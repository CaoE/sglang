/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// MHC (Multi-Head Channel) fused kernels for Xeon CPU.
//
// Three public entry points:
//   hc_pre_fused_cpu   – full hc_pre: RMSnorm + GEMM + sinkhorn + combine
//   hc_post_fused_cpu  – full hc_post: post*x + comb^T @ residual
//   hc_head_fused_cpu  – full hc_head: RMSnorm + GEMM + sigmoid gates + combine
//
// Data type contract:
//   * Data tensors (x, residual, y, out): bf16 only.
//   * Coefficient tensors (post, comb, hc_fn, hc_scale, hc_base): always float32.
//   * Internal reductions, dot products, and accumulators use float32.
//   * All arithmetic performed in float32; results converted back to bf16.
//   * hc_mult=4 required (DeepSeek V4 default).
//   * MHC projections are computed by fused tiny-GEMM loops; no scaled_x or
//     at::mm intermediate is materialized.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// Heuristics for the fused MHC loops.  Keep the active weight tile plus the
// converted-x cache near a private-cache working set, and keep the Phase 3
// accumulator tile around a few KiB per worker.  Values are derived from bytes
// and SIMD width instead of hard-coding per-kernel K/D block sizes.
constexpr int64_t kMhcTileCacheBudgetBytes = 36 * 1024;
constexpr int64_t kMhcScratchTargetBytes = 2 * 1024;
constexpr int64_t kMhcDefaultTokenBlock = 32;
constexpr int64_t kMhcMinDVecs = 8;
constexpr int64_t kMhcMaxDVecs = 32;

inline int64_t round_to_multiple(int64_t value, int64_t multiple) {
  return ((value + multiple / 2) / multiple) * multiple;
}

inline int64_t choose_t_block(int64_t /*T*/) {
  // Token blocking primarily controls weight-tile reuse.  32 tokens amortizes
  // weight loads well while keeping partials compact; Phase 1 dispatch below
  // decides whether to split K or use pure token-block parallelism.
  return kMhcDefaultTokenBlock;
}

inline int64_t choose_k_block(int64_t weight_rows, int64_t vec_size, int64_t max_block) {
  const int64_t bytes_per_k = (weight_rows + 1) * static_cast<int64_t>(sizeof(float));
  const int64_t raw = std::max(vec_size, kMhcTileCacheBudgetBytes / bytes_per_k);
  const int64_t grain = std::max<int64_t>(vec_size, 2 * vec_size);
  const int64_t rounded = round_to_multiple(raw, grain);
  return std::max(vec_size, std::min(max_block, rounded));
}

inline int64_t choose_d_block(int64_t vec_size) {
  const int64_t target_elems = kMhcScratchTargetBytes / static_cast<int64_t>(sizeof(float));
  const int64_t min_elems = kMhcMinDVecs * vec_size;
  const int64_t max_elems = kMhcMaxDVecs * vec_size;
  const int64_t rounded = round_to_multiple(target_elems, vec_size);
  return std::max(min_elems, std::min(max_elems, rounded));
}

template <typename func_t>
inline void parallel_mhc_phase1(int64_t n_t_blocks, int64_t n_k_blocks, const func_t& f) {
  const int64_t nth = static_cast<int64_t>(at::get_num_threads());
  if (n_t_blocks >= nth) {
    // Large prefill: enough token blocks exist to fill all workers.  Avoid
    // splitting K so Phase 2 does not pay unnecessary partial-reduce overhead.
    parallel_2d_tiled(static_cast<int>(n_t_blocks), static_cast<int>(n_k_blocks), static_cast<int>(nth), 1, f);
  } else {
    // Decode / small T: split both T and K to expose enough parallel tasks.
    parallel_2d(static_cast<int>(n_t_blocks), static_cast<int>(n_k_blocks), f);
  }
}

// In-place Sinkhorn normalization on a flat [HC*HC] float32 array (row-major).
//
// Matches the TileLang GPU kernel exactly:
//   First iter:      cm /= row_sum (then cm += eps);  cm /= (col_sum + eps)
//   Subsequent iters: cm /= (row_sum + eps);           cm /= (col_sum + eps)
// ---------------------------------------------------------------------------
template <int HC>
inline void sinkhorn_inplace(float* __restrict__ cm, int sinkhorn_iters, float eps) {
  float row_sum[HC];
  float col_sum[HC];

  // First iteration: asymmetric row step (mirrors TileLang kernel)
  for (int r = 0; r < HC; ++r) {
    row_sum[r] = 0.f;
    for (int c = 0; c < HC; ++c)
      row_sum[r] += cm[r * HC + c];
  }
  for (int r = 0; r < HC; ++r) {
    float inv = 1.0f / row_sum[r];
    for (int c = 0; c < HC; ++c)
      cm[r * HC + c] = cm[r * HC + c] * inv + eps;
  }
  for (int c = 0; c < HC; ++c) {
    col_sum[c] = 0.f;
    for (int r = 0; r < HC; ++r)
      col_sum[c] += cm[r * HC + c];
  }
  for (int r = 0; r < HC; ++r)
    for (int c = 0; c < HC; ++c)
      cm[r * HC + c] /= (col_sum[c] + eps);

  // Remaining (sinkhorn_iters-1) iterations: symmetric
  for (int iter = 1; iter < sinkhorn_iters; ++iter) {
    for (int r = 0; r < HC; ++r) {
      row_sum[r] = 0.f;
      for (int c = 0; c < HC; ++c)
        row_sum[r] += cm[r * HC + c];
    }
    for (int r = 0; r < HC; ++r) {
      float inv = 1.0f / (row_sum[r] + eps);
      for (int c = 0; c < HC; ++c)
        cm[r * HC + c] *= inv;
    }
    for (int c = 0; c < HC; ++c) {
      col_sum[c] = 0.f;
      for (int r = 0; r < HC; ++r)
        col_sum[c] += cm[r * HC + c];
    }
    for (int r = 0; r < HC; ++r)
      for (int c = 0; c < HC; ++c)
        cm[r * HC + c] /= (col_sum[c] + eps);
  }
}

// ---------------------------------------------------------------------------
// parse_mixes_and_sinkhorn<HC>
// From mixes[mix_hc] compute pre[HC], post[HC] and comb[HC*HC] in-place.
// ---------------------------------------------------------------------------
template <int HC>
inline void parse_mixes_and_sinkhorn(
    float* __restrict__ pre,      // [HC]   output
    float* __restrict__ post,     // [HC]   output
    float* __restrict__ cm,       // [HC*HC] output (row-major)
    const float* __restrict__ m,  // [mix_hc] input mixes
    float s0,
    float s1,
    float s2,
    const float* __restrict__ hc_base,
    int sinkhorn_iters,
    float eps) {
  // pre = sigmoid(m[:HC]*s0 + base[:HC]) + eps
  for (int h = 0; h < HC; ++h) {
    float v = m[h] * s0 + hc_base[h];
    pre[h] = 1.f / (1.f + std::exp(-v)) + eps;
  }
  // post = 2 * sigmoid(m[HC:2HC]*s1 + base[HC:2HC])
  for (int h = 0; h < HC; ++h) {
    float v = m[HC + h] * s1 + hc_base[HC + h];
    post[h] = 2.f / (1.f + std::exp(-v));
  }
  // comb logits: row-stable exp (subtract per-row max)
  float row_max[HC];
  for (int r = 0; r < HC; ++r)
    row_max[r] = -std::numeric_limits<float>::infinity();
  for (int r = 0; r < HC; ++r)
    for (int c = 0; c < HC; ++c) {
      float v = m[2 * HC + r * HC + c] * s2 + hc_base[2 * HC + r * HC + c];
      cm[r * HC + c] = v;
      if (v > row_max[r]) row_max[r] = v;
    }
  for (int r = 0; r < HC; ++r)
    for (int c = 0; c < HC; ++c)
      cm[r * HC + c] = std::exp(cm[r * HC + c] - row_max[r]);

  sinkhorn_inplace<HC>(cm, sinkhorn_iters, eps);
}

template <typename scalar_t, int HC>
static void hc_post_fuse_impl(
    scalar_t* __restrict__ out,             // [T, HC, d]  output
    const scalar_t* __restrict__ x,         // [T, d]      input
    const scalar_t* __restrict__ residual,  // [T, HC, d]  input
    const float* __restrict__ post,         // [T, HC]     float32
    const float* __restrict__ comb,         // [T, HC, HC] float32 row-major
    int64_t T,
    int64_t d) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();

  const int64_t nthreads = at::get_num_threads();
  const int64_t max_splits = std::max(int64_t(1), d / kVecSize);
  const int64_t token_head_tasks = T * HC;
  const int64_t K_SPLITS = (token_head_tasks >= nthreads) ? int64_t(1) : std::min(nthreads, max_splits);
  const int64_t k_chunk = (d + K_SPLITS - 1) / K_SPLITS;

  at::parallel_for(0, T * HC * K_SPLITS, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t tmp = idx / K_SPLITS;
      const int64_t ks = idx % K_SPLITS;
      const int64_t t = tmp / HC;
      const int h = static_cast<int>(tmp % HC);
      const int64_t k0 = ks * k_chunk;
      const int64_t k1 = std::min(k0 + k_chunk, d);

      const float post_val = post[t * HC + h];
      const float* comb_t = comb + t * HC * HC;
      const scalar_t* x_t = x + t * d;
      const scalar_t* res_t = residual + t * HC * d;
      scalar_t* out_th = out + (t * HC + h) * d;

      const fVec post_fvec(post_val);
      fVec comb_fvec[HC];
      for (int i = 0; i < HC; ++i)
        comb_fvec[i] = fVec(comb_t[i * HC + h]);

      int64_t k;
      // bf16: convert and accumulate
      for (k = k0; k <= k1 - kVecSize; k += kVecSize) {
        fVec acc0, acc1;
        std::tie(acc0, acc1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
        acc0 = post_fvec * acc0;
        acc1 = post_fvec * acc1;
        for (int i = 0; i < HC; ++i) {
          fVec r0, r1;
          std::tie(r0, r1) = at::vec::convert_to_float(bVec::loadu(res_t + i * d + k));
          acc0 += comb_fvec[i] * r0;
          acc1 += comb_fvec[i] * r1;
        }
        at::vec::convert_from_float<scalar_t>(acc0, acc1).store(out_th + k);
      }
      if (k < k1) {
        const int64_t rem = k1 - k;
        fVec acc0, acc1;
        std::tie(acc0, acc1) = at::vec::convert_to_float(bVec::loadu(x_t + k, rem));
        acc0 = post_fvec * acc0;
        acc1 = post_fvec * acc1;
        for (int i = 0; i < HC; ++i) {
          fVec r0, r1;
          std::tie(r0, r1) = at::vec::convert_to_float(bVec::loadu(res_t + i * d + k, rem));
          acc0 += comb_fvec[i] * r0;
          acc1 += comb_fvec[i] * r1;
        }
        at::vec::convert_from_float<scalar_t>(acc0, acc1).store(out_th + k, rem);
      }
    }
  });
}

// ---------------------------------------------------------------------------
// hc_pre_fuse_impl<scalar_t, HC>
// 2D-parallel fused path for hc_pre_fused_cpu (T > kSmallTokenThreshold).
//
// Fuses RMSnorm + GEMM (mix_hc=24 dot products) + sinkhorn + weighted combine
//
// Phase 1: parallel_mhc_phase1(n_t_blocks, n_k_blocks)
//   - upcast bf16→f32, fused sq_sum + mix_hc dot products → partials[]
//   - weight tiles reused across all tokens in the same t-block range
// Phase 2+3: parallel_for(T)
//   - reduce partials → inv_rms → mixes → sinkhorn → pre/post/comb
//   - d-tiled weighted combine → y[] with pre[] kept in registers
//
// T/K/D block sizes are selected from cache/SIMD heuristics at runtime.
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_pre_fuse_impl(
    scalar_t* __restrict__ y,            // [T, d]      bf16 output
    float* __restrict__ post_out,        // [T, HC]     float32 output
    float* __restrict__ comb_out,        // [T, HC*HC]  float32 output
    const scalar_t* __restrict__ x,      // [T, HC, d]  bf16 input
    const float* __restrict__ hc_fn,     // [mix_hc, hc_d] float32 row-major
    const float* __restrict__ hc_scale,  // [3]
    const float* __restrict__ hc_base,   // [mix_hc]
    int64_t T,
    int64_t d,
    int sinkhorn_iters,
    float hc_eps,
    float norm_eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();   // 32 bf16
  constexpr int64_t kFVecSize = fVec::size();  // 16 fp32
  constexpr int mix_hc = (2 + HC) * HC;
  constexpr int64_t K_BLOCK_CAP = 512;
  constexpr int64_t MAX_K_VECS = K_BLOCK_CAP / kVecSize;
  const int64_t hc_d = HC * d;
  const float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

  const int64_t T_BLOCK = choose_t_block(T);
  const int64_t K_BLOCK = choose_k_block(mix_hc, kVecSize, K_BLOCK_CAP);
  const int64_t D_BLOCK = choose_d_block(kVecSize);

  const int64_t n_t_blocks = div_up(T, T_BLOCK);
  const int64_t n_k_blocks = div_up(hc_d, K_BLOCK);

  // Partial buffer: [n_t_blocks, n_k_blocks, T_BLOCK, 1+mix_hc]
  constexpr int64_t P_STRIDE = 1 + mix_hc;  // 25
  auto partials_tensor = at::empty({n_t_blocks * n_k_blocks * T_BLOCK * P_STRIDE}, at::kFloat);
  float* const partials = partials_tensor.data_ptr<float>();

  // Per-thread scratch for Phase 3 d-tile accumulator
  const int64_t nth_sc = static_cast<int64_t>(at::get_num_threads());
  auto sc_tensor = at::empty({nth_sc * D_BLOCK}, at::kFloat);
  float* const sc_base = sc_tensor.data_ptr<float>();

  // ── Phase 1: fused sq + mix_hc dots ──────────────────────────────────────
  parallel_mhc_phase1(n_t_blocks, n_k_blocks, [&](int64_t tb0, int64_t tb1, int64_t kb0, int64_t kb1) {
    for (int64_t kb = kb0; kb < kb1; ++kb) {
      const int64_t k0 = kb * K_BLOCK;
      const int64_t klen = std::min(K_BLOCK, hc_d - k0);

      for (int64_t tb = tb0; tb < tb1; ++tb) {
        const int64_t t0 = tb * T_BLOCK;
        const int64_t tlen = std::min(T_BLOCK, T - t0);
        float* p_base = partials + (tb * n_k_blocks + kb) * T_BLOCK * P_STRIDE;

        const float* w_ptr[mix_hc];
        for (int h = 0; h < mix_hc; ++h)
          w_ptr[h] = hc_fn + h * hc_d + k0;

        fVec x_cache_lo[MAX_K_VECS], x_cache_hi[MAX_K_VECS];

        for (int64_t tl = 0; tl < tlen; ++tl) {
          const scalar_t* x_t = x + (t0 + tl) * hc_d + k0;
          float* p = p_base + tl * P_STRIDE;

          const int64_t n_full = klen / kVecSize;
          const int64_t k_tail = n_full * kVecSize;
          const int64_t rem = klen - k_tail;
          const int64_t rem0 = std::min(rem, kFVecSize);
          const int64_t rem1 = rem - rem0;

          // Batch-prime next token's x (NTA: stream-once, avoids L1 pollution).
          // Two evenly-spaced lines engage the L2 HW streamer without excess.
          if (tl + 1 < tlen) {
            const scalar_t* xn = x + (t0 + tl + 1) * hc_d + k0;
            __builtin_prefetch(xn, 0, 0);
            if (klen > kVecSize * 4) __builtin_prefetch(xn + kVecSize * 4, 0, 0);
          }

          // h=0: convert x once, cache, fuse sq + dot[0]
          // Bulk-prime all cold weight rows before the ki loop on tl==0 so the
          // prefetch distance is large enough to cover the entire weight tile;
          // the HW streamer handles subsequent sequential weight access.
          if (tl == 0) {
            for (int pw = 1; pw < mix_hc; ++pw)
              __builtin_prefetch(w_ptr[pw], 0, 3);
          }
          {
            const float* w0 = w_ptr[0];
            fVec sq_lo(0.f), sq_hi(0.f);
            fVec d0(0.f), d1(0.f);
            for (int64_t ki = 0; ki < n_full; ++ki) {
              const int64_t k = ki * kVecSize;
              std::tie(x_cache_lo[ki], x_cache_hi[ki]) = at::vec::convert_to_float(bVec::loadu(x_t + k));
              sq_lo += x_cache_lo[ki] * x_cache_lo[ki];
              sq_hi += x_cache_hi[ki] * x_cache_hi[ki];
              d0 += x_cache_lo[ki] * fVec::loadu(w0 + k);
              d1 += x_cache_hi[ki] * fVec::loadu(w0 + k + kFVecSize);
            }

            fVec x_tail_lo(0.f), x_tail_hi(0.f);
            if (rem > 0) {
              std::tie(x_tail_lo, x_tail_hi) = at::vec::convert_to_float(bVec::loadu(x_t + k_tail, rem));
              sq_lo += x_tail_lo * x_tail_lo;
              if (rem1 > 0) sq_hi += x_tail_hi * x_tail_hi;
              d0 += x_tail_lo * fVec::loadu(w0 + k_tail, rem0);
              if (rem1 > 0) d1 += x_tail_hi * fVec::loadu(w0 + k_tail + kFVecSize, rem1);
            }

            p[0] = vec_reduce_sum(sq_lo + sq_hi);
            p[1 + 0] = vec_reduce_sum(d0 + d1);

            // h=1..mix_hc-1: reuse cached x, 2-head interleaving for 4 FMA chains
            int h = 1;
            for (; h + 1 < mix_hc; h += 2) {
              const float* wa = w_ptr[h];
              const float* wb = w_ptr[h + 1];
              fVec a0(0.f), a1(0.f), b0(0.f), b1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                a0 += x_cache_lo[ki] * fVec::loadu(wa + k);
                b0 += x_cache_lo[ki] * fVec::loadu(wb + k);
                a1 += x_cache_hi[ki] * fVec::loadu(wa + k + kFVecSize);
                b1 += x_cache_hi[ki] * fVec::loadu(wb + k + kFVecSize);
              }
              if (rem > 0) {
                a0 += x_tail_lo * fVec::loadu(wa + k_tail, rem0);
                b0 += x_tail_lo * fVec::loadu(wb + k_tail, rem0);
                if (rem1 > 0) {
                  a1 += x_tail_hi * fVec::loadu(wa + k_tail + kFVecSize, rem1);
                  b1 += x_tail_hi * fVec::loadu(wb + k_tail + kFVecSize, rem1);
                }
              }
              p[1 + h] = vec_reduce_sum(a0 + a1);
              p[1 + h + 1] = vec_reduce_sum(b0 + b1);
            }
            // odd leftover (mix_hc=24 is even, so this is just for safety)
            if (h < mix_hc) {
              const float* w = w_ptr[h];
              fVec hd0(0.f), hd1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                hd0 += x_cache_lo[ki] * fVec::loadu(w + k);
                hd1 += x_cache_hi[ki] * fVec::loadu(w + k + kFVecSize);
              }
              if (rem > 0) {
                hd0 += x_tail_lo * fVec::loadu(w + k_tail, rem0);
                if (rem1 > 0) hd1 += x_tail_hi * fVec::loadu(w + k_tail + kFVecSize, rem1);
              }
              p[1 + h] = vec_reduce_sum(hd0 + hd1);
            }
          }
        }  // tl

        // Cold-start: prime next tb's first token's x for L2 streamer
        if (tb + 1 < tb1) {
          const scalar_t* x_next_tb = x + ((tb + 1) * T_BLOCK) * hc_d + k0;
          __builtin_prefetch(x_next_tb, 0, 0);
          if (klen > kVecSize * 4) __builtin_prefetch(x_next_tb + kVecSize * 4, 0, 0);
        }
      }  // tb
    }  // kb
  });  // Phase 1

  // ── Phase 2+3: reduce → sinkhorn → combine → y[] ─────────────────────────
  // Per-token independent: reduce partials, compute mixes/sinkhorn/pre/post/comb,
  // then d-tiled weighted combine + fused bf16 convert. pre_t[] stays in registers.
  at::parallel_for(0, T, 1, [&](int64_t begin, int64_t end) {
    const int64_t tid = at::get_thread_num();
    float* const sc = sc_base + tid * D_BLOCK;
    fVec v0, v1, pre_fvec;

    for (int64_t t = begin; t < end; ++t) {
      const int64_t tb = t / T_BLOCK;
      const int64_t tl = t % T_BLOCK;

      // Phase 2: reduce partials → mixes → sinkhorn (vectorized over mix_hc).
      // Requires kFVecSize < mix_hc <= 2*kFVecSize (satisfied for HC=4: mix_hc=24, kFVecSize=16).
      static_assert(
          mix_hc > kFVecSize && mix_hc <= 2 * kFVecSize,
          "hc_pre_fuse_impl: vectorized reduce requires kFVecSize < mix_hc <= 2*kFVecSize");
      constexpr int64_t dots_tail = mix_hc - kFVecSize;
      fVec acc_lo(0.f), acc_hi(0.f);
      double sq_total = 0.0;
      for (int64_t kb = 0; kb < n_k_blocks; ++kb) {
        const float* p = partials + (tb * n_k_blocks + kb) * T_BLOCK * P_STRIDE + tl * P_STRIDE;
        sq_total += static_cast<double>(p[0]);
        acc_lo += fVec::loadu(p + 1);
        acc_hi += fVec::loadu(p + 1 + kFVecSize, dots_tail);
      }
      const float inv_rms =
          static_cast<float>(1.0 / std::sqrt(sq_total / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));
      const fVec inv_rms_vec(inv_rms);
      float mixes[mix_hc];
      (acc_lo * inv_rms_vec).store(mixes);
      (acc_hi * inv_rms_vec).store(mixes + kFVecSize, dots_tail);

      float pre_t[HC], po[HC], cm[HC * HC];
      parse_mixes_and_sinkhorn<HC>(pre_t, po, cm, mixes, s0, s1, s2, hc_base, sinkhorn_iters, hc_eps);

      for (int h = 0; h < HC; ++h)
        post_out[t * HC + h] = po[h];
      for (int i = 0; i < HC * HC; ++i)
        comb_out[t * HC * HC + i] = cm[i];

      // Phase 3: d-tiled weighted combine + fused bf16 convert
      const scalar_t* x_t = x + t * hc_d;
      scalar_t* y_t = y + t * d;
      for (int64_t j0 = 0; j0 < d; j0 += D_BLOCK) {
        const int64_t jlen = std::min(D_BLOCK, d - j0);

        // h=0: direct assign
        // HC=4 heads × D_BLOCK elems fit comfortably in L1; the HW streamer
        // handles the sequential next-head read pattern automatically.
        {
          pre_fvec = fVec(pre_t[0]);
          const scalar_t* x_th = x_t + j0;
          int64_t kk = 0;
          for (; kk <= jlen - kVecSize; kk += kVecSize) {
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk));
            (pre_fvec * v0).store(sc + kk);
            (pre_fvec * v1).store(sc + kk + kFVecSize);
          }
          if (kk < jlen) {
            const int64_t rem = jlen - kk, rem0 = std::min(rem, kFVecSize), rem1 = rem - rem0;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk, rem));
            (pre_fvec * v0).store(sc + kk, rem0);
            if (rem1 > 0) (pre_fvec * v1).store(sc + kk + kFVecSize, rem1);
          }
        }

        // h=1..HC-1: accumulate
        for (int h = 1; h < HC; ++h) {
          pre_fvec = fVec(pre_t[h]);
          const scalar_t* x_th = x_t + (int64_t)h * d + j0;
          int64_t kk = 0;
          for (; kk <= jlen - kVecSize; kk += kVecSize) {
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk));
            (fVec::loadu(sc + kk) + pre_fvec * v0).store(sc + kk);
            (fVec::loadu(sc + kk + kFVecSize) + pre_fvec * v1).store(sc + kk + kFVecSize);
          }
          if (kk < jlen) {
            const int64_t rem = jlen - kk, rem0 = std::min(rem, kFVecSize), rem1 = rem - rem0;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk, rem));
            (fVec::loadu(sc + kk, rem0) + pre_fvec * v0).store(sc + kk, rem0);
            if (rem1 > 0) (fVec::loadu(sc + kk + kFVecSize, rem1) + pre_fvec * v1).store(sc + kk + kFVecSize, rem1);
          }
        }

        // fused convert: sc (hot in L1) → bf16
        {
          int64_t kk = 0;
          for (; kk <= jlen - kVecSize; kk += kVecSize) {
            at::vec::convert_from_float<scalar_t>(fVec::loadu(sc + kk), fVec::loadu(sc + kk + kFVecSize))
                .store(y_t + j0 + kk);
          }
          if (kk < jlen) {
            const int64_t rem = jlen - kk, rem0 = std::min(rem, kFVecSize), rem1 = rem - rem0;
            at::vec::convert_from_float<scalar_t>(
                fVec::loadu(sc + kk, rem0), rem1 > 0 ? fVec::loadu(sc + kk + kFVecSize, rem1) : fVec(0.f))
                .store(y_t + j0 + kk, rem);
          }
        }
      }  // d-tile
    }  // t
  });  // Phase 2+3
}

// ---------------------------------------------------------------------------
// hc_head_fuse_impl<scalar_t, HC>
// 2D-parallel fused path for T ≥ kSmallTokenThreshold.
//
// Borrows the CUDA TileLang strategy (mhc_pre_gemm_sqrsum_splitk):
//   adaptive Phase 1 chooses pure token-block parallelism for large prefill,
//   or 2D T×K splitting for decode/small T to expose enough work.
//
// Two parallel regions:
//   Phase 1 – parallel_mhc_phase1: upcast bf16→f32 + fused sq + HC dot-products
//             → partial_sq[tb,kb,tl] and partial_dot[tb,kb,tl,h]
//   Phase 2+3 – parallel_for(T): reduce partials, compute sigmoid gates, then
//             d-tiled weighted combine + fused bf16 convert → y[T, d]
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_head_fuse_impl(
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ x,
    const float* __restrict__ hc_fn,  // [HC, HC*d] row-major
    float hc_scale_val,
    const float* __restrict__ hc_base,
    int64_t T,
    int64_t d,
    float hc_eps,
    float norm_eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();   // 32 bf16
  constexpr int64_t kFVecSize = fVec::size();  // 16 fp32
  const int64_t hc_d = HC * d;

  constexpr int64_t K_BLOCK_CAP = 2048;
  constexpr int64_t MAX_K_VECS = K_BLOCK_CAP / kVecSize;

  const int64_t T_BLOCK = choose_t_block(T);
  const int64_t K_BLOCK = choose_k_block(HC, kVecSize, K_BLOCK_CAP);
  const int64_t D_BLOCK = choose_d_block(kVecSize);

  const int64_t n_t_blocks = div_up(T, T_BLOCK);
  const int64_t n_k_blocks = div_up(hc_d, K_BLOCK);

  // Partial buffer: [n_t_blocks, n_k_blocks, T_BLOCK, 1+HC]
  //   p[0] = partial sq,  p[1..HC] = partial dots
  constexpr int64_t P_STRIDE = 1 + HC;
  auto partials_tensor = at::empty({n_t_blocks * n_k_blocks * T_BLOCK * P_STRIDE}, at::kFloat);
  float* const partials = partials_tensor.data_ptr<float>();

  // Per-thread scratch for Phase 3 d-tile accumulator.
  const int64_t nth_sc = static_cast<int64_t>(at::get_num_threads());
  auto sc_tensor = at::empty({nth_sc * D_BLOCK}, at::kFloat);
  float* const sc_base = sc_tensor.data_ptr<float>();

  // ── Phase 1: adaptive token/K parallelism ────────────────────────────────
  // kb OUTER, tb INNER inside each task: weight tile [HC, K_BLOCK] is reused
  // across all token blocks assigned to that worker.
  parallel_mhc_phase1(n_t_blocks, n_k_blocks, [&](int64_t tb0, int64_t tb1, int64_t kb0, int64_t kb1) {
    // kb OUTER: keep the heuristic-sized weight tile hot across t_blocks.
    for (int64_t kb = kb0; kb < kb1; ++kb) {
      const int64_t k0 = kb * K_BLOCK;
      const int64_t klen = std::min(K_BLOCK, hc_d - k0);

      // tb INNER: all tokens assigned to this task share the weight tile.
      for (int64_t tb = tb0; tb < tb1; ++tb) {
        const int64_t t0 = tb * T_BLOCK;
        const int64_t tlen = std::min(T_BLOCK, T - t0);
        float* p_base = partials + (tb * n_k_blocks + kb) * T_BLOCK * P_STRIDE;

        // ── Per-token fused loops: h outer, K inner ────────────────────────
        // Weight tile [HC,K_BLOCK] stays in L1 across all tokens (kb-outer).
        // x-vector cache eliminates redundant bf16->fp32 conversions.
        // x_next batch-primed to L2 at tl start; w prefetch only on tl==0.
        const float* w_ptr[HC];
        for (int h = 0; h < HC; ++h)
          w_ptr[h] = hc_fn + h * hc_d + k0;

        fVec x_cache_lo[MAX_K_VECS], x_cache_hi[MAX_K_VECS];

        for (int64_t tl = 0; tl < tlen; ++tl) {
          const scalar_t* x_t = x + (t0 + tl) * hc_d + k0;
          float* p = p_base + tl * P_STRIDE;

          const int64_t n_full = klen / kVecSize;
          const int64_t k_tail = n_full * kVecSize;
          const int64_t rem = klen - k_tail;
          const int64_t rem0 = std::min(rem, kFVecSize);
          const int64_t rem1 = rem - rem0;

          // Batch-prime next token's x (NTA: stream-once).
          // Two evenly-spaced lines are sufficient to engage the HW streamer.
          if (tl + 1 < tlen) {
            const scalar_t* xn = x + (t0 + tl + 1) * hc_d + k0;
            __builtin_prefetch(xn, 0, 0);
            if (klen > kVecSize * 4) __builtin_prefetch(xn + kVecSize * 4, 0, 0);
          }

          // h=0: convert x once, fuse sq + dot0, and cache converted x.
          // Bulk-prime all cold weight rows before the ki loop so the prefetch
          // distance covers the entire weight tile (HW streamer takes over after).
          if (tl == 0) {
            for (int pw = 1; pw < HC; ++pw)
              __builtin_prefetch(w_ptr[pw], 0, 3);
          }
          {
            const float* w0 = w_ptr[0];
            fVec sq_lo(0.f), sq_hi(0.f);
            fVec d0(0.f), d1(0.f);
            for (int64_t ki = 0; ki < n_full; ++ki) {
              const int64_t k = ki * kVecSize;
              std::tie(x_cache_lo[ki], x_cache_hi[ki]) = at::vec::convert_to_float(bVec::loadu(x_t + k));
              sq_lo += x_cache_lo[ki] * x_cache_lo[ki];
              sq_hi += x_cache_hi[ki] * x_cache_hi[ki];
              d0 += x_cache_lo[ki] * fVec::loadu(w0 + k);
              d1 += x_cache_hi[ki] * fVec::loadu(w0 + k + kFVecSize);
            }

            fVec x_tail_lo(0.f), x_tail_hi(0.f);
            if (rem > 0) {
              std::tie(x_tail_lo, x_tail_hi) = at::vec::convert_to_float(bVec::loadu(x_t + k_tail, rem));
              sq_lo += x_tail_lo * x_tail_lo;
              if (rem1 > 0) sq_hi += x_tail_hi * x_tail_hi;
              d0 += x_tail_lo * fVec::loadu(w0 + k_tail, rem0);
              if (rem1 > 0) d1 += x_tail_hi * fVec::loadu(w0 + k_tail + kFVecSize, rem1);
            }

            p[0] = vec_reduce_sum(sq_lo + sq_hi);
            p[1 + 0] = vec_reduce_sum(d0 + d1);

            // h=1..HC-1: reuse cached x, 2-head interleaving for 4 FMA chains
            int h = 1;
            for (; h + 1 < HC; h += 2) {
              const float* wa = w_ptr[h];
              const float* wb = w_ptr[h + 1];
              fVec a0(0.f), a1(0.f), b0(0.f), b1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                a0 += x_cache_lo[ki] * fVec::loadu(wa + k);
                b0 += x_cache_lo[ki] * fVec::loadu(wb + k);
                a1 += x_cache_hi[ki] * fVec::loadu(wa + k + kFVecSize);
                b1 += x_cache_hi[ki] * fVec::loadu(wb + k + kFVecSize);
              }
              if (rem > 0) {
                a0 += x_tail_lo * fVec::loadu(wa + k_tail, rem0);
                b0 += x_tail_lo * fVec::loadu(wb + k_tail, rem0);
                if (rem1 > 0) {
                  a1 += x_tail_hi * fVec::loadu(wa + k_tail + kFVecSize, rem1);
                  b1 += x_tail_hi * fVec::loadu(wb + k_tail + kFVecSize, rem1);
                }
              }
              p[1 + h] = vec_reduce_sum(a0 + a1);
              p[1 + h + 1] = vec_reduce_sum(b0 + b1);
            }
            // odd leftover (HC=4: h=3 is the leftover)
            if (h < HC) {
              const float* w = w_ptr[h];
              fVec hd0(0.f), hd1(0.f);
              for (int64_t ki = 0; ki < n_full; ++ki) {
                const int64_t k = ki * kVecSize;
                hd0 += x_cache_lo[ki] * fVec::loadu(w + k);
                hd1 += x_cache_hi[ki] * fVec::loadu(w + k + kFVecSize);
              }
              if (rem > 0) {
                hd0 += x_tail_lo * fVec::loadu(w + k_tail, rem0);
                if (rem1 > 0) hd1 += x_tail_hi * fVec::loadu(w + k_tail + kFVecSize, rem1);
              }
              p[1 + h] = vec_reduce_sum(hd0 + hd1);
            }
          }
        }  // tl

        // Cold-start: prime next tb's first token's x for L2 streamer
        if (tb + 1 < tb1) {
          const scalar_t* x_next_tb = x + ((tb + 1) * T_BLOCK) * hc_d + k0;
          __builtin_prefetch(x_next_tb, 0, 0);
          if (klen > kVecSize * 2) __builtin_prefetch(x_next_tb + kVecSize * 2, 0, 0);
        }
      }  // tb
    }  // kb
  });  // Phase 1

  // ── Phase 2+3: reduce partials + sigmoid gate + combine → y[] ─────────────
  at::parallel_for(0, T, 1, [&](int64_t begin, int64_t end) {
    const int64_t tid = at::get_thread_num();
    float* const sc = sc_base + tid * D_BLOCK;
    fVec v0, v1, pre_fvec;

    for (int64_t t = begin; t < end; ++t) {
      const int64_t tb = t / T_BLOCK;
      const int64_t tl = t % T_BLOCK;

      // Phase 2: reduce partials + sigmoid gate → pre_t[] in registers
      double sq_total = 0.0;
      float pre_t[HC] = {};
      for (int64_t kb = 0; kb < n_k_blocks; ++kb) {
        const float* p = partials + (tb * n_k_blocks + kb) * T_BLOCK * P_STRIDE + tl * P_STRIDE;
        sq_total += static_cast<double>(p[0]);
        for (int h = 0; h < HC; ++h)
          pre_t[h] += p[1 + h];
      }
      const float inv_rms =
          static_cast<float>(1.0 / std::sqrt(sq_total / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));
      for (int h = 0; h < HC; ++h) {
        float gate = pre_t[h] * inv_rms * hc_scale_val + hc_base[h];
        pre_t[h] = 1.f / (1.f + std::exp(-gate)) + hc_eps;
      }

      // Phase 3: d-tiled weighted combine + fused bf16 convert
      const scalar_t* x_t = x + t * hc_d;
      scalar_t* y_t = y + t * d;
      for (int64_t j0 = 0; j0 < d; j0 += D_BLOCK) {
        const int64_t jlen = std::min(D_BLOCK, d - j0);

        // h=0: direct assign
        // HC=4 heads × D_BLOCK elems fit in L1; HW streamer handles sequential
        // next-head accesses automatically without per-vector manual prefetch.
        {
          pre_fvec = fVec(pre_t[0]);
          const scalar_t* x_th = x_t + j0;
          int64_t kk = 0;
          for (; kk <= jlen - kVecSize; kk += kVecSize) {
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk));
            (pre_fvec * v0).store(sc + kk);
            (pre_fvec * v1).store(sc + kk + kFVecSize);
          }
          if (kk < jlen) {
            const int64_t rem = jlen - kk, rem0 = std::min(rem, kFVecSize), rem1 = rem - rem0;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk, rem));
            (pre_fvec * v0).store(sc + kk, rem0);
            if (rem1 > 0) (pre_fvec * v1).store(sc + kk + kFVecSize, rem1);
          }
        }

        // h=1..HC-1: accumulate
        for (int h = 1; h < HC; ++h) {
          pre_fvec = fVec(pre_t[h]);
          const scalar_t* x_th = x_t + (int64_t)h * d + j0;
          int64_t kk = 0;
          for (; kk <= jlen - kVecSize; kk += kVecSize) {
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk));
            (fVec::loadu(sc + kk) + pre_fvec * v0).store(sc + kk);
            (fVec::loadu(sc + kk + kFVecSize) + pre_fvec * v1).store(sc + kk + kFVecSize);
          }
          if (kk < jlen) {
            const int64_t rem = jlen - kk, rem0 = std::min(rem, kFVecSize), rem1 = rem - rem0;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk, rem));
            (fVec::loadu(sc + kk, rem0) + pre_fvec * v0).store(sc + kk, rem0);
            if (rem1 > 0) (fVec::loadu(sc + kk + kFVecSize, rem1) + pre_fvec * v1).store(sc + kk + kFVecSize, rem1);
          }
        }

        // fused convert: sc (hot in L1) → bf16
        {
          int64_t kk = 0;
          for (; kk <= jlen - kVecSize; kk += kVecSize) {
            at::vec::convert_from_float<scalar_t>(fVec::loadu(sc + kk), fVec::loadu(sc + kk + kFVecSize))
                .store(y_t + j0 + kk);
          }
          if (kk < jlen) {
            const int64_t rem = jlen - kk, rem0 = std::min(rem, kFVecSize), rem1 = rem - rem0;
            at::vec::convert_from_float<scalar_t>(
                fVec::loadu(sc + kk, rem0), rem1 > 0 ? fVec::loadu(sc + kk + kFVecSize, rem1) : fVec(0.f))
                .store(y_t + j0 + kk, rem);
          }
        }
      }  // d-tile
    }  // t
  });  // Phase 2+3
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// hc_pre_fused_cpu
// Inputs:
//   x         [T, hc_mult, d]          bf16
//   hc_fn     [mix_hc, hc_mult*d]      float32   (projection weight)
//   hc_scale  [3]                      float32
//   hc_base   [mix_hc]                 float32
// Outputs: (y [T,d] bf16,  post [T,hc_mult] float32,
//           comb [T,hc_mult,hc_mult]  float32)
//
// Algorithm (fully fused, no intermediate scaled_x):
//   Per token: inv_rms = rsqrt(mean(x_flat^2) + rms_eps)
//   mixes     = (x_flat @ hc_fn.T) * inv_rms   [T, mix_hc]
//   sinkhorn + weighted combine → y, post, comb
// ---------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_fused_cpu(
    at::Tensor& x,
    at::Tensor& hc_fn,
    at::Tensor& hc_scale,
    at::Tensor& hc_base,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double rms_eps,
    double hc_eps) {
  RECORD_FUNCTION("sgl-kernel::hc_pre_fused_cpu", {});
  TORCH_CHECK(hc_mult == 4, "hc_pre_fused_cpu: only hc_mult=4 is supported");
  TORCH_CHECK(x.dim() == 3 && x.scalar_type() == at::kBFloat16, "hc_pre_fused_cpu: x must be bf16 [T, hc, d]");
  TORCH_CHECK(hc_fn.scalar_type() == at::kFloat, "hc_pre_fused_cpu: hc_fn must be float32");
  TORCH_CHECK(
      hc_scale.scalar_type() == at::kFloat && hc_scale.numel() == 3, "hc_pre_fused_cpu: hc_scale must be float32 [3]");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_pre_fused_cpu: hc_base must be float32");
  TORCH_CHECK(
      x.device().is_cpu() && hc_fn.device().is_cpu() && hc_scale.device().is_cpu() && hc_base.device().is_cpu(),
      "hc_pre_fused_cpu: all inputs must be CPU tensors");

  const int64_t T = x.size(0);
  const int64_t d = x.size(2);
  const int64_t hc_d = hc_mult * d;
  const int64_t mix_hc = (2 + hc_mult) * hc_mult;

  TORCH_CHECK(T > 0 && d > 0, "hc_pre_fused_cpu: T and d must be positive");
  TORCH_CHECK(x.size(1) == hc_mult, "hc_pre_fused_cpu: x shape must be [T, hc_mult, d]");
  TORCH_CHECK(
      hc_fn.dim() == 2 && hc_fn.size(0) == mix_hc && hc_fn.size(1) == hc_d,
      "hc_pre_fused_cpu: hc_fn must be [mix_hc, hc_mult*d]");
  TORCH_CHECK(hc_base.numel() == mix_hc, "hc_pre_fused_cpu: hc_base.numel() must equal mix_hc");

  // Accept non-contiguous inputs and use contiguous copies/views internally.
  const at::Tensor x_c = x.is_contiguous() ? x : x.contiguous();
  const at::Tensor hc_fn_c = hc_fn.is_contiguous() ? hc_fn : hc_fn.contiguous();
  const at::Tensor hc_scale_c = hc_scale.is_contiguous() ? hc_scale : hc_scale.contiguous();
  const at::Tensor hc_base_c = hc_base.is_contiguous() ? hc_base : hc_base.contiguous();

  auto f32_opts = at::TensorOptions().dtype(at::kFloat).device(x.device());
  auto post = at::empty({T, hc_mult}, f32_opts);
  auto comb = at::empty({T, hc_mult, hc_mult}, f32_opts);
  auto y = at::empty({T, d}, x.options());

  hc_pre_fuse_impl<c10::BFloat16, 4>(
      y.data_ptr<c10::BFloat16>(),
      post.data_ptr<float>(),
      comb.data_ptr<float>(),
      x_c.data_ptr<c10::BFloat16>(),
      hc_fn_c.data_ptr<float>(),
      hc_scale_c.data_ptr<float>(),
      hc_base_c.data_ptr<float>(),
      T,
      d,
      static_cast<int>(sinkhorn_iters),
      static_cast<float>(hc_eps),
      static_cast<float>(rms_eps));
  return {y, post, comb};
}

// ---------------------------------------------------------------------------
// hc_post_fused_cpu
// Inputs:
//   x         [T, d]         bf16  (sublayer output)
//   residual  [T, hc_mult, d] bf16 (pre-sublayer residual)
//   post      [T, hc_mult]   float32
//   comb      [T, hc_mult, hc_mult]  float32
// Output: out [T, hc_mult, d]  same dtype as x
//
// Formula: out[t,h,k] = post[t,h]*x[t,k] + sum_i comb[t,i,h]*residual[t,i,k]
// ---------------------------------------------------------------------------
at::Tensor hc_post_fused_cpu(at::Tensor& x, at::Tensor& residual, at::Tensor& post, at::Tensor& comb) {
  RECORD_FUNCTION("sgl-kernel::hc_post_fused_cpu", {});
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16, "hc_post_fused_cpu: x must be bf16 [T, d]");
  TORCH_CHECK(
      residual.dim() == 3 && residual.scalar_type() == x.scalar_type(),
      "hc_post_fused_cpu: residual must be bf16 [T, hc, d]");
  TORCH_CHECK(post.scalar_type() == at::kFloat, "hc_post_fused_cpu: post must be float32");
  TORCH_CHECK(comb.scalar_type() == at::kFloat, "hc_post_fused_cpu: comb must be float32");
  TORCH_CHECK(
      x.device().is_cpu() && residual.device().is_cpu() && post.device().is_cpu() && comb.device().is_cpu(),
      "hc_post_fused_cpu: all inputs must be CPU tensors");

  const int64_t T = x.size(0);
  const int64_t d = x.size(1);
  const int64_t hc = residual.size(1);
  TORCH_CHECK(hc == 4, "hc_post_fused_cpu: only hc_mult=4 is supported");
  TORCH_CHECK(T > 0 && d > 0, "hc_post_fused_cpu: T and d must be positive");
  TORCH_CHECK(residual.size(0) == T && residual.size(2) == d, "hc_post_fused_cpu: residual shape must be [T, hc, d]");
  TORCH_CHECK(
      post.dim() == 2 && post.size(0) == T && post.size(1) == hc, "hc_post_fused_cpu: post shape must be [T, hc]");
  TORCH_CHECK(
      comb.dim() == 3 && comb.size(0) == T && comb.size(1) == hc && comb.size(2) == hc,
      "hc_post_fused_cpu: comb shape must be [T, hc, hc]");

  // Accept non-contiguous inputs and use contiguous copies/views internally.
  const at::Tensor x_c = x.is_contiguous() ? x : x.contiguous();
  const at::Tensor residual_c = residual.is_contiguous() ? residual : residual.contiguous();
  const at::Tensor post_c = post.is_contiguous() ? post : post.contiguous();
  const at::Tensor comb_c = comb.is_contiguous() ? comb : comb.contiguous();

  auto out = at::empty({T, hc, d}, x.options());  // same dtype as x

  hc_post_fuse_impl<c10::BFloat16, 4>(
      out.data_ptr<c10::BFloat16>(),
      x_c.data_ptr<c10::BFloat16>(),
      residual_c.data_ptr<c10::BFloat16>(),
      post_c.data_ptr<float>(),
      comb_c.data_ptr<float>(),
      T,
      d);

  return out;
}

// ---------------------------------------------------------------------------
// hc_head_fused_cpu
// Inputs:
//   x         [T, hc_mult, d]       bf16
//   hc_fn     [hc_mult, hc_mult*d]  float32  (projection weight)
//   hc_scale  scalar [1]            float32
//   hc_base   [hc_mult]             float32
// Output: y [T, d]  bf16
//
// Algorithm (fully fused, no intermediate scaled_x):
//   Per token: inv_rms = rsqrt(mean(x_flat^2) + norm_eps)
//   mixes[h] = dot(x_flat, hc_fn[h,:]) * inv_rms
//   pre[h]   = sigmoid(mixes[h] * hc_scale + hc_base[h]) + hc_eps
//   y        = sum_h pre[h] * x[h,:]  (SIMD over d)
// ---------------------------------------------------------------------------
at::Tensor hc_head_fused_cpu(
    at::Tensor& x, at::Tensor& hc_fn, at::Tensor& hc_scale, at::Tensor& hc_base, double hc_eps, double norm_eps) {
  RECORD_FUNCTION("sgl-kernel::hc_head_fused_cpu", {});
  TORCH_CHECK(x.dim() == 3 && x.scalar_type() == at::kBFloat16, "hc_head_fused_cpu: x must be bf16 [T, hc, d]");
  const int64_t hc_mult = x.size(1);
  TORCH_CHECK(hc_mult == 4, "hc_head_fused_cpu: only hc_mult=4 is supported");
  TORCH_CHECK(hc_fn.scalar_type() == at::kFloat, "hc_head_fused_cpu: hc_fn must be float32");
  TORCH_CHECK(
      hc_scale.scalar_type() == at::kFloat && hc_scale.numel() == 1,
      "hc_head_fused_cpu: hc_scale must be a scalar float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_head_fused_cpu: hc_base must be float32");
  TORCH_CHECK(hc_base.numel() == hc_mult, "hc_head_fused_cpu: hc_base.numel() must equal hc_mult");
  TORCH_CHECK(
      x.device().is_cpu() && hc_fn.device().is_cpu() && hc_scale.device().is_cpu() && hc_base.device().is_cpu(),
      "hc_head_fused_cpu: all inputs must be CPU tensors");

  const int64_t T = x.size(0);
  const int64_t d = x.size(2);
  TORCH_CHECK(T > 0 && d > 0, "hc_head_fused_cpu: T and d must be positive");
  TORCH_CHECK(
      hc_fn.dim() == 2 && hc_fn.size(0) == hc_mult && hc_fn.size(1) == hc_mult * d,
      "hc_head_fused_cpu: hc_fn must be [hc_mult, hc_mult*d]");

  // Accept non-contiguous inputs; make contiguous copies if needed.
  const at::Tensor x_c = x.is_contiguous() ? x : x.contiguous();
  const at::Tensor hc_fn_c = hc_fn.is_contiguous() ? hc_fn : hc_fn.contiguous();
  const at::Tensor hc_scale_c = hc_scale.is_contiguous() ? hc_scale : hc_scale.contiguous();
  const at::Tensor hc_base_c = hc_base.is_contiguous() ? hc_base : hc_base.contiguous();
  const float hc_scale_val = hc_scale_c.data_ptr<float>()[0];

  auto y = at::empty({T, d}, x.options());

  hc_head_fuse_impl<c10::BFloat16, 4>(
      y.data_ptr<c10::BFloat16>(),
      x_c.data_ptr<c10::BFloat16>(),
      hc_fn_c.data_ptr<float>(),
      hc_scale_val,
      hc_base_c.data_ptr<float>(),
      T,
      d,
      static_cast<float>(hc_eps),
      static_cast<float>(norm_eps));

  return y;
}
