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
//   * Data tensors (x, residual, y, out): bf16 (typical) or float32.
//   * Coefficient tensors (post, comb, hc_fn, hc_scale, hc_base): always float32.
//   * Internal GEMM intermediate (scaled_x): always float32.
//   * All arithmetic performed in float32; results converted back to input dtype.
//   * hc_mult=4 required (DeepSeek V4 default).
//   * GEMM delegated to at::mm (MKL/OpenBLAS).

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>

#include <cmath>
#include <limits>

#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

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

// ---------------------------------------------------------------------------
// hc_pre_scale_impl<scalar_t>
// Pass A: RMSnorm per token on x[T, hc_d], write float32 scaled_x[T, hc_d].
// scalar_t is bf16 or float32; scaled_x is always float32 for GEMM.
// ---------------------------------------------------------------------------
template <typename scalar_t>
static void hc_pre_scale_impl(
    float* __restrict__ scaled_x,    // [T, hc_d]  float32 output
    const scalar_t* __restrict__ x,  // [T, hc_d]  input (bf16 or float32)
    int64_t T,
    int64_t hc_d,
    float rms_eps) {
  static_assert(
      std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, c10::BFloat16>,
      "hc_pre_scale_impl: only float32 and bf16 are supported");
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  at::parallel_for(0, T, 0, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      const scalar_t* x_t = x + t * hc_d;
      float* sx_t = scaled_x + t * hc_d;

      fVec sq_acc(0.f);
      int64_t k;

      if constexpr (std::is_same_v<scalar_t, float>) {
        // float32: bVec::size() == fVec::size() (typically 16)
        for (k = 0; k <= hc_d - kVecSize; k += kVecSize) {
          fVec xv = fVec::loadu(x_t + k);
          sq_acc += xv * xv;
        }
        if (k < hc_d) {
          fVec xv = fVec::loadu(x_t + k, hc_d - k);
          sq_acc += xv * xv;
        }
      } else {
        // bf16/fp16: bVec::size() == 2 * fVec::size(), convert_to_float splits into two
        for (k = 0; k <= hc_d - kVecSize; k += kVecSize) {
          fVec x0, x1;
          std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
          sq_acc += x0 * x0 + x1 * x1;
        }
        if (k < hc_d) {
          fVec x0, x1;
          std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k, hc_d - k));
          sq_acc += x0 * x0 + x1 * x1;
        }
      }

      const double sum_sq = static_cast<double>(vec_reduce_sum(sq_acc));
      const float rsqrt =
          static_cast<float>(1.0 / std::sqrt(sum_sq / static_cast<double>(hc_d) + static_cast<double>(rms_eps)));
      const fVec rsqrt_fvec(rsqrt);

      if constexpr (std::is_same_v<scalar_t, float>) {
        // float32: direct store
        for (k = 0; k <= hc_d - kVecSize; k += kVecSize) {
          fVec xv = fVec::loadu(x_t + k);
          (xv * rsqrt_fvec).store(sx_t + k);
        }
        if (k < hc_d) (fVec::loadu(x_t + k, hc_d - k) * rsqrt_fvec).store(sx_t + k, hc_d - k);
      } else {
        // bf16/fp16: convert and store to float32 output
        for (k = 0; k <= hc_d - kVecSize; k += kVecSize) {
          fVec x0, x1;
          std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
          (x0 * rsqrt_fvec).store(sx_t + k);
          (x1 * rsqrt_fvec).store(sx_t + k + fVec::size());
        }
        if (k < hc_d) {
          const int64_t rem = hc_d - k;
          fVec x0, x1;
          std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k, rem));
          const int64_t rem0 = std::min(rem, (int64_t)fVec::size());
          const int64_t rem1 = rem - rem0;
          (x0 * rsqrt_fvec).store(sx_t + k, rem0);
          if (rem1 > 0) (x1 * rsqrt_fvec).store(sx_t + k + fVec::size(), rem1);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// hc_pre_scale_splitk_impl<scalar_t>
// Small-token path: split K (hc_d) across threads for better parallelism.
//
// Phase 1 (parallel over T*K_SPLITS): compute partial sq_sum per (token, k-split)
// Phase 2 (parallel over T):          reduce partials → per-token inv_rms
// Phase 3 (parallel over T*K_SPLITS): scale x → scaled_x using inv_rms
//
// Unlike hc_pre_scale_impl (which parallels over T only), this function exposes
// T*K_SPLITS parallel tasks, saturating more cores when T is small (e.g. decode).
// ---------------------------------------------------------------------------
template <typename scalar_t>
static void hc_pre_scale_splitk_impl(
    float* __restrict__ scaled_x,    // [T, hc_d] float32 output
    const scalar_t* __restrict__ x,  // [T, hc_d] input (bf16 or float32)
    int64_t T,
    int64_t hc_d,
    float rms_eps) {
  static_assert(
      std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, c10::BFloat16>,
      "hc_pre_scale_splitk_impl: only float32 and bf16 are supported");
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();

  // Split hc_d into K_SPLITS chunks: aim for ≥ num_threads tasks total (T * K_SPLITS).
  // Each chunk must be at least kVecSize elements for SIMD to be effective.
  const int64_t nthreads = at::get_num_threads();
  const int64_t max_splits = std::max(int64_t(1), hc_d / kVecSize);
  const int64_t K_SPLITS = std::min(nthreads, max_splits);
  const int64_t k_chunk = (hc_d + K_SPLITS - 1) / K_SPLITS;  // ceil-div

  // Phase 1: partial sq_sum[T * K_SPLITS] (parallel over T * K_SPLITS)
  std::vector<double> partial_sq(static_cast<size_t>(T * K_SPLITS), 0.0);
  at::parallel_for(0, T * K_SPLITS, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t t = idx / K_SPLITS;
      const int64_t ks = idx % K_SPLITS;
      const int64_t k0 = ks * k_chunk;
      const int64_t k1 = std::min(k0 + k_chunk, hc_d);
      const scalar_t* x_t = x + t * hc_d;

      fVec sq_acc(0.f);
      int64_t k;
      if constexpr (std::is_same_v<scalar_t, float>) {
        for (k = k0; k <= k1 - (int64_t)fVec::size(); k += fVec::size()) {
          fVec xv = fVec::loadu(x_t + k);
          sq_acc += xv * xv;
        }
        if (k < k1) {
          fVec xv = fVec::loadu(x_t + k, k1 - k);
          sq_acc += xv * xv;
        }
      } else {
        for (k = k0; k <= k1 - kVecSize; k += kVecSize) {
          fVec x0v, x1v;
          std::tie(x0v, x1v) = at::vec::convert_to_float(bVec::loadu(x_t + k));
          sq_acc += x0v * x0v + x1v * x1v;
        }
        if (k < k1) {
          fVec x0v, x1v;
          std::tie(x0v, x1v) = at::vec::convert_to_float(bVec::loadu(x_t + k, k1 - k));
          sq_acc += x0v * x0v + x1v * x1v;
        }
      }
      partial_sq[static_cast<size_t>(idx)] = static_cast<double>(vec_reduce_sum(sq_acc));
    }
  });

  // Phase 2: reduce partials → inv_rms[T].
  // Serial: T <= kSmallTokenThreshold (≤8) and K_SPLITS ≈ nthreads (64-128),
  // so total work is ≤ 8 * 128 = 1024 scalar adds — parallel overhead not worth it.
  std::vector<float> inv_rms(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    double sum_sq = 0.0;
    for (int64_t ks = 0; ks < K_SPLITS; ++ks)
      sum_sq += partial_sq[static_cast<size_t>(t * K_SPLITS + ks)];
    inv_rms[static_cast<size_t>(t)] =
        static_cast<float>(1.0 / std::sqrt(sum_sq / static_cast<double>(hc_d) + static_cast<double>(rms_eps)));
  }

  // Phase 3: scale x → scaled_x (parallel over T * K_SPLITS)
  at::parallel_for(0, T * K_SPLITS, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t t = idx / K_SPLITS;
      const int64_t ks = idx % K_SPLITS;
      const int64_t k0 = ks * k_chunk;
      const int64_t k1 = std::min(k0 + k_chunk, hc_d);
      const scalar_t* x_t = x + t * hc_d;
      float* sx_t = scaled_x + t * hc_d;
      const float irms = inv_rms[static_cast<size_t>(t)];
      const fVec irms_vec(irms);

      int64_t k;
      if constexpr (std::is_same_v<scalar_t, float>) {
        for (k = k0; k <= k1 - (int64_t)fVec::size(); k += fVec::size()) {
          (fVec::loadu(x_t + k) * irms_vec).store(sx_t + k);
        }
        if (k < k1) (fVec::loadu(x_t + k, k1 - k) * irms_vec).store(sx_t + k, k1 - k);
      } else {
        for (k = k0; k <= k1 - kVecSize; k += kVecSize) {
          fVec x0v, x1v;
          std::tie(x0v, x1v) = at::vec::convert_to_float(bVec::loadu(x_t + k));
          (x0v * irms_vec).store(sx_t + k);
          (x1v * irms_vec).store(sx_t + k + (int64_t)fVec::size());
        }
        if (k < k1) {
          const int64_t rem = k1 - k;
          fVec x0v, x1v;
          std::tie(x0v, x1v) = at::vec::convert_to_float(bVec::loadu(x_t + k, rem));
          const int64_t rem0 = std::min(rem, (int64_t)fVec::size());
          const int64_t rem1 = rem - rem0;
          (x0v * irms_vec).store(sx_t + k, rem0);
          if (rem1 > 0) (x1v * irms_vec).store(sx_t + k + fVec::size(), rem1);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// hc_pre_combine_impl<scalar_t, HC>
// Pass B: mixes[T, mix_hc] + x[T, HC, d] → y[T,d], post[T,HC], comb[T,HC*HC]
// y is same dtype as x (bf16 or float32); post and comb are always float32.
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_pre_combine_impl(
    scalar_t* __restrict__ y,         // [T, d]      output
    float* __restrict__ post_out,     // [T, HC]     float32 output
    float* __restrict__ comb_out,     // [T, HC*HC]  float32 output
    const float* __restrict__ mixes,  // [T, mix_hc] float32 input
    const scalar_t* __restrict__ x,   // [T, HC, d]  input
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    int64_t T,
    int64_t d,
    int sinkhorn_iters,
    float hc_eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  constexpr int mix_hc = (2 + HC) * HC;
  const float s0 = hc_scale[0], s1 = hc_scale[1], s2 = hc_scale[2];

  at::parallel_for(0, T, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float pre[HC];
    alignas(64) float cm[HC * HC];

    for (int64_t t = begin; t < end; ++t) {
      const float* m = mixes + t * mix_hc;
      float* po = post_out + t * HC;
      float* co = comb_out + t * HC * HC;

      parse_mixes_and_sinkhorn<HC>(pre, po, cm, m, s0, s1, s2, hc_base, sinkhorn_iters, hc_eps);
      for (int i = 0; i < HC * HC; ++i)
        co[i] = cm[i];

      fVec pre_fvec[HC];
      for (int h = 0; h < HC; ++h)
        pre_fvec[h] = fVec(pre[h]);

      const scalar_t* x_t = x + t * HC * d;
      scalar_t* y_t = y + t * d;

      int64_t k;
      if constexpr (std::is_same_v<scalar_t, float>) {
        // float32: direct vectorized sum
        for (k = 0; k <= d - fVec::size(); k += fVec::size()) {
          fVec acc = fVec::loadu(x_t + k) * pre_fvec[0];
          for (int h = 1; h < HC; ++h)
            acc += fVec::loadu(x_t + h * d + k) * pre_fvec[h];
          acc.store(y_t + k);
        }
      } else {
        // bf16/fp16: convert and accumulate
        for (k = 0; k <= d - kVecSize; k += kVecSize) {
          fVec acc0, acc1;
          std::tie(acc0, acc1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
          acc0 *= pre_fvec[0];
          acc1 *= pre_fvec[0];
          for (int h = 1; h < HC; ++h) {
            fVec f0, f1;
            std::tie(f0, f1) = at::vec::convert_to_float(bVec::loadu(x_t + h * d + k));
            acc0 += f0 * pre_fvec[h];
            acc1 += f1 * pre_fvec[h];
          }
          at::vec::convert_from_float<scalar_t>(acc0, acc1).store(y_t + k);
        }
      }
      if (k < d) {
        if constexpr (std::is_same_v<scalar_t, float>) {
          const int64_t rem = d - k;
          fVec acc = fVec::loadu(x_t + k, rem) * pre_fvec[0];
          for (int h = 1; h < HC; ++h)
            acc += fVec::loadu(x_t + h * d + k, rem) * pre_fvec[h];
          acc.store(y_t + k, rem);
        } else {
          const int64_t rem = d - k;
          fVec acc0, acc1;
          std::tie(acc0, acc1) = at::vec::convert_to_float(bVec::loadu(x_t + k, rem));
          acc0 *= pre_fvec[0];
          acc1 *= pre_fvec[0];
          for (int h = 1; h < HC; ++h) {
            fVec f0, f1;
            std::tie(f0, f1) = at::vec::convert_to_float(bVec::loadu(x_t + h * d + k, rem));
            acc0 += f0 * pre_fvec[h];
            acc1 += f1 * pre_fvec[h];
          }
          at::vec::convert_from_float<scalar_t>(acc0, acc1).store(y_t + k, rem);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// hc_post_impl<scalar_t, HC>
// out[t,h,k] = post[t,h]*x[t,k] + sum_i comb[t,i,h]*residual[t,i,k]
// x and residual are bf16 or float32; post and comb are float32.
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_post_impl(
    scalar_t* __restrict__ out,             // [T, HC, d]  output
    const scalar_t* __restrict__ x,         // [T, d]      input
    const scalar_t* __restrict__ residual,  // [T, HC, d]  input
    const float* __restrict__ post,         // [T, HC]     float32
    const float* __restrict__ comb,         // [T, HC, HC] float32 row-major
    int64_t T,
    int64_t d) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  at::parallel_for(0, T * HC, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t t = idx / HC;
      const int h = static_cast<int>(idx % HC);

      const float post_val = post[t * HC + h];
      const float* comb_t = comb + t * HC * HC;
      const scalar_t* x_t = x + t * d;
      const scalar_t* res_t = residual + t * HC * d;
      scalar_t* out_th = out + (t * HC + h) * d;

      fVec post_fvec(post_val);
      fVec comb_fvec[HC];
      for (int i = 0; i < HC; ++i)
        comb_fvec[i] = fVec(comb_t[i * HC + h]);

      int64_t k;
      if constexpr (std::is_same_v<scalar_t, float>) {
        // float32: direct vectorized computation
        for (k = 0; k <= d - fVec::size(); k += fVec::size()) {
          fVec acc = post_fvec * fVec::loadu(x_t + k);
          for (int i = 0; i < HC; ++i)
            acc += comb_fvec[i] * fVec::loadu(res_t + i * d + k);
          acc.store(out_th + k);
        }
      } else {
        // bf16/fp16: convert and accumulate
        for (k = 0; k <= d - kVecSize; k += kVecSize) {
          // post term
          bVec xbv = bVec::loadu(x_t + k);
          fVec acc0, acc1;
          std::tie(acc0, acc1) = at::vec::convert_to_float(xbv);
          acc0 = post_fvec * acc0;
          acc1 = post_fvec * acc1;

          // residual terms: sum_i comb[t,i,h] * residual[t,i,k]
          for (int i = 0; i < HC; ++i) {
            bVec rbv = bVec::loadu(res_t + i * d + k);
            fVec r0, r1;
            std::tie(r0, r1) = at::vec::convert_to_float(rbv);
            acc0 += comb_fvec[i] * r0;
            acc1 += comb_fvec[i] * r1;
          }

          at::vec::convert_from_float<scalar_t>(acc0, acc1).store(out_th + k);
        }
      }
      if (k < d) {
        if constexpr (std::is_same_v<scalar_t, float>) {
          const int64_t rem = d - k;
          fVec acc = post_fvec * fVec::loadu(x_t + k, rem);
          for (int i = 0; i < HC; ++i)
            acc += comb_fvec[i] * fVec::loadu(res_t + i * d + k, rem);
          acc.store(out_th + k, rem);
        } else {
          const int64_t rem = d - k;
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
    }
  });
}

// ---------------------------------------------------------------------------
// hc_post_splitk_impl<scalar_t, HC>
// Same formula as hc_post_impl but parallelised over T * HC * K_SPLITS.
// Each output element out[t,h,k] is fully independent, so k can be split
// without any reduction step – unlike the RMSnorm sq-sum case.
// Used when T*HC is small (e.g. decode: T=1 → only 4 tasks in hc_post_impl).
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_post_splitk_impl(
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
  const int64_t K_SPLITS = std::min(nthreads, max_splits);
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
      if constexpr (std::is_same_v<scalar_t, float>) {
        for (k = k0; k <= k1 - (int64_t)fVec::size(); k += fVec::size()) {
          fVec acc = post_fvec * fVec::loadu(x_t + k);
          for (int i = 0; i < HC; ++i)
            acc += comb_fvec[i] * fVec::loadu(res_t + i * d + k);
          acc.store(out_th + k);
        }
      } else {
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
      }
      if (k < k1) {
        if constexpr (std::is_same_v<scalar_t, float>) {
          const int64_t rem = k1 - k;
          fVec acc = post_fvec * fVec::loadu(x_t + k, rem);
          for (int i = 0; i < HC; ++i)
            acc += comb_fvec[i] * fVec::loadu(res_t + i * d + k, rem);
          acc.store(out_th + k, rem);
        } else {
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
    }
  });
}

// ---------------------------------------------------------------------------
// hc_head_gemm_fuse_impl<scalar_t, HC>
// Fully fused hc_head path (large-T focused):
//   1) token-wise rms (inv_rms)
//   2) tiny GEMM (N=HC=4): mixes_raw[t,h] = dot(x_flat[t,:], hc_fn[h,:])
//   3) gate: pre[h] = sigmoid((mixes_raw[h] * inv_rms) * hc_scale + hc_base[h]) + hc_eps
//   4) combine: y[t,:] = sum_h pre[h] * x[t,h,:]
//
// Compared with the old path, this avoids materialising scaled_x[T, HC*d]
// and removes an extra large read/write pass over x for prefill workloads.
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_head_gemm_fuse_impl(
    scalar_t* __restrict__ y,         // [T, d]      output
    const scalar_t* __restrict__ x,   // [T, HC, d]  input
    const float* __restrict__ hc_fn,  // [HC, HC*d]  float32
    float hc_scale_val,
    const float* __restrict__ hc_base,  // [HC]
    int64_t T,
    int64_t d,
    float hc_eps,
    float norm_eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const int64_t hc_d = HC * d;

  at::parallel_for(0, T, 0, [&](int64_t begin, int64_t end) {
    std::vector<float> scratch(d);
    float* sc = scratch.data();

    for (int64_t t = begin; t < end; ++t) {
      const scalar_t* x_t = x + t * hc_d;
      scalar_t* y_t = y + t * d;

      // Pass 1: rms + tiny gemm (4 output channels) over flattened [HC*d].
      fVec sq_acc(0.f);
      fVec dot_acc[HC] = {fVec(0.f), fVec(0.f), fVec(0.f), fVec(0.f)};

      int64_t k = 0;
      if constexpr (std::is_same_v<scalar_t, float>) {
        for (; k <= hc_d - (int64_t)fVec::size(); k += fVec::size()) {
          fVec xv = fVec::loadu(x_t + k);
          sq_acc += xv * xv;
          for (int h = 0; h < HC; ++h) {
            dot_acc[h] += xv * fVec::loadu(hc_fn + h * hc_d + k);
          }
        }
        if (k < hc_d) {
          const int64_t rem = hc_d - k;
          fVec xv = fVec::loadu(x_t + k, rem);
          sq_acc += xv * xv;
          for (int h = 0; h < HC; ++h) {
            dot_acc[h] += xv * fVec::loadu(hc_fn + h * hc_d + k, rem);
          }
        }
      } else {
        for (; k <= hc_d - kVecSize; k += kVecSize) {
          fVec x0, x1;
          std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
          sq_acc += x0 * x0 + x1 * x1;
          for (int h = 0; h < HC; ++h) {
            const float* w = hc_fn + h * hc_d + k;
            dot_acc[h] += x0 * fVec::loadu(w);
            dot_acc[h] += x1 * fVec::loadu(w + (int64_t)fVec::size());
          }
        }
        if (k < hc_d) {
          const int64_t rem = hc_d - k;
          fVec x0, x1;
          std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k, rem));
          sq_acc += x0 * x0 + x1 * x1;
          const int64_t rem0 = std::min(rem, (int64_t)fVec::size());
          const int64_t rem1 = rem - rem0;
          for (int h = 0; h < HC; ++h) {
            const float* w = hc_fn + h * hc_d + k;
            dot_acc[h] += x0 * fVec::loadu(w, rem0);
            if (rem1 > 0) dot_acc[h] += x1 * fVec::loadu(w + (int64_t)fVec::size(), rem1);
          }
        }
      }

      const double sum_sq = static_cast<double>(vec_reduce_sum(sq_acc));
      const float inv_rms =
          static_cast<float>(1.0 / std::sqrt(sum_sq / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));

      float pre[HC];
      for (int h = 0; h < HC; ++h) {
        const float mix_h = static_cast<float>(vec_reduce_sum(dot_acc[h])) * inv_rms;
        const float gate_in = mix_h * hc_scale_val + hc_base[h];
        pre[h] = 1.f / (1.f + std::exp(-gate_in)) + hc_eps;
      }

      // Pass 2: combine with h-outer order for sequential x[t,h,:] streaming.
      if constexpr (std::is_same_v<scalar_t, float>) {
        std::memset(y_t, 0, d * sizeof(float));
        for (int h = 0; h < HC; ++h) {
          const fVec pre_fvec(pre[h]);
          const float* x_th = x_t + h * d;
          int64_t kk = 0;
          for (; kk <= d - (int64_t)fVec::size(); kk += fVec::size()) {
            (fVec::loadu(y_t + kk) + pre_fvec * fVec::loadu(x_th + kk)).store(y_t + kk);
          }
          if (kk < d) {
            const int64_t rem = d - kk;
            (fVec::loadu(y_t + kk, rem) + pre_fvec * fVec::loadu(x_th + kk, rem)).store(y_t + kk, rem);
          }
        }
      } else {
        std::memset(sc, 0, d * sizeof(float));
        for (int h = 0; h < HC; ++h) {
          const fVec pre_fvec(pre[h]);
          const scalar_t* x_th = x_t + h * d;
          int64_t kk = 0;
          for (; kk <= d - kVecSize; kk += kVecSize) {
            fVec x0, x1;
            std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_th + kk));
            (fVec::loadu(sc + kk) + pre_fvec * x0).store(sc + kk);
            (fVec::loadu(sc + kk + (int64_t)fVec::size()) + pre_fvec * x1).store(sc + kk + (int64_t)fVec::size());
          }
          if (kk < d) {
            const int64_t rem = d - kk;
            fVec x0, x1;
            std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_th + kk, rem));
            const int64_t rem0 = std::min(rem, (int64_t)fVec::size());
            const int64_t rem1 = rem - rem0;
            (fVec::loadu(sc + kk, rem0) + pre_fvec * x0).store(sc + kk, rem0);
            if (rem1 > 0)
              (fVec::loadu(sc + kk + (int64_t)fVec::size(), rem1) + pre_fvec * x1)
                  .store(sc + kk + (int64_t)fVec::size(), rem1);
          }
        }

        int64_t kk = 0;
        for (; kk <= d - kVecSize; kk += kVecSize) {
          at::vec::convert_from_float<scalar_t>(fVec::loadu(sc + kk), fVec::loadu(sc + kk + (int64_t)fVec::size()))
              .store(y_t + kk);
        }
        if (kk < d) {
          const int64_t rem = d - kk;
          const int64_t rem0 = std::min(rem, (int64_t)fVec::size());
          const int64_t rem1 = rem - rem0;
          at::vec::convert_from_float<scalar_t>(
              fVec::loadu(sc + kk, rem0), rem1 > 0 ? fVec::loadu(sc + kk + (int64_t)fVec::size(), rem1) : fVec(0.f))
              .store(y_t + kk, rem);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// hc_head_brgemm_fuse_impl<scalar_t, HC>
// Large-T fused path built on top of existing brgemm:
//   1) per-token inv_rms on x_flat
//   2) block GEMM raw_mixes = x_flat @ hc_fn.T via brgemm
//   3) scale raw_mixes by inv_rms, apply sigmoid gate
//   4) combine y = sum_h pre[h] * x[h,:]
//
// This removes scaled_x materialization while still reusing the existing GEMM
// kernel instead of a handwritten dot-product loop.
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_head_brgemm_fuse_impl(
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ x,
    const float* __restrict__ hc_fn,  // [HC, HC*d] row-major (NOT transposed)
    float hc_scale_val,
    const float* __restrict__ hc_base,
    int64_t T,
    int64_t d,
    float hc_eps,
    float norm_eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();
  const int64_t hc_d = HC * d;

  // ── Parallelism: grain=1 gives T independent tasks (e.g. 128 for T=128),
  // saturating ~40 threads with ~3 tokens/thread.  No K-split is needed
  // because T >> #threads; K-split only helps for T < #threads (decode), and
  // that case already falls back to the splitk path (kSmallTokenThreshold=8).
  //
  // N=HC=4 is below tinygemm_kernel's minimum block width (16).  Instead we
  // fuse the RMSnorm sq-sum with the HC dot-products in ONE pass over x[t,:],
  // so each 64-KB token row is read only TWICE (sq+gemm fused, then combine)
  // rather than three times (sq pass, GEMM pass, combine pass separately).
  at::parallel_for(0, T, 1, [&](int64_t begin, int64_t end) {
    // Per-thread scratch – sized for one token at a time to minimise memory.
    // a_buf: bf16→float upcast of one token row (float path: unused / zero-size)
    // sc   : fp32 combine accumulator (bf16 output path only)
    std::vector<float> a_buf(std::is_same_v<scalar_t, float> ? size_t{0} : static_cast<size_t>(hc_d));
    std::vector<float> sc(static_cast<size_t>(d));
    float* const a_buf_ptr = a_buf.empty() ? nullptr : a_buf.data();
    float* const sc_ptr = sc.data();

    for (int64_t t = begin; t < end; ++t) {
      const scalar_t* x_t = x + t * hc_d;
      scalar_t* y_t = y + t * d;

      // ── Step 1: optional bf16→float upcast + fused sq_acc + HC dot-products
      //    One pass over a_row[hc_d]. For float: a_row == x_t (zero-copy).
      const float* a_row;
      if constexpr (!std::is_same_v<scalar_t, float>) {
        // Upcast bf16 → float into a_buf
        int64_t k = 0;
        for (; k <= hc_d - kVecSize; k += kVecSize) {
          fVec x0, x1;
          std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
          x0.store(a_buf_ptr + k);
          x1.store(a_buf_ptr + k + (int64_t)fVec::size());
        }
        if (k < hc_d) {
          const int64_t rem = hc_d - k;
          fVec x0, x1;
          std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k, rem));
          const int64_t rem0 = std::min(rem, (int64_t)fVec::size());
          const int64_t rem1 = rem - rem0;
          x0.store(a_buf_ptr + k, rem0);
          if (rem1 > 0) x1.store(a_buf_ptr + k + (int64_t)fVec::size(), rem1);
        }
        a_row = a_buf_ptr;
      } else {
        a_row = reinterpret_cast<const float*>(x_t);
      }

      // Fused sq_acc + HC dot-product accumulators – single vectorised pass.
      // hc_fn[h, k] is contiguous per head (row-major [HC, hc_d]).
      fVec sq_acc(0.f);
      fVec dot_acc[HC];
      for (int h = 0; h < HC; ++h)
        dot_acc[h] = fVec(0.f);

      int64_t k = 0;
      for (; k <= hc_d - (int64_t)fVec::size(); k += fVec::size()) {
        const fVec xv = fVec::loadu(a_row + k);
        sq_acc += xv * xv;
        for (int h = 0; h < HC; ++h)
          dot_acc[h] += xv * fVec::loadu(hc_fn + h * hc_d + k);
      }
      if (k < hc_d) {
        const int64_t rem = hc_d - k;
        const fVec xv = fVec::loadu(a_row + k, rem);
        sq_acc += xv * xv;
        for (int h = 0; h < HC; ++h)
          dot_acc[h] += xv * fVec::loadu(hc_fn + h * hc_d + k, rem);
      }

      const float inv_rms = static_cast<float>(
          1.0 /
          std::sqrt(
              static_cast<double>(vec_reduce_sum(sq_acc)) / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));

      float pre[HC];
      for (int h = 0; h < HC; ++h) {
        const float mix_h = vec_reduce_sum(dot_acc[h]) * inv_rms;
        const float gate_in = mix_h * hc_scale_val + hc_base[h];
        pre[h] = 1.f / (1.f + std::exp(-gate_in)) + hc_eps;
      }

      // ── Step 2: h-outer combine (second pass over x_t) ─────────────────
      if constexpr (std::is_same_v<scalar_t, float>) {
        float* y_tf = reinterpret_cast<float*>(y_t);
        std::memset(y_tf, 0, d * sizeof(float));
        for (int h = 0; h < HC; ++h) {
          const fVec pre_fvec(pre[h]);
          const float* x_th = reinterpret_cast<const float*>(x_t + (int64_t)h * d);
          int64_t kk = 0;
          for (; kk <= d - (int64_t)fVec::size(); kk += fVec::size())
            (fVec::loadu(y_tf + kk) + pre_fvec * fVec::loadu(x_th + kk)).store(y_tf + kk);
          if (kk < d) {
            const int64_t rem = d - kk;
            (fVec::loadu(y_tf + kk, rem) + pre_fvec * fVec::loadu(x_th + kk, rem)).store(y_tf + kk, rem);
          }
        }
      } else {
        std::memset(sc_ptr, 0, d * sizeof(float));
        for (int h = 0; h < HC; ++h) {
          const fVec pre_fvec(pre[h]);
          const scalar_t* x_th = x_t + (int64_t)h * d;
          int64_t kk = 0;
          for (; kk <= d - kVecSize; kk += kVecSize) {
            fVec x0, x1;
            std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_th + kk));
            (fVec::loadu(sc_ptr + kk) + pre_fvec * x0).store(sc_ptr + kk);
            (fVec::loadu(sc_ptr + kk + (int64_t)fVec::size()) + pre_fvec * x1)
                .store(sc_ptr + kk + (int64_t)fVec::size());
          }
          if (kk < d) {
            const int64_t rem = d - kk;
            fVec x0, x1;
            std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_th + kk, rem));
            const int64_t rem0 = std::min(rem, (int64_t)fVec::size());
            const int64_t rem1 = rem - rem0;
            (fVec::loadu(sc_ptr + kk, rem0) + pre_fvec * x0).store(sc_ptr + kk, rem0);
            if (rem1 > 0)
              (fVec::loadu(sc_ptr + kk + (int64_t)fVec::size(), rem1) + pre_fvec * x1)
                  .store(sc_ptr + kk + (int64_t)fVec::size(), rem1);
          }
        }
        // Convert fp32 accumulator → bf16 output
        int64_t kk = 0;
        for (; kk <= d - kVecSize; kk += kVecSize) {
          at::vec::convert_from_float<scalar_t>(
              fVec::loadu(sc_ptr + kk), fVec::loadu(sc_ptr + kk + (int64_t)fVec::size()))
              .store(y_t + kk);
        }
        if (kk < d) {
          const int64_t rem = d - kk;
          const int64_t rem0 = std::min(rem, (int64_t)fVec::size());
          const int64_t rem1 = rem - rem0;
          at::vec::convert_from_float<scalar_t>(
              fVec::loadu(sc_ptr + kk, rem0),
              rem1 > 0 ? fVec::loadu(sc_ptr + kk + (int64_t)fVec::size(), rem1) : fVec(0.f))
              .store(y_t + kk, rem);
        }
      }
    }  // for t
  });  // parallel_for
}

// ---------------------------------------------------------------------------
// hc_head_combine_impl<scalar_t, HC>
// Pass B for hc_head: mixes[T, HC] + x[T, HC, d] → y[T, d]
//   pre[t,h] = sigmoid(mixes[t,h] * hc_scale + hc_base[h]) + hc_eps
//   y[t,k]   = sum_h pre[t,h] * x[t,h,k]
// x is bf16 or float32; mixes is float32; y is same dtype as x.
//
// Two optimizations over naive k-outer / h-inner:
//   1. Loop order: h-outer, k-inner → x[t,h,:] accessed sequentially, one
//      row at a time, so each cache line is loaded exactly once.
//      The naive k-outer loop jumps between 4 rows separated by d*sizeof(T)
//      bytes, causing cache thrashing when HC*d exceeds L1 size.
//   2. fp32 scratch buffer: accumulate into a per-thread float32 scratch
//      array for the entire token, then write y once at the end.
//      This avoids HC round-trips of reading/writing y in bf16, and for
//      the bf16 path eliminates HC−1 extra convert_from_float → rounding
//      errors that would otherwise accumulate across h iterations.
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_head_combine_impl(
    scalar_t* __restrict__ y,         // [T, d]      output
    const float* __restrict__ mixes,  // [T, HC]     float32 input
    const scalar_t* __restrict__ x,   // [T, HC, d]  input
    float hc_scale_val,
    const float* __restrict__ hc_base,  // [HC]
    int64_t T,
    int64_t d,
    float hc_eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  at::parallel_for(0, T, 0, [&](int64_t begin, int64_t end) {
    // Per-thread fp32 scratch: accumulate here, convert/write y once at end.
    // Fits in L1 cache (d * 4 bytes = 16KB for d=4096) and is reused across
    // all h iterations, so no repeated bf16 round-trips.
    std::vector<float> scratch(d);
    float* sc = scratch.data();

    alignas(64) float pre[HC];
    for (int64_t t = begin; t < end; ++t) {
      const float* m = mixes + t * HC;
      // Compute sigmoid pre-values once per token
      for (int h = 0; h < HC; ++h) {
        float v = m[h] * hc_scale_val + hc_base[h];
        pre[h] = 1.0f / (1.0f + std::exp(-v)) + hc_eps;
      }

      const scalar_t* x_t = x + t * HC * d;
      scalar_t* y_t = y + t * d;

      // Zero scratch buffer for this token
      std::memset(sc, 0, d * sizeof(float));

      // h-outer accumulation into fp32 scratch
      for (int h = 0; h < HC; ++h) {
        const fVec pre_fvec(pre[h]);
        const scalar_t* x_th = x_t + h * d;  // sequential access to x[t,h,:]

        if constexpr (std::is_same_v<scalar_t, float>) {
          int64_t k = 0;
          for (; k <= d - fVec::size(); k += fVec::size())
            (fVec::loadu(sc + k) + pre_fvec * fVec::loadu(x_th + k)).store(sc + k);
          if (k < d) {
            int64_t rem = d - k;
            (fVec::loadu(sc + k, rem) + pre_fvec * fVec::loadu(x_th + k, rem)).store(sc + k, rem);
          }
        } else {
          // bf16: convert x to fp32, accumulate into scratch (stays fp32)
          int64_t k = 0;
          for (; k <= d - kVecSize; k += kVecSize) {
            fVec x0, x1;
            std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_th + k));
            (fVec::loadu(sc + k) + pre_fvec * x0).store(sc + k);
            (fVec::loadu(sc + k + fVec::size()) + pre_fvec * x1).store(sc + k + fVec::size());
          }
          if (k < d) {
            int64_t rem = d - k;
            fVec x0, x1;
            std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_th + k, rem));
            int64_t rem0 = std::min(rem, (int64_t)fVec::size());
            int64_t rem1 = rem - rem0;
            (fVec::loadu(sc + k, rem0) + pre_fvec * x0).store(sc + k, rem0);
            if (rem1 > 0) (fVec::loadu(sc + k + fVec::size(), rem1) + pre_fvec * x1).store(sc + k + fVec::size(), rem1);
          }
        }
      }

      // Write scratch → y (convert fp32 → output dtype, single pass)
      if constexpr (std::is_same_v<scalar_t, float>) {
        std::memcpy(y_t, sc, d * sizeof(float));
      } else {
        int64_t k = 0;
        for (; k <= d - kVecSize; k += kVecSize)
          at::vec::convert_from_float<scalar_t>(fVec::loadu(sc + k), fVec::loadu(sc + k + fVec::size())).store(y_t + k);
        if (k < d) {
          int64_t rem = d - k;
          int64_t rem0 = std::min(rem, (int64_t)fVec::size());
          int64_t rem1 = rem - rem0;
          at::vec::convert_from_float<scalar_t>(
              fVec::loadu(sc + k, rem0), rem1 > 0 ? fVec::loadu(sc + k + fVec::size(), rem1) : fVec(0.f))
              .store(y_t + k, rem);
        }
      }
    }
  });
}

}  // anonymous namespace

// ============================================================================
// Public interface
// ============================================================================

// ---------------------------------------------------------------------------
// hc_pre_fused_cpu
// Inputs:
//   x         [T, hc_mult, d]          bf16 or float32
//   hc_fn     [mix_hc, hc_mult*d]      float32   (projection weight)
//   hc_scale  [3]                      float32
//   hc_base   [mix_hc]                 float32
// Outputs: (y [T,d] same dtype as x,  post [T,hc_mult] float32,
//           comb [T,hc_mult,hc_mult]  float32)
//
// Algorithm:
//   Pass A (parallel): scaled_x[T, hc_d] = x_flat * rsqrt  (RMSnorm scale)
//   GEMM: mixes = at::mm(scaled_x, hc_fn.T)  [T, mix_hc]
//   Pass B (parallel): sinkhorn + weighted combine
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
  TORCH_CHECK(
      x.dim() == 3 && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat),
      "hc_pre_fused_cpu: x must be bf16 or float32 [T, hc, d]");
  TORCH_CHECK(hc_fn.scalar_type() == at::kFloat, "hc_pre_fused_cpu: hc_fn must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_pre_fused_cpu: hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_pre_fused_cpu: hc_base must be float32");

  // Accept non-contiguous inputs and use contiguous copies/views internally.
  const at::Tensor x_c = x.is_contiguous() ? x : x.contiguous();
  const at::Tensor hc_fn_c = hc_fn.is_contiguous() ? hc_fn : hc_fn.contiguous();
  const at::Tensor hc_scale_c = hc_scale.is_contiguous() ? hc_scale : hc_scale.contiguous();
  const at::Tensor hc_base_c = hc_base.is_contiguous() ? hc_base : hc_base.contiguous();

  const int64_t T = x.size(0);
  const int64_t d = x.size(2);
  const int64_t hc_d = hc_mult * d;
  const int64_t mix_hc = (2 + hc_mult) * hc_mult;

  // Small-token fused path: avoid scaled_x materialization + at::mm overhead.
  // Parallelized on (T * mix_hc) to keep CPU utilization when T is small.
  // DeepSeek-v4 mix_hc is 24 for hc_mult=4.
  constexpr int64_t kSmallTokenThreshold = 8;
  const bool use_small_token_fused = (T <= kSmallTokenThreshold);

  auto f32_opts = at::TensorOptions().dtype(at::kFloat).device(x.device());
  auto mixes = at::empty({T, mix_hc}, f32_opts);

  // Produce scaled_x [T, hc_d] (float32), then GEMM → mixes [T, mix_hc].
  // Small-T path: split K (hc_d) for better multi-core utilisation (T*K_SPLITS tasks).
  // Large-T path: split over T tokens (T tasks, sufficient when T is large).
  auto scaled_x = at::empty({T, hc_d}, f32_opts);

  if (use_small_token_fused) {
    // K-split parallel RMSnorm scale (Phase 1: partial sq_sum, Phase 2: inv_rms, Phase 3: scale)
    if (x_c.scalar_type() == at::kBFloat16) {
      hc_pre_scale_splitk_impl<c10::BFloat16>(
          scaled_x.data_ptr<float>(), x_c.data_ptr<c10::BFloat16>(), T, hc_d, static_cast<float>(rms_eps));
    } else if (x_c.scalar_type() == at::kFloat) {
      hc_pre_scale_splitk_impl<float>(
          scaled_x.data_ptr<float>(), x_c.data_ptr<float>(), T, hc_d, static_cast<float>(rms_eps));
    } else {
      TORCH_CHECK(false, "hc_pre_fused_cpu: unexpected scalar_type");
    }
  } else {
    // T-parallel RMSnorm scale
    if (x_c.scalar_type() == at::kBFloat16) {
      hc_pre_scale_impl<c10::BFloat16>(
          scaled_x.data_ptr<float>(), x_c.data_ptr<c10::BFloat16>(), T, hc_d, static_cast<float>(rms_eps));
    } else if (x_c.scalar_type() == at::kFloat) {
      hc_pre_scale_impl<float>(scaled_x.data_ptr<float>(), x_c.data_ptr<float>(), T, hc_d, static_cast<float>(rms_eps));
    } else {
      TORCH_CHECK(false, "hc_pre_fused_cpu: unexpected scalar_type");
    }
  }

  // GEMM: mixes [T, mix_hc] = scaled_x @ hc_fn.T  (both paths use at::mm)
  mixes = at::mm(scaled_x, hc_fn_c.t());

  // Pass B: sinkhorn + weighted combine
  TORCH_CHECK(mixes.size(1) == mix_hc);

  auto post = at::empty({T, hc_mult}, f32_opts);           // always float32
  auto comb = at::empty({T, hc_mult, hc_mult}, f32_opts);  // always float32
  auto y = at::empty({T, d}, x.options());                 // same dtype as x

  if (x_c.scalar_type() == at::kBFloat16) {
    hc_pre_combine_impl<c10::BFloat16, 4>(
        y.data_ptr<c10::BFloat16>(),
        post.data_ptr<float>(),
        comb.data_ptr<float>(),
        mixes.contiguous().data_ptr<float>(),
        x_c.data_ptr<c10::BFloat16>(),
        hc_scale_c.data_ptr<float>(),
        hc_base_c.data_ptr<float>(),
        T,
        d,
        static_cast<int>(sinkhorn_iters),
        static_cast<float>(hc_eps));
  } else if (x_c.scalar_type() == at::kFloat) {
    hc_pre_combine_impl<float, 4>(
        y.data_ptr<float>(),
        post.data_ptr<float>(),
        comb.data_ptr<float>(),
        mixes.contiguous().data_ptr<float>(),
        x_c.data_ptr<float>(),
        hc_scale_c.data_ptr<float>(),
        hc_base_c.data_ptr<float>(),
        T,
        d,
        static_cast<int>(sinkhorn_iters),
        static_cast<float>(hc_eps));
  } else {
    TORCH_CHECK(false, "hc_pre_fused_cpu: unexpected scalar_type");
  }

  return {y, post, comb};
}

// ---------------------------------------------------------------------------
// hc_post_fused_cpu
// Inputs:
//   x         [T, d]         bf16 or float32  (sublayer output)
//   residual  [T, hc_mult, d] same dtype       (pre-sublayer residual)
//   post      [T, hc_mult]   float32
//   comb      [T, hc_mult, hc_mult]  float32
// Output: out [T, hc_mult, d]  same dtype as x
//
// Formula: out[t,h,k] = post[t,h]*x[t,k] + sum_i comb[t,i,h]*residual[t,i,k]
// ---------------------------------------------------------------------------
at::Tensor hc_post_fused_cpu(at::Tensor& x, at::Tensor& residual, at::Tensor& post, at::Tensor& comb) {
  RECORD_FUNCTION("sgl-kernel::hc_post_fused_cpu", {});
  TORCH_CHECK(
      x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat,
      "hc_post_fused_cpu: x must be bf16 or float32");
  TORCH_CHECK(residual.scalar_type() == x.scalar_type(), "hc_post_fused_cpu: residual must have same dtype as x");
  TORCH_CHECK(post.scalar_type() == at::kFloat, "hc_post_fused_cpu: post must be float32");
  TORCH_CHECK(comb.scalar_type() == at::kFloat, "hc_post_fused_cpu: comb must be float32");

  const int64_t T = x.size(0);
  const int64_t d = x.size(1);
  const int64_t hc = residual.size(1);
  TORCH_CHECK(hc == 4, "hc_post_fused_cpu: only hc_mult=4 is supported");

  // Accept non-contiguous inputs and use contiguous copies/views internally.
  const at::Tensor x_c = x.is_contiguous() ? x : x.contiguous();
  const at::Tensor residual_c = residual.is_contiguous() ? residual : residual.contiguous();
  const at::Tensor post_c = post.is_contiguous() ? post : post.contiguous();
  const at::Tensor comb_c = comb.is_contiguous() ? comb : comb.contiguous();

  auto out = at::empty({T, hc, d}, x.options());  // same dtype as x

  // Small T*HC: use K-split path to expose T*HC*K_SPLITS parallel tasks.
  // Large T*HC: the existing T*HC parallelism is sufficient.
  constexpr int64_t kSmallTaskThreshold = 8;
  const bool use_splitk = (T * hc <= kSmallTaskThreshold);

  if (x_c.scalar_type() == at::kBFloat16) {
    if (use_splitk)
      hc_post_splitk_impl<c10::BFloat16, 4>(
          out.data_ptr<c10::BFloat16>(),
          x_c.data_ptr<c10::BFloat16>(),
          residual_c.data_ptr<c10::BFloat16>(),
          post_c.data_ptr<float>(),
          comb_c.data_ptr<float>(),
          T,
          d);
    else
      hc_post_impl<c10::BFloat16, 4>(
          out.data_ptr<c10::BFloat16>(),
          x_c.data_ptr<c10::BFloat16>(),
          residual_c.data_ptr<c10::BFloat16>(),
          post_c.data_ptr<float>(),
          comb_c.data_ptr<float>(),
          T,
          d);
  } else if (x_c.scalar_type() == at::kFloat) {
    if (use_splitk)
      hc_post_splitk_impl<float, 4>(
          out.data_ptr<float>(),
          x_c.data_ptr<float>(),
          residual_c.data_ptr<float>(),
          post_c.data_ptr<float>(),
          comb_c.data_ptr<float>(),
          T,
          d);
    else
      hc_post_impl<float, 4>(
          out.data_ptr<float>(),
          x_c.data_ptr<float>(),
          residual_c.data_ptr<float>(),
          post_c.data_ptr<float>(),
          comb_c.data_ptr<float>(),
          T,
          d);
  } else {
    TORCH_CHECK(false, "hc_post_fused_cpu: unexpected scalar_type");
  }

  return out;
}

// ---------------------------------------------------------------------------
// hc_head_fused_cpu
// Inputs:
//   x         [T, hc_mult, d]     bf16 or float32
//   hc_fn     [hc_mult, hc_mult*d] float32  (projection weight)
//   hc_scale  scalar or [1]       float32
//   hc_base   [hc_mult]           float32
// Output: y [T, d]  same dtype as x
//
// Algorithm:
//   Pass A: scaled_x [T, hc_d] = x_flat * rsqrt(mean(x_flat^2)+norm_eps)
//   GEMM:   mixes    [T, hc_mult] = scaled_x @ hc_fn.T
//   Pass B: pre = sigmoid(mixes * hc_scale + hc_base) + hc_eps
//           y  = sum_h pre[:,h] * x[:,h,:]   (SIMD over d)
// ---------------------------------------------------------------------------
at::Tensor hc_head_fused_cpu(
    at::Tensor& x, at::Tensor& hc_fn, at::Tensor& hc_scale, at::Tensor& hc_base, double hc_eps, double norm_eps) {
  RECORD_FUNCTION("sgl-kernel::hc_head_fused_cpu", {});
  const int64_t hc_mult = x.size(1);
  TORCH_CHECK(hc_mult == 4, "hc_head_fused_cpu: only hc_mult=4 is supported");
  TORCH_CHECK(
      x.dim() == 3 && x.is_contiguous() && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat),
      "hc_head_fused_cpu: x must be contiguous bf16 or float32 [T, hc, d]");
  TORCH_CHECK(hc_fn.is_contiguous() && hc_fn.scalar_type() == at::kFloat);
  TORCH_CHECK(hc_base.is_contiguous() && hc_base.scalar_type() == at::kFloat);
  TORCH_CHECK(hc_base.numel() == hc_mult);
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat);

  const float hc_scale_val = hc_scale.item<float>();
  const int64_t T = x.size(0);
  const int64_t d = x.size(2);
  const int64_t hc_d = hc_mult * d;

  // Keep small-T fallback (splitk rms + at::mm + combine): for decode-like
  // workloads the existing path has good task granularity and low overhead.
  // Use fused tiny-gemm path for larger T to avoid scaled_x materialization.
  constexpr int64_t kSmallTokenThreshold = 8;
  const bool use_fused_head = (T > kSmallTokenThreshold);

  auto y = at::empty({T, d}, x.options());  // same dtype as x
  if (use_fused_head) {
    // hc_fn is [HC, HC*d] row-major (contiguous, checked above) – pass directly,
    // no transposition needed by the fused kernel.
    if (x.scalar_type() == at::kBFloat16) {
      hc_head_brgemm_fuse_impl<c10::BFloat16, 4>(
          y.data_ptr<c10::BFloat16>(),
          x.data_ptr<c10::BFloat16>(),
          hc_fn.data_ptr<float>(),
          hc_scale_val,
          hc_base.data_ptr<float>(),
          T,
          d,
          static_cast<float>(hc_eps),
          static_cast<float>(norm_eps));
    } else if (x.scalar_type() == at::kFloat) {
      hc_head_brgemm_fuse_impl<float, 4>(
          y.data_ptr<float>(),
          x.data_ptr<float>(),
          hc_fn.data_ptr<float>(),
          hc_scale_val,
          hc_base.data_ptr<float>(),
          T,
          d,
          static_cast<float>(hc_eps),
          static_cast<float>(norm_eps));
    } else {
      TORCH_CHECK(false, "hc_head_fused_cpu: unexpected scalar_type");
    }
    return y;
  }

  auto f32_opts = at::TensorOptions().dtype(at::kFloat).device(x.device());
  auto scaled_x = at::empty({T, hc_d}, f32_opts);
  if (x.scalar_type() == at::kBFloat16) {
    hc_pre_scale_splitk_impl<c10::BFloat16>(
        scaled_x.data_ptr<float>(), x.data_ptr<c10::BFloat16>(), T, hc_d, static_cast<float>(norm_eps));
  } else if (x.scalar_type() == at::kFloat) {
    hc_pre_scale_splitk_impl<float>(
        scaled_x.data_ptr<float>(), x.data_ptr<float>(), T, hc_d, static_cast<float>(norm_eps));
  } else {
    TORCH_CHECK(false, "hc_head_fused_cpu: unexpected scalar_type");
  }

  auto mixes = at::mm(scaled_x, hc_fn.t());
  if (x.scalar_type() == at::kBFloat16) {
    hc_head_combine_impl<c10::BFloat16, 4>(
        y.data_ptr<c10::BFloat16>(),
        mixes.contiguous().data_ptr<float>(),
        x.data_ptr<c10::BFloat16>(),
        hc_scale_val,
        hc_base.data_ptr<float>(),
        T,
        d,
        static_cast<float>(hc_eps));
  } else {
    hc_head_combine_impl<float, 4>(
        y.data_ptr<float>(),
        mixes.contiguous().data_ptr<float>(),
        x.data_ptr<float>(),
        hc_scale_val,
        hc_base.data_ptr<float>(),
        T,
        d,
        static_cast<float>(hc_eps));
  }
  return y;
}
