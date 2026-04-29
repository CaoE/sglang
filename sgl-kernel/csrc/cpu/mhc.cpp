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
//
// Uses 4-way unrolled sq accumulation for ILP (4 independent FMA chains).
// ---------------------------------------------------------------------------
template <typename scalar_t>
static void hc_pre_scale_impl(
    float* __restrict__ scaled_x,    // [T, hc_d]  float32 output
    const scalar_t* __restrict__ x,  // [T, hc_d]  input (bf16 or float32)
    int64_t T,
    int64_t hc_d,
    float rms_eps) {
  static_assert(std::is_same_v<scalar_t, c10::BFloat16>, "hc_pre_scale_impl: only bf16 is supported");
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();   // 32 bf16
  constexpr int64_t kFVecSize = fVec::size();  // 16 fp32
  // 4-way unroll: each bVec produces 2 fVec, so 4 bVec = 8 fVec = 4 sq chains
  constexpr int64_t KU = 4;
  constexpr int64_t STEP = KU * kVecSize;  // 128 bf16 elements per main-loop iter

  at::parallel_for(0, T, 0, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      const scalar_t* x_t = x + t * hc_d;
      float* sx_t = scaled_x + t * hc_d;

      // Pass 1: sq with 4 independent accumulator chains
      fVec sq0(0.f), sq1(0.f), sq2(0.f), sq3(0.f);
      fVec x0, x1;
      int64_t k = 0;
      for (; k <= hc_d - STEP; k += STEP) {
        std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
        sq0 += x0 * x0 + x1 * x1;
        std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k + kVecSize));
        sq1 += x0 * x0 + x1 * x1;
        std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k + 2 * kVecSize));
        sq2 += x0 * x0 + x1 * x1;
        std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k + 3 * kVecSize));
        sq3 += x0 * x0 + x1 * x1;
      }
      fVec sq_acc = (sq0 + sq1) + (sq2 + sq3);
      for (; k <= hc_d - kVecSize; k += kVecSize) {
        std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
        sq_acc += x0 * x0 + x1 * x1;
      }
      if (k < hc_d) {
        std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k, hc_d - k));
        sq_acc += x0 * x0 + x1 * x1;
      }

      const double sum_sq = static_cast<double>(vec_reduce_sum(sq_acc));
      const float rsqrt =
          static_cast<float>(1.0 / std::sqrt(sum_sq / static_cast<double>(hc_d) + static_cast<double>(rms_eps)));
      const fVec rsqrt_fvec(rsqrt);

      // Pass 2: scale and store to float32 output
      for (k = 0; k <= hc_d - kVecSize; k += kVecSize) {
        std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k));
        (x0 * rsqrt_fvec).store(sx_t + k);
        (x1 * rsqrt_fvec).store(sx_t + k + kFVecSize);
      }
      if (k < hc_d) {
        const int64_t rem = hc_d - k;
        const int64_t rem0 = std::min(rem, (int64_t)kFVecSize);
        const int64_t rem1 = rem - rem0;
        std::tie(x0, x1) = at::vec::convert_to_float(bVec::loadu(x_t + k, rem));
        (x0 * rsqrt_fvec).store(sx_t + k, rem0);
        if (rem1 > 0) (x1 * rsqrt_fvec).store(sx_t + k + kFVecSize, rem1);
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
  static_assert(std::is_same_v<scalar_t, c10::BFloat16>, "hc_pre_scale_splitk_impl: only bf16 is supported");
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
      // bf16: convert and accumulate
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
      if (k < d) {
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
      // bf16: convert and accumulate
      for (k = 0; k <= d - kVecSize; k += kVecSize) {
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
      if (k < d) {
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

      // Pass 1: rms + dot-products over flattened [HC*d], bf16 input.
      int64_t k = 0;
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

      const double sum_sq = static_cast<double>(vec_reduce_sum(sq_acc));
      const float inv_rms =
          static_cast<float>(1.0 / std::sqrt(sum_sq / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));

      float pre[HC];
      for (int h = 0; h < HC; ++h) {
        const float mix_h = static_cast<float>(vec_reduce_sum(dot_acc[h])) * inv_rms;
        const float gate_in = mix_h * hc_scale_val + hc_base[h];
        pre[h] = 1.f / (1.f + std::exp(-gate_in)) + hc_eps;
      }

      // Pass 2: h-outer combine, bf16 path: accumulate into fp32 scratch, then convert.
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

      // Convert fp32 accumulator → bf16 output
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
  });
}

// ---------------------------------------------------------------------------
// hc_head_splitk_fuse_impl<scalar_t, HC>
// Small-T fused path that saturates ~40 threads even for T=1.
//
// Problem: for T=1 the large-T path (grain=1 over T) only spawns 1 task,
// leaving 39 threads idle.  This version splits the K=HC*d dimension so
// that T × n_k_blocks tasks fill all threads in Phase 1.
//
// Three-phase structure (two parallel_for + one barrier between them):
//
//   Phase 1 – parallel over T × n_k_blocks (grain=1):
//     Each task processes a [k0, k1) slice of the flattened input.
//     Produces partials: partial_sq[t,kb] and partial_dot[t,kb,HC].
//
//   Phase 2 – parallel over T (grain=1):
//     Reduce partials → inv_rms, apply sigmoid gate → pre[t, HC].
//
//   Phase 3 – parallel over T × n_d_blocks (grain=1):
//     h-outer combine: y[t, k0:k1] = Σ_h pre[t,h] * x[t,h, k0:k1].
//     d is split into the same block size as K so both phases saturate threads.
//
// Memory layout of partials:
//   partials[(t * n_k_blocks + kb) * (1 + HC)]
//     [0]       = partial sq_acc (float)
//     [1..HC]   = partial dot_acc[h] (float, h=0..HC-1)
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_head_splitk_fuse_impl(
    scalar_t* __restrict__ y,         // [T, d]
    const scalar_t* __restrict__ x,   // [T, HC, d]
    const float* __restrict__ hc_fn,  // [HC, HC*d]  row-major
    float hc_scale_val,
    const float* __restrict__ hc_base,  // [HC]
    int64_t T,
    int64_t d,
    float hc_eps,
    float norm_eps) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();
  constexpr int64_t kFVecSize = fVec::size();  // 16 floats (AVX512)
  const int64_t hc_d = HC * d;

  // Block size: 512 floats = 2KB per K-block.  Each Phase-1 task reads the
  // block from x (2KB) plus HC weight slices (HC*2KB=8KB) — fits in L1 cache.
  constexpr int64_t K_BLOCK = 512;
  const int64_t n_k_blocks = div_up(hc_d, K_BLOCK);
  const int64_t n_d_blocks = div_up(d, K_BLOCK);

  // partials[t, kb, 1+HC]  (sq + HC dot products)
  constexpr int64_t partial_stride = 1 + HC;
  std::vector<float> partials(static_cast<size_t>(T * n_k_blocks * partial_stride), 0.f);
  std::vector<float> pre_buf(static_cast<size_t>(T * HC));

  // ── Phase 1: T × n_k_blocks tasks ────────────────────────────────────────
  at::parallel_for(0, T * n_k_blocks, 1, [&](int64_t begin, int64_t end) {
    // Per-thread upcast scratch: 512 f32 = 2 KB — stays in L1 across all heads.
    float a_float[K_BLOCK];
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t t = idx / n_k_blocks;
      const int64_t kb = idx % n_k_blocks;
      const int64_t k0 = kb * K_BLOCK;
      const int64_t len = std::min(K_BLOCK, hc_d - k0);

      const scalar_t* a = x + t * hc_d + k0;
      float* p = partials.data() + (t * n_k_blocks + kb) * partial_stride;

      // Step 1: upcast bf16 → float once into a_float (contiguous write).
      {
        int64_t k = 0;
        for (; k <= len - kVecSize; k += kVecSize) {
          fVec xf0, xf1;
          std::tie(xf0, xf1) = at::vec::convert_to_float(bVec::loadu(a + k));
          xf0.store(a_float + k);
          xf1.store(a_float + k + kFVecSize);
        }
        if (k < len) {
          const int64_t rem = len - k;
          const int64_t rem0 = std::min(rem, kFVecSize);
          const int64_t rem1 = rem - rem0;
          fVec xf0, xf1;
          std::tie(xf0, xf1) = at::vec::convert_to_float(bVec::loadu(a + k, rem));
          xf0.store(a_float + k, rem0);
          if (rem1 > 0) xf1.store(a_float + k + kFVecSize, rem1);
        }
      }

      // Step 2: sq — KU=4 unrolled float loop on a_float.
      {
        constexpr int64_t STEP = 4 * kFVecSize;
        fVec sq_u[4] = {fVec(0.f), fVec(0.f), fVec(0.f), fVec(0.f)};
        int64_t k = 0;
        for (; k <= len - STEP; k += STEP) {
          const fVec x0 = fVec::loadu(a_float + k);
          const fVec x1 = fVec::loadu(a_float + k + kFVecSize);
          const fVec x2 = fVec::loadu(a_float + k + 2 * kFVecSize);
          const fVec x3 = fVec::loadu(a_float + k + 3 * kFVecSize);
          sq_u[0] += x0 * x0;
          sq_u[1] += x1 * x1;
          sq_u[2] += x2 * x2;
          sq_u[3] += x3 * x3;
        }
        fVec sq_acc = (sq_u[0] + sq_u[1]) + (sq_u[2] + sq_u[3]);
        for (; k <= len - kFVecSize; k += kFVecSize) {
          const fVec xf = fVec::loadu(a_float + k);
          sq_acc += xf * xf;
        }
        if (k < len) {
          const fVec xf = fVec::loadu(a_float + k, len - k);
          sq_acc += xf * xf;
        }
        p[0] = vec_reduce_sum(sq_acc);
      }

      // Step 3: dot — h-outer KU=4 float loop; each head streams its weight
      // row contiguously (hc_fn[h][k0..k0+len]), no inter-head strides.
      for (int h = 0; h < HC; ++h) {
        const float* w = hc_fn + h * hc_d + k0;
        constexpr int64_t STEP = 4 * kFVecSize;
        fVec dot_u[4] = {fVec(0.f), fVec(0.f), fVec(0.f), fVec(0.f)};
        int64_t k = 0;
        for (; k <= len - STEP; k += STEP) {
          dot_u[0] += fVec::loadu(a_float + k) * fVec::loadu(w + k);
          dot_u[1] += fVec::loadu(a_float + k + kFVecSize) * fVec::loadu(w + k + kFVecSize);
          dot_u[2] += fVec::loadu(a_float + k + 2 * kFVecSize) * fVec::loadu(w + k + 2 * kFVecSize);
          dot_u[3] += fVec::loadu(a_float + k + 3 * kFVecSize) * fVec::loadu(w + k + 3 * kFVecSize);
        }
        fVec dot_acc = (dot_u[0] + dot_u[1]) + (dot_u[2] + dot_u[3]);
        for (; k <= len - kFVecSize; k += kFVecSize) {
          dot_acc += fVec::loadu(a_float + k) * fVec::loadu(w + k);
        }
        if (k < len) {
          dot_acc += fVec::loadu(a_float + k, len - k) * fVec::loadu(w + k, len - k);
        }
        p[1 + h] = vec_reduce_sum(dot_acc);
      }
    }
  });

  // ── Phase 2: reduce + gate (serial) ──────────────────────────────────────
  // Work = T × n_k_blocks × (1 + HC) floats ≈ 32 × 32 × 5 = 5120 ops at most.
  // Parallelising this would cost more in thread-scheduling overhead than it
  // saves, so we simply run it in the caller thread.
  for (int64_t t = 0; t < T; ++t) {
    double sq_total = 0.0;
    float dot_total[HC] = {};
    for (int64_t kb = 0; kb < n_k_blocks; ++kb) {
      const float* p = partials.data() + (t * n_k_blocks + kb) * partial_stride;
      sq_total += static_cast<double>(p[0]);
      for (int h = 0; h < HC; ++h)
        dot_total[h] += p[1 + h];
    }
    const float inv_rms =
        static_cast<float>(1.0 / std::sqrt(sq_total / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));
    float* pre = pre_buf.data() + t * HC;
    for (int h = 0; h < HC; ++h) {
      const float gate_in = dot_total[h] * inv_rms * hc_scale_val + hc_base[h];
      pre[h] = 1.f / (1.f + std::exp(-gate_in)) + hc_eps;
    }
  }

  // ── Phase 3: h-outer combine, T × n_d_blocks tasks ───────────────────────
  at::parallel_for(0, T * n_d_blocks, 1, [&](int64_t begin, int64_t end) {
    float sc[K_BLOCK];  // per-task fp32 scratch for bf16 output path
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t t = idx / n_d_blocks;
      const int64_t db = idx % n_d_blocks;
      const int64_t k0 = db * K_BLOCK;
      const int64_t len = std::min(K_BLOCK, d - k0);

      const float* pre = pre_buf.data() + t * HC;
      const scalar_t* x_t = x + t * hc_d;
      scalar_t* y_t = y + t * d;

      // bf16: accumulate into fp32 scratch, then convert to output.
      std::memset(sc, 0, len * sizeof(float));
      for (int h = 0; h < HC; ++h) {
        const fVec pv(pre[h]);
        const scalar_t* xh = x_t + (int64_t)h * d + k0;
        int64_t k = 0;
        for (; k <= len - kVecSize; k += kVecSize) {
          fVec xf0, xf1;
          std::tie(xf0, xf1) = at::vec::convert_to_float(bVec::loadu(xh + k));
          (fVec::loadu(sc + k) + pv * xf0).store(sc + k);
          (fVec::loadu(sc + k + kFVecSize) + pv * xf1).store(sc + k + kFVecSize);
        }
        if (k < len) {
          const int64_t rem = len - k;
          const int64_t rem0 = std::min(rem, kFVecSize);
          const int64_t rem1 = rem - rem0;
          fVec xf0, xf1;
          std::tie(xf0, xf1) = at::vec::convert_to_float(bVec::loadu(xh + k, rem));
          (fVec::loadu(sc + k, rem0) + pv * xf0).store(sc + k, rem0);
          if (rem1 > 0) (fVec::loadu(sc + k + kFVecSize, rem1) + pv * xf1).store(sc + k + kFVecSize, rem1);
        }
      }
      scalar_t* yt = y_t + k0;
      int64_t kk = 0;
      for (; kk <= len - kVecSize; kk += kVecSize) {
        at::vec::convert_from_float<scalar_t>(fVec::loadu(sc + kk), fVec::loadu(sc + kk + kFVecSize)).store(yt + kk);
      }
      if (kk < len) {
        const int64_t rem = len - kk;
        const int64_t rem0 = std::min(rem, kFVecSize);
        const int64_t rem1 = rem - rem0;
        at::vec::convert_from_float<scalar_t>(
            fVec::loadu(sc + kk, rem0), rem1 > 0 ? fVec::loadu(sc + kk + kFVecSize, rem1) : fVec(0.f))
            .store(yt + kk, rem);
      }
    }
  });
}

// ---------------------------------------------------------------------------
// hc_head_fuse_impl<scalar_t, HC>
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
static void hc_head_fuse_impl(
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
  constexpr int64_t kFVecSize = fVec::size();
  const int64_t hc_d = HC * d;

  // ── Parallelism: grain=1 gives T independent tasks (e.g. 128 for T=128),
  // saturating ~40 threads with ~3 tokens/thread.  No K-split is needed
  // because T >> #threads; K-split only helps for T < #threads (decode), and
  // that case already falls back to the splitk path (kSmallTokenThreshold=8).
  //
  // N=HC=4 is below tinygemm_kernel's minimum block width (16).  Instead we
  // fuse the RMSnorm sq-sum with the HC dot-products in ONE pass over x[t,:].
  // k-tiled Phase 1: upcast K_BLOCK bf16→float, consumed by all HC heads.
  // d-tiled Phase 2: accumulate K_BLOCK floats from HC heads, fused convert.
  // Both phases run sequentially and share ONE tile buffer per thread.
  // Scratch per thread: K_BLOCK floats = 2KB << 32KB L1.
  constexpr int64_t K_BLOCK = 512;  // must be >= STEP_DOT (=KU_DOT*kFVecSize)
  const int64_t num_threads = at::get_num_threads();
  const int64_t scratch_stride = K_BLOCK;  // reused by both phases
  auto scratch_tensor = at::empty({num_threads * scratch_stride}, at::kFloat);
  float* const scratch_base = scratch_tensor.data_ptr<float>();

  // Constants and per-thread vec accumulators are kept outside the t-loop so
  // the arrays are allocated once per thread, not once per token.
  constexpr int64_t KU_DOT = 8;
  constexpr int64_t STEP_DOT = KU_DOT * kFVecSize;

  at::parallel_for(0, T, 1, [&](int64_t begin, int64_t end) {
    const int64_t tid = at::get_thread_num();
    float* const a_tile_ptr = scratch_base + tid * scratch_stride;
    float* const sc_ptr = a_tile_ptr;  // reused: Phase 1 & 2 are sequential

    // Reusable fVec temporaries — declared once, reused across all loops.
    fVec v0, v1;
    fVec sq_u[KU_DOT];
    fVec sq_acc;
    fVec dot_acc[HC];
    fVec dot_u[KU_DOT];
    fVec pre_fvec;

    for (int64_t t = begin; t < end; ++t) {
      const scalar_t* x_t = x + t * hc_d;
      scalar_t* y_t = y + t * d;

      // ── k-tiled: upcast + sq+dot fused per tile ────────────────────────
      // Each tile upcasts K_BLOCK bf16→float into a_tile_ptr (2KB), then
      // all HC heads consume that same 2KB before advancing k.  No separate
      // full-width a_buf write needed; scratch is K_BLOCK+d instead of hc_d+d.
#pragma GCC unroll 4
      for (int h = 0; h < HC; ++h)
        dot_acc[h] = fVec(0.f);
      sq_acc = fVec(0.f);

      for (int64_t k0 = 0; k0 < hc_d; k0 += K_BLOCK) {
        const int64_t len = std::min(K_BLOCK, hc_d - k0);
        const scalar_t* x_k0 = x_t + k0;

        // Step 1 (tile): upcast bf16 → float into a_tile_ptr
        {
          int64_t k = 0;
          for (; k <= len - kVecSize; k += kVecSize) {
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_k0 + k));
            v0.store(a_tile_ptr + k);
            v1.store(a_tile_ptr + k + kFVecSize);
          }
          if (k < len) {
            const int64_t rem = len - k;
            const int64_t rem0 = std::min(rem, kFVecSize);
            const int64_t rem1 = rem - rem0;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_k0 + k, rem));
            v0.store(a_tile_ptr + k, rem0);
            if (rem1 > 0) v1.store(a_tile_ptr + k + kFVecSize, rem1);
          }
        }
        const float* a_tile = a_tile_ptr;

        // h=0: sq and dot fused for this tile ─────────────────────────────
        {
          const float* w0 = hc_fn + k0;
#pragma GCC unroll 8
          for (int u = 0; u < KU_DOT; ++u) {
            dot_u[u] = fVec(0.f);
            sq_u[u] = fVec(0.f);
          }
          int64_t kk = 0;
          for (; kk <= len - STEP_DOT; kk += STEP_DOT) {
#pragma GCC unroll 8
            for (int u = 0; u < KU_DOT; ++u) {
              v0 = fVec::loadu(a_tile + kk + u * kFVecSize);
              dot_u[u] += v0 * fVec::loadu(w0 + kk + u * kFVecSize);
              sq_u[u] += v0 * v0;
            }
          }
          fVec dacc = ((dot_u[0] + dot_u[1]) + (dot_u[2] + dot_u[3])) + ((dot_u[4] + dot_u[5]) + (dot_u[6] + dot_u[7]));
          sq_acc += ((sq_u[0] + sq_u[1]) + (sq_u[2] + sq_u[3])) + ((sq_u[4] + sq_u[5]) + (sq_u[6] + sq_u[7]));
          // dot+sq tile-tail together
          for (; kk <= len - 2 * kFVecSize; kk += 2 * kFVecSize) {
            v0 = fVec::loadu(a_tile + kk);
            v1 = fVec::loadu(a_tile + kk + kFVecSize);
            dacc += v0 * fVec::loadu(w0 + kk) + v1 * fVec::loadu(w0 + kk + kFVecSize);
            sq_acc += v0 * v0 + v1 * v1;
          }
          for (; kk <= len - kFVecSize; kk += kFVecSize) {
            v0 = fVec::loadu(a_tile + kk);
            dacc += v0 * fVec::loadu(w0 + kk);
            sq_acc += v0 * v0;
          }
          if (kk < len) {
            v0 = fVec::loadu(a_tile + kk, len - kk);
            dacc += v0 * fVec::loadu(w0 + kk, len - kk);
            sq_acc += v0 * v0;
          }
          dot_acc[0] += dacc;
        }

        // h=1..HC-1: dot only for this tile ───────────────────────────────
        for (int h = 1; h < HC; ++h) {
          const float* w = hc_fn + h * hc_d + k0;
#pragma GCC unroll 8
          for (int u = 0; u < KU_DOT; ++u)
            dot_u[u] = fVec(0.f);
          int64_t kk = 0;
          for (; kk <= len - STEP_DOT; kk += STEP_DOT) {
#pragma GCC unroll 8
            for (int u = 0; u < KU_DOT; ++u)
              dot_u[u] += fVec::loadu(a_tile + kk + u * kFVecSize) * fVec::loadu(w + kk + u * kFVecSize);
          }
          fVec acc = ((dot_u[0] + dot_u[1]) + (dot_u[2] + dot_u[3])) + ((dot_u[4] + dot_u[5]) + (dot_u[6] + dot_u[7]));
          for (; kk <= len - 2 * kFVecSize; kk += 2 * kFVecSize) {
            v0 = fVec::loadu(a_tile + kk);
            v1 = fVec::loadu(a_tile + kk + kFVecSize);
            acc += v0 * fVec::loadu(w + kk) + v1 * fVec::loadu(w + kk + kFVecSize);
          }
          for (; kk <= len - kFVecSize; kk += kFVecSize) {
            v0 = fVec::loadu(a_tile + kk);
            acc += v0 * fVec::loadu(w + kk);
          }
          if (kk < len) {
            v0 = fVec::loadu(a_tile + kk, len - kk);
            acc += v0 * fVec::loadu(w + kk, len - kk);
          }
          dot_acc[h] += acc;
        }
      }  // k-tile loop

      const float inv_rms = static_cast<float>(
          1.0 /
          std::sqrt(
              static_cast<double>(vec_reduce_sum(sq_acc)) / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));

      float pre[HC];
      for (int h = 0; h < HC; ++h) {
        const float gate_in = vec_reduce_sum(dot_acc[h]) * inv_rms * hc_scale_val + hc_base[h];
        pre[h] = 1.f / (1.f + std::exp(-gate_in)) + hc_eps;
      }

      // ── Step 2: d-tiled combine + fused bf16 convert ──────────────────
      // Each K_BLOCK-tile: h=0 direct-assigns (eliminates memset), h=1..HC-1
      // accumulates, then sc_ptr converts to bf16 y_t immediately while hot.
      // sc_ptr reuses a_tile_ptr (sequential with Phase 1, same 2KB buffer).
      for (int64_t j0 = 0; j0 < d; j0 += K_BLOCK) {
        const int64_t jlen = std::min(K_BLOCK, d - j0);

        // h=0: direct assign — no memset required
        {
          pre_fvec = fVec(pre[0]);
          const scalar_t* x_th = x_t + j0;  // h=0 segment starts at x_t
          int64_t kk = 0;
          for (; kk <= jlen - kVecSize; kk += kVecSize) {
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk));
            (pre_fvec * v0).store(sc_ptr + kk);
            (pre_fvec * v1).store(sc_ptr + kk + kFVecSize);
          }
          if (kk < jlen) {
            const int64_t rem = jlen - kk;
            const int64_t rem0 = std::min(rem, kFVecSize);
            const int64_t rem1 = rem - rem0;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk, rem));
            (pre_fvec * v0).store(sc_ptr + kk, rem0);
            if (rem1 > 0) (pre_fvec * v1).store(sc_ptr + kk + kFVecSize, rem1);
          }
        }

        // h=1..HC-1: accumulate into sc_ptr
        for (int h = 1; h < HC; ++h) {
          pre_fvec = fVec(pre[h]);
          const scalar_t* x_th = x_t + (int64_t)h * d + j0;
          int64_t kk = 0;
          for (; kk <= jlen - kVecSize; kk += kVecSize) {
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk));
            (fVec::loadu(sc_ptr + kk) + pre_fvec * v0).store(sc_ptr + kk);
            (fVec::loadu(sc_ptr + kk + kFVecSize) + pre_fvec * v1).store(sc_ptr + kk + kFVecSize);
          }
          if (kk < jlen) {
            const int64_t rem = jlen - kk;
            const int64_t rem0 = std::min(rem, kFVecSize);
            const int64_t rem1 = rem - rem0;
            std::tie(v0, v1) = at::vec::convert_to_float(bVec::loadu(x_th + kk, rem));
            (fVec::loadu(sc_ptr + kk, rem0) + pre_fvec * v0).store(sc_ptr + kk, rem0);
            if (rem1 > 0)
              (fVec::loadu(sc_ptr + kk + kFVecSize, rem1) + pre_fvec * v1).store(sc_ptr + kk + kFVecSize, rem1);
          }
        }

        // fused convert: sc_ptr (hot in L1) → bf16 y_t[j0..j0+jlen)
        int64_t kk = 0;
        for (; kk <= jlen - kVecSize; kk += kVecSize) {
          at::vec::convert_from_float<scalar_t>(fVec::loadu(sc_ptr + kk), fVec::loadu(sc_ptr + kk + kFVecSize))
              .store(y_t + j0 + kk);
        }
        if (kk < jlen) {
          const int64_t rem = jlen - kk;
          const int64_t rem0 = std::min(rem, kFVecSize);
          const int64_t rem1 = rem - rem0;
          at::vec::convert_from_float<scalar_t>(
              fVec::loadu(sc_ptr + kk, rem0), rem1 > 0 ? fVec::loadu(sc_ptr + kk + kFVecSize, rem1) : fVec(0.f))
              .store(y_t + j0 + kk, rem);
        }
      }  // d-tile loop
    }  // for t
  });  // parallel_for
}

// ---------------------------------------------------------------------------
// hc_head_combine_impl<scalar_t, HC>
// Pass B for hc_head: mixes[T, HC] + x[T, HC, d] → y[T, d]
//   pre[t,h] = sigmoid(mixes[t,h] * hc_scale + hc_base[h]) + hc_eps
//   y[t,k]   = sum_h pre[t,h] * x[t,h,k]
//
// Optimizations:
//   1. d-tiled (D_BLOCK=512 → 2KB sc buffer stays in L1).
//   2. h=0 direct-assigns (no memset), h=1..HC-1 accumulates.
//   3. Fused convert: sc → bf16 y_t immediately after all heads, while hot.
//   4. Pre-allocated per-thread scratch (no per-task heap alloc).
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
  constexpr int64_t kVecSize = bVec::size();
  constexpr int64_t kFVecSize = fVec::size();
  constexpr int64_t D_BLOCK = 512;  // 2KB fp32 scratch, fits L1

  // Pre-allocate per-thread scratch outside parallel region
  const int64_t num_threads = at::get_num_threads();
  auto sc_tensor = at::empty({num_threads * D_BLOCK}, at::kFloat);
  float* const sc_base = sc_tensor.data_ptr<float>();

  at::parallel_for(0, T, 0, [&](int64_t begin, int64_t end) {
    const int64_t tid = at::get_thread_num();
    float* const sc = sc_base + tid * D_BLOCK;
    fVec v0, v1, pre_fvec;

    alignas(64) float pre[HC];
    for (int64_t t = begin; t < end; ++t) {
      const float* m = mixes + t * HC;
      for (int h = 0; h < HC; ++h) {
        float v = m[h] * hc_scale_val + hc_base[h];
        pre[h] = 1.0f / (1.0f + std::exp(-v)) + hc_eps;
      }

      const scalar_t* x_t = x + t * HC * d;
      scalar_t* y_t = y + t * d;

      for (int64_t j0 = 0; j0 < d; j0 += D_BLOCK) {
        const int64_t jlen = std::min(D_BLOCK, d - j0);

        // h=0: direct assign (no memset)
        {
          pre_fvec = fVec(pre[0]);
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

        // h=1..HC-1: accumulate into sc
        for (int h = 1; h < HC; ++h) {
          pre_fvec = fVec(pre[h]);
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

        // Fused convert: sc (hot in L1) → bf16 y_t[j0..j0+jlen)
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
      }  // d-tile
    }  // t
  });
}

// ---------------------------------------------------------------------------
// hc_head_fuse_2d_impl<scalar_t, HC>
// 2D-parallel fused path for T ≥ kSmallTokenThreshold.
//
// Borrows the CUDA TileLang strategy (mhc_pre_gemm_sqrsum_splitk):
//   parallel_2d(n_t_blocks, n_k_blocks) with loop_2d cache-friendly ordering
//   so that hc_fn weight tiles stay in L2 while many token-blocks reuse them.
//
// Three phases (two parallel regions + one small parallel_for):
//   Phase 1 – parallel_2d: upcast bf16→f32 + fused sq + HC dot-products
//             → partial_sq[tb,kb,tl] and partial_dot[tb,kb,tl,h]
//   Phase 2 – parallel_for(T): reduce partials, compute inv_rms + sigmoid gate
//             → pre[T, HC]
//   Phase 3 – parallel_for(T): d-tiled weighted combine + fused bf16 convert
//             → y[T, d]
// ---------------------------------------------------------------------------
template <typename scalar_t, int HC>
static void hc_head_fuse_2d_impl(
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

  constexpr int64_t T_BLOCK = 32;
  constexpr int64_t K_BLOCK = 512;  // HC*1024*4B = 16KB weight tile, fits L1 (32KB)
  constexpr int64_t D_BLOCK = 512;

  const int64_t n_t_blocks = div_up(T, T_BLOCK);
  const int64_t n_k_blocks = div_up(hc_d, K_BLOCK);

  // Partial buffer: [n_t_blocks, n_k_blocks, T_BLOCK, 1+HC]
  //   p[0] = partial sq,  p[1..HC] = partial dots
  constexpr int64_t P_STRIDE = 1 + HC;
  auto partials_tensor = at::zeros({n_t_blocks * n_k_blocks * T_BLOCK * P_STRIDE}, at::kFloat);
  float* const partials = partials_tensor.data_ptr<float>();

  // ── Phase 1: parallel_2d(n_t_blocks, n_k_blocks) ─────────────────────────
  // kb OUTER, tb INNER: weight tile [HC, K_BLOCK] stays in L1 for all tokens.
  // ~50 threads, T=512 d=7168: parallel_2d(16, 28) → ~4 t_blocks × 3 k_blocks
  // per thread → weight reuse = 4×32 = 128 tokens per 16KB weight tile.
  // Single-pass fused loop: one bf16→fp32 conversion per vector for sq + all
  // HC dot products (saves HC-1 redundant conversions vs separate passes).
  parallel_2d(
      static_cast<int>(n_t_blocks),
      static_cast<int>(n_k_blocks),
      [&](int64_t tb0, int64_t tb1, int64_t kb0, int64_t kb1) {
        // kb OUTER: weight tile stays in L1 across all t_blocks
        for (int64_t kb = kb0; kb < kb1; ++kb) {
          const int64_t k0 = kb * K_BLOCK;
          const int64_t klen = std::min(K_BLOCK, hc_d - k0);

          // tb INNER: all tokens share L1-resident weight
          for (int64_t tb = tb0; tb < tb1; ++tb) {
            const int64_t t0 = tb * T_BLOCK;
            const int64_t tlen = std::min(T_BLOCK, T - t0);
            float* p_base = partials + (tb * n_k_blocks + kb) * T_BLOCK * P_STRIDE;

            // ── Per-token fused loops: h outer, K inner ────────────────────────
            // Keep weight loads contiguous (per-head sequential scan), while
            // eliminating redundant bf16->fp32 conversions via x-vector cache.
            // Prefetch next token's x (stride = hc_d*2B, beyond HW prefetcher).
            const float* w_ptr[HC];
            for (int h = 0; h < HC; ++h)
              w_ptr[h] = hc_fn + h * hc_d + k0;

            constexpr int64_t MAX_K_VECS = K_BLOCK / kVecSize;
            fVec x_cache_lo[MAX_K_VECS], x_cache_hi[MAX_K_VECS];

            for (int64_t tl = 0; tl < tlen; ++tl) {
              const scalar_t* x_t = x + (t0 + tl) * hc_d + k0;
              const scalar_t* x_next = (tl + 1 < tlen) ? x + (t0 + tl + 1) * hc_d + k0 : nullptr;
              float* p = p_base + tl * P_STRIDE;

              const int64_t n_full = klen / kVecSize;
              const int64_t k_tail = n_full * kVecSize;
              const int64_t rem = klen - k_tail;
              const int64_t rem0 = std::min(rem, kFVecSize);
              const int64_t rem1 = rem - rem0;

              // h=0: convert x once, fuse sq + dot0, and cache converted x.
              {
                const float* w0 = w_ptr[0];
                fVec sq_lo(0.f), sq_hi(0.f);
                fVec d0(0.f), d1(0.f);
                for (int64_t ki = 0; ki < n_full; ++ki) {
                  const int64_t k = ki * kVecSize;
                  if (x_next) __builtin_prefetch(x_next + k, 0, 2);
                  __builtin_prefetch(w0 + k + 256, 0, 3);
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

                // h=1..HC-1: reuse cached x conversions, keep weight contiguous.
                for (int h = 1; h < HC; ++h) {
                  const float* w = w_ptr[h];
                  fVec hd0(0.f), hd1(0.f);
                  for (int64_t ki = 0; ki < n_full; ++ki) {
                    const int64_t k = ki * kVecSize;
                    __builtin_prefetch(w + k + 256, 0, 3);
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
          }  // tb
        }  // kb
      });  // parallel_2d

  // ── Phase 2: reduce partials + sigmoid gate → pre[T, HC] ─────────────────
  auto pre_tensor = at::empty({T * HC}, at::kFloat);
  float* const pre = pre_tensor.data_ptr<float>();

  // Work per token ≈ n_k_blocks*5 adds + 1 sqrt + 4 exp ≈ trivial.
  // Sequential loop is faster than paying OpenMP fork/join overhead.
  for (int64_t t = 0; t < T; ++t) {
    const int64_t tb = t / T_BLOCK;
    const int64_t tl = t % T_BLOCK;

    double sq_total = 0.0;
    float dot_total[HC] = {};
    for (int64_t kb = 0; kb < n_k_blocks; ++kb) {
      const float* p = partials + (tb * n_k_blocks + kb) * T_BLOCK * P_STRIDE + tl * P_STRIDE;
      sq_total += static_cast<double>(p[0]);
      for (int h = 0; h < HC; ++h)
        dot_total[h] += p[1 + h];
    }
    const float inv_rms =
        static_cast<float>(1.0 / std::sqrt(sq_total / static_cast<double>(hc_d) + static_cast<double>(norm_eps)));

    for (int h = 0; h < HC; ++h) {
      float gate = dot_total[h] * inv_rms * hc_scale_val + hc_base[h];
      pre[t * HC + h] = 1.f / (1.f + std::exp(-gate)) + hc_eps;
    }
  }

  // ── Phase 3: d-tiled weighted combine + fused bf16 convert ────────────────
  //   y[t, k] = sum_h pre[t,h] * x[t,h,k]
  //   D_BLOCK-tiled: h=0 direct-assign, h=1..HC-1 accumulate, then convert.
  const int64_t num_threads_p3 = at::get_num_threads();
  auto sc_tensor = at::empty({num_threads_p3 * D_BLOCK}, at::kFloat);
  float* const sc_base = sc_tensor.data_ptr<float>();

  at::parallel_for(0, T, 0, [&](int64_t begin, int64_t end) {
    const int64_t tid = at::get_thread_num();
    float* const sc = sc_base + tid * D_BLOCK;
    fVec v0, v1, pre_fvec;

    for (int64_t t = begin; t < end; ++t) {
      const float* pre_t = pre + t * HC;
      const scalar_t* x_t = x + t * hc_d;
      scalar_t* y_t = y + t * d;

      for (int64_t j0 = 0; j0 < d; j0 += D_BLOCK) {
        const int64_t jlen = std::min(D_BLOCK, d - j0);

        // h=0: direct assign (no memset)
        {
          pre_fvec = fVec(pre_t[0]);
          const scalar_t* x_th = x_t + j0;
          int64_t kk = 0;
          for (; kk <= jlen - kVecSize; kk += kVecSize) {
            __builtin_prefetch(x_t + d + j0 + kk, 0, 3);  // prefetch h=1's x
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
            if (h + 1 < HC) __builtin_prefetch(x_t + (int64_t)(h + 1) * d + j0 + kk, 0, 3);
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
      }  // d-tile
    }  // t
  });  // Phase 3
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
  TORCH_CHECK(x.dim() == 3 && x.scalar_type() == at::kBFloat16, "hc_pre_fused_cpu: x must be bf16 [T, hc, d]");
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
    hc_pre_scale_splitk_impl<c10::BFloat16>(
        scaled_x.data_ptr<float>(), x_c.data_ptr<c10::BFloat16>(), T, hc_d, static_cast<float>(rms_eps));
  } else {
    hc_pre_scale_impl<c10::BFloat16>(
        scaled_x.data_ptr<float>(), x_c.data_ptr<c10::BFloat16>(), T, hc_d, static_cast<float>(rms_eps));
  }

  // GEMM: mixes [T, mix_hc] = scaled_x @ hc_fn.T  (both paths use at::mm)
  mixes = at::mm(scaled_x, hc_fn_c.t());

  // Pass B: sinkhorn + weighted combine
  TORCH_CHECK(mixes.size(1) == mix_hc);

  auto post = at::empty({T, hc_mult}, f32_opts);           // always float32
  auto comb = at::empty({T, hc_mult, hc_mult}, f32_opts);  // always float32
  auto y = at::empty({T, d}, x.options());                 // same dtype as x

  TORCH_CHECK(x_c.scalar_type() == at::kBFloat16, "hc_pre_fused_cpu: x must be bf16");
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
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "hc_post_fused_cpu: x must be bf16");
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

  TORCH_CHECK(x_c.scalar_type() == at::kBFloat16, "hc_post_fused_cpu: x must be bf16");
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
      x.dim() == 3 && x.is_contiguous() && x.scalar_type() == at::kBFloat16,
      "hc_head_fused_cpu: x must be contiguous bf16 [T, hc, d]");
  TORCH_CHECK(hc_fn.is_contiguous() && hc_fn.scalar_type() == at::kFloat);
  TORCH_CHECK(hc_base.is_contiguous() && hc_base.scalar_type() == at::kFloat);
  TORCH_CHECK(hc_base.numel() == hc_mult);
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat);

  const float hc_scale_val = hc_scale.item<float>();
  const int64_t T = x.size(0);
  const int64_t d = x.size(2);
  const int64_t hc_d = hc_mult * d;

  auto y = at::empty({T, d}, x.options());

  // Small-T: fully fused split-K path (no intermediate tensors).
  // Large-T: RMSnorm → at::mm (MKL SGEMM with optimal weight reuse) → combine.
  constexpr int64_t kSmallTokenThreshold = 32;

  if (T < kSmallTokenThreshold) {
    hc_head_splitk_fuse_impl<c10::BFloat16, 4>(
        y.data_ptr<c10::BFloat16>(),
        x.data_ptr<c10::BFloat16>(),
        hc_fn.data_ptr<float>(),
        hc_scale_val,
        hc_base.data_ptr<float>(),
        T,
        d,
        static_cast<float>(hc_eps),
        static_cast<float>(norm_eps));
  } else {
    hc_head_fuse_2d_impl<c10::BFloat16, 4>(
        y.data_ptr<c10::BFloat16>(),
        x.data_ptr<c10::BFloat16>(),
        hc_fn.data_ptr<float>(),
        hc_scale_val,
        hc_base.data_ptr<float>(),
        T,
        d,
        static_cast<float>(hc_eps),
        static_cast<float>(norm_eps));
  }

  return y;
}

/*
mm

  hc_head (T=   1, D=4096, bf16)           | C++   0.103ms | PyTorch   0.135ms | Compile   0.147ms | Speedup  1.31x |
Throughput (C++/PyTorch): 159426.8/121543.3 elem/ms hc_head (T=   2, D=4096, bf16)           | C++   0.097ms | PyTorch
0.191ms | Compile   0.204ms | Speedup  1.97x | Throughput (C++/PyTorch): 337878.3/171405.7 elem/ms hc_head (T=   4,
D=4096, bf16)           | C++   0.117ms | PyTorch   0.312ms | Compile   0.209ms | Speedup  2.66x | Throughput
(C++/PyTorch): 558047.4/210114.0 elem/ms hc_head (T=   8, D=4096, bf16)           | C++   0.120ms | PyTorch   0.325ms |
Compile   0.324ms | Speedup  2.71x | Throughput (C++/PyTorch): 1090776.0/402905.1 elem/ms hc_head (T=  16, D=4096, bf16)
| C++   0.112ms | PyTorch   0.397ms | Compile   0.332ms | Speedup  3.54x | Throughput (C++/PyTorch): 2339205.6/660430.1
elem/ms hc_head (T=  32, D=4096, bf16)           | C++   0.209ms | PyTorch   0.386ms | Compile   0.420ms |
Speedup  1.85x | Throughput (C++/PyTorch): 2506464.4/1357131.5 elem/ms hc_head (T=  64, D=4096, bf16)           | C++
0.209ms | PyTorch   0.520ms | Compile   0.511ms | Speedup  2.49x | Throughput (C++/PyTorch): 5015335.6/2015883.3 elem/ms
  hc_head (T= 128, D=4096, bf16)           | C++   0.291ms | PyTorch   0.812ms | Compile   0.920ms | Speedup  2.79x |
Throughput (C++/PyTorch): 7217137.8/2583019.4 elem/ms hc_head (T= 256, D=4096, bf16)           | C++   0.537ms | PyTorch
1.049ms | Compile   1.722ms | Speedup  1.95x | Throughput (C++/PyTorch): 7812850.7/3998973.9 elem/ms hc_head (T= 512,
D=4096, bf16)           | C++   0.512ms | PyTorch   1.142ms | Compile   3.539ms | Speedup  2.23x | Throughput
(C++/PyTorch): 16397726.9/7343948.8 elem/ms hc_head (T=1024, D=4096, bf16)           | C++   1.147ms | PyTorch   2.653ms
| Compile   6.449ms | Speedup  2.31x | Throughput (C++/PyTorch): 14630667.5/6324271.8 elem/ms hc_head (T=2048, D=4096,
bf16)           | C++   2.437ms | PyTorch   5.934ms | Compile  13.550ms | Speedup  2.44x | Throughput (C++/PyTorch):
13770900.9/5654958.5 elem/ms hc_head (T=   1, D=7168, bf16)           | C++   0.106ms | PyTorch   0.208ms | Compile
0.170ms | Speedup  1.95x | Throughput (C++/PyTorch): 269530.6/137888.3 elem/ms hc_head (T=   2, D=7168, bf16) | C++
0.106ms | PyTorch   0.303ms | Compile   0.257ms | Speedup  2.87x | Throughput (C++/PyTorch): 542996.8/189145.7 elem/ms
  hc_head (T=   4, D=7168, bf16)           | C++   0.121ms | PyTorch   0.308ms | Compile   0.326ms | Speedup  2.54x |
Throughput (C++/PyTorch): 944794.0/372340.2 elem/ms hc_head (T=   8, D=7168, bf16)           | C++   0.099ms | PyTorch
0.337ms | Compile   0.422ms | Speedup  3.42x | Throughput (C++/PyTorch): 2322276.0/679674.0 elem/ms hc_head (T=  16,
D=7168, bf16)           | C++   0.103ms | PyTorch   0.349ms | Compile   0.370ms | Speedup  3.38x | Throughput
(C++/PyTorch): 4444724.9/1315466.9 elem/ms hc_head (T=  32, D=7168, bf16)           | C++   0.315ms | PyTorch   0.322ms
| Compile   0.503ms | Speedup  1.02x | Throughput (C++/PyTorch): 2916266.2/2845963.3 elem/ms hc_head (T=  64, D=7168,
bf16)           | C++   0.362ms | PyTorch   0.690ms | Compile   0.785ms | Speedup  1.91x | Throughput (C++/PyTorch):
5074414.3/2657882.0 elem/ms hc_head (T= 128, D=7168, bf16)           | C++   0.768ms | PyTorch   0.794ms |
Compile   1.464ms | Speedup  1.03x | Throughput (C++/PyTorch): 4780563.3/4622045.3 elem/ms hc_head (T= 256, D=7168,
bf16)           | C++   0.894ms | PyTorch   1.169ms | Compile   3.554ms | Speedup  1.31x | Throughput (C++/PyTorch):
8209548.6/6277644.4 elem/ms hc_head (T= 512, D=7168, bf16)           | C++   1.046ms | PyTorch   2.698ms |
Compile   5.784ms | Speedup  2.58x | Throughput (C++/PyTorch): 14034122.5/5440914.3 elem/ms hc_head (T=1024, D=7168,
bf16)           | C++   2.152ms | PyTorch   4.561ms | Compile  10.009ms | Speedup  2.12x | Throughput (C++/PyTorch):
13643980.5/6437388.8 elem/ms hc_head (T=2048, D=7168, bf16)           | C++   5.021ms | PyTorch  14.409ms |
Compile  19.840ms | Speedup  2.87x | Throughput (C++/PyTorch): 11693883.7/4075322.7 elem/ms




  ====================================================



  hc_head (T=   1, D=4096, bf16)           | C++   0.102ms | PyTorch   0.133ms | Compile   0.149ms | Speedup  1.30x |
Throughput (C++/PyTorch): 159905.5/123023.2 elem/ms hc_head (T=   2, D=4096, bf16)           | C++   0.095ms | PyTorch
0.188ms | Compile   0.201ms | Speedup  1.98x | Throughput (C++/PyTorch): 343783.0/173913.6 elem/ms hc_head (T=   4,
D=4096, bf16)           | C++   0.113ms | PyTorch   0.309ms | Compile   0.222ms | Speedup  2.73x | Throughput
(C++/PyTorch): 578414.8/212145.8 elem/ms hc_head (T=   8, D=4096, bf16)           | C++   0.104ms | PyTorch   0.301ms |
Compile   0.298ms | Speedup  2.90x | Throughput (C++/PyTorch): 1260178.0/435135.2 elem/ms hc_head (T=  16, D=4096, bf16)
| C++   0.104ms | PyTorch   0.390ms | Compile   0.329ms | Speedup  3.73x | Throughput (C++/PyTorch): 2509406.2/672404.8
elem/ms hc_head (T=  32, D=4096, bf16)           | C++   0.097ms | PyTorch   0.366ms | Compile   0.424ms |
Speedup  3.77x | Throughput (C++/PyTorch): 5397378.8/1430811.0 elem/ms hc_head (T=  64, D=4096, bf16)           | C++
0.120ms | PyTorch   0.510ms | Compile   0.487ms | Speedup  4.27x | Throughput (C++/PyTorch): 8765506.9/2054139.1 elem/ms
  hc_head (T= 128, D=4096, bf16)           | C++   0.158ms | PyTorch   0.797ms | Compile   0.947ms | Speedup  5.05x |
Throughput (C++/PyTorch): 13291896.2/2630667.4 elem/ms hc_head (T= 256, D=4096, bf16)           | C++   0.310ms |
PyTorch   1.098ms | Compile   1.740ms | Speedup  3.55x | Throughput (C++/PyTorch): 13545595.9/3820009.5 elem/ms hc_head
(T= 512, D=4096, bf16)           | C++   0.461ms | PyTorch   1.155ms | Compile   3.627ms | Speedup  2.50x | Throughput
(C++/PyTorch): 18195168.2/7265114.3 elem/ms hc_head (T=1024, D=4096, bf16)           | C++   0.836ms | PyTorch   2.409ms
| Compile   6.538ms | Speedup  2.88x | Throughput (C++/PyTorch): 20074772.8/6965367.8 elem/ms hc_head (T=2048, D=4096,
bf16)           | C++   1.426ms | PyTorch   5.968ms | Compile  10.736ms | Speedup  4.18x | Throughput (C++/PyTorch):
23524951.3/5622121.9 elem/ms hc_head (T=   1, D=7168, bf16)           | C++   0.104ms | PyTorch   0.209ms | Compile
0.163ms | Speedup  2.01x | Throughput (C++/PyTorch): 276596.5/137295.1 elem/ms hc_head (T=   2, D=7168, bf16) | C++
0.106ms | PyTorch   0.305ms | Compile   0.263ms | Speedup  2.87x | Throughput (C++/PyTorch): 539040.9/187814.8 elem/ms
  hc_head (T=   4, D=7168, bf16)           | C++   0.123ms | PyTorch   0.311ms | Compile   0.319ms | Speedup  2.53x |
Throughput (C++/PyTorch): 932931.8/369273.6 elem/ms hc_head (T=   8, D=7168, bf16)           | C++   0.097ms | PyTorch
0.325ms | Compile   0.420ms | Speedup  3.35x | Throughput (C++/PyTorch): 2365699.0/705394.9 elem/ms hc_head (T=  16,
D=7168, bf16)           | C++   0.095ms | PyTorch   0.367ms | Compile   0.474ms | Speedup  3.87x | Throughput
(C++/PyTorch): 4835436.8/1248384.9 elem/ms hc_head (T=  32, D=7168, bf16)           | C++   0.111ms | PyTorch   0.350ms
| Compile   0.464ms | Speedup  3.16x | Throughput (C++/PyTorch): 8271016.4/2618145.6 elem/ms hc_head (T=  64, D=7168,
bf16)           | C++   0.157ms | PyTorch   0.701ms | Compile   0.801ms | Speedup  4.47x | Throughput (C++/PyTorch):
11693949.9/2617120.5 elem/ms hc_head (T= 128, D=7168, bf16)           | C++   0.275ms | PyTorch   0.956ms |
Compile   1.448ms | Speedup  3.47x | Throughput (C++/PyTorch): 13326454.6/3840891.6 elem/ms hc_head (T= 256, D=7168,
bf16)           | C++   0.431ms | PyTorch   1.313ms | Compile   3.628ms | Speedup  3.05x | Throughput (C++/PyTorch):
17045992.9/5589243.6 elem/ms hc_head (T= 512, D=7168, bf16)           | C++   0.730ms | PyTorch   2.820ms |
Compile   5.946ms | Speedup  3.86x | Throughput (C++/PyTorch): 20106094.1/5204947.6 elem/ms hc_head (T=1024, D=7168,
bf16)           | C++   1.373ms | PyTorch   4.842ms | Compile  10.568ms | Speedup  3.53x | Throughput (C++/PyTorch):
21379404.4/6063276.0 elem/ms hc_head (T=2048, D=7168, bf16)           | C++   2.452ms | PyTorch  13.951ms |
Compile  21.130ms | Speedup  5.69x | Throughput (C++/PyTorch): 23950461.6/4209158.7 elem/ms


  */
