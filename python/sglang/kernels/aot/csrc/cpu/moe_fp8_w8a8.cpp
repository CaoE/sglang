#include <c10/util/Unroll.h>

#include "common.h"
#include "gemm.h"
#include "vec.h"
namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
// no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += Vec::size()) {
    Vec data = Vec::loadu(input + d);
    data.store(out + d);
  }
}

inline void copy_stub2(at::Float8_e4m3fn* __restrict__ out, const at::Float8_e4m3fn* __restrict__ input, int64_t size) {
  // no remainder
  // #pragma GCC unroll 4
  for (int64_t d = 0; d < size; d++) {
    out[d] = input[d];
  }
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, float weight, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    bVec x = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x);
    x0 = x0 * weight_vec;
    x1 = x1 * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * weight);
  }
}

// acc from [topk, K] to [K]
template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t topk, int64_t K) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  if (topk == 1) {
    // do copy for topk = 1
    copy_stub(out, input, K);
  } else {
    // do sum for topk != 1
    int64_t d;
#pragma GCC unroll 4
    for (d = 0; d <= K - kVecSize; d += kVecSize) {
      fVec sum_fvec0 = fVec(0.f);
      fVec sum_fvec1 = fVec(0.f);
      for (int t = 0; t < topk; ++t) {
        bVec x_bvec = bVec::loadu(input + t * K + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        sum_fvec0 += x_fvec0;
        sum_fvec1 += x_fvec1;
      }
      bVec out_bvec = convert_from_float_ext<scalar_t>(sum_fvec0, sum_fvec1);
      out_bvec.store(out + d);
    }
    for (; d < K; ++d) {
      float sum_val = 0.f;
      for (int t = 0; t < topk; ++t) {
        sum_val += static_cast<float>(input[t * K + d]);
      }
      out[d] = static_cast<scalar_t>(sum_val);
    }
  }
}

// out = input + input2 * scale
template <typename scalar_t>
inline void add_mul_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ input2,
    float scale,
    int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_vec = fVec(scale);

  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    bVec x_bvec = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x_bvec);

    bVec y_bvec = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);

    x0 = x0 + y0 * s_vec;
    x1 = x1 + y1 * s_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + float(input2[d]) * scale);
  }
}

template <typename scalar_t>
inline void silu_and_mul_stub(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const scalar_t* __restrict__ input2, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);

  // no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += bVec::size()) {
    bVec x = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x);
    bVec y = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y);
    x0 = x0 / (one + x0.neg().exp_u20());
    x1 = x1 / (one + x1.neg().exp_u20());
    x0 = x0 * y0;
    x1 = x1 * y1;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
}

#define PER_TENSOR 1
#define PER_ROW 2
#define PER_GROUP 3

// Store result to output buffer with dtype conversion
// If act/wei are per_row or per_tensor quantized, apply scales
// If bias is not null, add bias
template <typename out_dtype, int64_t N, int act_quant_mode, int wei_quant_mode>
inline void store_out(
    const float* y_buf,
    out_dtype* c_ptr,
    int64_t M,
    int64_t lda,
    const float* scales_a,
    const float* scales_b,
    const float* bias) {
  float a_scale = 1.0, b_scale = 1.0;
  __m512 va_scale, vb_scale;
  if constexpr (act_quant_mode == PER_TENSOR) {
    a_scale = *scales_a;
  }
  if constexpr (wei_quant_mode == PER_TENSOR) {
    b_scale = *scales_b;
    vb_scale = _mm512_set1_ps(b_scale);
  }
  for (int i = 0; i < M; ++i) {
    if constexpr (act_quant_mode == PER_ROW) {
      a_scale = *(scales_a + i);
    }
    if constexpr (act_quant_mode != PER_GROUP) {
      va_scale = _mm512_set1_ps(a_scale);
    }
    constexpr int N_UNROLL = N / 16;
    c10::ForcedUnroll<N_UNROLL>{}([&](auto idx) {
      constexpr int j = idx * 16;
      __m512 y_vec = _mm512_loadu_ps(y_buf + i * N + j);
      __m512 bias_vec = bias ? _mm512_loadu_ps(bias + j) : _mm512_setzero_ps();
      if constexpr (act_quant_mode != PER_GROUP) {
        y_vec = _mm512_mul_ps(y_vec, va_scale);
      }
      if constexpr (wei_quant_mode == PER_ROW) {
        vb_scale = _mm512_loadu_ps(scales_b + j);
      }
      if constexpr (wei_quant_mode != PER_GROUP) {
        y_vec = _mm512_mul_ps(y_vec, vb_scale);
      }
      y_vec = _mm512_add_ps(y_vec, bias_vec);
      if constexpr (std::is_same<out_dtype, float>::value) {
        _mm512_storeu_ps(c_ptr + i * lda + j, y_vec);
      } else if constexpr (std::is_same<out_dtype, at::BFloat16>::value) {
        __m256i y_bf16_vec = at::vec::cvtfp32_bf16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_bf16_vec);
      } else if constexpr (std::is_same<out_dtype, at::Half>::value) {
        __m256i y_fp16_vec = at::vec::cvtfp32_fp16(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_ptr + i * lda + j), y_fp16_vec);
      } else {
        TORCH_CHECK(false, "Unsupported output dtype");
      }
    });
    constexpr int tail_start = N / 16 * 16;
    for (int j = tail_start; j < N; ++j) {
      if constexpr (wei_quant_mode == PER_ROW) {
        b_scale = scales_b[j];
      }
      c_ptr[i * lda + j] = static_cast<out_dtype>(y_buf[i * N + j] * a_scale * b_scale);
    }
  }  // for M
}

}  // anonymous namespace

// act : per channel quant
// weight: per block_N x block_K quant
template <typename scalar_t>
void fused_experts_fp8_a8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    at::Float8_e4m3fn* __restrict__ A_tmp,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    float* __restrict__ Ukernel_tmp,
    const at::Float8_e4m3fn* __restrict__ input,
    const at::Float8_e4m3fn* __restrict__ packed_w1,
    const at::Float8_e4m3fn* __restrict__ packed_w2,
    const float* __restrict__ As,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad) {
  //   1. intermediate_cache1 : [M * topk, N]
  //   2. intermediate_cache2 : [M * topk, K]
  //   3. A_tmp : [T, BLOCK_M * K]
  //   4. C_tmp : [T, 2 * BLOCK_M * BLOCK_N]
  //   7. intermediate_cache0 : [M * topk, 2N]
  //   8. B_tmp : [T, MAX_CACHE_BLOCK_SIZE, BLOCK_N, std::max(K, N)]
  //   9. ukernel buffer: [T, 2 * BLOCK_M * BLOCK_N]

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  int64_t B_tmp_size_per_thread = MAX_CACHE_BLOCK_SIZE * BLOCK_N * std::max(K, N);
  // stage 1: intermediate_cache0 = hidden_states @ w1
  const int64_t MB = div_up(num_tokens_post_pad, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  const int64_t KB = K / BLOCK_K;  // use div_up
  int64_t scale_size_N = div_up(2 * N, block_size_N);
  int64_t scale_size_K = div_up(K, block_size_K);
  int64_t blocks_n_per_group = block_size_N / BLOCK_N;  // use div_up
  int64_t num_groups = div_up(K, block_size_K);         // G, block_size_K is group size
  int64_t blocks_k_per_group = block_size_K / BLOCK_K;  // use div_up
  const int64_t stride_e = 2 * N * K;
  const int64_t stride_n = K;
  bool use_brgemm = true;
  int64_t block_size = BLOCK_M * BLOCK_N;
  int64_t num_thread = at::get_num_threads();
  at::parallel_for(0, MB * NB, 1, [&](int64_t begin, int64_t end) {
    // get local pointers
    int tid = get_thread_num();
    at::Float8_e4m3fn* __restrict__ A = A_tmp + tid * BLOCK_M * K;
    float* __restrict__ C0 = C_tmp + tid * 2 * BLOCK_M * BLOCK_N;
    float* __restrict__ C1 = C0 + BLOCK_M * BLOCK_N;
    alignas(64) float As_[BLOCK_M];
    float* __restrict__ ukernel_buf_1 = Ukernel_tmp + +tid * 2 * BLOCK_M * BLOCK_N;
    float* __restrict__ ukernel_buf_2 = ukernel_buf_1 + BLOCK_M * BLOCK_N;
    auto ldsa = 1;  // act per PER_ROW quant
    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB;
      int64_t nb = i % NB;
      int64_t nb1 = nb + NB;
      int64_t n_size = std::min(N - nb * BLOCK_N, BLOCK_N);

      // B shape [K, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const at::Float8_e4m3fn* __restrict__ B = packed_w1 + expert_id * stride_e;
      const float* __restrict__ Bs = w1s + expert_id * (num_groups) * (2 * N);
      // 1.a load A
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;
      int64_t m_size = offsets[mb + 1] - offsets[mb];

      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m] / topk;
        copy_stub2(A + m * K, input + index * K, K);
        As_[m] = As[index];
      }
      // A shape [m_size, K]
      const int64_t offset = offsets[mb];
      // ic0 + offset * 2 * N + nb * BLOCK_N,
      zero_buffer(C0, BLOCK_M * BLOCK_N);
      for (int kci = 0; kci < KB; ++kci) {
        // cpublas_can_pack = True, act_quant_mode = per row (2), wei_quant_mode = per group (3)
        tinygemm_kernel(
            /* C */ C0,
            /* A */ A + kci * BLOCK_K,
            /* scales_a */ As_ + kci / blocks_k_per_group,
            /* B */ B + (nb * KB + kci) * BLOCK_K * BLOCK_N,
            /* scales_b */ Bs + nb * BLOCK_N * num_groups + kci / blocks_k_per_group * BLOCK_N /*scales_b*/,
            /* M */ m_size,
            /* K */ BLOCK_K,
            /* lda */ K,
            /* ldc */ BLOCK_N,
            /* ldsa */ ldsa,
            /* ukernel_buf */ ukernel_buf_1,
            /* dqA_buf */ nullptr,
            /* dqB_buf */ nullptr);
      }
      store_out<scalar_t, BLOCK_N, 2, 3>(
          C0, ic0 + offset * 2 * N + nb * BLOCK_N, m_size, 2 * N /*lda*/, As_, nullptr, nullptr /*bias data*/);
      zero_buffer(C1, BLOCK_M * BLOCK_N);
      for (int kci = 0; kci < KB; ++kci) {
        // cpublas_can_pack = True, act_quant_mode = per row (2), wei_quant_mode = per group (3)
        tinygemm_kernel(
            /* C */ C1,
            /* A */ A + kci * BLOCK_K,
            /* scales_a */ As_ + kci / blocks_k_per_group,
            /* B */ B + (nb1 * KB + kci) * BLOCK_K * BLOCK_N,
            /* scales_b */ Bs + nb1 * BLOCK_N * num_groups + kci / blocks_k_per_group * BLOCK_N /*scales_b*/,
            /* M */ m_size,
            /* K */ BLOCK_K,
            /* lda */ K,
            /* ldc */ BLOCK_N,
            /* ldsa */ ldsa,
            /* ukernel_buf */ ukernel_buf_2,
            /* dqA_buf */ nullptr,
            /* dqB_buf */ nullptr);
      }
      store_out<scalar_t, BLOCK_N, 2, 3>(
          C1, ic0 + offset * 2 * N + nb1 * BLOCK_N, m_size, 2 * N /*lda*/, As_, nullptr, nullptr /*bias data*/);
    }
    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 1.5: intermediate_cache1 = silu(intermediate_cache0)
  at::parallel_for(0, M * topk, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      silu_and_mul_stub(ic1 + m * N, ic0 + m * 2 * N, ic0 + m * 2 * N + N, N);
    }
  });

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [E, K, N] as [E, OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  scale_size_N = div_up(K, block_size_N);
  scale_size_K = div_up(N, block_size_K);
  const int64_t stride_e2 = OC * IC;
  const int64_t stride_oc = IC;
  int64_t avg_M = std::max(int64_t(1), M * topk / E);
  use_brgemm = can_use_brgemm<at::Float8_e4m3fn>(avg_M);

  // parallel on [MB2, NB2]
  parallel_2d(MB2, NB2, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    int tid = get_thread_num();
    alignas(64) scalar_t C[BLOCK_M * BLOCK_K];

    loop_2d<at::Float8_e4m3fn>(mb0, mb1, nb0, nb1, BLOCK_N * IC, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t m_size = offsets[mb + 1] - offsets[mb];
      int64_t n_size = std::min(OC - nb * BLOCK_N, BLOCK_N);

      // A ptr from ic1 of [M * topk, N] in sorted order
      // so as to avoid copy A to tmp buffer again
      const scalar_t* __restrict__ A = ic1 + offsets[mb] * N;
      const int32_t* A_ids = sorted_ids + mb * BLOCK_M;

      // B shape [IC, n_size] in vnni format
      int32_t expert_id = expert_ids[mb];
      const at::Float8_e4m3fn* __restrict__ B = packed_w2 + expert_id * stride_e2 + nb * BLOCK_N * stride_oc;
      const float* __restrict__ Bs =
          w2s + expert_id * scale_size_N * scale_size_K + (nb / blocks_n_per_group) * scale_size_K;

      // do unpacking for the first row or a new expert
      int32_t pre_expert_id = mb == 0 ? -1 : expert_ids[mb - 1];
      bool do_unpack = (mb == mb0) || (expert_id != pre_expert_id);

      tinygemm_kernel<scalar_t>(
          /*   A            */ A,
          /*   B            */ B,
          /*   C            */ C,
          /*   Btmp         */ B_tmp + tid * B_tmp_size_per_thread + nb_offset * BLOCK_N * IC,
          /*   Ctmp         */ C_tmp + tid * 2 * BLOCK_M * BLOCK_N,
          /*   Bbias        */ nullptr,
          /*   scale        */ Bs,
          /*   M            */ m_size,
          /*   N            */ n_size,
          /*   K            */ IC,
          /*   lda          */ IC,
          /*   ldb          */ n_size,
          /*   ldc          */ BLOCK_N,
          /*   brg          */ use_brgemm,
          /*   block_size_K */ block_size_K,
          /*   do_unpack    */ do_unpack);

      // 2.b copy from C to ic2 in original order
      //   and also mul topk_weights in float32
      for (int64_t m = 0; m < m_size; ++m) {
        int32_t index = A_ids[m];
        float weight = topk_weights[index];
        copy_mul_stub(ic2 + index * K + nb * BLOCK_N, C + m * BLOCK_N, weight, n_size);
      }
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });

  // stage 3: out = intermediate_cache2.sum(dim=1)
  //   from [M, topk, K] to [M, K]
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      sum_stub(output + m * K, ic2 + m * topk * K, topk, K);
    }
  });
}

#define INSTANTIATE_MOE_FP8_A8_TEMPLATE(TYPE)           \
  template void fused_experts_fp8_a8_kernel_impl<TYPE>( \
      TYPE* __restrict__ output,                        \
      TYPE* __restrict__ ic0,                           \
      TYPE* __restrict__ ic1,                           \
      TYPE* __restrict__ ic2,                           \
      at::Float8_e4m3fn* __restrict__ A_tmp,            \
      TYPE* __restrict__ B_tmp,                         \
      float* __restrict__ C_tmp,                        \
      float* __restrict__ Ukernel_tmp,                  \
      const at::Float8_e4m3fn* __restrict__ input,      \
      const at::Float8_e4m3fn* __restrict__ packed_w1,  \
      const at::Float8_e4m3fn* __restrict__ packed_w2,  \
      const float* __restrict__ As,                     \
      const float* __restrict__ w1s,                    \
      const float* __restrict__ w2s,                    \
      int64_t block_size_N,                             \
      int64_t block_size_K,                             \
      const float* __restrict__ topk_weights,           \
      const int32_t* __restrict__ sorted_ids,           \
      const int32_t* __restrict__ expert_ids,           \
      const int32_t* __restrict__ offsets,              \
      int64_t M,                                        \
      int64_t N,                                        \
      int64_t K,                                        \
      int64_t E,                                        \
      int64_t topk,                                     \
      int64_t num_tokens_post_pad)

INSTANTIATE_MOE_FP8_A8_TEMPLATE(at::BFloat16);
INSTANTIATE_MOE_FP8_A8_TEMPLATE(at::Half);
