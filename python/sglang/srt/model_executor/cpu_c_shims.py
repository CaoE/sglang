# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""C shims for CPU custom ops under TorchInductor cpp_wrapper."""

import ctypes
import logging
import os
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)

_CPU_C_SHIMS_LOADED = False

_SHIM_SOURCE_CODE = r"""
#include <torch/extension.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <optional>
#include <vector>
#include <string>
#include <tuple>

void multimodal_rotary_embedding_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    const std::optional<std::vector<int64_t>>& mrope_section,
    bool mrope_interleaved,
    bool is_neox);

at::Tensor fp8_scaled_mm_cpu(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    std::vector<int64_t> block_size,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool is_vnni);

at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    int64_t moe_comp_method,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w1_zero,
    const std::optional<at::Tensor>& w2_zero,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    const std::optional<double>& alpha,
    const std::optional<double>& limit,
    bool is_vnni,
    const std::optional<std::string>& activation);

at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    const std::optional<at::Tensor>& fused_experts_out,
    const std::optional<double> routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    bool is_vnni);

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope(
    at::Tensor& hidden_states,
    at::Tensor& q_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& kv_a_proj_weight,
    at::Tensor& w_kc,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor> q_a_proj_scale,
    std::optional<at::Tensor> q_b_proj_scale,
    std::optional<at::Tensor> kv_a_proj_scale,
    std::optional<at::Tensor> w_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope_fused_weight(
    at::Tensor& hidden_states,
    at::Tensor& qkv_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& w_kc,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor> qkv_a_proj_scale,
    std::optional<at::Tensor> q_b_proj_scale,
    std::optional<at::Tensor> w_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size,
    int64_t q_lora_rank,
    int64_t kv_lora_rank,
    int64_t qk_rope_head_dim);

at::Tensor rmsnorm_cpu(at::Tensor& input, at::Tensor& weight, double eps);

void fused_add_rmsnorm_cpu(at::Tensor& input, at::Tensor& residual, at::Tensor& weight, double eps);

at::Tensor silu_and_mul_cpu(at::Tensor& input);

static inline at::Tensor* to_tensor(AtenTensorHandle h) {
    return h ? torch::aot_inductor::tensor_handle_to_tensor_pointer(h) : nullptr;
}

static inline std::optional<at::Tensor> to_opt_tensor(AtenTensorHandle h) {
    if (!h) return std::nullopt;
    return *torch::aot_inductor::tensor_handle_to_tensor_pointer(h);
}

static inline std::optional<double> to_opt_double(const double* p) {
    if (!p) return std::nullopt;
    return *p;
}

static inline std::optional<std::string> to_opt_string(const char* s) {
    if (!s) return std::nullopt;
    return std::string(s);
}

extern "C" {

int aoti_torch_cpu_multimodal_rotary_embedding_cpu(
    AtenTensorHandle positions,
    AtenTensorHandle query,
    AtenTensorHandle key,
    int64_t head_size,
    AtenTensorHandle cos_sin,
    const int64_t* const* mrope_section,
    int64_t mrope_section_size,
    bool mrope_interleaved,
    bool is_neox,
    void* unused_out)
{
    try {
        std::optional<std::vector<int64_t>> mrope_vec;
        if (mrope_section && *mrope_section && mrope_section_size > 0) {
            mrope_vec = std::vector<int64_t>(*mrope_section, *mrope_section + mrope_section_size);
        }
        multimodal_rotary_embedding_cpu(
            *to_tensor(positions), *to_tensor(query), *to_tensor(key),
            head_size, *to_tensor(cos_sin), mrope_vec, mrope_interleaved, is_neox
        );
        return AOTI_TORCH_SUCCESS;
    } catch (...) {
        return AOTI_TORCH_FAILURE;
    }
}

int aoti_torch_cpu_fp8_scaled_mm_cpu(
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    AtenTensorHandle scales2,
    const int64_t* block_size,
    int64_t block_size_len,
    AtenTensorHandle bias,
    int32_t out_dtype,
    bool is_vnni,
    AtenTensorHandle* out)
{
    try {
        std::vector<int64_t> bs_vec(block_size, block_size + block_size_len);
        at::Tensor res = fp8_scaled_mm_cpu(
            *to_tensor(mat1), *to_tensor(mat2), *to_tensor(scales2),
            bs_vec, to_opt_tensor(bias), static_cast<at::ScalarType>(out_dtype), is_vnni
        );
        *out = torch::aot_inductor::new_tensor_handle(std::move(res));
        return AOTI_TORCH_SUCCESS;
    } catch (...) {
        return AOTI_TORCH_FAILURE;
    }
}

int aoti_torch_cpu_fused_experts_cpu(
    AtenTensorHandle hidden_states,
    AtenTensorHandle w1,
    AtenTensorHandle w2,
    AtenTensorHandle topk_weights,
    AtenTensorHandle topk_ids,
    bool inplace,
    int64_t moe_comp_method,
    AtenTensorHandle w1_scale,
    AtenTensorHandle w2_scale,
    AtenTensorHandle w1_zero,
    AtenTensorHandle w2_zero,
    const int64_t* const* block_size,
    int64_t block_size_len,
    AtenTensorHandle w1_bias,
    AtenTensorHandle w2_bias,
    const double* alpha,
    const double* limit,
    bool is_vnni,
    const char* activation,
    AtenTensorHandle* out)
{
    try {
        std::optional<std::vector<int64_t>> bs_vec;
        if (block_size && *block_size && block_size_len > 0) {
            bs_vec = std::vector<int64_t>(*block_size, *block_size + block_size_len);
        }
        at::Tensor res = fused_experts_cpu(
            *to_tensor(hidden_states), *to_tensor(w1), *to_tensor(w2),
            *to_tensor(topk_weights), *to_tensor(topk_ids), inplace, moe_comp_method,
            to_opt_tensor(w1_scale), to_opt_tensor(w2_scale), to_opt_tensor(w1_zero), to_opt_tensor(w2_zero),
            bs_vec, to_opt_tensor(w1_bias), to_opt_tensor(w2_bias), to_opt_double(alpha), to_opt_double(limit),
            is_vnni, to_opt_string(activation)
        );
        *out = torch::aot_inductor::new_tensor_handle(std::move(res));
        return AOTI_TORCH_SUCCESS;
    } catch (...) {
        return AOTI_TORCH_FAILURE;
    }
}

int aoti_torch_cpu_shared_expert_cpu(
    AtenTensorHandle hidden_states,
    AtenTensorHandle w1,
    AtenTensorHandle w2,
    AtenTensorHandle fused_experts_out,
    const double* routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    AtenTensorHandle w1_scale,
    AtenTensorHandle w2_scale,
    const int64_t* const* block_size,
    int64_t block_size_len,
    bool is_vnni,
    AtenTensorHandle* out)
{
    try {
        std::optional<std::vector<int64_t>> bs_vec;
        if (block_size && *block_size && block_size_len > 0) {
            bs_vec = std::vector<int64_t>(*block_size, *block_size + block_size_len);
        }
        at::Tensor res = shared_expert_cpu(
            *to_tensor(hidden_states), *to_tensor(w1), *to_tensor(w2),
            to_opt_tensor(fused_experts_out), to_opt_double(routed_scaling_factor),
            inplace, use_int8_w8a8, use_fp8_w8a16,
            to_opt_tensor(w1_scale), to_opt_tensor(w2_scale), bs_vec, is_vnni
        );
        *out = torch::aot_inductor::new_tensor_handle(std::move(res));
        return AOTI_TORCH_SUCCESS;
    } catch (...) {
        return AOTI_TORCH_FAILURE;
    }
}

int aoti_torch_cpu_qkv_proj_with_rope(
    AtenTensorHandle hidden_states,
    AtenTensorHandle q_a_proj_weight,
    AtenTensorHandle q_b_proj_weight,
    AtenTensorHandle kv_a_proj_weight,
    AtenTensorHandle w_kc,
    AtenTensorHandle q_a_layernorm_weight,
    AtenTensorHandle kv_a_layernorm_weight,
    AtenTensorHandle positions,
    AtenTensorHandle cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    AtenTensorHandle q_a_proj_scale,
    AtenTensorHandle q_b_proj_scale,
    AtenTensorHandle kv_a_proj_scale,
    AtenTensorHandle w_scale,
    bool is_vnni,
    const int64_t* const* block_size,
    int64_t block_size_len,
    AtenTensorHandle* out0,
    AtenTensorHandle* out1,
    AtenTensorHandle* out2)
{
    try {
        std::optional<std::vector<int64_t>> bs_vec;
        if (block_size && *block_size && block_size_len > 0) {
            bs_vec = std::vector<int64_t>(*block_size, *block_size + block_size_len);
        }
        auto [t0, t1, t2] = qkv_proj_with_rope(
            *to_tensor(hidden_states), *to_tensor(q_a_proj_weight), *to_tensor(q_b_proj_weight),
            *to_tensor(kv_a_proj_weight), *to_tensor(w_kc), *to_tensor(q_a_layernorm_weight),
            *to_tensor(kv_a_layernorm_weight), *to_tensor(positions), *to_tensor(cos_sin_cache),
            eps, use_int8_w8a8, use_fp8_w8a16,
            to_opt_tensor(q_a_proj_scale), to_opt_tensor(q_b_proj_scale),
            to_opt_tensor(kv_a_proj_scale), to_opt_tensor(w_scale), is_vnni, bs_vec
        );
        *out0 = torch::aot_inductor::new_tensor_handle(std::move(t0));
        *out1 = torch::aot_inductor::new_tensor_handle(std::move(t1));
        *out2 = torch::aot_inductor::new_tensor_handle(std::move(t2));
        return AOTI_TORCH_SUCCESS;
    } catch (...) {
        return AOTI_TORCH_FAILURE;
    }
}

int aoti_torch_cpu_qkv_proj_with_rope_fused_weight(
    AtenTensorHandle hidden_states,
    AtenTensorHandle qkv_a_proj_weight,
    AtenTensorHandle q_b_proj_weight,
    AtenTensorHandle w_kc,
    AtenTensorHandle q_a_layernorm_weight,
    AtenTensorHandle kv_a_layernorm_weight,
    AtenTensorHandle positions,
    AtenTensorHandle cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    AtenTensorHandle qkv_a_proj_scale,
    AtenTensorHandle q_b_proj_scale,
    AtenTensorHandle w_scale,
    bool is_vnni,
    const int64_t* const* block_size,
    int64_t block_size_len,
    int64_t q_lora_rank,
    int64_t kv_lora_rank,
    int64_t qk_rope_head_dim,
    AtenTensorHandle* out0,
    AtenTensorHandle* out1,
    AtenTensorHandle* out2)
{
    try {
        std::optional<std::vector<int64_t>> bs_vec;
        if (block_size && *block_size && block_size_len > 0) {
            bs_vec = std::vector<int64_t>(*block_size, *block_size + block_size_len);
        }
        auto [t0, t1, t2] = qkv_proj_with_rope_fused_weight(
            *to_tensor(hidden_states), *to_tensor(qkv_a_proj_weight), *to_tensor(q_b_proj_weight),
            *to_tensor(w_kc), *to_tensor(q_a_layernorm_weight), *to_tensor(kv_a_layernorm_weight),
            *to_tensor(positions), *to_tensor(cos_sin_cache),
            eps, use_int8_w8a8, use_fp8_w8a16,
            to_opt_tensor(qkv_a_proj_scale), to_opt_tensor(q_b_proj_scale), to_opt_tensor(w_scale),
            is_vnni, bs_vec, q_lora_rank, kv_lora_rank, qk_rope_head_dim
        );
        *out0 = torch::aot_inductor::new_tensor_handle(std::move(t0));
        *out1 = torch::aot_inductor::new_tensor_handle(std::move(t1));
        *out2 = torch::aot_inductor::new_tensor_handle(std::move(t2));
        return AOTI_TORCH_SUCCESS;
    } catch (...) {
        return AOTI_TORCH_FAILURE;
    }
}

int aoti_torch_cpu_rmsnorm_cpu(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    double eps,
    AtenTensorHandle* out)
{
    try {
        at::Tensor res = rmsnorm_cpu(*to_tensor(input), *to_tensor(weight), eps);
        *out = torch::aot_inductor::new_tensor_handle(std::move(res));
        return AOTI_TORCH_SUCCESS;
    } catch (...) {
        return AOTI_TORCH_FAILURE;
    }
}

int aoti_torch_cpu_fused_add_rmsnorm_cpu(
    AtenTensorHandle input,
    AtenTensorHandle residual,
    AtenTensorHandle weight,
    double eps,
    void* unused_out)
{
    try {
        fused_add_rmsnorm_cpu(*to_tensor(input), *to_tensor(residual), *to_tensor(weight), eps);
        return AOTI_TORCH_SUCCESS;
    } catch (...) {
        return AOTI_TORCH_FAILURE;
    }
}

int aoti_torch_cpu_silu_and_mul_cpu(
    AtenTensorHandle input,
    AtenTensorHandle* out)
{
    try {
        at::Tensor res = silu_and_mul_cpu(*to_tensor(input));
        *out = torch::aot_inductor::new_tensor_handle(std::move(res));
        return AOTI_TORCH_SUCCESS;
    } catch (...) {
        return AOTI_TORCH_FAILURE;
    }
}

}
"""


def _get_common_ops_path() -> str:
    try:
        import sgl_kernel

        pkg_dir = os.path.dirname(sgl_kernel.__file__)
        for f in os.listdir(pkg_dir):
            if "common_ops" in f and f.endswith(".so"):
                return os.path.join(pkg_dir, f)
    except Exception as e:
        logger.warning(f"Failed to find sgl_kernel common_ops.so: {e}")
    return ""


def ensure_cpu_c_shims_loaded() -> None:
    """Ensure CPU C shims are loaded into the global symbol table (RTLD_GLOBAL)."""
    global _CPU_C_SHIMS_LOADED
    if _CPU_C_SHIMS_LOADED:
        return

    common_ops_so = _get_common_ops_path()
    if common_ops_so:
        try:
            # First, ensure common_ops symbols are visible globally
            ctypes.CDLL(common_ops_so, mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            logger.warning(f"Failed to load {common_ops_so} via RTLD_GLOBAL: {e}")

    # Check if aoti_torch_cpu_multimodal_rotary_embedding_cpu is already exported by common_ops
    handle = ctypes.CDLL(None)
    if hasattr(handle, "aoti_torch_cpu_multimodal_rotary_embedding_cpu"):
        _CPU_C_SHIMS_LOADED = True
        return

    # If not yet exported, build inline shim library linking against common_ops.so
    try:
        from torch.utils.cpp_extension import load_inline

        extra_ldflags = [common_ops_so] if common_ops_so else []
        shim_lib_path = load_inline(
            name="sgl_cpu_aoti_shims",
            cpp_sources=_SHIM_SOURCE_CODE,
            extra_ldflags=extra_ldflags,
            is_python_module=False,
            verbose=False,
        )
        ctypes.CDLL(shim_lib_path, mode=ctypes.RTLD_GLOBAL)
        _CPU_C_SHIMS_LOADED = True
        logger.info("Successfully loaded SGLang CPU AOTI C shims.")
    except Exception as e:
        logger.warning(
            f"Failed to compile/load SGLang CPU AOTI C shims inline: {e}. "
            "TorchInductor cpp_wrapper custom op calls may fall back."
        )


def get_cpu_custom_ops_to_c_shims() -> Dict[object, List[str]]:
    """Return Inductor custom_ops_to_c_shims dictionary for CPU ops."""
    shims = {}
    sgl_ops = getattr(torch.ops, "sgl_kernel", None)
    if sgl_ops is None:
        return shims

    def register(op_name: str, c_signature: str):
        op_obj = getattr(sgl_ops, op_name, None)
        if op_obj is not None and hasattr(op_obj, "default"):
            shims[op_obj.default] = [c_signature]

    # 1. multimodal_rotary_embedding_cpu (List[int]? mrope_section)
    register(
        "multimodal_rotary_embedding_cpu",
        "int aoti_torch_cpu_multimodal_rotary_embedding_cpu("
        "AtenTensorHandle positions, AtenTensorHandle query, AtenTensorHandle key, "
        "int64_t head_size, AtenTensorHandle cos_sin, "
        "const int64_t* const* mrope_section, int64_t mrope_section_size, "
        "bool mrope_interleaved, bool is_neox, void* out)",
    )

    # 2. fp8_scaled_mm_cpu (List[int] block_size)
    register(
        "fp8_scaled_mm_cpu",
        "int aoti_torch_cpu_fp8_scaled_mm_cpu("
        "AtenTensorHandle mat1, AtenTensorHandle mat2, AtenTensorHandle scales2, "
        "const int64_t* block_size, int64_t block_size_len, "
        "AtenTensorHandle bias, int32_t out_dtype, bool is_vnni, "
        "AtenTensorHandle* out)",
    )

    # 3. fused_experts_cpu (List[int]? block_size, str? activation)
    register(
        "fused_experts_cpu",
        "int aoti_torch_cpu_fused_experts_cpu("
        "AtenTensorHandle hidden_states, AtenTensorHandle w1, AtenTensorHandle w2, "
        "AtenTensorHandle topk_weights, AtenTensorHandle topk_ids, "
        "bool inplace, int64_t moe_comp_method, "
        "AtenTensorHandle w1_scale, AtenTensorHandle w2_scale, "
        "AtenTensorHandle w1_zero, AtenTensorHandle w2_zero, "
        "const int64_t* const* block_size, int64_t block_size_len, "
        "AtenTensorHandle w1_bias, AtenTensorHandle w2_bias, "
        "const double* alpha, const double* limit, bool is_vnni, "
        "const char* activation, AtenTensorHandle* out)",
    )

    # 4. shared_expert_cpu (List[int]? block_size)
    register(
        "shared_expert_cpu",
        "int aoti_torch_cpu_shared_expert_cpu("
        "AtenTensorHandle hidden_states, AtenTensorHandle w1, AtenTensorHandle w2, "
        "AtenTensorHandle fused_experts_out, const double* routed_scaling_factor, "
        "bool inplace, bool use_int8_w8a8, bool use_fp8_w8a16, "
        "AtenTensorHandle w1_scale, AtenTensorHandle w2_scale, "
        "const int64_t* const* block_size, int64_t block_size_len, "
        "bool is_vnni, AtenTensorHandle* out)",
    )

    # 5. qkv_proj_with_rope (List[int]? block_size)
    register(
        "qkv_proj_with_rope",
        "int aoti_torch_cpu_qkv_proj_with_rope("
        "AtenTensorHandle hidden_states, AtenTensorHandle q_a_proj_weight, "
        "AtenTensorHandle q_b_proj_weight, AtenTensorHandle kv_a_proj_weight, "
        "AtenTensorHandle w_kc, AtenTensorHandle q_a_layernorm_weight, "
        "AtenTensorHandle kv_a_layernorm_weight, AtenTensorHandle positions, "
        "AtenTensorHandle cos_sin_cache, double eps, bool use_int8_w8a8, bool use_fp8_w8a16, "
        "AtenTensorHandle q_a_proj_scale, AtenTensorHandle q_b_proj_scale, "
        "AtenTensorHandle kv_a_proj_scale, AtenTensorHandle w_scale, "
        "bool is_vnni, const int64_t* const* block_size, int64_t block_size_len, "
        "AtenTensorHandle* out0, AtenTensorHandle* out1, AtenTensorHandle* out2)",
    )

    # 6. qkv_proj_with_rope_fused_weight (List[int]? block_size)
    register(
        "qkv_proj_with_rope_fused_weight",
        "int aoti_torch_cpu_qkv_proj_with_rope_fused_weight("
        "AtenTensorHandle hidden_states, AtenTensorHandle qkv_a_proj_weight, "
        "AtenTensorHandle q_b_proj_weight, AtenTensorHandle w_kc, "
        "AtenTensorHandle q_a_layernorm_weight, AtenTensorHandle kv_a_layernorm_weight, "
        "AtenTensorHandle positions, AtenTensorHandle cos_sin_cache, "
        "double eps, bool use_int8_w8a8, bool use_fp8_w8a16, "
        "AtenTensorHandle qkv_a_proj_scale, AtenTensorHandle q_b_proj_scale, "
        "AtenTensorHandle w_scale, bool is_vnni, "
        "const int64_t* const* block_size, int64_t block_size_len, "
        "int64_t q_lora_rank, int64_t kv_lora_rank, int64_t qk_rope_head_dim, "
        "AtenTensorHandle* out0, AtenTensorHandle* out1, AtenTensorHandle* out2)",
    )

    # Core high-frequency compute ops
    register(
        "rmsnorm_cpu",
        "int aoti_torch_cpu_rmsnorm_cpu("
        "AtenTensorHandle input, AtenTensorHandle weight, double eps, "
        "AtenTensorHandle* out)",
    )
    register(
        "fused_add_rmsnorm_cpu",
        "int aoti_torch_cpu_fused_add_rmsnorm_cpu("
        "AtenTensorHandle input, AtenTensorHandle residual, AtenTensorHandle weight, "
        "double eps, void* unused_out)",
    )
    register(
        "silu_and_mul_cpu",
        "int aoti_torch_cpu_silu_and_mul_cpu("
        "AtenTensorHandle input, AtenTensorHandle* out)",
    )

    return shims


def enable_cpu_cpp_wrapper() -> None:
    """Configure TorchInductor for CPU C++ wrapper with direct C shims."""
    ensure_cpu_c_shims_loaded()
    shims = get_cpu_custom_ops_to_c_shims()
    if shims:
        torch._inductor.config.aot_inductor.custom_ops_to_c_shims.update(shims)
    torch._inductor.config.cpp_wrapper = True
    logger.info("TorchInductor CPU cpp_wrapper enabled with custom op C shims.")
