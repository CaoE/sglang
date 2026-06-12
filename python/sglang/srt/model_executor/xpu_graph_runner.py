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
"""Run the model with xpu graph and torch.compile."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
# from sglang.srt.layers.attention.fla.layernorm_gated import prefetch_sm_count
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


def register_fake_ops():
    """Register fake/abstract implementations for XPU sgl_kernel ops so that
    torch.compile (Dynamo) can trace through them using FakeTensors for shape
    and dtype propagation, without executing the real GPU kernels.

    sgl_kernel.fwd returns (Tensor, Tensor, Tensor, Tensor) = (out, softmax_lse, out_accum, softmax_lse_accum)
    where:
      out:             [total_q, num_heads_q, head_size_v]   (same dtype as q)
      softmax_lse:     [num_heads_q, total_q]                (float32)
      out_accum:       may be empty (when num_kv_splits == 1, out_accum == out)
      softmax_lse_accum: may be empty
    For shape inference we conservatively return the two primary outputs and
    two empty tensors for the accum buffers (Dynamo only needs correct shapes
    for tensors that are actually consumed downstream).
    """

    @torch.library.register_fake("sgl_kernel::fwd")
    def _(
        q,
        k,
        v,
        q_v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        sinks,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        is_rotary_interleaved,
        scheduler_metadata,
        num_kv_splits,
        pack_gqa,
        sm_margin,
    ):
        total_q = q.shape[0]
        num_heads_q = q.shape[1]
        head_size_v = v.shape[-1]
        out = q.new_empty(total_q, num_heads_q, head_size_v)
        softmax_lse = q.new_empty(num_heads_q, total_q, dtype=torch.float32)
        # out_accum and softmax_lse_accum are intermediate split-kv buffers;
        # they are only read when num_kv_splits > 1, which is determined at
        # runtime.  Return empty tensors with correct rank so downstream ops
        # that index into the list do not fail shape propagation.
        out_accum = q.new_empty(0)
        softmax_lse_accum = q.new_empty(0, dtype=torch.float32)
        return (out, softmax_lse, out_accum, softmax_lse_accum)

    @torch.library.register_fake("sgl_kernel::flash_mla_decode")
    def _(out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits):
        return


class XPUGraphRunner(CudaGraphRunner):
    """A XPUGraphRunner runs the forward pass of a model with xpu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        register_fake_ops()
        # # Pre-fetch gpu_subslice_count into a plain Python int before
        # # torch.compile starts tracing.  Without this, Dynamo would try to
        # # evaluate torch.xpu.get_device_properties() symbolically and fail
        # # because _XpuDeviceProperties is not a supported ConstantVariable type.
        # prefetch_sm_count(torch.device("xpu", model_runner.gpu_id))
        super().__init__(model_runner)

        # model_runner.server_args.disable_cuda_graph_padding
        # require_attn_tp_gather(model_runner.server_args)
        # model_runner.server_args.enable_pdmux

        # assert (
        #     self.model_runner.server_args.disable_piecewise_cuda_graph
        # ), "XPUGraphRunner does not support Piecewise Graph yet."

        # assert (
        #     not self.model_runner.server_args.enforce_piecewise_cuda_graph
        # ), "XPUGraphRunner does not support forced enabling Piecewise Graph yet."

        assert (
            not self.model_runner.server_args.enable_memory_saver
        ), "XPUGraphRunner does not support Torch Memory Saver yet."

        assert (
            not self.model_runner.server_args.enable_lora
        ), "XPUGraphRunner does not support LoRA yet."
        assert (
            not self.enable_two_batch_overlap
        ), "XPUGraphRunner does not support two batch overlap yet."
        assert (
            not self.require_mlp_tp_gather
        ), "XPUGraphRunner does not support MLP TP gather yet."
        assert (
            not self.require_mlp_sync
        ), "XPUGraphRunner does not support MLP sync yet."
        assert (
            not self.require_gathered_buffer
        ), "XPUGraphRunner does not support gathered buffer yet."
        assert (
            model_runner.spec_algorithm == SpeculativeAlgorithm.NONE
        ), "XPUGraphRunner does not support speculative inference yet."
        # TODO add compile support for encoder-decoder models
        assert (
            not self.is_encoder_decoder
        ), "XPUGraphRunner does not support encoder-decoder models yet."
        assert self.dp_size == 1, "XPUGraphRunner does not support DP yet."
        assert self.pp_size == 1, "XPUGraphRunner does not support PP yet."

    def _create_device_graph(self):
        return torch.xpu.XPUGraph()

    # def replay(
    #     self,
    #     forward_batch: ForwardBatch,
    #     skip_attn_backend_init: bool = False,
    #     pp_proxy_tensors: Optional[PPProxyTensors] = None,
    # ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
    #     # During extend (prefill), dist.all_reduce triggers OneCCL to call
    #     # begin_recording() on the default stream (confirmed by ARDebug log).
    #     # XPUGraph.replay() C++ checks the current stream's capture status
    #     # before executing. Since replay() is called with no stream context
    #     # manager active, current stream == default stream == the same stream
    #     # that OneCCL's recording is on.
    #     # Poll is_capturing() (pure CPU-side, no queue.wait()) until the
    #     # recording ends, then replay.
    #     import time
    #     cs = torch.xpu.current_stream()
    #     if cs.is_capturing():
    #         t0 = time.monotonic()
    #         while cs.is_capturing():
    #             if time.monotonic() - t0 > 30.0:
    #                 logger.warning(
    #                     "XPU default stream still in OneCCL recording state after 30s; "
    #                     "proceeding with graph replay."
    #                 )
    #                 break
    #             time.sleep(0.001)
    #     return super().replay(forward_batch, skip_attn_backend_init, pp_proxy_tensors)

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        # import os
        # _debug = os.environ.get("SGLANG_XPU_GRAPH_DEBUG", "0") == "1"
        # if _debug:
        #     cs = torch.xpu.current_stream()
        #     logger.warning(
        #         f"[XPUGraphDebug] _capture_graph ENTER: current_stream={cs.sycl_queue:#x} "
        #         f"capture_stream={stream.sycl_queue:#x} is_capturing={cs.is_capturing()}"
        #     )
        #     graph.enable_debug_mode()

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.model_runner.server_args.enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        graph_fn = (
            partial(memory_saver_adapter.cuda_graph, tag=GPU_MEMORY_TYPE_CUDA_GRAPH)
            if memory_saver_adapter.enabled
            else self.device_module.graph
        )
        with graph_fn(xpu_graph=graph, pool=pool, stream=stream):
            # if _debug:
            #     cs_in = torch.xpu.current_stream()
            #     logger.warning(
            #         f"[XPUGraphDebug] inside graph context: current_stream={cs_in.sycl_queue:#x} "
            #         f"is_capturing={cs_in.is_capturing()}"
            #     )
            out = run_once_fn()

        # if _debug:
        #     cs_out = torch.xpu.current_stream()
        #     logger.warning(
        #         f"[XPUGraphDebug] _capture_graph EXIT: current_stream={cs_out.sycl_queue:#x} "
        #         f"is_capturing={cs_out.is_capturing()}"
        #     )
        return out

    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        )
        torch.xpu.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.xpu.memory._dump_snapshot(f"xpu_graph_runner_memory_usage.pickle")
        torch.xpu.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by XPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_xpu_time_total"
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total"
            )
            + "\n\nMemory Usage is saved to xpu_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)
