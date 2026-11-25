# Copyright 2023-2026 SGLang Team
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

"""ViT XPU Graph Runner class."""

from __future__ import annotations

from typing import Hashable, List, Optional, Tuple

import torch

from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner


class ViTXpuGraphRunner(ViTCudaGraphRunner):
    """Generic ViT XPU Graph Runner.

    This runner captures the "blocks + merger + deepstack merger (optional)" part
    of a vision transformer into an XPU graph and replays it for identical shapes.
    """

    _graph_memory_pool = None

    def _create_graph(
        self,
        graph_key: Hashable,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
    ):
        if ViTXpuGraphRunner._graph_memory_pool is None:
            ViTXpuGraphRunner._graph_memory_pool = torch.xpu.graph_pool_handle()

        graph = torch.xpu.XPUGraph()
        vit = self.vit

        # Qwen2.5-VL
        if self._fullatt_block_indexes:
            cu_window = self.cu_window_len[graph_key]
            cu_window_kk = self.cu_window_len_kk[graph_key]
            max_window_len = int(cu_window_kk.max().item())

        cu_full = self.cu_full_len[graph_key]
        cu_full_kk = self.cu_full_len_kk[graph_key]
        max_full_len = int(cu_full_kk.max().item())

        backend = self._attn_backend

        if self._fullatt_block_indexes and 0 not in vit.fullatt_block_indexes:
            warmup_cu_ws = [cu_window, cu_window_kk, max_window_len]
        else:
            warmup_cu_ws = [cu_full, cu_full_kk, max_full_len]

        if backend in ("fa3", "fa4", "xpu_attn"):
            warmup_layouts = [[warmup_cu_ws[0], warmup_cu_ws[2]]]
        elif backend == "triton_attn":
            warmup_layouts = [warmup_cu_ws]
        else:
            raise RuntimeError(
                f"ViT XPU graph does not support attention backend: {backend}"
            )

        warmup_kwargs = dict(output_ws=self.block_ws[graph_key])
        if position_embeddings is not None:
            warmup_kwargs["position_embeddings"] = position_embeddings
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            warmup_kwargs["rotary_pos_emb_cos"] = rotary_pos_emb_cos
            warmup_kwargs["rotary_pos_emb_sin"] = rotary_pos_emb_sin

        capture_stream = torch.xpu.current_stream()
        with torch.no_grad():
            for layout in warmup_layouts:
                vit.blocks[0](
                    self.block_input[graph_key], cu_seqlens=layout, **warmup_kwargs
                )
        torch.xpu.synchronize()

        with (
            self._capture_context(),
            torch.xpu.graph(
                xpu_graph=graph,
                pool=ViTXpuGraphRunner._graph_memory_pool,
                stream=capture_stream,
            ),
        ):
            y = None
            deepstack_outs: List[torch.Tensor] = []
            deepstack_capture_idx = 0

            for layer_num, blk in enumerate(vit.blocks):
                if self._fullatt_block_indexes:
                    if layer_num in vit.fullatt_block_indexes:
                        cu_seqlens_now = cu_full
                        cu_seqlens_kk_now = cu_full_kk
                        max_len = max_full_len
                    else:
                        cu_seqlens_now = cu_window
                        cu_seqlens_kk_now = cu_window_kk
                        max_len = max_window_len
                else:
                    cu_seqlens_now = cu_full
                    cu_seqlens_kk_now = cu_full_kk
                    max_len = max_full_len

                if backend in ("fa3", "fa4", "xpu_attn"):
                    cu_seq_len_ws = [cu_seqlens_now, max_len]
                elif backend == "triton_attn":
                    cu_seq_len_ws = [cu_seqlens_now, cu_seqlens_kk_now, max_len]

                blk_in = self.block_input[graph_key] if layer_num == 0 else y
                if position_embeddings is not None:
                    y = blk(
                        blk_in,
                        cu_seqlens=cu_seq_len_ws,
                        position_embeddings=position_embeddings,
                        output_ws=self.block_ws[graph_key],
                    )
                elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
                    y = blk(
                        blk_in,
                        cu_seqlens=cu_seq_len_ws,
                        rotary_pos_emb_cos=rotary_pos_emb_cos,
                        rotary_pos_emb_sin=rotary_pos_emb_sin,
                        output_ws=self.block_ws[graph_key],
                    )

                # Optional deepstack support (Qwen3-VL)
                if (
                    self._deepstack_visual_indexes
                    and layer_num in self._deepstack_visual_indexes
                ):
                    if self._deepstack_merger_list is None:
                        raise RuntimeError(
                            "deepstack_visual_indexes exists but deepstack_merger_list is missing."
                        )
                    deepstack_out = self._deepstack_merger_list[deepstack_capture_idx](
                        y
                    )
                    deepstack_outs.append(deepstack_out)
                    deepstack_capture_idx += 1

            main_out = vit.merger(y)

            if deepstack_outs:
                self.block_output[graph_key] = torch.cat(
                    [main_out] + deepstack_outs, dim=1
                )
            else:
                self.block_output[graph_key] = main_out

        self.block_graphs[graph_key] = graph

    def replay(
        self,
        graph_key: Hashable,
        x_3d: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        output_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = x_3d.shape[0]
        if position_embeddings is not None:
            head_dim = position_embeddings[0].shape[1]
            used_cos_ws, used_sin_ws = self._get_sin_cos_ws(
                graph_key, seq_len, head_dim
            )
            used_cos_ws.copy_(position_embeddings[0])
            used_sin_ws.copy_(position_embeddings[1])
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            head_dim = rotary_pos_emb_cos.shape[1]
            used_cos_ws, used_sin_ws = self._get_sin_cos_ws(
                graph_key, seq_len, head_dim
            )
            used_cos_ws.copy_(rotary_pos_emb_cos)
            used_sin_ws.copy_(rotary_pos_emb_sin)

        # copy input
        self.block_input[graph_key].copy_(x_3d)

        # replay
        self.block_graphs[graph_key].replay()

        out = self.block_output[graph_key]

        # Optional output reordering (Qwen2.5-VL window permutation inverse)
        if output_indices is not None:
            out = out.index_select(0, output_indices)

        # Cache entries and split views outlive this invocation. Graph replay
        # (including another graph using the shared pool) may overwrite out.
        return out.clone() if output_indices is None else out
