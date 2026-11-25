from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.xpu.graph_runner.vit_xpu_graph_runner import (
    ViTXpuGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Block:
    attn = SimpleNamespace(
        qkv_backend_name="xpu_attn",
        num_attention_heads_per_partition=2,
        head_size=4,
    )

    def forward(self, x, cu_seqlens=None, output_ws=None):
        return x


def test_xpu_vit_graph_keys_and_buffer_ownership():
    """Distinct layouts get distinct graphs; capture/replay preserve caller data."""
    runner = ViTXpuGraphRunner(
        SimpleNamespace(
            blocks=[_Block()],
            device=torch.device("cpu"),
            dtype=torch.float32,
            use_data_parallel=True,
        )
    )

    def capture(graph_key, **kwargs):
        # Simulate in-place capture and a static output reused by replay.
        runner.block_input[graph_key].add_(10)
        runner.block_output[graph_key] = runner.block_input[graph_key].clone()
        runner.block_graphs[graph_key] = SimpleNamespace(
            replay=lambda: runner.block_output[graph_key].copy_(
                runner.block_input[graph_key] + 10
            )
        )

    runner._create_graph = capture
    x = torch.ones(8, 8)
    rotary = torch.zeros(8, 4)
    first_layout = torch.tensor([0, 4, 8], dtype=torch.int32)
    second_layout = torch.tensor([0, 2, 8], dtype=torch.int32)

    with patch.object(ViTXpuGraphRunner, "_graph_memory_pool", object()):
        out = runner.run(
            x=x,
            cu_seqlens=first_layout,
            cu_window_seqlens=None,
            position_embeddings=None,
            rotary_pos_emb_cos=rotary,
            rotary_pos_emb_sin=rotary,
        )
        retained = out[:4]
        torch.testing.assert_close(x, torch.ones_like(x))
        torch.testing.assert_close(out, torch.full_like(out, 11))
        later = runner.run(
            x=torch.full_like(x, 2),
            cu_seqlens=first_layout,
            cu_window_seqlens=None,
            position_embeddings=None,
            rotary_pos_emb_cos=rotary,
            rotary_pos_emb_sin=rotary,
        )
        runner.run(
            x=torch.full_like(x, 3),
            cu_seqlens=second_layout,
            cu_window_seqlens=None,
            position_embeddings=None,
            rotary_pos_emb_cos=rotary,
            rotary_pos_emb_sin=rotary,
        )

    torch.testing.assert_close(retained, torch.full_like(retained, 11))
    torch.testing.assert_close(later, torch.full_like(later, 12))
    assert len(runner.block_graphs) == 2
    assert {key[1][0] for key in runner.block_graphs} == {(0, 4, 8), (0, 2, 8)}
    assert all(ws.shape[0] == x.shape[0] for ws in runner.block_ws.values())
