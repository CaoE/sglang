import unittest
from types import SimpleNamespace

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)

from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.hardware_backend.xpu.graph_runner.vit_xpu_graph_runner import (
    ViTXpuGraphRunner,
)
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=15, suite="extra-test-1-gpu-xpu")


class TestXPUViTGraph(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise unittest.SkipTest("XPU not available")

        server_args = ServerArgs(
            model_path="openai/whisper-tiny",
            device="xpu",
            attention_backend="intel_xpu",
        )
        server_args.resolve_once()
        _set_envs_and_config(server_args)
        publish(server_args, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))

        dummy_group = SimpleNamespace(
            world_size=1, rank_in_group=0, rank=0, ca_comm=None, device_group=None
        )
        get_parallel().override_permanently(
            tp_group=dummy_group, attn_tp_group=dummy_group
        )

    def test_qwen2_5_vit_xpu_graph_capture_and_replay(self):
        config = Qwen2_5_VLVisionConfig(
            depth=2,
            embed_dim=128,
            hidden_size=256,
            num_heads=4,
            mlp_ratio=2.0,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            window_size=112,
            fullatt_block_indexes=[1],
        )
        vit = Qwen2_5_VisionTransformer(config).to(device="xpu", dtype=torch.bfloat16)
        runner = ViTXpuGraphRunner(vit)

        x1 = torch.randn(64, 3 * 2 * 14 * 14, device="xpu", dtype=torch.bfloat16)
        grid_thw1 = torch.tensor([[1, 8, 8]], device="xpu", dtype=torch.int32)

        # 1. Eager baseline
        vit.enable_cg = False
        eager_out1 = vit(x1, grid_thw1)

        # 2. XPU Graph capture
        vit.enable_cg = True
        vit.cuda_graph_runner = runner
        graph_out1 = vit(x1, grid_thw1)

        self.assertEqual(graph_out1.shape, (16, 3584))
        torch.testing.assert_close(graph_out1, eager_out1, rtol=1e-2, atol=1e-2)
        self.assertEqual(len(runner.block_graphs), 1)

        # 3. XPU Graph replay (same layout)
        replay_out1 = vit(x1, grid_thw1)
        torch.testing.assert_close(replay_out1, eager_out1, rtol=1e-2, atol=1e-2)
        self.assertEqual(len(runner.block_graphs), 1)

        # 4. Another shape creates a 2nd graph
        x2 = torch.randn(128, 3 * 2 * 14 * 14, device="xpu", dtype=torch.bfloat16)
        grid_thw2 = torch.tensor([[1, 16, 8]], device="xpu", dtype=torch.int32)

        vit.enable_cg = False
        eager_out2 = vit(x2, grid_thw2)

        vit.enable_cg = True
        graph_out2 = vit(x2, grid_thw2)

        self.assertEqual(graph_out2.shape, (32, 3584))
        torch.testing.assert_close(graph_out2, eager_out2, rtol=1e-2, atol=1e-2)
        self.assertEqual(len(runner.block_graphs), 2)


if __name__ == "__main__":
    unittest.main()
