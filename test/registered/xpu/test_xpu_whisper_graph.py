"""XPU Decode Graph integration test for encoder-decoder models (Whisper).

Verifies that:
1. Whisper encoder-decoder model successfully initializes and captures XPU decode graph.
2. Decode XPU graph successfully replays across exact (bs=2) and padded (bs=1 -> bs=2)
   workloads without fallback to eager.
3. Decoded token sequences with XPU decode graph match eager execution.

Usage:
    python3 -m unittest test_xpu_whisper_graph.TestXPUWhisperGraph.test_encoder_decoder_graph_runs
"""

import unittest
from array import array

import torch

from sglang.benchmark.one_batch import (
    _init_process_global_configs,
    load_model,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.runtime_context import SpawnRanks, publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")


class TestXPUWhisperGraph(CustomTestCase):
    """Whisper Decode Full Graph."""

    def test_encoder_decoder_graph_runs(self):
        torch.manual_seed(42)
        server_args = ServerArgs(
            model_path="openai/whisper-tiny",
            device="xpu",
            attention_backend="intel_xpu",
            mem_fraction_static=0.4,
            cuda_graph_config={"decode": {"backend": "full", "bs": [1, 2]}},
        )
        server_args.resolve_once()
        _set_envs_and_config(server_args)
        port_args = PortArgs.init_new(server_args)
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(world_rank=0, gpu_id=0),
        )
        _init_process_global_configs()

        bench_runner, tokenizer = load_model(
            server_args, port_args, gpu_id=0, tp_rank=0
        )
        torch_runner = bench_runner.torch_runner
        self.assertIsNotNone(torch_runner.decode_cuda_graph_runner)

        mel_features = torch.randn(80, 3000, dtype=torch.float32)
        prompt_ids = [50258, 50259, 50359, 50363]
        padded_ids = array("q", [0] * 1500 + prompt_ids)
        sampling_params = SamplingParams(temperature=0, max_new_tokens=8)

        def make_reqs(batch_size: int):
            reqs = []
            for i in range(batch_size):
                item = MultimodalDataItem(
                    modality=Modality.AUDIO, feature=mel_features.clone()
                )
                mm = MultimodalInputs(mm_items=[item])
                mm.num_image_tokens = 1500
                r = Req(
                    rid=i,
                    origin_input_text="",
                    origin_input_ids=array("q", padded_ids),
                    sampling_params=sampling_params,
                    origin_input_ids_unpadded=array("q", prompt_ids),
                )
                r.multimodal_inputs = mm
                r.full_untruncated_fill_ids = r.origin_input_ids
                r.logprob_start_len = -1
                r.extend_end = len(r.origin_input_ids)
                reqs.append(r)
            return reqs

        def run_enc_dec_pass(batch_size: int, use_graph: bool):
            bench_runner.clear()
            reqs = make_reqs(batch_size)
            saved_decode = torch_runner.decode_cuda_graph_runner
            decode_replays = []
            orig_decode_replay = saved_decode.backend.replay

            def wrapped_decode_replay(shape_key, *args, **kwargs):
                decode_replays.append(shape_key)
                return orig_decode_replay(shape_key, *args, **kwargs)

            try:
                if not use_graph:
                    torch_runner.decode_cuda_graph_runner = None
                else:
                    saved_decode.backend.replay = wrapped_decode_replay

                next_token_ids, _, batch = bench_runner.extend(reqs)
                output_ids = [[int(t)] for t in next_token_ids]
                for _ in range(4):
                    next_token_ids, _ = bench_runner.decode(next_token_ids, batch)
                    for i in range(batch_size):
                        output_ids[i].append(int(next_token_ids[i].item()))
                return output_ids, decode_replays
            finally:
                saved_decode.backend.replay = orig_decode_replay
                torch_runner.decode_cuda_graph_runner = saved_decode

        # Case 1: batch_size=2
        graph_ids_bs2, replays_bs2 = run_enc_dec_pass(batch_size=2, use_graph=True)
        eager_ids_bs2, eager_replays_bs2 = run_enc_dec_pass(
            batch_size=2, use_graph=False
        )
        self.assertEqual([k.size for k in replays_bs2], [2] * 4)
        self.assertEqual(eager_replays_bs2, [])
        self.assertEqual(graph_ids_bs2, eager_ids_bs2)

        # Case 2: batch_size=1
        graph_ids_bs1, replays_bs1 = run_enc_dec_pass(batch_size=1, use_graph=True)
        eager_ids_bs1, eager_replays_bs1 = run_enc_dec_pass(
            batch_size=1, use_graph=False
        )
        self.assertEqual([k.size for k in replays_bs1], [1] * 4)
        self.assertEqual(eager_replays_bs1, [])
        self.assertEqual(graph_ids_bs1, eager_ids_bs1)


if __name__ == "__main__":
    unittest.main()
