"""Inkling planning tests with header-only weights and synthetic device capacities."""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import inkling_config as ink, hybrid_auto as mem
from fastllm_pytools.util import make_normal_parser


class InklingAutoTest(unittest.TestCase):
    def setUp(self):
        self.meta = {"text_config": dict(hidden_size=2304, intermediate_size=1152, n_routed_experts=128,
                     num_experts_per_tok=8, num_attention_heads=24, num_key_value_heads=8, head_dim=128,
                     num_hidden_layers=24, loop_config=dict(core_start=6, core_end=18, num_passes=3),
                     local_layer_ids=[i for i in range(24) if i % 6 != 5], sliding_window_size=1024,
                     model_max_length=1048576, rel_extent=1024, n_layers_per_attn_res_block=6, unpadded_vocab_size=200058)}
        self.weights = [mem.Tensor("token_embd.weight", (201024, 2304), 492103616, "gguf:8")]
        for i in range(24):
            for suffix, shape in (("gate_up", (128, 2304, 2304)), ("down", (128, 2304, 1152))):
                n = shape[0] * shape[1] * shape[2]
                kind, size = ("gguf:40", n // 64 * 36) if i < 20 else ("gguf:8", n // 32 * 34)
                self.weights.append(mem.Tensor(f"blk.{i}.ffn_{suffix}_exps.weight", shape, size, kind))

    def plan(self, *options, n=2, free_gib=None, numa=True):
        args = make_normal_parser("test").parse_args(["model.gguf", "--auto", *options])
        info = {i: {"capability": 120} for i in range(n)}
        with patch.dict(os.environ, {}, clear=True), patch.object(mem, "inventory", return_value=(self.meta, self.weights)), redirect_stdout(io.StringIO()):
            result = ink.resolve(args, lambda d: dict(total=72*mem.GiB, free=(free_gib[d] if free_gib else 71)*mem.GiB),
                                 (info, {}), dict(total=128*mem.GiB, available=125*mem.GiB), numa)
        return args, result

    def test_topologies(self):
        self.assertEqual(self.plan()[0].tp, "cuda:0,1")
        self.assertEqual(self.plan(n=1)[1]["mode"], "cuda")
        self.assertEqual(self.plan(n=3)[0].tp, "cuda:0,1")
        self.assertEqual(self.plan(n=8)[0].tp, "cuda:0,1,2,3,4,5,6,7")
        self.assertEqual(self.plan(n=3, free_gib=[2, 71, 70])[0].tp, "cuda:1,2")
        args, plan = self.plan(free_gib=[2, 71])
        self.assertEqual((args.device, plan["mode"]), ("cuda:1", "cuda"))
        self.assertEqual(self.plan("--auto_gpu_memory_limit", "10g", numa=False)[0].moe_device, "cpu")
        self.assertEqual(self.plan("--tp", "cuda:1,0")[0].device, "cuda:1")
        with self.assertRaises(ValueError):
            self.plan("--tp", "cuda:0,1,2", n=3)
        with self.assertRaises(ValueError):
            self.plan("--moe_device", "numa", "--tp", "cuda:0,1")

    def test_cache_uses_physical_layers_and_kv_uses_visits(self):
        g = ink.geometry(self.meta)
        self.assertEqual((g.local, g.global_), (40, 8))
        w = ink.weight_sizes(self.weights, g)
        def record(rows, cols):
            raw = rows * cols // 32 * 34
            return (raw + 4 + 255) // 256 * 256
        stride = 20 * (3 * 2304 * 1152 // 16 * 12) + 4 * (record(2304, 2304) + record(2304, 1152))
        self.assertEqual(w.slot_bytes, stride)
        args, plan = self.plan("--auto_gpu_memory_limit", "10g")
        self.assertEqual((args.tp, args.moe_device), ("off", "numa"))
        self.assertGreater(plan["cache_slots_per_layer"], 0)
        self.assertEqual(args.moe_cuda_cache, plan["cache_slots_per_layer"] * stride)
        self.assertLessEqual(max(plan["weight_bytes_per_rank"] + plan["kv_bytes_per_rank"] +
                                 plan["workspace_bytes_per_rank"], plan["root_load_peak_bytes"]) + args.moe_cuda_cache, 10 * mem.GiB)

    def test_explicit_settings_and_failure(self):
        args, plan = self.plan("--device", "cuda:1", "--moe_device", "cpu", "--moe_cuda_cache", "2g",
                               "--max_batch", "2", "--tokens", "32768", "--max_context_length", "8192",
                               "--chunked_prefill_size", "1024")
        self.assertEqual((args.device, args.tp, args.moe_device, args.moe_cuda_cache), ("cuda:1", "off", "cpu", 2 * mem.GiB))
        self.assertEqual((args.max_batch, args.tokens, args.max_context_length, args.chunked_prefill_size), (2, 32768, 8192, 1024))
        args, _ = self.plan("--max_context_length", "4096", "--chunked_prefill_size", "8192")
        self.assertEqual(args.chunked_prefill_size, 8192)
        with self.assertRaisesRegex(ValueError, "no feasible layout"):
            self.plan("--auto_gpu_memory_limit", "1g")
        with self.assertRaisesRegex(ValueError, "host needs"):
            self.plan("--moe_device", "numa", "--auto_host_memory_limit", "1g")
        with self.assertRaisesRegex(ValueError, "--kv_cache_limit"):
            self.plan("--kv_cache_limit", "1g")

    def test_external_draft_budget(self):
        args = make_normal_parser("test").parse_args(["target.gguf", "--auto", "--dspark", "7",
                                                      "--speculative_draft_model_path", "draft.gguf"])
        target = {"general.architecture": "inkling2_loop"}
        md = {"general.architecture": "dflash", "dflash.attention.sliding_window_pattern": [True, False, True],
              "dflash.block_count": 3, "dflash.attention.head_count": 24, "dflash.attention.head_count_kv": 8,
              "dflash.attention.key_length": 128, "dflash.attention.value_length": 128,
              "dflash.feed_forward_length": 9216, "dflash.block_size": 8, "dflash.target_layers": [5, 23, 41, 48],
              "dflash.attention.sliding_window": 1025}
        weights = [mem.Tensor("joint_proj.weight", (8, 16), 136, "gguf:8"),
                   mem.Tensor("mask_embd.weight", (7, 8), 224, "gguf:0")]
        with patch.object(mem, "inventory", return_value=(md, weights)):
            draft, size = ink._draft_budget(args, target)
            self.assertEqual((size, draft["local_layer_ids"], draft["sliding_window_size"]), (480, [0, 2], 1024))
            args.dspark = 8
            with self.assertRaisesRegex(ValueError, "proposal count"):
                ink._draft_budget(args, target)


if __name__ == "__main__":
    unittest.main()
