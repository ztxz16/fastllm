"""Inkling planning tests with header-only weights and synthetic device capacities."""
import io
import json
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
                     model_max_length=1048576, rel_extent=1024, n_layers_per_attn_res_block=6, unpadded_vocab_size=200058, sconv_kernel_size=4)}
        self.weights = [mem.Tensor("token_embd.weight", (201024, 2304), 492103616, "gguf:8")]
        for i in range(24):
            for suffix, shape in (("gate_up", (128, 2304, 2304)), ("down", (128, 2304, 1152))):
                n = shape[0] * shape[1] * shape[2]
                kind, size = ("gguf:40", n // 64 * 36) if i < 20 else ("gguf:8", n // 32 * 34)
                self.weights.append(mem.Tensor(f"blk.{i}.ffn_{suffix}_exps.weight", shape, size, kind))

    def plan(self, *options, n=2, free_gib=None, numa=True, compact=False):
        args = make_normal_parser("test").parse_args(["model.gguf", "--auto", *options])
        info = {i: {"capability": 120, "compact_nvfp4_moe": compact} for i in range(n)}
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
                                 plan["workspace_bytes_per_rank"] + plan["prefix_cache_bytes_per_rank"], plan["root_load_peak_bytes"]) + args.moe_cuda_cache, 10 * mem.GiB)

    def native_expert_weights(self, kinds):
        g = ink.geometry(self.meta)
        tensors = []
        for layer, kind in enumerate(kinds):
            for name, rows, cols in (("w13_weight", 2 * g.inter, g.hidden), ("w2_weight", g.hidden, g.inter)):
                width = cols // 2 if kind == "U8" else cols
                shape = (g.experts, rows, width)
                tensors.append(mem.Tensor(f"model.llm.layers.{layer}.mlp.experts.{name}", shape,
                                          g.experts * rows * width, kind))
        return [self.weights[0], *tensors]

    def test_native_expert_storage_and_repack_peak(self):
        g = ink.geometry(self.meta)
        tensors = self.native_expert_weights(["U8", "F8_E4M3", "F8_E4M3"])
        fp4_bytes = sum(t.bytes for t in tensors if t.kind == "U8")
        fp8_bytes = sum(t.bytes for t in tensors if t.kind == "F8_E4M3")
        w = ink.weight_sizes(tensors, g)
        self.assertEqual(w.compact_fp4_savings, fp4_bytes * 3 // 8)
        self.assertEqual(w.repack_source, fp8_bytes // 2)
        self.assertEqual(w.slot_bytes * g.experts, fp4_bytes * 3 // 2 + fp8_bytes * 9 // 8)
        aux = (2 * 16 + 2 * ((2 * g.inter + g.hidden) * 4 + 2 * 24)) * g.experts
        self.assertEqual(w.cuda_experts, w.slot_bytes * g.experts + aux)

        self.weights = tensors
        expanded = self.plan(n=1)[1]
        compact = self.plan(n=1, compact=True)[1]
        self.assertEqual(expanded["weight_bytes_per_rank"] - compact["weight_bytes_per_rank"],
                         w.compact_fp4_savings)
        self.assertEqual(compact["root_load_peak_bytes"] - compact["weight_bytes_per_rank"],
                         max(w.largest, w.repack_source) + 512 * mem.MiB)
        self.assertEqual(self.plan("--moe_device", "cpu", n=1)[1]["weight_bytes_per_rank"],
                         self.plan("--moe_device", "cpu", n=1, compact=True)[1]["weight_bytes_per_rank"])

    def test_recurrent_cuda_core(self):
        self.weights = self.native_expert_weights(["U8"] * 20 + ["F8_E4M3"] * 4)
        options = ("--auto_gpu_memory_limit", "18g")
        for numa, host in ((False, "cpu"), (True, "numa")):
            _, plan = self.plan(*options, n=1, compact=True, numa=numa)
            self.assertEqual(plan["gpu_expert_layers"], list(range(6, 18)))
            m = plan["recurrent_expert_placement"]
            # Reproduce native SelectDeviceFromMap's sorted contiguous ranges.
            def expand(device_map, count):
                if isinstance(device_map, str):
                    return [device_map] * count
                return [name for name, n in sorted(device_map.items()) for _ in range(n)]
            devices = expand(m["front"], 24 - m["tail_layers"]) + expand(m["tail"], m["tail_layers"])
            self.assertEqual(devices, [host] * 6 + ["cuda:0"] * 12 + [host] * 6)
            use = sum(plan[key] for key in ("weight_bytes_per_rank", "workspace_bytes_per_rank",
                                            "prefix_cache_bytes_per_rank", "kv_bytes_per_rank"))
            self.assertLessEqual(max(use, plan["root_load_peak_bytes"]) + plan["moe_cuda_cache"], 18 * mem.GiB)
        for flags in (("--moe_device", "cpu"), ("--moe_cuda_cache", "1g")):
            self.assertNotIn("gpu_expert_layers", self.plan(*options, *flags, n=1, compact=True)[1])
        self.assertNotIn("gpu_expert_layers", self.plan(*options, n=1, compact=False)[1])
        self.assertNotIn("gpu_expert_layers", self.plan(n=1, compact=True)[1])

    def test_hybrid_workspace_covers_grouped_prefill(self):
        g = ink.geometry(self.meta)
        def workspace(rows, native):
            return ink._workspace_bytes(g, rows, 1, 1, True, 0, native_prefill=native)
        self.assertEqual(workspace(511, True), workspace(511, False))
        extra = workspace(8192, True) - workspace(8192, False)
        self.assertGreater(extra, 256 * mem.MiB)
        self.assertEqual(extra, workspace(16384, True) - workspace(16384, False))

    def test_prefix_pool_budget(self):
        options = ("--max_batch", "2", "--max_context_length", "8192")
        for ranks in (1, 2):
            self.assertEqual(self.plan(*options, n=ranks)[1]["prefix_cache_bytes_per_rank"], mem.GiB // ranks)
        for flag in ("--cache_history", "--prefix_cache"):
            self.assertEqual(self.plan(*options, flag, "false")[1]["prefix_cache_bytes_per_rank"], 0)
        self.assertEqual(self.plan(*options, "--kvmem", '{"resident_tokens":1024}')[1]["prefix_cache_bytes_per_rank"], 0)

    def test_fp8_target_kv_budget(self):
        options = ("--max_context_length", "8192", "--max_batch", "2")
        def kv(dtype, *extra):
            return self.plan(*options, "--kv_cache_dtype", dtype, *extra, n=1)[1]["kv_bytes_per_rank"]
        half = kv("float16")
        for dtype in ("fp8", "float8", "fp8_e4m3"):
            self.assertEqual(2 * kv(dtype), half)
        draft = dict(num_hidden_layers=3, local_layer_ids=[0, 2], num_attention_heads=24,
                     num_key_value_heads=8, head_dim=128, intermediate_size=9216,
                     sliding_window_size=1024, block_size=8, target_layer_ids=[5, 23, 41, 48],
                     max_position_embeddings=1536)
        mapped = (*options, "--kv_cache_dtype", "fp8_e4m3", "--kvmem", '{"resident_tokens":2048}')
        _, target_only = self.plan(*mapped)
        with patch.object(ink, "_draft_budget", return_value=(draft, 0)):
            half = kv("float16", "--dspark", "7")
            fp8 = kv("fp8_e4m3", "--dspark", "7")
            _, with_draft = self.plan(*mapped, "--dspark", "7")
        # The global draft backs 13 pages, each draft window backs 10. A target
        # window needs one more page than without speculation for tentative rows.
        pair_bytes = 2 * 2 * 8 * 128
        self.assertEqual(with_draft["kvmem_host_bytes"] - target_only["kvmem_host_bytes"],
                         2 * (13 + 2 * 10) * 128 * pair_bytes + 2 * 40 * 128 * 2048)
        self.assertEqual(with_draft["kvmem_hot_bytes_per_rank"] - target_only["kvmem_hot_bytes_per_rank"],
                         (2 * 1024 + 13 * 128) * pair_bytes)
        self.assertEqual(with_draft["kvmem_page_table_bytes_per_rank"] - target_only["kvmem_page_table_bytes_per_rank"],
                         2 * 3 * 13 * 8)
        draft_bytes = ink._kv_bytes(16384, 2, 128, 8, 128, 2, 1, 2, 1032, 7)
        self.assertEqual(2 * fp8 - half, draft_bytes)

    def test_private_kvmem_budgets(self):
        options = ("--kvmem", '{"max_tokens":2048,"resident_tokens":128,"prefill_tokens":256,"host_mib":2048}')
        for dtype, unit in (("float16", 2), ("bfloat16", 2), ("float32", 4), ("fp8_e4m3", 1)):
            row = 2 * unit * 8 * 128 * 48
            for ranks, batch in ((1, 1), (1, 2), (2, 2)):
                args, plan = self.plan(*options, "--kv_cache_dtype", dtype, "--max_batch", str(batch), n=ranks)
                self.assertEqual((args.max_context_length, args.chunked_prefill_size), (2048, 256))
                self.assertEqual(plan["kvmem_host_bytes"], batch * (8 * 2048 + 40 * 1152) * (row // 48))
                self.assertEqual(plan["kvmem_hot_bytes_per_rank"], batch * 128 * row // ranks)
                index = batch * 8 * (152 + 16 * 4 * ((8 * 128 + 24) // ranks + 2) + 16)
                self.assertEqual(plan["kvmem_selection_bytes_per_rank"], index)
                self.assertEqual(plan["kv_bytes_per_rank"], plan["kvmem_hot_bytes_per_rank"] + batch * 48 * 16 * 8 + index)
        self.assertEqual(self.plan(*options, "--moe_device", "numa")[1]["mode"], "hybrid")
        self.assertEqual(self.plan(*options, "--chunked_prefill_size", "1024")[0].chunked_prefill_size, 256)
        _, large = self.plan("--kvmem", '{"max_tokens":1048576,"resident_tokens":32768,"host_mib":100352}',
                             "--kv_cache_dtype", "fp8_e4m3", "--max_batch", "1", n=1)
        self.assertEqual(large["kvmem_hot_bytes_per_rank"], 592 * 1024**2)
        args, _ = self.plan("--kvmem", '{"resident_tokens":128}', "--max_context_length", "4096")
        self.assertEqual(json.loads(args.kvmem)["max_tokens"], 4096)
        args, _ = self.plan("--kvmem", '{"resident_tokens":128}', "--chunked_prefill_size", "1024")
        self.assertEqual(args.chunked_prefill_size, 1024)
        with self.assertRaisesRegex(ValueError, "host KV"):
            self.plan("--kvmem", '{"max_tokens":2048,"resident_tokens":128,"host_mib":1}')

    def test_sliding_host_retention(self):
        def plan(retention, host_mib=100352):
            cfg = dict(max_tokens=1048576, resident_tokens=2048, host_mib=host_mib, prefill_tokens=2048)
            if retention is not None:
                cfg["sliding_host_tokens"] = retention
            return self.plan("--kv_cache_dtype", "fp8_e4m3", "--kvmem", json.dumps(cfg), n=1)[1]
        small = plan(None)
        for retention in (0, 256, 1024):
            self.assertEqual(plan(retention)["kvmem_host_bytes"], small["kvmem_host_bytes"])
        for retained, backing in ((None, 1152), (4097, 4224), (1048576, 1048576)):
            result = plan(retained)
            self.assertEqual(result["kvmem_host_bytes"], 2048 * (8 * 1048576 + 40 * backing))
            self.assertEqual(result["kv_bytes_per_rank"], small["kv_bytes_per_rank"])
        budget = small["kvmem_host_bytes"] // 1024**2
        self.assertEqual(plan(None, budget)["kvmem_host_bytes"], small["kvmem_host_bytes"])
        with self.assertRaisesRegex(ValueError, "host KV"):
            plan(None, budget - 1)
        with self.assertRaisesRegex(ValueError, "host KV"):
            plan(1048576, budget)
        for invalid in (-1, True, 2147483648):
            with self.assertRaisesRegex(ValueError, "sliding_host_tokens"):
                plan(invalid)

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

    def test_speculative_pools_and_graph_workspace(self):
        # A partial last page on each request may be filled by the draft before commit.
        self.assertEqual(ink._kv_bytes(1024, 2, 64, 1, 1, 2, 1, 1, 65, 7),
                         4 * (1024 + 2 * 64 + 2 * 2 * 64))
        g = ink.geometry(self.meta)
        def overhead(batch):
            rows = batch * 8
            return (ink._workspace_bytes(g, 1024, rows, 1, False, 0, 7) -
                    ink._workspace_bytes(g, 1024, rows, 1, False, 0))
        self.assertEqual(overhead(48), 2 * overhead(24))
        self.assertGreater(overhead(48), 4 * 48 * 7 * g.vocab)

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
