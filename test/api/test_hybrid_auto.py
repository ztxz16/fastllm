"""Capacity invariants and metadata-only planning; no CUDA is required."""
import copy
import io
import types
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import hybrid_auto as auto
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model

GiB, MiB = auto.GiB, auto.MiB


def args(**changes):
    a = make_normal_parser("test").parse_args([])
    a.device, a.moe_device, a.tokens, a.max_batch = "cuda:0", "numa", 4096, 1
    a.chunked_prefill_size = 512
    a.moe_cuda_cache = "auto"
    vars(a).update(changes)
    return a


def fixture():
    geometry = auto.Geometry("fixture", 4, 65536, 128, 8, 2, 64,
                             [1024] * 4, [MiB] * 4, 4, 1024, 1024)
    costs = ([GiB] * 4, [MiB] * 4, MiB, MiB, GiB, False)
    gpu = {0: {"total": 8 * GiB, "free": 8 * GiB},
           1: {"total": 8 * GiB, "free": 8 * GiB}}
    ram = {"total": 32 * GiB, "available": 32 * GiB}
    return geometry, costs, gpu, ram


def write_gguf(path, metadata, tensors):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for k, v in metadata.items():
        header += string(k) + struct.pack("<I", 8 if isinstance(v, str) else 4)
        header += string(v) if isinstance(v, str) else struct.pack("<I", v)
    offset = 0
    for name, shape, size in tensors:
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", 1, offset)
        offset += size
    header += b"\0" * (-len(header) % 32)
    path.write_bytes(header + b"\0" * offset)


class HybridAutoTest(unittest.TestCase):
    def test_cli_opt_in_and_existing_defaults(self):
        defaults = make_normal_parser("test").parse_args([])
        self.assertFalse(auto.requested(defaults))
        self.assertEqual((defaults.moe_cuda_cache, defaults.moe_cpu_cache,
                          defaults.chunked_prefill_size, defaults.moe_device_layers), (0, 0, -1, -1))
        a = make_normal_parser("test").parse_args(sum((["--" + k, "auto"] for k in auto.AUTO_OPTIONS), []))
        self.assertTrue(auto.requested(a))
        self.assertTrue(all(getattr(a, k) == "auto" for k in auto.AUTO_OPTIONS))

    def test_resolved_parameters_reach_runtime_before_loading(self):
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            a = make_normal_parser("test").parse_args([
                path, "--device", "cuda", "--moe_device", "numa", "--threads", "1",
                "--moe_cuda_cache", "auto", "--moe_device_layers", "auto",
                "--chunked_prefill_size", "auto"])
            runtime = MagicMock()
            model = runtime.model.return_value
            model.get_max_input_len.return_value = 16384
            model.get_max_batch.return_value = 1
            def resolve(a, query):
                a.moe_cuda_cache = GiB
                a.moe_device_layers = -1
                a.chunked_prefill_size = 1024
                return {"moe_cuda_cache": GiB}
            def load(*unused, **kw):
                runtime.set_moe_cuda_cache.assert_called_once_with(GiB)
                self.assertEqual(a.chunked_prefill_size, 1024)
                return model
            runtime.model.side_effect = load
            with patch.object(auto, "resolve", side_effect=resolve), \
                    patch.dict(os.environ, {}, clear=True), \
                    patch.dict(sys.modules, {"ftllm": types.SimpleNamespace(llm=runtime)}), \
                    redirect_stdout(io.StringIO()):
                self.assertIs(make_normal_llm_model(a), model)
            model.set_chunked_prefill_size.assert_called_once_with(1024)
            self.assertEqual(model.hybrid_auto_plan, {"moe_cuda_cache": GiB})

    def test_cpu_and_gpu_paths_reject_auto_before_hardware_query(self):
        for main, moe, mode in [("cpu", "numa", "auto"), ("cuda:0", "cuda:0", "auto"),
                                ("cuda:0", "numa", "gpu")]:
            a = args(device=main, moe_device=moe)
            before = vars(a).copy()
            with patch.dict(os.environ, {"FASTLLM_EXECUTION_MODE": mode}, clear=True), \
                    self.assertRaises(ValueError):
                auto.resolve(a, lambda _: self.fail("unexpected CUDA access"))
            self.assertEqual(vars(a), before)

    def test_gguf_all_shards_and_missing_shard(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for i in range(2):
                write_gguf(p / ("tiny-%05d-of-00002.gguf" % (i + 1)),
                           {"general.architecture": "qwen4exp", "split.no": i, "split.count": 2,
                            "split.tensors.count": 2}, [("blk.%d.ffn_gate_exps.weight" % i, (8, 4, 2), 128)])
            _, tensors = auto.inventory(p / "tiny-00002-of-00002.gguf")
            self.assertEqual([t.bytes for t in tensors], [128, 128])
            self.assertEqual(tensors[0].shape, (2, 4, 8))
            (p / "tiny-00001-of-00002.gguf").unlink()
            with self.assertRaises(FileNotFoundError):
                auto.inventory(p / "tiny-00002-of-00002.gguf")

    def test_safetensors_validates_index_and_extents(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "config.json").write_text('{}')
            header = json.dumps({"x": {"dtype": "F16", "shape": [8], "data_offsets": [0, 16]}}).encode()
            (p / "model.safetensors").write_bytes(struct.pack('<Q', len(header)) + header + bytes(16))
            _, tensors = auto.inventory(p)
            self.assertEqual(tensors[0].bytes, 16)
            (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"missing": "model.safetensors"}}))
            with self.assertRaises(ValueError):
                auto.inventory(p)

    def test_ngram_and_inactive_draft_do_not_become_gpu_weights(self):
        g, _, _, _ = fixture()
        a = args(ngram_device="disk")
        tensors = [auto.Tensor("per_layer_token_embd.weight", (8, 8), 64 * GiB, "gguf:1"),
                   auto.Tensor("token_embd.weight", (8, 8), MiB, "gguf:1"),
                   auto.Tensor("blk.0.ffn_down_exps.weight", (8, 8, 8), GiB, "gguf:1"),
                   auto.Tensor("blk.4.ffn_down_exps.weight", (8, 8, 8), GiB, "gguf:1")]
        e, dense, common, host, largest, snapshot = auto.weight_costs(tensors, g, a)
        self.assertEqual((sum(e), sum(dense), common, host, largest), (GiB, 0, 0, MiB, GiB))
        a.ngram_device = "cpu"
        self.assertEqual(auto.weight_costs(tensors, g, a)[3], 64 * GiB + MiB)

    def test_hf_weight_conversion_counts_unpacked_values(self):
        g, _, _, _ = fixture()
        a = args(dtype="float16", moe_cuda_cache=0)
        t = [auto.Tensor("model.layers.0.mlp.experts.0.up_proj.weight", (8, 8), 64, "F8_E4M3"),
             auto.Tensor("model.layers.0.mlp.experts.0.down_proj.weight", (8, 4), 32, "U8")]
        self.assertEqual(auto.weight_costs(t, g, a)[0][0], 256)
        a.moe_cuda_cache = "auto"
        with self.assertRaises(ValueError):
            auto.weight_costs(t, g, a)

    def test_hf_integer_auxiliary_buffers_are_not_packed_weights(self):
        g, _, _, _ = fixture()
        t = [auto.Tensor("model.layers.0.mlp.experts.0.down_proj.weight", (8, 4), 32, "U8"),
             auto.Tensor("model.layers.1.ple.ple_embedding.layer_multipliers", (3,), 24, "I64")]
        self.assertEqual(auto.weight_costs(t, g, args())[1][1], 24)
        t.append(auto.Tensor("model.layers.0.mlp.experts.0.up_proj.weight", (8, 4), 128, "I32"))
        with self.assertRaises(ValueError):
            auto.weight_costs(t, g, args())

    def test_snapshot_fallback_counts_a_second_host_copy(self):
        g, c, gpu, ram = fixture()
        p = auto.plan(args(), g, c, "pp", {0: 1}, gpu, ram)
        c = (*c[:-1], True)
        q = auto.plan(args(), g, c, "pp", {0: 1}, gpu, ram)
        self.assertEqual(q["host_estimated_bytes"] - p["host_estimated_bytes"], 4 * GiB)

    def test_cgroup_parent_and_current_usage(self):
        for v2 in (True, False):
            with self.subTest(v2=v2), tempfile.TemporaryDirectory() as d:
                p = Path(d)
                proc, mount = p / "proc", p / "cgroup"
                (proc / "self").mkdir(parents=True)
                child = mount / "job" / "child"
                child.mkdir(parents=True)
                (proc / "meminfo").write_text("MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\n")
                (proc / "self/cgroup").write_text("0::/job/child\n" if v2 else "4:memory:/job/child\n")
                fs = "cgroup2 cgroup rw" if v2 else "cgroup memory rw,memory"
                (proc / "self/mountinfo").write_text("1 0 0:1 / %s rw - %s\n" % (mount, fs))
                limit, current = ("memory.max", "memory.current") if v2 else ("memory.limit_in_bytes", "memory.usage_in_bytes")
                (child / limit).write_text("max" if v2 else str(1 << 63))
                (child / current).write_text(str(GiB))
                (child.parent / limit).write_text(str(8 * GiB))
                (child.parent / current).write_text(str(3 * GiB))
                self.assertEqual(auto.host_memory(proc), {"total": 8 * GiB, "available": 5 * GiB})

    def test_available_gpu_memory_is_reserved_below_ratio(self):
        g, c, gpu, ram = fixture()
        gpu[0]["free"] = 7 * GiB
        a = args(gpu_mem_ratio=.75)
        p = auto.plan(a, g, c, "pp", {0: 1}, gpu, ram)
        self.assertEqual(p["gpu_budget_bytes"][0], 5 * GiB)  # 7 free - 2 reserve
        self.assertLessEqual(p["gpu_estimated_bytes"][0], 5 * GiB)
        self.assertEqual(p["resident_layers"], 0)
        self.assertEqual((a.tokens, a.chunked_prefill_size), (4096, 512))

    def test_more_kv_or_batch_cannot_increase_auto_cache(self):
        g, c, gpu, ram = fixture()
        base = auto.plan(args(), g, c, "pp", {0: 1}, gpu, ram)
        for changes in ({"tokens": 32768}, {"max_batch": 8}, {"mtp": 4}, {"moe_experts": 8}):
            p = auto.plan(args(**changes), g, c, "pp", {0: 1}, gpu, ram)
            self.assertLessEqual(p["moe_cuda_cache"], base["moe_cuda_cache"])
            self.assertGreater(p["workspace_bytes_per_device"], base["workspace_bytes_per_device"])

    def test_tp_shards_experts_but_does_not_shard_host_allocation(self):
        g, c, gpu, ram = fixture()
        a = args(moe_device_layers=2, moe_cuda_cache=0)
        p = auto.plan(a, g, c, "tp", {0: 1, 1: 1}, gpu, ram)
        self.assertEqual(p["gpu_estimated_bytes"][0], p["gpu_estimated_bytes"][1])
        q = auto.plan(args(moe_cuda_cache=0), g, c, "tp", {0: 1, 1: 1}, gpu, ram)
        self.assertEqual(q["host_estimated_bytes"] - p["host_estimated_bytes"], 2 * GiB)
        self.assertEqual(p["gpu_estimated_bytes"][0] - q["gpu_estimated_bytes"][0], GiB)

    def test_pp_resident_prefix_uses_its_own_device_partition(self):
        g, c, gpu, ram = fixture()
        a = args(moe_device_layers=2, moe_cuda_cache=0)
        p = auto.plan(a, g, c, "pp", {0: 1, 1: 1}, gpu, ram)
        q = auto.plan(args(moe_cuda_cache=0), g, c, "pp", {0: 1, 1: 1}, gpu, ram)
        for d in gpu:
            self.assertEqual(p["gpu_estimated_bytes"][d] - q["gpu_estimated_bytes"][d], GiB)

    def test_joint_plan_fits_host_limit_without_changing_explicit_chunk(self):
        g, c, gpu, ram = fixture()
        # 4 GiB of experts cannot fit alongside the load group + runtime.
        ram = {"total": 6 * GiB, "available": 6 * GiB}
        a = args(moe_device_layers="auto", chunked_prefill_size=512)
        p = auto.plan(a, g, c, "tp", {0: 1, 1: 1}, gpu, ram)
        self.assertGreater(p["resident_layers"], 0)
        self.assertEqual(p["chunked_prefill_size"], 512)
        self.assertLessEqual(p["host_estimated_bytes"], p["host_budget_bytes"])
        for device in gpu:
            self.assertLessEqual(p["gpu_estimated_bytes"][device], p["gpu_budget_bytes"][device])

    def test_explicit_cache_failure_does_not_reduce_context(self):
        g, c, gpu, ram = fixture()
        a = args(moe_cuda_cache=8 * GiB, chunked_prefill_size="auto")
        before = copy.deepcopy(vars(a))
        with self.assertRaises(ValueError):
            auto.plan(a, g, c, "pp", {0: 1}, gpu, ram)
        self.assertEqual(vars(a), before)

    def test_disk_host_cache_respects_effective_ram(self):
        g, c, gpu, _ = fixture()
        ram = {"total": 6 * GiB, "available": 6 * GiB}
        p = auto.plan(args(moe_device="disk", moe_cpu_cache="auto"), g, c, "pp", {0: 1}, gpu, ram)
        self.assertGreater(p["moe_cpu_cache"], 0)
        self.assertLess(p["moe_cpu_cache"], 4 * GiB)
        self.assertLessEqual(p["host_estimated_bytes"], 5 * GiB)

    def test_qwen_gguf_disk_auto_does_not_budget_unsupported_gpu_cache(self):
        g, costs, gpu, ram = fixture()
        g.family = "qwen4"
        a = args(path="unused.gguf", moe_device="disk", moe_cpu_cache="auto")
        with patch.object(auto, "inventory", return_value=({"general.architecture": "qwen4exp"}, [])), \
                patch.object(auto, "geometry", return_value=g), \
                patch.object(auto, "weight_costs", return_value=costs), \
                patch.object(auto, "host_memory", return_value=ram), \
                patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            result = auto.resolve(a, gpu.__getitem__)
        self.assertEqual(a.moe_cuda_cache, 0)
        self.assertGreater(a.moe_cpu_cache, 0)
        self.assertTrue(result["notes"])

    def test_draft_experts_follow_model_placement_and_preserve_mtp(self):
        g, costs, gpu, ram = fixture()
        g.kv.append(1024)
        g.state.append(MiB)
        experts, dense, *rest = costs
        costs = (experts + [GiB], dense + [MiB], *rest)
        a = args(mtp=3, moe_cuda_cache=0)
        before = vars(a).copy()
        g.family = "qwen4"
        qwen = auto.plan(a, g, costs, "tp", {0: 1, 1: 1}, gpu, ram)
        g.family = "glm5"
        glm = auto.plan(a, g, costs, "tp", {0: 1, 1: 1}, gpu, ram)
        self.assertEqual(glm["host_estimated_bytes"] - qwen["host_estimated_bytes"], GiB)
        for d in gpu:
            self.assertEqual(qwen["gpu_estimated_bytes"][d] - glm["gpu_estimated_bytes"][d], GiB // 2)
        self.assertEqual(vars(a), before)

    def test_auto_kv_precision_follows_explicit_activation_precision(self):
        g, costs, gpu, ram = fixture()
        a = args(atype="float32", kv_cache_dtype="auto")
        with patch.object(auto, "inventory", return_value=({}, [])), \
                patch.object(auto, "geometry", return_value=g) as geometry, \
                patch.object(auto, "weight_costs", return_value=costs), \
                patch.object(auto, "host_memory", return_value=ram), \
                patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            auto.resolve(a, gpu.__getitem__)
        geometry.assert_called_once_with({}, "float32", 0)
        self.assertEqual((a.atype, a.kv_cache_dtype), ("float32", "auto"))

    def test_deepseek_shared_compressed_cache_accounting(self):
        md = {"model_type": "deepseek_v41", "num_hidden_layers": 4,
              "max_position_embeddings": 65536, "hidden_size": 4096,
              "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
              "head_dim": 512, "index_head_dim": 128, "num_attention_heads": 64,
              "compress_ratios": [1, 1, 2, 2], "kv_source_layer_ids": [0, 2],
              "index_source_layer_ids": [0, 2], "sliding_window": 128}
        g = auto.geometry(md, "fp8_e4m3", 0)
        self.assertEqual(g.kv, [528 + 132, 0, 264 + 66, 0])
        self.assertEqual(g.state, [128 * 512 * 2] * 4)
        self.assertEqual(auto.geometry(md, "fp4", 0).kv, [288 + 68, 0, 144 + 34, 0])

    def test_pp_layer_assignment_matches_native_boundary(self):
        # Model calls SelectDeviceFromMap(layer + 1, block_cnt), using <=.
        self.assertEqual([auto.layer_device(i, 4, {0: 1, 1: 1}) for i in range(4)], [0, 0, 1, 1])
        self.assertEqual(auto.devices(args(device="cudapp=0,1")), ("pp", {0: 1, 1: 1}))


if __name__ == "__main__":
    unittest.main()
