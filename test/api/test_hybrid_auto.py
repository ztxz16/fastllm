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
    def test_qwen35_fp8_payload_and_scale_are_budgeted_once(self):
        metadata = dict(model_type='qwen3_5_text', num_hidden_layers=4,
                        hidden_size=1024, num_attention_heads=8,
                        num_key_value_heads=2, head_dim=128,
                        intermediate_size=2048, max_position_embeddings=8192,
                        layer_types=['linear_attention'] * 3 + ['full_attention'],
                        linear_num_key_heads=4, linear_num_value_heads=8,
                        linear_key_head_dim=64, linear_value_head_dim=64,
                        linear_conv_kernel_dim=4, mtp_num_hidden_layers=1)
        model = auto.geometry(metadata, 'float16', 0)
        tensors = [auto.Tensor('model.layers.0.mlp.up_proj.weight', (128, 128), 16384, 'F8_E4M3'),
                   auto.Tensor('model.layers.0.mlp.up_proj.weight_scale_inv', (1, 1), 4, 'F32')]
        costs = auto.weight_costs(tensors, model, args(dtype='fp8_e4m3'))
        self.assertEqual(costs[1][0], 16388)
        self.assertEqual(model.kv, [0, 0, 0, 1024])

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

    def test_independent_cache_respects_stage_ownership_and_tiny_budget(self):
        g, c, gpu, ram = fixture()
        g.experts = 256
        a = args(moe_device_layers=-1)
        p = auto.plan(a, g, c, 'pp', {0: 1, 1: 1}, gpu, ram, per_device_cache=True)
        self.assertEqual(p['moe_cuda_cache_per_device'], {0: 2 * GiB, 1: 2 * GiB})
        # No allocation may consume the budget already reserved for workspace.
        overhead = p['gpu_estimated_bytes'][1] - p['moe_cuda_cache_per_device'][1]
        gpu[1]['free'] = overhead + (gpu[1]['total'] - p['gpu_budget_bytes'][1]) + MiB
        p = auto.plan(a, g, c, 'pp', {0: 1, 1: 1}, gpu, ram, per_device_cache=True)
        self.assertEqual(p['moe_cuda_cache_per_device'][1], 0)
        self.assertGreater(p['moe_cuda_cache_per_device'][0], 0)
        self.assertLessEqual(p['gpu_estimated_bytes'][1], p['gpu_budget_bytes'][1])
        gpu[1]['free'] = gpu[1]['total']
        for mode, backend in [('tp', 'numa'), ('pp', 'disk')]:
            q = auto.plan(args(moe_device=backend), g, c, mode, {0: 1, 1: 1}, gpu, ram,
                          per_device_cache=True)
            self.assertNotIn('moe_cuda_cache_per_device', q)

    def test_implicit_embedding_reserves_first_pp_stage_and_full_gpu_mtp(self):
        g, c, gpu, ram = fixture()
        g.experts = 256
        c = ([4 * GiB] * 4, *c[1:])
        a = args(moe_device_layers=-1)
        base = auto.plan(a, g, c, 'pp', {0: 1, 1: 1}, gpu, ram, per_device_cache=True)
        p = auto.plan(a, g, c, 'pp', {0: 1, 1: 1}, gpu, ram, per_device_cache=True,
                      implicit_embedding_bytes=GiB)
        self.assertEqual(base['moe_cuda_cache_per_device'][0] - p['moe_cuda_cache_per_device'][0], GiB)
        self.assertEqual(base['moe_cuda_cache_per_device'][1], p['moe_cuda_cache_per_device'][1])
        q = auto.plan(a, g, c, 'pp', {0: 1, 1: 1}, gpu, ram, per_device_cache=True,
                      implicit_embedding_bytes=GiB, hybrid_cpu_embedding=True)
        self.assertEqual(q['moe_cuda_cache_per_device'], base['moe_cuda_cache_per_device'])
        # A full-GPU candidate cannot use the hybrid CPU-embedding exemption.
        a.moe_device_layers, a.mtp = 0, 3
        gpu = {d: dict(total=32 * GiB, free=32 * GiB) for d in (0, 1)}
        p = auto.plan(a, g, c, 'pp', {0: 1, 1: 1}, gpu, ram, allow_full_gpu=True,
                      implicit_embedding_bytes=GiB, hybrid_cpu_embedding=True)
        q = auto.plan(a, g, c, 'pp', {0: 1, 1: 1}, gpu, ram, allow_full_gpu=True)
        for d in gpu:
            self.assertEqual(p['gpu_estimated_bytes'][d] - q['gpu_estimated_bytes'][d], GiB)

    def test_qwen_gguf_embedding_budgets_expanded_fp32_table(self):
        g, _, _, _ = fixture()
        g.family = 'qwen4'
        tensors = [auto.Tensor('blk.0.ffn_up_exps.weight', (8, 64, 128), MiB, 'gguf:11'),
                   auto.Tensor('token_embd.weight', (1024, 128), 1024 * 128, 'gguf:23')]
        a = args(cuda_embedding=True)
        costs = auto.weight_costs(tensors, g, a)
        self.assertEqual(costs[2:4], (1024 * 128 * 4, 1024 * 128 * 4))
        a.low_gpu_mem = True
        costs = auto.weight_costs(tensors, g, a)
        self.assertEqual(costs[2:4], (0, 1024 * 128 * 4))

    def test_cpu_and_gpu_paths_reject_auto_before_hardware_query(self):
        for main, moe, mode in [("cpu", "numa", "auto"), ("cuda:0", "cuda:0", "auto"),
                                ("cuda:0", "numa", "gpu")]:
            a = args(device=main, moe_device=moe)
            before = vars(a).copy()
            with patch.dict(os.environ, {"FASTLLM_EXECUTION_MODE": mode}, clear=True), \
                    self.assertRaises(ValueError):
                auto.resolve(a, lambda _: self.fail("unexpected CUDA access"))
            self.assertEqual(vars(a), before)

    def test_pp_resident_experts_receive_a_parsed_front_device_map(self):
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
            for backend in ("numa", "disk"):
                with self.subTest(backend=backend):
                    a = make_normal_parser("test").parse_args([
                        path, "--device", "cudapp=0,1", "--tp", "off", "--threads", "1",
                        "--moe_device", backend, "--moe_device_layers", "22"])
                    runtime = MagicMock()
                    runtime.model.return_value.get_max_input_len.return_value = 32768
                    runtime.model.return_value.get_max_batch.return_value = 1
                    with patch.dict(os.environ, {}, clear=True), \
                            patch.dict(sys.modules, {"ftllm": types.SimpleNamespace(llm=runtime)}), \
                            redirect_stdout(io.StringIO()):
                        make_normal_llm_model(a)
                    runtime.set_device_map.assert_any_call({"cuda:0": 1, "cuda:1": 1}, True)
                    runtime.set_layered_moe_device_map.assert_called_once_with(backend)
                    runtime.set_moe_device_layers.assert_any_call(22)

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

    def test_fastllm_gguf_nested_and_symlink_shards_keep_quantized_sizes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            shared = p / "shared"
            shared.mkdir()
            (p / "config.json").write_text('{}')
            weight_map = {}
            expected = []
            for i, kind in enumerate((19, 42, 29, 12, 11)):
                key = "layers.%d.ffn.experts.%d.w1.weight" % (i % 4, i)
                name = "experts/%d/model.safetensors" % i
                payload = struct.pack("<iii", 1, 9999, kind) + bytes(256 + i)
                header = json.dumps({key: {"dtype": "fastllm", "shape": [256, 256],
                                         "data_offsets": [0, len(payload)]}}).encode()
                file = p / name
                file.parent.mkdir(parents=True)
                target = Path(shared) / str(i) if i == 0 else file
                target.write_bytes(struct.pack("<Q", len(header)) + header + payload)
                if i == 0:
                    file.symlink_to(target)
                weight_map[key] = name
                expected.append(auto.Tensor(key, (256, 256), len(payload), "gguf:%d" % kind))
            (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
            _, tensors = auto.inventory(p)
            self.assertEqual(tensors, expected)
            g, _, _, _ = fixture()
            self.assertEqual(sum(auto.weight_costs(tensors, g, args(dtype="float16"))[0]),
                             sum(t.bytes for t in expected))

    def test_safetensors_rejects_index_traversal_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "config.json").write_text('{}')
            for name in ("../outside.safetensors", "experts/../../outside.safetensors",
                         str(p / "model.safetensors"), "experts\\outside.safetensors"):
                with self.subTest(name=name):
                    (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": name}}))
                    with self.assertRaisesRegex(ValueError, "invalid safetensors shard path"):
                        auto.inventory(p)

    def test_fastllm_gguf_rejects_truncated_or_unsupported_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "config.json").write_text('{}')
            for payload in (bytes(8), struct.pack("<iii", 1, 9999, 19),
                            struct.pack("<iii", 2, 9999, 19) + bytes(32),
                            struct.pack("<iii", 1, 8, 19) + bytes(32),
                            struct.pack("<iii", 1, 9999, -1) + bytes(32)):
                with self.subTest(payload=payload[:12]):
                    header = json.dumps({"x": {"dtype": "fastllm", "shape": [256, 256],
                                              "data_offsets": [0, len(payload)]}}).encode()
                    (p / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
                    with self.assertRaisesRegex(ValueError, "fastllm"):
                        auto.inventory(p)

    def test_safetensors_subdirectories_and_path_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            model = parent / 'model'
            (model / 'experts').mkdir(parents=True)
            (model / 'config.json').write_text('{}')
            header = json.dumps({'x': {'dtype': 'F16', 'shape': [8],
                                      'data_offsets': [0, 16]}}).encode()
            payload = struct.pack('<Q', len(header)) + header + bytes(16)
            shard = model / 'experts' / 'rank0.safetensors'
            shard.write_bytes(payload)
            index = model / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'x': 'experts/rank0.safetensors'}}))
            _, tensors = auto.inventory(model)
            self.assertEqual(tensors, [auto.Tensor('x', (8,), 16, 'F16')])
            outside = parent / 'outside.safetensors'
            outside.write_bytes(payload)
            (model / 'linked.safetensors').symlink_to(outside)
            (model / 'linked_dir').symlink_to(parent, target_is_directory=True)
            for name in ('../outside.safetensors', 'linked.safetensors',
                         'linked_dir/outside.safetensors', str(shard.resolve())):
                with self.subTest(name=name):
                    index.write_text(json.dumps({'weight_map': {'x': name}}))
                    with self.assertRaisesRegex(ValueError, 'invalid safetensors shard path'):
                        auto.inventory(model)

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

    def test_deepseek_native_lookup_tables_stay_off_gpu(self):
        g, _, _, _ = fixture()
        a = args(ngram_device="disk")
        t = [auto.Tensor("layers.1.engram.embed.weight", (1024, 256), 262144, "F8_E4M3"),
             auto.Tensor("layers.1.engram.embed.scale", (1024, 8), 8192, "F8_E8M0"),
             auto.Tensor("embed.weight", (1024, 8), 16384, "BF16"),
             auto.Tensor("layers.0.ffn.experts.0.w1.weight", (8, 8), 64, "F8_E4M3")]
        self.assertTrue(all(auto.is_lookup_table(x.name) for x in t[:2]))
        self.assertFalse(auto.is_lookup_table("layers.1.engram.wkv.weight"))
        e, dense, common, host, largest, _ = auto.weight_costs(t, g, a)
        self.assertEqual((sum(e), sum(dense), common, host, largest), (64, 0, 0, 16384, 16384))

    def test_native_fp8_experts_support_numa_cuda_cache(self):
        from dataclasses import replace
        g, _, _, _ = fixture()
        for family in ("glm5", "deepseek_v41"):
            for kind in ("F8_E4M3", "F8_E4M3FN"):
                for dtype in ("auto", "fp8_e4m3"):
                    with self.subTest(family=family, kind=kind, dtype=dtype):
                        t = [auto.Tensor("layers.0.ffn.experts.0.up_proj.weight", (8, 8), 64, kind)]
                        e, *_ = auto.weight_costs(t, replace(g, family=family), args(dtype=dtype))
                        self.assertEqual(e[0], 64)
            t = [auto.Tensor("layers.0.ffn.experts.0.up_proj.weight", (8, 8), 64, "F8_E5M2")]
            with self.assertRaises(ValueError):
                auto.weight_costs(t, replace(g, family=family), args())

    def test_deepseek_pp_head_and_idle_vision_are_not_replicated(self):
        g = auto.geometry(self.dspark_metadata(), "float16", 0)
        tensors = [auto.Tensor("layers.0.ffn.experts.0.w1.weight", (8, 8), 64, "gguf:1"),
                   auto.Tensor("head.weight", (16, 8), 256, "BF16"),
                   auto.Tensor("norm.weight", (8,), 32, "F32"),
                   auto.Tensor("vision.blocks.0.attn.wqkv.weight", (8, 8), 128, "F16"),
                   auto.Tensor("aligner.w1.weight", (8, 8), 128, "F16"),
                   auto.Tensor("image_start", (8,), 32, "F32")]
        _, dense, common, host, _, _ = auto.weight_costs(tensors, g, args())
        self.assertEqual(common, 0)
        self.assertEqual(dense, [0, 0, 0, 544])
        self.assertEqual(host, 288)

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

    def test_native_packed_fp4_scales_are_counted_once(self):
        from dataclasses import replace
        g, _, _, _ = fixture()
        prefix = "layers.0.ffn.experts.0.w1"
        for family in ("deepseek_v41", "qwen4", "glm5"):
            for kind in ("I8", "U8"):
                for suffix in (".weight_scale_inv", ".weight_scale", ".scale_inv", ".scale"):
                    for dtype in ("auto", "nvfp4"):
                        with self.subTest(family=family, kind=kind, suffix=suffix, dtype=dtype):
                            # Eight rows of 64 values: 256 packed bytes, 16
                            # block32 scale bytes. The block16 fallback is 288.
                            w = auto.Tensor(prefix + ".weight", (8, 32), 256, kind)
                            scale = auto.Tensor(prefix + suffix, (8, 2), 16, "F8_E8M0")
                            for tensors in ([w, scale], [scale, w]):
                                costs = auto.weight_costs(tensors, replace(g, family=family), args(dtype=dtype))
                                self.assertEqual((costs[0][0], costs[4]), (288, 288))

    def test_packed_fp4_scale_replication_and_raw_mxfp4(self):
        g, _, _, _ = fixture()
        prefix = "model.layers.0.mlp.experts.0.down_proj"
        w = auto.Tensor(prefix + ".weight", (8, 32), 256, "U8")
        # Row-broadcast scales and larger source groups must still reserve
        # one scale per runtime block and output row.
        for shape, size, kind in [((2, 2), 4, "F8_E8M0"), ((8, 1), 8, "F8_E8M0"),
                                  ((8, 4), 32, "F8_E8M0"), ((8, 2), 16, "U8")]:
            with self.subTest(shape=shape, kind=kind):
                scale = auto.Tensor(prefix + ".weight_scale", shape, size, kind)
                self.assertEqual(auto.weight_costs([scale, w], g, args())[0][0], 288)

    def test_e4m3_packed_fp4_covers_compact_alignment_and_float_scales(self):
        from dataclasses import replace
        g, _, _, _ = fixture()
        prefix = "model.layers.0.mlp.experts.0.gate_proj"
        # Compact 16-value rows need 16 bytes including global scale/alignment;
        # 32-value rows need 24. For 64 values the float-scale fallback needs 48.
        for columns, expected in [(16, 48), (32, 72), (64, 144)]:
            w = auto.Tensor(prefix + ".weight", (3, columns // 2), 3 * columns // 2, "U8")
            scale = auto.Tensor(prefix + ".weight_scale", (3, columns // 16), 3 * columns // 16, "F8_E4M3")
            global_scale = auto.Tensor(prefix + ".weight_scale_2", (), 4, "F32")
            for family in ("qwen4", "glm5", "deepseek_v41"):
                for backend in ("numa", "cpu", "disk"):
                    with self.subTest(columns=columns, family=family, backend=backend):
                        costs = auto.weight_costs([scale, global_scale, w], replace(g, family=family),
                                                 args(moe_device=backend, moe_cuda_cache=0))
                        self.assertEqual(costs[0][0], expected + 4)
                        self.assertEqual(costs[4], expected)

    def test_packed_fp4_unrecognized_metadata_keeps_conservative_bound(self):
        g, _, _, _ = fixture()
        prefix = "layers.0.ffn.experts.0.w1"
        w = auto.Tensor(prefix + ".weight", (8, 32), 256, "I8")
        scales = [None, auto.Tensor(prefix + ".scale", (8, 2), 64, "F32"),
                  auto.Tensor(prefix + ".scale", (8, 2), 15, "F8_E8M0"),
                  auto.Tensor(prefix + ".scale", (3, 2), 6, "F8_E8M0"),
                  auto.Tensor(prefix + ".scale", (8, 4), 32, "U8"),
                  auto.Tensor(prefix + ".scale", (8, 2), 16, "F8_E4M3")]
        for scale in scales:
            with self.subTest(scale=scale):
                ts = [w] + ([scale] if scale else [])
                self.assertGreaterEqual(auto.weight_costs(ts, g, args())[0][0], 512)
        # Match loader priority: an invalid first scale cannot be replaced by
        # a convenient lower-priority tensor to shrink the budget.
        ts = [w, auto.Tensor(prefix + ".weight_scale_inv", (8, 2), 64, "F32"),
              auto.Tensor(prefix + ".scale", (8, 2), 16, "F8_E8M0")]
        self.assertGreaterEqual(auto.weight_costs(ts, g, args())[0][0], 512)

    def test_packed_fp4_explicit_conversion_and_expert_override(self):
        g, _, _, _ = fixture()
        prefix = "layers.0.ffn.experts.0.w1"
        ts = [auto.Tensor(prefix + ".weight", (8, 32), 256, "I8"),
              auto.Tensor(prefix + ".scale", (8, 2), 16, "F8_E8M0")]
        for dtype, minimum in [("float16", 1024), ("bfloat16", 1024), ("float32", 2048)]:
            with self.subTest(dtype=dtype):
                costs = auto.weight_costs(ts, g, args(dtype=dtype, moe_cuda_cache=0))
                self.assertGreaterEqual(costs[0][0], minimum)
                costs = auto.weight_costs(ts, g, args(dtype=dtype, moe_dtype="nvfp4"))
                self.assertEqual(costs[0][0], 288)
        costs = auto.weight_costs(ts, g, args(dtype="nvfp4", moe_dtype="float32", moe_cuda_cache=0))
        self.assertGreaterEqual(costs[0][0], 2048)

    def test_packed_fp4_real_geometry_fits_384gib_numa_plan(self):
        g = auto.Geometry("deepseek_v41", 40, 131072, 5120, 384, 6, 2304,
                          [4224] + [0] * 39, [131072] * 40, 1, 360448, 4096)
        tensors = []
        for layer in range(40):
            for projection, shape, scale_shape in [
                    ("gate_up_proj", (384, 4608, 2560), (384, 4608, 160)),
                    ("down_proj", (384, 5120, 1152), (384, 5120, 72))]:
                prefix = "model.layers.%d.mlp.experts.%s" % (layer, projection)
                import math
                tensors.extend([auto.Tensor(prefix + ".weight", shape, math.prod(shape), "I8"),
                                auto.Tensor(prefix + ".scale", scale_shape, math.prod(scale_shape), "F8_E8M0")])
        a = args(tokens=131072, max_context_length=131072, chunked_prefill_size=4096,
                 moe_device_layers=-1, moe_cuda_cache=20 * GiB)
        costs = auto.weight_costs(tensors, g, a)
        # 270 GiB of payload+scales on the block32 backend; keep about 285 GiB
        # for the larger block16 fallback instead of the old 522 GiB estimate.
        self.assertEqual(sum(costs[0]), 305764761600)
        gpu = {0: {"total": 72 * GiB, "free": 71 * GiB}}
        with patch.dict(os.environ, {}, clear=True):
            p = auto.plan(a, g, costs, "pp", {0: 1}, gpu,
                          {"total": 384 * GiB, "available": 352 * GiB})
            self.assertEqual(p["moe_device_layers"], -1)
            self.assertLess(p["host_estimated_bytes"], p["host_budget_bytes"])
            with self.assertRaises(ValueError):
                auto.plan(a, g, costs, "pp", {0: 1}, gpu,
                          {"total": 256 * GiB, "available": 256 * GiB})

    def test_host_reserve_keeps_floor_and_caps_large_machine_budget(self):
        g, c, gpu, _ = fixture()
        a = args(moe_device_layers=-1, moe_cuda_cache=0)
        for total, available, reserve in [(8 * GiB, 8 * GiB, GiB),
                                          (32 * GiB, 28 * GiB, 32 * GiB // 10),
                                          (128 * GiB, 120 * GiB, 4 * GiB)]:
            with self.subTest(total=total), patch.dict(os.environ, {}, clear=True):
                ram = {"total": total, "available": available}
                p = auto.plan(a, g, c, "pp", {0: 1}, gpu, ram)
                self.assertEqual(p["host_budget_bytes"], available - reserve)
                # The reserve remains unavailable even when total RAM is large.
                ram["available"] = p["host_estimated_bytes"] + reserve - 1
                with self.assertRaises(ValueError):
                    auto.plan(a, g, c, "pp", {0: 1}, gpu, ram)

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
        geometry.assert_called_once_with({}, "float32", 0, 0)
        self.assertEqual((a.atype, a.kv_cache_dtype), ("float32", "auto"))

    def test_disk_moe_keeps_cuda_shared_experts_in_dense_budget(self):
        from dataclasses import replace
        g, _, _, _ = fixture()
        g = replace(g, family="deepseek_v41")
        tensors = [auto.Tensor("layers.0.ffn.experts.0.w1.weight", (64, 32), 2048, "F8_E4M3"),
                   auto.Tensor("layers.0.ffn.shared_experts.w1.weight", (64, 32), 2048, "F8_E4M3")]
        a = args(moe_device="disk", cuda_shared_expert="true", dtype="auto")
        experts, dense, _, host, *_ = auto.weight_costs(tensors, g, a)
        self.assertEqual((experts[0], dense[0], host), (2048, 4096, 0))
        a.cuda_shared_expert = "false"
        experts, dense, _, host, *_ = auto.weight_costs(tensors, g, a)
        self.assertEqual((experts[0], dense[0], host), (2048, 0, 4096))

    def test_deepseek_shared_compressed_cache_accounting(self):
        md = {"model_type": "deepseek_v41", "num_hidden_layers": 4,
              "max_position_embeddings": 65536, "hidden_size": 4096,
              "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
              "head_dim": 512, "index_head_dim": 128, "num_attention_heads": 64,
              "compress_ratios": [1, 1, 2, 2], "kv_source_layer_ids": [0, 2],
              "index_source_layer_ids": [0, 2], "sliding_window": 128}
        g = auto.geometry(md, "fp8_e4m3", 0)
        self.assertEqual(g.kv, [528 + 132, 0, 264 + 66, 0])
        self.assertEqual(g.state, [128 * 528] * 4)
        self.assertEqual(auto.geometry(md, "fp4", 0).kv, [288 + 68, 0, 144 + 34, 0])
        self.assertEqual(g.streamed_kv, [])

    def dspark_metadata(self):
        return {"model_type": "deepseek_v41", "num_hidden_layers": 4,
                "max_position_embeddings": 1048576, "hidden_size": 4096,
                "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
                "head_dim": 512, "index_head_dim": 128, "num_attention_heads": 64,
                "compress_ratios": [1, 1, 2, 2, 0, 0], "kv_source_layer_ids": [0, 2],
                "index_source_layer_ids": [0, 2], "sliding_window": 128,
                "num_nextn_predict_layers": 2, "dspark_block_size": 5,
                "dspark_target_layer_ids": [2, 3], "dspark_noise_token_id": 100,
                "dspark_markov_rank": 32, "dspark_n_routed_experts": 128,
                "dspark_num_experts_per_tok": 3, "vocab_size": 1024}

    def test_v41_gguf_prefill_transport_does_not_consume_decode_cache_budget(self):
        g = auto.geometry(self.dspark_metadata(), "float16", 0)
        a = args(tokens=4096, max_context_length=4096, moe_device_layers=-1)
        costs = ([4*GiB]*g.layers, [MiB]*g.layers, 0, 0, MiB, False)
        gpu = {0: dict(total=16*GiB, free=16*GiB)}
        ram = dict(total=64*GiB, available=64*GiB)
        native = auto.plan(a, g, costs, "pp", {0: 1}, gpu, ram)
        g.gguf_expert_bytes = {1: 13*MiB+1}
        packed = auto.plan(a, g, costs, "pp", {0: 1}, gpu, ram)
        extra = 96*(13*MiB+256)+272*MiB
        self.assertEqual(packed["expert_transport_bytes_per_device"], extra)
        self.assertEqual(packed["workspace_bytes_per_device"]-native["workspace_bytes_per_device"], extra)
        self.assertGreaterEqual(native["moe_cuda_cache"]-packed["moe_cuda_cache"], extra-MiB)
        self.assertEqual(packed["host_estimated_bytes"], native["host_estimated_bytes"])
        # Resident-only layers need no host transfer ring.
        a.moe_device_layers = 0
        resident = auto.plan(a, g, costs, "pp", {0: 1},
                             {0: dict(total=32*GiB, free=32*GiB)}, ram, allow_full_gpu=True)
        self.assertEqual(resident["expert_transport_bytes_per_device"], 0)

    def test_v41_transport_only_charges_stages_with_host_experts(self):
        g = auto.geometry(self.dspark_metadata(), "float16", 0)
        a = args(tokens=4096, max_context_length=4096, moe_device_layers=2, moe_cuda_cache=0)
        costs = ([GiB] * 4, [MiB] * 4, 0, 0, MiB, False)
        gpu = {d: dict(total=16 * GiB, free=16 * GiB) for d in (0, 1)}
        ram = dict(total=64 * GiB, available=64 * GiB)
        native = auto.plan(a, g, costs, "pp", {0: 1, 1: 1}, gpu, ram)
        g.gguf_expert_bytes = {2: 13 * MiB}
        packed = auto.plan(a, g, costs, "pp", {0: 1, 1: 1}, gpu, ram)
        self.assertEqual(packed["gpu_estimated_bytes"][0], native["gpu_estimated_bytes"][0])
        self.assertEqual(packed["gpu_estimated_bytes"][1] - native["gpu_estimated_bytes"][1],
                         96 * 13 * MiB + 272 * MiB)

    def test_v41_dspark_transport_covers_gpu_only_helper_stages(self):
        g = auto.geometry(self.dspark_metadata(), "float16", 0)
        a = args(tokens=4096, max_context_length=4096, moe_device_layers=2,
                 moe_cuda_cache=0, dspark=5)
        costs = ([GiB] * 4, [MiB] * 4, 0, 0, MiB, False)
        gpu = {d: dict(total=16 * GiB, free=16 * GiB) for d in (0, 1)}
        ram = dict(total=64 * GiB, available=64 * GiB)
        native = auto.plan(a, g, costs, "pp", {0: 1, 1: 1}, gpu, ram)
        g.gguf_expert_bytes = {2: 13 * MiB}
        packed = auto.plan(a, g, costs, "pp", {0: 1, 1: 1}, gpu, ram)
        for device in (0, 1):
            self.assertEqual(packed["gpu_estimated_bytes"][device] - native["gpu_estimated_bytes"][device],
                             96 * 13 * MiB + 272 * MiB)
        # A helper's staging budget cannot be spent again on resident weights/cache.
        gpu[0]["free"] = packed["gpu_estimated_bytes"][0] - 1
        with self.assertRaises(ValueError):
            auto.plan(a, g, costs, "pp", {0: 1, 1: 1}, gpu, ram)

    def test_v41_resident_gguf_prefill_uses_phase_workspace(self):
        metadata = self.dspark_metadata()
        metadata['quantization_config'] = dict(quant_method='fp8', expert_dtype='fp4',
                                                weight_block_size=[32, 32])
        g = auto.geometry(metadata, 'fp4', 0)
        a = args(fast_prefill=True, atype='bfloat16')
        g.gguf_expert_bytes = {i: MiB for i in range(g.layers)}
        with patch.dict(os.environ, {}, clear=True):
            host = auto._workspace_bytes(a, g, 4096, 1048576, 'pp', 0)
            resident = auto._workspace_bytes(a, g, 4096, 1048576, 'pp', 2)
            generic = auto._workspace_bytes(a, g, 4096, 1048576, 'tp', 2)
            self.assertGreater(resident, host)  # resident route/MMQ buffers stay live
            self.assertLess(resident, generic)
            del g.gguf_expert_bytes[1]
            self.assertEqual(auto._workspace_bytes(a, g, 4096, 1048576, 'pp', 2), generic)

    def test_v41_host_staging_is_reused_only_with_early_cuda_placement(self):
        g = auto.geometry(self.dspark_metadata(), 'float16', 0)
        a = args(tokens=4096, max_context_length=4096, moe_device_layers=2,
                 moe_cuda_cache=0, fast_prefill=False)
        costs = ([GiB] * 4, [MiB] * 4, 0, 0, MiB, False)
        gpu = {0: dict(total=16 * GiB, free=16 * GiB)}
        ram = dict(total=64 * GiB, available=64 * GiB)
        with patch.dict(os.environ, {}, clear=True):
            native = auto.plan(a, g, costs, 'pp', {0: 1}, gpu, ram)
            for flag, value in [('FASTLLM_DSV41_REFERENCE_MATH', '1'),
                                ('FASTLLM_DSV41_DUMP_DIR', '/tmp/v41-dump')]:
                with patch.dict(os.environ, {flag: value}):
                    debug = auto.plan(a, g, costs, 'pp', {0: 1}, gpu, ram)
                self.assertEqual(debug['host_estimated_bytes'] - native['host_estimated_bytes'],
                                 native['workspace_bytes_per_device'])

    def test_v41_without_slabs_accounts_for_resident_allocation_pages(self):
        g = auto.geometry(self.dspark_metadata(), 'float16', 0)
        g.gguf_expert_bytes = {0: 4 * MiB, 1: 4 * MiB}
        a = args(tokens=4096, max_context_length=4096, moe_device_layers=2,
                 moe_cuda_cache=0, cuda_slab=64)
        costs = ([GiB] * 4, [MiB] * 4, 0, 0, MiB, False)
        gpu = {0: dict(total=4 * GiB, free=4 * GiB)}
        ram = dict(total=64 * GiB, available=64 * GiB)
        self.assertEqual(auto.plan(a, g, costs, 'pp', {0: 1}, gpu, ram)['resident_layers'], 2)
        a.cuda_slab = 0
        with self.assertRaises(ValueError):
            auto.plan(a, g, costs, 'pp', {0: 1}, gpu, ram)

    def test_v41_later_index_sources_share_persistent_keys(self):
        md = self.dspark_metadata()
        md["index_source_layer_ids"] = [0, 1, 2, 3]
        g = auto.geometry(md, "float16", 0)
        self.assertEqual(g.kv, [1280, 0, 640, 0])
        self.assertEqual(g.kv_reallocation, [1024, 0, 512, 0])
        self.assertEqual(sum(g.kv_padding), 2 * 256 * 1280)

    def test_v41_native_kv_layout_and_1m_capacity_budget(self):
        md = self.dspark_metadata()
        md.update(num_hidden_layers=40, compress_ratios=[2] * 20 + [1] * 20,
                  kv_source_layer_ids=[2, 8, 14, 20],
                  index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36])
        for dtype, per_token, window_row in [('float16', 3200, 1024),
                                            ('float32', 3200, 1024),
                                            ('fp8_e4m3', 1650, 528),
                                            ('fp4', 890, 528),
                                            ('fp4_e2m1', 890, 528)]:
            with self.subTest(dtype=dtype):
                g = auto.geometry(md, dtype, 0)
                self.assertEqual(sum(g.kv), per_token)
                self.assertEqual(sum(g.state), 40 * 128 * window_row)
        g = auto.geometry(md, 'fp4', 0)
        a = args(tokens=1048576, max_context_length=1048576, moe_device='disk',
                 moe_device_layers=-1, moe_cuda_cache=0, moe_cpu_cache=0)
        p = auto.plan(a, g, ([GiB] * 40, [MiB] * 40, 0, 0, GiB, False),
                      'pp', {0: 1}, {0: dict(total=32 * GiB, free=32 * GiB)},
                      dict(total=64 * GiB, available=64 * GiB))
        self.assertEqual(p['kv_logical_bytes_per_device'][0], 890 * 1048576)
        self.assertEqual(p['kv_capacity_bytes_per_device'][0], 890 * 1048576 * 3 // 2 + 4 * 256 * 356)
        self.assertEqual(p['kv_reallocation_bytes_per_device'][0], 288 * 1048576)
        self.assertLess(p['kv_capacity_bytes_per_device'][0] + p['kv_reallocation_bytes_per_device'][0], 1.6 * GiB)

    def test_v41_score_budget_honors_native_chunk_overrides(self):
        g = auto.geometry(self.dspark_metadata(), "float16", 0)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(auto._score_workspace(g, 4096, 1048576), 128 * MiB)
            with patch.dict(os.environ, {"FASTLLM_DSV41_INDEX_CHUNK": "64"}):
                self.assertEqual(auto._score_workspace(g, 4096, 1048576), 384 * MiB)
            with patch.dict(os.environ, {"FASTLLM_DSV41_INDEX_SCORE_MB": "32"}):
                self.assertEqual(auto._score_workspace(g, 4096, 1048576), 32 * MiB)
            # Even a tiny byte budget must fit a full score row.
            with patch.dict(os.environ, {"FASTLLM_DSV41_INDEX_SCORE_MB": "1"}):
                self.assertEqual(auto._score_workspace(g, 4096, 1048576), 6 * MiB)

    def test_v41_weight_budget_matches_native_precision_exceptions(self):
        g = auto.geometry(self.dspark_metadata(), "float16", 0)
        tensors = [auto.Tensor("layers.0.ffn.experts.0.w1.weight", (8, 8), 64, "F8_E4M3")]
        for name in ("layers.0.attn.wq_b", "layers.0.attn.wo_a", "layers.0.engram.wkv"):
            tensors += [auto.Tensor(name + ".weight", (64, 32), 2048, "F8_E4M3"),
                        auto.Tensor(name + ".scale", (2, 1), 2, "F8_E8M0")]
        tensors += [auto.Tensor("head.weight", (64, 32), 4096, "BF16"),
                    auto.Tensor("layers.0.ffn.gate.weight", (8, 32), 512, "BF16")]
        with patch.dict(os.environ, {}, clear=True):
            sizes = auto._deepseek_v41_weight_sizes(tensors, args())
            self.assertEqual(sizes["layers.0.attn.wq_b.weight"], 2056)
            self.assertEqual(sizes["layers.0.attn.wq_b.scale"], 0)
            self.assertNotIn("layers.0.attn.wo_a.weight", sizes)
            self.assertEqual(sizes["head.weight"], 8192)
            self.assertEqual(sizes["layers.0.ffn.gate.weight"], 1024)
            self.assertEqual(sizes["layers.0.engram.wkv.weight"], 2056)
            with patch.dict(os.environ, {"FASTLLM_DSV41_ENGRAM_WKV_FP8": "0"}):
                self.assertNotIn("layers.0.engram.wkv.weight",
                                 auto._deepseek_v41_weight_sizes(tensors, args()))
            for dtype in ("float16", "bfloat16", "float32"):
                self.assertNotIn("layers.0.attn.wq_b.weight",
                                 auto._deepseek_v41_weight_sizes(tensors, args(dtype=dtype)))
            # Missing scales must retain the existing conversion allowance.
            no_scale = [t for t in tensors if not t.name.endswith(".scale")]
            self.assertNotIn("layers.0.attn.wq_b.weight",
                             auto._deepseek_v41_weight_sizes(no_scale, args()))
            self.assertGreater(auto.weight_costs(no_scale, g, args())[1][0],
                               auto.weight_costs(tensors, g, args())[1][0])

    def test_v41_auto_shrinks_chunk_and_reserves_kv_expansion_at_1m(self):
        g = auto.geometry(self.dspark_metadata(), "float16", 0)
        costs = ([GiB] * 4, [MiB] * 4, 0, 0, GiB, False)
        ram = dict(total=64 * GiB, available=64 * GiB)
        a = args(tokens=1048576, max_context_length=1048576, moe_cuda_cache=0,
                 moe_device_layers=-1, chunked_prefill_size=256)
        with patch.dict(os.environ, {}, clear=True):
            p = auto.plan(a, g, costs, "pp", {0: 1},
                          {0: dict(total=32 * GiB, free=32 * GiB)}, ram)
            reserve = (p["gpu_estimated_bytes"][0] - p["workspace_bytes_per_device"] -
                       512 * MiB - sum(costs[1]) - sum(g.state))
            self.assertGreaterEqual(reserve, 1.5 * a.tokens * sum(g.kv) +
                                    a.tokens * max(g.kv_reallocation))
            # Fit the smallest candidate, with no room for a larger one.
            gpu = {0: dict(total=32 * GiB, free=p["gpu_estimated_bytes"][0] + 1)}
            a.gpu_mem_ratio = 1
            a.chunked_prefill_size = "auto"
            fit = auto.plan(a, g, costs, "pp", {0: 1}, gpu, ram)
            self.assertEqual((fit["context_tokens"], fit["chunked_prefill_size"]),
                             (1048576, 256))
            a.chunked_prefill_size = 512
            with self.assertRaises(ValueError):
                auto.plan(a, g, costs, "pp", {0: 1}, gpu, ram)

    def test_v41_disk_workspace_reuse_has_guarded_scope_and_full_context_scores(self):
        md = self.dspark_metadata()
        md["quantization_config"] = {"quant_method": "fp8", "expert_dtype": "fp4",
                                     "weight_block_size": [32, 32]}
        g = auto.geometry(md, "float16", 0, 7)
        a = args(moe_device="disk", fast_prefill=True)
        with patch.dict(os.environ, {}, clear=True):
            bounded = auto._workspace_bytes(a, g, 2048, 1048576, "pp", 0)
            old = auto._workspace_bytes(a, g, 2048, 1048576, "tp", 0)
            self.assertLess(bounded, old)
            numa = copy.copy(a)
            numa.moe_device = "numa"
            self.assertEqual(auto._workspace_bytes(numa, g, 2048, 1048576, "pp", 0), bounded)
            self.assertGreater(bounded, auto._workspace_bytes(a, g, 256, 1048576, "pp", 0))
            for changes in (dict(dtype="float32"), dict(atype="float32"),
                            dict(max_batch=2), dict(fast_prefill=False)):
                other = copy.copy(a)
                vars(other).update(changes)
                self.assertEqual(auto._workspace_bytes(other, g, 2048, 1048576, "pp", 0), old)
            self.assertEqual(auto._workspace_bytes(a, g, 2048, 1048576, "pp", 1), old)
            with patch.dict(os.environ, {"FASTLLM_DSV41_DUMP_DIR": "/tmp/dump"}):
                self.assertEqual(auto._workspace_bytes(a, g, 2048, 1048576, "pp", 0), old)
            with patch.dict(os.environ, {"FASTLLM_DSV41_REFERENCE_MATH": "false"}):
                # Native V41EnvFlag treats every non-empty value except "0" as enabled.
                self.assertEqual(auto._workspace_bytes(a, g, 2048, 1048576, "pp", 0), old)
            with patch.dict(os.environ, {"FASTLLM_DSV41_INDEX_SCORE_MB": "1024"}):
                self.assertGreater(auto._workspace_bytes(a, g, 2048, 1048576, "pp", 0), bounded)

    def test_v41_reused_workspace_preserves_chunk_boundary_and_kv_reserve(self):
        md = self.dspark_metadata()
        md["quantization_config"] = {"quant_method": "fp8", "expert_dtype": "fp4",
                                     "weight_block_size": [32, 32]}
        g = auto.geometry(md, "float16", 0)
        a = args(moe_device="disk", fast_prefill=True, tokens=1048576,
                 max_context_length=1048576, moe_device_layers=-1, moe_cpu_cache=0,
                 moe_cuda_cache=0, chunked_prefill_size=2048, gpu_mem_ratio=1)
        costs = ([GiB] * 4, [MiB] * 4, 0, 0, GiB, False)
        ram = dict(total=64 * GiB, available=64 * GiB)
        with patch.dict(os.environ, {}, clear=True):
            p = auto.plan(a, g, costs, "pp", {0: 1}, {0: dict(total=32 * GiB, free=32 * GiB)}, ram)
            gpu = {0: dict(total=32 * GiB, free=p["gpu_estimated_bytes"][0])}
            a.chunked_prefill_size = "auto"
            fit = auto.plan(a, g, costs, "pp", {0: 1}, gpu, ram)
            self.assertEqual((fit["context_tokens"], fit["chunked_prefill_size"]), (1048576, 2048))
            gpu[0]["free"] -= 1
            smaller = auto.plan(a, g, costs, "pp", {0: 1}, gpu, ram)
            self.assertEqual(smaller["chunked_prefill_size"], 1024)
            a.chunked_prefill_size = 2048
            with self.assertRaises(ValueError):
                auto.plan(a, g, costs, "pp", {0: 1}, gpu, ram)

    def test_v41_disk_8192_candidate_respects_capacity_context_and_explicit_chunk(self):
        md = self.dspark_metadata()
        md["quantization_config"] = {"quant_method": "fp8", "expert_dtype": "fp4",
                                     "weight_block_size": [32, 32]}
        g = auto.geometry(md, "fp4", 0)
        a = args(moe_device="disk", fast_prefill=True, tokens=1048576,
                 max_context_length=1048576, moe_device_layers=-1, moe_cpu_cache=0,
                 moe_cuda_cache=0, chunked_prefill_size=8192, gpu_mem_ratio=1)
        costs = ([GiB] * 4, [MiB] * 4, 0, 0, GiB, False)
        ram = dict(total=64 * GiB, available=64 * GiB)
        roomy = {0: dict(total=32 * GiB, free=32 * GiB)}
        with patch.dict(os.environ, {}, clear=True):
            def plan(gpu):
                return auto.plan(a, g, costs, "pp", {0: 1}, gpu, ram)
            explicit = plan(roomy)
            gpu = {0: dict(total=32 * GiB, free=explicit["gpu_estimated_bytes"][0])}
            a.chunked_prefill_size = "auto"
            fit = plan(gpu)
            self.assertEqual((fit["context_tokens"], fit["chunked_prefill_size"]), (1048576, 8192))
            gpu[0]["free"] -= 1
            self.assertEqual(plan(gpu)["chunked_prefill_size"], 4096)
            a.chunked_prefill_size = 8192
            with self.assertRaises(ValueError):
                plan(gpu)
            a.chunked_prefill_size = 8196
            self.assertEqual(plan(roomy)["chunked_prefill_size"], 8196)
            a.chunked_prefill_size = "auto"
            a.tokens = a.max_context_length = 6000
            self.assertEqual(plan(roomy)["chunked_prefill_size"], 6000)
            a.tokens = a.max_context_length = 1048576
            a.moe_device = "numa"
            self.assertEqual(plan(roomy)["chunked_prefill_size"], 8192)
            a.moe_device, g.family = "disk", "fixture"
            self.assertEqual(plan(roomy)["chunked_prefill_size"], 4096)

    def test_dspark_has_bounded_draft_kv_and_budgets_every_draft_stage(self):
        md = self.dspark_metadata()
        off = auto.geometry(md, "float16", 0)
        on = auto.geometry(md, "float16", 0, 7)
        short = auto.geometry(md, "float16", 0, 2)
        self.assertEqual(on.kv, off.kv + [0, 0])
        self.assertEqual((on.dspark_block_size, short.dspark_block_size), (7, 5))
        self.assertGreater(sum(on.state), sum(off.state) + 2 * 128 * 512 * 2)
        self.assertGreater(on.activation_width, off.activation_width)
        tensors = [auto.Tensor("layers.0.ffn.experts.0.w1.weight", (8, 8), 64, "F8_E4M3")]
        for stage in range(3):
            prefix = "mtp.%d.ffn.experts.0.w1" % stage
            tensors += [auto.Tensor(prefix + ".weight", (8, 32), 256, "I8"),
                        auto.Tensor(prefix + ".scale", (8, 2), 16, "F8_E8M0"),
                        auto.Tensor("mtp.%d.attn.wq_a.weight" % stage, (8, 8), 128, "BF16")]
        a = args(dspark=7, speculative_algorithm="dspark")
        experts, dense, *_ = auto.weight_costs(tensors, on, a)
        self.assertEqual(experts, [64, 0, 0, 0, 288, 288])
        self.assertEqual(dense, [0, 0, 0, 0, 128, 128])
        self.assertEqual(auto.weight_costs(tensors, off, args())[0], [64, 0, 0, 0])
        auto.validate_options(a)
        with self.assertRaises(ValueError):
            auto.validate_options(args(dspark=7, speculative_algorithm="dspark",
                                       speculative_draft_model_path="external"))
        with self.assertRaises(ValueError):
            auto.geometry({**md, "model_type": "qwen4_exp"}, "float16", 0, 7)
        for changes in ({"num_nextn_predict_layers": 0}, {"dspark_target_layer_ids": [4]},
                        {"compress_ratios": [1, 1, 2, 2, 1, 0]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                auto.geometry({**md, **changes}, "float16", 0, 7)

    def test_dspark_draft_experts_follow_tail_layer_residency_and_dense_final_stage(self):
        g = auto.geometry(self.dspark_metadata(), "float16", 0, 7)
        costs = ([GiB] * 6, [MiB] * 4 + [2 * MiB, 3 * MiB], 0, 0, GiB, False)
        gpu = {d: dict(total=32 * GiB, free=32 * GiB) for d in (0, 1)}
        ram = dict(total=128 * GiB, available=128 * GiB)
        # Main prefix 0..2 and draft stage 0 (owner=2) are GPU resident;
        # main layer 3 and draft stage 1 (owner=3) remain on the host.
        a = args(dspark=7, speculative_algorithm="dspark", moe_cuda_cache=0, moe_device_layers=1)
        partial = auto.plan(a, g, costs, "pp", {0: 1, 1: 1}, gpu, ram)
        host = auto.plan(args(dspark=7, moe_cuda_cache=0, moe_device_layers=-1),
                         g, costs, "pp", {0: 1, 1: 1}, gpu, ram)
        self.assertEqual(host['host_estimated_bytes'] - partial['host_estimated_bytes'], 4 * GiB)
        self.assertEqual(partial['gpu_estimated_bytes'][0] - host['gpu_estimated_bytes'][0], GiB)
        self.assertEqual(partial['gpu_estimated_bytes'][1] - host['gpu_estimated_bytes'][1], 3 * GiB)
        without_draft_dense = (costs[0], [MiB] * 4 + [0, 0], *costs[2:])
        q = auto.plan(a, g, without_draft_dense, "pp", {0: 1, 1: 1}, gpu, ram)
        self.assertEqual(partial['gpu_estimated_bytes'][0], q['gpu_estimated_bytes'][0])
        self.assertEqual(partial['gpu_estimated_bytes'][1] - q['gpu_estimated_bytes'][1], 5 * MiB)
        full = auto.plan(args(dspark=7, moe_cuda_cache=0, moe_device_layers=0),
                         g, costs, "pp", {0: 1, 1: 1}, gpu, ram, allow_full_gpu=True)
        self.assertEqual(host['host_estimated_bytes'] - full['host_estimated_bytes'], 6 * GiB)

    def test_dspark_1m_budget_leaves_less_cache_without_shortening_context(self):
        md = self.dspark_metadata()
        gpu = {0: dict(total=64 * GiB, free=64 * GiB)}
        ram = dict(total=512 * GiB, available=512 * GiB)
        plans = []
        for drafts in (0, 7):
            g = auto.geometry(md, "float16", 0, drafts)
            costs = ([16 * GiB] * 4 + ([GiB] * 2 if drafts else []),
                     [MiB] * len(g.kv), 0, 0, GiB, False)
            a = args(dspark=drafts, tokens=1048576, max_context_length=1048576,
                     chunked_prefill_size=4096, moe_device_layers=-1)
            plans.append(auto.plan(a, g, costs, "pp", {0: 1}, gpu, ram))
        off, on = plans
        self.assertEqual((on['context_tokens'], on['total_tokens']), (1048576, 1048576))
        self.assertLess(on['moe_cuda_cache'], off['moe_cuda_cache'])
        self.assertGreaterEqual(on['host_estimated_bytes'] - off['host_estimated_bytes'], 2 * GiB)
        self.assertLessEqual(on['gpu_estimated_bytes'][0], on['gpu_budget_bytes'][0])

    def test_exact_qsa_streaming_geometry_excludes_the_full_context_index(self):
        md = {"model_type": "qwen4_exp", "num_hidden_layers": 4,
              "max_position_embeddings": 262144, "hidden_size": 2560,
              "num_experts": 512, "num_experts_per_tok": 8, "moe_intermediate_size": 1024,
              "num_key_value_heads": 2, "head_dim": 256}
        for mtp in (0, 3):
            g = auto.geometry(md, "float16", mtp)
            self.assertEqual(g.streamed_kv, [0, 0, 0, 2048] + ([2048] if mtp else []))
            self.assertEqual(g.kv, [0, 0, 0, 2176] + ([2176] if mtp else []))
        for dtype in ("bfloat16", "float32", "fp8_e4m3"):
            self.assertEqual(auto.geometry(md, dtype, 0).streamed_kv, [])
        for changes in ({"head_dim": 64}, {"indexer_budget": 0}, {"indexer_compress_ratio": 8},
                        {"compress_ratios": [0, 0, 0, 8]}):
            self.assertEqual(auto.geometry({**md, **changes}, "float16", 0).streamed_kv, [])

    def test_streaming_retains_index_and_counts_draft_host_budget_and_staging(self):
        g, costs, gpu, ram = fixture()
        g.family, g.context = "qwen4", 131072
        g.kv, g.streamed_kv = [2176] * 4, [2048] * 4
        a = args(atype="float16", kv_cache_dtype="float16", tokens=131072,
                 max_context_length=131072, moe_device_layers=-1, moe_cuda_cache=0)
        full = auto.plan(a, g, costs, "pp", {0: 1, 1: 1}, gpu, ram)
        streamed = auto.plan(a, g, costs, "pp", {0: 1, 1: 1}, gpu, ram, kv_resident=32768)
        self.assertGreater(streamed["kvmem"]["host_mib"], 1280)  # 4 target + possible draft + page
        self.assertEqual(streamed["workspace_bytes_per_device"] - full["workspace_bytes_per_device"],
                         131076 * 2048)
        for d in gpu:
            # Two layers/card save 384 MiB; full-layer staging costs 256 MiB
            # and page metadata/index alignment slightly reduce the net saving.
            saved = full["gpu_estimated_bytes"][d] - streamed["gpu_estimated_bytes"][d]
            self.assertGreater(saved, 120 * MiB)
            self.assertLess(saved, 128 * MiB)
        a.moe_device_layers = 0
        with self.assertRaises(ValueError):
            auto.plan(a, g, costs, "pp", {0: 1, 1: 1}, gpu, ram,
                      kv_resident=32768, allow_full_gpu=True)

    def test_pp_layer_assignment_matches_native_boundary(self):
        # Model calls SelectDeviceFromMap(layer + 1, block_cnt), using <=.
        self.assertEqual([auto.layer_device(i, 4, {0: 1, 1: 1}) for i in range(4)], [0, 0, 1, 1])
        self.assertEqual(auto.devices(args(device="cudapp=0,1")), ("pp", {0: 1, 1: 1}))


if __name__ == "__main__":
    unittest.main()
