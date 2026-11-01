"""Clover placement limits and checkpoint/OpenAI sampling semantics."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import auto_config, hybrid_auto
from fastllm_pytools.util import make_normal_parser
from test_unrestricted_sampling import SamplingModel, completion, RawRequest
from test_qwen35_reasoning import FakeQwen35Model
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import ChatCompletionRequest


class CloverAutoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = dict(model_type="clover_moe", block_size=32,
            max_position_embeddings=131072, layer_types=["linear_attention"] * 30 + ["full_attention"] * 10,
            num_attention_heads=16, linear_num_key_heads=16, linear_num_value_heads=32, num_key_value_heads=2, head_dim=256,
            hidden_size=2048, moe_intermediate_size=512, shared_expert_intermediate_size=512,
            num_experts=256, num_experts_per_tok=8, vocab_size=248320)
        (Path(self.temp.name) / "config.json").write_text(json.dumps(self.config))
        self.weights = [hybrid_auto.Tensor("model.layers.0.mlp.experts.gate_up_proj", (), 65 << 30, "BF16")]

    def args(self, *extra):
        args = make_normal_parser("test").parse_args([self.temp.name, "--auto", *extra])
        args.path = args.model
        return args

    def resolve(self, args, free=140 << 30):
        with patch.object(hybrid_auto, "inventory", return_value=(self.config, self.weights)), redirect_stdout(io.StringIO()):
            return auto_config.resolve(args, lambda d: dict(free=free, total=144 << 30),
                hardware=({0: {"capability": 90}}, {}), ram=dict(total=512 << 30, available=400 << 30))

    def test_native_capacity_and_original_precision(self):
        args = self.args()
        result = self.resolve(args)
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual((args.dtype, args.atype, args.kv_cache_dtype), ("bfloat16",) * 3)
        self.assertEqual((args.max_batch, args.tokens, args.speculative_num_draft_tokens), (1, 131072, 16))
        self.assertEqual(result["kv_bytes"], 131072 * 20480)
        self.assertEqual(result["snapshot_bytes"], 16 * result["recurrent_state_bytes"])
        self.assertEqual(args.chunked_prefill_size, 4096)

    def test_tp_capacity_and_kv_replication(self):
        from fastllm_pytools import clover_config
        single = self.resolve(self.args("--max_context_length", "4096"))
        for ranks in (2, 4):
            args = self.args("--tp", "cuda:" + ",".join(map(str, range(ranks))),
                             "--max_context_length", "4096")
            with patch.object(hybrid_auto, "inventory", return_value=(self.config, self.weights)), redirect_stdout(io.StringIO()):
                result = clover_config.resolve(args, lambda d: dict(free=140 << 30, total=144 << 30), self.config,
                    hardware=({d: {"capability": 90} for d in range(ranks)}, {}), ram=dict(available=400 << 30))
            self.assertEqual(result["mode"], "thread_tp")
            self.assertEqual(result["weights_bytes"] * ranks, single["weights_bytes"])
            self.assertEqual(result["recurrent_state_bytes"] * ranks, single["recurrent_state_bytes"])
            self.assertEqual(result["kv_bytes"], single["kv_bytes"] // 2)
            self.assertEqual(args.device, args.tp)
        for spec in ("cuda:0,0", "cuda:0:1,1:2", "cuda:0,1,2"):
            with self.assertRaises(ValueError):
                clover_config.prepare(self.args("--tp", spec), self.config)

    def test_explicit_fp16(self):
        args = self.args("--atype", "float16")
        result = self.resolve(args)
        self.assertEqual(result["dtype"], "float16")
        self.assertEqual((args.dtype, args.atype, args.kv_cache_dtype), ("float16",) * 3)

    def test_legacy_auto_fp16(self):
        for capability in (60, 61, 70, 75):
            args = self.args()
            with patch.object(hybrid_auto, "inventory", return_value=(self.config, self.weights)), redirect_stdout(io.StringIO()):
                result = auto_config.resolve(args, lambda d: dict(free=140 << 30, total=144 << 30),
                    hardware=({0: {"capability": capability}}, {}), ram=dict(total=512 << 30, available=400 << 30))
            self.assertEqual(result["dtype"], "float16")
            self.assertEqual((args.dtype, args.atype, args.kv_cache_dtype), ("float16",) * 3)

    def test_batch_capacity_is_accounted_per_request(self):
        single = self.resolve(self.args())
        for size in (2, 4, 8):
            with self.subTest(size=size):
                args = self.args("--max_batch", str(size))
                batch = self.resolve(args)
                self.assertEqual((args.max_batch, batch["max_batch"]), (size, size))
                self.assertEqual(batch["weights_bytes"], single["weights_bytes"])
                for key in ("recurrent_state_bytes", "snapshot_bytes", "kv_bytes"):
                    self.assertEqual(batch[key], size * single[key])
        with self.assertRaisesRegex(ValueError, "self_speculative"):
            off = self.args("--max_batch", "2", "--speculative_algorithm", "off")
            off.speculative_algorithm = ""
            self.resolve(off)

    def test_explicit_off_uses_hf_block_width(self):
        args = self.args("--speculative_algorithm", "off", "--max_context_length", "4096",
                         "--chunked_prefill_size", "512")
        # The CLI's normalizer clears off but preserves its explicit marker.
        args.speculative_algorithm = ""
        result = self.resolve(args)
        self.assertEqual(args.speculative_algorithm, "")
        self.assertEqual(args.speculative_num_draft_tokens, 32)
        self.assertEqual(result["snapshot_bytes"], 0)
        self.assertEqual(args.tokens, 4096)
        self.assertEqual(args.chunked_prefill_size, 512)

    def test_large_batch_keeps_full_cache_budget(self):
        single = self.resolve(self.args("--max_context_length", "4096"))
        for size in (9, 16, 32, 50, 64):
            with self.subTest(size=size):
                args = self.args("--max_batch", str(size), "--max_context_length", "4096")
                with patch.object(hybrid_auto, "inventory", return_value=(self.config, self.weights)), redirect_stdout(io.StringIO()):
                    result = auto_config.resolve(args, lambda d: dict(free=256 << 30, total=256 << 30),
                        hardware=({0: {"capability": 90}}, {}), ram=dict(total=512 << 30, available=400 << 30))
                for key in ("recurrent_state_bytes", "snapshot_bytes", "kv_bytes"):
                    self.assertEqual(result[key], size * single[key])
        short = self.resolve(self.args("--max_batch", "32", "--max_context_length", "4096",
                                       "--gpu_mem_ratio", "0.98"))
        self.assertLess(short["estimated_gpu_bytes"], 140 << 30)
        # Larger batches retain a bounded MoE graph workspace budget as well
        # as their full per-request caches and eager/logits scratch.
        with self.assertRaisesRegex(ValueError, "one CUDA"):
            self.resolve(self.args("--max_batch", "50", "--max_context_length", "4096",
                                   "--gpu_mem_ratio", "0.98"))
        with self.assertRaisesRegex(ValueError, "one CUDA"):
            self.resolve(self.args("--max_batch", "50", "--gpu_mem_ratio", "0.98"))

    def test_insufficient_memory_is_transactional(self):
        args = self.args(); original = copy.deepcopy(vars(args))
        with self.assertRaisesRegex(ValueError, "one CUDA"):
            self.resolve(args, free=40 << 30)
        self.assertEqual(vars(args), original)

    def test_unsupported_settings_fail_before_loading(self):
        for options in (("--dtype", "int8"), ("--max_batch", "65"),
                        ("--max_context_length", "131073"), ("--speculative_num_draft_tokens", "7"),
                        ("--device", "cuda:0,1"),
                        ("--max_batch", "2", "--speculative_num_draft_tokens", "8")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.resolve(self.args(*options))

    def test_prefix_cache_host_budget(self):
        with patch.dict(os.environ, {"FASTLLM_PREFIX_CACHE": "true"}):
            enabled = self.resolve(self.args("--prefix_cache_snapshot_max_records", "3"))
            history = self.resolve(self.args("--cache_history", "true", "--prefix_cache_snapshot_max_records", "3"))
            disabled = self.resolve(self.args("--prefix_cache", "false"))
        self.assertGreater(enabled["prefix_cache_host_bytes"], 0)
        self.assertEqual(enabled["prefix_cache_host_bytes"], history["prefix_cache_host_bytes"])
        self.assertEqual(disabled["prefix_cache_host_bytes"], 0)
        self.assertEqual(enabled["estimated_gpu_bytes"], disabled["estimated_gpu_bytes"])

    def test_prefix_capacity_tracks_concurrency_and_explicit_limits(self):
        with patch.dict(os.environ, {}, clear=True):
            args = self.args("--max_batch", "13", "--max_context_length", "8192")
            automatic = self.resolve(args)
            self.assertEqual(automatic["prefix_cache_max_records"], 52)
            with patch.dict(os.environ, {"FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST": "6"}):
                self.assertEqual(self.resolve(args)["prefix_cache_max_records"], 78)
            with patch.dict(os.environ, {"FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS": "7 "}):
                self.assertEqual(self.resolve(args)["prefix_cache_max_records"], 52)
            with patch.dict(os.environ, {"FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS": "7"}):
                self.assertEqual(self.resolve(args)["prefix_cache_max_records"], 7)
                args.prefix_cache_snapshot_max_records = 9
                self.assertEqual(self.resolve(args)["prefix_cache_max_records"], 9)
            args.prefix_cache_snapshot_max_records = 1 << 30
            with self.assertRaisesRegex(ValueError, "host RAM"):
                self.resolve(args)


class CloverSamplingModel(SamplingModel):
    default_generation_config = dict(top_k=0, top_p=1., temperature=1.,
        do_sample=False, repetition_penalty=1.05, presence_penalty=.9, frequency_penalty=0.)

    def launch_stream_response(self, query, **kwargs):
        self.sampling_args = dict(kwargs)
        kwargs.pop("clover_options", None)
        return FakeQwen35Model.launch_stream_response(self, query, **kwargs)


class CloverSamplingAPITest(unittest.IsolatedAsyncioTestCase):
    async def test_tools_use_xml_parser_without_native_constraint(self):
        tool = {"type": "function", "function": {"name": "lookup_stock",
            "parameters": {"type": "object", "properties": {
                "sku": {"type": "string"}}, "required": ["sku"]}}}
        for choice in ("auto", "required", {"type": "function", "function": {"name": "lookup_stock"}}):
            with self.subTest(tool_choice=choice):
                model = CloverSamplingModel(output='</think><tool_call>\n<function=lookup_stock>\n'
                    '<parameter=sku>SKU-42</parameter>\n</function>\n</tool_call>')
                model.config = {"model_type": "clover_moe", "vocab_size": 248320}
                model.get_type = lambda: "clover_moe"
                model.hf_tokenizer = SimpleNamespace(chat_template='<tool_call><function=',
                    get_vocab=lambda: {"<tool_call>": 1, "</tool_call>": 2})
                server = completion(model)
                request = ChatCompletionRequest(model=server.model_name,
                    messages=[{"role": "user", "content": "查询 SKU-42 库存"}],
                    tools=[tool], tool_choice=choice, max_tokens=128)
                response = await server.create_chat_completion(request, RawRequest())
                self.assertNotIn("tool_call_constraint", model.sampling_args)
                self.assertEqual(response.choices[0].finish_reason, "tool_calls")
                call = response.choices[0].message.tool_calls[0]
                self.assertEqual(call.function.name, "lookup_stock")
                self.assertEqual(json.loads(call.function.arguments), {"sku": "SKU-42"})

    async def test_defaults_and_independent_additive_penalties(self):
        for overrides, presence, frequency, repetition, sample in (
                ({}, .9, 0., 1.05, False),
                ({"presence_penalty": 0., "frequency_penalty": .4}, 0., .4, 1.05, False),
                ({"repetition_penalty": 1.2, "temperature": 1.0}, .9, 0., 1.2, True),
                ({"temperature": 0., "do_sample": True}, .9, 0., 1.05, False)):
            with self.subTest(overrides=overrides):
                model = CloverSamplingModel()
                model.config = {"model_type": "clover_moe", "vocab_size": 248320}
                model.default_generation_config = dict(top_k=0, top_p=1., temperature=1.,
                    do_sample=False, repetition_penalty=1.05, presence_penalty=.9, frequency_penalty=0.)
                server = completion(model)
                request = ChatCompletionRequest(model=server.model_name,
                    messages=[{"role": "user", "content": "hello"}], max_tokens=32, **overrides)
                await server.create_chat_completion(request, RawRequest())
                args = model.sampling_args
                self.assertEqual(args["clover_options"]["presence_penalty"], presence)
                self.assertEqual(args["clover_options"]["frequency_penalty"], frequency)
                self.assertEqual(args["repeat_penalty"], repetition)
                self.assertEqual(args["do_sample"], sample)


if __name__ == "__main__":
    unittest.main()
