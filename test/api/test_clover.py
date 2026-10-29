"""Clover placement limits and checkpoint/OpenAI sampling semantics."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
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
            linear_num_key_heads=16, linear_num_value_heads=32, num_key_value_heads=2, head_dim=256)
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

    def test_insufficient_memory_is_transactional(self):
        args = self.args(); original = copy.deepcopy(vars(args))
        with self.assertRaisesRegex(ValueError, "one SM80"):
            self.resolve(args, free=40 << 30)
        self.assertEqual(vars(args), original)

    def test_unsupported_settings_fail_before_loading(self):
        for options in (("--dtype", "float16"), ("--max_batch", "2"),
                        ("--max_context_length", "131073"), ("--speculative_num_draft_tokens", "7"),
                        ("--device", "cuda:0,1"), ("--prefix_cache", "true")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.resolve(self.args(*options))


class CloverSamplingModel(SamplingModel):
    def launch_stream_response(self, query, **kwargs):
        self.sampling_args = dict(kwargs)
        kwargs.pop("clover_options", None)
        return FakeQwen35Model.launch_stream_response(self, query, **kwargs)


class CloverSamplingAPITest(unittest.IsolatedAsyncioTestCase):
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
