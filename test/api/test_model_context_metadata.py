import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.openai_server.fastllm_model import FastLLmModel
from fastllm_pytools.tui import DeployConfig, build_fastllm_argv
from fastllm_pytools.util import add_server_args, make_normal_parser, make_normal_llm_model


class _FakeModel:
    def __init__(self, model_context_window, kv_cache_token_limit,
                 native_context_window = None, configured_limit = None,
                 model_type = None):
        self._model_context_window = model_context_window
        self._kv_cache_token_limit = kv_cache_token_limit
        self.native_context_window = native_context_window
        self.configured_context_window_limit = configured_limit
        self._model_type = model_type

    def get_max_input_len(self):
        return self._model_context_window

    def get_kv_cache_token_limit(self):
        return self._kv_cache_token_limit

    def get_type(self):
        return self._model_type


class FastLLmModelContextMetadataTest(unittest.TestCase):
    def test_resolved_window_keeps_original_model_metadata(self):
        model = _FakeModel(1000000, 1100000, native_context_window=262144,
                           configured_limit=1000000)
        model.context_config = {"configured": True, "context_window": 1000000}
        metadata = FastLLmModel("extended", model)
        self.assertEqual(metadata.context_window, 1000000)
        self.assertEqual(metadata.model_context_window, 262144)
        self.assertEqual(metadata.response["models"][0]["auto_compact_token_limit"], 900000)

    def test_resolved_bounded_cache_window_is_not_physical_retention(self):
        model = _FakeModel(1000000, 4096, native_context_window=262144)
        model.context_config = {"configured": True, "context_window": 1000000}
        self.assertEqual(FastLLmModel("bounded", model).context_window, 1000000)

    def test_launcher_preserves_rope_json_as_one_argument(self):
        rope = '{"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768}'
        config = DeployConfig(model="/tmp/model", command="server", rope_scaling=rope,
                              max_context_length="131072")
        argv = build_fastllm_argv(config)
        self.assertEqual(argv[argv.index("--rope_scaling") + 1], rope)
        self.assertEqual(argv[argv.index("--max_context_length") + 1], "131072")

    def test_model_limit_is_used_when_kv_cache_is_larger(self):
        metadata = FastLLmModel(
            "Qwen3.6-27B-FP8",
            _FakeModel(262144, 358400),
        )

        model = metadata.response["models"][0]
        self.assertEqual(model["context_window"], 262144)
        self.assertEqual(model["max_context_window"], 262144)
        self.assertEqual(model["model_context_window"], 262144)
        self.assertEqual(model["kv_cache_token_limit"], 358400)
        self.assertEqual(model["auto_compact_token_limit"], 235929)

    def test_kv_cache_limit_is_used_when_it_is_smaller(self):
        metadata = FastLLmModel(
            "large-context-model",
            _FakeModel(1048576, 358400),
        )

        self.assertEqual(metadata.context_window, 358400)

    def test_configured_per_session_limit_is_reported(self):
        metadata = FastLLmModel(
            "limited-model",
            _FakeModel(
                131072,
                358400,
                native_context_window = 262144,
                configured_limit = 131072,
            ),
        )

        model = metadata.response["data"][0]
        self.assertEqual(model["context_window"], 131072)
        self.assertEqual(model["model_context_window"], 262144)
        self.assertEqual(model["configured_context_window_limit"], 131072)

    def test_legacy_constructor_keeps_safe_fallback(self):
        metadata = FastLLmModel("unknown-model")

        self.assertEqual(metadata.context_window, 32768)

    def test_kimi_k3_reasoning_efforts_are_discoverable(self):
        metadata = FastLLmModel(
            "kimi-k3",
            _FakeModel(262144, 262144, model_type = "kimi_k3"),
        )

        model = metadata.response["data"][0]
        self.assertEqual(
            model["supported_reasoning_efforts"], ["low", "high", "max"])
        self.assertEqual(model["supportedReasoningEfforts"],
                         ["low", "high", "max"])
        self.assertEqual(model["default_reasoning_effort"], "max")
        self.assertEqual(model["defaultReasoningEffort"], "max")

    def test_qwen3_5_reasoning_efforts_are_discoverable(self):
        metadata = FastLLmModel(
            "qwen3.5",
            _FakeModel(262144, 262144, model_type = "qwen3_5"),
        )

        model = metadata.response["data"][0]
        self.assertEqual(
            model["supported_reasoning_efforts"],
            ["low", "medium", "xhigh"])
        self.assertEqual(
            model["supportedReasoningEfforts"],
            ["low", "medium", "xhigh"])
        self.assertEqual(model["default_reasoning_effort"], "xhigh")
        self.assertEqual(model["defaultReasoningEffort"], "xhigh")

    def test_glm5_next_reasoning_efforts_are_discoverable(self):
        metadata = FastLLmModel(
            "glm-5.3-flash",
            _FakeModel(2048, 2048, model_type = "glm5_next"),
        )

        model = metadata.response["data"][0]
        self.assertEqual(
            model["supported_reasoning_efforts"], ["low", "high", "max"])
        self.assertEqual(model["supportedReasoningEfforts"],
                         ["low", "high", "max"])
        self.assertEqual(model["default_reasoning_effort"], "max")
        self.assertEqual(model["defaultReasoningEffort"], "max")

    def test_qwen4_exp_reasoning_efforts_are_discoverable(self):
        metadata = FastLLmModel(
            "qwen4-exp",
            _FakeModel(262144, 262144, model_type = "qwen4_exp"),
        )

        model = metadata.response["data"][0]
        self.assertEqual(
            model["supported_reasoning_efforts"],
            ["low", "medium", "xhigh"])
        self.assertEqual(
            model["supportedReasoningEfforts"],
            ["low", "medium", "xhigh"])
        self.assertEqual(model["default_reasoning_effort"], "xhigh")
        self.assertEqual(model["defaultReasoningEffort"], "xhigh")

    def test_mmproj_model_advertises_image_input(self):
        native_model = _FakeModel(262144, 262144, model_type = "qwen3_5")
        native_model.mmproj_path = "/models/mmproj.gguf"

        model = FastLLmModel("qwen3.5", native_model).response["data"][0]

        self.assertEqual(model["input_modalities"], ["text", "image"])
        self.assertEqual(model["inputModalities"], ["text", "image"])


class ServerContextArgumentTest(unittest.TestCase):
    def test_mmproj_argument_is_parsed(self):
        args = make_normal_parser("test").parse_args([
            "--mmproj", "/models/mmproj.gguf",
        ])

        self.assertEqual(args.mmproj, "/models/mmproj.gguf")

    def test_hyphenated_alias_is_parsed(self):
        for server in (False, True):
            parser = make_normal_parser("context parameters")
            if server:
                add_server_args(parser)
            args = parser.parse_args(["--max-context-length", "131072", "--rope-scaling", "yarn"])
            self.assertEqual(args.max_context_length, 131072)
            self.assertEqual(args.rope_scaling, "yarn")

    def test_tui_adds_context_limit_to_server_command(self):
        argv = build_fastllm_argv(DeployConfig(
            command = "server",
            model = "/models/qwen",
            max_context_length = "131072",
        ))

        option_index = argv.index("--max_context_length")
        self.assertEqual(argv[option_index + 1], "131072")


class ContextLoadingTest(unittest.TestCase):
    def load(self, path, limit, extra=(), warmup_error=None):
        model = MagicMock()
        model.native_context_window = 32768
        model.configured_context_window_limit = None
        model.get_max_input_len.return_value = 32768
        model.get_max_batch.return_value = 1
        model.warmup.side_effect = warmup_error

        def shrink(value):
            model.get_max_input_len.return_value = min(value, model.get_max_input_len())
            return model.get_max_input_len()

        def construct(*args, **kwargs):
            if kwargs["max_context_length"] > 0:
                model.get_max_input_len.return_value = kwargs["max_context_length"]
            return model

        model.set_max_context_length.side_effect = shrink
        runtime = MagicMock()
        runtime.model.side_effect = construct
        args = make_normal_parser("legacy context").parse_args([
            str(path), "--device", "cpu", "--dtype", "float16", "--threads", "1",
            "--max_context_length", str(limit), *extra])
        with patch.dict(os.environ, {}, clear=True), \
                patch.dict(sys.modules, {"ftllm": SimpleNamespace(llm=runtime)}), \
                patch("fastllm_pytools.util.get_gguf_model_config", return_value=None), \
                redirect_stdout(io.StringIO()):
            if warmup_error is None:
                self.assertIs(make_normal_llm_model(args), model)
                model.release_memory.assert_not_called()
            else:
                with self.assertRaises(type(warmup_error)) as raised:
                    make_normal_llm_model(args)
                self.assertIs(raised.exception, warmup_error)
        return model, runtime.model.call_args.kwargs

    def test_legacy_file_limits_keep_shrink_only_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("gguf", "flm"):
                path = Path(directory, "model." + suffix)
                path.touch()
                for limit in (4096, 65536):
                    with self.subTest(suffix=suffix, limit=limit):
                        model, options = self.load(path, limit)
                        self.assertEqual(options["max_context_length"], -1)
                        model.set_max_context_length.assert_called_once_with(limit)
                        self.assertEqual(model.get_max_input_len(), min(limit, 32768))
                        self.assertEqual(model.native_context_window, 32768)
                        self.assertEqual(model.configured_context_window_limit, limit)
                        calls = [call[0] for call in model.mock_calls]
                        self.assertLess(calls.index("set_max_context_length"), calls.index("warmup"))

    def test_custom_graph_keeps_legacy_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory, "custom.py")
            custom.write_text("__model__ = object\n")
            model, options = self.load(directory, 4096, ["--custom", str(custom)])
            self.assertIsNotNone(options["graph"])
            self.assertEqual(options["max_context_length"], -1)
            model.set_max_context_length.assert_called_once_with(4096)

    def test_hf_limit_is_resolved_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            model, options = self.load(directory, 4096)
            self.assertEqual(options["max_context_length"], 4096)
            model.set_max_context_length.assert_not_called()

    def test_failed_capacity_check_releases_loaded_model(self):
        with tempfile.TemporaryDirectory() as directory:
            error = RuntimeError("Context capacity insufficient: requested 4096 tokens, calibrated KV capacity 2048")
            model, _ = self.load(directory, 4096, warmup_error=error)
            model.release_memory.assert_called_once_with()

    def test_explicit_rope_on_legacy_format_reaches_capability_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "model.gguf")
            path.touch()
            model, options = self.load(path, 4096, ["--rope_scaling", "yarn"])
            self.assertEqual(options["max_context_length"], 4096)
            self.assertEqual(options["rope_scaling"], "yarn")
            model.set_max_context_length.assert_not_called()


if __name__ == "__main__":
    unittest.main()
