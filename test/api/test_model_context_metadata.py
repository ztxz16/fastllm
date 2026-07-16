import argparse
import os
import sys
import unittest


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.openai_server.fastllm_model import FastLLmModel
from fastllm_pytools.tui import DeployConfig, build_fastllm_argv
from fastllm_pytools.util import add_server_args


class _FakeModel:
    def __init__(self, model_context_window, kv_cache_token_limit,
                 native_context_window = None, configured_limit = None):
        self._model_context_window = model_context_window
        self._kv_cache_token_limit = kv_cache_token_limit
        self.native_context_window = native_context_window
        self.configured_context_window_limit = configured_limit

    def get_max_input_len(self):
        return self._model_context_window

    def get_kv_cache_token_limit(self):
        return self._kv_cache_token_limit


class FastLLmModelContextMetadataTest(unittest.TestCase):
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


class ServerContextArgumentTest(unittest.TestCase):
    def test_hyphenated_alias_is_parsed(self):
        parser = argparse.ArgumentParser()
        add_server_args(parser)

        args = parser.parse_args(["--max-context-length", "131072"])

        self.assertEqual(args.max_context_length, 131072)

    def test_tui_adds_context_limit_to_server_command(self):
        argv = build_fastllm_argv(DeployConfig(
            command = "server",
            model = "/models/qwen",
            max_context_length = "131072",
        ))

        option_index = argv.index("--max_context_length")
        self.assertEqual(argv[option_index + 1], "131072")


if __name__ == "__main__":
    unittest.main()
