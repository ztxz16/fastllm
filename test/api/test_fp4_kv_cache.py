import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools")))
from fastllm_pytools.tui import (
    KV_CACHE_DTYPE_CHOICES, DeployConfig, build_fastllm_argv, config_from_dict,
)
from fastllm_pytools.util import make_normal_parser


class FP4KVConfigTest(unittest.TestCase):
    def test_default_and_explicit_cache_type(self):
        parser = make_normal_parser("test")
        self.assertEqual(parser.parse_args([]).kv_cache_dtype, "auto")
        for dtype in ("fp4", "nvfp4", "fp4_e2m1", "fp8_e4m3", "float16"):
            with self.subTest(dtype=dtype):
                self.assertEqual(parser.parse_args(["--kv_cache_dtype", dtype]).kv_cache_dtype, dtype)

    def test_launcher_preserves_fp4_for_tp_server(self):
        config = config_from_dict({"command": "server", "model": "/models/qwen", "device": "tp",
                                   "tp": "2", "dtype": "auto", "kv_cache_dtype": "fp4"})
        self.assertIn("fp4", dict(KV_CACHE_DTYPE_CHOICES))
        argv = build_fastllm_argv(config)
        self.assertEqual(argv[argv.index("--kv_cache_dtype") + 1], "fp4")
        self.assertEqual(config.dtype, "auto")
        self.assertNotIn("--dtype", argv)  # Keep the default source weight type.
        self.assertEqual(argv[argv.index("--tp") + 1], "2")

    def test_webui_keeps_cache_option_on_model_server(self):
        argv = build_fastllm_argv(DeployConfig(command="webui", kv_cache_dtype="fp4"))
        self.assertNotIn("--kv_cache_dtype", argv)


if __name__ == "__main__":
    unittest.main()
