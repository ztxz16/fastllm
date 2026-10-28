import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools.util import _configure_windows_cuda_graph


class WindowsGraphDefaultTest(unittest.TestCase):
    def configure(self, *, platform="win32", env=None, supported=True, **args):
        values = dict(device="cuda:0", tp="", low=False, low_gpu_mem=False)
        values.update(args)
        with patch("sys.platform", platform), patch.dict(
                os.environ, env or {}, clear=True), patch(
                "fastllm_pytools.util._cuda_graph_auto_supported",
                return_value=supported) as probe:
            _configure_windows_cuda_graph(SimpleNamespace(**values))
            return dict(os.environ), probe.call_args_list

    def test_cuda_default_including_cpu_embedding_and_thread_tp(self):
        for options in ({}, {"low_gpu_mem": True}, {"tp": "cuda:0,1"}):
            with self.subTest(options=options):
                env, calls = self.configure(**options)
                self.assertEqual(env["FASTLLM_CUDA_GRAPH"], "1")
                self.assertEqual(len(calls), 1)

    def test_explicit_override_does_not_probe_or_change_other_flags(self):
        for value in ("0", "off", "false", "1"):
            with self.subTest(value=value):
                expected = {"FASTLLM_CUDA_GRAPH": value,
                            "FASTLLM_QWEN35_MTP_VERIFY_CUDA_GRAPH": "0"}
                env, calls = self.configure(env=expected)
                self.assertEqual(env, expected)
                self.assertEqual(calls, [])

    def test_cpu_low_unsupported_and_other_platforms_keep_existing_policy(self):
        for options in ({"device": "cpu"}, {"low": True},
                        {"supported": False}, {"platform": "linux"},
                        {"platform": "darwin"}):
            with self.subTest(options=options):
                env, _ = self.configure(**options)
                self.assertNotIn("FASTLLM_CUDA_GRAPH", env)


if __name__ == "__main__":
    unittest.main()
