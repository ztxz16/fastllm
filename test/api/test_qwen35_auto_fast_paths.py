import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.util import (
    _configure_qwen35_auto_fast_paths,
)


def _args(**overrides):
    values = {
        "tp": "cuda:0,1",
        "device": "cuda:0",
        "low": False,
        "cuda_embedding": False,
        "max_batch": 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class Qwen35AutoFastPathsTest(unittest.TestCase):
    def test_enables_tested_cuda_tp_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=True, mtp=0)

            self.assertEqual(os.environ["FASTLLM_CUDA_GRAPH"], "1")
            self.assertEqual(
                os.environ["FASTLLM_QWEN35_GPU_TOKEN_HANDOFF"], "1")
            self.assertEqual(
                os.environ["FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH"], "64")
            self.assertTrue(args.cuda_embedding)

    def test_caps_automatic_graph_batch(self):
        with patch.dict(os.environ, {}, clear=True):
            _configure_qwen35_auto_fast_paths(
                _args(max_batch=128), is_qwen35_model=True, mtp=0)

            self.assertEqual(
                os.environ["FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH"], "64")

    def test_respects_explicit_disable_overrides(self):
        overrides = {
            "FASTLLM_CUDA_GRAPH": "0",
            "FASTLLM_GPU_TOKEN_HANDOFF": "0",
        }
        with patch.dict(os.environ, overrides, clear=True):
            args = _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=True, mtp=0)

            self.assertNotIn(
                "FASTLLM_QWEN35_GPU_TOKEN_HANDOFF", os.environ)
            self.assertNotIn(
                "FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH", os.environ)
            self.assertFalse(args.cuda_embedding)

    def test_does_not_enable_handoff_for_mtp(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=True, mtp=1)

            self.assertNotIn("FASTLLM_CUDA_GRAPH", os.environ)
            self.assertNotIn(
                "FASTLLM_QWEN35_GPU_TOKEN_HANDOFF", os.environ)
            self.assertFalse(args.cuda_embedding)

    def test_does_not_change_other_models(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=False, mtp=0)

            self.assertEqual(dict(os.environ), {})
            self.assertFalse(args.cuda_embedding)


if __name__ == "__main__":
    unittest.main()
