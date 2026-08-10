import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.util import (
    _configure_qwen35_auto_fast_paths,
    _configure_triton_compiler_python,
    _find_triton_python,
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
                os.environ["FASTLLM_GPU_TOKEN_HANDOFF"], "1")
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

            self.assertEqual(os.environ["FASTLLM_GPU_TOKEN_HANDOFF"], "0")
            self.assertNotIn(
                "FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH", os.environ)
            self.assertFalse(args.cuda_embedding)

    def test_does_not_enable_handoff_for_mtp(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=True, mtp=1)

            self.assertNotIn("FASTLLM_CUDA_GRAPH", os.environ)
            self.assertNotIn(
                "FASTLLM_GPU_TOKEN_HANDOFF", os.environ)
            self.assertFalse(args.cuda_embedding)

    def test_does_not_change_other_models(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=False, mtp=0)

            self.assertEqual(dict(os.environ), {})
            self.assertFalse(args.cuda_embedding)


class TritonPythonAutoDetectionTest(unittest.TestCase):
    def test_checks_only_current_python(self):
        current_python = "/tmp/current-venv/bin/python"
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util.sys.executable", current_python), patch(
                    "fastllm_pytools.util._triton_python_works",
                    return_value=True) as checker:
            self.assertEqual(_find_triton_python(), current_python)

        checker.assert_called_once_with(current_python)

    def test_returns_empty_when_current_python_lacks_triton(self):
        current_python = "/tmp/current-venv/bin/python"
        with patch(
                "fastllm_pytools.util.sys.executable", current_python), patch(
                    "fastllm_pytools.util._triton_python_works",
                    return_value=False) as checker:
            self.assertEqual(_find_triton_python(), "")

        checker.assert_called_once_with(current_python)

    def test_enables_current_interpreter(self):
        current_python = "/opt/current-venv/bin/python"
        with patch.dict(
                os.environ,
                {"FASTLLM_CUDA_TRITON_PYTHON": "/custom/python"},
                clear=True), patch(
                "fastllm_pytools.util._find_triton_python",
                return_value=current_python):
            detected = _configure_triton_compiler_python()

            self.assertEqual(detected, current_python)
            self.assertEqual(
                os.environ["FASTLLM_CUDA_TRITON_PYTHON"], detected)
            self.assertEqual(os.environ["FASTLLM_CUDA_TRITON"], "1")

    def test_disables_triton_when_current_interpreter_is_unavailable(self):
        output = io.StringIO()
        with patch.dict(
                os.environ,
                {
                    "FASTLLM_CUDA_TRITON": "1",
                    "FASTLLM_CUDA_TRITON_PYTHON": "/custom/python",
                },
                clear=True), patch(
                    "fastllm_pytools.util.sys.executable",
                    "/usr/bin/python3"), patch(
                    "fastllm_pytools.util._find_triton_python",
                    return_value=""), redirect_stdout(output):
            self.assertEqual(_configure_triton_compiler_python(), "")

            self.assertEqual(os.environ["FASTLLM_CUDA_TRITON"], "0")
            self.assertNotIn("FASTLLM_CUDA_TRITON_PYTHON", os.environ)
            self.assertIn("Triton is unavailable", output.getvalue())
            self.assertIn("--triton has been disabled", output.getvalue())


if __name__ == "__main__":
    unittest.main()
