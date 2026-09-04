import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import mock_open, patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.util import (
    _configure_qwen35_auto_fast_paths,
    _configure_sm89_fp8_linear_triton,
    _configure_triton_compiler_python,
    _find_triton_python,
    _is_nvidia_cuda_platform,
    _uses_non_nvidia_cuda_compatible_build,
)


def _args(**overrides):
    values = {
        "tp": "cuda:0,1",
        "device": "cuda:0",
        "low": False,
        "speculative_algorithm": "",
        "cuda_embedding": False,
        "max_batch": 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class Qwen35AutoFastPathsTest(unittest.TestCase):
    def test_enables_tested_cuda_tp_defaults(self):
        capabilities = {0: 80, 1: 89}
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util._is_nvidia_cuda_platform",
                return_value=True), patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities",
                return_value=capabilities):
            args = _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=True, mtp=0)

            self.assertEqual(os.environ["FASTLLM_CUDA_GRAPH"], "1")
            self.assertEqual(
                os.environ["FASTLLM_GPU_TOKEN_HANDOFF"], "1")
            self.assertEqual(
                os.environ["FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH"], "64")
            self.assertTrue(args.cuda_embedding)

    def test_caps_automatic_graph_batch(self):
        capabilities = {0: 80, 1: 80}
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util._is_nvidia_cuda_platform",
                return_value=True), patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities",
                return_value=capabilities):
            _configure_qwen35_auto_fast_paths(
                _args(max_batch=128), is_qwen35_model=True, mtp=0)

            self.assertEqual(
                os.environ["FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH"], "64")

    def test_does_not_auto_enable_graph_on_sm75_or_older(self):
        for capability in (70, 75):
            with self.subTest(compute_capability=capability), patch.dict(
                    os.environ, {}, clear=True), patch(
                    "fastllm_pytools.util._is_nvidia_cuda_platform",
                    return_value=True), patch(
                    "fastllm_pytools.util._nvidia_cuda_compute_capabilities",
                    return_value={0: capability, 1: capability}):
                args = _configure_qwen35_auto_fast_paths(
                    _args(), is_qwen35_model=True, mtp=0)

                self.assertNotIn("FASTLLM_CUDA_GRAPH", os.environ)
                self.assertNotIn(
                    "FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH", os.environ)
                self.assertEqual(os.environ["FASTLLM_GPU_TOKEN_HANDOFF"], "1")
                self.assertTrue(args.cuda_embedding)

    def test_single_cuda_device_uses_its_compute_capability(self):
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util._is_nvidia_cuda_platform",
                return_value=True), patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities",
                return_value={3: 75}) as capability_query:
            _configure_qwen35_auto_fast_paths(
                _args(tp="", device="cuda:3"),
                is_qwen35_model=True,
                mtp=0,
            )

            capability_query.assert_called_once_with([3])
            self.assertNotIn("FASTLLM_CUDA_GRAPH", os.environ)

    def test_does_not_auto_enable_graph_for_mixed_sm75_tp(self):
        capabilities = {0: 89, 1: 75}
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util._is_nvidia_cuda_platform",
                return_value=True), patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities",
                return_value=capabilities):
            _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=True, mtp=0)

            self.assertNotIn("FASTLLM_CUDA_GRAPH", os.environ)

    def test_does_not_auto_enable_graph_when_capability_is_unknown(self):
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util._is_nvidia_cuda_platform",
                return_value=True), patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities",
                return_value={}):
            _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=True, mtp=0)

            self.assertNotIn("FASTLLM_CUDA_GRAPH", os.environ)

    def test_cuda_like_non_nvidia_does_not_query_nvidia_driver(self):
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util._is_nvidia_cuda_platform",
                return_value=False), patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities"
                ) as capability_query:
            _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=True, mtp=0)

            capability_query.assert_not_called()
            self.assertEqual(os.environ["FASTLLM_CUDA_GRAPH"], "1")

    def test_explicit_graph_enable_skips_hardware_detection(self):
        overrides = {"FASTLLM_CUDA_GRAPH": "1"}
        with patch.dict(os.environ, overrides, clear=True), patch(
                "fastllm_pytools.util._is_nvidia_cuda_platform"
                ) as vendor_query, patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities"
                ) as capability_query:
            args = _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=True, mtp=0)

            vendor_query.assert_not_called()
            capability_query.assert_not_called()
            self.assertEqual(os.environ["FASTLLM_CUDA_GRAPH"], "1")
            self.assertEqual(
                os.environ["FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH"], "64")
            self.assertTrue(args.cuda_embedding)

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

    def test_does_not_enable_fast_paths_for_dflash(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _configure_qwen35_auto_fast_paths(
                _args(speculative_algorithm="dflash"),
                is_qwen35_model=True,
                mtp=0,
            )

            self.assertNotIn("FASTLLM_CUDA_GRAPH", os.environ)
            self.assertNotIn(
                "FASTLLM_GPU_TOKEN_HANDOFF", os.environ)
            self.assertNotIn(
                "FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH", os.environ)
            self.assertFalse(args.cuda_embedding)

    def test_does_not_change_other_models(self):
        with patch.dict(os.environ, {}, clear=True):
            args = _configure_qwen35_auto_fast_paths(
                _args(), is_qwen35_model=False, mtp=0)

            self.assertEqual(dict(os.environ), {})
            self.assertFalse(args.cuda_embedding)


class NvidiaCudaPlatformDetectionTest(unittest.TestCase):
    def test_recognizes_non_nvidia_build_flags(self):
        for flag in ("USE_ROCM", "USE_IVCOREX"):
            build_info = '{"%s": true}' % flag
            with self.subTest(flag=flag), patch(
                    "builtins.open", mock_open(read_data=build_info)):
                self.assertTrue(_uses_non_nvidia_cuda_compatible_build())

    def test_non_nvidia_build_skips_hardware_probes(self):
        with patch(
                "fastllm_pytools.util._uses_non_nvidia_cuda_compatible_build",
                return_value=True), patch(
                "fastllm_pytools.util.glob.glob") as pci_probe, patch(
                "fastllm_pytools.util.subprocess.run") as vendor_tool:
            self.assertFalse(_is_nvidia_cuda_platform())

            pci_probe.assert_not_called()
            vendor_tool.assert_not_called()

    def test_unknown_platform_does_not_fall_back_to_vendor_tools(self):
        with patch(
                "fastllm_pytools.util._uses_non_nvidia_cuda_compatible_build",
                return_value=False), patch(
                "fastllm_pytools.util.glob.glob", return_value=[]), patch(
                "fastllm_pytools.util.subprocess.run") as vendor_tool:
            self.assertFalse(_is_nvidia_cuda_platform())

            vendor_tool.assert_not_called()

    def test_non_nvidia_pci_device_is_not_misdetected(self):
        for device_class in ("0x030000", "0x120000"):
            def fake_open(path, *args, **kwargs):
                if str(path).endswith("/vendor"):
                    return io.StringIO("0x1234\n")
                if str(path).endswith("/class"):
                    return io.StringIO(device_class + "\n")
                raise FileNotFoundError(path)

            with self.subTest(device_class=device_class), patch(
                    "fastllm_pytools.util."
                    "_uses_non_nvidia_cuda_compatible_build",
                    return_value=False), patch(
                    "fastllm_pytools.util.glob.glob",
                    return_value=["/sys/bus/pci/devices/fake"]), patch(
                    "builtins.open", side_effect=fake_open), patch(
                    "fastllm_pytools.util.subprocess.run") as vendor_tool:
                self.assertFalse(_is_nvidia_cuda_platform())
                vendor_tool.assert_not_called()

    def test_nvidia_gpu_with_bmc_display_is_detected(self):
        def fake_open(path, *args, **kwargs):
            device = os.path.basename(os.path.dirname(str(path)))
            if str(path).endswith("/vendor"):
                vendor = "0x10de" if device == "nvidia" else "0x1a03"
                return io.StringIO(vendor + "\n")
            if str(path).endswith("/class"):
                return io.StringIO("0x030000\n")
            raise FileNotFoundError(path)

        with patch(
                "fastllm_pytools.util._uses_non_nvidia_cuda_compatible_build",
                return_value=False), patch(
                "fastllm_pytools.util.glob.glob",
                return_value=[
                    "/sys/bus/pci/devices/nvidia",
                    "/sys/bus/pci/devices/bmc",
                ]), patch(
                "builtins.open", side_effect=fake_open), patch(
                "fastllm_pytools.util.subprocess.run") as vendor_tool:
            self.assertTrue(_is_nvidia_cuda_platform())
            vendor_tool.assert_not_called()

    def test_mixed_compute_gpu_vendors_are_not_treated_as_nvidia(self):
        def fake_open(path, *args, **kwargs):
            device = os.path.basename(os.path.dirname(str(path)))
            if str(path).endswith("/vendor"):
                vendor = "0x10de" if device == "nvidia" else "0x1234"
                return io.StringIO(vendor + "\n")
            if str(path).endswith("/class"):
                return io.StringIO("0x030000\n")
            raise FileNotFoundError(path)

        with patch(
                "fastllm_pytools.util._uses_non_nvidia_cuda_compatible_build",
                return_value=False), patch(
                "fastllm_pytools.util.glob.glob",
                return_value=[
                    "/sys/bus/pci/devices/nvidia",
                    "/sys/bus/pci/devices/other",
                ]), patch(
                "builtins.open", side_effect=fake_open), patch(
                "fastllm_pytools.util.subprocess.run") as vendor_tool:
            self.assertFalse(_is_nvidia_cuda_platform())
            vendor_tool.assert_not_called()


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


class Sm89Fp8LinearTritonAutoConfigTest(unittest.TestCase):
    def test_enables_only_fp8_linear_on_sm89(self):
        current_python = "/opt/current-venv/bin/python"
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities",
                return_value={0: 89, 1: 89}), patch(
                "fastllm_pytools.util._find_triton_python",
                return_value=current_python), redirect_stdout(output):
            detected = _configure_sm89_fp8_linear_triton(_args())

            self.assertEqual(detected, current_python)
            self.assertEqual(
                os.environ["FASTLLM_CUDA_TRITON_PYTHON"], current_python)
            self.assertEqual(
                os.environ["FASTLLM_CUDA_TRITON_LINEAR_FP8"], "1")
            self.assertNotIn("FASTLLM_CUDA_TRITON", os.environ)
            self.assertIn("enabled automatically", output.getvalue())

    def test_keeps_builtin_cuda_when_triton_is_unavailable(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities",
                return_value={0: 89, 1: 89}), patch(
                "fastllm_pytools.util._find_triton_python",
                return_value=""), redirect_stdout(output):
            detected = _configure_sm89_fp8_linear_triton(_args())

            self.assertEqual(detected, "")
            self.assertNotIn(
                "FASTLLM_CUDA_TRITON_LINEAR_FP8", os.environ)
            self.assertNotIn("FASTLLM_CUDA_TRITON_PYTHON", os.environ)
            self.assertNotIn("FASTLLM_CUDA_TRITON", os.environ)
            self.assertIn("built-in CUDA", output.getvalue())

    def test_skips_other_architectures(self):
        with patch.dict(os.environ, {}, clear=True), patch(
                "fastllm_pytools.util._nvidia_cuda_compute_capabilities",
                return_value={0: 86, 1: 90}), patch(
                "fastllm_pytools.util._find_triton_python") as detector:
            self.assertEqual(
                _configure_sm89_fp8_linear_triton(_args()), "")

            detector.assert_not_called()

    def test_respects_explicit_disable(self):
        for environment in (
                {"FASTLLM_CUDA_TRITON": "0"},
                {"FASTLLM_CUDA_TRITON_LINEAR_FP8": "0"}):
            with self.subTest(environment=environment), patch.dict(
                    os.environ, environment, clear=True), patch(
                    "fastllm_pytools.util._find_triton_python") as detector:
                self.assertEqual(
                    _configure_sm89_fp8_linear_triton(_args()), "")

                detector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
