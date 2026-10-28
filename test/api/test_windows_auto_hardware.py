"""Windows auto planning uses CUDA and physical RAM without Linux files."""
import ctypes
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import auto_config, hybrid_auto


class WindowsAutoHardwareTest(unittest.TestCase):
    def test_cuda_driver_uses_windows_abi_and_visible_ordinals(self):
        driver = MagicMock()

        def write(pointer, value):
            pointer._obj.value = value
            return 0

        driver.cuInit.return_value = 0
        driver.cuDeviceGetCount.side_effect = lambda p: write(p, 1)
        driver.cuDeviceGet.side_effect = lambda p, ordinal: write(p, ordinal)
        attributes = {16: 82, 13: 1695000, 36: 9751000, 37: 384, 75: 8, 76: 6}
        driver.cuDeviceGetAttribute.side_effect = lambda p, key, device: write(p, attributes[key])
        driver.cuDeviceGetPCIBusId.return_value = 1
        with patch.object(auto_config.os, "name", "nt"), \
                patch.object(ctypes, "WinDLL", return_value=driver, create=True) as load, \
                patch.object(auto_config, "Path") as paths:
            hardware, peers = auto_config.cuda_hardware()
        load.assert_called_once_with("nvcuda.dll")
        paths.assert_not_called()
        self.assertEqual(hardware[0]["capability"], 86)
        self.assertEqual(hardware[0]["sm"], 82)
        self.assertEqual(hardware[0]["memory_bandwidth"], 936096000000)
        self.assertEqual(peers, {(0, 0): True})

    def test_memory_budget_excludes_page_file(self):
        kernel = MagicMock()

        def query(pointer):
            status = pointer._obj
            self.assertEqual(status.length, 64)
            status.total_phys = 32 << 30
            status.avail_phys = 20 << 30
            status.total_page = 1024 << 30
            status.avail_page = 900 << 30
            return 1

        kernel.GlobalMemoryStatusEx.side_effect = query
        with patch.object(hybrid_auto.os, "name", "nt"), \
                patch.object(ctypes, "WinDLL", return_value=kernel, create=True) as load:
            ram = hybrid_auto.host_memory()
        load.assert_called_once_with("kernel32", use_last_error=True)
        self.assertEqual(ram, {"total": 32 << 30, "available": 20 << 30, "cma": 0})

    def test_missing_windows_driver_reports_actual_library(self):
        with patch.object(auto_config.os, "name", "nt"), \
                patch.object(ctypes, "WinDLL", side_effect=OSError("missing"), create=True):
            with self.assertRaisesRegex(ValueError, "nvcuda.dll"):
                auto_config.cuda_hardware()

    def test_failed_memory_query_does_not_create_a_budget(self):
        kernel = MagicMock()
        kernel.GlobalMemoryStatusEx.return_value = 0
        with patch.object(hybrid_auto.os, "name", "nt"), \
                patch.object(ctypes, "WinDLL", return_value=kernel, create=True), \
                patch.object(ctypes, "get_last_error", return_value=123, create=True), \
                patch.object(ctypes, "WinError", return_value=OSError("memory query failed"), create=True):
            with self.assertRaisesRegex(OSError, "memory query failed"):
                hybrid_auto.host_memory()


if __name__ == "__main__":
    unittest.main()
