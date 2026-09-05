#!/usr/bin/env python3
"""Unit tests for desktop shared-library classification."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from desktop.collect_libraries import find_nss_libraries, is_driver_library


class DriverLibraryClassificationTest(unittest.TestCase):
    def test_host_driver_libraries_are_recognized(self):
        names = (
            "libcuda.so.1",
            "libcudadebugger.so.1",
            "libEGL_nvidia.so.0",
            "libGLX_nvidia.so.0",
            "libnvcuvid.so.1",
            "libnvidia-ml.so.1",
            "libnvoptix.so.1",
            "libvdpau_nvidia.so.1",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(is_driver_library(name))

    def test_redistributable_cuda_libraries_are_not_drivers(self):
        names = (
            "libcublas.so.13",
            "libcudart.so.12",
            "libnccl.so.2",
            "libnvJitLink.so.12",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertFalse(is_driver_library(name))

    def test_nss_modules_in_private_directory_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core = root / "libnss3.so"
            core.touch()
            private = root / "nss"
            private.mkdir()
            names = ("libsoftokn3.so", "libfreebl3.so", "libfreeblpriv3.so", "libnssckbi.so")
            for name in names:
                (private / name).touch()
            with patch("desktop.collect_libraries.find_optional_libraries", return_value={"libnss3.so": core}):
                self.assertEqual(find_nss_libraries(), {name: private / name for name in names})
                (private / "libsoftokn3.so").unlink()
                with self.assertRaisesRegex(RuntimeError, "libsoftokn3"):
                    find_nss_libraries()


if __name__ == "__main__":
    unittest.main()
