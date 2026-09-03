#!/usr/bin/env python3
"""Unit tests for desktop shared-library classification."""

import unittest

from desktop.collect_libraries import is_driver_library


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


if __name__ == "__main__":
    unittest.main()
