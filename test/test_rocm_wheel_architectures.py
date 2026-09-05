"""Architecture selection must follow SDK metadata without claiming extra GPUs."""
import importlib.util
from pathlib import Path
import sys
import unittest


spec = importlib.util.spec_from_file_location(
    'rocm_wheel_builder', Path(__file__).resolve().parents[1] / 'tools/scripts/build_rocm_wheel.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ArchitectureSelectionTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'The ROCm wheel builder targets Linux')
    def test_device_packages_respect_platform_and_extras(self):
        targets = builder.device_targets([
            'rocm-sdk-core==10.0.0',
            'rocm-sdk-device-gfx1100==10.0.0; extra == "device-gfx1100"',
            'ROCM_SDK_DEVICE_GFX942==10.0.0; sys_platform == "linux" and extra == "device-all"',
            'rocm-sdk-device-gfx1010==10.0.0; sys_platform == "win32" and extra == "device-all"',
            'rocm-sdk-device-gfx950==10.0.0; extra == "unrelated"',
        ])
        self.assertEqual(targets, {'gfx1100', 'gfx942'})

    def test_all_includes_new_sdk_targets_without_editing_a_static_list(self):
        self.assertEqual(builder.select_architectures('all', {'gfx1310', 'gfx90a', 'gfx908'}),
                         ['gfx908', 'gfx90a', 'gfx1310'])

    def test_explicit_subset_keeps_order_and_removes_duplicates(self):
        self.assertEqual(builder.select_architectures(' gfx1100, gfx942;gfx1100 ',
                                                     {'gfx1100', 'gfx942', 'gfx950'}),
                         ['gfx1100', 'gfx942'])

    def test_no_silent_drop_of_unsupported_or_malformed_requests(self):
        for value in ('gfx906', 'gfx1100;gfx906', '', 'all;gfx1100', 'gfx1100;', 'native'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                builder.select_architectures(value, {'gfx1100'})


if __name__ == '__main__':
    unittest.main()
