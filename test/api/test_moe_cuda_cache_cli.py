import argparse
import os
import sys
import unittest


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.util import _memory_size_bytes, make_normal_parser


class MoeCudaCacheCliTest(unittest.TestCase):
    def test_disabled_by_default(self):
        args = make_normal_parser("test").parse_args([])
        self.assertEqual(args.moe_cuda_cache, 0)

    def test_binary_size_units(self):
        expected = {
            "3g": 3 << 30,
            "3GiB": 3 << 30,
            "512m": 512 << 20,
            "1.5gb": 3 << 29,
            "4096": 4096,
            "0": 0,
        }
        for value, bytes_ in expected.items():
            with self.subTest(value=value):
                self.assertEqual(_memory_size_bytes(value), bytes_)

    def test_both_option_spellings(self):
        for option in ("--moe_cuda_cache", "--moe-cuda-cache"):
            with self.subTest(option=option):
                args = make_normal_parser("test").parse_args([option, "3g"])
                self.assertEqual(args.moe_cuda_cache, 3 << 30)

    def test_invalid_sizes_are_rejected(self):
        for value in ("-1g", "nan", "inf", "1e300g", "3t", "",
                      "1e-300g", "0.5", 1 << 64, str(1 << 64),
                      "18446744073709551615.1", "17179869184g"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _memory_size_bytes(value)

    def test_exact_large_byte_counts(self):
        maximum = (1 << 64) - 1
        for value in (maximum, str(maximum), str(maximum - 1)):
            with self.subTest(value=value):
                self.assertEqual(_memory_size_bytes(value), int(value))
        self.assertEqual(_memory_size_bytes("9007199254740993"), (1 << 53) + 1)
        self.assertEqual(_memory_size_bytes("0.000000000931322574615478515625g"), 1)


if __name__ == "__main__":
    unittest.main()
