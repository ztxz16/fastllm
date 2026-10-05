import argparse
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.util import _memory_size_bytes, make_normal_parser, make_normal_llm_model


class MoeCudaCacheCliTest(unittest.TestCase):
    def test_disabled_by_default(self):
        args = make_normal_parser("test").parse_args([])
        self.assertEqual(args.moe_cuda_cache, 0)
        self.assertEqual(args.moe_cpu_cache, 0)

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

    def test_disk_cache_budgets_are_independent(self):
        for option in ("--moe_cpu_cache", "--moe-cpu-cache"):
            args = make_normal_parser("test").parse_args([
                "--moe_device", "disk", "--moe_cuda_cache", "3g", option, "32g"])
            self.assertEqual(args.moe_cuda_cache, 3 << 30)
            self.assertEqual(args.moe_cpu_cache, 32 << 30)

    def test_exact_large_byte_counts(self):
        maximum = (1 << 64) - 1
        for value in (maximum, str(maximum), str(maximum - 1)):
            with self.subTest(value=value):
                self.assertEqual(_memory_size_bytes(value), int(value))
        self.assertEqual(_memory_size_bytes("9007199254740993"), (1 << 53) + 1)
        self.assertEqual(_memory_size_bytes("0.000000000931322574615478515625g"), 1)

    def test_policy_defaults_and_aliases(self):
        defaults = make_normal_parser("test").parse_args([])
        self.assertEqual(defaults.moe_cache_half_life, 128)
        self.assertEqual(defaults.moe_cache_update_interval, 1)
        self.assertEqual(defaults.moe_cache_max_replacements, 96)
        self.assertEqual(defaults.moe_cache_prefill_prior, 0)
        for separator in ("_", "-"):
            args = make_normal_parser("test").parse_args([
                "--" + "moe_cache_half_life".replace("_", separator), "32",
                "--" + "moe_cache_max_bytes".replace("_", separator), "30m",
                "--" + "moe_cache_rank_by_bytes".replace("_", separator)])
            self.assertEqual(args.moe_cache_half_life, 32)
            self.assertEqual(args.moe_cache_max_bytes, 30 << 20)
            self.assertTrue(args.moe_cache_rank_by_bytes)

    def test_invalid_policy_values(self):
        for option, values in {
            "half_life": ["-1", "nan", "inf", "1e100"],
            "update_interval": ["0", "-1", "2147483648"],
            "max_replacements": ["-1", "2147483648"],
            "factor": ["0.9", "nan"],
            "margin": ["-1", "inf"],
            "prefill_prior": ["-0.1", "1.1", "nan"],
        }.items():
            for value in values:
                with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        make_normal_parser("test").parse_args(["--moe_cache_" + option, value])

    def test_policy_reaches_runtime_before_model_creation(self):
        with tempfile.TemporaryDirectory() as path:
            with open(os.path.join(path, "config.json"), "w") as config:
                json.dump({"model_type": "llama"}, config)
            runtime = MagicMock()
            model = runtime.model.return_value
            model.get_max_input_len.return_value = 4096
            model.get_max_batch.return_value = 1
            expected = dict(half_life=32., update_interval=4, max_replacements=12,
                            max_bytes=30 << 20, min_heat=2., margin=.5, factor=1.25,
                            min_residence=3, prefill_prior=.25, rank_by_bytes=True)
            runtime.model.side_effect = lambda *a, **k: (
                runtime.set_moe_cache_policy.assert_called_once_with(**expected) or model)
            args = make_normal_parser("test").parse_args([
                path, "--device", "cpu", "--threads", "1", "--dtype", "float16",
                "--moe_cache_half_life", "32", "--moe_cache_update_interval", "4",
                "--moe_cache_max_replacements", "12", "--moe_cache_max_bytes", "30m",
                "--moe_cache_min_heat", "2", "--moe_cache_margin", ".5",
                "--moe_cache_factor", "1.25", "--moe_cache_min_residence", "3",
                "--moe_cache_prefill_prior", ".25", "--moe_cache_rank_by_bytes"])
            with patch.dict(os.environ, {}, clear=True), \
                    patch.dict(sys.modules, {"ftllm": types.SimpleNamespace(llm=runtime)}), \
                    redirect_stdout(io.StringIO()):
                self.assertIs(make_normal_llm_model(args), model)


if __name__ == "__main__":
    unittest.main()
