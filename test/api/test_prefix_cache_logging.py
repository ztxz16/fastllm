#!/usr/bin/env python3
"""CPU checks for the user-facing persistent-prefix log policy."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "fastllm_pytools"))
import util
from prefix_cache_logging import configure_prefix_cache_logging, prefix_cache_log_level


class PrefixCacheLogging(unittest.TestCase):
    def setUp(self):
        self.parser = util.make_normal_parser("prefix-cache logging")

    def test_default_is_info(self):
        args = self.parser.parse_args(["unused-model"])
        self.assertEqual(args.prefix_cache_log_level, "info")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configure_prefix_cache_logging(args), "info")
            self.assertEqual(os.environ["FASTLLM_PREFIX_CACHE_LOG_LEVEL"], "info")

    def test_cli_levels(self):
        for level in ("off", "info", "debug"):
            args = self.parser.parse_args(["unused-model", "--prefix_cache_log_level", level])
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(configure_prefix_cache_logging(args), level)
                self.assertEqual(prefix_cache_log_level(), level)

    def test_invalid_level_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["unused-model", "--prefix_cache_log_level", "verbose"])


if __name__ == "__main__":
    unittest.main()
