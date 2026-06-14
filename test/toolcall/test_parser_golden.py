#!/usr/bin/env python3
import unittest
from pathlib import Path

from lib.golden import (
    DEFAULT_CHUNK_SIZES,
    get_parser,
    load_jsonl,
    run_non_stream_case,
    run_stream_case,
)


CASES_DIR = Path(__file__).resolve().parent / "cases"


class ToolParserGoldenTest(unittest.TestCase):
    def test_deepseek_v4_dsml_golden_cases(self):
        cases_path = CASES_DIR / "deepseek_v4_dsml.jsonl"
        for case in load_jsonl(cases_path):
            modes = case.get("mode", ["non_stream", "stream"])

            if "non_stream" in modes:
                with self.subTest(id=case["id"], mode="non_stream"):
                    parser = get_parser(case["parser"])
                    run_non_stream_case(self, parser, case)

            if "stream" in modes:
                chunk_sizes = case.get("stream", {}).get(
                    "chunk_sizes", DEFAULT_CHUNK_SIZES)
                for chunk_size in chunk_sizes:
                    with self.subTest(id=case["id"],
                                      mode="stream",
                                      chunk_size=chunk_size):
                        parser = get_parser(case["parser"])
                        run_stream_case(self, parser, case, chunk_size)


if __name__ == "__main__":
    unittest.main()
