import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.util import make_normal_llm_model, make_normal_parser


class SpeculativeDraftCliAliasesTest(unittest.TestCase):
    def configure_without_target(self, argv):
        args = make_normal_parser("test").parse_args(argv)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            make_normal_llm_model(args)
        self.assertEqual(error.exception.code, 0)
        return args

    def write_draft_config(self, directory, config):
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as file:
            json.dump(config, file)

    def test_short_draft_aliases(self):
        draft_path = "/models/qwen-dflash2"

        for option in ("--draft", "--draft_model_path"):
            with self.subTest(option=option):
                args = make_normal_parser("test").parse_args(
                    [option, draft_path]
                )
                self.assertEqual(
                    args.speculative_draft_model_path,
                    draft_path,
                )

    def test_draft_tokens_is_positive(self):
        args = make_normal_parser("test").parse_args(
            ["--draft_tokens", "5"]
        )
        self.assertEqual(args.draft_tokens, 5)

    def test_dflash_uses_checkpoint_default_when_unspecified(self):
        with tempfile.TemporaryDirectory() as draft_path:
            self.write_draft_config(draft_path, {
                "architectures": ["DFlash2DraftModel"],
                "dflash_config": {"block_size": 8},
            })
            with patch.dict(os.environ, {}, clear=True):
                self.configure_without_target(["--draft", draft_path])
                self.assertEqual(os.environ["FASTLLM_DFLASH_BLOCK_SIZE"], "8")

    def test_dflash_converts_actual_draft_tokens_to_block_size(self):
        with tempfile.TemporaryDirectory() as draft_path:
            self.write_draft_config(draft_path, {
                "architectures": ["DFlash2DraftModel"],
                "dflash_config": {"block_size": 8},
            })
            with patch.dict(os.environ, {}, clear=True):
                self.configure_without_target([
                    "--draft", draft_path,
                    "--draft_tokens", "5",
                ])
                self.assertEqual(os.environ["FASTLLM_DFLASH_BLOCK_SIZE"], "6")

    def test_external_dspark_caps_draft_tokens(self):
        with tempfile.TemporaryDirectory() as draft_path:
            self.write_draft_config(draft_path, {
                "architectures": ["DSparkDraftModel"],
                "block_size": 7,
            })
            with patch.dict(os.environ, {}, clear=True):
                self.configure_without_target([
                    "--draft", draft_path,
                    "--draft_tokens", "4",
                ])
                self.assertEqual(os.environ["FASTLLM_DSPARK_TOKENS"], "4")

    def test_external_dspark_uses_checkpoint_default_when_unspecified(self):
        with tempfile.TemporaryDirectory() as draft_path:
            self.write_draft_config(draft_path, {
                "architectures": ["DSparkDraftModel"],
                "block_size": 7,
            })
            with patch.dict(os.environ, {}, clear=True):
                self.configure_without_target(["--draft", draft_path])
                self.assertNotIn("FASTLLM_DSPARK_TOKENS", os.environ)

    def test_embedded_dspark_accepts_draft_tokens(self):
        with patch.dict(os.environ, {}, clear=True):
            args = self.configure_without_target([
                "--draft_tokens", "9",
            ])
            self.assertEqual(args.speculative_algorithm, "dspark")
            self.assertEqual(os.environ["FASTLLM_DSPARK_TOKENS"], "9")


if __name__ == "__main__":
    unittest.main()
