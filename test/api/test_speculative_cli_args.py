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

    def write_mtp_checkpoint(self, directory):
        self.write_draft_config(directory, {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "mtp_num_hidden_layers": 1,
        })
        path = os.path.join(directory, "mtp.safetensors")
        with open(path, "wb") as file:
            file.write(b"")
        return path

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

    def test_mtp_directory_is_auto_detected(self):
        with tempfile.TemporaryDirectory() as draft_path:
            self.write_mtp_checkpoint(draft_path)
            stale_speculative_env = {
                "FASTLLM_DSPARK_MODEL_PATH": "/stale/dspark",
                "FASTLLM_DSPARK_TOKENS": "4",
                "FASTLLM_DSPARK_CONFIDENCE_THRESHOLD": "0.5",
                "FASTLLM_DFLASH_MODEL_PATH": "/stale/dflash",
                "FASTLLM_DFLASH_BLOCK_SIZE": "8",
            }
            with patch.dict(os.environ, stale_speculative_env, clear=True):
                args = self.configure_without_target(["--draft", draft_path])
                self.assertEqual(args.speculative_algorithm, "mtp")
                self.assertEqual(args.mtp, 5)
                self.assertEqual(args.speculative_draft_model_path,
                                 os.path.abspath(draft_path))
                for env_name in stale_speculative_env:
                    self.assertNotIn(env_name, os.environ)

    def test_mtp_file_is_auto_detected_and_uses_draft_tokens(self):
        with tempfile.TemporaryDirectory() as draft_path:
            mtp_path = self.write_mtp_checkpoint(draft_path)
            with patch.dict(os.environ, {}, clear=True):
                args = self.configure_without_target([
                    "--draft", mtp_path,
                    "--draft_tokens", "7",
                ])
                self.assertEqual(args.speculative_algorithm, "mtp")
                self.assertEqual(args.mtp, 7)
                self.assertEqual(args.speculative_draft_model_path,
                                 os.path.abspath(mtp_path))

    def test_mtp_rejects_conflicting_token_counts(self):
        with tempfile.TemporaryDirectory() as draft_path:
            self.write_mtp_checkpoint(draft_path)
            args = make_normal_parser("test").parse_args([
                "--draft", draft_path,
                "--mtp", "4",
                "--draft_tokens", "5",
            ])
            with self.assertRaisesRegex(ValueError, "different MTP draft counts"):
                make_normal_llm_model(args)

    def test_external_mtp_requires_qwen35_config(self):
        with tempfile.TemporaryDirectory() as draft_path:
            self.write_draft_config(draft_path, {
                "model_type": "llama",
                "mtp_num_hidden_layers": 1,
            })
            with open(os.path.join(draft_path, "mtp.safetensors"), "wb") as file:
                file.write(b"")
            args = make_normal_parser("test").parse_args([
                "--draft", draft_path,
            ])
            with self.assertRaisesRegex(ValueError,
                                        "must be Qwen3.5"):
                make_normal_llm_model(args)


if __name__ == "__main__":
    unittest.main()
