import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools.util import apply_multimodal_warmup_env, make_normal_parser
from fastllm_pytools.qwen35_multimodal_native import _compute_video_grid, get_qwen35_multimodal_config


class MultimodalWarmupTest(unittest.TestCase):
    def test_default_retains_lazy_behavior(self):
        args = make_normal_parser("test").parse_args([])
        self.assertFalse(args.multimodal)
        with patch.dict(os.environ, {"FASTLLM_QWEN35_MM_MAX_PATCHES": "65536"}, clear=True):
            apply_multimodal_warmup_env(args, False)
            self.assertNotIn("FASTLLM_QWEN35_MM_MAX_PATCHES", os.environ)

    def test_processor_limits_cover_images_and_video(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            model = Path(folder)
            (model / "config.json").write_text(json.dumps({"vision_config": {
                "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2}}))
            (model / "preprocessor_config.json").write_text(json.dumps({
                "size": {"longest_edge": 4096 ** 2}}))
            args = make_normal_parser("test").parse_args(["--multimodal", "--path", folder])
            apply_multimodal_warmup_env(args, True)
            self.assertEqual(os.environ["FASTLLM_QWEN35_MM_MAX_PATCHES"], "65536")
            (model / "video_preprocessor_config.json").write_text(json.dumps({
                "size": {"longest_edge": 4096 ** 2 * 4}}))
            apply_multimodal_warmup_env(args, True)
            self.assertEqual(os.environ["FASTLLM_QWEN35_MM_MAX_PATCHES"], "262144")
            grid, _ = _compute_video_grid(
                [SimpleNamespace(shape=(8192, 8192, 3))], get_qwen35_multimodal_config(folder))
            self.assertLessEqual(int(grid.prod()), int(os.environ["FASTLLM_QWEN35_MM_MAX_PATCHES"]))
            # GGUF uses the processor/config beside --ori, just like request preprocessing.
            args.ori = folder
            args.path = str(model / "model.gguf")
            apply_multimodal_warmup_env(args, True)
            self.assertEqual(os.environ["FASTLLM_QWEN35_MM_MAX_PATCHES"], "262144")

    def test_unsupported_model_fails_before_loading(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Qwen3.5/Qwen3.8"):
                apply_multimodal_warmup_env(argparse.Namespace(multimodal=True), False)


if __name__ == "__main__":
    unittest.main()
