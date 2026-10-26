"""GGUF vision metadata and imported-weight capacity accounting."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import deepseek_v41_gguf as vision, gguf_metadata, hybrid_auto as memory
from fastllm_pytools.util import make_normal_parser
from test_deepseek_v41_gguf import write_fixture


class VisionTest(unittest.TestCase):
    def setUp(self):
        self.target = dict(gguf_architecture="deepseek41", model_type="deepseek_v41",
                           hidden_size=32, num_hidden_layers=4, max_position_embeddings=1024,
                           n_routed_experts=2, num_experts_per_tok=1, moe_intermediate_size=16,
                           num_attention_heads=4, head_dim=8, index_head_dim=8,
                           compress_ratios=[0, 0, 2, 1], kv_source_layer_ids=[2, 3],
                           index_source_layer_ids=[2], sliding_window=8)
        self.metadata = {"general.architecture": "clip", "clip.projector_type": "deepseek41v",
                         "clip.vision.projection_dim": 32, "clip.vision.embedding_length": 16,
                         "clip.vision.feed_forward_length": 64, "clip.vision.block_count": 2,
                         "clip.vision.attention.head_count": 4, "clip.vision.patch_size": 14,
                         "clip.vision.projector.scale_factor": 2}

    def test_reads_vision_configuration_from_gguf(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "mmproj.gguf")
            write_fixture(path, self.metadata, "v.post_ln.weight")
            config = vision.vision_config(path, self.target)["vision_config"]
            self.assertEqual(config["hidden_size"], 16)
            self.assertEqual(config["num_hidden_layers"], 2)
            self.assertEqual(config["downsample_ratio"], 2)

    def test_rejects_incompatible_projectors(self):
        for key, value in (("general.architecture", "dflash"), ("clip.projector_type", "other"),
                           ("clip.vision.projection_dim", 64), ("clip.vision.patch_size", 0)):
            with self.subTest(key=key), patch.object(vision, "read_gguf_metadata",
                                                   return_value={**self.metadata, key: value}):
                with self.assertRaises(ValueError):
                    vision.vision_config("unused.gguf", self.target)
        with patch.object(vision, "read_gguf_metadata", return_value=self.metadata):
            with self.assertRaises(ValueError):
                vision.vision_config("unused.gguf", {**self.target, "gguf_architecture": "qwen3"})

    def test_image_token_is_read_from_backbone_tokenizer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "model.gguf")
            write_fixture(path, {"general.architecture": "deepseek41",
                                 "deepseek41.block_count": 4, "deepseek41.embedding_length": 32,
                                 "tokenizer.ggml.tokens": ["hello", "<｜deepseek_image｜>", "world"]},
                          "output.weight")
            config = gguf_metadata.get_gguf_model_config(path)
            self.assertEqual(config["image_token_id"], 1)
            self.assertEqual(config["vocab_size"], 3)

    def test_planner_includes_projector_and_converted_allocations(self):
        args = make_normal_parser("test").parse_args(["-p", "model.gguf", "--mmproj", "mmproj.gguf"])
        experts = [memory.Tensor("blk.0.ffn_gate_exps.weight", (2, 16, 32), 544, "gguf:16")]
        tensors = [memory.Tensor("mm.1.weight", (32, 32), 1088, "gguf:8"),
                   memory.Tensor("v.post_ln.weight", (16,), 32, "gguf:1")]
        with patch.object(memory, "inventory", side_effect=[(self.target.copy(), experts.copy()),
                                                            ({}, tensors)]), \
                patch.object(vision, "read_gguf_metadata", return_value=self.metadata):
            config, combined, model = memory.model_inputs(args)
        self.assertEqual(config["vision_config"]["num_hidden_layers"], 2)
        self.assertEqual([t.name for t in combined[1:]],
                         ["vision.mm.1.weight", "vision.v.post_ln.weight"])
        with_vision = memory.weight_costs(combined, model, args)
        args.mmproj = ""
        baseline = memory.weight_costs(experts, model, args)
        converted_bytes = 32 * 32 * 2 + 16 * 4
        self.assertEqual(with_vision[3] - baseline[3], converted_bytes)
        self.assertEqual(with_vision[2] - baseline[2], converted_bytes + 2 * memory.GiB)


if __name__ == "__main__":
    unittest.main()
