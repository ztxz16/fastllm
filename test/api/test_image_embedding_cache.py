import argparse
import copy
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools")))
from fastllm_pytools.qwen35_multimodal_native import build_qwen35_multimodal_payload
from fastllm_pytools.util import apply_image_embedding_cache_env, make_normal_parser


class ImageEmbeddingCacheTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES": str(512 << 20)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.inputs = {
            "image_arrays": [np.arange(48, dtype=np.uint8).reshape(4, 4, 3)],
            "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
            "multimodal_config": {"merge_size": 2, "image_mean": [0.5] * 3},
        }
        self.config = {"image_token_id": 1, "video_token_id": 2,
                       "vision_start_token_id": 3, "vision_end_token_id": 4}

    def payload(self, inputs):
        config, data = build_qwen35_multimodal_payload(inputs, None, self.config)
        tensors = {}
        end = 0
        for item in config["tensors"]:
            self.assertEqual(item["offset_bytes"], end)
            end += item["nbytes"]
            tensor = np.frombuffer(data[item["offset_bytes"]:end], dtype=item["dtype"]).reshape(item["shape"])
            tensors.setdefault(item["name"], []).append(tensor)
        self.assertEqual(end, len(data))
        return tensors

    def key(self, inputs):
        return self.payload(inputs)["image_cache_keys"][0].tobytes()

    def test_same_pixels_stable_and_payload_unchanged(self):
        self.assertEqual(self.key(self.inputs), self.key(copy.deepcopy(self.inputs)))
        self.assertEqual(len(self.key(self.inputs)), 32)
        tensors = self.payload(self.inputs)
        np.testing.assert_array_equal(tensors["image_frames"][0], self.inputs["image_arrays"][0])
        self.assertEqual(tensors["image_frames"][0].dtype, np.float32)

    def test_content_shape_grid_and_processing_change_identity(self):
        baseline = self.key(self.inputs)
        changed = copy.deepcopy(self.inputs)
        changed["image_arrays"][0][0, 0, 0] += 1
        self.assertNotEqual(baseline, self.key(changed))
        changed = copy.deepcopy(self.inputs)
        changed["image_arrays"][0] = changed["image_arrays"][0].reshape(2, 8, 3)
        self.assertNotEqual(baseline, self.key(changed))
        changed = copy.deepcopy(self.inputs)
        changed["image_grid_thw"][0] = [1, 1, 4]
        self.assertNotEqual(baseline, self.key(changed))
        changed = copy.deepcopy(self.inputs)
        changed["multimodal_config"]["image_mean"][0] = 0.25
        self.assertNotEqual(baseline, self.key(changed))

    def test_equivalent_layout_and_processor_order_keep_identity(self):
        inputs = copy.deepcopy(self.inputs)
        inputs["image_arrays"][0] = np.asfortranarray(
            inputs["image_arrays"][0], dtype=np.float32
        )
        inputs["multimodal_config"] = dict(
            reversed(list(inputs["multimodal_config"].items()))
        )
        self.assertEqual(self.key(self.inputs), self.key(inputs))

    def test_each_image_has_independent_key_and_order(self):
        a = self.inputs["image_arrays"][0]
        b = a + 1
        inputs = copy.deepcopy(self.inputs)
        inputs["image_arrays"] = [a, b, a]
        inputs["image_grid_thw"] = np.repeat(inputs["image_grid_thw"], 3, axis=0)
        keys = self.payload(inputs)["image_cache_keys"][0]
        np.testing.assert_array_equal(keys[0], keys[2])
        self.assertNotEqual(keys[0].tobytes(), keys[1].tobytes())
        self.assertEqual(keys[0].tobytes(), self.key(self.inputs))

    def test_disabled_does_not_hash_or_send_keys(self):
        with patch.dict(os.environ, {"FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES": "0"}), patch(
            "fastllm_pytools.qwen35_multimodal_native.hashlib.sha256",
            side_effect=AssertionError("disabled cache must not hash"),
        ):
            self.assertNotIn("image_cache_keys", self.payload(self.inputs))

    def test_video_only_does_not_get_image_keys(self):
        self.assertNotIn("image_cache_keys", self.payload({
            "video_arrays": [np.zeros((2, 4, 4, 3), dtype=np.uint8)],
            "video_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
        }))

    def test_text_only_does_not_hash_or_send_keys(self):
        with patch(
            "fastllm_pytools.qwen35_multimodal_native.hashlib.sha256",
            side_effect=AssertionError("text-only requests must not hash images"),
        ):
            self.assertNotIn("image_cache_keys", self.payload({}))

    def test_cli_capacity_and_environment(self):
        parser = make_normal_parser("test")
        self.assertIsNone(parser.parse_args([]).image_embedding_cache)
        apply_image_embedding_cache_env(parser.parse_args([]))
        self.assertEqual(os.environ["FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES"], str(512 << 20))
        for option in ("--image-embedding-cache", "--image_embedding_cache"):
            args = parser.parse_args([option, "1g"])
            apply_image_embedding_cache_env(args)
            self.assertEqual(os.environ["FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES"], str(1 << 30))
        apply_image_embedding_cache_env(argparse.Namespace(image_embedding_cache=0))
        self.assertEqual(os.environ["FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES"], "0")


if __name__ == "__main__":
    unittest.main()
