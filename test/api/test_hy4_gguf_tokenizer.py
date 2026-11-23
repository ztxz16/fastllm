import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import gguf_metadata


@unittest.skipUnless(importlib.util.find_spec("tokenizers") and importlib.util.find_spec("transformers"),
                     "requires tokenizers and transformers")
class Hy4GGUFTokenizerTest(unittest.TestCase):
    def test_hy4_split_order_and_merge_boundaries(self):
        tokens = [".", "f", "n", ".f", ".fn", "1", "2", "3", "4", "5", "12", "123", "1234", "45"]
        metadata = {
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "hy_v4",
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.merges": [". f", ".f n", "1 2", "12 3", "123 4", "4 5"],
        }
        with patch.object(gguf_metadata, "read_gguf_metadata", return_value=metadata):
            tokenizer = gguf_metadata.try_load_gguf_tokenizer("fixture.gguf")
        self.assertIsNotNone(tokenizer)
        text = "a一あb.fn12345"
        splits = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        self.assertEqual([text[start:end] for _, (start, end) in splits], ["a", "一あ", "b", ".fn", "123", "45"])
        self.assertEqual(tokenizer.encode(".fn12345", add_special_tokens=False),
                         [tokens.index(part) for part in (".fn", "123", "45")])
