import gc
import sys
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.fastllm_pytools.openai_server.tool_parsers.abstract_tool_parser import ToolParser
from tools.fastllm_pytools.openai_server.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


class Tokenizer:
    def __init__(self):
        self.tokens = {"<tool_call>": 1, "</tool_call>": 2}
        self.added_tokens_decoder = {}
        self.reads = 0

    def __len__(self):
        raise AssertionError("Do not enumerate the full vocabulary to count it")

    def __bool__(self):
        # HF tokenizers are truthy independently of vocabulary enumeration.
        return True

    def get_vocab(self):
        self.reads += 1
        return dict(self.tokens)


class SharedToolVocabTest(unittest.TestCase):
    def test_concurrent_parsers_share_vocab_but_not_stream_state(self):
        tokenizer = Tokenizer()
        with ThreadPoolExecutor(max_workers=8) as pool:
            parsers = list(pool.map(lambda _: Qwen3CoderToolParser(tokenizer), range(220)))
        self.assertEqual(tokenizer.reads, 1)
        self.assertTrue(all(p.vocab is parsers[0].vocab for p in parsers))
        with self.assertRaises(TypeError):
            parsers[0].vocab["<tool_call>"] = 99
        parsers[0].prev_tool_call_arr.append({"name": "private"})
        self.assertTrue(all(not p.prev_tool_call_arr for p in parsers[1:]))

    def test_different_tokenizers_and_added_tokens(self):
        first, second = Tokenizer(), Tokenizer()
        second.tokens["<tool_call>"] = 7
        old = ToolParser(first).vocab
        self.assertEqual(ToolParser(second).vocab["<tool_call>"], 7)
        first.tokens["new"] = 3
        first.added_tokens_decoder[3] = "new"
        new = ToolParser(first).vocab
        self.assertNotIn("new", old)
        self.assertEqual(new["new"], 3)
        self.assertEqual(first.reads, 2)

    def test_cache_does_not_keep_tokenizer_alive(self):
        tokenizer = Tokenizer()
        ToolParser(tokenizer).vocab
        reference = weakref.ref(tokenizer)
        del tokenizer
        gc.collect()
        self.assertIsNone(reference())

    def test_custom_unhashable_or_non_weakref_tokenizer_falls_back(self):
        class Unhashable(Tokenizer):
            __hash__ = None

        class Slotted:
            __slots__ = ()

            def get_vocab(self):
                return {"token": 1}

        for tokenizer in (Unhashable(), Slotted()):
            self.assertEqual(dict(ToolParser(tokenizer).vocab), tokenizer.get_vocab())


if __name__ == "__main__":
    unittest.main()
