"""Token-exact chat preparation without a GPU model or checkpoint."""
from array import array
import os
import pickle
import unittest
from unittest.mock import patch

try:
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
except ImportError as error:
    raise unittest.SkipTest("requires Hugging Face fast tokenizers") from error

from tools.fastllm_pytools.chat_input import ChatEncoder, ChatInputPool
from tools.fastllm_pytools.chat_template import apply_hf_chat_template


def tokenizer(prefix_space=False):
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=prefix_space)
    backend.train_from_iterator(["hello 你好"], trainers.BpeTrainer(
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    backend.add_special_tokens([AddedToken(token, special=True, normalized=False)
                                for token in ("<turn>", "@end@", "<ends>", "<ends>extra")])
    return PreTrainedTokenizerFast(tokenizer_object=backend, chat_template=(
        "{% for m in messages %}<turn>{{ m.role }} {{ m.content }}@end@{% endfor %}"))


class ChatEncoderTest(unittest.TestCase):
    def test_capability_and_exact_reuse(self):
        for prefix_space in (False, True):
            tok = tokenizer(prefix_space)
            encoder = ChatEncoder(tok, "any", 2500)
            if prefix_space:
                self.assertFalse(encoder.markers)
            else:
                self.assertIn("<turn>", encoder.markers.values())
                self.assertNotIn("<ends>", encoder.markers.values())
            prefix = {"role": "system", "content": "😀e\u0301"}
            for index, content in enumerate(("A", "A中", "<ends>", "<ends>extra", "B", "x" * 10000, "C")):
                messages = [prefix, {"role": "user", "content": content}]
                old = encoder.entries.get("session")
                with patch.object(encoder, "tokenizer", wraps=tok) as call:
                    result = encoder.encode(messages, {}, "session")
                    if old and index == 1:
                        self.assertNotIn(prefix["content"], call.call_args.args[0])
                expected = apply_hf_chat_template(tok, messages, tokenize=True)
                self.assertEqual(result.tolist(), expected)
                self.assertLessEqual(encoder.used, encoder.budget)
                if index == 2:
                    prefix = {"role": "system", "content": "changed"}
            # Tokenizer control kwargs require the original full encode.
            options = {"chat_template_kwargs": {"truncation": True, "max_length": 3}}
            result = encoder.encode(messages, options, "session")
            self.assertEqual(result.tolist(), apply_hf_chat_template(tok, messages, tokenize=True, **options))


@unittest.skipUnless(os.name == "posix", "requires POSIX descriptor passing")
class WorkerTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_returns_native_ids_and_recovers_after_exit(self):
        tok = tokenizer()
        pool = ChatInputPool(pickle.dumps((tok, "other", 10000)), 1)
        self.addCleanup(pool.close)
        messages = [{"role": "user", "content": "first"}]
        expected = array("i", apply_hf_chat_template(tok, messages, tokenize=True))
        self.assertEqual(await pool.prepare(messages, {}, "abc"), expected)
        lane = pool.lanes[0]
        lane.proc.kill()
        lane.proc.wait()
        # The caller executes the original preparation for the interrupted call.
        with self.assertLogs(level="ERROR"):
            self.assertIsNone(await pool.prepare(messages, {}, "abc"))
        self.assertEqual(await pool.prepare(messages, {}, "abc"), expected)
