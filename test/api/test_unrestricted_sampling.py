"""Non-positive API top-k must reach the backend as full-vocabulary sampling."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

from test_qwen35_reasoning import FakeQwen35Model, completion, RawRequest
from tools.fastllm_pytools.openai_server.protocal.anthropic_protocol import AnthropicMessageRequest
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import ChatCompletionRequest


class SamplingModel(FakeQwen35Model):
    config = {"vocab_size": 152576}
    def get_vocab_size(self):
        return self.config["vocab_size"]

    def launch_stream_response(self, query, **kwargs):
        self.sampling_args = kwargs
        return super().launch_stream_response(query, **kwargs)


class UnrestrictedSamplingTest(unittest.IsolatedAsyncioTestCase):
    async def test_sampling_forwarding_for_both_apis(self):
        for anthropic in (False, True):
            for top_k, temperature, expected, sample in ((0, 1, 152576, True),
                    (-1, 1, 152576, True), (1, 1, 1, True), (50, 1, 50, True),
                    (0, 0, 1, False)):
                with self.subTest(anthropic=anthropic, top_k=top_k, temperature=temperature):
                    model = SamplingModel()
                    instance = completion(model)
                    req_type = AnthropicMessageRequest if anthropic else ChatCompletionRequest
                    req = req_type(model=instance.model_name, messages=[{'role': 'user', 'content': 'answer'}],
                                   top_k=top_k, temperature=temperature, top_p=.95, max_tokens=128)
                    method = instance.create_anthropic_message if anthropic else instance.create_chat_completion
                    await method(req, RawRequest())
                    self.assertEqual(model.sampling_args['top_k'], expected)
                    self.assertEqual(model.sampling_args['do_sample'], sample)
                    self.assertEqual(model.sampling_args['top_p'], .95 if sample else 1.)

    async def test_server_default_zero_is_normalized(self):
        model = SamplingModel()
        model.default_generation_config = {**model.default_generation_config, 'top_k': 0}
        instance = completion(model)
        await instance.create_anthropic_message(AnthropicMessageRequest(
            model=instance.model_name, messages=[{'role':'user','content':'answer'}], max_tokens=128), RawRequest())
        self.assertEqual(model.sampling_args['top_k'], 152576)

class NativeVocabularyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = Path(__file__).resolve().parents[2] / "tools" / "fastllm_pytools"
        if not any((package / name).exists() for name in (
                "libfastllm_tools.so", "libfastllm_tools-cu11.so",
                "libfastllm_tools-cpu.so", "fastllm_tools.dll", "libfastllm_tools.dylib")):
            raise unittest.SkipTest("FastLLM native library is not available")
        from tools.fastllm_pytools import llm
        cls.llm = llm

    def test_vocab_bounds(self):
        llm = self.llm
        for config, expected in (({'vocab_size': 128}, 128),
                                 ({'text_config': {'vocab_size': 256}}, 256)):
            model = SimpleNamespace(config=config, hf_tokenizer=None)
            self.assertEqual(llm.model.get_vocab_size(model), expected)
        tokenizer = SimpleNamespace(get_vocab=lambda: {'a': 0, 'added': 7})
        self.assertEqual(llm.model.get_vocab_size(SimpleNamespace(config={}, hf_tokenizer=tokenizer)), 8)
        with patch.object(llm.fastllm_lib, 'get_tokenizer_vocab_size', return_value=128):
            self.assertEqual(llm.model.get_vocab_size(SimpleNamespace(config={}, hf_tokenizer=None, model=1)), 128)


if __name__ == '__main__':
    unittest.main()
