"""Context errors must reach clients without becoming generated text."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_qwen35_reasoning import (
    FakeQwen35Model, RawRequest, completion, request,
)
from tools.fastllm_pytools.generation_errors import PromptTooLongError
from tools.fastllm_pytools.openai_server.protocal.anthropic_protocol import (
    AnthropicMessageRequest,
)
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (
    ErrorResponse,
)


class ContextErrorModel(FakeQwen35Model):
    def stream_response_handle_async(self, handle):
        async def generate():
            if self.output:
                yield self.output
            raise PromptTooLongError()
        return generate()


class ContextLengthAPITest(unittest.IsolatedAsyncioTestCase):
    async def test_context_error_is_reported_with_and_without_thinking(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                for thinking in (False, True):
                    for partial in ("", "partial output"):
                        with self.subTest(api=api, stream=stream,
                                          thinking=thinking, partial=partial):
                            instance = completion(ContextErrorModel(partial))
                            instance.enable_thinking = thinking
                            # Another request may reuse the integer handle
                            # freed by the native terminal error.
                            instance.conversation_handles["new-request"] = 101
                            if api == "openai":
                                result = await instance.create_chat_completion(
                                    request(stream=stream), RawRequest())
                            else:
                                result = await instance.create_anthropic_message(
                                    AnthropicMessageRequest(
                                        model="qwen3.5", max_tokens=128,
                                        stream=stream,
                                        messages=[{"role": "user", "content": "answer"}]),
                                    RawRequest())
                            if stream:
                                generator, background = result
                                events = []
                                async for chunk in generator:
                                    for line in chunk.splitlines():
                                        if line.startswith("data: ") and line != "data: [DONE]":
                                            event = json.loads(line[6:])
                                            events.append(event)
                                            if "error" in event:
                                                self.assertEqual(
                                                    instance.conversation_handles,
                                                    {"new-request": 101})
                                errors = [event["error"] for event in events
                                          if "error" in event]
                                self.assertEqual(len(errors), 1)
                                self.assertEqual(errors[0]["message"], str(PromptTooLongError()))
                                self.assertFalse(any(
                                    choice.get("finish_reason")
                                    for event in events
                                    for choice in event.get("choices", [])))
                                await background()
                            else:
                                self.assertIsInstance(result, ErrorResponse)
                                self.assertEqual(result.code, 400)
                                self.assertEqual(result.message, str(PromptTooLongError()))
                            # FakeQwen35Model.abort_handle fails if called.
                            self.assertEqual(instance.conversation_handles,
                                             {"new-request": 101})

    async def test_literal_prompt_too_long_is_still_model_text(self):
        for stream in (False, True):
            instance = completion(FakeQwen35Model("prompt too long"))
            instance.enable_thinking = False
            result = await instance.create_chat_completion(
                request(stream=stream), RawRequest())
            if stream:
                generator, background = result
                wire = "".join([chunk async for chunk in generator])
                self.assertNotIn('"error"', wire)
                self.assertIn("prompt too long", wire)
                await background()
            else:
                self.assertEqual(result.choices[0].message.content,
                                 "prompt too long")


class NativeContextLengthTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        package = Path(__file__).resolve().parents[2] / "tools" / "fastllm_pytools"
        if not any((package / name).exists() for name in (
                "libfastllm_tools.so", "libfastllm_tools-cu11.so",
                "libfastllm_tools-cpu.so", "fastllm_tools.dll",
                "libfastllm_tools.dylib")):
            raise unittest.SkipTest("FastLLM native library is not available")
        from tools.fastllm_pytools import llm
        cls.llm = llm

    def make_model(self, hf, save_history):
        model = self.llm.model.__new__(self.llm.model)
        model.model = 123
        model.save_history = save_history
        model.current_tokenizer_cache = {7: [["prompt"], [[1]]]} if save_history else {}
        model.tokenizer_cache = Mock()
        model._can_apply_hf_chat_template = lambda: hf
        model._uses_hf_deepseek_v4_tokenizer = lambda: False
        model.hf_tokenizer = SimpleNamespace(decode=lambda tokens: "prompt too long")
        model._decode_fastllm_token = lambda token: b"prompt too long"
        return model

    async def test_native_error_raises_and_discards_failed_tokenizer_cache(self):
        for hf in (False, True):
            for save_history in (False, True):
                for asynchronous in (False, True):
                    with self.subTest(hf=hf, save_history=save_history,
                                      asynchronous=asynchronous):
                        model = self.make_model(hf, save_history)
                        native = Mock()
                        native.can_fetch_response_llm_model.return_value = True
                        native.fetch_response_llm_model.return_value = -2
                        native.fetch_response_tokens_batch_llm_model.return_value = -2
                        with patch.object(self.llm, "fastllm_lib", native):
                            with self.assertRaises(PromptTooLongError):
                                if asynchronous:
                                    _ = [chunk async for chunk in model.stream_response_handle_async(7)]
                                else:
                                    list(model.stream_response_handle(7))
                        self.assertNotIn(7, model.current_tokenizer_cache)
                        model.tokenizer_cache.add.assert_not_called()
                        native.abort_response_llm_model.assert_not_called()

    async def test_native_normal_text_and_end_are_unchanged(self):
        for hf in (False, True):
            for save_history in (False, True):
                for asynchronous in (False, True):
                    with self.subTest(hf=hf, save_history=save_history,
                                      asynchronous=asynchronous):
                        model = self.make_model(hf, save_history)
                        native = Mock()
                        native.can_fetch_response_llm_model.return_value = True
                        native.fetch_response_llm_model.side_effect = [42, -1]
                        batches = iter([42, -1])

                        def fetch_batch(model_id, handle, buffer, capacity):
                            token = next(batches)
                            buffer[0] = token
                            return 1 if token >= 0 else token

                        native.fetch_response_tokens_batch_llm_model.side_effect = fetch_batch
                        with patch.object(self.llm, "fastllm_lib", native):
                            if asynchronous:
                                chunks = [chunk async for chunk in model.stream_response_handle_async(7)]
                            else:
                                chunks = list(model.stream_response_handle(7))
                        self.assertEqual("".join(chunks), "prompt too long")


if __name__ == "__main__":
    unittest.main()
