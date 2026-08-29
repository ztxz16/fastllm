#!/usr/bin/env python3
import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_API_DIR = Path(__file__).resolve().parent
ORIGINAL_SYS_PATH = list(sys.path)
sys.path = [
    path for path in sys.path
    if Path(path or os.getcwd()).resolve() != TEST_API_DIR
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fastllm_pytools.openai_server.fastllm_completion import (  # noqa: E402
    FastLLmCompletion,
)
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionResponse,
)
sys.path[:] = ORIGINAL_SYS_PATH


class FakeDotsModel:
    default_generation_config = {
        "repetition_penalty": 1.0,
        "top_p": 0.8,
        "top_k": 1,
        "temperature": 1.0,
    }
    force_chat_template = False
    tool_call_parser = "auto"
    hf_tokenizer = None

    def __init__(self, output="<think>brief thought</think>42"):
        self.output = output

    def get_type(self):
        return "dots3_note"

    def _is_deepseek_v4(self):
        return False

    def get_input_token_len(self, messages, enable_thinking=False):
        return 17

    def launch_stream_response(
        self,
        query,
        max_length=8192,
        min_length=0,
        do_sample=True,
        top_p=0.8,
        top_k=1,
        temperature=1.0,
        repeat_penalty=1.0,
        tools=None,
        one_by_one=True,
        enable_thinking=None,
        images=None,
        videos=None,
        stop_token_ids=None,
    ):
        return 101

    def stream_response_handle_async(self, handle):
        async def generator():
            yield self.output

        return generator()

    def abort_handle(self, handle):
        raise AssertionError("abort_handle should not be called")


class RawRequest:
    async def is_disconnected(self):
        return False


def completion(model=None):
    instance = FastLLmCompletion.__new__(FastLLmCompletion)
    instance.model_name = "dots3-note"
    instance.model = model or FakeDotsModel()
    instance.think = False
    instance.enable_thinking = True
    instance.hide_input = True
    instance.conversation_handles = {}
    return instance


def request(**kwargs):
    values = {
        "model": "dots3-note",
        "messages": [{"role": "user", "content": "answer"}],
        "max_tokens": 128,
    }
    values.update(kwargs)
    return ChatCompletionRequest(**values)


class DotsReasoningTest(unittest.IsolatedAsyncioTestCase):
    def test_reasoning_response_requires_native_template_and_thinking(self):
        instance = completion()
        self.assertTrue(instance._uses_tagged_reasoning_response(True))
        self.assertFalse(instance._uses_tagged_reasoning_response(False))
        instance.model.force_chat_template = True
        self.assertFalse(instance._uses_tagged_reasoning_response(True))

    def test_streaming_is_safe_at_every_byte_boundary(self):
        wire = "<think>think step</think>final answer"
        for chunk_size in (1, 2, 7, len(wire)):
            with self.subTest(chunk_size=chunk_size):
                state = {
                    "active": True,
                    "buffer": "",
                    "started": False,
                    "format": "think",
                }
                reasoning = ""
                content = ""
                for offset in range(0, len(wire), chunk_size):
                    messages, content_delta = (
                        completion()._consume_tagged_reasoning_delta(
                            wire[offset:offset + chunk_size], state))
                    reasoning += "".join(
                        message.reasoning_content or ""
                        for message in messages)
                    content += content_delta

                self.assertEqual(reasoning, "think step")
                self.assertEqual(content, "final answer")

    async def test_non_stream_response_separates_reasoning_and_content(self):
        response = await completion().create_chat_completion(
            request(), RawRequest())

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertEqual(response.choices[0].message.reasoning_content,
                         "brief thought")
        self.assertEqual(response.choices[0].message.content, "42")

    async def test_truncated_thinking_is_reasoning_only(self):
        response = await completion(FakeDotsModel(
            "<think>unfinished thought")).create_chat_completion(
                request(), RawRequest())

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertEqual(response.choices[0].message.reasoning_content,
                         "unfinished thought")
        self.assertEqual(response.choices[0].message.content, "")


if __name__ == "__main__":
    unittest.main()
