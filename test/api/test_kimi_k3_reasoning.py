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
)
sys.path[:] = ORIGINAL_SYS_PATH


THINK_CLOSE = "<|close|>think<|sep|>"
RESPONSE_OPEN = "<|open|>response<|sep|>"
RESPONSE_CLOSE = "<|close|>response<|sep|>"
MESSAGE_CLOSE = "<|close|>message<|sep|>"


class FakeKimiK3Model:
    def _is_kimi_k3(self):
        return True


def completion():
    instance = FastLLmCompletion.__new__(FastLLmCompletion)
    instance.model = FakeKimiK3Model()
    return instance


class KimiK3ReasoningTest(unittest.TestCase):
    def test_non_thinking_response_removes_xtml(self):
        wire = "42" + RESPONSE_CLOSE + MESSAGE_CLOSE

        self.assertEqual(
            completion()._strip_kimi_k3_response_wrapper(wire), "42")

    def test_non_stream_splits_reasoning_and_removes_xtml(self):
        wire = (
            "I should calculate carefully."
            + THINK_CLOSE
            + RESPONSE_OPEN
            + "42"
            + RESPONSE_CLOSE
            + MESSAGE_CLOSE
        )

        content, reasoning = completion()._split_kimi_k3_reasoning(
            wire, emit_reasoning_content=True, preserve_xtml=False)

        self.assertEqual(reasoning, "I should calculate carefully.")
        self.assertEqual(content, "42")
        self.assertNotIn("<|close|>", content)

    def test_truncated_thinking_is_reasoning_only(self):
        content, reasoning = completion()._split_kimi_k3_reasoning(
            "unfinished thought", True, False)

        self.assertEqual(content, "")
        self.assertEqual(reasoning, "unfinished thought")

    def test_streaming_is_safe_at_every_byte_boundary(self):
        wire = (
            "think step"
            + THINK_CLOSE
            + RESPONSE_OPEN
            + "final answer"
            + RESPONSE_CLOSE
            + MESSAGE_CLOSE
        )
        for chunk_size in (1, 2, 7, len(wire)):
            with self.subTest(chunk_size=chunk_size):
                state = {
                    "active": True,
                    "buffer": "",
                    "format": "kimi_k3",
                    "phase": "reasoning",
                    "reasoning_started": False,
                    "preserve_xtml": False,
                    "content_buffer": "",
                    "content_started": False,
                    "content_done": False,
                }
                reasoning = ""
                content = ""
                for offset in range(0, len(wire), chunk_size):
                    chunk = wire[offset:offset + chunk_size]
                    messages, content_delta = (
                        completion()._consume_kimi_k3_reasoning_delta(
                            chunk, state))
                    reasoning += "".join(
                        message.reasoning_content or ""
                        for message in messages)
                    content += content_delta

                self.assertEqual(reasoning, "think step")
                self.assertEqual(content, "final answer")
                self.assertNotIn("<|", reasoning + content)

    def test_non_thinking_streaming_is_safe_at_every_byte_boundary(self):
        wire = "final answer" + RESPONSE_CLOSE + MESSAGE_CLOSE
        for chunk_size in (1, 2, 7, len(wire)):
            with self.subTest(chunk_size=chunk_size):
                state = {
                    "active": False,
                    "buffer": "",
                    "format": "kimi_k3",
                    "phase": "content",
                    "reasoning_started": False,
                    "preserve_xtml": False,
                    "content_buffer": "",
                    "content_started": False,
                    "content_done": False,
                }
                content = ""
                for offset in range(0, len(wire), chunk_size):
                    chunk = wire[offset:offset + chunk_size]
                    messages, content_delta = (
                        completion()._consume_tagged_reasoning_delta(
                            chunk, state))
                    self.assertEqual(messages, [])
                    content += content_delta

                self.assertEqual(content, "final answer")
                self.assertNotIn("<|", content)

    def test_effort_defaults_to_max_and_rejects_unsupported_value(self):
        request = ChatCompletionRequest(
            model="kimi-k3",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(
            completion()._resolve_kimi_k3_reasoning_effort(request), "max")

        request.reasoning_effort = "medium"
        with self.assertRaisesRegex(ValueError, "low, high, max"):
            completion()._resolve_kimi_k3_reasoning_effort(request)


if __name__ == "__main__":
    unittest.main()
