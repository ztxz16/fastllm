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


class FakeGlm5NextModel:
    force_chat_template = False
    tool_call_parser = "auto"

    def _is_glm5_next(self):
        return True

    def get_type(self):
        return "glm5_next"


def completion():
    instance = FastLLmCompletion.__new__(FastLLmCompletion)
    instance.model = FakeGlm5NextModel()
    return instance


class Glm5NextReasoningTest(unittest.TestCase):
    def test_official_template_is_always_tagged_reasoning(self):
        self.assertTrue(completion()._uses_tagged_reasoning_response(True))
        self.assertTrue(completion()._uses_tagged_reasoning_response(False))

    def test_non_stream_splits_reasoning_and_content(self):
        content, reasoning = completion()._split_glm5_next_reasoning(
            "先计算。 </think>答案是 2。", True)

        self.assertEqual(reasoning, "先计算。 ")
        self.assertEqual(content, "答案是 2。")

    def test_truncated_thinking_is_reasoning_only(self):
        content, reasoning = completion()._split_glm5_next_reasoning(
            "尚未完成的思考", True)

        self.assertEqual(content, "")
        self.assertEqual(reasoning, "尚未完成的思考")

    def test_streaming_is_stable_at_every_byte_boundary(self):
        wire = "先计算。</think>答案是 2。"
        for chunk_size in (1, 2, 5, len(wire)):
            with self.subTest(chunk_size=chunk_size):
                state = {
                    "active": True,
                    "buffer": "",
                    "started": False,
                    "format": "think",
                    "phase": "reasoning",
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

                self.assertEqual(reasoning, "先计算。")
                self.assertEqual(content, "答案是 2。")

    def test_effort_defaults_to_max_and_rejects_unsupported_value(self):
        request = ChatCompletionRequest(
            model="glm-5.3-flash",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(
            completion()._resolve_glm5_next_reasoning_effort(request), "max")

        request.reasoning_effort = "high"
        effort = completion()._resolve_glm5_next_reasoning_effort(request)
        self.assertEqual(effort, "high")
        self.assertEqual(
            completion()._resolve_chat_template_kwargs(request, None, effort),
            {"reasoning_effort": "high"},
        )

        request.reasoning_effort = "medium"
        with self.assertRaisesRegex(ValueError, "low, high, max"):
            completion()._resolve_glm5_next_reasoning_effort(request)

    def test_tool_constraint_uses_glm_xml_wire_name(self):
        request = ChatCompletionRequest(
            model="glm-5.3-flash",
            messages=[{"role": "user", "content": "查天气"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                },
            }],
        )

        descriptor = completion()._build_tool_call_constraint_descriptor(
            request)

        self.assertEqual(descriptor.model_type, "glm47")
        self.assertEqual(descriptor.constraint_type, "glm47_tool_call")


if __name__ == "__main__":
    unittest.main()
