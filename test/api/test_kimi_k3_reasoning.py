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
    tool_call_parser = "auto"
    force_chat_template = ""

    def _is_kimi_k3(self):
        return True

    def get_type(self):
        return "kimi_k3"


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

    def test_auto_tool_guidance_is_added_to_existing_system_message(self):
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Create index.html."},
        ]
        tools = [{"type": "function", "function": {"name": "write"}}]

        guided = completion()._apply_kimi_k3_auto_tool_guidance(
            messages, tools, "auto")

        self.assertEqual([message["role"] for message in guided],
                         ["system", "user"])
        self.assertIn("call the appropriate available tools",
                      guided[0]["content"])
        self.assertEqual(messages[0]["content"], "You are a coding agent.")

    def test_auto_tool_guidance_preserves_explicit_tool_choice(self):
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "write"}}]

        self.assertIs(
            completion()._apply_kimi_k3_auto_tool_guidance(
                messages, tools, "required"),
            messages,
        )

    def test_required_tool_guidance_strengthens_latest_user_action(self):
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "写一首绝句到文件"},
        ]
        tools = [{"type": "function", "function": {"name": "write"}}]

        guided = completion()._apply_kimi_k3_required_tool_guidance(
            messages, tools, "required")

        self.assertIn("必须立即调用合适的工具", guided[-1]["content"])
        self.assertIn("纯文本回复无效", guided[-1]["content"])
        self.assertIn("直接调用 write 工具", guided[-1]["content"])
        self.assertIn("filePath 和完整 content", guided[-1]["content"])
        self.assertEqual(messages[-1]["content"], "写一首绝句到文件")

    def test_explicit_file_action_uses_kimi_native_required_mode(self):
        tools = [{"type": "function", "function": {"name": "write"}}]

        self.assertEqual(
            completion()._resolve_kimi_k3_auto_tool_choice(
                [{"role": "user",
                  "content": "帮我写一个html的贪吃蛇"}],
                tools,
                "auto",
            ),
            "required",
        )
        self.assertEqual(
            completion()._resolve_kimi_k3_auto_tool_choice(
                [{"role": "user", "content": "写一首绝句到文件"}],
                tools,
                "auto",
            ),
            "required",
        )
        self.assertEqual(
            completion()._resolve_kimi_k3_auto_tool_choice(
                [{"role": "user", "content": "Please create index.html."}],
                tools,
                "auto",
            ),
            "required",
        )

    def test_auto_tool_choice_keeps_greetings_and_explanations_optional(self):
        tools = [{"type": "function", "function": {"name": "write"}}]

        for content in ("你好", "如何写一个 HTML 文件？"):
            self.assertEqual(
                completion()._resolve_kimi_k3_auto_tool_choice(
                    [{"role": "user", "content": content}],
                    tools,
                    "auto",
                ),
                "auto",
            )

    def test_tool_result_does_not_keep_kimi_required_mode_active(self):
        tools = [{"type": "function", "function": {"name": "write"}}]
        messages = [
            {"role": "user", "content": "Please create index.html."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "write:0",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": '{"filePath":"index.html","content":"ok"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "write:0",
                "content": "Wrote file successfully.",
            },
        ]

        self.assertEqual(
            completion()._resolve_kimi_k3_auto_tool_choice(
                messages, tools, "auto"),
            "auto",
        )

    def test_repeated_schema_errors_disable_more_kimi_tool_calls(self):
        tools = [{"type": "function", "function": {"name": "write"}}]
        messages = [
            {"role": "user", "content": "写一首绝句到文件"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {
                "role": "tool",
                "tool_call_id": "write:0",
                "content": "SchemaError(Missing key at [\"content\"])",
            },
            {"role": "assistant", "content": "", "tool_calls": []},
            {
                "role": "tool",
                "tool_call_id": "write:1",
                "content": "The write tool was called with invalid arguments.",
            },
        ]

        self.assertEqual(
            completion()._resolve_kimi_k3_auto_tool_choice(
                messages, tools, "auto"),
            "none",
        )

    def test_effective_required_tool_choice_is_used_by_response_parser(self):
        request = ChatCompletionRequest(
            model="kimi-k3",
            messages=[{"role": "user", "content": "写一首绝句到文件"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object"},
                },
            }],
            tool_choice="auto",
        )

        effective_request = completion()._with_effective_tool_choice(
            request, "required")
        parsed = completion()._create_function_call_parser(
            effective_request).parse_non_stream("已经完成。")

        self.assertEqual(request.tool_choice, "auto")
        self.assertEqual(effective_request.tool_choice, "required")
        self.assertTrue(parsed.has_invalid_tool_block)
        self.assertEqual(
            [diagnostic.code for diagnostic in parsed.diagnostics],
            ["tool_choice_violation"],
        )

    def test_default_sampling_switches_only_inside_kimi_tool_call(self):
        request = ChatCompletionRequest(
            model="kimi-k3",
            messages=[{"role": "user", "content": "Create index.html."}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object"},
                },
            }],
        )

        constraint = completion()._build_tool_call_generation_constraint(
            request)

        self.assertEqual(constraint["content_sampling"], {
            "type": "tool_content_sampling",
            "format": "kimi_k3_xtml",
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 1.0,
        })

    def test_explicit_sampling_disables_kimi_tool_sampling_switch(self):
        request = ChatCompletionRequest(
            model="kimi-k3",
            messages=[{"role": "user", "content": "Create index.html."}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {"type": "object"},
                },
            }],
            top_k=5,
        )

        constraint = completion()._build_tool_call_generation_constraint(
            request)

        self.assertNotIn("content_sampling", constraint)


if __name__ == "__main__":
    unittest.main()
