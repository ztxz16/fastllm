import copy
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools")))

from fastllm_pytools.qwen35_multimodal_native import (
    build_qwen35_prompt, normalize_qwen35_conversation,
)


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, conversation, **kwargs):
        self.conversation = conversation
        self.kwargs = kwargs
        return "<|im_start|>assistant\n"


class Qwen35MultimodalPromptTest(unittest.TestCase):
    def test_multimodal_tool_history_survives_normalization_and_rendering(self):
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": "private image payload"},
                {"type": "text", "text": "Inspect both images."},
            ]},
            {"role": "assistant", "content": None,
             "reasoning_content": "Inspect the second image.", "tool_calls": [{
                 "id": "call_image", "type": "function", "function": {
                     "name": "read_image", "arguments": {"path": "second.png"},
                 },
             }]},
            {"role": "tool", "tool_call_id": "call_image", "name": "read_image",
             "content": [{"type": "text", "text": "Second image:"},
                         {"type": "image", "image": "second image payload"}]},
        ]
        original = copy.deepcopy(conversation)
        normalized = normalize_qwen35_conversation(conversation, image_count=2, video_count=0)
        tokenizer = FakeTokenizer()
        build_qwen35_prompt(
            tokenizer=tokenizer, conversation=normalized, image_grid_thw=None,
            video_grid_thw=None, video_timestamps=None, merge_size=2,
            add_generation_prompt=True, enable_thinking=False,
        )
        self.assertEqual(tokenizer.conversation[1], original[1])
        self.assertEqual(tokenizer.conversation[2]["tool_call_id"], "call_image")
        self.assertEqual(tokenizer.conversation[2]["name"], "read_image")
        self.assertEqual(tokenizer.conversation[0]["content"][0], {"type": "image"})
        self.assertEqual(tokenizer.conversation[2]["content"][1], {"type": "image"})
        self.assertEqual(conversation, original)

    def prompt(self, tokenizer, **kwargs):
        return build_qwen35_prompt(
            tokenizer=tokenizer,
            conversation=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            image_grid_thw=None, video_grid_thw=None, video_timestamps=None,
            merge_size=2, add_generation_prompt=True, enable_thinking=False,
            **kwargs,
        )

    def test_tools_and_template_kwargs_reach_multimodal_prompt(self):
        tokenizer = FakeTokenizer()
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        build_qwen35_prompt(
            tokenizer=tokenizer,
            conversation=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            image_grid_thw=None,
            video_grid_thw=None,
            video_timestamps=None,
            merge_size=2,
            add_generation_prompt=True,
            enable_thinking=True,
            tools=tools,
            tool_choice="auto",
            chat_template_kwargs={"reasoning_effort": "medium"},
        )
        self.assertEqual(tokenizer.kwargs["tools"], tools)
        self.assertEqual(tokenizer.kwargs["tool_choice"], "auto")
        self.assertEqual(tokenizer.kwargs["reasoning_effort"], "medium")
        self.assertTrue(tokenizer.kwargs["enable_thinking"])

    def test_absent_tools_are_not_injected(self):
        tokenizer = FakeTokenizer()
        build_qwen35_prompt(
            tokenizer=tokenizer,
            conversation=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            image_grid_thw=None,
            video_grid_thw=None,
            video_timestamps=None,
            merge_size=2,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.assertNotIn("tools", tokenizer.kwargs)
        self.assertNotIn("tool_choice", tokenizer.kwargs)
        self.assertFalse(tokenizer.kwargs["enable_thinking"])

    def test_unsupported_tool_choice_preserves_tools_and_thinking(self):
        class Tokenizer:
            def apply_chat_template(self, conversation, tokenize=False,
                                    add_generation_prompt=True, tools=None, enable_thinking=True):
                self.tools = tools
                self.enable_thinking = enable_thinking
                return "prompt"

        tokenizer = Tokenizer()
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        self.assertEqual(self.prompt(tokenizer, tools=tools, tool_choice="auto"), "prompt")
        self.assertEqual(tokenizer.tools, tools)
        self.assertFalse(tokenizer.enable_thinking)

    def test_template_internal_type_error_is_not_retried(self):
        class Tokenizer:
            calls = 0

            def apply_chat_template(self, conversation, **kwargs):
                self.calls += 1
                if kwargs.get("tools"):
                    raise TypeError("template implementation failed")
                return "silently lost tools"

        tokenizer = Tokenizer()
        with self.assertRaisesRegex(TypeError, "template implementation failed"):
            self.prompt(tokenizer, tools=[{"type": "function"}])
        self.assertEqual(tokenizer.calls, 1)

    def test_unknown_signature_does_not_retry_internal_error(self):
        tokenizer = FakeTokenizer()
        with patch("fastllm_pytools.qwen35_multimodal_native.inspect.signature", side_effect=ValueError), patch.object(
            tokenizer, "apply_chat_template", side_effect=TypeError("opaque failure")
        ) as call:
            with self.assertRaisesRegex(TypeError, "opaque failure"):
                self.prompt(tokenizer, tools=[{"type": "function"}])
            self.assertEqual(call.call_count, 1)

    def test_missing_tools_support_fails_explicitly(self):
        class Tokenizer:
            def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):
                return "legacy prompt"

        with self.assertRaisesRegex(ValueError, "accepts tools"):
            self.prompt(Tokenizer(), tools=[{"type": "function"}])
        self.assertEqual(self.prompt(Tokenizer()), "legacy prompt")
        self.assertEqual(self.prompt(Tokenizer(), tools=[]), "legacy prompt")

    def test_native_fallback_rejects_tools_and_preserves_plain_prompt(self):
        with self.assertRaisesRegex(ValueError, "native fallback does not support tools"):
            self.prompt(None, tools=[{"type": "function"}])
        with self.assertRaisesRegex(ValueError, "native fallback does not support tools"):
            self.prompt(None, chat_template_kwargs={"tools": [{"type": "function"}]})
        self.assertEqual(self.prompt(None, tools=[]),
                         "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_template_kwargs_cannot_change_tokenization_or_generation_boundary(self):
        tokenizer = FakeTokenizer()
        self.prompt(tokenizer, chat_template_kwargs={
            "tokenize": True, "add_generation_prompt": False, "enable_thinking": True,
            "reasoning_effort": "xhigh",
        })
        self.assertFalse(tokenizer.kwargs["tokenize"])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertEqual(tokenizer.kwargs["reasoning_effort"], "xhigh")


if __name__ == "__main__":
    unittest.main()
