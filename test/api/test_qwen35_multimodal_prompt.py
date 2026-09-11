import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools")))

from fastllm_pytools.qwen35_multimodal_native import build_qwen35_prompt


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, conversation, **kwargs):
        self.kwargs = kwargs
        return "<|im_start|>assistant\n"


class Qwen35MultimodalPromptTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
