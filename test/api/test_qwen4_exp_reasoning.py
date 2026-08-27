#!/usr/bin/env python3
import copy
import json
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
from tools.fastllm_pytools.openai_server.tool_parsers import (  # noqa: E402
    ToolParserManager,
)
sys.path[:] = ORIGINAL_SYS_PATH


class FakeQwen4ExpModel:
    default_generation_config = {
        "repetition_penalty": 1.0,
        "top_p": 0.8,
        "top_k": 1,
        "temperature": 1.0,
    }
    force_chat_template = False
    tool_call_parser = "auto"
    hf_tokenizer = None

    def __init__(self, output="Qwen4 thought</think>Qwen4 answer"):
        self.output = output
        self.input_kwargs = None
        self.launch_kwargs = None

    def get_type(self):
        return "qwen4_exp"

    def _is_deepseek_v4(self):
        return False

    def get_input_token_len(
        self,
        messages,
        enable_thinking=False,
        tools=None,
        thinking_effort=None,
        tool_choice=None,
        chat_template_kwargs=None,
    ):
        self.input_kwargs = {
            "enable_thinking": enable_thinking,
            "chat_template_kwargs": copy.deepcopy(chat_template_kwargs),
        }
        return 19

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
        chat_template_kwargs=None,
    ):
        self.launch_kwargs = {
            "enable_thinking": enable_thinking,
            "chat_template_kwargs": copy.deepcopy(chat_template_kwargs),
        }
        return 404

    def stream_response_handle_async(self, handle):
        async def generator():
            yield self.output

        return generator()

    def abort_handle(self, handle):
        raise AssertionError("abort_handle should not be called")


class DummyQwenTokenizer:
    chat_template = "<tool_call><function=NAME>"

    def get_vocab(self):
        return {"<tool_call>": 248058, "</tool_call>": 248059}


class RawRequest:
    async def is_disconnected(self):
        return False


def completion(model=None):
    instance = FastLLmCompletion.__new__(FastLLmCompletion)
    instance.model_name = "qwen4-exp"
    instance.model = model or FakeQwen4ExpModel()
    instance.think = False
    instance.enable_thinking = True
    instance.hide_input = True
    instance.conversation_handles = {}
    return instance


def request(**kwargs):
    values = {
        "model": "qwen4-exp",
        "messages": [{"role": "user", "content": "answer"}],
        "max_tokens": 128,
    }
    values.update(kwargs)
    return ChatCompletionRequest(**values)


class Qwen4ExpReasoningTest(unittest.IsolatedAsyncioTestCase):
    def test_native_reasoning_efforts_and_tagged_response(self):
        instance = completion()
        self.assertTrue(instance._is_qwen4_exp_model())
        self.assertTrue(instance._uses_tagged_reasoning_response(True))
        self.assertFalse(instance._uses_tagged_reasoning_response(False))
        self.assertEqual(
            instance._resolve_qwen3_5_reasoning_effort(request()), "xhigh")
        for effort in ("low", "medium", "xhigh"):
            with self.subTest(effort=effort):
                self.assertEqual(
                    instance._resolve_qwen3_5_reasoning_effort(
                        request(reasoning_effort=effort)),
                    effort,
                )

    async def test_non_stream_request_splits_reasoning_and_content(self):
        model = FakeQwen4ExpModel()
        response = await completion(model).create_chat_completion(
            request(reasoning_effort="low"), RawRequest())

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertEqual(
            model.input_kwargs["chat_template_kwargs"],
            {"reasoning_effort": "low"},
        )
        self.assertEqual(
            model.launch_kwargs["chat_template_kwargs"],
            {"reasoning_effort": "low"},
        )
        self.assertEqual(
            response.choices[0].message.reasoning_content, "Qwen4 thought")
        self.assertEqual(response.choices[0].message.content, "Qwen4 answer")

    def test_xml_tool_parser_is_selected_and_parses_call(self):
        tokenizer = DummyQwenTokenizer()
        parser_class = ToolParserManager.get_tool_parser_auto(
            "qwen4_exp", tokenizer.chat_template)
        self.assertEqual(parser_class.__name__, "Qwen3CoderToolParser")

        tool_request = request(tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }])
        output = (
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\n北京\n</parameter>\n"
            "</function>\n</tool_call>")
        parsed = parser_class(tokenizer).extract_tool_calls(
            output, tool_request)

        self.assertTrue(parsed.tools_called)
        self.assertEqual(parsed.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(parsed.tool_calls[0].function.arguments),
            {"city": "北京"},
        )


if __name__ == "__main__":
    unittest.main()
