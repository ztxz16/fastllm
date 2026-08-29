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
    ErrorResponse,
)
sys.path[:] = ORIGINAL_SYS_PATH


def weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def time_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get time.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        },
    }


WEATHER_CALL = (
    "<dots_function_call>\n"
    '<invoke name="get_weather">\n'
    '<parameter name="city">\n北京\n</parameter>\n'
    "</invoke>\n"
    "</dots_function_call>"
)


class DummyTokenizer:
    chat_template = "<dots_function_call>"


class FakeDotsToolModel:
    default_generation_config = {
        "repetition_penalty": 1.0,
        "top_p": 0.8,
        "top_k": 1,
        "temperature": 1.0,
    }
    force_chat_template = False
    tool_call_parser = "auto"
    hf_tokenizer = DummyTokenizer()

    def __init__(self, output):
        self.output = output
        self.launch_called = False
        self.launch_messages = None
        self.launch_tools = None
        self.received_constraint = None

    def get_type(self):
        return "dots3_note"

    def _is_deepseek_v4(self):
        return False

    def get_input_token_len(self, messages, **kwargs):
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
        tool_call_constraint=None,
    ):
        self.launch_called = True
        self.launch_messages = copy.deepcopy(query)
        self.launch_tools = copy.deepcopy(tools)
        self.received_constraint = copy.deepcopy(tool_call_constraint)
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


def completion(model):
    instance = FastLLmCompletion.__new__(FastLLmCompletion)
    instance.model_name = "dots3-note"
    instance.model = model
    instance.think = False
    instance.enable_thinking = False
    instance.hide_input = True
    instance.conversation_handles = {}
    return instance


def request(tool_choice="auto", tools=None):
    return ChatCompletionRequest(
        model="dots3-note",
        messages=[{"role": "user", "content": "say hello"}],
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=128,
    )


class DotsToolChoiceTest(unittest.IsolatedAsyncioTestCase):
    def test_required_preserves_list_system_content(self):
        model = FakeDotsToolModel(WEATHER_CALL)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Original policy."}],
            },
            {"role": "user", "content": "say hello"},
        ]

        guided, selected_tools = completion(model)._apply_dots_tool_choice(
            messages, [weather_tool()], "required")

        self.assertEqual(guided[0]["role"], "system")
        self.assertEqual(guided[0]["content"][0]["text"],
                         "Original policy.")
        self.assertIn("must call at least one",
                      guided[0]["content"][-1]["text"])
        self.assertEqual(selected_tools, [weather_tool()])
        self.assertEqual(messages[0]["content"],
                         [{"type": "text", "text": "Original policy."}])

    async def test_none_hides_tools_and_skips_tool_parser(self):
        model = FakeDotsToolModel("hello")
        chat_request = request("none", [weather_tool()])

        response = await completion(model).create_chat_completion(
            chat_request, RawRequest())

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertTrue(model.launch_called)
        self.assertIsNone(model.launch_tools)
        self.assertIsNone(model.received_constraint)
        self.assertEqual(model.launch_messages,
                         [{"role": "user", "content": "say hello"}])
        self.assertEqual(response.choices[0].message.content, "hello")
        self.assertIsNone(response.choices[0].message.tool_calls)
        self.assertIsNotNone(chat_request.tools)

    async def test_required_adds_system_guidance_and_accepts_call(self):
        model = FakeDotsToolModel(WEATHER_CALL)

        response = await completion(model).create_chat_completion(
            request("required", [weather_tool(), time_tool()]),
            RawRequest(),
        )

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertIn("must call at least one", model.launch_messages[0]["content"])
        self.assertIn("immediately without explaining",
                      model.launch_messages[0]["content"])
        self.assertIn("tool_choice=required",
                      model.launch_messages[-1]["content"])
        self.assertIn("无需解释", model.launch_messages[-1]["content"])
        self.assertEqual(
            [tool["function"]["name"] for tool in model.launch_tools],
            ["get_weather", "get_time"],
        )
        self.assertEqual(
            model.received_constraint["name_constraint"]["format"],
            "dots_xml",
        )
        self.assertTrue(
            model.received_constraint["descriptor"]["requires_tool_call"])
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")
        call = response.choices[0].message.tool_calls[0]
        self.assertEqual(call.function.name, "get_weather")
        self.assertEqual(json.loads(call.function.arguments), {"city": "北京"})

    async def test_named_choice_exposes_and_allows_only_named_tool(self):
        model = FakeDotsToolModel(WEATHER_CALL)
        named_choice = {
            "type": "function",
            "function": {"name": "get_weather"},
        }

        response = await completion(model).create_chat_completion(
            request(named_choice, [weather_tool(), time_tool()]),
            RawRequest(),
        )

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertIn("'get_weather'", model.launch_messages[0]["content"])
        self.assertIn("'get_weather'", model.launch_messages[-1]["content"])
        self.assertEqual(
            [tool["function"]["name"] for tool in model.launch_tools],
            ["get_weather"],
        )
        self.assertEqual(
            model.received_constraint["name_constraint"]["allowed_names"],
            ["get_weather"],
        )
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")

    async def test_named_choice_without_call_is_rejected(self):
        model = FakeDotsToolModel("hello")
        named_choice = {
            "type": "function",
            "function": {"name": "get_weather"},
        }

        response = await completion(model).create_chat_completion(
            request(named_choice, [weather_tool(), time_tool()]),
            RawRequest(),
        )

        self.assertIsInstance(response, ErrorResponse)
        self.assertIn("tool_choice_violation", response.message)

    async def test_unknown_named_choice_is_rejected_before_launch(self):
        model = FakeDotsToolModel(WEATHER_CALL)
        named_choice = {
            "type": "function",
            "function": {"name": "missing_tool"},
        }

        response = await completion(model).create_chat_completion(
            request(named_choice, [weather_tool()]), RawRequest())

        self.assertIsInstance(response, ErrorResponse)
        self.assertIn("unknown function", response.message)
        self.assertFalse(model.launch_called)


if __name__ == "__main__":
    unittest.main()
