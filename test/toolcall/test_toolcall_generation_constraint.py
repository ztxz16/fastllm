#!/usr/bin/env python3
import copy
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fastllm_pytools.openai_server.fastllm_completion import (  # noqa: E402
    FastLLmCompletion,
)
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionResponse,
    ExtractedToolCallInformation,
)
from tools.fastllm_pytools.openai_server.protocal.anthropic_protocol import (  # noqa: E402
    AnthropicMessageRequest,
    AnthropicMessageResponse,
)


def _weather_tool(strict=False):
    function = {
        "name": "get_weather",
        "description": "Get weather.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _request(tools=None, tool_choice="auto"):
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "查天气"}],
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=128,
    )


def _dsml_call(name="get_weather"):
    return (
        "<｜DSML｜tool_calls>"
        f"<｜DSML｜invoke name=\"{name}\">"
        "<｜DSML｜parameter name=\"city\" string=\"true\">北京</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )


class _RawRequest:
    async def is_disconnected(self):
        return False


class _NoToolParser:
    def extract_tool_calls(self, text, request):
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=text,
        )


class _BaseRecordingModel:
    default_generation_config = {
        "repetition_penalty": 1.0,
        "top_p": 0.8,
        "top_k": 1,
        "temperature": 1.0,
    }
    force_chat_template = False
    tool_call_parser = "deepseek_v4"
    hf_tokenizer = None

    def __init__(self, output):
        self.output = output
        self.launch_called = False
        self.launch_kwargs = None
        self.input_token_messages = None

    def get_type(self):
        return "deepseek_v4"

    def _is_deepseek_v4(self):
        return False

    def get_input_token_len(self, messages, enable_thinking=False):
        self.input_token_messages = copy.deepcopy(messages)
        return 3

    def stream_response_handle_async(self, handle):
        async def generator():
            yield self.output

        return generator()

    def abort_handle(self, handle):
        raise AssertionError("abort_handle should not be called")


class _ConstraintAwareModel(_BaseRecordingModel):
    def __init__(self, output):
        super().__init__(output)
        self.received_constraint = None

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
        self.received_constraint = copy.deepcopy(tool_call_constraint)
        self.launch_kwargs = {
            "tools": copy.deepcopy(tools),
            "tool_call_constraint": copy.deepcopy(tool_call_constraint),
        }
        return 101


class _ConstraintUnsupportedModel(_BaseRecordingModel):
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
        self.launch_called = True
        self.launch_kwargs = {"tools": copy.deepcopy(tools)}
        return 102


def _completion(model):
    completion = FastLLmCompletion.__new__(FastLLmCompletion)
    completion.model_name = "dummy"
    completion.model = model
    completion.think = False
    completion.enable_thinking = False
    completion.hide_input = True
    completion.conversation_handles = {}
    return completion


def _anthropic_request(tools=None):
    return AnthropicMessageRequest(
        model="dummy",
        max_tokens=128,
        messages=[{"role": "user", "content": "查天气"}],
        tools=tools,
    )


class ToolCallGenerationConstraintTest(unittest.IsolatedAsyncioTestCase):
    async def _run_completion(self, model, request):
        completion = _completion(model)
        return await completion.create_chat_completion(request, _RawRequest())

    async def _run_anthropic_completion(self, model, request):
        completion = _completion(model)
        completion._create_tool_parser = lambda: _NoToolParser()
        return await completion.create_anthropic_message(request, _RawRequest())

    async def test_required_tool_builds_and_passes_generation_constraint(self):
        model = _ConstraintAwareModel(_dsml_call())
        request = _request(
            tools=[_weather_tool(strict=True)],
            tool_choice="required",
        )

        response = await self._run_completion(model, request)

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertTrue(model.launch_called)
        constraint = model.received_constraint
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint["backend"],
                         "fastllm_toolcall_prototype")
        self.assertEqual(constraint["structural_tag"]["format"],
                         "deepseek_v4_dsml")
        descriptor = constraint["descriptor"]
        self.assertEqual(descriptor["constraint_type"], "deepseek_v4_dsml")
        self.assertEqual(descriptor["tool_names"], ["get_weather"])
        self.assertEqual(descriptor["allowed_tool_names"], ["get_weather"])
        self.assertTrue(descriptor["requires_tool_call"])
        self.assertEqual(descriptor["strict_tool_names"], ["get_weather"])
        self.assertEqual(
            constraint["json_schemas"]["get_weather"]["required"], ["city"])
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.tool_calls[0].function.name,
                         "get_weather")
        self.assertEqual(json.loads(
            choice.message.tool_calls[0].function.arguments), {"city": "北京"})

    async def test_constraint_absent_without_tools(self):
        model = _ConstraintAwareModel("普通回复")
        request = _request(tools=None)

        response = await self._run_completion(model, request)

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertTrue(model.launch_called)
        self.assertIsNone(model.received_constraint)
        self.assertIsNone(model.launch_kwargs["tools"])
        self.assertEqual(response.choices[0].finish_reason, "stop")

    async def test_unsupported_backend_path_does_not_crash(self):
        model = _ConstraintUnsupportedModel(_dsml_call())
        request = _request(
            tools=[_weather_tool()],
            tool_choice="required",
        )

        with self.assertLogs(level="DEBUG") as logs:
            response = await self._run_completion(model, request)

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertTrue(model.launch_called)
        self.assertNotIn("tool_call_constraint", model.launch_kwargs)
        self.assertIn("does not support tool_call_constraint",
                      "\n".join(logs.output))
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")

    async def test_anthropic_tools_build_and_pass_generation_constraint(self):
        model = _ConstraintAwareModel("普通回复")
        request = _anthropic_request(tools=[{
            "name": "get_weather",
            "description": "Get weather.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }])

        response = await self._run_anthropic_completion(model, request)

        self.assertIsInstance(response, AnthropicMessageResponse)
        self.assertTrue(model.launch_called)
        self.assertEqual(model.launch_kwargs["tools"][0]["function"]["name"],
                         "get_weather")
        constraint = model.received_constraint
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint["backend"],
                         "fastllm_toolcall_prototype")
        self.assertEqual(constraint["structural_tag"]["format"],
                         "deepseek_v4_dsml")
        descriptor = constraint["descriptor"]
        self.assertEqual(descriptor["constraint_type"], "deepseek_v4_dsml")
        self.assertEqual(descriptor["tool_names"], ["get_weather"])
        self.assertEqual(descriptor["allowed_tool_names"], ["get_weather"])
        self.assertFalse(descriptor["requires_tool_call"])

    async def test_anthropic_constraint_absent_without_tools(self):
        model = _ConstraintAwareModel("普通回复")
        request = _anthropic_request(tools=None)

        response = await self._run_anthropic_completion(model, request)

        self.assertIsInstance(response, AnthropicMessageResponse)
        self.assertTrue(model.launch_called)
        self.assertIsNone(model.received_constraint)
        self.assertIsNone(model.launch_kwargs["tools"])


if __name__ == "__main__":
    unittest.main()
