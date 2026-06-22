#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.golden import get_parser, split_text_by_chunk_size  # noqa: E402
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (  # noqa: E402
    ChatCompletionRequest,
    FunctionCall,
    ToolCall,
)
from tools.fastllm_pytools.openai_server.toolcall_adapter import (  # noqa: E402
    build_toolcall_context,
    map_tool_calls_to_external,
    named_tool_choice_name,
    prepare_request_for_model,
    validate_tool_calls,
)


def _weather_tool(name: str = "get_weather") -> Dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _time_tool() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "get_time",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        },
    }


class ToolCallAdapterContextTest(unittest.TestCase):
    def make_request(self, **overrides) -> ChatCompletionRequest:
        payload = {
            "model": "dummy",
            "messages": [{"role": "user", "content": "查天气"}],
            "tools": [_weather_tool(), _time_tool()],
            "tool_choice": "auto",
        }
        payload.update(overrides)
        return ChatCompletionRequest(**payload)

    def test_build_context_without_tools(self):
        request = self.make_request(tools=None, tool_choice="none")

        context = build_toolcall_context(request, model_type="deepseek_v4")

        self.assertFalse(context.has_tools)
        self.assertEqual(context.external_tools, ())
        self.assertEqual(context.model_tools, ())
        self.assertEqual(context.allowed_external_tool_names, frozenset())
        self.assertEqual(context.allowed_model_tool_names, frozenset())
        self.assertEqual(context.tool_choice, "none")
        self.assertEqual(context.model_type, "deepseek_v4")

    def test_build_context_with_tools_and_auto_choice(self):
        request = self.make_request()

        context = build_toolcall_context(request)

        self.assertTrue(context.has_tools)
        self.assertEqual(len(context.external_tools), 2)
        self.assertEqual(context.external_tools, context.model_tools)
        self.assertEqual(
            context.allowed_external_tool_names,
            frozenset({"get_weather", "get_time"}),
        )
        self.assertEqual(
            context.allowed_model_tool_names,
            frozenset({"get_weather", "get_time"}),
        )
        self.assertEqual(context.tool_choice, "auto")

    def test_build_context_records_alias_names_without_applying_them(self):
        request = self.make_request()

        context = build_toolcall_context(
            request, alias_map={"get_weather": "weather"})

        self.assertEqual(
            context.allowed_external_tool_names,
            frozenset({"get_weather", "get_time"}),
        )
        self.assertEqual(
            context.allowed_model_tool_names,
            frozenset({"weather", "get_time"}),
        )
        self.assertEqual(context.external_tools, context.model_tools)

    def test_named_tool_choice_name(self):
        request = self.make_request(
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            })
        context = build_toolcall_context(request)

        self.assertEqual(named_tool_choice_name(context.tool_choice),
                         "get_weather")

    def test_prepare_request_no_alias_is_identity(self):
        request = self.make_request()
        context = build_toolcall_context(request)

        prepared = prepare_request_for_model(request, context)

        self.assertIs(prepared, request)
        self.assertEqual(prepared.tools[0].function.name, "get_weather")
        self.assertEqual(prepared.tools[1].function.name, "get_time")

    def test_map_tool_calls_no_alias_returns_same_visible_names(self):
        request = self.make_request()
        context = build_toolcall_context(request)
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(
                    name="get_weather",
                    arguments=json.dumps({"city": "北京"}),
                ),
            )
        ]

        mapped = map_tool_calls_to_external(tool_calls, context)

        self.assertIsNot(mapped, tool_calls)
        self.assertEqual(mapped[0].function.name, "get_weather")
        self.assertEqual(json.loads(mapped[0].function.arguments),
                         {"city": "北京"})

    def test_validate_tool_calls_permissive_by_default(self):
        request = self.make_request()
        context = build_toolcall_context(request)
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(name="get_wether", arguments="{}"),
            )
        ]

        result = validate_tool_calls(tool_calls, context)

        self.assertTrue(result.valid)
        self.assertEqual(result.errors, ())
        self.assertEqual(result.tool_names, ("get_wether",))

    def test_validate_tool_calls_strict_mode_records_unknown_name(self):
        request = self.make_request()
        context = build_toolcall_context(request, strict_mode=True)
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(name="get_wether", arguments="{}"),
            )
        ]

        result = validate_tool_calls(tool_calls, context)

        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].code, "invalid_tool_name")
        self.assertEqual(result.errors[0].tool_name, "get_wether")
        self.assertEqual(result.errors[0].index, 0)


class ToolCallAdapterIntegrationTest(unittest.TestCase):
    RAW_TOOL_CALL = (
        "<｜DSML｜tool_calls>"
        "<｜DSML｜invoke name=\"get_weather\">"
        "<｜DSML｜parameter name=\"city\" string=\"true\">北京</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )

    def make_request(self) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="dummy",
            messages=[{"role": "user", "content": "查北京天气"}],
            tools=[_weather_tool()],
            tool_choice="auto",
        )

    def test_non_stream_parser_behavior_unchanged(self):
        request = self.make_request()
        context = build_toolcall_context(request, model_type="deepseek_v4")
        prepared_request = prepare_request_for_model(request, context)

        direct = get_parser("deepseek_v4").extract_tool_calls(
            self.RAW_TOOL_CALL, request)
        adapted = get_parser("deepseek_v4").extract_tool_calls(
            self.RAW_TOOL_CALL, prepared_request)

        self.assertTrue(adapted.tools_called)
        self.assertEqual(adapted.content, direct.content)
        self.assertEqual(
            [call.function.name for call in adapted.tool_calls],
            [call.function.name for call in direct.tool_calls],
        )
        self.assertEqual(
            [json.loads(call.function.arguments) for call in adapted.tool_calls],
            [json.loads(call.function.arguments) for call in direct.tool_calls],
        )

    def test_stream_parser_behavior_unchanged(self):
        request = self.make_request()
        context = build_toolcall_context(request, model_type="deepseek_v4")
        prepared_request = prepare_request_for_model(request, context)

        direct = self._collect_stream(get_parser("deepseek_v4"), request)
        adapted = self._collect_stream(
            get_parser("deepseek_v4"), prepared_request)

        self.assertEqual(adapted, direct)
        self.assertEqual(adapted, {"get_weather": {"city": "北京"}})

    def _collect_stream(self, parser, request: ChatCompletionRequest) -> Dict:
        previous_text = ""
        current_text = ""
        previous_token_ids: List[int] = []
        current_token_ids: List[int] = []
        tool_calls: Dict[int, Dict[str, str]] = {}
        next_token_id = 0

        for chunk in split_text_by_chunk_size(self.RAW_TOOL_CALL, 7):
            delta_token_ids = list(
                range(next_token_id, next_token_id + len(chunk)))
            next_token_id += len(chunk)
            current_text += chunk
            current_token_ids += delta_token_ids
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
                request=request,
            )
            previous_text = current_text
            previous_token_ids = list(current_token_ids)

            if delta is None:
                continue
            for tool_delta in delta.tool_calls:
                state = tool_calls.setdefault(
                    tool_delta.index, {"name": "", "arguments": ""})
                if tool_delta.function is None:
                    continue
                if tool_delta.function.name:
                    state["name"] = tool_delta.function.name
                if tool_delta.function.arguments:
                    state["arguments"] += tool_delta.function.arguments

        return {
            state["name"]: json.loads(state["arguments"])
            for state in tool_calls.values()
        }


if __name__ == "__main__":
    unittest.main()
