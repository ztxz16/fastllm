#!/usr/bin/env python3
import copy
import json
import sys
import unittest
from pathlib import Path
from typing import Dict, Iterable, List
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from tools.fastllm_pytools.openai_server.toolcall_parser import (  # noqa: E402
    FunctionCallParser,
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


def _strict_weather_tool(name: str = "get_weather") -> Dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get weather.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "rain": {"type": "boolean"},
                    "tags": {"type": "array"},
                    "meta": {"type": "object"},
                },
                "required": ["city"],
            },
        },
    }


def _strict_todowrite_tool() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "todowrite",
            "description": "Create and maintain a structured task list.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string"},
                                "priority": {"type": "string"},
                            },
                            "required": ["content", "status", "priority"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    }


def _time_tool() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get time.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        },
    }


def _request(tools=None, tool_choice="auto",
             parallel_tool_calls=None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "查天气"}],
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )


def _dsml_call(name: str) -> str:
    return (
        "<｜DSML｜tool_calls>"
        f"<｜DSML｜invoke name=\"{name}\">"
        "<｜DSML｜parameter name=\"city\" string=\"true\">北京</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )


def _time_dsml_call() -> str:
    return (
        "<｜DSML｜invoke name=\"get_time\">"
        "<｜DSML｜parameter name=\"timezone\" string=\"true\">Asia/Shanghai</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
    )


def _parallel_dsml_call() -> str:
    return (
        "<｜DSML｜tool_calls>"
        "<｜DSML｜invoke name=\"get_weather\">"
        "<｜DSML｜parameter name=\"city\" string=\"true\">北京</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
        f"{_time_dsml_call()}"
        "</｜DSML｜tool_calls>"
    )


def _tool_call_dict(name: str, arguments) -> Dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _chunks(text: str, chunk_size: int) -> Iterable[str]:
    for index in range(0, len(text), chunk_size):
        yield text[index:index + chunk_size]


class FunctionCallParserTest(unittest.TestCase):
    def test_build_tool_index_from_openai_tools(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool(), _time_tool()]))

        self.assertEqual(parser.tool_index, {
            "get_weather": 0,
            "get_time": 1,
        })

    def test_empty_tools_means_no_toolcall_parsing(self):
        parser = FunctionCallParser.from_request(_request(tools=None))

        self.assertFalse(parser.has_tools)
        self.assertFalse(parser.has_tool_call(_dsml_call("get_weather")))
        result = parser.parse_non_stream(_dsml_call("get_weather"))

        self.assertFalse(result.tools_called)
        self.assertFalse(result.has_invalid_tool_block)
        self.assertEqual(result.content, _dsml_call("get_weather"))
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(result.invalid_tool_calls, [])

    def test_has_tool_call_delegates_to_parser_detector(self):
        class DetectorParser:
            def has_tool_call(self, text: str) -> bool:
                return text == "detected"

        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()]))
        parser.parser = DetectorParser()

        self.assertTrue(parser.has_tool_call("detected"))
        self.assertFalse(parser.has_tool_call("plain text"))

    def test_named_tool_choice_match_passes(self):
        parser = FunctionCallParser.from_request(
            _request(
                tools=[_weather_tool(), _time_tool()],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            ))

        self.assertIsNotNone(parser.tool_choice)
        result = parser.parse_non_stream(_dsml_call("get_weather"))

        self.assertTrue(result.tools_called)
        self.assertEqual(result.diagnostics, [])
        self.assertEqual(result.valid_tool_calls[0].function.name, "get_weather")

    def test_named_tool_choice_mismatch_is_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(
                tools=[_weather_tool(), _time_tool()],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_time"},
                },
            ))

        result = parser.parse_non_stream(_dsml_call("get_weather"))

        self.assertFalse(result.tools_called)
        self.assertTrue(result.has_invalid_tool_block)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 1)
        self.assertEqual(result.diagnostics[0].code, "tool_choice_violation")
        self.assertEqual(result.diagnostics[0].tool_name, "get_weather")

    def test_required_tool_choice_without_call_is_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()], tool_choice="required"))

        result = parser.parse_non_stream("普通回复")

        self.assertFalse(result.tools_called)
        self.assertTrue(result.has_invalid_tool_block)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].code, "tool_choice_violation")

    def test_required_tool_choice_with_valid_call_passes(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()], tool_choice="required"))

        result = parser.parse_non_stream(_dsml_call("get_weather"))

        self.assertTrue(result.tools_called)
        self.assertFalse(result.has_invalid_tool_block)
        self.assertEqual(result.diagnostics, [])

    def test_parallel_tool_calls_false_with_one_call_passes(self):
        parser = FunctionCallParser.from_request(
            _request(
                tools=[_weather_tool(), _time_tool()],
                parallel_tool_calls=False,
            ))

        result = parser.parse_non_stream(_dsml_call("get_weather"))

        self.assertTrue(result.tools_called)
        self.assertFalse(result.has_invalid_tool_block)
        self.assertEqual(result.diagnostics, [])

    def test_parallel_tool_calls_false_with_two_calls_is_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(
                tools=[_weather_tool(), _time_tool()],
                parallel_tool_calls=False,
            ))

        result = parser.parse_non_stream(_parallel_dsml_call())

        self.assertFalse(result.tools_called)
        self.assertTrue(result.has_invalid_tool_block)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 2)
        self.assertEqual(result.diagnostics[-1].code,
                         "parallel_tool_calls_violation")

    def test_strict_schema_required_key_present_passes(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_weather_tool()]))

        result = parser.validate_tool_calls([
            _tool_call_dict(
                "get_weather",
                json.dumps({"city": "北京", "days": 3},
                           ensure_ascii=False),
            )
        ])

        self.assertTrue(result.valid)
        self.assertEqual(len(result.valid_tool_calls), 1)
        self.assertEqual(result.invalid_tool_calls, [])
        self.assertEqual(result.diagnostics, [])

    def test_strict_schema_required_key_missing_is_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_weather_tool()]))

        result = parser.validate_tool_calls([
            _tool_call_dict("get_weather", json.dumps({"days": 3}))
        ])

        self.assertFalse(result.valid)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 1)
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].code,
                         "missing_required_argument")
        self.assertEqual(result.diagnostics[0].argument_name, "city")

    def test_opencode_todowrite_todos_array_passes(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_todowrite_tool()]))

        result = parser.validate_tool_calls([
            _tool_call_dict(
                "todowrite",
                json.dumps({
                    "todos": [
                        {
                            "content": "inspect OpenAI request",
                            "status": "pending",
                            "priority": "medium",
                        },
                        {
                            "content": "inspect tool call arguments",
                            "status": "pending",
                            "priority": "medium",
                        },
                    ],
                }),
            )
        ])

        self.assertTrue(result.valid)
        self.assertEqual(len(result.valid_tool_calls), 1)
        self.assertEqual(result.invalid_tool_calls, [])
        self.assertEqual(result.diagnostics, [])

    def test_opencode_todowrite_wrong_top_level_keys_are_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_todowrite_tool()]))

        wrong_arguments = [
            {"todo": [{"content": "inspect OpenAI request"}]},
            {"todoList": [{"content": "inspect OpenAI request"}]},
            {"todolist": [{"content": "inspect OpenAI request"}]},
            {"todoS": [{"content": "inspect OpenAI request"}]},
            {"description": "inspect OpenAI request"},
            {"arguments": {}},
        ]
        for arguments in wrong_arguments:
            with self.subTest(arguments=arguments):
                result = parser.validate_tool_calls([
                    _tool_call_dict("todowrite", json.dumps(arguments))
                ])

                self.assertFalse(result.valid)
                self.assertEqual(result.valid_tool_calls, [])
                self.assertEqual(len(result.invalid_tool_calls), 1)
                self.assertEqual(len(result.diagnostics), 1)
                self.assertEqual(result.diagnostics[0].code,
                                 "missing_required_argument")
                self.assertEqual(result.diagnostics[0].argument_name, "todos")

    def test_opencode_todowrite_missing_todo_item_fields_are_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_todowrite_tool()]))

        result = parser.validate_tool_calls([
            _tool_call_dict(
                "todowrite",
                json.dumps({
                    "todos": [
                        {
                            "content": "inspect OpenAI request",
                        },
                    ],
                }),
            )
        ])

        self.assertFalse(result.valid)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 1)
        diagnostic_names = {
            diagnostic.argument_name for diagnostic in result.diagnostics
        }
        self.assertIn("todos[0].status", diagnostic_names)
        self.assertIn("todos[0].priority", diagnostic_names)
        self.assertTrue(all(
            diagnostic.code == "missing_required_argument"
            for diagnostic in result.diagnostics
        ))

    def test_strict_schema_wrong_primitive_type_is_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_weather_tool()]))

        result = parser.validate_tool_calls([
            _tool_call_dict("get_weather", json.dumps({"city": 123}))
        ])

        self.assertFalse(result.valid)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 1)
        self.assertEqual(result.diagnostics[0].code, "invalid_argument_type")
        self.assertEqual(result.diagnostics[0].argument_name, "city")

    def test_strict_schema_arguments_must_be_json(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_weather_tool()]))

        result = parser.validate_tool_calls([
            _tool_call_dict("get_weather", "{bad json")
        ])

        self.assertFalse(result.valid)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 1)
        self.assertEqual(result.diagnostics[0].code,
                         "malformed_arguments_json")

    def test_strict_schema_arguments_must_be_object(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_weather_tool()]))

        result = parser.validate_tool_calls([
            _tool_call_dict("get_weather", json.dumps(["北京"]))
        ])

        self.assertFalse(result.valid)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 1)
        self.assertEqual(result.diagnostics[0].code,
                         "malformed_arguments_json")

    def test_non_strict_schema_missing_required_key_is_allowed(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()]))

        result = parser.validate_tool_calls([
            _tool_call_dict("get_weather", json.dumps({}))
        ])

        self.assertTrue(result.valid)
        self.assertEqual(len(result.valid_tool_calls), 1)
        self.assertEqual(result.invalid_tool_calls, [])
        self.assertEqual(result.diagnostics, [])

    def test_constraint_descriptor_auto_without_strict_is_dry_run(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool(), _time_tool()]))

        descriptor = parser.build_constraint_descriptor()

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.constraint_type, "deepseek_v4_dsml")
        self.assertEqual(descriptor.model_type, "deepseek_v4")
        self.assertEqual(descriptor.tool_names, ("get_weather", "get_time"))
        self.assertEqual(descriptor.allowed_tool_names,
                         ("get_weather", "get_time"))
        self.assertEqual(descriptor.tool_choice, "auto")
        self.assertFalse(descriptor.requires_tool_call)
        self.assertEqual(descriptor.schemas, {})
        self.assertEqual(descriptor.parameter_names, {
            "get_weather": ("city",),
            "get_time": ("timezone",),
        })
        self.assertEqual(descriptor.strict_tool_names, ())

    def test_constraint_descriptor_required_requires_tool_call(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()], tool_choice="required"))

        descriptor = parser.build_constraint_descriptor()

        self.assertTrue(descriptor.requires_tool_call)
        self.assertIsNone(descriptor.named_tool_choice)
        self.assertEqual(descriptor.allowed_tool_names, ("get_weather",))

    def test_constraint_descriptor_named_tool_choice_restricts_enum(self):
        parser = FunctionCallParser.from_request(
            _request(
                tools=[_weather_tool(), _time_tool()],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_time"},
                },
            ))

        descriptor = parser.build_constraint_descriptor()

        self.assertTrue(descriptor.requires_tool_call)
        self.assertEqual(descriptor.named_tool_choice, "get_time")
        self.assertEqual(descriptor.allowed_tool_names, ("get_time",))
        self.assertEqual(
            descriptor.tool_choice,
            {"type": "function", "function": {"name": "get_time"}},
        )

    def test_constraint_descriptor_strict_includes_schema(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_weather_tool(), _time_tool()]))

        descriptor = parser.build_constraint_descriptor()

        self.assertEqual(descriptor.strict_tool_names, ("get_weather",))
        self.assertEqual(
            descriptor.schemas["get_weather"]["required"], ["city"])
        self.assertEqual(
            descriptor.schemas["get_weather"]["properties"]["city"]["type"],
            "string",
        )
        self.assertNotIn("get_time", descriptor.schemas)
        self.assertEqual(descriptor.parameter_names["get_weather"],
                         ("city", "days", "rain", "tags", "meta"))
        self.assertEqual(descriptor.parameter_names["get_time"],
                         ("timezone",))

    def test_constraint_descriptor_schema_is_snapshot(self):
        tool = _strict_weather_tool()
        parser = FunctionCallParser.from_request(_request(tools=[tool]))

        descriptor = parser.build_constraint_descriptor()
        tool["function"]["parameters"]["required"].append("unit")

        self.assertEqual(descriptor.schemas["get_weather"]["required"],
                         ["city"])
        self.assertEqual(descriptor.to_dict()["schemas"]["get_weather"]
                         ["required"], ["city"])

    def test_constraint_descriptor_to_dict_is_json_serializable_copy(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_strict_weather_tool()]))
        descriptor = parser.build_constraint_descriptor()

        dumped = descriptor.to_dict()
        dumped["schemas"]["get_weather"]["required"].append("unit")
        dumped["parameter_names"]["get_weather"].append("unit")

        json.dumps(dumped, ensure_ascii=False)
        self.assertEqual(descriptor.schemas["get_weather"]["required"],
                         ["city"])
        self.assertEqual(descriptor.parameter_names["get_weather"],
                         ("city", "days", "rain", "tags", "meta"))

    def test_constraint_descriptor_parallel_flag_is_included(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()], parallel_tool_calls=False))

        descriptor = parser.build_constraint_descriptor()

        self.assertFalse(descriptor.parallel_tool_calls)
        self.assertFalse(descriptor.to_dict()["parallel_tool_calls"])

    def test_constraint_descriptor_absent_without_tools(self):
        parser = FunctionCallParser.from_request(_request(tools=None))

        self.assertIsNone(parser.build_constraint_descriptor())

    def test_non_stream_valid_tool_name_passes(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()]))

        result = parser.parse_non_stream(_dsml_call("get_weather"))

        self.assertTrue(result.tools_called)
        self.assertFalse(result.has_invalid_tool_block)
        self.assertEqual(result.invalid_tool_calls, [])
        self.assertEqual(result.diagnostics, [])
        self.assertEqual(len(result.valid_tool_calls), 1)
        self.assertEqual(
            result.valid_tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(result.valid_tool_calls[0].function.arguments),
            {"city": "北京"},
        )

    def test_non_stream_unknown_tool_name_is_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()]))

        result = parser.parse_non_stream(_dsml_call("get_wearher"))

        self.assertFalse(result.tools_called)
        self.assertTrue(result.has_invalid_tool_block)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 1)
        self.assertEqual(
            result.invalid_tool_calls[0].function.name, "get_wearher")
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].code, "invalid_tool_name")
        self.assertEqual(result.diagnostics[0].tool_name, "get_wearher")
        self.assertIsNone(result.diagnostics[0].closest_tool_name)

    def test_compat_mode_records_closest_match_diagnostics(self):
        with patch.dict("os.environ", {"FT_TOOLCALL_COMPAT_MODE": "ON"}):
            parser = FunctionCallParser.from_request(
                _request(tools=[_weather_tool()]))

        result = parser.parse_non_stream(_dsml_call("get_wearher"))

        self.assertFalse(result.tools_called)
        self.assertTrue(result.has_invalid_tool_block)
        self.assertEqual(len(result.diagnostics), 1)
        diagnostic = result.diagnostics[0]
        self.assertEqual(diagnostic.code, "invalid_tool_name")
        self.assertEqual(diagnostic.tool_name, "get_wearher")
        self.assertEqual(diagnostic.allowed_tool_names, ("get_weather",))
        self.assertEqual(diagnostic.closest_tool_name, "get_weather")
        self.assertGreater(diagnostic.similarity_ratio, 0.9)

    def test_forward_unknown_tools_is_off_by_default(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()]))

        result = parser.parse_non_stream(_dsml_call("get_wearher"))

        self.assertFalse(result.tools_called)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 1)

    def test_forward_unknown_tools_preserves_raw_unknown_name(self):
        with patch.dict("os.environ",
                        {"FT_TOOLCALL_FORWARD_UNKNOWN_TOOLS": "ON"}):
            parser = FunctionCallParser.from_request(
                _request(tools=[_weather_tool()]))

        result = parser.parse_non_stream(_dsml_call("get_wearher"))
        validation = parser.validate_tool_calls(result.valid_tool_calls)

        self.assertTrue(result.tools_called)
        self.assertFalse(result.has_invalid_tool_block)
        self.assertEqual(result.invalid_tool_calls, [])
        self.assertEqual(len(result.valid_tool_calls), 1)
        self.assertEqual(result.valid_tool_calls[0].function.name,
                         "get_wearher")
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].code, "invalid_tool_name")
        self.assertTrue(validation.valid)
        self.assertEqual(validation.invalid_tool_calls, [])
        self.assertEqual(len(validation.diagnostics), 1)

    def test_non_stream_malformed_tool_block_is_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()]))
        raw_output = (
            "<｜DSML｜tool_calls>"
            "<｜DSML｜invoke name=\"get_weather\">"
            "<｜DSML｜parameter name=\"city\" string=\"true\">北京</｜DSML｜parameter>"
            "</｜DSML｜tool_calls>"
        )

        result = parser.parse_non_stream(raw_output)

        self.assertFalse(result.tools_called)
        self.assertTrue(result.has_invalid_tool_block)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(result.invalid_tool_calls, [])
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].code, "malformed_tool_block")

    def test_missing_tool_name_is_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()]))

        result = parser.validate_tool_calls([
            {
                "id": "call_1",
                "type": "function",
                "function": {"arguments": "{}"},
            }
        ])

        self.assertFalse(result.valid)
        self.assertEqual(result.valid_tool_calls, [])
        self.assertEqual(len(result.invalid_tool_calls), 1)
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].code, "missing_tool_name")
        self.assertIsNone(result.diagnostics[0].tool_name)
        self.assertEqual(result.diagnostics[0].index, 0)

    def test_stream_valid_tool_name_passes(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()]))

        results = self._stream_results(parser, _dsml_call("get_weather"))

        valid_calls = [
            call
            for result in results
            for call in result.valid_tool_calls
        ]
        diagnostics = [
            diagnostic
            for result in results
            for diagnostic in result.diagnostics
        ]
        self.assertEqual(diagnostics, [])
        self.assertEqual(len(valid_calls), 1)
        self.assertEqual(valid_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(valid_calls[0].function.arguments),
            {"city": "北京"},
        )

    def test_stream_unknown_tool_name_is_invalid(self):
        parser = FunctionCallParser.from_request(
            _request(tools=[_weather_tool()]))

        results = self._stream_results(parser, _dsml_call("get_wearher"))

        valid_calls = [
            call
            for result in results
            for call in result.valid_tool_calls
        ]
        invalid_calls = [
            call
            for result in results
            for call in result.invalid_tool_calls
        ]
        diagnostics = [
            diagnostic
            for result in results
            for diagnostic in result.diagnostics
        ]
        self.assertEqual(valid_calls, [])
        self.assertEqual(len(invalid_calls), 1)
        self.assertEqual(invalid_calls[0].function.name, "get_wearher")
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].code, "invalid_tool_name")
        self.assertEqual(diagnostics[0].tool_name, "get_wearher")

    def test_stream_forward_unknown_tools_preserves_raw_unknown_name(self):
        with patch.dict("os.environ",
                        {"FT_TOOLCALL_FORWARD_UNKNOWN_TOOLS": "ON"}):
            parser = FunctionCallParser.from_request(
                _request(tools=[_weather_tool()]))

        results = self._stream_results(parser, _dsml_call("get_wearher"))

        valid_calls = [
            call
            for result in results
            for call in result.valid_tool_calls
        ]
        diagnostics = [
            diagnostic
            for result in results
            for diagnostic in result.diagnostics
        ]
        self.assertEqual(len(valid_calls), 1)
        self.assertEqual(valid_calls[0].function.name, "get_wearher")
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].code, "invalid_tool_name")

    def test_stream_named_tool_choice_mismatch_is_suppressed(self):
        parser = FunctionCallParser.from_request(
            _request(
                tools=[_weather_tool(), _time_tool()],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_time"},
                },
            ))

        results = self._stream_results(parser, _dsml_call("get_weather"))

        valid_calls = [
            call
            for result in results
            for call in result.valid_tool_calls
        ]
        diagnostics = [
            diagnostic
            for result in results
            for diagnostic in result.diagnostics
        ]
        self.assertEqual(valid_calls, [])
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].code, "tool_choice_violation")

    def test_stream_parallel_tool_calls_false_suppresses_second_call(self):
        parser = FunctionCallParser.from_request(
            _request(
                tools=[_weather_tool(), _time_tool()],
                parallel_tool_calls=False,
            ))

        results = self._stream_results(parser, _parallel_dsml_call())

        valid_calls = [
            call
            for result in results
            for call in result.valid_tool_calls
        ]
        diagnostics = [
            diagnostic
            for result in results
            for diagnostic in result.diagnostics
        ]
        self.assertEqual(len(valid_calls), 1)
        self.assertEqual(valid_calls[0].function.name, "get_weather")
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].code,
                         "parallel_tool_calls_violation")

    def test_request_tools_are_not_mutated(self):
        request = _request(tools=[_weather_tool(), _time_tool()])
        original_tools = copy.deepcopy(request.tools)
        parser = FunctionCallParser.from_request(request)

        parser.parse_non_stream(_dsml_call("get_weather"))

        self.assertEqual(request.tools, original_tools)

    def _stream_results(self, parser: FunctionCallParser,
                        text: str) -> List:
        previous_text = ""
        current_text = ""
        previous_token_ids: List[int] = []
        current_token_ids: List[int] = []
        results = []
        next_token_id = 0

        for chunk in _chunks(text, 5):
            delta_token_ids = list(
                range(next_token_id, next_token_id + len(chunk)))
            next_token_id += len(chunk)
            current_text += chunk
            current_token_ids += delta_token_ids
            result = parser.parse_stream_chunk(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
            )
            previous_text = current_text
            previous_token_ids = list(current_token_ids)
            results.append(result)

        return results


if __name__ == "__main__":
    unittest.main()
