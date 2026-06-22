#!/usr/bin/env python3
import copy
import json
import sys
import unittest
from pathlib import Path
from typing import Dict, Iterable, List


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


def _request(tools=None, tool_choice="auto") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "查天气"}],
        tools=tools,
        tool_choice=tool_choice,
    )


def _dsml_call(name: str) -> str:
    return (
        "<｜DSML｜tool_calls>"
        f"<｜DSML｜invoke name=\"{name}\">"
        "<｜DSML｜parameter name=\"city\" string=\"true\">北京</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )


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

    def test_tool_choice_is_stored_but_not_enforced(self):
        parser = FunctionCallParser.from_request(
            _request(
                tools=[_weather_tool(), _time_tool()],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_time"},
                },
            ))

        self.assertIsNotNone(parser.tool_choice)
        result = parser.parse_non_stream(_dsml_call("get_weather"))

        self.assertTrue(result.tools_called)
        self.assertEqual(result.diagnostics, [])
        self.assertEqual(result.valid_tool_calls[0].function.name, "get_weather")

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
