#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from tools.fastllm_pytools.openai_server.toolcall_parser import (  # noqa: E402
    FunctionCallParser,
)
from tools.fastllm_pytools.openai_server.tool_parsers import (  # noqa: E402
    ToolParserManager,
)


class DummyTokenizer:
    def get_vocab(self):
        return {}


def make_request(parallel_tool_calls=None, strict=False):
    return ChatCompletionRequest(
        model="dots3-note",
        messages=[{"role": "user", "content": "查天气和时间"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "strict": strict,
                    "parameters": {
                        "type": "object",
                        "$defs": {
                            "metadata": {"type": "object"},
                        },
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                            "ratio": {"type": "number"},
                            "rain": {"type": "boolean"},
                            "tags": {"type": "array"},
                            "metadata": {"$ref": "#/$defs/metadata"},
                            "nullable": {
                                "anyOf": [
                                    {"type": "integer"},
                                    {"type": "null"},
                                ],
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        tool_choice="auto",
        parallel_tool_calls=parallel_tool_calls,
    )


WIRE = (
    "I will check."
    "<dots_function_call>\n"
    '<invoke name="get_weather">\n'
    '<parameter name="city">\n北京\n</parameter>\n'
    '<parameter name="days">\n3\n</parameter>\n'
    '<parameter name="ratio">\n1.5\n</parameter>\n'
    '<parameter name="rain">\ntrue\n</parameter>\n'
    '<parameter name="tags">\n["hot", "dry"]\n</parameter>\n'
    '<parameter name="metadata">\n{"unit":"c"}\n</parameter>\n'
    '<parameter name="nullable">\nnull\n</parameter>\n'
    "</invoke>\n"
    '<invoke name="get_time">\n</invoke>\n'
    "</dots_function_call>"
)


class DotsToolParserTest(unittest.TestCase):
    def make_parser(self):
        return ToolParserManager.get_tool_parser("dots")(DummyTokenizer())

    def make_function_parser(self, request):
        return FunctionCallParser.from_request(
            request,
            tool_parser_name="dots",
            parser=self.make_parser(),
        )

    def stream_with_function_parser(self, wire, request, chunk_size=5):
        parser = self.make_function_parser(request)
        previous = ""
        current = ""
        content = ""
        calls = []
        diagnostics = []
        for offset in range(0, len(wire), chunk_size):
            chunk = wire[offset:offset + chunk_size]
            current += chunk
            result = parser.parse_stream_chunk(
                previous_text=previous,
                current_text=current,
                delta_text=chunk,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
            )
            previous = current
            content += result.content or ""
            calls.extend(result.valid_tool_calls)
            diagnostics.extend(result.diagnostics)

        flushed = parser.flush_stream_tool_calls()
        calls.extend(flushed.valid_tool_calls)
        diagnostics.extend(flushed.diagnostics)
        return content, calls, diagnostics

    def test_auto_selects_dots_from_model_and_template(self):
        by_model = ToolParserManager.get_tool_parser_auto(
            "dots3_note", "")
        by_template = ToolParserManager.get_tool_parser_auto(
            "unknown", "Use <dots_function_call> for tools")

        self.assertEqual(by_model.__name__, "DotsToolParser")
        self.assertEqual(by_template.__name__, "DotsToolParser")

    def test_non_stream_parallel_calls_and_schema_types(self):
        result = self.make_parser().extract_tool_calls(WIRE, make_request())

        self.assertTrue(result.tools_called)
        self.assertEqual(result.content, "I will check.")
        self.assertEqual(
            [call.function.name for call in result.tool_calls],
            ["get_weather", "get_time"],
        )
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments),
            {
                "city": "北京",
                "days": 3,
                "ratio": 1.5,
                "rain": True,
                "tags": ["hot", "dry"],
                "metadata": {"unit": "c"},
                "nullable": None,
            },
        )
        self.assertEqual(
            json.loads(result.tool_calls[1].function.arguments), {})

    def test_json_object_fallback(self):
        wire = (
            "<dots_function_call>"
            '{"name":"get_weather","arguments":{"city":"上海"}}'
            "</dots_function_call>"
        )
        result = self.make_parser().extract_tool_calls(wire, make_request())

        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments),
            {"city": "上海"},
        )

    def test_string_literal_null_is_preserved(self):
        wire = (
            "<dots_function_call>"
            '<invoke name="get_weather">'
            '<parameter name="city">null</parameter>'
            "</invoke>"
            "</dots_function_call>"
        )

        result = self.make_function_parser(
            make_request(strict=True)).parse_non_stream(wire)
        stream_content, stream_calls, stream_diagnostics = (
            self.stream_with_function_parser(
                wire, make_request(strict=True)))

        self.assertTrue(result.tools_called)
        self.assertFalse(result.has_invalid_tool_block)
        self.assertEqual(
            json.loads(result.valid_tool_calls[0].function.arguments)["city"],
            "null",
        )
        self.assertEqual(stream_content, "")
        self.assertEqual(stream_diagnostics, [])
        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(
            json.loads(stream_calls[0].function.arguments)["city"],
            "null",
        )

    def test_boolean_conversion_requires_an_explicit_value(self):
        for value, expected in (("true", True), ("1", True),
                                ("false", False), ("0", False)):
            with self.subTest(value=value):
                wire = (
                    "<dots_function_call>"
                    '<invoke name="get_weather">'
                    f'<parameter name="rain">{value}</parameter>'
                    "</invoke>"
                    "</dots_function_call>"
                )
                result = self.make_parser().extract_tool_calls(
                    wire, make_request())
                self.assertEqual(
                    json.loads(
                        result.tool_calls[0].function.arguments)["rain"],
                    expected,
                )

        invalid_wire = (
            "<dots_function_call>"
            '<invoke name="get_weather">'
            '<parameter name="rain">maybe</parameter>'
            "</invoke>"
            "</dots_function_call>"
        )
        non_strict = self.make_parser().extract_tool_calls(
            invalid_wire, make_request())
        strict = self.make_function_parser(
            make_request(strict=True)).parse_non_stream(invalid_wire)

        self.assertEqual(
            json.loads(
                non_strict.tool_calls[0].function.arguments)["rain"],
            "maybe",
        )
        self.assertTrue(strict.has_invalid_tool_block)
        self.assertFalse(strict.tools_called)
        self.assertEqual(strict.diagnostics[0].code, "invalid_argument_type")
        self.assertEqual(strict.diagnostics[0].argument_name, "rain")

    def test_non_finite_numbers_are_not_emitted_as_json_constants(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                wire = (
                    "<dots_function_call>"
                    '<invoke name="get_weather">'
                    f'<parameter name="ratio">{value}</parameter>'
                    "</invoke>"
                    "</dots_function_call>"
                )
                non_strict = self.make_parser().extract_tool_calls(
                    wire, make_request())
                arguments = json.loads(
                    non_strict.tool_calls[0].function.arguments)
                strict = self.make_function_parser(
                    make_request(strict=True)).parse_non_stream(wire)

                self.assertEqual(arguments["ratio"], value)
                self.assertTrue(strict.has_invalid_tool_block)
                self.assertFalse(strict.tools_called)
                self.assertEqual(
                    strict.diagnostics[0].code, "invalid_argument_type")
                self.assertEqual(strict.diagnostics[0].argument_name, "ratio")

    def test_json_fallback_rejects_non_finite_constants(self):
        wire = (
            "<dots_function_call>"
            '{"name":"get_weather","arguments":{"ratio":NaN}}'
            "</dots_function_call>"
        )

        with self.assertLogs(level="WARNING") as logs:
            result = self.make_function_parser(
                make_request()).parse_non_stream(wire)

        self.assertTrue(result.has_invalid_tool_block)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.diagnostics[0].code, "malformed_tool_block")
        self.assertIn("non-finite JSON constant", "\n".join(logs.output))

    def test_unknown_tool_is_reported_by_validation_layer(self):
        wire = (
            "<dots_function_call>"
            '<invoke name="get_wearher"></invoke>'
            "</dots_function_call>"
        )
        parser = FunctionCallParser.from_request(
            make_request(),
            tool_parser_name="dots",
            parser=self.make_parser(),
        )
        result = parser.parse_non_stream(wire)

        self.assertTrue(result.has_invalid_tool_block)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.diagnostics[0].code, "invalid_tool_name")
        self.assertEqual(result.diagnostics[0].tool_name, "get_wearher")

    def test_truncated_call_is_not_returned(self):
        result = self.make_parser().extract_tool_calls(
            '<dots_function_call><invoke name="get_time">',
            make_request(),
        )

        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])

    def test_stream_and_non_stream_preserve_all_surrounding_text(self):
        call = (
            "<dots_function_call>"
            '<invoke name="get_time"></invoke>'
            "</dots_function_call>"
        )
        wire = "before\n" + call + "middle" + call + "\nafter"

        non_stream = self.make_parser().extract_tool_calls(
            wire, make_request())
        stream_content, stream_calls, stream_diagnostics = (
            self.stream_with_function_parser(wire, make_request()))

        self.assertEqual(non_stream.content, "before\nmiddle\nafter")
        self.assertEqual(stream_content, non_stream.content)
        self.assertEqual(len(non_stream.tool_calls), 2)
        self.assertEqual(len(stream_calls), 2)
        self.assertEqual(stream_diagnostics, [])

    def test_stream_and_non_stream_hide_trailing_incomplete_block(self):
        complete_call = (
            "<dots_function_call>"
            '<invoke name="get_time"></invoke>'
            "</dots_function_call>"
        )
        wire = (
            "before" + complete_call + "middle"
            '<dots_function_call><invoke name="get_time">'
        )

        non_stream = self.make_parser().extract_tool_calls(
            wire, make_request())
        stream_content, stream_calls, stream_diagnostics = (
            self.stream_with_function_parser(wire, make_request()))

        self.assertEqual(non_stream.content, "beforemiddle")
        self.assertEqual(stream_content, non_stream.content)
        self.assertEqual(len(non_stream.tool_calls), 1)
        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(stream_diagnostics, [])

    def test_strict_stream_rejects_invalid_scalar_values(self):
        cases = (("rain", "maybe"), ("ratio", "NaN"))
        for name, value in cases:
            with self.subTest(name=name, value=value):
                wire = (
                    "<dots_function_call>"
                    '<invoke name="get_weather">'
                    f'<parameter name="{name}">{value}</parameter>'
                    "</invoke>"
                    "</dots_function_call>"
                )

                content, calls, diagnostics = (
                    self.stream_with_function_parser(
                        wire, make_request(strict=True)))

                self.assertEqual(content, "")
                self.assertEqual(calls, [])
                self.assertEqual(len(diagnostics), 1)
                self.assertEqual(
                    diagnostics[0].code, "invalid_argument_type")
                self.assertEqual(diagnostics[0].argument_name, name)

    def test_streaming_is_stable_at_every_boundary(self):
        for chunk_size in (1, 2, 7, 31, len(WIRE)):
            with self.subTest(chunk_size=chunk_size):
                parser = self.make_parser()
                previous = ""
                current = ""
                content = ""
                calls = {}
                for offset in range(0, len(WIRE), chunk_size):
                    chunk = WIRE[offset:offset + chunk_size]
                    current += chunk
                    delta = parser.extract_tool_calls_streaming(
                        previous,
                        current,
                        chunk,
                        [],
                        [],
                        [],
                        make_request(),
                    )
                    previous = current
                    if delta is None:
                        continue
                    content += delta.content or ""
                    for call in delta.tool_calls:
                        state = calls.setdefault(
                            call.index, {"name": None, "arguments": ""})
                        state["name"] = call.function.name or state["name"]
                        state["arguments"] += call.function.arguments or ""

                self.assertEqual(content, "I will check.")
                self.assertNotIn("dots_function_call", content)
                self.assertEqual(
                    [calls[index]["name"] for index in sorted(calls)],
                    ["get_weather", "get_time"],
                )
                self.assertEqual(
                    json.loads(calls[0]["arguments"])["days"], 3)
                self.assertEqual(json.loads(calls[1]["arguments"]), {})


if __name__ == "__main__":
    unittest.main()
