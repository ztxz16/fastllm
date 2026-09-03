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
from tools.fastllm_pytools.openai_server.tool_parsers.qwen3coder_tool_parser import (  # noqa: E402
    Qwen3CoderToolParser,
)


class _DummyTokenizer:
    def get_vocab(self):
        return {"<tool_call>": 1, "</tool_call>": 2}


def _write_request():
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "write a file"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["filePath", "content"],
                },
            },
        }],
        stream=True,
    )


def _todo_request(stream=True):
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "write todos"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "todowrite",
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
                                },
                                "required": ["content", "status"],
                            },
                        },
                    },
                    "required": ["todos"],
                },
            },
        }],
        stream=stream,
    )


def _collect_arguments(chunks, request=None):
    calls, _, _ = _collect_stream(chunks, request or _write_request())
    if not calls:
        return "", ""
    return calls[0]["name"], calls[0]["arguments"]


def _merge_stream_delta(parsed, calls):
    if parsed is None:
        return ""
    for tool_call in parsed.tool_calls or []:
        call = calls.setdefault(tool_call.index, {
            "id": None,
            "name": "",
            "arguments": "",
        })
        if tool_call.id:
            call["id"] = tool_call.id
        function = tool_call.function
        if function is not None and function.name:
            call["name"] = function.name
        if function is not None and function.arguments:
            call["arguments"] += function.arguments
    return parsed.content or ""


def _collect_stream(chunks, request):
    parser = Qwen3CoderToolParser(_DummyTokenizer())
    previous = ""
    content = ""
    calls = {}
    for delta in chunks:
        current = previous + delta
        parsed = parser.extract_tool_calls_streaming(
            previous, current, delta, [], [], [], request)
        content += _merge_stream_delta(parsed, calls)
        previous = current

    parsed = parser.extract_tool_calls_streaming(
        previous, previous, "", [], [], [999], request)
    content += _merge_stream_delta(parsed, calls)
    return [calls[index] for index in sorted(calls)], content, parser


def _function_tool(name, properties=None, required=None, strict=False,
                   definitions=None):
    parameters = {
        "type": "object",
        "properties": properties or {},
    }
    if required:
        parameters["required"] = required
    if definitions:
        parameters["$defs"] = definitions
    return {
        "type": "function",
        "function": {
            "name": name,
            "strict": strict,
            "parameters": parameters,
        },
    }


def _request(tools, stream=True, tool_choice="auto",
             parallel_tool_calls=None):
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "use tools"}],
        tools=tools,
        stream=stream,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )


def _wire_call(name, parameters=None):
    body = "".join(
        f"\n<parameter={key}>\n{value}\n</parameter>"
        for key, value in (parameters or [])
    )
    return f"<tool_call>\n<function={name}>{body}\n</function>\n</tool_call>"


class Qwen3CoderToolParserTest(unittest.TestCase):
    def test_complete_tool_call_in_single_delta(self):
        name, arguments = _collect_arguments([
            ("<tool_call>\n<function=write>"
             "\n<parameter=filePath>\n/tmp/a.cpp\n</parameter>"
             "\n<parameter=content>\nint main() {}\n</parameter>"
             "\n</function>\n</tool_call>"),
        ])

        self.assertEqual(name, "write")
        self.assertEqual(json.loads(arguments), {
            "filePath": "/tmp/a.cpp",
            "content": "int main() {}",
        })

    def test_final_parameter_and_function_end_in_same_delta(self):
        content = 'int main() {\n    return "ok" == "ok" ? 0 : 1;\n}\n'
        name, arguments = _collect_arguments([
            "<tool_call>",
            "\n<function=write>",
            "\n<parameter=filePath>\n/tmp/bench.cpp\n</parameter>",
            ("\n<parameter=content>\n" + content +
             "</parameter>\n</function>\n</tool_call>"),
        ])

        self.assertEqual(name, "write")
        self.assertEqual(json.loads(arguments), {
            "filePath": "/tmp/bench.cpp",
            "content": content.rstrip("\n"),
        })

    def test_function_and_multiple_parameters_in_one_delta(self):
        name, arguments = _collect_arguments([
            "<tool_call>",
            ("<function=write>"
             "<parameter=filePath>/tmp/a.cpp</parameter>"
             "<parameter=content>int main() {}</parameter>"
             "</function></tool_call>"),
        ])

        self.assertEqual(name, "write")
        self.assertEqual(json.loads(arguments), {
            "filePath": "/tmp/a.cpp",
            "content": "int main() {}",
        })

    def test_array_parameter_non_streaming(self):
        todos = [{"content": "inspect request", "status": "pending"}]
        parser = Qwen3CoderToolParser(_DummyTokenizer())
        request = _todo_request(stream=False)
        output = (
            "<tool_call>\n<function=todowrite>\n<parameter=todos>\n" +
            json.dumps(todos) +
            "\n</parameter>\n</function>\n</tool_call>")

        parsed = parser.extract_tool_calls(output, request)

        self.assertTrue(parsed.tools_called)
        self.assertEqual(json.loads(parsed.tool_calls[0].function.arguments), {
            "todos": todos,
        })

    def test_array_parameter_streaming(self):
        todos = [{"content": "inspect request", "status": "pending"}]
        name, arguments = _collect_arguments([
            "<tool_call>",
            "\n<function=todowrite>",
            ("\n<parameter=todos>\n" + json.dumps(todos) +
             "\n</parameter>"),
            "\n</function>\n</tool_call>",
        ], request=_todo_request())

        self.assertEqual(name, "todowrite")
        self.assertEqual(json.loads(arguments), {"todos": todos})

    def test_single_call_is_independent_of_every_two_way_split(self):
        wire = _wire_call("write", [
            ("filePath", "/tmp/a.cpp"),
            ("content", "int main() {}"),
        ])
        for split in range(1, len(wire)):
            with self.subTest(split=split):
                calls, _, parser = _collect_stream(
                    [wire[:split], wire[split:]], _write_request())
                self.assertIsNone(parser.streaming_parse_error())
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["name"], "write")
                self.assertEqual(json.loads(calls[0]["arguments"]), {
                    "filePath": "/tmp/a.cpp",
                    "content": "int main() {}",
                })

    def test_parallel_calls_are_independent_of_semantic_splits(self):
        tools = [
            _function_tool("weather", {"city": {"type": "string"}}),
            _function_tool("time", {"zone": {"type": "string"}}),
        ]
        request = _request(tools, parallel_tool_calls=True)
        units = [
            "<tool_call>",
            "\n<function=weather>",
            "\n<parameter=city>\n北京\n</parameter>",
            "\n</function>",
            "\n</tool_call>",
            "\n",
            "<tool_call>",
            "\n<function=time>",
            "\n<parameter=zone>\nAsia/Shanghai\n</parameter>",
            "\n</function>",
            "\n</tool_call>",
        ]

        for mask in range(1 << (len(units) - 1)):
            chunks = []
            current = units[0]
            for boundary, unit in enumerate(units[1:]):
                if mask & (1 << boundary):
                    chunks.append(current)
                    current = unit
                else:
                    current += unit
            chunks.append(current)

            calls, content, parser = _collect_stream(chunks, request)
            with self.subTest(mask=mask, chunks=len(chunks)):
                self.assertIsNone(parser.streaming_parse_error())
                self.assertEqual(content, "")
                self.assertEqual(
                    [(call["name"], json.loads(call["arguments"]))
                     for call in calls],
                    [
                        ("weather", {"city": "北京"}),
                        ("time", {"zone": "Asia/Shanghai"}),
                    ],
                )
                self.assertEqual(len({call["id"] for call in calls}), 2)

    def test_parallel_calls_are_independent_of_every_two_way_split(self):
        request = _request([
            _function_tool("weather", {"city": {"type": "string"}}),
            _function_tool("time", {"zone": {"type": "string"}}),
        ], parallel_tool_calls=True)
        wire = (_wire_call("weather", [("city", "北京")]) + "\n" +
                _wire_call("time", [("zone", "Asia/Shanghai")]))

        for split in range(1, len(wire)):
            with self.subTest(split=split):
                calls, content, parser = _collect_stream(
                    [wire[:split], wire[split:]], request)
                self.assertIsNone(parser.streaming_parse_error())
                self.assertEqual(content, "")
                self.assertEqual(
                    [(call["name"], json.loads(call["arguments"]))
                     for call in calls],
                    [
                        ("weather", {"city": "北京"}),
                        ("time", {"zone": "Asia/Shanghai"}),
                    ],
                )

    def test_mixed_content_matches_non_stream_for_every_split(self):
        request = _request([
            _function_tool("weather", {"city": {"type": "string"}}),
            _function_tool("time", {"zone": {"type": "string"}}),
        ], parallel_tool_calls=True)
        wire = ("Before " + _wire_call("weather", [("city", "北京")]) +
                "\n" + _wire_call("time", [("zone", "Asia/Shanghai")]) +
                " after ")
        non_stream = Qwen3CoderToolParser(
            _DummyTokenizer()).extract_tool_calls(wire, request)
        expected_calls = [
            (call.function.name, json.loads(call.function.arguments))
            for call in non_stream.tool_calls
        ]

        for split in range(1, len(wire)):
            with self.subTest(split=split):
                calls, content, parser = _collect_stream(
                    [wire[:split], wire[split:]], request)
                self.assertIsNone(parser.streaming_parse_error())
                self.assertEqual(content, non_stream.content)
                self.assertEqual(
                    [(call["name"], json.loads(call["arguments"]))
                     for call in calls],
                    expected_calls,
                )

    def test_same_name_parallel_calls_keep_separate_indices(self):
        request = _request([
            _function_tool("weather", {"city": {"type": "string"}}),
        ], parallel_tool_calls=True)
        wire = (_wire_call("weather", [("city", "北京")]) + "\n" +
                _wire_call("weather", [("city", "上海")]))

        calls, _, parser = _collect_stream([wire], request)

        self.assertEqual(
            [json.loads(call["arguments"]) for call in calls],
            [{"city": "北京"}, {"city": "上海"}],
        )
        self.assertEqual(len(parser.prev_tool_call_arr), 2)

    def test_stream_preserves_content_outside_tool_blocks(self):
        request = _request([
            _function_tool("weather", {"city": {"type": "string"}}),
        ])
        wire = "Before " + _wire_call("weather", [("city", "北京")]) + " after"
        split = wire.index("<tool_call>") + 6

        calls, content, _ = _collect_stream(
            [wire[:split], wire[split:-3], wire[-3:]], request)

        self.assertEqual(content, "Before  after")
        self.assertEqual(len(calls), 1)

    def test_protocol_tags_inside_string_argument_are_data(self):
        value = (
            "Document <tool_call> and </tool_call> plus <function=x> and "
            "</function>, then <parameter=x> and </parameter> safely; "
            "final literal: </parameter>"
        )
        wire = _wire_call("write", [
            ("filePath", "/tmp/tags.txt"),
            ("content", value),
        ])
        request = _write_request()

        non_stream = Qwen3CoderToolParser(
            _DummyTokenizer()).extract_tool_calls(wire, request)
        self.assertTrue(non_stream.tools_called)
        self.assertEqual(
            json.loads(non_stream.tool_calls[0].function.arguments),
            {"filePath": "/tmp/tags.txt", "content": value},
        )

        for split in range(1, len(wire)):
            with self.subTest(split=split):
                calls, content, parser = _collect_stream(
                    [wire[:split], wire[split:]], request)
                self.assertIsNone(parser.streaming_parse_error())
                self.assertEqual(content, "")
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(calls[0]["arguments"]), {
                    "filePath": "/tmp/tags.txt",
                    "content": value,
                })

    def test_non_stream_requires_complete_tool_markup(self):
        request = _request([
            _function_tool("weather", {"city": {"type": "string"}}),
        ], stream=False)
        malformed = [
            "<tool_call><function=weather><parameter=city>北京</parameter>"
            "</function>",
            "<tool_call><function=weather><parameter=city>北京</parameter>",
            "<tool_call><function=weather><parameter=city>北京",
            "<tool_call><function=weather></function></tool_call>"
            "<tool_call><function=weather>",
        ]

        for wire in malformed:
            with self.subTest(wire=wire):
                parser = Qwen3CoderToolParser(_DummyTokenizer())
                result = parser.extract_tool_calls(wire, request)
                self.assertFalse(result.tools_called)
                self.assertEqual(result.tool_calls, [])

    def test_bare_function_markup_is_plain_content(self):
        request = _request([_function_tool("ping")], stream=False)
        wire = "Example syntax: <function=ping></function>"
        parser = Qwen3CoderToolParser(_DummyTokenizer())

        result = parser.extract_tool_calls(wire, request)

        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, wire)

    def test_schema_combinators_and_references_preserve_json_types(self):
        properties = {
            "direct": {"type": "array", "items": {"type": "integer"}},
            "union": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
            },
            "any_of": {
                "anyOf": [
                    {"type": "array", "items": {"type": "integer"}},
                    {"type": "null"},
                ],
            },
            "one_of": {
                "oneOf": [{"type": "integer"}, {"type": "string"}],
            },
            "reference": {"$ref": "#/$defs/payload"},
            "literal_null": {"type": "string"},
        }
        request = _request([
            _function_tool(
                "typed",
                properties,
                definitions={"payload": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                }},
            ),
        ], stream=False)
        wire = _wire_call("typed", [
            ("direct", "[1, 2]"),
            ("union", "[3, 4]"),
            ("any_of", "[5, 6]"),
            ("one_of", "42"),
            ("reference", '{"value": 7}'),
            ("literal_null", "null"),
        ])

        parsed = Qwen3CoderToolParser(_DummyTokenizer()).extract_tool_calls(
            wire, request)

        self.assertTrue(parsed.tools_called)
        self.assertEqual(json.loads(parsed.tool_calls[0].function.arguments), {
            "direct": [1, 2],
            "union": [3, 4],
            "any_of": [5, 6],
            "one_of": 42,
            "reference": {"value": 7},
            "literal_null": "null",
        })

    def test_invalid_scalars_are_not_silently_changed_or_raised(self):
        properties = {
            "flag": {"type": "boolean"},
            "number": {"type": "number"},
        }
        request = _request([
            _function_tool("typed", properties),
        ])

        for number in ("NaN", "Infinity", "-Infinity", "1e309"):
            with self.subTest(number=number):
                wire = _wire_call("typed", [
                    ("flag", "maybe"),
                    ("number", number),
                ])
                calls, _, parser = _collect_stream([wire], request)
                self.assertIsNone(parser.streaming_parse_error())
                self.assertEqual(json.loads(calls[0]["arguments"]), {
                    "flag": "maybe",
                    "number": number,
                })

    def test_nonfinite_values_nested_in_json_are_preserved_for_validation(self):
        request = _request([
            _function_tool("typed", {
                "items": {"type": "array"},
                "payload": {"type": "object"},
            }),
        ])
        invalid_values = [
            ("[1e309]", '{"value":1e309}'),
            ("[NaN]", '{"value":Infinity}'),
        ]

        for items, payload in invalid_values:
            with self.subTest(items=items, payload=payload):
                wire = _wire_call("typed", [
                    ("items", items),
                    ("payload", payload),
                ])
                calls, _, parser = _collect_stream([wire], request)
                self.assertIsNone(parser.streaming_parse_error())
                self.assertEqual(json.loads(calls[0]["arguments"]), {
                    "items": items,
                    "payload": payload,
                })

    def test_boolean_conversion_ignores_protocol_padding(self):
        request = _request([
            _function_tool("typed", {"flag": {"type": "boolean"}}),
        ])
        calls, _, parser = _collect_stream([
            _wire_call("typed", [("flag", "  false  ")]),
        ], request)

        self.assertIsNone(parser.streaming_parse_error())
        self.assertEqual(json.loads(calls[0]["arguments"]), {"flag": False})

    def test_invalid_converted_values_are_rejected_by_strict_facade(self):
        request = _request([
            _function_tool(
                "typed",
                {
                    "flag": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "number": {"type": "number"},
                },
                required=["flag", "count", "number"],
                strict=True,
            ),
        ], stream=False)
        parser = FunctionCallParser.from_request(
            request,
            tool_parser_name="qwen3_coder",
            tokenizer=_DummyTokenizer(),
        )
        wire = _wire_call("typed", [
            ("flag", "maybe"),
            ("count", "null"),
            ("number", "1e309"),
        ])

        result = parser.parse_non_stream(wire)

        self.assertFalse(result.tools_called)
        self.assertEqual(
            {diagnostic.argument_name for diagnostic in result.diagnostics},
            {"flag", "count", "number"},
        )
        self.assertTrue(all(
            diagnostic.code == "invalid_argument_type"
            for diagnostic in result.diagnostics
        ))

    def test_parallel_false_detects_two_calls_from_one_delta(self):
        request = _request([
            _function_tool("weather", {"city": {"type": "string"}}),
            _function_tool("time", {"zone": {"type": "string"}}),
        ], parallel_tool_calls=False)
        parser = FunctionCallParser.from_request(
            request,
            tool_parser_name="qwen3_coder",
            tokenizer=_DummyTokenizer(),
        )
        wire = (_wire_call("weather", [("city", "北京")]) + "\n" +
                _wire_call("time", [("zone", "Asia/Shanghai")]))

        result = parser.parse_stream_chunk(
            previous_text="",
            current_text=wire,
            delta_text=wire,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )

        self.assertEqual(len(result.valid_tool_calls), 1)
        self.assertEqual(result.valid_tool_calls[0].function.name, "weather")
        self.assertEqual(len(result.invalid_tool_calls), 1)
        self.assertTrue(any(
            diagnostic.code == "parallel_tool_calls_violation"
            for diagnostic in result.diagnostics
        ))

    def test_stream_flushes_plain_text_that_looks_like_partial_marker(self):
        request = _request([
            _function_tool("weather", {"city": {"type": "string"}}),
        ])
        parser = FunctionCallParser.from_request(
            request,
            tool_parser_name="qwen3_coder",
            tokenizer=_DummyTokenizer(),
        )
        text = "ordinary response ending in <tool"

        result = parser.parse_stream_chunk(
            previous_text="",
            current_text=text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        flushed = parser.flush_stream_tool_calls()

        self.assertEqual((result.content or "") + (flushed.content or ""),
                         text)
        self.assertEqual(flushed.diagnostics, [])

    def test_stream_finalization_is_idempotent_for_malformed_block(self):
        request = _request([
            _function_tool("weather", {"city": {"type": "string"}}),
        ])
        parser = FunctionCallParser.from_request(
            request,
            tool_parser_name="qwen3_coder",
            tokenizer=_DummyTokenizer(),
        )
        wire = "<tool_call><function=weather>"
        parser.parse_stream_chunk(
            previous_text="",
            current_text=wire,
            delta_text=wire,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )

        first = parser.finalize_stream()
        flushed = parser.flush_stream_tool_calls()

        self.assertTrue(any(diagnostic.code == "malformed_tool_block"
                            for diagnostic in first))
        self.assertTrue(flushed.has_invalid_tool_block)
        self.assertEqual(flushed.diagnostics, first)

    def test_incomplete_stream_reports_malformed_tool_block(self):
        request = _request([
            _function_tool(
                "weather",
                {"city": {"type": "string"}},
                required=["city"],
                strict=True,
            ),
        ], tool_choice="required")
        parser = FunctionCallParser.from_request(
            request,
            tool_parser_name="qwen3_coder",
            tokenizer=_DummyTokenizer(),
        )
        wire = (_wire_call("weather", [("city", "北京")]) +
                "<tool_call><function=weather>")

        parsed = parser.parse_stream_chunk(
            previous_text="",
            current_text=wire,
            delta_text=wire,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        diagnostics = parser.finalize_stream()

        self.assertEqual(len(parsed.valid_tool_calls), 0)
        self.assertTrue(any(diagnostic.code == "malformed_tool_block"
                            for diagnostic in diagnostics))


if __name__ == "__main__":
    unittest.main()
