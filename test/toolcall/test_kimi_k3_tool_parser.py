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
from tools.fastllm_pytools.openai_server.tool_parsers import (  # noqa: E402
    ToolParserManager,
)


OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"


def open_tag(tag, attrs=""):
    attrs = f" {attrs}" if attrs else ""
    return f"{OPEN}{tag}{attrs}{SEP}"


def close_tag(tag):
    return f"{CLOSE}{tag}{SEP}"


def argument(key, argument_type, value):
    return (
        open_tag("argument", f'key="{key}" type="{argument_type}"')
        + value
        + close_tag("argument")
    )


def call(name, index, *arguments):
    return (
        open_tag("call", f'tool="{name}" index="{index}"')
        + "".join(arguments)
        + close_tag("call")
    )


WIRE = (
    open_tag("response")
    + "I will check."
    + close_tag("response")
    + open_tag("tools")
    + call(
        "get_weather",
        1,
        argument("city", "string", "北京"),
        argument("days", "number", "3"),
        argument("options", "object", '{"metric":true}'),
    )
    + call("get_time", 2)
    + close_tag("tools")
    + close_tag("message")
)


class DummyTokenizer:
    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def request():
    return ChatCompletionRequest(
        model="kimi-k3",
        messages=[{"role": "user", "content": "北京天气如何？"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "number"},
                            "options": {"type": "object"},
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
    )


class KimiK3ToolParserTest(unittest.TestCase):
    def make_parser(self):
        return ToolParserManager.get_tool_parser("kimi_k3")(DummyTokenizer())

    def test_default_tool_choice_and_auto_parser(self):
        self.assertEqual(request().tool_choice, "auto")
        parser_class = ToolParserManager.get_tool_parser_auto(
            "kimi_k3", None)
        self.assertEqual(parser_class.__name__, "KimiK3ToolParser")

    def test_non_stream_xtml_is_unwrapped_and_typed(self):
        result = self.make_parser().extract_tool_calls(WIRE, request())

        self.assertTrue(result.tools_called)
        self.assertEqual(result.content, "I will check.")
        self.assertEqual(
            [tool_call.function.name for tool_call in result.tool_calls],
            ["get_weather", "get_time"],
        )
        self.assertEqual(
            [tool_call.id for tool_call in result.tool_calls],
            ["get_weather:0", "get_time:1"],
        )
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments),
            {
                "city": "北京",
                "days": 3,
                "options": {"metric": True},
            },
        )
        self.assertEqual(
            json.loads(result.tool_calls[1].function.arguments), {})

    def test_plain_response_does_not_leak_close_markers(self):
        wire = (
            open_tag("response")
            + "answer"
            + close_tag("response")
            + close_tag("message")
        )
        result = self.make_parser().extract_tool_calls(wire, request())

        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, "answer")

    def test_streaming_is_safe_at_every_byte_boundary(self):
        for chunk_size in (1, 2, 5, 17, len(WIRE)):
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
                        request(),
                    )
                    previous = current
                    if delta is None:
                        continue
                    content += delta.content or ""
                    for tool_call in delta.tool_calls:
                        calls[tool_call.index] = tool_call

                self.assertEqual(content, "I will check.")
                self.assertNotIn("<|", content)
                self.assertEqual(
                    [calls[index].function.name for index in sorted(calls)],
                    ["get_weather", "get_time"],
                )
                self.assertEqual(
                    json.loads(calls[0].function.arguments)["city"], "北京")


if __name__ == "__main__":
    unittest.main()
