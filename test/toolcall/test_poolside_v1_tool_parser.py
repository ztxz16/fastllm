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


class DummyTokenizer:
    def get_vocab(self):
        return {"<tool_call>": 25, "</tool_call>": 26}


def make_request():
    return ChatCompletionRequest(
        model="laguna",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "mode": {"type": "integer"},
                            "metadata": {"type": "object"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "now",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
        tool_choice="auto",
    )


WIRE = (
    "I will write it.</think>"
    "<tool_call>write_file"
    "<arg_key>path</arg_key><arg_value>/tmp/demo.py</arg_value>"
    "<arg_key>content</arg_key><arg_value>  print(\"你好\")\n  </arg_value>"
    "<arg_key>mode</arg_key><arg_value>420</arg_value>"
    "<arg_key>metadata</arg_key><arg_value>{\"safe\": true}</arg_value>"
    "</tool_call>\n<tool_call>now</tool_call>"
)


class PoolsideV1ToolParserTest(unittest.TestCase):
    def make_parser(self):
        return ToolParserManager.get_tool_parser("poolside_v1")(DummyTokenizer())

    def test_auto_selects_poolside_for_laguna(self):
        parser_cls = ToolParserManager.get_tool_parser_auto(
            "laguna", "<tool_call><arg_key><arg_value>"
        )
        self.assertEqual(parser_cls.__name__, "PoolsideV1ToolParser")

    def test_non_stream_no_newline_mixed_types_and_zero_args(self):
        result = self.make_parser().extract_tool_calls(WIRE, make_request())

        self.assertTrue(result.tools_called)
        self.assertEqual(result.content, "I will write it.</think>")
        self.assertEqual([call.function.name for call in result.tool_calls],
                         ["write_file", "now"])
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments),
            {
                "path": "/tmp/demo.py",
                "content": "  print(\"你好\")\n  ",
                "mode": 420,
                "metadata": {"safe": True},
            },
        )
        self.assertEqual(json.loads(result.tool_calls[1].function.arguments), {})

    def test_truncated_call_is_not_returned_non_stream(self):
        result = self.make_parser().extract_tool_calls(
            "prefix<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp",
            make_request(),
        )
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])

    def test_streaming_is_stable_for_every_byte_split(self):
        for chunk_size in (1, 2, 5, 13, len(WIRE)):
            with self.subTest(chunk_size=chunk_size):
                parser = self.make_parser()
                previous = ""
                current = ""
                content = ""
                calls = {}
                emitted_content_fragments = []
                for offset in range(0, len(WIRE), chunk_size):
                    chunk = WIRE[offset:offset + chunk_size]
                    current += chunk
                    delta = parser.extract_tool_calls_streaming(
                        previous, current, chunk, [], [], [], make_request()
                    )
                    previous = current
                    if delta is None:
                        continue
                    if delta.content:
                        content += delta.content
                        emitted_content_fragments.append(delta.content)
                    for call in delta.tool_calls:
                        state = calls.setdefault(call.index, {"name": None, "args": ""})
                        state["name"] = call.function.name or state["name"]
                        state["args"] += call.function.arguments or ""

                self.assertEqual(content, "I will write it.</think>")
                self.assertEqual([calls[i]["name"] for i in sorted(calls)],
                                 ["write_file", "now"])
                self.assertEqual(
                    json.loads(calls[0]["args"]),
                    {
                        "path": "/tmp/demo.py",
                        "content": "  print(\"你好\")\n  ",
                        "mode": 420,
                        "metadata": {"safe": True},
                    },
                )
                self.assertEqual(json.loads(calls[1]["args"]), {})
                self.assertNotIn("<tool_call>", "".join(emitted_content_fragments))


if __name__ == "__main__":
    unittest.main()
