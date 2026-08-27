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
        model="glm-5.3-flash",
        messages=[],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            },
        }],
    )


WIRE = (
    "需要查询。</think>"
    "<tool_call>get_weather"
    "<arg_key>city</arg_key><arg_value>杭州</arg_value>"
    "<arg_key>days</arg_key><arg_value>2</arg_value>"
    "</tool_call>"
)


class Glm5NextToolParserTest(unittest.TestCase):
    def make_parser(self):
        return ToolParserManager.get_tool_parser("glm47")(DummyTokenizer())

    def test_auto_selects_glm_xml_parser(self):
        parser_cls = ToolParserManager.get_tool_parser_auto(
            "glm5_next", "<tool_call><arg_key><arg_value>")
        self.assertEqual(parser_cls.__name__, "Glm4MoeModelToolParser")

    def test_non_stream_parses_native_glm5_format(self):
        result = self.make_parser().extract_tool_calls(WIRE, make_request())

        self.assertTrue(result.tools_called)
        self.assertEqual(result.content, "需要查询。</think>")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments),
            {"city": "杭州", "days": 2},
        )


if __name__ == "__main__":
    unittest.main()
