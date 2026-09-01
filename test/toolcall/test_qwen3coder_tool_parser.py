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


def _collect_arguments(chunks):
    parser = Qwen3CoderToolParser(_DummyTokenizer())
    request = _write_request()
    previous = ""
    tool_name = ""
    arguments = ""
    for delta in chunks:
        current = previous + delta
        parsed = parser.extract_tool_calls_streaming(
            previous, current, delta, [], [], [], request)
        if parsed is not None:
            for tool_call in parsed.tool_calls or []:
                function = tool_call.function
                if function is not None and function.name:
                    tool_name = function.name
                if function is not None and function.arguments:
                    arguments += function.arguments
        previous = current
    return tool_name, arguments


class Qwen3CoderToolParserTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
