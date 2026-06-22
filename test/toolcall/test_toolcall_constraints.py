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
from tools.fastllm_pytools.openai_server.toolcall_constraints import (  # noqa: E402
    apply_tool_call_constraint_to_decoder,
    compile_tool_call_constraint,
)
from tools.fastllm_pytools.openai_server.toolcall_parser import (  # noqa: E402
    FunctionCallParser,
)


def _weather_tool(strict=False):
    function = {
        "name": "get_weather",
        "description": "Get weather.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    }
    if strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def _time_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get time.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        },
    }


def _request(tools=None, tool_choice="auto", parallel_tool_calls=None):
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "查工具"}],
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )


def _descriptor(tools=None, tool_choice="auto", parallel_tool_calls=None):
    return FunctionCallParser.build_constraint_descriptor_from_request(
        _request(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        ))


class _ToolCallConstraintDecoder:
    def __init__(self):
        self.payload = None

    def set_tool_call_constraint(self, payload):
        self.payload = payload


class _StructuralTagDecoder:
    def __init__(self):
        self.payload = None

    def set_structural_tag(self, payload):
        self.payload = payload


class _StructuredOutputsDecoder:
    def __init__(self):
        self.payload = None

    def set_structured_outputs(self, payload):
        self.payload = payload


class _UnsupportedDecoder:
    pass


class ToolCallConstraintCompilerTest(unittest.TestCase):
    def test_compile_deepseek_v4_structural_tag_and_name_grammar(self):
        spec = compile_tool_call_constraint(
            _descriptor(tools=[_weather_tool(), _time_tool()]))

        self.assertEqual(spec.backend, "fastllm_toolcall_prototype")
        self.assertEqual(spec.structural_tag["format"], "deepseek_v4_dsml")
        self.assertEqual(spec.structural_tag["invoke"]["allowed_names"],
                         ["get_weather", "get_time"])
        self.assertIn('"get_weather"', spec.name_grammar)
        self.assertIn('"get_time"', spec.name_grammar)
        self.assertIn("<｜DSML｜tool_calls>", spec.name_grammar)
        json.dumps(spec.to_dict(), ensure_ascii=False)

    def test_named_tool_choice_restricts_name_enum(self):
        spec = compile_tool_call_constraint(
            _descriptor(
                tools=[_weather_tool(), _time_tool()],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_time"},
                },
            ))

        self.assertEqual(spec.structural_tag["invoke"]["allowed_names"],
                         ["get_time"])
        self.assertIn('"get_time"', spec.name_grammar)
        self.assertNotIn('"get_weather"', spec.name_grammar)

    def test_strict_schema_is_copied_to_spec(self):
        tool = _weather_tool(strict=True)
        descriptor = _descriptor(tools=[tool])
        spec = compile_tool_call_constraint(descriptor)
        tool["function"]["parameters"]["required"].append("days")

        self.assertEqual(
            spec.json_schemas["get_weather"]["required"], ["city"])
        spec_dict = spec.to_dict()
        spec_dict["json_schemas"]["get_weather"]["required"].append("days")
        self.assertEqual(
            spec.json_schemas["get_weather"]["required"], ["city"])

    def test_parallel_false_sets_max_tool_calls(self):
        spec = compile_tool_call_constraint(
            _descriptor(
                tools=[_weather_tool(), _time_tool()],
                parallel_tool_calls=False,
            ))

        self.assertEqual(spec.structural_tag["max_tool_calls"], 1)

    def test_non_deepseek_descriptor_does_not_emit_dsml_grammar(self):
        descriptor = _descriptor(tools=[_weather_tool()]).to_dict()
        descriptor["constraint_type"] = "qwen_tool_call"
        descriptor["model_type"] = "qwen"

        spec = compile_tool_call_constraint(descriptor)

        self.assertIsNone(spec.structural_tag)
        self.assertIsNone(spec.name_grammar)
        self.assertIn("no structural_tag prototype", spec.notes[-1])

    def test_tool_call_constraint_decoder_accepts_spec(self):
        decoder = _ToolCallConstraintDecoder()
        result = apply_tool_call_constraint_to_decoder(
            decoder, _descriptor(tools=[_weather_tool()]))

        self.assertTrue(result.applied)
        self.assertEqual(result.mode, "tool_call_constraint")
        self.assertEqual(decoder.payload["backend"],
                         "fastllm_toolcall_prototype")

    def test_structural_tag_decoder_accepts_structural_tag(self):
        decoder = _StructuralTagDecoder()
        result = apply_tool_call_constraint_to_decoder(
            decoder, _descriptor(tools=[_weather_tool()]))

        self.assertTrue(result.applied)
        self.assertEqual(result.mode, "structural_tag")
        self.assertEqual(decoder.payload["format"], "deepseek_v4_dsml")

    def test_structured_outputs_decoder_accepts_structural_tag_json(self):
        decoder = _StructuredOutputsDecoder()
        result = apply_tool_call_constraint_to_decoder(
            decoder, _descriptor(tools=[_weather_tool()]))

        self.assertTrue(result.applied)
        self.assertEqual(result.mode, "structured_outputs")
        payload = json.loads(decoder.payload["structural_tag"])
        self.assertEqual(payload["format"], "deepseek_v4_dsml")

    def test_unsupported_decoder_does_not_crash(self):
        decoder = _UnsupportedDecoder()
        result = apply_tool_call_constraint_to_decoder(
            decoder, _descriptor(tools=[_weather_tool()]))

        self.assertFalse(result.applied)
        self.assertIsNone(result.mode)
        self.assertIsNotNone(result.spec)
        self.assertIn("does not expose", result.message)

    def test_no_descriptor_is_not_applied(self):
        decoder = _ToolCallConstraintDecoder()
        result = apply_tool_call_constraint_to_decoder(decoder, None)

        self.assertFalse(result.applied)
        self.assertIsNone(decoder.payload)
        self.assertIsNone(result.spec)


if __name__ == "__main__":
    unittest.main()
