#!/usr/bin/env python3
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fastllm_pytools.openai_server.fastllm_completion import FastLLmCompletion
from tools.fastllm_pytools.openai_server.tool_parsers.hermes_tool_parser import (
    Hermes2ProToolParser,
)
from tools.fastllm_pytools.openai_server.tool_parsers.qwen3coder_tool_parser import (
    Qwen3CoderToolParser,
)

XML_TEMPLATE = "<tool_call><function=NAME><parameter=ARG>VALUE</parameter></function></tool_call>"
JSON_TEMPLATE = '<tool_call>{"name": NAME, "arguments": ARGUMENTS}</tool_call>'


def _completion(
    template=XML_TEMPLATE, parser="auto", force_template=False,
    model_type="qwen4_exp",
):
    completion = FastLLmCompletion.__new__(FastLLmCompletion)
    completion.model = SimpleNamespace(
        hf_tokenizer=SimpleNamespace(chat_template=template),
        tool_call_parser=parser,
        force_chat_template=force_template,
        get_type=lambda: model_type,
    )
    return completion


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


class QwenToolParserSelectionTest(unittest.TestCase):
    def _assert_xml_guidance(self, completion):
        messages = [{"role": "user", "content": "Call get_weather."}]
        tools = _tools()
        before = copy.deepcopy((messages, tools))
        self.assertIs(
            completion._resolve_tool_parser_class(), Qwen3CoderToolParser)
        guided, selected = completion._apply_qwen4_tool_choice(
            messages, tools, "required")
        self.assertEqual((messages, tools), before)
        self.assertEqual(guided[1:], messages)
        self.assertIn("<function=FUNCTION_NAME>", guided[0]["content"])
        self.assertEqual(selected, tools)

    def _assert_no_guidance(self, completion, choice="required", tools=None):
        messages = [{"role": "user", "content": "Call get_weather."}]
        tools = _tools() if tools is None else tools
        guided, selected = completion._apply_qwen4_tool_choice(
            messages, tools, choice)
        self.assertIs(guided, messages)
        self.assertIs(selected, tools)

    def test_auto_xml_parser_enables_guidance(self):
        self._assert_xml_guidance(_completion())

    def test_auto_hermes_parser_does_not_receive_xml_guidance(self):
        completion = _completion(template=JSON_TEMPLATE)
        self.assertIs(
            completion._resolve_tool_parser_class(), Hermes2ProToolParser)
        self._assert_no_guidance(completion)

    def test_explicit_hermes_override_wins_over_xml_template(self):
        completion = _completion(parser="hermes")
        self.assertIs(
            completion._resolve_tool_parser_class(), Hermes2ProToolParser)
        self._assert_no_guidance(completion)

    def test_custom_json_template_auto_detection_disables_xml_guidance(self):
        completion = _completion(template=JSON_TEMPLATE, force_template=True)
        self.assertIs(
            completion._resolve_tool_parser_class(), Hermes2ProToolParser)
        self._assert_no_guidance(completion)

    def test_custom_xml_template_auto_detection_enables_guidance(self):
        self._assert_xml_guidance(_completion(force_template=True))

    def test_explicit_xml_override_wins_over_json_template(self):
        self._assert_xml_guidance(_completion(
            template=JSON_TEMPLATE, parser="qwen3_coder", force_template=True))

    def test_named_tool_with_non_xml_parser_is_not_filtered(self):
        tools = _tools()
        second = copy.deepcopy(tools[0])
        second["function"]["name"] = "get_time"
        tools.append(second)
        self._assert_no_guidance(
            _completion(parser="hermes"),
            choice={"type": "function", "function": {"name": "get_weather"}},
            tools=tools)

    def test_no_tool_auto_and_none_requests_do_not_resolve_parser(self):
        completion = _completion()
        completion.model.hf_tokenizer = None
        messages = [{"role": "user", "content": "Hello."}]
        with patch.object(
            completion, "_resolve_tool_parser_class",
            side_effect=AssertionError("Parser selection must be skipped"),
        ):
            for tools, choice in (
                (None, "required"), ([], "required"),
                (_tools(), "auto"), (_tools(), "none"), (_tools(), None),
            ):
                with self.subTest(choice=choice, has_tools=bool(tools)):
                    guided, selected = completion._apply_qwen4_tool_choice(
                        messages, tools, choice)
                    self.assertIs(guided, messages)
                    self.assertIs(selected, tools)

    def test_other_model_families_are_unchanged(self):
        completion = _completion(model_type="qwen3_moe")
        with patch.object(
            completion, "_resolve_tool_parser_class",
            side_effect=AssertionError("Other models must not be changed"),
        ):
            self._assert_no_guidance(completion)

    def test_qwen_xml_parser_subclasses_receive_guidance(self):
        class CustomQwenParser(Qwen3CoderToolParser):
            pass

        completion = _completion()
        with patch.object(
            completion, "_resolve_tool_parser_class",
            return_value=CustomQwenParser,
        ):
            messages = [{"role": "user", "content": "Call get_weather."}]
            guided, _ = completion._apply_qwen4_tool_choice(
                messages, _tools(), "required")
            self.assertIn("<function=FUNCTION_NAME>", guided[0]["content"])

    def test_parser_creation_uses_the_shared_resolver(self):
        completion = _completion()
        constructor = Mock(return_value=sentinel.parser)
        with patch.object(
            completion, "_resolve_tool_parser_class", return_value=constructor,
        ) as resolver:
            self.assertIs(completion._create_tool_parser(), sentinel.parser)
        resolver.assert_called_once_with()
        constructor.assert_called_once_with(completion.model.hf_tokenizer)


if __name__ == "__main__":
    unittest.main()
