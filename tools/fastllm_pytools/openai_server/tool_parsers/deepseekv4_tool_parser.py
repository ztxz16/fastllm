# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import uuid
from collections.abc import Sequence
from typing import Any, Optional, Union

import regex as re

from .abstract_tool_parser import ToolParser, ToolParserManager
from ..protocal.openai_protocol import *

import logging
logger = logging.getLogger(__name__)


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


def _partial_tags_overlap(text: str, tags: list[str]) -> int:
    return max((_partial_tag_overlap(text, tag) for tag in tags), default=0)


@ToolParserManager.register_module("deepseek_v4")
class DeepSeekV4ToolParser(ToolParser):
    """
    DeepSeek V4 DSML tool parser.

    Example:
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    """

    tool_call_start_token: str = "<｜DSML｜tool_calls>"
    tool_call_end_token: str = "</｜DSML｜tool_calls>"
    alt_tool_call_start_token: str = "<\\DSML\\tool_calls>"
    alt_tool_call_end_token: str = "</\\DSML\\tool_calls>"

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self.current_tool_index: int = 0
        self._sent_content_idx: int = 0
        self.tool_call_start_tokens = [
            self.tool_call_start_token,
            self.alt_tool_call_start_token,
        ]
        dsml_tag_regex = r"(?:｜DSML｜|\\DSML\\)"

        self.tool_call_complete_regex = re.compile(
            r"<" + dsml_tag_regex + r"tool_calls>"
            + r"(.*?)"
            + r"</" + dsml_tag_regex + r"tool_calls>",
            re.DOTALL,
        )
        self.invoke_complete_regex = re.compile(
            r'<'
            + dsml_tag_regex
            + r'invoke\s+name="([^"]+)"\s*>(.*?)</'
            + dsml_tag_regex
            + r'invoke>',
            re.DOTALL,
        )
        self.parameter_complete_regex = re.compile(
            r'<'
            + dsml_tag_regex
            + r'parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</'
            + dsml_tag_regex
            + r'parameter>',
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

    def get_token_ids(self, text: str) -> list[int]:
        return [0]

    def _generate_tool_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    def _decode_param_value(self, value: str, is_string: str) -> Any:
        if is_string == "true":
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _parse_invoke_params(self, invoke_str: str) -> dict[str, Any]:
        param_dict: dict[str, Any] = {}
        for param_name, is_string, param_val in self.parameter_complete_regex.findall(invoke_str):
            param_dict[param_name] = self._decode_param_value(param_val, is_string)
        return param_dict

    def _find_tool_call_start(self, text: str) -> int:
        positions = [
            text.find(token) for token in self.tool_call_start_tokens
            if text.find(token) >= 0
        ]
        return min(positions) if positions else -1

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        first_tool_idx = self._find_tool_call_start(model_output)
        if first_tool_idx < 0:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

        try:
            tool_calls = []
            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                for invoke_name, invoke_content in self.invoke_complete_regex.findall(
                    tool_call_match):
                    param_dict = self._parse_invoke_params(invoke_content)
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=invoke_name,
                                arguments=json.dumps(param_dict, ensure_ascii=False),
                            ),
                        ))

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output)

            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content)
        except Exception:
            logger.exception("Error extracting DeepSeek V4 tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

    def _reset_streaming_state(self):
        self.current_tool_index = 0
        self._sent_content_idx = 0
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    def _extract_delta_tool_calls(
        self,
        current_text: str,
    ) -> list[DeltaToolCall]:
        complete_invokes = self.invoke_complete_regex.findall(current_text)
        delta_tool_calls: list[DeltaToolCall] = []

        while len(complete_invokes) > self.current_tool_index:
            invoke_name, invoke_body = complete_invokes[self.current_tool_index]
            param_dict = self._parse_invoke_params(invoke_body)
            args_json = json.dumps(param_dict, ensure_ascii=False)
            idx = self.current_tool_index
            self.current_tool_index += 1

            self.prev_tool_call_arr.append(
                {"name": invoke_name, "arguments": param_dict})
            self.streamed_args_for_tool.append(args_json)

            delta_tool_calls.append(
                DeltaToolCall(
                    index=idx,
                    id=self._generate_tool_call_id(),
                    type="function",
                    function=DeltaFunctionCall(
                        name=invoke_name,
                        arguments=args_json,
                    ),
                ))

        return delta_tool_calls

    def _extract_content(self, current_text: str) -> Optional[str]:
        first_tool_idx = self._find_tool_call_start(current_text)
        if first_tool_idx < 0:
            overlap = _partial_tags_overlap(current_text, self.tool_call_start_tokens)
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = first_tool_idx

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx:sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        if not previous_text:
            self._reset_streaming_state()

        content = self._extract_content(current_text)
        delta_tool_calls = self._extract_delta_tool_calls(current_text)

        if delta_tool_calls or content:
            return DeltaMessage(content=content, tool_calls=delta_tool_calls)

        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")

        return None
