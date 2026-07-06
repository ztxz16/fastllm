# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copy from vLLM project

import ast
import json
from collections.abc import Sequence
from typing import Any, Optional, Union

import regex as re
from .abstract_tool_parser import (ToolParser, ToolParserManager, random_tool_call_id)
from ..protocal.openai_protocol import *

import logging
logger = logging.getLogger(__name__)

@ToolParserManager.register_module(["glm45", "glm47"])
class Glm4MoeModelToolParser(ToolParser):

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        self.tool_calls_start_token = self.tool_call_start_token

        self.func_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>",
                                          re.DOTALL)
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL)
        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self._buffer = ""

    def _is_string_type(
            self, tool_name: str, arg_name: str,
            tools: Optional[list[ChatCompletionToolsParam]]) -> bool:
        if tools is None:
            return False
        for tool in tools:
            if tool.function.name == tool_name:
                if tool.function.parameters is None:
                    return False
                arg_type = tool.function.parameters.get(
                    "properties", {}).get(arg_name, {}).get("type", None)
                return arg_type == "string"
        logger.warning("No tool named '%s'.", tool_name)
        return False

    def _deserialize(self, value: str) -> Any:
        try:
            return json.loads(value)
        except Exception:
            pass

        try:
            return ast.literal_eval(value)
        except Exception:
            pass
        return value

    def _parse_tool_call(self, body: str,
                         request: ChatCompletionRequest) -> ToolCall:
        # GLM-5.x places the first <arg_key> immediately after the function
        # name. Older GLM-4.5 templates may insert a newline there.
        first_arg_idx = body.find("<arg_key>")
        if first_arg_idx == -1:
            tc_name = body.strip()
            tc_args = ""
        else:
            tc_name = body[:first_arg_idx].strip()
            tc_args = body[first_arg_idx:]
        if not tc_name:
            raise ValueError("GLM tool call does not contain a function name")

        arg_dct = {}
        for key, value in self.func_arg_regex.findall(tc_args):
            arg_key = key.strip()
            if self._is_string_type(tc_name, arg_key, request.tools):
                arg_val = value
            else:
                arg_val = self._deserialize(value.strip())
            logger.debug("arg_key = %s, arg_val = %s", arg_key, arg_val)
            arg_dct[arg_key] = arg_val

        return ToolCall(
            type="function",
            function=FunctionCall(
                name=tc_name,
                arguments=json.dumps(arg_dct, ensure_ascii=False)))

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matched_tool_calls = self.func_call_regex.findall(model_output)
        logger.debug("model_output: %s", model_output)
        try:
            tool_calls = [
                self._parse_tool_call(match, request)
                for match in matched_tool_calls
            ]
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)
        else:
            if len(tool_calls) > 0:
                content = model_output[:model_output.
                                       find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(tools_called=True,
                                                    tool_calls=tool_calls,
                                                    content=content
                                                    if content else None)
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

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
        if previous_text == "":
            self.current_tool_name_sent = False
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.streamed_args_for_tool = []
            self._buffer = ""

        self._buffer += delta_text
        cur_text = self._buffer
        start_idx = cur_text.find(self.tool_call_start_token)
        if start_idx == -1:
            keep_len = 0
            max_prefix_len = min(len(cur_text),
                                 len(self.tool_call_start_token) - 1)
            for prefix_len in range(max_prefix_len, 0, -1):
                if self.tool_call_start_token.startswith(cur_text[-prefix_len:]):
                    keep_len = prefix_len
                    break
            self._buffer = cur_text[-keep_len:] if keep_len else ""
            cur_text = cur_text[:-keep_len] if keep_len else cur_text
            if self.current_tool_id > 0:
                cur_text = ""
            return DeltaMessage(content=cur_text)
        logger.debug("cur_text = %s", cur_text)
        deltas: list[DeltaToolCall] = []
        text_portion = cur_text[:start_idx]
        parse_pos = start_idx
        while True:
            if cur_text.find(self.tool_call_start_token, parse_pos) != parse_pos:
                break
            body_start = parse_pos + len(self.tool_call_start_token)
            end_idx = cur_text.find(self.tool_call_end_token, body_start)
            if end_idx == -1:
                break
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            tool_call = self._parse_tool_call(cur_text[body_start:end_idx],
                                              request)
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments)
            }
            self.streamed_args_for_tool[
                self.current_tool_id] = tool_call.function.arguments
            deltas.append(
                DeltaToolCall(index=self.current_tool_id,
                              id=tool_call.id,
                              type=tool_call.type,
                              function=DeltaFunctionCall(
                                  name=tool_call.function.name,
                                  arguments=tool_call.function.arguments)))
            self.current_tool_id += 1
            parse_pos = end_idx + len(self.tool_call_end_token)
            while cur_text.startswith(("\n", "\r", " ", "\t"), parse_pos):
                parse_pos += 1

        self._buffer = cur_text[parse_pos:]
        if deltas:
            return DeltaMessage(content=text_portion if text_portion else None,
                                tool_calls=deltas)

        self._buffer = cur_text[start_idx:]
        return DeltaMessage(content=text_portion if text_portion else None)
