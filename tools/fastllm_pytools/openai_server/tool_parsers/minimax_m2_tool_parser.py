# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project (Adapter for fastllm)

import json
import uuid
from collections.abc import Sequence
from typing import Any, Optional, Union

import regex as re
from .abstract_tool_parser import (ToolParser, ToolParserManager,
                                   random_tool_call_id)
from ..protocal.openai_protocol import *

import logging
logger = logging.getLogger(__name__)


@ToolParserManager.register_module(["minimax_m2"])
class MinimaxM2ToolParser(ToolParser):

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"
        self.invoke_start_prefix: str = "<invoke name="
        self.invoke_end_token: str = "</invoke>"
        self.parameter_prefix: str = "<parameter name="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        self.current_tool_index: int = 0
        self.header_sent: bool = False
        self.current_tool_string_id: Optional[str] = None
        self.current_function_name: Optional[str] = None
        self.current_param_name: Optional[str] = None
        self.current_param_value: str = ""
        self.param_count: int = 0
        self.in_param: bool = False
        self.in_function: bool = False
        self.accumulated_text: str = ""
        self.json_started: bool = False
        self.json_closed: bool = False
        self.accumulated_params: dict = {}

        self._reset_streaming_state()

        self.tool_call_complete_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
        self.invoke_complete_regex = re.compile(
            r"<invoke name=(.*?)</invoke>", re.DOTALL)
        self.parameter_complete_regex = re.compile(
            r"<parameter name=(.*?)</parameter>", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (self.tool_call_start_token_id is None
                or self.tool_call_end_token_id is None):
            raise RuntimeError(
                "MiniMax M2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")

    def _generate_tool_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_string_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.accumulated_params = {}
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    def _extract_name(self, name_str: str) -> str:
        name_str = name_str.strip()
        if ((name_str.startswith('"') and name_str.endswith('"'))
                or (name_str.startswith("'") and name_str.endswith("'"))):
            return name_str[1:-1]
        return name_str

    def _convert_param_value(self, value: str, param_name: str,
                             param_config: dict,
                             func_name: str) -> Any:
        if value.lower() in ("null", "none", "nil"):
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the "
                    "string value.", param_name, func_name)
            return value

        schema = param_config[param_name]
        if not isinstance(schema, dict):
            return value

        param_types = self._extract_types_from_schema(schema)
        return self._convert_value_with_types(value, param_types)

    def _extract_types_from_schema(self, schema: Any) -> list[str]:
        if schema is None or not isinstance(schema, dict):
            return ["string"]

        types: set[str] = set()

        if "type" in schema:
            type_value = schema["type"]
            if isinstance(type_value, str):
                types.add(type_value)
            elif isinstance(type_value, list):
                for t in type_value:
                    if isinstance(t, str):
                        types.add(t)

        if "enum" in schema and isinstance(schema["enum"], list):
            for val in schema["enum"]:
                if val is None:
                    types.add("null")
                elif isinstance(val, bool):
                    types.add("boolean")
                elif isinstance(val, int):
                    types.add("integer")
                elif isinstance(val, float):
                    types.add("number")
                elif isinstance(val, str):
                    types.add("string")
                elif isinstance(val, list):
                    types.add("array")
                elif isinstance(val, dict):
                    types.add("object")

        for choice_field in ("anyOf", "oneOf", "allOf"):
            if choice_field in schema and isinstance(schema[choice_field],
                                                     list):
                for choice in schema[choice_field]:
                    types.update(self._extract_types_from_schema(choice))

        if not types:
            return ["string"]
        return list(types)

    def _convert_value_with_types(self, value: str,
                                  param_types: list[str]) -> Any:
        if value.lower() in ("null", "none", "nil"):
            return None

        normalized = [t.lower() for t in param_types]
        type_priority = [
            "integer", "int", "number", "float", "boolean", "bool",
            "object", "array", "string", "str", "text",
        ]

        for ptype in type_priority:
            if ptype not in normalized:
                continue
            if ptype in ["string", "str", "text"]:
                return value
            elif ptype in ["integer", "int"]:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
            elif ptype in ["number", "float"]:
                try:
                    val = float(value)
                    return val if val != int(val) else int(val)
                except (ValueError, TypeError):
                    continue
            elif ptype in ["boolean", "bool"]:
                lower_val = value.lower().strip()
                if lower_val in ["true", "1", "yes", "on"]:
                    return True
                elif lower_val in ["false", "0", "no", "off"]:
                    return False
                continue
            elif ptype in ["object", "array"]:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    continue

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _parse_single_invoke(
            self, invoke_str: str,
            tools: Optional[list[ChatCompletionToolsParam]]
    ) -> Optional[ToolCall]:
        name_match = re.search(r"^([^>]+)", invoke_str)
        if not name_match:
            return None

        function_name = self._extract_name(name_match.group(1))

        param_config = {}
        if tools:
            for tool in tools:
                if (hasattr(tool, "function")
                        and tool.function.name == function_name
                        and hasattr(tool.function, "parameters")):
                    params = tool.function.parameters
                    if isinstance(params, dict) and "properties" in params:
                        param_config = params["properties"]
                    break

        param_dict = {}
        for match in self.parameter_complete_regex.findall(invoke_str):
            param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
            if param_match:
                param_name = self._extract_name(param_match.group(1))
                param_value = param_match.group(2).strip()
                if param_value.startswith("\n"):
                    param_value = param_value[1:]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                param_dict[param_name] = self._convert_param_value(
                    param_value, param_name, param_config, function_name)

        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name,
                arguments=json.dumps(param_dict, ensure_ascii=False)),
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            tool_calls = []

            for tool_call_match in self.tool_call_complete_regex.findall(
                    model_output):
                for invoke_match in self.invoke_complete_regex.findall(
                        tool_call_match):
                    tool_call = self._parse_single_invoke(
                        invoke_match, request.tools if request else None)
                    if tool_call:
                        tool_calls.append(tool_call)

            if not tool_calls:
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

            self.prev_tool_call_arr.clear()
            for tool_call in tool_calls:
                self.prev_tool_call_arr.append({
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                })

            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = (model_output[:first_tool_idx]
                       if first_tool_idx > 0 else None)

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        except Exception:
            logger.exception("Error extracting tool calls")
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
        if not delta_text:
            if (delta_token_ids
                    and self.tool_call_end_token_id not in delta_token_ids):
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text))
                if (complete_calls > 0
                        and len(self.prev_tool_call_arr) > 0):
                    open_calls = (
                        current_text.count(self.tool_call_start_token) -
                        current_text.count(self.tool_call_end_token))
                    if open_calls == 0:
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    return DeltaMessage(content="")
            return None

        if not previous_text:
            self._reset_streaming_state()

        self.accumulated_text = current_text

        if self.json_closed and not self.in_function:
            invoke_ends = current_text.count(self.invoke_end_token)
            if invoke_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.in_function = False
                self.accumulated_params = {}
                return None

        if not self.is_tool_call_started:
            if (self.tool_call_start_token_id in delta_token_ids
                    or self.tool_call_start_token in delta_text):
                self.is_tool_call_started = True
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[:delta_text.index(
                        self.tool_call_start_token)]
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                if (current_text.rstrip().endswith(self.tool_call_end_token)
                        and delta_text.strip() == ""):
                    return None
                return DeltaMessage(content=delta_text)

        invoke_starts_count = current_text.count(self.invoke_start_prefix)
        if self.current_tool_index >= invoke_starts_count:
            return None

        invoke_start_positions: list[int] = []
        idx = 0
        while True:
            idx = current_text.find(self.invoke_start_prefix, idx)
            if idx == -1:
                break
            invoke_start_positions.append(idx)
            idx += len(self.invoke_start_prefix)

        if self.current_tool_index >= len(invoke_start_positions):
            return None

        invoke_start_idx = invoke_start_positions[self.current_tool_index]
        invoke_end_idx = current_text.find(self.invoke_end_token,
                                           invoke_start_idx)
        if invoke_end_idx == -1:
            tool_text = current_text[invoke_start_idx:]
        else:
            tool_text = current_text[
                invoke_start_idx:invoke_end_idx + len(self.invoke_end_token)]

        if not self.header_sent:
            if self.invoke_start_prefix in tool_text:
                func_start = (tool_text.find(self.invoke_start_prefix) +
                              len(self.invoke_start_prefix))
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    function_name_raw = tool_text[func_start:func_end]
                    self.current_function_name = self._extract_name(
                        function_name_raw)
                    self.current_tool_string_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    if len(self.prev_tool_call_arr) <= self.current_tool_index:
                        self.prev_tool_call_arr.append({
                            "name": self.current_function_name,
                            "arguments": "{}",
                        })
                        if len(self.streamed_args_for_tool
                               ) <= self.current_tool_index:
                            self.streamed_args_for_tool.append("")

                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            id=self.current_tool_string_id,
                            function=DeltaFunctionCall(
                                name=self.current_function_name,
                                arguments=""),
                            type="function",
                        )
                    ])
            return None

        if self.in_function:
            if not self.json_started:
                self.json_started = True
                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[
                        self.current_tool_index] += "{"
                return DeltaMessage(tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        function=DeltaFunctionCall(arguments="{"),
                    )
                ])

            if not self.json_closed and self.invoke_end_token in tool_text:
                total_param_count = tool_text.count(self.parameter_prefix)

                if self.param_count >= total_param_count:
                    self.json_closed = True

                    invoke_start = (
                        tool_text.find(self.invoke_start_prefix) +
                        len(self.invoke_start_prefix))
                    invoke_content_end = tool_text.find(
                        self.invoke_end_token, invoke_start)
                    if invoke_content_end != -1:
                        invoke_content = tool_text[
                            invoke_start:invoke_content_end]
                        try:
                            parsed_tool = self._parse_single_invoke(
                                invoke_content,
                                request.tools if request else None)
                            if (parsed_tool
                                    and self.current_tool_index < len(
                                        self.prev_tool_call_arr)):
                                self.prev_tool_call_arr[
                                    self.current_tool_index][
                                        "arguments"] = json.loads(
                                            parsed_tool.function.arguments)
                        except Exception:
                            pass

                    result = DeltaMessage(tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="}"),
                        )
                    ])
                    if self.current_tool_index < len(
                            self.streamed_args_for_tool):
                        self.streamed_args_for_tool[
                            self.current_tool_index] += "}"

                    self.in_function = False
                    self.accumulated_params = {}
                    return result
                else:
                    return None

            param_starts = []
            idx = 0
            while True:
                idx = tool_text.find(self.parameter_prefix, idx)
                if idx == -1:
                    break
                param_starts.append(idx)
                idx += len(self.parameter_prefix)

            if (not self.in_param
                    and self.param_count < len(param_starts)):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" in remaining:
                    name_end = remaining.find(">")
                    param_name_raw = remaining[:name_end]
                    self.current_param_name = self._extract_name(
                        param_name_raw)

                    value_start = param_start + name_end + 1
                    value_text = tool_text[value_start:]
                    if value_text.startswith("\n"):
                        value_text = value_text[1:]

                    param_end_idx = value_text.find(self.parameter_end_token)
                    if param_end_idx == -1:
                        next_param_idx = value_text.find(
                            self.parameter_prefix)
                        func_end_idx = value_text.find(self.invoke_end_token)

                        if (next_param_idx != -1
                                and (func_end_idx == -1
                                     or next_param_idx < func_end_idx)):
                            param_end_idx = next_param_idx
                        elif func_end_idx != -1:
                            param_end_idx = func_end_idx
                        else:
                            if self.invoke_end_token in tool_text:
                                param_end_idx = len(value_text)
                            else:
                                return None

                    if param_end_idx != -1:
                        param_value = value_text[:param_end_idx]
                        if param_value.endswith("\n"):
                            param_value = param_value[:-1]

                        self.accumulated_params[
                            self.current_param_name] = param_value

                        param_config = {}
                        if request and request.tools:
                            for tool in request.tools:
                                if (hasattr(tool, "function")
                                        and tool.function.name ==
                                        self.current_function_name
                                        and hasattr(tool.function,
                                                    "parameters")):
                                    params = tool.function.parameters
                                    if (isinstance(params, dict)
                                            and "properties" in params):
                                        param_config = params["properties"]
                                    break

                        converted_value = self._convert_param_value(
                            param_value, self.current_param_name,
                            param_config, self.current_function_name or "")

                        serialized_value = json.dumps(
                            converted_value, ensure_ascii=False)

                        if self.param_count == 0:
                            json_fragment = (
                                f'"{self.current_param_name}": '
                                f'{serialized_value}')
                        else:
                            json_fragment = (
                                f', "{self.current_param_name}": '
                                f'{serialized_value}')

                        self.param_count += 1
                        if self.current_tool_index < len(
                                self.streamed_args_for_tool):
                            self.streamed_args_for_tool[
                                self.current_tool_index] += json_fragment
                        return DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                function=DeltaFunctionCall(
                                    arguments=json_fragment),
                            )
                        ])

        return None
