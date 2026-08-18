# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for FastLLM from vLLM's Dots tool parser.

import json
import logging
import math
from collections.abc import Sequence
from typing import Any, Optional, Union

import regex as re

from .abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
    random_tool_call_id,
)
from ..protocal.openai_protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)


logger = logging.getLogger(__name__)


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _load_strict_json(value: str) -> Any:
    return json.loads(
        value,
        parse_constant=_reject_non_finite_json_constant,
    )


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Return the length of a tag prefix held at the end of ``text``."""
    max_length = min(len(text), len(tag) - 1)
    for length in range(max_length, 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


@ToolParserManager.register_module("dots")
class DotsToolParser(ToolParser):
    """Parse the XML tool-call protocol used by Dots3-Note.

    The canonical wire format is::

        <dots_function_call>
        <invoke name="search">
        <parameter name="query">weather in Shanghai</parameter>
        </invoke>
        </dots_function_call>

    Multiple wrappers and multiple ``invoke`` elements per wrapper are
    supported.  The model's JSON-object fallback is accepted as well.
    """

    tool_call_start_token = "<dots_function_call>"
    tool_call_end_token = "</dots_function_call>"

    _block_regex = re.compile(
        rf"{re.escape(tool_call_start_token)}\s*(.*?)\s*"
        rf"{re.escape(tool_call_end_token)}",
        re.DOTALL,
    )
    _invoke_regex = re.compile(
        r"<invoke\s+name\s*=\s*(?P<name>[^>]+)>"
        r"(?P<body>.*?)</invoke>",
        re.DOTALL,
    )
    _parameter_regex = re.compile(
        r"<parameter\s+name\s*=\s*(?P<name>[^>]+)>"
        r"(?P<value>.*?)</parameter>",
        re.DOTALL,
    )

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._buffer = ""

    @staticmethod
    def _extract_name(value: str) -> str:
        value = value.strip()
        if (len(value) >= 2 and value[0] == value[-1]
                and value[0] in {'"', "'"}):
            return value[1:-1]
        return value

    @staticmethod
    def _convert_param_value(value: str, param_type: Any) -> Any:
        if isinstance(param_type, list):
            param_type = next(
                (item for item in param_type if item != "null"), "string")
        if not isinstance(param_type, str):
            param_type = str(param_type)
        param_type = param_type.lower()

        if param_type in {"string", "str", "text"}:
            return value
        if value.lower() == "null":
            return None
        if param_type in {"integer", "int"}:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        if param_type in {"number", "float"}:
            try:
                number = float(value)
                if not math.isfinite(number):
                    return value
                return int(number) if number.is_integer() else number
            except (TypeError, ValueError):
                return value
        if param_type in {"boolean", "bool"}:
            lowered = value.lower()
            if lowered in {"true", "1"}:
                return True
            if lowered in {"false", "0"}:
                return False
            return value
        try:
            return _load_strict_json(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return value

    def _resolve_param_type(
        self,
        schema: Any,
        definitions: dict[str, Any],
        depth: int = 0,
    ) -> Optional[Any]:
        if not isinstance(schema, dict) or depth > 10:
            return None
        if "type" in schema:
            return schema["type"]

        reference = schema.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            return self._resolve_param_type(
                definitions.get(reference.rsplit("/", 1)[-1]),
                definitions,
                depth + 1,
            )

        for keyword in ("anyOf", "oneOf", "allOf"):
            alternatives = schema.get(keyword)
            if not isinstance(alternatives, list):
                continue
            for alternative in alternatives:
                if (isinstance(alternative, dict)
                        and alternative.get("type") == "null"):
                    continue
                resolved = self._resolve_param_type(
                    alternative, definitions, depth + 1)
                if resolved is not None:
                    return resolved
        return None

    @staticmethod
    def _tool_schema(
        name: str,
        tools: Optional[list[Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        for tool in tools or []:
            function = _get_value(tool, "function")
            if _get_value(function, "name") != name:
                continue
            schema = _get_value(function, "parameters")
            if not isinstance(schema, dict):
                break
            properties = schema.get("properties", {})
            definitions = schema.get("$defs", {})
            return (
                properties if isinstance(properties, dict) else {},
                definitions if isinstance(definitions, dict) else {},
            )
        return {}, {}

    def _parse_xml_invoke(
        self,
        match: Any,
        tools: Optional[list[Any]],
    ) -> dict[str, Any]:
        name = self._extract_name(match.group("name"))
        properties, definitions = self._tool_schema(name, tools)
        arguments: dict[str, Any] = {}
        for parameter in self._parameter_regex.finditer(match.group("body")):
            parameter_name = self._extract_name(parameter.group("name"))
            value = parameter.group("value").strip()
            parameter_type: Any = "string"
            if parameter_name in properties:
                parameter_type = (
                    self._resolve_param_type(
                        properties[parameter_name], definitions)
                    or "string"
                )
            arguments[parameter_name] = self._convert_param_value(
                value, parameter_type)
        return {"name": name, "arguments": arguments}

    def _parse_block(
        self,
        content: str,
        tools: Optional[list[Any]],
    ) -> list[dict[str, Any]]:
        content = content.strip()
        if content.startswith("<invoke"):
            calls = [
                self._parse_xml_invoke(match, tools)
                for match in self._invoke_regex.finditer(content)
            ]
            if not calls:
                raise ValueError("Dots tool-call block contains no invoke")
            return calls

        parsed = _load_strict_json(content)
        if not isinstance(parsed, dict):
            raise TypeError("Dots JSON tool call must be an object")
        return [parsed]

    @staticmethod
    def _normalize_call(parsed: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = parsed.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Dots tool call is missing a function name")
        arguments = parsed.get("arguments", parsed.get("parameters", {})) or {}
        if not isinstance(arguments, dict):
            raise TypeError("Dots tool-call arguments must be an object")
        return name, arguments

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        marker_index = model_output.find(self.tool_call_start_token)
        if marker_index == -1:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        blocks = list(self._block_regex.finditer(model_output))
        tool_calls: list[ToolCall] = []
        for block in blocks:
            try:
                parsed_calls = self._parse_block(block.group(1), request.tools)
                for parsed in parsed_calls:
                    name, arguments = self._normalize_call(parsed)
                    tool_calls.append(
                        ToolCall(function=FunctionCall(
                            name=name,
                            arguments=json.dumps(
                                arguments,
                                ensure_ascii=False,
                                allow_nan=False,
                            ),
                        )))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                logger.warning("Failed to parse Dots tool call: %s", error)

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        self.prev_tool_call_arr = [
            {
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
            for call in tool_calls
        ]
        normal_parts: list[str] = []
        previous_end = 0
        for block in blocks:
            normal_parts.append(model_output[previous_end:block.start()])
            previous_end = block.end()
        normal_parts.append(model_output[previous_end:])
        normal_text = "".join(normal_parts)
        unmatched_marker_index = normal_text.find(self.tool_call_start_token)
        if unmatched_marker_index != -1:
            normal_text = normal_text[:unmatched_marker_index]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=normal_text if normal_text.strip() else None,
        )

    def _append_stream_call(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_calls: list[DeltaToolCall],
    ) -> None:
        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
        )
        self.current_tool_id += 1
        self.prev_tool_call_arr.append({
            "name": name,
            "arguments": serialized,
        })
        self.streamed_args_for_tool.append(serialized)
        tool_calls.append(DeltaToolCall(
            index=self.current_tool_id,
            id=random_tool_call_id(),
            type="function",
            function=DeltaFunctionCall(
                name=name,
                arguments=serialized,
            ),
        ))

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
        del current_text, previous_token_ids, current_token_ids, delta_token_ids
        if not previous_text:
            self._buffer = ""
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []

        self._buffer += delta_text
        normal_parts: list[str] = []
        tool_calls: list[DeltaToolCall] = []

        while self._buffer:
            marker_index = self._buffer.find(self.tool_call_start_token)
            if marker_index == -1:
                partial_length = _partial_tag_overlap(
                    self._buffer, self.tool_call_start_token)
                if partial_length:
                    normal_parts.append(self._buffer[:-partial_length])
                    self._buffer = self._buffer[-partial_length:]
                else:
                    normal_parts.append(self._buffer)
                    self._buffer = ""
                break

            if marker_index > 0:
                normal_parts.append(self._buffer[:marker_index])
                self._buffer = self._buffer[marker_index:]

            end_index = self._buffer.find(
                self.tool_call_end_token, len(self.tool_call_start_token))
            if end_index == -1:
                break

            content = self._buffer[
                len(self.tool_call_start_token):end_index]
            self._buffer = self._buffer[
                end_index + len(self.tool_call_end_token):]
            try:
                parsed_calls = self._parse_block(content, request.tools)
                for parsed in parsed_calls:
                    name, arguments = self._normalize_call(parsed)
                    self._append_stream_call(name, arguments, tool_calls)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                # Keep malformed markup out of streamed assistant content.  The
                # FunctionCallParser finalizer reports the invalid block.
                logger.warning(
                    "Failed to parse streamed Dots tool call: %s", error)

        content_delta = "".join(normal_parts)
        if not content_delta and not tool_calls:
            return None
        return DeltaMessage(
            content=content_delta or None,
            tool_calls=tool_calls,
        )
