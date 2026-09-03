# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import uuid
from collections.abc import Sequence
from typing import Any, Optional, Union

from .abstract_tool_parser import ToolParser, ToolParserManager
from ..protocal.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from ..tool_schema import convert_text_value, get_value


logger = logging.getLogger(__name__)


class _IncompleteToolCall(ValueError):
    """Raised when a valid tool call may be completed by later text."""


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Return the length of a tag prefix held at the end of ``text``."""
    max_length = min(len(text), len(tag) - 1)
    for length in range(max_length, 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


def _skip_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _is_partial_token(text: str, cursor: int, token: str) -> bool:
    remainder = text[cursor:]
    return len(remainder) < len(token) and token.startswith(remainder)


@ToolParserManager.register_module(["qwen3_coder"])
class Qwen3CoderToolParser(ToolParser):
    """Parser for Qwen's XML function-call wire format.

    Streaming is deliberately block-buffered. A speculative decoder may put
    any number of token boundaries into one delta, so parser correctness must
    depend on the generated text rather than on how that text was chunked.
    Complete blocks are drained in a loop and incomplete tails remain buffered
    for the next delta.
    """

    tool_call_start_token = "<tool_call>"
    tool_call_end_token = "</tool_call>"
    tool_call_prefix = "<function="
    function_end_token = "</function>"
    parameter_prefix = "<parameter="
    parameter_end_token = "</parameter>"

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

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
                "Qwen3 XML Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!")

        self._reset_streaming_state()

    def get_token_ids(self, text: str) -> list[int]:
        return [self.vocab.get(text)]

    @staticmethod
    def _generate_tool_call_id() -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self) -> None:
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self._stream_buffer = ""
        self._stream_error: Optional[str] = None
        self._stream_has_content_since_tool = False

    @staticmethod
    def _tool_parameters(
        function_name: str,
        tools: Optional[list[ChatCompletionToolsParam]],
    ) -> dict[str, Any]:
        for tool in tools or []:
            function = get_value(tool, "function")
            if get_value(function, "name") != function_name:
                continue
            parameters = get_value(function, "parameters")
            return parameters if isinstance(parameters, dict) else {}
        return {}

    @staticmethod
    def _parameter_schema(
        parameter_name: str,
        parameters: dict[str, Any],
    ) -> Any:
        properties = parameters.get("properties")
        if isinstance(properties, dict) and parameter_name in properties:
            return properties[parameter_name]
        additional = parameters.get("additionalProperties")
        if isinstance(additional, dict):
            return additional
        return None

    @staticmethod
    def _strip_protocol_newlines(value: str) -> str:
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        return value

    @staticmethod
    def _parse_named_opening(
        text: str,
        cursor: int,
        prefix: str,
        description: str,
    ) -> tuple[str, int]:
        if not text.startswith(prefix, cursor):
            if _is_partial_token(text, cursor, prefix):
                raise _IncompleteToolCall(
                    f"Qwen output ended inside a {description} opening tag")
            raise ValueError(f"Expected a Qwen {description} opening tag")

        name_start = cursor + len(prefix)
        name_end = text.find(">", name_start)
        if name_end == -1:
            raise _IncompleteToolCall(
                f"Qwen output ended inside a {description} opening tag")
        if "\n" in text[name_start:name_end] or "\r" in text[
                name_start:name_end]:
            raise ValueError(
                f"Qwen {description} opening tag contains a newline")

        name = text[name_start:name_end].strip()
        if not name:
            raise ValueError(f"Qwen {description} has an empty name")
        return name, name_end + 1

    def _find_parameter_end(self, text: str, value_start: int) -> int:
        """Find a structural parameter terminator, ignoring tag-like data.

        Qwen does not escape XML-looking text inside parameter values. A
        ``</parameter>`` is therefore structural only when the next
        non-whitespace text can continue the wire grammar: another parameter
        or the end of the function. This keeps ordinary code and documentation
        that mention protocol tags intact. A value containing a complete,
        structurally valid closing sequence remains inherently ambiguous in
        the model's unescaped wire format.
        """
        search_from = value_start
        continuations = (self.parameter_prefix, self.function_end_token)
        while True:
            end = text.find(self.parameter_end_token, search_from)
            if end == -1:
                raise _IncompleteToolCall(
                    "Qwen output ended inside a parameter value")

            continuation = _skip_whitespace(
                text, end + len(self.parameter_end_token))
            if any(text.startswith(token, continuation)
                   for token in continuations):
                return end
            if any(_is_partial_token(text, continuation, token)
                   for token in continuations):
                raise _IncompleteToolCall(
                    "Qwen output ended after a parameter closing tag")

            # The delimiter is part of the argument text. Search for the next
            # one instead of letting data alter the protocol nesting.
            search_from = end + len(self.parameter_end_token)

    def _parse_parameter_at(
        self,
        text: str,
        cursor: int,
    ) -> tuple[str, str, int]:
        name, value_start = self._parse_named_opening(
            text,
            cursor,
            self.parameter_prefix,
            "parameter",
        )
        value_end = self._find_parameter_end(text, value_start)
        raw_value = self._strip_protocol_newlines(
            text[value_start:value_end])
        return (
            name,
            raw_value,
            value_end + len(self.parameter_end_token),
        )

    def _build_tool_call(
        self,
        function_name: str,
        raw_arguments: list[tuple[str, str]],
        tools: Optional[list[ChatCompletionToolsParam]],
    ) -> ToolCall:
        function_name = function_name.strip()
        if not function_name:
            raise ValueError("Qwen tool call has an empty function name")

        parameters = self._tool_parameters(function_name, tools)
        properties = parameters.get("properties")
        arguments: dict[str, Any] = {}
        for parameter_name, raw_value in raw_arguments:
            if parameter_name in arguments:
                raise ValueError(
                    f"Tool {function_name!r} repeats parameter "
                    f"{parameter_name!r}")

            parameter_schema = self._parameter_schema(parameter_name,
                                                       parameters)
            if parameter_schema is None:
                if isinstance(properties, dict) and properties:
                    logger.warning(
                        "Parsed parameter '%s' is not defined for tool '%s'; "
                        "preserving its string value.", parameter_name,
                        function_name)
                converted_value = raw_value
            else:
                converted_value = convert_text_value(
                    raw_value,
                    parameter_schema,
                    parameters,
                )
            arguments[parameter_name] = converted_value

        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
        )
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name,
                arguments=serialized,
            ),
        )

    def _parse_function_at(
        self,
        text: str,
        cursor: int,
        tools: Optional[list[ChatCompletionToolsParam]],
    ) -> tuple[ToolCall, int]:
        function_name, cursor = self._parse_named_opening(
            text,
            cursor,
            self.tool_call_prefix,
            "function",
        )
        raw_arguments: list[tuple[str, str]] = []

        while True:
            cursor = _skip_whitespace(text, cursor)
            if text.startswith(self.function_end_token, cursor):
                cursor += len(self.function_end_token)
                return self._build_tool_call(
                    function_name,
                    raw_arguments,
                    tools,
                ), cursor
            if text.startswith(self.parameter_prefix, cursor):
                name, raw_value, cursor = self._parse_parameter_at(
                    text, cursor)
                raw_arguments.append((name, raw_value))
                continue
            if (_is_partial_token(text, cursor, self.parameter_prefix)
                    or _is_partial_token(text, cursor,
                                         self.function_end_token)):
                raise _IncompleteToolCall(
                    f"Qwen output ended inside tool {function_name!r}")
            raise ValueError(
                f"Tool {function_name!r} contains malformed parameter markup")

    def _parse_tool_block_prefix(
        self,
        text: str,
        cursor: int,
        tools: Optional[list[ChatCompletionToolsParam]],
    ) -> tuple[list[ToolCall], int]:
        if not text.startswith(self.tool_call_start_token, cursor):
            if _is_partial_token(text, cursor, self.tool_call_start_token):
                raise _IncompleteToolCall(
                    "Qwen output ended inside a tool-call opening tag")
            raise ValueError("Expected a Qwen tool-call opening tag")

        cursor += len(self.tool_call_start_token)
        tool_calls: list[ToolCall] = []
        while True:
            cursor = _skip_whitespace(text, cursor)
            if text.startswith(self.tool_call_end_token, cursor):
                if not tool_calls:
                    raise ValueError(
                        "Qwen tool-call block contains no complete function")
                return tool_calls, cursor + len(self.tool_call_end_token)
            if text.startswith(self.tool_call_prefix, cursor):
                tool_call, cursor = self._parse_function_at(
                    text, cursor, tools)
                tool_calls.append(tool_call)
                continue
            if (_is_partial_token(text, cursor, self.tool_call_prefix)
                    or _is_partial_token(text, cursor,
                                         self.tool_call_end_token)):
                raise _IncompleteToolCall(
                    "Qwen output ended inside a tool-call block")
            raise ValueError(
                "Qwen tool-call block contains malformed function markup")

    def _split_complete_blocks(
        self,
        text: str,
        tools: Optional[list[ChatCompletionToolsParam]],
    ) -> tuple[str, list[ToolCall]]:
        normal_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        cursor = 0

        while True:
            start = text.find(self.tool_call_start_token, cursor)
            stray_end = text.find(self.tool_call_end_token, cursor)
            if start == -1:
                if stray_end != -1:
                    raise ValueError("Qwen output has an unmatched tool-call end")
                normal_segment = text[cursor:]
                if normal_segment.strip():
                    normal_parts.append(normal_segment)
                break
            if stray_end != -1 and stray_end < start:
                raise ValueError("Qwen output has an unmatched tool-call end")

            normal_segment = text[cursor:start]
            if normal_segment.strip():
                normal_parts.append(normal_segment)
            parsed_calls, cursor = self._parse_tool_block_prefix(
                text, start, tools)
            tool_calls.extend(parsed_calls)

        return "".join(normal_parts), tool_calls

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

        try:
            normal_text, tool_calls = self._split_complete_blocks(
                model_output, request.tools)
        except (TypeError, ValueError) as error:
            logger.warning("Failed to parse Qwen tool call: %s", error)
            prefix = model_output[:marker_index]
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=prefix or None,
            )

        self.prev_tool_call_arr = [{
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        } for tool_call in tool_calls]
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=normal_text if normal_text.strip() else None,
        )

    def _append_stream_call(
        self,
        tool_call: ToolCall,
        deltas: list[DeltaToolCall],
    ) -> None:
        self.current_tool_id += 1
        call_id = self._generate_tool_call_id()
        name = tool_call.function.name
        arguments = tool_call.function.arguments

        self.prev_tool_call_arr.append({
            "name": name,
            "arguments": arguments,
        })
        self.streamed_args_for_tool.append(arguments)
        deltas.append(
            DeltaToolCall(
                index=self.current_tool_id,
                id=call_id,
                type="function",
                function=DeltaFunctionCall(
                    name=name,
                    arguments=arguments,
                ),
            ))

    def streaming_parse_error(self) -> Optional[str]:
        if self._stream_error:
            return self._stream_error
        if self.tool_call_start_token in self._stream_buffer:
            return "Qwen tool-call stream ended inside a tool-call block"
        return None

    def flush_streaming_content(self) -> Optional[str]:
        """Return ordinary text held while disambiguating a partial marker."""
        if (not self._stream_buffer or self._stream_error
                or self.tool_call_start_token in self._stream_buffer):
            return None
        pending = self._stream_buffer
        self._stream_buffer = ""
        if (self.prev_tool_call_arr and not pending.strip()
                and not self._stream_has_content_since_tool):
            return None
        return pending

    def _append_normal_segment(
        self,
        segment: str,
        normal_parts: list[str],
    ) -> None:
        if not segment:
            return
        if segment.strip():
            self._stream_has_content_since_tool = True
            normal_parts.append(segment)
        elif self._stream_has_content_since_tool:
            normal_parts.append(segment)

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
        del current_text, previous_token_ids, current_token_ids
        if not previous_text:
            self._reset_streaming_state()

        if self._stream_error:
            return None

        self._stream_buffer += delta_text
        normal_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []
        at_stream_end = bool(not delta_text and delta_token_ids)

        while self._stream_buffer:
            marker_index = self._stream_buffer.find(
                self.tool_call_start_token)
            if marker_index == -1:
                overlap = (0 if at_stream_end else _partial_tag_overlap(
                    self._stream_buffer, self.tool_call_start_token))
                safe_end = len(self._stream_buffer) - overlap
                safe_text = self._stream_buffer[:safe_end]
                if safe_text.strip():
                    content_end = len(safe_text.rstrip())
                    self._append_normal_segment(
                        safe_text[:content_end], normal_parts)
                    self._stream_buffer = (
                        safe_text[content_end:] +
                        self._stream_buffer[safe_end:])
                if at_stream_end:
                    pending = self.flush_streaming_content()
                    if pending:
                        normal_parts.append(pending)
                break

            if marker_index > 0:
                prefix = self._stream_buffer[:marker_index]
                self._append_normal_segment(prefix, normal_parts)
                self._stream_buffer = self._stream_buffer[marker_index:]

            try:
                parsed_calls, consumed = self._parse_tool_block_prefix(
                    self._stream_buffer,
                    0,
                    request.tools,
                )
            except _IncompleteToolCall:
                break
            except (TypeError, ValueError) as error:
                logger.warning("Failed to parse streamed Qwen tool call: %s",
                               error)
                self._stream_error = str(error)
                self._stream_buffer = ""
                break
            else:
                self._stream_buffer = self._stream_buffer[consumed:]
                for tool_call in parsed_calls:
                    self._append_stream_call(tool_call, tool_deltas)
                self._stream_has_content_since_tool = False

        content_delta = "".join(normal_parts)
        content = content_delta or None
        if not content and not tool_deltas:
            return None
        return DeltaMessage(content=content, tool_calls=tool_deltas)
