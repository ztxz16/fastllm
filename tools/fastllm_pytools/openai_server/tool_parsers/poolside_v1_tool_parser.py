# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM and SGLang projects

"""Parser for the Poolside/Laguna ``poolside_v1`` tool-call format.

The wire format deliberately leaves string values unquoted, so argument types
must be recovered from the tool schema::

    <tool_call>function_name
    <arg_key>name</arg_key><arg_value>value</arg_value>
    </tool_call>

Laguna may omit the newline after the function name.  The streaming parser is
therefore tag driven and also streams long string values instead of buffering
the complete tool call.
"""

import ast
import json
import logging
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
    ChatCompletionToolsParam,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)


logger = logging.getLogger(__name__)


@ToolParserManager.register_module("poolside_v1")
class PoolsideV1ToolParser(ToolParser):
    """Schema-aware Poolside parser with incremental string streaming."""

    tool_call_start_token = "<tool_call>"
    tool_call_end_token = "</tool_call>"
    arg_key_start = "<arg_key>"
    arg_key_end = "</arg_key>"
    arg_value_start = "<arg_value>"
    arg_value_end = "</arg_value>"
    tool_calls_start_token = tool_call_start_token

    _tool_call_re = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
    _tool_detail_re = re.compile(
        r"<tool_call>\s*([^\n<]+?)\s*\n?\s*(<arg_key>.*?)?</tool_call>",
        re.DOTALL,
    )
    # A key cannot contain '<'.  This prevents a malformed orphan key from
    # backtracking across the next <arg_key> boundary.
    _arg_pair_re = re.compile(
        r"<arg_key>([^<]*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
        re.DOTALL,
    )

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to PoolsideV1ToolParser")

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        self._buffer = ""
        self._in_tool_call = False
        self._current_tool_name: Optional[str] = None
        self._pending_key: Optional[str] = None
        self._streaming_string_value = False
        self._has_completed_tool_call = False
        self._tool_call_ids: list[str] = []
        self._args_started: list[bool] = []
        self._args_closed: list[bool] = []
        self._seen_keys: list[set[str]] = []

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    @staticmethod
    def _tools_enabled(request: ChatCompletionRequest) -> bool:
        return bool(request.tools) and request.tool_choice != "none"

    @staticmethod
    def _is_string_type(
        tool_name: str,
        arg_name: str,
        tools: Optional[list[ChatCompletionToolsParam]],
    ) -> bool:
        for tool in tools or []:
            if tool.function.name != tool_name or tool.function.parameters is None:
                continue
            arg_type = tool.function.parameters.get("properties", {}).get(
                arg_name, {}
            ).get("type")
            return str(arg_type).lower() in {"string", "str", "text", "enum"}
        return False

    @staticmethod
    def _deserialize(value: str) -> Any:
        for decoder in (json.loads, ast.literal_eval):
            try:
                result = decoder(value)
                # literal_eval can produce sets/bytes/complex values, which
                # cannot be returned as OpenAI JSON arguments.
                json.dumps(result)
                return result
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                pass
        return value

    @staticmethod
    def _json_escape_string_content(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)[1:-1] if value else ""

    @staticmethod
    def _partial_suffix_length(text: str, token: str) -> int:
        for length in range(min(len(text), len(token) - 1), 0, -1):
            if text.endswith(token[:length]):
                return length
        return 0

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        tool_calls: list[ToolCall] = []
        try:
            for block in self._tool_call_re.findall(model_output):
                detail = self._tool_detail_re.search(block)
                if detail is None:
                    continue
                name = detail.group(1).strip()
                if not name:
                    continue
                arguments: dict[str, Any] = {}
                for raw_key, raw_value in self._arg_pair_re.findall(
                    detail.group(2) or ""
                ):
                    key = raw_key.strip()
                    if self._is_string_type(name, key, request.tools):
                        value = raw_value
                    else:
                        value = self._deserialize(raw_value.strip())
                    arguments[key] = value
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )
        except Exception:
            logger.exception("Failed to parse poolside_v1 tool call")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        content = model_output[: model_output.find(self.tool_call_start_token)]
        if not content or not content.strip():
            content = None
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content
        )

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
        if not self._tools_enabled(request):
            return DeltaMessage(content=delta_text) if delta_text else None
        if previous_text == "":
            self._reset_stream_state()

        self._buffer += delta_text
        pending: dict[int, DeltaToolCall] = {}
        content: Optional[str] = None

        while True:
            if not self._in_tool_call:
                start = self._buffer.find(self.tool_call_start_token)
                if start < 0:
                    hold = self._partial_suffix_length(
                        self._buffer, self.tool_call_start_token
                    )
                    emit = self._buffer[:-hold] if hold else self._buffer
                    self._buffer = self._buffer[-hold:] if hold else ""
                    if emit and not (self._has_completed_tool_call and emit.isspace()):
                        content = (content or "") + emit
                    break
                if start > 0 and not (
                    self._has_completed_tool_call and self._buffer[:start].isspace()
                ):
                    content = (content or "") + self._buffer[:start]
                self._buffer = self._buffer[
                    start + len(self.tool_call_start_token) :
                ]
                self._begin_tool_call()
                continue

            if not self.current_tool_name_sent:
                # Whitespace before a name is formatting, not part of it.
                self._buffer = self._buffer.lstrip(" \t\r\n")
                newline = self._buffer.find("\n")
                key_start = self._buffer.find(self.arg_key_start)
                call_end = self._buffer.find(self.tool_call_end_token)
                boundaries = [x for x in (newline, key_start, call_end) if x >= 0]
                if not boundaries:
                    break
                boundary = min(boundaries)
                name = self._buffer[:boundary].strip()
                if boundary == newline:
                    self._buffer = self._buffer[boundary + 1 :]
                else:
                    self._buffer = self._buffer[boundary:]
                if not name:
                    if boundary == call_end:
                        self._buffer = self._buffer[len(self.tool_call_end_token) :]
                        self._revert_tool_call()
                        self._finish_tool_call()
                        continue
                    # Malformed call without a name: drain it without exposing
                    # a bogus OpenAI tool-call object.
                    end = self._buffer.find(self.tool_call_end_token)
                    if end < 0:
                        break
                    self._buffer = self._buffer[end + len(self.tool_call_end_token) :]
                    self._revert_tool_call()
                    self._finish_tool_call()
                    continue
                self._current_tool_name = name
                self.current_tool_name_sent = True
                self._emit_name(pending, name)
                continue

            if self._streaming_string_value:
                value_end = self._buffer.find(self.arg_value_end)
                if value_end >= 0:
                    raw = self._buffer[:value_end]
                    self._buffer = self._buffer[
                        value_end + len(self.arg_value_end) :
                    ]
                    fragment = self._json_escape_string_content(raw) + '"'
                    self._append_stream_fragment(pending, fragment)
                    self._streaming_string_value = False
                    self._pending_key = None
                    continue

                hold = self._partial_suffix_length(self._buffer, self.arg_value_end)
                safe_length = len(self._buffer) - hold
                if safe_length > 0:
                    raw = self._buffer[:safe_length]
                    self._buffer = self._buffer[safe_length:]
                    fragment = self._json_escape_string_content(raw)
                    if fragment:
                        self._append_stream_fragment(pending, fragment)
                break

            if self._pending_key is not None:
                # Recover when the model abandons one key and starts another.
                key_start = self._buffer.find(self.arg_key_start)
                value_start = self._buffer.find(self.arg_value_start)
                call_end = self._buffer.find(self.tool_call_end_token)
                candidates = [
                    (pos, kind)
                    for pos, kind in (
                        (key_start, "key"),
                        (value_start, "value"),
                        (call_end, "end"),
                    )
                    if pos >= 0
                ]
                if not candidates:
                    break
                pos, kind = min(candidates)
                self._buffer = self._buffer[pos:]
                if kind == "end":
                    self._buffer = self._buffer[len(self.tool_call_end_token) :]
                    fragment = self._close_arguments()
                    if fragment:
                        self._append_stream_fragment(pending, fragment)
                        self._finalize_arguments()
                    self._finish_tool_call()
                    continue
                if kind == "key":
                    key_end = self._buffer.find(self.arg_key_end)
                    if key_end < 0:
                        break
                    self._pending_key = self._buffer[
                        len(self.arg_key_start) : key_end
                    ].strip()
                    self._buffer = self._buffer[key_end + len(self.arg_key_end) :]
                    continue

                key = self._pending_key.strip()
                if self._is_string_type(self._current_tool_name or "", key,
                                        request.tools):
                    self._buffer = self._buffer[len(self.arg_value_start) :]
                    if key in self._seen_keys[self.current_tool_id]:
                        # Consume a duplicate value atomically and ignore it.
                        duplicate_end = self._buffer.find(self.arg_value_end)
                        if duplicate_end < 0:
                            break
                        self._buffer = self._buffer[
                            duplicate_end + len(self.arg_value_end) :
                        ]
                        self._pending_key = None
                        continue
                    self._seen_keys[self.current_tool_id].add(key)
                    separator = "{" if not self._args_started[self.current_tool_id] else ", "
                    fragment = separator + json.dumps(key, ensure_ascii=False) + ': "'
                    self._args_started[self.current_tool_id] = True
                    self._streaming_string_value = True
                    self._append_stream_fragment(pending, fragment)
                    continue

                value_end = self._buffer.find(self.arg_value_end)
                if value_end < 0:
                    break
                raw = self._buffer[len(self.arg_value_start) : value_end].strip()
                self._buffer = self._buffer[value_end + len(self.arg_value_end) :]
                self._pending_key = None
                fragment = self._make_non_string_fragment(key, raw)
                if fragment:
                    self._append_stream_fragment(pending, fragment)
                continue

            call_end = self._buffer.find(self.tool_call_end_token)
            key_start = self._buffer.find(self.arg_key_start)
            if call_end >= 0 and (key_start < 0 or call_end < key_start):
                self._buffer = self._buffer[call_end + len(self.tool_call_end_token) :]
                fragment = self._close_arguments()
                if fragment:
                    self._append_stream_fragment(pending, fragment)
                    self._finalize_arguments()
                self._finish_tool_call()
                continue
            if key_start < 0:
                break
            self._buffer = self._buffer[key_start:]
            key_end = self._buffer.find(self.arg_key_end)
            if key_end < 0:
                break
            self._pending_key = self._buffer[
                len(self.arg_key_start) : key_end
            ].strip()
            self._buffer = self._buffer[key_end + len(self.arg_key_end) :]

        tool_calls = list(pending.values())
        if content is None and not tool_calls:
            return None
        return DeltaMessage(content=content, tool_calls=tool_calls)

    def _ensure_tool_state(self) -> None:
        while len(self._tool_call_ids) <= self.current_tool_id:
            self._tool_call_ids.append(random_tool_call_id())
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append({})
            self._args_started.append(False)
            self._args_closed.append(False)
            self._seen_keys.append(set())

    def _begin_tool_call(self) -> None:
        self.current_tool_id += 1
        self._ensure_tool_state()
        self.current_tool_name_sent = False
        self._current_tool_name = None
        self._pending_key = None
        self._streaming_string_value = False
        self._in_tool_call = True

    def _finish_tool_call(self) -> None:
        self._has_completed_tool_call = True
        self._in_tool_call = False
        self.current_tool_name_sent = False
        self._current_tool_name = None
        self._pending_key = None
        self._streaming_string_value = False

    def _revert_tool_call(self) -> None:
        if self.current_tool_id < 0:
            return
        self._tool_call_ids.pop()
        self.streamed_args_for_tool.pop()
        self.prev_tool_call_arr.pop()
        self._args_started.pop()
        self._args_closed.pop()
        self._seen_keys.pop()
        self.current_tool_id -= 1

    def _get_delta(self, pending: dict[int, DeltaToolCall]) -> DeltaToolCall:
        if self.current_tool_id not in pending:
            pending[self.current_tool_id] = DeltaToolCall(
                index=self.current_tool_id,
                function=DeltaFunctionCall(),
            )
        return pending[self.current_tool_id]

    def _emit_name(self, pending: dict[int, DeltaToolCall], name: str) -> None:
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": name,
            "arguments": {},
        }
        delta = self._get_delta(pending)
        delta.id = self._tool_call_ids[self.current_tool_id]
        delta.type = "function"
        delta.function.name = name
        delta.function.arguments = delta.function.arguments or ""

    def _append_stream_fragment(
        self, pending: dict[int, DeltaToolCall], fragment: str
    ) -> None:
        self.streamed_args_for_tool[self.current_tool_id] += fragment
        delta = self._get_delta(pending)
        delta.function.arguments = (delta.function.arguments or "") + fragment

    def _make_non_string_fragment(self, key: str, raw: str) -> Optional[str]:
        if not key or key in self._seen_keys[self.current_tool_id]:
            return None
        value = self._deserialize(raw)
        separator = "{" if not self._args_started[self.current_tool_id] else ", "
        self._args_started[self.current_tool_id] = True
        self._seen_keys[self.current_tool_id].add(key)
        return (
            separator
            + json.dumps(key, ensure_ascii=False)
            + ": "
            + json.dumps(value, ensure_ascii=False)
        )

    def _close_arguments(self) -> Optional[str]:
        if self._args_closed[self.current_tool_id]:
            return None
        self._args_closed[self.current_tool_id] = True
        if self._args_started[self.current_tool_id]:
            return "}"
        return "{}"

    def _finalize_arguments(self) -> None:
        try:
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = json.loads(
                self.streamed_args_for_tool[self.current_tool_id]
            )
        except (json.JSONDecodeError, IndexError):
            logger.warning("Incomplete poolside_v1 arguments for tool %d",
                           self.current_tool_id)
