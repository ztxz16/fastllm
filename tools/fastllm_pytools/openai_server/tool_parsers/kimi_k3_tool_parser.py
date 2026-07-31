# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for Kimi K3's XTML response format.

Kimi K3 emits assistant responses in sibling ``response`` and ``tools``
channels.  The structural markers are tokenizer tokens, but reach the OpenAI
serving layer as their literal spellings, for example::

    <|open|>response<|sep|>I'll check.<|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="get_weather" index="1"<|sep|>
        <|open|>argument key="city" type="string"<|sep|>Beijing
        <|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>

This parser follows the Kimi K3 parsers used by vLLM: response wrappers are
removed, argument values are restored to JSON types, and streaming holds
partial XTML markers so ``<|close|>`` fragments never leak to clients.
"""

import json
import logging
from collections.abc import Sequence
from typing import Optional, Union

import regex as re

from .abstract_tool_parser import ToolParser, ToolParserManager
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

_OPEN = r"<\|open\|>"
_CLOSE = r"<\|close\|>"
_SEP = r"<\|sep\|>"
_TEXT_UNTIL_SEP = r"(?:(?!" + _SEP + r").)*?"


def _partial_marker_overlap(text: str, marker: str) -> int:
    max_len = min(len(text), len(marker) - 1)
    for length in range(max_len, 0, -1):
        if text.endswith(marker[:length]):
            return length
    return 0


@ToolParserManager.register_module(["kimi_k3"])
class KimiK3ToolParser(ToolParser):
    """Parse Kimi K3 XTML into OpenAI-compatible tool calls."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        self.tools_open = "<|open|>tools<|sep|>"
        self.tools_close = "<|close|>tools<|sep|>"
        self.call_open = "<|open|>call"
        self.response_open = "<|open|>response<|sep|>"
        self.response_close = "<|close|>response<|sep|>"
        self.tool_call_start_token = self.tools_open

        self._tools_open_re = re.compile(
            _OPEN + r"\s*tools\s*" + _SEP)
        self._tools_close_re = re.compile(
            _CLOSE + r"\s*tools\s*" + _SEP)
        self._call_open_re = re.compile(
            _OPEN + r"\s*call(?:\s|" + _SEP + r")")
        self._response_open_re = re.compile(
            _OPEN + r"\s*response\s*" + _SEP)
        self._response_close_re = re.compile(
            _CLOSE + r"\s*response\s*" + _SEP)
        self._message_close_re = re.compile(
            _CLOSE + r"\s*message\s*" + _SEP)
        self._end_of_msg_re = re.compile(r"<\|end_of_msg\|>")
        self._call_re = re.compile(
            _OPEN
            + r"\s*call\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r")"
            + _SEP
            + r"(?P<body>.*?)"
            + _CLOSE
            + r"\s*call\s*"
            + _SEP,
            re.DOTALL,
        )
        self._argument_re = re.compile(
            _OPEN
            + r"\s*argument\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r")"
            + _SEP
            + r"(?P<value>.*?)"
            + _CLOSE
            + r"\s*argument\s*"
            + _SEP,
            re.DOTALL,
        )
        self._json_re = re.compile(
            _OPEN
            + r"\s*json(?:\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r"))?"
            + _SEP
            + r"(?P<value>.*?)"
            + _CLOSE
            + r"\s*json\s*"
            + _SEP,
            re.DOTALL,
        )
        self._attribute_re = re.compile(
            r'(?P<key>\w+)="(?P<value>[^"]*)"')
        self._response_re = re.compile(
            _OPEN
            + r"\s*response\s*"
            + _SEP
            + r"(?P<content>.*?)"
            + _CLOSE
            + r"\s*response\s*"
            + _SEP,
            re.DOTALL,
        )

        self._sent_content_index = 0
        self._sent_tool_call_count = 0

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

    def has_tool_call(self, text: str) -> bool:
        return (
            self._tools_open_re.search(text) is not None
            or self._call_open_re.search(text) is not None
        )

    def get_token_ids(self, text: str) -> list[int]:
        # K3 parses accumulated XTML text and never inspects token ids.  Avoid
        # re-tokenizing every streamed fragment: the custom Kimi tokenizer
        # deliberately warns whenever Hugging Face's add_special_tokens
        # argument sends encode() through its generic compatibility path.
        return [1]

    def _attributes(self, text: str) -> dict[str, str]:
        return {
            match["key"]: match["value"].replace(
                "&quot;", '"').replace("&amp;", "&")
            for match in self._attribute_re.finditer(text)
        }

    def _decode_call(self, attrs: str, body: str) -> Optional[ToolCall]:
        call_attrs = self._attributes(attrs)
        tool_name = call_attrs.get("tool", "")
        tool_index = call_attrs.get("index", "")
        if not tool_name:
            return None

        arguments = {}
        for match in self._argument_re.finditer(body):
            argument_attrs = self._attributes(match["attrs"])
            key = argument_attrs.get("key", "")
            argument_type = argument_attrs.get("type", "string")
            raw_value = match["value"]
            if argument_type == "string":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    arguments[key] = raw_value

        # The official tokenizer uses a json block when an earlier assistant
        # message carries an already-serialized argument object.  Accept the
        # same shape for generated output as a defensive compatibility path.
        if not arguments:
            json_match = self._json_re.search(body)
            if json_match is not None:
                try:
                    parsed = json.loads(json_match["value"])
                    if isinstance(parsed, dict):
                        arguments = parsed
                except json.JSONDecodeError:
                    pass

        tool_call_id = tool_name
        if tool_index:
            try:
                # K3 XTML indices are one-based; API call aliases are
                # zero-based and can be rendered back into tool-result turns.
                tool_call_id = f"{tool_name}:{int(tool_index) - 1}"
            except ValueError:
                tool_call_id = f"{tool_name}:{tool_index}"

        return ToolCall(
            id=tool_call_id,
            type="function",
            function=FunctionCall(
                name=tool_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _strip_response_content(self, text: str) -> Optional[str]:
        response_open = self._response_open_re.search(text)
        if response_open is not None:
            response_close = self._response_close_re.search(
                text, response_open.end())
            if response_close is not None:
                text = text[response_open.end():response_close.start()]
            else:
                text = text[response_open.end():]
        else:
            # The generation prefix may already contain response_open.
            text = self._response_close_re.sub("", text)
        text = self._message_close_re.sub("", text)
        text = self._end_of_msg_re.sub("", text)
        return text or None

    def _content(self, model_output: str, before_tools: str) -> Optional[str]:
        response = self._response_re.search(model_output)
        if response is not None:
            return response["content"] or None
        return self._strip_response_content(before_tools)

    def _extract_stream_content(self, current_text: str) -> Optional[str]:
        response_open = self._response_open_re.search(current_text)
        body_start = response_open.end() if response_open is not None else 0
        tools_open = self._tools_open_re.search(current_text, body_start)
        call_open = self._call_open_re.search(current_text, body_start)
        response_close = self._response_close_re.search(
            current_text, body_start)

        terminal_positions = [
            match.start()
            for match in (tools_open, call_open, response_close)
            if match is not None
        ]
        if terminal_positions:
            sendable_index = min(terminal_positions)
        else:
            overlap = max(
                _partial_marker_overlap(current_text, self.response_open),
                _partial_marker_overlap(current_text, self.response_close),
                _partial_marker_overlap(current_text, self.tools_open),
                _partial_marker_overlap(current_text, self.call_open),
            )
            sendable_index = len(current_text) - overlap

        if sendable_index <= body_start:
            return None
        if self._sent_content_index < body_start:
            self._sent_content_index = body_start
        if sendable_index <= self._sent_content_index:
            return None

        content = current_text[
            self._sent_content_index:sendable_index]
        self._sent_content_index = sendable_index
        return content or None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        tools_open = self._tools_open_re.search(model_output)
        call_open = self._call_open_re.search(model_output)
        if tools_open is None and call_open is None:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self._content(model_output, model_output),
            )

        try:
            wrapper = tools_open if tools_open is not None else call_open
            before_tools = model_output[:wrapper.start()]
            section_start = (
                tools_open.end() if tools_open is not None
                else call_open.start()
            )
            tools_close = self._tools_close_re.search(
                model_output, section_start)
            section = (
                model_output[section_start:]
                if tools_close is None
                else model_output[section_start:tools_close.start()]
            )
            tool_calls = [
                tool_call
                for match in self._call_re.finditer(section)
                if (tool_call := self._decode_call(
                    match["attrs"], match["body"])) is not None
            ]
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=self._content(model_output, before_tools),
            )
        except Exception:
            logger.exception("Error extracting Kimi K3 tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
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
        # Parsing from accumulated text makes every possible chunk boundary
        # safe, including chunks that end halfway through a control marker.
        content = self._extract_stream_content(current_text)
        tools_open = self._tools_open_re.search(current_text)
        call_open = self._call_open_re.search(current_text)
        if tools_open is None and call_open is None:
            return DeltaMessage(content=content) if content else None

        section_start = (
            tools_open.end() if tools_open is not None
            else call_open.start()
        )
        section = current_text[section_start:]
        calls = [
            tool_call
            for match in self._call_re.finditer(section)
            if (tool_call := self._decode_call(
                match["attrs"], match["body"])) is not None
        ]
        if len(calls) <= self._sent_tool_call_count:
            return DeltaMessage(content=content) if content else None

        new_calls = calls[self._sent_tool_call_count:]
        deltas = [
            DeltaToolCall(
                index=self._sent_tool_call_count + offset,
                id=tool_call.id,
                type="function",
                function=DeltaFunctionCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for offset, tool_call in enumerate(new_calls)
        ]
        self._sent_tool_call_count = len(calls)
        return DeltaMessage(content=content, tool_calls=deltas)
