# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DeepSeek-V4.1 DSML tool parser.

V4.1 keeps V4's DSML markup token (``｜DSML｜``) but renames every tag with a
leading space: ``<｜DSML｜ calls>`` / ``<｜DSML｜ invoke>`` /
``<｜DSML｜ parameter>``.  The reference implementation of both the encoder and
the decoder ships with the checkpoint as ``encoding.py``; FastLLM vendors it as
``ftllm.encoding_dsv41``.  This parser therefore never re-implements the DSML
grammar: the complete-output path delegates to the official
``parse_message_from_completion_text`` and every argument dictionary — in both
the streaming and the non-streaming path — is built with the official
``decode_dsml_to_arguments``.  A tolerant regex scan is kept only as a fallback
for output the strict official parser rejects (truncated generations, missing
EOS, stray text after the tool-call block).
"""

import uuid
from collections.abc import Sequence
from typing import Any, Optional, Union

import regex as re

from .abstract_tool_parser import ToolParser, ToolParserManager
from ..protocal.openai_protocol import *

from ftllm.encoding_dsv41 import (
    decode_dsml_to_arguments,
    dsml_token,
    eos_token,
    thinking_end_token,
    thinking_start_token,
    tool_call_tag_name,
    tool_calls_block_name,
    tool_parameter_tag_name,
    parse_message_from_completion_text,
)

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


# The ASCII fallback spelling mirrors the one DeepSeek-V4 accepts: some
# deployments strip the full-width markup token from the vocabulary.
ALT_DSML_TOKEN = "\\DSML\\"


@ToolParserManager.register_module(["deepseek_v41", "deepseek_v41_text"])
class DeepSeekV41ToolParser(ToolParser):
    """
    DeepSeek V4.1 DSML tool parser.

    Example:
    <｜DSML｜ calls>
    <｜DSML｜ invoke name="get_weather">
    <｜DSML｜ parameter name="location" string="true">北京</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>
    """

    tool_call_start_token: str = f"<{dsml_token}{tool_calls_block_name}>"
    tool_call_end_token: str = f"</{dsml_token}{tool_calls_block_name}>"
    alt_tool_call_start_token: str = f"<{ALT_DSML_TOKEN}{tool_calls_block_name}>"
    alt_tool_call_end_token: str = f"</{ALT_DSML_TOKEN}{tool_calls_block_name}>"

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
        dsml_tag_regex = r"(?:" + re.escape(dsml_token) + r"|\\DSML\\)"
        block = re.escape(tool_calls_block_name)
        invoke = re.escape(tool_call_tag_name)
        parameter = re.escape(tool_parameter_tag_name)

        self.tool_call_complete_regex = re.compile(
            r"<" + dsml_tag_regex + block
            + r">(.*?)</" + dsml_tag_regex + block + r">",
            re.DOTALL,
        )
        self.invoke_complete_regex = re.compile(
            r"<" + dsml_tag_regex + invoke
            + r'\s+name="([^"]+)"\s*>(.*?)</' + dsml_tag_regex + invoke + r">",
            re.DOTALL,
        )
        self.parameter_complete_regex = re.compile(
            r"<" + dsml_tag_regex + parameter
            + r'\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</'
            + dsml_tag_regex + parameter + r">",
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

    def _parse_invoke_params(self, invoke_str: str) -> dict[str, tuple[str, str]]:
        """Collect DSML parameters as the official ``(value, string)`` pairs."""
        params: dict[str, tuple[str, str]] = {}
        for name, is_string, value in self.parameter_complete_regex.findall(
                invoke_str):
            params[name] = (value, is_string)
        return params

    def _invoke_arguments(self, tool_name: str, invoke_str: str) -> str:
        # decode_dsml_to_arguments is the official DSML -> JSON conversion:
        # string="true" values are JSON-quoted, everything else is passed
        # through verbatim so numbers/objects keep their exact spelling.
        return decode_dsml_to_arguments(
            tool_name=tool_name,
            tool_args=self._parse_invoke_params(invoke_str),
        )["arguments"]

    def _find_tool_call_start(self, text: str) -> int:
        positions = [
            text.find(token) for token in self.tool_call_start_tokens
            if text.find(token) >= 0
        ]
        return min(positions) if positions else -1

    def _extract_with_official_parser(
        self,
        model_output: str,
    ) -> Optional[ExtractedToolCallInformation]:
        """Strict path: hand the whole completion to the official decoder.

        The server already split the reasoning block off in most
        configurations, so the thinking mode is inferred from the text itself.
        Returns None when the official parser rejects the text, in which case
        the caller falls back to the tolerant scan.
        """
        text = model_output
        # emit_reasoning_content=False leaves the whole <think>…</think> block
        # inline; the official parser rejects a literal <think> inside the
        # reasoning text, so peel it off and restore it afterwards.
        think_prefix = ""
        if text.startswith(thinking_start_token):
            think_prefix = thinking_start_token
            text = text[len(thinking_start_token):]
        thinking_mode = "thinking" if thinking_end_token in text else "chat"
        if not text.endswith(eos_token):
            text = text + eos_token
        try:
            message = parse_message_from_completion_text(text, thinking_mode)
        except Exception:
            return None

        raw_tool_calls = message.get("tool_calls") or []
        if not raw_tool_calls:
            return None

        tool_calls = [
            ToolCall(
                type="function",
                function=FunctionCall(
                    name=call["function"]["name"],
                    arguments=call["function"]["arguments"],
                ),
            )
            for call in raw_tool_calls
        ]
        content = message.get("content") or None
        if thinking_mode == "thinking" and message.get("reasoning_content"):
            # emit_reasoning_content=False keeps <think>…</think> inline; the
            # V4 behaviour is to leave that text in `content`.
            reasoning = message["reasoning_content"]
            content = (think_prefix + reasoning + thinking_end_token
                       + (content or "")) or None
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=tool_calls, content=content)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        first_tool_idx = self._find_tool_call_start(model_output)
        if first_tool_idx < 0:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

        official = self._extract_with_official_parser(model_output)
        if official is not None:
            return official

        try:
            tool_calls = []
            for tool_call_match in self.tool_call_complete_regex.findall(
                    model_output):
                for invoke_name, invoke_content in (
                        self.invoke_complete_regex.findall(tool_call_match)):
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=invoke_name,
                                arguments=self._invoke_arguments(
                                    invoke_name, invoke_content),
                            ),
                        ))

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output)

            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content)
        except Exception:
            logger.exception("Error extracting DeepSeek V4.1 tool calls")
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
            args_json = self._invoke_arguments(invoke_name, invoke_body)
            idx = self.current_tool_index
            self.current_tool_index += 1

            self.prev_tool_call_arr.append(
                {"name": invoke_name, "arguments": args_json})
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
            overlap = _partial_tags_overlap(
                current_text, self.tool_call_start_tokens)
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
