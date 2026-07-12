# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
import uuid
from collections.abc import Sequence
from typing import Any, Optional, Union

import regex as re

from .abstract_tool_parser import ToolParser, ToolParserManager
from ..protocal.openai_protocol import *

import logging
logger = logging.getLogger(__name__)


def _get_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


def _partial_tags_overlap(text: str, tags: list[str]) -> int:
    return max((_partial_tag_overlap(text, tag) for tag in tags), default=0)


@ToolParserManager.register_module(["hy_v3", "hunyuan"])
class HYV3ToolParser(ToolParser):
    """
    Hunyuan/HY-V3 native tool parser.

    HY-V3 emits tool calls in the tokenizer-specific form:

    <tool_calls:suffix>
    <tool_call:suffix>{name}<tool_sep:suffix>
    <arg_key:suffix>{key}</arg_key:suffix>
    <arg_value:suffix>{value}</arg_value:suffix>
    </tool_call:suffix>
    </tool_calls:suffix>
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []
        self.current_tool_index: int = 0
        self._sent_content_idx: int = 0

        self.token_suffix = self._token_suffix()
        self.tool_calls_start_token = self._resolve_token("tool_calls")
        self.tool_calls_end_token = self._resolve_token(
            "tool_calls", closing=True)
        self.tool_call_start_token = self._resolve_token("tool_call")
        self.tool_call_end_token = self._resolve_token(
            "tool_call", closing=True)
        self.tool_sep_token = self._resolve_token("tool_sep")
        self.arg_key_start_token = self._resolve_token("arg_key")
        self.arg_key_end_token = self._resolve_token("arg_key", closing=True)
        self.arg_value_start_token = self._resolve_token("arg_value")
        self.arg_value_end_token = self._resolve_token(
            "arg_value", closing=True)

        self.tool_call_start_tokens = [
            self.tool_calls_start_token,
            self.tool_call_start_token,
        ]

        self.tool_call_start_token_id = self._token_id(
            self.tool_call_start_token)
        self.tool_calls_start_token_id = self._token_id(
            self.tool_calls_start_token)
        self.tool_call_end_token_id = self._token_id(self.tool_call_end_token)

        self.tool_call_complete_regex = re.compile(
            re.escape(self.tool_call_start_token)
            + r"(.*?)"
            + re.escape(self.tool_sep_token)
            + r"(.*?)"
            + re.escape(self.tool_call_end_token),
            re.DOTALL,
        )
        self.tool_call_partial_regex = re.compile(
            re.escape(self.tool_call_start_token)
            + r"(.*?)"
            + re.escape(self.tool_sep_token)
            + r"(.*)",
            re.DOTALL,
        )
        self.func_arg_regex = re.compile(
            re.escape(self.arg_key_start_token)
            + r"(.*?)"
            + re.escape(self.arg_key_end_token)
            + r"\s*"
            + re.escape(self.arg_value_start_token)
            + r"(.*?)"
            + re.escape(self.arg_value_end_token),
            re.DOTALL,
        )

    def get_token_ids(self, text: str) -> list[int]:
        token_id = self._token_id(text)
        return [token_id] if token_id is not None else [0]

    def has_tool_call(self, text: str) -> bool:
        return any(token in text for token in self.tool_call_start_tokens)

    def _token_suffix(self) -> str:
        init_kwargs = getattr(self.model_tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict) and init_kwargs.get("token_suffix"):
            return str(init_kwargs["token_suffix"])

        for token in self._safe_vocab():
            match = re.match(r"^<tool_call(:[^>]*)>$", token)
            if match:
                return match.group(1)
        return ":opensource"

    def _safe_vocab(self) -> dict[str, int]:
        try:
            return self.vocab
        except Exception:
            return {}

    def _token_id(self, token: str) -> Optional[int]:
        token_id = self._safe_vocab().get(token)
        return token_id if token_id is not None else None

    def _resolve_token(self, name: str, closing: bool = False) -> str:
        vocab = self._safe_vocab()
        prefix = "</" if closing else "<"
        fallback = f"{prefix}{name}{self.token_suffix}>"
        bare = f"{prefix}{name}>"

        exact_candidates = [fallback, bare]
        for candidate in exact_candidates:
            if candidate in vocab:
                return candidate

        if closing:
            pattern = re.compile(r"^</" + re.escape(name) + r"(?::[^>]*)?>$")
        else:
            pattern = re.compile(r"^<" + re.escape(name) + r"(?::[^>]*)?>$")

        candidates = [token for token in vocab if pattern.match(token)]
        if candidates:
            suffix_candidate = f"{prefix}{name}{self.token_suffix}>"
            if suffix_candidate in candidates:
                return suffix_candidate
            opensource_candidate = f"{prefix}{name}:opensource>"
            if opensource_candidate in candidates:
                return opensource_candidate
            return sorted(candidates, key=lambda item: (len(item), item))[0]

        return fallback

    def _generate_tool_call_id(self) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    def _find_first_tool_start(self, text: str) -> int:
        positions = [
            text.find(token) for token in self.tool_call_start_tokens
            if text.find(token) >= 0
        ]
        return min(positions) if positions else -1

    def _tool_region(self, model_output: str) -> tuple[int, str]:
        calls_start_idx = model_output.find(self.tool_calls_start_token)
        first_tool_idx = self._find_first_tool_start(model_output)
        if calls_start_idx >= 0:
            body_start = calls_start_idx + len(self.tool_calls_start_token)
            body_end = model_output.find(self.tool_calls_end_token, body_start)
            if body_end < 0:
                body_end = len(model_output)
            return calls_start_idx, model_output[body_start:body_end]

        if first_tool_idx >= 0:
            return first_tool_idx, model_output[first_tool_idx:]
        return -1, ""

    def _arguments_schema(self, func_name: str,
                          tools: Optional[list[ChatCompletionToolsParam]]
                          ) -> dict[str, Any]:
        for tool in tools or []:
            function = _get_value(tool, "function")
            if _get_value(function, "name") != func_name:
                continue
            parameters = _get_value(function, "parameters") or {}
            properties = _get_value(parameters, "properties")
            if isinstance(properties, dict):
                return properties
            if isinstance(parameters, dict):
                return parameters
        return {}

    def _arg_schema(self, func_name: str, arg_name: str,
                    tools: Optional[list[ChatCompletionToolsParam]]
                    ) -> dict[str, Any]:
        schema = self._arguments_schema(func_name, tools).get(arg_name, {})
        return schema if isinstance(schema, dict) else {}

    def _schema_type(self, schema: dict[str, Any]) -> Optional[str]:
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            for item in schema_type:
                if item != "null":
                    return str(item)
            return str(schema_type[0]) if schema_type else None
        if schema_type is None:
            for key in ("anyOf", "oneOf"):
                variants = schema.get(key)
                if isinstance(variants, list):
                    for variant in variants:
                        if not isinstance(variant, dict):
                            continue
                        variant_type = variant.get("type")
                        if variant_type and variant_type != "null":
                            return str(variant_type)
            return None
        return str(schema_type)

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

    def _parse_value(self, func_name: str, arg_name: str, value: str,
                     request: ChatCompletionRequest) -> Any:
        value = value.strip()
        if value.lower() == "null":
            return None

        schema_type = self._schema_type(
            self._arg_schema(func_name, arg_name, request.tools))
        if schema_type in ("string", "str", "text", "enum"):
            return value
        if schema_type in ("integer", "int", "long", "short"):
            try:
                return int(value)
            except ValueError:
                logger.warning(
                    "HY-V3 argument %s.%s is not an integer: %s",
                    func_name, arg_name, value)
                return value
        if schema_type in ("number", "float", "double"):
            try:
                return float(value)
            except ValueError:
                logger.warning("HY-V3 argument %s.%s is not a number: %s",
                               func_name, arg_name, value)
                return value
        if schema_type in ("boolean", "bool"):
            lowered = value.lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
            logger.warning("HY-V3 argument %s.%s is not a boolean: %s",
                           func_name, arg_name, value)
            return value
        if schema_type in ("array", "object"):
            parsed = self._deserialize(value)
            if schema_type == "array" and isinstance(parsed, list):
                return parsed
            if schema_type == "object" and isinstance(parsed, dict):
                return parsed
            return value

        return self._deserialize(value)

    def _parse_tool_call(self, func_name: str, func_args: str,
                         request: ChatCompletionRequest) -> ToolCall:
        name = func_name.strip()
        if not name:
            raise ValueError("HY-V3 tool call does not contain a function name")

        args: dict[str, Any] = {}
        for arg_key, arg_value in self.func_arg_regex.findall(func_args):
            key = arg_key.strip()
            args[key] = self._parse_value(name, key, arg_value, request)

        stripped_args = func_args.strip()
        if not args and stripped_args:
            parsed = self._deserialize(stripped_args)
            if isinstance(parsed, dict):
                args = parsed

        return ToolCall(
            id=self._generate_tool_call_id(),
            type="function",
            function=FunctionCall(
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
            ),
        )

    def _store_tool_call(self, tool_call: ToolCall,
                         index: Optional[int] = None) -> str:
        arguments_obj = self._deserialize(tool_call.function.arguments)
        entry = {
            "name": tool_call.function.name,
            "arguments": arguments_obj,
        }
        if index is None:
            self.prev_tool_call_arr.append(entry)
            self.streamed_args_for_tool.append(tool_call.function.arguments)
            return tool_call.function.arguments

        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= index:
            self.streamed_args_for_tool.append("")
        self.prev_tool_call_arr[index] = entry
        self.streamed_args_for_tool[index] = tool_call.function.arguments
        return tool_call.function.arguments

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        first_tool_idx, tool_body = self._tool_region(model_output)
        if first_tool_idx < 0:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

        try:
            matches = self.tool_call_complete_regex.findall(tool_body)
            if not matches:
                matches = self.tool_call_partial_regex.findall(tool_body)

            tool_calls = [
                self._parse_tool_call(name, args, request)
                for name, args in matches
            ]

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output)

            self.prev_tool_call_arr.clear()
            self.streamed_args_for_tool.clear()
            for tool_call in tool_calls:
                self._store_tool_call(tool_call)

            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None
            if content is not None and len(content.strip()) == 0:
                content = None
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content)
        except Exception:
            logger.exception("Error extracting HY-V3 tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output)

    def _reset_streaming_state(self) -> None:
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self.current_tool_index = 0
        self._sent_content_idx = 0

    def _extract_content(self, current_text: str) -> Optional[str]:
        first_tool_idx = self._find_first_tool_start(current_text)
        if first_tool_idx < 0:
            overlap = _partial_tags_overlap(current_text,
                                            self.tool_call_start_tokens)
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = first_tool_idx

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx:sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _extract_delta_tool_calls(
        self,
        current_text: str,
        request: ChatCompletionRequest,
    ) -> list[DeltaToolCall]:
        complete_calls = list(self.tool_call_complete_regex.findall(current_text))
        deltas: list[DeltaToolCall] = []

        while len(complete_calls) > self.current_tool_index:
            name, args = complete_calls[self.current_tool_index]
            idx = self.current_tool_index
            tool_call = self._parse_tool_call(name, args, request)
            self._store_tool_call(tool_call, idx)
            deltas.append(
                DeltaToolCall(
                    index=idx,
                    id=tool_call.id,
                    type=tool_call.type,
                    function=DeltaFunctionCall(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                ))
            self.current_tool_index += 1

        return deltas

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
        delta_tool_calls = self._extract_delta_tool_calls(current_text, request)

        if content or delta_tool_calls:
            return DeltaMessage(content=content, tool_calls=delta_tool_calls)
        return None
