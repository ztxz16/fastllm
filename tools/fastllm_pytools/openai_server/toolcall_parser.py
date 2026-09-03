import copy
import difflib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .protocal.openai_protocol import (
    ChatCompletionRequest,
    ExtractedToolCallInformation,
)
from .tool_parsers import ToolParserManager
from .tool_schema import (
    dereference_schema,
    get_value as _get_value,
    json_type_name,
    load_strict_json,
    schema_types,
)


@dataclass(frozen=True)
class ToolCallDiagnostic:
    code: str
    message: str
    tool_name: Optional[str] = None
    index: Optional[int] = None
    argument_name: Optional[str] = None
    allowed_tool_names: tuple[str, ...] = ()
    closest_tool_name: Optional[str] = None
    similarity_ratio: Optional[float] = None


@dataclass
class ToolCallParseResult:
    content: Optional[str] = None
    tools_called: bool = False
    valid_tool_calls: List[Any] = field(default_factory=list)
    invalid_tool_calls: List[Any] = field(default_factory=list)
    diagnostics: List[ToolCallDiagnostic] = field(default_factory=list)
    has_invalid_tool_block: bool = False


@dataclass(frozen=True)
class ToolCallValidationResult:
    valid: bool
    diagnostics: List[ToolCallDiagnostic] = field(default_factory=list)
    valid_tool_calls: List[Any] = field(default_factory=list)
    invalid_tool_calls: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCallConstraintDescriptor:
    constraint_type: str
    model_type: str
    tool_names: tuple[str, ...]
    allowed_tool_names: tuple[str, ...]
    tool_choice: Any
    requires_tool_call: bool
    named_tool_choice: Optional[str]
    parallel_tool_calls: Optional[bool]
    schemas: Dict[str, Any] = field(default_factory=dict)
    parameter_names: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    strict_tool_names: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "model_type": self.model_type,
            "tool_names": list(self.tool_names),
            "allowed_tool_names": list(self.allowed_tool_names),
            "tool_choice": self.tool_choice,
            "requires_tool_call": self.requires_tool_call,
            "named_tool_choice": self.named_tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "schemas": copy.deepcopy(self.schemas),
            "parameter_names": {
                name: list(names)
                for name, names in self.parameter_names.items()
            },
            "strict_tool_names": list(self.strict_tool_names),
        }


class FunctionCallParser:
    """SGLang-style facade over existing FastLLM tool parsers.

    This class does not implement a second detector framework. It delegates
    wire-format parsing to `ToolParserManager` parsers, then adds request-tool
    validation and diagnostics.
    """

    def __init__(
        self,
        tools: Optional[Iterable[Any]],
        tool_choice: Any = "auto",
        tool_parser_name: str = "deepseek_v4",
        tokenizer: Optional[Any] = None,
        parser: Optional[Any] = None,
        model: str = "toolcall-parser",
        parallel_tool_calls: Optional[bool] = None,
    ):
        self.tools = list(tools or [])
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.tool_parser_name = tool_parser_name
        self.tool_index = self._build_tool_index(self.tools)
        self._request = ChatCompletionRequest(
            model=model,
            messages=[],
            tools=self.tools or None,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        if parser is None:
            parser_cls = ToolParserManager.get_tool_parser(tool_parser_name)
            parser = parser_cls(tokenizer or _EmptyToolTokenizer())
        self.parser = parser
        self.compat_mode = _env_enabled("FT_TOOLCALL_COMPAT_MODE")
        self.forward_unknown_tools = _env_enabled(
            "FT_TOOLCALL_FORWARD_UNKNOWN_TOOLS")
        self.invalid_stream_tool_indices: set[int] = set()
        self.valid_stream_tool_indices: set[int] = set()
        self.stream_index_map: Dict[int, int] = {}
        self.stream_diagnostics: List[ToolCallDiagnostic] = []
        self.stream_tool_call_fragments: Dict[int, Dict[str, str]] = {}
        self.buffered_stream_tool_calls: Dict[int, List[Any]] = {}
        self.stream_text = ""
        self._stream_final_diagnostics: Optional[
            List[ToolCallDiagnostic]] = None

    @classmethod
    def from_request(
        cls,
        request: ChatCompletionRequest,
        tool_parser_name: str = "deepseek_v4",
        tokenizer: Optional[Any] = None,
        parser: Optional[Any] = None,
    ) -> "FunctionCallParser":
        return cls(
            tools=request.tools,
            tool_choice=request.tool_choice,
            tool_parser_name=tool_parser_name,
            tokenizer=tokenizer,
            parser=parser,
            model=request.model,
            parallel_tool_calls=getattr(request, "parallel_tool_calls", None),
        )

    @classmethod
    def build_constraint_descriptor_from_request(
        cls,
        request: ChatCompletionRequest,
        tool_parser_name: str = "deepseek_v4",
    ) -> Optional[ToolCallConstraintDescriptor]:
        return cls(
            tools=request.tools,
            tool_choice=request.tool_choice,
            tool_parser_name=tool_parser_name,
            parser=_NoopToolParser(),
            model=request.model,
            parallel_tool_calls=getattr(request, "parallel_tool_calls", None),
        ).build_constraint_descriptor()

    @property
    def has_tools(self) -> bool:
        return bool(self.tools)

    def has_tool_call(self, text: str) -> bool:
        if not self.has_tools or self.tool_choice == "none":
            return False
        detector = getattr(self.parser, "has_tool_call", None)
        if callable(detector):
            return bool(detector(text))
        tokens = list(getattr(self.parser, "tool_call_start_tokens", []) or [])
        token = getattr(self.parser, "tool_call_start_token", None)
        if token:
            tokens.append(token)
        return any(token in text for token in tokens)

    @property
    def has_valid_streamed_tool_calls(self) -> bool:
        return bool(self.valid_stream_tool_indices)

    def get_token_ids(self, text: str) -> list[int]:
        get_token_ids = getattr(self.parser, "get_token_ids", None)
        if callable(get_token_ids):
            return get_token_ids(text)
        return [0]

    def build_constraint_descriptor(
        self,
    ) -> Optional[ToolCallConstraintDescriptor]:
        if not self.has_tools or self.tool_choice == "none":
            return None

        named_tool_choice = self._named_tool_choice()
        if named_tool_choice is not None:
            allowed_tool_names = (named_tool_choice,)
        else:
            allowed_tool_names = tuple(self.tool_index)

        strict_tool_names: List[str] = []
        schemas: Dict[str, Any] = {}
        parameter_names: Dict[str, tuple[str, ...]] = {}
        for name in self.tool_index:
            function = self._tool_function(name)
            parameters = _get_value(function, "parameters")
            names = _schema_property_names(parameters)
            if names:
                parameter_names[name] = tuple(names)
            if not _get_value(function, "strict"):
                continue
            strict_tool_names.append(name)
            if parameters is not None:
                schemas[name] = copy.deepcopy(parameters)

        return ToolCallConstraintDescriptor(
            constraint_type=self._constraint_type(),
            model_type=self.tool_parser_name,
            tool_names=tuple(self.tool_index),
            allowed_tool_names=allowed_tool_names,
            tool_choice=_normalize_tool_choice_for_descriptor(
                self.tool_choice),
            requires_tool_call=self._requires_tool_call(),
            named_tool_choice=named_tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            schemas=schemas,
            parameter_names=parameter_names,
            strict_tool_names=tuple(strict_tool_names),
        )

    def parse_non_stream(self, text: str) -> ToolCallParseResult:
        if not self.has_tools or self.tool_choice == "none":
            return ToolCallParseResult(content=text)

        extracted = self.parser.extract_tool_calls(text, self._request)
        validation = self.validate_tool_calls(extracted.tool_calls)
        if self.has_tool_call(text) and not validation.valid_tool_calls:
            if not validation.invalid_tool_calls:
                diagnostic = ToolCallDiagnostic(
                    code="malformed_tool_block",
                    message="tool call markup was detected but no valid tool call was parsed",
                )
                return ToolCallParseResult(
                    content=extracted.content,
                    tools_called=False,
                    valid_tool_calls=[],
                    invalid_tool_calls=[],
                    diagnostics=[diagnostic],
                    has_invalid_tool_block=True,
                )
        return self._result_from_extracted(extracted, validation)

    def parse_stream_chunk(
        self,
        *,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Iterable[int],
        current_token_ids: Iterable[int],
        delta_token_ids: Iterable[int],
    ) -> ToolCallParseResult:
        if not self.has_tools or self.tool_choice == "none":
            return ToolCallParseResult(content=delta_text)

        self.stream_text = current_text
        delta = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=list(previous_token_ids),
            current_token_ids=list(current_token_ids),
            delta_token_ids=list(delta_token_ids),
            request=self._request,
        )
        if delta is None:
            return ToolCallParseResult()

        valid_tool_calls: List[Any] = []
        invalid_tool_calls: List[Any] = []
        diagnostics: List[ToolCallDiagnostic] = []

        for tool_call in delta.tool_calls:
            raw_index = _tool_call_index(tool_call)
            if raw_index is None:
                diagnostics.append(
                    ToolCallDiagnostic(
                        code="missing_tool_index",
                        message="stream tool call is missing index",
                    ))
                invalid_tool_calls.append(tool_call)
                continue
            if raw_index in self.invalid_stream_tool_indices:
                invalid_tool_calls.append(tool_call)
                continue

            name = _tool_call_name(tool_call)
            if name:
                choice_diagnostic = self._tool_choice_diagnostic(
                    name, raw_index)
                if choice_diagnostic is not None:
                    self.invalid_stream_tool_indices.add(raw_index)
                    invalid_tool_calls.append(tool_call)
                    diagnostics.append(choice_diagnostic)
                    continue
                if name not in self.tool_index:
                    diagnostic = self._invalid_tool_name_diagnostic(
                        name, raw_index)
                    if self.forward_unknown_tools:
                        parallel_diagnostic = self._parallel_stream_diagnostic(
                            raw_index)
                        if parallel_diagnostic is not None:
                            self.invalid_stream_tool_indices.add(raw_index)
                            invalid_tool_calls.append(tool_call)
                            diagnostics.append(parallel_diagnostic)
                            continue
                        self.valid_stream_tool_indices.add(raw_index)
                        self.stream_index_map.setdefault(
                            raw_index, len(self.stream_index_map))
                        diagnostics.append(diagnostic)
                        self._record_stream_tool_call(raw_index, tool_call)
                        valid_tool_calls.append(
                            _copy_tool_call_with_index(
                                tool_call, self.stream_index_map[raw_index]))
                        continue
                    self.invalid_stream_tool_indices.add(raw_index)
                    invalid_tool_calls.append(tool_call)
                    diagnostics.append(diagnostic)
                    continue
                parallel_diagnostic = self._parallel_stream_diagnostic(
                    raw_index)
                if parallel_diagnostic is not None:
                    self.invalid_stream_tool_indices.add(raw_index)
                    invalid_tool_calls.append(tool_call)
                    diagnostics.append(parallel_diagnostic)
                    continue
                self.valid_stream_tool_indices.add(raw_index)
                self.stream_index_map.setdefault(raw_index,
                                                 len(self.stream_index_map))
                if self._is_strict_tool(name):
                    self._record_stream_tool_call(raw_index, tool_call)
                    self._buffer_stream_tool_call(raw_index, tool_call)
                    continue
            elif raw_index not in self.valid_stream_tool_indices:
                self.invalid_stream_tool_indices.add(raw_index)
                invalid_tool_calls.append(tool_call)
                diagnostics.append(
                    ToolCallDiagnostic(
                        code="missing_tool_name",
                        message="stream tool call is missing function.name",
                        index=raw_index,
                    ))
                continue
            else:
                state = self.stream_tool_call_fragments.get(raw_index, {})
                if self._is_strict_tool(state.get("name")):
                    self._record_stream_tool_call(raw_index, tool_call)
                    self._buffer_stream_tool_call(raw_index, tool_call)
                    continue

            self._record_stream_tool_call(raw_index, tool_call)
            valid_tool_calls.append(
                _copy_tool_call_with_index(
                    tool_call, self.stream_index_map[raw_index]))

        self.stream_diagnostics.extend(diagnostics)
        return ToolCallParseResult(
            content=delta.content,
            tools_called=bool(valid_tool_calls),
            valid_tool_calls=valid_tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            diagnostics=diagnostics,
            has_invalid_tool_block=bool(invalid_tool_calls or diagnostics),
        )

    def validate_tool_calls(
        self, tool_calls: Iterable[Any]) -> ToolCallValidationResult:
        valid_tool_calls: List[Any] = []
        invalid_tool_calls: List[Any] = []
        diagnostics: List[ToolCallDiagnostic] = []

        for index, tool_call in enumerate(tool_calls or []):
            name = _tool_call_name(tool_call)
            if not name:
                invalid_tool_calls.append(tool_call)
                diagnostics.append(
                    ToolCallDiagnostic(
                        code="missing_tool_name",
                        message="tool call is missing function.name",
                        index=index,
                    ))
                continue
            diagnostic = self._tool_choice_diagnostic(name, index)
            if diagnostic is not None:
                invalid_tool_calls.append(tool_call)
                diagnostics.append(diagnostic)
                continue
            if name in self.tool_index:
                schema_diagnostics = self._strict_schema_diagnostics(
                    tool_call, name, index)
                if schema_diagnostics:
                    invalid_tool_calls.append(tool_call)
                    diagnostics.extend(schema_diagnostics)
                    continue
                valid_tool_calls.append(tool_call)
                continue
            diagnostic = self._invalid_tool_name_diagnostic(name, index)
            if self.forward_unknown_tools:
                valid_tool_calls.append(tool_call)
                diagnostics.append(diagnostic)
                continue
            invalid_tool_calls.append(tool_call)
            diagnostics.append(diagnostic)

        return ToolCallValidationResult(
            valid=not invalid_tool_calls,
            diagnostics=diagnostics,
            valid_tool_calls=valid_tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )

    def _result_from_extracted(
        self,
        extracted: ExtractedToolCallInformation,
        validation: ToolCallValidationResult,
    ) -> ToolCallParseResult:
        diagnostics = list(validation.diagnostics)
        has_invalid_tool_block = bool(validation.invalid_tool_calls)
        valid_tool_calls = list(validation.valid_tool_calls)
        invalid_tool_calls = list(validation.invalid_tool_calls)
        if self._requires_tool_call() and not validation.valid_tool_calls:
            required_name = self._named_tool_choice()
            message = (
                f"tool_choice requires function {required_name!r} but no "
                "valid tool call was produced"
                if required_name is not None else
                "tool_choice='required' was set but no valid tool call was produced"
            )
            diagnostics.append(
                ToolCallDiagnostic(
                    code="tool_choice_violation",
                    message=message,
                ))
            has_invalid_tool_block = True
        if (self.parallel_tool_calls is False
                and len(validation.valid_tool_calls) > 1):
            diagnostics.append(
                ToolCallDiagnostic(
                    code="parallel_tool_calls_violation",
                    message="parallel_tool_calls=false was set but multiple valid tool calls were produced",
                ))
            invalid_tool_calls.extend(valid_tool_calls)
            valid_tool_calls = []
            has_invalid_tool_block = True
        return ToolCallParseResult(
            content=extracted.content,
            tools_called=bool(valid_tool_calls),
            valid_tool_calls=valid_tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            diagnostics=diagnostics,
            has_invalid_tool_block=has_invalid_tool_block,
        )

    def finalize_stream(self) -> List[ToolCallDiagnostic]:
        return self._finalize_stream_diagnostics()

    def flush_stream_tool_calls(self) -> ToolCallParseResult:
        diagnostics = self._finalize_stream_diagnostics()
        if diagnostics:
            return ToolCallParseResult(
                diagnostics=diagnostics,
                has_invalid_tool_block=True,
            )
        content = None
        content_flusher = getattr(self.parser, "flush_streaming_content", None)
        if callable(content_flusher):
            content = content_flusher()
        tool_calls: List[Any] = []
        for raw_index in sorted(self.buffered_stream_tool_calls):
            tool_calls.extend(self.buffered_stream_tool_calls[raw_index])
        self.buffered_stream_tool_calls.clear()
        return ToolCallParseResult(
            content=content,
            tools_called=bool(tool_calls),
            valid_tool_calls=tool_calls,
        )

    def _finalize_stream_diagnostics(self) -> List[ToolCallDiagnostic]:
        if self._stream_final_diagnostics is not None:
            return list(self._stream_final_diagnostics)
        diagnostics: List[ToolCallDiagnostic] = []
        parser_error = None
        parser_error_fn = getattr(self.parser, "streaming_parse_error", None)
        if callable(parser_error_fn):
            parser_error = parser_error_fn()
        if parser_error:
            diagnostics.append(
                ToolCallDiagnostic(
                    code="malformed_tool_block",
                    message=str(parser_error),
                ))
        elif (self.has_tool_call(self.stream_text)
                and not self.valid_stream_tool_indices
                and not self.invalid_stream_tool_indices):
            extracted = self.parser.extract_tool_calls(
                self.stream_text, self._request)
            if not extracted.tool_calls:
                diagnostics.append(
                    ToolCallDiagnostic(
                        code="malformed_tool_block",
                        message=(
                            "tool call markup was detected but the stream "
                            "ended before a complete tool call was parsed"
                        ),
                    ))
        if self._requires_tool_call() and not self.has_valid_streamed_tool_calls:
            required_name = self._named_tool_choice()
            message = (
                f"tool_choice requires function {required_name!r} but no "
                "valid stream tool call was produced"
                if required_name is not None else
                "tool_choice='required' was set but no valid stream tool call was produced"
            )
            diagnostics.append(
                ToolCallDiagnostic(
                    code="tool_choice_violation",
                    message=message,
                ))
        for raw_index in sorted(self.valid_stream_tool_indices):
            state = self.stream_tool_call_fragments.get(raw_index, {})
            name = state.get("name")
            if not name:
                continue
            diagnostics.extend(
                self._strict_schema_diagnostics(
                    {
                        "function": {
                            "name": name,
                            "arguments": state.get("arguments", ""),
                        }
                    },
                    name,
                    self.stream_index_map.get(raw_index, raw_index),
                ))
        self.stream_diagnostics.extend(diagnostics)
        self._stream_final_diagnostics = list(diagnostics)
        return diagnostics

    @staticmethod
    def _build_tool_index(tools: Iterable[Any]) -> Dict[str, int]:
        index: Dict[str, int] = {}
        for position, tool in enumerate(tools or []):
            function = _get_value(tool, "function")
            name = _get_value(function, "name") if function is not None else None
            if name:
                index[name] = position
        return index

    def _invalid_tool_name_diagnostic(
        self,
        name: str,
        index: int,
    ) -> ToolCallDiagnostic:
        allowed_names = tuple(self.tool_index)
        closest_name, ratio = _closest_tool_name(name, allowed_names)
        if self.compat_mode and closest_name is not None:
            message = (
                f"tool name {name!r} is not in request tools; "
                f"closest allowed tool is {closest_name!r} "
                f"(similarity={ratio:.3f})"
            )
        else:
            message = f"tool name {name!r} is not in request tools"
            closest_name = None
            ratio = None
            allowed_names = ()
        return ToolCallDiagnostic(
            code="invalid_tool_name",
            message=message,
            tool_name=name,
            index=index,
            allowed_tool_names=allowed_names,
            closest_tool_name=closest_name,
            similarity_ratio=ratio,
        )

    def _requires_tool_call(self) -> bool:
        return (self.tool_choice == "required"
                or self._named_tool_choice() is not None)

    def _named_tool_choice(self) -> Optional[str]:
        if isinstance(self.tool_choice, str) or self.tool_choice is None:
            return None
        function = _get_value(self.tool_choice, "function")
        if function is None:
            return None
        return _get_value(function, "name")

    def _tool_choice_diagnostic(
        self,
        name: str,
        index: int,
    ) -> Optional[ToolCallDiagnostic]:
        required_name = self._named_tool_choice()
        if required_name is None or name == required_name:
            return None
        return ToolCallDiagnostic(
            code="tool_choice_violation",
            message=(
                f"tool_choice requires function {required_name!r} "
                f"but model produced {name!r}"
            ),
            tool_name=name,
            index=index,
            allowed_tool_names=(required_name,),
        )

    def _parallel_stream_diagnostic(
        self,
        raw_index: int,
    ) -> Optional[ToolCallDiagnostic]:
        if self.parallel_tool_calls is not False:
            return None
        if raw_index in self.valid_stream_tool_indices:
            return None
        if not self.valid_stream_tool_indices:
            return None
        return ToolCallDiagnostic(
            code="parallel_tool_calls_violation",
            message=(
                "parallel_tool_calls=false was set but stream produced "
                "more than one tool call"
            ),
            index=raw_index,
        )

    def _record_stream_tool_call(self, raw_index: int, tool_call: Any) -> None:
        state = self.stream_tool_call_fragments.setdefault(
            raw_index, {"name": "", "arguments": ""})
        function = _get_value(tool_call, "function")
        name = _get_value(function, "name")
        if name:
            state["name"] = name
        arguments = _get_value(function, "arguments")
        if arguments:
            state["arguments"] += str(arguments)

    def _buffer_stream_tool_call(self, raw_index: int, tool_call: Any) -> None:
        external_index = self.stream_index_map[raw_index]
        self.buffered_stream_tool_calls.setdefault(raw_index, []).append(
            _copy_tool_call_with_index(tool_call, external_index))

    def _strict_schema_diagnostics(
        self,
        tool_call: Any,
        name: str,
        index: int,
    ) -> List[ToolCallDiagnostic]:
        function = self._tool_function(name)
        if not _get_value(function, "strict"):
            return []

        arguments = _tool_call_arguments(tool_call)
        if not isinstance(arguments, str):
            return [
                ToolCallDiagnostic(
                    code="malformed_arguments_json",
                    message="function.arguments must be a JSON string",
                    tool_name=name,
                    index=index,
                )
            ]

        try:
            parsed_arguments = load_strict_json(arguments)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return [
                ToolCallDiagnostic(
                    code="malformed_arguments_json",
                    message=f"function.arguments is not valid JSON: {exc}",
                    tool_name=name,
                    index=index,
                )
            ]

        if not isinstance(parsed_arguments, dict):
            return [
                ToolCallDiagnostic(
                    code="malformed_arguments_json",
                    message="function.arguments must decode to a JSON object",
                    tool_name=name,
                    index=index,
                )
            ]

        parameters = _get_value(function, "parameters") or {}
        return _schema_value_diagnostics(
            parsed_arguments,
            parameters,
            "",
            name,
            index,
            root_schema=parameters,
        )

    def _tool_function(self, name: str) -> Any:
        position = self.tool_index.get(name)
        if position is None:
            return None
        tool = self.tools[position]
        return _get_value(tool, "function")

    def _is_strict_tool(self, name: Optional[str]) -> bool:
        if name is None:
            return False
        return bool(_get_value(self._tool_function(name), "strict"))

    def _constraint_type(self) -> str:
        if self.tool_parser_name == "deepseek_v4":
            return "deepseek_v4_dsml"
        if self.tool_parser_name in {"dots", "dots3_note"}:
            return "dots_xml"
        return f"{self.tool_parser_name}_tool_call"


def _tool_call_name(tool_call: Any) -> Optional[str]:
    function = _get_value(tool_call, "function")
    if function is None:
        return None
    return _get_value(function, "name")


def _tool_call_arguments(tool_call: Any) -> Any:
    function = _get_value(tool_call, "function")
    if function is None:
        return None
    return _get_value(function, "arguments")


def _tool_call_index(tool_call: Any) -> Optional[int]:
    return _get_value(tool_call, "index")


def _copy_tool_call_with_index(tool_call: Any, index: int) -> Any:
    if hasattr(tool_call, "model_copy"):
        return tool_call.model_copy(update={"index": index})
    if hasattr(tool_call, "copy"):
        return tool_call.copy(update={"index": index})
    if isinstance(tool_call, dict):
        copied = dict(tool_call)
        copied["index"] = index
        return copied
    return tool_call


def _closest_tool_name(
    actual_name: str,
    allowed_names: tuple[str, ...],
) -> tuple[Optional[str], Optional[float]]:
    if not allowed_names:
        return None, None
    closest_name = max(
        allowed_names,
        key=lambda name: difflib.SequenceMatcher(
            None, actual_name, name).ratio(),
    )
    ratio = difflib.SequenceMatcher(None, actual_name, closest_name).ratio()
    return closest_name, ratio


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "on", "true", "yes"}


def _required_arguments(parameters: Any) -> List[str]:
    required = _get_value(parameters, "required") or []
    if not isinstance(required, list):
        return []
    return [name for name in required if isinstance(name, str)]


def _schema_value_diagnostics(
    value: Any,
    schema: Any,
    path: str,
    tool_name: str,
    index: int,
    *,
    root_schema: Any = None,
    depth: int = 0,
) -> List[ToolCallDiagnostic]:
    if schema is True or schema is None:
        return []
    if schema is False:
        return [_schema_value_diagnostic(
            "invalid_argument_value",
            f"argument {_display_schema_path(path)!r} is not allowed",
            path,
            tool_name,
            index,
        )]
    if not isinstance(schema, dict):
        return []
    if root_schema is None:
        root_schema = schema
    if depth > 64:
        return [_schema_value_diagnostic(
            "invalid_argument_value",
            f"argument {_display_schema_path(path)!r} exceeds schema nesting limit",
            path,
            tool_name,
            index,
        )]

    schema = dereference_schema(schema, root_schema)
    if not isinstance(schema, dict):
        return []

    child_kwargs = {
        "root_schema": root_schema,
        "depth": depth + 1,
    }
    diagnostics: List[ToolCallDiagnostic] = []

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for candidate in all_of:
            diagnostics.extend(_schema_value_diagnostics(
                value, candidate, path, tool_name, index, **child_kwargs))

    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and not any(
            not _schema_value_diagnostics(
                value, candidate, path, tool_name, index, **child_kwargs)
            for candidate in any_of):
        diagnostics.append(_schema_combinator_diagnostic(
            value, schema, "anyOf", path, tool_name, index, root_schema))

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        match_count = sum(
            not _schema_value_diagnostics(
                value, candidate, path, tool_name, index, **child_kwargs)
            for candidate in one_of)
        if match_count != 1:
            diagnostics.append(_schema_value_diagnostic(
                "invalid_argument_value",
                f"argument {_display_schema_path(path)!r} must match exactly "
                f"one oneOf schema; matched {match_count}",
                path,
                tool_name,
                index,
            ))

    negated = schema.get("not")
    if negated is not None and not _schema_value_diagnostics(
            value, negated, path, tool_name, index, **child_kwargs):
        diagnostics.append(_schema_value_diagnostic(
            "invalid_argument_value",
            f"argument {_display_schema_path(path)!r} matches a forbidden schema",
            path,
            tool_name,
            index,
        ))

    conditional = schema.get("if")
    if conditional is not None:
        condition_matches = not _schema_value_diagnostics(
            value, conditional, path, tool_name, index, **child_kwargs)
        selected = schema.get("then" if condition_matches else "else")
        if selected is not None:
            diagnostics.extend(_schema_value_diagnostics(
                value, selected, path, tool_name, index, **child_kwargs))

    expected_type = _get_value(schema, "type")
    if (value is None and schema.get("nullable") is True):
        type_matches = True
    else:
        type_matches = (expected_type is None
                        or _matches_json_schema_type(value, expected_type))
    if not type_matches:
        diagnostics.append(_schema_value_diagnostic(
            "invalid_argument_type",
            f"argument {_display_schema_path(path)!r} expected type "
            f"{expected_type!r} but got {json_type_name(value)!r}",
            path,
            tool_name,
            index,
        ))
        return diagnostics

    if "const" in schema and not _json_values_equal(value, schema["const"]):
        diagnostics.append(_schema_value_diagnostic(
            "invalid_argument_value",
            f"argument {_display_schema_path(path)!r} must equal {schema['const']!r}",
            path,
            tool_name,
            index,
        ))
    enum = schema.get("enum")
    if (isinstance(enum, list)
            and not any(_json_values_equal(value, item) for item in enum)):
        diagnostics.append(_schema_value_diagnostic(
            "invalid_argument_value",
            f"argument {_display_schema_path(path)!r} is not one of the allowed values",
            path,
            tool_name,
            index,
        ))

    if isinstance(value, dict):
        for child_name in _required_arguments(schema):
            if child_name in value:
                continue
            child_path = _join_schema_path(path, child_name)
            diagnostics.append(
                ToolCallDiagnostic(
                    code="missing_required_argument",
                    message=f"required argument {child_path!r} is missing",
                    tool_name=tool_name,
                    index=index,
                    argument_name=child_path,
                ))

        properties = _get_value(schema, "properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        additional = schema.get("additionalProperties", True)
        for child_name, child_value in value.items():
            child_path = _join_schema_path(path, child_name)
            if child_name in properties:
                diagnostics.extend(_schema_value_diagnostics(
                    child_value,
                    properties[child_name],
                    child_path,
                    tool_name,
                    index,
                    **child_kwargs,
                ))
            elif additional is False:
                diagnostics.append(_schema_value_diagnostic(
                    "unexpected_argument",
                    f"argument {child_path!r} is not defined by the tool schema",
                    child_path,
                    tool_name,
                    index,
                ))
            elif isinstance(additional, dict):
                diagnostics.extend(_schema_value_diagnostics(
                    child_value,
                    additional,
                    child_path,
                    tool_name,
                    index,
                    **child_kwargs,
                ))

    if isinstance(value, list):
        item_schema = _get_value(schema, "items")
        for item_index, item_value in enumerate(value):
            diagnostics.extend(_schema_value_diagnostics(
                item_value,
                item_schema,
                f"{path}[{item_index}]",
                tool_name,
                index,
                **child_kwargs,
            ))

    return diagnostics


def _schema_value_diagnostic(
    code: str,
    message: str,
    path: str,
    tool_name: str,
    index: int,
) -> ToolCallDiagnostic:
    return ToolCallDiagnostic(
        code=code,
        message=message,
        tool_name=tool_name,
        index=index,
        argument_name=path or None,
    )


def _schema_combinator_diagnostic(
    value: Any,
    schema: Any,
    keyword: str,
    path: str,
    tool_name: str,
    index: int,
    root_schema: Any,
) -> ToolCallDiagnostic:
    allowed_types = schema_types(schema, root_schema)
    if (allowed_types
            and not _matches_json_schema_type(value, list(allowed_types))):
        return _schema_value_diagnostic(
            "invalid_argument_type",
            f"argument {_display_schema_path(path)!r} expected one of types "
            f"{allowed_types!r} but got {json_type_name(value)!r}",
            path,
            tool_name,
            index,
        )
    return _schema_value_diagnostic(
        "invalid_argument_value",
        f"argument {_display_schema_path(path)!r} does not match any {keyword} schema",
        path,
        tool_name,
        index,
    )


def _join_schema_path(path: str, child_name: str) -> str:
    return f"{path}.{child_name}" if path else child_name


def _display_schema_path(path: str) -> str:
    return path or "$"


def _json_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return (len(left) == len(right)
                and all(_json_values_equal(a, b)
                        for a, b in zip(left, right)))
    if isinstance(left, dict):
        return (left.keys() == right.keys()
                and all(_json_values_equal(left[key], right[key])
                        for key in left))
    return left == right


def _schema_property_names(parameters: Any) -> List[str]:
    if _get_value(parameters, "type") not in (None, "object"):
        return []
    properties = _get_value(parameters, "properties")
    if not isinstance(properties, dict):
        return []
    return [name for name in properties if isinstance(name, str)]


def _matches_json_schema_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_json_schema_type(value, item)
                   for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return (isinstance(value, float) and math.isfinite(value)
                and value.is_integer())
    if expected_type == "number":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and math.isfinite(value)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True


def _normalize_tool_choice_for_descriptor(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str) or tool_choice is None:
        return tool_choice
    tool_type = _get_value(tool_choice, "type")
    function = _get_value(tool_choice, "function")
    name = _get_value(function, "name")
    if tool_type == "function" and name:
        return {
            "type": "function",
            "function": {"name": name},
        }
    return str(tool_choice)


class _EmptyToolTokenizer:
    def get_vocab(self):
        return {}


class _NoopToolParser:
    pass
