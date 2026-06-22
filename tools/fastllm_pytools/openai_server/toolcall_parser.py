import copy
import difflib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .protocal.openai_protocol import (
    ChatCompletionRequest,
    ExtractedToolCallInformation,
)
from .tool_parsers import ToolParserManager


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
        self._stream_finalized = False

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
        if not self.has_tools:
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
        if not self.has_tools:
            return None

        named_tool_choice = self._named_tool_choice()
        if named_tool_choice is not None:
            allowed_tool_names = (named_tool_choice,)
        else:
            allowed_tool_names = tuple(self.tool_index)

        strict_tool_names: List[str] = []
        schemas: Dict[str, Any] = {}
        for name in self.tool_index:
            function = self._tool_function(name)
            if not _get_value(function, "strict"):
                continue
            strict_tool_names.append(name)
            parameters = _get_value(function, "parameters")
            if parameters is not None:
                schemas[name] = copy.deepcopy(parameters)

        return ToolCallConstraintDescriptor(
            constraint_type=self._constraint_type(),
            model_type=self.tool_parser_name,
            tool_names=tuple(self.tool_index),
            allowed_tool_names=allowed_tool_names,
            tool_choice=_normalize_tool_choice_for_descriptor(
                self.tool_choice),
            requires_tool_call=self._requires_tool_call()
            or named_tool_choice is not None,
            named_tool_choice=named_tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            schemas=schemas,
            strict_tool_names=tuple(strict_tool_names),
        )

    def parse_non_stream(self, text: str) -> ToolCallParseResult:
        if not self.has_tools:
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
        if not self.has_tools:
            return ToolCallParseResult(content=delta_text)

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
            diagnostics.append(
                ToolCallDiagnostic(
                    code="tool_choice_violation",
                    message="tool_choice='required' was set but no valid tool call was produced",
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
        tool_calls: List[Any] = []
        for raw_index in sorted(self.buffered_stream_tool_calls):
            tool_calls.extend(self.buffered_stream_tool_calls[raw_index])
        self.buffered_stream_tool_calls.clear()
        return ToolCallParseResult(
            tools_called=bool(tool_calls),
            valid_tool_calls=tool_calls,
        )

    def _finalize_stream_diagnostics(self) -> List[ToolCallDiagnostic]:
        if self._stream_finalized:
            return []
        self._stream_finalized = True
        diagnostics: List[ToolCallDiagnostic] = []
        if self._requires_tool_call() and not self.has_valid_streamed_tool_calls:
            diagnostics.append(
                ToolCallDiagnostic(
                    code="tool_choice_violation",
                    message="tool_choice='required' was set but no valid stream tool call was produced",
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
        return self.tool_choice == "required"

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
            parsed_arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError) as exc:
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
        diagnostics: List[ToolCallDiagnostic] = []
        for argument_name in _required_arguments(parameters):
            if argument_name not in parsed_arguments:
                diagnostics.append(
                    ToolCallDiagnostic(
                        code="missing_required_argument",
                        message=(
                            f"required argument {argument_name!r} is missing"
                        ),
                        tool_name=name,
                        index=index,
                        argument_name=argument_name,
                    ))

        properties = _get_value(parameters, "properties") or {}
        for argument_name, value in parsed_arguments.items():
            property_schema = _get_value(properties, argument_name)
            expected_type = _get_value(property_schema, "type")
            if expected_type is None:
                continue
            if _matches_json_schema_type(value, expected_type):
                continue
            diagnostics.append(
                ToolCallDiagnostic(
                    code="invalid_argument_type",
                    message=(
                        f"argument {argument_name!r} expected type "
                        f"{expected_type!r} but got {_json_type_name(value)!r}"
                    ),
                    tool_name=name,
                    index=index,
                    argument_name=argument_name,
                ))
        return diagnostics

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


def _get_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _required_arguments(parameters: Any) -> List[str]:
    required = _get_value(parameters, "required") or []
    if not isinstance(required, list):
        return []
    return [name for name in required if isinstance(name, str)]


def _matches_json_schema_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_json_schema_type(value, item)
                   for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, int) or isinstance(value, float)
        ) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


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
