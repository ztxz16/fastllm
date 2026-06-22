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
    ):
        self.tools = list(tools or [])
        self.tool_choice = tool_choice
        self.tool_parser_name = tool_parser_name
        self.tool_index = self._build_tool_index(self.tools)
        self._request = ChatCompletionRequest(
            model=model,
            messages=[],
            tools=self.tools or None,
            tool_choice=tool_choice,
        )
        if parser is None:
            parser_cls = ToolParserManager.get_tool_parser(tool_parser_name)
            parser = parser_cls(tokenizer or _EmptyToolTokenizer())
        self.parser = parser
        self.invalid_stream_tool_indices: set[int] = set()
        self.valid_stream_tool_indices: set[int] = set()
        self.stream_index_map: Dict[int, int] = {}
        self.stream_diagnostics: List[ToolCallDiagnostic] = []

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
        )

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
                if name not in self.tool_index:
                    self.invalid_stream_tool_indices.add(raw_index)
                    invalid_tool_calls.append(tool_call)
                    diagnostics.append(
                        ToolCallDiagnostic(
                            code="invalid_tool_name",
                            message=f"tool name {name!r} is not in request tools",
                            tool_name=name,
                            index=raw_index,
                        ))
                    continue
                self.valid_stream_tool_indices.add(raw_index)
                self.stream_index_map.setdefault(raw_index,
                                                 len(self.stream_index_map))
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
            if name in self.tool_index:
                valid_tool_calls.append(tool_call)
                continue
            invalid_tool_calls.append(tool_call)
            diagnostics.append(
                ToolCallDiagnostic(
                    code="invalid_tool_name",
                    message=f"tool name {name!r} is not in request tools",
                    tool_name=name,
                    index=index,
                ))

        return ToolCallValidationResult(
            valid=not diagnostics,
            diagnostics=diagnostics,
            valid_tool_calls=valid_tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )

    def _result_from_extracted(
        self,
        extracted: ExtractedToolCallInformation,
        validation: ToolCallValidationResult,
    ) -> ToolCallParseResult:
        return ToolCallParseResult(
            content=extracted.content,
            tools_called=bool(validation.valid_tool_calls),
            valid_tool_calls=validation.valid_tool_calls,
            invalid_tool_calls=validation.invalid_tool_calls,
            diagnostics=validation.diagnostics,
            has_invalid_tool_block=bool(validation.invalid_tool_calls),
        )

    @staticmethod
    def _build_tool_index(tools: Iterable[Any]) -> Dict[str, int]:
        index: Dict[str, int] = {}
        for position, tool in enumerate(tools or []):
            function = _get_value(tool, "function")
            name = _get_value(function, "name") if function is not None else None
            if name:
                index[name] = position
        return index


def _tool_call_name(tool_call: Any) -> Optional[str]:
    function = _get_value(tool_call, "function")
    if function is None:
        return None
    return _get_value(function, "name")


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


def _get_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class _EmptyToolTokenizer:
    def get_vocab(self):
        return {}
