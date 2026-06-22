from dataclasses import dataclass, field
from typing import Any, FrozenSet, Iterable, List, Mapping, Optional, Tuple


@dataclass
class ToolCallAdapterContext:
    """Request-local toolcall adapter state.

    Milestone 1 intentionally keeps this as a no-behavior-change context. Later
    milestones can use the same surface to add explicit aliases and diagnostics
    without spreading toolcall policy across parser and server code.
    """

    external_tools: Tuple[Any, ...] = ()
    model_tools: Tuple[Any, ...] = ()
    tool_choice: Any = None
    alias_map: Mapping[str, str] = field(default_factory=dict)
    allowed_external_tool_names: FrozenSet[str] = field(default_factory=frozenset)
    allowed_model_tool_names: FrozenSet[str] = field(default_factory=frozenset)
    strict_mode: bool = False
    compat_mode: bool = False
    model_type: Optional[str] = None

    @property
    def has_tools(self) -> bool:
        return bool(self.external_tools)


@dataclass(frozen=True)
class ToolCallValidationError:
    code: str
    message: str
    tool_name: Optional[str] = None
    index: Optional[int] = None


@dataclass(frozen=True)
class ToolCallValidationResult:
    valid: bool
    errors: Tuple[ToolCallValidationError, ...] = ()
    tool_names: Tuple[str, ...] = ()


def build_toolcall_context(
    request: Any,
    model_type: Optional[str] = None,
    alias_map: Optional[Mapping[str, str]] = None,
    strict_mode: bool = False,
    compat_mode: bool = False,
) -> ToolCallAdapterContext:
    tools = tuple(_get_value(request, "tools") or ())
    aliases = dict(alias_map or {})
    external_tool_names = tuple(_iter_tool_names(tools))
    model_tool_names = tuple(
        aliases.get(name, name) for name in external_tool_names)

    return ToolCallAdapterContext(
        external_tools=tools,
        model_tools=tools,
        tool_choice=_get_value(request, "tool_choice"),
        alias_map=aliases,
        allowed_external_tool_names=frozenset(external_tool_names),
        allowed_model_tool_names=frozenset(model_tool_names),
        strict_mode=strict_mode,
        compat_mode=compat_mode,
        model_type=model_type,
    )


def prepare_request_for_model(
    request: Any,
    context: ToolCallAdapterContext,
) -> Any:
    """Return the request that should be sent to the model.

    This is a deliberate no-op in Milestone 1. It gives later alias/compat work
    a single integration point while proving the default path is unchanged.
    """

    return request


def map_tool_calls_to_external(
    tool_calls: Iterable[Any],
    context: ToolCallAdapterContext,
) -> List[Any]:
    """Return externally visible tool calls.

    This is also a no-op in Milestone 1. Returning a list makes callers safe to
    iterate more than once without mutating the parser-owned collection.
    """

    return list(tool_calls or [])


def validate_tool_calls(
    tool_calls: Iterable[Any],
    context: ToolCallAdapterContext,
) -> ToolCallValidationResult:
    names: List[str] = []
    errors: List[ToolCallValidationError] = []

    allowed_names = (
        context.allowed_model_tool_names or context.allowed_external_tool_names)
    for index, tool_call in enumerate(tool_calls or []):
        name = _tool_call_name(tool_call)
        if name:
            names.append(name)
        if context.strict_mode and name and allowed_names and name not in allowed_names:
            errors.append(
                ToolCallValidationError(
                    code="invalid_tool_name",
                    message=f"tool name {name!r} is not in allowed tools",
                    tool_name=name,
                    index=index,
                ))

    return ToolCallValidationResult(
        valid=not errors,
        errors=tuple(errors),
        tool_names=tuple(names),
    )


def named_tool_choice_name(tool_choice: Any) -> Optional[str]:
    if isinstance(tool_choice, str) or tool_choice is None:
        return None
    function = _get_value(tool_choice, "function")
    if function is None:
        return None
    return _get_value(function, "name")


def _iter_tool_names(tools: Iterable[Any]) -> Iterable[str]:
    for tool in tools or []:
        function = _get_value(tool, "function")
        name = _get_value(function, "name") if function is not None else None
        if name:
            yield name


def _tool_call_name(tool_call: Any) -> Optional[str]:
    function = _get_value(tool_call, "function")
    if function is None:
        return None
    return _get_value(function, "name")


def _get_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
