import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ToolCallConstraintSpec:
    """Backend-neutral prototype for future toolcall constrained decoding."""

    version: int
    backend: str
    descriptor: Dict[str, Any]
    structural_tag: Optional[Dict[str, Any]] = None
    name_grammar: Optional[str] = None
    json_schemas: Dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "backend": self.backend,
            "descriptor": copy.deepcopy(self.descriptor),
            "structural_tag": copy.deepcopy(self.structural_tag),
            "name_grammar": self.name_grammar,
            "json_schemas": copy.deepcopy(self.json_schemas),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ConstraintApplyResult:
    applied: bool
    mode: Optional[str]
    spec: Optional[ToolCallConstraintSpec]
    message: str = ""


def compile_tool_call_constraint(
    descriptor: Any,
) -> Optional[ToolCallConstraintSpec]:
    descriptor_dict = _descriptor_to_dict(descriptor)
    if descriptor_dict is None:
        return None

    constraint_type = descriptor_dict.get("constraint_type")
    structural_tag = None
    name_grammar = None
    notes = [
        "prototype only: native FastLLM decoding does not consume this yet",
        "no xgrammar dependency is required to build this spec",
    ]
    if constraint_type == "deepseek_v4_dsml":
        structural_tag = _build_deepseek_v4_structural_tag(descriptor_dict)
        name_grammar = _build_deepseek_v4_name_grammar(descriptor_dict)
    else:
        notes.append(
            f"no structural_tag prototype for constraint_type={constraint_type!r}")

    return ToolCallConstraintSpec(
        version=1,
        backend="fastllm_toolcall_prototype",
        descriptor=descriptor_dict,
        structural_tag=structural_tag,
        name_grammar=name_grammar,
        json_schemas=copy.deepcopy(descriptor_dict.get("schemas") or {}),
        notes=tuple(notes),
    )


def apply_tool_call_constraint_to_decoder(
    decoder: Any,
    descriptor_or_spec: Any,
) -> ConstraintApplyResult:
    spec = (
        descriptor_or_spec
        if isinstance(descriptor_or_spec, ToolCallConstraintSpec)
        else compile_tool_call_constraint(descriptor_or_spec)
    )
    if spec is None:
        return ConstraintApplyResult(
            applied=False,
            mode=None,
            spec=None,
            message="no toolcall constraint descriptor",
        )

    setter = getattr(decoder, "set_tool_call_constraint", None)
    if callable(setter):
        setter(spec.to_dict())
        return ConstraintApplyResult(
            applied=True,
            mode="tool_call_constraint",
            spec=spec,
        )

    if spec.structural_tag is not None:
        structural_setter = getattr(decoder, "set_structural_tag", None)
        if callable(structural_setter):
            structural_setter(copy.deepcopy(spec.structural_tag))
            return ConstraintApplyResult(
                applied=True,
                mode="structural_tag",
                spec=spec,
            )

        structured_outputs_setter = getattr(
            decoder, "set_structured_outputs", None)
        if callable(structured_outputs_setter):
            structured_outputs_setter({
                "structural_tag": json.dumps(
                    spec.structural_tag, ensure_ascii=False),
            })
            return ConstraintApplyResult(
                applied=True,
                mode="structured_outputs",
                spec=spec,
            )

    logging.debug(
        "Decoder does not support toolcall constraint application; "
        "continuing without backend constraint.")
    return ConstraintApplyResult(
        applied=False,
        mode=None,
        spec=spec,
        message="decoder does not expose a supported constraint API",
    )


def _descriptor_to_dict(descriptor: Any) -> Optional[Dict[str, Any]]:
    if descriptor is None:
        return None
    if hasattr(descriptor, "to_dict"):
        return descriptor.to_dict()
    if isinstance(descriptor, dict):
        return copy.deepcopy(descriptor)
    raise TypeError(
        "toolcall constraint descriptor must be a dict or expose to_dict()")


def _build_deepseek_v4_structural_tag(
    descriptor: Dict[str, Any],
) -> Dict[str, Any]:
    allowed_tool_names = list(descriptor.get("allowed_tool_names") or [])
    return {
        "type": "structural_tag",
        "format": "deepseek_v4_dsml",
        "tool_call_start": "<｜DSML｜tool_calls>",
        "tool_call_end": "</｜DSML｜tool_calls>",
        "alternate_tool_call_start": "<\\DSML\\tool_calls>",
        "alternate_tool_call_end": "</\\DSML\\tool_calls>",
        "invoke": {
            "tag": "invoke",
            "name_attribute": "name",
            "allowed_names": allowed_tool_names,
        },
        "parameter": {
            "tag": "parameter",
            "required_attributes": ["name", "string"],
        },
        "requires_tool_call": bool(descriptor.get("requires_tool_call")),
        "max_tool_calls": (
            1 if descriptor.get("parallel_tool_calls") is False else None
        ),
        "schemas": copy.deepcopy(descriptor.get("schemas") or {}),
    }


def _build_deepseek_v4_name_grammar(descriptor: Dict[str, Any]) -> str:
    allowed_tool_names = list(descriptor.get("allowed_tool_names") or [])
    tool_name_rule = " | ".join(
        _quote_ebnf_literal(name) for name in allowed_tool_names)
    if not tool_name_rule:
        tool_name_rule = '""'
    return "\n".join([
        'root ::= tool_calls',
        'tool_calls ::= "<｜DSML｜tool_calls>" invoke+ "</｜DSML｜tool_calls>"',
        'invoke ::= "<｜DSML｜invoke name=\\"" tool_name "\\">" '
        'parameter* "</｜DSML｜invoke>"',
        f"tool_name ::= {tool_name_rule}",
        'parameter ::= "<｜DSML｜parameter" parameter_attrs ">" '
        'parameter_value "</｜DSML｜parameter>"',
        'parameter_attrs ::= /[^>]*/',
        'parameter_value ::= /[^<]*/',
    ])


def _quote_ebnf_literal(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)
