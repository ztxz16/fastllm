# SPDX-License-Identifier: Apache-2.0

import json
import math
from typing import Any, Optional


_TYPE_ALIASES = {
    "arr": "array",
    "binary": "boolean",
    "bool": "boolean",
    "char": "string",
    "decimal": "number",
    "double": "number",
    "enum": "string",
    "str": "string",
    "text": "string",
    "varchar": "string",
}


def get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def json_type_name(value: Any) -> str:
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


def _normalize_type(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    for prefix, canonical in (
            ("unsigned", "integer"),
            ("uint", "integer"),
            ("int", "integer"),
            ("long", "integer"),
            ("short", "integer"),
            ("float", "number"),
            ("num", "number"),
            ("dict", "object"),
            ("list", "array")):
        if normalized.startswith(prefix):
            return canonical
    return _TYPE_ALIASES.get(normalized, normalized)


def _resolve_json_pointer(root_schema: Any, reference: str) -> Any:
    if not reference.startswith("#/"):
        return None
    current = root_schema
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def dereference_schema(
    schema: Any,
    root_schema: Any,
    *,
    max_depth: int = 16,
) -> Any:
    """Resolve local JSON Schema references while preserving sibling keys."""
    current = schema
    visited: set[str] = set()
    for _ in range(max_depth):
        if not isinstance(current, dict):
            return current
        reference = current.get("$ref")
        if not isinstance(reference, str) or reference in visited:
            return current
        resolved = _resolve_json_pointer(root_schema, reference)
        if not isinstance(resolved, dict):
            return current
        visited.add(reference)
        merged = dict(resolved)
        merged.update({key: value for key, value in current.items()
                       if key != "$ref"})
        current = merged
    return current


def schema_types(
    schema: Any,
    root_schema: Any = None,
    *,
    max_depth: int = 16,
) -> tuple[str, ...]:
    """Return JSON types allowed by a schema in stable preference order."""
    if root_schema is None:
        root_schema = schema

    result: list[str] = []

    def append_type(value: Any) -> None:
        normalized = _normalize_type(value)
        if normalized and normalized not in result:
            result.append(normalized)

    def visit(candidate: Any, depth: int) -> None:
        if depth > max_depth:
            return
        candidate = dereference_schema(candidate, root_schema,
                                       max_depth=max_depth)
        if not isinstance(candidate, dict):
            return

        declared = candidate.get("type")
        if isinstance(declared, list):
            for item in declared:
                append_type(item)
        elif declared is not None:
            append_type(declared)

        if candidate.get("nullable") is True:
            append_type("null")

        for keyword in ("anyOf", "oneOf", "allOf"):
            alternatives = candidate.get(keyword)
            if isinstance(alternatives, list):
                for alternative in alternatives:
                    visit(alternative, depth + 1)

        if declared is None:
            if isinstance(candidate.get("properties"), dict):
                append_type("object")
            elif "items" in candidate:
                append_type("array")
            elif isinstance(candidate.get("enum"), list):
                for value in candidate["enum"]:
                    append_type(json_type_name(value))
            elif "const" in candidate:
                append_type(json_type_name(candidate["const"]))

    visit(schema, 0)
    return tuple(result)


def load_strict_json(value: str) -> Any:
    """Load standards-compliant JSON and reject non-finite numbers."""

    def reject_constant(constant: str) -> None:
        raise ValueError(
            f"non-finite JSON constant {constant!r} is not allowed")

    def parse_finite_float(number: str) -> float:
        parsed = float(number)
        if not math.isfinite(parsed):
            raise ValueError(
                f"non-finite JSON number {number!r} is not allowed")
        return parsed

    return json.loads(
        value,
        parse_constant=reject_constant,
        parse_float=parse_finite_float,
    )


def _convert_as_type(value: str, target_type: str) -> tuple[bool, Any]:
    if target_type == "string":
        return True, value
    if target_type == "integer":
        try:
            return True, int(value)
        except (TypeError, ValueError):
            return False, value
    if target_type == "number":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False, value
        if not math.isfinite(number):
            return False, value
        return True, int(number) if number.is_integer() else number
    if target_type == "boolean":
        lowered = value.strip().lower()
        if lowered == "true":
            return True, True
        if lowered == "false":
            return True, False
        return False, value
    if target_type in {"array", "object"}:
        try:
            parsed = load_strict_json(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False, value
        expected_class = list if target_type == "array" else dict
        return ((True, parsed) if isinstance(parsed, expected_class)
                else (False, value))
    return False, value


def convert_text_value(
    value: str,
    schema: Any,
    root_schema: Any = None,
) -> Any:
    """Convert an XML parameter string according to its JSON Schema type.

    Failed conversions intentionally preserve the original string.  The
    validation layer can then reject it for strict tools without silently
    changing the model's requested argument.
    """
    types = schema_types(schema, root_schema)
    if not types:
        return value

    if value.strip().lower() == "null" and "null" in types:
        return None

    for target_type in types:
        if target_type == "null":
            continue
        converted, result = _convert_as_type(value, target_type)
        if converted:
            return result
    return value
