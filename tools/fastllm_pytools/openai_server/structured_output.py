"""Model-independent JSON output guidance and final-response validation.

This is prompt-guided generation, not token-level constrained decoding. Stream
deltas are provisional; an invalid final answer ends in an error, never a
successful completion. Tool calls and token-limit truncation remain distinct
from a completed JSON answer.
"""

import copy
import json


def responses_response_format(text):
    if text is None or "format" not in text:
        return None
    format_spec = text["format"]
    if not isinstance(format_spec, dict):
        raise ValueError("text.format must be an object")
    if format_spec.get("type") == "json_schema":
        return {"type": "json_schema", "json_schema": {
            key: value for key, value in format_spec.items() if key != "type"
        }}
    return copy.deepcopy(format_spec)


def _schema(format_spec):
    if format_spec is None:
        return None
    kind = format_spec.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return {"type": "object"}
    if kind != "json_schema":
        raise ValueError("response format type must be text, json_object or json_schema")
    options = format_spec.get("json_schema")
    if not isinstance(options, dict) or not isinstance(options.get("schema"), (dict, bool)):
        raise ValueError("json_schema.schema must be a JSON Schema object or boolean")
    if options.get("strict") is not None and not isinstance(options["strict"], bool):
        raise ValueError("json_schema.strict must be a boolean")
    return options["schema"]


def _validator(schema):
    # Lazy import: ordinary inference does not need the JSON Schema library.
    from jsonschema import validators
    from referencing import Registry
    from referencing.exceptions import NoSuchResource

    def no_remote_references(uri):
        raise NoSuchResource(ref=uri)

    validator_type = validators.validator_for(schema)
    validator_type.check_schema(schema)
    # Schemas must be self-contained. Never fetch a client-supplied $ref URL.
    return validator_type(schema, registry=Registry(retrieve=no_remote_references))


def prepare_structured_output(messages, format_spec):
    schema = _schema(format_spec)
    if schema is None:
        return messages
    try:
        _validator(schema)
    except Exception as error:
        raise ValueError(f"Invalid response format schema: {error}") from error
    instruction = (
        "Return raw JSON only. Your final answer will be parsed directly with "
        "JSON.parse(). Never use backticks. "
        "Your final answer must be a single valid JSON value matching the JSON "
        "Schema below. Do not include markdown fences or any text outside the JSON. "
        "Tool calls may still be used when necessary; tool availability does not "
        "require tool use. If the supplied information suffices, answer directly.\n"
        "JSON Schema:\n" + json.dumps(schema, ensure_ascii=False, allow_nan=False)
    )
    if isinstance(schema, dict) and schema.get("type") in ("object", "array"):
        opening, closing = ("{", "}") if schema["type"] == "object" else ("[", "]")
        instruction += f"\nStart your final answer with {opening} and end it with {closing}."
    messages = copy.deepcopy(messages)
    if messages and messages[0].get("role") in ("system", "developer"):
        content = messages[0].get("content")
        if isinstance(content, list):
            content.append({"type": "text", "text": instruction})
        else:
            messages[0]["content"] = (content + "\n\n" if content else "") + instruction
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages


def validate_structured_output(content, format_spec):
    schema = _schema(format_spec)
    if schema is None:
        return

    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    try:
        value = json.loads(content, parse_constant=reject_constant)
        _validator(schema).validate(value)
    except Exception as error:
        raise ValueError(f"Model output does not match the requested response format: {error}") from error
