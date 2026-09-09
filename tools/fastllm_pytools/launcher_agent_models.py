"""Discover reasoning capabilities from the active model, including model aliases."""

import json
import urllib.request


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def with_model_metadata(service, api_key):
    service = dict(service)
    request = urllib.request.Request(service["endpoint"].rstrip("/") + "/v1/models",
        headers={"Authorization": "Bearer " + (api_key or "fastllm-local")})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read(1024 * 1024))
        entries = payload.get("data") or payload.get("models") or []
        for entry in entries:
            if isinstance(entry, dict) and (entry.get("id") or entry.get("slug")) == service["modelName"]:
                service["modelMetadata"] = entry
                break
    except (OSError, ValueError, AttributeError, TypeError):
        # Older or external model servers may omit capability metadata. Opening
        # an agent must still work, without advertising invented effort levels.
        pass
    return service


def reasoning_options(service):
    metadata = service.get("modelMetadata") or {}
    raw = metadata.get("supported_reasoning_efforts", metadata.get("supportedReasoningEfforts",
        metadata.get("supported_reasoning_levels", [])))
    offered = set()
    for value in raw if isinstance(raw, list) else []:
        if isinstance(value, dict):
            value = value.get("reasoningEffort", value.get("effort"))
        if isinstance(value, str) and value in REASONING_EFFORTS:
            offered.add(value)
    efforts = [value for value in REASONING_EFFORTS if value in offered]
    default = metadata.get("default_reasoning_effort", metadata.get("defaultReasoningEffort",
        metadata.get("default_reasoning_level")))
    if default not in efforts:
        default = efforts[-1] if efforts else None
    return efforts, default
