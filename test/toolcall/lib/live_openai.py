import copy
import json
import socket
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CASES_DIR = Path(__file__).resolve().parents[1] / "cases"
LIVE_CASES_PATH = CASES_DIR / "live_openai.jsonl"


class LiveOpenAIError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def load_jsonl(path: Path = LIVE_CASES_PATH) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            case = json.loads(line)
            case["_path"] = str(path)
            case["_lineno"] = lineno
            yield case


def load_live_cases() -> List[Dict[str, Any]]:
    return list(load_jsonl())


def chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def summarize_text(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, separators=(",", ":"))
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def case_prompt_summary(case: Dict[str, Any]) -> str:
    messages = case.get("request", {}).get("messages", [])
    parts = []
    for message in messages:
        parts.append(
            f"{message.get('role')}:{summarize_text(message.get('content'), 80)}"
        )
    return " | ".join(parts)


def request_tool_names(request_payload: Dict[str, Any]) -> List[str]:
    names = []
    for tool in request_payload.get("tools") or []:
        function = tool.get("function") or {}
        if function.get("name"):
            names.append(function["name"])
    return names


def prepare_payload(
    case: Dict[str, Any],
    model: str,
    stream: bool,
    temperature_override: Optional[float] = None,
) -> Dict[str, Any]:
    payload = copy.deepcopy(case["request"])
    payload["model"] = model
    payload["stream"] = stream
    if temperature_override is not None:
        payload["temperature"] = temperature_override
    return payload


def post_json(
    base_url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        chat_completions_url(base_url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise LiveOpenAIError(
            "http_error",
            f"HTTP {exc.code} from {chat_completions_url(base_url)}: "
            f"{summarize_text(raw)}",
        ) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise LiveOpenAIError("http_error", f"request failed: {exc}") from exc

    try:
        return json.loads(raw)
    except Exception as exc:
        raise LiveOpenAIError(
            "invalid_json_response",
            f"response is not JSON: {summarize_text(raw)}",
        ) from exc


def post_sse(
    base_url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        chat_completions_url(base_url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: List[Dict[str, Any]] = []
    saw_done = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    saw_done = True
                    continue
                try:
                    chunks.append(json.loads(data))
                except Exception as exc:
                    raise LiveOpenAIError(
                        "invalid_json_response",
                        f"SSE data is not JSON: {summarize_text(data)}",
                    ) from exc
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise LiveOpenAIError(
            "http_error",
            f"HTTP {exc.code} from {chat_completions_url(base_url)}: "
            f"{summarize_text(raw)}",
        ) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise LiveOpenAIError("http_error", f"stream request failed: {exc}") from exc
    return chunks, saw_done


def _choice_from_response(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices")
    if not choices:
        raise LiveOpenAIError("invalid_json_response",
                              "response has no choices")
    return choices[0]


def _tool_call_from_message(index: int, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    function = tool_call.get("function") or {}
    return {
        "index": index,
        "id": tool_call.get("id"),
        "type": tool_call.get("type"),
        "name": function.get("name"),
        "arguments": function.get("arguments") or "",
    }


def extract_non_stream_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    choice = _choice_from_response(response)
    message = choice.get("message") or {}
    return [
        _tool_call_from_message(index, tool_call)
        for index, tool_call in enumerate(message.get("tool_calls") or [])
    ]


def reconstruct_stream_tool_calls(
    chunks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    state: Dict[int, Dict[str, Any]] = {}
    finish_reason = None
    content_fragments: List[str] = []
    names_observed: Dict[int, List[str]] = defaultdict(list)
    argument_fragment_counts: Dict[int, int] = defaultdict(int)

    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            finish_reason = choice.get("finish_reason")
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content_fragments.append(delta["content"])
        for tool_delta in delta.get("tool_calls") or []:
            index = tool_delta.get("index")
            if not isinstance(index, int):
                raise LiveOpenAIError(
                    "stream_tool_call_reconstruction_failed",
                    f"stream tool_call has invalid index: {index!r}",
                )
            item = state.setdefault(index, {
                "index": index,
                "id": None,
                "type": None,
                "name": None,
                "arguments": "",
            })
            if tool_delta.get("id"):
                if item["id"] not in (None, tool_delta["id"]):
                    raise LiveOpenAIError(
                        "stream_tool_call_reconstruction_failed",
                        f"tool_call index {index} id changed from "
                        f"{item['id']!r} to {tool_delta['id']!r}",
                    )
                item["id"] = tool_delta["id"]
            if tool_delta.get("type"):
                item["type"] = tool_delta["type"]
            function = tool_delta.get("function") or {}
            if function.get("name"):
                if item["name"] not in (None, function["name"]):
                    raise LiveOpenAIError(
                        "stream_tool_call_reconstruction_failed",
                        f"tool_call index {index} name changed from "
                        f"{item['name']!r} to {function['name']!r}",
                    )
                item["name"] = function["name"]
                names_observed[index].append(function["name"])
            if function.get("arguments"):
                item["arguments"] += function["arguments"]
                argument_fragment_counts[index] += 1

    tool_calls = [state[index] for index in sorted(state)]
    trace = {
        "finish_reason": finish_reason,
        "content": "".join(content_fragments),
        "names_observed": {
            str(index): values for index, values in names_observed.items()
        },
        "argument_fragment_counts": {
            str(index): count
            for index, count in argument_fragment_counts.items()
        },
    }
    return tool_calls, trace


def _parse_arguments(arguments: str, case_id: str, index: int) -> Any:
    try:
        return json.loads(arguments)
    except Exception as exc:
        raise LiveOpenAIError(
            "invalid_arguments_json",
            f"{case_id}: tool_call {index} arguments are not JSON: "
            f"{summarize_text(arguments)}",
        ) from exc


def validate_tool_calls(
    case: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    case_id = case["id"]
    expected = case.get("expected", {})
    allowed_names = expected.get("allowed_tool_names") or []
    required_keys = expected.get("required_argument_keys") or []

    if expected.get("must_call_tool") and not tool_calls:
        raise LiveOpenAIError("no_tool_call", f"{case_id}: no tool call returned")
    if expected.get("forbid_tool_call") and tool_calls:
        raise LiveOpenAIError(
            "unexpected_tool_call",
            f"{case_id}: unexpected tool calls returned: "
            f"{[call.get('name') for call in tool_calls]}",
        )

    parsed_arguments = []
    for index, tool_call in enumerate(tool_calls):
        name = tool_call.get("name")
        if allowed_names and name not in allowed_names:
            raise LiveOpenAIError(
                "invalid_tool_name",
                f"{case_id}: tool_call {index} name {name!r} not in "
                f"{allowed_names!r}",
                {
                    "invalid_tool_name": name,
                    "tool_names": [
                        call.get("name") for call in tool_calls
                    ],
                    "allowed_tool_names": allowed_names,
                },
            )
        if expected.get("arguments_must_be_json"):
            arguments = _parse_arguments(tool_call.get("arguments") or "",
                                         case_id, index)
            parsed_arguments.append(arguments)
            for key in required_keys:
                if not isinstance(arguments, dict) or key not in arguments:
                    raise LiveOpenAIError(
                        "missing_required_argument",
                        f"{case_id}: tool_call {index} missing required "
                        f"argument {key!r}: {arguments!r}",
                    )
    return {"parsed_arguments": parsed_arguments}


def validate_content(case: Dict[str, Any], content: Optional[str]):
    expected = case.get("expected", {})
    if expected.get("final_content_non_empty") and not content:
        raise LiveOpenAIError(
            "empty_content",
            f"{case['id']}: expected non-empty assistant content",
        )


def _openai_tool_call_shape(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": tool_call.get("id") or f"call_{tool_call.get('index', 0)}",
        "type": "function",
        "function": {
            "name": tool_call.get("name"),
            "arguments": tool_call.get("arguments") or "{}",
        },
    }


def run_non_stream_case(
    case: Dict[str, Any],
    base_url: str,
    model: str,
    timeout: float,
    temperature_override: Optional[float] = None,
    validate_response_content: bool = True,
) -> Dict[str, Any]:
    payload = prepare_payload(case, model, stream=False,
                              temperature_override=temperature_override)
    response = post_json(base_url, payload, timeout)
    choice = _choice_from_response(response)
    message = choice.get("message") or {}
    tool_calls = extract_non_stream_tool_calls(response)
    validate_tool_calls(case, tool_calls)
    if validate_response_content:
        validate_content(case, message.get("content"))
    expected_finish_reason = case.get("expected", {}).get("finish_reason")
    finish_reason = choice.get("finish_reason")
    if expected_finish_reason and finish_reason != expected_finish_reason:
        raise LiveOpenAIError(
            "stream_missing_finish_reason",
            f"{case['id']}: finish_reason {finish_reason!r} != "
            f"{expected_finish_reason!r}",
        )
    return {
        "case_id": case["id"],
        "mode": "non_stream",
        "passed": True,
        "request": payload,
        "raw_response": response,
        "finish_reason": finish_reason,
        "content": message.get("content"),
        "tool_calls": tool_calls,
        "tool_names": [call.get("name") for call in tool_calls],
    }


def run_stream_case(
    case: Dict[str, Any],
    base_url: str,
    model: str,
    timeout: float,
    temperature_override: Optional[float] = None,
) -> Dict[str, Any]:
    payload = prepare_payload(case, model, stream=True,
                              temperature_override=temperature_override)
    chunks, saw_done = post_sse(base_url, payload, timeout)
    tool_calls, trace = reconstruct_stream_tool_calls(chunks)
    validate_tool_calls(case, tool_calls)
    validate_content(case, trace.get("content"))
    expected = case.get("expected", {})
    if expected.get("must_see_done", True) and not saw_done:
        raise LiveOpenAIError(
            "stream_missing_done",
            f"{case['id']}: stream did not emit [DONE]",
        )
    expected_finish_reason = expected.get("finish_reason")
    if expected_finish_reason and trace.get("finish_reason") != expected_finish_reason:
        raise LiveOpenAIError(
            "stream_missing_finish_reason",
            f"{case['id']}: finish_reason {trace.get('finish_reason')!r} != "
            f"{expected_finish_reason!r}",
        )
    return {
        "case_id": case["id"],
        "mode": "stream",
        "passed": True,
        "request": payload,
        "stream_chunks": chunks,
        "saw_done": saw_done,
        "finish_reason": trace.get("finish_reason"),
        "content": trace.get("content"),
        "tool_calls": tool_calls,
        "tool_names": [call.get("name") for call in tool_calls],
        "argument_fragment_counts": [
            trace["argument_fragment_counts"].get(str(call["index"]), 0)
            for call in tool_calls
        ],
        "trace": trace,
    }


def run_roundtrip_case(
    case: Dict[str, Any],
    base_url: str,
    model: str,
    timeout: float,
    temperature_override: Optional[float] = None,
) -> Dict[str, Any]:
    first = run_non_stream_case(case, base_url, model, timeout,
                                temperature_override=temperature_override,
                                validate_response_content=False)
    if not first["tool_calls"]:
        raise LiveOpenAIError("no_tool_call",
                              f"{case['id']}: first turn had no tool call")

    first_request = first["request"]
    first_tool_call = first["tool_calls"][0]
    tool_result = case.get("tool_result", {})
    follow_up = tool_result.get("follow_up_user_message",
                                "请根据工具结果回答。")
    messages = copy.deepcopy(first_request.get("messages") or [])
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            _openai_tool_call_shape(call) for call in first["tool_calls"]
        ],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": first_tool_call.get("id"),
        "content": tool_result.get("content", {}),
    })
    messages.append({"role": "user", "content": follow_up})

    second_payload = copy.deepcopy(first_request)
    second_payload["messages"] = messages
    second_payload["stream"] = False
    second_payload["tool_choice"] = "auto"
    second_response = post_json(base_url, second_payload, timeout)
    second_choice = _choice_from_response(second_response)
    second_message = second_choice.get("message") or {}
    second_tool_calls = extract_non_stream_tool_calls(second_response)
    if second_tool_calls:
        validate_tool_calls(case, second_tool_calls)
    validate_content(case, second_message.get("content"))

    return {
        "case_id": case["id"],
        "mode": "roundtrip",
        "passed": True,
        "request": first_request,
        "raw_response": first["raw_response"],
        "tool_calls": first["tool_calls"],
        "tool_names": [call.get("name") for call in first["tool_calls"]],
        "finish_reason": second_choice.get("finish_reason"),
        "second_request": second_payload,
        "second_raw_response": second_response,
        "second_content": second_message.get("content"),
        "second_tool_calls": second_tool_calls,
    }


def run_live_openai_case(
    case: Dict[str, Any],
    base_url: str,
    model: str,
    timeout: float,
    temperature_override: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        mode = case.get("mode")
        if mode == "non_stream":
            return run_non_stream_case(case, base_url, model, timeout,
                                       temperature_override)
        if mode == "stream":
            return run_stream_case(case, base_url, model, timeout,
                                   temperature_override)
        if mode == "roundtrip":
            return run_roundtrip_case(case, base_url, model, timeout,
                                      temperature_override)
        raise LiveOpenAIError("unknown_failure",
                              f"{case['id']}: unsupported mode {mode!r}")
    except LiveOpenAIError as exc:
        return {
            "case_id": case.get("id"),
            "mode": case.get("mode"),
            "passed": False,
            "error_code": exc.code,
            "error_message": exc.message,
            "diagnostics": exc.details,
        }
    except Exception as exc:  # pragma: no cover - defensive live diagnostics
        return {
            "case_id": case.get("id"),
            "mode": case.get("mode"),
            "passed": False,
            "error_code": "unknown_failure",
            "error_message": f"unexpected error: {exc}",
        }


def result_summary_line(result: Dict[str, Any]) -> str:
    if not result.get("passed"):
        return (f"FAIL {result.get('case_id')}: "
                f"error_code={result.get('error_code')}: "
                f"{result.get('error_message')}")
    mode = result.get("mode")
    if mode == "stream":
        return (
            f"PASS {result['case_id']}: finish_reason={result.get('finish_reason')}, "
            f"tool_calls={len(result.get('tool_calls') or [])}, "
            f"argument_fragments={result.get('argument_fragment_counts')}, "
            f"chunks={len(result.get('stream_chunks') or [])}, "
            f"done={result.get('saw_done')}, "
            f"tool_names={result.get('tool_names')}"
        )
    if mode == "roundtrip":
        return (
            f"PASS {result['case_id']}: tool_calls="
            f"{len(result.get('tool_calls') or [])}, "
            f"tool_names={result.get('tool_names')}, "
            f"final_content={summarize_text(result.get('second_content'), 120)}"
        )
    return (
        f"PASS {result['case_id']}: finish_reason={result.get('finish_reason')}, "
        f"tool_calls={len(result.get('tool_calls') or [])}, "
        f"tool_names={result.get('tool_names')}"
    )


def verbose_lines(case: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
    request = result.get("request") or case.get("request", {})
    lines = [
        f"  case={case.get('id')}",
        f"  mode={case.get('mode')}",
        f"  tool_choice={request.get('tool_choice')!r}",
        f"  tools={request_tool_names(request)}",
        f"  prompt={case_prompt_summary(case)}",
    ]
    if not result.get("passed"):
        lines.append(f"  error_code={result.get('error_code')}")
        lines.append(f"  error_message={result.get('error_message')}")
        return lines
    lines.append(f"  finish_reason={result.get('finish_reason')}")
    lines.append(f"  tool_calls={len(result.get('tool_calls') or [])}")
    if result.get("mode") == "stream":
        lines.append(f"  stream_chunks={len(result.get('stream_chunks') or [])}")
        lines.append(f"  saw_done={result.get('saw_done')}")
        lines.append(
            f"  argument_fragment_counts={result.get('argument_fragment_counts')}"
        )
    for index, tool_call in enumerate(result.get("tool_calls") or []):
        args = tool_call.get("arguments") or ""
        try:
            parsed = json.loads(args)
            json_ok = True
        except Exception:
            parsed = None
            json_ok = False
        required_keys = case.get("expected", {}).get("required_argument_keys") or []
        keys_ok = (
            not required_keys or
            (isinstance(parsed, dict) and all(key in parsed for key in required_keys))
        )
        lines.append(
            f"  tool_call[{index}]: id={tool_call.get('id')!r} "
            f"name={tool_call.get('name')!r} json_ok={json_ok} "
            f"required_keys_ok={keys_ok} args={summarize_text(args, 160)}"
        )
    return lines
