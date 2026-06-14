import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (  # noqa: E402
    ChatCompletionRequest,
)
from tools.fastllm_pytools.openai_server.tool_parsers import ToolParserManager  # noqa: E402


DEFAULT_CHUNK_SIZES = [1, 2, 5, 13]
DSML_LEAK_MARKERS = ("DSML", "tool_calls", "invoke name", "parameter name")


class DummyTokenizer:
    def get_vocab(self):
        return {}


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            case = json.loads(line)
            case["_path"] = str(path)
            case["_lineno"] = lineno
            yield case


def make_request(case: Dict[str, Any]) -> ChatCompletionRequest:
    tool_names = [
        tool_call["name"]
        for tool_call in case.get("expected", {}).get("tool_calls", [])
    ]
    if not tool_names:
        tool_names = ["dummy"]

    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in tool_names
    ]

    return ChatCompletionRequest(
        model="tool-parser-golden",
        messages=[],
        tools=tools,
        tool_choice="auto",
    )


def get_parser(parser_name: str):
    parser_cls = ToolParserManager.get_tool_parser(parser_name)
    return parser_cls(DummyTokenizer())


def _case_label(case: Dict[str, Any]) -> str:
    return f"{case.get('id', '<unknown>')} ({case.get('_path')}:{case.get('_lineno')})"


def normalize_arguments(arguments_str: str) -> Any:
    try:
        return json.loads(arguments_str)
    except Exception as exc:
        raise AssertionError(
            f"function.arguments is not valid JSON: {arguments_str!r}"
        ) from exc


def assert_tool_calls_equal(testcase, actual_tool_calls, expected_tool_calls,
                            case: Dict[str, Any]):
    label = _case_label(case)
    testcase.assertEqual(
        len(actual_tool_calls), len(expected_tool_calls),
        f"{label}: tool call count mismatch")
    for index, (actual_call, expected_call) in enumerate(
            zip(actual_tool_calls, expected_tool_calls)):
        testcase.assertIsNotNone(actual_call.function,
                                 f"{label}: tool call {index} has no function")
        testcase.assertEqual(
            actual_call.function.name,
            expected_call["name"],
            f"{label}: tool call {index} name mismatch",
        )
        testcase.assertEqual(
            normalize_arguments(actual_call.function.arguments),
            expected_call["arguments"],
            f"{label}: tool call {index} arguments mismatch",
        )


def run_non_stream_case(testcase, parser, case: Dict[str, Any]):
    request = make_request(case)
    actual = parser.extract_tool_calls(case["raw_output"], request)
    expected = case["expected"]
    label = _case_label(case)

    testcase.assertEqual(actual.tools_called, expected["tools_called"],
                         f"{label}: tools_called mismatch")
    testcase.assertEqual(actual.content, expected["content"],
                         f"{label}: content mismatch")
    assert_tool_calls_equal(testcase, actual.tool_calls,
                            expected["tool_calls"], case)


def split_text_by_chunk_size(text: str, chunk_size: int) -> Iterable[str]:
    for index in range(0, len(text), chunk_size):
        yield text[index:index + chunk_size]


def _token_ids_for(parser, chunk: str, fallback_start: int) -> List[int]:
    get_token_ids = getattr(parser, "get_token_ids", None)
    if get_token_ids is None:
        return list(range(fallback_start, fallback_start + len(chunk)))
    token_ids = get_token_ids(chunk)
    if token_ids is None:
        return []
    return list(token_ids)


def _assert_no_dsml_leak(testcase, content: str):
    for marker in DSML_LEAK_MARKERS:
        testcase.assertNotIn(marker, content)


def run_stream_case(testcase, parser, case: Dict[str, Any], chunk_size: int):
    request = make_request(case)
    expected = case["expected"]
    stream_config = case.get("stream", {})

    previous_text = ""
    current_text = ""
    previous_token_ids: List[int] = []
    current_token_ids: List[int] = []
    content_parts: List[str] = []
    tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
    next_token_id = 0

    for chunk in split_text_by_chunk_size(case["raw_output"], chunk_size):
        delta_token_ids = _token_ids_for(parser, chunk, next_token_id)
        next_token_id += max(len(delta_token_ids), len(chunk), 1)
        current_text += chunk
        current_token_ids += delta_token_ids

        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=request,
        )

        previous_text = current_text
        previous_token_ids = list(current_token_ids)

        if delta is None:
            continue
        if delta.content:
            content_parts.append(delta.content)
        for tool_delta in delta.tool_calls:
            state = tool_calls_by_index.setdefault(
                tool_delta.index, {"name": None, "arguments": ""})
            if tool_delta.function is None:
                continue
            if tool_delta.function.name:
                state["name"] = tool_delta.function.name
            if tool_delta.function.arguments:
                state["arguments"] += tool_delta.function.arguments

    collected_content = "".join(content_parts)
    label = _case_label(case)
    if stream_config.get("assert_no_dsml_leak", False):
        _assert_no_dsml_leak(testcase, collected_content)

    expected_content = expected["content"] or ""
    testcase.assertEqual(
        collected_content, expected_content,
        f"{label}: streaming content mismatch at chunk_size={chunk_size}")

    actual_tool_calls = []
    for index in sorted(tool_calls_by_index):
        state = tool_calls_by_index[index]
        actual_tool_calls.append(
            _CollectedToolCall(state["name"], state["arguments"]))

    assert_tool_calls_equal(testcase, actual_tool_calls,
                            expected["tool_calls"], case)


class _CollectedFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _CollectedToolCall:
    def __init__(self, name: str, arguments: str):
        self.function = _CollectedFunction(name, arguments)
