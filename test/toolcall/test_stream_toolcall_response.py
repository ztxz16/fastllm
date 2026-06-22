#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fastllm_pytools.openai_server.fastllm_completion import (  # noqa: E402
    FastLLmCompletion,
)
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (  # noqa: E402
    ChatCompletionRequest,
)


def _weather_tool(name="get_weather"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _time_tool(name="get_time"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get time.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        },
    }


def _request(tools=None):
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "查工具"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=128,
        stream=True,
    )


def _invoke(name: str, params: str) -> str:
    return (
        f"<｜DSML｜invoke name=\"{name}\">"
        f"{params}"
        "</｜DSML｜invoke>"
    )


def _param(name: str, is_string: bool, value: str) -> str:
    return (
        f"<｜DSML｜parameter name=\"{name}\" "
        f"string=\"{str(is_string).lower()}\">{value}</｜DSML｜parameter>"
    )


def _tool_calls(*invokes: str) -> str:
    return "<｜DSML｜tool_calls>" + "".join(invokes) + "</｜DSML｜tool_calls>"


def _weather_call(name="get_weather") -> str:
    return _tool_calls(_invoke(name, _param("city", True, "北京")))


def _weather_and_time_call(weather_name="get_weather") -> str:
    return _tool_calls(
        _invoke(weather_name, _param("city", True, "北京")),
        _invoke("get_time", _param("timezone", True, "Asia/Shanghai")),
    )


async def _result_generator(text: str, chunk_size: int = 7):
    for index in range(0, len(text), chunk_size):
        yield text[index:index + chunk_size]


class _RawRequest:
    async def is_disconnected(self):
        return False


class _DummyModel:
    force_chat_template = False
    tool_call_parser = "deepseek_v4"
    hf_tokenizer = None

    def get_type(self):
        return "deepseek_v4"


def _completion():
    completion = FastLLmCompletion.__new__(FastLLmCompletion)
    completion.model_name = "dummy"
    completion.model = _DummyModel()
    completion.conversation_handles = {}
    return completion


class StreamToolCallResponseTest(unittest.IsolatedAsyncioTestCase):
    async def _collect_stream(
        self,
        text: str,
        request: ChatCompletionRequest,
    ) -> Tuple[List[Dict], bool]:
        completion = _completion()
        chunks: List[Dict] = []
        saw_done = False
        async for event in completion.chat_completion_stream_generator(
            request=request,
            raw_request=_RawRequest(),
            result_generator=_result_generator(text),
            request_id="chatcmpl-stream-test",
            input_token_len=3,
            think=False,
        ):
            for line in event.splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    saw_done = True
                else:
                    chunks.append(json.loads(payload))
        return chunks, saw_done

    def _reconstruct_tool_calls(self, chunks: List[Dict]) -> Dict[int, Dict]:
        tool_calls: Dict[int, Dict[str, str]] = {}
        for chunk in chunks:
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                for tool_call in delta.get("tool_calls", []):
                    index = tool_call["index"]
                    state = tool_calls.setdefault(
                        index, {"name": "", "arguments": ""})
                    function = tool_call.get("function") or {}
                    if function.get("name"):
                        state["name"] = function["name"]
                    if function.get("arguments"):
                        state["arguments"] += function["arguments"]
        return tool_calls

    def _finish_reason(self, chunks: List[Dict]) -> str:
        reasons = [
            choice.get("finish_reason")
            for chunk in chunks
            for choice in chunk.get("choices", [])
            if choice.get("finish_reason") is not None
        ]
        self.assertEqual(len(reasons), 1)
        return reasons[0]

    async def test_valid_tool_call_streams_tool_delta_and_done(self):
        chunks, saw_done = await self._collect_stream(
            _weather_call(),
            _request(tools=[_weather_tool()]),
        )

        tool_calls = self._reconstruct_tool_calls(chunks)
        self.assertTrue(saw_done)
        self.assertEqual(self._finish_reason(chunks), "tool_calls")
        self.assertEqual(set(tool_calls), {0})
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(json.loads(tool_calls[0]["arguments"]),
                         {"city": "北京"})

    async def test_unknown_tool_name_is_suppressed(self):
        with self.assertLogs(level="WARNING") as logs:
            chunks, saw_done = await self._collect_stream(
                _weather_call("get_wearher"),
                _request(tools=[_weather_tool()]),
            )

        serialized = json.dumps(chunks, ensure_ascii=False)
        self.assertTrue(saw_done)
        self.assertEqual(self._finish_reason(chunks), "stop")
        self.assertEqual(self._reconstruct_tool_calls(chunks), {})
        self.assertNotIn("DSML", serialized)
        self.assertNotIn("get_wearher", serialized)
        self.assertIn("invalid_tool_name", "\n".join(logs.output))

    async def test_valid_second_tool_after_invalid_first_is_compacted(self):
        with self.assertLogs(level="WARNING") as logs:
            chunks, saw_done = await self._collect_stream(
                _weather_and_time_call(weather_name="get_wearher"),
                _request(tools=[_time_tool()]),
            )

        tool_calls = self._reconstruct_tool_calls(chunks)
        self.assertTrue(saw_done)
        self.assertEqual(self._finish_reason(chunks), "tool_calls")
        self.assertEqual(set(tool_calls), {0})
        self.assertEqual(tool_calls[0]["name"], "get_time")
        self.assertEqual(json.loads(tool_calls[0]["arguments"]),
                         {"timezone": "Asia/Shanghai"})
        self.assertIn("invalid_tool_name", "\n".join(logs.output))

    async def test_parallel_valid_tools_preserve_order(self):
        chunks, saw_done = await self._collect_stream(
            _weather_and_time_call(),
            _request(tools=[_weather_tool(), _time_tool()]),
        )

        tool_calls = self._reconstruct_tool_calls(chunks)
        self.assertTrue(saw_done)
        self.assertEqual(self._finish_reason(chunks), "tool_calls")
        self.assertEqual([tool_calls[index]["name"] for index in sorted(tool_calls)],
                         ["get_weather", "get_time"])


if __name__ == "__main__":
    unittest.main()
