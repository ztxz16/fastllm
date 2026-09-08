"""Exercise Qwen EOF recovery through the OpenAI response generators."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fastllm_pytools.openai_server.fastllm_completion import FastLLmCompletion
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (
    ChatCompletionRequest,
    ErrorResponse,
)


class _Tokenizer:
    chat_template = "<tool_call><function=<parameter="

    def get_vocab(self):
        return {"<tool_call>": 1, "</tool_call>": 2}


class _Model:
    force_chat_template = False
    tool_call_parser = "qwen3_coder"
    hf_tokenizer = _Tokenizer()

    def get_type(self):
        return "qwen3_5"


class _RawRequest:
    async def is_disconnected(self):
        return False


_TOOL = {"type": "function", "function": {
    "name": "read_file", "strict": True, "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}}
_FUNCTION = (
    "<tool_call><function=read_file>"
    "<parameter=path>README.md</parameter></function>"
)


class QwenToolCallTruncationTest(unittest.IsolatedAsyncioTestCase):
    async def _response(self, text, *, stream, max_tokens=24,
                        chunk_size=7, thinking=False, tool_choice="auto",
                        parallel_tool_calls=None):
        completion = FastLLmCompletion.__new__(FastLLmCompletion)
        completion.model = _Model()
        completion.model_name = "dummy"
        completion.conversation_handles = {}
        request = ChatCompletionRequest(
            model="dummy", messages=[], tools=[_TOOL], stream=stream,
            max_tokens=max_tokens, tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

        async def generated():
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]

        kwargs = dict(
            request=request, raw_request=_RawRequest(),
            result_generator=generated(), request_id="chatcmpl-truncation",
            input_token_len=3, emit_reasoning_content=thinking,
            response_statistics={"output_tokens": 24},
        )
        if not stream:
            response = await completion.chat_completion_full_generator(
                **kwargs, handle=0)
            if isinstance(response, ErrorResponse):
                return {"errors": [response.message], "calls": []}
            choice = response.choices[0]
            return {
                "errors": [], "finish": choice.finish_reason,
                "calls": [c.model_dump() for c in choice.message.tool_calls or []],
                "reasoning": choice.message.reasoning_content or "",
                "usage": response.usage.model_dump(),
            }
        events = []
        done = 0
        async for event in completion.chat_completion_stream_generator(
                **kwargs, think=False):
            for line in event.splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    done += 1
                else:
                    events.append(json.loads(payload))
        self.assertEqual(done, 1)
        choices = [c for e in events for c in e.get("choices", [])]
        finishes = [c["finish_reason"] for c in choices if c.get("finish_reason")]
        errors = [e["error"]["message"] for e in events if "error" in e]
        if not errors:
            self.assertEqual(len(finishes), 1)
        return {
            "errors": errors, "finish": finishes[0] if finishes else None,
            "calls": [t for c in choices for t in c["delta"].get("tool_calls", [])],
            "reasoning": "".join(c["delta"].get("reasoning_content") or ""
                                 for c in choices),
            "usage": next((e["usage"] for e in events if e.get("usage")), None),
        }

    async def test_complete_function_recovers_at_stop_or_length(self):
        for stream in (False, True):
            for max_tokens in (24, 128):
                for chunk_size in (1, 7, 1024):
                    with self.subTest(stream=stream, max_tokens=max_tokens,
                                      chunk_size=chunk_size):
                        result = await self._response(
                            _FUNCTION, stream=stream, max_tokens=max_tokens,
                            chunk_size=chunk_size)
                        self.assertEqual(result["errors"], [])
                        self.assertEqual(result["finish"], "tool_calls")
                        self.assertEqual(len(result["calls"]), 1)
                        self.assertEqual(result["calls"][0]["function"]["name"],
                                         "read_file")
                        self.assertEqual(json.loads(
                            result["calls"][0]["function"]["arguments"]),
                            {"path": "README.md"})
                        self.assertEqual(result["usage"]["completion_tokens"], 24)

    async def test_incomplete_arguments_keep_length_without_a_call(self):
        for stream in (False, True):
            for tool_choice in ("auto", "required"):
                for text in (_FUNCTION[:-1], _FUNCTION.split("</parameter>")[0]):
                    with self.subTest(stream=stream, choice=tool_choice, text=text):
                        result = await self._response(
                            text, stream=stream, tool_choice=tool_choice)
                        self.assertEqual(result["errors"], [])
                        self.assertEqual(result["calls"], [])
                        self.assertEqual(result["finish"], "length")
                        self.assertEqual(result["usage"]["completion_tokens"], 24)

    async def test_incomplete_arguments_without_length_remain_errors(self):
        for stream in (False, True):
            result = await self._response(
                _FUNCTION[:-1], stream=stream, max_tokens=128)
            self.assertTrue(result["errors"])
            self.assertEqual(result["calls"], [])

    async def test_recovery_does_not_bypass_validation_at_length(self):
        for stream in (False, True):
            for text in (
                _FUNCTION.replace("read_file", "unknown_tool"),
                "<tool_call><function=read_file></function>",
                _FUNCTION + "garbage",
                "<tool_call></tool_call>",
            ):
                with self.subTest(stream=stream, text=text):
                    result = await self._response(text, stream=stream)
                    self.assertTrue(result["errors"])
                    self.assertEqual(result["calls"], [])

    async def test_normal_reasoning_is_preserved_with_recovered_call(self):
        for stream in (False, True):
            result = await self._response(
                "<think>Read the file.</think>" + _FUNCTION,
                stream=stream, thinking=True)
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["finish"], "tool_calls")
            self.assertEqual(result["reasoning"], "Read the file.")
            self.assertEqual(len(result["calls"]), 1)

    async def test_recovery_respects_explicit_unknown_tool_forwarding(self):
        with patch.dict("os.environ", {"FT_TOOLCALL_FORWARD_UNKNOWN_TOOLS": "ON"}):
            for stream in (False, True):
                result = await self._response(
                    _FUNCTION.replace("read_file", "unknown_tool"), stream=stream)
                self.assertEqual(result["errors"], [])
                self.assertEqual(result["finish"], "tool_calls")
                self.assertEqual(len(result["calls"]), 1)
                self.assertEqual(result["calls"][0]["function"]["name"],
                                 "unknown_tool")

    async def test_recovery_respects_parallel_tool_call_limit(self):
        for stream in (False, True):
            result = await self._response(
                _FUNCTION + "</tool_call>" + _FUNCTION,
                stream=stream, parallel_tool_calls=False)
            self.assertTrue(result["errors"])
            self.assertEqual(result["calls"], [])

    async def test_parallel_recovery_preserves_order_and_unique_ids(self):
        text = _FUNCTION + "</tool_call>" + _FUNCTION.replace("README.md", "a.py")
        for stream in (False, True):
            result = await self._response(text, stream=stream)
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["finish"], "tool_calls")
            self.assertEqual([json.loads(c["function"]["arguments"])["path"]
                              for c in result["calls"]], ["README.md", "a.py"])
            self.assertEqual(len({c["id"] for c in result["calls"]}), 2)

    async def test_incomplete_second_call_keeps_stream_length(self):
        result = await self._response(
            _FUNCTION + "</tool_call>" + _FUNCTION[:-1], stream=True)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["finish"], "length")
        self.assertEqual(len(result["calls"]), 1)


if __name__ == "__main__":
    unittest.main()
