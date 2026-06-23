#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fastllm_pytools.openai_server.fastllm_completion import (  # noqa: E402
    FastLLmCompletion,
)
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from lib.software_dev_tools import (  # noqa: E402
    run_tests_tool,
    search_code_tool,
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


def _strict_weather_tool(name="get_weather"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Get weather.",
            "strict": True,
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


def _request(tools=None, tool_choice="auto", parallel_tool_calls=None):
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "查天气"}],
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        max_tokens=128,
    )


def _dsml_call(name: str) -> str:
    return (
        "<｜DSML｜tool_calls>"
        f"<｜DSML｜invoke name=\"{name}\">"
        "<｜DSML｜parameter name=\"city\" string=\"true\">北京</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )


def _param(name: str, is_string: bool, value: str) -> str:
    return (
        f"<｜DSML｜parameter name=\"{name}\" "
        f"string=\"{str(is_string).lower()}\">{value}</｜DSML｜parameter>"
    )


def _invoke(name: str, *params: str) -> str:
    return (
        f"<｜DSML｜invoke name=\"{name}\">"
        f"{''.join(params)}"
        "</｜DSML｜invoke>"
    )


def _tool_calls(*invokes: str) -> str:
    return "<｜DSML｜tool_calls>" + "".join(invokes) + "</｜DSML｜tool_calls>"


def _search_code_dsml_call() -> str:
    return _tool_calls(_invoke(
        "search_code",
        _param("query", True, "FunctionCallParser"),
        _param("path", True, "tools/fastllm_pytools/openai_server"),
        _param("file_pattern", True, "*.py"),
    ))


def _run_tests_dsml_call() -> str:
    return _tool_calls(_invoke(
        "run_tests",
        _param("command", False, '["python","-m","unittest","test.toolcall"]'),
        _param("cwd", True, "."),
        _param("timeout_seconds", False, "120"),
    ))


def _dsml_call_with_prefix(name: str) -> str:
    return "我来查一下。\n" + _dsml_call(name)


def _malformed_dsml_call() -> str:
    return (
        "<｜DSML｜tool_calls>"
        "<｜DSML｜invoke name=\"get_weather\">"
        "<｜DSML｜parameter name=\"city\" string=\"true\">北京</｜DSML｜parameter>"
        "</｜DSML｜tool_calls>"
    )


def _empty_args_dsml_call() -> str:
    return (
        "<｜DSML｜tool_calls>"
        "<｜DSML｜invoke name=\"get_weather\">"
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )


def _parallel_dsml_call() -> str:
    return (
        "<｜DSML｜tool_calls>"
        "<｜DSML｜invoke name=\"get_weather\">"
        "<｜DSML｜parameter name=\"city\" string=\"true\">北京</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
        "<｜DSML｜invoke name=\"get_time\">"
        "<｜DSML｜parameter name=\"timezone\" string=\"true\">Asia/Shanghai</｜DSML｜parameter>"
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>"
    )


async def _result_generator(text: str):
    yield text


class _RawRequest:
    async def is_disconnected(self):
        return False


class _DummyModel:
    force_chat_template = False
    tool_call_parser = "deepseek_v4"
    hf_tokenizer = None

    def get_type(self):
        return "deepseek_v4"

    def abort_handle(self, handle):
        raise AssertionError("abort_handle should not be called")


def _completion():
    completion = FastLLmCompletion.__new__(FastLLmCompletion)
    completion.model_name = "dummy"
    completion.model = _DummyModel()
    completion.conversation_handles = {}
    return completion


class NonStreamToolCallResponseTest(unittest.IsolatedAsyncioTestCase):
    async def _run_full_generator(self, text: str, request: ChatCompletionRequest):
        completion = _completion()
        return await completion.chat_completion_full_generator(
            request=request,
            raw_request=_RawRequest(),
            handle=0,
            result_generator=_result_generator(text),
            request_id="chatcmpl-test",
            input_token_len=3,
        )

    async def test_valid_tool_call_preserves_openai_response_shape(self):
        response = await self._run_full_generator(
            _dsml_call("get_weather"),
            _request(tools=[_weather_tool()]),
        )

        self.assertIsInstance(response, ChatCompletionResponse)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertIsNone(choice.message.content)
        self.assertEqual(len(choice.message.tool_calls), 1)
        tool_call = choice.message.tool_calls[0]
        self.assertEqual(tool_call.function.name, "get_weather")
        self.assertEqual(json.loads(tool_call.function.arguments),
                         {"city": "北京"})

    async def test_prefix_content_is_preserved_for_valid_tool_call(self):
        response = await self._run_full_generator(
            _dsml_call_with_prefix("get_weather"),
            _request(tools=[_weather_tool()]),
        )

        self.assertIsInstance(response, ChatCompletionResponse)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.content, "我来查一下。\n")
        self.assertEqual(choice.message.tool_calls[0].function.name,
                         "get_weather")

    async def test_software_dev_search_code_response_shape(self):
        response = await self._run_full_generator(
            _search_code_dsml_call(),
            _request(tools=[search_code_tool()]),
        )

        self.assertIsInstance(response, ChatCompletionResponse)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertIsNone(choice.message.content)
        self.assertEqual(len(choice.message.tool_calls), 1)
        tool_call = choice.message.tool_calls[0]
        self.assertEqual(tool_call.function.name, "search_code")
        self.assertEqual(json.loads(tool_call.function.arguments), {
            "query": "FunctionCallParser",
            "path": "tools/fastllm_pytools/openai_server",
            "file_pattern": "*.py",
        })

    async def test_software_dev_run_tests_array_arguments(self):
        response = await self._run_full_generator(
            _run_tests_dsml_call(),
            _request(tools=[run_tests_tool()]),
        )

        self.assertIsInstance(response, ChatCompletionResponse)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        tool_call = choice.message.tool_calls[0]
        self.assertEqual(tool_call.function.name, "run_tests")
        self.assertEqual(json.loads(tool_call.function.arguments), {
            "command": ["python", "-m", "unittest", "test.toolcall"],
            "cwd": ".",
            "timeout_seconds": 120,
        })

    async def test_unknown_tool_name_returns_error_without_leaking_dsml(self):
        with self.assertLogs(level="WARNING") as logs:
            response = await self._run_full_generator(
                _dsml_call("get_wearher"),
                _request(tools=[_weather_tool()]),
            )

        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(response.code, 400)
        self.assertIn("invalid_tool_name", response.message)
        self.assertIn("get_wearher", response.message)
        self.assertNotIn("DSML", response.message)
        self.assertNotIn("tool_calls", response.message)
        self.assertIn("invalid_tool_name", "\n".join(logs.output))

    async def test_compat_mode_error_includes_closest_match_diagnostic(self):
        with patch.dict("os.environ", {"FT_TOOLCALL_COMPAT_MODE": "ON"}):
            with self.assertLogs(level="WARNING"):
                response = await self._run_full_generator(
                    _dsml_call("get_wearher"),
                    _request(tools=[_weather_tool()]),
                )

        self.assertIsInstance(response, ErrorResponse)
        self.assertIn("invalid_tool_name", response.message)
        self.assertIn("get_wearher", response.message)
        self.assertIn("closest='get_weather'", response.message)
        self.assertIn("ratio=", response.message)

    async def test_forward_unknown_tools_returns_raw_unknown_tool_call(self):
        with patch.dict("os.environ",
                        {"FT_TOOLCALL_FORWARD_UNKNOWN_TOOLS": "ON"}):
            response = await self._run_full_generator(
                _dsml_call("get_wearher"),
                _request(tools=[_weather_tool()]),
            )

        self.assertIsInstance(response, ChatCompletionResponse)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.tool_calls[0].function.name,
                         "get_wearher")
        self.assertEqual(json.loads(
            choice.message.tool_calls[0].function.arguments), {"city": "北京"})

    async def test_malformed_tool_block_returns_error_without_leaking_dsml(self):
        with self.assertLogs(level="WARNING") as logs:
            response = await self._run_full_generator(
                _malformed_dsml_call(),
                _request(tools=[_weather_tool()]),
            )

        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(response.code, 400)
        self.assertIn("malformed_tool_block", response.message)
        self.assertNotIn("DSML", response.message)
        self.assertNotIn("tool_calls", response.message)
        self.assertIn("malformed_tool_block", "\n".join(logs.output))

    async def test_required_tool_choice_no_call_returns_error(self):
        with self.assertLogs(level="WARNING") as logs:
            response = await self._run_full_generator(
                "普通回复",
                _request(tools=[_weather_tool()], tool_choice="required"),
            )

        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(response.code, 400)
        self.assertIn("tool_choice_violation", response.message)
        self.assertIn("tool_choice_violation", "\n".join(logs.output))

    async def test_named_tool_choice_mismatch_returns_error(self):
        with self.assertLogs(level="WARNING") as logs:
            response = await self._run_full_generator(
                _dsml_call("get_weather"),
                _request(
                    tools=[_weather_tool(), _time_tool()],
                    tool_choice={
                        "type": "function",
                        "function": {"name": "get_time"},
                    },
                ),
            )

        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(response.code, 400)
        self.assertIn("tool_choice_violation", response.message)
        self.assertIn("tool_choice_violation", "\n".join(logs.output))

    async def test_parallel_tool_calls_false_returns_error_for_two_calls(self):
        with self.assertLogs(level="WARNING") as logs:
            response = await self._run_full_generator(
                _parallel_dsml_call(),
                _request(
                    tools=[_weather_tool(), _time_tool()],
                    parallel_tool_calls=False,
                ),
            )

        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(response.code, 400)
        self.assertIn("parallel_tool_calls_violation", response.message)
        self.assertIn("parallel_tool_calls_violation", "\n".join(logs.output))

    async def test_strict_schema_valid_call_passes(self):
        response = await self._run_full_generator(
            _dsml_call("get_weather"),
            _request(tools=[_strict_weather_tool()]),
        )

        self.assertIsInstance(response, ChatCompletionResponse)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.tool_calls[0].function.name,
                         "get_weather")
        self.assertEqual(json.loads(
            choice.message.tool_calls[0].function.arguments), {"city": "北京"})

    async def test_strict_schema_missing_required_returns_error(self):
        with self.assertLogs(level="WARNING") as logs:
            response = await self._run_full_generator(
                _empty_args_dsml_call(),
                _request(tools=[_strict_weather_tool()]),
            )

        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(response.code, 400)
        self.assertIn("missing_required_argument", response.message)
        self.assertIn("city", response.message)
        self.assertIn("missing_required_argument", "\n".join(logs.output))

    async def test_constraint_descriptor_dry_run_does_not_change_response(self):
        completion = _completion()
        request = _request(
            tools=[_strict_weather_tool()],
            tool_choice="required",
            parallel_tool_calls=False,
        )

        descriptor = completion._build_tool_call_constraint_descriptor(request)
        response = await completion.chat_completion_full_generator(
            request=request,
            raw_request=_RawRequest(),
            handle=0,
            result_generator=_result_generator(_dsml_call("get_weather")),
            request_id="chatcmpl-test",
            input_token_len=3,
        )

        self.assertEqual(descriptor.constraint_type, "deepseek_v4_dsml")
        self.assertTrue(descriptor.requires_tool_call)
        self.assertEqual(descriptor.strict_tool_names, ("get_weather",))
        self.assertFalse(descriptor.parallel_tool_calls)
        self.assertIsInstance(response, ChatCompletionResponse)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.tool_calls[0].function.name,
                         "get_weather")

    async def test_constraint_descriptor_dry_run_does_not_create_parser(self):
        completion = _completion()
        completion.model.hf_tokenizer = object()
        request = _request(tools=[_strict_weather_tool()])

        def fail_create_tool_parser():
            raise AssertionError("descriptor helper should not create parser")

        completion._create_tool_parser = fail_create_tool_parser
        descriptor = completion._build_tool_call_constraint_descriptor(request)

        self.assertEqual(descriptor.constraint_type, "deepseek_v4_dsml")
        self.assertEqual(descriptor.strict_tool_names, ("get_weather",))

    async def test_plain_text_without_tools_is_unchanged(self):
        response = await self._run_full_generator(
            "普通回复",
            _request(tools=None),
        )

        self.assertIsInstance(response, ChatCompletionResponse)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.message.content, "普通回复")
        self.assertIsNone(choice.message.tool_calls)


if __name__ == "__main__":
    unittest.main()
