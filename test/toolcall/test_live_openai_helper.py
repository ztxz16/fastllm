import copy
import json
import unittest

from lib import live_openai
from lib.live_openai import (
    LiveOpenAIError,
    load_live_cases,
    prepare_payload,
    reconstruct_stream_tool_calls,
    run_roundtrip_case,
    validate_tool_calls,
)
from lib.software_dev_tools import run_tests_tool


class LiveOpenAIHelperTest(unittest.TestCase):
    def test_live_cases_are_manual(self):
        cases = load_live_cases()
        self.assertGreaterEqual(len(cases), 1)
        self.assertTrue(all(case.get("status") == "manual" for case in cases))

    def test_prepare_payload_sets_model_stream_and_does_not_mutate_case(self):
        case = {
            "id": "live_dummy",
            "request": {
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0,
            },
        }
        original = copy.deepcopy(case)
        payload = prepare_payload(case, "dummy-model", stream=True,
                                  temperature_override=0.2)
        self.assertEqual(case, original)
        self.assertEqual(payload["model"], "dummy-model")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["temperature"], 0.2)

    def test_reconstruct_stream_tool_calls(self):
        chunks = [
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\"",
                            },
                        }]
                    },
                    "finish_reason": None,
                }]
            },
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": ":\"北京\"}",
                            },
                        }]
                    },
                    "finish_reason": None,
                }]
            },
            {
                "choices": [{
                    "delta": {},
                    "finish_reason": "tool_calls",
                }]
            },
        ]
        tool_calls, trace = reconstruct_stream_tool_calls(chunks)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["id"], "call_1")
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(tool_calls[0]["arguments"], "{\"city\":\"北京\"}")
        self.assertEqual(trace["finish_reason"], "tool_calls")
        self.assertEqual(trace["argument_fragment_counts"]["0"], 2)

    def test_invalid_tool_name_reports_diagnostics_without_fuzzy_pass(self):
        case = {
            "id": "live_dummy",
            "expected": {
                "must_call_tool": True,
                "allowed_tool_names": ["get_weather"],
                "arguments_must_be_json": True,
            },
        }
        with self.assertRaises(LiveOpenAIError) as ctx:
            validate_tool_calls(case, [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "name": "get_wearher",
                "arguments": "{\"city\":\"北京\"}",
            }])
        self.assertEqual(ctx.exception.code, "invalid_tool_name")
        self.assertEqual(ctx.exception.details["invalid_tool_name"],
                         "get_wearher")
        self.assertEqual(ctx.exception.details["allowed_tool_names"],
                         ["get_weather"])

    def test_roundtrip_allows_first_tool_call_with_null_content(self):
        case = {
            "id": "live_roundtrip_dummy",
            "mode": "roundtrip",
            "request": {
                "messages": [{"role": "user", "content": "weather"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }],
                "tool_choice": "required",
            },
            "tool_result": {
                "content": {"city": "北京", "temperature": 22},
                "follow_up_user_message": "answer",
            },
            "expected": {
                "must_call_tool": True,
                "allowed_tool_names": ["get_weather"],
                "arguments_must_be_json": True,
                "required_argument_keys": ["city"],
                "final_content_non_empty": True,
            },
        }
        responses = [
            {
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"北京\"}",
                            },
                        }],
                    },
                }]
            },
            {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "北京 22 度。",
                    },
                }]
            },
        ]
        seen_payloads = []

        def fake_post_json(_base_url, payload, _timeout):
            seen_payloads.append(copy.deepcopy(payload))
            return responses.pop(0)

        original_post_json = live_openai.post_json
        live_openai.post_json = fake_post_json
        try:
            result = run_roundtrip_case(
                case,
                base_url="http://127.0.0.1:8080/v1",
                model="dummy",
                timeout=1,
            )
        finally:
            live_openai.post_json = original_post_json

        self.assertEqual(result["second_content"], "北京 22 度。")
        self.assertEqual(len(seen_payloads), 2)
        second_messages = seen_payloads[1]["messages"]
        self.assertEqual(second_messages[1]["role"], "assistant")
        self.assertIsNone(second_messages[1]["content"])
        self.assertEqual(
            second_messages[1]["tool_calls"][0]["function"]["name"],
            "get_weather",
        )
        self.assertEqual(second_messages[2]["role"], "tool")
        self.assertEqual(second_messages[2]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(second_messages[2]["content"]),
                         {"city": "北京", "temperature": 22})

    def test_software_dev_run_tests_roundtrip_preserves_tool_result(self):
        case = {
            "id": "live_run_tests_roundtrip_dummy",
            "mode": "roundtrip",
            "request": {
                "messages": [{
                    "role": "user",
                    "content": "请运行单元测试并总结结果。",
                }],
                "tools": [run_tests_tool()],
                "tool_choice": "required",
            },
            "tool_result": {
                "content": {
                    "exit_code": 0,
                    "stdout": "Ran 91 tests in 0.156s\nOK",
                    "stderr": "",
                },
                "follow_up_user_message": "请根据测试结果回答。",
            },
            "expected": {
                "must_call_tool": True,
                "allowed_tool_names": ["run_tests"],
                "arguments_must_be_json": True,
                "required_argument_keys": ["command"],
                "final_content_non_empty": True,
            },
        }
        responses = [
            {
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_tests",
                            "type": "function",
                            "function": {
                                "name": "run_tests",
                                "arguments": (
                                    "{\"command\":[\"python\",\"-m\","
                                    "\"unittest\",\"discover\"],\"cwd\":\".\"}"
                                ),
                            },
                        }],
                    },
                }]
            },
            {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "测试已通过。",
                    },
                }]
            },
        ]
        seen_payloads = []

        def fake_post_json(_base_url, payload, _timeout):
            seen_payloads.append(copy.deepcopy(payload))
            return responses.pop(0)

        original_post_json = live_openai.post_json
        live_openai.post_json = fake_post_json
        try:
            result = run_roundtrip_case(
                case,
                base_url="http://127.0.0.1:8080/v1",
                model="dummy",
                timeout=1,
            )
        finally:
            live_openai.post_json = original_post_json

        self.assertEqual(result["second_content"], "测试已通过。")
        self.assertEqual(len(seen_payloads), 2)
        second_messages = seen_payloads[1]["messages"]
        self.assertEqual(second_messages[1]["role"], "assistant")
        self.assertIsNone(second_messages[1]["content"])
        self.assertEqual(
            second_messages[1]["tool_calls"][0]["function"]["name"],
            "run_tests",
        )
        self.assertEqual(second_messages[2]["role"], "tool")
        self.assertEqual(second_messages[2]["tool_call_id"], "call_tests")
        tool_content = json.loads(second_messages[2]["content"])
        self.assertEqual(tool_content["exit_code"], 0)
        self.assertIn("91 tests", tool_content["stdout"])


if __name__ == "__main__":
    unittest.main()
