"""Generic client effort levels fall back to Qwen's native default over HTTP."""
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from test_qwen35_reasoning import FakeQwen35Model, completion
from tools.fastllm_pytools import server


class QwenReasoningDefaultTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeQwen35Model(output="42")
        self.completion = completion(self.model)
        self.completion.enable_thinking = False
        server_patch = patch.object(
            server, "fastllm_completion", self.completion, create=True)
        server_patch.start()
        self.addCleanup(server_patch.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def send(self, api, effort, stream, *, omit=False):
        body = {"model": self.completion.model_name, "stream": stream}
        if api == "responses":
            body.update(input="Answer", max_output_tokens=128)
            if not omit:
                body["reasoning"] = {"effort": effort}
        else:
            body.update(messages=[{"role": "user", "content": "Answer"}],
                        max_tokens=128)
            if not omit:
                if api == "messages":
                    body["output_config"] = {"effort": effort}
                else:
                    body["reasoning_effort"] = effort
        response = self.client.post("/v1/" + api, json=body)
        self.assertEqual(response.status_code, 200, response.text)
        payloads = ([json.loads(line[6:]) for line in response.text.splitlines()
                     if line.startswith("data: ") and line[6:] != "[DONE]"]
                    if stream else [response.json()])
        self.assertTrue(payloads)
        for payload in payloads:
            self.assertFalse(payload.get("error"), payload)
            self.assertNotEqual(payload.get("type"), "error", payload)

    def assert_generation(self, effort, enabled):
        for kwargs in (self.model.input_kwargs, self.model.launch_kwargs):
            self.assertEqual(kwargs["chat_template_kwargs"]["reasoning_effort"], effort)
            self.assertEqual(kwargs["enable_thinking"], enabled)
        self.assertFalse(self.completion.enable_thinking)

    def test_generic_efforts_fall_back_across_apis_and_qwen_families(self):
        for model_type in ("qwen3_5", "qwen3_5_moe", "qwen4_exp"):
            with patch.object(self.model, "get_type", return_value=model_type):
                for api in ("chat/completions", "responses", "messages"):
                    for stream in (False, True):
                        for effort in ("high", "max", "minimal", 75):
                            with self.subTest(model=model_type, api=api,
                                              stream=stream, effort=effort):
                                self.send(api, effort, stream)
                                self.assert_generation("xhigh", True)

    def test_native_efforts_and_service_thinking_default_are_preserved(self):
        for api in ("chat/completions", "responses", "messages"):
            for stream in (False, True):
                for effort in (None, "none", "low", "medium", "xhigh"):
                    with self.subTest(api=api, stream=stream, effort=effort):
                        self.send(api, effort, stream, omit=effort is None)
                        self.assert_generation(
                            "xhigh" if effort in (None, "none") else effort,
                            effort not in (None, "none"))


if __name__ == "__main__":
    unittest.main()
