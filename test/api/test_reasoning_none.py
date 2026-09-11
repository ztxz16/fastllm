"""An agent's explicit off selection overrides the running service default."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from test_anthropic_messages import CapturingModel, completion, server
from test_qwen35_reasoning import request


class ReasoningNoneTest(unittest.TestCase):
    def setUp(self):
        self.model = CapturingModel(output="42")
        self.completion = completion(self.model)
        self.completion.enable_thinking = True
        server_patch = patch.object(server, "fastllm_completion", self.completion, create=True)
        server_patch.start(); self.addCleanup(server_patch.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def test_off_reaches_counting_and_rendering_through_each_api(self):
        for api in ("chat/completions", "responses", "messages"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    body = {"model":"qwen3.5", "stream":stream}
                    if api == "responses":
                        body.update(input="Answer", max_output_tokens=128, reasoning={"effort":"none"})
                    else:
                        body.update(messages=[{"role":"user", "content":"Answer"}], max_tokens=128)
                        if api == "messages":
                            # Claude Code has a dedicated thinking switch.
                            body.update(thinking={"type":"disabled"}, output_config={"effort":None})
                        else:
                            body.update(reasoning_effort="none")
                    response = self.client.post("/v1/" + api, json=body)
                    self.assertEqual(response.status_code, 200, response.text)
                    for kwargs in (self.model.input_kwargs, self.model.launch_kwargs):
                        self.assertFalse(kwargs["enable_thinking"])
                        self.assertEqual(kwargs["chat_template_kwargs"]["reasoning_effort"], "medium")
                    self.assertEqual(self.model.counted_prompt, self.model.generated_prompt)
                    self.assertTrue(self.model.generated_prompt.endswith("<think>\n\n</think>\n\n"))
                    self.assertTrue(self.completion.enable_thinking)

    def test_anthropic_none_and_followup_effort_override_the_default(self):
        for effort, enabled in (("none", False), ("low", True)):
            with self.subTest(effort=effort):
                response = self.client.post("/v1/messages", json={
                    "model":"qwen3.5", "max_tokens":128,
                    "messages":[{"role":"user", "content":"Answer"}],
                    "output_config":{"effort":effort},
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.model.input_kwargs["enable_thinking"], enabled)
                self.assertEqual(self.model.launch_kwargs["enable_thinking"], enabled)

    def test_none_keeps_native_template_effort_valid_for_all_model_families(self):
        for model_type, resolver, expected in (
                ("qwen3_5", "_resolve_qwen3_5_reasoning_effort", "medium"),
                ("qwen4_exp", "_resolve_qwen3_5_reasoning_effort", "xhigh"),
                ("kimi_k3", "_resolve_kimi_k3_reasoning_effort", "max"),
                ("glm5_next", "_resolve_glm5_next_reasoning_effort", "max")):
            with self.subTest(model_type=model_type), patch.object(self.model, "get_type", return_value=model_type):
                self.assertEqual(getattr(self.completion, resolver)(request(reasoning_effort="none")), expected)


if __name__ == "__main__":
    unittest.main()
