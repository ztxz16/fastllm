"""Codex titles must complete while the model is busy, without changing chat."""
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Keep test/api/openai.py and http.py from shadowing installed packages.
ORIGINAL_SYS_PATH = list(sys.path)
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != str(Path(__file__).parent)]
sys.path.insert(0, str(ROOT / "test" / "toolcall"))
sys.path.insert(0, str(ROOT))
from test_toolcall_generation_constraint import _ConstraintUnsupportedModel, _RawRequest, _completion
from tools.fastllm_pytools.openai_server.codex_title import TITLE_PREFIX, TITLE_SCHEMA, local_codex_title
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import ErrorResponse, ResponsesRequest
sys.path[:] = ORIGINAL_SYS_PATH


def title_request(prompt="做一个海边鹈鹕骑车的html，并部署到局域网", **kwargs):
    return ResponsesRequest(model="dummy", input=(TITLE_PREFIX +
        ". Write in the user's language. Do not answer the request.\n\nUser prompt:\n" + prompt),
        text={"format": {"type": "json_schema", "name": "codex_output_schema", "strict": True,
                         "schema": TITLE_SCHEMA}}, **kwargs)


class CodexLocalTitleTest(unittest.IsolatedAsyncioTestCase):
    async def test_opt_in_completes_stream_and_nonstream_without_model(self):
        for stream in (False, True):
            model = _ConstraintUnsupportedModel("must not generate")
            completion = _completion(model)
            raw = _RawRequest()
            raw.headers = {"x-fastllm-codex-title-mode": "local"}
            response = await completion.create_response(title_request(stream=stream), raw)
            if stream:
                events = [json.loads(line[6:]) async for chunk in response[0]
                          for line in chunk.splitlines() if line.startswith("data: ")]
                self.assertEqual(events[-1]["type"], "response.completed")
                response = events[-1]["response"]
                text = response["output_text"]
                self.assertEqual(response["usage"]["total_tokens"], 0)
            else:
                text = response.output_text
                self.assertEqual(response.status, "completed")
                self.assertEqual(response.usage.total_tokens, 0)
            self.assertEqual(json.loads(text)["title"], "做一个海边鹈鹕骑车的html，并部署到局域网")
            self.assertFalse(model.launch_called)
            self.assertIsNone(model.input_token_messages)

    async def test_without_header_uses_model(self):
        model = _ConstraintUnsupportedModel('{"title":"模型标题"}')
        response = await _completion(model).create_response(title_request(), _RawRequest())
        self.assertTrue(model.launch_called)
        self.assertEqual(json.loads(response.output_text)["title"], "模型标题")

    async def test_opt_in_still_checks_model_and_leaves_other_json_to_model(self):
        model = _ConstraintUnsupportedModel('{"title":"模型标题"}')
        completion = _completion(model)
        raw = _RawRequest()
        raw.headers = {"x-fastllm-codex-title-mode": "local"}
        request = title_request()
        request.model = "missing-model"
        response = await completion.create_response(request, raw)
        self.assertIsInstance(response, ErrorResponse)
        self.assertEqual(response.code, 404)
        self.assertFalse(model.launch_called)
        request = title_request()
        request.text["format"]["name"] = "ordinary_schema"
        response = await completion.create_response(request, raw)
        self.assertTrue(model.launch_called)
        self.assertEqual(json.loads(response.output_text)["title"], "模型标题")

    def test_exact_auxiliary_match_and_length(self):
        for prompt in ("你好", "A" * 100, "修复 " + "问题" * 30):
            request = title_request(prompt)
            title = local_codex_title(request, [{"role": "user", "content": request.input}])
            self.assertTrue(1 <= len(title) <= 36)
        request = title_request()
        self.assertIsNone(local_codex_title(request, [{"role": "user", "content": "给项目命名"}]))
        self.assertIsNone(local_codex_title(request, [{"role": "tool", "content": request.input}]))
        request.text["format"]["name"] = "ordinary_schema"
        self.assertIsNone(local_codex_title(request, [{"role": "user", "content": request.input}]))


if __name__ == "__main__":
    unittest.main()
