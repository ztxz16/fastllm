import asyncio
import copy
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Running this script directly must not shadow stdlib http with test/api/http.py.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != str(Path(__file__).parent)]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "toolcall"))

from test_toolcall_generation_constraint import (
    _ConstraintUnsupportedModel, _RawRequest, _completion, _dsml_call, _weather_tool,
)
from tools.fastllm_pytools.openai_server.protocal.openai_protocol import (
    ChatCompletionRequest, ErrorResponse, ResponsesRequest,
)
from tools.fastllm_pytools.openai_server.structured_output import (
    prepare_structured_output, validate_structured_output,
)


SCHEMA = {
    "type": "object", "properties": {"title": {"type": "string", "minLength": 1, "maxLength": 36}},
    "required": ["title"], "additionalProperties": False,
}
FORMAT = {"type": "json_schema", "json_schema": {"name": "task", "schema": SCHEMA, "strict": True}}


class StructuredOutputTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_streams_keep_their_own_schema_and_content(self):
        class ConcurrentModel(_ConstraintUnsupportedModel):
            def __init__(self):
                super().__init__("")
                self.outputs = {}

            def launch_stream_response(self, query, **kwargs):
                handle = len(self.outputs) + 1
                marker = query[-1]["content"]
                self.outputs[handle] = json.dumps({"marker": marker})
                return handle

            async def stream_response_handle_async(self, handle):
                for char in self.outputs[handle]:
                    await asyncio.sleep(0)
                    yield char

        completion = _completion(ConcurrentModel())

        async def run(marker):
            request = ChatCompletionRequest(
                model="dummy", messages=[{"role": "user", "content": marker}], stream=True,
                response_format={"type": "json_schema", "json_schema": {"schema": {
                    "type": "object", "properties": {"marker": {"const": marker}},
                    "required": ["marker"], "additionalProperties": False}}})
            stream, _ = await completion.create_chat_completion(request, _RawRequest())
            events = [json.loads(line[6:]) async for chunk in stream for line in chunk.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
            content = "".join((c.get("delta") or {}).get("content") or ""
                              for event in events for c in event.get("choices", []))
            self.assertEqual(json.loads(content), {"marker": marker})

        await asyncio.gather(run("alpha"), run("beta"))
        self.assertEqual(completion.conversation_handles, {})

    async def run_request(self, output, *, responses=False, stream=False, **kwargs):
        model = _ConstraintUnsupportedModel(output)
        completion = _completion(model)
        if responses:
            request = ResponsesRequest(model="dummy", input="Name this task.", stream=stream,
                                       text={"format": {"type": "json_schema", **FORMAT["json_schema"]}},
                                       **kwargs)
            result = await completion.create_response(request, _RawRequest())
        else:
            request = ChatCompletionRequest(model="dummy", messages=[{"role": "user", "content": "Name this task."}],
                                            stream=stream, response_format=FORMAT, **kwargs)
            result = await completion.create_chat_completion(request, _RawRequest())
        if stream and not isinstance(result, ErrorResponse):
            result = [json.loads(line[6:]) async for chunk in result[0] for line in chunk.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
        self.assertEqual(completion.conversation_handles, {})
        return result, model

    async def test_both_apis_stream_and_nonstream(self):
        for responses in (False, True):
            for stream in (False, True):
                with self.subTest(responses=responses, stream=stream):
                    result, model = await self.run_request('{"title":"检查目录"}', responses=responses, stream=stream)
                    self.assertIn('"maxLength": 36', model.input_token_messages[0]["content"])
                    if stream:
                        if responses:
                            self.assertEqual(result[-1]["type"], "response.completed")
                            self.assertEqual(result[-1]["response"]["output_text"], '{"title":"检查目录"}')
                        else:
                            self.assertEqual(result[-1]["choices"][0]["finish_reason"], "stop")
                    elif responses:
                        self.assertEqual(result.output_text, '{"title":"检查目录"}')
                    else:
                        self.assertEqual(result.choices[0].message.content, '{"title":"检查目录"}')

    async def test_invalid_output_cannot_report_success(self):
        for output in ('检查目录', '{"title":""}', '{"title":"ok","extra":1}', '```json\n{"title":"ok"}\n```'):
            for responses in (False, True):
                for stream in (False, True):
                    with self.subTest(output=output, responses=responses, stream=stream):
                        result, _ = await self.run_request(output, responses=responses, stream=stream)
                        if not stream:
                            self.assertIsInstance(result, ErrorResponse)
                            self.assertEqual(result.type, "invalid_response_format")
                            self.assertEqual(result.code, 500)
                        elif responses:
                            self.assertEqual(result[-1]["type"], "response.failed")
                            self.assertNotIn("response.completed", [r["type"] for r in result])
                        else:
                            self.assertEqual(result[-1]["error"]["type"], "invalid_response_format")
                            self.assertFalse(any(c.get("finish_reason") == "stop" for r in result for c in r.get("choices", [])))

    async def test_truncation_is_incomplete_not_schema_error(self):
        for stream in (False, True):
            result, _ = await self.run_request('{"title":', responses=True, stream=stream, max_output_tokens=1)
            if stream:
                self.assertEqual(result[-1]["type"], "response.incomplete")
            else:
                self.assertEqual(result.status, "incomplete")

    async def test_tool_calls_are_not_validated_as_final_json(self):
        for stream in (False, True):
            result, model = await self.run_request(_dsml_call(), stream=stream,
                                                  tools=[_weather_tool()], tool_choice="required")
            self.assertTrue(model.launch_kwargs["tools"])
            if stream:
                self.assertEqual(result[-1]["choices"][0]["finish_reason"], "tool_calls")
            else:
                self.assertEqual(result.choices[0].finish_reason, "tool_calls")

    async def test_invalid_format_rejected_before_model_launch(self):
        for format_spec in ({"type": "unknown"}, {"type": "json_schema"},
                            {"type": "json_schema", "json_schema": {"schema": {"type": "invalid"}}}):
            model = _ConstraintUnsupportedModel("unused")
            completion = _completion(model)
            request = ChatCompletionRequest(model="dummy", messages=[{"role": "user", "content": "Hi"}],
                                            response_format=format_spec)
            result = await completion.create_chat_completion(request, _RawRequest())
            self.assertIsInstance(result, ErrorResponse)
            self.assertEqual(result.code, 400)
            self.assertFalse(model.launch_called)

    def test_generic_guidance_preserves_roles_content_and_input(self):
        for leading in ([], [{"role": "system", "content": "Be helpful."}],
                        [{"role": "developer", "content": [{"type": "text", "text": "Be helpful."}]}]):
            messages = leading + [{"role": "user", "content": "Hi"}]
            before = copy.deepcopy(messages)
            prepared = prepare_structured_output(messages, FORMAT)
            self.assertEqual(messages, before)
            self.assertEqual(prepared[-1], messages[-1])
            self.assertIn("JSON Schema", str(prepared[0]["content"]))
            for plain_format in (None, {"type": "text"}):
                self.assertIs(prepare_structured_output(messages, plain_format), messages)

    def test_nested_schema_arrays_enums_and_local_refs(self):
        schema = {"type": "array", "items": {"$ref": "#/$defs/item"}, "$defs": {
            "item": {"type": "object", "properties": {"color": {"enum": ["red", "blue"]}, "count": {"type": "integer", "minimum": 1}},
                     "required": ["color", "count"], "additionalProperties": False}}}
        format_spec = {"type": "json_schema", "json_schema": {"schema": schema}}
        validate_structured_output('[{"color":"red","count":2}]', format_spec)
        for invalid in ('[{"color":"green","count":2}]', '[{"color":"red","count":0}]', '[{"color":"red","count":true}]'):
            with self.assertRaises(ValueError):
                validate_structured_output(invalid, format_spec)

    def test_json_object_and_nonstandard_numbers(self):
        validate_structured_output('{"count":2}', {"type": "json_object"})
        for invalid in ('[]', 'null', '{"count":NaN}', '{"count":Infinity}'):
            with self.assertRaises(ValueError):
                validate_structured_output(invalid, {"type": "json_object"})

    def test_remote_schema_references_are_not_fetched(self):
        with self.assertRaises(ValueError):
            validate_structured_output('{}', {"type": "json_schema", "json_schema": {
                "schema": {"$ref": "http://127.0.0.1:1/private"}}})


if __name__ == "__main__":
    unittest.main()
