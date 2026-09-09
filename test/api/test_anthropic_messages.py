"""Exercise Claude Code message formats through the production HTTP route."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from test_qwen35_reasoning import FakeQwen35Model, completion
from fastapi.testclient import TestClient
from tools.fastllm_pytools import server
from jinja2.exceptions import TemplateError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


class TemplateModel(FakeQwen35Model):
    """Render the upstream model template without loading model weights."""
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(
            {"[UNK]": 0, "<tool_call>": 1, "</tool_call>": 2}, unk_token="[UNK]")),
        chat_template=(Path(__file__).parent / "fixtures/qwen35_chat_template.jinja").read_text(),
    )

    def render(self, messages, **kwargs):
        return self.hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=kwargs.get("tools"), enable_thinking=kwargs.get("enable_thinking"))


class CapturingModel(TemplateModel):

    def get_input_token_len(self, messages, **kwargs):
        self.counted_prompt = self.render(messages, **kwargs)
        self.counted_messages = copy.deepcopy(messages)
        return super().get_input_token_len(messages, **kwargs)

    def launch_stream_response(self, query, tool_call_constraint=None, **kwargs):
        self.generated_prompt = self.render(query, **kwargs)
        self.generated_messages = copy.deepcopy(query)
        return super().launch_stream_response(query, **kwargs)


class AnthropicMessagesTest(unittest.TestCase):
    def setUp(self):
        self.model = CapturingModel(output="这是 FastLLM 项目。")
        instance = completion(self.model)
        instance.enable_thinking = False
        self.server_patch = patch.object(server, "fastllm_completion", instance, create=True)
        self.server_patch.start()
        self.addCleanup(self.server_patch.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def assert_message_response(self, response, stream):
        self.assertEqual(response.status_code, 200, response.text)
        if stream:
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ")]
            self.assertEqual(events[0]["type"], "message_start")
            self.assertEqual(events[-1]["type"], "message_stop")
            text = "".join(event["delta"].get("text", "") for event in events
                           if event["type"] == "content_block_delta")
            self.assertEqual(text, self.model.output)
        else:
            self.assertEqual(response.json()["content"],
                             [{"type": "text", "text": self.model.output}])
        self.assertEqual(self.model.counted_messages, self.model.generated_messages)
        self.assertEqual(self.model.counted_prompt, self.model.generated_prompt)

    def test_system_after_user_accepts_text_and_blocks(self):
        for stream in (False, True):
            for use_blocks in (False, True):
                with self.subTest(stream=stream, use_blocks=use_blocks):
                    system = "先查看项目说明。"
                    if use_blocks:
                        system = [{"type": "text", "text": system,
                                   "cache_control": {"type": "ephemeral"}}]
                    response = self.client.post("/v1/messages?beta=true", json={
                        "model": "qwen3.5", "max_tokens": 128, "stream": stream,
                        "system": system,
                        "messages": [
                            {"role": "user", "content": "这个项目是什么？"},
                            {"role": "system", "content": system},
                        ],
                    })
                    self.assert_message_response(response, stream)
                    self.assertEqual(self.model.generated_messages, [
                        {"role": "system", "content": "先查看项目说明。\n先查看项目说明。"},
                        {"role": "user", "content": "这个项目是什么？"},
                    ])

    def test_system_messages_preserve_tool_history_and_followup_order(self):
        messages = [
            {"role": "user", "content": "这个项目是什么？"},
            {"role": "system", "content": "当前目录是项目根目录。"},
            {"role": "assistant", "content": [{
                "type": "tool_use", "id": "tool-read", "name": "Read",
                "input": {"file_path": "README.md"},
            }]},
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "tool-read",
                "content": [{"type": "text", "text": "# FastLLM"}],
            }]},
            {"role": "system", "content": "根据已读取的文件回答。"},
            {"role": "assistant", "content": "这是推理引擎。"},
            {"role": "user", "content": "支持哪些模型？"},
            {"role": "system", "content": "继续查看模型列表。"},
        ]
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.client.post("/v1/messages", json={
                    "model": "qwen3.5", "max_tokens": 128, "stream": stream,
                    "messages": messages,
                    "tools": [{"name": "Read", "input_schema": {
                        "type": "object", "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    }}],
                })
                self.assert_message_response(response, stream)
                expected = copy.deepcopy(messages)
                expected[2] = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "tool-read", "type": "function", "function": {
                        "name": "Read", "arguments": {"file_path": "README.md"},
                    },
                }]}
                expected[3] = {"role": "tool", "content": "# FastLLM",
                               "tool_call_id": "tool-read"}
                expected = [{"role": "system", "content":
                    "当前目录是项目根目录。\n根据已读取的文件回答。\n继续查看模型列表。"}] + [
                        message for message in expected if message["role"] != "system"]
                self.assertEqual(self.model.generated_messages, expected)
                self.assertIn("<tool_response>\n# FastLLM\n</tool_response>", self.model.generated_prompt)

    def test_template_errors_return_anthropic_errors_without_starting_a_stream(self):
        for stream in (False, True):
            for method in ("get_input_token_len", "launch_stream_response"):
                with self.subTest(stream=stream, method=method):
                    with patch.object(self.model, method, side_effect=TemplateError("Unsupported template input")):
                        response = self.client.post("/v1/messages", json={
                            "model": "qwen3.5", "max_tokens": 128, "stream": stream,
                            "messages": [{"role": "user", "content": "answer"}],
                        })
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(response.json()["type"], "error")
                    self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
                    self.assertIn("Unsupported template input", response.json()["error"]["message"])
                    self.assertEqual(server.fastllm_completion.conversation_handles, {})

    def test_unknown_roles_still_fail_request_validation(self):
        response = self.client.post("/v1/messages", json={
            "model": "qwen3.5", "max_tokens": 128,
            "messages": [{"role": "invalid", "content": "answer"}],
        })
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"],
                         ["body", "messages", 0, "role"])
        self.assertIsNone(self.model.launch_kwargs)


if __name__ == "__main__":
    unittest.main()
