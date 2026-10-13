"""Thinking stays separate from visible text and survives tool round trips."""
import json
import unittest

from test_naive_n05_tools import NaiveModel, CALL, TOOL, RawRequest, completion
from tools.fastllm_pytools.openai_server.protocal.anthropic_protocol import AnthropicMessageRequest


class RecordingModel(NaiveModel):
    def get_input_token_len(self, messages, **kwargs):
        self.counted_messages = messages
        return super().get_input_token_len(messages, **kwargs)

    def launch_stream_response(self, messages, tool_call_constraint=None, **kwargs):
        self.generated_messages = messages
        return super().launch_stream_response(messages, **kwargs)


class AnthropicThinkingTest(unittest.IsolatedAsyncioTestCase):
    async def generate(self, output, stream, *, thinking=True, messages=None, max_tokens=1024):
        model = RecordingModel(output)
        instance = completion(model)
        req = AnthropicMessageRequest(
            model=instance.model_name, messages=messages or [{"role": "user", "content": "天气？"}],
            max_tokens=max_tokens, stream=stream,
            thinking={"type": "enabled" if thinking else "disabled"},
            tools=[{"name": TOOL["function"]["name"], "input_schema": TOOL["function"]["parameters"]}])
        response = await instance.create_anthropic_message(req, RawRequest())
        if not stream:
            data = response.model_dump()
        else:
            events = [json.loads(line[6:]) async for event in response[0]
                      for line in event.splitlines() if line.startswith("data: ")]
            self.assertFalse([e for e in events if e["type"] == "error"], events)
            self.assertEqual(events[0]["type"], "message_start")
            self.assertEqual(events[-1]["type"], "message_stop")
            blocks, stopped, pending_args = [], set(), {}
            for event in events:
                kind = event["type"]
                if kind == "content_block_start":
                    self.assertEqual(event["index"], len(blocks))
                    blocks.append(event["content_block"])
                elif kind == "content_block_delta":
                    idx, delta = event["index"], event["delta"]
                    self.assertNotIn(idx, stopped)
                    block = blocks[idx]
                    if delta["type"] == "input_json_delta":
                        self.assertEqual(block["type"], "tool_use")
                        pending_args[idx] = pending_args.get(idx, "") + delta["partial_json"]
                    else:
                        field = "thinking" if delta["type"] == "thinking_delta" else "text"
                        self.assertEqual(block["type"], field)
                        block[field] += delta[field]
                elif kind == "content_block_stop":
                    self.assertNotIn(event["index"], stopped)
                    stopped.add(event["index"])
                elif kind == "message_delta":
                    stop_reason = event["delta"]["stop_reason"]
            self.assertEqual(stopped, set(range(len(blocks))))
            for idx, args in pending_args.items():
                blocks[idx]["input"] = json.loads(args)
            data = {"content": blocks, "stop_reason": stop_reason}
        self.assertEqual(model.counted_messages, model.generated_messages)
        self.assertEqual(instance.conversation_handles, {})
        return data, model

    async def test_thinking_text_and_tool_roundtrip(self):
        # A tool delimiter inside reasoning must not produce a phantom call.
        thought = '先考虑 <tool_call>，然后查询。'
        for stream in (False, True):
            with self.subTest(stream=stream):
                data, _ = await self.generate('<think>' + thought + '</think>正在查询。' + CALL, stream)
                blocks = data['content']
                self.assertEqual([b['type'] for b in blocks], ['thinking', 'text', 'tool_use'])
                self.assertEqual(blocks[0]['thinking'], thought)
                self.assertEqual(blocks[1]['text'], '正在查询。')
                self.assertEqual(blocks[2]['input'], {'city': '北京'})
                self.assertEqual(data['stop_reason'], 'tool_use')
                data2, model = await self.generate('<think>已查询。</think>晴天。', stream, messages=[
                    {'role': 'user', 'content': '天气？'},
                    {'role': 'assistant', 'content': blocks},
                    {'role': 'user', 'content': [{'type': 'tool_result',
                        'tool_use_id': blocks[2]['id'], 'content': '晴天'}]},
                ])
                assistant = model.generated_messages[1]
                self.assertEqual(assistant['reasoning_content'], thought)
                self.assertEqual(assistant['content'], '正在查询。')
                self.assertEqual(assistant['tool_calls'][0]['function']['name'], 'get_weather')
                self.assertEqual(data2['content'][-1]['text'], '晴天。')

    async def test_truncated_reasoning_does_not_become_text(self):
        for stream in (False, True):
            data, _ = await self.generate('<think>还在思考</thi', stream, max_tokens=1)
            self.assertEqual(data['content'], [{'type': 'thinking', 'thinking': '还在思考</thi'}])
            self.assertEqual(data['stop_reason'], 'max_tokens')

    async def test_disabled_thinking_keeps_literal_text(self):
        for stream in (False, True):
            data, _ = await self.generate('字面量 <think>示例</think>。', stream, thinking=False)
            self.assertEqual(data['content'], [{'type': 'text', 'text': '字面量 <think>示例</think>。'}])

    async def test_empty_thought_has_no_empty_thinking_block(self):
        for stream in (False, True):
            data, _ = await self.generate('<think></think>答案', stream)
            self.assertEqual(data['content'], [{'type': 'text', 'text': '答案'}])


if __name__ == '__main__':
    unittest.main()
