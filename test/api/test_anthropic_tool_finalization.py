"""A final Qwen tool call must not turn into a successful text-only stop."""
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from test_anthropic_messages import CapturingModel
from test_qwen35_reasoning import completion
from tools.fastllm_pytools import server


class ChunkedModel(CapturingModel):
    def stream_response_handle_async(self, handle):
        async def generate():
            # Include empty decoder chunks; only real EOF finalizes a call.
            for char in self.output:
                yield char
                yield ""
        return generate()


class AnthropicToolFinalizationTest(unittest.TestCase):
    def setUp(self):
        self.model = CapturingModel()
        self.instance = completion(self.model)
        self.instance.enable_thinking = False
        patcher = patch.object(server, "fastllm_completion", self.instance, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def send(self, text, *, stream=True, max_tokens=32000):
        self.model.output = text
        return self.client.post("/v1/messages", json={
            "model": "qwen3.5", "max_tokens": max_tokens, "stream": stream,
            "messages": [{"role": "user", "content": "Write the file."}],
            "tools": [{"name": "Write", "input_schema": {
                "type": "object", "properties": {
                    "file_path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["file_path", "content"]}}],
        })

    @staticmethod
    def events(response):
        return [json.loads(line[6:]) for line in response.text.splitlines()
                if line.startswith("data: ")]

    def test_complete_call_at_eof_is_delivered_exactly_once(self):
        call = ("<tool_call><function=Write><parameter=file_path>done.txt</parameter>"
                "<parameter=content>done</parameter></function>")
        for model_cls in (CapturingModel, ChunkedModel):
            self.model = model_cls()
            self.instance.model = self.model
            for stream in (False, True):
                for tail in ("", "\n", "</tool_", "</tool_call>"):
                    with self.subTest(chunked=model_cls is ChunkedModel, stream=stream, tail=tail):
                        response = self.send("我现在写入文件。" + call + tail, stream=stream)
                        self.assertEqual(response.status_code, 200, response.text)
                        if stream:
                            events = self.events(response)
                            self.assertFalse([e for e in events if e['type'] == 'error'], events)
                            starts = [e for e in events if e['type'] == 'content_block_start'
                                      and e['content_block']['type'] == 'tool_use']
                            self.assertEqual(len(starts), 1, response.text)
                            index = starts[0]['index']
                            self.assertEqual(starts[0]['content_block']['name'], 'Write')
                            args = ''.join(e['delta'].get('partial_json', '') for e in events
                                           if e['type'] == 'content_block_delta' and e['index'] == index)
                            self.assertEqual(json.loads(args), {'file_path': 'done.txt', 'content': 'done'})
                            stops = [e for e in events if e['type'] == 'content_block_stop' and e['index'] == index]
                            self.assertEqual(len(stops), 1)
                            self.assertEqual(events[-2]['delta']['stop_reason'], 'tool_use')
                            self.assertEqual(events[-1]['type'], 'message_stop')
                        else:
                            body = response.json()
                            self.assertEqual(body['stop_reason'], 'tool_use')
                            calls = [b for b in body['content'] if b['type'] == 'tool_use']
                            self.assertEqual(len(calls), 1)
                            self.assertEqual(calls[0]['input'], {'file_path': 'done.txt', 'content': 'done'})
                        self.assertEqual(self.instance.conversation_handles, {})

    def test_malformed_call_cannot_silently_finish_as_text(self):
        for stream in (False, True):
            for tail in ('<tool_call><function=Write><parameter=content>unfinished',
                         '<tool_call><function=Missing></function></tool_call>'):
                with self.subTest(stream=stream, tail=tail):
                    response = self.send('我现在开始处理。' + tail, stream=stream)
                    if stream:
                        events = self.events(response)
                        self.assertTrue([e for e in events if e['type'] == 'error'], response.text)
                        self.assertFalse([e for e in events if e['type'] == 'message_stop'])
                    else:
                        self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(self.instance.conversation_handles, {})

    def test_budget_truncation_remains_recoverable(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.send('我现在开始。<tool_call><function=Write><parameter=content>unfinished',
                                     stream=stream, max_tokens=1)
                self.assertEqual(response.status_code, 200, response.text)
                if stream:
                    events = self.events(response)
                    self.assertEqual(events[-2]['delta']['stop_reason'], 'max_tokens')
                    self.assertEqual(events[-1]['type'], 'message_stop')
                else:
                    self.assertEqual(response.json()['stop_reason'], 'max_tokens')

    def test_final_buffered_call_follows_an_already_emitted_call(self):
        def call(path):
            return ('<tool_call><function=Write><parameter=file_path>' + path
                    + '</parameter><parameter=content>done</parameter></function>')
        response = self.send(call('first.txt') + '</tool_call>' + call('last.txt'))
        events = self.events(response)
        starts = [e for e in events if e['type'] == 'content_block_start'
                  and e['content_block']['type'] == 'tool_use']
        self.assertEqual([e['index'] for e in starts], [0, 1])
        self.assertEqual(len({e['content_block']['id'] for e in starts}), 2)
        for event, path in zip(starts, ['first.txt', 'last.txt']):
            args = ''.join(e['delta'].get('partial_json', '') for e in events
                           if e['type'] == 'content_block_delta' and e['index'] == event['index'])
            self.assertEqual(json.loads(args)['file_path'], path)
        self.assertEqual(events[-2]['delta']['stop_reason'], 'tool_use')

    def test_ordinary_final_answer_preserves_pending_marker_text(self):
        response = self.send('比较 a < b，以及末尾的 <tool_')
        events = self.events(response)
        text = ''.join(e['delta'].get('text', '') for e in events if e['type'] == 'content_block_delta')
        self.assertEqual(text, self.model.output)
        self.assertEqual(events[-2]['delta']['stop_reason'], 'end_turn')


if __name__ == '__main__':
    unittest.main()
