"""Check Harness/OpenAI token budgets through the production HTTP route."""
import json
import unittest
from unittest.mock import patch

from test_qwen35_reasoning import FakeQwen35Model, completion
from fastapi.testclient import TestClient
from tools.fastllm_pytools import server


class CapturingModel(FakeQwen35Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured_launch_kwargs = None

    def launch_stream_response(self, *args, **kwargs):
        self.captured_launch_kwargs = dict(kwargs)
        kwargs.pop("session_key", None)
        return super().launch_stream_response(*args, **kwargs)


class CompletionTokenLimitTest(unittest.TestCase):
    def setUp(self):
        self.model = CapturingModel(output='hello')
        instance = completion(self.model)
        instance.enable_thinking = False
        mocked = patch.object(server, 'fastllm_completion', instance, create=True)
        mocked.start()
        self.addCleanup(mocked.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def request(self, **fields):
        return self.client.post('/v1/chat/completions', json={
            'model': 'qwen3.5', 'messages': [{'role': 'user', 'content': 'hello'}], **fields})

    def test_session_metadata_reaches_launch_without_leaking(self):
        for route, body in (
            ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'hello'}]}),
            ('/v1/responses', {'input': 'hello'}),
            ('/v1/messages', {'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 16}),
        ):
            for headers, expected in (({'X-Session-Id': 'a'}, 'a'),
                                      ({'X-SMG-Routing-Key': 'b'}, 'b'), ({}, None)):
                with self.subTest(route=route, expected=expected):
                    response = self.client.post(route, headers=headers, json={'model': 'qwen3.5', **body})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(self.model.captured_launch_kwargs.get('session_key'), expected)

    def test_modern_and_legacy_fields_reach_native_generation(self):
        for stream in (False, True):
            for field in ('max_completion_tokens', 'max_tokens'):
                with self.subTest(stream=stream, field=field):
                    response = self.request(stream=stream, **{field: 8192})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(self.model.captured_launch_kwargs['max_length'], 8192)
                    if stream:
                        self.assertIn('data: [DONE]', response.text)
                    else:
                        self.assertEqual(response.json()['choices'][0]['message']['content'], 'hello')

    def test_modern_budget_controls_the_terminal_finish_reason(self):
        response = self.request(max_completion_tokens=1)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['choices'][0]['finish_reason'], 'length')

    def test_default_and_unlimited_budgets_report_natural_chat_completion(self):
        self.model.get_response_statistics = lambda handle: {"output_tokens": 32769}
        for stream in (False, True):
            for fields in ({}, {"max_tokens": None}, {"max_tokens": 0}, {"max_tokens": -1},
                           {"max_completion_tokens": None}):
                with self.subTest(stream=stream, fields=fields):
                    self._check_unlimited_chat(stream, fields)

    def _check_unlimited_chat(self, stream, fields):
        response = self.request(stream=stream, **fields)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.model.captured_launch_kwargs['max_length'], -1)
        if stream:
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            reasons = [c['finish_reason'] for event in events
                       for c in event.get('choices', []) if c.get('finish_reason')]
            self.assertEqual(reasons, ['stop'])
            usage = [event['usage'] for event in events if event.get('usage')][-1]
            self.assertEqual(usage['completion_tokens'], 32769)
            self.assertIn('data: [DONE]', response.text)
        else:
            self.assertEqual(response.json()['choices'][0]['finish_reason'], 'stop')
            self.assertEqual(response.json()['usage']['completion_tokens'], 32769)

    def test_default_and_unlimited_responses_budgets_are_not_incomplete(self):
        self.model.get_response_statistics = lambda handle: {"output_tokens": 32769}
        for stream in (False, True):
            for fields in ({}, {"max_output_tokens": None}, {"max_output_tokens": -1}):
                with self.subTest(stream=stream, fields=fields):
                    self._check_unlimited_responses(stream, fields)

    def _check_unlimited_responses(self, stream, fields):
        response = self.client.post('/v1/responses', json={
            'model': 'qwen3.5', 'input': 'hello', 'stream': stream, **fields})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.model.captured_launch_kwargs['max_length'], -1)
        if stream:
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith('data: ')]
            terminal = [event for event in events
                        if event['type'] in ('response.completed', 'response.incomplete')]
            self.assertEqual(len(terminal), 1)
            self.assertEqual(terminal[0]['type'], 'response.completed')
            result = terminal[0]['response']
        else:
            result = response.json()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['usage']['output_tokens'], 32769)
        self.assertIsNone(result.get('incomplete_details'))

    def test_unlimited_anthropic_budget_reports_end_turn(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.client.post('/v1/messages', json={
                    'model': 'qwen3.5', 'messages': [{'role': 'user', 'content': 'hello'}],
                    'max_tokens': -1, 'stream': stream})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.model.captured_launch_kwargs['max_length'], -1)
                if stream:
                    events = [json.loads(line[6:]) for line in response.text.splitlines()
                              if line.startswith('data: ')]
                    reasons = [event['delta']['stop_reason'] for event in events
                               if event['type'] == 'message_delta']
                    self.assertEqual(reasons, ['end_turn'])
                else:
                    self.assertEqual(response.json()['stop_reason'], 'end_turn')

    def test_modern_limit_wins_when_both_fields_are_present(self):
        response = self.request(max_completion_tokens=64, max_tokens=128)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.model.captured_launch_kwargs['max_length'], 64)

    def test_null_or_missing_modern_field_uses_unlimited_or_legacy_budget(self):
        for fields, expected in [({}, -1), ({'max_completion_tokens': None}, -1),
                                 ({'max_completion_tokens': None, 'max_tokens': 128}, 128)]:
            with self.subTest(fields=fields):
                response = self.request(**fields)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.model.captured_launch_kwargs['max_length'], expected)

    def test_invalid_modern_limit_is_rejected_before_launch(self):
        for value in (0, -1):
            with self.subTest(value=value):
                response = self.request(max_completion_tokens=value)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIsNone(self.model.captured_launch_kwargs)


if __name__ == '__main__':
    unittest.main()
