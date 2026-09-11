"""Check Harness/OpenAI token budgets through the production HTTP route."""
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

    def test_modern_limit_wins_when_both_fields_are_present(self):
        response = self.request(max_completion_tokens=64, max_tokens=128)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.model.captured_launch_kwargs['max_length'], 64)

    def test_null_or_missing_modern_field_preserves_existing_default(self):
        for fields, expected in [({}, 32768), ({'max_completion_tokens': None}, 32768),
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
