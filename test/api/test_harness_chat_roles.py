"""Exercise Harness developer instructions against the Qwen chat template."""
import copy
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from jinja2.exceptions import TemplateError
from test_anthropic_messages import CapturingModel
from test_qwen35_reasoning import FakeQwen35Model, completion
from tools.fastllm_pytools import server


class HarnessChatRolesTest(unittest.TestCase):
    def setUp(self):
        self.model = CapturingModel(output='The answer is 42.')
        instance = completion(self.model)
        instance.enable_thinking = False
        mocked = patch.object(server, 'fastllm_completion', instance, create=True)
        mocked.start()
        self.addCleanup(mocked.stop)
        self.client = TestClient(server.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def request(self, messages, stream=False):
        return self.client.post('/v1/chat/completions', json={
            'model': 'qwen3.5', 'max_completion_tokens': 128,
            'messages': messages, 'stream': stream})

    def assert_ok(self, response, stream):
        self.assertEqual(response.status_code, 200, response.text)
        if stream:
            self.assertIn('data: [DONE]', response.text)
        else:
            self.assertEqual(response.json()['choices'][0]['message']['content'], self.model.output)
        self.assertEqual(self.model.counted_messages, self.model.generated_messages)
        self.assertEqual(self.model.counted_prompt, self.model.generated_prompt)

    def test_harness_developer_prompt_accepts_text_and_blocks(self):
        for stream in (False, True):
            for content in ('Use read to inspect files.', [{'type': 'text', 'text': 'Use read to inspect files.'}]):
                with self.subTest(stream=stream, content=content):
                    messages = [{'role': 'developer', 'content': content},
                                {'role': 'user', 'content': 'Read fixture.txt.'}]
                    original = copy.deepcopy(messages)
                    self.assert_ok(self.request(messages, stream), stream)
                    self.assertEqual(self.model.generated_messages[0],
                                     {'role': 'system', 'content': 'Use read to inspect files.'})
                    self.assertEqual(messages, original)

    def test_mixed_instructions_preserve_tool_history(self):
        messages = [
            {'role': 'system', 'content': 'Follow project instructions.'},
            {'role': 'developer', 'content': 'Use read.'},
            {'role': 'user', 'content': 'Read fixture.txt.'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'Read the file.',
             'tool_calls': [{'id': 'call_read', 'type': 'function', 'function': {
                 'name': 'read', 'arguments': '{"file_path":"fixture.txt"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_read', 'content': 'answer=42'},
            {'role': 'developer', 'content': 'Report the answer.'},
            {'role': 'user', 'content': 'Continue.'},
        ]
        expected = [
            {'role': 'system', 'content': 'Follow project instructions.\nUse read.\nReport the answer.'},
            *copy.deepcopy(messages[2:5]), copy.deepcopy(messages[6])]
        expected[2]['tool_calls'][0]['function']['arguments'] = {'file_path': 'fixture.txt'}
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assert_ok(self.request(messages, stream), stream)
                self.assertEqual(self.model.generated_messages, expected)
                self.assertIn('<tool_response>\nanswer=42\n</tool_response>', self.model.generated_prompt)

    def test_other_model_families_keep_their_developer_role(self):
        model = FakeQwen35Model(output='42')
        model.get_type = lambda: 'deepseek_v4'
        instance = completion(model)
        instance.enable_thinking = False
        with patch.object(server, 'fastllm_completion', instance), \
             patch.object(model, 'launch_stream_response', wraps=model.launch_stream_response) as launch:
            response = self.request([{'role': 'developer', 'content': 'Keep native role.'},
                                     {'role': 'user', 'content': 'Answer.'}])
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(launch.call_args.args[0][0]['role'], 'developer')

    def test_template_errors_are_actionable_client_errors(self):
        for stream in (False, True):
            for method in ('get_input_token_len', 'launch_stream_response'):
                with self.subTest(stream=stream, method=method):
                    with patch.object(self.model, method, side_effect=TemplateError('Unsupported template input')):
                        response = self.request([{'role': 'user', 'content': 'Answer.'}], stream)
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(response.json()['object'], 'error')
                    self.assertEqual(response.json()['code'], 400)
                    self.assertIn('Unsupported template input', response.json()['message'])
                    self.assertEqual(server.fastllm_completion.conversation_handles, {})


if __name__ == '__main__':
    unittest.main()
