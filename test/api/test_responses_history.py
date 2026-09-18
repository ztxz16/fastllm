import os
import sys
import unittest
from types import SimpleNamespace

TEST_API_DIR = os.path.abspath(os.path.dirname(__file__))
ORIGINAL_SYS_PATH = list(sys.path)
sys.path = [path for path in sys.path
            if os.path.abspath(path or os.getcwd()) != TEST_API_DIR]
sys.path.insert(0, os.path.abspath(os.path.join(TEST_API_DIR, '..', '..', 'tools')))

from fastllm_pytools.openai_server.fastllm_completion import FastLLmCompletion
from fastllm_pytools.openai_server.protocal.openai_protocol import ResponsesRequest
from fastllm_pytools.encoding_dsv41 import encode_messages, eos_token
sys.path[:] = ORIGINAL_SYS_PATH


def message(role, text):
    return {'type': 'message', 'role': role,
            'content': [{'type': 'output_text' if role == 'assistant' else 'input_text', 'text': text}]}


def call(call_id, name, arguments):
    return {'type': 'function_call', 'call_id': call_id,
            'name': name, 'arguments': arguments}


class ResponsesHistoryTest(unittest.TestCase):
    def convert(self, items, *, native_v41=False, force_chat_template=False,
                instructions=None):
        completion = object.__new__(FastLLmCompletion)
        completion.model = SimpleNamespace(
            _is_deepseek_v41=lambda: native_v41,
            force_chat_template=force_chat_template)
        request = ResponsesRequest(model='test-model', input=items,
                                   instructions=instructions)
        return completion._build_chat_request_from_responses(request).messages

    def test_v41_new_turn_instructions_preserve_cached_prefix(self):
        history = [message('developer', 'Use the project tools.'),
                   message('user', 'First question.')]
        first = self.convert(history, native_v41=True, instructions='Be concise.')
        second = self.convert(history + [
            message('assistant', 'First answer.'),
            message('developer', 'Additional skills for this turn.'),
            message('user', 'Next question.'),
        ], native_v41=True, instructions='Be concise.')
        self.assertEqual(second[:len(first)], first)
        self.assertEqual(second[0], {
            'role': 'system', 'content': 'Be concise.\n\nUse the project tools.'})
        self.assertEqual([m['role'] for m in second],
                         ['system', 'user', 'assistant', 'system', 'user'])
        self.assertEqual(second[3]['content'], 'Additional skills for this turn.')
        first_prompt = encode_messages(first, thinking_mode='chat')
        second_prompt = encode_messages(second, thinking_mode='chat')
        self.assertTrue(second_prompt.startswith(
            first_prompt + 'First answer.' + eos_token))
        self.assertIn('<｜System｜>Additional skills for this turn.', second_prompt)

    def test_v41_midconversation_instruction_keeps_tool_history(self):
        messages = self.convert([
            message('user', 'Inspect the file.'),
            call('read', 'read_file', '{"path":"a.py"}'),
            {'type': 'function_call_output', 'call_id': 'read', 'output': 'source'},
            message('developer', 'Report the result without editing.'),
        ], native_v41=True)
        self.assertEqual([m['role'] for m in messages],
                         ['user', 'assistant', 'tool', 'system'])
        self.assertEqual(messages[1]['tool_calls'][0]['id'], 'read')
        self.assertEqual(messages[2]['tool_call_id'], 'read')
        self.assertTrue(encode_messages(messages, thinking_mode='chat').endswith(
            '<｜System｜>Report the result without editing.<｜Assistant｜></think>'))

    def test_other_templates_keep_leading_system_normalization(self):
        for native_v41, force_chat_template in [(False, False), (True, True)]:
            with self.subTest(native_v41=native_v41, force_chat_template=force_chat_template):
                messages = self.convert([
                    message('user', 'Question.'),
                    message('developer', 'Additional instructions.'),
                ], native_v41=native_v41, force_chat_template=force_chat_template,
                    instructions='Initial instructions.')
                self.assertEqual(messages, [
                    {'role': 'system', 'content':
                     'Initial instructions.\n\nAdditional instructions.'},
                    {'role': 'user', 'content': 'Question.'},
                ])

    def test_progress_and_parallel_calls_remain_one_assistant_turn(self):
        messages = self.convert([
            message('user', 'Read both files, then fix the failing test.'),
            message('assistant', 'I will inspect both files.'),
            call('call_a', 'read_file', '{"path":"a.py"}'),
            call('call_b', 'read_file', '{"path":"test_a.py"}'),
            {'type': 'function_call_output', 'call_id': 'call_a', 'output': 'source'},
            {'type': 'function_call_output', 'call_id': 'call_b', 'output': 'test'},
        ])
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'tool', 'tool'])
        self.assertEqual(messages[1]['content'], 'I will inspect both files.')
        self.assertEqual(messages[1]['tool_calls'], [
            {'id': 'call_a', 'type': 'function', 'function': {
                'name': 'read_file', 'arguments': '{"path":"a.py"}'}},
            {'id': 'call_b', 'type': 'function', 'function': {
                'name': 'read_file', 'arguments': '{"path":"test_a.py"}'}},
        ])
        self.assertEqual([m['tool_call_id'] for m in messages[2:]], ['call_a', 'call_b'])

    def test_call_before_text_joins_the_same_assistant_turn(self):
        messages = self.convert([
            message('user', 'Run tests.'),
            call('call_test', 'run_tests', '{}'),
            message('assistant', 'Running the tests now.'),
        ])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]['content'], 'Running the tests now.')
        self.assertEqual(messages[1]['tool_calls'][0]['id'], 'call_test')

    def test_tool_result_and_next_user_keep_turn_boundaries(self):
        messages = self.convert([
            message('user', 'Run tests.'),
            call('call_first', 'run_tests', '{}'),
            {'type': 'function_call_output', 'call_id': 'call_first', 'output': 'one failure'},
            message('assistant', 'I will inspect the failure.'),
            call('call_read', 'read_file', '{"path":"test_a.py"}'),
            {'type': 'function_call_output', 'call_id': 'call_read', 'output': 'test'},
            message('assistant', 'The fix is ready.'),
            message('user', 'Run the tests again.'),
            call('call_second', 'run_tests', '{}'),
        ])
        self.assertEqual([m['role'] for m in messages],
                         ['user', 'assistant', 'tool', 'assistant', 'tool', 'assistant', 'user', 'assistant'])
        self.assertEqual(messages[3]['tool_calls'][0]['id'], 'call_read')
        self.assertEqual(messages[-1]['tool_calls'][0]['id'], 'call_second')

    def test_separate_text_messages_are_not_combined(self):
        messages = self.convert([
            message('user', 'Continue.'),
            message('assistant', 'Previous status.'),
            message('assistant', 'Next action.'),
            call('call_next', 'run_tests', '{}'),
        ])
        self.assertEqual(len(messages), 3)
        self.assertNotIn('tool_calls', messages[1])
        self.assertEqual(messages[2]['content'], 'Next action.')
        self.assertEqual(messages[2]['tool_calls'][0]['id'], 'call_next')


if __name__ == '__main__':
    unittest.main()
