"""Counted chat requests must launch those exact IDs with native KV history on."""
from collections import OrderedDict
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# Always exercise this checkout, not a possibly older installed ftllm package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
NATIVE_LIBRARIES = ('libfastllm_tools.so', 'libfastllm_tools-cu11.so',
                    'libfastllm_tools-cpu.so', 'libfastllm_tools.dylib', 'fastllm_tools.dll')
if not any((REPO_ROOT / 'tools/fastllm_pytools' / name).is_file()
           for name in NATIVE_LIBRARIES):
    raise unittest.SkipTest('Build the native library in tools/fastllm_pytools first')
from tools.fastllm_pytools import llm


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        # A merge spanning an old text-cache boundary must not replace the
        # tokenizer's full-prompt result with separately encoded fragments.
        result = []
        while text:
            if text.startswith('ab'):
                result.append(1000)
                text = text[2:]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = ''.join(message.get('content') or '' for message in messages)
        for message in messages:
            if message.get('tool_calls'):
                text += json.dumps(message['tool_calls'])
        if kwargs.get('tools'):
            text += json.dumps(kwargs['tools'])
        return self.encode(text) if tokenize else text


class CountedHistoryTokensTest(unittest.TestCase):
    def setUp(self):
        self.ids = []

        def launch(model_id, count, ids, *args):
            self.ids.append(list(ids[:count]))
            return 42

        self.native = SimpleNamespace(launch_response_llm_model=launch,
                                      set_save_history=Mock())
        native_patch = patch.object(llm, 'fastllm_lib', self.native)
        native_patch.start()
        self.addCleanup(native_patch.stop)
        self.model = llm.model.__new__(llm.model)
        m = self.model
        m.model = 1
        m.hf_tokenizer = Tokenizer()
        m.enable_thinking = True
        m.thread_local_obj = threading.local()
        m.text_input_token_cache_limit = 4
        m.text_input_token_cache = OrderedDict()
        m.text_input_token_inflight = {}
        m.text_input_token_cache_lock = threading.Lock()
        m.tokenizer_cache = llm.TokenizerCache()
        m.current_tokenizer_cache = {}
        m.config = {}
        m.get_struct = lambda: 'naive_n05_flash'
        m._can_apply_hf_chat_template = lambda: True
        m._uses_hf_deepseek_v4_tokenizer = lambda: False
        m._is_kimi_k3 = lambda: False
        m._apply_tool_call_constraint_to_native = lambda constraint: None
        m.set_save_history(True)
        self.messages = [{'role': 'user', 'content': 'ab'}]

    def test_counted_ids_win_over_old_fragment_cache(self):
        m = self.model
        m.tokenizer_cache.add(['a'], [[ord('a')]])
        m.current_tokenizer_cache[42] = [['stale request'], [[7]]]
        self.assertEqual(m.get_input_token_len(self.messages), 1)
        with patch.object(m.tokenizer_cache, 'tokenize_with_cache',
                          side_effect=AssertionError('Repeated tokenization')):
            m.launch_stream_response(self.messages)
        self.assertEqual(self.ids, [[1000]])
        self.assertNotIn(42, m.current_tokenizer_cache)
        self.assertTrue(m.save_history)
        self.native.set_save_history.assert_called_once_with(1, True)
        self.assertIsNone(m.thread_local_obj.pending_text_input_token_cache)

    def test_normalized_tool_arguments_and_unicode(self):
        messages = [{'role': 'user', 'content': '中文🙂'},
                    {'role': 'assistant', 'content': '', 'tool_calls': [{
                        'type': 'function', 'function': {
                            'name': 'read', 'arguments': '{"path":"/tmp/a"}'}}]}]
        m = self.model
        m.get_input_token_len(messages)
        expected = m.thread_local_obj.pending_text_input_token_cache['input_ids'][:]
        with patch.object(m.tokenizer_cache, 'tokenize_with_cache',
                          side_effect=AssertionError('Repeated tokenization')):
            m.launch_stream_response(messages)
        self.assertEqual(self.ids[-1], expected)

    def test_changed_request_rejects_one_shot_ids(self):
        changes = [{'enable_thinking': False}, {'add_generation_prompt': False},
                   {'tools': [{'type': 'function', 'function': {'name': 'read'}}]},
                   {'thinking_effort': 'high'}, {'tool_choice': 'auto'},
                   {'chat_template_kwargs': {'keep_all_reasoning': False}}]
        for kwargs in changes:
            with self.subTest(kwargs=kwargs):
                m = self.model
                m.get_input_token_len(self.messages)
                with patch.object(m.tokenizer_cache, 'tokenize_with_cache',
                                  wraps=m.tokenizer_cache.tokenize_with_cache) as fallback:
                    m.launch_stream_response(self.messages, **kwargs)
                fallback.assert_called_once()
                self.assertIsNone(m.thread_local_obj.pending_text_input_token_cache)
                self.assertTrue(m.current_tokenizer_cache[42][0][0])
        m.get_input_token_len(self.messages)
        m.launch_stream_response([{'role': 'user', 'content': 'different'}])
        self.assertEqual(self.ids[-1], m.hf_tokenizer.encode('different'))

    def test_uncounted_and_second_launch_keep_legacy_fallback(self):
        m = self.model
        m.get_input_token_len(self.messages)
        m.launch_stream_response(self.messages)
        with patch.object(m.tokenizer_cache, 'tokenize_with_cache',
                          wraps=m.tokenizer_cache.tokenize_with_cache) as fallback:
            m.launch_stream_response(self.messages)
        fallback.assert_called_once()
        self.assertEqual(self.ids, [[1000], [1000]])
        self.assertEqual(m.current_tokenizer_cache[42], [['ab'], [[1000]]])

    def test_history_disabled_still_reuses_counted_ids(self):
        m = self.model
        m.set_save_history(False)
        m.get_input_token_len(self.messages)
        with patch.object(m.hf_tokenizer, 'apply_chat_template',
                          side_effect=AssertionError('Repeated template rendering')):
            m.launch_stream_response(self.messages)
        self.assertEqual(self.ids, [[1000]])
        self.assertEqual(m.current_tokenizer_cache, {})

    def test_tool_property_order_does_not_reuse_other_rendered_ids(self):
        m = self.model
        expected = []
        for properties in ({'first': 1, 'second': 2}, {'second': 2, 'first': 1}):
            tools = [{'type': 'function', 'function': {
                'name': 'read', 'parameters': {'properties': properties}}}]
            expected.append(m.hf_tokenizer.apply_chat_template(
                self.messages, tokenize=True, tools=tools))
            m.get_input_token_len(self.messages, tools=tools)
            m.launch_stream_response(self.messages, tools=tools)
        self.assertNotEqual(expected[0], expected[1])
        self.assertEqual(self.ids, expected)

    def test_counted_ids_are_thread_local(self):
        barrier = threading.Barrier(2)
        errors = []

        def request(text):
            try:
                messages = [{'role': 'user', 'content': text}]
                self.model.get_input_token_len(messages)
                barrier.wait(timeout=5)
                self.model.launch_stream_response(messages)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=request, args=(text,)) for text in ('ab', 'xy')]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertCountEqual(self.ids, [[1000], [ord('x'), ord('y')]])

    def test_reordered_tool_inputs_reject_one_shot_ids(self):
        first, second = {'first': 1, 'second': 2}, {'second': 2, 'first': 1}
        messages = lambda arguments: [{'role': 'assistant', 'content': '',
            'tool_calls': [{'type': 'function', 'function': {
                'name': 'read', 'arguments': arguments}}]}]
        tools = lambda properties: [{'type': 'function', 'function': {
            'name': 'read', 'parameters': {'properties': properties}}}]
        cases = [(messages(first), {}, messages(second), {}),
                 (self.messages, {'tools': tools(first)}, self.messages, {'tools': tools(second)}),
                 (self.messages, {'chat_template_kwargs': first}, self.messages, {'chat_template_kwargs': second})]
        for before, count_args, after, launch_args in cases:
            with self.subTest(launch_args=launch_args, messages=after):
                m = self.model
                m.get_input_token_len(before, **count_args)
                with patch.object(m.tokenizer_cache, 'tokenize_with_cache',
                                  wraps=m.tokenizer_cache.tokenize_with_cache) as fallback:
                    m.launch_stream_response(after, **launch_args)
                fallback.assert_called_once()
                self.assertEqual(self.ids[-1], m.hf_tokenizer.apply_chat_template(
                    m.trans_conversation(after), tokenize=True, **launch_args))

    def test_thinking_alias_matches_launch_template(self):
        class AliasTokenizer(Tokenizer):
            def apply_chat_template(self, messages, tokenize=False, **kwargs):
                text = 'thinking on' if kwargs.get('thinking', True) else 'thinking off'
                return self.encode(text) if tokenize else text

        m = self.model
        m.hf_tokenizer = AliasTokenizer()
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                m.get_input_token_len(self.messages, enable_thinking=thinking)
                m.launch_stream_response(self.messages, enable_thinking=thinking)
                expected = llm.apply_hf_chat_template(m.hf_tokenizer, self.messages,
                    tokenize=True, enable_thinking=thinking, thinking_alias=True)
                self.assertEqual(self.ids[-1], expected)


if __name__ == '__main__':
    unittest.main()
