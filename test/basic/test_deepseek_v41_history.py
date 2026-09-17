import copy
import json
import unittest

from ftllm.encoding_dsv41 import encode_messages, eos_token
from ftllm.deepseek_v41_history import ToolHistory


class HistoryTest(unittest.TestCase):
    def message(self, call_id='c1'):
        return dict(role='assistant', content=None, tool_calls=[dict(
            id=call_id, type='function', function=dict(name='exec_command',
            arguments=json.dumps({'cmd': 'pwd', 'workdir': '/tmp'})))])

    def test_exact_noncanonical_roundtrip(self):
        from ftllm.openai_server.tool_parsers.deepseekv41_tool_parser import DeepSeekV41ToolParser
        from types import SimpleNamespace
        parser = DeepSeekV41ToolParser(SimpleNamespace(get_vocab=lambda: {}))
        base = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="exec_command">\n<｜DSML｜ parameter name="workdir" string="true">/tmp</｜DSML｜ parameter>\n<｜DSML｜ parameter name="cmd" string="true">pwd</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
        for raw in (base, '\n'+base, '\n\n'+base, base.replace('｜DSML｜', '\\DSML\\')):
            extracted = parser.extract_tool_calls(raw, SimpleNamespace(tools=[]))
            msg = dict(role='assistant', content=extracted.content, tool_calls=[c.model_dump() for c in extracted.tool_calls])
            self.assertTrue(msg['tool_calls'])
            cache = ToolHistory()
            cache.remember(raw, msg)
            if not (msg.get('content') or '').strip():
                msg['content'] = None
            for call in msg['tool_calls']:
                call['function']['arguments'] = json.dumps(json.loads(call['function']['arguments']), sort_keys=True)
            messages = [dict(role='user',content='inspect'), msg,
                        dict(role='tool',tool_call_id=msg['tool_calls'][0]['id'],content='result')]
            prefix = encode_messages(messages[:1], thinking_mode='chat')
            restored = encode_messages(messages, thinking_mode='chat', tool_history=cache)
            self.assertTrue(restored.startswith(prefix+raw+eos_token))
            self.assertNotEqual(encode_messages(messages, thinking_mode='chat'), restored)

    def test_validate_ids_content_and_arguments(self):
        m = self.message(); cache = ToolHistory(); cache.remember('raw', m)
        self.assertEqual(cache.restore(m), 'raw')
        changed = copy.deepcopy(m); changed['tool_calls'][0]['function']['arguments'] = {'workdir':'/tmp','cmd':'pwd'}
        self.assertEqual(cache.restore(changed), 'raw')
        for key, val in [('id','other'), ('function',dict(name='other',arguments='{}'))]:
            changed = copy.deepcopy(m); changed['tool_calls'][0][key] = val
            self.assertIsNone(cache.restore(changed))
        changed = copy.deepcopy(m); changed['content'] = 'changed'
        self.assertIsNone(cache.restore(changed))
        changed = copy.deepcopy(m); changed['tool_calls'][0]['function']['arguments']='{"cmd":"rm"}'
        self.assertIsNone(cache.restore(changed))
        self.assertIsNone(ToolHistory().restore(m))

    def test_parallel_order_and_bounds(self):
        cache = ToolHistory(max_entries=1)
        a=self.message('a'); b=self.message('b')
        cache.remember('first',a); cache.remember('second',b)
        self.assertIsNone(cache.restore(a)); self.assertEqual(cache.restore(b),'second')
        cache=ToolHistory(max_bytes=5); cache.remember('large',a)
        self.assertEqual(cache.bytes,0)
        a['tool_calls'] += b['tool_calls']; cache=ToolHistory(); cache.remember('parallel',a)
        a['tool_calls'].reverse(); self.assertIsNone(cache.restore(a))

    def test_thinking_history_without_client_reasoning(self):
        msg = self.message()
        raw = 'original thought</think>\n\n<｜DSML｜ calls>exact output</｜DSML｜ calls>'
        saved = dict(msg, reasoning_content='original thought')
        cache = ToolHistory()
        cache.remember(raw, saved, thinking=True)
        self.assertEqual(cache.restore(msg, thinking=True), raw)
        self.assertEqual(cache.restore(saved, thinking=True), raw)
        self.assertIsNone(cache.restore(msg))
        self.assertIsNone(cache.restore(dict(msg, reasoning_content='edited'), thinking=True))
        user = dict(role='user', content='inspect')
        messages = [user, msg, dict(role='tool', tool_call_id='c1', content='result')]
        prefix = encode_messages([user], thinking_mode='thinking')
        restored = encode_messages(messages, thinking_mode='thinking', tool_history=cache,
                                   drop_thinking=False)
        self.assertTrue(restored.startswith(prefix + raw + eos_token))
        later = messages + [dict(role='user', content='next question')]
        self.assertNotIn('original thought', encode_messages(
            later, thinking_mode='thinking', tool_history=cache, drop_thinking=True))

    def test_no_client_override_or_thinking_replay(self):
        m=self.message(); m['_raw_content']='injected'; cache=ToolHistory(); cache.remember('raw',m)
        self.assertNotIn('injected',encode_messages([dict(role='user',content='hi'),m],thinking_mode='chat'))
        self.assertNotIn('raw',encode_messages([dict(role='user',content='hi'),m],thinking_mode='thinking',tool_history=cache))

class ServerHistoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_full_and_stream_preserve_thinking_for_codex(self):
        from types import SimpleNamespace
        from ftllm.openai_server.fastllm_completion import FastLLmCompletion
        from ftllm.openai_server.protocal.openai_protocol import ChatCompletionRequest
        raw = ('Check the directory first.</think>\n'
               '<｜DSML｜ calls>\n<｜DSML｜ invoke name="exec_command">\n'
               '<｜DSML｜ parameter name="cmd" string="true">pwd</｜DSML｜ parameter>\n'
               '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        request = ChatCompletionRequest(model='test', messages=[dict(role='user', content='inspect')],
            tools=[dict(type='function', function=dict(name='exec_command', parameters=dict(
                type='object', properties=dict(cmd=dict(type='string')), required=['cmd'])))])
        async def connected():
            return False
        for stream in (False, True):
            for chunk_size in (1, 7, len(raw)):
                cache = ToolHistory()
                registered = []
                def remember(text, content, tool_calls, thinking=False, reasoning_content=None):
                    message = dict(role='assistant', content=content, tool_calls=tool_calls,
                                   reasoning_content=reasoning_content)
                    cache.remember(text, message, thinking=thinking)
                    registered.append(message)
                completion = FastLLmCompletion.__new__(FastLLmCompletion)
                completion.model_name = 'test'
                completion.conversation_handles = {}
                completion.model = SimpleNamespace(
                    _is_deepseek_v41=lambda: True,
                    get_type=lambda: 'deepseek_v41', tool_call_parser='deepseek_v41',
                    force_chat_template=False, hf_tokenizer=SimpleNamespace(get_vocab=lambda: {}),
                    remember_deepseek_v41_tool_output=remember)
                async def generated():
                    for start in range(0, len(raw), chunk_size):
                        yield raw[start:start+chunk_size]
                kwargs = dict(request=request, raw_request=SimpleNamespace(is_disconnected=connected),
                    result_generator=generated(), request_id='test', input_token_len=3,
                    think=False, emit_reasoning_content=True)
                if stream:
                    async for _ in completion.chat_completion_stream_generator(**kwargs):
                        pass
                else:
                    await completion.chat_completion_full_generator(handle=0, **kwargs)
                self.assertEqual(len(registered), 1)
                msg = registered[0]
                self.assertEqual(msg.pop('reasoning_content'), 'Check the directory first.')
                self.assertEqual(cache.restore(msg, thinking=True), raw)
                prefix_messages = [dict(role='system', content='', tools=request.model_dump()['tools']),
                                   dict(role='user', content='inspect')]
                prefix = encode_messages(prefix_messages, thinking_mode='thinking')
                continuation = encode_messages(prefix_messages + [msg, dict(
                    role='tool', tool_call_id=msg['tool_calls'][0]['id'], content='/tmp')],
                    thinking_mode='thinking', tool_history=cache)
                self.assertTrue(continuation.startswith(prefix + raw + eos_token))

if __name__=='__main__': unittest.main()
