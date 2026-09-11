"""Run actual isolated Harness against controlled SSE cases; no model/GPU requests."""
import tempfile
import unittest
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / 'tools/fastllm_pytools/harness_recovery.mjs'
RUNTIME = Path(os.environ.get('FASTLLM_HARNESS_TEST_RUNTIME', '/nonexistent'))


def run(case, root, recover=True, max_recoveries=2, steering=None):
    directory = root / case
    directory.mkdir(parents=True, exist_ok=True)
    workspace = directory / 'workspace'
    workspace.mkdir(exist_ok=True)
    (workspace / 'fixture.txt').write_text('answer=42\n')
    requests = []
    mode = case.removesuffix('_exhausted')
    exhausted = case.endswith('_exhausted')
    emitted = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *_):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            (directory / 'requests.json').write_text(json.dumps(requests, indent=2))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'close')
            self.end_headers()
            def send(data):
                emitted.append(data)
                text = data if isinstance(data, str) else json.dumps(data)
                self.wfile.write(('data: ' + text + '\n\n').encode())
                self.wfile.flush()
            def chunk(delta, finish=None, out=20):
                data = {'id': 'chatcmpl-test', 'object': 'chat.completion.chunk',
                        'created': 1, 'model': body['model'],
                        'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                if finish:
                    data['usage'] = {'prompt_tokens': 9000, 'completion_tokens': out,
                                     'total_tokens': 9000 + out}
                send(data)
            def tool(arguments, finish):
                chunk({'tool_calls': [{'index': 0, 'id': 'call_read', 'type': 'function',
                                      'function': {'name': 'read', 'arguments': json.dumps(arguments)}}]})
                chunk({}, finish, out=8192 if finish=='length' else 20)
            def error(message, kind):
                send({'error': {'object': 'error', 'message': message, 'code': 400, 'type': kind}})
            try:
                chunk({'role': 'assistant'})
                user_text = '\n'.join(str(m.get('content', '')) for m in body['messages'] if m['role'] == 'user')
                if steering and 'USER_STEER_FIRST' in user_text:
                    chunk({'content': 'STEERING_LAST_RECEIVED' if 'USER_STEER_LAST' in user_text else 'STEERING_FIRST_RECEIVED'})
                    chunk({}, 'stop')
                elif len(requests) > 1 and not exhausted and mode in ('length_text', 'reasoning_only_stop', 'length_tool', 'invalid_tool_stream', 'whitespace_stop'):
                    if mode != 'length_text' and not any(m['role']=='tool' for m in body['messages']):
                        tool({'file_path': 'fixture.txt'}, 'tool_calls')
                    else:
                        chunk({'content': 'Recovered: the answer is 42.'})
                        chunk({}, 'stop')
                elif mode=='length_text':
                    chunk({'content': 'I will inspect the file and then'})
                    chunk({}, 'length', out=8192)
                elif mode=='length_tool':
                    tool({'file_path': 'fixture.txt'}, 'length')
                elif mode=='reasoning_only_stop':
                    chunk({'reasoning_content': 'I should inspect fixture.txt to answer.'})
                    chunk({}, 'stop')
                elif mode=='whitespace_stop':
                    chunk({'content': '  \n'})
                    chunk({}, 'stop')
                elif mode=='missing_finish':
                    chunk({'content': 'Starting to inspect the file...'})
                elif mode=='invalid_tool_stream':
                    # A streamed call must not execute when its attempt later fails.
                    chunk({'tool_calls': [{'index': 0, 'id': 'call_read', 'type': 'function',
                                          'function': {'name': 'read', 'arguments': '{"file_path":"fixture.txt"}'}}]})
                    error('Invalid tool call: incomplete_tool_call: tool call was not closed', 'invalid_tool_call')
                elif mode=='tool_schema_error':
                    if len(requests)==1:
                        tool({'file_path': 42}, 'tool_calls')
                    else:
                        chunk({'content': 'I received the tool validation error and can continue.'})
                        chunk({}, 'stop')
                elif mode=='context_overflow':
                    if len(requests)==1:
                        tool({'file_path': 'fixture.txt'}, 'tool_calls')
                    else:
                        error('Prompt too long: maximum context length is 32768 tokens; request needs 40000 tokens.', 'context_length_exceeded')
                elif mode=='empty_stop':
                    chunk({}, 'stop', out=0)
                send('[DONE]')
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                (directory / 'emitted.json').write_text(json.dumps(emitted, indent=2))
                self.close_connection = True

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = {'api': 'openai-completions', 'baseURL': f'http://127.0.0.1:{server.server_port}/v1',
                'apiKeyEnv': 'FTLLM_HARNESS_API_KEY', 'reasoning': 'xhigh',
                'models': [{'id': 'qwen3.8-fixture', 'contextWindow': 32768,
                            'maxTokens': 8192, 'reasoningEfforts': {'xhigh': 'xhigh'}}],
                'compat': {'supportsStore': False, 'supportsReasoningEffort': True,
                           'thinkingFormat': 'openai', 'supportsDeveloperRole': False,
                           'maxTokensField': 'max_tokens'},
                # Bound only this synthetic test. Product default remains five retries.
                'retryPolicy': {'mode': 'normal', 'maxRetries': 1,
                                'backoff': {'initialDelayMs': 20, 'maxDelayMs': 20, 'jitterRatio': 0}}}
    patch = [
        {'id': 'agent-default-model', 'config': {'provider': 'fastllm', 'model': 'qwen3.8-fixture'}},
        {'id': 'llm-pi-ai', 'config': {'providers': {'fastllm': provider}}},
        {'id': 'session-title-llm', 'disabled': True},
        {'id': 'session-log-deepseek', 'disabled': True},
        {'id': 'session-telemetry-otel', 'disabled': True},
    ]
    if recover:
        patch.insert(0, {'insert': [{'id': 'fastllm-harness-recovery', 'name': str(PLUGIN),
                                     'config': {'provider': 'fastllm', 'maxRecoveries': max_recoveries}}]})
    if steering:
        patch.insert(0, {'insert': [{'id': 'test-queue-steering',
                                     'name': str(Path(__file__).with_name('harness_queue_steering.mjs')),
                                     'config': {**steering, 'auditPath': str(directory/'steering-audit.json')}}]})
    (directory / 'patch.json').write_text(json.dumps(patch, indent=2))
    env = {k:v for k,v in os.environ.items() if not k.lower().endswith('_proxy') and not k.startswith(('DSH_', 'DEEPSEEK_', 'OPENAI_'))}
    env.update(DSH_HOME=str(directory/'home'), FTLLM_HARNESS_API_KEY='local-fixture-key', DSH_TELEMETRY_MODE='OFF')
    cmd = ['node', str(RUNTIME/'node_modules/@deepseek-ai/dsh/lib/bin.js'), '--profile', 'headless',
           '--patch', str(directory/'patch.json'), 'Read fixture.txt with the read tool and report the answer.']
    start = time.monotonic()
    with (directory/'stdout.log').open('w') as out, (directory/'stderr.log').open('w') as err:
        proc = subprocess.Popen(cmd, cwd=workspace, env=env, stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            code = proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:code=proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code=proc.wait()
    server.shutdown()
    server.server_close()
    result = {'case': case, 'returncode': code, 'driver_timeout': timed_out,
              'seconds': round(time.monotonic()-start, 3), 'requests': len(requests),
              'tool_results': [m for b in requests for m in b.get('messages', []) if m['role']=='tool'],
              'stdout': (directory/'stdout.log').read_text(),
              'stderr': (directory/'stderr.log').read_text()}
    (directory/'result.json').write_text(json.dumps(result, indent=2))
    return result


@unittest.skipUnless((RUNTIME/'node_modules/@deepseek-ai/dsh/lib/bin.js').is_file(),
                     'Set FASTLLM_HARNESS_TEST_RUNTIME to an installed official Harness runtime')
class HarnessRecoveryIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get('FASTLLM_HARNESS_TEST_OUTPUT')
        cls.temporary = None if configured else tempfile.TemporaryDirectory(prefix='harness-recovery-test-')
        cls.root = Path(configured or cls.temporary.name).resolve()
        cls.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.temporary: cls.temporary.cleanup()

    def case(self, name, code, requests):
        result = run(name, self.root)
        self.assertFalse(result['driver_timeout'], result)
        self.assertEqual(result['returncode'], code, result)
        self.assertEqual(result['requests'], requests, result)
        return result

    def test_zero_recovery_setting_is_honored_by_the_real_loader(self):
        result = run('reasoning_only_stop_exhausted', self.root / 'disabled', max_recoveries=0)
        self.assertFalse(result['driver_timeout'])
        self.assertEqual((result['returncode'], result['requests']), (1, 1), result)
        self.assertIn('after 0 attempts', result['stderr'])

    def steering_case(self, name, *, queue_at=1, target='next-step', max_recoveries=2):
        result = run(name, self.root / f'steering-{target}-{queue_at}-{max_recoveries}',
                     max_recoveries=max_recoveries,
                     steering={'queueAt': queue_at, 'target': target})
        self.assertFalse(result['driver_timeout'], result)
        self.assertEqual(result['returncode'], 0, result)
        self.assertEqual(result['requests'], queue_at + (2 if target == 'next-turn' else 1), result)
        self.assertEqual(result['tool_results'], [], result)
        self.assertEqual(result['stdout'].strip(), 'STEERING_LAST_RECEIVED', result)
        directory = self.root / f'steering-{target}-{queue_at}-{max_recoveries}' / name
        bodies = json.loads((directory/'requests.json').read_text())
        first = str(bodies[queue_at]['messages'])
        self.assertEqual(first.count('USER_STEER_FIRST'), 1)
        if target == 'next-step':
            self.assertEqual(first.count('USER_STEER_LAST'), 1)
            self.assertLess(first.index('USER_STEER_FIRST'), first.index('USER_STEER_LAST'))
        else:
            self.assertNotIn('USER_STEER_LAST', first)
            self.assertEqual(str(bodies[-1]['messages']).count('USER_STEER_LAST'), 1)
        self.assertFalse(any(m.get('tool_calls') or m['role']=='tool' for m in bodies[queue_at]['messages']))
        audit = json.loads((directory/'steering-audit.json').read_text())
        self.assertEqual(audit['accepted'], ['USER_STEER_FIRST', 'USER_STEER_LAST'])
        self.assertEqual(audit['claimed'], ['USER_STEER_FIRST', 'USER_STEER_LAST'])
        self.assertEqual(audit['preStep'], ['USER_STEER_FIRST', 'USER_STEER_LAST'])
        self.assertEqual(audit['discarded'], [])
        self.assertEqual(audit['pending'], [])

    def test_queued_steering_takes_precedence_over_error_recovery(self):
        self.steering_case('reasoning_only_stop_exhausted')

    def test_queued_followups_keep_their_individual_turns(self):
        self.steering_case('reasoning_only_stop_exhausted', target='next-turn')

    def test_queued_steering_is_processed_even_when_recovery_is_exhausted(self):
        self.steering_case('reasoning_only_stop_exhausted', queue_at=3)

    def test_queued_steering_does_not_need_automatic_recovery_budget(self):
        self.steering_case('reasoning_only_stop_exhausted', max_recoveries=0)

    def test_queued_steering_does_not_execute_rejected_tool_calls(self):
        for name in ['length_tool_exhausted', 'invalid_tool_stream_exhausted']:
            with self.subTest(case=name):
                self.steering_case(name)

    def test_text_continues_with_original_text_in_history(self):
        result = self.case('length_text', 0, 2)
        bodies = json.loads((self.root/'length_text/requests.json').read_text())
        history = bodies[1]['messages']
        self.assertTrue(any(m['role']=='assistant' and 'I will inspect' in str(m.get('content')) for m in history))
        self.assertTrue(any('previous response reached the output token limit' in str(m.get('content')) for m in history))
        self.assertIn('answer is 42', result['stdout'])

    def test_reasoning_only_recovers_through_one_tool_execution(self):
        result = self.case('reasoning_only_stop', 0, 3)
        self.assertEqual(len(result['tool_results']), 1, result)
        self.assertIn('answer=42', result['tool_results'][0]['content'])

    def test_whitespace_answer_recovers_instead_of_completing(self):
        result = self.case('whitespace_stop', 0, 3)
        self.assertEqual(len(result['tool_results']), 1, result)
        self.assertIn('answer is 42', result['stdout'])

    def test_truncated_tool_call_does_not_execute_before_repair(self):
        result = self.case('length_tool', 0, 3)
        self.assertEqual(len(result['tool_results']), 1, result)
        bodies = json.loads((self.root/'length_tool/requests.json').read_text())
        self.assertFalse(any(m.get('tool_calls') or m['role']=='tool' for m in bodies[1]['messages']))
        self.assertIn('answer=42', result['tool_results'][0]['content'])

    def test_rejected_streamed_call_is_executed_only_after_repair(self):
        result = self.case('invalid_tool_stream', 0, 3)
        self.assertEqual(len(result['tool_results']), 1, result)
        bodies = json.loads((self.root/'invalid_tool_stream/requests.json').read_text())
        self.assertFalse(any(m.get('tool_calls') or m['role']=='tool' for m in bodies[1]['messages']))
        self.assertIn('answer is 42', result['stdout'])

    def test_text_continuation_limit_is_finite_across_turns(self):
        result = self.case('length_text_exhausted', 1, 3)
        self.assertIn('FASTLLM_RECOVERY_EXHAUSTED', result['stderr'])

    def test_reasoning_only_cannot_finish_successfully_after_limit(self):
        result = self.case('reasoning_only_stop_exhausted', 1, 3)
        self.assertIn('FASTLLM_RECOVERY_EXHAUSTED', result['stderr'])

    def test_invalid_tools_stop_after_two_repairs_without_execution(self):
        result = self.case('invalid_tool_stream_exhausted', 1, 3)
        self.assertEqual(result['tool_results'], [])
        self.assertIn('FASTLLM_RECOVERY_EXHAUSTED', result['stderr'])

    def test_truncated_tools_stop_after_two_repairs_without_execution(self):
        result = self.case('length_tool_exhausted', 1, 3)
        self.assertEqual(result['tool_results'], [])
        self.assertIn('FASTLLM_RECOVERY_EXHAUSTED', result['stderr'])

    def test_ordinary_tool_validation_still_returns_to_model(self):
        result = self.case('tool_schema_error', 0, 2)
        self.assertIn('invalid arguments', result['tool_results'][0]['content'])

    def test_compaction_failure_is_not_hidden_by_recovery(self):
        result = self.case('context_overflow', 1, 3)
        self.assertIn('CONTEXT_WINDOW_EXCEEDED', result['stderr'])
        bodies = json.loads((self.root/'context_overflow/requests.json').read_text())
        self.assertIn('compaction engine', bodies[-1]['messages'][-1]['content'])
        self.assertNotIn('Your last response', str(bodies[-1]['messages']))

    def test_transport_retry_limit_is_preserved(self):
        result = self.case('missing_finish', 1, 2)
        self.assertIn('TRANSPORT', result['stderr'])

    def test_empty_response_retry_limit_is_preserved(self):
        result = self.case('empty_stop', 1, 2)
        self.assertIn('EMPTY_RESPONSE', result['stderr'])


if __name__=='__main__':
    unittest.main()
