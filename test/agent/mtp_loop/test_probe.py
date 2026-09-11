import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analyze import analyze_text, find_cycle
from probe import HERE, main, make_payload, probe, sse_data


class CycleTests(unittest.TestCase):
    def test_real_legacy_response(self):
        fixture = HERE / 'fixtures'
        text = (fixture / 'legacy_reasoning.txt').read_text(encoding='utf-8')
        expected = json.loads((fixture / 'legacy_cycle.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), expected['response_sha256'])
        result = analyze_text(text)
        self.assertEqual(result['unit'], 'characters')
        for key, value in expected['character_cycle'].items():
            self.assertEqual(result['cycle'][key], value, key)

    def test_integer_tokens_and_partial_last_cycle(self):
        block = tuple(range(100, 300))
        tokens = (7, 8, 9) + block * 4 + block[:53] + (999,)
        cycle = find_cycle(tokens)
        self.assertEqual((cycle['start'], cycle['end'], cycle['period']), (3, 856, 200))
        self.assertEqual(cycle['full_repetitions'], 4)
        self.assertEqual(cycle['trailing_units'], 53)

    def test_single_token_loop(self):
        cycle = find_cycle([42] * 1024)
        self.assertEqual((cycle['period'], cycle['full_repetitions']), (1, 1024))

    def test_repeated_quotation_is_not_a_continuous_cycle(self):
        block = tuple(range(100, 300))
        tokens = block + (1,) + block + (2,) + block + (3,) + block
        self.assertIsNone(find_cycle(tokens))

    def test_two_copies_are_not_a_cycle(self):
        self.assertIsNone(find_cycle(tuple(range(200)) * 2))

    def test_short_and_nonrepeated_text(self):
        self.assertIsNone(analyze_text('测试已完成。')['cycle'])
        self.assertIsNone(find_cycle(tuple(range(8192))))

    def test_cycle_after_frequent_unrelated_anchor(self):
        # A frequent quoted phrase must not hide a different later cycle.
        prefix = ''.join('This quotation is repeated in the notes. ' + str(i) + '\n' for i in range(40))
        block = ''.join(chr(0x4e00 + i) for i in range(200))
        text = prefix + block * 4
        cycle = find_cycle(text)
        self.assertEqual(cycle['period'], 200)
        self.assertEqual(cycle['full_repetitions'], 4)

    def test_token_units_are_not_character_counts(self):
        class WordTokenizer:
            def encode(self, text, add_special_tokens):
                assert not add_special_tokens
                return SimpleNamespace(ids=[int(w[1:]) for w in text.split()])

            def decode(self, ids):
                return ' '.join(f'w{i}' for i in ids)

        text = (' '.join(f'w{i}' for i in range(100)) + ' ') * 4
        result = analyze_text(text, WordTokenizer())
        self.assertEqual((result['unit'], result['length']), ('tokens', 400))
        self.assertEqual(result['cycle']['period'], 100)
        self.assertNotEqual(result['length'], result['chars'])


@contextlib.contextmanager
def fake_server(body, status=200):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(status)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            # Split UTF-8 characters and SSE lines across writes.
            for offset in range(0, len(body), 7):
                self.wfile.write(body[offset:offset + 7])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/chat/completions', received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def event(value):
    return ('data: ' + json.dumps(value, ensure_ascii=False) + '\r\n\r\n').encode()


class StreamTests(unittest.TestCase):
    def test_sse_comments_and_multiline_fields(self):
        lines = [b': keepalive\r\n', b'event: chunk\r\n', b'data: {"a":\r\n',
                 b'data: 1}\r\n', b'\r\n', b'data: [DONE]\r\n', b'\r\n']
        self.assertEqual(list(sse_data(lines)), ['{"a":\n1}', '[DONE]'])

    def test_thinking_content_usage_and_length_are_recorded(self):
        body = event({'choices': [{'delta': {'reasoning_content': '正在检查。'}}]})
        body += event({'choices': [{'delta': {'content': '结果正确。'}, 'finish_reason': 'length'}]})
        body += event({'choices': [], 'usage': {'completion_tokens': 8}}) + b'data: [DONE]\n\n'
        with tempfile.TemporaryDirectory() as directory, fake_server(body) as (url, requests):
            out = Path(directory) / 'probe'
            result = probe(url, {'model': 'fixture', 'stream': True}, out)
            self.assertIsNone(result['error'])
            self.assertTrue(result['done'])
            self.assertEqual(result['finish'], 'length')
            self.assertEqual(result['usage']['completion_tokens'], 8)
            self.assertIsNone(result['reasoning']['cycle'])
            self.assertEqual((out / 'reasoning.txt').read_text(), '正在检查。')
            self.assertEqual((out / 'content.txt').read_text(), '结果正确。')
            self.assertEqual(requests[0]['model'], 'fixture')

    def test_incomplete_stream_preserves_partial_text(self):
        body = event({'choices': [{'delta': {'reasoning_content': '尚未完成'}}]})
        with tempfile.TemporaryDirectory() as directory, fake_server(body) as (url, _):
            out = Path(directory) / 'probe'
            result = probe(url, {}, out)
            self.assertIn('Incomplete SSE', result['error'])
            self.assertFalse(result['done'])
            self.assertEqual((out / 'reasoning.txt').read_text(), '尚未完成')
            self.assertTrue((out / 'result.json').exists())

    def test_api_error_is_not_a_successful_generation(self):
        body = event({'error': {'message': 'fixture error'}})
        with tempfile.TemporaryDirectory() as directory, fake_server(body) as (url, _):
            result = probe(url, {}, Path(directory) / 'probe')
            self.assertIn('fixture error', result['error'])

    def test_http_error_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory, fake_server(b'failed', 500) as (url, _):
            result = probe(url, {}, Path(directory) / 'probe')
            self.assertIn('500', result['error'])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                probe('http://127.0.0.1:1', {}, Path(directory))

    def test_fixture_payload_has_no_machine_dependencies(self):
        data = json.loads((HERE / 'cases.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data['cases']), 6)
        before = json.dumps(data, ensure_ascii=False)
        for case in data['cases']:
            payload = make_payload(case, data['tools'], model='my-served-model', temperature=.6,
                                   top_k=20, top_p=.95, max_tokens=16384, effort='medium')
            self.assertEqual(payload['model'], 'my-served-model')
            self.assertEqual(payload['tool_choice'], 'none')
            self.assertEqual(payload['reasoning_effort'], 'medium')
            self.assertTrue(payload['chat_template_kwargs']['enable_thinking'])
            payload['messages'][0]['content'] = 'changed in one request'
        self.assertEqual(json.dumps(data, ensure_ascii=False), before)

    def test_cli_cycle_exit_code_and_error_priority(self):
        text = '重新检查假设，然后回到相同的结论。' * 40
        body = event({'choices': [{'delta': {'reasoning_content': text}, 'finish_reason': 'length'}]})
        for complete, expected_exit in [(True, 2), (False, 1)]:
            with self.subTest(complete=complete):
                response = body + (b'data: [DONE]\n\n' if complete else b'')
                with tempfile.TemporaryDirectory() as directory, fake_server(response) as (url, _):
                    out = Path(directory) / 'run'
                    argv = ['probe.py', '--base-url', url.removesuffix('/chat/completions'),
                            '--model', 'test-model', '--cases', 'circular_requirements_zh',
                            '--out', str(out), '--fail-on-cycle']
                    with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(), expected_exit)
                    summary = json.loads((out / 'summary.json').read_text())
                    self.assertEqual(summary['exact_cycles'], 1)
                    self.assertEqual(summary['errors'], 0 if complete else 1)


if __name__ == '__main__':
    unittest.main()
