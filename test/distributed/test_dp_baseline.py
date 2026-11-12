"""固定服务配置与JSONL回归；真实CLI/HTTP，CUDA、模型分词及原生协议使用替身。"""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

import diagnose_qwen3_dp_lengths as runner
from qwen3_request_inputs import read_requests
import test_process_dp as fixtures
from fastllm_pytools import cli, dp_baseline, parallel_config


class BaselineConfigTests(unittest.TestCase):
    def test_preview_is_read_only_and_matches_formal_arguments(self):
        for profile, (dp, tp) in dp_baseline.PROFILES.items():
            out = io.StringIO()
            with redirect_stdout(out), mock.patch.object(cli, 'main') as entry, \
                    mock.patch.object(parallel_config, 'cuda_inventory') as cuda:
                self.assertEqual(dp_baseline.main(['--profile', profile, '--model', '/missing/model', '--print-command']), 0)
            entry.assert_not_called()
            cuda.assert_not_called()
            plan = json.loads(out.getvalue())
            args = cli.args_parser().parse_args(plan['command'][3:])
            self.assertEqual((args.command, args.dp, args.tp_backend, args.tp), ('serve', dp, 'process', str(tp)))
            self.assertEqual((args.dp_routing, args.prefix_cache, args.cuda_graph), ('round_robin', 'false', True))
            self.assertEqual((args.tp_queue_size, plan['recommended_client_concurrency']), (2*dp, dp))
            self.assertEqual(args.custom_allreduce, tp == 2)
            self.assertEqual((args.host, args.port), ('127.0.0.1', 8080))

    def test_launcher_calls_formal_cli_and_restores_argv(self):
        previous = sys.argv
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'worker'
            binary.write_text('#!/bin/false\n')
            binary.chmod(0o700)
            argv = ['--profile', 'dp4_tp1', '--model', folder, '--generate', str(binary), '--worker', str(binary)]
            def entry():
                self.assertEqual(cli.args_parser().parse_args(sys.argv[1:]).dp, 4)
                return 7
            with mock.patch.object(cli, 'main', side_effect=entry), redirect_stdout(io.StringIO()):
                self.assertEqual(dp_baseline.main(argv), 7)
            self.assertIs(sys.argv, previous)
            binary.unlink()
            with mock.patch.object(cli, 'main') as start, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                dp_baseline.main(argv)
            start.assert_not_called()

    def test_jsonl_rejections_and_example_contract(self):
        example = runner.ROOT / 'test/distributed/data/qwen3_requests_example.jsonl'
        items = read_requests(example.read_text())
        self.assertEqual(len(items), 8)
        self.assertEqual(sum(i['payload']['stream'] for i in items), 4)
        good = dict(name='a', request=dict(messages=[dict(role='user', content='你好')]))
        invalid = ['', '{', '[]', json.dumps(good)+'\n'+json.dumps(good),
                   json.dumps(dict(good, request=dict(good['request'], temperature=0.2))),
                   json.dumps(dict(good, request=dict(good['request'], top_k=True))),
                   json.dumps(dict(good, request=dict(good['request'], tools=[]))),
                   json.dumps(dict(good, request=dict(good['request'], stream=True, stream_options={'include_usage': False}))),
                   json.dumps(dict(good, request=dict(messages=[dict(role='assistant', content='你好')])))]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_requests(text)

    def test_full_conversation_is_reference_identity(self):
        a = dict(messages=[dict(role='user', content='相同问题'), dict(role='assistant', content='答复'),
                           dict(role='user', content='后续问题甲')], max_tokens=4, stream=False)
        b = dict(a, messages=[*a['messages'][:-1], dict(role='user', content='后续问题乙')])
        self.assertNotEqual(runner.input_key(a), runner.input_key(b))
        self.assertEqual(runner.input_key(a), runner.input_key(dict(a, stream=True, stream_options={'include_usage': True})))


class BaselineRequestTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ProcessDPTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.rename_model('dp_routing_lengths')
        self.root, self.fixture = fixture.root, fixture
        fixture.fixture.tokenizer.apply_chat_template.side_effect = lambda messages, **_: [1]*(32+len(messages[-1]['content']))
        self.source = self.root / 'requests.jsonl'
        rows = [dict(name=f'followup_{i}_{stream}', request=dict(messages=[dict(role='user', content='请整理活动计划'),
                    dict(role='assistant', content='请补充条件'), dict(role='user', content=content)],
                    max_tokens=2+2*i, stream=stream))
                for i, content in enumerate(('室内', '室外但可能下雨')) for stream in (False, True)]
        self.source.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows))
        cache = self.root / 'CMakeCache.txt'
        cache.write_text('CUDA_ARCH:UNINITIALIZED=70\n')
        binary = str(fixture.fixture.fixture.binary)
        self.args = ['--model', str(fixture.model), '--generate', binary, '--worker', binary,
                     '--build-cache', str(cache), '--timeout', '10', '--requests', str(self.source),
                     '--output', str(self.root / 'reports')]

    def invoke(self, profile='dp4_tp1', *extra):
        with mock.patch.object(runner, 'capture', return_value='Python fixture; no GPU'), \
                mock.patch.object(runner, 'gpu_sample', return_value=[dict(uuid='GPU-fixture', index=0,
                    **{'memory.used': 123.5, 'memory.total': 16000.0})]), redirect_stdout(io.StringIO()):
            code = runner.main([*self.args, '--profile', profile, *extra])
        output = sorted((self.root / 'reports').iterdir())[-1]
        return code, output, json.loads((output / 'summary.json').read_text())

    def test_formal_http_profiles_and_conversation_references(self):
        for profile, (dp, tp) in dp_baseline.PROFILES.items():
            with self.subTest(profile=profile):
                code, output, report = self.invoke(profile)
                self.assertEqual((code, report['status'], report['complete']), (0, 'PASS', True), report)
                self.assertEqual([r['concurrency'] for r in report['cases']], [1, dp])
                self.assertEqual([len(r['samples']) for r in report['cases']], [5, 5])
                self.assertEqual(len(json.loads((output / 'references.json').read_text())), 2)
                self.assertEqual(len(json.loads((output / 'reference_texts.json').read_text())), 4)
                self.assertEqual((output / 'requests.jsonl').read_bytes(), self.source.read_bytes())
                command = shlex.split((output / 'server.command.txt').read_text())[4:]
                expected = dp_baseline.server_flags(profile, self.fixture.model, self.args[3], self.args[5],
                                                     output / 'server', 10.0, port=0)
                self.assertEqual(command, expected)
                self.assertEqual(report['metadata']['profile'], profile)
                for row in report['cases']:
                    self.assertEqual(row['failed_elements'], 0)
                    for sample in row['samples']:
                        self.assertEqual({r['generated_tokens'] for r in sample['records']}, {2, 4})
                        self.assertTrue(all(r['completion_ms'] >= r['client_wait_ms'] for r in sample['records']))

    def test_bad_jsonl_fails_before_model_and_writes_reports(self):
        self.source.write_text('{"name":"bad","request":{"messages":[]}}')
        with mock.patch.object(runner, 'make_normal_llm_model') as model:
            code, output, report = self.invoke()
        model.assert_not_called()
        self.assertEqual((code, report['status']), (1, 'FAIL'))
        self.assertIn('JSONL line 1', report['cases'][0]['reason'])
        for suffix in ('json', 'csv', 'md', 'xlsx'):
            self.assertTrue((output / ('summary.'+suffix)).is_file())

    def test_capacity_rejected_without_starting_http_service(self):
        self.fixture.fixture.tokenizer.apply_chat_template.side_effect = lambda *_, **__: [1]*2048
        with mock.patch.object(runner.MixedService, 'start') as server:
            code, _, report = self.invoke()
        server.assert_not_called()
        self.assertEqual((code, report['status']), (1, 'FAIL'))
        self.assertIn('input followup_', report['cases'][0]['reason'])

    def test_profile_cannot_be_overridden_by_dp_or_length_arguments(self):
        for extra in (['--dp', '2'], ['--tp', '1'], ['--prompt-lengths', '64', '512']):
            with mock.patch.object(runner, 'make_normal_llm_model') as model, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main([*self.args, '--profile', 'dp4_tp1', *extra])
            model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
