"""正式模型工厂和 CLI 的纯 Python 回归；模拟原生协议，不编译、不使用 GPU。"""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest import mock

import test_qwen3_session as fixture_module
from fastllm_pytools import cli, process_tp
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model


class ProcessEntryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.model = self.root / 'cancel'
        self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps(dict(model_type='qwen3', vocab_size=17, eos_token_id=16)))
        self.tokenizer = mock.Mock(chat_template='template', eos_token_id=16)
        self.tokenizer.apply_chat_template.return_value = [1, 2]
        self.tokenizer.decode.side_effect = lambda ids, **_: '答' * len(ids)
        transformers = mock.Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = self.tokenizer
        patcher = mock.patch.dict(sys.modules, {'transformers': transformers})
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.base = [str(self.model), '--tp_backend', 'process', '--tp', '2',
                     '--tp-binary', str(self.fixture.binary), '--tp-worker', str(self.fixture.binary),
                     '--tp-output', str(self.root / 'output'), '--tp-timeout', '3', '--tp-request-timeout', '2']

    def args(self, extra=()):
        return make_normal_parser('test').parse_args(self.base + list(extra))

    def test_factory_reuses_native_workers_and_releases(self):
        with make_normal_llm_model(self.args()) as model:
            self.assertEqual(model.tp_backend, 'process')
            workers = model.ready['workers']
            self.assertEqual(model.response('你好', max_length=1), '答')
            self.assertEqual(''.join(model.stream_response('你好', max_length=1)), '答')
            self.assertEqual(model.ready['workers'], workers)
            self.assertEqual(model.session._next_id, 3)
            self.assertNotIn('fastllm_pytools.llm', sys.modules)
        self.assertEqual(model.state, 'closed')
        self.assertTrue((model.output / 'session/shutdown.json').exists())

    def test_generator_close_cancels_and_next_request_reuses_session(self):
        with make_normal_llm_model(self.args()) as model:
            stream = model.stream_response('你好', max_length=16)
            self.assertEqual(next(stream), '答')
            stream.close()
            self.assertEqual(model.state, 'ready')
            report = json.loads((model.output / 'session/request-000001.response.json').read_text())
            self.assertTrue(report['cancelled'])
            self.assertEqual(model.response('恢复', max_length=1), '答')

    def test_invalid_prompt_or_sampling_never_reaches_native(self):
        with make_normal_llm_model(self.args()) as model:
            for kwargs in ({'repeat_penalty': 1.1}, {'max_length': 0}, {'temperature': float('nan')}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    model.response('你好', **kwargs)
            self.tokenizer.apply_chat_template.return_value = [1] * 129
            with self.assertRaisesRegex(ValueError, 'Prompt contains 129'):
                model.response('超长')
            self.assertEqual(model.session._next_id, 1)

    def test_model_close_cancels_live_stream_and_reaps_workers(self):
        with make_normal_llm_model(self.args()) as model:
            stream = model.stream_response('你好', max_length=16)
            self.assertEqual(next(stream), '答')
            model.close()
            stream.close()
            self.assertEqual(model.state, 'closed')
            self.assertFalse(Path(f'/proc/{model.session._process.pid}').exists())

    def test_graph_factory_passes_flag_and_checks_native_path(self):
        with make_normal_llm_model(self.args(['--cuda-graph'])) as model:
            self.assertTrue(model.session.cuda_graph)
            self.assertEqual(model.ready['execution'], 'decode_cuda_graph')
            self.assertEqual(model.response('你好', max_length=1), '答')
            self.assertIn('--cuda-graph 1', (model.output / 'session/command.txt').read_text())
        self.assertEqual(model.session.shutdown['graph_after_unload'][0]['workspace_bytes'], 0)

    def test_unimplemented_optimizations_are_rejected(self):
        for name in ('FASTLLM_PREFIX_CACHE', 'FASTLLM_CUDA_GRAPH', 'FASTLLM_CUDA_CUSTOM_ALLREDUCE'):
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: '1'}), \
                    self.assertRaisesRegex(ValueError, name):
                make_normal_llm_model(self.args())

    def test_config_cannot_change_command_or_enable_unknown_option(self):
        config = self.root / 'options.json'
        for update in ({'command': 'export'}, {'tp_backend': 'process', 'unknown': 'value'}):
            config.write_text(json.dumps(dict(vars(self.args()), **update)))
            with mock.patch('sys.argv', ['ftllm', 'run', str(config)]), self.assertRaises(ValueError):
                cli.main()

    def test_history_is_encoded_and_never_silently_truncated(self):
        with make_normal_llm_model(self.args()) as model:
            model.response('第二问', history=[('第一问', '回答')], max_length=1)
            messages = self.tokenizer.apply_chat_template.call_args.args[0]
            self.assertEqual([m['content'] for m in messages], ['第一问', '回答', '第二问'])
            self.assertFalse(self.tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])

    def test_explicit_unsupported_options_rejected_before_model_load(self):
        for extra in (['--dtype', 'int4'], ['--atype', 'float32'], ['--prefix-cache', 'true'],
                      ['--kv-cache-layout', 'paged'], ['--max_batch', '2'], ['--device', 'cpu'],
                      ['--lora', 'adapter'], ['--enable_thinking', 'true'], ['--triton'],
                      ['--linear-backend', 'machete']):
            with self.subTest(extra=extra), mock.patch.object(process_tp, 'Qwen3ProcessSession') as native:
                with self.assertRaisesRegex(ValueError, 'does not support'):
                    make_normal_llm_model(self.args(extra))
                native.assert_not_called()

    def test_bad_model_or_binary_does_not_import_old_library(self):
        for update, message in (({'model_type': 'qwen3_moe'}, 'dense Qwen3'),
                                ({'model_type': 'qwen3', 'quantization_config': {'quant_method': 'awq'}}, 'dense Qwen3')):
            (self.model / 'config.json').write_text(json.dumps(update))
            with self.assertRaisesRegex(ValueError, message):
                make_normal_llm_model(self.args())
        with self.assertRaisesRegex(ValueError, 'missing or not executable'):
            process_tp.native_binary(self.root / 'missing', 'unused')

    def test_visible_gpu_mapping_and_no_silent_clamp(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-abc,GPU-def'}):
            for spec, result in [('auto', '0,1'), ('2', '0,1'), ('0', '0'), ('1', '0'), ('1,0', '1,0')]:
                self.assertEqual(process_tp.process_devices(spec), result)
            for spec in ('3', '0,0', '2,3', '-1', '', 'cuda:0,1', '0:2,1:1', '65'):
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    process_tp.process_devices(spec)
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaises(ValueError):
                process_tp.process_devices('1')

    def test_cli_routes_process_without_loading_thread_backend(self):
        for command, target in [('run', 'run_chat'), ('chat', 'run_chat'), ('server', 'run_server'), ('serve', 'run_server')]:
            with self.subTest(command=command), mock.patch('sys.argv', ['ftllm', command, *self.base]), \
                    mock.patch.object(process_tp, target, return_value=0) as action:
                self.assertEqual(cli.main(), 0)
                action.assert_called_once()
        with mock.patch('sys.argv', ['ftllm', 'export', *self.base, '-o', 'x']):
            with self.assertRaisesRegex(ValueError, 'supports chat/run/server/serve'):
                cli.main()

    def test_cli_config_file_selects_same_backend_and_model(self):
        config = self.root / 'options.json'
        config.write_text(json.dumps(vars(self.args())))
        with mock.patch('sys.argv', ['ftllm', 'run', str(config), '--prompt', '你好', '--max-new-tokens', '1']), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            cli.main()
        self.assertEqual(output.getvalue(), '答\n')
        directories = list((self.root / 'output').iterdir())
        self.assertEqual(len(directories), 1)
        self.assertTrue((directories[0] / 'session/shutdown.json').is_file())

    def test_cli_chat_eof_and_clear_preserve_cleanup(self):
        with mock.patch('sys.argv', ['ftllm', 'chat', *self.base, '--max-new-tokens', '1']), \
                mock.patch('sys.stdin', io.StringIO('你好\nclear\n你好\n')), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cli.main()
        self.assertEqual(len(self.tokenizer.apply_chat_template.call_args.args[0]), 1)
        directory = next((self.root / 'output').iterdir())
        self.assertTrue((directory / 'session/shutdown.json').exists())

    def test_default_thread_dispatch_is_unchanged(self):
        with mock.patch('sys.argv', ['ftllm', 'chat', 'model']), \
                mock.patch('fastllm_pytools.chat.fastllm_chat') as thread_chat:
            cli.main()
        thread_chat.assert_called_once()
        self.assertEqual(thread_chat.call_args.args[0].tp_backend, 'thread')

    def test_packaged_layout_imports_without_repository_tools_or_shared_library(self):
        package = self.root / 'package'
        shutil.copytree(Path(process_tp.__file__).parent, package / 'ftllm', ignore=shutil.ignore_patterns('__pycache__'))
        env = dict(os.environ, PYTHONPATH=str(package))
        result = subprocess.run([sys.executable, '-m', 'ftllm.cli', 'server', '--help'], cwd=self.root,
                                env=env, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

        self.assertIn('--tp-binary', result.stdout)
        result = subprocess.run([sys.executable, '-c', 'from ftllm import process_tp; '
                                 'from ftllm.qwen3_process_server import ChatService'],
                                cwd=self.root, env=env, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_acceptance_runner_uses_packaged_factory_and_cli(self):
        import run_qwen3_entrypoint as runner

        package = self.root / 'package'
        shutil.copytree(Path(process_tp.__file__).parent, package / 'ftllm', ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('fastllm_tp_generate', 'fastllm_tp_worker'):
            shutil.copy2(self.fixture.binary, package / 'ftllm' / name)
        # 只有原生协调器与 tokenizer 是测试替身，正式 CLI/工厂/HTTP/清理逻辑全部照常执行。
        (package / 'transformers.py').write_text('''class Tokenizer:
    chat_template = 'template'
    eos_token_id = 1
    def apply_chat_template(self, *args, **kwargs): return [1, 2]
    def decode(self, ids, **kwargs): return '答' * len(ids)
class AutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs): return Tokenizer()
''')
        model = self.root / 'ok'
        shutil.copytree(self.model, model)
        (model / 'config.json').write_text(json.dumps(dict(model_type='qwen3', vocab_size=17, eos_token_id=1)))
        self.tokenizer.eos_token_id = 1
        cache = self.root / 'cache.txt'
        cache.write_text('CUDA_ARCH:STRING=70\n')
        argv = ['runner', '--generate', str(self.fixture.binary), '--worker', str(self.fixture.binary),
                '--model', str(model), '--build-cache', str(cache), '--package-root', str(package),
                '--output', str(self.root / 'reports')]

        def reference(binary, worker, model, request, devices, output):
            return dict(backend=process_tp.BACKEND, request=request, token_ids=[1], finish_reason='eos')

        prior_path = sys.path[:]
        modules = {k: v for k, v in sys.modules.items() if k == 'ftllm' or k.startswith('ftllm.')}
        try:
            with mock.patch('sys.argv', argv), mock.patch.object(runner, 'run_native', side_effect=reference), \
                    mock.patch.object(runner, 'capture', return_value='fixture'), redirect_stdout(io.StringIO()):
                result = runner.main()
            directory = next((self.root / 'reports').iterdir())
            report = json.loads((directory / 'summary.json').read_text())
            self.assertEqual(result, 0, report)
            self.assertEqual(len(report['cases']), 12)
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((directory / ('summary.' + suffix)).is_file())
            # 同一验收入口失败时仍保存已执行部分，不得把缺少 case 的报告记为 PASS。
            with mock.patch('sys.argv', argv), mock.patch.object(runner, 'run_native', side_effect=RuntimeError('native failed')), \
                    mock.patch.object(runner, 'capture', return_value='fixture'), redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 1)
            failed = next(p for p in (self.root / 'reports').iterdir() if p != directory)
            partial = json.loads((failed / 'summary.json').read_text())
            self.assertEqual(partial['status'], 'FAIL')
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['cases'][0]['case'], 'native_reference')
            self.assertIn('native failed', partial['cases'][0]['reason'])
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((failed / ('summary.' + suffix)).is_file())
        finally:
            sys.path[:] = prior_path
            for key in list(sys.modules):
                if key == 'ftllm' or key.startswith('ftllm.'):
                    del sys.modules[key]
            sys.modules.update(modules)


if __name__ == '__main__':
    unittest.main()
