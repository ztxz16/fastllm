"""Attention DP/EP 的正式入口和可见设备规划；不加载权重或原生库。"""

from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from test_parallel_config import inventory, ROOT
from fastllm_pytools import cli, parallel_config, process_tp
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model
from run_parallel_config import real_cuda_case


class AttentionEPConfigTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.parser = make_normal_parser('attention ep')
        stack.enter_context(mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'))
        self.query = stack.enter_context(mock.patch.object(parallel_config, 'cuda_inventory', return_value=inventory()))
        self.binary = stack.enter_context(mock.patch.object(process_tp, 'native_binary', side_effect=AssertionError('binary lookup')))
        self.session = stack.enter_context(mock.patch.object(process_tp, 'Qwen3ProcessSession', side_effect=AssertionError('worker launch')))

    def args(self, *flags):
        return self.parser.parse_args(['missing-model', '--tp-backend', 'process', '--attention-dp', '4', '--ep', '4', *flags])

    def preview(self, *flags):
        with mock.patch.object(sys, 'argv', ['ftllm', 'serve', 'missing-model', '--tp-backend', 'process',
                    '--attention-dp', '4', '--ep', '4', '--parallel-plan', *flags]), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(), 0)
        self.binary.assert_not_called()
        self.session.assert_not_called()
        return json.loads(out.getvalue())

    def test_four_ranks_use_four_devices_not_sixteen(self):
        p = self.preview()
        self.assertEqual((p['mode'], p['dp_size'], p['tp_size'], p['attention_dp_size'], p['ep_size'], p['total_gpus']),
                         ('attention_dp_ep', 1, 1, 4, 4, 4))
        self.assertEqual(p['attention_groups'], [[0], [1], [2], [3]])
        self.assertEqual(p['ep_groups'], [[0, 1, 2, 3]])
        self.assertEqual([(r['rank'], r['attention_rank'], r['ep_rank'], r['tp_rank'], r['device']['logical_id'])
                          for r in p['ranks']], [(i, i, i, 0, i) for i in range(4)])
        self.assertTrue(p['attention_ep_execution_available'])
        self.assertEqual(len(p['replicas']), 1)

    def test_two_ranks_leave_other_devices_unused(self):
        p = self.preview('--attention-dp', '2', '--ep', '2')
        self.assertEqual(p['total_gpus'], 2)
        self.assertEqual([r['device'] for r in p['ranks']], inventory()['devices'][:2])

    def test_single_rank_moe_identified_by_config_only(self):
        model = self.root / 'toy'
        model.mkdir()
        (model / 'config.json').write_text(json.dumps({'model_type': 'qwen2_moe'}))
        args = self.parser.parse_args([str(model), '--tp-backend', 'process'])
        p = parallel_config.build_parallel_plan(args)
        self.assertEqual((p['mode'], p['total_gpus']), ('attention_dp_ep', 1))
        self.assertEqual(p['ep_groups'], [[0]])
        self.binary.assert_not_called()
        self.session.assert_not_called()

    def test_reordered_driver_uuid_mapping(self):
        self.query.return_value = inventory((0, 3, 1, 2))
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,3,1,2'):
            p = self.preview()
        self.assertEqual([r['device'] for r in p['ranks']], inventory((0, 3, 1, 2))['devices'])

    def test_strict_types_cli_and_json_all_entrypoints(self):
        for name in ('attention_dp', 'ep'):
            for value in (0, -1, 3, 8, True, False, 2.0, '2', None, []):
                options = dict(model='missing', tp_backend='process', attention_dp=4, ep=4)
                options[name] = value
                path = self.root / 'options.json'
                path.write_text(json.dumps(options))
                for entry in (make_normal_llm_model, process_tp.ProcessTPModel):
                    with self.subTest(name=name, value=value, entry=entry.__name__), self.assertRaises(ValueError):
                        entry(self.parser.parse_args([str(path)]))
            for value in ('0', '-1', '1.5', 'true'):
                with self.subTest(name=name, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.args('--' + name.replace('_', '-'), value)
        self.query.assert_not_called()

    def test_conflicting_axes_backend_devices_and_tp(self):
        cases = [(['--ep', '2'], 'attention_dp == ep'), (['--dp', '2'], '--dp 1'),
                 (['--tp-backend', 'thread'], '--tp-backend process'),
                 (['--device', 'cuda:0'], 'cannot combine'), (['--moe_device', 'cuda:0'], 'cannot combine'),
                 (['--custom-allreduce'], 'AllGather/ReduceScatter')]
        cases += [(['--tp', value], '--tp 1') for value in ('0', '2', 'auto', '0,1', 'cuda:0', 'false', '1.0')]
        for flags, reason in cases:
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, reason):
                parallel_config.build_parallel_plan(self.args(*flags))
        for value in (True, False, 1.0, [1]):
            args = self.args()
            args.tp = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '--tp 1'):
                parallel_config.build_parallel_plan(args)
        self.query.assert_not_called()

    def test_insufficient_empty_duplicate_visibility(self):
        del os.environ['CUDA_VISIBLE_DEVICES']
        self.query.return_value = inventory((0, 1))
        with self.assertRaisesRegex(ValueError, 'needs 4, found 2'):
            parallel_config.build_parallel_plan(self.args())
        self.query.return_value = inventory(())
        with self.assertRaisesRegex(ValueError, 'no visible'):
            parallel_config.build_parallel_plan(self.args())
        self.query.return_value = inventory((0, 0, 1, 2))
        with self.assertRaisesRegex(ValueError, 'duplicate CUDA'):
            parallel_config.build_parallel_plan(self.args())
        for mask in ('', '0,0', '0,,1', '-1'):
            with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES=mask), self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES'):
                parallel_config.build_parallel_plan(self.args())

    def test_prefix_routing_targets_attention_ranks(self):
        p = self.preview('--prefix-cache', 'true', '--dp-routing', 'prefix', '--cuda-graph')
        self.assertEqual((p['dp_size'], p['dp_routing']), (1, 'prefix'))
        with self.assertRaisesRegex(ValueError, 'prefix-cache true'):
            self.preview('--dp-routing', 'prefix')

    def test_json_aliases_precedence_and_duplicate_rejection(self):
        path = self.root / 'options.json'
        path.write_text(json.dumps({'model': 'missing', 'attention-dp': 2, 'ep': 2,
                                    'tp_backend': 'process', 'parallel-plan': True}))
        with mock.patch.object(sys, 'argv', ['ftllm', 'serve', str(path), '--attention-dp', '4', '--ep', '4']), \
                redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(), 0)
        self.assertEqual(json.loads(out.getvalue())['attention_dp_size'], 2)
        path.write_text(json.dumps({'attention-dp': 2, 'attention_dp': 4}))
        with self.assertRaisesRegex(ValueError, 'both attention-dp and attention_dp'):
            make_normal_llm_model(self.parser.parse_args([str(path)]))

    def test_execution_uses_moe_factory_and_rejects_dense_adapter(self):
        with mock.patch('fastllm_pytools.process_moe.ProcessMoeModel') as model:
            self.assertIs(make_normal_llm_model(self.args()), model.return_value)
            model.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'requires ProcessMoeModel'):
            process_tp.ProcessTPModel(self.args())
        self.binary.assert_not_called()
        self.session.assert_not_called()

    def test_preview_rejected_by_model_factory(self):
        with self.assertRaisesRegex(ValueError, 'CLI preview'):
            make_normal_llm_model(self.args('--parallel-plan'))
        self.query.assert_not_called()

    def test_server_report_verifies_shared_group_and_failures(self):
        plan = self.preview()
        metadata = {}

        def run(command, **kwargs):
            kwargs['stdout'].write(json.dumps(plan))
            self.assertIn('--attention-dp', command)
            return subprocess.CompletedProcess(command, 0)

        with mock.patch('run_parallel_config.subprocess.run', side_effect=run):
            row = real_cuda_case(self.root, 1, 1, metadata, 4, 4)
            self.assertEqual((row['status'], row['failed_elements']), ('PASS', 0))
            self.assertEqual(row['case'], 'real_cuda_attention_dp4_ep4')
            self.assertEqual(len(metadata['gpu']), 4)
            plan['ranks'][2]['ep_rank'] = 0
            row = real_cuda_case(self.root, 1, 1, {}, 4, 4)
            self.assertEqual(row['status'], 'FAIL')
            self.assertIn('rank mapping', row['reason'])

    def test_default_json_export_keeps_legacy_mode(self):
        path = self.root / 'defaults.json'
        with redirect_stdout(io.StringIO()):
            cli.save_defaults_to_json(self.parser, str(path))
        from fastllm_pytools.util import load_model_args
        args = self.parser.parse_args([str(path)])
        load_model_args(args)
        self.assertEqual((args.attention_dp, args.ep), (1, 1))
        self.assertIsNone(parallel_config.prepare_parallel_entry(args))
        self.query.assert_not_called()

    def test_all_formal_commands_preview_without_model_imports(self):
        for command in ('chat', 'run', 'serve', 'server'):
            script = ("import sys\nfrom unittest.mock import patch\nfrom fastllm_pytools import cli\n"
                      f"sys.argv=['ftllm',{command!r},'--attention-dp','4','--ep','4','--tp-backend','process','--parallel-plan']\n"
                      f"with patch('fastllm_pytools.parallel_config.cuda_inventory',return_value={inventory()!r}): cli.main()\n"
                      "assert not any(m in sys.modules for m in ('torch','transformers','fastllm_pytools.llm','fastllm_pytools.process_tp'))\n")
            p = subprocess.run([sys.executable, '-c', script], env=dict(os.environ, PYTHONPATH=str(ROOT / 'tools')),
                               capture_output=True, text=True, timeout=15)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout)['total_gpus'], 4)


if __name__ == '__main__':
    unittest.main()
