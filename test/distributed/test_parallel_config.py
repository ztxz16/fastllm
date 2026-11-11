"""DP 分配的正式 CLI/工厂回归；仅替换 CUDA 查询，不编译或使用真实 GPU。"""

from contextlib import ExitStack, redirect_stdout, redirect_stderr
import ctypes
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from fastllm_pytools import cli, parallel_config, process_tp
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model


def inventory(order=(0, 1, 2, 3)):
    return {'driver_cuda_version': 12060,
            'devices': [{'logical_id': logical, 'uuid': str(uuid.UUID(int=physical + 1)),
                         'name': f'fixture GPU {physical}'} for logical, physical in enumerate(order)]}


class ParallelConfigTests(unittest.TestCase):
    def setUp(self):
        self.parser = make_normal_parser('test')
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3'}))
        self.query = stack.enter_context(mock.patch.object(parallel_config, 'cuda_inventory', return_value=inventory()))
        self.factory = stack.enter_context(mock.patch.object(process_tp, 'native_binary', side_effect=AssertionError('native binary lookup')))
        self.session = stack.enter_context(mock.patch.object(process_tp, 'Qwen3ProcessSession', side_effect=AssertionError('worker launch')))

    def args(self, *flags):
        return self.parser.parse_args(['missing-model', '--tp-backend', 'process', *flags])

    def preview(self, *flags):
        with mock.patch.object(sys, 'argv', ['ftllm', 'serve', 'missing-model', '--tp-backend', 'process',
                                            '--parallel-plan', *flags]), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(), 0)
        self.factory.assert_not_called()
        self.session.assert_not_called()
        return json.loads(output.getvalue())

    def test_default_preserves_old_backend_and_no_device_query(self):
        args = self.parser.parse_args(['model'])
        self.assertEqual((args.dp, args.tp, args.tp_backend, args.parallel_plan), (1, '', 'thread', False))
        self.assertIsNone(parallel_config.prepare_parallel_entry(args))
        self.query.assert_not_called()

    def test_pure_dp_two_and_four_replicas(self):
        for count in (2, 4):
            with self.subTest(dp=count):
                result = self.preview('--dp', str(count))
                self.assertEqual((result['dp_size'], result['tp_size'], result['total_gpus']), (count, 1, count))
                self.assertEqual([[d['logical_id'] for d in r['devices']] for r in result['replicas']],
                                 [[i] for i in range(count)])
                self.assertFalse(result['multi_replica_execution_available'])
                self.assertTrue(result['preview_only'])

    def test_dp_tp_combination_uses_disjoint_replicas(self):
        result = self.preview('--dp', '2', '--tp', '2')
        self.assertEqual((result['dp_size'], result['tp_size'], result['total_gpus']), (2, 2, 4))
        self.assertEqual([[d['logical_id'] for d in r['devices']] for r in result['replicas']], [[0, 1], [2, 3]])
        self.assertEqual(len({d['uuid'] for r in result['replicas'] for d in r['devices']}), 4)

    def test_unused_gpus_are_not_assigned(self):
        result = self.preview('--dp', '2', '--tp', '1')
        self.assertEqual(result['visible_device_count'], 4)
        self.assertEqual([r['devices'][0]['uuid'] for r in result['replicas']],
                         [d['uuid'] for d in inventory()['devices'][:2]])

    def test_cuda_order_is_taken_from_driver_not_physical_ids(self):
        self.query.return_value = inventory((7, 3, 5, 1))
        for mask in ('7,3,5,1', 'GPU-a,GPU-b,GPU-c,GPU-d'):
            with self.subTest(mask=mask), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': mask}):
                result = self.preview('--dp', '2', '--tp', '2')
                self.assertEqual(result['cuda_visible_devices'], mask)
                self.assertEqual([d['uuid'] for d in result['replicas'][1]['devices']],
                                 [str(uuid.UUID(int=6)), str(uuid.UUID(int=2))])

    def test_no_mask_uses_actual_visible_devices(self):
        del os.environ['CUDA_VISIBLE_DEVICES']
        self.assertIsNone(self.preview('--dp', '2')['cuda_visible_devices'])

    def test_driver_count_not_environment_length_controls_capacity(self):
        self.query.return_value = inventory((0,))
        with self.assertRaisesRegex(ValueError, 'lists 4 entries but CUDA exposes 1'):
            self.preview('--dp', '2')
        del os.environ['CUDA_VISIBLE_DEVICES']
        with self.assertRaisesRegex(ValueError, 'needs 2, found 1'):
            self.preview('--dp', '2')

    def test_truncated_or_stale_mask_is_rejected_even_with_enough_gpus(self):
        for mask in ('0,1,invalid', '0,1,GPU-alias-of-0', '0'):
            with self.subTest(mask=mask), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': mask}):
                self.query.return_value = inventory((0, 1))
                with self.assertRaisesRegex(ValueError, 'CUDA exposes 2'):
                    self.preview('--dp', '2')

    def test_empty_inventory_and_driver_error_are_not_fallbacks(self):
        self.query.return_value = inventory(())
        with self.assertRaisesRegex(ValueError, 'no visible CUDA'):
            self.preview('--dp', '2')
        self.query.side_effect = RuntimeError('cuInit failed')
        with self.assertRaisesRegex(RuntimeError, 'cuInit failed'):
            self.preview('--dp', '2')

    def test_repeated_uuid_is_rejected(self):
        self.query.return_value = inventory((1, 1, 2, 3))
        with self.assertRaisesRegex(ValueError, 'duplicate CUDA device UUIDs'):
            self.preview('--dp', '2')

    def test_invalid_and_duplicate_visibility_masks(self):
        for mask in ('', '-1', '0,-1', '0,', ',0', '0,,1', '0,0', '00,0', 'GPU-a,GPU-a'):
            with self.subTest(mask=mask), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': mask}):
                with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES'):
                    self.preview('--dp', '2')
        self.query.assert_not_called()

    def test_cli_dp_requires_positive_integer(self):
        for value in ('0', '-1', '1.5', 'true'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args('--dp', value)

    def test_dp_tp_types_and_boundaries(self):
        for value in ('0', '-1', 'auto', '0,1', 'cuda:0,1', '65', '1.5', True, False, 1.0, [1]):
            with self.subTest(value=value):
                args = self.args('--dp', '2')
                args.tp = value
                with self.assertRaisesRegex(ValueError, 'DP>1 requires --tp'):
                    parallel_config.build_parallel_plan(args)
        self.query.assert_not_called()

    def test_large_tp_checks_capacity_before_allocation(self):
        with self.assertRaisesRegex(ValueError, 'needs 128, found 4'):
            self.preview('--dp', '2', '--tp', '64')

    def test_backend_device_and_operation_conflicts(self):
        for flags, message in ((['--tp-backend', 'thread'], 'requires --tp-backend process'),
                               (['--device', 'cuda:0'], 'cannot combine'),
                               (['--moe_device', 'cuda:0'], 'cannot combine'),
                               (['--custom-allreduce'], 'requires process TP=2')):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, message):
                self.preview('--dp', '2', *flags)
        result = self.preview('--dp', '2', '--tp', '2', '--custom-allreduce', '--cuda-graph')
        self.assertEqual(result['tp_size'], 2)

    def test_single_process_replica_legacy_device_forms(self):
        for value, expected in (('0', [0]), ('1', [0]), ('2', [0, 1]), ('1,0', [1, 0]), ('auto', [0, 1, 2, 3])):
            with self.subTest(tp=value):
                result = self.preview('--tp', value)
                self.assertEqual([d['logical_id'] for d in result['replicas'][0]['devices']], expected)

    def test_single_thread_replica_preview_and_no_silent_clamp(self):
        for value, expected in (('', [0]), ('1', [0]), ('2', [0, 1]), ('1,0', [1, 0]), ('auto', [0, 1, 2, 3])):
            with self.subTest(tp=value):
                result = self.preview('--tp-backend', 'thread', '--tp', value)
                self.assertEqual([d['logical_id'] for d in result['replicas'][0]['devices']], expected)
        for value in ('5', '0,0', '4,5', '0:2,1:1'):
            with self.subTest(tp=value), self.assertRaises(ValueError):
                self.preview('--tp-backend', 'thread', '--tp', value)

    def test_single_process_preview_missing_tp_keeps_existing_error(self):
        with self.assertRaisesRegex(ValueError, 'requires --tp'):
            self.preview()

    def test_dp_inference_cannot_silently_execute_one_replica(self):
        for entry in (make_normal_llm_model, process_tp.ProcessTPModel):
            with self.subTest(entry=entry.__name__), self.assertRaisesRegex(NotImplementedError, '多副本执行尚未接入'):
                entry(self.args('--dp', '2'))
        with mock.patch.object(sys, 'argv', ['ftllm', 'serve', 'missing', '--tp-backend', 'process', '--dp', '2']):
            with self.assertRaisesRegex(NotImplementedError, '多副本执行尚未接入'):
                cli.main()
        self.factory.assert_not_called()
        self.session.assert_not_called()

    def test_invalid_config_precedes_model_lookup(self):
        for entry in (make_normal_llm_model, process_tp.ProcessTPModel):
            with self.subTest(entry=entry.__name__), self.assertRaisesRegex(ValueError, 'needs 8, found 4'):
                entry(self.args('--dp', '4', '--tp', '2'))
        self.factory.assert_not_called()

    def test_factory_never_returns_preview_as_model(self):
        with self.assertRaisesRegex(ValueError, 'CLI preview'):
            make_normal_llm_model(self.args('--dp', '2', '--parallel-plan'))
        self.query.assert_not_called()

    def test_config_json_is_validated_through_both_entrypoints(self):
        path = self.root / 'options.json'
        for value in (0, -1, True, False, 2.0, '2', None, []):
            path.write_text(json.dumps({'model': 'missing', 'dp': value, 'tp_backend': 'process'}))
            for entry in (make_normal_llm_model, process_tp.ProcessTPModel):
                with self.subTest(value=value, entry=entry.__name__), self.assertRaisesRegex(ValueError, '--dp'):
                    entry(self.parser.parse_args([str(path)]))
            with mock.patch.object(sys, 'argv', ['ftllm', 'serve', str(path)]):
                with self.assertRaisesRegex(ValueError, '--dp'):
                    cli.main()
        self.query.assert_not_called()

    def test_json_boolean_flags_cannot_be_strings(self):
        for key in ('parallel_plan', 'cuda_graph', 'custom_allreduce'):
            args = self.args('--dp', '2')
            setattr(args, key, 'false')
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'boolean'):
                parallel_config.prepare_parallel_entry(args)

    def test_json_preview_aliases_and_existing_config_precedence(self):
        path = self.root / 'options.json'
        for key in ('parallel_plan', 'parallel-plan'):
            path.write_text(json.dumps({'model': 'missing', 'dp': 2, 'tp': '1', 'tp_backend': 'process', key: True}))
            with mock.patch.object(sys, 'argv', ['ftllm', 'run', str(path), '--dp', '3']), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(), 0)
            self.assertEqual(json.loads(output.getvalue())['dp_size'], 2)
        path.write_text(json.dumps({'parallel_plan': False, 'parallel-plan': True}))
        with self.assertRaisesRegex(ValueError, 'both parallel-plan and parallel_plan'):
            make_normal_llm_model(self.parser.parse_args([str(path)]))

    def test_config_export_roundtrip_preserves_defaults(self):
        path = self.root / 'defaults.json'
        with redirect_stdout(io.StringIO()):
            cli.save_defaults_to_json(self.parser, str(path))
        args = self.parser.parse_args([str(path)])
        from fastllm_pytools.util import load_model_args
        load_model_args(args)
        self.assertEqual(args.dp, 1)
        self.assertIs(args.parallel_plan, False)
        self.assertIsNone(parallel_config.prepare_parallel_entry(args))
        self.query.assert_not_called()

    def test_preview_routes_all_formal_inference_commands(self):
        for command in ('chat', 'run', 'serve', 'server'):
            with self.subTest(command=command), mock.patch.object(sys, 'argv',
                    ['ftllm', command, '--dp', '2', '--tp-backend', 'process', '--parallel-plan']), \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(), 0)
                self.assertEqual(json.loads(output.getvalue())['dp_size'], 2)
        for command in ('export', 'bench', 'webui'):
            args = self.args('--dp', '2', '--parallel-plan')
            args.command = command
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'supports chat/run/server/serve'):
                parallel_config.prepare_parallel_entry(args, allow_preview=True)

    def test_single_replica_normal_entry_does_not_probe_gpu(self):
        for backend in ('thread', 'process'):
            args = self.parser.parse_args(['model', '--tp-backend', backend, '--tp', 'auto'])
            self.assertIsNone(parallel_config.prepare_parallel_entry(args))
        self.query.assert_not_called()

    def test_packaged_preview_needs_no_model_library_or_worker(self):
        package = self.root / 'package'
        shutil.copytree(ROOT / 'tools/fastllm_pytools', package / 'ftllm', ignore=shutil.ignore_patterns('__pycache__'))
        script = ("import sys\nfrom unittest.mock import patch\nfrom ftllm import cli\n"
                  "sys.argv=['ftllm','serve','--dp','2','--tp-backend','process','--parallel-plan']\n"
                  f"with patch('ftllm.parallel_config.cuda_inventory', return_value={inventory()!r}):\n"
                  "    cli.main()\n"
                  "assert not any(x in sys.modules for x in ('torch','transformers','ftllm.llm','ftllm.process_tp'))\n")
        result = subprocess.run([sys.executable, '-c', script], cwd=self.root,
                                env=dict(os.environ, PYTHONPATH=str(package)), capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['total_gpus'], 2)


class _Function:
    def __init__(self, name, driver):
        self.name, self.driver = name, driver

    def __call__(self, *args):
        self.driver.calls.append(self.name)
        if self.name in self.driver.errors:
            return self.driver.errors[self.name]
        if self.name == 'cuDriverGetVersion':
            args[0]._obj.value = 12060
        elif self.name == 'cuDeviceGetCount':
            args[0]._obj.value = len(self.driver.order)
        elif self.name == 'cuDeviceGet':
            args[0]._obj.value = self.driver.order[args[1]]
        elif self.name == 'cuDeviceGetUuid_v2':
            args[0]._obj.bytes[:] = uuid.UUID(int=args[1] + 1).bytes
        elif self.name == 'cuDeviceGetName':
            args[0].value = f'GPU {args[2]}'.encode()
        return 0


class _Driver:
    def __init__(self, order=(7, 3)):
        self.order, self.calls, self.errors = order, [], {}
        for name in ('cuInit', 'cuDriverGetVersion', 'cuDeviceGetCount', 'cuDeviceGet', 'cuDeviceGetUuid_v2', 'cuDeviceGetName'):
            setattr(self, name, _Function(name, self))


class CudaInventoryTests(unittest.TestCase):
    def test_driver_ordinals_uuid_and_no_context_creation(self):
        driver = _Driver()
        with mock.patch.object(ctypes, 'CDLL', return_value=driver):
            self.assertEqual(parallel_config.cuda_inventory(),
                             {'driver_cuda_version': 12060, 'devices': [
                                 {'logical_id': 0, 'uuid': str(uuid.UUID(int=8)), 'name': 'GPU 7'},
                                 {'logical_id': 1, 'uuid': str(uuid.UUID(int=4)), 'name': 'GPU 3'}]})
        self.assertEqual(driver.calls, ['cuDriverGetVersion', 'cuInit', 'cuDeviceGetCount'] +
                         ['cuDeviceGet', 'cuDeviceGetUuid_v2', 'cuDeviceGetName'] * 2)
        self.assertEqual(driver.cuInit.argtypes, [ctypes.c_uint])

    def test_no_device_and_each_api_error(self):
        driver = _Driver()
        driver.errors['cuInit'] = 100
        with mock.patch.object(ctypes, 'CDLL', return_value=driver):
            self.assertEqual(parallel_config.cuda_inventory()['devices'], [])
        for name in ('cuInit', 'cuDriverGetVersion', 'cuDeviceGetCount', 'cuDeviceGet', 'cuDeviceGetUuid_v2', 'cuDeviceGetName'):
            driver = _Driver()
            driver.errors[name] = 999
            with self.subTest(api=name), mock.patch.object(ctypes, 'CDLL', return_value=driver):
                with self.assertRaisesRegex(RuntimeError, name + ' failed'):
                    parallel_config.cuda_inventory()

    def test_driver_missing_or_symbol_missing(self):
        with mock.patch.object(ctypes, 'CDLL', side_effect=OSError('missing')):
            with self.assertRaisesRegex(RuntimeError, 'cannot query CUDA devices'):
                parallel_config.cuda_inventory()
        driver = _Driver()
        del driver.cuDeviceGetUuid_v2
        with mock.patch.object(ctypes, 'CDLL', return_value=driver):
            with self.assertRaisesRegex(RuntimeError, 'missing cuDeviceGetUuid_v2'):
                parallel_config.cuda_inventory()


if __name__ == '__main__':
    unittest.main()
