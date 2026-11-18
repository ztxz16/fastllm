"""专家GEMM归约诊断的CPU回归；模拟快照与执行器，不加载CUDA或编译。"""

from contextlib import contextmanager, redirect_stdout
import ctypes
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import probe_qwen_moe_gemm_reduction as probe
from run_qwen_moe_full import request
from test_qwen_moe_block_batches import CONFIG, write_blocks


class FakeReplay:
    """/** 模拟默认重放和候选归约；记录完整矩阵行数并注入基准、稳定性和中断故障。 */"""

    def __init__(self, fault=None):
        self.fault, self.calls = fault, []

    @contextmanager
    def operation(self, target):
        counts = {2: 0, 4: 0}

        def run(mode, wave):
            counts[wave] += 1
            self.calls.append((mode, wave, len(target['x'][wave])))
            if self.fault == 'interrupt' and mode == 'no_reduced':
                raise KeyboardInterrupt('injected')
            if mode == 'current' or self.fault == 'candidate':
                value = target['y'][wave].copy()
            else:
                value = target['y'][4].copy()
                if wave == 2:
                    value = value[target['positions']]
            if self.fault == 'baseline' and mode == 'current' and wave == 2:
                value[0, 0] += 1
            if self.fault == 'unstable' and mode == 'no_reduced' and counts[wave] == 3:
                value[0, 0] += 1
            if self.fault == 'nan' and mode == 'no_reduced':
                value[0, 0] = np.nan
            return value

        yield run


class ArrayTensor:
    """/** 提供权重读取测试所需的CPU张量接口，不导入PyTorch。 */"""

    def __init__(self, array):
        self.array = np.asarray(array)
        self.shape, self.dtype = self.array.shape, self.array.dtype

    def to(self, *, dtype):
        return ArrayTensor(self.array.astype(dtype))

    def contiguous(self):
        return ArrayTensor(np.ascontiguousarray(self.array))

    def numpy(self):
        return self.array


class GemmReductionTests(unittest.TestCase):
    def test_flag_comes_from_local_header_and_rejects_missing_or_invalid_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = root/'include/cublas_api.h'
            header.parent.mkdir()
            for literal in ('0x10', '16', '0', '3', 'UNKNOWN'):
                header.write_text('enum { CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = '+literal+' };')
                with self.subTest(literal=literal):
                    if literal in ('0x10', '16'):
                        flag, evidence = probe.reduction_flag(root)
                        self.assertEqual(flag, 16)
                        self.assertEqual(evidence['sha256'], probe.file_hash(header))
                    else:
                        with self.assertRaises(ValueError):
                            probe.reduction_flag(root)
            header.unlink()
            with self.assertRaises(ValueError):
                probe.reduction_flag(root)

    def test_math_mode_rejects_api_errors_and_silent_mismatch(self):
        for fault in (None, 'set', 'get', 'mismatch'):
            def get_mode(handle, result):
                ctypes.cast(result, ctypes.POINTER(ctypes.c_int))[0] = 0 if fault == 'mismatch' else 16
                return 7 if fault == 'get' else 0

            def check(status):
                if status:
                    raise RuntimeError(f'cuBLAS error {status}')

            lib = SimpleNamespace(cublasGetMathMode=mock.Mock(side_effect=get_mode),
                                  cublasSetMathMode=mock.Mock(return_value=7 if fault == 'set' else 0))
            gemm = SimpleNamespace(lib=lib, handle=ctypes.c_void_p(1), check=check)
            with self.subTest(fault=fault):
                if fault:
                    with self.assertRaises((ValueError, RuntimeError)):
                        probe.set_math_mode(gemm, 16)
                else:
                    probe.set_math_mode(gemm, 16)
                lib.cublasSetMathMode.assert_called_once_with(gemm.handle, 16)

    def test_pairing_uses_slots_and_preserves_full_batch_for_both_gemms(self):
        with tempfile.TemporaryDirectory() as temp:
            blocks = {w: write_blocks(Path(temp)/str(w), w) for w in (4, 2)}
            short, full = blocks[2][2][1], blocks[4][2][1]
            short['expert_2_indices'] = full['expert_2_indices'][[1, 3]]
            short['expert_2_packed'] = full['expert_2_packed'][[1, 3]]
            blocks[2][1][1]['expert_1_down'][0, 0] += .125
            cases = probe.targets(blocks, {'2': 'gateup', '1': 'down'})
            self.assertEqual([(c['owner'], c['phase']) for c in cases], [(1, 'down'), (2, 'gateup')])
            self.assertEqual(cases[1]['positions'].tolist(), [1, 3])
            self.assertEqual([len(cases[1]['x'][w]) for w in (2, 4)], [2, 4])
            # Expert 2的激活已经不同，不能继续把其Down差异归因于Down GEMM。
            with self.assertRaisesRegex(ValueError, 'inputs already differ'):
                probe.targets(blocks, {'2': 'down'})
            short['expert_2_indices'][-1] = 1000
            with self.assertRaisesRegex(ValueError, 'pair GEMM rows'):
                probe.targets(blocks, {'2': 'gateup'})

    def test_weights_keep_gate_up_layout_and_detect_checkpoint_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp).resolve()
            (model/'weights.safetensors').write_text('CPU fixture')
            prefix = 'model.layers.0.mlp.experts.2.'
            values = {prefix+part+'.weight': np.full((2, 2), value, dtype=np.float32)
                      for part, value in (('gate_proj', 1), ('up_proj', 2), ('down_proj', 3))}
            mapping = {name: 'weights.safetensors' for name in values}
            torch = SimpleNamespace(float16=np.float16, float32=np.float32, bfloat16='bf16',
                cat=lambda tensors, dim: ArrayTensor(np.concatenate([t.numpy() for t in tensors], axis=dim)))

            @contextmanager
            def safe_open(path, **kwargs):
                yield SimpleNamespace(get_tensor=lambda name: ArrayTensor(values[name]))

            evidence = {}
            result = probe.load_weight(torch, safe_open, model, CONFIG, mapping,
                                       dict(expert=2, phase='gateup'), evidence).numpy()
            np.testing.assert_array_equal(result, [[1, 1], [1, 1], [2, 2], [2, 2]])
            self.assertEqual(result.dtype, np.float16)
            down = probe.load_weight(torch, safe_open, model, CONFIG, mapping,
                                     dict(expert=2, phase='down'), evidence).numpy()
            np.testing.assert_array_equal(down, np.full((2, 2), 3))
            values[prefix+'gate_proj.weight'][0, 0] = 4
            with self.assertRaisesRegex(ValueError, 'changed between modes'):
                probe.load_weight(torch, safe_open, model, CONFIG, mapping, dict(expert=2, phase='gateup'), evidence)
            values[prefix+'gate_proj.weight'][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                probe.load_weight(torch, safe_open, model, CONFIG, mapping, dict(expert=2, phase='gateup'), {})

    def run_main(self, fault=None):
        """/** 用完整历史快照执行CLI；所有GPU执行均替换为CPU模拟，检查报告及失败门禁。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, build, baseline = (root/name for name in ('model', 'build', 'baseline'))
            for path in (model, build, baseline):
                path.mkdir()
            (model/'config.json').write_text(json.dumps(CONFIG))
            (model/'model.safetensors.index.json').write_text('{}')
            (build/'fastllm_tp_worker').write_text('mock, never execute')
            (build/'CMakeCache.txt').write_text('mock architecture')
            job = request([1, 2], 4)
            old = dict(scope='formal_moe_block_batch_trace_diagnostic', trace_attribution_valid=True,
                expert_pairing_valid=True, layer=0, cuda_graph=False, prefix_cache=False,
                request=job, git='native-build', traces={}, expert_first_difference={'2': 'gateup'},
                binary_sha256={'fastllm_tp_worker': probe.file_hash(build/'fastllm_tp_worker')},
                model_config_sha256=probe.file_hash(model/'config.json'),
                model_index_sha256=probe.file_hash(model/'model.safetensors.index.json'))
            for wave in (4, 2):
                directory = baseline/f'first{wave}/trace'
                write_blocks(directory, wave)
                blocks = probe.load_blocks(directory, wave, CONFIG, job)
                old['traces'][str(wave)] = {str(rank): meta for rank, (meta, _) in blocks.items()}
            if fault == 'provenance':
                old['model_config_sha256'] = 'wrong'
            if fault == 'worker':
                old['binary_sha256']['fastllm_tp_worker'] = 'wrong'
            if fault == 'trace':
                old['traces']['2']['0']['binary_sha256'] = 'wrong'
            (baseline/'summary.json').write_text(json.dumps(dict(status='FAIL', complete=True, metadata=old)))
            engine = FakeReplay(fault)
            argv = ['probe', '--baseline', str(baseline), '--model', str(model), '--build', str(build),
                    '--output', str(root/'out')]
            with mock.patch('sys.argv', argv), mock.patch.object(probe, 'capture', return_value='mock'), \
                    mock.patch.object(probe, 'GpuReplay', return_value=engine) as factory, redirect_stdout(io.StringIO()):
                code = probe.main()
            output, = (root/'out').iterdir()
            report = json.loads((output/'summary.json').read_text())
            for ext in ('json', 'csv', 'md', 'xlsx'):
                self.assertTrue((output/f'summary.{ext}').is_file())
            for sample in report['metadata'].get('samples', []):
                self.assertEqual(probe.file_hash(output/sample['path']), sample['sha256'])
            return code, report, engine.calls, factory.call_count, len(list(output.rglob('*.npy')))

    def test_baseline_failures_remain_visible_when_candidate_eliminates_difference(self):
        code, report, calls, factory, raw_count = self.run_main()
        self.assertEqual((code, factory, raw_count), (1, 1, 4))
        self.assertTrue(report['complete'])
        self.assertEqual(report['completed_case_count'], 29)
        self.assertTrue(report['metadata']['baseline_replay_passed'])
        self.assertTrue(report['metadata']['candidate_batch_invariant'])
        failed = [r for r in report['cases'] if r['status'] == 'FAIL']
        self.assertEqual(len(failed), 5)
        self.assertTrue(all(r['group'] == 'current_batch_compare' for r in failed))
        self.assertEqual({(wave, size) for _, wave, size in calls}, {(2, 2), (4, 4)})
        self.assertEqual([mode for mode, _, _ in calls], ['current']*12+['no_reduced']*12)
        self.assertEqual(len(report['metadata']['samples']), 20)

    def test_current_replay_mismatch_blocks_candidate_and_preserves_partial_report(self):
        code, report, calls, _, _ = self.run_main('baseline')
        self.assertEqual(code, 1)
        self.assertFalse(report['complete'])
        self.assertFalse(report['metadata']['baseline_replay_passed'])
        self.assertIsNone(report['metadata']['candidate_batch_invariant'])
        self.assertEqual({mode for mode, _, _ in calls}, {'current'})

    def test_candidate_batch_difference_and_repeat_instability_are_failures(self):
        for fault, group in (('candidate', 'no_reduced_batch_compare'), ('unstable', 'candidate_stability')):
            with self.subTest(fault=fault):
                code, report, _, _, _ = self.run_main(fault)
                self.assertEqual(code, 1)
                self.assertTrue(report['complete'])
                self.assertTrue(report['metadata']['baseline_replay_passed'])
                self.assertFalse(report['metadata']['candidate_batch_invariant'])
                self.assertTrue(any(r['status'] == 'FAIL' and r.get('group') == group for r in report['cases']))

    def test_provenance_rejects_changed_model_worker_or_trace_before_gpu_start(self):
        for fault in ('provenance', 'worker', 'trace'):
            with self.subTest(fault=fault):
                code, report, calls, factory, _ = self.run_main(fault)
                self.assertEqual((code, calls, factory), (1, [], 0))
                self.assertFalse(report['complete'])

    def test_interruption_and_nonfinite_values_invalidate_candidate_but_keep_reports(self):
        for fault in ('interrupt', 'nan'):
            with self.subTest(fault=fault):
                code, report, _, _, _ = self.run_main(fault)
                self.assertEqual(code, 1)
                self.assertFalse(report['complete'])
                self.assertTrue(report['metadata']['baseline_replay_passed'])
                self.assertIsNone(report['metadata']['candidate_batch_invariant'])


if __name__ == '__main__':
    unittest.main()
