"""批次不变模式的配置、拒绝路径和验收报告回归；仅 CPU，不编译或执行 CUDA。"""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

import test_parallel_config
from fastllm_pytools.moe_batch_invariant import BACKEND, CONFIG, validate_ready
from fastllm_pytools.parallel_config import prepare_parallel_entry
from fastllm_pytools.util import make_normal_parser
import run_qwen_moe_batch_invariant as full
import run_sm70_expert_gemm as gemm
from run_qwen_moe_batch_performance import compare_samples


def ready():
    """/** 构造包含所有 worker 实际配置的原生加载回执。 */"""
    return dict(batch_invariant=dict(requested=True, effective=True, backend=BACKEND, config=CONFIG.copy(), scope='moe_experts'),
                workers=[dict(batch_invariant=True, expert_gemm_config=CONFIG.copy()) for _ in range(4)])


class BatchInvariantTests(unittest.TestCase):
    def test_reject_missing_fallback_and_rank_mismatch(self):
        validate_ready(ready(), True)
        validate_ready({}, False)
        for fault in ('missing', 'fallback', 'wrong_backend', 'wrong_tile', 'rank', 'unexpected'):
            report = ready()
            if fault == 'missing':
                report.pop('batch_invariant')
            elif fault == 'fallback':
                report['batch_invariant']['effective'] = False
            elif fault == 'wrong_backend':
                report['batch_invariant']['backend'] = 'cublas'
            elif fault == 'wrong_tile':
                report['batch_invariant']['config'][1] = 32
            elif fault == 'rank':
                report['workers'][2]['expert_gemm_config'][1] = 32
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                validate_ready(report, fault != 'unexpected')

    def test_public_mode_defaults_and_rejects_dense(self):
        parser = make_normal_parser('batch invariant tests')
        self.assertFalse(parser.parse_args([]).batch_invariant)
        with self.assertRaisesRegex(ValueError, 'Attention DP/EP'):
            prepare_parallel_entry(parser.parse_args(['--batch-invariant']))
        args = parser.parse_args(['--batch-invariant', '--tp-backend', 'process', '--attention-dp', '4', '--ep', '4'])
        prepare_parallel_entry(args)
        self.assertTrue(args.batch_invariant)

    def test_bitwise_and_numeric_are_separate(self):
        positive = np.array([0.0, 1.0], dtype=np.float16)
        negative = np.array([-0.0, 1.0], dtype=np.float16)
        self.assertEqual(gemm.metrics(positive, negative)['status'], 'PASS')
        self.assertEqual(gemm.metrics(positive, negative, bitwise=True)['failed_elements'], 1)
        bad = np.array([0.0, 2.0], dtype=np.float16)
        self.assertEqual(gemm.metrics(bad, positive)['status'], 'FAIL')
        with self.assertRaises(ValueError):
            gemm.metrics(np.array([np.nan]), np.array([0.0]))

    def test_full_token_sequence_and_length_are_checked(self):
        self.assertEqual(full.compare_tokens([1, 2, 3], [1, 2, 3])['token_mismatches'], 0)
        difference = full.compare_tokens([1, 2, 4], [1, 2, 3, 5])
        self.assertEqual(difference['first_divergence_token_1based'], 3)
        self.assertEqual(difference['token_mismatches'], 2)

    def test_hot_cache_checks_active_request_not_idle_rank_hits(self):
        states = [dict(request_active=False, expert_gemm_backend=BACKEND, expert_gemm_submissions=2,
                       prefix_cache=dict(hits=0, last_reused_tokens=0)) for _ in range(4)]
        states[0]['prefix_cache'].update(hits=1, last_reused_tokens=3)
        report = dict(status='PASS', dp_rank=0, token_ids=[1, 2, 3], states=states,
                      request=dict(token_ids=[1, 2, 3, 4], max_new_tokens=3))
        session = mock.Mock(_states=states)
        full.verify_paths(session, [report], False, 'hot')
        states[0]['prefix_cache']['last_reused_tokens'] = 0
        with self.assertRaisesRegex(AssertionError, 'expected prefix'):
            full.verify_paths(session, [report], False, 'hot')

    def test_speedup_requires_identical_histories_and_cache(self):
        sample = dict(requests=[{'token_ids': [1]}], prompt_length=1, output_tokens=2, concurrency=1,
                      cuda_graph=False, cache_state='off', warmup=1, iterations=1, repeats=5,
                      reused_tokens=[[0]] * 5, responses=[[{'token_ids': [3, 4]}]] * 5, samples_ms=[2.] * 5)
        candidate = copy.deepcopy(sample)
        candidate['samples_ms'] = [1.] * 5
        result = compare_samples(sample, candidate, 'PASS', 'PASS')
        self.assertEqual(result['speedup'], 2.)
        candidate['responses'][0][0]['token_ids'][1] = 5
        different = compare_samples(sample, candidate, 'PASS', 'PASS')
        self.assertIsNone(different['speedup'])
        self.assertTrue(different['measurements_valid'])
        candidate = copy.deepcopy(sample)
        candidate['reused_tokens'] = [[1]] * 5
        self.assertIsNone(compare_samples(sample, candidate, 'PASS', 'PASS')['speedup'])
        self.assertIsNone(compare_samples(sample, sample, 'FAIL', 'PASS')['speedup'])

    def test_incomplete_measurements_cannot_pass_without_speedup(self):
        sample = dict(requests=[{'token_ids': [1]}], prompt_length=1, output_tokens=2, concurrency=1,
                      cuda_graph=False, cache_state='off', warmup=1, iterations=1, repeats=5,
                      reused_tokens=[[0]] * 5, responses=[[{'token_ids': [3, 4]}]] * 5, samples_ms=[2.] * 5)
        for fault in ('repeats', 'requests', 'tokens'):
            candidate = copy.deepcopy(sample)
            if fault == 'repeats':
                candidate['responses'].pop()
            elif fault == 'requests':
                candidate['responses'][0] = []
            else:
                candidate['responses'][0] = [{'token_ids': [3]}]
            with self.subTest(fault=fault):
                result = compare_samples(sample, candidate, 'PASS', 'PASS')
                self.assertFalse(result['measurements_valid'])
                self.assertIsNone(result['speedup'])

    def test_safetensors_bfloat_conversion_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = json.dumps({'weight': dict(dtype='BF16', shape=[1, 2], data_offsets=[0, 4])}).encode()
            path = root / 'model.safetensors'
            path.write_bytes(struct.pack('<Q', len(header)) + header + np.array([0x3f80, 0xc000], dtype='<u2').tobytes())
            np.testing.assert_array_equal(gemm.load_weight(root, {'weight': path.name}, 'weight'), [[1, -2]])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'offsets'):
                gemm.load_weight(root, {'weight': path.name}, 'weight')

    def test_failure_is_nonzero_and_four_reports_survive(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / 'CMakeCache.txt').write_text('CUDA_ARCH:STRING=70\n')
                sessions = []

                class Session:
                    """/** 模拟公开会话，故意注入 Graph Token 首差以验证门禁及失败报告。 */"""

                    def __init__(self, *args, **kwargs):
                        self.output, self.graph = args[4], kwargs['cuda_graph']
                        self.batch_invariant = kwargs['batch_invariant']
                        self.ready = ready()
                        self.shutdown = dict(shutdown_complete=True)
                        self._states = [dict(request_active=False, expert_gemm_backend=BACKEND,
                            expert_gemm_submissions=2, captures=int(self.graph), replays=int(self.graph),
                            prefix_cache=dict(hits=0, last_reused_tokens=0)) for _ in range(4)]
                        sessions.append(self)

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def generate(self, job, **kwargs):
                        return dict(status='PASS', request=job, token_ids=[1, 2, 9 if fail and self.graph else 3],
                                    dp_rank=0, states=copy.deepcopy(self._states), steps=[])

                argv = ['runner', '--model', str(root), '--build', str(root), '--output', str(root / 'out'),
                        '--prompt-lengths', '8', '--output-tokens', '3', '--concurrencies', '1',
                        '--cache-states', 'off', '--orders', 'simultaneous']
                with mock.patch.object(full, 'QwenMoeProcessSession', Session), mock.patch.object(full, 'capture', return_value='mock'), \
                        mock.patch('sys.argv', argv), redirect_stdout(io.StringIO()):
                    result = full.main()
                report = next((root / 'out').glob('*/summary.json'))
                data = json.loads(report.read_text())
                self.assertEqual(result, int(fail))
                self.assertEqual(data['status'], 'FAIL' if fail else 'PASS')
                self.assertTrue(data['complete'])
                self.assertFalse(data['metadata']['full_planned_matrix'])
                self.assertTrue(all(s.batch_invariant for s in sessions))
                for extension in ('json', 'csv', 'md', 'xlsx'):
                    self.assertTrue((report.parent / f'summary.{extension}').is_file())


if __name__ == '__main__':
    unittest.main()
