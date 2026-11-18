"""并发诊断的纯Python回归：模拟正式会话结果，不编译、不启动GPU。"""

from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import probe_qwen_moe_concurrency as probe


class ProbeTests(unittest.TestCase):
    def run_probe(self, fault=None):
        """/** 通过真实CLI和四格式报告，模拟两种模式在不同轮次出现同一输出变体。 */"""
        calls = []

        class Session:
            def __init__(self, binary, worker, model, devices, output, **options):
                calls.append(options)
                self.output, self.options = output, options
                self.count, self.lock = 0, threading.Lock()
                self._states = [dict(captures=int(options['cuda_graph']), replays=int(options['cuda_graph']),
                                     prefix_cache=dict(last_reused_tokens=0)) for _ in range(4)]

            def __enter__(self):
                self.output.mkdir()
                return self

            def __exit__(self, *args):
                self.shutdown = dict(shutdown_complete=True)

            def generate(self, job, teacher_forced_tokens=None):
                assert teacher_forced_tokens is None
                with self.lock:
                    self.count += 1
                    number = self.count
                repeat, rank = divmod(number-1, 4)
                if fault == 'generation' and not self.options['cuda_graph'] and repeat == 2:
                    raise RuntimeError('injected worker failure')
                if fault == 'interrupt' and repeat == 2:
                    raise KeyboardInterrupt()
                change = fault != 'stable' and repeat == (5 if self.options['cuda_graph'] else 2) and rank < 2
                tokens = [1, 1, 2, 3] if change else [1, 1, 1, 1]
                steps = []
                for i, token in enumerate(tokens, 1):
                    step = dict(token_id=token, execution='decode_cuda_graph' if self.options['cuda_graph'] and i > 1 else 'eager',
                        joint_round=dict(actions=[1 if i == 1 else 2]*4, submitted_token_counts=[2 if i == 1 else 1]*4),
                        trace_schedule=dict(round_1based=number*5+i, request_ids=[number]*4,
                                            token_positions_1based=[i]*4, input_lengths=[i+1]*4))
                    if i in self.options['trace_token_positions']:
                        logits = [0.0]*8
                        logits[1], logits[2] = 10.0, 10.001 if change and i == 3 else 9.999
                        if i == 4 and change:
                            logits[3] = 20.0
                        step['logits'] = logits
                    steps.append(step)
                if fault == 'missing' and repeat == 1:
                    steps[2].pop('logits', None)
                report = dict(id=number, dp_rank=rank, status='PASS', request=copy.deepcopy(job), token_ids=tokens,
                              steps=steps, states=self._states, teacher_forced_tokens=None)
                (self.output / f'request-{number:06d}.json').write_text(json.dumps(report))
                return report

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, build = root / 'model', root / 'build'
            model.mkdir()
            build.mkdir()
            (model / 'config.json').write_text('{"vocab_size": 8}')
            for name in ('CMakeCache.txt', 'fastllm_moe_generate', 'fastllm_tp_worker'):
                (build / name).write_text('mock data, never executable')
            argv = ['probe', '--model', str(model), '--build', str(build), '--prompt-length', '2',
                    '--output-tokens', '4', '--trace-tokens', '2', '3', '4', '--output', str(root / 'out')]
            with mock.patch('sys.argv', argv), mock.patch.object(probe, 'capture', return_value='mock hardware'), \
                    mock.patch.object(probe, 'QwenMoeProcessSession', Session), \
                    mock.patch.dict('sys.modules', {'transformers': None, 'torch': None}), redirect_stdout(io.StringIO()):
                code = probe.main()
            output, = (root / 'out').iterdir()
            report = json.loads((output / 'summary.json').read_text())
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((output / f'summary.{suffix}').is_file())
            details = {p.stem: json.loads(p.read_text()) for p in output.glob('*_vs_*.json')}
            raw_count = len(list(output.glob('*/request-*.json')))
        self.assertTrue(all(o['trace'] and not o['prefix_cache'] and o['routing'] == 'round_robin' for o in calls))
        self.assertTrue(all(o['trace_token_positions'] == [2, 3, 4] for o in calls))
        return code, report, details, raw_count

    def test_same_variant_counts_do_not_hide_per_request_failures(self):
        code, report, details, count = self.run_probe()
        self.assertEqual(code, 1)
        self.assertTrue(report['complete'])
        self.assertEqual(report['expected_case_count'], 60)
        self.assertEqual(report['metadata']['reproduction_status'], 'OBSERVED_TOKEN_DIVERGENCE')
        self.assertEqual(report['metadata']['eager_variants'], {'V1': 18, 'V2': 2})
        self.assertEqual(report['metadata']['graph_variants'], {'V1': 18, 'V2': 2})
        rows = report['cases']
        self.assertEqual(sum(r['status'] == 'FAIL' for r in rows if r['case'].endswith('_vs_first')), 4)
        self.assertEqual(sum(r['status'] == 'FAIL' for r in rows if r['case'].endswith('_vs_eager')), 4)
        differing = next(d for d in details.values() if d['first_divergence_trace'] is not None)
        self.assertEqual(differing['first_divergence_trace']['token_1based'], 3)
        self.assertTrue(differing['first_divergence_trace']['same_input_history'])
        self.assertEqual(differing['traces'][-1]['numeric_comparison'], 'SKIP: different generated input histories')
        self.assertEqual(differing['compared_steps'], 2)
        self.assertAlmostEqual(differing['max_abs_error'], .002)
        self.assertEqual(count, 48)
        self.assertTrue(all('median_ms' not in row and 'speedup' not in row for row in rows))

    def test_stable_trace_reports_not_observed_not_a_fix(self):
        code, report, _, _ = self.run_probe('stable')
        self.assertEqual(code, 0)
        self.assertEqual(report['metadata']['reproduction_status'], 'NOT_OBSERVED')
        self.assertIn('未复现不证明原问题消失', report['metadata']['report_description'])

    def test_missing_trace_and_worker_failure_keep_partial_reports(self):
        for fault in ('missing', 'generation', 'interrupt'):
            with self.subTest(fault=fault):
                code, report, _, raw = self.run_probe(fault)
                self.assertEqual(code, 1)
                self.assertFalse(report['complete'])
                self.assertGreater(raw, 0)
                self.assertEqual(report['status'], 'FAIL')

    def test_invalid_options_never_start_native(self):
        for extra in (['--trace-tokens', '0'], ['--trace-tokens', '2', '2'], ['--repeats', '4'],
                      ['--prompt-length', '2048'], ['--timeout', 'nan']):
            with self.subTest(extra=extra), mock.patch('sys.argv', ['probe', '--model', '/unused', *extra]), \
                    mock.patch.object(probe, 'QwenMoeProcessSession') as session, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    probe.main()
                self.assertEqual(error.exception.code, 2)
                session.assert_not_called()

    def test_uncaptured_divergence_and_invalid_logits_are_not_treated_as_near_ties(self):
        """/** 未采集首次分歧时不猜测候选；分叉历史不计算数值误差，非法logits必须拒绝。 */"""
        step = dict(logits=[1., 2.], execution='eager', joint_round={}, trace_schedule={})
        a = dict(request={'token_ids': [1]}, token_ids=[0, 1], steps=[copy.deepcopy(step), copy.deepcopy(step)])
        b = dict(request={'token_ids': [1]}, token_ids=[1, 1], steps=[copy.deepcopy(step), copy.deepcopy(step)])
        result = probe.compare_requests(a, b, [2])
        self.assertFalse(result['token_comparison']['match'])
        self.assertIsNone(result['first_divergence_trace'])
        self.assertEqual(result['compared_steps'], 0)
        self.assertIsNone(result['max_abs_error'])
        for values in ([], [1.], [float('nan'), 2.], [float('inf'), 2.]):
            a['steps'][1]['logits'] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                probe.compare_requests(a, b, [2])


if __name__ == '__main__':
    unittest.main()
