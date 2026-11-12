"""测速器的纯Python回归：正式模型工厂+协议替身，不编译、不作为GPU性能结果。"""

from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import unittest
from unittest import mock
import zipfile

import test_process_dp as dp_tests
import benchmark_qwen3_dp as bench


class BenchmarkDPTests(unittest.TestCase):
    def setUp(self):
        fixture = dp_tests.ProcessDPTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.rename_model('dp_smoke')
        self.fixture, self.root = fixture, fixture.root
        self.cache = self.root / 'CMakeCache.txt'
        self.cache.write_text('CUDA_ARCH:UNINITIALIZED=70\n')
        self.base = ['--model', str(fixture.model), '--generate', str(fixture.fixture.fixture.binary),
                     '--worker', str(fixture.fixture.fixture.binary), '--build-cache', str(self.cache),
                     '--requests', '4', '--max-new-tokens', '2', '--output', str(self.root / 'reports')]

    def invoke(self, *flags):
        with mock.patch.object(bench, 'capture', return_value='Python protocol fixture; no GPU'), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = bench.main(self.base + list(flags))
        directory = next((self.root / 'reports').iterdir())
        return status, directory, json.loads((directory / 'summary.json').read_text())

    def test_full_matrix_factory_reports_and_ratios(self):
        status, output, report = self.invoke()
        self.assertEqual(status, 0, report)
        self.assertEqual((report['status'], report['complete'], len(report['cases'])), ('PASS', True, 20))
        by_name = {row['case']: row for row in report['cases']}
        for row in report['cases']:
            self.assertEqual(len(row['samples']), 5)
            self.assertEqual(row['failed_elements'], 0)
            baseline = by_name[row['baseline_case']]
            self.assertEqual((baseline['profile'], baseline['tp'], baseline['concurrency'], baseline['dp']),
                             (row['profile'], row['tp'], row['concurrency'], 1))
            self.assertAlmostEqual(row['speedup_vs_dp1'], baseline['median_batch_ms'] / row['median_batch_ms'])
            self.assertAlmostEqual(row['output_tokens_per_s_per_gpu'], row['output_tokens_per_s'] / row['gpu_count'])
        self.assertEqual(report['metadata']['request']['eos_token_ids'], [])
        self.assertEqual(report['metadata']['compile_architecture'], ['CUDA_ARCH:UNINITIALIZED=70'])
        self.assertTrue((output / 'summary.md').exists())
        self.assertTrue((output / 'summary.csv').exists())
        self.assertTrue((output / 'samples.csv').exists())
        with zipfile.ZipFile(output / 'summary.xlsx') as workbook:
            self.assertIn('Samples', workbook.read('xl/workbook.xml').decode())
        self.assertFalse(list((output / 'models').rglob('request-*.json')))
        raw = json.loads(next((output / 'raw').glob('*_r0_i0.json')).read_text())
        self.assertTrue(all(record['report']['token_ids'] == [1, 1] for record in raw['records']))

    def test_statistics_uses_wall_time_and_preserves_units(self):
        samples = [dict(elapsed_s=s, latency_ms=[10., 30.], ttft_ms=[2., 4.], itl_ms=[1., 3.]) for s in (1., 2., 3., 4., 5.)]
        row = bench.statistics_row(samples, requests=12, tokens=10, gpus=4)
        self.assertEqual(row['median_batch_ms'], 3000.)
        self.assertEqual(row['p95_batch_ms'], 4800.)
        self.assertEqual(row['output_tokens_per_s'], 40.)
        self.assertEqual(row['output_tokens_per_s_per_gpu'], 10.)
        self.assertEqual(row['requests_per_s'], 4.)
        self.assertEqual(row['median_request_ms'], 20.)
        self.assertEqual(row['p95_request_ms'], 30.)
        self.assertAlmostEqual(row['batch_cv_pct'], 100 * (2 ** .5) / 3)

    def test_bad_matrix_or_workload_rejected_before_model(self):
        cases = [('--configurations', '2x1'), ('--configurations', '1x3'), ('--configurations', '1x1', '1x1'),
                 ('--repeats', '4'), ('--warmup', '0'), ('--requests', '3'), ('--concurrency', '0'),
                 ('--concurrency', '1', '1'), ('--max-new-tokens', '1'), ('--timeout', 'nan')]
        for flags in cases:
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.parse_args(self.base + list(flags))

    def test_bad_token_output_is_failed_with_raw_evidence(self):
        original = bench.run_wave
        calls = 0

        def corrupt(*args, **kwargs):
            nonlocal calls
            wave = original(*args, **kwargs)
            calls += 1
            if calls == 3:  # 预热、第一轮成功后破坏第二轮。
                wave['records'][0]['report']['token_ids'][0] += 1
            return wave

        with mock.patch.object(bench, 'run_wave', side_effect=corrupt):
            status, output, report = self.invoke('--configurations', '1x1', '--profiles', 'common', '--concurrency', '1')
        self.assertEqual(status, 1)
        row = report['cases'][0]
        self.assertEqual((row['status'], row['failed_elements'], len(row['samples'])), ('FAIL', 1, 1))
        self.assertEqual(row['max_abs_error'], 1.)
        self.assertTrue((output / 'raw/common_dp1_tp1_c1_r1_i0.json').exists())
        self.assertTrue((output / 'summary.xlsx').exists())

    def test_missing_graph_evidence_cannot_pass(self):
        original = bench.run_wave

        def corrupt(*args, **kwargs):
            wave = original(*args, **kwargs)
            wave['records'][0]['report']['graph'][0]['replays'] = 0
            return wave

        with mock.patch.object(bench, 'run_wave', side_effect=corrupt):
            status, _, report = self.invoke('--configurations', '1x1', '--profiles', 'optimized', '--concurrency', '1')
        self.assertEqual(status, 1)
        self.assertIn('Graph', report['cases'][0]['reason'])

    def test_cold_prefix_cannot_be_reported_as_warm(self):
        original = bench.check_wave

        def corrupt(wave, *args, **kwargs):
            if kwargs.get('hot'):
                wave = copy.deepcopy(wave)
                for state in wave['records'][0]['report']['graph']:
                    state['prefix_cache']['last_reused_tokens'] = 0
            return original(wave, *args, **kwargs)

        with mock.patch.object(bench, 'check_wave', side_effect=corrupt):
            status, _, report = self.invoke('--configurations', '1x1', '--profiles', 'optimized', '--concurrency', '1')
        self.assertEqual(status, 1)
        self.assertIn('warm prefix', report['cases'][0]['reason'])

    def test_startup_failure_saves_partial_report(self):
        with mock.patch.object(bench, 'make_normal_llm_model', side_effect=RuntimeError('test startup failure')):
            status, output, report = self.invoke()
        self.assertEqual((status, report['status'], report['complete']), (1, 'FAIL', False))
        self.assertIn('test startup failure', report['cases'][0]['reason'])
        self.assertTrue((output / 'summary.xlsx').exists())

    def test_interrupted_wave_closes_model_before_waiting_for_pool(self):
        model = mock.Mock()
        model.generate.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            bench.run_wave(model, dict(max_new_tokens=2), 1, 1)
        model.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
