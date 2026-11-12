"""混合路由验收器回归：使用正式工厂及独立Python协议进程，禁止代替GPU性能结论。"""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import unittest
from unittest import mock

import benchmark_qwen3_dp_routing as runner
import test_process_dp as fixtures


class RoutingBenchmarkTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ProcessDPTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.rename_model('dp_routing')
        self.root = fixture.root
        cache = self.root / 'CMakeCache.txt'
        cache.write_text('CUDA_ARCH:UNINITIALIZED=70\n')
        request = self.root / 'request.json'
        request.write_text(json.dumps(fixture.request))
        binary = str(fixture.fixture.fixture.binary)
        self.args = ['--model', str(fixture.model), '--generate', binary, '--worker', binary,
                     '--build-cache', str(cache), '--request-file', str(request), '--dp', '2', '--tp', '2',
                     '--output', str(self.root / 'reports')]

    def invoke(self):
        with mock.patch.object(runner.bench, 'capture', return_value='Python protocol fixture, no GPU'), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = runner.main(self.args)
        output = next((self.root / 'reports').iterdir())
        return code, output, json.loads((output / 'summary.json').read_text())

    def test_formal_routes_mixed_cache_output_and_all_report_formats(self):
        code, output, report = self.invoke()
        self.assertEqual((code, report['status'], report['complete']), (0, 'PASS', True), report)
        self.assertEqual(report['metadata']['completed_waves'], 20)
        self.assertEqual(report['metadata']['output_tokens_per_wave'], 480)
        self.assertNotIn('为全部副本建立前缀', (output / 'summary.md').read_text())
        for row in report['cases']:
            self.assertEqual(len(row['samples']), 5)
            self.assertEqual(row['failed_elements'], 0)
            self.assertGreater(row['speedup_vs_round_robin'], 0)
            self.assertEqual(row['reused_prompt_tokens'] + row['computed_prompt_tokens'], 5 * 24 * 64)
        checks = json.loads((output / 'functional/summary.json').read_text())
        self.assertEqual(checks['status'], 'PASS', checks)
        self.assertEqual(len(checks['cases']), 3)
        for folder in (output, output / 'functional'):
            for extension in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((folder / ('summary.' + extension)).exists())
        raw = json.loads((output / 'raw/round_robin_dp2_tp2_c1_0.json').read_text())
        self.assertEqual(len(raw['records']), 24)
        self.assertEqual({len(r['report']['token_ids']) for r in raw['records']}, {8, 32})
        self.assertFalse(list((output / 'models').rglob('request-*.json')))

    def test_wrong_timed_output_fails_with_raw_and_partial_report(self):
        original = runner.bench.run_requests
        count = 0

        def corrupt(*args, **kwargs):
            nonlocal count
            wave = original(*args, **kwargs)
            count += 1
            if count == 2:
                wave['records'][0]['report']['token_ids'][0] += 1
            return wave

        with mock.patch.object(runner.bench, 'run_requests', side_effect=corrupt):
            code, output, report = self.invoke()
        self.assertEqual((code, report['status'], report['complete']), (1, 'FAIL', False))
        self.assertEqual(report['cases'][0]['failed_elements'], 1)
        self.assertTrue((output / 'raw/round_robin_dp2_tp2_c1_0.json').exists())


if __name__ == '__main__':
    unittest.main()
