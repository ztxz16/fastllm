"""混合长度诊断回归；正式CLI及HTTP真实运行，仅CUDA、原生协议和tokenizer使用替身。"""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import signal
import threading
import unittest
from unittest import mock

import diagnose_qwen3_dp_lengths as runner
import test_process_dp as fixtures


class DPLengthTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ProcessDPTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.rename_model('dp_routing_lengths')
        self.root = fixture.root
        fixture.fixture.tokenizer.apply_chat_template.side_effect = lambda messages, **_: [1] * (32 + 4*messages[0]['content'].count('苹果是水果。'))
        cache = self.root / 'CMakeCache.txt'
        cache.write_text('CUDA_ARCH:UNINITIALIZED=70\n')
        binary = str(fixture.fixture.fixture.binary)
        self.args = ['--model', str(fixture.model), '--generate', binary, '--worker', binary,
                     '--build-cache', str(cache), '--dp', '2', '--tp', '2', '--timeout', '10',
                     '--prompt-lengths', '64', '256', '512', '--output-lengths', '2', '4',
                     '--output', str(self.root / 'reports')]

    def invoke(self, *extra):
        with mock.patch.object(runner, 'capture', return_value='Python fixture; no GPU'), \
                mock.patch.object(runner, 'gpu_sample', return_value=[dict(uuid='GPU-fixture', index=0, **{'memory.used': 123.5, 'memory.total': 16000.0})]), \
                redirect_stdout(io.StringIO()):
            code = runner.main([*self.args, *extra])
        output = sorted((self.root / 'reports').iterdir())[-1]
        return code, output, json.loads((output / 'summary.json').read_text())

    def test_real_http_mixed_lengths_tp_and_reports(self):
        code, output, report = self.invoke()
        self.assertEqual((code, report['status'], report['complete']), (0, 'PASS', True), report)
        self.assertEqual([r['concurrency'] for r in report['cases']], [1, 2, 4])
        self.assertEqual(report['metadata']['repeat_orders'], [[1, 2, 4], [2, 4, 1], [4, 1, 2], [1, 2, 4], [2, 4, 1]])
        inputs = json.loads((output / 'inputs.json').read_text())
        for row in report['cases']:
            self.assertEqual(len(row['samples']), 5)
            self.assertEqual(row['requests'], 60)
            self.assertEqual(row['failed_elements'], 0)
            self.assertGreater(row['p95_first_content_ms'], 0)
            self.assertGreaterEqual(row['p95_queue_ms'], 0)
            self.assertEqual(len(row['by_input']), 12)
            for sample in row['samples']:
                self.assertEqual({r['prompt_tokens'] for r in sample['records']}, {64, 256, 512})
                self.assertEqual({r['generated_tokens'] for r in sample['records']}, {2, 4})
                self.assertEqual([r['payload'] for r in sample['records']], [item['payload'] for item in inputs])
                for record in sample['records']:
                    latency = 1000*(record['end']-record['start'])
                    self.assertGreaterEqual(record['client_wait_ms'], 0)
                    self.assertAlmostEqual(record['completion_ms'], record['client_wait_ms'] + latency)
                    self.assertLessEqual(record['completion_ms'], sample['elapsed_s']*1000)
        self.assertEqual(report['cases'][0]['speedup_vs_c1'], 1)
        for suffix in ('json', 'csv', 'md', 'xlsx'):
            self.assertTrue((output / ('summary.' + suffix)).is_file())
        self.assertEqual(report['metadata']['gpu_memory_observed'][0]['max_used_mib'], 123.5)
        self.assertIn('p95_completion_ms', (output / 'summary.csv').read_text().splitlines()[0])
        self.assertIn('p95_client_wait_ms', (output / 'inputs.csv').read_text().splitlines()[0])

        # 即使另外两档已经PASS，缺少C=DP也不能生成整体PASS报告。
        rows = [row for row in report['cases'] if row['concurrency'] != 2]
        self.assertEqual(runner.save(output, rows, report['metadata']), 'FAIL')
        self.assertFalse(json.loads((output / 'summary.json').read_text())['complete'])

    def test_real_http_pure_dp(self):
        code, _, report = self.invoke('--dp', '4', '--tp', '1')
        self.assertEqual(code, 0, report)
        self.assertEqual([r['concurrency'] for r in report['cases']], [1, 4, 8])

    def test_wrong_native_output_fails_with_raw_evidence(self):
        original = runner.MixedService.native
        def corrupt(service, record):
            value = original(service, record)
            if record['id'].startswith('r0-dp2_tp2_c2-'):
                value['token_ids'][0] += 1
            return value
        with mock.patch.object(runner.MixedService, 'native', corrupt):
            code, output, report = self.invoke()
        self.assertEqual((code, report['status'], report['complete']), (1, 'FAIL', False))
        self.assertEqual(len(report['cases'][0]['samples']), 1)
        self.assertEqual(report['cases'][1]['failed_elements'], 1)
        self.assertEqual(report['cases'][2]['samples'], [])
        self.assertTrue((output / 'raw/r0-dp2_tp2_c2.json').is_file())

    def test_completion_statistics_include_client_wait(self):
        samples = [dict(elapsed_s=4, records=[
            dict(start=11, end=11.5, first_content=None, generated_tokens=2, queue_ms=100, service_ms=300,
                 client_wait_ms=1000, completion_ms=1500),
            dict(start=13, end=14, first_content=13.2, generated_tokens=4, queue_ms=200, service_ms=700,
                 client_wait_ms=3000, completion_ms=4000)])]
        metrics = runner.stats(samples)
        self.assertEqual(metrics['median_request_ms'], 750)
        self.assertEqual(metrics['median_client_wait_ms'], 2000)
        self.assertEqual(metrics['median_completion_ms'], 2750)
        self.assertEqual(metrics['p95_completion_ms'], 3875)
        self.assertEqual(metrics['output_tokens_per_s'], 1.5)
        self.assertAlmostEqual(metrics['median_first_content_ms'], 200)

    def test_sampling_error_cannot_be_hidden_by_successful_requests(self):
        original = runner.MemorySampler.sample
        def failed(sampler):
            original(sampler)
            sampler.errors.append('sampling failure')
        with mock.patch.object(runner.MemorySampler, 'sample', failed):
            code, _, report = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn('sampling failure', report['cases'][0]['reason'])

    def test_interruption_retains_partial_report_and_reaps(self):
        original = runner.exchange
        previous = signal.getsignal(signal.SIGTERM)
        sent = threading.Event()
        def interrupt(*args, **kwargs):
            if args[2].startswith('r0-') and not sent.is_set():
                # 信号在主线程处理；实际HTTP客户端和正式服务仍执行自己的清理。
                import os
                sent.set()
                os.kill(os.getpid(), signal.SIGTERM)
            return original(*args, **kwargs)
        with mock.patch.object(runner, 'exchange', side_effect=interrupt):
            code, _, report = self.invoke()
        self.assertEqual((code, report['status']), (1, 'FAIL'))
        self.assertIn('interrupted', report['cases'][0]['reason'])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_capacity_invalid_before_model_start(self):
        with mock.patch.object(runner, 'make_normal_llm_model') as model, redirect_stderr(io.StringIO()):
            for extra in (['--prompt-lengths', '64', '2049'], ['--prompt-lengths', '64', '2048'],
                          ['--repeats', '4'], ['--timeout', 'nan']):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    runner.main([*self.args, *extra])
            model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
