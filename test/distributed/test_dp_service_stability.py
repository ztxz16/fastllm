"""正式CLI/HTTP稳定性验收器回归；仅原生进程、CUDA枚举和tokenizer使用替身。"""

from contextlib import redirect_stdout
import io
import json
import signal
import unittest
from unittest import mock

import run_qwen3_dp_service as runner
import test_process_dp as fixtures
from fastllm_pytools.process_dp import ProcessDPModel


class DPServiceStabilityTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ProcessDPTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.rename_model('dp_routing_service')
        self.root = fixture.root
        cache = self.root / 'CMakeCache.txt'
        cache.write_text('CUDA_ARCH:UNINITIALIZED=70\n')
        binary = str(fixture.fixture.fixture.binary)
        self.args = ['--model', str(fixture.model), '--generate', binary, '--worker', binary,
                     '--build-cache', str(cache), '--dp', '2', '--tp', '2', '--timeout', '10',
                     '--round-seconds', '0.02', '--output', str(self.root / 'reports')]

    def invoke(self, *extra):
        with mock.patch.object(runner, 'capture', return_value='Python protocol fixture; no GPU'), redirect_stdout(io.StringIO()):
            code = runner.main([*self.args, *extra])
        output = sorted((self.root / 'reports').iterdir())[-1]
        return code, output, json.loads((output / 'summary.json').read_text())

    def test_real_cli_http_sse_lifecycle_reports_and_reaping(self):
        code, output, report = self.invoke()
        self.assertEqual((code, report['status'], report['complete']), (0, 'PASS', True), report)
        self.assertEqual(len(report['cases']), 12)
        for row in report['cases']:
            if 'requests' in row:
                self.assertGreaterEqual(row['requests'], 4)
                self.assertGreater(row['http_requests'], 0)
                self.assertGreater(row['sse_requests'], 0)
                self.assertEqual(row['failed_elements'], 0)
        for suffix in ('json', 'md', 'csv', 'xlsx'):
            self.assertTrue((output / ('summary.' + suffix)).exists())
        self.assertGreater(report['cases'][-1]['reaped_processes'], 1)
        self.assertTrue((output / 'raw/disconnect-sse-recovery.json').exists())

    def test_corrupted_http_output_fails_and_retains_partial_report(self):
        original = runner.exchange

        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            if result['id'].startswith('round-0') and not result['payload']['stream'] and result['status'] == 200:
                result['body']['choices'][0]['message']['content'] = 'WRONG'
            return result
        with mock.patch.object(runner, 'exchange', side_effect=corrupt):
            code, output, report = self.invoke('--routing', 'round_robin', '--tp', '1')
        self.assertEqual((code, report['status'], report['complete']), (1, 'FAIL', False), report)
        self.assertEqual(report['cases'][-1]['case'], 'mixed_http_sse_round_01', report)
        self.assertIn('HTTP response differs', report['cases'][-1]['reason'])
        self.assertTrue((output / 'raw/round-0.json').exists())

    def test_pure_dp_round_robin_lifecycle(self):
        code, _, report = self.invoke('--dp', '4', '--tp', '1', '--routing', 'round_robin')
        self.assertEqual((code, report['status'], report['complete']), (0, 'PASS', True), report)

    def test_truncated_sse_cannot_pass_as_completed(self):
        original = runner.exchange

        def truncate(*args, **kwargs):
            result = original(*args, **kwargs)
            if result['id'].startswith('round-0') and result['payload']['stream']:
                result['body'] = result['body'][:-1]
            return result
        with mock.patch.object(runner, 'exchange', side_effect=truncate):
            code, _, report = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn('missing/duplicate SSE DONE', report['cases'][-1]['reason'], report)

    def test_wrong_native_tokens_report_errors_and_cleanup(self):
        original = runner.ServiceRun.native

        def corrupt(service, record):
            result = original(service, record)
            if record['id'].startswith('round-0'):
                result['token_ids'][0] += 1
            return result
        with mock.patch.object(runner.ServiceRun, 'native', corrupt):
            code, _, report = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(report['cases'][-1]['failed_elements'], 1, report)

    def test_runner_sigterm_preserves_report_and_reaps_service(self):
        original = runner.ServiceRun.soak
        previous = signal.getsignal(signal.SIGTERM)

        def interrupt(service, index, seconds):
            if index == 0:
                signal.raise_signal(signal.SIGTERM)
            return original(service, index, seconds)
        with mock.patch.object(runner.ServiceRun, 'soak', interrupt):
            code, _, report = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn('SIGTERM', report['cases'][-1]['reason'], report)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_http_disconnect_between_prepare_and_dp_dispatch(self):
        original = ProcessDPModel.generate

        def delayed(model, request, *, cancel_event=None, on_token=None):
            if request['max_new_tokens'] == 1024 and on_token is None:
                self.assertTrue(cancel_event.wait(5), 'disconnect was not observed')
            return original(model, request, cancel_event=cancel_event, on_token=on_token)
        with mock.patch.object(ProcessDPModel, 'generate', delayed):
            code, _, report = self.invoke()
        self.assertEqual(code, 0, report)
        row = next(c for c in report['cases'] if c['case'] == 'http_disconnect_and_recovery')
        self.assertFalse(row['native_cancelled'])


if __name__ == '__main__':
    unittest.main()
