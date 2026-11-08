"""Simulated admission checks; no CUDA, sleep, network or process mutation."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / 'tools' / 'tp5_resource_check.py'
CODE = runpy.run_path(str(SCRIPT))
MAIN = CODE['main']


class AdmissionTests(unittest.TestCase):
    def invoke(self, argv, gpu_rows='', apps='', busy=0, failure=False):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if failure:
                return subprocess.CompletedProcess(command, 1, '', 'GPU driver query unavailable')
            text = apps if command[1].startswith('--query-compute-apps=') else gpu_rows
            return subprocess.CompletedProcess(command, 0, text, '')

        counters = iter([(1000, 500), (1100, 600 - busy)])
        out = io.StringIO()
        with patch.object(sys, 'argv', ['probe'] + argv), \
                patch.object(subprocess, 'run', side_effect=run), \
                patch('time.sleep'), \
                patch.dict(MAIN.__globals__, cpu_counters=lambda: next(counters)), \
                contextlib.redirect_stdout(out):
            result = MAIN()
        return result, json.loads(out.getvalue()), calls

    def test_unused_required_device_ignores_foreign_device(self):
        result, report, calls = self.invoke(
            ['--gpus', '0'], '0, GPU-a, 0, 0\n5, GPU-b, 14000, 100\n', 'GPU-b, 123\n')
        self.assertEqual(result, 0)
        self.assertEqual(report['status'], 'READY')
        self.assertEqual([x['index'] for x in report['gpus']], [0])
        self.assertTrue(all('--id=0' in x for x in calls))

    def test_foreign_process_blocks_an_idle_looking_device(self):
        result, report, _ = self.invoke(['--gpus', '0'], '0, GPU-a, 0, 0\n', 'GPU-a, 123\n')
        self.assertEqual(result, 3)
        self.assertIn('GPU_0_IN_USE', report['reasons'])

    def test_resident_memory_blocks_even_at_zero_utilization(self):
        result, report, _ = self.invoke(['--gpus', '0'], '0, GPU-a, 12, 0\n')
        self.assertEqual(result, 3)

    def test_utilization_blocks_even_at_zero_reported_memory(self):
        result, report, _ = self.invoke(['--gpus', '0'], '0, GPU-a, 0, 10\n')
        self.assertEqual(result, 3)

    def test_driver_query_failure_is_unknown_not_ready(self):
        result, report, _ = self.invoke(['--gpus', '0'], failure=True)
        self.assertEqual(result, 2)
        self.assertEqual(report['status'], 'UNKNOWN')

    def test_missing_device_is_unknown(self):
        result, report, _ = self.invoke(['--gpus', '0,1'], '0, GPU-a, 0, 0\n')
        self.assertEqual(result, 2)
        self.assertEqual(report['status'], 'UNKNOWN')

    def test_cpu_threshold_and_cpu_only_never_query_gpu(self):
        for busy, expected in [(50, 0), (51, 3)]:
            result, report, calls = self.invoke(['--cpu-only'], busy=busy)
            self.assertEqual(result, expected)
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
