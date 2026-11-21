"""分批验收编排的CPU检查：只运行替身命令，不加载模型、不编译或访问GPU。"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import attention_ep_acceptance as runner
import run_qwen_moe_full as full
import run_qwen_moe_layer as layer
from run_qwen_moe_batch_performance import compare_samples


class AcceptanceTests(unittest.TestCase):
    def test_partition_covers_original_synthetic_and_performance_matrices(self):
        """/** 展开独立批次，检查覆盖不丢失、无重复且每份脚本不超过50项。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(build=root / 'build', model=root / 'model', python=sys.executable,
                reference_python=sys.executable,
                requests=runner.ROOT / 'test/distributed/data/qwen_moe_requests.jsonl')
            jobs = runner.jobs_for(args, root)
            self.assertEqual(len(jobs), 29)
            self.assertEqual(len({job['name'] for job in jobs}), len(jobs))
            self.assertTrue(all(0 < job['expected_report_rows'] <= 50 for job in jobs))
            self.assertEqual(sum(job['expected_report_rows'] for job in jobs if job['name'].startswith('synthetic_')), 96)
            self.assertEqual(sum(job['expected_report_rows'] for job in jobs if job['group'] == 'performance'), 162)
            restored = [json.loads(line) for path in sorted(root.glob('hf_requests_part*.jsonl'))
                        for line in path.read_text().splitlines()]
            original = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
            self.assertEqual(restored, original)
            for job in jobs:
                if 'candidate' in job['name']:
                    self.assertIn('--batch-invariant', job['command'])

    def test_missing_report_cannot_pass_with_zero_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            job = dict(name='missing', directory=temp, expected_report_rows=2)
            self.assertEqual(runner.inspect_job(job, 0)['status'], 'FAIL')

    def test_prepared_script_quotes_paths_and_saves_exit_code(self):
        """/** 带空格和单引号的目录仍保存真实退出码，脚本不将命令文本当作插值执行。 */"""
        with tempfile.TemporaryDirectory(prefix="acceptance ' path ") as temp:
            root = Path(temp)
            args = SimpleNamespace(output=root, build=root / 'build', model=root / 'model',
                python=sys.executable, reference_python=sys.executable, groups=['validation'], only=None)

            def planned(options, output):
                return [dict(name='fixture', group='validation', command=[sys.executable, '-c', 'raise SystemExit(7)'],
                             directory=str(output / 'fixture'), state='PENDING', expected_report_rows=1)]

            printed = io.StringIO()
            with mock.patch.object(runner, 'jobs_for', side_effect=planned), \
                    mock.patch.object(runner, 'capture', return_value='CPU fixture'), \
                    mock.patch.object(runner.subprocess, 'check_output', return_value=b''), redirect_stdout(printed):
                runner.prepare(args)
            output = Path(printed.getvalue().strip())
            completed = subprocess.run(['bash', str(output / 'fixture/run.sh')], capture_output=True)
            self.assertEqual(completed.returncode, 7)
            self.assertEqual((output / 'fixture/exit.code').read_text().strip(), '7')

    def test_incomplete_and_failed_child_reports_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / 'reports/one/summary.json'
            report.parent.mkdir(parents=True)
            job = dict(name='partial', directory=temp, expected_report_rows=2)
            runner.write_json(report, dict(status='PASS', cases=[dict(case='a', status='PASS')]))
            self.assertEqual(runner.inspect_job(job, 0)['status'], 'FAIL')
            runner.write_json(report, dict(status='PASS', cases=[dict(case='a', status='PASS'), dict(case='b', status='FAIL')]))
            self.assertEqual(runner.inspect_job(job, 0)['status'], 'FAIL')

    def test_nonzero_exit_is_preserved_even_with_passing_report(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'reports/one/summary.json'
            path.parent.mkdir(parents=True)
            runner.write_json(path, dict(status='PASS', cases=[dict(case='a', status='PASS')]))
            job = dict(name='exit', directory=temp, expected_report_rows=1)
            self.assertEqual(runner.inspect_job(job, 7)['status'], 'FAIL')
            self.assertEqual(runner.inspect_job(job, 0)['status'], 'PASS')

    def test_malformed_report_and_missing_declared_cases_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'reports/one/summary.json'
            path.parent.mkdir(parents=True)
            job = dict(name='bad-report', directory=temp, expected_report_rows=1)
            path.write_text('{')
            self.assertEqual(runner.inspect_job(job, 0)['status'], 'FAIL')
            runner.write_json(path, dict(status='PASS', expected_case_count=2,
                                        cases=[dict(case='a', status='PASS')]))
            self.assertEqual(runner.inspect_job(job, 0)['status'], 'FAIL')

    def test_failed_batch_does_not_hide_later_results(self):
        """/** 真实子进程只执行shell替身，验证失败后继续、证据保存和非零总退出码。 */"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = []
            for name, code in (('failed', 7), ('passed', 0)):
                directory = root / name
                directory.mkdir()
                report = directory / 'reports/one/summary.json'
                report.parent.mkdir(parents=True)
                runner.write_json(report, dict(status='PASS', cases=[dict(case='a', status='PASS')]))
                script = directory / 'run.sh'
                script.write_text(f'#!/bin/sh\nexit {code}\n')
                jobs.append(dict(name=name, group='validation', directory=str(directory), expected_report_rows=1,
                                 script_sha256=runner.fingerprint(script), state='PENDING'))
            runner.write_json(root / 'manifest.json', dict(jobs=jobs, git='test', git_status='',
                requested_command='CPU fixture', required_files=[], frozen_files={}, blockers=runner.BLOCKERS,
                python=sys.executable, reference_python=sys.executable))
            with mock.patch.object(runner, 'capture', return_value='CPU fixture'), redirect_stdout(io.StringIO()):
                self.assertEqual(runner.execute(root), 1)
            report = json.loads((root / 'summary.json').read_text())
            self.assertEqual([row['status'] for row in report['cases']], ['FAIL', 'PASS'])
            self.assertFalse(report['metadata']['final_delivery_eligible'])
            self.assertEqual((root / 'failed/exit.code').read_text().strip(), '7')
            self.assertEqual((root / 'passed/exit.code').read_text().strip(), '0')
            self.assertEqual((root / 'exit.code').read_text().strip(), '1')
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((root / f'summary.{suffix}').is_file())

    def test_oversized_synthetic_is_rejected_before_setup(self):
        argv = ['runner', '--binary', '/unused', '--worker', '/unused', '--build-cache', '/unused']
        with mock.patch('sys.argv', argv), mock.patch.object(full, 'Suite') as suite, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                full.main()
            self.assertEqual(error.exception.code, 2)
            suite.assert_not_called()

    def test_oversized_layer_is_rejected_before_gpu(self):
        argv = ['runner', '--check', '/unused', '--worker', '/unused', '--build-cache', '/unused']
        with mock.patch('sys.argv', argv), mock.patch.object(layer, 'invoke') as invoke, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                layer.main()
            self.assertEqual(error.exception.code, 2)
            invoke.assert_not_called()

    def test_speedup_is_unavailable_for_different_output_history(self):
        sample = dict(requests=[dict(token_ids=[1])], prompt_length=1, output_tokens=2, concurrency=1,
                      cuda_graph=False, cache_state='off', warmup=1, iterations=1, repeats=5,
                      reused_tokens=[[0]] * 5, responses=[[dict(token_ids=[3, 4])]] * 5, samples_ms=[2.] * 5)
        candidate = dict(sample, responses=[[dict(token_ids=[3, 5])]] * 5, samples_ms=[1.] * 5)
        result = compare_samples(sample, candidate, 'PASS', 'PASS')
        self.assertTrue(result['measurements_valid'])
        self.assertFalse(result['comparison_eligible'])
        self.assertIsNone(result['speedup'])


if __name__ == '__main__':
    unittest.main()
