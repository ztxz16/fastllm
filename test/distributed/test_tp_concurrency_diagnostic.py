"""并发交替诊断与观察进程的纯Python回归；GPU查询和原生执行使用替身，不编译。"""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import benchmark_observer as monitor
import diagnose_qwen3_tp_concurrency as diag
import test_benchmark_qwen3_dp as benchmark_tests


class ConcurrencyDiagnosticTests(unittest.TestCase):
    def setUp(self):
        fixture = benchmark_tests.BenchmarkDPTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        request = dict(token_ids=[1, 2], max_new_tokens=2, temperature=0.0, top_k=0, top_p=1.0, seed=0, eos_token_ids=[])
        self.request_file = self.root / 'request.json'
        self.request_file.write_text(json.dumps(request))
        self.flags = fixture.base[:8] + ['--request-file', str(self.request_file), '--requests', '4',
                                        '--no-telemetry', '--output', str(self.root / 'reports')]

    def invoke(self, *flags):
        with mock.patch.object(diag.bench, 'capture', return_value='Python fixture; no GPU'), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = diag.main(self.flags + list(flags))
        output = next((self.root / 'reports').iterdir())
        return status, output, json.loads((output / 'summary.json').read_text())

    def test_interleaved_formal_factory_reuses_model_and_saves_full_evidence(self):
        original = diag.bench.run_wave
        models = []

        def record(model, *args, **kwargs):
            models.append(id(model))
            return original(model, *args, **kwargs)

        with mock.patch.object(diag.bench, 'run_wave', side_effect=record):
            status, output, report = self.invoke()
        self.assertEqual((status, report['status']), (0, 'PASS'), report)
        self.assertTrue(report['complete'])
        self.assertEqual((len(models), len(set(models))), (23, 1))
        self.assertEqual(report['metadata']['completed_waves'], 20)
        self.assertEqual(report['metadata']['repeats'], 10)
        self.assertFalse(report['metadata']['telemetry'])
        self.assertEqual(report['metadata']['request'], json.loads(self.request_file.read_text()))
        rows = report['cases']
        for row in rows:
            self.assertEqual((len(row['samples']), row['failed_elements']), (10, 0))
            self.assertEqual(row['reference_cache_state'], 'warm')
        self.assertEqual(rows[1]['throughput_ratio_vs_c1'], rows[0]['median_batch_ms'] / rows[1]['median_batch_ms'])
        timeline = json.loads((output / 'timeline.json').read_text())
        self.assertEqual([(r['block'], r['position'], r['concurrency']) for r in timeline], diag.schedule(5))
        self.assertTrue(all(a['end'] <= b['start'] for a, b in zip(timeline, timeline[1:])))
        self.assertEqual(len(report['metadata']['within_block_c4_over_c1_ratios']), 5)
        for extension in ('json', 'md', 'csv', 'xlsx'):
            self.assertTrue((output / f'summary.{extension}').exists())
        with zipfile.ZipFile(output / 'observations.xlsx') as workbook:
            self.assertIn('Timeline', workbook.read('xl/workbook.xml').decode())

    def test_failed_timed_output_retains_partial_samples_and_raw(self):
        original = diag.bench.run_wave
        count = 0

        def corrupt(*args, **kwargs):
            nonlocal count
            count += 1
            wave = original(*args, **kwargs)
            if count == 5:  # prime、两次warmup、C1成功后破坏第一轮C4。
                wave['records'][0]['report']['token_ids'][0] += 1
            return wave

        with mock.patch.object(diag.bench, 'run_wave', side_effect=corrupt):
            status, output, report = self.invoke()
        self.assertEqual((status, report['status'], report['metadata']['completed_waves']), (1, 'FAIL', 1))
        self.assertFalse(report['complete'])
        self.assertEqual([len(row['samples']) for row in report['cases']], [1, 0])
        self.assertEqual(report['cases'][1]['failed_elements'], 1)
        self.assertTrue((output / 'raw/b0_p1_c4.json').exists())

    def test_common_control_uses_eager_disabled_cache(self):
        status, output, report = self.invoke('--profile', 'common', '--tp', '1')
        self.assertEqual(status, 0, report)
        self.assertFalse((output / 'raw/prime.json').exists())
        self.assertTrue(all(row['execution'] == 'eager' and row['reference_cache_state'] == 'disabled' for row in report['cases']))

    def test_bad_configuration_is_rejected(self):
        for flags in [('--blocks', '4'), ('--requests', '5'), ('--warmup', '0'), ('--timeout', 'nan'),
                      ('--telemetry-interval', '0'), ('--telemetry-interval', 'nan')]:
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                diag.parse_args(self.flags + list(flags))

    def test_gpu_summary_excludes_boundary_and_unselected_devices(self):
        frames = [dict(start=0, end=1, gpu=[dict(uuid='GPU-a', **{'clocks.current.sm': 999.})]),
                  dict(start=2, end=3, gpu=[dict(uuid='GPU-a', **{'clocks.current.sm': 100.}),
                                           dict(uuid='GPU-b', **{'clocks.current.sm': 200.})])]
        rows = diag.summarize_gpu(frames, [dict(start=.5, end=4, concurrency=1)], 1, {'a'})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['samples'], 1)
        self.assertEqual(rows[0]['clocks.current.sm']['median'], 100.)

    def test_observer_startup_failure_returns_report_and_closes_model(self):
        with mock.patch.object(diag, 'observer', side_effect=RuntimeError('observer startup failed')):
            status, output, report = self.invoke()
        self.assertEqual(status, 1)
        self.assertIn('observer startup failed', report['cases'][-1]['reason'])
        self.assertTrue((output / 'summary.xlsx').exists())
        self.assertTrue(list((output / 'models').rglob('shutdown.json')))

    def test_observer_ending_before_last_measurement_cannot_pass(self):
        self.flags.remove('--no-telemetry')

        @contextmanager
        def ended_early(output, *args, **kwargs):
            yield
            (output / 'observer.status.json').write_text(json.dumps(dict(samples=1, errors=[], end_monotonic=0)))

        with mock.patch.object(diag, 'observer', side_effect=ended_early):
            status, _, report = self.invoke()
        self.assertEqual((status, report['metadata']['completed_waves']), (1, 20))
        self.assertTrue(any('observer failed' in row['reason'] for row in report['cases']))


class ObserverTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_proc_tree_excludes_sampler_and_uses_process_identity(self):
        for pid, parent in ((1, 0), (2, 1), (3, 1), (4, 3), (9, 0)):
            directory = self.root / str(pid)
            directory.mkdir()
            fields = ['0'] * 20
            for index, value in {0: 'S', 1: str(parent), 11: '20', 12: '10', 17: '4', 19: '100'}.items():
                fields[index] = value
            (directory / 'stat').write_text(f'{pid} (worker (rank)) ' + ' '.join(fields))
            (directory / 'status').write_text('Cpus_allowed_list:\t20-39\n')
        rows = monitor.process_tree(1, 3, self.root)
        self.assertEqual([row['pid'] for row in rows], [1, 2])
        self.assertEqual(rows[1]['name'], 'worker (rank)')
        self.assertEqual(rows[1]['cpu_seconds'], 30 / os.sysconf('SC_CLK_TCK'))
        self.assertEqual(rows[1]['allowed_cpus'], '20-39')

    def test_gpu_csv_preserves_identity_discrete_index_and_unavailable_values(self):
        result = subprocess.CompletedProcess([], 0, 'GPU-abc, 3, P0, 1200, 877, 50, 20, 60, N/A, 300\n', '')
        with mock.patch.object(monitor.subprocess, 'run', return_value=result) as run:
            row = monitor.gpu_sample(monitor.GPU_FIELDS)[0]
        self.assertEqual((row['uuid'], row['index'], row['power.draw']), ('GPU-abc', 3, 'N/A'))
        self.assertIs(type(row['index']), int)
        self.assertEqual(run.call_args.kwargs['timeout'], 3)

    def test_failed_query_and_malformed_csv_are_errors(self):
        for result in (subprocess.CompletedProcess([], 1, '', 'GPU unavailable'),
                       subprocess.CompletedProcess([], 0, 'GPU-a, 0\n', '')):
            with mock.patch.object(monitor.subprocess, 'run', return_value=result), self.assertRaises(RuntimeError):
                monitor.gpu_sample(monitor.GPU_FIELDS)

    def test_sampler_records_optional_field_fallback_and_later_failure(self):
        process = dict(pid=1, ppid=0, start_ticks=100, cpu_seconds=1.)
        with mock.patch.object(monitor.signal, 'signal'), mock.patch.object(monitor.os, 'getppid', return_value=1), \
                mock.patch.object(monitor, 'process_tree', side_effect=[[dict(process)], [dict(process, cpu_seconds=2.)]]), \
                mock.patch.object(monitor, 'cgroup_cpu', return_value={'unavailable': 'fixture'}), \
                mock.patch.object(monitor, 'gpu_sample', side_effect=[RuntimeError('field unsupported'), [{'uuid': 'GPU-a'}],
                                                                    [{'uuid': 'GPU-a'}], RuntimeError('GPU lost')]):
            status = monitor.observe(self.root, 1, 0)
        result = json.loads((self.root / 'observer.status.json').read_text())
        self.assertEqual(status, 1)
        self.assertTrue(result['warnings'])
        self.assertIn('GPU lost', result['errors'][0])
        frames = [json.loads(line) for line in (self.root / 'telemetry.jsonl').read_text().splitlines()]
        self.assertEqual(len(frames), 3)
        self.assertIsNone(frames[0]['processes'][0]['cpu_interval_start'])
        self.assertEqual(frames[1]['processes'][0]['cpu_interval_start'], frames[0]['cpu_time'])
        self.assertEqual(frames[1]['processes'][0]['cpu_interval_end'], frames[1]['cpu_time'])

    def test_observer_is_reaped_when_measured_work_raises(self):
        (self.root / 'observer.ready.json').write_text('{"error": null}')
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch.object(monitor.subprocess, 'Popen', return_value=process), self.assertRaisesRegex(RuntimeError, 'work failed'):
            with monitor.observer(self.root):
                raise RuntimeError('work failed')
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)

    def test_real_observer_process_with_fake_nvidia_smi_leaves_no_child(self):
        executable = self.root / 'nvidia-smi'
        executable.write_text(f'#!{sys.executable}\nimport sys\nfields=sys.argv[1].split("=",1)[1].split(",")\n'
                              'print(",".join("GPU-fixture" if f=="uuid" else "P0" if f=="pstate" else "0" for f in fields))\n')
        executable.chmod(0o755)
        with mock.patch.dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ.get('PATH', '')):
            with monitor.observer(self.root, interval=.5) as pid:
                pass
        self.assertFalse(Path(f'/proc/{pid}').exists())
        status = json.loads((self.root / 'observer.status.json').read_text())
        self.assertEqual(status['errors'], [])
        self.assertGreaterEqual(status['samples'], 1)
        frame = json.loads((self.root / 'telemetry.jsonl').read_text().splitlines()[0])
        self.assertTrue(any(row['pid'] == os.getpid() for row in frame['processes']))


if __name__ == '__main__':
    unittest.main()
