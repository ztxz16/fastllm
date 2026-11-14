"""Quota, isolation and bounded native-calibration policy; no model or GPU."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from fastllm_pytools import cpu_tuning as tuning


class CpuTuningTest(unittest.TestCase):
    def layout(self, directory, v2=True):
        root = Path(directory)
        proc, mount = root / 'proc', root / 'cgroup'
        (proc / 'self').mkdir(parents=True)
        child = mount / 'parent/worker'
        child.mkdir(parents=True)
        (proc / 'self/cgroup').write_text('0::/parent/worker\n' if v2 else '2:cpu,cpuacct:/parent/worker\n')
        (proc / 'self/mountinfo').write_text('1 0 0:1 / %s rw - %s cgroup %s\n' %
                                         (mount, 'cgroup2' if v2 else 'cgroup', 'rw' if v2 else 'rw,cpu,cpuacct'))
        return proc, child

    def test_nested_v2_quota_and_unlimited_child(self):
        with tempfile.TemporaryDirectory() as d:
            proc, child = self.layout(d)
            (child / 'cpu.max').write_text('max 100000')
            (child.parent / 'cpu.max').write_text('1000000 100000')
            (child.parent.parent / 'cpu.max').write_text('2000000 100000')
            self.assertEqual(tuning.cpu_quota(proc), 10)
            (child / 'cpu.max').write_text('50000 100000')
            self.assertEqual(tuning.cpu_quota(proc), .5)

    def test_v1_quota(self):
        with tempfile.TemporaryDirectory() as d:
            proc, child = self.layout(d, False)
            (child / 'cpu.cfs_quota_us').write_text('800000')
            (child / 'cpu.cfs_period_us').write_text('100000')
            self.assertEqual(tuning.cpu_quota(proc), 8)

    def test_quota_leaves_controller_headroom(self):
        self.assertEqual(tuning.thread_limit(40, 10), 8)
        self.assertEqual(tuning.thread_limit(40, 8), 6)
        self.assertEqual(tuning.thread_limit(100, 100), 96)
        self.assertEqual(tuning.thread_limit(100, 40), 36)
        self.assertEqual(tuning.thread_limit(4, 40), 3)
        self.assertEqual(tuning.thread_limit(40, .5), 1)
        self.assertEqual(tuning.candidates(9), (1, 2, 4, 6, 8, 9))
        self.assertEqual(tuning.candidates(tuning.thread_limit(40, 10)), (1, 2, 4, 6, 8))

    def test_fewer_workers_win_a_near_tie(self):
        result = {'samples_ms': {'1': 4, '2': 2, '4': 1.4, '6': 1.1, '8': .91, '9': .9}}
        with patch.object(tuning, '_measure', return_value=result):
            chosen, detail = tuning.select_threads(7168, 2048, 16, 9)
        self.assertEqual(chosen, 8)
        self.assertEqual(detail['selected'], 8)

    def test_exl3_kind_uses_its_own_calibration(self):
        with patch.object(tuning,'_measure',return_value={'samples_ms':{'1':3,'2':2,'4':1}}) as probe:
            chosen,detail=tuning.select_threads(5120,2304,-6,4)
        probe.assert_called_once_with(5120,2304,-6,(1,2,4))
        self.assertEqual(chosen,4)
        self.assertEqual(detail['method'],'quantized-gemv')

    def test_failed_calibration_preserves_original_limit(self):
        for limit in (1, 6, 9, 28, 128):
            for error in (subprocess.TimeoutExpired('probe', 10), OSError('library unavailable'),
                          subprocess.CalledProcessError(1, 'probe'), ValueError('no samples')):
                with self.subTest(limit=limit, error=type(error).__name__), \
                        patch.object(tuning, '_measure', side_effect=error):
                    chosen, detail = tuning.select_threads(7168, 2048, 16, limit)
                self.assertEqual(chosen, limit)
                self.assertEqual(detail['fallback'], type(error).__name__)

    def test_partial_samples_do_not_choose_only_completed_small_counts(self):
        result = {'samples_ms': {'1': 4, '2': 2, '4': 1}}
        with patch.object(tuning, '_measure', return_value=result):
            chosen, detail = tuning.select_threads(5120, 2304, 16, 28)
        self.assertEqual(chosen, 28)
        self.assertEqual(detail['fallback'], 'ValueError')

    def test_budget_limited_numa_sweep_keeps_measured_large_counts(self):
        result = {'samples_ms': {'124': 2, '64': 1, '32': 1.03},
                  'measurement_order': 'descending', 'budget_exhausted': True,
                  'unmeasured_threads': [2, 4, 8, 16]}
        with patch.object(tuning, 'numa_node_count', return_value=2), \
                patch.object(tuning, '_measure', return_value=result):
            chosen, detail = tuning.select_threads(5120, 2304, 17, 124, numa=True, rows=48, topk=6)
        self.assertEqual(chosen, 32)
        self.assertNotIn('fallback', detail)
        self.assertTrue(detail['budget_exhausted'])

    def test_budget_limited_sweep_cannot_skip_an_unmeasured_large_count(self):
        result = {'samples_ms': {'124': 2, '32': 1},
                  'measurement_order': 'descending', 'budget_exhausted': True,
                  'unmeasured_threads': [2, 4, 8, 16, 64]}
        with patch.object(tuning, 'numa_node_count', return_value=2), \
                patch.object(tuning, '_measure', return_value=result):
            chosen, detail = tuning.select_threads(5120, 2304, 17, 124, numa=True, rows=48, topk=6)
        self.assertEqual(chosen, 124)
        self.assertEqual(detail['fallback'], 'ValueError')

    def test_worker_failures_preserve_actionable_diagnostics(self):
        error = subprocess.CalledProcessError(-11, ['worker'], output='binding workers',
                                             stderr='native shard setup failed')
        with patch.object(tuning, '_measure', side_effect=error):
            chosen, detail = tuning.select_threads(5120, 2304, 17, 124)
        self.assertEqual(chosen, 124)
        self.assertEqual(detail['returncode'], -11)
        self.assertIn('native shard setup failed', detail['stderr_tail'])
        self.assertEqual(detail['stdout_tail'], 'binding workers')
        self.assertIn('error', detail)

    def test_timeout_diagnostics_accept_bytes_and_bound_log_size(self):
        error = subprocess.TimeoutExpired('worker', 10, output=b'x' * 4096,
                                          stderr=b'last native diagnostic')
        with patch.object(tuning, '_measure', side_effect=error):
            _, detail = tuning.select_threads(5120, 2304, 17, 124)
        self.assertEqual(len(detail['stdout_tail']), 2048)
        self.assertEqual(detail['stderr_tail'], 'last native diagnostic')

    def test_numa_batch_is_not_truncated_to_a_fixed_row_limit(self):
        for rows in (1, 6, 24, 48, 257):
            with self.subTest(rows=rows), patch.object(tuning, 'numa_node_count', return_value=2), \
                    patch.object(tuning, '_measure', return_value={'samples_ms': {'2': 1, '4': .5}}) as probe:
                _, detail = tuning.select_threads(5120, 2304, -6, 4, numa=True, rows=rows, topk=6)
            probe.assert_called_once_with(5120, 2304, -6, (2, 4), rows, 6)
            self.assertEqual(detail['rows'], rows)

    def test_invalid_candidate_requires_fallback(self):
        for invalid in (0, -1, float('inf'), float('nan')):
            result = {'samples_ms': {'1': 4, '2': 2, '4': invalid}}
            with self.subTest(invalid=invalid), patch.object(tuning, '_measure', return_value=result):
                chosen, detail = tuning.select_threads(5120, 2304, 16, 4)
            self.assertEqual(chosen, 4)
            self.assertEqual(detail['fallback'], 'ValueError')

    def test_numa_candidates_are_balanced_and_use_verifier_workload(self):
        samples = {'samples_ms': {'2': 8, '4': 4, '8': 2, '16': 1.5, '32': 1.3, '64': 1, '124': 1.2}}
        with patch.object(tuning, 'numa_node_count', return_value=2), \
                patch.object(tuning, '_measure', return_value=samples) as measure:
            chosen, detail = tuning.select_threads(5120, 2304, 17, 124, numa=True, rows=6, topk=6)
        self.assertEqual(chosen, 64)
        self.assertEqual(detail['method'], 'numa-moe')
        measure.assert_called_once_with(5120, 2304, 17, (2, 4, 8, 16, 32, 64, 124), 6, 6)

    def test_numa_layout_respects_affinity_and_explicit_node_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for node, cpus in enumerate(('0-3', '4-7')):
                path = root / ('node%d' % node)
                path.mkdir()
                (path / 'cpulist').write_text(cpus)
            with patch.dict(tuning.os.environ, {}, clear=True), \
                    patch.object(tuning.os, 'sched_getaffinity', create=True, return_value=set(range(8))):
                self.assertEqual(tuning.numa_node_count(root), 2)
            with patch.dict(tuning.os.environ, {'FT_NUMAS': '1'}, clear=True), \
                    patch.object(tuning.os, 'sched_getaffinity', create=True, return_value=set(range(4))):
                self.assertEqual(tuning.numa_node_count(root), 1)
            with patch.dict(tuning.os.environ, {}, clear=True), \
                    patch.object(tuning.os, 'sched_getaffinity', create=True, return_value=set(range(4))):
                self.assertEqual(tuning.numa_node_count(root), 1)

    def test_numa_nonzero_and_sparse_nodes_with_partial_cpu_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for node, cpus in ((1, '4-7'), (3, '12-15')):
                path = root / ('node%d' % node)
                path.mkdir()
                (path / 'cpulist').write_text(cpus)
            for mask, requested, expected in (({6}, '1', 1), ({6, 14}, '0', 2),
                                               ({6, 14}, '1', 1), ({6}, '8', 1),
                                               ({0}, '0', 0)):
                with self.subTest(mask=mask, requested=requested), \
                        patch.dict(tuning.os.environ, {'FT_NUMAS': requested}, clear=True), \
                        patch.object(tuning.os, 'sched_getaffinity', create=True, return_value=mask):
                    self.assertEqual(tuning.numa_node_count(root), expected)

    def test_unavailable_numa_layout_retains_linear_fallback(self):
        with patch.object(tuning, 'numa_node_count', return_value=0), \
                patch.object(tuning, '_measure', return_value={'samples_ms': {'1': 2, '2': 1}}) as measure:
            chosen, detail = tuning.select_threads(5120, 2304, 16, 2, numa=True, rows=6, topk=6)
        self.assertEqual(chosen, 2)
        self.assertEqual(detail['method'], 'quantized-gemv')
        measure.assert_called_once_with(5120, 2304, 16, (1, 2))

    def test_standard_numa_dispatch_preserves_gate_down_pair(self):
        with patch.object(tuning, 'numa_node_count', return_value=1), \
                patch.object(tuning, '_measure', return_value={'samples_ms': {'1': 2, '2': 1}}) as measure:
            chosen, detail = tuning.select_threads(2560, 640, 22, 2, numa=True, rows=4, topk=10, down_type=42)
        self.assertEqual(chosen, 2)
        measure.assert_called_once_with(2560, 640, 22, (1, 2), 4, 10, 42)

    def test_worker_is_separate_and_has_hard_timeout(self):
        tuning._measure.cache_clear()
        value = {'samples_ms': {'1': 1, '2': .6, '4': .4}}
        done = subprocess.CompletedProcess([], 0, stdout='native initialization\n' + json.dumps(value), stderr='')
        with patch.object(tuning.subprocess, 'run', return_value=done) as run:
            self.assertEqual(tuning._measure(7168, 2048, 16, (1, 2, 4)), value)
        self.assertEqual(run.call_args.kwargs['timeout'], 10.0)
        self.assertIn('--worker', run.call_args.args[0])
        tuning._measure.cache_clear()

    def test_linear_worker_does_not_require_process_libc_symbols(self):
        value = {'samples_ms': {'1': 1, '2': .6}}
        native = Mock()
        native.benchmark_cpu_linear_threads.return_value = json.dumps(value).encode()
        library = Path('/native/libfastllm_tools.so')
        def load(path):
            if path is None:
                raise AttributeError('process does not export libc symbols')
            return native
        output = io.StringIO()
        with patch('ctypes.CDLL', side_effect=load) as cdll, \
                patch.object(Path, 'glob', return_value=[library]), redirect_stdout(output):
            tuning._worker(json.dumps([5120, 2304, 16, [1, 2], 1000, 0, 1]))
        self.assertEqual(json.loads(output.getvalue())['samples_ms'], value['samples_ms'])
        cdll.assert_called_once_with(str(library))

    def test_worker_propagates_native_error_instead_of_empty_samples(self):
        native = Mock()
        native.benchmark_cpu_moe_threads.return_value = b'{"error":"unsupported MoE calibration shard shape"}'
        with patch('ctypes.CDLL', return_value=native), \
                patch.object(Path, 'glob', return_value=[Path('/native/libfastllm_tools.so')]), \
                patch.dict(tuning.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'unsupported MoE calibration shard shape'):
                tuning._worker(json.dumps([5120, 2304, 17, [2, 4], 1000, 48, 6]))


if __name__ == '__main__':
    unittest.main()
