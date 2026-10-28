"""Quota, isolation and bounded native-calibration policy; no model or GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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

    def test_invalid_candidate_requires_fallback(self):
        for invalid in (0, -1, float('inf'), float('nan')):
            result = {'samples_ms': {'1': 4, '2': 2, '4': invalid}}
            with self.subTest(invalid=invalid), patch.object(tuning, '_measure', return_value=result):
                chosen, detail = tuning.select_threads(5120, 2304, 16, 4)
            self.assertEqual(chosen, 4)
            self.assertEqual(detail['fallback'], 'ValueError')

    def test_worker_is_separate_and_has_hard_timeout(self):
        tuning._measure.cache_clear()
        value = {'samples_ms': {'1': 1, '2': .6, '4': .4}}
        done = subprocess.CompletedProcess([], 0, stdout='native initialization\n' + json.dumps(value), stderr='')
        with patch.object(tuning.subprocess, 'run', return_value=done) as run:
            self.assertEqual(tuning._measure(7168, 2048, 16, (1, 2, 4)), value)
        self.assertEqual(run.call_args.kwargs['timeout'], 10.0)
        self.assertIn('--worker', run.call_args.args[0])
        tuning._measure.cache_clear()


if __name__ == '__main__':
    unittest.main()
