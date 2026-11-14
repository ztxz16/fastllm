"""定向验收的CPU流程检查；替换原生会话，不编译、不启动GPU。"""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_qwen_moe_full as runner


class FakeSession:
    def __init__(self, fail_start=False, captures=1, bad_logits=False):
        self.fail_start = fail_start
        self.bad_logits = bad_logits
        self.ready = {'workers': []}
        self.shutdown = {'workers_reaped': True}
        self._states = [dict(captures=captures, replays=1) for _ in range(4)]

    def __enter__(self):
        if self.fail_start:
            raise RuntimeError('session disconnected during startup')
        return self

    def __exit__(self, *args):
        pass

    def generate(self, request, teacher_forced_tokens=None):
        return dict(request=request, token_ids=[1], teacher_forced_tokens=teacher_forced_tokens,
                    steps=[dict(logits=[0.5 if self.bad_logits else 0.0, 1.0], token_id=1,
                                execution='decode_cuda_graph')])


def small_fixture(model, **options):
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'vocab_size': 2}))
    return {}, {}


class ReproduceTests(unittest.TestCase):
    def run_focused(self, sessions):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'CMakeCache.txt'
            cache.write_text('CPU test; no compilation')
            args = SimpleNamespace(output=root, build_cache=cache, timeout=1,
                                   binary=Path('/unused/generate'), worker=Path('/unused/worker'),
                                   synthetic_case='f16_bias_ep4_graph1_prefix0', repeat=len(sessions))
            with mock.patch.object(runner, 'capture', return_value='CPU test'), \
                 mock.patch.object(runner, 'fixture', side_effect=small_fixture), \
                 mock.patch.object(runner, 'full_logits', return_value=[0.0, 1.0]), \
                 mock.patch.object(runner, 'QwenMoeProcessSession', side_effect=sessions) as native, \
                 redirect_stdout(io.StringIO()):
                suite = runner.Suite(args)
                suite.synthetic()
            report = json.loads((suite.output / 'summary.json').read_text())
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((suite.output / f'summary.{suffix}').is_file())
            self.assertEqual(len(native.call_args_list), len(sessions))
            self.assertEqual(len({call.args[4] for call in native.call_args_list}), len(sessions))
            for call in native.call_args_list:
                self.assertEqual(call.args[3], '0,1,2,3')
                self.assertTrue(call.kwargs['cuda_graph'])
                self.assertFalse(call.kwargs['prefix_cache'])
            return report

    def test_later_success_does_not_hide_startup_failure(self):
        report = self.run_focused([FakeSession(True), FakeSession(), FakeSession()])
        self.assertEqual(report['status'], 'FAIL')
        self.assertTrue(report['complete'])
        self.assertEqual([row['status'] for row in report['cases']], ['FAIL', 'PASS', 'PASS'])
        self.assertIn('startup', report['cases'][0]['reason'])
        self.assertEqual(len({row['case'] for row in report['cases']}), 3)
        self.assertEqual(report['metadata']['scope'], 'focused_synthetic')

    def test_complete_repeated_success_is_labeled_as_focused(self):
        report = self.run_focused([FakeSession(), FakeSession()])
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['expected_case_count'], 2)
        self.assertEqual(report['metadata']['selected_case'], 'f16_bias_ep4_graph1_prefix0')
        self.assertTrue(all(row['failed_elements'] == 0 for row in report['cases']))

    def test_focus_retains_graph_path_checks(self):
        report = self.run_focused([FakeSession(captures=0)])
        self.assertEqual(report['status'], 'FAIL')
        self.assertIn('Graph capture/replay evidence missing', report['cases'][0]['reason'])

    def test_same_greedy_token_still_requires_correct_logits(self):
        report = self.run_focused([FakeSession(bad_logits=True)])
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['cases'][0]['failed_elements'], 1)
        self.assertEqual(report['cases'][0]['max_abs_error'], 0.5)

    def test_invalid_selection_rejected_before_launch(self):
        base = ['run_qwen_moe_full.py', '--binary', '/unused', '--worker', '/unused', '--build-cache', '/unused']
        for extra in (['--synthetic-case', 'typo'], ['--repeat', '0'], ['--repeat', '2'],
                      ['--synthetic-case', 'f16_bias_ep4_graph1_prefix0', '--model', '/unused'],
                      ['--synthetic-case', 'f16_bias_ep4_graph1_prefix0', '--performance']):
            with self.subTest(extra=extra), mock.patch.object(runner.sys, 'argv', base + extra), \
                 mock.patch.object(runner, 'Suite') as suite, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(error.exception.code, 2)
                suite.assert_not_called()


if __name__ == '__main__':
    unittest.main()
