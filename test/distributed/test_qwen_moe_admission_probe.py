"""固定准入诊断的纯Python回归；模拟原生协议，不编译或加载CUDA。"""

from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import probe_qwen_moe_admission as probe


class AdmissionProbeTests(unittest.TestCase):
    def run_probe(self, fault=None, waves=None):
        """/** 用协议替身验证固定轮次、真实输入回馈、路径校验及完整/失败四格式报告。 */"""
        sessions = []

        class Transport:
            def __init__(self, binary, worker, model, devices, output, **options):
                self.options, self.output = options, output
                self.graph = options['cuda_graph']
                self.states = [dict(request_active=False, request_tokens=0, request_capacity=0,
                    request_kv_bytes=0, captures=0, replays=0, prefix_cache=dict(last_reused_tokens=0)) for _ in range(4)]
                self.ready = dict(states=copy.deepcopy(self.states))
                self.history, self.generation = [], [0]*4
                self.batch = 0
                sessions.append(self)

            def __enter__(self):
                self.output.mkdir(parents=True)
                return self

            def __exit__(self, exc_type, *unused):
                self.shutdown = dict(shutdown_complete=exc_type is None)

            def exchange(self, op, *, steps, trace):
                assert op == 'step'
                actions = [s['action'] for s in steps]
                if 1 in actions and not any(s['request_active'] for s in self.states):
                    self.batch += 1
                    self.wave = actions.count(1)
                if self.batch == 3 and fault in ('worker', 'interrupt'):
                    raise KeyboardInterrupt() if fault == 'interrupt' else RuntimeError('injected worker error')
                self.history.append((copy.deepcopy(steps), trace))
                graph = self.graph and 1 not in actions and 2 in actions
                tokens, logits = [], []
                for rank, (step, state) in enumerate(zip(steps, self.states)):
                    action = step['action']
                    if action == 1:
                        assert not state['request_active']
                        self.generation[rank] = 0
                        state.update(request_active=True, request_tokens=len(step['token_ids']),
                                     request_capacity=step['capacity'], request_kv_bytes=20)
                    elif action == 2:
                        assert state['request_active']
                        expected = 2 if fault == 'mixed' and self.wave == 2 and rank < 2 and self.generation[rank] >= 3 else 1
                        assert step['token_ids'] == [expected], 'must feed actual generated token'
                        state['request_tokens'] += 1
                    elif action == 3:
                        assert state['request_active']
                        state.update(request_active=False, request_tokens=0, request_capacity=0, request_kv_bytes=0)
                    else:
                        assert not state['request_active']
                    if graph:
                        state['captures'] = 1
                        state['replays'] += 0 if fault == 'replay' else 1
                    if action in (1, 2):
                        self.generation[rank] += 1
                        changed = fault == 'mixed' and self.wave == 2 and rank < 2 and self.generation[rank] >= 3
                        token = 2 if changed else 1
                        values = [0., 10., 10.001 if changed else 9.999, 0., 0., 0., 0., 0.]
                        tokens.append(token)
                        logits.append(values if trace else [])
                    else:
                        tokens.append(None)
                        logits.append([])
                if fault == 'logits' and trace:
                    logits[0] = []
                if fault == 'kv' and 3 in actions:
                    self.states[actions.index(3)]['request_kv_bytes'] = 20
                return dict(status='PASS', states=copy.deepcopy(self.states), token_ids=tokens, logits=logits,
                            execution='eager' if fault == 'path' else 'decode_cuda_graph' if graph else 'eager')

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, build = root / 'model', root / 'build'
            model.mkdir()
            build.mkdir()
            (model / 'config.json').write_text('{"vocab_size": 8}')
            for name in ('CMakeCache.txt', 'fastllm_moe_generate', 'fastllm_tp_worker'):
                (build / name).write_text('mock data; never execute')
            argv = ['probe', '--model', str(model), '--build', str(build), '--prompt-length', '2',
                    '--output-tokens', '4', '--trace-tokens', '2', '3', '4', '--output', str(root / 'out')]
            if waves:
                argv += ['--first-wave', *map(str, waves)]
            with mock.patch('sys.argv', argv), mock.patch.object(probe, 'capture', return_value='mock hardware'), \
                    mock.patch.object(probe, 'RoundTransport', Transport), redirect_stdout(io.StringIO()):
                code = probe.main()
            output, = (root / 'out').iterdir()
            summary = json.loads((output / 'summary.json').read_text())
            for extension in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((output / f'summary.{extension}').is_file())
            raws = {str(p.relative_to(output)): json.loads(p.read_text()) for p in output.glob('*/*/rank*.json')}
            details = {p.stem: json.loads(p.read_text()) for p in output.glob('*_rank*.json')}
        self.assertTrue(all(not s.options['prefix_cache'] for s in sessions))
        return code, summary, sessions, raws, details

    def test_mixed_admission_can_change_outputs_in_both_modes(self):
        code, report, sessions, raws, details = self.run_probe('mixed')
        self.assertEqual(code, 1)
        self.assertTrue(report['complete'])
        self.assertEqual(report['expected_case_count'], 80)
        self.assertEqual(sum(r['status'] == 'FAIL' for r in report['cases']), 20)
        self.assertEqual(report['metadata']['reproduction_status'], 'OBSERVED_TOKEN_DIVERGENCE')
        self.assertTrue(all('median_ms' not in r and 'speedup' not in r for r in report['cases']))
        self.assertEqual(len(raws), 96)
        for session in sessions:
            first = [s['action'] for s in session.history[0][0]]
            if first == [1, 1, 0, 0]:
                self.assertEqual([s['action'] for s in session.history[1][0]], [2, 2, 1, 1])
                self.assertEqual([s['action'] for s in session.history[2][0]], [2, 2, 2, 2])
            else:
                self.assertEqual(first, [1, 1, 1, 1])
                self.assertEqual([s['action'] for s in session.history[1][0]], [2, 2, 2, 2])
        actual = raws['graph_first2_then2/repeat01/rank0.json']
        self.assertEqual(actual['token_ids'], [1, 1, 2, 2])
        self.assertEqual([s['execution'] for s in actual['steps']], ['eager', 'eager', 'decode_cuda_graph', 'decode_cuda_graph'])
        comparison = details['graph_first2_then2_repeat01_rank0']['comparisons']
        self.assertTrue(comparison['vs_eager']['token_comparison']['match'])
        self.assertTrue(comparison['vs_first']['token_comparison']['match'])
        self.assertEqual(comparison['vs_all4']['first_divergence_trace']['token_1based'], 3)
        self.assertTrue(comparison['vs_all4']['first_divergence_trace']['same_input_history'])
        self.assertFalse(comparison['vs_all4']['traces'][-1]['same_input_history'])

    def test_stable_outputs_only_report_not_observed(self):
        code, report, _, _, _ = self.run_probe()
        self.assertEqual(code, 0)
        self.assertTrue(report['complete'])
        self.assertEqual(report['metadata']['reproduction_status'], 'NOT_OBSERVED')

    def test_one_and_three_request_first_waves_use_exact_joint_rounds(self):
        code, report, sessions, _, _ = self.run_probe(waves=[1, 4, 3])
        self.assertEqual(code, 0)
        self.assertEqual(report['metadata']['first_waves'], [4, 1, 3])
        self.assertEqual(report['expected_case_count'], 120)
        for session, wave in zip(sessions, [4, 1, 3]*2):
            self.assertEqual([s['action'] for s in session.history[0][0]], [1]*wave + [0]*(4-wave))
            self.assertEqual([s['action'] for s in session.history[1][0]], [2]*wave + [1]*(4-wave))

    def test_path_replay_trace_kv_and_worker_failures_are_not_accepted(self):
        for fault in ('path', 'replay', 'logits', 'kv', 'worker', 'interrupt'):
            with self.subTest(fault=fault):
                code, report, _, raw, _ = self.run_probe(fault)
                self.assertEqual(code, 1)
                self.assertFalse(report['complete'])
                self.assertEqual(report['metadata']['reproduction_status'], 'INCOMPLETE')
                self.assertTrue(raw)

    def test_invalid_parameters_never_start_native(self):
        for extra in (['--first-wave', '2'], ['--first-wave', '4', '4'], ['--output-tokens', '2'],
                      ['--trace-tokens', '0'], ['--trace-tokens', '2', '2'], ['--repeats', '4'], ['--timeout', 'nan']):
            with self.subTest(extra=extra), mock.patch('sys.argv', ['probe', '--model', '/unused', *extra]), \
                    mock.patch.object(probe, 'RoundTransport') as transport, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    probe.main()
                self.assertEqual(error.exception.code, 2)
                transport.assert_not_called()


if __name__ == '__main__':
    unittest.main()
