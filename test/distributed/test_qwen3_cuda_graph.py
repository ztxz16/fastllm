"""Graph 开关、执行证据和生命周期的纯 Python 回归；不替代服务器 CUDA 验收。"""

import copy
from pathlib import Path
import unittest

import test_qwen3_session as session_fixture
from qwen3_process_session import Qwen3ProcessSession
from fastllm_pytools.tp_graph import validate_graph
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model
import test_qwen3_decode_runner as decode_fixture
from run_qwen3_decode import verify_graph


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.fixture = session_fixture.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def session(self, mode='graph_normal', enabled=True):
        f = self.fixture
        return Qwen3ProcessSession(f.binary, f.binary, f.root / mode, '0,1', f.root / (mode + f'-graph-{enabled}'),
                                   timeout=5, cuda_graph=enabled)

    def test_eager_session_reports_integer_counters_and_reuses_workers(self):
        with self.session(enabled=False) as session:
            self.assertEqual(session.ready['execution'], 'eager')
            for count, expected in ((1, 0), (3, 2), (2, 3)):
                report = session.generate(dict(self.fixture.request, max_new_tokens=count))
                for state in report['graph']:
                    self.assertFalse(state['enabled'])
                    self.assertEqual(state['eager_decodes'], expected)
                    self.assertEqual(state['captures'], 0)
        self.assertEqual(session.shutdown['graph_after_unload'][0]['eager_decodes'], 0)

    def test_float_counters_rejected_on_wire_in_both_modes(self):
        # 重现原生 double 序列化后成为 JSON 小数的问题，覆盖 Eager 也会读取的启动状态。
        for enabled in (False, True):
            for event in ('ready', 'result', 'shutdown'):
                with self.subTest(enabled=enabled, event=event):
                    session = self.session('graph_float_' + event, enabled=enabled)
                    with self.assertRaisesRegex(RuntimeError, 'invalid CUDA Graph diagnostics'):
                        with session:
                            session.generate(self.fixture.request)
                    self.assertEqual(session.state, 'failed')
                    self.assertFalse(Path(f'/proc/{session._process.pid}').exists())

    def test_session_replays_and_cleans_up_each_request(self):
        with self.session() as session:
            self.assertIn('--cuda-graph 1', (session.output / 'command.txt').read_text())
            for count, captures, replays in ((1, 0, 0), (3, 1, 2), (2, 2, 3)):
                report = session.generate(dict(self.fixture.request, max_new_tokens=count))
                self.assertEqual((report['graph'][0]['captures'], report['graph'][0]['replays']), (captures, replays))
                self.assertFalse(report['graph'][0]['active'])
        self.assertEqual(session.shutdown['graph_after_unload'][0]['replays'], 0)
        self.assertEqual(session.state, 'closed')

    def test_old_binary_cannot_silently_ignore_graph(self):
        session = self.session('graph_missing')
        with self.assertRaisesRegex(RuntimeError, 'Graph'):
            session.__enter__()
        self.assertEqual(session.state, 'failed')
        self.assertFalse(Path(f'/proc/{session._process.pid}').exists())

    def test_fallback_leak_or_stale_counters_fail_and_reap(self):
        for mode in ('graph_fallback', 'graph_leak', 'graph_stale'):
            with self.subTest(mode=mode):
                with self.session(mode) as session:
                    count = 3 if mode == 'graph_stale' else 1
                    with self.assertRaisesRegex(RuntimeError, 'Graph'):
                        session.generate(dict(self.fixture.request, max_new_tokens=count))
                    self.assertEqual(session.state, 'failed')
                    self.assertFalse(Path(f'/proc/{session._process.pid}').exists())

    def test_graph_diagnostic_type_and_rank_consistency(self):
        valid = dict(enabled=True, active=False, captures=1, replays=3, warmups=1, eager_decodes=0, workspace_bytes=0)
        validate_graph([valid, dict(valid)], 2, True)
        mutations = [dict(enabled=False), dict(active=1), dict(replays=True), dict(replays=-1),
                     dict(warmups=0), dict(eager_decodes=1), dict(active=True, workspace_bytes=4096),
                     dict(replays=4), dict(workspace_bytes=1), dict(captures=2, warmups=2)]
        for update in mutations:
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                validate_graph([valid, dict(valid, **update)], 2, True)
        for value in (None, [], {}, [valid]):
            with self.assertRaises(RuntimeError):
                validate_graph(value, 2, True)

    def test_thread_factory_rejects_process_only_flag(self):
        args = make_normal_parser('test').parse_args(['--cuda-graph'])
        with self.assertRaisesRegex(ValueError, 'requires --tp_backend process'):
            make_normal_llm_model(args)

    def test_decode_runner_rejects_missing_or_incorrect_execution_evidence(self):
        fixture = decode_fixture.DecodeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        report = fixture.report([0, 1])
        verify_graph(report, fixture.tasks, 2, False)
        bad = copy.deepcopy(report)
        for result in bad['results']:
            for rank in result['graph']:
                rank['eager_decodes'] = 0
        with self.assertRaisesRegex(AssertionError, 'lifecycle'):
            verify_graph(bad, fixture.tasks, 2, False)
        with self.assertRaisesRegex(AssertionError, 'execution'):
            verify_graph(report, fixture.tasks, 2, True)

    def test_decode_graph_lifetime_with_prefill_and_reload(self):
        ops = ['query', 'begin', 'decode', 'prefill', 'decode', 'end', 'begin', 'decode', 'unload', 'load']
        tasks = [dict(name=str(i), op=op) for i, op in enumerate(ops)]
        expected = [(0, 0, False), (0, 0, False), (1, 1, True), (1, 1, True), (1, 2, True),
                    (1, 2, False), (1, 2, False), (2, 3, True), (0, 0, False), (0, 0, False)]
        results = []
        for captures, replays, active in expected:
            state = dict(enabled=True, captures=captures, replays=replays, warmups=captures,
                         active=active, workspace_bytes=4096 if active else 0, eager_decodes=0)
            results.append(dict(graph=[dict(state), dict(state)]))
        report = dict(execution='decode_cuda_graph', results=results, graph_after_unload=results[-1]['graph'])
        verify_graph(report, tasks, 2, True)
        for index in (3, 5, 7, 8):
            wrong = copy.deepcopy(report)
            wrong['results'][index]['graph'] = copy.deepcopy(results[index - 1]['graph'])
            if index == 3:
                for rank in wrong['results'][index]['graph']:
                    rank['replays'] += 1
            with self.subTest(index=index), self.assertRaises((AssertionError, RuntimeError)):
                verify_graph(wrong, tasks, 2, True)


if __name__ == '__main__':
    unittest.main()
