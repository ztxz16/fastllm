"""自定义规约客户端、实际路径判定及拒绝路径；不编译，不代替 GPU 数值验收。"""

import copy
from pathlib import Path
import unittest

import test_qwen3_session as fixture
from fastllm_pytools.qwen3_process_session import Qwen3ProcessSession
from fastllm_pytools.tp_graph import validate_allreduce
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model
from run_qwen3_decode import verify_allreduce


def states(custom=0, nccl=0):
    return [dict(allreduce=dict(enabled=True, active=True, custom_calls=custom,
                               nccl_calls=nccl, workspace_bytes=2*1024*1024+4)) for _ in range(2)]


class AllReduceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def session(self, mode='ar_normal', graph=False):
        f = self.fixture
        return Qwen3ProcessSession(f.binary, f.binary, f.root / mode, '0,1',
                                  f.root / f'{mode}-{graph}', timeout=5,
                                  cuda_graph=graph, custom_allreduce=True)

    def test_eager_graph_replay_and_model_unload_preserve_ipc_counters(self):
        for graph in (False, True):
            with self.session(graph=graph) as session:
                self.assertIn('--custom-allreduce 1', (session.output / 'command.txt').read_text())
                old = 0
                for count in (1, 3, 2):
                    report = session.generate(dict(self.fixture.request, max_new_tokens=count))
                    now = report['graph'][0]['allreduce']['custom_calls']
                    self.assertEqual(now-old, 2*(count+int(graph and count > 1)))
                    old = now
            self.assertEqual(session.shutdown['graph_after_unload'][0]['allreduce']['custom_calls'], old)
            self.assertEqual(session.state, 'closed')

    def test_missing_fallback_float_stale_or_reset_evidence_kills_group(self):
        for graph in (False, True):
            for mode in ('ar_missing', 'ar_fallback', 'ar_float_ready', 'ar_float_result',
                         'ar_float_shutdown', 'ar_stale', 'ar_reset'):
                with self.subTest(graph=graph, mode=mode):
                    session = self.session(mode, graph)
                    with self.assertRaisesRegex(RuntimeError, 'AllReduce'):
                        with session:
                            session.generate(dict(self.fixture.request, max_new_tokens=3))
                    self.assertEqual(session.state, 'failed')
                    self.assertFalse(Path(f'/proc/{session._process.pid}').exists())

    def test_argument_and_thread_backend_rejections(self):
        f = self.fixture
        for value, devices in ((1, '0,1'), ('yes', '0,1'), (True, '0'), (True, '0,1,2')):
            with self.assertRaises(ValueError):
                Qwen3ProcessSession(f.binary, f.binary, f.root, devices, f.root/'unused', custom_allreduce=value)
        args = make_normal_parser('test').parse_args(['--custom-allreduce'])
        with self.assertRaisesRegex(ValueError, 'requires --tp_backend process'):
            make_normal_llm_model(args)

    def test_diagnostic_types_ranks_resources_and_progress(self):
        good = states(4, 2)
        validate_allreduce(good, 2, True, previous=states(2, 1), require_progress=True)
        for update in (dict(enabled=1), dict(active=False), dict(custom_calls=4.0),
                       dict(nccl_calls=True), dict(custom_calls=-1), dict(workspace_bytes=0),
                       dict(workspace_bytes=1), dict(custom_calls=5)):
            bad = copy.deepcopy(good)
            bad[1]['allreduce'].update(update)
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                validate_allreduce(bad, 2, True)
        for bad in (None, [], [{}], [{}, {}]):
            with self.assertRaises(RuntimeError):
                validate_allreduce(bad, 2, True)
        with self.assertRaisesRegex(RuntimeError, 'not executed'):
            validate_allreduce(good, 2, True, previous=good, require_progress=True)
        with self.assertRaisesRegex(RuntimeError, 'regressed'):
            validate_allreduce(states(3, 2), 2, True, previous=good)

    def test_decode_counts_warmup_boundary_interleave_unload_and_reload(self):
        config = dict(hidden_size=1024, num_hidden_layers=2)
        tasks = [dict(name='begin', op='begin', token_ids=[1]*64), # exactly 65536
                 dict(name='first', op='decode'), dict(name='second', op='decode'),
                 dict(name='large', op='prefill', token_ids=[1]*65),
                 dict(name='unload', op='unload'), dict(name='reload', op='load'),
                 dict(name='new', op='begin', token_ids=[1]*63),
                 dict(name='again', op='decode'), dict(name='end', op='end')]
        for graph, totals in ((False, [4,8,12,12,12,12,16,20,20]),
                              (True, [4,12,16,16,16,16,20,28,28])):
            rows = [dict(graph=states(total, 4 if index >= 3 else 0)) for index, total in enumerate(totals)]
            report = dict(results=rows, graph_after_unload=states(totals[-1], 4))
            verify_allreduce(report, tasks, config, graph)
            bad = copy.deepcopy(report)
            bad['results'][2]['graph'] = states(totals[1], 0)
            with self.assertRaisesRegex(AssertionError, 'execution counts'):
                verify_allreduce(bad, tasks, config, graph)


if __name__ == '__main__':
    unittest.main()
