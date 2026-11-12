"""前缀缓存协议/生命周期回归；不编译，不把模拟诊断当作 GPU 数值验收。"""

import copy
from pathlib import Path
import unittest

import test_qwen3_session as fixture
from fastllm_pytools.qwen3_process_session import Qwen3ProcessSession, GenerationRejected
from fastllm_pytools.tp_graph import validate_prefix
from run_qwen3_prefix_cache import make_tasks
from run_qwen3_decode import expectations


def states(**changes):
    p = dict(enabled=True, entries=1, device_bytes=64, lookups=2, hits=1, reused_tokens=3,
             last_reused_tokens=3, last_computed_tokens=1, evictions=0, clears=0)
    p.update(changes)
    return [dict(prefix_cache=dict(p)) for _ in range(2)]


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def session(self, mode='prefix_normal', graph=False, custom=False):
        f = self.fixture
        return Qwen3ProcessSession(f.binary, f.binary, f.root/mode, '0,1',
                                  f.root/f'{mode}-{graph}-{custom}', timeout=5,
                                  cuda_graph=graph, custom_allreduce=custom, prefix_cache=True)

    def test_four_modes_reuse_clear_reject_and_unload(self):
        for graph in (False, True):
            for custom in (False, True):
                with self.subTest(graph=graph, custom=custom), self.session(graph=graph, custom=custom) as session:
                    session.clear_prefix_cache()
                    for ids, reused in (([1,2,3,4], 0), ([1,2,3,4], 3), ([1,2,3,4,5], 4),
                                        ([1,2], 1), ([1,7,9], 1), ([3,7], 0), ([3], 0)):
                        result = session.generate(dict(self.fixture.request, token_ids=ids, max_new_tokens=3))
                        self.assertEqual(result['graph'][0]['prefix_cache']['last_reused_tokens'], reused)
                    before = copy.deepcopy(session._graph_state)
                    with self.assertRaises(GenerationRejected):
                        session.generate(dict(self.fixture.request, max_new_tokens=0))
                    self.assertEqual(session._graph_state, before)
                    session.clear_prefix_cache()
                    result = session.generate(self.fixture.request)
                    self.assertEqual(result['graph'][0]['prefix_cache']['last_reused_tokens'], 0)
                    self.assertIn('--prefix-cache 1', (session.output/'command.txt').read_text())
                self.assertEqual(session.state, 'closed')
                validate_prefix(session.shutdown['graph_after_unload'], 2, True, reset=True)

    def test_bad_diagnostics_fail_and_reap_process(self):
        for mode in ('prefix_missing', 'prefix_float_ready', 'prefix_float_result',
                     'prefix_float_shutdown', 'prefix_fallback', 'prefix_stale', 'prefix_leak'):
            with self.subTest(mode=mode):
                session = self.session(mode)
                with self.assertRaisesRegex(RuntimeError, 'prefix cache'):
                    with session:
                        session.generate(self.fixture.request)
                self.assertEqual(session.state, 'failed')
                self.assertFalse(Path(f'/proc/{session._process.pid}').exists())

    def test_types_limits_and_rank_mismatch(self):
        validate_prefix(states(), 2, True)
        # 各 rank 的 KV 头分片可不同，因此资源字节不要求相等。
        unequal = states()
        unequal[1]['prefix_cache']['device_bytes'] = 32
        validate_prefix(unequal, 2, True)
        for update in (dict(enabled=1), dict(hits=True), dict(entries=9), dict(device_bytes=0),
                       dict(device_bytes=64*1024*1024+1), dict(lookups=-1), dict(hits=3),
                       dict(last_reused_tokens=2048), dict(last_computed_tokens=0), dict(hits=0)):
            bad = states()
            bad[1]['prefix_cache'].update(update)
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                validate_prefix(bad, 2, True)
        for bad in (None, [], [{}, {}]):
            with self.assertRaises(RuntimeError):
                validate_prefix(bad, 2, True)

    def test_request_delta_clear_and_reset_evidence(self):
        before = states(lookups=1, hits=0, reused_tokens=0, last_reused_tokens=0, last_computed_tokens=4)
        validate_prefix(states(), 2, True, previous=before, prompt_tokens=4)
        for bad in (states(lookups=1), states(reused_tokens=4), states(last_computed_tokens=2)):
            with self.assertRaises(RuntimeError):
                validate_prefix(bad, 2, True, previous=before, prompt_tokens=4)
        cleared = states(entries=0, device_bytes=0, clears=1)
        validate_prefix(cleared, 2, True, previous=states(), cleared=True)
        with self.assertRaises(RuntimeError):
            validate_prefix(states(), 2, True, previous=states(), cleared=True)
        zero = states(**{k: 0 for k in states()[0]['prefix_cache'] if k != 'enabled'})
        validate_prefix(zero, 2, True, previous=states(), reset=True)
        with self.assertRaises(RuntimeError):
            validate_prefix(states(), 2, True, reset=True)

    def test_cache_flag_requires_boolean(self):
        f = self.fixture
        for value in (1, 'true', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Qwen3ProcessSession(f.binary, f.binary, f.root, '0,1', f.root/'unused', prefix_cache=value)

    def test_server_task_plan_is_valid_and_exercises_graph_clear(self):
        tasks = make_tasks()
        expected = expectations(tasks)
        self.assertEqual(len({t['name'] for t in tasks}), len(tasks))
        self.assertTrue(any(t['op'] == 'clear_prefix' and e['active'] for t, e in zip(tasks, expected)))
        self.assertTrue(any(e['tokens'] == 129 for e in expected))
        self.assertLess(sum('ids' in e for e in expected)*151936, 8*1024*1024)
        for t, e in zip(tasks, expected):
            if t['op'] == 'begin':
                self.assertTrue(1 <= len(t['token_ids']) <= 128)
            self.assertLessEqual(e['tokens'], e['capacity'])

    def test_budget_and_allreduce_boundary_plans(self):
        budget = [t for t in make_tasks('budget') if t['op'] == 'begin']
        lengths = [len(t['token_ids']) for t in budget]
        self.assertEqual(lengths[:3], [127, 1, 1])
        self.assertEqual((sum(lengths[:2])*512*1024, sum(lengths[:3])*512*1024),
                         (64*1024*1024, 64*1024*1024+512*1024))
        wide = [t for t in make_tasks('wide') if t['op'] == 'begin']
        self.assertEqual(wide[0]['token_ids'], wide[1]['token_ids'])
        self.assertGreater(len(wide[0]['token_ids'])*1024, 65536)

    def test_allreduce_verifier_uses_executed_suffix_length(self):
        from run_qwen3_decode import verify_allreduce
        from test_process_allreduce import states as ar_states
        tasks = [dict(name='cold', op='begin', token_ids=[1]*65), dict(name='end', op='end'),
                 dict(name='hit', op='begin', token_ids=[1]*65)]
        rows = []
        for custom, computed in ((0,65), (0,65), (2,1)):
            observed = ar_states(custom, 2)
            for state in observed:
                state['prefix_cache'] = dict(enabled=True, last_computed_tokens=computed)
            rows.append(dict(graph=observed))
        report = dict(results=rows, graph_after_unload=ar_states(2,2))
        verify_allreduce(report, tasks, dict(hidden_size=1024, num_hidden_layers=1), False)
        rows[-1]['graph'][0]['prefix_cache']['last_computed_tokens'] = 65
        with self.assertRaisesRegex(AssertionError, 'execution counts'):
            verify_allreduce(report, tasks, dict(hidden_size=1024, num_hidden_layers=1), False)


if __name__ == '__main__':
    unittest.main()
