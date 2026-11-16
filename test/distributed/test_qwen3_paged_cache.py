"""分页协议和边界测试；仅运行Python，不编译或加载CUDA。"""
import copy
import unittest
from types import SimpleNamespace

from run_qwen3_prefix_cache import make_tasks
from run_qwen3_decode import expected_cache
from fastllm_pytools.process_dp import ProcessDPModel
from fastllm_pytools.tp_graph import validate_prefix, validate_prefix_directory


def paged_states():
    prefix = dict(enabled=True, entries=1, device_bytes=512,
                  lookups=2, hits=1, reused_tokens=128,
                  last_reused_tokens=128, last_computed_tokens=1,
                  evictions=0, clears=0, directory_version=2,
                  cache_layout='paged', page_len=128, max_pages=300,
                  pool_bytes=153600, active_pages=2, cached_pages=1,
                  attention_backend='native_paged',
                  snapshots=[dict(position_begin=0, token_ids=[1]*128)])
    return [dict(prefix_cache=copy.deepcopy(prefix)) for _ in range(2)]


class PagedCacheTests(unittest.TestCase):
    def test_paged_capacity_and_local_resources(self):
        states = paged_states()
        states[1]['prefix_cache'].update(device_bytes=80*1024*1024, pool_bytes=160*1024*1024, evictions=3)
        validate_prefix(states, 2, True)

    def test_reject_partial_pages_and_wrong_backend(self):
        for update in (dict(last_reused_tokens=127), dict(page_len=0), dict(max_pages=0),
                       dict(page_len=64), dict(max_pages=299), dict(attention_backend='legacy_contiguous'), dict(device_bytes=153601),
                       dict(snapshots=[dict(position_begin=0, token_ids=[1]*127)])):
            states = paged_states()
            states[0]['prefix_cache'].update(update)
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                validate_prefix(states, 2, True)

    def test_clear_preserves_active_page_references(self):
        before = paged_states()
        after = copy.deepcopy(before)
        for state in after:
            state['prefix_cache'].update(entries=0, device_bytes=0, cached_pages=0, snapshots=[], clears=1)
        validate_prefix(after, 2, True, previous=before, cleared=True)
        self.assertEqual(after[0]['prefix_cache']['active_pages'], 2)

    def test_page_boundary_tasks(self):
        tasks = make_tasks('paged_boundaries', 128)
        lengths = [len(t['token_ids']) for t in tasks if t['op'] == 'begin']
        self.assertEqual(lengths, [127,127,128,128,129,129,255,255,256,256,257,257])
        self.assertEqual(len({t['name'] for t in tasks}), len(tasks))

    def test_active_bytes_follow_pages_not_reserved_capacity(self):
        config = dict(num_key_value_heads=2, num_attention_heads=2, hidden_size=8,
                      num_hidden_layers=1, head_dim=4)
        state = dict(active=True, tokens=129, capacity=2048, position_begin=0)
        cache = expected_cache(config, state, 0, 2, page_size=128)
        self.assertEqual(cache['device_bytes'], 2*128*2*1*4*2)

    def test_dp_routing_uses_complete_pages(self):
        model = object.__new__(ProcessDPModel)
        model._prefixes = [[dict(position_begin=0, token_ids=[1]*256),
                            dict(position_begin=7, token_ids=[2]*256)]]
        model.replicas = [SimpleNamespace(session=SimpleNamespace(_graph_state=paged_states()))]
        for length, expected in ((127, 0), (128, 0), (129, 128), (256, 128), (257, 256)):
            with self.subTest(length=length):
                self.assertEqual(model._prefix_score(0, [1]*length), expected)
        self.assertEqual(model._prefix_score(0, [2]*257), 0)

    def test_rank_local_eviction_and_clear_counters(self):
        before = paged_states()
        after = copy.deepcopy(before)
        after[1]['prefix_cache'].update(evictions=1, clears=1)
        for state in after:
            state['prefix_cache'].update(entries=0, snapshots=[])
        validate_prefix(after, 2, True, previous=before)
        regressed = copy.deepcopy(after)
        regressed[1]['prefix_cache']['clears'] = 0
        with self.assertRaisesRegex(RuntimeError, 'regressed'):
            validate_prefix(regressed, 2, True, previous=after)

    def test_directory_keeps_position_identity(self):
        prefix = paged_states()[0]['prefix_cache']
        prefix['snapshots'].append(dict(position_begin=23, token_ids=[1]*128))
        prefix['entries'] = 2
        self.assertEqual(len(validate_prefix_directory(prefix)), 2)


if __name__ == '__main__':
    unittest.main()
