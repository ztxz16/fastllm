"""前缀路由的正式工厂回归；原生传输使用Python子进程，不编译或访问GPU。"""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import threading
import unittest
from unittest import mock

import test_process_dp as fixtures
from fastllm_pytools import process_tp
from fastllm_pytools.parallel_config import validate_parallel_options
from fastllm_pytools.qwen3_process_session import GenerationCancelled
from fastllm_pytools.tp_graph import validate_prefix, validate_prefix_directory
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProcessDPTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.rename_model('dp_routing')
        self.request = dict(self.fixture.request, token_ids=[1, 2, 3, 4])

    def args(self, *extra):
        return self.fixture.args('--prefix-cache', 'true', '--dp-routing', 'prefix', *extra)

    def test_affinity_partial_prefix_and_dp_tp_sizes(self):
        for dp, tp in ((2, 1), (4, 1), (2, 2)):
            with self.subTest(dp=dp, tp=tp), make_normal_llm_model(self.args('--dp', str(dp), '--tp', str(tp))) as model:
                a = model.generate(self.request)
                b = model.generate(dict(self.request, token_ids=[5, 6, 7, 8]))
                self.assertEqual((a['dp_rank'], b['dp_rank']), (0, 1))
                for ids, rank, reused in (([1, 2, 3, 4], 0, 3), ([5, 6, 10], 1, 2)):
                    result = model.generate(dict(self.request, token_ids=ids))
                    self.assertEqual(result['token_ids'], [ids[0]])
                    self.assertEqual(result['dp_rank'], rank)
                    self.assertEqual(result['dp_routing']['estimated_reused_tokens'], reused)
                    self.assertEqual(result['graph'][0]['prefix_cache']['last_reused_tokens'], reused)
                self.assertEqual(model.ready['dp_routing'], 'prefix')

    def test_default_rotation_and_single_token_ties(self):
        for flags, request in ((('--dp-routing', 'round_robin'), self.request),
                               ((), dict(self.request, token_ids=[1]))):
            with make_normal_llm_model(self.args(*flags)) as model:
                self.assertEqual([model.generate(request)['dp_rank'] for _ in range(4)], [0, 1, 0, 1])

    def test_busy_cached_replica_does_not_leave_other_gpu_idle(self):
        with make_normal_llm_model(self.args()) as model, ThreadPoolExecutor(1) as pool:
            model.generate(self.request)
            emitted, cancel = threading.Event(), threading.Event()
            future = pool.submit(model.generate, dict(self.request, max_new_tokens=128),
                                 cancel_event=cancel, on_token=lambda _: emitted.set())
            try:
                self.assertTrue(emitted.wait(3))
                result = model.generate(self.request)
                self.assertEqual(result['dp_rank'], 1)
                self.assertEqual(result['dp_routing']['idle_ranks'], [1])
                self.assertEqual(result['dp_routing']['estimated_reused_tokens'], 0)
            finally:
                cancel.set()
            with self.assertRaises(GenerationCancelled):
                future.result(timeout=5)

    def test_cancelled_prefill_directory_is_retained_and_clear_removes_it(self):
        with make_normal_llm_model(self.args()) as model:
            cancel = threading.Event()
            with self.assertRaises(GenerationCancelled):
                model.generate(dict(self.request, max_new_tokens=128), cancel_event=cancel, on_token=lambda _: cancel.set())
            result = model.generate(self.request)
            self.assertEqual(result['dp_rank'], 0)
            self.assertEqual(result['dp_routing']['estimated_reused_tokens'], 3)
            model.clear_prefix_cache()
            self.assertTrue(all(not entries for entries in model._prefixes))
            result = model.generate(self.request)
            self.assertEqual(result['dp_routing']['estimated_reused_tokens'], 0)
            self.assertEqual(result['graph'][0]['prefix_cache']['last_reused_tokens'], 0)

    def test_evicted_snapshot_is_not_a_full_prefix_hit(self):
        with make_normal_llm_model(self.args()) as model:
            for second in range(2, 11):
                before = model.replicas[0].session.prefix_directory()
                result = model.generate(dict(self.request, token_ids=[1, second, 15]))
                self.assertEqual(result['dp_rank'], 0)
            self.assertEqual(result['graph'][0]['prefix_cache']['entries'], 8)
            self.assertEqual(result['graph'][0]['prefix_cache']['evictions'], 1)
            after = model.replicas[0].session.prefix_directory()
            evicted = next(entry['token_ids'] for entry in before if entry not in after)
            result = model.generate(dict(self.request, token_ids=evicted))
            self.assertEqual(result['dp_routing']['estimated_reused_tokens'], 1)
            self.assertEqual(result['graph'][0]['prefix_cache']['last_reused_tokens'], 1)

    def test_text_stream_and_http_use_same_routing(self):
        from fastllm_pytools.qwen3_process_server import ChatService
        with make_normal_llm_model(self.args()) as model:
            tokenizer = self.fixture.fixture.tokenizer
            self.assertEqual(model.response('你好', max_length=1), '答')
            self.assertEqual(''.join(model.stream_response('你好', max_length=1)), '答')
            self.assertEqual(tokenizer.apply_chat_template.call_count, 2)
            self.assertEqual(model.replicas[0].session._next_id, 3)
            service = ChatService(model, tokenizer, self.fixture.model, 'qwen3', model.output,
                                  queue_size=4, request_timeout=5)
            try:
                self.assertEqual(service.health()['dp_routing'], 'prefix')
                job = service.submit(dict(model='qwen3', messages=[dict(role='user', content='你好')], max_tokens=1), 'prefix-http')
                self.assertTrue(job.done.wait(5))
                self.assertEqual(job.dp_rank, 0)
                self.assertEqual(job.status, 200, job.response)
            finally:
                service.close()

    def test_stream_close_refreshes_cache_and_releases_lease(self):
        with make_normal_llm_model(self.args()) as model:
            stream = model.stream_response('你好', max_length=128)
            self.assertEqual(next(stream), '答')
            stream.close()
            self.assertFalse(model._busy)
            self.assertEqual(model.generate(dict(self.request, token_ids=[1, 2]))['dp_rank'], 0)

    def test_cancelled_waiter_never_dispatches(self):
        with make_normal_llm_model(self.args()) as model, ThreadPoolExecutor(3) as pool:
            cancels = [threading.Event(), threading.Event()]
            started = [threading.Event(), threading.Event()]
            futures = [pool.submit(model.generate, dict(self.request, token_ids=[i+1, 3], max_new_tokens=128),
                                   cancel_event=cancels[i], on_token=lambda _, i=i: started[i].set()) for i in range(2)]
            try:
                self.assertTrue(all(event.wait(3) for event in started))
                before = model._next_id
                cancelled = threading.Event()
                cancelled.set()
                with self.assertRaises(GenerationCancelled):
                    model.generate(self.request, cancel_event=cancelled)
                self.assertEqual(model._next_id, before)
            finally:
                for event in cancels:
                    event.set()
                for future in futures:
                    with self.assertRaises(GenerationCancelled):
                        future.result(timeout=5)

    def test_missing_directory_refuses_policy_and_corruption_fails_pool(self):
        self.fixture.rename_model('dp_routing_missing')
        with self.assertRaisesRegex(RuntimeError, 'directory'):
            make_normal_llm_model(self.args())
        self.fixture.rename_model('dp_routing_corrupt')
        with make_normal_llm_model(self.args()) as model:
            with self.assertRaisesRegex(RuntimeError, 'directory'):
                model.generate(self.request)
            self.assertEqual(model.state, 'failed')
            with self.assertRaisesRegex(RuntimeError, 'pool is failed'):
                model.generate(self.request)

    def test_cli_json_and_factory_reject_conflicts_before_start(self):
        args = self.args()
        for changes in (dict(dp_routing=None), dict(dp_routing=True), dict(dp_routing=[]),
                        dict(dp_routing='invalid'), dict(dp=1), dict(tp_backend='thread'), dict(prefix_cache='false')):
            with self.subTest(changes=changes):
                values = dict(vars(args), **changes)
                with self.assertRaises(ValueError):
                    validate_parallel_options(type(args)(**values))
                path = self.fixture.root / 'routing.json'
                path.write_text(json.dumps(values))
                with mock.patch.object(process_tp, 'Qwen3ProcessSession') as session, self.assertRaises(ValueError):
                    make_normal_llm_model(make_normal_parser('test').parse_args([str(path)]))
                session.assert_not_called()

    def test_partial_clear_failure_disables_routing_and_invalidates_directory(self):
        with make_normal_llm_model(self.args()) as model:
            model.generate(self.request)
            with mock.patch.object(model.replicas[1], 'clear_prefix_cache', side_effect=RuntimeError('clear failed')):
                with self.assertRaisesRegex(RuntimeError, 'clear failed'):
                    model.clear_prefix_cache()
            self.assertTrue(all(not entries for entries in model._prefixes))
            self.assertEqual(model.state, 'failed')
            self.assertFalse(model._maintenance)
            with self.assertRaisesRegex(RuntimeError, 'pool is failed'):
                model.generate(self.request)

    def test_clear_timeout_before_native_work_preserves_pool(self):
        with make_normal_llm_model(self.args()) as model:
            model.generate(self.request)
            before = copy.deepcopy(model._prefixes)
            with mock.patch.object(model._condition, 'wait_for', side_effect=[True, False]):
                with self.assertRaisesRegex(TimeoutError, 'callers'):
                    model.clear_prefix_cache()
            self.assertEqual(model.state, 'ready')
            self.assertEqual(model._prefixes, before)
            self.assertFalse(model._maintenance)
            self.assertEqual(model.generate(self.request)['dp_routing']['estimated_reused_tokens'], 3)

    def test_directory_shape_version_and_rank_consistency(self):
        with make_normal_llm_model(self.args('--tp', '2')) as model:
            result = model.generate(self.request)
        state = result['graph'][0]['prefix_cache']
        for changes in (dict(directory_version=True), dict(directory_version=2), dict(snapshots=[]),
                        dict(snapshots=[dict(position_begin=0, token_ids=[True])]),
                        dict(snapshots=[dict(position_begin=1048576, token_ids=[1, 2])])):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                validate_prefix_directory(dict(state, **changes), required=True)
        bad = copy.deepcopy(result['graph'])
        bad[1]['prefix_cache']['snapshots'][0]['token_ids'][0] = 8
        with self.assertRaisesRegex(RuntimeError, 'directory ranks'):
            validate_prefix(bad, 2, True)


if __name__ == '__main__':
    unittest.main()
