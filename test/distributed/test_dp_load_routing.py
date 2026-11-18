"""正式DP工厂的负载路由回归；原生页面统计与执行闸门来自Python协议子进程。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import threading
import time
import unittest

import test_process_dp as fixtures
import test_qwen3_session as sessions
from fastllm_pytools.qwen3_process_session import GenerationCancelled
from fastllm_pytools.qwen3_process_server import ChatService, HTTPError
from fastllm_pytools.util import make_normal_llm_model


class LoadRoutingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProcessDPTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.rename_model('dp_load_gate')
        self.request = self.fixture.request

    def wait_for(self, predicate):
        """/** 有界等待可观测状态，不以线程启动先后代替实际协议证据。 */"""
        deadline = time.monotonic() + 4
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail('expected routing state was not reached')
            time.sleep(0.01)

    def release(self, device, native_id):
        (self.fixture.model / f'release-{device}-{native_id}').touch()

    @contextmanager
    def running(self, dp=2, tp=1, free=None):
        """/** 真实子进程停在生成闸门；返回前全部副本已上报负载，退出回收全部请求和进程。 */"""
        for path in self.fixture.model.glob('release-*'):
            path.unlink()
        for path in self.fixture.model.glob('begin-*'):
            path.unlink()
        for rank in range(dp):
            counts = (free or [[100] * tp for _ in range(dp)])[rank]
            state = dict(workers=[dict(rank=i, total_pages=100, free_pages=value) for i, value in enumerate(counts)],
                         kv_cache_usage=max(1-value/100 for value in counts))
            (self.fixture.model / f'pressure-{rank*tp}.json').write_text(json.dumps(state))
        model = make_normal_llm_model(self.fixture.args('--dp', str(dp), '--tp', str(tp)))
        pool = ThreadPoolExecutor(dp + 8)
        cancels = [threading.Event() for _ in range(dp)]
        futures = []
        try:
            self.assertEqual(model.routing, 'load')
            for rank in range(dp):
                futures.append(pool.submit(model.generate, self.request, cancel_event=cancels[rank]))
                self.wait_for(lambda rank=rank: (self.fixture.model / f'begin-{rank*tp}-1').exists())
                self.wait_for(lambda rank=rank: model.load_snapshot()[rank]['kv_cache_usage'] ==
                              max(1-value/100 for value in (free or [[100]*tp for _ in range(dp)])[rank]))
            yield model, pool, futures, cancels
        finally:
            model.close()
            model.close()
            pool.shutdown(wait=True)
            self.assertEqual([len(q) for q in model._queues], [0]*dp)
            self.assertEqual(model._busy, set())
            self.fixture.assert_reaped(model.ready['workers'])

    def test_kv_thresholds_and_immediate_accounting(self):
        for free in (100, 51, 50, 49, 0):
            with self.subTest(free=free), self.running(free=[[free], [free]]) as (model, _, _, _):
                first, second = model._reserve(), model._reserve()
                self.assertEqual([first['rank'], second['rank']], [0, 1])
                expected = 2 + 6*max(0, 1-free/100-0.5)
                for state in model.load_snapshot():
                    self.assertEqual((state['waiting'], state['running']), (1, 1))
                    self.assertAlmostEqual(state['score'], expected)
                third = model._reserve()
                self.assertEqual(third['rank'], 0)
                self.assertTrue(model._withdraw(third))
                self.assertFalse(model._withdraw(third))

    def test_tp_uses_worst_worker_and_chooses_lower_pressure(self):
        with self.running(tp=2, free=[[50, 0], [100, 50]]) as (model, _, _, _):
            self.assertEqual([s['kv_cache_usage'] for s in model.load_snapshot()], [1.0, 0.5])
            previous = [model._reserve() for _ in range(2)]
            self.assertEqual([t['rank'] for t in previous], [0, 1])
            chosen = model._reserve()
            self.assertEqual(chosen['rank'], 1)
            self.assertEqual([s['score'] for s in chosen['decision']['replicas']], [5.0, 2.0])
            for ticket in [*previous, chosen]:
                model._withdraw(ticket)
            self.assertEqual(model._reserve()['rank'], 1, 'scan start advances by one, not from the chosen rank')

    def test_fixed_fifo_and_actual_tokens_for_all_layouts(self):
        for dp, tp in ((2, 1), (4, 1), (2, 2)):
            with self.subTest(dp=dp, tp=tp), self.running(dp, tp) as (model, pool, active, _):
                tickets = [model._reserve() for _ in range(dp+1)]
                self.assertEqual([t['rank'] for t in tickets], list(range(dp))+[0])
                queued = [pool.submit(model._generate_reserved, ticket, dict(self.request, token_ids=[i+3]))
                          for i, ticket in enumerate(tickets)]
                self.release(tp, 1)
                self.assertEqual(active[1].result(3)['dp_rank'], 1)
                self.wait_for(lambda: (self.fixture.model / f'begin-{tp}-2').exists())
                self.release(tp, 2)
                self.assertEqual(queued[1].result(3)['token_ids'], [4])
                self.assertFalse(queued[0].done())
                self.assertFalse(queued[-1].done())
                self.assertEqual(model.load_snapshot()[1]['running'], 0)
                self.assertEqual(model.load_snapshot()[0]['waiting'], 2)
                for rank in range(dp):
                    self.release(rank*tp, 1)
                    self.release(rank*tp, 2)
                self.release(0, 3)
                for i, future in enumerate(queued):
                    report = future.result(3)
                    self.assertEqual(report['token_ids'], [i+3])
                    self.assertEqual(report['dp_rank'], tickets[i]['rank'])
                    self.assertEqual(report['load_samples'][-1]['kv_cache_usage'], 0)
                for future in active:
                    self.assertEqual(future.result(3)['token_ids'], self.request['token_ids'][:1])

    def test_waiter_cancel_timeout_and_running_cancel_release_once(self):
        with self.running() as (model, pool, active, cancels):
            cancel = threading.Event()
            ticket = model._reserve(cancel)
            future = pool.submit(model._generate_reserved, ticket, self.request, cancel_event=cancel)
            cancel.set()
            with self.assertRaises(GenerationCancelled):
                future.result(3)
            ticket = model._reserve(deadline=time.monotonic()+0.05)
            with self.assertRaises(TimeoutError):
                model._generate_reserved(ticket, self.request)
            self.assertEqual([s['waiting'] for s in model.load_snapshot()], [0, 0])
            cancels[0].set()
            with self.assertRaises(GenerationCancelled):
                active[0].result(3)
            self.assertEqual(model.load_snapshot()[0]['running'], 0)
            self.assertEqual(model.load_snapshot()[0]['kv_cache_usage'], 0)
            self.assertEqual(model.state, 'ready')

    def test_http_admission_queues_limits_and_cancels(self):
        model = make_normal_llm_model(self.fixture.args())
        (self.fixture.root / 'http').mkdir()
        service = ChatService(model, model.hf_tokenizer, model.model_path, 'test', self.fixture.root / 'http',
                              queue_size=2, request_timeout=5)
        payload = dict(model='test', messages=[dict(role='user', content='hi')], max_tokens=1)
        try:
            active = []
            for rank in range(2):
                active.append(service.submit(payload, f'active-{rank}'))
                self.wait_for(lambda rank=rank: (self.fixture.model / f'begin-{rank}-1').exists())
            a, b = service.submit(payload, 'a'), service.submit(payload, 'b')
            self.assertEqual((a.dp_rank, b.dp_rank), (0, 1))
            self.assertEqual([s['waiting'] for s in service.health()['replica_load']], [1, 1])
            with self.assertRaises(HTTPError) as caught:
                service.submit(payload, 'full')
            self.assertEqual(caught.exception.status, 429)
            self.assertTrue(service.cancel('a'))
            self.assertTrue(a.done.wait(2))
            self.assertFalse(service.cancel('a'))
            self.assertEqual([s['waiting'] for s in model.load_snapshot()], [0, 1])
            for rank in range(2):
                self.release(rank, 1)
            self.release(1, 2)
            for job in [*active, b]:
                self.assertTrue(job.done.wait(3))
                self.assertEqual(job.status, 200)
                self.assertEqual(job.response['choices'][0]['message']['content'], '答')
        finally:
            service.close()

    def test_http_deadline_removes_queued_and_running_requests(self):
        model = make_normal_llm_model(self.fixture.args())
        (self.fixture.root / 'http-timeout').mkdir()
        service = ChatService(model, model.hf_tokenizer, model.model_path, 'test', self.fixture.root / 'http-timeout',
                              queue_size=4, request_timeout=0.3)
        payload = dict(model='test', messages=[dict(role='user', content='hi')], max_tokens=1)
        try:
            jobs = [service.submit(payload, f'timeout-{i}') for i in range(4)]
            for job in jobs:
                self.assertTrue(job.done.wait(3))
                self.assertEqual(job.status, 504)
            self.wait_for(lambda: not model._busy)
            self.assertEqual([s['waiting'] for s in model.load_snapshot()], [0, 0])
            self.assertEqual(model.state, 'ready')
        finally:
            service.close()

    def test_missing_native_capability_requires_explicit_legacy_policy(self):
        self.fixture.rename_model('load_legacy')
        with self.assertRaisesRegex(RuntimeError, 'load_stats=1'):
            make_normal_llm_model(self.fixture.args())
        with make_normal_llm_model(self.fixture.args('--dp-routing', 'round_robin')) as model:
            self.assertEqual(model.generate(self.request)['token_ids'], self.request['token_ids'][:1])
        for flags in (('--dp', '1'), ('--attention-dp', '2', '--ep', '2')):
            with self.assertRaises(ValueError):
                make_normal_llm_model(self.fixture.args(*flags, '--dp-routing', 'load'))

    def test_clear_pauses_new_execution_and_retains_queued_ticket(self):
        with self.running() as (model, pool, active, cancels):
            ticket = model._reserve()
            queued = pool.submit(model._generate_reserved, ticket, self.request)
            cleared = pool.submit(model.clear_prefix_cache)
            self.wait_for(lambda: model._maintenance)
            cancels[0].set()
            with self.assertRaises(GenerationCancelled):
                active[0].result(3)
            self.assertEqual(ticket['state'], 'queued')
            self.assertFalse(queued.done())
            self.assertFalse(cleared.done())
            cancels[1].set()
            with self.assertRaises(GenerationCancelled):
                active[1].result(3)
            self.assertEqual(len(cleared.result(3)), 2)
            self.release(0, 3)  # 原生编号2属于clear_prefix_cache。
            self.assertEqual(queued.result(3)['token_ids'], self.request['token_ids'][:1])

    def test_worker_failure_wakes_waiters_and_records_abandoned_admission(self):
        with self.running() as (model, pool, active, _):
            ticket = model._reserve()
            abandoned = model._reserve()
            queued = pool.submit(model._generate_reserved, ticket, self.request)
            os.kill(model.replicas[0].ready['workers'][0]['pid'], signal.SIGKILL)
            with self.assertRaises(RuntimeError):
                active[0].result(3)
            with self.assertRaises(RuntimeError):
                queued.result(3)
            self.assertEqual(model.state, 'failed')
            self.assertEqual(abandoned['state'], 'cancelled')
            record = json.loads((model.output / f'dispatch-{abandoned["id"]:06d}.json').read_text())
            self.assertEqual(record['status'], 'FAIL')
            self.assertEqual(record['dp_rank'], abandoned['rank'])
            with self.assertRaises(RuntimeError):
                model.generate(self.request)

    def test_close_with_paused_stream_releases_accounting(self):
        self.fixture.rename_model('dp_smoke')
        with make_normal_llm_model(self.fixture.args()) as model:
            stream = model.stream_response('hi', max_length=128)
            try:
                self.assertEqual(next(stream), '答')
                model.close()
                self.assertTrue(all(s['waiting'] == s['running'] == 0 for s in model.load_snapshot()))
            finally:
                stream.close()


class LoadProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture = sessions.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_nonstreaming_receives_actual_load_and_terminal_cleanup(self):
        with self.fixture.session() as session:
            observed = []
            session.load_observer = observed.append
            result = session.generate(self.fixture.request)
            self.assertEqual(result['token_ids'], [1])
            self.assertEqual([s['kv_cache_usage'] for s in observed], [0.75, 0.0])
            self.assertEqual([s['phase'] for s in result['load_samples']], ['running', 'complete'])

    def test_reject_malformed_or_missing_load_events(self):
        for mode in ('load_bad_id', 'load_bad_seq', 'load_bad_pages', 'load_bad_usage',
                     'load_dirty_final', 'load_missing_final', 'load_duplicate_final'):
            with self.subTest(mode=mode), self.fixture.session(mode) as session:
                session.load_observer = lambda _: None
                with self.assertRaises(RuntimeError):
                    session.generate(self.fixture.request)
                self.assertEqual(session.state, 'failed')


if __name__ == '__main__':
    unittest.main()
