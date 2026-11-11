"""正式工厂的多副本生命周期与并发回归；只替换 CUDA 枚举、tokenizer 和原生协议。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import unittest
from unittest import mock

from test_parallel_config import inventory
import test_process_tp_entrypoint as entry_tests
from fastllm_pytools import cli, parallel_config, process_tp
from fastllm_pytools.process_dp import ProcessDPModel
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser
from fastllm_pytools.qwen3_process_session import GenerationCancelled, GenerationObserverError
from fastllm_pytools.qwen3_process_server import ChatService, HTTPError


class ProcessDPTests(unittest.TestCase):
    def setUp(self):
        fixture = entry_tests.ProcessEntryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root, self.model = fixture, fixture.root, fixture.model
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='3,2,1,0'))
        stack.enter_context(mock.patch.object(parallel_config, 'cuda_inventory', return_value=inventory((3, 2, 1, 0))))
        self.request = dict(token_ids=[1, 2], max_new_tokens=1, eos_token_ids=[],
                            temperature=0.0, top_k=0, top_p=1.0, seed=0)

    def args(self, *flags):
        return self.fixture.args(['--dp', '2', '--tp', '1', '--tp-timeout', '5', *flags])

    def rename_model(self, mode):
        self.model = self.fixture.model.rename(self.root / mode)
        self.fixture.model = self.model
        self.fixture.base[0] = str(self.model)

    def assert_reaped(self, workers):
        for worker in workers:
            self.assertFalse(Path(f"/proc/{worker['pid']}").exists(), worker)

    def test_pure_dp_defaults_and_tp_replica_groups(self):
        for dp, tp in ((2, ''), (2, '1'), (2, '2'), (4, '1')):
            with self.subTest(dp=dp, tp=tp), make_normal_llm_model(self.args('--dp', str(dp), '--tp', tp)) as model:
                self.assertIsInstance(model, ProcessDPModel)
                size = int(tp or 1)
                self.assertEqual([r.session.devices for r in model.replicas],
                                 [list(range(rank * size, (rank + 1) * size)) for rank in range(dp)])
                workers = model.ready['workers']
                self.assertEqual(len({w['pid'] for w in workers}), dp * size)
                self.assertEqual(model.ready['model_load_count'], dp)
                for i in range(2 * dp):
                    result = model.generate(dict(self.request, token_ids=[i + 1]))
                    self.assertEqual((result['dp_rank'], result['dp_request_id'], result['token_ids']),
                                     (i % dp, i + 1, [i + 1]))
                self.assertEqual(model.ready['workers'], workers)
            self.assertEqual(model.state, 'closed')
            model.close()
            self.assert_reaped(workers)

    def test_device_environment_is_snapshotted_without_mutating_parent(self):
        with make_normal_llm_model(self.args()) as model:
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '3,2,1,0')
            self.assertEqual([r.ready['visible_devices'] for r in model.replicas], ['3,2,1,0'] * 2)
            self.assertEqual([r.devices for r in model.replicas], ['0', '1'])

    def test_actual_requests_overlap_on_two_processes(self):
        self.rename_model('dp_overlap')
        with make_normal_llm_model(self.args()) as model, ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(model.generate, dict(self.request, token_ids=[i])) for i in (3, 4)]
            results = [f.result(timeout=6) for f in futures]
            self.assertEqual([r['token_ids'] for r in results], [[3], [4]])
            self.assertEqual({r['dp_rank'] for r in results}, {0, 1})

    def test_text_calls_can_overlap_and_cli_uses_dp_factory(self):
        self.rename_model('dp_overlap')
        with make_normal_llm_model(self.args()) as model, ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(model.response, '你好', max_length=1) for _ in range(2)]
            self.assertEqual([f.result(timeout=6) for f in futures], ['答', '答'])
        with mock.patch.object(sys, 'argv', ['ftllm', 'run', *self.fixture.base, '--dp', '2', '--tp', '1',
                                             '--prompt', '你好', '--max-new-tokens', '1']), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            cli.main()
        self.assertEqual(output.getvalue(), '答\n')

    def test_cancelled_replica_is_reusable_and_does_not_cancel_other(self):
        with make_normal_llm_model(self.args()) as model, ThreadPoolExecutor(2) as pool:
            event, emitted = threading.Event(), threading.Event()
            future = pool.submit(model.generate, dict(self.request, max_new_tokens=16),
                                 cancel_event=event, on_token=lambda _: emitted.set())
            self.assertTrue(emitted.wait(3))
            self.assertEqual(model.generate(self.request)['dp_rank'], 1)
            event.set()
            with self.assertRaises(GenerationCancelled):
                future.result(timeout=3)
            self.assertEqual(model.generate(self.request)['dp_rank'], 0)
            self.assertEqual(model.state, 'ready')

    def test_waiter_cancel_and_shutdown_release_all_callers(self):
        model = make_normal_llm_model(self.args())
        self.addCleanup(model.close)
        workers = model.ready['workers']
        with ThreadPoolExecutor(3) as pool:
            started = [threading.Event(), threading.Event()]
            futures = [pool.submit(model.generate, dict(self.request, max_new_tokens=16),
                                   on_token=lambda _, i=i: started[i].set()) for i in range(2)]
            self.assertTrue(all(event.wait(3) for event in started))
            cancel = threading.Event()
            waiting = pool.submit(model.generate, self.request, cancel_event=cancel)
            cancel.set()
            with self.assertRaises(GenerationCancelled):
                waiting.result(timeout=2)
            model.close()
            for future in futures:
                with self.assertRaises(GenerationCancelled):
                    future.result(timeout=2)
        self.assert_reaped(workers)

    def test_startup_failure_rolls_back_earlier_replica(self):
        self.rename_model('dp_startup_failure')
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            make_normal_llm_model(self.args())
        launches = [json.loads(p.read_text()) for p in self.model.glob('launch-*.json')]
        self.assertEqual(len(launches), 2)
        self.assert_reaped(launches)
        output = next((self.root / 'output').iterdir())
        self.assertTrue((output / 'startup.failure.json').exists())
        self.assertTrue(next((output / 'replica-0').glob('*/session/shutdown.json')).is_file())

    def test_worker_failure_blocks_new_dispatch_without_retry(self):
        with make_normal_llm_model(self.args()) as model:
            os.kill(model.replicas[0].session._process.pid, signal.SIGKILL)
            with self.assertRaises((RuntimeError, OSError)):
                model.generate(self.request)
            self.assertEqual(model.state, 'failed')
            with self.assertRaisesRegex(RuntimeError, 'DP pool is failed'):
                model.generate(self.request)
            self.assertEqual(model.replicas[1].session._next_id, 1)

    def test_cli_termination_during_startup_reaps_and_restores_handler(self):
        from fastllm_pytools import process_dp
        original, created = process_dp.ProcessTPModel, []
        previous = signal.getsignal(signal.SIGTERM)
        def start(*args, **kwargs):
            if created:
                handler = signal.getsignal(signal.SIGTERM)
                self.assertTrue(callable(handler))
                handler(signal.SIGTERM, None)
            replica = original(*args, **kwargs)
            created.append(replica)
            return replica
        with mock.patch.object(process_dp, 'ProcessTPModel', side_effect=start), \
                mock.patch.object(sys, 'argv', ['ftllm', 'run', *self.fixture.base, '--dp', '2', '--tp', '1']), \
                self.assertRaises(SystemExit) as error:
            cli.main()
        self.assertEqual(error.exception.code, 128 + signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(created[0].state, 'closed')
        self.assert_reaped(created[0].ready['workers'])

    def test_invalid_request_and_observer_failure_preserve_replicas(self):
        with make_normal_llm_model(self.args()) as model:
            with self.assertRaises(ValueError):
                model.generate(dict(self.request, token_ids=[99]))
            with self.assertRaises(GenerationObserverError):
                model.generate(self.request, on_token=lambda _: (_ for _ in ()).throw(ValueError('observer')))
            self.assertEqual(model.generate(self.request)['token_ids'], [1])
            self.assertEqual(model.state, 'ready')

    def test_stream_close_releases_replica_and_resources(self):
        with make_normal_llm_model(self.args()) as model:
            stream = model.stream_response('你好', max_length=16)
            self.assertEqual(next(stream), '答')
            stream.close()
            self.assertFalse(model._busy)
            self.assertEqual(model.response('继续', max_length=1), '答')

    def test_model_close_does_not_wait_for_a_paused_text_iterator(self):
        model = make_normal_llm_model(self.args())
        self.addCleanup(model.close)
        stream = model.stream_response('你好', max_length=16)
        try:
            self.assertEqual(next(stream), '答')
            model.close()
            self.assertEqual(model.state, 'closed')
            self.assert_reaped(model.ready['workers'])
        finally:
            stream.close()

    def test_graph_prefix_and_custom_allreduce_stay_replica_local(self):
        with make_normal_llm_model(self.args('--tp', '2', '--cuda-graph', '--custom-allreduce',
                                            '--prefix-cache', 'true')) as model:
            for _ in range(4):
                model.generate(self.request)
            for replica in model.replicas:
                self.assertEqual(replica.session._graph_state[0]['prefix_cache']['hits'], 1)
                self.assertIn('--custom-allreduce 1', (replica.output / 'session/command.txt').read_text())
            model.clear_prefix_cache()
            self.assertTrue(all(r.session._graph_state[0]['prefix_cache']['entries'] == 0 for r in model.replicas))

    def test_json_uses_same_factory_and_rejects_unsupported_before_launch(self):
        config = self.root / 'options.json'
        config.write_text(json.dumps(vars(self.args())))
        with make_normal_llm_model(make_normal_parser('test').parse_args([str(config)])) as model:
            self.assertEqual(model.dp_size, 2)
        with mock.patch.object(process_tp, 'Qwen3ProcessSession') as native:
            with self.assertRaisesRegex(ValueError, 'does not support dtype'):
                make_normal_llm_model(self.args('--dtype', 'int4'))
            native.assert_not_called()

    def test_http_consumers_overlap_and_preserve_request_identity(self):
        self.rename_model('dp_overlap')
        model = make_normal_llm_model(self.args())
        self.addCleanup(model.close)
        service = ChatService(model, model.hf_tokenizer, self.model, 'qwen3', model.output,
                              queue_size=4, request_timeout=4)
        self.addCleanup(service.close)
        jobs = [service.submit(dict(model='qwen3', messages=[dict(role='user', content='你好')], max_tokens=1),
                               request_id=f'request-{i}') for i in range(2)]
        for job in jobs:
            self.assertTrue(job.done.wait(6))
            self.assertEqual(job.status, 200, job.response)
            self.assertEqual(job.response['id'], 'chatcmpl-' + job.id)
        self.assertEqual({j.dp_rank for j in jobs}, {0, 1})
        self.assertEqual({j.dp_request_id for j in jobs}, {1, 2})
        self.assertEqual(service.health()['model_load_count'], 2)

    def test_http_cancel_queue_and_shutdown_multiple_active_requests(self):
        model = make_normal_llm_model(self.args())
        self.addCleanup(model.close)
        service = ChatService(model, model.hf_tokenizer, self.model, 'qwen3', model.output,
                              queue_size=2, request_timeout=4)
        self.addCleanup(service.close)
        payload = dict(model='qwen3', messages=[dict(role='user', content='你好')], max_tokens=16)
        jobs = [service.submit(payload, request_id=f'active-{i}') for i in range(2)]
        deadline = time.monotonic() + 3
        while len(service.health()['active_request_ids']) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(service.health()['active_request_ids']), 2)
        with self.assertRaises(HTTPError) as error:
            service.submit(payload, request_id='active-0')
        self.assertEqual(error.exception.status, 409)
        queued = service.submit(payload, request_id='queued')
        service.cancel('queued')
        self.assertTrue(queued.done.wait(1))
        self.assertEqual(queued.status, 499)
        service.close()
        self.assertTrue(all(job.done.is_set() and job.status == 503 for job in jobs))
        self.assert_reaped(model.ready['workers'])

    def test_server_acceptance_reports_success_and_partial_failure(self):
        import run_qwen3_dp as runner
        self.rename_model('dp_smoke')
        cache = self.root / 'CMakeCache.txt'
        cache.write_text('CUDA_ARCH:STRING=70\n')
        argv = ['run_qwen3_dp', '--model', str(self.model), '--generate', str(self.fixture.fixture.binary),
                '--worker', str(self.fixture.fixture.binary), '--build-cache', str(cache),
                '--output', str(self.root / 'acceptance')]

        for flags in ([], ['--tp', '2', '--cuda-graph', '--custom-allreduce', '--prefix-cache']):
            with self.subTest(flags=flags), mock.patch.object(sys, 'argv', argv + flags), \
                    mock.patch.object(runner, 'capture', return_value='fixture, no GPU or compiler'), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(runner, 'capture', return_value='fixture'), \
                mock.patch.object(runner, 'make_normal_llm_model', side_effect=RuntimeError('startup failed')), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
        reports = [json.loads(p.read_text()) for p in (self.root / 'acceptance').glob('*/summary.json')]
        self.assertEqual(sum(s['status'] == 'PASS' and s['completed_case_count'] == 10 for s in reports), 2)
        failed = next(s for s in reports if s['status'] == 'FAIL')
        self.assertFalse(failed['complete'])
        self.assertIn('startup failed', failed['cases'][0]['reason'])
        for path in (self.root / 'acceptance').iterdir():
            for suffix in ('json', 'md', 'csv', 'xlsx'):
                self.assertTrue((path / f'summary.{suffix}').exists())


if __name__ == '__main__':
    unittest.main()
