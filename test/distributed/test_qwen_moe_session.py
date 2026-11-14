"""联合调度的协议模拟测试：不加载 CUDA、不编译；GPU 数值另由服务器验收。"""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import sys
import threading
import time
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import test_parallel_config  # 设置 tools 导入路径。
from fastllm_pytools import qwen_moe_process_session as moe


class FakeTransport:
    def __init__(self, binary, worker, model, devices, output, **options):
        self.devices = list(map(int, devices.split(',')))
        self.graph = options['cuda_graph']
        self.history, self.fail, self.delay = [], False, 0.003
        self.states = [dict(request_active=False, request_tokens=0, request_capacity=0, request_kv_bytes=0,
                            reserved_kv_bytes=0, workspace_bytes=0, captures=0, replays=0,
                            prefix_cache=dict(snapshots=[], entries=0, device_bytes=0)) for _ in self.devices]
        self.ready = dict(backend=moe.BACKEND, states=self.states, workers=[],
                          cancellation='round_boundary', token_events=True)
        self.current = [0] * len(self.devices)

    def __enter__(self):
        return self

    def exchange(self, op, **fields):
        if self.fail:
            raise RuntimeError('injected worker failure')
        time.sleep(self.delay)
        if op == 'clear_prefix_cache':
            assert not any(s['request_active'] for s in self.states)
            for s in self.states:
                s['prefix_cache'] = dict(snapshots=[], entries=0, device_bytes=0)
            return dict(states=copy.deepcopy(self.states))
        steps = fields['steps']
        self.history.append(copy.deepcopy(steps))
        graph = self.graph and not any(s['action'] == 1 for s in steps) and any(s['action'] == 2 for s in steps)
        tokens = []
        for i, (step, state) in enumerate(zip(steps, self.states)):
            action = step['action']
            if action == 1:
                assert not state['request_active']
                self.current[i] = step['token_ids'][0]
                state.update(request_active=True, request_tokens=len(step['token_ids']),
                             request_capacity=step['capacity'], request_kv_bytes=step['capacity'] * 4)
                state['prefix_cache']['snapshots'] = [dict(position_begin=0, token_ids=step['token_ids'])]
            elif action == 2:
                assert state['request_active']
                state['request_tokens'] += 1
            elif action == 3:
                state.update(request_active=False, request_tokens=0, request_capacity=0, request_kv_bytes=0)
            else:
                assert not state['request_active']
            if graph:
                state['replays'] += 1
                state['captures'] = 1
            tokens.append(self.current[i] if action in (1, 2) else None)
        return dict(states=copy.deepcopy(self.states), token_ids=tokens,
                    logits=[[float(j) for j in range(13)] for _ in steps], round_ms=0.1,
                    execution='decode_cuda_graph' if graph else 'eager')

    def close(self):
        assert not any(s['request_active'] for s in self.states)
        self.shutdown = dict(shutdown_complete=True)

    def _dispose(self, state):
        self.disposed = state


def request(token=1, length=4):
    return dict(token_ids=[token, 2, 3], max_new_tokens=length, top_k=0, top_p=1.0,
                temperature=0.0, seed=0, eos_token_ids=[])


class MoeSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'config.json').write_text(json.dumps(dict(vocab_size=13)))
        self.patch = patch.object(moe, 'RoundTransport', FakeTransport)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def session(self, **options):
        return moe.QwenMoeProcessSession('/unused', '/unused', self.root, '0,1,2,3', self.root,
                                         timeout=3, request_log=False, **options)

    def test_concurrent_variable_lengths_and_graph_reuse(self):
        with self.session(cuda_graph=True, trace=True) as session, ThreadPoolExecutor(8) as pool:
            jobs = [pool.submit(session.generate, request(i + 1, i + 2)) for i in range(8)]
            results = [f.result(5) for f in jobs]
            for i, result in enumerate(results):
                self.assertEqual(result['token_ids'], [i + 1] * (i + 2))
                self.assertFalse(result['states'][result['dp_rank']]['request_active'])
            history = session._transport.history
            self.assertTrue(any(sum(s['action'] in (1, 2) for s in steps) > 1 for steps in history))
            self.assertTrue(any(any(s['action'] == 1 for s in row) and any(s['action'] == 2 for s in row) for row in history))
            self.assertTrue(all(s['captures'] == 1 for s in session._states))

    def test_eos_and_callback_cancel_do_not_poison_peers(self):
        with self.session() as session, ThreadPoolExecutor(2) as pool:
            eos = request(5, 40)
            eos['eos_token_ids'] = [5]
            self.assertEqual(session.generate(eos)['finish_reason'], 'eos')
            cancel = threading.Event()
            a = pool.submit(session.generate, request(2, 100), cancel_event=cancel, on_token=lambda _: cancel.set())
            b = pool.submit(session.generate, request(3, 6))
            with self.assertRaises(moe.GenerationCancelled):
                a.result(5)
            self.assertEqual(b.result(5)['token_ids'], [3]*6)
            self.assertEqual(session.generate(request())['status'], 'PASS')

    def test_observer_exception_is_request_local(self):
        with self.session() as session:
            def fail(_):
                raise ValueError('callback failed')
            with self.assertRaises(moe.GenerationObserverError):
                session.generate(request(length=30), on_token=fail)
            self.assertEqual(session.generate(request())['status'], 'PASS')

    def test_slow_consumer_cancels_without_stalling_other_rank(self):
        with self.session() as session, ThreadPoolExecutor(2) as pool:
            session._transport.delay = 0.001
            a = pool.submit(session.generate, request(length=200), on_token=lambda _: time.sleep(0.01))
            b = pool.submit(session.generate, request(2, 4))
            self.assertEqual(b.result(2)['status'], 'PASS')
            with self.assertRaises(moe.GenerationCancelled):
                a.result(5)

    def test_queued_cancel_and_invalid_request_never_enter_native(self):
        with self.session() as session:
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(moe.GenerationCancelled):
                session.generate(request(), cancel_event=cancel)
            with self.assertRaises(ValueError):
                session.generate(request(99))
            with self.assertRaises(ValueError):
                session.generate(request(length=2048))
            self.assertFalse(session._transport.history)

    def test_prefix_route_and_drain_clear(self):
        with self.session(prefix_cache=True, routing='prefix') as session:
            a = session.generate(request(3))
            b = session.generate(request(3))
            self.assertEqual(a['dp_rank'], b['dp_rank'])
            with ThreadPoolExecutor(2) as pool:
                future = pool.submit(session.generate, request(4, 12))
                session.clear_prefix_cache()
                future.result(5)
            session.clear_prefix_cache()
            self.assertTrue(all(not s['prefix_cache']['snapshots'] for s in session._states))

    def test_group_failure_unblocks_all_and_forbids_reuse(self):
        session = self.session().__enter__()
        session._transport.fail = True
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(session.generate, request()) for _ in range(4)]
            for future in futures:
                with self.assertRaises(RuntimeError):
                    future.result(5)
        self.assertEqual(session.state, 'failed')
        self.assertEqual(session._transport.disposed, 'failed')
        with self.assertRaises(RuntimeError):
            session.close()

    def test_shutdown_cancels_active_requests(self):
        session = self.session().__enter__()
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(session.generate, request(length=1000))
            time.sleep(0.02)
            session.close()
            with self.assertRaises(moe.GenerationCancelled):
                future.result(5)
        self.assertEqual(session.state, 'closed')

    def test_teacher_forcing_changes_inputs_only_and_requires_trace(self):
        with self.session(trace=True) as session:
            result = session.generate(request(1, 3), teacher_forced_tokens=[7, 8])
            self.assertEqual(result['token_ids'], [1, 1, 1])
            rank = result['dp_rank']
            inputs = [row[rank]['token_ids'] for row in session._transport.history if row[rank]['action'] == 2]
            self.assertEqual(inputs, [[7], [8]])
        with self.session() as session, self.assertRaises(ValueError):
            session.generate(request(1, 3), teacher_forced_tokens=[7, 8])

    def test_deadline_releases_slot_and_preserves_group(self):
        with self.session() as session:
            session.timeout = 0.001
            with self.assertRaises(moe.GenerationCancelled):
                session.generate(request(length=30))
            session.timeout = 3
            self.assertEqual(session.generate(request())['status'], 'PASS')

    def test_factory_text_and_service_use_joint_session(self):
        from fastllm_pytools.util import make_normal_parser, make_normal_llm_model
        from fastllm_pytools.qwen3_process_server import ChatService
        from test_parallel_config import inventory

        (self.root / 'config.json').write_text(json.dumps(dict(model_type='qwen2_moe', vocab_size=13)))
        tokenizer = Mock(chat_template='template', eos_token_id=12)
        tokenizer.apply_chat_template.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **_: '答' * len(ids)
        transformers = Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        args = make_normal_parser('moe factory').parse_args([str(self.root), '--tp-backend', 'process',
            '--attention-dp', '4', '--ep', '4', '--tp', '1', '--tp-output', str(self.root / 'output'),
            '--tp-timeout', '3', '--tp-request-timeout', '2', '--tp-queue-size', '1', '--tp-request-log', 'off'])
        with patch.dict(sys.modules, {'transformers': transformers}), \
                patch('fastllm_pytools.parallel_config.cuda_inventory', return_value=inventory()), \
                patch('fastllm_pytools.process_tp.native_binary', return_value=Path(sys.executable)), \
                patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1,2,3'}):
            with make_normal_llm_model(args) as model, ThreadPoolExecutor(4) as pool:
                self.assertEqual(model.concurrency, 4)
                results = list(pool.map(lambda _: model.response('你好', max_length=6), range(4)))
                self.assertEqual(results, ['答'*6]*4)
                self.assertEqual(''.join(model.stream_response('你好', max_length=4)), '答'*4)
                service = ChatService(model, tokenizer, self.root, 'moe', model.output, queue_size=1, request_timeout=2)
                try:
                    self.assertEqual(service.health()['backend'], moe.BACKEND)
                    self.assertEqual(service.health()['ep_size'], 4)
                    job = service.submit(dict(model='moe', messages=[dict(role='user', content='你好')], max_tokens=4))
                    self.assertTrue(job.done.wait(3))
                    self.assertEqual(job.status, 200)
                    self.assertEqual(job.response['choices'][0]['message']['content'], '答'*4)
                finally:
                    service.close()


if __name__ == '__main__':
    unittest.main()
