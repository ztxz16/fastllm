"""Attention DP/EP 联合轮次会话；一个控制线程驱动全部 rank，用户回调独立消费。"""

from collections import deque
from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
import queue
import shlex
import socket
import threading
import time

from .qwen3_process_chat import validate_request
from .qwen3_process_session import (Qwen3ProcessSession, GenerationCancelled,
                                    GenerationObserverError)
from .tp_process_utils import isolated_process, wait_case

BACKEND = 'qwen_moe_attention_dp_ep'


class RoundTransport(Qwen3ProcessSession):
    """/** 复用有界帧与进程组回收；取消在联合轮次边界执行，不发送 dense TP 数据报。 */"""

    def _send_cancel(self):
        pass

    def __enter__(self):
        self._stack = ExitStack()
        self.state = 'starting'
        try:
            self.output.mkdir(parents=True, exist_ok=False)
            self._socket, child = socket.socketpair()
            self._stack.callback(self._socket.close)
            self._stack.callback(child.close)
            command = [str(self.binary), '--worker', str(self.worker), '--model', str(self.model),
                       '--devices', ','.join(map(str, self.devices)), '--session-fd', str(child.fileno()),
                       '--cuda-graph', str(int(self.cuda_graph)), '--prefix-cache', str(int(self.prefix_cache)),
                       '--timeout-ms', str(max(1, int(self.timeout * 1000)))]
            (self.output / 'command.txt').write_text(shlex.join(command) + '\n')
            stdout = self._stack.enter_context((self.output / 'native.stdout.log').open('wb'))
            stderr = self._stack.enter_context((self.output / 'native.stderr.log').open('wb'))
            self._process = self._stack.enter_context(isolated_process(
                command, pass_fds=(child.fileno(),), stdout=stdout, stderr=stderr, env=self.environment))
            child.close()
            self.ready = self._read(time.monotonic() + self.timeout)
            workers = self.ready.get('workers', [])
            if (self.ready.get('status') != 'PASS' or self.ready.get('event') != 'ready' or
                    self.ready.get('backend') != BACKEND or self.ready.get('protocol') != 1 or
                    self.ready.get('model_load_count') != 1 or len(workers) != len(self.devices) or
                    any(w.get('rank') != i or w.get('device') != d or type(w.get('pid')) is not int or w['pid'] <= 0
                        for i, (w, d) in enumerate(zip(workers, self.devices))) or
                    len({w['pid'] for w in workers}) != len(workers)):
                raise RuntimeError('invalid Attention DP/EP startup response')
            (self.output / 'ready.json').write_text(json.dumps(self.ready, indent=2))
            self.state = 'ready'
            return self
        except BaseException:
            self._dispose('failed')
            raise

    def exchange(self, op, **fields):
        """/** 仅调度线程调用；协议失败后由会话关闭全部 worker。 */"""
        result = self._exchange(dict(id=self._next_id, op=op, **fields))
        if result.get('status') != 'PASS':
            raise RuntimeError(f'MoE round failed: {result}')
        return result

    def close(self):
        """/** 正常关闭要求原生卸载确认、所有预留归零和进程退出；失败执行整组回收。 */"""
        try:
            self.shutdown = self.exchange('shutdown')
            if self.shutdown.get('shutdown_complete') is not True or any(
                    s['request_active'] or s['reserved_kv_bytes'] or s['workspace_bytes'] or
                    s['prefix_cache']['device_bytes'] for s in self.shutdown['states']):
                raise RuntimeError('MoE shutdown leaked device resources')
            if wait_case(self._process, self.timeout) != 0:
                raise RuntimeError('MoE coordinator exited with failure')
            (self.output / 'shutdown.json').write_text(json.dumps(self.shutdown, indent=2))
        except BaseException:
            self._dispose('failed')
            raise
        self._dispose('closed')


class QwenMoeProcessSession:
    """/** 每 rank 一个请求，所有专家协同执行；队列、回调及取消不阻塞其他 rank。 */"""

    def __init__(self, binary, worker, model, devices, output, *, timeout=1800, trace=False,
                 cuda_graph=False, prefix_cache=False, routing='round_robin', queue_size=8,
                 request_log=True, environment=None):
        if routing not in ('round_robin', 'prefix') or (routing == 'prefix' and not prefix_cache):
            raise ValueError('prefix routing requires prefix cache')
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError('queue_size must be positive')
        self._transport = RoundTransport(binary, worker, model, devices, output, timeout=timeout,
            cuda_graph=cuda_graph, prefix_cache=prefix_cache, request_log=request_log, environment=environment)
        self.concurrency = len(self._transport.devices)
        if self.concurrency not in (1, 2, 4):
            raise ValueError('Attention DP/EP requires 1, 2 or 4 ranks')
        self.config = json.loads((Path(model) / 'config.json').read_text())
        self.output, self.timeout, self.trace = Path(output), timeout, trace
        self.cuda_graph, self.prefix_cache = cuda_graph, prefix_cache
        self.routing, self.queue_size, self.request_log = routing, queue_size, request_log
        self._condition = threading.Condition()
        self._pending, self._slots = deque(), [None] * self.concurrency
        self._stopping = self._clear = False
        self._cursor, self._job_id = 0, 0
        self.state, self.failure, self.shutdown = 'new', None, None

    def __enter__(self):
        if self.state != 'new':
            raise RuntimeError('MoE session cannot restart')
        self._transport.__enter__()
        self.ready = copy.deepcopy(self._transport.ready)
        self._states = self.ready['states']
        self.state = 'ready'
        self._thread = threading.Thread(target=self._run, name='attention-ep-rounds')
        try:
            self._thread.start()
        except BaseException:
            self._transport._dispose('failed')
            self.state = 'failed'
            raise
        return self

    def generate(self, request, *, cancel_event=None, on_token=None, teacher_forced_tokens=None):
        """/**
         * 入队并在调用线程消费token；完成通知仅在rank已释放后发出。
         * @param teacher_forced_tokens 仅trace诊断允许的后续输入ID，长度为max_new_tokens-1。
         *        它替换下一步输入而不替换采样结果，复用正式KV/算子/通信路径；HTTP不暴露此参数。
         * @return 实际采样结果和可选logits；教师强制报告同时保留真正使用的输入历史。
         */"""
        request = json.loads(json.dumps(request, allow_nan=False))
        validate_request(request, self.config['vocab_size'])
        if teacher_forced_tokens is not None:
            if (not self.trace or request['eos_token_ids'] or not isinstance(teacher_forced_tokens, list) or
                    len(teacher_forced_tokens) != request['max_new_tokens'] - 1 or
                    any(type(x) is not int or not 0 <= x < self.config['vocab_size'] for x in teacher_forced_tokens)):
                raise ValueError('teacher forcing requires trace, no EOS and exactly max_new_tokens-1 valid IDs')
            teacher_forced_tokens = list(teacher_forced_tokens)
        if on_token is not None and not callable(on_token):
            raise ValueError('on_token must be callable')
        now = time.monotonic()
        job = dict(request=request, cancel=cancel_event if cancel_event is not None else threading.Event(),
                   done=threading.Event(), events=queue.Queue(64), tokens=[], steps=[], finish=None,
                   start=now, deadline=now + self.timeout, rank=None, error=None, observer=None,
                   forced=teacher_forced_tokens)
        with self._condition:
            if self.state != 'ready' or self._stopping:
                raise RuntimeError(f'MoE session is {self.state}: {self.failure}')
            if len(self._pending) + sum(j is not None for j in self._slots) >= self.queue_size + self.concurrency:
                raise ValueError('Attention DP/EP request queue is full')
            self._job_id += 1
            job['id'] = self._job_id
            self._pending.append(job)
            self._condition.notify_all()
        try:
            while not job['done'].is_set() or not job['events'].empty():
                try:
                    token = job['events'].get(timeout=0.05)
                except queue.Empty:
                    continue
                if on_token is not None and job['observer'] is None:
                    try:
                        on_token(token)
                    except Exception as error:
                        job['observer'] = error
                        job['cancel'].set()
            if job['observer'] is not None:
                raise GenerationObserverError(job['report'], job['observer'])
            if job['error'] is not None:
                raise job['error']
            return job['report']
        finally:
            job['cancel'].set()

    def _finish(self, job):
        """/** 保存请求级证据再通知调用方；失败报告仍保留已完成 token。 */"""
        report = dict(id=job['id'], dp_request_id=job['id'], dp_rank=job['rank'], backend=BACKEND,
                      request=job['request'], token_ids=job['tokens'], steps=job['steps'],
                      teacher_forced_tokens=job['forced'],
                      status='FAIL' if job['error'] else 'PASS', finish_reason=job['finish'],
                      states=copy.deepcopy(self._states), reusable=self.state == 'ready',
                      elapsed_ms=(time.monotonic() - job['start']) * 1000,
                      queue_ms=job.get('queue_ms', (time.monotonic() - job['start']) * 1000))
        if job['error'] is not None:
            report.update(error=str(job['error']), cancelled=job['cancel'].is_set())
            if isinstance(job['error'], GenerationCancelled):
                job['error'].report = report
        job['report'] = report
        try:
            if self.request_log:
                (self.output / f"request-{job['id']:06d}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        except BaseException as error:
            job['error'] = RuntimeError(f'cannot persist MoE request: {error}')
            report.update(status='FAIL', error=str(job['error']))
            raise
        finally:
            job['done'].set()

    def _cancelled(self, job):
        if job['cancel'].is_set() or self._stopping or time.monotonic() >= job['deadline']:
            job['cancel'].set()
            job['error'] = GenerationCancelled(dict(error='cancelled, shutting down or request deadline exceeded'))
            return True
        return False

    def _rank(self, free, request):
        """/** 仅依据原生目录选择空闲 rank；匹配长度相同时按轮转打散。 */"""
        order = sorted(free, key=lambda rank: (rank - self._cursor) % self.concurrency)
        def score(rank):
            best = 0
            for entry in self._states[rank]['prefix_cache']['snapshots']:
                if entry['position_begin'] != 0:
                    continue
                same = 0
                for a, b in zip(entry['token_ids'], request['token_ids'][:-1]):
                    if a != b:
                        break
                    same += 1
                best = max(best, same)
            return best
        rank = max(order, key=score) if self.routing == 'prefix' else order[0]
        self._cursor = (rank + 1) % self.concurrency
        return rank

    def _round(self):
        """/** 锁内构造准入/结束动作，锁外驱动整个通信组，最后核验并提交各 rank 输出。 */"""
        with self._condition:
            steps = [dict(action=0, token_ids=[]) for _ in self._slots]
            for rank, job in enumerate(self._slots):
                if job is None:
                    continue
                end = self._cancelled(job) or job['finish'] is not None
                token = None if end else (job['forced'][len(job['tokens'])-1] if job['forced'] is not None else job['tokens'][-1])
                steps[rank] = dict(action=3 if end else 2, token_ids=[] if end else [token])
            for job in list(self._pending):
                if self._cancelled(job):
                    self._pending.remove(job)
                    self._finish(job)
            free = [i for i, job in enumerate(self._slots) if job is None]
            while free and self._pending and not self._clear and not self._stopping:
                job = self._pending.popleft()
                rank = self._rank(free, job['request'])
                free.remove(rank)
                self._slots[rank] = job
                job['rank'], job['queue_ms'] = rank, (time.monotonic() - job['start']) * 1000
                r = job['request']
                steps[rank] = dict(action=1, token_ids=r['token_ids'], position_begin=0,
                                   capacity=len(r['token_ids']) + r['max_new_tokens'] - 1,
                                   sampling={k: r[k] for k in ('seed', 'top_k', 'top_p', 'temperature')})
        if not any(s['action'] for s in steps):
            return
        result = self._transport.exchange('step', steps=steps, trace=self.trace)
        states, tokens = result['states'], result['token_ids']
        if len(states) != self.concurrency or len(tokens) != self.concurrency:
            raise RuntimeError('MoE round rank count mismatch')
        graph = self.cuda_graph and not any(s['action'] == 1 for s in steps) and any(s['action'] == 2 for s in steps)
        if result['execution'] != ('decode_cuda_graph' if graph else 'eager'):
            raise RuntimeError('MoE round silently changed execution backend')
        with self._condition:
            previous, self._states = self._states, states
            for rank, step in enumerate(steps):
                active = step['action'] in (1, 2)
                state, token = states[rank], tokens[rank]
                if (state['request_active'] != active or (active and (type(token) is not int or
                        not 0 <= token < self.config['vocab_size'])) or (not active and token is not None) or
                        state['replays'] - previous[rank]['replays'] != int(graph)):
                    raise RuntimeError(f'invalid MoE rank {rank} output or Graph evidence')
                job = self._slots[rank]
                if step['action'] == 3:
                    if state['request_kv_bytes'] or state['request_tokens'] or state['request_capacity']:
                        raise RuntimeError('MoE finished rank still has active KV')
                    self._finish(job)
                    self._slots[rank] = None
                elif active:
                    job['tokens'].append(token)
                    item = dict(token_id=token, round_ms=result['round_ms'], execution=result['execution'],
                                elapsed_ms=(time.monotonic() - job['start']) * 1000)
                    if self.trace:
                        values = result['logits'][rank]
                        if len(values) != self.config['vocab_size'] or not all(math.isfinite(x) for x in values):
                            raise RuntimeError('invalid MoE trace logits')
                        item['logits'] = values
                    job['steps'].append(item)
                    try:
                        job['events'].put_nowait(token)
                    except queue.Full:
                        job['cancel'].set()
                    if token in job['request']['eos_token_ids']:
                        job['finish'] = 'eos'
                    elif len(job['tokens']) == job['request']['max_new_tokens']:
                        job['finish'] = 'length'

    def _run(self):
        """/** 单个 native/worker 故障使整组不可复用；唤醒所有等待者，关闭进程树。 */"""
        try:
            while True:
                with self._condition:
                    while not self._pending and not any(self._slots) and not self._clear and not self._stopping:
                        self._condition.wait()
                    if not any(self._slots) and self._clear:
                        result = self._transport.exchange('clear_prefix_cache')
                        self._states = result['states']
                        if any(s['prefix_cache']['entries'] or s['prefix_cache']['device_bytes'] for s in self._states):
                            raise RuntimeError('MoE prefix cache clear failed')
                        self._clear = False
                        self._condition.notify_all()
                    if self._stopping and not self._pending and not any(self._slots):
                        break
                self._round()
            self._transport.close()
            self.shutdown = self._transport.shutdown
            self.state = 'closed'
        except BaseException as error:
            self.failure, self.state = str(error), 'failed'
            try:
                self._transport._dispose('failed')
            except BaseException as cleanup:
                self.failure += f'; cleanup: {cleanup}'
            with self._condition:
                for job in list(self._pending) + [j for j in self._slots if j is not None]:
                    job['error'] = RuntimeError(self.failure)
                    try:
                        self._finish(job)
                    except BaseException as report_error:
                        job.setdefault('report', dict(status='FAIL', error=str(report_error)))
                    finally:
                        job['done'].set()
                self._pending.clear()
                self._slots = [None] * self.concurrency
        finally:
            with self._condition:
                self._condition.notify_all()

    def clear_prefix_cache(self):
        """/** 暂停准入，等待活动请求排空后清理；排队请求继续保留截止时间。 */"""
        with self._condition:
            if self.state != 'ready' or self._stopping:
                raise RuntimeError('MoE session is unavailable')
            self._clear = True
            self._condition.notify_all()
            if not self._condition.wait_for(lambda: not self._clear or self.state != 'ready', self.timeout):
                raise TimeoutError('prefix cache clear deadline exceeded')
            if self.state != 'ready':
                raise RuntimeError(self.failure)
            return copy.deepcopy(self._states)

    def close(self):
        """/** 停止准入并取消排队及活动请求；等待边界释放及原生退出。 */"""
        with self._condition:
            if self.state == 'new':
                self.state = 'closed'
                return
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(self.timeout + 40)
        if self._thread.is_alive():
            raise RuntimeError('MoE scheduler did not stop')
        if self.state == 'failed':
            raise RuntimeError(self.failure)

    def __exit__(self, *_):
        self.close()
