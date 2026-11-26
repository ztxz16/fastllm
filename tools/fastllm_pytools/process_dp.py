"""完整模型副本 DP：Python 只管理副本和请求，前向、采样及 TP 通信仍在 C++/CUDA。"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .parallel_config import build_parallel_plan
from .process_tp import ProcessTPModel, validate_options
from .qwen3_process_chat import validate_request
from .qwen3_process_session import BACKEND, GenerationCancelled


class ProcessDPModel:
    """/**
     * 管理互斥 GPU 上的完整模型副本；每个副本有独立协调器、TP worker 和 KV。
     * 全部副本就绪后才接受请求；三种策略都固定请求归属并在健康副本上排队。
     * 启动失败回收已启动副本；运行故障停止整个池的新分配，不自动重试。
     * 调用者须使用上下文管理器或 close；Token 回调不得重入生成或关闭模型。
     */"""

    tp_backend = 'process'
    backend = BACKEND

    def __init__(self, args):
        from .util import load_model_args

        load_model_args(args)
        validate_options(args, allow_dp=True)
        if getattr(args, 'dp', 1) <= 1:
            raise ValueError('ProcessDPModel requires DP>1; use the model factory for DP=1')
        self.plan = build_parallel_plan(args)
        self.dp_size = self.concurrency = self.plan['dp_size']
        self.tp_size = self.plan['tp_size']
        self.routing = self.plan['dp_routing']
        self.timeout = getattr(args, 'tp_timeout', 180)
        self.request_log = getattr(args, 'tp_request_log', 'on') == 'on'
        self.output = Path(getattr(args, 'tp_output', 'process-tp-results')).resolve() / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_dp_' + uuid.uuid4().hex[:8])
        self.output.mkdir(parents=True, exist_ok=False)
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._prepare_lock = threading.Lock()
        self._status, self._failure = 'starting', None
        self._maintenance = False
        self._busy, self._next_rank, self._next_id = set(), 0, 1
        self.replicas = []
        self._prefixes = [[] for _ in range(self.dp_size)]
        self._queues = [deque() for _ in range(self.dp_size)]
        self._reservations = {}
        self._loads = [None for _ in range(self.dp_size)]
        environment = dict(os.environ)
        environment.setdefault('USE_TORCH', '0')

        try:
            for assignment in self.plan['replicas']:
                replica_args = copy.deepcopy(args)
                replica_args.dp, replica_args.tp = 1, str(self.tp_size)
                replica_args.dp_routing = 'round_robin'
                replica_args.tp_output = str(self.output / f"replica-{assignment['dp_rank']}")
                devices = [device['logical_id'] for device in assignment['devices']]
                replica = ProcessTPModel(replica_args, _device_ids=devices, _environment=environment,
                                         _defer_startup=True)
                self.replicas.append(replica)
                replica.session.start()

            # 先让全部原生进程开始加载，再按固定 rank 收集 ready；完成顺序不改变设备映射。
            for replica in self.replicas:
                replica.session.wait_ready()
                if (replica.ready.get('cancellation') != 'request_id_datagram' or
                        replica.ready.get('token_events') is not True):
                    raise RuntimeError('all DP replicas require cancellation and token events')

            if self.routing == 'prefix':
                for rank in range(self.dp_size):
                    self._refresh_prefixes(rank)
            if self.routing == 'load':
                for rank, replica in enumerate(self.replicas):
                    if replica.session.load_state is None:
                        raise RuntimeError('load routing requires native load_stats=1; rebuild coordinator and worker')
                    self._loads[rank] = copy.deepcopy(replica.session.load_state)
                    replica.session.load_observer = lambda state, rank=rank: self._update_load(rank, state)

            first = self.replicas[0]
            self.model_path, self.config, self.hf_tokenizer = first.model_path, first.config, first.hf_tokenizer
            self.devices = [replica.devices for replica in self.replicas]
            self.ready = dict(event='ready', status='PASS', backend=BACKEND, dp_size=self.dp_size,
                              tp_size=self.tp_size, model_load_count=self.dp_size, dp_routing=self.routing,
                              cancellation='request_id_datagram', token_events=True,
                              workers=[dict(worker, dp_rank=rank) for rank, replica in enumerate(self.replicas)
                                       for worker in replica.ready['workers']],
                              replicas=[dict(dp_rank=rank, devices=replica.session.devices,
                                             output=str(replica.output), ready=replica.ready)
                                        for rank, replica in enumerate(self.replicas)])
            if len({worker['pid'] for worker in self.ready['workers']}) != self.dp_size * self.tp_size:
                raise RuntimeError('DP replicas share worker PIDs')
            (self.output / 'allocation.json').write_text(json.dumps(self.plan, ensure_ascii=False, indent=2))
            (self.output / 'replicas.ready.json').write_text(json.dumps(self.ready, ensure_ascii=False, indent=2))
            self._status = 'ready'
            args.path = str(self.model_path)
        except BaseException as error:
            try:
                self.close()
            finally:
                self._status = 'failed'
                (self.output / 'startup.failure.json').write_text(json.dumps(
                    dict(status='FAIL', error=f'{type(error).__name__}: {error}'), ensure_ascii=False))
            raise

    @property
    def state(self):
        with self._condition:
            if self._status == 'ready' and (self._failure is not None or
                    any(replica.state != 'ready' for replica in self.replicas)):
                return 'failed'
            return self._status

    def _refresh_prefixes(self, rank):
        """/** 只使用原生已确认的完整目录替换本地记录；取消仍可能保留有效Prompt快照。 */"""
        replica = self.replicas[rank]
        directory = replica.session.prefix_directory()
        if any(token >= replica.config['vocab_size'] for entry in directory for token in entry['token_ids']):
            raise RuntimeError('prefix directory token outside model vocabulary')
        self._prefixes[rank] = directory

    def _prefix_score(self, rank, ids):
        """/** 匹配Token前缀及RoPE起点0，至少留一个Prompt Token重算，与原生恢复边界相同。 */"""
        best = 0
        for entry in self._prefixes[rank]:
            if entry['position_begin'] != 0:
                continue
            common = 0
            for a, b in zip(entry['token_ids'], ids[:-1]):
                if a != b:
                    break
                common += 1
            best = max(best, common)
        states = getattr(self.replicas[rank].session, '_graph_state', None)
        prefix = states[0].get('prefix_cache', {}) if states else {}
        page = prefix.get('page_len', 1) if prefix.get('directory_version') == 2 else 1
        return best // page * page

    def _update_load(self, rank, state):
        """/** 接收会话已校验的真实KV统计，不用异步事件覆盖本地请求计数。 */"""
        with self._condition:
            self._loads[rank] = copy.deepcopy(state)

    def load_snapshot(self):
        """/** 在同一锁内读取等待/运行数和最近KV压力，供路由、日志和健康接口使用。 */"""
        with self._condition:
            result = []
            for rank in range(self.dp_size):
                waiting, running = len(self._queues[rank]), int(rank in self._busy)
                load = self._loads[rank]
                usage = load['kv_cache_usage'] if load else 0.0
                result.append(dict(dp_rank=rank, waiting=waiting, running=running,
                    kv_cache_usage=usage if load else None, sampled_at=load['sampled_at'] if load else None,
                    score=waiting + running + waiting * 6 * max(0.0, usage - 0.5)))
            return result

    def _reserve(self, cancel_event=None, deadline=None, request=None):
        """/** 原子选择并登记固定副本；HTTP可先预约再分词，取消和超时沿用同一截止时间。 */"""
        with self._condition:
            if self.state != 'ready':
                raise RuntimeError(f'DP pool is {self.state}; routing unavailable')
            started = time.monotonic()
            deadline = started + self.timeout if deadline is None else deadline
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled(dict(id=0, status='FAIL', cancelled=True, error='cancelled before DP admission'))
            if deadline <= started:
                raise TimeoutError('DP admission deadline exceeded')
            snapshot = self.load_snapshot()
            candidates = [(self._next_rank + offset) % self.dp_size for offset in range(self.dp_size)]
            scores = {}
            if self.routing == 'load':
                rank = min(candidates, key=lambda item: snapshot[item]['score'])
            elif self.routing == 'prefix':
                if request is None:
                    raise ValueError('prefix routing requires tokenized request')
                scores = {rank: self._prefix_score(rank, request['token_ids']) for rank in candidates}
                rank = max(candidates, key=lambda item: scores[item])
            else:
                rank = candidates[0]
            self._next_rank = (self._next_rank + 1) % self.dp_size
            request_id, self._next_id = self._next_id, self._next_id + 1
            ticket = dict(owner=self, rank=rank, id=request_id, state='queued', started=started,
                          deadline=deadline, cancel=cancel_event, error=None, logged=False,
                          decision=dict(policy=self.routing, replicas=snapshot, selected_score=snapshot[rank]['score'],
                                        idle_ranks=[r for r in candidates if r not in self._busy],
                                        estimated_reused_tokens=scores.get(rank), wait_ms=0.0))
            self._queues[rank].append(ticket)
            self._reservations[request_id] = ticket
            self._condition.notify_all()
            return ticket

    def _write_dispatch(self, ticket, record):
        """/** 每个预约只保存一份终态记录，覆盖尚未进入执行线程的撤销。 */"""
        if not ticket['logged']:
            if self.request_log:
                (self.output / f'dispatch-{ticket["id"]:06d}.json').write_text(
                    json.dumps(record, ensure_ascii=False, indent=2))
            ticket['logged'] = True

    def _withdraw(self, ticket, error=None):
        """/** 幂等撤销尚未执行的预约；运行中只由执行位释放路径减账。 */"""
        with self._condition:
            if ticket.get('owner') is not self:
                raise ValueError('foreign DP reservation')
            if ticket['state'] != 'queued':
                return False
            self._queues[ticket['rank']].remove(ticket)
            self._reservations.pop(ticket['id'], None)
            ticket['state'], ticket['error'] = 'cancelled', error
            ticket['decision']['wait_ms'] = (time.monotonic() - ticket['started']) * 1000
            try:
                self._write_dispatch(ticket, dict(dp_rank=ticket['rank'], dp_request_id=ticket['id'], status='FAIL',
                    error=str(error or 'DP reservation cancelled before execution'), dp_routing=ticket['decision']))
            except Exception as failure:
                self._failure = f'cannot finalize DP dispatch: {failure}'
            self._condition.notify_all()
            return True

    def _cancel_queued(self, error):
        """/** 整池停止时撤销全部等待预约；调用者持有调度锁。 */"""
        for ticket in list(self._reservations.values()):
            self._withdraw(ticket, error)

    @contextmanager
    def _load_lease(self, cancel_event=None, reservation=None, request=None):
        """/** 按固定FIFO获取执行位，异常或取消只释放一次；运行位保持到原生清理/流关闭。 */"""
        ticket = reservation if reservation is not None else self._reserve(cancel_event, request=request)
        if not isinstance(ticket, dict) or ticket.get('owner') is not self:
            raise ValueError('foreign DP reservation')
        rank, request_id = ticket['rank'], ticket['id']
        acquired = False
        record = dict(dp_rank=rank, dp_request_id=request_id, status='PASS', dp_routing=ticket['decision'])
        try:
            with self._condition:
                while True:
                    if ticket['state'] != 'queued':
                        if ticket['error'] is not None:
                            raise ticket['error']
                        raise GenerationCancelled(dict(id=0, status='FAIL', cancelled=True, error='DP reservation cancelled'))
                    if self.state != 'ready':
                        raise RuntimeError(f'DP pool is {self.state}: {self._failure or "unavailable"}')
                    if any(event is not None and event.is_set() for event in (cancel_event, ticket['cancel'])):
                        raise GenerationCancelled(dict(id=0, status='FAIL', cancelled=True, error='cancelled in DP queue'))
                    remaining = ticket['deadline'] - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('waiting in DP replica queue timed out')
                    if not self._maintenance and rank not in self._busy and self._queues[rank][0] is ticket:
                        self._queues[rank].popleft()
                        self._busy.add(rank)
                        ticket['state'] = 'running'
                        acquired = True
                        ticket['decision']['wait_ms'] = (time.monotonic() - ticket['started']) * 1000
                        break
                    self._condition.wait(min(0.05, remaining))
            yield rank, request_id, self.replicas[rank], ticket['decision']
        except BaseException as error:
            record.update(status='FAIL', error=f'{type(error).__name__}: {error}')
            if hasattr(error, 'report'):
                error.report = dict(error.report, dp_rank=rank, dp_request_id=request_id, dp_routing=ticket['decision'])
            raise
        finally:
            with self._condition:
                try:
                    if acquired and self._status == 'ready' and self.replicas[rank].state != 'ready':
                        self._failure = f'replica {rank} is {self.replicas[rank].state}'
                    elif acquired and self._status == 'ready' and self.routing == 'prefix':
                        self._refresh_prefixes(rank)
                    self._write_dispatch(ticket, record)
                except Exception as error:
                    self._failure = f'cannot finalize DP dispatch: {error}'
                    raise
                finally:
                    if acquired:
                        self._busy.discard(rank)
                        self._reservations.pop(request_id, None)
                        ticket['state'] = 'finished'
                    else:
                        self._withdraw(ticket)
                    if self.state == 'failed':
                        self._cancel_queued(RuntimeError('DP pool failed'))
                    self._condition.notify_all()

    @contextmanager
    def _lease(self, cancel_event=None, request=None, reservation=None):
        """/** 三种策略统一预约健康副本并在其本地排队，忙碌不等于不可用。 */"""
        with self._load_lease(cancel_event, reservation, request) as lease:
            yield lease

    @contextmanager
    def _idle_lease(self, cancel_event=None, request=None):
        """/** 独占一个空闲副本直至请求/流结束；等待可取消，故障不重试。 */"""
        started = time.monotonic()
        deadline = started + self.timeout
        with self._condition:
            while True:
                if self.state != 'ready':
                    raise RuntimeError(f'DP pool is {self.state}: {self._failure or "unavailable"}')
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelled(dict(id=0, status='FAIL', cancelled=True,
                                                   error='cancelled before DP dispatch'))
                candidates = [(self._next_rank + offset) % self.dp_size for offset in range(self.dp_size)]
                idle = [rank for rank in candidates if rank not in self._busy] if not self._maintenance else []
                scores = {rank: self._prefix_score(rank, request['token_ids']) for rank in idle} if self.routing == 'prefix' else {}
                rank = max(idle, key=lambda r: scores.get(r, 0)) if idle else None
                if rank is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('waiting for an idle DP replica timed out')
                self._condition.wait(min(0.05, remaining))
            self._busy.add(rank)
            self._next_rank = (rank + 1) % self.dp_size
            request_id, self._next_id = self._next_id, self._next_id + 1

        decision = dict(policy=self.routing, idle_ranks=idle, estimated_reused_tokens=scores.get(rank),
                        wait_ms=(time.monotonic() - started) * 1000)
        record = dict(dp_request_id=request_id, dp_rank=rank, status='PASS', dp_routing=decision)
        try:
            yield rank, request_id, self.replicas[rank], decision
        except BaseException as error:
            record.update(status='FAIL', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            with self._condition:
                try:
                    if self._status == 'ready' and self.replicas[rank].state != 'ready':
                        self._prefixes[rank] = []
                        self._failure = f'replica {rank} is {self.replicas[rank].state}'
                    elif self._status == 'ready' and self.routing == 'prefix':
                        self._refresh_prefixes(rank)
                    if self.request_log:
                        (self.output / f'dispatch-{request_id:06d}.json').write_text(
                            json.dumps(record, ensure_ascii=False, indent=2))
                except Exception as error:
                    self._prefixes[rank] = []
                    self._failure = f'cannot finalize DP dispatch: {error}'
                    raise
                finally:
                    self._busy.remove(rank)
                    self._condition.notify_all()

    def generate(self, request, *, cancel_event=None, on_token=None):
        """/** 生成已编码请求；返回原生结果及全局 DP 请求编号、副本编号。 */"""
        return self._generate_reserved(None, request, cancel_event=cancel_event, on_token=on_token)

    def _generate_reserved(self, reservation, request, *, cancel_event=None, on_token=None):
        """/** HTTP复用入队时的预约；直接调用在校验Token后预约，执行时不重新选副本。 */"""
        try:
            request = copy.deepcopy(request)
            validate_request(request, self.config['vocab_size'])
        except BaseException:
            if reservation is not None:
                self._withdraw(reservation)
            raise
        with self._lease(cancel_event, request, reservation) as (rank, request_id, replica, decision):
            try:
                result = replica.generate(request, cancel_event=cancel_event, on_token=on_token)
            except Exception as error:
                if hasattr(error, 'report'):
                    error.report = dict(error.report, dp_rank=rank, dp_request_id=request_id, dp_routing=decision)
                raise
            return dict(result, dp_rank=rank, dp_request_id=request_id, dp_routing=decision)

    def _prepare_request(self, query, history=None, max_length=32, top_k=0, top_p=1.0,
                         temperature=0.0, repeat_penalty=1.0, seed=0):
        """/** 分配前只分词一次；采用正式TP请求构造器，不在分发后改变Token或采样参数。 */"""
        with self._prepare_lock:
            return self.replicas[0]._request(query, history, max_length, top_k, top_p, temperature, repeat_penalty, seed)

    def response(self, query, *args, **kwargs):
        request = self._prepare_request(query, *args, **kwargs)
        report = self.generate(request)
        with self._prepare_lock:
            return self.hf_tokenizer.decode(report['token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def stream_response(self, query, *args, **kwargs):
        """/** 流结束或迭代器关闭后才归还副本，复用 TP 取消与 KV 回收语义。 */"""
        request = self._prepare_request(query, *args, **kwargs)
        with self._lease(request=request) as (_, _, replica, _):
            stream = replica._stream_request(request)
            try:
                yield from stream
            finally:
                stream.close()

    def clear_prefix_cache(self):
        """/** 暂停新分配并等待现有调用释放副本后清空全部快照；等待有超时。 */"""
        with self._condition:
            if not self._condition.wait_for(lambda: not self._maintenance or self.state != 'ready', self.timeout):
                raise TimeoutError('waiting for DP cache maintenance timed out')
            if self.state != 'ready':
                raise RuntimeError(f'DP pool is {self.state}')
            self._maintenance = True
        clearing = False
        try:
            with self._condition:
                if not self._condition.wait_for(lambda: not self._busy or self.state != 'ready', self.timeout):
                    raise TimeoutError('waiting for DP callers before cache clear timed out')
                if self.state != 'ready':
                    raise RuntimeError(f'DP pool is {self.state}')
                clearing = True
            results = []
            for rank, replica in enumerate(self.replicas):
                results.append(replica.clear_prefix_cache())
                if self.routing == 'prefix':
                    self._refresh_prefixes(rank)
            return results
        except BaseException:
            if clearing:
                with self._condition:
                    self._prefixes = [[] for _ in range(self.dp_size)]
                    self._failure = 'DP prefix cache clear failed'
            raise
        finally:
            with self._condition:
                self._maintenance = False
                self._condition.notify_all()

    def close(self):
        """/** 停止分配并并行取消、回收全部副本；重复调用安全，清理失败报错。 */"""
        with self._close_lock:
            with self._condition:
                if self._status == 'closed':
                    return
                self._status = 'closing'
                self._cancel_queued(RuntimeError('DP pool closing'))
                self._prefixes = [[] for _ in range(self.dp_size)]
                self._condition.notify_all()
            errors = []
            if self.replicas:
                with ThreadPoolExecutor(max_workers=len(self.replicas)) as pool:
                    futures = [pool.submit(replica.close) for replica in self.replicas]
                    for future in futures:
                        try:
                            future.result()
                        except BaseException as error:
                            errors.append(f'{type(error).__name__}: {error}')
            with self._condition:
                # 原生工作已回收；暂停在 yield 的调用者可稍后关闭迭代器，不能阻塞停服。
                self._busy.clear()
                for ticket in self._reservations.values():
                    ticket['state'] = 'finished'
                self._reservations.clear()
                self._status = 'failed' if errors else 'closed'
                self._condition.notify_all()
            (self.output / 'replicas.shutdown.json').write_text(json.dumps(
                dict(status='FAIL' if errors else 'PASS', errors=errors,
                     replicas=[dict(dp_rank=i, state=replica.state) for i, replica in enumerate(self.replicas)]),
                ensure_ascii=False, indent=2))
            if errors:
                raise RuntimeError('DP cleanup failed: ' + '; '.join(errors))

    release_memory = close

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
