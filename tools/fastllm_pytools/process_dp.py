"""完整模型副本 DP：Python 只管理副本和请求，前向、采样及 TP 通信仍在 C++/CUDA。"""

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
     * 全部副本就绪后才接受请求；按空闲副本轮转，每个请求只提交一次。
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
        self.timeout = getattr(args, 'tp_timeout', 180)
        self.output = Path(getattr(args, 'tp_output', 'process-tp-results')).resolve() / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_dp_' + uuid.uuid4().hex[:8])
        self.output.mkdir(parents=True, exist_ok=False)
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._status, self._failure = 'starting', None
        self._maintenance = False
        self._busy, self._next_rank, self._next_id = set(), 0, 1
        self.replicas = []
        environment = dict(os.environ)
        environment.setdefault('USE_TORCH', '0')

        try:
            for assignment in self.plan['replicas']:
                replica_args = copy.deepcopy(args)
                replica_args.dp, replica_args.tp = 1, str(self.tp_size)
                replica_args.tp_output = str(self.output / f"replica-{assignment['dp_rank']}")
                devices = [device['logical_id'] for device in assignment['devices']]
                replica = ProcessTPModel(replica_args, _device_ids=devices, _environment=environment)
                self.replicas.append(replica)
                if (replica.ready.get('cancellation') != 'request_id_datagram' or
                        replica.ready.get('token_events') is not True):
                    raise RuntimeError('all DP replicas require cancellation and token events')

            first = self.replicas[0]
            self.model_path, self.config, self.hf_tokenizer = first.model_path, first.config, first.hf_tokenizer
            self.devices = [replica.devices for replica in self.replicas]
            self.ready = dict(event='ready', status='PASS', backend=BACKEND, dp_size=self.dp_size,
                              tp_size=self.tp_size, model_load_count=self.dp_size,
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

    @contextmanager
    def _lease(self, cancel_event=None):
        """/** 独占一个空闲副本直至请求/流结束；等待可取消，故障不重试。 */"""
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while True:
                if self.state != 'ready':
                    raise RuntimeError(f'DP pool is {self.state}: {self._failure or "unavailable"}')
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelled(dict(id=0, status='FAIL', cancelled=True,
                                                   error='cancelled before DP dispatch'))
                candidates = [(self._next_rank + offset) % self.dp_size for offset in range(self.dp_size)]
                rank = next((rank for rank in candidates if rank not in self._busy), None) if not self._maintenance else None
                if rank is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('waiting for an idle DP replica timed out')
                self._condition.wait(min(0.05, remaining))
            self._busy.add(rank)
            self._next_rank = (rank + 1) % self.dp_size
            request_id, self._next_id = self._next_id, self._next_id + 1

        record = dict(dp_request_id=request_id, dp_rank=rank, status='PASS')
        try:
            yield rank, request_id, self.replicas[rank]
        except BaseException as error:
            record.update(status='FAIL', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            with self._condition:
                try:
                    if self._status == 'ready' and self.replicas[rank].state != 'ready':
                        self._failure = f'replica {rank} is {self.replicas[rank].state}'
                    (self.output / f'dispatch-{request_id:06d}.json').write_text(
                        json.dumps(record, ensure_ascii=False, indent=2))
                except Exception as error:
                    self._failure = f'cannot persist DP dispatch: {error}'
                    raise
                finally:
                    self._busy.remove(rank)
                    self._condition.notify_all()

    def generate(self, request, *, cancel_event=None, on_token=None):
        """/** 生成已编码请求；返回原生结果及全局 DP 请求编号、副本编号。 */"""
        validate_request(request, self.config['vocab_size'])
        with self._lease(cancel_event) as (rank, request_id, replica):
            try:
                result = replica.generate(request, cancel_event=cancel_event, on_token=on_token)
            except Exception as error:
                if hasattr(error, 'report'):
                    error.report = dict(error.report, dp_rank=rank, dp_request_id=request_id)
                raise
            return dict(result, dp_rank=rank, dp_request_id=request_id)

    def response(self, query, *args, **kwargs):
        with self._lease() as (_, _, replica):
            return replica.response(query, *args, **kwargs)

    def stream_response(self, query, *args, **kwargs):
        """/** 流结束或迭代器关闭后才归还副本，复用 TP 取消与 KV 回收语义。 */"""
        with self._lease() as (_, _, replica):
            stream = replica.stream_response(query, *args, **kwargs)
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
        try:
            with self._condition:
                if not self._condition.wait_for(lambda: not self._busy or self.state != 'ready', self.timeout):
                    raise TimeoutError('waiting for DP callers before cache clear timed out')
                if self.state != 'ready':
                    raise RuntimeError(f'DP pool is {self.state}')
            return [replica.clear_prefix_cache() for replica in self.replicas]
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
