"""CPU 后台传输跟踪；调度器通过通知队列取得结果，后台线程不执行 TP/EP 通信。"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
import time

from .mapping import plan_transfer
from .protocol import Identity, Manifest, Peer, ProtocolError


class State(str, Enum):
    WAITING = "waiting"
    TRANSFERRING = "transferring"
    READY = "ready"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class Completion:
    identity: Identity
    state: State
    error: str = ""


@dataclass
class Task:
    identity: Identity
    source: Manifest
    target: Manifest
    copies: tuple
    release: object
    state: State = State.WAITING
    ready: set = field(default_factory=set)
    submitted: set = field(default_factory=set)
    pending: dict = field(default_factory=dict)
    final: bool = False
    error: str = ""
    notified: bool = False
    released: bool = False


class TransferManager:
    """/**
     * 按交接编号管理传输，不持有模型权重或调度计算的权限。
     * backend 提供 submit(copies)、query(handle)、retire(handle)；query 只有确认
     * 实际 GPU 访问结束才返回 True，失败时任务保留占用，不把异常当作完成。
     * release 回调只解除传输保护，接收 KV 的请求引用由调度器继续持有。
     */"""

    def __init__(self, owner: Peer, backend, *, poll_interval=0.002):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.owner, self.backend = owner, backend
        self.poll_interval = poll_interval
        self.notifications = queue.SimpleQueue()
        self._tasks = {}
        self._condition = threading.Condition(threading.RLock())
        self._closing = False
        self._statistics = dict(completed_batches=0, copied_bytes=0, gpu_copy_ms=0.0)
        self._thread = threading.Thread(target=self._run, name="ft-pd-transfer", daemon=True)
        self._thread.start()

    def register(self, source, target, release):
        """/**
         * 校验完整映射后登记任务；重放相同交接不再次提交。
         * 接收位置必须由本地接收分配/对端登记得到，并在服务层核对当前授权记录。
         * @return 任务状态；拒绝不会产生 GPU 写入或调用 release。
         */"""
        identity = source.identity
        expected = identity.target if identity.mode == "pull" else identity.source
        if self.owner != expected:
            raise ProtocolError("transfer initiated by wrong instance or boot")
        copies = plan_transfer(source, target)
        with self._condition:
            if self._closing:
                raise RuntimeError("transfer manager is closing")
            old = self._tasks.get(identity.transfer)
            if old is not None:
                if old.identity != identity or old.source != source or old.target != target:
                    raise ProtocolError("transfer ID reused with different data")
                return old.state
            self._tasks[identity.transfer] = Task(identity, source, target, copies, release)
            self._condition.notify_all()
            return State.WAITING

    def publish_ready(self, identity, rank, layer, *, final=False):
        """/** 记录源 rank 的逐层就绪；重复通知幂等，拒绝旧交接或越界层。 */"""
        with self._condition:
            task = self._task(identity)
            if (rank, layer) not in {(c.source.rank, c.source.layer) for c in task.copies}:
                raise ProtocolError("ready notification does not match expected source")
            if task.state in (State.WAITING, State.TRANSFERRING):
                task.ready.add((rank, layer))
                task.final = task.final or final
                self._condition.notify_all()

    def seal(self, identity):
        """/** 源端确认本次 Prefill 已发布完整 KV 范围，不能代替逐层就绪。 */"""
        with self._condition:
            task = self._task(identity)
            task.final = True
            self._condition.notify_all()

    def cancel(self, identity, reason="cancelled"):
        """/** 停止后续提交；已提交访问全部结束前保持占用。 */"""
        with self._condition:
            task = self._task(identity)
            if task.state in (State.READY, State.CANCELLED, State.FAILED):
                return task.state
            task.state, task.error = State.CANCELLING, reason
            self._condition.notify_all()
            return task.state

    def status(self, identity):
        with self._condition:
            return self._task(identity).state

    def failures(self):
        """/** 暴露后台实际错误供健康判断；正常取消原因不计为传输设备故障。 */"""
        with self._condition:
            return [task.error for task in self._tasks.values()
                    if task.error and task.state not in (State.CANCELLING, State.CANCELLED)]

    def statistics(self):
        """/** 返回实际完成批次与 CUDA event 耗时总和；并发 stream 的时间不能视作墙钟时间。 */"""
        with self._condition:
            return dict(self._statistics)

    def forget(self, identity):
        """/** 调度器完成通知处理后移除终态记录；活动任务禁止遗忘。 */"""
        with self._condition:
            task = self._task(identity)
            if not task.notified or not task.released:
                raise RuntimeError("cannot forget an in-flight transfer")
            del self._tasks[identity.transfer]

    def _task(self, identity):
        task = self._tasks.get(identity.transfer)
        if task is None or task.identity != identity:
            raise ProtocolError("unknown transfer or stale identity")
        return task

    def _finish(self, task, state):
        if task.notified:
            return
        if task.pending:
            raise RuntimeError("cannot finish with GPU accesses in flight")
        if not task.released:
            task.release()
            task.released = True
        task.state, task.notified = state, True
        self.notifications.put(Completion(task.identity, state, task.error))

    def _advance(self, task):
        # 查询成功后才解除后端映射引用；CUDA 错误仍保留句柄及源/目标保护。
        for key, handle in list(task.pending.items()):
            if self.backend.query(handle):
                error = getattr(handle, "error", "")
                self.backend.retire(handle)
                del task.pending[key]
                if not error:
                    self._statistics["completed_batches"] += 1
                    self._statistics["copied_bytes"] += getattr(handle, "bytes", 0)
                    self._statistics["gpu_copy_ms"] += getattr(handle, "gpu_ms", 0.0)
                if error:
                    task.error = error
        if task.state == State.CANCELLING:
            if not task.pending:
                self._finish(task, State.CANCELLED)
            return
        if task.error:
            if not task.pending:
                self._finish(task, State.FAILED)
            return

        groups = defaultdict(list)
        for index, copy in enumerate(task.copies):
            if index not in task.submitted and (copy.source.rank, copy.source.layer) in task.ready:
                groups[(copy.source.rank, copy.source.layer, copy.target.rank)].append((index, copy))
        for key, entries in groups.items():
            handle = self.backend.submit(tuple(copy for _, copy in entries))
            task.pending[key] = handle
            task.submitted.update(index for index, _ in entries)
            task.state = State.TRANSFERRING
        if task.final and len(task.submitted) == len(task.copies) and not task.pending:
            self._finish(task, State.READY)

    def _run(self):
        while True:
            with self._condition:
                active = [t for t in self._tasks.values() if not t.notified]
                if self._closing and not active:
                    return
                if not active:
                    self._condition.wait()
                    continue
                for task in active:
                    try:
                        self._advance(task)
                    except Exception as error:
                        # 不以 query 错误推断 GPU 已停止；保留未确认的占用，交给服务退出收尾。
                        task.error = str(error)
                        if not task.pending:
                            try:
                                self._finish(task, State.FAILED)
                            except Exception as cleanup:
                                task.error = f"{error}; release: {cleanup}"
                self._condition.wait(self.poll_interval)

    def close(self, timeout=None):
        """/**
         * 取消所有非终态任务并等待在途访问排空。
         * 超时仅报错，不释放仍被访问的内存；调用方须隔离服务并保留控制面继续排空。
         */"""
        with self._condition:
            self._closing = True
            for task in self._tasks.values():
                if not task.notified:
                    task.state, task.error = State.CANCELLING, "service shutdown"
            self._condition.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("PD transfer drain incomplete; KV remains protected")
        self.backend.close()
