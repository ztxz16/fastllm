"""CPU 后台传输跟踪；调度器通过通知队列取得结果，后台线程不执行 TP/EP 通信。"""

from . import _core
from dataclasses import dataclass, field
from enum import Enum
from .protocol import Identity, Manifest, Peer


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
        return _core.invoke('tasks.TransferManager.__init__', locals())

    def register(self, source, target, release):
        return _core.invoke('tasks.TransferManager.register', locals())

    def publish_ready(self, identity, rank, layer, *, final=False):
        return _core.invoke('tasks.TransferManager.publish_ready', locals())

    def seal(self, identity):
        return _core.invoke('tasks.TransferManager.seal', locals())

    def cancel(self, identity, reason="cancelled"):
        return _core.invoke('tasks.TransferManager.cancel', locals())

    def status(self, identity):
        return _core.invoke('tasks.TransferManager.status', locals())

    def failures(self):
        return _core.invoke('tasks.TransferManager.failures', locals())

    def statistics(self):
        return _core.invoke('tasks.TransferManager.statistics', locals())

    def forget(self, identity):
        return _core.invoke('tasks.TransferManager.forget', locals())

    def _task(self, identity):
        return _core.invoke('tasks.TransferManager._task', locals())

    def _finish(self, task, state):
        return _core.invoke('tasks.TransferManager._finish', locals())

    def _advance(self, task):
        # 查询成功后才解除后端映射引用；CUDA 错误仍保留句柄及源/目标保护。
        return _core.invoke('tasks.TransferManager._advance', locals())

    def _run(self):
        return _core.invoke('tasks.TransferManager._run', locals())

    def close(self, timeout=None):
        return _core.invoke('tasks.TransferManager.close', locals())
