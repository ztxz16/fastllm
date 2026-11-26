"""服务级路由、健康状态和缓存目录；与模型内部 DP 调度解耦。"""

from . import _core
from collections import OrderedDict
from dataclasses import dataclass, field
import time


POLICIES = ("round_robin", "random", "power_of_two", "cache_aware", "kv_aware",
            "consistent_hash", "sticky_least_loaded", "rendezvous_hash")


@dataclass
class Worker:
    name: str
    url: str
    role: str
    model: str
    boot: str
    healthy: bool = False
    failures: int = 0
    successes: int = 0
    inflight: int = 0
    backend_load: object = None
    load_at: float = 0.0
    history: OrderedDict = field(default_factory=OrderedDict)
    cached: dict = field(default_factory=dict)
    cache_sequence: int = 0


@dataclass(frozen=True)
class Lease:
    worker: str
    boot: str
    ticket: int


class Registry:
    """/**
     * 原子管理健康候选、请求计数、会话绑定与缓存记录。
     * 未健康服务保留注册信息供持续探测；显式注销才删除。
     * inflight 按一次选路加一、响应或失败结束减一，重放 release 不重复扣减。
     */"""

    def __init__(self, *, policy="cache_aware", prefill_policy=None, decode_policy=None,
                 cache_threshold=0.3, balance_absolute=64, balance_relative=1.5,
                 failure_threshold=3, success_threshold=2, session_ttl=7200,
                 history_limit=4096, load_ttl=10, clock=time.monotonic, rng=None):
        return _core.invoke('routing.Registry.__init__', locals())

    def register(self, name, url, role, model, boot):
        return _core.invoke('routing.Registry.register', locals())

    def unregister(self, name):
        return _core.invoke('routing.Registry.unregister', locals())

    def health_result(self, name, boot, success):
        return _core.invoke('routing.Registry.health_result', locals())

    def update_load(self, name, boot, count):
        return _core.invoke('routing.Registry.update_load', locals())

    def cache_event(self, name, boot, sequence, operation, key, tokens=()):
        return _core.invoke('routing.Registry.cache_event', locals())

    def cache_snapshot(self, name, boot, entries):
        return _core.invoke('routing.Registry.cache_snapshot', locals())

    @staticmethod
    def _prefix(tokens, entries):
        return _core.invoke('routing.Registry._prefix', locals())

    @staticmethod
    def _hash(text):
        return _core.invoke('routing.Registry._hash', locals())

    def select(self, role, model, tokens=(), session=None, *, exclude=()):
        return _core.invoke('routing.Registry.select', locals())

    def worker(self, lease):
        return _core.invoke('routing.Registry.worker', locals())

    def release(self, lease):
        return _core.invoke('routing.Registry.release', locals())

    def snapshot(self):
        return _core.invoke('routing.Registry.snapshot', locals())
