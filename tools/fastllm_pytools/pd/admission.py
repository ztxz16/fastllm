"""KV 容量承诺、等待队列与本地抢占决策；不直接释放原生 KV。"""

from . import _core
from dataclasses import dataclass, field
import time


@dataclass
class Allocation:
    key: str
    used: int
    growth: int
    state: str = "receiving"
    pins: int = 0
    touched: float = 0.0
    resume: dict = field(default_factory=dict)


class Admission:
    """/**
     * 按 Token 槽计费，同一池必须采用相同每 Token KV 字节数。
     * used 是实际 KV，growth 是尚未分配的准入承诺；计算与传输 pin 禁止抢占。
     * 淘汰和抢占返回候选 key，调用者确认原生资源已解除后才能 retire。
     */"""

    def __init__(self, capacity, growth=512, *, clock=time.monotonic):
        return _core.invoke('admission.Admission.__init__', locals())

    @property
    def available(self):
        return _core.invoke('admission.Admission.available', locals())

    def request(self, key, tokens, *, decode=True, remaining=None):
        return _core.invoke('admission.Admission.request', locals())

    def ready(self, key):
        return _core.invoke('admission.Admission.ready', locals())

    def grow(self, key, count=1, *, remaining=None):
        return _core.invoke('admission.Admission.grow', locals())

    def pin(self, key):
        return _core.invoke('admission.Admission.pin', locals())

    def unpin(self, key):
        return _core.invoke('admission.Admission.unpin', locals())

    def cache(self, key):
        return _core.invoke('admission.Admission.cache', locals())

    def victim(self):
        return _core.invoke('admission.Admission.victim', locals())

    def retire(self, key):
        return _core.invoke('admission.Admission.retire', locals())
