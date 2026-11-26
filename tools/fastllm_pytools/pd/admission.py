"""KV 容量承诺、等待队列与本地抢占决策；不直接释放原生 KV。"""

from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import time

from .protocol import integer


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
        self.capacity = integer(capacity, "KV capacity", 1)
        self.growth = integer(growth, "decode growth budget")
        self.clock = clock
        self.records, self.waiting = {}, OrderedDict()
        self.lock = threading.RLock()

    @property
    def available(self):
        with self.lock:
            return self.capacity - sum(r.used + r.growth for r in self.records.values())

    def request(self, key, tokens, *, decode=True, remaining=None):
        """/** 原子预约；不足只进入等待队列，不挤占其他活动请求的增长承诺。 */"""
        integer(tokens, "KV tokens", 1)
        reserve = self.growth if decode else 0
        if remaining is not None:
            reserve = min(reserve, integer(remaining, "remaining tokens"))
        if tokens + reserve > self.capacity:
            raise ValueError("request exceeds total KV capacity with growth budget")
        with self.lock:
            old = self.records.get(key)
            if old:
                if (old.used, old.growth) != (tokens, reserve):
                    raise ValueError("admission key reused with different allocation")
                return True
            old_wait = self.waiting.get(key)
            if old_wait is not None and old_wait != (tokens, reserve):
                raise ValueError("waiting key reused with different allocation")
            self.waiting.setdefault(key, (tokens, reserve))
            if next(iter(self.waiting)) != key or tokens + reserve > self.available:
                return False
            del self.waiting[key]
            self.records[key] = Allocation(key, tokens, reserve, touched=self.clock())
            return True

    def ready(self, key):
        with self.lock:
            record = self.records[key]
            record.state, record.touched = "running", self.clock()

    def grow(self, key, count=1, *, remaining=None):
        """/** 使用已有预算追加 KV；预算耗尽后不足则返回 False，由调度器选择本地抢占。 */"""
        integer(count, "growth count", 1)
        with self.lock:
            record = self.records[key]
            if record.state != "running":
                raise ValueError("only a running request may grow")
            deficit = max(0, count - record.growth)
            if deficit > self.available:
                return False
            record.used += count
            record.growth = max(0, record.growth - count)
            wanted = self.growth if remaining is None else min(self.growth, integer(remaining, "remaining"))
            record.growth += min(max(0, wanted - record.growth), self.available)
            record.touched = self.clock()
            return True

    def pin(self, key):
        with self.lock:
            self.records[key].pins += 1

    def unpin(self, key):
        with self.lock:
            record = self.records[key]
            if record.pins <= 0:
                raise ValueError("duplicate KV unpin")
            record.pins -= 1

    def cache(self, key):
        with self.lock:
            record = self.records[key]
            record.state, record.growth, record.touched = "cached", 0, self.clock()

    def victim(self):
        """/** 先选无 pin 的 LRU 缓存，再选最后准入的运行请求；不隐式释放任何数据。 */"""
        with self.lock:
            cached = [r for r in self.records.values() if not r.pins and r.state == "cached"]
            if cached:
                return min(cached, key=lambda r: r.touched).key
            running = [r for r in self.records.values() if not r.pins and r.state == "running"]
            return running[-1].key if running else None

    def retire(self, key):
        """/** 原生释放已确认后回收预算；在途 pin 未清零时明确拒绝。 */"""
        with self.lock:
            record = self.records.get(key)
            if record and record.pins:
                raise RuntimeError("KV is still in flight")
            self.records.pop(key, None)
            self.waiting.pop(key, None)
