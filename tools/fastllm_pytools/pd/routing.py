"""服务级路由、健康状态和缓存目录；与模型内部 DP 调度解耦。"""

from collections import OrderedDict
from dataclasses import dataclass, field
import bisect
import hashlib
import math
import random
import threading
import time
from urllib.parse import urlsplit

from .protocol import identifier, integer


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
        self.policies = {"full": policy, "p": prefill_policy or policy, "d": decode_policy or policy}
        if any(p not in POLICIES for p in self.policies.values()):
            raise ValueError("unsupported router policy")
        if (any(not math.isfinite(v) for v in (cache_threshold, balance_absolute, balance_relative,
                session_ttl, load_ttl)) or not 0 <= cache_threshold <= 1 or balance_absolute < 0 or balance_relative < 1):
            raise ValueError("invalid cache-aware thresholds")
        if min(failure_threshold, success_threshold, session_ttl, history_limit, load_ttl) <= 0:
            raise ValueError("invalid router limits")
        self.cache_threshold, self.balance_absolute, self.balance_relative = cache_threshold, balance_absolute, balance_relative
        self.failure_threshold, self.success_threshold = failure_threshold, success_threshold
        self.session_ttl, self.history_limit, self.load_ttl = session_ttl, history_limit, load_ttl
        self._clock, self._rng = clock, rng or random.SystemRandom()
        self._lock = threading.RLock()
        self._workers, self._bindings, self._leases, self._cursor = {}, {}, {}, {}
        self._next_ticket = 0

    def register(self, name, url, role, model, boot):
        """/** 注册明确的服务角色及启动身份；同地址重启使旧会话绑定与缓存信息失效。 */"""
        identifier(name, "service name")
        identifier(model, "model")
        identifier(boot, "boot")
        address = urlsplit(url)
        if (address.scheme not in ("http", "https") or not address.hostname or address.username or
                address.password or address.query or address.fragment or address.path not in ("", "/")):
            raise ValueError("service URL must be an HTTP(S) origin")
        if role not in ("full", "p", "d"):
            raise ValueError("invalid service role")
        with self._lock:
            old = self._workers.get(name)
            if old and (old.url, old.role, old.model, old.boot) == (url.rstrip("/"), role, model, boot):
                return
            if old and old.boot == boot:
                raise ValueError("service identity changed without a new boot ID")
            self.unregister(name)
            self._workers[name] = Worker(name, url.rstrip("/"), role, model, boot)

    def unregister(self, name):
        with self._lock:
            self._workers.pop(name, None)
            self._bindings = {k: v for k, v in self._bindings.items() if v[0] != name}

    def health_result(self, name, boot, success):
        """/** 按连续失败/成功阈值更新健康；迟到的上次启动探测无效。 */"""
        with self._lock:
            worker = self._workers.get(name)
            if worker is None or worker.boot != boot:
                return
            if success:
                worker.failures = 0
                worker.successes += 1
                if worker.successes >= self.success_threshold:
                    worker.healthy = True
            else:
                worker.successes = 0
                worker.failures += 1
                if worker.failures >= self.failure_threshold:
                    worker.healthy = False
                    self._bindings = {k: v for k, v in self._bindings.items() if v[0] != name}

    def update_load(self, name, boot, count):
        integer(count, "unfinished requests")
        with self._lock:
            worker = self._workers.get(name)
            if worker is not None and worker.boot == boot:
                worker.backend_load, worker.load_at = count, self._clock()

    def cache_event(self, name, boot, sequence, operation, key, tokens=()):
        """/**
         * 消费实际缓存新增/删除事件；序号断层时废弃旧目录，等待 reset 后重新同步。
         * 历史分发记录不注入实际 KV 目录，避免把 cache_aware 的推测当作缓存事实。
         */"""
        integer(sequence, "cache sequence", 1)
        token_tuple = tuple(integer(t, "token", 0, 0xffffffff) for t in tokens)
        if operation not in ("add", "remove", "reset"):
            raise ValueError("invalid cache operation")
        identifier(key, "cache key")
        with self._lock:
            worker = self._workers.get(name)
            if worker is None or worker.boot != boot or sequence <= worker.cache_sequence:
                return False
            if operation != "reset" and sequence != worker.cache_sequence + 1:
                worker.cached.clear()
                worker.cache_sequence = -1
                return False
            if worker.cache_sequence == -1 and operation != "reset":
                return False
            if operation == "reset":
                worker.cached.clear()
            elif operation == "add":
                if not token_tuple:
                    raise ValueError("cache add requires token IDs")
                worker.cached[key] = token_tuple
            else:
                worker.cached.pop(key, None)
            worker.cache_sequence = sequence
            return True

    def cache_snapshot(self, name, boot, entries):
        """/** 原子替换一次实际缓存目录，选路不会观察到 reset 与 add 之间的半份快照。 */"""
        parsed = {identifier(key, "cache key"): tuple(integer(t, "token", 0, 0xffffffff) for t in ids)
                  for key, ids in entries.items()}
        with self._lock:
            worker = self._workers.get(name)
            if worker is None or worker.boot != boot:
                return False
            worker.cached = parsed
            worker.cache_sequence = max(0, worker.cache_sequence) + 1
            return True

    @staticmethod
    def _prefix(tokens, entries):
        best = 0
        for entry in entries:
            length = 0
            for a, b in zip(tokens, entry):
                if a != b:
                    break
                length += 1
            best = max(best, length)
        return best

    @staticmethod
    def _hash(text):
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")

    def select(self, role, model, tokens=(), session=None, *, exclude=()):
        """/**
         * 在指定角色的健康服务池内选路并登记一次在途请求。
         * session 只用于亲和性；不承载历史，也不决定 KV 保留期限。
         * @return 带启动身份的 Lease；无候选时抛 LookupError，不随意选择不健康服务。
         */"""
        tokens = tuple(integer(t, "token", 0, 0xffffffff) for t in tokens)
        if session is not None:
            identifier(session, "session")
        with self._lock:
            now = self._clock()
            self._bindings = {k: v for k, v in self._bindings.items() if v[2] > now}
            candidates = sorted((w for w in self._workers.values() if w.healthy and w.role == role
                                 and w.model == model and w.name not in exclude), key=lambda w: w.name)
            if not candidates:
                raise LookupError(f"no healthy {role} service for {model}")
            policy = self.policies[role]
            cursor_key = (role, model)
            cursor = self._cursor.get(cursor_key, 0) % len(candidates)
            ordered = candidates[cursor:] + candidates[:cursor]
            lowest = min(ordered, key=lambda w: w.inflight)
            session_key = (role, model, session) if session else None
            if policy == "round_robin":
                chosen = ordered[0]
            elif policy == "random":
                chosen = self._rng.choice(candidates)
            elif policy == "power_of_two":
                pair = self._rng.sample(candidates, min(2, len(candidates)))
                chosen = min(pair, key=lambda w: w.backend_load if w.backend_load is not None
                             and now - w.load_at <= self.load_ttl else w.inflight)
            elif policy in ("cache_aware", "kv_aware"):
                maximum, minimum = max(w.inflight for w in candidates), lowest.inflight
                imbalance = maximum - minimum > self.balance_absolute and maximum > minimum * self.balance_relative
                matches = {w.name: self._prefix(tokens, w.history.values() if policy == "cache_aware"
                                                else w.cached.values()) for w in candidates}
                best = max(ordered, key=lambda w: matches[w.name])
                chosen = best if (not imbalance and tokens and
                                  matches[best.name] / len(tokens) > self.cache_threshold) else lowest
            elif policy == "sticky_least_loaded":
                binding = self._bindings.get(session_key) if session_key else None
                chosen = next((w for w in candidates if binding and (w.name, w.boot) == binding[:2]), None)
                if chosen is None:
                    counts = {w.name: 0 for w in candidates}
                    for key, value in self._bindings.items():
                        if key[:2] == (role, model) and value[0] in counts:
                            counts[value[0]] += 1
                    chosen = min(ordered, key=lambda w: counts[w.name]) if session_key else lowest
                if session_key:
                    self._bindings[session_key] = (chosen.name, chosen.boot, now + self.session_ttl)
            else:
                key = session or hashlib.sha256(repr(tuple(tokens)).encode()).hexdigest()
                if policy == "rendezvous_hash":
                    chosen = max(candidates, key=lambda w: self._hash(f"{key}\0{w.url}"))
                else:
                    ring = sorted((self._hash(f"{w.url}\0{i}"), w.name) for w in candidates for i in range(128))
                    index = bisect.bisect_left([h for h, _ in ring], self._hash(key)) % len(ring)
                    chosen = self._workers[ring[index][1]]
            self._cursor[cursor_key] = (candidates.index(chosen) + 1) % len(candidates)
            if tokens:
                token_tuple = tuple(integer(t, "token", 0, 0xffffffff) for t in tokens)
                key = hashlib.sha256(repr(token_tuple).encode()).hexdigest()
                chosen.history[key] = token_tuple
                chosen.history.move_to_end(key)
                while len(chosen.history) > self.history_limit:
                    chosen.history.popitem(last=False)
            chosen.inflight += 1
            self._next_ticket += 1
            lease = Lease(chosen.name, chosen.boot, self._next_ticket)
            self._leases[lease.ticket] = lease
            return lease

    def worker(self, lease):
        with self._lock:
            worker = self._workers.get(lease.worker)
            if worker is None or worker.boot != lease.boot:
                raise LookupError("service was removed or restarted")
            return worker

    def release(self, lease):
        with self._lock:
            if self._leases.get(lease.ticket) != lease:
                return
            del self._leases[lease.ticket]
            worker = self._workers.get(lease.worker)
            if worker is not None and worker.boot == lease.boot:
                worker.inflight -= 1

    def snapshot(self):
        with self._lock:
            return [dict(name=w.name, url=w.url, role=w.role, model=w.model, boot=w.boot,
                         healthy=w.healthy, inflight=w.inflight) for w in self._workers.values()]
