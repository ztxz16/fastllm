"""Router 请求级重试；与 GPU 传输重发、D 本地抢占分别管理。"""

from dataclasses import dataclass
import math
import random
import time

from .protocol import integer, new_id


@dataclass
class RetryBudget:
    deadline: float
    attempts: int = 1
    base: float = 0.05
    maximum: float = 2.0
    issued: int = 0
    responded: bool = False
    cancelled: bool = False

    def __post_init__(self):
        integer(self.attempts, "retry maximum attempts", 1, 100)
        if not all(math.isfinite(x) for x in (self.deadline, self.base, self.maximum)) or not 0 < self.base <= self.maximum:
            raise ValueError("invalid retry timing")

    def begin(self, now=None):
        """/** 每次尝试使用全新交接编号，所有尝试共用同一截止时间。 */"""
        now = time.monotonic() if now is None else now
        if self.cancelled or self.responded or now >= self.deadline or self.issued >= self.attempts:
            raise TimeoutError("request retry budget exhausted")
        self.issued += 1
        return new_id()

    def delay(self, *, status=None, transport_error=False, now=None, rng=random):
        """/** 返回退避秒数或 None；客户端响应开始后绝不重试。 */"""
        now = time.monotonic() if now is None else now
        retryable = transport_error or status == 429 or isinstance(status, int) and 500 <= status <= 599
        if not retryable or self.cancelled or self.responded or self.issued >= self.attempts:
            return None
        delay = rng.uniform(0, min(self.maximum, self.base * 2 ** max(0, self.issued - 1)))
        return delay if now + delay < self.deadline else None
