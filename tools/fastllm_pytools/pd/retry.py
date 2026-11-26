"""Router 请求级重试；与 GPU 传输重发、D 本地抢占分别管理。"""

from . import _core
from dataclasses import dataclass
import random


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
        return _core.invoke('retry.RetryBudget.__post_init__', locals())

    def begin(self, now=None):
        return _core.invoke('retry.RetryBudget.begin', locals())

    def delay(self, *, status=None, transport_error=False, now=None, rng=random):
        return _core.invoke('retry.RetryBudget.delay', locals())
