"""D 解码结束后的有期限 KV 保留；空闲绑定与 KV 生命周期相互独立。"""

from . import _core
from dataclasses import dataclass, field
import time
import threading
from .protocol import Manifest


@dataclass
class Retained:
    manifest: object
    tokens: tuple
    expires: float
    release: object
    readers: dict = field(default_factory=dict)


class History:
    def __init__(self, peer, ttl=480, *, clock=time.monotonic):
        return _core.invoke('history.History.__init__', locals())

    def retain(self, ticket, manifest, tokens, release):
        return _core.invoke('history.History.retain', locals())

    def acquire(self, ticket, identity, tokens):
        return _core.invoke('history.History.acquire', locals())

    def release(self, ticket, identity):
        return _core.invoke('history.History.release', locals())

    def expire(self):
        return _core.invoke('history.History.expire', locals())

    def close(self):
        return _core.invoke('history.History.close', locals())
