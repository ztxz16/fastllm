"""D 解码结束后的有期限 KV 保留；空闲绑定与 KV 生命周期相互独立。"""

from dataclasses import dataclass, field
import threading
import time
import math

from .protocol import ProtocolError, prefix_manifest


@dataclass
class Retained:
    manifest: object
    tokens: tuple
    expires: float
    release: object
    readers: dict = field(default_factory=dict)


class History:
    def __init__(self, peer, ttl=480, *, clock=time.monotonic):
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("history retention must be positive")
        self.peer, self.ttl, self.clock = peer, ttl, clock
        self.lock = threading.RLock()
        self.entries = {}
        self.closing = False

    def retain(self, ticket, manifest, tokens, release):
        """/** 仅在 Decode 完成后开始计时；运行时长不计入保留期。 */"""
        with self.lock:
            if self.closing or ticket in self.entries or len(tokens) != manifest.token_end:
                raise ProtocolError("invalid history retention")
            self.entries[ticket] = Retained(manifest, tuple(tokens), self.clock() + self.ttl, release)

    def acquire(self, ticket, identity, tokens):
        """/** 下一轮 P 拉取前取得租约；过期拒绝新读取，已有读取仍保持保护。 */"""
        with self.lock:
            item = self.entries.get(ticket)
            if identity.direction != "d_to_p" or identity.mode != "pull" or identity.source != self.peer:
                raise ProtocolError("invalid history transfer identity")
            if self.closing or item is None or self.clock() >= item.expires:
                return None
            count = 0
            for a, b in zip(item.tokens, tokens[:-1]):
                if a != b:
                    break
                count += 1
            if not count:
                return None
            manifest = prefix_manifest(item.manifest, identity, self.peer, tokens[:count])
            old = item.readers.get(identity.transfer)
            if old is not None and old != manifest:
                raise ProtocolError("history lease replay changed")
            item.readers[identity.transfer] = manifest
            return manifest

    def release(self, ticket, identity):
        with self.lock:
            item = self.entries.get(ticket)
            if item is None:
                return
            old = item.readers.get(identity.transfer)
            if old is not None:
                if old.identity != identity:
                    raise ProtocolError("history lease identity mismatch")
                del item.readers[identity.transfer]
            self.expire()

    def expire(self):
        with self.lock:
            for ticket, item in list(self.entries.items()):
                if self.clock() >= item.expires and not item.readers:
                    item.release()
                    del self.entries[ticket]

    def close(self):
        """/** 拒绝新读取并回收无读者的保留；未归还的远端租约不能被关闭强行释放。 */"""
        with self.lock:
            self.closing = True
            for item in self.entries.values():
                item.expires = min(item.expires, self.clock())
            self.expire()
            return not self.entries
