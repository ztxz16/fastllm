"""按逻辑 Token/KV head 区间映射 TP 分片与分页/连续存储。"""

from . import _core
from dataclasses import dataclass
from .protocol import Manifest, Region


@dataclass(frozen=True)
class Copy:
    source: Region
    target: Region
    token_begin: int
    token_end: int
    head_begin: int
    head_end: int
    vector_bytes: int

    @property
    def bytes(self):
        return _core.invoke('mapping.Copy.bytes', locals())


def plan_transfer(source: Manifest, target: Manifest):
    return _core.invoke('mapping.plan_transfer', locals())
