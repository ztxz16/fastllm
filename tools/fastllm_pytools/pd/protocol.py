"""PD 控制消息的有界 JSON 协议与不可变身份；不承载 KV 数据。"""

from . import _core
from dataclasses import dataclass
from typing import Tuple


VERSION = 1
MAX_META_BYTES = 16 * 1024 * 1024
MAX_REGIONS = 131072
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


class ProtocolError(ValueError):
    """控制消息在提交任何 GPU 读写前被拒绝。"""


def integer(value, name, minimum=0, maximum=(1 << 63) - 1):
    return _core.invoke('protocol.integer', locals())


def identifier(value, name):
    return _core.invoke('protocol.identifier', locals())


def new_id():
    return _core.invoke('protocol.new_id', locals())


def _object(cls, value):
    return _core.invoke('protocol._object', locals())


def ipc_handle(value):
    return _core.invoke('protocol.ipc_handle', locals())


@dataclass(frozen=True)
class Peer:
    instance: str
    boot: str
    host: str

    def __post_init__(self):
        return _core.invoke('protocol.Peer.__post_init__', locals())


@dataclass(frozen=True)
class Identity:
    request: str
    transfer: str
    source: Peer
    target: Peer
    direction: str
    mode: str

    def __post_init__(self):
        return _core.invoke('protocol.Identity.__post_init__', locals())


@dataclass(frozen=True)
class Layout:
    model: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype: str
    tp_size: int
    storage: str
    page_size: int

    def __post_init__(self):
        return _core.invoke('protocol.Layout.__post_init__', locals())

    @property
    def layout_id(self):
        return _core.invoke('protocol.Layout.layout_id', locals())

    def compatible(self, other):
        return _core.invoke('protocol.Layout.compatible', locals())


@dataclass(frozen=True)
class Region:
    rank: int
    device: str
    layer: int
    component: str
    token_begin: int
    token_end: int
    head_begin: int
    head_end: int
    memory: str
    allocation_bytes: int
    offset: int
    token_stride: int
    head_stride: int

    def validate(self, layout, begin, end):
        return _core.invoke('protocol.Region.validate', locals())

    def byte_offset(self, token, head):
        return _core.invoke('protocol.Region.byte_offset', locals())


@dataclass(frozen=True)
class Manifest:
    version: int
    identity: Identity
    owner: Peer
    layout: Layout
    token_begin: int
    token_end: int
    prefix_digest: str
    regions: Tuple[Region, ...]

    def __post_init__(self):
        return _core.invoke('protocol.Manifest.__post_init__', locals())

    def to_json(self):
        return _core.invoke('protocol.Manifest.to_json', locals())

    @classmethod
    def from_json(cls, payload):
        return _core.invoke('protocol.Manifest.from_json', locals())


def token_digest(tokens):
    return _core.invoke('protocol.token_digest', locals())


def prefix_manifest(manifest, identity, owner, tokens):
    return _core.invoke('protocol.prefix_manifest', locals())


def suffix_manifest(manifest, begin):
    return _core.invoke('protocol.suffix_manifest', locals())
