"""PD 控制消息的有界 JSON 协议与不可变身份；不承载 KV 数据。"""

from dataclasses import asdict, dataclass, fields
import base64
import hashlib
import json
import math
import re
from typing import Tuple
import uuid


VERSION = 1
MAX_META_BYTES = 16 * 1024 * 1024
MAX_REGIONS = 131072
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,192}\Z")


class ProtocolError(ValueError):
    """控制消息在提交任何 GPU 读写前被拒绝。"""


def integer(value, name, minimum=0, maximum=(1 << 63) - 1):
    """/** 校验协议整数，拒绝 bool、浮点截断及越界。 */"""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProtocolError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def identifier(value, name):
    """/** 校验无路径语义的标识；标识只用于匹配，不能当作文件名。 */"""
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ProtocolError(f"invalid {name}")
    return value


def new_id():
    """/** 为实例启动、请求或交接产生新的独立身份。 */"""
    return uuid.uuid4().hex


def _object(cls, value):
    """/** 严格反序列化字段集合；不接受未知字段或缺失的必填字段。 */"""
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
        raise ProtocolError(f"invalid {cls.__name__} fields")
    return cls(**value)


def ipc_handle(value):
    """/** 解码 CUDA 的 64 字节 IPC 句柄；拒绝地址字符串和非法编码。 */"""
    if not isinstance(value, str) or len(value) > 128:
        raise ProtocolError("invalid CUDA IPC handle")
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise ProtocolError("invalid CUDA IPC handle") from error
    if len(result) != 64:
        raise ProtocolError("CUDA IPC handle must contain 64 bytes")
    return result


@dataclass(frozen=True)
class Peer:
    instance: str
    boot: str
    host: str

    def __post_init__(self):
        for name in ("instance", "boot", "host"):
            identifier(getattr(self, name), name)


@dataclass(frozen=True)
class Identity:
    request: str
    transfer: str
    source: Peer
    target: Peer
    direction: str
    mode: str

    def __post_init__(self):
        identifier(self.request, "request")
        identifier(self.transfer, "transfer")
        if not isinstance(self.source, Peer) or not isinstance(self.target, Peer):
            raise ProtocolError("source and target must be Peer identities")
        if self.source.host != self.target.host:
            raise ProtocolError("cross-host KV transfer is not supported")
        if self.source == self.target:
            raise ProtocolError("source and target must be different service instances")
        if self.direction not in ("p_to_d", "d_to_p") or self.mode not in ("pull", "push"):
            raise ProtocolError("invalid transfer direction or mode")
        if self.direction == "d_to_p" and self.mode != "pull":
            raise ProtocolError("D to P supports pull only")


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
        identifier(self.model, "model fingerprint")
        integer(self.layers, "layers", 1, 4096)
        integer(self.kv_heads, "kv_heads", 1, 65536)
        integer(self.head_dim, "head_dim", 1, 65536)
        integer(self.tp_size, "tp_size", 1, 64)
        integer(self.page_size, "page_size", 0, 1048576)
        if self.dtype not in DTYPE_BYTES or self.storage not in ("paged", "continuous"):
            raise ProtocolError("unsupported KV dtype or storage")
        if self.storage == "paged" and self.page_size == 0:
            raise ProtocolError("paged KV requires a positive page size")

    @property
    def layout_id(self):
        text = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def compatible(self, other):
        """/** 比较数据语义；TP 规模和本地存储差异交给映射器处理。 */"""
        return all(getattr(self, key) == getattr(other, key)
                   for key in ("model", "layers", "kv_heads", "head_dim", "dtype"))


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
        """/**
         * 校验一块 KV 的逻辑矩形及底层分配范围，不打开远端内存。
         * 数据语义为 FP16/BF16/FP32 [token, head, head_dim]；最后一维连续，
         * 前两维以字节步长描述，分页 KV 每个物理页单独形成 region。
         * @return 正常返回表示逻辑范围与字节范围有效，否则抛 ProtocolError。
         */"""
        integer(self.rank, "rank", 0, layout.tp_size - 1)
        identifier(self.device, "GPU UUID")
        integer(self.layer, "layer", 0, layout.layers - 1)
        integer(self.token_begin, "token_begin", begin, end - 1)
        integer(self.token_end, "token_end", self.token_begin + 1, end)
        integer(self.head_begin, "head_begin", 0, layout.kv_heads - 1)
        integer(self.head_end, "head_end", self.head_begin + 1, layout.kv_heads)
        integer(self.allocation_bytes, "allocation_bytes", 1)
        integer(self.offset, "offset", 0, self.allocation_bytes - 1)
        element = DTYPE_BYTES[layout.dtype]
        width = layout.head_dim * element
        integer(self.token_stride, "token_stride", width)
        integer(self.head_stride, "head_stride", width)
        if self.component not in ("k", "v"):
            raise ProtocolError("KV component must be k or v")
        ipc_handle(self.memory)
        if any(value % element for value in (self.offset, self.token_stride, self.head_stride)):
            raise ProtocolError("misaligned KV region")
        tokens, heads = self.token_end - self.token_begin, self.head_end - self.head_begin
        if not (self.token_stride >= (heads - 1) * self.head_stride + width or
                self.head_stride >= (tokens - 1) * self.token_stride + width):
            raise ProtocolError("overlapping KV rows")
        last = self.offset + (tokens - 1) * self.token_stride + (heads - 1) * self.head_stride + width
        if last > self.allocation_bytes:
            raise ProtocolError("KV region exceeds exported allocation")

    def byte_offset(self, token, head):
        return self.offset + (token - self.token_begin) * self.token_stride + (head - self.head_begin) * self.head_stride


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
        if type(self.version) is not int or self.version != VERSION:
            raise ProtocolError("unsupported PD protocol version")
        if not isinstance(self.identity, Identity) or not isinstance(self.layout, Layout):
            raise ProtocolError("invalid identity or layout")
        if self.owner not in (self.identity.source, self.identity.target):
            raise ProtocolError("manifest owner does not belong to this transfer")
        integer(self.token_begin, "token_begin")
        integer(self.token_end, "token_end", self.token_begin + 1)
        if not isinstance(self.prefix_digest, str) or not re.fullmatch(r"[a-f0-9]{64}", self.prefix_digest):
            raise ProtocolError("invalid token prefix digest")
        if not isinstance(self.regions, tuple) or not 1 <= len(self.regions) <= MAX_REGIONS:
            raise ProtocolError("invalid region count")
        for region in self.regions:
            if not isinstance(region, Region):
                raise ProtocolError("invalid KV region")
            region.validate(self.layout, self.token_begin, self.token_end)

    def to_json(self):
        payload = json.dumps(asdict(self), ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        if len(payload.encode()) > MAX_META_BYTES:
            raise ProtocolError("KV manifest exceeds metadata limit")
        return payload

    @classmethod
    def from_json(cls, payload):
        """/** 解析有界 manifest；所有嵌套身份和范围在返回前校验。 */"""
        if not isinstance(payload, (str, bytes)) or len(payload) > MAX_META_BYTES:
            raise ProtocolError("invalid metadata size")
        def unique(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ProtocolError(f"duplicate metadata key: {key}")
                value[key] = item
            return value
        try:
            value = json.loads(payload, object_pairs_hook=unique)
            identity = value["identity"]
            identity["source"] = _object(Peer, identity["source"])
            identity["target"] = _object(Peer, identity["target"])
            value["identity"] = _object(Identity, identity)
            value["owner"] = _object(Peer, value["owner"])
            value["layout"] = _object(Layout, value["layout"])
            value["regions"] = tuple(_object(Region, item) for item in value["regions"])
            return _object(cls, value)
        except (KeyError, TypeError, json.JSONDecodeError, RecursionError) as error:
            raise ProtocolError(f"invalid KV manifest: {error}") from error


def token_digest(tokens):
    """/** 按明确的 uint32 小端 Token 序列计算前缀身份，不使用进程随机 hash。 */"""
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(integer(token, "token", 0, 0xffffffff).to_bytes(4, "little"))
    return digest.hexdigest()


def prefix_manifest(manifest, identity, owner, tokens):
    """/** 将已持有的导出引用绑定到新交接的公共前缀；不改变底层地址及占用。 */"""
    from dataclasses import replace
    end = len(tokens)
    if manifest.token_begin != 0 or not 0 < end <= manifest.token_end:
        raise ProtocolError("prefix outside exported token range")
    regions = tuple(replace(r, token_end=min(r.token_end, end)) for r in manifest.regions if r.token_begin < end)
    return Manifest(VERSION, identity, owner, manifest.layout, 0, end, token_digest(tokens), regions)


def suffix_manifest(manifest, begin):
    """/** 裁去已在接收端恢复的前缀；保留完整前缀摘要，偏移随逻辑 Token 起点调整。 */"""
    from dataclasses import replace
    integer(begin, "suffix begin", manifest.token_begin, manifest.token_end - 1)
    regions = tuple(replace(r, token_begin=max(begin, r.token_begin),
                            offset=r.byte_offset(max(begin, r.token_begin), r.head_begin))
                    for r in manifest.regions if r.token_end > begin)
    return replace(manifest, token_begin=begin, regions=regions)
