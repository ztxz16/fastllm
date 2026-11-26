"""按逻辑 Token/KV head 区间映射 TP 分片与分页/连续存储。"""

from dataclasses import dataclass
from .protocol import DTYPE_BYTES, Manifest, ProtocolError, Region


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
        return (self.token_end - self.token_begin) * (self.head_end - self.head_begin) * self.vector_bytes


def plan_transfer(source: Manifest, target: Manifest):
    """/**
     * 生成完整且无重复目标写入的 KV 拷贝计划，提交前发现缺片或不兼容。
     * 目标副本由调度器提前确定；本函数只在已选 TP 组之间切分、汇集和复制。
     * 同一 KV head 的源复制只选择一个来源；目标复制分别填充。
     * @param source 已校验的源 manifest，实际数据就绪由传输模块另行判断。
     * @param target 由本地接收端分配生成的 manifest，禁止采用未经确认的远端地址。
     * @return 不可变 Copy 序列；目标缺口、重叠或身份不符时抛异常，无 GPU 副作用。
     */"""
    if (source.identity != target.identity or source.owner != source.identity.source or
            target.owner != target.identity.target):
        raise ProtocolError("source/target transfer identity mismatch")
    if not source.layout.compatible(target.layout) or source.prefix_digest != target.prefix_digest:
        raise ProtocolError("model, KV semantics or prompt prefix mismatch")
    if (source.token_begin, source.token_end) != (target.token_begin, target.token_end):
        raise ProtocolError("source/target logical token range mismatch")

    def overlaps(a, b):
        return (a.token_begin < b.token_end and b.token_begin < a.token_end and
                a.head_begin < b.head_end and b.head_begin < a.head_end)

    by_key = {}
    for region in source.regions:
        by_key.setdefault((region.layer, region.component), []).append(region)
    targets = {}
    writes = []
    width = target.layout.head_dim * DTYPE_BYTES[target.layout.dtype]
    for dst in target.regions:
        key = (dst.rank, dst.layer, dst.component)
        previous = targets.setdefault(key, [])
        if any(overlaps(dst, other) for other in previous):
            raise ProtocolError("overlapping target logical regions")
        previous.append(dst)
        candidates = sorted((s for s in by_key.get((dst.layer, dst.component), []) if overlaps(s, dst)),
                            key=lambda s: (s.rank, s.token_begin, s.head_begin))
        token_edges = sorted({dst.token_begin, dst.token_end} | {
            x for s in candidates for x in (max(s.token_begin, dst.token_begin), min(s.token_end, dst.token_end))})
        head_edges = sorted({dst.head_begin, dst.head_end} | {
            x for s in candidates for x in (max(s.head_begin, dst.head_begin), min(s.head_end, dst.head_end))})
        for tb, te in zip(token_edges, token_edges[1:]):
            for hb, he in zip(head_edges, head_edges[1:]):
                src = next((s for s in candidates if s.token_begin <= tb and s.token_end >= te
                            and s.head_begin <= hb and s.head_end >= he), None)
                if src is None:
                    raise ProtocolError(f"missing KV layer={dst.layer} {dst.component} token={tb}:{te} head={hb}:{he}")
                writes.append(Copy(src, dst, tb, te, hb, he, width))
    # 所有目标层与 K/V 均须覆盖完整 Token 范围和全局 head；不以“已有分片传完”冒充完整。
    for layer in range(target.layout.layers):
        for component in ("k", "v"):
            regions = [r for r in target.regions if r.layer == layer and r.component == component]
            edges = sorted({target.token_begin, target.token_end} | {
                edge for r in regions for edge in (r.token_begin, r.token_end)})
            for begin, end in zip(edges, edges[1:]):
                intervals = sorted((r.head_begin, r.head_end) for r in regions
                                   if r.token_begin <= begin and r.token_end >= end)
                covered = 0
                for left, right in intervals:
                    if left > covered:
                        break
                    covered = max(covered, right)
                if covered != target.layout.kv_heads:
                    raise ProtocolError(f"incomplete target layout layer={layer} {component}")
    return tuple(writes)
