#include "pd_service.h"

namespace fastllm {
namespace pd {
namespace {

/** 检查交接仍可接受调度预留；不申请 KV，也不提交模型计算。 */
void Live(O s, O r) {
    protocol(!cancelled(r) && !flag(r, "released") && !flag(s, "failure") &&
                 !truth(get(s, "closed").attr("is_set")()) && real(get(r, "expires_at")) > wall(),
             "replica reservation cancelled, expired or service unavailable");
}

/**
 * 按正式 TP 分头规则生成各 rank 的逻辑范围；EP view 的 TP 大小为一。
 * @param heads 模型的全局 KV head 数。
 * @param count 副本内部 TP rank 数。
 * @param rank 副本内部 rank 编号。
 * @return 全局 KV head 半开区间；TP 大于 KV head 数时复制对应 head。
 */
std::pair<I, I> Heads(I heads, I count, I rank) {
    if (count <= heads) {
        I begin = heads / count * rank + std::min(rank, heads % count);
        return {begin, begin + heads / count + (rank < heads % count)};
    }
    I offset = 0;
    for (I head = 0; head < heads; ++head) {
        I width = (count - offset + heads - head - 1) / (heads - head);
        if (rank < offset + width)
            return {head, head + 1};
        offset += width;
    }
    fail("invalid TP head assignment");
    return {};
}

/**
 * 从服务实际使用的本地 CUDA 编号构造 UUID 拓扑，避免跨进程比较逻辑编号。
 * @param s 当前服务，backend 查询仅返回设备能力，不打开 IPC 映射。
 * @return 启动身份、所有副本的分头描述和本进程可执行的有向拷贝边。
 */
O Snapshot(O s) {
    O topology = get(get(s, "transfers"), "backend").attr("topology")();
    O visible = at(topology, "devices");
    L replicas;
    I index = 0;
    for (auto replica : get(s, "replicas")) {
        O layout = get(replica, "layout"), devices = get(get(replica, "session"), "devices");
        I count = py::len(devices), heads = num(get(layout, "kv_heads"));
        protocol(count > 0 && count == num(get(layout, "tp_size")), "invalid replica TP topology");
        L ranks;
        for (I rank = 0; rank < count; ++rank) {
            I device = integer(item(devices, py::int_(rank)), "CUDA device", 0, I(py::len(visible)) - 1);
            auto range = Heads(heads, count, rank);
            ranks.append(D("rank"_a = rank, "device"_a = item(visible, py::int_(device)),
                           "head_begin"_a = range.first, "head_end"_a = range.second));
        }
        replicas.append(D("index"_a = index++, "layout"_a = asdict(layout), "ranks"_a = ranks));
    }
    return D("peer"_a = asdict(get(s, "peer")), "replicas"_a = replicas,
             "access"_a = at(topology, "access"));
}

/**
 * 校验远端拓扑的启动身份和分头结构；只接受受支持的完整副本描述。
 * @param topology 认证控制通道传来的拓扑快照，不包含地址或 IPC handle。
 * @param peer 本次交接中预期的服务启动身份。
 * @return 经校验的副本列表，index 与列表位置一致。
 */
O Validate(O topology, O peer) {
    protocol(equal(at(topology, "peer"), asdict(peer)), "stale topology peer or boot");
    O replicas = at(topology, "replicas"), access = at(topology, "access");
    protocol(py::isinstance<L>(replicas) && py::len(replicas) > 0 && py::len(replicas) <= 1024 &&
                 py::isinstance<L>(access) && py::len(access) <= 65536, "invalid topology size");
    I index = 0;
    for (auto replica : replicas) {
        protocol(integer(at(replica, "index"), "replica index") == index++, "invalid replica index");
        O layout = cls("protocol", "_object")(cls("protocol", "Layout"), at(replica, "layout"));
        I count = num(get(layout, "tp_size")), heads = num(get(layout, "kv_heads"));
        O ranks = at(replica, "ranks");
        protocol(py::isinstance<L>(ranks) && I(py::len(ranks)) == count, "invalid topology ranks");
        for (I rank = 0; rank < count; ++rank) {
            O shard = item(ranks, py::int_(rank));
            auto range = Heads(heads, count, rank);
            protocol(integer(at(shard, "rank"), "rank") == rank &&
                         integer(at(shard, "head_begin"), "head begin") == range.first &&
                         integer(at(shard, "head_end"), "head end") == range.second,
                     "topology head assignment mismatch");
            identifier(at(shard, "device"), "GPU UUID");
        }
    }
    for (auto edge : access) {
        protocol(py::isinstance<L>(edge) && py::len(edge) == 2, "invalid P2P edge");
        identifier(item(edge, py::int_(0)), "source UUID");
        identifier(item(edge, py::int_(1)), "target UUID");
    }
    return replicas;
}

/**
 * 按正式映射器的最小源 rank 规则检查实际需要的边，不要求 TP 全互通。
 * @param source P 副本布局及 rank 描述。
 * @param target D 副本布局及 rank 描述。
 * @param access 发起传输进程可执行的 [源 UUID, 目标 UUID] 列表。
 * @return 每个目标 head 均有来源且可达时为 true；不修改调度或 KV 状态。
 */
bool Reachable(O source, O target, O access) {
    O from = cls("protocol", "_object")(cls("protocol", "Layout"), at(source, "layout"));
    O to = cls("protocol", "_object")(cls("protocol", "Layout"), at(target, "layout"));
    if (!truth(from.attr("compatible")(to)))
        return false;
    std::set<std::pair<std::string, std::string>> edges;
    for (auto edge : access)
        edges.emplace(str(item(edge, py::int_(0))), str(item(edge, py::int_(1))));
    for (auto dst : at(target, "ranks")) {
        I head = num(at(dst, "head_begin")), end = num(at(dst, "head_end"));
        while (head < end) {
            bool found = false;
            for (auto src : at(source, "ranks")) {
                if (num(at(src, "head_begin")) <= head && num(at(src, "head_end")) > head) {
                    if (!edges.count({str(at(src, "device")), str(at(dst, "device"))}))
                        return false;
                    head = std::min(end, num(at(src, "head_end")));
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }
    }
    return true;
}

/** 返回已选择副本在快照中的描述；调用者须持有服务锁，防止重复选择。 */
O Select(O s, O r, O candidates, O replicas) {
    O chosen = s.attr("_select_replica")(r, candidates);
    for (auto index : candidates)
        if (item(get(s, "replicas"), index).is(chosen))
            return item(replicas, index);
    fail("reserved replica no longer belongs to reachable candidates");
    return none();
}
} // namespace

/**
 * 在登记远端内存或提交传输前，将实际 manifest 与已绑定副本逐 rank 核对。
 * @param record 当前交接记录；旧的单副本直接控制入口可无 pair。
 * @param manifest 已经通过协议解析的实际 KV 描述。
 * @param side source 或 target，指定应使用配对中的哪一端。
 */
void ValidatePairManifest(O record, O manifest, const char *side) {
    O pair = get(record, "pair");
    if (pair.is_none())
        return;
    O expected = at(pair, side), ranks = at(expected, "ranks");
    protocol(equal(get(manifest, "identity"), recordIdentity(record)) &&
                 equal(asdict(get(manifest, "layout")), at(expected, "layout")),
             "manifest does not match reserved layout or identity");
    for (auto region : get(manifest, "regions")) {
        I rank = integer(get(region, "rank"), "manifest rank", 0, I(py::len(ranks)) - 1);
        O shard = item(ranks, py::int_(rank));
        protocol(equal(get(region, "device"), at(shard, "device")) &&
                     equal(get(region, "head_begin"), at(shard, "head_begin")) &&
                     equal(get(region, "head_end"), at(shard, "head_end")),
                 "manifest GPU/head range differs from reserved replica");
    }
}

/** 注册拓扑预检和双端调度预留；所有决策在 C++，HTTP 仅承载数据。 */
void BindPairing(Handlers &h) {
    h["service.Service.topology"] = [](D a) -> O {
        O s = self(a), id = cls("service", "identity_from")(at(at(a, "body"), "identity"));
        protocol(str(get(s, "role")) == "d" && equal(get(id, "target"), get(s, "peer")) &&
                     equal(get(get(id, "source"), "host"), get(get(s, "peer"), "host")),
                 "invalid topology recipient or host");
        return Snapshot(s);
    };

    h["service.Service.reserve"] = [](D a) -> O {
        O s = self(a), body = at(a, "body");
        protocol(str(get(s, "role")) == "d", "only D accepts target reservation");
        O r;
        {
            Lock lock(get(s, "lock"));
            r = item(s.attr("_new")(body), py::int_(0));
            Live(s, r);
            protocol(!get(r, "pair").is_none() ||
                         (!flag(r, "preparing") && get(r, "replica").is_none()),
                     "target already preparing");
            protocol(py::isinstance<py::str>(get(r, "peer_url")) && flag(r, "peer_url"),
                     "missing P control URL");
            // 配对未完成时，也禁止旧的单副本 create 入口抢先申请 KV。
            put(r, "pair_source_url", get(r, "peer_url"));
        }
        O id = recordIdentity(r);
        O remote = at(body, "source_topology"), sources = Validate(remote, get(id, "source"));
        I sourceIndex = integer(at(body, "source_index"), "source replica", 0, I(py::len(sources)) - 1);
        O source = item(sources, py::int_(sourceIndex)), local = Snapshot(s);
        O targets = Validate(local, get(id, "target"));
        O access = at(str(get(id, "mode")) == "pull" ? local : remote, "access");
        L candidates;
        for (auto target : targets)
            if (Reachable(source, py::reinterpret_borrow<O>(target), access))
                candidates.append(at(target, "index"));
        Lock lock(get(s, "lock"));
        Live(s, r);
        O pair = get(r, "pair");
        if (!pair.is_none()) {
            protocol(equal(at(pair, "source"), source) &&
                         truth(candidates.attr("__contains__")(at(at(pair, "target"), "index"))),
                     "source or reachability changed on reservation replay");
            return D("pair"_a = pair);
        }
        protocol(!flag(r, "preparing") && get(r, "replica").is_none(), "target already preparing");
        O target = Select(s, r, candidates, targets);
        pair = D("source"_a = source, "target"_a = target);
        put(r, "pair", pair);
        return D("pair"_a = pair);
    };

    h["service.Service.pair"] = [](D a) -> O {
        O s = self(a), body = at(a, "body");
        protocol(str(get(s, "role")) == "p", "only P coordinates replica pairing");
        O r = item(s.attr("_new")(body), py::int_(0)), id = recordIdentity(r);
        // 同一交接串行协商，绝不持有整个服务的锁等待远端 RPC。
        Lock pairing(get(r, "pair_lock"));
        {
            Lock lock(get(s, "lock"));
            Live(s, r);
            O sourceUrl = at(body, "source_url");
            protocol(py::isinstance<py::str>(sourceUrl) && truth(sourceUrl), "missing P control URL");
            protocol(!flag(r, "pair_source_url") || equal(get(r, "pair_source_url"), sourceUrl),
                     "P control URL changed on pair replay");
            if (!get(r, "pair").is_none())
                return D("pair"_a = get(r, "pair"));
            protocol(!flag(r, "preparing"), "source already preparing");
            put(r, "pair_source_url", sourceUrl);
        }
        O remote = rpc(s, get(r, "peer_url"), "/internal/topology", fields(r),
                       std::max(0.001, deadline(s, r) - now()));
        O targets = Validate(remote, get(id, "target")), local = Snapshot(s);
        O sources = Validate(local, get(id, "source"));
        O access = at(str(get(id, "mode")) == "pull" ? remote : local, "access");
        L candidates;
        for (auto source : sources)
            for (auto target : targets)
                if (Reachable(py::reinterpret_borrow<O>(source), py::reinterpret_borrow<O>(target), access)) {
                    candidates.append(at(source, "index"));
                    break;
                }
        O source;
        {
            Lock lock(get(s, "lock"));
            Live(s, r);
            protocol(!flag(r, "preparing"), "source started preparing during pair negotiation");
            source = Select(s, r, candidates, sources);
        }
        D reserve("identity"_a = asdict(id), "request"_a = get(r, "request"),
                  "peer_url"_a = at(body, "source_url"), "expires_at"_a = get(r, "expires_at"),
                  "source_topology"_a = local, "source_index"_a = at(source, "index"));
        O response = rpc(s, get(r, "peer_url"), "/internal/reserve", reserve,
                         std::max(0.001, deadline(s, r) - now()));
        O pair = at(response, "pair"), target = at(pair, "target");
        I targetIndex = integer(at(target, "index"), "target replica", 0, I(py::len(targets)) - 1);
        protocol(equal(at(pair, "source"), source) && equal(target, item(targets, py::int_(targetIndex))) &&
                     Reachable(source, target, access), "invalid reserved replica pair");
        {
            Lock lock(get(s, "lock"));
            Live(s, r);
            put(r, "pair", pair);
        }
        return D("pair"_a = pair);
    };
}
} // namespace pd
} // namespace fastllm
