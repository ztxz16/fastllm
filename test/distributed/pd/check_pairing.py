"""正式 C++ 副本配对控制测试；GPU 能力和控制 RPC 为注入桩，不替代模型验收。"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from fastllm_pytools.pd.native import NativeReplica
from fastllm_pytools.pd.protocol import Identity, Peer, Manifest, Region, ProtocolError, token_digest
from fastllm_pytools.pd.service import Service
from report import run


def check(value, reason):
    if not value:
        raise AssertionError(reason)


def rejects(call, message):
    try:
        call()
    except (ProtocolError, ValueError, RuntimeError) as error:
        check(message in str(error), f"wrong rejection: {error}")
        return
    raise AssertionError("expected rejection: " + message)


class Backend:
    def __init__(self, devices, edges):
        self.devices, self.edges, self.submitted = devices, edges, 0

    def topology(self):
        return dict(devices=list(self.devices), access=[list(edge) for edge in self.edges])

    def submit(self, copies):
        self.submitted += 1
        raise AssertionError("reservation/rejection must not submit GPU copies")

    def close(self):
        check(self.submitted == 0, "unexpected GPU submission")


@contextmanager
def services(p_groups=((0,), (2,)), d_groups=((1,), (3,)), *, policy="load", edges=None,
             p_visible=None, d_visible=None, heads=4, target_model="model"):
    visible = [f"gpu{i}" for i in range(4)]
    if edges is None:
        edges = [(visible[a], visible[b]) for a, b in ((0, 1), (1, 0), (2, 3), (3, 2))]
    peers = [Peer("p", "p-boot", "host"), Peer("d", "d-boot", "host")]
    objects = []
    calls = []
    config = dict(model_type="qwen3", num_hidden_layers=2, num_key_value_heads=heads,
                  hidden_size=8, num_attention_heads=4, vocab_size=100)
    for role, peer, groups, devices in zip(("p", "d"), peers, (p_groups, d_groups),
                                          (p_visible or visible, d_visible or visible)):
        replicas = []
        for group in groups:
            session = SimpleNamespace(devices=list(group), page_size=128, storage="paged", state="ready")
            replicas.append(NativeReplica(session, peer, config,
                                          target_model if role == "d" else "model"))
        objects.append(Service(role, peer, replicas, token="test", timeout=5, routing=policy,
                               backend=Backend(devices, edges)))
    p, d = objects

    def rpc(url, path, body=None, **kwargs):
        calls.append(path)
        service = {"p-url": p, "d-url": d}[url]
        method = path.removeprefix("/internal/")
        if method == "cancel":
            from fastllm_pytools.pd.service import identity_from
            return service.cancel(identity_from(body["identity"]))
        return getattr(service, method)(body)

    with patch("fastllm_pytools.pd.http.rpc", rpc):
        try:
            yield p, d, calls
        finally:
            p.close()
            d.close()


def body(p, d, mode="pull", request="a", length=2):
    identity = Identity(request, "transfer-" + request, p.peer, d.peer, "p_to_d", mode)
    return dict(identity=asdict(identity), request=dict(token_ids=[1] * length, max_new_tokens=8,
                eos_token_ids=[]), peer_url="d-url", source_url="p-url", expires_at=time.time() + 60)


def record(service, payload):
    return service.records[payload["identity"]["transfer"]]


def selection(mode, policy, length=2):
    with services(policy=policy) as (p, d, calls):
        # 故意令 D 的游标与 P 相反；仍必须选到 P2P 可达的组合。
        d.cursor = 1
        results = []
        for index in range(4):
            payload = body(p, d, mode, str(index), length)
            pair = p.pair(payload)["pair"]
            check(pair["source"]["index"] == pair["target"]["index"], "independent DP mispair")
            check(record(p, payload).pair == record(d, payload).pair, "pair not bound on both ends")
            check(record(p, payload).manifest is None and record(d, payload).manifest is None,
                  "pairing allocated KV")
            check(record(p, payload).long_prompt == (length >= 512), "long Prompt classification")
            results.append(pair["source"]["index"])
        if policy != "prefix":
            check(results == [0, 1, 0, 1], "load/round robin ignored outstanding reservations")
        return dict(pairs=results, reservations=4, prefill_calls=calls.count("/internal/prefill"))


def topology_case(p_groups, d_groups, expected, *, edges=None, heads=4):
    with services(p_groups, d_groups, edges=edges, heads=heads) as (p, d, calls):
        payload = body(p, d)
        if expected:
            pair = p.pair(payload)["pair"]
            return dict(source_ranks=len(pair["source"]["ranks"]), target_ranks=len(pair["target"]["ranks"]))
        rejects(lambda: p.pair(payload), "no reachable replica pair")
        check(not d.records and record(p, payload).replica is None, "rejection reserved a replica")
        check(not p.failure and not d.failure, "unsupported pairing poisoned service health")
        return dict(rejected=True, kv_submissions=0)


def source_filter():
    with services(p_groups=((0,), (2,)), d_groups=((3,),)) as (p, d, _):
        # 不可达的 P0 评分更低，也应先被过滤。
        p.replicas[1].active = "busy"
        pair = p.pair(body(p, d))["pair"]
        check(pair["source"]["index"] == 1, "P load selected unreachable replica")


def target_load():
    edges = [(f"gpu{i}", f"gpu{j}") for i in range(4) for j in range(4)]
    with services(p_groups=((0,),), edges=edges) as (p, d, _):
        d.replicas[0].active = "busy"
        pair = p.pair(body(p, d))["pair"]
        check(pair["target"]["index"] == 1, "D ignored load within reachable candidates")


def concurrent():
    with services() as (p, d, _):
        payloads = [body(p, d, request=str(i)) for i in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            pairs = list(pool.map(p.pair, payloads))
        counts = [sum(x["pair"]["source"]["index"] == i for x in pairs) for i in range(2)]
        check(counts == [4, 4], "concurrent reservations missing from load")
        check(all(x["pair"]["source"]["index"] == x["pair"]["target"]["index"] for x in pairs),
              "concurrent request lost its pair")
        return dict(concurrency=4, requests=8, allocations=0)


def replay(change=None):
    with services() as (p, d, _):
        payload = body(p, d)
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(p.pair, [deepcopy(payload), deepcopy(payload)]))
        check(result[0] == result[1] and len(d.records) == 1, "duplicate pairing selected twice")
        saved = deepcopy(result[0]["pair"])
        if change == "url":
            payload["source_url"] = "wrong-url"
            rejects(lambda: p.pair(payload), "P control URL changed")
        elif change == "request":
            payload["request"]["token_ids"] = [2, 2]
            rejects(lambda: p.pair(payload), "changed data")
        elif change == "pair":
            payload["pair"] = deepcopy(saved)
            payload["pair"]["target"]["index"] = 1
            rejects(lambda: p.create(payload), "pair missing or changed")
        else:
            check(p.pair(payload)["pair"] == saved, "replay changed pair")
        check(record(p, payload).pair == saved and not record(p, payload).preparing,
              "invalid replay modified reservation or started compute")


def terminal(expire=False):
    with services() as (p, d, _):
        payload = body(p, d)
        paired = p.pair(payload)
        r = record(p, payload)
        if expire:
            r.expires_at = time.time() - 1
        else:
            p.cancel(r.identity)
        rejects(lambda: p.pair(payload), "cancelled, expired")
        payload["pair"] = paired["pair"]
        rejects(lambda: p.create(payload), "cancellation or expiry")
        check(not r.preparing and r.manifest is None, "terminal reservation started model work")


def stale():
    with services() as (p, d, _):
        payload = body(p, d)
        payload["identity"]["target"]["boot"] = "old"
        rejects(lambda: p.pair(payload), "invalid topology recipient")
        check(not d.records, "old boot created D reservation")


def remapped():
    with services(((1,),), ((0,),), p_visible=["gpu3", "gpu2"],
                  d_visible=["gpu3", "gpu2"]) as (p, d, _):
        pair = p.pair(body(p, d))["pair"]
        check(pair["source"]["ranks"][0]["device"] == "gpu2" and
              pair["target"]["ranks"][0]["device"] == "gpu3", "logical ID confused with UUID")


def incompatible():
    with services(target_model="other") as (p, d, _):
        rejects(lambda: p.pair(body(p, d)), "no reachable replica pair")
        check(not d.records, "model mismatch allocated D reservation")


def unpaired():
    with services() as (p, d, _):
        rejects(lambda: p.create(body(p, d)), "pair missing or changed")
        check(all(not r.preparing for r in p.records.values()), "unpaired DP entered computation")


def cancel_during_pair():
    from fastllm_pytools.pd import http
    with services() as (p, d, _):
        payload = body(p, d)
        original = http.rpc

        def delayed(url, path, *args, **kwargs):
            result = original(url, path, *args, **kwargs)
            if path == "/internal/reserve":
                p.cancel(record(p, payload).identity)
            return result

        with patch("fastllm_pytools.pd.http.rpc", delayed):
            rejects(lambda: p.pair(payload), "cancelled, expired")
        check(record(p, payload).pair is None and record(p, payload).released,
              "late reservation revived cancelled P")
        d.cancel(record(d, payload).identity)
        check(record(d, payload).released, "target reservation not released")


def create_during_pair(side):
    from fastllm_pytools.pd import http
    with services(((0,),), ((1,),)) as (p, d, _):
        payload = body(p, d)
        receiver = dict(payload, peer_url="p-url")
        rejected = []
        original = http.rpc
        topology = d.transfers.backend.topology

        def concurrent_rpc(url, path, *args, **kwargs):
            if side == "p" and path == "/internal/topology":
                rejects(lambda: p.create(payload), "pair missing or changed")
                rejected.append(True)
            return original(url, path, *args, **kwargs)

        def concurrent_topology():
            if side == "d" and d.records:
                rejects(lambda: d.create(receiver), "pair missing or changed")
                rejected.append(True)
            return topology()

        with patch("fastllm_pytools.pd.http.rpc", concurrent_rpc), \
                patch.object(d.transfers.backend, "topology", concurrent_topology):
            p.pair(payload)
        check(rejected == [True], "preparation race was not exercised")
        check(not record(p, payload).preparing and not record(d, payload).preparing,
              "legacy single-replica preparation bypassed incomplete pair")


def fake_manifest(r, peer, side):
    from fastllm_pytools.pd.protocol import Layout
    description = r.pair[side]
    layout = Layout(**description["layout"])
    regions = []
    for shard in description["ranks"]:
        width = (shard["head_end"] - shard["head_begin"]) * layout.head_dim * 2
        for layer in range(layout.layers):
            for component in ("k", "v"):
                regions.append(Region(shard["rank"], shard["device"], layer, component, 0, 2,
                    shard["head_begin"], shard["head_end"], base64.b64encode(bytes(64)).decode(),
                    width * 2, 0, width, layout.head_dim * 2))
    return Manifest(1, r.identity, peer, layout, 0, 2, token_digest([1, 1]), tuple(regions))


def wrong_manifest(mode, field):
    with services() as (p, d, _):
        payload = body(p, d, mode)
        p.pair(payload)
        pr, dr = record(p, payload), record(d, payload)
        source, target = fake_manifest(pr, p.peer, "source"), fake_manifest(dr, d.peer, "target")
        chosen = source if mode == "pull" else target
        region = chosen.regions[0]
        replacement = replace(region, **({"device": "wrong-gpu"} if field == "gpu" else {"head_end": 3}))
        wrong = replace(chosen, regions=(replacement, *chosen.regions[1:]))
        # 协议合法但副本/分片错配，必须在读取租约及 GPU 提交前拒绝。
        dr.manifest = target
        try:
            rejects(lambda: d.pull(wrong) if mode == "pull" else p.register_target(wrong),
                    "GPU/head range differs")
            check(dr.remote is None and pr.remote is None and pr.readers == 0,
                  "rejection changed remote binding")
        finally:
            dr.manifest = None  # 注入的描述不持有真实资源；退出仅收尾调度预留。


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = [(f"{mode}_{policy}", lambda m=mode, p=policy: selection(m, p))
             for mode in ("pull", "push") for policy in ("load", "round_robin", "prefix")]
    cases += [(f"{mode}_long_prompt", lambda m=mode: selection(m, "load", 513)) for mode in ("pull", "push")]
    cases += [(f"{mode}_reject_{field}", lambda m=mode, f=field: wrong_manifest(m, f))
              for mode in ("pull", "push") for field in ("gpu", "head")]
    cases += [
        ("tp2_corresponding_edges", lambda: topology_case(((0, 2),), ((1, 3),), True)),
        ("tp1_tp2_missing_edge", lambda: topology_case(((0,),), ((1, 3),), False)),
        ("tp2_tp1_missing_edge", lambda: topology_case(((0, 2),), ((1,),), False)),
        ("tp1_tp2_split", lambda: topology_case(((0,),), ((1, 3),), True, edges=[("gpu0", "gpu1"), ("gpu0", "gpu3")])),
        ("tp2_tp1_merge", lambda: topology_case(((0, 2),), ((1,),), True, edges=[("gpu0", "gpu1"), ("gpu2", "gpu1")])),
        ("replicated_head_source_rank", lambda: topology_case(((0, 2),), ((1,),), True, heads=1, edges=[("gpu0", "gpu1")])),
        ("directional_edge_rejected", lambda: topology_case(((0,),), ((1,),), False, edges=[("gpu1", "gpu0")])),
        ("p_filter_before_load", source_filter), ("d_load_after_filter", target_load),
        ("concurrent_reservations", concurrent), ("duplicate_pair", replay),
        ("changed_url", lambda: replay("url")), ("changed_request", lambda: replay("request")),
        ("changed_pair", lambda: replay("pair")), ("cancelled_reservation", terminal),
        ("expired_reservation", lambda: terminal(True)), ("stale_boot", stale),
        ("logical_device_remapped", remapped), ("incompatible_model", incompatible),
        ("unpaired_dp_rejected", unpaired),
        ("cancel_during_pair", cancel_during_pair),
        ("p_create_during_pair", lambda: create_during_pair("p")),
        ("d_create_during_pair", lambda: create_during_pair("d")),
    ]
    return run(cases, args.output, backend="C++ Service/NativeReplica; injected GPU topology and RPC",
               scope="35 个正式 C++ 副本配对控制用例；不包含真实 CUDA、模型计算或并行组合验收")


if __name__ == "__main__":
    raise SystemExit(main())
