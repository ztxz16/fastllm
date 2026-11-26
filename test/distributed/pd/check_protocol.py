"""协议、TP 映射、路由、保留期和异步取消的 CPU 用例；不替代 GPU 验收。"""

import argparse
import base64
from dataclasses import replace
import json
from pathlib import Path
import queue
import sys
import threading
import time
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from fastllm_pytools.pd.protocol import Identity, Peer, Layout, Region, Manifest, ProtocolError, token_digest
from fastllm_pytools.pd.mapping import plan_transfer
from fastllm_pytools.pd.tasks import TransferManager, State
from fastllm_pytools.pd.routing import Registry, POLICIES
from fastllm_pytools.pd.retry import RetryBudget
from fastllm_pytools.pd.history import History
from fastllm_pytools.pd.admission import Admission
from fastllm_pytools.parallel_config import is_attention_ep
from report import run

P, D = Peer("p", "boot-p", "host"), Peer("d", "boot-d", "host")
IDENTITY = Identity("request", "transfer", P, D, "p_to_d", "pull")


def check(condition, reason):
    if not condition:
        raise AssertionError(reason)


def rejects(function):
    try:
        function()
    except (ProtocolError, ValueError, RuntimeError):
        return
    raise AssertionError("expected explicit rejection")


def manifest(peer, tp, *, mode="pull", paged=False):
    layout = Layout("model", 2, 4, 2, "float16", tp, "paged" if paged else "continuous", 2 if paged else 4)
    regions = []
    for rank in range(tp):
        hb, he = (rank * 4 // tp, (rank + 1) * 4 // tp) if tp <= 4 else (rank // 2, rank // 2 + 1)
        for layer in range(2):
            for component in ("k", "v"):
                for begin in range(0, 4, layout.page_size):
                    end = min(begin + layout.page_size, 4)
                    handle = bytes([rank, layer, component == "v", begin, peer == D]) + bytes(59)
                    regions.append(Region(rank, "gpu" + str(rank), layer, component, begin, end, hb, he,
                        base64.b64encode(handle).decode(), (end - begin) * (he - hb) * 4, 0, (he - hb) * 4, 4))
    return Manifest(1, replace(IDENTITY, mode=mode), peer, layout, 0, 4, token_digest([1, 2, 3, 4]), tuple(regions))


def mapping_case(p, d, paged, mode):
    source, target = manifest(P, p, mode=mode, paged=paged), manifest(D, d, mode=mode, paged=not paged)
    copies = plan_transfer(source, target)
    actual = {}
    for copy in copies:
        for token in range(copy.token_begin, copy.token_end):
            for head in range(copy.head_begin, copy.head_end):
                key = (copy.target.rank, copy.target.layer, copy.target.component, token, head)
                check(key not in actual, "duplicate destination write")
                actual[key] = (copy.source.layer, copy.source.component, token, head)
    expected = {(r.rank, r.layer, r.component, t, h): (r.layer, r.component, t, h)
                for r in target.regions for t in range(r.token_begin, r.token_end) for h in range(r.head_begin, r.head_end)}
    check(actual == expected, "head/token content mapping changed")
    return dict(copies=len(copies), destination_vectors=len(actual))


class Backend:
    def __init__(self):
        self.started, self.done = threading.Event(), threading.Event()
        self.handles, self.retired = [], []
    def submit(self, copies):
        handle = object()
        self.handles.append(handle)
        self.started.set()
        return handle
    def query(self, handle):
        return self.done.is_set()
    def retire(self, handle):
        self.retired.append(handle)
    def close(self):
        check(len(self.handles) == len(self.retired), "in-flight handle closed")


def cancellation():
    backend, releases = Backend(), []
    manager = TransferManager(D, backend)
    try:
        source, target = manifest(P, 1), manifest(D, 1)
        manager.register(source, target, lambda: releases.append(1))
        manager.publish_ready(IDENTITY, 0, 0)
        check(backend.started.wait(2), "transfer never submitted")
        manager.cancel(IDENTITY)
        check(not releases, "released before transfer completion")
        backend.done.set()
        event = manager.notifications.get(timeout=2)
        check(event.state == State.CANCELLED and releases == [1], "cancel did not drain exactly once")
        manager.cancel(IDENTITY)
        check(releases == [1], "duplicate release")
    finally:
        backend.done.set()
        manager.close(2)


def complete_all_layers():
    backend, releases = Backend(), []
    backend.done.set()
    manager = TransferManager(D, backend)
    try:
        source, target = manifest(P, 1), manifest(D, 1)
        manager.register(source, target, lambda: releases.append(1))
        manager.seal(IDENTITY)
        manager.publish_ready(IDENTITY, 0, 0)
        check(backend.started.wait(2), "first layer not submitted")
        check(manager.status(IDENTITY) != State.READY, "final flag accepted without missing layer")
        manager.publish_ready(IDENTITY, 0, 1)
        check(manager.notifications.get(timeout=2).state == State.READY, "complete layers not accepted")
    finally:
        manager.close(2)


def routing(policy):
    registry = Registry(policy=policy)
    for name in ("p0", "p1"):
        registry.register(name, "http://" + name + ":8000", "p", "model", "boot")
        registry.health_result(name, "boot", True); registry.health_result(name, "boot", True)
    lease = registry.select("p", "model", [1, 2], session="conversation")
    check(lease.worker in ("p0", "p1"), "selected outside healthy role pool")
    registry.release(lease); registry.release(lease)
    check(all(w["inflight"] == 0 for w in registry.snapshot()), "lease released twice")


def health():
    registry = Registry()
    registry.register("p0", "http://p0:8000", "p", "model", "boot")
    for _ in range(2): registry.health_result("p0", "boot", True)
    for _ in range(2): registry.health_result("p0", "boot", False)
    check(registry.snapshot()[0]["healthy"], "removed before third failure")
    registry.health_result("p0", "boot", False)
    check(not registry.snapshot()[0]["healthy"], "third failure not removed")
    registry.health_result("p0", "boot", True)
    check(not registry.snapshot()[0]["healthy"], "recovered after one success")
    registry.health_result("p0", "boot", True)
    check(registry.snapshot()[0]["healthy"], "two successes did not recover")


def retained_lifetime():
    clock, releases = [0.0], []
    history = History(D, 480, clock=lambda: clock[0])
    source = manifest(D, 1)
    history.retain("old", source, [1, 2, 3, 4], lambda: releases.append(1))
    identity = Identity("new", "back", D, P, "d_to_p", "pull")
    check(history.acquire("old", identity, [1, 2, 3, 4, 5]).token_end == 4, "history prefix mismatch")
    clock[0] = 481
    history.expire()
    check(not releases and history.acquire("old", replace(identity, transfer="late"), [1, 2, 3, 4, 5]) is None,
          "expired retention released active reader or accepted new reader")
    history.release("old", identity)
    check(releases == [1], "drained expired retention not released")


def retry():
    budget = RetryBudget(100, attempts=3)
    first = budget.begin(now=0)
    check(budget.delay(status=503, now=0) is not None, "retryable error rejected")
    check(budget.begin(now=1) != first, "retry reused transfer ID")
    budget.responded = True
    check(budget.delay(status=503, now=1) is None, "retried after response")


def capacity():
    pool = Admission(1000, growth=100)
    check(pool.request("a", 800), "first request refused")
    check(not pool.request("b", 100), "overcommitted growth budget")
    pool.ready("a"); pool.pin("a")
    rejects(lambda: pool.retire("a"))
    pool.unpin("a"); pool.retire("a")
    check(pool.request("b", 100), "waiting request did not resume")


def cache_thresholds():
    registry = Registry(policy="cache_aware")
    for name in ("p0", "p1"):
        registry.register(name, "http://" + name, "p", "model", "boot")
        registry.health_result(name, "boot", True)
        registry.health_result(name, "boot", True)
    first = registry.select("p", "model", list(range(10)), session="history")
    chosen = first.worker
    registry.release(first)
    busy = registry.select("p", "model", list(range(10)))
    check(busy.worker == chosen, "prefix affinity not retained under balanced load")
    selected = registry.select("p", "model", [0, 1, 2, 99, 98, 97, 96, 95, 94, 93])
    check(selected.worker != chosen, "30 percent must not count as exceeding 0.3")
    registry.release(selected)
    registry.release(busy)


def stale_boot():
    registry = Registry()
    registry.register("p0", "http://p0", "p", "model", "old")
    registry.register("p0", "http://p0", "p", "model", "new")
    registry.health_result("p0", "old", True)
    registry.health_result("p0", "old", True)
    check(not registry.snapshot()[0]["healthy"], "old boot probe revived new instance")


def target_hole():
    source, target = manifest(P, 1), manifest(D, 1)
    rejects(lambda: plan_transfer(source, replace(target, regions=target.regions[:-1])))


def duplicate_target():
    source, target = manifest(P, 1), manifest(D, 1)
    rejects(lambda: plan_transfer(source, replace(target, regions=target.regions + (target.regions[0],))))


def reject_without_submission():
    backend = Backend()
    manager = TransferManager(D, backend)
    try:
        target = replace(manifest(D, 1), identity=replace(IDENTITY, transfer="wrong"))
        rejects(lambda: manager.register(manifest(P, 1), target, lambda: None))
        check(not backend.handles, "rejection modified target through backend")
    finally:
        manager.close(2)


def retention_shutdown():
    released = []
    history = History(D)
    history.retain("old", manifest(D, 1), [1, 2, 3, 4], lambda: released.append(1))
    identity = Identity("new", "return", D, P, "d_to_p", "pull")
    history.acquire("old", identity, [1, 2, 3, 4, 5])
    check(not history.close() and not released, "shutdown freed an active remote reader")
    check(history.acquire("old", replace(identity, transfer="other"), [1, 2]) is None,
          "shutdown accepted another reader")
    history.release("old", identity)
    check(history.close() and released == [1], "shutdown did not release after reader drained")


def parallel_mode(mode, dp, tp, attention, ep, expected):
    with tempfile.TemporaryDirectory() as path:
        (Path(path) / "config.json").write_text(json.dumps(dict(model_type="qwen2_moe")))
        args = SimpleNamespace(path=path, tp_backend="process", moe_parallel_mode=mode,
                               dp=dp, tp=tp, attention_dp=attention, ep=ep)
        if expected is None:
            rejects(lambda: is_attention_ep(args))
        else:
            check(is_attention_ep(args) == expected, "MoE TP/EP selector routed to wrong executor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = []
    for p, d in ((1, 1), (2, 1), (1, 4), (4, 2), (2, 4), (8, 1)):
        for mode in ("pull", "push"):
            cases.append((f"map_tp{p}_tp{d}_{mode}", lambda p=p, d=d, mode=mode: mapping_case(p, d, True, mode)))
    cases += [("policy_" + policy, lambda policy=policy: routing(policy)) for policy in POLICIES]
    cases += [("moe_mode_" + name, lambda values=values: parallel_mode(*values)) for name, values in (
        ("default", ("auto", 1, "1", 1, 1, True)),
        ("tp1", ("tp", 1, "1", 1, 1, False)),
        ("tp2", ("tp", 1, "0,1", 1, 1, False)),
        ("dp2", ("tp", 2, "1", 1, 1, False)),
        ("ep2", ("ep", 1, "1", 2, 2, True)),
        ("conflict", ("tp", 1, "1", 2, 2, None)))]
    cases += [("cancel_drain", cancellation), ("completion_all_layers", complete_all_layers),
              ("health_thresholds", health), ("history_480_and_inflight", retained_lifetime),
              ("request_retry", retry), ("capacity_growth", capacity),
              ("cache_strict_threshold", cache_thresholds), ("reject_stale_boot", stale_boot),
              ("reject_missing_target_range", target_hole), ("reject_overlapping_target", duplicate_target),
              ("reject_without_side_effect", reject_without_submission), ("history_shutdown_drain", retention_shutdown),
              ("metadata_roundtrip", lambda: check(Manifest.from_json(manifest(P, 2).to_json()) == manifest(P, 2), "roundtrip mismatch")),
              ("reject_wrong_request", lambda: rejects(lambda: plan_transfer(manifest(P, 1),
                  replace(manifest(D, 1), identity=replace(IDENTITY, request="other"))))),
              ("reject_model_mismatch", lambda: rejects(lambda: plan_transfer(manifest(P, 1),
                  replace(manifest(D, 1), layout=replace(manifest(D, 1).layout, model="other")))))]
    return run(cases, args.output, backend="CPU protocol simulation; no CUDA execution")


if __name__ == "__main__":
    raise SystemExit(main())
