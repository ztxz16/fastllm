"""副本证据回归：调用正式 C++ 接口，注入原生状态；不执行 GPU 或模型计算。"""

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from fastllm_pytools.pd.service import Service
from generation_reference import deployment
from report import run


def fixture(size, rank=None, *, cold=False, page_len=128):
    attention = rank is not None
    workers = [dict(rank=i, device=2 * i, pid=100 + i) for i in range(size)]
    ready = dict(backend="qwen_moe_attention_dp_ep" if attention else "qwen3_process_generate_cublas_nccl",
                 workers=workers)
    session = SimpleNamespace(ready=ready, devices=[2 * rank] if attention else [2 * i for i in range(size)],
                              rank=rank, storage="continuous" if attention else "paged")
    states = [dict(enabled=False, allreduce=dict(enabled=False),
                   prefix_cache=dict(enabled=False, page_len=0 if attention or cold else page_len))
              for _ in range(1 if attention else size)]
    calls = []

    def command(action):
        calls.append(action)
        if action == "cache":
            return deepcopy(states)
        if cold and not attention:
            for state in states:
                state["prefix_cache"]["page_len"] = page_len
        return dict(kv_cache_usage=0.0, workers=[])

    replica = SimpleNamespace(session=session, config=dict(model_type="qwen2_moe"), _command=command)
    health = dict(healthy=True, role="full", model="model", peer=dict(instance="full", boot="boot", host="host"))
    service = SimpleNamespace(lock=threading.RLock(), records={}, replicas=[replica], health=lambda: health)
    return service, calls


def snapshot(size, rank=None):
    service, calls = fixture(size, rank)
    original = deepcopy(service.replicas[0].session.ready)
    result = Service.evidence(service)
    replica = result["replicas"][0]
    expected = size if rank is None else 1
    assert len(replica["workers"]) == len(replica["states"]) == len(replica["devices"]) == expected
    assert calls == ["usage", "cache"]
    assert service.replicas[0].session.ready == original, "shared ready metadata was modified"
    if rank is None:
        assert "parallel" not in replica and "group_workers" not in replica
    else:
        assert replica["workers"] == [original["workers"][rank]]
        assert replica["group_workers"] == original["workers"]
        assert replica["parallel"] == dict(mode="attention_dp_ep", attention_dp=size, ep=size, rank=rank)
    return result


def reject_rank_device(kind):
    service, calls = fixture(2, 1)
    session = service.replicas[0].session
    if kind == "rank":
        session.rank = 2
    elif kind == "worker":
        session.ready["workers"][1]["rank"] = 0
    else:
        session.devices = [99]
    try:
        Service.evidence(service)
    except ValueError as error:
        assert "Attention DP/EP" in str(error)
        assert not calls, "invalid evidence submitted native commands"
        return dict(rejected=True, native_commands=0)
    raise AssertionError("invalid rank/device evidence was accepted")


def archive_config(size, rank=None):
    value = snapshot(size, rank)
    worker = dict(url="http://full", role="full", healthy=True, boot="boot", model="model")

    def urlopen(request, timeout):
        data = dict(workers=[worker]) if request.full_url.endswith("/health") else value
        return io.BytesIO(json.dumps(data).encode())

    with patch("generation_reference.urllib.request.urlopen", urlopen):
        return deployment("http://router", "test", "full")["config"][0]["replicas"][0]


def config_boundary():
    tp = archive_config(2)
    ep2, ep4 = archive_config(2, 0), archive_config(4, 0)
    assert tp["tp"] == 2 and "parallel" not in tp
    assert ep2["tp"] == ep4["tp"] == 1
    assert ep2["parallel"]["ep"] == 2 and ep4["parallel"]["ep"] == 4
    assert ep2 != ep4, "archive lost EP group size"
    return dict(tp=2, attention_tp=1, group_size_distinguished=True)


def check_snapshot(size, rank=None):
    result = snapshot(size, rank)
    return dict(workers=len(result["replicas"][0]["workers"]),
                states=len(result["replicas"][0]["states"]))


def cold_paged_snapshot():
    service, _ = fixture(2, cold=True, page_len=256)
    first = Service.evidence(service)
    second = Service.evidence(service)
    lengths = [state["prefix_cache"]["page_len"] for state in first["replicas"][0]["states"]]
    assert lengths == [256, 256], "first evidence retained the uninitialized page length"
    assert first == second, "cold and initialized evidence disagree"
    return dict(workers=2, page_len=256, cold_matches_initialized=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ("ordinary_tp1", lambda: check_snapshot(1)),
        ("ordinary_tp2", lambda: check_snapshot(2)),
        ("attention_ep2_rank0", lambda: check_snapshot(2, 0)),
        ("attention_ep2_rank1", lambda: check_snapshot(2, 1)),
        ("attention_ep4_rank3", lambda: check_snapshot(4, 3)),
        ("reject_view_rank", lambda: reject_rank_device("rank")),
        ("reject_worker_rank", lambda: reject_rank_device("worker")),
        ("reject_view_device", lambda: reject_rank_device("device")),
        ("reference_parallel_config", config_boundary),
        ("ordinary_tp2_cold_page_len", cold_paged_snapshot),
    ]
    return run(cases, args.output, backend="production C++ evidence; injected native snapshots",
               scope="10 个副本证据与基线配置用例，含普通 TP 冷启动页大小；不代表模型生成或 GPU 验收")


if __name__ == "__main__":
    raise SystemExit(main())
