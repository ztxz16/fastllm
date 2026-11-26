"""正式 HTTP 入口的端到端性能对照；六个 case，每项五轮，不宣称未测量的 GPU 重叠。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

from check_generation import request
from report import run


def batch(url, token, payload, concurrency):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        samples = list(pool.map(lambda _: request(url, token, payload), range(concurrency)))
    return dict(wall_s=time.perf_counter() - started, requests=samples)


def p95(values):
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def evidence(args):
    snapshots = []
    for url in args.service:
        query = urllib.request.Request(url.rstrip("/") + "/internal/evidence",
                                       headers={"Authorization": "Bearer " + args.token})
        with urllib.request.urlopen(query, timeout=300) as response:
            snapshots.append(json.load(response))
    if {s["health"]["role"] for s in snapshots} != {"p", "d", "full"}:
        raise AssertionError("supply P, D and full reference service evidence URLs")
    if len({s["health"]["model"] for s in snapshots}) != 1:
        raise AssertionError("model fingerprints differ")
    flags = {state["enabled"] for s in snapshots for r in s["replicas"] for state in r["states"]}
    caches = {state["prefix_cache"]["enabled"] for s in snapshots for r in s["replicas"] for state in r["states"]}
    if len(flags) != 1 or caches != {args.cache_state == "warm"}:
        raise AssertionError("Graph/prefix cache configuration differs from requested comparison")
    return snapshots


def performance(args, name, tokens, concurrency):
    before = evidence(args)
    (args.output / (name + ".before.json")).write_text(json.dumps(before, indent=2))
    payload = dict(token_ids=tokens, max_new_tokens=64, temperature=0.0, top_k=0, top_p=1.0,
                   seed=20261126, eos_token_ids=[], stream=True, return_token_ids=True)
    # 精确比较与 warmup 在测量轮次之外；模型与服务配置由部署 profile 固定。
    expected = request(args.reference, args.token, payload)["token_ids"]
    if request(args.router, args.token, payload)["token_ids"] != expected or len(expected) != 64:
        raise AssertionError("pre-measurement exact comparison failed")
    for url in (args.reference, args.router):
        batch(url, args.token, payload, concurrency)
    samples = {"reference": [], "pd": []}
    for iteration in range(5):
        # 交替先后顺序，降低固定测量顺序造成的偏差。
        order = (("reference", args.reference), ("pd", args.router))
        for label, url in order if iteration % 2 == 0 else reversed(order):
            measured = batch(url, args.token, payload, concurrency)
            samples[label].append(measured)
            if any(item["token_ids"] != expected for item in measured["requests"]):
                (args.output / (name + ".samples.json")).write_text(json.dumps(samples, indent=2))
                raise AssertionError("timed run changed actual output; exact FAIL")
    (args.output / (name + ".samples.json")).write_text(json.dumps(samples, indent=2))
    after = evidence(args)
    (args.output / (name + ".after.json")).write_text(json.dumps(after, indent=2))
    for snapshot, previous in zip(after, before):
        if snapshot["health"]["role"] not in ("d", "full"):
            continue
        key = "replays" if snapshot["replicas"][0]["states"][0]["enabled"] else "eager_decodes"
        current = sum(s[key] for r in snapshot["replicas"] for s in r["states"])
        old = sum(s[key] for r in previous["replicas"] for s in r["states"])
        if current <= old:
            raise AssertionError(f"no actual {key} execution evidence")
    walls = [x["wall_s"] for x in samples["pd"]]
    reference = [x["wall_s"] for x in samples["reference"]]
    requests = [r for row in samples["pd"] for r in row["requests"]]
    ttft = [r["arrivals_s"][0] * 1000 for r in requests]
    itl = [(b - a) * 1000 for r in requests for a, b in zip(r["arrivals_s"], r["arrivals_s"][1:])]
    if not itl:
        raise AssertionError("no inter-token timing samples")
    return dict(prompt_tokens=len(tokens), output_tokens=64, concurrency=concurrency, repeats=5,
        median_ms=statistics.median(walls) * 1000, p95_ms=p95(walls) * 1000,
        cv=statistics.pstdev(walls) / statistics.mean(walls),
        ttft_median_ms=statistics.median(ttft), ttft_p95_ms=p95(ttft),
        inter_token_median_ms=statistics.median(itl), inter_token_p95_ms=p95(itl),
        output_tokens_per_second=64 * concurrency / statistics.median(walls),
        speedup=statistics.median(reference) / statistics.median(walls), exact="PASS",
        overlap="not measured; requires GPU timeline", cache_policy=args.cache_state)


def main():
    parser = argparse.ArgumentParser()
    for name in ("router", "reference", "profile", "cache-state"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prompt-token-ids", type=Path, required=True)
    parser.add_argument("--service", action="append", required=True, help="repeat for every P/D/full service; used outside timing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.token = os.environ.get("FT_PD_TOKEN", "")
    if not args.token:
        parser.error("FT_PD_TOKEN is required")
    if args.cache_state not in ("off", "warm"):
        parser.error("cache-state must be off or warm; cold runs require a separate cache-clear protocol")
    ids = json.loads(args.prompt_token_ids.read_text())
    if not isinstance(ids, list) or not ids or any(type(x) is not int or not 0 <= x <= 0xffffffff for x in ids):
        parser.error("expected nonempty Token ID array")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for length in (128, 512, 1024):
        tokens = (ids * math.ceil(length / len(ids)))[:length]
        for concurrency in (1, 4):
            name = f"prompt{length}_output64_concurrency{concurrency}"
            cases.append((name, lambda n=name, t=tokens, c=concurrency: performance(args, n, t, c)))
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"], text=True)
    return run(cases, args.output, backend="formal streaming HTTP -> native workers",
        metadata_extra=dict(gpu=gpu, profile=args.profile, warmup=1, iterations=1, repeats=5,
                            cuda="see deployed worker ready.json", architecture="see deployed build log",
                            cache_state=args.cache_state, reference=args.reference, router=args.router),
        scope="6 个端到端性能 case；相同请求范围，5 轮原始样本；CUDA 时间线重叠须另行采集")


if __name__ == "__main__":
    raise SystemExit(main())
