"""对已部署的正式 Router 入口做 Token exact 比对；每个部署组合独立执行 15 个 case。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

from report import run


def request(url, token, payload):
    data = json.dumps(payload, allow_nan=False).encode()
    query = urllib.request.Request(url.rstrip("/") + "/v1/completions", data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    began = time.perf_counter()
    with urllib.request.urlopen(query, timeout=300) as response:
        if not payload.get("stream"):
            body = json.load(response)
            ids = body["token_ids"]
            return dict(token_ids=ids, response=body, elapsed_s=time.perf_counter() - began)
        ids, deltas, times, done = [], [], [], False
        for line in response:
            if not line.startswith(b"data: "):
                continue
            value = line[6:].strip()
            if value == b"[DONE]":
                done = True
                break
            body = json.loads(value)
            if "error" in body:
                raise RuntimeError(body["error"])
            new_ids = body.get("token_ids", [])
            ids.extend(new_ids)
            if new_ids:
                times.append(time.perf_counter() - began)
            deltas.append(body)
        if not done or not ids:
            raise RuntimeError("stream ended without token IDs and DONE")
        return dict(token_ids=ids, deltas=deltas, arrivals_s=times, elapsed_s=time.perf_counter() - began)


def compare(args, output, name, tokens, *, generated=8, streaming=False, sampling=False, session=None):
    payload = dict(token_ids=tokens, max_new_tokens=generated, top_k=5 if sampling else 0,
        temperature=0.7 if sampling else 0.0, top_p=1.0, seed=20261126, eos_token_ids=[],
        return_token_ids=True, stream=streaming)
    if session:
        payload["session_id"] = session
    reference = request(args.reference, args.token, payload)
    actual = request(args.router, args.token, payload)
    expected_ids, actual_ids = reference["token_ids"], actual["token_ids"]
    first = next((i for i, (a, b) in enumerate(zip(expected_ids, actual_ids)) if a != b),
                 min(len(expected_ids), len(actual_ids)) if len(expected_ids) != len(actual_ids) else None)
    evidence = dict(request=payload, reference=reference, actual=actual,
                    comparison="exact", first_divergence=first,
                    candidate_comparison="not available: public endpoint does not export logits")
    (output / (name + ".evidence.json")).write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
    if len(expected_ids) != generated or len(actual_ids) != generated:
        raise AssertionError("output length/position mismatch; evidence retained")
    if expected_ids != actual_ids:
        raise AssertionError(f"exact FAIL at output {first}; numeric warnings cannot override this failure")
    return dict(prompt_tokens=len(tokens), output_tokens=generated, exact="PASS",
                latency_s=actual["elapsed_s"], comparison_scope="actual output token IDs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", required=True)
    parser.add_argument("--reference", required=True, help="同模型同配置完整服务的 Router URL")
    parser.add_argument("--profile", required=True, help="如 pull_p_tp2_d_tp1_graph0_cache0")
    parser.add_argument("--prompt-token-ids", required=True, type=Path, help="固定 Token ID 数组 JSON")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.token = os.environ.get("FT_PD_TOKEN", "")
    if not args.token:
        parser.error("FT_PD_TOKEN must be set")
    ids = json.loads(args.prompt_token_ids.read_text())
    if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
        parser.error("prompt token file must be a nonempty uint32 array")
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap",
        "--format=csv,noheader"], text=True).strip()
    cases = []
    for length in (1, 128, 511, 512, 513):
        tokens = (ids * ((length + len(ids) - 1) // len(ids)))[:length]
        for streaming in (False, True):
            name = f"{args.profile}_prompt{length}_stream{int(streaming)}"
            cases.append((name, lambda n=name, t=tokens, s=streaming: compare(args, args.output, n, t, streaming=s)))
    name = args.profile + "_first_token_only"
    cases.append((name, lambda n=name: compare(args, args.output, n, ids[:128], generated=1)))
    name = args.profile + "_sampling_fixed_seed"
    cases.append((name, lambda n=name: compare(args, args.output, n, ids[:128], sampling=True)))
    for concurrency in (2, 4):
        name = f"{args.profile}_concurrent{concurrency}"
        def concurrent_case(n=name, count=concurrency):
            with ThreadPoolExecutor(max_workers=count) as pool:
                futures = [pool.submit(compare, args, args.output, n + f"_request{i}",
                    (ids * (514 // len(ids) + 1))[i:i + 512], streaming=True) for i in range(count)]
                for future in futures:
                    future.result()
            return dict(concurrency=count, exact="PASS")
        cases.append((name, concurrent_case))
    name = args.profile + "_multi_turn"
    def multi_turn():
        session = uuid.uuid4().hex
        tokens = ids[:128]
        for turn in range(3):
            payload = dict(token_ids=tokens, max_new_tokens=8, temperature=0.0, top_k=0, top_p=1.0,
                seed=20261126, eos_token_ids=[], return_token_ids=True, stream=True, session_id=session)
            reference = request(args.reference, args.token, payload)
            actual = request(args.router, args.token, payload)
            (args.output / (name + f"_turn{turn}.evidence.json")).write_text(json.dumps(
                dict(request=payload, actual=actual, reference=reference), indent=2))
            if actual["token_ids"] != reference["token_ids"] or len(actual["token_ids"]) != 8:
                raise AssertionError(f"multi-turn exact FAIL: turn {turn}")
            tokens = tokens + actual["token_ids"] + ids[:4]
        return dict(turns=3, exact="PASS", scope="complete history supplied; enable return-KV to exercise reuse")
    cases.append((name, multi_turn))
    return run(cases, args.output, backend="formal HTTP Router -> native workers",
        metadata_extra=dict(gpu=gpu, cuda="recorded by deployed worker command and build logs", profile=args.profile,
                            reference=args.reference, router=args.router, architecture="see GPU compute_cap above"),
        scope="单个部署组合的 15 个正式入口 exact 用例，含并发与多轮；不含故障注入或性能验收")


if __name__ == "__main__":
    raise SystemExit(main())
