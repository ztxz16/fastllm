"""已部署真实 P/D 服务的生命周期验证；调用正式控制接口，不注入伪造 GPU 完成事件。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from fastllm_pytools.pd.http import rpc, dumps
from fastllm_pytools.pd.protocol import Identity, Peer, new_id
from report import run


def invoke(args, url, path, body=None):
    return rpc(url, path, body, token=args.token, timeout=args.timeout)


def evidence(args, name):
    result = {role: invoke(args, url, "/internal/evidence") for role, url in (("p", args.prefill), ("d", args.decode))}
    (args.output / (name + ".evidence.json")).write_text(json.dumps(result, indent=2))
    return result


def setup(args):
    p, d = (invoke(args, url, "/health") for url in (args.prefill, args.decode))
    if p["role"] != "p" or d["role"] != "d" or p["model"] != d["model"] or not p["healthy"] or not d["healthy"]:
        raise AssertionError("healthy matching P/D services required")
    identity = Identity(new_id(), new_id(), Peer(**p["peer"]), Peer(**d["peer"]), "p_to_d", args.mode)
    request = dict(token_ids=args.ids[:128], max_new_tokens=8, eos_token_ids=[],
                   temperature=0.0, top_k=0, top_p=1.0, seed=20261126)
    pbody = dict(identity=asdict(identity), request=request, peer_url=args.decode,
                 expires_at=time.time() + args.timeout)
    dbody = dict(pbody, peer_url=args.prefill)
    return identity, pbody, dbody


def cancel(args, identity):
    # 两种模式都先让 P 停止后续写入/计算，再让 D 排空本地读取；远端租约保持到实际完成。
    errors = []
    for url in (args.prefill, args.decode):
        try:
            invoke(args, url, "/internal/cancel", dict(identity=asdict(identity)))
        except urllib.error.HTTPError as error:
            if error.code != 400:  # 尚未登记的另一端没有可取消的请求。
                errors.append(str(error))
    if errors:
        raise RuntimeError("; ".join(errors))


def wait_released(args, identity):
    deadline = time.monotonic() + args.timeout
    while True:
        snapshots = [invoke(args, url, "/internal/evidence") for url in (args.prefill, args.decode)]
        records = [r for snapshot in snapshots for r in snapshot["requests"] if r["transfer"] == identity.transfer]
        if ((not records or all(r["released"] for r in records)) and
                not any(r["health"]["pending_control"] for r in snapshots)):
            return snapshots
        if time.monotonic() >= deadline:
            raise TimeoutError("cancelled request still owns KV or computation")
        time.sleep(0.05)


def control_case(args, name, stage):
    identity, pbody, dbody = setup(args)
    before = evidence(args, name + ".before")
    try:
        if stage in ("old_boot", "expired"):
            if stage == "old_boot":
                dbody["identity"] = asdict(replace(identity, target=replace(identity.target, boot=new_id())))
            else:
                dbody["expires_at"] = time.time() - 1
            try:
                invoke(args, args.decode, "/internal/prepare", dbody)
            except urllib.error.HTTPError as error:
                if error.code != 400:
                    raise
            else:
                raise AssertionError("stale control request accepted")
            after = evidence(args, name + ".after")
            if before["d"]["requests"] != after["d"]["requests"]:
                raise AssertionError("rejected request changed target records")
            return dict(rejected=True, target_unchanged=True)
        invoke(args, args.prefill, "/internal/announce", pbody)
        if stage == "source_ready":
            invoke(args, args.prefill, "/internal/prefill", pbody)
        elif stage == "target_ready":
            invoke(args, args.decode, "/internal/prepare", dbody)
        elif stage == "transfer_ready":
            with ThreadPoolExecutor(max_workers=2) as pool:
                p = pool.submit(invoke, args, args.prefill, "/internal/prefill", pbody)
                d = pool.submit(invoke, args, args.decode, "/internal/prepare", dbody)
                source, _ = p.result(), d.result()
            if args.mode == "pull":
                invoke(args, args.decode, "/internal/pull", source)
            deadline = time.monotonic() + args.timeout
            while True:
                states = invoke(args, args.decode, "/internal/evidence")
                record = next(r for r in states["requests"] if r["transfer"] == identity.transfer)
                if record["error"]:
                    raise AssertionError(record["error"])
                if record["completed"]:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("actual transfer did not finish")
                time.sleep(0.05)
        cancel(args, identity)
        cancel(args, identity)  # 重复取消不得再次结束新请求或重复归还页。
        wait_released(args, identity)
        after = evidence(args, name + ".after")
        if any(not s["health"]["healthy"] for s in after.values()):
            raise AssertionError("normal cancellation made a native group unhealthy")
        return dict(cancelled=True, released=True, transfer=identity.transfer)
    finally:
        cancel(args, identity)


def disconnect_case(args, name):
    payload = dict(token_ids=args.ids[:128], max_new_tokens=128, eos_token_ids=[],
                   temperature=0.0, stream=True, return_token_ids=True)
    query = urllib.request.Request(args.router.rstrip("/") + "/v1/completions", dumps(payload),
        headers={"Authorization": "Bearer " + args.token, "Content-Type": "application/json"})
    request_id = None
    with urllib.request.urlopen(query, timeout=args.timeout) as stream:
        for line in stream:
            if line.startswith(b"data: "):
                event = json.loads(line[6:])
                if event.get("token_ids"):
                    request_id = event["id"]
                    break
    if request_id is None:
        raise AssertionError("no actual first token before disconnect")
    deadline = time.monotonic() + args.timeout
    while True:
        snapshots = [invoke(args, url, "/internal/evidence") for url in (args.prefill, args.decode)]
        records = [r for s in snapshots for r in s["requests"] if r["request"] == request_id]
        if records and all(r["released"] for r in records):
            evidence(args, name + ".after")
            if len({r["transfer"] for r in records}) != 1:
                raise AssertionError("Router retried after client received output")
            return dict(request=request_id, released=True, retries=0)
        if time.monotonic() >= deadline:
            raise TimeoutError("client disconnect did not drain request")
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser()
    for name in ("prefill", "decode", "router"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("pull", "push"), required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--prompt-token-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.token = os.environ.get("FT_PD_TOKEN", "")
    args.ids = json.loads(args.prompt_token_ids.read_text())
    if not args.token or not isinstance(args.ids, list) or not args.ids or any(type(t) is not int or t < 0 for t in args.ids):
        parser.error("FT_PD_TOKEN and a valid nonempty Token ID array are required")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [(stage, lambda stage=stage: control_case(args, stage, stage)) for stage in
             ("old_boot", "expired", "announce_only", "source_ready", "target_ready", "transfer_ready")]
    cases.append(("client_disconnect", lambda: disconnect_case(args, "client_disconnect")))
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"], text=True)
    return run(cases, args.output, backend="deployed P/D native service + Router",
        metadata_extra=dict(gpu=gpu, mode=args.mode, cuda="see service evidence worker fields"),
        scope="7 个实际服务拒绝、取消和断连场景；要求隔离部署无其他客户端，不含破坏性 GPU/rank 故障")


if __name__ == "__main__":
    raise SystemExit(main())
