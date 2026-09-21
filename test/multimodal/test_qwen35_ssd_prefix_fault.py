#!/usr/bin/env python3
"""Damage one late rank-3 object in an isolated copy, then check GPU fallback.

Reuses a successful HTTP fixture to avoid repeating cold model inference.
Original evidence is never modified: the damaged hardlink is replaced first.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal

from test_qwen35_ssd_prefix_cache_e2e import (
    CACHE_ROOT, EVIDENCE_ROOT, equal_output, events, library_digest, make_inputs,
    markers_since, require, save_json, send_request, server, wait_committed,
)


def read_commit(path):
    envelope = json.loads(path.read_text())
    require(hashlib.sha256(envelope["body"].encode()).hexdigest() == envelope["sha256"],
            "Invalid fixture commit checksum")
    return json.loads(envelope["body"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    for label in (args.label, args.source_label):
        require(label.replace("-", "").replace("_", "").isalnum(), "Use safe labels")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Fault deadline")))
    signal.alarm(args.timeout)
    manifest = json.loads((EVIDENCE_ROOT / args.source_label / "manifest.json").read_text())
    require(library_digest() == manifest["library_sha256"], "Runtime differs from cold fixture")
    reference = manifest["groups"]["images"]["references"]["repeat"]
    require(manifest["groups"]["images"]["write_complete"], "Incomplete fixture")
    key = reference["committed"]["line"].split("key=", 1)[1].rstrip(".")
    original = Path(manifest["groups"]["images"]["cache_dir"])
    checkpoint = read_commit(next(original.glob("v2/commits/*/" + key + ".commit")))
    state = checkpoint["state"]
    require(state["length"] == 8192, "Fixture needs an 8192-token checkpoint")
    bundle = next(item for item in reversed(state["kv"])
                  if all(part["rank"] == 3 and part["begin"] >= 6144 for part in item["segments"]))
    chunk = bundle["chunks"][-1]
    relative = Path("v2/objects") / ("fastllm@0x0@0@" + chunk["sha256"] + ".data")
    source_bytes = (original / relative).read_bytes()
    require(source_bytes[:8] == b"FLKV002\n" and len(source_bytes) == 80 + chunk["bytes"],
            "Unexpected object header")
    require(hashlib.sha256(source_bytes[80:]).hexdigest() == chunk["sha256"], "Fixture already damaged")
    cache = CACHE_ROOT / args.label
    evidence = EVIDENCE_ROOT / args.label
    require(not cache.exists() and not evidence.exists(), "Preserve prior evidence; choose a new label")
    evidence.mkdir()
    # Rebuild the disposable copy's SQLite index from authoritative commits.
    for directory in ("objects", "commits", "identities"):
        shutil.copytree(original / "v2" / directory, cache / "v2" / directory, copy_function=os.link)
    damaged = bytearray(source_bytes)
    damaged[80 + len(damaged[80:]) // 2] ^= 1
    target = cache / relative
    temporary = target.with_suffix(".fault-tmp")
    temporary.write_bytes(damaged)
    temporary.replace(target)
    require(target.stat().st_ino != (original / relative).stat().st_ino, "Fault must not alter source inode")
    run = {"status": "running", "attempt": "rank3-corruption", "evidence_dir": str(evidence),
           "servers": [], "requests": [], "startup_timeout": 600, "commit_timeout": 120,
           "source_label": args.source_label, "library_sha256": manifest["library_sha256"],
           "fault": {"object": str(relative), "rank": 3, "begin": bundle["segments"][0]["begin"],
                     "original_sha256": chunk["sha256"],
                     "damaged_sha256": hashlib.sha256(damaged[80:]).hexdigest()},
           "limitations": ["No allocator-level page-leak counter or power-loss simulation"]}
    result = evidence / "result.json"
    save_json(result, run)
    try:
        messages = make_inputs(manifest["nonce"])["images"]["repeat"]
        with server(run, "fallback", cache) as (proc, logfile):
            actual, offset = send_request(run, "rank3-corruption/fallback", messages, logfile)
            markers = markers_since(logfile, offset)
            require(any("restore miss: tokens=8192 reason=blob_checksum" in line for line in markers),
                    "Expected damaged 8192-token candidate to be rejected")
            require(actual["cached_tokens"] == 2048 and
                    any(event["tokens"] == 2048 for event in events(markers, "restored")),
                    "Expected shorter complete 2048-token SSD fallback")
            require(equal_output(actual, reference), "Fallback output differs from independent cold fixture")
            actual["output_equal"] = True
            actual["committed"] = wait_committed(run, proc, logfile, offset, 8192)
        repaired = target.read_bytes()
        require(repaired == source_bytes, "Recomputed object has not repaired original bytes")
        require((original / relative).read_bytes() == source_bytes, "Source fixture changed")
        run["fault"]["repaired_bytes_equal"] = True
        # Prove the repaired checkpoint survives another server process.
        with server(run, "repaired-restart", cache) as (_, logfile):
            actual, _ = send_request(run, "rank3-corruption/repaired-restart", messages, logfile)
            require(actual["cached_tokens"] == 8192 and
                    any(event["tokens"] == 8192 for event in events(actual["markers"], "restored")),
                    "Repaired checkpoint did not restore after restart")
            require(equal_output(actual, reference), "Repaired restart output differs from cold fixture")
            actual["output_equal"] = True
        run["status"] = "ok"
    except BaseException as error:
        run.update(status="failed", error=repr(error))
        raise
    finally:
        save_json(result, run)
    signal.alarm(0)
    print(json.dumps({"status": run["status"], "result": str(result)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
