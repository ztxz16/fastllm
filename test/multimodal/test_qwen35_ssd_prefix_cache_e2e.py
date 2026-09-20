#!/usr/bin/env python3
"""Opt-in TP4 SSD restart acceptance; requires the fastllm2 environment.

Examples (each new label preserves its cache and all evidence):
  python test/multimodal/test_qwen35_ssd_prefix_cache_e2e.py --label smoke --case text
  python test/multimodal/test_qwen35_ssd_prefix_cache_e2e.py --label tree --case branches --phase write
  python test/multimodal/test_qwen35_ssd_prefix_cache_e2e.py --label tree --case branches --phase restore
  python test/multimodal/test_qwen35_ssd_prefix_cache_e2e.py --label media --case images/changed
  python test/multimodal/test_qwen35_ssd_prefix_cache_e2e.py --label conversion --case precision

This script starts GPU servers only when explicitly run. It never terminates
pre-existing processes, deletes a cache, or claims native conversion numerical
equivalence. The precision case checks HTTP restore accounting and conversion
markers; a same-source in-memory conversion/logits comparison is a separate
required native test. A successful HTTP result is not that numerical result.
"""

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from test_qwen35_prefix_cache_e2e import (
    DRAFT, FTLLM, LIBRARY, MODEL, NAME, ROOT, chat, say, scenarios,
)


CACHE_ROOT = Path("/mnt/nvme/LLM/fastllm-kv-cache1")
EVIDENCE_ROOT = ROOT / "build-acceptance/20260920-ssd-prefix-cache"
CONDA_ENV = Path("/home/sy/miniconda3/envs/fastllm2")
DISK_BYTES = 256 * 1024**3
CHUNK = 2048
CASE_NAMES = (
    "text/repeat", "branches/A", "branches/B", "branches/A-return",
    "images/repeat", "images/changed", "images/append", "precision/convert",
)
SSD_EVENT = re.compile(r"\[Prefix SSD\].*?\b(committed|restored):\s*tokens=(\d+)\b")
DTYPE_FIELD = re.compile(r"\b(source_dtype|target_dtype)=([A-Za-z0-9_]+)")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def library_digest():
    digest = hashlib.sha256()
    with Path(LIBRARY).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cases(spec):
    selected = set()
    for selector in spec.split(","):
        selector = selector.strip()
        matches = [name for name in CASE_NAMES
                   if selector == "all" or name == selector or name.startswith(selector + "/")]
        require(matches, "Unknown --case selector: " + selector)
        selected.update(matches)
    return [name for name in CASE_NAMES if name in selected]


def make_inputs(nonce):
    existing = {name: messages for name, messages, _ in scenarios()}

    def tagged(name, group):
        messages = copy.deepcopy(existing[name])
        messages[0]["content"] = "SSD验收命名空间=" + nonce + "/" + group + "。\n" + messages[0]["content"]
        return messages

    text = tagged("text/cold", "text")
    common = tagged("text/cold", "branches")
    branches = {}
    for letter in ("A", "B"):
        # The divergence is followed by several complete prefill chunks. A hit
        # only on the common 8K trunk cannot pass the branch checkpoint check.
        suffix = (f"分支 {letter} 的记录：这里属于当前分支，请保留本分支的标记。\n" * 180)
        branches[letter] = common + [
            {"role": "assistant", "content": "项目代号是青山。"},
            {"role": "user", "content": suffix + f"只回答当前分支标记 {letter}。"},
        ]
    return {
        "text": {"repeat": text},
        "branches": {"common": common, **branches, "A-return": copy.deepcopy(branches["A"])},
        "images": {
            "repeat": tagged("first/cold", "images"),
            "changed": tagged("first/changed", "images"),
            "append": tagged("first/append-B", "images"),
        },
        "precision": {"convert": tagged("first/cold", "precision")},
    }


def command_for(dtype):
    # Exactly the production inference arguments; only the precision case uses
    # float16 for its source run. SSD settings are supplied through environment.
    return [FTLLM, "server", MODEL, "--tp", "0,1,2,3", "--gpu_mem_ratio", "0.95",
            "--chunked_prefill_size", "2048", "--kv_cache_dtype", dtype,
            "--enable_thinking", "true", "--speculative_algorithm", "dflash",
            "--speculative_draft_model_path", DRAFT, "--draft_tokens", "6",
            "--model_name", NAME, "--host", "0.0.0.0", "--port", "8080", "--multimodal"]


def assert_resources_idle():
    with socket.socket() as probe:
        # Match the server's socket policy: a closed server can leave accepted
        # connections in TIME_WAIT without any process still listening.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", 8080))
        except OSError as error:
            raise RuntimeError("Port 8080 is occupied; no existing process will be stopped") from error
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    selected = {}
    for row in gpu_rows.splitlines():
        index, uuid, memory = [field.strip() for field in row.split(",")]
        if int(index) in range(4):
            selected[uuid] = int(index)
            require(int(memory) <= 64, f"GPU {index} has {memory} MiB in use; refusing to start")
    require(len(selected) == 4, "All four physical GPUs 0-3 must be available")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
        text=True, timeout=15)
    for row in processes.splitlines():
        uuid = row.split(",", 1)[0].strip()
        require(uuid not in selected, "GPU 0-3 have a compute process: " + row)
    return gpu_rows


def markers_since(logfile, offset=0):
    with logfile.open("rb") as source:
        source.seek(offset)
        return [line for line in source.read().decode(errors="replace").splitlines()
                if "[Prefix SSD]" in line or "[PrefixCache]" in line
                or "prefix cache" in line or "[Vision]" in line or "[Prompt]" in line]


def events(markers, kind):
    result = []
    for line in markers:
        match = SSD_EVENT.search(line)
        if match and match.group(1) == kind:
            result.append({"tokens": int(match.group(2)), "line": line,
                           **dict(DTYPE_FIELD.findall(line))})
    return result


def canonical_dtype(value):
    value = value.lower()
    return {"fp16": "float16", "half": "float16", "fp8": "fp8_e4m3"}.get(value, value)


def stop_owned(proc):
    # The process-group ID belongs to this Popen(start_new_session=True), never
    # an arbitrary process found by its port, GPU allocation, or executable name.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=35)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=15)


@contextmanager
def server(run, label, cache_dir, dtype="fp8_e4m3", reference=False):
    command = command_for(dtype)
    configured = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3", "FASTLLM_ACTIVATE_NUMA": "ON",
        "FASTLLM_NUMA_THREADS": "27", "FASTLLM_CUDA_DFLASH_TP_BACKBONE": "force",
        "FASTLLM_PREFIX_CACHE": "1", "FASTLLM_MULTIMODAL_PREFIX_CACHE": "0" if reference else "1",
        "PYTHONUNBUFFERED": "1",
    }
    if not reference:
        configured.update(FASTLLM_PREFIX_CACHE_DIR=str(cache_dir),
                          FASTLLM_PREFIX_CACHE_DISK_BYTES=str(DISK_BYTES),
                          FASTLLM_PREFIX_CACHE_RESTORE_POLICY="always")
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("FASTLLM_PREFIX_CACHE_"):
            env.pop(key)
    env.update(configured)
    record = {"name": label, "command": command, "environment": configured,
              "gpu_before": assert_resources_idle()}
    run["servers"].append(record)
    logfile = Path(run["evidence_dir"]) / (run["attempt"] + "-" + label + ".log")
    record["log"] = str(logfile)
    proc = None
    try:
        with logfile.open("xb") as sink:
            proc = subprocess.Popen(command, env=env, stdout=sink, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        record["pid"] = proc.pid
        say(f"{label}: starting own server pid={proc.pid}")
        deadline = time.monotonic() + run["startup_timeout"]
        while time.monotonic() < deadline:
            require(proc.poll() is None, f"{label}: server exited ({proc.returncode}); see {logfile}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2) as response:
                    if any(model["id"] == NAME for model in json.load(response)["data"]):
                        break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError(f"{label}: startup deadline exceeded; see {logfile}")
        yield proc, logfile
    finally:
        if proc is not None:
            stop_owned(proc)
            record["returncode"] = proc.returncode


def send_request(run, name, messages, logfile):
    offset = logfile.stat().st_size
    say(name + ": sending")
    response = chat(messages)
    usage = response["usage"]
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    record = {"name": name, "prompt_tokens": prompt, "cached_tokens": cached,
              "response": response, "markers": markers_since(logfile, offset)}
    run["requests"].append(record)
    require(prompt > 8192, f"{name}: actual prompt must exceed 8192 tokens, got {prompt}")
    require(0 <= cached < prompt, f"{name}: invalid cached token accounting {cached}/{prompt}")
    require(cached % CHUNK == 0, f"{name}: restored non-natural prefill boundary {cached}")
    say(f"{name}: cached={cached}/{prompt}, time={response['seconds']:.2f}s")
    return record, offset


def wait_committed(run, proc, logfile, offset, minimum):
    deadline = time.monotonic() + run["commit_timeout"]
    while time.monotonic() < deadline:
        require(proc.poll() is None, "Server exited before committing the required checkpoint")
        found = events(markers_since(logfile, offset), "committed")
        qualifying = [event for event in found if event["tokens"] >= minimum]
        if qualifying:
            return max(qualifying, key=lambda event: event["tokens"])
        time.sleep(0.25)
    raise RuntimeError(f"No [Prefix SSD] committed checkpoint >= {minimum} within deadline; see {logfile}")


def equal_output(actual, expected):
    return all(actual["response"][key] == expected["response"][key]
               for key in ("content", "reasoning", "finish"))


def source_boundary(record):
    return ((record["prompt_tokens"] - 1) // CHUNK) * CHUNK


def write_group(run, manifest, manifest_path, inputs, group):
    info = manifest["groups"].setdefault(group, {})
    require(not info.get("write_complete"), f"{group}: write phase already completed; use --phase restore")
    cache_dir = Path(manifest["cache_dir"]) / group
    require(not cache_dir.exists() or not any(cache_dir.iterdir()),
            f"{cache_dir} already contains data; use a new label for a fresh cold write")
    references = {}
    if group == "images":
        # Modified-image and appended-image references must not use the cache
        # being tested: otherwise an invalid hit could validate itself.
        with server(run, "images-reference", None, reference=True) as (_, logfile):
            for name in ("changed", "append"):
                record, _ = send_request(run, "reference/images/" + name, inputs[group][name], logfile)
                require(record["cached_tokens"] == 0, "Image cold reference unexpectedly used prefix cache")
                require(not events(record["markers"], "restored"), "Image cold reference restored SSD state")
                references[name] = record
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = {"text": ("repeat",), "branches": ("common", "A", "B"),
             "images": ("repeat",), "precision": ("convert",)}[group]
    dtype = "float16" if group == "precision" else "fp8_e4m3"
    with server(run, group + "-write", cache_dir, dtype) as (proc, logfile):
        for index, name in enumerate(names):
            record, offset = send_request(run, "write/" + group + "/" + name, inputs[group][name], logfile)
            if index == 0:
                require(record["cached_tokens"] == 0, "Fresh namespace unexpectedly reused a prefix")
            minimum = source_boundary(record)
            record["committed"] = wait_committed(run, proc, logfile, offset, minimum)
            record["required_restore_tokens"] = minimum
            references[name] = record
            info.update(cache_dir=str(cache_dir), source_dtype=dtype, references=references)
            save_json(manifest_path, manifest)
        if group == "branches":
            require(references["A"]["prompt_tokens"] == references["B"]["prompt_tokens"],
                    "A/B token counts differ; this is not the promised equal-length branch test")
            require(source_boundary(references["A"]) > references["common"]["prompt_tokens"],
                    "Branch-specific checkpoint must extend beyond the common trunk")
            references["A-return"] = references["A"]
    info.update(cache_dir=str(cache_dir), source_dtype=dtype, references=references, write_complete=True)
    save_json(manifest_path, manifest)


def restore_case(run, manifest, manifest_path, inputs, name):
    group, case = name.split("/", 1)
    info = manifest["groups"].get(group, {})
    require(info.get("write_complete"), f"{group}: run --phase write with this label first")
    attempted = info.setdefault("restore_attempts", {})
    require(case not in attempted, f"{name}: this namespace already attempted restore; use a new label "
            "to avoid testing against objects written by an earlier restore")
    attempted[case] = {"attempt": run["attempt"], "status": "started"}
    save_json(manifest_path, manifest)
    reference = info["references"][case]
    # Restart separately for each A/B/A entry. A later in-memory hit cannot
    # masquerade as persistent recovery of a branch evicted before restart.
    with server(run, name.replace("/", "-") + "-restore", Path(info["cache_dir"])) as (_, logfile):
        record, _ = send_request(run, "restore/" + name, inputs[group][case], logfile)
        require(record["prompt_tokens"] == reference["prompt_tokens"], name + ": prompt length changed")
        restored = events(record["markers"], "restored")
        if name == "images/changed":
            # This fixture places the changed picture before the first natural
            # 2048 checkpoint. There is no legal earlier complete checkpoint.
            require(record["cached_tokens"] == 0 and not restored,
                    name + ": reused a checkpoint affected by the changed first image")
        else:
            origin = info["references"]["repeat"] if name == "images/append" else reference
            minimum = origin["required_restore_tokens"]
            require(record["cached_tokens"] >= minimum, name + ": expected persisted complete prefix")
            require(any(event["tokens"] == record["cached_tokens"] for event in restored),
                    name + ": cached usage lacks matching [Prefix SSD] restored marker")
        if group == "precision":
            converted = [event for event in restored
                         if canonical_dtype(event.get("source_dtype", "")) == "float16"
                         and canonical_dtype(event.get("target_dtype", "")) == "fp8_e4m3"
                         and event["tokens"] == record["cached_tokens"]]
            require(converted, "FP16 source must be explicitly reported as restored into FP8 target KV")
            record["conversion_evidence"] = converted
            record["remaining_input_tokens"] = record["prompt_tokens"] - record["cached_tokens"]
            record["numerical_validation"] = "not_performed_requires_same_source_in_memory_conversion_reference"
            record["prefix_forward_verification"] = "restore_log_and_usage_accounting_only_not_kernel_trace"
            run["limitations"].append("precision: HTTP conversion contract only; native same-source "
                                      "converted-state/logits and kernel-path verification remain required")
        else:
            record["output_equal"] = equal_output(record, reference)
            require(record["output_equal"], name + ": greedy content/reasoning/finish differs from reference")
    attempted[case]["status"] = "ok"
    save_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="Stable run namespace; reuse only to advance phases/cases")
    parser.add_argument("--phase", choices=("all", "write", "restore"), default="all")
    parser.add_argument("--case", default="text", help="Comma-separated groups or cases: " + ", ".join(CASE_NAMES))
    parser.add_argument("--deadline", type=int, default=3600, help="Hard total wall-clock deadline in seconds")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--commit-timeout", type=int, default=120)
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.label) is not None, "Unsafe label")
    require(min(args.deadline, args.startup_timeout, args.commit_timeout) > 0, "Timeouts must be positive")
    require(Path(sys.prefix).resolve() == CONDA_ENV.resolve(), "Run with conda activate fastllm2")
    selected = select_cases(args.case)
    evidence = EVIDENCE_ROOT / args.label
    evidence.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence / "manifest.json"
    digest = library_digest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        require(manifest["library_sha256"] == digest, "Installed runtime changed since write; use a new label")
    else:
        require(args.phase != "restore", "No write manifest for this label")
        cache_dir = CACHE_ROOT / args.label
        require(not cache_dir.exists(), "Cache namespace exists without manifest; preserving it, choose a new label")
        manifest = {"schema": 1, "label": args.label, "nonce": secrets.token_hex(12),
                    "cache_dir": str(cache_dir), "library_sha256": digest, "groups": {}}
        save_json(manifest_path, manifest)
    attempt = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(3)
    resultfile = evidence / (attempt + ".json")
    run = {"status": "failed", "contract": "http_ssd_restart_acceptance", "phase": args.phase,
           "cases": selected, "attempt": attempt, "evidence_dir": str(evidence),
           "cache_dir": manifest["cache_dir"], "library_sha256": digest, "servers": [],
           "requests": [], "limitations": [], "startup_timeout": args.startup_timeout,
           "commit_timeout": args.commit_timeout, "hard_deadline_seconds": args.deadline}
    lockfile = evidence / ".running"
    lockfd = os.open(lockfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(lockfd, str(os.getpid()).encode())
    os.close(lockfd)

    def interrupted(signum, _frame):
        raise RuntimeError(f"Interrupted by signal {signum}; cleaning up only this run's process group")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, args.deadline)
    try:
        assert_resources_idle()
        inputs = make_inputs(manifest["nonce"])
        if args.phase == "all":
            for group in dict.fromkeys(name.split("/", 1)[0] for name in selected):
                if not manifest["groups"].get(group, {}).get("write_complete"):
                    write_group(run, manifest, manifest_path, inputs, group)
                    save_json(resultfile, run)
                for name in selected:
                    if name.split("/", 1)[0] == group:
                        restore_case(run, manifest, manifest_path, inputs, name)
                        save_json(resultfile, run)
        elif args.phase == "write":
            for group in dict.fromkeys(name.split("/", 1)[0] for name in selected):
                write_group(run, manifest, manifest_path, inputs, group)
                save_json(resultfile, run)
        elif args.phase == "restore":
            for name in selected:
                restore_case(run, manifest, manifest_path, inputs, name)
                save_json(resultfile, run)
        run["status"] = "ok"
    except BaseException as error:
        run["error"] = repr(error)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        save_json(resultfile, run)
        lockfile.unlink()
        say("HTTP contract result: " + run["status"] + " " + str(resultfile))


if __name__ == "__main__":
    main()
