#!/usr/bin/env python3
"""Time process launch, HTTP readiness and first real streamed generation.

Uses the user's production inference arguments, adding only the existing
--startup-progress ndjson instrumentation. Never stops an existing process.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import urllib.request

from test_qwen35_ssd_prefix_cache_e2e import (
    CACHE_ROOT, DISK_BYTES, EVIDENCE_ROOT, assert_resources_idle, command_for,
    equal_output, events, library_digest, make_inputs, require, save_json, send_request, stop_owned,
)
from test_qwen35_prefix_cache_e2e import LIBRARY, NAME, chat, image_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--timeout", type=int, default=720)
    parser.add_argument("--monitor", action="store_true",
                        help="Collect 1-second sysstat, vmstat and GPU telemetry from before launch")
    parser.add_argument("--restore-fixture", type=Path,
                        help="Reuse a previously successful images fixture to check old SSD prefix restoration in the same run")
    args = parser.parse_args()
    require(args.label.replace("-", "").replace("_", "").isalnum(), "Use a safe label")
    evidence = EVIDENCE_ROOT / "startup-hash" / args.label
    require(not evidence.exists(), "Preserve previous evidence; choose a new label")
    assert_resources_idle()
    evidence.mkdir(parents=True)
    resultfile, logfile = evidence / "result.json", evidence / "server.log"
    fixture = json.loads(args.restore_fixture.read_text()) if args.restore_fixture else None
    cache_dir = CACHE_ROOT
    if fixture:
        require(fixture["library_sha256"] == library_digest(), "Fixture native library differs")
        require(fixture["groups"]["images"]["write_complete"], "Fixture image write is incomplete")
        cache_dir = Path(fixture["groups"]["images"]["cache_dir"])
    configured = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "FASTLLM_ACTIVATE_NUMA": "ON",
                  "FASTLLM_NUMA_THREADS": "27", "FASTLLM_CUDA_DFLASH_TP_BACKBONE": "force",
                  "FASTLLM_PREFIX_CACHE": "1", "FASTLLM_MULTIMODAL_PREFIX_CACHE": "1",
                  "FASTLLM_PREFIX_CACHE_DIR": str(cache_dir),
                  "FASTLLM_PREFIX_CACHE_DISK_BYTES": str(DISK_BYTES),
                  "FASTLLM_PREFIX_CACHE_RESTORE_POLICY": "always", "PYTHONUNBUFFERED": "1"}
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("FASTLLM_PREFIX_CACHE_"):
            env.pop(key)
    env.update(configured)
    command = command_for("fp8_e4m3") + ["--startup-progress", "ndjson"]
    module = Path(LIBRARY).parent / "persistent_prefix.py"
    result = {"status": "running", "label": args.label, "command": command,
              "environment": configured, "library_sha256": library_digest(),
              "identity_module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
              "events": [], "markers": {}, "stage_logs": [],
              "limitations": ["Single startup; OS page cache not cleared",
                              "Stage log observations polled every 0.1 seconds",
                              "Short first-request fixture does not validate long-context SSD restore"]}
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Startup deadline")))
    signal.alarm(args.timeout)
    proc = None
    monitors = []
    offset, pending = 0, ""

    def start_monitor(name, command):
        path = evidence / (name + ".log")
        with path.open("xb") as output:
            monitor = subprocess.Popen(["stdbuf", "-oL", "-eL", *command],
                env={**os.environ, "LC_ALL": "C", "TZ": "UTC", "S_TIME_FORMAT": "ISO", "S_COLORS": "never"},
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"name": name, "command": command, "path": str(path), "pid": monitor.pid,
                  "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        result["monitoring"]["collectors"].append(record)
        monitors.append((monitor, record))

    def stop_monitors():
        for monitor, record in monitors:
            record["unexpected_exit"] = monitor.poll() is not None
            if monitor.poll() is None:
                try:
                    os.killpg(monitor.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(monitor.pid, signal.SIGKILL)
                monitor.wait(timeout=5)
            record.update(returncode=monitor.returncode,
                          stopped_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          bytes=Path(record["path"]).stat().st_size)

    def collect():
        nonlocal offset, pending
        with logfile.open("rb") as source:
            source.seek(offset)
            data = source.read()
            offset = source.tell()
        lines = re.split(r"[\r\n]", pending + data.decode(errors="replace"))
        pending = lines.pop()
        for line in lines:
            observed = time.monotonic() - started
            if line.startswith("FTLLM_PROGRESS "):
                event = json.loads(line[len("FTLLM_PROGRESS "):])
                result["events"].append({"observed_seconds": observed, **event})
            elif any(marker in line for marker in ("[Prefix SSD]", "[Vision]", "[Fastllm]",
                                                    "[Qwen3.5 DFlash2]", "[Quantized prefill", "[FP8 prefill")):
                result["stage_logs"].append({"observed_seconds": observed, "line": line})
            for name, marker in (("hash_start", "[Prefix SSD] hashing model/runtime"),
                                 ("hash_end", "[Prefix SSD] identity="),
                                 ("uvicorn_ready", "Application startup complete.")):
                if marker in line:
                    result["markers"].setdefault(name, {"seconds": observed, "line": line})

    try:
        if args.monitor:
            for executable in ("stdbuf", "pidstat", "iostat", "mpstat", "vmstat", "nvidia-smi"):
                require(shutil.which(executable), "Missing monitoring tool: " + executable)
            result["monitoring"] = {"interval_seconds": 1, "collectors": [],
                "scope": "Whole-system CPU/disk/memory; GPUs 0-3; server PID and threads",
                "background_processes": subprocess.check_output(
                    ["ps", "-eo", "pid,comm,pcpu,pmem", "--sort=-pcpu"], text=True).splitlines()[:16]}
            result["limitations"].extend([
                "System CPU/disk metrics can include other services; process metrics are recorded separately",
                "One-second monitoring samples are not kernel-level tracing; sampler overhead is not subtracted"])
            start_monitor("iostat", ["iostat", "-x", "-m", "-t", "-y", "1"])
            start_monitor("mpstat", ["mpstat", "-P", "ALL", "1"])
            start_monitor("vmstat", ["vmstat", "-w", "-t", "-y", "1"])
            start_monitor("gpu", ["nvidia-smi", "-i", "0,1,2,3",
                "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits", "--loop-ms=1000"])
            time.sleep(2)
            for monitor, record in monitors:
                require(monitor.poll() is None, "Monitoring tool exited: " + record["path"])
            assert_resources_idle()
        with logfile.open("xb") as sink:
            result["started_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            started = time.monotonic()
            proc = subprocess.Popen(command, env=env, stdout=sink, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        result["pid"] = proc.pid
        if args.monitor:
            start_monitor("pidstat", ["pidstat", "-H", "-h", "-u", "-r", "-d", "-w", "-t",
                                      "-p", str(proc.pid), "1"])
            result["monitoring"]["pidstat_attached_seconds"] = time.monotonic() - started
        save_json(resultfile, result)
        print("Started owned server pid=%d; timing from process launch" % proc.pid, flush=True)
        next_status = started + 30
        while time.monotonic() - started < args.timeout - 90:
            collect()
            require(proc.poll() is None, "Server exited; see " + str(logfile))
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=1) as response:
                    ready = any(model["id"] == NAME for model in json.load(response)["data"])
                if ready:
                    result["http_ready_seconds"] = time.monotonic() - started
                    break
            except (OSError, ValueError, KeyError):
                pass
            if time.monotonic() >= next_status:
                save_json(resultfile, result)
                print("Startup still running: %.1fs" % (time.monotonic() - started), flush=True)
                next_status = time.monotonic() + 30
            time.sleep(0.1)
        else:
            raise TimeoutError("HTTP startup deadline")
        collect()
        print("HTTP ready: %.3fs" % result["http_ready_seconds"], flush=True)
        request_started = time.monotonic()
        response = chat([{"role": "user", "content": [image_url("red"),
                         {"type": "text", "text": "只回答图片的背景颜色。"}]}])
        require(response["ttft"] is not None and response["usage"].get("completion_tokens", 0) > 0,
                "First generation did not return model tokens")
        result.update(first_token_seconds=request_started - started + response["ttft"],
                      first_request_done_seconds=time.monotonic() - started, response=response)
        collect()
        if args.monitor:
            for monitor, record in monitors:
                require(monitor.poll() is None, "Monitoring collector exited early: " + record["path"])
        require("hash_start" in result["markers"] and "hash_end" in result["markers"],
                "Missing identity phase evidence")
        result["hash_seconds"] = (result["markers"]["hash_end"]["seconds"] -
                                  result["markers"]["hash_start"]["seconds"])
        if args.compare:
            reference = json.loads(args.compare.read_text())
            require(reference["status"] == "ok", "Reference startup did not pass")
            require(result["library_sha256"] == reference["library_sha256"], "Native runtime changed")
            identities = lambda record: dict(re.findall(r"\b(identity|family)=([0-9a-f]{64})",
                                                        record["markers"]["hash_end"]["line"]))
            require(len(identities(result)) == 2 and identities(result) == identities(reference),
                    "Persistent cache identity/configuration changed")
            result["response_equal"] = all(response[key] == reference["response"][key]
                                           for key in ("content", "reasoning", "finish", "usage"))
            require(result["response_equal"], "First model response differs from baseline")
        if fixture:
            result["requests"] = []
            messages = make_inputs(fixture["nonce"])["images"]["repeat"]
            actual, _ = send_request(result, "existing-ssd/repeat", messages, logfile)
            expected = fixture["groups"]["images"]["references"]["repeat"]
            boundary = expected["required_restore_tokens"]
            require(actual["cached_tokens"] == boundary and
                    any(event["tokens"] == boundary for event in events(actual["markers"], "restored")),
                    "Existing SSD checkpoint did not restore")
            require(equal_output(actual, expected), "Existing SSD restore output differs from cold reference")
            actual["output_equal"] = True
            result["restore_fixture"] = str(args.restore_fixture)
            result["limitations"] = [line for line in result["limitations"]
                                     if not line.startswith("Short first-request fixture")]
            collect()
        result["status"] = "ok"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            stop_monitors()
        finally:
            if proc is not None:
                stop_owned(proc)
                result["returncode"] = proc.returncode
                collect()
            save_json(resultfile, result)
    signal.alarm(0)
    print(json.dumps({key: result[key] for key in ("status", "http_ready_seconds", "hash_seconds",
                     "first_token_seconds", "first_request_done_seconds")}), flush=True)


if __name__ == "__main__":
    main()
