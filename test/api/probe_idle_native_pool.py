#!/usr/bin/env python3
"""Measure idle CPU cost of the installed native pool, without loading a model.

Separate bounded child processes compare importing ftllm alone with creating
32 CPU workers. Import may itself create workers through NUMA initialization.
No operations are submitted to those workers. No GPU is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def snapshot():
    hz = os.sysconf("SC_CLK_TCK")
    threads = {}
    for path in Path("/proc/self/task").iterdir():
        try:
            raw = (path / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            threads[path.name] = (int(fields[11]) + int(fields[12])) / hz
        except FileNotFoundError:
            pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return time.monotonic(), usage.ru_utime + usage.ru_stime, threads


def child(threads):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ftllm import llm
    original = snapshot()[2]
    configured_before = llm.get_cpu_threads()
    if threads:
        llm.set_cpu_threads(threads)
    # The pool starts with a 3-second spin period. Measure the steady idle
    # behavior separately, after that period; do not submit any operations.
    time.sleep(4)
    before = snapshot()
    time.sleep(5)
    after = snapshot()
    wall = after[0] - before[0]
    rows = [{"tid": tid, "created_by_pool": tid not in original,
             "cpu_seconds": seconds - before[2].get(tid, seconds)}
            for tid, seconds in after[2].items()]
    library = Path(llm.fastllm_lib._name)
    record = {"threads_requested": threads, "wall_seconds": wall,
              "native_threads_before": configured_before, "native_threads_after": llm.get_cpu_threads(),
              "environment": {name: os.environ.get(name) for name in
                              ("FASTLLM_ACTIVATE_NUMA", "FASTLLM_NUMA_THREADS", "FT_THREADS")},
              "process_cpu_seconds": after[1] - before[1],
              "process_cpu_percent": 100 * (after[1] - before[1]) / wall,
              "threads_before": len(original), "threads_after": len(after[2]),
              "created_thread_cpu_seconds": sum(row["cpu_seconds"] for row in rows if row["created_by_pool"]),
              "threads": rows, "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
    print("IDLE_POOL_RESULT " + json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-threads", type=int, choices=(0, 32))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child_threads is not None:
        child(args.child_threads)
        return
    if args.output is None:
        parser.error("Provide --output")
    if args.output.exists():
        raise RuntimeError("Preserve earlier evidence; choose another output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "runs": [],
              "limitations": ["Controlled idle-pool reproduction, not a deployment call-stack trace",
                              "Single ordered pair; CPU cost alone does not predict startup seconds saved"]}
    try:
        for count in (0, 32):
            proc = subprocess.run([sys.executable, __file__, "--child-threads", str(count)],
                                  capture_output=True, text=True, timeout=30)
            args.output.with_name(args.output.stem + "-threads-%d.log" % count).write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                raise RuntimeError("Idle probe child failed: %d" % proc.returncode)
            records = [line[len("IDLE_POOL_RESULT "):] for line in proc.stdout.splitlines()
                       if line.startswith("IDLE_POOL_RESULT ")]
            if len(records) != 1:
                raise RuntimeError("Missing idle-pool record")
            record = json.loads(records[0])
            result["runs"].append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "threads"}), flush=True)
        result["status"] = "ok"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
