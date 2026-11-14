"""两组独立 TP2 的共同起跑、重叠证据和墙钟吞吐。"""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

from .execution import checkpoint, collect, isolated_process, require_correctness, run_job, wait_case, worker_environment
from .reporting import ROOT


def gate_wait(job, case, rank):
    directory = Path(job["gate"])
    name = case["case"]
    (directory / f"{name}.g{job['group_index']}.r{rank}.ready").touch()
    release = directory / f"{name}.go"
    deadline = time.monotonic() + job["timeout"]
    while not release.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("双组起跑等待超时")
        time.sleep(0.001)


def overlap_metrics(rows):
    intervals = [(min(r["batch_start_ns"] for r in row["ranks"]),
                  max(r["batch_end_ns"] for r in row["ranks"])) for row in rows]
    durations = [end - start for start, end in intervals]
    if len(intervals) != 2 or min(durations) <= 0:
        raise ValueError("需要两组有效的批次时间区间")
    overlap = max(0, min(end for _, end in intervals) - max(start for start, _ in intervals))
    span = max(end for _, end in intervals) - min(start for start, _ in intervals)
    return {"overlap_ratio": overlap / min(durations),
            "wall_calls_per_second": sum(row["iterations"] for row in rows) * 1e9 / span,
            "batch_span_ms": span / 1e6}


def run_parallel(jobs, args, directory):
    directory.mkdir(parents=True, exist_ok=False)
    gate = directory / "gate"
    gate.mkdir()
    processes, directories = [], []
    failure = None
    with ExitStack() as stack:
        for index, job in enumerate(jobs):
            child = directory / f"group{index}"
            child.mkdir()
            directories.append(child)
            job.update(directory=str(child.resolve()), gate=str(gate.resolve()), group_index=index)
            environment = worker_environment(job)
            path = child / "job.json"
            path.write_text(json.dumps(job, indent=2))
            stdout = stack.enter_context((child / "stdout.log").open("w"))
            stderr = stack.enter_context((child / "stderr.log").open("w"))
            command = [args.python, "-m", "allreduce_bench.worker", str(path.resolve())]
            process = stack.enter_context(isolated_process(command, cwd=ROOT / "test/distributed", env=environment,
                                                             stdout=stdout, stderr=stderr))
            processes.append(process)
        try:
            for case in jobs[0]["cases"]:
                deadline = time.monotonic() + args.timeout
                ready = [gate / f"{case['case']}.g{g}.r{rank}.ready" for g in range(2) for rank in range(2)]
                while not all(path.exists() for path in ready):
                    for process in processes:
                        exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                        if exited is not None:
                            raise RuntimeError("某一并发组提前退出，终止两组并保留日志")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("并发组未全部准备好")
                    time.sleep(0.01)
                (gate / f"{case['case']}.go").touch()
            for process in processes:
                code = wait_case(process, args.timeout)
                if code:
                    raise RuntimeError(f"concurrent worker exit {code}")
        except Exception as exc:
            failure = str(exc)
    rows = []
    for job, child in zip(jobs, directories):
        rows.extend(collect(job, child))
    if failure:
        rows.append(dict(case="concurrent_worker_failure", stage="concurrent", status="FAIL", reason=failure))
    for case in jobs[0]["cases"]:
        peers = [row for row in rows if row["case"] == case["case"] and row["status"] == "PASS"]
        if len(peers) == 2:
            overlap = overlap_metrics(peers)
            for row in peers:
                row.update(overlap)
                if overlap["overlap_ratio"] < 0.5:
                    row.update(status="FAIL", reason="两组批次窗口重叠不足 50%，不能作为并发成绩")
    return rows


def run(args, output, meta):
    from .plan import BACKENDS, MODES, SEED, devices

    groups = [devices(group) for group in args.groups.split(";")]
    if len(groups) != 2 or any(len(g) != 2 for g in groups) or len(set(sum(groups, []))) != 4:
        raise ValueError("--groups 必须是两组互不重叠的两卡，例如 0,1;2,3")
    if groups[0] != devices(args.pair):
        raise ValueError("--pair 必须与 --groups 的第一组一致")
    if args.repeats < 5 or min(args.iterations, args.warmup) < 1:
        raise ValueError("并发测试要求至少五轮且 iterations/warmup 为正")
    require_correctness(args, meta)
    rows = []
    cases = [dict(case=f"concurrent_n{n}", n=n, pattern="random") for n in (256, 4096, 65536)]
    for repeat in range(args.repeats):
        for mode in MODES:
            for backend in BACKENDS[repeat % 3:] + BACKENDS[:repeat % 3]:
                jobs = [dict(stage="concurrent", backend=backend, mode=mode, devices=pair,
                             native=str(Path(args.native).resolve()), timeout=args.timeout, seed=SEED,
                             strict_sizes=[256, 4096, 65536], cases=cases, measure=True,
                             warmup=args.warmup, iterations=args.iterations, repeat=repeat,
                             expected_runtime=meta["correctness_runtime"], runtime_device_match=index == 0)
                        for index, pair in enumerate(groups)]
                isolated = []
                for index, job in enumerate(jobs):
                    job["scenario"] = "isolated"
                    isolated.extend(run_job(job, args, output / f"r{repeat}_{backend}_{mode}_solo{index}"))
                rows.extend(isolated)
                checkpoint(output, rows, meta)
                if any(row["status"] != "PASS" for row in isolated):
                    return rows
                for job in jobs:
                    job["scenario"] = "concurrent"
                together = run_parallel(jobs, args, output / f"r{repeat}_{backend}_{mode}_together")
                for row in together:
                    reference = next((r for r in isolated if r["case"] == row["case"] and
                                      r.get("devices") == row.get("devices") and r["status"] == "PASS"), None)
                    if row["status"] == "PASS" and reference:
                        row["slowdown"] = row["median_us"] / reference["median_us"]
                rows.extend(together)
                checkpoint(output, rows, meta)
                if any(row["status"] != "PASS" for row in together):
                    return rows
    meta["concurrency_notes"] = ["two independent TP2 groups, not one TP4 collective",
                                  "both groups validate each measured shape before timing",
                                  "wall throughput includes identical input reset and host submission; distinct from event latency",
                                  "overlap measures batch wall-clock windows, not a hardware kernel concurrency trace"]
    return rows
