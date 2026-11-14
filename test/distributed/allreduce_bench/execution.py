"""隔离每个测试组并有界回收全部后代；控制面时间不作为算子延迟。"""

import json
from pathlib import Path
import sys

from .reporting import ROOT, write_report

sys.path.insert(0, str(ROOT / "tools"))
from tp_process_utils import isolated_process, wait_case


def collect(job, directory):
    results = []
    ranks = []
    for rank in range(len(job["devices"])):
        path = directory / f"rank{rank}.jsonl"
        records = []
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # 超时终止可能留下半行；保留已完整输出的 case，缺项随后判 FAIL。
        ranks.append(records)
    for index, spec in enumerate(job["cases"]):
        records = [rows[index] for rows in ranks if len(rows) > index]
        if len(records) != len(ranks) or any(row["case"] != spec["case"] for row in records):
            results.append(dict(stage=job["stage"], case=spec["case"], backend=job["backend"],
                                mode=job["mode"], devices=job["devices"], n=spec["n"], status="FAIL",
                                reason="缺少完整的各 rank 结果；见原始 worker 日志"))
            break
        row = dict(records[0])
        row["ranks"] = records
        row["max_abs_error"] = max(r.get("max_abs_error", 0) for r in records)
        row["mean_abs_error"] = sum(r.get("mean_abs_error", 0) for r in records) / len(records)
        row["failed_elements"] = sum(r.get("failed_elements", 0) for r in records)
        row["status"] = "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL"
        if len({r.get("path") for r in records}) != 1:
            row.update(status="FAIL", reason="各 rank 的实际执行路径不一致")
        results.append(row)
    return results


def run_job(job, args, directory):
    directory.mkdir(parents=True, exist_ok=False)
    job["directory"] = str(directory.resolve())
    path = directory / "job.json"
    path.write_text(json.dumps(job, indent=2))
    command = [args.python, "-m", "allreduce_bench.worker", str(path.resolve())]
    failure = None
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        try:
            with isolated_process(command, cwd=ROOT / "test/distributed", stdout=stdout, stderr=stderr) as process:
                code = wait_case(process, args.timeout)
                if code:
                    failure = f"worker group exit {code}"
        except Exception as exc:
            failure = str(exc)
    rows = collect(job, directory)
    if failure:
        rows.append(dict(stage=job["stage"], case="worker_failure", backend=job["backend"],
                         mode=job["mode"], devices=job["devices"], status="FAIL", reason=failure))
    return rows


def checkpoint(output, rows, meta):
    meta["worker_metadata"] = [json.loads(p.read_text()) for p in sorted(output.glob("*/rank*.metadata.json"))]
    write_report(output, rows, meta, status="RUNNING")


def smoke(args, output, meta):
    from .plan import devices

    pair = devices(args.pair)
    if len(pair) != 2:
        raise ValueError("主比较要求一组两卡")
    if not Path(args.native).is_file():
        raise FileNotFoundError("先在服务器显式构建 native adapter: " + args.native)
    rows = []
    for backend in ("ft", "vllm", "nccl"):
        job = dict(stage="smoke", backend=backend, mode="eager", devices=pair,
                   native=str(Path(args.native).resolve()), timeout=args.timeout,
                   seed=20261114, strict_sizes=[4, 4096],
                   cases=[dict(case=f"smoke_n{n}", n=n, pattern="known") for n in (4, 4096)])
        rows.extend(run_job(job, args, output / backend))
        checkpoint(output, rows, meta)
    meta["native"] = str(Path(args.native).resolve())
    return rows


def correctness(args, output, meta):
    from .plan import BACKENDS, CORE_SIZES, MODES, SEED, correctness_cases, devices

    pair = devices(args.pair)
    if len(pair) != 2 or args.stress_iterations <= 0:
        raise ValueError("correctness 要求两卡和正数 stress-iterations")
    if not Path(args.native).is_file():
        raise FileNotFoundError(args.native)
    rows = []
    for group in ([pair[0]], devices(args.devices)):
        if len(group) == 2:
            continue
        job = dict(stage="correctness", backend="ft", mode="eager", devices=group,
                   native=str(Path(args.native).resolve()), timeout=args.timeout, seed=SEED,
                   expected_init_error="custom AllReduce requires TP=2",
                   cases=[dict(case=f"invalid_world_{len(group)}", n=0)])
        rows.extend(run_job(job, args, output / f"invalid_world_{len(group)}"))
        checkpoint(output, rows, meta)
    for mode in MODES:
        for backend in BACKENDS:
            cases = correctness_cases(mode, args.stress_iterations)
            if backend == "ft" and mode == "eager":
                cases[:0] = [dict(case="reject_zero", n=0, action="reject_zero"),
                             dict(case="after_rejection", n=4, pattern="known")]
            job = dict(stage="correctness", backend=backend, mode=mode, devices=pair,
                       native=str(Path(args.native).resolve()), timeout=args.timeout, seed=SEED,
                       strict_sizes=CORE_SIZES, cases=cases)
            rows.extend(run_job(job, args, output / f"{backend}_{mode}"))
            checkpoint(output, rows, meta)
    meta["scope"] = "TP2 FP32; independent CPU reference; all ranks checked"
    meta["worker_metadata"] = [json.loads(p.read_text()) for p in sorted(output.glob("*/rank*.metadata.json"))]
    return rows
