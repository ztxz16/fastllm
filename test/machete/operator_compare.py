#!/usr/bin/env python3
"""通过正式Linear测试程序比较Native/Machete；可选Nsight，报告中保留失败case。"""
import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from report import ChartSpec, metadata, save, timing, write_dashboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--batch-sizes", default="1,7,8,9,16,17,32,64")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--nsight", action="store_true")
    parser.add_argument("--nsys", default="nsys")
    args = parser.parse_args()
    if args.repeats < 5 or args.iterations < 1:
        parser.error("repeats至少5，iterations至少1")
    modes = args.modes.split(",")
    if any(mode not in ("eager", "graph") for mode in modes):
        parser.error("modes只能为eager,graph")
    batches = list(dict.fromkeys(int(x) for x in args.batch_sizes.split(",")))
    if not batches or any(x <= 0 for x in batches):
        parser.error("batch-sizes必须为正整数")
    root = Path(args.result_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    meta = metadata(shlex.join(sys.argv))
    meta.update(warmup=5, repeats=args.repeats, iterations=args.iterations,
                timing="CUDA events on production per-thread stream",
                nsys_artifacts="temporary .nsys-rep/.sqlite; summary CSV retained")
    rows, images = [], []

    def execute(bits, atype, batch, group, bias, mode, backend, n=None, expected=None, quant_mode="symmetric"):
        n = args.n if n is None else n
        name = f"w{bits}a16-{atype}-m{batch}-n{n}-k{args.k}-g{group}-bias{bias}-{mode}-{backend}-{quant_mode}"
        command = [str(Path(args.executable).resolve()), backend, str(bits), atype,
                   str(batch), str(n), str(args.k), str(group), str(bias),
                   str(args.repeats), str(args.iterations), str(int(mode == "graph")), str(args.device), quant_mode]
        row = dict(case=name, backend=backend, bits=bits, atype=atype, batch=batch,
                   n=n, k=args.k, group=group, bias=bias, mode=mode, quant_mode=quant_mode, command=shlex.join(command))
        try:
            with tempfile.TemporaryDirectory(prefix="fastllm-machete-nsys-") as tmp:
                run = command
                if args.nsight and not expected:
                    run = [args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none",
                           "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                           "-o", str(Path(tmp) / "profile"), *command]
                process = subprocess.run(run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                text = process.stdout
                (root / f"{name}.log").write_text(text)
                if expected:
                    row.update(status="PASS" if process.returncode != 0 and expected in text else "FAIL",
                               rejection_test=True, expected_reason=expected, exit_code=process.returncode)
                else:
                    records = re.findall(r"^MACHETE_RESULT (.+)$", text, re.M)
                    if not records:
                        raise RuntimeError(f"exit={process.returncode}; missing result; see {name}.log")
                    data = json.loads(records[-1])
                    if not re.search(rf"\[linear-backend\].*actual={backend}\b", text):
                        raise RuntimeError("未确认目标后端")
                    row.update(data)
                    row.update(timing(data["samples_ms"]))
                    row["effective_tops"] = 2 * batch * n * args.k / (row["median_ms"] * 1e9)
                    if process.returncode:
                        row["status"] = "FAIL"
                    if args.nsight:
                        reports = list(Path(tmp).glob("*.nsys-rep"))
                        if len(reports) != 1:
                            raise RuntimeError("Nsight report missing")
                        for report in ("cuda_gpu_kern_sum", "cuda_api_sum"):
                            stats = subprocess.run([args.nsys, "stats", "--report", report,
                                                    "--format", "csv", str(reports[0])],
                                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                            (root / f"{name}-{report}.csv").write_text(stats.stdout)
                            if stats.returncode:
                                raise RuntimeError(f"nsys stats {report} failed")
        except Exception as exc:
            row.update(status="FAIL", failure=str(exc))
        rows.append(row)
        save(root, rows, meta, images)

    for bits in (4, 8):
        for atype in ("fp16", "bf16"):
            for group in (sorted(set((64, 128, args.k))) if bits == 4 else (args.k,)):
                if args.k % group:
                    continue
                for bias in (0, 1):
                    for mode in modes:
                        for batch in batches:
                            for backend in ("native", "machete"):
                                if backend == "native" and atype == "bf16":
                                    rows.append(dict(case=f"w{bits}-{atype}-g{group}-b{batch}-{mode}-bias{bias}-native",
                                                     status="UNSUPPORTED", backend="native",
                                                     reason="native dispatch lacks this exact BF16 INT8/INT4_GROUP combination"))
                                    continue
                                execute(bits, atype, batch, group, bias, mode, backend)
                        selected = [r for r in rows if r.get("status") == "PASS" and
                                    (r.get("bits"), r.get("atype"), r.get("group"), r.get("bias"), r.get("mode")) ==
                                    (bits, atype, group, bias, mode)]
                        for row in selected:
                            peer = next((p for p in selected if p["batch"] == row["batch"] and p["backend"] == "native"), None)
                            if row["backend"] == "machete" and peer:
                                row["machete_speedup_vs_native_x"] = peer["median_ms"] / row["median_ms"]
                        if selected:
                            images.append(write_dashboard(root / "images" / f"w{bits}-{atype}-g{group}-bias{bias}-{mode}.png",
                                selected, [ChartSpec("MEDIAN", "median_ms", "ms"),
                                           ChartSpec("P95", "p95_ms", "ms"),
                                           ChartSpec("EFFECTIVE TOPS", "effective_tops", "TOPS"),
                                           ChartSpec("MAX ABS ERROR", "max_abs_error", "")],
                                f"W{bits}A16 {atype.upper()} {mode.upper()}"))
    # 非法形状和不支持的分组必须在正式入口拒绝，而不是静默fallback。
    for bits in (4, 8):
        for atype in ("fp16", "bf16"):
            for quant_mode in (("asymmetric", "min") if bits == 4 else ("asymmetric",)):
                for backend in ("native", "machete"):
                    if backend == "native" and atype == "bf16":
                        continue
                    execute(bits, atype, 8, args.k, 1, "eager", backend, quant_mode=quant_mode)
    execute(4, "fp16", 1, args.k, 0, "eager", "machete", n=64,
            expected="requires_N_multiple128_K_multiple64")
    if args.k % 32 == 0 and args.k != 32:
        execute(4, "fp16", 1, 32, 0, "eager", "machete", expected="unsupported_group_size")
    status = save(root, rows, meta, images)
    print(f"Summary: {status}\nReport: {root / 'summary.xlsx'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
