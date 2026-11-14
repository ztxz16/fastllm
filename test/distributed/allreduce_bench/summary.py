"""合并实际结果，保持模式、卡对、路径和计时范围的比较边界。"""

from collections import defaultdict
import json
from pathlib import Path
import statistics

from .timing import statistics_for


def aggregate(rows, repeats):
    groups = defaultdict(list)
    result = []
    for row in rows:
        if "samples_us" not in row:
            if row["status"] != "PASS":
                result.append(row)
            continue
        key = (row["stage"], row["backend"], row["mode"], tuple(row["devices"]),
               row["n"], row.get("scenario", "single"))
        groups[key].append(row)
    for key, rounds in groups.items():
        stage, backend, mode, pair, n, scenario = key
        row = dict(stage=stage, case=f"{stage}_n{n}", backend=backend, mode=mode,
                   devices=list(pair), n=n, scenario=scenario, status="PASS")
        round_ids = {r.get("repeat") for r in rounds}
        signatures = {(r.get("warmup"), r.get("iterations"), r.get("path")) for r in rounds}
        if (len(rounds) != repeats or round_ids != set(range(repeats)) or
                len(signatures) != 1 or any(r["status"] != "PASS" for r in rounds)):
            row.update(status="FAIL", reason="轮数不完整、重复轮次、配置/路径变化或某轮失败")
        else:
            samples = [sample for r in rounds for sample in r["samples_us"]]
            medians = [r["median_us"] for r in rounds]
            row.update(statistics_for(samples, medians), rounds=[{
                "repeat": r["repeat"], "median_us": r["median_us"], "samples_us": r["samples_us"]} for r in rounds],
                warmup=rounds[0]["warmup"], iterations=rounds[0]["iterations"], repeats=repeats,
                path=rounds[0]["path"], max_abs_error=max(r["max_abs_error"] for r in rounds),
                mean_abs_error=statistics.mean(r["mean_abs_error"] for r in rounds),
                failed_elements=sum(r["failed_elements"] for r in rounds))
            for field in ("slowdown", "wall_calls_per_second", "overlap_ratio"):
                if all(field in r for r in rounds):
                    row[field] = statistics.median(r[field] for r in rounds)
        result.append(row)
    for row in result:
        if row["status"] != "PASS" or row.get("backend") == "ft":
            continue
        matching = ("stage", "mode", "devices", "n", "scenario", "warmup", "iterations", "repeats")
        baseline = next((other for other in result if other.get("backend") == "ft" and
                         other["status"] == "PASS" and all(other.get(k) == row.get(k) for k in matching)), None)
        if baseline:
            row["speedup"] = row["median_us"] / baseline["median_us"]
            row["comparison_paths"] = {"ft": baseline["path"], row["backend"]: row["path"]}
    return result


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = [r for r in rows if r["stage"] == "performance" and r["status"] == "PASS"]
    images = []
    for mode, pair in sorted({(r["mode"], tuple(r["devices"])) for r in eligible}):
        fig, axis = plt.subplots(figsize=(8, 5))
        for backend in ("ft", "vllm", "nccl"):
            series = sorted((r for r in eligible if r["backend"] == backend and r["mode"] == mode and
                             tuple(r["devices"]) == pair), key=lambda r: r["n"])
            if series:
                axis.plot([r["n"] * 4 for r in series], [r["median_us"] for r in series],
                          marker="o", label=backend)
        axis.axvline(65536 * 4, linestyle="--", color="grey", label="FT custom limit")
        axis.set(xscale="log", yscale="log", xlabel="FP32 bytes per rank", ylabel="Median latency (us)",
                 title=f"AllReduce / {mode} / devices {pair}")
        axis.grid(True, which="both", alpha=0.2)
        axis.legend()
        fig.tight_layout()
        name = f"latency_{mode}_{'_'.join(map(str, pair))}"
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=160)
            images.append(path.name)
        plt.close(fig)
    return images


def run(args, output, meta):
    if not args.input_root:
        raise ValueError("report 要求 --input-root 指向本次七步报告根目录")
    root = Path(args.input_root).resolve()
    result = []
    meta["sources"] = []
    for name in ("01-plan", "02-preflight", "03-smoke", "04-correctness", "05-performance", "06-concurrent"):
        path = root / name / "summary.json"
        if not path.is_file():
            result.append(dict(stage="report", case=name, status="FAIL", reason="缺少阶段报告：" + str(path)))
            continue
        report = json.loads(path.read_text())
        meta["sources"].append({"path": str(path), "status": report["status"], "metadata": report["metadata"]})
        required_status = "PLANNED" if name == "01-plan" else "PASS"
        if report["status"] != required_status:
            result.append(dict(stage="report", case=name, status="FAIL", reason="源阶段未完整通过"))
        if name in ("05-performance", "06-concurrent"):
            result.extend(aggregate(report["cases"], report["metadata"]["options"]["repeats"]))
        elif name != "01-plan":
            result.extend(report["cases"])
    meta["statistics"] = {"median_us": "median of independent round medians",
                          "p95_us": "95th percentile of pooled rank-max samples",
                          "cv_percent": "sample standard deviation / mean of round medians * 100",
                          "speedup": "comparison median / FT median; >1 means FT faster",
                          "wall_calls_per_second": "combined two-group rate; duplicated in the two group rows, do not sum again"}
    if not args.no_plots and any(r["stage"] == "performance" and r["status"] == "PASS" for r in result):
        try:
            meta["plots"] = plot(result, output)
        except ImportError as exc:
            meta["plot_error"] = str(exc)
            result.append(dict(stage="report", case="plots", status="FAIL",
                               reason="当前解释器缺少 matplotlib；数值结果已保留，可配置环境后重跑 report"))
    else:
        meta["plots"] = []
    return result
