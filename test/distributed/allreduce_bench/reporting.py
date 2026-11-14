"""无第三方依赖的报告写入；保留部分失败结果及原始精度。"""

import csv
import json
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test" / "nvfp4"))
from xlsx_report import write_xlsx

FIELDS = ["stage", "case", "backend", "mode", "devices", "n", "scenario", "repeat", "status", "path",
          "max_abs_error", "mean_abs_error", "failed_elements", "median_us",
          "p95_us", "cv_percent", "calls_per_second", "speedup", "slowdown",
          "overlap_ratio", "wall_calls_per_second", "reason"]


def capture(command, timeout=15):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {"command": shlex.join(command), "returncode": p.returncode,
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": shlex.join(command), "returncode": -1, "error": str(exc)}


def metadata(args):
    return {"command": shlex.join([sys.executable, *sys.argv]),
            "git": capture(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "git_dirty": capture(["git", "-C", str(ROOT), "status", "--short"]),
            "python": sys.version, "options": vars(args),
            "environment": {key: value for key, value in os.environ.items()
                            if key.startswith("NCCL_") or key in (
                                "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "PYTORCH_CUDA_ALLOC_CONF",
                                "VLLM_CUSTOM_ALLREDUCE_ALGO", "VLLM_BATCH_INVARIANT")}}


def artifact_signature(args):
    native = Path(args.native).resolve()
    return {"native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "ft_source_sha256": hashlib.sha256((ROOT / "src/distributed/process_allreduce.cu").read_bytes()).hexdigest(),
            "data_generator_sha256": hashlib.sha256(Path(__file__).with_name("data.py").read_bytes()).hexdigest()}


def cell(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_report(output, rows, meta, status=None):
    output = Path(output)
    if status is None:
        status = "PASS" if rows and all(r["status"] == "PASS" for r in rows) else "FAIL"
    report = {"schema": 1, "status": status, "metadata": meta, "cases": rows}
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    display = [[cell(row.get(key)) for key in FIELDS] for row in rows]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(display)
    lines = [f"# AllReduce operator comparison: {status}", "",
             "仅代表本报告实际执行的算子配置，不代表模型推理性能。", "",
             "| " + " | ".join(FIELDS) + " |",
             "| " + " | ".join(["---"] * len(FIELDS)) + " |"]
    lines += ["| " + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in row) + " |"
              for row in display]
    lines += ["", "```json", json.dumps(meta, ensure_ascii=False, indent=2), "```"]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_xlsx(output / "summary.xlsx", [
        ("Cases", FIELDS, display),
        ("Metadata", ["key", "value"], [[k, cell(v)] for k, v in meta.items()]),
    ])
    return status
