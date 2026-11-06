"""Native/Machete共享报告：保留原始样本，展示值固定三位小数。"""
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test"))
sys.path.insert(0, str(REPO / "test/nvfp4"))
from xlsx_report import write_xlsx
from common.performance_report import ChartSpec, write_dashboard


def metadata(command):
    result = {"command": command, "seed": 0x9e3779b9,
              "scale_policy": "source FP32 -> activation dtype; offset=-scale*zero or min",
              "speedup_definition": "native_ms / machete_ms; >1 means Machete faster"}
    for key, cmd in {
        "git": ["git", "rev-parse", "HEAD"],
        "git_status": ["git", "status", "--short"],
        "gpu": ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
    }.items():
        try:
            result[key] = subprocess.check_output(cmd, cwd=REPO, text=True).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            result[key] = str(exc)
    return result


def timing(samples):
    if len(samples) < 5 or any(x <= 0 for x in samples):
        raise ValueError("至少5轮有效耗时样本")
    ordered = sorted(samples)
    index = (len(ordered) - 1) * .95
    lo = int(index)
    p95 = ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (index - lo)
    return {"median_ms": statistics.median(samples), "p95_ms": p95,
            "cv": statistics.pstdev(samples) / statistics.mean(samples)}


def save(root, rows, meta, images=()):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    status = "FAIL" if not rows or any(r.get("status") == "FAIL" for r in rows) else (
        "INCOMPLETE" if any(r.get("status") == "UNPAIRED" for r in rows) or
        not any(r.get("status") == "PASS" for r in rows) else "PASS")
    (root / "summary.json").write_text(json.dumps(
        {"status": status, "metadata": meta, "cases": rows}, ensure_ascii=False, indent=2))
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]

    def cell(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, float):
            return f"{value:.3f}"
        return value

    display = [[cell(row.get(key)) for key in fields] for row in rows]
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(display)
    lines = [f"# Native / Machete\n\nSummary: {status}",
             "\n加速比为 native / machete；大于1表示Machete更快。",
             "\n| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |" for row in display)
    lines.extend(f"\n![{Path(p).stem}]({Path(p).relative_to(root)})" for p in images)
    (root / "summary.md").write_text("\n".join(lines))
    write_xlsx(root / "summary.xlsx", [
        ("Cases", fields, display),
        ("Metadata", ["key", "value"], [[k, cell(v)] for k, v in meta.items()]),
    ], image_sheets=[("Trends", images)] if images else None)
    return status
