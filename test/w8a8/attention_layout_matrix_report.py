#!/usr/bin/env python3
"""汇总Continuous/Paged Attention测试矩阵的结构化结果和趋势图。"""

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test"))
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))

from common.performance_report import (  # noqa: E402
    ChartSpec, markdown_images, model_performance_chart_specs, write_dashboard)
from xlsx_report import write_xlsx  # noqa: E402


MATRIX = (
    ("continuous", "auto"),
    ("paged", "native_paged"),
    ("paged", "flashinfer_paged"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="汇总Attention布局测试矩阵")
    parser.add_argument("--kind", required=True,
                        choices=("model-performance", "decode-nsys"))
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def metric_ms(row, name):
    value = row.get(name)
    return None if value is None else value * 1000.0


def model_records(payload, layout, requested_backend):
    records = []
    for result_backend in ("fastllm", "vllm"):
        for row in payload.get(result_backend, []):
            records.append({
                "status": "PASS",
                "layout": layout,
                "requested_attention_backend": requested_backend,
                "series": f"{layout}-{requested_backend}",
                "backend": result_backend,
                "mode": row["mode"],
                "scenario": row["scenario"],
                "workload": row["workload"],
                "batch": row["batch"],
                "actual_kv_cache_layouts": ",".join(
                    row.get("actual_kv_cache_layouts", [])),
                "actual_attention_backends": ",".join(
                    row.get("actual_attention_backends", [])),
                "ttft_ms": metric_ms(row, "ttft_s"),
                "tpot_ms": metric_ms(row, "tpot_s"),
                "itl_ms": metric_ms(row, "itl_s"),
                "e2el_ms": metric_ms(row, "e2el_s"),
                "prefill_tok_s": row.get("prefill_tok_s"),
                "output_tok_s": row.get("output_tok_s"),
            })
    return records


def nsys_records(payload, layout, requested_backend):
    records = []
    for row in payload:
        nsys = row["nsys"]
        timeline = nsys["timeline"]
        cpu = nsys.get("cpu_trace") or {}
        decode_steps = nsys["decode_steps"]
        launch_count = cpu.get("launch_api_count")
        records.append({
            "status": row.get("result", "PASS"),
            "layout": layout,
            "requested_attention_backend": requested_backend,
            "series": f"{layout}-{requested_backend}",
            "backend": row["backend"],
            "batch": row["batch"],
            "actual_kv_cache_layouts": ",".join(
                row.get("actual_kv_cache_layouts", [])),
            "actual_attention_backends": ",".join(
                row.get("actual_attention_backends", [])),
            "tpot_ms": row.get("tpot_ms"),
            "output_tok_s": row.get("output_tok_s"),
            "kernel_ms_per_step": nsys.get("kernel_ms_per_decode_step"),
            "kernel_count_per_step": nsys.get("kernel_instances_per_decode_step"),
            "gpu_idle_pct": (None if timeline.get("idle_ratio") is None else
                             timeline["idle_ratio"] * 100.0),
            "submit_ms_per_step": cpu.get("launch_api_union_ms_per_decode_step"),
            "submit_count_per_step": (None if launch_count is None else
                                      launch_count / decode_steps),
        })
    return records


def load_matrix(kind, result_dir):
    filename = ("model-performance-compare.json" if kind == "model-performance"
                else "decode-nsys-compare.json")
    records = []
    cases = []
    for layout, backend in MATRIX:
        case_dir = result_dir / f"{layout}-{backend}"
        source = case_dir / filename
        case = {
            "layout": layout,
            "attention_backend": backend,
            "strict": backend != "auto",
            "status": "PASS" if source.is_file() else "FAIL",
            "report_directory": str(case_dir),
        }
        if source.is_file():
            payload = json.loads(source.read_text(encoding="utf-8"))
            loader = model_records if kind == "model-performance" else nsys_records
            records.extend(loader(payload, layout, backend))
        else:
            case["failure"] = f"missing {filename}"
        cases.append(case)
    return cases, records


def write_csv(path, records, headers):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: (f"{value:.3f}" if isinstance(value, float) else value)
                for key, value in record.items()
            })


def create_charts(kind, result_dir, records):
    image_dir = result_dir / "images"
    fastllm = [row for row in records if row["backend"] == "fastllm"]
    paths = []
    if kind == "model-performance":
        groups = sorted({(row["mode"], row["scenario"], view)
                         for row in fastllm for view in ("prefill", "decode")})
        for mode, scenario, workload in groups:
            rows = [row for row in fastllm
                    if (row["mode"], row["scenario"]) == (mode, scenario)]
            name = f"model-layout-{mode}-{scenario}-{workload}.png"
            paths.append(write_dashboard(
                image_dir / name, rows,
                model_performance_chart_specs(workload),
                f"FASTLLM LAYOUT {mode} {scenario} {workload}",
                backend_field="series"))
    elif fastllm:
        paths.append(write_dashboard(
            image_dir / "decode-nsys-layout-trends.png", fastllm, (
                ChartSpec("TPOT", "tpot_ms", "ms"),
                ChartSpec("OUTPUT THROUGHPUT", "output_tok_s", "tok/s"),
                ChartSpec("KERNEL TIME", "kernel_ms_per_step", "ms/step"),
                ChartSpec("GPU IDLE RATIO", "gpu_idle_pct", "%"),
            ), "FASTLLM DECODE NSYS LAYOUT TRENDS", backend_field="series"))
        if any(row.get("submit_ms_per_step") is not None for row in fastllm):
            paths.append(write_dashboard(
                image_dir / "decode-nsys-layout-submission-trends.png",
                fastllm, (
                    ChartSpec("KERNEL COUNT", "kernel_count_per_step", "count/step"),
                    ChartSpec("KERNEL SUBMISSIONS", "submit_count_per_step", "count/step"),
                    ChartSpec("SUBMIT API TIME", "submit_ms_per_step", "ms/step"),
                    ChartSpec("GPU IDLE RATIO", "gpu_idle_pct", "%"),
                ), "FASTLLM DECODE NSYS LAYOUT SUBMISSION",
                backend_field="series"))
    return paths


def make_report(kind, result_dir, cases, records):
    stem = f"{kind}-attention-layout-matrix"
    headers = list(records[0]) if records else [
        "status", "layout", "requested_attention_backend", "backend", "batch"]
    chart_paths = create_charts(kind, result_dir, records)
    passed = all(case["status"] == "PASS" for case in cases)
    lines = [
        f"# {kind} Attention布局矩阵", "",
        "| 状态 | KV Cache布局 | Attention Backend | Strict | 结果目录 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        relative = Path(case["report_directory"]).relative_to(result_dir)
        lines.append(
            f"| {case['status']} | {case['layout']} | "
            f"{case['attention_backend']} | {case['strict']} | `{relative}` |")
    lines.extend(markdown_images(chart_paths, "跨布局Batch趋势图"))
    (result_dir / f"{stem}.md").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    write_csv(result_dir / f"{stem}.csv", records, headers)
    (result_dir / f"{stem}.json").write_text(json.dumps({
        "summary": {"result": "PASS" if passed else "FAIL",
                    "case_count": len(cases), "record_count": len(records)},
        "cases": cases,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_xlsx(result_dir / f"{stem}.xlsx", [
        ("矩阵状态", ["状态", "KV Cache布局", "Attention Backend", "Strict", "目录"], [
            [case["status"], case["layout"], case["attention_backend"],
             case["strict"], case["report_directory"]] for case in cases]),
        ("合并明细", headers,
         [[record.get(header) for header in headers] for record in records]),
    ], image_sheets=[("跨布局趋势图", chart_paths)])
    print(f"Summary: {'PASS' if passed else 'FAIL'}")
    print(f"Matrix report: {result_dir / (stem + '.xlsx')}")


def main():
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    cases, records = load_matrix(args.kind, result_dir)
    make_report(args.kind, result_dir, cases, records)


if __name__ == "__main__":
    main()
