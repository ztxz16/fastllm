#!/usr/bin/env python3
"""将NVFP4算子性能日志汇总为Markdown和CSV表格。"""

import argparse
import csv
import re
import shlex
from pathlib import Path

from xlsx_report import write_xlsx


def parse_args():
    parser = argparse.ArgumentParser(description="汇总NVFP4算子性能日志")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def command_tokens(text):
    for line in text.splitlines():
        if line.startswith("COMMAND:"):
            try:
                return shlex.split(line.removeprefix("COMMAND:").strip())
            except ValueError:
                return []
    return []


def option(tokens, name, default="-"):
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError):
        return default


def parameters(tokens):
    result = {}
    for index, token in enumerate(tokens[:-1]):
        if token == "--param" and "=" in tokens[index + 1]:
            key, value = tokens[index + 1].split("=", 1)
            result[key] = value
    return result


def match(text, pattern, default="-"):
    found = re.search(pattern, text)
    return found.group(1) if found else default


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = command_tokens(text)
    params = parameters(tokens)
    op = option(tokens, "--op")
    if op == "linear_nvfp4":
        m, n, k = params.get("batch", "-"), params.get("out", "-"), params.get("in", "-")
    elif op == "nvfp4_swiglu_quant":
        m, n, k = params.get("rows", "-"), "-", params.get("hidden", "-")
    else:
        m, n, k = params.get("batch", "-"), params.get("inter", "-"), params.get("hidden", "-")
    strict = "yes" if any("STRICT=1" in token for token in tokens) else "no"
    return {
        "case": path.stem,
        "op": op,
        "M": m,
        "N/inter": n,
        "K/hidden": k,
        "topk": params.get("topk", "-"),
        "experts": params.get("experts", "-"),
        "dtype": params.get("input_type", "-"),
        "warmup": option(tokens, "--warmup"),
        "iters": option(tokens, "--iters"),
        "latency_ms": match(text, r"latency:\s+avg_ms=([0-9.eE+-]+)"),
        "io_speed": match(text, r"io_speed=([^,\n]+)"),
        "compute_speed": match(text, r"compute_speed=([^,\n]+)"),
        "strict": strict,
    }


def markdown(rows, columns):
    output = ["# NVFP4算子性能汇总", "", "| " + " | ".join(columns) + " |",
              "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(row[column]).strip() for column in columns) + " |")
    output.append("")
    return "\n".join(output)


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    rows = [parse_log(path) for path in sorted(log_dir.glob("op_*perf*.log"))]
    if not rows:
        raise RuntimeError(f"没有找到算子性能日志: {log_dir}/op_*perf*.log")
    columns = ["case", "op", "M", "N/inter", "K/hidden", "topk", "experts",
               "dtype", "warmup", "iters", "latency_ms", "io_speed",
               "compute_speed", "strict"]
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = prefix.with_suffix(".md")
    csv_path = prefix.with_suffix(".csv")
    md_text = markdown(rows, columns)
    md_path.write_text(md_text, encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    xlsx_path = prefix.with_suffix(".xlsx")
    write_xlsx(xlsx_path, [("算子性能", columns,
                            [[row[column] for column in columns] for row in rows])])
    print(md_text)
    print(f"Markdown: {md_path}")
    print(f"CSV: {csv_path}")
    print(f"Excel: {xlsx_path}")


if __name__ == "__main__":
    main()
