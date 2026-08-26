#!/usr/bin/env python3
"""Compare paired target-only and DFlash2 HLE JSONL results."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def load_records(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            case_id = str(record.get("case_id") or "")
            if not case_id:
                raise ValueError(f"Record without case_id in {path}")
            if case_id in records:
                raise ValueError(f"Duplicate case_id {case_id} in {path}")
            records[case_id] = record
    return records


def metrics(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    values = list(records.values())
    total = len(values)
    correct = sum(item.get("correct") is True for item in values)
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in values)
    request_seconds = sum(float(item.get("latency_ms") or 0.0) for item in values) / 1000.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "output_tokens": output_tokens,
        "request_seconds": request_seconds,
        "output_tokens_per_second": output_tokens / request_seconds if request_seconds else None,
        "errors": sum(bool(item.get("error")) for item in values),
    }


def compare(
    target: Dict[str, Dict[str, Any]], dflash: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], int]:
    target_ids = set(target)
    dflash_ids = set(dflash)
    if target_ids != dflash_ids:
        return {
            "error": "case_id sets differ",
            "target_only_missing": sorted(dflash_ids - target_ids),
            "dflash_missing": sorted(target_ids - dflash_ids),
        }, 2
    target_metrics = metrics(target)
    dflash_metrics = metrics(dflash)
    both_correct = target_only_correct = dflash_only_correct = both_wrong = 0
    identical_outputs = 0
    for case_id in sorted(target_ids):
        target_correct = target[case_id].get("correct") is True
        dflash_correct = dflash[case_id].get("correct") is True
        if target_correct and dflash_correct:
            both_correct += 1
        elif target_correct:
            target_only_correct += 1
        elif dflash_correct:
            dflash_only_correct += 1
        else:
            both_wrong += 1
        target_sha = target[case_id].get("response_sha256")
        dflash_sha = dflash[case_id].get("response_sha256")
        identical_outputs += bool(
            target_sha
            and dflash_sha
            and not target[case_id].get("error")
            and not dflash[case_id].get("error")
            and target_sha == dflash_sha
        )
    target_tps = target_metrics["output_tokens_per_second"]
    dflash_tps = dflash_metrics["output_tokens_per_second"]
    return {
        "target_only": target_metrics,
        "dflash2": dflash_metrics,
        "speedup": dflash_tps / target_tps if target_tps and dflash_tps else None,
        "paired": {
            "both_correct": both_correct,
            "target_only_correct": target_only_correct,
            "dflash2_only_correct": dflash_only_correct,
            "both_wrong": both_wrong,
            "identical_outputs": identical_outputs,
            "identical_output_rate": identical_outputs / len(target_ids) if target_ids else 0.0,
        },
    }, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_only", type=Path)
    parser.add_argument("dflash2", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result, status = compare(load_records(args.target_only), load_records(args.dflash2))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
