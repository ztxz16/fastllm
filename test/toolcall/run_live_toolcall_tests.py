#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from lib.live_openai import (
    load_live_cases,
    request_tool_names,
    result_summary_line,
    run_live_openai_case,
    verbose_lines,
)


def _case_matches(case: Dict[str, Any], case_id: Optional[str]) -> bool:
    return case_id is None or case.get("id") == case_id


def _selected_cases(case_id: Optional[str]) -> List[Dict[str, Any]]:
    return [
        case for case in load_live_cases()
        if case.get("status") == "manual" and _case_matches(case, case_id)
    ]


def _list_cases(cases: Iterable[Dict[str, Any]]):
    for case in cases:
        tags = ",".join(case.get("tags") or [])
        print(
            f"{case.get('id')}\tmode={case.get('mode')}\t"
            f"status={case.get('status')}\ttags={tags}"
        )


def _dump_result(
    dump_dir: Path,
    label: str,
    case: Dict[str, Any],
    result: Dict[str, Any],
):
    dump_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case.get("id"),
        "mode": case.get("mode"),
        "status": case.get("status"),
        "tags": case.get("tags", []),
        "expected": case.get("expected", {}),
        "result": result,
    }
    path = dump_dir / f"{label}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _update_variant_stats(
    variants_by_case: Dict[str, Counter],
    result: Dict[str, Any],
):
    case_id = result.get("case_id")
    names = []
    if result.get("passed"):
        names = result.get("tool_names") or []
    else:
        diagnostics = result.get("diagnostics") or {}
        names = diagnostics.get("tool_names") or []
        invalid = diagnostics.get("invalid_tool_name")
        if invalid and invalid not in names:
            names.append(invalid)
    for name in names:
        variants_by_case[case_id][name] += 1


def _print_summary(
    results: List[Dict[str, Any]],
    cases_by_id: Dict[str, Dict[str, Any]],
    variants_by_case: Dict[str, Counter],
):
    total = len(results)
    passed = sum(1 for result in results if result.get("passed"))
    failed = total - passed
    print(f"Summary: total={total} passed={passed} failed={failed} skipped=0")

    failures = Counter(
        result.get("error_code", "unknown_failure")
        for result in results if not result.get("passed")
    )
    if failures:
        print("failure_by_error_code:")
        for code, count in sorted(failures.items()):
            print(f"  {code}: {count}")

    if variants_by_case:
        print("tool_name_variants_by_case:")
        for case_id in sorted(variants_by_case):
            variants = ", ".join(
                f"{name}: {count}"
                for name, count in sorted(variants_by_case[case_id].items())
            )
            print(f"  {case_id}: {variants}")

    quality = defaultdict(Counter)
    for result in results:
        case = cases_by_id.get(result.get("case_id"), {})
        dimension = case.get("expected", {}).get(
            "quality_dimension", "unspecified")
        quality[dimension]["passed" if result.get("passed") else "failed"] += 1
    if quality:
        print("quality_by_dimension:")
        for dimension in sorted(quality):
            items = ", ".join(
                f"{key}: {value}" for key, value in sorted(quality[dimension].items())
            )
            print(f"  {dimension}: {items}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual OpenAI-compatible live toolcall smoke tests.")
    parser.add_argument("--base-url",
                        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    parser.add_argument("--model", help="Model name to send in requests")
    parser.add_argument("--case-id", help="Run only one case id")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override request.temperature")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="Return exit code 0 even if cases fail")
    parser.add_argument("--dump-dir",
                        help="Write per-case JSON diagnostics to this directory")
    args = parser.parse_args()

    cases = _selected_cases(args.case_id)
    if args.list_cases:
        _list_cases(cases)
        return 0

    if not cases:
        print(f"No live cases matched case_id={args.case_id!r}")
        return 1
    if not args.base_url or not args.model:
        print("--base-url and --model are required unless --list-cases is used")
        return 1
    if args.repeat < 1:
        print("--repeat must be >= 1")
        return 1

    results: List[Dict[str, Any]] = []
    variants_by_case: Dict[str, Counter] = defaultdict(Counter)
    cases_by_id = {case["id"]: case for case in cases}
    dump_dir = Path(args.dump_dir) if args.dump_dir else None

    for case in cases:
        for repeat_index in range(args.repeat):
            result = run_live_openai_case(
                case,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                temperature_override=args.temperature,
            )
            label = case["id"]
            if args.repeat > 1:
                label = f"{label}#{repeat_index + 1}/{args.repeat}"
            print(result_summary_line({**result, "case_id": label}))
            if args.verbose:
                for line in verbose_lines(case, result):
                    print(line)
            if dump_dir:
                safe_label = label.replace("/", "_").replace("#", "_")
                _dump_result(dump_dir, safe_label, case, result)
            _update_variant_stats(variants_by_case, result)
            results.append(result)

    _print_summary(results, cases_by_id, variants_by_case)
    if args.report_only:
        return 0
    return 0 if all(result.get("passed") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
