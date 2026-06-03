#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm


BENCHMARK_ID = "mmlu_pro"
LETTERS = "ABCDEFGHIJ"
DEFAULT_DATASET = "TIGER-Lab/MMLU-Pro"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that answers multiple-choice questions accurately."


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "model"


def parse_json_object(value: str, name: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name} must be a JSON object.")
    return parsed


def get_option_labels(options: Sequence[str]) -> str:
    return LETTERS[: len(options)]


def parse_options(value: Any, row: Dict[str, Any]) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
        parts = re.split(r"\n(?=[A-J][\).:]\s)", stripped)
        if len(parts) > 1:
            return [re.sub(r"^[A-J][\).:]\s*", "", part).strip() for part in parts]
    option_keys = [
        f"option_{letter.lower()}" for letter in LETTERS
    ] + [
        f"option_{letter}" for letter in LETTERS
    ] + [
        letter for letter in LETTERS
    ]
    options = []
    for key in option_keys:
        if key in row and str(row[key]).strip():
            options.append(str(row[key]).strip())
    return options


def normalize_answer(row: Dict[str, Any], labels: str) -> str:
    answer = str(row.get("answer", "")).strip().upper()
    if answer and answer in labels:
        return answer
    if answer:
        match = re.search(rf"(?<![A-Za-z])([{re.escape(labels)}])(?![A-Za-z])", answer)
        if match:
            return match.group(1).upper()
    answer_index = row.get("answer_index")
    if answer_index is None or answer_index == "":
        answer_index = row.get("answer_idx")
    if answer_index is not None and answer_index != "":
        try:
            idx = int(answer_index)
        except (TypeError, ValueError):
            idx = -1
        if 0 <= idx < len(labels):
            return labels[idx]
    raise ValueError(f"Cannot normalize answer from row: {row}")


def normalize_example(row: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    question = str(row.get("question", "")).strip()
    options = parse_options(row.get("options"), row)
    if not question:
        raise ValueError(f"Example {idx} has no question.")
    if len(options) < 2 or len(options) > len(LETTERS):
        raise ValueError(f"Example {idx} has invalid options length: {len(options)}")

    labels = get_option_labels(options)
    answer = normalize_answer(row, labels)
    question_id = row.get("question_id", row.get("id", idx))
    case_id = f"{split}:{question_id}"
    return {
        "benchmark_id": BENCHMARK_ID,
        "case_id": case_id,
        "question_id": question_id,
        "split": split,
        "category": str(row.get("category", row.get("subject", "unknown"))).strip() or "unknown",
        "src": str(row.get("src", "")).strip(),
        "question": question,
        "options": options,
        "answer": answer,
        "answer_index": labels.index(answer),
        "labels": labels,
    }


def load_local_rows(data_file: Path) -> Iterable[Dict[str, Any]]:
    suffix = data_file.suffix.lower()
    if suffix == ".jsonl":
        with data_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if suffix == ".json":
        with data_file.open("r", encoding="utf-8") as fin:
            parsed = json.load(fin)
        if isinstance(parsed, dict):
            parsed = parsed.get("data", parsed.get("examples", []))
        if not isinstance(parsed, list):
            raise ValueError("JSON data file must be a list, or contain a data/examples list.")
        for row in parsed:
            yield row
        return
    if suffix == ".csv":
        with data_file.open("r", encoding="utf-8", newline="") as fin:
            for row in csv.DictReader(fin):
                yield row
        return
    raise ValueError("Unsupported local data file. Use .jsonl, .json, or .csv.")


def load_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: Iterable[Dict[str, Any]]
    if args.data_file:
        data_file = Path(args.data_file)
        rows = load_local_rows(data_file)
        split = args.split
    else:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: datasets. Install with "
                "`pip install -r test/mmlu_pro/requirements.txt`, or pass --data-file."
            ) from exc
        load_kwargs: Dict[str, Any] = {"split": args.split}
        if args.dataset_revision:
            load_kwargs["revision"] = args.dataset_revision
        rows = load_dataset(args.dataset_name, **load_kwargs)
        split = args.split

    examples = []
    for idx, row in enumerate(rows):
        try:
            examples.append(normalize_example(dict(row), idx, split))
        except ValueError as exc:
            if args.skip_bad_examples:
                print(f"Skip bad example {idx}: {exc}")
                continue
            raise
    return examples


def filter_examples(examples: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = examples
    if args.category:
        wanted = {category.lower() for category in args.category}
        selected = [item for item in selected if item["category"].lower() in wanted]

    if args.shuffle:
        rng = random.Random(args.seed)
        selected = list(selected)
        rng.shuffle(selected)

    if args.sample_per_category is not None:
        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in selected:
            by_category[item["category"]].append(item)
        sampled: List[Dict[str, Any]] = []
        for category in sorted(by_category):
            sampled.extend(by_category[category][: args.sample_per_category])
        selected = sampled

    if args.start:
        selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def build_prompt(example: Dict[str, Any], cot: bool) -> str:
    option_lines = "\n".join(
        f"{label}. {option}" for label, option in zip(example["labels"], example["options"])
    )
    answer_format = f"one of {', '.join(example['labels'])}"
    if cot:
        instruction = (
            "Think through the problem, then finish with exactly this format on the final line: "
            "The answer is (X)."
        )
    else:
        instruction = (
            "Choose the single best answer. Do not explain. "
            "Return only the answer letter, for example: A."
        )
    return (
        "The following is a multiple-choice question from MMLU-Pro.\n\n"
        f"Question:\n{example['question']}\n\n"
        f"Choices:\n{option_lines}\n\n"
        f"The answer must be {answer_format}.\n"
        f"{instruction}"
    )


def build_payload(args: argparse.Namespace, prompt: str, extra_body: Dict[str, Any]) -> Dict[str, Any]:
    messages = []
    if not args.no_system_prompt and args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "stream": False,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    payload.update(extra_body)
    return payload


def extract_answer(text: str, labels: str) -> Optional[str]:
    if not text:
        return None
    escaped = re.escape(labels)
    patterns = [
        rf"\\boxed\s*{{\s*([{escaped}])\s*}}",
        rf"(?:final\s+answer|answer|correct\s+answer)\s*(?:is|:)?\s*[\(\[\{{]?\s*([{escaped}])\s*[\)\]\}}]?",
        rf"^\s*[\(\[\{{]?\s*([{escaped}])\s*[\)\]\}}\.:\s]*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    candidates = re.findall(rf"(?<![A-Za-z])([{escaped}])(?![A-Za-z])", text)
    if candidates:
        return candidates[-1].upper()
    return None


def post_chat_completion(
    args: argparse.Namespace,
    example: Dict[str, Any],
    prompt: str,
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    url = f"{normalize_base_url(args.base_url)}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    payload = build_payload(args, prompt, extra_body)

    last_error = ""
    with requests.Session() as session:
        session.trust_env = args.use_env_proxy
        for attempt in range(args.max_retries + 1):
            start = time.perf_counter()
            try:
                response = session.post(url, headers=headers, json=payload, timeout=args.request_timeout)
                latency_ms = (time.perf_counter() - start) * 1000.0
                if response.status_code in (429, 500, 502, 503, 504) and attempt < args.max_retries:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    time.sleep(args.retry_backoff * (2 ** attempt))
                    continue
                if response.status_code != 200:
                    return {
                        "raw_output": "",
                        "usage": {},
                        "latency_ms": latency_ms,
                        "error": f"HTTP {response.status_code}: {response.text[:1000]}",
                    }
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return {
                        "raw_output": "",
                        "usage": data.get("usage", {}),
                        "latency_ms": latency_ms,
                        "error": "response has no choices",
                    }
                message = choices[0].get("message", {})
                return {
                    "raw_output": str(message.get("content", "")),
                    "usage": data.get("usage", {}),
                    "latency_ms": latency_ms,
                    "finish_reason": choices[0].get("finish_reason"),
                    "response_model": data.get("model"),
                    "error": None,
                }
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                if attempt < args.max_retries:
                    time.sleep(args.retry_backoff * (2 ** attempt))
                    continue
                return {
                    "raw_output": "",
                    "usage": {},
                    "latency_ms": None,
                    "error": last_error,
                }

    return {
        "raw_output": "",
        "usage": {},
        "latency_ms": None,
        "error": last_error or "unknown error",
    }


def evaluate_one(
    args: argparse.Namespace,
    example: Dict[str, Any],
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = build_prompt(example, args.cot)
    api_result = post_chat_completion(args, example, prompt, extra_body)
    prediction = extract_answer(api_result.get("raw_output", ""), example["labels"])
    correct = prediction == example["answer"] if prediction else False
    usage = api_result.get("usage") or {}
    return {
        "benchmark_id": BENCHMARK_ID,
        "case_id": example["case_id"],
        "question_id": example["question_id"],
        "split": example["split"],
        "category": example["category"],
        "src": example["src"],
        "model": args.model,
        "prompt": prompt,
        "question": example["question"],
        "options": example["options"],
        "answer": example["answer"],
        "answer_index": example["answer_index"],
        "prediction": prediction,
        "correct": correct,
        "raw_output": api_result.get("raw_output", ""),
        "latency_ms": api_result.get("latency_ms"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": api_result.get("finish_reason"),
        "response_model": api_result.get("response_model"),
        "error": api_result.get("error"),
        "metadata": {
            "cot": args.cot,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "labels": example["labels"],
        },
    }


def read_existing_results(output_file: Path) -> Tuple[set, List[Dict[str, Any]]]:
    if not output_file.exists():
        return set(), []
    completed = set()
    records = []
    with output_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            case_id = record.get("case_id")
            if case_id:
                completed.add(case_id)
            records.append(record)
    return completed, records


def write_jsonl_record(fout, record: Dict[str, Any]) -> None:
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    fout.flush()


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, Dict[str, Any]] = {}
    total = len(records)
    correct = sum(1 for item in records if item.get("correct") is True)
    answered = sum(1 for item in records if item.get("prediction"))
    errors = sum(1 for item in records if item.get("error"))
    latency_values = [
        float(item["latency_ms"])
        for item in records
        if item.get("latency_ms") is not None
    ]
    for item in records:
        category = item.get("category") or "unknown"
        stats = by_category.setdefault(
            category,
            {"total": 0, "correct": 0, "answered": 0, "errors": 0, "accuracy": 0.0},
        )
        stats["total"] += 1
        stats["correct"] += 1 if item.get("correct") is True else 0
        stats["answered"] += 1 if item.get("prediction") else 0
        stats["errors"] += 1 if item.get("error") else 0
    for stats in by_category.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

    return {
        "benchmark_id": BENCHMARK_ID,
        "total": total,
        "answered": answered,
        "invalid": total - answered,
        "errors": errors,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "valid_accuracy": correct / answered if answered else 0.0,
        "latency_ms_avg": sum(latency_values) / len(latency_values) if latency_values else None,
        "latency_ms_min": min(latency_values) if latency_values else None,
        "latency_ms_max": max(latency_values) if latency_values else None,
        "by_category": dict(sorted(by_category.items())),
    }


def resolve_output_file(args: argparse.Namespace) -> Path:
    if args.output_file:
        return Path(args.output_file)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = sanitize_filename(args.model)
    return Path(args.output_dir) / f"mmlu_pro_{model_name}_{args.split}_{timestamp}.jsonl"


def run(args: argparse.Namespace) -> int:
    extra_body = parse_json_object(args.extra_body, "--extra-body")
    examples = filter_examples(load_examples(args), args)
    if not examples:
        print("No examples selected.")
        return 1

    output_file = resolve_output_file(args)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file = output_file.with_suffix(".summary.json")
    if output_file.exists() and not args.resume and not args.overwrite:
        raise SystemExit(f"Output file already exists: {output_file}. Pass --resume or --overwrite.")

    completed, existing_records = (set(), [])
    if args.resume and not args.overwrite:
        completed, existing_records = read_existing_results(output_file)
    examples_to_run = [item for item in examples if item["case_id"] not in completed]

    mode = "w" if args.overwrite or not output_file.exists() else "a"
    print(f"Dataset: {args.data_file or args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Selected examples: {len(examples)}")
    print(f"Already completed: {len(completed)}")
    print(f"Running examples: {len(examples_to_run)}")
    print(f"Output: {output_file}")

    new_records: List[Dict[str, Any]] = []
    with output_file.open(mode, encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(evaluate_one, args, example, extra_body): example
                for example in examples_to_run
            }
            progress = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="MMLU-Pro",
            )
            for future in progress:
                record = future.result()
                write_jsonl_record(fout, record)
                new_records.append(record)
                done = len(existing_records) + len(new_records)
                acc = sum(1 for item in existing_records + new_records if item.get("correct")) / done
                progress.set_postfix(acc=f"{acc:.4f}")

    all_records = existing_records + new_records
    if not all_records:
        _, all_records = read_existing_results(output_file)
    summary = summarize(all_records)
    with summary_file.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_file}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI-compatible chat/completions API on MMLU-Pro."
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "no-key"))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-file", default="", help="Optional local .jsonl/.json/.csv data file.")
    parser.add_argument("--category", action="append", help="Category filter. Can be passed multiple times.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-per-category", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--use-env-proxy", action="store_true", help="Use HTTP(S)_PROXY from the environment.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cot", action="store_true", help="Ask the model to reason and end with 'The answer is (X).'.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument(
        "--extra-body",
        default="",
        help="JSON object merged into every chat/completions payload, e.g. "
        "'{\"chat_template_kwargs\":{\"enable_thinking\":false}}'.",
    )
    parser.add_argument("--output-dir", default="test/mmlu_pro/results")
    parser.add_argument("--output-file", default="")
    parser.add_argument("--resume", action="store_true", help="Skip case_id values already present in --output-file.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-bad-examples", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.sample_per_category is not None and args.sample_per_category < 1:
        parser.error("--sample-per-category must be >= 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
