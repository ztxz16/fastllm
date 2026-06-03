#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import json
import random
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm


BENCHMARK_ID = "gsm8k"
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that solves math problems accurately."


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


def normalize_number_text(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", "", value)
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = value.rstrip(".")
    return value


def decimal_from_text(value: str) -> Optional[Decimal]:
    value = normalize_number_text(value)
    if not value:
        return None
    frac_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", value)
    if frac_match:
        try:
            denominator = Decimal(frac_match.group(2))
            if denominator == 0:
                return None
            return Decimal(frac_match.group(1)) / denominator
        except InvalidOperation:
            return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def extract_gold_answer(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer)
    if match:
        return normalize_number_text(match.group(1))
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?", answer)
    if numbers:
        return normalize_number_text(numbers[-1])
    raise ValueError(f"Cannot extract numeric answer: {answer}")


def extract_prediction(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"\\boxed\s*\{\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)\s*\}",
        r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)",
        r"(?:the\s+answer\s+is|answer\s*:|final\s+answer\s*:?)\s*([-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_number_text(match.group(1))

    numbers = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?", text)
    if numbers:
        return normalize_number_text(numbers[-1])
    return None


def answers_equal(prediction: Optional[str], answer: str) -> bool:
    if prediction is None:
        return False
    pred_dec = decimal_from_text(prediction)
    ans_dec = decimal_from_text(answer)
    if pred_dec is not None and ans_dec is not None:
        return pred_dec == ans_dec
    return normalize_number_text(prediction) == normalize_number_text(answer)


def normalize_example(row: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    question = str(row.get("question", "")).strip()
    answer_raw = str(row.get("answer", "")).strip()
    if not question:
        raise ValueError(f"Example {idx} has no question.")
    if not answer_raw:
        raise ValueError(f"Example {idx} has no answer.")
    answer = extract_gold_answer(answer_raw)
    question_id = row.get("question_id", row.get("id", idx))
    return {
        "benchmark_id": BENCHMARK_ID,
        "case_id": f"{split}:{question_id}",
        "question_id": question_id,
        "split": split,
        "question": question,
        "answer": answer,
        "answer_raw": answer_raw,
    }


def load_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.data_file:
        rows = load_local_rows(Path(args.data_file))
    else:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: datasets. Install with "
                "`pip install -r test/gsm8k/requirements.txt`, or pass --data-file."
            ) from exc
        load_kwargs: Dict[str, Any] = {"split": args.split}
        if args.dataset_revision:
            load_kwargs["revision"] = args.dataset_revision
        rows = load_dataset(args.dataset_name, args.dataset_config, **load_kwargs)

    examples = []
    for idx, row in enumerate(rows):
        examples.append(normalize_example(dict(row), idx, args.split))
    return examples


def filter_examples(examples: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = examples
    if args.shuffle:
        rng = random.Random(args.seed)
        selected = list(selected)
        rng.shuffle(selected)
    if args.start:
        selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def build_prompt(example: Dict[str, Any], cot: bool) -> str:
    if cot:
        instruction = (
            "Solve the problem step by step. Keep the reasoning concise. "
            "Finish with exactly one final line in this format: The answer is <number>."
        )
    else:
        instruction = "Return only the final numeric answer. Do not explain."
    return (
        "The following is a grade-school math word problem from GSM8K.\n\n"
        f"Problem:\n{example['question']}\n\n"
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


def post_chat_completion(args: argparse.Namespace, prompt: str, extra_body: Dict[str, Any]) -> Dict[str, Any]:
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


def evaluate_one(args: argparse.Namespace, example: Dict[str, Any], extra_body: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt(example, args.cot)
    api_result = post_chat_completion(args, prompt, extra_body)
    prediction = extract_prediction(api_result.get("raw_output", ""))
    correct = answers_equal(prediction, example["answer"])
    usage = api_result.get("usage") or {}
    return {
        "benchmark_id": BENCHMARK_ID,
        "case_id": example["case_id"],
        "question_id": example["question_id"],
        "split": example["split"],
        "model": args.model,
        "prompt": prompt,
        "question": example["question"],
        "answer": example["answer"],
        "answer_raw": example["answer_raw"],
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


def percentile(values: Sequence[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * percent / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def summarize(records: Sequence[Dict[str, Any]], runtime_seconds: Optional[float] = None) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for item in records if item.get("correct") is True)
    answered = sum(1 for item in records if item.get("prediction") is not None)
    errors = sum(1 for item in records if item.get("error"))
    latency_values = [
        float(item["latency_ms"])
        for item in records
        if item.get("latency_ms") is not None
    ]
    prompt_tokens = sum(int(item.get("input_tokens") or 0) for item in records)
    completion_tokens = sum(int(item.get("output_tokens") or 0) for item in records)
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
        "latency_ms_p50": percentile(latency_values, 50),
        "latency_ms_p90": percentile(latency_values, 90),
        "latency_ms_p95": percentile(latency_values, 95),
        "latency_ms_min": min(latency_values) if latency_values else None,
        "latency_ms_max": max(latency_values) if latency_values else None,
        "input_tokens_total": prompt_tokens,
        "output_tokens_total": completion_tokens,
        "runtime_seconds": runtime_seconds,
        "throughput_items_per_second": total / runtime_seconds if runtime_seconds else None,
        "input_tokens_per_second": prompt_tokens / runtime_seconds if runtime_seconds else None,
        "output_tokens_per_second": completion_tokens / runtime_seconds if runtime_seconds else None,
    }


def resolve_output_file(args: argparse.Namespace) -> Path:
    if args.output_file:
        return Path(args.output_file)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = sanitize_filename(args.model)
    mode = "cot" if args.cot else "direct"
    return Path(args.output_dir) / f"gsm8k_{model_name}_{args.split}_{mode}_{timestamp}.jsonl"


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

    start = time.perf_counter()
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
                desc="GSM8K",
            )
            for future in progress:
                record = future.result()
                write_jsonl_record(fout, record)
                new_records.append(record)
                done = len(existing_records) + len(new_records)
                acc = sum(1 for item in existing_records + new_records if item.get("correct")) / done
                progress.set_postfix(acc=f"{acc:.4f}")

    runtime_seconds = time.perf_counter() - start
    all_records = existing_records + new_records
    if not all_records:
        _, all_records = read_existing_results(output_file)
        runtime_seconds = None
    summary = summarize(all_records, runtime_seconds)
    with summary_file.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K reasoning with an OpenAI-compatible API.")
    parser.add_argument("--base-url", default="http://localhost:8080", help="API base URL without /v1.")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="ds")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-file", default="")
    parser.add_argument("--output-dir", default="test/gsm8k/results")
    parser.add_argument("--output-file", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--cot", action="store_true", help="Ask the model to solve step by step.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--extra-body", default="")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--use-env-proxy", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
