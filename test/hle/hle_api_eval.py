#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible endpoint on a reproducible HLE subset."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm


BENCHMARK_ID = "hle"
DEFAULT_DATASET = "cais/hle"
DEFAULT_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


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


def canonical_answer_type(value: Any) -> str:
    compact = re.sub(r"[^a-z]", "", str(value or "").lower())
    if compact in {"multiplechoice", "mc", "mcq"}:
        return "multipleChoice"
    if compact in {"exactmatch", "shortanswer", "exact"}:
        return "exactMatch"
    return str(value or "unknown").strip() or "unknown"


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
            raise ValueError("JSON data file must be a list, or contain data/examples.")
        yield from parsed
        return
    if suffix == ".csv":
        with data_file.open("r", encoding="utf-8", newline="") as fin:
            yield from csv.DictReader(fin)
        return
    raise ValueError("Unsupported local data file. Use .jsonl, .json, or .csv.")


def normalize_example(row: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not question:
        raise ValueError(f"Example {idx} has no question.")
    if not answer:
        raise ValueError(f"Example {idx} has no answer.")
    question_id = str(row.get("id", row.get("question_id", idx)))
    image = row.get("image") or ""
    if not isinstance(image, str):
        raise ValueError(
            f"Example {idx} image must be an URL/data-URL string; got {type(image).__name__}."
        )
    return {
        "benchmark_id": BENCHMARK_ID,
        "case_id": f"{split}:{question_id}",
        "question_id": question_id,
        "split": split,
        "question": question,
        "image": image,
        "answer": answer,
        "answer_type": canonical_answer_type(row.get("answer_type")),
        "category": str(row.get("category", "unknown")).strip() or "unknown",
        "raw_subject": str(row.get("raw_subject", "")).strip(),
    }


def load_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.data_file:
        rows: Iterable[Dict[str, Any]] = load_local_rows(Path(args.data_file))
    else:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Missing dependency: datasets. Run test/hle/setup.sh, or pass --data-file."
            ) from exc
        kwargs: Dict[str, Any] = {"split": args.split}
        if args.dataset_revision:
            kwargs["revision"] = args.dataset_revision
        rows = load_dataset(args.dataset_name, **kwargs)

    examples = []
    for idx, row in enumerate(rows):
        try:
            examples.append(normalize_example(dict(row), idx, args.split))
        except ValueError as exc:
            if args.skip_bad_examples:
                print(f"Skip bad example {idx}: {exc}")
                continue
            raise
    return examples


def filter_examples(
    examples: List[Dict[str, Any]], args: argparse.Namespace
) -> List[Dict[str, Any]]:
    selected = list(examples)
    if args.text_only:
        selected = [item for item in selected if not item["image"]]
    if args.answer_type:
        wanted_types = {canonical_answer_type(value) for value in args.answer_type}
        selected = [item for item in selected if item["answer_type"] in wanted_types]
    if args.category:
        wanted_categories = {value.casefold() for value in args.category}
        selected = [
            item for item in selected if item["category"].casefold() in wanted_categories
        ]
    if args.shuffle:
        random.Random(args.seed).shuffle(selected)
    if args.start:
        selected = selected[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def selection_sha256(examples: Sequence[Dict[str, Any]]) -> str:
    payload = "\n".join(item["case_id"] for item in examples).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_messages(
    example: Dict[str, Any], system_prompt: str, no_system_prompt: bool
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if not no_system_prompt and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if example["image"]:
        content: Any = [
            {"type": "text", "text": example["question"]},
            {"type": "image_url", "image_url": {"url": example["image"]}},
        ]
    else:
        content = example["question"]
    messages.append({"role": "user", "content": content})
    return messages


def build_payload(
    args: argparse.Namespace,
    example: Dict[str, Any],
    extra_body: Dict[str, Any],
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    effective_max_tokens = args.max_tokens if max_tokens is None else max_tokens
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": build_messages(example, args.system_prompt, args.no_system_prompt),
        "stream": False,
        "temperature": args.temperature,
    }
    if effective_max_tokens > 0:
        payload["max_tokens"] = effective_max_tokens
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    payload.update(extra_body)
    return payload


def post_chat_completion(
    args: argparse.Namespace,
    example: Dict[str, Any],
    extra_body: Dict[str, Any],
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    url = f"{normalize_base_url(args.base_url)}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    payload = build_payload(args, example, extra_body, max_tokens=max_tokens)
    last_error = ""
    with requests.Session() as session:
        session.trust_env = args.use_env_proxy
        for attempt in range(args.max_retries + 1):
            started = time.perf_counter()
            try:
                response = session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=args.request_timeout if args.request_timeout > 0 else None,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                if (
                    response.status_code in (429, 500, 502, 503, 504)
                    and attempt < args.max_retries
                ):
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    time.sleep(args.retry_backoff * (2**attempt))
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
                    "raw_output": str(message.get("content") or ""),
                    "reasoning_content": str(message.get("reasoning_content") or ""),
                    "usage": data.get("usage", {}),
                    "latency_ms": latency_ms,
                    "finish_reason": choices[0].get("finish_reason"),
                    "response_model": data.get("model"),
                    "error": None,
                }
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                if attempt < args.max_retries:
                    time.sleep(args.retry_backoff * (2**attempt))
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


def answer_region(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    last_close = lowered.rfind("</think>")
    if last_close >= 0:
        text = text[last_close + len("</think>") :]
    elif lowered.rfind("<think>") >= 0:
        return None

    matches = re.findall(
        r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.*?)\s*$", text
    )
    for candidate in reversed(matches):
        if candidate.strip():
            return candidate.strip()
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text, flags=re.IGNORECASE)
    if boxed:
        return boxed[-1].strip()
    return None


def normalize_choice(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    compact = re.sub(r"[*_`]", "", value).strip()
    match = re.match(
        r"(?:the\s+)?(?:answer\s+is\s+)?(?:option|choice)?\s*"
        r"[\(\[\{]?\s*([A-Z])(?:\s*[\)\]\}]|\s*[\.!,:;-]|\s*$)",
        compact,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def normalize_exact(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(". ")
    return normalized or None


def score_answer(
    raw_output: str, answer: str, answer_type: str
) -> Tuple[Optional[str], bool, str]:
    prediction = answer_region(raw_output)
    if answer_type == "multipleChoice":
        pred_choice = normalize_choice(prediction)
        gold_choice = normalize_choice(answer)
        return pred_choice, pred_choice is not None and pred_choice == gold_choice, "choice_exact"
    pred_exact = normalize_exact(prediction)
    gold_exact = normalize_exact(answer)
    return pred_exact, pred_exact is not None and pred_exact == gold_exact, "normalized_exact"


def extract_confidence(raw_output: str) -> Optional[float]:
    matches = re.findall(
        r"(?im)^\s*confidence\s*:\s*(\d+(?:\.\d+)?)\s*%?\s*$", raw_output
    )
    if not matches:
        return None
    value = float(matches[-1])
    return value if 0.0 <= value <= 100.0 else None


def evaluate_one(
    args: argparse.Namespace,
    example: Dict[str, Any],
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    api_result = post_chat_completion(args, example, extra_body)
    raw_output = api_result.get("raw_output", "")
    prediction, correct, score_method = score_answer(
        raw_output, example["answer"], example["answer_type"]
    )
    usage = api_result.get("usage") or {}
    response_sha = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    return {
        "benchmark_id": BENCHMARK_ID,
        "case_id": example["case_id"],
        "question_id": example["question_id"],
        "split": example["split"],
        "category": example["category"],
        "raw_subject": example["raw_subject"],
        "answer_type": example["answer_type"],
        "model": args.model,
        "question": example["question"],
        "answer": example["answer"],
        "prediction": prediction,
        "correct": correct,
        "score_method": score_method,
        "confidence": extract_confidence(raw_output),
        "raw_output": raw_output,
        "response_sha256": response_sha,
        "latency_ms": api_result.get("latency_ms"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": api_result.get("finish_reason"),
        "response_model": api_result.get("response_model"),
        "error": api_result.get("error"),
        "metadata": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens or None,
            "has_image": bool(example["image"]),
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
            records.append(record)
            if record.get("case_id"):
                completed.add(record["case_id"])
    return completed, records


def percentile(values: Sequence[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize_group(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    correct = sum(item.get("correct") is True for item in records)
    answered = sum(item.get("prediction") is not None for item in records)
    errors = sum(bool(item.get("error")) for item in records)
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in records)
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in records)
    latencies = [
        float(item["latency_ms"])
        for item in records
        if item.get("latency_ms") is not None
    ]
    request_seconds = sum(latencies) / 1000.0
    return {
        "total": total,
        "answered": answered,
        "invalid": total - answered,
        "errors": errors,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "request_seconds_total": request_seconds,
        "output_tokens_per_second": (
            output_tokens / request_seconds if request_seconds else None
        ),
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else None,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p90": percentile(latencies, 90),
        "latency_ms_max": max(latencies) if latencies else None,
    }


def summarize(
    records: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    run_wall_seconds: Optional[float],
) -> Dict[str, Any]:
    by_answer_type: Dict[str, Any] = {}
    by_category: Dict[str, Any] = {}
    for answer_type in sorted({item["answer_type"] for item in records}):
        by_answer_type[answer_type] = summarize_group(
            [item for item in records if item["answer_type"] == answer_type]
        )
    for category in sorted({item["category"] for item in records}):
        by_category[category] = summarize_group(
            [item for item in records if item["category"] == category]
        )
    return {
        "benchmark_id": BENCHMARK_ID,
        "dataset": args.data_file or args.dataset_name,
        "dataset_revision": args.dataset_revision or None,
        "selection_sha256": selection_sha256(selected),
        "scoring_note": (
            "multipleChoice uses deterministic answer-letter matching; exactMatch is only "
            "a normalized-string lower bound and is not the official HLE judge score"
        ),
        "parameters": {
            "workers": args.workers,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens or None,
            "warmup_tokens": args.warmup_tokens,
            "text_only": args.text_only,
            "answer_type": args.answer_type,
            "shuffle": args.shuffle,
            "seed": args.seed,
        },
        **summarize_group(records),
        "run_wall_seconds": run_wall_seconds,
        "by_answer_type": by_answer_type,
        "by_category": by_category,
    }


def resolve_output_file(args: argparse.Namespace) -> Path:
    if args.output_file:
        return Path(args.output_file)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = sanitize_filename(args.model)
    return Path(args.output_dir) / f"hle_{model_name}_{args.split}_{timestamp}.jsonl"


def run(args: argparse.Namespace) -> int:
    extra_body = parse_json_object(args.extra_body, "--extra-body")
    selected = filter_examples(load_examples(args), args)
    if not selected:
        print("No examples selected.")
        return 1

    output_file = resolve_output_file(args)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file = output_file.with_suffix(".summary.json")
    if output_file.exists() and not args.resume and not args.overwrite:
        raise SystemExit(
            f"Output file already exists: {output_file}. Pass --resume or --overwrite."
        )

    completed, existing_records = (set(), [])
    if args.resume and not args.overwrite:
        completed, existing_records = read_existing_results(output_file)
    examples_to_run = [item for item in selected if item["case_id"] not in completed]
    mode = "w" if args.overwrite or not output_file.exists() else "a"

    print(f"Dataset: {args.data_file or args.dataset_name}")
    print(f"Selected examples: {len(selected)}")
    print(f"Selection SHA-256: {selection_sha256(selected)}")
    print(f"Already completed: {len(completed)}")
    print(f"Running examples: {len(examples_to_run)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_file}")

    if args.warmup_tokens > 0 and examples_to_run:
        print(f"Warmup: 1 request, max_tokens={args.warmup_tokens}")
        warmup = post_chat_completion(
            args, examples_to_run[0], extra_body, max_tokens=args.warmup_tokens
        )
        if warmup.get("error"):
            raise SystemExit(f"Warmup failed: {warmup['error']}")

    started = time.perf_counter()
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
                desc="HLE",
            )
            for future in progress:
                record = future.result()
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                new_records.append(record)
                all_so_far = existing_records + new_records
                correct = sum(item.get("correct") is True for item in all_so_far)
                tokens = sum(int(item.get("output_tokens") or 0) for item in all_so_far)
                seconds = sum(float(item.get("latency_ms") or 0.0) for item in all_so_far) / 1000
                progress.set_postfix(
                    acc=f"{correct / len(all_so_far):.4f}",
                    tps=f"{tokens / seconds:.2f}" if seconds else "n/a",
                )

    run_wall_seconds = time.perf_counter() - started
    all_records = existing_records + new_records
    if not all_records:
        _, all_records = read_existing_results(output_file)
        run_wall_seconds = None
    summary = summarize(all_records, selected, args, run_wall_seconds)
    with summary_file.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
        fout.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI-compatible API on Humanity's Last Exam."
    )
    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1616")
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "ds"))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-file", default="")
    parser.add_argument("--output-dir", default="test/hle/results")
    parser.add_argument("--output-file", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--answer-type", action="append")
    parser.add_argument("--category", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help=(
            "Maximum completion tokens; 0 omits the field and uses the "
            "server's default completion limit."
        ),
    )
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--extra-body", default="")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--use-env-proxy", action="store_true")
    parser.add_argument("--skip-bad-examples", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.max_tokens < 0:
        parser.error("--max-tokens must be >= 0 (0 omits the request limit)")
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens must be >= 0")
    if args.request_timeout < 0:
        parser.error("--request-timeout must be >= 0 (0 disables the timeout)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
