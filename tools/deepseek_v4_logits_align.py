#!/usr/bin/env python3
"""Utilities for comparing DeepSeek-V4 raw logits across runtimes.

The ``prepare`` command creates a lightweight layer-truncated Hugging Face
directory.  It only writes a new config/index and symlinks the selected
checkpoint shards, so the original checkpoint is never modified or copied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


_SHARED_MAIN_WEIGHTS = {
    "embed.weight",
    "head.weight",
    "norm.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to replace existing subset entry: {destination}"
        )
    destination.symlink_to(source)


def prepare_subset(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source == output:
        raise ValueError("subset output must differ from the source checkpoint")
    if args.layers <= 0:
        raise ValueError("--layers must be positive")

    config_path = source / "config.json"
    index_path = source / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "source must contain config.json and model.safetensors.index.json"
        )

    config = _load_json(config_path)
    original_layers = int(config["num_hidden_layers"])
    if args.layers > original_layers:
        raise ValueError(
            f"requested {args.layers} layers, checkpoint only has {original_layers}"
        )

    index = _load_json(index_path)
    selected_weight_map: dict[str, str] = {}
    layer_pattern = re.compile(r"^layers\.(\d+)\.")
    for name, shard in index["weight_map"].items():
        match = layer_pattern.match(name)
        if name in _SHARED_MAIN_WEIGHTS or (
            match is not None and int(match.group(1)) < args.layers
        ):
            selected_weight_map[name] = shard

    missing_shared = _SHARED_MAIN_WEIGHTS - selected_weight_map.keys()
    if missing_shared:
        raise ValueError(f"checkpoint is missing shared weights: {missing_shared}")
    selected_shards = sorted(set(selected_weight_map.values()))

    output.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if not entry.is_file():
            continue
        if entry.name in {"config.json", "model.safetensors.index.json"}:
            continue
        if entry.suffix == ".safetensors" and entry.name not in selected_shards:
            continue
        _safe_symlink(entry, output / entry.name)

    subset_config = dict(config)
    subset_config["num_hidden_layers"] = args.layers
    subset_config["num_hash_layers"] = min(
        int(subset_config.get("num_hash_layers", 0)), args.layers
    )
    if isinstance(subset_config.get("compress_ratios"), list):
        subset_config["compress_ratios"] = subset_config["compress_ratios"][
            : args.layers
        ]
    # The target-model comparison intentionally excludes DSpark/MTP.
    subset_config["num_nextn_predict_layers"] = 0
    subset_config["dspark_block_size"] = 0
    subset_config["dspark_target_layer_ids"] = []

    subset_index = dict(index)
    subset_index["weight_map"] = selected_weight_map
    metadata = dict(subset_index.get("metadata", {}))
    metadata["total_size"] = sum((source / name).stat().st_size for name in selected_shards)
    subset_index["metadata"] = metadata

    _write_json(output / "config.json", subset_config)
    _write_json(output / "model.safetensors.index.json", subset_index)
    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "layers": args.layers,
                "weights": len(selected_weight_map),
                "shards": selected_shards,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def make_input(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    input_ids = tokenizer.encode(args.text, add_special_tokens=args.add_special_tokens)
    if not input_ids:
        raise ValueError("the encoded input is empty")
    payload = {"text": args.text, "input_ids": input_ids}
    _write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _read_input_ids(path: str) -> list[int]:
    payload = _load_json(Path(path))
    values = payload["input_ids"] if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("input-id file must contain a non-empty list")
    return [int(value) for value in values]


def dump_vllm(args: argparse.Namespace) -> None:
    # The verified local vLLM setup intentionally uses this compatibility
    # override: flashinfer-python is 0.6.14 while its SM120 cubin package is
    # labelled 0.6.13.
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    from vllm import LLM, SamplingParams
    from vllm.logprobs import FlatLogprobs

    input_ids = _read_input_ids(args.input_ids)
    config = _load_json(Path(args.model) / "config.json")
    vocab_size = int(config["vocab_size"])
    max_model_len = max(args.max_model_len, len(input_ids) + args.steps + 1)

    engine = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        kv_cache_dtype=args.kv_cache_dtype,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        disable_custom_all_reduce=True,
        enable_flashinfer_autotune=False,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.steps,
        ignore_eos=True,
        logprobs=-1,
        flat_logprobs=True,
        detokenize=False,
        allowed_token_ids=(
            [args.allowed_token_id] if args.allowed_token_id is not None else None
        ),
    )
    request_output = engine.generate(
        [{"prompt_token_ids": input_ids}], sampling, use_tqdm=False
    )[0]
    completion = request_output.outputs[0]
    flat = completion.logprobs
    if not isinstance(flat, FlatLogprobs):
        raise TypeError(f"expected FlatLogprobs, got {type(flat)!r}")

    logits = np.full((len(flat), vocab_size), np.nan, dtype=np.float32)
    for position, (start, end) in enumerate(
        zip(flat.start_indices, flat.end_indices)
    ):
        for token_id, value in zip(flat.token_ids[start:end], flat.logprobs[start:end]):
            if 0 <= token_id < vocab_size:
                logits[position, token_id] = value
    missing = np.isnan(logits).sum(axis=1)
    if np.any(missing):
        raise RuntimeError(f"vLLM did not return the full vocabulary: missing={missing}")

    np.savez(
        args.output,
        engine=np.asarray("vllm"),
        model=np.asarray(args.model),
        input_ids=np.asarray(input_ids, dtype=np.int32),
        output_ids=np.asarray(completion.token_ids, dtype=np.int32),
        logits=logits,
        vocab_size=np.asarray([vocab_size], dtype=np.int32),
        mode=np.asarray(
            json.dumps(
                {
                    "tp": args.tp,
                    "steps": args.steps,
                    "max_model_len": max_model_len,
                    "kv_cache_dtype": args.kv_cache_dtype,
                    "dspark": False,
                    "logprobs_mode": "raw_logits",
                    "allowed_token_id": args.allowed_token_id,
                }
            )
        ),
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "output_ids": completion.token_ids,
                "shape": list(logits.shape),
            },
            indent=2,
        )
    )


def _step_metrics(left: np.ndarray, right: np.ndarray, topk: int) -> dict[str, Any]:
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    diff = left64 - right64
    left_centered = left64 - left64.mean()
    right_centered = right64 - right64.mean()
    centered_diff = left_centered - right_centered
    # Match greedy decoding's np.argmax tie rule: among equal logits, the
    # smaller token id wins.  Reversing np.argsort would select the larger id
    # and can report a false top-1 mismatch at an exact tie.
    token_ids = np.arange(left64.size)
    topk = min(topk, left64.size)
    left_top = np.lexsort((token_ids, -left64))[:topk].tolist()
    right_top = np.lexsort((token_ids, -right64))[:topk].tolist()
    cosine = float(
        np.dot(left64, right64)
        / (np.linalg.norm(left64) * np.linalg.norm(right64))
    )
    centered_cosine = float(
        np.dot(left_centered, right_centered)
        / (np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    )
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(math.sqrt(np.mean(np.square(diff)))),
        "cosine": cosine,
        "centered_max_abs": float(np.max(np.abs(centered_diff))),
        "centered_mean_abs": float(np.mean(np.abs(centered_diff))),
        "centered_rmse": float(math.sqrt(np.mean(np.square(centered_diff)))),
        "centered_cosine": centered_cosine,
        "top1_left": left_top[0],
        "top1_right": right_top[0],
        "top1_equal": left_top[0] == right_top[0],
        "topk_left": left_top,
        "topk_right": right_top,
        "topk_overlap": len(set(left_top) & set(right_top)),
    }


def compare(args: argparse.Namespace) -> None:
    with np.load(args.left, allow_pickle=False) as left_file:
        left = {name: left_file[name] for name in left_file.files}
    with np.load(args.right, allow_pickle=False) as right_file:
        right = {name: right_file[name] for name in right_file.files}
    if not np.array_equal(left["input_ids"], right["input_ids"]):
        raise ValueError("the two runs used different input ids")
    if left["logits"].shape != right["logits"].shape:
        raise ValueError(
            f"logits shapes differ: {left['logits'].shape} vs {right['logits'].shape}"
        )

    report = {
        "left": args.left,
        "right": args.right,
        "input_tokens": int(left["input_ids"].size),
        "shape": list(left["logits"].shape),
        "output_ids_left": left["output_ids"].tolist(),
        "output_ids_right": right["output_ids"].tolist(),
        "output_ids_equal": bool(
            np.array_equal(left["output_ids"], right["output_ids"])
        ),
        "steps": {},
    }
    for step in range(left["logits"].shape[0]):
        label = "prefill" if step == 0 else f"decode_{step}"
        report["steps"][label] = _step_metrics(
            left["logits"][step], right["logits"][step], args.topk
        )
    _write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--layers", type=int, default=8)
    prepare_parser.set_defaults(func=prepare_subset)

    input_parser = subparsers.add_parser("make-input")
    input_parser.add_argument("--model", required=True)
    input_parser.add_argument("--text", required=True)
    input_parser.add_argument("--output", required=True)
    input_parser.add_argument("--add-special-tokens", action="store_true")
    input_parser.set_defaults(func=make_input)

    vllm_parser = subparsers.add_parser("vllm")
    vllm_parser.add_argument("--model", required=True)
    vllm_parser.add_argument("--input-ids", required=True)
    vllm_parser.add_argument("--output", required=True)
    vllm_parser.add_argument("--tp", type=int, default=8)
    vllm_parser.add_argument("--steps", type=int, default=2)
    vllm_parser.add_argument("--max-model-len", type=int, default=256)
    vllm_parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    vllm_parser.add_argument("--kv-cache-dtype", default="fp8")
    vllm_parser.add_argument(
        "--allowed-token-id",
        type=int,
        help=(
            "restrict sampling to this token while retaining processor-free raw "
            "logits; useful for giving both runtimes the same decode history"
        ),
    )
    vllm_parser.set_defaults(func=dump_vllm)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--topk", type=int, default=20)
    compare_parser.set_defaults(func=compare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
