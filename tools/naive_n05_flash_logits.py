#!/usr/bin/env python3
"""Capture and compare raw Naive-N0.5 logits, using the checkpoint's HF code.

FastLLM mode accepts the normal ftllm CLI flags plus --input-ids/--output/--steps.
Reference mode needs transformers>=5.17, torch, kernels, accelerate, and a CUDA
GPU. It retains native FP8 experts on CPU and stages one layer at a time; all
model equations and FP8 GEMMs run through the original Transformers modules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile
import time

import numpy as np


def compare(reference, actual):
    ref, got = np.load(reference), np.load(actual)
    if not np.array_equal(ref["input_ids"], got["input_ids"]):
        raise ValueError("The input token IDs differ")
    a, b = ref["logits"].astype(np.float64), got["logits"].astype(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Logit shapes differ or contain NaN/Inf")
    if not np.array_equal(ref["output_ids"][:-1], got["output_ids"][:-1]):
        raise ValueError("Decode prefixes differ; use --teacher-forcing for the reference")
    metrics = []
    for x, y in zip(a, b):
        p, q = np.exp(x - x.max()), np.exp(y - y.max())
        p, q = p / p.sum(), q / q.sum()
        metrics.append({
            "mae": float(np.abs(x - y).mean()),
            "rmse": float(np.sqrt(np.square(x - y).mean())),
            "max_abs": float(np.abs(x - y).max()),
            "cosine": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
            "kl_ref_actual": float(np.sum(p * (np.log(p) - np.log(q)))),
            "top1_reference": int(x.argmax()), "top1_actual": int(y.argmax()),
            "top10_overlap": len(set(x.argsort()[-10:]) & set(y.argsort()[-10:])),
        })
    print(json.dumps(metrics, indent=2))


def reference():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeat-to", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-forcing", help="FastLLM NPZ whose generated IDs feed subsequent steps")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.integrations.finegrained_fp8 import FP8Experts
    from dots3_note_transformers_logits import move_non_routed_tensors, install_expert_swap

    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_path = Path(args.model).expanduser().resolve()
    payload = json.loads(Path(args.input_ids).read_text())
    ids = payload["input_ids"] if isinstance(payload, dict) else payload
    if args.repeat_to:
        ids = [ids[i % len(ids)] for i in range(args.repeat_to)]
    forced = np.load(args.teacher_forcing)["output_ids"] if args.teacher_forcing else None
    # Some downloads contain all shards but omit the HF index. Build the
    # missing manifest beside symlinks, leaving the checkpoint untouched.
    with tempfile.TemporaryDirectory(prefix="naive-hf-") as directory:
        root = Path(directory)
        for path in model_path.iterdir():
            if path.is_file():
                (root / path.name).symlink_to(path)
        if not (root / "model.safetensors.index.json").exists():
            mapping, size = {}, 0
            for shard in sorted(model_path.glob("*.safetensors")):
                with shard.open("rb") as handle:
                    header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
                for name, item in header.items():
                    if name == "__metadata__":
                        continue
                    mapping[name] = shard.name
                    size += item["data_offsets"][1] - item["data_offsets"][0]
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"metadata": {"total_size": size}, "weight_map": mapping}))
        started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            root, trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, device_map="cpu", experts_implementation="eager")
        print(f"Reference loaded in {time.monotonic() - started:.1f}s", flush=True)
    model.eval()
    device = torch.device(args.device)
    for layer, decoder in enumerate(model.model.layers):
        if hasattr(decoder.mlp, "experts"):
            assert isinstance(decoder.mlp.experts, FP8Experts)
            install_expert_swap(decoder.mlp.experts, layer, device)
    resident = move_non_routed_tensors(model, FP8Experts, device)
    print(f"Reference dense GPU weights: {resident / 2**30:.2f} GiB", flush=True)
    logits, output_ids, greedy_ids = [], [], []
    current = torch.tensor([ids], device=device)
    cache = None
    with torch.inference_mode():
        for step in range(args.steps):
            result = model(input_ids=current, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            values = result.logits[0, -1].float().cpu().numpy()
            greedy = int(values.argmax())
            token = int(forced[step]) if forced is not None else greedy
            logits.append(values)
            greedy_ids.append(greedy)
            output_ids.append(token)
            current = torch.tensor([[token]], device=device)
            print(f"Reference step {step}: greedy={greedy}, next={token}", flush=True)
    np.savez(args.output, input_ids=np.asarray(ids), output_ids=np.asarray(output_ids),
             greedy_ids=np.asarray(greedy_ids), logits=np.stack(logits),
             engine=np.asarray("original_transformers_fp8_eager"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: naive_n05_flash_logits.py {fastllm|reference|compare} ...")
    mode = sys.argv.pop(1)
    if mode == "fastllm":
        from dots3_note_fastllm_logits import main
        main()
    elif mode == "reference":
        reference()
    elif mode == "compare" and len(sys.argv) == 3:
        compare(*sys.argv[1:])
    else:
        raise SystemExit("Usage: naive_n05_flash_logits.py {fastllm|reference|compare} ...")
