#!/usr/bin/env python3
"""Capture processor-free GLM-5.3-Flash logits with Transformers on CPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-output")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def read_input_ids(path: str) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload["input_ids"] if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("input-id file must contain a non-empty list")
    return [int(value) for value in values]


def resident_gib() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**30


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import transformers
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
    )

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    model_path = str(Path(args.model).expanduser().resolve())
    input_ids = read_input_ids(args.input_ids)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.use_cache = args.steps > 1

    load_started = time.perf_counter()
    model = Glm5NextForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    load_rss_gib = resident_gib()
    print(
        json.dumps(
            {
                "stage": "loaded",
                "seconds": load_seconds,
                "rss_gib": load_rss_gib,
                "transformers": transformers.__version__,
                "torch": torch.__version__,
            }
        ),
        flush=True,
    )

    current_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(current_ids)
    past_key_values = None
    output_ids: list[int] = []
    logits: list[np.ndarray] = []
    step_seconds: list[float] = []

    layer_hidden: list[list[np.ndarray | None]] | None = None
    final_hidden: list[np.ndarray | None] | None = None
    embedding_hidden: list[np.ndarray | None] | None = None
    hook_handles = []
    active_step = [0]
    if args.hidden_output:
        text_model = model.model.language_model
        layer_hidden = [
            [None] * len(text_model.layers) for _ in range(args.steps)
        ]
        final_hidden = [None] * args.steps
        embedding_hidden = [None] * args.steps

        def capture_embedding(_module, _inputs, output):
            embedding_hidden[active_step[0]] = (
                output[0, -1].float().cpu().numpy().copy()
            )

        def capture_layer(layer_index: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                layer_hidden[active_step[0]][layer_index] = (
                    hidden[0, -1].float().cpu().numpy().copy()
                )

            return hook

        def capture_final(_module, _inputs, output):
            final_hidden[active_step[0]] = (
                output[0, -1].float().cpu().numpy().copy()
            )

        hook_handles.append(
            text_model.embed_tokens.register_forward_hook(capture_embedding)
        )
        hook_handles.extend(
            layer.register_forward_hook(capture_layer(index))
            for index, layer in enumerate(text_model.layers)
        )
        hook_handles.append(text_model.norm.register_forward_hook(capture_final))

    with torch.inference_mode():
        for step in range(args.steps):
            active_step[0] = step
            step_started = time.perf_counter()
            outputs = model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=args.steps > 1,
                return_dict=True,
            )
            raw_logits = outputs.logits[0, -1].float().cpu()
            token_id = int(torch.argmax(raw_logits).item())
            logits.append(raw_logits.numpy().copy())
            output_ids.append(token_id)
            step_seconds.append(time.perf_counter() - step_started)
            print(
                json.dumps(
                    {
                        "stage": "forward",
                        "step": step,
                        "token_id": token_id,
                        "seconds": step_seconds[-1],
                        "rss_gib": resident_gib(),
                    }
                ),
                flush=True,
            )
            if step + 1 < args.steps:
                past_key_values = outputs.past_key_values
                current_ids = torch.tensor([[token_id]], dtype=torch.long)
                attention_mask = torch.ones(
                    (1, len(input_ids) + step + 1), dtype=torch.long
                )

    for handle in hook_handles:
        handle.remove()

    if args.hidden_output:
        assert (
            layer_hidden is not None
            and final_hidden is not None
            and embedding_hidden is not None
        )
        if any(value is None for step in layer_hidden for value in step):
            raise RuntimeError("one or more decoder-layer hooks did not run")
        if any(value is None for value in final_hidden):
            raise RuntimeError("the final hidden-state hook did not run")
        if any(value is None for value in embedding_hidden):
            raise RuntimeError("the embedding hook did not run")
        hidden_path = Path(args.hidden_output)
        hidden_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            hidden_path,
            embedding_hidden=np.stack(embedding_hidden).astype(
                np.float32, copy=False
            ),
            layer_hidden=np.stack(
                [np.stack(step) for step in layer_hidden]
            ).astype(np.float32, copy=False),
            final_hidden=np.stack(final_hidden).astype(
                np.float32, copy=False
            ),
        )

    stacked_logits = np.stack(logits).astype(np.float32, copy=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        engine=np.asarray("transformers-cpu-bf16"),
        model=np.asarray(model_path),
        input_ids=np.asarray(input_ids, dtype=np.int32),
        output_ids=np.asarray(output_ids, dtype=np.int32),
        logits=stacked_logits,
        vocab_size=np.asarray([stacked_logits.shape[-1]], dtype=np.int32),
        load_seconds=np.asarray([load_seconds], dtype=np.float64),
        load_rss_gib=np.asarray([load_rss_gib], dtype=np.float64),
        step_seconds=np.asarray(step_seconds, dtype=np.float64),
        mode=np.asarray(
            json.dumps(
                {
                    "device": "cpu",
                    "dtype": "bfloat16",
                    "threads": args.threads,
                    "transformers": transformers.__version__,
                    "torch": torch.__version__,
                    "attention": "eager",
                    "fp8_checkpoint": "dequantized",
                }
            )
        ),
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_ids": output_ids,
                "shape": list(stacked_logits.shape),
                "hidden_output": args.hidden_output,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
