#!/usr/bin/env python3
"""Capture Dots3-Note Transformers logits with routed experts swapped per layer.

The released checkpoint is too large to keep all routed experts on one GPU.
Weights are loaded in their native FP8 representation on CPU.  Non-routed text
weights stay on one GPU, while each routed-expert tensor is copied to the GPU
only for the duration of its layer's grouped-MM call.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import time
import types
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-output")
    parser.add_argument(
        "--indexer-output",
        help="save only each DSA layer's final-query TopK indices",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--repeat-to",
        type=int,
        default=0,
        help="repeat the supplied input-id pattern to this prompt length",
    )
    parser.add_argument(
        "--disable-dsa",
        action="store_true",
        help="disable DSA only for a <=2048 dense-equivalence baseline",
    )
    parser.add_argument("--indexer-query-chunk-size", type=int, default=128)
    parser.add_argument("--sparse-query-chunk-size", type=int, default=8)
    parser.add_argument("--sparse-head-chunk-size", type=int, default=4)
    parser.add_argument("--dense-query-chunk-size", type=int, default=256)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help="feed the initial prompt through the cache in bounded chunks",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--experts-implementation",
        default="grouped_mm",
        choices=("grouped_mm", "batched_mm", "eager"),
    )
    return parser.parse_args()


def read_input_ids(path: str) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload["input_ids"] if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("input-id file must contain a non-empty list")
    return [int(value) for value in values]


def tensor_bytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def move_non_routed_tensors(module, fp8_experts_class, device):
    """Move direct parameters/buffers recursively, stopping at FP8Experts."""
    import torch

    moved_parameters: dict[int, torch.nn.Parameter] = {}
    moved_buffers: dict[int, torch.Tensor] = {}
    total_bytes = 0

    def visit(current) -> None:
        nonlocal total_bytes
        if isinstance(current, fp8_experts_class):
            return

        for name, parameter in list(current._parameters.items()):
            if parameter is None:
                continue
            key = id(parameter)
            moved = moved_parameters.get(key)
            if moved is None:
                moved = torch.nn.Parameter(
                    parameter.detach().to(device), requires_grad=False
                )
                moved_parameters[key] = moved
                total_bytes += tensor_bytes(moved)
            current._parameters[name] = moved

        for name, buffer in list(current._buffers.items()):
            if buffer is None:
                continue
            key = id(buffer)
            moved = moved_buffers.get(key)
            if moved is None:
                moved = buffer.detach().to(device)
                moved_buffers[key] = moved
                total_bytes += tensor_bytes(moved)
            current._buffers[name] = moved

        for child in current.children():
            visit(child)

    visit(module)
    return total_bytes


def install_expert_swap(experts, layer: int, device) -> int:
    """Keep canonical parameters on CPU and install a GPU-copying forward."""
    import torch

    cpu_parameters = dict(experts._parameters)
    if not cpu_parameters:
        raise RuntimeError(f"layer {layer} routed experts have no parameters")
    for name, parameter in cpu_parameters.items():
        if parameter is not None and parameter.device.type != "cpu":
            raise RuntimeError(
                f"layer {layer} expert parameter {name} is unexpectedly on "
                f"{parameter.device}"
            )

    resident_bytes = sum(
        tensor_bytes(parameter)
        for parameter in cpu_parameters.values()
        if parameter is not None
    )
    original_forward = experts.forward

    def swapping_forward(self, *args, **kwargs):
        started = time.monotonic()
        print(
            f"[transformers] layer {layer:02d}: staging "
            f"{resident_bytes / 2**30:.2f} GiB routed experts",
            flush=True,
        )
        gpu_parameters = {
            name: (
                None
                if parameter is None
                else torch.nn.Parameter(
                    parameter.detach().to(device), requires_grad=False
                )
            )
            for name, parameter in cpu_parameters.items()
        }
        self._parameters.clear()
        self._parameters.update(gpu_parameters)
        try:
            result = original_forward(*args, **kwargs)
            # The grouped kernels are asynchronous.  Their weight storage must
            # remain alive until the result has finished writing.
            torch.cuda.synchronize(device)
            return result
        finally:
            self._parameters.clear()
            self._parameters.update(cpu_parameters)
            del gpu_parameters
            print(
                f"[transformers] layer {layer:02d}: done in "
                f"{time.monotonic() - started:.2f}s",
                flush=True,
            )

    experts.forward = types.MethodType(swapping_forward, experts)
    return resident_bytes


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.repeat_to < 0:
        raise ValueError("--repeat-to must be non-negative")
    if args.prefill_chunk_size < 0:
        raise ValueError("--prefill-chunk-size must be non-negative")
    if args.disable_dsa and args.indexer_output:
        raise ValueError("--indexer-output requires DSA")
    for name in (
        "indexer_query_chunk_size",
        "sparse_query_chunk_size",
        "sparse_head_chunk_size",
        "dense_query_chunk_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    import torch
    from transformers import Dots3NoteConfig
    from transformers.integrations.finegrained_fp8 import FP8Experts
    from transformers.models.dots3_note.modeling_dots3_note import (
        Dots3NoteTextForCausalLM,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("this reference runner requires CUDA")
    device = torch.device(args.device)
    input_ids = read_input_ids(args.input_ids)
    if args.repeat_to > 0:
        input_ids = [
            input_ids[index % len(input_ids)]
            for index in range(args.repeat_to)
        ]
    if args.disable_dsa and len(input_ids) > 2048:
        raise ValueError("--disable-dsa supports at most 2048 prompt tokens")
    model_path = str(Path(args.model).expanduser().resolve())

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    config = Dots3NoteConfig.from_pretrained(
        model_path, local_files_only=True
    )
    config.use_dsa = not args.disable_dsa
    config.use_cache = True

    # The reference implementation intentionally favors readability.  Smaller
    # query/head tiles keep its temporary gathers bounded on a 24 GiB GPU while
    # preserving the exact DSA equations.
    if config.use_dsa:
        from transformers.models.dots3_note import modeling_dots3_note

        original_eager_attention = (
            modeling_dots3_note.eager_attention_forward
        )
        original_sparse_attention = (
            modeling_dots3_note.dsa_sparse_attention_forward
        )

        @functools.wraps(original_eager_attention)
        def bounded_eager_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            *call_args,
            **call_kwargs,
        ):
            query_length = query.shape[2]
            if query_length <= args.dense_query_chunk_size:
                return original_eager_attention(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    *call_args,
                    **call_kwargs,
                )
            output_chunks = []
            weight_chunks = []
            keep_weights = bool(
                call_kwargs.get("output_attentions", False)
            )
            for start in range(0, query_length, args.dense_query_chunk_size):
                stop = min(start + args.dense_query_chunk_size, query_length)
                mask_chunk = (
                    None
                    if attention_mask is None
                    else attention_mask[:, :, start:stop]
                )
                output_chunk, weight_chunk = original_eager_attention(
                    module,
                    query[:, :, start:stop],
                    key,
                    value,
                    mask_chunk,
                    *call_args,
                    **call_kwargs,
                )
                output_chunks.append(output_chunk)
                if keep_weights:
                    weight_chunks.append(weight_chunk)
            return (
                torch.cat(output_chunks, dim=1),
                torch.cat(weight_chunks, dim=2) if keep_weights else None,
            )

        @functools.wraps(original_sparse_attention)
        def bounded_sparse_attention(*call_args, **call_kwargs):
            call_kwargs.setdefault(
                "query_chunk_size", args.sparse_query_chunk_size
            )
            call_kwargs.setdefault(
                "head_chunk_size", args.sparse_head_chunk_size
            )
            return original_sparse_attention(*call_args, **call_kwargs)

        modeling_dots3_note.dsa_sparse_attention_forward = (
            bounded_sparse_attention
        )
        modeling_dots3_note.eager_attention_forward = (
            bounded_eager_attention
        )

    print("[transformers] loading native FP8 text weights on CPU", flush=True)
    load_started = time.monotonic()
    model = Dots3NoteTextForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        experts_implementation=args.experts_implementation,
    )
    model.eval()
    if config.use_dsa:
        for decoder_layer in model.model.layers:
            indexer = getattr(decoder_layer.self_attn, "indexer", None)
            if indexer is not None:
                indexer.query_chunk_size = args.indexer_query_chunk_size
    print(
        f"[transformers] loaded in {time.monotonic() - load_started:.2f}s",
        flush=True,
    )

    routed_bytes = 0
    for layer, decoder_layer in enumerate(model.model.layers):
        if layer == 0:
            continue
        experts = decoder_layer.mlp.experts
        if not isinstance(experts, FP8Experts):
            raise TypeError(
                f"layer {layer} expected FP8Experts, got {type(experts)!r}"
            )
        routed_bytes += install_expert_swap(experts, layer, device)

    persistent_bytes = move_non_routed_tensors(model, FP8Experts, device)
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "[transformers] placement: "
        f"{persistent_bytes / 2**30:.2f} GiB persistent CUDA, "
        f"{routed_bytes / 2**30:.2f} GiB routed CPU",
        flush=True,
    )

    hidden_records: dict[str, np.ndarray] = {}
    indexer_records: dict[str, np.ndarray] = {}
    active_step = [-1]
    hook_handles = []
    if args.hidden_output:
        def capture(name):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                hidden_records[
                    f"step_{active_step[0]}_{name}"
                ] = tensor.detach().float().cpu().numpy()
            return hook

        def capture_gate(name):
            def hook(_module, _inputs, output):
                topk_idx, topk_weight = output
                hidden_records[
                    f"step_{active_step[0]}_{name}_expert_index"
                ] = topk_idx.detach().to(torch.int32).cpu().numpy()
                hidden_records[
                    f"step_{active_step[0]}_{name}_expert_score"
                ] = topk_weight.detach().float().cpu().numpy()
            return hook

        hook_handles.append(
            model.model.embed_tokens.register_forward_hook(
                capture("embedding_output")
            )
        )
        for layer, decoder_layer in enumerate(model.model.layers):
            prefix = f"layer_{layer:02d}"
            hook_handles.append(
                decoder_layer.self_attn.register_forward_hook(
                    capture(f"{prefix}_attention")
                )
            )
            indexer = getattr(decoder_layer.self_attn, "indexer", None)
            if indexer is not None:
                def capture_indexer(name):
                    def hook(_module, _inputs, output):
                        hidden_records[
                            f"step_{active_step[0]}_{name}_last_topk"
                        ] = output[:, -1].detach().to(
                            torch.int32
                        ).cpu().numpy()
                    return hook

                hook_handles.append(
                    indexer.register_forward_hook(
                        capture_indexer(f"{prefix}_indexer")
                    )
                )
            hook_handles.append(
                decoder_layer.mlp.register_forward_hook(
                    capture(f"{prefix}_mlp")
                )
            )
            if layer > 0:
                hook_handles.append(
                    decoder_layer.mlp.gate.register_forward_hook(
                        capture_gate(prefix)
                    )
                )
            hook_handles.append(
                decoder_layer.register_forward_hook(
                    capture(f"{prefix}_output")
                )
            )

    if args.indexer_output:
        def capture_indexer_only(name):
            def hook(_module, _inputs, output):
                indexer_records[
                    f"step_{active_step[0]}_{name}_last_topk"
                ] = output[:, -1].detach().to(
                    torch.int32
                ).cpu().numpy()
            return hook

        for layer, decoder_layer in enumerate(model.model.layers):
            indexer = getattr(decoder_layer.self_attn, "indexer", None)
            if indexer is not None:
                hook_handles.append(
                    indexer.register_forward_hook(
                        capture_indexer_only(f"layer_{layer:02d}_indexer")
                    )
                )

    logits_steps: list[np.ndarray] = []
    output_ids: list[int] = []
    current_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.empty(
        (1, 0), dtype=torch.long, device=device
    )
    past_key_values = None

    with torch.inference_mode():
        for step in range(args.steps):
            active_step[0] = step
            if step == 0 and args.prefill_chunk_size > 0:
                input_chunks = current_ids.split(
                    args.prefill_chunk_size, dim=1
                )
            else:
                input_chunks = (current_ids,)
            for input_chunk in input_chunks:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], input_chunk.shape[1]),
                            dtype=attention_mask.dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
                outputs = model(
                    input_ids=input_chunk,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
            step_logits = outputs.logits[0, -1].float()
            token_id = int(torch.argmax(step_logits).item())
            logits_steps.append(step_logits.cpu().numpy())
            output_ids.append(token_id)
            current_ids = torch.tensor(
                [[token_id]], dtype=torch.long, device=device
            )
            print(
                f"[transformers] step {step}: token={token_id}", flush=True
            )

    stacked_logits = np.stack(logits_steps).astype(np.float32, copy=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        engine=np.asarray("transformers"),
        model=np.asarray(model_path),
        input_ids=np.asarray(input_ids, dtype=np.int32),
        output_ids=np.asarray(output_ids, dtype=np.int32),
        logits=stacked_logits,
        vocab_size=np.asarray([stacked_logits.shape[-1]], dtype=np.int32),
        mode=np.asarray(
            json.dumps(
                {
                    "device": str(device),
                    "dtype": "bfloat16",
                    "checkpoint_quantization": "fp8_e4m3_block128",
                    "attn_implementation": "eager",
                    "experts_implementation": args.experts_implementation,
                    "use_dsa": config.use_dsa,
                    "prompt_tokens": len(input_ids),
                    "indexer_query_chunk_size": (
                        args.indexer_query_chunk_size
                    ),
                    "sparse_query_chunk_size": (
                        args.sparse_query_chunk_size
                    ),
                    "sparse_head_chunk_size": (
                        args.sparse_head_chunk_size
                    ),
                    "dense_query_chunk_size": (
                        args.dense_query_chunk_size
                    ),
                    "prefill_chunk_size": args.prefill_chunk_size,
                    "steps": args.steps,
                }
            )
        ),
    )
    if args.hidden_output:
        hidden_output_path = Path(args.hidden_output)
        hidden_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(hidden_output_path, **hidden_records)
    if args.indexer_output:
        indexer_output_path = Path(args.indexer_output)
        indexer_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(indexer_output_path, **indexer_records)
    for handle in hook_handles:
        handle.remove()
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_ids": output_ids,
                "shape": list(stacked_logits.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
