#!/usr/bin/env python3
"""Run the official Transformers Qwen4-Exp model on CPU and compare logits.

The released checkpoint stores 512 routed experts separately and shards the
very large PLE embedding into 128 tensors.  Transformers represents both as
monolithic parameters, which would require materializing roughly 177 billion
parameters before a two-token check.  This runner keeps the official
``Qwen4ExpForCausalLM`` graph and replaces only those two storage containers
with CPU lazy readers.  Attention, Gated DeltaNet, hyper-connections, PLE hash
logic, routing, shared experts, norms, and the language-model head all execute
through the upstream Transformers implementation.

The FastLLM dump directory is produced with ``FASTLLM_QWEN4_DUMP_DIR``.  The
script must be run with a Transformers checkout that contains ``qwen4_exp`` on
``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
    Qwen4ExpTextRotaryEmbedding,
)

from qwen4_exp_reference_check import (
    LANGUAGE_PREFIX,
    SafeTensorCheckpoint,
    metric,
    read_dump,
)


class LazyCheckpointExperts(nn.Module):
    """Transformers expert interface backed by per-expert checkpoint tensors."""

    def __init__(self, checkpoint: SafeTensorCheckpoint, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.checkpoint = checkpoint
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.last_expert_ids: list[int] = []

    def _weight(self, expert_idx: int, projection: str) -> torch.Tensor:
        name = (
            f"{LANGUAGE_PREFIX}layers.{self.layer_idx}.mlp.experts."
            f"{expert_idx}.{projection}.weight"
        )
        return self.checkpoint.tensor(name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_ids = torch.unique(top_k_index, sorted=True).cpu().tolist()
        self.last_expert_ids = [int(index) for index in expert_ids]

        for expert_idx in self.last_expert_ids:
            token_idx, top_k_pos = torch.where(top_k_index == expert_idx)
            current_state = hidden_states.index_select(0, token_idx)
            gate = F.linear(current_state, self._weight(expert_idx, "gate_proj"))
            up = F.linear(current_state, self._weight(expert_idx, "up_proj"))
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, self._weight(expert_idx, "down_proj")
            )
            current_hidden_states *= top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states)

        return final_hidden_states


class LazyCheckpointEmbedding(nn.Module):
    """Embedding-compatible view over Qwen4-Exp's sharded FP8 PLE table."""

    def __init__(self, checkpoint: SafeTensorCheckpoint, checkpoint_prefix: str):
        super().__init__()
        self.checkpoint = checkpoint
        self.checkpoint_prefix = checkpoint_prefix
        first_name = checkpoint_prefix + "shard_0.weight"
        self.rows_per_shard, self.embedding_dim = checkpoint.shape(first_name)
        self.scale = float(checkpoint.tensor(checkpoint_prefix + "weight_scale").item())
        # Qwen4ExpTextNGramEmbedding explicitly consults ``weight.device``.
        self.register_buffer("weight", torch.empty(0), persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        original_shape = tuple(indices.shape)
        flat_indices = indices.reshape(-1).cpu().tolist()
        positions_by_shard: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for output_position, global_row in enumerate(flat_indices):
            shard_idx, local_row = divmod(int(global_row), self.rows_per_shard)
            positions_by_shard[shard_idx].append((output_position, local_row))

        output = torch.empty(len(flat_indices), self.embedding_dim, dtype=torch.float32)
        for shard_idx, positions_and_rows in positions_by_shard.items():
            positions, rows = zip(*positions_and_rows)
            name = self.checkpoint_prefix + f"shard_{shard_idx}.weight"
            values = self.checkpoint.rows(name, rows) * self.scale
            output[list(positions)] = values
        return output.reshape(*original_shape, self.embedding_dim).to(indices.device)


def checkpoint_name(model_parameter_name: str) -> str:
    if model_parameter_name == "lm_head.weight":
        return model_parameter_name
    if not model_parameter_name.startswith("model."):
        raise KeyError(f"cannot map Transformers parameter: {model_parameter_name}")
    return LANGUAGE_PREFIX + model_parameter_name.removeprefix("model.")


def replace_parameter(model: nn.Module, name: str, value: torch.Tensor) -> None:
    parent_name, attribute = name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    if attribute not in parent._parameters:
        raise KeyError(f"{name} is not a registered parameter")
    parent._parameters[attribute] = nn.Parameter(value, requires_grad=False)


def replace_buffer(model: nn.Module, name: str, value: torch.Tensor) -> None:
    parent_name, attribute = name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    if attribute not in parent._buffers:
        raise KeyError(f"{name} is not a registered buffer")
    parent._buffers[attribute] = value


def build_transformers_model(
    model_dir: Path, checkpoint: SafeTensorCheckpoint
) -> tuple[Qwen4ExpForCausalLM, list[LazyCheckpointExperts]]:
    top_config = json.loads((model_dir / "config.json").read_text())
    config = Qwen4ExpTextConfig.from_dict(top_config["text_config"])
    config._attn_implementation = "eager"
    config.use_cache = False
    config.output_router_logits = False

    print("constructing official Qwen4ExpForCausalLM on meta device", flush=True)
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)

    lazy_experts: list[LazyCheckpointExperts] = []
    for layer_idx, layer in enumerate(model.model.layers):
        experts = LazyCheckpointExperts(checkpoint, config, layer_idx)
        layer.mlp.experts = experts
        lazy_experts.append(experts)

    for layer_idx, layer in enumerate(model.model.layers):
        if layer.ple is None:
            continue
        prefix = (
            f"{LANGUAGE_PREFIX}layers.{layer_idx}.ple.ple_embedding."
            "ngram_embedding."
        )
        layer.ple.ple_embedding.ngram_embedding = LazyCheckpointEmbedding(checkpoint, prefix)

    # Rotary buffers are derived from config and are not stored in the checkpoint.
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config)

    parameters = list(model.named_parameters())
    for index, (name, _) in enumerate(parameters, start=1):
        source_name = checkpoint_name(name)
        if source_name not in checkpoint.weight_map:
            raise KeyError(f"checkpoint tensor not found for {name}: {source_name}")
        replace_parameter(model, name, checkpoint.raw(source_name).float())
        if index % 100 == 0 or index == len(parameters):
            print(f"loaded regular parameters: {index}/{len(parameters)}", flush=True)

    for name, value in list(model.named_buffers()):
        if value.device.type != "meta":
            continue
        source_name = checkpoint_name(name)
        if source_name not in checkpoint.weight_map:
            raise KeyError(f"checkpoint buffer not found for {name}: {source_name}")
        replace_buffer(model, name, checkpoint.raw(source_name))

    meta_tensors = [
        name
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if value.device.type == "meta"
    ]
    if meta_tensors:
        raise RuntimeError(f"unmaterialized meta tensors: {meta_tensors}")

    model.eval()
    return model, lazy_experts


def print_metric(name: str, result: dict[str, float | int]) -> None:
    print(
        f"{name:22s} max={result['max_abs']:.6g} "
        f"mean={result['mean_abs']:.6g} rel_l2={result['relative_l2']:.6g} "
        f"cos={result['cosine']:.9f}",
        flush=True,
    )


@torch.inference_mode()
def run_check(
    model_dir: Path,
    dump_dir: Path,
    threads: int,
    min_logits_cosine: float,
    max_logits_relative_l2: float,
) -> dict[str, object]:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(min(4, threads))
    top_config = json.loads((model_dir / "config.json").read_text())
    block_size = tuple(top_config.get("quantization_config", {}).get("weight_block_size", [128, 128]))
    checkpoint = SafeTensorCheckpoint(model_dir, torch.device("cpu"), block_size)

    start = time.monotonic()
    model, lazy_experts = build_transformers_model(model_dir, checkpoint)
    load_seconds = time.monotonic() - start
    print(f"model storage ready in {load_seconds:.3f}s", flush=True)

    results: dict[str, dict[str, float | int]] = {}

    def compare(name: str, value: torch.Tensor) -> None:
        result = metric(value, read_dump(dump_dir / f"{name}.f32"))
        results[name] = result
        print_metric(name, result)

    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        def layer_hook(_module, _args, output, index=layer_idx):
            compare(f"layer_{index}_output", output)

        handles.append(layer.register_forward_hook(layer_hook))

    final_hidden: list[torch.Tensor] = []

    def final_hook(_module, _args, output):
        final_hidden.append(output)
        compare("final_hidden", output)

    handles.append(model.model.hyper_connection_mixer.register_forward_hook(final_hook))

    input_ids = read_dump(dump_dir / "input_ids.f32").to(torch.int64)
    position_ids = read_dump(dump_dir / "position_ids.f32").to(torch.int64)
    forward_start = time.monotonic()
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        use_cache=False,
        output_router_logits=False,
        logits_to_keep=1,
    )
    forward_seconds = time.monotonic() - forward_start
    for handle in handles:
        handle.remove()

    logits = outputs.logits
    compare("logits", logits)
    dumped_logits = read_dump(dump_dir / "logits.f32").reshape(-1)
    reference_logits = logits.reshape(-1)
    top_count = min(10, reference_logits.numel())
    transformers_top = torch.topk(reference_logits, top_count).indices.cpu().tolist()
    fastllm_top = torch.topk(dumped_logits, top_count).indices.cpu().tolist()
    logits_metric = results["logits"]
    alignment_checks = {
        "argmax_match": int(reference_logits.argmax().item())
        == int(dumped_logits.argmax().item()),
        "cosine_at_least_minimum": logits_metric["cosine"] >= min_logits_cosine,
        "relative_l2_at_most_maximum": (
            logits_metric["relative_l2"] <= max_logits_relative_l2
        ),
    }
    summary: dict[str, object] = {
        "device": "cpu",
        "threads": threads,
        "transformers_version": transformers.__version__,
        "transformers_source": str(Path(transformers.__file__).resolve()),
        "input_ids": input_ids.reshape(-1).tolist(),
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "transformers_argmax": int(reference_logits.argmax().item()),
        "fastllm_argmax": int(dumped_logits.argmax().item()),
        "transformers_top10": transformers_top,
        "fastllm_top10": fastllm_top,
        "top10_set_overlap": len(set(transformers_top) & set(fastllm_top)),
        "alignment": {
            "passed": all(alignment_checks.values()),
            "minimum_logits_cosine": min_logits_cosine,
            "maximum_logits_relative_l2": max_logits_relative_l2,
            "checks": alignment_checks,
        },
        "routed_experts": [experts.last_expert_ids for experts in lazy_experts],
        "metrics": results,
    }
    print(f"official Transformers CPU forward completed in {forward_seconds:.3f}s", flush=True)
    print(
        "logits alignment: "
        + ("PASS" if summary["alignment"]["passed"] else "FAIL"),
        flush=True,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "metrics"}, indent=2))
    del outputs, logits, final_hidden, model
    gc.collect()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--min-logits-cosine", type=float, default=0.999)
    parser.add_argument("--max-logits-relative-l2", type=float, default=0.02)
    parser.add_argument("--json", type=Path, help="optional complete result JSON")
    args = parser.parse_args()
    summary = run_check(
        args.model_dir,
        args.dump_dir,
        args.threads,
        args.min_logits_cosine,
        args.max_logits_relative_l2,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["alignment"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
