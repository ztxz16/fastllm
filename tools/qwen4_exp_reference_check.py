#!/usr/bin/env python3
"""Layerwise eager-reference check for Qwen4-Exp / Qwen3.8-Flash-Next.

The checker intentionally does not instantiate the complete Hugging Face model:
the released PLE table alone is tens of GiB and the text model has hundreds of
GiB of expert parameters.  Instead it implements the public Transformers eager
equations, loads one decoder layer at a time, dequantizes only routed FP8
experts, and slices only the selected PLE rows from safetensors.  This makes a
full intermediate/logit comparison practical on a large CPU host or one GPU.

FastLLM dumps are enabled with FASTLLM_QWEN4_DUMP_DIR.  Their binary format is:
int32 rank, int32 dimensions[rank], then contiguous float32 values.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


LANGUAGE_PREFIX = "model.language_model."
MASK64 = (1 << 64) - 1


class SafeTensorCheckpoint:
    def __init__(self, model_dir: Path, device: torch.device, block_size: tuple[int, int]):
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self.device = device
        self.block_size = block_size

    def _path(self, name: str) -> Path:
        if name not in self.weight_map:
            raise KeyError(f"checkpoint tensor not found: {name}")
        return self.model_dir / self.weight_map[name]

    def raw(self, name: str) -> torch.Tensor:
        with safe_open(self._path(name), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def shape(self, name: str) -> tuple[int, ...]:
        with safe_open(self._path(name), framework="pt", device="cpu") as handle:
            return tuple(handle.get_slice(name).get_shape())

    def rows(self, name: str, rows: Iterable[int]) -> torch.Tensor:
        pieces = []
        with safe_open(self._path(name), framework="pt", device="cpu") as handle:
            source = handle.get_slice(name)
            for row in rows:
                pieces.append(source[row : row + 1])
        return torch.cat(pieces, dim=0).float().to(self.device)

    def tensor(self, name: str, *, dequantize: bool = True) -> torch.Tensor:
        value = self.raw(name)
        scale_name = name + "_scale_inv"
        if dequantize and scale_name in self.weight_map:
            scales = self.raw(scale_name).float()
            block_rows, block_columns = self.block_size
            expanded = scales.repeat_interleave(block_rows, dim=0).repeat_interleave(
                block_columns, dim=1
            )
            value = value.float() * expanded[: value.shape[0], : value.shape[1]]
        else:
            value = value.float()
        return value.to(self.device)


def read_dump(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        rank_bytes = handle.read(4)
        if len(rank_bytes) != 4:
            raise ValueError(f"invalid FastLLM dump header: {path}")
        rank = struct.unpack("<i", rank_bytes)[0]
        dims = struct.unpack(f"<{rank}i", handle.read(4 * rank))
        values = np.frombuffer(handle.read(), dtype="<f4").copy()
    expected = math.prod(dims)
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, found {values.size}")
    return torch.from_numpy(values.reshape(dims))


def rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    group_size: int | None = None,
    delta_weight: bool = True,
) -> torch.Tensor:
    original_shape = value.shape
    working = value.float()
    if group_size is not None:
        working = working.reshape(*working.shape[:-1], -1, group_size)
    working = working * torch.rsqrt(working.square().mean(dim=-1, keepdim=True) + eps)
    working = working.reshape(original_shape)
    multiplier = 1.0 + weight.float() if delta_weight else weight.float()
    return working * multiplier


def rotate_partial(value: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, rotary_dim, 2, device=value.device, dtype=torch.float32) / rotary_dim)
    )
    frequencies = positions.float().unsqueeze(-1) * inv_freq
    cos = torch.cat([frequencies, frequencies], dim=-1).cos()
    sin = torch.cat([frequencies, frequencies], dim=-1).sin()
    while cos.ndim < value.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotary = value[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = torch.cat([-rotary[..., half:], rotary[..., :half]], dim=-1)
    rotated = rotary * cos + rotated_half * sin
    return torch.cat([rotated, value[..., rotary_dim:]], dim=-1)


def metric(reference: torch.Tensor, dump: torch.Tensor) -> dict[str, float | int]:
    reference = reference.detach().float().cpu().reshape(-1)
    dump = dump.detach().float().cpu().reshape(-1)
    if reference.numel() != dump.numel():
        raise ValueError(f"shape mismatch: reference={tuple(reference.shape)} dump={tuple(dump.shape)}")
    difference = reference - dump
    reference_norm = torch.linalg.vector_norm(reference)
    cosine = F.cosine_similarity(reference, dump, dim=0).item()
    return {
        "elements": reference.numel(),
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-12)
        ).item(),
        "cosine": cosine,
    }


class Qwen4EagerReference:
    def __init__(self, model_dir: Path, dump_dir: Path, device: torch.device):
        self.model_dir = model_dir
        self.dump_dir = dump_dir
        self.device = device
        top_config = json.loads((model_dir / "config.json").read_text())
        self.config = top_config["text_config"]
        quant = top_config.get("quantization_config", {})
        block_size = tuple(quant.get("weight_block_size", [128, 128]))
        self.checkpoint = SafeTensorCheckpoint(model_dir, device, block_size)

        self.hidden_size = int(self.config["hidden_size"])
        self.hc_count = int(self.config["hc_count"])
        self.eps = float(self.config["rms_norm_eps"])
        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.num_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config["head_dim"])
        self.rotary_dim = int(self.head_dim * float(self.config["partial_rotary_factor"]))
        rope = self.config.get("rope_parameters", {})
        self.rope_theta = float(rope.get("rope_theta", self.config.get("rope_theta", 10_000_000.0)))
        self.num_experts = int(self.config["num_experts"])
        self.top_k = int(self.config["num_experts_per_tok"])
        self.layer_types = list(self.config["layer_types"])
        self.results: dict[str, dict[str, float | int]] = {}

    def weight(self, suffix: str) -> torch.Tensor:
        return self.checkpoint.tensor(LANGUAGE_PREFIX + suffix)

    def compare(self, name: str, value: torch.Tensor) -> None:
        dump_path = self.dump_dir / f"{name}.f32"
        if not dump_path.exists():
            raise FileNotFoundError(f"missing FastLLM dump: {dump_path}")
        result = metric(value, read_dump(dump_path))
        self.results[name] = result
        print(
            f"{name:22s} max={result['max_abs']:.6g} "
            f"mean={result['mean_abs']:.6g} rel_l2={result['relative_l2']:.6g} "
            f"cos={result['cosine']:.9f}",
            flush=True,
        )

    def hyper_mix(self, hidden: torch.Tensor, prefix: str, combine: bool = True):
        normalized = rms_norm(
            hidden,
            self.weight(prefix + "hc_norm.weight"),
            self.eps,
            group_size=self.hidden_size,
        )
        low_rank = F.linear(normalized, self.weight(prefix + "input_mix_weight_down.weight"))
        low_rank = F.silu(low_rank / self.hc_count)
        mix = torch.sigmoid(
            F.linear(low_rank, self.weight(prefix + "input_mix_weight_up.weight"))
        )
        mixed = (
            mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
            * normalized.reshape(*normalized.shape[:-1], self.hc_count, self.hidden_size)
        ).mean(dim=-2)
        if not combine:
            return mixed
        injection = 2.0 * torch.sigmoid(
            F.linear(normalized, self.weight(prefix + "block_inject_weight.weight"))
            / self.hc_count
        )
        return mixed, hidden, injection

    @staticmethod
    def hyper_combine(hyper: torch.Tensor, block: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        added = block.unsqueeze(-2) * injection.unsqueeze(-1)
        return hyper + added.flatten(-2)

    def ple(self, hidden: torch.Tensor, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
        prefix = f"layers.{layer}.ple."
        embedding_prefix = prefix + "ple_embedding.ngram_embedding."
        ngram_size = int(self.config["ngram_size"])
        heads_per_ngram = int(self.config["heads_per_ngram"])
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        ple_dim = int(self.config["ple_embed_dim"])
        head_dim = ple_dim // ngram_heads
        eos = int(self.config["eos_token_id"])

        metadata_prefix = prefix + "ple_embedding."
        multipliers = self.checkpoint.raw(
            LANGUAGE_PREFIX + metadata_prefix + "layer_multipliers"
        ).to(torch.int64).tolist()
        vocab_sizes = self.checkpoint.raw(
            LANGUAGE_PREFIX + metadata_prefix + "ngram_heads_vocab_sizes"
        ).to(torch.int64).tolist()
        offsets = self.checkpoint.raw(
            LANGUAGE_PREFIX + metadata_prefix + "ngram_heads_offsets"
        ).to(torch.int64).tolist()
        first_shard = LANGUAGE_PREFIX + embedding_prefix + "shard_0.weight"
        rows_per_shard = self.checkpoint.shape(first_shard)[0]
        scale = self.weight(embedding_prefix + "weight_scale").reshape(-1)[0]

        ids = input_ids.reshape(-1).cpu().to(torch.int64).tolist()
        previous1 = eos
        previous2 = eos
        selected: list[tuple[int, int]] = []
        for current in ids:
            shifted = [current, previous1, previous2]
            for ngram in range(2, ngram_size + 1):
                mixed = (shifted[0] * multipliers[0]) & MASK64
                for position in range(1, ngram):
                    mixed ^= (shifted[position] * multipliers[position]) & MASK64
                signed = mixed if mixed < (1 << 63) else mixed - (1 << 64)
                start = (ngram - 2) * heads_per_ngram
                for local_head in range(heads_per_ngram):
                    head = start + local_head
                    global_row = offsets[head] + signed % vocab_sizes[head]
                    selected.append((global_row // rows_per_shard, global_row % rows_per_shard))
            if current == eos:
                previous1 = previous2 = eos
            else:
                previous2, previous1 = previous1, current

        rows: list[torch.Tensor] = []
        for shard, row in selected:
            name = LANGUAGE_PREFIX + embedding_prefix + f"shard_{shard}.weight"
            rows.append(self.checkpoint.rows(name, [row])[0] * scale)
        embeddings = torch.stack(rows).reshape(1, len(ids), ple_dim)

        key = F.linear(embeddings, self.weight(prefix + "key_proj.weight"))
        value = F.linear(embeddings, self.weight(prefix + "value_proj.weight"))
        key = rms_norm(
            key,
            self.weight(prefix + "norm_key.weight"),
            self.eps,
            group_size=self.hidden_size,
        ).reshape(1, len(ids), self.hc_count, self.hidden_size)
        query = rms_norm(
            hidden,
            self.weight(prefix + "norm_query.weight"),
            self.eps,
            group_size=self.hidden_size,
        ).reshape(1, len(ids), self.hc_count, self.hidden_size)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        normalized = rms_norm(
            gated,
            self.weight(prefix + "norm_conv.weight"),
            self.eps,
            group_size=self.hidden_size,
        )
        convolution_weight = self.weight(prefix + "conv1d.weight")
        history = (convolution_weight.shape[-1] - 1) * ngram_size
        convolution = F.conv1d(
            F.pad(normalized.transpose(1, 2), (history, 0)),
            convolution_weight,
            groups=self.hc_count * self.hidden_size,
            dilation=ngram_size,
        ).transpose(1, 2)
        return gated + F.silu(convolution)

    def linear_attention(self, value: torch.Tensor, layer: int) -> torch.Tensor:
        prefix = f"layers.{layer}.linear_attn."
        batch, sequence, _ = value.shape
        key_heads = int(self.config["linear_num_key_heads"])
        value_heads = int(self.config["linear_num_value_heads"])
        key_dim = int(self.config["linear_key_head_dim"])
        value_dim = int(self.config["linear_value_head_dim"])
        key_columns = key_heads * key_dim
        value_columns = value_heads * value_dim

        qkv = F.linear(value, self.weight(prefix + "in_proj_qkv.weight"))
        z = F.linear(value, self.weight(prefix + "in_proj_z.weight"))
        beta = F.linear(value, self.weight(prefix + "in_proj_b.weight"))
        alpha = F.linear(value, self.weight(prefix + "in_proj_a.weight"))
        conv_weight = self.weight(prefix + "conv1d.weight")
        qkv = F.conv1d(
            qkv.transpose(1, 2),
            conv_weight,
            padding=conv_weight.shape[-1] - 1,
            groups=conv_weight.shape[0],
        )[:, :, :sequence]
        qkv = F.silu(qkv.transpose(1, 2))
        query, key, recurrent_value = torch.split(
            qkv, [key_columns, key_columns, value_columns], dim=-1
        )
        query = query.reshape(batch, sequence, key_heads, key_dim)
        key = key.reshape(batch, sequence, key_heads, key_dim)
        recurrent_value = recurrent_value.reshape(batch, sequence, value_heads, value_dim)
        repeat = value_heads // key_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
        query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
        query = query / math.sqrt(key_dim)
        beta = torch.sigmoid(beta)
        decay = -self.weight(prefix + "A_log").float().exp() * F.softplus(
            alpha.float() + self.weight(prefix + "dt_bias").float()
        )

        state = torch.zeros(batch, value_heads, key_dim, value_dim, device=self.device)
        outputs = []
        for token in range(sequence):
            q_t = query[:, token].float()
            k_t = key[:, token].float()
            v_t = recurrent_value[:, token].float()
            state = state * decay[:, token].exp().unsqueeze(-1).unsqueeze(-1)
            memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - memory) * beta[:, token].unsqueeze(-1)
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
        core = torch.stack(outputs, dim=1)
        core = core.reshape(-1, value_dim)
        core = rms_norm(
            core,
            self.weight(prefix + "norm.weight"),
            self.eps,
            delta_weight=False,
        )
        core = core * torch.sigmoid(z.reshape(-1, value_dim).float())
        return F.linear(
            core.reshape(batch, sequence, value_columns),
            self.weight(prefix + "out_proj.weight"),
        )

    def full_attention(self, value: torch.Tensor, positions: torch.Tensor, layer: int) -> torch.Tensor:
        prefix = f"layers.{layer}.self_attn."
        batch, sequence, _ = value.shape
        q_gate = F.linear(value, self.weight(prefix + "q_proj.weight"))
        q_gate = q_gate.reshape(batch, sequence, self.num_heads, 2 * self.head_dim)
        query, gate = q_gate.split(self.head_dim, dim=-1)
        key = F.linear(value, self.weight(prefix + "k_proj.weight")).reshape(
            batch, sequence, self.num_kv_heads, self.head_dim
        )
        attention_value = F.linear(value, self.weight(prefix + "v_proj.weight")).reshape(
            batch, sequence, self.num_kv_heads, self.head_dim
        )
        query = rms_norm(query, self.weight(prefix + "q_norm.weight"), self.eps)
        key = rms_norm(key, self.weight(prefix + "k_norm.weight"), self.eps)
        query = rotate_partial(query, positions, self.rotary_dim, self.rope_theta)
        key = rotate_partial(key, positions, self.rotary_dim, self.rope_theta)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        attention_value = attention_value.transpose(1, 2).repeat_interleave(
            self.num_heads // self.num_kv_heads, dim=1
        )
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool, device=self.device), diagonal=1
        )
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1)
        context = (probabilities @ attention_value.float()).transpose(1, 2).reshape(
            batch, sequence, self.num_heads * self.head_dim
        )
        context = context * torch.sigmoid(gate.reshape(batch, sequence, -1))
        return F.linear(context, self.weight(prefix + "o_proj.weight"))

    def moe(self, value: torch.Tensor, layer: int) -> torch.Tensor:
        prefix = f"layers.{layer}.mlp."
        flat = value.reshape(-1, self.hidden_size)
        router = torch.softmax(F.linear(flat, self.weight(prefix + "gate.weight")).float(), dim=-1)
        scores, experts = torch.topk(router, self.top_k, dim=-1)
        if bool(self.config.get("norm_topk_prob", True)):
            scores = scores / scores.sum(dim=-1, keepdim=True)
        scores = scores * float(self.config.get("routed_scaling_factor", 1.0))

        routed = torch.zeros_like(flat)
        for expert in torch.unique(experts).cpu().tolist():
            token_indices, ranks = torch.where(experts == expert)
            current = flat.index_select(0, token_indices)
            expert_prefix = prefix + f"experts.{expert}."
            gate = F.linear(current, self.weight(expert_prefix + "gate_proj.weight"))
            up = F.linear(current, self.weight(expert_prefix + "up_proj.weight"))
            current = F.silu(gate) * up
            current = F.linear(current, self.weight(expert_prefix + "down_proj.weight"))
            current = current * scores[token_indices, ranks].unsqueeze(-1)
            routed.index_add_(0, token_indices, current)

        shared_gate = F.linear(flat, self.weight(prefix + "shared_expert.gate_proj.weight"))
        shared_up = F.linear(flat, self.weight(prefix + "shared_expert.up_proj.weight"))
        shared = F.silu(shared_gate) * shared_up
        shared = F.linear(shared, self.weight(prefix + "shared_expert.down_proj.weight"))
        shared = shared * torch.sigmoid(
            F.linear(flat, self.weight(prefix + "shared_expert_gate.weight"))
        )
        return (routed + shared).reshape_as(value)

    def logits(self, hidden: torch.Tensor, rows_per_chunk: int = 4096) -> torch.Tensor:
        name = "lm_head.weight"
        vocab, _ = self.checkpoint.shape(name)
        pieces = []
        with safe_open(self.checkpoint._path(name), framework="pt", device="cpu") as handle:
            source = handle.get_slice(name)
            for start in range(0, vocab, rows_per_chunk):
                weight = source[start : min(start + rows_per_chunk, vocab)].float().to(self.device)
                pieces.append(F.linear(hidden, weight).cpu())
        return torch.cat(pieces, dim=-1)

    @torch.inference_mode()
    def run(self) -> dict[str, object]:
        input_ids = read_dump(self.dump_dir / "input_ids.f32").to(torch.int64).to(self.device)
        positions = read_dump(self.dump_dir / "position_ids.f32").to(torch.int64).to(self.device)
        embedding_name = LANGUAGE_PREFIX + "embed_tokens.weight"
        hidden = self.checkpoint.rows(embedding_name, input_ids.reshape(-1).cpu().tolist())
        hidden = hidden.reshape(1, input_ids.shape[1], self.hidden_size)
        hidden = hidden.repeat(1, 1, self.hc_count)
        self.compare("embedding", hidden)

        ple_layers = {int(index) - 1 for index in self.config.get("ple_layer_ids", [])}
        for layer in range(self.num_layers):
            if layer in ple_layers:
                hidden = hidden + self.ple(hidden, input_ids, layer)
                self.compare(f"layer_{layer}_ple", hidden)

            attn_input, hyper, injection = self.hyper_mix(
                hidden, f"layers.{layer}.attn_hyper_connection."
            )
            if self.layer_types[layer] == "linear_attention":
                attn_output = self.linear_attention(attn_input, layer)
            else:
                attn_output = self.full_attention(attn_input, positions, layer)
            self.compare(f"layer_{layer}_attention", attn_output)
            hidden = self.hyper_combine(hyper, attn_output, injection)

            mlp_input, hyper, injection = self.hyper_mix(
                hidden, f"layers.{layer}.mlp_hyper_connection."
            )
            mlp_output = self.moe(mlp_input, layer)
            hidden = self.hyper_combine(hyper, mlp_output, injection)
            self.compare(f"layer_{layer}_output", hidden)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        final_hidden = self.hyper_mix(hidden, "hyper_connection_mixer.", combine=False)
        self.compare("final_hidden", final_hidden)
        reference_logits = self.logits(final_hidden[:, -1:, :])
        self.compare("logits", reference_logits)
        dumped_logits = read_dump(self.dump_dir / "logits.f32").reshape(-1)
        reference_flat = reference_logits.reshape(-1)
        top_count = min(10, reference_flat.numel())
        reference_top = torch.topk(reference_flat, top_count).indices.cpu().tolist()
        dump_top = torch.topk(dumped_logits, top_count).indices.cpu().tolist()
        summary: dict[str, object] = {
            "input_ids": input_ids.cpu().reshape(-1).tolist(),
            "reference_argmax": int(reference_flat.argmax().item()),
            "fastllm_argmax": int(dumped_logits.argmax().item()),
            "reference_top10": reference_top,
            "fastllm_top10": dump_top,
            "top10_set_overlap": len(set(reference_top) & set(dump_top)),
            "metrics": self.results,
        }
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--json", type=Path, help="optional path for the complete metrics JSON")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    checker = Qwen4EagerReference(args.model_dir, args.dump_dir, device)
    summary = checker.run()
    print(json.dumps({key: value for key, value in summary.items() if key != "metrics"}, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
