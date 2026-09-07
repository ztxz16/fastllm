"""Conservative discovery of built-in MTP without importing a tensor runtime."""

import json
import re
import struct
from pathlib import Path

from .gguf_metadata import (
    get_gguf_model_config,
    read_gguf_metadata,
    read_gguf_tensor_names,
)


def _safetensors_mtp_names(path):
    with path.open("rb") as stream:
        header_size, = struct.unpack("<Q", stream.read(8))
        if not 0 < header_size <= 8 * 1024 * 1024:
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    payload_size = path.stat().st_size - 8 - header_size
    names = set()
    for name, tensor in header.items():
        if not name.startswith("mtp."):
            continue
        start, end = tensor["data_offsets"]
        if not 0 <= start < end <= payload_size:
            raise ValueError("truncated MTP tensor")
        names.add(name)
    return names


def _hf_mtp_names(path):
    # Match the native HF loader: an unreferenced mtp.safetensors sidecar is
    # not loaded merely because it happens to be in the model directory.
    index_path = path / "model.safetensors.index.json"
    files = {"model.safetensors"}
    if index_path.is_file():
        if index_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("safetensors index is too large")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files = set(index["weight_map"].values())
        if not files or len(files) > 4096:
            raise ValueError("invalid safetensors index")
    names = set()
    for filename in files:
        names.update(_safetensors_mtp_names(path / filename))
    return names


def _gguf_mtp_names(path, architecture, layers):
    metadata = read_gguf_metadata(path, {architecture + ".block_count"})
    block_count = int(metadata.get(architecture + ".block_count", 0))
    if block_count <= layers:
        raise ValueError("invalid GGUF MTP block count")
    paths = [path]
    split = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name, re.IGNORECASE)
    if split:
        count = int(split[3])
        if not 1 <= count <= 4096:
            raise ValueError("invalid GGUF split count")
        paths = [path.with_name(f"{split[1]}-{part:05d}-of-{count:05d}.gguf")
                 for part in range(1, count + 1)]
    source_names = set()
    for shard in paths:
        source_names.update(read_gguf_tensor_names(shard))
    root_names = {
        "nextn.eh_proj.weight": "fc.weight",
        "nextn.enorm.weight": "pre_fc_norm_embedding.weight",
        "nextn.hnorm.weight": "pre_fc_norm_hidden.weight",
        "nextn.shared_head_norm.weight": "norm.weight",
    }
    layer_names = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_gate_inp.weight": "mlp.gate.weight",
    }
    for projection in ("q", "k", "v"):
        layer_names[f"attn_{projection}.weight"] = f"self_attn.{projection}_proj.weight"
    for projection in ("q", "k"):
        layer_names[f"attn_{projection}_norm.weight"] = f"self_attn.{projection}_norm.weight"
    for projection in ("gate", "up", "down"):
        layer_names[f"ffn_{projection}.weight"] = f"mlp.{projection}_proj.weight"
        layer_names[f"ffn_{projection}_exps.weight"] = f"mlp.experts.{projection}_proj.weight"
        layer_names[f"ffn_{projection}_shexp.weight"] = f"mlp.shared_expert.{projection}_proj.weight"
    names = set()
    for name in source_names:
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if not match:
            continue
        layer = int(match[1]) - (block_count - layers)
        if not 0 <= layer < layers:
            continue
        suffix = match[2]
        if suffix in root_names:
            names.add("mtp." + root_names[suffix])
        elif suffix in layer_names:
            names.add(f"mtp.layers.{layer}." + layer_names[suffix])
    return names


def _has_mlp(names, prefix):
    return prefix + "down_proj.weight" in names and (
        prefix + "gateup_proj.weight" in names
        or {prefix + "gate_proj.weight", prefix + "up_proj.weight"} <= names
    )


def _has_mtp_weights(names, text_config, layers):
    required = {"mtp.fc.weight", "mtp.norm.weight",
                "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"}
    if not required <= names:
        return False
    for layer in range(layers):
        prefix = f"mtp.layers.{layer}."
        required = {prefix + suffix for suffix in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        )}
        if not required <= names:
            return False
        if prefix + "self_attn.mergeqkv.weight" not in names and not {
            prefix + f"self_attn.{projection}_proj.weight" for projection in ("q", "k", "v")
        } <= names:
            return False
        mlp = prefix + "mlp."
        if _has_mlp(names, mlp):
            continue
        experts = int(text_config.get("num_experts", 0))
        if not 0 < experts <= 4096 or int(text_config.get("num_experts_per_tok", 0)) <= 0:
            return False
        if mlp + "gate.weight" not in names:
            return False
        packed = {mlp + "experts.gate_up_proj", mlp + "experts.down_proj"} <= names
        if not (packed or _has_mlp(names, mlp + "experts.") or all(
            _has_mlp(names, mlp + f"experts.{expert}.") for expert in range(experts)
        )):
            return False
        if (int(text_config.get("shared_expert_intermediate_size", 0)) > 0
                or int(text_config.get("n_shared_experts", 0)) > 0):
            if not _has_mlp(names, mlp + "shared_expert."):
                return False
    return True


def detect_mtp_support(model_path, model_config):
    """Return a UI reason code; only architectures accepted by util.py opt in."""
    path = Path(model_path)
    try:
        is_gguf = path.is_file() and path.suffix.lower() == ".gguf"
        if is_gguf:
            model_config = get_gguf_model_config(str(path))
        elif not path.is_dir():
            return "unverified"
        text_config = model_config.get("text_config", model_config)
        architectures = model_config.get("architectures", [])
        model_types = {model_config.get("model_type"), text_config.get("model_type")}
        if not (set(architectures) & {
            "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
        } or model_types & {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}):
            return "unsupported_architecture" if model_config else "unverified"
        if is_gguf:
            architecture = model_config["gguf_architecture"]
            fields = {"mtp_num_hidden_layers": "nextn_predict_layers",
                      "num_experts": "expert_count", "num_experts_per_tok": "expert_used_count",
                      "n_shared_experts": "expert_shared_count",
                      "shared_expert_intermediate_size": "expert_shared_feed_forward_length"}
            metadata = read_gguf_metadata(path, {architecture + "." + key for key in fields.values()})
            text_config = {key: metadata.get(architecture + "." + suffix, 0)
                           for key, suffix in fields.items()}
        layers = int(text_config.get("mtp_num_hidden_layers", 0))
        if layers <= 0:
            return "no_mtp"
        if layers > 1024:
            return "unverified"
        if is_gguf:
            names = _gguf_mtp_names(path, architecture, layers)
        else:
            names = _hf_mtp_names(path)
        return "enabled" if _has_mtp_weights(names, text_config, layers) else "missing_weights"
    except FileNotFoundError:
        return "missing_weights"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, struct.error, OverflowError):
        return "unverified"
