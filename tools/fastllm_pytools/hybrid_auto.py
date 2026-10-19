"""Pre-load memory planning for hybrid MoE inference.

This module resolves opt-in CLI values into the existing explicit configuration.
It reads tensor headers, never weight payloads. Estimates include a conservative
workspace allowance; they are capacity plans, not throughput autotuning.
"""

import ast
import copy
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import struct

from . import gguf_metadata as gguf

GiB = 1 << 30
MiB = 1 << 20
AUTO_OPTIONS = ("moe_cuda_cache", "moe_cpu_cache", "chunked_prefill_size", "moe_device_layers")


def requested(args):
    return any(getattr(args, name, None) == "auto" for name in AUTO_OPTIONS)


def _memory_limit_bytes(value, label):
    """Return an optional explicit planner cap in bytes."""
    if value is None or str(value).strip().lower() in ("", "auto"):
        return None
    try:
        from .util import _memory_size_bytes
        result = _memory_size_bytes(value)
    except Exception as error:
        raise ValueError("Hybrid auto: %s is invalid" % label) from error
    if result <= 0:
        raise ValueError("Hybrid auto: %s must be positive" % label)
    return result


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    bytes: int
    kind: str


def _gguf_header(path):
    metadata, tensors = {}, []
    with open(path, "rb") as stream:
        if gguf._read_exact(stream, 4) != b"GGUF" or gguf._read_number(stream, "I") not in (2, 3):
            raise ValueError("Hybrid auto: invalid GGUF header")
        count, pairs = gguf._read_number(stream, "Q"), gguf._read_number(stream, "Q")
        for _ in range(pairs):
            key = gguf._read_string(stream)
            keep = not key.startswith(("tokenizer.", "quantize.", "general.tags"))
            value = gguf._read_value(stream, gguf._read_number(stream, "I"), keep)
            if keep:
                metadata[key] = value
        entries = []
        for _ in range(count):
            name = gguf._read_string(stream)
            ndim = gguf._read_number(stream, "I")
            if not 1 <= ndim <= 4:
                raise ValueError("Hybrid auto: invalid GGUF tensor rank")
            shape = tuple(gguf._read_number(stream, "Q") for _ in range(ndim))
            kind, offset = gguf._read_number(stream, "I"), gguf._read_number(stream, "Q")
            if not all(shape):
                raise ValueError("Hybrid auto: empty tensor")
            entries.append((offset, name, shape, kind))
        alignment = int(metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Hybrid auto: invalid GGUF alignment")
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        data_size = os.fstat(stream.fileno()).st_size - data_start
        entries.sort()
        for i, (offset, name, shape, kind) in enumerate(entries):
            end = entries[i + 1][0] if i + 1 < len(entries) else data_size
            if not 0 <= offset < end <= data_size:
                raise ValueError("Hybrid auto: invalid GGUF tensor extent")
            # Include alignment padding rather than duplicating ggml's type table.
            tensors.append(Tensor(name, shape[::-1], end - offset, "gguf:%d" % kind))
    return metadata, tensors


def inventory(path):
    """Read a complete local checkpoint; validate shards before budgeting."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".gguf":
        metadata, first = _gguf_header(path)
        count = int(metadata.get("split.count", 1))
        if count == 1:
            if len({t.name for t in first}) != len(first):
                raise ValueError("Hybrid auto: duplicate tensor names")
            return metadata, first
        match = re.fullmatch(r"(.*)-\d{5}-of-(\d{5})\.gguf", path.name)
        if not match or int(match[2]) != count:
            raise ValueError("Hybrid auto: cannot resolve GGUF shards")
        tensors = []
        for i in range(count):
            shard = path.with_name("%s-%05d-of-%05d.gguf" % (match[1], i + 1, count))
            md, part = (metadata, first) if shard == path else _gguf_header(shard)
            if int(md.get("split.no", -1)) != i or int(md.get("split.count", 0)) != count:
                raise ValueError("Hybrid auto: inconsistent GGUF shards")
            tensors.extend(part)
        if len(tensors) != int(metadata.get("split.tensors.count", len(tensors))):
            raise ValueError("Hybrid auto: incomplete GGUF tensor directory")
    else:
        with (path / "config.json").open() as stream:
            metadata = json.load(stream)
        index = path / "model.safetensors.index.json"
        if index.exists():
            with index.open() as stream:
                weight_map = json.load(stream)["weight_map"]
            files = sorted(set(weight_map.values()))
        else:
            files = ["model.safetensors"]
            weight_map = None
        tensors = []
        for name in files:
            file = path / name
            if file.resolve().parent != path.resolve():
                raise ValueError("Hybrid auto: invalid safetensors shard path")
            with file.open("rb") as stream:
                size = struct.unpack("<Q", gguf._read_exact(stream, 8))[0]
                if not 0 < size <= min(128 * MiB, file.stat().st_size - 8):
                    raise ValueError("Hybrid auto: invalid safetensors header")
                header = json.loads(gguf._read_exact(stream, size))
            for key, item in header.items():
                if key == "__metadata__":
                    continue
                start, end = item["data_offsets"]
                if not 0 <= start < end <= file.stat().st_size - size - 8:
                    raise ValueError("Hybrid auto: invalid safetensors tensor extent")
                if weight_map is not None and weight_map.get(key) != name:
                    raise ValueError("Hybrid auto: safetensors index/header mismatch")
                tensors.append(Tensor(key, tuple(item["shape"]), end - start, item["dtype"]))
        if weight_map is not None and set(weight_map) != {t.name for t in tensors}:
            raise ValueError("Hybrid auto: incomplete safetensors checkpoint")
    if len({t.name for t in tensors}) != len(tensors):
        raise ValueError("Hybrid auto: duplicate tensor names")
    return metadata, tensors


def host_memory(proc=Path("/proc")):
    """Effective Linux RAM, including every visible cgroup ancestor limit.

    Do not count swap or assume that charged file pages can be reclaimed. A
    parent's current usage includes siblings, which can constrain this process.
    """
    info = {}
    for line in (proc / "meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0]) * 1024
    total, available = info["MemTotal"], info["MemAvailable"]
    groups = [line.split(":", 2) for line in (proc / "self/cgroup").read_text().splitlines()]
    mounts = (proc / "self/mountinfo").read_text().splitlines()
    for line in mounts:
        before, after = line.split(" - ", 1)
        fields, fs = before.split(), after.split()
        if fs[0] not in ("cgroup", "cgroup2"):
            continue
        v2 = fs[0] == "cgroup2"
        if not v2 and "memory" not in fs[2].split(","):
            continue
        group = next((g[2] for g in groups if (g[0] == "0" and v2) or
                      (not v2 and "memory" in g[1].split(","))), None)
        if group is None:
            continue
        unescape = lambda s: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)
        root, mount = Path(unescape(fields[3])), Path(unescape(fields[4]))
        try:
            relative = Path(group).relative_to(root)
        except ValueError:
            # A cgroup namespace exposes membership as / relative to the mount.
            if group != "/":
                raise ValueError("Hybrid auto: cannot locate memory cgroup")
            relative = Path(".")
        if ".." in relative.parts:
            raise ValueError("Hybrid auto: invalid cgroup membership path")
        current = mount / relative
        while True:
            limit_file = current / ("memory.max" if v2 else "memory.limit_in_bytes")
            usage_file = current / ("memory.current" if v2 else "memory.usage_in_bytes")
            if limit_file.exists():
                raw = limit_file.read_text().strip()
                if raw != "max":
                    limit = int(raw)
                    usage = int(usage_file.read_text())
                    total, available = min(total, limit), min(available, max(0, limit - usage))
            if current == mount:
                break
            current = current.parent
    return {"total": total, "available": available}


@dataclass
class Geometry:
    family: str
    layers: int
    context: int
    hidden: int
    experts: int
    topk: int
    intermediate: int
    kv: list                 # bytes/token/layer; conservatively replicated under TP
    state: list              # recurrent or fixed-window state bytes/layer/request
    score_divisor: int
    activation_width: int
    default_chunk: int
    streamed_kv: list = field(default_factory=list)  # exact host-backed KV bytes/token/layer, excluding index
    dspark_block_size: int = 0  # draft forward retains the trained block when verifying fewer tokens
    attention_score_heads: int = 1  # dense GQA fallback may materialize one score matrix per query head


def geometry(metadata, kv_dtype, mtp, dspark=0):
    """Model adapters expose costs, while placement and budgeting stay shared."""
    arch = metadata.get("general.architecture", "")
    config = metadata.get("text_config", metadata)
    family = str(arch or config.get("model_type", ""))
    if family in ("qwen4exp", "qwen4_exp", "qwen4_exp_text", "qwen3_8_flash_next", "qwen3_8_flash_next_text"):
        family = "qwen4"
    elif family in ("glm5next", "glm5-next", "glm5_next", "glm5_next_text"):
        family = "glm5"
    elif family == "qwen3":
        if mtp:
            raise ValueError("Auto: dense Qwen3 does not have embedded MTP")
    elif family in ("deepseek_v41", "deepseek_v41_text"):
        family = "deepseek_v41"
        config = dict(config)
        for primary, alias in (("num_hidden_layers", "n_layers"), ("hidden_size", "dim"),
                               ("max_position_embeddings", "max_seq_len"), ("num_attention_heads", "n_heads"),
                               ("n_routed_experts", "num_experts"), ("moe_intermediate_size", "moe_inter_dim")):
            if primary not in config and alias in config:
                config[primary] = config[alias]
    else:
        raise ValueError("Auto supports dense Qwen3, Qwen4, GLM-5.3 and DeepSeek-V4.1")
    if dspark and family != "deepseek_v41":
        raise ValueError("Hybrid auto supports embedded DSpark only for DeepSeek-V4.1")
    def get(hf, gg, default=None):
        value = metadata.get(arch + "." + gg, default) if arch else config.get(hf, default)
        if value is None:
            raise ValueError("Hybrid auto: missing model dimension " + (gg if arch else hf))
        return value
    layers = int(get("num_hidden_layers", "block_count"))
    if family == "glm5" and arch:
        layers -= int(get("num_nextn_predict_layers", "nextn_predict_layers", 0))
    hidden = int(get("hidden_size", "embedding_length"))
    context = int(get("max_position_embeddings", "context_length"))
    if family == "qwen3":
        heads = int(get("num_attention_heads", "attention.head_count"))
        kv_heads = int(get("num_key_value_heads", "attention.head_count_kv", heads))
        inter = int(get("intermediate_size", "feed_forward_length"))
        if min(layers, hidden, context, heads, kv_heads, inter) <= 0 or heads % kv_heads:
            raise ValueError("Auto: invalid Qwen3 dimensions")
        dim = int(get("head_dim", "attention.key_length", hidden // heads))
        unit = {"auto": 2, "float16": 2, "bfloat16": 2, "float32": 4}.get(kv_dtype)
        if dim <= 0 or unit is None:
            raise ValueError("Auto: unsupported Qwen3 head dimension or KV precision")
        # One always-active FFN reuses the shared activation/latency budget;
        # actual tensor classification keeps all dense weights on CUDA.
        return Geometry(family, layers, context, hidden, 1, 1, inter,
                        [kv_heads * dim * 2 * unit] * layers, [0] * layers,
                        1, hidden * 8 + heads * dim * 6, 4096,
                        attention_score_heads=heads)
    experts = int(get("n_routed_experts" if family != "qwen4" else "num_experts", "expert_count"))
    topk = int(get("num_experts_per_tok", "expert_used_count"))
    inter = int(get("moe_intermediate_size", "expert_feed_forward_length"))
    if min(layers, hidden, context, experts, topk, inter) <= 0:
        raise ValueError("Hybrid auto: non-positive model dimensions")
    unit = {"auto": 2, "float16": 2, "bfloat16": 2, "float32": 4, "fp8_e4m3": 1}.get(kv_dtype)
    if unit is None and not (family == "deepseek_v41" and kv_dtype == "fp4"):
        raise ValueError("Hybrid auto: unsupported KV precision")
    kv, state = [0] * layers, [0] * layers
    streamed_kv = []
    dspark_block = 0
    if family == "qwen4":
        heads = int(get("num_key_value_heads", "attention.head_count_kv"))
        dim = int(get("head_dim", "attention.key_length"))
        interval = int(get("full_attention_interval", "full_attention_interval", 4))
        ratios = get("compress_ratios", "attention.compress_ratios", [0 if (i + 1) % interval else 4 for i in range(layers)])
        if not arch and "layer_types" in config:
            ratios = [0 if kind == "linear_attention" else int(config.get("indexer_compress_ratio", 4))
                      for kind in config["layer_types"]]
        if len(ratios) != layers or not any(r > 0 for r in ratios):
            raise ValueError("Hybrid auto: invalid Qwen attention layout")
        linear_dim = int(get("linear_key_head_dim", "ssm.state_size", 128))
        linear_heads = int(get("linear_num_value_heads", "ssm.time_step_rank", 48))
        linear_width = (int(metadata[arch + ".ssm.inner_size"]) if arch and arch + ".ssm.inner_size" in metadata
                        else int(config.get("linear_value_head_dim", 128)) * linear_heads)
        key_heads = int(get("linear_num_key_heads", "ssm.group_count", 16))
        kernel = int(get("linear_conv_kernel_dim", "ssm.conv_kernel", 4))
        idxdim = int(get("indexer_head_dim", "attention.indexer.key_length", 128))
        for i in range(layers):
            if ratios[i] > 0:
                kv[i] = heads * dim * 2 * unit + math.ceil(idxdim * 4 / ratios[i])
            else:
                state[i] = linear_heads * linear_dim * (linear_width // linear_heads) * 4 + max(3 * linear_width, linear_width + 2 * key_heads * linear_dim) * kernel * 4
        if mtp:
            kv.append(heads * dim * 2 * unit + math.ceil(idxdim * 4 / min(r for r in ratios if r > 0)))
            state.append(0)
        score_divisor = min(r for r in ratios if r > 0)
        # Only Qwen4 Hybrid has exact QSA streaming. Other KVMem adapters
        # change attention selection and must never be enabled by this policy.
        if (kv_dtype == "float16" and dim in (128, 256) and
                all(r in (0, 4) for r in ratios) and
                int(config.get("indexer_compress_ratio", 4)) == 4 and
                int(get("indexer_budget", "attention.indexer.top_k", 2048)) > 0):
            streamed_kv = [heads * dim * 4 if n else 0 for n in kv]
        width = hidden * int(get("hc_count", "hyper_connection.count", 4)) * 8 + linear_width * 12
        chunk = 4096
    elif family == "glm5":
        # Native GLM supports this published architecture and compressed MLA.
        if os.environ.get("FASTLLM_GLM5_NEXT_EXPANDED_DSA", "0") not in ("", "0", "false", "off"):
            raise ValueError("Hybrid auto requires GLM compressed MLA")
        if arch:
            linear = [i for i, n in enumerate(get("", "attention.head_count_kv")) if n == 0 and i < layers]
        else:
            linear = config["linear_attn_config"]["kda_layers"]
        rank = int(get("kv_lora_rank", "attention.kv_lora_rank"))
        for i in range(layers):
            if i in linear:
                state[i] = 64 * 128 * 128 * 4 + 3 * 64 * 128 * 4 * 4
            else:
                kv[i] = (rank + 64 + 128 // 4) * max(2, unit)
        if mtp:
            kv.append((rank + 64 + 128 // 4) * max(2, unit))
            state.append(0)
        score_divisor, width, chunk = 4, hidden * 4 * 8 + 64 * 128 * 12, 1024
    else:
        if mtp:
            raise ValueError("Hybrid auto: DeepSeek-V4.1 uses DSpark, not MTP")
        dim, idxdim = int(get("head_dim", "")), int(get("index_head_dim", ""))
        ratios = config["compress_ratios"]
        def row(dim, index=False):
            if kv_dtype == "fp4":
                return dim // 2 + dim // (32 if index else 16)
            if kv_dtype == "fp8_e4m3":
                return dim + dim // 32
            return dim * unit
        for i in config["kv_source_layer_ids"]:
            kv[i] += math.ceil(row(dim) / max(1, ratios[i]))
        for i in config["index_source_layer_ids"]:
            kv[i] += math.ceil(row(idxdim, True) / max(1, ratios[i]))
        state = [int(config["sliding_window"]) * dim * max(2, unit or 2)] * layers
        score_divisor = min((r for r in ratios if r > 0), default=1)
        width = hidden * int(config.get("hc_mult", 4)) * 8 + int(config["num_attention_heads"]) * dim * 6
        chunk = 4096
        if dspark:
            drafts = int(config.get("num_nextn_predict_layers", config.get("n_mtp_layers", 0)))
            trained = int(config.get("dspark_block_size", 0))
            targets = config.get("dspark_target_layer_ids", [])
            if (not 0 < drafts <= layers or trained <= 0 or dspark < 1 or
                    int(config.get("dspark_noise_token_id", -1)) < 0 or
                    int(config.get("dspark_markov_rank", 0)) <= 0 or
                    int(config.get("dspark_n_routed_experts", 0)) <= 0 or
                    int(config.get("dspark_num_experts_per_tok", 0)) <= 0 or
                    not targets or any(not 0 <= i < layers for i in targets) or
                    any(r != 0 for r in ratios[layers:layers + drafts])):
                raise ValueError("Hybrid auto: invalid embedded DeepSeek-V4.1 DSpark configuration")
            dspark_block = max(trained, dspark)
            # Draft layers have only BF16 sliding-window state, not full-context
            # KV. Dense draft work runs on the final PP device (replicated bound
            # for TP); experts follow the final N target-layer placements.
            kv.extend([0] * drafts)
            state.extend([int(config["sliding_window"]) * dim * 2] * drafts)
            # Preserve target hidden features and reserve proposal/verify logits.
            width += hidden * (len(targets) + 1) * 2
            state[-1] += (2 * dspark_block + 1) * int(get("vocab_size", "")) * 4
    return Geometry(family, layers, context, hidden, experts, topk, inter, kv, state,
                    score_divisor, width, chunk, streamed_kv, dspark_block)


def expert_layer(layer, model):
    """Match DSpark's ApplyMoeDeviceMapForLayer(main_layers - drafts + stage)."""
    if model.family == "deepseek_v41" and layer >= model.layers:
        return layer - (len(model.kv) - model.layers)
    return layer


def devices(args):
    """Return execution topology; CUDA ordinals are the process-visible IDs."""
    from .util import expand_cudapp_device, _thread_tp_cuda_device_spec, _thread_tp_cuda_device_ids, _uses_thread_tp
    tp = getattr(args, "tp", "") or os.environ.get("FASTLLM_TP", "")
    if _uses_thread_tp(tp):
        spec = _thread_tp_cuda_device_spec(tp)
        # Unequal TP splits need the loader's alignment rules; do not guess.
        if not re.fullmatch(r"cuda:\d+(?:,\d+)*", spec):
            raise ValueError("Hybrid auto currently requires equal TP shares")
        return "tp", {i: 1 for i in _thread_tp_cuda_device_ids(spec)}
    spec = expand_cudapp_device(str(args.device or ""))
    try:
        mapping = ast.literal_eval(spec)
    except (ValueError, SyntaxError):
        mapping = {spec: 1}
    if not isinstance(mapping, dict):
        raise ValueError("Hybrid auto requires a CUDA device or a weighted CUDA PP map")
    result = {}
    # Match std::map ordering used by SelectDeviceFromMap.
    for name, weight in sorted(mapping.items()):
        match = re.fullmatch(r"cuda(?::(\d+))?", name)
        if not match or not isinstance(weight, int) or weight <= 0:
            raise ValueError("Hybrid auto requires CUDA-only main devices")
        device = int(match[1] or 0)
        if device in result:
            raise ValueError("Hybrid auto: duplicate CUDA device")
        result[device] = weight
    if not result:
        raise ValueError("Hybrid auto: no CUDA devices")
    return "pp", result


def layer_device(layer, layers, mapping):
    cumulative, total = 0, sum(mapping.values())
    for device, weight in mapping.items():
        cumulative += weight
        if (layer + 1) * total <= cumulative * layers:
            return device
    return next(reversed(mapping))


def is_lookup_table(name):
    return ("ple_embd" in name or name == "per_layer_token_embd.weight" or "ple_embedding.weight" in name or
            "ngram_embedding" in name or ".engram.embed." in name or
            "engram" in name and "embedding" in name)


def _is_routed_expert(name):
    return "_exps." in name or bool(re.search(r"\.experts\.(?:\d+\.|(?:gate|up|down|gate_up)_proj)", name))


def _packed_fp4_sizes(tensors, args):
    """Packed payload plus runtime scales/padding, without reading weight data.

    Follow the loader's scale-name priority and recognize only validated native
    layouts. Unknown layouts and explicit precision conversions keep the existing
    conservative conversion budget in weight_costs.
    """
    by_name = {t.name: t for t in tensors}
    sizes = {}
    for weight in tensors:
        if weight.kind not in ("U8", "I8") or not weight.name.endswith(".weight") or len(weight.shape) < 2:
            continue
        expert = _is_routed_expert(weight.name)
        shared = "_shexp." in weight.name or ".shared_experts." in weight.name
        dtype = (args.moe_dtype or args.dtype) if expert or shared else args.dtype
        if dtype not in ("auto", "nvfp4"):
            continue
        prefix = weight.name[:-len(".weight")]
        candidates = (weight.name + "_scale_inv", weight.name + "_scale",
                      prefix + ".scale_inv", prefix + ".scale")
        scale = next((by_name[n] for n in candidates if n in by_name), None)
        if scale is None or scale.kind not in ("F8_E8M0", "U8", "F8_E4M3") or len(scale.shape) < 2:
            continue
        if (min(*weight.shape, *scale.shape) <= 0 or
                weight.bytes != math.prod(weight.shape) or scale.bytes != math.prod(scale.shape)):
            continue
        rows, columns = math.prod(weight.shape[:-1]), weight.shape[-1] * 2
        scale_rows, scale_columns = math.prod(scale.shape[:-1]), scale.shape[-1]
        if rows % scale_rows or columns % scale_columns:
            continue
        block_rows, block_columns = rows // scale_rows, columns // scale_columns
        if block_rows & (block_rows - 1) or block_columns & (block_columns - 1) or block_columns < 16:
            continue
        # Raw U8 scales are E8M0 only for the loader's exact MXFP4 shape.
        if scale.kind == "U8" and not (len(weight.shape) == len(scale.shape) == 2 and
                                       block_rows == 1 and block_columns == 32):
            continue
        blocks16 = (columns + 15) // 16
        if scale.kind == "F8_E4M3":
            if block_columns != 16:
                continue
            # E4M3 can use compact byte scales + an aligned row-global FP32
            # multiplier, or 4-byte block scales on CPU/older CUDA devices.
            # The same costs feed both host and GPU placement: cover both.
            compact_row = (4 + blocks16 * 9 + 3) // 4 * 4
            row_bytes = max(compact_row, blocks16 * 12)
        else:
            # Native block32 uses 17 bytes/32 values. Keep the 9-byte block16
            # fallback bound as CPU capabilities/overrides are not queried here.
            # It also covers scale replication when source rows share a scale.
            row_bytes = blocks16 * 9
        sizes[weight.name] = max(weight.bytes + scale.bytes, rows * row_bytes)
        sizes[scale.name] = 0  # Already included, even if the header lists it first.
    return sizes


def weight_costs(tensors, model, args):
    """Expert bytes by layer, replicated dense upper bounds and host tables."""
    expert = [0] * len(model.kv)
    dense = [0] * len(model.kv)
    common, host, largest = 0, 0, 0
    snapshot = False
    packed_sizes = _packed_fp4_sizes(tensors, args)
    for tensor in tensors:
        name = tensor.name
        match = re.search(r"(?:^blk\.|(?:^|\.)layers\.)(\d+)\.", name)
        layer = int(match[1]) if match else None
        is_draft = name.startswith(("mtp.", "nextn.")) or (layer is not None and layer >= model.layers)
        if is_draft:
            if model.family == "deepseek_v41":
                stage = re.match(r"^mtp\.(\d+)\.", name)
                if not model.dspark_block_size or not stage:
                    continue
                layer = model.layers + int(stage[1])
                if layer >= len(model.kv):
                    continue  # DsparkTensorNeeded ignores stages outside the checkpoint configuration.
            elif args.mtp:
                layer = model.layers
            else:
                continue
        if layer is not None and layer >= len(dense):
            raise ValueError("Hybrid auto: tensor layer outside model")
        is_expert = _is_routed_expert(name)
        is_shared = "_shexp." in name or ".shared_experts." in name
        size = tensor.bytes
        if is_expert and tensor.kind in ("gguf:13", "gguf:14"):
            # NUMA Q5_K/Q6_K R4 cannot currently be restored by the cache;
            # the entire cache group takes its established host snapshot.
            snapshot = True
        if is_expert and name.endswith(".weight") and not tensor.kind.startswith("gguf:"):
            target = args.moe_dtype or args.dtype
            dense_expert = target in ("float16", "bfloat16", "float32") or (
                target == "auto" and tensor.kind in ("F16", "BF16", "F32"))
            if args.moe_device != "disk" and args.moe_cuda_cache == "auto" and dense_expert:
                raise ValueError("Hybrid auto CUDA cache requires GGUF, FP8 or packed FP4 expert weights")
            if (args.moe_device != "disk" and args.moe_cuda_cache == "auto" and
                    model.family in ("glm5", "deepseek_v41") and
                    not ((tensor.kind in ("U8", "I8") and target in ("auto", "nvfp4")) or
                         (tensor.kind in ("F8_E4M3", "F8_E4M3FN") and target in ("auto", "fp8_e4m3")))):
                raise ValueError("Hybrid auto: this model's NUMA CUDA cache requires GGUF, native FP8 E4M3 or packed FP4 experts")
        if not tensor.kind.startswith("gguf:"):
            dtype = (args.moe_dtype or args.dtype) if is_expert or is_shared else args.dtype
            if dtype not in ("auto", "float16", "bfloat16", "float32", "fp8_e4m3", "nvfp4"):
                raise ValueError("Hybrid auto: unsupported HF target weight precision " + dtype)
            # Norms/scales retain source precision. Compressed FP8/NVFP4
            # sources and their scale tensors are budgeted from their headers.
            if name in packed_sizes:
                size = packed_sizes[name]
            elif tensor.kind in ("F16", "BF16", "F32"):
                size = max(size, math.prod(tensor.shape) * (4 if dtype == "float32" else 2))
            elif tensor.kind in ("U8", "I8"):
                # Unknown packed layouts retain the conservative bound. Explicit
                # conversions count two unpacked values per stored byte.
                size *= 8 if dtype == "float32" else 4 if dtype in ("float16", "bfloat16") else 2
            elif tensor.kind in ("F8_E4M3", "F8_E4M3FN", "F8_E5M2", "F8_E8M0"):
                if dtype in ("float16", "bfloat16", "float32") or (dtype == "auto" and not is_expert):
                    # Model adapters may force auxiliary FP8 projections to BF16.
                    size = max(size, math.prod(tensor.shape) * (4 if dtype == "float32" else 2))
            elif tensor.kind in ("I16", "I32", "I64", "U16", "U32", "U64", "BOOL") and not name.endswith(".weight"):
                # Auxiliary integer buffers include PLE hash multipliers and
                # vocabulary offsets; these are not quantized weight matrices.
                size = max(size, math.prod(tensor.shape) * 4)
            else:
                raise ValueError("Hybrid auto: unsupported HF storage " + tensor.kind)
        is_table = is_lookup_table(name)
        is_embedding = name in ("token_embd.weight", "embed.weight") or name.endswith("embed_tokens.weight")
        if is_table:
            if args.ngram_device != "disk":
                host += size
        elif is_embedding:
            # The Qwen GGUF adapter expands token embeddings to FP32.
            if model.family in ("qwen4", "qwen3") and tensor.kind.startswith("gguf:"):
                size = math.prod(tensor.shape) * 4
            host += size
            if model.family == "qwen3":
                # Cover implicit CUDA embedding and Prepare()'s output-head
                # copy for tied checkpoints, including the 0.6B model.
                common += size
                if not any(t.name in ("lm_head.weight", "output.weight") for t in tensors):
                    common += size
            elif args.cuda_embedding and not args.low_gpu_mem:
                common += size
        elif is_shared and str(args.cuda_shared_expert).lower() in ("", "false", "0", "off"):
            host += size
        elif is_expert:
            if layer is None:
                raise ValueError("Hybrid auto: cannot place expert " + name)
            expert[layer] += size
        elif layer is None:
            common += size
        else:
            dense[layer] += size
        if not is_table:
            largest = max(largest, size)
    if not any(expert) and model.family != "qwen3":
        raise ValueError("Hybrid auto: no routed experts found")
    return expert, dense, common, host, largest, snapshot


def plan(args, model, costs, mode, mapping, gpu, ram, *, rank=None, allow_full_gpu=False,
         kv_resident=0, per_device_cache=False, implicit_embedding_bytes=0,
         hybrid_cpu_embedding=False):
    """Deterministic capacity solver. Explicit fields are never overwritten."""
    experts, dense, common, host, largest, snapshot = costs
    context = int(args.max_context_length) if args.max_context_length > 0 else model.context
    tokens = int(args.tokens) if args.tokens > 0 else context
    if context > model.context or tokens <= 0:
        raise ValueError("Hybrid auto requires a context within the checkpoint's native window")
    context = min(context, tokens)
    batch = max(1, int(args.max_batch))
    page = args.page_size if args.page_size > 0 else 128
    tokens = math.ceil(tokens / page) * page
    streaming = bool(kv_resident)
    kv_host_bytes = 0
    if streaming:
        if (model.family != "qwen4" or not model.streamed_kv or args.atype != "float16" or
                args.kv_cache_dtype != "float16" or batch != 1 or args.tokens != context or
                not 0 < kv_resident < context or kv_resident % 4):
            raise ValueError("Hybrid auto: unsupported exact KV streaming configuration")
        # Match Qwen4's validation: include a possible draft layer even with
        # MTP off, plus provisional tokens and page alignment. The compressed
        # QSA index remains on the GPU for the entire logical context.
        logical = (context + args.mtp + 4) // 4 * 4
        layer_bytes = max(model.streamed_kv)
        target_bytes = sum(model.streamed_kv[:model.layers])
        kv_host_bytes = math.ceil(logical * (target_bytes + layer_bytes) / MiB) * MiB
        page_metadata = ((3 * (logical // 4) + 2 * (kv_resident // 4) + 3) & ~1) * 4 + 16
    ratio = float(args.gpu_mem_ratio)
    if not 0 < ratio <= 1:
        raise ValueError("Hybrid auto: gpu_mem_ratio must be in (0, 1]")
    gpu_limit = {d: gpu[d]["free"] - math.ceil(gpu[d]["total"] * (1 - ratio)) for d in mapping}
    gpu_memory_limit = _memory_limit_bytes(
        getattr(args, "auto_gpu_memory_limit", None), "auto_gpu_memory_limit"
    )
    if gpu_memory_limit is not None:
        gpu_limit = {d: min(limit, gpu_memory_limit) for d, limit in gpu_limit.items()}
    # The library queries free memory after creating the CUDA contexts. Keep
    # additional space for graphs, allocator granularity and library handles.
    guard = 512 * MiB
    host_limit = ram["available"] - max(GiB, ram["total"] // 10)
    host_memory_limit = _memory_limit_bytes(
        getattr(args, "auto_host_memory_limit", None), "auto_host_memory_limit"
    )
    if host_memory_limit is not None:
        host_limit = min(host_limit, host_memory_limit)
    chunks = ([4096, 2048, 1024, 512, 256] if args.chunked_prefill_size == "auto" else
              [args.chunked_prefill_size if args.chunked_prefill_size > 0 else model.default_chunk])
    chunks = list(dict.fromkeys(min(c, context) for c in chunks))
    if args.moe_device_layers == "auto":
        # Only the unified policy may choose full GPU residency.
        residents = ([model.layers] if model.family == "qwen3" else
                     range(model.layers if allow_full_gpu else model.layers - 1, -1, -1))
    else:
        host_layers = args.moe_device_layers
        if host_layers > model.layers:
            raise ValueError("Hybrid auto: moe_device_layers exceeds model depth")
        residents = [0 if host_layers < 0 else model.layers - host_layers]
    cache_auto = args.moe_cuda_cache == "auto"
    cpu_auto = args.moe_cpu_cache == "auto"
    topk = max(model.topk, args.moe_experts)
    selected, selected_score = None, None
    speculative = max(args.mtp, getattr(args, "dspark", 0))
    for chunk in chunks:
        rows = batch * max(chunk + (1 if args.mtp else 0), speculative + 1, model.dspark_block_size)
        # QSA/DSA score matrix, recurrent scan activations, routed expert
        # intermediates and one dense dequantization working set.
        workspace = (rows * math.ceil(context / model.score_divisor) * 4 * model.attention_score_heads +
                     rows * (model.activation_width + topk * (model.hidden + 3 * model.intermediate)) * 4 +
                     model.hidden * model.intermediate * 3 * 2)
        if streaming:
            # Prefill stages one complete KV layer per device; the buffer is
            # reused across layers. Do not count it as bounded by resident KV.
            workspace += logical * layer_bytes
        for resident in residents:
            used = {d: common + workspace + guard for d in mapping}
            if implicit_embedding_bytes and not (hybrid_cpu_embedding and resident < model.layers):
                # Qwen PP embeds on its first stage; TP and a full-GPU MTP
                # pipeline may also need the shared table on other devices.
                targets = list(mapping) if mode == "tp" or args.mtp else [next(iter(mapping))]
                for d in targets:
                    used[d] += implicit_embedding_bytes
            host_experts = 0
            host_by_device = dict.fromkeys(mapping, 0)
            for layer in range(len(experts)):
                targets = list(mapping) if mode == "tp" else [layer_device(layer, model.layers, mapping)]
                # Dense TP replication is an upper bound, including unsharded
                # norms and attention state; expert matrices are column/row split.
                owner = expert_layer(layer, model)
                resident_bytes = experts[layer] if owner < resident else 0
                if layer >= model.layers and args.mtp and model.family == "qwen4":
                    resident_bytes = experts[layer]  # Qwen draft experts stay on CUDA
                host_experts += experts[layer] - resident_bytes
                for d in targets:
                    host_by_device[d] += experts[layer] - resident_bytes
                    kv_bytes = (tokens + (speculative if model.dspark_block_size else 0)) * model.kv[layer]
                    if streaming and model.streamed_kv[layer]:
                        raw = model.streamed_kv[layer]
                        kv_bytes = logical * (model.kv[layer] - raw) + kv_resident * raw + page_metadata
                    used[d] += dense[layer] + kv_bytes + batch * (speculative + 1) * model.state[layer]
                if resident_bytes:
                    expert_targets = list(mapping) if mode == "tp" else [
                        layer_device(owner, resident, mapping) if owner < model.layers else next(reversed(mapping))]
                    for d in expert_targets:
                        used[d] += math.ceil(resident_bytes / len(expert_targets))
            if host_experts == 0 and (not allow_full_gpu or streaming):
                continue
            remaining = {d: gpu_limit[d] - used[d] for d in mapping}
            independent = per_device_cache and cache_auto and mode == "pp" and args.moe_device != "disk"
            cache = {}
            for d in mapping:
                ceiling = remaining[d] if independent else min(remaining.values())
                available_experts = host_by_device[d] if independent else host_experts
                budget = max(0, min(ceiling, available_experts)) // MiB * MiB if cache_auto else args.moe_cuda_cache
                if cache_auto and args.moe_device != "disk":
                    # The existing host cache requires at least 16 whole records.
                    record = math.ceil(max(experts[resident:model.layers], default=0) / model.experts)
                    if budget < 16 * (record + 256):
                        budget = 0
                cache[d] = budget
            if any(remaining[d] < cache[d] for d in mapping):
                continue
            # The scalar ceiling also enables registration; allocation uses the
            # individual budgets. Explicit caches and TP retain uniform budgets.
            cuda_cache = max(cache.values())
            # One streamed load group, host input/output staging and tokenizer.
            host_required = host + max(largest, max(experts, default=0)) + GiB + workspace
            if args.moe_device != "disk":
                host_required += host_experts
                pinned = os.environ.get("FT_PINNED_WEIGHT", "1").lower() not in ("0", "false", "off")
                if snapshot or not pinned or (cuda_cache > 0 and args.moe_device == "cpu"):
                    host_required += host_experts
            if streaming:
                # Full pinned KV replaces the normal request mirror. Retain
                # the index allowance; prefix reuse is disabled by KVMem.
                host_required += kv_host_bytes + logical * (sum(model.kv) - sum(model.streamed_kv))
            else:
                host_required += tokens * sum(model.kv)  # request mirrors / saved prefix
            cpu_cache = 0 if args.moe_device != "disk" else (
                max(0, min(host_experts, host_limit - host_required)) if cpu_auto else args.moe_cpu_cache)
            cpu_cache = cpu_cache // MiB * MiB if cpu_auto else cpu_cache
            if host_required + cpu_cache > host_limit:
                continue
            candidate = {"family": model.family, "mode": mode, "context_tokens": context,
                    "total_tokens": tokens, "max_batch": batch, "chunked_prefill_size": chunk,
                    "resident_layers": resident, "resident_expert_layers": sum(x > 0 for x in experts[:resident]), "moe_device_layers": model.layers - resident if resident else -1,
                    "moe_cuda_cache": cuda_cache, "moe_cpu_cache": cpu_cache,
                    "gpu_estimated_bytes": {d: used[d] + cache[d] for d in mapping},
                    "gpu_budget_bytes": gpu_limit, "host_estimated_bytes": host_required + cpu_cache,
                    "host_budget_bytes": host_limit, "workspace_bytes_per_device": workspace,
                    "auto_gpu_memory_limit": gpu_memory_limit,
                    "auto_host_memory_limit": host_memory_limit}
            if independent and host_experts:
                candidate["moe_cuda_cache_per_device"] = cache
            if streaming:
                candidate["kvmem"] = {"max_tokens": context, "resident_tokens": kv_resident,
                                      "host_mib": kv_host_bytes // MiB, "prefill_tokens": chunk}
            if rank is not None:
                score = rank(candidate)
            else:
                # Prefer useful expert storage, rather than layer count:
                # uneven layers can lower the common per-card cache ceiling.
                resident_bytes = sum(experts) - host_experts
                coverage = resident_bytes + min(host_experts, sum(cache.values()))
                score = (coverage // (64 * MiB), resident_bytes, -resident)
            if selected_score is None or score > selected_score:
                selected, selected_score = candidate, score
        if selected is not None and rank is None:
            return selected
    if selected is not None:
        return selected
    raise ValueError("Hybrid auto: requested context, batch, explicit caches and weights do not fit the measured memory budget; "
                     "reduce an explicit cache/resident allocation, use disk experts, or choose smaller context/chunk explicitly")


def validate_options(args):
    if str(args.moe_device).lower() not in ("cpu", "numa", "disk") or os.environ.get("FASTLLM_EXECUTION_MODE", "auto") not in ("auto", "hybrid"):
        raise ValueError("Hybrid auto requires CUDA + --moe_device cpu/numa/disk in auto/hybrid execution mode")
    if (args.kvmem or args.kv_cache_limit not in ("", "auto") or args.dtype_config or
            args.custom or args.lora or args.multimodal or args.rope_scaling or
            getattr(args, "speculative_algorithm", "") not in ("", "off", "mtp", "dspark") or
            (getattr(args, "speculative_algorithm", "") == "dspark" and
             (getattr(args, "speculative_draft_model_path", "") or args.dspark <= 0))):
        raise ValueError("Hybrid auto does not yet budget KVMem, custom KV limits, weight overrides, adapters, vision or external speculative algorithms; use explicit parameters")
    if any(k.startswith("FASTLLM_MOE_CUDA_CACHE_BYTES_") for k in os.environ):
        raise ValueError("Hybrid auto cannot combine with per-device cache environment overrides; use explicit cache parameters")


def model_inputs(args):
    metadata, tensors = inventory(args.path)
    kv_dtype = args.atype if args.kv_cache_dtype == "auto" and args.atype != "auto" else args.kv_cache_dtype
    model = geometry(metadata, kv_dtype, args.mtp, args.dspark)
    draft = getattr(args, "speculative_draft_model_path", "")
    if draft:
        _, draft_tensors = inventory(draft)
        tensors += [Tensor(t.name if t.name.startswith("mtp.") else "mtp." + t.name,
                           t.shape, t.bytes, t.kind) for t in draft_tensors]
    return metadata, tensors, model


def prepare(args, metadata, tensors, model, mode, mapping, *, costs=None):
    """Validate backend capabilities and normalize budgets without mutating args."""
    from .util import _memory_size_bytes
    planning_args = copy.copy(args)
    for name in ("moe_cuda_cache", "moe_cpu_cache"):
        if getattr(planning_args, name) != "auto":
            setattr(planning_args, name, _memory_size_bytes(getattr(planning_args, name)))
    if model.family in ("glm5", "deepseek_v41") and args.moe_device == "cpu" and args.moe_cuda_cache == "auto":
        raise ValueError("Hybrid auto: this model's CUDA expert cache requires --moe_device numa or disk")
    if model.family == "deepseek_v41" and mode == "tp" and len(mapping) > 1:
        config = metadata.get("text_config", metadata)
        heads = int(config.get("num_attention_heads", config.get("n_heads", 64)))
        groups, ranks = int(config.get("o_groups", 8)), len(mapping)
        if (heads <= 0 or groups <= 0 or heads % ranks or (heads // ranks) % 32 or
                heads % groups or (heads // ranks) % (heads // groups)):
            raise ValueError("Hybrid auto: DeepSeek attention would fall back from TP; choose an aligned TP degree or a single CUDA device")
    notes = []
    if (model.family == "qwen4" and "general.architecture" in metadata and
            args.moe_device == "disk" and args.moe_cuda_cache == "auto"):
        # The disk backend currently admits GGUF GPU records only for GLM.
        planning_args.moe_cuda_cache = 0
        notes.append("Qwen4 GGUF disk experts use the CPU cache; automatic CUDA expert cache is 0")
    if costs is None:
        costs = weight_costs(tensors, model, planning_args)
    return planning_args, costs, notes


def resolve(args, query_gpu):
    """Validate and calculate completely before modifying any CLI field."""
    if not requested(args):
        return None
    validate_options(args)
    mode, mapping = devices(args)
    metadata, tensors, model = model_inputs(args)
    planning_args, costs, notes = prepare(args, metadata, tensors, model, mode, mapping)
    gpu = {d: query_gpu(d) for d in mapping}
    result = plan(planning_args, model, costs, mode, mapping, gpu, host_memory())
    if notes:
        result["notes"] = notes
    for name in AUTO_OPTIONS:
        if getattr(args, name) == "auto":
            setattr(args, name, result[name])
    # Bound native automatic KV allocation by the capacity included in the plan.
    if args.tokens <= 0:
        args.tokens = result["total_tokens"]
    if args.max_batch <= 0:
        args.max_batch = result["max_batch"]
    print("[hybrid auto] " + json.dumps(result, sort_keys=True), flush=True)
    return result
