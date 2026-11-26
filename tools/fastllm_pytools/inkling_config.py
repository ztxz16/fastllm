"""Capacity planning for Inkling's rank-local CUDA experts and single-GPU host cache."""
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import hybrid_auto as memory

MiB, GiB = memory.MiB, memory.GiB


@dataclass
class Geometry:
    hidden: int
    inter: int
    experts: int
    top: int
    heads: int
    kv: int
    dim: int
    context: int
    window: int
    local: int
    global_: int
    relative_extent: int
    residual_block: int
    vocab: int
    conv: int


def geometry(meta):
    gg = meta.get("general.architecture") == "inkling2_loop"
    text = meta if gg else meta["text_config"]

    def field(native, key):
        return text["inkling2_loop." + key] if gg else text[native]

    def heads(native, key):
        value = field(native, key)
        if isinstance(value, list):
            if len(value) != layers or len(set(value)) != 1:
                raise ValueError("Inkling --auto requires uniform attention head counts")
            return value[0]
        return value

    layers = field("num_hidden_layers", "block_count")
    if gg:
        start, end, passes = [text["inkling2_loop.loop." + k] for k in ("start", "end", "count")]
        pattern = field("", "attention.sliding_window_pattern")
    else:
        loop = text["loop_config"]
        start, end, passes = (loop[k] for k in ("core_start", "core_end", "num_passes"))
        pattern = [i in text["local_layer_ids"] for i in range(layers)]
    if not (0 <= start < end <= layers and passes >= 1 and len(pattern) == layers):
        raise ValueError("Invalid Inkling loop or sliding attention geometry")
    local = sum(pattern) + (passes - 1) * sum(pattern[start:end])
    visits = layers + (passes - 1) * (end - start)
    return Geometry(field("hidden_size", "embedding_length"), field("intermediate_size", "expert_feed_forward_length"),
                    field("n_routed_experts", "expert_count"), field("num_experts_per_tok", "expert_used_count"),
                    heads("num_attention_heads", "attention.head_count"), heads("num_key_value_heads", "attention.head_count_kv"),
                    field("head_dim", "attention.key_length"), field("model_max_length", "context_length"),
                    field("sliding_window_size", "attention.sliding_window"), local, visits - local,
                    field("rel_extent", "attention.relative_extent"), field("n_layers_per_attn_res_block", "attn_res.block_size"),
                    field("unpadded_vocab_size", "vocab_size_unpadded"), field("sconv_kernel_size", "shortconv.l_cache"))


@dataclass
class ExpertStorage:
    cuda_experts: int = 0
    compact_fp4_savings: int = 0
    repack_source: int = 0
    slot_bytes: int = 0


@dataclass
class Weights(ExpertStorage):
    replicated: int = 0
    sharded: int = 0
    largest: int = 0
    native_layers: dict = field(default_factory=dict)


def weight_sizes(tensors, g):
    w = Weights()
    for t in tensors:
        name, n = t.name, math.prod(t.shape)
        w.largest = max(w.largest, t.bytes)
        if ".mlp.experts." in name or "_exps." in name:
            if len(t.shape) != 3 or not name.endswith(("w13_weight", "w2_weight", "_exps.weight")):
                continue  # The record sizes below already include scales and outer factors.
            if t.shape[0] != g.experts:
                raise ValueError("Inkling expert count disagrees with " + name)
            savings = 0
            if t.kind == "U8":
                host = 3 * t.bytes // (2 * g.experts)
                gpu = host * g.experts
                # The compact GPU record uses one E4M3 byte per 8 packed code bytes.
                if g.hidden % 128 == 0 and g.inter % 128 == 0 and g.top <= 16:
                    savings = gpu - (t.bytes + t.bytes // 8)
                    w.compact_fp4_savings += savings
                gpu += g.experts * (8 + 2 * 4)  # Pointers and outer scales.
            elif t.kind == "F8_E4M3":
                host = (n + n // 32 * 4) // g.experts
                # Grouped packing releases the source. Budget the larger indexed
                # fallback with one FP32 scale per block and its optional bias.
                gpu = host * g.experts + g.experts * (t.shape[1] * 4 + 3 * 8)
            elif t.kind.startswith("gguf:"):
                if t.kind in ("gguf:39", "gguf:40"):  # MXFP4 / NVFP4 -> 12 bytes per 16 elements on host.
                    host = n // (16 * g.experts) * 12
                else:
                    raw = math.ceil(t.bytes / g.experts)
                    host = ((raw + 3) // 4 * 4 + 4 + 255) // 256 * 256
                gpu = t.bytes + 2 * g.experts * 8
            else:
                raise ValueError("Unsupported Inkling expert encoding " + t.kind)
            if t.kind in ("U8", "F8_E4M3"):
                layer = w.native_layers.setdefault(name.rsplit(".", 1)[0], ExpertStorage())
                layer.cuda_experts += gpu
                layer.slot_bytes += host
                layer.compact_fp4_savings += savings
                # Indexed FP8 scales are host vectors during repack. FP4
                # repacking takes the compact 9-byte source blocks as input.
                source = t.bytes if t.kind == "F8_E4M3" else t.bytes + t.bytes // 8
                layer.repack_source += source
            w.slot_bytes += host
            w.cuda_experts += gpu
            continue
        # GetTensorMap casts projections to BF16/FP16, embeddings and small parameters to FP32.
        embedding = name in ("model.llm.embed.weight", "token_embd.weight")
        output = name in ("model.llm.unembed.weight", "output.weight")
        attn = name.endswith((".attn.wq_du.weight", ".attn.wr_du.weight", ".attn.wk_dv.weight",
                              ".attn.wv_dv.weight", ".attn.wo_ud.weight", ".attn.gate_attn.weight",
                              ".attn.attn_sink_proj.weight", ".attn.k_sconv.weight_proj", ".attn.v_sconv.weight_proj",
                              ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_rel.weight",
                              ".attn_gate.weight", ".attn_output.weight", ".attn_sink_proj.weight",
                              ".attn_k_conv_proj.weight", ".attn_v_conv_proj.weight"))
        visual = name.startswith("model.visual.") and name.endswith(".weight") and ".ln_" not in name
        size = n * (2 if output or attn or visual else 4)
        if embedding or output or attn:
            w.sharded += size
        else:
            w.replicated += size
    w.repack_source = max((layer.repack_source for layer in w.native_layers.values()), default=0)
    if not w.slot_bytes or not w.sharded:
        raise ValueError("Inkling --auto needs expert matrices and backbone projections")
    return w


def _recurrent_experts(meta, weights):
    text = meta.get("text_config", {})
    loop = text.get("loop_config", {})
    if loop.get("num_passes", 0) <= 1:
        return None
    start, end = loop["core_start"], loop["core_end"]
    layers = text["num_hidden_layers"]
    names = [f"model.llm.layers.{i}.mlp.experts" for i in range(layers)]
    if end - start == layers or not all(name in weights.native_layers for name in names):
        return None
    core = [weights.native_layers[name] for name in names[start:end]]
    storage = ExpertStorage(cuda_experts=sum(w.cuda_experts for w in core),
                            compact_fp4_savings=sum(w.compact_fp4_savings for w in core),
                            repack_source=max(w.repack_source for w in core),
                            slot_bytes=sum(w.slot_bytes for w in core))
    return start, end, layers, storage


def _recurrent_device_map(start, end, layers, host, gpu):
    # SelectDeviceFromMap traverses keys in lexical order. Use two consecutive
    # maps to represent host / GPU / host for either spelling of the host device.
    if host < gpu:
        front = {host: start, gpu: end - start} if start else gpu
        return dict(front=front, tail=host, tail_layers=layers - end)
    tail = {gpu: end - start, host: layers - end} if end < layers else gpu
    return dict(front=host, tail=tail, tail_layers=layers - start)


def _device_id(spec, info):
    match = re.fullmatch(r"cuda(?::(\d+))?", spec)
    if not match or int(match[1] or 0) not in info:
        raise ValueError("Inkling --auto needs a visible CUDA backbone, got " + str(spec))
    return int(match[1] or 0)


def _layouts(tp, root, info, budgets, g, draft):
    from .util import _normalize_thread_tp_arg, _uses_thread_tp
    eligible = sorted(info, key=lambda d: (-budgets[d], d))
    divisors = [g.kv, g.heads, g.experts]
    if draft:
        divisors.extend(draft[k] for k in ("num_attention_heads", "num_key_value_heads", "intermediate_size"))

    layouts = []
    if _uses_thread_tp(tp) and str(tp).lower() not in ("true", "on", "auto"):
        if str(tp).isdigit() and int(tp) > len(info):
            raise ValueError("Inkling TP requests more GPUs than are visible")
        spec = _normalize_thread_tp_arg(tp)
        if not re.fullmatch(r"cuda:\d+(?:,\d+)*", spec):
            raise ValueError("Inkling --auto requires equal TP ranks")
        ids = tuple(int(v) for v in spec[5:].split(","))
        if (len(ids) != len(set(ids)) or not set(ids) <= info.keys() or root not in (None, ids[0])
                or any(v % len(ids) for v in divisors) or len({info[d]["capability"] >= 80 for d in ids}) != 1):
            raise ValueError("Invalid Inkling TP device list or indivisible attention/expert groups")
        layouts.append(ids)
    else:
        if not tp or _uses_thread_tp(tp):
            for bf16 in {card["capability"] >= 80 for card in info.values()}:
                ranks = [d for d in eligible if (info[d]["capability"] >= 80) == bf16]
                if root is not None:
                    if root not in ranks or not _uses_thread_tp(tp):
                        continue
                    ranks = [root] + [d for d in ranks if d != root]
                layouts.extend(tuple(ranks[:n]) for n in range(2, len(ranks) + 1) if all(v % n == 0 for v in divisors))
        layouts.sort(key=lambda ids: (-len(ids), -min(budgets[d] for d in ids), ids))
        layouts.extend((d,) for d in eligible if root in (None, d))
    return layouts


def _kv_bytes(capacity, batch, page, heads, dim, unit, global_layers, local_layers, window, extra=0):
    # The draft reserves its uncommitted block on each request's global visits.
    capacity += batch * ((extra + page - 1) // page) * page
    local_tokens = min(capacity, batch * ((window + page - 2) // page + 1) * page)
    return 2 * unit * heads * dim * (global_layers * capacity + local_layers * local_tokens)


def _workspace_bytes(g, rows, decode_rows, ranks, hybrid, draft_taps, steps=0):
    # target() keeps the live projections and one residual per block.
    residuals = (g.global_ + g.local) // g.residual_block
    backbone = rows * g.hidden * 4 * (12 + residuals + draft_taps)
    relative_bias = 4 * min(rows, 512) * (g.heads // ranks) * g.relative_extent
    route_bytes = 4 * g.top * (g.hidden + 3 * g.inter)
    moe = rows * route_bytes
    if hybrid:
        # Eager host routing uses bounded chunks. Capture also retains its decode intermediates.
        moe = min(moe, 64 * MiB) + decode_rows * route_bytes
    workspace = backbone + relative_bias + moe + 4 * decode_rows * g.vocab + 512 * MiB
    if steps:
        batch = decode_rows // (steps + 1)
        # Both sampling workspaces (two FP32 matrices each), proposal q, and
        # the verifier's convolution histories. Graph shapes reuse the largest buffers.
        workspace += 4 * g.vocab * (2 * (batch + decode_rows) + batch * steps)
        workspace += 8 * batch * (g.local + g.global_) * (g.conv + steps) * (g.hidden + g.kv * g.dim // ranks)
    return workspace


def _draft_budget(args, meta):
    if not args.dspark:
        return {}, 0
    if args.speculative_draft_model_path:
        md, tensors = memory.inventory(args.speculative_draft_model_path)
        if (meta.get("general.architecture") != "inkling2_loop" or md.get("general.architecture") != "dflash"
                or not any(t.name == "joint_proj.weight" for t in tensors)):
            raise ValueError("Inkling --auto requires a matching GGUF target and DSpark draft")
        pattern = md["dflash.attention.sliding_window_pattern"]
        draft = {key: md["dflash." + field] for key, field in (
            ("num_hidden_layers", "block_count"), ("num_attention_heads", "attention.head_count"),
            ("num_key_value_heads", "attention.head_count_kv"), ("head_dim", "attention.key_length"),
            ("intermediate_size", "feed_forward_length"), ("block_size", "block_size"),
            ("target_layer_ids", "target_layers"))}
        draft["local_layer_ids"] = [i for i, local in enumerate(pattern) if local]
        if "dflash.context_length" in md:
            draft["max_position_embeddings"] = md["dflash.context_length"]
        draft["sliding_window_size"] = md["dflash.attention.sliding_window"] - 1
        if (len(pattern) != draft["num_hidden_layers"] or any(type(local) is not bool for local in pattern)
                or draft["head_dim"] != md["dflash.attention.value_length"] or draft["sliding_window_size"] < 1
                or not 0 < args.dspark < draft["block_size"]):
            raise ValueError("Invalid Inkling GGUF DSpark geometry or proposal count")
        # LoadDraftGGUF expands projection matrices to the dense dtype and vectors to FP32.
        size = sum(math.prod(t.shape) * (2 if len(t.shape) == 2 and t.name != "mask_embd.weight" else 4)
                   for t in tensors)
        return draft, size
    path = Path(args.path) / "mtp.safetensors"
    if not meta.get("mtp_config") or not path.is_file():
        raise ValueError("Inkling --auto requires the native mtp.safetensors or an explicit matching GGUF draft")
    return meta["mtp_config"], 2 * path.stat().st_size


def resolve(args, query_gpu, hardware=None, ram=None, use_numa=None):
    from .auto_config import cpu_threads, cuda_hardware, cuda_graph_enabled
    from .env import env
    from .util import _arg_enabled, _memory_size_bytes

    # A planned token count takes precedence over the runtime's byte limit.
    if args.kv_cache_limit not in ("", "auto"):
        raise ValueError("Inkling --auto sizes KV with --tokens and --max_context_length; "
                         "use manual placement for --kv_cache_limit")
    explicit = set(getattr(args, "_auto_explicit", ()))
    meta, tensors = memory.inventory(args.path)
    g = geometry(meta)
    w = weight_sizes(tensors, g)
    draft, draft_bytes = _draft_budget(args, meta)
    w.replicated += draft_bytes  # Conservatively budget the draft, including the folded head, on every rank.
    info, _ = cuda_hardware() if hardware is None else hardware
    if not info or not 0 < args.gpu_mem_ratio <= 1:
        raise ValueError("Inkling --auto needs visible CUDA devices and a GPU memory ratio in (0,1]")
    host_limit, _, _ = memory.host_budget(args, memory.host_memory() if ram is None else ram)
    limit = memory._memory_limit_bytes(args.auto_gpu_memory_limit, "auto_gpu_memory_limit")
    budgets = {d: memory.gpu_budget(query_gpu(d), args.gpu_mem_ratio, limit) for d in info}

    compact = set()
    if w.compact_fp4_savings:
        if hardware is None:
            from .llm import cuda_supports_compact_nvfp4_moe
            compact = {d for d in info if cuda_supports_compact_nvfp4_moe(d)}
        else:
            compact = {d for d, card in info.items() if card.get("compact_nvfp4_moe", False)}

    root = _device_id(args.device, info) if "device" in explicit else None
    numa = env.use_numas if use_numa is None else use_numa
    host = "numa" if numa else "cpu"
    moe = args.moe_device if "moe_device" in explicit else ""
    cache = None if "moe_cuda_cache" not in explicit or args.moe_cuda_cache == "auto" else _memory_size_bytes(args.moe_cuda_cache)
    force_host = moe in ("cpu", "numa", "numas") or bool(cache)
    if moe in ("cpu", "numa", "numas"):
        host = "numa" if moe == "numas" else moe
        if host == "numa" and not numa:
            raise ValueError("Inkling NUMA experts require a NUMA build")
    elif moe:
        requested = _device_id(moe, info)
        if force_host or root not in (None, requested):
            raise ValueError("Inkling expert placement conflicts with the backbone or expert cache")
        root = requested
    if args.moe_cpu_cache not in (0, "auto") or args.moe_device_layers not in (-1, 0, "auto"):
        raise ValueError("Inkling --auto requires automatic expert layer placement or --moe_device_layers 0, "
                         "and does not support a separate CPU expert cache")
    force_gpu = bool(moe and not force_host) or args.moe_device_layers == 0
    tp = args.tp or os.environ.get("FASTLLM_TP", "")
    layouts = _layouts(tp, root, info, budgets, g, draft)
    if len(layouts) == 1 and len(layouts[0]) > 1:
        force_gpu = True
    if force_host and force_gpu:
        raise ValueError("Inkling cannot combine tensor-parallel CUDA experts with a host expert placement")

    batch = max(1, args.max_batch)
    context = args.max_context_length if args.max_context_length > 0 else min(g.context, args.tokens if args.tokens > 0 else 131072)
    kvmem = json.loads(args.kvmem) if args.kvmem is not None else {}
    if not isinstance(kvmem, dict):
        raise ValueError("--kvmem must be a JSON object")
    offload = args.kvmem is not None and kvmem.get("enabled", True)
    if offload:
        if "max_tokens" not in kvmem:
            kvmem["max_tokens"] = context if args.max_context_length > 0 else min(context, 32768)
            args.kvmem = json.dumps(kvmem)
        logical = kvmem["max_tokens"]
        if type(logical) is not int or logical <= 0 or logical > g.context or (args.max_context_length > 0 and logical > context):
            raise ValueError("KVMem max_tokens exceeds the configured Inkling context")
        context = logical
    if context > g.context or (args.tokens > 0 and args.tokens < context):
        raise ValueError("Inkling context exceeds the checkpoint or shared token budget")
    contexts = [context]
    if not offload and args.tokens <= 0 and args.max_context_length <= 0:
        while contexts[-1] > 8192:
            contexts.append(max(8192, contexts[-1] // 2))
    page = args.page_size if args.page_size > 0 else 128
    unit = {"auto": 2, "float16": 2, "bfloat16": 2, "float32": 4,
            "fp8": 1, "float8": 1, "fp8_e4m3": 1}.get(args.kv_cache_dtype)
    if not unit:
        raise ValueError("Inkling KV requires float16, bfloat16, float32 or fp8_e4m3")
    graph = cuda_graph_enabled()
    dense = w.replicated + w.sharded
    head_width = 64 if g.dim <= 64 else 128
    decode_rows = batch * (args.dspark + 1)
    draft_taps = len(draft.get("target_layer_ids", ()))
    host_kv = 0
    if offload:
        hot = kvmem.get("resident_tokens", 8192)
        if type(hot) is not int or not 0 < hot <= logical or hot % page:
            raise ValueError("KVMem resident_tokens must fit max_tokens and align to a KV page")
        retained = kvmem.get("sliding_host_tokens", 0)
        if type(retained) is not int or not 0 <= retained <= 2147483647:
            raise ValueError("KVMem sliding_host_tokens must be a nonnegative 32-bit integer")
        def aligned(n):
            return (n + page - 1) // page * page

        def host_slots(tokens, num_global, num_local, window):
            provisional = draft["block_size"] - 1 if draft else 0
            sliding = min(tokens, aligned(max(window, retained) + provisional + page - 1))
            return num_global * tokens + num_local * sliding

        target_tokens = aligned(logical)
        pages = target_tokens // page
        target_visits = g.local + g.global_
        target_pair = 2 * unit * g.kv * head_width
        host_kv = batch * host_slots(target_tokens, g.global_, g.local, g.window) * target_pair
        target_slots = g.global_ * hot + g.local * min(hot, aligned(g.window))
        hot_kv = batch * target_slots * target_pair
        tables = batch * target_visits * pages * 8
        if draft:
            draft_layers = draft["num_hidden_layers"]
            draft_pair = 4 * draft["num_key_value_heads"] * draft["head_dim"]
            draft_tokens = aligned(min(logical, draft.get("max_position_embeddings", logical)) + draft["block_size"] - 1)
            draft_hot = min(hot, draft_tokens)
            local_layers = len(draft["local_layer_ids"])
            draft_host_slots = host_slots(draft_tokens, draft_layers - local_layers, local_layers,
                                          draft["sliding_window_size"])
            host_kv += batch * draft_host_slots * draft_pair
            draft_slots = ((draft_layers - local_layers) * draft_hot
                           + local_layers * min(draft_hot, aligned(draft["sliding_window_size"])))
            hot_kv += batch * draft_slots * draft_pair
            tables += batch * draft_layers * (draft_tokens // page) * 8
        if host_kv > kvmem.get("host_mib", 8192) * MiB:
            raise ValueError(f"KVMem needs {math.ceil(host_kv / MiB)} MiB of host KV for max_batch={batch}")
        if "prefill_tokens" in kvmem:
            if type(kvmem["prefill_tokens"]) is not int or kvmem["prefill_tokens"] < 1:
                raise ValueError("KVMem prefill_tokens must be a positive integer")
            args.chunked_prefill_size = kvmem["prefill_tokens"]
    prefix_flag = args.prefix_cache or os.environ.get("FASTLLM_PREFIX_CACHE") or "true"
    prefix_enabled = (not offload and _arg_enabled(args.cache_history or "true") and
                      prefix_flag.lower() in ("true", "on", "yes", "1"))
    # Match the rank-local lazy BufferPool ceiling, including in simulated VRAM.
    prefix_budget = 256 * MiB * max(4, min(16, batch)) if prefix_enabled else 0
    fixed_chunk = args.chunked_prefill_size not in (-1, "auto")
    chunks = [args.chunked_prefill_size] if fixed_chunk else [8192, 4096, 2048, 1024, 512, 256]
    if fixed_chunk and chunks[0] < 1:
        raise ValueError("Inkling prefill chunk must be positive")
    recurrent = None
    if not ({"moe_device", "moe_device_layers", "moe_cuda_cache"} & explicit):
        recurrent = _recurrent_experts(meta, w)
    failures = []
    for context in contexts:
        tokens = args.tokens if args.tokens > 0 else context * batch
        if offload:
            kv = hot_kv
        else:
            capacity = (tokens + page - 1) // page * page
            extra = draft["block_size"] - 1 if draft else 0
            kv = _kv_bytes(capacity, batch, page, g.kv, head_width, unit, g.global_, g.local, g.window + extra)
            if draft:
                local = len(draft["local_layer_ids"])
                kv += _kv_bytes(capacity, batch, page, draft["num_key_value_heads"], draft["head_dim"], 2,
                                draft["num_hidden_layers"] - local, local, draft["sliding_window_size"] + draft["block_size"], extra)
        for hybrid in (False, True):
            if (hybrid and force_gpu) or (not hybrid and force_host):
                continue
            host_bytes = host_kv + dense + 2 * w.largest + (w.slot_bytes * g.experts if hybrid else 0)
            for ids in layouts:
                world = len(ids)
                if hybrid and world > 1:
                    continue
                expert_total = w.cuda_experts - (w.compact_fp4_savings if compact.issuperset(ids) else 0)
                expert_gpu = 0 if hybrid else math.ceil(expert_total / world)
                resident = w.replicated + math.ceil(w.sharded / world) + expert_gpu
                # Rank zero can still own the loaded dense checkpoint while shards are created.
                source = max(w.largest, 0 if hybrid else w.repack_source)
                load_peak = resident + (dense if world > 1 else 0) + math.ceil(source / world) + 512 * MiB
                rank_kv = (kv + world - 1) // world
                if offload:
                    width = g.kv * head_width // world
                    selection_bytes = batch * g.global_ * (152 + 4 * pages * (width + g.heads // world + 2)
                                                           + 16 * (hot // page))
                    rank_kv += tables + selection_bytes
                rank_prefix = prefix_budget // world
                rank_budget = min(budgets[d] for d in ids)
                # Prefer putting the repeated core on CUDA when it fits with
                # both prefill workspaces and staging for the remaining host layers.
                tiers = [None]
                if hybrid and recurrent and ids[0] in compact:
                    tiers.append(recurrent[3])
                candidates = []
                for core in tiers:
                    slot_bytes = w.slot_bytes - (core.slot_bytes if core else 0)
                    gpu_core = core.cuda_experts - core.compact_fp4_savings if core else 0
                    host_used = host_bytes - (w.slot_bytes - slot_bytes) * g.experts
                    if host_used > host_limit:
                        failures.append(f"host needs {host_used // MiB} MiB, budget {host_limit // MiB} MiB")
                        continue
                    peak = load_peak + gpu_core
                    if core:
                        peak += max(0, core.repack_source - source)
                    for chunk in chunks:
                        if not fixed_chunk:
                            chunk = min(chunk, context)
                        rows = max(chunk, args.prefill_batch_tokens or 0, decode_rows)
                        scratch = _workspace_bytes(g, rows, decode_rows, world, hybrid, draft_taps, args.dspark)
                        if core:
                            scratch = max(scratch, _workspace_bytes(g, rows, decode_rows, world, False, draft_taps, args.dspark))
                        runtime = resident + gpu_core + rank_kv + rank_prefix + scratch
                        spare = min(rank_budget - runtime, budgets[ids[0]] - peak)
                        slots = min(g.experts, max(0, spare) // slot_bytes) if hybrid else 0
                        cache_bytes = (slots * slot_bytes if cache is None else cache) if hybrid else 0
                        if spare < cache_bytes or (core and slots < g.top):
                            failures.append(f"{'hybrid' if hybrid else 'CUDA'} {ids} needs {max(runtime, peak) // MiB} MiB before cache")
                            continue
                        chunk_rank, cache_rank = memory.interactive_moe_priority(chunk, cache_bytes)
                        priority = (chunk_rank, bool(core), cache_rank) if hybrid else ()
                        candidates.append((priority, chunk, cache_bytes, scratch, gpu_core, peak, host_used, slot_bytes))
                if not candidates:
                    continue
                _, chunk, cache_bytes, scratch, gpu_core, load_peak, host_used, slot_bytes = max(candidates)
                args.device = f"cuda:{ids[0]}"
                args.tp = "cuda:" + ",".join(map(str, ids)) if world > 1 else "off"
                args.moe_device = host if hybrid else args.device
                args.moe_device_layers = -1 if hybrid else 0
                args.moe_cuda_cache, args.moe_cpu_cache = cache_bytes, 0
                args.max_batch, args.tokens, args.max_context_length = batch, tokens, context
                args.chunked_prefill_size = chunk
                if args.threads <= 0:
                    args.threads = cpu_threads()
                result = dict(model="inkling2_loop", mode="hybrid" if hybrid else "tp" if world > 1 else "cuda",
                              device=args.device, tp=args.tp, moe_device=args.moe_device, max_batch=batch,
                              tokens=tokens, max_context_length=context, chunked_prefill_size=chunk,
                              moe_cuda_cache=cache_bytes, cache_slots_per_layer=min(g.experts, cache_bytes // slot_bytes),
                              gpu_budget_bytes={d: budgets[d] for d in ids}, kv_bytes_per_rank=rank_kv,
                              prefix_cache_bytes_per_rank=rank_prefix, workspace_bytes_per_rank=scratch,
                              weight_bytes_per_rank=resident + gpu_core, root_load_peak_bytes=load_peak,
                              host_estimated_bytes=host_used, host_budget_bytes=host_limit,
                              automatic={"policy": "inkling-tp-or-single-hybrid", "cuda_graph": graph})
                if gpu_core:
                    start, end, layers = recurrent[:3]
                    result.update(gpu_expert_layers=list(range(start, end)),
                                  recurrent_expert_placement=_recurrent_device_map(start, end, layers, host, args.device))
                if offload:
                    result.update(kvmem_host_bytes=host_kv, kvmem_resident_tokens=hot,
                                  kvmem_hot_bytes_per_rank=hot_kv // world, kvmem_page_table_bytes_per_rank=tables,
                                  kvmem_selection_bytes_per_rank=selection_bytes)
                print("[auto] " + json.dumps(result, sort_keys=True), flush=True)
                return result
    raise ValueError("Inkling --auto: no feasible layout. " + "; ".join(dict.fromkeys(failures[-3:])))
