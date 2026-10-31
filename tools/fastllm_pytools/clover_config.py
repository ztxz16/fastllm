"""Single-GPU native Clover placement and generation configuration."""
import json
import math
import os
import re


def workspace_reserve(config, batch, width):
    """Small batches retain MoE graphs; larger batches reuse eager scratch."""
    graph_batch = min(batch, 32)  # CloverModel::MoeGraphs::MaxBatch
    extra_rows = (batch - graph_batch) * width
    # Two FP32 logits sets and the 16-bit head output may overlap. Include
    # projections, routing and expert scratch; prefill runs one request at a time.
    per_row = (12 * int(config["vocab_size"]) + 32 * int(config["hidden_size"]) +
               4 * int(config["num_experts"]) + 4 * int(config["num_experts_per_tok"]) *
               (int(config["hidden_size"]) + int(config["moe_intermediate_size"])))
    return (graph_batch + 2) * (1 << 30) + extra_rows * per_row


def prepare(args, config):
    """Reject settings that the native Clover implementation cannot honor."""
    if "enable_thinking" not in getattr(args, "_auto_explicit", ()):
        args.enable_thinking = "true"
    precision_keys = ("dtype", "atype", "kv_cache_dtype", "moe_dtype", "moe_atype")
    requested = set()
    for key in precision_keys:
        value = getattr(args, key, "")
        if value not in ("", "auto", "bf16", "bfloat16", "fp16", "float16"):
            raise ValueError("Clover requires BF16 or FP16: --" + key)
        if value not in ("", "auto"):
            requested.add("float16" if value in ("fp16", "float16") else "bfloat16")
    if len(requested) > 1:
        raise ValueError("Clover weights, activations and attention KV must use the same precision")
    precision = next(iter(requested), "bfloat16")
    # The native model also enforces FP16 below SM80 for direct C++ callers.
    if not getattr(args, "auto", False):
        from .auto_config import cuda_hardware
        try:
            info, _ = cuda_hardware()
        except ValueError:
            info = {}  # Native initialization reports missing CUDA support.
        match = re.fullmatch(r"cuda(?::(\d+))?", args.device or "cuda")
        if match and 0 < info.get(int(match[1] or 0), {}).get("capability", 0) < 80:
            precision = "float16"
    for key in precision_keys:
        setattr(args, key, precision)
    if (args.tp or args.mtp or args.dspark or args.speculative_draft_model_path or
            args.lora or args.dtype_config or args.mmproj or args.kvmem):
        raise ValueError("Clover requires one GPU, original weights and no external draft/adapter/KVMem")
    if args.speculative_algorithm not in ("", "clover"):
        raise ValueError("Clover supports --speculative_algorithm clover or off")
    if args.low_gpu_mem or args.moe_experts > 0 or args.prefill_batch_tokens > 0:
        raise ValueError("Clover requires resident weights, the checkpoint expert count and serial prefill")
    if args.max_batch > 64:
        raise ValueError("Clover currently supports --max_batch up to 64")
    args.max_batch = max(1, args.max_batch)
    if args.max_batch > 1 and "speculative_algorithm" in getattr(args, "_auto_explicit", ()) and not args.speculative_algorithm:
        raise ValueError("Clover batch generation requires self_speculative mode")
    if args.rope_scaling:
        raise ValueError("Clover currently uses the checkpoint's native RoPE")
    native_context = int(config["max_position_embeddings"])
    if args.max_context_length > native_context:
        raise ValueError("Clover context exceeds the checkpoint's native window")
    width = args.speculative_num_draft_tokens
    if args.max_batch > 1 and width != -1 and width < 16:
        raise ValueError("Clover batch generation requires block_size 16 or 32")
    if width != -1 and (width <= 0 or int(config["block_size"]) % width):
        raise ValueError("Clover block width must divide the checkpoint block_size")


def resolve(args, query_gpu, config, *, hardware=None, ram=None):
    from . import hybrid_auto as memory
    from .auto_config import cuda_hardware
    explicit = set(getattr(args, "_auto_explicit", ()))
    prepare(args, config)
    _, tensors = memory.inventory(args.path)
    if any(t.kind != "BF16" for t in tensors):
        raise ValueError("Clover --auto currently supports the original BF16 checkpoint")
    weights = sum(t.bytes * (2 if ("norm.weight" in t.name or ".A_log" in t.name or
                  ".dt_bias" in t.name or ".bd_softmax_" in t.name) else 1) for t in tensors)
    info, _ = hardware if hardware is not None else cuda_hardware()
    devices = list(info)
    if args.device:
        match = re.fullmatch(r"cuda(?::(\d+))?", args.device)
        if not match:
            raise ValueError("Clover requires --device cuda:N with exactly one GPU")
        devices = [int(match[1] or 0)]
    if any(d not in info for d in devices):
        raise ValueError("Clover selected CUDA ordinal is not visible")
    ratio = float(args.gpu_mem_ratio)
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("gpu_mem_ratio must be in (0, 1]")
    context = args.max_context_length if args.max_context_length > 0 else int(config["max_position_embeddings"])
    if args.tokens > 0:
        if "max_context_length" in explicit and args.tokens < context:
            raise ValueError("Clover --tokens cannot be below --max_context_length")
        context = min(context, args.tokens)
    types = config["layer_types"]
    linear = types.count("linear_attention")
    attention = types.count("full_attention")
    channels = (2 * int(config["linear_num_key_heads"]) + int(config["linear_num_value_heads"])) * 128
    state = linear * (4 * int(config["linear_num_value_heads"]) * 128 * 128 + 2 * channels * 3)
    kv_per_token = attention * 2 * int(config["num_key_value_heads"]) * int(config["head_dim"]) * 2
    speculative = args.speculative_algorithm == "clover" or "speculative_algorithm" not in explicit
    width = args.speculative_num_draft_tokens
    if width <= 0:
        width = 16 if speculative else int(config["block_size"])
    # Native weights, clean recurrent state, accepted-prefix snapshots, full
    # KV capacity, and scratch/allocator reserve. This is a capacity estimate,
    # not a throughput prediction. Every active request owns independent caches.
    kv = math.ceil(context / 256) * 256 * kv_per_token
    reserve = workspace_reserve(config, args.max_batch, width)
    # Keep the full-snapshot bound: mixed widths or a shrinking active batch
    # can select the original layout even when compact records usually fit.
    snapshots = state * width if speculative else 0
    state *= args.max_batch
    snapshots *= args.max_batch
    kv *= args.max_batch
    required = weights + state + snapshots + kv + reserve
    limit = memory._memory_limit_bytes(getattr(args, "auto_gpu_memory_limit", None), "auto_gpu_memory_limit")
    choices = []
    for d in devices:
        available = query_gpu(d)
        budget = min(available["free"] - 512 * memory.MiB, int(available["total"] * ratio))
        if limit is not None:
            budget = min(budget, limit)
        if budget >= required:
            choices.append((budget, d))
    if not choices:
        raise ValueError("Clover needs one CUDA GPU with at least %.2f GiB available under the memory budget" %
                         (required / memory.GiB))
    _, device = max(choices)
    if info[device].get("capability", 0) < 80:
        for key in ("dtype", "atype", "kv_cache_dtype", "moe_dtype", "moe_atype"):
            setattr(args, key, "float16")
    device = "cuda:%d" % device
    if args.moe_device and args.moe_device != device:
        raise ValueError("Clover experts must reside on the same GPU as the main model")
    host_limit = memory._memory_limit_bytes(getattr(args, "auto_host_memory_limit", None), "auto_host_memory_limit")
    available_host = (ram if ram is not None else memory.host_memory())["available"]
    if host_limit is not None:
        available_host = min(available_host, host_limit)
    prefix_flag = args.prefix_cache or os.environ.get("FASTLLM_PREFIX_CACHE") or "true"
    prefix_enabled = str(prefix_flag).lower() in ("1", "true", "yes", "on")
    records = args.prefix_cache_snapshot_max_records
    if records <= 0:
        try:
            records = max(1, int(os.environ.get("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS", "8")))
        except ValueError:
            records = 8
    # Conservative bound: immutable KV chunks are shared, but distinct cached
    # conversations and active requests can each retain a full context.
    prefix_host = (records + args.max_batch) * (kv // args.max_batch + state // args.max_batch +
                   4 * int(config.get("vocab_size", 248320))) if prefix_enabled else 0
    if available_host < weights + prefix_host + 2 * memory.GiB:
        raise ValueError("Insufficient host RAM for Clover weights and prefix-cache budget")
    args.device = args.moe_device = device
    args.max_context_length = args.tokens = context
    args.cuda_embedding = True
    if args.chunked_prefill_size in ("auto", -1, 0):
        args.chunked_prefill_size = 4096
    for key in ("moe_cuda_cache", "moe_cpu_cache", "moe_device_layers"):
        if key in explicit and getattr(args, key) not in (0, "0", "", "auto"):
            raise ValueError("Clover keeps all experts resident; --%s is unsupported" % key)
        setattr(args, key, 0)
    if args.threads <= 0:
        args.threads = 4
    if "speculative_algorithm" not in explicit:
        args.speculative_algorithm = "clover"
    args.speculative_num_draft_tokens = width
    result = dict(model="clover_moe", mode="single_gpu", device=device, max_batch=args.max_batch,
                  dtype=args.atype, max_context_length=context, weights_bytes=weights,
                  recurrent_state_bytes=state, snapshot_bytes=snapshots,
                  kv_bytes=kv, workspace_reserve_bytes=reserve, estimated_gpu_bytes=required,
                  prefix_cache_host_bytes=prefix_host,
                  speculative_algorithm=args.speculative_algorithm, block_size=width,
                  automatic={"policy": "single GPU; BF16 or legacy FP16; native context", "cuda_graph": False,
                             "moe_cuda_graph": os.environ.get("FASTLLM_CUDA_GRAPH", "").lower() not in ("0", "false", "off", "no")})
    print("[auto] " + json.dumps(result, sort_keys=True), flush=True)
    return result
