"""Single-GPU native Clover placement and generation configuration."""
import json
import math
import os
import re


def prepare(args, config):
    """Reject settings that the native Clover implementation cannot honor."""
    if "enable_thinking" not in getattr(args, "_auto_explicit", ()):
        args.enable_thinking = "true"
    for key in ("dtype", "atype", "kv_cache_dtype", "moe_dtype", "moe_atype"):
        value = getattr(args, key, "")
        if value not in ("", "auto", "bf16", "bfloat16"):
            raise ValueError("Clover currently requires BF16: --" + key)
        setattr(args, key, "bfloat16")
    if (args.tp or args.mtp or args.dspark or args.speculative_draft_model_path or
            args.lora or args.dtype_config or args.mmproj or args.kvmem):
        raise ValueError("Clover requires one GPU, original BF16 weights and no external draft/adapter/KVMem")
    if args.speculative_algorithm not in ("", "clover"):
        raise ValueError("Clover supports --speculative_algorithm clover or off")
    if args.low_gpu_mem or args.moe_experts > 0 or args.prefill_batch_tokens > 0:
        raise ValueError("Clover requires resident weights, the checkpoint expert count and serial prefill")
    if any(str(getattr(args, key, "")).lower() not in ("", "0", "false", "off")
           for key in ("cache_history", "prefix_cache")):
        raise ValueError("Clover cross-request prefix caching is not implemented yet")
    if args.max_batch > 8:
        raise ValueError("Clover currently supports --max_batch up to 8")
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
    reserve = (args.max_batch + 2) * memory.GiB
    snapshots = state * width if speculative else 0
    state *= args.max_batch
    snapshots *= args.max_batch
    kv *= args.max_batch
    required = weights + state + snapshots + kv + reserve
    limit = memory._memory_limit_bytes(getattr(args, "auto_gpu_memory_limit", None), "auto_gpu_memory_limit")
    choices = []
    for d in devices:
        if info[d].get("capability", 0) < 80:
            continue
        available = query_gpu(d)
        budget = min(available["free"] - 512 * memory.MiB, int(available["total"] * ratio))
        if limit is not None:
            budget = min(budget, limit)
        if budget >= required:
            choices.append((budget, d))
    if not choices:
        raise ValueError("Clover needs one SM80+ GPU with at least %.2f GiB available under the memory budget" %
                         (required / memory.GiB))
    _, device = max(choices)
    device = "cuda:%d" % device
    if args.moe_device and args.moe_device != device:
        raise ValueError("Clover experts must reside on the same GPU as the main model")
    host_limit = memory._memory_limit_bytes(getattr(args, "auto_host_memory_limit", None), "auto_host_memory_limit")
    available_host = (ram if ram is not None else memory.host_memory())["available"]
    if host_limit is not None:
        available_host = min(available_host, host_limit)
    if available_host < weights + 2 * memory.GiB:
        raise ValueError("Insufficient host RAM to load original Clover weights")
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
                  dtype="bfloat16", max_context_length=context, weights_bytes=weights,
                  recurrent_state_bytes=state, snapshot_bytes=snapshots,
                  kv_bytes=kv, workspace_reserve_bytes=reserve, estimated_gpu_bytes=required,
                  speculative_algorithm=args.speculative_algorithm, block_size=width,
                  automatic={"policy": "single GPU; original BF16; native context", "cuda_graph": False,
                             "moe_cuda_graph": os.environ.get("FASTLLM_CUDA_GRAPH", "").lower() not in ("0", "false", "off", "no")})
    print("[auto] " + json.dumps(result, sort_keys=True), flush=True)
    return result
