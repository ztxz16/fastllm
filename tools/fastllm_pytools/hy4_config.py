"""Plan the single-GPU, host-expert path for native HY4 Flash checkpoints."""
import json
import math
import os
import re

from . import hybrid_auto as memory


def resolve(args, query_gpu, hardware, ram, use_numa):
    from .auto_config import cuda_hardware
    from .env import env
    from .util import _memory_size_bytes, _uses_thread_tp

    explicit = getattr(args, "_auto_explicit", ())
    info, _ = hardware if hardware is not None else cuda_hardware()
    candidates = list(info)
    if "device" in explicit:
        match = re.fullmatch(r"cuda(?::(\d+))?", args.device)
        if not match or int(match[1] or 0) not in info:
            raise ValueError("HY4 --auto needs a visible CUDA device; use manual configuration for pipeline parallelism")
        candidates = [int(match[1] or 0)]
    numa = env.use_numas if use_numa is None else use_numa
    host = args.moe_device if "moe_device" in explicit else ("numa" if numa else "cpu")
    if host not in ("numa", "cpu") or _uses_thread_tp(args.tp or os.environ.get("FASTLLM_TP", "")):
        raise ValueError("HY4 --auto uses a CUDA backbone and CPU or NUMA experts")
    if args.moe_device_layers not in (-1, "auto") or args.moe_cpu_cache not in (0, "auto"):
        raise ValueError("HY4 --auto requires all routed experts in host memory")

    _, tensors = memory.inventory(args.path)
    expert_bytes = dense_bytes = lookup_bytes = 0
    for tensor in tensors:
        if memory._is_routed_expert(tensor.name):
            expert_bytes += tensor.bytes
        elif memory.is_lookup_table(tensor.name):
            lookup_bytes += tensor.bytes
        else:
            # Native GGUF expands dense projections to at least BF16.
            dense_bytes += max(tensor.bytes, 2 * math.prod(tensor.shape))
    host_budget, _, _ = memory.host_budget(args, ram if ram is not None else memory.host_memory())
    if host_budget < expert_bytes + lookup_bytes + dense_bytes:
        raise ValueError("HY4 host memory budget is too small for the checkpoint and host experts")
    batch = args.max_batch if args.max_batch > 0 else 1
    context = args.max_context_length if args.max_context_length > 0 else 8192
    slots = args.tokens if args.tokens > 0 else batch * context
    if slots < context:
        if args.max_context_length > 0:
            raise ValueError("HY4 --tokens cannot be below --max_context_length")
        context = slots
    # Bound each of the 42 layers by a full FP32 MLA latent, RoPE and index key.
    # Reserve scratch separately from the expert cache.
    reserve = dense_bytes + slots * 42 * (512 + 64 + 128) * 4 + 2 * memory.GiB
    limit = memory._memory_limit_bytes(args.auto_gpu_memory_limit, "auto_gpu_memory_limit")
    if not 0 < args.gpu_mem_ratio <= 1:
        raise ValueError("gpu_mem_ratio must be in (0, 1]")
    budgets = {}
    for device in candidates:
        gpu = query_gpu(device)
        budget = min(gpu["free"] - 512 * memory.MiB, int(gpu["total"] * args.gpu_mem_ratio))
        budgets[device] = min(budget, limit) if limit is not None else budget
    if not budgets or max(budgets.values()) < reserve:
        raise ValueError("HY4 --auto needs one GPU with enough memory for dense weights, KV and workspace")
    device = max(budgets, key=budgets.get)
    remaining = budgets[device] - reserve
    cache = min(expert_bytes, remaining) // memory.MiB * memory.MiB
    if "moe_cuda_cache" in explicit and args.moe_cuda_cache != "auto":
        cache = _memory_size_bytes(args.moe_cuda_cache)
    if cache > remaining:
        raise ValueError("HY4 --moe_cuda_cache exceeds the available GPU budget")
    args.device, args.moe_device = f"cuda:{device}", host
    args.moe_cuda_cache, args.moe_cpu_cache, args.moe_device_layers = cache, 0, -1
    args.max_batch, args.max_context_length, args.tokens = batch, context, slots
    if args.chunked_prefill_size == "auto":
        args.chunked_prefill_size = -1
    result = {"model": "hy_v4_flash", "device": args.device, "moe_device": host,
              "max_context_length": context, "moe_cuda_cache": cache,
              "gpu_budget_bytes": budgets[device], "host_budget_bytes": host_budget,
              "automatic": {"policy": "hy4-single-cuda-host-experts",
                            "cuda_graph": os.environ.get("FASTLLM_CUDA_GRAPH", "1") not in ("0", "")}}
    print("[auto] " + json.dumps(result, sort_keys=True), flush=True)
    return result
