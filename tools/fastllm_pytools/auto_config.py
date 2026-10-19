"""Opt-in startup policy for the shared hybrid capacity planner.

Scores rank configurations for interactive use. They are deliberately estimates,
not advertised throughput: routing locality and the real workload are unknown.
"""
import copy
import ctypes
import json
import math
import os
from pathlib import Path

from . import hybrid_auto as memory


def apply_model_defaults(args, metadata):
    """Apply model-specific auto defaults before speculative setup/budgeting."""
    config = metadata.get("text_config", metadata)
    if (not getattr(args, "auto", False) or
            config.get("model_type", metadata.get("model_type")) not in
            ("deepseek_v41", "deepseek_v41_text")):
        return
    explicit = set(getattr(args, "_auto_explicit", ()))
    speculation = {"mtp", "dspark", "speculative_algorithm", "speculative_draft_model_path",
                   "draft_tokens", "speculative_num_draft_tokens", "speculative_dspark_block_size"}
    if not explicit & speculation and not (args.mtp or args.dspark or args.speculative_algorithm or
                                           args.speculative_draft_model_path):
        args.dspark, args.speculative_algorithm = 7, "dspark"
    if "fast_prefill" not in explicit:
        args.fast_prefill = True


def cuda_hardware():
    """Read visible CUDA ordinals, bandwidth bounds and peer access capability."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise ValueError("--auto requires the Linux NVIDIA CUDA driver") from error
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
        "cuDeviceGetPCIBusId": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        "cuDeviceCanAccessPeer": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    }
    for name, signature in signatures.items():
        fn = getattr(driver, name)
        fn.argtypes, fn.restype = signature, ctypes.c_int
    count = ctypes.c_int()
    if driver.cuInit(0) or driver.cuDeviceGetCount(ctypes.byref(count)) or count.value < 1:
        raise ValueError("--auto requires a visible CUDA GPU")
    devices, handles = {}, {}
    for ordinal in range(count.value):
        handle = ctypes.c_int()
        if driver.cuDeviceGet(ctypes.byref(handle), ordinal):
            raise ValueError("--auto cannot inspect CUDA device %d" % ordinal)
        handles[ordinal] = handle.value
        def attr(number, fallback):
            value = ctypes.c_int()
            return value.value if driver.cuDeviceGetAttribute(ctypes.byref(value), number, handle.value) == 0 and value.value > 0 else fallback
        bus = ctypes.create_string_buffer(32)
        pci = ""
        if driver.cuDeviceGetPCIBusId(bus, len(bus), handle.value) == 0:
            pci = bus.value.decode("ascii").lower()
        # sysfs is keyed by PCI identity, never by nvidia-smi's device index.
        link = 12e9
        try:
            path = Path("/sys/bus/pci/devices") / pci
            # Idle GPUs downshift their current PCIe generation. Budget with
            # the endpoint's negotiated-width/max-speed bound, not idle speed.
            speed = float((path / "max_link_speed").read_text().split()[0])
            lanes = int((path / "current_link_width").read_text())
            if speed > 0 and lanes > 0:
                link = speed * 1e9 * lanes / 8 * .8
        except (OSError, ValueError):
            pass
        devices[ordinal] = {"sm": attr(16, 40), "clock_khz": attr(13, 1000000),
                            "memory_bandwidth": 2 * attr(36, 5000000) * 1000 * attr(37, 256) / 8,
                            "link_bandwidth": link, "capability": attr(75, 7) * 10 + attr(76, 0)}
    peers = {}
    for a in devices:
        for b in devices:
            value = ctypes.c_int()
            peers[a, b] = a == b or (driver.cuDeviceCanAccessPeer(ctypes.byref(value), handles[a], handles[b]) == 0 and bool(value.value))
    return devices, peers


def cpu_threads():
    """Honor native thread overrides, otherwise leave launch/IO headroom."""
    if "FT_THREADS" in os.environ:
        return int(os.environ["FT_THREADS"])
    if "FASTLLM_NUMA_THREADS" in os.environ:
        nodes = max(1, len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))))
        requested = int(os.environ.get("FT_NUMAS", "0"))
        if 0 < requested <= nodes:
            nodes = requested
        return int(os.environ["FASTLLM_NUMA_THREADS"]) * nodes
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else range(os.cpu_count() or 1)
    cores = set()
    for cpu in affinity:
        try:
            path = Path("/sys/devices/system/cpu/cpu%d/topology" % cpu)
            cores.add(((path / "physical_package_id").read_text(), (path / "core_id").read_text()))
        except OSError:
            cores.add((0, cpu))
    return max(1, len(cores) - min(4, len(cores) // 4))


def topologies(args, explicit, hardware, gpu):
    from .util import _uses_thread_tp, _normalize_thread_tp_arg
    fixed_tp = args.tp if "tp" in explicit else os.environ.get("FASTLLM_TP", "")
    fixed_device = "device" in explicit and bool(args.device)
    if _uses_thread_tp(fixed_tp) or fixed_device:
        trial = copy.copy(args)
        trial.tp = _normalize_thread_tp_arg(fixed_tp) if fixed_tp else "off"
        if not fixed_device:
            trial.device = "cuda:0"
        mode, mapping = memory.devices(trial)
        yield trial.device, trial.tp, mode, mapping
        return
    by_capacity = sorted(gpu, key=lambda d: (-gpu[d]["free"], d))
    subsets = [(d,) for d in by_capacity]
    subsets += [tuple(sorted(by_capacity[:n])) for n in range(2, len(by_capacity) + 1)]
    for ids in subsets:
        if len(ids) == 1:
            yield "cuda:%d" % ids[0], "off", "pp", {ids[0]: 1}
            continue
        sm = [hardware[d]["sm"] for d in ids]
        divisor = math.gcd(*sm)
        mapping = {d: hardware[d]["sm"] // divisor for d in ids}
        if len(set(sm)) == 1:
            device = "cudapp=" + ",".join(str(d) for d in ids)
        else:
            device = repr({"cuda:%d" % d: weight for d, weight in mapping.items()})
        yield device, "off", "pp", mapping
        # First release automatically considers the validated two-rank layout.
        # Other explicit equal TP degrees still use the existing budget checks.
        if ("tp" not in explicit and not fixed_tp and len(ids) == 2 and
                len({hardware[d]["capability"] for d in ids}) == 1 and max(sm) <= min(sm) * 1.2):
            yield "cuda:" + ",".join(map(str, ids)), "cuda:" + ",".join(map(str, ids)), "tp", {d: 1 for d in ids}


def latency_score(candidate, model, costs, args, mode, mapping, hardware, peers):
    """Relative latency for a 4K prompt + 512 outputs; no model reload/search.

    Dense bandwidth, active expert traffic, PCIe streaming and TP collectives
    compete with chunk-dependent cache capacity. Cache locality is a smooth
    prior, not a measured cache hit rate. Keep this policy separate from safety.
    """
    experts, dense, common, _, _, _ = costs
    resident = sum(size for layer, size in enumerate(experts)
                   if memory.expert_layer(layer, model) < candidate["resident_layers"])
    if args.mtp and model.family == "qwen4":
        resident += sum(experts[model.layers:])
    host = sum(experts) - resident
    ids = list(mapping)
    single_bandwidth = min(hardware[d]["memory_bandwidth"] for d in ids) * .45
    bandwidth = single_bandwidth
    link = min(hardware[d]["link_bandwidth"] for d in ids)
    compute = min(hardware[d]["sm"] * hardware[d]["clock_khz"] * 1000 for d in ids) * 256
    if mode == "tp":
        bandwidth *= len(ids)
        compute *= len(ids)
        link *= len(ids)
    cpu_bandwidth = max(1, args.threads) * 1.0e9
    cache = candidate.get("moe_cuda_cache_per_device", dict.fromkeys(ids, candidate["moe_cuda_cache"]))
    covered = min(host, sum(cache.values()))
    hit = (covered / host) ** .6 if host else 1.
    active = min(1., max(model.topk, args.moe_experts) / model.experts)
    misses = host * active * (1 - hit)
    # CPU experts and streamed GPU misses overlap in the shared hybrid runtime.
    miss_time = misses / (cpu_bandwidth + link * .5)
    if args.moe_device == "disk" and host:
        ram_hit = min(1., candidate["moe_cpu_cache"] / host) ** .6
        miss_time += misses * (1 - ram_hit) / .8e9
    # Common tensors (notably the output head) are not uniformly TP sharded.
    dense_time = sum(dense) / bandwidth + common / single_bandwidth
    gpu_time = (resident * active + host * active * hit) / bandwidth
    if mode == "tp" and len(ids) > 1:
        peer = all(peers.get((a, b), False) for a in ids for b in ids)
        # Attention and MoE each merge their tensor-parallel output.
        communication = 2 * model.layers * (len(ids) - 1) * (24e-6 if peer else 65e-6)
    # PP only transfers the activation at stage boundaries.
    else:
        communication = max(0, len(ids) - 1) * 20e-6
    decode = dense_time + max(gpu_time, miss_time) + communication + model.layers * 20e-6
    prompt = min(4096, candidate["context_tokens"])
    chunk = candidate["chunked_prefill_size"]
    active_params = model.layers * 3 * model.hidden * model.intermediate * max(model.topk, args.moe_experts)
    prefill = prompt * (2 * active_params + sum(dense) + common) / compute
    prefill += max(0, host - covered) * math.ceil(prompt / chunk) / (link + cpu_bandwidth)
    prefill += math.ceil(prompt / chunk) * model.layers * 50e-6
    # Long single-request decode dominates this interactive default. Explicit
    # concurrency adds value to TP's parallel compute; it never changes capacity.
    return prefill + 512 * decode * max(1, args.max_batch)


def kv_residencies(args, model):
    """Supported exact streaming pools, without changing context capacity."""
    from .util import _fastllm_env_flag_enabled
    if (not model.streamed_kv or model.family != "qwen4" or args.kvmem is not None or
            args.atype != "float16" or args.kv_cache_dtype != "float16" or
            args.max_batch != 1 or args.tokens != args.max_context_length or
            _fastllm_env_flag_enabled("FASTLLM_SKIP_WARMUP")):
        return [0]
    # KVMem cannot preserve prefix/history reuse. Honor explicit requests for
    # either feature, including the existing prefix-cache environment override.
    prefix = args.prefix_cache or os.environ.get("FASTLLM_PREFIX_CACHE", "")
    history = args.cache_history if "cache_history" in getattr(args, "_auto_explicit", ()) else ""
    if any(str(v).lower() not in ("", "false", "0", "off") for v in (prefix, history)):
        return [0]
    return [0] + [n for n in (32768, 16384, 8192) if n < args.max_context_length]


def full_kv_has_expert_room(candidate, model, costs, mode, mapping, mtp):
    """Full KV is roomy only if each GPU can also hold all its experts."""
    if "kvmem" in candidate:
        return False
    host = dict.fromkeys(mapping, 0)
    for layer, size in enumerate(costs[0]):
        if layer < candidate["resident_layers"] or (mtp and layer >= model.layers):
            continue
        devices = list(mapping) if mode == "tp" else [memory.layer_device(layer, model.layers, mapping)]
        for device in devices:
            host[device] += math.ceil(size / len(devices))
    cache = candidate.get("moe_cuda_cache_per_device", dict.fromkeys(mapping, candidate["moe_cuda_cache"]))
    return all(candidate["gpu_estimated_bytes"][d] + max(0, host[d] - cache[d]) <=
               candidate["gpu_budget_bytes"][d] for d in mapping)


def resolve(args, query_gpu, *, hardware=None, ram=None, use_numa=None):
    """Resolve one --auto invocation transactionally, before model allocation."""
    from .env import env
    from .util import _fastllm_env_flag_enabled
    graph_enabled = ("FASTLLM_CUDA_GRAPH" not in os.environ or
                     _fastllm_env_flag_enabled("FASTLLM_CUDA_GRAPH"))
    explicit = set(getattr(args, "_auto_explicit", ()))
    base = copy.copy(args)
    for name in memory.AUTO_OPTIONS:
        if name not in explicit:
            setattr(base, name, "auto")
    if "threads" not in explicit or base.threads == -1:
        base.threads = cpu_threads()
    if base.threads < 1:
        raise ValueError("--auto requires positive CPU threads")
    if base.max_batch <= 0:
        base.max_batch = 1
    metadata, tensors, model = memory.model_inputs(base)
    # Dense Qwen3's graph warmup currently hits stream-capture-unsupported
    # in the deferred CUDA pool. Keep its new auto adapter on eager CUDA;
    # an explicit environment override still opts into graph execution.
    disable_qwen3_graph = model.family == "qwen3" and "FASTLLM_CUDA_GRAPH" not in os.environ
    if disable_qwen3_graph:
        graph_enabled = False
    apply_model_defaults(base, metadata)
    # DeepSeek V4.1 targets its 1M window; other models retain the 128K policy.
    if base.max_context_length <= 0:
        target = 1048576 if model.family == "deepseek_v41" else 131072
        base.max_context_length = min(model.context, base.tokens if base.tokens > 0 else target)
    if base.tokens <= 0:
        base.tokens = base.max_context_length * base.max_batch
    if "atype" not in explicit and model.family != "deepseek_v41":
        base.atype = "bfloat16" if model.family == "glm5" else "float16"
    if "kv_cache_dtype" not in explicit:
        base.kv_cache_dtype = "float16"
    # Re-evaluate geometry with the selected defaults (explicit precisions win).
    kv_dtype = base.atype if base.kv_cache_dtype == "auto" and base.atype != "auto" else base.kv_cache_dtype
    model = memory.geometry(metadata, kv_dtype, base.mtp, base.dspark)
    if hardware is None:
        hardware = cuda_hardware()
    info, peers = hardware
    gpu = {d: query_gpu(d) for d in info}
    ram = ram if ram is not None else memory.host_memory()
    numa = env.use_numas if use_numa is None else use_numa
    backends = [base.moe_device] if "moe_device" in explicit and base.moe_device else ["numa" if numa else "cpu", "disk"]
    tables = [base.ngram_device] if "ngram_device" in explicit else ["cpu", "disk"]
    table_bytes = sum(t.bytes for t in tensors if memory.is_lookup_table(t.name))
    if "ngram_device" not in explicit and table_bytes > max(4 * memory.GiB, ram["total"] // 10):
        # Huge sparse lookup tables add substantial startup IO/RAM, while the
        # existing disk path reads only requested rows and benefits from page cache.
        tables = ["disk", "cpu"]
    if model.family == "glm5" and "ngram_device" not in explicit:
        tables = ["cpu"]  # GLM has no PLE/Engram table.
    choices = []
    rejection = "no supported CUDA topology"
    for backend in backends:
        # Disk is a capacity fallback. Do not trade host-resident experts for IO
        # merely because an approximate cache-locality score looks favourable.
        kv_options = kv_residencies(base, model)
        for kv_resident in kv_options:
            previous_choices = len(choices)
            for table in tables:
                trial = copy.copy(base)
                trial.moe_device, trial.ngram_device = backend, table
                if backend == "cpu" and model.family != "qwen4" and trial.moe_cuda_cache == "auto":
                    trial.moe_cuda_cache = 0
                memory.validate_options(trial)
                cached_costs = None
                implicit_embedding_bytes = None
                for device, tp, mode, mapping in topologies(trial, explicit, info, gpu):
                    if any(d not in gpu for d in mapping):
                        raise ValueError("--auto: selected CUDA ordinal is not visible")
                    trial.device, trial.tp = device, tp
                    try:
                        prepared, costs, notes = memory.prepare(trial, metadata, tensors, model, mode, mapping,
                                                                costs=cached_costs)
                        cached_costs = costs
                        if implicit_embedding_bytes is None:
                            implicit_embedding_bytes = 0
                            if (model.family == "qwen4" and graph_enabled and
                                    not prepared.cuda_embedding and not prepared.low_gpu_mem):
                                embedding_args = copy.copy(prepared)
                                embedding_args.cuda_embedding = True
                                implicit_embedding_bytes = memory.weight_costs(tensors, model, embedding_args)[2] - costs[2]
                        cpu_embedding = (model.family == "qwen4" and prepared.mtp > 0 and
                                         mode == "pp" and len(mapping) > 1 and
                                         not {"cuda_embedding", "low_gpu_mem"} & explicit)
                        prefer_cache = (model.family == "qwen4" and backend != "disk" and
                                        prepared.moe_device_layers == "auto" and prepared.moe_cuda_cache == "auto")
                        def rank(p):
                            score = latency_score(p, model, costs, prepared, mode, mapping, info, peers)
                            # Keep full GPU execution when it fits. Otherwise
                            # prefer flexible caching; partial residency remains
                            # available when host memory cannot hold all experts.
                            placement = (2 if p["resident_layers"] == model.layers else
                                         int(p["resident_layers"] == 0 and p["moe_cuda_cache"] > 0)) if prefer_cache else 0
                            return (placement, -score, -p["host_estimated_bytes"], -p["resident_layers"])
                        result = memory.plan(prepared, model, costs, mode, mapping, gpu, ram,
                                             rank=rank, allow_full_gpu=True, kv_resident=kv_resident,
                                             per_device_cache=model.family == "qwen4",
                                             implicit_embedding_bytes=implicit_embedding_bytes,
                                             hybrid_cpu_embedding=cpu_embedding)
                    except ValueError as error:
                        rejection = str(error)
                        continue
                    result.update(device=device, tp=tp, moe_device=backend, ngram_device=table,
                                  threads=base.threads, atype=base.atype, kv_cache_dtype=base.kv_cache_dtype,
                                  max_context_length=base.max_context_length, tokens=base.tokens)
                    score = latency_score(result, model, costs, prepared, mode, mapping, info, peers)
                    if kv_resident:
                        notes.append("Exact QSA KV streaming leaves GPU memory for experts; prefix/history reuse disabled")
                    if notes:
                        result["notes"] = notes
                    roomy = model.family == "qwen4" and full_kv_has_expert_room(
                        result, model, costs, mode, mapping, prepared.mtp)
                    choices.append((score, copy.copy(trial), result, roomy))
            # At long context prefer a 32K streaming pool unless full KV also
            # leaves room for every expert. Smaller pools are capacity fallbacks.
            if len(choices) > previous_choices and (kv_resident or base.max_context_length < 65536 or 32768 not in kv_options):
                break
        if choices:
            break
    if not choices:
        raise ValueError("--auto could not find a safe configuration: " + rejection)
    # Prefer pipeline placement for Qwen without overriding explicit TP.
    # Retain TP as a capacity fallback; explicit TP produces no PP candidates.
    eligible = choices
    if model.family == "qwen4":
        eligible = [item for item in choices if item[2]["mode"] == "pp"] or choices
    if (model.family == "qwen4" and base.moe_device_layers == "auto" and
            base.moe_cuda_cache == "auto" and eligible[0][2]["moe_device"] != "disk"):
        # Apply the same placement preference across topology/KV candidates.
        full = [item for item in eligible if item[2]["resident_layers"] == model.layers]
        cached = [item for item in eligible if item[2]["resident_layers"] == 0 and item[2]["moe_cuda_cache"] > 0]
        eligible = full or cached or eligible
    if any("kvmem" in item[2] for item in eligible):
        eligible = [item for item in eligible if "kvmem" in item[2] or item[3]]
    fastest = min(item[0] for item in eligible)
    # Small predicted wins are within the error of this uncalibrated model.
    # Prefer fewer devices/collectives when estimates are within ten percent.
    # Where every expert fits as well, keep full KV on an otherwise tied topology.
    close = [item for item in eligible if item[0] <= fastest * 1.10]
    _, chosen, result, _ = min(close, key=lambda item: (
        len(item[2]["gpu_budget_bytes"]), item[2]["mode"] == "tp", "kvmem" in item[2], item[0]))
    for name in memory.AUTO_OPTIONS:
        if getattr(chosen, name) == "auto":
            setattr(chosen, name, result[name])
    if "kvmem" in result:
        chosen.kvmem = json.dumps(result["kvmem"], sort_keys=True)
        # Model detection supplies a history-cache default before planning. It
        # must not re-enable reuse after the planner chose exact KV streaming.
        for name in ("cache_history", "prefix_cache"):
            if name not in explicit:
                setattr(chosen, name, "false")
    if (model.family == "qwen4" and chosen.mtp > 0 and result["mode"] == "pp" and
            len(result["gpu_budget_bytes"]) > 1 and result["resident_layers"] < model.layers and
            not {"cuda_embedding", "low_gpu_mem"} & explicit):
        # Target and draft run on opposite PP stages. Graph's implicit CUDA
        # embedding would move the entire shared weight between those stages.
        chosen.low_gpu_mem, chosen.cuda_embedding = True, False
        result["low_gpu_mem"] = True
        result.setdefault("notes", []).append("PP MTP uses CPU embedding to avoid moving its shared weight between GPUs")
    changed = (*memory.AUTO_OPTIONS, "device", "tp", "moe_device", "ngram_device", "threads",
               "atype", "kv_cache_dtype", "max_context_length", "tokens", "max_batch", "kvmem",
               "low_gpu_mem", "cuda_embedding", "cache_history", "prefix_cache",
               "dspark", "speculative_algorithm", "fast_prefill")
    result["automatic"] = {"policy": "interactive-4k-input-512-output", "topology_candidates": len(choices),
                           "ranking": "estimated latency; not measured throughput", "mtp": args.mtp}
    result["automatic"]["kv_streaming"] = "kvmem" in result
    result["automatic"]["cuda_graph"] = graph_enabled
    result["automatic"]["dspark"] = chosen.dspark
    result["automatic"]["fast_prefill"] = chosen.fast_prefill
    result["automatic"]["gpu_memory_limit"] = result.get("auto_gpu_memory_limit")
    result["automatic"]["host_memory_limit"] = result.get("auto_host_memory_limit")
    if chosen.dspark:
        result["automatic"]["dspark_confidence_threshold"] = chosen.speculative_dspark_confidence_threshold
    for name in changed:
        setattr(args, name, getattr(chosen, name))
    if disable_qwen3_graph:
        os.environ["FASTLLM_CUDA_GRAPH"] = "0"
        result.setdefault("notes", []).append("Dense Qwen3 auto uses eager CUDA to avoid unsupported graph capture")
    print("[auto] " + json.dumps(result, sort_keys=True), flush=True)
    return result
