"""Opt-in, offline startup tuning. The existing auto policy is only a baseline.

Search and measurements run before serving, in disposable processes. Historical
timings prioritize candidates; they never bypass a fresh capacity check or run.
"""
import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from . import auto_config as auto, hybrid_auto as memory, autotune_cost as cost

GiB, MiB = memory.GiB, memory.MiB
SCHEMA = 2
PLACEMENT_OPTIONS = ("moe_device", "moe_device_layers", "moe_cuda_cache", "moe_cpu_cache")
TUNABLE = (*PLACEMENT_OPTIONS, "chunked_prefill_size")
PROMPT = """A team is planning a reliable document search service. Documents arrive
through a queue, are validated, split into passages, indexed, and made searchable.
Users need accurate citations and predictable response times. The team must handle
updates, duplicates, failed uploads, access permissions, and incomplete documents.
Capacity is limited, so memory, storage traffic and compute time must be measured
separately. A cache can save repeated work but must be invalidated after an update.
The rollout should include reproducible tests, observability, a rollback procedure,
and a way to compare the service with its previous version using the same queries.
"""
TASK = "\nWrite a detailed implementation and test plan with 40 numbered items. Explain each item in several sentences.\n"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _candidate(args, plan, numa_layers=0):
    settings = {name: getattr(args, name) for name in TUNABLE}
    placement = plan.get("autotune_placement")
    identity = {"settings": settings, "placement": placement,
                "host_gpu_cache": plan.get("moe_cuda_cache_per_device")}
    nearby = copy.deepcopy(identity)
    # Free RAM drifts between launches. Nearby cache sizes may reuse a timing
    # prior for ordering, while the exact configuration is always measured anew.
    for key in ("moe_cuda_cache", "moe_cpu_cache"):
        nearby["settings"][key] = round(settings[key] / GiB)
    if nearby["host_gpu_cache"]:
        nearby["host_gpu_cache"] = {d: round(n / GiB) for d, n in nearby["host_gpu_cache"].items()}
    return {"id": _digest(identity)[:20], "history_key": _digest(nearby)[:20], **identity, "plan": plan,
            "gpu_layers": plan["resident_layers"], "numa_layers": numa_layers}


def _front_map(resident, numa, mapping):
    front = {}
    for layer in range(resident):
        device = memory.layer_device(layer, resident, mapping)
        key = "cuda:%d" % device
        front[key] = front.get(key, 0) + 1
    front["numa"] = numa
    return front


def candidates(base, baseline, metadata, tensors, model, gpu, ram):
    """Bounded grid with the baseline's GPU prefix/topology and fixed precision.

    NUMA and disk have independent CUDA pools. Charge the disk pool to each GPU
    and its CPU cache to host storage, then use the unchanged NUMA capacity solver.
    Draft experts follow their corresponding target layer, just as in the runtime.
    """
    from .env import env
    mode, mapping = memory.devices(base)
    prepared, costs, _ = memory.prepare(base, metadata, tensors, model, mode, mapping)
    experts, dense, common, host, largest, snapshot = costs
    implicit_embedding = _embedding_bytes(base, tensors, model, costs)
    explicit = set(getattr(base, "_auto_explicit", ()))
    resident = baseline["resident_layers"]
    host_layers = model.layers - resident
    numa_layers = baseline.get("autotune_placement", {}).get("front", {}).get(
        "numa", host_layers if base.moe_device != "disk" else 0)
    result = [_candidate(base, baseline, numa_layers)]
    # Auto's mixed baseline reports only the disk suffix in moe_device_layers.
    # New candidates still keep its GPU prefix, not that shorter disk suffix.
    prepared.moe_device_layers = model.layers - resident if resident else -1
    chunks = ([base.chunked_prefill_size] if "chunked_prefill_size" in explicit else
              sorted({min(base.max_context_length, base.tokens, c) for c in (512, 1024, 2048, 4096, 8192)}))
    # Include cache/chunk trade-offs for the existing backend, preserving every
    # explicit placement field. The new entry point never edits these constraints.
    backends = [base.moe_device]
    if base.moe_device != "disk" and not explicit.intersection(PLACEMENT_OPTIONS):
        backends.append("disk")
    for backend in backends:
        for chunk in chunks:
            trial = copy.copy(prepared)
            trial.chunked_prefill_size = chunk
            trial.moe_device = backend
            for name in ("moe_cuda_cache", "moe_cpu_cache"):
                if name not in explicit:
                    setattr(trial, name, "auto")
            try:
                plan = memory.plan(trial, model, costs, mode, mapping, gpu, ram, allow_full_gpu=True,
                                   implicit_embedding_bytes=implicit_embedding)
            except ValueError:
                continue
            for name in memory.AUTO_OPTIONS:
                if getattr(trial, name) == "auto":
                    setattr(trial, name, plan[name])
            plan["automatic"] = baseline["automatic"]
            result.append(_candidate(trial, plan, host_layers if trial.moe_device != "disk" else 0))
    if env.use_numas and not explicit.intersection(PLACEMENT_OPTIONS) and host_layers > 1:
        # Every integer NUMA/disk boundary is capacity-checked. This is metadata
        # enumeration, not one full model load per candidate.
        cuts = range(1, host_layers)
        for numa in cuts:
            boundary = resident + numa
            disk_bytes = sum(size for i, size in enumerate(experts)
                             if memory.expert_layer(i, model) >= boundary)
            kept = [size if memory.expert_layer(i, model) < boundary else 0
                    for i, size in enumerate(experts)]
            for chunk in chunks:
                trial = copy.copy(prepared)
                trial.moe_device, trial.chunked_prefill_size = "numa", chunk
                trial._host_moe_shared = True
                trial.moe_cuda_cache = trial.moe_cpu_cache = 0
                adjusted = (kept, dense, common, host, largest, snapshot)
                try:
                    zero = memory.plan(trial, model, adjusted, mode, mapping, gpu, ram,
                                       implicit_embedding_bytes=implicit_embedding)
                except ValueError:
                    continue
                cpu_cache = min(disk_bytes, zero["host_budget_bytes"] - zero["host_estimated_bytes"])
                cpu_cache = max(0, cpu_cache // MiB * MiB)
                total_cache = min(zero["gpu_budget_bytes"][d] - zero["gpu_estimated_bytes"][d] for d in mapping)
                for host_cache in sorted({0, *(min(3 * GiB, max(0, total_cache) // (n * MiB) * MiB)
                                              for n in (4, 2))}):
                    disk_cache = min(disk_bytes, total_cache - host_cache) // MiB * MiB
                    if disk_cache < host_cache:  # Native per-device overrides cannot exceed scalar.
                        continue
                    trial.moe_cuda_cache = host_cache
                    adjusted = (kept, dense, common + disk_cache, host + cpu_cache, largest, snapshot)
                    try:
                        plan = memory.plan(trial, model, adjusted, mode, mapping, gpu, ram,
                                           implicit_embedding_bytes=implicit_embedding)
                    except ValueError:
                        continue
                    launch = copy.copy(trial)
                    launch.moe_device, launch.moe_device_layers = "disk", model.layers - boundary
                    launch.moe_cuda_cache, launch.moe_cpu_cache = disk_cache, cpu_cache
                    plan.update({name: getattr(launch, name) for name in TUNABLE if name != "moe_device"})
                    plan["automatic"] = baseline["automatic"]
                    plan["autotune_placement"] = {"front": _front_map(resident, numa, mapping),
                                                   "disk_layers": launch.moe_device_layers}
                    plan["moe_cuda_cache_per_device"] = dict.fromkeys(mapping, host_cache)
                    result.append(_candidate(launch, plan, numa))
    unique = {}
    for item in result:
        unique.setdefault(item["id"], item)
    return list(unique.values()), costs


def _embedding_bytes(args, tensors, model, costs):
    if args.low_gpu_mem or args.cuda_embedding:
        return 0
    other = copy.copy(args)
    other.cuda_embedding = True
    # Conservative even when graphs are disabled, matching the old planner's
    # implicit embedding allowance without touching graph environment settings.
    return max(0, memory.weight_costs(tensors, model, other)[2] - costs[2])


def seed_estimates(pool, base, model, costs, hardware):
    """An explicitly uncalibrated prior until native measurements are available."""
    _, mapping = memory.devices(base)
    info, peers = hardware
    for item in pool:
        args = copy.copy(base)
        vars(args).update(item["settings"])
        p = copy.deepcopy(item["plan"])
        if item["placement"]:
            # Rank only: NUMA weights cannot miss in RAM. This smooth locality
            # prior does not model real expert reuse; measurements decide winners.
            n = item["numa_layers"]
            start = item["gpu_layers"]
            p["moe_cpu_cache"] += sum(costs[0][start:start + n])
            p["moe_cuda_cache_per_device"] = {
                d: p["moe_cuda_cache"] + item["host_gpu_cache"][d]
                for d in mapping}
        item["estimated_seconds"] = auto.latency_score(p, model, costs, args, "pp", mapping, info, peers)


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * MiB), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_files(path):
    """Include every backbone shard and the adjacent GGUF Engram sidecar."""
    root = Path(path).resolve()
    if root.is_dir():
        return sorted(p for p in root.rglob("*") if p.is_file() and
                      p.suffix in (".json", ".safetensors", ".gguf", ".bin"))
    files = [root]
    if root.suffix == ".gguf":
        from .gguf_metadata import read_gguf_metadata
        metadata = read_gguf_metadata(root, {"split.count", "deepseek41.engram.present", "deepseek41.engram.file"})
        count = int(metadata.get("split.count", 1))
        if count > 1:
            match = re.fullmatch(r"(.*)-\d{5}-of-(\d{5})\.gguf", root.name)
            if not match or int(match[2]) != count:
                raise ValueError("autotune cannot resolve GGUF shards")
            files = [root.with_name("%s-%05d-of-%05d.gguf" % (match[1], i + 1, count)) for i in range(count)]
        if metadata.get("deepseek41.engram.present"):
            name = metadata["deepseek41.engram.file"]
            if not name or name in (".", "..") or "/" in name or "\\" in name:
                raise ValueError("autotune requires an adjacent GGUF Engram file")
            files.append(root.with_name(name))
    return files


def fingerprint(base, tensors, hardware, prompt):
    from . import llm
    from .util import make_normal_parser
    root = Path(base.path).resolve()
    files = checkpoint_files(root)
    signature = [(str(p.relative_to(root) if root.is_dir() else p.name),
                  p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    normal_fields = vars(make_normal_parser("autotune").parse_args([]))
    settings = {k: getattr(base, k, None) for k in normal_fields
                if not k.startswith("autotune") and k not in (*TUNABLE, "_auto_explicit", "auto")}
    settings["constraints"] = {k: getattr(base, k) for k in TUNABLE if k in base._auto_explicit}
    environment = {k: v for k, v in os.environ.items()
                   if k.startswith(("FASTLLM_", "FT_", "CUDA_", "OMP_", "NUMA_"))}
    cpu = Path("/proc/cpuinfo").read_text()
    cpu = [line for line in cpu.splitlines() if line.startswith(("model name", "flags", "physical id", "core id"))]
    node_layout = [(p.name, p.read_text()) for p in sorted(Path("/sys/devices/system/node").glob("node*/cpulist"))]
    driver = Path("/proc/driver/nvidia/version").read_text()
    code = {p.name: _file_hash(p) for p in (Path(__file__), Path(cost.__file__), Path(auto.__file__), Path(memory.__file__),
                                          *(Path(__file__).with_name(name) for name in
                                            ("util.py", "llm.py", "benchmark.py", "encoding_dsv41.py")),
                                          Path(llm.fastllm_lib._name))}
    # Store only the digest, never the environment or arbitrary user settings.
    return _digest({"schema": SCHEMA, "files": signature, "root": str(root), "code": code,
                    "tensors": [(t.name, t.kind, t.bytes) for t in tensors], "cpu": cpu,
                    "nodes": node_layout, "affinity": sorted(os.sched_getaffinity(0)),
                    "driver": driver, "gpu": hardware[0], "peers": sorted((a, b, v) for (a, b), v in hardware[1].items()),
                    "settings": settings, "environment": environment, "prompt": prompt})


def read_history(path, key):
    try:
        value = json.loads(Path(path).read_text())
        if value["schema"] != SCHEMA or value["fingerprint"] != key:
            return []
        return [item for item in value["trials"] if isinstance(item.get("history_key"), str)
                and valid_measurement(item.get("measurement"))]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []


def valid_measurement(value):
    try:
        samples = value["samples"]
        return (len(samples) == 2 and all(
            isinstance(sample["input_hash"], str) and sample["input_tokens"] >= 128 and
            sample["cached_input_tokens"] == 0 and sample["output_tokens"] >= 64 and
            all(math.isfinite(sample[k]) and sample[k] > 0
                for k in ("ttft_seconds", "decode_seconds_per_token", "score_seconds"))
            for sample in samples) and math.isfinite(value["score_seconds"]) and value["score_seconds"] > 0 and
            math.isclose(value["score_seconds"], statistics.median(s["score_seconds"] for s in samples)))
    except (TypeError, KeyError, ValueError):
        return False


def choose_winner(trials):
    if not trials:
        raise RuntimeError("Autotune time budget exhausted before the baseline completed")
    baseline = trials[0]
    if not valid_measurement(baseline.get("measurement")):
        raise RuntimeError("Autotune baseline did not complete safely; refusing to launch an unverified candidate")
    best = baseline
    for trial in trials[1:]:
        if not valid_measurement(trial.get("measurement")):
            continue
        a, b = baseline["measurement"], trial["measurement"]
        if any(x["input_hash"] != y["input_hash"] for x, y in zip(a["samples"], b["samples"])):
            continue
        # Require a repeatable >=5% gain and a gain larger than observed spread.
        spread = max((max(s["score_seconds"] for s in m["samples"]) -
                      min(s["score_seconds"] for s in m["samples"])) / m["score_seconds"] for m in (a, b))
        gain = 1 - b["score_seconds"] / a["score_seconds"]
        repeatable = all(y["score_seconds"] < x["score_seconds"] * .97
                         for x, y in zip(a["samples"], b["samples"]))
        if repeatable and gain >= max(.05, spread) and b["score_seconds"] < best["measurement"]["score_seconds"]:
            best = trial
    return best


def fits(candidate, args, query_gpu):
    """Re-read live CMA/cgroup/GPU limits after each child releases its model."""
    ram = memory.host_memory(reclaim_file_cache=True)
    host = ram["available"] - min(4 * GiB, max(GiB, ram["total"] // 10))
    cap = memory._memory_limit_bytes(args.auto_host_memory_limit, "auto_host_memory_limit")
    if cap is not None:
        host = min(host, cap)
    plan = candidate["plan"]
    if plan["host_estimated_bytes"] > host:
        return False
    cap = memory._memory_limit_bytes(args.auto_gpu_memory_limit, "auto_gpu_memory_limit")
    for d, used in plan["gpu_estimated_bytes"].items():
        gpu = query_gpu(int(d))
        limit = gpu["free"] - math.ceil(gpu["total"] * (1 - args.gpu_mem_ratio))
        if cap is not None:
            limit = min(limit, cap)
        if used > limit:
            return False
    return True


class WorkerCleanupError(RuntimeError):
    """Do not load another model while a killed worker still owns resources."""


def _group_alive(pgid):
    # Zombies own no model storage; killpg(0) alone cannot distinguish them
    # from helpers still freeing pinned weights or blocked inside the kernel.
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            # Unrelated users' processes may be hidden by proc mount policy.
            continue
    return False


def _stop(process):
    # Also retire any native helper processes left in this worker's group after
    # the Python leader exits. The group is created exclusively for this probe.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    exited = False
    try:
        process.wait(timeout=5)
        exited = True
    except subprocess.TimeoutExpired:
        pass
    if not exited or _group_alive(process.pid):
        deadline = time.monotonic() + 30
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            # Releasing ~100 GiB of pinned NUMA storage and CUDA contexts can
            # outlive SIGKILL by several seconds while kernel teardown finishes.
            process.wait(timeout=30)
        except subprocess.TimeoutExpired as error:
            raise WorkerCleanupError("autotune worker %d has not exited after SIGKILL; refusing another model load" % process.pid) from error
        while _group_alive(process.pid):
            if time.monotonic() >= deadline:
                raise WorkerCleanupError("autotune helpers in group %d have not exited; refusing another model load" % process.pid)
            time.sleep(.1)


def probe(base, candidate, prompt, directory, deadline):
    """One fresh interpreter/native runtime, no listening port or shared model."""
    from .util import make_normal_parser
    directory = Path(directory)
    directory.mkdir(parents=True)
    # Server authentication/listener settings are not needed by the worker.
    fields = vars(make_normal_parser("autotune").parse_args([]))
    args = {k: getattr(base, k, v) for k, v in fields.items()}
    args.update(candidate["settings"])
    args.update(auto=False, autotune=False, _autotune_plan=candidate["plan"])
    args["model"] = base.path  # Do not replay a saved JSON config in the worker.
    args["_auto_explicit"] = sorted(set(base._auto_explicit) | set(TUNABLE) |
                                     {"device", "tp", "fast_prefill", "dspark", "cuda_slab"})
    _write(directory / "input.json", {"args": args, "prompt": prompt, "parent_pid": os.getpid(),
                                      "deadline": deadline if math.isfinite(deadline) else None})
    process = None
    peak_rss, minimum_ram = 0, None
    try:
        with (directory / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", __package__ + ".autotune", str(directory)],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            last_notice = 0
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    completed = directory / "result.json"
                    if completed.exists():
                        value = json.loads(completed.read_text())
                        if valid_measurement(value):
                            value.update(peak_rss_bytes=peak_rss, minimum_usable_ram_bytes=minimum_ram,
                                         calibration_error="calibration reached time budget; timed requests completed")
                            return value  # finally still reaps the process group before another load.
                    raise TimeoutError("autotune time budget exhausted (including model loading)")
                ram = memory.host_memory(reclaim_file_cache=True)
                minimum_ram = ram["available"] if minimum_ram is None else min(minimum_ram, ram["available"])
                if ram["available"] < min(GiB, ram["total"] // 50):
                    raise RuntimeError("autotune stopped: usable non-CMA/cgroup RAM below safety floor")
                try:
                    status = Path("/proc/%d/status" % process.pid).read_text()
                    values = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in status.splitlines()
                              if line.startswith(("VmHWM:", "VmRSS:", "VmSwap:"))}
                    peak_rss = max(peak_rss, values.get("VmHWM", 0))
                    if values.get("VmSwap", 0):
                        raise RuntimeError("autotune worker started swapping")
                except FileNotFoundError:
                    pass
                if now - last_notice >= 30:
                    print("[autotune] trial=%s RSS=%.2fGiB usable_RAM=%.2fGiB remaining=%ds" %
                          (candidate["id"], peak_rss / GiB, ram["available"] / GiB, max(0, deadline - now)), flush=True)
                    last_notice = now
                time.sleep(1)
            if process.returncode != 0:
                raise RuntimeError("autotune worker exited %s; see %s" % (process.returncode, directory / "worker.log"))
        value = json.loads((directory / "result.json").read_text())
        if not valid_measurement(value):
            raise ValueError("invalid or insufficient autotune measurement")
        value.update(peak_rss_bytes=peak_rss, minimum_usable_ram_bytes=minimum_ram)
        return value
    finally:
        try:
            if process is not None:
                _stop(process)
        finally:
            (directory / "input.json").unlink(missing_ok=True)


def resolve(args, query_gpu):
    """Only invoked by --autotune, never by --auto."""
    if not isinstance(args.autotune_budget, int) or args.autotune_budget <= 0:
        raise ValueError("--autotune-budget must be a positive integer number of seconds")
    if args.custom:
        raise ValueError("--autotune cannot validate placement overridden by --custom")
    if sys.platform != "linux":
        raise ValueError("--autotune currently requires Linux/CUDA")
    metadata, tensors, model = memory.model_inputs(args)
    if model.family != "deepseek_v41":
        raise ValueError("--autotune currently supports DeepSeek-V4.1; use --auto for other models")
    prompt = Path(args.autotune_prompt).read_text(encoding="utf-8") if args.autotune_prompt else PROMPT
    if not prompt.strip():
        raise ValueError("--autotune-prompt must not be empty")
    base = copy.deepcopy(args)
    base.auto = True
    hardware = auto.cuda_hardware()
    ram = memory.host_memory(reclaim_file_cache=True)
    # Tuning reloads several processes. Leave an extra GiB for the parent and
    # ordinary free-memory drift; this is exclusive to the new entry point.
    ram["available"] = max(0, ram["available"] - GiB)
    baseline = auto.resolve(base, query_gpu, hardware=hardware, ram=ram)
    if baseline["mode"] != "pp":
        raise ValueError("--autotune currently supports PP; keep --auto for TP")
    if baseline["context_tokens"] < 640:
        raise ValueError("autotune requires at least 640 tokens of planned context")
    # Resolve geometry again after auto's KV and speculative defaults.
    metadata, tensors, model = memory.model_inputs(base)
    _, mapping = memory.devices(base)
    gpu = {d: dict(query_gpu(d), capability=hardware[0][d].get("capability", 0)) for d in mapping}
    pool, costs = candidates(base, baseline, metadata, tensors, model, gpu, ram)
    key = fingerprint(base, tensors, hardware, prompt)
    path = Path(args.autotune_profile).expanduser() if args.autotune_profile else Path.home() / ".cache/fastllm/autotune" / (key + ".json")
    seed_estimates(pool, base, model, costs, hardware)
    previous = read_history(path, key)
    history = {trial["history_key"]: trial["measurement"]["score_seconds"] for trial in previous}
    predictor = cost.CostModel(model.layers, mapping)
    # Compatible calibration files may seed ordering, but no saved plan is
    # considered verified until it has completed a fresh timed trial.
    try:
        for trial in previous:
            predictor.add(trial)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        predictor = cost.CostModel(model.layers, mapping)
    run = Path(tempfile.mkdtemp(prefix=path.stem[:12] + "-", dir=_report_directory(path)))
    report = {"schema": SCHEMA, "fingerprint": key, "created": time.time(),
              "objective": {"input_tokens": min(4096, baseline["context_tokens"] - 512),
                            "output_tokens": 512, "sampling": "greedy",
                            "score": "TTFT + remaining output tokens * measured decode interval"},
              "candidate_count": len(pool), "trials": [], "logs": str(run),
              "search_space": {"gpu_prefix": baseline["resident_layers"], "topology": base.device,
                               "numa_cuts": "every feasible integer boundary",
                               "cache_splits": "0, quarter, half; host GPU cache at most 3 GiB"}}
    deadline = time.monotonic() + args.autotune_budget
    while True:
        if time.monotonic() >= deadline:
            report["stop_reason"] = "time budget exhausted"
            break
        durations = [t["duration_seconds"] for t in report["trials"] if "measurement" in t]
        if durations and deadline-time.monotonic() < statistics.median(durations)*1.1:
            report["stop_reason"] = "remaining budget below observed load/measure duration"
            break
        candidate, rankings = cost.select_next(pool, report["trials"], predictor, history)
        report["predictions"] = rankings
        if time.monotonic() >= deadline:
            report["stop_reason"] = "time budget exhausted during candidate evaluation"
            break
        if candidate is None:
            report["stop_reason"] = "all feasible candidates tested"
            break
        i = len(report["trials"])
        trial = copy.deepcopy(candidate)
        trial["prediction"] = next(p for p in rankings if p["id"] == trial["id"])
        print("[autotune] testing %d (pool=%d) GPU=%d NUMA=%d disk=%d chunk=%d" %
              (i + 1, len(pool), trial["gpu_layers"], trial["numa_layers"],
               model.layers - trial["gpu_layers"] - trial["numa_layers"], trial["settings"]["chunked_prefill_size"]), flush=True)
        started = time.monotonic()
        try:
            if not fits(candidate, base, query_gpu):
                raise RuntimeError("live memory budget changed; candidate no longer fits")
            trial["measurement"] = probe(base, candidate, prompt, run / str(i), deadline)
            try:
                predictor.add(trial)
            except (OSError, ValueError, KeyError, TypeError) as error:
                trial["calibration_error"] = str(error)
        except WorkerCleanupError as error:
            trial["error"] = str(error)
            report["trials"].append(trial)
            _write(path, report)
            raise
        except (OSError, RuntimeError, ValueError, TimeoutError) as error:
            trial["error"] = str(error)
            print("[autotune] " + str(error), flush=True)
        trial["duration_seconds"] = time.monotonic()-started
        report["trials"].append(trial)
        _write(path, report)
        if i == 0 and "measurement" not in trial:
            break
    winner = choose_winner(report["trials"])
    if not fits(winner, base, query_gpu):
        raise RuntimeError("Autotune finished but live memory decreased; refusing an unsafe final launch")
    report["winner"] = winner["id"]
    report["baseline_retained"] = winner["id"] == pool[0]["id"]
    _write(path, report)
    vars(args).update(vars(base))
    vars(args).update(winner["settings"])
    print("[autotune] selected=%s baseline_retained=%s profile=%s" %
          (winner["id"], report["baseline_retained"], path), flush=True)
    return winner["plan"]


def _report_directory(path):
    directory = path.parent / (path.stem + ".runs")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _request(model, tokens, output_tokens):
    from . import llm
    from .benchmark import _launch_raw_response
    start = time.perf_counter()
    handle = _launch_raw_response(model, tokens, output_tokens,
                                 {"do_sample": False, "top_p": 1., "top_k": 1,
                                  "temperature": 1., "repeat_penalty": 1.})
    buffer, stats = (ctypes.c_int * 64)(), (ctypes.c_int * 3)(-1, -1, -1)
    first, last, first_count, count = None, None, 0, 0
    try:
        while True:
            n = llm.fastllm_lib.fetch_response_tokens_batch_with_statistics_llm_model(
                model.model, handle, buffer, len(buffer), stats)
            now = time.perf_counter()
            if n < 0:
                handle = None
                if n != -1:
                    raise RuntimeError("autotune inference failed: %s" % n)
                break
            if n:
                if first is None:
                    first, first_count = now, n
                count += n
                last = now
            else:
                time.sleep(.0005)
    finally:
        if handle is not None:
            llm.fastllm_lib.abort_response_llm_model(model.model, handle)
    if first is None or count <= first_count:
        raise RuntimeError("autotune prompt produced too few tokens")
    tpot = (last - first) / (count - first_count)
    return {"input_tokens": len(tokens), "output_tokens": count, "first_batch_tokens": first_count,
            "start_seconds": start,
            "cached_input_tokens": stats[0], "ttft_seconds": first - start,
            "decode_seconds_per_token": tpot, "prefill_tokens_per_second": len(tokens) / (first - start),
            "score_seconds": first - start + max(0, output_tokens - first_count) * tpot,
            "input_hash": _digest(tokens)}


def _render_prompt(model, messages):
    # Official DeepSeek formatting also applies to standalone GGUF checkpoints,
    # whose native tokenizer has no hf_tokenizer object.
    if model._is_deepseek_v4() and not model.force_chat_template:
        return model._deepseek_encode_messages(None)(
            messages, thinking_mode="thinking" if model.enable_thinking else "chat")
    if model.hf_tokenizer is not None:
        from .llm import apply_hf_chat_template
        return apply_hf_chat_template(model.hf_tokenizer, messages, enable_thinking=model.enable_thinking)
    return model.apply_chat_template(messages, add_generation_prompt=True)


def _parent_lifetime(parent_pid):
    # A killed CLI must not leave a detached worker holding the model/GPU memory.
    # SIGKILL also takes effect while the worker is inside a blocking native call.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "cannot bind autotune worker to parent lifetime")
    if os.getppid() != parent_pid:
        raise RuntimeError("autotune parent exited before worker initialization")


def worker(directory):
    from .util import make_normal_llm_model
    from .benchmark import _encode_prompt
    directory = Path(directory)
    request = json.loads((directory / "input.json").read_text())
    _parent_lifetime(request["parent_pid"])
    args = argparse.Namespace(**request["args"])
    # Fail before loading a large checkpoint when Python and native builds differ.
    from . import llm
    start_capture = getattr(llm.fastllm_lib, "start_autotune_recording", None)
    stop_capture = getattr(llm.fastllm_lib, "stop_autotune_recording", None)
    if start_capture is None or stop_capture is None:
        raise RuntimeError("--autotune calibration requires a matching native library")
    start_capture.argtypes, start_capture.restype = [ctypes.c_char_p], ctypes.c_bool
    stop_capture.argtypes, stop_capture.restype = [], ctypes.c_bool
    # JSON object keys are strings; native cache setter expects CUDA ordinals.
    plan = args._autotune_plan
    if "moe_cuda_cache_per_device" in plan:
        plan["moe_cuda_cache_per_device"] = {int(k): v for k, v in plan["moe_cuda_cache_per_device"].items()}
    # Equalize checkpoint page-cache state without flushing unrelated host data.
    for path in checkpoint_files(args.path):
        if path.is_file() and path.suffix in (".safetensors", ".gguf", ".bin"):
            with path.open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    model = make_normal_llm_model(args)
    try:
        target = min(4096, plan["context_tokens"] - 512)
        if target < 128:
            raise ValueError("autotune requires at least 640 tokens of context")
        text = request["prompt"]
        # Repetition fills context; tokens remain natural-language passages, not
        # uniform random IDs. Each trial uses identical token sequences and order.
        encoded = _encode_prompt(model, text)
        if not encoded:
            raise ValueError("autotune prompt has no tokens")
        context = text * (target // len(encoded) + 2)
        samples = []
        for marker, length, output in (("Warmup", 128, 64), ("Analysis", target, 512),
                                       ("Design", target, 512), ("Calibration", target, 64)):
            if marker == "Calibration":
                value = {"samples": samples, "score_seconds": statistics.median(s["score_seconds"] for s in samples)}
                _write(directory / "result.json", value)
                # Timed requests are already finished. A separate request may
                # synchronize operators to collect curves without contaminating them.
                estimate = 3*(statistics.mean(s["ttft_seconds"] for s in samples)+
                              64*statistics.mean(s["decode_seconds_per_token"] for s in samples))+15
                if time.monotonic()+estimate >= (request.get("deadline") or float("inf")):
                    value["calibration_error"] = "insufficient remaining calibration budget"
                    break
            messages = [{"role": "user", "content": marker + ":\n" + context + TASK}]
            rendered = _render_prompt(model, messages)
            full = _encode_prompt(model, rendered)
            # Retain the request marker and complete generation suffix when
            # shortening a repeated passage to the exact input length.
            tail = min(96, length // 2)
            tokens = full[:length - tail] + full[-tail:]
            if marker == "Calibration":
                trace = directory / "trace.jsonl"
                if not start_capture(str(trace).encode()):
                    raise RuntimeError("could not start bounded autotune capture")
                try:
                    observed = _request(model, tokens, output)
                finally:
                    complete = stop_capture()
                try:
                    if not complete or observed["cached_input_tokens"]:
                        raise ValueError("incomplete or prefix-cached calibration request")
                    profile = cost.capture(trace, observed)
                    calibration = directory / "calibration.json"
                    _write(calibration, profile)
                    value["calibration"] = {"path": str(calibration), "events": len(profile["events"]),
                                             "samples": len(profile["samples"])}
                except (OSError, ValueError) as error:
                    value["calibration_error"] = str(error)
            else:
                observed = _request(model, tokens, output)
                print("[autotune sample] " + json.dumps(observed), flush=True)
                if marker != "Warmup":
                    samples.append(observed)
        _write(directory / "result.json", value)
    finally:
        model.release_memory()


if __name__ == "__main__":
    worker(sys.argv[1])
