"""Content identities and startup configuration for native prefix persistence.

Identity is computed once per model construction, from tensor bytes rather than
paths or mtimes. The native model copies the exported settings at construction;
request processing never hashes model files or consults another model's env.
"""
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import ctypes
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import threading
import time

FORMAT_VERSION = 2
DEFAULT_DISK_BYTES = 256 << 30
_READ_BYTES = 8 << 20
_HASH_WORKERS = 8
_ENV_LOCK = threading.RLock()
_INPUT_ENV = ("FASTLLM_PREFIX_CACHE_DIR", "FASTLLM_PREFIX_CACHE_DISK_BYTES",
              "FASTLLM_PREFIX_CACHE_RESTORE_POLICY")
_DERIVED_ENV = tuple("FASTLLM_PREFIX_CACHE_" + suffix for suffix in (
    "IDENTITY", "FAMILY_IDENTITY", "TARGET_IDENTITY", "ENCODER_IDENTITY",
    "DRAFT_IDENTITY", "MANIFEST_VERSION"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stat_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _check_unchanged(path, handle, before):
    if (_stat_identity(before) != _stat_identity(os.fstat(handle.fileno())) or
            _stat_identity(before) != _stat_identity(path.stat())):
        raise ValueError("Model file changed while building persistent-cache identity")


def _file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for data in iter(lambda: handle.read(_READ_BYTES), b""):
            digest.update(data)
        _check_unchanged(path, handle, before)
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def _is_encoder_tensor(name):
    return any(part in name.split(".") for part in
               ("visual", "vision_model", "vision_tower"))


def _tensor_layout(path):
    """Validate raw tensor ranges once, keeping no weight payload in memory."""
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Invalid safetensors header in persistent-cache model")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 0 < header_bytes <= min(100_000_000, before.st_size - 8):
            raise ValueError("Invalid safetensors header length")
        raw_header = handle.read(header_bytes)
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors tensor table")
        tensors = []
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(descriptor, dict):
                raise ValueError("Invalid safetensors tensor descriptor")
            offsets = descriptor.get("data_offsets")
            shape = descriptor.get("shape")
            dtype = descriptor.get("dtype")
            if (not isinstance(offsets, list) or len(offsets) != 2 or
                    any(type(value) is not int for value in offsets) or
                    not isinstance(shape, list) or
                    any(type(value) is not int or value < 0 for value in shape) or
                    not isinstance(dtype, str) or not dtype):
                raise ValueError("Invalid safetensors tensor metadata")
            start, end = offsets
            if not 0 <= start <= end <= before.st_size - 8 - header_bytes:
                raise ValueError("Safetensors tensor is outside its data buffer")
            tensors.append((start, end, name, shape, dtype))
        previous_end = 0
        for start, end, name, shape, dtype in sorted(tensors):
            if start < previous_end:
                raise ValueError("Overlapping safetensors tensor ranges")
            previous_end = end
        _check_unchanged(path, handle, before)
    return before, 8 + header_bytes, tensors


def _model_plan(model_path):
    root = Path(model_path).expanduser().resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index_path = root / "model.safetensors.index.json"
    weight_map = None
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Persistent cache requires a nonempty safetensors weight_map")
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
    layouts, names = [], set()
    for name in shards:
        shard = Path(name)
        if shard.is_absolute() or ".." in shard.parts:
            raise ValueError("Checkpoint shard path must be relative to the model directory")
        path = root / shard
        before, offset, tensors = _tensor_layout(path)
        present = {tensor[2] for tensor in tensors}
        if weight_map is not None:
            expected = {key for key, value in weight_map.items() if value == name}
            if not expected.issubset(present):
                raise ValueError("Safetensors index references a missing tensor")
        # Native SafeTensors also loads tensors absent from the index.
        if names & present:
            raise ValueError("Duplicate tensor names in checkpoint shards")
        names.update(present)
        layouts.append((path, before, offset, tensors))
    if not names:
        raise ValueError("Persistent cache requires model weight tensors")
    return {"root": root, "config": config, "layouts": layouts, "weights": {}}


def _hash_tensor(task, buffers, buffer_bytes, progress):
    model_index, path, before, offset, tensor = task
    start, end, name, shape, dtype = tensor
    if not hasattr(buffers, "data"):
        buffers.data = bytearray(buffer_bytes)
    view = memoryview(buffers.data)
    digest = hashlib.sha256()
    # Each task has its own file offset; never seek a shared handle. The same
    # reusable buffer feeds the complete tensor SHA-256 in original byte order.
    with path.open("rb", buffering=0) as handle:
        _check_unchanged(path, handle, before)
        handle.seek(offset + start)
        remaining = end - start
        while remaining:
            count = handle.readinto(view[:min(remaining, len(view))])
            if not count:
                raise ValueError("Truncated model tensor while hashing identity")
            digest.update(view[:count])
            remaining -= count
            if progress:
                progress(model_index, read_bytes=count)
        _check_unchanged(path, handle, before)
    return model_index, name, {"dtype": dtype, "shape": shape,
                              "bytes": end - start, "sha256": digest.hexdigest()}


def _hash_models(plans, *, progress=False):
    tasks = [(index, path, before, offset, tensor)
             for index, plan in enumerate(plans)
             for path, before, offset, tensors in plan["layouts"] for tensor in tensors]
    # One pool for target, vision and draft. Large tensors go first so a large
    # embedding or the single-file draft cannot become a serial tail.
    tasks.sort(key=lambda task: (-(task[4][1] - task[4][0]), task[0], str(task[1]), task[4][2]))
    workers = min(_HASH_WORKERS, len(tasks))
    buffer_bytes = min(_READ_BYTES, max(task[4][1] - task[4][0] for task in tasks))
    buffers = threading.local()
    started = time.monotonic()
    last_report = started
    hashed_bytes, completed = [0] * len(plans), [0] * len(plans)
    totals = [sum(len(layout[3]) for layout in plan["layouts"]) for plan in plans]
    progress_lock = threading.Lock()

    def report(index=0, *, read_bytes=0, finished=0, force=False):
        nonlocal last_report
        with progress_lock:
            hashed_bytes[index] += read_bytes
            completed[index] += finished
            now = time.monotonic()
            if force or now - last_report >= 5:
                for index, plan in enumerate(plans):
                    print("[Prefix SSD] hashing %s: %.2f GiB read, tensors=%d/%d, elapsed=%.1fs" %
                          (plan["root"].name, hashed_bytes[index] / (1 << 30), completed[index],
                           totals[index], now - started), flush=True)
                last_report = now

    if progress:
        print("[Prefix SSD] tensor hashing: models=%d, workers=%d, max_in_flight=%d, read_buffers=%.2f MiB" %
              (len(plans), workers, 2 * workers, workers * buffer_bytes / (1 << 20)), flush=True)
    iterator, pending = iter(tasks), set()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prefix-hash") as pool:
        def submit_next():
            task = next(iterator, None)
            if task is not None:
                pending.add(pool.submit(_hash_tensor, task, buffers, buffer_bytes, report if progress else None))

        for _ in range(2 * workers):
            submit_next()
        try:
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, name, record = future.result()
                    plans[index]["weights"][name] = record
                    if progress:
                        report(index, finished=1)
                for _ in finished:
                    submit_next()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    if progress:
        report(force=True)
    # Catch changes to an already-finished shard while other shards were read.
    for plan in plans:
        for path, before, _, _ in plan["layouts"]:
            if _stat_identity(path.stat()) != _stat_identity(before):
                raise ValueError("Model file changed while building persistent-cache identity")


def _model_files(root):
    files = {}
    # Includes tokenizer, templates, processor configuration and local model code.
    # Shard filenames/packing are not semantic; the tensor names and raw bytes are.
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (any(part.startswith(".") for part in relative.parts) or
                not path.is_file() or path.name == "model.safetensors.index.json"):
            continue
        if (path.suffix in (".json", ".jinja", ".jinja2", ".py", ".model", ".tiktoken") or
                path.name in ("merges.txt", "vocab.txt")):
            files[relative.as_posix()] = _file_record(path)
    return files


def _runtime_record(library_path):
    root = Path(__file__).resolve().parent
    implementations = {name: _file_record(root / name) for name in (
        "qwen35_multimodal_native.py", "llm.py")}
    triton_override = os.environ.get("FASTLLM_CUDA_TRITON_SERVER_SCRIPT")
    if triton_override:
        implementations["triton_server_override"] = _file_record(
            Path(triton_override).expanduser())
    else:
        for directory in (root, root.parent):
            script = directory / "fastllm_triton_server.py"
            if script.is_file():
                implementations["triton_server"] = _file_record(script)
                break
    versions = {}
    for name in ("numpy", "Pillow", "transformers", "tokenizers", "jinja2",
                 "safetensors", "triton", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"library": _file_record(Path(library_path)),
            "python_implementations": implementations, "dependencies": versions}


def build_identity(model_path, execution, library_path, draft_path=None, *, progress=False):
    target_plan = _model_plan(model_path)
    config = target_plan["config"]
    text_config = config.get("text_config", config)
    if (config.get("model_type") not in ("qwen3_5", "qwen3_5_text") or
            not isinstance(text_config, dict) or
            text_config.get("num_experts", 0) or text_config.get("num_local_experts", 0)):
        raise ValueError("Persistent prefix v2 currently requires dense Qwen3.5/Qwen3.8 layout")
    draft_plan = _model_plan(draft_path) if draft_path else None
    if draft_plan and "DFlash2DraftModel" not in draft_plan["config"].get("architectures", []):
        raise ValueError("Persistent prefix v2 supports only DFlash2 draft checkpoints")
    _hash_models([target_plan, draft_plan] if draft_plan else [target_plan], progress=progress)
    weights, files = target_plan["weights"], _model_files(target_plan["root"])
    runtime = _runtime_record(library_path)
    target_weights = {key: value for key, value in weights.items() if not _is_encoder_tensor(key)}
    encoder_weights = {key: value for key, value in weights.items() if _is_encoder_tensor(key)}
    target = {"tensors": target_weights, "files": files}
    encoder = {
        "tensors": encoder_weights,
        "config": {key: config.get(key) for key in (
            "vision_config", "quantization_config", "dtype", "torch_dtype")},
        "files": {key: value for key, value in files.items()
                  if "processor" in key or key.endswith(".py")},
        "runtime": runtime,
        "execution": {key: execution.get(key) for key in (
            "dtype", "atype", "dtype_config", "tp", "pp", "device", "env")},
    }
    # KV storage precision and the draft model do not affect image embeddings.
    encoder["execution"]["env"] = {
        key: value for key, value in execution.get("env", {}).items()
        if "DFLASH" not in key and "MTP" not in key
    }
    draft = None
    if draft_plan:
        draft = {"tensors": draft_plan["weights"], "files": _model_files(draft_plan["root"])}
    manifest = {
        "format_version": FORMAT_VERSION, "target": target, "encoder": encoder,
        "draft": draft, "execution": copy.deepcopy(execution), "runtime": runtime,
    }
    manifest["target_identity"] = _digest(target)
    manifest["encoder_identity"] = _digest(encoder)
    manifest["draft_identity"] = _digest(draft) if draft is not None else "none"
    return manifest


def identity_keys(manifest):
    family = copy.deepcopy(manifest)
    family["execution"].pop("kv_cache_dtype", None)
    return {"identity": _digest(manifest), "family_identity": _digest(family),
            "target_identity": manifest["target_identity"],
            "encoder_identity": manifest["encoder_identity"],
            "draft_identity": manifest["draft_identity"]}


def _execution_record(args):
    fields = ("dtype", "kv_cache_dtype", "atype", "moe_dtype", "moe_atype",
              "tp", "pp", "device", "moe_device", "page_size", "rope_scaling",
              "max_context_length", "chunked_prefill_size", "dtype_config",
              "chat_template", "cuda_embedding", "cuda_shared_expert", "low",
              "enable_amx", "speculative_algorithm", "draft_tokens",
              "speculative_num_draft_tokens")
    execution = {key: getattr(args, key, None) for key in fields}
    for key in ("dtype_config", "chat_template"):
        value = execution[key]
        if value and os.path.isfile(value):
            execution[key] = Path(value).read_text(encoding="utf-8")
    aliases = {"fp16": "float16", "half": "float16", "bf16": "bfloat16",
               "fp8": "fp8_e4m3", "float8": "fp8_e4m3"}
    dtype = str(execution.get("kv_cache_dtype") or "auto").lower()
    execution["kv_cache_dtype"] = aliases.get(dtype, dtype)
    # Include numerical execution overrides but never the derived identity, cache
    # quota/policy, or a draft filesystem path (its content is hashed separately).
    execution["env"] = {key: value for key, value in sorted(os.environ.items())
                        if (key.startswith(("FASTLLM_CUDA_", "FASTLLM_QWEN35_",
                                            "FASTLLM_MULTICUDA_", "FASTLLM_DFLASH_")) or
                            key in ("CUDA_VISIBLE_DEVICES", "FASTLLM_TP", "FASTLLM_PP",
                                    "FASTLLM_GPU_TOKEN_HANDOFF", "FASTLLM_ACTIVATE_NUMA",
                                    "FASTLLM_NUMA_THREADS")) and
                        not key.endswith(("_PATH", "_PYTHON")) and
                        not any(part in key for part in ("DEBUG", "PROFILE", "LOG"))}
    return execution


def _native_capability(native_library):
    function = getattr(native_library, "get_persistent_prefix_cache_version", None)
    if function is None:
        raise RuntimeError("SSD prefix cache requested, but the loaded native runtime lacks persistent-cache v2 support")
    function.argtypes = []
    function.restype = ctypes.c_int
    if function() != FORMAT_VERSION:
        raise RuntimeError("SSD prefix cache requires native persistent-cache v2 with LMCache, SQLite and OpenSSL")


def _write_manifest(root, identity, manifest):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    (root / "v2").mkdir(exist_ok=True, mode=0o700)
    target = root / "v2" / "identities"
    target.mkdir(exist_ok=True, mode=0o700)
    fd, temp = tempfile.mkstemp(prefix=".identity-", dir=target)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target / (identity + ".json"))
        dir_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def configure_persistent_prefix(args, is_qwen35_model, library_path, native_library=None):
    """Validate and export startup settings; use the context manager in loaders."""
    for name in _DERIVED_ENV:
        os.environ.pop(name, None)
    directory = getattr(args, "prefix_cache_dir", None)
    if directory is None:
        directory = os.environ.get("FASTLLM_PREFIX_CACHE_DIR", "")
    if not directory:
        os.environ.pop("FASTLLM_PREFIX_CACHE_DIR", None)
        return None
    algorithm = str(getattr(args, "speculative_algorithm", "") or "").strip().lower()
    if (not is_qwen35_model or algorithm not in ("", "off", "dflash") or
            any(getattr(args, key, None) for key in ("mtp", "dspark", "lora", "ori", "mmproj", "custom"))):
        raise ValueError("Persistent prefix v2 requires dense Qwen3.5/Qwen3.8, optionally DFlash2; MTP/DSpark/LoRA/custom weights are unsupported")
    prefix_cache = str(getattr(args, "prefix_cache", "") or
                       os.environ.get("FASTLLM_PREFIX_CACHE", "true")).lower()
    if prefix_cache in ("0", "off", "false"):
        raise ValueError("Persistent prefix cache requires prefix_cache=true")
    disk_gb = getattr(args, "prefix_cache_disk_gb", None)
    if disk_gb is not None:
        if (not math.isfinite(disk_gb) or disk_gb <= 0 or
                disk_gb > ((1 << 63) - 1) / (1 << 30)):
            raise ValueError("prefix_cache_disk_gb must fit a finite positive signed 64-bit byte quota")
        disk_bytes = int(disk_gb * (1 << 30))
    else:
        try:
            disk_bytes = int(os.environ.get("FASTLLM_PREFIX_CACHE_DISK_BYTES", DEFAULT_DISK_BYTES))
        except ValueError as error:
            raise ValueError("FASTLLM_PREFIX_CACHE_DISK_BYTES must be a positive integer") from error
    if not 0 < disk_bytes <= (1 << 63) - 1:
        raise ValueError("Persistent cache quota must fit a positive signed 64-bit byte count")
    policy = getattr(args, "prefix_cache_restore_policy", None)
    if policy is None:
        policy = os.environ.get("FASTLLM_PREFIX_CACHE_RESTORE_POLICY", "auto")
    if policy not in ("auto", "always", "never"):
        raise ValueError("prefix_cache_restore_policy must be auto, always or never")
    draft_path = getattr(args, "speculative_draft_model_path", "") if algorithm == "dflash" else None
    if algorithm == "dflash" and not draft_path:
        raise ValueError("Persistent DFlash cache requires a draft checkpoint directory")
    _native_capability(native_library)
    root = Path(directory).expanduser().resolve()
    print("[Prefix SSD] hashing model/runtime contents for format v2...", flush=True)
    manifest = build_identity(args.path, _execution_record(args), library_path, draft_path, progress=True)
    keys = identity_keys(manifest)
    _write_manifest(root, keys["identity"], manifest)
    exported = {"FASTLLM_PREFIX_CACHE_" + name.upper(): value for name, value in keys.items()}
    exported.update({"FASTLLM_PREFIX_CACHE_DIR": str(root),
                     "FASTLLM_PREFIX_CACHE_DISK_BYTES": str(disk_bytes),
                     "FASTLLM_PREFIX_CACHE_RESTORE_POLICY": policy,
                     "FASTLLM_PREFIX_CACHE_MANIFEST_VERSION": str(FORMAT_VERSION)})
    os.environ.update(exported)
    print("[Prefix SSD] identity=%s, family=%s, format=v2, directory=%s, quota=%d, restore=%s" %
          (keys["identity"], keys["family_identity"], root, disk_bytes, policy), flush=True)
    return exported


@contextmanager
def persistent_prefix_environment(args, is_qwen35_model, library_path, native_library=None):
    """Scope native constructor env while preserving the user's raw settings."""
    with _ENV_LOCK:
        original = {name: os.environ.get(name) for name in _INPUT_ENV}
        try:
            yield configure_persistent_prefix(args, is_qwen35_model, library_path, native_library)
        finally:
            for name in _DERIVED_ENV:
                os.environ.pop(name, None)
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
