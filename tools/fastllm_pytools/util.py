import argparse
import json
import os
import sys
import subprocess
import glob
import math

def _positive_int(value: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive integer")
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value

def _has_cuda_device() -> bool:
    if os.path.exists("/dev/nvidia0") or os.path.isdir("/proc/driver/nvidia/gpus"):
        return True
    try:
        return subprocess.run(["nvidia-smi", "-L"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=8).returncode == 0
    except Exception:
        return False

def _normalize_mtp_arg(value) -> int:
    try:
        value = int(value)
    except Exception:
        value = 0
    return max(0, value)

def _total_memory_gib() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0

def _uses_cuda_device(device) -> bool:
    if not device:
        return False
    return "cuda" in str(device).lower() or str(device).lower().startswith("cudapp=")

def _uses_multicuda_device(device) -> bool:
    if not device:
        return False
    return "multicuda" in str(device).lower()

def _uses_thread_tp(tp) -> bool:
    if tp is None:
        return False
    spec = str(tp).strip().lower()
    return spec not in ["", "false", "off", "none", "disable"]

def _arg_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in ["", "false", "0", "off", "none", "disable"]

def _cuda_device_count() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                text=True,
                                timeout=8)
        if result.returncode == 0:
            return len([line for line in result.stdout.splitlines() if line.strip()])
    except Exception:
        pass
    try:
        return len(glob.glob("/dev/nvidia[0-9]*"))
    except Exception:
        return 0

def _cuda_driver_device_info(device_ids):
    """Return CUDA-ordinal SM counts and PCI bus ids without creating a context."""
    try:
        import ctypes
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                 ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetAttribute.restype = ctypes.c_int
        driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                               ctypes.c_int]
        driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
        if driver.cuInit(0) != 0:
            return {}

        # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.  CUDA ordinals are used
        # deliberately: their FASTEST_FIRST order can differ from nvidia-smi.
        multiprocessor_count = 16
        result = {}
        for ordinal in device_ids:
            device = ctypes.c_int()
            sm_count = ctypes.c_int()
            bus_id = ctypes.create_string_buffer(32)
            if driver.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
                return {}
            if driver.cuDeviceGetAttribute(ctypes.byref(sm_count),
                                           multiprocessor_count,
                                           device.value) != 0:
                return {}
            pci_bus_id = ""
            if driver.cuDeviceGetPCIBusId(bus_id, len(bus_id), device.value) == 0:
                pci_bus_id = bus_id.value.decode("ascii", errors="ignore")
            result[ordinal] = {
                "sm_count": sm_count.value,
                "pci_bus_id": pci_bus_id,
            }
        return result
    except Exception:
        return {}

def _auto_balanced_cuda_spec(device_ids) -> str:
    infos = _cuda_driver_device_info(device_ids)
    sm_counts = [infos.get(i, {}).get("sm_count", 0) for i in device_ids]
    if len(device_ids) <= 1 or any(count <= 0 for count in sm_counts) or \
            len(set(sm_counts)) == 1:
        return "cuda:" + ",".join(str(i) for i in device_ids)

    divisor = 0
    for count in sm_counts:
        divisor = math.gcd(divisor, count)
    divisor = max(1, divisor)
    return "cuda:" + ",".join(
        f"{device_id}:{sm_count // divisor}"
        for device_id, sm_count in zip(device_ids, sm_counts)
    )

def _parse_cpu_list(value):
    cpus = []
    for item in str(value).strip().split(","):
        if not item:
            continue
        if "-" in item:
            first, last = item.split("-", 1)
            cpus.extend(range(int(first), int(last) + 1))
        else:
            cpus.append(int(item))
    return cpus

def _thread_tp_cuda_device_ids(tp):
    spec = _thread_tp_cuda_device_spec(tp)
    if ":" not in spec:
        return []
    result = []
    for part in spec.split(":", 1)[1].split(","):
        device = part.strip().split(":", 1)[0].split("-", 1)[0]
        if device.isdigit():
            result.append(int(device))
    return result

def _configure_multicuda_worker_affinity(tp, threads):
    """Keep GPU launch workers off the NUMA MoE worker cores when possible."""
    if "FASTLLM_MULTICUDA_WORKER_CPU_BASE" in os.environ:
        return
    device_ids = _thread_tp_cuda_device_ids(tp)
    if len(device_ids) <= 1:
        return
    try:
        node_paths = sorted(glob.glob("/sys/devices/system/node/node[0-9]*"))
        if not node_paths:
            return
        infos = _cuda_driver_device_info(device_ids)
        gpu_nodes = []
        for device_id in device_ids:
            bus_id = infos.get(device_id, {}).get("pci_bus_id", "").lower()
            numa_path = f"/sys/bus/pci/devices/{bus_id}/numa_node"
            if bus_id and os.path.exists(numa_path):
                with open(numa_path, "r", encoding="utf-8") as f:
                    node = int(f.read().strip())
                if node >= 0:
                    gpu_nodes.append(node)
        target_node = max(set(gpu_nodes), key=gpu_nodes.count) if gpu_nodes else 0
        target_node = min(target_node, len(node_paths) - 1)

        used_cpus = set()
        per_node_threads = max(0, int(threads) // len(node_paths))
        for node_path in node_paths:
            with open(os.path.join(node_path, "cpulist"), "r", encoding="utf-8") as f:
                used_cpus.update(_parse_cpu_list(f.read())[:per_node_threads])

        allowed = set(os.sched_getaffinity(0))
        physical_cpus = []
        for cpu_path in glob.glob(os.path.join(node_paths[target_node], "cpu[0-9]*")):
            cpu_id = int(os.path.basename(cpu_path)[3:])
            sibling_path = os.path.join(cpu_path, "topology", "thread_siblings_list")
            with open(sibling_path, "r", encoding="utf-8") as f:
                siblings = _parse_cpu_list(f.read())
            if siblings and cpu_id == min(siblings) and cpu_id in allowed and \
                    cpu_id not in used_cpus:
                physical_cpus.append(cpu_id)
        physical_cpus.sort()

        worker_count = len(device_ids)
        for end in range(len(physical_cpus), worker_count - 1, -1):
            selected = physical_cpus[end - worker_count:end]
            if selected == list(range(selected[0], selected[0] + worker_count)):
                os.environ["FASTLLM_MULTICUDA_WORKER_CPU_BASE"] = str(selected[0])
                print("[tp] MultiCuda launch workers use reserved CPU(s): " +
                      ",".join(str(cpu) for cpu in selected))
                return
    except Exception:
        # Unbound C++ workers remain the safe fallback on unusual topologies.
        return

def _first_thread_tp_cuda_device(tp) -> str:
    spec = str(tp or "").strip()
    lower = spec.lower()
    if lower in ["", "auto", "true", "on"]:
        return "cuda:0"
    if lower.isdigit():
        return "cuda:0"

    first_part = spec.split(",")[0].strip()
    first_lower = first_part.lower()
    if first_lower.startswith("multicuda:") or first_lower.startswith("cuda:"):
        first_part = first_part.split(":", 1)[1].strip()
    elif first_lower in ["multicuda", "cuda"]:
        return "cuda:0"

    device_id = ""
    for ch in first_part:
        if ch.isdigit():
            device_id += ch
        elif device_id:
            break
    return "cuda:" + (device_id if device_id != "" else "0")

def _thread_tp_cuda_device_spec(tp) -> str:
    spec = str(tp or "").strip()
    lower = spec.lower()
    if lower in ["", "false", "off", "none", "disable"]:
        return ""
    if lower in ["auto", "true", "on"]:
        count = _cuda_device_count()
        if count <= 1:
            return "cuda:0"
        return _auto_balanced_cuda_spec(list(range(count)))
    if lower.isdigit():
        requested = int(lower)
        if requested == 0:
            return "cuda:0"
        count = _cuda_device_count()
        if count > 0:
            requested = min(requested, count)
        # A numeric TP request means equal logical ranks. Model-specific split
        # units (for example DeepSeek-V4's eight output groups) can quantize a
        # small hardware ratio back to equal attention shards while leaving FFN
        # shards asymmetric, which is slower than a consistently equal split.
        # Explicit ratios remain available, while `--tp auto` keeps SM-count
        # balancing for users who request topology-based weighting.
        return "cuda:" + ",".join(str(i) for i in range(requested))

    if lower.startswith("multicuda:") or lower.startswith("cuda:"):
        spec = spec.split(":", 1)[1].strip()
    elif lower in ["multicuda", "cuda"]:
        return "cuda:0"
    return "cuda:" + spec

def _thread_tp_cuda_device_count(tp) -> int:
    spec = _thread_tp_cuda_device_spec(tp)
    if spec == "":
        return 0
    payload = spec.split(":", 1)[1] if ":" in spec else spec
    return len([item for item in payload.split(",") if item.strip() != ""])

def _normalize_thread_tp_arg(tp) -> str:
    spec = str(tp or "").strip()
    lower = spec.lower()
    if lower in ["", "false", "off", "none", "disable"]:
        return spec
    return _thread_tp_cuda_device_spec(spec)

def _explain_thread_tp_arg(original, normalized):
    spec = str(original or "").strip()
    if spec == "":
        return
    lower = spec.lower()
    if lower in ["false", "off", "none", "disable"]:
        print(f"[tp] --tp {spec}: thread-level tensor parallel is disabled")
        return
    normalized = str(normalized or "").strip()
    if lower == "0":
        print(f"[tp] --tp 0: interpreted as CUDA device id 0 => {normalized}")
    elif lower.isdigit():
        print(f"[tp] --tp {spec}: interpreted as using {int(lower)} CUDA device(s) => {normalized}")
    elif lower in ["auto", "true", "on"]:
        print(f"[tp] --tp {spec}: automatically using detected CUDA devices => {normalized}")
    elif lower.startswith("cuda:") or lower.startswith("multicuda:"):
        print(f"[tp] --tp {spec}: normalized explicit device list => {normalized}")
    else:
        print(f"[tp] --tp {spec}: completed as CUDA device list => {normalized}")

def apply_page_size_default(args):
    if (getattr(args, "page_size", -1) <= 0 and
        (_uses_multicuda_device(getattr(args, "device", "")) or
         _uses_multicuda_device(getattr(args, "moe_device", "")))):
        try:
            args.page_size = int(os.environ.get("FASTLLM_MULTICUDA_PAGE_SIZE", "16"))
        except:
            args.page_size = 16
    return args

def apply_prefix_cache_env(args):
    prefix_cache = getattr(args, "prefix_cache", "")
    if (prefix_cache != ""):
        os.environ["FASTLLM_PREFIX_CACHE"] = str(prefix_cache)

    env_args = [
        ("prefix_cache_snapshot_interval_pages", "FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES"),
        ("prefix_cache_snapshot_max_per_request", "FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST"),
        ("prefix_cache_snapshot_max_records", "FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS"),
    ]
    for arg_name, env_name in env_args:
        value = getattr(args, arg_name, -1)
        try:
            value = int(value)
        except:
            value = -1
        if (value > 0):
            os.environ[env_name] = str(value)
    return args

def _fastllm_env_flag_enabled(name: str, fallback_name: str = "") -> bool:
    value = os.environ.get(name)
    if value is None and fallback_name:
        value = os.environ.get(fallback_name)
    if value is None:
        return False
    return str(value).strip().lower() in ["1", "true", "on", "yes"]

def _configure_qwen35_auto_fast_paths(args, is_qwen35_model: bool, mtp: int):
    """Select the tested Qwen3.5 CUDA TP fast path without deployment env vars.

    Environment variables remain authoritative debugging overrides.  The
    automatic path is deliberately limited to the configuration for which the
    scheduler can safely fall back row-by-row: CUDA thread TP, no MTP, and no
    low-memory mode.
    """
    tp_arg = getattr(args, "tp", "")
    device = getattr(args, "device", "")
    eligible = (is_qwen35_model and mtp == 0 and
                not bool(getattr(args, "low", False)) and
                _uses_thread_tp(tp_arg) and _uses_cuda_device(device))

    if eligible and "FASTLLM_CUDA_GRAPH" not in os.environ:
        os.environ["FASTLLM_CUDA_GRAPH"] = "1"

    handoff_env = "FASTLLM_GPU_TOKEN_HANDOFF"
    if eligible and handoff_env not in os.environ:
        os.environ[handoff_env] = "1"

    graph_enabled = _fastllm_env_flag_enabled("FASTLLM_CUDA_GRAPH")
    handoff_enabled = _fastllm_env_flag_enabled(handoff_env)
    if is_qwen35_model and (graph_enabled or handoff_enabled):
        # Qwen3.5 handoff keeps sampled tokens on device, and graph replay also
        # benefits from avoiding a host embedding round trip.
        args.cuda_embedding = True

    graph_batch_env = "FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH"
    requested_batch = int(getattr(args, "max_batch", -1) or -1)
    if (eligible and graph_enabled and requested_batch > 0 and
            graph_batch_env not in os.environ):
        os.environ[graph_batch_env] = str(min(requested_batch, 64))

    if eligible:
        print(
            "[Fastllm] Qwen3.5 auto fast paths: cuda_graph=%s, "
            "gpu_token_handoff=%s, cuda_embedding=%s, graph_max_batch=%s."
            % (
                "on" if graph_enabled else "off",
                "on" if handoff_enabled else "off",
                "on" if bool(getattr(args, "cuda_embedding", False)) else "off",
                os.environ.get(graph_batch_env, "default"),
            ),
            flush=True,
        )
    return args

def _triton_python_works(python: str) -> bool:
    if not python or not os.path.isfile(python) or not os.access(python, os.X_OK):
        return False
    try:
        return subprocess.run(
            [python, "-c", "import triton"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).returncode == 0
    except Exception:
        return False

def _find_triton_python() -> str:
    if not sys.executable:
        return ""
    # Keep the current virtualenv's Python path instead of resolving its
    # symlink: resolving it may bypass pyvenv.cfg and hide installed packages.
    current_python = os.path.abspath(os.path.expanduser(sys.executable))
    return current_python if _triton_python_works(current_python) else ""

def _configure_triton_compiler_python() -> str:
    python_env_name = "FASTLLM_CUDA_TRITON_PYTHON"
    triton_env_name = "FASTLLM_CUDA_TRITON"
    detected = _find_triton_python()
    if detected:
        os.environ[python_env_name] = detected
        os.environ[triton_env_name] = "1"
        print(
            "[Fastllm] Triton enabled with the current Python environment: %s"
            % detected,
            flush=True,
        )
    else:
        os.environ.pop(python_env_name, None)
        os.environ[triton_env_name] = "0"
        current_python = sys.executable or "unknown"
        print(
            "[Fastllm] Triton is unavailable in the current Python "
            "environment (%s); --triton has been disabled and built-in "
            "CUDA will be used." % current_python,
            flush=True,
        )
    return detected

def _is_moe_architecture(architecture: str, model_type: str = "", text_model_type: str = "") -> bool:
    return (architecture in [
        "DeepseekV3ForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "MiniMaxM1ForCausalLM",
        "MiniMaxText01ForCausalLM",
        "HunYuanMoEV1ForCausalLM",
        "Ernie4_5_MoeForCausalLM",
        "PanguProMoEForCausalLM",
        "Glm4MoeForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Qwen3NextForCausalLM",
        "MiniMaxM2ForCausalLM",
        "HYV3ForCausalLM",
        "LagunaForCausalLM",
        "KimiK3ForConditionalGeneration",
    ] or model_type in [
        "deepseek_v4", "glm_moe_dsa", "qwen3_5_moe", "hy_v3", "laguna",
        "kimi_k3",
    ] or text_model_type == "qwen3_5_moe_text")

def _prefers_multicuda_tp(architecture: str, model_type: str = "") -> bool:
    return (architecture == "DeepseekV4ForCausalLM" or
            model_type == "deepseek_v4")

def _prefers_laguna_hybrid_tp(architecture: str, model_type: str = "") -> bool:
    return (architecture == "LagunaForCausalLM" or
            model_type == "laguna")

def make_normal_parser(des: str, add_help = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = des, add_help = add_help)
    parser.add_argument('model', nargs='?', help = '模型路径，fastllm模型文件或HF模型文件夹或配置文件')
    parser.add_argument('-p', '--path', type = str, required = False, default = '', help = '模型路径，fastllm模型文件或HF模型文件夹')
    parser.add_argument('-t', '--threads', type = int, default = -1,  help = '线程数量')
    parser.add_argument('-l', '--low', action = 'store_true', help = '是否使用低内存模式')
    parser.add_argument('--dtype', type = str, default = "auto", help = '权重类型（读取HF模型时有效；auto默认使用float16，带缩放因子的FP8源权重保持FP8）')
    parser.add_argument('--moe_dtype', type = str, default = "", help = 'MOE层使用的权重类型（读取HF模型时有效）')
    parser.add_argument('--moe_atype', type = str, default = "", help = 'MOE层激活类型，可使用auto、float32、float16或bfloat16')
    parser.add_argument('--atype', type = str, default = "auto", help = '推理类型，可使用float32或float16')
    parser.add_argument('--kv_cache_dtype', type = str, default = "auto", help = 'KV Cache类型，可使用auto、float16、bfloat16或fp8_e4m3')
    parser.add_argument('--cuda_embedding', action = 'store_true', help = '在cuda上进行embedding')
    parser.add_argument('--kv_cache_limit', type = str, default = "auto",  help = 'kv缓存最大使用量')
    parser.add_argument('--max_batch', type = int, default = -1,  help = '每次最多同时推理的询问数量')
    parser.add_argument('--chunked_prefill_size', type = int, default = -1, help = '分块 prefill 的切片大小（首块与后续块相同），如 8192')
    parser.add_argument('--device', type = str, help = '使用的设备')
    parser.add_argument('--tp', type = str, default = "", help = '线程级张量并行设备；裸数字X表示使用前X张卡，0表示0号卡，也可写 0,1 或 auto')
    parser.add_argument('--moe_device', type = str, default = "", help = 'moe使用的设备')
    parser.add_argument('--moe_device_layers', type = int, default = -1, help = '后面多少层moe使用moe_device，-1表示全部moe层使用moe_device')
    parser.add_argument('--moe_experts', type = int, default = -1, help = 'moe使用的专家数')
    parser.add_argument("--cache_history", type = str, default = "", help = "缓存历史对话")
    parser.add_argument("--cache_fast", type = str, default = "", help = "是否启用快速缓存（会消耗一定显存）")
    parser.add_argument("--enable_thinking", type = str, default = "", help = "是否开启硬思考开关（需要模型支持）")
    parser.add_argument("--cuda_shared_expert", "--cuda_se", type = str, default = "true", help = "是否使用cuda来执行共享专家")
    parser.add_argument("--enable_amx", "--amx", type = str, default = "false", help = "是否开启amx加速")
    parser.add_argument("--tokens", type = int, default = -1, help = "设置总的token数量（用于计算paged cache的最大页数）")
    parser.add_argument("--page_size", type = int, default = -1, help = "设置paged cache每页的大小（token数），默认multicuda为16，其它设备使用后端默认值")
    parser.add_argument("--prefix_cache", "--prefix-cache", dest = "prefix_cache", type = str, default = "",
                        help = "是否启用前缀缓存（true/false），对应 FASTLLM_PREFIX_CACHE")
    parser.add_argument("--prefix_cache_snapshot_interval_pages", "--prefix-cache-snapshot-interval-pages",
                        dest = "prefix_cache_snapshot_interval_pages", type = int, default = -1,
                        help = "前缀缓存快照间隔页数，对应 FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES")
    parser.add_argument("--prefix_cache_snapshot_max_per_request", "--prefix-cache-snapshot-max-per-request",
                        dest = "prefix_cache_snapshot_max_per_request", type = int, default = -1,
                        help = "单请求最多保留的前缀缓存快照数，对应 FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST")
    parser.add_argument("--prefix_cache_snapshot_max_records", "--prefix-cache-snapshot-max-records",
                        dest = "prefix_cache_snapshot_max_records", type = int, default = -1,
                        help = "全局最多保留的前缀缓存快照数，对应 FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS")
    parser.add_argument("--gpu_mem_ratio", type = float, default = 0.9, help = "GPU显存使用比例，如0.9表示使用90%%的显存")
    parser.add_argument("--cuda_slab", type = int, default = 0, help = "CUDA模型权重slab大小（MB），0表示关闭")
    parser.add_argument("--mtp", type = int, default = 0, help = "Qwen3.5 MTP每步生成的draft token数，0表示关闭（默认），当前最大8")
    parser.add_argument("--dspark", type = int, default = 0,
                        help = "启用模型内置 DSpark，并指定每轮 draft token 数；例如 --dspark 7")
    parser.add_argument("--speculative_algorithm", "--speculative-algorithm",
                        dest = "speculative_algorithm", type = str, default = "",
                        help = "投机解码算法；当前支持 dspark")
    parser.add_argument("--speculative_draft_model_path", "--speculative-draft-model-path", "--dspark_model",
                        dest = "speculative_draft_model_path", type = str, default = "",
                        help = "DSpark draft model 的 Hugging Face 目录")
    parser.add_argument("--speculative_dspark_block_size", "--speculative-dspark-block-size",
                        dest = "speculative_dspark_block_size", type = int, default = -1,
                        help = "DSpark block size；默认读取 draft config")
    parser.add_argument("--speculative_dspark_confidence_threshold", "--speculative-dspark-confidence-threshold",
                        dest = "speculative_dspark_confidence_threshold", type = float, default = 0.5,
                        help = "DSpark confidence 前缀阈值，范围 [0,1]；0 表示固定验证完整 block")
    parser.add_argument("--triton", action = "store_true", help = "启用Triton CUDA算子")
    
    parser.add_argument('--custom', type = str, default = "", help = '指定描述自定义模型的python文件')
    parser.add_argument('--lora', type = str, default = "", help = '指定lora路径')
    parser.add_argument('--cache_dir', type = str, default = "", help = '指定缓存模型文件的路径')
    parser.add_argument('--dtype_config', type = str, default = "", help = '指定权重类型配置文件')
    parser.add_argument('--ori', type = str, default = "", help = '原始模型权重，读取GGUF文件时可以使用')

    parser.add_argument('--tool_call_parser', type = str, default = "auto", help = '使用的tool_call_parser类型')
    parser.add_argument('--chat_template', type = str, default = "", help = '使用的chat_template文件')

    return parser

def add_server_args(parser):
    parser.add_argument("--model_name", type = str, default = '', help = "部署的模型名称, 调用api时会进行名称核验")
    parser.add_argument("--host", type = str, default="0.0.0.0", help = "API server host")
    parser.add_argument("--port", type = int, default = 8080, help = "API server port")
    parser.add_argument("--api_key", type = str, default = "", help = "API Key")
    parser.add_argument("--max_context_length", "--max-context-length", dest = "max_context_length",
                        type = _positive_int, default = -1,
                        help = "限制单会话输入和输出合计的最大token数；默认取模型上限和KV Cache总容量的较小值")
    parser.add_argument("--temperature", type = float, default = None, help = "覆盖服务端默认 temperature，未指定则使用模型默认值")
    parser.add_argument("--top_p", type = float, default = None, help = "覆盖服务端默认 top_p，未指定则使用模型默认值")
    parser.add_argument("--top_k", type = int, default = None, help = "覆盖服务端默认 top_k，未指定则使用模型默认值")
    parser.add_argument("--repeat_penalty", "--repetition_penalty", dest = "repeat_penalty",
                        type = float, default = None, help = "覆盖服务端默认 repeat_penalty，未指定则使用模型默认值")
    parser.add_argument("--think", type = str, default = "false", help="if <think> lost")
    parser.add_argument("--hide_input", action = 'store_true', help = "不显示请求信息")
    parser.add_argument("--dev_mode", action = 'store_true', help = "开发模式, 启用后能够获取对话列表并主动停止")
    parser.add_argument(
        "--startup-progress",
        choices = ["off", "ndjson"],
        default = "off",
        help = "启动进度输出格式；ndjson 会向 stderr 输出 FTLLM_PROGRESS 事件",
    )

def expand_cudapp_device(device_str):
    if not device_str or not device_str.startswith("cudapp="):
        return device_str
    spec = device_str[len("cudapp="):]
    if ',' in spec:
        raw_device_ids = [device_id.strip() for device_id in spec.split(',')]
        if any(device_id == '' for device_id in raw_device_ids):
            raise ValueError(f"invalid cudapp device list: {spec}")
        device_ids = [int(device_id) for device_id in raw_device_ids]
        return str({f'cuda:{device_id}': 1 for device_id in device_ids})
    if ':' in spec:
        weights = [int(w) for w in spec.split(':')]
    else:
        n = int(spec)
        weights = [1] * n
    return str({f'cuda:{i}': w for i, w in enumerate(weights)})

def make_normal_llm_model(args, startup_progress = None):
    if startup_progress is not None:
        startup_progress.progress("initializing", 0, 1)
    if (args.model and args.model != ''):
        if (args.model.endswith(".json") and os.path.exists(args.model)):
            with open(args.model, "r", encoding = "utf-8") as file:
                args_config = json.load(file)
                for it in args_config.keys():
                    if (it == "FASTLLM_ACTIVATE_NUMA" or it == "FASTLLM_NUMA_THREADS"):
                        os.environ[it] = str(args_config[it])
                    setattr(args, it, args_config[it])

    user_set_device = bool(args.device and args.device != "")
    user_set_moe_device = bool(args.moe_device and args.moe_device != "")
    mtp = _normalize_mtp_arg(getattr(args, "mtp", 0))
    args.mtp = mtp
    speculative_algorithm = str(
        getattr(args, "speculative_algorithm", "") or "").strip().lower()
    speculative_draft_path = str(
        getattr(args, "speculative_draft_model_path", "") or "").strip()
    dspark_tokens = int(getattr(args, "dspark", 0) or 0)
    if dspark_tokens < 0:
        raise ValueError("--dspark must be >= 0")
    if dspark_tokens > 0 and not speculative_algorithm:
        speculative_algorithm = "dspark"
    if speculative_draft_path and not speculative_algorithm:
        speculative_algorithm = "dspark"
    if speculative_algorithm and speculative_algorithm != "dspark":
        raise ValueError("--speculative_algorithm currently only supports dspark")
    if (speculative_algorithm == "dspark" and not speculative_draft_path and
            dspark_tokens <= 0):
        raise ValueError(
            "DSpark requires either --dspark N for an embedded checkpoint or "
            "--speculative_draft_model_path")
    if speculative_draft_path:
        os.environ.pop("FASTLLM_DSPARK_TOKENS", None)
        speculative_draft_path = os.path.abspath(
            os.path.expanduser(speculative_draft_path))
        draft_config_path = os.path.join(speculative_draft_path, "config.json")
        if not os.path.isfile(draft_config_path):
            raise ValueError(
                "DSpark draft directory has no config.json: %s" %
                speculative_draft_path)
        with open(draft_config_path, "r", encoding = "utf-8") as file:
            draft_config = json.load(file)
        draft_architectures = draft_config.get("architectures", [])
        if "DSparkDraftModel" not in draft_architectures:
            raise ValueError(
                "draft checkpoint is not DSparkDraftModel: %s" %
                speculative_draft_path)
        configured_block = int(draft_config.get("block_size", 0))
        requested_block = int(
            getattr(args, "speculative_dspark_block_size", -1))
        if dspark_tokens > 0:
            requested_block = dspark_tokens
        if requested_block > 0 and requested_block != configured_block:
            raise ValueError(
                "FastLLM currently requires the DSpark runtime block size "
                "to match the checkpoint (requested=%d, checkpoint=%d)" %
                (requested_block, configured_block))
        os.environ["FASTLLM_DSPARK_MODEL_PATH"] = speculative_draft_path
        confidence_threshold = float(getattr(
            args, "speculative_dspark_confidence_threshold", 0.5))
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "--speculative_dspark_confidence_threshold must be in [0, 1]")
        os.environ["FASTLLM_DSPARK_CONFIDENCE_THRESHOLD"] = str(
            confidence_threshold)
        args.speculative_draft_model_path = speculative_draft_path
        args.speculative_algorithm = "dspark"
    else:
        os.environ.pop("FASTLLM_DSPARK_MODEL_PATH", None)
        if dspark_tokens > 0:
            os.environ["FASTLLM_DSPARK_TOKENS"] = str(dspark_tokens)
            confidence_threshold = float(getattr(
                args, "speculative_dspark_confidence_threshold", 0.5))
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError(
                    "--speculative_dspark_confidence_threshold must be in [0, 1]")
            os.environ["FASTLLM_DSPARK_CONFIDENCE_THRESHOLD"] = str(
                confidence_threshold)
            args.speculative_algorithm = "dspark"
        else:
            os.environ.pop("FASTLLM_DSPARK_TOKENS", None)
            os.environ.pop("FASTLLM_DSPARK_CONFIDENCE_THRESHOLD", None)

    usenuma = False
    try:
        from ftllm.env import env
        usenuma = env.use_numas
    except:
        pass
    if (args.path == '' or args.path is None):
        args.path = args.model
    if (args.path == '' or args.path is None):
        print("model can't be empty. (Example: ftllm run MODELNAME)")
        exit(0)
    if (_arg_enabled(getattr(args, "triton", False))):
        if not _configure_triton_compiler_python():
            args.triton = False
    if not(os.path.exists(args.path)):
        if (hasattr(args, "model_name") and args.model_name == ''):
            args.model_name = args.path
        from ftllm.download import HFDNormalDownloader
        from ftllm.download import find_metadata
        from ftllm.download import search_model
        if (not(os.path.exists(get_fastllm_cache_path(args.path, args.cache_dir))) and not(find_metadata(args.path))):
            print("Can't find model \"" + args.path + "\", try to find similar one.")
            search_result = search_model(args.path)
            if (len(search_result) > 0):
                args.path = search_result[0]["id"]
                print("Replace model to \"" + args.path + "\"")
            else:
                exit(0)
        downloader = HFDNormalDownloader(args.path, local_dir = get_fastllm_cache_path(args.path, args.cache_dir))
        downloader.run()
        args.path = str(downloader.local_dir)
    
    config_path = os.path.join(args.path, "config.json")
    if (not(os.path.exists(config_path)) and args.ori != "" and os.path.exists(os.path.join(args.ori, "config.json"))):
        config_path = os.path.join(args.ori, "config.json")
    is_moe_model = False
    is_thread_tp_moe_model = False
    is_multicuda_tp_model = False
    is_laguna_hybrid_tp_model = False
    is_laguna_model = False
    is_qwen35_model = False
    if (os.path.exists(config_path)):
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)
            architecture = config["architectures"][0]
            model_type = config.get("model_type", "")
            is_laguna_model = (architecture == 'LagunaForCausalLM' or
                                model_type == 'laguna')
            text_model_type = ""
            if isinstance(config.get("text_config"), dict):
                text_model_type = config["text_config"].get("model_type", "")
            is_qwen35_model = (
                architecture in (
                    "Qwen3_5ForConditionalGeneration",
                    "Qwen3_5MoeForConditionalGeneration",
                ) or
                model_type in ("qwen3_5", "qwen3_5_moe") or
                text_model_type in ("qwen3_5_text", "qwen3_5_moe_text")
            )
            if speculative_algorithm == "dspark":
                if speculative_draft_path:
                    if (architecture != "KimiK3ForConditionalGeneration" and
                            model_type != "kimi_k3"):
                        raise ValueError(
                            "external DSpark draft checkpoints currently target "
                            "Kimi-K3, got architecture=%s model_type=%s" %
                            (architecture, model_type))
                else:
                    is_deepseek_v4 = (
                        architecture in ("DeepseekV4ForCausalLM",
                                         "DeepSeekV4ForCausalLM") or
                        model_type == "deepseek_v4")
                    if not is_deepseek_v4:
                        raise ValueError(
                            "--dspark N requires a DeepSeek-V4 checkpoint with "
                            "embedded mtp.* DSpark weights, got architecture=%s "
                            "model_type=%s" % (architecture, model_type))
                    checkpoint_block = int(config.get(
                        "dspark_block_size", 0) or 0)
                    target_layers = config.get("dspark_target_layer_ids", [])
                    noise_token = int(config.get(
                        "dspark_noise_token_id", -1) or -1)
                    if (checkpoint_block <= 0 or not target_layers or
                            noise_token < 0):
                        raise ValueError(
                            "DeepSeek-V4 checkpoint is missing embedded DSpark "
                            "configuration")
                    if dspark_tokens < checkpoint_block:
                        raise ValueError(
                            "--dspark must be at least the checkpoint training "
                            "block size (requested=%d, checkpoint=%d)" %
                            (dspark_tokens, checkpoint_block))
            is_moe_model = _is_moe_architecture(architecture, model_type, text_model_type)

            is_step3p5 = (architecture == 'Step3p5ForCausalLM' or
                          model_type == 'step3p5' or
                          text_model_type == 'step3p5')
            is_step3p7 = (architecture == 'Step3p7ForConditionalGeneration' or
                          model_type == 'step3p7')
            if is_step3p5:
                is_thread_tp_moe_model = True
                if (args.cache_history == ""):
                    args.cache_history = "true"
                if (args.moe_device == "" and not(args.device and args.device != "")):
                    total_mem_gib = _total_memory_gib()
                    can_hold_cpu_moe = total_mem_gib >= 220.0
                    if (_has_cuda_device() and can_hold_cpu_moe):
                        args.device = "cuda"
                        args.moe_device = "cpu"
                    else:
                        args.device = "cpu"
                        args.moe_device = "disk"
                if (args.chunked_prefill_size <= 0):
                    args.chunked_prefill_size = 128
                if (args.tokens <= 0 and not is_step3p7 and not _uses_thread_tp(getattr(args, "tp", ""))):
                    args.tokens = 32768

            if (architecture == 'Qwen3ForCausalLM' or architecture == 'Qwen3MoeForCausalLM' or
                architecture == 'DeepseekV4ForCausalLM' or model_type == 'deepseek_v4' or
                architecture == 'LagunaForCausalLM' or model_type == 'laguna' or
                architecture == 'Qwen3_5MoeForConditionalGeneration' or
                model_type == 'qwen3_5_moe' or text_model_type == 'qwen3_5_moe_text' or
                architecture == 'Glm4MoeForCausalLM' or architecture == 'GlmMoeDsaForCausalLM' or
                architecture == 'HYV3ForCausalLM' or model_type == 'glm_moe_dsa' or
                model_type == 'hy_v3' or
                architecture == 'KimiK3ForConditionalGeneration' or
                model_type == 'kimi_k3'):
                if (args.enable_thinking == ""):
                    args.enable_thinking = "true"
            if ((architecture == 'Qwen3_5ForConditionalGeneration' or
                 model_type == 'qwen3_5' or text_model_type == 'qwen3_5_text') and
                (not user_set_device) and _has_cuda_device()):
                args.device = "cuda"
            if (architecture == 'Qwen3MoeForCausalLM' or model_type == 'qwen3_moe' or
                architecture == 'HYV3ForCausalLM' or model_type == 'hy_v3'):
                is_thread_tp_moe_model = True
            if (architecture == 'Qwen3_5MoeForConditionalGeneration' or
                model_type == 'qwen3_5_moe' or text_model_type == 'qwen3_5_moe_text'):
                is_thread_tp_moe_model = True
            if (architecture == 'MiniMaxM2ForCausalLM' or model_type == 'minimax_m2'):
                is_thread_tp_moe_model = True
            if (_prefers_laguna_hybrid_tp(architecture, model_type)):
                is_laguna_hybrid_tp_model = True
            if (_prefers_multicuda_tp(architecture, model_type)):
                is_multicuda_tp_model = True
            if (is_moe_model):
                if (args.cache_history == ""):
                    args.cache_history = "true"
                if ((not(args.device and args.device != ""))):
                    args.device = "cuda"
                    if (not user_set_moe_device):
                        args.moe_device = "cpu"
                        if (usenuma):
                            args.moe_device = "numa"
            if ("quantization_config" in config):
                quantization_config = config["quantization_config"]
                try:
                    if (args.dtype == "auto" and quantization_config['bits'] == 4 and quantization_config['group_size']):
                        args.dtype = "int4g" + str(quantization_config["group_size"])
                except:
                    pass
                try:
                    if (args.dtype == "auto" and quantization_config['quant_method'] == "fp8" and 
                        (quantization_config['fmt'] == "e4m3" or quantization_config['fmt'] == "float8_e4m3fn")):
                        args.dtype = "fp8_e4m3"
                except:
                    pass
                try:
                    if (args.path.lower().find("-fp8") != -1):
                        args.dtype = "fp8_e4m3";
                except:
                    pass
        except:
            if speculative_algorithm:
                raise
            pass
    raw_tp_arg = getattr(args, "tp", "")
    normalized_tp_arg = _normalize_thread_tp_arg(raw_tp_arg)
    if raw_tp_arg != "" and normalized_tp_arg != raw_tp_arg:
        args.tp = normalized_tp_arg
    _explain_thread_tp_arg(raw_tp_arg, normalized_tp_arg)

    if (_uses_thread_tp(getattr(args, "tp", ""))):
        tp_device = _first_thread_tp_cuda_device(args.tp)
        cuda_spec = _thread_tp_cuda_device_spec(args.tp)
        if is_multicuda_tp_model:
            target_spec = cuda_spec
            if len(_thread_tp_cuda_device_ids(cuda_spec)) > 1:
                target_spec = "multicuda:" + cuda_spec.split(":", 1)[1]
            if user_set_device and str(args.device).strip().lower() != target_spec:
                print("[tp] DeepSeek-V4 --tp overrides the main device "
                      f"{args.device} => {target_spec}")
            # DeepSeek-V4 executes multi-device tensor parallel through the
            # MultiCuda executor. A one-device --tp remains a normal CUDA path.
            args.device = target_spec
            if (not user_set_moe_device):
                args.moe_device = target_spec
        elif is_laguna_hybrid_tp_model:
            multicuda_spec = "multicuda:" + cuda_spec.split(":", 1)[1]
            if (not user_set_device):
                args.device = tp_device
            if (not user_set_moe_device):
                args.moe_device = multicuda_spec
        else:
            if (not user_set_device):
                args.device = tp_device
            if (not user_set_moe_device):
                args.moe_device = (_thread_tp_cuda_device_spec(args.tp) or args.device) if is_thread_tp_moe_model else args.device
    if ((is_multicuda_tp_model or is_laguna_hybrid_tp_model) and
            _uses_multicuda_device(args.moe_device)):
        # Large MoE checkpoints have tens of thousands of routed-expert tensors.
        # Pack their TP shards into slabs to avoid exhausting the CUDA driver's
        # allocation-count limit before device memory is full.
        if args.cuda_slab <= 0:
            # Laguna TP=4 owns 64 experts/rank.  Its merged gate-up and down
            # sources are 6 MiB and 3 MiB, and one layer is exactly 576 MiB per
            # rank.  A 96 MiB slab packs both shapes and the whole layer without
            # tail waste, which is required to fit the FP8 checkpoint on 32 GB
            # Blackwell cards.  Other layouts retain the established default.
            args.cuda_slab = (96 if is_laguna_hybrid_tp_model and
                              _thread_tp_cuda_device_count(args.tp) == 4
                              else 256)
    if ((args.device and args.device.find("numa") != -1) or args.moe_device.find("numa") != -1 or
        (args.device and args.device.find("tfacc") != -1) or args.moe_device.find("tfacc") != -1):
        os.environ["FASTLLM_ACTIVATE_NUMA"] = "ON"
        if (args.threads == -1):
            try:
                import glob
                numa_nodes = sorted(glob.glob("/sys/devices/system/node/node[0-9]*"))
                numa_count = len(numa_nodes)
                if numa_count > 0:
                    physical_cores_per_numa = set()
                    for entry in os.listdir(numa_nodes[0]):
                        if entry.startswith("cpu") and entry[3:].isdigit():
                            siblings_path = os.path.join(numa_nodes[0], entry, "topology", "thread_siblings_list")
                            if os.path.exists(siblings_path):
                                with open(siblings_path, "r") as f:
                                    physical_cores_per_numa.add(f.read().strip())
                    cpus_per_numa = len(physical_cores_per_numa) if physical_cores_per_numa else 1
                    args.threads = max(1, numa_count * (cpus_per_numa - 4))
                else:
                    args.threads = 4
            except:
                args.threads = 4
    if (args.threads == -1):
        try:
            available_cores = len(os.sched_getaffinity(0))  # 参数 0 表示当前进程
            args.threads = max(1, min(32, available_cores - 2))
        except:
            args.threads = max(1, min(32, os.cpu_count() - 2))
    if is_multicuda_tp_model and _uses_multicuda_device(args.device) and \
            "numa" in str(args.moe_device).lower():
        _configure_multicuda_worker_affinity(args.tp, args.threads)
    if ("FT_THREADS" not in os.environ and "FASTLLM_NUMA_THREADS" not in os.environ):
        os.environ["FT_THREADS"] = str(args.threads)
    atype_was_auto = (args.atype == "auto")
    if (args.atype == "auto"):
        if (args.device in ["cpu", "numa", "tfacc"]):
            args.atype = "float32"
    if (args.moe_device == ""):
        args.moe_device = args.device
    raw_main_device = str(args.device or "").strip()
    os.environ["FASTLLM_CUDAPP_SERIAL"] = "1" if raw_main_device.lower().startswith("cudapp=") else "0"

    tp_arg = getattr(args, "tp", "")
    if (tp_arg != ""):
        os.environ["FASTLLM_TP"] = tp_arg
        if (_uses_thread_tp(tp_arg)):
            if (atype_was_auto):
                args.atype = "bfloat16" if is_laguna_model else "float16"
            if (not(args.device and args.device != "")):
                args.device = _first_thread_tp_cuda_device(tp_arg)
    if (args.moe_atype == "" and is_moe_model and args.dtype == "fp8_e4m3"):
        if (is_laguna_model and _uses_thread_tp(tp_arg)):
            args.moe_atype = "bfloat16"
        elif (_uses_cuda_device(args.moe_device)):
            args.moe_atype = "float16"
        elif (_uses_thread_tp(tp_arg)):
            args.moe_atype = "bfloat16"
    if (args.device and args.device != ""):
        expanded = expand_cudapp_device(args.device)
        if expanded != args.device:
            print(f"[device] cudapp expand: {args.device} => {expanded}", flush=True)
            args.device = expanded
    if (args.moe_device and args.moe_device != ""):
        args.moe_device = expand_cudapp_device(args.moe_device)
    _configure_qwen35_auto_fast_paths(args, is_qwen35_model, mtp)
    from ftllm import llm
    llm.set_moe_device_layers(-1)
    if (args.device and args.device != ""):
        try:
            import ast
            device_map = ast.literal_eval(args.device)
            if (isinstance(device_map, list) or isinstance(device_map, dict)):
                llm.set_device_map(device_map)
            else:
                llm.set_device_map(args.device)
        except:
            llm.set_device_map(args.device)
    if (args.moe_device and args.device != ""):
        try:
            import ast
            moe_device_map = ast.literal_eval(args.moe_device)
            if (args.moe_device_layers >= 0):
                front_moe_device = args.device
                if (_uses_thread_tp(tp_arg) and is_thread_tp_moe_model):
                    front_moe_device = _thread_tp_cuda_device_spec(tp_arg) or args.device
                llm.set_device_map(front_moe_device, True)
                if (isinstance(moe_device_map, list) or isinstance(moe_device_map, dict)):
                    llm.set_layered_moe_device_map(moe_device_map)
                else:
                    llm.set_layered_moe_device_map(args.moe_device)
                llm.set_moe_device_layers(args.moe_device_layers)
            elif (isinstance(moe_device_map, list) or isinstance(moe_device_map, dict)):
                llm.set_device_map(moe_device_map, True)
            else:
                llm.set_device_map(args.moe_device, True)
        except:
            if (args.moe_device_layers >= 0):
                front_moe_device = args.device
                if (_uses_thread_tp(tp_arg) and is_thread_tp_moe_model):
                    front_moe_device = _thread_tp_cuda_device_spec(tp_arg) or args.device
                llm.set_device_map(front_moe_device, True)
                llm.set_layered_moe_device_map(args.moe_device)
                llm.set_moe_device_layers(args.moe_device_layers)
            else:
                llm.set_device_map(args.moe_device, True)
    llm.set_cpu_threads(args.threads)
    llm.set_cpu_low_mem(args.low)
    if (args.cuda_embedding):
        llm.set_cuda_embedding(True)
    llm.set_cuda_shared_expert(
        args.cuda_shared_expert.lower() not in ["", "false", "0", "off"]
    )
    if (args.enable_amx.lower() not in ["", "false", "0", "off"]):
        llm.set_enable_amx(True)
    if (args.tokens > 0):
        llm.set_max_tokens(args.tokens)
    apply_page_size_default(args)
    if (args.page_size > 0):
        llm.set_page_size(args.page_size)
    apply_prefix_cache_env(args)
    if (hasattr(args, 'gpu_mem_ratio')):
        llm.set_gpu_mem_ratio(args.gpu_mem_ratio)
    if (hasattr(args, 'cuda_slab') and hasattr(llm, 'set_cuda_slab')):
        llm.set_cuda_slab(args.cuda_slab)
    os.environ["FASTLLM_QWEN35_ENABLE_MTP"] = str(mtp)
    graph = None
    if (args.custom != ""):
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_module", args.custom)
        if spec is None:
            raise ImportError(f"Cannot load module at {args.custom}")
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        if (hasattr(custom_module, "__model__")):
            graph = getattr(custom_module, "__model__")
    if (args.dtype_config != "" and os.path.exists(args.dtype_config)):
        with open(args.dtype_config, "r", encoding="utf-8") as file:
            args.dtype_config = file.read()
    if (args.chat_template != "" and os.path.exists(args.chat_template)):
        with open(args.chat_template, "r", encoding="utf-8") as file:
            args.chat_template = file.read()
    if startup_progress is not None:
        startup_progress.progress("initializing", 1, 1)
        llm.set_model_load_progress_callback(startup_progress.model_load_progress)
    try:
        model = llm.model(args.path, dtype = args.dtype, kv_cache_dtype = args.kv_cache_dtype,
                            moe_dtype = args.moe_dtype, graph = graph, tokenizer_type = "auto", lora = args.lora,
                            dtype_config = args.dtype_config, ori_model_path = args.ori, chat_template = args.chat_template, tool_call_parser = args.tool_call_parser)
        llm.report_model_load_progress("weights_finalize", 0, 1)
        if (args.enable_thinking.lower() in ["", "false", "0", "off"]):
            model.enable_thinking = False
        model.set_atype(args.atype)
        if (args.moe_atype != ""):
            model.set_moe_atype(args.moe_atype)
        if (args.cache_history.lower() not in ["", "false", "0", "off"]):
            model.set_save_history(True)
            if (args.cache_fast in ["", "false", "0", "off"]):
                llm.set_cpu_historycache(True)
        if (args.moe_experts > 0):
            model.set_moe_experts(args.moe_experts)
        if (args.max_batch > 0):
            model.set_max_batch(args.max_batch)
        model.native_context_window = model.get_max_input_len()
        model.configured_context_window_limit = None
        max_context_length = getattr(args, "max_context_length", -1)
        if (max_context_length == 0 or max_context_length < -1):
            raise ValueError("--max_context_length must be a positive integer")
        if (max_context_length > 0):
            model.configured_context_window_limit = max_context_length
            effective_context_length = model.set_max_context_length(max_context_length)
            print("[Fastllm] Per-session context window limit: %d tokens "
                  "(requested=%d, model max=%d)." %
                  (effective_context_length, max_context_length, model.native_context_window))
        if (args.kv_cache_limit != "" and args.kv_cache_limit != "auto"):
            model.set_kv_cache_limit(args.kv_cache_limit)
        if (args.chunked_prefill_size > 0):
            model.set_chunked_prefill_size(args.chunked_prefill_size)
        llm.report_model_load_progress("weights_finalize", 1, 1)
        llm.report_model_load_progress("warmup", 0, 1)
        model.warmup()
        llm.report_model_load_progress("warmup", 1, 1)
        effective_max_batch = model.get_max_batch()
        if (mtp > 0 and args.max_batch > 0 and
                effective_max_batch > 0 and effective_max_batch < args.max_batch):
            raise RuntimeError(
                "MTP startup validation failed: requested --max_batch=%d, but "
                "the current model/TP/MTP/GPU memory configuration safely "
                "supports at most %d concurrent requests. Lower --max_batch "
                "to %d or less, reduce --mtp, or use GPUs with more memory."
                % (args.max_batch, effective_max_batch, effective_max_batch)
            )
        return model
    finally:
        if startup_progress is not None:
            llm.set_model_load_progress_callback(None)

def make_download_parser(add_help = True):
    parser = argparse.ArgumentParser(
            description="Downloads a model or dataset from Hugging Face",
            usage="ftllm download [REPO_ID] [OPTIONS]",
            add_help = add_help
    )
        
    # 位置参数
    parser.add_argument("repo_id", nargs="?", help="Hugging Face repo ID")
    # 选项参数
    parser.add_argument("--include", nargs="+", default=[], help="Include patterns")
    parser.add_argument("--exclude", nargs="+", default=[], help="Exclude patterns")
    parser.add_argument("--hf_username", help="HF username")
    parser.add_argument("--hf_token", help="HF access token")
    parser.add_argument("--tool", choices=["aria2c", "wget"], default="aria2c", help="Download tool")
    parser.add_argument("-x", type=int, default=4, help="Threads for aria2c")
    parser.add_argument("-j", type=int, default=5, help="Concurrent downloads")
    parser.add_argument("--dataset", action="store_true", help="Download dataset")
    parser.add_argument("--local-dir", help="Local directory path")
    parser.add_argument("--revision", default="main", help="Revision to download")
    #parser.add_argument("-h", "--help", action="store_true", help="Show help")
        
    return parser

def get_fastllm_cache_path(model_name: str, cache_path = ""):
    system = sys.platform

    if cache_path == "":
        if system == "win32":
            # Windows: %LOCALAPPDATA%\Temp 或 C:\Users\<user>\AppData\Local\Temp
            cache_path = os.getenv('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local')) + '\\Temp'
        elif system == "darwin":
            # macOS: ~/Library/Caches
            cache_path = os.path.expanduser('~/Library/Caches')
        else:
            # Linux 和其他 Unix-like 系统: ~/.cache 或 $XDG_CACHE_HOME
            cache_path = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        cache_path = os.path.join(cache_path, "fastllm")

        cache_dir = os.getenv("FASTLLM_CACHEDIR")
        if (cache_dir and os.path.isdir(cache_dir)):
            cache_path = cache_dir

    cache_path = os.path.join(cache_path, model_name)
    return cache_path
