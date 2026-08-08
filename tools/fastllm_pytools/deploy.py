import json
import os
import re
import shlex
import tempfile
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 2
_TRUTHY = {"1", "true", "yes", "on"}
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class DeployConfig:
    name: str = ""
    command: str = "server"
    model: str = ""
    model_name: str = ""
    host: str = "0.0.0.0"
    port: str = "8080"
    device: str = "cuda"
    cuda_device_id: str = "0"
    tp: str = "2"
    cudapp: str = "2"
    device_custom: str = ""
    enable_moe_hybrid: bool = False
    moe_device: str = "numa"
    moe_device_layers: str = "10000"
    moe_device_custom: str = ""
    dtype: str = "auto"
    dtype_custom: str = ""
    moe_dtype: str = "auto"
    moe_dtype_custom: str = ""
    activation_dtype: str = "auto"
    gpu_mem_ratio: str = "0.9"
    chunked_prefill_size: str = "auto"
    kv_cache_dtype: str = "auto"
    moe_atype: str = "auto"
    enable_thinking: str = "auto"
    tokens: str = "auto"
    threads: str = "auto"
    kv_cache_limit: str = "auto"
    mtp: str = "auto"
    max_batch: str = "auto"
    max_context_length: str = "auto"
    temperature: str = ""
    top_p: str = ""
    top_k: str = ""
    repeat_penalty: str = ""
    api_key: str = ""
    hide_input: bool = False
    cache_dir: str = ""
    ori: str = ""
    low_memory: bool = False
    cuda_embedding: bool = False
    prefix_cache: bool = True
    prefix_snapshot_interval_pages: str = "auto"
    startup_timeout: str = "180"
    default_max_tokens: str = "16384"
    extra_args: str = ""
    env_vars: str = ""


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(str(value).strip()))


def _optional(value: object, auto_values=("", "auto")) -> str:
    text = str(value).strip()
    return "" if text in auto_values else text


def _add(argv: List[str], option: str, value: object) -> None:
    text = str(value).strip()
    if text:
        argv.extend([option, text])


def parse_env_vars(value: str) -> Dict[str, str]:
    raw = str(value).strip()
    if not raw:
        return {}
    result: Dict[str, str] = {}
    for part in shlex.split(raw.replace(";", " ")):
        if "=" not in part:
            raise ValueError(f"环境变量缺少 '=': {part}")
        key, item_value = part.split("=", 1)
        if not _ENV_NAME.fullmatch(key):
            raise ValueError(f"环境变量名无效: {key}")
        result[key] = item_value
    return result


def get_saved_commands_path() -> str:
    config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return os.path.join(_expand(config_home), "fastllm", "tui_commands.json")


def default_model_name_from_path(model_path: str) -> str:
    model_path = _expand(model_path)
    if not model_path:
        return ""
    normalized = os.path.normpath(model_path)
    return os.path.basename(normalized) or normalized.strip(os.sep) or normalized


def effective_model_name(config: DeployConfig) -> str:
    return config.model_name.strip() or default_model_name_from_path(config.model)


def _normalize_device(config: DeployConfig) -> None:
    value = str(config.device).strip()
    lower = value.lower()
    if lower in ("", "auto"):
        config.device = "cuda"
        config.cuda_device_id = str(config.cuda_device_id).strip() or "0"
    elif lower.startswith("cuda:"):
        config.device = "cuda"
        config.cuda_device_id = value.split(":", 1)[1].strip() or "0"
    elif lower.startswith("multicuda:"):
        config.device = "tp"
        config.tp = value.split(":", 1)[1].strip()
    elif lower == "multicuda":
        config.device = "tp"
        config.tp = str(config.tp).strip() or "2"
    elif lower.startswith("cudapp="):
        config.device = "cudapp"
        config.cudapp = value.split("=", 1)[1].strip()


def _normalize_moe(config: DeployConfig, source: dict) -> None:
    device = str(config.moe_device).strip().lower()
    if "enable_moe_hybrid" not in source and device in ("numa", "disk"):
        config.enable_moe_hybrid = True
        if "moe_device_layers" not in source:
            config.moe_device_layers = "-1"
    if config.enable_moe_hybrid:
        config.moe_device = device if device in ("numa", "disk") else "numa"
        if not str(config.moe_device_layers).strip():
            config.moe_device_layers = "10000"


def config_from_dict(data: dict) -> DeployConfig:
    config = DeployConfig()
    valid = {item.name for item in fields(DeployConfig)}
    aliases = {
        "atype": "activation_dtype",
        "prefix_cache_snapshot_interval_pages": "prefix_snapshot_interval_pages",
        "low": "low_memory",
    }
    source = dict(data or {})
    for old, new in aliases.items():
        if old in source and new not in source:
            source[new] = source[old]
    for key, value in source.items():
        if key in valid:
            setattr(config, key, value)
    _normalize_device(config)
    _normalize_moe(config, source)
    return config


def clone_config(config: DeployConfig) -> DeployConfig:
    return config_from_dict(asdict(config))


def load_saved_configs(path: Optional[str] = None) -> List[DeployConfig]:
    path = path or get_saved_commands_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    raw = payload.get("commands", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        return []
    return [config_from_dict(item) for item in raw if isinstance(item, dict)]


def _atomic_json_write(path: str, payload: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".fastllm-", suffix=".tmp", dir=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            dirfd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_saved_configs(configs: Sequence[DeployConfig], path: Optional[str] = None) -> None:
    path = path or get_saved_commands_path()
    payload = {
        "version": SCHEMA_VERSION,
        "commands": [asdict(config) for config in configs],
    }
    _atomic_json_write(path, payload)


def _device_args(config: DeployConfig) -> Tuple[str, str]:
    if config.device == "cuda":
        value = str(config.cuda_device_id).strip()
        if value.lower().startswith("cuda:"):
            value = value.split(":", 1)[1].strip()
        return "cuda:" + (value or "0"), ""
    if config.device == "cudapp":
        return "cudapp=" + str(config.cudapp).strip(), ""
    if config.device == "tp":
        return "", str(config.tp).strip()
    if config.device == "cpu":
        return "cpu", ""
    if config.device == "custom":
        return str(config.device_custom).strip(), ""
    return str(config.device).strip(), ""


def build_fastllm_argv(config: DeployConfig) -> List[str]:
    argv = ["ftllm", config.command]
    model = _expand(config.model)
    if model:
        argv.append(model)
    device, tp = _device_args(config)
    _add(argv, "--device", device)
    _add(argv, "--tp", tp)
    if config.enable_moe_hybrid:
        _add(argv, "--moe_device", config.moe_device)
        _add(argv, "--moe_device_layers", config.moe_device_layers)
    _add(argv, "--gpu_mem_ratio", _optional(config.gpu_mem_ratio))
    _add(argv, "--chunked_prefill_size", _optional(config.chunked_prefill_size))
    _add(argv, "--kv_cache_dtype", _optional(config.kv_cache_dtype))
    _add(argv, "--moe_atype", _optional(config.moe_atype))
    _add(argv, "--enable_thinking", _optional(config.enable_thinking))
    _add(argv, "--tokens", _optional(config.tokens))
    dtype = config.dtype_custom.strip() if config.dtype == "custom" else _optional(config.dtype)
    moe_dtype = (config.moe_dtype_custom.strip()
                 if config.moe_dtype == "custom" else _optional(config.moe_dtype))
    _add(argv, "--dtype", dtype)
    _add(argv, "--moe_dtype", moe_dtype)
    _add(argv, "--atype", _optional(config.activation_dtype))
    _add(argv, "-t", _optional(config.threads))
    _add(argv, "--kv_cache_limit", _optional(config.kv_cache_limit))
    _add(argv, "--mtp", _optional(config.mtp))
    _add(argv, "--max_batch", _optional(config.max_batch))
    _add(argv, "--cache_dir", _expand(config.cache_dir))
    if model.lower().endswith(".gguf"):
        _add(argv, "--ori", _expand(config.ori))
    if config.low_memory:
        argv.append("-l")
    if config.cuda_embedding:
        argv.append("--cuda_embedding")
    if config.command == "server":
        _add(argv, "--model_name", effective_model_name(config))
        _add(argv, "--host", config.host)
        _add(argv, "--port", config.port)
        _add(argv, "--api_key", config.api_key)
        _add(argv, "--max_context_length", _optional(config.max_context_length))
        _add(argv, "--default_max_tokens", config.default_max_tokens)
        _add(argv, "--temperature", config.temperature)
        _add(argv, "--top_p", config.top_p)
        _add(argv, "--top_k", config.top_k)
        _add(argv, "--repeat_penalty", config.repeat_penalty)
        if config.hide_input:
            argv.append("--hide_input")
    elif config.command == "webui":
        _add(argv, "--port", config.port)
    if str(config.extra_args).strip():
        argv.extend(shlex.split(config.extra_args))
    return argv


def build_fastllm_env(config: DeployConfig) -> Dict[str, str]:
    env = parse_env_vars(config.env_vars)
    env.setdefault("FASTLLM_PREFIX_CACHE", "1" if config.prefix_cache else "0")
    snapshot = _optional(config.prefix_snapshot_interval_pages)
    if snapshot:
        env.setdefault("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", snapshot)
    env.setdefault("FASTLLM_CUDA_EMBEDDING", "1" if config.cuda_embedding else "0")
    return env


def build_child_environment(config: DeployConfig,
                            inherited: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    source = dict(os.environ if inherited is None else inherited)
    allowed = {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE",
        "TERM", "TMPDIR", "TEMP", "TMP", "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "VIRTUAL_ENV", "CONDA_PREFIX",
        "PYTHONPATH", "PYTHONUNBUFFERED",
    }
    result = {key: value for key, value in source.items()
              if key in allowed or key.startswith("LC_")}
    result.update(build_fastllm_env(config))
    return result


def build_fastllm_command(config: DeployConfig) -> str:
    prefix = [f"{key}={shlex.quote(value)}"
              for key, value in build_fastllm_env(config).items()]
    return " ".join(prefix + [shlex.quote(part) for part in build_fastllm_argv(config)])


def _positive_or_auto(value: object) -> bool:
    text = str(value).strip()
    if text in ("", "auto"):
        return True
    try:
        return int(text) > 0
    except ValueError:
        return False


def _positive(value: object) -> bool:
    try:
        return int(str(value).strip()) > 0
    except ValueError:
        return False


def _mtp(value: object) -> bool:
    text = str(value).strip()
    if text in ("", "auto"):
        return True
    try:
        return 0 <= int(text) <= 9
    except ValueError:
        return False


def _float(value: object, low=None, high=None, optional=True) -> bool:
    text = str(value).strip()
    if optional and not text:
        return True
    try:
        number = float(text)
    except ValueError:
        return False
    return (low is None or number >= low) and (high is None or number <= high)


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def validate_config(config: DeployConfig) -> List[str]:
    errors: List[str] = []
    model = _expand(config.model)
    if not model:
        errors.append("模型路径不能为空。")
    elif model.lower().endswith(".gguf"):
        if not os.path.isfile(model):
            errors.append("GGUF模型路径必须是已存在的本地 .gguf 文件。")
    elif not os.path.isdir(model):
        errors.append("模型路径必须是已存在的本地模型目录，或 .gguf 文件。")
    ori = _expand(config.ori)
    if model.lower().endswith(".gguf") and ori and not os.path.isdir(ori):
        errors.append("模型配置文件夹必须是已存在的本地目录。")
    if config.command not in ("server", "webui", "run"):
        errors.append("命令必须是 server、webui 或 run。")
    if config.command in ("server", "webui"):
        try:
            port = int(str(config.port).strip())
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            errors.append("端口必须是 1-65535 的整数。")
    for label, value in (
        ("预处理分片大小", config.chunked_prefill_size),
        ("最大Batch", config.max_batch),
        ("单会话上下文", config.max_context_length),
        ("tokens数量", config.tokens),
        ("线程数", config.threads),
        ("前缀快照页数", config.prefix_snapshot_interval_pages),
    ):
        if not _positive_or_auto(value):
            errors.append(f"{label}必须是正整数或 auto。")
    if not _positive(config.startup_timeout):
        errors.append("启动超时必须是正整数。")
    if not _positive(config.default_max_tokens):
        errors.append("默认输出token上限必须是正整数。")
    if not _mtp(config.mtp):
        errors.append("MTP 必须是 0-9 的整数或 auto。")
    if not _float(config.gpu_mem_ratio, low=0, high=1, optional=False) or float(config.gpu_mem_ratio) <= 0:
        errors.append("显存利用率必须是 0 到 1 之间的数字。")
    if config.command == "server" and not str(config.host).strip():
        errors.append("监听地址不能为空。")
    if config.activation_dtype not in ("auto", "float16", "float32", "bfloat16"):
        errors.append("activation dtype 必须是 auto、float16、float32 或 bfloat16。")
    if config.kv_cache_dtype not in ("auto", "float16", "bfloat16", "fp8_e4m3", "turbo3"):
        errors.append("KV Cache 类型无效。")
    if config.device == "cuda" and not str(config.cuda_device_id).replace("cuda:", "", 1).isdigit():
        errors.append("CUDA卡号必须是非负整数。")
    if config.dtype == "custom" and not config.dtype_custom.strip():
        errors.append("选择自定义权重类型时必须填写自定义权重类型。")
    if config.moe_dtype == "custom" and not config.moe_dtype_custom.strip():
        errors.append("选择自定义MOE类型时必须填写自定义MOE类型。")
    if not _float(config.temperature, low=0):
        errors.append("temperature 必须大于等于 0，或留空。")
    if not _float(config.top_p, low=0, high=1):
        errors.append("top_p 必须在 0 到 1 之间，或留空。")
    if config.top_k and not _positive(config.top_k):
        errors.append("top_k 必须是正整数，或留空。")
    if not _float(config.repeat_penalty, low=0):
        errors.append("repeat_penalty 必须大于等于 0，或留空。")
    try:
        extra = shlex.split(config.extra_args) if config.extra_args.strip() else []
    except ValueError as exc:
        errors.append(f"额外参数无法解析: {exc}")
        extra = []
    try:
        env = build_fastllm_env(config)
    except ValueError as exc:
        errors.append(str(exc))
        env = {}
    turbo_gate = env.get("FASTLLM_QWEN35_TURBO3_KV")
    if config.kv_cache_dtype == "turbo3" and not _truthy(turbo_gate):
        errors.append("Turbo3 必须显式设置 FASTLLM_QWEN35_TURBO3_KV=1。")
    if config.kv_cache_dtype != "turbo3" and _truthy(turbo_gate):
        errors.append("非 Turbo3 profile 不得启用 FASTLLM_QWEN35_TURBO3_KV。")
    if config.cuda_embedding and env.get("FASTLLM_CUDA_EMBEDDING") == "0":
        errors.append("CUDA embedding 配置互相冲突。")
    if not config.cuda_embedding and env.get("FASTLLM_CUDA_EMBEDDING") == "1":
        errors.append("CPU embedding profile 不得启用 FASTLLM_CUDA_EMBEDDING。")
    forbidden = {"--batch", "--default-max-tokens"}
    if any(item in forbidden for item in extra):
        errors.append("额外参数包含不适用于 ftllm server 的参数。")
    return errors
