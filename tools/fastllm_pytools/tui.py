import json
import os
import shlex
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass, fields
from typing import Callable, List, Optional, Sequence, Tuple, Union

try:
    import curses
except ImportError:
    curses = None


Choice = Tuple[str, str]
ModelGroup = Tuple[str, str, Sequence[Choice]]
PATH_COMPLETION_FIELDS = {"model", "cache_dir", "ori"}
DIRECTORY_COMPLETION_FIELDS = {"cache_dir", "ori"}
CONTENT_MAX_WIDTH = 96
PANEL_MAX_HEIGHT = 28
PANEL_PADDING_X = 2
PANEL_PADDING_Y = 1
DEFAULT_ESCDELAY_MS = 25
QWEN_MODELSCOPE_MODEL_CHOICES: Sequence[Choice] = (
    ("Qwen/Qwen3.6-27B-FP8", "Qwen3.6-27B-FP8"),
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    ("Qwen/Qwen3-4B", "Qwen3-4B"),
    ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("Qwen/Qwen3-14B", "Qwen3-14B"),
    ("Qwen/Qwen3-32B", "Qwen3-32B"),
    ("Qwen/Qwen3-30B-A3B", "Qwen3-30B-A3B"),
    ("Qwen/Qwen3-235B-A22B", "Qwen3-235B-A22B"),
)
DEEPSEEK_MODELSCOPE_MODEL_CHOICES: Sequence[Choice] = (
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-Distill-Qwen-14B"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek-R1-Distill-Qwen-32B"),
)
MINIMAX_MODELSCOPE_MODEL_CHOICES: Sequence[Choice] = (
    ("MiniMax/MiniMax-Text-01", "MiniMax-Text-01"),
    ("MiniMax/MiniMax-M1-40k", "MiniMax-M1-40k"),
    ("MiniMax/MiniMax-M1-80k", "MiniMax-M1-80k"),
)
HOT_MODELSCOPE_MODEL_CHOICES: Sequence[Choice] = (
    ("Qwen/Qwen3.6-27B-FP8", "Qwen3.6-27B-FP8"),
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("Qwen/Qwen3-30B-A3B", "Qwen3-30B-A3B"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek-R1-Distill-Qwen-32B"),
    ("MiniMax/MiniMax-M1-40k", "MiniMax-M1-40k"),
)
CUSTOM_MODELSCOPE_MODEL_CHOICES: Sequence[Choice] = (
    ("custom", "自定义模型ID"),
)
MODELSCOPE_MODEL_GROUPS: Sequence[ModelGroup] = (
    ("hot", "热门模型", HOT_MODELSCOPE_MODEL_CHOICES),
    ("qwen", "千问系列", QWEN_MODELSCOPE_MODEL_CHOICES),
    ("deepseek", "DeepSeek系列", DEEPSEEK_MODELSCOPE_MODEL_CHOICES),
    ("minimax", "MiniMax系列", MINIMAX_MODELSCOPE_MODEL_CHOICES),
    ("custom", "自定义", CUSTOM_MODELSCOPE_MODEL_CHOICES),
)
MODELSCOPE_MODEL_GROUP_CHOICES: Sequence[Choice] = tuple(
    (group_key, group_label) for group_key, group_label, _ in MODELSCOPE_MODEL_GROUPS
)


def _flatten_modelscope_model_choices() -> Sequence[Choice]:
    choices: List[Choice] = []
    seen = set()
    for _, _, group_choices in MODELSCOPE_MODEL_GROUPS:
        for value, label in group_choices:
            if value in seen:
                continue
            choices.append((value, label))
            seen.add(value)
    return tuple(choices)


MODELSCOPE_MODEL_CHOICES: Sequence[Choice] = _flatten_modelscope_model_choices()


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
    temperature: str = ""
    top_p: str = ""
    top_k: str = ""
    repeat_penalty: str = ""
    api_key: str = ""
    hide_input: bool = False
    cache_dir: str = ""
    ori: str = ""
    extra_args: str = ""
    env_vars: str = ""


@dataclass
class ModelScopeDownloadConfig:
    model_id: str = "Qwen/Qwen3-0.6B"
    model_id_custom: str = ""
    target_dir: str = ""
    max_workers: str = "4"
    custom_args: str = ""


@dataclass
class FormField:
    key: str
    label: str
    kind: str
    help: str
    choices: Sequence[Choice] = ()
    visible: Optional[Callable[[DeployConfig], bool]] = None

    def is_visible(self, config: DeployConfig) -> bool:
        return self.visible(config) if self.visible is not None else True


COMMAND_CHOICES: Sequence[Choice] = (
    ("server", "API Server"),
    ("webui", "WebUI"),
    ("run", "本地聊天"),
)

DEVICE_CHOICES: Sequence[Choice] = (
    ("cuda", "CUDA 单卡"),
    ("tp", "多卡张量并行"),
    ("cudapp", "多卡串行"),
    ("cpu", "CPU"),
)

MOE_DEVICE_CHOICES: Sequence[Choice] = (
    ("numa", "NUMA CPU"),
    ("disk", "Disk"),
)

DTYPE_CHOICES: Sequence[Choice] = (
    ("auto", "自动"),
    ("float16", "float16"),
    ("float32", "float32"),
    ("int8", "int8"),
    ("int4", "int4"),
    ("int4g128", "int4g128"),
    ("int4g256", "int4g256"),
    ("fp8_e4m3", "fp8_e4m3"),
    ("custom", "自定义"),
)

MOE_DTYPE_CHOICES: Sequence[Choice] = (
    ("auto", "自动/不指定"),
    ("float16", "float16"),
    ("int8", "int8"),
    ("int4", "int4"),
    ("int4g128", "int4g128"),
    ("int4g256", "int4g256"),
    ("fp8_e4m3", "fp8_e4m3"),
    ("custom", "自定义"),
)

KV_CACHE_DTYPE_CHOICES: Sequence[Choice] = (
    ("auto", "auto"),
    ("float16", "float16"),
    ("bfloat16", "bfloat16"),
    ("fp8_e4m3", "fp8"),
)

MOE_ATYPE_CHOICES: Sequence[Choice] = (
    ("auto", "自动/不指定"),
    ("float32", "float32"),
    ("float16", "float16"),
    ("bfloat16", "bfloat16"),
)

ENABLE_THINKING_CHOICES: Sequence[Choice] = (
    ("auto", "自动"),
    ("true", "开启"),
    ("false", "关闭"),
)


FIELDS: Sequence[FormField] = (
    FormField("model", "模型路径", "text", "必填，必须是已存在的本地模型目录或 GGUF 文件；可选择已下载模型，或按 Tab 补全路径。"),
    FormField(
        "ori",
        "模型配置文件夹",
        "text",
        "GGUF 文件对应的原始模型目录；会传给 --ori，用于读取 config.json 等模型配置。",
        visible=lambda c: is_gguf_model(c.model),
    ),
    FormField("name", "配置名称", "text", "保存在首页命令列表中的名称；新建时自动生成，可按需修改。"),
    FormField(
        "command",
        "部署方式",
        "choice",
        "server 提供 OpenAI API；webui 启动网页聊天；run 启动本地终端聊天。",
        COMMAND_CHOICES,
    ),
    FormField(
        "model_name",
        "API模型名",
        "text",
        "OpenAI API /v1/models 暴露的模型名；留空时使用模型路径最后一级目录名。",
        visible=lambda c: c.command == "server",
    ),
    FormField(
        "host",
        "监听地址",
        "text",
        "API Server 监听地址，常用 0.0.0.0 或 127.0.0.1。",
        visible=lambda c: c.command == "server",
    ),
    FormField(
        "port",
        "端口",
        "text",
        "server 默认 8080；webui 默认 1616。",
        visible=lambda c: c.command in ("server", "webui"),
    ),
    FormField("device", "主设备", "choice", "主干网络使用的计算设备。", DEVICE_CHOICES),
    FormField(
        "cuda_device_id",
        "CUDA卡号",
        "text",
        "单卡 CUDA 使用的卡号；留空默认 0。",
        visible=lambda c: c.device == "cuda",
    ),
    FormField(
        "tp",
        "TP卡数/ID",
        "text",
        "多卡张量并行：输入 1 表示单卡；输入 4 表示使用 0、1、2、3 号卡；输入 0,2,3 表示指定卡号。",
        visible=lambda c: c.device == "tp",
    ),
    FormField(
        "cudapp",
        "串行参数",
        "text",
        "多卡串行：生成 --device cudapp=...；只填写 ... 的内容。输入 4 表示 4 卡串行；输入 0,1,2 表示指定卡号。",
        visible=lambda c: c.device == "cudapp",
    ),
    FormField(
        "device_custom",
        "自定义主设备",
        "text",
        "例如 multicuda:0:4,1:5,cpu:1 或 {'cuda:0':3,'cuda:1':2}。",
        visible=lambda c: c.device == "custom",
    ),
    FormField(
        "enable_moe_hybrid",
        "打开MOE混合推理",
        "bool",
        "打开后可把最后若干层 MOE 专家放到 NUMA CPU 或磁盘上。",
    ),
    FormField(
        "moe_device",
        "MOE推理设备",
        "choice",
        "选择放置指定 MOE 层的设备。",
        MOE_DEVICE_CHOICES,
        visible=lambda c: c.enable_moe_hybrid,
    ),
    FormField(
        "moe_device_layers",
        "MOE设备层数",
        "text",
        "输入 8 表示最后 8 层 MOE 放到所选设备；输入 -1 表示全部 MOE 层都放到所选设备。",
        visible=lambda c: c.enable_moe_hybrid,
    ),
    FormField(
        "gpu_mem_ratio",
        "显存利用率",
        "text",
        "GPU显存使用比例，如 0.9 表示最多使用 90% 的显存。",
    ),
    FormField(
        "chunked_prefill_size",
        "预处理分片大小",
        "text",
        "分块 prefill 的切片大小；调小可以减少显存占用，auto 表示不指定。",
    ),
    FormField("kv_cache_dtype", "缓存类型", "choice", "KV Cache 类型，可使用 auto、float16、bfloat16 或 fp8。", KV_CACHE_DTYPE_CHOICES),
    FormField("mtp", "MTP", "text", "Qwen3.5 MTP 每步生成的 draft token 数；0 表示关闭，1-8 开启，auto 表示不指定。"),
    FormField("max_batch", "最大Batch", "text", "每次最多同时推理的询问数量；auto 表示不指定。"),
    FormField("moe_atype", "MOE激活类型", "choice", "MOE层激活类型，可使用 auto、float32、float16 或 bfloat16。", MOE_ATYPE_CHOICES),
    FormField("enable_thinking", "思考开关", "choice", "是否开启硬思考开关，需要模型支持。", ENABLE_THINKING_CHOICES),
    FormField("tokens", "tokens数量", "text", "设置总 token 数量；auto 表示不指定。"),
    FormField("threads", "线程数", "text", "线程数量；auto 表示由 fastllm 自动估算。"),
    FormField(
        "sampling_params",
        "采样参数",
        "submenu",
        "进入二级菜单，配置 temperature、top_p、top_k 和 repeat_penalty。",
        visible=lambda c: c.command == "server",
    ),
    FormField(
        "api_key",
        "API Key",
        "text",
        "设置后 server 会校验 Bearer token；留空表示不校验。",
        visible=lambda c: c.command == "server",
    ),
    FormField("extra_args", "其它参数", "text", "直接追加到 ftllm 命令末尾，例如 --cuda_slab 1024。"),
    FormField("env_vars", "环境变量", "text", "启动前设置环境变量，格式 KEY=VALUE KEY2=VALUE2，例如 FASTLLM_ACTIVATE_NUMA=ON。"),
)

SAMPLING_FIELDS: Sequence[FormField] = (
    FormField("temperature", "temperature", "text", "覆盖服务端默认 temperature；留空表示使用模型默认值。"),
    FormField("top_p", "top_p", "text", "覆盖服务端默认 top_p；留空表示使用模型默认值。"),
    FormField("top_k", "top_k", "text", "覆盖服务端默认 top_k；留空表示使用模型默认值。"),
    FormField("repeat_penalty", "repeat_penalty", "text", "覆盖服务端默认 repeat_penalty；留空表示使用模型默认值。"),
)

BASIC_FIELD_KEYS = {
    "name",
    "command",
    "model",
    "ori",
    "model_name",
    "host",
    "port",
    "device",
    "cuda_device_id",
    "tp",
    "cudapp",
    "device_custom",
    "enable_moe_hybrid",
    "moe_device",
    "moe_device_layers",
}


def iter_deploy_fields(config: DeployConfig, show_advanced: bool):
    for field in FIELDS:
        is_basic = field.key in BASIC_FIELD_KEYS
        if show_advanced and is_basic:
            continue
        if not show_advanced and not is_basic:
            continue
        if field.is_visible(config):
            yield field


def visible_deploy_fields(config: DeployConfig, show_advanced: bool) -> List[FormField]:
    return list(iter_deploy_fields(config, show_advanced))


DOWNLOAD_FIELDS: Sequence[FormField] = (
    FormField("model_id", "模型", "choice", "从常用 ModelScope 模型列表选择。", MODELSCOPE_MODEL_CHOICES),
    FormField(
        "model_id_custom",
        "自定义模型ID",
        "text",
        "填写 ModelScope 模型ID，例如 Qwen/Qwen3-0.6B。",
        visible=lambda c: c.model_id == "custom",
    ),
    FormField("target_dir", "下载目录", "text", "默认使用系统缓存目录，支持 Tab 补全。"),
    FormField("max_workers", "下载并发", "text", "ModelScope SDK 下载线程数，填写正整数。"),
    FormField(
        "custom_args",
        "自定义参数",
        "text",
        "可选，snapshot_download 参数，格式 key=value; key=value。例如 allow_patterns=*.json,*.safetensors; token=xxx。",
    ),
)


def _choice_label(choices: Sequence[Choice], value: str) -> str:
    for choice_value, label in choices:
        if choice_value == value:
            return label
    return value


def _previous_index(index: int, count: int) -> int:
    return (index - 1) % count if count > 0 else 0


def _next_index(index: int, count: int) -> int:
    return (index + 1) % count if count > 0 else 0


def _modelscope_group_index_for_model(model_id: str) -> int:
    for group_index, (_, _, choices) in enumerate(MODELSCOPE_MODEL_GROUPS):
        if any(value == model_id for value, _ in choices):
            return group_index
    return 0


def _resolve_custom(value: str, custom_value: str) -> str:
    if value == "custom":
        return custom_value.strip()
    if value == "auto":
        return ""
    return value.strip()


def is_gguf_model(model_path: str) -> bool:
    return str(model_path).strip().lower().endswith(".gguf")


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    first = name[0]
    if not (first == "_" or "A" <= first <= "Z" or "a" <= first <= "z"):
        return False
    return all(ch == "_" or "A" <= ch <= "Z" or "a" <= ch <= "z" or "0" <= ch <= "9" for ch in name)


def parse_env_vars(value: str) -> dict:
    raw = str(value).strip()
    if not raw:
        return {}
    parts = shlex.split(raw.replace(";", " "))
    env = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"环境变量缺少 '=': {part}")
        key, env_value = part.split("=", 1)
        if not _is_valid_env_name(key):
            raise ValueError(f"环境变量名无效: {key}")
        env[key] = env_value
    return env


def build_fastllm_env(config: DeployConfig) -> dict:
    return parse_env_vars(config.env_vars)


def _env_prefix(config: DeployConfig) -> List[str]:
    return [f"{key}={shlex.quote(value)}" for key, value in build_fastllm_env(config).items()]


def _cuda_device_id(value: str) -> str:
    value = str(value).strip()
    if value.lower().startswith("cuda:"):
        value = value.split(":", 1)[1].strip()
    return value or "0"


def _resolve_main_device_args(config: DeployConfig) -> Tuple[str, str]:
    if config.device == "cuda":
        return "cuda:" + _cuda_device_id(config.cuda_device_id), ""
    if config.device == "cudapp":
        return "cudapp=" + config.cudapp.strip(), ""
    if config.device == "tp":
        return "", config.tp.strip()
    if config.device == "cpu":
        return "cpu", ""
    return _resolve_custom(config.device, config.device_custom), ""


def _is_valid_cuda_device_id(value: str) -> bool:
    return _cuda_device_id(value).isdigit()


def _is_valid_tp_spec(value: str) -> bool:
    value = str(value).strip()
    if not value:
        return False
    lower = value.lower()
    if lower.startswith("cuda:") or lower.startswith("multicuda:"):
        value = value.split(":", 1)[1].strip()
        if not value:
            return False
    if value.isdigit():
        return int(value) >= 0
    parts = [part.strip() for part in value.split(",")]
    return len(parts) >= 1 and all(part.isdigit() for part in parts)


def _is_valid_cudapp_spec(value: str) -> bool:
    value = str(value).strip()
    if not value:
        return False
    if value.isdigit():
        return int(value) > 0
    parts = [part.strip() for part in value.split(",")]
    return len(parts) >= 2 and len(set(parts)) == len(parts) and all(part.isdigit() for part in parts)


def _is_valid_moe_device_layers(value: str) -> bool:
    value = str(value).strip()
    if not value:
        return False
    try:
        layers = int(value)
    except ValueError:
        return False
    return layers == -1 or layers > 0


def _is_auto_or_empty(value: str) -> bool:
    return str(value).strip() in ("", "auto")


def _is_positive_int_or_auto(value: str) -> bool:
    if _is_auto_or_empty(value):
        return True
    try:
        return int(str(value).strip()) > 0
    except ValueError:
        return False


def _is_mtp_value(value: str) -> bool:
    if _is_auto_or_empty(value):
        return True
    try:
        mtp = int(str(value).strip())
    except ValueError:
        return False
    return 0 <= mtp <= 8


def _is_positive_float_or_auto(value: str) -> bool:
    if _is_auto_or_empty(value):
        return True
    try:
        return float(str(value).strip()) > 0
    except ValueError:
        return False


def _is_ratio(value: str) -> bool:
    try:
        ratio = float(str(value).strip())
    except ValueError:
        return False
    return 0 < ratio <= 1


def _is_optional_float(value: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
    value = str(value).strip()
    if value == "":
        return True
    try:
        number = float(value)
    except ValueError:
        return False
    if min_value is not None and number < min_value:
        return False
    if max_value is not None and number > max_value:
        return False
    return True


def _is_optional_positive_int(value: str) -> bool:
    value = str(value).strip()
    if value == "":
        return True
    try:
        return int(value) > 0
    except ValueError:
        return False


def _optional_text(value: str, auto_values: Sequence[str] = ("", "auto")) -> str:
    value = str(value).strip()
    return "" if value in auto_values else value


def _add_option(argv: List[str], name: str, value: str):
    if value != "":
        argv.extend([name, value])


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width


def _clip_display(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    width = 0
    output = []
    for char in str(text):
        char_width = 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
        if width + char_width > max_width:
            break
        output.append(char)
        width += char_width
    return "".join(output)


def _pad_display(text: str, width: int) -> str:
    clipped = _clip_display(text, width)
    return clipped + " " * max(0, width - _display_width(clipped))


def _expand_user_path(value: str) -> str:
    return os.path.expanduser(value) if value.startswith("~") else value


def get_fastllm_cache_dir() -> str:
    if os.name == "nt":
        cache_home = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    elif sys.platform == "darwin":
        cache_home = os.path.expanduser("~/Library/Caches")
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    cache_home = os.path.expanduser(os.path.expandvars(cache_home))
    return os.path.join(cache_home, "fastllm")


def default_model_name_from_path(model_path: str) -> str:
    model_path = _expand_user_path(model_path.strip())
    if not model_path:
        return ""
    normalized = os.path.normpath(model_path)
    if normalized == os.curdir:
        normalized = os.path.abspath(normalized)
    return os.path.basename(normalized) or normalized.strip(os.sep) or normalized


def effective_model_name(config: DeployConfig) -> str:
    return config.model_name.strip() or default_model_name_from_path(config.model)


def _is_path_completion_field(field: FormField) -> bool:
    return field.key in PATH_COMPLETION_FIELDS


def _is_directory_completion_field(field: FormField) -> bool:
    return field.key in DIRECTORY_COMPLETION_FIELDS


def _is_download_path_completion_field(field: FormField) -> bool:
    return field.key == "target_dir"


def _resolve_modelscope_model_id(config: ModelScopeDownloadConfig) -> str:
    if config.model_id == "custom":
        return config.model_id_custom.strip()
    return config.model_id.strip()


def _parse_modelscope_custom_value(value: str):
    value = value.strip()
    if "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    try:
        return int(value)
    except ValueError:
        pass
    return value


def parse_modelscope_custom_args(value: str) -> dict:
    kwargs = {}
    if not value.strip():
        return kwargs
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"自定义参数缺少 '=': {part}")
        key, raw_value = part.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"自定义参数key为空: {part}")
        kwargs[key] = _parse_modelscope_custom_value(raw_value)
    return kwargs


def validate_modelscope_download(config: ModelScopeDownloadConfig) -> List[str]:
    errors = []
    if not _resolve_modelscope_model_id(config):
        errors.append("模型ID不能为空。")
    if not config.target_dir.strip():
        errors.append("下载目录不能为空。")
    if config.max_workers.strip():
        try:
            workers = int(config.max_workers)
            if workers <= 0:
                errors.append("下载并发必须是正整数。")
        except ValueError:
            errors.append("下载并发必须是正整数。")
    try:
        parse_modelscope_custom_args(config.custom_args)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def default_modelscope_target_dir(model_id: str) -> str:
    name = model_id.split("/")[-1] if model_id else "model"
    return os.path.join(get_modelscope_cache_dir(), name)


def get_modelscope_cache_dir() -> str:
    return os.path.join(get_fastllm_cache_dir(), "modelscope")


def list_downloaded_model_dirs() -> List[Choice]:
    cache_dir = get_modelscope_cache_dir()
    if not os.path.isdir(cache_dir):
        return []
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return []

    choices: List[Choice] = []
    for name in names:
        if name.startswith("."):
            continue
        path = os.path.join(cache_dir, name)
        if os.path.isdir(path):
            choices.append((path, name))
    choices.sort(key=lambda item: item[1].lower())
    return choices


def update_default_modelscope_target_dir(config: ModelScopeDownloadConfig, old_model_id: str = ""):
    model_id = _resolve_modelscope_model_id(config)
    current_target = config.target_dir.strip()
    old_default = default_modelscope_target_dir(old_model_id) if old_model_id else ""
    if not model_id:
        if old_default and _expand_user_path(current_target) == _expand_user_path(old_default):
            config.target_dir = ""
        return
    if not current_target or (old_default and _expand_user_path(current_target) == _expand_user_path(old_default)):
        config.target_dir = default_modelscope_target_dir(model_id)


def download_modelscope_model(config: ModelScopeDownloadConfig) -> str:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise RuntimeError("缺少 modelscope 包，请先执行: pip install modelscope") from exc

    target_dir = _expand_user_path(config.target_dir.strip())
    os.makedirs(target_dir, exist_ok=True)
    kwargs = {
        "model_id": _resolve_modelscope_model_id(config),
        "revision": "master",
        "local_dir": target_dir,
        "max_workers": int(config.max_workers.strip() or "4"),
    }
    kwargs.update(parse_modelscope_custom_args(config.custom_args))
    return snapshot_download(**kwargs)


def get_saved_commands_path() -> str:
    config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return os.path.join(config_home, "fastllm", "tui_commands.json")


def clone_config(config: DeployConfig) -> DeployConfig:
    return config_from_dict(asdict(config))


def normalize_main_device_config(config: DeployConfig):
    device = str(config.device).strip()
    lower = device.lower()
    if lower in ("", "auto"):
        config.device = "cuda"
        config.cuda_device_id = config.cuda_device_id.strip() or "0"
    elif lower.startswith("cudapp="):
        spec = device.split("=", 1)[1].strip()
        if _is_valid_cudapp_spec(spec):
            config.device = "cudapp"
            config.cudapp = spec
    elif lower == "cudapp":
        config.device = "cudapp"
        config.cudapp = config.cudapp.strip() or "2"
    elif lower.startswith("cuda:"):
        spec = device.split(":", 1)[1].strip()
        if spec.isdigit() or spec == "":
            config.device = "cuda"
            config.cuda_device_id = spec or "0"
        elif _is_valid_tp_spec(spec):
            config.device = "tp"
            config.tp = spec
    elif lower.startswith("multicuda:"):
        spec = device.split(":", 1)[1].strip()
        if _is_valid_tp_spec(spec):
            config.device = "tp"
            config.tp = spec
        elif spec.isdigit():
            config.device = "cuda"
            config.cuda_device_id = spec
        else:
            return
    elif lower == "multicuda":
        config.device = "tp"
        config.tp = config.tp.strip() or "2"


def normalize_moe_hybrid_config(config: DeployConfig, has_enable_field: bool, has_layers_field: bool):
    moe_device = str(config.moe_device).strip().lower()
    if not has_enable_field and moe_device in ("numa", "disk"):
        config.enable_moe_hybrid = True
        if not has_layers_field:
            config.moe_device_layers = "-1"

    if config.enable_moe_hybrid:
        if moe_device not in ("numa", "disk"):
            config.moe_device = "numa"
        else:
            config.moe_device = moe_device
        if not str(config.moe_device_layers).strip():
            config.moe_device_layers = "10000"


def config_from_dict(data: dict) -> DeployConfig:
    config = DeployConfig()
    valid_keys = {field.name for field in fields(DeployConfig)}
    has_enable_moe_hybrid = "enable_moe_hybrid" in data
    has_moe_device_layers = "moe_device_layers" in data
    for key, value in data.items():
        if key in valid_keys:
            setattr(config, key, value)
    normalize_main_device_config(config)
    normalize_moe_hybrid_config(config, has_enable_moe_hybrid, has_moe_device_layers)
    return config


def config_title(config: DeployConfig) -> str:
    if config.name.strip():
        return config.name.strip()
    model = config.model.strip() or "未设置模型"
    return f"{config.command} {model}"


def next_config_name(configs: Sequence[DeployConfig]) -> str:
    used_names = {config.name.strip() for config in configs if config.name.strip()}
    index = 1
    while True:
        name = f"配置{index}"
        if name not in used_names:
            return name
        index += 1


def is_default_config_name(name: str) -> bool:
    name = name.strip()
    return name.startswith("配置") and name[2:].isdigit()


def new_deploy_config(configs: Sequence[DeployConfig]) -> DeployConfig:
    config = DeployConfig()
    config.name = next_config_name(configs)
    return config


def apply_command_defaults(config: DeployConfig, old_command: str, new_command: str):
    if old_command == new_command:
        return
    if new_command == "server" and config.port in ("", "1616"):
        config.port = "8080"
    elif new_command == "webui" and config.port in ("", "8080"):
        config.port = "1616"


def apply_model_name_default(config: DeployConfig, old_model: str, new_model: str):
    if old_model == new_model:
        return
    if not is_default_config_name(config.name):
        return
    default_name = default_model_name_from_path(new_model)
    if default_name:
        config.name = default_name


def apply_main_device_defaults(config: DeployConfig, old_device: str, new_device: str):
    if old_device == new_device:
        return
    if new_device == "cuda" and not config.cuda_device_id.strip():
        config.cuda_device_id = "0"
    elif new_device == "cudapp" and not config.cudapp.strip():
        config.cudapp = "2"
    elif new_device == "tp" and not config.tp.strip():
        config.tp = "2"


def apply_moe_hybrid_defaults(config: DeployConfig):
    if config.moe_device not in ("numa", "disk"):
        config.moe_device = "numa"
    if not str(config.moe_device_layers).strip():
        config.moe_device_layers = "10000"


def load_saved_configs(path: Optional[str] = None) -> List[DeployConfig]:
    path = path or get_saved_commands_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    raw_commands = data.get("commands", data) if isinstance(data, dict) else data
    if not isinstance(raw_commands, list):
        return []
    return [config_from_dict(item) for item in raw_commands if isinstance(item, dict)]


def save_saved_configs(configs: Sequence[DeployConfig], path: Optional[str] = None):
    path = path or get_saved_commands_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "version": 1,
        "commands": [asdict(config) for config in configs],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def complete_path_prefix(raw_prefix: str, directories_only: bool = False) -> List[str]:
    raw_prefix = raw_prefix.strip()
    if raw_prefix.endswith(("/", os.sep)):
        raw_dir = raw_prefix
        partial = ""
    else:
        raw_dir = os.path.dirname(raw_prefix)
        partial = os.path.basename(raw_prefix)

    search_dir = os.path.expanduser(raw_dir or ".")
    if not os.path.isdir(search_dir):
        return []

    try:
        names = os.listdir(search_dir)
    except OSError:
        return []

    matches = []
    for name in names:
        if partial and not name.startswith(partial):
            continue
        full_path = os.path.join(search_dir, name)
        is_dir = os.path.isdir(full_path)
        if directories_only and not is_dir:
            continue
        if raw_dir:
            separator = "" if raw_dir.endswith(("/", os.sep)) else os.sep
            value = raw_dir + separator + name
        else:
            value = name
        if is_dir:
            value += os.sep
        matches.append((not is_dir, value))

    matches.sort(key=lambda item: item[1].lower())
    return [value for _, value in matches]


def build_fastllm_argv(config: DeployConfig) -> List[str]:
    argv = ["ftllm", config.command]
    model = _expand_user_path(config.model.strip())
    if model:
        argv.append(model)

    device, tp = _resolve_main_device_args(config)
    dtype = _resolve_custom(config.dtype, config.dtype_custom)
    moe_dtype = _resolve_custom(config.moe_dtype, config.moe_dtype_custom)

    _add_option(argv, "--device", device)
    _add_option(argv, "--tp", tp)
    if config.enable_moe_hybrid:
        _add_option(argv, "--moe_device", config.moe_device.strip())
        _add_option(argv, "--moe_device_layers", config.moe_device_layers.strip())
    _add_option(argv, "--gpu_mem_ratio", _optional_text(config.gpu_mem_ratio))
    _add_option(argv, "--chunked_prefill_size", _optional_text(config.chunked_prefill_size))
    _add_option(argv, "--kv_cache_dtype", _optional_text(config.kv_cache_dtype))
    _add_option(argv, "--moe_atype", _optional_text(config.moe_atype))
    _add_option(argv, "--enable_thinking", _optional_text(config.enable_thinking))
    _add_option(argv, "--tokens", _optional_text(config.tokens))
    _add_option(argv, "--dtype", dtype)
    _add_option(argv, "--moe_dtype", moe_dtype)
    _add_option(argv, "-t", _optional_text(config.threads))
    _add_option(argv, "--kv_cache_limit", _optional_text(config.kv_cache_limit))
    _add_option(argv, "--mtp", _optional_text(config.mtp))
    _add_option(argv, "--max_batch", _optional_text(config.max_batch))
    _add_option(argv, "--cache_dir", _expand_user_path(config.cache_dir.strip()))
    if is_gguf_model(config.model):
        _add_option(argv, "--ori", _expand_user_path(config.ori.strip()))

    if config.command == "server":
        _add_option(argv, "--model_name", effective_model_name(config))
        _add_option(argv, "--host", config.host.strip())
        _add_option(argv, "--port", config.port.strip())
        _add_option(argv, "--api_key", config.api_key.strip())
        _add_option(argv, "--temperature", config.temperature.strip())
        _add_option(argv, "--top_p", config.top_p.strip())
        _add_option(argv, "--top_k", config.top_k.strip())
        _add_option(argv, "--repeat_penalty", config.repeat_penalty.strip())
        if config.hide_input:
            argv.append("--hide_input")
    elif config.command == "webui":
        _add_option(argv, "--port", config.port.strip())

    extra_args = config.extra_args.strip()
    if extra_args:
        argv.extend(shlex.split(extra_args))
    return argv


def build_fastllm_command(config: DeployConfig) -> str:
    try:
        env_prefix = _env_prefix(config)
    except ValueError as exc:
        env_prefix = [f"<环境变量错误:{exc}>"]
    return " ".join(env_prefix + [shlex.quote(part) for part in build_fastllm_argv(config)])


def validate_config(config: DeployConfig) -> List[str]:
    errors = []
    model_path = _expand_user_path(config.model.strip())
    if not model_path:
        errors.append("模型路径不能为空。")
    elif is_gguf_model(model_path):
        if not os.path.isfile(model_path):
            errors.append("GGUF模型路径必须是已存在的本地 .gguf 文件。")
    elif not os.path.isdir(model_path):
        errors.append("模型路径必须是已存在的本地模型目录，或 .gguf 文件。")

    ori_path = _expand_user_path(config.ori.strip())
    if is_gguf_model(config.model) and ori_path and not os.path.isdir(ori_path):
        errors.append("模型配置文件夹必须是已存在的本地目录。")

    if config.command in ("server", "webui"):
        try:
            port = int(config.port)
            if port < 1 or port > 65535:
                errors.append("端口必须在 1-65535 之间。")
        except ValueError:
            errors.append("端口必须是整数。")

    for label, value in (
        ("预处理分片大小", config.chunked_prefill_size),
        ("最大Batch", config.max_batch),
        ("tokens数量", config.tokens),
        ("线程数", config.threads),
    ):
        if not _is_positive_int_or_auto(value):
            errors.append(f"{label}必须是正整数或 auto。")

    if not _is_mtp_value(config.mtp):
        errors.append("MTP 必须是 0-8 的整数或 auto。")

    if not _is_ratio(config.gpu_mem_ratio):
        errors.append("显存利用率必须是 0 到 1 之间的数字，例如 0.9。")

    if config.command == "server" and not config.host.strip():
        errors.append("监听地址不能为空。")
    if config.device == "cuda" and not _is_valid_cuda_device_id(config.cuda_device_id):
        errors.append("CUDA卡号必须是非负整数；留空表示 0。")
    if config.device == "cudapp" and not _is_valid_cudapp_spec(config.cudapp):
        errors.append("串行参数格式不对。请输入正整数卡数，例如 4；或输入至少两个卡号，例如 0,1,2。")
    if config.device == "tp" and not _is_valid_tp_spec(config.tp):
        errors.append("TP卡数/ID格式不对。请输入卡数，例如 1 或 4；或输入卡号，例如 0、cuda:1、0,2,3。")
    if config.device == "custom" and not config.device_custom.strip():
        errors.append("选择自定义主设备时必须填写自定义主设备。")
    if config.enable_moe_hybrid:
        if config.moe_device not in ("numa", "disk"):
            errors.append("MOE推理设备只能选择 NUMA CPU 或 Disk。")
        if not _is_valid_moe_device_layers(config.moe_device_layers):
            errors.append("MOE设备层数格式不对。请输入正整数，例如 8；或输入 -1 表示全部 MOE 层。")
    if config.dtype == "custom" and not config.dtype_custom.strip():
        errors.append("选择自定义权重类型时必须填写自定义权重类型。")
    if config.moe_dtype == "custom" and not config.moe_dtype_custom.strip():
        errors.append("选择自定义MOE类型时必须填写自定义MOE类型。")

    if not _is_optional_float(config.temperature, min_value=0):
        errors.append("temperature 必须是大于等于 0 的数字，或留空使用模型默认值。")
    if not _is_optional_float(config.top_p, min_value=0, max_value=1):
        errors.append("top_p 必须是 0 到 1 之间的数字，或留空使用模型默认值。")
    if not _is_optional_positive_int(config.top_k):
        errors.append("top_k 必须是正整数，或留空使用模型默认值。")
    if not _is_optional_float(config.repeat_penalty, min_value=0):
        errors.append("repeat_penalty 必须是大于等于 0 的数字，或留空使用模型默认值。")

    if config.extra_args.strip():
        try:
            shlex.split(config.extra_args)
        except ValueError as exc:
            errors.append(f"额外参数无法解析: {exc}")
    if config.env_vars.strip():
        try:
            parse_env_vars(config.env_vars)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def _tui_escdelay_ms() -> int:
    raw_value = os.environ.get("FASTLLM_TUI_ESCDELAY", os.environ.get("ESCDELAY", str(DEFAULT_ESCDELAY_MS)))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_ESCDELAY_MS
    return max(0, min(value, 1000))


def _configure_curses_keys(stdscr):
    try:
        curses.set_escdelay(_tui_escdelay_ms())
    except (AttributeError, curses.error):
        pass
    try:
        stdscr.notimeout(False)
    except curses.error:
        pass


class FastllmCursesTUI:
    def __init__(self, config: Optional[DeployConfig] = None):
        self.config = config or DeployConfig()
        self.selected = 0
        self.home_selected = 0
        self.edit_index: Optional[int] = None
        self.saved_configs = load_saved_configs()
        self.home_button_selected = self._default_home_button_index(self._home_buttons(None))
        self.form_button_selected = 0
        self.form_show_advanced = False
        self.download_button_selected = 0
        self.status = "上下选择命令，左右/Tab选择按钮，Enter执行按钮。"

    def run(self) -> Tuple[str, DeployConfig]:
        return curses.wrapper(self._main)

    def _main(self, stdscr) -> Tuple[str, DeployConfig]:
        curses.curs_set(0)
        stdscr.keypad(True)
        _configure_curses_keys(stdscr)
        return self._home_loop(stdscr)

    def _home_loop(self, stdscr) -> Tuple[str, DeployConfig]:
        pending_delete: Optional[int] = None
        while True:
            self.home_selected = min(self.home_selected, max(0, len(self.saved_configs) - 1))
            buttons = self._home_buttons(pending_delete)
            self.home_button_selected = min(self.home_button_selected, max(0, len(buttons) - 1))
            self._draw_home(stdscr, pending_delete)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                self.home_selected = _previous_index(self.home_selected, len(self.saved_configs))
                pending_delete = None
            elif key in (curses.KEY_DOWN, ord("j")):
                self.home_selected = _next_index(self.home_selected, len(self.saved_configs))
                pending_delete = None
            elif key in (curses.KEY_LEFT, curses.KEY_BTAB):
                self.home_button_selected = (self.home_button_selected - 1) % len(buttons)
            elif key in (curses.KEY_RIGHT, 9):
                self.home_button_selected = (self.home_button_selected + 1) % len(buttons)
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                result, pending_delete = self._run_home_action(stdscr, buttons[self.home_button_selected][0], pending_delete)
                if result is not None:
                    return result
            elif key in (ord("n"), ord("N")):
                result, pending_delete = self._run_home_action(stdscr, "new", pending_delete)
                if result is not None:
                    return result
            elif key in (ord("e"), ord("E")):
                result, pending_delete = self._run_home_action(stdscr, "edit", pending_delete)
                if result is not None:
                    return result
            elif key in (ord("s"), ord("S")):
                result, pending_delete = self._run_home_action(stdscr, "start", pending_delete)
                if result is not None:
                    return result
            elif key in (ord("d"), ord("D")):
                result, pending_delete = self._run_home_action(stdscr, "delete", pending_delete)
                if result is not None:
                    return result
            elif key in (ord("q"), ord("Q"), 27):
                return "quit", DeployConfig()

    def _home_buttons(self, pending_delete: Optional[int]) -> Sequence[Choice]:
        if not self.saved_configs:
            return (("new", "新建"), ("download", "下载模型"), ("quit", "退出"))
        delete_label = "确认删除" if pending_delete == self.home_selected else "删除"
        return (
            ("start", "启动"),
            ("edit", "编辑"),
            ("new", "新建"),
            ("delete", delete_label),
            ("download", "下载模型"),
            ("quit", "退出"),
        )

    def _default_home_button_index(self, buttons: Sequence[Choice]) -> int:
        preferred = "start" if self.saved_configs else "new"
        return next((index for index, (action, _) in enumerate(buttons) if action == preferred), 0)

    def _reset_home_button_selection(self, pending_delete: Optional[int] = None):
        self.home_button_selected = self._default_home_button_index(self._home_buttons(pending_delete))

    def _run_home_action(
        self,
        stdscr,
        action: str,
        pending_delete: Optional[int],
    ) -> Tuple[Optional[Tuple[str, DeployConfig]], Optional[int]]:
        if action == "new":
            result = self._form_loop(stdscr, new_deploy_config(self.saved_configs), None)
            if result[0] in ("start", "quit"):
                return result, None
            self.status = "上下选择命令，左右/Tab选择按钮，Enter执行按钮。"
            self._reset_home_button_selection()
            return None, None
        if action == "download":
            result = self._download_loop(stdscr)
            if result[0] in ("modelscope_download", "quit"):
                return result, None
            self.status = "上下选择命令，左右/Tab选择按钮，Enter执行按钮。"
            self._reset_home_button_selection()
            return None, None
        if action == "edit":
            if not self.saved_configs:
                result = self._form_loop(stdscr, new_deploy_config(self.saved_configs), None)
            else:
                result = self._form_loop(stdscr, clone_config(self.saved_configs[self.home_selected]), self.home_selected)
            if result[0] in ("start", "quit"):
                return result, None
            self.status = "上下选择命令，左右/Tab选择按钮，Enter执行按钮。"
            self._reset_home_button_selection()
            return None, None
        if action == "start":
            if self.saved_configs:
                config = clone_config(self.saved_configs[self.home_selected])
                errors = validate_config(config)
                if errors:
                    self.status = errors[0]
                else:
                    return ("start", config), None
            return None, None
        if action == "delete":
            if self.saved_configs:
                if pending_delete == self.home_selected:
                    removed = config_title(self.saved_configs.pop(self.home_selected))
                    save_saved_configs(self.saved_configs)
                    self.home_selected = min(self.home_selected, max(0, len(self.saved_configs) - 1))
                    self.status = f"已删除: {removed}"
                    self._reset_home_button_selection()
                    return None, None
                self.status = "再次执行确认删除按钮以删除当前命令。"
                return None, self.home_selected
            return None, None
        if action == "quit":
            return ("quit", DeployConfig()), None
        return None, pending_delete

    def _form_loop(self, stdscr, config: DeployConfig, edit_index: Optional[int]) -> Tuple[str, DeployConfig]:
        self.config = config
        self.edit_index = edit_index
        self.selected = 0
        self.form_button_selected = 0
        self.form_show_advanced = False
        self.status = "基础配置只显示常用参数；上下选择高级选项，Enter进入更多参数。"
        while True:
            visible_fields = self._visible_fields()
            selectable_count = len(visible_fields) + 1
            self.selected = min(self.selected, max(0, selectable_count - 1))
            buttons = self._form_buttons()
            self.form_button_selected = min(self.form_button_selected, max(0, len(buttons) - 1))
            self._draw(stdscr, visible_fields)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                self.selected = _previous_index(self.selected, selectable_count)
            elif key in (curses.KEY_DOWN, ord("j")):
                self.selected = _next_index(self.selected, selectable_count)
            elif key in (curses.KEY_LEFT, curses.KEY_BTAB):
                self.form_button_selected = (self.form_button_selected - 1) % len(buttons)
            elif key in (curses.KEY_RIGHT, 9):
                self.form_button_selected = (self.form_button_selected + 1) % len(buttons)
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                action = "advanced" if self._advanced_toggle_selected(visible_fields) else buttons[self.form_button_selected][0]
                result = self._run_form_action(stdscr, action, visible_fields)
                if result is not None:
                    return result
            elif key in (ord("s"), ord("S")):
                result = self._run_form_action(stdscr, "start", visible_fields)
                if result is not None:
                    return result
            elif key in (ord("w"), ord("W")):
                result = self._run_form_action(stdscr, "save", visible_fields)
                if result is not None:
                    return result
            elif key in (ord("a"), ord("A")):
                result = self._run_form_action(stdscr, "advanced", visible_fields)
                if result is not None:
                    return result
            elif key in (ord("b"), ord("B"), 27):
                return "back", self.config
            elif key in (ord("q"), ord("Q")):
                return "quit", self.config

    def _form_buttons(self) -> Sequence[Choice]:
        return (
            ("edit", "编辑"),
            ("save", "保存"),
            ("start", "启动"),
            ("back", "返回"),
            ("quit", "退出"),
        )

    def _run_form_action(
        self,
        stdscr,
        action: str,
        visible_fields: Sequence[FormField],
    ) -> Optional[Tuple[str, DeployConfig]]:
        if action == "edit":
            if self._advanced_toggle_selected(visible_fields):
                self._toggle_advanced_options()
                return None
            self._edit_field(stdscr, visible_fields[self.selected])
            return None
        if action == "advanced":
            self._toggle_advanced_options()
            return None
        if action == "save":
            errors = validate_config(self.config)
            if errors:
                self.status = errors[0]
                return None
            self._save_current_config()
            return "back", self.config
        if action == "start":
            errors = validate_config(self.config)
            if errors:
                self.status = errors[0]
                return None
            self._save_current_config()
            return "start", self.config
        if action == "back":
            return "back", self.config
        if action == "quit":
            return "quit", self.config
        return None

    def _advanced_toggle_label(self) -> str:
        return "基础选项" if self.form_show_advanced else "高级选项"

    def _advanced_toggle_help(self) -> str:
        if self.form_show_advanced:
            return "返回基础配置。"
        return "进入高级选项，配置显存、缓存、batch、思考、采样、API Key、其它参数和环境变量。"

    def _advanced_toggle_selected(self, visible_fields: Sequence[FormField]) -> bool:
        return self.selected >= len(visible_fields)

    def _toggle_advanced_options(self):
        self.form_show_advanced = not self.form_show_advanced
        self.selected = 0
        if self.form_show_advanced:
            self.status = "高级选项已打开；采样参数在二级菜单中配置。"
        else:
            self.status = "已返回基础配置。"

    def _download_loop(self, stdscr) -> Tuple[str, Union[ModelScopeDownloadConfig, DeployConfig]]:
        config = ModelScopeDownloadConfig()
        update_default_modelscope_target_dir(config)
        field_selected = 0
        self.download_button_selected = 0
        self.status = "从模型列表选择模型；下载目录默认使用系统缓存路径。"
        while True:
            visible_fields = [field for field in DOWNLOAD_FIELDS if field.is_visible(config)]
            field_selected = min(field_selected, len(visible_fields) - 1)
            buttons = self._download_buttons()
            self.download_button_selected = min(self.download_button_selected, max(0, len(buttons) - 1))
            self._draw_download(stdscr, config, visible_fields, field_selected)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                field_selected = _previous_index(field_selected, len(visible_fields))
            elif key in (curses.KEY_DOWN, ord("j")):
                field_selected = _next_index(field_selected, len(visible_fields))
            elif key in (curses.KEY_LEFT, curses.KEY_BTAB):
                self.download_button_selected = (self.download_button_selected - 1) % len(buttons)
            elif key in (curses.KEY_RIGHT, 9):
                self.download_button_selected = (self.download_button_selected + 1) % len(buttons)
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                result = self._run_download_action(stdscr, buttons[self.download_button_selected][0], config, visible_fields, field_selected)
                if result is not None:
                    return result
            elif key in (ord("f"), ord("F")):
                result = self._run_download_action(stdscr, "edit", config, visible_fields, field_selected)
                if result is not None:
                    return result
            elif key in (ord("d"), ord("D")):
                result = self._run_download_action(stdscr, "download", config, visible_fields, field_selected)
                if result is not None:
                    return result
            elif key in (ord("b"), ord("B"), 27):
                return "back", DeployConfig()
            elif key in (ord("q"), ord("Q")):
                return "quit", DeployConfig()

    def _download_buttons(self) -> Sequence[Choice]:
        return (
            ("edit", "编辑字段"),
            ("download", "下载模型"),
            ("back", "返回"),
            ("quit", "退出"),
        )

    def _run_download_action(
        self,
        stdscr,
        action: str,
        config: ModelScopeDownloadConfig,
        visible_fields: Sequence[FormField],
        field_selected: int,
    ) -> Optional[Tuple[str, Union[ModelScopeDownloadConfig, DeployConfig]]]:
        if action == "edit":
            field = visible_fields[field_selected]
            if field.kind == "choice":
                old_model_id = _resolve_modelscope_model_id(config)
                value = self._choose_download(stdscr, field, config)
                if value is None:
                    return None
                setattr(config, field.key, value)
                if field.key == "model_id":
                    update_default_modelscope_target_dir(config, old_model_id)
                self.status = f"{field.label}已更新。"
                return None
            else:
                old_model_id = _resolve_modelscope_model_id(config)
                value = self._input_download_text(stdscr, field, config)
                if value is not None:
                    setattr(config, field.key, value)
                    if field.key == "model_id_custom":
                        update_default_modelscope_target_dir(config, old_model_id)
                    self.status = f"{field.label}已更新。"
            return None
        if action == "download":
            errors = validate_modelscope_download(config)
            if errors:
                self.status = errors[0]
                return None
            return "modelscope_download", config
        if action == "back":
            return "back", DeployConfig()
        if action == "quit":
            return "quit", DeployConfig()
        return None

    def _draw_download(
        self,
        stdscr,
        config: ModelScopeDownloadConfig,
        fields: Sequence[FormField],
        field_selected: int,
    ):
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 20 or width < 78:
            self._safe_addstr(stdscr, 0, 0, "终端窗口太小，至少需要 78x20。请调整窗口大小。", curses.A_BOLD)
            self._refresh(stdscr)
            return

        panel = self._draw_panel(stdscr, "ModelScope 模型下载", 20)
        label_x = panel["left"]
        value_x = panel["left"] + 18

        self._safe_addstr(stdscr, panel["content_top"], panel["left"], "上下选择字段，左右/Tab选择按钮，Enter执行按钮。")

        form_top = panel["content_top"] + 2
        for row, index in enumerate(range(len(fields)), start=form_top):
            field = fields[index]
            attr = curses.A_REVERSE if index == field_selected else curses.A_NORMAL
            label = f"{field.label}:"
            value = getattr(config, field.key)
            if field.kind == "choice":
                value = _choice_label(field.choices, value)
            value_text = str(value) if value != "" else "(空)"
            self._safe_addstr(stdscr, row, label_x, label.ljust(16), attr)
            self._safe_addstr(stdscr, row, value_x, _clip_display(value_text, max(1, panel["right"] - value_x)), attr)

        model_id = _resolve_modelscope_model_id(config) or "<model_id>"
        command = f"modelscope snapshot_download {model_id} -> {config.target_dir or '<target_dir>'}"
        self._safe_addstr(stdscr, panel["content_bottom"] - 3, panel["left"], _clip_display(command, panel["width"]), curses.A_DIM)
        self._draw_buttons(stdscr, panel["content_bottom"] - 1, panel["left"], self._download_buttons(), self.download_button_selected, panel["width"])
        self._safe_addstr(stdscr, panel["content_bottom"], panel["left"], _clip_display(self.status, panel["width"]), curses.A_BOLD)
        self._refresh(stdscr)

    def _input_download_text(self, stdscr, field: FormField, config: ModelScopeDownloadConfig) -> Optional[str]:
        if _is_download_path_completion_field(field):
            return self._input_text_with_completion_for_value(
                stdscr,
                field,
                str(getattr(config, field.key)),
                directories_only=True,
            )
        return self._input_text_for_value(stdscr, field, str(getattr(config, field.key)))

    def _choose_download(self, stdscr, field: FormField, config: ModelScopeDownloadConfig) -> Optional[str]:
        if field.key == "model_id":
            return self._choose_modelscope_model(stdscr, field, config)
        choices = list(field.choices)
        current = getattr(config, field.key)
        index = next((i for i, (value, _) in enumerate(choices) if value == current), 0)
        while True:
            self._draw_download_choice_popup(stdscr, field, choices, index, config)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                index = _previous_index(index, len(choices))
            elif key in (curses.KEY_DOWN, ord("j")):
                index = _next_index(index, len(choices))
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                return choices[index][0]
            elif key in (27, ord("q"), ord("Q")):
                self.status = "已取消选择。"
                return None

    def _choose_modelscope_model(self, stdscr, field: FormField, config: ModelScopeDownloadConfig) -> Optional[str]:
        current = getattr(config, field.key)
        group_index = _modelscope_group_index_for_model(current)
        while True:
            group_index = self._choose_modelscope_group(stdscr, field, config, group_index)
            if group_index is None:
                return None
            _, group_label, choices = MODELSCOPE_MODEL_GROUPS[group_index]
            model_index = next((i for i, (value, _) in enumerate(choices) if value == current), 0)
            selected = self._choose_modelscope_model_in_group(
                stdscr,
                field,
                config,
                group_label,
                choices,
                model_index,
            )
            if selected == "__back__":
                continue
            return selected

    def _choose_modelscope_group(
        self,
        stdscr,
        field: FormField,
        config: ModelScopeDownloadConfig,
        selected: int,
    ) -> Optional[int]:
        choices = list(MODELSCOPE_MODEL_GROUP_CHOICES)
        while True:
            self._draw_download_choice_popup(stdscr, field, choices, selected, config, "选择模型分类")
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = _previous_index(selected, len(choices))
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = _next_index(selected, len(choices))
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                return selected
            elif key in (27, ord("q"), ord("Q")):
                self.status = "已取消选择。"
                return None

    def _choose_modelscope_model_in_group(
        self,
        stdscr,
        field: FormField,
        config: ModelScopeDownloadConfig,
        group_label: str,
        choices: Sequence[Choice],
        selected: int,
    ) -> str:
        choices = list(choices)
        while True:
            self._draw_download_choice_popup(stdscr, field, choices, selected, config, group_label)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = _previous_index(selected, len(choices))
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = _next_index(selected, len(choices))
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                return choices[selected][0]
            elif key in (curses.KEY_LEFT, curses.KEY_BACKSPACE, 127, ord("b"), ord("B")):
                return "__back__"
            elif key in (27, ord("q"), ord("Q")):
                self.status = "已返回模型分类。"
                return "__back__"

    def _draw_download_choice_popup(
        self,
        stdscr,
        field: FormField,
        choices: Sequence[Choice],
        selected: int,
        config: ModelScopeDownloadConfig,
        title: Optional[str] = None,
    ):
        visible_fields = [item for item in DOWNLOAD_FIELDS if item.is_visible(config)]
        selected_field = next((i for i, item in enumerate(visible_fields) if item.key == field.key), 0)
        self._draw_download(stdscr, config, visible_fields, selected_field)
        height, width = stdscr.getmaxyx()
        box_width = min(max(48, max(_display_width(label) for _, label in choices) + 8), width - 4)
        box_height = min(len(choices) + 4, height - 4)
        top = max(1, (height - box_height) // 2)
        left = max(2, (width - box_width) // 2)
        self._safe_addstr(stdscr, top, left, "+" + "-" * (box_width - 2) + "+", curses.A_BOLD)
        self._safe_addstr(stdscr, top + 1, left, "| " + _pad_display(title or field.label, box_width - 4) + " |", curses.A_BOLD)
        for row, (_, label) in enumerate(choices[: box_height - 4], start=top + 2):
            choice_index = row - top - 2
            attr = curses.A_REVERSE if choice_index == selected else curses.A_NORMAL
            self._safe_addstr(stdscr, row, left, "| " + _pad_display(label, box_width - 4) + " |", attr)
        self._safe_addstr(stdscr, top + box_height - 1, left, "+" + "-" * (box_width - 2) + "+", curses.A_BOLD)
        self._refresh(stdscr)

    def _save_current_config(self):
        saved = clone_config(self.config)
        if self.edit_index is None:
            self.saved_configs.append(saved)
            self.edit_index = len(self.saved_configs) - 1
            self.home_selected = self.edit_index
        else:
            self.saved_configs[self.edit_index] = saved
            self.home_selected = self.edit_index
        save_saved_configs(self.saved_configs)

    def _draw_home(self, stdscr, pending_delete: Optional[int]):
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 14 or width < 72:
            self._safe_addstr(stdscr, 0, 0, "终端窗口太小，至少需要 72x14。请调整窗口大小。", curses.A_BOLD)
            self._refresh(stdscr)
            return

        panel = self._draw_panel(stdscr, "fastllm TUI 命令列表", 14)
        self._safe_addstr(stdscr, panel["content_top"], panel["left"], _clip_display(f"配置文件: {get_saved_commands_path()}", panel["width"]), curses.A_DIM)

        list_top = panel["content_top"] + 2
        list_height = max(1, panel["content_bottom"] - 6 - list_top + 1)
        if not self.saved_configs:
            self._safe_addstr(stdscr, list_top, panel["left"], "还没有保存的命令。按 N 新建一个部署配置。", curses.A_DIM)
        else:
            top_index = 0
            if self.home_selected >= list_height:
                top_index = self.home_selected - list_height + 1
            bottom_index = min(len(self.saved_configs), top_index + list_height)
            for row, index in enumerate(range(top_index, bottom_index), start=list_top):
                config = self.saved_configs[index]
                attr = curses.A_REVERSE if index == self.home_selected else curses.A_NORMAL
                summary = f"{index + 1}. {config_title(config)}"
                self._safe_addstr(stdscr, row, panel["left"], _clip_display(summary, panel["width"]), attr)

        selected_config = self.saved_configs[self.home_selected] if self.saved_configs else None
        if selected_config is not None:
            command = build_fastllm_command(selected_config)
            self._safe_addstr(stdscr, panel["content_bottom"] - 4, panel["left"], "当前命令:", curses.A_BOLD)
            self._safe_addstr(stdscr, panel["content_bottom"] - 3, panel["left"], _clip_display(command, panel["width"]))
        buttons = self._home_buttons(pending_delete)
        self._draw_buttons(stdscr, panel["content_bottom"] - 1, panel["left"], buttons, self.home_button_selected, panel["width"])
        self._safe_addstr(stdscr, panel["content_bottom"], panel["left"], _clip_display(self.status, panel["width"]), curses.A_BOLD)
        self._refresh(stdscr)

    def _draw_buttons(
        self,
        stdscr,
        y: int,
        x: int,
        buttons: Sequence[Choice],
        selected: int,
        max_width: Optional[int] = None,
    ):
        height, width = stdscr.getmaxyx()
        if y < 0 or y >= height:
            return
        right = min(width, x + max_width) if max_width is not None else width
        cur_x = x
        for index, (_, label) in enumerate(buttons):
            text = f" [ {label} ] "
            text_width = _display_width(text)
            if cur_x + text_width > right:
                break
            attr = curses.A_REVERSE | curses.A_BOLD if index == selected else curses.A_NORMAL
            self._safe_addstr(stdscr, y, cur_x, text, attr)
            cur_x += text_width + 1

    @staticmethod
    def _panel_frame(stdscr, min_height: int = 18) -> dict:
        height, width = stdscr.getmaxyx()
        panel_width = min(CONTENT_MAX_WIDTH + 2, max(2, width))
        panel_height = min(max(min_height, min(PANEL_MAX_HEIGHT, height)), height)
        panel_left = max(0, (width - panel_width) // 2)
        panel_top = max(0, (height - panel_height) // 2)
        panel_bottom = panel_top + panel_height - 1
        content_left = panel_left + 1 + PANEL_PADDING_X
        content_right = panel_left + panel_width - 1 - PANEL_PADDING_X
        content_width = max(1, content_right - content_left)
        content_top = min(panel_bottom, panel_top + 1 + PANEL_PADDING_Y)
        content_bottom = max(content_top, panel_bottom - 1 - PANEL_PADDING_Y)
        return {
            "top": panel_top,
            "bottom": panel_bottom,
            "content_top": content_top,
            "content_bottom": content_bottom,
            "left": content_left,
            "right": content_left + content_width,
            "width": content_width,
            "panel_left": panel_left,
            "panel_width": panel_width,
            "panel_height": panel_height,
        }

    def _draw_panel(self, stdscr, title: str, min_height: int = 18) -> dict:
        panel = self._panel_frame(stdscr, min_height)
        top = panel["top"]
        bottom = panel["bottom"]
        left = panel["panel_left"]
        width = panel["panel_width"]
        if width < 2:
            return panel

        top_line = "+" + "-" * (width - 2) + "+"
        if title:
            title_text = f"  {title}  "
            title_width = _display_width(title_text)
            if title_width < width - 2:
                top_line = "+" + title_text + "-" * (width - 2 - title_width) + "+"
        bottom_line = "+" + "-" * (width - 2) + "+"
        self._safe_addstr(stdscr, top, left, top_line, curses.A_BOLD)
        for row in range(top + 1, bottom):
            self._safe_addstr(stdscr, row, left, "|", curses.A_DIM)
            self._safe_addstr(stdscr, row, left + width - 1, "|", curses.A_DIM)
        self._safe_addstr(stdscr, bottom, left, bottom_line, curses.A_BOLD)
        return panel

    def _visible_fields(self) -> List[FormField]:
        return visible_deploy_fields(self.config, self.form_show_advanced)

    def _draw(self, stdscr, fields: Sequence[FormField]):
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 18 or width < 72:
            self._safe_addstr(stdscr, 0, 0, "终端窗口太小，至少需要 72x18。请调整窗口大小。", curses.A_BOLD)
            self._refresh(stdscr)
            return

        title = "fastllm TUI 部署向导 - 高级选项" if self.form_show_advanced else "fastllm TUI 部署向导 - 基础配置"
        panel = self._draw_panel(stdscr, title, 18)
        label_x = panel["left"]
        value_x = panel["left"] + 18
        buttons = self._form_buttons()
        self._safe_addstr(stdscr, panel["content_top"], panel["left"], "上下选择字段或选项，左右/Tab选择底部按钮，Enter执行。")

        help_y = panel["content_bottom"] - 5
        form_top = panel["content_top"] + 2
        form_height = max(1, help_y - form_top - 1)
        item_count = len(fields) + 1
        top_index = 0
        if self.selected >= form_height:
            top_index = self.selected - form_height + 1
        bottom_index = min(item_count, top_index + form_height)

        for row, index in enumerate(range(top_index, bottom_index), start=form_top):
            attr = curses.A_REVERSE if index == self.selected else curses.A_NORMAL
            if index < len(fields):
                field = fields[index]
                label = f"{field.label}:"
                value = self._display_value(field)
                self._safe_addstr(stdscr, row, label_x, label.ljust(16), attr)
                self._safe_addstr(stdscr, row, value_x, _clip_display(value, max(1, panel["right"] - value_x)), attr)
            else:
                text = f"[{self._advanced_toggle_label()}]"
                self._safe_addstr(stdscr, row, panel["left"], text, attr | curses.A_BOLD)

        help_line = self._advanced_toggle_help() if self._advanced_toggle_selected(fields) else fields[self.selected].help
        self._safe_addstr(stdscr, help_y, panel["left"], _clip_display(help_line, panel["width"]), curses.A_DIM)
        command = build_fastllm_command(self.config)
        self._safe_addstr(stdscr, panel["content_bottom"] - 4, panel["left"], "命令预览:", curses.A_BOLD)
        self._safe_addstr(stdscr, panel["content_bottom"] - 3, panel["left"], _clip_display(command, panel["width"]))
        if _display_width(command) > panel["width"]:
            self._safe_addstr(stdscr, panel["content_bottom"] - 2, panel["left"], _clip_display(command[panel["width"] :], panel["width"]))
        self._draw_buttons(stdscr, panel["content_bottom"] - 1, panel["left"], buttons, self.form_button_selected, panel["width"])
        self._safe_addstr(stdscr, panel["content_bottom"], panel["left"], _clip_display(self.status, panel["width"]), curses.A_BOLD)
        self._refresh(stdscr)

    def _display_value(self, field: FormField) -> str:
        if field.kind == "submenu":
            if field.key == "sampling_params":
                items = [
                    ("temperature", self.config.temperature),
                    ("top_p", self.config.top_p),
                    ("top_k", self.config.top_k),
                    ("repeat_penalty", self.config.repeat_penalty),
                ]
                active = [f"{name}={value}" for name, value in items if str(value).strip()]
                return ", ".join(active) if active else "(使用模型默认值)"
            return "进入"
        value = getattr(self.config, field.key)
        if field.kind == "bool":
            return "是" if value else "否"
        if field.kind == "choice":
            return _choice_label(field.choices, value)
        if field.key == "model_name" and value == "":
            default_name = default_model_name_from_path(self.config.model)
            return f"{default_name} (默认)" if default_name else "(默认使用模型路径名)"
        if value == "":
            return "(空)"
        return str(value)

    def _edit_field(self, stdscr, field: FormField):
        if field.kind == "submenu":
            if field.key == "sampling_params":
                self._sampling_loop(stdscr)
            else:
                self.status = "未知二级菜单。"
        elif field.kind == "bool":
            setattr(self.config, field.key, not getattr(self.config, field.key))
            if field.key == "enable_moe_hybrid" and getattr(self.config, field.key):
                apply_moe_hybrid_defaults(self.config)
            self.status = f"{field.label}已切换。"
        elif field.kind == "choice":
            value = self._choose(stdscr, field)
            if value is not None:
                old_command = self.config.command
                old_device = self.config.device
                setattr(self.config, field.key, value)
                if field.key == "command" and old_command != value:
                    self._apply_command_defaults(old_command, value)
                if field.key == "device" and old_device != value:
                    apply_main_device_defaults(self.config, old_device, value)
                self.status = f"{field.label}已更新。"
        else:
            old_model = self.config.model
            value = self._input_text(stdscr, field)
            if value is not None:
                setattr(self.config, field.key, value)
                if field.key == "model":
                    apply_model_name_default(self.config, old_model, value)
                self.status = f"{field.label}已更新。"

    def _apply_command_defaults(self, old_command: str, new_command: str):
        apply_command_defaults(self.config, old_command, new_command)

    def _sampling_loop(self, stdscr):
        selected = 0
        button_selected = 0
        buttons: Sequence[Choice] = (("edit", "编辑"), ("back", "返回"))
        self.status = "采样参数未填写时使用模型默认值。"
        while True:
            selected = min(selected, max(0, len(SAMPLING_FIELDS) - 1))
            button_selected = min(button_selected, max(0, len(buttons) - 1))
            self._draw_sampling(stdscr, selected, buttons, button_selected)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = _previous_index(selected, len(SAMPLING_FIELDS))
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = _next_index(selected, len(SAMPLING_FIELDS))
            elif key in (curses.KEY_LEFT, curses.KEY_BTAB):
                button_selected = (button_selected - 1) % len(buttons)
            elif key in (curses.KEY_RIGHT, 9):
                button_selected = (button_selected + 1) % len(buttons)
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                action = buttons[button_selected][0]
                if action == "edit":
                    value = self._input_text(stdscr, SAMPLING_FIELDS[selected])
                    if value is not None:
                        setattr(self.config, SAMPLING_FIELDS[selected].key, value)
                        self.status = f"{SAMPLING_FIELDS[selected].label}已更新。"
                elif action == "back":
                    self.status = "已返回高级选项。"
                    return
            elif key in (ord("f"), ord("F")):
                value = self._input_text(stdscr, SAMPLING_FIELDS[selected])
                if value is not None:
                    setattr(self.config, SAMPLING_FIELDS[selected].key, value)
                    self.status = f"{SAMPLING_FIELDS[selected].label}已更新。"
            elif key in (ord("b"), ord("B"), 27):
                self.status = "已返回高级选项。"
                return
            elif key in (ord("q"), ord("Q")):
                self.status = "已返回高级选项。"
                return

    def _draw_sampling(self, stdscr, selected: int, buttons: Sequence[Choice], button_selected: int):
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 18 or width < 72:
            self._safe_addstr(stdscr, 0, 0, "终端窗口太小，至少需要 72x18。请调整窗口大小。", curses.A_BOLD)
            self._refresh(stdscr)
            return

        panel = self._draw_panel(stdscr, "fastllm TUI 部署向导 - 采样参数", 18)
        label_x = panel["left"]
        value_x = panel["left"] + 18
        self._safe_addstr(stdscr, panel["content_top"], panel["left"], "上下选择采样字段，左右/Tab选择底部按钮，Enter执行。")

        form_top = panel["content_top"] + 2
        for row, index in enumerate(range(len(SAMPLING_FIELDS)), start=form_top):
            field = SAMPLING_FIELDS[index]
            attr = curses.A_REVERSE if index == selected else curses.A_NORMAL
            value = self._display_value(field)
            self._safe_addstr(stdscr, row, label_x, f"{field.label}:".ljust(16), attr)
            self._safe_addstr(stdscr, row, value_x, _clip_display(value, max(1, panel["right"] - value_x)), attr)

        help_y = panel["content_bottom"] - 5
        selected_field = SAMPLING_FIELDS[selected]
        self._safe_addstr(stdscr, help_y, panel["left"], _clip_display(selected_field.help, panel["width"]), curses.A_DIM)
        command = build_fastllm_command(self.config)
        self._safe_addstr(stdscr, panel["content_bottom"] - 4, panel["left"], "命令预览:", curses.A_BOLD)
        self._safe_addstr(stdscr, panel["content_bottom"] - 3, panel["left"], _clip_display(command, panel["width"]))
        if _display_width(command) > panel["width"]:
            self._safe_addstr(stdscr, panel["content_bottom"] - 2, panel["left"], _clip_display(command[panel["width"] :], panel["width"]))
        self._draw_buttons(stdscr, panel["content_bottom"] - 1, panel["left"], buttons, button_selected, panel["width"])
        self._safe_addstr(stdscr, panel["content_bottom"], panel["left"], _clip_display(self.status, panel["width"]), curses.A_BOLD)
        self._refresh(stdscr)

    def _choose(self, stdscr, field: FormField) -> Optional[str]:
        choices = list(field.choices)
        current = getattr(self.config, field.key)
        index = next((i for i, (value, _) in enumerate(choices) if value == current), 0)
        while True:
            self._draw_choice_popup(stdscr, field, choices, index)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                index = _previous_index(index, len(choices))
            elif key in (curses.KEY_DOWN, ord("j")):
                index = _next_index(index, len(choices))
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                return choices[index][0]
            elif key in (27, ord("q"), ord("Q")):
                self.status = "已取消选择。"
                return None

    def _draw_choice_popup(self, stdscr, field: FormField, choices: Sequence[Choice], selected: int):
        self._draw(stdscr, self._visible_fields())
        height, width = stdscr.getmaxyx()
        content_width = max([_display_width(field.label)] + [_display_width(label) for _, label in choices])
        box_width = min(max(42, content_width + 4), width - 4)
        box_height = min(len(choices) + 4, height - 4)
        top = max(1, (height - box_height) // 2)
        left = max(2, (width - box_width) // 2)
        inner_width = max(1, box_width - 4)
        self._safe_addstr(stdscr, top, left, "+" + "-" * (box_width - 2) + "+", curses.A_BOLD)
        self._safe_addstr(stdscr, top + 1, left, "| " + _pad_display(field.label, inner_width) + " |", curses.A_BOLD)
        for row, (_, label) in enumerate(choices[: box_height - 4], start=top + 2):
            choice_index = row - top - 2
            attr = curses.A_REVERSE if choice_index == selected else curses.A_NORMAL
            self._safe_addstr(stdscr, row, left, "| " + _pad_display(label, inner_width) + " |", attr)
        self._safe_addstr(stdscr, top + box_height - 1, left, "+" + "-" * (box_width - 2) + "+", curses.A_BOLD)
        self._refresh(stdscr)

    def _input_text(self, stdscr, field: FormField) -> Optional[str]:
        if field.key == "model":
            return self._input_model_path(stdscr, field)
        if _is_path_completion_field(field):
            return self._input_text_with_completion_for_value(
                stdscr,
                field,
                str(getattr(self.config, field.key)),
                directories_only=_is_directory_completion_field(field),
            )
        return self._input_text_for_value(stdscr, field, str(getattr(self.config, field.key)))

    def _input_model_path(self, stdscr, field: FormField) -> Optional[str]:
        downloaded_models = list_downloaded_model_dirs()
        current = _expand_user_path(str(getattr(self.config, field.key)).strip())
        if not downloaded_models:
            self.status = f"未找到已下载模型，将手动输入: {get_modelscope_cache_dir()}"
            return self._input_text_with_completion_for_value(
                stdscr,
                field,
                str(getattr(self.config, field.key)),
                directories_only=False,
            )

        choices = downloaded_models + [("__manual__", "手动输入路径")]
        selected = next(
            (
                index
                for index, (path, _) in enumerate(choices)
                if path != "__manual__" and _expand_user_path(path) == current
            ),
            len(choices) - 1 if current else 0,
        )
        while True:
            self._draw_model_path_popup(stdscr, field, choices, selected)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = _previous_index(selected, len(choices))
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = _next_index(selected, len(choices))
            elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
                value = choices[selected][0]
                if value == "__manual__":
                    return self._input_text_with_completion_for_value(
                        stdscr,
                        field,
                        str(getattr(self.config, field.key)),
                        directories_only=False,
                    )
                return value
            elif key in (27, ord("q"), ord("Q")):
                self.status = "已取消选择模型路径。"
                return None

    def _draw_model_path_popup(self, stdscr, field: FormField, choices: Sequence[Choice], selected: int):
        self._draw(stdscr, self._visible_fields())
        height, width = stdscr.getmaxyx()
        label_width = max(_display_width(label) for _, label in choices)
        box_width = min(max(58, label_width + 8), width - 4)
        box_height = min(len(choices) + 5, height - 4)
        visible_count = max(1, box_height - 5)
        top_index = 0
        if selected >= visible_count:
            top_index = selected - visible_count + 1
        bottom_index = min(len(choices), top_index + visible_count)
        top = max(1, (height - box_height) // 2)
        left = max(2, (width - box_width) // 2)
        inner_width = max(1, box_width - 4)

        self._safe_addstr(stdscr, top, left, "+" + "-" * (box_width - 2) + "+", curses.A_BOLD)
        self._safe_addstr(stdscr, top + 1, left, "| " + _pad_display("选择已下载模型", inner_width) + " |", curses.A_BOLD)
        for row, index in enumerate(range(top_index, bottom_index), start=top + 2):
            _, label = choices[index]
            attr = curses.A_REVERSE if index == selected else curses.A_NORMAL
            self._safe_addstr(stdscr, row, left, "| " + _pad_display(label, inner_width) + " |", attr)

        selected_path, _ = choices[selected]
        hint = "手动输入路径，支持 Tab 补全。" if selected_path == "__manual__" else selected_path
        self._safe_addstr(stdscr, top + box_height - 2, left, "| " + _pad_display(_clip_display(hint, inner_width), inner_width) + " |", curses.A_DIM)
        self._safe_addstr(stdscr, top + box_height - 1, left, "+" + "-" * (box_width - 2) + "+", curses.A_BOLD)
        self._refresh(stdscr)

    def _input_text_for_value(self, stdscr, field: FormField, current: str) -> Optional[str]:
        return self._input_text_popup(stdscr, field, current)

    def _input_text_with_completion_for_value(
        self,
        stdscr,
        field: FormField,
        current: str,
        directories_only: bool = False,
    ) -> Optional[str]:
        return self._input_text_popup(stdscr, field, current, allow_completion=True, directories_only=directories_only)

    def _input_text_popup(
        self,
        stdscr,
        field: FormField,
        current: str,
        allow_completion: bool = False,
        directories_only: bool = False,
    ) -> Optional[str]:
        prompt = "> "
        value = ""
        cursor = 0
        message = (
            "Tab补全，Enter保存，Esc取消，空输入保持当前值，-清空。"
            if allow_completion
            else "Enter保存，Esc取消，空输入保持当前值，-清空。"
        )

        curses.curs_set(1)
        try:
            while True:
                input_y, input_x, visible_width, view_start = self._draw_text_input_popup(
                    stdscr,
                    field,
                    current,
                    value,
                    cursor,
                    prompt,
                    message,
                )
                visible_cursor = _display_width(value[view_start:cursor])
                stdscr.move(input_y, min(input_x + visible_cursor, input_x + visible_width - 1))
                stdscr.refresh()

                key = stdscr.get_wch()
                if key in ("\n", "\r") or key == curses.KEY_ENTER:
                    if value == "":
                        return None
                    if value == "-":
                        return ""
                    return value.strip()
                if key == "\x1b":
                    self.status = "已取消输入。"
                    return None
                if key in ("\t", curses.KEY_BTAB):
                    if allow_completion:
                        value, cursor, message = self._complete_input_value(value, cursor, directories_only)
                    continue
                if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                    if cursor > 0:
                        value = value[: cursor - 1] + value[cursor:]
                        cursor -= 1
                    continue
                if key == curses.KEY_DC:
                    if cursor < len(value):
                        value = value[:cursor] + value[cursor + 1:]
                    continue
                if key == curses.KEY_LEFT:
                    cursor = max(0, cursor - 1)
                    continue
                if key == curses.KEY_RIGHT:
                    cursor = min(len(value), cursor + 1)
                    continue
                if key == curses.KEY_HOME:
                    cursor = 0
                    continue
                if key == curses.KEY_END:
                    cursor = len(value)
                    continue
                if isinstance(key, str) and key == "\x15":
                    value = ""
                    cursor = 0
                    continue
                if isinstance(key, str) and key.isprintable():
                    value = value[:cursor] + key + value[cursor:]
                    cursor += len(key)
                    message = (
                        "Tab补全，Enter保存，Esc取消，空输入保持当前值，-清空。"
                        if allow_completion
                        else "Enter保存，Esc取消，空输入保持当前值，-清空。"
                    )
        finally:
            curses.curs_set(0)

    def _draw_text_input_popup(
        self,
        stdscr,
        field: FormField,
        current: str,
        value: str,
        cursor: int,
        prompt: str,
        message: str,
    ) -> Tuple[int, int, int, int]:
        height, width = stdscr.getmaxyx()
        box_width = min(max(56, min(84, width - 4)), width - 4)
        box_height = min(8, height - 2)
        top = max(1, (height - box_height) // 2)
        left = max(2, (width - box_width) // 2)
        inner_width = max(1, box_width - 4)

        self._safe_addstr(stdscr, top, left, "+" + "-" * (box_width - 2) + "+", curses.A_BOLD)
        self._draw_text_box_line(stdscr, top + 1, left, box_width, field.label, curses.A_BOLD)
        self._draw_text_box_line(stdscr, top + 2, left, box_width, "")
        self._draw_text_box_line(stdscr, top + 3, left, box_width, f"当前值: {current or '空'}", curses.A_DIM)
        self._draw_text_box_line(stdscr, top + 4, left, box_width, message, curses.A_DIM)

        input_y = top + 5
        input_x = left + 2 + _display_width(prompt)
        visible_width = max(1, inner_width - _display_width(prompt))
        view_start = max(0, cursor - visible_width + 1)
        visible_value = _clip_display(value[view_start:], visible_width)
        input_line = prompt + visible_value
        self._draw_text_box_line(stdscr, input_y, left, box_width, input_line, curses.A_REVERSE)
        self._draw_text_box_line(stdscr, top + 6, left, box_width, "")
        self._safe_addstr(stdscr, top + box_height - 1, left, "+" + "-" * (box_width - 2) + "+", curses.A_BOLD)
        return input_y, input_x, visible_width, view_start

    def _draw_text_box_line(self, stdscr, y: int, left: int, box_width: int, text: str, attr: int = 0):
        inner_width = max(1, box_width - 4)
        self._safe_addstr(stdscr, y, left, "| " + _pad_display(text, inner_width) + " |", attr)

    def _complete_input_value(self, value: str, cursor: int, directories_only: bool = False) -> Tuple[str, int, str]:
        prefix = value[:cursor]
        suffix = value[cursor:]
        matches = complete_path_prefix(prefix, directories_only)
        if not matches:
            target = "本地目录" if directories_only else "本地路径"
            return value, cursor, f"没有匹配的{target}。"

        if len(matches) == 1:
            completed = matches[0]
            new_value = completed + suffix
            return new_value, len(completed), f"已补全: {completed}"

        common = os.path.commonprefix(matches)
        if len(common) > len(prefix):
            new_value = common + suffix
            return new_value, len(common), f"已补全公共前缀，候选 {len(matches)} 个。"

        preview = ", ".join(matches[:5])
        if len(matches) > 5:
            preview += f" 等 {len(matches)} 个"
        return value, cursor, "候选: " + preview

    @staticmethod
    def _safe_addstr(stdscr, y: int, x: int, text: str, attr: int = 0):
        height, width = stdscr.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        max_width = max(0, width - x)
        if max_width <= 0:
            return
        try:
            stdscr.addnstr(y, x, _clip_display(text, max_width), max_width, attr)
        except curses.error:
            pass

    def _refresh(self, stdscr):
        stdscr.refresh()


def _prompt_choice(field: FormField, config: DeployConfig) -> str:
    current = getattr(config, field.key)
    print(f"\n{field.label}:")
    for index, (value, label) in enumerate(field.choices, start=1):
        marker = "*" if value == current else " "
        print(f"  {index}. {label} {marker}")
    raw = _read_line("选择序号，留空保持当前值: ").strip()
    if raw == "":
        return current
    try:
        index = int(raw) - 1
        if 0 <= index < len(field.choices):
            return field.choices[index][0]
    except ValueError:
        pass
    print("输入无效，保持当前值。")
    return current


def _prompt_text(field: FormField, config: DeployConfig) -> str:
    if field.key == "model":
        selected_model = _prompt_downloaded_model_path()
        if selected_model is not None:
            return selected_model

    current = str(getattr(config, field.key))
    prompt = f"{field.label} [{current or '空'}]，输入 - 清空"
    if _is_path_completion_field(field):
        target = "本地目录" if _is_directory_completion_field(field) else "本地路径"
        prompt += f"，Tab补全{target}"
        raw = _read_line_with_path_completion(prompt + ": ", _is_directory_completion_field(field)).strip()
    else:
        raw = _read_line(prompt + ": ").strip()
    if raw == "":
        return current
    if raw == "-":
        return ""
    return raw


def _prompt_downloaded_model_path() -> Optional[str]:
    choices = list_downloaded_model_dirs()
    if not choices:
        return None

    print("\n已下载模型:")
    print(f"缓存目录: {get_modelscope_cache_dir()}")
    for index, (path, label) in enumerate(choices, start=1):
        print(f"  {index}. {label} ({path})")
    raw = _read_line("选择已下载模型编号，留空手动输入路径: ").strip()
    if raw == "":
        return None
    try:
        index = int(raw) - 1
    except ValueError:
        print("输入无效，改为手动输入路径。")
        return None
    if 0 <= index < len(choices):
        return choices[index][0]
    print("编号不存在，改为手动输入路径。")
    return None


def _read_line(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _read_line_with_path_completion(prompt: str, directories_only: bool = False) -> str:
    if not sys.stdin.isatty():
        return _read_line(prompt)

    try:
        import readline
    except ImportError:
        return _read_line(prompt)

    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()

    def completer(text: str, state: int) -> Optional[str]:
        matches = complete_path_prefix(text, directories_only)
        if state < len(matches):
            return matches[state]
        return None

    try:
        readline.set_completer(completer)
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
        return _read_line(prompt)
    finally:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)


def _save_plain_config(saved_configs: List[DeployConfig], config: DeployConfig, edit_index: Optional[int]):
    if edit_index is None:
        saved_configs.append(clone_config(config))
    else:
        saved_configs[edit_index] = clone_config(config)
    save_saved_configs(saved_configs)


def _prompt_deploy_field(config: DeployConfig, field: FormField):
    print(f"\n{field.help}")
    if field.kind == "submenu":
        if field.key == "sampling_params":
            raw = _read_line("是否编辑采样参数? (y/N): ").strip().lower()
            if raw in ("y", "yes"):
                print("\n采样参数")
                for sampling_field in SAMPLING_FIELDS:
                    _prompt_deploy_field(config, sampling_field)
        return
    if field.kind == "choice":
        old_command = config.command
        old_device = config.device
        value = _prompt_choice(field, config)
        setattr(config, field.key, value)
        if field.key == "command" and old_command != value:
            apply_command_defaults(config, old_command, value)
        if field.key == "device" and old_device != value:
            apply_main_device_defaults(config, old_device, value)
    elif field.kind == "bool":
        current = getattr(config, field.key)
        raw = _read_line(f"{field.label} [{'Y' if current else 'N'}] (y/n): ").strip().lower()
        if raw in ("y", "yes", "1", "true", "on"):
            setattr(config, field.key, True)
            if field.key == "enable_moe_hybrid":
                apply_moe_hybrid_defaults(config)
        elif raw in ("n", "no", "0", "false", "off"):
            setattr(config, field.key, False)
    else:
        old_model = config.model
        value = _prompt_text(field, config)
        setattr(config, field.key, value)
        if field.key == "model":
            apply_model_name_default(config, old_model, value)


def run_plain_form(
    config: Optional[DeployConfig] = None,
    saved_configs: Optional[List[DeployConfig]] = None,
    edit_index: Optional[int] = None,
) -> Tuple[str, DeployConfig]:
    config = config or DeployConfig()
    print("fastllm TUI 部署向导")
    print("逐项填写基础配置；留空保持默认值。")
    for field in iter_deploy_fields(config, False):
        _prompt_deploy_field(config, field)

    raw = _read_line("\n是否编辑高级选项? (y/N): ").strip().lower()
    if raw in ("y", "yes"):
        print("\n高级选项")
        for field in iter_deploy_fields(config, True):
            _prompt_deploy_field(config, field)

    errors = validate_config(config)
    if errors:
        print("\n配置有误:")
        for error in errors:
            print(f"- {error}")
        return "quit", config

    print("\n命令预览:")
    print(build_fastllm_command(config))
    raw = _read_line("\n输入 s 保存并启动，w 保存返回，其他键返回: ").strip().lower()
    if raw == "s":
        if saved_configs is not None:
            _save_plain_config(saved_configs, config, edit_index)
        return "start", config
    if raw == "w":
        if saved_configs is not None:
            _save_plain_config(saved_configs, config, edit_index)
            print(f"已保存到 {get_saved_commands_path()}")
        return "back", config
    return "back", config


def run_plain_wizard(config: Optional[DeployConfig] = None) -> Tuple[str, Union[DeployConfig, ModelScopeDownloadConfig]]:
    if config is not None:
        saved_configs = load_saved_configs()
        return run_plain_form(config, saved_configs, None)

    saved_configs = load_saved_configs()
    while True:
        print("fastllm TUI 命令列表")
        print(f"配置文件: {get_saved_commands_path()}")
        if not saved_configs:
            print("还没有保存的命令。直接回车或输入 n 新建。")
        else:
            for index, item in enumerate(saved_configs, start=1):
                print(f"{index}. [{item.command}] {config_title(item)}")
                print(f"   {build_fastllm_command(item)}")

        raw = _read_line("\n输入编号启动，e编号编辑，n新建，m下载模型，q退出: ").strip().lower()
        if raw in ("", "n"):
            if raw == "" and saved_configs:
                continue
            action, edited = run_plain_form(new_deploy_config(saved_configs), saved_configs, None)
            if action in ("start", "quit"):
                return action, edited
        elif raw == "q":
            return "quit", DeployConfig()
        elif raw == "m":
            return run_plain_modelscope_download()
        elif raw.startswith("e"):
            try:
                index = int(raw[1:]) - 1
            except ValueError:
                print("编辑格式应为 e编号，例如 e1。")
                continue
            if not (0 <= index < len(saved_configs)):
                print("编号不存在。")
                continue
            action, edited = run_plain_form(clone_config(saved_configs[index]), saved_configs, index)
            if action in ("start", "quit"):
                return action, edited
        else:
            try:
                index = int(raw) - 1
            except ValueError:
                print("输入无效。")
                continue
            if not (0 <= index < len(saved_configs)):
                print("编号不存在。")
                continue
            selected = clone_config(saved_configs[index])
            errors = validate_config(selected)
            if errors:
                print("配置有误: " + errors[0])
                continue
            return "start", selected


def run_plain_modelscope_download() -> Tuple[str, Union[ModelScopeDownloadConfig, DeployConfig]]:
    config = ModelScopeDownloadConfig()
    print("ModelScope 模型下载")
    for index, (_, label, _) in enumerate(MODELSCOPE_MODEL_GROUPS, start=1):
        print(f"{index}. {label}")
    raw = _read_line("选择模型分类，留空使用热门模型: ").strip()
    group_index = 0
    if raw:
        try:
            group_index = int(raw) - 1
        except ValueError:
            group_index = 0
        if not (0 <= group_index < len(MODELSCOPE_MODEL_GROUPS)):
            group_index = 0
    _, group_label, model_choices = MODELSCOPE_MODEL_GROUPS[group_index]
    print(f"{group_label}:")
    for index, (model_id, label) in enumerate(model_choices, start=1):
        print(f"{index}. {label} ({model_id})")
    raw = _read_line("选择模型编号，留空使用第一个: ").strip()
    model_index = 0
    if raw:
        try:
            model_index = int(raw) - 1
        except ValueError:
            model_index = 0
        if not (0 <= model_index < len(model_choices)):
            model_index = 0
    config.model_id = model_choices[model_index][0]
    if config.model_id == "custom":
        config.model_id_custom = _read_line("自定义模型ID: ").strip()
    model_id = _resolve_modelscope_model_id(config)
    default_target = default_modelscope_target_dir(model_id)
    target_dir = _read_line_with_path_completion(f"下载目录 [{default_target}]: ").strip()
    config.target_dir = target_dir or default_target
    workers = _read_line(f"下载并发 [{config.max_workers}]: ").strip()
    if workers:
        config.max_workers = workers
    custom_args = _read_line("自定义参数 key=value; key=value [空]: ").strip()
    if custom_args:
        config.custom_args = custom_args

    errors = validate_modelscope_download(config)
    if errors:
        print("配置有误: " + errors[0])
        return "back", DeployConfig()
    raw = _read_line(f"下载 {model_id} 到 {config.target_dir} ? (y/N): ").strip().lower()
    if raw in ("y", "yes"):
        return "modelscope_download", config
    return "back", DeployConfig()


def _should_use_plain(plain: bool) -> bool:
    if plain:
        return True
    if curses is None:
        return True
    if os.name == "nt":
        return True
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return True
    return False


def _execute(argv: Sequence[str], env_overrides: Optional[dict] = None) -> int:
    env = None
    if env_overrides:
        env = os.environ.copy()
        env.update(env_overrides)
    try:
        return subprocess.call(list(argv), env=env)
    except KeyboardInterrupt:
        return 130


def FastllmTUI(plain: bool = False) -> int:
    if _should_use_plain(plain):
        action, config = run_plain_wizard()
    else:
        try:
            action, config = FastllmCursesTUI().run()
        except curses.error:
            action, config = run_plain_wizard()

    if action == "modelscope_download":
        assert isinstance(config, ModelScopeDownloadConfig)
        print(f"Downloading {config.model_id} to {_expand_user_path(config.target_dir)}")
        try:
            local_path = download_modelscope_model(config)
            print(f"Model downloaded to: {local_path}")
            return 0
        except Exception as exc:
            print(f"ModelScope download failed: {exc}", file=sys.stderr)
            return 1
    if action == "start":
        assert isinstance(config, DeployConfig)
        argv = build_fastllm_argv(config)
        print("Running: " + build_fastllm_command(config))
        return _execute(argv, build_fastllm_env(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(FastllmTUI())
