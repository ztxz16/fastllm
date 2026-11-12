"""Launch contracts for model services. New modalities register a backend here.

The launcher owns process/GPU isolation; a backend owns its command, health
probe, capabilities and public API contract. No inference implementation lives
in the launcher or in the studio.
"""
import os
import math
import re
import shutil
import subprocess
from pathlib import Path


def media_body_limit(path):
    """Match native video uploads while keeping other media requests at 32 MiB."""
    video = re.fullmatch(r"(?:/api/media/[^/]+/)?videos", path)
    return (256 if video else 32) * 1024 * 1024


def visible_gpu_environment(value, environment=None):
    """Resolve launcher-visible indices against an inherited CUDA mask.

    Child device arguments always use logical indices after this mask. Never
    widen a parent/container mask, even when an explicit UUID is supplied.
    """
    environment = os.environ if environment is None else environment
    if not value.strip():
        return {}
    identifiers = [part.strip() for part in value.split(",")]
    identifiers = [str(int(part)) if part.isascii() and part.isdigit() else part for part in identifiers]
    if any(not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+", part)
           for part in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("可用 GPU 必须是不重复的卡号或 UUID，以逗号分隔。")
    parent = environment.get("CUDA_VISIBLE_DEVICES")
    gpu_uuids = {}
    if any(part.startswith("GPU-") for part in identifiers):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=4, check=False)
            for line in result.stdout.splitlines():
                index, uuid = line.split(",", 1)
                gpu_uuids[index.strip()] = uuid.strip()
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    def canonical(identifier):
        identifier = str(int(identifier)) if identifier.isascii() and identifier.isdigit() else identifier
        return gpu_uuids.get(identifier, identifier)
    if parent is not None:
        allowed = []
        for part in parent.split(","):
            part = part.strip()
            # CUDA stops enumeration at an invalid entry; dropping it would
            # expose devices outside the inherited visibility boundary.
            if (not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+", part)
                    or canonical(part) in {canonical(item) for item in allowed}):
                break
            allowed.append(part)
        resolved = []
        for identifier in identifiers:
            if identifier.isdigit() and int(identifier) < len(allowed):
                resolved.append(allowed[int(identifier)])
            elif not identifier.isdigit() and identifier in allowed:
                resolved.append(identifier)
            elif not identifier.isdigit() and len([item for item in allowed if canonical(item) == identifier]) == 1:
                resolved.append(next(item for item in allowed if canonical(item) == identifier))
            else:
                raise ValueError("所选 GPU 超出了 Launcher 的 CUDA_VISIBLE_DEVICES 范围。")
        identifiers = resolved
    if len({canonical(item) for item in identifiers}) != len(identifiers):
        raise ValueError("不能重复选择同一张 GPU。")
    return {"CUDA_VISIBLE_DEVICES": ",".join(identifiers)}


class ChatBackend:
    id = "llm"
    kind = "llm"
    label = "LLM · FastLLM"
    capabilities = ("chat",)
    health_path = "/v1/models"

    def model_name(self, config):
        return config.model_name.strip() or Path(config.model.rstrip("/\\")).name

    def api_contract(self, state):
        return {"kind": self.kind, "capabilities": list(self.capabilities),
                "endpoint": state["endpoint"].rstrip("/") + "/v1/chat/completions"}


def _validate_memory_limits(config, add, label):
    if config.video_gpu_memory_limit.strip():
        try:
            limit = float(config.video_gpu_memory_limit)
            if not math.isfinite(limit) or not (limit == 0 or (1 + 1 / 2**30 <= limit <= 1048576)):
                raise ValueError()
        except ValueError:
            add(f"{label}显存预算须为大于 1 的 GiB 数值，或填 0 使用默认策略。", "video_gpu_memory_limit")
    if config.video_host_memory_limit.strip():
        try:
            limit = float(config.video_host_memory_limit)
            if not math.isfinite(limit) or not (limit == 0 or 2 <= limit <= 1048576):
                raise ValueError()
        except ValueError:
            add(f"{label}内存预算须为至少 2 的 GiB 数值，或填 0 使用默认策略。", "video_host_memory_limit")


class H3Backend:
    id = "minimax_h3"
    kind = "video"
    label = "视频 · MiniMax H3 / Pika H3"
    capabilities = ("video.generate",)
    health_path = "/health"

    @staticmethod
    def media_path(method, path):
        if method == "GET" and path == "health":
            return "/health"
        if ((method in {"GET", "POST"} and path == "videos") or
                (method == "GET" and re.fullmatch(r"videos/[A-Za-z0-9_-]+(?:/content)?", path)) or
                (method == "POST" and re.fullmatch(r"videos/[A-Za-z0-9_-]+/cancel", path))):
            return "/v1/" + path
        return None

    def model_name(self, config):
        root = Path(config.model).expanduser()
        return "pika-h3-flex" if any((root / task / name).exists()
            for task in ("fl2va", "ref2va")
            for name in ("manifest.json", "adapter.safetensors")) else "minimax-h3"

    @staticmethod
    def executable(config):
        explicit = config.server_binary.strip() or os.environ.get("FTLLM_H3_SERVER", "")
        if explicit:
            return os.path.expanduser(explicit)
        bundled = Path(__file__).resolve().parent / "minimax_h3_server"
        if bundled.is_file():
            return str(bundled)
        found = shutil.which("minimax_h3_server")
        if found:
            return found
        repository = Path(__file__).resolve().parents[2]
        for build in ("build", "build-cuda"):
            candidate = repository / build / "src/models/minimax_h3/minimax_h3_server"
            if candidate.is_file():
                return str(candidate)
        return "minimax_h3_server"

    def argv(self, config):
        # Explicit selection uses all selected GPUs, remapped to 0..N-1.
        devices = ",".join(str(i) for i in range(len(config.visible_gpus.split(",")))) if config.visible_gpus.strip() else "0"
        output = config.output_dir.strip() or str(Path.home() / ".local/share/fastllm/videos" / config.port)
        command = [self.executable(config), "--model", os.path.expanduser(config.model.strip()),
                   "--host", config.host.strip(), "--port", config.port.strip(),
                   "--devices", devices, "--output", os.path.expanduser(output)]
        demo = Path(__file__).resolve().parent / "h3_assets/demo.html"
        if demo.is_file():
            command.extend(("--demo", str(demo)))
        for flag, value in (("--base-model", config.base_model), ("--api-key", config.api_key)):
            if value.strip():
                command.extend((flag, os.path.expanduser(value.strip()) if flag == "--base-model" else value.strip()))
        if config.video_gpu_memory_limit.strip():
            command.extend(("--gpu-memory-limit", config.video_gpu_memory_limit.strip()))
        if config.video_host_memory_limit.strip():
            command.extend(("--host-memory-limit", config.video_host_memory_limit.strip()))
        return command

    def validate(self, config):
        errors = []
        def add(message, *fields):
            errors.append({"message": message, "fields": list(fields)})
        if config.command != "server":
            add("视频模型需要使用 API Server 启动方式。", "command")
        if not Path(config.model).expanduser().is_dir() or not config.model.strip():
            add("请选择已存在的 MiniMax H3 或 Pika H3 模型目录。", "model")
        if config.base_model.strip() and not Path(config.base_model).expanduser().is_dir():
            add("MiniMax H3 基座目录不存在。", "base_model")
        executable = self.executable(config)
        if not shutil.which(executable):
            add("找不到 minimax_h3_server；请编译 H3 服务并填写可执行文件路径，或设置 FTLLM_H3_SERVER。", "server_binary")
        try:
            if not 1 <= int(config.port) <= 65535:
                raise ValueError()
        except ValueError:
            add("端口必须在 1-65535 之间。", "port")
        if not config.host.strip():
            add("监听地址不能为空。", "host")
        _validate_memory_limits(config, add, "视频")
        return errors

    def api_contract(self, state):
        base = state["endpoint"].rstrip("/") + "/v1/videos"
        return {"kind": self.kind, "capabilities": list(self.capabilities),
                "endpoint": base, "create": "POST " + base,
                "status": "GET " + base + "/{id}",
                "content": "GET " + base + "/{id}/content",
                "cancel": "POST " + base + "/{id}/cancel",
                "modes": ["t2va", "first_frame", "last_frame", "first_last", "ref2va"],
                "conditioning_health": "GET " + state["endpoint"].rstrip("/") + "/health",
                "keyframes": {"format": "multipart/form-data", "fields": ["first_frame", "last_frame"],
                              "first_frame_alias": "input_reference"},
                "references": {"format": "multipart/form-data", "field": "reference", "ordered": True,
                               "types": ["image/*", "video/*", "audio/*"],
                               "max_images": 9, "max_videos": 3, "max_audio": 3, "max_count": 12,
                               "clip_seconds": [2, 15], "total_video_seconds": 15, "total_audio_seconds": 15,
                               "requires_visual": True, "exclusive_with_keyframes": True,
                               "requires": "transformer_ref", "availability_field": "reference_model_available",
                               "prompt_labels": ["Picture 1", "Video 1", "Audio 1"]},
                "uploads": {"max_file_bytes": 128 * 1024 * 1024, "max_request_bytes": media_body_limit("videos")},
                "request": {"model": state["modelName"], "prompt": "A cat walking through a sunny garden",
                            "size": "640x384", "seconds": 5},
                "defaults": {"size": "640x384", "seconds": 5,
                             "num_inference_steps": 6 if state["modelName"] == "pika-h3-flex" else 50}}


class H3ImageBackend:
    id = "minimax_h3_image"
    kind = "image"
    label = "图像 · MiniMax H3 Image"
    capabilities = ("image.generate", "image.edit")
    health_path = "/health"

    @staticmethod
    def media_path(method, path):
        if ((method in {"GET", "POST"} and path == "images/jobs") or
                (method == "POST" and path in {"images/generations", "images/edits"}) or
                (method == "GET" and re.fullmatch(r"images/jobs/image_[0-9]+_[0-9]+(?:/content)?", path)) or
                (method == "POST" and re.fullmatch(r"images/jobs/image_[0-9]+_[0-9]+/cancel", path))):
            return "/v1/" + path
        return None

    def model_name(self, config):
        return "minimax-h3-image"

    @staticmethod
    def executable(config):
        explicit = config.server_binary.strip() or os.environ.get("FTLLM_H3_IMAGE_SERVER", "")
        if explicit:
            return os.path.expanduser(explicit)
        name = "minimax_h3_image_server"
        bundled = Path(__file__).resolve().parent / name
        if bundled.is_file():
            return str(bundled)
        found = shutil.which(name)
        if found:
            return found
        repository = Path(__file__).resolve().parents[2]
        for build in ("build", "build-cuda"):
            candidate = repository / build / "src/models/minimax_h3" / name
            if candidate.is_file():
                return str(candidate)
        return name

    def argv(self, config):
        output = config.output_dir.strip() or str(Path.home() / ".local/share/fastllm/images" / config.port)
        command = [self.executable(config), "--model", os.path.expanduser(config.model.strip()),
                   "--host", config.host.strip(), "--port", config.port.strip(), "--device", "0",
                   "--output", os.path.expanduser(output)]
        count = len([gpu for gpu in config.visible_gpus.split(",") if gpu.strip()])
        if count > 1:
            command.extend(("--devices", ",".join(str(i) for i in range(count))))
        package = Path(__file__).resolve().parent
        for demo in (package / "h3_image_assets/demo.html", package.parent / "minimax_h3_image/demo.html"):
            if demo.is_file():
                command.extend(("--demo", str(demo)))
                break
        if config.api_key.strip():
            command.extend(("--api-key", config.api_key.strip()))
        for flag, value in (("--gpu-memory-limit", config.video_gpu_memory_limit),
                            ("--host-memory-limit", config.video_host_memory_limit)):
            if value.strip():
                command.extend((flag, value.strip()))
        return command

    def validate(self, config):
        errors = []
        def add(message, *fields):
            errors.append({"message": message, "fields": list(fields)})
        if config.command != "server":
            add("图像模型需要使用 API Server 启动方式。", "command")
        model = Path(config.model.strip()).expanduser()
        if not config.model.strip() or not (model.is_dir() or (model.is_file() and model.suffix == ".safetensors")):
            add("请选择完整 MiniMax H3 Image 模型目录或 FP8 safetensors 文件；权重类型会自动识别。", "model")
        if not shutil.which(self.executable(config)):
            add("找不到 minimax_h3_image_server；请编译图像服务或设置 FTLLM_H3_IMAGE_SERVER。", "server_binary")
        count = len([gpu for gpu in config.visible_gpus.split(",") if gpu.strip()])
        if count not in {0, 1, 2, 4, 8}:
            add("MiniMax H3 Image 支持 1、2、4 或 8 张 GPU。", "visible_gpus")
        try:
            if not 1 <= int(config.port) <= 65535:
                raise ValueError()
        except ValueError:
            add("端口必须在 1-65535 之间。", "port")
        if not config.host.strip():
            add("监听地址不能为空。", "host")
        _validate_memory_limits(config, add, "图像")
        return errors

    def api_contract(self, state):
        base = state["endpoint"].rstrip("/") + "/v1/images/jobs"
        return {"kind": self.kind, "capabilities": list(self.capabilities), "endpoint": base,
                "create": "POST " + base, "status": "GET " + base + "/{id}",
                "content": "GET " + base + "/{id}/content", "cancel": "POST " + base + "/{id}/cancel",
                "request": {"model": state["modelName"], "prompt": "A red panda beside a mountain stream, autumn forest",
                            "size": "2048x2048", "num_inference_steps": 6},
                "edit": "POST " + state["endpoint"].rstrip("/") + "/v1/images/edits",
                "references": {"format": "multipart/form-data", "field": "image[]", "max_count": 16,
                               "types": ["image/png", "image/jpeg", "image/webp"], "prompt_labels": "<Picture 1>, <Picture 2>, ..."},
                "defaults": {"size": "2048x2048", "num_inference_steps": 6}}


BACKENDS = {backend.id: backend for backend in (ChatBackend(), H3Backend(), H3ImageBackend())}


def get_backend(config):
    try:
        return BACKENDS[config.backend]
    except KeyError:
        raise ValueError("不支持的模型后端：" + config.backend) from None


def backend_catalog():
    return [{"id": item.id, "kind": item.kind, "label": item.label,
             "capabilities": list(item.capabilities)} for item in BACKENDS.values()]
