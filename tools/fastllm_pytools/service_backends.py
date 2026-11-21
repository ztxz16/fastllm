"""Launch contracts for model services. New modalities register a backend here.

The launcher owns process/GPU isolation; a backend owns its command, health
probe, capabilities and public API contract. No inference implementation lives
in the launcher or in the studio.
"""
import os
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .model_service import has_pika_adapter, identify_model


PACKAGE = Path(__file__).resolve().parent


def _build_outputs(relative, builds):
    # Source checkouts use tools/fastllm_pytools; a CMake staging tree uses tools/ftllm.
    if PACKAGE.parent.name == "tools":
        root = PACKAGE.parent.parent
        if PACKAGE.name == "ftllm":
            yield root / relative
        for build in builds:
            yield root / build / relative


def _native_executable(build_path, builds):
    name = build_path.name
    bundled = PACKAGE / name
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    found = shutil.which(name)
    if found:
        return found
    for candidate in _build_outputs(build_path, builds):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return name


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


def launcher_host(config, backend):
    if config.command == "webui" or (backend and backend.kind == "interactive"):
        return "127.0.0.1"
    return config.host.strip() or "127.0.0.1"


def logical_devices(config):
    """Native device IDs after applying the visibility mask; default to the first GPU."""
    return [str(index) for index in range(config.visible_gpus.count(",") + 1)]


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


def _h3_runtime_options(config):
    options = []
    for flag, value in (("--gpu-memory-limit", config.video_gpu_memory_limit),
                        ("--host-memory-limit", config.video_host_memory_limit)):
        if value.strip():
            options.extend((flag, value.strip()))
    if config.video_parallel_mode != "auto":
        options.extend(("--parallel-mode", config.video_parallel_mode))
    return options


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
        manifest = root / "h3_quantization.json" if root.is_dir() else root
        if manifest.name == "h3_quantization.json" and manifest.is_file():
            try:
                with manifest.open(encoding="utf-8") as stream:
                    source = json.loads(stream.read(65537))
                if isinstance(source, dict) and source.get("pika"):
                    return "pika-h3-flex"
            except (OSError, ValueError):
                pass  # The native loader reports malformed model manifests.
        return "pika-h3-flex" if has_pika_adapter(root) else "minimax-h3"

    @staticmethod
    def executable():
        return _native_executable(Path("src/models/minimax_h3/minimax_h3_server"),
                                  ("build", "build-cuda"))

    def argv(self, config):
        # Explicit selection uses all selected GPUs, remapped to 0..N-1.
        devices = ",".join(logical_devices(config))
        output = config.output_dir.strip() or str(Path.home() / ".local/share/fastllm/videos" / config.port)
        command = [self.executable(), "--model", os.path.expanduser(config.model.strip()),
                   "--host", config.host.strip(), "--port", config.port.strip(),
                   "--devices", devices, "--output", os.path.expanduser(output)]
        demo = Path(__file__).resolve().parent / "h3_assets/demo.html"
        if demo.is_file():
            command.extend(("--demo", str(demo)))
        for flag, value in (("--base-model", config.base_model), ("--api-key", config.api_key)):
            if value.strip():
                command.extend((flag, os.path.expanduser(value.strip()) if flag == "--base-model" else value.strip()))
        command.extend(_h3_runtime_options(config))
        return command

    def validate(self, config):
        errors = []
        def add(message, *fields):
            errors.append({"message": message, "fields": list(fields)})
        if config.command != "server":
            add("视频模型需要使用 API Server 启动方式。", "command")
        model = Path(config.model).expanduser()
        if not config.model.strip() or not (model.is_dir() or
                (model.is_file() and model.name == "h3_quantization.json")):
            add("请选择 MiniMax H3、Pika H3 模型目录或 h3_quantization.json。", "model")
        if config.video_parallel_mode not in {"auto", "tp", "sp"}:
            add("视频并行方式须为 auto、tp 或 sp。", "video_parallel_mode")
        if config.base_model.strip() and not Path(config.base_model).expanduser().is_dir():
            add("MiniMax H3 基座目录不存在。", "base_model")
        executable = self.executable()
        if not shutil.which(executable):
            add("当前安装缺少 MiniMax H3 视频服务（minimax_h3_server）。请安装含该服务的版本或构建该目标。", "model")
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
    def executable():
        return _native_executable(Path("src/models/minimax_h3/minimax_h3_image_server"),
                                  ("build", "build-cuda"))

    def argv(self, config):
        output = config.output_dir.strip() or str(Path.home() / ".local/share/fastllm/images" / config.port)
        command = [self.executable(), "--model", os.path.expanduser(config.model.strip()),
                   "--host", config.host.strip(), "--port", config.port.strip(), "--device", "0",
                   "--output", os.path.expanduser(output)]
        devices = logical_devices(config)
        if len(devices) > 1:
            command.extend(("--devices", ",".join(devices)))
        package = Path(__file__).resolve().parent
        for demo in (package / "h3_image_assets/demo.html", package.parent / "minimax_h3_image/demo.html"):
            if demo.is_file():
                command.extend(("--demo", str(demo)))
                break
        if config.api_key.strip():
            command.extend(("--api-key", config.api_key.strip()))
        command.extend(_h3_runtime_options(config))
        return command

    def validate(self, config):
        errors = []
        def add(message, *fields):
            errors.append({"message": message, "fields": list(fields)})
        if config.command != "server":
            add("图像模型需要使用 API Server 启动方式。", "command")
        model = Path(config.model.strip()).expanduser()
        if not config.model.strip() or not (
                model.is_dir() or (model.is_file() and model.suffix in {".safetensors", ".gguf"})):
            add("请选择完整 MiniMax H3 Image 模型目录或 FP8 / INT8 / NVFP4 safetensors 或 GGUF 文件；权重类型会自动识别。", "model")
        if not shutil.which(self.executable()):
            add("当前安装缺少 MiniMax H3 图像服务（minimax_h3_image_server）。请安装含该服务的版本或构建该目标。", "model")
        if len(logical_devices(config)) not in {1, 2, 4, 8}:
            add("MiniMax H3 Image 支持 1、2、4 或 8 张 GPU。", "visible_gpus")
        if config.video_parallel_mode not in {"auto", "tp", "sp"}:
            add("图像并行方式须为 auto、tp 或 sp。", "video_parallel_mode")
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


def _native_library(name, builds):
    candidates = [PACKAGE / name, *_build_outputs(name, (*builds, "build", "build-cuda"))]
    candidates.extend(Path(root) / name for root in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                      if root and os.path.isabs(root))
    return next((path for path in candidates if path.is_file()), PACKAGE / name)


def media_assets(name):
    bundled = PACKAGE / (name + "_assets")
    return bundled if bundled.is_dir() else PACKAGE.parent / name


class InteractiveBackend:
    accepts_api_key = False
    kind = "interactive"
    capabilities = ("interactive.demo",)

    def model_name(self, config):
        return self.name

    def output(self, config):
        return str(Path(config.output_dir.strip()).expanduser() if config.output_dir.strip()
                   else Path.home() / ".local/share/fastllm" / self.id / config.port.strip())

    def validate(self, config):
        errors = []
        def add(message, field):
            errors.append({"message": message, "fields": [field]})
        if config.command != "server":
            add("交互模型需要以服务方式启动。", "command")
        if not config.model.strip() or not Path(config.model.strip()).expanduser().is_dir():
            add("请选择模型权重目录。", "model")
        if not re.fullmatch(r"[0-9]{1,5}", config.port.strip()) or not 1 <= int(config.port) <= 65535:
            add("端口必须在 1-65535 之间。", "port")
        if config.api_key.strip() and not self.accepts_api_key:
            add("交互服务通过启动器鉴权，请清空服务 API Key。", "api_key")
        self.validate_runtime(config, add)
        return errors

    def api_contract(self, state):
        base = state["endpoint"].rstrip("/")
        return {"kind": self.kind, "capabilities": list(self.capabilities),
                "baseUrl": base, "endpoint": base + "/", "routes": self.routes}


class NomasBackend(InteractiveBackend):
    id = "hy_nomas"
    name = "HY-Nomas"
    label = "场景交互 · HY-Nomas"
    health_path = "/health"
    routes = ["POST /v1/worlds", "PATCH /v1/worlds/{id}", "POST /v1/worlds/{id}/actions",
              "GET /v1/videos/{id}/events", "DELETE /v1/worlds/{id}"]

    @staticmethod
    def library():
        return _native_library("libfastllm_nomas.so", ("build-hy-nomas", "build-nomas"))

    def encoder(self, config):
        if config.nomas_text_encoder.strip():
            return Path(config.nomas_text_encoder.strip()).expanduser()
        root = Path(config.model.strip()).expanduser()
        if (root / "text_encoder/tokenizer.json").is_file():
            return root / "text_encoder"
        return root.parent / "Qwen2.5-VL-7B-Instruct"

    def validate_runtime(self, config, add):
        if len(logical_devices(config)) not in (1, 2, 4, 8):
            add("HY-Nomas 支持 1、2、4 或 8 张 GPU。", "visible_gpus")
        if not self.library().is_file():
            add("当前安装缺少 HY-Nomas 推理库。请安装含 HY-Nomas 的版本或构建 fastllm_nomas 目标。", "model")
        if not (self.encoder(config) / "tokenizer.json").is_file():
            add("请选择包含 tokenizer.json 的 Qwen2.5-VL-7B-Instruct 目录。", "nomas_text_encoder")

    def argv(self, config):
        devices = logical_devices(config)
        return [sys.executable, str(PACKAGE / "nomas_server.py"), "--library", str(self.library()),
                "--model", os.path.expanduser(config.model.strip()), "--text-encoder", str(self.encoder(config)),
                "--output", self.output(config), "--precision", "bf16", "--host", config.host,
                "--port", config.port.strip(), "--devices" if len(devices) > 1 else "--device", ",".join(devices)]


class DuoBackend(InteractiveBackend):
    id = "duo"
    name = "Duo"
    label = "音视频对话 · Duo"
    health_path = "/resolutions"
    routes = ["GET /resolutions", "POST /sessions", "POST /sessions/custom", "POST /sessions/{id}/turns",
              "GET /sessions/{id}/events", "GET /sessions/{id}/turn-{n}/avc", "DELETE /sessions/{id}"]

    @staticmethod
    def executable():
        return _native_executable(Path("src/models/duo/live/duo_live"),
                                  ("build-duo", "build", "build-cuda"))

    def assets(self, config):
        return Path(config.duo_assets.strip()).expanduser() if config.duo_assets.strip() else media_assets("duo")

    def validate_runtime(self, config, add):
        model = Path(config.model.strip()).expanduser()
        if model.is_dir() and not all((model / name).is_dir() for name in ("realization", "intent", "intent_adapter", "codec", "speech")):
            add("Duo 需要完整的 distill-quant 权重及相应的 causal-base 组件。", "model")
        if not shutil.which(self.executable()):
            add("当前安装缺少 Duo 交互服务。请安装含 Duo 的版本或构建 duo_live 目标。", "model")
        if len(logical_devices(config)) not in (1, 4):
            add("Duo 支持单卡或四卡，请选择 1 或 4 张 GPU。", "visible_gpus")
        required = ("live.html", "avc-player.js", "presets/default.txt", "presets/default-256x192.bf16",
                    "presets/default-512x384.bf16")
        assets = self.assets(config)
        if not all((assets / entry).is_file() for entry in required):
            add("Duo 资源目录需要包含网页、播放器及两个分辨率的角色预设。", "duo_assets")
        if config.duo_intent_cache.strip() and not Path(config.duo_intent_cache.strip()).expanduser().is_dir():
            add("Intent NF4 缓存目录不存在。", "duo_intent_cache")

    def argv(self, config):
        args = [self.executable(), os.path.expanduser(config.model.strip()), str(self.assets(config)),
                self.output(config), config.port.strip(), "four" if len(logical_devices(config)) == 4 else "single"]
        if config.duo_intent_cache.strip():
            args.extend(["--intent-cache", os.path.expanduser(config.duo_intent_cache.strip())])
        return args


class LingBotBackend(InteractiveBackend):
    accepts_api_key = True
    id = "lingbot_world"
    name = "lingbot-world-2.0-1.3b"
    label = "LingBot World 2.0"
    health_path = "/health"
    routes = ["POST /v1/worlds", "POST /v1/worlds/{id}/actions", "GET /v1/jobs/{id}", "DELETE /v1/worlds/{id}"]

    def library(self):
        return _native_library("libfastllm_lingbot.so", ("build-lingbot",))

    def validate_runtime(self, config, add):
        if len(logical_devices(config)) != 1:
            add("LingBot World 目前支持单卡。", "visible_gpus")
        if not self.library().is_file():
            add("当前安装缺少 libfastllm_lingbot.so，请构建 fastllm_lingbot。", "model")
        root = Path(config.model).expanduser()
        for asset in ("native/t5.safetensors", "native/vae.safetensors", "google/umt5-xxl/tokenizer.json"):
            if not (root / asset).is_file():
                add(f"LingBot World 缺少 {asset}，请先准备本地模型组件。", "model")

    def argv(self, config):
        argv = [sys.executable, str(media_assets("lingbot_world") / "server.py"), "--model", config.model,
                "--library", str(self.library()), "--device", "0", "--host", config.host,
                "--port", config.port, "--output", self.output(config)]
        if config.api_key.strip():
            argv.extend(["--api-key", config.api_key.strip()])
        return argv


class QwenImageBackend:
    id = "qwen_image"
    kind = "image"
    label = "Qwen-Image-2.1"
    health_path = "/health"
    capabilities = ("image.generate", "image.edit", "interactive.demo")

    demo_routes = {
        "GET": r"health|openapi\.json|v1/images/jobs/image_[0-9]+_[0-9]+|v1/images/files/image_[0-9]+_[0-9]+\.png",
        "POST": r"v1/images/jobs|v1/images/jobs/image_[0-9]+_[0-9]+/cancel|v1/images/(?:generations|edits)",
    }

    def model_name(self, config):
        return "qwen-image-2.1"

    @staticmethod
    def executable():
        return _native_executable(Path("src/models/qwen_image/qwen_image_server"), ("build", "build-cuda"))

    def validate(self, config):
        errors = []
        if not shutil.which(self.executable()):
            errors.append({"message": "当前安装缺少 qwen_image_server。", "fields": ["model"]})
        if len(logical_devices(config)) not in (1, 2, 4, 8):
            errors.append({"message": "Qwen-Image 支持 1、2、4 或 8 张 GPU。", "fields": ["visible_gpus"]})
        if not config.port.isascii() or not config.port.isdigit() or not 1 <= int(config.port) <= 65535:
            errors.append({"message": "端口必须在 1-65535 之间。", "fields": ["port"]})
        if not config.host.strip():
            errors.append({"message": "监听地址不能为空。", "fields": ["host"]})
        return errors

    def argv(self, config):
        devices = ",".join(logical_devices(config))
        output = config.output_dir or str(Path.home() / ".local/share/fastllm/images" / config.port)
        cmd = [self.executable(), "--model", config.model, "--host", config.host, "--port", config.port,
               "--devices", devices, "--output", os.path.expanduser(output), "--demo", str(media_assets("qwen_image") / "demo.html")]
        if config.api_key:
            cmd.extend(["--api-key", config.api_key])
        return cmd

    def api_contract(self, state):
        base = state["endpoint"].rstrip("/") + "/v1/images"
        return {"kind": self.kind, "capabilities": list(self.capabilities), "endpoint": base + "/jobs",
                "create": "POST " + base + "/jobs", "status": "GET " + base + "/jobs/{id}",
                "content": "GET " + base + "/files/{id}.png", "cancel": "POST " + base + "/jobs/{id}/cancel"}

    def media_path(self, method, path):
        route = "v1/" + path
        pattern = self.demo_routes.get(method)
        return "/" + route if pattern and re.fullmatch(pattern, route) else None


BACKENDS = {backend.id: backend for backend in (ChatBackend(), H3Backend(), H3ImageBackend(), QwenImageBackend(),
                                                 NomasBackend(), LingBotBackend(), DuoBackend())}


def get_backend(config):
    model = identify_model(config.model)
    if model.error:
        raise ValueError(model.error)
    return BACKENDS[model.backend]


def backend_catalog():
    return [{"id": item.id, "kind": item.kind, "label": item.label,
             "capabilities": list(item.capabilities)} for item in BACKENDS.values()]
