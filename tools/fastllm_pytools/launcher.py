import argparse
import hashlib
import hmac
import importlib.util
import ipaddress
import json
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .modelscope_download import PROGRESS_PREFIX as MODELSCOPE_PROGRESS_PREFIX
from .launcher_mtp import detect_mtp_support
from .startup_progress import PROGRESS_PREFIX
from .tui import (
    DEFAULT_MODELSCOPE_MODEL_ID,
    DeployConfig,
    MODELSCOPE_MODEL_GROUPS,
    build_fastllm_env,
    build_fastllm_argv,
    complete_path_prefix,
    config_from_dict,
    default_modelscope_target_dir,
    effective_model_name,
    get_saved_commands_path,
    load_saved_configs,
    new_deploy_config,
    save_saved_configs,
    validate_config,
)


ASSET_DIRECTORY = Path(__file__).with_name("launcher_assets")
LOG_LIMIT = 2000
LOG_MESSAGE_LIMIT = 16 * 1024
REQUEST_BODY_LIMIT = 1024 * 1024
SENSITIVE_ENV_PATTERN = re.compile(
    r"(?:^|_)(?:PASSWORD|PASSWD|TOKEN|SECRET|API_KEY|PRIVATE_KEY|ACCESS_KEY)(?:_|$)",
    re.IGNORECASE,
)
PROGRESS_LABELS = {
    "initializing": "Initializing model",
    "tokenizer": "Loading tokenizer",
    "weights_prepare": "Preparing model weights",
    "weights_load": "Loading model weights",
    "weights_finalize": "Finalizing model weights",
    "warmup": "Warming up model",
    "server_starting": "Starting local API",
    "ready": "Local API is ready",
}
MODELSCOPE_INSTALL_ERROR = (
    "ModelScope is not installed. Run: "
    "python -m pip install 'modelscope>=1.34.0,<2'."
)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
INFERENCE_SPEED_PATTERN = re.compile(
    r"^\[(Prompt|Decode)\]\s+.+?\bSpeed:\s*"
    r"((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+tokens\s*/\s*s\.?\s*$"
)
CONTEXT_WINDOW_PATTERN = re.compile(
    r"\bModel context window:\s*(\d{1,10})\s+tokens per session "
    r"\(model=(?:\d+|None), shared KV cache=(?:\d+|None), configured limit=(?:\d+|None)\)\s*$"
)
GIB = 1024 ** 3
MODEL_CONFIG_SIZE_LIMIT = 8 * 1024 * 1024
MODEL_WEIGHT_ENTRY_LIMIT = 4096
FOLDER_BROWSER_ENTRY_LIMIT = 300
# Keep headroom for allocator overhead, KV cache, and other processes.
GPU_USABLE_RATIO = 0.82
HOST_MEMORY_USABLE_RATIO = 0.78
# The largest supported PLE tables are roughly 95 GiB when resident in memory.
NGRAM_MEMORY_RESERVE = 96 * GIB
MOE_ARCHITECTURES = frozenset({
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV4ForCausalLM",
    "DeepSeekV4ForCausalLM",
    "Dots3NoteForCausalLM",
    "Ernie4_5_MoeForCausalLM",
    "Glm4MoeForCausalLM",
    "Glm5NextForConditionalGeneration",
    "GlmMoeDsaForCausalLM",
    "HunYuanMoEV1ForCausalLM",
    "HYV3ForCausalLM",
    "KimiK3ForConditionalGeneration",
    "LagunaForCausalLM",
    "MiniMaxM1ForCausalLM",
    "MiniMaxM2ForCausalLM",
    "MiniMaxText01ForCausalLM",
    "PanguProMoEForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
    "Step3p5ForCausalLM",
    "Step3p7ForConditionalGeneration",
})
MOE_MODEL_TYPES = frozenset({
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v4",
    "dots3_note",
    "glm5_next",
    "glm5_next_text",
    "glm_moe_dsa",
    "hy_v3",
    "kimi_k3",
    "laguna",
    "minimax_m1",
    "minimax_m2",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_8_flash_next",
    "qwen3_8_flash_next_text",
    "qwen3_moe",
    "qwen3_next",
    "qwen4_exp",
    "qwen4_exp_text",
    "step3p5",
    "step3p7",
})
NGRAM_ARCHITECTURES = frozenset({
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen4ExpForConditionalGeneration",
})
NGRAM_MODEL_TYPES = frozenset({
    "qwen3_8_flash_next",
    "qwen3_8_flash_next_text",
    "qwen4_exp",
    "qwen4_exp_text",
})
TP_UNSUPPORTED_ARCHITECTURES = frozenset({
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "Glm4MoeForCausalLM",
    "GlmMoeDsaForCausalLM",
    "Step3p5ForCausalLM",
    "Step3p7ForConditionalGeneration",
})
TP_SUPPORTED_MOE_ARCHITECTURES = frozenset({
    "DeepseekV4ForCausalLM",
    "DeepSeekV4ForCausalLM",
    "HYV3ForCausalLM",
    "LagunaForCausalLM",
    "MiniMaxM2ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
})
TP_SUPPORTED_MOE_MODEL_TYPES = frozenset({
    "deepseek_v4",
    "hy_v3",
    "laguna",
    "minimax_m2",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_moe",
    "qwen3_next",
})


class LauncherError(RuntimeError):
    pass


@dataclass
class DownloadConfig:
    model_id: str = DEFAULT_MODELSCOPE_MODEL_ID
    target_dir: str = ""
    revision: str = "master"
    max_workers: int = 4
    token: str = ""


def _modelscope_is_available() -> bool:
    try:
        return importlib.util.find_spec("modelscope") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _child_process_options(
    environment: Dict[str, str], cwd: Optional[str] = None
) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
        "env": environment,
    }
    if cwd is not None:
        options["cwd"] = cwd
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return options


def _coerce_config(payload: Any) -> DeployConfig:
    if not isinstance(payload, dict):
        raise LauncherError("Invalid configuration format.")
    config = config_from_dict(payload)
    if config.command not in ("server", "webui"):
        raise LauncherError("The browser launcher supports only ftllm server or ftllm webui.")
    return config


def _coerce_download_config(payload: Any) -> DownloadConfig:
    if not isinstance(payload, dict):
        raise LauncherError("Invalid download configuration format.")

    def text_value(name: str, default: str = "") -> str:
        value = payload.get(name, default)
        return "" if value is None else str(value)

    model_id = text_value("modelId").strip()
    raw_target_dir = text_value("targetDir").strip()
    target_dir = (
        os.path.abspath(os.path.expanduser(raw_target_dir))
        if raw_target_dir
        else ""
    )
    revision = text_value("revision", "master").strip() or "master"
    token = text_value("token")
    try:
        raw_max_workers = payload.get("maxWorkers", 4)
        max_workers = 0 if isinstance(raw_max_workers, bool) else int(raw_max_workers)
    except (TypeError, ValueError):
        max_workers = 0
    return DownloadConfig(
        model_id=model_id,
        target_dir=target_dir,
        revision=revision,
        max_workers=max_workers,
        token=token,
    )


def _validate_download_config(config: DownloadConfig) -> List[str]:
    errors = []
    if not config.model_id:
        errors.append("ModelScope model ID is required.")
    elif (
        len(config.model_id) > 512
        or CONTROL_CHARACTER_PATTERN.search(config.model_id)
        or any(character.isspace() for character in config.model_id)
        or config.model_id.startswith("-")
        or "/" not in config.model_id
    ):
        errors.append("ModelScope model ID must use the owner/model-name format.")
    if not config.target_dir:
        errors.append("Model destination directory is required.")
    elif (
        len(config.target_dir) > 4096
        or CONTROL_CHARACTER_PATTERN.search(config.target_dir)
    ):
        errors.append("Model destination directory contains invalid characters.")
    elif os.path.isfile(config.target_dir):
        errors.append("Model destination directory cannot be an existing file.")
    if (
        not config.revision
        or len(config.revision) > 255
        or CONTROL_CHARACTER_PATTERN.search(config.revision)
        or any(character.isspace() for character in config.revision)
        or config.revision.startswith("-")
    ):
        errors.append("Model revision is invalid.")
    if config.max_workers < 1 or config.max_workers > 64:
        errors.append("Concurrent downloads must be an integer from 1 to 64.")
    if len(config.token) > 8192 or CONTROL_CHARACTER_PATTERN.search(config.token):
        errors.append("Access token contains invalid characters.")
    return errors


def _download_arguments(config: DownloadConfig) -> List[str]:
    return [
        "download",
        "--model",
        config.model_id,
        "--local_dir",
        config.target_dir,
        "--revision",
        config.revision,
        "--max-workers",
        str(config.max_workers),
    ]


def _download_argv(config: DownloadConfig) -> List[str]:
    worker = Path(__file__).with_name("modelscope_download.py")
    return [sys.executable, str(worker), *_download_arguments(config)]


def _download_display_command(config: DownloadConfig) -> str:
    argv = ["modelscope", *_download_arguments(config)]
    suffix = "  [Token injected through environment]" if config.token else ""
    return " ".join(shlex.quote(part) for part in argv) + suffix


def _display_endpoint(config: DeployConfig) -> str:
    host = (
        "127.0.0.1"
        if config.command == "webui"
        else config.host.strip() or "127.0.0.1"
    )
    return f"http://{_browser_host(host)}:{config.port.strip()}"


def _redacted_command(
    argv: List[str], environment_overrides: Dict[str, str]
) -> str:
    safe_argv = list(argv)
    for index, value in enumerate(safe_argv):
        if value == "--api_key":
            if index + 1 < len(safe_argv):
                safe_argv[index + 1] = "••••••••"
        elif value.startswith("--api_key="):
            safe_argv[index] = "--api_key=••••••••"
    environment = []
    for name, value in environment_overrides.items():
        shown = "••••••••" if SENSITIVE_ENV_PATTERN.search(name) else value
        environment.append(f"{name}={shlex.quote(shown)}")
    return " ".join(environment + [shlex.quote(part) for part in safe_argv])


def _log_level(line: str) -> str:
    lower = line.lower()
    if any(
        word in lower
        for word in ("error", "exception", "traceback", "failed")
    ):
        return "error"
    if "warn" in lower:
        return "warning"
    return "info"


def _empty_runtime_speed() -> Dict[str, Any]:
    return {
        "prefillTokensPerSecond": None,
        "decodeTokensPerSecond": None,
        "prefillUpdatedAt": None,
        "decodeUpdatedAt": None,
    }


def _empty_runtime_state() -> Dict[str, Any]:
    return {
        "phase": "stopped",
        "command": "server",
        "pid": None,
        "ready": False,
        "model": "",
        "modelName": "",
        "profileName": "",
        "endpoint": "",
        "progress": 0.0,
        "progressStage": "",
        "progressLabel": "Model has not been started",
        "progressIndeterminate": False,
        "message": "Choose a launch item to start a service",
        "startedAt": None,
        "sessionId": "",
        "exitCode": None,
        "speed": _empty_runtime_speed(),
        "contextWindowTokens": None,
    }


def _empty_download_state() -> Dict[str, Any]:
    return {
        "phase": "idle",
        "pid": None,
        "modelId": "",
        "destination": "",
        "progress": 0.0,
        "progressIndeterminate": False,
        "downloadedBytes": 0,
        "totalBytes": 0,
        "completedFiles": 0,
        "totalFiles": 0,
        "message": "Download has not started",
        "startedAt": None,
        "exitCode": None,
    }


class LauncherRuntime:
    """Own launcher subprocesses and expose thread-safe serializable state.

    Each subprocess kind has a monotonically increasing generation. Reader and
    watcher threads ignore events from older generations so a late callback
    cannot overwrite the state of a replacement process.
    """

    def __init__(self, config_path: str = "", popen_factory=None, webui_history_dir: str = "",
                 agent_workspace_root: str = "", allow_remote_workspace_agent: bool = True,
                 disable_workspace_agent: bool = False):
        self.config_path = os.path.abspath(os.path.expanduser(
            config_path or get_saved_commands_path()
        ))
        self._popen = popen_factory or subprocess.Popen
        self._lock = threading.RLock()
        self._process = None
        self._generation = 0
        self._stopping_generation = -1
        self._state = _empty_runtime_state()
        self._service_api_key = ""
        self._webui_app = None
        self._webui_session = ""
        self._webui_history_dir = webui_history_dir
        self._agent_workspace_root = agent_workspace_root
        self._allow_remote_workspace_agent = allow_remote_workspace_agent
        self._disable_workspace_agent = disable_workspace_agent
        self._download_process = None
        self._download_generation = 0
        self._download_stopping_generation = -1
        self._download_state = _empty_download_state()
        self._logs = deque(maxlen=LOG_LIMIT)
        self._next_log_id = 1
        self._last_progress_stage = ""
        self._shutdown_callback = None

    def set_shutdown_callback(self, callback):
        self._shutdown_callback = callback

    def profiles(self) -> List[Dict[str, Any]]:
        return [asdict(config) for config in load_saved_configs(self.config_path)]

    def default_profile(self) -> Dict[str, Any]:
        configs = load_saved_configs(self.config_path)
        config = new_deploy_config(configs)
        config.command = "server"
        config.config_mode = "long_context"
        config.low_gpu_mem = True
        return asdict(config)

    def _close_webui_locked(self):
        if self._webui_app is not None:
            self._webui_app.state.runtime.close()
        self._webui_app = None
        self._webui_session = ""

    def embedded_webui(self, session_id: str, launcher_host: str):
        """Reuse the standalone WebUI with an immutable active-service client."""
        with self._lock:
            if (self._state["command"] != "server" or not self._state["ready"]
                    or self._state["phase"] != "running" or self._process is None
                    or self._process.poll() is not None):
                raise LauncherError("Start an API Server before opening WebUI.")
            if not session_id or session_id != self._state["sessionId"]:
                raise LauncherError("The model service changed. Reopen WebUI.")
            if self._webui_app is not None and self._webui_session == session_id:
                return self._webui_app
            self._close_webui_locked()
            from .webui_server import add_webui_args, create_app
            args = add_webui_args(argparse.ArgumentParser()).parse_args([])
            args.host = launcher_host
            args.api_base = self._state["endpoint"].rstrip("/") + "/v1"
            args.api_model = self._state["modelName"]
            args.api_key = self._service_api_key
            args.agent_runtime = "auto"
            args.embedded = True
            args.allow_remote_workspace_agent = self._allow_remote_workspace_agent
            args.disable_workspace_agent = self._disable_workspace_agent
            if self._agent_workspace_root:
                args.agent_workspace_root = self._agent_workspace_root
            if self._webui_history_dir:
                args.history_dir = self._webui_history_dir
            self._webui_app = create_app(args)
            self._webui_session = session_id
            return self._webui_app

    def download_defaults(self) -> Dict[str, Any]:
        model_id = DEFAULT_MODELSCOPE_MODEL_ID
        return {
            "modelId": model_id,
            "targetDir": default_modelscope_target_dir(model_id),
            "revision": "master",
            "maxWorkers": 4,
            "token": "",
        }

    def download_catalog(self) -> List[Dict[str, Any]]:
        groups = []
        for key, label, choices in MODELSCOPE_MODEL_GROUPS:
            models = [
                {"id": model_id, "label": model_label}
                for model_id, model_label in choices
                if model_id != "custom"
            ]
            if models:
                groups.append({"id": key, "label": label, "models": models})
        return groups

    def save_profile(self, index: Optional[int], payload: Any) -> Dict[str, Any]:
        config = _coerce_config(payload)
        configs = load_saved_configs(self.config_path)
        if not config.name.strip():
            config.name = new_deploy_config(configs).name
        if isinstance(index, int) and not isinstance(index, bool):
            if index < 0 or index >= len(configs):
                raise LauncherError("The profile to save no longer exists. Refresh and try again.")
            configs[index] = config
            saved_index = index
        else:
            configs.append(config)
            saved_index = len(configs) - 1
        save_saved_configs(configs, self.config_path)
        return {
            "index": saved_index,
            "profile": asdict(config),
            "profiles": [asdict(item) for item in configs],
        }

    def delete_profile(self, index: int) -> Dict[str, Any]:
        configs = load_saved_configs(self.config_path)
        if index < 0 or index >= len(configs):
            raise LauncherError("The profile to delete no longer exists. Refresh and try again.")
        deleted = configs.pop(index)
        save_saved_configs(configs, self.config_path)
        return {
            "deleted": asdict(deleted),
            "profiles": [asdict(item) for item in configs],
        }

    def preview(self, payload: Any) -> Dict[str, Any]:
        config = _coerce_config(payload)
        errors = validate_config(config)
        try:
            command = _redacted_command(
                build_fastllm_argv(config), build_fastllm_env(config)
            )
        except ValueError as error:
            command = ""
            errors.append(str(error))
        return {
            "command": command,
            "endpoint": _display_endpoint(config),
            "errors": errors,
        }

    def preview_download(self, payload: Any) -> Dict[str, Any]:
        config = _coerce_download_config(payload)
        errors = _validate_download_config(config)
        if not _modelscope_is_available():
            errors.append(MODELSCOPE_INSTALL_ERROR)
        return {
            "command": _download_display_command(config) if not errors else "",
            "destination": config.target_dir,
            "errors": errors,
        }

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {**self._state, "reportedAt": time.time(), "speed": dict(self._state["speed"])}

    def download_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._download_state)

    def logs(self, since: int = 0) -> Dict[str, Any]:
        with self._lock:
            entries = [dict(entry) for entry in self._logs if entry["id"] > since]
            return {"entries": entries, "lastId": self._next_log_id - 1}

    def clear_logs(self):
        with self._lock:
            self._logs.clear()

    def start_download(self, payload: Any) -> Dict[str, Any]:
        config = _coerce_download_config(payload)
        errors = _validate_download_config(config)
        if not _modelscope_is_available():
            errors.append(MODELSCOPE_INSTALL_ERROR)
        if errors:
            raise LauncherError("\n".join(errors))

        parent = os.path.dirname(config.target_dir) or os.getcwd()
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as error:
            raise LauncherError(f"Unable to create model destination directory: {error}") from error

        environment = os.environ.copy()
        if config.token:
            environment["MODELSCOPE_API_TOKEN"] = config.token
        environment["PYTHONUNBUFFERED"] = "1"
        popen_options = _child_process_options(environment, cwd=parent)
        argv = _download_argv(config)
        display_command = _download_display_command(config)

        with self._lock:
            if (
                self._download_process is not None
                and self._download_process.poll() is None
            ):
                raise LauncherError(
                    "A model download is already running. Wait for it to finish or cancel it first."
                )
            self._download_generation += 1
            generation = self._download_generation
            self._download_stopping_generation = -1
            self._download_state.update({
                "phase": "starting",
                "pid": None,
                "modelId": config.model_id,
                "destination": config.target_dir,
                "progress": 0.0,
                "progressIndeterminate": True,
                "downloadedBytes": 0,
                "totalBytes": 0,
                "completedFiles": 0,
                "totalFiles": 0,
                "message": "Connecting to ModelScope...",
                "startedAt": time.time(),
                "exitCode": None,
            })
            self._append_log(
                "modelscope", "info", f"Download command: {display_command}"
            )
            try:
                process = self._popen(argv, **popen_options)
            except Exception as error:
                self._download_state.update({
                    "phase": "failed",
                    "progressIndeterminate": False,
                    "message": f"Failed to start download process: {error}",
                })
                self._append_log(
                    "modelscope", "error", f"Failed to start download process: {error}"
                )
                raise LauncherError(f"Unable to start model download: {error}") from error
            self._download_process = process
            self._download_state.update({
                "phase": "downloading",
                "pid": process.pid,
                "message": "Downloading model...",
            })

        self._start_stream_readers(
            process,
            self._consume_download_stream,
            "ftllm-launcher-download",
            generation,
        )
        threading.Thread(
            target=self._watch_download,
            args=(process, generation),
            name="ftllm-launcher-download",
            daemon=True,
        ).start()
        return self.download_state()

    def _consume_download_stream(self, stream, source: str, generation: int):
        if stream is None:
            return
        try:
            for raw_line in stream:
                line = ANSI_ESCAPE_PATTERN.sub("", raw_line).strip()
                if not line:
                    continue
                event = _parse_modelscope_download_progress(line)
                if event is not None:
                    self._handle_download_progress(event, generation)
                    continue
                self._append_log("modelscope", _log_level(line), line)
        except Exception as error:
            self._append_log(
                "launcher", "warning", f"Failed to read ModelScope {source}: {error}"
            )
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _handle_download_progress(self, event: Dict[str, int], generation: int):
        with self._lock:
            if generation != self._download_generation:
                return
            total_bytes = event["totalBytes"]
            downloaded_bytes = event["downloadedBytes"]
            total_files = event["totalFiles"]
            completed_files = event["completedFiles"]
            if total_bytes > 0:
                progress = downloaded_bytes * 100.0 / total_bytes
            elif total_files > 0:
                progress = completed_files * 100.0 / total_files
            else:
                progress = 0.0
            progress = max(
                float(self._download_state.get("progress", 0.0)),
                min(99.9, progress),
            )
            self._download_state.update({
                "phase": "downloading",
                "progress": progress,
                "progressIndeterminate": total_bytes <= 0 and total_files <= 0,
                "downloadedBytes": downloaded_bytes,
                "totalBytes": total_bytes,
                "completedFiles": completed_files,
                "totalFiles": total_files,
                "message": f"Downloading model... Overall progress {progress:.1f}%",
            })

    def _watch_download(self, process, generation: int):
        exit_code = process.wait()
        with self._lock:
            if generation != self._download_generation:
                return
            cancelling = self._download_stopping_generation == generation
            if self._download_process is process:
                self._download_process = None
            completed = exit_code == 0 and not cancelling
            self._download_state.update({
                "phase": "completed" if completed else (
                    "cancelled" if cancelling else "failed"
                ),
                "pid": None,
                "progress": 100.0 if completed else self._download_state["progress"],
                "progressIndeterminate": False,
                "message": (
                    "Model download completed"
                    if completed
                    else "Download cancelled. It can be resumed from the same directory."
                    if cancelling
                    else f"Model download failed (code={exit_code})"
                ),
                "exitCode": exit_code,
            })
            destination = self._download_state["destination"]
            message = self._download_state["message"]
        level = "info" if completed else ("warning" if cancelling else "error")
        self._append_log(
            "modelscope",
            level,
            f"Model saved to {destination}" if completed else message,
        )

    def stop_download(self) -> Dict[str, Any]:
        with self._lock:
            process = self._download_process
            generation = self._download_generation
            if process is None or process.poll() is not None:
                return dict(self._download_state)
            if self._download_stopping_generation == generation:
                return dict(self._download_state)
            self._download_stopping_generation = generation
            self._download_state.update({
                "phase": "cancelling",
                "progressIndeterminate": True,
                "message": "Cancelling download...",
            })
        self._append_log("modelscope", "warning", "Cancelling model download...")
        self._terminate_process_tree(process)
        return self.download_state()

    def _append_log(self, source: str, level: str, message: str):
        text = str(message).rstrip("\r\n")
        if not text:
            return
        if len(text) > LOG_MESSAGE_LIMIT:
            text = text[:LOG_MESSAGE_LIMIT] + "... (log truncated)"
        with self._lock:
            self._logs.append({
                "id": self._next_log_id,
                "timestamp": time.time(),
                "source": source,
                "level": level,
                "message": text,
            })
            self._next_log_id += 1

    @staticmethod
    def _start_stream_readers(process, target, name_prefix: str, generation: int):
        for stream, source in (
            (process.stdout, "stdout"),
            (process.stderr, "stderr"),
        ):
            threading.Thread(
                target=target,
                args=(stream, source, generation),
                name=f"{name_prefix}-{source}",
                daemon=True,
            ).start()

    def _update_state_for_generation(
        self, generation: int, process=None, **changes
    ) -> bool:
        with self._lock:
            if generation != self._generation:
                return False
            if process is not None and self._process is not process:
                return False
            self._state.update(changes)
            return True

    def start(self, payload: Any) -> Dict[str, Any]:
        config = _coerce_config(payload)
        errors = validate_config(config)
        if errors:
            raise LauncherError("\n".join(errors))
        port_host = (
            "127.0.0.1" if config.command == "webui" else config.host.strip()
        )
        argv = build_fastllm_argv(config)
        api_key = config.api_key.strip()
        for index, value in enumerate(argv):
            if value == "--api_key" and index + 1 < len(argv):
                api_key = argv[index + 1]
            elif value.startswith("--api_key="):
                api_key = value.split("=", 1)[1]
        environment_overrides = build_fastllm_env(config)
        display_command = _redacted_command(argv, environment_overrides)
        endpoint = _display_endpoint(config)
        argv = list(argv)
        if config.command == "server" and "--startup-progress" not in argv:
            argv.extend(["--startup-progress", "ndjson"])
        if argv[0] == "ftllm":
            argv = [sys.executable, "-m", f"{__package__}.cli"] + argv[1:]
        else:
            executable = shutil.which(argv[0])
            if executable:
                argv[0] = executable

        environment = os.environ.copy()
        environment.update(environment_overrides)
        environment["PYTHONUNBUFFERED"] = "1"
        popen_options = _child_process_options(environment)

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise LauncherError("A model service is already running. Stop it first.")
            if not _port_is_available(port_host, int(config.port)):
                raise LauncherError(
                    f"Service port {port_host}:{config.port.strip()} is already in use."
                )
            self._close_webui_locked()
            self._generation += 1
            self._service_api_key = api_key
            generation = self._generation
            self._stopping_generation = -1
            self._state.update({
                "phase": "starting",
                "command": config.command,
                "pid": None,
                "ready": False,
                "model": os.path.expanduser(config.model.strip()),
                "modelName": effective_model_name(config),
                "profileName": config.name.strip(),
                "endpoint": endpoint,
                "progress": 0.0,
                "progressStage": "initializing",
                "progressLabel": (
                    PROGRESS_LABELS["initializing"]
                    if config.command == "server"
                    else "Starting chat WebUI"
                ),
                "progressIndeterminate": True,
                "message": f"Starting ftllm {config.command}...",
                "startedAt": time.time(),
                "sessionId": secrets.token_hex(16),
                "exitCode": None,
                "speed": _empty_runtime_speed(),
                "contextWindowTokens": None,
            })
            self._last_progress_stage = ""
            self._append_log(
                "launcher", "info", f"Launch command: {display_command}"
            )
            try:
                process = self._popen(argv, **popen_options)
            except Exception as error:
                self._state.update({
                    "phase": "failed",
                    "progressIndeterminate": False,
                    "message": f"Failed to start process: {error}",
                })
                self._append_log(
                    "launcher", "error", f"Failed to start process: {error}"
                )
                raise LauncherError(
                    f"Unable to start ftllm {config.command}: {error}"
                ) from error
            self._process = process
            self._state["pid"] = process.pid

        self._start_stream_readers(
            process,
            self._consume_stream,
            "ftllm-launcher",
            generation,
        )
        threading.Thread(
            target=self._watch_process,
            args=(process, generation),
            name="ftllm-launcher-process",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._probe_readiness,
            args=(
                process,
                generation,
                endpoint,
                api_key,
                config.command,
            ),
            name="ftllm-launcher-readiness",
            daemon=True,
        ).start()
        return self.state()

    def _consume_stream(self, stream, source: str, generation: int):
        if stream is None:
            return
        try:
            for raw_line in stream:
                line = ANSI_ESCAPE_PATTERN.sub("", raw_line).rstrip("\r\n")
                if source == "stderr" and line.startswith(PROGRESS_PREFIX):
                    self._handle_progress(line[len(PROGRESS_PREFIX):], generation)
                    continue
                self._handle_inference_speed(line, generation)
                self._handle_context_window(line, generation)
                self._append_log("ftllm", _log_level(line), line)
        except Exception as error:
            self._append_log("launcher", "warning", f"Failed to read {source}: {error}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _handle_inference_speed(self, line: str, generation: int):
        """Capture the engine's latest throughput samples without estimating tokens."""
        match = INFERENCE_SPEED_PATTERN.fullmatch(line)
        if not match:
            return
        value = float(match[2])
        if not math.isfinite(value):
            return
        kind = "prefill" if match[1] == "Prompt" else "decode"
        with self._lock:
            if (generation != self._generation or self._state["command"] != "server"
                    or self._state["phase"] not in ("starting", "running")):
                return
            self._state["speed"] = {
                **self._state["speed"],
                kind + "TokensPerSecond": value,
                kind + "UpdatedAt": time.time(),
            }

    def _handle_context_window(self, line: str, generation: int):
        # server.py reports the effective per-session limit after warmup,
        # accounting for model, shared KV capacity, and explicit context settings.
        match = CONTEXT_WINDOW_PATTERN.search(line)
        if not match or int(match[1]) <= 0:
            return
        with self._lock:
            if (generation != self._generation or self._state["command"] != "server"
                    or self._state["phase"] not in ("starting", "running")):
                return
            self._state["contextWindowTokens"] = int(match[1])

    def _handle_progress(self, raw_event: str, generation: int):
        try:
            event = json.loads(raw_event)
        except json.JSONDecodeError:
            self._append_log("ftllm", "warning", raw_event)
            return
        stage = str(event.get("stage", ""))
        try:
            percent = float(event.get("percent", 0.0) or 0.0)
        except (TypeError, ValueError):
            percent = 0.0
        event_type = str(event.get("type", ""))
        label = PROGRESS_LABELS.get(stage, str(event.get("message", stage)))
        indeterminate = bool(event.get("indeterminate", False))
        changes = {
            "progress": max(0.0, min(100.0, percent)),
            "progressStage": stage,
            "progressLabel": label,
            "progressIndeterminate": indeterminate,
            "message": label,
        }
        if event_type == "startup.ready":
            changes.update({
                "phase": "running",
                "ready": True,
                "progress": 100.0,
                "progressIndeterminate": False,
                "message": "Local API is ready",
            })
        elif event_type == "startup.error":
            changes.update({
                "phase": "failed",
                "ready": False,
                "progressIndeterminate": False,
                "message": str(event.get("message", "Model startup failed")),
            })
        with self._lock:
            if generation != self._generation:
                return
            self._state.update(changes)
            should_log_stage = bool(
                stage and stage != self._last_progress_stage
            )
            if should_log_stage:
                self._last_progress_stage = stage
        if should_log_stage:
            self._append_log("launcher", "info", label)

    def _probe_readiness(
        self,
        process,
        generation: int,
        endpoint: str,
        api_key: str,
        command: str,
    ):
        headers = {}
        if api_key and command == "server":
            headers["Authorization"] = f"Bearer {api_key}"
        path = "/v1/models" if command == "server" else "/_stcore/health"
        url = endpoint.rstrip("/") + path
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while process.poll() is None:
            with self._lock:
                if (
                    generation != self._generation
                    or self._process is not process
                ):
                    return
                if self._state["ready"]:
                    return
            try:
                request = urllib.request.Request(url, headers=headers)
                with opener.open(request, timeout=0.7) as response:
                    if response.status == 200:
                        self._update_state_for_generation(
                            generation,
                            process=process,
                            phase="running",
                            ready=True,
                            progress=100.0,
                            progressStage="ready",
                            progressLabel=(
                                PROGRESS_LABELS["ready"]
                                if command == "server"
                                else "Chat WebUI is ready"
                            ),
                            progressIndeterminate=False,
                            message=(
                                "Local API is ready"
                                if command == "server"
                                else "Chat WebUI is ready"
                            ),
                        )
                        return
            except (OSError, urllib.error.URLError, ValueError):
                pass
            time.sleep(0.7)

    def _watch_process(self, process, generation: int):
        exit_code = process.wait()
        with self._lock:
            if generation != self._generation:
                return
            self._close_webui_locked()
            stopping = self._stopping_generation == generation
            command = self._state.get("command", "server")
            service_name = "WebUI" if command == "webui" else "Model service"
            if self._process is process:
                self._process = None
            self._state.update({
                "phase": "stopped" if stopping or exit_code == 0 else "failed",
                "pid": None,
                "ready": False,
                "progressIndeterminate": False,
                "message": (
                    f"{service_name} stopped"
                    if stopping or exit_code == 0
                    else f"{service_name} exited unexpectedly (code={exit_code})"
                ),
                "exitCode": exit_code,
            })
        level = "info" if stopping or exit_code == 0 else "error"
        self._append_log("launcher", level, self.state()["message"])

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self._close_webui_locked()
            process = self._process
            generation = self._generation
            if process is None or process.poll() is not None:
                self._process = None
                self._state.update(_empty_runtime_state())
                self._state["message"] = "No service is currently running"
                return dict(self._state)
            if self._stopping_generation == generation:
                return dict(self._state)
            self._stopping_generation = generation
            self._state.update({
                "phase": "stopping",
                "ready": False,
                "progressIndeterminate": True,
                "message": "Stopping the current service...",
            })
        self._append_log("launcher", "info", "Stopping the current service...")
        self._terminate_process_tree(process)
        return self.state()

    def _terminate_process_tree(self, process):
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=8,
                    check=False,
                )
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            else:
                process_group = os.getpgid(process.pid)
                os.killpg(process_group, signal.SIGTERM)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process_group, signal.SIGKILL)
                    process.wait(timeout=3)
        except (OSError, subprocess.SubprocessError) as error:
            self._append_log("launcher", "warning", f"Failed to stop process tree: {error}")
            try:
                process.kill()
                process.wait(timeout=3)
            except OSError:
                pass
            except subprocess.TimeoutExpired:
                self._append_log(
                    "launcher", "warning", "Timed out waiting for child processes to exit."
                )

    def request_shutdown(self):
        callback = self._shutdown_callback
        if callback is not None:
            timer = threading.Timer(0.15, callback)
            timer.daemon = True
            timer.start()

    def close(self):
        self.stop_download()
        self.stop()


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return ""


def _port_is_available(host: str, port: int) -> bool:
    host = host or "0.0.0.0"
    normalized_host = host.strip("[]")
    family = socket.AF_INET6 if ":" in normalized_host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as listener:
            listener.bind((normalized_host, port))
        return True
    except (OSError, OverflowError):
        return False


def _parse_modelscope_download_progress(line: str) -> Optional[Dict[str, int]]:
    value = str(line).strip()
    if not value.startswith(MODELSCOPE_PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(value[len(MODELSCOPE_PROGRESS_PREFIX):])
    except (json.JSONDecodeError, TypeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("type") not in ("download.plan", "download.progress")
    ):
        return None
    result = {}
    for key in ("downloadedBytes", "totalBytes", "completedFiles", "totalFiles"):
        raw = payload.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            return None
        result[key] = raw
    result["downloadedBytes"] = min(
        result["downloadedBytes"], result["totalBytes"]
    )
    result["completedFiles"] = min(
        result["completedFiles"], result["totalFiles"]
    )
    return result


def _memory_info() -> Dict[str, int]:
    values = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as file:
            for line in file:
                key, raw = line.split(":", 1)
                number = raw.strip().split()[0]
                values[key] = int(number) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return {
        "total": values.get("MemTotal", 0),
        "available": values.get("MemAvailable", 0),
    }


def _gpu_info() -> List[Dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    query = (
        "index,name,memory.total,memory.free,utilization.gpu,temperature.gpu,"
        "driver_version"
    )
    try:
        result = subprocess.run(
            [
                executable,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    output = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        output.append({
            "index": parts[0],
            "name": parts[1],
            "memoryTotalMiB": parts[2],
            "memoryFreeMiB": parts[3],
            "utilization": parts[4],
            "temperature": parts[5],
            "driver": parts[6],
        })
    return output


def detect_hardware(model_path: str = "") -> Dict[str, Any]:
    cpu_model = ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except (OSError, IndexError):
        pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1

    numa_nodes = []
    node_root = Path("/sys/devices/system/node")
    if node_root.is_dir():
        for node in sorted(node_root.glob("node[0-9]*")):
            numa_nodes.append({
                "name": node.name,
                "cpus": _read_text(str(node / "cpulist")),
                "memory": _read_text(str(node / "meminfo")).splitlines()[:1],
            })

    disk_target = os.path.expanduser(model_path) if model_path else os.getcwd()
    if not os.path.exists(disk_target):
        disk_target = os.path.dirname(disk_target) or os.getcwd()
    try:
        disk = shutil.disk_usage(disk_target)
        disk_info = {"path": disk_target, "total": disk.total, "free": disk.free}
    except OSError:
        disk_info = {"path": disk_target, "total": 0, "free": 0}

    try:
        from .env import env
        build = dict(env.build_info)
    except Exception:
        build = {}
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": {
            "model": cpu_model or platform.processor() or "Unknown CPU",
            "logical": os.cpu_count() or 1,
            "available": affinity,
        },
        "memory": _memory_info(),
        "gpus": _gpu_info(),
        "numa": numa_nodes,
        "disk": disk_info,
        "build": build,
    }


def _folder_browser_drives() -> List[Dict[str, str]]:
    if os.name != "nt":
        return []
    try:
        import ctypes
        # Enumerate drive letters without probing removable or network volumes.
        mask = ctypes.windll.kernel32.GetLogicalDrives()
    except (AttributeError, OSError):
        return []
    return [
        {"name": f"{chr(65 + index)}:", "path": f"{chr(65 + index)}:\\"}
        for index in range(26) if mask & (1 << index)
    ]


def browse_folders(path: str = "") -> Dict[str, Any]:
    """Return bounded folders and files from the machine running Launcher."""
    raw_path = str(path or "").strip()
    try:
        current = os.path.abspath(
            os.path.expanduser(raw_path or os.path.expanduser("~"))
        )
    except (OSError, ValueError) as error:
        raise LauncherError("Invalid folder path.") from error

    selected_file = current if os.path.isfile(current) else ""
    if selected_file:
        current = os.path.dirname(current)
    else:
        # A partially typed model path is useful as a starting point. Walk up
        # until an existing directory is found instead of falling back at once.
        while not os.path.isdir(current):
            parent = os.path.dirname(current)
            if parent == current:
                if raw_path:
                    raise LauncherError("Folder root is unavailable.")
                current = ""
                break
            current = parent

    if not current:
        home = os.path.abspath(os.path.expanduser("~"))
        current = home if os.path.isdir(home) else os.getcwd()

    folders = []
    files = []
    truncated = False
    try:
        with os.scandir(current) as entries:
            for entry in entries:
                try:
                    is_directory = entry.is_dir(follow_symlinks=True)
                    is_file = not is_directory and entry.is_file(follow_symlinks=True)
                except OSError:
                    continue
                if not is_directory and not is_file:
                    continue
                if len(folders) + len(files) >= FOLDER_BROWSER_ENTRY_LIMIT:
                    truncated = True
                    break
                (folders if is_directory else files).append({
                    "name": entry.name,
                    "path": os.path.abspath(entry.path),
                })
    except OSError as error:
        raise LauncherError(f"Unable to read folder: {error}") from error

    folders.sort(key=lambda item: (item["name"].casefold(), item["name"]))
    files.sort(key=lambda item: (item["name"].casefold(), item["name"]))
    parent = os.path.dirname(current)
    return {
        "path": current,
        "parent": "" if parent == current else parent,
        "drives": _folder_browser_drives(),
        "folders": folders,
        "files": files,
        "selectedFile": selected_file,
        "truncated": truncated,
    }


def _first_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "").strip()


def _positive_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def _read_model_config_for_recommendation(model_path: str) -> Dict[str, Any]:
    path = Path(os.path.abspath(os.path.expanduser(model_path)))
    if path.is_dir():
        config_path = path / "config.json"
    elif path.name.lower() == "config.json":
        config_path = path
    else:
        config_path = path.parent / "config.json"
    try:
        if not config_path.is_file() or config_path.stat().st_size > MODEL_CONFIG_SIZE_LIMIT:
            return {}
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return config if isinstance(config, dict) else {}


def _is_model_weight_file(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() in (".safetensors", ".gguf", ".flm", ".pt", ".pth"):
        return True
    return re.fullmatch(r"(?:pytorch_model|model)(?:[-_.].*)?\.bin", name) is not None


def _sum_model_weight_bytes(model_path: str) -> int:
    path = Path(os.path.abspath(os.path.expanduser(model_path)))
    try:
        if path.is_file():
            return path.stat().st_size if _is_model_weight_file(path) else 0
        if not path.is_dir():
            return 0
        total = 0
        for index, entry in enumerate(path.iterdir()):
            if index >= MODEL_WEIGHT_ENTRY_LIMIT:
                break
            try:
                if entry.is_file() and _is_model_weight_file(entry):
                    total += entry.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return 0


def _parse_parameter_billions(identity: str) -> float:
    values = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*b(?=$|[\s._-])", identity, re.IGNORECASE):
        value = _positive_number(match.group(1))
        if value < 10000:
            values.append(value)
    return max(values, default=0.0)


def _first_positive_config_number(*values: Any) -> float:
    for value in values:
        number = _positive_number(value)
        if number:
            return number
    return 0.0


def _detect_model_quantization(
    config: Dict[str, Any],
    text_config: Dict[str, Any],
    identity: str,
    model_path: str,
) -> Dict[str, Any]:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        quantization = text_config.get("quantization_config")
    if not isinstance(quantization, dict):
        quantization = {}
    lower_identity = f"{identity} {model_path}".lower()
    bits = int(_positive_number(quantization.get("bits")))
    group_size = int(_positive_number(quantization.get("group_size")))
    group_match = re.search(r"int4g(?:roup)?[_-]?(128|256)", lower_identity)
    if not group_size and group_match:
        group_size = int(group_match.group(1))
    if bits == 4 or re.search(r"(?:int4|awq|gptq)", lower_identity):
        dtype = f"int4g{group_size}" if group_size in (128, 256) else "int4"
        label = f"INT4 group {group_size}" if group_size else "INT4"
        return {
            "name": label,
            "dtype": dtype,
            "sourceBytesPerParameter": 0.62,
            "convertible": False,
        }
    quant_method = _first_text(quantization.get("quant_method")).lower()
    quant_format = _first_text(quantization.get("fmt")).lower()
    if (
        quant_method == "fp8"
        or quant_format in ("e4m3", "float8_e4m3fn")
        or re.search(r"(?:^|[\s._-])fp8(?=$|[\s._-])", lower_identity)
    ):
        return {
            "name": "FP8 E4M3",
            "dtype": "fp8_e4m3",
            "sourceBytesPerParameter": 1.05,
            "convertible": False,
        }
    if bits == 8 or re.search(r"(?:^|[\s._-])int8(?=$|[\s._-])", lower_identity):
        return {
            "name": "INT8",
            "dtype": "int8",
            "sourceBytesPerParameter": 1.05,
            "convertible": False,
        }
    configured_dtype = _first_text(
        text_config.get("torch_dtype")
        or text_config.get("dtype")
        or config.get("torch_dtype")
        or config.get("dtype")
    ).lower()
    source_bytes = 4.0 if configured_dtype in ("float32", "fp32") else 2.0
    suffix = Path(model_path).suffix.lower()
    fixed_format = suffix in (".gguf", ".flm")
    return {
        "name": suffix[1:].upper() if fixed_format else (configured_dtype or "Auto"),
        "dtype": "auto",
        "sourceBytesPerParameter": source_bytes,
        "convertible": not fixed_format and Path(model_path).is_dir(),
    }


def inspect_launch_model(model_path: str, name: str = "") -> Dict[str, Any]:
    model_path = str(model_path or "").strip()[:4096]
    config = _read_model_config_for_recommendation(model_path)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = {}
    architecture = _first_text(
        config.get("architectures") or text_config.get("architectures")
    )
    model_type = _first_text(config.get("model_type")).lower()
    text_model_type = _first_text(text_config.get("model_type")).lower()
    identity = " ".join(
        value for value in (name, model_path, architecture, model_type, text_model_type) if value
    )
    quantization = _detect_model_quantization(
        config,
        text_config,
        identity,
        model_path,
    )
    weight_bytes = _sum_model_weight_bytes(model_path)
    parameter_billions = _parse_parameter_billions(identity)
    if not parameter_billions and weight_bytes:
        parameter_billions = (
            weight_bytes
            / max(0.25, quantization["sourceBytesPerParameter"])
            / 1e9
        )
    expert_count = _first_positive_config_number(
        config.get("num_experts"),
        config.get("num_local_experts"),
        config.get("n_routed_experts"),
        text_config.get("num_experts"),
        text_config.get("num_local_experts"),
        text_config.get("n_routed_experts"),
    )
    type_values = {model_type, text_model_type}
    is_moe = (
        architecture in MOE_ARCHITECTURES
        or bool(type_values & MOE_MODEL_TYPES)
        or expert_count > 1
        or re.search(
            r"(?:\bmoe\b|\ba\d+(?:\.\d+)?b\b|deepseek[\s._-]*v[234])",
            identity,
            re.IGNORECASE,
        )
        is not None
    )
    uses_ngram = (
        architecture in NGRAM_ARCHITECTURES
        or bool(type_values & NGRAM_MODEL_TYPES)
    )
    return {
        "architecture": architecture,
        "modelType": model_type,
        "textModelType": text_model_type,
        "parameterBillions": parameter_billions,
        "weightBytes": weight_bytes,
        "quantization": quantization["name"],
        "recommendedDtype": quantization["dtype"],
        "sourceBytesPerParameter": quantization["sourceBytesPerParameter"],
        "convertible": quantization["convertible"],
        "isMoe": is_moe,
        "usesNgram": uses_ngram,
        "configFound": bool(config),
    }


def _estimate_launch_model_bytes(metadata: Dict[str, Any]) -> int:
    weight_bytes = int(_positive_number(metadata.get("weightBytes")))
    if not weight_bytes:
        parameters = _positive_number(metadata.get("parameterBillions")) * 1e9
        source_bytes = _positive_number(metadata.get("sourceBytesPerParameter")) or 2.0
        weight_bytes = int(parameters * source_bytes)
    if weight_bytes <= 0:
        return 0
    return int(weight_bytes * 1.1 + 1.25 * GIB)


def _normalize_recommendation_hardware(hardware: Dict[str, Any]) -> Dict[str, Any]:
    cpu = hardware.get("cpu") if isinstance(hardware.get("cpu"), dict) else {}
    memory = hardware.get("memory") if isinstance(hardware.get("memory"), dict) else {}
    build = hardware.get("build") if isinstance(hardware.get("build"), dict) else {}
    gpus = []
    for position, item in enumerate(hardware.get("gpus") or []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index", position))
        except (TypeError, ValueError):
            index = position
        total_bytes = int(_positive_number(item.get("memoryTotalMiB")) * 1024 ** 2)
        free_bytes = int(_positive_number(item.get("memoryFreeMiB")) * 1024 ** 2)
        if not free_bytes:
            free_bytes = total_bytes
        usable_bytes = int(min(total_bytes * GPU_USABLE_RATIO, free_bytes * 0.95))
        if total_bytes <= 0 or usable_bytes <= 0:
            continue
        gpus.append({
            "index": max(0, index),
            "name": _first_text(item.get("name")) or f"CUDA {index}",
            "totalBytes": total_bytes,
            "freeBytes": free_bytes,
            "usableBytes": usable_bytes,
        })
    gpus.sort(key=lambda gpu: (-gpu["usableBytes"], gpu["index"]))
    total_memory = int(_positive_number(memory.get("total")))
    available_memory = int(_positive_number(memory.get("available"))) or total_memory
    numa = hardware.get("numa") if isinstance(hardware.get("numa"), list) else []
    return {
        "cpuThreads": max(
            1,
            int(_positive_number(cpu.get("available") or cpu.get("logical")) or 1),
        ),
        "totalMemoryBytes": total_memory,
        "availableMemoryBytes": available_memory,
        "numaNodes": max(1, len(numa)),
        "gpus": gpus,
        "build": build,
    }


def _model_supports_tensor_parallel(metadata: Dict[str, Any]) -> bool:
    architecture = str(metadata.get("architecture") or "")
    if architecture in TP_UNSUPPORTED_ARCHITECTURES:
        return False
    model_type = str(metadata.get("modelType") or "")
    text_model_type = str(metadata.get("textModelType") or "")
    if metadata.get("isMoe"):
        return (
            architecture in TP_SUPPORTED_MOE_ARCHITECTURES
            or model_type in TP_SUPPORTED_MOE_MODEL_TYPES
            or text_model_type in TP_SUPPORTED_MOE_MODEL_TYPES
        )
    return model_type not in ("step3p5", "step3p7")


def _choose_gpu_launch_plan(
    metadata: Dict[str, Any], hardware: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    gpus = hardware["gpus"]
    if not gpus:
        return None
    required_bytes = _estimate_launch_model_bytes(metadata)
    if required_bytes <= 0:
        return {"gpus": gpus[:1]}
    maximum = len(gpus) if _model_supports_tensor_parallel(metadata) else 1
    for count in range(maximum, 0, -1):
        selected = gpus[:count]
        if all(gpu["usableBytes"] >= required_bytes / count for gpu in selected):
            return {"gpus": selected}
    return None


def recommend_launch_config(
    model_path: str,
    hardware: Optional[Dict[str, Any]] = None,
    name: str = "",
    config_mode: str = "custom",
    enable_speculative_decoding: bool = False,
) -> Dict[str, Any]:
    if config_mode not in ("long_context", "high_concurrency", "custom"):
        raise LauncherError("Unknown configuration mode.")
    model_path = str(model_path or "").strip()[:4096]
    if not model_path:
        raise LauncherError("Model path is required for automatic configuration.")
    expanded_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.exists(expanded_path):
        raise LauncherError("Model path does not exist; choose a local model first.")
    metadata = inspect_launch_model(expanded_path, str(name or "")[:512])
    normalized_hardware = _normalize_recommendation_hardware(
        hardware if hardware is not None else detect_hardware(expanded_path)
    )
    parameter_billions = _positive_number(metadata.get("parameterBillions"))
    weight_bytes = _positive_number(metadata.get("weightBytes"))
    if parameter_billions > 70 or weight_bytes >= 80 * GIB:
        maximum_batch = "1"
    elif parameter_billions > 30 or weight_bytes >= 40 * GIB:
        maximum_batch = "2"
    else:
        maximum_batch = "4"
    config = {
        "config_mode": config_mode,
        "device": "auto",
        "cuda_device_id": "0",
        "tp": "2",
        "threads": "auto",
        "gpu_mem_ratio": "0.9",
        "low_gpu_mem": config_mode == "long_context",
        "max_batch": maximum_batch,
        "max_context_length": "auto",
        "kv_cache_dtype": "auto",
        "kv_cache_limit": "auto",
        "tokens": "auto",
        "enable_moe_hybrid": False,
        "moe_device": "numa",
        "moe_device_layers": "-1",
        "moe_device_custom": "",
        "moe_atype": "auto",
        "ngram_device": "auto",
        # Draft compatibility is model-specific. Keep speculative decoding
        # opt-in instead of inferring MTP or an external draft checkpoint.
        "speculative_algorithm": "auto",
        "speculative_draft_model_path": "",
        "mtp": "auto",
        "draft_tokens": "auto",
    }
    strategy = "automatic"
    gpu_plan = _choose_gpu_launch_plan(metadata, normalized_hardware)
    if gpu_plan:
        selected = sorted(gpu_plan["gpus"], key=lambda gpu: gpu["index"])
        if len(selected) == 1:
            strategy = "cuda"
            config["device"] = "cuda"
            config["cuda_device_id"] = str(selected[0]["index"])
        else:
            strategy = "tensor_parallel"
            config["device"] = "tp"
            config["tp"] = ",".join(str(gpu["index"]) for gpu in selected)
    elif metadata.get("isMoe") and normalized_hardware["gpus"]:
        best_gpu = normalized_hardware["gpus"][0]
        runtime_bytes = _estimate_launch_model_bytes(metadata)
        available_memory = normalized_hardware["availableMemoryBytes"]
        memory_fits = (
            runtime_bytes <= 0
            or available_memory <= 0
            or runtime_bytes <= available_memory * HOST_MEMORY_USABLE_RATIO
        )
        config["device"] = "cuda"
        config["cuda_device_id"] = str(best_gpu["index"])
        config["max_batch"] = "1"
        config["enable_moe_hybrid"] = True
        if memory_fits:
            config["moe_device"] = (
                "numa" if normalized_hardware["numaNodes"] > 1 else "cpu"
            )
            strategy = (
                "hybrid_numa"
                if config["moe_device"] == "numa"
                else "hybrid_cpu"
            )
        else:
            config["moe_device"] = "disk"
            strategy = "hybrid_disk"
    elif normalized_hardware["build"].get("USE_ROCM"):
        strategy = "automatic"
        config["device"] = "auto"
    else:
        config["device"] = (
            "numa" if normalized_hardware["numaNodes"] > 1 else "cpu"
        )
        strategy = "numa" if config["device"] == "numa" else "cpu"
        runtime_bytes = _estimate_launch_model_bytes(metadata)
        available_memory = normalized_hardware["availableMemoryBytes"]
        if (
            metadata.get("isMoe")
            and available_memory > 0
            and runtime_bytes > available_memory * HOST_MEMORY_USABLE_RATIO
        ):
            config["enable_moe_hybrid"] = True
            config["moe_device"] = "disk"
            strategy = "cpu_disk"

    ngram_disk = False
    if metadata.get("usesNgram"):
        runtime_bytes = _estimate_launch_model_bytes(metadata)
        memory_headroom = normalized_hardware["availableMemoryBytes"] - runtime_bytes
        if config["moe_device"] == "disk" or memory_headroom < NGRAM_MEMORY_RESERVE:
            config["ngram_device"] = "disk"
            ngram_disk = True

    if config_mode == "long_context":
        # Reserve the cache for a single long conversation. Context capacity
        # still follows the model and the runtime's available-memory limit.
        config["max_batch"] = "1"
    elif config_mode == "high_concurrency":
        # Let the engine size batching from its actual cache budget and model capabilities.
        config["max_batch"] = "auto"

    speculative_reason = "not_requested"
    if enable_speculative_decoding:
        speculative_reason = detect_mtp_support(
            expanded_path, _read_model_config_for_recommendation(expanded_path)
        )
        if speculative_reason == "enabled":
            build = normalized_hardware["build"]
            if (config["device"] in ("cuda", "tp")
                    and build.get("USE_CUDA") is not False and not build.get("USE_ROCM")):
                config["mtp"] = "3"
                config["speculative_algorithm"] = "mtp"
            else:
                speculative_reason = "cuda_required"

    selected_gpu_ids = []
    if config["device"] == "cuda":
        selected_gpu_ids = [int(config["cuda_device_id"])]
    elif config["device"] == "tp":
        selected_gpu_ids = [int(value) for value in config["tp"].split(",")]
    return {
        "version": 1,
        "strategy": strategy,
        "config": config,
        "speculative": {
            "requested": bool(enable_speculative_decoding),
            "enabled": speculative_reason == "enabled",
            "reason": speculative_reason,
        },
        "detected": {
            "architecture": metadata["architecture"],
            "modelType": metadata["modelType"],
            "parameterBillions": round(parameter_billions, 2),
            "weightGiB": round(weight_bytes / GIB, 2),
            "quantization": metadata["quantization"],
            "isMoe": bool(metadata["isMoe"]),
            "usesNgram": bool(metadata["usesNgram"]),
            "configFound": bool(metadata["configFound"]),
        },
        "hardware": {
            "cpuThreads": normalized_hardware["cpuThreads"],
            "availableMemoryGiB": round(
                normalized_hardware["availableMemoryBytes"] / GIB, 1
            ),
            "numaNodes": normalized_hardware["numaNodes"],
            "gpus": [
                {
                    "index": gpu["index"],
                    "name": gpu["name"],
                    "totalGiB": round(gpu["totalBytes"] / GIB, 1),
                    "freeGiB": round(gpu["freeBytes"] / GIB, 1),
                }
                for gpu in normalized_hardware["gpus"]
            ],
            "selectedGpuIds": selected_gpu_ids,
        },
        "adjustments": {
            "ngramOnDisk": ngram_disk,
            "metadataLimited": not metadata["configFound"] or weight_bytes <= 0,
        },
    }


def create_launcher_app(
    runtime: LauncherRuntime,
    control_token: str,
    launcher_addresses: Optional[List[Dict[str, str]]] = None,
    launcher_host: str = "127.0.0.1",
):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from starlette.concurrency import run_in_threadpool
    except ImportError as error:
        raise LauncherError(
            "The launcher requires fastapi and uvicorn. Install ftllm[server]."
        ) from error

    if not ASSET_DIRECTORY.is_dir():
        raise LauncherError(f"Launcher web assets were not found: {ASSET_DIRECTORY}")
    if not control_token:
        raise LauncherError("Launcher control token must not be empty.")

    advertised_addresses = [
        dict(address) for address in (launcher_addresses or [])
    ]
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    webui_cookie = "ftllm-webui-" + hashlib.sha256(control_token.encode()).hexdigest()[:12]
    webui_token = hmac.new(control_token.encode(), b"embedded-webui", hashlib.sha256).hexdigest()

    @app.middleware("http")
    async def protect_launcher(request: Request, call_next):
        is_webui = request.url.path.startswith("/webui/")
        if is_webui:
            supplied = request.cookies.get(webui_cookie, "")
            if not hmac.compare_digest(supplied, webui_token):
                return JSONResponse({"detail": "Open WebUI from the Launcher."}, status_code=403)
            origin = request.headers.get("origin")
            if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Invalid request origin."}, status_code=403)
        if request.url.path.startswith("/api/"):
            content_length = request.headers.get("content-length", "0")
            try:
                if int(content_length or "0") > REQUEST_BODY_LIMIT:
                    return JSONResponse(
                        {"error": "Request body is too large."}, status_code=413
                    )
            except ValueError:
                return JSONResponse({"error": "Invalid Content-Length."}, status_code=400)
            supplied = request.headers.get("x-ftllm-launcher-token", "")
            if not hmac.compare_digest(supplied, control_token):
                return JSONResponse(
                    {"error": "Invalid control token. Reopen the page from the ftllm launch URL."},
                    status_code=403,
                )
        response = await call_next(request)
        if is_webui:
            # Downloaded resources must not become executable documents.
            response.headers.setdefault("Content-Security-Policy", (
                "default-src 'self'; script-src 'none'; style-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'self'"
            ))
            if "/attachments/" in request.url.path:
                response.headers["Content-Security-Policy"] += "; sandbox allow-downloads"
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data: blob:; media-src 'self' data: blob:; "
                "connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
            )
            response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if is_webui or request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(LauncherError)
    async def launcher_error_handler(_request: Request, error: LauncherError):
        return JSONResponse({"error": str(error)}, status_code=400)

    @app.exception_handler(json.JSONDecodeError)
    async def invalid_json_handler(_request: Request, _error: json.JSONDecodeError):
        return JSONResponse({"error": "Invalid JSON request body."}, status_code=400)

    @app.get("/")
    async def index():
        return FileResponse(ASSET_DIRECTORY / "index.html")

    @app.get("/api/bootstrap")
    async def bootstrap():
        try:
            from . import __version__
        except Exception:
            __version__ = "unknown"
        return {
            "version": __version__,
            "profiles": runtime.profiles(),
            "defaultProfile": runtime.default_profile(),
            "runtime": runtime.state(),
            "download": runtime.download_state(),
            "downloadDefaults": runtime.download_defaults(),
            "downloadCatalog": runtime.download_catalog(),
            "logs": runtime.logs(),
            "configPath": runtime.config_path,
            "launcherAddresses": advertised_addresses,
        }

    @app.post("/api/preview")
    async def preview(request: Request):
        payload = await request.json()
        return await run_in_threadpool(runtime.preview, payload)

    @app.post("/api/recommend")
    async def recommend(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise LauncherError("Invalid automatic configuration request.")
        model_path = str(payload.get("model") or "")[:4096]
        name = str(payload.get("name") or "")[:512]
        config_mode = str(payload.get("config_mode") or "custom")
        return await run_in_threadpool(
            recommend_launch_config,
            model_path,
            None,
            name,
            config_mode,
            payload.get("enable_speculative_decoding") is True,
        )

    @app.post("/api/profiles")
    async def save_profile(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise LauncherError("Invalid configuration format.")
        return runtime.save_profile(payload.get("index"), payload.get("config"))

    @app.delete("/api/profiles/{index}")
    async def delete_profile(index: int):
        return runtime.delete_profile(index)

    @app.post("/api/runtime/start")
    async def start_runtime(request: Request):
        payload = await request.json()
        return await run_in_threadpool(runtime.start, payload)

    @app.post("/api/runtime/stop")
    async def stop_runtime():
        return await run_in_threadpool(runtime.stop)

    @app.get("/api/runtime")
    async def runtime_state():
        return runtime.state()

    @app.post("/api/webui/open")
    async def open_webui(request: Request):
        payload = await request.json()
        session_id = payload.get("sessionId", "") if isinstance(payload, dict) else ""
        await run_in_threadpool(runtime.embedded_webui, session_id, launcher_host)
        response = JSONResponse({"url": f"/webui/{session_id}/"})
        response.set_cookie(
            webui_cookie, webui_token, path="/webui/", httponly=True,
            samesite="strict", secure=request.url.scheme == "https",
        )
        return response

    async def embedded_webui(scope, receive, send):
        session_id = scope.get("path_params", {}).get("session_id", "")
        try:
            webui_app = await run_in_threadpool(runtime.embedded_webui, session_id, launcher_host)
        except LauncherError as error:
            await JSONResponse({"detail": str(error)}, status_code=409)(scope, receive, send)
            return
        await webui_app(scope, receive, send)

    app.mount("/webui/{session_id}", embedded_webui, name="embedded-webui")

    @app.post("/api/download/preview")
    async def preview_download(request: Request):
        payload = await request.json()
        return await run_in_threadpool(runtime.preview_download, payload)

    @app.post("/api/download/start")
    async def start_download(request: Request):
        payload = await request.json()
        return await run_in_threadpool(runtime.start_download, payload)

    @app.post("/api/download/cancel")
    async def cancel_download():
        return await run_in_threadpool(runtime.stop_download)

    @app.get("/api/download")
    async def download_state():
        return runtime.download_state()

    @app.get("/api/logs")
    async def logs(since: int = 0):
        return runtime.logs(max(0, since))

    @app.delete("/api/logs")
    async def clear_logs():
        runtime.clear_logs()
        return {"ok": True}

    @app.get("/api/paths")
    async def paths(prefix: str = "", directories_only: bool = False):
        prefix = prefix[:4096]
        return {
            "paths": (
                await run_in_threadpool(
                    complete_path_prefix,
                    prefix,
                    directories_only,
                )
            )[:40]
        }

    @app.get("/api/folders")
    async def folders(path: str = ""):
        return await run_in_threadpool(browse_folders, path[:4096])

    @app.get("/api/hardware")
    async def hardware(model_path: str = ""):
        return await run_in_threadpool(detect_hardware, model_path[:4096])

    @app.post("/api/shutdown")
    async def shutdown():
        runtime.request_shutdown()
        return {"ok": True}

    # Both entry points serve exactly the same component resources.
    app.mount("/assets/webui", StaticFiles(directory=str(
        Path(__file__).with_name("webui_assets"))), name="webui-assets")

    @app.get("/assets/webui_locales.js")
    async def webui_locales():
        return FileResponse(Path(__file__).with_name("webui_locales.js"),
                            media_type="application/javascript; charset=utf-8")

    app.mount(
        "/assets",
        StaticFiles(directory=str(ASSET_DIRECTORY)),
        name="launcher-assets",
    )
    return app


def _browser_host(host: str) -> str:
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host in ("::", "[::]"):
        return "[::1]"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]")
    if normalized.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_usable_ip_address(address) -> bool:
    return not (
        address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
    )


def _interface_ip_addresses() -> List[str]:
    addresses = set()

    def add_address(value: Any):
        normalized = str(value or "").split("%", 1)[0].strip().strip("[]")
        try:
            address = ipaddress.ip_address(normalized)
        except ValueError:
            return
        if not _is_usable_ip_address(address):
            return
        addresses.add(str(address))

    ip_command = shutil.which("ip")
    if ip_command:
        try:
            result = subprocess.run(
                [ip_command, "-j", "address", "show", "up"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            interfaces = json.loads(result.stdout) if result.returncode == 0 else []
            if not isinstance(interfaces, list):
                interfaces = []
            for interface in interfaces:
                if not isinstance(interface, dict):
                    continue
                name = str(interface.get("ifname", ""))
                flags = set(interface.get("flags") or [])
                virtual = (
                    name == "docker0"
                    or name.startswith(("veth", "virbr", "podman", "cni"))
                    or re.fullmatch(r"br-[0-9a-f]{6,}", name) is not None
                )
                if (
                    virtual
                    or "LOOPBACK" in flags
                    or "NO-CARRIER" in flags
                    or interface.get("operstate") == "DOWN"
                ):
                    continue
                for info in interface.get("addr_info") or []:
                    if isinstance(info, dict):
                        add_address(info.get("local"))
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass

    host_name = socket.gethostname()
    if host_name:
        try:
            resolved = socket.getaddrinfo(
                host_name, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
        except socket.gaierror:
            resolved = []
        for result in resolved:
            add_address(result[4][0])

    route_probes = (
        (socket.AF_INET, ("192.0.2.1", 9)),
        (socket.AF_INET6, ("2001:db8::1", 9, 0, 0)),
    )
    for family, target in route_probes:
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as probe:
                probe.connect(target)
                add_address(probe.getsockname()[0])
        except OSError:
            pass

    return sorted(
        addresses,
        key=lambda value: (
            ipaddress.ip_address(value).version,
            int(ipaddress.ip_address(value)),
        ),
    )


def _launcher_access_addresses(
    host: str,
    port: int,
    interface_addresses: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    normalized_host = str(host or "").strip().strip("[]") or "0.0.0.0"
    candidates = (
        _interface_ip_addresses()
        if interface_addresses is None
        else interface_addresses
    )
    records: List[Dict[str, str]] = []
    seen_urls = set()

    def add_record(scope: str, label: str, address: str):
        url = f"http://{_browser_host(address)}:{port}"
        if url in seen_urls:
            return
        seen_urls.add(url)
        records.append({"scope": scope, "label": label, "url": url})

    if normalized_host in ("0.0.0.0", "::"):
        version = 4 if normalized_host == "0.0.0.0" else 6
        loopback = "127.0.0.1" if version == 4 else "::1"
        add_record("local", "Local address", loopback)
        for value in candidates:
            try:
                address = ipaddress.ip_address(str(value).split("%", 1)[0])
            except ValueError:
                continue
            if address.version != version:
                continue
            if address.is_global:
                add_record("public", "Public address", str(address))
            elif _is_usable_ip_address(address):
                add_record("lan", "LAN address", str(address))
    else:
        try:
            bound_address = ipaddress.ip_address(normalized_host)
        except ValueError:
            if _is_loopback_host(normalized_host):
                add_record("local", "Local address", "127.0.0.1")
            else:
                add_record("custom", "Access address", normalized_host)
        else:
            if bound_address.is_loopback:
                add_record("local", "Local address", str(bound_address))
            elif bound_address.is_global:
                add_record("public", "Public address", str(bound_address))
            else:
                add_record("lan", "LAN address", str(bound_address))

    scope_order = {"local": 0, "lan": 1, "public": 2, "custom": 3}
    return sorted(
        records,
        key=lambda item: (scope_order.get(item["scope"], 9), item["url"]),
    )


def fastllm_launcher(args) -> int:
    host = str(
        getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    ).strip()
    host = host or "127.0.0.1"
    port = int(getattr(args, "port", 8000))
    if port < 1 or port > 65535:
        print("Launcher port must be in 1-65535.", file=sys.stderr)
        return 2

    try:
        import uvicorn
    except ImportError:
        print(
            "ftllm launch requires uvicorn. Install ftllm[server].",
            file=sys.stderr,
        )
        return 1

    control_token = secrets.token_urlsafe(32)
    runtime = LauncherRuntime(
        getattr(args, "config", ""),
        agent_workspace_root=getattr(args, "agent_workspace_root", ""),
        allow_remote_workspace_agent=bool(getattr(args, "allow_remote_workspace_agent", True)),
        disable_workspace_agent=bool(getattr(args, "disable_workspace_agent", False)),
    )
    launcher_addresses = _launcher_access_addresses(host, port)
    try:
        app = create_launcher_app(runtime, control_token, launcher_addresses, launcher_host=host)
    except LauncherError as error:
        print(str(error), file=sys.stderr)
        return 1

    browser_base_url = launcher_addresses[0]["url"]
    browser_url = f"{browser_base_url}/?token={control_token}"

    class LauncherServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if not self.started:
                return
            print("FastLLM Launcher access URLs (control token included):", flush=True)
            for address in launcher_addresses:
                control_url = (
                    f"{address['url']}/?token={control_token}"
                )
                print(f"  {address['label']}: {control_url}", flush=True)
            print(
                "Press Ctrl+C to stop the launcher and its managed tasks.",
                flush=True,
            )
            if not _is_loopback_host(host):
                print(
                    "Warning: Launcher control traffic is unencrypted on a "
                    "non-loopback address; use it only on a trusted network.",
                    file=sys.stderr,
                    flush=True,
                )
            if getattr(args, "no_browser", False):
                return
            threading.Thread(
                target=webbrowser.open,
                args=(browser_url,),
                name="ftllm-launcher-browser",
                daemon=True,
            ).start()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = LauncherServer(config)
    runtime.set_shutdown_callback(lambda: setattr(server, "should_exit", True))
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
    return 0
