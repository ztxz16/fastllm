import hmac
import importlib.util
import ipaddress
import json
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
    "initializing": "正在初始化模型",
    "tokenizer": "正在加载分词器",
    "weights_prepare": "正在准备模型权重",
    "weights_load": "正在读取模型权重",
    "weights_finalize": "正在整理模型权重",
    "warmup": "正在预热模型",
    "server_starting": "正在启动本地 API",
    "ready": "本地 API 已就绪",
}
MODELSCOPE_INSTALL_ERROR = (
    "缺少 modelscope，请先执行："
    "python -m pip install 'modelscope>=1.34.0,<2'。"
)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


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
        raise LauncherError("配置格式无效。")
    config = config_from_dict(payload)
    if config.command not in ("server", "webui"):
        raise LauncherError("浏览器启动器只支持 ftllm server 或 ftllm webui。")
    return config


def _coerce_download_config(payload: Any) -> DownloadConfig:
    if not isinstance(payload, dict):
        raise LauncherError("下载配置格式无效。")

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
        errors.append("ModelScope 模型 ID 不能为空。")
    elif (
        len(config.model_id) > 512
        or CONTROL_CHARACTER_PATTERN.search(config.model_id)
        or any(character.isspace() for character in config.model_id)
        or config.model_id.startswith("-")
        or "/" not in config.model_id
    ):
        errors.append("ModelScope 模型 ID 应为 owner/model-name 格式。")
    if not config.target_dir:
        errors.append("模型保存目录不能为空。")
    elif (
        len(config.target_dir) > 4096
        or CONTROL_CHARACTER_PATTERN.search(config.target_dir)
    ):
        errors.append("模型保存目录包含无效字符。")
    elif os.path.isfile(config.target_dir):
        errors.append("模型保存目录不能是已有文件。")
    if (
        not config.revision
        or len(config.revision) > 255
        or CONTROL_CHARACTER_PATTERN.search(config.revision)
        or any(character.isspace() for character in config.revision)
        or config.revision.startswith("-")
    ):
        errors.append("模型 Revision 无效。")
    if config.max_workers < 1 or config.max_workers > 64:
        errors.append("并发下载数必须是 1-64 的整数。")
    if len(config.token) > 8192 or CONTROL_CHARACTER_PATTERN.search(config.token):
        errors.append("Access Token 包含无效字符。")
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
    suffix = "  [Token 通过环境变量注入]" if config.token else ""
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
        "progressLabel": "尚未启动模型",
        "progressIndeterminate": False,
        "message": "填写配置后即可启动本地 API",
        "startedAt": None,
        "exitCode": None,
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
        "message": "尚未开始下载",
        "startedAt": None,
        "exitCode": None,
    }


class LauncherRuntime:
    """Own launcher subprocesses and expose thread-safe serializable state.

    Each subprocess kind has a monotonically increasing generation. Reader and
    watcher threads ignore events from older generations so a late callback
    cannot overwrite the state of a replacement process.
    """

    def __init__(self, config_path: str = "", popen_factory=None):
        self.config_path = os.path.abspath(os.path.expanduser(
            config_path or get_saved_commands_path()
        ))
        self._popen = popen_factory or subprocess.Popen
        self._lock = threading.RLock()
        self._process = None
        self._generation = 0
        self._stopping_generation = -1
        self._state = _empty_runtime_state()
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
        return asdict(config)

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
                raise LauncherError("要保存的配置不存在，请刷新页面后重试。")
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
            raise LauncherError("要删除的配置不存在，请刷新页面后重试。")
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
            return dict(self._state)

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
            raise LauncherError(f"无法创建模型保存目录：{error}") from error

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
                    "已有模型正在下载，请先等待完成或取消任务。"
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
                "message": "正在连接 ModelScope…",
                "startedAt": time.time(),
                "exitCode": None,
            })
            self._append_log(
                "modelscope", "info", f"下载命令：{display_command}"
            )
            try:
                process = self._popen(argv, **popen_options)
            except Exception as error:
                self._download_state.update({
                    "phase": "failed",
                    "progressIndeterminate": False,
                    "message": f"启动下载进程失败：{error}",
                })
                self._append_log(
                    "modelscope", "error", f"启动下载进程失败：{error}"
                )
                raise LauncherError(f"无法启动模型下载：{error}") from error
            self._download_process = process
            self._download_state.update({
                "phase": "downloading",
                "pid": process.pid,
                "message": "正在下载模型…",
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
                "launcher", "warning", f"读取 ModelScope {source} 失败：{error}"
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
                "message": f"正在下载模型… 总进度 {progress:.1f}%",
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
                    "模型下载完成"
                    if completed
                    else "下载已取消，可从原目录继续下载。"
                    if cancelling
                    else f"模型下载失败（code={exit_code}）"
                ),
                "exitCode": exit_code,
            })
            destination = self._download_state["destination"]
            message = self._download_state["message"]
        level = "info" if completed else ("warning" if cancelling else "error")
        self._append_log(
            "modelscope",
            level,
            f"模型已保存到 {destination}" if completed else message,
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
                "message": "正在取消下载…",
            })
        self._append_log("modelscope", "warning", "正在取消模型下载…")
        self._terminate_process_tree(process)
        return self.download_state()

    def _append_log(self, source: str, level: str, message: str):
        text = str(message).rstrip("\r\n")
        if not text:
            return
        if len(text) > LOG_MESSAGE_LIMIT:
            text = text[:LOG_MESSAGE_LIMIT] + "…（日志已截断）"
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
        if config.command == "webui":
            environment["STREAMLIT_SERVER_HEADLESS"] = "true"
        popen_options = _child_process_options(environment)

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise LauncherError("已有模型服务正在运行，请先停止当前服务。")
            if not _port_is_available(port_host, int(config.port)):
                raise LauncherError(
                    f"服务端口 {port_host}:{config.port.strip()} 已被占用。"
                )
            self._generation += 1
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
                    else "正在启动聊天 WebUI"
                ),
                "progressIndeterminate": True,
                "message": f"正在启动 ftllm {config.command}…",
                "startedAt": time.time(),
                "exitCode": None,
            })
            self._last_progress_stage = ""
            self._append_log(
                "launcher", "info", f"启动命令：{display_command}"
            )
            try:
                process = self._popen(argv, **popen_options)
            except Exception as error:
                self._state.update({
                    "phase": "failed",
                    "progressIndeterminate": False,
                    "message": f"启动进程失败：{error}",
                })
                self._append_log(
                    "launcher", "error", f"启动进程失败：{error}"
                )
                raise LauncherError(
                    f"无法启动 ftllm {config.command}：{error}"
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
                config.api_key,
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
                self._append_log("ftllm", _log_level(line), line)
        except Exception as error:
            self._append_log("launcher", "warning", f"读取 {source} 失败：{error}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

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
                "message": "本地 API 已就绪",
            })
        elif event_type == "startup.error":
            changes.update({
                "phase": "failed",
                "ready": False,
                "progressIndeterminate": False,
                "message": str(event.get("message", "模型启动失败")),
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
                                else "聊天 WebUI 已就绪"
                            ),
                            progressIndeterminate=False,
                            message=(
                                "本地 API 已就绪"
                                if command == "server"
                                else "聊天 WebUI 已就绪"
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
            stopping = self._stopping_generation == generation
            command = self._state.get("command", "server")
            service_name = "WebUI" if command == "webui" else "模型服务"
            if self._process is process:
                self._process = None
            self._state.update({
                "phase": "stopped" if stopping or exit_code == 0 else "failed",
                "pid": None,
                "ready": False,
                "progressIndeterminate": False,
                "message": (
                    f"{service_name}已停止"
                    if stopping or exit_code == 0
                    else f"{service_name}异常退出（code={exit_code}）"
                ),
                "exitCode": exit_code,
            })
        level = "info" if stopping or exit_code == 0 else "error"
        self._append_log("launcher", level, self.state()["message"])

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            process = self._process
            generation = self._generation
            if process is None or process.poll() is not None:
                self._process = None
                self._state.update(_empty_runtime_state())
                self._state["message"] = "当前服务未运行"
                return dict(self._state)
            if self._stopping_generation == generation:
                return dict(self._state)
            self._stopping_generation = generation
            self._state.update({
                "phase": "stopping",
                "ready": False,
                "progressIndeterminate": True,
                "message": "正在停止当前服务…",
            })
        self._append_log("launcher", "info", "正在停止当前服务…")
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
            self._append_log("launcher", "warning", f"停止进程树失败：{error}")
            try:
                process.kill()
                process.wait(timeout=3)
            except OSError:
                pass
            except subprocess.TimeoutExpired:
                self._append_log(
                    "launcher", "warning", "等待子进程退出超时。"
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
            "model": cpu_model or platform.processor() or "未知 CPU",
            "logical": os.cpu_count() or 1,
            "available": affinity,
        },
        "memory": _memory_info(),
        "gpus": _gpu_info(),
        "numa": numa_nodes,
        "disk": disk_info,
        "build": build,
    }


def create_launcher_app(
    runtime: LauncherRuntime,
    control_token: str,
    launcher_addresses: Optional[List[Dict[str, str]]] = None,
):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from starlette.concurrency import run_in_threadpool
    except ImportError as error:
        raise LauncherError(
            "启动器需要 fastapi 和 uvicorn，请安装 ftllm[server]。"
        ) from error

    if not ASSET_DIRECTORY.is_dir():
        raise LauncherError(f"找不到启动器页面资源：{ASSET_DIRECTORY}")
    if not control_token:
        raise LauncherError("启动器控制令牌不能为空。")

    advertised_addresses = [
        dict(address) for address in (launcher_addresses or [])
    ]
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def protect_launcher(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            content_length = request.headers.get("content-length", "0")
            try:
                if int(content_length or "0") > REQUEST_BODY_LIMIT:
                    return JSONResponse(
                        {"error": "请求内容过大。"}, status_code=413
                    )
            except ValueError:
                return JSONResponse({"error": "Content-Length 无效。"}, status_code=400)
            supplied = request.headers.get("x-ftllm-launcher-token", "")
            if not hmac.compare_digest(supplied, control_token):
                return JSONResponse(
                    {"error": "控制令牌无效，请从 ftllm launch 重新打开页面。"},
                    status_code=403,
                )
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(LauncherError)
    async def launcher_error_handler(_request: Request, error: LauncherError):
        return JSONResponse({"error": str(error)}, status_code=400)

    @app.exception_handler(json.JSONDecodeError)
    async def invalid_json_handler(_request: Request, _error: json.JSONDecodeError):
        return JSONResponse({"error": "请求 JSON 格式无效。"}, status_code=400)

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

    @app.post("/api/profiles")
    async def save_profile(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise LauncherError("配置格式无效。")
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

    @app.get("/api/hardware")
    async def hardware(model_path: str = ""):
        return await run_in_threadpool(detect_hardware, model_path[:4096])

    @app.post("/api/shutdown")
    async def shutdown():
        runtime.request_shutdown()
        return {"ok": True}

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
        add_record("local", "本机地址", loopback)
        for value in candidates:
            try:
                address = ipaddress.ip_address(str(value).split("%", 1)[0])
            except ValueError:
                continue
            if address.version != version:
                continue
            if address.is_global:
                add_record("public", "公网地址", str(address))
            elif _is_usable_ip_address(address):
                add_record("lan", "局域网地址", str(address))
    else:
        try:
            bound_address = ipaddress.ip_address(normalized_host)
        except ValueError:
            if _is_loopback_host(normalized_host):
                add_record("local", "本机地址", "127.0.0.1")
            else:
                add_record("custom", "访问地址", normalized_host)
        else:
            if bound_address.is_loopback:
                add_record("local", "本机地址", str(bound_address))
            elif bound_address.is_global:
                add_record("public", "公网地址", str(bound_address))
            else:
                add_record("lan", "局域网地址", str(bound_address))

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
    runtime = LauncherRuntime(getattr(args, "config", ""))
    launcher_addresses = _launcher_access_addresses(host, port)
    try:
        app = create_launcher_app(runtime, control_token, launcher_addresses)
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
            print("FastLLM Launcher 访问地址（已包含控制令牌）：", flush=True)
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
