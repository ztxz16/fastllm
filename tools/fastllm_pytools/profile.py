import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .deploy import (
    DeployConfig,
    _atomic_json_write,
    build_child_environment,
    build_fastllm_argv,
    clone_config,
    config_from_dict,
    effective_model_name,
    load_saved_configs,
    validate_config,
)


class ProfileError(RuntimeError):
    pass


class ProfileLockError(ProfileError):
    pass


def get_profile_state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))
    return Path(os.path.expandvars(os.path.expanduser(base))) / "fastllm" / "profiles"

def _redact_argv(argv: Iterable[str]) -> List[str]:
    output = []
    hide_next = False
    for item in argv:
        if hide_next:
            output.append("<redacted>")
            hide_next = False
            continue
        output.append(item)
        hide_next = item in ("--api_key", "--api-key")
    return output


def _public_value(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ("api_key", "env_vars") and item:
                result[key] = "<redacted>"
            elif key == "argv" and isinstance(item, list):
                result[key] = _redact_argv(item)
            else:
                result[key] = _public_value(item)
        return result
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    return value


class ProfileManager:
    def __init__(self, config_path: Optional[str] = None,
                 state_dir: Optional[str] = None):
        self.config_path = config_path
        self.state_dir = Path(state_dir) if state_dir else get_profile_state_dir()
        self.state_path = self.state_dir / "runtime.json"
        self.lock_path = self.state_dir / "profile.lock"
        self.log_dir = self.state_dir / "logs"
        self._lock_handle = None

    def _ensure_dirs(self) -> None:
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self.state_dir, 0o700)
            os.chmod(self.log_dir, 0o700)
        except OSError:
            pass

    @contextlib.contextmanager
    def lock(self, blocking: bool = False):
        self._ensure_dirs()
        handle = open(self.lock_path, "a+", encoding="utf-8")
        os.chmod(self.lock_path, 0o600)
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            handle.close()
            raise ProfileLockError("另一个 profile 操作正在进行") from exc
        self._lock_handle = handle
        try:
            yield
        finally:
            self._lock_handle = None
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def load_state(self) -> dict:
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {"status": "stopped", "active": None}
        return state if isinstance(state, dict) else {"status": "stopped", "active": None}

    def save_state(self, state: dict) -> None:
        self._ensure_dirs()
        _atomic_json_write(str(self.state_path), state)

    def profiles(self) -> List[DeployConfig]:
        return [item for item in load_saved_configs(self.config_path)
                if item.command == "server" and item.name.strip()]

    def profile_map(self) -> Dict[str, DeployConfig]:
        result: Dict[str, DeployConfig] = {}
        duplicates = set()
        for item in self.profiles():
            name = item.name.strip()
            if name in result:
                duplicates.add(name)
            result[name] = item
        if duplicates:
            raise ProfileError("profile 名称重复: " + ", ".join(sorted(duplicates)))
        return result

    def get_profile(self, name: str) -> DeployConfig:
        try:
            return self.profile_map()[name]
        except KeyError as exc:
            raise ProfileError(f"profile 不存在: {name}") from exc

    def validate(self, name: Optional[str] = None) -> Dict[str, List[str]]:
        profiles = self.profile_map()
        names = [name] if name else sorted(profiles)
        result = {}
        for item_name in names:
            if item_name not in profiles:
                result[item_name] = [f"profile 不存在: {item_name}"]
                continue
            result[item_name] = validate_config(profiles[item_name])
        return result

    @staticmethod
    def _proc_start_time(pid: int) -> Optional[str]:
        try:
            raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        except (OSError, ValueError):
            return None
        end = raw.rfind(")")
        if end < 0:
            return None
        fields = raw[end + 2:].split()
        return fields[19] if len(fields) > 19 else None

    def process_matches(self, state: dict) -> bool:
        pid = state.get("pid")
        expected = state.get("proc_start_time")
        if not isinstance(pid, int) or pid <= 0 or not expected:
            return False
        return self._proc_start_time(pid) == str(expected)

    def status(self) -> dict:
        state = self.load_state()
        state["running"] = self.process_matches(state)
        if state.get("status") == "running" and not state["running"]:
            state["status"] = "stale"
        return state

    def _log_path(self, name: str) -> Path:
        self._ensure_dirs()
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self.log_dir / f"{safe}-{stamp}-{time.time_ns() % 1000000000:09d}.log"

    @staticmethod
    def _resolve_launch_argv(argv: List[str], env: Dict[str, str]) -> List[str]:
        executable = shutil.which(argv[0], path=env.get("PATH"))
        if executable:
            env.pop("FASTLLM_PROFILE_PYTHON", None)
            return [executable, *argv[1:]]
        python = env.pop("FASTLLM_PROFILE_PYTHON", "").strip() or sys.executable
        if not python or not os.path.isfile(python) or not os.access(python, os.X_OK):
            raise ProfileError("找不到 ftllm 命令，且 Python profile 启动器不可执行")
        return [python, "-m", "ftllm.cli", *argv[1:]]

    def _spawn(self, config: DeployConfig, log_path: Path):
        argv = build_fastllm_argv(config)
        if "--startup-progress" not in argv:
            argv.extend(["--startup-progress", "ndjson"])
        env = build_child_environment(config)
        argv = self._resolve_launch_argv(argv, env)
        handle = open(log_path, "ab", buffering=0)
        os.chmod(log_path, 0o600)
        try:
            process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=handle,
                stderr=subprocess.STDOUT, env=env, start_new_session=True,
                close_fds=True)
        except BaseException:
            handle.close()
            raise
        process._fastllm_log_handle = handle
        return process, argv

    @staticmethod
    def _close_process_log(process) -> None:
        handle = getattr(process, "_fastllm_log_handle", None)
        if handle is not None:
            handle.close()
            process._fastllm_log_handle = None

    @staticmethod
    def public_state(state: dict) -> dict:
        return _public_value(state)

    @staticmethod
    def _http_json(url: str, timeout: float) -> dict:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ProfileError(f"readiness HTTP {response.status}: {url}")
            return json.loads(response.read().decode("utf-8"))

    def _probe_ready(self, config: DeployConfig) -> bool:
        host = str(config.host).strip()
        if host in ("", "0.0.0.0", "::"):
            host = "127.0.0.1"
        base = f"http://{host}:{int(config.port)}"
        try:
            health = self._http_json(base + "/health", 2.0)
            models = self._http_json(base + "/v1/models", 2.0)
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError, ProfileError):
            return False
        if not health.get("ready"):
            return False
        expected = effective_model_name(config)
        entries = models.get("data", []) if isinstance(models, dict) else []
        return expected in {item.get("id") for item in entries if isinstance(item, dict)}

    def _wait_ready(self, process, config: DeployConfig, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise ProfileError(f"profile 进程提前退出，code={return_code}")
            if self._probe_ready(config):
                return
            time.sleep(0.25)
        raise ProfileError(f"profile 启动超时（{timeout:g}s）")

    def _stop_process(self, state: dict, grace: float = 30.0) -> None:
        if not self.process_matches(state):
            return
        pid = int(state["pid"])
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + max(0.0, grace)
        while time.monotonic() < deadline:
            if self._proc_start_time(pid) != str(state.get("proc_start_time")):
                return
            time.sleep(0.1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _start_snapshot(self, name: str, config: DeployConfig) -> dict:
        errors = validate_config(config)
        if errors:
            raise ProfileError(f"profile {name} 校验失败: " + "; ".join(errors))
        log_path = self._log_path(name)
        process, argv = self._spawn(config, log_path)
        snapshot = {
            "name": name,
            "config": asdict(clone_config(config)),
            "pid": int(process.pid),
            "proc_start_time": self._proc_start_time(process.pid),
            "log_path": str(log_path),
            "argv": _redact_argv(argv),
            "started_at": time.time(),
        }
        if not snapshot["proc_start_time"]:
            try:
                process.terminate()
                process.wait(timeout=2)
            except BaseException:
                try:
                    process.kill()
                except BaseException:
                    pass
            self._close_process_log(process)
            raise ProfileError("无法读取新进程身份")
        try:
            self._wait_ready(process, config, float(config.startup_timeout))
        except BaseException as exc:
            self._stop_process(snapshot, grace=1.0)
            self._close_process_log(process)
            try:
                setattr(exc, "log_path", str(log_path))
            except BaseException:
                pass
            raise
        self._close_process_log(process)
        return snapshot

    def start(self, name: str) -> dict:
        with self.lock():
            state = self.load_state()
            if self.process_matches(state):
                raise ProfileError(f"已有 profile 运行: {state.get('active')}")
            config = self.get_profile(name)
            pending = {
                "status": "starting", "active": None,
                "target": {"name": name, "config": asdict(clone_config(config))},
            }
            self.save_state(pending)
            try:
                snapshot = self._start_snapshot(name, config)
            except BaseException as exc:
                failed = {**pending, "status": "failed", "error": str(exc)}
                self.save_state(failed)
                raise
            result = {"status": "running", "active": name, **snapshot}
            self.save_state(result)
            return result

    def stop(self, grace: float = 30.0) -> dict:
        with self.lock():
            state = self.load_state()
            self._stop_process(state, grace=grace)
            result = {"status": "stopped", "active": None,
                      "stopped_at": time.time(), "previous": state}
            self.save_state(result)
            return result

    def switch(self, name: str, grace: float = 30.0) -> dict:
        with self.lock():
            target_config = self.get_profile(name)
            errors = validate_config(target_config)
            if errors:
                raise ProfileError(f"profile {name} 校验失败: " + "; ".join(errors))
            previous_state = self.load_state()
            previous_snapshot = None
            if self.process_matches(previous_state):
                previous_snapshot = {
                    "name": previous_state.get("active") or previous_state.get("name"),
                    "config": previous_state.get("config"),
                    "log_path": previous_state.get("log_path"),
                }
            transition = {
                "status": "switching", "active": None,
                "previous": previous_snapshot,
                "target": {"name": name, "config": asdict(clone_config(target_config))},
                "started_at": time.time(),
            }
            self.save_state(transition)
            if previous_snapshot:
                self._stop_process(previous_state, grace=grace)
            try:
                target = self._start_snapshot(name, target_config)
            except BaseException as target_exc:
                transition["target_error"] = str(target_exc)
                transition["target_log_path"] = getattr(target_exc, "log_path", None)
                if previous_snapshot and previous_snapshot.get("config"):
                    previous_config = config_from_dict(previous_snapshot["config"])
                    previous_name = previous_snapshot.get("name") or "previous"
                    try:
                        restored = self._start_snapshot(previous_name, previous_config)
                    except BaseException as rollback_exc:
                        transition["rollback_log_path"] = getattr(
                            rollback_exc, "log_path", None)
                        failed = {
                            **transition, "status": "failed",
                            "rollback_error": str(rollback_exc),
                        }
                        self.save_state(failed)
                        raise ProfileError(
                            f"目标启动失败且回滚失败: {target_exc}; {rollback_exc}") from rollback_exc
                    rolled_back = {
                        **transition, "status": "rolled_back",
                        "active": previous_name, "rollback": restored,
                        **restored,
                    }
                    self.save_state(rolled_back)
                    raise ProfileError(f"目标启动失败，已回滚: {target_exc}") from target_exc
                failed = {**transition, "status": "failed", "error": str(target_exc)}
                self.save_state(failed)
                raise
            result = {
                **transition, "status": "running", "active": name,
                "completed_at": time.time(), **target,
            }
            self.save_state(result)
            return result

    def logs(self, name: Optional[str] = None, lines: int = 100) -> str:
        state = self.load_state()
        path = state.get("log_path")
        if name and name != state.get("active"):
            candidates = sorted(self.log_dir.glob(f"{name}-*.log"))
            path = str(candidates[-1]) if candidates else None
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-max(1, lines):])


def add_profile_subparsers(parent) -> None:
    actions = parent.add_subparsers(dest="profile_action", required=True)
    actions.add_parser("list", help="列出命名 profile")
    show = actions.add_parser("show", help="显示 profile")
    show.add_argument("name")
    validate = actions.add_parser("validate", help="校验 profile")
    validate.add_argument("name", nargs="?")
    actions.add_parser("status", help="显示运行状态")
    start = actions.add_parser("start", help="启动 profile")
    start.add_argument("name")
    stop = actions.add_parser("stop", help="停止 active profile")
    stop.add_argument("--grace", type=float, default=30.0)
    switch = actions.add_parser("switch", help="切换 profile 并自动回滚")
    switch.add_argument("name")
    switch.add_argument("--grace", type=float, default=30.0)
    logs = actions.add_parser("logs", help="查看日志")
    logs.add_argument("name", nargs="?")
    logs.add_argument("-n", "--lines", type=int, default=100)


def profile_main(args, manager: Optional[ProfileManager] = None) -> int:
    manager = manager or ProfileManager()
    action = args.profile_action
    if action == "list":
        state = manager.status()
        for config in manager.profiles():
            marker = "*" if state.get("active") == config.name and state.get("running") else " "
            print(f"{marker} {config.name}")
        return 0
    if action == "show":
        print(json.dumps(_public_value(asdict(manager.get_profile(args.name))), ensure_ascii=False, indent=2))
        return 0
    if action == "validate":
        result = manager.validate(args.name)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if any(result.values()) else 0
    if action == "status":
        print(json.dumps(manager.public_state(manager.status()), ensure_ascii=False, indent=2))
        return 0
    if action == "start":
        print(json.dumps(manager.public_state(manager.start(args.name)), ensure_ascii=False, indent=2))
        return 0
    if action == "stop":
        print(json.dumps(manager.public_state(manager.stop(args.grace)), ensure_ascii=False, indent=2))
        return 0
    if action == "switch":
        print(json.dumps(manager.public_state(manager.switch(args.name, args.grace)), ensure_ascii=False, indent=2))
        return 0
    if action == "logs":
        print(manager.logs(args.name, args.lines), end="")
        return 0
    raise ProfileError(f"未知 profile 操作: {action}")
