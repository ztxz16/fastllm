"""Install the companion wheel with pip into the current user's application data."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


RUNTIME_VERSION = "0.3.3"
PACKAGE_SPEC = f"ftllm-agent-runtime=={RUNTIME_VERSION}"
INSTALLATION_ID = f"ftllm-agent-runtime-{RUNTIME_VERSION}"
_MANAGED_MODULE = __package__ + "._installed_agent_runtime"
_IMPORT_LOCK = threading.RLock()


def installation_root() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    return base.expanduser().resolve() / "ftllm/agent-runtime" / INSTALLATION_ID


def managed_package() -> Path | None:
    root = installation_root()
    try:
        manifest = json.loads((root / "installation.json").read_text(encoding="utf-8"))
        package = root / "ftllm_agent_runtime"
        if manifest.get("id") == INSTALLATION_ID and all(
            (package / name).is_file() for name in (
                "__init__.py", "runtime.py", "bin/pi", "bin/rg", "bin/fd",
                "bin/package.json", "bin/photon_rs_bg.wasm", "extensions/project_tools.ts",
                "bin/theme/dark.json", "bin/theme/light.json", "bin/theme/theme-schema.json",
            )
        ):
            return package
    except (OSError, ValueError):
        pass
    return None


def load_pi_agent_runtime():
    with _IMPORT_LOCK:
        package = managed_package()
        if package is None:
            # Existing pip environments and complete portable bundles work as-is.
            return importlib.import_module("ftllm_agent_runtime").PiAgentRuntime
        module = sys.modules.get(_MANAGED_MODULE)
        if module is None or Path(module.__file__).parent != package:
            _forget_managed_module()
            spec = importlib.util.spec_from_file_location(_MANAGED_MODULE, package / "__init__.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[_MANAGED_MODULE] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                _forget_managed_module()
                raise
        return module.PiAgentRuntime


def _forget_managed_module():
    for name in tuple(sys.modules):
        if name == _MANAGED_MODULE or name.startswith(_MANAGED_MODULE + "."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _check_platform():
    if (sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}
            or sys.version_info < (3, 9)):
        raise RuntimeError("Pi installation requires Linux x86-64 and Python 3.9 or newer.")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(int(x) for x in version.split(".")) < (2, 17):
        raise RuntimeError("Pi installation requires glibc 2.17 or newer.")


def _runtime_info():
    runtime = load_pi_agent_runtime()(api_base="http://127.0.0.1:1/v1", model="runtime-check")
    info = runtime.info()
    path = str(runtime.binary.parent) + os.pathsep + os.environ.get("PATH", "")
    missing = [name for name, aliases in (("rg", ("rg",)), ("fd", ("fd", "fdfind")))
               if not any(shutil.which(alias, path=path) for alias in aliases)]
    if missing:
        raise RuntimeError("Missing Pi search tools: " + ", ".join(missing))
    shell = "powershell" if sys.platform == "win32" else "bash"
    if not shutil.which(shell, path=path):
        raise RuntimeError(f"{shell} is required for Pi workspace commands.")
    return info


def _run_command(command, cancelled, *, label, timeout=600, environment=None):
    if cancelled.is_set():
        raise RuntimeError("Pi installation was cancelled.")
    # File-backed output avoids a full stdout pipe blocking pip's download.
    # Only a bounded error tail is exposed to the control API.
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if cancelled.wait(0.1):
                    raise RuntimeError("Pi installation was cancelled.")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{label} timed out. Retry the installation.")
            if process.returncode:
                output.seek(max(0, output.tell() - 8000))
                details = output.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"{label} failed (exit {process.returncode}).\n{details}")
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()


def _install_wheel(target, cancelled):
    environment = os.environ.copy()
    # Keep pip's configured indexes, proxy, certificates and download cache.
    # Installation destinations must belong to this job, even with pip --user defaults.
    environment.pop("PIP_PREFIX", None)
    environment.pop("PIP_TARGET", None)
    environment["PIP_USER"] = "0"
    _run_command([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-input", "--progress-bar", "off", "--only-binary=:all:",
        "--target", str(target), PACKAGE_SPEC,
    ], cancelled, label="pip install", environment=environment)


def _verify_package(target, cancelled):
    # Verify in a fresh interpreter before making the installation discoverable.
    # This avoids an already-imported older companion masking broken files.
    script = """
import importlib.metadata, pathlib, subprocess, sys
sys.path.insert(0, sys.argv[1])
from ftllm_agent_runtime import PiAgentRuntime
assert importlib.metadata.version('ftllm-agent-runtime') == sys.argv[2]
runtime = PiAgentRuntime(api_base='http://127.0.0.1:1/v1', model='runtime-check')
assert runtime.info()['available']
assert runtime.binary.is_relative_to(pathlib.Path(sys.argv[1]))
for name in ('pi', 'rg', 'fd'):
    subprocess.run([str(runtime.binary.parent / name), '--version'], check=True, timeout=15)
"""
    _run_command([sys.executable, "-c", script, str(target), RUNTIME_VERSION],
                 cancelled, label="Pi runtime verification", timeout=60)


def install_runtime(progress, cancelled):
    import fcntl

    _check_platform()
    root = installation_root()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ".install.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another Launcher is installing Pi. Wait for it to finish.") from error
        with tempfile.TemporaryDirectory(prefix=".install-", dir=root.parent) as temporary:
            staging = Path(temporary) / "runtime"
            progress("pip", 0, 0)
            _install_wheel(staging, cancelled)
            progress("verify", 0, 0)
            _verify_package(staging, cancelled)
            (staging / "installation.json").write_text(
                json.dumps({"id": INSTALLATION_ID, "version": RUNTIME_VERSION}) + "\n", encoding="utf-8")
            if cancelled.is_set():
                raise RuntimeError("Pi installation was cancelled.")
            with _IMPORT_LOCK:
                backup = Path(temporary) / "previous"
                if root.exists():
                    root.rename(backup)
                try:
                    staging.rename(root)
                except BaseException:
                    if backup.exists():
                        backup.rename(root)
                    raise
                _forget_managed_module()


class AgentRuntimeInstaller:
    """The control API starts one background job and polls its bounded state."""

    def __init__(self, on_installed=None):
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._generation = 0
        self._thread = None
        self._on_installed = on_installed
        self._state = {"phase": "unchecked", "available": False, "supported": True,
                       "component": "", "error": ""}

    def state(self):
        with self._lock:
            if self._state["phase"] == "unchecked":
                self._launch(False)
            return dict(self._state)

    def start(self):
        with self._lock:
            if self._cancelled.is_set():
                raise RuntimeError("The Launcher is shutting down.")
            _check_platform()
            if self._state["phase"] not in {"installing", "ready"}:
                self._launch(True)
            return dict(self._state)

    def _launch(self, install):
        self._generation += 1
        generation = self._generation
        self._state.update(phase="installing" if install else "checking", error="",
                           component="")
        self._thread = threading.Thread(target=self._run, args=(generation, install), daemon=True)
        self._thread.start()

    def _update(self, generation, **values):
        with self._lock:
            if generation == self._generation and not self._cancelled.is_set():
                self._state.update(values)

    def _run(self, generation, install):
        platform_error = ""
        try:
            _check_platform()
        except Exception as error:
            platform_error = str(error)
        if install and platform_error:
            self._update(generation, phase="unsupported", supported=False, error=platform_error)
            return
        try:
            if install:
                install_runtime(lambda name, done, total: self._update(
                    generation, component=name), self._cancelled)
            info = _runtime_info()
            if install and self._on_installed and not self._cancelled.is_set():
                self._on_installed()
            # A portable runtime can already work on platforms for which the
            # public pip installer does not provide a compatible wheel.
            self._update(generation, phase="ready", available=True, supported=not platform_error,
                         piVersion=info["pi_version"], error="")
        except Exception as error:
            phase = "unsupported" if platform_error else "failed" if install else "missing"
            self._update(generation, phase=phase, available=False, supported=not platform_error,
                         error=platform_error or str(error))

    def close(self):
        self._cancelled.set()
