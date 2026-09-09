"""Fixed DeepSeek Harness integration, running outside the Launcher process."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .harness_install import install_runtime, runtime_command, terminate_process
from .launcher_agent_proxy import browser_address, create_harness_proxy
from .launcher_agent_management import ManagedAgentRuntime
from .launcher_agent_models import reasoning_options, with_model_metadata


def harness_directory():
    return Path.home() / ".fastllm" / "deepseek-harness"


class HarnessRuntime(ManagedAgentRuntime):
    def __init__(self, directory=None):
        self.agent = "harness"
        self.directory = Path(directory) if directory else harness_directory()
        self._lock = threading.RLock()
        self._operations = threading.RLock()
        self._cancelled = threading.Event()
        self._thread = None
        self._process = None
        self._proxy = None
        self._state = {"phase": "stopped", "sessionId": "", "url": "", "error": "", "stage": ""}

    def state(self):
        with self._lock:
            return dict(self._state, installed=self._command() is not None, **self._management_state())

    def state_for_browser(self, browser_origin):
        with self._lock:
            state = self.state()
            if state["phase"] == "running" and self._proxy:
                state["url"] = self._proxy.url_for(browser_origin)
            return state

    def start(self, service, api_key, bind_host, browser_origin, *, install=False):
        with self._operations, self._lock:
            browser_address(browser_origin, "DeepSeek Harness")
            if self._state["phase"] in {"installing", "upgrading", "removing", "starting", "running"}:
                return self.state_for_browser(browser_origin)
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Harness is still stopping. Retry shortly.")
            self._cancelled = threading.Event()
            self._state = {"phase": "starting", "sessionId": service["sessionId"], "url": "", "error": "", "stage": ""}
            self._thread = threading.Thread(target=self._run,
                args=(dict(service), api_key, bind_host, browser_origin, self._cancelled, install),
                name="ftllm-harness", daemon=True)
            self._thread.start()
            return dict(self._state)

    def _command(self):
        managed = runtime_command(self.directory / "runtime")
        if managed:
            return managed
        command = shutil.which("dsh")
        if command:
            return [command]
        return None

    @staticmethod
    def _patch(service, bind_host):
        context = service.get("contextWindowTokens") or 8192
        efforts, default = reasoning_options(service)
        # Harness names the disabled level "off"; FastLLM expects "none" on
        # the wire so it overrides a service with thinking enabled by default.
        default = "off" if default == "none" else default
        return [
            {"id": "webserver", "config": {"host": bind_host, "port": 0}},
            {"id": "agent-default-model", "config": {"provider": "fastllm", "model": service["modelName"],
                **({"reasoningEffort": default} if default else {})}},
            # Embedded browsers choose server directories inside the page,
            # including when Launcher itself runs on a desktop machine.
            {"id": "directory-picker", "disabled": True},
            {"insert": [
                {"id": "launcher-directory-picker", "name": "@deepseek-ai/dsh-host-directory-picker-browse"},
                {"id": "launcher-directory-picker-ui", "name": "@deepseek-ai/dsh-client-ui-directory-picker-browse"},
            ]},
            {"id": "llm-pi-ai", "config": {"providers": {"fastllm": {
                "displayName": "FastLLM", "api": "openai-completions",
                "baseURL": service["endpoint"].rstrip("/") + "/v1",
                "apiKeyEnv": "FTLLM_HARNESS_API_KEY",
                **({"reasoning": default} if default else {}),
                "models": [{"id": service["modelName"], "name": service["modelName"],
                            "reasoningEfforts": {"off" if effort == "none" else effort: effort
                                                 for effort in efforts} if efforts else False,
                            "contextWindow": context, "maxTokens": min(8192, context // 2)}],
                "compat": {"supportsStore": False, "supportsReasoningEffort": True, "thinkingFormat": "openai"},
            }}}},
            {"id": "session-log-deepseek", "disabled": True},
            {"id": "session-telemetry-otel", "disabled": True},
        ]

    def _run(self, service, api_key, bind_host, browser_origin, cancelled, install):
        try:
            command = self._command()
            if command is None:
                if not install:
                    raise RuntimeError("Harness is not installed. Click Install and open Harness to install it.")
                def progress(stage, done, total):
                    with self._lock:
                        if not cancelled.is_set():
                            self._state.update(phase="installing", stage=stage, done=done, total=total)
                install_runtime(self.directory, progress, cancelled)
                command = self._command()
            with self._lock:
                if cancelled.is_set():
                    return
                self._state.update(phase="starting", stage="")
            service = with_model_metadata(service, api_key)
            self.directory.mkdir(parents=True, exist_ok=True)
            workspace = self.directory / "workspace"
            workspace.mkdir(exist_ok=True)
            environment = os.environ.copy()
            environment["DSH_HOME"] = str(self.directory / "home")
            environment["FTLLM_HARNESS_API_KEY"] = api_key or "fastllm-local"
            environment["PATH"] = str(Path(command[0]).parent) + os.pathsep + environment.get("PATH", "")
            # Configuration and launch tokens belong to this process, while
            # the Harness home and workspace survive stop/restart.
            with tempfile.TemporaryDirectory(prefix="launch-", dir=self.directory) as temporary:
                patch = Path(temporary) / "launcher.patch.yml"
                patch.write_text(json.dumps(self._patch(service, "127.0.0.1")), encoding="utf-8")
                log = Path(temporary) / "harness.log"
                with log.open("wb") as output, log.open("rb") as reader:
                    with self._lock:
                        if cancelled.is_set():
                            return
                        process = subprocess.Popen(
                            [*command, "--profile", "web", "--patch", str(patch), "--no-open"],
                            cwd=workspace, env=environment, stdin=subprocess.DEVNULL,
                            stdout=output, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
                        self._process = process
                    try:
                        self._watch(process, reader, api_key, bind_host, browser_origin, cancelled)
                    finally:
                        terminate_process(process)
        except Exception as error:
            diagnostic = re.sub(r"([?&]token=)[^\s)]+", r"\1[redacted]", str(error))
            if api_key:
                diagnostic = diagnostic.replace(api_key, "[redacted]")
            with self._lock:
                if not cancelled.is_set():
                    self._state.update(phase="failed", url="", error=diagnostic)

    def _watch(self, process, reader, api_key, bind_host, browser_origin, cancelled):
        deadline = time.monotonic() + 120
        tail, proxy = "", None
        try:
            while not cancelled.wait(.1):
                chunk = reader.read(65536).decode("utf-8", errors="replace")
                tail = (tail + chunk)[-16000:]
                if process.poll() is not None:
                    diagnostic = re.sub(r"([?&]token=)[^\s)]+", r"\1[redacted]", tail)
                    if api_key:
                        diagnostic = diagnostic.replace(api_key, "[redacted]")
                    raise RuntimeError("DeepSeek Harness exited.\n" + diagnostic[-4000:])
                if proxy is None:
                    match = re.search(r"dsh web: (http://[^\s)]+)", tail)
                    if match:
                        proxy = create_harness_proxy(match[1], bind_host, browser_origin)
                        proxy.start(cancelled)
                        with self._lock:
                            if not cancelled.is_set():
                                self._proxy = proxy
                                self._state.update(phase="running", url=proxy.url)
                    elif time.monotonic() > deadline:
                        raise RuntimeError("DeepSeek Harness startup timed out. Retry opening it.")
        finally:
            with self._lock:
                if self._proxy is proxy:
                    self._proxy = None
            if proxy:
                # Close native SSE streams before asking the proxy to drain
                # its requests, so a normal stop does not force-cancel them.
                try:
                    terminate_process(process)
                finally:
                    proxy.stop()

    def stop(self):
        with self._operations:
            with self._lock:
                self._cancelled.set()
                thread = self._thread
                self._state.update(phase="stopped", sessionId="", url="", error="", stage="")
            if thread and thread is not threading.current_thread():
                thread.join(timeout=12)
            with self._lock:
                self._process = None
            return self.state()
