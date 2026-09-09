"""Lifecycle shared by the application's fixed native agent adapters."""

import os
import re
import shutil
import threading
import uuid
from pathlib import Path

from .harness_install import terminate_process
from .launcher_agent_install import AGENTS, install_runtime, runtime_command
from .launcher_agent_proxy import browser_address
from .launcher_agent_management import ManagedAgentRuntime
from .launcher_agent_models import reasoning_options, with_model_metadata


class AgentRuntime(ManagedAgentRuntime):
    def __init__(self, agent, directory=None):
        self.agent = agent
        self.name = AGENTS[agent]["name"]
        self.directory = Path(directory).absolute() if directory else Path.home() / ".fastllm" / agent
        self._lock = threading.RLock()
        self._operations = threading.RLock()
        self._cancelled = threading.Event()
        self._thread = self._process = None
        self._state = {"phase": "stopped", "sessionId": "", "url": "", "error": "", "stage": "", "epoch": ""}

    def _command(self):
        managed = runtime_command(self.directory / "runtime", self.agent)
        command = shutil.which(self.agent)
        return managed or ([command] if command else None)

    def state(self):
        with self._lock:
            return dict(self._state, installed=self._command() is not None,
                        workspace=str(self.directory / "workspace"), **self._management_state())

    def state_for_browser(self, browser_origin):
        return self.state()

    def start(self, service, api_key, bind_host, browser_origin, *, install=False):
        with self._operations, self._lock:
            if self.agent == "opencode":
                browser_address(browser_origin)
            if self._state["phase"] in {"installing", "upgrading", "removing", "starting", "running"}:
                return self.state_for_browser(browser_origin)
            if self._thread and self._thread.is_alive():
                raise RuntimeError(f"{self.name} is still stopping. Retry shortly.")
            self._cancelled = threading.Event()
            self._state = {"phase": "starting", "sessionId": service["sessionId"], "url": "", "error": "",
                           "stage": "", "epoch": uuid.uuid4().hex}
            self._thread = threading.Thread(target=self._run,
                args=(dict(service), api_key, bind_host, browser_origin, self._cancelled, install),
                name=f"ftllm-{self.agent}", daemon=True)
            self._thread.start()
            return self.state()

    def _update(self, cancelled, **values):
        with self._lock:
            if self._cancelled is cancelled and not cancelled.is_set():
                self._state.update(values)

    def _run(self, service, api_key, bind_host, browser_origin, cancelled, install):
        try:
            command = self._command()
            if not command:
                if not install:
                    raise RuntimeError(f"{self.name} is not installed. Click Install and open to install it.")
                install_runtime(self.directory, self.agent,
                    lambda stage, done, total: self._update(cancelled, phase="installing", stage=stage, done=done, total=total),
                    cancelled)
                command = self._command()
            if cancelled.is_set():
                return
            self._update(cancelled, phase="starting", stage="")
            service = with_model_metadata(service, api_key)
            efforts, default = reasoning_options(service)
            self._update(cancelled, modelName=service["modelName"], reasoningEfforts=efforts, defaultReasoningEffort=default)
            if cancelled.is_set():
                return
            (self.directory / "workspace").mkdir(parents=True, exist_ok=True)
            self._serve(command, service, api_key, bind_host, browser_origin, cancelled)
        except Exception as error:
            message = str(error)
            if api_key:
                message = message.replace(api_key, "[redacted]")
            message = re.sub(r"([?&]token=)[^\s)]+", r"\1[redacted]", message)
            self._update(cancelled, phase="failed", url="", error=message)

    def _environment(self, command, api_key):
        environment = os.environ.copy()
        # Runtime configuration is private to the child; do not inherit another
        # agent's endpoint, login, configuration overrides or automatic features.
        for key in list(environment):
            if key.startswith(("OPENCODE_", "CODEX_")) or key in {
                    "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}:
                environment.pop(key)
        environment["FTLLM_AGENT_API_KEY"] = api_key or "fastllm-local"
        environment["PATH"] = str(Path(command[0]).parent) + os.pathsep + environment.get("PATH", "")
        return environment

    def stop(self):
        with self._operations:
            with self._lock:
                self._cancelled.set()
                process, thread = self._process, self._thread
                self._state.update(phase="stopped", sessionId="", url="", error="", stage="")
            if process:
                terminate_process(process)
            if thread and thread is not threading.current_thread():
                thread.join(timeout=12)
            with self._lock:
                if not thread or not thread.is_alive():
                    self._process = None
            return self.state()
