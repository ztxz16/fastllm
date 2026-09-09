"""OpenCode's original web app, connected to the active FastLLM model."""

import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

from .harness_install import terminate_process
from .launcher_agent_runtime import AgentRuntime
from .launcher_agent_proxy import AgentProxy
from .launcher_agent_models import REASONING_EFFORTS, reasoning_options


class OpenCodeRuntime(AgentRuntime):
    def __init__(self, directory=None):
        super().__init__("opencode", directory)
        self._proxy = None

    def state_for_browser(self, browser_origin):
        with self._lock:
            state = self.state()
            if state["phase"] == "running" and self._proxy:
                state["url"] = self._proxy.url_for(browser_origin)
            return state

    @staticmethod
    def configuration(service):
        context = service.get("contextWindowTokens") or 8192
        model = service["modelName"]
        efforts, default = reasoning_options(service)
        return {"$schema": "https://opencode.ai/config.json", "autoupdate": False,
                "share": "disabled", "enabled_providers": ["fastllm"], "model": "fastllm/" + model,
                "small_model": "fastllm/" + model, "compaction": {"auto": False, "prune": False},
                "permission": {"external_directory": "ask"},
                "provider": {"fastllm": {"npm": "@ai-sdk/openai-compatible", "name": "FastLLM",
                    "options": {"baseURL": service["endpoint"].rstrip("/") + "/v1",
                                "apiKey": "{env:FTLLM_AGENT_API_KEY}"},
                    "models": {model: {"name": model, "tool_call": True, "reasoning": bool(efforts),
                                       "options": {"reasoningEffort": default} if default else {},
                                       # OpenCode merges its built-in variants with this map.
                                       # Explicitly disable generic levels the model rejects.
                                       "variants": {effort: ({"reasoningEffort": effort} if effort in efforts
                                                              else {"disabled": True})
                                                    for effort in REASONING_EFFORTS},
                                       "limit": {"context": context, "output": min(8192, context // 2)}}}}}}

    def _serve(self, command, service, api_key, bind_host, browser_origin, cancelled):
        environment = self._environment(command, api_key)
        for kind in ("config", "data", "cache", "state"):
            path = self.directory / kind
            path.mkdir(parents=True, exist_ok=True)
            environment[f"XDG_{kind.upper()}_HOME"] = str(path)
        environment.update({"OPENCODE_CONFIG_CONTENT": json.dumps(self.configuration(service)),
            "OPENCODE_DISABLE_AUTOUPDATE": "true", "OPENCODE_DISABLE_MODELS_FETCH": "true",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "true", "OPENCODE_DISABLE_AUTOCOMPACT": "true",
            "OPENCODE_DISABLE_PRUNE": "true", "OPENCODE_SERVER_USERNAME": "opencode",
            "OPENCODE_SERVER_PASSWORD": secrets.token_urlsafe(32)})
        password = environment["OPENCODE_SERVER_PASSWORD"]
        with tempfile.TemporaryDirectory(prefix="launch-", dir=self.directory) as temporary:
            log = Path(temporary) / "opencode.log"
            with log.open("wb") as output, log.open("rb") as reader:
                with self._lock:
                    if cancelled.is_set():
                        return
                    self._process = process = subprocess.Popen(
                        [*command, "serve", "--pure", "--hostname", "127.0.0.1", "--port", "0"],
                        cwd=self.directory / "workspace", env=environment, stdin=subprocess.DEVNULL,
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
                proxy, tail, deadline = None, "", time.monotonic() + 120
                try:
                    while not cancelled.wait(.1):
                        tail = (tail + reader.read(65536).decode("utf-8", errors="replace"))[-16000:]
                        if process.poll() is not None:
                            raise RuntimeError("OpenCode exited.\n" + tail[-4000:].replace(password, "[redacted]"))
                        if proxy is None:
                            match = re.search(r"opencode server listening on (http://127\.0\.0\.1:\d+)", tail)
                            if match:
                                proxy = AgentProxy(match[1], password, bind_host, browser_origin)
                                proxy.start(cancelled)
                                with self._lock:
                                    self._proxy = proxy
                                    self._update(cancelled, phase="running", url=proxy.url)
                            elif time.monotonic() > deadline:
                                raise RuntimeError("OpenCode startup timed out. Retry opening it.")
                finally:
                    with self._lock:
                        if self._proxy is proxy:
                            self._proxy = None
                    if proxy:
                        proxy.stop()
                    terminate_process(process)
