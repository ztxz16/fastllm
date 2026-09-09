"""Codex app-server adapter for the shared native session interface."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .launcher_agent_session import SessionRuntime
from .launcher_agent_models import reasoning_options
from .openai_server.fastllm_model import FastLLmModel


class CodexRuntime(SessionRuntime):
    def __init__(self, directory=None):
        super().__init__("codex", directory)

    def _configure(self, service, temporary):
        context = service.get("contextWindowTokens") or 8192
        metadata = FastLLmModel(service["modelName"], SimpleNamespace(get_max_input_len=lambda: context)).response
        efforts, default = reasoning_options(service)
        # app-server's private catalog uses presets, whereas /v1/models also
        # exposes string lists for OpenAI-compatible clients.
        metadata["models"][0].update(
            supported_reasoning_levels=[{"effort": effort, "description": effort} for effort in efforts],
            default_reasoning_level=default)
        metadata["models"][0]["auto_compact_token_limit"] = 2**60
        catalog = Path(temporary) / "models.json"
        catalog.write_text(json.dumps(metadata), encoding="utf-8")
        config = {
            "model": service["modelName"], "model_provider": "fastllm",
            "model_context_window": context, "model_auto_compact_token_limit": 2**60,
            "model_catalog_json": str(catalog), "approval_policy": "on-request",
            "approvals_reviewer": "user", "sandbox_mode": "workspace-write", "web_search": "disabled",
            "analytics.enabled": False, "check_for_update_on_startup": False,
            "model_providers.fastllm.name": "FastLLM",
            "model_providers.fastllm.base_url": service["endpoint"].rstrip("/") + "/v1",
            "model_providers.fastllm.wire_api": "responses",
            "model_providers.fastllm.env_key": "FTLLM_AGENT_API_KEY",
            "model_providers.fastllm.requires_openai_auth": False,
            "model_providers.fastllm.request_max_retries": 0,
            "model_providers.fastllm.stream_max_retries": 0,
        }
        arguments = []
        if default:
            config["model_reasoning_effort"] = default
        for key, value in config.items():
            arguments.extend(["-c", key + "=" + json.dumps(value)])
        return arguments

    def _serve(self, command, service, api_key, bind_host, browser_origin, cancelled):
        environment = self._environment(command, api_key)
        data = self.directory / "home"
        data.mkdir(parents=True, exist_ok=True)
        environment["CODEX_HOME"] = str(data)
        with tempfile.TemporaryDirectory(prefix="launch-", dir=self.directory) as temporary:
            arguments = self._configure(service, temporary)
            self._launch([*command, "app-server", *arguments], service, environment, cancelled)

    def _prepare_rpc(self, method, params):
        if method == "thread/list":
            params.update(limit=100, modelProviders=["fastllm"], sortKey="updated_at")
        if method in {"thread/start", "thread/resume"}:
            params.update(model=self._service["modelName"], modelProvider="fastllm",
                          approvalPolicy="on-request", approvalsReviewer="user", sandbox="workspace-write")
        if method == "thread/read":
            params["includeTurns"] = True
        if method == "turn/start":
            params["input"] = [{"type":"text", "text":params.pop("text"), "text_elements":[]}]
        return params
