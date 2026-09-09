"""Claude Code Agent SDK adapter, using the active FastLLM Anthropic endpoint."""

import json

from .launcher_agent_install import runtime_command
from .launcher_agent_session import SessionRuntime
from .launcher_agent_models import reasoning_options


class ClaudeRuntime(SessionRuntime):
    def __init__(self, directory=None):
        super().__init__("claude", directory)

    def _command(self):
        # The SDK bundles Claude Code; a standalone system CLI cannot replace
        # the bridge's SDK dependency. The user's ~/.claude stays separate.
        return runtime_command(self.directory / "runtime", self.agent)

    def _prepare_rpc(self, method, params):
        if method == "turn/start" and "effort" not in params:
            default = reasoning_options(self._service)[1]
            if default:
                params["effort"] = default
        return params

    def _serve(self, command, service, api_key, bind_host, browser_origin, cancelled):
        environment = self._environment(command, api_key)
        for key in list(environment):
            if key.startswith(("ANTHROPIC_", "CLAUDE_", "CLAUDECODE")):
                environment.pop(key)
        data = self.directory / "home"
        data.mkdir(parents=True, exist_ok=True)
        environment.update({
            "CLAUDE_CONFIG_DIR": str(data),
            "ANTHROPIC_BASE_URL": service["endpoint"].rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": api_key or "fastllm-local",
            "ANTHROPIC_MODEL": service["modelName"],
            "ANTHROPIC_DEFAULT_OPUS_MODEL": service["modelName"],
            "ANTHROPIC_DEFAULT_SONNET_MODEL": service["modelName"],
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": service["modelName"],
            "CLAUDE_CODE_SUBAGENT_MODEL": service["modelName"],
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(min(8192, (service.get("contextWindowTokens") or 8192) // 2)),
            "FTLLM_CLAUDE_SERVICE": json.dumps(service),
        })
        self._launch(command, service, environment, cancelled)
