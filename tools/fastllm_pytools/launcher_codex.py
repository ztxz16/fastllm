"""Codex app-server stdio bridge. Only the fixed UI operations are exposed."""

import json
import os
import queue
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from .harness_install import terminate_process
from .launcher_agent_runtime import AgentRuntime
from .launcher_agent_models import reasoning_options
from .openai_server.fastllm_model import FastLLmModel


class CodexConnection:
    def __init__(self, process, on_event):
        self.process, self.on_event = process, on_event
        self._lock, self._write_lock = threading.Lock(), threading.Lock()
        self._pending, self._next_id, self._closed = {}, 0, False
        self._event_cursor = 0
        self.reader = threading.Thread(target=self._read, name="ftllm-codex-rpc", daemon=True)
        self.reader.start()

    def send(self, message):
        with self._write_lock:
            try:
                self.process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
                self.process.stdin.flush()
            except (OSError, ValueError) as error:
                raise RuntimeError("Codex is disconnected. Reopen it from Launcher.") from error

    def call(self, method, params, timeout=30):
        with self._lock:
            if self._closed:
                raise RuntimeError("Codex is disconnected. Reopen it from Launcher.")
            self._next_id += 1
            request_id, response = self._next_id, queue.Queue(maxsize=1)
            self._pending[request_id] = response
        try:
            self.send({"id": request_id, "method": method, "params": params})
            try:
                message = response.get(timeout=timeout)
            except queue.Empty as error:
                raise RuntimeError(f"Codex {method} timed out. Check the session before retrying.") from error
            if "error" in message:
                raise RuntimeError(message["error"].get("message", "Codex request failed."))
            result = message.get("result", {})
            if isinstance(result, dict):
                result["_eventCursor"] = message.get("_eventCursor", 0)
            return result
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def _read(self):
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if "method" in message:
                    self._event_cursor = self.on_event(message) or self._event_cursor
                elif "id" in message:
                    message["_eventCursor"] = self._event_cursor
                    with self._lock:
                        pending = self._pending.get(message["id"])
                        if pending and pending.empty():
                            pending.put_nowait(message)
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                self._closed = True
                for pending in self._pending.values():
                    if pending.empty():
                        pending.put_nowait({"error": {"message": "Codex app-server disconnected."}})

    def close(self):
        terminate_process(self.process)
        self.reader.join(timeout=3)
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()


class CodexRuntime(AgentRuntime):
    METHODS = {
        "thread/list": {"cursor", "searchTerm", "archived"},
        "thread/start": {"cwd"}, "thread/resume": {"threadId"},
        "thread/read": {"threadId"}, "thread/name/set": {"threadId", "name"},
        "thread/archive": {"threadId"}, "turn/start": {"threadId", "text", "effort"},
        "turn/interrupt": {"threadId", "turnId"},
    }
    APPROVALS = {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}

    def __init__(self, directory=None):
        super().__init__("codex", directory)
        self._connection = None
        self._service = {}
        # This is a bounded delivery buffer, not conversation history. Clients
        # that miss events reload the full persisted thread from app-server.
        self._events, self._sequence = deque(maxlen=4096), 0
        self._requests, self._turns = {}, {}

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
            with (Path(temporary) / "codex.log").open("w+b") as log:
                with self._lock:
                    if cancelled.is_set():
                        return
                    self._events.clear(); self._requests.clear(); self._turns.clear(); self._sequence = 0
                    self._service = service
                    self._process = process = subprocess.Popen([*command, "app-server", *arguments],
                        cwd=self.directory / "workspace", env=environment, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=log, start_new_session=os.name != "nt")
                connection = CodexConnection(process, lambda message: self._event(message, cancelled))
                with self._lock:
                    self._connection = connection
                try:
                    connection.call("initialize", {"clientInfo": {"name": "fastllm_launcher",
                        "title": "FastLLM Launcher", "version": "1.0"}, "capabilities": {"experimentalApi": True}}, timeout=30)
                    connection.send({"method": "initialized", "params": {}})
                    self._update(cancelled, phase="running")
                    while not cancelled.wait(.1):
                        if process.poll() is not None or not connection.reader.is_alive():
                            log.seek(0, 2)
                            size = log.tell(); log.seek(max(0, size - 4000))
                            raise RuntimeError("Codex exited.\n" + log.read().decode("utf-8", errors="replace"))
                finally:
                    connection.close()
                    with self._lock:
                        if self._connection is connection:
                            self._connection = None
                            self._requests.clear(); self._turns.clear()

    def _event(self, message, cancelled):
        with self._lock:
            if cancelled.is_set() or self._cancelled is not cancelled:
                return
            method, params = message["method"], message.get("params") or {}
            if method.startswith("codex/event/"):
                return  # Duplicate legacy stream; the UI consumes v2 events.
            self._sequence += 1
            self._events.append({"sequence": self._sequence, **message})
            if "id" in message:
                self._requests[str(message["id"])] = message
            if method == "serverRequest/resolved":
                self._requests.pop(str(params.get("requestId")), None)
            if method == "turn/started":
                self._turns[params.get("threadId")] = params["turn"]["id"]
            elif method == "turn/completed":
                self._turns.pop(params.get("threadId"), None)
                self._requests = {key: value for key, value in self._requests.items()
                                  if value.get("params", {}).get("turnId") != params["turn"]["id"]}
            return self._sequence

    def events(self, after=0, epoch=""):
        with self._lock:
            reset = epoch != self._state["epoch"] or (self._events and after < self._events[0]["sequence"] - 1)
            return {"epoch": self._state["epoch"], "cursor": self._sequence, "reset": bool(reset),
                    "events": [item for item in self._events if not reset and item["sequence"] > after],
                    "requests": list(self._requests.values()), "turns": dict(self._turns)}

    def _ready(self):
        with self._lock:
            if self._state["phase"] != "running" or self._connection is None or self._cancelled.is_set():
                raise RuntimeError("Open Codex before using a session.")
            return self._connection

    def rpc(self, method, params):
        if not isinstance(method, str) or method not in self.METHODS or not isinstance(params, dict) or set(params) - self.METHODS[method]:
            raise RuntimeError("Unsupported Codex operation or parameters.")
        params = dict(params)
        for key, value in params.items():
            if key == "archived":
                if not isinstance(value, bool):
                    raise RuntimeError("archived must be a boolean.")
            elif not isinstance(value, str):
                raise RuntimeError(f"{key} must be text.")
        connection = self._ready()
        if method == "thread/list":
            params.update(limit=100, modelProviders=["fastllm"], sortKey="updated_at")
        if method in {"thread/start", "thread/resume"}:
            params.update(model=self._service["modelName"], modelProvider="fastllm",
                          approvalPolicy="on-request", approvalsReviewer="user", sandbox="workspace-write")
            if method == "thread/start":
                workspace = Path(params.get("cwd") or self.directory / "workspace").expanduser().resolve()
                if not workspace.is_dir():
                    raise RuntimeError("Choose an existing workspace directory on the server.")
                params["cwd"] = str(workspace)
        if method == "thread/read":
            params["includeTurns"] = True
        if method == "turn/start":
            if "effort" in params and params["effort"] not in reasoning_options(self._service)[0]:
                raise RuntimeError("This reasoning effort is not supported by the current model.")
            text = params.pop("text", "")
            if not text.strip():
                raise RuntimeError("Enter a message first.")
            params["input"] = [{"type": "text", "text": text, "text_elements": []}]
        return connection.call(method, params)

    def respond(self, request_id, result):
        connection = self._ready()
        with self._lock:
            request = self._requests.get(str(request_id))
            if not request:
                raise RuntimeError("This Codex request has already been resolved.")
            method = request["method"]
            if not isinstance(result, dict):
                raise RuntimeError("Invalid Codex response.")
            if method in self.APPROVALS:
                if set(result) != {"decision"} or result["decision"] not in ("accept", "decline", "cancel"):
                    raise RuntimeError("Choose allow once, decline, or cancel.")
                allowed = request.get("params", {}).get("availableDecisions")
                if allowed and result["decision"] not in allowed:
                    raise RuntimeError("This decision is not available for the request.")
            elif method == "item/tool/requestUserInput":
                questions = {question["id"] for question in request["params"]["questions"]}
                answers = result.get("answers")
                if (set(result) != {"answers"} or not isinstance(answers, dict) or set(answers) != questions
                        or any(not isinstance(value, dict) or not isinstance(value.get("answers"), list)
                               or any(not isinstance(answer, str) for answer in value["answers"]) for value in answers.values())):
                    raise RuntimeError("Answer each Codex question.")
            elif method == "item/permissions/requestApproval":
                # Extra grants remain explicit and scoped to the current turn.
                requested = request["params"].get("permissions", {})
                if result not in ({"permissions": {}, "scope": "turn"}, {"permissions": requested, "scope": "turn"}):
                    raise RuntimeError("Choose allow this turn or decline.")
            else:
                raise RuntimeError("This Codex request is unsupported. Cancel the turn to continue.")
            connection.send({"id": request["id"], "result": result})
            self._requests.pop(str(request_id), None)
        return {"ok": True}
