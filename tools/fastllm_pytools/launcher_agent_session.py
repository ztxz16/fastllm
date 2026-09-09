"""Shared session RPC transport, event delivery and approval validation for native agents."""

import json
import os
import queue
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path

from .harness_install import terminate_process
from .launcher_agent_runtime import AgentRuntime
from .launcher_agent_models import reasoning_options


class SessionConnection:
    def __init__(self, process, on_event, name):
        self.name = name
        self.process, self.on_event = process, on_event
        self._lock, self._write_lock = threading.Lock(), threading.Lock()
        self._pending, self._next_id, self._closed = {}, 0, False
        self._event_cursor = 0
        self.reader = threading.Thread(target=self._read, name="ftllm-agent-rpc", daemon=True)
        self.reader.start()

    def send(self, message):
        with self._write_lock:
            try:
                self.process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
                self.process.stdin.flush()
            except (OSError, ValueError) as error:
                raise RuntimeError(f"{self.name} is disconnected. Reopen it from Launcher.") from error

    def call(self, method, params, timeout=30):
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self.name} is disconnected. Reopen it from Launcher.")
            self._next_id += 1
            request_id, response = self._next_id, queue.Queue(maxsize=1)
            self._pending[request_id] = response
        try:
            self.send({"id": request_id, "method": method, "params": params})
            try:
                message = response.get(timeout=timeout)
            except queue.Empty as error:
                raise RuntimeError(f"{self.name} {method} timed out. Check the session before retrying.") from error
            if "error" in message:
                raise RuntimeError(message["error"].get("message", f"{self.name} request failed."))
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
                        pending.put_nowait({"error": {"message": f"{self.name} disconnected."}})

    def close(self):
        terminate_process(self.process)
        self.reader.join(timeout=3)
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()


class SessionRuntime(AgentRuntime):
    METHODS = {
        "thread/list": {"cursor", "searchTerm", "archived"},
        "thread/start": {"cwd"}, "thread/resume": {"threadId"},
        "thread/read": {"threadId"}, "thread/name/set": {"threadId", "name"},
        "thread/archive": {"threadId"}, "turn/start": {"threadId", "text", "effort"},
        "turn/interrupt": {"threadId", "turnId"},
    }
    APPROVALS = {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}

    def __init__(self, agent, directory=None):
        super().__init__(agent, directory)
        self._connection = None
        self._service = {}
        # This is a bounded delivery buffer, not conversation history. Clients
        # that miss events reload the full persisted thread from app-server.
        self._events, self._sequence = deque(maxlen=4096), 0
        self._requests, self._turns = {}, {}

    def _launch(self, command, service, environment, cancelled):
        with tempfile.TemporaryDirectory(prefix="launch-", dir=self.directory) as temporary:
            with (Path(temporary) / "session.log").open("w+b") as log:
                with self._lock:
                    if cancelled.is_set():
                        return
                    self._events.clear(); self._requests.clear(); self._turns.clear(); self._sequence = 0
                    self._service = service
                    self._process = process = subprocess.Popen(command,
                        cwd=self.directory / "workspace", env=environment, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=log, start_new_session=os.name != "nt")
                connection = SessionConnection(process, lambda message: self._event(message, cancelled), self.name)
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
                            raise RuntimeError(f"{self.name} exited.\n" + log.read().decode("utf-8", errors="replace"))
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
                raise RuntimeError(f"Open {self.name} before using a session.")
            return self._connection

    def rpc(self, method, params):
        if not isinstance(method, str) or method not in self.METHODS or not isinstance(params, dict) or set(params) - self.METHODS[method]:
            raise RuntimeError(f"Unsupported {self.name} operation or parameters.")
        params = dict(params)
        for key, value in params.items():
            if key == "archived":
                if not isinstance(value, bool):
                    raise RuntimeError("archived must be a boolean.")
            elif not isinstance(value, str):
                raise RuntimeError(f"{key} must be text.")
        connection = self._ready()
        if method == "thread/start":
            workspace = Path(params.get("cwd") or self.directory / "workspace").expanduser().resolve()
            if not workspace.is_dir():
                raise RuntimeError("Choose an existing workspace directory on the server.")
            params["cwd"] = str(workspace)
        if method == "turn/start":
            if "effort" in params and params["effort"] not in reasoning_options(self._service)[0]:
                raise RuntimeError("This reasoning effort is not supported by the current model.")
            text = params.get("text", "")
            if not text.strip():
                raise RuntimeError("Enter a message first.")
        return connection.call(method, self._prepare_rpc(method, params))

    def respond(self, request_id, result):
        connection = self._ready()
        with self._lock:
            request = self._requests.get(str(request_id))
            if not request:
                raise RuntimeError(f"This {self.name} request has already been resolved.")
            method = request["method"]
            if not isinstance(result, dict):
                raise RuntimeError(f"Invalid {self.name} response.")
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
                    raise RuntimeError(f"Answer each {self.name} question.")
            elif method == "item/permissions/requestApproval":
                # Extra grants remain explicit and scoped to the current turn.
                requested = request["params"].get("permissions", {})
                if result not in ({"permissions": {}, "scope": "turn"}, {"permissions": requested, "scope": "turn"}):
                    raise RuntimeError("Choose allow this turn or decline.")
            else:
                raise RuntimeError(f"This {self.name} request is unsupported. Cancel the turn to continue.")
            connection.send({"id": request["id"], "result": result})
            self._requests.pop(str(request_id), None)
        return {"ok": True}

    def _prepare_rpc(self, method, params):
        return params
