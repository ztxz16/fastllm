"""Safe, dependency-free Python bridge to Pi's JSONL RPC mode."""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional
from urllib.parse import urlsplit


BRIDGE_VERSION = "0.2.1"
PI_VERSION = "0.84.4"
_RUNTIME_TOOLS = ("runtime_info",)
_PROJECT_TOOLS = (
    "list_project_files",
    "read_project_file",
    "search_project_files",
)
_WEB_TOOLS = (
    "web_search",
    "read_web_page",
)
_ALL_TOOLS = _RUNTIME_TOOLS + _PROJECT_TOOLS + _WEB_TOOLS
_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh", "max"}


class PiAgentError(RuntimeError):
    """Raised when the packaged Pi process cannot complete an agent run."""


class PiAgentCancelled(PiAgentError):
    """Raised after a caller requests cancellation of an active Pi run."""


def _resource_path(*parts: str) -> Path:
    resource = resources.files("ftllm_agent_runtime")
    for part in parts:
        resource = resource.joinpath(part)
    return Path(str(resource))


def _normalize_api_base(value: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("api_base cannot be empty")
    if not normalized.endswith("/v1"):
        normalized += "/v1"
    return normalized


def _safe_name(value: str, fallback: str) -> str:
    name = Path(str(value or "")).name.strip() or fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:120] or fallback


class _WebToolBridge:
    """Expose a Python WebAgent to the isolated Pi process over localhost."""

    MAX_REQUEST_BYTES = 16 * 1024
    MAX_SOURCES = 40

    def __init__(self, backend: Any):
        if not callable(getattr(backend, "search", None)):
            raise TypeError("web_backend must provide search(query, limit)")
        if not callable(getattr(backend, "read_page", None)):
            raise TypeError("web_backend must provide read_page(url, limit)")
        self.backend = backend
        self.token = secrets.token_urlsafe(32)
        self.sources: list[Dict[str, Any]] = []
        self._source_by_url: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                bridge._handle(self)

            def log_message(self, _format: str, *args: Any) -> None:
                del args

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="ftllm-pi-web-tools",
            daemon=True,
        )
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    @staticmethod
    def _result_value(result: Any, key: str) -> str:
        if isinstance(result, Mapping):
            value = result.get(key, "")
        else:
            value = getattr(result, key, "")
        return str(value or "")

    def _register_results(self, results: Iterable[Any]) -> list[Dict[str, Any]]:
        selected = []
        with self._lock:
            for result in results:
                url = self._result_value(result, "url").strip()
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    continue
                source = self._source_by_url.get(url)
                if source is None:
                    if len(self.sources) >= self.MAX_SOURCES:
                        break
                    source = {
                        "index": len(self.sources) + 1,
                        "title": self._result_value(result, "title")[:500],
                        "url": url[:4096],
                        "snippet": self._result_value(result, "snippet")[:2000],
                    }
                    self.sources.append(source)
                    self._source_by_url[url] = source
                selected.append(dict(source))
        return selected

    def public_sources(self) -> list[Dict[str, Any]]:
        with self._lock:
            return [dict(source) for source in self.sources]

    @staticmethod
    def _send(
        handler: BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            if handler.path != "/tool":
                self._send(handler, 404, {"error": "not found"})
                return
            if handler.headers.get("Authorization") != f"Bearer {self.token}":
                self._send(handler, 403, {"error": "forbidden"})
                return
            length = int(handler.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.MAX_REQUEST_BYTES:
                raise ValueError("invalid web tool request size")
            payload = json.loads(handler.rfile.read(length))
            action = str(payload.get("action", ""))
            if action == "search":
                query = " ".join(str(payload.get("query", "")).split())
                if not query or len(query) > 500:
                    raise ValueError("search query must contain 1-500 characters")
                limit = max(1, min(int(payload.get("limit", 6)), 10))
                results = self.backend.search(query, limit=limit)
                self._send(
                    handler, 200, {"results": self._register_results(results)}
                )
                return
            if action == "read":
                index = int(payload.get("source", 0))
                with self._lock:
                    if index < 1 or index > len(self.sources):
                        raise ValueError(f"unknown web source index: {index}")
                    source = dict(self.sources[index - 1])
                limit = max(500, min(int(payload.get("limit", 9000)), 12000))
                content = self.backend.read_page(source["url"], limit=limit)
                self._send(handler, 200, {"source": source, "content": content})
                return
            raise ValueError(f"unknown web tool action: {action}")
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send(handler, 400, {"error": str(error)})
        except Exception as error:
            self._send(handler, 502, {"error": str(error)})

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


class PiAgentRuntime:
    """Launch one isolated Pi agent process per request.

    This first Linux prototype deliberately disables every Pi built-in tool.
    The bundled extension can inspect read-only source snapshots or call a
    caller-supplied, SSRF-protected web backend through a temporary localhost
    bridge.
    """

    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str = "",
        timeout: float = 300.0,
        context_window: int = 40_000,
        max_tokens: int = 4_096,
        max_turns: int = 8,
        binary: Optional[str] = None,
    ):
        self.api_base = _normalize_api_base(api_base)
        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("model cannot be empty")
        self.api_key = str(api_key or "")
        self.timeout = max(5.0, float(timeout))
        self.context_window = max(4_096, int(context_window))
        self.max_tokens = max(64, int(max_tokens))
        self.max_turns = max(1, int(max_turns))
        self.binary = Path(binary).resolve() if binary else _resource_path("bin", "pi")
        self.extension = _resource_path("extensions", "project_tools.ts")
        self._validate_installation()

    def _validate_installation(self) -> None:
        machine = platform.machine().lower()
        if not sys.platform.startswith("linux") or machine not in {"x86_64", "amd64"}:
            raise PiAgentError("ftllm-agent-runtime currently supports Linux x86-64 only")
        if not self.binary.is_file():
            raise PiAgentError(f"packaged Pi executable is missing: {self.binary}")
        if not os.access(self.binary, os.X_OK):
            try:
                self.binary.chmod(self.binary.stat().st_mode | 0o111)
            except OSError as error:
                raise PiAgentError(f"packaged Pi executable is not executable: {error}") from error
        if not self.extension.is_file():
            raise PiAgentError(f"packaged Pi extension is missing: {self.extension}")

    def info(self) -> Dict[str, Any]:
        try:
            completed = subprocess.run(
                [str(self.binary), "--version"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
            )
            detected = completed.stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise PiAgentError(f"could not execute packaged Pi: {error}") from error
        return {
            "available": True,
            "bridge_version": BRIDGE_VERSION,
            "pi_version": detected or PI_VERSION,
            "binary": str(self.binary),
            "binary_bytes": self.binary.stat().st_size,
            "platform": "linux_x86_64",
            "tools": list(_ALL_TOOLS),
        }

    def _models_config(self) -> Dict[str, Any]:
        identity = self.model.lower()
        compat: Dict[str, Any] = {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": True,
            "supportsUsageInStreaming": True,
            "supportsFinishReason": True,
            "maxTokensField": "max_tokens",
            "supportsStrictMode": False,
        }
        reasoning = any(
            marker in identity
            for marker in ("qwen", "deepseek", "kimi", "glm", "reasoning")
        )
        if "qwen" in identity:
            compat["thinkingFormat"] = "qwen-chat-template"
        return {
            "providers": {
                "fastllm": {
                    "baseUrl": self.api_base,
                    "api": "openai-completions",
                    "apiKey": "$FTLLM_PI_API_KEY",
                    "authHeader": True,
                    "compat": compat,
                    "models": [
                        {
                            "id": self.model,
                            "name": self.model,
                            "reasoning": reasoning,
                            "input": ["text"],
                            "contextWindow": self.context_window,
                            "maxTokens": self.max_tokens,
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                }
            }
        }

    @staticmethod
    def _copy_project_files(
        work_dir: Path, files: Iterable[Mapping[str, Any]]
    ) -> list[Dict[str, Any]]:
        project_dir = work_dir / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for index, item in enumerate(files, 1):
            text = str(item.get("text", ""))
            original_name = Path(str(item.get("name", ""))).name or f"file-{index}"
            safe_name = _safe_name(original_name, f"file-{index}")
            target = project_dir / f"{index:03d}-{safe_name}"
            target.write_text(text, encoding="utf-8")
            manifest.append(
                {
                    "index": index,
                    "name": original_name,
                    "path": str(target),
                    "size": int(item.get("size", len(text.encode("utf-8")))),
                    "truncated": bool(item.get("truncated", False)),
                }
            )
        if not manifest:
            raise ValueError("at least one project file is required")
        return manifest

    def _command(
        self,
        system_prompt: str,
        thinking_level: str,
        active_tools: Iterable[str],
    ) -> list[str]:
        return [
            str(self.binary),
            "--mode",
            "rpc",
            "--no-session",
            "--provider",
            "fastllm",
            "--model",
            self.model,
            "--thinking",
            thinking_level,
            "--system-prompt",
            system_prompt,
            "--no-builtin-tools",
            "--tools",
            ",".join(active_tools),
            "--extension",
            str(self.extension),
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-themes",
            "--no-approve",
            "--offline",
        ]

    def stream(
        self,
        prompt: str,
        files: Iterable[Mapping[str, Any]],
        system_prompt: str,
        thinking_level: str = "off",
        web_backend: Optional[Any] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield normalized events from a complete Pi agent run."""

        if cancel_event is not None and cancel_event.is_set():
            raise PiAgentCancelled("Pi agent request was cancelled")
        level = str(thinking_level or "off").lower()
        if level not in _THINKING_LEVELS:
            raise ValueError(f"unsupported thinking level: {thinking_level}")
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")

        with tempfile.TemporaryDirectory(prefix="ftllm-pi-agent-") as temporary:
            root = Path(temporary)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            file_items = list(files)
            if not file_items and web_backend is None:
                raise ValueError("at least one project file or web backend is required")
            manifest = (
                self._copy_project_files(root, file_items) if file_items else []
            )
            (config_dir / "models.json").write_text(
                json.dumps(self._models_config(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            web_bridge = _WebToolBridge(web_backend) if web_backend else None
            active_tools = list(_RUNTIME_TOOLS)
            if file_items:
                active_tools.extend(_PROJECT_TOOLS)
            if web_bridge:
                active_tools.extend(_WEB_TOOLS)
            environment = os.environ.copy()
            environment.update(
                {
                    "FTLLM_AGENT_PROJECT_MANIFEST": json.dumps(
                        manifest, ensure_ascii=False
                    ),
                    "FTLLM_AGENT_BRIDGE_VERSION": BRIDGE_VERSION,
                    "FTLLM_AGENT_PI_VERSION": PI_VERSION,
                    "FTLLM_PI_API_KEY": self.api_key or "local-fastllm",
                    "PI_CODING_AGENT_DIR": str(config_dir),
                    "PI_CODING_AGENT_SESSION_DIR": str(root / "sessions"),
                    "PI_PACKAGE_DIR": str(self.binary.parent),
                    "PI_OFFLINE": "1",
                    "PI_SKIP_VERSION_CHECK": "1",
                    "PI_TELEMETRY": "0",
                }
            )
            if web_bridge:
                environment.update(
                    {
                        "FTLLM_AGENT_WEB_BRIDGE_URL": web_bridge.url,
                        "FTLLM_AGENT_WEB_BRIDGE_TOKEN": web_bridge.token,
                    }
                )

            try:
                process = subprocess.Popen(
                    self._command(system_prompt, level, active_tools),
                    cwd=root,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except BaseException:
                if web_bridge:
                    web_bridge.close()
                raise
            if process.stdin is None or process.stdout is None or process.stderr is None:
                process.kill()
                if web_bridge:
                    web_bridge.close()
                raise PiAgentError("could not open Pi RPC pipes")

            records: queue.Queue[tuple[str, Optional[str]]] = queue.Queue()
            stderr_tail: deque[str] = deque(maxlen=40)

            def read_stream(kind: str, stream) -> None:
                try:
                    for line in stream:
                        records.put((kind, line.rstrip("\n").rstrip("\r")))
                finally:
                    records.put((kind, None))

            stdout_thread = threading.Thread(
                target=read_stream,
                args=("stdout", process.stdout),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=read_stream,
                args=("stderr", process.stderr),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            command = json.dumps(
                {"id": "ftllm-prompt", "type": "prompt", "message": prompt},
                ensure_ascii=False,
            )
            try:
                process.stdin.write(command + "\n")
                process.stdin.flush()
            except BaseException:
                process.kill()
                process.wait(timeout=3)
                if web_bridge:
                    web_bridge.close()
                raise

            deadline = time.monotonic() + self.timeout
            stdout_closed = False
            saw_agent_end = False
            saw_assistant_output = False
            turns = 0
            try:
                while not saw_agent_end:
                    if cancel_event is not None and cancel_event.is_set():
                        raise PiAgentCancelled("Pi agent request was cancelled")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PiAgentError(
                            f"Pi agent exceeded the {self.timeout:g}s request timeout"
                        )
                    try:
                        poll_interval = 0.1 if cancel_event is not None else 1.0
                        kind, line = records.get(
                            timeout=min(poll_interval, remaining))
                    except queue.Empty:
                        if process.poll() is not None:
                            break
                        continue
                    if line is None:
                        if kind == "stdout":
                            stdout_closed = True
                            if process.poll() is not None:
                                break
                        continue
                    if kind == "stderr":
                        stderr_tail.append(line)
                        continue
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise PiAgentError(f"invalid Pi RPC JSON: {line[:300]}") from error

                    event_type = str(event.get("type", ""))
                    if event_type == "response" and not event.get("success", False):
                        raise PiAgentError(str(event.get("error") or event))
                    if event_type == "turn_start":
                        if turns >= self.max_turns:
                            raise PiAgentError(
                                f"Pi agent exceeded the {self.max_turns} turn limit"
                            )
                        yield {"type": "turn_start"}
                    elif event_type == "message_update":
                        update = event.get("assistantMessageEvent", {}) or {}
                        update_type = str(update.get("type", ""))
                        if update_type == "text_delta":
                            delta = str(update.get("delta", ""))
                            saw_assistant_output = saw_assistant_output or bool(delta)
                            yield {"type": "text_delta", "text": delta}
                        elif update_type == "thinking_delta":
                            delta = str(update.get("delta", ""))
                            saw_assistant_output = saw_assistant_output or bool(delta)
                            yield {
                                "type": "thinking_delta",
                                "text": delta,
                            }
                    elif event_type == "tool_execution_start":
                        yield {
                            "type": "tool_start",
                            "name": str(event.get("toolName", "")),
                            "arguments": event.get("args", {}),
                        }
                    elif event_type == "tool_execution_end":
                        tool_name = str(event.get("toolName", ""))
                        yield {
                            "type": "tool_end",
                            "name": tool_name,
                            "is_error": bool(event.get("isError", False)),
                        }
                        if web_bridge and tool_name in {
                            "web_search", "read_web_page"
                        }:
                            yield {
                                "type": "web_sources",
                                "sources": web_bridge.public_sources(),
                            }
                    elif event_type == "turn_end":
                        turns += 1
                    elif event_type == "agent_end":
                        saw_agent_end = True

                return_code = process.poll()
                if not saw_agent_end:
                    details = "\n".join(stderr_tail).strip()
                    suffix = f": {details}" if details else ""
                    state = "closed stdout" if stdout_closed else f"exited {return_code}"
                    raise PiAgentError(f"Pi agent {state} before agent_end{suffix}")
                if not saw_assistant_output:
                    raise PiAgentError("Pi agent completed without assistant output")
                done_event: Dict[str, Any] = {"type": "done", "turns": turns}
                if web_bridge:
                    done_event["web_sources"] = web_bridge.public_sources()
                yield done_event
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
                if web_bridge:
                    web_bridge.close()
