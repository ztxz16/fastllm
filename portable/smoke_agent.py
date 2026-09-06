#!/usr/bin/env python3
"""Exercise the packaged Pi executable against a disposable local model stub."""

from __future__ import annotations

import json
import argparse
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ftllm_agent_runtime.runtime import PiAgentRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-studio", action="store_true", help="Also verify Studio runtime discovery")
    options = parser.parse_args()
    marker = "ftllm-portable-agent-file-ok"
    requests = []

    class ModelHandler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            if len(requests) == 1:
                calls = [
                    ("read", {"path": "README.md"}),
                    ("find", {"pattern": "README.md", "path": "."}),
                    ("grep", {"pattern": marker, "path": "."}),
                ]
                delta = {"role": "assistant", "tool_calls": [
                    {"index": index, "id": f"portable-{name}", "type": "function",
                     "function": {"name": name, "arguments": json.dumps(arguments)}}
                    for index, (name, arguments) in enumerate(calls)
                ]}
                finish = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "portable-agent-ok"}
                finish = "stop"
            chunks = []
            for content, reason in ((delta, None), ({}, finish)):
                chunk = {
                    "id": "portable-smoke", "object": "chat.completion.chunk",
                    "created": 1, "model": "portable-smoke",
                    "choices": [{"index": 0, "delta": content, "finish_reason": reason}],
                }
                chunks.append("data: " + json.dumps(chunk) + "\n\n")
            payload = ("".join(chunks) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with tempfile.TemporaryDirectory(prefix="ftllm-agent-smoke-") as temporary:
        workspace = Path(temporary)
        (workspace / "README.md").write_text(marker + "\n", encoding="utf-8")
        server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            runtime = PiAgentRuntime(
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                model="portable-smoke", timeout=45, max_turns=3,
            )
            events = list(runtime.stream(
                "Read README.md and report success.", [], "Read the requested file.",
                working_directory=str(workspace),
            ))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        reads = [event for event in events if event["type"] == "tool_end" and event["name"] == "read"]
        if not reads or any(event["is_error"] for event in reads) or marker not in reads[0]["result"]:
            raise RuntimeError(f"Packaged Pi did not read the workspace file: {reads}")
        for name, expected in (("find", "README.md"), ("grep", marker)):
            results = [event for event in events if event["type"] == "tool_end" and event["name"] == name]
            if not results or any(event["is_error"] for event in results) or expected not in results[0]["result"]:
                raise RuntimeError(f"Packaged Pi {name} failed: {results}")
        if len(requests) != 2 or not any(
            message.get("role") == "tool" and marker in str(message.get("content"))
            for message in requests[-1]["messages"]
        ):
            raise RuntimeError("Pi did not send the file result back to the model")
        text = "".join(event.get("text", "") for event in events if event["type"] == "text_delta")
        if "portable-agent-ok" not in text or events[-1]["type"] != "done":
            raise RuntimeError("Pi did not finish the model/tool round trip")
        if options.check_studio:
            from fastapi.testclient import TestClient
            from ftllm.webui_server import add_webui_args, create_app

            args = add_webui_args(argparse.ArgumentParser()).parse_args([
                "--agent-runtime", "auto", "--api-base", "http://127.0.0.1:1/v1",
                "--api-model", "portable-smoke", "--history-dir", str(workspace / "history"),
                "--agent-workspace-root", str(workspace),
            ])
            with TestClient(create_app(args)) as client:
                response = client.get("/api/config")
                response.raise_for_status()
                config = response.json()
                if config["agent_runtime"] != "pi" or not config["pi_agent"]["available"]:
                    raise RuntimeError("Studio did not discover the packaged Pi runtime")
                if not config["workspace_agent_enabled"]:
                    raise RuntimeError("Studio directory Agent is disabled by default")
    print("Packaged Pi Agent smoke test passed (real executable, read/find/grep, local model stub).")


if __name__ == "__main__":
    main()
