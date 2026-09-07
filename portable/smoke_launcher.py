#!/usr/bin/env python3
"""Start a portable Launcher without a desktop and verify its authenticated API."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path


CONTROL_URL = re.compile(r"http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9_-]+)")


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            except ProcessLookupError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Portable bundle root")
    parser.add_argument("--entrypoint", choices=("ftllm", "launch.sh"), default="ftllm")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    executable = bundle / args.entrypoint
    if not executable.is_file():
        parser.error(f"missing bundled ftllm executable: {executable}")

    port = available_port()
    with tempfile.TemporaryDirectory(prefix="ftllm-portable-smoke-") as temporary:
        config = Path(temporary) / "config with spaces.json"
        environment = os.environ.copy()
        environment.pop("DISPLAY", None)
        environment.pop("WAYLAND_DISPLAY", None)
        environment["XDG_CONFIG_HOME"] = str(Path(temporary) / "xdg-config")
        environment["XDG_CACHE_HOME"] = str(Path(temporary) / "xdg-cache")
        if args.entrypoint == "ftllm":
            subprocess.run(
                [
                    "bash", "--noprofile", "--norc", "-e", "-c",
                    'source "$1/env.sh"\n'
                    '[[ "$(command -v ftllm)" == "$1/ftllm" ]]\n'
                    'ftllm server --help',
                    "portable-cli-smoke", str(bundle),
                ],
                check=True, timeout=30, cwd=temporary, env=environment,
                stdout=subprocess.DEVNULL,
            )
        command = [str(executable)]
        if args.entrypoint == "ftllm":
            command.append("launch")
        process = subprocess.Popen(
            command + [
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-browser",
                "--config",
                str(config),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            cwd=temporary,
            start_new_session=True,
        )
        lines: "queue.Queue[str]" = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line.rstrip())

        threading.Thread(target=read_output, daemon=True).start()
        captured = []
        token = None
        deadline = time.monotonic() + 60
        try:
            while time.monotonic() < deadline and token is None:
                if process.poll() is not None and lines.empty():
                    break
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                captured.append(
                    CONTROL_URL.sub(
                        f"http://127.0.0.1:{port}/?token=[redacted]", line
                    )
                )
                match = CONTROL_URL.search(line)
                if match and int(match.group(1)) == port:
                    token = match.group(2)
            if token is None:
                raise RuntimeError(
                    "launcher did not report a control URL:\n" + "\n".join(captured[-30:])
                )

            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/bootstrap",
                headers={"X-FTLLM-Launcher-Token": token},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.load(response)
            if not isinstance(payload.get("profiles"), list):
                raise RuntimeError("launcher bootstrap response has no profile list")
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
                if b"FastLLM" not in response.read(65536):
                    raise RuntimeError("launcher root did not return the FastLLM UI")
            for asset in ("app.js", "styles.css", "template.html", "standalone.js"):
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/assets/webui/{asset}", timeout=10
                ) as response:
                    if response.status != 200 or not response.read():
                        raise RuntimeError(f"missing shared Studio asset: {asset}")
        finally:
            stop_process_group(process)

    print(f"Headless Launcher smoke test passed: {args.entrypoint}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
