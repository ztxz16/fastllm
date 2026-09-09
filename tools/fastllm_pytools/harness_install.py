"""On-demand, private Node/npm installation for the embedded Harness."""

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.request import urlopen


HARNESS_VERSION = "0.1.5-alpha.1"
NODE_VERSION = "24.20.0"
# From https://nodejs.org/download/release/v24.20.0/SHASUMS256.txt
NODE_HASHES = {
    "linux-x64": "855d581f8a4eb1a8117e3426de25fe02770592febcfb31369aee1ffbfee9e8ec",
    "linux-arm64": "3515603e2487879a39bc75716f1a2affd027500c64ba50e845cf72cb33219013",
    "darwin-x64": "9e5b2644cf107befb6aefca676b96d3296bc10138096f022ed378d6233ed81f4",
    "darwin-arm64": "40e5607e5ecb3db9192723776da2d75d966260fc74a7a9e731c1bd67dda96bc8",
    "win-x64": "6cac9ffbca8f6a47091e4b5c772e0606049c3871cb67d900c0cedde630e545ba",
    "win-arm64": "31c6799744de8a54601643098040c68c3697e56c94e407d61d0e5fa5f34191d7",
}


def runtime_command(root):
    entry = root / "node_modules/@deepseek-ai/dsh/lib/bin.js"
    # Keep the manually installed preview environment usable.
    nodes = (root / "node/bin/node", root / "node/node.exe",
             root / "node_modules/node/bin/node", root / "node_modules/node/bin/node.exe")
    return next(([str(node), str(entry)] for node in nodes if node.is_file() and entry.is_file()), None)


def _check_cancelled(cancelled, name="Harness"):
    if cancelled.is_set():
        raise RuntimeError(f"{name} installation was cancelled.")


def _node_archive():
    system = {"Linux": "linux", "Darwin": "darwin", "Windows": "win"}.get(platform.system())
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    key = f"{system}-{arch}"
    if key not in NODE_HASHES:
        raise RuntimeError("Private Node runtime installation requires Linux, macOS or Windows on x64/ARM64.")
    extension = "zip" if system == "win" else "tar.gz"
    return f"node-v{NODE_VERSION}-{key}.{extension}", NODE_HASHES[key]


def _download_node(destination, progress, cancelled):
    name, checksum = _node_archive()
    progress("download", 0, 0)
    digest, done = hashlib.sha256(), 0
    with urlopen(f"https://nodejs.org/download/release/v{NODE_VERSION}/{name}", timeout=5) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        while True:
            _check_cancelled(cancelled)
            chunk = response.read(128 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            progress("download", done, total)
    if digest.hexdigest() != checksum:
        raise RuntimeError("Node.js download checksum mismatch. Retry the installation.")


def _extract_node(archive, target, cancelled):
    # Copy only regular runtime files. Never materialize archive links/devices,
    # or archive paths outside the staging directory (also on older Python).
    def destination(name):
        parts = PurePosixPath(name).parts
        if not parts or name.startswith("/") or ".." in parts or "\\" in name or ":" in name:
            raise RuntimeError("Invalid Node.js archive path.")
        relative = PurePosixPath(*parts[1:])
        if str(relative) in {"bin/node", "node.exe", "npm.cmd", "npx.cmd"} or str(relative).startswith(("lib/node_modules/npm/", "node_modules/npm/")):
            path = target.joinpath(*relative.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _check_cancelled(cancelled)
                path = destination(member.filename)
                if path and not member.is_dir():
                    with bundle.open(member) as source, path.open("wb") as output:
                        shutil.copyfileobj(source, output)
    else:
        with tarfile.open(archive) as bundle:
            for member in bundle:
                _check_cancelled(cancelled)
                path = destination(member.name)
                if path and member.isfile():
                    with bundle.extractfile(member) as source, path.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    path.chmod(0o755 if member.mode & 0o111 else 0o644)
        for name in ("npm", "npx"):
            (target / "bin" / name).symlink_to(f"../lib/node_modules/npm/bin/{name}-cli.js")


def terminate_process(process):
    # Both the stop request and the worker's finally block can reach here.
    lock = process.__dict__.setdefault("_ftllm_termination_lock", threading.Lock())
    with lock:
        if getattr(process, "_ftllm_terminated", False):
            return
        _terminate_process_tree(process)
        process._ftllm_terminated = True


def _terminate_process_tree(process):
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        return
    # The session leader can exit before its tools. Wait for the process group,
    # including children that ignore SIGTERM, instead of only waiting for Popen.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            process.poll()  # Reap the leader so it cannot keep an empty group alive.
            os.killpg(process.pid, 0)
            time.sleep(.05)
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def _run_command(command, root, environment, progress, cancelled, stage, timeout=1200, name="Harness"):
    _check_cancelled(cancelled, name)
    progress(stage, 0, 0)
    # Separate descriptors prevent reading progress from moving the child's
    # write position. Bound both the API diagnostics and each read.
    with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
        log = Path(temporary) / "install.log"
        with log.open("wb") as output, log.open("rb") as reader:
            process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
            try:
                deadline, tail, fetched = time.monotonic() + timeout, "", 0
                while True:
                    _check_cancelled(cancelled, name)
                    chunk = reader.read(65536).decode("utf-8", errors="replace")
                    tail = (tail + chunk)[-8000:]
                    fetched += chunk.count("npm http fetch")
                    if chunk:
                        progress(stage, fetched, 0)
                    if process.poll() is not None:
                        if process.returncode:
                            # npm may log registry URLs configured with credentials.
                            tail = re.sub(r"https?://\S+", "[registry URL]", tail)
                            raise RuntimeError(f"{name} {stage} failed (exit {process.returncode}).\n{tail[-4000:]}")
                        return
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{name} installation timed out. Retry the installation.")
                    cancelled.wait(.1)
            finally:
                terminate_process(process)


@contextmanager
def _installation_lock(directory, name="Harness"):
    with (directory / ".install.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.write(b"\0"); lock.flush(); lock.seek(0)
            acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError(f"Another Launcher is installing {name}. Wait for it to finish, then retry.") from error
        yield


def install_runtime(directory, progress, cancelled, *, upgrade=False):
    directory.mkdir(parents=True, exist_ok=True)
    with _installation_lock(directory):
        root = directory / "runtime"
        if not upgrade and runtime_command(root):
            return
        with tempfile.TemporaryDirectory(prefix=".install-", dir=directory) as temporary:
            staging = Path(temporary) / "runtime"
            staging.mkdir()
            archive = Path(temporary) / "node.archive"
            _download_node(archive, progress, cancelled)
            progress("extract", 0, 0)
            _extract_node(archive, staging / "node", cancelled)
            archive.unlink()
            node = staging / ("node/node.exe" if os.name == "nt" else "node/bin/node")
            npm = staging / ("node/node_modules/npm/bin/npm-cli.js" if os.name == "nt" else "node/lib/node_modules/npm/bin/npm-cli.js")
            environment = os.environ.copy()
            environment["PATH"] = str(node.parent) + os.pathsep + environment.get("PATH", "")
            (staging / "package.json").write_text(json.dumps({"private": True}), encoding="utf-8")
            _run_command([str(node), str(npm), "install", "--prefix", str(staging), "--global=false",
                "--save-exact", "--omit=dev", "--include=optional", "--ignore-scripts=false",
                "--no-audit", "--no-fund", "--loglevel=http", f"@deepseek-ai/dsh@{HARNESS_VERSION}"],
                staging, environment, progress, cancelled, "dependencies")
            command = runtime_command(staging)
            if command is None:
                raise RuntimeError("Harness installation is incomplete. Retry the installation.")
            _run_command([*command, "--version"], staging, environment, progress, cancelled, "verify", timeout=60)
            _check_cancelled(cancelled)
            backup = Path(temporary) / "previous"
            if root.exists():
                root.rename(backup)
            try:
                staging.rename(root)
            except BaseException:
                if backup.exists():
                    backup.rename(root)
                raise
