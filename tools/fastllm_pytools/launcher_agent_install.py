"""Explicit installation of pinned, private OpenCode and Codex runtimes."""

import json
import os
import tempfile
from pathlib import Path

from .harness_install import (_check_cancelled, _download_node, _extract_node,
                              _installation_lock, _run_command)


AGENTS = {
    "opencode": {"name": "OpenCode", "package": "opencode-ai", "version": "1.18.26",
                 "entry": "opencode-ai/bin/opencode.exe"},
    "codex": {"name": "Codex", "package": "@openai/codex", "version": "0.153.4",
              "entry": "@openai/codex/bin/codex.js"},
}


def runtime_command(root, agent):
    entry = root / "node_modules" / AGENTS[agent]["entry"]
    node = root / ("node/node.exe" if os.name == "nt" else "node/bin/node")
    if not entry.is_file() or not node.is_file():
        return None
    return [str(entry)] if agent == "opencode" else [str(node), str(entry)]


def install_runtime(directory, agent, progress, cancelled, *, upgrade=False):
    spec = AGENTS[agent]
    directory.mkdir(parents=True, exist_ok=True)
    with _installation_lock(directory, spec["name"]):
        root = directory / "runtime"
        if not upgrade and runtime_command(root, agent):
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
            npm = staging / ("node/node_modules/npm/bin/npm-cli.js" if os.name == "nt"
                             else "node/lib/node_modules/npm/bin/npm-cli.js")
            environment = os.environ.copy()
            environment["PATH"] = str(node.parent) + os.pathsep + environment.get("PATH", "")
            (staging / "package.json").write_text(json.dumps({"private": True}), encoding="utf-8")
            _run_command([str(node), str(npm), "install", "--prefix", str(staging), "--global=false",
                          "--save-exact", "--omit=dev", "--include=optional", "--ignore-scripts=false",
                          "--no-audit", "--no-fund", "--loglevel=http",
                          f'{spec["package"]}@{spec["version"]}'],
                         staging, environment, progress, cancelled, "dependencies", name=spec["name"])
            command = runtime_command(staging, agent)
            if not command:
                raise RuntimeError(f'{spec["name"]} installation is incomplete. Retry the installation.')
            _run_command([*command, "--version"], staging, environment, progress, cancelled,
                         "verify", timeout=60, name=spec["name"])
            _check_cancelled(cancelled, spec["name"])
            backup = Path(temporary) / "previous"
            if root.exists():
                root.rename(backup)
            try:
                staging.rename(root)
            except BaseException:
                if backup.exists():
                    backup.rename(root)
                raise
