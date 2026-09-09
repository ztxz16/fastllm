"""Managed runtime operations exposed only by fixed, built-in agent plugins."""

import json
import shutil
import threading

from . import harness_install, launcher_agent_install


class ManagedAgentRuntime:
    def _runtime_spec(self):
        if self.agent == "harness":
            return {"name": "DeepSeek Harness", "package": "@deepseek-ai/dsh",
                    "version": harness_install.HARNESS_VERSION}
        return launcher_agent_install.AGENTS[self.agent]

    def _managed_command(self):
        root = self.directory / "runtime"
        return (harness_install.runtime_command(root) if self.agent == "harness"
                else launcher_agent_install.runtime_command(root, self.agent))

    def _management_state(self):
        root = self.directory / "runtime"
        spec = self._runtime_spec()
        version = ""
        try:
            version = json.loads((root / "node_modules" / spec["package"] / "package.json")
                                 .read_text(encoding="utf-8")).get("version", "")
        except (OSError, ValueError, AttributeError):
            pass
        return {"managed": root.exists() or root.is_symlink(),
                "source": "managed" if self._managed_command() else "path" if self._command() else "none",
                "version": version if isinstance(version, str) else "", "targetVersion": spec["version"]}

    def manage(self, operation):
        if operation not in {"install", "upgrade", "remove", "cancel"}:
            raise RuntimeError("Unknown agent runtime operation")
        with self._operations:
            if operation == "cancel":
                return self.stop()
            with self._lock:
                if self._state["phase"] in {"installing", "upgrading", "removing", "starting"}:
                    raise RuntimeError("An agent operation is already in progress. Wait or cancel it first.")
                if operation == "install" and self._managed_command():
                    return self.state()
            # Stop the child before replacing its files. The operation lock also
            # excludes a concurrent open, including the legacy install-and-open API.
            self.stop()
            with self._lock:
                if self._thread and self._thread.is_alive():
                    raise RuntimeError("The agent is still stopping. Retry shortly.")
                self._cancelled = threading.Event()
                self._state.update(phase={"install": "installing", "upgrade": "upgrading", "remove": "removing"}[operation],
                                   stage="", done=0, total=0, error="", sessionId="", url="")
                self._thread = threading.Thread(target=self._manage_runtime,
                    args=(operation, self._cancelled), name=f"ftllm-{self.agent}-{operation}", daemon=True)
                self._thread.start()
                return self.state()

    def _manage_runtime(self, operation, cancelled):
        def update(**values):
            with self._lock:
                if self._cancelled is cancelled and not cancelled.is_set():
                    self._state.update(values)

        def progress(stage, done, total):
            update(stage=stage, done=done, total=total)

        try:
            if operation == "remove":
                self.directory.mkdir(parents=True, exist_ok=True)
                with harness_install._installation_lock(self.directory, self._runtime_spec()["name"]):
                    harness_install._check_cancelled(cancelled)
                    root = self.directory / "runtime"
                    # Only the private runtime belongs to the installer. Keep
                    # sessions, configuration and workspaces, and never follow a symlink.
                    if root.is_symlink():
                        root.unlink()
                    elif root.exists():
                        shutil.rmtree(root)
            else:
                installer = harness_install if self.agent == "harness" else launcher_agent_install
                args = (self.directory,) if self.agent == "harness" else (self.directory, self.agent)
                installer.install_runtime(*args, progress, cancelled, upgrade=operation == "upgrade")
            update(phase="stopped", stage="", done=0, total=0)
        except Exception as error:
            update(phase="failed", stage="", error=str(error))
