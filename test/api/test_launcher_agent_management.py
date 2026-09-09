import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastllm_pytools import harness_install, launcher_agent_install
from fastllm_pytools.launcher import LauncherRuntime, create_launcher_app
from fastllm_pytools.launcher_codex import CodexRuntime
from fastllm_pytools.launcher_harness import HarnessRuntime
from fastllm_pytools.launcher_opencode import OpenCodeRuntime
from test_launcher_agents import FAKE_CODEX, SERVICE


def installed_files(root, agent, version="old"):
    spec = ({"package":"@deepseek-ai/dsh", "entry":"@deepseek-ai/dsh/lib/bin.js"}
            if agent == "harness" else launcher_agent_install.AGENTS[agent])
    for path in (root / ("node/node.exe" if os.name == "nt" else "node/bin/node"),
                 root / "node_modules" / spec["entry"]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    (root / "node_modules" / spec["package"] / "package.json").write_text(json.dumps({"version":version}))


class AgentManagementTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        no_system = patch("shutil.which", return_value=None)
        no_system.start(); self.addCleanup(no_system.stop)

    def runtimes(self):
        for factory in (HarnessRuntime, OpenCodeRuntime, CodexRuntime):
            runtime = factory(self.root / factory.__name__)
            self.addCleanup(runtime.stop)
            yield runtime

    def finish(self, runtime):
        runtime._thread.join(timeout=5)
        self.assertFalse(runtime._thread.is_alive())
        return runtime.state()

    def test_management_install_progress_cancel_retry_and_no_implicit_start(self):
        for runtime in self.runtimes():
            with self.subTest(agent=runtime.agent):
                installer = harness_install if runtime.agent == "harness" else launcher_agent_install
                entered = threading.Event()

                def install(*args, upgrade=False):
                    progress, cancelled = args[-2:]
                    progress("download", 25, 100); entered.set()
                    cancelled.wait(5)
                    raise RuntimeError("cancelled")

                with patch.object(installer, "install_runtime", side_effect=install):
                    runtime.manage("install")
                    self.assertTrue(entered.wait(2))
                    self.assertEqual(runtime.state()["done"], 25)
                    self.assertEqual(runtime.state()["sessionId"], "")
                    with self.assertRaisesRegex(RuntimeError, "already in progress"):
                        runtime.manage("upgrade")
                    self.assertEqual(runtime.start(SERVICE, "", "127.0.0.1", "http://localhost:8000")["phase"], "installing")
                    runtime.manage("cancel")
                    self.assertEqual(self.finish(runtime)["phase"], "stopped")
                    self.assertFalse(runtime.state()["installed"])

                def complete(*args, upgrade=False):
                    installed_files(runtime.directory / "runtime", runtime.agent, "new")

                with patch.object(installer, "install_runtime", side_effect=complete) as install:
                    runtime.manage("install")
                    state = self.finish(runtime)
                    self.assertEqual((state["phase"], state["source"], state["version"]), ("stopped", "managed", "new"))
                    self.assertTrue(state["installed"])
                    self.assertIsNone(runtime._process)
                    runtime.manage("install")
                    install.assert_called_once()

    def test_upgrade_replaces_valid_runtime_only_after_successful_verification(self):
        for runtime in self.runtimes():
            with self.subTest(agent=runtime.agent):
                installer = harness_install if runtime.agent == "harness" else launcher_agent_install
                installed_files(runtime.directory / "runtime", runtime.agent)
                mode = "fail"

                def extract(archive, destination, cancelled):
                    node = destination / ("node.exe" if os.name == "nt" else "bin/node")
                    node.parent.mkdir(parents=True); node.touch()

                def run(command, staging, environment, progress, cancelled, stage, **kwargs):
                    if stage == "dependencies":
                        spec = runtime._runtime_spec()
                        self.assertIn(spec["package"] + "@" + spec["version"], command)
                        installed_files(staging, runtime.agent, spec["version"])
                    elif mode == "fail":
                        raise RuntimeError("verification failed")

                with patch.object(installer, "_download_node", side_effect=lambda archive, *args:archive.touch()), \
                        patch.object(installer, "_extract_node", side_effect=extract), \
                        patch.object(installer, "_run_command", side_effect=run):
                    runtime.manage("upgrade")
                    state = self.finish(runtime)
                    self.assertEqual(state["phase"], "failed")
                    self.assertIn("verification failed", state["error"])
                    self.assertEqual(state["version"], "old")
                    self.assertTrue(state["installed"])
                    mode = "success"
                    runtime.manage("upgrade")
                    state = self.finish(runtime)
                    self.assertEqual(state["phase"], "stopped")
                    self.assertEqual(state["version"], state["targetVersion"])
                    self.assertFalse(list(runtime.directory.glob(".install-*")))

    def test_remove_preserves_sessions_workspace_and_system_installation(self):
        for runtime in self.runtimes():
            with self.subTest(agent=runtime.agent):
                installed_files(runtime.directory / "runtime", runtime.agent)
                for folder in ("home", "data", "workspace"):
                    saved = runtime.directory / folder / "keep"
                    saved.parent.mkdir(); saved.write_text("user data")
                system = self.root / (runtime.agent + "-system")
                system.write_text("external binary")
                with patch("shutil.which", return_value=str(system)):
                    runtime.manage("remove")
                    state = self.finish(runtime)
                    self.assertFalse(state["managed"])
                    self.assertTrue(state["installed"])
                    self.assertEqual(state["source"], "path")
                    self.assertEqual(system.read_text(), "external binary")
                for folder in ("home", "data", "workspace"):
                    self.assertEqual((runtime.directory / folder / "keep").read_text(), "user data")

    @unittest.skipIf(os.name == "nt", "POSIX symlink fixture")
    def test_remove_does_not_follow_runtime_symlink(self):
        runtime = CodexRuntime(self.root / "codex"); self.addCleanup(runtime.stop)
        external = self.root / "external"; installed_files(external, "codex")
        runtime.directory.mkdir(); (runtime.directory / "runtime").symlink_to(external, target_is_directory=True)
        runtime.manage("remove")
        self.assertEqual(self.finish(runtime)["phase"], "stopped")
        self.assertTrue((external / "node_modules/@openai/codex/package.json").is_file())
        self.assertFalse((runtime.directory / "runtime").is_symlink())

    def test_upgrade_stops_live_agent_before_installing_and_keeps_session_home(self):
        runtime = CodexRuntime(self.root / "codex"); self.addCleanup(runtime.stop)
        script = self.root / "codex.py"; script.write_text(FAKE_CODEX)
        with patch.object(runtime, "_command", return_value=[sys.executable, str(script)]):
            runtime.start(SERVICE, "key", "127.0.0.1", "http://localhost")
            deadline = time.monotonic() + 5
            while runtime.state()["phase"] == "starting" and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(runtime.state()["phase"], "running")
            process = runtime._process

            def install(*args, **kwargs):
                self.assertIsNotNone(process.poll())
                self.assertTrue((runtime.directory / "home/probe.json").is_file())
                self.assertTrue(kwargs["upgrade"])

            with patch.object(launcher_agent_install, "install_runtime", side_effect=install) as installing:
                runtime.manage("upgrade")
                self.assertEqual(self.finish(runtime)["phase"], "stopped")
                installing.assert_called_once()

    def test_plugin_api_requires_auth_and_only_manages_registered_agents_without_model(self):
        launcher = LauncherRuntime(str(self.root / "profiles.json"), plugins_dir=str(self.root / "plugins"))
        self.addCleanup(launcher.close)
        for runtime in self.runtimes():
            setattr(launcher, runtime.agent, runtime)
        headers = {"X-FTLLM-Launcher-Token":"control", "X-FTLLM-Plugin-Request":"1"}
        with TestClient(create_launcher_app(launcher, "control")) as client:
            for agent in ("harness", "opencode", "codex"):
                runtime = getattr(launcher, agent)
                url = f"/api/plugins/{agent}/runtime/install"
                with patch.object(runtime, "manage", return_value={"phase":"installing"}) as manage:
                    for invalid in ({}, {"X-FTLLM-Launcher-Token":"control"}, {**headers, "Origin":"http://evil.test"}):
                        self.assertEqual(client.post(url, headers=invalid, json={}).status_code, 403)
                    manage.assert_not_called()
                    launcher.plugins.set_enabled(agent, False)
                    self.assertEqual(client.post(url, headers=headers, json={}).status_code, 200)
                    manage.assert_called_once_with("install")
            for agent, operation in (("studio", "install"), ("models", "remove"), ("custom", "install"), ("codex", "exec")):
                response = client.post(f"/api/plugins/{agent}/runtime/{operation}", headers=headers, json={})
                self.assertEqual(response.status_code, 400, response.text)
            catalog = client.get("/api/plugins", headers=headers).json()["plugins"]
            for plugin in catalog:
                if plugin["id"] in {"harness", "opencode", "codex"}:
                    self.assertTrue(plugin["runtime"]["manageable"])
                    self.assertTrue(plugin["runtime"]["targetVersion"])
                else:
                    self.assertNotIn("runtime", plugin)


if __name__ == "__main__":
    unittest.main()
