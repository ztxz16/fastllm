import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import agent_runtime_install as installer
from fastllm_pytools.launcher import LauncherRuntime, create_launcher_app


class RuntimeInstallTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wheelhouse = self.root / "wheels"
        self.wheelhouse.mkdir()
        env = patch.dict(os.environ, {
            "XDG_DATA_HOME": str(self.root / "data"),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_NO_INDEX": "1",
            "PIP_FIND_LINKS": str(self.wheelhouse),
            "PIP_CACHE_DIR": str(self.root / "cache"),
            "PIP_USER": "1",
        })
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(installer._forget_managed_module)

    def make_wheel(self):
        if importlib.util.find_spec("pip") is None:
            self.skipTest("pip is needed for the offline wheel-install integration test")
        source = Path(__file__).resolve().parents[2] / "tools/ftllm_agent_runtime/src/ftllm_agent_runtime"
        files = {"ftllm_agent_runtime/" + name: (source / name).read_bytes() for name in (
            "__init__.py", "runtime.py", "extensions/project_tools.ts",
        )}
        for name, version in (("pi", "0.84.4"), ("rg", "14.1.1"), ("fd", "10.2.0")):
            files["ftllm_agent_runtime/bin/" + name] = f"#!/bin/sh\necho {version}\n".encode()
        for name in ("package.json", "photon_rs_bg.wasm", "theme/dark.json",
                     "theme/light.json", "theme/theme-schema.json"):
            files["ftllm_agent_runtime/bin/" + name] = b"{}"
        metadata = f"ftllm_agent_runtime-{installer.RUNTIME_VERSION}.dist-info/"
        files[metadata + "METADATA"] = (
            f"Metadata-Version: 2.1\nName: ftllm-agent-runtime\nVersion: {installer.RUNTIME_VERSION}\n"
        ).encode()
        files[metadata + "WHEEL"] = b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-manylinux2014_x86_64\n"
        files[metadata + "RECORD"] = "".join(f"{name},,\n" for name in (*files, metadata + "RECORD")).encode()
        wheel = self.wheelhouse / f"ftllm_agent_runtime-{installer.RUNTIME_VERSION}-py3-none-manylinux2014_x86_64.whl"
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                info = zipfile.ZipInfo(name)
                info.external_attr = (0o100755 if name.endswith(("/bin/pi", "/bin/rg", "/bin/fd")) else 0o100644) << 16
                archive.writestr(info, data)
        return wheel

    def wait_job(self, job):
        deadline = time.monotonic() + 5
        while job.state()["phase"] in {"unchecked", "checking", "installing"}:
            if time.monotonic() > deadline:
                self.fail("installer job did not finish")
            time.sleep(.01)
        return job.state()

    def test_windows_preinstalled_runtime_uses_powershell_without_bash(self):
        runtime = SimpleNamespace(binary=self.root / "bin/pi.exe", info=lambda: {"pi_version": "0.84.4"})
        with patch.object(installer, "load_pi_agent_runtime", return_value=lambda **kw: runtime), \
                patch.object(installer.sys, "platform", "win32"), \
                patch.object(installer.shutil, "which", side_effect=lambda name, **kw: name if name != "bash" else None) as which:
            self.assertEqual(installer._runtime_info()["pi_version"], "0.84.4")
            self.assertIn("powershell", [call.args[0] for call in which.call_args_list])
            self.assertNotIn("bash", [call.args[0] for call in which.call_args_list])

    def test_windows_portable_runtime_is_ready_without_enabling_pip_installer(self):
        job = installer.AgentRuntimeInstaller()
        self.addCleanup(job.close)
        with patch.object(installer.sys, "platform", "win32"), \
                patch.object(installer, "_runtime_info", return_value={"pi_version": "0.84.4"}), \
                patch.object(installer, "install_runtime") as download:
            state = self.wait_job(job)
            self.assertEqual(state["phase"], "ready")
            self.assertTrue(state["available"])
            self.assertFalse(state["supported"])
            with self.assertRaisesRegex(RuntimeError, "Linux x86-64"):
                job.start()
            download.assert_not_called()

    def test_offline_pip_install_is_discoverable_now_and_after_restart(self):
        self.make_wheel()
        stages = []
        installer.install_runtime(lambda stage, *args: stages.append(stage), threading.Event())
        self.assertEqual(stages, ["pip", "verify"])
        package = installer.managed_package()
        self.assertIsNotNone(package)
        runtime = installer.load_pi_agent_runtime()(api_base="http://127.0.0.1:1", model="test")
        self.assertEqual(runtime.binary, package / "bin/pi")
        self.assertEqual(installer._runtime_info()["pi_version"], "0.84.4")
        self.assertFalse(list(installer.installation_root().parent.glob(".install-*")))
        script = "import sys; sys.path.insert(0, sys.argv[1]); from fastllm_pytools.agent_runtime_install import _runtime_info; print(_runtime_info()['pi_version'])"
        result = subprocess.check_output([sys.executable, "-c", script, str(Path(installer.__file__).parents[1])], text=True)
        self.assertEqual(result.strip(), "0.84.4")

    def test_pip_uses_launcher_python_and_preserves_configured_index(self):
        target = self.root / "target"
        with patch.dict(os.environ, {"PIP_INDEX_URL": "https://mirror.example/simple"}), \
                patch.object(installer, "_run_command") as run:
            installer._install_wheel(target, threading.Event())
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn(installer.PACKAGE_SPEC, command)
        self.assertEqual(command[command.index("--target") + 1], str(target))
        self.assertNotIn("--index-url", command)
        self.assertNotIn("--break-system-packages", command)
        self.assertEqual(run.call_args.kwargs["environment"]["PIP_INDEX_URL"], "https://mirror.example/simple")
        self.assertEqual(run.call_args.kwargs["environment"]["PIP_USER"], "0")

    def test_pip_failure_preserves_previous_installation(self):
        self.make_wheel().unlink()
        root = installer.installation_root()
        root.mkdir(parents=True)
        (root / "existing").write_text("previous")
        with self.assertRaisesRegex(RuntimeError, "pip install failed"):
            installer.install_runtime(lambda *args: None, threading.Event())
        self.assertEqual((root / "existing").read_text(), "previous")
        self.assertIsNone(installer.managed_package())
        self.assertFalse(list(root.parent.glob(".install-*")))

    def test_failed_verification_preserves_previous_installation(self):
        self.make_wheel()
        root = installer.installation_root()
        root.mkdir(parents=True)
        (root / "existing").write_text("previous")
        with patch.object(installer, "_verify_package", side_effect=RuntimeError("verification failed")):
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                installer.install_runtime(lambda *args: None, threading.Event())
        self.assertEqual((root / "existing").read_text(), "previous")
        self.assertIsNone(installer.managed_package())
        self.assertFalse(list(root.parent.glob(".install-*")))

    def test_missing_resources_do_not_count_as_installed(self):
        root = installer.installation_root()
        (root / "ftllm_agent_runtime/bin").mkdir(parents=True)
        (root / "ftllm_agent_runtime/bin/pi").write_text("incomplete")
        (root / "installation.json").write_text(json.dumps({"id": installer.INSTALLATION_ID}))
        self.assertIsNone(installer.managed_package())

    def test_external_companion_remains_usable(self):
        factory = object()
        with patch.object(installer.importlib, "import_module", return_value=SimpleNamespace(PiAgentRuntime=factory)) as load:
            self.assertIs(installer.load_pi_agent_runtime(), factory)
        load.assert_called_once_with("ftllm_agent_runtime")

    def test_pip_error_output_is_bounded_and_reported(self):
        script = "import sys; print('x' * 20000); print('pip test error'); sys.exit(1)"
        with self.assertRaisesRegex(RuntimeError, "pip test error") as raised:
            installer._run_command([sys.executable, "-c", script], threading.Event(), label="pip install")
        self.assertLess(len(str(raised.exception)), 8200)

    def test_cancellation_terminates_install_process(self):
        cancelled = threading.Event()
        pidfile = self.root / "pid"
        script = "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
        timer = threading.Timer(.5, cancelled.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            installer._run_command([sys.executable, "-c", script, str(pidfile)], cancelled, label="pip install")
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)

    def test_timeout_terminates_install_process(self):
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            installer._run_command([sys.executable, "-c", "import time; time.sleep(30)"],
                                   threading.Event(), label="pip install", timeout=.1)

    def test_other_launcher_installation_is_not_overwritten(self):
        import fcntl
        root = installer.installation_root()
        root.parent.mkdir(parents=True)
        with (root.parent / ".install.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(installer, "_install_wheel") as install:
                with self.assertRaisesRegex(RuntimeError, "Another Launcher"):
                    installer.install_runtime(lambda *args: None, threading.Event())
                install.assert_not_called()

    def test_duplicate_install_clicks_run_one_job_and_allow_retry(self):
        started, release = threading.Event(), threading.Event()
        calls, enabled = [], []

        def install(*args):
            calls.append(True)
            started.set()
            if not release.wait(3):
                raise RuntimeError("test timeout")
            if len(calls) == 1:
                raise RuntimeError("temporary download failure")

        job = installer.AgentRuntimeInstaller(lambda: enabled.append(True))
        self.addCleanup(job.close)
        with patch.object(installer, "install_runtime", side_effect=install), \
                patch.object(installer, "_runtime_info", return_value={"pi_version": "0.84.4"}):
            self.assertEqual(job.start()["phase"], "installing")
            self.assertTrue(started.wait(2))
            self.assertEqual(job.start()["phase"], "installing")
            self.assertEqual(len(calls), 1)
            release.set()
            self.assertEqual(self.wait_job(job)["phase"], "failed")
            job.start()
            self.assertEqual(self.wait_job(job)["phase"], "ready")
            self.assertEqual(len(calls), 2)
            self.assertEqual(enabled, [True])

    def test_unsupported_platform_does_not_download(self):
        job = installer.AgentRuntimeInstaller()
        self.addCleanup(job.close)
        with patch.object(installer.platform, "machine", return_value="aarch64"), \
                patch.object(installer, "install_runtime") as download:
            with self.assertRaisesRegex(RuntimeError, "Linux x86-64"):
                job.start()
            self.assertEqual(self.wait_job(job)["phase"], "unsupported")
            download.assert_not_called()

    def test_api_install_requires_control_token_and_ignores_custom_sources(self):
        from fastapi.testclient import TestClient
        runtime = LauncherRuntime(str(self.root / "launcher.json"))
        self.addCleanup(runtime.close)
        with TestClient(create_launcher_app(runtime, "test-token")) as client, \
                patch.object(runtime._agent_installer, "start", return_value={"phase": "installing"}) as start:
            self.assertEqual(client.post("/api/agent-runtime/install").status_code, 403)
            start.assert_not_called()
            response = client.post("/api/agent-runtime/install",
                                   headers={"X-FTLLM-Launcher-Token": "test-token"},
                                   json={"url": "https://example.com/untrusted.whl", "command": "anything"})
            self.assertEqual(response.status_code, 200)
            start.assert_called_once_with()

    def test_install_enables_existing_studio_without_closing_it(self):
        runtime = LauncherRuntime(str(self.root / "launcher.json"))
        self.addCleanup(runtime.close)
        enabled = []
        studio = SimpleNamespace(_pi_agent_lock=threading.Lock(), pi_agent="old", pi_agent_error="missing",
                                 _configure_pi_agent=lambda: enabled.append(True), close=lambda: None)
        app = SimpleNamespace(state=SimpleNamespace(runtime=studio))
        runtime._webui_app = app
        runtime._enable_installed_agent()
        self.assertIs(runtime._webui_app, app)
        self.assertIsNone(studio.pi_agent)
        self.assertEqual(studio.pi_agent_error, "")
        self.assertEqual(enabled, [True])


if __name__ == "__main__":
    unittest.main()
