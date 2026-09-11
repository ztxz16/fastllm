import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastllm_pytools.launcher import LauncherRuntime, create_launcher_app
from fastllm_pytools.launcher_harness import HarnessRuntime
from fastllm_pytools.ui_plugins import PluginRegistry


FAKE_HARNESS = '''import json, os, sys, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
patch = json.loads(Path(sys.argv[sys.argv.index('--patch') + 1]).read_text())
home = Path(os.environ['DSH_HOME']); home.mkdir(exist_ok=True)
(home / 'probe.json').write_text(json.dumps({'patch':patch, 'cwd':os.getcwd()}))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        authority = '127.0.0.1:' + str(self.server.server_port)
        if self.headers.get('Host') != authority:
            self.send_response(403); self.end_headers(); return
        if self.path == '/?token=private-launch-token':
            self.send_response(303)
            self.send_header('Location', '/')
            self.send_header('Set-Cookie', 'harness-session=private-cookie; HttpOnly; SameSite=Strict; Path=/')
            self.end_headers(); return
        if self.headers.get('Cookie') != 'harness-session=private-cookie':
            self.send_response(401); self.end_headers(); return
        if self.path == '/events':
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            self.wfile.write(b'data: ready\\n\\n'); self.wfile.flush()
            time.sleep(60); return
        self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
        self.wfile.write(b'<textarea id="draft"></textarea><p id="ready">Harness ready</p>')
    def log_message(self, *args): pass
server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print('dsh web: http://127.0.0.1:' + str(server.server_port) + '/?token=private-launch-token', flush=True)
server.serve_forever()
'''


class HarnessRuntimeTest(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("FTLLM_HARNESS_EXPERIMENTAL_RECOVERY", None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.harness = HarnessRuntime(self.root)
        self.addCleanup(self.harness.stop)
        self.service = {"sessionId": "model-a", "modelName": "local-model",
                        "endpoint": "http://127.0.0.1:8001", "contextWindowTokens": 32768}

    def wait_phase(self, *phases):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            result = self.harness.state()
            if result["phase"] in phases:
                return result
            time.sleep(.02)
        self.fail(str(self.harness.state()))

    def test_model_configuration_and_process_lifecycle(self):
        script = self.root / "fake.py"
        script.write_text(FAKE_HARNESS)
        with patch.object(self.harness, "_command", return_value=[sys.executable, str(script)]):
            self.harness.start(self.service, "private-api-key", "127.0.0.1", "http://localhost:8000")
            result = self.wait_phase("running", "failed")
            self.assertEqual(result["phase"], "running", result)
            self.assertTrue(result["url"].startswith("http://localhost:"))
            self.assertNotIn("private-launch-token", result["url"])
            proxy = self.harness._proxy
            with TestClient(proxy._app(), base_url=proxy.origin) as browser:
                self.assertEqual(browser.get("/").status_code, 403)
                self.assertEqual(browser.get(result["url"]).status_code, 200)
            process = self.harness._process
            self.harness.start(self.service, "private-api-key", "127.0.0.1", "http://localhost:8000")
            self.assertIs(self.harness._process, process)
            probe = json.loads((self.root / "home/probe.json").read_text())
            config = next(p["config"] for p in probe["patch"] if p.get("id") == "llm-pi-ai")
            model = config["providers"]["fastllm"]
            self.assertEqual(model["baseURL"], "http://127.0.0.1:8001/v1")
            self.assertEqual(model["models"][0]["contextWindow"], 32768)
            self.assertEqual(model["compat"]["maxTokensField"], "max_tokens")
            self.assertFalse(model["compat"]["supportsDeveloperRole"])
            self.assertFalse(any(entry["id"] == "fastllm-harness-recovery"
                                 for row in probe["patch"] for entry in row.get("insert", [])))
            self.assertEqual(next(p["config"]["host"] for p in probe["patch"] if p.get("id") == "webserver"), "127.0.0.1")
            self.assertNotIn("private-api-key", json.dumps(probe))
            self.assertEqual(probe["cwd"], str(self.root / "workspace"))
            self.harness.stop()
            self.assertIsNotNone(process.poll())
            self.assertFalse(proxy.thread.is_alive())
            self.assertIsNone(self.harness._proxy)
            self.assertEqual(self.harness.state()["phase"], "stopped")
            self.assertTrue((self.root / "home/probe.json").exists())
            self.assertFalse(list(self.root.glob("launch-*")))
            retained = (self.root / "logs/latest.log").read_text()
            self.assertIn("dsh web:", retained)
            self.assertNotIn("private-launch-token", retained)

    def test_recovery_extension_requires_explicit_experimental_opt_in(self):
        script = self.root / "fake.py"
        script.write_text(FAKE_HARNESS)
        with patch.object(self.harness, "_command", return_value=[sys.executable, str(script)]):
            for setting in ("0", "false", "1"):
                with self.subTest(setting=setting), patch.dict(os.environ,
                        {"FTLLM_HARNESS_EXPERIMENTAL_RECOVERY": setting}):
                    self.harness.start(self.service, "", "127.0.0.1", "http://localhost:8000")
                    result = self.wait_phase("running", "failed")
                    self.assertEqual(result["phase"], "running", result)
                    probe = json.loads((self.root / "home/probe.json").read_text())
                    recovery = [entry for row in probe["patch"] for entry in row.get("insert", [])
                                if entry["id"] == "fastllm-harness-recovery"]
                    self.assertEqual(len(recovery), 1 if setting == "1" else 0)
                    if recovery:
                        self.assertTrue(Path(recovery[0]["name"]).is_file())
                        self.assertEqual(recovery[0]["config"], {"provider": "fastllm", "maxRecoveries": 2})
                    self.harness.stop()

    def test_startup_failure_redacts_tokens_and_model_key(self):
        script = self.root / "fail.py"
        script.write_text("print('error private-api-key http://localhost/?token=private-launch-token', flush=True)\n")
        with patch.object(self.harness, "_command", return_value=[sys.executable, str(script)]):
            self.harness.start(self.service, "private-api-key", "127.0.0.1", "http://localhost:8000")
            result = self.wait_phase("failed")
        self.assertNotIn("private-api-key", result["error"])
        self.assertNotIn("private-launch-token", result["error"])
        self.assertEqual(result["url"], "")
        retained = (self.root / "logs/latest.log").read_text()
        self.assertIn("error", retained)
        self.assertNotIn("private-api-key", retained)
        self.assertNotIn("private-launch-token", retained)

    def test_log_retention_is_bounded_private_and_redacted(self):
        source = self.root / "raw.log"
        source.write_text("x" * 70000 + "\nAuthorization: Bearer private-bearer\n"
                          "http://localhost/?token=private-token&api_key=private-query\nprivate-key\n")
        self.harness._save_log(source, "private-key")
        latest = self.root / "logs/latest.log"
        retained = latest.read_text()
        self.assertLess(latest.stat().st_size, 65536)
        for secret in ("private-bearer", "private-token", "private-query", "private-key"):
            self.assertNotIn(secret, retained)
        if os.name != "nt":
            self.assertEqual(latest.stat().st_mode & 0o777, 0o600)
        source.write_text("second run\n")
        self.harness._save_log(source, "")
        self.assertEqual(latest.read_text(), "second run\n")
        self.assertEqual((self.root / "logs/previous.log").read_text(), retained)

    def test_stop_finishes_active_harness_stream_before_closing_proxy(self):
        import httpx
        script = self.root / "stream.py"; script.write_text(FAKE_HARNESS)
        with patch.object(self.harness, "_command", return_value=[sys.executable, str(script)]):
            self.harness.start(self.service, "", "127.0.0.1", "http://localhost:8000")
            state = self.wait_phase("running", "failed")
            self.assertEqual(state["phase"], "running", state)
            proxy = self.harness._proxy
            with httpx.Client(trust_env=False, timeout=5, follow_redirects=True) as browser:
                self.assertEqual(browser.get(state["url"]).status_code, 200)
                with browser.stream("GET", proxy.origin + "/events") as response:
                    lines = response.iter_lines()
                    self.assertEqual(next(lines), "data: ready")
                    self.harness.stop()
                    self.assertEqual(list(lines), [""])
            self.assertFalse(proxy.thread.is_alive())
            self.assertFalse(self.harness._thread.is_alive())

    def test_stop_during_startup_cleans_up_the_child(self):
        script = self.root / "wait.py"
        script.write_text("import time\ntime.sleep(120)\n")
        with patch.object(self.harness, "_command", return_value=[sys.executable, str(script)]):
            self.harness.start(self.service, "", "127.0.0.1", "http://localhost:8000")
            deadline = time.monotonic() + 5
            while self.harness._process is None and time.monotonic() < deadline:
                time.sleep(.02)
            process = self.harness._process
            self.assertIsNotNone(process)
            self.harness.stop()
            self.assertIsNotNone(process.poll())
            self.assertEqual(self.harness.state()["phase"], "stopped")

    def test_installation_failure_and_unsupported_origin_report_errors(self):
        with patch("shutil.which", return_value=None), patch(
                "fastllm_pytools.launcher_harness.install_runtime", side_effect=RuntimeError("Download failed")):
            self.assertFalse(self.harness.state()["installed"])
            self.harness.start(self.service, "", "127.0.0.1", "http://localhost:8000", install=True)
            self.assertIn("Download failed", self.wait_phase("failed")["error"])
        with self.assertRaisesRegex(RuntimeError, "HTTP"):
            self.harness.start(self.service, "", "127.0.0.1", "https://localhost:8000")

    def test_open_does_not_install_without_explicit_install_request(self):
        with patch.object(self.harness, "_command", return_value=None), patch(
                "fastllm_pytools.launcher_harness.install_runtime") as installer:
            self.harness.start(self.service, "", "127.0.0.1", "http://localhost:8000")
            self.assertIn("Click Install and open Harness", self.wait_phase("failed")["error"])
            installer.assert_not_called()

    def test_explicit_install_installs_once_and_starts_automatically(self):
        release, entered = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        command = None
        script = self.root / "ready.py"
        script.write_text(FAKE_HARNESS)

        def install(directory, progress, cancelled):
            nonlocal command
            progress("download", 1024, 4096)
            entered.set()
            self.assertTrue(release.wait(5))
            command = [sys.executable, str(script)]

        with patch.object(self.harness, "_command", side_effect=lambda: command), patch(
                "fastllm_pytools.launcher_harness.install_runtime", side_effect=install) as installer:
            self.harness.start(self.service, "", "127.0.0.1", "http://localhost:8000", install=True)
            self.assertTrue(entered.wait(5))
            state = self.harness.state()
            self.assertEqual((state["phase"], state["stage"], state["done"], state["total"]),
                             ("installing", "download", 1024, 4096))
            self.harness.start(self.service, "", "127.0.0.1", "http://localhost:8000", install=True)
            installer.assert_called_once()
            release.set()
            self.assertEqual(self.wait_phase("running", "failed")["phase"], "running")
            self.harness.stop()
            self.harness.start(self.service, "", "127.0.0.1", "http://localhost:8000")
            self.assertEqual(self.wait_phase("running", "failed")["phase"], "running")
            installer.assert_called_once()

    def test_cancel_installation_does_not_start_harness_and_allows_retry(self):
        entered = threading.Event()

        def install(directory, progress, cancelled):
            progress("dependencies", 0, 0)
            entered.set()
            self.assertTrue(cancelled.wait(5))
            raise RuntimeError("cancelled")

        with patch.object(self.harness, "_command", return_value=None), patch(
                "fastllm_pytools.launcher_harness.install_runtime", side_effect=install) as installer:
            for _ in range(2):
                entered.clear()
                self.harness.start(self.service, "", "127.0.0.1", "http://localhost:8000", install=True)
                self.assertTrue(entered.wait(5))
                self.harness.stop()
                self.assertFalse(self.harness._thread.is_alive())
                self.assertIsNone(self.harness._process)
                self.assertEqual(self.harness.state()["phase"], "stopped")
            self.assertEqual(installer.call_count, 2)


class LauncherHarnessAPITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = LauncherRuntime(os.path.join(self.temp.name, "profiles.json"),
            plugins_dir=os.path.join(self.temp.name, "plugins"))
        self.addCleanup(self.runtime.close)
        self.client = TestClient(create_launcher_app(self.runtime, "control"))
        self.addCleanup(self.client.close)
        self.headers = {"X-FTLLM-Launcher-Token": "control"}

    def test_harness_endpoints_require_launcher_authentication_and_running_model(self):
        for method, path in (("GET", "/api/harness"), ("POST", "/api/harness/open"),
                             ("POST", "/api/harness/install"), ("POST", "/api/harness/stop")):
            self.assertEqual(self.client.request(method, path).status_code, 403)
        self.assertEqual(self.client.post("/api/harness/open", headers=self.headers).status_code, 400)
        self.assertEqual(self.client.post("/api/harness/install", headers=self.headers).status_code, 400)
        with patch.object(self.runtime.harness, "start", return_value={"phase": "starting"}) as start:
            self.runtime._state.update(command="server", phase="running", ready=True,
                                       modelName="local", sessionId="model-a", endpoint="http://127.0.0.1:8001")
            self.runtime._process = SimpleNamespace(poll=lambda: None)
            try:
                response = self.client.post("/api/harness/open", headers=self.headers)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(start.call_args.args[0]["modelName"], "local")
                self.assertFalse(start.call_args.kwargs["install"])
                response = self.client.post("/api/harness/install", headers=self.headers)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(start.call_args.kwargs["install"])
            finally:
                self.runtime._process = None

    def test_stopping_model_also_stops_harness(self):
        with patch.object(self.runtime.harness, "stop") as stop:
            self.runtime.stop()
            stop.assert_called_once()

    def test_harness_open_and_poll_support_new_browser_hostnames_without_restart(self):
        runtime = self.runtime.harness = HarnessRuntime(Path(self.temp.name) / "harness")
        script = Path(self.temp.name) / "harness.py"; script.write_text(FAKE_HARNESS)
        self.runtime._state.update(command="server", phase="running", ready=True, sessionId="model-a",
                                   modelName="local", endpoint="http://127.0.0.1:8001")
        self.runtime._process = SimpleNamespace(poll=lambda:None)
        try:
            with patch.object(runtime, "_command", return_value=[sys.executable, str(script)]):
                first = {**self.headers, "host":"localhost:8000"}
                second = {**self.headers, "host":"harness-alias.example:8000"}
                self.assertEqual(self.client.post("/api/harness/open", headers=first).status_code, 200)
                deadline = time.monotonic() + 5
                while runtime.state()["phase"] not in {"running", "failed"} and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(runtime.state()["phase"], "running", runtime.state())
                process, proxy = runtime._process, runtime._proxy
                original = self.client.get("/api/harness", headers=first).json()
                polled = self.client.get("/api/harness", headers=second).json()
                opened = self.client.post("/api/harness/open", headers=second).json()
                self.assertTrue(original["url"].startswith("http://localhost:"))
                self.assertTrue(polled["url"].startswith("http://harness-alias.example:"))
                self.assertEqual(polled["url"], opened["url"])
                with TestClient(proxy._app(), base_url=polled["url"].split("/_ftllm/")[0]) as browser:
                    self.assertEqual(browser.get("/").status_code, 403)
                    response = browser.get(polled["url"])
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("frame-ancestors http://harness-alias.example:8000", response.headers["content-security-policy"])
                    self.assertNotIn("private-cookie", response.headers.get("set-cookie", ""))
                self.assertEqual(self.client.get("/api/harness", headers=first).json()["url"], original["url"])
                self.assertIs(runtime._process, process)
                self.assertIs(runtime._proxy, proxy)
        finally:
            self.runtime._process = None

    def test_harness_uses_builtin_registry_and_disabling_blocks_launch_and_install(self):
        status = {"phase":"running", "installed":True, "url":"http://localhost/?token=private"}
        with patch.object(self.runtime.harness, "state", return_value=status), patch.object(
                self.runtime.harness, "stop") as stop, patch.object(self.runtime.harness, "start") as start:
            catalog = self.client.get("/api/plugins", headers=self.headers).json()
            plugin = next(p for p in catalog["plugins"] if p["id"] == "harness")
            self.assertTrue(plugin["builtin"])
            self.assertEqual(plugin["view"], "harness")
            self.assertEqual(plugin["runtime"]["phase"], "running")
            self.assertTrue(plugin["runtime"]["installed"])
            self.assertTrue(plugin["runtime"]["manageable"])
            self.assertNotIn("private", json.dumps(catalog))
            files = self.runtime.plugins.files("harness")
            self.assertEqual(set(files), {"plugin.json", "app.js", "page.html", "styles.css"})
            headers = {**self.headers, "X-FTLLM-Plugin-Request":"1"}
            response = self.client.post("/api/plugins/harness/enabled", headers=headers, json={"enabled":False})
            self.assertEqual(response.status_code, 200, response.text)
            stop.assert_called_once()
            self.assertFalse(PluginRegistry(self.runtime.plugins.directory).get("harness")["enabled"])
            for path in ("/api/harness/open", "/api/harness/install"):
                response = self.client.post(path, headers=self.headers)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("disabled", response.text)
            start.assert_not_called()
            self.runtime.plugins.set_enabled("harness", True)
            start.assert_not_called()
            stop.assert_called_once()

    def test_disabling_harness_cancels_installation_without_stopping_model(self):
        self.runtime.harness = HarnessRuntime(Path(self.temp.name) / "harness")
        entered = threading.Event()

        def install(directory, progress, cancelled):
            progress("dependencies", 0, 0)
            entered.set()
            self.assertTrue(cancelled.wait(5))
            raise RuntimeError("cancelled")

        self.runtime._state.update(command="server", phase="running", ready=True, sessionId="model-a",
                                   modelName="local", endpoint="http://127.0.0.1:8001")
        model_process = SimpleNamespace(poll=lambda: None)
        self.runtime._process = model_process
        try:
            with patch.object(self.runtime.harness, "_command", return_value=None), patch(
                    "fastllm_pytools.launcher_harness.install_runtime", side_effect=install):
                response = self.client.post("/api/harness/install", headers=self.headers)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(entered.wait(5))
                self.runtime.plugins.set_enabled("harness", False)
                self.assertEqual(self.runtime.harness.state()["phase"], "stopped")
                self.assertFalse(self.runtime.harness._thread.is_alive())
                self.assertIs(self.runtime._process, model_process)
        finally:
            self.runtime._process = None


if __name__ == "__main__":
    unittest.main()
