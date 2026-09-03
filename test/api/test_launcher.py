import asyncio
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.launcher import (
    ASSET_DIRECTORY,
    MODELSCOPE_PROGRESS_PREFIX,
    LauncherError,
    LauncherRuntime,
    _launcher_access_addresses,
    _parse_modelscope_download_progress,
    create_launcher_app,
    fastllm_launcher,
)
from fastllm_pytools.tui import DeployConfig


def unused_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class LauncherConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp.name, "model")
        self.draft_path = os.path.join(self.temp.name, "draft")
        os.makedirs(self.model_path)
        os.makedirs(self.draft_path)
        self.config_path = os.path.join(self.temp.name, "launcher.json")
        self.runtime = LauncherRuntime(self.config_path)

    def tearDown(self):
        self.runtime.close()
        self.temp.cleanup()

    def config(self, **changes):
        payload = {
            "name": "Qwen launcher",
            "command": "server",
            "model": self.model_path,
            "model_name": "qwen-local",
            "host": "127.0.0.1",
            "port": "18080",
            "device": "cpu",
            "gpu_mem_ratio": "0.9",
        }
        payload.update(changes)
        return payload

    def test_preview_reuses_tui_command_builder(self):
        preview = self.runtime.preview(self.config(
            speculative_algorithm="dflash",
            speculative_draft_model_path=self.draft_path,
            draft_tokens="7",
        ))

        self.assertEqual(preview["errors"], [])
        self.assertIn("ftllm server", preview["command"])
        self.assertIn("--speculative_algorithm dflash", preview["command"])
        self.assertIn("--draft_tokens 7", preview["command"])
        self.assertEqual(preview["endpoint"], "http://127.0.0.1:18080")

    def test_profile_round_trip_uses_tui_config_file(self):
        saved = self.runtime.save_profile(None, self.config())
        self.assertEqual(saved["index"], 0)
        self.assertEqual(self.runtime.profiles()[0]["model_name"], "qwen-local")

        with open(self.config_path, "r", encoding="utf-8") as config_file:
            on_disk = json.load(config_file)
        self.assertEqual(on_disk["version"], 1)
        self.assertEqual(on_disk["commands"][0]["command"], "server")

        result = self.runtime.delete_profile(0)
        self.assertEqual(result["profiles"], [])

    def test_invalid_dflash_configuration_is_reported(self):
        preview = self.runtime.preview(self.config(
            speculative_algorithm="dflash",
            speculative_draft_model_path="",
        ))

        self.assertTrue(any("DFlash2" in error for error in preview["errors"]))

    def test_dflash_draft_path_must_be_a_directory(self):
        draft_file = os.path.join(self.temp.name, "draft.safetensors")
        with open(draft_file, "wb") as file:
            file.write(b"")

        preview = self.runtime.preview(self.config(
            speculative_algorithm="dflash",
            speculative_draft_model_path=draft_file,
        ))

        self.assertTrue(any("必须是目录" in error for error in preview["errors"]))

    def test_embedded_mtp_requires_positive_token_count(self):
        preview = self.runtime.preview(self.config(
            speculative_algorithm="mtp",
            speculative_draft_model_path="",
            mtp="0",
        ))

        self.assertTrue(any("内置 MTP" in error for error in preview["errors"]))

    def test_mtp_token_counts_must_match(self):
        preview = self.runtime.preview(self.config(
            speculative_algorithm="mtp",
            speculative_draft_model_path=self.draft_path,
            mtp="4",
            draft_tokens="5",
        ))

        self.assertTrue(any("必须一致" in error for error in preview["errors"]))

    def test_api_key_is_redacted_from_preview(self):
        preview = self.runtime.preview(self.config(api_key="local-secret"))

        self.assertNotIn("local-secret", preview["command"])
        self.assertIn("••••••••", preview["command"])

    def test_api_key_in_extra_arguments_is_redacted_from_preview(self):
        preview = self.runtime.preview(self.config(
            extra_args="--api_key=extra-secret",
        ))

        self.assertNotIn("extra-secret", preview["command"])
        self.assertIn("--api_key=••••••••", preview["command"])

    def test_ipv6_wildcard_uses_ipv6_loopback_for_display(self):
        preview = self.runtime.preview(self.config(host="::"))

        self.assertEqual(preview["endpoint"], "http://[::1]:18080")

    def test_wildcard_launcher_addresses_are_grouped_by_network_scope(self):
        addresses = _launcher_access_addresses(
            "0.0.0.0",
            8000,
            ["8.8.8.8", "10.20.30.40", "fc00::10"],
        )

        self.assertEqual(
            addresses,
            [
                {
                    "scope": "local",
                    "label": "本机地址",
                    "url": "http://127.0.0.1:8000",
                },
                {
                    "scope": "lan",
                    "label": "局域网地址",
                    "url": "http://10.20.30.40:8000",
                },
                {
                    "scope": "public",
                    "label": "公网地址",
                    "url": "http://8.8.8.8:8000",
                },
            ],
        )

    def test_ipv6_wildcard_only_advertises_ipv6_addresses(self):
        addresses = _launcher_access_addresses(
            "::",
            8000,
            ["10.20.30.40", "fc00::10", "2001:4860:4860::8888"],
        )

        self.assertEqual(
            [address["scope"] for address in addresses],
            ["local", "lan", "public"],
        )
        self.assertEqual(addresses[0]["url"], "http://[::1]:8000")
        self.assertEqual(addresses[1]["url"], "http://[fc00::10]:8000")

    def test_json_scalar_types_are_normalized(self):
        preview = self.runtime.preview(self.config(
            enable_moe_hybrid="false",
            port=18080,
        ))

        self.assertNotIn("--moe_device", preview["command"])
        self.assertEqual(preview["endpoint"], "http://127.0.0.1:18080")

    def test_webui_profile_builds_webui_command(self):
        preview = self.runtime.preview(self.config(
            command="webui",
            port="1616",
            webui_max_token="8192",
            webui_think="true",
        ))

        self.assertEqual(preview["errors"], [])
        self.assertIn("ftllm webui", preview["command"])
        self.assertIn("--max_token 8192", preview["command"])
        self.assertIn("--think true", preview["command"])
        self.assertNotIn("--model_name", preview["command"])
        self.assertNotIn("--host", preview["command"])
        self.assertEqual(preview["endpoint"], "http://127.0.0.1:1616")

    def test_download_preview_hides_token(self):
        payload = {
            "modelId": "Qwen/Qwen3-0.6B",
            "targetDir": os.path.join(self.temp.name, "downloads", "qwen"),
            "revision": "master",
            "maxWorkers": "4",
            "token": "private-modelscope-token",
        }
        with patch(
            "fastllm_pytools.launcher.importlib.util.find_spec",
            return_value=object(),
        ):
            preview = self.runtime.preview_download(payload)

        self.assertEqual(preview["errors"], [])
        self.assertIn("modelscope download", preview["command"])
        self.assertNotIn("private-modelscope-token", preview["command"])
        self.assertIn("环境变量", preview["command"])

    def test_download_progress_parser_rejects_invalid_events(self):
        valid = MODELSCOPE_PROGRESS_PREFIX + json.dumps({
            "version": 1,
            "type": "download.progress",
            "downloadedBytes": 650,
            "totalBytes": 1000,
            "completedFiles": 1,
            "totalFiles": 2,
        })
        self.assertEqual(
            _parse_modelscope_download_progress(valid)["downloadedBytes"],
            650,
        )
        self.assertIsNone(_parse_modelscope_download_progress("Progress 65%"))
        self.assertIsNone(_parse_modelscope_download_progress(
            MODELSCOPE_PROGRESS_PREFIX + '{"version": 2}'
        ))

    def test_occupied_server_port_is_rejected_before_spawn(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(LauncherError, "已被占用"):
                self.runtime.start(self.config(port=str(port)))

    def test_launcher_assets_are_packaged_as_external_resources(self):
        for filename in ("index.html", "styles.css", "app.js"):
            self.assertTrue((ASSET_DIRECTORY / filename).is_file(), filename)
        html = (ASSET_DIRECTORY / "index.html").read_text(encoding="utf-8")
        javascript = (ASSET_DIRECTORY / "app.js").read_text(encoding="utf-8")
        self.assertIn('src="/assets/app.js"', html)
        self.assertNotIn("<script>", html)
        self.assertRegex(
            html,
            r'id="server-api-key-field" class="field">\s*<span>API Key</span>',
        )

        cached_ids_source = javascript.split("const ids = [", 1)[1].split(
            "];", 1
        )[0]
        cached_ids = set(re.findall(r'"([a-z0-9-]+)"', cached_ids_source))
        html_ids = set(re.findall(r'\bid="([^"]+)"', html))
        self.assertEqual(cached_ids - html_ids, set())

        form_fields = re.findall(r'\bdata-field="([^"]+)"', html)
        self.assertEqual(len(form_fields), len(set(form_fields)))
        self.assertEqual(
            set(form_fields) - set(DeployConfig.__dataclass_fields__),
            set(),
        )

    def test_control_api_requires_launcher_token(self):
        from fastapi.testclient import TestClient

        addresses = [{
            "scope": "local",
            "label": "本机地址",
            "url": "http://127.0.0.1:8000",
        }]
        client = TestClient(create_launcher_app(
            self.runtime,
            "control-secret",
            addresses,
        ))
        denied = client.get("/api/bootstrap")
        allowed = client.get(
            "/api/bootstrap",
            headers={"X-FTLLM-Launcher-Token": "control-secret"},
        )

        self.assertEqual(denied.status_code, 403)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.json()["launcherAddresses"], addresses)
        self.assertEqual(allowed.headers["cache-control"], "no-store")

    def test_control_api_rejects_an_empty_server_token(self):
        with self.assertRaisesRegex(LauncherError, "令牌不能为空"):
            create_launcher_app(self.runtime, "")

    def test_control_api_rejects_invalid_json(self):
        from fastapi.testclient import TestClient

        client = TestClient(create_launcher_app(self.runtime, "control-secret"))
        response = client.post(
            "/api/preview",
            content="{",
            headers={"X-FTLLM-Launcher-Token": "control-secret"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON", response.json()["error"])


class LauncherProcessTest(unittest.TestCase):
    def test_launcher_opens_browser_after_startup_by_default(self):
        import uvicorn

        class InlineThread:
            def __init__(self, target, args=(), **_kwargs):
                self.target = target
                self.args = args

            def start(self):
                self.target(*self.args)

        async def fake_startup(server, sockets=None):
            server.started = True

        def fake_run(server):
            asyncio.run(server.startup())

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                host="127.0.0.1",
                port=8000,
                no_browser=False,
                config=os.path.join(directory, "launcher.json"),
            )
            with (
                patch.object(uvicorn.Server, "startup", fake_startup),
                patch.object(uvicorn.Server, "run", fake_run),
                patch(
                    "fastllm_pytools.launcher.threading.Thread",
                    InlineThread,
                ),
                patch(
                    "fastllm_pytools.launcher.webbrowser.open",
                    return_value=True,
                ) as browser_open,
                redirect_stdout(io.StringIO()),
            ):
                result = fastllm_launcher(args)

        self.assertEqual(result, 0)
        browser_open.assert_called_once()
        self.assertRegex(
            browser_open.call_args.args[0],
            r"^http://127\.0\.0\.1:8000/\?token=.+$",
        )

    def test_concurrent_start_only_spawns_one_process(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model")
            os.makedirs(model_path)
            entered_popen = threading.Event()
            release_popen = threading.Event()
            spawn_count = 0

            def delayed_popen(*args, **kwargs):
                nonlocal spawn_count
                spawn_count += 1
                entered_popen.set()
                release_popen.wait(timeout=2)
                return subprocess.Popen(*args, **kwargs)

            runtime = LauncherRuntime(
                os.path.join(directory, "config.json"),
                popen_factory=delayed_popen,
            )
            fake_argv = [
                sys.executable, "-u", "-c", "import time;time.sleep(30)"
            ]
            payload = {
                "name": "concurrent start",
                "command": "server",
                "model": model_path,
                "host": "127.0.0.1",
                "port": str(unused_tcp_port()),
                "device": "cpu",
                "gpu_mem_ratio": "0.9",
            }
            outcomes = []

            def start_runtime():
                try:
                    outcomes.append(runtime.start(payload)["phase"])
                except LauncherError as error:
                    outcomes.append(str(error))

            try:
                with patch(
                    "fastllm_pytools.launcher.build_fastllm_argv",
                    return_value=fake_argv,
                ):
                    first = threading.Thread(target=start_runtime)
                    second = threading.Thread(target=start_runtime)
                    first.start()
                    self.assertTrue(entered_popen.wait(timeout=2))
                    second.start()
                    release_popen.set()
                    first.join(timeout=3)
                    second.join(timeout=3)

                self.assertEqual(spawn_count, 1)
                self.assertEqual(len(outcomes), 2)
                self.assertTrue(any("已有模型服务" in item for item in outcomes))
            finally:
                release_popen.set()
                runtime.close()

    def test_progress_event_and_process_tree_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model")
            os.makedirs(model_path)
            runtime = LauncherRuntime(os.path.join(directory, "config.json"))
            event = {
                "type": "startup.ready",
                "stage": "ready",
                "percent": 100,
                "message": "API server is ready",
            }
            script = (
                "import json,sys,time;"
                f"print('FTLLM_PROGRESS '+json.dumps({event!r}),file=sys.stderr,flush=True);"
                "time.sleep(30)"
            )
            fake_argv = [sys.executable, "-u", "-c", script]
            payload = {
                "name": "process test",
                "command": "server",
                "model": model_path,
                "host": "127.0.0.1",
                "port": str(unused_tcp_port()),
                "device": "cpu",
                "gpu_mem_ratio": "0.9",
            }
            try:
                with patch(
                    "fastllm_pytools.launcher.build_fastllm_argv",
                    return_value=fake_argv,
                ):
                    started = runtime.start(payload)
                self.assertEqual(started["phase"], "starting")
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if runtime.state()["phase"] == "running":
                        break
                    time.sleep(0.02)
                self.assertEqual(runtime.state()["phase"], "running")
                self.assertTrue(runtime.state()["ready"])

                runtime.stop()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and runtime.state()["pid"]:
                    time.sleep(0.02)
                self.assertEqual(runtime.state()["phase"], "stopped")
                self.assertIsNone(runtime.state()["pid"])
            finally:
                runtime.close()

    def test_webui_start_does_not_add_server_progress_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model")
            os.makedirs(model_path)
            runtime = LauncherRuntime(os.path.join(directory, "config.json"))
            fake_argv = [sys.executable, "-u", "-c", "import time;time.sleep(30)"]
            payload = {
                "name": "webui process test",
                "command": "webui",
                "model": model_path,
                "port": str(unused_tcp_port()),
                "device": "cpu",
                "gpu_mem_ratio": "0.9",
            }
            try:
                with patch(
                    "fastllm_pytools.launcher.build_fastllm_argv",
                    return_value=fake_argv,
                ):
                    started = runtime.start(payload)
                self.assertEqual(started["command"], "webui")
                self.assertNotIn("--startup-progress", fake_argv)
            finally:
                runtime.close()

    def test_download_progress_and_process_tree_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LauncherRuntime(os.path.join(directory, "config.json"))
            event = {
                "version": 1,
                "type": "download.progress",
                "downloadedBytes": 650,
                "totalBytes": 1000,
                "completedFiles": 1,
                "totalFiles": 2,
            }
            script = (
                "import json,time;"
                f"print({MODELSCOPE_PROGRESS_PREFIX!r}+json.dumps({event!r}),flush=True);"
                "time.sleep(30)"
            )
            fake_argv = [sys.executable, "-u", "-c", script]
            payload = {
                "modelId": "Qwen/Qwen3-0.6B",
                "targetDir": os.path.join(directory, "downloaded-model"),
                "revision": "master",
                "maxWorkers": 4,
                "token": "memory-only-token",
            }
            try:
                with (
                    patch(
                        "fastllm_pytools.launcher.importlib.util.find_spec",
                        return_value=object(),
                    ),
                    patch(
                        "fastllm_pytools.launcher._download_argv",
                        return_value=fake_argv,
                    ),
                ):
                    runtime.start_download(payload)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if runtime.download_state()["progress"] >= 65:
                        break
                    time.sleep(0.02)
                self.assertEqual(runtime.download_state()["progress"], 65.0)
                self.assertNotIn("memory-only-token", json.dumps(runtime.logs()))

                runtime.stop_download()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if runtime.download_state()["phase"] == "cancelled":
                        break
                    time.sleep(0.02)
                self.assertEqual(runtime.download_state()["phase"], "cancelled")
                self.assertIsNone(runtime.download_state()["pid"])
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
