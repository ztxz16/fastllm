"""Service isolation and generation workflows, using real local child servers."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from fastllm_pytools.launcher import LauncherError, LauncherRuntime, create_launcher_app
from fastllm_pytools.launcher_services import LauncherServicePool
from fastllm_pytools.media_agent import GenerationBridge
from fastllm_pytools.service_backends import visible_gpu_environment
from fastllm_pytools.tui import DeployConfig, build_fastllm_env, build_fastllm_argv, config_from_dict


WORKER = r'''
import argparse, json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
p=argparse.ArgumentParser()
p.add_argument('--port',type=int); p.add_argument('--host'); p.add_argument('--api-key',default='')
p.add_argument('--api_key',default=''); p.add_argument('--kind',default='video')
args,_=p.parse_known_args(); jobs={}; started=time.monotonic()
class H(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def answer(self,value,status=200,mime='application/json'):
  data=value if isinstance(value,bytes) else json.dumps(value).encode()
  self.send_response(status); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
 def auth(self):
  if self.headers.get('Authorization')!='Bearer '+(args.api_key or 'test-service-key'):
   self.answer({'error':{'message':'unauthorized'}},401); return False
  return True
 def do_GET(self):
  if self.path=='/health': return self.answer({'ready':time.monotonic()-started>float(os.environ.get('READY_DELAY','.4'))},200 if time.monotonic()-started>float(os.environ.get('READY_DELAY','.4')) else 503)
  if not self.auth(): return
  if self.path=='/v1/models': return self.answer({'data':[{'id':'stub-'+args.kind}], 'visible':os.environ.get('CUDA_VISIBLE_DEVICES')})
  if self.path=='/v1/videos': return self.answer({'data':list(jobs.values())})
  if self.path.endswith('/content'): return self.answer(b'fake-mp4',mime='video/mp4')
  return self.answer(jobs[self.path.split('/')[-1]])
 def do_POST(self):
  if not self.auth(): return
  if self.path.endswith('/cancel'):
   job=jobs[self.path.split('/')[-2]]; job['status']='cancelled'; return self.answer(job)
  data=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
  job=dict(data,id='video_1_'+str(len(jobs)),status='queued' if data['prompt']=='hold' else 'completed',created_at=time.time(),progress=100,stage='done',current=6,total=6,error=None)
  jobs[job['id']]=job; return self.answer(job)
ThreadingHTTPServer((args.host,args.port),H).serve_forever()
'''


def port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return str(listener.getsockname()[1])


class ServiceTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.worker = self.directory / "server.py"
        self.worker.write_text("#!" + sys.executable + "\n" + WORKER)
        self.worker.chmod(0o755)
        self.commands = []
        def popen(argv, **kwargs):
            self.commands.append((argv, kwargs["env"].copy()))
            if "-m" in argv:
                argv = [str(self.worker), "--kind", "llm"] + argv[4:]
            return subprocess.Popen(argv, **kwargs)
        runtime = LauncherRuntime(str(self.directory / "profiles.json"), popen_factory=popen,
            webui_history_dir=str(self.directory / "history"),
            agent_workspace_root=str(self.directory), plugins_dir=str(self.directory / "plugins"))
        self.pool = LauncherServicePool(runtime)
        self.addCleanup(self.pool.close)
        self.client = TestClient(create_launcher_app(self.pool, "control-key"))
        self.headers = {"X-FTLLM-Launcher-Token": "control-key"}

    def profile(self, backend="llm", **changes):
        config = dict(name=backend, model=str(self.directory), backend=backend, config_mode="auto",
            host="127.0.0.1", port=port(), api_key="test-service-key", visible_gpus="0",
            server_binary=str(self.worker), output_dir=str(self.directory / "output"))
        config.update(changes)
        return config

    def start(self, backend="llm", **changes):
        response = self.client.post("/api/runtime/start", headers=self.headers, json=self.profile(backend, **changes))
        self.assertEqual(response.status_code, 200, response.text)
        session = response.json()["sessionId"]
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            if self.pool.service(session).state()["ready"]:
                return session
            time.sleep(.05)
        self.fail(str(self.pool.state()) + str(self.pool.logs()))

    def test_gpu_mask_is_composed_and_auto_configuration_preserves_it(self):
        self.assertEqual(visible_gpu_environment("1,0", {"CUDA_VISIBLE_DEVICES": "6,7"}), {"CUDA_VISIBLE_DEVICES": "7,6"})
        self.assertEqual(visible_gpu_environment("1", {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}), {"CUDA_VISIBLE_DEVICES": "GPU-b"})
        for selection, parent in [("2", "6,7"), ("0", "-1"), ("GPU-c", "GPU-a,GPU-b"),
                                  ("0,0", "6,7"), ("0,00", "6,7"), ("1", "6,-1,7"),
                                  ("1", "6,,7"), ("1", "6,6,7")]:
            with self.assertRaises(ValueError):
                visible_gpu_environment(selection, {"CUDA_VISIBLE_DEVICES": parent})
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "6,7"}):
            config = config_from_dict(self.profile(visible_gpus="1"))
            self.assertIn("--auto", build_fastllm_argv(config))
            self.assertEqual(build_fastllm_env(config)["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(DeployConfig().backend, "llm")
        with self.assertRaises(ValueError):
            visible_gpu_environment("0,00", {})
        with patch("fastllm_pytools.service_backends.subprocess.run", return_value=SimpleNamespace(stdout="6, GPU-a\n7, GPU-b\n")):
            self.assertEqual(visible_gpu_environment("GPU-b", {"CUDA_VISIBLE_DEVICES": "6,7"}), {"CUDA_VISIBLE_DEVICES": "7"})
            with self.assertRaises(ValueError):
                visible_gpu_environment("6,GPU-a", {})
        config = config_from_dict(self.profile("minimax_h3", visible_gpus="0,1"))
        command = build_fastllm_argv(config)
        self.assertEqual(command[command.index("--devices") + 1], "0,1")

    def test_services_run_and_stop_independently_with_separate_gpu_environments(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "6,7"}):
            chat = self.start(visible_gpus="0")
            video = self.start("minimax_h3", visible_gpus="1")
        self.assertEqual([item[1]["CUDA_VISIBLE_DEVICES"] for item in self.commands], ["6", "7"])
        self.assertNotIn("--startup-progress", self.commands[1][0])
        self.assertEqual(len(self.pool.states()), 2)
        self.assertNotIn("test-service-key", json.dumps(self.pool.state()) + json.dumps(self.pool.logs()))
        opened = self.client.post("/api/webui/open", headers=self.headers, json={"sessionId": chat})
        self.assertEqual(opened.status_code, 200, opened.text)
        self.assertEqual(self.pool.state()["sessionId"], video)
        self.client.post("/api/services/" + video + "/stop", headers=self.headers)
        self.assertTrue(self.pool.service(chat).state()["ready"])
        self.assertEqual(self.client.get("/webui/" + chat + "/api/config").status_code, 200)

    def test_video_api_contract_authentication_and_scoped_proxy(self):
        started = time.monotonic()
        video = self.start("minimax_h3", env_vars="READY_DELAY=1.5")
        self.assertGreaterEqual(time.monotonic() - started, 1.5)
        info = self.client.get("/api/runtime/api-info", headers=self.headers).json()
        self.assertEqual(info["kind"], "video")
        self.assertEqual(info["request"]["model"], "minimax-h3")
        self.assertNotIn("test-service-key", json.dumps(self.pool.catalog()))
        path = "/api/media/" + video + "/videos"
        self.assertEqual(self.client.get(path).status_code, 403)
        created = self.client.post(path, headers=self.headers, json={"prompt": "hello", "size": "640x384"})
        self.assertEqual(created.status_code, 200, created.text)
        job = created.json()["id"]
        self.assertEqual(self.client.get(path + "/" + job, headers=self.headers).json()["status"], "completed")
        content = self.client.get(path + "/" + job + "/content", headers=self.headers)
        self.assertEqual(content.content, b"fake-mp4")
        self.assertEqual(content.headers["content-type"], "video/mp4")
        bad = self.client.get("/api/media/" + video + "/http://example.org/", headers=self.headers)
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.client.post("/api/runtime/start", headers=self.headers, json=[]).status_code, 400)
        # Equivalent port spellings must not bypass the starting-service guard.
        active_port = self.pool.state()["endpoint"].rsplit(":", 1)[1]
        with self.assertRaises(LauncherError):
            self.pool.start(self.profile(port="0" + active_port))

    def test_shared_history_excludes_concurrent_writes_but_stops_only_its_owner(self):
        first = self.start()
        second = self.start()
        a = self.pool.embedded_webui(first, "127.0.0.1").state.runtime
        b = self.pool.embedded_webui(second, "127.0.0.1").state.runtime
        conversation = a.store.create_conversation()
        self.assertEqual(a.store.database_path, b.store.database_path)
        control = a.begin_generation(conversation)
        self.assertTrue(b.is_generating(conversation))
        with self.assertRaises(RuntimeError):
            b.begin_generation(conversation)
        self.assertTrue(b.cancel_generation(conversation))
        self.assertTrue(control.cancelled)
        # Cancellation does not release the reservation until the writer exits.
        with self.assertRaises(RuntimeError):
            b.begin_generation(conversation)
        a._end_generation(conversation, control)
        replacement = b.begin_generation(conversation)
        a.close()
        self.assertFalse(replacement.cancelled)
        b._end_generation(conversation, replacement)
        self.assertFalse(b.is_generating(conversation))

    def test_media_request_rechecks_identity_after_service_slot_is_reused(self):
        session = self.start("minimax_h3")
        item = self.pool.service(session)
        def reused(_):
            with item._lock:
                item._state["sessionId"] = "replacement"
            return item
        with patch.object(self.pool, "service", side_effect=reused):
            with self.assertRaisesRegex(LauncherError, "服务已退出或更换"):
                self.pool.media_request(session, "POST", "videos", b'{"prompt":"wrong model"}')

    def test_stop_and_restart_are_serialized_and_closed_pool_rejects_start(self):
        session = self.start()
        item = self.pool.service(session)
        entered, release, starting = threading.Event(), threading.Event(), threading.Event()
        original_stop = item.stop
        def delayed_stop():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release stop")
            return original_stop()
        def restart():
            starting.set()
            return self.pool.start(self.profile())
        with ThreadPoolExecutor(max_workers=2) as executor, patch.object(item, "stop", side_effect=delayed_stop):
            stopping = executor.submit(self.pool.stop, session)
            self.assertTrue(entered.wait(3))
            restarting = executor.submit(restart)
            self.assertTrue(starting.wait(3))
            try:
                acquired = self.pool.lock.acquire(blocking=False)
                if acquired:
                    self.pool.lock.release()
                self.assertFalse(acquired)
            finally:
                release.set()
            stopping.result(timeout=5)
            replacement = restarting.result(timeout=5)
        self.assertNotEqual(replacement["sessionId"], session)
        self.pool.close()
        with self.assertRaises(LauncherError):
            self.pool.start(self.profile())

    def test_native_agent_installation_management_follows_active_owner(self):
        first = self.start()
        second = self.start()
        class Agent:
            _thread = None
            def __init__(self, phase):
                self.phase = phase
            def state(self):
                return {"phase": self.phase}
            def stop(self):
                self.phase = "stopped"
                return self.state()
            def manage(self, operation):
                self.phase = "upgrading"
                return self.state()
        a, b = Agent("running"), Agent("stopped")
        self.pool.service(first).codex = a
        self.pool.service(second).codex = b
        self.pool.plugins.manage_runtime("codex", "upgrade")
        self.assertEqual(a.phase, "stopped")
        self.pool.select(first)
        self.assertEqual(self.pool.plugins._runtimes["codex"]()["phase"], "upgrading")
        with self.assertRaises(LauncherError):
            self.pool.open_agent("codex", "127.0.0.1", "http://localhost")
        self.pool.plugins.manage_runtime("codex", "cancel")
        self.assertEqual(b.phase, "stopped")

    def test_agent_helper_saves_video_and_cancellation_only_affects_its_jobs(self):
        video = self.start("minimax_h3")
        downloads = []
        bridge = GenerationBridge(self.pool, lambda name, body: downloads.append((name, body)))
        self.addCleanup(bridge.close)
        command = [sys.executable, str(TOOLS / "fastllm_pytools/media_agent.py"), "--bridge", bridge.url, "--token", bridge.token]
        result = subprocess.run(command + ["generate", "--service", video, "--prompt", "hello", "--output", "videos/result.mp4", "--wait", "0"], cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.directory / "videos/result.mp4").read_bytes(), b"fake-mp4")
        self.assertEqual(downloads[0][1], b"fake-mp4")
        bad = subprocess.run(command + ["generate", "--service", video, "--prompt", "hello", "--output", "../escape.mp4"], cwd=self.directory, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        pending = subprocess.run(command + ["generate", "--service", video, "--prompt", "hold", "--output", "videos/hold.mp4", "--wait", "0"], cwd=self.directory, capture_output=True, text=True)
        job_id = json.loads(pending.stdout)["id"]
        bridge.close(cancel=True)
        status = json.loads(self.pool.media_request(video, "GET", "videos/" + job_id)[2])
        self.assertEqual(status["status"], "cancelled")

    def test_pi_workspace_calls_video_and_returns_a_downloadable_attachment(self):
        chat = self.start()
        self.start("minimax_h3")
        app = self.pool.embedded_webui(chat, "127.0.0.1")
        runtime = app.state.runtime
        class Pi:
            def stream(agent, **kwargs):
                command_line = kwargs["system_prompt"].split("工具命令前缀：\n", 1)[1].split("\n", 1)[0]
                command = shlex.split(command_line)
                catalog = json.loads(subprocess.check_output(command + ["catalog"]))
                result = subprocess.run(command + ["generate", "--service", catalog[0]["id"], "--prompt", "hello", "--output", "videos/agent.mp4", "--wait", "0"], cwd=kwargs["working_directory"], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                yield {"type": "text_delta", "delta": "Video saved to videos/agent.mp4"}
        runtime.pi_agent = Pi()
        runtime.pi_agent_error = ""
        conversation_id = runtime.store.create_conversation()
        messages = [{"role": "user", "content": "Generate a video"}]
        events = [json.loads(line) for line in runtime.generate_workspace_agent(conversation_id, messages,
            {"agent_workspace": str(self.directory)}, "Video")]
        final = events[-1]["message"]
        self.assertNotIn("error", final, str(events))
        self.assertEqual(final["attachments"][0]["kind"], "video")
        self.client.post("/api/webui/open", headers=self.headers, json={"sessionId": chat})
        response = self.client.get("/webui/" + chat + final["attachments"][0]["url"])
        self.assertEqual(response.content, b"fake-mp4")

    def test_browser_profiles_service_selection_and_video_demo(self):
        try:
            from playwright.sync_api import sync_playwright, expect
        except ImportError:
            self.skipTest("Playwright is optional")
        import uvicorn
        chat = self.start()
        video = self.start("minimax_h3")
        other_video = self.start("minimax_h3")
        server = uvicorn.Server(uvicorn.Config(create_launcher_app(self.pool, "control-key"), host="127.0.0.1", port=int(port()), log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True); thread.start()
        def stop_server():
            server.should_exit = True; thread.join(timeout=5)
        self.addCleanup(stop_server)
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.route("**/api/hardware*", lambda route: route.fulfill(json={"gpus": [
                {"index": "0", "name": "Test GPU A", "uuid": "GPU-a"},
                {"index": "1", "name": "Test GPU B", "uuid": "GPU-b"}]}))
            errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.config.port}/?token=control-key")
            page.locator("#language-select").select_option("en-US")
            page.locator("#new-profile").click()
            page.locator("#model-backend").select_option("minimax_h3")
            expect(page.locator('[data-field="server_binary"]')).to_be_visible()
            expect(page.locator('[data-field="visible_gpus"]')).to_be_visible()
            page.locator('#gpu-choices input[value="1"]').check()
            expect(page.locator('[data-field="visible_gpus"]')).to_have_value("1")
            expect(page.locator("#configuration-mode-settings")).not_to_be_visible()
            page.locator("#close-profile-editor").click()
            page.locator("#confirmation-confirm").click()
            page.locator("#service-selector").select_option(video)
            page.locator("#open-api-info").click()
            expect(page.locator("#api-contract")).to_have_value(re.compile(r"/v1/videos"))
            self.assertIn("/v1/videos", page.locator("#api-contract").input_value())
            page.locator("#api-info-close").click()
            page.locator("#video-service").select_option(other_video, force=True)
            page.locator("#open-endpoint").click()
            expect(page.locator("#video-service")).to_have_value(video)
            expect(page.locator("#video-studio")).to_be_visible()
            page.locator("#video-prompt").fill("A cat walking")
            page.locator("#video-generate").click()
            expect(page.locator(".video-job")).to_have_count(1)
            expect(page.locator(".video-job p")).to_have_text("A cat walking")
            page.get_by_role("button", name="Preview and download").click()
            expect(page.locator(".video-job video")).to_have_count(1)
            expect(page.get_by_role("link", name="Download video")).to_be_visible()
            page.locator("#video-prompt").fill("A second video")
            page.locator("#video-generate").click()
            expect(page.locator(".video-job")).to_have_count(2)
            expect(page.locator(".video-job p").first).to_have_text("A second video")
            page.locator("#language-select").select_option("zh-CN")
            expect(page.get_by_role("link", name="下载视频")).to_be_visible()
            expect(page.locator(".video-job video")).to_have_count(1)
            page.locator("#language-select").select_option("en-US")
            page.locator("#studio-chat-tab").click()
            expect(page.locator("#webui-content")).to_be_visible()
            self.assertEqual(page.locator("#studio-chat-select").input_value(), chat)
            page.locator("#studio-video-tab").click()
            if os.environ.get("FTLLM_MEDIA_SCREENSHOT"):
                page.screenshot(path=os.environ["FTLLM_MEDIA_SCREENSHOT"])
            self.assertEqual(errors, [])
            browser.close()


class BridgeLifecycleTest(unittest.TestCase):
    def test_cancel_catches_jobs_created_after_bridge_closes(self):
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        class Services:
            def media_request(self, service, method, path, body=None):
                if path.endswith("/cancel"):
                    cancelled.set()
                    return 200, "application/json", b'{}'
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release POST")
                return 200, "application/json", b'{"id":"late_job"}'
        bridge = GenerationBridge(Services())
        self.addCleanup(bridge.close)
        def create():
            request = urllib.request.Request(bridge.url + "/service/videos", data=b'{}',
                headers={"Authorization": "Bearer " + bridge.token})
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.load(response)
        with ThreadPoolExecutor(max_workers=1) as executor:
            created = executor.submit(create)
            try:
                self.assertTrue(entered.wait(3))
                bridge.close(cancel=True)
            finally:
                release.set()
            self.assertEqual(created.result(timeout=5)["id"], "late_job")
        self.assertTrue(cancelled.is_set())
