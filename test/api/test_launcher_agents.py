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
from fastllm_pytools.launcher_codex import CodexRuntime
from fastllm_pytools.launcher_claude import ClaudeRuntime
from fastllm_pytools.launcher_opencode import OpenCodeRuntime
from fastllm_pytools import launcher_agent_install as installer


SERVICE = {"sessionId":"model-a", "modelName":"local-model", "endpoint":"http://127.0.0.1:18001",
           "contextWindowTokens":32768}

FAKE_CODEX = '''import json, sys, os
from pathlib import Path
Path(os.environ['CODEX_HOME'], 'probe.json').write_text(json.dumps({'args':sys.argv, 'cwd':os.getcwd()}))
def send(message):
    print(json.dumps(message), flush=True)
def event(method, **params):
    send({'method':method, 'params':dict(threadId='thread-a', **params)})
thread = {'id':'thread-a', 'name':'Saved session', 'cwd':os.getcwd(), 'turns':[], 'updatedAt':1}
for line in sys.stdin:
    request=json.loads(line)
    method=request.get('method')
    if method == 'initialized': continue
    result={}
    if method in ('thread/start','thread/resume','thread/read'): result={'thread':thread}
    if method == 'thread/list': result={'data':[thread], 'nextCursor':None}
    if method == 'turn/start':
        turn={'id':'turn-a', 'items':[], 'status':'inProgress'}
        event('turn/started',turn=turn)
        event('item/started',turnId='turn-a',item={'id':'message-a','type':'agentMessage','text':''})
        event('item/agentMessage/delta',turnId='turn-a',itemId='message-a',delta='Hello')
        send({'id':'approval-a','method':'item/commandExecution/requestApproval',
              'params':{'threadId':'thread-a','turnId':'turn-a','itemId':'command-a','command':'echo hello'}})
        result={'turn':turn}
    if method == 'turn/interrupt' or (method is None and request.get('id') == 'approval-a'):
        event('serverRequest/resolved',requestId='approval-a')
        event('item/completed',turnId='turn-a',item={'id':'message-a','type':'agentMessage','text':'Hello complete'})
        turn={'id':'turn-a','items':[{'id':'message-a','type':'agentMessage','text':'Hello complete'}],'status':'completed'}
        thread['turns']=[turn]
        event('turn/completed',turn=turn)
        if method is None: continue
    send({'id':request['id'],'result':result})
'''


class AgentRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def wait(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(.02)
        self.fail("Timed out")

    def test_neither_agent_installs_on_open_and_cancelled_install_can_retry(self):
        for factory in (CodexRuntime, OpenCodeRuntime, ClaudeRuntime):
            runtime = factory(self.root / factory.__name__)
            self.addCleanup(runtime.stop)
            entered = threading.Event()

            def install(directory, agent, progress, cancelled):
                progress("download", 1024, 4096); entered.set()
                cancelled.wait(5)
                raise RuntimeError("cancelled")

            with patch.object(runtime, "_command", return_value=None), patch(
                    "fastllm_pytools.launcher_agent_runtime.install_runtime", side_effect=install) as installing:
                runtime.start(SERVICE, "key", "127.0.0.1", "http://localhost:8000")
                self.wait(lambda: runtime.state()["phase"] == "failed")
                installing.assert_not_called()
                for _ in range(2):
                    entered.clear()
                    runtime.start(SERVICE, "key", "127.0.0.1", "http://localhost:8000", install=True)
                    self.assertTrue(entered.wait(5))
                    state = runtime.state()
                    self.assertEqual((state["phase"], state["done"], state["total"]), ("installing", 1024, 4096))
                    runtime.stop()
                    self.assertFalse(runtime._thread.is_alive())
                    self.assertEqual(runtime.state()["phase"], "stopped")
                self.assertEqual(installing.call_count, 2)

    def test_codex_stdio_events_approval_resume_and_process_cleanup(self):
        runtime = CodexRuntime(self.root / "codex")
        self.addCleanup(runtime.stop)
        script = self.root / "fake.py"; script.write_text(FAKE_CODEX)
        with patch.object(runtime, "_command", return_value=[sys.executable, str(script)]):
            runtime.start(SERVICE, "private-model-key", "127.0.0.1", "https://localhost")
            self.wait(lambda: runtime.state()["phase"] == "running")
            process = runtime._process
            probe = json.loads((runtime.directory / "home/probe.json").read_text())
            self.assertNotIn("private-model-key", json.dumps(probe))
            self.assertIn('model_providers.fastllm.wire_api="responses"', probe["args"])
            self.assertIn('model_context_window=32768', probe["args"])
            self.assertEqual(probe["cwd"], str(runtime.directory / "workspace"))
            self.assertEqual(runtime.rpc("thread/start", {})["thread"]["id"], "thread-a")
            result = runtime.rpc("turn/start", {"threadId":"thread-a", "text":"Do it"})
            events = runtime.events(0, runtime.state()["epoch"])
            self.assertIn("thread-a", events["turns"])
            self.assertEqual(len(events["requests"]), 1)
            self.assertTrue(any(e["method"] == "item/agentMessage/delta" for e in events["events"]))
            with self.assertRaises(RuntimeError):
                runtime.respond("approval-a", {"decision":"acceptForSession"})
            runtime.respond("approval-a", {"decision":"accept"})
            self.wait(lambda: not runtime.events()["turns"])
            self.assertFalse(runtime.events()["requests"])
            resumed = runtime.rpc("thread/resume", {"threadId":"thread-a"})
            self.assertEqual(resumed["thread"]["turns"][0]["items"][0]["text"], "Hello complete")
            with self.assertRaises(RuntimeError):
                runtime.rpc("command/exec", {"command":"true"})
            with self.assertRaises(RuntimeError):
                runtime.rpc("thread/start", {"config":{}})
            runtime.stop()
            self.assertIsNotNone(process.poll())
            self.assertFalse(runtime._thread.is_alive())
            self.assertFalse(list(runtime.directory.glob("launch-*")))

    def test_event_gap_requires_persisted_history_reload(self):
        runtime = CodexRuntime(self.root / "codex")
        for index in range(4100):
            runtime._event({"method":"item/agentMessage/delta", "params":{"delta":str(index)}}, runtime._cancelled)
        response = runtime.events(0, "")
        self.assertTrue(response["reset"])
        self.assertFalse(response["events"])
        self.assertEqual(response["cursor"], 4100)

    def test_opencode_private_configuration_uses_chat_completions_without_compaction(self):
        config = OpenCodeRuntime.configuration(SERVICE)
        self.assertEqual(config["provider"]["fastllm"]["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(config["provider"]["fastllm"]["options"]["apiKey"], "{env:FTLLM_AGENT_API_KEY}")
        self.assertEqual(config["compaction"], {"auto":False, "prune":False})
        self.assertFalse(config["autoupdate"])
        runtime = OpenCodeRuntime(self.root / "opencode")
        with self.assertRaisesRegex(RuntimeError, "HTTP"):
            runtime.start(SERVICE, "", "127.0.0.1", "https://localhost")

    def test_installer_stages_pinned_packages_and_preserves_failed_previous_install(self):
        for agent in ("opencode", "codex"):
            directory = self.root / agent; directory.mkdir()
            previous = directory / "runtime"; previous.mkdir(); (previous / "keep").write_text("old")
            mode = "fail"

            def download(archive, *args): archive.touch()
            def extract(archive, target, *args):
                node = target / "bin/node"; node.parent.mkdir(parents=True); node.touch()
            def run(command, staging, environment, progress, cancelled, stage, **kwargs):
                if stage == "dependencies":
                    self.assertIn(installer.AGENTS[agent]["package"] + "@" + installer.AGENTS[agent]["version"], command)
                    entry = staging / "node_modules" / installer.AGENTS[agent]["entry"]
                    entry.parent.mkdir(parents=True); entry.touch()
                elif mode == "fail": raise RuntimeError("bad binary")
            with patch.object(installer, "_download_node", side_effect=download), patch.object(
                    installer, "_extract_node", side_effect=extract), patch.object(installer, "_run_command", side_effect=run):
                with self.assertRaisesRegex(RuntimeError, "bad binary"):
                    installer.install_runtime(directory, agent, lambda *args:None, threading.Event())
                self.assertTrue((previous / "keep").is_file())
                mode = "success"
                installer.install_runtime(directory, agent, lambda *args:None, threading.Event())
                self.assertIsNotNone(installer.runtime_command(previous, agent))
                self.assertFalse((previous / "keep").exists())


class AgentAPITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.runtime = LauncherRuntime(self.temp.name + "/profiles.json", plugins_dir=self.temp.name + "/plugins")
        self.addCleanup(self.runtime.close)
        self.client = TestClient(create_launcher_app(self.runtime, "control")); self.addCleanup(self.client.close)
        self.headers = {"X-FTLLM-Launcher-Token":"control"}

    def test_authentication_manual_install_and_plugin_lifecycle(self):
        for agent in ("opencode", "codex"):
            runtime = getattr(self.runtime, agent)
            for verb, suffix in (("GET", ""), ("POST", "/open"), ("POST", "/install"), ("POST", "/stop")):
                self.assertEqual(self.client.request(verb, "/api/agents/" + agent + suffix).status_code, 403)
            self.assertEqual(self.client.post(f"/api/agents/{agent}/open", headers=self.headers).status_code, 400)
            self.runtime._state.update(SERVICE, command="server", phase="running", ready=True)
            process = SimpleNamespace(poll=lambda:None); self.runtime._process = process
            try:
                with patch.object(runtime, "start", return_value={"phase":"starting"}) as start, patch.object(runtime, "stop") as stop:
                    for suffix, install in (("open", False), ("install", True)):
                        response = self.client.post(f"/api/agents/{agent}/{suffix}", headers=self.headers)
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(start.call_args.kwargs["install"], install)
                    self.runtime.plugins.set_enabled(agent, False)
                    stop.assert_called_once()
                    self.assertIs(self.runtime._process, process)
                    self.assertEqual(self.client.post(f"/api/agents/{agent}/install", headers=self.headers).status_code, 400)
                    start.reset_mock(); self.runtime.plugins.set_enabled(agent, True); start.assert_not_called()
            finally:
                self.runtime._process = None

    def test_plugin_catalog_omits_secrets_and_model_stop_stops_both_agents(self):
        with patch.object(self.runtime.opencode, "state", return_value={"phase":"running", "installed":True, "url":"secret"}):
            catalog = self.client.get("/api/plugins", headers=self.headers).json()
            self.assertNotIn("secret", json.dumps(catalog))
            for agent in ("opencode", "codex"):
                plugin = next(p for p in catalog["plugins"] if p["id"] == agent)
                self.assertTrue(plugin["native"]); self.assertTrue(plugin["builtin"])
        with patch.object(self.runtime.codex, "stop") as codex, patch.object(self.runtime.opencode, "stop") as opencode:
            self.runtime.stop(); codex.assert_called_once(); opencode.assert_called_once()

    def test_codex_rpc_validation_and_assets(self):
        for body in ([], {"method":[]}, {"method":"command/exec"}, {"method":"thread/start", "params":{"config":{}}}):
            result = self.client.post("/api/agents/codex/rpc", headers=self.headers, json=body)
            self.assertEqual(result.status_code, 400, result.text)
        for path in ("/plugin-core/native-agent.js", "/plugin-core/native-agent.css", "/plugin-core/runtime-manager.js", "/ui_plugins/codex/app.js"):
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_opencode_open_and_poll_use_the_current_browser_address(self):
        runtime = self.runtime.opencode = OpenCodeRuntime(Path(self.temp.name) / "opencode")
        script = Path(self.temp.name) / "opencode.py"
        script.write_text("import time\nprint('opencode server listening on http://127.0.0.1:9', flush=True)\ntime.sleep(60)\n")
        self.runtime._state.update(SERVICE, command="server", phase="running", ready=True)
        self.runtime._process = SimpleNamespace(poll=lambda:None)
        try:
            with patch.object(runtime, "_command", return_value=[sys.executable, str(script)]):
                first = {**self.headers, "host":"localhost:8000"}
                second = {**self.headers, "host":"192.0.2.10:8000"}
                self.assertEqual(self.client.post("/api/agents/opencode/open", headers=first).status_code, 200)
                deadline = time.monotonic() + 5
                while runtime.state()["phase"] not in {"running", "failed"} and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(runtime.state()["phase"], "running", runtime.state())
                process = runtime._process
                initial = self.client.get("/api/agents/opencode", headers=first).json()
                polled = self.client.get("/api/agents/opencode", headers=second).json()
                opened = self.client.post("/api/agents/opencode/open", headers=second).json()
                self.assertTrue(initial["url"].startswith("http://localhost:"))
                self.assertTrue(polled["url"].startswith("http://192.0.2.10:"))
                self.assertEqual(polled["url"], opened["url"])
                self.assertEqual(initial["epoch"], opened["epoch"])
                self.assertIs(runtime._process, process)
                self.assertEqual(self.client.get("/api/agents/opencode", headers=first).json()["url"], initial["url"])
        finally:
            self.runtime._process = None


if __name__ == "__main__":
    unittest.main()
