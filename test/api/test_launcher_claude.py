import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastllm_pytools.launcher import LauncherRuntime, create_launcher_app
from fastllm_pytools.launcher_claude import ClaudeRuntime
from test_launcher_agents import SERVICE


BRIDGE = Path(__file__).resolve().parents[2] / "tools/fastllm_pytools/plugin_assets/claude-bridge.mjs"
FAKE_SDK = r'''
import fs from "node:fs";
import path from "node:path";
const home = process.env.CLAUDE_CONFIG_DIR;
export async function getSessionInfo(id) {
  return fs.existsSync(path.join(home,id)) ? {sessionId:id} : undefined;
}
export function query({prompt,options:o}) {
  const id = o.resume || o.sessionId;
  fs.writeFileSync(path.join(home,id),"native session");
  fs.writeFileSync(path.join(home,"probe.json"),JSON.stringify({cwd:o.cwd,model:o.model,
    effort:o.effort,thinking:o.thinking,resume:o.resume,sessionId:o.sessionId,permissionMode:o.permissionMode,
    settingSources:o.settingSources,extraBody:JSON.parse(o.env.CLAUDE_CODE_EXTRA_BODY),base:process.env.ANTHROPIC_BASE_URL,token:process.env.ANTHROPIC_AUTH_TOKEN,
    config:home,defaultModel:process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL,foreign:process.env.CLAUDE_CODE_USE_BEDROCK}));
  const stream=(async function*() {
    if (prompt === "template failure") {
      yield {type:"result",is_error:true,errors:["API Error: 400 — Unsupported model template input"]};
      return;
    }
    const mid="message-"+Date.now();
    if (prompt.startsWith("multiple blocks")) {
      const partial = !prompt.includes("without partial events");
      const content = [
        {type:"redacted_thinking",data:"opaque"},
        {type:"thinking",thinking:"Check the project first."},
        {type:"text",text:"## Before\n\nI will write **example.txt**."},
        {type:"tool_use",id:"tool-multi",name:"Write",input:{file_path:"example.txt",content:"safe"}},
        {type:"text",text:"## After\n\nThe write has been requested."},
      ];
      if (partial) yield {type:"stream_event",event:{type:"message_start",message:{id:mid}}};
      for (const [index, block] of content.entries()) {
        if (partial) {
          const start = block.type === "text" ? {...block,text:""}
            : block.type === "thinking" ? {...block,thinking:""}
            : block.type === "tool_use" ? {...block,input:{}} : block;
          yield {type:"stream_event",event:{type:"content_block_start",index,content_block:start}};
          if (["text", "thinking"].includes(block.type)) {
            const value = block.text || block.thinking;
            for (const chunk of [value.slice(0,5), value.slice(5)])
              yield {type:"stream_event",event:{type:"content_block_delta",index,
                delta:block.type === "text" ? {type:"text_delta",text:chunk} : {type:"thinking_delta",thinking:chunk}}};
          }
        }
        // The native SDK emits one assistant frame per completed block,
        // before content_block_stop, reusing the same message id.
        yield {type:"assistant",message:{id:mid,content:[block]}};
        if (partial) yield {type:"stream_event",event:{type:"content_block_stop",index}};
      }
      const result=await o.canUseTool("Write",content[3].input,
        {signal:o.abortController.signal,toolUseID:"tool-multi",title:"Write example.txt?"});
      if (o.abortController.signal.aborted) throw new Error("aborted");
      yield {type:"user",message:{content:[{type:"tool_result",tool_use_id:"tool-multi",content:result.behavior}]}};
      yield {type:"result",is_error:false};
      return;
    }
    yield {type:"stream_event",event:{type:"message_start",message:{id:mid}}};
    yield {type:"stream_event",event:{type:"content_block_start",index:0,content_block:{type:"text",text:""}}};
    yield {type:"stream_event",event:{type:"content_block_delta",index:0,delta:{type:"text_delta",text:"## Reply\n\n**Hello**"}}};
    yield {type:"assistant",message:{id:mid,content:[{type:"text",text:"## Reply\n\n**Hello**"}]}};
    if (prompt.includes("question")) {
      const input={questions:[{question:"Which file?",options:[{label:"README.md",description:"Project notes"}]}]};
      const result=await o.canUseTool("AskUserQuestion",input,{signal:o.abortController.signal,toolUseID:"ask-1"});
      fs.writeFileSync(path.join(home,"answer.json"),JSON.stringify(result));
    } else {
      yield {type:"assistant",message:{id:mid+"-tool",content:[{type:"tool_use",id:"tool-1",name:"Write",input:{file_path:"example.txt",content:"safe"}}]}};
      const result=await o.canUseTool("Write",{file_path:"example.txt",content:"safe"},
        {signal:o.abortController.signal,toolUseID:"tool-1",title:"Write example.txt?"});
      fs.writeFileSync(path.join(home,"answer.json"),JSON.stringify(result));
      if (o.abortController.signal.aborted) throw new Error("aborted");
      yield {type:"user",message:{content:[{type:"tool_result",tool_use_id:"tool-1",content:result.behavior}]}};
    }
    yield {type:"result",is_error:false};
  })();
  stream.close=()=>{};
  return stream;
}
'''


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the SDK bridge test")
class ClaudeRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sdk = self.root / "sdk.mjs"
        sdk.write_text(FAKE_SDK)
        self.runtime = ClaudeRuntime(self.root / "claude")
        self.addCleanup(self.runtime.stop)
        command = patch.object(self.runtime, "_command", return_value=[shutil.which("node"), str(BRIDGE), str(sdk)])
        command.start(); self.addCleanup(command.stop)
        metadata = patch("fastllm_pytools.launcher_agent_runtime.with_model_metadata", side_effect=lambda service, key:service)
        metadata.start(); self.addCleanup(metadata.stop)
        self.service = dict(SERVICE, modelMetadata={"supported_reasoning_efforts":["low", "medium", "xhigh"]})

    def wait(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = predicate()
            if result: return result
            time.sleep(.02)
        self.fail(f"Timed out: {self.runtime.state()}")

    def start(self):
        self.runtime.start(self.service, "local-private-key", "127.0.0.1", "https://localhost")
        self.wait(lambda:self.runtime.state()["phase"] == "running")

    def turn(self, thread, text="Write a file", effort="low"):
        result = self.runtime.rpc("turn/start", {"threadId":thread["id"], "text":text, "effort":effort})
        request = self.wait(lambda:self.runtime.events()["requests"])[0]
        return result["turn"], request

    def test_sdk_stream_approval_cwd_effort_and_persistent_resume(self):
        with patch.dict(os.environ, {"CLAUDE_CODE_USE_BEDROCK":"1", "ANTHROPIC_BASE_URL":"http://wrong-provider"}):
            self.start()
        thread = self.runtime.rpc("thread/start", {"cwd":str(self.root)})["thread"]
        turn, request = self.turn(thread)
        self.assertIn("Write example.txt", request["params"]["reason"])
        live = self.runtime.rpc("thread/read", {"threadId":thread["id"]})["thread"]
        self.assertIn("**Hello**", live["turns"][0]["items"][1]["text"])
        probe = json.loads((self.runtime.directory / "home/probe.json").read_text())
        self.assertEqual((probe["cwd"], probe["model"], probe["effort"]), (str(self.root), SERVICE["modelName"], "low"))
        self.assertEqual(probe["base"], SERVICE["endpoint"])
        self.assertEqual(probe["token"], "local-private-key")
        self.assertEqual(probe["permissionMode"], "default")
        self.assertEqual(probe["settingSources"], [])
        self.assertNotIn("foreign", probe)
        self.assertEqual(probe["defaultModel"], SERVICE["modelName"])
        self.assertEqual(probe["extraBody"], {"thinking":{"type":"adaptive"}, "output_config":{"effort":"low"}})
        self.runtime.respond(request["id"], {"decision":"accept"})
        self.wait(lambda:not self.runtime.events()["turns"])
        self.runtime.stop(); self.start()
        restored = self.runtime.rpc("thread/resume", {"threadId":thread["id"]})
        self.assertEqual(restored["thread"]["cwd"], str(self.root))
        self.assertEqual(restored["thread"]["turns"][0]["status"], "completed")
        _, request = self.turn(thread, effort="xhigh")
        probe = json.loads((self.runtime.directory / "home/probe.json").read_text())
        self.assertEqual(probe["resume"], thread["id"])
        self.assertEqual(probe["effort"], "xhigh")
        self.runtime.respond(request["id"], {"decision":"decline"})
        self.wait(lambda:not self.runtime.events()["turns"])
        self.runtime.rpc("thread/name/set", {"threadId":thread["id"], "name":"My project"})
        self.assertEqual(self.runtime.rpc("thread/list", {"searchTerm":"My project"})["data"][0]["id"], thread["id"])
        self.runtime.rpc("thread/archive", {"threadId":thread["id"]})
        self.assertEqual(self.runtime.rpc("thread/list", {})["data"], [])

    def test_cancel_question_and_rpc_validation(self):
        self.start()
        thread = self.runtime.rpc("thread/start", {"cwd":str(self.root)})["thread"]
        for method, params in [("thread/start", {"cwd":"/directory-does-not-exist"}),
                               ("turn/start", {"threadId":thread["id"], "text":"x", "effort":"high"}),
                               ("thread/read", {"threadId":"../../outside"}),
                               ("turn/start", {"text":"x", "permissionMode":"bypassPermissions"})]:
            with self.subTest(method=method), self.assertRaises(RuntimeError):self.runtime.rpc(method, params)
        turn, request = self.turn(thread)
        with self.assertRaises(RuntimeError):self.runtime.respond(request["id"], {"decision":"acceptForSession"})
        self.runtime.rpc("turn/interrupt", {"threadId":thread["id"], "turnId":turn["id"]})
        self.wait(lambda:not self.runtime.events()["turns"])
        self.assertEqual(self.runtime.events()["requests"], [])
        _, request = self.turn(thread, "Ask a question")
        self.assertEqual(request["method"], "item/tool/requestUserInput")
        self.runtime.respond(request["id"], {"answers":{"0":{"answers":["README.md"]}}})
        self.wait(lambda:not self.runtime.events()["turns"])
        answer = json.loads((self.runtime.directory / "home/answer.json").read_text())
        self.assertEqual(answer["updatedInput"]["answers"], {"Which file?":"README.md"})

    def test_multiple_content_blocks_keep_stream_identity_and_persist_in_order(self):
        for partial in (True, False):
            with self.subTest(partial=partial):
                self.start()
                thread = self.runtime.rpc("thread/start", {"cwd":str(self.root)})["thread"]
                prompt = "multiple blocks" + ("" if partial else " without partial events")
                _, request = self.turn(thread, prompt)
                live = self.runtime.rpc("thread/read", {"threadId":thread["id"]})["thread"]["turns"][0]
                items = live["items"]
                self.assertEqual([item["type"] for item in items],
                    ["userMessage", "reasoning", "agentMessage", "mcpToolCall", "agentMessage"])
                self.assertEqual(len({item["id"] for item in items}), len(items))
                self.assertEqual(items[1]["content"], ["Check the project first."])
                self.assertEqual([item["text"] for item in items if item["type"] == "agentMessage"],
                    ["## Before\n\nI will write **example.txt**.", "## After\n\nThe write has been requested."])
                self.assertEqual(items[3]["arguments"], {"file_path":"example.txt", "content":"safe"})
                if partial:
                    events = self.runtime.events(epoch=self.runtime.state()["epoch"])["events"]
                    started = [event["params"]["item"] for event in events if event["method"] == "item/started"]
                    completed = [event["params"]["item"] for event in events
                        if event["method"] == "item/completed" and event["params"]["item"]["type"] != "userMessage"]
                    self.assertEqual([(item["id"], item["type"]) for item in started],
                                     [(item["id"], item["type"]) for item in completed])
                    self.assertEqual(completed, items[1:])
                self.runtime.respond(request["id"], {"decision":"accept"})
                self.wait(lambda:not self.runtime.events()["turns"])
                finished = self.runtime.rpc("thread/read", {"threadId":thread["id"]})["thread"]["turns"]
                self.assertEqual(finished[0]["status"], "completed")
                self.assertEqual(finished[0]["items"][3]["result"], "allow")
                self.runtime.stop(); self.start()
                restored = self.runtime.rpc("thread/resume", {"threadId":thread["id"]})["thread"]["turns"]
                self.assertEqual(restored, finished)
                self.runtime.stop()

    def test_launcher_routes_require_auth_and_enabled_builtin(self):
        launcher = LauncherRuntime(str(self.root / "profiles.json"), plugins_dir=str(self.root / "plugins"))
        self.addCleanup(launcher.close)
        launcher.claude = self.runtime
        self.start()
        client = TestClient(create_launcher_app(launcher, "launcher-key"))
        self.addCleanup(client.close)
        self.assertEqual(client.get("/api/agents/claude/events").status_code, 403)
        headers = {"X-FTLLM-Launcher-Token":"launcher-key"}
        result = client.post("/api/agents/claude/rpc", headers=headers,
                             json={"method":"thread/start", "params":{"cwd":str(self.root)}})
        self.assertEqual(result.status_code, 200, result.text)
        launcher.plugins.set_enabled("claude", False)
        self.assertEqual(self.runtime.state()["phase"], "stopped")
        self.assertNotEqual(client.post("/api/agents/claude/rpc", headers=headers,
            json={"method":"thread/list"}).status_code, 200)

    def test_missing_capabilities_preserve_model_defaults_and_shutdown_saves_history(self):
        self.service["modelMetadata"] = {}
        self.start()
        thread = self.runtime.rpc("thread/start", {"cwd":str(self.root)})["thread"]
        self.runtime.rpc("turn/start", {"threadId":thread["id"], "text":"Write a file"})
        self.wait(lambda:self.runtime.events()["requests"])
        probe = json.loads((self.runtime.directory / "home/probe.json").read_text())
        self.assertNotIn("effort", probe)
        self.assertEqual(probe["extraBody"], {"thinking":None, "output_config":{"effort":None}})
        self.runtime.stop(); self.start()
        restored = self.runtime.rpc("thread/read", {"threadId":thread["id"]})["thread"]
        self.assertEqual(restored["turns"][0]["status"], "interrupted")
        self.assertIn("**Hello**", restored["turns"][0]["items"][1]["text"])


if __name__ == "__main__":
    unittest.main()
