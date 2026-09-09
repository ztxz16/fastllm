import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastllm_pytools.ui_plugins import (
    BUNDLED_PLUGINS, PluginConflict, PluginError, PluginRegistry,
    install_plugin_routes, validate_bundle,
)


def bundle(plugin_id="monitor", text="hello", **manifest):
    return {"plugin.json": json.dumps({"apiVersion": 1, "id": plugin_id, "name": "Monitor",
                                      "entry": "index.html", "slot": "page", "capabilities": ["hardware.read"], **manifest}),
            "index.html": "<!doctype html><html><head></head><body><p>" + text + "</p></body></html>"}


def theme_bundle(plugin_id="blue-skin", **theme):
    return {"plugin.json": json.dumps({"apiVersion": 1, "id": plugin_id, "name": "Blue skin",
                                      "slot": "theme", "theme": theme or {"light": {"primary": "#2563eb"}}})}


class UIPluginTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "plugins"
        self.registry = PluginRegistry(self.root)
        self.model = Mock()
        self.model.complete.return_value = (json.dumps({"summary": "monitor", "files": bundle()}), "")
        self.hardware = Mock(return_value={"memory": {"total": 123, "available": 45}})
        app = FastAPI()
        install_plugin_routes(app, self.registry, lambda: self.model, hardware=self.hardware)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.headers = {"X-FTLLM-Plugin-Request": "1"}

    def post(self, path, payload):
        return self.client.post("/api/plugins/" + path, json=payload, headers=self.headers)

    def test_core_and_builtin_sources_are_read_only(self):
        original = (BUNDLED_PLUGINS / "models" / "plugin.json").read_bytes()
        with self.assertRaises(PluginError):
            self.registry.apply("models", bundle("models"), "builtin")
        with self.assertRaises(PluginError):
            self.registry.apply("harness", bundle("harness"), "builtin")
        with self.assertRaises(PluginError):
            self.registry.delete("harness", "builtin")
        with self.assertRaises(PluginError):
            PluginRegistry(BUNDLED_PLUGINS / "modified")
        self.assertEqual(original, (BUNDLED_PLUGINS / "models" / "plugin.json").read_bytes())

    def test_paths_native_code_and_unknown_capabilities_are_rejected(self):
        for name in ("../launcher.py", "/tmp/x.js", "C:/x.js", "a:stream.js", "a/../../x.js", "a\\x.js", ".hidden.js", "a//x.js", "evil.py"):
            with self.subTest(name=name), self.assertRaises(PluginError):
                validate_bundle("monitor", {**bundle(), name: "bad"})
        for changes in ({"native": True}, {"capabilities": ["shell"]}, {"slot": []}, {"entry": "missing.html"}):
            with self.subTest(changes=changes), self.assertRaises(PluginError):
                validate_bundle("monitor", bundle(**changes))
        for plugin_id in ("../core", "UPPER", "a/b", "", "a" * 49):
            with self.subTest(plugin_id=plugin_id), self.assertRaises(PluginError):
                self.registry.apply(plugin_id, bundle(plugin_id), "")
        self.assertFalse(self.root.exists())

    def test_discovery_hot_edits_bad_revision_and_removal(self):
        self.registry.apply("monitor", bundle(), "")
        first = self.registry.get("monitor")
        (self.root / "monitor/index.html").write_text("changed")
        second = self.registry.get("monitor")
        self.assertNotEqual(first["revision"], second["revision"])
        (self.root / "monitor/plugin.json").write_text("{")
        failed = self.registry.get("monitor")
        self.assertEqual(failed["revision"], second["revision"])
        self.assertTrue(failed["error"])
        self.assertEqual(self.registry.files("monitor")["index.html"], "changed")
        import shutil
        shutil.rmtree(self.root / "monitor")
        self.assertNotIn("monitor", [p["id"] for p in self.registry.list()["plugins"]])

    def test_shell_widgets_have_bounded_sizes_and_cannot_replace_pages(self):
        for slot in ("topbar", "sidebar", "statusbar"):
            manifest = validate_bundle("monitor", bundle(slot=slot, size={"width": 280, "height": 44}))
            self.assertEqual(manifest["slot"], slot)
            self.assertEqual(manifest["size"], {"width": 280, "height": 44})
            with self.assertRaises(PluginError):
                validate_bundle("monitor", bundle(slot=slot, replaces="launch"))
        for size in ({"width": 9999}, {"height": 0}, {"height": True}, {"position": "fixed"}, []):
            with self.subTest(size=size), self.assertRaises(PluginError):
                validate_bundle("monitor", bundle(slot="topbar", size=size))

    def test_theme_is_declarative_and_rejects_css_and_unsafe_layout(self):
        manifest = validate_bundle("blue-skin", theme_bundle(light={"primary": "#123"}, layout={"sidebarSide": "right", "radius": 0}))
        self.assertEqual(manifest["entry"], "")
        self.assertEqual(manifest["capabilities"], [])
        for theme in ({"light": {"primary": "url(https://example.com)"}},
                      {"light": {"primary": "#fff;display:none"}}, {"light": {"display": "#fff"}},
                      {"layout": {"sidebarWidth": 10000}}, {"layout": {"radius": True}},
                      {"layout": {"font": "url(x)"}}, {"layout": {"sidebarSide": []}},
                      {"css": "body{display:none}"}):
            with self.subTest(theme=theme), self.assertRaises(PluginError):
                validate_bundle("blue-skin", theme_bundle(**theme))
        with self.assertRaises(PluginError):
            validate_bundle("blue-skin", {**theme_bundle(), "app.js": "alert(1)"})
        with self.assertRaises(PluginError):
            validate_bundle("monitor", bundle(slot="theme", theme={"light": {"primary": "#123"}}))

    def test_theme_enable_is_exclusive_and_reset_preserves_widgets(self):
        self.registry.apply("monitor", bundle(slot="topbar"), "")
        first = self.registry.apply("blue-skin", theme_bundle(), "")
        self.registry.apply("red-skin", theme_bundle("red-skin", light={"primary": "#a00"}), "")
        self.assertFalse(self.registry.get("blue-skin")["enabled"])
        self.registry.set_enabled("blue-skin", True)
        self.assertFalse(self.registry.get("red-skin")["enabled"])
        second = self.registry.apply("blue-skin", theme_bundle(light={"primary": "#248"}), first["revision"])
        self.registry.rollback("blue-skin", second["revision"])
        self.assertEqual(self.registry.get("blue-skin")["revision"], first["revision"])
        self.assertEqual(self.post("reset-theme", {}).status_code, 200)
        other = PluginRegistry(self.root)
        self.assertFalse(other.get("blue-skin")["enabled"])
        self.assertFalse(other.get("red-skin")["enabled"])
        self.assertTrue(other.get("monitor")["enabled"])
        # Editing a disabled skin must not unexpectedly change the active skin.
        self.registry.apply("blue-skin", theme_bundle(light={"primary": "#369"}), first["revision"])
        self.assertFalse(other.get("blue-skin")["enabled"])

    def test_model_shell_target_is_enforced_before_publication(self):
        for slot, files in (("topbar", bundle(slot="topbar")), ("theme", theme_bundle("monitor"))):
            self.model.complete.return_value = (json.dumps({"files": files}), "")
            response = self.post("propose", {"id": "monitor", "instruction": "定制主界面", "target": slot})
            self.assertEqual(response.status_code, 200, response.text)
            request = json.loads(self.model.complete.call_args.args[0][1]["content"])
            self.assertEqual(request["target"], slot)
            self.assertFalse(self.root.exists())
        self.assertEqual(self.post("propose", {"id": "monitor", "instruction": "增加硬件栏", "target": "topbar"}).status_code, 400)
        self.assertEqual(self.post("propose", {"id": "monitor", "instruction": "定制", "target": []}).status_code, 400)

    def test_compare_and_swap_and_rollback_across_registry_instances(self):
        first = self.registry.apply("monitor", bundle(), "")
        other = PluginRegistry(self.root)
        second = other.apply("monitor", bundle(text="second"), first["revision"])
        with self.assertRaises(PluginConflict):
            self.registry.apply("monitor", bundle(text="stale"), first["revision"])
        restored = self.registry.rollback("monitor", second["revision"])
        self.assertEqual(restored["revision"], first["revision"])
        self.registry.set_enabled("monitor", False)
        self.assertFalse(PluginRegistry(self.root).get("monitor")["enabled"])

    def test_delete_removes_custom_files_history_and_cached_state(self):
        for files in (bundle(), bundle(slot="studio"), theme_bundle("monitor")):
            with self.subTest(slot=json.loads(files["plugin.json"])["slot"]):
                first = self.registry.apply("monitor", files, "")
                updated = {**files, "plugin.json": json.dumps({**json.loads(files["plugin.json"]), "name": "Updated"})}
                second = self.registry.apply("monitor", updated, first["revision"])
                self.registry.set_enabled("monitor", False)
                response = self.client.request("DELETE", "/api/plugins/monitor", headers=self.headers,
                                               json={"expectedRevision": second["revision"]})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse((self.root / "monitor").exists())
                self.assertFalse((self.root / ".history/monitor.json").exists())
                self.assertNotIn("monitor", self.registry._last_good)
                self.assertNotIn("monitor", self.registry._state())
                self.assertFalse(any(item["id"] == "monitor" for item in PluginRegistry(self.root).list()["plugins"]))
                with self.assertRaises(PluginError):
                    self.registry.rollback("monitor", "")

    def test_delete_rejects_stale_revision_builtin_and_untrusted_requests(self):
        first = self.registry.apply("monitor", bundle(), "")
        second = self.registry.apply("monitor", bundle(text="new"), first["revision"])
        response = self.client.request("DELETE", "/api/plugins/monitor", headers=self.headers,
                                       json={"expectedRevision": first["revision"]})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.registry.get("monitor")["revision"], second["revision"])
        self.assertEqual(self.client.request("DELETE", "/api/plugins/monitor", json={}).status_code, 403)
        builtin = next(item for item in self.registry.list()["plugins"] if item["builtin"])
        self.assertEqual(self.client.request("DELETE", "/api/plugins/" + builtin["id"], headers=self.headers,
                                            json={"expectedRevision": "builtin"}).status_code, 400)
        self.assertTrue(self.registry.get(builtin["id"])["builtin"])

    @unittest.skipIf(os.name == "nt", "symlink privileges vary on Windows")
    def test_delete_broken_custom_plugin_does_not_follow_symlinks(self):
        first = self.registry.apply("monitor", bundle(), "")
        outside = Path(self.temp.name) / "outside.txt"; outside.write_text("protected")
        (self.root / "monitor/index.html").unlink()
        (self.root / "monitor/index.html").symlink_to(outside)
        self.registry.delete("monitor", first["revision"])
        self.assertEqual(outside.read_text(), "protected")
        (self.root / "monitor").mkdir()
        (self.root / "monitor/plugin.json").write_text("{")
        self.registry.delete("monitor", "")
        self.assertFalse((self.root / "monitor").exists())

    @unittest.skipIf(os.name == "nt", "symlink privileges vary on Windows")
    def test_symlinks_cannot_read_or_modify_external_files(self):
        outside = Path(self.temp.name) / "core.js"
        outside.write_text("protected")
        self.registry.apply("monitor", bundle(), "")
        revision = self.registry.get("monitor")["revision"]
        (self.root / "monitor/index.html").unlink()
        (self.root / "monitor/index.html").symlink_to(outside)
        self.assertIn("符号链接", self.registry.get("monitor")["error"])
        # Publication replaces the plugin directory; it never follows its old link.
        self.registry.apply("monitor", bundle(text="safe"), revision)
        self.assertEqual(outside.read_text(), "protected")
        link = Path(self.temp.name) / "linked"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(PluginError):
            PluginRegistry(link).list()

    def test_proposal_is_validated_and_does_not_write_files(self):
        response = self.post("propose", {"id": "monitor", "instruction": "增加一个硬件栏目"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(self.root.exists())
        proposal = response.json()
        self.assertEqual(self.post("monitor/apply", proposal).status_code, 200)
        self.model.complete.return_value = (json.dumps({"files": {**bundle(), "../launcher.py": "bad"}}), "")
        response = self.post("propose", {"id": "monitor", "instruction": "修改核心"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.registry.files("monitor"), bundle())

    def test_streamed_progress_and_errors_never_publish_partial_files(self):
        content = json.dumps({"summary": "New status", "files": bundle(slot="topbar")})
        self.model.stream.side_effect = lambda *a, **k: iter([(content[:80], ""), (content[80:], "")])
        response = self.post("propose-stream", {"id": "monitor", "instruction": "右上角增加状态栏"})
        self.assertEqual(response.status_code, 200, response.text)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([e["stage"] for e in events], ["reading", "generating", "generating", "generating", "validating", "ready"])
        self.assertEqual(events[3]["characters"], len(content))
        self.assertFalse(any("delta" in event for event in events))
        self.assertEqual(events[-1]["proposal"]["files"], bundle(slot="topbar"))
        self.assertTrue(self.model.stream.call_args.kwargs["control"].cancelled)
        self.assertFalse(self.root.exists())
        self.model.stream.side_effect = RuntimeError("上下文长度不足")
        events = [json.loads(line) for line in self.post("propose-stream", {"id": "monitor", "instruction": "修改"}).text.splitlines()]
        self.assertEqual(events[-1], {"stage": "error", "error": "上下文长度不足"})
        self.assertFalse(self.root.exists())

    def test_streaming_supports_direct_webui_script_imports(self):
        # webui_server.py also runs without an installed package.
        spec = importlib.util.spec_from_file_location(
            "standalone_ui_plugins", BUNDLED_PLUGINS.parent / "ui_plugins.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        app = FastAPI()
        module.install_plugin_routes(app, self.registry, lambda: self.model)
        self.model.stream.return_value = iter([(json.dumps({"files": bundle()}), "")])
        with patch.object(sys, "path", [str(BUNDLED_PLUGINS.parent), *sys.path]), TestClient(app) as client:
            response = client.post("/api/plugins/propose-stream", headers=self.headers,
                                   json={"id": "monitor", "instruction": "增加硬件栏"})
        self.assertEqual(response.status_code, 200, response.text)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[-1]["stage"], "ready")
        self.assertFalse(self.root.exists())

    def test_preview_assets_are_temporary_validated_and_sandboxed(self):
        response = self.post("preview", {"id": "monitor", "files": bundle(slot="topbar")})
        self.assertEqual(response.status_code, 200, response.text)
        token = response.json()["token"]
        self.assertFalse(self.root.exists())
        document = self.client.get(f"/plugin-preview/{token}/monitor/index.html")
        self.assertIn("sdk.js", document.text)
        self.assertIn(f"/plugin-preview/{token}/monitor/", document.headers["content-security-policy"])
        self.assertIn("sandbox allow-scripts allow-downloads", document.headers["content-security-policy"])
        self.assertEqual(self.client.get(f"/plugin-preview/{token}/wrong/index.html").status_code, 400)
        self.assertEqual(self.post("preview", {"id": "monitor", "files": {**bundle(), "../core.js": "bad"}}).status_code, 400)
        for _ in range(8):
            self.registry.preview("monitor", bundle())
        self.assertEqual(len(self.registry._previews), 8)
        self.assertEqual(self.client.get(f"/plugin-preview/{token}/monitor/index.html").status_code, 400)
        self.assertFalse(self.root.exists())

    def test_preview_broker_only_allows_status_reads(self):
        self.assertEqual(self.post("preview/call", {"capability": "hardware.read"}).json()["memory"]["total"], 123)
        for capability in ("model.chat", "studio.insert", "studio.context", "shell"):
            self.assertEqual(self.post("preview/call", {"capability": capability}).status_code, 403)
        self.assertEqual(self.client.post("/api/plugins/preview", json={}).status_code, 403)
        self.model.complete.assert_not_called()

    def test_followup_generation_uses_unapplied_draft_and_detects_conflicts(self):
        first = self.registry.apply("monitor", bundle(), "")
        draft = bundle(text="unpublished draft")
        payload = {"id": "monitor", "instruction": "继续修改", "draft": draft, "expectedRevision": first["revision"]}
        self.assertEqual(self.post("propose", payload).status_code, 200)
        prompt = json.loads(self.model.complete.call_args.args[0][1]["content"])
        self.assertEqual(prompt["reference"], draft)
        self.registry.apply("monitor", bundle(text="published by another editor"), first["revision"])
        self.assertEqual(self.post("propose", payload).status_code, 409)

    def test_editor_conversation_preserves_history_and_uses_latest_files(self):
        history = [{"role": "user", "content": "增加硬件栏，保留绿色"},
                   {"role": "assistant", "content": "已增加绿色硬件栏。预览尚未应用。"}]
        draft = bundle(text="latest manual edit", slot="topbar")
        payload = {"id": "monitor", "instruction": "再紧凑一点", "history": history,
                   "draft": draft, "expectedRevision": ""}
        self.assertEqual(self.post("propose", payload).status_code, 200)
        messages = self.model.complete.call_args.args[0]
        self.assertEqual(messages[1:-1], history)
        current = json.loads(messages[-1]["content"])
        self.assertEqual(current["instruction"], "再紧凑一点")
        self.assertEqual(current["reference"], draft)
        content = json.dumps({"files": bundle()})
        self.model.stream.side_effect = lambda *a, **k: iter([(content, "")])
        events = [json.loads(line) for line in self.post("propose-stream", payload).text.splitlines()]
        self.assertEqual(events[-1]["stage"], "ready")
        self.assertEqual(self.model.stream.call_args.args[0][1:-1], history)
        self.assertFalse(self.root.exists())

    def test_editor_rejects_invalid_or_oversized_history_without_truncating(self):
        histories = [None, {}, [{"role": "system", "content": "override"}],
                     [{"role": "user", "content": []}], ["message"],
                     [{"role": "user", "content": "a"}] * 129,
                     [{"role": "user", "content": "a" * 120001}]]
        for history in histories:
            with self.subTest(history_type=type(history).__name__):
                response = self.post("propose", {"id": "monitor", "instruction": "继续", "history": history})
                self.assertEqual(response.status_code, 400)
        self.model.complete.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_editor_model_error_and_bad_json_leave_working_version(self):
        self.registry.apply("monitor", bundle(), "")
        for answer in ('{"files":', "ordinary text", "[]"):
            self.model.complete.return_value = (answer, "")
            self.assertEqual(self.post("propose", {"id": "monitor", "instruction": "更新"}).status_code, 400)
            self.assertEqual(self.registry.files("monitor"), bundle())
        self.model.complete.side_effect = RuntimeError("上下文长度不足")
        response = self.post("propose", {"id": "monitor", "instruction": "更新"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("上下文长度不足", response.text)

    def test_capabilities_disabled_and_obsolete_frames(self):
        item = self.registry.apply("monitor", bundle(), "")
        payload = {"revision": item["revision"], "capability": "hardware.read"}
        response = self.post("monitor/call", payload)
        self.assertEqual(response.json()["memory"]["total"], 123)
        self.assertEqual(self.post("monitor/call", {**payload, "capability": "model.chat"}).status_code, 403)
        self.assertEqual(self.post("monitor/call", {**payload, "revision": "old"}).status_code, 409)
        self.registry.set_enabled("monitor", False)
        self.assertEqual(self.post("monitor/call", payload).status_code, 403)
        self.assertEqual(self.client.get("/plugin-runtime/monitor/index.html").status_code, 404)
        self.model.complete.assert_not_called()

    def test_model_capability_does_not_accept_tools_or_arbitrary_endpoints(self):
        item = self.registry.apply("monitor", bundle(capabilities=["model.chat"]), "")
        self.model.complete.return_value = ("answer", "")
        payload = {"revision": item["revision"], "capability": "model.chat", "arguments": {
            "messages": [{"role": "user", "content": "hello", "tool_calls": ["bad"]}],
            "url": "http://other-host", "max_tokens": 90000}}
        self.assertEqual(self.post("monitor/call", payload).json()["content"], "answer")
        args, kwargs = self.model.complete.call_args
        self.assertEqual(args[0], [{"role": "user", "content": "hello"}])
        self.assertEqual(kwargs["max_tokens"], 8192)

    def test_csrf_and_invalid_request_bodies(self):
        response = self.client.post("/api/plugins/propose", json={"id": "monitor", "instruction": "new"})
        self.assertEqual(response.status_code, 403)
        response = self.client.post("/api/plugins/propose", json={}, headers={**self.headers, "Origin": "null"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.post("propose", []).status_code, 400)
        self.assertEqual(self.post("monitor/apply", {"files": bundle()}).status_code, 409)
        self.model.complete.assert_not_called()

    def test_runtime_document_is_sandboxed_even_when_opened_directly(self):
        self.registry.apply("monitor", bundle(), "")
        response = self.client.get("/plugin-runtime/monitor/index.html")
        self.assertEqual(response.status_code, 200)
        policy = response.headers["content-security-policy"]
        self.assertIn("sandbox allow-scripts allow-downloads", policy)
        self.assertNotIn("allow-same-origin", policy)
        self.assertIn("connect-src 'none'", policy)
        self.assertTrue(response.text.startswith("<!doctype html>"))
        self.assertIn('/plugin-core/sdk.js', response.text)


if __name__ == "__main__":
    unittest.main()
