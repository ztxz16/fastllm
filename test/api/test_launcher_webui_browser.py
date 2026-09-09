"""Optional Chromium regression tests; requires Playwright and its browser."""
import argparse
import base64
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tools')))
from fastllm_pytools.launcher import LauncherRuntime, create_launcher_app
from fastllm_pytools.webui_server import GenerationCancelled, add_webui_args, create_app

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipIf(sync_playwright is None, 'Playwright is not installed')
class LauncherWebUIBrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        cls.browser = cls.playwright.chromium.launch(args=['--no-sandbox'])
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        import uvicorn
        from fastapi import FastAPI

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = LauncherRuntime(os.path.join(self.temp.name, 'profiles.json'),
                                       webui_history_dir=os.path.join(self.temp.name, 'history'),
                                       plugins_dir=os.path.join(self.temp.name, 'plugins'))
        self.addCleanup(self.close_runtime)
        self.runtime._process = SimpleNamespace(poll=lambda: None)
        self.runtime._state.update(command='server', phase='running', ready=True,
                                   sessionId='model-a', modelName='browser-model',
                                   endpoint='http://127.0.0.1:19001')
        launcher = create_launcher_app(self.runtime, 'browser-key')
        args = add_webui_args(argparse.ArgumentParser()).parse_args([])
        args.api_model = 'standalone-model'
        args.agent_runtime = 'builtin'
        args.history_dir = os.path.join(self.temp.name, 'standalone-history')
        args.plugins_dir = os.path.join(self.temp.name, 'plugins')
        standalone = create_app(args)
        self.standalone = standalone.state.runtime
        self.addCleanup(self.standalone.close)
        # Exercise standalone WebUI without Launcher's unrelated middleware.
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        app.mount('/standalone', standalone)
        app.mount('/', launcher)
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, log_level='error', ws='none'))
        self.server_thread = threading.Thread(target=self.server.run,
                                              kwargs={'sockets': [listener]}, daemon=True)
        self.server_thread.start()
        self.addCleanup(self.stop_server)
        deadline = time.monotonic() + 5
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(self.server.started)
        self.context = self.browser.new_context(locale='en-US')
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.page.on('console', lambda message: self.errors.append(message.text)
                     if 'Content Security Policy' in message.text else None)
        self.url = f'http://127.0.0.1:{port}'
        self.page.goto(self.url + '/?token=browser-key')
        expect(self.page.locator('#open-webui')).to_be_enabled()
        self.page.clock.install()

    def tearDown(self):
        self.assertFalse(self.errors)

    def close_runtime(self):
        self.runtime._process = None
        self.runtime.close()

    def stop_server(self):
        self.server.should_exit = True
        self.server_thread.join(timeout=5)
        self.assertFalse(self.server_thread.is_alive())

    def assert_loaded(self):
        expect(self.page.locator('#webui-content')).to_be_visible()
        expect(self.page.locator('#webui-content').locator('#prompt')).to_be_visible()
        # Successful loads must clear the previous attempt's timeout.
        self.page.clock.fast_forward(35000)
        expect(self.page.locator('#webui-content')).to_be_visible()
        expect(self.page.locator('#webui-retry')).to_be_hidden()

    def install_test_plugin(self, text='first', **manifest):
        from test_ui_plugins import bundle
        files = bundle(text=text, **manifest)
        files['index.html'] = '''<!doctype html><html><head></head><body>
          <output id="value">loading</output><button id="insert">Insert</button>
          <script>
          (async () => {
            let isolated = false;
            try { parent.document.body.textContent = 'unsafe'; } catch (_) { isolated = true; }
            const report = await ftllm.call('hardware.read');
            document.querySelector('#value').textContent = LABEL + ':' + isolated + ':' + Boolean(report.memory);
            document.querySelector('#insert').onclick = async () => {
              const context = await ftllm.call('studio.context');
              await ftllm.call('studio.insert', {text:'Plugin addition'});
              document.querySelector('#value').textContent = context.draft;
            };
          })();
          </script></body></html>'''.replace('LABEL', json.dumps(text))
        current = next((p for p in self.runtime.plugins.list()['plugins'] if p['id'] == 'monitor'), None)
        self.runtime.plugins.apply('monitor', files, current['revision'] if current else '')
        self.page.clock.fast_forward(2100)
        return files

    def test_plugin_hot_add_replace_remove_and_isolation(self):
        self.install_test_plugin()
        nav = self.page.locator('[data-view-button="plugin-monitor"]')
        expect(nav).to_be_visible(); nav.click()
        self.assertLess(nav.bounding_box()['height'], 60)
        frame = self.page.frame_locator('#view-plugin-monitor iframe')
        expect(frame.locator('#value')).to_have_text('first:true:true')
        self.assertIsNotNone(self.runtime._process)
        self.install_test_plugin('second')
        expect(frame.locator('#value')).to_have_text('second:true:true')
        expect(nav).to_have_attribute('aria-current', 'page')
        self.runtime.plugins.set_enabled('monitor', False)
        self.page.clock.fast_forward(2100)
        expect(nav).to_have_count(0)
        expect(self.page.locator('#profile-browser')).to_be_visible()
        self.assertIsNotNone(self.runtime._process)

    def test_customizer_entry_icon_and_no_studio_entry(self):
        button = self.page.locator('.navigation > .plugin-manager-button')
        expect(button).to_have_text('自定义界面')
        emblem = button.locator('.plugin-manager-emblem')
        expect(emblem.locator('svg')).to_be_visible()
        self.assertIn('linear-gradient', emblem.evaluate('node => getComputedStyle(node).backgroundImage'))
        self.page.locator('#open-webui').click(); self.assert_loaded()
        expect(self.page.locator('#webui-content .plugin-manager-button')).to_have_count(0)
        self.screenshot('customizer-entry-light')
        self.page.locator('#theme-select').select_option('dark')
        button.hover()
        self.screenshot('customizer-entry-dark')
        self.page.emulate_media(reduced_motion='reduce')
        expect(emblem).to_have_css('transform', 'none')
        self.page.goto(self.url + '/standalone/')
        expect(self.page.locator('#webui-root #prompt')).to_be_visible()
        expect(self.page.locator('#webui-root .plugin-manager-button')).to_have_count(0)
        self.page.goto(self.url + '/standalone/#customize')
        expect(self.page.locator('#webui-root .plugin-manager')).to_be_visible()

    def test_custom_plugin_delete_unloads_without_changing_studio_draft(self):
        self.install_test_plugin(slot='topbar')
        self.install_test_plugin('second', slot='topbar')
        self.page.locator('#open-webui').click(); self.assert_loaded()
        pane = self.page.locator('body > .app-shell #webui-content')
        pane.locator('#prompt').fill('Keep this conversation draft')
        self.page.locator('.navigation > .plugin-manager-button').click()
        editor = self.page.locator('body > .plugin-manager')
        editor.locator('.customizer-library > summary').click()
        row = editor.locator('[data-plugin-id=monitor]')
        expect(editor.locator('[data-plugin-id=studio] .plugin-delete')).to_have_count(0)
        expect(editor.locator('.plugin-delete')).to_have_count(1)
        row.locator('.plugin-delete').click()
        row.locator('.plugin-delete-confirm').get_by_role('button', name='取消', exact=True).click()
        self.assertTrue((self.runtime.plugins.directory / 'monitor').is_dir())
        row.locator('.plugin-delete').click()
        self.screenshot('custom-plugin-delete')
        row.locator('.plugin-confirm-delete').click()
        expect(row).to_have_count(0)
        expect(self.page.locator('body > .app-shell .plugin-shell-topbar iframe')).to_have_count(0)
        expect(editor.locator('.customizer-screen .plugin-shell-topbar iframe')).to_have_count(0)
        self.assertFalse((self.runtime.plugins.directory / 'monitor').exists())
        self.assertFalse((self.runtime.plugins.directory / '.history/monitor.json').exists())
        expect(pane.locator('#prompt')).to_have_value('Keep this conversation draft')
        self.assertIsNotNone(self.runtime._process)

    def test_plugin_replacement_can_restore_original_page(self):
        self.install_test_plugin(replaces='hardware')
        self.page.locator('[data-view-button="hardware"]').click()
        frame = self.page.frame_locator('#view-hardware iframe')
        expect(frame.locator('#value')).to_have_text('first:true:true')
        self.runtime.plugins.set_enabled('monitor', False)
        self.page.clock.fast_forward(2100)
        expect(self.page.locator('#view-hardware iframe')).to_have_count(0)
        expect(self.page.locator('#refresh-hardware')).to_be_visible()

    def test_plugin_script_error_keeps_recovery_controls_working(self):
        from test_ui_plugins import bundle
        files = bundle()
        files['index.html'] = '<!doctype html><script>throw new Error("broken plugin fixture")</script>'
        self.runtime.plugins.apply('monitor', files, '')
        self.page.clock.fast_forward(2100)
        self.page.locator('[data-view-button="plugin-monitor"]').click()
        expect(self.page.locator('#view-plugin-monitor > .plugin-status')).to_contain_text('broken plugin fixture')
        self.page.locator('.navigation > .plugin-manager-button').click()
        expect(self.page.locator('body > .plugin-manager')).to_be_visible()
        self.assertEqual(self.errors, ['broken plugin fixture'])
        self.errors.clear()

    def test_studio_plugin_preserves_draft_and_uses_scoped_bridge(self):
        self.install_test_plugin(slot='studio', capabilities=['hardware.read', 'studio.context', 'studio.insert'])
        self.page.locator('#open-webui').click(); self.assert_loaded()
        pane = self.page.locator('#webui-content')
        pane.locator('#prompt').fill('Keep this draft')
        frame = self.page.frame_locator('#webui-content iframe.plugin-frame')
        expect(frame.locator('#value')).to_have_text('first:true:true')
        frame.locator('#insert').click()
        expect(frame.locator('#value')).to_have_text('Keep this draft')
        expect(pane.locator('#prompt')).to_have_value('Keep this draftPlugin addition')
        self.runtime.plugins.set_enabled('monitor', False)
        self.page.clock.fast_forward(2100)
        expect(pane.locator('iframe.plugin-frame')).to_have_count(0)
        expect(pane.locator('#prompt')).to_have_value('Keep this draftPlugin addition')

    def test_plugin_model_preview_apply_and_core_recovery_ui(self):
        from test_ui_plugins import bundle
        files = bundle(text='Generated page')
        self.page.locator('#open-webui').click(); self.assert_loaded()
        model = self.runtime._webui_app.state.runtime.api_client
        with patch.object(model, 'stream', side_effect=lambda *a, **k: iter([(json.dumps({'summary':'New monitor', 'files':files}), '')])) as complete:
            self.page.locator('.navigation > .plugin-manager-button').click()
            dialog = self.page.locator('body > .plugin-manager')
            dialog.locator('.customizer-options > summary').click()
            dialog.locator('[name="id"]').fill('monitor')
            dialog.locator('[name="instruction"]').fill('增加一个硬件监控栏目')
            dialog.locator('.plugin-editor [type=submit]').click()
            expect(dialog.locator('.plugin-preview')).to_be_visible()
            self.assertFalse((self.runtime.plugins.directory / 'monitor').exists())
            dialog.locator('.plugin-apply').click()
            expect(dialog.locator('.plugin-result')).to_have_text('已应用，界面已更新。')
            self.assertEqual(complete.call_count, 1)
            dialog.locator('.plugin-heading [aria-label="关闭"]').click()
            self.page.clock.fast_forward(2100)
            self.page.locator('[data-view-button="plugin-monitor"]').click()
            expect(self.page.frame_locator('#view-plugin-monitor iframe').locator('p')).to_have_text('Generated page')
            expect(self.page.locator('.navigation > .plugin-manager-button')).to_be_visible()

    def test_customizer_stream_progress_live_draft_preview_and_manual_edits(self):
        from test_ui_plugins import bundle
        self.page.locator('#open-webui').click(); self.assert_loaded()
        model = self.runtime._webui_app.state.runtime.api_client
        files = bundle(slot='topbar')
        files['index.html'] = '''<!doctype html><output>loading</output><script>
          ftllm.call('hardware.read').then(value => {
            document.querySelector('output').textContent = 'Preview memory: ' + Boolean(value.memory);
          });</script>'''
        content = json.dumps({'summary': 'Hardware status', 'files': files})
        release = threading.Event(); self.addCleanup(release.set)
        def stream(*args, **kwargs):
            yield content[:90], ''
            release.wait(10)
            yield content[90:], ''
        with patch.object(model, 'stream', side_effect=stream):
            button = self.page.locator('.navigation > .plugin-manager-button')
            expect(button).to_have_text('自定义界面'); button.click()
            editor = self.page.locator('body > .plugin-manager')
            self.assertEqual(editor.evaluate('node => node.tagName'), 'MAIN')
            self.assertTrue(self.page.url.endswith('#customize'))
            expect(editor.locator('.customizer-screen .app-shell')).to_be_visible()
            editor.locator('.customizer-options > summary').click()
            editor.locator('[name=id]').fill('monitor')
            editor.locator('[name=target]').select_option('topbar')
            editor.locator('[name=instruction]').fill('右上角增加硬件状态栏')
            editor.locator('.plugin-editor [type=submit]').click()
            expect(editor).to_have_attribute('data-stage', 'generating')
            expect(editor.locator('.customizer-user p')).to_have_text('右上角增加硬件状态栏')
            expect(editor.locator('.customizer-reply')).to_have_text('模型正在生成修改…')
            expect(editor.locator('.plugin-output')).to_have_count(0)
            expect(editor.locator('.plugin-timing')).to_contain_text('90 字符')
            self.page.clock.fast_forward(2100)
            expect(editor.locator('.plugin-timing')).to_contain_text('已用 2 秒')
            self.screenshot('customizer-generating')
            release.set()
            expect(editor.locator('.plugin-result')).to_have_text('预览已更新，尚未应用修改。')
            expect(editor.locator('.customizer-reply')).to_have_text('Hardware status')
            preview = editor.frame_locator('.customizer-screen .plugin-shell-topbar iframe')
            expect(preview.locator('output')).to_have_text('Preview memory: true')
            self.assertFalse((self.runtime.plugins.directory / 'monitor').exists())
            expect(self.page.locator('body > .app-shell > .workspace > .topbar iframe')).to_have_count(0)
            expect(editor.locator('.customizer-screen #stop-runtime')).to_be_disabled()
            editor.locator('.customizer-code > summary').click()
            editor.locator('.plugin-files').select_option('index.html')
            editor.locator('.plugin-after').fill('<!doctype html><output>Edited preview</output>')
            self.page.clock.fast_forward(500)
            expect(preview.locator('output')).to_have_text('Edited preview')
            editor.locator('.plugin-files').select_option('plugin.json')
            editor.locator('.plugin-after').fill('{')
            self.page.clock.fast_forward(500)
            expect(editor.locator('.plugin-apply')).to_be_disabled()
            expect(editor.locator('.plugin-result')).to_contain_text('plugin.json')
            expect(preview.locator('output')).to_have_text('Edited preview')
            editor.locator('.plugin-after').fill(files['plugin.json'])
            self.page.clock.fast_forward(500)
            expect(editor.locator('.plugin-apply')).to_be_enabled()
            self.screenshot('customizer-preview')
            editor.locator('.plugin-apply').click()
            expect(editor.locator('.plugin-result')).to_have_text('已应用，界面已更新。')
            editor.locator('.plugin-heading [aria-label=关闭]').click()
            expect(editor).to_be_hidden()
            expect(self.page.frame_locator('body > .app-shell .plugin-shell-topbar iframe').locator('output')).to_have_text('Edited preview')

    def test_customizer_conversation_keeps_rounds_and_sends_history_with_latest_draft(self):
        from test_ui_plugins import bundle
        first = bundle(slot='topbar', text='First version')
        second = bundle(slot='topbar', text='Compact version')
        self.page.locator('#open-webui').click(); self.assert_loaded()
        model = self.runtime._webui_app.state.runtime.api_client
        answers = [iter([(json.dumps({'summary': '已增加绿色硬件栏。', 'files': first}), '')]),
                   iter([(json.dumps({'summary': '已缩小间距，保留绿色。', 'files': second}), '')]),
                   iter([(json.dumps({'summary': '已调整文字大小。', 'files': second}), '')])]
        with patch.object(model, 'stream', side_effect=answers) as generate:
            self.page.locator('.navigation > .plugin-manager-button').click()
            editor = self.page.locator('body > .plugin-manager')
            editor.locator('.customizer-options > summary').click()
            editor.locator('[name=id]').fill('monitor')
            editor.locator('[name=instruction]').fill('增加硬件栏，使用绿色')
            editor.locator('.plugin-editor [type=submit]').click()
            expect(editor.locator('.plugin-apply')).to_be_enabled()
            expect(editor.locator('[name=instruction]')).to_have_value('')
            editor.locator('.customizer-code > summary').click()
            editor.locator('.plugin-files').select_option('index.html')
            manual = '<!doctype html><p>Latest manual edit</p>'
            editor.locator('.plugin-after').fill(manual)
            self.page.clock.fast_forward(500)
            expect(editor.locator('.plugin-apply')).to_be_enabled()
            editor.locator('.customizer-code > summary').click()
            editor.locator('[name=instruction]').fill('再紧凑一点，颜色保持刚才的')
            editor.locator('.plugin-editor [type=submit]').click()
            expect(editor.locator('.plugin-apply')).to_be_enabled()
            expect(editor.locator('.customizer-user p')).to_have_text(['增加硬件栏，使用绿色', '再紧凑一点，颜色保持刚才的'])
            expect(editor.locator('.customizer-reply')).to_have_text(['已增加绿色硬件栏。', '已缩小间距，保留绿色。'])
            messages = generate.call_args_list[1].args[0]
            self.assertEqual(messages[1], {'role': 'user', 'content': '增加硬件栏，使用绿色'})
            self.assertIn('已增加绿色硬件栏。', messages[2]['content'])
            self.assertEqual(json.loads(messages[-1]['content'])['reference']['index.html'], manual)
            self.assertFalse((self.runtime.plugins.directory / 'monitor').exists())
            self.screenshot('customizer-conversation')
            editor.locator('.plugin-apply').click()
            expect(editor.locator('.customizer-turn-status').nth(1)).to_have_text('已应用到当前界面。')
            editor.locator('.plugin-heading [aria-label=关闭]').click()
            self.page.locator('.navigation > .plugin-manager-button').click()
            expect(editor.locator('.customizer-turn')).to_have_count(2)
            editor.locator('.customizer-new-chat').click()
            expect(editor.locator('.customizer-turn')).to_have_count(0)
            expect(editor.locator('.plugin-preview')).to_be_hidden()
            editor.locator('.customizer-options > summary').click()
            editor.locator('[name=id]').fill('monitor')
            editor.locator('[name=instruction]').fill('文字再大一点')
            editor.locator('.plugin-editor [type=submit]').click()
            expect(editor.locator('.plugin-apply')).to_be_enabled()
            expect(editor.locator('.customizer-turn')).to_have_count(1)
            self.assertEqual(len(generate.call_args_list[2].args[0]), 2)
            self.assertEqual(json.loads(generate.call_args_list[2].args[0][-1]['content'])['reference'], second)

    def test_customizer_sessions_restore_drafts_rename_delete_and_survive_reload(self):
        from test_ui_plugins import bundle
        requests = []
        def generate(route):
            payload = route.request.post_data_json; requests.append(payload)
            files = bundle(payload['id'], slot='topbar', text=payload['id'])
            route.fulfill(status=200, content_type='application/x-ndjson', body=json.dumps({'stage': 'ready', 'proposal': {
                'id': payload['id'], 'files': files, 'manifest': {'slot': 'topbar'}, 'summary': '完成 ' + payload['id'], 'expectedRevision': ''}}) + '\n')
        self.page.route('**/api/plugins/propose-stream', generate)
        self.page.locator('.navigation > .plugin-manager-button').click()
        editor = self.page.locator('body > .plugin-manager')
        for identifier, instruction in [('alpha', '增加硬件状态栏'), ('beta', '增加统计信息')]:
            if identifier == 'beta':
                editor.locator('.customizer-new-chat').click()
                expect(editor.locator('.customizer-turn')).to_have_count(0)
                expect(editor.locator('[name=instruction]')).to_have_value('')
                expect(editor.locator('.plugin-preview')).to_be_hidden()
            editor.locator('.customizer-options > summary').click()
            editor.locator('[name=id]').fill(identifier)
            editor.locator('[name=target]').select_option('topbar')
            editor.locator('[name=instruction]').fill(instruction)
            editor.locator('.plugin-editor [type=submit]').click()
            expect(editor.locator('.plugin-apply')).to_be_enabled()
            if identifier == 'alpha':
                editor.locator('.customizer-code > summary').click()
                editor.locator('.plugin-files').select_option('index.html')
                editor.locator('.plugin-after').fill('<!doctype html><p>Alpha edited draft</p>')
                self.page.clock.fast_forward(500)
                expect(editor.locator('.plugin-apply')).to_be_enabled()
                editor.locator('.customizer-code > summary').click()
            editor.locator('[name=instruction]').fill('尚未发送：' + identifier)
        self.assertEqual([request['history'] for request in requests], [[], []])
        editor.locator('.customizer-toggle-sessions').click()
        rows = editor.locator('.customizer-session')
        expect(rows).to_have_count(2)
        alpha = rows.filter(has_text='增加硬件状态栏')
        alpha.locator('[data-session-action=rename]').click()
        alpha.locator('[aria-label=对话名称]').fill('硬件监控')
        alpha.locator('[data-session-action=save-name]').click()
        editor.locator('.customizer-session-search').fill('硬件监控')
        expect(rows).to_have_count(1)
        rows.locator('[data-session-action=switch]').click()
        expect(editor.locator('.plugin-apply')).to_be_enabled()
        expect(editor.locator('.customizer-reply')).to_have_text('完成 alpha')
        expect(editor.locator('[name=instruction]')).to_have_value('尚未发送：alpha')
        expect(editor.frame_locator('.customizer-screen .plugin-shell-topbar iframe').locator('p')).to_have_text('Alpha edited draft')
        self.page.reload()
        expect(editor.locator('.customizer-session-title')).to_have_text('硬件监控')
        expect(editor.locator('.plugin-apply')).to_be_enabled()
        expect(editor.locator('[name=instruction]')).to_have_value('尚未发送：alpha')
        expect(editor.frame_locator('.customizer-screen .plugin-shell-topbar iframe').locator('p')).to_have_text('Alpha edited draft')
        editor.locator('.customizer-toggle-sessions').click()
        expect(rows).to_have_count(2)
        self.screenshot('customizer-session-management')
        beta = rows.filter(has_text='增加统计信息')
        beta.locator('[data-session-action=delete]').click()
        beta.locator('[data-session-action=cancel]').click()
        expect(rows).to_have_count(2)
        beta.locator('[data-session-action=delete]').click()
        beta.locator('[data-session-action=confirm-delete]').click()
        expect(rows).to_have_count(1)
        editor.locator('.customizer-toggle-sessions').click()
        editor.locator('.plugin-apply').click()
        expect(editor.locator('.plugin-result')).to_have_text('已应用，界面已更新。')
        editor.locator('.customizer-toggle-sessions').click()
        rows.locator('[data-session-action=delete]').click()
        rows.locator('[data-session-action=confirm-delete]').click()
        expect(editor.locator('.customizer-session-title')).to_have_text('新对话')
        expect(editor.locator('.customizer-turn')).to_have_count(0)
        self.assertEqual(self.runtime.plugins.files('alpha')['index.html'], '<!doctype html><p>Alpha edited draft</p>')
        self.page.reload()
        expect(editor.locator('.customizer-session-title')).to_have_text('新对话')
        expect(editor.locator('.customizer-turn')).to_have_count(0)

    def test_customizer_new_sessions_get_distinct_automatic_names_and_context(self):
        from test_ui_plugins import bundle
        requests = []
        def generate(route):
            payload = route.request.post_data_json; requests.append(payload)
            files = bundle(payload['id'], slot='topbar')
            route.fulfill(status=200, content_type='application/x-ndjson', body=json.dumps({'stage': 'ready', 'proposal': {
                'id': payload['id'], 'files': files, 'manifest': {'slot': 'topbar'}, 'summary': '完成', 'expectedRevision': ''}}) + '\n')
        self.page.route('**/api/plugins/propose-stream', generate)
        self.page.locator('.navigation > .plugin-manager-button').click()
        editor = self.page.locator('body > .plugin-manager')
        for index in range(2):
            if index:
                editor.locator('.customizer-new-chat').click()
            editor.locator('[name=instruction]').fill(f'增加栏目 {index}')
            editor.locator('.plugin-editor [type=submit]').click()
            expect(editor.locator('.plugin-apply')).to_be_enabled()
        self.assertNotEqual(requests[0]['id'], requests[1]['id'])
        self.assertEqual([request['history'] for request in requests], [[], []])
        self.assertNotIn('draft', requests[1])
        expect(editor.locator('.customizer-user p')).to_have_text('增加栏目 1')

    def test_customizer_switch_ignores_old_preview_validation(self):
        from test_ui_plugins import bundle
        files = bundle(slot='topbar')
        self.page.route('**/api/plugins/propose-stream', lambda route: route.fulfill(
            status=200, content_type='application/x-ndjson', body=json.dumps({'stage': 'ready', 'proposal': {
                'id': 'monitor', 'files': files, 'manifest': {'slot': 'topbar'}, 'summary': '硬件栏', 'expectedRevision': ''}}) + '\n'))
        self.page.locator('.navigation > .plugin-manager-button').click()
        editor = self.page.locator('body > .plugin-manager')
        editor.locator('.customizer-options > summary').click()
        editor.locator('[name=id]').fill('monitor')
        editor.locator('[name=instruction]').fill('增加硬件栏')
        editor.locator('.plugin-editor [type=submit]').click()
        expect(editor.locator('.plugin-apply')).to_be_enabled()
        pending = []
        self.page.route('**/api/plugins/preview', lambda route: pending.append(route))
        editor.locator('.customizer-code > summary').click()
        editor.locator('.plugin-files').select_option('index.html')
        editor.locator('.plugin-after').fill('<!doctype html><p>Late response</p>')
        self.page.clock.fast_forward(500)
        self.page.wait_for_function('true')
        self.assertEqual(len(pending), 1)
        editor.locator('.customizer-new-chat').click()
        expect(editor.locator('.customizer-new-chat')).to_be_enabled()
        expect(editor.locator('.customizer-turn')).to_have_count(0)
        checked = self.runtime.plugins.preview('monitor', pending[0].request.post_data_json['files'])
        pending[0].fulfill(status=200, content_type='application/json', body=json.dumps(checked))
        expect(editor.locator('.plugin-preview')).to_be_hidden()
        expect(editor.locator('.customizer-screen .plugin-shell-topbar iframe')).to_have_count(0)
        expect(editor.locator('.customizer-turn')).to_have_count(0)
        # Reopening a conversation also restores its last valid preview. That
        # fallback must not run after the user has already selected a new chat.
        self.page.unroute('**/api/plugins/preview')
        editor.locator('.customizer-toggle-sessions').click()
        editor.locator('.customizer-session').filter(has_text='增加硬件栏').locator('[data-session-action=switch]').click()
        expect(editor.locator('.plugin-apply')).to_be_enabled()
        editor.locator('.plugin-heading [aria-label=关闭]').click()
        self.page.route('**/api/plugins/preview', lambda route: pending.append(route))
        with self.page.expect_request('**/api/plugins/preview'):
            self.page.locator('.navigation > .plugin-manager-button').click()
        editor.locator('.customizer-new-chat').click()
        expect(editor.locator('.customizer-new-chat')).to_be_enabled()
        checked = self.runtime.plugins.preview('monitor', pending[1].request.post_data_json['files'])
        pending[1].fulfill(status=200, content_type='application/json', body=json.dumps(checked))
        self.page.wait_for_timeout(100)
        self.assertEqual(len(pending), 2)
        expect(editor.locator('.plugin-preview')).to_be_hidden()
        expect(editor.locator('.customizer-turn')).to_have_count(0)

    def test_customizer_restores_invalid_edit_and_last_good_preview(self):
        from test_ui_plugins import bundle
        files = bundle(slot='topbar', text='Good preview')
        self.page.route('**/api/plugins/propose-stream', lambda route: route.fulfill(
            status=200, content_type='application/x-ndjson', body=json.dumps({'stage': 'ready', 'proposal': {
                'id': 'monitor', 'files': files, 'manifest': {'slot': 'topbar'}, 'summary': '硬件栏', 'expectedRevision': ''}}) + '\n'))
        self.page.locator('.navigation > .plugin-manager-button').click()
        editor = self.page.locator('body > .plugin-manager')
        editor.locator('.customizer-options > summary').click()
        editor.locator('[name=id]').fill('monitor')
        editor.locator('[name=instruction]').fill('增加硬件栏')
        editor.locator('.plugin-editor [type=submit]').click()
        expect(editor.locator('.plugin-apply')).to_be_enabled()
        editor.locator('.customizer-code > summary').click()
        editor.locator('.plugin-files').select_option('plugin.json')
        editor.locator('.plugin-after').fill('{')
        self.page.clock.fast_forward(500)
        expect(editor.locator('.plugin-result')).to_contain_text('plugin.json')
        self.page.reload()
        expect(editor.locator('.plugin-after')).to_have_value('{')
        expect(editor.locator('.plugin-apply')).to_be_disabled()
        expect(editor.locator('.plugin-result')).to_contain_text('plugin.json')
        expect(editor.frame_locator('.customizer-screen .plugin-shell-topbar iframe').locator('p')).to_have_text('Good preview')
        editor.locator('.plugin-after').fill(files['plugin.json'])
        self.page.clock.fast_forward(500)
        expect(editor.locator('.plugin-apply')).to_be_enabled()

    def test_customizer_storage_conflict_does_not_overwrite_another_tab(self):
        self.page.locator('.navigation > .plugin-manager-button').click()
        editor = self.page.locator('body > .plugin-manager')
        expect(editor.locator('.customizer-new-chat')).to_be_enabled()
        self.page.evaluate('''async () => {
          const db = await new Promise((resolve, reject) => {
            const request = indexedDB.open('ftllm-interface-editor', 1);
            request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
          });
          await new Promise((resolve, reject) => {
            const tx = db.transaction('conversations', 'readwrite'), store = tx.objectStore('conversations');
            const request = store.get('/');
            request.onsuccess = () => {
              const saved = request.result; saved.revision++;
              saved.conversations[0].title = 'Saved in another tab'; store.put(saved, '/');
            };
            tx.oncomplete = resolve; tx.onerror = () => reject(tx.error);
          }); db.close();
        }''')
        editor.locator('[name=instruction]').fill('This tab keeps its unsaved input')
        self.page.clock.fast_forward(350)
        expect(editor.locator('.customizer-storage-error')).to_contain_text('其他页面已更新')
        expect(editor.locator('[name=instruction]')).to_have_value('This tab keeps its unsaved input')
        self.page.reload()
        expect(editor.locator('.customizer-session-title')).to_have_text('Saved in another tab')

    def test_customizer_failed_followup_keeps_reply_and_previous_draft(self):
        from test_ui_plugins import bundle
        files = bundle(slot='topbar', text='Last good preview')
        self.page.locator('#open-webui').click(); self.assert_loaded()
        model = self.runtime._webui_app.state.runtime.api_client
        answers = [iter([(json.dumps({'summary': '已增加硬件栏。', 'files': files}), '')]), RuntimeError('上下文长度不足')]
        with patch.object(model, 'stream', side_effect=answers):
            self.page.locator('.navigation > .plugin-manager-button').click()
            editor = self.page.locator('body > .plugin-manager')
            editor.locator('.customizer-options > summary').click()
            editor.locator('[name=id]').fill('monitor')
            editor.locator('[name=instruction]').fill('增加硬件栏')
            editor.locator('.plugin-editor [type=submit]').click()
            expect(editor.locator('.plugin-apply')).to_be_enabled()
            editor.locator('[name=instruction]').fill('再显示一些信息')
            editor.locator('.plugin-editor [type=submit]').click()
            expect(editor.locator('.plugin-result')).to_have_text('上下文长度不足')
            expect(editor.locator('.customizer-reply')).to_have_text(['已增加硬件栏。', '上下文长度不足'])
            expect(editor.locator('.plugin-apply')).to_be_enabled()
            expect(editor.frame_locator('.customizer-screen .plugin-shell-topbar iframe').locator('p')).to_have_text('Last good preview')
            editor.locator('.plugin-apply').click()
            expect(editor.locator('.customizer-turn-status').first).to_have_text('已应用到当前界面。')
            expect(editor.locator('.customizer-reply').nth(1)).to_have_text('上下文长度不足')
            self.assertEqual(self.runtime.plugins.files('monitor'), files)

    def test_customizer_cancel_and_browser_back_preserve_studio_draft(self):
        self.page.locator('#open-webui').click(); self.assert_loaded()
        pane = self.page.locator('body > .app-shell #webui-content')
        pane.locator('#prompt').fill('Keep original draft')
        controls = []
        def stream(*args, **kwargs):
            control = kwargs['control']; controls.append(control)
            yield '{"files":', ''
            control.event.wait(10)
            control.check()
        model = self.runtime._webui_app.state.runtime.api_client
        with patch.object(model, 'stream', side_effect=stream):
            expect(pane.locator('.plugin-manager-button')).to_have_count(0)
            self.page.locator('.navigation > .plugin-manager-button').click()
            editor = self.page.locator('body > .plugin-manager')
            expect(editor).to_be_visible()
            editor.locator('[name=instruction]').fill('更换皮肤')
            editor.locator('.plugin-editor [type=submit]').click()
            expect(editor.locator('.plugin-timing')).to_contain_text('9 字符')
            expect(editor.locator('.customizer-toggle-sessions')).to_be_disabled()
            editor.locator('.plugin-cancel').click()
            expect(editor.locator('.plugin-result')).to_have_text('已取消生成，尚未应用修改。')
            expect(editor.locator('.customizer-user p')).to_have_text('更换皮肤')
            expect(editor.locator('.customizer-reply')).to_have_text('已取消生成，尚未应用修改。')
            self.assertTrue(controls[0].event.wait(2))
            self.assertFalse(self.runtime.plugins.directory.exists())
            self.page.go_back()
            expect(editor).to_be_hidden()
            expect(pane.locator('#prompt')).to_have_value('Keep original draft')
            pane.locator('#prompt').click()
            self.page.go_forward()
            expect(editor).to_be_visible()
            expect(editor.locator('.customizer-turn')).to_have_count(1)
            editor.locator('.plugin-heading [aria-label=关闭]').click()

    def test_customizer_clicks_and_forms_only_change_preview(self):
        storage = self.page.evaluate('JSON.stringify(localStorage)')
        writes = []
        self.page.on('request', lambda request: writes.append(request.url)
                     if request.method not in ('GET', 'HEAD') else None)
        self.page.locator('.navigation > .plugin-manager-button').click()
        editor = self.page.locator('body > .plugin-manager')
        snapshot = editor.locator('.customizer-preview-page')
        expect(snapshot.locator('#stop-runtime')).to_be_disabled()
        snapshot.locator('[data-open-view=hardware]').click()
        expect(snapshot.locator('#view-hardware')).to_have_class(re.compile(r'\bactive\b'))
        snapshot.locator('[data-view-button=launch]').click()
        snapshot.locator('#new-profile').click()
        expect(snapshot.locator('#profile-editor-modal')).to_be_visible()
        snapshot.locator('#model-path').fill('/preview/model')
        snapshot.locator('[data-config-mode][value=custom]').check()
        expect(snapshot.locator('#profile-parameters')).to_be_visible()
        snapshot.locator('#enable-speculative-decoding').check()
        expect(snapshot.locator('#enable-speculative-decoding')).to_be_checked()
        expect(snapshot.locator('#save-profile')).to_be_disabled()
        snapshot.locator('#model-path').press('Enter')
        snapshot.locator('#close-profile-editor').click()
        expect(snapshot.locator('#profile-editor-modal')).to_be_hidden()
        snapshot.locator('#theme-select').select_option('dark')
        snapshot.locator('[data-open-view=webui]').click()
        expect(snapshot.locator('#view-webui')).to_have_class(re.compile(r'\bactive\b'))
        expect(snapshot.locator('#open-webui')).to_be_hidden()
        expect(self.page.locator('body > .app-shell #view-launch')).to_have_class(re.compile(r'\bactive\b'))
        expect(self.page.locator('body > .app-shell #model-path')).not_to_have_value('/preview/model')
        expect(self.page.locator('html')).to_have_attribute('data-theme', 'light')
        self.assertEqual(self.page.evaluate('JSON.stringify(localStorage)'), storage)
        self.assertEqual(writes, [])
        self.screenshot('customizer-interactive-preview')

    def test_customizer_previews_skin_before_apply_and_supports_mobile(self):
        from test_ui_plugins import theme_bundle
        files = theme_bundle(light={'background': '#eef4ff', 'primary': '#2563eb'}, dark={'background': '#101827'})
        self.page.route('**/api/plugins/propose-stream', lambda route: route.fulfill(
            status=200, content_type='application/x-ndjson', body=json.dumps({'stage':'ready', 'proposal':{
                'id':'blue-skin', 'summary':'Blue skin', 'files':files, 'expectedRevision':'',
                'manifest':{'slot':'theme'}}}) + '\n'))
        self.page.locator('.navigation > .plugin-manager-button').click()
        editor = self.page.locator('body > .plugin-manager')
        editor.locator('.customizer-options > summary').click()
        editor.locator('[name=id]').fill('blue-skin')
        editor.locator('[name=instruction]').fill('蓝色皮肤')
        editor.locator('.plugin-editor [type=submit]').click()
        expect(editor.locator('.plugin-result')).to_have_text('预览已更新，尚未应用修改。')
        snapshot = editor.locator('.customizer-preview-page')
        expect(snapshot).to_have_css('background-color', 'rgb(238, 244, 255)')
        expect(self.page.locator('html')).not_to_have_attribute('data-plugin-skin', 'blue-skin')
        expect(editor.locator('.customizer-view,.customizer-mode')).to_have_count(0)
        snapshot.locator('[data-view-button=webui]').click()
        expect(snapshot.locator('#webui-content #prompt')).to_be_visible()
        expect(snapshot.locator('#webui-content .main')).to_have_css('background-color', 'rgb(238, 244, 255)')
        snapshot.locator('#theme-select').select_option('dark')
        expect(snapshot.locator('#webui-content .main')).to_have_css('background-color', 'rgb(16, 24, 39)')
        expect(self.page.locator('html')).to_have_attribute('data-theme', 'light')
        editor.locator('.customizer-refresh').click()
        expect(editor.locator('.plugin-result')).to_have_text('预览已更新，尚未应用修改。')
        expect(snapshot.locator('#theme-select')).to_have_value('dark')
        expect(snapshot.locator('#view-webui')).to_have_class(re.compile(r'\bactive\b'))
        expect(snapshot.locator('#webui-content .main')).to_have_css('background-color', 'rgb(16, 24, 39)')
        editor.locator('.customizer-size').select_option('390')
        expect(snapshot.locator('.app-shell > .sidebar')).to_have_css('display', 'grid')
        snapshot.locator('#mobileMenu').click()
        expect(snapshot.locator('#sidebar')).to_have_class(re.compile(r'\bopen\b'))
        backdrop = snapshot.locator('#sidebarBackdrop')
        backdrop.click(position={'x': backdrop.bounding_box()['width'] - 5, 'y': 5})
        expect(snapshot.locator('#sidebar')).not_to_have_class(re.compile(r'\bopen\b'))
        self.screenshot('customizer-mobile-preview')
        self.page.set_viewport_size({'width': 390, 'height': 844})
        editor.locator('.customizer-size').scroll_into_view_if_needed()
        expect(snapshot).to_be_visible()
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        self.screenshot('customizer-mobile-editor')

    def test_shell_widget_stays_mounted_across_pages_and_can_move_or_unload(self):
        self.install_test_plugin()
        self.page.locator('[data-view-button="plugin-monitor"]').click()
        self.install_test_plugin(slot='topbar', size={'width': 280, 'height': 44})
        expect(self.page.locator('#profile-browser')).to_be_visible()
        widget = self.page.locator('.plugin-shell-topbar iframe')
        frame = self.page.frame_locator('.plugin-shell-topbar iframe')
        expect(frame.locator('#value')).to_have_text('first:true:true')
        expect(self.page.locator('[data-view-button="plugin-monitor"]')).to_have_count(0)
        self.assertEqual(widget.bounding_box()['height'], 44)
        self.page.evaluate("window.widgetIdentity = document.querySelector('.plugin-shell-topbar iframe')")
        for view in ('webui', 'hardware', 'launch'):
            self.page.locator(f'[data-view-button="{view}"]').click()
            expect(widget).to_be_visible()
            self.assertTrue(self.page.evaluate("window.widgetIdentity === document.querySelector('.plugin-shell-topbar iframe')"))
        for width in (390, 320):
            self.page.set_viewport_size({'width': width, 'height': 844})
            expect(widget).to_be_visible()
            self.assertLessEqual(widget.bounding_box()['width'], width)
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            self.page.locator('.navigation > .plugin-manager-button').click()
            self.page.locator('body > .plugin-manager [aria-label="关闭"]').click()
        self.install_test_plugin('second', slot='sidebar', size={'height': 60})
        expect(self.page.locator('.plugin-shell-topbar')).to_be_hidden()
        expect(self.page.frame_locator('.plugin-shell-sidebar iframe').locator('#value')).to_have_text('second:true:true')
        self.install_test_plugin('third', slot='statusbar', size={'height': 32})
        expect(self.page.locator('.plugin-shell-sidebar')).to_be_hidden()
        expect(self.page.frame_locator('.plugin-shell-statusbar iframe').locator('#value')).to_have_text('third:true:true')
        self.runtime.plugins.set_enabled('monitor', False)
        self.page.clock.fast_forward(2100)
        expect(self.page.locator('.plugin-shell-statusbar')).to_be_hidden()
        self.assertIsNotNone(self.runtime._process)

    def test_skin_hot_updates_launcher_and_studio_without_losing_draft(self):
        from test_ui_plugins import bundle, theme_bundle
        self.page.locator('#open-webui').click(); self.assert_loaded()
        pane = self.page.locator('body > .app-shell #webui-content')
        pane.locator('#prompt').fill('Keep draft through skin updates')
        self.page.evaluate("window.skinHost = document.querySelector('#webui-content > div')")
        widget = bundle(slot='topbar', capabilities=[])
        widget['index.html'] = '''<!doctype html><output id="palette"></output><script>
          addEventListener('ftllm-context', e => {
            document.querySelector('#palette').textContent = e.detail.palette.primary || 'default';
          });</script>'''
        self.runtime.plugins.apply('monitor', widget, '')
        palette = self.page.frame_locator('body > .app-shell .plugin-shell-topbar iframe').locator('#palette')
        files = theme_bundle(light={'primary': '#2563eb', 'background': '#eef4ff', 'sidebar': '#dae7ff'},
                             dark={'primary': '#60a5fa', 'background': '#101827', 'sidebar': '#18233b'},
                             layout={'sidebarSide': 'right', 'sidebarWidth': 240, 'radius': 16, 'font': 'serif'})
        first = self.runtime.plugins.apply('blue-skin', files, '')
        self.page.clock.fast_forward(2100)
        expect(self.page.locator('html')).to_have_attribute('data-plugin-skin', 'blue-skin')
        expect(self.page.locator('body')).to_have_css('background-color', 'rgb(238, 244, 255)')
        expect(pane.locator('.main')).to_have_css('background-color', 'rgb(238, 244, 255)')
        expect(pane.locator('#newChat')).to_have_css('background-color', 'rgb(37, 99, 235)')
        expect(palette).to_have_text('#2563eb')
        sidebar = self.page.locator('.app-shell > .sidebar')
        self.assertEqual(sidebar.bounding_box()['width'], 240)
        self.assertGreater(sidebar.bounding_box()['x'], self.page.locator('.app-shell > .workspace').bounding_box()['x'])
        self.screenshot('plugin-skin-light')
        self.page.locator('#theme-select').select_option('dark')
        expect(self.page.locator('body')).to_have_css('background-color', 'rgb(16, 24, 39)')
        expect(pane.locator('.main')).to_have_css('background-color', 'rgb(16, 24, 39)')
        expect(palette).to_have_text('#60a5fa')
        self.screenshot('plugin-skin-dark')
        for width in (390, 320):
            self.page.set_viewport_size({'width': width, 'height': 844})
            self.page.clock.run_for(300)
            self.assertEqual(sidebar.bounding_box()['x'], 0)
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            expect(pane.locator('#prompt')).to_be_visible()
            pane.locator('#prompt').click()
        self.page.set_viewport_size({'width': 1280, 'height': 720})
        second = self.runtime.plugins.apply('blue-skin', theme_bundle(dark={'background': '#24182a'}), first['revision'])
        self.page.clock.fast_forward(2100)
        expect(pane.locator('.main')).to_have_css('background-color', 'rgb(36, 24, 42)')
        self.runtime.plugins.rollback('blue-skin', second['revision'])
        self.page.clock.fast_forward(2100)
        expect(pane.locator('.main')).to_have_css('background-color', 'rgb(16, 24, 39)')
        self.page.locator('.navigation > .plugin-manager-button').click()
        self.page.locator('body > .plugin-manager .customizer-library > summary').click()
        self.page.locator('body > .plugin-manager .plugin-reset-theme').click()
        expect(self.page.locator('html')).not_to_have_attribute('data-plugin-skin', 'blue-skin')
        expect(palette).to_have_text('default')
        self.page.clock.fast_forward(2100)
        expect(pane.locator(':scope > div')).not_to_have_attribute('data-plugin-skin', 'blue-skin')
        expect(pane.locator('#prompt')).to_have_value('Keep draft through skin updates')
        self.assertTrue(self.page.evaluate("window.skinHost === document.querySelector('#webui-content > div')"))
        self.assertEqual(self.page.locator('html').evaluate("e => e.style.getPropertyValue('--bg')"), '')

    def test_skin_generation_preview_and_recovery_are_readable(self):
        from test_ui_plugins import theme_bundle
        self.page.locator('#open-webui').click(); self.assert_loaded()
        model = self.runtime._webui_app.state.runtime.api_client
        files = theme_bundle(light={key: '#ffffff' for key in ('primary', 'background', 'surface', 'text', 'mutedText', 'border')})
        with patch.object(model, 'stream', side_effect=lambda *a, **k: iter([(json.dumps({'summary': 'Custom skin', 'files': files}), '')])) as complete:
            self.page.locator('.navigation > .plugin-manager-button').click()
            dialog = self.page.locator('body > .plugin-manager')
            dialog.locator('.customizer-options > summary').click()
            dialog.locator('[name="id"]').fill('blue-skin')
            dialog.locator('[name="target"]').select_option('theme')
            dialog.locator('[name="instruction"]').fill('更换主界面皮肤')
            dialog.locator('.plugin-editor [type=submit]').click()
            expect(dialog.locator('.plugin-preview')).to_be_visible()
            self.assertFalse((self.runtime.plugins.directory / 'blue-skin').exists())
            self.assertEqual(json.loads(complete.call_args.args[0][1]['content'])['target'], 'theme')
            dialog.locator('.plugin-apply').click()
            expect(self.page.locator('html')).to_have_attribute('data-plugin-skin', 'blue-skin')
            expect(dialog).to_have_css('color', 'rgb(37, 53, 46)')
            dialog.locator('.plugin-heading [aria-label="关闭"]').click()
            recovery = self.page.locator('.navigation > .plugin-manager-button')
            expect(recovery).to_have_css('color', 'rgb(55, 65, 81)')
            recovery.click()
            dialog.locator('.customizer-library > summary').click()
            dialog.locator('.plugin-reset-theme').click()
            expect(self.page.locator('html')).not_to_have_attribute('data-plugin-skin', 'blue-skin')

    def test_standalone_studio_skin_applies_and_disables(self):
        from test_ui_plugins import theme_bundle
        self.runtime.plugins.apply('blue-skin', theme_bundle(light={'primary': '#2563eb', 'background': '#eef4ff'}), '')
        self.page.goto(self.url + '/standalone/')
        pane = self.page.locator('#webui-root')
        expect(pane.locator('.main')).to_have_css('background-color', 'rgb(238, 244, 255)')
        expect(pane.locator('#newChat')).to_have_css('background-color', 'rgb(37, 99, 235)')
        pane.locator('#prompt').fill('Standalone draft')
        self.runtime.plugins.set_enabled('blue-skin', False)
        self.page.clock.fast_forward(2100)
        expect(pane.locator('[data-plugin-skin]')).to_have_count(0)
        expect(pane.locator('#prompt')).to_have_value('Standalone draft')

    def test_theme_defaults_to_light_and_remembers_choice(self):
        picker = self.page.locator('#theme-select')
        root = self.page.locator('html')
        self.page.emulate_media(color_scheme='dark')
        self.page.reload()
        expect(picker.locator('option')).to_have_count(2)
        expect(picker).to_have_value('light')
        expect(root).to_have_attribute('data-theme', 'light')
        self.page.emulate_media(color_scheme='light')
        self.page.emulate_media(color_scheme='dark')
        expect(root).to_have_attribute('data-theme', 'light')
        picker.select_option('dark')
        self.page.emulate_media(color_scheme='light')
        self.page.reload()
        expect(picker).to_have_value('dark')
        expect(root).to_have_attribute('data-theme', 'dark')
        self.page.locator('#language-select').select_option('zh-CN')
        expect(picker).to_have_value('dark')
        expect(picker.locator('option:checked')).to_have_text('黑夜模式')
        picker.select_option('light')
        self.page.reload()
        expect(picker).to_have_value('light')
        expect(root).to_have_attribute('data-theme', 'light')
        # Older saved system preferences fall back to the new light default.
        self.page.evaluate("localStorage.setItem('ftllm-launcher-theme', 'system')")
        self.page.emulate_media(color_scheme='dark')
        self.page.reload()
        expect(picker).to_have_value('light')
        expect(root).to_have_attribute('data-theme', 'light')

    def test_saved_theme_is_applied_before_main_script_runs(self):
        self.page.evaluate("localStorage.setItem('ftllm-launcher-theme', 'dark')")
        self.page.emulate_media(color_scheme='light')
        self.page.route('**/assets/app.js', lambda route: route.abort())
        self.page.reload()
        expect(self.page.locator('html')).to_have_attribute('data-theme', 'dark')
        self.assertEqual(self.page.locator('body').evaluate('node => getComputedStyle(node).colorScheme'), 'dark')
        self.assert_dark_surface(self.page.locator('body'))

    def assert_dark_surface(self, locator):
        # Wait for existing hover/color transitions to finish after a theme change.
        expect(locator).to_have_css('background-color', re.compile(r'rgb\([0-8]?\d, [0-8]?\d, [0-8]?\d\)'))

    def test_dark_theme_updates_studio_dialogs_and_mobile_without_losing_draft(self):
        self.page.locator('#open-webui').click()
        self.assert_loaded()
        pane = self.page.locator('#webui-content')
        pane.locator('#prompt').fill('Keep this draft across theme changes')
        self.page.evaluate("window.themeTestHost = document.querySelector('#webui-content > div')")
        self.page.locator('#theme-select').select_option('dark')
        expect(pane.locator(':scope > div')).to_have_attribute('data-theme', 'dark')
        self.assert_dark_surface(pane.locator('.main'))
        self.assert_dark_surface(pane.locator('.composer'))
        self.assert_dark_surface(pane.locator('.suggestion').first)
        self.screenshot('launcher-dark-studio')
        pane.locator('#agentButton').click()
        expect(pane.locator('#agentDialog')).to_be_visible()
        self.assert_dark_surface(pane.locator('#agentDialog'))
        self.screenshot('launcher-dark-agent-dialog')
        pane.locator('#agentDialog').evaluate('node => node.close()')
        self.page.locator('#theme-select').select_option('light')
        expect(pane.locator(':scope > div')).to_have_attribute('data-theme', 'light')
        expect(pane.locator('#prompt')).to_have_value('Keep this draft across theme changes')
        self.assertTrue(self.page.evaluate("window.themeTestHost === document.querySelector('#webui-content > div')"))
        self.page.locator('#theme-select').select_option('dark')
        self.page.locator('[data-view-button="launch"]').click()
        self.page.locator('#new-profile').click()
        self.assert_dark_surface(self.page.locator('#launch-form'))
        self.assert_dark_surface(self.page.locator('#model-path'))
        self.screenshot('launcher-dark-profile-dialog')
        self.page.locator('#close-profile-editor').click()
        self.assert_dark_surface(self.page.locator('.confirmation-window'))
        self.page.locator('#confirmation-confirm').click()
        self.screenshot('launcher-dark-models')
        self.page.locator('#language-select').select_option('zh-CN')
        for width in (390, 320):
            self.page.set_viewport_size({'width': width, 'height': 844})
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            expect(self.page.locator('#theme-select')).to_be_visible()
            expect(self.page.locator('#language-select')).to_be_visible()
            expect(self.page.locator('#shutdown-launcher')).to_be_visible()
        self.screenshot('launcher-dark-mobile')

    def test_markdown_tables_lists_links_and_streaming_code_survive_reload(self):
        self.page.locator('#open-webui').click()
        self.assert_loaded()
        pane = self.page.locator('#webui-content')
        partial = threading.Event()
        release = threading.Event()
        source = (
            '## 功能介绍\n\n'
            '| 功能 | 说明 | 分数 |\n| :--- | :---: | ---: |\n'
            '| **操作** | 方向键 / WASD | 10 |\n'
            '| `a\\|b` | *空格暂停*<br>继续 | 20 |\n\n'
            '> 引用 **说明**\n\n'
            '3. 第一步\n   - 子项目\n4. 第二步\n\n'
            '- [x] 已完成\n- [ ] 未完成\n\n'
            '~~旧说明~~ [参数介绍](https://example.com/parameters)\n\n'
            '[危险链接](javascript:alert(1)) <img src=x onerror=alert(1)>\n\n'
            '```html\n<h1>Still streaming</h1>')

        def stream(*args, **kwargs):
            yield source, '| 思考 | 状态 |\n| --- | --- |\n| 分析 | 完成 |'
            partial.set()
            release.wait(10)
            yield '\n```', ''

        with patch.object(self.runtime._webui_app.state.runtime.api_client, 'stream', side_effect=stream):
            try:
                pane.locator('#prompt').fill('Show Markdown')
                pane.locator('#sendButton').click()
                self.assertTrue(partial.wait(5))
                message = pane.locator('.message.assistant > .message-body > .message-text')
                expect(message.locator('tbody tr')).to_have_count(2)
                expect(message.locator('th')).to_have_text(['功能', '说明', '分数'])
                expect(message.locator('tbody td code')).to_have_text('a|b')
                expect(message.locator('tbody td').nth(1)).to_have_css('text-align', 'center')
                expect(message.locator('tbody td').nth(2)).to_have_css('text-align', 'right')
                expect(message.locator('blockquote strong')).to_have_text('说明')
                expect(message.locator('ol')).to_have_attribute('start', '3')
                expect(message.locator('ol ul li')).to_have_text('子项目')
                expect(message.locator('input[type="checkbox"]')).to_have_count(2)
                expect(message.locator('input[type="checkbox"]').first).to_be_checked()
                expect(message.locator('del')).to_have_text('旧说明')
                expect(message.locator('a')).to_have_count(1)
                expect(message.locator('a')).to_have_attribute('rel', 'noopener noreferrer')
                expect(message.locator('img,script')).to_have_count(0)
                expect(message.locator('.code-block code')).to_have_text('<h1>Still streaming</h1>')
                expect(message.locator('.preview-code')).to_be_visible()
                pane.locator('.reasoning summary').click()
                expect(pane.locator('.reasoning-content table')).to_be_visible()
            finally:
                release.set()
            expect(pane.locator('#stopButton')).to_be_hidden()
        self.page.reload()
        self.page.locator('#open-webui').click()
        expect(pane.locator('.message.assistant > .message-body > .message-text table')).to_be_visible()
        self.page.locator('#theme-select').select_option('dark')
        self.screenshot('markdown-dark')
        self.page.set_viewport_size({'width': 390, 'height': 844})
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        self.screenshot('markdown-mobile')

    def test_html_preview_runs_scripts_with_isolated_storage_and_closes_cleanly(self):
        self.page.locator('#open-webui').click()
        self.assert_loaded()
        pane = self.page.locator('#webui-content')
        source = '''<!doctype html><html><head><style>
body { background: rgb(12, 24, 36); color: white; font: 24px sans-serif; }
</style></head><body><h1>HTML demo</h1><button id="counter">0</button>
<p id="isolation"></p><script>
localStorage.setItem('preview-test', 'inside');
let isolated = false;
try { parent.document.body.dataset.previewEscaped = 'yes'; } catch (_) { isolated = true; }
document.querySelector('#isolation').textContent = isolated ? 'Isolated' : 'Unsafe';
document.querySelector('#counter').onclick = event => {
  event.target.textContent = Number(event.target.textContent) + 1;
};
</script></body></html>'''
        with patch.object(self.runtime._webui_app.state.runtime.api_client, 'stream',
                          side_effect=lambda *a, **k: iter([('```HTML\n' + source + '\n```', '')])):
            pane.locator('#prompt').fill('Build a page')
            pane.locator('#sendButton').click()
            expect(pane.locator('#stopButton')).to_be_hidden()
            expect(pane.locator('.preview-code')).to_be_visible()
        self.page.evaluate("localStorage.setItem('preview-test', 'outside')")
        pane.locator('#prompt').fill('Preserve this draft')
        pane.locator('.preview-code').click()
        dialog = pane.locator('#htmlPreviewDialog')
        expect(dialog).to_be_visible()
        frame = pane.frame_locator('#htmlPreviewBody iframe')
        expect(frame.locator('h1')).to_have_text('HTML demo')
        expect(frame.locator('body')).to_have_css('background-color', 'rgb(12, 24, 36)')
        expect(frame.locator('#isolation')).to_have_text('Isolated')
        frame.locator('#counter').click()
        expect(frame.locator('#counter')).to_have_text('1')
        self.assertEqual(self.page.evaluate("localStorage.getItem('preview-test')"), 'outside')
        self.assertIsNone(self.page.locator('body').get_attribute('data-preview-escaped'))
        self.screenshot('html-preview')
        pane.locator('#closeHTMLPreview').click()
        expect(dialog).to_be_hidden()
        expect(pane.locator('iframe')).to_have_count(0)
        expect(pane.locator('#prompt')).to_have_value('Preserve this draft')
        self.page.locator('#language-select').select_option('zh-CN')
        self.page.locator('#theme-select').select_option('dark')
        expect(pane.locator('.preview-code')).to_have_text('预览')
        pane.locator('.preview-code').click()
        expect(frame.locator('#counter')).to_have_text('0')
        self.page.set_viewport_size({'width': 390, 'height': 844})
        self.assertLessEqual(dialog.bounding_box()['width'], 390)
        self.screenshot('html-preview-mobile')
        pane.locator('#closeHTMLPreview').click()
        pane.locator('.preview-code').click()
        self.page.keyboard.press('Escape')
        expect(dialog).to_be_hidden()
        expect(pane.locator('iframe')).to_have_count(0)

    def test_pi_install_retry_enables_agent_and_preserves_draft(self):
        job = {'phase': 'missing', 'supported': True, 'available': False, 'error': ''}
        installed = False

        def load_runtime():
            if not installed:
                raise ImportError('Pi runtime missing')
            return lambda **kwargs: SimpleNamespace(info=lambda: {
                'available': True, 'pi_version': '0.84.4', 'tools': ['read', 'find', 'grep']})

        def start():
            job.update(phase='installing', component='pip')
            return dict(job)

        with patch.object(self.runtime._agent_installer, 'state', side_effect=lambda: dict(job)), \
                patch.object(self.runtime._agent_installer, 'start', side_effect=start) as install, \
                patch('fastllm_pytools.agent_runtime_install.load_pi_agent_runtime', side_effect=load_runtime):
            self.page.reload()
            self.page.locator('[data-view-button="webui"]').click()
            expect(self.page.locator('#webui-content #prompt')).to_be_visible()
            expect(self.page.locator('#install-agent-runtime')).to_be_enabled()
            expect(self.page.locator('#install-agent-runtime')).to_have_text('Install Agent dependencies')
            expect(self.page.locator('#agent-runtime-message')).to_contain_text('pip')
            expect(self.page.locator('#newAgent')).to_be_disabled()
            expect(self.page.locator('#installRuntime')).to_be_visible()
            self.screenshot('launcher-pi-install-en')
            self.page.locator('#language-select').select_option('zh-CN')
            expect(self.page.locator('#install-agent-runtime')).to_have_text('安装 Agent 依赖')
            self.page.set_viewport_size({'width': 390, 'height': 844})
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            self.screenshot('launcher-pi-install-zh-mobile')
            self.page.set_viewport_size({'width': 1280, 'height': 720})
            self.page.locator('#language-select').select_option('en-US')
            self.page.locator('#prompt').fill('Keep this draft while installing Pi')
            self.page.locator('#installRuntime').click()
            expect(self.page.locator('#install-agent-runtime')).to_be_disabled()
            expect(self.page.locator('#agent-runtime-progress')).to_be_visible()
            expect(self.page.locator('#agent-runtime-message')).to_have_text('Installing Agent dependencies with pip…')
            self.assertIsNone(self.page.locator('#agent-runtime-progress').get_attribute('value'))
            install.assert_called_once_with()

            job.update(phase='failed', error='Temporary download failure')
            self.page.clock.run_for(1100)
            expect(self.page.locator('#agent-runtime-error')).to_have_text('Temporary download failure')
            expect(self.page.locator('#install-agent-runtime')).to_have_text('Retry installation')
            self.page.locator('#install-agent-runtime').click()
            self.assertEqual(install.call_count, 2)

            installed = True
            self.runtime._enable_installed_agent()
            job.update(phase='ready', available=True, error='')
            self.page.route('**/webui/**/api/config', lambda route: route.abort())
            self.page.clock.run_for(1100)
            expect(self.page.locator('#newAgent')).to_be_disabled()
            self.page.unroute('**/webui/**/api/config')
            self.page.clock.run_for(1100)
            expect(self.page.locator('#agent-runtime-card')).to_be_hidden()
            expect(self.page.locator('#newAgent')).to_be_enabled()
            expect(self.page.locator('#installRuntime')).to_be_hidden()
            expect(self.page.locator('#prompt')).to_have_value('Keep this draft while installing Pi')

    def test_launch_item_can_be_added_saved_and_edited(self):
        editor = self.page.locator('#profile-editor-modal')
        for selector in ('#new-profile', '[data-new-profile]'):
            with self.subTest(entrypoint=selector):
                self.page.locator(selector).click()
                expect(editor).to_be_visible()
                expect(self.page.locator('#profile-editor-title')).to_have_text('Add launch item')
                expect(editor.locator('[data-field="model"]')).to_be_focused()
                self.page.locator('#close-profile-editor').click()
                self.page.locator('#confirmation-confirm').click()
                expect(editor).to_be_hidden()

        self.page.locator('#new-profile').click()
        editor.locator('[data-config-mode][value="custom"]').check()
        editor.locator('[data-field="device"]').select_option('cpu')
        editor.locator('[data-field="name"]').fill('Saved launch item')
        model_path = os.path.join(self.temp.name, 'model')
        os.mkdir(model_path)
        editor.locator('[data-field="model"]').fill(model_path)
        editor.locator('[data-field="chunked_prefill_size"]').fill('8192')
        with patch('fastllm_pytools.launcher.detect_hardware', return_value={'gpus': []}):
            self.page.locator('#auto-configure-profile').click()
            expect(self.page.locator('#automatic-config-status')).to_have_class('automatic-config-status success')
        expect(editor.locator('[data-field="chunked_prefill_size"]')).to_have_value('auto')
        expect(editor.locator('[data-field="dtype"]')).to_have_count(0)
        cache = editor.locator('#editor-basic [data-field="kv_cache_dtype"]')
        expect(cache).to_be_visible()
        cache.select_option('fp8_e4m3')
        context_length = editor.locator('[data-field="max_context_length"]')
        expect(context_length).to_be_visible()
        context_length.fill('4096')
        self.page.locator('#save-profile').click()
        expect(editor).to_be_hidden()
        expect(self.page.locator('#profile-count')).to_have_text('1')
        self.assertEqual(self.runtime.profiles()[0]['max_context_length'], '4096')
        self.assertEqual(self.runtime.profiles()[0]['kv_cache_dtype'], 'fp8_e4m3')
        self.assertEqual(self.runtime.profiles()[0]['chunked_prefill_size'], 'auto')

        self.page.reload()
        self.page.locator('[data-profile-action="edit"]').click()
        expect(editor).to_be_visible()
        expect(editor.locator('[data-field="name"]')).to_have_value('Saved launch item')
        expect(editor.locator('[data-field="model"]')).to_have_value(model_path)
        expect(context_length).to_have_value('4096')
        expect(cache).to_have_value('fp8_e4m3')
        self.page.locator('#auto-configure-profile').click()
        chooser = self.page.locator('#automatic-config-dialog')
        expect(chooser.locator('input[name="automatic-configuration-mode"]')).to_have_count(2)
        expect(chooser.locator('input[value="custom"]')).to_have_count(0)
        expect(chooser.locator('input[value="long_context"]')).to_be_checked()
        self.page.locator('#automatic-config-dialog-cancel').click()
        expect(editor.locator('[data-config-mode][value="custom"]')).to_be_checked()
        expect(context_length).to_have_value('4096')
        self.page.locator('#close-profile-editor').click()
        expect(editor).to_be_hidden()
        self.assertEqual(self.runtime.profiles()[0]['config_mode'], 'custom')

    def test_inference_speed_bar_updates_across_views_and_service_restarts(self):
        bar = self.page.locator('#inference-status-bar')
        expect(bar).to_be_visible()
        expect(self.page.locator('.topbar #inference-status-bar')).to_be_visible()
        expect(self.page.locator('#prefill-speed')).to_have_text('0')
        expect(self.page.locator('#decode-speed')).to_have_text('0')
        self.runtime._handle_inference_speed(
            '[Prompt] 8192 Tokens. Speed: 4321.25 tokens / s.', self.runtime._generation)
        self.runtime._handle_inference_speed(
            '[Decode] alive = 1, pending = 0, context usages: 5.0%, Speed: 108.65 tokens / s.',
            self.runtime._generation)
        self.page.clock.fast_forward(1000)
        expect(self.page.locator('#prefill-speed')).to_have_text('4,321.3')
        expect(self.page.locator('#decode-speed')).to_have_text('108.7')
        self.page.locator('[data-view-button="logs"]').click()
        expect(bar).to_be_visible()
        self.page.clock.fast_forward(1500)
        expect(self.page.locator('#decode-speed')).to_have_text('108.7')
        self.page.clock.fast_forward(2000)
        expect(self.page.locator('#prefill-speed')).to_have_text('0')
        expect(self.page.locator('#decode-speed')).to_have_text('0')
        self.runtime._handle_inference_speed(
            '[Decode] alive = 1, pending = 0, context usages: 5.0%, Speed: 108.65 tokens / s.',
            self.runtime._generation)
        self.page.clock.fast_forward(1000)
        expect(self.page.locator('#decode-speed')).to_have_text('108.7')
        expect(self.page.locator('#prefill-speed')).to_have_text('0')
        self.page.locator('#open-webui').click()
        self.assert_loaded()
        expect(bar).to_be_visible()
        expect(self.page.locator('#decode-speed')).to_have_text('0')
        with self.runtime._lock:
            self.runtime._state.update(phase='stopped', ready=False)
        self.page.clock.fast_forward(1000)
        expect(bar).to_be_hidden()
        with self.runtime._lock:
            self.runtime._state.update(phase='starting', ready=False,
                                       speed={key: None for key in self.runtime.state()['speed']})
        self.page.clock.fast_forward(1000)
        expect(bar).to_be_hidden()
        with self.runtime._lock:
            self.runtime._state.update(phase='running', ready=True, sessionId='model-b')
        self.page.clock.fast_forward(1000)
        expect(bar).to_be_visible()
        expect(self.page.locator('#decode-speed')).to_have_text('0')

    def test_context_capacity_stays_visible_when_speeds_expire_and_resets_on_restart(self):
        context = self.page.locator('#context-window')
        metric = self.page.locator('#context-window-metric')
        expect(context).to_have_text('—')
        expect(metric).to_have_attribute('title', 'Context capacity has not been reported yet.')
        self.runtime._handle_context_window(
            'INFO: Model context window: 262144 tokens per session '
            '(model=262144, shared KV cache=335360, configured limit=None)', self.runtime._generation)
        self.page.clock.fast_forward(1000)
        expect(context).to_have_text('256K')
        expect(metric).to_have_attribute('title',
            'Available context per session (input + output): 262,144 tokens. 1K = 1024 tokens.')
        self.page.locator('[data-view-button="logs"]').click()
        self.page.clock.fast_forward(4000)
        expect(context).to_be_visible()
        expect(context).to_have_text('256K')
        expect(self.page.locator('#decode-speed')).to_have_text('0')
        self.page.reload()
        expect(context).to_have_text('256K')
        self.page.clock.install()
        with self.runtime._lock:
            self.runtime._state.update(phase='starting', ready=False,
                                       sessionId='model-b', contextWindowTokens=None)
        self.page.clock.fast_forward(1000)
        expect(context).to_be_hidden()
        with self.runtime._lock:
            self.runtime._state.update(phase='running', ready=True)
        self.page.clock.fast_forward(1000)
        expect(context).to_be_visible()
        expect(context).to_have_text('—')
        self.runtime._handle_context_window(
            'INFO: Model context window: 167168 tokens per session '
            '(model=262144, shared KV cache=167168, configured limit=None)', self.runtime._generation)
        self.page.clock.fast_forward(1000)
        expect(context).to_have_text('163.25K')

    def test_speed_timeout_works_while_runtime_poll_is_stalled(self):
        self.runtime._handle_inference_speed(
            '[Decode] alive = 1, pending = 0, contextLen = 128, Speed: 42.0 tokens / s.',
            self.runtime._generation)
        self.page.clock.fast_forward(1000)
        expect(self.page.locator('#decode-speed')).to_have_text('42')
        pending = []
        self.page.route('**/api/runtime', lambda route: pending.append(route), times=1)
        with self.page.expect_request('**/api/runtime'):
            self.page.clock.fast_forward(700)
        self.page.clock.fast_forward(3000)
        expect(self.page.locator('#decode-speed')).to_have_text('0')
        with self.page.expect_response('**/api/runtime') as response:
            pending[0].fulfill(status=200, content_type='application/json', body=json.dumps(self.runtime.state()))
        response.value.body()
        expect(self.page.locator('#decode-speed')).to_have_text('0')
        self.assertEqual(len(pending), 1)

    def test_presets_stay_simple_and_editing_reuses_the_saved_mode(self):
        model_path = os.path.join(self.temp.name, 'model')
        os.mkdir(model_path)
        with open(os.path.join(model_path, 'config.json'), 'w') as output:
            json.dump({'max_position_embeddings': 32768}, output)
        hardware = {'cpu': {'available': 4}, 'memory': {'available': 16 * 1024 ** 3},
                    'gpus': [], 'numa': [], 'build': {}}
        editor = self.page.locator('#profile-editor-modal')
        with patch('fastllm_pytools.launcher.detect_hardware', return_value=hardware):
            for index, (mode, batch, context) in enumerate((('long_context', '1', 'auto'),
                                                           ('high_concurrency', 'auto', 'auto'))):
                with self.subTest(mode=mode):
                    self.page.locator('#new-profile').click()
                    expect(editor.locator('[data-config-mode][value="long_context"]')).to_be_checked()
                    editor.locator(f'[data-config-mode][value="{mode}"]').check()
                    expect(self.page.locator('#profile-parameters')).to_be_hidden()
                    expect(self.page.locator('#auto-configure-profile')).to_be_hidden()
                    expect(self.page.locator('#save-profile')).to_be_disabled()
                    editor.locator('[data-field="model"]').fill(model_path)
                    self.page.clock.fast_forward(700)
                    expect(self.page.locator('#automatic-config-status')).to_have_class('automatic-config-status success')
                    self.page.locator('#save-profile').click()
                    expect(editor).to_be_hidden()
                    saved = self.runtime.profiles()[index]
                    self.assertEqual((saved['config_mode'], saved['max_batch'], saved['max_context_length']),
                                     (mode, batch, context))
                    self.assertEqual(saved['low_gpu_mem'], mode == 'long_context')
                    self.assertEqual(saved['chunked_prefill_size'], 'auto')
                    self.assertNotIn('--chunked_prefill_size', self.runtime.preview(saved)['command'])
                    self.runtime.save_profile(index, {**saved, 'dtype': 'float16'})
                    self.page.reload()
                    self.page.locator(f'[data-profile-action="edit"][data-profile-index="{index}"]').click()
                    expect(self.page.locator('#configuration-mode-settings')).to_be_hidden()
                    expect(editor.locator(f'[data-config-mode][value="{mode}"]')).to_be_checked()
                    expect(editor.locator('[data-field="max_context_length"]')).to_be_visible()
                    self.assertEqual(editor.locator('[data-field="low_gpu_mem"]').is_checked(),
                                     mode == 'long_context')
                    editor.locator('[data-field="max_batch"]').fill('3')
                    expect(editor.locator('[data-field="dtype"]')).to_have_count(0)
                    editor.locator('[data-field="moe_dtype"]').select_option('int8')
                    editor.locator('[data-field="chunked_prefill_size"]').fill('1024')
                    self.page.locator('#auto-configure-profile').click()
                    chooser = self.page.locator('#automatic-config-dialog')
                    expect(chooser).to_be_visible()
                    expect(chooser.locator('input[name="automatic-configuration-mode"]')).to_have_count(2)
                    expect(chooser.locator('input[value="custom"]')).to_have_count(0)
                    expect(chooser.locator(f'input[name="automatic-configuration-mode"][value="{mode}"]')).to_be_checked()
                    with self.page.expect_request('**/api/recommend') as request:
                        self.page.locator('#automatic-config-dialog-apply').click()
                    self.assertEqual(request.value.post_data_json['config_mode'], mode)
                    expect(chooser).to_be_hidden()
                    expect(editor.locator('[data-field="max_batch"]')).to_have_value(batch)
                    expect(editor.locator('[data-field="max_context_length"]')).to_have_value(context)
                    expect(editor.locator('[data-field="moe_dtype"]')).to_have_value('int8')
                    expect(editor.locator('[data-field="chunked_prefill_size"]')).to_have_value('auto')
                    editor.locator('[data-field="low_gpu_mem"]').check()
                    editor.locator('[data-field="kv_cache_dtype"]').select_option('fp4')
                    self.page.clock.fast_forward(700)
                    expect(self.page.locator('#command-preview')).to_contain_text('--low_gpu_mem')
                    expect(self.page.locator('#command-preview')).to_contain_text('--kv_cache_dtype fp4')
                    expect(self.page.locator('#command-preview')).to_contain_text('--dtype float16')
                    expect(self.page.locator('#command-preview')).not_to_contain_text('--chunked_prefill_size')
                    self.page.locator('#close-profile-editor').click()
                    self.page.locator('#confirmation-confirm').click()
                    expect(editor).to_be_hidden()

    def test_model_picker_selects_both_directories_and_files(self):
        folder = os.path.join(self.temp.name, 'model folder')
        os.mkdir(folder)
        model_file = os.path.join(folder, 'model test.gguf')
        with open(model_file, 'w') as output:
            output.write('GGUF')
        self.page.locator('#new-profile').click()
        self.page.locator('[data-config-mode][value="custom"]').check()
        model = self.page.locator('#model-path')
        model.fill(self.temp.name)
        self.page.locator('#choose-model-folder').click()
        self.page.locator('#folder-picker-list button').filter(has_text='model folder').click()
        expect(self.page.locator('#folder-picker-current')).to_have_value(folder)
        self.page.locator('#folder-picker-select').click()
        expect(model).to_have_value(folder)
        self.page.locator('#choose-model-folder').click()
        self.page.locator('#folder-picker-list button').filter(has_text='model test.gguf').click()
        expect(self.page.locator('#folder-picker-select')).to_have_text('Select this file')
        self.page.locator('#folder-picker-select').click()
        expect(model).to_have_value(model_file)
        expect(self.page.locator('#folder-picker-modal')).to_be_hidden()

    def test_model_picker_switches_windows_drives_and_recovers_from_unavailable_volume(self):
        drives = [{'name': letter + ':', 'path': letter + ':\\'} for letter in ('C', 'D', 'E')]
        folder = 'D:\\模型 目录'
        model_file = folder + '\\model.gguf'
        share = '\\\\nas\\models'

        def browse(path=''):
            path = path or 'C:\\'
            if path == 'E:\\':
                from fastllm_pytools.launcher import LauncherError
                raise LauncherError('Folder root is unavailable.')
            selected = model_file if path == model_file else ''
            if selected:
                path = folder
            return {'path': path, 'parent': 'D:\\' if path == folder else '', 'drives': drives,
                    'folders': [{'name': '模型 目录', 'path': folder}] if path == 'D:\\' else [],
                    'files': [{'name': 'model.gguf', 'path': model_file}] if path == folder else [],
                    'selectedFile': selected, 'truncated': False}

        self.page.locator('#new-profile').click()
        self.page.locator('[data-config-mode][value="custom"]').check()
        with patch('fastllm_pytools.launcher.browse_folders', side_effect=browse):
            self.page.locator('#choose-model-folder').click()
            drive = self.page.locator('#folder-picker-drive')
            location = self.page.locator('#folder-picker-current')
            expect(drive).to_be_visible()
            expect(drive).to_have_value('C:\\')
            expect(self.page.locator('#folder-picker-up')).to_be_disabled()
            drive.select_option('E:\\')
            expect(self.page.locator('#folder-picker-status')).to_contain_text('Folder root is unavailable.')
            expect(self.page.locator('#folder-picker-select')).to_be_disabled()
            drive.select_option('D:\\')
            expect(location).to_have_value('D:\\')
            self.page.locator('#folder-picker-list button').filter(has_text='模型 目录').click()
            expect(location).to_have_value(folder)
            self.page.locator('#folder-picker-up').click()
            expect(location).to_have_value('D:\\')
            location.fill(share)
            location.press('Enter')
            expect(drive).to_have_value('')
            expect(location).to_have_value(share)
            location.fill(model_file)
            location.press('Enter')
            expect(self.page.locator('#folder-picker-select')).to_have_text('Select this file')
            expect(location).to_have_value(folder)
            expect(drive).to_have_value('D:\\')
            self.page.locator('#folder-picker-select').click()
            expect(self.page.locator('#model-path')).to_have_value(model_file)
            expect(self.page.locator('#folder-picker-modal')).to_be_hidden()

    def test_draft_picker_selects_folders_and_mtp_files_without_changing_main_model(self):
        folder = os.path.join(self.temp.name, 'Draft 模型目录')
        os.mkdir(folder)
        checkpoint = os.path.join(folder, 'mtp.safetensors')
        with open(checkpoint, 'w') as output:
            output.write('test checkpoint')
        self.page.locator('#new-profile').click()
        self.page.locator('[data-config-mode][value="custom"]').check()
        model = self.page.locator('#model-path')
        draft = self.page.locator('#draft-model-path')
        model.fill(self.temp.name)
        self.page.locator('[data-editor-section="speculative"]').click()
        self.page.locator('#choose-draft-model-folder').click()
        expect(self.page.locator('#folder-picker-title')).to_have_text('Choose a draft model file or folder')
        self.page.locator('#folder-picker-list button').filter(has_text='Draft 模型目录').click()
        expect(self.page.locator('#folder-picker-list button').filter(has_text='mtp.safetensors')).to_be_visible()
        self.screenshot('launcher-draft-folder-picker')
        self.page.locator('#folder-picker-select').click()
        expect(draft).to_have_value(folder)
        expect(model).to_have_value(self.temp.name)
        self.page.locator('#choose-draft-model-folder').click()
        self.page.keyboard.press('Escape')
        expect(self.page.locator('#choose-draft-model-folder')).to_be_focused()
        expect(draft).to_have_value(folder)
        self.page.locator('#choose-draft-model-folder').click()
        self.page.locator('#folder-picker-list button').filter(has_text='mtp.safetensors').click()
        self.page.locator('#folder-picker-select').click()
        expect(draft).to_have_value(checkpoint)
        expect(model).to_have_value(self.temp.name)
        self.page.locator('#choose-model-folder').click()
        expect(self.page.locator('#folder-picker-title')).to_have_text('Choose a model file or folder')
        self.page.locator('#folder-picker-list button').filter(has_text='Draft 模型目录').click()
        self.page.locator('#folder-picker-select').click()
        expect(model).to_have_value(folder)
        expect(draft).to_have_value(checkpoint)
        self.page.locator('#save-profile').click()
        expect(self.page.locator('#profile-editor-modal')).to_be_hidden()
        self.page.reload()
        self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
        expect(draft).to_have_value(checkpoint)
        expect(model).to_have_value(folder)

        self.page.set_viewport_size({'width': 390, 'height': 844})
        draft.scroll_into_view_if_needed()
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))

    def test_explicit_speculative_off_clears_saved_mtp_and_survives_reopening(self):
        for legacy_algorithm in ('auto', 'mtp'):
            with self.subTest(legacy_algorithm=legacy_algorithm):
                saved = self.runtime.save_profile(None, {
                    'name': 'MTP profile', 'model': self.temp.name, 'device': 'cpu',
                    'mtp': '3', 'draft_tokens': '3', 'speculative_algorithm': legacy_algorithm,
                    'speculative_draft_model_path': self.temp.name, 'enable_speculative_decoding': True})
                index = saved['index']
                self.page.reload()
                self.page.locator(f'[data-profile-action="edit"][data-profile-index="{index}"]').click()
                self.page.locator('[data-editor-section="speculative"]').click()
                algorithm = self.page.locator('[data-field="speculative_algorithm"]')
                expect(algorithm).to_have_value(legacy_algorithm)
                algorithm.select_option('off')
                expect(self.page.locator('[data-field="mtp"]')).to_have_value('0')
                expect(self.page.locator('[data-field="mtp"]')).to_be_disabled()
                expect(self.page.locator('#draft-model-path')).to_have_value('')
                expect(self.page.locator('#choose-draft-model-folder')).to_be_disabled()
                expect(self.page.locator('#enable-speculative-decoding')).not_to_be_checked()
                self.page.clock.fast_forward(700)
                expect(self.page.locator('#command-preview')).to_contain_text('--speculative_algorithm off')
                expect(self.page.locator('#command-preview')).not_to_contain_text('--mtp')
                expect(self.page.locator('#command-preview')).not_to_contain_text('--draft_tokens')
                self.screenshot('launcher-speculative-off')
                self.page.locator('#save-profile').click()
                expect(self.page.locator('#profile-editor-modal')).to_be_hidden()
                profile = self.runtime.profiles()[index]
                self.assertEqual(profile['speculative_algorithm'], 'off')
                self.assertEqual(profile['mtp'], '0')
                self.assertEqual(profile['speculative_draft_model_path'], '')
                self.assertFalse(profile['enable_speculative_decoding'])
                self.page.reload()
                self.page.locator(f'[data-profile-action="edit"][data-profile-index="{index}"]').click()
                expect(algorithm).to_have_value('off')
                self.page.locator('[data-editor-section="speculative"]').click()
                algorithm.select_option('mtp')
                expect(self.page.locator('[data-field="mtp"]')).to_be_enabled()
                expect(self.page.locator('#choose-draft-model-folder')).to_be_enabled()
                self.page.locator('[data-field="mtp"]').fill('3')
                self.page.clock.fast_forward(700)
                expect(self.page.locator('#command-preview')).to_contain_text('--speculative_algorithm mtp')
                expect(self.page.locator('#command-preview')).to_contain_text('--mtp 3')

    def test_speculative_switch_detects_mtp_and_restores_the_saved_preference(self):
        from test_launcher_mtp import cuda_hardware, write_mtp_checkpoint, write_safetensors
        from pathlib import Path

        model_path = Path(self.temp.name) / 'mtp-model'
        write_mtp_checkpoint(model_path)
        toggle = self.page.locator('#enable-speculative-decoding')
        status = self.page.locator('#automatic-config-status')
        with patch('fastllm_pytools.launcher.detect_hardware', return_value=cuda_hardware()):
            self.page.locator('#new-profile').click()
            expect(toggle).not_to_be_checked()
            self.page.locator('#model-path').fill(str(model_path))
            self.page.clock.fast_forward(700)
            expect(status).to_have_class('automatic-config-status success')
            expect(self.page.locator('[data-field="mtp"]')).to_have_value('auto')
            toggle.check()
            self.page.clock.fast_forward(1)
            expect(status).to_contain_text('Speculative decoding is enabled (3 draft tokens)')
            expect(self.page.locator('[data-field="mtp"]')).to_have_value('3')
            toggle.uncheck()
            self.page.clock.fast_forward(1)
            expect(status).to_have_class('automatic-config-status success')
            expect(self.page.locator('[data-field="mtp"]')).to_have_value('auto')
            toggle.check()
            self.page.clock.fast_forward(1)
            expect(status).to_contain_text('Speculative decoding is enabled')
            self.screenshot('launcher-mtp-enabled')
            self.page.locator('#save-profile').click()
            expect(self.page.locator('#profile-editor-modal')).to_be_hidden()
            saved = self.runtime.profiles()[0]
            self.assertTrue(saved['enable_speculative_decoding'])
            self.assertEqual(saved['mtp'], '3')
            self.page.reload()
            self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
            expect(toggle).to_be_checked()
            expect(toggle).to_be_hidden()
            write_safetensors(model_path / 'model.safetensors', {'model.embed_tokens.weight'})
            self.page.locator('#auto-configure-profile').click()
            expect(self.page.locator('#automatic-enable-speculative-decoding')).to_be_checked()
            with self.page.expect_request('**/api/recommend') as request:
                self.page.locator('#automatic-config-dialog-apply').click()
            self.assertTrue(request.value.post_data_json['enable_speculative_decoding'])
            expect(status).to_contain_text('MTP weights are missing or incomplete')
            expect(self.page.locator('[data-field="mtp"]')).to_have_value('auto')
            expect(toggle).to_be_checked()

    def test_speculative_algorithms_show_one_count_and_clear_incompatible_values(self):
        self.runtime.save_profile(None, {'name': 'Algorithm changes', 'model': self.temp.name,
                                        'device': 'cpu', 'mtp': '3', 'speculative_algorithm': 'mtp',
                                        'speculative_draft_model_path': self.temp.name})
        self.page.reload()
        self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
        self.page.locator('[data-editor-section="speculative"]').click()
        algorithm = self.page.locator('[data-field="speculative_algorithm"]')
        mtp = self.page.locator('[data-field="mtp"]')
        draft = self.page.locator('[data-field="draft_tokens"]')
        for name in ('dflash', 'dspark'):
            algorithm.select_option(name)
            expect(mtp).to_be_hidden()
            expect(mtp).to_have_value('auto')
            expect(draft).to_be_visible()
            expect(draft).to_have_value('3')
            self.page.clock.fast_forward(700)
            expect(self.page.locator('#validation-summary')).to_be_hidden()
            expect(self.page.locator('#command-preview')).not_to_contain_text('--mtp')
        self.screenshot('launcher-algorithm-dependent-fields')
        algorithm.select_option('mtp')
        expect(draft).to_be_hidden()
        expect(draft).to_have_value('auto')
        mtp.fill('7')
        self.page.clock.fast_forward(700)
        expect(self.page.locator('#command-preview')).to_contain_text('--mtp 7')
        expect(self.page.locator('#command-preview')).not_to_contain_text('--draft_tokens')
        algorithm.select_option('auto')
        mtp.fill('auto')
        expect(mtp).to_be_visible()
        mtp.fill('3')
        algorithm.select_option('off')
        expect(mtp).to_be_hidden()
        expect(draft).to_be_hidden()
        expect(self.page.locator('#speculative-mode-hint')).to_contain_text('is off')

    def test_field_errors_can_be_located_and_clear_after_correction(self):
        self.runtime.save_profile(None, {'name': 'Validation', 'model': self.temp.name,
                                        'device': 'cpu', 'max_batch': '0'})
        self.page.reload()
        self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
        batch = self.page.locator('[data-field="max_batch"]')
        summary = self.page.locator('#validation-summary')
        expect(batch).to_have_attribute('aria-invalid', 'true')
        expect(self.page.locator('#launch-error-max_batch')).to_contain_text('positive integer')
        expect(summary).to_have_text('Errors: 1 · locate')
        self.page.locator('#editor-advanced').evaluate('(node) => node.open = false')
        summary.click()
        expect(batch).to_be_visible()
        expect(batch).to_be_focused()
        self.screenshot('launcher-inline-error-location')
        self.page.locator('#language-select').select_option('zh-CN')
        expect(summary).to_have_text('1 项错误 · 点击定位')
        expect(self.page.locator('#launch-error-max_batch')).to_contain_text('正整数')
        self.page.locator('#theme-select').select_option('dark')
        expect(batch).to_have_css('border-color', 'rgb(244, 135, 113)')
        for width in (390, 320):
            self.page.set_viewport_size({'width': width, 'height': 844})
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            summary.click()
            expect(batch).to_be_focused()
            expect(batch).to_be_in_viewport()
            expect(self.page.locator('#save-profile')).to_be_in_viewport()
            expect(self.page.locator('#start-runtime')).to_be_in_viewport()
        self.screenshot('launcher-inline-error-dark-mobile')
        batch.fill('1')
        self.page.clock.fast_forward(700)
        expect(summary).to_be_hidden()
        expect(batch).not_to_have_attribute('aria-invalid', 'true')
        expect(self.page.locator('#launch-error-max_batch')).to_have_count(0)

    def test_saved_changes_show_restart_requirement_and_preserve_running_service(self):
        from fastllm_pytools.tui import build_fastllm_argv, build_fastllm_env, config_from_dict
        config = {'name': 'Running profile', 'model': self.temp.name, 'device': 'cpu', 'gpu_mem_ratio': '0.9'}
        saved = self.runtime.save_profile(None, config)['profile']
        deploy = config_from_dict(saved)
        self.runtime._launch_signature = (tuple(build_fastllm_argv(deploy)), tuple(sorted(build_fastllm_env(deploy).items())))
        self.runtime._state.update(profileName=saved['name'], model=self.temp.name)
        self.page.reload()
        self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
        expect(self.page.locator('#save-state')).to_have_text('Saved')

        expect(self.page.locator('#launch-action-hint')).to_contain_text('next model start')
        self.page.locator('[data-editor-section="advanced"]').click()
        self.page.locator('[data-field="gpu_mem_ratio"]').fill('0.8')
        self.page.locator('#save-profile').click()
        expect(self.page.locator('#profile-editor-modal')).to_be_hidden()
        expect(self.page.locator('#toast-region')).to_contain_text('running service is unchanged')
        self.page.reload()
        self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
        expect(self.page.locator('#save-state')).to_have_text('Saved · restart required')
        self.screenshot('launcher-saved-restart-required')
        self.assertEqual(self.runtime.state()['sessionId'], 'model-a')
        self.assertEqual(self.runtime.state()['phase'], 'running')
        self.page.locator('[data-editor-section="advanced"]').click()
        self.page.locator('[data-field="gpu_mem_ratio"]').fill('0.9')
        self.page.locator('#save-profile').click()
        expect(self.page.locator('#profile-editor-modal')).to_be_hidden()
        self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
        expect(self.page.locator('#save-state')).to_have_text('Saved')

        self.page.locator('#close-profile-editor').click()
        self.runtime.save_profile(None, {**config, 'name': 'Other profile', 'gpu_mem_ratio': '0.7'})
        self.page.reload()
        self.page.locator('[data-profile-action="edit"][data-profile-index="1"]').click()
        expect(self.page.locator('#save-state')).to_have_text('Saved')

    def test_edit_automatic_configuration_defers_changes_until_apply(self):
        self.runtime.save_profile(None, {'name': 'Mode picker', 'model': self.temp.name,
                                        'config_mode': 'long_context', 'max_batch': '7', 'device': 'cpu'})
        self.page.reload()
        self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
        requests = []
        self.page.on('request', lambda request: requests.append(request)
                     if request.url.endswith('/api/recommend') else None)
        batch = self.page.locator('[data-field="max_batch"]')
        chooser = self.page.locator('#automatic-config-dialog')
        self.page.locator('#auto-configure-profile').click()
        expect(chooser).to_be_visible()
        high = chooser.locator('input[value="high_concurrency"]')
        high.check()
        self.page.locator('#automatic-enable-speculative-decoding').check()
        expect(batch).to_have_value('7')
        self.assertEqual(requests, [])
        self.page.keyboard.press('Escape')
        expect(chooser).to_be_hidden()
        expect(self.page.locator('#profile-editor-modal')).to_be_visible()
        expect(batch).to_have_value('7')
        expect(self.page.locator('#auto-configure-profile')).to_be_focused()
        self.page.locator('#auto-configure-profile').click()
        expect(chooser.locator('input[value="long_context"]')).to_be_checked()
        expect(self.page.locator('#automatic-enable-speculative-decoding')).not_to_be_checked()
        high.check()
        self.page.locator('#automatic-enable-speculative-decoding').check()
        self.screenshot('edit-automatic-configuration-picker')
        with patch('fastllm_pytools.launcher.detect_hardware', return_value={'gpus': []}):
            self.page.locator('#automatic-config-dialog-apply').click()
            expect(chooser).to_be_hidden()
        expect(batch).to_have_value('auto')
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].post_data_json['config_mode'], 'high_concurrency')
        self.assertTrue(requests[0].post_data_json['enable_speculative_decoding'])
        self.page.locator('#save-profile').click()
        expect(self.page.locator('#profile-editor-modal')).to_be_hidden()
        saved = self.runtime.profiles()[0]
        self.assertEqual(saved['config_mode'], 'high_concurrency')
        self.assertTrue(saved['enable_speculative_decoding'])
        self.page.reload()
        self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
        self.page.locator('#auto-configure-profile').click()
        expect(chooser.locator('input[value="high_concurrency"]')).to_be_checked()
        expect(self.page.locator('#automatic-enable-speculative-decoding')).to_be_checked()

    def test_edit_mode_picker_handles_failure_and_cancels_pending_changes(self):
        self.runtime.save_profile(None, {'name': 'Cancel recommendation', 'model': self.temp.name,
                                        'config_mode': 'long_context', 'max_batch': '7', 'device': 'cpu'})
        self.page.reload()
        self.page.locator('[data-profile-action="edit"][data-profile-index="0"]').click()
        self.page.locator('#auto-configure-profile').click()
        chooser = self.page.locator('#automatic-config-dialog')
        chooser.locator('input[value="high_concurrency"]').check()
        self.page.route('**/api/recommend', lambda route: route.fulfill(
            status=500, content_type='application/json', body='{"error":"Test recommendation error"}'), times=1)
        self.page.locator('#automatic-config-dialog-apply').click()
        expect(self.page.locator('#automatic-config-dialog-error')).to_have_text('Test recommendation error')
        expect(chooser).to_be_visible()
        expect(self.page.locator('[data-field="max_batch"]')).to_have_value('7')
        pending = []
        self.page.route('**/api/recommend', lambda route: pending.append(route), times=1)
        with self.page.expect_request('**/api/recommend'):
            self.page.locator('#automatic-config-dialog-apply').click()
        expect(chooser.locator('input[value="high_concurrency"]')).to_be_disabled()
        self.page.locator('#automatic-config-dialog-cancel').click()
        expect(chooser).to_be_hidden()
        with self.page.expect_response('**/api/recommend') as response:
            pending[0].fulfill(status=200, content_type='application/json',
                               body=json.dumps({'config': {'max_batch': '99'}}))
        response.value.body()
        expect(self.page.locator('[data-field="max_batch"]')).to_have_value('7')
        self.page.locator('#close-profile-editor').click()
        expect(self.page.locator('#confirmation-modal')).to_be_hidden()
        expect(self.page.locator('#profile-editor-modal')).to_be_hidden()
        self.assertEqual(self.runtime.profiles()[0]['config_mode'], 'long_context')

    def test_disabling_speculative_switch_ignores_the_inflight_enabled_result(self):
        from test_launcher_mtp import cuda_hardware, write_mtp_checkpoint
        from pathlib import Path

        model_path = Path(self.temp.name) / 'mtp-model'
        write_mtp_checkpoint(model_path)
        pending = []
        status = self.page.locator('#automatic-config-status')
        toggle = self.page.locator('#enable-speculative-decoding')
        with patch('fastllm_pytools.launcher.detect_hardware', return_value=cuda_hardware()):
            self.page.locator('#new-profile').click()
            toggle.check()
            self.page.route('**/api/recommend', lambda route: pending.append(route), times=1)
            self.page.locator('#model-path').fill(str(model_path))
            with self.page.expect_request('**/api/recommend'):
                self.page.clock.fast_forward(700)
            toggle.uncheck()
            self.page.clock.fast_forward(1)
            expect(status).to_have_class('automatic-config-status success')
            with self.page.expect_response('**/api/recommend') as response:
                pending[0].fulfill(status=200, content_type='application/json', body=json.dumps({
                    'config': {'mtp': '3', 'speculative_algorithm': 'mtp'},
                    'speculative': {'requested': True, 'enabled': True, 'reason': 'enabled'}}))
            response.value.body()
            self.page.locator('#save-profile').click()
            expect(self.page.locator('#profile-editor-modal')).to_be_hidden()
            saved = self.runtime.profiles()[0]
            self.assertFalse(saved['enable_speculative_decoding'])
            self.assertEqual(saved['mtp'], 'auto')

    def test_switching_to_custom_ignores_a_pending_recommendation(self):
        pending = []
        self.page.route('**/api/recommend', lambda route: pending.append(route), times=1)
        self.page.locator('#new-profile').click()
        self.page.locator('#model-path').fill(self.temp.name)
        with self.page.expect_request('**/api/recommend'):
            self.page.clock.fast_forward(700)
        expect(self.page.locator('#save-profile')).to_be_disabled()
        self.page.locator('[data-config-mode][value="custom"]').check()
        batch = self.page.locator('[data-field="max_batch"]')
        batch.fill('7')
        with self.page.expect_response('**/api/recommend') as response:
            pending[0].fulfill(status=200, content_type='application/json',
                               body=json.dumps({'config': {'max_batch': '99'}}))
        response.value.body()
        self.page.locator('#save-profile').click()
        expect(self.page.locator('#profile-editor-modal')).to_be_hidden()
        saved = self.runtime.profiles()[0]
        self.assertEqual(saved['config_mode'], 'custom')
        self.assertEqual(saved['max_batch'], '7')

    def test_network_error_can_retry_without_reloading_launcher(self):
        self.assert_resource_failure_recovers('**/assets/webui/template.html')

    def test_component_module_failure_can_retry(self):
        self.assert_resource_failure_recovers('**/assets/webui/app.js')

    def test_markdown_module_failure_can_retry(self):
        self.assert_resource_failure_recovers('**/assets/webui/markdown.js')

    def test_markdown_parser_failure_can_retry(self):
        self.assert_resource_failure_recovers('**/assets/webui/marked.js')

    def test_component_stylesheet_failure_can_retry(self):
        self.assert_resource_failure_recovers('**/assets/webui/styles.css')

    def test_locales_failure_can_retry(self):
        self.assert_resource_failure_recovers('**/assets/webui_locales.js')

    def assert_pending_asset_can_retry(self, pattern):
        pending = []
        self.page.route(pattern, lambda route: pending.append(route), times=1)
        with self.page.expect_request(pattern):
            self.page.locator('#open-webui').click()
        self.page.clock.fast_forward(30001)
        expect(self.page.locator('#webui-retry')).to_be_visible()
        self.assertEqual(len(pending), 1)
        # Retry must finish while the first script request is still hung.
        self.page.locator('#webui-retry').click()
        self.assert_loaded()
        pending[0].abort('connectionfailed')
        self.assert_loaded()

    def test_stalled_module_can_retry_before_old_request_finishes(self):
        self.assert_pending_asset_can_retry('**/assets/webui/app.js')

    def test_stalled_locales_can_retry_before_old_request_finishes(self):
        self.assert_pending_asset_can_retry('**/assets/webui_locales.js')

    def assert_resource_failure_recovers(self, pattern):
        self.page.route(pattern, lambda route: route.abort('connectionfailed'))
        self.page.locator('#open-webui').click()
        expect(self.page.locator('#webui-retry')).to_be_visible()
        expect(self.page.locator('#webui-status')).to_have_text('Unable to load Studio. Try reopening it.')
        self.page.unroute(pattern)
        self.page.locator('[data-view-button="launch"]').click()
        self.page.locator('#open-webui').click()
        self.page.locator('#webui-retry').click()
        self.assert_loaded()

    def assert_timeout_recovers(self, pattern, response):
        pending = []
        self.page.route(pattern, lambda route: pending.append(route), times=1)
        with self.page.expect_request(pattern):
            self.page.locator('#open-webui').click()
        self.page.clock.fast_forward(30001)
        expect(self.page.locator('#webui-retry')).to_be_visible()
        expect(self.page.locator('#webui-status')).to_have_text('Studio loading timed out. Try reopening it.')
        self.assertEqual(len(pending), 1)
        # Completing the stale request must not undo timeout cleanup or poison
        # the next attempt (including the aborted /api/webui/open fetch).
        pending[0].fulfill(status=200, **response)
        self.page.unroute(pattern)
        self.page.locator('#webui-retry').click()
        self.assert_loaded()

    def test_open_api_timeout_can_retry(self):
        self.assert_timeout_recovers('**/api/webui/open', {
            'content_type': 'application/json', 'body': json.dumps({'url': '/webui/stale/'})})

    def test_component_timeout_can_retry(self):
        self.assert_timeout_recovers('**/assets/webui/template.html', {
            'content_type': 'text/html', 'body': '<p>stale document</p>'})

    def test_model_switch_clears_the_old_loading_attempt(self):
        pending = []
        pattern = '**/assets/webui/template.html'
        self.page.route(pattern, lambda route: pending.append(route), times=1)
        with self.page.expect_request(pattern):
            self.page.locator('#open-webui').click()
        with self.runtime._lock:
            self.runtime._close_webui_locked()
            self.runtime._state.update(sessionId='model-b', modelName='next-model')
        self.page.clock.fast_forward(2500)
        expect(self.page.locator('#webui-content #modelName')).to_have_text('next-model')
        self.assert_loaded()
        self.assertEqual(len(pending), 1)
        pending[0].fulfill(status=200, content_type='text/html', body='<p>old model</p>')
        expect(self.page.locator('#webui-content').locator('#modelName')).to_have_text('next-model')

    def screenshot(self, name):
        directory = os.environ.get('FTLLM_WEBUI_SCREENSHOTS')
        if directory:
            os.makedirs(directory, exist_ok=True)
            self.page.screenshot(path=os.path.join(directory, name + '.png'), animations='disabled')

    def test_chat_is_part_of_launcher_and_preserves_shared_features(self):
        expect(self.page.locator('#open-webui')).to_have_text('Open Studio')
        expect(self.page.locator('[data-view-button="webui"]')).to_have_text('Studio')
        self.page.locator('#open-webui').click()
        self.assert_loaded()
        expect(self.page.locator('#current-view-title')).to_have_text('Studio')
        pane = self.page.locator('#webui-content')
        expect(self.page.locator('.app-shell > .sidebar')).to_be_visible()
        self.assertEqual(self.page.locator('iframe').count(), 0)
        self.assertEqual(self.page.locator('[data-view-button]').evaluate_all(
            '(nodes) => nodes.map(node => node.dataset.viewButton)'),
            ['launch', 'download', 'logs', 'hardware', 'webui'])
        self.screenshot('launcher-empty')
        with patch.object(self.runtime._webui_app.state.runtime.api_client, 'stream',
                          side_effect=lambda *a, **k: iter([('**Hello**\n\n```python\nprint(1)\n```', 'Reasoning')])) as stream:
            pane.locator('#prompt').fill('First turn')
            pane.locator('#sendButton').click()
            expect(pane.locator('.message.assistant')).to_have_count(1)
            expect(pane.locator('#stopButton')).to_be_hidden()
            expect(pane.locator('.code-block')).to_be_visible()
            self.page.locator('[data-view-button="logs"]').click()
            expect(self.page.locator('#view-logs')).to_be_visible()
            self.page.locator('[data-view-button="webui"]').click()
            expect(pane.locator('.message.assistant')).to_have_count(1)
            pane.locator('#fileInput').set_input_files({
                'name': 'notes.txt', 'mimeType': 'text/plain', 'buffer': b'Shared component attachment'})
            expect(pane.locator('.pending-file span')).to_have_text('notes.txt')
            pane.locator('#prompt').fill('Read this file')
            pane.locator('#sendButton').click()
            expect(pane.locator('.message.assistant')).to_have_count(2)
            expect(pane.locator('#stopButton')).to_be_hidden()
            self.assertEqual([m['role'] for m in stream.call_args.args[0] if m['role'] != 'system'],
                             ['user', 'assistant', 'user'])
            pane.locator('#fileInput').set_input_files({
                'name': 'pixel.png', 'mimeType': 'image/png', 'buffer': base64.b64decode(
                    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aM8sAAAAASUVORK5CYII=')})
            expect(pane.locator('.pending-file span')).to_have_text('pixel.png')
            pane.locator('#prompt').fill('Describe the image')
            pane.locator('#sendButton').click()
            expect(pane.locator('.message.assistant')).to_have_count(3)
            expect(pane.locator('#stopButton')).to_be_hidden()
            picture = pane.locator('.attachment-card img')
            expect(picture).to_have_js_property('complete', True)
            self.assertEqual(picture.evaluate('(image) => image.naturalWidth'), 1)
        link = pane.locator('.attachment-card.document')
        response = self.context.request.get(link.get_attribute('href'))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.body(), b'Shared component attachment')
        pane.locator('#agentButton').click()
        expect(pane.locator('#agentDialog')).to_be_visible()
        expect(pane.locator('.agent-card')).to_have_count(4)
        pane.locator('#closeAgent').click()
        pane.locator('#topSettings').click()
        expect(pane.locator('#settingsDialog')).to_be_visible()
        pane.locator('#settingTokens').fill('128')
        pane.locator('#saveSettings').click()
        expect(pane.locator('#settingsDialog')).not_to_be_visible()
        self.page.locator('#language-select').select_option('zh-CN')
        expect(pane.locator('#newChat')).to_have_text('新建对话')
        self.assertEqual(self.page.url, self.url + '/')
        self.assertEqual(self.page.title(), 'FastLLM Launcher')
        self.screenshot('launcher-chat')
        self.page.reload()
        self.page.locator('#open-webui').click()
        expect(pane.locator('.message.assistant')).to_have_count(3)

    def test_history_menu_and_layout_fit_the_launcher_content_width(self):
        self.page.locator('#open-webui').click()
        self.assert_loaded()
        pane = self.page.locator('#webui-content')
        for width in (1440, 1024, 768, 390, 320):
            with self.subTest(width=width):
                self.page.set_viewport_size({'width': width, 'height': 900})
                self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth + 2'))
                self.assertTrue(pane.locator('.main').evaluate('(node) => node.scrollWidth <= node.clientWidth + 2'))
                expect(self.page.locator('[data-view-button="launch"]')).to_be_visible()
                expect(pane.locator('#sendButton')).to_be_visible()
                send = pane.locator('#sendButton').bounding_box()
                bounds = pane.bounding_box()
                self.assertLessEqual(send['x'] + send['width'], bounds['x'] + bounds['width'])
        self.page.set_viewport_size({'width': 390, 'height': 844})
        pane.locator('#mobileMenu').click()
        expect(pane.locator('#sidebar')).to_have_class('sidebar open')
        pane.locator('.conversation-more').first.click()
        menu = pane.locator('#conversationActionMenu').bounding_box()
        bounds = pane.bounding_box()
        self.assertGreaterEqual(menu['x'], bounds['x'])
        self.assertLessEqual(menu['x'] + menu['width'], bounds['x'] + bounds['width'])
        self.assertGreaterEqual(menu['y'], bounds['y'])
        self.assertLessEqual(menu['y'] + menu['height'], bounds['y'] + bounds['height'])
        pane.locator('#renameConversationAction').click()
        pane.locator('#renameTitle').fill('Renamed inside Launcher')
        pane.locator('#saveRename').click()
        expect(pane.locator('.conversation-title').first).to_have_text('Renamed inside Launcher')
        self.screenshot('launcher-mobile-history')
        pane.locator('#sidebarBackdrop').click(position={'x': 380, 'y': 300})
        self.screenshot('launcher-mobile-chat')

    def test_stop_generation_and_switch_model_dispose_the_component(self):
        self.page.locator('#open-webui').click()
        self.assert_loaded()
        pane = self.page.locator('#webui-content')
        controls = []

        def slow_stream(*args, control, **kwargs):
            controls.append(control)
            yield ('Partial response', '')
            control.event.wait(10)
            raise GenerationCancelled()

        with patch.object(self.runtime._webui_app.state.runtime.api_client, 'stream', slow_stream):
            pane.locator('#prompt').fill('Stop this response')
            pane.locator('#sendButton').click()
            expect(pane.locator('#stopButton')).to_be_visible()
            pane.locator('#stopButton').click()
            expect(pane.locator('#stopButton')).to_be_hidden()
            self.assertTrue(controls[0].cancelled)
            pane.locator('#prompt').fill('Switch while generating')
            pane.locator('#sendButton').click()
            expect(pane.locator('#stopButton')).to_be_visible()
            with self.runtime._lock:
                self.runtime._close_webui_locked()
                self.runtime._state.update(sessionId='model-b', modelName='next-model')
            self.page.clock.fast_forward(2500)
            expect(pane.locator('#modelName')).to_have_text('next-model')
            expect(pane.locator('#stopButton')).to_be_hidden()
            self.assertTrue(all(control.cancelled for control in controls))

    def test_workspace_agent_can_select_a_project_and_chat(self):
        workspace = os.path.join(self.temp.name, 'projects')
        project = os.path.join(workspace, 'demo')
        os.makedirs(project)
        self.runtime._agent_workspace_root = workspace
        calls = []

        def stream(**kwargs):
            calls.append(kwargs)
            yield {'type': 'tool_start', 'id': 'read-project', 'name': 'read',
                   'arguments': {'path': 'README.md'}}
            yield {'type': 'tool_end', 'id': 'read-project', 'name': 'read',
                   'result': 'Demo project', 'is_error': False}
            yield {'type': 'text_delta', 'text': 'Project inspected.\n\n| File | State |\n| --- | --- |\n| README.md | **Read** |\n\n```html\n<h1>Agent preview</h1>\n```'}
            yield {'type': 'done', 'turns': 1}

        fake_pi = lambda **kwargs: SimpleNamespace(
            info=lambda: {'available': True}, stream=stream)
        with patch.dict(sys.modules, {'ftllm_agent_runtime': SimpleNamespace(PiAgentRuntime=fake_pi)}):
            self.page.locator('#open-webui').click()
            self.assert_loaded()
            pane = self.page.locator('#webui-content')
            expect(pane.locator('#newAgent')).to_be_enabled()
            expect(pane.locator('#agentUnavailable')).to_be_hidden()
            pane.locator('#newAgent').click()
            expect(pane.locator('#workspaceDialog')).to_be_visible()
            pane.locator('.workspace-directory').filter(has_text='demo').click()
            expect(pane.locator('#workspacePath')).to_have_value(project)
            pane.locator('#createWorkspace').click()
            expect(pane.locator('#workspaceContextPath')).to_have_text(project)
            pane.locator('#prompt').fill('Inspect this project')
            pane.locator('#sendButton').click()
            expect(pane.locator('.message.assistant')).to_contain_text('Project inspected.')
            expect(pane.locator('#stopButton')).to_be_hidden()
            self.assertEqual(str(calls[0]['working_directory']), project)
            expect(pane.locator('.message.assistant table td strong')).to_have_text('Read')
            pane.locator('.preview-code').click()
            expect(pane.frame_locator('#htmlPreviewBody iframe').locator('h1')).to_have_text('Agent preview')
            pane.locator('#closeHTMLPreview').click()
            self.screenshot('launcher-pi-agent')

    def test_workspace_unavailable_reason_explains_runtime_and_remote_policy(self):
        def config(route):
            response = route.fetch()
            data = response.json()
            data.update(workspace_agent_enabled=False, pi_agent={'available': False, 'error': 'Missing runtime'})
            route.fulfill(response=response, json=data)

        self.page.route('**/webui/*/api/config', config)
        self.page.locator('#open-webui').click()
        self.assert_loaded()
        pane = self.page.locator('#webui-content')
        expect(pane.locator('#newAgent')).to_be_disabled()
        expect(pane.locator('#agentUnavailable')).to_contain_text('Install it from Launcher Studio')
        expect(pane.locator('#installRuntime')).to_have_text('Install Agent dependencies')
        expect(pane.locator('#agentUnavailable')).to_contain_text('--allow-remote-workspace-agent')
        self.page.locator('#language-select').select_option('zh-CN')
        expect(pane.locator('#agentUnavailable')).to_contain_text('目录 Agent 未对远程访问开放')
        expect(self.page.locator('[data-view-button="webui"]')).to_have_text('工作室')
        expect(self.page.locator('#current-view-title')).to_have_text('工作室')

    def test_workspace_disabled_option_has_its_own_explanation(self):
        self.runtime._disable_workspace_agent = True
        self.page.locator('#open-webui').click()
        self.assert_loaded()
        pane = self.page.locator('#webui-content')
        expect(pane.locator('#newAgent')).to_be_disabled()
        expect(pane.locator('#agentUnavailable')).to_contain_text('Remove --disable-workspace-agent')
        expect(pane.locator('#newChat')).to_be_enabled()
        self.page.locator('#language-select').select_option('zh-CN')
        expect(pane.locator('#agentUnavailable')).to_contain_text('目录 Agent 已关闭')

    def test_standalone_webui_uses_the_same_component(self):
        self.page.goto(self.url + '/standalone/')
        pane = self.page.locator('#webui-root')
        expect(pane.locator('#prompt')).to_be_visible()
        expect(pane.locator('.brand')).to_be_visible()
        expect(pane.locator('#languageButton')).to_be_visible()
        with patch.object(self.standalone.api_client, 'stream',
                          side_effect=lambda *a, **k: iter([('Shared standalone reply\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n```html\n<h1>Standalone preview</h1>\n```', '')])):
            pane.locator('#prompt').fill('Hello standalone')
            pane.locator('#sendButton').click()
            expect(pane.locator('.message.assistant')).to_have_count(1)
            expect(pane.locator('#stopButton')).to_be_hidden()
        expect(pane.locator('.message.assistant table')).to_be_visible()
        pane.locator('.preview-code').click()
        expect(pane.frame_locator('#htmlPreviewBody iframe').locator('h1')).to_have_text('Standalone preview')
        pane.locator('#closeHTMLPreview').click()
        self.assertIn('/standalone/?chat=', self.page.url)
        self.screenshot('standalone-webui')
        self.page.reload()
        expect(pane.locator('.message.assistant')).to_have_count(1)

    def test_standalone_customizer_previews_whole_studio_and_preserves_draft(self):
        self.page.goto(self.url + '/standalone/')
        pane = self.page.locator('#webui-root')
        expect(pane.locator('.plugin-manager')).to_have_count(1)
        pane.locator('#prompt').fill('Standalone customization draft')
        original_system = pane.locator('#settingSystem').input_value()
        expect(pane.locator('.plugin-manager-button')).to_have_count(0)
        self.page.evaluate("location.hash = 'customize'")
        editor = pane.locator('.plugin-manager')
        expect(editor).to_be_visible()
        snapshot = editor.locator('.customizer-screen')
        expect(snapshot.locator('#prompt')).to_have_value('Standalone customization draft')
        expect(snapshot.locator('#sendButton')).to_be_disabled()
        snapshot.locator('#prompt').fill('Only in preview')
        snapshot.locator('#prompt').press('Enter')
        snapshot.locator('#topSettings').click()
        expect(snapshot.locator('#settingsDialog')).to_be_visible()
        self.assertFalse(snapshot.locator('#settingsDialog').evaluate('node => node.matches(":modal")'))
        snapshot.locator('#settingSystem').fill('Preview settings')
        # A preview dialog does not block the surrounding customization editor.
        editor.locator('[name=instruction]').fill('Add a status widget')
        snapshot.locator('#settingSystem').press('Escape')
        expect(snapshot.locator('#settingsDialog')).to_be_hidden()
        expect(editor).to_be_visible()
        snapshot.locator('#topSettings').click()
        snapshot.locator('#saveSettings').click()
        expect(snapshot.locator('#settingsDialog')).to_be_hidden()
        editor.locator('.plugin-heading [aria-label=关闭]').click()
        expect(editor).to_be_hidden()
        expect(pane.locator('#prompt')).to_have_value('Standalone customization draft')
        expect(pane.locator('.app').locator('#messages .message.user')).to_have_count(0)
        expect(pane.locator('#settingSystem')).to_have_value(original_system)


if __name__ == '__main__':
    unittest.main()
