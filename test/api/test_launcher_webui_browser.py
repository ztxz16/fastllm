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

    def test_native_agents_install_only_after_click_and_use_plugin_management(self):
        self.page.clock.resume()
        for agent, name in (("opencode", "OpenCode"), ("codex", "Codex")):
            state = {"phase":"stopped", "installed":False, "sessionId":"", "url":"", "error":""}
            def start(*args, **kwargs):
                self.assertTrue(kwargs['install'])
                state.update(phase='installing', sessionId='model-a', stage='download', done=1048576, total=4194304)
                return dict(state)
            def stop():
                state.update(phase='stopped', sessionId='', error='')
                return dict(state)
            runtime = getattr(self.runtime, agent)
            with patch.object(runtime, 'state', side_effect=lambda:dict(state)), patch.object(
                    runtime, 'start', side_effect=start) as opening, patch.object(runtime, 'stop', side_effect=stop):
                self.page.locator(f'[data-view-button="{agent}"]').click()
                expect(self.page.locator(f'#{agent}-retry')).to_have_text(f'Install and open {name}')
                opening.assert_not_called()
                self.page.locator('[data-view-button="launch"]').click()
                self.page.locator(f'[data-view-button="{agent}"]').click()
                opening.assert_not_called()
                self.page.locator(f'#{agent}-retry').click()
                expect(self.page.locator(f'#{agent}-progress-detail')).to_have_text('1.0 / 4.0 MiB')
                self.page.locator('[data-view-button="launch"]').click()
                self.page.locator(f'[data-view-button="{agent}"]').click()
                expect(self.page.locator(f'#{agent}-progress')).to_be_visible()
                self.screenshot(f'{agent}-manual-install')
                self.page.locator(f'#{agent}-stop').click()
                expect(self.page.locator(f'#{agent}-progress')).to_be_hidden()
                self.assertEqual(opening.call_count, 1)
                self.runtime.plugins.set_enabled(agent, False)
                self.runtime.plugins.set_enabled(agent, True)
                self.assertEqual(opening.call_count, 1)

    def test_agent_groups_and_runtime_management_without_model_service(self):
        from pathlib import Path
        from fastllm_pytools import harness_install, launcher_agent_install
        from fastllm_pytools.launcher_harness import HarnessRuntime
        from fastllm_pytools.launcher_codex import CodexRuntime
        from fastllm_pytools.launcher_opencode import OpenCodeRuntime
        from fastllm_pytools.launcher_claude import ClaudeRuntime
        from test_launcher_agent_management import installed_files

        self.page.clock.resume()
        self.runtime._process = None
        self.runtime._state.update(phase='stopped', ready=False, sessionId='')
        for factory, agent in ((HarnessRuntime, 'harness'), (OpenCodeRuntime, 'opencode'), (CodexRuntime, 'codex'), (ClaudeRuntime, 'claude')):
            setattr(self.runtime, agent, factory(Path(self.temp.name) / agent))
        released = threading.Event()
        self.addCleanup(released.set)

        def install(directory, *args, upgrade=False):
            agent = directory.name
            progress, cancelled = args[-2:]
            progress('download', 1, 4)
            while not released.wait(.02):
                if cancelled.is_set():
                    raise RuntimeError('cancelled')
            installed_files(directory / 'runtime', agent, getattr(self.runtime, agent)._runtime_spec()['version'] if upgrade else 'old')

        with patch('shutil.which', return_value=None), \
                patch.object(harness_install, 'install_runtime', side_effect=install), \
                patch.object(launcher_agent_install, 'install_runtime', side_effect=install):
            navigation = self.page.locator('.app-shell > .sidebar > .navigation')
            expect(navigation.locator('[role="heading"]')).to_have_text(['Model management', 'agent'])
            self.assertEqual(navigation.locator('[data-view-button]').evaluate_all('(nodes) => nodes.map(n => n.dataset.viewButton)'),
                             ['launch', 'download', 'logs', 'hardware', 'webui', 'harness', 'opencode', 'codex', 'claude'])
            for button in navigation.locator('[data-view-button]').all():
                expect(button).to_be_in_viewport(ratio=1)
            expect(self.page.locator('#view-webui [data-manage-agent]')).to_have_count(0)
            navigation.locator('[data-manage-agent]').click()
            manager = self.page.locator('.plugin-runtime-manager')
            expect(manager).to_be_visible()
            expect(manager.locator('[data-agent] option')).to_have_count(4)
            manager.locator('[data-operation="install"]').click()
            expect(manager.locator('[data-status]')).to_have_text('Installing')
            expect(manager.locator('[data-progress]')).to_have_attribute('value', '1')
            expect(manager.locator('[data-operation="upgrade"]')).to_be_disabled()
            manager.locator('[data-close]').click()
            navigation.locator('[data-manage-agent]').click()
            expect(manager.locator('[data-status]')).to_have_text('Installing')
            manager.locator('[data-operation="cancel"]').click()
            expect(manager.locator('[data-status]')).to_have_text('Not installed')
            released.set()
            self.page.on('dialog', lambda dialog: dialog.accept())
            for agent in ('harness', 'opencode', 'codex', 'claude'):
                manager.locator('[data-agent]').select_option(agent)
                manager.locator('[data-operation="install"]').click()
                expect(manager.locator('[data-version]')).to_contain_text('Private runtime · old')
                manager.locator('[data-operation="upgrade"]').click()
                expect(manager.locator('[data-version]')).to_contain_text('Private runtime · ' + getattr(self.runtime, agent)._runtime_spec()['version'])
                workspace = getattr(self.runtime, agent).directory / 'workspace'
                workspace.mkdir(); (workspace / 'keep.txt').write_text('keep me')
                manager.locator('[data-operation="remove"]').click()
                expect(manager.locator('[data-status]')).to_have_text('Not installed')
                self.assertEqual((workspace / 'keep.txt').read_text(), 'keep me')
                expect(manager.locator('[data-operation="remove"]')).to_be_disabled()
            manager.locator('[data-close]').click()
            self.page.locator('[data-view-button="codex"]').click()
            expect(self.page.locator('.management-label')).to_have_text('agent')
            self.page.locator('#view-codex [data-manage-agent]').click()
            expect(manager.locator('[data-agent]')).to_have_value('codex')
            self.page.set_viewport_size({'width':390, 'height':844})
            expect(manager).to_be_visible()
            box = manager.bounding_box()
            self.assertGreaterEqual(box['x'], 0)
            self.assertLessEqual(box['x'] + box['width'], 390)
            self.screenshot('agent-management-mobile')
            manager.locator('[data-close]').click()
            self.page.locator('#language-select').select_option('zh-CN')
            expect(navigation.locator('[role="heading"]')).to_have_text(['模型管理', 'agent'])
            navigation.locator('[data-manage-agent]').click()
            expect(manager.locator('h2')).to_have_text('agent 管理')
            self.assertIsNone(self.runtime._process)

    def test_codex_stream_approval_drafts_and_reload_use_app_server(self):
        from pathlib import Path
        from fastllm_pytools.launcher_codex import CodexRuntime
        from test_launcher_agents import FAKE_CODEX
        self.page.clock.resume()
        self.runtime.codex = CodexRuntime(Path(self.temp.name) / 'codex')
        script = Path(self.temp.name) / 'codex.py'; script.write_text(FAKE_CODEX)
        with patch.object(self.runtime.codex, '_command', return_value=[sys.executable, str(script)]):
            self.page.locator('[data-view-button="codex"]').click()
            expect(self.page.locator('#codex-content')).to_be_visible()
            self.page.locator('.codex-session[data-thread-id="thread-a"]').click()
            self.page.locator('#codex-prompt').fill('Create a small file')
            self.page.locator('#codex-send').click()
            expect(self.page.locator('#codex-messages')).to_contain_text('Hello')
            approval = self.page.locator('.codex-approval')
            expect(approval).to_contain_text('echo hello')
            self.screenshot('codex-command-approval')
            approval.get_by_role('button', name='Allow once', exact=True).click()
            expect(self.page.locator('#codex-messages')).to_contain_text('Hello complete')
            expect(approval).to_have_count(0)
            self.page.locator('#codex-prompt').fill('Draft for another turn')
            self.page.locator('[data-view-button="launch"]').click()
            self.page.locator('[data-view-button="codex"]').click()
            expect(self.page.locator('#codex-prompt')).to_have_value('Draft for another turn')
            process = self.runtime.codex._process
            self.page.reload()
            self.page.locator('[data-view-button="codex"]').click()
            expect(self.page.locator('#codex-prompt')).to_have_value('Draft for another turn')
            expect(self.page.locator('#codex-messages')).to_contain_text('Hello complete')
            self.assertIs(self.runtime.codex._process, process)
            self.screenshot('codex-session-restored')
            self.page.set_viewport_size({'width':390, 'height':844})
            self.screenshot('codex-mobile')
            self.page.locator('#codex-stop').click()
            expect(self.page.locator('#codex-content')).to_be_hidden()

    def test_codex_workspace_picker_selects_server_directory_and_keeps_model_picker_independent(self):
        from pathlib import Path
        from fastllm_pytools.launcher_codex import CodexRuntime
        from test_launcher_agents import FAKE_CODEX
        self.page.clock.resume()
        self.runtime.codex = CodexRuntime(Path(self.temp.name) / 'codex')
        project = Path(self.temp.name) / '项目 workspace'; project.mkdir()
        nested = project / 'nested'; nested.mkdir()
        ordinary_file = project / 'notes.txt'; ordinary_file.write_text('not a directory')
        script = Path(self.temp.name) / 'codex.py'
        script.write_text(FAKE_CODEX.replace("method=request.get('method')", """method=request.get('method')
    if method == 'thread/start':
        thread['cwd'] = request['params']['cwd']
        Path(os.environ['CODEX_HOME'], 'thread-cwd').write_text(thread['cwd'])"""))
        with patch.object(self.runtime.codex, '_command', return_value=[sys.executable, str(script)]):
            self.page.locator('[data-view-button="codex"]').click()
            workspace = self.page.locator('#codex-workspace')
            browse = self.page.locator('#codex-browse-workspace')
            expect(browse).to_be_enabled()
            workspace.fill(str(project))
            self.page.locator('#codex-prompt').fill('Preserve my draft')
            browse.click()
            picker = self.page.locator('#folder-picker-modal')
            expect(self.page.locator('#folder-picker-title')).to_have_text('Choose a workspace folder')
            expect(picker.locator('.file-icon')).to_have_count(0)
            picker.locator('.folder-picker-entry').filter(has_text='nested').click()
            expect(self.page.locator('#folder-picker-current')).to_have_value(str(nested))
            self.page.locator('#folder-picker-up').click()
            expect(self.page.locator('#folder-picker-current')).to_have_value(str(project))
            self.screenshot('codex-workspace-picker')
            self.page.keyboard.press('Escape')
            expect(picker).to_be_hidden(); expect(browse).to_be_focused()
            expect(workspace).to_have_value(str(project))
            browse.click()
            # Typing a file path still selects its containing directory in this mode.
            self.page.locator('#folder-picker-current').fill(str(ordinary_file))
            self.page.locator('#folder-picker-current').press('Enter')
            expect(self.page.locator('#folder-picker-current')).to_have_value(str(project))
            expect(self.page.locator('#folder-picker-select')).to_have_text('Select this folder')
            self.page.locator('#folder-picker-select').click()
            expect(workspace).to_have_value(str(project)); expect(workspace).to_be_focused()
            expect(self.page.locator('#codex-prompt')).to_have_value('Preserve my draft')
            self.page.locator('#codex-prompt').press('Enter')
            expect(self.page.locator('.codex-session.active')).to_have_count(1)
            self.assertEqual((self.runtime.codex.directory / 'home/thread-cwd').read_text(), str(project.resolve()))
            expect(self.page.locator('#codex-session-workspace')).to_have_text(str(project))
            expect(self.page.locator('#codex-workspace-setup')).to_be_hidden()
            self.page.reload()
            self.page.locator('[data-view-button="codex"]').click()
            expect(self.page.locator('#codex-session-workspace')).to_have_text(str(project))
            self.page.locator('[data-view-button="launch"]').click()
            self.page.locator('#new-profile').click()
            self.page.locator('[data-config-mode][value="custom"]').check()
            self.page.locator('#model-path').fill(str(project))
            self.page.locator('#choose-model-folder').click()
            expect(self.page.locator('#folder-picker-title')).to_have_text('Choose a model file or folder')
            picker.locator('.folder-picker-entry').filter(has_text='notes.txt').click()
            expect(self.page.locator('#folder-picker-select')).to_have_text('Select this file')
            self.page.locator('#folder-picker-select').click()
            expect(self.page.locator('#model-path')).to_have_value(str(ordinary_file))
            expect(workspace).to_have_value(str(project))

    def start_claude_sdk(self):
        import shutil
        from pathlib import Path
        from fastllm_pytools.launcher_claude import ClaudeRuntime
        from test_launcher_claude import BRIDGE, FAKE_SDK

        self.page.clock.resume()
        self.runtime.claude = ClaudeRuntime(Path(self.temp.name) / 'claude')
        sdk = Path(self.temp.name) / 'sdk.mjs'; sdk.write_text(FAKE_SDK)
        command = patch.object(self.runtime.claude, '_command', return_value=[shutil.which('node'), str(BRIDGE), str(sdk)])
        command.start(); self.addCleanup(command.stop)
        metadata = patch('fastllm_pytools.launcher_agent_runtime.with_model_metadata',
            side_effect=lambda service, key:dict(service, modelMetadata={
                'supported_reasoning_efforts':['none', 'low', 'medium', 'xhigh']}))
        metadata.start(); self.addCleanup(metadata.stop)
        self.page.locator('[data-view-button="claude"]').click()
        expect(self.page.locator('#claude-content')).to_be_visible()

    def test_claude_workspace_markdown_approval_and_session_resume(self):
        from pathlib import Path
        self.start_claude_sdk()
        project = Path(self.temp.name) / 'Claude project'; project.mkdir()
        (project / 'notes.txt').write_text('not a directory')
        self.page.locator('#claude-workspace').fill(str(project))
        self.page.locator('#claude-browse-workspace').click()
        expect(self.page.locator('#folder-picker-title')).to_have_text('Choose a workspace folder')
        expect(self.page.locator('#folder-picker-modal .file-icon')).to_have_count(0)
        self.page.locator('#folder-picker-select').click()
        expect(self.page.locator('#claude-workspace')).to_have_value(str(project))
        self.screenshot('claude-new-light')
        self.page.locator('#claude-effort').select_option('low')
        prompt = self.page.locator('#claude-prompt')
        prompt.fill('Write a file'); prompt.press('Shift+Enter')
        expect(prompt).to_have_value('Write a file\n')
        prompt.press('Enter')
        expect(self.page.locator('#claude-approvals')).to_contain_text('Write example.txt?')
        expect(self.page.locator('#claude-messages .agentMessage h2')).to_have_text('Reply')
        expect(self.page.locator('#claude-messages .agentMessage .codex-markdown strong')).to_have_text('Hello')
        user = self.page.locator('#claude-messages .userMessage')
        assistant = self.page.locator('#claude-messages .agentMessage')
        self.assertGreater(user.bounding_box()['x'], assistant.bounding_box()['x'])
        expect(self.page.locator('#claude-session-workspace')).to_have_text(str(project))
        expect(self.page.locator('#claude-workspace-setup')).to_be_hidden()
        self.screenshot('claude-approval-light')
        self.page.locator('#claude-approvals').get_by_role('button', name='Allow once', exact=True).click()
        expect(self.page.locator('#claude-cancel')).to_be_hidden()
        self.page.reload()
        self.page.locator('[data-view-button="claude"]').click()
        expect(self.page.locator('#claude-session-workspace')).to_have_text(str(project))
        expect(self.page.locator('#claude-messages .agentMessage h2')).to_have_text('Reply')
        expect(self.page.locator('#claude-effort')).to_have_value('low')
        prompt.fill('Ask a question'); prompt.press('Enter')
        expect(self.page.locator('#claude-approvals')).to_contain_text('Which file?')
        self.page.locator('#claude-approvals input').fill('README.md')
        self.page.locator('#claude-approvals').get_by_role('button', name='Submit answers').click()
        expect(self.page.locator('#claude-cancel')).to_be_hidden()
        probe = json.loads((self.runtime.claude.directory / 'home/probe.json').read_text())
        self.assertEqual(probe['effort'], 'low')
        self.assertIn('resume', probe)

    def test_claude_multiple_blocks_remain_distinct_after_reload(self):
        self.start_claude_sdk()
        self.page.locator('#claude-prompt').fill('multiple blocks')
        self.page.locator('#claude-prompt').press('Enter')
        expect(self.page.locator('#claude-approvals')).to_contain_text('Write example.txt?')

        def assert_blocks():
            messages = self.page.locator('#claude-messages')
            expect(messages.locator('.agentMessage')).to_have_count(2)
            expect(messages.locator('.agentMessage h2')).to_have_text(['Before', 'After'])
            expect(messages.locator('.reasoning')).to_have_count(1)
            expect(messages.locator('.reasoning .codex-markdown')).to_have_text('Check the project first.')
            expect(messages.locator('.mcpToolCall')).to_have_count(1)

        assert_blocks()
        self.page.locator('#claude-approvals').get_by_role('button', name='Allow once', exact=True).click()
        expect(self.page.locator('#claude-send')).to_be_enabled()
        self.page.reload()
        self.page.locator('[data-view-button="claude"]').click()
        assert_blocks()

    def test_claude_none_effort_is_sent_and_restored(self):
        self.start_claude_sdk()
        effort = self.page.locator('#claude-effort')
        expect(effort.locator('option[value="none"]')).to_have_text('None (none)')
        effort.select_option('none')
        self.page.locator('#claude-prompt').fill('Reply without thinking')
        self.page.locator('#claude-prompt').press('Enter')
        expect(self.page.locator('#claude-approvals')).to_contain_text('Write example.txt?')
        probe = json.loads((self.runtime.claude.directory / 'home/probe.json').read_text())
        self.assertEqual(probe['thinking'], {'type':'disabled'})
        self.assertEqual(probe['extraBody']['thinking'], {'type':'disabled'})
        self.assertIsNone(probe['extraBody']['output_config']['effort'])
        self.page.locator('#claude-approvals').get_by_role('button', name='Decline', exact=True).click()
        expect(self.page.locator('#claude-send')).to_be_enabled()
        self.page.reload()
        self.page.locator('[data-view-button="claude"]').click()
        expect(effort).to_have_value('none')
        self.page.locator('#claude-new').click()
        expect(effort).to_have_value('xhigh')

    def test_claude_template_error_is_visible_and_another_message_can_be_sent(self):
        self.start_claude_sdk()
        prompt = self.page.locator('#claude-prompt')
        prompt.fill('template failure')
        prompt.press('Enter')
        expect(self.page.locator('#claude-chat-error')).to_contain_text('Unsupported model template input')
        expect(self.page.locator('#claude-send')).to_be_enabled()
        expect(self.page.locator('#claude-cancel')).to_be_hidden()
        prompt.fill('Ask a question')
        prompt.press('Enter')
        expect(self.page.locator('#claude-chat-error')).to_be_hidden()
        expect(self.page.locator('#claude-approvals')).to_contain_text('Which file?')
        self.page.locator('#claude-approvals input').fill('README.md')
        self.page.locator('#claude-approvals').get_by_role('button', name='Submit answers').click()
        expect(self.page.locator('#claude-send')).to_be_enabled()

    def test_claude_theme_mobile_layout_drafts_and_cancel(self):
        from pathlib import Path
        self.start_claude_sdk()
        project_a = Path(self.temp.name) / 'project-a'; project_a.mkdir()
        project_b = Path(self.temp.name) / 'project-b'; project_b.mkdir()
        workspace = self.page.locator('#claude-workspace')
        prompt = self.page.locator('#claude-prompt')
        workspace.fill(str(project_a)); prompt.fill('Draft A')
        workspace.fill(str(project_b)); prompt.fill('Draft B')
        workspace.fill(str(project_a)); expect(prompt).to_have_value('Draft A')
        self.page.evaluate("window.ftllmLauncherTheme.setPreference('dark')")
        expect(self.page.locator('html')).to_have_attribute('data-theme', 'dark')
        prompt.press('Enter')
        expect(self.page.locator('#claude-approvals')).to_contain_text('Write example.txt?')
        self.screenshot('claude-dark')
        self.page.locator('#claude-cancel').click()
        expect(self.page.locator('#claude-approvals .codex-approval')).to_have_count(0)
        expect(self.page.locator('#claude-send')).to_be_enabled()
        self.page.set_viewport_size({'width':390, 'height':844})
        expect(self.page.locator('#claude-toggle-sessions')).to_be_visible()
        expect(self.page.locator('#claude-send')).to_be_in_viewport(ratio=1)
        self.assertLessEqual(self.page.evaluate('document.documentElement.scrollWidth'), 390)
        self.screenshot('claude-mobile-dark')
        self.page.locator('#claude-toggle-sessions').click()
        expect(self.page.locator('#claude-sidebar')).to_be_visible()

    def start_codex_projects(self):
        from pathlib import Path
        from fastllm_pytools.launcher_codex import CodexRuntime
        self.page.clock.resume()
        projects = [Path(self.temp.name) / name for name in ('项目 A', 'project B')]
        for project in projects:
            project.mkdir()
        self.runtime.codex = CodexRuntime(Path(self.temp.name) / 'codex')
        script = Path(self.temp.name) / 'codex-projects.py'
        script.write_text('projects = ' + repr([str(path) for path in projects]) + '\n' + '''
import json, os, sys
from pathlib import Path
def send(message):
    print(json.dumps(message), flush=True)
def event(method, thread, **params):
    send({'method':method, 'params':dict(threadId=thread['id'], **params)})
threads = {str(index):dict(id=str(index), cwd=projects[project], name=name, turns=[], updatedAt=index)
           for index, project, name in ((1, 0, 'Saved A'), (2, 0, 'Another A'), (3, 1, 'Saved B'))}
for line in sys.stdin:
    request = json.loads(line)
    with Path(os.environ['CODEX_HOME'], 'requests.jsonl').open('a') as log:
        log.write(json.dumps(request) + '\\n')
    method, params = request.get('method'), request.get('params', {})
    if method == 'initialized': continue
    result = {}
    if method == 'thread/start':
        key = str(len(threads) + 1)
        threads[key] = dict(id=key, cwd=params['cwd'], name='New task', turns=[], updatedAt=len(threads)+1)
        result = {'thread':threads[key]}
    if method in ('thread/read', 'thread/resume'):
        result = {'thread':threads[params['threadId']]}
    if method == 'thread/list':
        result = {'data':[thread for thread in threads.values() if not thread.get('archived')
                          and params.get('searchTerm', '').lower() in thread['name'].lower()], 'nextCursor':None}
    if method == 'thread/archive': threads[params['threadId']]['archived'] = True
    if method == 'turn/start':
        thread = threads[params['threadId']]
        key = thread['id'] + '-' + str(len(thread['turns']))
        turn = dict(id=key, status='inProgress', items=[
            dict(id='user-'+key, type='userMessage', content=params['input']),
            dict(id='agent-'+key, type='agentMessage', text='**Reply** from ' + thread['cwd'])])
        thread['turns'].append(turn)
        event('turn/started', thread, turn=turn)
        for item in turn['items']: event('item/started', thread, turnId=key, item=item)
        result = {'turn':turn}
    if method == 'turn/interrupt':
        thread = threads[params['threadId']]
        turn = thread['turns'][-1]; turn['status'] = 'completed'
        event('turn/completed', thread, turn=turn)
    send({'id':request['id'], 'result':result})
''')
        command = patch.object(self.runtime.codex, '_command', return_value=[sys.executable, str(script)])
        command.start(); self.addCleanup(command.stop)
        self.page.locator('[data-view-button="codex"]').click()
        expect(self.page.locator('.codex-session')).to_have_count(3)
        return projects

    def test_codex_new_sessions_remain_visible_before_history_is_indexed(self):
        self.start_codex_projects()
        unindexed = {'4', '5'}

        def intercept(route):
            method = route.request.post_data_json['method']
            if method not in ('thread/list', 'thread/start'):
                route.continue_(); return
            response = route.fetch()
            result = response.json()
            if method == 'thread/list':
                result['data'] = [thread for thread in result['data'] if thread['id'] not in unindexed]
            else:
                # app-server has no preview until it records the first turn.
                result['thread'].update(name='', preview='')
            route.fulfill(response=response, json=result)

        self.page.route('**/api/agents/codex/rpc', intercept)
        prompt = self.page.locator('#codex-prompt')
        search = self.page.locator('#codex-search')

        def search_sessions(text):
            with self.page.expect_response(lambda response: response.url.endswith('/api/agents/codex/rpc')
                    and response.request.post_data_json.get('method') == 'thread/list'):
                search.fill(text)

        prompt.fill('Inspect the first project')
        prompt.press('Enter')
        expect(self.page.locator('#codex-messages')).to_contain_text('Reply')
        expect(self.page.locator('#codex-cancel')).to_be_visible()
        first = self.page.locator('#codex-session-list [data-thread-id="4"]')
        expect(first).to_be_visible()
        expect(first).to_contain_text('Inspect the first project')
        expect(first).to_have_attribute('aria-current', 'page')
        search_sessions('Inspect')
        expect(first).to_be_visible()
        search_sessions('Saved A')
        expect(first).to_have_count(0)
        search_sessions('')
        expect(first).to_be_visible()

        self.page.locator('#codex-new').click()
        prompt.fill('Inspect the second project')
        prompt.press('Enter')
        second = self.page.locator('#codex-session-list [data-thread-id="5"]')
        expect(second).to_be_visible()
        expect(second).to_have_attribute('aria-current', 'page')
        search_sessions('Inspect')
        expect(first).to_be_visible()
        expect(second).to_be_visible()
        first.click()
        expect(first).to_have_attribute('aria-current', 'page')
        expect(self.page.locator('#codex-cancel')).to_be_visible()

        # A finished turn can still precede the history index update.
        search_sessions('')
        self.page.locator('#codex-cancel').click()
        expect(self.page.locator('#codex-send')).to_be_enabled()
        expect(first).to_be_visible()
        self.page.locator('#codex-archive').click()
        expect(first).to_have_count(0)
        search_sessions('Saved')
        search_sessions('')
        expect(first).to_have_count(0)
        expect(second).to_be_visible()

        # Reconcile with the indexed summary without adding a duplicate.
        unindexed.clear()
        search_sessions('New task')
        expect(second).to_have_count(1)
        expect(second).to_contain_text('New task')

    def test_codex_reasoning_effort_is_sent_and_remembered_per_session(self):
        def metadata(service, api_key):
            return dict(service, modelMetadata={"supported_reasoning_efforts": ["low", "medium", "xhigh"],
                                               "default_reasoning_effort": "xhigh"})
        with patch('fastllm_pytools.launcher_agent_runtime.with_model_metadata', side_effect=metadata):
            self.start_codex_projects()
        effort = self.page.locator('#codex-effort')
        expect(effort).to_be_enabled(); expect(effort).to_have_value('xhigh')
        self.assertEqual(effort.locator('option').evaluate_all('(nodes) => nodes.map(n => n.value)'),
                         ['low', 'medium', 'xhigh'])
        self.page.locator('[data-thread-id="1"]').click()
        effort.select_option('low')
        self.page.locator('[data-thread-id="3"]').click()
        expect(effort).to_have_value('xhigh')
        effort.select_option('medium')
        self.page.locator('[data-thread-id="1"]').click()
        expect(effort).to_have_value('low')
        self.page.locator('#codex-prompt').fill('Check reasoning')
        self.page.locator('#codex-prompt').press('Enter')
        expect(self.page.locator('#codex-messages')).to_contain_text('Reply')
        expect(effort).to_be_disabled()
        self.page.locator('#codex-cancel').click()
        expect(effort).to_be_enabled()
        self.page.locator('#codex-new').click()
        expect(effort).to_have_value('xhigh')
        effort.select_option('medium')
        self.page.locator('#codex-prompt').fill('New conversation effort')
        self.page.locator('#codex-prompt').press('Enter')
        expect(self.page.locator('[data-thread-id="4"]')).to_have_class('codex-session active')
        expect(self.page.locator('#codex-messages')).to_contain_text('Reply')
        expect(effort).to_have_value('medium')
        self.page.locator('#codex-cancel').click()
        self.page.reload()
        self.page.locator('[data-view-button="codex"]').click()
        expect(effort).to_have_value('medium')
        self.page.locator('[data-thread-id="1"]').click()
        expect(effort).to_have_value('low')
        self.screenshot('codex-reasoning-light')
        self.page.locator('#theme-select').select_option('dark')
        self.screenshot('codex-reasoning-dark')
        self.page.set_viewport_size({'width':390, 'height':844})
        expect(effort).to_be_visible()
        expect(self.page.locator('#codex-send')).to_be_visible()
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        self.screenshot('codex-reasoning-mobile')
        requests = [json.loads(line) for line in (self.runtime.codex.directory / 'home/requests.jsonl').read_text().splitlines()]
        self.assertEqual([(r['params']['threadId'], r['params']['effort']) for r in requests
                          if r.get('method') == 'turn/start'], [('1', 'low'), ('4', 'medium')])

    def test_codex_none_effort_is_sent_and_restored(self):
        with patch('fastllm_pytools.launcher_agent_runtime.with_model_metadata', side_effect=lambda service, key:
                dict(service, modelMetadata={'supported_reasoning_efforts':['none', 'low', 'medium', 'xhigh']})):
            self.start_codex_projects()
        effort = self.page.locator('#codex-effort')
        expect(effort.locator('option[value="none"]')).to_have_text('None (none)')
        effort.select_option('none')
        self.page.locator('#codex-prompt').fill('Reply without thinking')
        self.page.locator('#codex-prompt').press('Enter')
        expect(self.page.locator('#codex-messages')).to_contain_text('Reply')
        requests = [json.loads(line) for line in (self.runtime.codex.directory / 'home/requests.jsonl').read_text().splitlines()]
        self.assertEqual([r['params']['effort'] for r in requests if r.get('method') == 'turn/start'], ['none'])
        self.page.locator('#codex-cancel').click()
        expect(self.page.locator('#codex-send')).to_be_enabled()
        self.page.reload()
        self.page.locator('[data-view-button="codex"]').click()
        expect(effort).to_have_value('none')
        self.page.locator('#codex-new').click()
        expect(effort).to_have_value('xhigh')

    def test_codex_unknown_model_disables_effort_with_explanation(self):
        self.start_codex_projects()
        effort = self.page.locator('#codex-effort')
        expect(effort).to_be_disabled()
        expect(effort).to_have_value('')
        expect(effort).to_have_attribute('title', 'This model does not advertise adjustable reasoning effort.')

    def test_codex_groups_threads_by_workspace_and_restores_bound_directory(self):
        project_a, project_b = self.start_codex_projects()
        group_a = self.page.locator('.codex-project').filter(has=self.page.locator('[data-thread-id="1"]'))
        group_b = self.page.locator('.codex-project').filter(has=self.page.locator('[data-thread-id="3"]'))
        expect(self.page.locator('.codex-project')).to_have_count(2)
        expect(group_a.locator('.codex-session')).to_have_count(2)
        expect(group_b.locator('.codex-session')).to_have_count(1)
        prompt = self.page.locator('#codex-prompt')
        directory = self.page.locator('#codex-session-workspace')
        setup = self.page.locator('#codex-workspace-setup')
        self.page.locator('[data-thread-id="1"]').click()
        expect(directory).to_have_text(str(project_a)); expect(setup).to_be_hidden()
        prompt.fill('Draft in A')
        self.page.locator('[data-thread-id="3"]').click()
        expect(directory).to_have_text(str(project_b)); expect(prompt).to_have_value('')
        prompt.fill('Draft in B')
        self.page.locator('[data-thread-id="1"]').click()
        expect(prompt).to_have_value('Draft in A')
        group_a.locator('.codex-project-new').click()
        expect(setup).to_be_visible(); expect(directory).to_have_text(str(project_a))
        prompt.fill('New draft in A')
        group_b.locator('.codex-project-new').click()
        expect(prompt).to_have_value(''); expect(directory).to_have_text(str(project_b))
        prompt.fill('New task in B')
        group_a.locator('.codex-project-new').click()
        expect(prompt).to_have_value('New draft in A')
        group_b.locator('.codex-project-new').click()
        expect(prompt).to_have_value('New task in B')
        expect(self.page.locator('.codex-session')).to_have_count(3)
        prompt.press('Enter')
        expect(self.page.locator('[data-thread-id="4"]')).to_have_class('codex-session active')
        expect(self.page.locator('#codex-messages')).to_contain_text('Reply from ' + str(project_b))
        expect(setup).to_be_hidden(); expect(directory).to_have_text(str(project_b))
        expect(group_b.locator('.codex-session')).to_have_count(2)
        self.page.locator('#codex-cancel').click()
        expect(self.page.locator('#codex-send')).to_be_enabled()
        self.screenshot('codex-projects-light')
        self.page.reload()
        self.page.locator('[data-view-button="codex"]').click()
        expect(directory).to_have_text(str(project_b)); expect(setup).to_be_hidden()
        expect(self.page.locator('#codex-messages')).to_contain_text('Reply from ' + str(project_b))
        self.page.locator('#codex-archive').click()
        expect(setup).to_be_visible(); expect(directory).to_have_text(str(project_b))
        expect(self.page.locator('.codex-session')).to_have_count(3)
        self.page.locator('[data-thread-id="1"]').click()
        expect(prompt).to_have_value('Draft in A'); expect(directory).to_have_text(str(project_a))
        self.page.locator('#codex-new').click()
        expect(prompt).to_have_value('New draft in A')
        self.page.locator('#codex-workspace').fill(str(project_a / 'missing'))
        prompt.press('Enter')
        expect(self.page.locator('#codex-chat-error')).to_be_visible()
        expect(self.page.locator('.codex-session.active')).to_have_count(0)
        expect(prompt).to_have_value('New draft in A')
        self.page.locator('[data-thread-id="3"]').click()
        expect(prompt).to_have_value('Draft in B'); expect(directory).to_have_text(str(project_b))
        expect(self.page.locator('#codex-chat-error')).to_be_hidden()
        self.page.locator('#codex-search').fill('Saved A')
        expect(self.page.locator('.codex-project')).to_have_count(1)
        expect(self.page.locator('.codex-session')).to_have_count(1)
        expect(directory).to_have_text(str(project_b))
        log = self.runtime.codex.directory / 'home/requests.jsonl'
        requests = [json.loads(line) for line in log.read_text().splitlines()]
        starts = [r['params'] for r in requests if r.get('method') == 'thread/start']
        self.assertEqual([params['cwd'] for params in starts], [str(project_b)])
        turns = [r['params'] for r in requests if r.get('method') == 'turn/start']
        self.assertEqual([params['threadId'] for params in turns], ['4'])
        for request in requests:
            if request.get('method') in ('thread/resume', 'turn/start'):
                self.assertNotIn('cwd', request['params'])

    def test_codex_directory_changes_restore_drafts_without_overwriting_other_projects(self):
        project_a, project_b = self.start_codex_projects()
        project_c = project_a.parent / 'new project'; project_c.mkdir()
        group_a = self.page.locator('.codex-project').filter(has=self.page.locator('[data-thread-id="1"]'))
        group_b = self.page.locator('.codex-project').filter(has=self.page.locator('[data-thread-id="3"]'))
        prompt = self.page.locator('#codex-prompt')
        workspace = self.page.locator('#codex-workspace')
        group_a.locator('.codex-project-new').click(); prompt.fill('Draft for A')
        group_b.locator('.codex-project-new').click(); prompt.fill('Draft for B')
        group_a.locator('.codex-project-new').click()
        expect(prompt).to_have_value('Draft for A')

        # A directly entered directory must restore B before persisting its input.
        workspace.fill(str(project_b))
        expect(prompt).to_have_value('Draft for B')
        expect(self.page.locator('#codex-session-workspace')).to_have_text(str(project_b))
        prompt.fill('Edited draft for B')

        # The folder dialog dispatches the same input event after a selection.
        self.page.locator('#codex-browse-workspace').click()
        self.page.locator('#folder-picker-current').fill(str(project_a))
        self.page.locator('#folder-picker-current').press('Enter')
        expect(self.page.locator('#folder-picker-select')).to_be_enabled()
        self.page.locator('#folder-picker-select').click()
        expect(workspace).to_have_value(str(project_a))
        expect(prompt).to_have_value('Draft for A')
        saved = json.loads(self.page.evaluate('localStorage.getItem("ftllm.codex.sessions")'))['drafts']
        self.assertEqual(saved['workspace:' + str(project_a)], 'Draft for A')
        self.assertEqual(saved['workspace:' + str(project_b)], 'Edited draft for B')

        # A new destination carries the input, while a deliberately empty saved
        # draft must stay empty when revisited from a different directory.
        workspace.fill(str(project_c))
        expect(prompt).to_have_value('Draft for A')
        prompt.fill('Draft for C')
        workspace.fill(str(project_b))
        expect(prompt).to_have_value('Edited draft for B')
        prompt.clear()
        workspace.fill(str(project_a))
        expect(prompt).to_have_value('Draft for A')
        workspace.fill(str(project_b))
        expect(prompt).to_have_value('')
        self.page.reload()
        self.page.locator('[data-view-button="codex"]').click()
        expect(workspace).to_be_enabled()
        expect(workspace).to_have_value(str(project_b))
        expect(prompt).to_have_value('')
        workspace.fill(str(project_c))
        expect(prompt).to_have_value('Draft for C')
        workspace.fill(str(project_a))
        expect(prompt).to_have_value('Draft for A')

    def test_codex_enter_respects_composition_and_messages_align_by_role(self):
        project_a, _ = self.start_codex_projects()
        self.page.locator('[data-thread-id="1"]').click()
        expect(self.page.locator('#codex-send')).to_be_enabled()
        prompt = self.page.locator('#codex-prompt')
        prompt.fill('First line'); prompt.press('Shift+Enter'); prompt.type('第二行')
        expect(prompt).to_have_value('First line\n第二行')
        prompt.dispatch_event('compositionstart')
        prompt.press('Enter')
        prompt.dispatch_event('keydown', {'key':'Enter', 'isComposing':True})
        prompt.dispatch_event('compositionend')
        prompt.dispatch_event('keydown', {'key':'Enter', 'keyCode':229})
        prompt.dispatch_event('keydown', {'key':'Enter', 'repeat':True})
        expect(self.page.locator('#codex-messages .userMessage')).to_have_count(0)
        # The composition confirmation can insert a line break in a synthetic event;
        # sending must still submit exactly the current text once.
        sent = prompt.input_value()
        prompt.press('Enter')
        user = self.page.locator('#codex-messages .userMessage')
        agent = self.page.locator('#codex-messages .agentMessage')
        expect(user).to_have_count(1); expect(user).to_contain_text('第二行')
        expect(agent.locator('.codex-markdown strong')).to_have_text('Reply')
        prompt.fill('Next draft'); prompt.press('Enter')
        expect(user).to_have_count(1); expect(prompt).to_have_value('Next draft')
        for width, height, theme in ((1280, 900, 'light'), (1280, 900, 'dark'), (390, 844, 'dark')):
            self.page.set_viewport_size({'width':width, 'height':height})
            self.page.locator('#theme-select').select_option(theme)
            user_box, agent_box = user.bounding_box(), agent.bounding_box()
            self.assertGreater(user_box['x'], agent_box['x'])
            self.assertGreater(user_box['x'] + user_box['width'], agent_box['x'] + agent_box['width'])
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            self.screenshot(f'codex-conversation-{theme}-{width}')
        expect(self.page.locator('#codex-sidebar')).to_be_hidden()
        self.page.locator('#codex-toggle-sessions').click()
        expect(self.page.locator('#codex-sidebar')).to_be_visible()
        self.page.locator('[data-thread-id="2"]').click()
        expect(self.page.locator('#codex-sidebar')).to_be_hidden()
        expect(self.page.locator('#codex-session-workspace')).to_have_text(str(project_a))
        expect(prompt).to_have_value('')
        self.page.locator('#codex-toggle-sessions').click()
        self.page.locator('[data-thread-id="1"]').click()
        expect(prompt).to_have_value('Next draft')
        self.page.locator('#codex-cancel').click()
        expect(self.page.locator('#codex-send')).to_be_enabled()
        prompt.press('Control+Enter')
        expect(user).to_have_count(2)
        log = self.runtime.codex.directory / 'home/requests.jsonl'
        requests = [json.loads(line) for line in log.read_text().splitlines()]
        turns = [r['params'] for r in requests if r.get('method') == 'turn/start']
        self.assertEqual([params['input'][0]['text'] for params in turns], [sent, 'Next draft'])
        self.page.locator('#language-select').select_option('zh-CN')
        expect(self.page.locator('#view-codex .codex-compose-hint')).to_have_text('Enter 发送，Shift+Enter 换行')
        expect(self.page.locator('#codex-toggle-sessions')).to_have_text('项目与会话')
        self.screenshot('codex-conversation-zh-mobile')
        self.page.set_viewport_size({'width':1280, 'height':900})
        self.screenshot('codex-conversation-zh-dark')
        self.page.locator('#codex-new').click()
        expect(self.page.locator('#codex-workspace-setup')).to_be_visible()
        expect(self.page.locator('label[for="codex-workspace"]')).to_have_text('会话工作目录')
        self.page.locator('#theme-select').select_option('light')
        self.screenshot('codex-new-conversation-zh-light')

    def test_codex_migrates_the_old_new_conversation_draft_only_once(self):
        # Install the legacy data after pagehide has saved the outgoing page's input.
        self.page.add_init_script('''if (!sessionStorage.getItem('legacy-draft-seeded')) {
            localStorage.setItem('ftllm.codex.sessions', JSON.stringify({
                selected:'', workspace:'', drafts:{'':'Unsent draft from the previous version'}
            }));
            sessionStorage.setItem('legacy-draft-seeded', 'true');
        }''')
        self.page.reload()
        self.start_codex_projects()
        prompt = self.page.locator('#codex-prompt')
        expect(prompt).to_have_value('Unsent draft from the previous version')
        prompt.press('Enter')
        expect(self.page.locator('#codex-messages .userMessage')).to_contain_text('Unsent draft from the previous version')
        self.page.locator('#codex-cancel').click()
        expect(self.page.locator('#codex-send')).to_be_enabled()
        self.page.locator('#codex-new').click()
        expect(prompt).to_have_value('')
        self.page.reload()
        self.page.locator('[data-view-button="codex"]').click()
        expect(self.page.locator('#codex-send')).to_be_enabled()
        expect(prompt).to_have_value('')

    def test_codex_slow_resume_cannot_replace_another_conversations_workspace(self):
        project_a, project_b = self.start_codex_projects()
        pending = []

        def intercept(route):
            payload = route.request.post_data_json
            if payload.get('method') == 'thread/resume' and payload['params']['threadId'] == '1':
                pending.append(route)
            else:
                route.continue_()

        self.page.route('**/api/agents/codex/rpc', intercept)
        self.page.locator('[data-thread-id="1"]').click()
        expect(self.page.locator('#codex-send')).to_be_disabled()
        expect(self.page.locator('#codex-session-workspace')).to_have_text(str(project_a))
        self.page.locator('#codex-prompt').fill('A draft while loading')
        self.page.locator('[data-thread-id="3"]').click()
        expect(self.page.locator('#codex-send')).to_be_enabled()
        self.page.locator('#codex-prompt').fill('B draft')
        self.assertEqual(len(pending), 1)
        with self.page.expect_response(lambda response: response.request.post_data_json.get('method') == 'thread/resume'
                                       if response.request.url.endswith('/codex/rpc') else False):
            pending[0].continue_()
        # A later request completes after the stale resume response was handled.
        self.page.locator('#codex-search').fill('Saved')
        expect(self.page.locator('.codex-session')).to_have_count(2)
        expect(self.page.locator('#codex-session-workspace')).to_have_text(str(project_b))
        expect(self.page.locator('#codex-prompt')).to_have_value('B draft')
        expect(self.page.locator('.codex-session.active')).to_have_attribute('data-thread-id', '3')
        self.page.unroute('**/api/agents/codex/rpc', intercept)
        self.page.locator('[data-thread-id="1"]').click()
        expect(self.page.locator('#codex-prompt')).to_have_value('A draft while loading')
        expect(self.page.locator('#codex-session-workspace')).to_have_text(str(project_a))

    def test_codex_markdown_streaming_and_history_use_safe_renderer(self):
        from pathlib import Path
        from fastllm_pytools.launcher_codex import CodexRuntime
        from test_launcher_agents import FAKE_CODEX
        self.page.clock.resume()
        self.runtime.codex = CodexRuntime(Path(self.temp.name) / 'codex')
        source = ('## Markdown answer\n\n**Bold** and *italic* with `inline code`.\n\n'
                  '| Name | Value |\n| --- | ---: |\n| **Item** | 42 |\n\n'
                  '> Quoted **text**\n\n1. First\n   - Nested\n2. Second\n\n'
                  '- [x] Done\n- [ ] Pending\n\n'
                  '[Documentation](https://example.com/docs) [Unsafe](javascript:alert(1))\n\n'
                  '<img src=x onerror=alert(1)>\n\n```html\n<h1>Streaming code</h1>')
        script = Path(self.temp.name) / 'codex.py'
        script.write_text(FAKE_CODEX.replace("delta='Hello'", 'delta=' + repr(source))
            .replace("'Hello complete'", repr(source + '\n```'))
            .replace("event('turn/started',turn=turn)", """event('turn/started',turn=turn)
        event('item/started',turnId='turn-a',item={'id':'user-a','type':'userMessage','content':[{'type':'text','text':'**User text**'}]})
        event('item/started',turnId='turn-a',item={'id':'reason-a','type':'reasoning','summary':['**Thinking**']})
        event('item/started',turnId='turn-a',item={'id':'command-a','type':'commandExecution','command':'echo test','aggregatedOutput':'**plain output** <img src=x onerror=alert(1)>'})"""))
        with patch.object(self.runtime.codex, '_command', return_value=[sys.executable, str(script)]):
            self.page.locator('[data-view-button="codex"]').click()
            self.page.locator('.codex-session[data-thread-id="thread-a"]').click()
            self.page.locator('#codex-prompt').fill('Show Markdown')
            self.page.locator('#codex-send').click()
            message = self.page.locator('#codex-messages .agentMessage .codex-markdown')
            expect(message.locator('h2')).to_have_text('Markdown answer')
            expect(message.locator('th')).to_have_text(['Name', 'Value'])
            expect(message.locator('td').nth(1)).to_have_css('text-align', 'right')
            expect(message.locator('blockquote strong')).to_have_text('text')
            expect(message.locator('ol ul li')).to_have_text('Nested')
            expect(message.locator('input[type="checkbox"]')).to_have_count(2)
            expect(message.locator('input[type="checkbox"]').first).to_be_checked()
            expect(message.locator('a')).to_have_count(1)
            expect(message.locator('a')).to_have_attribute('rel', 'noopener noreferrer')
            expect(message.locator('img,script')).to_have_count(0)
            expect(message.locator('.code-block code')).to_have_text('<h1>Streaming code</h1>')
            expect(self.page.locator('#codex-messages .userMessage .codex-markdown strong')).to_have_text('User text')
            self.page.locator('#codex-messages .reasoning summary').click()
            expect(self.page.locator('#codex-messages .reasoning .codex-markdown strong')).to_have_text('Thinking')
            command = self.page.locator('#codex-messages .commandExecution')
            command.locator('summary').click()
            expect(command.locator('pre')).to_contain_text('**plain output**')
            expect(command.locator('strong,img')).to_have_count(0)
            self.context.grant_permissions(['clipboard-read', 'clipboard-write'])
            message.locator('.copy-code').click()
            expect(message.locator('.copy-code')).to_have_text('Copied')
            self.assertEqual(self.page.evaluate('navigator.clipboard.readText()'), '<h1>Streaming code</h1>')
            self.page.locator('.codex-approval').get_by_role('button', name='Allow once', exact=True).click()
            expect(self.page.locator('#codex-send')).to_be_enabled()
            self.page.reload()
            self.page.locator('[data-view-button="codex"]').click()
            expect(message.locator('h2')).to_have_text('Markdown answer')
            expect(message.locator('.code-block code')).to_have_text('<h1>Streaming code</h1>')
            self.screenshot('codex-markdown-light')
            self.page.locator('#theme-select').select_option('dark')
            expect(message).to_have_css('color', 'rgb(212, 212, 212)')
            self.screenshot('codex-markdown-dark')
            self.page.set_viewport_size({'width':390,'height':844})
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
            expect(self.page.locator('#codex-session-workspace')).to_be_visible()
            expect(self.page.locator('#codex-workspace-setup')).to_be_hidden()
            expect(self.page.locator('#codex-toggle-sessions')).to_be_visible()
            self.screenshot('codex-markdown-mobile')

    def test_codex_markdown_failure_can_retry_without_blocking_the_launcher(self):
        from pathlib import Path
        from fastllm_pytools.launcher_codex import CodexRuntime
        from test_launcher_agents import FAKE_CODEX
        self.page.clock.resume()
        self.runtime.codex = CodexRuntime(Path(self.temp.name) / 'codex')
        script = Path(self.temp.name) / 'codex.py'; script.write_text(FAKE_CODEX.replace("delta='Hello'", "delta='**Hello**'"))
        failure = lambda route: route.abort()
        self.page.route('**/assets/webui/markdown.js?codex=*', failure)
        with patch.object(self.runtime.codex, '_command', return_value=[sys.executable, str(script)]):
            self.page.locator('[data-view-button="codex"]').click()
            expect(self.page.locator('#codex-chat-error')).to_contain_text('Could not load Markdown')
            self.page.locator('#codex-prompt').fill('Keep chatting')
            self.page.locator('#codex-send').click()
            message = self.page.locator('#codex-messages .agentMessage .codex-markdown')
            expect(message).to_have_text('**Hello**')
            expect(message.locator('strong')).to_have_count(0)
            self.page.unroute('**/assets/webui/markdown.js?codex=*', failure)
            self.page.locator('[data-view-button="launch"]').click()
            self.page.locator('[data-view-button="codex"]').click()
            expect(message.locator('strong')).to_have_text('Hello')
            expect(self.page.locator('#codex-chat-error')).to_be_hidden()

    def assert_codex_preserves_pending_draft(self, new_thread, pending_method):
        from pathlib import Path
        from fastllm_pytools.launcher_codex import CodexRuntime
        from test_launcher_agents import FAKE_CODEX
        self.page.clock.resume()
        self.runtime.codex = CodexRuntime(Path(self.temp.name) / 'codex')
        script = Path(self.temp.name) / 'codex.py'; script.write_text(FAKE_CODEX)
        pending = []

        def intercept(route):
            if route.request.post_data_json.get('method') == pending_method:
                pending.append(route)
            else:
                route.continue_()

        with patch.object(self.runtime.codex, '_command', return_value=[sys.executable, str(script)]):
            self.page.locator('[data-view-button="codex"]').click()
            expect(self.page.locator('#codex-content')).to_be_visible()
            if not new_thread:
                self.page.locator('.codex-session[data-thread-id="thread-a"]').click()
            self.page.route('**/api/agents/codex/rpc', intercept)
            prompt = self.page.locator('#codex-prompt')
            prompt.fill('First submitted message')
            self.page.locator('#codex-send').click()
            deadline = time.monotonic() + 5
            while not pending and time.monotonic() < deadline:
                self.page.wait_for_timeout(20)
            self.assertEqual(len(pending), 1)
            prompt.fill('New unsent draft while waiting')
            pending[0].continue_()
            expect(self.page.locator('#codex-messages')).to_contain_text('Hello')
            expect(self.page.locator('#codex-send')).to_be_disabled()
            expect(prompt).to_have_value('New unsent draft while waiting')
            self.page.locator('.codex-approval').get_by_role('button', name='Allow once', exact=True).click()
            expect(self.page.locator('#codex-send')).to_be_enabled()
            expect(prompt).to_have_value('New unsent draft while waiting')
            saved = json.loads(self.page.evaluate('localStorage.getItem("ftllm.codex.sessions")'))
            self.assertEqual(saved['drafts']['thread-a'], 'New unsent draft while waiting')
            self.page.unroute('**/api/agents/codex/rpc', intercept)
            self.page.reload()
            self.page.locator('[data-view-button="codex"]').click()
            expect(prompt).to_have_value('New unsent draft while waiting')

    def test_codex_preserves_edits_during_turn_start(self):
        self.assert_codex_preserves_pending_draft(False, 'turn/start')

    def test_codex_preserves_edits_during_first_thread_creation(self):
        self.assert_codex_preserves_pending_draft(True, 'thread/start')

    def test_codex_preserves_edits_during_first_turn_start(self):
        self.assert_codex_preserves_pending_draft(True, 'turn/start')

    def test_opencode_embeds_from_two_launcher_addresses_at_the_same_time(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from fastllm_pytools.launcher_agent_proxy import AgentProxy

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(b'<textarea id="draft"></textarea>')

            def log_message(self, *args):
                pass

        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        proxy = AgentProxy(f'http://127.0.0.1:{upstream.server_port}', 'test-password', '127.0.0.1',
                           self.url.replace('127.0.0.1', 'localhost'))
        self.addCleanup(proxy.stop)
        proxy.start(threading.Event())
        self.runtime.opencode._proxy = proxy
        state = {'phase':'running', 'installed':True, 'sessionId':'model-a', 'url':proxy.url, 'error':''}
        with patch.object(self.runtime.opencode, 'state', side_effect=lambda:dict(state)):
            self.page.goto(self.url.replace('127.0.0.1', 'localhost') + '/?token=browser-key')
            self.page.locator('[data-view-button="opencode"]').click()
            first = self.page.frame_locator('#opencode-content iframe').locator('#draft')
            first.fill('First browser draft')
            other = self.context.new_page()
            other.on('pageerror', lambda error:self.errors.append(str(error)))
            other.on('console', lambda message:self.errors.append(message.text)
                     if 'Content Security Policy' in message.text else None)
            other.goto(self.url + '/?token=browser-key')
            other.locator('[data-view-button="opencode"]').click()
            second = other.frame_locator('#opencode-content iframe').locator('#draft')
            second.fill('Second browser draft')
            expect(self.page.locator('#opencode-content iframe')).to_have_attribute('src', re.compile(r'^http://localhost:'))
            expect(other.locator('#opencode-content iframe')).to_have_attribute('src', re.compile(r'^http://127\.0\.0\.1:'))
            self.page.wait_for_timeout(1200)
            expect(first).to_have_value('First browser draft')
            expect(second).to_have_value('Second browser draft')
            other.close()

    def test_opencode_theme_sync_keeps_the_frame_and_draft_and_rejects_other_senders(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from fastllm_pytools.launcher_agent_proxy import AgentProxy
        self.page.clock.resume()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/javascript' if self.path == '/native.js' else 'text/html')
                self.send_header('Content-Security-Policy', "script-src 'self'; style-src 'self' 'unsafe-inline'")
                self.end_headers()
                if self.path == '/native.js':
                    self.wfile.write(b'''window.bootID = Math.random();
                    function apply(value) { document.documentElement.dataset.colorScheme = value || 'light'; }
                    apply(localStorage.getItem('opencode-color-scheme'));
                    window.addEventListener('storage', event => {
                        if (event.key === 'opencode-color-scheme') apply(event.newValue);
                    });''')
                else:
                    self.wfile.write(b'''<!doctype html><html><head><script defer src="/native.js"></script>
                    <style>html,body{height:100%;margin:0}textarea{max-width:90%}</style></head>
                    <body><textarea aria-label="Message" id="draft"></textarea></body></html>''')

            def log_message(self, *args):
                pass

        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close); self.addCleanup(upstream.shutdown)
        proxy = AgentProxy(f'http://127.0.0.1:{upstream.server_port}', 'test-password', '127.0.0.1', self.url)
        self.addCleanup(proxy.stop); proxy.start(threading.Event())
        self.runtime.opencode._proxy = proxy
        state = {'phase':'running', 'installed':True, 'sessionId':'model-a', 'url':proxy.url, 'error':''}
        with patch.object(self.runtime.opencode, 'state', side_effect=lambda:dict(state)):
            self.page.locator('[data-view-button="opencode"]').click()
            frame = self.page.frame_locator('#opencode-content iframe')
            root = frame.locator('html')
            draft = frame.locator('#draft')
            expect(root).to_have_attribute('data-color-scheme', 'light')
            draft.fill('Keep this unsent message')
            boot = root.evaluate('() => window.bootID')
            for theme in ('dark', 'light', 'dark'):
                self.page.locator('#theme-select').select_option(theme)
                expect(root).to_have_attribute('data-color-scheme', theme)
                expect(draft).to_have_value('Keep this unsent message')
                self.assertEqual(root.evaluate('() => window.bootID'), boot)
            # Matching message data from another window/origin cannot change the frame.
            root.evaluate('''() => {
                const data = {type:'ftllm:opencode-appearance', theme:'light'};
                window.postMessage(data, location.origin);
                window.dispatchEvent(new MessageEvent('message', {
                    data, origin:'http://untrusted.example', source:parent
                }));
            }''')
            self.page.locator('[data-view-button="launch"]').click()
            self.page.locator('[data-view-button="opencode"]').click()
            expect(root).to_have_attribute('data-color-scheme', 'dark')
            expect(draft).to_have_value('Keep this unsent message')
            self.assertEqual(root.evaluate('() => window.bootID'), boot)
            # A fresh document also receives the current theme after native initialization.
            root.evaluate('() => location.reload()')
            expect(root).to_have_attribute('data-ftllm-theme', 'dark')
            expect(root).to_have_attribute('data-color-scheme', 'dark')
            self.page.set_viewport_size({'width':390, 'height':844})
            expect(draft).to_be_visible()
            self.assertTrue(root.evaluate('() => document.documentElement.scrollWidth <= innerWidth'))

    def test_harness_uses_each_browser_address_and_reconnects_after_external_restart(self):
        from pathlib import Path
        from fastllm_pytools.launcher_harness import HarnessRuntime
        from test_launcher_harness import FAKE_HARNESS
        self.page.clock.resume()
        runtime = self.runtime.harness = HarnessRuntime(Path(self.temp.name) / 'harness')
        script = Path(self.temp.name) / 'harness.py'; script.write_text(FAKE_HARNESS)
        localhost = self.url.replace('127.0.0.1', 'localhost')
        with patch.object(runtime, '_command', return_value=[sys.executable, str(script)]):
            self.page.goto(localhost + '/?token=browser-key')
            self.page.locator('[data-view-button="harness"]').click()
            first = self.page.frame_locator('#harness-content iframe').locator('#draft')
            first.fill('First browser draft')
            process = runtime._process
            other = self.context.new_page()
            self.addCleanup(other.close)
            other.on('pageerror', lambda error:self.errors.append(str(error)))
            other.on('console', lambda message:self.errors.append(message.text)
                     if 'Content Security Policy' in message.text else None)
            other.goto(self.url + '/?token=browser-key')
            other.locator('[data-view-button="harness"]').click()
            second = other.frame_locator('#harness-content iframe').locator('#draft')
            second.fill('Second browser draft')
            expect(self.page.locator('#harness-content iframe')).to_have_attribute('src', re.compile(r'^http://localhost:'))
            expect(other.locator('#harness-content iframe')).to_have_attribute('src', re.compile(r'^http://127\.0\.0\.1:'))
            expect(first).to_have_value('First browser draft')
            self.assertIs(runtime._process, process)
            # Both tabs miss the stopped/starting states, as can happen while
            # they are in the background and a different client restarts Harness.
            old = runtime.state_for_browser(localhost)
            old_second = runtime.state_for_browser(self.url)
            self.context.route('**/api/harness', lambda route:route.fulfill(
                json=old if route.request.url.startswith(localhost + '/') else old_second))
            runtime.stop()
            runtime.start(self.runtime._state, '', '127.0.0.1', localhost)
            deadline = time.monotonic() + 5
            while runtime.state()['phase'] not in {'running', 'failed'} and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(runtime.state()['phase'], 'running', runtime.state())
            self.assertIsNot(runtime._process, process)
            self.assertIsNotNone(process.poll())
            self.context.unroute('**/api/harness')
            new_first = runtime.state_for_browser(localhost)['url']
            new_second = runtime.state_for_browser(self.url)['url']
            self.assertNotEqual(new_first, old['url'])
            expect(self.page.locator('#harness-content iframe')).to_have_attribute('src', new_first)
            expect(other.locator('#harness-content iframe')).to_have_attribute('src', new_second)
            expect(first).to_have_value('')
            expect(second).to_have_value('')
            first.fill('Draft after reconnect')
            self.page.locator('[data-view-button="launch"]').click()
            self.page.locator('[data-view-button="harness"]').click()
            expect(first).to_have_value('Draft after reconnect')

    def test_harness_embeds_below_studio_and_retains_page_when_switching_views(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(b'<textarea id="draft"></textarea><p id="isolation"></p><script>'
                    b'try { parent.document.body; document.querySelector("p").textContent="unsafe"; }'
                    b'catch (_) { document.querySelector("p").textContent="isolated"; }</script>')

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        state = {'phase': 'stopped', 'sessionId': '', 'url': '', 'error': '', 'installed': True}

        def start(*args, **kwargs):
            self.assertFalse(kwargs['install'])
            state.update(phase='running', sessionId='model-a',
                         url=f'http://127.0.0.1:{server.server_port}/')
            return dict(state)

        def stop():
            state.update(phase='stopped', sessionId='', url='')
            return dict(state)

        with patch.object(self.runtime.harness, 'start', side_effect=start), \
                patch.object(self.runtime.harness, 'state', side_effect=lambda: dict(state)), \
                patch.object(self.runtime.harness, 'stop', side_effect=stop):
            nav = self.page.locator('.navigation > [data-view-button="harness"]')
            self.assertEqual(nav.evaluate('node => node.previousElementSibling.dataset.viewButton'), 'webui')
            nav.click()
            expect(self.page.locator('#harness-content iframe')).to_be_visible()
            embedded = self.page.frame_locator('#harness-content iframe')
            expect(embedded.locator('#isolation')).to_have_text('isolated')
            embedded.locator('#draft').fill('Harness draft stays here')
            self.page.locator('.navigation > [data-view-button="launch"]').click()
            nav.click()
            expect(embedded.locator('#draft')).to_have_value('Harness draft stays here')
            self.page.locator('#harness-stop').click()
            expect(self.page.locator('#harness-content iframe')).to_have_count(0)
            self.assertIsNotNone(self.runtime._process)

    def test_harness_installation_progress_survives_navigation_and_supports_cancel_retry(self):
        state = {'phase':'stopped', 'sessionId':'', 'url':'', 'error':'', 'installed':False}

        def start(*args, **kwargs):
            self.assertTrue(kwargs['install'])
            state.update(phase='installing', sessionId='model-a', stage='download',
                         done=1048576, total=4194304, error='')
            return dict(state)

        def stop():
            state.update(phase='stopped', stage='', sessionId='')
            return dict(state)

        with patch.object(self.runtime.harness, 'start', side_effect=start) as opening, \
                patch.object(self.runtime.harness, 'state', side_effect=lambda: dict(state)), \
                patch.object(self.runtime.harness, 'stop', side_effect=stop) as stopping:
            opening.assert_not_called()
            nav = self.page.locator('[data-view-button="harness"]')
            nav.click()
            expect(self.page.locator('#harness-retry')).to_have_text('Install and open Harness')
            expect(self.page.locator('#harness-install-note')).to_be_visible()
            self.page.clock.fast_forward(2100)
            opening.assert_not_called()
            expect(self.page.locator('#harness-progress')).to_be_hidden()
            self.page.locator('[data-view-button="launch"]').click()
            nav.click()
            self.page.clock.fast_forward(1100)
            opening.assert_not_called()
            self.page.reload()
            self.page.clock.install()
            self.page.locator('[data-view-button="harness"]').click()
            expect(self.page.locator('#harness-retry')).to_have_text('Install and open Harness')
            self.page.clock.fast_forward(1100)
            opening.assert_not_called()
            self.page.locator('#harness-retry').click()
            expect(self.page.locator('#harness-progress')).to_be_visible()
            expect(self.page.locator('#harness-progress-stage')).to_have_text('Downloading the runtime…')
            expect(self.page.locator('#harness-progress-detail')).to_have_text('1.0 / 4.0 MiB')
            expect(self.page.locator('#harness-progress-bar')).to_have_attribute('value', '1048576')
            expect(self.page.locator('#harness-install-note')).to_contain_text('450 MiB')
            self.screenshot('harness-installing')
            expect(self.page.locator('#harness-retry')).to_be_disabled()
            self.page.locator('[data-view-button="launch"]').click()
            nav.click()
            expect(self.page.locator('#harness-progress')).to_be_visible()
            opening.assert_called_once()
            state.update(stage='dependencies', done=12, total=0)
            self.page.clock.fast_forward(1100)
            expect(self.page.locator('#harness-progress-detail')).to_have_text('12 package requests completed')
            self.assertIsNone(self.page.locator('#harness-progress-bar').get_attribute('value'))
            self.page.locator('#harness-stop').click()
            expect(self.page.locator('#harness-progress')).to_be_hidden()
            expect(self.page.locator('#harness-retry')).to_have_text('Install and open Harness')
            stopping.assert_called_once()
            self.page.locator('[data-view-button="launch"]').click()
            nav.click()
            self.page.clock.fast_forward(1100)
            opening.assert_called_once()
            self.page.locator('#harness-retry').click()
            expect(self.page.locator('#harness-progress')).to_be_visible()
            self.assertEqual(opening.call_count, 2)
            state.update(phase='failed', error='Download failed')
            self.page.clock.fast_forward(1100)
            expect(self.page.locator('#harness-error')).to_have_text('Download failed')
            expect(self.page.locator('#harness-retry')).to_have_text('Retry')
            self.page.locator('#harness-retry').click()
            expect(self.page.locator('#harness-progress')).to_be_visible()
            self.assertEqual(opening.call_count, 3)
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

    def test_harness_is_managed_by_customizer_without_implicit_installation(self):
        state = {'phase':'stopped', 'sessionId':'', 'url':'', 'error':'', 'installed':False}

        def start(*args, **kwargs):
            self.assertTrue(kwargs['install'])
            state.update(phase='installing', sessionId='model-a', stage='dependencies', done=12, total=0)
            return dict(state)

        def stop():
            state.update(phase='stopped', sessionId='', stage='')
            return dict(state)

        with patch.object(self.runtime.harness, 'start', side_effect=start) as opening, \
                patch.object(self.runtime.harness, 'state', side_effect=lambda: dict(state)), \
                patch.object(self.runtime.harness, 'stop', side_effect=stop) as stopping:
            self.page.locator('body > .app-shell .navigation > .plugin-manager-button').click()
            manager = self.page.locator('body > .plugin-manager')
            manager.locator('.customizer-library > summary').click()
            row = manager.locator('.plugin-row[data-plugin-id="harness"]')
            expect(row.locator('.plugin-runtime-status')).to_have_text('未安装，需手动点击安装')
            self.screenshot('harness-plugin-manager')
            row.get_by_role('button', name='管理', exact=True).click()
            runtime_manager = self.page.locator('.plugin-runtime-manager')
            expect(runtime_manager).to_be_visible()
            expect(runtime_manager.locator('[data-agent]')).to_have_value('harness')
            opening.assert_not_called()
            runtime_manager.locator('[data-close]').click()
            manager.locator('.plugin-heading > button').click()
            self.page.locator('[data-view-button="harness"]').click()
            expect(self.page.locator('#harness-retry')).to_have_text('Install and open Harness')
            opening.assert_not_called()
            self.page.locator('#harness-retry').click()
            expect(self.page.locator('#harness-progress')).to_be_visible()
            self.page.locator('body > .app-shell .navigation > .plugin-manager-button').click()
            expect(row.locator('.plugin-runtime-status')).to_have_text('安装中')
            row.get_by_role('button', name='停用', exact=True).click()
            expect(row.locator('.plugin-runtime-status')).to_have_text('已停用')
            stopping.assert_called_once()
            expect(row.get_by_role('button', name='管理', exact=True)).to_be_enabled()
            row.get_by_role('button', name='启用', exact=True).click()
            expect(row.locator('.plugin-runtime-status')).to_have_text('未安装，需手动点击安装')
            opening.assert_called_once()
            row.get_by_role('button', name='管理', exact=True).click()
            expect(runtime_manager).to_be_visible()
            runtime_manager.locator('[data-close]').click()
            manager.locator('.plugin-heading > button').click()
            expect(self.page.locator('#harness-retry')).to_be_enabled()
            expect(self.page.locator('#harness-progress')).to_be_hidden()
            opening.assert_called_once()
            self.assertIsNotNone(self.runtime._process)

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
            ['launch', 'download', 'logs', 'hardware', 'webui', 'harness', 'opencode', 'codex', 'claude'])
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
        backdrop = pane.locator('#sidebarBackdrop')
        backdrop_bounds = backdrop.bounding_box()
        backdrop.click(position={'x': backdrop_bounds['width'] - 10, 'y': backdrop_bounds['height'] / 2})
        expect(pane.locator('#sidebar')).not_to_have_class('sidebar open')
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
