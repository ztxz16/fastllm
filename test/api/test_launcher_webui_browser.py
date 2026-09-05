"""Optional Chromium regression tests; requires Playwright and its browser."""
import argparse
import base64
import json
import os
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

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = LauncherRuntime(os.path.join(self.temp.name, 'profiles.json'),
                                       webui_history_dir=os.path.join(self.temp.name, 'history'))
        self.addCleanup(self.close_runtime)
        self.runtime._process = SimpleNamespace(poll=lambda: None)
        self.runtime._state.update(command='server', phase='running', ready=True,
                                   sessionId='model-a', modelName='browser-model',
                                   endpoint='http://127.0.0.1:19001')
        app = create_launcher_app(self.runtime, 'browser-key')
        args = add_webui_args(argparse.ArgumentParser()).parse_args([])
        args.api_model = 'standalone-model'
        args.agent_runtime = 'builtin'
        args.history_dir = os.path.join(self.temp.name, 'standalone-history')
        standalone = create_app(args)
        self.standalone = standalone.state.runtime
        self.addCleanup(self.standalone.close)
        app.mount('/standalone', standalone)
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

    def test_network_error_can_retry_without_reloading_launcher(self):
        self.assert_resource_failure_recovers('**/assets/webui/template.html')

    def test_component_module_failure_can_retry(self):
        self.assert_resource_failure_recovers('**/assets/webui/app.js')

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

    def test_standalone_webui_uses_the_same_component(self):
        self.page.goto(self.url + '/standalone/')
        pane = self.page.locator('#webui-root')
        expect(pane.locator('#prompt')).to_be_visible()
        expect(pane.locator('.brand')).to_be_visible()
        expect(pane.locator('#languageButton')).to_be_visible()
        with patch.object(self.standalone.api_client, 'stream',
                          side_effect=lambda *a, **k: iter([('Shared standalone reply', '')])):
            pane.locator('#prompt').fill('Hello standalone')
            pane.locator('#sendButton').click()
            expect(pane.locator('.message.assistant')).to_have_count(1)
            expect(pane.locator('#stopButton')).to_be_hidden()
        self.assertIn('/standalone/?chat=', self.page.url)
        self.screenshot('standalone-webui')
        self.page.reload()
        expect(pane.locator('.message.assistant')).to_have_count(1)


if __name__ == '__main__':
    unittest.main()
