"""Optional Chromium regression tests; requires Playwright and its browser."""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tools')))
from fastllm_pytools.launcher import LauncherRuntime, create_launcher_app

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
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
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
        self.page.goto(f'http://127.0.0.1:{port}/?token=browser-key')
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
        expect(self.page.locator('#webui-frame')).to_be_visible()
        expect(self.page.frame_locator('#webui-frame').locator('#prompt')).to_be_visible()
        # Successful loads must clear the previous attempt's timeout.
        self.page.clock.fast_forward(35000)
        expect(self.page.locator('#webui-frame')).to_be_visible()
        expect(self.page.locator('#webui-retry')).to_be_hidden()

    def test_network_error_can_retry_without_reloading_launcher(self):
        pattern = '**/webui/model-a/'
        self.page.route(pattern, lambda route: route.abort('connectionfailed'))
        self.page.locator('#open-webui').click()
        expect(self.page.locator('#webui-retry')).to_be_visible()
        expect(self.page.locator('#webui-status')).to_have_text('Unable to load WebUI. Try reopening it.')
        self.page.unroute(pattern)
        self.page.locator('#back-to-management').click()
        self.page.locator('#open-webui').click()
        self.page.locator('#webui-retry').click()
        self.assert_loaded()

    def assert_timeout_recovers(self, pattern, response):
        pending = []
        self.page.route(pattern, lambda route: pending.append(route))
        with self.page.expect_request(pattern):
            self.page.locator('#open-webui').click()
        self.page.clock.fast_forward(30001)
        expect(self.page.locator('#webui-retry')).to_be_visible()
        expect(self.page.locator('#webui-status')).to_have_text('WebUI loading timed out. Try reopening it.')
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

    def test_iframe_timeout_can_retry(self):
        self.assert_timeout_recovers('**/webui/model-a/', {
            'content_type': 'text/html', 'body': '<p>stale document</p>'})

    def test_model_switch_clears_the_old_loading_attempt(self):
        pending = []
        pattern = '**/webui/model-a/'
        self.page.route(pattern, lambda route: pending.append(route))
        with self.page.expect_request(pattern):
            self.page.locator('#open-webui').click()
        with self.runtime._lock:
            self.runtime._close_webui_locked()
            self.runtime._state.update(sessionId='model-b', modelName='next-model')
        self.page.clock.fast_forward(2500)
        expect(self.page.locator('#webui-frame')).to_have_attribute('src', '/webui/model-b/')
        self.assert_loaded()
        self.assertEqual(len(pending), 1)
        pending[0].fulfill(status=200, content_type='text/html', body='<p>old model</p>')
        expect(self.page.frame_locator('#webui-frame').locator('#modelName')).to_have_text('next-model')


if __name__ == '__main__':
    unittest.main()
