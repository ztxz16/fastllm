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
            yield {'type': 'text_delta', 'text': 'Project inspected.'}
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
        expect(pane.locator('#agentUnavailable')).to_contain_text('Install ftllm-agent-runtime')
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
