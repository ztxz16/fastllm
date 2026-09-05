"""Integration checks for the original WebUI mounted inside Launcher."""
import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tools')))
from fastapi.testclient import TestClient
from starlette.requests import Request
from fastllm_pytools.launcher import LauncherRuntime, create_launcher_app


class LauncherWebUITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = LauncherRuntime(os.path.join(self.temp.name, 'profiles.json'),
                                       webui_history_dir=os.path.join(self.temp.name, 'history'))
        self.runtime._process = SimpleNamespace(poll=lambda: None)
        self.runtime._state.update(command='server', phase='running', ready=True,
                                   sessionId='model-a', modelName='active-model', endpoint='http://127.0.0.1:19001')
        self.runtime._service_api_key = 'private-model-key'
        self.app = create_launcher_app(self.runtime, 'launcher-key')
        self.client = TestClient(self.app)
        self.headers = {'X-FTLLM-Launcher-Token': 'launcher-key'}
        self.base = '/webui/model-a'

    def tearDown(self):
        self.client.close()
        self.runtime._process = None
        self.runtime.close()
        self.temp.cleanup()

    def open(self):
        response = self.client.post('/api/webui/open', headers=self.headers,
                                    json={'sessionId': self.runtime._state['sessionId']})
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def test_authentication_covers_page_assets_api_and_downloads(self):
        for path in ('/', '/assets/webui_locales.js', '/api/config', '/api/conversations/x/attachments/y'):
            self.assertEqual(self.client.get(self.base + path).status_code, 403)
        self.assertEqual(self.client.post('/api/webui/open', json={'sessionId': 'model-a'}).status_code, 403)
        response = self.open()
        cookie = response.headers['set-cookie'].lower()
        for flag in ('httponly', 'samesite=strict', 'path=/webui/'):
            self.assertIn(flag, cookie)
        self.assertNotIn('private-model-key', cookie)
        self.assertNotIn('launcher-key', cookie)
        self.assertEqual(self.client.get('/api/runtime').status_code, 403)
        self.assertEqual(self.client.post(self.base + '/api/conversations', json={},
                                         headers={'Origin': 'https://untrusted.example'}).status_code, 403)
        self.assertEqual(self.client.post(self.base + '/api/conversations', json={},
                                         headers={'Origin': 'http://testserver'}).status_code, 200)

    def test_both_entry_points_use_shared_component_assets(self):
        self.open()
        page = self.client.get(self.base + '/')
        self.assertEqual(page.status_code, 200)
        self.assertIn('id="webui-root"', page.text)
        self.assertIn(f'data-base-path="{self.base}"', page.text)
        self.assertIn(f'src="{self.base}/assets/webui/standalone.js"', page.text)
        self.assertNotIn('__WEBUI_BASE_PATH__', page.text)
        self.assertNotIn("'unsafe-inline'", page.headers['content-security-policy'])
        self.assertIn("frame-ancestors 'self'", page.headers['content-security-policy'])
        self.assertEqual(page.headers['cache-control'], 'no-store')
        for asset in ('app.js', 'styles.css', 'template.html', 'standalone.js'):
            child = self.client.get(self.base + '/assets/webui/' + asset)
            parent = self.client.get('/assets/webui/' + asset)
            self.assertEqual(child.status_code, 200)
            self.assertEqual(parent.status_code, 200)
            self.assertEqual(child.content, parent.content)
        template = self.client.get('/assets/webui/template.html').text
        for control in ('conversationList', 'agentButton', 'attachButton'):
            self.assertIn(f'id="{control}"', template)
        for asset in ('webui_locales.js', 'fastllm_icon.svg'):
            self.assertEqual(self.client.get(self.base + '/assets/' + asset).status_code, 200)
        parent = self.client.get('/')
        self.assertEqual(parent.headers['x-frame-options'], 'DENY')
        self.assertNotIn("'unsafe-inline'", parent.headers['content-security-policy'])
        self.assertNotIn('<iframe', parent.text)
        self.assertIn('id="webui-content"', parent.text)
        self.assertIn('<h1>模型管理</h1>', parent.text)

    def test_active_model_and_credentials_are_snapshotted_on_server_side(self):
        self.open()
        webui = self.runtime._webui_app.state.runtime
        self.assertEqual(webui.api_client.model_name, 'active-model')
        self.assertEqual(webui.api_client.base_url, 'http://127.0.0.1:19001/v1')
        self.assertEqual(webui.api_client.api_key, 'private-model-key')
        config = self.client.get(self.base + '/api/config')
        self.assertNotIn('private-model-key', config.text)
        self.assertTrue(config.json()['workspace_agent_enabled'])
        first_app = self.runtime._webui_app
        self.open()
        self.assertIs(self.runtime._webui_app, first_app)

    def test_stopped_webui_and_stale_sessions_are_rejected(self):
        self.open()
        for change in ({'ready': False}, {'phase': 'starting'}, {'command': 'webui'}, {'sessionId': 'model-b'}):
            with self.subTest(change=change), patch.dict(self.runtime._state, change):
                self.assertEqual(self.client.get(self.base + '/api/config').status_code, 409)
                self.assertEqual(self.client.post('/api/webui/open', headers=self.headers,
                                                 json={'sessionId': 'model-a'}).status_code, 400)
        self.runtime._process = None
        self.assertEqual(self.client.get(self.base + '/').status_code, 409)

    def test_remote_launcher_preserves_webui_workspace_policy(self):
        remote = TestClient(create_launcher_app(self.runtime, 'launcher-key', launcher_host='0.0.0.0'))
        try:
            response = remote.post('/api/webui/open', headers=self.headers, json={'sessionId': 'model-a'})
            self.assertEqual(response.status_code, 200)
            config = remote.get(self.base + '/api/config').json()
            self.assertFalse(config['workspace_agent_enabled'])
            self.assertEqual(remote.get(self.base + '/api/agent/directories').status_code, 403)
        finally:
            remote.close()

    def test_saved_history_survives_app_recreation_and_model_switch(self):
        self.open()
        record = self.client.post(self.base + '/api/conversations', json={'title': 'Saved chat'}).json()
        webui = self.runtime._webui_app.state.runtime
        control = webui.begin_generation(record['id'])
        self.runtime._state.update(sessionId='model-b', modelName='next-model')
        self.open()
        self.assertTrue(control.cancelled)
        self.assertEqual(self.client.get(self.base + '/api/config').status_code, 409)
        self.base = '/webui/model-b'
        saved = self.client.get(self.base + '/api/conversations/' + record['id']).json()
        self.assertEqual(saved['title'], 'Saved chat')
        self.assertEqual(self.client.get(self.base + '/api/config').json()['model'], 'next-model')
        self.runtime._process = None
        self.runtime.stop()
        self.assertIsNone(self.runtime._webui_app)

    def test_stop_cancels_every_active_webui_generation(self):
        self.open()
        webui = self.runtime._webui_app.state.runtime
        controls = [webui.begin_generation(name) for name in ('one', 'two')]
        self.runtime._process = None
        self.runtime.stop()
        self.assertTrue(all(control.cancelled for control in controls))
        self.assertIsNone(self.runtime._webui_app)
        with self.assertRaisesRegex(RuntimeError, '重新打开 WebUI'):
            webui.begin_generation('late-request')

    def switch_model(self):
        self.runtime._process = None
        self.runtime.stop()
        self.runtime._process = SimpleNamespace(poll=lambda: None)
        self.runtime._state.update(command='server', phase='running', ready=True,
                                   sessionId='model-b', modelName='next-model',
                                   endpoint='http://127.0.0.1:19001')
        self.open()

    def assert_old_request_cannot_overwrite(self, method='POST', suffix='/chat',
                                          payload=None, pause_at='json'):
        self.open()
        record = self.client.post(self.base + '/api/conversations', json={}).json()
        path = '/api/conversations/' + record['id']
        old_runtime = self.runtime._webui_app.state.runtime
        entered, release = threading.Event(), threading.Event()
        url = self.base + path + suffix
        kwargs = {'json': payload or {'prompt': 'old pending prompt'}}

        def wait_for_release():
            if not release.wait(5):
                raise AssertionError('Timed out waiting for the replacement chat')

        if pause_at in ('json', 'body'):
            original = getattr(Request, pause_at)

            async def delayed_body(request):
                if request.url.path == url:
                    entered.set()
                    await asyncio.to_thread(wait_for_release)
                return await original(request)

            pause = patch.object(Request, pause_at, delayed_body)
        elif pause_at == 'registered':
            original = old_runtime.begin_generation

            def delayed_registration(conversation_id):
                control = original(conversation_id)
                entered.set()
                wait_for_release()
                return control

            pause = patch.object(old_runtime, 'begin_generation', delayed_registration)
        else:
            def delayed_stream(*args, **kwargs):
                yield ('old partial response', '')
                entered.set()
                wait_for_release()
                yield ('old late response', '')

            pause = patch.object(old_runtime.api_client, 'stream', delayed_stream)

        if pause_at == 'body':
            kwargs = {'content': b'old upload', 'headers': {
                'X-Filename': 'old.txt', 'Content-Type': 'text/plain'}}

        with pause, ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.client.request, method, url, **kwargs)
            try:
                self.assertTrue(entered.wait(5), 'Old request never reached the pause')
                self.switch_model()
                new_runtime = self.runtime._webui_app.state.runtime
                with patch.object(new_runtime.api_client, 'stream',
                                  return_value=iter([('new response', '')])):
                    response = self.client.post('/webui/model-b' + path + '/chat',
                                                json={'prompt': 'new prompt to preserve'})
                self.assertEqual(response.status_code, 200, response.text)
                saved = self.client.get('/webui/model-b' + path).json()
            finally:
                release.set()
            response = pending.result(timeout=5)
        if pause_at == 'stream':
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.text.splitlines()[-1])['type'], 'cancelled')
        else:
            self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.client.get('/webui/model-b' + path).json(), saved)
        self.assertFalse(old_runtime.is_generating(record['id']))
        if pause_at == 'body':
            self.assertFalse(list(old_runtime.store.upload_root.rglob('*.txt')))

    def test_pending_chat_is_rejected_after_model_switch(self):
        self.assert_old_request_cannot_overwrite()

    def test_registered_chat_cannot_save_after_model_switch(self):
        self.assert_old_request_cannot_overwrite(pause_at='registered')

    def test_late_stream_cancellation_cannot_overwrite_new_history(self):
        self.assert_old_request_cannot_overwrite(pause_at='stream')

    def test_pending_settings_cannot_overwrite_new_history(self):
        self.assert_old_request_cannot_overwrite(method='PATCH', suffix='',
                                               payload={'title': 'old title'})

    def test_pending_upload_cannot_write_after_model_switch(self):
        self.assert_old_request_cannot_overwrite(suffix='/attachments', pause_at='body')

    def test_attachment_upload_and_download_use_original_size_limit(self):
        self.open()
        record = self.client.post(self.base + '/api/conversations', json={}).json()
        url = self.base + '/api/conversations/' + record['id'] + '/attachments'
        content = b'test document\n' * 100000  # Larger than Launcher's 1 MiB JSON request limit.
        response = self.client.post(url, content=content,
                                    headers={'X-Filename': 'notes.txt', 'Content-Type': 'text/plain'})
        self.assertEqual(response.status_code, 200, response.text)
        attachment = response.json()
        download = self.client.get(self.base + attachment['url'])
        self.assertEqual(download.content, content)
        self.assertIn("script-src 'none'", download.headers['content-security-policy'])
        self.assertIn('sandbox', download.headers['content-security-policy'])

    def test_chat_reuses_webui_streaming_history_and_cancellation_route(self):
        self.open()
        record = self.client.post(self.base + '/api/conversations', json={}).json()
        path = self.base + '/api/conversations/' + record['id']
        webui = self.runtime._webui_app.state.runtime
        for prompt in ('First', 'Next'):
            with patch.object(webui.api_client, 'stream', return_value=iter([('**Hello**', 'Thinking')])) as stream:
                response = self.client.post(path + '/chat', json={'prompt': prompt, 'attachments': []})
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line) for line in response.text.splitlines()]
            self.assertEqual(events[-1]['type'], 'done')
            self.assertIn('**Hello**', response.text)
        messages = stream.call_args.args[0]
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'user'])
        self.assertEqual(len(self.client.get(path).json()['messages']), 4)
        self.assertFalse(self.client.post(path + '/cancel').json()['cancelled'])


if __name__ == '__main__':
    unittest.main()
