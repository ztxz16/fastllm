"""Service failure/queue regressions using only the Python standard library."""
import base64
import http.client
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('lingbot_server',
    Path(__file__).resolve().parents[2]/'tools/lingbot_world/server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class FakeNative:
    def __init__(self, *args):
        self.worlds = {}
        self.next_id = 0
        self.failure = None

    def create(self, options, progress):
        self.next_id += 1
        self.worlds[self.next_id] = 0
        return self.next_id

    def step(self, handle, options, progress):
        if self.failure:
            raise RuntimeError(self.failure)
        self.worlds[handle] += 1
        return [b'frame0', b'frame1']

    def delete(self, handle):
        del self.worlds[handle]


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.native = FakeNative()
        with patch.object(server, 'Native', return_value=self.native):
            self.service = server.Service(SimpleNamespace(output=self.directory.name,
                library='unused', model='unused', device=0, vae_decode_precision='bf16', max_worlds=2))
            deadline = time.monotonic()+5
            while self.service.status == 'loading' and time.monotonic() < deadline:
                time.sleep(.01)
        self.assertEqual(self.service.status, 'ready')

    def create(self):
        job = self.service.submit(dict(prompt='A garden.', width=64, height=64,
            image='data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\n').decode()))
        return self.wait(job)

    def wait(self, job):
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            current = self.service.get('jobs', job['id'])
            if current['status'] in ('completed', 'failed'):
                return current
            time.sleep(.01)
        self.fail('worker did not finish')

    def test_inference_failure_preserves_existing_world(self):
        first = self.create()
        self.native.failure = 'inference cancelled'
        job = self.wait(self.service.submit({'action': 'forward'}, first['world_id']))
        self.assertEqual(job['status'], 'failed')
        self.assertFalse(job['world_deleted'])
        world = self.service.get('worlds', first['world_id'])
        self.assertEqual(world['segments'], 1)
        self.assertFalse(world['pending'])
        self.native.failure = None
        job = self.wait(self.service.submit({'action': 'stay'}, first['world_id']))
        self.assertEqual(job['result']['segment'], 1)

    def test_artifact_failure_removes_advanced_world(self):
        first = self.create()
        original = Path.write_bytes
        calls = 0
        def fail_second(file, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                original(file, b'partial')
                raise OSError('disk full')
            return original(file, data)
        with patch.object(Path, 'write_bytes', fail_second):
            job = self.wait(self.service.submit({'action': 'forward'}, first['world_id']))
        self.assertEqual(job['status'], 'failed')
        self.assertTrue(job['world_deleted'])
        self.assertIn('disk full', job['error'])
        self.assertNotIn(first['world_id'], self.service.worlds)
        self.assertFalse(self.native.worlds)
        self.assertFalse(list(Path(self.directory.name).glob(job['id']+'*')))
        self.assertEqual(len(list(Path(self.directory.name).glob(first['id']+'*'))), 2)
        self.assertEqual(self.service.status, 'ready')

    def test_failed_create_does_not_consume_world_slot(self):
        self.native.failure = 'invalid model input'
        job = self.create()
        self.assertEqual(job['status'], 'failed')
        self.assertTrue(job['world_deleted'])
        self.assertFalse(self.service.worlds)
        self.assertFalse(self.native.worlds)

    def test_mp4_delivery_and_png_fallback(self):
        world = dict(id='world_test', segments=1, width=64, height=64)
        rgb = bytes((18, 83, 220)) * (64*64*2)
        def encode(ffmpeg, frames, job, path):
            self.assertEqual(frames, rgb)
            self.assertEqual((job['width'], job['height'], job['fps']), (64, 64, 16))
            path.write_bytes(b'clip')
            return True
        with patch.object(server, 'encode_clip', side_effect=encode):
            result = self.service.save_segment('job_test', world, rgb, [])
        self.assertEqual((result['format'], result['frame_count']), ('mp4', 2))
        self.assertEqual((self.service.output / Path(result['url']).name).read_bytes(), b'clip')
        self.assertNotIn('frames', result)
        with patch.object(server, 'encode_clip', return_value=False):
            result = self.service.save_segment('job_fallback', world, rgb, [])
        self.assertEqual((result['format'], result['frame_count']), ('png_frames', 2))
        self.assertEqual(len(result['frames']), 2)
        self.assertEqual((self.service.output / Path(result['frames'][0]).name).read_bytes(),
                         server.png_rgb(rgb[:64*64*3], 64, 64))

    def test_mp4_byte_ranges(self):
        filename = 'job_' + 'a' * 24 + '.mp4'
        (self.service.output / filename).write_bytes(b'0123456789')
        self.service.args.api_key = ''
        handler = server.handler(self.service, Path('unused'))
        with server.ThreadingHTTPServer(('127.0.0.1', 0), handler) as listener:
            worker = threading.Thread(target=listener.serve_forever, daemon=True)
            worker.start()
            try:
                for requested, status, span, data in [
                    (None, 200, None, b'0123456789'),
                    ('bytes=2-4', 206, 'bytes 2-4/10', b'234'),
                    ('bytes=-3', 206, 'bytes 7-9/10', b'789'),
                    ('bytes=8-', 206, 'bytes 8-9/10', b'89'),
                    ('bytes=10-', 416, 'bytes */10', b''),
                ]:
                    with self.subTest(requested=requested):
                        conn = http.client.HTTPConnection(*listener.server_address, timeout=2)
                        try:
                            headers = {'Range': requested} if requested else {}
                            conn.request('GET', '/v1/worlds/files/' + filename, headers=headers)
                            response = conn.getresponse()
                            self.assertEqual((response.status, response.getheader('Content-Range'), response.read()),
                                             (status, span, data))
                        finally:
                            conn.close()
            finally:
                listener.shutdown()
                worker.join()

    def test_invalid_action_types(self):
        for action in [[], {}, 1, None]:
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'camera action'):
                server.options({'action': action}, False)

    def test_prompt_cannot_be_truncated_at_native_boundary(self):
        with self.assertRaisesRegex(ValueError, 'without NUL'):
            server.options({'prompt': 'A room.\0Ignored suffix.'}, False)


if __name__ == '__main__':
    unittest.main()
