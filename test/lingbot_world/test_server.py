"""Service failure/queue regressions using only the Python standard library."""
import base64
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
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
                library='unused', model='unused', device=0, max_worlds=2))
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

    def test_invalid_action_types(self):
        for action in [[], {}, 1, None]:
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'camera action'):
                server.options({'action': action}, False)

    def test_prompt_cannot_be_truncated_at_native_boundary(self):
        with self.assertRaisesRegex(ValueError, 'without NUL'):
            server.options({'prompt': 'A room.\0Ignored suffix.'}, False)


if __name__ == '__main__':
    unittest.main()
