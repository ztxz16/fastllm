"""Native HTTP persistence/assets regression; no checkpoint, GPU or Torch required.

python3 test/minimax_h3_image/test_server.py --binary /path/to/minimax_h3_image_server
"""
import argparse
import base64
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--binary', default=os.environ.get('FTLLM_H3_IMAGE_SERVER', ''))
args, remaining = parser.parse_known_args()
BINARY = str(Path(args.binary).resolve()) if args.binary else ''
ROOT = Path(__file__).resolve().parents[2]
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5WQAAAAASUVORK5CYII=')


@unittest.skipUnless(BINARY, 'Set --binary or FTLLM_H3_IMAGE_SERVER to run native HTTP tests')
class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ftllm-image-http-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.output = cls.directory / 'output'
        cls.output.mkdir()
        # Deliberately non-chronological directory insertion order, more than the history cap.
        order = list(range(140))
        random.Random(42).shuffle(order)
        cls.expected = {f'image_{1700000000000000+i}_1' for i in range(12, 140)}
        for i in order:
            cls.fixture(i, 1700000000 + i)
        cls.fixture(1000, 1800000000, status='failed')
        cls.fixture(1001, 1800000001, image=False)
        (cls.output / 'image_999_1.json').write_text('{incomplete')
        # Test the wheel asset layout as well as the source fallback.
        cls.assets = cls.directory / 'h3_image_assets'
        cls.assets.mkdir()
        for name in ('demo.html', 'openapi.json'):
            shutil.copy2(ROOT / 'tools/minimax_h3_image' / name, cls.assets / name)
        for name in ('image-workspace.js', 'image-workspace.css'):
            shutil.copy2(ROOT / 'tools/fastllm_pytools/ui_plugins/studio' / name, cls.assets / name)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        cls.base = f'http://127.0.0.1:{port}'
        cls.log = (cls.directory / 'server.log').open('w')
        cls.addClassCleanup(cls.log.close)
        cls.process = subprocess.Popen([BINARY, '--model', str(cls.directory / 'missing-model'),
            '--host', '127.0.0.1', '--port', str(port), '--output', str(cls.output),
            '--demo', str(cls.assets / 'demo.html'), '--api-key', 'test-image-key',
            '--gpu-memory-limit', '4.5', '--host-memory-limit', '2'],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=cls.log, stderr=subprocess.STDOUT)
        cls.addClassCleanup(cls.stop)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                raise RuntimeError((cls.directory / 'server.log').read_text())
            try:
                if cls.request('/v1/images/jobs')[0] == 200:
                    return
            except OSError:
                time.sleep(.05)
        raise RuntimeError('HTTP test server failed to start')

    @classmethod
    def stop(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait()

    @classmethod
    def fixture(cls, i, created, status='completed', image=True):
        id_ = f'image_{1700000000000000+i}_1'
        body = dict(id=id_, status=status, created_at=created, prompt='saved result',
                    size='2048x2048', seed=i, min_generation_area=4194304,
                    references=[dict(label='Picture 1', width=64, height=96)])
        (cls.output / (id_ + '.json')).write_text(json.dumps(body))
        if image:
            (cls.output / (id_ + '.png')).write_bytes(PNG)

    @classmethod
    def request(cls, path, token='test-image-key'):
        headers = {} if token is None else {'Authorization': 'Bearer ' + token}
        try:
            response = urllib.request.urlopen(urllib.request.Request(cls.base + path, headers=headers), timeout=3)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, response.read(), response.headers

    def test_memory_policy_health_without_gpu(self):
        code, body, _ = self.request('/health', None)
        self.assertEqual(code, 503)
        data = json.loads(body)
        self.assertEqual(data['gpu_memory_limit_gib'], 4.5)
        self.assertEqual(data['host_memory_limit_gib'], 2)
        self.assertFalse(data['gpu_memory_reserved'])
        self.assertIn('host_weight_cache_gib', data)
        self.assertIn('layer_streaming', data)
        self.assertEqual(data['tensor_parallel_size'], 1)
        self.assertEqual(data['sequence_parallel_size'], 1)
        self.assertEqual(data['dit_parallel_mode'], 'tp')

    def test_reject_invalid_memory_budgets_before_startup(self):
        for flag, values in (('--gpu-memory-limit', ('1', '1.0000000001', '-1', 'nan', 'inf', '4GiB', '1048577')),
                             ('--host-memory-limit', ('1.99', '-1', 'nan', 'inf', '2GiB', '1048577'))):
            for value in values:
                result = subprocess.run([BINARY, '--model', '/unused', flag, value],
                                        capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0, (flag, value))
                self.assertIn(flag.lstrip('-'), result.stderr)

    def test_restart_keeps_newest_128_complete_results(self):
        code, data, _ = self.request('/v1/images/jobs')
        self.assertEqual(code, 200)
        jobs = json.loads(data)['data']
        self.assertEqual({j['id'] for j in jobs}, self.expected)
        self.assertTrue(all(j['reference_count'] == 1 and j['canvas_size'] == '2048x2048' for j in jobs))
        newest = 'image_1700000000000139_1'
        code, content, _ = self.request('/v1/images/jobs/' + newest + '/content')
        self.assertEqual((code, content), (200, PNG))
        # Pruning metadata in memory must never remove archived PNGs from disk.
        self.assertEqual(len(list(self.output.glob('*.png'))), 141)

    def test_authentication_and_missing_content(self):
        for token in (None, 'wrong-key'):
            self.assertEqual(self.request('/v1/images/jobs', token)[0], 401)
            self.assertEqual(self.request('/v1/models', token)[0], 401)
        self.assertEqual(self.request('/v1/images/jobs/image_0_0/content')[0], 404)

    def test_bundled_demo_assets_are_public_and_have_correct_mime(self):
        for path, mime in [('/demo', 'text/html'), ('/openapi.json', 'application/json'),
                           ('/assets/image-workspace.js', 'application/javascript'),
                           ('/assets/image-workspace.css', 'text/css')]:
            code, content, headers = self.request(path, None)
            self.assertEqual(code, 200, path)
            self.assertEqual(headers.get_content_type(), mime, path)
            self.assertTrue(content)
        schema = json.loads(self.request('/openapi.json', None)[1])
        self.assertIn('/v1/images/edits', schema['paths'])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]] + remaining)
