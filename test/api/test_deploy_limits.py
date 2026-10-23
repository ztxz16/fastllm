"""Exercise real proxy sockets and inherited Linux limits without loading a model."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

from test_deploy import TOOLS, configuration, free_ports
from fastllm_pytools.deploy.config import resolve_config
from fastllm_pytools.deploy.process import configure_nofile, owned


class LimitTests(unittest.TestCase):
    def config(self, **extra):
        return {'replicas': [{'max_inflight': 380}, {'max_inflight': 380}],
                'router': {'queue_limit_per_replica': 380}, **extra}

    def test_config_accepts_optional_positive_integer(self):
        data = configuration(Path('/tmp/deploy-limit-test'))
        for value in (0, -1, True, '65536', []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_config({**data, 'nofile': value}, Path('/tmp/deployment.json'))
        resolved = resolve_config({**data, 'nofile': 65536}, Path('/tmp/deployment.json'))
        self.assertEqual(resolved['nofile'], 65536)
        self.assertNotIn('nofile', resolved['serve_args'])

    def test_budget_checks_without_changing_caller_then_applies_to_supervisor(self):
        with patch('resource.getrlimit', return_value=(1024, 1048576)), patch('resource.setrlimit') as setter:
            result = configure_nofile(self.config())
            self.assertEqual(result['required'], 2536)
            self.assertEqual(result['soft'], 2536)
            setter.assert_not_called()
            configure_nofile(self.config(), apply=True)
            import resource
            setter.assert_called_once_with(resource.RLIMIT_NOFILE, (2536, 1048576))

    def test_explicit_target_and_existing_higher_limits(self):
        with patch('resource.getrlimit', return_value=(1024, 1048576)), patch('resource.setrlimit') as setter:
            self.assertEqual(configure_nofile(self.config(nofile=65536), apply=True)['soft'], 65536)
            setter.assert_called_once()
        with patch('resource.getrlimit', return_value=(65536, 1048576)), patch('resource.setrlimit') as setter:
            self.assertEqual(configure_nofile(self.config(), apply=True)['soft'], 65536)
            setter.assert_not_called()

    def test_rejects_insufficient_explicit_or_hard_limits(self):
        with self.assertRaisesRegex(ValueError, 'capacity requirement'):
            configure_nofile(self.config(nofile=1024))
        with patch('resource.getrlimit', return_value=(1024, 1024)), patch('resource.setrlimit') as setter:
            with self.assertRaisesRegex(RuntimeError, 'hard limit 1024'):
                configure_nofile(self.config(), apply=True)
            setter.assert_not_called()

    def test_unlimited_soft_limit_is_not_lowered(self):
        import resource
        with patch('resource.getrlimit', return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY)), \
                patch('resource.setrlimit') as setter:
            self.assertEqual(configure_nofile(self.config(), apply=True)['soft'], resource.RLIM_INFINITY)
            setter.assert_not_called()


ASYNC_WORKER = r'''
import argparse, asyncio
from aiohttp import web
p = argparse.ArgumentParser(); p.add_argument('--port', type=int)
a, _ = p.parse_known_args()
release = asyncio.Event()
async def models(request):
    return web.json_response({'data': [{'id': 'dummy'}]})
async def stream(request):
    await request.read()
    response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
    await response.prepare(request)
    await response.write(b'ok')
    await release.wait()
    await response.write_eof()
    return response
async def finish(request):
    release.set()
    return web.json_response({'ok': True})
app = web.Application()
app.router.add_get('/v1/models', models)
app.router.add_post('/v1/messages', stream)
app.router.add_post('/finish', finish)
web.run_app(app, host='127.0.0.1', port=a.port, access_log=None)
'''


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process limits')
class LiveLimitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = configuration(self.root, free_ports(3))
        self.data['serve_args'][-1] = '380'
        self.data['router']['queue_limit_per_replica'] = 380
        self.data['startup_timeout_seconds'] = 15
        worker = self.root/'worker.py'
        worker.write_text(ASYNC_WORKER)
        self.data['worker'] = {'script': str(worker)}
        self.path = self.root/'deployment.json'
        self.path.write_text(json.dumps(self.data))
        self.env = {**os.environ, 'PYTHONPATH': str(TOOLS)}

    def command(self, action):
        return [sys.executable, '-m', 'fastllm_pytools.deploy', action, '--config', str(self.path)]

    def start_with_limit(self, hard=None):
        # Only this fresh launcher process gets reduced limits, not the test runner.
        code = ('import resource,os,sys; '
                'soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE); '
                f'resource.setrlimit(resource.RLIMIT_NOFILE,(1024,{hard if hard else "hard"})); '
                'os.execv(sys.executable,[sys.executable,*sys.argv[1:]])')
        return subprocess.run([sys.executable, '-c', code, *self.command('start')[1:]],
                              env=self.env, capture_output=True, text=True, timeout=45)

    def tearDown(self):
        subprocess.run(self.command('stop'), env=self.env, capture_output=True, timeout=30)
        self.temp.cleanup()

    def test_insufficient_hard_limit_fails_before_starting_workers(self):
        result = self.start_with_limit(hard=1024)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('RLIMIT_NOFILE hard limit 1024', result.stderr)
        self.assertFalse((self.root/'state/state.json').exists())

    def test_760_streams_keep_health_responsive_with_a_1024_limit_launcher(self):
        result = self.start_with_limit()
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads((self.root/'state/state.json').read_text())
        manifest = json.loads((Path(state['run_dir'])/'manifest.json').read_text())
        limits = manifest['resource_limits']['nofile']
        self.assertEqual(limits['soft_before'], 1024)
        self.assertGreaterEqual(limits['soft'], 2536)
        self.assertEqual(json.loads(result.stdout)['resource_limits']['nofile'], limits)
        for record in [state['supervisor'], *manifest['processes'].values()]:
            self.assertTrue(owned(record))
            line = next(line for line in Path(f"/proc/{record['pid']}/limits").read_text().splitlines()
                        if line.startswith('Max open files'))
            self.assertEqual(int(line.split()[3]), limits['soft'])
        base = f"http://127.0.0.1:{self.data['router']['port']}"

        async def exercise():
            from aiohttp import ClientSession, ClientTimeout, TCPConnector
            async with ClientSession(connector=TCPConnector(limit=0), timeout=ClientTimeout(total=30)) as client:
                responses = []
                async def open_stream(index):
                    response = await client.post(base+'/v1/messages', json={'stream': True},
                                                 headers={'X-Session-Id': f'stream-{index}'})
                    responses.append(response)
                    self.assertEqual(response.status, 200)
                    self.assertEqual(await response.content.readexactly(2), b'ok')
                try:
                    await asyncio.gather(*(open_stream(i) for i in range(760)))
                    pid = manifest['processes']['router']['pid']
                    self.assertGreaterEqual(len(list(Path(f'/proc/{pid}/fd').iterdir())), 1520)
                    # Check several times while all streams remain open.
                    for _ in range(3):
                        async with client.get(base+'/health', timeout=ClientTimeout(total=3)) as response:
                            self.assertEqual(response.status, 200)
                            self.assertEqual((await response.json())['healthy_replicas'], 2)
                        async with client.get(base+'/deploy/status', timeout=ClientTimeout(total=3)) as response:
                            live = await response.json()
                            self.assertEqual([r['active'] for r in live['replicas']], [380, 380])
                        await asyncio.sleep(.1)
                    for replica in self.data['replicas']:
                        async with client.post(f"http://127.0.0.1:{replica['port']}/finish") as response:
                            self.assertEqual(response.status, 200)
                    await asyncio.gather(*(response.read() for response in responses))
                finally:
                    for response in responses:
                        response.close()
        asyncio.run(exercise())


if __name__ == '__main__':
    unittest.main()
