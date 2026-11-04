"""Run with: python -m unittest discover -s test -p test_pd_router.py"""
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import ClientSession, web
from tools.fastllm_pytools.router import create_router, load_config


def configuration(directory):
    return {
        'version': 1, 'model_id': 'checkpoint-revision', 'state_dir': directory,
        'prefill': [{'name': 'p1', 'endpoint': '127.0.0.1:19000',
                     'url': 'http://127.0.0.1:19001', 'command': ['ftllm', 'serve', '/model']}],
        'decode': [{'name': 'd1', 'url': 'http://127.0.0.1:19002',
                    'command': ['ftllm', 'serve', '/model']}],
    }


class ConfigurationTest(unittest.TestCase):
    def test_deployment_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            config = configuration(directory)
            path.write_text(json.dumps(config))
            loaded = load_config(path)
            pd = json.loads(loaded['decode'][0]['command'][-1])
            self.assertEqual(pd['prefill'], ['127.0.0.1:19000'])
            self.assertEqual(pd['model_id'], 'checkpoint-revision')
            self.assertNotIn('chunk_tokens', pd)
            self.assertNotIn('receive_device', pd)
            prefill_pd = json.loads(loaded['prefill'][0]['command'][-1])
            self.assertEqual(prefill_pd['chunk_tokens'], 2048)
            self.assertEqual(prefill_pd['cache_bytes'], 1 << 30)
            self.assertNotIn('prefill', prefill_pd)
            self.assertEqual(loaded['decode'][0]['command'][-2], '--pd')
            config['decode'][0]['prefill'] = ['unknown:19000']
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)
            del config['decode'][0]['prefill']
            config['prefill'][0]['command'].append('--pd={}')
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)


class ReadinessTest(unittest.IsolatedAsyncioTestCase):
    async def test_prefill_dependencies_gate_decode_admission(self):
        readiness = {'p1': True, 'p2': True, 'd1': True, 'd2': True}
        runners, urls = [], {}
        try:
            for name in readiness:
                async def models(request, name=name):
                    return web.json_response({'data': []}, status=200 if readiness[name] else 503)
                app = web.Application()
                app.router.add_get('/v1/models', models)
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, '127.0.0.1', 0)
                await site.start()
                runners.append(runner)
                urls[name] = 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])
            with tempfile.TemporaryDirectory() as directory:
                config = configuration(directory)
                config['prefill'].append({'name': 'p2', 'endpoint': '127.0.0.1:19010',
                    'url': urls['p2'], 'command': ['ftllm', 'serve', '/model']})
                config['prefill'][0]['url'] = urls['p1']
                config['decode'][0].update(url=urls['d1'], prefill=['127.0.0.1:19000'])
                config['decode'].append({'name': 'd2', 'url': urls['d2'],
                    'prefill': ['127.0.0.1:19010'], 'command': ['ftllm', 'serve', '/model']})
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(config))
                router = create_router(load_config(path), Path(directory) / 'routing')
                async with ClientSession() as router.client:
                    await router.check_health()
                    self.assertEqual(router.healthy, [True, True])
                    readiness['p1'] = False
                    await router.check_health()
                    self.assertEqual(router.healthy, [False, True])
                    self.assertEqual(router.select(), 1)
                    self.assertEqual((await router.health(None)).status, 503)
                    readiness['p1'] = True
                    await router.check_health()
                    self.assertEqual(router.healthy, [True, True])
                    self.assertEqual((await router.health(None)).status, 200)
        finally:
            for runner in runners:
                await runner.cleanup()


if __name__ == '__main__':
    unittest.main()
