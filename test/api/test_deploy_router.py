import asyncio
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import ClientSession, web
from yarl import URL

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from fastllm_pytools.deploy.router import Router


def config(backends, capacity=2, **limits):
    return {'name': 'test', 'api_key': '', 'replicas': [
        {'name': f'replica{i}', 'url': base, 'max_inflight': capacity} for i, base in enumerate(backends)],
        'router': {'max_body_bytes': 1024*1024, 'max_sessions': 100, 'affinity_ttl_seconds': 60,
                   'health_interval_seconds': 60, 'queue_limit_per_replica': 1,
                   'queue_timeout_seconds': 1, 'request_timeout_seconds': 10, **limits}}


async def serve(app):
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    return runner, 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])


def backend_app(handler):
    app = web.Application()
    async def models(request):
        return web.json_response({'data': [{'id': 'test'}]})
    app.router.add_get('/v1/models', models)
    app.router.add_route('*', '/v1/{path:.*}', handler)
    return app


class AffinityTests(unittest.TestCase):
    def test_new_sessions_use_available_capacity_without_moving_existing_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = Router(config(['http://a', 'http://b'], 1, queue_limit_per_replica=0), tmp)
            router.healthy = [True, True]
            for session in ('active', 'idle-a', 'idle-b', 'idle-c'):
                router.select(session)
            # The first replica has fewer sessions but its only slot is busy.
            router.entries.pop('idle-b')
            router.pending[0] = 1
            router.references['active'] = 1
            self.assertEqual(router.select('new'), 1)
            self.assertEqual(router.select('active'), 0)
            router.pending[1] = 1
            with self.assertRaises(web.HTTPTooManyRequests):
                router.select('overflow')
            self.assertNotIn('overflow', router.entries)

    def test_affinity_ttl_lru_persistence_and_active_protection(self):
        with tempfile.TemporaryDirectory() as tmp, patch('fastllm_pytools.deploy.router.time.time') as clock:
            clock.return_value = 100
            settings = config(['http://a', 'http://b'], 1, max_sessions=2)
            router = Router(settings, tmp)
            router.healthy = [True, True]
            first = router.select('first')
            router.references['first'] = 1
            router.select('idle')
            clock.return_value = 101
            router.select('new')
            self.assertNotIn('idle', router.entries)
            router.references['new'] = 1
            with self.assertRaises(web.HTTPServiceUnavailable):
                router.select('overflow')
            clock.return_value = 200
            router.references.pop('new')
            self.assertEqual(router.select('first'), first)
            self.assertNotIn('new', router.entries)
            router.persist()
            restored = Router(settings, tmp)
            restored.healthy = [True, True]
            self.assertEqual(restored.select('first'), first)
            changed = copy.deepcopy(settings)
            changed['replicas'][0]['url'] = 'http://changed'
            with self.assertRaises(ValueError):
                Router(changed, tmp)

    def test_unhealthy_affinity_is_not_silently_moved_and_unkeyed_uses_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = Router(config(['http://a', 'http://b']), tmp)
            router.healthy = [True, True]
            rank = router.select('sticky')
            router.pending[rank] = 2
            self.assertEqual(router.select(), 1-rank)
            self.assertEqual(router.select('sticky'), rank)
            router.healthy[rank] = False
            self.assertEqual(router.select(), 1-rank)
            with self.assertRaises(web.HTTPServiceUnavailable):
                router.select('sticky')

    def test_session_cache_is_not_lifetime_generation_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = Router(config(['http://a', 'http://b'], 1), tmp)
            router.healthy = [True, True]
            for i in range(20):
                router.select(str(i))
            self.assertEqual(router.assigned, [10, 10])


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_compressed_request_body_and_encoding_are_preserved(self):
        received = []
        async def backend(request):
            received.append((request.headers.get('Content-Encoding'), await request.read()))
            return web.json_response({'ok': True})
        upstream_app = web.Application(handler_args={'auto_decompress': False})
        async def models(request):
            return web.json_response({'data': []})
        upstream_app.router.add_get('/v1/models', models)
        upstream_app.router.add_post('/v1/messages', backend)
        upstream, base = await serve(upstream_app)
        with tempfile.TemporaryDirectory() as tmp:
            router = Router(config([base]), tmp)
            proxy, url = await serve(router.app())
            payload = gzip.compress(b'{"messages":[{"role":"user","content":"hello"}]}')
            try:
                async with ClientSession() as client:
                    async with client.post(url+'/v1/messages', data=payload, headers={
                            'Content-Type': 'application/json', 'Content-Encoding': 'gzip'}) as response:
                        self.assertEqual(response.status, 200)
                        await response.read()
                self.assertEqual(received, [('gzip', payload)])
                self.assertEqual(router.pending, [0])
            finally:
                await proxy.cleanup()
                await upstream.cleanup()

    async def test_transparent_requests_with_and_without_session_and_authentication(self):
        calls = []
        async def backend(request):
            calls.append((request.raw_path, request.headers.get('Authorization'), request.headers.get('x-api-key')))
            return web.Response(body=await request.read(), headers={'X-Request-Id': 'upstream'}, status=201)
        upstream, base = await serve(backend_app(backend))
        with tempfile.TemporaryDirectory() as tmp:
            settings = config([base, base])
            settings['api_key'] = 'secret'
            router = Router(settings, tmp)
            proxy, url = await serve(router.app())
            try:
                async with ClientSession() as client:
                    async with client.post(url+'/v1/messages', data=b'no-auth') as response:
                        self.assertEqual(response.status, 401)
                    async with client.get(url+'/health') as response:
                        self.assertEqual(response.status, 200)
                    replicas = []
                    for _ in range(4):
                        async with client.post(URL(url+'/v1/chat/completions?name=a%2Fb', encoded=True), data=b'{"literal":1}',
                                headers={'Authorization': 'Bearer secret'}) as response:
                            self.assertEqual(response.status, 201)
                            self.assertEqual(response.headers['X-Request-Id'], 'upstream')
                            self.assertEqual(await response.read(), b'{"literal":1}')
                            replicas.append(response.headers['X-FastLLM-Replica'])
                    self.assertEqual(set(replicas), {'replica0', 'replica1'})
                    sticky = []
                    for _ in range(3):
                        async with client.post(url+'/v1/messages', data=b'unchanged',
                                headers={'x-api-key': 'secret', 'X-Session-Id': 'same'}) as response:
                            sticky.append(response.headers['X-FastLLM-Replica'])
                            self.assertEqual(await response.read(), b'unchanged')
                    self.assertEqual(len(set(sticky)), 1)
                    async with client.get(url+'/deploy/status', headers={'x-api-key': 'secret'}) as response:
                        state = await response.json()
                        self.assertEqual(state['aggregate_max_inflight'], 4)
                        self.assertNotIn('secret', json.dumps(state))
                    async with client.post(url+'/deploy/release', headers={
                            'x-api-key': 'secret', 'X-Session-Id': 'same'}) as response:
                        self.assertEqual(response.status, 200)
                self.assertNotIn('same', router.entries)
                self.assertEqual(len(calls), 7)
                self.assertTrue(all(path.endswith('name=a%2Fb') for path, _, _ in calls[:4]))
                self.assertEqual(calls[0][1], 'Bearer secret')
                self.assertEqual(calls[-1][2], 'secret')
            finally:
                await proxy.cleanup()
                await upstream.cleanup()

    async def test_stream_first_chunk_queue_bounds_timeouts_and_client_disconnect(self):
        released = asyncio.Event()
        calls = []
        async def backend(request):
            calls.append(1)
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            await response.write(b'data: first\n\n')
            try:
                await asyncio.Event().wait()
            finally:
                released.set()
            return response
        upstream, base = await serve(backend_app(backend))
        with tempfile.TemporaryDirectory() as tmp:
            router = Router(config([base], 1, queue_timeout_seconds=.2), tmp)
            proxy, url = await serve(router.app())
            try:
                async with ClientSession() as client:
                    first = await client.post(url+'/v1/messages', json={}, headers={'X-Session-Id': 'one'})
                    self.assertEqual(await asyncio.wait_for(first.content.read(13), 2), b'data: first\n\n')
                    queued = asyncio.create_task(client.post(url+'/v1/messages', json={}, headers={'X-Session-Id': 'two'}))
                    for _ in range(100):
                        if router.pending[0] == 2:
                            break
                        await asyncio.sleep(.001)
                    async with client.post(url+'/v1/messages', json={}) as full:
                        self.assertEqual(full.status, 429)
                    async with client.post(url+'/deploy/release', headers={'X-Session-Id': 'one'}) as busy:
                        self.assertEqual(busy.status, 409)
                    timed_out = await queued
                    self.assertEqual(timed_out.status, 504)
                    await timed_out.read()
                    timed_out.release()
                    self.assertEqual(router.pending, [1])
                    first.close()
                    await asyncio.wait_for(released.wait(), 2)
                    await asyncio.sleep(.05)
                    self.assertEqual(router.pending, [0])
                    self.assertEqual(router.active, [0])
                    self.assertEqual(router.references, {})
                    self.assertEqual(len(calls), 1)
                    self.assertIn('client_cancelled', (Path(tmp)/'requests.jsonl').read_text())
            finally:
                await proxy.cleanup()
                await upstream.cleanup()

    async def test_upstream_http_error_is_forwarded_once_and_health_reflects_loss(self):
        calls = []
        async def backend(request):
            calls.append(1)
            return web.json_response({'error': 'bad request'}, status=400, headers={'Retry-After': '5'})
        upstream, base = await serve(backend_app(backend))
        with tempfile.TemporaryDirectory() as tmp:
            router = Router(config([base]), tmp)
            proxy, url = await serve(router.app())
            try:
                async with ClientSession() as client:
                    async with client.post(url+'/v1/messages', json={}) as response:
                        self.assertEqual(response.status, 400)
                        self.assertEqual(response.headers['Retry-After'], '5')
                        self.assertEqual(await response.json(), {'error': 'bad request'})
                    self.assertEqual(len(calls), 1)
                    await upstream.cleanup()
                    await router.check_health()
                    async with client.get(url+'/health') as response:
                        self.assertEqual(response.status, 503)
                    async with client.post(url+'/v1/messages', json={}) as response:
                        self.assertEqual(response.status, 503)
            finally:
                await proxy.cleanup()
                await upstream.cleanup()


if __name__ == '__main__':
    unittest.main()
