"""Transparent streaming HTTP router with bounded queues and idle session affinity."""
import argparse
import asyncio
import contextlib
import hmac
import json
from pathlib import Path
import time

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from multidict import CIMultiDict
from yarl import URL

from .process import write_json

HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length'}


def end_to_end(headers):
    excluded = HOP_HEADERS | {s.strip().lower() for s in headers.get('Connection', '').split(',')}
    return CIMultiDict((k, v) for k, v in headers.items() if k.lower() not in excluded)


class Router:
    def __init__(self, config, directory):
        self.config = config
        self.settings = config['router']
        self.backends = [r['url'] for r in config['replicas']]
        self.capacities = [r['max_inflight'] for r in config['replicas']]
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory/'affinity.json'
        self.entries = {}
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved['backends'] != self.backends:
                raise ValueError('Affinity snapshot belongs to different backends')
            self.entries = saved['sessions']
            if any(type(e['rank']) is not int or not 0 <= e['rank'] < len(self.backends)
                   or type(e['last_seen']) not in (int, float) for e in self.entries.values()):
                raise ValueError('Invalid affinity snapshot')
        self.active = [0] * len(self.backends)
        self.pending = [0] * len(self.backends)
        self.healthy = [False] * len(self.backends)
        self.references = {}
        self.slots = [asyncio.Semaphore(n) for n in self.capacities]
        self.cursor = 0
        self.client = None

    @property
    def assigned(self):
        return [sum(e['rank'] == i for e in self.entries.values()) for i in range(len(self.backends))]

    def persist(self):
        write_json(self.path, {'backends': self.backends, 'sessions': self.entries})

    def expire(self):
        now = time.time()
        for session, entry in list(self.entries.items()):
            if not self.references.get(session) and now-entry['last_seen'] >= self.settings['affinity_ttl_seconds']:
                del self.entries[session]

    def select(self, session=None):
        self.expire()
        if session in self.entries:
            rank = self.entries[session]['rank']
            if not self.healthy[rank]:
                raise web.HTTPServiceUnavailable(text='Session replica is unavailable')
            self.entries[session]['last_seen'] = time.time()
            return rank
        candidates = [i for i, health in enumerate(self.healthy) if health]
        if not candidates:
            raise web.HTTPServiceUnavailable(text='No healthy replicas')
        # New sessions and unkeyed requests can use any available replica.
        # Existing sessions above retain affinity even when their queue is full.
        candidates = [i for i in candidates if self.pending[i] <
                      self.capacities[i] + self.settings['queue_limit_per_replica']]
        if not candidates:
            raise web.HTTPTooManyRequests(text='All replica queues are full', headers={'Retry-After': '1'})
        if session:
            while len(self.entries) >= self.settings['max_sessions']:
                idle = [s for s in self.entries if not self.references.get(s)]
                if not idle:
                    raise web.HTTPServiceUnavailable(text='Affinity cache occupied by active sessions')
                del self.entries[min(idle, key=lambda s: self.entries[s]['last_seen'])]
            counts = self.assigned
            rank = min(candidates, key=lambda i: (counts[i]/self.capacities[i],
                       self.pending[i]/self.capacities[i], (i-self.cursor) % len(self.backends)))
            self.entries[session] = {'rank': rank, 'last_seen': time.time()}
            self.persist()
        else:
            rank = min(candidates, key=lambda i: (self.pending[i]/self.capacities[i],
                       (i-self.cursor) % len(self.backends)))
        self.cursor = (rank+1) % len(self.backends)
        return rank

    async def check_health(self):
        headers = {'Authorization': 'Bearer '+self.config['api_key']} if self.config['api_key'] else {}
        async def check(index, base):
            try:
                async with self.client.get(base+'/v1/models', headers=headers,
                                           timeout=ClientTimeout(total=3)) as response:
                    response.raise_for_status()
                    await response.json()
                self.healthy[index] = True
            except (OSError, ValueError, asyncio.TimeoutError) as error:
                self.healthy[index] = False
            except Exception:
                self.healthy[index] = False
        await asyncio.gather(*(check(i, url) for i, url in enumerate(self.backends)))

    async def lifecycle(self, app):
        timeout = ClientTimeout(total=self.settings['request_timeout_seconds'], sock_connect=15, sock_read=None)
        async with ClientSession(timeout=timeout, connector=TCPConnector(limit=0), auto_decompress=False,
                                 trust_env=False) as client:
            self.client = client
            await self.check_health()
            async def monitor():
                while True:
                    await asyncio.sleep(self.settings['health_interval_seconds'])
                    await self.check_health()
            monitor_task = asyncio.create_task(monitor())
            try:
                yield
            finally:
                monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor_task

    async def health(self, request):
        healthy = all(self.healthy)
        return web.json_response({'status': 'ok' if healthy else 'degraded',
                                  'healthy_replicas': sum(self.healthy), 'replicas': len(self.backends)},
                                 status=200 if healthy else 503)

    async def state(self, request):
        self.expire()
        assigned = self.assigned
        return web.json_response({'name': self.config['name'], 'replica_count': len(self.backends),
            'aggregate_max_inflight': sum(self.capacities), 'replicas': [
                {'name': replica['name'], 'healthy': self.healthy[i], 'max_inflight': self.capacities[i],
                 'active': self.active[i], 'queued': self.pending[i]-self.active[i], 'sessions': assigned[i]}
                for i, replica in enumerate(self.config['replicas'])]})

    async def release(self, request):
        session = request.headers.get('X-Session-Id') or request.headers.get('X-SMG-Routing-Key')
        if not session:
            raise web.HTTPBadRequest(text='A session ID header is required')
        if self.references.get(session):
            raise web.HTTPConflict(text='Session has active or queued requests')
        self.entries.pop(session, None)
        self.persist()
        return web.json_response({'released': True})

    async def forward(self, request):
        session = request.headers.get('X-Session-Id') or request.headers.get('X-SMG-Routing-Key')
        if session and len(session) > 512:
            raise web.HTTPBadRequest(text='Session ID exceeds 512 characters')
        rank = self.select(session)
        if self.pending[rank] >= self.capacities[rank] + self.settings['queue_limit_per_replica']:
            raise web.HTTPTooManyRequests(text='Replica queue is full', headers={'Retry-After': '1'})
        self.pending[rank] += 1
        if session:
            self.references[session] = self.references.get(session, 0) + 1
        response, acquired = None, False
        record = {'start': time.time(), 'replica': self.config['replicas'][rank]['name'],
                  'path': request.path, 'status': None, 'bytes': 0}
        try:
            try:
                await asyncio.wait_for(self.slots[rank].acquire(), self.settings['queue_timeout_seconds'])
            except asyncio.TimeoutError as error:
                raise web.HTTPGatewayTimeout(text='Replica queue wait timed out') from error
            acquired = True
            self.active[rank] += 1
            body = await request.read()
            headers = end_to_end(request.headers)
            headers['Accept-Encoding'] = 'identity'
            async with self.client.request(request.method, URL(self.backends[rank]+request.raw_path, encoded=True),
                    headers=headers, data=body, allow_redirects=False) as upstream:
                record.update(status=upstream.status, headers_at=time.time())
                headers = end_to_end(upstream.headers)
                headers['X-FastLLM-Replica'] = self.config['replicas'][rank]['name']
                response = web.StreamResponse(status=upstream.status, headers=headers)
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                    record['bytes'] += len(chunk)
                await response.write_eof()
                return response
        except asyncio.CancelledError:
            record['error'] = 'client_cancelled'
            raise
        except web.HTTPException as error:
            record.update(status=error.status, error=error.reason)
            raise
        except Exception as error:
            record['error'] = type(error).__name__
            if response is not None and response.prepared:
                if request.transport:
                    request.transport.close()
                return response
            raise web.HTTPBadGateway(text='Replica request failed') from error
        finally:
            if acquired:
                self.active[rank] -= 1
                self.slots[rank].release()
            self.pending[rank] -= 1
            if session:
                self.references[session] -= 1
                if not self.references[session]:
                    del self.references[session]
                self.entries[session]['last_seen'] = time.time()
                self.persist()
            record['end'] = time.time()
            with (self.directory/'requests.jsonl').open('a') as output:
                output.write(json.dumps(record)+'\n')

    def app(self):
        @web.middleware
        async def authenticate(request, handler):
            key = self.config['api_key']
            if key and request.path != '/health':
                bearer = request.headers.get('Authorization', '')
                anthropic = request.headers.get('x-api-key', '')
                if not (hmac.compare_digest(bearer.encode(), ('Bearer '+key).encode())
                        or hmac.compare_digest(anthropic.encode(), key.encode())):
                    raise web.HTTPUnauthorized(text='Unauthorized')
            return await handler(request)
        # Preserve the wire body together with Content-Encoding when proxying.
        app = web.Application(client_max_size=self.settings['max_body_bytes'],
                              middlewares=[authenticate], handler_args={'auto_decompress': False})
        app.cleanup_ctx.append(self.lifecycle)
        app.router.add_get('/health', self.health)
        app.router.add_get('/deploy/status', self.state)
        app.router.add_post('/deploy/release', self.release)
        app.router.add_route('*', '/v1/{path:.*}', self.forward)
        return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    directory = parser.parse_args().run_dir
    config = json.loads((directory/'config.json').read_text())
    router = Router(config, directory/'routing')
    web.run_app(router.app(), host=config['router']['host'], port=config['router']['port'],
                handler_cancellation=True, access_log=None, shutdown_timeout=config['stop_timeout_seconds'])
