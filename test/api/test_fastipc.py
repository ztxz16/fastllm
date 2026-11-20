"""Exercise the real shared pools, native admission and separate Linux replica processes."""

import asyncio
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def child(path, rank, capacity):
    from fastapi import FastAPI, Request
    from starlette.responses import Response, StreamingResponse
    from fastllm_pytools.deploy.ipc_server import serve_worker

    app = FastAPI()

    @app.post("/v1/echo")
    async def echo(request: Request):
        return Response(await request.body(), headers={"x-rank": str(rank)})

    @app.get("/v1/stream")
    async def stream():
        async def chunks():
            for _ in range(16):
                yield bytes(range(256)) * 1024

        return StreamingResponse(chunks())

    @app.get("/v1/block")
    async def block():
        await asyncio.Future()

    @app.get("/v1/active_conversations")
    async def active():
        return {"active_conversations": [{"conversation_id": str(rank)}], "count": 1}

    @app.post("/v1/cancel")
    async def cancel():
        return Response(
            json.dumps({"rank": rank}), status_code=200 if rank == 1 else 404, media_type="application/json"
        )

    asyncio.run(serve_worker(app, path, capacity, 2 << 20))


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux shared-memory transport")
class FastIPCTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("c++"):
            raise unittest.SkipTest("FastIPC tests require a C++ compiler")
        build = tempfile.TemporaryDirectory()
        cls.addClassCleanup(build.cleanup)
        so = Path(build.name) / "fastipc.so"
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-pthread",
                str(ROOT / "fastipc/fastipc.cpp"),
                "-o",
                str(so),
            ],
            check=True,
        )
        cls.lib = C.CDLL(str(so))

    async def asyncSetUp(self):
        from fastllm_pytools.deploy.ipc_gateway import Gateway

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)
        self.children = []
        cap = 256 if self._testMethodName.startswith("test_512") else 1
        cfg = dict(
            name="test",
            api_key=None,
            replicas=[dict(name=str(i), max_inflight=cap) for i in range(2)],
            router=dict(
                queue_limit_per_replica=1,
                queue_timeout_seconds=0.3,
                request_timeout_seconds=20,
                max_sessions=4096,
                affinity_ttl_seconds=30,
                max_body_bytes=2 << 20,
            ),
        )
        for i in range(2):
            p = subprocess.Popen([sys.executable, __file__, "--worker", str(path / str(i)), str(i), str(cap)])
            self.addCleanup(p.wait, 5)
            self.addCleanup(p.terminate)
            self.children.append(p)
        for _ in range(200):
            if all((path / str(i)).exists() for i in range(2)):
                break
            await asyncio.sleep(0.05)
        self.gateway = Gateway(self.lib, cfg, path)
        await self.gateway.client.connect(path)
        self.addCleanup(self.gateway.client.close)

    async def request(
        self, path="/v1/echo", body=b"", key="", disconnected=None, slow=None, headers=(), on_read=None
    ):
        method = "POST" if path in ("/v1/echo", "/v1/cancel", "/deploy/release") else "GET"
        scope = dict(
            type="http",
            http_version="1.1",
            method=method,
            path=path,
            raw_path=path.encode(),
            query_string=b"",
            headers=[(b"x-session-id", key.encode()), *headers],
            scheme="http",
        )
        first = True

        async def receive():
            nonlocal first
            if first:
                if on_read:
                    on_read()
                first = False
                return dict(type="http.request", body=body)
            if disconnected is None:
                await asyncio.Future()
            await disconnected.wait()
            return dict(type="http.disconnect")

        headers = {}
        status = None
        chunks = []

        async def send(msg):
            nonlocal status, headers
            if msg["type"] == "http.response.start":
                status = msg["status"]
                headers = dict(msg["headers"])
            else:
                if slow:
                    await slow.wait()
                chunks.append(msg["body"])

        await self.gateway(scope, receive, send)
        return status, headers, b"".join(chunks)

    async def test_512_fair_long_body_and_stable_fds(self):
        before = len(os.listdir("/proc/self/fd"))
        rows = await asyncio.gather(*(self.request(body=str(i).encode(), key=str(i)) for i in range(512)))
        ranks = [r[1][b"x-fastllm-replica"] for r in rows]
        self.assertEqual((ranks.count(b"0"), ranks.count(b"1")), (256, 256))
        self.assertEqual([r[2] for r in rows], [str(i).encode() for i in range(512)])
        large = bytes(range(256)) * 3200
        row = await self.request(body=large, key="7")
        self.assertEqual((row[0], row[1][b"x-fastllm-replica"], row[2]), (200, ranks[7], large))
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), before + 1)

    async def test_input_arena_and_exhaustion_fallback(self):
        peer = self.gateway.client.ranks[0]
        from unittest.mock import patch
        from fastllm_pytools.deploy import fastipc

        with patch.object(fastipc, "buffer", wraps=fastipc.buffer) as overflow:
            for size in (200000, 1500000, 300000):
                body = bytes([size % 251]) * size
                self.assertEqual((await self.request(body=body, key="arena"))[2], body)
            self.assertEqual(overflow.call_count, 1)
        self.assertEqual(len(peer.arena.free), 1)

    async def test_slow_stream_keeps_backpressure_off_other_rank(self):
        gate = asyncio.Event()
        slow = asyncio.create_task(self.request("/v1/stream", key="a", slow=gate))
        await asyncio.sleep(0.04)
        self.assertEqual((await self.request(body=b"fast", key="b"))[2], b"fast")
        self.assertEqual((await self.request("/deploy/release", key="a"))[0], 409)
        gate.set()
        self.assertEqual((await slow)[2], bytes(range(256)) * 1024 * 16)
        self.assertEqual((await self.request("/deploy/release", key="a"))[0], 200)

    async def test_cancel_wait_and_refill_and_control_at_capacity(self):
        stop = asyncio.Event()
        held = [asyncio.create_task(self.request("/v1/block", key=s, disconnected=stop)) for s in ("a", "b")]
        await asyncio.sleep(0.04)
        queued = asyncio.create_task(self.request(body=b"refilled", key="a"))
        await asyncio.sleep(0.01)
        self.assertEqual((await self.request(body=b"excess", key="a"))[0], 429)
        self.assertEqual(json.loads((await self.request("/v1/active_conversations"))[2])["count"], 2)
        self.assertEqual(json.loads((await self.request("/v1/cancel"))[2]), {"rank": 1})
        stop.set()
        await asyncio.gather(*held)
        self.assertEqual((await queued)[2], b"refilled")
        again = await self.request(body=b"next", key="b")
        self.assertEqual(again[2], b"next")

    async def test_gateway_auth_and_admission_before_upload(self):
        def unread():
            self.fail("Rejected request body was read")

        self.gateway.cfg["api_key"] = "private"
        self.assertEqual((await self.request(on_read=unread))[0], 401)
        self.assertEqual((await self.request("/health", on_read=unread))[0], 200)
        for header in [(b"authorization", b"Bearer private"), (b"x-api-key", b"private")]:
            self.assertEqual((await self.request(headers=[header]))[0], 200)
            self.assertEqual((await self.request("/docs", headers=[header], on_read=unread))[0], 404)
        with self.gateway.client.reserve(b"busy"), self.gateway.client.reserve(b"busy"):
            result = await self.request(key="busy", headers=[header], on_read=unread)
            self.assertEqual(result[0], 429)
            self.assertEqual(result[1][b"retry-after"], b"1")

    async def test_idle_session_moves_while_active_sessions_stay_on_owner(self):
        client = self.gateway.client
        with client.reserve(b"sticky") as owner:
            with client.reserve(b"sticky") as again:
                self.assertIs(owner, again)
        with client.reserve(b"") as free_rank:
            self.assertIsNot(free_rank, owner)
        with client.reserve(b"") as busy_rank:
            self.assertIs(busy_rank, owner)
            with client.reserve(b"sticky") as moved:
                self.assertIsNot(moved, owner)

    async def test_worker_exit_and_size_limit(self):
        held = asyncio.create_task(self.request("/v1/block", key="a"))
        await asyncio.sleep(0.04)
        self.children[0].terminate()
        self.assertEqual((await held)[0], 502)
        self.assertEqual((await self.request("/health"))[0], 503)
        self.assertEqual((await self.request(key="a"))[0], 503)
        self.assertEqual((await self.request(key="other"))[0], 200)
        self.assertEqual((await self.request(body=b"z" * ((2 << 20) + 1)))[0], 413)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        child(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        unittest.main()
