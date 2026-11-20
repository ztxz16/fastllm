"""Single public HTTP endpoint. Replica-local tokenization, generation and SSE encoding."""

import asyncio
from contextlib import nullcontext
import hmac
import json
from pathlib import Path
import sys

import uvicorn

from .fastipc import (
    AdmissionError,
    Client,
    CREDIT,
    DATA,
    END,
    LENGTH,
    META_LIMIT,
    decode_headers,
    encode_headers,
    pack_meta,
)


async def reply(send, status, data):
    body = json.dumps(data, separators=(",", ":")).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if status == 429:
        headers.append((b"retry-after", b"1"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class Gateway:
    def __init__(self, lib, config, sockets):
        self.client = Client(lib, config)
        self.cfg, self.sockets = config, sockets
        self.settings = config["router"]

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await receive()
            try:
                await self.client.connect(self.sockets)
            except Exception as e:
                self.client.close()
                await send({"type": "lifespan.startup.failed", "message": str(e)})
                return
            await send({"type": "lifespan.startup.complete"})
            await receive()
            self.client.close()
            await send({"type": "lifespan.shutdown.complete"})
            return
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1003})
            return
        path = scope["path"]
        headers = dict(scope["headers"])
        key = headers.get(b"x-session-id") or headers.get(b"x-smg-routing-key", b"")
        if len(key) > 512 or b"\x00" in key:
            return await reply(send, 400, {"error": "Invalid replica session key"})
        if path == "/health":
            if scope["method"] not in ("GET", "HEAD"):
                return await reply(send, 405, {"error": "Method not allowed"})
            ok = sum(r["healthy"] for r in self.client.state())
            n = len(self.client.ranks)
            return await reply(
                send,
                200 if ok == n else 503,
                dict(status="ok" if ok == n else "unavailable", replicas=n, healthy_replicas=ok),
            )
        if self.cfg.get("api_key"):
            secret = self.cfg["api_key"].encode()
            if not (
                hmac.compare_digest(headers.get(b"authorization", b""), b"Bearer " + secret)
                or hmac.compare_digest(headers.get(b"x-api-key", b""), secret)
            ):
                return await reply(send, 401, {"error": "Unauthorized"})
        if path in ("/deploy/status", "/deploy/release"):
            if path == "/deploy/status" and scope["method"] in ("GET", "HEAD"):
                states = self.client.state()
                return await reply(
                    send,
                    200,
                    dict(
                        name=self.cfg["name"],
                        replicas=states,
                        replica_count=len(states),
                        aggregate_max_inflight=sum(r["max_inflight"] for r in states),
                    ),
                )
            if path == "/deploy/release" and scope["method"] == "POST":
                if not key:
                    return await reply(send, 400, {"error": "Session ID required"})
                if self.client.lib.fastipc_forget(self.client.router, key):
                    return await reply(send, 409, {"error": "Session is active"})
                return await reply(send, 200, {"released": True})
            return await reply(send, 405, {"error": "Method not allowed"})
        content_length = headers.get(b"content-length", b"")
        if content_length.isdigit() and int(content_length) > self.settings["max_body_bytes"]:
            return await reply(send, 413, {"error": "Request body too large"})
        if not path.startswith("/v1/"):
            return await reply(send, 404, {"error": "Not found"})
        control = (path, scope["method"]) in (("/v1/cancel", "POST"), ("/v1/active_conversations", "GET"))
        try:
            admission = nullcontext() if control else self.client.reserve(key)
            with admission as peer:
                await self.proxy(scope, receive, send, peer)
        except AdmissionError as e:
            await reply(send, e.status, {"error": str(e)})

    async def read_body(self, receive, send):
        chunks = []
        length = 0
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunk = msg.get("body", b"")
            length += len(chunk)
            if length > self.settings["max_body_bytes"]:
                return await reply(send, 413, {"error": "Request body too large"})
            chunks.append(chunk)
            if not msg.get("more_body"):
                break
        return b"".join(chunks)

    async def proxy(self, scope, receive, send, peer):
        path = scope["path"]
        try:
            meta = pack_meta(
                [
                    scope["method"],
                    path,
                    scope["raw_path"].decode("latin1"),
                    scope["query_string"].decode("latin1"),
                    encode_headers(scope["headers"]),
                    scope.get("scheme", "http"),
                    scope.get("client"),
                    scope.get("server"),
                    scope.get("http_version", "1.1"),
                    scope.get("root_path", ""),
                ]
            )
        except ValueError:
            return await reply(send, 431, {"error": "Request headers too large"})
        try:
            body = await asyncio.wait_for(
                self.read_body(receive, send), self.settings["request_timeout_seconds"]
            )
        except asyncio.TimeoutError:
            return await reply(send, 408, {"error": "Request body timed out"})
        if body is None:
            return
        if peer is None:
            work = self.control(path, meta, body, send)
        else:
            work = self.forward(peer, meta, body, send)
        relay = asyncio.create_task(work)
        del body, meta, work

        async def disconnected():
            if (await receive())["type"] == "http.disconnect":
                relay.cancel()

        monitor = asyncio.create_task(disconnected())
        try:
            await relay
        except asyncio.CancelledError:
            if not monitor.done():
                raise
        finally:
            monitor.cancel()

    async def control(self, path, meta, body, send):
        async def one(peer):
            status = None
            chunks = []

            async def collect(message):
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                else:
                    chunks.append(message.get("body", b""))

            await self.forward(peer, meta, body, collect, control=True)
            return status, json.loads(b"".join(chunks))

        responses = await asyncio.gather(*(one(p) for p in self.client.ranks))
        if path == "/v1/cancel":
            status, data = next(
                (r for r in responses if r[0] == 200),
                next((r for r in responses if r[0] != 404), responses[0]),
            )
            return await reply(send, status, data)
        for status, data in responses:
            if status != 200:
                return await reply(send, status, data)
        active = [item for _, data in responses for item in data["active_conversations"]]
        await reply(send, 200, {"active_conversations": active, "count": len(active)})

    async def forward(self, peer, meta, body, send, control=False):
        started = False
        stream = None

        async def run():
            nonlocal started, stream, meta, body
            stream = await peer.submit(meta, body, control)
            meta = body = None
            prefix = bytearray()
            while True:
                op, chunk = await stream.messages.get()
                if op not in (DATA, END):
                    raise ConnectionError("Replica stopped during request")
                if not started:
                    prefix.extend(chunk)
                    prefix_len = LENGTH.unpack_from(prefix)[0] if len(prefix) >= LENGTH.size else None
                    if prefix_len is not None and prefix_len > META_LIMIT:
                        raise ValueError("Invalid FastIPC response header")
                    if prefix_len is not None and len(prefix) >= LENGTH.size + prefix_len:
                        offset = LENGTH.size + prefix_len
                        status, headers = json.loads(prefix[LENGTH.size : offset])
                        values = [
                            (k, v) for k, v in decode_headers(headers) if k.lower() != b"x-fastllm-replica"
                        ]
                        values.append(
                            (b"x-fastllm-replica", self.cfg["replicas"][peer.rank]["name"].encode())
                        )
                        await send({"type": "http.response.start", "status": status, "headers": values})
                        started = True
                        chunk = bytes(prefix[offset:])
                        prefix.clear()
                    elif op == END:
                        raise ValueError("Truncated FastIPC response header")
                if started and (chunk or op == END):
                    await send({"type": "http.response.body", "body": chunk, "more_body": op != END})
                if op == END:
                    return
                peer.link.post(CREDIT, stream.slot, stream.ident)

        try:
            await asyncio.wait_for(run(), self.settings["request_timeout_seconds"])
        except (OSError, ValueError, asyncio.TimeoutError) as e:
            if started:
                raise
            status = 504 if isinstance(e, asyncio.TimeoutError) else 502
            await reply(send, status, {"error": str(e) or "Replica request timed out"})
        finally:
            if stream:
                peer.cancel(stream)


if __name__ == "__main__":
    from ..llm import fastllm_lib

    config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
    opts = config["router"]
    uvicorn.run(
        Gateway(fastllm_lib, config, Path(sys.argv[2])),
        host=opts["host"],
        port=opts["port"],
        access_log=False,
        lifespan="on",
        ws="none",
        interface="asgi3",
    )
