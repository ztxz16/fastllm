"""Serve the replica's existing ASGI app on a multiplexed local channel."""

import asyncio
import logging
import mmap
import os
import socket

from .fastipc import (
    Link,
    HELLO,
    READY,
    REQUEST,
    LARGE,
    DATA,
    END,
    CREDIT,
    CANCEL,
    CLOSED,
    FAILED,
    INPUT,
    OUTPUT,
    STRIDE,
    META_LIMIT,
    pack_meta,
    read_input,
    pool_size,
    ARENA,
    CONSUMED,
    OFFSET,
    encode_headers,
    decode_headers,
)


class Response:
    def __init__(self, worker, slot, ident):
        self.worker, self.slot, self.ident = worker, slot, ident
        self.pending = bytearray()
        self.outstanding = self.finished = self.scheduled = False
        self.space = asyncio.Event()
        self.task = None

    def schedule(self):
        if not self.scheduled and not self.outstanding:
            self.scheduled = True
            asyncio.get_running_loop().call_soon(self.flush)

    def flush(self):
        self.scheduled = False
        w = self.worker
        if w.jobs.get(self.slot) is not self or self.outstanding:
            return
        size = len(self.pending)
        if size or self.finished:
            start = self.slot * STRIDE + INPUT
            w.pool[start : start + size] = self.pending
            self.pending.clear()
            w.link.post(END if self.finished else DATA, self.slot, self.ident, size)
            self.space.set()
            if self.finished:
                del w.jobs[self.slot]
            else:
                self.outstanding = True

    async def append(self, data):
        pos = 0
        while pos < len(data):
            if len(self.pending) == OUTPUT:
                self.flush()
                if len(self.pending) == OUTPUT:
                    self.space.clear()
                    await self.space.wait()
            n = min(OUTPUT - len(self.pending), len(data) - pos)
            self.pending.extend(memoryview(data)[pos : pos + n])
            pos += n
        self.schedule()

    async def run(self, scope, body):
        started = complete = sent_body = False

        async def receive():
            nonlocal sent_body, body
            if not sent_body:
                sent_body = True
                chunk, body = body, None
                return {"type": "http.request", "body": chunk, "more_body": False}
            await asyncio.Future()  # Cancellation of the request task signals disconnect.

        async def send(msg):
            nonlocal started, complete
            if msg["type"] == "http.response.start" and not started:
                await self.append(pack_meta([msg["status"], encode_headers(msg.get("headers", []))]))
                started = True
            elif msg["type"] == "http.response.body" and started and not complete:
                await self.append(msg.get("body", b""))
                complete = not msg.get("more_body", False)
            else:
                raise ValueError("Unexpected replica ASGI response")

        await self.worker.app(scope, receive, send)
        if not complete:
            raise ValueError("Incomplete replica ASGI response")
        self.finished = True
        self.flush()


class WorkerPeer:
    def __init__(self, sock, app, state, capacity, limit):
        self.app, self.state, self.capacity, self.limit = app, state, capacity, limit
        self.pool = None
        self.jobs = {}
        self.link = Link(sock, self.receive, self.close)

    def close(self):
        for job in self.jobs.values():
            job.task.cancel()
        self.jobs.clear()
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def completed(self, job, task):
        error = None if task.cancelled() else task.exception()
        if error:
            logging.error(
                "FastIPC replica request failed", exc_info=(type(error), error, error.__traceback__)
            )
        if (task.cancelled() or error) and self.jobs.get(job.slot) is job:
            del self.jobs[job.slot]
            self.link.post(CLOSED if task.cancelled() else FAILED, job.slot, job.ident)

    def receive(self, op, slot, ident, size, fd):
        if op == HELLO:
            if (
                self.pool is not None
                or size != self.capacity
                or os.fstat(fd).st_size != pool_size(size, self.limit)
            ):
                raise ValueError("Invalid FastIPC handshake")
            self.pool = mmap.mmap(fd, pool_size(size, self.limit))
            self.link.post(READY)
            return
        if self.pool is None or slot > self.capacity:
            raise ValueError("Invalid FastIPC slot")
        job = self.jobs.get(slot)
        if op in (CREDIT, CANCEL):
            if job is None or job.ident != ident:
                return  # Terminal notification crossed an in-flight cancel.
            if op == CANCEL:
                job.task.cancel()
                if job.task.done():
                    # ASGI returned with a terminal block waiting for credit.
                    del self.jobs[slot]
                    self.link.post(CLOSED, slot, ident)
            else:
                job.outstanding = False
                job.flush()
            return
        if job is not None or op not in (REQUEST, LARGE, ARENA) or size > self.limit + META_LIMIT + 4:
            raise ValueError("Invalid FastIPC request")
        if op == LARGE:
            if os.fstat(fd).st_size != size:
                raise ValueError("Invalid FastIPC overflow region")
            with mmap.mmap(fd, size, access=mmap.ACCESS_READ) as large:
                meta, body = read_input(large, 0, size)
        elif op == ARENA:
            (start,) = OFFSET.unpack_from(self.pool, slot * STRIDE)
            if start < (self.capacity + 1) * STRIDE or start + size > len(self.pool):
                raise ValueError("FastIPC input outside staging area")
            meta, body = read_input(self.pool, start, size)
            self.link.post(CONSUMED, slot, ident)
        else:
            if size > INPUT:
                raise ValueError("FastIPC input exceeds slot")
            start = slot * STRIDE
            meta, body = read_input(self.pool, start, size)
        if len(body) > self.limit:
            raise ValueError("FastIPC body exceeds limit")
        method, path, raw, query, headers, scheme, client, server, version, root = meta
        scope = dict(
            type="http",
            asgi={"version": "3.0", "spec_version": "2.3"},
            method=method,
            path=path,
            raw_path=raw.encode("latin1"),
            query_string=query.encode("latin1"),
            headers=decode_headers(headers),
            scheme=scheme,
            client=client,
            server=server,
            http_version=version,
            root_path=root,
            state=dict(self.state),
        )
        job = self.jobs[slot] = Response(self, slot, ident)
        job.task = asyncio.create_task(job.run(scope, body))
        job.task.add_done_callback(lambda t: self.completed(job, t))


async def serve_worker(app, path, capacity, limit, on_ready=None):
    async with app.router.lifespan_context(app) as state:
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
            listener.bind(str(path))
            listener.listen(4)
            listener.setblocking(False)
            if on_ready is not None:
                on_ready()
            while True:
                sock, _ = await asyncio.get_running_loop().sock_accept(listener)
                WorkerPeer(sock, app, state or {}, capacity, limit)
