"""Host-local multiplexed IPC. Each replica has one duplex socket and one shared pool."""

import array
import bisect
import asyncio
import ctypes as C
import errno
import json
import mmap
import os
import socket
import struct
from collections import deque
from contextlib import contextmanager

HELLO, READY, REQUEST, LARGE, DATA, END, CREDIT, CANCEL, CLOSED, FAILED, ARENA, CONSUMED = range(12)
FRAME = struct.Struct("=IIQI")  # operation, slot, unique generation, length
LENGTH = struct.Struct("=I")
INPUT = 16 * 1024
OUTPUT = 32 * 1024
STRIDE = INPUT + OUTPUT
MAX_FRAMES = 64
META_LIMIT = 1024 * 1024
OFFSET = struct.Struct("=Q")


def pool_size(capacity, limit):
    staging = min(limit + META_LIMIT + 4, max(1 << 20, min(16 << 20, capacity * 65536)))
    return (capacity + 1) * STRIDE + staging


class InputArena:
    """Variable-size requests borrow pages until the replica consumes their body."""

    def __init__(self, start, end):
        self.free = [(start, end)]

    def take(self, size):
        size = (size + 4095) & ~4095
        for i, (start, end) in enumerate(self.free):
            if end - start >= size:
                self.free[i : i + 1] = [(start + size, end)] if start + size < end else []
                return start, start + size
        return None

    def give(self, block):
        start, end = block
        i = bisect.bisect_left(self.free, block)
        if i and self.free[i - 1][1] == start:
            i -= 1
            start = self.free.pop(i)[0]
        if i < len(self.free) and end == self.free[i][0]:
            end = self.free.pop(i)[1]
        self.free.insert(i, (start, end))


def signatures(lib):
    i, p, key = C.c_int, C.c_void_p, C.c_char_p
    for name, result, args in (
        ("fastipc_router_create", p, [C.POINTER(i), i, i, i, C.c_double]),
        ("fastipc_router_destroy", None, [p]),
        ("fastipc_reserve", i, [p, key]),
        ("fastipc_release", None, [p, i, key]),
        ("fastipc_forget", i, [p, key]),
        ("fastipc_rank_offline", None, [p, i]),
        ("fastipc_rank_state", None, [p, i, C.POINTER(i)]),
        ("fastipc_create_buffer", i, [C.c_size_t]),
    ):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result


def buffer(lib, size):
    fd = lib.fastipc_create_buffer(size)
    if fd < 0:
        raise OSError(-fd, os.strerror(-fd))
    return fd


class Link:
    """One event loop owns a peer. Merge ready notifications without a batching timer."""

    def __init__(self, sock, received, disconnected):
        self.sock = sock
        sock.setblocking(False)
        self.loop = asyncio.get_running_loop()
        self.received, self.disconnected = received, disconnected
        self.pending = deque()
        self.scheduled = False
        self.closed = False
        self.loop.add_reader(sock.fileno(), self.read)

    def post(self, op, slot=0, ident=0, size=0, fd=None):
        if self.closed:
            if fd is not None:
                os.close(fd)
            raise ConnectionError("FastIPC peer is closed")
        frame = FRAME.pack(op, slot, ident, size)
        if fd is None and self.pending:
            prev, prev_fd = self.pending[-1]
            if prev_fd is None and len(prev) < MAX_FRAMES * FRAME.size:
                prev.extend(frame)
                return
        self.pending.append((bytearray(frame), fd))
        if not self.scheduled:
            self.scheduled = True
            self.loop.call_soon(self.flush)

    def flush(self):
        if self.closed:
            return
        try:
            while self.pending:
                packet, fd = self.pending[0]
                if fd is None:
                    sent = self.sock.send(packet)
                else:
                    sent = self.sock.sendmsg(
                        [packet], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
                    )
                if sent != len(packet):
                    raise ConnectionError("Incomplete FastIPC control packet")
                self.pending.popleft()
                if fd is not None:
                    os.close(fd)
        except BlockingIOError:
            self.loop.add_writer(self.sock.fileno(), self.flush)
            return
        except OSError:
            self.close()
            return
        self.loop.remove_writer(self.sock.fileno())
        self.scheduled = False

    def read(self):
        try:
            for _ in range(16):
                data, control, flags, _ = self.sock.recvmsg(
                    FRAME.size * MAX_FRAMES, socket.CMSG_SPACE(4), socket.MSG_CMSG_CLOEXEC
                )
                fds = []
                try:
                    for level, kind, payload in control:
                        if (level, kind) == (socket.SOL_SOCKET, socket.SCM_RIGHTS):
                            values = array.array("i")
                            values.frombytes(payload)
                            fds.extend(values)
                    if not data:
                        self.close()
                        return
                    if (
                        flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
                        or len(data) % FRAME.size
                        or len(fds) > 1
                        or (fds and len(data) != FRAME.size)
                    ):
                        raise ValueError("Malformed FastIPC control packet")
                    for op, slot, ident, size in FRAME.iter_unpack(data):
                        if (op in (HELLO, LARGE)) != bool(fds):
                            raise ValueError("Invalid FastIPC descriptor count")
                        self.received(op, slot, ident, size, fds[0] if fds else None)
                finally:
                    for fd in fds:
                        os.close(fd)
        except BlockingIOError:
            pass
        except (OSError, ValueError, struct.error):
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.loop.remove_reader(self.sock.fileno())
        self.loop.remove_writer(self.sock.fileno())
        self.sock.close()
        for _, fd in self.pending:
            if fd is not None:
                os.close(fd)
        self.pending.clear()
        self.disconnected()


def encode_headers(headers):
    return [(k.decode("latin1"), v.decode("latin1")) for k, v in headers]


def decode_headers(headers):
    return [(k.encode("latin1"), v.encode("latin1")) for k, v in headers]


def pack_meta(obj):
    data = json.dumps(obj, ensure_ascii=True, separators=(",", ":")).encode()
    if len(data) > META_LIMIT:
        raise ValueError("FastIPC metadata too large")
    return LENGTH.pack(len(data)) + data


def read_input(region, start, total):
    (size,) = LENGTH.unpack_from(region, start)
    if size > META_LIMIT or size > total - LENGTH.size:
        raise ValueError("Invalid FastIPC metadata size")
    end = start + LENGTH.size + size
    return json.loads(region[start + LENGTH.size : end]), region[end : start + total]


class Stream:
    def __init__(self, slot, ident):
        self.slot, self.ident = slot, ident
        self.messages = asyncio.Queue()
        self.cancelled = False
        self.block = None


class Replica:
    def __init__(self, client, rank, sock, capacity):
        self.client, self.rank = client, rank
        self.free = asyncio.Queue()
        for i in range(capacity):
            self.free.put_nowait(i)
        self.control = asyncio.Queue(maxsize=1)
        self.control.put_nowait(capacity)
        self.capacity = capacity
        self.streams = {}
        self.sequence = 0
        self.ready = asyncio.Event()
        size = pool_size(capacity, client.settings["max_body_bytes"])
        self.arena = InputArena((capacity + 1) * STRIDE, size)
        fd = buffer(client.lib, size)
        try:
            self.pool = mmap.mmap(fd, size)
            self.link = Link(sock, self.received, self.lost)
        except BaseException:
            os.close(fd)
            raise
        self.link.post(HELLO, size=capacity, fd=fd)

    async def submit(self, meta, body, control=False):
        slots = self.control if control else self.free
        if slots.empty():
            slot = await asyncio.wait_for(slots.get(), self.client.settings["queue_timeout_seconds"])
        else:
            slot = slots.get_nowait()
        try:
            if self.link.closed:
                raise ConnectionError("FastIPC worker disconnected")
            return self.send_request(slot, meta, body)
        except BaseException:
            slots.put_nowait(slot)
            raise

    def send_request(self, slot, meta, body):
        total = len(meta) + len(body)
        self.sequence += 1
        s = Stream(slot, self.sequence)
        if total > INPUT:
            s.block = self.arena.take(total)
        if total <= INPUT or s.block:
            start = s.block[0] if s.block else slot * STRIDE
            if s.block:
                OFFSET.pack_into(self.pool, slot * STRIDE, start)
            try:
                self.pool[start : start + len(meta)] = meta
                self.pool[start + len(meta) : start + total] = body
                self.link.post(ARENA if s.block else REQUEST, slot, s.ident, total)
            except BaseException:
                self.consumed(s)
                raise
        else:
            fd = buffer(self.client.lib, total)
            try:
                with mmap.mmap(fd, total) as region:
                    region[: len(meta)] = meta
                    region[len(meta) :] = body
            except BaseException:
                os.close(fd)
                raise
            self.link.post(LARGE, slot, s.ident, total, fd)
        self.streams[slot] = s
        return s

    def consumed(self, s):
        if s.block:
            self.arena.give(s.block)
            s.block = None

    def release(self, s):
        self.consumed(s)
        del self.streams[s.slot]
        slots = self.control if s.slot == self.capacity else self.free
        slots.put_nowait(s.slot)

    def cancel(self, s):
        if self.streams.get(s.slot) is s and not s.cancelled:
            s.cancelled = True
            self.link.post(CANCEL, s.slot, s.ident)

    def received(self, op, slot, ident, size, fd):
        if op == READY:
            self.ready.set()
            return
        s = self.streams.get(slot)
        if s is None or ident != s.ident:
            raise ValueError("Unexpected FastIPC response generation")
        if op == CONSUMED:
            self.consumed(s)
            return
        if op not in (DATA, END, CLOSED, FAILED) or size > OUTPUT:
            raise ValueError("Malformed FastIPC result")
        if not s.cancelled:
            start = slot * STRIDE + INPUT
            s.messages.put_nowait((op, self.pool[start : start + size]))
        if op != DATA:
            self.release(s)

    def lost(self):
        self.client.lib.fastipc_rank_offline(self.client.router, self.rank)
        self.ready.set()
        for s in list(self.streams.values()):
            s.messages.put_nowait((FAILED, b""))
            self.release(s)
        self.pool.close()


class AdmissionError(Exception):
    def __init__(self, code):
        self.status = 429 if code == errno.EAGAIN else 503
        super().__init__(os.strerror(code))


class Client:
    def __init__(self, lib, cfg):
        signatures(lib)
        self.lib = lib
        self.ranks = []
        self.cfg = cfg
        caps = [r["max_inflight"] for r in cfg["replicas"]]
        self.settings = s = cfg["router"]
        self.router = lib.fastipc_router_create(
            (C.c_int * len(caps))(*caps),
            len(caps),
            s["queue_limit_per_replica"],
            s["max_sessions"],
            s["affinity_ttl_seconds"],
        )
        if not self.router:
            raise ValueError("Invalid FastIPC deployment capacity")

    async def connect(self, directory):
        for i, r in enumerate(self.cfg["replicas"]):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                sock.connect(str(directory / str(i)))
                peer = Replica(self, i, sock, r["max_inflight"])
            except BaseException:
                sock.close()
                raise
            self.ranks.append(peer)
            await asyncio.wait_for(peer.ready.wait(), 10)
            if peer.link.closed:
                raise ConnectionError("FastIPC handshake failed")

    @contextmanager
    def reserve(self, key):
        if not self.router:
            raise ConnectionError("FastIPC gateway closed")
        rank = self.lib.fastipc_reserve(self.router, key)
        if rank < 0:
            raise AdmissionError(-rank)
        try:
            yield self.ranks[rank]
        finally:
            if self.router:
                self.lib.fastipc_release(self.router, rank, key)

    def state(self):
        result = []
        for p, r in zip(self.ranks, self.cfg["replicas"]):
            v = (C.c_int * 2)()
            self.lib.fastipc_rank_state(self.router, p.rank, v)
            active = sum(s.slot != p.capacity and not s.cancelled for s in p.streams.values())
            result.append(
                dict(
                    name=r["name"],
                    healthy=not p.link.closed,
                    max_inflight=p.capacity,
                    active=active,
                    queued=v[0] - active,
                    sessions=v[1],
                )
            )
        return result

    def close(self):
        for p in self.ranks:
            p.link.close()
        if self.router:
            self.lib.fastipc_router_destroy(self.router)
            self.router = None
