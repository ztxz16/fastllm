"""One blocking native output reader per model, shared by asynchronous streams."""
import asyncio
import atexit
import ctypes
import threading
from collections import defaultdict, deque

_readers = {}
_registry_lock = threading.Lock()
_batch_tokens = 64
_symbols = ("get_response_generation_llm_model", "abort_response_generation_llm_model",
            "wake_response_output_llm_model", "fetch_responses_llm_model")


def _configure(lib):
    if not all(hasattr(lib, name) for name in _symbols):
        return False
    integer = ctypes.c_int
    pointer = ctypes.POINTER(integer)
    generation = ctypes.c_uint64
    lib.get_response_generation_llm_model.argtypes = [integer, integer]
    lib.get_response_generation_llm_model.restype = generation
    lib.abort_response_generation_llm_model.argtypes = [integer, integer, generation]
    lib.abort_response_generation_llm_model.restype = None
    lib.wake_response_output_llm_model.argtypes = [integer]
    lib.wake_response_output_llm_model.restype = None
    lib.fetch_responses_llm_model.argtypes = [integer, pointer, ctypes.POINTER(generation),
        integer, integer, pointer, integer, pointer, integer, integer, ctypes.POINTER(generation)]
    lib.fetch_responses_llm_model.restype = integer
    return True


class OutputStream:
    def __init__(self, reader, handle):
        self.reader, self.handle = reader, handle
        self.loop = asyncio.get_running_loop()
        # At most one queued batch and the batch currently being decoded/sent.
        self.queue = asyncio.Queue(maxsize=1)
        self.generation = 0
        self.enabled = True  # Only the collector thread changes credits.
        self.cancelled = False
        self.terminal = None
        reader.post("add", self)

    async def read(self):
        if self.terminal is not None:
            status, statistics = self.terminal
            self.terminal = None
            return status, [], statistics
        try:
            packet = await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return 0, [], (-1, -1, -1)
        if isinstance(packet, BaseException):
            raise packet
        tokens, statistics, status = packet
        if status and tokens:
            self.terminal = (status, statistics)
        elif not status:
            self.reader.post("credit", self)
        return (len(tokens) if tokens else status), tokens, statistics

    def close(self):
        if not self.cancelled:
            self.cancelled = True
            self.reader.post("remove", self)


class _Reader:
    def __init__(self, lib, model):
        self.lib, self.model = lib, model
        self.condition = threading.Condition()
        self.commands = deque()
        self.stopping = False
        self.stopped = False
        self.in_read = False
        self.wake_scheduled = False
        self.thread = threading.Thread(target=self.run, name="fastllm-output", daemon=True)
        self.thread.start()

    def post(self, command, stream):
        with self.condition:
            if self.stopping or self.stopped:
                if command == "add":
                    raise RuntimeError("Model output reader is closed")
                return
            self.commands.append((command, stream))
            self.condition.notify()
            if not self.wake_scheduled:
                self.wake_scheduled = True
                # All consumers awakened in one event-loop turn share this control wake.
                try:
                    stream.loop.call_soon_threadsafe(self.wake)
                except RuntimeError:
                    self.wake_scheduled = False
                    if self.in_read:
                        self.lib.wake_response_output_llm_model(self.model)

    def wake(self):
        with self.condition:
            self.wake_scheduled = False
            if self.in_read and not self.stopping and not self.stopped:
                self.lib.wake_response_output_llm_model(self.model)

    @staticmethod
    def deliver(batch):
        for stream, packet in batch:
            if stream.cancelled:
                continue
            if isinstance(packet, BaseException) and stream.queue.full():
                stream.queue.get_nowait()
            stream.queue.put_nowait(packet)

    def dispatch(self, packets):
        loops = defaultdict(list)
        for stream, packet in packets:
            loops[stream.loop].append((stream, packet))
        for loop, batch in loops.items():
            try:
                loop.call_soon_threadsafe(self.deliver, batch)
            except RuntimeError:
                # A closed event loop can no longer consume its request.
                for stream, _ in batch:
                    stream.cancelled = True
                    self.post("remove", stream)

    def run(self):
        entries = {}
        version = ctypes.c_uint64(0)
        capacity = 0
        failure = RuntimeError("Model output reader was closed")
        try:
            while True:
                with self.condition:
                    while not self.commands and not self.stopping and not any(
                            stream.enabled for stream in entries.values()):
                        self.condition.wait()
                    if self.stopping:
                        break
                    commands = list(self.commands)
                    self.commands.clear()
                for command, stream in commands:
                    if command == "add":
                        if stream.handle in entries:
                            self.dispatch([(stream, RuntimeError("Request already has an output consumer"))])
                            continue
                        stream.generation = self.lib.get_response_generation_llm_model(self.model, stream.handle)
                        if stream.generation:
                            entries[stream.handle] = stream
                        else:
                            self.dispatch([(stream, ([], (-1, -1, -1), -1))])
                    elif entries.get(stream.handle) is stream:
                        if command == "credit":
                            stream.enabled = True
                        else:
                            self.lib.abort_response_generation_llm_model(
                                self.model, stream.handle, stream.generation)
                            del entries[stream.handle]
                active = [stream for stream in entries.values() if stream.enabled]
                count = len(active)
                if not count:
                    continue
                if count > capacity:
                    capacity = max(count, capacity * 2)
                    tokens = (ctypes.c_int * (capacity * _batch_tokens))()
                    records = (ctypes.c_int * (capacity * 7))()
                    handles = (ctypes.c_int * capacity)()
                    generations = (ctypes.c_uint64 * capacity)()
                for i, stream in enumerate(active):
                    handles[i], generations[i] = stream.handle, stream.generation
                with self.condition:
                    if self.stopping or self.commands:
                        continue
                    self.in_read = True
                try:
                    ready = self.lib.fetch_responses_llm_model(self.model, handles, generations,
                        count, _batch_tokens, tokens, len(tokens), records, capacity, 1000,
                        ctypes.byref(version))
                finally:
                    with self.condition:
                        self.in_read = False
                if ready < 0:
                    raise RuntimeError("Native model output reader stopped (%d)" % ready)
                packets = []
                for i in range(ready):
                    row, offset, length, status, cached, missed, generated = records[i*7:i*7+7]
                    stream = active[row]
                    stream.enabled = False
                    if status:
                        del entries[stream.handle]
                    packets.append((stream, (tokens[offset:offset+length],
                        (cached, missed, generated), status)))
                self.dispatch(packets)
        except BaseException as error:
            failure = error
        finally:
            with self.condition:
                self.in_read = False
                self.stopped = True
                pending = [stream for command, stream in self.commands if command == "add"]
                self.commands.clear()
            self.dispatch([(stream, failure) for stream in list(entries.values()) + pending])

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
            if self.in_read:
                self.lib.wake_response_output_llm_model(self.model)
        self.thread.join()


def open_output_stream(lib, model, handle):
    """Return None for an older core, preserving its per-request fetch implementation."""
    key = (id(lib), model)
    with _registry_lock:
        reader = _readers.get(key)
        if reader is None:
            if not _configure(lib):
                return None
            reader = _readers[key] = _Reader(lib, model)
        return OutputStream(reader, handle)


def close_output_reader(lib, model):
    with _registry_lock:
        reader = _readers.pop((id(lib), model), None)
    if reader is not None:
        reader.close()


def _close_all():
    with _registry_lock:
        readers = list(_readers.values())
        _readers.clear()
    for reader in readers:
        reader.close()


atexit.register(_close_all)
