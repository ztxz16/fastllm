"""Qwen3 常驻多进程 TP 客户端：串行复用权重，每个请求独立管理 KV。"""

from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
import select
import shlex
import socket
import struct
import threading
import time

from tp_process_utils import isolated_process, wait_case

BACKEND = "qwen3_process_generate_cublas_nccl"
EMPTY = dict(active=False, tokens=0, capacity=0, device_bytes=0)


class GenerationRejected(RuntimeError):
    """单个请求被拒绝且 KV 已清理；同一会话可以继续处理下一个请求。"""

    def __init__(self, report):
        self.report = report
        super().__init__(report.get("error", "generation rejected"))


class GenerationCancelled(GenerationRejected):
    """当前请求已在计算边界取消并释放 KV，模型和 worker 继续驻留。"""


class Qwen3ProcessSession:
    """同步会话；线程调用串行化，超时或协议错误后整组回收，不自动重试请求。"""

    def __init__(self, binary, worker, model, devices, output, *, timeout=1800, trace=False):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("session timeout must be finite and positive")
        device_ids = [int(value) for value in devices.split(",")]
        if not device_ids or min(device_ids) < 0 or len(set(device_ids)) != len(device_ids):
            raise ValueError("devices must be distinct nonnegative indices")
        self.binary, self.worker, self.model = (Path(value).resolve() for value in (binary, worker, model))
        self.devices = device_ids
        self.output = Path(output).resolve()
        self.timeout, self.trace = timeout, trace
        self.state = "new"
        self.ready = self.shutdown = None
        self._stack = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._cancel_event = None
        self._cancel_sent = False

    def _receive(self, size, deadline):
        chunks = bytearray()
        while len(chunks) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("session response deadline exceeded")
            self._send_cancel()
            if not select.select([self._socket], [], [], min(0.05, remaining))[0]:
                continue
            self._socket.settimeout(remaining)
            chunk = self._socket.recv(min(65536, size - len(chunks)))
            if not chunk:
                raise RuntimeError(f"session disconnected; native logs: {self.output}")
            chunks.extend(chunk)
        return chunks

    def _send_cancel(self):
        if self._cancel_event is not None and self._cancel_event.is_set() and not self._cancel_sent:
            self._cancel_socket.send(struct.pack("!I", self._next_id))
            self._cancel_sent = True

    def _read(self, deadline):
        size = struct.unpack("!I", self._receive(4, deadline))[0]
        if not 0 < size <= 512 * 1024 * 1024:
            raise RuntimeError("invalid session response size")
        result = json.loads(self._receive(size, deadline))
        if not isinstance(result, dict):
            raise RuntimeError("session response must be object")
        return result

    def _exchange(self, message):
        data = json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(data) > 1024 * 1024:
            raise ValueError("session request exceeds 1 MiB")
        deadline = time.monotonic() + self.timeout
        self._send_cancel()
        self._socket.settimeout(self.timeout)
        self._socket.sendall(struct.pack("!I", len(data)) + data)
        result = self._read(deadline)
        if type(result.get("id")) is not int or result["id"] != message["id"]:
            raise RuntimeError("session response ID mismatch")
        self._next_id += 1
        return result

    def _empty(self, cache):
        if cache != [EMPTY for _ in self.devices]:
            raise RuntimeError("session returned nonempty or invalid KV state")

    def _dispose(self, state):
        self.state = "closing" if state == "closed" else state
        try:
            if self._stack is not None:
                stack, self._stack = self._stack, None
                stack.close()
        except BaseException:
            self.state = "failed"
            raise
        self.state = state

    def __enter__(self):
        with self._lock:
            if self.state != "new":
                raise RuntimeError("session cannot be restarted")
            self.state = "starting"
            self._stack = ExitStack()
            try:
                self.output.mkdir(parents=True, exist_ok=False)
                parent, child = socket.socketpair()
                self._socket = parent
                self._stack.callback(parent.close)
                self._stack.callback(child.close)
                cancel_parent, cancel_child = socket.socketpair(type=socket.SOCK_DGRAM)
                self._cancel_socket = cancel_parent
                cancel_parent.setblocking(False)
                self._stack.callback(cancel_parent.close)
                self._stack.callback(cancel_child.close)
                command = [str(self.binary), "--worker", str(self.worker), "--model", str(self.model),
                           "--devices", ",".join(map(str, self.devices)), "--session-fd", str(child.fileno()),
                           "--trace-logits", "1" if self.trace else "0", "--cancel-fd", str(cancel_child.fileno())]
                (self.output / "command.txt").write_text(shlex.join(command) + "\n")
                stdout = self._stack.enter_context((self.output / "native.stdout.log").open("wb"))
                stderr = self._stack.enter_context((self.output / "native.stderr.log").open("wb"))
                self._process = self._stack.enter_context(isolated_process(
                    command, pass_fds=(child.fileno(), cancel_child.fileno()), stdout=stdout, stderr=stderr))
                child.close()
                cancel_child.close()
                self.ready = self._read(time.monotonic() + self.timeout)
                if (self.ready.get("event") != "ready" or self.ready.get("status") != "PASS" or
                    self.ready.get("protocol") != 1 or self.ready.get("backend") != BACKEND or
                    self.ready.get("model_load_count") != 1):
                    raise RuntimeError("invalid session startup response")
                workers = self.ready.get("workers", [])
                if (len(workers) != len(self.devices) or
                    any(w.get("rank") != rank or w.get("device") != device or
                        type(w.get("pid")) is not int or w["pid"] <= 0
                        for rank, (w, device) in enumerate(zip(workers, self.devices))) or
                    len({w["pid"] for w in workers}) != len(workers)):
                    raise RuntimeError("invalid session workers")
                self._workers = copy.deepcopy(workers)
                self._empty(self.ready.get("cache"))
                (self.output / "ready.json").write_text(json.dumps(self.ready, ensure_ascii=False, indent=2))
                self.state = "ready"
                return self
            except BaseException:
                self._dispose("failed")
                raise

    def generate(self, request, *, cancel_event=None):
        if not self._lock.acquire(timeout=self.timeout):
            raise TimeoutError("session busy: waiting for previous request timed out")
        try:
            if self.state != "ready":
                raise RuntimeError(f"session is {self.state}")
            if cancel_event is not None and self.ready.get("cancellation") != "request_id_datagram":
                raise RuntimeError("native session does not support cancellation; rebuild fastllm_tp_generate")
            # 先检查本地序列化；尚未提交的非法输入不能破坏正在复用的会话。
            request = json.loads(json.dumps(request, allow_nan=False))
            message = dict(id=self._next_id, op="generate", request=request)
            if len(json.dumps(message, ensure_ascii=False).encode("utf-8")) > 1024 * 1024:
                raise ValueError("session request exceeds 1 MiB")
            path = self.output / f"request-{self._next_id:06d}"
            path.with_suffix(".json").write_text(json.dumps(request, ensure_ascii=False, indent=2))
            try:
                self._cancel_event, self._cancel_sent = cancel_event, False
                result = self._exchange(message)
                path.with_suffix(".response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                if (result.get("event") != "result" or result.get("backend") != BACKEND or
                    result.get("workers") != self._workers or result.get("reusable") is not True):
                    raise RuntimeError(f"session failed: {result.get('error', 'invalid response')}")
                if result.get("status") == "FAIL":
                    self._empty(result.get("cache_after_error"))
                elif result.get("status") == "PASS":
                    self._empty(result.get("cache_after_generate"))
                    ids = result.get("token_ids")
                    if (result.get("request") != request or not isinstance(ids, list) or not ids or
                        any(type(token) is not int or token < 0 for token in ids) or
                        len(ids) > request.get("max_new_tokens", 32) or
                        result.get("finish_reason") not in ("eos", "length")):
                        raise RuntimeError("invalid session generated output")
                    eos = request.get("eos_token_ids", [])
                    finish = "eos" if ids[-1] in eos else "length"
                    if (result["finish_reason"] != finish or any(token in eos for token in ids[:-1]) or
                        (finish == "length" and len(ids) != request.get("max_new_tokens", 32))):
                        raise RuntimeError("invalid session stopping condition")
                else:
                    raise RuntimeError("invalid session result status")
            except BaseException:
                self._dispose("failed")
                raise
            if result["status"] == "FAIL":
                if result.get("cancelled") is True:
                    raise GenerationCancelled(result)
                raise GenerationRejected(result)
            return result
        finally:
            self._cancel_event = None
            self._lock.release()

    def close(self):
        with self._lock:
            if self.state in ("closed", "failed"):
                return
            if self.state == "new":
                self._dispose("closed")
                return
            try:
                self.shutdown = self._exchange(dict(id=self._next_id, op="shutdown"))
                if (self.shutdown.get("event") != "shutdown" or self.shutdown.get("status") != "PASS" or
                    self.shutdown.get("shutdown_complete") is not True):
                    raise RuntimeError("session shutdown failed")
                self._empty(self.shutdown.get("cache_after_unload"))
                if wait_case(self._process, min(30, self.timeout)) != 0:
                    raise RuntimeError("session exited with failure")
                self._dispose("closed")
                (self.output / "shutdown.json").write_text(json.dumps(self.shutdown, indent=2))
            except BaseException:
                self._dispose("failed")
                raise

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            with self._lock:
                self._dispose("failed")
        else:
            self.close()
