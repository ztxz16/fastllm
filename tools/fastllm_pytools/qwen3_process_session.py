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

from .tp_process_utils import isolated_process, wait_case
from .tp_graph import validate_graph, validate_allreduce, validate_prefix, validate_prefix_directory
from .tp_load import validate_load

BACKEND = "qwen3_process_generate_cublas_nccl"
EMPTY = dict(active=False, tokens=0, capacity=0, device_bytes=0)


class GenerationRejected(RuntimeError):
    """单个请求被拒绝且 KV 已清理；同一会话可以继续处理下一个请求。"""

    def __init__(self, report):
        self.report = report
        super().__init__(report.get("error", "generation rejected"))


class GenerationCancelled(GenerationRejected):
    """当前请求已在计算边界取消并释放 KV，模型和 worker 继续驻留。"""


class GenerationObserverError(RuntimeError):
    """Token 回调失败；已取消并读完原生结果，确认 KV 为空后保留会话。"""

    def __init__(self, report, error):
        self.report = report
        super().__init__(f"token observer failed: {type(error).__name__}: {error}")


class Qwen3ProcessSession:
    """同步会话；线程调用串行化，超时或协议错误后整组回收，不自动重试请求。"""

    def __init__(self, binary, worker, model, devices, output, *, timeout=1800, trace=False, cuda_graph=False, custom_allreduce=False, prefix_cache=False, environment=None, request_log=True, page_size=-1, cache_tokens=-1):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("session timeout must be finite and positive")
        device_ids = [int(value) for value in devices.split(",")]
        if not device_ids or min(device_ids) < 0 or len(set(device_ids)) != len(device_ids):
            raise ValueError("devices must be distinct nonnegative indices")
        self.binary, self.worker, self.model = (Path(value).resolve() for value in (binary, worker, model))
        self.devices = device_ids
        self.environment = dict(environment) if environment is not None else None
        self.output = Path(output).resolve()
        if type(request_log) is not bool:
            raise ValueError('request_log must be boolean')
        self.request_log = request_log
        if type(cuda_graph) is not bool:
            raise ValueError("cuda_graph must be boolean")
        if type(custom_allreduce) is not bool or (custom_allreduce and len(device_ids) != 2):
            raise ValueError("custom_allreduce requires a boolean and TP=2")
        self.custom_allreduce = custom_allreduce
        if type(prefix_cache) is not bool:
            raise ValueError('prefix_cache must be boolean')
        for name, value in (('page_size', page_size), ('cache_tokens', cache_tokens)):
            if type(value) is not int or (value != -1 and not 1 <= value <= 2147483647):
                raise ValueError(f'{name} must be -1 or a positive int32')
        self.page_size, self.cache_tokens = page_size, cache_tokens
        self.prefix_cache = prefix_cache
        self.cuda_graph = cuda_graph
        self._graph_state = None
        self.timeout, self.trace = timeout, trace
        self.state = "new"
        self.ready = self.shutdown = None
        self._stack = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._cancel_event = None
        self._cancel_sent = False
        self._observer_error = None

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
        requested = self._observer_error is not None or (self._cancel_event is not None and self._cancel_event.is_set())
        if requested and not self._cancel_sent:
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

    def _exchange(self, message, on_event=None, on_load=None):
        data = json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(data) > 1024 * 1024:
            raise ValueError("session request exceeds 1 MiB")
        deadline = time.monotonic() + self.timeout
        self._send_cancel()
        self._socket.settimeout(self.timeout)
        self._socket.sendall(struct.pack("!I", len(data)) + data)
        while True:
            result = self._read(deadline)
            if type(result.get("id")) is not int or result["id"] != message["id"]:
                raise RuntimeError("session response ID mismatch")
            if result.get('event') == 'load':
                if on_load is None:
                    raise RuntimeError('unexpected KV load event')
                on_load(result)
                continue
            if result.get("event") != "token":
                break
            if on_event is None:
                raise RuntimeError("unexpected token event")
            on_event(result)
        self._next_id += 1
        return result

    def _empty(self, cache):
        if cache != [EMPTY for _ in self.devices]:
            raise RuntimeError("session returned nonempty or invalid KV state")

    def _check_graph(self, report, *, reset=False, decodes=None, cleared=False):
        key = "graph_after_unload" if reset else "graph"
        if not self.cuda_graph and not self.custom_allreduce and not self.prefix_cache and key not in report:
            return  # Eager 客户端兼容旧原生报告；Graph 必须具有执行证据。
        expected = "decode_cuda_graph" if self.cuda_graph else "eager"
        if report.get("execution") != expected:
            raise RuntimeError("CUDA Graph execution mode mismatch")
        states = validate_graph(report.get(key), len(self.devices), self.cuda_graph)
        if self.prefix_cache or any('prefix_cache' in state for state in states):
            validate_prefix(states, len(self.devices), self.prefix_cache, previous=self._graph_state,
                            reset=reset or self._graph_state is None, cleared=cleared,
                            prompt_tokens=len(report['request']['token_ids']) if decodes is not None else None)
        if self.custom_allreduce or any('allreduce' in state for state in states):
            ar = validate_allreduce(states, len(self.devices), self.custom_allreduce, previous=self._graph_state,
                                    require_progress=self.custom_allreduce and decodes is not None and decodes > 0)
            if self._graph_state is None and (ar[0]['custom_calls'] or ar[0]['nccl_calls']):
                raise RuntimeError('AllReduce counters not zero at startup')
        counts = ('captures', 'replays', 'warmups', 'eager_decodes')
        if reset or self._graph_state is None:
            if any(states[0][key] for key in counts):
                raise RuntimeError("CUDA Graph state not reset")
        else:
            before, after = self._graph_state[0], states[0]
            if any(after[key] < before[key] for key in counts):
                raise RuntimeError("CUDA Graph counters regressed")
            if decodes is not None:
                key = 'replays' if self.cuda_graph else 'eager_decodes'
                captures = int(decodes > 0) if self.cuda_graph else 0
                if after[key] - before[key] != decodes or after['captures'] - before['captures'] != captures:
                    raise RuntimeError("CUDA Graph decode count mismatch")
        self._graph_state = copy.deepcopy(states)

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

    def start(self):
        """/**
         * 创建原生进程并立即返回，让多个 DP 副本的权重加载可以重叠。
         * 会话持有进程组和通信资源；调用者随后必须等待就绪或关闭会话。
         * 仅允许从 new 启动一次，启动异常会回收已创建的资源。
         * @return 处于 starting 状态的当前会话；失败通过异常报告。
         */"""
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
                if self.page_size > 0:
                    command += ["--page-size", str(self.page_size)]
                if self.cache_tokens > 0:
                    command += ["--tokens", str(self.cache_tokens)]
                if self.cuda_graph:
                    command += ["--cuda-graph", "1"]
                if self.custom_allreduce:
                    command += ["--custom-allreduce", "1"]
                if self.prefix_cache:
                    command += ["--prefix-cache", "1"]
                (self.output / "command.txt").write_text(shlex.join(command) + "\n")
                stdout = self._stack.enter_context((self.output / "native.stdout.log").open("wb"))
                stderr = self._stack.enter_context((self.output / "native.stderr.log").open("wb"))
                self._process = self._stack.enter_context(isolated_process(
                    command, pass_fds=(child.fileno(), cancel_child.fileno()), stdout=stdout, stderr=stderr,
                    env=self.environment))
                child.close()
                cancel_child.close()
                return self
            except BaseException:
                self._dispose("failed")
                raise

    def wait_ready(self):
        """/**
         * 等待并校验原生加载结果，确认 worker、KV 和执行后端后开放请求。
         * 仅接受 starting 状态；超时从本次等待开始计时，异常会回收整组进程。
         * @return 处于 ready 状态的当前会话；失败通过异常报告。
         */"""
        with self._lock:
            if self.state != "starting":
                raise RuntimeError(f"cannot wait for startup: session is {self.state}")
            try:
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
                self._check_graph(self.ready)
                self.load_state = None
                self.load_observer = None
                if type(self.ready.get('load_stats')) is int and self.ready['load_stats'] == 1:
                    self.load_state = validate_load(self.ready.get('load'), len(self.devices), idle=True)
                (self.output / "ready.json").write_text(json.dumps(self.ready, ensure_ascii=False, indent=2))
                self.state = "ready"
                return self
            except BaseException:
                self._dispose("failed")
                raise

    def __enter__(self):
        self.start()
        return self.wait_ready()

    def generate(self, request, *, cancel_event=None, on_token=None):
        """同步生成；on_token 必须快速返回，回调异常会取消并排空结果后保留已清理的会话。"""
        if not self._lock.acquire(timeout=self.timeout):
            raise TimeoutError("session busy: waiting for previous request timed out")
        try:
            if self.state != "ready":
                raise RuntimeError(f"session is {self.state}")
            if cancel_event is not None and self.ready.get("cancellation") != "request_id_datagram":
                raise RuntimeError("native session does not support cancellation; rebuild fastllm_tp_generate")
            if on_token is not None and (not callable(on_token) or self.ready.get("token_events") is not True or
                                        self.ready.get("cancellation") != "request_id_datagram"):
                raise ValueError("token callback requires native token events and cancellation support")
            # 先检查本地序列化；尚未提交的非法输入不能破坏正在复用的会话。
            request = json.loads(json.dumps(request, allow_nan=False))
            message = dict(id=self._next_id, op="generate", request=request)
            observer = self.load_observer
            if observer is not None:
                message['stream_load'] = True
            if on_token is not None:
                message["stream_tokens"] = True
            if len(json.dumps(message, ensure_ascii=False).encode("utf-8")) > 1024 * 1024:
                raise ValueError("session request exceeds 1 MiB")
            path = self.output / f"request-{self._next_id:06d}"
            if self.request_log:
                path.with_suffix(".json").write_text(json.dumps(request, ensure_ascii=False, indent=2))
            tokens = []
            loads = []

            def on_load(event):
                if (type(event.get('sequence')) is not int or event['sequence'] != len(loads) or
                        len(loads) >= 4096 or event.get('phase') not in ('running', 'complete') or
                        (loads and loads[-1]['phase'] == 'complete')):
                    raise RuntimeError('invalid or out-of-order KV load event')
                state = validate_load(event, len(self.devices), idle=event['phase'] == 'complete',
                                      running=event['phase'] == 'running', previous=self.load_state)
                self.load_state = state
                loads.append(dict(state, sequence=event['sequence'], phase=event['phase']))
                observer(copy.deepcopy(state))

            def on_event(event):
                if observer is not None and (not loads or loads[-1]['phase'] != 'running'):
                    raise RuntimeError('token event outside running KV load phase')
                token = event.get("token_id")
                if (type(event.get("index")) is not int or event["index"] != len(tokens) or
                        type(token) is not int or not 0 <= token <= 4294967295 or
                        len(tokens) >= request.get("max_new_tokens", 32) or len(tokens) >= 2048 or
                        (tokens and tokens[-1] in request.get("eos_token_ids", []))):
                    raise RuntimeError("invalid or out-of-order token event")
                tokens.append(token)
                if self._observer_error is None:
                    try:
                        on_token(token)
                    except Exception as error:
                        self._observer_error = error
                        self._send_cancel()

            try:
                self._cancel_event, self._cancel_sent = cancel_event, False
                result = self._exchange(message, on_event if on_token is not None else None,
                                        on_load if observer is not None else None)
                if observer is not None:
                    if (not loads or loads[-1]['phase'] != 'complete' or
                            (result.get('status') == 'PASS' and len(loads) < 2)):
                        raise RuntimeError(f'missing final KV load statistics: {result.get("error", "invalid report")}')
                    result['load_samples'] = loads
                if self.request_log:
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
                self._check_graph(result, decodes=len(result["token_ids"]) - 1 if result["status"] == "PASS" else None)
                if on_token is not None:
                    expected = result.get("token_ids") if result["status"] == "PASS" else [step["token_id"] for step in result.get("steps", [])]
                    if (not isinstance(expected, list) or any(type(token) is not int for token in expected) or
                            tokens != expected):
                        raise RuntimeError("token events do not match final result")
            except BaseException:
                self._dispose("failed")
                raise
            if self._observer_error is not None:
                raise GenerationObserverError(result, self._observer_error) from self._observer_error
            if result["status"] == "FAIL":
                if result.get("cancelled") is True:
                    raise GenerationCancelled(result)
                raise GenerationRejected(result)
            return result
        finally:
            self._cancel_event = None
            self._observer_error = None
            self._lock.release()

    def pd_command(self, request):
        """/** 调用原生 PD 存储或计算入口；只供持有副本调度权的本地交接服务使用。 */"""
        with self._lock:
            if self.state != "ready" or self.ready.get("pd_storage_protocol") != 1:
                raise RuntimeError("native worker does not support PD storage protocol 1")
            try:
                result = self._exchange({"id": self._next_id, "op": "pd", "request": request})
                if result.get("event") != "pd_result" or result.get("status") != "PASS":
                    raise RuntimeError("invalid native PD response")
                return result["result"]
            except BaseException:
                self._dispose("failed")
                raise

    def prefix_directory(self):
        """/** 返回最近一次已校验的原生目录副本；调用方独占副本时使用，不查询GPU或推测缓存。 */"""
        if self.state != 'ready' or not self.prefix_cache or not self._graph_state:
            raise RuntimeError('prefix directory requires a ready prefix-cache session')
        directory = validate_prefix_directory(self._graph_state[0]['prefix_cache'], required=True)
        return copy.deepcopy(directory)

    def clear_prefix_cache(self):
        """串行等待当前请求结束后清空 GPU 快照，保留模型、规约器及累计计数。"""
        if not self._lock.acquire(timeout=self.timeout):
            raise TimeoutError('session busy: prefix cache clear timed out')
        try:
            if self.state != 'ready':
                raise RuntimeError(f'session is {self.state}')
            try:
                result = self._exchange(dict(id=self._next_id, op='clear_prefix_cache'))
                if result.get('event') != 'prefix_cache_cleared' or result.get('status') != 'PASS':
                    raise RuntimeError('prefix cache clear failed')
                self._empty(result.get('cache'))
                self._check_graph(result, cleared=True)
                (self.output / f'clear-{result["id"]:06d}.json').write_text(json.dumps(result, indent=2))
                return result
            except BaseException:
                self._dispose('failed')
                raise
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            if self.state in ("closed", "failed"):
                return
            if self.state in ("new", "starting"):
                # 尚未完成 ready 握手，不能发送 shutdown 或继续等待权重加载。
                self._dispose("closed")
                return
            try:
                self.shutdown = self._exchange(dict(id=self._next_id, op="shutdown"))
                if (self.shutdown.get("event") != "shutdown" or self.shutdown.get("status") != "PASS" or
                    self.shutdown.get("shutdown_complete") is not True):
                    raise RuntimeError("session shutdown failed")
                self._empty(self.shutdown.get("cache_after_unload"))
                self._check_graph(self.shutdown, reset=True)
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
