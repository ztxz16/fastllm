#!/usr/bin/env python3
"""Qwen3 多进程 TP HTTP 入口：有界 FIFO、SSE/非流式聊天、协作取消；默认仅监听本机。"""

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import re
import select
import shlex
import signal
import socket
import sys
import threading
import time
import uuid

from qwen3_process_chat import ROOT, prepare_request
from qwen3_process_session import BACKEND, GenerationCancelled, GenerationObserverError, GenerationRejected, Qwen3ProcessSession


class HTTPError(ValueError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def error_body(message):
    return dict(error=dict(message=str(message), type="request_error"))


def validate_payload(payload, model_name):
    """只接受明确支持的聊天语义；未知参数、工具和多模态均显式拒绝。"""
    if not isinstance(payload, dict):
        raise HTTPError(400, "request must be a JSON object")
    fields = {"model", "messages", "max_tokens", "temperature", "top_p", "top_k", "seed", "stream", "stream_options"}
    if set(payload) - fields:
        raise HTTPError(400, "unsupported fields: " + ", ".join(sorted(set(payload) - fields)))
    if payload.get("model") != model_name:
        raise HTTPError(404, "unknown model")
    if type(payload.get("stream", False)) is not bool:
        raise HTTPError(400, "stream must be boolean")
    if "stream_options" in payload:
        options = payload["stream_options"]
        if (not payload.get("stream") or not isinstance(options, dict) or
                set(options) - {"include_usage"} or type(options.get("include_usage", False)) is not bool):
            raise HTTPError(400, "stream_options only supports include_usage:boolean with stream=true")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > 64:
        raise HTTPError(400, "messages must contain 1..64 text messages")
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or set(message) != {"role", "content"} or
                message["role"] not in ("system", "user", "assistant") or
                not isinstance(message["content"], str) or not message["content"].strip()):
            raise HTTPError(400, "messages require role and nonempty text content")
        if message["role"] == "system" and index != 0:
            raise HTTPError(400, "system message is only allowed first")
        try:
            message["content"].encode("utf-8")
        except UnicodeError as error:
            raise HTTPError(400, "message content must be valid Unicode") from error
    dialogue = messages[1:] if messages[0]["role"] == "system" else messages
    if (not dialogue or dialogue[-1]["role"] != "user" or
            any(message["role"] != ("user" if i % 2 == 0 else "assistant") for i, message in enumerate(dialogue))):
        raise HTTPError(400, "messages must alternate user/assistant and end with user")
    for key, default, minimum, maximum in (("max_tokens", 32, 1, 2048), ("top_k", 0, 0, 4294967295),
                                            ("seed", 0, 0, 4294967295)):
        value = payload.get(key, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise HTTPError(400, f"invalid {key}")
    for key, default in (("temperature", 0.0), ("top_p", 1.0)):
        value = payload.get(key, default)
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise HTTPError(400, f"invalid {key}")
        if (key == "temperature" and value < 0) or (key == "top_p" and not 0 < value <= 1):
            raise HTTPError(400, f"invalid {key}")


@dataclass
class Job:
    id: str
    payload: dict
    deadline: float
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    cancel_status: int = 499
    cancel_reason: str = "request cancelled"
    status: int = 0
    response: dict = field(default_factory=dict)
    native_id: int | None = None
    created: int = field(default_factory=lambda: int(time.time()))
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=64))


class IncrementalTextDecoder:
    """复用同一 tokenizer；保留 UTF-8 未完成字节产生的尾部替换字符，禁止修改已发送前缀。"""

    def __init__(self, tokenizer):
        self.tokenizer, self.ids, self.text = tokenizer, [], ""

    def push(self, token):
        self.ids.append(token)
        return self._decode(final=False)

    def finish(self):
        return self._decode(final=True)

    def _decode(self, final):
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not final:
            text = text.rstrip("\ufffd")
        if not text.startswith(self.text):
            raise ValueError("tokenizer changed an already emitted text prefix")
        delta, self.text = text[len(self.text):], text
        return delta


class ChatService:
    """唯一消费线程持有会话与 tokenizer；HTTP 线程只入队、查询或设置取消标志。"""

    def __init__(self, session, tokenizer, model, model_name, output, *, queue_size=8, request_timeout=120):
        if (type(queue_size) is not int or queue_size < 1 or
                not math.isfinite(request_timeout) or request_timeout <= 0):
            raise ValueError("queue_size and request_timeout must be positive")
        if session.ready.get("cancellation") != "request_id_datagram":
            raise ValueError("native cancellation support required; rebuild fastllm_tp_generate")
        if session.ready.get("token_events") is not True:
            raise ValueError("native token events required; rebuild fastllm_tp_generate")
        self.session, self.tokenizer = session, tokenizer
        self.model, self.model_name, self.output = Path(model), model_name, Path(output)
        self.queue_size, self.request_timeout = queue_size, request_timeout
        self._condition = threading.Condition()
        self._queue, self._jobs = deque(), {}
        self._active = None
        self._stopping = False
        self.failure = None
        self._thread = threading.Thread(target=self._run, name="qwen3-http-scheduler")
        self._thread.start()

    def submit(self, payload, request_id=None):
        validate_payload(payload, self.model_name)
        request_id = request_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", request_id):
            raise HTTPError(400, "X-Request-ID must contain 1..64 letters, digits, _, . or -")
        # 生成内部唯一日志名；用户重复使用已完成的外部 ID 不会覆盖历史记录。
        job = Job(request_id, json.loads(json.dumps(payload, allow_nan=False)), time.monotonic() + self.request_timeout)
        with self._condition:
            if self._stopping or self.failure is not None or self.session.state != "ready":
                raise HTTPError(503, "model service unavailable")
            if request_id in self._jobs:
                raise HTTPError(409, "request ID is already active or queued")
            if len(self._queue) >= self.queue_size:
                raise HTTPError(429, "request queue is full")
            self._jobs[request_id] = job
            self._queue.append(job)
            self._condition.notify()
        return job

    def _finish(self, job, status, response):
        # 调用者持有 condition；先落盘再唤醒 HTTP，不能将日志失败伪装为成功。
        try:
            path = self.output / ("http-" + uuid.uuid4().hex + ".json")
            path.write_text(json.dumps(dict(id=job.id, native_id=job.native_id, request=job.payload, status=status, response=response),
                                       ensure_ascii=False, allow_nan=False, indent=2))
        except Exception as error:
            self.failure = f"cannot persist HTTP result: {error}"
            status, response = 503, error_body("cannot persist request result")
        job.status, job.response = status, response
        self._jobs.pop(job.id, None)
        job.done.set()

    def cancel(self, request_id, status=499, reason="request cancelled", *, expected_job=None):
        with self._condition:
            job = self._jobs.get(request_id)
            if job is None or (expected_job is not None and job is not expected_job):
                return False
            if not job.cancel.is_set():
                job.cancel_status, job.cancel_reason = status, reason
                job.cancel.set()
            if job is not self._active:
                self._queue.remove(job)
                self._finish(job, job.cancel_status, error_body(job.cancel_reason))
            return True

    def health(self):
        with self._condition:
            ready = not self._stopping and self.failure is None and self.session.state == "ready"
            return dict(status="ready" if ready else "unavailable", backend=BACKEND, model=self.model_name,
                        active_request_id=self._active.id if self._active else None,
                        queued_request_ids=[job.id for job in self._queue], queue_capacity=self.queue_size,
                        workers=self.session.ready["workers"], model_load_count=1)

    def _emit(self, job, text):
        # 回调不能阻塞原生事件读取；慢客户端取消当前请求并继续读取原生取消确认。
        if not text or job.cancel.is_set():
            return
        try:
            job.events.put_nowait(text)
        except queue.Full:
            self.cancel(job.id, 503, "stream consumer too slow", expected_job=job)

    def _generate(self, job):
        payload = job.payload
        request = prepare_request(self.model, self.tokenizer, None, messages=payload["messages"],
            max_new_tokens=payload.get("max_tokens", 32), temperature=payload.get("temperature", 0.0),
            top_k=payload.get("top_k", 0), top_p=payload.get("top_p", 1.0), seed=payload.get("seed", 0))
        if job.cancel.is_set() or time.monotonic() >= job.deadline:
            raise HTTPError(job.cancel_status if job.cancel.is_set() else 504,
                            job.cancel_reason if job.cancel.is_set() else "request deadline exceeded")
        timer = threading.Timer(max(0, job.deadline - time.monotonic()), self.cancel,
                                args=(job.id, 504, "request deadline exceeded"), kwargs=dict(expected_job=job))
        decoder = IncrementalTextDecoder(self.tokenizer) if payload.get("stream") else None
        timer.start()
        try:
            report = self.session.generate(request, cancel_event=job.cancel,
                on_token=(lambda token: self._emit(job, decoder.push(token))) if decoder else None)
            job.native_id = report["id"]
        except (GenerationRejected, GenerationObserverError) as error:
            job.native_id = error.report["id"]
            raise
        finally:
            timer.cancel()
            timer.join()
        if decoder:
            try:
                self._emit(job, decoder.finish())
            except ValueError as error:
                raise HTTPError(500, f"final text decoding failed: {error}") from error
            text = decoder.text
            if job.cancel.is_set():
                # 取消期间可能丢弃过增量；即使原生恰好已完成，也不能发送成功 DONE。
                raise HTTPError(job.cancel_status, job.cancel_reason)
        else:
            text = self.tokenizer.decode(report["token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return dict(id="chatcmpl-" + job.id, object="chat.completion", created=job.created, model=self.model_name,
                    choices=[dict(index=0, message=dict(role="assistant", content=text),
                                  finish_reason="stop" if report["finish_reason"] == "eos" else "length")],
                    usage=dict(prompt_tokens=len(request["token_ids"]), completion_tokens=len(report["token_ids"]),
                               total_tokens=len(request["token_ids"]) + len(report["token_ids"])))

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._queue or self._stopping)
                    if self._stopping:
                        break
                    job = self._queue.popleft()
                    self._active = job
                try:
                    if self.failure is not None:
                        raise HTTPError(503, "model service unavailable")
                    if time.monotonic() >= job.deadline:
                        raise HTTPError(504, "request deadline exceeded in queue")
                    response, status = self._generate(job), 200
                except GenerationCancelled:
                    status, response = job.cancel_status, error_body(job.cancel_reason)
                except GenerationObserverError as error:
                    status, response = 500, error_body(error)
                except HTTPError as error:
                    status, response = error.status, error_body(error)
                except (GenerationRejected, ValueError) as error:
                    if self.session.state != "ready":
                        self.failure = str(error)
                        status, response = 503, error_body("native session failed; inspect server logs")
                    else:
                        status, response = 400, error_body(error)
                except Exception as error:
                    self.failure = str(error)
                    status, response = 503, error_body("generation failed; inspect server logs")
                with self._condition:
                    self._finish(job, status, response)
                    self._active = None
        except BaseException as error:
            self.failure = str(error)
        finally:
            with self._condition:
                self._stopping = True
                for job in list(self._jobs.values()):
                    self._finish(job, 503, error_body("model service stopped"))
                self._queue.clear()
                self._active = None
            try:
                self.session.close()
            except BaseException as error:
                self.failure = f"session cleanup failed: {error}"

    def close(self):
        with self._condition:
            self._stopping = True
            for request_id in list(self._jobs):
                self.cancel(request_id, 503, "server shutting down")
            self._condition.notify_all()
        # 取消不强行打断 NCCL；会话总超时负责失败时回收整个进程组。
        self._thread.join(self.session.timeout + 40)
        if self._thread.is_alive():
            raise RuntimeError("scheduler did not stop within session timeout")
        if self.failure is not None or self.session.state != "closed":
            raise RuntimeError(self.failure or "session did not close cleanly")


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass  # 每个已提交请求单独保存 JSON，避免并发 stderr 混入文本输出。

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def _reply(self, status, body, request_id=None):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        if request_id:
            self.send_header("X-Request-ID", request_id)
        if status == 429:
            self.send_header("Retry-After", "1")
        self.end_headers()
        self.wfile.write(data)

    def _disconnected(self):
        if not select.select([self.connection], [], [], 0)[0]:
            return False
        # 不支持流水化/复用连接；请求读完后客户端再发数据也视为放弃当前请求。
        self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        return True

    def _sse_data(self, value):
        data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
        self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def _stream_reply(self, service, job):
        """逐条发送增量；成功结束必须等原生结果/KV 清理和审计落盘，错误流没有 DONE。"""
        started = False
        heartbeat = time.monotonic()

        def chunk(delta, finish=None):
            return dict(id="chatcmpl-" + job.id, object="chat.completion.chunk", created=job.created,
                        model=service.model_name, choices=[dict(index=0, delta=delta, finish_reason=finish)])

        def start():
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.send_header("X-Request-ID", job.id)
            self.end_headers()
            self._sse_data(chunk(dict(role="assistant", content="")))

        while True:
            if self._disconnected():
                service.cancel(job.id, 499, "client disconnected", expected_job=job)
                return
            if time.monotonic() >= job.deadline and not job.done.is_set():
                service.cancel(job.id, 504, "request deadline exceeded", expected_job=job)
            try:
                text = job.events.get(timeout=0.05)
            except queue.Empty:
                if job.done.is_set():
                    # done 可能在 get 超时与检查之间置位；必须再排空最后一个文本增量。
                    try:
                        text = job.events.get_nowait()
                    except queue.Empty:
                        break
                else:
                    if started and time.monotonic() - heartbeat >= 1:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        heartbeat = time.monotonic()
                    continue
            if not started:
                start()
                started = True
            self._sse_data(chunk(dict(content=text)))

        if job.status != 200:
            if started:
                self._sse_data(dict(error=dict(job.response["error"], code=job.status)))
            else:
                self._reply(job.status, job.response, job.id)
            return
        if not started:
            start()
        self._sse_data(chunk({}, job.response["choices"][0]["finish_reason"]))
        if job.payload.get("stream_options", {}).get("include_usage", False):
            usage = chunk({})
            usage.update(choices=[], usage=job.response["usage"])
            self._sse_data(usage)
        self._sse_data("[DONE]")

    def _dispatch(self):
        service = self.server.service
        if self.command == "GET" and self.path == "/health":
            body = service.health()
            return self._reply(200 if body["status"] == "ready" else 503, body)
        if self.command == "GET" and self.path == "/v1/models":
            return self._reply(200, dict(object="list", data=[dict(id=service.model_name, object="model", owned_by="fastllm")]))
        if self.command == "DELETE" and self.path.startswith("/v1/requests/"):
            request_id = self.path[len("/v1/requests/"):]
            if not service.cancel(request_id):
                raise HTTPError(404, "request not active or queued")
            return self._reply(202, dict(id=request_id, cancellation_requested=True))
        if self.command != "POST" or self.path != "/v1/chat/completions":
            raise HTTPError(404, "unknown endpoint")
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
            raise HTTPError(400, "one Content-Length is required; chunked requests are unsupported")
        if len(lengths[0]) > 6:
            raise HTTPError(413, "JSON body exceeds 65536 bytes")
        size = int(lengths[0])
        if not 0 < size <= 65536:
            raise HTTPError(413, "JSON body must contain 1..65536 bytes")
        if self.headers.get_content_type() != "application/json":
            raise HTTPError(415, "Content-Type must be application/json")
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise HTTPError(400, "truncated request body")
        try:
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError("duplicate JSON key")
                    result[key] = value
                return result
            payload = json.loads(raw, object_pairs_hook=pairs)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise HTTPError(400, "invalid JSON body") from error
        request_id = self.headers.get("X-Request-ID")
        job = service.submit(payload, request_id)
        try:
            if payload.get("stream"):
                return self._stream_reply(service, job)
            while not job.done.wait(0.05):
                if self._disconnected():
                    service.cancel(job.id, 499, "client disconnected", expected_job=job)
                    return
                if time.monotonic() >= job.deadline:
                    service.cancel(job.id, 504, "request deadline exceeded", expected_job=job)
            self._reply(job.status, job.response, job.id)
        finally:
            if not job.done.is_set():
                service.cancel(job.id, 499, "client disconnected", expected_job=job)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionError, TimeoutError, OSError):
            pass
        finally:
            self.close_connection = True

    def do_GET(self):
        try:
            self._dispatch()
        except HTTPError as error:
            self._reply(error.status, error_body(error))

    do_POST = do_DELETE = do_GET


class ChatHTTPServer(ThreadingHTTPServer):
    """限制连接线程数；慢速请求不能无限创建线程挤占推理进程。"""
    daemon_threads = False

    def __init__(self, address, service):
        self.service = service
        self._slots = threading.BoundedSemaphore(service.queue_size + 8)
        self._connections = set()
        self._connections_lock = threading.Lock()
        super().__init__(address, ChatHandler)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            with self._connections_lock:
                self._connections.add(request)
            super().process_request(request, client_address)
        except BaseException:
            with self._connections_lock:
                self._connections.discard(request)
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._connections_lock:
                self._connections.discard(request)
            self._slots.release()

    def server_close(self):
        # 先唤醒仍在读请求体/等待客户端读取的线程，再由标准库 join。
        # 单纯 socket timeout 会被不断到达的零碎数据刷新，不能保证停服有界。
        self.socket.close()
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        super().server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen3")
    parser.add_argument("--binary", type=Path, default=ROOT / "build-process-tp-cu126/fastllm_tp_generate")
    parser.add_argument("--worker", type=Path, default=ROOT / "build-process-tp-cu126/fastllm_tp_worker")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--session-timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "test/distributed/results/http-server")
    args = parser.parse_args()
    if (not 0 <= args.port <= 65535 or args.queue_size < 1 or not args.model_name or
            not math.isfinite(args.request_timeout) or not math.isfinite(args.session_timeout) or
            not 0 < args.request_timeout <= args.session_timeout):
        parser.error("invalid port/queue/model name/timeouts; require 0 < request-timeout <= session-timeout")
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    (output / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    service = server = None
    try:
        os.environ["USE_TORCH"] = "0"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        with Qwen3ProcessSession(args.binary, args.worker, args.model, args.devices, output / "session",
                                timeout=args.session_timeout) as session:
            service = ChatService(session, tokenizer, args.model.resolve(), args.model_name, output,
                                  queue_size=args.queue_size, request_timeout=args.request_timeout)
            try:
                server = ChatHTTPServer((args.host, args.port), service)
                server.timeout = 0.2
                ready = dict(host=args.host, port=server.server_address[1], logs=str(output), **service.health())
                (output / "ready.json").write_text(json.dumps(ready, indent=2))
                print(json.dumps(ready, ensure_ascii=False), flush=True)
                while not stop.is_set() and service.failure is None:
                    server.handle_request()
            finally:
                try:
                    service.close()
                finally:
                    if server is not None:
                        server.server_close()
        (output / "shutdown.json").write_text(json.dumps(dict(status="PASS", shutdown_complete=True)))
        return 0
    except Exception as error:
        (output / "failure.json").write_text(json.dumps(dict(status="FAIL", error=f"{type(error).__name__}: {error}")))
        print(f"FAIL: {error}; logs={output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
