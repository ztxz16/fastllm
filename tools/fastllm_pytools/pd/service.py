"""单机 P/D 控制服务；HTTP 只携带请求与 meta，CUDA IPC 搬运 KV。"""

from . import _core

from dataclasses import dataclass, field, asdict
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import queue
import sys
import threading
import time

from ..qwen3_process_chat import validate_request
from .cuda_ipc import CudaIPC
from .http import dumps, rpc
from .protocol import Identity, Peer, Layout, ProtocolError, Manifest, MAX_META_BYTES, _object, new_id, prefix_manifest, suffix_manifest
from .tasks import TransferManager, State
from .mapping import plan_transfer
from .history import History


def identity_from(value):
    return _core.invoke('service.identity_from', locals())


@dataclass
class Record:
    identity: Identity
    request: dict
    peer_url: str
    replica: object = None
    manifest: object = None
    remote: object = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    ready: threading.Event = field(default_factory=threading.Event)
    prepared: threading.Event = field(default_factory=threading.Event)
    error: str = ""
    started: bool = False
    readers: int = 0
    released: bool = False
    generating: bool = False
    complete: bool = False
    completion_state: str = ""
    preparing: bool = False
    prefill_done: bool = False
    ready_layers: set = field(default_factory=set)
    long_prompt: bool = False
    history: object = None
    history_result: object = None
    generation_done: bool = False
    expires_at: float = 0.0
    admitted: bool = False
    source_stopped: bool = False
    expected_layers: set = field(default_factory=set)
    published_layers: set = field(default_factory=set)
    sealed: bool = False
    compute_active: bool = False
    history_active: bool = False
    local_reused: int = 0
    remote_reused: int = 0
    pair: object = None
    pair_source_url: str = ""
    pair_lock: object = field(default_factory=threading.RLock)


class Service:
    """/**
     * 一个服务可包含多个独立 TP 副本；副本选择和交接任务分别管理。
     * 接收端保留权威目标 manifest；推送只接受已登记的目标，拉取只写本地目标。
     * 取消后的源 pin 等远端读取租约归还，目标等实际传输排空后才释放。
     */"""

    def __init__(self, role, peer, replicas, *, token, timeout=300, backend=None, long_threshold=512,
                 return_kv=False, retention=480, routing="load"):
        return _core.invoke('service.Service.__init__', locals())

    def _expire_history(self):
        return _core.invoke('service.Service._expire_history', locals())

    def _send_terminal(self, url, path, body):
        return _core.invoke('service.Service._send_terminal', locals())

    def _record(self, identity):
        return _core.invoke('service.Service._record', locals())

    def handshake(self, body):
        return _core.invoke('service.Service.handshake', locals())

    def _negotiate(self, record):
        return _core.invoke('service.Service._negotiate', locals())

    def _new(self, body):
        return _core.invoke('service.Service._new', locals())

    def _select_replica(self, record, candidates=None):
        return _core.invoke('service.Service._select_replica', locals())

    def topology(self, body):
        return _core.invoke('service.Service.topology', locals())

    def pair(self, body):
        return _core.invoke('service.Service.pair', locals())

    def reserve(self, body):
        return _core.invoke('service.Service.reserve', locals())

    def create(self, body):
        return _core.invoke('service.Service.create', locals())

    def _restore_history(self, record):
        return _core.invoke('service.Service._restore_history', locals())

    def _prefill_long(self, record):
        return _core.invoke('service.Service._prefill_long', locals())

    def _layer_ready(self, record, rank, layer):
        return _core.invoke('service.Service._layer_ready', locals())

    def readiness(self, identity):
        return _core.invoke('service.Service.readiness', locals())

    def _poll_source(self, record):
        return _core.invoke('service.Service._poll_source', locals())

    def register_target(self, manifest):
        return _core.invoke('service.Service.register_target', locals())

    def _start_push(self, record):
        return _core.invoke('service.Service._start_push', locals())

    def lease(self, identity, acquire):
        return _core.invoke('service.Service.lease', locals())

    def pull(self, source):
        return _core.invoke('service.Service.pull', locals())

    def _submit(self, record, source, target):
        return _core.invoke('service.Service._submit', locals())

    def _publish(self, record):
        return _core.invoke('service.Service._publish', locals())

    def _release_source(self, record):
        return _core.invoke('service.Service._release_source', locals())

    def received(self, identity, state, error=""):
        return _core.invoke('service.Service.received', locals())

    def _release_target(self, record):
        return _core.invoke('service.Service._release_target', locals())

    def _notify(self):
        return _core.invoke('service.Service._notify', locals())

    def cancel(self, identity):
        return _core.invoke('service.Service.cancel', locals())

    def generate(self, identity):
        return _core.stream('service.Service.generate', locals())

    def generate_full(self, body):
        return _core.stream('service.Service.generate_full', locals())

    def health(self):
        return _core.invoke('service.Service.health', locals())

    def evidence(self):
        return _core.invoke('service.Service.evidence', locals())

    def close(self):
        return _core.invoke('service.Service.close', locals())


def serve(service, host, port):
    """/** 运行带共享认证的内部服务；结果是持续 NDJSON，避免逐 Token 建连接。 */"""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, value, status=200):
            data = dumps(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + service.token)

        def do_GET(self):
            if not self._auth():
                return self._send(dict(error="unauthorized"), 401)
            if self.path == "/internal/evidence":
                try:
                    return self._send(service.evidence())
                except Exception as error:
                    return self._send(dict(error=str(error)), 500)
            if self.path != "/health":
                return self._send(dict(error="not found"), 404)
            value = service.health()
            self._send(value, 200 if value["healthy"] else 503)

        def do_POST(self):
            identity = None
            streamed = False
            try:
                if not self._auth():
                    return self._send(dict(error="unauthorized"), 401)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_META_BYTES:
                    return self._send(dict(error="invalid body size"), 413)
                body = json.loads(self.rfile.read(size))
                if self.path == "/internal/handshake":
                    return self._send(service.handshake(body))
                if self.path == "/internal/topology":
                    return self._send(service.topology(body))
                if self.path == "/internal/pair":
                    return self._send(service.pair(body))
                if self.path == "/internal/reserve":
                    return self._send(service.reserve(body))
                if self.path == "/internal/full":
                    identity = identity_from(body["identity"])
                    iterator = service.generate_full(body)
                else:
                    iterator = None
                if self.path == "/internal/announce":
                    service._new(body)
                    return self._send(dict(accepted=True))
                if self.path in ("/internal/prefill", "/internal/prepare"):
                    expected = "/internal/prefill" if service.role == "p" else "/internal/prepare"
                    if self.path != expected:
                        raise ProtocolError("wrong service role")
                    return self._send(service.create(body))
                if self.path in ("/internal/register-target", "/internal/pull"):
                    manifest = Manifest.from_json(dumps(body["manifest"]))
                    result = service.register_target(manifest) if self.path.endswith("register-target") else service.pull(manifest)
                    return self._send(result)
                identity = identity_from(body["identity"])
                if self.path == "/internal/history/acquire":
                    manifest = service.history.acquire(body["ticket"], identity, body["token_ids"]) if service.return_kv else None
                    return self._send(dict(manifest=json.loads(manifest.to_json()) if manifest else None))
                if self.path == "/internal/history/release":
                    service.history.release(body["ticket"], identity)
                    return self._send(dict(released=True))
                if self.path == "/internal/lease":
                    if type(body["acquire"]) is not bool:
                        raise ProtocolError("acquire must be boolean")
                    return self._send(service.lease(identity, body["acquire"]))
                if self.path == "/internal/readiness":
                    return self._send(service.readiness(identity))
                if self.path == "/internal/received":
                    return self._send(service.received(identity, body["state"], body.get("error", "")))
                if self.path == "/internal/cancel":
                    return self._send(service.cancel(identity))
                if self.path not in ("/internal/generate", "/internal/full"):
                    return self._send(dict(error="not found"), 404)
                if iterator is None:
                    iterator = service.generate(identity)
                first = next(iterator)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                streamed = True
                self.wfile.write(dumps(first) + b"\n")
                self.wfile.flush()
                for event in iterator:
                    self.wfile.write(dumps(event) + b"\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                if identity is not None:
                    service.cancel(identity)
            except Exception as error:
                if not streamed:
                    self._send(dict(error=str(error)), 400 if isinstance(error, (ValueError, KeyError)) else 503)
                else:
                    self.wfile.write(dumps(dict(event="error", error=str(error))) + b"\n")
            finally:
                if "iterator" in locals() and iterator is not None:
                    iterator.close()

    server = ThreadingHTTPServer((host, port), Handler)
    listener = threading.Thread(target=server.serve_forever, daemon=True, name="ft-service-control")
    listener.start()
    try:
        while listener.is_alive():
            listener.join(1)
    finally:
        # 排空期间保持控制入口可达，让对端能够归还读取租约。
        # 未确认 GPU 排空时不退出模型上下文，不以超时为理由释放导出显存。
        while True:
            try:
                service.close()
                break
            except TimeoutError as error:
                print(f"PD shutdown quarantined: {error}; retaining workers and control endpoint", file=sys.stderr, flush=True)
        server.shutdown()
        server.server_close()
        listener.join()
