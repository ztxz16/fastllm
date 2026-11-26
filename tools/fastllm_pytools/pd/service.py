"""单机 P/D 控制服务；HTTP 只携带请求与 meta，CUDA IPC 搬运 KV。"""

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
    """/** 严格读取控制帧身份，重试和重启不能复用已有请求记录。 */"""
    value = dict(value)
    value["source"], value["target"] = _object(Peer, value["source"]), _object(Peer, value["target"])
    return _object(Identity, value)


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


class Service:
    """/**
     * 一个服务可包含多个独立 TP 副本；副本选择和交接任务分别管理。
     * 接收端保留权威目标 manifest；推送只接受已登记的目标，拉取只写本地目标。
     * 取消后的源 pin 等远端读取租约归还，目标等实际传输排空后才释放。
     */"""

    def __init__(self, role, peer, replicas, *, token, timeout=300, backend=None, long_threshold=512,
                 return_kv=False, retention=480, routing="load"):
        if role not in ("p", "d", "full") or not replicas or not token:
            raise ValueError("P/D service requires role, replicas and internal token")
        if not math.isfinite(timeout) or timeout <= 0 or type(long_threshold) is not int or long_threshold < 1:
            raise ValueError("service timeout and long Prompt threshold must be positive")
        self.role, self.peer, self.replicas = role, peer, replicas
        self.token, self.timeout = token, timeout
        self.long_threshold = long_threshold
        if routing not in ("load", "round_robin", "prefix"):
            raise ValueError("invalid internal DP policy")
        self.routing, self.cursor = routing, 0
        self.return_kv = return_kv
        self.history = History(peer, retention)
        self.records = {}
        self.layouts = {}
        self.agreements = set()
        self.control_outbox = {}
        self.cache_sequence, self.cache_snapshot = 0, {}
        self.lock = threading.RLock()
        self.transfers = TransferManager(peer, backend or CudaIPC())
        self.closed = threading.Event()
        self.failure = ""
        self.notifier = threading.Thread(target=self._notify, daemon=True, name="ft-pd-notify")
        self.notifier.start()
        self.reaper = threading.Thread(target=self._expire_history, daemon=True, name="ft-pd-history")
        self.reaper.start()

    def _expire_history(self):
        while not self.closed.wait(1):
            try:
                self.history.expire()
                with self.lock:
                    controls = list(self.control_outbox.items())
                for key, (url, path, body) in controls:
                    try:
                        rpc(url, path, body, token=self.token, timeout=min(self.timeout, 5))
                    except Exception:
                        continue
                    with self.lock:
                        self.control_outbox.pop(key, None)
                with self.lock:
                    expired = [r.identity for r in self.records.values()
                               if not r.released and r.expires_at <= time.time()]
                for identity in expired:
                    try:
                        # Router 退出也不能留下无人管理的交接；取消仍等待实际访问排空。
                        self.cancel(identity)
                    except Exception:
                        continue
                with self.lock:
                    for key, record in list(self.records.items()):
                        if (not record.released or record.expires_at > time.time() or
                                record.started and not record.complete):
                            continue
                        if record.started:
                            self.transfers.forget(record.identity)
                        del self.records[key]
            except Exception:
                # 原生组异常时不伪造解除 pin；健康状态由原生会话失败状态决定。
                continue

    def _send_terminal(self, url, path, body):
        """/** 实际访问结束后的幂等通知可重发；网络丢包不能丢失解除占用通知。 */"""
        try:
            rpc(url, path, body, token=self.token, timeout=min(self.timeout, 5))
        except Exception:
            with self.lock:
                self.control_outbox[new_id()] = (url, path, body)

    def _record(self, identity):
        record = self.records.get(identity.transfer)
        if record is None or record.identity != identity:
            raise ProtocolError("unknown transfer, request or boot identity")
        return record

    def handshake(self, body):
        """/** 在分配或读写之前交换固定布局；不同 TP/存储布局可以协商，模型语义必须一致。 */"""
        target, remote = _object(Peer, body["target"]), _object(Peer, body["source"])
        layout = _object(Layout, body["layout"])
        if target != self.peer or remote == self.peer or remote.host != self.peer.host:
            raise ProtocolError("layout handshake belongs to another instance, boot or host")
        own = self.replicas[0].layout
        if not own.compatible(layout):
            raise ProtocolError("layout handshake model/dtype/shape mismatch")
        with self.lock:
            self.layouts = {key: value for key, value in self.layouts.items()
                            if key[0] != remote.instance or key[1] == remote.boot}
            self.layouts[(remote.instance, remote.boot, layout.layout_id)] = layout
        return dict(peer=asdict(self.peer), layout=asdict(own), layout_id=own.layout_id)

    def _negotiate(self, record):
        """/** 缓存当前启动代次的握手；请求身份、位置、有效范围仍在每次交接时独立校验。 */"""
        remote = record.identity.target if self.role == "p" else record.identity.source
        own = record.replica.layout
        key = (remote, own.layout_id)
        with self.lock:
            if key in self.agreements:
                return
        reply = rpc(record.peer_url, "/internal/handshake",
            dict(source=asdict(self.peer), target=asdict(remote), layout=asdict(own)),
            token=self.token, timeout=min(self.timeout, max(0.001, record.expires_at - time.time())))
        layout = _object(Layout, reply["layout"])
        if _object(Peer, reply["peer"]) != remote or reply["layout_id"] != layout.layout_id or not own.compatible(layout):
            raise ProtocolError("remote layout handshake changed identity or model semantics")
        with self.lock:
            self.agreements = {item for item in self.agreements if item[0].instance != remote.instance or item[0].boot == remote.boot}
            self.agreements.add(key)

    def _new(self, body):
        identity = identity_from(body["identity"])
        expected = identity.source if self.role == "p" else identity.target
        if expected != self.peer or identity.direction != "p_to_d":
            raise ProtocolError("wrong service role, instance or boot")
        request = body["request"]
        validate_request(request, self.replicas[0].config["vocab_size"])
        expires_at = body["expires_at"]
        if type(expires_at) not in (int, float) or not math.isfinite(expires_at) or expires_at <= time.time():
            raise ProtocolError("request control lease has expired")
        with self.lock:
            if self.closed.is_set() or self.failure:
                raise RuntimeError(self.failure or "service closing")
            old = self.records.get(identity.transfer)
            if old:
                if (old.identity != identity or old.request != request or old.peer_url != body["peer_url"] or
                        old.history != (body.get("history") if self.return_kv else None)):
                    raise ProtocolError("transfer identity replayed with changed data")
                return old, False
            record = Record(identity, request, body["peer_url"])
            record.expires_at = expires_at
            record.long_prompt = len(request["token_ids"]) >= self.long_threshold
            record.history = body.get("history") if self.return_kv else None
            self.records[identity.transfer] = record
            return record, True

    def _select_replica(self, record):
        """/** 调用方持有服务锁；选择与登记一起完成，保证并发请求看到已有排队承诺。 */"""
        candidates = self.replicas[self.cursor:] + self.replicas[:self.cursor]
        if self.routing == "round_robin":
            replica = candidates[0]
        elif self.routing == "prefix":
            from .routing import Registry
            replica = max(candidates, key=lambda r: Registry._prefix(record.request["token_ids"], r.cached.values()))
        else:
            def score(replica):
                waiting = sum(r.replica is replica and not r.admitted and not r.released
                              for r in self.records.values())
                return replica.score(waiting)
            replica = min(candidates, key=score)
        self.cursor = (self.replicas.index(replica) + 1) % len(self.replicas)
        record.replica = replica
        return replica

    def create(self, body):
        """/** 接受请求后在本地调度队列等待；P 与 D 的计算/分配可并行。 */"""
        record, fresh = self._new(body)
        with self.lock:
            fresh = not record.preparing
            record.preparing = True
        if not fresh:
            if not record.prepared.wait(self.timeout):
                raise TimeoutError("duplicate request still preparing")
            if record.error:
                raise RuntimeError(record.error)
            if record.released or record.cancelled.is_set():
                raise ProtocolError("transfer already released or cancelled")
            return dict(manifest=json.loads(record.manifest.to_json()))
        deadline = time.monotonic() + min(self.timeout, max(0, record.expires_at - time.time()))
        try:
            with self.lock:
                replica = self._select_replica(record)
            self._negotiate(record)
            replica.acquire(record.identity, deadline, record.cancelled)
            record.admitted = True
            tokens = record.request["token_ids"]
            replica.wait_capacity(len(tokens), self.role == "d", record.request["max_new_tokens"] - 1,
                                  deadline, record.cancelled)
            if self.role == "p":
                if record.long_prompt or record.history:
                    manifest = replica.prepare(record.identity, tokens, 1, deadline=deadline, cancelled=record.cancelled)
                    record.local_reused = replica.restore_local(record.identity, tokens)
                else:
                    manifest = replica.prefill(record.identity, tokens, deadline=deadline, cancelled=record.cancelled)
                    replica.finish(record.identity)
                    record.prefill_done = True
                    record.ready_layers = {(r.rank, r.layer) for r in manifest.regions}
            else:
                manifest = replica.prepare(record.identity, tokens, record.request["max_new_tokens"],
                                           deadline=deadline, cancelled=record.cancelled)
            with self.lock:
                record.manifest = manifest
                record.prepared.set()
            if self.role == "p" and record.history:
                self._restore_history(record)
            if self.role == "p" and record.cancelled.is_set():
                record.replica.finish(record.identity)
                record.source_stopped = True
                raise InterruptedError("cancelled before Prefill")
            if self.role == "p" and record.long_prompt:
                record.compute_active = True
                threading.Thread(target=self._prefill_long, args=(record,), daemon=True,
                                 name="ft-pd-prefill").start()
            elif self.role == "p" and record.history:
                record.compute_active = True
                self._prefill_long(record)
                if record.error:
                    raise RuntimeError(record.error)
            if record.cancelled.is_set():
                raise InterruptedError("cancelled during preparation")
            if self.role == "d" and record.identity.mode == "push":
                rpc(record.peer_url, "/internal/register-target", dict(manifest=json.loads(manifest.to_json())),
                    token=self.token, timeout=self.timeout)
            elif self.role == "p" and record.identity.mode == "push":
                self._start_push(record)
            return dict(manifest=json.loads(manifest.to_json()))
        except BaseException as error:
            with self.lock:
                record.error = str(error)
                record.cancelled.set()
                record.prepared.set()
                record.ready.set()
            if (self.role == "p" and record.replica and not record.compute_active and
                    not record.history_active and record.replica.session.state == "ready"):
                record.replica.finish(record.identity)
                record.source_stopped = True
            self.cancel(record.identity)
            if record.replica and record.manifest is None:
                record.replica.finish(record.identity)
            if record.manifest is None:
                record.released = True
            raise

    def _restore_history(self, record):
        """/** 按需从上一轮 D 拉取；来源不可用时重算，已提交传输失败必须排空后报错。 */"""
        history = record.history
        identity = Identity(record.identity.request, new_id(), Peer(**history["peer"]), self.peer, "d_to_p", "pull")
        body = dict(ticket=history["ticket"], identity=asdict(identity), token_ids=record.request["token_ids"])
        try:
            result = rpc(history["url"], "/internal/history/acquire", body, token=self.token, timeout=self.timeout)
        except Exception:
            # acquire 的回复可能丢失；当前没有提交 GPU 读取，归还可能已创建的租约。
            self._send_terminal(history["url"], "/internal/history/release",
                                dict(ticket=history["ticket"], identity=asdict(identity)))
            return
        if result.get("manifest") is None:
            return
        manager = TransferManager(self.peer, CudaIPC())
        record.history_active = True
        try:
            source = Manifest.from_json(dumps(result["manifest"]))
            if source.token_end <= record.local_reused:
                return
            target = prefix_manifest(record.manifest, identity, self.peer, record.request["token_ids"][:source.token_end])
            if record.local_reused:
                source = suffix_manifest(source, record.local_reused)
                target = suffix_manifest(target, record.local_reused)
            manager.register(source, target, lambda: None)
            for rank, layer in {(c.source.rank, c.source.layer) for c in plan_transfer(source, target)}:
                manager.publish_ready(identity, rank, layer)
            manager.seal(identity)
            deadline = time.monotonic() + min(self.timeout, max(0, record.expires_at - time.time()))
            draining_at = None
            while True:
                if record.cancelled.is_set() or time.monotonic() >= deadline:
                    manager.cancel(identity)
                    if draining_at is None:
                        draining_at = time.monotonic()
                    elif time.monotonic() - draining_at >= self.timeout:
                        self.failure = "history transfer could not drain; source and target remain quarantined"
                        raise TimeoutError(self.failure)
                try:
                    completion = manager.notifications.get(timeout=0.05)
                    break
                except queue.Empty:
                    continue
            if completion.state != State.READY:
                raise RuntimeError(completion.error or "history transfer failed")
            _, key, _ = record.replica.exports[record.identity.transfer]
            record.replica.session.pd_command(dict(action="restore_prefix", key=key,
                tokens=source.token_end, capacity=0, position=0))
            record.remote_reused = source.token_end - record.local_reused
        finally:
            manager.close(timeout=self.timeout)
            record.history_active = False
            self._send_terminal(history["url"], "/internal/history/release",
                                dict(ticket=history["ticket"], identity=asdict(identity)))

    def _prefill_long(self, record):
        try:
            record.replica.prepared_prefill(record.identity, record.request["token_ids"],
                lambda rank, layer: self._layer_ready(record, rank, layer))
            record.replica.finish(record.identity)
            with self.lock:
                record.prefill_done = True
                record.ready_layers = {(r.rank, r.layer) for r in record.manifest.regions}
                if record.started:
                    self._publish(record)
                if record.cancelled.is_set():
                    self._release_source(record)
        except Exception as error:
            with self.lock:
                record.error = str(error)
                record.cancelled.set()
        finally:
            record.compute_active = False
            if record.cancelled.is_set():
                if record.replica.session.state == "ready":
                    record.replica.finish(record.identity)
                    record.source_stopped = True
                self.cancel(record.identity)

    def _layer_ready(self, record, rank, layer):
        with self.lock:
            record.ready_layers.add((rank, layer))
            if record.started:
                self._publish(record)

    def readiness(self, identity):
        with self.lock:
            record = self._record(identity)
            return dict(ready=sorted(record.ready_layers), done=record.prefill_done,
                        error=record.error, cancelled=record.cancelled.is_set())

    def _poll_source(self, record):
        try:
            while not record.cancelled.is_set():
                state = rpc(record.peer_url, "/internal/readiness", dict(identity=asdict(record.identity)),
                            token=self.token, timeout=self.timeout)
                if state["error"] or state["cancelled"]:
                    raise RuntimeError(state["error"] or "source cancelled")
                with self.lock:
                    record.ready_layers = {tuple(item) for item in state["ready"]}
                    record.prefill_done = state["done"]
                    self._publish(record)
                if state["done"]:
                    return
                record.cancelled.wait(0.002)
        except Exception as error:
            self.transfers.cancel(record.identity, str(error))

    def register_target(self, manifest):
        """/** D 直接向 P 登记接收位置；P 完成 Prefill 且目标就绪后提交写入。 */"""
        if self.role != "p" or manifest.identity.mode != "push" or manifest.owner != manifest.identity.target:
            raise ProtocolError("invalid push target registration")
        with self.lock:
            record = self._record(manifest.identity)
            if record.cancelled.is_set():
                raise ProtocolError("request cancelled")
            if record.remote and record.remote != manifest:
                raise ProtocolError("target registration changed")
            record.remote = manifest
        self._start_push(record)
        return dict(registered=True)

    def _start_push(self, record):
        with self.lock:
            if record.started or record.manifest is None or record.remote is None or record.cancelled.is_set():
                return
            self._submit(record, record.manifest, record.remote)

    def lease(self, identity, acquire):
        """/** 拉取前保护源导出；已取消请求拒绝新读取，正在读取的租约必须显式归还。 */"""
        with self.lock:
            if self.role != "p" or identity.mode != "pull":
                raise ProtocolError("only P source exports accept pull leases")
            if not acquire and identity.source == self.peer and identity.transfer not in self.records:
                return dict(leased=False, ignored="no remaining export")
            record = self._record(identity)
            if acquire:
                if record.cancelled.is_set() or record.released or record.manifest is None:
                    raise ProtocolError("source export unavailable")
                record.readers = 1
            else:
                record.readers = 0
                self._release_source(record)
        return dict(leased=acquire)

    def pull(self, source):
        if self.role != "d" or source.identity.mode != "pull" or source.owner != source.identity.source:
            raise ProtocolError("invalid pull source")
        with self.lock:
            record = self._record(source.identity)
            if record.started:
                if record.remote != source:
                    raise ProtocolError("source changed on repeated pull")
                return dict(submitted=True)
            if record.manifest is None or record.cancelled.is_set():
                raise ProtocolError("receiver unavailable")
            record.remote = source
            # 锁内取得源租约，使取消不会夹在取得租约与登记本地任务之间。
            try:
                rpc(record.peer_url, "/internal/lease", dict(identity=asdict(record.identity), acquire=True),
                    token=self.token, timeout=self.timeout)
                self._submit(record, source, record.manifest)
                threading.Thread(target=self._poll_source, args=(record,), daemon=True,
                                 name="ft-pd-source-ready").start()
            except BaseException:
                # 即使 acquire 回复丢失，也尚未提交 GPU 访问，可幂等归还可能已建立的租约。
                # 若已成功 register，则取消排空交由完成通知归还，不能在这里提前归还。
                if record.started:
                    self.transfers.cancel(record.identity, "pull setup failed")
                    raise
                self._send_terminal(record.peer_url, "/internal/lease",
                                    dict(identity=asdict(record.identity), acquire=False))
                raise
        return dict(submitted=True)

    def _submit(self, record, source, target):
        record.expected_layers = {(c.source.rank, c.source.layer) for c in plan_transfer(source, target)}
        self.transfers.register(source, target, lambda: None)
        record.started = True
        self._publish(record)

    def _publish(self, record):
        newly_ready = (record.expected_layers & record.ready_layers) - record.published_layers
        for rank, layer in newly_ready:
            self.transfers.publish_ready(record.identity, rank, layer)
        record.published_layers.update(newly_ready)
        if record.prefill_done and not record.sealed:
            self.transfers.seal(record.identity)
            record.sealed = True

    def _release_source(self, record):
        if record.readers or record.released or record.manifest is None:
            return
        if not record.prefill_done and not record.source_stopped:
            return
        if record.started and self.transfers.status(record.identity) not in (State.READY, State.FAILED, State.CANCELLED):
            return
        record.replica.release(record.identity)
        record.released = True

    def received(self, identity, state, error=""):
        """/** 推送发起方确认真实访问已结束后通知 D；不能用 submitted 代替 READY。 */"""
        with self.lock:
            if self.role != "d" or identity.mode != "push" or state not in ("ready", "failed", "cancelled"):
                raise ProtocolError("invalid completion notification")
            if identity.target == self.peer and identity.transfer not in self.records:
                return dict(received=False, ignored="no active transfer")
            record = self._record(identity)
            if record.completion_state:
                if record.completion_state != state:
                    raise ProtocolError("completion replay changed terminal state")
                return dict(received=True)
            record.completion_state = state
            record.complete = True
            if state != "ready":
                record.error = error or state
            record.ready.set()
            if record.cancelled.is_set():
                self._release_target(record)
        return dict(received=True)

    def _release_target(self, record):
        if record.released or record.generating or record.manifest is None:
            return
        if not record.complete and (record.started or record.identity.mode == "push"):
            return
        record.replica.release(record.identity)
        record.replica.finish(record.identity)
        record.released = True

    def _notify(self):
        while not self.closed.is_set():
            try:
                event = self.transfers.notifications.get(timeout=0.1)
            except queue.Empty:
                continue
            with self.lock:
                record = self._record(event.identity)
                record.complete = True
                record.error = event.error if event.state != State.READY else ""
            try:
                if self.role == "p":
                    self._send_terminal(record.peer_url, "/internal/received", dict(identity=asdict(event.identity),
                        state=event.state.value, error=record.error))
                    with self.lock:
                        self._release_source(record)
                else:
                    self._send_terminal(record.peer_url, "/internal/lease",
                                        dict(identity=asdict(event.identity), acquire=False))
            except Exception as error:
                record.error = str(error)
            finally:
                record.ready.set()
                if self.role == "d" and record.cancelled.is_set():
                    with self.lock:
                        self._release_target(record)

    def cancel(self, identity):
        with self.lock:
            record = self._record(identity)
            record.cancelled.set()
            if record.manifest is None and not record.preparing:
                record.released = True
            if record.started and not record.complete:
                self.transfers.cancel(identity)
            elif self.role == "p":
                self._release_source(record)
            else:
                self._release_target(record)
            drained = record.complete or record.released or not record.started
        if self.role == "d" and identity.mode == "push" and not record.complete:
            result = rpc(record.peer_url, "/internal/cancel", dict(identity=asdict(identity)),
                         token=self.token, timeout=min(5, self.timeout))
            if result.get("drained") is True:
                with self.lock:
                    record.complete = True
                    self._release_target(record)
                    drained = True
        return dict(cancelled=True, drained=drained)

    def generate(self, identity):
        record = self._record(identity)
        deadline = time.monotonic() + min(self.timeout, max(0, record.expires_at - time.time()))
        while not record.ready.wait(0.05):
            if record.cancelled.is_set():
                raise InterruptedError("cancelled while receiving KV")
            if time.monotonic() >= deadline:
                self.cancel(identity)
                raise TimeoutError("KV receive timeout")
        if record.error or record.cancelled.is_set():
            raise RuntimeError(record.error or "request cancelled")
        with self.lock:
            if self.role != "d" or record.generating or record.generation_done or record.released:
                raise ProtocolError("generation already started or released")
            record.generating = True
        try:
            record.replica.commit(identity, len(record.request["token_ids"]))
            generated = []
            for event in record.replica.generate(identity, record.request, record.cancelled, deadline=deadline):
                generated.extend(event["token_ids"])
                yield dict(event, request=identity.request, transfer=identity.transfer)
            if self.return_kv and not record.cancelled.is_set():
                tokens = record.request["token_ids"] + generated[:-1]
                manifest = record.replica.export(identity, tokens, optional=True)
                if manifest is not None:
                    self.history.retain(identity.transfer, manifest, tokens, lambda: record.replica.release(identity))
                    record.history_result = dict(ticket=identity.transfer, peer=asdict(self.peer),
                                                 expires_at=time.time() + self.history.ttl)
            yield dict(event="finished", request=identity.request, transfer=identity.transfer, history=record.history_result)
        finally:
            # commit 前失败仍可能持有接收 pin；此时 ready 已确认传输排空。
            if record.history_result is None:
                record.replica.release(identity)
            record.replica.finish(identity)
            with self.lock:
                record.generating = False
                record.generation_done = True
                record.released = True

    def generate_full(self, body):
        """/** 完整推理服务复用正式 generate 入口；Router 仍持有客户端连接。 */"""
        if self.role != "full":
            raise ProtocolError("complete generation requires a full service")
        record, fresh = self._new(body)
        with self.lock:
            if not fresh or record.preparing:
                raise ProtocolError("complete request already submitted")
            record.preparing = record.generating = True
            replica = self._select_replica(record)
        events = queue.SimpleQueue()

        def produce():
            try:
                deadline = time.monotonic() + min(self.timeout, max(0, record.expires_at - time.time()))
                replica.acquire(record.identity, deadline, record.cancelled)
                record.admitted = True
                replica.wait_capacity(len(record.request["token_ids"]), True,
                    record.request["max_new_tokens"] - 1, deadline, record.cancelled)
                index = 0
                def on_token(token):
                    nonlocal index
                    if record.cancelled.is_set() or time.monotonic() >= deadline:
                        raise InterruptedError("complete request cancelled or timed out")
                    finish = ("eos" if token in record.request["eos_token_ids"] else
                              "length" if index + 1 == record.request["max_new_tokens"] else None)
                    events.put(dict(event="token_delta", index=index, token_ids=[token], finish_reason=finish))
                    index += 1
                replica.session.generate(record.request, cancel_event=record.cancelled, on_token=on_token)
                replica.refresh_cache()
                events.put(dict(event="finished"))
            except Exception as error:
                events.put(error)
            finally:
                replica.finish(record.identity)
                with self.lock:
                    record.released = record.generation_done = True
                    record.generating = False

        thread = threading.Thread(target=produce, daemon=True, name="ft-full-generation")
        thread.start()
        try:
            while True:
                event = events.get(timeout=self.timeout)
                if isinstance(event, Exception):
                    raise event
                yield dict(event, request=record.identity.request, transfer=record.identity.transfer)
                if event["event"] == "finished":
                    break
        finally:
            record.cancelled.set()
            thread.join(self.timeout)
            if thread.is_alive():
                self.failure = "complete generation did not stop after cancellation"
                raise TimeoutError(self.failure)

    def health(self):
        with self.lock:
            cached = {}
            for replica in self.replicas:
                with replica.condition:
                    cached.update(replica.cached)
            if cached != self.cache_snapshot:
                self.cache_sequence += 1
                self.cache_snapshot = cached
            return dict(peer=asdict(self.peer), role=self.role, model=self.replicas[0].model_id,
                healthy=not self.failure and not self.closed.is_set() and not self.transfers.failures() and
                        all(r.session.state == "ready" for r in self.replicas),
                load=sum(not r.released for r in self.records.values()),
                preemptions=sum(r.preemptions for r in self.replicas),
                pending_control=len(self.control_outbox),
                transfer_statistics=self.transfers.statistics(),
                cache_version=self.cache_sequence, cache_snapshot=cached)

    def evidence(self):
        """/** 读取正式执行器的计数和布局供验收；调用可能等待当前计算，不属于健康探测。 */"""
        with self.lock:
            transfers = [dict(request=r.identity.request, transfer=r.identity.transfer, local_reused=r.local_reused,
                remote_reused=r.remote_reused, prefill_done=r.prefill_done, completed=r.complete,
                released=r.released, error=r.error) for r in self.records.values()]
        return dict(health=self.health(), requests=transfers, replicas=[dict(model_type=r.config["model_type"],
            storage=getattr(r.session, "storage", "paged"), devices=r.session.devices,
            workers=r.session.ready.get("workers", []),
            states=r._command("cache"), usage=r._command("usage")) for r in self.replicas])

    def close(self):
        self.failure = "service closing"
        for record in list(self.records.values()):
            try:
                self.cancel(record.identity)
            except Exception:
                continue
        self.transfers.close(timeout=self.timeout)
        deadline = time.monotonic() + self.timeout
        while True:
            retained_done = self.history.close()
            with self.lock:
                unfinished = [r for r in self.records.values() if not r.released]
                controls_pending = bool(self.control_outbox)
            if retained_done and not unfinished and not controls_pending:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("shutdown has unconfirmed remote accesses; KV must remain protected")
            for record in unfinished:
                try:
                    self.cancel(record.identity)
                except Exception:
                    continue
            self.closed.wait(0.05)
        self.closed.set()
        self.notifier.join(self.timeout)
        self.reaper.join(self.timeout)


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
