"""正式进程执行器的 PD 适配器；KV 不拷贝到 Python 或 CPU 内存。"""

import base64
from dataclasses import replace
import math
import threading
import time
import socket
import struct
import tempfile
from pathlib import Path

from .protocol import Layout, Manifest, ProtocolError, Region, VERSION, new_id, token_digest


class NativeReplica:
    """/**
     * 一个 TP 组的本地调度所有者；原生计算串行，已导出的页可独立驻留。
     * 各请求交接 key 与服务身份分别校验，所有原生调用复用会话序号机制。
     */"""

    def __init__(self, session, peer, config, model_id, *, growth=512):
        if config.get("model_type") not in ("qwen3", "qwen2_moe") or config.get("quantization_config"):
            raise ValueError("native PD adapter requires unquantized Qwen3 or Qwen2 MoE")
        self.session, self.peer = session, peer
        self.config, self.model_id, self.growth = config, model_id, growth
        self.condition = threading.Condition(threading.RLock())
        self.active = None
        self.exports = {}
        self.waiters = []
        self.closed = False
        self.cached = {}
        self.preemptions = 0
        self.kv_usage = 0.0

    @property
    def layout(self):
        """/** 固定模型与分片布局用于服务握手；具体 KV 位置仍逐请求导出。 */"""
        cfg = self.config
        return Layout(self.model_id, cfg["num_hidden_layers"], cfg["num_key_value_heads"],
            cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]), "float16",
            len(self.session.devices), getattr(self.session, "storage", "paged"), self.session.page_size)

    def refresh_cache(self):
        """/** 读取原生前缀目录的全部 TP rank 交集，不能把历史请求当作仍存在的 KV。 */"""
        states = self._command("cache")
        common = None
        for state in states:
            entries = {tuple(s["token_ids"]) for s in state["prefix_cache"]["snapshots"]
                       if s["position_begin"] == 0 and s["token_ids"]}
            common = entries if common is None else common & entries
        with self.condition:
            self.cached = {token_digest(ids): list(ids) for ids in common or ()}

    def _command(self, action, key=None, **fields):
        if key is not None:
            fields = dict(key=key, tokens=0, capacity=0, position=0, **fields)
        result = self.session.pd_command(dict(action=action, **fields))
        if action == "usage":
            self.kv_usage = result["kv_cache_usage"]
        return result

    def score(self, waiting):
        """/** 与普通进程 DP 使用相同分数；读取最近页池压力，不阻塞当前 GPU 计算。 */"""
        return waiting + int(self.active is not None) + waiting * 6 * max(0, self.kv_usage - 0.5)

    def acquire(self, identity, deadline, cancelled):
        """/** 在副本本地排队；忙碌但健康的副本允许等待，取消不触发计算。 */"""
        with self.condition:
            ticket = object()
            self.waiters.append(ticket)
            try:
                while self.active is not None or self.waiters[0] is not ticket:
                    if cancelled.is_set() or self.closed:
                        raise InterruptedError("request cancelled while queued")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("replica queue deadline exceeded")
                    self.condition.wait(min(remaining, 0.05))
                if cancelled.is_set() or self.closed or time.monotonic() >= deadline:
                    raise InterruptedError("request no longer admissible")
                self.active = identity
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    def wait_capacity(self, prompt, decode, remaining, deadline, cancelled):
        """/** 检查真实页池空闲/可淘汰页；增长预算仅参与检查，不预分配。 */"""
        while True:
            usage = self._command("usage")
            states = usage["workers"]
            page = self.session.page_size if self.session.page_size > 0 else 128
            reserve = min(self.growth, remaining) if decode else 0
            need = math.ceil((prompt + reserve) / page)
            if states and all(s["total_pages"] == 0 or s["free_pages"] >= need for s in states):
                return
            if states and any(s["total_pages"] and need > s["total_pages"] for s in states):
                raise ValueError("request exceeds KV pool capacity including decode budget")
            if cancelled.wait(0.05):
                raise InterruptedError("cancelled while waiting for KV capacity")
            if time.monotonic() >= deadline:
                raise TimeoutError("KV admission deadline exceeded")

    def _manifest(self, identity, tokens, regions):
        if not regions:
            raise RuntimeError("native worker did not allocate receiving KV")
        layout = self.layout
        converted = []
        for native in regions:
            fields = dict(native)
            if fields.pop("head_dim") != layout.head_dim or fields.pop("page_size") != layout.page_size:
                raise ProtocolError("native model layout mismatch")
            fields["memory"] = base64.b64encode(bytes.fromhex(fields.pop("memory_hex"))).decode()
            for field in ("allocation_bytes", "offset", "token_stride", "head_stride"):
                value = fields[field]
                if int(value) != value:
                    raise ProtocolError("native byte range is not integral")
                fields[field] = int(value)
            converted.append(Region(**fields))
        return Manifest(VERSION, identity, self.peer, layout, 0, len(tokens), token_digest(tokens), tuple(converted))

    def prefill(self, identity, tokens, *, deadline=None, cancelled=None):
        with self.condition:
            if self.active != identity:
                raise ProtocolError("prefill request does not own replica")
        self._command("prefill", token_ids=tokens, capacity=len(tokens))
        self.refresh_cache()
        return self.export(identity, tokens, deadline=deadline, cancelled=cancelled)

    def prepared_prefill(self, identity, tokens, on_layer):
        """/** 独立通知线程接收 GPU 层完成消息；完整 Prefill 返回后补齐丢失的通知。 */"""
        with self.condition:
            saved, key, manifest = self.exports[identity.transfer]
            if saved != identity or self.active != identity:
                raise ProtocolError("prepared Prefill identity mismatch")
        with tempfile.TemporaryDirectory(prefix="ft-pd-") as directory:
            path = str(Path(directory) / "ready.sock")
            receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            receiver.bind(path)
            receiver.settimeout(0.05)
            done = threading.Event()
            failures = []
            def listen():
                while not done.is_set():
                    try:
                        packet = receiver.recv(25)
                        if len(packet) != 24:
                            continue
                        local_key, rank, layer = struct.unpack("=QQQ", packet)
                        if local_key == int(key, 16) and rank < manifest.layout.tp_size and layer < manifest.layout.layers:
                            on_layer(rank, layer)
                    except socket.timeout:
                        continue
                    except Exception as error:
                        failures.append(error)
                        return
            listener = threading.Thread(target=listen, daemon=True, name="ft-pd-layer-ready")
            listener.start()
            try:
                self.session.pd_command(dict(action="prepared_prefill", key=key, token_ids=tokens, notify_path=path))
            finally:
                done.set()
                listener.join()
                receiver.close()
            if failures:
                raise failures[0]
        self.refresh_cache()

    def export(self, identity, tokens, *, deadline=None, cancelled=None, optional=False):
        """/** 导出真实页并登记唯一 pin；释放计算请求后页仍受 pin 保护。 */"""
        with self.condition:
            if self.active != identity or identity.transfer in self.exports:
                raise ProtocolError("export does not own active request or was already exported")
        key = new_id()[:16]
        deadline = deadline if deadline is not None else time.monotonic() + self.session.timeout
        while True:
            if (cancelled is not None and cancelled.is_set()) or time.monotonic() >= deadline:
                raise TimeoutError("KV export allocation cancelled or expired")
            native = self.session.pd_command(dict(action="export", key=key, tokens=len(tokens), capacity=0, position=0))
            if native["regions"]:
                break
            if optional:
                return None
            if cancelled is not None:
                cancelled.wait(0.05)
            else:
                time.sleep(0.05)
        try:
            manifest = self._manifest(identity, tokens, native["regions"])
        except BaseException:
            self._command("release", key)
            raise
        with self.condition:
            self.exports[identity.transfer] = (identity, key, manifest)
        self.refresh_cache()
        return manifest

    def prepare(self, identity, tokens, output_tokens, *, deadline=None, cancelled=None):
        with self.condition:
            if self.active != identity or identity.transfer in self.exports:
                raise ProtocolError("prepare does not own active request or was already prepared")
        key = new_id()[:16]
        deadline = deadline if deadline is not None else time.monotonic() + self.session.timeout
        while True:
            if (cancelled is not None and cancelled.is_set()) or time.monotonic() >= deadline:
                raise TimeoutError("KV receiving allocation cancelled or expired")
            native = self.session.pd_command(dict(action="prepare", key=key, tokens=len(tokens),
                capacity=len(tokens) + output_tokens - 1, position=0))
            if native["regions"]:
                break
            if cancelled is not None:
                cancelled.wait(0.05)
            else:
                time.sleep(0.05)
        try:
            manifest = self._manifest(identity, tokens, native["regions"])
        except BaseException:
            self._command("release", key)
            self._command("end")
            raise
        with self.condition:
            self.exports[identity.transfer] = (identity, key, manifest)
        self.refresh_cache()
        return manifest

    def commit(self, identity, tokens):
        """/** 完成接收后重算 Prompt 最后一个 Token，首个输出仍完全由 D 采样。 */"""
        with self.condition:
            if self.active != identity:
                raise ProtocolError("commit does not own replica")
            saved, key, _ = self.exports[identity.transfer]
            if saved != identity or self.exports[identity.transfer][2].token_end != tokens:
                raise ProtocolError("stale transfer commit")
            self.session.pd_command(dict(action="commit", key=key, tokens=tokens - 1, capacity=0, position=0))
            self.release(identity)

    def restore_local(self, identity, tokens):
        """/** 优先恢复 P 本地已有前缀到固定交接位置；实际有效长度以原生回复为准。 */"""
        with self.condition:
            saved, key, _ = self.exports[identity.transfer]
            if saved != identity or self.active != identity:
                raise ProtocolError("local prefix restore identity mismatch")
        reply = self.session.pd_command(dict(action="restore_local", key=key, token_ids=tokens,
                                            tokens=0, capacity=0, position=0))
        count = reply["restored_tokens"]
        if type(count) is not int or not 0 <= count < len(tokens):
            raise ProtocolError("invalid local restored prefix length")
        return count

    def release(self, identity):
        """/** 调用者确认远端读写已经排空后解除 pin；幂等且拒绝旧启动/旧请求。 */"""
        with self.condition:
            saved = self.exports.get(identity.transfer)
            if saved is None:
                return
            if saved[0] != identity:
                raise ProtocolError("export release identity mismatch")
            self._command("release", saved[1])
            del self.exports[identity.transfer]
            self._command("usage")
            self.condition.notify_all()

    def finish(self, identity):
        with self.condition:
            if self.active != identity:
                return
            self._command("end")
            self.active = None
            self._command("usage")
            self.condition.notify_all()

    def generate(self, identity, request, cancelled, *, deadline=None):
        """/** 逐步调用原生 Decode 和正式采样；取消只在计算边界处理。 */"""
        sampling = dict(temperature=request.get("temperature", 0.0), top_k=request.get("top_k", 0),
                        top_p=request.get("top_p", 1.0), seed=request.get("seed", 0))
        token = request["token_ids"][-1]
        maximum = request.get("max_new_tokens", 32)
        eos = request.get("eos_token_ids", [])
        generated = []
        page = self.session.page_size if self.session.page_size > 0 else 128
        deadline = deadline if deadline is not None else time.monotonic() + self.session.timeout
        try:
            for index in range(maximum):
                if cancelled.is_set() or time.monotonic() >= deadline:
                    raise InterruptedError("decode cancelled or deadline exceeded")
                if self.active != identity:
                    raise ProtocolError("decode does not own replica")
                position = len(request["token_ids"]) - 1 + index
                if getattr(self.session, "storage", "paged") == "paged" and position % page == 0:
                    usage = self._command("usage")["workers"]
                    if any(state["free_pages"] == 0 for state in usage):
                        # 运行中的请求没有传输 pin；释放计算引用后回到副本队尾。
                        # 已生成 Token 和原生按交接编号保存的 RNG 均保留。
                        self.finish(identity)
                        self.preemptions += 1
                        self.acquire(identity, deadline, cancelled)
                        history = request["token_ids"] + generated
                        self.wait_capacity(len(history), True, maximum - index - 1, deadline, cancelled)
                        if position:
                            self._command("prefill", token_ids=history[:-1],
                                          capacity=len(request["token_ids"]) + maximum - 1)
                        else:
                            # 长度为一的初始 Prompt：重新建立空的已提交请求再计算首步。
                            self.prepare(identity, history, maximum)
                            self.commit(identity, 1)
                token = self._command("step", input=token, sampling=sampling,
                                      first=index == 0, context=identity.transfer)["token"]
                generated.append(token)
                finish = "eos" if token in eos else "length" if index + 1 == maximum else None
                yield dict(event="token_delta", index=index, token_ids=[token], finish_reason=finish)
                if finish:
                    break
        finally:
            if self.session.state == "ready":
                self._command("sampling_end", context=identity.transfer)
