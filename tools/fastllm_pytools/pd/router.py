"""独立服务级 Router：选择 P/D、转交 meta、重试并持有唯一客户端连接。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import select
import socket
import threading
import time
import urllib.error

from .http import call, dumps, events, rpc
from .protocol import Identity, Peer, MAX_META_BYTES, new_id
from .retry import RetryBudget


class Router:
    def __init__(self, registry, services, *, token, model_id, tokenizer, model_path,
                 mode="pull", timeout=300, attempts=1, health_interval=60, health_timeout=5, long_threshold=512,
                 return_kv=False):
        if mode not in ("pull", "push") or not services or not token:
            raise ValueError("invalid router deployment")
        if any(not math.isfinite(v) or v <= 0 for v in (timeout, health_interval, health_timeout)):
            raise ValueError("router timeouts must be finite and positive")
        self.registry, self.services = registry, services
        roles = {role for role, _ in services}
        if roles not in ({"full"}, {"p", "d"}):
            raise ValueError("router requires full workers or both P and D pools")
        self.full = roles == {"full"}
        self.boot = new_id()
        self.token, self.model_id, self.tokenizer, self.model_path = token, model_id, tokenizer, model_path
        self.mode, self.timeout, self.attempts = mode, timeout, attempts
        self.health_interval, self.health_timeout = health_interval, health_timeout
        self.long_threshold = long_threshold
        self.return_kv, self.histories = return_kv, {}
        self.peers, self.lock = {}, threading.RLock()
        self.cache_versions = {}
        self.stopped = threading.Event()
        self.monitor = threading.Thread(target=self._health, name="ft-router-health", daemon=True)

    def _probe(self, role, url):
        info = rpc(url, "/health", token=self.token, timeout=self.health_timeout)
        peer = Peer(**info["peer"])
        if info["role"] != role or info["model"] != self.model_id or info.get("healthy") is not True:
            raise ValueError("service role/model/health mismatch")
        with self.lock:
            if (role, url) not in self.services:
                return
            if any(other.instance == peer.instance for address, other in self.peers.items() if address != url):
                raise ValueError("service instance ID is already registered at another URL")
            self.registry.register(peer.instance, url, role, info["model"], peer.boot)
            self.registry.health_result(peer.instance, peer.boot, True)
            self.registry.update_load(peer.instance, peer.boot, info["load"])
            version_key = (peer.instance, peer.boot)
            if self.cache_versions.get(version_key) != info.get("cache_version"):
                self.registry.cache_snapshot(peer.instance, peer.boot, info.get("cache_snapshot", {}))
                self.cache_versions[version_key] = info.get("cache_version")
            self.peers[url] = peer

    def _health(self):
        while not self.stopped.is_set():
            with self.lock:
                self.histories = {key: item for key, item in self.histories.items()
                                  if item["expires_at"] > time.time()}
            for role, url in self.services:
                try:
                    self._probe(role, url)
                except Exception:
                    with self.lock:
                        peer = self.peers.get(url)
                    if peer:
                        self.registry.health_result(peer.instance, peer.boot, False)
            self.stopped.wait(self.health_interval)

    def start(self):
        # 启动探测与后续周期探测共用同一判断；恢复仍要求连续成功次数。
        for role, url in self.services:
            try:
                self._probe(role, url)
            except Exception:
                pass
        self.monitor.start()

    def register(self, role, url):
        """/** 显式登记后主动探测；新服务仍须满足连续成功门限才加入候选池。 */"""
        allowed = {"full"} if self.full else {"p", "d"}
        if role not in allowed:
            raise ValueError("service role is incompatible with this router deployment")
        url = url.rstrip("/")
        from urllib.parse import urlsplit
        address = urlsplit(url)
        if (address.scheme not in ("http", "https") or not address.hostname or address.username or
                address.password or address.query or address.fragment or address.path):
            raise ValueError("worker URL must be an HTTP(S) origin")
        with self.lock:
            if any(existing == url and current != role for current, existing in self.services):
                raise ValueError("service URL already registered with another role")
            if (role, url) not in self.services:
                self.services = [*self.services, (role, url)]
        self._probe(role, url)

    def unregister(self, url):
        """/** 显式注销停止后续探测，并清理绑定；已发出的请求仍自行完成或取消。 */"""
        url = url.rstrip("/")
        with self.lock:
            self.services = [(role, value) for role, value in self.services if value != url]
            peer = self.peers.pop(url, None)
            if peer:
                self.registry.unregister(peer.instance)
                self.cache_versions.pop((peer.instance, peer.boot), None)

    def close(self):
        self.stopped.set()
        self.monitor.join(self.health_timeout * len(self.services) + 1)

    def _remaining(self, budget):
        if budget.cancelled or time.monotonic() >= budget.deadline:
            raise TimeoutError("request cancelled or deadline exceeded")
        return budget.deadline - time.monotonic()

    def _cancel_pair(self, identity, p, d):
        # 先通知传输发起方停止提交，接收端继续保留页面直到实际完成通知。
        ordered = (d, p) if identity.mode == "pull" else (p, d)
        for worker in ordered:
            try:
                rpc(worker.url, "/internal/cancel", dict(identity=asdict(identity)),
                    token=self.token, timeout=self.health_timeout)
            except Exception:
                pass  # 无确认时服务侧保持 pin；不会将网络失败当作 GPU 已排空。

    def generate(self, request, budget, *, session=None, live=True):
        """/** 请求级重试包围整次交接；已经向客户端响应后禁止重新执行。 */"""
        if self.full:
            yield from self._generate_full(request, budget, session=session, live=live)
            return
        request_id = new_id()
        while True:
            transfer = budget.begin()
            leases, pair, identity = [], None, None
            try:
                for role in ("p", "d"):
                    leases.append(self.registry.select(role, self.model_id, request["token_ids"], session=session))
                p, d = (self.registry.worker(lease) for lease in leases)
                pair = p, d
                with self.lock:
                    source, target = self.peers[p.url], self.peers[d.url]
                if source.boot != p.boot or target.boot != d.boot:
                    raise ConnectionError("selected worker restarted")
                identity = Identity(request_id, transfer, source, target, "p_to_d", self.mode)
                expires_at = time.time() + self._remaining(budget)
                p_body = dict(identity=asdict(identity), request=request, peer_url=d.url, expires_at=expires_at)
                if session and self.return_kv:
                    with self.lock:
                        history = self.histories.get(session)
                        if history and history["expires_at"] > time.time():
                            p_body["history"] = history
                d_body = dict(identity=asdict(identity), request=request, peer_url=p.url, expires_at=expires_at)
                # 先登记身份，避免快 D 在 P 收到请求之前登记目标。
                rpc(p.url, "/internal/announce", p_body, token=self.token, timeout=self._remaining(budget))
                if self.mode == "pull" and len(request["token_ids"]) < self.long_threshold:
                    source_meta = rpc(p.url, "/internal/prefill", p_body, token=self.token, timeout=self._remaining(budget))
                    rpc(d.url, "/internal/prepare", d_body, token=self.token, timeout=self._remaining(budget))
                    rpc(d.url, "/internal/pull", source_meta, token=self.token, timeout=self._remaining(budget))
                else:
                    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ft-pd-pair") as executor:
                        futures = [executor.submit(rpc, p.url, "/internal/prefill", p_body,
                                   token=self.token, timeout=self._remaining(budget)),
                                   executor.submit(rpc, d.url, "/internal/prepare", d_body,
                                   token=self.token, timeout=self._remaining(budget))]
                        results = [future.result(timeout=self._remaining(budget)) for future in futures]
                    if self.mode == "pull":
                        rpc(d.url, "/internal/pull", results[0], token=self.token, timeout=self._remaining(budget))
                with call(d.url, "/internal/generate", dict(identity=asdict(identity)),
                          token=self.token, timeout=self._remaining(budget)) as response:
                    index, finished, buffered = 0, False, []
                    for event in events(response):
                        self._remaining(budget)
                        if event.get("event") == "error":
                            raise ConnectionError(event.get("error", "decode failed"))
                        if event.get("request") != request_id or event.get("transfer") != transfer:
                            raise ValueError("stale or mismatched internal result")
                        if event["event"] == "token_delta":
                            tokens = event.get("token_ids")
                            if event.get("index") != index or not isinstance(tokens, list) or not tokens:
                                raise ValueError("missing or duplicate output position")
                            if any(type(t) is not int or t < 0 for t in tokens):
                                raise ValueError("invalid generated token")
                            index += len(tokens)
                            if live:
                                yield event
                            else:
                                buffered.append(event)
                        elif event["event"] == "finished":
                            if session and self.return_kv:
                                with self.lock:
                                    if event.get("history"):
                                        self.histories[session] = dict(event["history"], url=d.url)
                                    else:
                                        self.histories.pop(session, None)
                            finished = True
                            break
                        else:
                            raise ValueError("unknown internal result event")
                    if not finished or not index:
                        raise ConnectionError("decode result stream ended without completion")
                    if not live:
                        yield from buffered
                return
            except GeneratorExit:
                budget.cancelled = True
                raise
            except Exception as error:
                status = error.code if isinstance(error, urllib.error.HTTPError) else None
                delay = budget.delay(status=status, transport_error=status is None and isinstance(error, (
                    TimeoutError, ConnectionError, urllib.error.URLError)))
                if delay is None:
                    raise
            finally:
                if identity and pair:
                    self._cancel_pair(identity, *pair)
                for lease in leases:
                    self.registry.release(lease)
            if self.stopped.wait(delay):
                raise InterruptedError("router stopped")

    def _generate_full(self, request, budget, *, session, live):
        """/** 非 PD 只选择一个完整服务；策略、健康、请求重试和输出边界与 PD 共用。 */"""
        request_id = new_id()
        while True:
            transfer = budget.begin()
            lease, identity, worker = None, None, None
            try:
                lease = self.registry.select("full", self.model_id, request["token_ids"], session=session)
                worker = self.registry.worker(lease)
                with self.lock:
                    target = self.peers[worker.url]
                if target.boot != worker.boot:
                    raise ConnectionError("complete service restarted")
                identity = Identity(request_id, transfer, Peer("router", self.boot, target.host), target, "p_to_d", "pull")
                body = dict(identity=asdict(identity), request=request, peer_url="",
                            expires_at=time.time() + self._remaining(budget))
                with call(worker.url, "/internal/full", body, token=self.token,
                          timeout=self._remaining(budget)) as response:
                    index, finished, buffered = 0, False, []
                    for event in events(response):
                        self._remaining(budget)
                        if event.get("event") == "error":
                            raise ConnectionError(event.get("error", "complete generation failed"))
                        if event.get("request") != request_id or event.get("transfer") != transfer:
                            raise ValueError("stale complete result")
                        if event["event"] == "finished":
                            finished = True
                            break
                        ids = event.get("token_ids")
                        if (event["event"] != "token_delta" or event.get("index") != index or
                                not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids)):
                            raise ValueError("invalid complete output sequence")
                        index += len(ids)
                        if live:
                            yield event
                        else:
                            buffered.append(event)
                    if not finished or not index:
                        raise ConnectionError("complete stream ended without completion")
                    if not live:
                        yield from buffered
                return
            except GeneratorExit:
                budget.cancelled = True
                raise
            except Exception as error:
                delay = budget.delay(status=error.code if isinstance(error, urllib.error.HTTPError) else None,
                    transport_error=not isinstance(error, urllib.error.HTTPError) and
                    isinstance(error, (TimeoutError, ConnectionError, urllib.error.URLError)))
                if delay is None:
                    raise
            finally:
                if identity is not None:
                    try:
                        rpc(worker.url, "/internal/cancel", dict(identity=asdict(identity)),
                            token=self.token, timeout=self.health_timeout)
                    except Exception:
                        pass
                if lease is not None:
                    self.registry.release(lease)
            if self.stopped.wait(delay):
                raise InterruptedError("router stopped")

    def prepare(self, body):
        from ..qwen3_process_chat import prepare_request, validate_request
        for field in ("tools", "tool_choice", "response_format", "logit_bias", "suffix"):
            if body.get(field):
                raise ValueError(f"PD endpoint does not yet implement {field}")
        for field, default in (("repetition_penalty", 1), ("presence_penalty", 0), ("frequency_penalty", 0)):
            if body.get(field, default) != default:
                raise ValueError(f"PD endpoint does not yet implement {field}")
        if "token_ids" in body:
            request = {name: body.get(name, default) for name, default in (
                ("token_ids", []), ("max_new_tokens", 32), ("temperature", 0.0),
                ("top_k", 0), ("top_p", 1.0), ("seed", 0), ("eos_token_ids", []))}
            validate_request(request, len(self.tokenizer))
            return request
        return prepare_request(self.model_path, self.tokenizer, body.get("prompt", ""),
            raw="messages" not in body, messages=body.get("messages"),
            max_new_tokens=body.get("max_tokens", 32), temperature=body.get("temperature", 0.0),
            top_p=body.get("top_p", 1.0), top_k=body.get("top_k", 0), seed=body.get("seed", 0))


def serve(router, host, port):
    """/** 对外保持单一 OpenAI 风格结果入口；客户端断开后取消两端。 */"""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _reply(self, status, value):
            data = dumps(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self._reply(404, dict(error="not found"))
            self._reply(200, dict(workers=router.registry.snapshot()))

        def do_POST(self):
            streaming, responded = False, False
            iterator = None
            budget = RetryBudget(time.monotonic() + router.timeout, router.attempts)
            watcher_stop = threading.Event()
            try:
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + router.token):
                    return self._reply(401, dict(error="unauthorized"))
                if self.path not in ("/v1/chat/completions", "/v1/completions",
                                     "/internal/workers/register", "/internal/workers/unregister"):
                    return self._reply(404, dict(error="not found"))
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_META_BYTES:
                    return self._reply(413, dict(error="invalid body size"))
                body = json.loads(self.rfile.read(size))
                if self.path == "/internal/workers/register":
                    router.register(body["role"], body["url"])
                    return self._reply(200, dict(registered=True))
                if self.path == "/internal/workers/unregister":
                    router.unregister(body["url"])
                    return self._reply(200, dict(unregistered=True))
                if body.get("n", 1) != 1 or body.get("stop") or body.get("logprobs"):
                    raise ValueError("PD endpoint currently supports n=1 without stop strings or logprobs")
                request = router.prepare(body)
                chat = self.path == "/v1/chat/completions"
                include_ids = body.get("return_token_ids", False)
                if type(include_ids) is not bool:
                    raise ValueError("return_token_ids must be boolean")
                streaming = body.get("stream", False)
                if type(streaming) is not bool:
                    raise ValueError("stream must be boolean")

                def watch_disconnect():
                    while not watcher_stop.wait(0.1):
                        try:
                            if select.select([self.connection], [], [], 0)[0] and not self.connection.recv(1, socket.MSG_PEEK):
                                budget.cancelled = True
                                return
                        except OSError:
                            budget.cancelled = True
                            return
                watcher = threading.Thread(target=watch_disconnect, daemon=True)
                watcher.start()
                from ..qwen3_process_server import IncrementalTextDecoder
                decoder = IncrementalTextDecoder(router.tokenizer)
                result, texts, finish = [], [], None
                iterator = router.generate(request, budget, session=body.get("session_id"), live=streaming)
                for event in iterator:
                    result.extend(event["token_ids"])
                    text = "".join(decoder.push(token) for token in event["token_ids"])
                    texts.append(text)
                    finish = event.get("finish_reason")
                    if streaming:
                        if not responded:
                            budget.responded = responded = True
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream")
                            self.end_headers()
                        choice = dict(index=0, finish_reason=None)
                        choice.update(dict(delta=dict(content=text)) if chat else dict(text=text))
                        chunk = dict(id=event["request"], object="chat.completion.chunk" if chat else "text_completion",
                                     model=router.model_id, choices=[choice])
                        if include_ids:
                            chunk["token_ids"] = event["token_ids"]
                        self.wfile.write(b"data: " + dumps(chunk) + b"\n\n")
                        self.wfile.flush()
                tail = decoder.finish()
                if streaming:
                    choice = dict(index=0, finish_reason="stop" if finish == "eos" else finish)
                    choice.update(dict(delta=dict(content=tail)) if chat else dict(text=tail))
                    self.wfile.write(b"data: " + dumps(dict(choices=[choice])) + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    budget.responded = responded = True
                    choice = dict(index=0, finish_reason="stop" if finish == "eos" else finish)
                    content = "".join(texts) + tail
                    choice.update(dict(message=dict(role="assistant", content=content)) if chat else dict(text=content))
                    response = dict(object="chat.completion" if chat else "text_completion", model=router.model_id,
                        choices=[choice],
                        usage=dict(prompt_tokens=len(request["token_ids"]), completion_tokens=len(result),
                                   total_tokens=len(request["token_ids"]) + len(result)))
                    if include_ids:
                        response["token_ids"] = result
                    self._reply(200, response)
            except (BrokenPipeError, ConnectionResetError):
                budget.cancelled = True
            except Exception as error:
                if not responded:
                    self._reply(400 if isinstance(error, (ValueError, KeyError)) else 503, dict(error=str(error)))
            finally:
                watcher_stop.set()
                if iterator is not None:
                    iterator.close()

    router.start()
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        router.close()
