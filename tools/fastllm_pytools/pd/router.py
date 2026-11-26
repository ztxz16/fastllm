"""独立服务级 Router：选择 P/D、转交 meta、重试并持有唯一客户端连接。"""

from . import _core

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
        return _core.invoke('router.Router.__init__', locals())

    def _probe(self, role, url):
        return _core.invoke('router.Router._probe', locals())

    def _health(self):
        return _core.invoke('router.Router._health', locals())

    def start(self):
        # 启动探测与后续周期探测共用同一判断；恢复仍要求连续成功次数。
        return _core.invoke('router.Router.start', locals())

    def register(self, role, url):
        return _core.invoke('router.Router.register', locals())

    def unregister(self, url):
        return _core.invoke('router.Router.unregister', locals())

    def close(self):
        return _core.invoke('router.Router.close', locals())

    def _remaining(self, budget):
        return _core.invoke('router.Router._remaining', locals())

    def _cancel_pair(self, identity, p, d):
        # 先通知传输发起方停止提交，接收端继续保留页面直到实际完成通知。
        return _core.invoke('router.Router._cancel_pair', locals())

    def generate(self, request, budget, *, session=None, live=True):
        return _core.stream('router.Router.generate', locals())

    def _generate_full(self, request, budget, *, session, live):
        return _core.stream('router.Router._generate_full', locals())

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
