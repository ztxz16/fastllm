"""真实 HTTP 控制连接上的 Router 重试边界；后端是故障注入桩，不代表 GPU 验收。"""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from fastllm_pytools.pd.protocol import Peer
from fastllm_pytools.pd.router import Router
from fastllm_pytools.pd.routing import Registry
from fastllm_pytools.pd.retry import RetryBudget
from report import run


@contextmanager
def worker(role, failure):
    """/** 提供真实 HTTP 故障边界；不假装执行模型或搬运 KV。 */"""
    attempts, cancels = [], []
    peer = Peer(role + "0", "boot", "host")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data, status=200):
            encoded = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.reply(dict(role=role, model="model", peer=asdict(peer), healthy=True,
                            load=0, cache_version=1, cache_snapshot={}))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/internal/cancel":
                cancels.append(body)
                return self.reply(dict(cancelled=True, drained=True))
            if self.path not in ("/internal/generate", "/internal/full"):
                return self.reply(dict(manifest={}))
            identity = body["identity"]
            attempts.append(identity)
            first = len(attempts) == 1
            if first and isinstance(failure, int):
                return self.reply(dict(error="injected"), failure)
            if first and failure == "disconnect":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.send_response(200)
            self.end_headers()
            event = dict(event="token_delta", request=identity["request"], transfer=identity["transfer"],
                         index=0, token_ids=[7], finish_reason="length")
            self.wfile.write(json.dumps(event).encode() + b"\n")
            if first and failure in ("partial", "cancel"):
                self.wfile.flush()
                return
            self.wfile.write(json.dumps(dict(event="finished", request=identity["request"],
                                            transfer=identity["transfer"])).encode() + b"\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (role, f"http://127.0.0.1:{server.server_port}"), attempts, cancels
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def case(pd, failure, responded=False):
    with ExitStack() as stack:
        target, attempts, cancels = stack.enter_context(worker("d" if pd else "full", failure))
        services = [target]
        if pd:
            source, _, _ = stack.enter_context(worker("p", None))
            services.insert(0, source)
        registry = Registry(policy="round_robin")
        router = Router(registry, services, token="test-token", model_id="model", tokenizer=None,
                        model_path=None, attempts=2, timeout=5)
        for _ in range(2):
            for role, url in services:
                router._probe(role, url)
        budget = RetryBudget(time.monotonic() + 5, attempts=2)
        payload = dict(token_ids=[1, 2], max_new_tokens=1, eos_token_ids=[])
        iterator = router.generate(payload, budget, live=responded or failure == "cancel")
        actual, error = [], None
        try:
            for event in iterator:
                actual.extend(event["token_ids"])
                if responded:
                    budget.responded = True
                if failure == "cancel":
                    budget.cancelled = True
        except Exception as caught:
            error = caught
        finally:
            iterator.close()
        retryable = failure in (429, 500, 503, "disconnect") or failure == "partial" and not responded
        expected_attempts = 2 if retryable else 1
        if len(attempts) != expected_attempts:
            raise AssertionError(f"expected {expected_attempts} attempts, got {len(attempts)}: {error!r}")
        if retryable:
            if error is not None or actual != [7]:
                raise AssertionError(f"retry did not replace buffered failed output: {error!r}, {actual}")
            if attempts[0]["request"] != attempts[1]["request"] or attempts[0]["transfer"] == attempts[1]["transfer"]:
                raise AssertionError("request ID changed or retry reused transfer ID")
        elif error is None:
            raise AssertionError("failure incorrectly reported as success")
        if any(item["inflight"] for item in registry.snapshot()):
            raise AssertionError("Router left a request lease after final completion")
        if not cancels:
            raise AssertionError("failed request did not notify cancellation")
        return dict(attempts=len(attempts), token_ids=actual, terminal_error=repr(error),
                    identities=attempts, scope="Router + HTTP injection; no KV execution")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = []
    for pd in (False, True):
        for failure in (400, 401, 404, 429, 500, 503, "disconnect", "partial", "cancel"):
            cases.append((f"{'pd' if pd else 'full'}_{failure}",
                          lambda p=pd, f=failure: case(p, f)))
        cases.append((f"{'pd' if pd else 'full'}_after_response", lambda p=pd: case(p, "partial", True)))
    return run(cases, args.output, backend="production Router + real HTTP fault stub",
               scope="20 个完整服务/PD Router 重试边界用例；不替代 CUDA 故障和模型集成验收")


if __name__ == "__main__":
    raise SystemExit(main())
