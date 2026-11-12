"""HTTP/队列/取消的纯 Python 回归；通过真实 socket 和正式会话客户端驱动模拟协调器。"""

from concurrent.futures import ThreadPoolExecutor
import http.client
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from unittest import mock

import test_qwen3_session as session_tests
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from qwen3_process_server import ChatHTTPServer, ChatService, HTTPError, validate_payload
import run_qwen3_http as runner


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.fixture = session_tests.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps(dict(model_type="qwen3", vocab_size=17, eos_token_id=16)))
        self.tokenizer = mock.Mock(chat_template="template", eos_token_id=16)
        self.tokenizer.apply_chat_template.return_value = [1, 2]
        self.tokenizer.decode.return_value = "回答"
        self.payload = dict(model="qwen3", messages=[dict(role="user", content="你好")], max_tokens=1)
        self.session = self.fixture.session("cancel").__enter__()
        self.addCleanup(self.session.close)
        self.service = ChatService(self.session, self.tokenizer, self.model, "qwen3", self.root, queue_size=1, request_timeout=3)
        self.expected_service_failure = False
        self.addCleanup(self.close_service)
        self.server = ChatHTTPServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs=dict(poll_interval=0.02))
        self.thread.start()
        self.addCleanup(self.stop_http)
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.addCleanup(self.pool.shutdown)

    def close_service(self):
        if self.expected_service_failure:
            with self.assertRaises(RuntimeError):
                self.service.close()
        else:
            self.service.close()

    def stop_http(self):
        self.server.shutdown()
        self.thread.join(3)
        self.server.server_close()

    def call(self, method="POST", path="/v1/chat/completions", payload=None, request_id=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=8)
        try:
            headers = {"Content-Type": "application/json"}
            if request_id:
                headers["X-Request-ID"] = request_id
            body = json.dumps(payload if payload is not None else self.payload) if method == "POST" else None
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def wait_health(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            health = self.call("GET", "/health")[1]
            if predicate(health):
                return health
            time.sleep(0.01)
        self.fail("health condition not reached")

    def test_completion_template_usage_and_reuse(self):
        for _ in range(2):
            status, body = self.call()
            self.assertEqual(status, 200)
            self.assertEqual(body["choices"][0]["message"]["content"], "回答")
            self.assertEqual(body["usage"], dict(prompt_tokens=2, completion_tokens=1, total_tokens=3))
        self.assertEqual(len(list(self.session.output.glob("request-*.response.json"))), 2)
        self.assertEqual(self.service.health()["model_load_count"], 1)
        self.assertFalse(self.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_fifo_queue_full_and_cancel_waiter(self):
        active = self.pool.submit(self.call, payload=dict(self.payload, max_tokens=16), request_id="active")
        self.wait_health(lambda h: h["active_request_id"] == "active")
        waiting = self.pool.submit(self.call, request_id="waiter")
        self.wait_health(lambda h: h["queued_request_ids"] == ["waiter"])
        self.assertEqual(self.call(request_id="overflow")[0], 429)
        self.assertEqual(self.call(request_id="active")[0], 409)
        self.assertEqual(self.call("DELETE", "/v1/requests/waiter")[0], 202)
        self.assertEqual(waiting.result()[0], 499)
        next_job = self.pool.submit(self.call, request_id="next")
        self.wait_health(lambda h: h["queued_request_ids"] == ["next"])
        self.assertEqual(self.call("DELETE", "/v1/requests/active")[0], 202)
        self.assertEqual(active.result()[0], 499)
        self.assertEqual(next_job.result()[0], 200)
        audits = {r['id']: r for r in (json.loads(p.read_text()) for p in self.root.glob('http-*.json'))}
        for rid in ('active', 'next'):
            timing = audits[rid]['timing']
            self.assertLessEqual(timing['enqueued_at'], timing['dequeued_at'])
            self.assertLessEqual(timing['dequeued_at'], timing['finished_at'])
            self.assertAlmostEqual(timing['queue_ms'], 1000*(timing['dequeued_at']-timing['enqueued_at']))
        self.assertIsNone(audits['waiter']['timing']['dequeued_at'])
        self.assertIsNone(audits['waiter']['timing']['service_ms'])
        self.assertGreaterEqual(audits['waiter']['timing']['queue_ms'], 0)
        self.assertEqual(self.session.state, "ready")
        self.assertEqual(self.call("DELETE", "/v1/requests/active")[0], 404)

    def test_disconnect_cancels_active_then_allows_next(self):
        connection = socket.create_connection(self.server.server_address)
        body = json.dumps(dict(self.payload, max_tokens=16)).encode()
        connection.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                            f"X-Request-ID: disconnect\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)
        self.wait_health(lambda h: h["active_request_id"] == "disconnect")
        connection.close()
        self.wait_health(lambda h: h["active_request_id"] is None)
        self.assertEqual(self.call()[0], 200)
        reports = [json.loads(p.read_text()) for p in self.root.glob("http-*.json")]
        self.assertTrue(any(r["id"] == "disconnect" and r["status"] == 499 for r in reports))

    def test_timeout_releases_request_and_keeps_model(self):
        self.service.request_timeout = 0.15
        status, body = self.call(payload=dict(self.payload, max_tokens=16))
        self.assertEqual(status, 504)
        self.assertIn("deadline", body["error"]["message"])
        self.assertEqual(self.call()[0], 200)

    def test_payload_rejections_do_not_reach_gpu(self):
        for extra, expected in ((dict(model="wrong"), 404), (dict(stream="true"), 400), (dict(tools=[]), 400),
                                (dict(max_tokens=0), 400), (dict(max_tokens=True), 400),
                                (dict(top_p=float("nan")), 400), (dict(temperature=-1), 400), (dict(top_p=10**400), 400),
                                (dict(messages=[]), 400), (dict(messages=[dict(role="user", content=[])]), 400)):
            with self.subTest(extra=extra):
                self.assertEqual(self.call(payload=dict(self.payload, **extra))[0], expected)
        self.assertFalse(list(self.session.output.glob("request-*.response.json")))
        self.assertEqual(self.call()[0], 200)

    def test_invalid_unicode_and_content_length_are_rejected(self):
        invalid = dict(self.payload, messages=[dict(role="user", content="\ud800")])
        self.assertEqual(self.call(payload=invalid)[0], 400)
        for value in ("²", "1"*5000):
            connection = http.client.HTTPConnection(*self.server.server_address)
            try:
                connection.request("POST", "/v1/chat/completions", b"{}",
                                   {"Content-Type": "application/json", "Content-Length": value})
                response = connection.getresponse()
                self.assertIn(response.status, (400, 413))
                response.read()
            finally:
                connection.close()
        self.assertEqual(self.call()[0], 200)

    def test_protocol_failure_is_service_error_not_bad_request(self):
        def fail(*args, **kwargs):
            self.session._dispose("failed")
            raise ValueError("malformed native JSON")
        self.expected_service_failure = True
        with mock.patch.object(self.session, "generate", side_effect=fail):
            self.assertEqual(self.call()[0], 503)
        self.assertEqual(self.call("GET", "/health")[0], 503)
        self.assertEqual(self.call()[0], 503)

    def test_deadline_is_enforced_without_http_handler_polling(self):
        self.service.request_timeout = 0.1
        job = self.service.submit(dict(self.payload, max_tokens=16))
        self.assertTrue(job.done.wait(2))
        self.assertEqual(job.status, 504)
        self.assertEqual(self.session.state, "ready")

    def test_multiple_waiters_are_fifo(self):
        self.service.queue_size = 2
        active = self.pool.submit(self.call, payload=dict(self.payload, max_tokens=16), request_id="active")
        self.wait_health(lambda h: h["active_request_id"] == "active")
        first = self.pool.submit(self.call, request_id="first")
        self.wait_health(lambda h: h["queued_request_ids"] == ["first"])
        second = self.pool.submit(self.call, request_id="second")
        self.wait_health(lambda h: h["queued_request_ids"] == ["first", "second"])
        self.call("DELETE", "/v1/requests/active")
        self.assertEqual(active.result()[0], 499)
        self.assertEqual(first.result()[0], 200)
        self.assertEqual(second.result()[0], 200)
        self.assertLess(runner.audit(self.root, "first")["native_id"], runner.audit(self.root, "second")["native_id"])

    def test_server_runner_sends_utf8_and_checks_exact_outputs(self):
        result = runner.call(self.server.server_address, "POST", "/v1/chat/completions", self.payload, "utf8")
        # 正式 runner 的路径约定在独立目录下；这里直接读取 fixture 的会话日志。
        record = runner.audit(self.root, "utf8")
        report = json.loads((self.session.output / f"request-{record['native_id']:06d}.response.json").read_text())
        with mock.patch.object(runner, "native_report", return_value=report):
            metrics = runner.verify_completion(self.root, "utf8", result, report, self.tokenizer, self.session.ready["workers"])
            self.assertEqual(metrics["failed_elements"], 0)
            with self.assertRaises(AssertionError) as caught:
                runner.verify_completion(self.root, "utf8", result, dict(report, token_ids=[3]), self.tokenizer,
                                         self.session.ready["workers"])
            self.assertEqual(caught.exception.metrics["failed_elements"], 1)

    def test_server_runner_start_failure_preserves_all_reports(self):
        cache = self.root / "cache.txt"
        cache.write_text("CUDA_ARCH:STRING=70\n")
        argv = ["runner", "--generate", "/bin/false", "--worker", "/bin/false", "--model", str(self.model),
                "--build-cache", str(cache), "--output", str(self.root / "reports")]
        transformers = mock.Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = self.tokenizer
        with mock.patch("sys.argv", argv), mock.patch.dict(sys.modules, {"transformers": transformers}), \
             mock.patch.object(runner, "capture", return_value="fixture"), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
        directory = next((self.root / "reports").iterdir())
        report = json.loads((directory / "summary.json").read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["complete"])
        for suffix in ("json", "md", "csv", "xlsx"):
            self.assertTrue((directory / f"summary.{suffix}").is_file())

    def test_long_prompt_rejected_and_recovery(self):
        self.tokenizer.apply_chat_template.side_effect = [[1]*2049, [1, 2]]
        self.assertEqual(self.call()[0], 400)
        self.assertEqual(self.call()[0], 200)

    def test_chat_history_uses_same_template(self):
        messages = [dict(role="system", content="简短回答"), dict(role="user", content="一"),
                    dict(role="assistant", content="二"), dict(role="user", content="三")]
        self.assertEqual(self.call(payload=dict(self.payload, messages=messages))[0], 200)
        self.assertEqual(self.tokenizer.apply_chat_template.call_args.args[0], messages)
        with self.assertRaises(HTTPError):
            validate_payload(dict(self.payload, messages=messages[:-1]), "qwen3")

    def test_http_body_errors_and_health(self):
        for body, expected in ((b'{"model":"qwen3","model":"qwen3"}', 400), (b'{', 400), (b'x'*65537, 413)):
            connection = http.client.HTTPConnection(*self.server.server_address)
            try:
                connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, expected)
                response.read()
            finally:
                connection.close()
        self.assertEqual(self.call("GET", "/v1/models")[1]["data"][0]["id"], "qwen3")
        self.assertEqual(self.call("GET", "/unknown")[0], 404)

    def test_shutdown_cancels_active_and_queued_and_reaps(self):
        active = self.pool.submit(self.call, payload=dict(self.payload, max_tokens=16), request_id="active")
        self.wait_health(lambda h: h["active_request_id"] == "active")
        queued = self.pool.submit(self.call, request_id="queued")
        self.wait_health(lambda h: h["queued_request_ids"] == ["queued"])
        pid = self.session._process.pid
        self.service.close()
        self.assertEqual(active.result()[0], 503)
        self.assertEqual(queued.result()[0], 503)
        self.assertEqual(self.call("GET", "/health")[0], 503)
        self.assertFalse(Path(f"/proc/{pid}").exists())
        self.assertEqual(self.session.state, "closed")

    def test_shutdown_does_not_wait_for_incomplete_request_body(self):
        connection = socket.create_connection(self.server.server_address)
        self.addCleanup(connection.close)
        connection.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                           b"Content-Type: application/json\r\nContent-Length: 1000\r\n\r\n{")
        runner.wait_for(lambda: bool(self.server._connections), timeout=2)
        self.service.close()
        closer = threading.Thread(target=self.stop_http)
        closer.start()
        closer.join(2)
        self.assertFalse(closer.is_alive(), "shutdown blocked on incomplete HTTP body")


if __name__ == "__main__":
    unittest.main()
