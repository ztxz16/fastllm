"""真实 HTTP socket + 模拟协调器的 SSE 回归；不编译、不执行 GPU。"""

import http.client
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import test_qwen3_http as http_tests
import run_qwen3_http as runner
from qwen3_process_server import ChatService, IncrementalTextDecoder


def next_event(response):
    data = []
    while True:
        line = response.readline(65536)
        if not line:
            return None
        if line == b"\n" and data:
            text = "\n".join(data)
            return text if text == "[DONE]" else json.loads(text)
        if line.startswith(b"data: "):
            data.append(line[6:].decode("utf-8").rstrip("\n"))


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.ctx = http_tests.HTTPTests()
        self.ctx.setUp()
        self.addCleanup(self.ctx.doCleanups)

    def open_stream(self, **extra):
        request_id = extra.pop("request_id", "stream-test")
        connection = http.client.HTTPConnection(*self.ctx.server.server_address, timeout=5)
        self.addCleanup(connection.close)
        payload = dict(self.ctx.payload, stream=True, **extra)
        connection.request("POST", "/v1/chat/completions", json.dumps(payload).encode(),
                           {"Content-Type": "application/json", "X-Request-ID": request_id})
        response = connection.getresponse()
        self.addCleanup(response.close)
        return response

    def replace_session(self, mode):
        self.ctx.service.close()
        self.ctx.session = self.ctx.fixture.session(mode).__enter__()
        self.ctx.addCleanup(self.ctx.session.close)
        self.ctx.service = ChatService(self.ctx.session, self.ctx.tokenizer, self.ctx.model, "qwen3", self.ctx.root,
                                       queue_size=1, request_timeout=3)
        self.ctx.server.service = self.ctx.service

    def all_events(self, response):
        result = []
        while True:
            event = next_event(response)
            if event is None:
                return result
            result.append(event)

    def test_success_matches_nonstream_and_has_usage_and_done(self):
        normal = self.ctx.call()[1]
        response = self.open_stream(stream_options=dict(include_usage=True))
        self.assertEqual(response.status, 200)
        self.assertIn("text/event-stream", response.getheader("Content-Type"))
        events = self.all_events(response)
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        chunks = [e for e in events[:-1] if e["choices"]]
        self.assertEqual("".join(e["choices"][0]["delta"].get("content", "") for e in chunks),
                         normal["choices"][0]["message"]["content"])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertEqual(events[-2]["usage"], normal["usage"])
        self.assertEqual(events[-2]["choices"], [])
        self.assertEqual(len({e["id"] for e in events[:-1]}), 1)
        self.assertEqual(len({e["created"] for e in events[:-1]}), 1)
        report = json.loads((self.ctx.session.output / "request-000002.response.json").read_text())
        self.assertEqual(report["status"], "PASS")

    def test_first_content_is_available_before_completion_and_disconnect_cancels(self):
        response = self.open_stream(max_tokens=16)
        self.assertEqual(next_event(response)["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(next_event(response)["choices"][0]["delta"]["content"], "回答")
        self.assertEqual(self.ctx.service.health()["active_request_id"], "stream-test")
        self.assertFalse((self.ctx.session.output / "request-000001.response.json").exists())
        response.close()
        self.ctx.wait_health(lambda h: h["active_request_id"] is None)
        self.assertEqual(self.ctx.call()[0], 200)

    def test_explicit_cancel_stream_has_error_not_success_terminator(self):
        response = self.open_stream(max_tokens=16)
        next_event(response)
        next_event(response)
        self.assertEqual(self.ctx.call("DELETE", "/v1/requests/stream-test")[0], 202)
        events = self.all_events(response)
        self.assertEqual(events[-1]["error"]["code"], 499)
        self.assertNotIn("[DONE]", events)
        self.assertFalse(any(e.get("choices") and e["choices"][0]["finish_reason"] for e in events))
        self.assertEqual(self.ctx.call()[0], 200)

    def test_stream_timeout_is_error_and_session_reusable(self):
        self.ctx.service.request_timeout = 0.15
        response = self.open_stream(max_tokens=16)
        events = self.all_events(response)
        self.assertEqual(events[-1]["error"]["code"], 504)
        self.assertNotIn("[DONE]", events)
        self.assertEqual(self.ctx.call()[0], 200)

    def test_bad_prompt_returns_http_error_before_sse(self):
        self.ctx.tokenizer.apply_chat_template.return_value = [1]*129
        response = self.open_stream()
        self.assertEqual(response.status, 400)
        self.assertIn("application/json", response.getheader("Content-Type"))
        self.assertIn("Prompt", json.loads(response.read())["error"]["message"])

    def test_stream_options_validation(self):
        for options in ({"stream_options": {"include_usage": True}},
                        {"stream": True, "stream_options": {"include_usage": 1}},
                        {"stream": True, "stream_options": {"extra": True}}):
            self.assertEqual(self.ctx.call(payload=dict(self.ctx.payload, **options))[0], 400)

    def test_eos_without_visible_text_still_finishes_and_counts_token(self):
        self.ctx.tokenizer.eos_token_id = 1
        self.ctx.tokenizer.decode.return_value = ""
        events = self.all_events(self.open_stream(stream_options=dict(include_usage=True)))
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(events[1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(events[-2]["usage"]["completion_tokens"], 1)

    def test_acceptance_runner_verifies_real_http_stream_not_just_status(self):
        result = runner.stream_call(self.ctx.server.server_address,
            dict(self.ctx.payload, stream=True, stream_options=dict(include_usage=True)), "runner", self.ctx.root)
        record = runner.audit(self.ctx.root, "runner")
        report = json.loads((self.ctx.session.output / f"request-{record['native_id']:06d}.response.json").read_text())
        with mock.patch.object(runner, "native_report", return_value=report):
            metrics = runner.verify_stream(self.ctx.root, "runner", result, report, self.ctx.tokenizer,
                                           self.ctx.session.ready["workers"], True)
            self.assertEqual(metrics["failed_elements"], 0)
            with self.assertRaisesRegex(AssertionError, "DONE"):
                runner.verify_stream(self.ctx.root, "runner", (200, result[1][:-1]), report, self.ctx.tokenizer,
                                     self.ctx.session.ready["workers"], True)
            modified = json.loads(json.dumps(result[1]))
            modified[1]["choices"][0]["delta"]["content"] = "wrong"
            with self.assertRaisesRegex(AssertionError, "text differs"):
                runner.verify_stream(self.ctx.root, "runner", (200, modified), report, self.ctx.tokenizer,
                                     self.ctx.session.ready["workers"], True)
        self.assertTrue((self.ctx.root / "runner.sse.json").is_file())

    def test_utf8_fragments_are_joined_without_replacement_corruption(self):
        self.replace_session("stream_burst")
        self.ctx.tokenizer.decode.side_effect = lambda ids, **kw: ["", "\ufffd", "你", "你好"][len(ids)]
        response = self.open_stream(max_tokens=3)
        events = self.all_events(response)
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1] if e["choices"])
        self.assertEqual(text, "你好")
        self.assertEqual(events[-1], "[DONE]")
        self.assertTrue(all(call.kwargs["clean_up_tokenization_spaces"] is False
                            for call in self.ctx.tokenizer.decode.call_args_list))

    def test_actual_trailing_replacement_character_is_flushed(self):
        self.ctx.tokenizer.decode.return_value = "末尾\ufffd"
        events = self.all_events(self.open_stream())
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1] if e["choices"])
        self.assertEqual(text, "末尾\ufffd")

    def test_nonmonotonic_decoder_fails_current_stream_and_recovers(self):
        self.replace_session("stream_burst")
        self.ctx.tokenizer.decode.side_effect = lambda ids, **kw: "A" if len(ids) == 1 else "B"
        response = self.open_stream(max_tokens=2)
        events = self.all_events(response)
        self.assertEqual(events[-1]["error"]["code"], 500)
        self.assertNotIn("[DONE]", events)
        self.assertEqual(self.ctx.session.state, "ready")
        self.ctx.tokenizer.decode.side_effect = None
        self.assertEqual(self.ctx.call()[0], 200)

    def test_slow_consumer_queue_is_bounded_and_cannot_emit_success(self):
        self.replace_session("stream_burst")
        self.ctx.tokenizer.decode.side_effect = lambda ids, **kw: "x"*len(ids)
        job = self.ctx.service.submit(dict(self.ctx.payload, stream=True, max_tokens=128), "slow")
        self.assertTrue(job.done.wait(3))
        self.assertEqual(job.status, 503)
        self.assertEqual(job.events.qsize(), 64)
        self.assertIn("slow", job.response["error"]["message"])
        self.assertEqual(self.ctx.session.state, "ready")
        self.assertEqual(self.ctx.call()[0], 200)

    def test_stale_disconnect_does_not_cancel_reused_external_id(self):
        old = self.ctx.service.submit(self.ctx.payload, "reused")
        self.assertTrue(old.done.wait(2))
        new = self.ctx.service.submit(dict(self.ctx.payload, max_tokens=16), "reused")
        self.assertFalse(self.ctx.service.cancel("reused", expected_job=old))
        self.assertFalse(new.cancel.is_set())
        self.ctx.service.cancel("reused", expected_job=new)
        self.assertTrue(new.done.wait(2))

    def test_generation_error_midstream_has_no_done_and_stops_bad_session(self):
        self.replace_session("stream_mismatch")
        self.ctx.expected_service_failure = True
        response = self.open_stream()
        events = self.all_events(response)
        self.assertEqual(events[-1]["error"]["code"], 503)
        self.assertNotIn("[DONE]", events)
        self.assertEqual(self.ctx.session.state, "failed")


class DecoderTests(unittest.TestCase):
    def test_empty_and_split_byte_output(self):
        tokenizer = mock.Mock()
        tokenizer.decode.side_effect = ["\ufffd", "\ufffd", "😀", "😀\n", "😀\n"]
        decoder = IncrementalTextDecoder(tokenizer)
        chunks = [decoder.push(i) for i in range(4)] + [decoder.finish()]
        self.assertEqual(chunks, ["", "", "😀", "\n", ""])

    def test_acceptance_parser_rejects_truncated_events_and_false_done(self):
        for data in (b'data: {"id":1}', b'data: {"id":1}\n'):
            with self.assertRaisesRegex(AssertionError, "truncated"):
                runner.read_event(io.BytesIO(data))
        with self.assertRaisesRegex(AssertionError, "success"):
            runner.verify_stream_error((200, ["[DONE]", dict(error=dict(code=499))]), 499)


if __name__ == "__main__":
    unittest.main()
