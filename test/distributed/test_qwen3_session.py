"""常驻客户端的纯 Python 协议/生命周期回归；模拟协调器，不编译、不调用 GPU。"""

import atexit
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import select
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from qwen3_process_session import BACKEND, EMPTY, GenerationCancelled, GenerationObserverError, GenerationRejected, Qwen3ProcessSession
import qwen3_process_chat as chat
import run_qwen3_session as runner
from run_qwen3_prefill import make_model
from qwen3_layer_reference import prefill


def fixture():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    mode = Path(args["--model"]).name
    channel = socket.socket(fileno=int(args["--session-fd"]))
    cancellation = socket.socket(fileno=int(args["--cancel-fd"]))
    # 使用真正的第二个 PID，避免 getpid()+1 偶然撞上并行测试的活进程。
    peer = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read()'], stdin=subprocess.PIPE)
    def close_peer():
        peer.stdin.close()
        peer.wait(timeout=5)
    atexit.register(close_peer)
    workers = [dict(rank=i, device=i, pid=pid) for i, pid in enumerate((os.getpid(), peer.pid))]
    cache = [EMPTY.copy(), EMPTY.copy()]
    graph_enabled = args.get("--cuda-graph") == "1"
    graph = dict(enabled=graph_enabled, active=False, captures=0, replays=0, warmups=0, eager_decodes=0, workspace_bytes=0)
    custom_enabled = args.get('--custom-allreduce') == '1'
    ar = dict(enabled=custom_enabled, active=custom_enabled, custom_calls=0, nccl_calls=0,
              workspace_bytes=2*1024*1024+4 if custom_enabled else 388)
    prefix_enabled = args.get('--prefix-cache') == '1'
    prefix = dict(enabled=prefix_enabled, entries=0, device_bytes=0, lookups=0, hits=0,
                  reused_tokens=0, last_reused_tokens=0, last_computed_tokens=0, evictions=0, clears=0)
    saved_prompt = []

    def send(value):
        nonlocal saved_prompt
        if value['event'] != 'token' and mode != 'graph_missing':
            event = value['event']
            if event == 'result' and value['status'] == 'PASS':
                if prefix_enabled:
                    ids = value['request']['token_ids']
                    reused = 0
                    while reused < min(len(saved_prompt), len(ids)-1) and saved_prompt[reused] == ids[reused]:
                        reused += 1
                    prefix.update(entries=1, device_bytes=16*len(ids), lookups=prefix['lookups']+1,
                                  hits=prefix['hits']+int(reused > 0), reused_tokens=prefix['reused_tokens']+reused,
                                  last_reused_tokens=reused, last_computed_tokens=len(ids)-reused)
                    saved_prompt = ids[:]
                decodes = len(value['token_ids']) - 1
                ar['custom_calls' if custom_enabled else 'nccl_calls'] += 2 * (1 + decodes + int(graph_enabled and decodes > 0))
                if graph_enabled:
                    graph['replays'] += decodes
                    graph['captures'] += int(decodes > 0)
                    graph['warmups'] = graph['captures']
                else:
                    graph['eager_decodes'] += decodes
            state = dict(graph)
            if mode != 'prefix_missing':
                state['prefix_cache'] = dict(prefix)
                if event == 'shutdown' and mode != 'prefix_leak':
                    state['prefix_cache'] = {key: (prefix_enabled if key == 'enabled' else 0) for key in prefix}
                if mode == 'prefix_float_' + event:
                    state['prefix_cache']['hits'] = float(state['prefix_cache']['hits'])
                if mode == 'prefix_fallback':
                    state['prefix_cache']['enabled'] = False
                if mode == 'prefix_stale' and event == 'result':
                    state['prefix_cache'].update(lookups=0, hits=0, reused_tokens=0)
            if mode != 'ar_missing':
                state['allreduce'] = dict(ar)
                if mode == 'ar_float_' + event:
                    state['allreduce']['custom_calls'] = float(ar['custom_calls'])
                if mode == 'ar_fallback':
                    state['allreduce'].update(enabled=False, active=False)
                if mode == 'ar_stale' and event == 'result':
                    state['allreduce']['custom_calls'] = 0
                if mode == 'ar_reset' and event == 'shutdown':
                    state['allreduce']['custom_calls'] = 0
            if event == 'shutdown':
                state.update(captures=0, replays=0, warmups=0, eager_decodes=0)
            if event == 'result' and mode == 'graph_fallback':
                state['eager_decodes'] = 1
            if event == 'result' and mode == 'graph_leak':
                state.update(active=True, workspace_bytes=4096, captures=1, warmups=1, replays=1)
            if event == 'result' and mode == 'graph_stale':
                state.update(captures=0, replays=0, warmups=0)
            if mode == 'graph_float_' + event:
                for key in ('captures', 'replays', 'warmups', 'eager_decodes', 'workspace_bytes'):
                    state[key] = float(state[key])
            value['execution'] = 'decode_cuda_graph' if graph_enabled else 'eager'
            value['graph_after_unload' if event == 'shutdown' else 'graph'] = [dict(state), dict(state)]
        data = json.dumps(value, indent=2).encode()
        framed = struct.pack("!I", len(data)) + data
        # 验证读取可以正确组装分段 header/body，多行 JSON 不依赖 stdout。
        for offset in range(0, len(framed), 7):
            channel.sendall(framed[offset:offset+7])

    def read(size):
        result = b""
        while len(result) < size:
            part = channel.recv(size-len(result))
            if not part:
                raise EOFError()
            result += part
        return result

    print("NCCL INFO simulated noisy stdout", flush=True)
    send(dict(event="ready", status="PASS", protocol=1, backend=BACKEND,
              model_load_count=1, workers=workers, cache=cache, cancellation="request_id_datagram", token_events=True))
    while True:
        try:
            message = json.loads(read(struct.unpack("!I", read(4))[0]))
        except EOFError:
            return
        if message["op"] == "shutdown":
            send(dict(event="shutdown", id=message["id"], status="PASS", shutdown_complete=True,
                      cache_after_unload=cache))
            return
        if message['op'] == 'clear_prefix_cache':
            saved_prompt = []
            prefix.update(entries=0, device_bytes=0, clears=prefix['clears'] + int(prefix_enabled))
            send(dict(event='prefix_cache_cleared', id=message['id'], status='PASS', cache=cache))
            continue
        request = message["request"]
        if mode == "timeout":
            child = subprocess.Popen([sys.executable, "-c", "import signal; signal.raise_signal(signal.SIGSTOP)"])
            Path(args["--model"]).with_suffix(".pids.json").write_text(json.dumps([os.getpid(), child.pid]))
            time.sleep(60)
        if mode == "crash":
            return
        if mode == "oversize":
            channel.sendall(struct.pack("!I", 512*1024*1024+1))
            return
        if mode == "bad_json":
            channel.sendall(struct.pack("!I", 1) + b"{")
            return
        result = dict(event="result", id=message["id"], status="PASS", backend=BACKEND,
                      workers=workers, reusable=True, request=request, token_ids=[request["token_ids"][0]],
                      finish_reason="eos" if request["token_ids"][0] in request.get("eos_token_ids", []) else "length",
                      cache_after_generate=cache)
        if (mode in ('graph_normal', 'graph_stale') or mode.startswith(('ar_', 'prefix_'))) and request.get('max_new_tokens', 0) > 0:
            result['token_ids'] *= request['max_new_tokens']
            result['steps'] = [dict(token_id=token) for token in result['token_ids']]
        if message.get("stream_tokens"):
            token = request["token_ids"][0]
            token_event = dict(event="token", id=message["id"], index=0, token_id=token)
            if mode == "stream_bad_index":
                token_event["index"] = 1
            if mode == "stream_bad_id":
                token_event["id"] += 1
            if mode == "stream_bad_token":
                token_event["token_id"] = 1.0
            if mode != "stream_missing":
                send(token_event)
            result["steps"] = [dict(token_id=token)]
            if mode == "stream_mismatch":
                result["token_ids"] = [token+1]
            if mode == "stream_error_float":
                result.update(status="FAIL", steps=[dict(token_id=float(token))], cache_after_error=cache)
            if mode in ("stream_burst", "stream_eos"):
                count = request.get("max_new_tokens", 1)
                if mode == "stream_eos":
                    count = 1
                    result["finish_reason"] = "eos"
                for index in range(1, count):
                    send(dict(event="token", id=message["id"], index=index, token_id=token))
                result["token_ids"] = [token]*count
                result["steps"] = [dict(token_id=token)]*count
        if mode == "cancel" and request.get("max_new_tokens", 1) > 1:
            while True:
                if select.select([cancellation], [], [], 5)[0]:
                    cancelled_id = struct.unpack("!I", cancellation.recv(4))[0]
                    if cancelled_id == message["id"]:
                        break
                else:
                    raise RuntimeError("fixture cancellation timed out")
            result.update(status="FAIL", cancelled=True, error="generation cancelled", cache_after_error=cache)
        if request.get("max_new_tokens") == 0:
            result.update(status="FAIL", error="invalid generation length", cache_after_error=cache)
        if mode == "bad_id":
            result["id"] += 1
        if mode == "dirty_kv":
            result["cache_after_generate"] = [dict(EMPTY, active=True), EMPTY]
        if mode == "float_id":
            result["token_ids"] = [1.0]
        if mode == "fatal":
            result.update(status="FAIL", error="worker failed", reusable=False)
        send(result)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "fake-native"
        self.binary.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
                               "from test_qwen3_session import fixture\nfixture()\n")
        self.binary.chmod(0o700)
        self.request = dict(token_ids=[1, 2], max_new_tokens=1, eos_token_ids=[])

    def session(self, mode="normal", timeout=5):
        return Qwen3ProcessSession(self.binary, self.binary, self.root / mode, "0,1",
                                   self.root / (mode + "-logs"), timeout=timeout)

    def test_reuse_and_rejection_preserve_same_workers(self):
        with self.session() as session:
            workers = session.ready["workers"]
            for token in (1, 3, 1):
                report = session.generate(dict(self.request, token_ids=[token]))
                self.assertEqual(report["token_ids"], [token])
                self.assertEqual(report["workers"], workers)
            with self.assertRaisesRegex(GenerationRejected, "generation length"):
                session.generate(dict(self.request, max_new_tokens=0))
            self.assertEqual(session.state, "ready")
            self.assertEqual(session.generate(self.request)["token_ids"], [1])
        self.assertEqual(session.state, "closed")
        self.assertTrue(session.shutdown["shutdown_complete"])
        self.assertEqual(len(list(session.output.glob("request-*.response.json"))), 5)
        self.assertIn("NCCL", (session.output / "native.stdout.log").read_text())
        session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.generate(self.request)

    def test_concurrent_callers_are_serialized(self):
        with self.session() as session, ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(session.generate, dict(self.request, token_ids=[i])) for i in range(8)]
            results = [future.result() for future in futures]
            self.assertEqual([r["token_ids"] for r in results], [[i] for i in range(8)])
            self.assertEqual(sorted(r["id"] for r in results), list(range(1, 9)))

    def test_cancellation_releases_kv_and_preserves_workers(self):
        import threading
        with self.session("cancel") as session:
            workers = session.ready["workers"]
            event = threading.Event()
            timer = threading.Timer(0.1, event.set)
            timer.start()
            try:
                with self.assertRaises(GenerationCancelled) as caught:
                    session.generate(dict(self.request, max_new_tokens=16), cancel_event=event)
                self.assertEqual(caught.exception.report["cache_after_error"], [EMPTY, EMPTY])
            finally:
                timer.cancel()
                timer.join()
            self.assertEqual(session.state, "ready")
            self.assertEqual(session.generate(self.request)["workers"], workers)

    def test_late_cancel_id_does_not_affect_following_request(self):
        import threading
        with self.session("cancel") as session:
            session.generate(self.request)
            session._cancel_socket.send(struct.pack("!I", 1))
            event = threading.Event()
            timer = threading.Timer(0.2, event.set)
            timer.start()
            start = time.monotonic()
            try:
                with self.assertRaises(GenerationCancelled):
                    session.generate(dict(self.request, max_new_tokens=16), cancel_event=event)
                self.assertGreaterEqual(time.monotonic() - start, 0.15)
            finally:
                timer.cancel()
                timer.join()

    def test_stream_callback_arrives_before_result_and_can_cancel(self):
        import threading
        event, tokens = threading.Event(), []
        with self.session("cancel") as session:
            def callback(token):
                self.assertFalse((session.output / "request-000001.response.json").exists())
                tokens.append(token)
                event.set()
            with self.assertRaises(GenerationCancelled):
                session.generate(dict(self.request, max_new_tokens=16), cancel_event=event, on_token=callback)
            self.assertEqual(tokens, [1])
            self.assertEqual(session.generate(self.request)["token_ids"], [1])

    def test_callback_failure_cancels_and_drains_without_losing_session(self):
        with self.session("cancel") as session:
            def fail(token):
                raise ValueError("decoder failed")
            with self.assertRaisesRegex(GenerationObserverError, "decoder failed"):
                session.generate(dict(self.request, max_new_tokens=16), on_token=fail)
            self.assertEqual(session.state, "ready")
            self.assertEqual(session.generate(self.request)["id"], 2)

    def test_stream_protocol_corruption_fails_closed(self):
        for mode in ("stream_bad_index", "stream_bad_id", "stream_bad_token", "stream_missing", "stream_mismatch", "stream_error_float"):
            with self.subTest(mode=mode), self.session(mode) as session:
                with self.assertRaises(RuntimeError):
                    session.generate(self.request, on_token=lambda token: None)
                self.assertEqual(session.state, "failed")

    def test_invalid_local_request_does_not_poison_session(self):
        with self.session() as session:
            for request in (dict(self.request, temperature=float("nan")), dict(self.request, extra="a"*(1024*1024))):
                with self.assertRaises(ValueError):
                    session.generate(request)
            self.assertEqual(session.generate(self.request)["id"], 1)

    def test_protocol_failure_closes_session(self):
        for mode in ("bad_id", "dirty_kv", "float_id", "fatal", "crash", "oversize", "bad_json"):
            with self.subTest(mode=mode):
                with self.session(mode) as session:
                    pid = session._process.pid
                    with self.assertRaises((RuntimeError, ValueError)):
                        session.generate(self.request)
                    self.assertEqual(session.state, "failed")
                    self.assertFalse(Path(f"/proc/{pid}").exists())
                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        session.generate(self.request)

    def test_timeout_reaps_coordinator_and_stopped_child(self):
        with self.session("timeout", timeout=0.5) as session:
            with self.assertRaises(TimeoutError):
                session.generate(self.request)
            pids = json.loads((self.root / "timeout.pids.json").read_text())
            self.assertEqual(session.state, "failed")
            for pid in pids:
                self.assertFalse(Path(f"/proc/{pid}").exists())

    def test_caller_error_aborts_session(self):
        with self.assertRaisesRegex(ValueError, "caller failed"):
            with self.session() as session:
                pid = session._process.pid
                raise ValueError("caller failed")
        self.assertEqual(session.state, "failed")
        self.assertFalse(Path(f"/proc/{pid}").exists())

    def test_start_failure_releases_resources(self):
        session = Qwen3ProcessSession("/bin/false", "/bin/false", self.root, "0,1", self.root / "failed")
        with self.assertRaisesRegex(RuntimeError, "disconnected"):
            session.__enter__()
        self.assertEqual(session.state, "failed")
        self.assertFalse(Path(f"/proc/{session._process.pid}").exists())
        session.close()

    def test_cleanup_error_cannot_report_closed(self):
        session = self.session().__enter__()
        self.addCleanup(session.close)
        pid = session._process.pid
        session._stack.callback(mock.Mock(side_effect=RuntimeError("cleanup failed")))
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            session.close()
        self.assertEqual(session.state, "failed")
        self.assertFalse(Path(f"/proc/{pid}").exists())
        self.assertFalse((session.output / "shutdown.json").exists())

    def test_runner_does_not_mark_failed_session_shutdown_as_pass(self):
        model = self.root / "fatal"
        make_model(model, dict(layers=1))
        cache = self.root / "cache.txt"
        cache.write_text("CUDA_ARCH:STRING=70\n")
        argv = ["runner", "--generate", str(self.binary), "--worker", str(self.binary), "--decode-check", "/bin/false",
                "--build-cache", str(cache), "--model", str(model), "--output", str(self.root / "reports")]
        with mock.patch("sys.argv", argv), mock.patch.object(runner, "capture", return_value="fixture"), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
        directory = next((self.root / "reports").iterdir())
        report = json.loads((directory / "summary.json").read_text())
        shutdown = next(row for row in report["cases"] if row["case"] == "real/shutdown")
        self.assertEqual(shutdown["status"], "FAIL")
        self.assertIn("failed", shutdown["reason"])

    def test_runner_failure_saves_partial_reports(self):
        model = self.root / "model"
        make_model(model, dict(layers=1))
        cache = self.root / "cache.txt"
        cache.write_text("CUDA_ARCH:STRING=70\n")
        argv = ["runner", "--generate", "/bin/false", "--worker", "/bin/false", "--decode-check", "/bin/false",
                "--build-cache", str(cache), "--model", str(model), "--output", str(self.root / "reports")]
        with mock.patch("sys.argv", argv), mock.patch.object(runner, "capture", return_value="fixture"), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
        directory = next((self.root / "reports").iterdir())
        report = json.loads((directory / "summary.json").read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["complete"])
        self.assertIn("disconnected", report["cases"][0]["reason"])
        for suffix in ("json", "md", "csv", "xlsx"):
            self.assertTrue((directory / f"summary.{suffix}").is_file())

    def test_session_output_checks_actual_sampling_and_eos(self):
        config, tensors = make_model(self.root / "numeric-model", dict(layers=1))
        request = dict(self.request, temperature=0, top_k=0, top_p=1, seed=123)
        logits = prefill(config, tensors, request["token_ids"])
        token = max(range(len(logits)), key=logits.__getitem__)
        report = dict(request=request, token_ids=[token], steps=[dict(token_id=token, logits=logits)], finish_reason="length")
        runner.verify_output(report, request, config["vocab_size"])
        wrong = (token+1) % config["vocab_size"]
        with self.assertRaisesRegex(AssertionError, "incorrect sampled token"):
            runner.verify_output(dict(report, token_ids=[wrong], steps=[dict(token_id=wrong, logits=logits)]),
                                 request, config["vocab_size"])
        with self.assertRaisesRegex(AssertionError, "finish reason"):
            runner.verify_output(dict(report, finish_reason="eos"), request, config["vocab_size"])

    def test_interactive_frontend_loads_once_and_records_each_prompt(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps(dict(model_type="qwen3", vocab_size=17, eos_token_id=16)))
        tokenizer = mock.Mock(chat_template="template", eos_token_id=16)
        tokenizer.apply_chat_template.side_effect = [[1, 2], [1]*129, [3, 4]]
        tokenizer.decode.side_effect = ["回答一", "回答二"]
        transformers = mock.Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        argv = ["chat", "--model", str(model), "--interactive", "--max-new-tokens", "1",
                "--binary", str(self.binary), "--worker", str(self.binary), "--output", str(self.root / "text")]
        with mock.patch("sys.argv", argv), mock.patch("sys.stdin", io.StringIO("你好\n太长\n再见\n")), \
             mock.patch.dict(sys.modules, {"transformers": transformers}), \
             redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            self.assertEqual(chat.main(), 1)
        self.assertEqual(stdout.getvalue(), "回答一\n回答二\n")
        transformers.AutoTokenizer.from_pretrained.assert_called_once()
        directory = next((self.root / "text").iterdir())
        summary = json.loads((directory / "response.json").read_text())
        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertTrue(summary["shutdown_complete"])
        self.assertEqual(json.loads((directory / "text-000002.json").read_text())["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
