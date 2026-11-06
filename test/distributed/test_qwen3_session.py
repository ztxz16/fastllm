"""常驻客户端的纯 Python 协议/生命周期回归；模拟协调器，不编译、不调用 GPU。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from qwen3_process_session import BACKEND, EMPTY, GenerationRejected, Qwen3ProcessSession
import qwen3_process_chat as chat
import run_qwen3_session as runner
from run_qwen3_prefill import make_model
from qwen3_layer_reference import prefill


def fixture():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    mode = Path(args["--model"]).name
    channel = socket.socket(fileno=int(args["--session-fd"]))
    workers = [dict(rank=i, device=i, pid=os.getpid()+i) for i in range(2)]
    cache = [EMPTY.copy(), EMPTY.copy()]

    def send(value):
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
              model_load_count=1, workers=workers, cache=cache))
    while True:
        try:
            message = json.loads(read(struct.unpack("!I", read(4))[0]))
        except EOFError:
            return
        if message["op"] == "shutdown":
            send(dict(event="shutdown", id=message["id"], status="PASS", shutdown_complete=True,
                      cache_after_unload=cache))
            return
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
                      finish_reason="length", cache_after_generate=cache)
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
