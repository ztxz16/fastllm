"""生成前端、采样判定、数值验收的纯 Python 回归，不编译或启动 GPU。"""

import copy
from collections import UserDict
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
import signal
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from qwen3_process_chat import prepare_request, parse_report, validate_request
import qwen3_process_chat as chat
import run_process_tp as process_runner
from run_qwen3_generation import BACKEND, candidates, main, replay_tasks, verify_generation
from run_qwen3_prefill import make_model
from qwen3_layer_reference import prefill
from run_qwen3_decode import BACKEND as DECODE_BACKEND, EMPTY, expectations, expected_cache


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.config, self.tensors = make_model(self.model, dict(layers=1))
        self.config["model_type"] = "qwen3"
        self.config["eos_token_id"] = 15
        (self.model / "config.json").write_text(json.dumps(self.config))
        self.tokenizer = mock.Mock(chat_template="template", eos_token_id=16)
        self.tokenizer.apply_chat_template.return_value = [1, 2]
        self.tokenizer.encode.return_value = [3, 4]
        self.request = dict(token_ids=[1, 2], max_new_tokens=3, temperature=0, top_k=0, top_p=1, seed=123, eos_token_ids=[])

    def test_chat_template_and_eos(self):
        (self.model / "generation_config.json").write_text(json.dumps(dict(eos_token_id=[14, 16])))
        request = prepare_request(self.model, self.tokenizer, "你好")
        self.assertEqual(request["token_ids"], [1, 2])
        self.assertEqual(request["eos_token_ids"], [14, 16])
        self.tokenizer.apply_chat_template.assert_called_once_with([dict(role="user", content="你好")],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)

    def test_raw_eos_override_and_ignore(self):
        request = prepare_request(self.model, self.tokenizer, "hello", raw=True, eos_ids=[5])
        self.assertEqual(request["token_ids"], [3, 4])
        self.assertEqual(request["eos_token_ids"], [5])
        self.tokenizer.encode.assert_called_once_with("hello", add_special_tokens=False)
        self.assertEqual(prepare_request(self.model, self.tokenizer, "hello", ignore_eos=True)["eos_token_ids"], [])

    def test_chat_template_return_type_does_not_depend_on_library_default(self):
        for default_return_dict in (False, True):
            with self.subTest(default_return_dict=default_return_dict):
                # 模拟旧版列表与新版 BatchEncoding（UserDict）默认返回值。
                self.tokenizer.apply_chat_template.side_effect = lambda *args, **kwargs: (
                    UserDict(input_ids=[1, 2], attention_mask=[1, 1])
                    if kwargs.get("return_dict", default_return_dict) else [1, 2])
                request = prepare_request(self.model, self.tokenizer, "你好")
                self.assertEqual(request["token_ids"], [1, 2])
                self.assertTrue(all(type(token) is int for token in request["token_ids"]))
                self.assertEqual(request["eos_token_ids"], [15, 16])

    def test_local_template_fallback_and_missing_template(self):
        self.tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, "no chat template"):
            prepare_request(self.model, self.tokenizer, "hello")
        (self.model / "chat_template.jinja").write_text("local template")
        prepare_request(self.model, self.tokenizer, "hello", thinking=True)
        self.assertEqual(self.tokenizer.chat_template, "local template")
        self.assertTrue(self.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_limits_not_silently_truncated(self):
        for change in (dict(token_ids=[]), dict(token_ids=[1]*129), dict(token_ids=[17]),
                       dict(max_new_tokens=0), dict(max_new_tokens=2048), dict(seed=-1),
                       dict(top_k=True), dict(top_p=0), dict(top_p=float("nan")),
                       dict(temperature=float("inf")), dict(temperature=-1), dict(eos_token_ids=[-1])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request(dict(self.request, **change), 17)
        validate_request(dict(self.request, token_ids=[1], max_new_tokens=2048), 17)
        validate_request(dict(self.request, token_ids=[1]*128, max_new_tokens=1), 17)

    def test_non_qwen3_rejected(self):
        (self.model / "config.json").write_text(json.dumps(dict(model_type="qwen3_moe")))
        with self.assertRaisesRegex(ValueError, "dense Qwen3"):
            prepare_request(self.model, self.tokenizer, "hello")

    def test_parse_report_rejects_broken_outer_json(self):
        self.assertEqual(parse_report('NCCL INFO\n{"status":"PASS"}\n')["status"], "PASS")
        with self.assertRaises(ValueError):
            parse_report('{broken\n{"status":"PASS"}')

    def test_sampling_allowed_sets(self):
        self.assertEqual(candidates([1, 3, 3], self.request), [1])
        self.assertEqual(candidates([0]*4, dict(self.request, temperature=1, top_p=0.5)), [0, 1])
        self.assertEqual(candidates([0, 3, 2, 1], dict(self.request, temperature=1, top_k=2, top_p=0.5)), [1])
        with self.assertRaises(AssertionError):
            candidates([float("nan")], self.request)

    def make_report(self, request=None, config=None, tensors=None):
        request = self.request if request is None else request
        config = self.config if config is None else config
        tensors = self.tensors if tensors is None else tensors
        prefix, ids, steps = request["token_ids"][:], [], []
        for _ in range(request["max_new_tokens"]):
            logits = prefill(config, tensors, prefix)
            token = candidates(logits, request)[0]
            ids.append(token)
            steps.append(dict(token_id=token, logits=logits))
            if token in request["eos_token_ids"]:
                break
            prefix.append(token)
        empty = [dict(active=False, tokens=0, capacity=0, device_bytes=0) for _ in range(2)]
        return dict(status="PASS", backend=BACKEND, request=request, shutdown_complete=True,
            workers=[dict(rank=i, device=i, pid=100+i, cuda_runtime_version=12060, nccl_version=22203) for i in range(2)],
            cache_after_generate=copy.deepcopy(empty), cache_after_unload=copy.deepcopy(empty),
            token_ids=ids, steps=steps, finish_reason="eos" if ids[-1] in request["eos_token_ids"] else "length")

    def test_generation_output_and_last_token_not_consumed(self):
        report = self.make_report()
        verify_generation(report, self.request, [0, 1], 17)
        tasks = replay_tasks(self.request["token_ids"], report["token_ids"])
        self.assertEqual(tasks[0]["capacity"], 4)
        self.assertEqual([t["token_id"] for t in tasks if t["op"] == "decode"], report["token_ids"][:-1])

    def test_eos_first_token_and_no_extra_decode(self):
        first = self.make_report()["token_ids"][0]
        request = dict(self.request, eos_token_ids=[first])
        report = self.make_report(request)
        verify_generation(report, request, [0, 1], 17)
        self.assertEqual(report["token_ids"], [first])
        self.assertEqual([t["op"] for t in replay_tasks(request["token_ids"], [first])], ["begin", "end"])

    def test_wrong_finish_token_or_leaked_cache_rejected(self):
        for mutation in (lambda r: r.update(finish_reason="eos"),
                         lambda r: r["steps"][0].update(token_id=99),
                         lambda r: r["cache_after_generate"][0].update(active=True),
                         lambda r: r["steps"][0].update(logits=[float("nan")]*17)):
            report = self.make_report()
            mutation(report)
            with self.assertRaises(AssertionError):
                verify_generation(report, self.request, [0, 1], 17)

    def test_json_token_ids_require_integers_with_diagnostic(self):
        # 复现 C++ double 序列化的 JSON，而非在判定里将小数强转 int 掩盖错误。
        report = json.loads(json.dumps(self.make_report()))
        verify_generation(report, self.request, [0, 1], 17)
        for field in ("token_ids", "steps"):
            for invalid in (1.0, 1.5, True, "1", -1, 17):
                broken = copy.deepcopy(report)
                if field == "token_ids":
                    broken["token_ids"][0] = invalid
                else:
                    broken["steps"][0]["token_id"] = invalid
                broken = json.loads(json.dumps(broken))
                with self.subTest(field=field, invalid=invalid), self.assertRaisesRegex(
                        AssertionError, r"invalid generated ID at step=0: .*vocab=17"):
                    verify_generation(broken, self.request, [0, 1], 17)

        # Python 中 True == 1、1.0 == 1；逐步 trace 也必须拒绝这些类型。
        broken = copy.deepcopy(report)
        broken["token_ids"][0] = 1
        for invalid in (True, 1.0):
            broken["steps"][0]["token_id"] = invalid
            with self.assertRaisesRegex(AssertionError, "invalid generated ID"):
                verify_generation(broken, self.request, [0, 1], 17)

    def test_generation_cannot_continue_after_eos(self):
        report = self.make_report()
        request = dict(self.request, eos_token_ids=[report["token_ids"][0]])
        report["request"] = request
        with self.assertRaisesRegex(AssertionError, "continued after EOS"):
            verify_generation(report, request, [0, 1], 17)

    def test_chat_output_error_cannot_report_success(self):
        request = self.request
        report = self.make_report(request)
        self.tokenizer.decode.return_value = "你好"
        argv = ["chat", "--model", str(self.model), "--prompt", "你好", "--ignore-eos",
                "--max-new-tokens", "3", "--seed", "123", "--output", str(self.root / "chat")]
        class BrokenOutput(io.StringIO):
            def write(self, text):
                raise BrokenPipeError("output pipe closed")
        transformers = mock.Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = self.tokenizer
        with mock.patch("sys.argv", argv), mock.patch.dict(sys.modules, {"transformers": transformers}), \
             mock.patch.object(chat, "run_native", return_value=report), \
             redirect_stdout(BrokenOutput()), redirect_stderr(io.StringIO()):
            self.assertEqual(chat.main(), 1)
        response = json.loads(next((self.root / "chat").glob("*/response.json")).read_text())
        self.assertEqual(response["status"], "FAIL")
        self.assertIn("BrokenPipeError", response["error"])

    def test_chat_timeout_reaps_stopped_descendant(self):
        # 测试进程也作为 subreaper 兜底；旧实现只 wait 协调器，会遗留已停止的孙进程。
        process_runner.enable_child_subreaper()
        binary = self.root / "native-fixture"
        identities = self.root / "timeout.pids.json"
        source = Path(__file__).resolve().parent
        binary.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(source)!r})\nfrom test_process_tp_runner import fixture_parent\n"
            f"raise SystemExit(fixture_parent('timeout', Path({str(identities)!r})))\n")
        binary.chmod(0o700)
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                chat.run_native(binary, Path('/unused-worker'), self.model, self.request, "0,1", self.root, timeout=0.5)
            pids = json.loads(identities.read_text())
            self.assertFalse(Path(f"/proc/{pids['worker']}").exists(), "chat frontend left worker alive or zombie")
            with self.assertRaises(ChildProcessError):
                os.waitpid(pids["worker"], os.WNOHANG)
        finally:
            if identities.exists():
                # 即使回归失败也由外层测试回收遗留子进程，不能污染容器 PID 1。
                pids = json.loads(identities.read_text())
                deadline = time.monotonic() + 5
                try:
                    while os.waitpid(pids["worker"], os.WNOHANG)[0] == 0:
                        # waitpid 已确认它仍是本测试的活子进程，此时 PID 不会被复用。
                        os.kill(pids["worker"], signal.SIGKILL)
                        if time.monotonic() >= deadline:
                            self.fail("fixture worker cleanup timed out")
                        time.sleep(0.01)
                except ChildProcessError:
                    pass

    def test_runner_preserves_numeric_failure_and_all_report_formats(self):
        def fake_generate(args, directory, model, request, label="generate"):
            report = self.make_report(request)
            # 保持 argmax 不变，模拟 TP 与单卡共享但独立 CPU 可识别的数值错误。
            for step in report["steps"]:
                step["logits"] = [value + 1 for value in step["logits"]]
            return 0, report

        def fake_replay(binary, worker, model, directory, devices, label, **kwargs):
            tasks = json.loads((directory / "tasks.json").read_text())
            results = []
            for task, state in zip(tasks, expectations(tasks)):
                item = dict(task=task["name"], op=task["op"], cache=[expected_cache(self.config, state, 0, 1)])
                if "ids" in state:
                    logits = [value+1 for value in prefill(self.config, self.tensors, state["ids"])]
                    item.update(logits=logits, argmax=max(range(17), key=logits.__getitem__))
                results.append(item)
            return 0, dict(status="PASS", backend=DECODE_BACKEND, results=results, after_unload=[EMPTY.copy()],
                workers=[dict(rank=0, device=0, pid=123, cuda_runtime_version=12060, nccl_version=22203)])

        cache = self.root / "cache.txt"
        cache.write_text("CUDA_ARCH:STRING=70\n")
        argv = ["runner", "--generate", "/bin/false", "--sampling-check", "/bin/false", "--decode-check", "/bin/false",
                "--worker", "/bin/false", "--build-cache", str(cache), "--output", str(self.root)]
        golden = dict(status="PASS", cases=[dict(status="PASS", failed_elements=0)]*24)
        with mock.patch("sys.argv", argv), mock.patch("run_qwen3_generation.capture", return_value="test"), \
             mock.patch("run_qwen3_generation.cases", return_value=[("corrupt", dict(tokens=2))]), \
             mock.patch("run_qwen3_generation.make_model", return_value=(self.config, self.tensors)), \
             mock.patch("run_qwen3_generation.run_binary", return_value=(0, golden)), \
             mock.patch("run_qwen3_generation.generate", side_effect=fake_generate), \
             mock.patch("run_qwen3_generation.invoke", side_effect=fake_replay), redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 1)
        output = next(self.root.glob("*_generation_*"))
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "FAIL")
        self.assertIn("CPU", summary["cases"][1]["reason"])
        self.assertGreater(summary["cases"][1]["failed_elements"], 0)
        self.assertTrue(json.loads((output / "corrupt/reference.json").read_text()))
        for suffix in ("md", "json", "csv", "xlsx"):
            self.assertTrue((output / ("summary." + suffix)).is_file())


if __name__ == "__main__":
    unittest.main()
