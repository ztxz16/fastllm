"""请求 KV/Decode 验收判定回归；只运行 Python，不触发编译或 GPU。"""

import copy
from contextlib import redirect_stdout
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from qwen3_layer_reference import prefill
from run_qwen3_prefill import compare_logits, make_model
from run_qwen3_decode import (BACKEND, EMPTY, cases, compare_results, expectations, expected_cache,
                              main, make_tasks, verify, verify_failure)


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config, self.tensors = make_model(Path(self.temp.name) / "model", dict(layers=1))
        self.tasks = make_tasks(self.config, dict(tokens=2, steps=2))
        self.metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
        self.reference = {}

    def report(self, devices):
        results = []
        eager_decodes = 0
        for task, state in zip(self.tasks, expectations(self.tasks)):
            result = dict(task=task["name"], op=task["op"],
                cache=[expected_cache(self.config, state, rank, len(devices)) for rank in range(len(devices))])
            if "ids" in state:
                key = (tuple(state["ids"]), state["rope_position"])
                if key not in self.reference:
                    self.reference[key] = prefill(self.config, self.tensors, *key)
                values = self.reference[key][:]
                result.update(logits=values, argmax=max(range(len(values)), key=values.__getitem__))
            if task['op'] == 'decode':
                eager_decodes += 1
            elif task['op'] == 'unload':
                eager_decodes = 0
            result['graph'] = [dict(enabled=False, active=False, captures=0, replays=0, warmups=0,
                                    eager_decodes=eager_decodes, workspace_bytes=0) for _ in devices]
            results.append(result)
        return dict(status="PASS", backend=BACKEND, execution="eager",
            graph_after_unload=[dict(enabled=False, active=False, captures=0, replays=0, warmups=0,
                                     eager_decodes=0, workspace_bytes=0) for _ in devices], results=results, after_unload=[EMPTY.copy() for _ in devices],
            workers=[dict(rank=i, device=device, pid=100+i, cuda_runtime_version=12060, nccl_version=22203)
                     for i, device in enumerate(devices)])

    def check_numbers(self, tp, single):
        return compare_results(tp, single, self.tasks, self.config, self.tensors, None,
                               self.metrics, 0.008, 0.008, {})

    def test_complete_sequences_and_reference(self):
        tp, single = self.report([0, 1]), self.report([0])
        verify(tp, [0, 1], self.tasks, self.config)
        verify(single, [0], self.tasks, self.config)
        refs = self.check_numbers(tp, single)
        self.assertEqual(len(refs), 3)
        self.assertEqual(self.metrics["failed"], 0)
        self.assertGreater(self.metrics["count"], 0)

    def test_stale_or_leaked_cache_rejected(self):
        original = self.report([0, 1])
        for index, field, value in ((1, "tokens", 0), (1, "capacity", 0), (1, "device_bytes", 0),
                                    (2, "tokens", 99), (len(self.tasks)-1, "device_bytes", 4)):
            report = copy.deepcopy(original)
            report["results"][index]["cache"][0][field] = value
            with self.assertRaisesRegex(AssertionError, "KV"):
                verify(report, [0, 1], self.tasks, self.config)
        original["after_unload"][0]["active"] = True
        with self.assertRaisesRegex(AssertionError, "unload"):
            verify(original, [0, 1], self.tasks, self.config)

    def test_both_backends_wrong_cpu_reference_still_rejects(self):
        tp, single = self.report([0, 1]), self.report([0])
        for report in (tp, single):
            report["results"][1]["logits"][0] += 1
        with self.assertRaisesRegex(AssertionError, "CPU"):
            self.check_numbers(tp, single)

    def test_failed_numeric_check_saves_reference_and_reports(self):
        tp, single = self.report([0, 1]), self.report([0])
        for report in (tp, single):
            result = report["results"][1]
            result["logits"][0] += 1
            result["argmax"] = max(range(len(result["logits"])), key=result["logits"].__getitem__)
        root = Path(self.temp.name)
        cache = root / "CMakeCache.txt"
        cache.write_text("CUDA_ARCH:STRING=70\n")
        args = ["run_qwen3_decode.py", "--check", "/bin/false", "--worker", "/bin/false",
                "--build-cache", str(cache), "--output", str(root)]
        with mock.patch("sys.argv", args), mock.patch("run_qwen3_decode.capture", return_value="test"), \
             mock.patch("run_qwen3_decode.cases", return_value=[("numeric_failure", dict(layers=1, tokens=2, steps=2))]), \
             mock.patch("run_qwen3_decode.invoke", side_effect=[(0, tp), (0, single)]), redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 1)
        output = next(root.glob("*_decode_*"))
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "FAIL")
        self.assertIn("CPU", summary["cases"][0]["reason"])
        self.assertGreater(summary["cases"][0]["failed_elements"], 0)
        refs = json.loads((output / "numeric_failure/reference.json").read_text())
        self.assertEqual(refs[0]["token_ids"], self.tasks[1]["token_ids"])
        self.assertNotEqual(refs[0]["logits"], tp["results"][1]["logits"])
        for suffix in ("json", "md", "csv", "xlsx"):
            self.assertGreater((output / ("summary." + suffix)).stat().st_size, 0)

    def test_reference_is_sensitive_to_missing_and_stale_history(self):
        ids = [1, 4, 7, 10, 13, 15]
        correct = prefill(self.config, self.tensors, ids)
        wrong = [prefill(self.config, self.tensors, ids[-1:], len(ids)-1),
                 prefill(self.config, self.tensors, [2, 3, 5, 6, 8, ids[-1]])]
        for values in wrong:
            with self.assertRaises(AssertionError):
                compare_logits(values, correct, self.metrics, "lost/stale history", 0.008, 0.008)

    def test_long_fixture_detects_truncating_early_kv(self):
        spec = dict(cases())["max_context"]
        config, tensors = make_model(Path(self.temp.name) / "long-model", spec)
        tasks = make_tasks(config, spec)
        self.assertEqual(tasks[1]["token_ids"], [14] * 128)
        self.assertTrue(all(task["token_id"] == 16 for task in tasks if task["op"] == "decode"))
        correct = prefill(config, tensors, [14] * 128 + [16] * 128)
        truncated = prefill(config, tensors, [16] * 128, 128)
        with self.assertRaises(AssertionError):
            compare_logits(truncated, correct, self.metrics, "truncated history", 0.008, 0.008)

    def test_cached_decode_differs_from_recompute_rejected(self):
        tp, single = self.report([0, 1]), self.report([0])
        # 同时篡改单卡和 TP 的重算输出：单卡/TP 比较仍相同，但缓存/重算必须失败。
        for report in (tp, single):
            report["results"][2]["logits"][0] += 1
        with self.assertRaisesRegex(AssertionError, "cached vs full"):
            self.check_numbers(tp, single)

    def test_repeat_is_stricter_than_numeric_tolerance(self):
        tp, single = self.report([0, 1]), self.report([0])
        index = next(i for i, task in enumerate(self.tasks) if "repeat_of" in task)
        for report in (tp, single):
            report["results"][index]["logits"][0] += 1e-5
        with self.assertRaisesRegex(AssertionError, "repeat"):
            self.check_numbers(tp, single)

    def test_missing_nonfinite_or_wrong_identity_fails(self):
        original = self.report([0, 1])
        for bad in (float("nan"), float("inf")):
            report = copy.deepcopy(original)
            report["results"][1]["logits"][0] = bad
            with self.assertRaisesRegex(AssertionError, "logits"):
                verify(report, [0, 1], self.tasks, self.config)
        for field, value in (("task", "wrong"), ("op", "decode"), ("argmax", -1), ("logits", [])):
            report = copy.deepcopy(original)
            report["results"][1][field] = value
            with self.assertRaises(AssertionError):
                verify(report, [0, 1], self.tasks, self.config)

    def test_error_must_occur_on_intended_task_and_reap(self):
        spec = dict(reason="capacity exhausted", group_failed=True)
        report = dict(status="FAIL", backend=BACKEND, error=spec["reason"], group_failed=True,
                      workers_reaped=True, shutdown_complete=True, current_task=self.tasks[-1]["name"])
        verify_failure(1, report, spec, self.tasks)
        for changed in (dict(report, current_task="earlier"), dict(report, workers_reaped=False),
                        dict(report, shutdown_complete=False), dict(report, cleanup_error="timeout")):
            with self.assertRaises(AssertionError):
                verify_failure(1, changed, spec, self.tasks)

    def test_uneven_and_replicated_kv_accounting(self):
        state = dict(active=True, tokens=2, capacity=8, position_begin=123)
        sizes = [expected_cache(self.config, state, rank, 2)["device_bytes"] for rank in (0, 1)]
        self.assertEqual(sizes, [256, 128])  # 1 layer * 8 slots * K/V * (2/1 heads) * 4 dims * FP16.
        config = dict(self.config, num_key_value_heads=1)
        self.assertEqual([expected_cache(config, state, rank, 2)["device_bytes"] for rank in (0, 1)], [128, 128])

    def test_plans_cover_boundary_and_preserve_cache_during_recompute(self):
        names = [name for name, _ in cases()]
        self.assertEqual(len(names), len(set(names)))
        for _, spec in cases():
            tasks = make_tasks(self.config, spec)
            self.assertLessEqual(len(tasks), 8192)
            self.assertEqual(len(tasks), len({task["name"] for task in tasks}))
            if "reject" in spec:
                continue
            states = expectations(tasks)
            for index, task in enumerate(tasks):
                if task["op"] == "prefill":
                    for field in ("active", "tokens", "capacity", "position_begin"):
                        self.assertEqual(states[index][field], states[index-1][field])
            if spec.get("capacity") == 2048:
                self.assertEqual(max(state["tokens"] for state in states), 2048)

    def test_unload_and_reload_reset_position_and_model(self):
        tasks = make_tasks(self.config, dict(reload=True, position=11, tokens=1, steps=1))
        states = expectations(tasks)
        for task, state in zip(tasks, states):
            if task["op"] in ("load", "unload"):
                self.assertFalse(state["active"])
                self.assertEqual(state["tokens"], 0)
            if task["name"] == "cycle-1-step-0":
                self.assertEqual(state["model"], "reloaded")
                self.assertEqual(state["rope_position"], 11)
                self.assertEqual(state["tokens"], 1)


if __name__ == "__main__":
    unittest.main()
