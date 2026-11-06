"""单层数学参考与验收器回归，不调用 GPU、不编译。"""

import copy
import math
from pathlib import Path
import tempfile
import unittest

from qwen3_layer_reference import forward, half
from run_qwen3_layer import BACKEND, compare, make_fixture, verify, verify_failure
from run_qwen3_shards import encode


class LayerReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.tensors, self.tasks = make_fixture(self.root, dict(bias=True))
        self.metrics = dict(max=0.0, sum=0.0, count=0, failed=0)

    def test_zero_projection_is_identity(self):
        for name, tensor in self.tensors.items():
            if "norm.weight" not in name:
                tensor["data"] = bytes(len(tensor["data"]))
        for task in self.tasks:
            self.assertEqual(forward(self.config, self.tensors, task), [half(x) for x in task["input"]])

    def test_output_bias_added_once(self):
        for name, tensor in self.tensors.items():
            if "norm.weight" not in name:
                tensor["data"] = bytes(len(tensor["data"]))
            if name.endswith(("self_attn.o_proj.bias", "mlp.down_proj.bias")):
                tensor["data"] = encode([0.25] * math.prod(tensor["shape"]), tensor["dtype"])
        for task in self.tasks:
            amount = 0.5 if task["stage"] == 0 else 0.25
            self.assertEqual(forward(self.config, self.tensors, task), [half(half(x) + amount) for x in task["input"]])

    def test_full_matches_composed_stages(self):
        task = self.tasks[2]
        attention = forward(self.config, self.tensors, dict(task, stage=1))
        composed = forward(self.config, self.tensors, dict(task, stage=2, input=attention))
        self.assertEqual(composed, forward(self.config, self.tensors, task))

    def test_future_tokens_do_not_change_prefix(self):
        original = forward(self.config, self.tensors, self.tasks[2])
        changed = forward(self.config, self.tensors, self.tasks[3])
        self.assertEqual(original[:-self.config["hidden_size"]], changed[:-self.config["hidden_size"]])
        self.assertNotEqual(original[-self.config["hidden_size"]:], changed[-self.config["hidden_size"]:])

    def test_single_token_attention_independent_of_qk(self):
        task = dict(self.tasks[0], tokens=1, input=self.tasks[0]["input"][:self.config["hidden_size"]])
        expected = forward(self.config, self.tensors, task)
        for name, tensor in self.tensors.items():
            if name.endswith(("q_proj.weight", "k_proj.weight")):
                tensor["data"] = bytes(len(tensor["data"]))
        self.assertEqual(expected, forward(self.config, self.tensors, task))

    def test_corruption_and_nan_are_counted(self):
        for value in (1.0, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(AssertionError):
                    compare([value], [0.0], self.metrics, "bad")
        self.assertEqual(self.metrics["failed"], 2)

    def test_nonfinite_reference_is_always_rejected(self):
        for expected in (float("nan"), float("inf"), -float("inf")):
            for actual in (0.0, expected):
                for exact in (False, True):
                    with self.subTest(expected=expected, actual=actual, exact=exact):
                        with self.assertRaises(AssertionError):
                            compare([actual], [expected], self.metrics, "invalid reference", exact=exact)
        self.assertEqual(self.metrics["failed"], 12)

    def test_exact_rank_check_is_stricter(self):
        compare([0.001], [0.0], self.metrics, "within tolerance")
        with self.assertRaises(AssertionError):
            compare([0.001], [0.0], self.metrics, "rank mismatch", exact=True)

    def report(self):
        return dict(status="PASS", backend=BACKEND, workers=[
            dict(rank=i, device=i, pid=100+i, cuda_runtime_version=12060, nccl_version=22203) for i in range(2)],
            results=[dict(task=t["name"], outputs=[t["input"], t["input"]]) for t in self.tasks])

    def test_missing_rank_task_or_wrong_backend_is_rejected(self):
        report = self.report()
        verify(report, [0, 1], self.tasks)
        for changed in (dict(report, backend="cpu"), dict(report, results=report["results"][:-1])):
            with self.assertRaises(AssertionError):
                verify(changed, [0, 1], self.tasks)
        changed = copy.deepcopy(report)
        changed["results"][0]["outputs"].pop()
        with self.assertRaises(AssertionError):
            verify(changed, [0, 1], self.tasks)

    def test_expected_failure_requires_reason_and_cleanup(self):
        spec = dict(reason="unsupported config", group_failed=True)
        report = dict(status="FAIL", backend=BACKEND, error="unsupported config", group_failed=True,
                      workers_reaped=True, shutdown_complete=True)
        verify_failure(1, report, spec)
        for code, changed in ((-9, report), (1, dict(report, workers_reaped=False)), (1, dict(report, error="unrelated"))):
            with self.assertRaises(AssertionError):
                verify_failure(code, changed, spec)

    def test_local_rejection_requires_successful_shutdown(self):
        spec = dict(reason="invalid request", group_failed=False)
        report = dict(status="FAIL", backend=BACKEND, error="invalid request", group_failed=False, workers_reaped=False)
        with self.assertRaises(AssertionError):
            verify_failure(1, report, spec)
        report["shutdown_complete"] = True
        verify_failure(1, report, spec)
        for changed in (dict(report, shutdown_complete=False), dict(report, cleanup_error="rank exit timeout")):
            with self.assertRaises(AssertionError):
                verify_failure(1, changed, spec)


if __name__ == "__main__":
    unittest.main()
