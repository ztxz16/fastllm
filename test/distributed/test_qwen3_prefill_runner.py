"""完整 prefill 数学参考和验收判定回归，不编译、不调用 GPU。"""

import copy
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from qwen3_layer_reference import linear, matrix, norm, prefill
from run_qwen3_prefill import BACKEND, compare_logits, invoke, make_model, make_tasks, prepare_reload, verify, verify_failure
from run_qwen3_shards import encode


class PrefillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.tensors = make_model(self.root / "model", {})
        self.tasks = make_tasks(self.config, {})
        self.metrics = dict(max=0.0, sum=0.0, count=0, failed=0)

    def test_zero_decoder_projections_leave_embedding_for_final_head(self):
        for name, tensor in self.tensors.items():
            if name.startswith("model.layers.") and "norm.weight" not in name:
                tensor["data"] = bytes(len(tensor["data"]))
        for ids in ([0], [0, 2, self.config["vocab_size"] - 1]):
            embedding = matrix(self.tensors["model.embed_tokens.weight"])[ids[-1]]
            normalized = norm(embedding, matrix(self.tensors["model.norm.weight"]), self.config["rms_norm_eps"])
            expected = linear([normalized], matrix(self.tensors["lm_head.weight"]), rounded=False)[0]
            self.assertEqual(prefill(self.config, self.tensors, ids), expected)

    def test_all_decoder_layers_affect_logits(self):
        ids = self.tasks[0]["token_ids"]
        expected = prefill(self.config, self.tensors, ids)
        for layer in range(self.config["num_hidden_layers"]):
            tensors = copy.deepcopy(self.tensors)
            for name, tensor in tensors.items():
                if name.startswith(f"model.layers.{layer}.") and "norm.weight" not in name:
                    tensor["data"] = bytes(len(tensor["data"]))
            self.assertNotEqual(prefill(self.config, tensors, ids), expected)

    def test_tied_head_matches_explicit_embedding_copy(self):
        config, tensors = make_model(self.root / "tied", dict(tied=True))
        expected = prefill(config, tensors, [1, 5, 0])
        tensors["lm_head.weight"] = copy.deepcopy(tensors["model.embed_tokens.weight"])
        self.assertEqual(prefill(config, tensors, [1, 5, 0]), expected)

    def test_final_head_changes_vocabulary_order(self):
        ids = [1, 4, 8]
        expected = prefill(self.config, self.tensors, ids)
        tensor = self.tensors["lm_head.weight"]
        rows = matrix(tensor)
        tensor["data"] = encode([v for row in reversed(rows) for v in row], tensor["dtype"])
        self.assertEqual(prefill(self.config, self.tensors, ids), list(reversed(expected)))

    def test_reload_changes_logits_and_keeps_request(self):
        original_ids = self.tasks[0]["token_ids"][:]
        _, updated = prepare_reload(self.root, self.config, self.tensors, self.tasks)
        self.assertEqual(len(self.tasks), 2)
        self.assertEqual(self.tasks[1]["token_ids"], original_ids)
        before = prefill(self.config, self.tensors, original_ids)
        after = prefill(self.config, updated, original_ids)
        self.assertTrue(any(value != 0 for value in before))
        self.assertEqual(after, [-value for value in before])
        with self.assertRaises(AssertionError):
            compare_logits(before, after, self.metrics, "stale reload", 0.008, 0.008)

    def test_nonfinite_actual_or_reference_never_passes(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for actual, expected in ((value, 0.0), (0.0, value), (value, value)):
                with self.assertRaises(AssertionError):
                    compare_logits([actual], [expected], self.metrics, "nonfinite", 0.05, 0.01)
        self.assertEqual(self.metrics["failed"], 9)

    def report(self):
        logits = prefill(self.config, self.tensors, [1, 5])
        return dict(status="PASS", backend=BACKEND,
            workers=[dict(rank=i, device=i, pid=100+i, cuda_runtime_version=12060, nccl_version=22203) for i in range(2)],
            results=[dict(task=task["name"], logits=logits[:], argmax=max(range(len(logits)), key=logits.__getitem__))
                     for task in self.tasks])

    def test_missing_vocab_tail_or_wrong_argmax_rejected(self):
        report = self.report()
        verify(report, [0, 1], self.tasks, self.config["vocab_size"])
        for field, value in (("logits", report["results"][0]["logits"][:-1]), ("argmax", -1)):
            changed = copy.deepcopy(report)
            changed["results"][0][field] = value
            with self.assertRaises(AssertionError):
                verify(changed, [0, 1], self.tasks, self.config["vocab_size"])

    def test_wrong_backend_rank_or_task_rejected(self):
        report = self.report()
        for changed in (dict(report, backend="cpu"), dict(report, results=report["results"][:-1]),
                        dict(report, workers=report["workers"][:1])):
            with self.assertRaises(AssertionError):
                verify(changed, [0, 1], self.tasks, self.config["vocab_size"])

    def test_rejection_requires_reason_and_cleanup(self):
        spec = dict(reason="token ID outside vocabulary", group_failed=True)
        report = dict(status="FAIL", backend=BACKEND, error=spec["reason"], group_failed=True,
                      workers_reaped=True, shutdown_complete=True)
        verify_failure(1, report, spec)
        for changed in (dict(report, workers_reaped=False), dict(report, shutdown_complete=False),
                        dict(report, error="unrelated"), dict(report, cleanup_error="timeout")):
            with self.assertRaises(AssertionError):
                verify_failure(1, changed, spec)

    def test_missing_executable_keeps_command_and_error(self):
        with self.assertRaises(FileNotFoundError):
            invoke(self.root / "missing", self.root / "worker", self.root / "model", self.root, [0, 1], "tp")
        self.assertTrue((self.root / "tp.command.txt").is_file())
        self.assertIn("FileNotFoundError", (self.root / "tp.stderr.log").read_text())

    def test_timeout_and_interrupt_preserve_logs(self):
        for error in (KeyboardInterrupt(), subprocess.TimeoutExpired("checker", 1500)):
            with mock.patch("run_qwen3_prefill.wait_case", side_effect=error):
                with self.assertRaises(type(error)):
                    invoke(Path("/bin/false"), self.root / "worker", self.root / "model", self.root, [0, 1], "tp")
            self.assertIn(type(error).__name__, (self.root / "tp.stderr.log").read_text())


if __name__ == "__main__":
    unittest.main()
