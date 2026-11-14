"""纯 CPU 测试：验证报告契约，不加载 CUDA、不构建扩展。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).with_name("benchmark_allreduce.py")


class PlanTests(unittest.TestCase):
    def test_disjoint_batch_windows_have_zero_overlap(self):
        from allreduce_bench.concurrency import overlap_metrics
        rows = [dict(iterations=10, ranks=[dict(batch_start_ns=10, batch_end_ns=20)]),
                dict(iterations=10, ranks=[dict(batch_start_ns=30, batch_end_ns=40)])]
        self.assertEqual(overlap_metrics(rows)["overlap_ratio"], 0)
        rows[1]["ranks"][0].update(batch_start_ns=15, batch_end_ns=25)
        self.assertEqual(overlap_metrics(rows)["overlap_ratio"], 0.5)

    def test_timing_uses_slowest_rank_and_rejects_missing_samples(self):
        from allreduce_bench.timing import rank_max_samples, statistics_for
        values = rank_max_samples([{"samples_us": [1, 10, 3]}, {"samples_us": [4, 2, 6]}])
        self.assertEqual(values, [4, 10, 6])
        self.assertEqual(statistics_for(values)["median_us"], 6)
        with self.assertRaises(ValueError):
            rank_max_samples([{"samples_us": [1, 2]}, {"samples_us": [3]}])
        with self.assertRaises(ValueError):
            statistics_for([0, 1])

    def test_reference_detects_stale_results_and_nonfinite_corruption(self):
        from array import array
        from allreduce_bench.data import compare, inputs
        first = inputs(32, "changed", 12, 0)
        second = inputs(32, "changed", 12, 1)
        old_output = array("f", [a + b for a, b in zip(*first)])
        self.assertEqual(compare(old_output, first)["failed_elements"], 0)
        self.assertEqual(compare(old_output, second)["failed_elements"], 32)
        special = inputs(14, "special", 12)
        output = array("f", [a + b for a, b in zip(*special)])
        self.assertEqual(compare(output, special)["failed_elements"], 0)
        output[0] = 0
        self.assertEqual(compare(output, special)["failed_elements"], 1)

    def test_missing_peer_result_cannot_be_reported_as_pass(self):
        from allreduce_bench.execution import collect
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(stage="smoke", backend="ft", mode="eager", devices=[0, 1],
                       cases=[dict(case="one", n=4), dict(case="two", n=8)])
            first = dict(case="one", path="ft_custom", status="PASS", max_abs_error=0,
                         mean_abs_error=0, failed_elements=0)
            second = dict(first, case="two")
            (root / "rank0.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
            (root / "rank1.jsonl").write_text(json.dumps(first) + "\n")
            rows = collect(job, root)
            self.assertEqual([row["status"] for row in rows], ["PASS", "FAIL"])

    def test_preflight_start_failure_keeps_all_report_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"
            command = [sys.executable, str(SCRIPT), "preflight", "--output", str(output),
                       "--python", "/nonexistent/allreduce-python"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("allreduce-python", report["cases"][0]["reason"])
            for suffix in ("json", "md", "csv", "xlsx"):
                self.assertTrue((output / f"summary.{suffix}").is_file())

    def test_plan_is_not_gpu_pass_and_never_overwrites_history(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan"
            command = [sys.executable, str(SCRIPT), "plan", "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["status"], "PLANNED")
            self.assertTrue(all(row["status"] == "PLANNED" for row in report["cases"]))
            for suffix in ("json", "md", "csv", "xlsx"):
                self.assertTrue((output / f"summary.{suffix}").is_file())
            with zipfile.ZipFile(output / "summary.xlsx") as book:
                self.assertIsNone(book.testzip())
            before = (output / "summary.json").read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(before, (output / "summary.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
