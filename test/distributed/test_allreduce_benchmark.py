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
