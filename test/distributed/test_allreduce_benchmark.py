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
