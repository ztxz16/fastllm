"""不加载CUDA、不编译：检查后端配置、路径确认、统计和失败报告约定。"""
import ast
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from report import REPO, save, timing
from model_compare import confirmed


class MacheteContractTests(unittest.TestCase):
    def test_median_p95_cv(self):
        stats = timing([1, 2, 3, 4, 5])
        self.assertEqual(stats["median_ms"], 3)
        self.assertAlmostEqual(stats["p95_ms"], 4.8)
        self.assertGreater(stats["cv"], 0)

    def test_requires_five_rounds(self):
        for samples in ([1, 2], [0, 1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                timing(samples)

    def test_actual_backend_is_required(self):
        confirmed("[fastllm][linear-backend] requested=machete actual=machete device=0", "machete")
        for text in ("", "requested=machete", "[linear-backend] actual=native",
                     "[linear-backend] actual=machete\n[linear-backend] actual=native"):
            with self.assertRaises(RuntimeError):
                confirmed(text, "machete")

    def test_reports_preserve_partial_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(save(tmp, [{"status": "PASS", "median_ms": .123456},
                                        {"status": "FAIL", "failure": "target not selected"}], {}), "FAIL")
            root = Path(tmp)
            result = json.loads((root / "summary.json").read_text())
            self.assertEqual(result["cases"][0]["median_ms"], .123456)
            self.assertIn("0.123", (root / "summary.csv").read_text())
            with zipfile.ZipFile(root / "summary.xlsx") as archive:
                self.assertIsNone(archive.testzip())

    def test_unpaired_is_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(save(tmp, [{"status": "UNPAIRED"}], {}), "INCOMPLETE")
            self.assertEqual(save(tmp, [{"status": "UNSUPPORTED"}], {}), "INCOMPLETE")

    def test_python_selector_without_loading_library(self):
        source = ast.parse((REPO / "tools/fastllm_pytools/llm.py").read_text())
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and
                     node.name in ("set_linear_backend", "set_linear_backend_trace")]
        namespace = {"os": os}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "selectors", "exec"), namespace)
        with patch.dict(os.environ, {}, clear=True):
            for value in ("native", "machete", "auto"):
                namespace["set_linear_backend"](value)
                self.assertEqual(os.environ["FASTLLM_LINEAR_BACKEND"], value)
            with self.assertRaises(ValueError):
                namespace["set_linear_backend"]("unknown")
            namespace["set_linear_backend_trace"](False)
            self.assertEqual(os.environ["FASTLLM_LINEAR_BACKEND_TRACE"], "0")

    def test_int4_nibble_order_conversion(self):
        for high in range(16):
            for low in range(16):
                source = high * 16 + low
                target = (source >> 4) | ((source << 4) & 255)
                self.assertEqual((target & 15, target >> 4), (high, low))

    def test_offset_semantics(self):
        for bits in (4, 8):
            zero = 1 << (bits - 1)
            scale = .00317
            for q in range(1 << bits):
                self.assertAlmostEqual(scale * (q - zero), scale * q + (-scale * zero))

    def test_standalone_headers_do_not_import_torch(self):
        directory = REPO / "src/devices/cuda/linear/machete"
        for path in directory.glob("*.cuh"):
            self.assertNotIn("torch::", path.read_text())
            self.assertNotIn("#include <torch", path.read_text())


if __name__ == "__main__":
    unittest.main()
