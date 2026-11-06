"""只检查 Python 参数入口，不导入推理库或触发本地编译。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser


class ProcessTPOptionsTest(unittest.TestCase):
    def setUp(self):
        self.parser = make_normal_parser("test")

    def test_existing_tp_defaults_to_thread(self):
        args = self.parser.parse_args(["missing-model", "--tp", "2"])
        self.assertEqual(args.tp_backend, "thread")
        self.assertEqual(args.tp, "2")

    def test_process_aliases(self):
        for flag in ("--tp-backend", "--tp_backend"):
            with self.subTest(flag=flag):
                args = self.parser.parse_args(["missing-model", "--tp", "2", flag, "process"])
                self.assertEqual(args.tp_backend, "process")

    def test_process_fails_before_model_lookup_or_download(self):
        args = self.parser.parse_args(["nonexistent/model", "--tp", "2", "--tp-backend", "process"])
        with self.assertRaisesRegex(NotImplementedError, "No model has been loaded"):
            make_normal_llm_model(args)

    def test_process_without_tp_cannot_fall_back(self):
        args = self.parser.parse_args(["missing-model", "--tp-backend", "process"])
        with self.assertRaises(NotImplementedError):
            make_normal_llm_model(args)

    def test_json_cannot_bypass_backend_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for backend, error in (("process", NotImplementedError), ("invalid", ValueError)):
                with self.subTest(backend=backend):
                    path.write_text(json.dumps({"model": "missing-model", "tp_backend": backend}), encoding="utf-8")
                    args = self.parser.parse_args([str(path)])
                    with self.assertRaises(error):
                        make_normal_llm_model(args)


if __name__ == "__main__":
    unittest.main()
