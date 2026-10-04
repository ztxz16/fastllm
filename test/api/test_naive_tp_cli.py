import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser


class NaiveTensorParallelCliTest(unittest.TestCase):
    def configure(self, options, model_type="naive_n05_flash", saved=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model_type": model_type}))
            model_path = str(root)
            if saved is not None:
                path = root / "launch.json"
                path.write_text(json.dumps({"model": str(root), **saved}))
                model_path = str(path)
            args = make_normal_parser("test").parse_args([model_path, "-t", "4", *options])
            fake = types.ModuleType("ftllm")
            fake.llm = MagicMock()
            fake.llm.model.return_value.get_max_input_len.return_value = 4096
            fake.llm.model.return_value.get_max_batch.return_value = 1
            with patch.dict(os.environ, {}, clear=True), \
                    patch.dict(sys.modules, {"ftllm": fake}), \
                    patch("fastllm_pytools.util._has_cuda_device", return_value=True), \
                    redirect_stdout(io.StringIO()):
                make_normal_llm_model(args)
            fake.llm.set_cuda_slab.assert_called_once_with(args.cuda_slab)
            return args

    def test_tp_defaults_and_explicit_slab(self):
        for ranks in (2, 4, 8):
            for slab in (None, 0, 32):
                with self.subTest(ranks=ranks, slab=slab):
                    options = ["--tp", str(ranks)]
                    if slab is not None:
                        options += ["--cuda_slab", str(slab)]
                    args = self.configure(options)
                    self.assertEqual(args.cuda_slab, 16 if slab is None else slab)
                    self.assertEqual(args.atype, "bfloat16")

    def test_saved_explicit_zero(self):
        args = self.configure([], saved={"tp": "8", "cuda_slab": 0})
        self.assertEqual(args.cuda_slab, 0)

    def test_serial_and_single_device_defaults(self):
        for options in ([], ["--tp", "1"], ["--device", '["cuda:0", "cuda:1"]']):
            with self.subTest(options=options):
                self.assertEqual(self.configure(options).cuda_slab, 0)

    def test_other_model_defaults(self):
        args = self.configure(["--tp", "2"], model_type="llama")
        self.assertEqual(args.cuda_slab, 0)
        self.assertEqual(args.atype, "float16")


if __name__ == "__main__":
    unittest.main()
