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
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser, _is_naive_n05_family


class NaiveTensorParallelCliTest(unittest.TestCase):
    def configure(self, options, model_type="naive_n05_flash", saved=None, config=None, environment=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(
                {"model_type": model_type, **(config or {})}))
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
            with patch.dict(os.environ, environment or {}, clear=True), \
                    patch.dict(sys.modules, {"ftllm": fake}), \
                    patch("fastllm_pytools.util._has_cuda_device", return_value=True), \
                    redirect_stdout(io.StringIO()):
                make_normal_llm_model(args)
                self.native_env = dict(os.environ)
            fake.llm.set_cuda_slab.assert_called_once_with(args.cuda_slab)
            self.loaded_model = fake.llm.model.return_value
            return args

    def test_prefill_batching_cli_and_saved_settings(self):
        for options, saved in [(['--tp', '8', '--prefill_batch_tokens', '16384',
                                  '--prefill_max_wait_ms', '20', '--chunked_prefill_size', '8192'], None),
                               ([], {'tp': '8', 'prefill_batch_tokens': 16384,
                                     'prefill_max_wait_ms': 20, 'chunked_prefill_size': 8192})]:
            with self.subTest(saved=saved):
                self.configure(options, saved=saved)
                self.loaded_model.set_prefill_batching.assert_called_once_with(16384, 20)
                self.loaded_model.set_chunked_prefill_size.assert_called_once_with(8192)

    def test_prefill_batching_defaults_and_invalid_values(self):
        self.configure(['--tp', '8'])
        self.loaded_model.set_prefill_batching.assert_not_called()
        for options in [['--prefill_batch_tokens', '-1'], ['--prefill_max_wait_ms', '10'],
                        ['--prefill_batch_tokens', '8192', '--prefill_max_wait_ms', '-1']]:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'prefill'):
                self.configure(options)
        with self.assertRaisesRegex(ValueError, 'prefill'):
            self.configure([], saved={'prefill_batch_tokens': 2147483648})

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

    def test_mimo_dsa_tp_defaults(self):
        args = self.configure(["--tp", "8"], model_type="mimo_v2", config={
            "architectures": ["MiMoV2ForCausalLM"], "enable_dsa": True})
        self.assertEqual(args.atype, "bfloat16")
        self.assertEqual(args.cuda_slab, 16)
        self.assertEqual(args.cache_history, "true")

    def test_mimo_family_requires_dsa(self):
        for enabled in (None, False, "true"):
            self.assertFalse(_is_naive_n05_family(
                {"model_type": "mimo_v2", "enable_dsa": enabled}))
        self.assertTrue(_is_naive_n05_family({
            "architectures": ["MiMoV2DSAForCausalLM"], "enable_dsa": True}))

    def test_serial_and_single_device_defaults(self):
        for options in ([], ["--tp", "1"], ["--device", '["cuda:0", "cuda:1"]']):
            with self.subTest(options=options):
                self.assertEqual(self.configure(options).cuda_slab, 0)

    def test_other_model_defaults(self):
        args = self.configure(["--tp", "2"], model_type="llama")
        self.assertEqual(args.cuda_slab, 0)
        self.assertEqual(args.atype, "float16")

    def test_ep8_cli_and_saved_settings(self):
        for options, saved in [(["--tp", "8", "--ep", "8"], None),
                               ([], {"tp": "8", "ep": 8})]:
            with self.subTest(saved=saved):
                args = self.configure(options, saved=saved)
                self.assertEqual(args.ep, 8)
                self.assertEqual(self.native_env["FASTLLM_EP"], "8")
                self.assertEqual(self.native_env["FASTLLM_TP"], "cuda:0,1,2,3,4,5,6,7")

    def test_ep_explicit_disable_overrides_environment(self):
        self.configure(["--tp", "8", "--ep", "1"], environment={
            "FASTLLM_EP": "8"})
        self.assertEqual(self.native_env["FASTLLM_EP"], "1")

    def test_ep_environment_and_explicit_enable(self):
        args = self.configure(["--tp", "8"], environment={"FASTLLM_EP": "8"})
        self.assertEqual(args.ep, 8)
        self.configure(["--tp", "8", "--ep", "8"], environment={"FASTLLM_EP": "1"})
        self.assertEqual(self.native_env["FASTLLM_EP"], "8")

    def test_ep_requires_eight_tp_devices(self):
        for tp in (None, "1", "4"):
            with self.subTest(tp=tp), self.assertRaisesRegex(ValueError, "requires --tp 8"):
                self.configure(["--ep", "8"] + (["--tp", tp] if tp else []))

    def test_ep_rejects_unsupported_model_and_saved_value(self):
        with self.assertRaisesRegex(ValueError, "currently supports MiMo"):
            self.configure(["--tp", "8", "--ep", "8"], model_type="llama")
        with self.assertRaisesRegex(ValueError, "supports 1"):
            self.configure([], saved={"tp": "8", "ep": 4})

    def test_tp_does_not_enable_ep_implicitly(self):
        args = self.configure(["--tp", "8"])
        self.assertIsNone(args.ep)
        self.assertNotIn("FASTLLM_EP", self.native_env)


if __name__ == "__main__":
    unittest.main()
