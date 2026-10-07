import io
import json
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser


def write_gguf(path, architecture):
    def string(value):
        value = value.encode()
        return struct.pack("<Q", len(value)) + value
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) +
                     string("general.architecture") + struct.pack("<I", 8) +
                     string(architecture))


class CudaWeightSlabDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gguf = self.root / "qwen.gguf"
        write_gguf(self.gguf, "qwen4exp")
        self.llm = MagicMock()
        self.llm.model.return_value.get_max_input_len.return_value = 8192
        self.llm.model.return_value.get_max_batch.return_value = 1

    def load(self, path=None, extra=(), *, device="cuda:0", moe="numa"):
        args = make_normal_parser("test").parse_args([
            str(path or self.gguf), "--device", device, "--moe_device", moe,
            "--threads", "1", "--tokens", "8192", *extra])
        with patch.dict(sys.modules, {"ftllm": SimpleNamespace(llm=self.llm)}), \
                patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            make_normal_llm_model(args)
        return args

    def test_host_offload_gguf_sets_slab_before_constructing_model(self):
        for moe in ("cpu", "numa", "disk"):
            with self.subTest(moe=moe):
                self.llm.reset_mock()
                args = self.load(moe=moe)
                self.assertEqual(args.cuda_slab, 64)
                self.llm.set_cuda_slab.assert_called_once_with(64)
                names = [entry[0] for entry in self.llm.mock_calls]
                self.assertLess(names.index("set_cuda_slab"), names.index("model"))

    def test_explicit_zero_and_custom_size_override_automatic_default(self):
        for size in (0, 32, 128):
            with self.subTest(size=size):
                args = self.load(extra=("--cuda_slab", str(size)))
                self.assertEqual(args.cuda_slab, size)
                self.llm.set_cuda_slab.assert_called_with(size)

    def test_saved_json_zero_disables_slab_after_an_automatic_model(self):
        self.load()
        config = self.root / "saved.json"
        config.write_text(json.dumps({"path": str(self.gguf), "cuda_slab": 0}))
        self.load(config)
        self.assertEqual(self.llm.set_cuda_slab.call_args_list, [call(64), call(0)])

    def test_pure_cpu_and_other_gguf_models_keep_slab_disabled(self):
        self.load(device="cpu", moe="cpu")
        self.llm.set_cuda_slab.assert_called_with(0)
        other = self.root / "other.gguf"
        write_gguf(other, "llama")
        self.load(other)
        self.llm.set_cuda_slab.assert_called_with(0)

    def test_hf_host_offload_does_not_inherit_gguf_default(self):
        self.load()
        model = self.root / "hf"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
        self.load(model)
        self.assertEqual(self.llm.set_cuda_slab.call_args_list, [call(64), call(0)])

    def test_gpu_expert_default_and_explicit_opt_out(self):
        self.load(moe="cuda:0")
        self.llm.set_cuda_slab.assert_called_with(225)
        self.load(moe="cuda:0", extra=("--cuda_slab", "0"))
        self.llm.set_cuda_slab.assert_called_with(0)

    def test_layered_gpu_experts_keep_existing_slab_default(self):
        self.load(extra=("--moe_device_layers", "2"))
        self.llm.set_cuda_slab.assert_called_with(225)

    def test_glm_gguf_resident_tp_packs_weights_and_respects_explicit_override(self):
        glm = self.root / "glm.gguf"
        write_gguf(glm, "glm5next")
        tp = ("--tp", "0,1", "--moe_device_layers", "30")
        self.load(glm, extra=tp)
        self.llm.set_cuda_slab.assert_called_with(64)
        for size in (0, 128):
            self.load(glm, extra=tp+("--cuda_slab", str(size)))
            self.llm.set_cuda_slab.assert_called_with(size)
        self.load(glm, extra=("--tp", "0,1"))
        self.llm.set_cuda_slab.assert_called_with(0)

    def test_existing_multicuda_defaults_and_explicit_zero(self):
        for model_type, devices, expected in [
                ("deepseek_v4", "0,1", 256),
                ("laguna", "0,1,2,3", 96)]:
            with self.subTest(model_type=model_type):
                model = self.root / model_type
                model.mkdir()
                (model / "config.json").write_text(json.dumps({"model_type": model_type}))
                tp = ("--tp", "cuda:"+devices)
                self.load(model, moe="multicuda:"+devices, extra=tp)
                self.llm.set_cuda_slab.assert_called_with(expected)
                self.load(model, moe="multicuda:"+devices,
                          extra=tp+("--cuda_slab", "0"))
                self.llm.set_cuda_slab.assert_called_with(0)


if __name__ == "__main__":
    unittest.main()
