"""Small CPU fixtures for the NVFP4 exporter; requires torch and safetensors."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file


EXPORTER = Path(__file__).resolve().parents[1] / "tools" / "naive_n05_export_nvfp4.py"
spec = importlib.util.spec_from_file_location("naive_n05_export_nvfp4", EXPORTER)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class Nvfp4ExportTest(unittest.TestCase):
    def test_round_to_even_and_nibble_order(self):
        positive = [.25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6]
        weight = torch.tensor([positive + [-value for value in positive]])
        packed, scales, global_scale = exporter.quantize(weight)
        self.assertEqual(packed.tolist(), [[0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE]])
        expected = torch.tensor([[0, 1, 1, 2, 2, 4, 4, 6, 0, -1, -1, -2, -2, -4, -4, -6]])
        torch.testing.assert_close(exporter.dequantize(packed, scales, global_scale), expected.float())

    def test_export_zero_and_scaled_experts_preserves_other_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "output"
            source.mkdir()
            config = {"model_type": "naive_n05_flash", "quantization_config": {
                "quant_method": "fp8", "weight_block_size": [128, 128]}}
            (source / "config.json").write_text(json.dumps(config))
            (source / "tokenizer_config.json").write_text('{"test": true}\n')
            zero = "model.layers.1.mlp.experts.0.gate_proj.weight"
            scaled = "model.layers.1.mlp.experts.1.down_proj.weight"
            levels = torch.tensor([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6, 0])
            dense = {"model.norm.weight": torch.tensor([1., -2.], dtype=torch.bfloat16),
                     "model.layers.1.mlp.gate.weight": torch.tensor([[.125, -.25]])}
            tensors = {zero: torch.zeros(128, 128).to(torch.float8_e4m3fn),
                       zero + "_scale_inv": torch.ones(1, 1),
                       scaled: levels.repeat(128, 16).to(torch.float8_e4m3fn),
                       scaled + "_scale_inv": torch.tensor([[1., 2.]]), **dense}
            save_file(tensors, source / "model.safetensors")
            command = [sys.executable, str(EXPORTER), "--model", str(source),
                       "--output", str(output), "--device", "cpu"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            converted = load_file(output / "model.safetensors")
            expected_names = set(dense)
            for name in (zero, scaled):
                expected_names.update((name, name + "_scale", name + "_scale_2"))
                self.assertEqual(converted[name].dtype, torch.uint8)
                self.assertEqual(converted[name + "_scale"].dtype, torch.float8_e4m3fn)
                self.assertEqual(converted[name + "_scale_2"].dtype, torch.float32)
                reconstructed = exporter.dequantize(
                    converted[name], converted[name + "_scale"], converted[name + "_scale_2"])
                expected = tensors[name].float()
                if name == scaled:
                    expected[:, 128:] *= 2
                torch.testing.assert_close(reconstructed, expected)
            self.assertEqual(set(converted), expected_names)
            report = json.loads((output / "nvfp4_export.json").read_text())
            self.assertEqual(report["expert_matrices"], 2)
            self.assertEqual(report["weight_error_samples"], [{"name": zero, "relative_rmse": 0.0}])
            for name, tensor in dense.items():
                self.assertEqual(converted[name].dtype, tensor.dtype)
                self.assertTrue(torch.equal(converted[name].view(torch.uint8), tensor.view(torch.uint8)))
                self.assertEqual(report["nonexpert_sha256"][name],
                                 hashlib.sha256(tensor.view(torch.uint8).numpy()).hexdigest())
            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(index["weight_map"], dict.fromkeys(expected_names, "model.safetensors"))
            self.assertEqual(index["metadata"]["total_size"],
                             sum(t.numel() * t.element_size() for t in converted.values()))
            self.assertEqual((output / "tokenizer_config.json").read_bytes(),
                             (source / "tokenizer_config.json").read_bytes())
            quantization = json.loads((output / "config.json").read_text())["quantization_config"]
            self.assertEqual(quantization["quant_algo"], "NVFP4")
            self.assertEqual(quantization["group_size"], 16)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
