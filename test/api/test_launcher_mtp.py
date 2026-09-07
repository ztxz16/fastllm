"""MTP discovery must reflect files the loader will actually use, without a GPU."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools.launcher import LauncherRuntime, create_launcher_app, recommend_launch_config
from fastllm_pytools.launcher_mtp import detect_mtp_support
from fastllm_pytools.tui import build_fastllm_argv, config_from_dict


MTP_NAMES = {
    "mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    *{"mtp.layers.0." + suffix for suffix in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    )},
}


def write_safetensors(path, names):
    header = {name: {"dtype": "F16", "shape": [1], "data_offsets": [i * 2, i * 2 + 2]}
              for i, name in enumerate(sorted(names))}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0\0" * len(names))


def write_mtp_checkpoint(directory):
    path = Path(directory)
    path.mkdir(exist_ok=True)
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "text_config": {"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1}}
    (path / "config.json").write_text(json.dumps(config))
    write_safetensors(path / "model.safetensors", MTP_NAMES)
    return config


def cuda_hardware():
    return {"cpu": {"available": 4}, "memory": {"available": 64 * 1024 ** 3},
            "gpus": [{"index": 0, "memoryTotalMiB": 24576, "memoryFreeMiB": 24576}],
            "numa": [], "build": {"USE_CUDA": True}}


def write_mtp_gguf(path, names=None):
    def string(value):
        value = value.encode()
        return struct.pack("<Q", len(value)) + value
    if names is None:
        names = ["blk.48." + suffix + ".weight" for suffix in (
            "nextn.eh_proj", "nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm",
            "attn_norm", "post_attention_norm", "attn_q", "attn_k", "attn_v",
            "attn_output", "attn_q_norm", "attn_k_norm", "ffn_gate", "ffn_up", "ffn_down",
        )]
    metadata = string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
    for key, value in (("block_count", 49), ("nextn_predict_layers", 1)):
        metadata += string("qwen35." + key) + struct.pack("<II", 4, value)
    directory = b"".join(string(name) + struct.pack("<IQIQ", 1, 1, 0, i * 4)
                         for i, name in enumerate(names))
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, len(names), 3) + metadata + directory
                     + b"\0" * (len(names) * 4 + 32))
    return names


class LauncherMtpTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.config = write_mtp_checkpoint(self.path)

    def test_switch_is_opt_in_and_preserves_model_defaults_in_every_mode(self):
        for mode in ("long_context", "high_concurrency", "custom"):
            for enabled in (False, True):
                with self.subTest(mode=mode, enabled=enabled):
                    result = recommend_launch_config(str(self.path), cuda_hardware(), config_mode=mode,
                                                     enable_speculative_decoding=enabled)
                    self.assertEqual(result["speculative"]["enabled"], enabled)
                    self.assertEqual(result["config"]["mtp"], "3" if enabled else "auto")
                    argv = build_fastllm_argv(config_from_dict({"model": str(self.path), **result["config"]}))
                    self.assertEqual("--mtp" in argv, enabled)
                    if enabled:
                        self.assertEqual(argv[argv.index("--mtp") + 1], "3")
                    for flag in ("--dtype", "--moe_dtype", "--chunked_prefill_size", "--max_context_length"):
                        self.assertNotIn(flag, argv)
        with patch("fastllm_pytools.launcher.detect_mtp_support", side_effect=AssertionError):
            recommend_launch_config(str(self.path), cuda_hardware())

    def test_missing_incomplete_and_truncated_weights_stay_off(self):
        weight = self.path / "model.safetensors"
        for names in (set(), {"mtp.fc.weight"}, MTP_NAMES - {"mtp.layers.0.self_attn.q_proj.weight"}):
            write_safetensors(weight, names)
            self.assertEqual(detect_mtp_support(self.path, self.config), "missing_weights")
        write_safetensors(weight, MTP_NAMES)
        weight.write_bytes(weight.read_bytes()[:-1])
        self.assertEqual(detect_mtp_support(self.path, self.config), "unverified")
        weight.unlink()
        self.assertEqual(detect_mtp_support(self.path, self.config), "missing_weights")

    def test_index_references_and_separate_mtp_file_match_the_native_loader(self):
        write_safetensors(self.path / "model.safetensors", {"model.embed_tokens.weight"})
        write_safetensors(self.path / "mtp.safetensors", MTP_NAMES)
        self.assertEqual(detect_mtp_support(self.path, self.config), "missing_weights")
        index = {"weight_map": {name: "mtp.safetensors" for name in MTP_NAMES}}
        index["weight_map"]["model.embed_tokens.weight"] = "model.safetensors"
        (self.path / "model.safetensors.index.json").write_text(json.dumps(index))
        self.assertEqual(detect_mtp_support(self.path, self.config), "enabled")
        (self.path / "mtp.safetensors").unlink()
        self.assertEqual(detect_mtp_support(self.path, self.config), "missing_weights")

    def test_mtp_config_alone_and_other_architectures_do_not_enable_mtp(self):
        self.config["text_config"]["mtp_num_hidden_layers"] = 0
        self.assertEqual(detect_mtp_support(self.path, self.config), "no_mtp")
        for model_type in ("qwen3", "llama", "glm5_next", "qwen3_8_flash_next", "deepseek_v4"):
            self.assertEqual(detect_mtp_support(self.path, {"model_type": model_type,
                             "mtp_num_hidden_layers": 1}), "unsupported_architecture")
        self.assertEqual(detect_mtp_support(self.path, {}), "unverified")
        self.assertEqual(detect_mtp_support(self.path, {"text_config": None}), "unverified")

    def test_cpu_and_rocm_configurations_keep_mtp_off(self):
        for hardware in ({**cuda_hardware(), "gpus": []},
                         {**cuda_hardware(), "build": {"USE_CUDA": False}},
                         {**cuda_hardware(), "build": {"USE_ROCM": True}}):
            result = recommend_launch_config(str(self.path), hardware, enable_speculative_decoding=True)
            self.assertEqual(result["speculative"]["reason"], "cuda_required")
            self.assertEqual(result["config"]["mtp"], "auto")

    def test_dense_and_moe_weight_layouts(self):
        names = {name for name in MTP_NAMES if ".mlp." not in name}
        self.config["text_config"].update(num_experts=2, num_experts_per_tok=1,
                                          shared_expert_intermediate_size=128)
        names.add("mtp.layers.0.mlp.gate.weight")
        for prefix in ("experts.0.", "experts.1.", "shared_expert."):
            names.update("mtp.layers.0.mlp." + prefix + suffix
                         for suffix in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))
        write_safetensors(self.path / "model.safetensors", names)
        self.assertEqual(detect_mtp_support(self.path, self.config), "enabled")
        names.remove("mtp.layers.0.mlp.experts.1.down_proj.weight")
        write_safetensors(self.path / "model.safetensors", names)
        self.assertEqual(detect_mtp_support(self.path, self.config), "missing_weights")

    def test_gguf_and_split_gguf_use_tensor_directories(self):
        path = self.path / "model.gguf"
        names = write_mtp_gguf(path)
        self.assertEqual(detect_mtp_support(path, {}), "enabled")
        write_mtp_gguf(path, names[:-1])
        self.assertEqual(detect_mtp_support(path, {}), "missing_weights")
        part1 = self.path / "model-00001-of-00002.gguf"
        part2 = self.path / "model-00002-of-00002.gguf"
        write_mtp_gguf(part1, names[:5])
        write_mtp_gguf(part2, names[5:])
        self.assertEqual(detect_mtp_support(part1, {}), "enabled")
        part2.unlink()
        self.assertEqual(detect_mtp_support(part1, {}), "missing_weights")
        path.write_bytes(b"GGUF\x03")
        self.assertEqual(detect_mtp_support(path, {}), "unverified")

    def test_api_and_saved_profile_preserve_the_preference(self):
        from fastapi.testclient import TestClient
        runtime = LauncherRuntime(str(self.path / "profiles.json"))
        self.addCleanup(runtime.close)
        with TestClient(create_launcher_app(runtime, "test-key")) as client, patch(
                "fastllm_pytools.launcher.detect_hardware", return_value=cuda_hardware()):
            response = client.post("/api/recommend", headers={"X-FTLLM-Launcher-Token": "test-key"},
                                   json={"model": str(self.path), "config_mode": "long_context",
                                         "enable_speculative_decoding": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["speculative"]["enabled"])
        saved = runtime.save_profile(None, {"model": str(self.path), "name": "MTP model",
                                           **response.json()["config"], "enable_speculative_decoding": True})
        self.assertTrue(runtime.profiles()[saved["index"]]["enable_speculative_decoding"])
        self.assertNotIn("--enable_speculative_decoding", runtime.preview(saved["profile"])["command"])
        self.assertFalse(config_from_dict({}).enable_speculative_decoding)


if __name__ == "__main__":
    unittest.main()
