"""Public DiT dispatch tests. No native program or accelerator is required."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import cli, service_backends
from fastllm_pytools.model_service import identify_model
from fastllm_pytools.tui import build_fastllm_argv, config_from_dict, save_saved_configs


class ModelServiceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "renamed"
        self.model.mkdir()

    def test_detect_after_edit_and_reload(self):
        config = config_from_dict({"model": str(self.model), "backend": "duo"})
        self.assertEqual(config.backend, "llm")
        (self.model / "model_index.json").write_text('{"_class_name":"MiniMaxH3ImagePipeline"}')
        self.assertEqual(config.backend, "minimax_h3_image")
        path = self.root / "profiles.json"
        save_saved_configs([config], path)
        self.assertNotIn("backend", json.loads(path.read_text())["commands"][0])
        (self.model / "model_index.json").write_text('{"_class_name":"QwenImage21Pipeline"}')
        self.assertEqual(config.backend, "qwen_image")
        self.assertEqual(build_fastllm_argv(config)[:3], ["ftllm", "server", str(self.model)])
        from fastllm_pytools.launcher import DownloadConfig, _downloaded_model_path
        (self.model / "transformer").mkdir()
        (self.model / "transformer/weights.gguf").touch()
        self.assertEqual(_downloaded_model_path(DownloadConfig(target_dir=str(self.model))), str(self.model))

    def test_qwen_cli_and_launcher_remap_devices(self):
        (self.model / "model_index.json").write_text('{"_class_name":"QwenImage21Pipeline"}')
        runtime = self.root / "qwen_image_server"
        runtime.touch(mode=0o755)
        config = config_from_dict({"model": str(self.model), "visible_gpus": "1,0", "api_key": "k"})
        from fastllm_pytools.tui import build_fastllm_env
        with patch.object(service_backends, "PACKAGE", self.root), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5,2"}):
            public = build_fastllm_argv(config)
            with patch.dict(os.environ, build_fastllm_env(config)), patch("os.execvpe") as execute:
                cli.main(public[1:])
            exe, native, env = execute.call_args.args
            self.assertEqual(exe, str(runtime))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,5")
            self.assertEqual(native[native.index("--devices") + 1], "0,1")
            self.assertEqual(native[native.index("--api-key") + 1], "k")
            with patch("os.execvpe") as execute:
                cli.main(["serve", "--path", str(self.model), "--devices", "1,0", "--api-key", "k"])
                self.assertEqual(execute.call_args.args[1:], (native, env))
            for invalid in (["--parallel-mode", "sp"], ["--devices", "2"], ["--dtype", "fp16"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit), patch("os.execvpe") as execute:
                    cli.main(["server", str(self.model), *invalid])
                execute.assert_not_called()

    def test_lingbot_native_assets_and_single_device(self):
        for name in ("transformers/model.safetensors.index.json", "google/umt5-xxl/tokenizer.json",
                     "native/t5.safetensors", "native/vae.safetensors"):
            file = self.model / name
            file.parent.mkdir(exist_ok=True, parents=True)
            file.touch()
        (self.root / "libfastllm_lingbot.so").touch()
        package = self.root / "package"
        package.mkdir()
        (package / "lingbot_world_assets").symlink_to(service_backends.media_assets("lingbot_world"))
        with patch.object(service_backends, "PACKAGE", package), \
                patch.dict(os.environ, {"LD_LIBRARY_PATH": str(self.root), "CUDA_VISIBLE_DEVICES": "2,5"}):
            with patch("os.execvpe") as execute:
                cli.main(["server", str(self.model), "--devices", "1", "--host", "0.0.0.0", "--api-key", "k"])
                _, argv, env = execute.call_args.args
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "5")
                self.assertTrue(Path(argv[1]).is_file())
                self.assertEqual(argv[argv.index("--host") + 1], "0.0.0.0")
                self.assertEqual(argv[argv.index("--api-key") + 1], "k")
                self.assertEqual(argv[argv.index("--library") + 1], str(self.root / "libfastllm_lingbot.so"))
            for missing in (False, True):
                if missing:
                    (self.model / "native/t5.safetensors").unlink()
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit), patch("os.execvpe"):
                    cli.main(["server", str(self.model), "--devices", "0" if missing else "0,1"])

    def test_pika_adapter_directory_matches_native_discovery(self):
        adapter = self.model / "fl2va"
        adapter.mkdir()
        (adapter / "adapter.safetensors").touch()
        config = config_from_dict({"model": str(self.model)})
        self.assertEqual(config.backend, "minimax_h3")
        self.assertEqual(service_backends.get_backend(config).model_name(config), "pika-h3-flex")
        self.assertIn(str(self.model), identify_model(str(adapter)).error)

    def test_unknown_pipeline_and_components(self):
        (self.model / "model_index.json").write_text('{"_class_name":"UnregisteredPipeline"}')
        self.assertIn("UnregisteredPipeline", identify_model(str(self.model)).error)
        (self.model / "model_index.json").unlink()
        (self.model / "release.json").write_text('{"schema":"duo-release-v1"}')
        (self.model / "realization").mkdir()
        self.assertIn("distill-quant", identify_model(str(self.model)).error)
        self.assertIn(str(self.model), identify_model(str(self.model / "realization")).error)


if __name__ == "__main__":
    unittest.main()
