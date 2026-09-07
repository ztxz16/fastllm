import asyncio
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.launcher import (
    ASSET_DIRECTORY,
    MODELSCOPE_PROGRESS_PREFIX,
    LauncherError,
    LauncherRuntime,
    _launcher_access_addresses,
    _parse_modelscope_download_progress,
    browse_folders,
    create_launcher_app,
    fastllm_launcher,
    recommend_launch_config,
)
from fastllm_pytools.tui import DeployConfig, build_fastllm_argv, config_from_dict


def unused_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class LauncherConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp.name, "model")
        self.draft_path = os.path.join(self.temp.name, "draft")
        os.makedirs(self.model_path)
        os.makedirs(self.draft_path)
        self.config_path = os.path.join(self.temp.name, "launcher.json")
        self.runtime = LauncherRuntime(self.config_path)

    def tearDown(self):
        self.runtime.close()
        self.temp.cleanup()

    def config(self, **changes):
        payload = {
            "name": "Qwen launcher",
            "command": "server",
            "model": self.model_path,
            "model_name": "qwen-local",
            "host": "127.0.0.1",
            "port": "18080",
            "device": "cpu",
            "gpu_mem_ratio": "0.9",
        }
        payload.update(changes)
        return payload

    def hardware(self, gpu_memory_gib=(), memory_gib=128, numa_nodes=2):
        gib = 1024 ** 3
        return {
            "cpu": {"logical": 64, "available": 32},
            "memory": {"total": memory_gib * gib, "available": memory_gib * gib},
            "gpus": [
                {
                    "index": index,
                    "name": f"Test GPU {index}",
                    "memoryTotalMiB": memory * 1024,
                    "memoryFreeMiB": memory * 1024,
                }
                for index, memory in enumerate(gpu_memory_gib)
            ],
            "numa": [{} for _ in range(numa_nodes)],
            "build": {"USE_CUDA": bool(gpu_memory_gib)},
        }

    def write_model(self, directory, config, weight_gib):
        os.makedirs(directory)
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as file:
            json.dump(config, file)
        weight_path = os.path.join(directory, "model.safetensors")
        with open(weight_path, "wb") as file:
            file.truncate(weight_gib * 1024 ** 3)

    def test_preview_reuses_tui_command_builder(self):
        preview = self.runtime.preview(self.config(
            speculative_algorithm="dflash",
            speculative_draft_model_path=self.draft_path,
            draft_tokens="7",
        ))

        self.assertEqual(preview["errors"], [])
        self.assertIn("ftllm server", preview["command"])
        self.assertIn("--speculative_algorithm dflash", preview["command"])
        self.assertIn("--draft_tokens 7", preview["command"])
        self.assertEqual(preview["endpoint"], "http://127.0.0.1:18080")

    def test_profile_round_trip_uses_tui_config_file(self):
        saved = self.runtime.save_profile(None, self.config())
        self.assertEqual(saved["index"], 0)
        self.assertEqual(self.runtime.profiles()[0]["model_name"], "qwen-local")

        with open(self.config_path, "r", encoding="utf-8") as config_file:
            on_disk = json.load(config_file)
        self.assertEqual(on_disk["version"], 1)
        self.assertEqual(on_disk["commands"][0]["command"], "server")

        result = self.runtime.delete_profile(0)
        self.assertEqual(result["profiles"], [])

    def test_configuration_mode_is_saved_without_changing_legacy_profiles(self):
        self.assertEqual(self.runtime.default_profile()["config_mode"], "long_context")
        self.assertTrue(self.runtime.default_profile()["low_gpu_mem"])
        for mode in ("long_context", "high_concurrency", "custom"):
            saved = self.runtime.save_profile(None, self.config(config_mode=mode))
            self.assertEqual(self.runtime.profiles()[saved["index"]]["config_mode"], mode)
            self.assertNotIn("--config_mode", self.runtime.preview(saved["profile"])["command"])
        legacy = config_from_dict({"max_batch": "3", "max_context_length": "12345"})
        self.assertEqual(legacy.config_mode, "custom")
        self.assertFalse(legacy.low_gpu_mem)
        self.assertEqual(legacy.max_batch, "3")
        self.assertEqual(legacy.max_context_length, "12345")

    def test_configuration_modes_preserve_model_weight_context_and_chunk_defaults(self):
        model_path = os.path.join(self.temp.name, "mode-test")
        self.write_model(model_path, {
            "architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 32768,
        }, 4)
        for mode, batch in (("long_context", "1"), ("high_concurrency", "auto"), ("custom", "4")):
            recommendation = recommend_launch_config(model_path, self.hardware((24,)), config_mode=mode)
            config = recommendation["config"]
            self.assertEqual(config["config_mode"], mode)
            self.assertEqual(config["max_batch"], batch)
            self.assertEqual(config["max_context_length"], "auto")
            self.assertEqual(config["low_gpu_mem"], mode == "long_context")
            for field in ("dtype", "moe_dtype", "chunked_prefill_size"):
                self.assertNotIn(field, config)
            argv = build_fastllm_argv(config_from_dict({"model": model_path, **config}))
            self.assertEqual("--low_gpu_mem" in argv, mode == "long_context")
            for option in ("--dtype", "--moe_dtype", "--chunked_prefill_size", "--max_context_length"):
                self.assertNotIn(option, argv)
        with self.assertRaisesRegex(LauncherError, "Unknown configuration mode"):
            recommend_launch_config(model_path, config_mode="invalid")

    def test_insufficient_gpu_memory_does_not_trigger_weight_conversion(self):
        model_path = os.path.join(self.temp.name, "Dense-20B")
        self.write_model(model_path, {
            "architectures": ["Qwen3ForCausalLM"], "torch_dtype": "float16",
        }, 40)
        for mode in ("long_context", "high_concurrency", "custom"):
            recommendation = recommend_launch_config(model_path, self.hardware((24,)), config_mode=mode)
            self.assertEqual(recommendation["strategy"], "numa")
            self.assertNotIn("dtype", recommendation["config"])

    def test_invalid_dflash_configuration_is_reported(self):
        preview = self.runtime.preview(self.config(
            speculative_algorithm="dflash",
            speculative_draft_model_path="",
        ))

        self.assertTrue(any("DFlash2" in error for error in preview["errors"]))

    def test_dflash_draft_path_must_be_a_directory(self):
        draft_file = os.path.join(self.temp.name, "draft.safetensors")
        with open(draft_file, "wb") as file:
            file.write(b"")

        preview = self.runtime.preview(self.config(
            speculative_algorithm="dflash",
            speculative_draft_model_path=draft_file,
        ))

        self.assertTrue(any("必须是目录" in error for error in preview["errors"]))

    def test_embedded_mtp_requires_positive_token_count(self):
        preview = self.runtime.preview(self.config(
            speculative_algorithm="mtp",
            speculative_draft_model_path="",
            mtp="0",
        ))

        self.assertTrue(any("内置 MTP" in error for error in preview["errors"]))

    def test_mtp_token_counts_must_match(self):
        preview = self.runtime.preview(self.config(
            speculative_algorithm="mtp",
            speculative_draft_model_path=self.draft_path,
            mtp="4",
            draft_tokens="5",
        ))

        self.assertTrue(any("必须一致" in error for error in preview["errors"]))

    def test_api_key_is_redacted_from_preview(self):
        preview = self.runtime.preview(self.config(api_key="local-secret"))

        self.assertNotIn("local-secret", preview["command"])
        self.assertIn("••••••••", preview["command"])

    def test_api_key_in_extra_arguments_is_redacted_from_preview(self):
        preview = self.runtime.preview(self.config(
            extra_args="--api_key=extra-secret",
        ))

        self.assertNotIn("extra-secret", preview["command"])
        self.assertIn("--api_key=••••••••", preview["command"])

    def test_ipv6_wildcard_uses_ipv6_loopback_for_display(self):
        preview = self.runtime.preview(self.config(host="::"))

        self.assertEqual(preview["endpoint"], "http://[::1]:18080")

    def test_wildcard_launcher_addresses_are_grouped_by_network_scope(self):
        addresses = _launcher_access_addresses(
            "0.0.0.0",
            8000,
            ["8.8.8.8", "10.20.30.40", "fc00::10"],
        )

        self.assertEqual(
            addresses,
            [
                {
                    "scope": "local",
                    "label": "Local address",
                    "url": "http://127.0.0.1:8000",
                },
                {
                    "scope": "lan",
                    "label": "LAN address",
                    "url": "http://10.20.30.40:8000",
                },
                {
                    "scope": "public",
                    "label": "Public address",
                    "url": "http://8.8.8.8:8000",
                },
            ],
        )

    def test_ipv6_wildcard_only_advertises_ipv6_addresses(self):
        addresses = _launcher_access_addresses(
            "::",
            8000,
            ["10.20.30.40", "fc00::10", "2001:4860:4860::8888"],
        )

        self.assertEqual(
            [address["scope"] for address in addresses],
            ["local", "lan", "public"],
        )
        self.assertEqual(addresses[0]["url"], "http://[::1]:8000")
        self.assertEqual(addresses[1]["url"], "http://[fc00::10]:8000")

    def test_json_scalar_types_are_normalized(self):
        preview = self.runtime.preview(self.config(
            enable_moe_hybrid="false",
            port=18080,
        ))

        self.assertNotIn("--moe_device", preview["command"])
        self.assertEqual(preview["endpoint"], "http://127.0.0.1:18080")

    def test_hybrid_device_map_and_ngram_storage_reach_ftllm(self):
        config = config_from_dict(self.config(
            enable_moe_hybrid=True,
            moe_device="custom",
            moe_device_custom="{'cuda':1,'numa':8,'disk':1}",
            moe_device_layers="-1",
            ngram_device="disk",
        ))

        argv = build_fastllm_argv(config)

        self.assertEqual(
            argv[argv.index("--moe_device") + 1],
            "{'cuda':1,'numa':8,'disk':1}",
        )
        self.assertEqual(argv[argv.index("--moe_device_layers") + 1], "-1")
        self.assertEqual(argv[argv.index("--ngram_device") + 1], "disk")

    def test_auto_device_omits_explicit_device_argument(self):
        config = config_from_dict(self.config(device="auto"))

        self.assertNotIn("--device", build_fastllm_argv(config))

    def test_invalid_custom_hybrid_device_map_is_rejected(self):
        preview = self.runtime.preview(self.config(
            enable_moe_hybrid=True,
            moe_device="custom",
            moe_device_custom="{'cuda': 0, 'numa': -1}",
            moe_device_layers="-1",
        ))

        self.assertTrue(any("设备映射格式无效" in error for error in preview["errors"]))

    def test_dense_model_recommendation_uses_available_gpus(self):
        model = os.path.join(self.temp.name, "Qwen3-7B")
        self.write_model(model, {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
        }, 14)

        recommendation = recommend_launch_config(
            model,
            self.hardware(gpu_memory_gib=(24, 24)),
        )

        self.assertEqual(recommendation["strategy"], "tensor_parallel")
        self.assertEqual(recommendation["config"]["device"], "tp")
        self.assertEqual(recommendation["config"]["tp"], "0,1")
        self.assertFalse(recommendation["config"]["enable_moe_hybrid"])
        self.assertEqual(recommendation["config"]["speculative_algorithm"], "auto")
        self.assertEqual(recommendation["config"]["mtp"], "auto")

    def test_large_moe_recommendation_enables_numa_hybrid_inference(self):
        model = os.path.join(self.temp.name, "DeepSeek-V3-100B")
        self.write_model(model, {
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "n_routed_experts": 64,
            "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"},
        }, 100)

        recommendation = recommend_launch_config(
            model,
            self.hardware(gpu_memory_gib=(24,), memory_gib=180),
        )

        self.assertEqual(recommendation["strategy"], "hybrid_numa")
        self.assertEqual(recommendation["config"]["device"], "cuda")
        self.assertTrue(recommendation["config"]["enable_moe_hybrid"])
        self.assertEqual(recommendation["config"]["moe_device"], "numa")
        self.assertEqual(recommendation["config"]["moe_device_layers"], "-1")

        low_memory = recommend_launch_config(
            model,
            self.hardware(gpu_memory_gib=(24,), memory_gib=64),
        )
        self.assertEqual(low_memory["strategy"], "hybrid_disk")
        self.assertEqual(low_memory["config"]["moe_device"], "disk")

    def test_qwen_ngram_table_moves_to_disk_under_memory_pressure(self):
        model = os.path.join(self.temp.name, "Qwen4-Exp-30B")
        self.write_model(model, {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "n_routed_experts": 64,
        }, 30)

        recommendation = recommend_launch_config(
            model,
            self.hardware(gpu_memory_gib=(24,), memory_gib=64),
        )

        self.assertTrue(recommendation["detected"]["usesNgram"])
        self.assertEqual(recommendation["config"]["ngram_device"], "disk")
        self.assertTrue(recommendation["adjustments"]["ngramOnDisk"])

    def test_webui_profile_builds_webui_command(self):
        preview = self.runtime.preview(self.config(
            command="webui",
            port="1616",
            webui_max_token="8192",
            webui_think="true",
        ))

        self.assertEqual(preview["errors"], [])
        self.assertIn("ftllm webui", preview["command"])
        self.assertIn("--max_token 8192", preview["command"])
        self.assertNotIn("--think", preview["command"])
        self.assertNotIn("--device", preview["command"])
        self.assertNotIn("--model_name", preview["command"])
        self.assertNotIn("--host", preview["command"])
        self.assertEqual(preview["endpoint"], "http://127.0.0.1:1616")

    def test_webui_saved_inference_options_produce_valid_api_client_args(self):
        from fastllm_pytools.cli import args_parser

        config = DeployConfig(**self.config(
            command="webui", port="1616", device="cuda",
            enable_moe_hybrid=True, moe_device="numa",
            webui_think="true", api_key="test-api-key",
            extra_args="--api_base http://127.0.0.1:8081/v1",
        ))
        args = args_parser().parse_args(build_fastllm_argv(config)[1:])
        self.assertEqual(args.api_base, "http://127.0.0.1:8081/v1")
        self.assertEqual(args.api_key, "test-api-key")
        self.assertEqual(args.port, 1616)
        self.assertFalse(hasattr(args, "moe_device"))

    def test_download_preview_hides_token(self):
        payload = {
            "modelId": "Qwen/Qwen3-0.6B",
            "targetDir": os.path.join(self.temp.name, "downloads", "qwen"),
            "revision": "master",
            "maxWorkers": "4",
            "token": "private-modelscope-token",
        }
        with patch(
            "fastllm_pytools.launcher.importlib.util.find_spec",
            return_value=object(),
        ):
            preview = self.runtime.preview_download(payload)

        self.assertEqual(preview["errors"], [])
        self.assertIn("modelscope download", preview["command"])
        self.assertNotIn("private-modelscope-token", preview["command"])
        self.assertIn("Token injected through environment", preview["command"])

    def test_download_progress_parser_rejects_invalid_events(self):
        valid = MODELSCOPE_PROGRESS_PREFIX + json.dumps({
            "version": 1,
            "type": "download.progress",
            "downloadedBytes": 650,
            "totalBytes": 1000,
            "completedFiles": 1,
            "totalFiles": 2,
        })
        self.assertEqual(
            _parse_modelscope_download_progress(valid)["downloadedBytes"],
            650,
        )
        self.assertIsNone(_parse_modelscope_download_progress("Progress 65%"))
        self.assertIsNone(_parse_modelscope_download_progress(
            MODELSCOPE_PROGRESS_PREFIX + '{"version": 2}'
        ))

    def test_occupied_server_port_is_rejected_before_spawn(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(LauncherError, "already in use"):
                self.runtime.start(self.config(port=str(port)))

    def test_launcher_assets_are_packaged_as_external_resources(self):
        for filename in ("index.html", "styles.css", "app.js", "launcher-icon.png"):
            self.assertTrue((ASSET_DIRECTORY / filename).is_file(), filename)
        html = (ASSET_DIRECTORY / "index.html").read_text(encoding="utf-8")
        javascript = (ASSET_DIRECTORY / "app.js").read_text(encoding="utf-8")
        self.assertIn('src="/assets/app.js"', html)
        self.assertIn('rel="icon" type="image/png" href="/assets/launcher-icon.png"', html)
        self.assertIn('class="brand-mark" src="/assets/launcher-icon.png"', html)
        self.assertIn('id="profile-browser"', html)
        self.assertIn('id="new-profile"', html)
        self.assertIn('id="close-profile-editor"', html)
        self.assertIn('id="auto-configure-profile"', html)
        self.assertIn('id="clear-profile-config"', html)
        self.assertIn('id="choose-model-folder"', html)
        self.assertIn('id="folder-picker-modal"', html)
        self.assertIn('id="folder-picker-list"', html)
        self.assertIn('id="folder-picker-select"', html)
        self.assertIn('id="confirmation-modal"', html)
        self.assertIn('id="confirmation-cancel"', html)
        self.assertIn('id="confirmation-confirm"', html)
        self.assertLess(
            html.index('data-field="model"'),
            html.index('data-field="name"'),
        )
        self.assertIn('/api/folders?', javascript)
        self.assertIn("function showConfirmation", javascript)
        self.assertNotIn("window.confirm", javascript)
        self.assertIn('data-field="enable_moe_hybrid"', html)
        self.assertIn('data-field="ngram_device"', html)
        self.assertIn('data-field="moe_device_custom"', html)
        self.assertIn(
            'id="profile-editor-modal" class="profile-editor-modal hidden" role="dialog"',
            html,
        )
        self.assertIn(
            'id="launch-form" class="configuration-card profile-editor"',
            html,
        )
        self.assertNotIn('id="start-runtime-top"', html)
        for action in ("start", "edit", "delete"):
            self.assertIn(f'profileActionButton("{action}"', javascript)
        self.assertNotIn("<script>", html)
        self.assertRegex(
            html,
            r'id="server-api-key-field" class="field">\s*<span>API Key</span>',
        )

        cached_ids_source = javascript.split("const ids = [", 1)[1].split(
            "];", 1
        )[0]
        cached_ids = set(re.findall(r'"([a-z0-9-]+)"', cached_ids_source))
        html_ids = set(re.findall(r'\bid="([^"]+)"', html))
        self.assertEqual(cached_ids - html_ids, set())

        form_fields = re.findall(r'\bdata-field="([^"]+)"', html)
        self.assertEqual(len(form_fields), len(set(form_fields)))
        self.assertEqual(
            set(form_fields) - set(DeployConfig.__dataclass_fields__),
            set(),
        )

        locale_directory = ASSET_DIRECTORY / "locales"
        resources = {}
        for locale in ("zh-CN", "en-US"):
            path = locale_directory / f"{locale}.json"
            self.assertTrue(path.is_file(), locale)
            resources[locale] = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(resources[locale]["locale"], locale)
            for pattern in resources[locale]["patterns"]:
                re.compile(pattern["source"])

        self.assertIn('id="language-select"', html)
        dynamic_messages = {
            json.loads(match)
            for match in re.findall(r'(?<![A-Za-z0-9_])t\(("(?:[^"\\]|\\.)*")', javascript)
        }
        self.assertEqual(
            dynamic_messages - set(resources["zh-CN"]["messages"]),
            set(),
        )
        static_chinese = {
            value.strip()
            for value in re.findall(r">([^<>]+)<", html)
            if re.search(r"[一-龥]", value)
        }
        static_chinese.update(
            re.findall(
                r'(?:aria-label|placeholder|title)="([^\"]*[一-龥][^\"]*)"',
                html,
            )
        )
        self.assertEqual(
            static_chinese - set(resources["zh-CN"]["static"]),
            set(),
        )
        self.assertEqual(
            static_chinese - set(resources["en-US"]["static"]),
            set(),
        )
        english_values = [
            value
            for key, value in resources["en-US"]["static"].items()
            if key != "简体中文"
        ] + list(resources["en-US"]["messages"].values()) + [
            pattern["target"] for pattern in resources["en-US"]["patterns"]
        ]
        self.assertNotRegex("\n".join(english_values), r"[一-龥]")

    def test_folder_browser_lists_folders_and_files_and_resolves_file_path(self):
        browser_root = os.path.join(self.temp.name, "browser")
        os.makedirs(os.path.join(browser_root, "zeta"))
        os.makedirs(os.path.join(browser_root, "Alpha"))
        model_file = os.path.join(browser_root, "model.gguf")
        with open(model_file, "wb") as file:
            file.write(b"gguf")

        listing = browse_folders(browser_root)
        self.assertEqual(listing["path"], browser_root)
        self.assertEqual(
            [folder["name"] for folder in listing["folders"]],
            ["Alpha", "zeta"],
        )
        self.assertFalse(listing["truncated"])
        self.assertNotIn("model.gguf", [folder["name"] for folder in listing["folders"]])
        self.assertEqual(listing["files"], [{"name": "model.gguf", "path": model_file}])

        file_listing = browse_folders(model_file)
        self.assertEqual(file_listing["path"], browser_root)
        self.assertEqual(file_listing["selectedFile"], model_file)

    def test_file_browser_bounds_combined_directory_and_file_entries(self):
        for name in ("a.gguf", "b.flm", "c.bin"):
            with open(os.path.join(self.model_path, name), "w"):
                pass
        os.mkdir(os.path.join(self.model_path, "folder"))
        with patch("fastllm_pytools.launcher.FOLDER_BROWSER_ENTRY_LIMIT", 2):
            listing = browse_folders(self.model_path)
        self.assertEqual(len(listing["folders"]) + len(listing["files"]), 2)
        self.assertTrue(listing["truncated"])

    def test_folder_browser_api_returns_server_directories(self):
        from fastapi.testclient import TestClient

        browser_root = os.path.join(self.temp.name, "browser-api")
        child = os.path.join(browser_root, "model")
        os.makedirs(child)
        client = TestClient(create_launcher_app(self.runtime, "control-secret"))
        drives = [{"name": "D:", "path": "D:\\"}]
        with patch("fastllm_pytools.launcher._folder_browser_drives", return_value=drives):
            response = client.get(
                "/api/folders",
                params={"path": browser_root},
                headers={"X-FTLLM-Launcher-Token": "control-secret"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["path"], browser_root)
        self.assertEqual(response.json()["drives"], drives)
        self.assertEqual(response.json()["folders"], [{
            "name": "model",
            "path": child,
        }])

    def test_windows_folder_browser_lists_drives_and_selects_files_on_another_volume(self):
        import ctypes
        import ntpath

        folder = "D:\\模型 目录"
        model_file = folder + "\\model.gguf"
        entry = SimpleNamespace(name="model.gguf", path=model_file,
                                is_dir=lambda **kwargs: False, is_file=lambda **kwargs: True)
        windows_os = SimpleNamespace(name="nt", path=ntpath,
                                     scandir=lambda path: nullcontext(iter([entry] if path == folder else [])))
        kernel32 = SimpleNamespace(GetLogicalDrives=lambda: (1 << 2) | (1 << 3) | (1 << 25))
        with patch("fastllm_pytools.launcher.os", windows_os), \
                patch("ctypes.windll", SimpleNamespace(kernel32=kernel32), create=True), \
                patch("ntpath.isdir", side_effect=lambda path: path in ("C:\\", "D:\\", folder)), \
                patch("ntpath.isfile", side_effect=lambda path: path == model_file):
            listing = browse_folders(model_file)
            self.assertEqual(listing["path"], folder)
            self.assertEqual(listing["parent"], "D:\\")
            self.assertEqual(listing["selectedFile"], model_file)
            self.assertEqual(listing["files"], [{"name": "model.gguf", "path": model_file}])
            self.assertEqual(listing["drives"], [
                {"name": "C:", "path": "C:\\"},
                {"name": "D:", "path": "D:\\"},
                {"name": "Z:", "path": "Z:\\"},
            ])
            self.assertEqual(browse_folders("D:\\")["parent"], "")
            with self.assertRaisesRegex(LauncherError, "Folder root is unavailable"):
                browse_folders("Z:\\")

    def test_control_api_requires_launcher_token(self):
        from fastapi.testclient import TestClient

        addresses = [{
            "scope": "local",
            "label": "Local address",
            "url": "http://127.0.0.1:8000",
        }]
        client = TestClient(create_launcher_app(
            self.runtime,
            "control-secret",
            addresses,
        ))
        denied = client.get("/api/bootstrap")
        allowed = client.get(
            "/api/bootstrap",
            headers={"X-FTLLM-Launcher-Token": "control-secret"},
        )

        self.assertEqual(denied.status_code, 403)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.json()["launcherAddresses"], addresses)
        self.assertEqual(allowed.headers["cache-control"], "no-store")

    def test_automatic_configuration_api_uses_model_and_hardware(self):
        from fastapi.testclient import TestClient

        with open(os.path.join(self.model_path, "config.json"), "w", encoding="utf-8") as file:
            json.dump({
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
            }, file)
        client = TestClient(create_launcher_app(self.runtime, "control-secret"))
        with patch(
            "fastllm_pytools.launcher.detect_hardware",
            return_value=self.hardware(gpu_memory_gib=(24,)),
        ):
            response = client.post(
                "/api/recommend",
                json={"model": self.model_path, "name": "Qwen3"},
                headers={"X-FTLLM-Launcher-Token": "control-secret"},
            )

        self.assertEqual(response.status_code, 200)
        recommendation = response.json()
        self.assertEqual(recommendation["config"]["device"], "cuda")
        self.assertEqual(recommendation["config"]["cuda_device_id"], "0")
        self.assertEqual(recommendation["config"]["speculative_algorithm"], "auto")

    def test_control_api_rejects_an_empty_server_token(self):
        with self.assertRaisesRegex(LauncherError, "token must not be empty"):
            create_launcher_app(self.runtime, "")

    def test_control_api_rejects_invalid_json(self):
        from fastapi.testclient import TestClient

        client = TestClient(create_launcher_app(self.runtime, "control-secret"))
        response = client.post(
            "/api/preview",
            content="{",
            headers={"X-FTLLM-Launcher-Token": "control-secret"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON", response.json()["error"])


class LauncherProcessTest(unittest.TestCase):
    def test_context_window_uses_reported_session_limit_and_ignores_stale_logs(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as directory:
            runtime = LauncherRuntime(os.path.join(directory, "config.json"))
            self.addCleanup(runtime.close)
            self.assertIsNone(runtime.state()["contextWindowTokens"])
            runtime._state.update(phase="starting")
            for limit, capacity, configured in ((262144, 335360, "None"),
                                                (167168, 167168, "None"),
                                                (8192, 335360, "8192")):
                line = (f"Model context window: {limit} tokens per session "
                        f"(model=262144, shared KV cache={capacity}, configured limit={configured})")
                runtime._consume_stream(io.StringIO(
                    "\x1b[32m2026-09-06 16:14:10,407 447441 server.py[line:326] INFO: "
                    + line + "\x1b[0m\n"), "stderr", runtime._generation)
                self.assertEqual(runtime.state()["contextWindowTokens"], limit)
            runtime.clear_logs()
            client = TestClient(create_launcher_app(runtime, "test-key"))
            response = client.get("/api/runtime", headers={"X-FTLLM-Launcher-Token": "test-key"})
            self.assertEqual(response.json()["contextWindowTokens"], 8192)
            for invalid in ("[Fastllm] KV Cache Token limit: 335360 tokens.",
                            line.replace("8192 tokens", "0 tokens"),
                            line.replace("8192 tokens", "-1 tokens"),
                            line.replace("8192 tokens", "NaN tokens")):
                runtime._handle_context_window(invalid, runtime._generation)
                self.assertEqual(runtime.state()["contextWindowTokens"], 8192)
            runtime._state["contextWindowTokens"] = None
            runtime._handle_context_window(line, runtime._generation - 1)
            self.assertIsNone(runtime.state()["contextWindowTokens"])
            for phase in ("stopped", "stopping", "failed"):
                runtime._state.update(phase=phase)
                runtime._handle_context_window(line, runtime._generation)
                self.assertIsNone(runtime.state()["contextWindowTokens"])
            runtime._state.update(phase="running", command="webui")
            runtime._handle_context_window(line, runtime._generation)
            self.assertIsNone(runtime.state()["contextWindowTokens"])
            runtime._state.update(command="server")
            runtime._handle_context_window(line, runtime._generation)
            self.assertEqual(runtime.state()["contextWindowTokens"], 8192)
            runtime.stop()
            self.assertIsNone(runtime.state()["contextWindowTokens"])

    def test_speed_samples_are_read_from_engine_logs_and_exposed_by_api(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as directory:
            runtime = LauncherRuntime(os.path.join(directory, "config.json"))
            self.addCleanup(runtime.close)
            runtime._state.update(phase="running", ready=True)
            prompt = "[Prompt] 1024 Tokens. Speed: 2345.678900 tokens / s."
            decode = "[Decode] alive = 2, pending = 1, context usages: 12.5%, Speed: 98.765432 tokens / s."
            with patch("fastllm_pytools.launcher.time.time", return_value=1000.0):
                runtime._consume_stream(io.StringIO("\x1b[32m" + prompt + "\x1b[0m\n" + decode + "\n"),
                                        "stdout", runtime._generation)
            speed = runtime.state()["speed"]
            self.assertEqual(speed, {"prefillTokensPerSecond": 2345.6789, "decodeTokensPerSecond": 98.765432,
                                     "prefillUpdatedAt": 1000.0, "decodeUpdatedAt": 1000.0})
            runtime.clear_logs()
            self.assertEqual(runtime.state()["speed"], speed)
            client = TestClient(create_launcher_app(runtime, "test-key"))
            response = client.get("/api/runtime", headers={"X-FTLLM-Launcher-Token": "test-key"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["speed"], speed)
            speed["decodeTokensPerSecond"] = 0
            self.assertEqual(runtime.state()["speed"]["decodeTokensPerSecond"], 98.765432)
            with patch("fastllm_pytools.launcher.time.time", return_value=1002.0):
                runtime._handle_inference_speed(
                    "[Prompt] Long Prefill ... (2048/8192, 25%). Speed: 4.5e3 tokens / s.", runtime._generation)
                runtime._handle_inference_speed(
                    "[Decode] alive = 1, pending = 0, contextLen = 8192, Speed: 123.0 tokens / s.", runtime._generation)
            self.assertEqual(runtime.state()["speed"]["prefillTokensPerSecond"], 4500.0)
            self.assertEqual(runtime.state()["speed"]["decodeTokensPerSecond"], 123.0)
            self.assertEqual(runtime.state()["speed"]["decodeUpdatedAt"], 1002.0)

    def test_speed_ignores_invalid_and_previous_service_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LauncherRuntime(os.path.join(directory, "config.json"))
            self.addCleanup(runtime.close)
            empty = runtime.state()["speed"]
            runtime._state.update(phase="running")
            for line in ("user output Speed: 42 tokens / s.", "[Prompt] Long Prefill ... (10%)",
                         *[f"[Decode] alive = 1, Speed: {value} tokens / s."
                           for value in ("nan", "inf", "-1", "1e9999")]):
                runtime._handle_inference_speed(line, runtime._generation)
            self.assertEqual(runtime.state()["speed"], empty)
            valid = "[Decode] alive = 1, pending = 0, contextLen = 128, Speed: 42 tokens / s."
            runtime._handle_inference_speed(valid, runtime._generation - 1)
            self.assertEqual(runtime.state()["speed"], empty)
            for phase in ("stopped", "failed", "stopping"):
                runtime._state.update(phase=phase)
                runtime._handle_inference_speed(valid, runtime._generation)
                self.assertEqual(runtime.state()["speed"], empty)
            runtime._state.update(phase="running", command="webui")
            runtime._handle_inference_speed(valid, runtime._generation)
            self.assertEqual(runtime.state()["speed"], empty)
            runtime._state.update(command="server")
            runtime._handle_inference_speed(valid, runtime._generation)
            self.assertEqual(runtime.state()["speed"]["decodeTokensPerSecond"], 42)
            runtime.stop()
            self.assertEqual(runtime.state()["speed"], empty)

    def test_launcher_opens_browser_after_startup_by_default(self):
        import uvicorn

        class InlineThread:
            def __init__(self, target, args=(), **_kwargs):
                self.target = target
                self.args = args

            def start(self):
                self.target(*self.args)

        async def fake_startup(server, sockets=None):
            server.started = True

        def fake_run(server):
            asyncio.run(server.startup())

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                host="127.0.0.1",
                port=8000,
                no_browser=False,
                config=os.path.join(directory, "launcher.json"),
            )
            terminal_output = io.StringIO()
            with (
                patch.object(uvicorn.Server, "startup", fake_startup),
                patch.object(uvicorn.Server, "run", fake_run),
                patch(
                    "fastllm_pytools.launcher.threading.Thread",
                    InlineThread,
                ),
                patch(
                    "fastllm_pytools.launcher.webbrowser.open",
                    return_value=True,
                ) as browser_open,
                redirect_stdout(terminal_output),
                redirect_stderr(terminal_output),
            ):
                result = fastllm_launcher(args)

        self.assertEqual(result, 0)
        browser_open.assert_called_once()
        self.assertRegex(
            browser_open.call_args.args[0],
            r"^http://127\.0\.0\.1:8000/\?token=.+$",
        )
        self.assertIn("Local address", terminal_output.getvalue())
        self.assertNotRegex(terminal_output.getvalue(), r"[一-龥]")

    def test_concurrent_start_only_spawns_one_process(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model")
            os.makedirs(model_path)
            entered_popen = threading.Event()
            release_popen = threading.Event()
            spawn_count = 0

            def delayed_popen(*args, **kwargs):
                nonlocal spawn_count
                spawn_count += 1
                entered_popen.set()
                release_popen.wait(timeout=2)
                return subprocess.Popen(*args, **kwargs)

            runtime = LauncherRuntime(
                os.path.join(directory, "config.json"),
                popen_factory=delayed_popen,
            )
            fake_argv = [
                sys.executable, "-u", "-c", "import time;time.sleep(30)"
            ]
            payload = {
                "name": "concurrent start",
                "command": "server",
                "model": model_path,
                "host": "127.0.0.1",
                "port": str(unused_tcp_port()),
                "device": "cpu",
                "gpu_mem_ratio": "0.9",
            }
            outcomes = []

            def start_runtime():
                try:
                    outcomes.append(runtime.start(payload)["phase"])
                except LauncherError as error:
                    outcomes.append(str(error))

            try:
                with patch(
                    "fastllm_pytools.launcher.build_fastllm_argv",
                    return_value=fake_argv,
                ):
                    first = threading.Thread(target=start_runtime)
                    second = threading.Thread(target=start_runtime)
                    first.start()
                    self.assertTrue(entered_popen.wait(timeout=2))
                    second.start()
                    release_popen.set()
                    first.join(timeout=3)
                    second.join(timeout=3)

                self.assertEqual(spawn_count, 1)
                self.assertEqual(len(outcomes), 2)
                self.assertTrue(any("already running" in item for item in outcomes))
            finally:
                release_popen.set()
                runtime.close()

    def test_progress_event_and_process_tree_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model")
            os.makedirs(model_path)
            runtime = LauncherRuntime(os.path.join(directory, "config.json"))
            event = {
                "type": "startup.ready",
                "stage": "ready",
                "percent": 100,
                "message": "API server is ready",
            }
            runtime._state["speed"]["decodeTokensPerSecond"] = 99
            runtime._state["contextWindowTokens"] = 262144
            def checked_popen(*args, **kwargs):
                self.assertTrue(all(value is None for value in runtime.state()["speed"].values()))
                self.assertIsNone(runtime.state()["contextWindowTokens"])
                return subprocess.Popen(*args, **kwargs)
            runtime._popen = checked_popen
            script = (
                "import json,sys,time;"
                f"print('FTLLM_PROGRESS '+json.dumps({event!r}),file=sys.stderr,flush=True);"
                "print('[Decode] alive = 1, pending = 0, contextLen = 128, Speed: 42.0 tokens / s.',flush=True);"
                "time.sleep(30)"
            )
            fake_argv = [sys.executable, "-u", "-c", script]
            payload = {
                "name": "process test",
                "command": "server",
                "model": model_path,
                "host": "127.0.0.1",
                "port": str(unused_tcp_port()),
                "device": "cpu",
                "gpu_mem_ratio": "0.9",
            }
            try:
                with patch(
                    "fastllm_pytools.launcher.build_fastllm_argv",
                    return_value=fake_argv,
                ):
                    started = runtime.start(payload)
                self.assertEqual(started["phase"], "starting")
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if runtime.state()["phase"] == "running" and runtime.state()["speed"]["decodeTokensPerSecond"] == 42:
                        break
                    time.sleep(0.02)
                self.assertEqual(runtime.state()["phase"], "running")
                self.assertTrue(runtime.state()["ready"])
                self.assertEqual(runtime.state()["speed"]["decodeTokensPerSecond"], 42)

                runtime.stop()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and runtime.state()["pid"]:
                    time.sleep(0.02)
                self.assertEqual(runtime.state()["phase"], "stopped")
                self.assertIsNone(runtime.state()["pid"])
            finally:
                runtime.close()

    def test_webui_start_does_not_add_server_progress_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model")
            os.makedirs(model_path)
            runtime = LauncherRuntime(os.path.join(directory, "config.json"))
            fake_argv = [sys.executable, "-u", "-c", "import time;time.sleep(30)"]
            payload = {
                "name": "webui process test",
                "command": "webui",
                "model": model_path,
                "port": str(unused_tcp_port()),
                "device": "cpu",
                "gpu_mem_ratio": "0.9",
            }
            try:
                with patch(
                    "fastllm_pytools.launcher.build_fastllm_argv",
                    return_value=fake_argv,
                ):
                    started = runtime.start(payload)
                self.assertEqual(started["command"], "webui")
                self.assertNotIn("--startup-progress", fake_argv)
            finally:
                runtime.close()

    def test_download_progress_and_process_tree_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LauncherRuntime(os.path.join(directory, "config.json"))
            event = {
                "version": 1,
                "type": "download.progress",
                "downloadedBytes": 650,
                "totalBytes": 1000,
                "completedFiles": 1,
                "totalFiles": 2,
            }
            script = (
                "import json,time;"
                f"print({MODELSCOPE_PROGRESS_PREFIX!r}+json.dumps({event!r}),flush=True);"
                "time.sleep(30)"
            )
            fake_argv = [sys.executable, "-u", "-c", script]
            payload = {
                "modelId": "Qwen/Qwen3-0.6B",
                "targetDir": os.path.join(directory, "downloaded-model"),
                "revision": "master",
                "maxWorkers": 4,
                "token": "memory-only-token",
            }
            try:
                with (
                    patch(
                        "fastllm_pytools.launcher.importlib.util.find_spec",
                        return_value=object(),
                    ),
                    patch(
                        "fastllm_pytools.launcher._download_argv",
                        return_value=fake_argv,
                    ),
                ):
                    runtime.start_download(payload)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if runtime.download_state()["progress"] >= 65:
                        break
                    time.sleep(0.02)
                self.assertEqual(runtime.download_state()["progress"], 65.0)
                self.assertNotIn("memory-only-token", json.dumps(runtime.logs()))

                runtime.stop_download()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if runtime.download_state()["phase"] == "cancelled":
                        break
                    time.sleep(0.02)
                self.assertEqual(runtime.download_state()["phase"], "cancelled")
                self.assertIsNone(runtime.download_state()["pid"])
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
