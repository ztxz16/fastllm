import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("verify_wheel", REPO / "tools/windows/verify_wheel.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class WheelVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wheel = self.root / "ftllm-9.8.7-py3-none-win_amd64.whl"
        self.files = {
            "ftllm-9.8.7.dist-info/METADATA": b"Name: ftllm\nVersion: 9.8.7\n",
            "ftllm-9.8.7.dist-info/WHEEL": b"Tag: py3-none-win_amd64\n",
            "ftllm-9.8.7.dist-info/entry_points.txt": b"[console_scripts]\nftllm = ftllm.cli:main\n",
            "ftllm/build_info.json": b'{"USE_CUDA": false}',
        }
        for name in ("__init__.py", "cli.py", "llm.py", "launcher.py", "fastllm_tools.dll", "fastllm_triton_server.py",
                     "launcher_assets/index.html", "launcher_assets/locales/zh-CN.json",
                     "ui_plugins/studio/app.js", "ui_plugins/studio/template.html",
                     "ui_plugins/models/plugin.json", "plugin_assets/host.js"):
            self.files["ftllm/" + name] = b"test payload"

    def write_wheel(self):
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for name, content in self.files.items():
                archive.writestr(name, content)

    def test_version_comes_from_metadata_and_cpu_needs_no_cuda_tools(self):
        self.write_wheel()
        result = verifier.verify(self.wheel, version="9.8.7", backend="cpu")
        self.assertEqual(result["version"], "9.8.7")
        self.assertFalse(result["cuda_architectures_checked"])
        with self.assertRaisesRegex(ValueError, "Expected 1.0"):
            verifier.verify(self.wheel, version="1.0")
        with self.assertRaisesRegex(ValueError, "Expected cuda"):
            verifier.verify(self.wheel, backend="cuda")

    def test_missing_native_payload_and_wrong_platform_are_rejected(self):
        del self.files["ftllm/fastllm_tools.dll"]
        self.write_wheel()
        with self.assertRaisesRegex(ValueError, "Missing wheel member"):
            verifier.verify(self.wheel)
        self.files["ftllm-9.8.7.dist-info/WHEEL"] = b"Tag: py3-none-any\n"
        self.write_wheel()
        with self.assertRaisesRegex(ValueError, "Unexpected Windows wheel tags"):
            verifier.verify(self.wheel)

    def test_source_mismatch_is_rejected(self):
        self.write_wheel()
        source = self.root / "source/tools/fastllm_pytools"
        source.mkdir(parents=True)
        (source / "launcher.py").write_text("changed after building", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Source asset mismatch"):
            verifier.verify(self.wheel, source_root=self.root / "source")

    def test_incremental_wheel_cannot_keep_an_old_compiler_service(self):
        source_root = self.root / "source"
        (source_root / "tools/fastllm_pytools").mkdir(parents=True)
        service = source_root / "tools/fastllm_triton_server.py"
        service.write_bytes(b"updated compiler service")
        self.write_wheel()
        with self.assertRaisesRegex(ValueError, "Source asset mismatch: ftllm/fastllm_triton_server.py"):
            verifier.verify(self.wheel, source_root=source_root)
        self.files["ftllm/fastllm_triton_server.py"] = service.read_bytes()
        self.write_wheel()
        self.assertTrue(verifier.verify(self.wheel, source_root=source_root)["source_assets_checked"])

    def test_architecture_check_requires_requested_machine_code_and_ptx(self):
        verifier.check_architectures("60-real;90-real;120", ["60", "90", "120"], ["120f"])
        verifier.check_architectures("120-virtual", [], ["120f"])
        with self.assertRaisesRegex(ValueError, "machine code for SM86"):
            verifier.check_architectures("86-real", ["90"], [])
        with self.assertRaisesRegex(ValueError, "PTX for SM120"):
            verifier.check_architectures("120", ["120"], [])


class ArchiveVerificationTests(unittest.TestCase):
    def test_tampering_is_detected_even_with_python_optimization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "desktop.zip"
            files = {
                "FastLLM-Launcher.exe": b"executable", "resources/app/main.js": b"app",
                "ftllm/runtime/python.exe": b"python",
                "BUILD-INFO.json": json.dumps({"desktop": {"application_sha256": {
                    "main.js": hashlib.sha256(b"app").hexdigest()}}}).encode(),
            }
            manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in files.items())
            command = [sys.executable, "-I", "-B", "-O", str(REPO / "desktop/tests/verify_windows_archive.py"),
                       str(archive), "--report", str(root / "report.json")]

            def write_zip():
                with zipfile.ZipFile(archive, "w") as bundle:
                    for name, data in files.items():
                        bundle.writestr("FastLLM/" + name, data)
                    bundle.writestr("FastLLM/MANIFEST.sha256", manifest)
                archive.with_suffix(".zip.sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest())

            write_zip()
            good = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertTrue(json.loads((root / "report.json").read_text())["passed"])
            files["resources/app/main.js"] = b"tampered"
            write_zip()
            bad = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("File hash mismatch", bad.stderr)


if __name__ == "__main__":
    unittest.main()
