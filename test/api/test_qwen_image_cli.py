"""CLI forwarding must preserve native options and package-local assets."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("qwen_cli", ROOT / "tools/fastllm_pytools/qwen_image.py")
qwen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen)


class QwenImageCLITest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.package = Path(self.directory.name) / "package with spaces"
        self.package.mkdir()
        for name in ("qwen_image_server", "qwen_image_generate"):
            binary = self.package / name
            binary.touch()
            binary.chmod(0o755)
        self.source = patch.object(qwen, "__file__", str(self.package / "qwen_image.py"))
        self.source.start()
        self.addCleanup(self.source.stop)

    def test_server_uses_bundled_assets_and_preserves_arguments(self):
        args = ["--model", "/models/a b", "--devices", "0,1", "--port", "8008"]
        with patch.object(qwen.os, "execv") as execute:
            qwen.main(["server", *args])
        binary = str(self.package / "qwen_image_server")
        execute.assert_called_once_with(binary, [binary, *args, "--demo",
            str(self.package / "qwen_image_assets/demo.html")])

    def test_explicit_demo_is_preserved(self):
        args = ["--model", "/models/qwen", "--demo", "/custom/demo.html"]
        with patch.object(qwen.os, "execv") as execute:
            qwen.main(["server", *args])
        binary = str(self.package / "qwen_image_server")
        execute.assert_called_once_with(binary, [binary, *args])

    def test_generate_forwards_native_help(self):
        with patch.object(qwen.os, "execv") as execute:
            qwen.main(["generate", "--help"])
        binary = str(self.package / "qwen_image_generate")
        execute.assert_called_once_with(binary, [binary, "--help"])

    def test_missing_backend_fails_before_exec(self):
        (self.package / "qwen_image_server").unlink()
        with patch.object(qwen.os, "execv") as execute, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as error:
                qwen.main(["server", "--model", "/models/qwen"])
        self.assertEqual(error.exception.code, 2)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
