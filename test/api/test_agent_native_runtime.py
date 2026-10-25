import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastllm_pytools import agent_native_runtime as runtime


class NativeRuntimeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.node = self.root / "node"
        self.node.write_bytes(b"\x7fELFtest")

    def test_compatible_node_does_not_require_patchelf_or_change_files(self):
        with patch.object(runtime.platform, "system", return_value="Linux"), patch.object(
                runtime.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), patch.object(
                runtime.shutil, "which") as which:
            self.assertIsNone(runtime.prepare_node(self.node))
            which.assert_not_called()
        self.assertEqual(self.node.read_bytes(), b"\x7fELFtest")

    def test_old_host_without_private_runtime_reports_actionable_error(self):
        failure = subprocess.CompletedProcess([], 1, b"", b"GLIBC_2.28 not found")
        with patch.object(runtime.platform, "system", return_value="Linux"), patch.object(
                runtime.subprocess, "run", return_value=failure), patch.object(runtime.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "supported Linux.*patchelf"):
                runtime.prepare_node(self.node)
        self.assertEqual(self.node.read_bytes(), b"\x7fELFtest")

    def test_private_python_libraries_are_applied_only_to_node_and_verified(self):
        failure = subprocess.CompletedProcess([], 1, b"", b"GLIBC_2.28 not found")
        success = subprocess.CompletedProcess([], 0, b"", b"")
        with patch.object(runtime.platform, "system", return_value="Linux"), patch.object(
                runtime.subprocess, "run", side_effect=[failure, success, success, success]) as run, patch.object(
                runtime.shutil, "which", return_value="patchelf"), patch.object(
                runtime.subprocess, "check_output", side_effect=["/private/ld.so", str(self.root)]):
            runtime.prepare_node(self.node)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[1], ["/private/ld.so", "--library-path", str(self.root), str(self.node), "--version"])
        self.assertIn("--no-sort", commands[2])
        self.assertEqual(commands[2][-1], str(self.node))
        self.assertEqual(commands[3], [str(self.node), "-e", "process.exit(0)"])
        self.assertFalse(any("env" in call.kwargs for call in run.call_args_list))
