"""Exercise DLL discovery without importing llm.py's native library."""
import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[2] / "fastllm_pytools/llm.py"


class WindowsDllSearchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root / "ftllm"
        self.package.mkdir()
        self.environment = {}
        self.register = Mock(side_effect=lambda path: object())
        self.namespace = {
            "__file__": str(self.package / "llm.py"),
            "List": list,
            "os": SimpleNamespace(path=os.path, pathsep=os.pathsep,
                                  environ=self.environment, add_dll_directory=self.register),
            "sys": SimpleNamespace(prefix=str(self.root / "python"), path=[]),
            "site": SimpleNamespace(getsitepackages=lambda: [], getusersitepackages=lambda: ""),
            "importlib_metadata": SimpleNamespace(distribution=Mock(side_effect=LookupError)),
            "_windows_dll_directory_handles": [],
        }
        # Native imports happen before callers can use llm.py. Load only these
        # helpers so this regression runs on CPU CI without a built FastLLM DLL.
        names = {"_discover_windows_dll_dirs", "_prepare_windows_dll_search_path"}
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        helpers = ast.Module(body=[node for node in tree.body
                                   if isinstance(node, ast.FunctionDef) and node.name in names],
                             type_ignores=[])
        exec(compile(helpers, str(SOURCE), "exec"), self.namespace)

    def prepare(self):
        return self.namespace["_prepare_windows_dll_search_path"]()

    def test_path_only_dependency_is_registered_and_handle_retained(self):
        dependency = self.root / "CUDA 测试 bin"
        dependency.mkdir()
        self.environment["PATH"] = str(dependency)
        handles = {}

        def register(path):
            handles[path] = object()
            return handles[path]

        self.register.side_effect = register
        self.assertIn(str(dependency), self.prepare())
        self.assertIn(handles[str(dependency)], self.namespace["_windows_dll_directory_handles"])
        self.assertIn(str(dependency), self.environment["PATH"].split(os.pathsep))

    def test_path_ignores_empty_and_invalid_entries_and_deduplicates(self):
        dependency = self.root / "dependencies"
        dependency.mkdir()
        regular_file = self.root / "file.dll"
        regular_file.touch()
        self.environment["PATH"] = os.pathsep.join([
            "", str(dependency), str(dependency) + os.sep + ".", f'"{dependency}"',
            str(self.package), str(self.root / "missing"), str(regular_file), "",
        ])
        self.assertEqual(self.prepare(), [str(self.package), str(dependency)])
        self.assertEqual(self.register.call_count, 2)

    def test_unusable_path_entry_does_not_prevent_other_registrations(self):
        blocked = self.root / "blocked"
        available = self.root / "available"
        blocked.mkdir()
        available.mkdir()
        self.environment["PATH"] = os.pathsep.join([str(blocked), str(available)])
        handle = object()

        def register(path):
            if path == str(blocked):
                raise OSError("Directory registration failed")
            return handle

        self.register.side_effect = register
        self.prepare()
        self.register.assert_any_call(str(available))
        self.assertIn(handle, self.namespace["_windows_dll_directory_handles"])


if __name__ == "__main__":
    unittest.main()
