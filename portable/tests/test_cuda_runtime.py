import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "portable_cuda_runtime", Path(__file__).resolve().parents[1] / "cuda_runtime.py")
runtime_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime_check)


class CudaRuntimeFilesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "runtime"
        for module, names in runtime_check.CUDA_RUNTIME_FILES.values():
            for name in names:
                path = self.root / "nvidia" / module / "lib" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\x7fELF" + bytes(60))
        self.lookup = patch.object(runtime_check.importlib.metadata, "distribution",
                                   return_value=SimpleNamespace(locate_file=lambda name: self.root / name))
        self.lookup.start()
        self.addCleanup(self.lookup.stop)

    def test_complete_runtime(self):
        self.assertEqual(len(runtime_check.check_cuda_runtime(self.root)), 4)

    def test_metadata_does_not_hide_missing_nccl(self):
        (self.root / "nvidia/nccl/lib/libnccl.so.2").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing bundled CUDA library: libnccl"):
            runtime_check.check_cuda_runtime(self.root)

    def test_external_library_does_not_count_as_bundled(self):
        path = self.root / "nvidia/nccl/lib/libnccl.so.2"
        external = self.root.parent / "host-libnccl.so.2"
        path.rename(external)
        path.symlink_to(external)
        with self.assertRaisesRegex(RuntimeError, "Missing bundled CUDA library: libnccl"):
            runtime_check.check_cuda_runtime(self.root)


if __name__ == "__main__":
    unittest.main()
