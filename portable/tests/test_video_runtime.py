import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "portable_video_runtime", Path(__file__).resolve().parents[1] / "video_runtime.py")
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


class VideoRuntimeTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        paths = ["runtime/ftllm/" + name for name in
                 ("minimax_h3_server", "minimax_h3_generate", "libfastllm_minimax_h3.so")]
        paths += ["media/bin/ffmpeg", "media/bin/ffprobe"]
        for name in paths:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x7fELF" + bytes(60))
            path.chmod(0o755)
        lookup = patch.object(video.importlib.metadata, "distribution",
            return_value=SimpleNamespace(locate_file=lambda name: self.root / "runtime" / name))
        lookup.start()
        self.addCleanup(lookup.stop)

    def test_complete_runtime(self):
        self.assertEqual(len(video.check_video_runtime(self.root)), 5)

    def test_missing_backend_fails(self):
        (self.root / "runtime/ftllm/minimax_h3_server").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing bundled video runtime: minimax_h3_server"):
            video.check_video_runtime(self.root)

    def test_external_media_tool_is_not_bundled(self):
        path = self.root / "media/bin/ffprobe"
        external = self.root / "outside-ffprobe"
        path.rename(external)
        path.symlink_to(external)
        with self.assertRaisesRegex(RuntimeError, "Missing bundled video runtime: ffprobe"):
            video.check_video_runtime(self.root)


if __name__ == "__main__":
    unittest.main()
