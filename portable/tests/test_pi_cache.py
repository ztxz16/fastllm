"""Pi downloads must remain pinned and work without network after caching."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


FETCH_PATH = Path(__file__).resolve().parents[2] / "tools/ftllm_agent_runtime/scripts/fetch_pi.py"
SPEC = importlib.util.spec_from_file_location("fetch_pi", FETCH_PATH)
fetch_pi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_pi)


class PiCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.archive = self.cache / f"pi-{fetch_pi.PI_VERSION}-linux-x64.tar.gz"
        self.data = b"verified-test-archive"
        self.digest = patch.object(fetch_pi, "ARCHIVE_SHA256", fetch_pi.sha256(self.data))
        self.digest.start()
        self.addCleanup(self.digest.stop)

    def test_online_download_populates_verified_offline_cache(self):
        with patch.object(fetch_pi, "fetch", return_value=self.data) as network:
            self.assertEqual(fetch_pi.load_archive(None, self.cache, False), self.data)
            network.assert_called_once_with(fetch_pi.ARCHIVE_URL)
        with patch.object(fetch_pi, "fetch", side_effect=AssertionError("network in offline mode")):
            self.assertEqual(fetch_pi.load_archive(None, self.cache, True), self.data)

    def test_missing_or_corrupt_offline_cache_never_downloads(self):
        with patch.object(fetch_pi, "fetch", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(RuntimeError, "Offline"):
                fetch_pi.load_archive(None, self.cache, True)
            self.archive.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "Offline"):
                fetch_pi.load_archive(None, self.cache, True)

    def test_bad_download_is_not_cached(self):
        with patch.object(fetch_pi, "fetch", return_value=b"wrong-release"):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                fetch_pi.load_archive(None, self.cache, False)
        self.assertFalse(self.archive.exists())

    def test_explicit_archive_is_verified_in_offline_mode(self):
        self.archive.write_bytes(self.data)
        with patch.object(fetch_pi, "fetch", side_effect=AssertionError("unexpected network")):
            self.assertEqual(fetch_pi.load_archive(self.archive, None, True), self.data)
            self.archive.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                fetch_pi.load_archive(self.archive, None, True)


if __name__ == "__main__":
    unittest.main()
