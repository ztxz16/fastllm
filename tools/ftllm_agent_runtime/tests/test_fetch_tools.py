import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_pi
import fetch_tools


class SearchToolPackagingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()

    def cached_tool(self, name, info):
        filename = info["url"].rsplit("/", 1)[-1]
        prefix = filename.removesuffix(".tar.gz")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for member_name in (name, *info["licenses"]):
                payload = member_name.encode()
                member = tarfile.TarInfo(prefix + "/" + member_name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        data = buffer.getvalue()
        (self.cache / filename).write_bytes(data)
        return dict(info, sha256=fetch_pi.sha256(data))

    def test_verified_offline_archives_include_executables_and_all_licenses(self):
        definitions = {name: self.cached_tool(name, info) for name, info in fetch_tools.TOOLS.items()}
        binary_dir, licenses = self.root / "bin", self.root / "licenses"
        with patch.object(fetch_tools, "TOOLS", definitions), \
                patch.object(fetch_pi, "fetch", side_effect=AssertionError("unexpected network")):
            fetch_tools.install_tools(binary_dir, licenses, self.cache, True)
        for name, info in definitions.items():
            self.assertEqual((binary_dir / name).read_bytes(), name.encode())
            self.assertTrue((binary_dir / name).stat().st_mode & 0o111)
            for license_name in info["licenses"]:
                self.assertEqual((licenses / name / license_name).read_text(), license_name)
        self.assertEqual(json.loads((licenses / "manifest.json").read_text()), definitions)

    def test_missing_offline_cache_never_downloads_or_installs(self):
        with patch.object(fetch_pi, "fetch", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(RuntimeError, "Offline"):
                fetch_tools.install_tools(self.root / "bin", self.root / "licenses", self.cache, True)
        self.assertFalse((self.root / "bin").exists())

    def test_bad_download_is_rejected_before_installation(self):
        with patch.object(fetch_pi, "fetch", return_value=b"wrong archive"):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                fetch_tools.install_tools(self.root / "bin", self.root / "licenses", self.cache, False)
        self.assertFalse((self.root / "bin").exists())
        self.assertEqual(list(self.cache.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
