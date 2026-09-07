"""Check metadata registration, relocation, and retry behavior."""

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from desktop.setup_desktop import ENTRIES, setup


class DesktopIconsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ftllm-icons-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "包 with spaces"
        shutil.copytree(Path(__file__).resolve().parents[1] / "icons", self.bundle / "support/icons")
        (self.bundle / "support/desktop").mkdir()
        for desktop, executable, name in ENTRIES:
            (self.bundle / desktop).write_text(f"[Desktop Entry]\nIcon={name}\n")
            (self.bundle / executable).touch()
        (self.bundle / "launch.sh").touch()
        (self.bundle / "ftllm").touch()
        environment = patch.dict(os.environ, {
            "XDG_DATA_HOME": str(self.root / "user data"),
            "XDG_CACHE_HOME": str(self.root / "cache"),
            "DBUS_SESSION_BUS_ADDRESS": "test-session",
        })
        environment.start()
        self.addCleanup(environment.stop)
        which = patch("desktop.setup_desktop.shutil.which", side_effect=lambda name: "/usr/bin/gio" if name == "gio" else None)
        which.start()
        self.addCleanup(which.stop)

    def test_icons_are_registered_for_scripts_and_desktop_files_and_refresh_after_move(self):
        originals = {desktop: (self.bundle / desktop).read_bytes() for desktop, _, _ in ENTRIES}
        with patch("desktop.setup_desktop.run_optional", return_value=True) as run:
            self.assertTrue(setup(self.bundle, quiet=True))
            self.assertEqual(run.call_count, 8)
            for desktop, executable, name in ENTRIES:
                icon = self.root / "user data/icons/hicolor/scalable/apps" / (name + ".svg")
                for filename in (desktop, executable):
                    run.assert_any_call([
                        "/usr/bin/gio", "set", "-t", "string", str(self.bundle / filename),
                        "metadata::custom-icon", icon.as_uri(),
                    ])
                self.assertEqual((self.bundle / desktop).read_bytes(), originals[desktop])
            cli_calls = [call.args[0] for call in run.call_args_list if call.args[0][4] == str(self.bundle / "ftllm")]
            self.assertEqual(cli_calls, [[
                "/usr/bin/gio", "set", "-t", "unset", str(self.bundle / "ftllm"),
                "metadata::custom-icon",
            ]])
            run.reset_mock()
            self.assertTrue(setup(self.bundle, quiet=True))
            run.assert_not_called()
            moved = self.root / "moved 包"
            self.bundle.rename(moved)
            self.assertTrue(setup(moved, quiet=True))
            self.assertEqual(run.call_count, 8)
            self.assertTrue(all(str(moved) in call.args[0][4] for call in run.call_args_list))

    def test_failed_metadata_is_retried(self):
        with patch("desktop.setup_desktop.run_optional", return_value=False):
            self.assertFalse(setup(self.bundle, quiet=True))
        with patch("desktop.setup_desktop.run_optional", return_value=True) as run:
            self.assertTrue(setup(self.bundle, quiet=True))
            self.assertEqual(run.call_count, 8)


if __name__ == "__main__":
    unittest.main()
