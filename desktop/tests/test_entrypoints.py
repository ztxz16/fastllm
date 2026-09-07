"""Exercise portable entrypoints through real shells and the desktop launcher."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

try:
    from gi.repository import Gio
except ImportError:
    Gio = None


DESKTOP = Path(__file__).resolve().parents[1]
ROOT = DESKTOP.parent


class EntrypointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="ftllm-native-launcher-")
        cls.addClassCleanup(temporary.cleanup)
        cls.native_launcher = Path(temporary.name) / "entrypoint"
        subprocess.run([
            os.environ.get("CC", "cc"), "-O2", "-Wall", "-Wextra", "-Werror",
            str(DESKTOP / "entrypoint.c"), "-o", str(cls.native_launcher),
        ], check=True, timeout=30)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ftllm-entrypoints-")
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        original = self.temporary / "original"
        payload = original / "support"
        (payload / "libexec").mkdir(parents=True)
        (payload / "runtime/bin").mkdir(parents=True)
        for name in ("ftllm", "ftllm-launch-webui", "Fastllm-Launcher"):
            self.install(self.native_launcher, original / name)
        self.install(DESKTOP / "launch.sh", original / "launch.sh")
        self.install(DESKTOP / "entrypoint.sh", payload / "entrypoint.sh")
        self.install(ROOT / "portable/launch.sh", payload / "launch.sh")
        self.install(DESKTOP / "launcher.sh", payload / "FastLLM-Launcher")
        self.install(DESKTOP / "terminal.sh", payload / "terminal.sh")
        self.install(DESKTOP / "setup_desktop.py", payload / "setup_desktop.py")
        shutil.copytree(DESKTOP / "icons", payload / "icons")
        (payload / "runtime/bin/python3").symlink_to(sys.executable)
        self.install(ROOT / "portable/activate.sh", payload / "libexec/activate.sh")
        (payload / "desktop").mkdir()
        for name in ("Fastllm-Launcher.desktop", "ftllm-launch-webui.desktop"):
            self.install(DESKTOP / name, payload / "desktop" / name)
        # Shell metacharacters in a moved bundle must remain literal data.
        self.bundle = self.temporary / "moved 中文 'quote' $dollar `tick` %k"
        original.rename(self.bundle)
        self.payload = self.bundle / "support"
        self.result = self.temporary / "result.json"
        self.environment = {
            **os.environ, "ENTRYPOINT_RESULT": str(self.result),
            "XDG_DATA_HOME": str(self.temporary / "user-data"),
            "XDG_CACHE_HOME": str(self.temporary / "user-cache"),
        }
        for name in ("ftllm", "FastLLM-Launcher.bin"):
            executable = self.payload / name
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['ENTRYPOINT_RESULT']).write_text(json.dumps({\n"
                "    'argv': sys.argv, 'cwd': os.getcwd(), 'env': dict(os.environ)\n"
                "}))\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)

    @staticmethod
    def install(source, destination):
        shutil.copyfile(source, destination)
        destination.chmod(0o755)

    def read_result(self):
        return json.loads(self.result.read_text(encoding="utf-8"))

    def test_relocated_wrappers_preserve_arguments_and_electron_paths(self):
        arguments = ["--config", "模型 dir/config.json", "$(false)", "", 'a"b']
        for entrypoint, prefix in (("ftllm", []), ("ftllm-launch-webui", ["launch"]), ("launch.sh", ["launch"]), ("Fastllm-Launcher", [])):
            with self.subTest(entrypoint=entrypoint):
                environment = self.environment.copy()
                environment.pop("FTLLM_RUNTIME_DIR", None)
                environment.pop("FTLLM_LAUNCHER_DATA_DIR", None)
                subprocess.run(
                    [str(self.bundle / entrypoint), *arguments],
                    env=environment, cwd=self.temporary, check=True, timeout=10,
                )
                result = self.read_result()
                self.assertEqual(result["argv"][1:], prefix + arguments)
                if entrypoint == "Fastllm-Launcher":
                    self.assertEqual(result["env"]["FTLLM_RUNTIME_DIR"], str(self.payload))
                    self.assertEqual(result["env"]["FTLLM_LAUNCHER_DATA_DIR"], str(self.payload / "data"))

    def test_terminal_keeps_an_interactive_bundled_environment(self):
        # A pipe lets the test send commands to the real interactive Bash without
        # a graphical terminal emulator. The no-job-control warning is expected.
        subprocess.run(
            [str(self.payload / "terminal.sh")],
            input="ftllm --help\nexit\n", text=True, capture_output=True,
            cwd=self.temporary, env=self.environment, check=True, timeout=10,
        )
        result = self.read_result()
        self.assertEqual(result["argv"], [str(self.payload / "ftllm"), "--help"])
        self.assertEqual(result["cwd"], str(self.bundle))
        self.assertEqual(result["env"]["FTLLM_HOME"], str(self.payload))
        self.assertTrue(result["env"]["PATH"].startswith(str(self.payload) + ":"))

    def test_setup_flag_installs_icons_without_forwarding_to_ftllm(self):
        commands = self.temporary / "setup-commands"
        commands.mkdir()
        for name in ("dirname", "basename"):
            (commands / name).symlink_to(shutil.which(name))
        original = (self.payload / "desktop/ftllm-launch-webui.desktop").read_bytes()
        result = subprocess.run(
            [str(self.bundle / "ftllm"), "--setup-desktop"],
            env={**self.environment, "PATH": str(commands)},
            capture_output=True, text=True, check=True, timeout=10,
        )
        self.assertIn("2 个 FastLLM 图标", result.stdout)
        self.assertFalse(self.result.exists())
        self.assertEqual((self.payload / "desktop/ftllm-launch-webui.desktop").read_bytes(), original)
        icons = self.temporary / "user-data/icons/hicolor/scalable/apps"
        self.assertEqual(len(list(icons.glob("*.svg"))), 2)

    def test_native_launcher_resolves_bundle_through_an_external_symlink(self):
        link = self.temporary / "symlink to ftllm"
        link.symlink_to(self.bundle / "ftllm")
        subprocess.run([str(link), "--help"], env=self.environment, check=True, timeout=10)
        self.assertEqual(self.read_result()["argv"], [str(self.payload / "ftllm"), "--help"])

    def test_graphical_launch_without_a_tty_opens_a_terminal(self):
        commands = self.temporary / "terminal-commands"
        commands.mkdir()
        for name in ("dirname", "basename", "bash"):
            (commands / name).symlink_to(shutil.which(name))
        terminal = commands / "x-terminal-emulator"
        terminal.write_text(
            '#!/bin/sh\nprintf "started" > "$TERMINAL_RESULT"\n'
            '[ "$1" != "-e" ] || shift\nexec "$@" < "$TERMINAL_INPUT"\n'
        )
        terminal.chmod(0o755)
        terminal_input = self.temporary / "terminal-input"
        terminal_input.write_text("ftllm --help\nexit\n")
        terminal_result = self.temporary / "terminal-result"
        environment = {
            **self.environment, "PATH": str(commands), "DISPLAY": ":test",
            "TERMINAL_INPUT": str(terminal_input), "TERMINAL_RESULT": str(terminal_result),
        }
        for name, arguments in (("ftllm", ["--help"]), ("ftllm-launch-webui", ["launch"])):
            with self.subTest(entrypoint=name):
                terminal_result.unlink(missing_ok=True)
                subprocess.run(
                    [str(self.bundle / name)], stdin=subprocess.DEVNULL,
                    capture_output=True, env=environment, check=True, timeout=10,
                )
                self.assertEqual(terminal_result.read_text(), "started")
                self.assertEqual(self.read_result()["argv"][1:], arguments)

    @unittest.skipIf(Gio is None, "PyGObject is needed to exercise desktop entries")
    def test_desktop_entries_launch_using_gio_after_relocation(self):
        # Substitute only the terminal emulator, so GIO still parses the actual
        # shipped Exec/Terminal fields and expands %k as a desktop would.
        commands = self.temporary / "commands"
        commands.mkdir()
        for name in ("sh", "bash", "dirname", "basename"):
            (commands / name).symlink_to(shutil.which(name))
        terminal = commands / "xterm"
        terminal.write_text(
            '#!/bin/sh\n[ "$1" != "-e" ] || shift\n'
            'exec "$@" < "$ENTRYPOINT_INPUT"\n'
        )
        terminal.chmod(0o755)
        input_path = self.temporary / "commands.txt"
        input_path.write_text("ftllm --help\nexit\n")
        environment = {
            **self.environment, "PATH": str(commands),
            "ENTRYPOINT_INPUT": str(input_path),
        }
        for entrypoint, expected in (
            ("Fastllm-Launcher.desktop", "FastLLM-Launcher.bin"),
            ("ftllm-launch-webui.desktop", "ftllm"),
        ):
            with self.subTest(entrypoint=entrypoint):
                self.result.unlink(missing_ok=True)
                # GIO may start a D-Bus daemon that inherits output handles;
                # use a file rather than waiting for pipe EOF from descendants.
                with (self.temporary / "gio.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable, "-c",
                            "import sys; from gi.repository import Gio; "
                            "Gio.DesktopAppInfo.new_from_filename(sys.argv[1]).launch([], None)",
                            str(self.payload / "desktop" / entrypoint),
                        ],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                        cwd=self.temporary, env=environment, check=True, timeout=10,
                    )
                deadline = time.monotonic() + 5
                while not self.result.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(self.result.exists(), (self.temporary / "gio.log").read_text())
                self.assertEqual(self.read_result()["argv"][0], str(self.payload / expected))
                if entrypoint == "ftllm-launch-webui.desktop":
                    self.assertEqual(self.read_result()["argv"][1:], ["launch"])


if __name__ == "__main__":
    unittest.main()
