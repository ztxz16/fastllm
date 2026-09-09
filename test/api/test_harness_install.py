import hashlib
import io
import os
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastllm_pytools import harness_install as installer


class HarnessInstallTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cancelled = threading.Event()
        self.events = []
        self.progress = lambda *args: self.events.append(args)

    def test_download_checks_checksum_and_reports_actual_bytes(self):
        payload = b"node archive" * 20000
        checksum = hashlib.sha256(payload).hexdigest()
        for digest, succeeds in ((checksum, True), ("0" * 64, False)):
            response = io.BytesIO(payload)
            response.headers = {"Content-Length": str(len(payload))}
            with patch.object(installer, "urlopen", return_value=response), patch.object(
                    installer, "_node_archive", return_value=("node.tar.gz", digest)):
                if succeeds:
                    installer._download_node(self.root / "node", self.progress, self.cancelled)
                    self.assertEqual(self.events[-1], ("download", len(payload), len(payload)))
                else:
                    with self.assertRaisesRegex(RuntimeError, "checksum"):
                        installer._download_node(self.root / "node", self.progress, self.cancelled)

    def test_archives_copy_runtime_files_without_following_links_or_traversal(self):
        archive = self.root / "node.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for name in ("node/bin/node", "node/lib/node_modules/npm/bin/npm-cli.js"):
                member = tarfile.TarInfo(name)
                member.size = 4; member.mode = 0o755
                bundle.addfile(member, io.BytesIO(b"node"))
            link = tarfile.TarInfo("node/lib/node_modules/npm/escape")
            link.type = tarfile.SYMTYPE; link.linkname = "/tmp"
            bundle.addfile(link)
        target = self.root / "extracted"
        installer._extract_node(archive, target, self.cancelled)
        self.assertTrue(os.access(target / "bin/node", os.X_OK))
        self.assertTrue((target / "bin/npm").is_file())
        self.assertFalse((target / "lib/node_modules/npm/escape").exists())
        with zipfile.ZipFile(self.root / "bad.zip", "w") as bundle:
            bundle.writestr("node/../../escaped", "bad")
        with self.assertRaisesRegex(RuntimeError, "archive path"):
            installer._extract_node(self.root / "bad.zip", self.root / "bad", self.cancelled)
        self.assertFalse((self.root / "escaped").exists())
        with zipfile.ZipFile(self.root / "windows.zip", "w") as bundle:
            bundle.writestr("node/node.exe", "node")
            bundle.writestr("node/node_modules/npm/bin/npm-cli.js", "npm")
        installer._extract_node(self.root / "windows.zip", self.root / "windows", self.cancelled)
        self.assertTrue((self.root / "windows/node.exe").is_file())

    def test_failed_or_cancelled_install_keeps_previous_files_and_retry_publishes(self):
        root = self.root / "runtime"
        root.mkdir()
        (root / "keep").write_text("previous")
        mode = "fail"

        def download(archive, *args):
            archive.touch()

        def extract(archive, target, *args):
            node = target / "bin/node"
            node.parent.mkdir(parents=True)
            node.write_text("node")

        def run(command, staging, environment, progress, cancelled, stage, **kwargs):
            self.assertEqual((root / "keep").read_text(), "previous")
            if stage == "dependencies":
                entry = staging / "node_modules/@deepseek-ai/dsh/lib/bin.js"
                entry.parent.mkdir(parents=True)
                entry.write_text("harness")
                self.assertEqual(Path(environment["PATH"].split(os.pathsep)[0]), staging / "node/bin")
            if stage == "verify":
                if mode == "fail":
                    raise RuntimeError("broken dependency")
                if mode == "cancel":
                    cancelled.set()

        with patch.object(installer, "_download_node", side_effect=download), patch.object(
                installer, "_extract_node", side_effect=extract), patch.object(installer, "_run_command", side_effect=run) as runner:
            for mode in ("fail", "cancel"):
                with self.assertRaises(RuntimeError):
                    installer.install_runtime(self.root, self.progress, self.cancelled)
                self.assertEqual((root / "keep").read_text(), "previous")
                self.assertFalse(list(self.root.glob(".install-*")))
                self.cancelled.clear()
            mode = "success"
            installer.install_runtime(self.root, self.progress, self.cancelled)
            self.assertIsNotNone(installer.runtime_command(root))
            self.assertFalse((root / "keep").exists())
            runner.reset_mock()
            installer.install_runtime(self.root, self.progress, self.cancelled)
            runner.assert_not_called()

    def test_concurrent_installation_reports_busy_and_releases_lock(self):
        with installer._installation_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "Another Launcher"):
                with installer._installation_lock(self.root):
                    self.fail("acquired twice")
        with installer._installation_lock(self.root):
            pass

    def test_failed_command_reports_bounded_diagnostics_without_registry_credentials(self):
        with self.assertRaises(RuntimeError) as error:
            installer._run_command([sys.executable, "-c",
                "print('x'*12000); print('https://user:secret@registry.example/fail'); raise SystemExit(1)"],
                self.root, os.environ.copy(), self.progress, self.cancelled, "dependencies")
        self.assertIn("exit 1", str(error.exception))
        self.assertNotIn("secret", str(error.exception))
        self.assertLess(len(str(error.exception)), 4200)

    def test_cancelling_command_terminates_child(self):
        pidfile = self.root / "pid"

        def cancel():
            deadline = time.monotonic() + 5
            while not pidfile.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.cancelled.set()

        thread = threading.Thread(target=cancel)
        thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                installer._run_command([sys.executable, "-c",
                    "import os,time,pathlib; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(120)"],
                    self.root, os.environ.copy(), self.progress, self.cancelled, "dependencies")
            pid = int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        finally:
            thread.join(timeout=5)

    @unittest.skipIf(os.name == "nt", "POSIX process groups")
    def test_termination_cleans_up_tools_after_the_leader_exits(self):
        child = '''import json, os, signal, socket, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen()
Path(sys.argv[1]).write_text(json.dumps({'port':listener.getsockname()[1]}))
time.sleep(60)
'''
        parent = '''import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]],
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while not Path(sys.argv[2]).exists(): time.sleep(.01)
if sys.argv[3] == 'wait': time.sleep(60)
'''
        import json
        for mode in ("exit", "wait"):
            with self.subTest(leader=mode):
                ready = self.root / (mode + ".json")
                process = subprocess.Popen([sys.executable, "-c", parent, child, str(ready), mode],
                                           start_new_session=True)
                try:
                    deadline = time.monotonic() + 5
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(ready.exists())
                    port = json.loads(ready.read_text())["port"]
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        pass
                    if mode == "exit":
                        process.wait(timeout=5)
                    installer.terminate_process(process)
                    deadline = time.monotonic() + 3
                    while True:
                        try:
                            with socket.create_connection(("127.0.0.1", port), timeout=.1):
                                pass
                        except OSError:
                            break
                        self.assertLess(time.monotonic(), deadline, "Agent's tool is still listening after stop")
                        time.sleep(.02)
                    self.assertIsNotNone(process.poll())
                    with patch.object(installer.os, "killpg") as kill:
                        installer.terminate_process(process)
                        kill.assert_not_called()
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
