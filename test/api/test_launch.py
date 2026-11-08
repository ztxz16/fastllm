#!/usr/bin/env python3
"""Tests for tools/fastllm_pytools/launch.py: one new folder per run, nothing appended or overwritten."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "fastllm_pytools"
sys.path.insert(0, str(TOOLS))
import launch  # noqa: E402

READY = "INFO:     Uvicorn running on http://0.0.0.0:8080"


def child(code: str):
    return [sys.executable, "-c", textwrap.dedent(code)]


def tree_hashes(root: Path):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "logs"

    def tearDown(self):
        self.tmp.cleanup()

    def runs(self):
        return sorted((self.root / "runs").iterdir())

    def test_one_new_folder_complete_log_no_duplicate(self):
        rc = launch.run(self.root, child("""
            import sys
            print("hello stdout")
            print("hello stderr", file=sys.stderr)
            print("INFO:     Uvicorn running on http://0.0.0.0:8080")
            print("[fastllm-event] category=decode request=1 event=progress")
        """), mode="debug", tty=None)
        self.assertEqual(rc, 0)
        runs = self.runs()
        self.assertEqual(len(runs), 1)
        names = sorted(p.name for p in runs[0].iterdir())
        self.assertEqual(names, ["diagnostics", "run-end.json", "run-meta.json", "server.log"])
        log = (runs[0] / "server.log").read_text()
        for s in ("hello stdout", "hello stderr", READY, "category=decode"):
            self.assertIn(s, log)
        self.assertTrue(log.startswith("[ftllm.launch] run_dir="))
        self.assertFalse(list(runs[0].rglob("raw.log")))
        end = json.loads((runs[0] / "run-end.json").read_text())
        self.assertEqual(end["child_returncode"], 0)
        self.assertEqual(end["server_log_lines"], 4)

    def test_second_run_never_touches_first(self):
        launch.run(self.root, child("print('first')"), tty=None)
        before = tree_hashes(self.root)
        time.sleep(0.01)
        launch.run(self.root, child("print('second')"), tty=None)
        runs = self.runs()
        self.assertEqual(len(runs), 2)
        after = tree_hashes(self.root)
        for k, v in before.items():
            self.assertEqual(after[k], v, k)
        self.assertIn("second", (runs[1] / "server.log").read_text())
        self.assertNotIn("second", (runs[0] / "server.log").read_text())

    def test_name_collision_fails_instead_of_reusing(self):
        launch.run(self.root, child("print(1)"), tty=None, run_id_fn=lambda: "fixed")
        before = tree_hashes(self.root)
        with self.assertRaises(FileExistsError):
            launch.run(self.root, child("print(2)"), tty=None, run_id_fn=lambda: "fixed")
        self.assertEqual(tree_hashes(self.root), before)

    def test_run_id_has_time_and_pid(self):
        rid = launch.default_run_id(pid=4242)
        self.assertRegex(rid, r"^\d{8}T\d{6}\.\d{6}Z-4242$")

    def test_exit_code_passed_through(self):
        self.assertEqual(launch.run(self.root, child("import sys; sys.exit(3)"), tty=None), 3)
        end = json.loads((self.runs()[0] / "run-end.json").read_text())
        self.assertEqual(end["child_returncode"], 3)

    def test_child_env_points_into_run_folder(self):
        launch.run(self.root, child("""
            import os
            print("DIAG=" + os.environ["FASTLLM_DIAGNOSTIC_LOG_DIR"])
            print("RUN=" + os.environ["FASTLLM_RUN_DIR"])
        """), tty=None)
        run = self.runs()[0]
        log = (run / "server.log").read_text()
        self.assertIn("DIAG=" + str(run / "diagnostics"), log)
        self.assertIn("RUN=" + str(run), log)
        meta = json.loads((run / "run-meta.json").read_text())
        self.assertEqual(meta["env"]["FASTLLM_DIAGNOSTIC_LOG_DIR"], str(run / "diagnostics"))

    def test_user_log_only_in_user_mode(self):
        launch.run(self.root, child(f"print({READY!r})"), mode="user", tty=None)
        launch.run(self.root, child(f"print({READY!r})"), mode="debug", tty=None)
        a, b = self.runs()
        self.assertTrue((a / "user.log").is_file())
        self.assertFalse((b / "user.log").exists())

    def test_record_file_copied_with_hash(self):
        src = Path(self.tmp.name) / "deploy.sh"
        src.write_text("echo deploy\n")
        launch.run(self.root, child("print(1)"), tty=None, record_files=[src])
        run = self.runs()[0]
        self.assertEqual((run / "recorded" / "deploy.sh").read_text(), "echo deploy\n")
        meta = json.loads((run / "run-meta.json").read_text())
        self.assertEqual(meta["recorded_files"][0]["sha256"], hashlib.sha256(b"echo deploy\n").hexdigest())

    def test_sigterm_forwarded_once_and_shutdown_logged(self):
        # The launcher runs as its own process; the child counts the signals it receives.
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(TOOLS)!r})
            import launch
            from pathlib import Path
            raise SystemExit(launch.run(Path({str(self.root)!r}), [sys.executable, "-c", {textwrap.dedent('''
                import signal, sys, time
                hits = []
                signal.signal(signal.SIGTERM, lambda s, f: hits.append(s))
                print("ready", flush=True)
                while not hits: time.sleep(0.01)
                time.sleep(0.3)
                print("shutdown signals=%d" % len(hits), flush=True)
            ''')!r}], tty=None))
        """)
        p = subprocess.Popen([sys.executable, "-c", script])
        deadline = time.time() + 10
        log = None
        while time.time() < deadline:
            logs = list((self.root / "runs").glob("*/server.log")) if (self.root / "runs").exists() else []
            if logs and "ready" in logs[0].read_text():
                log = logs[0]
                break
            time.sleep(0.02)
        self.assertIsNotNone(log)
        os.kill(p.pid, signal.SIGTERM)
        self.assertEqual(p.wait(timeout=10), 0)
        self.assertIn("shutdown signals=1", log.read_text())
        end = json.loads((log.parent / "run-end.json").read_text())
        self.assertEqual(end["signals_forwarded"], [int(signal.SIGTERM)])

    def test_cli_requires_command(self):
        with self.assertRaises(SystemExit):
            launch._main(["--log_dir", str(self.root)])


if __name__ == "__main__":
    unittest.main()
