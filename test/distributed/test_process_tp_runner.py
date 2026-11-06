#!/usr/bin/env python3
"""用真实 Python 子进程验证验收器清理；不加载推理库，不编译。"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import run_process_tp as runner


def fixture_worker(mode, parent):
    if mode == "watch":
        print("ready", flush=True)
        while os.getppid() == parent:
            time.sleep(0.01)
    else:
        # 父进程退出时，孤儿进程组可能收到 SIGHUP/SIGCONT；不能靠它们退出。
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        os.kill(os.getpid(), signal.SIGSTOP)
        time.sleep(120)


def fixture_parent(mode, identity_path):
    worker_mode = "watch" if mode == "parent_death" else "stop"
    worker = subprocess.Popen([sys.executable, __file__, "--worker", worker_mode,
                               str(os.getpid()), str(identity_path)], stdout=subprocess.PIPE)
    if worker_mode == "watch":
        if worker.stdout.readline() != b"ready\n":
            raise RuntimeError("fixture worker did not become ready")
    else:
        _, status = os.waitpid(worker.pid, os.WUNTRACED)
        if not os.WIFSTOPPED(status):
            raise RuntimeError("fixture worker did not stop")
    identity_path.write_text(json.dumps({"parent": os.getpid(), "worker": worker.pid}), encoding="utf-8")
    print("fixture stderr", file=sys.stderr, flush=True)

    data = {"status": "PASS", "workers": [{"pid": worker.pid}]}
    if mode in ("startup_timeout", "interrupt_startup"):
        print('{"workers":', end="", flush=True)
    elif mode == "invalid_output":
        print("not JSON", flush=True)
    else:
        if mode == "bad_startup":
            data["workers"].append({"pid": 0})
        print(json.dumps(data), flush=True)
    if mode in ("success", "invalid_output", "nonzero"):
        return 7 if mode == "nonzero" else 0
    if mode in ("interrupt", "interrupt_startup"):
        time.sleep(0.1)
        os.kill(os.getppid(), signal.SIGINT)
    time.sleep(120)
    return 0


class RunnerCleanupTest(unittest.TestCase):
    def setUp(self):
        runner.enable_child_subreaper()
        self.directory = tempfile.TemporaryDirectory(prefix="fastllm-runner-regression-")
        self.output = Path(self.directory.name)

    def tearDown(self):
        # 被测清理回归时也要兜底，防止测试自身遗留 SIGSTOP 进程。
        for path in self.output.glob("*.pids.json"):
            identities = json.loads(path.read_text(encoding="utf-8"))
            for pid in identities.values():
                try:
                    fd = os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                try:
                    command = Path(f"/proc/{pid}/cmdline").read_bytes()
                    if str(self.output).encode() in command:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(fd)
            deadline = time.monotonic() + 5
            while True:
                try:
                    pid, _ = os.waitpid(-identities["parent"], os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    if time.monotonic() >= deadline:
                        self.fail("fixture cleanup timed out")
                    time.sleep(0.01)
        self.directory.cleanup()

    def command(self, mode):
        return [sys.executable, str(Path(__file__).resolve()), "--fixture", mode,
                str(self.output / f"{mode}.pids.json")]

    def assert_reaped(self, mode):
        identities = json.loads((self.output / f"{mode}.pids.json").read_text(encoding="utf-8"))
        for kind, pid in identities.items():
            with self.subTest(process=kind, pid=pid):
                self.assertFalse(Path(f"/proc/{pid}").exists(), "process is still alive or a zombie")
                with self.assertRaises(ChildProcessError):
                    os.waitpid(pid, os.WNOHANG)

    def test_timeout_kills_stopped_worker_and_preserves_logs(self):
        row = runner.run_case("timeout", self.command("timeout"), self.output, 1)
        self.assertEqual(row["status"], "FAIL")
        self.assertIn("timed out", row["reason"])
        self.assertIn("workers", (self.output / "timeout.stdout.log").read_text())
        self.assertIn("fixture stderr", (self.output / "timeout.stderr.log").read_text())
        self.assert_reaped("timeout")

    def test_success_cleans_remaining_workers(self):
        row = runner.run_case("success", self.command("success"), self.output, 5)
        self.assertEqual(row["status"], "PASS", row)
        self.assert_reaped("success")

    def test_invalid_output_cleans_remaining_workers(self):
        row = runner.run_case("invalid_output", self.command("invalid_output"), self.output, 5)
        self.assertEqual(row["status"], "FAIL")
        self.assert_reaped("invalid_output")

    def test_nonzero_exit_overrides_pass_json(self):
        row = runner.run_case("nonzero", self.command("nonzero"), self.output, 5)
        self.assertEqual(row["status"], "FAIL")
        self.assert_reaped("nonzero")

    def test_keyboard_interrupt_cleans_group(self):
        with self.assertRaises(KeyboardInterrupt):
            runner.run_case("interrupt", self.command("interrupt"), self.output, 5)
        self.assert_reaped("interrupt")

    def test_parent_death_observes_worker_self_exit(self):
        row = runner.parent_death_case(self.command("parent_death"), 5, self.output)
        self.assertEqual(row["status"], "PASS", row)
        self.assert_reaped("parent_death")

    def test_parent_death_cannot_pass_by_killing_worker(self):
        row = runner.parent_death_case(self.command("stopped_parent_death"), 5, self.output)
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual(row["reason"], "worker survived parent death")
        self.assert_reaped("stopped_parent_death")

    def test_parent_death_startup_failure_closes_pidfds(self):
        count = len(list(Path("/proc/self/fd").iterdir()))
        row = runner.parent_death_case(self.command("bad_startup"), 5, self.output)
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), count)
        self.assert_reaped("bad_startup")

    def test_parent_death_startup_timeout_preserves_partial_output(self):
        row = runner.parent_death_case(self.command("startup_timeout"), 1, self.output)
        self.assertEqual(row["status"], "FAIL")
        self.assertIn("in time", row["reason"])
        self.assertEqual((self.output / "parent_death.stdout.log").read_text(), '{"workers":')
        self.assert_reaped("startup_timeout")

    def test_parent_death_interrupt_cleans_group(self):
        with self.assertRaises(KeyboardInterrupt):
            runner.parent_death_case(self.command("interrupt_startup"), 5, self.output)
        self.assert_reaped("interrupt_startup")

    def test_does_not_kill_or_reap_unrelated_children(self):
        live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
        exited = subprocess.Popen([sys.executable, "-c", "raise SystemExit(23)"], start_new_session=True)
        try:
            runner.wait_case(exited, 5)
            row = runner.run_case("success", self.command("success"), self.output, 5)
            self.assertEqual(row["status"], "PASS", row)
            self.assertIsNone(live.poll())
            self.assertEqual(exited.wait(timeout=5), 23)
            self.assert_reaped("success")
        finally:
            live.kill()
            live.wait(timeout=5)
            exited.kill()
            exited.wait(timeout=5)

    def test_missing_executable_is_reported(self):
        row = runner.run_case("missing", [str(self.output / "does-not-exist")], self.output, 5)
        self.assertEqual(row["status"], "FAIL")
        self.assertIn("No such file", row["reason"])


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RunnerCleanupTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "backend": "python/process-cleanup",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "max_abs_error": None, "mean_abs_error": None, "failed_elements": None}
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        fixture_worker(sys.argv[2], int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--fixture":
        raise SystemExit(fixture_parent(sys.argv[2], Path(sys.argv[3])))
    else:
        raise SystemExit(0 if run_tests()["status"] == "PASS" else 1)
