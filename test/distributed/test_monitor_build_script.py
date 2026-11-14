#!/usr/bin/env python3
"""在隔离 PATH 中用替身验证服务器脚本；不调用真实 CMake、编译器或 GPU。"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("run_process_tp_monitor_v100.sh")
TOOL = r'''
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
root = Path(os.environ["FIXTURE_ROOT"])
with (root / "calls.jsonl").open("a") as output:
    output.write(json.dumps([name, *sys.argv[1:]]) + "\n")
if name == "cmake" and sys.argv[1] != "--build":
    build = Path(sys.argv[sys.argv.index("-B") + 1])
    build.mkdir(parents=True, exist_ok=True)
    counter = root / "configure-count"
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count))
    mode = os.environ["FIXTURE_MODE"]
    if mode == "configure_failure":
        sys.exit(17)
    if mode == "missing_cache":
        sys.exit(0)
    values = {"USE_CUDA": "ON", "USE_NUMAS": "OFF", "BUILD_PROCESS_TP": "ON", "CUDA_ARCH": "70"}
    if mode == "reset_once" and count == 1:
        values = {"USE_CUDA": "OFF", "USE_NUMAS": "ON", "CUDA_ARCH": "native"}
    if mode == "invalid_cache":
        values.update(json.loads(os.environ["FIXTURE_OVERRIDES"]))
    cache = build / "CMakeCache.txt"
    cache.write_text("".join(f"{key}:STRING={value}\n" for key, value in values.items()))
elif name == "cmake" and os.environ["FIXTURE_MODE"] == "build_failure":
    sys.exit(18)
'''


class MonitorBuildScriptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fastllm-script-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        script = self.root / "test/distributed" / SCRIPT.name
        script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, script)
        self.script = script
        self.build = self.root / "build with spaces"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.bash = shutil.which("bash")
        self.assertIsNotNone(self.bash)

        # PATH 不包含系统工具目录，真实 cmake/nvcc 即使已安装也不能被误调用。
        for name in ("awk", "dirname"):
            executable = shutil.which(name)
            self.assertIsNotNone(executable)
            (self.bin / name).symlink_to(executable)
        for name in ("cmake", "nvcc", "git", "nvidia-smi", "fixture-python"):
            executable = self.bin / name
            executable.write_text(f"#!{sys.executable}\n" + TOOL)
            executable.chmod(0o755)

    def run_script(self, mode="healthy", overrides=None, custom="0"):
        environment = dict(os.environ, PATH=str(self.bin), FIXTURE_ROOT=str(self.root),
                           FIXTURE_MODE=mode, FIXTURE_OVERRIDES=json.dumps(overrides or {}),
                           TP_MONITOR_BUILD=str(self.build), TP_MONITOR_CUDA=str(self.root),
                           TP_MONITOR_PYTHON=str(self.bin / "fixture-python"), TP_MONITOR_CUSTOM=custom,
                           CUDA_VISIBLE_DEVICES="2,3")
        result = subprocess.run([self.bash, str(self.script)], env=environment,
                                capture_output=True, text=True, timeout=15)
        self.calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        self.configures = [call for call in self.calls if call[0] == "cmake" and call[1] != "--build"]
        self.builds = [call for call in self.calls if call[:2] == ["cmake", "--build"]]
        self.tests = [call for call in self.calls if call[0] == "fixture-python"]
        return result

    def test_valid_configuration_builds_once(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.configures), 1)
        self.assertEqual(len(self.builds), 1)
        self.assertEqual(len(self.tests), 1)
        self.assertIn("-DUSE_CUDA=ON", self.configures[0])
        self.assertIn("-DBUILD_PROCESS_TP=ON", self.configures[0])
        self.assertIn("fastllm_process_tp_package", self.builds[0])

    def test_cache_reset_reapplies_all_options_before_build(self):
        result = self.run_script("reset_once")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.configures), 2)
        self.assertEqual(self.configures[0], self.configures[1])
        self.assertEqual(len(self.builds), 1)
        self.assertEqual(len(self.tests), 1)
        self.assertLess(self.calls.index(self.configures[-1], self.calls.index(self.configures[0]) + 1),
                        self.calls.index(self.builds[0]))

    def test_persistent_invalid_cache_does_not_build(self):
        for key, value in (("USE_CUDA", "OFF"), ("USE_NUMAS", "ON"),
                           ("BUILD_PROCESS_TP", "OFF"), ("CUDA_ARCH", "native")):
            with self.subTest(key=key):
                (self.root / "calls.jsonl").write_text("")
                result = self.run_script("invalid_cache", {key: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(self.configures), 2)
                self.assertFalse(self.builds)
                self.assertFalse(self.tests)
                self.assertIn("停止构建", result.stderr)

    def test_missing_cache_does_not_build(self):
        result = self.run_script("missing_cache")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.configures), 2)
        self.assertFalse(self.builds)
        self.assertFalse(self.tests)

    def test_configure_failure_is_not_retried(self):
        result = self.run_script("configure_failure")
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(self.configures), 1)
        self.assertFalse(self.builds)
        self.assertFalse(self.tests)

    def test_build_failure_does_not_run_tests(self):
        result = self.run_script("build_failure")
        self.assertEqual(result.returncode, 18)
        self.assertEqual(len(self.builds), 1)
        self.assertFalse(self.tests)

    def test_custom_configuration_runs_both_suites(self):
        result = self.run_script(custom="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.tests), 2)
        self.assertNotIn("--custom-allreduce", self.tests[0])
        self.assertIn("--custom-allreduce", self.tests[1])


if __name__ == "__main__":
    unittest.main()
