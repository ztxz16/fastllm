import json
import os
import stat
import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.deploy import (
    DeployConfig,
    build_child_environment,
    build_fastllm_argv,
    load_saved_configs,
    save_saved_configs,
    validate_config,
)
from fastllm_pytools.profile import (
    ProfileError,
    ProfileLockError,
    ProfileManager,
    add_profile_subparsers,
)


MODEL = ("/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/models/"
         "Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-LOW-MTP-IQ4_XS.gguf")
ORI = "/run/media/ezra/13D010B6FDBC1A06/1CatVLLM/models/Qwen3.6-27B-bf16"


def profile(name, kv="fp8_e4m3", turbo="0"):
    return DeployConfig(
        name=name,
        command="server",
        model=MODEL,
        ori=ORI,
        model_name="qwen3.6-fastllm",
        host="127.0.0.1",
        port="18002",
        activation_dtype="float16",
        low_memory=True,
        kv_cache_dtype=kv,
        tokens="262144",
        max_batch="5",
        mtp="2",
        threads="2",
        chunked_prefill_size="8192",
        prefix_snapshot_interval_pages="64",
        cuda_embedding=True,
        startup_timeout="1",
        default_max_tokens="16384",
        env_vars=(
            f"FASTLLM_QWEN35_TURBO3_KV={turbo} "
            "FASTLLM_QWEN35_INTERLEAVE_LONG_PREFILL=1 "
            "FASTLLM_QWEN35_BATCHED_MTP=0"),
    )


class FakeProcess:
    def __init__(self, pid, return_code=None):
        self.pid = pid
        self.return_code = return_code

    def poll(self):
        return self.return_code


class FakeManager(ProfileManager):
    def __init__(self, *args, outcomes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = list(outcomes or [])
        self.identities = {}
        self.next_pid = 1000
        self.stop_calls = []

    def _spawn(self, config, log_path):
        pid = self.next_pid
        self.next_pid += 1
        self.identities[pid] = f"start-{pid}"
        Path(log_path).touch(mode=0o600)
        return FakeProcess(pid), build_fastllm_argv(config) + [
            "--startup-progress", "ndjson"]

    def _proc_start_time(self, pid):
        return self.identities.get(int(pid))

    def _wait_ready(self, process, config, timeout):
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, BaseException):
            raise outcome

    def _stop_process(self, state, grace=30.0):
        self.stop_calls.append(dict(state))
        pid = state.get("pid")
        if isinstance(pid, int):
            self.identities.pop(pid, None)


class DeploySchemaTests(unittest.TestCase):
    def test_v1_migration_and_v2_atomic_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "profiles.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "commands": [{
                    "name": "legacy", "command": "server",
                    "atype": "float16", "device": "cuda:0",
                    "prefix_cache_snapshot_interval_pages": "16",
                }]}, handle)
            migrated = load_saved_configs(path)[0]
            self.assertEqual(migrated.activation_dtype, "float16")
            self.assertEqual(migrated.cuda_device_id, "0")
            self.assertEqual(migrated.prefix_snapshot_interval_pages, "16")
            save_saved_configs([migrated], path)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["version"], 2)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_exact_argv_and_environment_isolation(self):
        config = profile("speed")
        argv = build_fastllm_argv(config)
        self.assertIn("--max_batch", argv)
        self.assertNotIn("--batch", argv)
        self.assertIn("--atype", argv)
        self.assertIn("float16", argv)
        self.assertIn("--default_max_tokens", argv)
        self.assertIn("--cuda_embedding", argv)
        self.assertIn("-l", argv)
        env = build_child_environment(config, {
            "PATH": "/bin",
            "FASTLLM_QWEN35_TURBO3_KV": "1",
            "FASTLLM_QWEN35_ENABLE_MTP": "9",
            "AUTH_TOKEN": "must-not-leak",
        })
        self.assertEqual(env["FASTLLM_QWEN35_TURBO3_KV"], "0")
        self.assertNotIn("FASTLLM_QWEN35_ENABLE_MTP", env)
        self.assertNotIn("AUTH_TOKEN", env)

    def test_profile_launcher_prefers_command_then_python_module(self):
        env = {"PATH": "/bin", "FASTLLM_PROFILE_PYTHON": sys.executable}
        module_argv = ProfileManager._resolve_launch_argv(
            ["ftllm", "server", "model"], env)
        self.assertEqual(module_argv[:3], [sys.executable, "-m", "ftllm.cli"])
        self.assertNotIn("FASTLLM_PROFILE_PYTHON", env)

        with tempfile.TemporaryDirectory() as directory:
            executable = os.path.join(directory, "ftllm")
            Path(executable).write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(executable, 0o700)
            env = {"PATH": directory, "FASTLLM_PROFILE_PYTHON": sys.executable}
            direct_argv = ProfileManager._resolve_launch_argv(
                ["ftllm", "server", "model"], env)
            self.assertEqual(direct_argv, [executable, "server", "model"])
            self.assertNotIn("FASTLLM_PROFILE_PYTHON", env)

    def test_turbo3_double_gate_and_speed_exclusion(self):
        missing = profile("capacity", kv="turbo3", turbo="0")
        self.assertTrue(any("Turbo3" in item for item in validate_config(missing)))
        capacity = profile("capacity", kv="turbo3", turbo="1")
        capacity.cuda_embedding = False
        self.assertEqual(validate_config(capacity), [])
        speed_bad = profile("speed", kv="fp8_e4m3", turbo="1")
        self.assertTrue(any("非 Turbo3" in item for item in validate_config(speed_bad)))


class ProfileLifecycleTests(unittest.TestCase):
    def make_manager(self, directory, configs, outcomes=None):
        config_path = os.path.join(directory, "profiles.json")
        save_saved_configs(configs, config_path)
        return FakeManager(
            config_path=config_path,
            state_dir=os.path.join(directory, "state"),
            outcomes=outcomes)

    def test_duplicate_names_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.make_manager(
                directory, [profile("same"), profile("same")])
            with self.assertRaisesRegex(ProfileError, "名称重复"):
                manager.profile_map()

    def test_lock_contention(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.make_manager(directory, [profile("speed")])
            contender = ProfileManager(
                config_path=manager.config_path,
                state_dir=str(manager.state_dir))
            with manager.lock():
                with self.assertRaises(ProfileLockError):
                    with contender.lock():
                        pass

    def test_stale_pid_identity_is_not_signaled(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.make_manager(directory, [profile("speed")])
            manager.save_state({
                "status": "running", "active": "old", "pid": 777,
                "proc_start_time": "reused"})
            result = manager.stop()
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(manager.stop_calls, [{
                "status": "running", "active": "old", "pid": 777,
                "proc_start_time": "reused"}])

    def test_start_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.make_manager(directory, [profile("speed")])
            state = manager.start("speed")
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["active"], "speed")
            self.assertTrue(manager.process_matches(state))
        with tempfile.TemporaryDirectory() as directory:
            manager = self.make_manager(
                directory, [profile("speed")],
                outcomes=[ProfileError("startup timeout")])
            with self.assertRaisesRegex(ProfileError, "startup timeout"):
                manager.start("speed")
            self.assertEqual(manager.load_state()["status"], "failed")

    def test_switch_success(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.make_manager(
                directory, [profile("a"), profile("b")])
            first = manager.start("a")
            state = manager.switch("b")
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["active"], "b")
            self.assertEqual(state["previous"]["name"], "a")
            self.assertFalse(manager.process_matches(first))

    def test_target_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.make_manager(
                directory, [profile("a"), profile("b")])
            manager.start("a")
            manager.outcomes = [ProfileError("target failed"), "ok"]
            with self.assertRaisesRegex(ProfileError, "已回滚"):
                manager.switch("b")
            state = manager.load_state()
            self.assertEqual(state["status"], "rolled_back")
            self.assertEqual(state["active"], "a")
            self.assertTrue(manager.process_matches(state))

    def test_target_and_rollback_failure_records_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.make_manager(
                directory, [profile("a"), profile("b")])
            manager.start("a")
            manager.outcomes = [
                ProfileError("target failed"), ProfileError("rollback failed")]
            with self.assertRaisesRegex(ProfileError, "回滚失败"):
                manager.switch("b")
            state = manager.load_state()
            self.assertEqual(state["status"], "failed")
            self.assertIn("target_error", state)
            self.assertIn("rollback_error", state)

    def test_readiness_requires_health_and_expected_model(self):
        state = {"ready": True, "model": "qwen3.6-fastllm"}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    payload = {"ready": state["ready"]}
                elif self.path == "/v1/models":
                    payload = {"data": [{"id": state["model"]}]}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = profile("ready")
            config.port = str(server.server_port)
            manager = ProfileManager()
            manager._wait_ready(FakeProcess(1), config, 0.5)
            state["model"] = "wrong"
            with self.assertRaisesRegex(ProfileError, "启动超时"):
                manager._wait_ready(FakeProcess(1), config, 0.05)
            state["model"] = "qwen3.6-fastllm"
            state["ready"] = False
            with self.assertRaisesRegex(ProfileError, "启动超时"):
                manager._wait_ready(FakeProcess(1), config, 0.05)
        finally:
            server.shutdown()
            server.server_close()

    def test_readiness_reports_early_exit(self):
        manager = ProfileManager()
        with self.assertRaisesRegex(ProfileError, "提前退出"):
            manager._wait_ready(FakeProcess(1, return_code=7), profile("exit"), 1)

    def test_public_state_redacts_secrets(self):
        state = {
            "config": {"api_key": "secret", "env_vars": "TOKEN=x"},
            "argv": ["ftllm", "--api_key", "secret"],
        }
        public = ProfileManager.public_state(state)
        self.assertEqual(public["config"]["api_key"], "<redacted>")
        self.assertEqual(public["config"]["env_vars"], "<redacted>")
        self.assertNotIn("secret", public["argv"])

    def test_cli_parser_registers_commands(self):
        parser = ArgumentParser()
        add_profile_subparsers(parser)
        args = parser.parse_args(["switch", "speed", "--grace", "2"])
        self.assertEqual(args.profile_action, "switch")
        self.assertEqual(args.name, "speed")
        self.assertEqual(args.grace, 2)


if __name__ == "__main__":
    unittest.main()
