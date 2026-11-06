"""仅验证验收器能拒绝错误输出；不替代 C++ 分片读取器的真实运行。"""

import copy
import json
import signal
import struct
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest import mock
import zipfile

from run_process_tp import save_report
from run_qwen3_shards import decode, encode, fixture, invoke, reference, rejection_case, verify_rank, verify_rejection


class ShardReportValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config, self.tensors = fixture(self.directory / "model", bias=True, tied=True, indexed=True)
        self.expected, self.metadata = reference(self.config, self.tensors, 1, 4)
        self.report = dict(status="PASS", backend="qwen3_rank_local_safetensors", plan_only=False,
                           tensors=[], **self.metadata)
        for index, (name, target) in enumerate(self.expected.items()):
            filename = f"{index}.bin"
            (self.directory / filename).write_bytes(target["data"])
            self.report["tensors"].append(dict(
                name=name, dtype=target["dtype"], shape=target["shape"], zero_fill=target["zero"],
                file=filename, bytes=len(target["data"]), bytes_read=0 if target["zero"] else len(target["data"]),
                slices=[dict(source=s, row_begin=a, row_end=b, column_begin=c, column_end=d)
                        for s, a, b, c, d in target["slices"]]))
        self.report["tensor_bytes"] = sum(len(t["data"]) for t in self.expected.values())
        self.report["payload_bytes_read"] = sum(len(t["data"]) for t in self.expected.values() if not t["zero"])
        self.report["max_tensor_buffer_bytes"] = max(len(t["data"]) for t in self.expected.values())
        self.metrics = dict(max=0.0, sum=0.0, count=0, failed=0)

    def validate(self):
        reconstructed = {name: bytearray(len(t["data"])) for name, t in self.tensors.items()}
        coverage = copy.deepcopy(reconstructed)
        verify_rank(self.report, self.directory, self.expected, self.metadata, self.tensors,
                    reconstructed, coverage, self.metrics)

    def test_reference_report_is_accepted(self):
        self.validate()
        self.assertGreater(self.metrics["count"], 0)
        self.assertEqual(self.metrics["failed"], 0)

    def test_wrong_payload_is_rejected_and_counted(self):
        path = self.directory / self.report["tensors"][0]["file"]
        data = bytearray(path.read_bytes())
        data[1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(AssertionError, "output differs"):
            self.validate()
        self.assertEqual(self.metrics["failed"], 1)
        self.assertGreater(self.metrics["max"], 0)

    def test_wrong_head_ownership_is_rejected(self):
        self.report["kv_head_begin"] += 1
        with self.assertRaises(AssertionError):
            self.validate()

    def test_wrong_rank_is_rejected(self):
        self.report["rank"] = 999
        with self.assertRaises(AssertionError):
            self.validate()

    def test_wrong_world_size_is_rejected(self):
        self.report["world_size"] = 999
        with self.assertRaises(AssertionError):
            self.validate()

    def test_overread_is_rejected(self):
        self.report["tensors"][0]["bytes_read"] += 2
        with self.assertRaises(AssertionError):
            self.validate()

    def test_missing_tensor_is_rejected(self):
        self.report["tensors"].pop()
        with self.assertRaisesRegex(AssertionError, "target list"):
            self.validate()

    def test_duplicate_tensor_is_rejected(self):
        self.report["tensors"].append(self.report["tensors"][0])
        with self.assertRaisesRegex(AssertionError, "target list"):
            self.validate()

    def test_dtype_roundtrip(self):
        values = [-15.625, -1.0, 0.0, 0.125, 15.625]
        for dtype in ("F16", "BF16", "F32"):
            with self.subTest(dtype=dtype):
                self.assertEqual(decode(encode(values, dtype), dtype), values)


class ShardFailureReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_partial_success_cannot_report_pass(self):
        metadata = dict(expected_case_count=2)
        rows = [dict(case="first", status="PASS")]
        self.assertEqual(save_report(self.directory, rows, metadata), "FAIL")
        partial = json.loads((self.directory / "summary.json").read_text())
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["completed_case_count"], 1)
        self.assertEqual(partial["expected_case_count"], 2)
        self.assertIn("1/2", (self.directory / "summary.md").read_text())
        rows.append(dict(case="second", status="PASS"))
        self.assertEqual(save_report(self.directory, rows, metadata), "PASS")
        self.assertTrue(json.loads((self.directory / "summary.json").read_text())["complete"])
        with zipfile.ZipFile(self.directory / "summary.xlsx") as archive:
            self.assertIsNone(archive.testzip())

    def test_default_report_contract_is_unchanged(self):
        self.assertEqual(save_report(self.directory, [dict(case="old", status="PASS")], {}), "PASS")

    def test_crash_is_not_an_expected_rejection(self):
        report = dict(status="FAIL", backend="qwen3_rank_local_safetensors", reason="invalid rank/world size")
        verify_rejection(1, report, "invalid rank")
        for code in (0, 2, -signal.SIGSEGV, -signal.SIGABRT):
            with self.subTest(code=code), self.assertRaises(AssertionError):
                verify_rejection(code, report, "invalid rank")

    def test_missing_checker_keeps_logs(self):
        logs = self.directory / "logs"
        with self.assertRaises(FileNotFoundError):
            invoke(self.directory / "missing", Path("model"), 0, 2, logs)
        self.assertEqual((logs / "stdout.log").read_bytes(), b"")
        self.assertIn("missing", (logs / "stderr.log").read_text())
        self.assertEqual(json.loads((logs / "invocation.json").read_text())["cwd"], str(Path.cwd()))

    def test_interrupt_and_timeout_keep_partial_logs(self):
        for exception in (KeyboardInterrupt(), subprocess.TimeoutExpired("checker", 30)):
            with self.subTest(error=type(exception).__name__):
                logs = self.directory / type(exception).__name__

                def fail(command, *, stdout, stderr, **kwargs):
                    stdout.write(b'{"status":')
                    stderr.write(b"before failure")
                    raise exception

                with mock.patch("run_qwen3_shards.subprocess.run", side_effect=fail):
                    with self.assertRaises(type(exception)):
                        invoke(Path("checker"), Path("model"), 0, 2, logs)
                self.assertEqual((logs / "stdout.log").read_bytes(), b'{"status":')
                self.assertEqual((logs / "stderr.log").read_bytes(), b"before failure")

    def test_duplicate_fixtures_contain_decoded_duplicate_keys(self):
        for kind in ("duplicate_config", "duplicate_escaped_key", "duplicate_nested_key",
                     "duplicate_index", "duplicate_tensor", "duplicate_tensor_dtype"):
            with self.subTest(case=kind):
                directory = self.directory / kind
                directory.mkdir()
                result = (1, dict(status="FAIL", backend="qwen3_rank_local_safetensors", reason="duplicate JSON key"))
                with mock.patch("run_qwen3_shards.invoke", return_value=result):
                    rejection_case(Path("not-executed"), directory, kind)
                if kind == "duplicate_index":
                    text = (directory / "model/model.safetensors.index.json").read_text()
                elif kind.startswith("duplicate_tensor"):
                    data = (directory / "model/model.safetensors").read_bytes()
                    text = data[8:8 + struct.unpack("<Q", data[:8])[0]].decode()
                else:
                    text = (directory / "model/config.json").read_text()
                duplicates = []

                def collect(pairs):
                    seen = set()
                    for key, value in pairs:
                        if key in seen:
                            duplicates.append(key)
                        seen.add(key)
                    return dict(pairs)

                json.loads(text, object_pairs_hook=collect)
                self.assertEqual(len(duplicates), 1)


if __name__ == "__main__":
    unittest.main()
