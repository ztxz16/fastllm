"""GPU 权重验收器的独立防误报测试；不调用 CUDA、不编译 C++。"""

import copy
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

from run_qwen3_device_weights import (BACKEND, compare_payload, execute, fp16_reference,
                                      prepare_reload, verify, verify_rejection)
from run_qwen3_shards import fixture, reference


class DeviceWeightValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, self.tensors = fixture(self.root / "model", dtype="BF16", tied=True, bias=True)
        self.metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
        self.report = dict(status="PASS", backend=BACKEND, workers=[
            dict(rank=i, device=i, pid=100 + i, cuda_runtime_version=12060, nccl_version=22203)
            for i in range(2)], cycles=[])
        for cycle_index in range(2):
            directory = self.root / f"cycle-{cycle_index}"
            directory.mkdir()
            ranks = []
            for rank in range(2):
                expected, _ = reference(self.config, self.tensors, rank, 2)
                entries = []
                for index, (name, target) in enumerate(expected.items()):
                    data = fp16_reference(target)
                    filename = f"{rank}-{index}.bin"
                    (directory / filename).write_bytes(data)
                    entries.append(dict(name=name, shape=target["shape"], dtype="F16", source_dtype="BF16",
                                        file=filename, bytes=len(data)))
                ranks.append(dict(rank=rank, tensors=entries, device_bytes=sum(t["bytes"] for t in entries),
                                  source_bytes_read=sum(len(t["data"]) for t in expected.values() if not t["zero"])))
            self.report["cycles"].append(dict(directory=directory.name, ranks=ranks))

    def validate(self):
        verify(self.report, self.root, self.config, self.tensors, [0, 1], 2, self.metrics)

    def test_correct_reload(self):
        self.validate()
        self.assertGreater(self.metrics["count"], 0)
        self.assertEqual(self.metrics["failed"], 0)

    def test_corrupt_second_rank_reload(self):
        entry = self.report["cycles"][1]["ranks"][1]["tensors"][0]
        path = self.root / "cycle-1" / entry["file"]
        data = bytearray(path.read_bytes())
        data[1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(AssertionError, "GPU payload differs"):
            self.validate()
        self.assertEqual(self.metrics["failed"], 1)

    def test_signed_zero_mismatch_is_counted(self):
        with self.assertRaises(AssertionError):
            compare_payload(b"\0\x80", b"\0\0", self.metrics, "zero")
        self.assertEqual(self.metrics["failed"], 1)

    def test_missing_cycle_or_rank(self):
        original = copy.deepcopy(self.report)
        self.report["cycles"].pop()
        with self.assertRaisesRegex(AssertionError, "cycle"):
            self.validate()
        self.report = original
        self.report["cycles"][0]["ranks"].pop()
        with self.assertRaisesRegex(AssertionError, "rank"):
            self.validate()

    def test_wrong_device_or_backend(self):
        self.report["workers"][1]["device"] = 0
        with self.assertRaisesRegex(AssertionError, "identity"):
            self.validate()
        self.report["workers"][1]["device"] = 1
        self.report["backend"] = "cpu"
        with self.assertRaises(AssertionError):
            self.validate()

    def test_missing_or_duplicate_tensor(self):
        entries = self.report["cycles"][0]["ranks"][0]["tensors"]
        entries.append(entries[0])
        with self.assertRaisesRegex(AssertionError, "tensor list"):
            self.validate()

    def test_invalid_accounting(self):
        self.report["cycles"][0]["ranks"][0]["device_bytes"] += 2
        with self.assertRaisesRegex(AssertionError, "accounting"):
            self.validate()

    def test_nan_is_not_accepted(self):
        with self.assertRaises(AssertionError):
            compare_payload(struct.pack("<e", float("nan")), b"\0\0", self.metrics, "nan")
        self.assertEqual(self.metrics["failed"], 1)

    def test_half_rounding_reference(self):
        values = [0.0, -0.0, 2**-25, 3 * 2**-25, 1 + 2**-11, 1 + 3 * 2**-11]
        result = fp16_reference(dict(dtype="F32", data=struct.pack("<6f", *values)))
        self.assertEqual(struct.unpack("<6H", result), (0, 0x8000, 0, 2, 0x3c00, 0x3c02))
        with self.assertRaises(ValueError):
            fp16_reference(dict(dtype="F32", data=struct.pack("<f", 65536)))

    def test_rejection_requires_full_group_cleanup(self):
        report = dict(status="FAIL", backend=BACKEND, error="rank 1: bad value",
                      group_failed=True, workers_reaped=True)
        verify_rejection(1, report, "bad value")
        for field in ("group_failed", "workers_reaped"):
            with self.assertRaises(AssertionError):
                verify_rejection(1, dict(report, **{field: False}), "bad value")
        with self.assertRaises(AssertionError):
            verify_rejection(-9, report, "bad value")

    def test_reload_rejects_stale_gpu_weights(self):
        updated = prepare_reload(self.root, self.config, self.tensors, indexed=True)
        with self.assertRaisesRegex(AssertionError, "GPU payload differs"):
            verify(self.report, self.root, self.config, self.tensors, [0, 1], 2, self.metrics, updated)

    def test_reload_accepts_changed_gpu_weights(self):
        updated = prepare_reload(self.root, self.config, self.tensors)
        for rank, result in enumerate(self.report["cycles"][1]["ranks"]):
            expected, _ = reference(*updated, rank, 2)
            for entry in result["tensors"]:
                data = fp16_reference(expected[entry["name"]])
                (self.root / "cycle-1" / entry["file"]).write_bytes(data)
        verify(self.report, self.root, self.config, self.tensors, [0, 1], 2, self.metrics, updated)
        self.assertEqual(self.metrics["failed"], 0)

    def test_missing_checker_preserves_reason(self):
        with self.assertRaises(FileNotFoundError):
            execute(self.root / "missing", self.root / "worker", self.root, [0, 1], {})
        self.assertIn("FileNotFoundError", (self.root / "stderr.log").read_text())
        self.assertTrue((self.root / "command.txt").is_file())

    def test_interrupted_or_timed_out_checker_preserves_reason(self):
        for error in (KeyboardInterrupt(), subprocess.TimeoutExpired("checker", 420)):
            with self.subTest(error=type(error).__name__):
                with mock.patch("run_qwen3_device_weights.wait_case", side_effect=error):
                    with self.assertRaises(type(error)):
                        execute(Path("/bin/false"), self.root / "worker", self.root, [0, 1], {})
                self.assertIn(type(error).__name__, (self.root / "stderr.log").read_text())

    def test_invalid_pid_is_rejected(self):
        self.report["workers"][1]["pid"] = 0
        with self.assertRaisesRegex(AssertionError, "PID"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
