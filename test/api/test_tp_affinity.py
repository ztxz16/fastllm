"""Automatic TP placement must not collide with the NUMA worker layout."""
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools import util


class TpAffinityTest(unittest.TestCase):
    def configure(self, env, multiple_caches=False):
        nodes = [f"/sys/devices/system/node/node{i}" for i in range(2)]

        def glob(pattern):
            if pattern.endswith("node[0-9]*"):
                return nodes
            return [nodes[0] + f"/cpu{i}" for i in range(32)]

        def read(path, *args, **kwargs):
            path = str(path)
            if path.endswith("numa_node"):
                value = "0"
            elif path.endswith("cpulist"):
                value = "0-31" if "node0" in path else "32-63"
            else:
                cpu = int(path.split("/cpu")[-1].split("/")[0])
                value = str(cpu // 4 if multiple_caches else 0) if path.endswith("/id") else str(cpu)
            return io.StringIO(value)

        with patch.dict(os.environ, env, clear=True), \
                patch.object(util.glob, "glob", side_effect=glob), \
                patch.object(util.os.path, "exists", return_value=True), \
                patch.object(util.os, "sched_getaffinity", return_value=set(range(32))), \
                patch.object(util, "_cuda_driver_device_info", return_value={0: {"pci_bus_id": "0000:01:00.0"}}), \
                patch("builtins.open", side_effect=read):
            util._configure_multicuda_worker_affinity("0,1", 30)
            return os.environ.get("FASTLLM_MULTICUDA_WORKER_CPU_BASE")

    def test_single_numa_node_reserves_only_unused_cores(self):
        self.assertEqual(self.configure({"FT_NUMAS": "1"}), "30")

    def test_llc_spread_keeps_workers_unbound(self):
        self.assertIsNone(self.configure({"FT_NUMAS": "1"}, multiple_caches=True))

    def test_explicit_binding_is_preserved(self):
        self.assertEqual(self.configure({"FASTLLM_MULTICUDA_WORKER_CPU_BASE": "8"}, True), "8")

    def test_explicit_numa_thread_count_is_respected(self):
        self.assertIsNone(self.configure({"FT_NUMAS": "1", "FT_THREADS": "32"}))

    def test_per_node_thread_count_is_respected(self):
        self.assertIsNone(self.configure({"FT_NUMAS": "1", "FASTLLM_NUMA_THREADS": "32"}))

    def test_disabled_llc_spread_uses_sequential_order(self):
        self.assertEqual(self.configure({"FT_NUMAS": "1", "FASTLLM_NUMAS_DISABLE_LLC_SPREAD": "1"}, True), "30")


if __name__ == "__main__":
    unittest.main()
