"""Check the shipped SM120 MegaMoE graphs without loading a CUDA runtime."""
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "tools/fastllm_pytools/kernels/mxfp4_ep"


def kernel_binaries(nodes):
    for node in nodes:
        if node["type"] == "kernel":
            yield node["binary"]
        elif node["type"] == "graph":
            yield from kernel_binaries(node["nodes"])


class Sm120EpPrefillPackagesTest(unittest.TestCase):
    def test_packages_cover_full_prefill_shapes(self):
        for ep in (4, 8):
            with self.subTest(ep=ep):
                package = PACKAGES / f"sm120_sms110_ep{ep}_h4096_i2048_e256_k8"
                manifest = json.loads((package / "manifest.json").read_text())
                buckets = manifest["buckets"]
                capacities = {bucket["capacity"] for bucket in buckets}
                self.assertTrue({256, 512, 1024, 2048} <= capacities, sorted(capacities))

                referenced = set()
                for bucket in buckets:
                    self.assertEqual(len(bucket["ranks"]), ep)
                    for rank in bucket["ranks"]:
                        local = rank["buffers"]["local"]
                        ranges = local["private_ranges"]
                        self.assertTrue({"nvlink_barrier_counter", "fc1_output_sf"} <=
                                        {region["name"] for region in ranges})
                        for region in ranges:
                            self.assertGreaterEqual(region["offset"], 0)
                            self.assertGreater(region["bytes"], 0)
                            self.assertLessEqual(region["offset"] + region["bytes"], local["bytes"])
                        binaries = set(kernel_binaries(rank["nodes"]))
                        self.assertTrue(binaries, f"Empty graph in {bucket['capacity']}-row bucket")
                        referenced.update(binaries)

                self.assertEqual(referenced, set(manifest["binaries"]))
                self.assertEqual(referenced, {path.stem for path in package.glob("*.cubin")})
                for digest, size in manifest["binaries"].items():
                    data = (package / (digest + ".cubin")).read_bytes()
                    self.assertEqual(len(data), size)
                    self.assertEqual(hashlib.sha256(data).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
