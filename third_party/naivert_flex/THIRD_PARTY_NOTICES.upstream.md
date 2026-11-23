# Third-party notices

NaiveRT vendors the following source trees unchanged (each directory carries its own LICENSE file and
a manifest of the pinned upstream identity where the vendor tool records one):

| directory | upstream | version | license |
|---|---|---|---|
| `naivert/prefill/kernels/flashinfer_comm`, `naivert/prefill/kernels/flashinfer_rope` | FlashInfer (mixed communication, RoPE) | 0.6.12 | Apache License 2.0 |
| `naivert/prefill/kernels/deepep` | DeepEP (DeepSeek) | 1.2.1 | MIT License |
| `naivert/kernels/naive_n0p5/prefill_sglang` | SGLang kernels (expert scatter) | pinned in the vendor manifest | Apache License 2.0 |
| `naivert/kernels/naive_n0p5/prefill_ext/naive_ops` | SGLang / NaiveOpsLib prefill kernels (indexer logits, top-k, sparse attention) | pinned in `prefill_ext/manifest.json` | Apache License 2.0 |
| `naivert/kernels/naive_n0p5/prefill_ext/fa4_cute` | FlashAttention-4 CuTe DSL sources (via SGLang) | pinned in `prefill_ext/manifest.json` | BSD 3-Clause License |

Build-time dependencies not vendored: NVIDIA CUTLASS v4.7.1 headers (BSD 3-Clause), DeepGEMM
(`sgl-deep-gemm`, MIT), nvidia-cutlass-dsl (NVIDIA license), PyTorch (BSD-style), FlashInfer Python
package (Apache 2.0). `tools/vendor_prefill_*.py --check` reproduces each vendored tree from its pinned
upstream and compares every file.
