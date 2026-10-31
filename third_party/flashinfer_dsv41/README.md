# FlashInfer DSV4.1 mixed-KV sparse decode

Imported from the installed `flashinfer-python 0.7.2` wheel used by the standalone
2026-10-30 study. Upstream: https://github.com/flashinfer-ai/flashinfer .
`upstream-sha256.json` identifies every original included header. Existing Apache-2.0
and BSD-3-Clause notices are retained. This directory is private to the SM120 object.

Local adaptations:
- `decode.cuh`: optional device pointer-to-pointer parameters and a request stride let an existing
  CUDA Graph consume the current request's ring/compressed cache addresses. No
  attention arithmetic, precision, tile, or reduction is changed. The optional
  request-index specialization derives the ring window and masks compressed
  indices during gather, removing a separate preparation kernel. Verification
  uses the same arithmetic with causal ring/compressed-index masks: uncommitted
  rows are excluded from the ring and future compressed rows are invalidated.
- `fastdiv.cuh` and `fast_mod_div_compat.cuh`: reuse the repository's vendored
  CUDA 12.9 compatibility headers from `flashinfer_glm53` (see its revision.json).

The FastLLM adapter keeps the current token BF16 and merges the old-KV split
outputs, attention sink and current-token contribution in one final kernel.
The merge accumulates in FP32 and rounds once to BF16. Split/merge reassociates
floating-point arithmetic; this is not a bitwise-equivalent replacement.
For verification, a separate FP32 merge includes every BF16 row in the query's
causal chunk prefix. Ring writes stay deferred until acceptance. The full-decode
request validator requires consecutive positions and shared cache pointers;
these are preconditions of the low-level verification adapter as well.
The dispatch supports SM120/SM121, 1-128 independent decode requests or a
contiguous verification chunk of 1-128 tokens, 1-128 heads, D512, window128,
FP8 ring and FP4 compressed cache. It is not tied to a DSpark draft length.
Compressed top-k width is 1-1024 and compression ratio is 0, 1 or 2.
`FASTLLM_DSV41_FLASHINFER` defaults on for eligible inputs; `=0` disables it
and `=1` explicitly enables it. Other explicit values are treated as disabled.
Other shapes, architectures and long prefill chunks retain native attention. The CUDA unit is compiled separately for sm_120a and sm_121a, with fast math.

Repository regression: build `cuda_dsv41_flashinfer_test` with CUDA 12.9+ and
SM120/SM121 enabled, then run `ctest -R '^cuda_dsv41_flashinfer$' --output-on-failure`.
The test compares dynamic graph replays to an independent FP64 CPU oracle,
including changed cache pointers, invalid indices, partial windows, and dispatch
rejections. It requires no Python package or model weights. The allowed relative
L2 is 0.015, with a maximum absolute error/reference-maximum limit of 0.025.

The ordinary sparse-attention entry uses the same packed FlashInfer kernel and
BF16 causal-prefix merge, with a small GPU index preparation pass. It supports
dense [batch, sequence] inputs with 1-128 total query rows, 1-128 heads and a
sequence length up to the 128-token window. It preserves the native operator's
precondition that compressed indices have already been causally filtered by the
indexer; the gather additionally checks the allocated compressed-cache bounds.
Temporary workspace comes from FastLLM's existing stream/graph-aware pool.
Unsupported layouts, explicit reference/legacy math, unavailable kernel setup,
or optional workspace exhaustion retain the native implementation. The existing
FlashInfer environment flag controls both ordinary and full-plan paths.

Validation on SM120: 204 independent-decode, 520 plan-verification and 432
ordinary-entry cases passed the independent FP64 CPU oracle; ordinary cases
each replay their graph three times. Maximum relative L2 was 0.003223, 0.002750
and 0.002751 respectively, with zero compute-sanitizer errors. Coverage includes
chunk lengths 1-128, padded head groups, empty/partial/wrapped windows, changing
cache pointers, compression boundaries, future/invalid indices and two separate
sequences. Environment-policy checks and 26 dispatch guards cover rejection.
SM121 is compile-checked only. Model execution and mailbox replay stress pass;
bitwise model-logit identity is not claimed. Unsupported devices and layouts
retain the native implementation.
