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
  indices during gather, removing a separate preparation kernel.
- `fastdiv.cuh` and `fast_mod_div_compat.cuh`: reuse the repository's vendored
  CUDA 12.9 compatibility headers from `flashinfer_glm53` (see its revision.json).

The FastLLM adapter keeps the current token BF16 and merges the old-KV split
outputs, attention sink and current-token contribution in one final kernel.
The merge accumulates in FP32 and rounds once to BF16. Split/merge reassociates
floating-point arithmetic; this is not a bitwise-equivalent replacement.
The dispatch is restricted to SM120/SM121, non-speculative independent decode
requests, 1-128 heads, D512, window128, FP8 ring and FP4 compressed cache.
`FASTLLM_DSV41_FLASHINFER=1` opts into this path; unset/0 retains native attention.
Other shapes, architectures, prefill and speculative verification retain native
attention. The CUDA unit is compiled separately for sm_120a and sm_121a, with fast math.

Initial standalone validation on SM120: 204 operator cases passed relative L2 <= 0.015 (maximum
0.003139), including empty/partial caches and supported head/request counts.
Repeated CUDA Graph execution passed memory checking. SM121 was compile-checked,
not tested on hardware. Model execution and long mailbox replay stress also pass;
strict model-logit identity is not claimed, since even repeated native execution
varies. The backend remains opt-in while broader hardware validation is pending.

Repository regression: build `cuda_dsv41_flashinfer_test` with CUDA 12.9+ and
SM120/SM121 enabled, then run `ctest -R '^cuda_dsv41_flashinfer$' --output-on-failure`.
The test compares dynamic graph replays to an independent FP64 CPU oracle,
including changed cache pointers, invalid indices, partial windows, and dispatch
rejections. It requires no Python package or model weights. The allowed relative
L2 is 0.015, with a maximum absolute error/reference-maximum limit of 0.025.

The repository CPU-oracle suite passed 204 graph cases and 12 dispatch guards
on SM120 (maximum relative L2 0.003223), with zero compute-sanitizer errors.
