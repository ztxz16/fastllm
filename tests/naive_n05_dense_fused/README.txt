BF16 dense verification reference test

Requires eight peer-accessible B300 GPUs, a build with FASTLLM_ENABLE_NAIVE_FLEX_SM103,
PyTorch and the CUDA runtime on the library path. Example:
  python3 tests/naive_n05_dense_fused/validate.py build-fastllm/tools/ftllm/libfastllm_tools.so

The test uses identical deterministic inputs and independent per-rank weights.
The reference is RMSNorm with both BF16 boundaries, BF16 gate/up cuBLAS GEMMs,
BF16 SiLU and multiply, BF16 down GEMM, then fixed-rank-order FP32 sum -> BF16.
Threshold: maximum per-row relative L2 < 0.006 for intermediate and final output.
Cases: rows 1..8,9,13,16,31,33,64; amplitudes 0,.01,1,10; six dynamic CUDA Graph
cases with three changing-input replays. Guards, finiteness and exact rank agreement
are required. Invalid pointers, aliases, epsilon and row counts must reject before
writing. The host admission test makes one rank reject or disagree on row count,
then verifies that the next valid group can still proceed.

The C ABI run function assumes its caller has completed the all-rank admission
and uses the state's exclusive per-rank ordered streams. This test deliberately
validates the admission and numerical launch interfaces separately.
