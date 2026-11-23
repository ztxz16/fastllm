Target verification commit validation (CUDA, eight GPUs)

Set CUDA_HOME to the toolkit used to build fastllm_tools, then compile this
standalone executable from the repo root:

g++ -O2 -std=c++17 -DUSE_CUDA tests/naive_n05_commit_handoff/commit.cpp \
  -Iinclude -Ithird_party/json11 -I$CUDA_HOME/include \
  -Lbuild-fastllm/tools/ftllm -Wl,-rpath,$PWD/build-fastllm/tools/ftllm \
  -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64 \
  -lfastllm_tools -lcudart -pthread -o /tmp/naive_commit_test
/tmp/naive_commit_test

The independent CPU oracle compares the entire K/V allocation, including staging
and guard rows, byte for byte. The original per-layer kernel is also compared.
Coverage: BF16/FP8, different K/V widths, accepted counts 0..8, ring wrap and long
positions, and descriptor counts 1/39/78/111/112/113/117/156/224/225/312. Counts
around 112 exercise the batch launch boundary. Every check runs on all eight GPUs.
The printed microbenchmark is kernel/submission timing, not end-to-end decoding.

tests/naive_n05_draft_fused/head.py also validates batched asynchronous export:
independent request inputs, seeds, anchors and temperatures; pinned host outputs;
one completion wait per batch; exact q/tokens/confidence equality to the retained
synchronous Finish API. Its independent numerical reference retains L2 checks.

Scheduler validation must cover batch sizes 1..8, mixed accepted lengths, shrinking,
cancellation, arrivals, capacity changes, prefix restore, configuration fallbacks
and tails 1..8. Check live request identities and physical bindings after reuse.
No test-only environment switch is needed in the production path. Capability
mismatches return to the existing path before submitting any mutation; an error
after a joined commit has started drains the producer streams and propagates.

For asynchronous next-backbone execution, include five simultaneous requests
with output budgets 16/24/32/40/48. This exercises a normal subgroup followed by
a single-token tail before the prefetched result is consumed. A tail/prefill
without the proposal's per-rank dependency must join lookahead first: independent
multi-GPU resident collectives can deadlock even when their buffers differ.
Only acceptance of the exact group that consumed lookahead may retire its
in-flight ownership without another host wait. Cancellation and cache recycling
must retain request owners until every prefetched reader has completed.

Verification page preflight regression

page_checks.cpp is a diagnostic LD_PRELOAD interposer for real model requests.
Build it with the same headers and CUDA runtime as fastllm_tools:

g++ -O2 -std=c++17 -shared -fPIC -DUSE_CUDA \
  tests/naive_n05_commit_handoff/page_checks.cpp \
  -Isrc/models -Iinclude -Iinclude/models -Iinclude/utils -Iinclude/blocks \
  -Iinclude/devices/cuda -Iinclude/devices/multicuda -Ithird_party/json11 \
  -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64 \
  -lcudart -ldl -pthread -o /tmp/naive_page_checks.so

Place this .so first in LD_PRELOAD, before any other model interposers and the
tested fastllm_tools library. After model warmup, enable it through ctypes:
  audit = ctypes.CDLL(None)
  audit.fastllm_page_checks_enable(1)
Run real speculative requests with batches 1..8, including a 512-token request
that crosses physical page boundaries, arrivals/cancellation and prefix restore.
fastllm_page_checks_count(batch, 0) returns the number of checked preparations;
fastllm_page_checks_count(batch, 1) must return one completed forced COW per batch.
Both counters return unsigned long long; set ctypes.restype accordingly.
Disable the audit before releasing the model. Never use this interposer for
performance or memory measurements.

The test retains a real partial tail page without changing the binding epoch.
It verifies physical capacity and private writable tails after each successful
preparation. Forced COW compares the entire copied KV page, the optional indexer
sidecar and the uploaded GPU page-table entry against independent snapshots.
Different batch sizes select different ranks, K/V parts and request rows. Test
snapshots use host memory; the production path allocates no additional tensor.
Ordinary small owned batches may preflight on the caller; page growth/COW,
external callers and workloads above the inline-check budget keep rank workers.
