Fused draft operator validation (8 x SM90 or SM103, torch, safetensors)

Build the normal fastllm_tools target, then run each script separately, with no
other GPU workload. Inputs use the actual checkpoint weights and an independent
FP32 matmul reference with the original BF16 intermediate rounding boundaries.
The original reference tests require relative L2 <= 0.006 on every row, not just
the batch average. The expanded batch attention suite records and bounds known
scalar-reference outliers as described below.
Attention includes the original BF16 probability materialization in the reference.

Example:
python tests/naive_n05_draft_fused/backbone.py --library /abs/libfastllm_tools.so \
  --draft /abs/draft/model.safetensors --rows 7 --bench --out backbone.json
python tests/naive_n05_draft_fused/context.py --library /abs/libfastllm_tools.so \
  --draft /abs/draft/model.safetensors --rows 7 --out context.json
python tests/naive_n05_draft_fused/head.py --library /abs/libfastllm_tools.so \
  --draft /abs/draft/model.safetensors --target-head /abs/target/head-shard.safetensors \
  --rows 7 --layers 1 --out head.json
python tests/naive_n05_draft_fused/normalized.py --library /abs/libfastllm_tools.so \
  --draft /abs/draft/model.safetensors --rows 7 --out normalized.json

Also run backbone with --rows 2, 4 and 8. Context checks accepted lengths 1/3/7/8,
positions through 65529, retained KV and guard rows, and ragged 2/3/4-request pointer
tables. Backbone and head exercise 1/2/4-request CUDA graph replay domains.
Head checks the BF16 corrected-logit boundary exactly, full q softmax L2 < 2e-6,
confidence probabilities and deterministic replay of an explicitly supplied seed.
Sampling law is Gumbel-max(logits / temperature); the identical unperturbed logits
are exported as q for the existing rejection sampler. Different draws from the
legacy inverse-CDF sampler are expected; token equality is not an operator test.

These tests do not replace actual scheduler lifecycle checks (request arrival,
cancellation, shrinking batches, unsupported configuration fallback). No default
or global environment switch selects the fusion: capability guards keep legacy
paths for unsupported hardware, geometry, sampling constraints or memory budget.

The head test additionally submits 1/2/4 independent requests into pinned host
outputs, waits once for the batch, and checks q/tokens/confidence bit for bit
against the synchronous Finish API. Submit retains the same arithmetic; callers
must keep all destination storage live until their owner stream completes.

Prepared context checks compare the shared rank-major descriptor packet against
eager output and an independent reference (19 geometries, each with host and
device accepted counts), including unsupported
batch-size admission. History, guard rows and padding remain bitwise equal;
new KV must independently satisfy the same per-row L2 limit. The report also
records bitwise agreement with eager and the maximum eager/prepared difference.
Prepared context stages input directly into CTA shared memory and computes RoPE
in the consumer. FC's existing grid join retires compact-KV trimming before KV
output, avoiding an extra initial grid barrier. Its SSQ fold uses a warp reduction.
The separate eager kernels remain the unsupported-plan fallback.
The shared packet belongs to one serialized execution domain: every rank must
finish reading it before the next upload or destruction.
Device-count checks delay the producer on another stream, pad features to eight
rows, and publish distinct accepted lengths per request. They verify that the
context consumer waits for its producer and preserves retained KV and guards.
The serving path reuses this packet and the sampler's existing result allocation;
CPU result export may overlap context submission. Accepted counts stay valid
until all rank consumers finish, including cancellation and exception cleanup.
Mixed groups and unsupported plans retain the ordinary host commit path.

The cuda_mtp_rejection CTest covers exact ragged accepted prefixes for batch
1/2/4/8 and the batch 9/16 fallback, with 1/3/7/8 proposals per request. It checks
device consumer counts against host results and retains the existing graph
replay, changing sampling parameters, filtering and distribution-law tests.

Normalized writeback checks 1/2/4/8 requests against the retained raw body and
independent FP32 RMSNorm, including the BF16 boundary before gamma multiplication.
Run it with rows 2, 4, 7 and 8. Graph replay checks exact deterministic results
and audits programmatic dependency edges (or ordinary edges on unsupported
runtimes). Position metadata is consumed only after the programmatic wait;
weight prefetch may precede it. Each request retains its own epoch and workspace.

Composite verification checks use a child target graph and the same head/input
prefix as serving. Batches 1/2 cover every candidate length 1..7; batches 3..8
also exercise short prefixes and odd request tails at eight changing positions.
Tokens, full-vocabulary q, optional confidence, packed IDs/positions/lengths and
guard words match the full seven-step reference prefix exactly. Unused export
rows and disabled confidence outputs must remain untouched. The export owner runs on a
separate stream and waits for the graph's head-ready event. The child graph runs
five fused backbones against separate scratch KV/hidden buffers, exercising the
early completion boundary while other resident GPU work is still running. Eight
distinct input/seed cases prevent accidental cross-request reuse from passing.
The head joins mbarrier initialization across all producer/consumer warps before
the first TMA transaction. The existing full-grid residency check and unsupported
device fallback remain in place.

Runtime candidate length no longer requires a seven-candidate verification
graph or next-body prefetch. Each model prepares one configured full-budget
composite per request count; short request tails retain the ordinary fallback.
The current checkpoint still computes all seven body rows. Head projection
retains its eight-row MMA tile, while Markov sampling and probability/confidence
exports stop at the requested prefix; threshold zero skips confidence entirely.
Composite and ordinary verification share one graph executable: ordinary replay
disables only the head/input prefix, retaining the target child. Tests alternate
these modes and check that disabled prefix nodes leave all input guards intact.
All-rank plan construction completes before replacing the ordinary executable;
an unsupported plan retains the original path.

Batched backbone tiles (SM103)

RunBody handles arbitrary prepared request counts (1..64) as bounded row tiles;
odd tails retain the scalar kernel. MLP shares packed gate/up/down weights across
four or two independent eight-row lanes. Four-request attention shares QKV/O
weights and uses four 32-CTA subgrids, with eight KV splits per request. The
paired fallback uses two 64-CTA subgrids with 16 splits per request. Each lane
retains its KV pointer table, counters, epoch and mailbox. Attention and MLP
capabilities are checked across all participating ranks before capture; the
body uses four requests when MLP admits them and independently falls back for
attention. Windows beyond the admitted split capacity retain the paired or
original 128-CTA kernel. No GPU allocation, new
weight copy or environment variable is introduced by this path.

python tests/naive_n05_draft_fused/mlp_batch.py --library /abs/libfastllm_tools.so \
  --draft /abs/draft/model.safetensors --rows 7 --out mlp-batch.json
python tests/naive_n05_draft_fused/attention_batch.py --library /abs/libfastllm_tools.so \
  --draft /abs/draft/model.safetensors --rows 7 --out attention-batch.json

Both test 1/2/3/4/8/16 requests, independent lane inputs, output guards, graph
replay and changed inputs; attention also changes positions and checks retained
KV, speculative KV and padding. Repeat for rows 2/4/8. Use --requests 17,32,63,64
--layers 1 for larger tile boundaries. Attention --window 8192 tests the largest
admitted split geometry; --window 8193 (or 16384) tests the scalar fallback.

MLP retains the independent FP32 matmul reference and per-row L2 <= 0.006.
Attention additionally compares every batched row against the retained scalar
path (per-row L2 <= 0.006). Expanded random inputs expose existing scalar-path
outliers against the fully independent reference because intermediate BF16
rounding changes softmax-sensitive rows. Reports retain both errors; a batched
reference row must meet 0.006 or stay within 0.0001 of that same scalar row's
reference error. The original backbone reference tests remain unchanged.
Normalized tests audit the reduced number of programmatic edges and admit the
independent attention/MLP fallback combinations.

The batched MLP normalizes each padded row once, using the eventual output
buffer until the existing post-SiLU grid join retires every GU reader. Input and
output are distinct prepared lane buffers. The first eight bits of each lane's
existing arrival word publish normalized rows; its high bits keep the grid
arrival count. The first lane is cleared by that grid join, and the other lanes
are cleared before kernel completion. Consumers acquire readiness and bypass L1
for these cross-CTA inputs. Helper warps continue weight DMA during normalization.
This removes duplicated per-CTA normalization and permits two 2048-K weight pages
without added global storage. All lanes publish their TP partials before any
lane waits for results; each retains the original mailbox and rank summation.
Scalar admission, odd tails, request isolation and graph replay remain covered
by mlp_batch.py and normalized.py, including count changes on the same state.


Four-request and paired head/context tiles

Composite head shares one LM-head weight pass across four or two requests and
advances independent Markov chains in separate warp groups, retaining each
request's RNG, mailbox, epoch and asynchronous q export. Before a weight pane
can be overwritten, all head projection readers join the producer. The composite
checks above compare 1..8 requests exactly against separate scalar head runs.
Prepared context shares FC weights across four or two requests; its KV projection and
mailbox domains stay per-request. Context tests additionally compare two alias
cases (shared cache with full-window trimming, and a later input inside the
first request's cache) bit for bit against sequential prepared launches. Aliased
requests execute sequentially inside the tile.
Both entry points retain scalar tails and capability fallback; no extra global
workspace or persistent weight copy is required. Use --bench on head.py or
context.py for the actual composite entry timing. Those event measurements also
include submission skew and collective waiting; use NSYS for kernel attribution.

True batches of eight or more requests

Serving uses the existing batched body GEMMs and attention for wide batches,
reusing one layer of scratch. The small resident body remains unchanged.
Composite head uses one dense projection for all requests, parallel vocabulary
scans and independent Markov chains. Prepared context uses one FC GEMM and a
five-matrix KV batched GEMM, borrowing the existing partial allocation for BLAS
workspace. Contiguous slabs replace equivalent per-lane payloads, without new
weight copies. Requests with overlapping context input/cache buffers use the
ordered existing path. States grown after initial preparation keep the older
path beyond their contiguous capacity; serving prepares max_batch first.

python tests/naive_n05_draft_fused/wide_head.py --library /abs/libfastllm_tools.so \
  --draft /abs/draft/model.safetensors --target-head /abs/target/head-shard.safetensors \
  --rows 7 --requests 8,15,16,31,32,63,64 --bench --out wide-head.json
python tests/naive_n05_draft_fused/context.py --library /abs/libfastllm_tools.so \
  --draft /abs/draft/model.safetensors --rows 7 --requests 8,15,16,31,32,63,64 \
  --bench --out context-wide.json

Wide-head checks per-row head L2 < 0.006, exact corrected-logit BF16 boundaries,
independent Philox/Gumbel sampling, short DS1/DS3 graph prefixes, packed guards
and disabled graph prefixes. Batched probability output must be bitwise equal
to the retained exporter; its separate Torch softmax comparison allows 5 ppm
for FP32 exp/reduction order. This does not change the original head test.
Ragged probability prefixes, temperatures, seeds and confidence are covered.
Wide context retains the original 0.006 per-row bound, exact history/guards and
ordered alias checks, including distinct device-published accepted lengths.

The native body fixture tests complete five-layer TP8 output, immutable context
replay, batch growth and shrinkage against both independent Torch and retained
scalar paths. Chained BF16 error is reported separately from operator L2.
Generate and run it as follows:
python tests/naive_n05_draft_fused/body_fixture.py --draft /abs/draft/model.safetensors \
  --out /abs/body-fixture
build-fastllm/naive_n05_draft_test /abs/body-fixture batch

The wide body collective borrows the context partial allocation at disjoint
phase boundaries, avoiding the additional NCCL runtime arena. Eight independent
rank slices use rank-ordered FP32 sums; publication joins precede reads and a
retirement join completes before scratch reuse. No target communication state
is shared. The configured max_batch is reserved on first use, even when a
single-request bootstrap precedes serving preparation.
Run reduce.py with --rows 2/4/7/8 (separately) and --requests 64. It checks exact
FP32 rank-ordered sums, output guards, invalid shapes, graph replay and changed
inputs at batch 8/15/16/31/32/63/64, followed by shrinkage to eight. wide_head.py
also interleaves this collective with head replay on the same scratch/epochs.


The cuBLAS handle exists only while constructing projection graph nodes. Replay
uses state-owned matrices and borrowed workspace; no persistent BLAS handle arena
is retained. Head/context tests overwrite newly reclaimed device allocations
after graph construction, replay changed inputs, and verify both operator output
and the complete allocation canaries before graph destruction.
