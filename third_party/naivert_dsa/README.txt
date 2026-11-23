FastLLM DSA verification fusion
=============================

Source: NaiveAI-Labs/NaiveRT at 79d5704ac32ff5fa756e9fb3a76829c236444d86.
LICENSE and NOTICE retain Apache-2.0 attribution. UPSTREAM.json records original
and installed hashes. No NaiveRT runtime or checkpoint conversion is required.

Execution paths
---------------
The full-layer kernel combines the pending MLP residual, optional draft feature
export, input RMSNorm, QKV/Indexer projections, Indexer normalization, RoPE,
cache append, query quantization, index scoring, exact Top-K, sparse attention,
output projection, TP8 reduction and the attention residual. Post-attention
norm/router use the existing shared FastLLM preparation path.

For eight MX projection rows from one request, the Blackwell path can split
scoring, exact selection and sparse attention/output into three graph nodes.
It reads the interleaved BF16 Indexer cache and reuses the layer scratch.
The device, cluster, shape and output layout checks run before graph replay.

Partial fusion combines query quantization, index scoring, exact Top-K, score
ordering and sparse attention. It remains useful when the full-layer adapter
rejects the input. Rejection retains the complete native fallback; neither path
requires a runtime environment switch.

Support and ownership
---------------------
- SM103 with cooperative launch, at least 128 SMs, enough shared memory and one
  resident CTA per SM. Other hardware uses the existing operators.
- BF16 Q/K/V; eight Q heads, one KV head; QK dim 192, V dim 128; sixteen
  128-dimensional Indexer heads; E4M3 query quantization; Top-K 2048.
- BF16 or per-row-scaled FP8 Indexer cache, up to 65536 logical tokens. Native
  descriptors support independent K/V page tables and graph-time bindings.
  The default cache precision and BF16 checkpoint weights are unchanged.
- Full-layer API: 1..16 total rows, each request 1..8 rows. Partial-fusion API:
  1..31 total rows, splitting longer requests into ordered eight-row launches.
- Serving selects each path for 4..8 verification rows per request in capacity
  buckets above 8192. Smaller shapes retain the existing kernels. Separate or
  context-sharded Indexer caches and unsupported layouts retain native dispatch.
- Each request owns its causal lengths and cache binding. Scratch/output/indices
  belong to the caller and are tracked by CUDA graph pointer validation.
  Independent streams must own independent mutable buffers.
- TP8 uses the existing SWA mailbox with one ordered stream per rank group.
  Concurrent TP groups require separate collective state. Every rank agrees on
  admission before a GPU writer starts. A CUDA launch error after admission is
  reported, never silently retried through a fallback after partial writes.

Kernel implementation
---------------------
- Full-layer BF16 projections use swizzled shared inputs and vectorized global
  weight loads. QKV and Indexer weights share one projection wave: two output
  tiles and four K splits per CTA. RMSNorm is computed once per row.
- A single request packs rotated Indexer queries during publication and uses
  the frontend's existing synchronization to publish control state. Multiple
  requests retain independent packing/initialization and reuse scratch in order.
- Full-layer Indexer uses seven cache producer warps; partial fusion retains
  one. The three-stage asynchronous ring and BF16/FP8 MMA variants are retained.
  TMEM allocation permission is released only after the final request.
- Exact Top-K uses shared per-query histograms for small contexts and grid
  selection for longer contexts. Full-layer compact finishing uses bitmaps to
  emit ascending tokens; candidate overflow retains full radix-pass selection.
- When grid selection needs at most one Indexer tile per CTA, the sweep stores
  its first exact histogram in unused ring stage 1. The existing grid join
  publishes it, avoiding a repeated scan/join without adding shared storage.
  Device lengths select this path on every replay; local and multi-tile cases
  retain ordinary histogram scans.
- Full-layer attention uses eight-head QK/PV fragments and a distributed split
  combine derived from NaiveRT. Tail tiles own disjoint columns/mailbox slots;
  no terminal grid join is needed.
- The internal publication-only diagnostic field/control flow is retained to
  preserve measured SM103 code generation. Production Prepare always clears it;
  it is not a serving option or environment switch.

Numerical contracts and validation
----------------------------------
Partial fusion retains score-descending order with lowest-token ties and native
FastLLM split-attention arithmetic, BF16 rounding and attention sinks.

Full-layer projection accumulation order differs from the GEMV/cuBLAS fallback.
It consumes the exact selected set in ascending token order, with explicit dense
identities at the 2048-token boundary. Compact attention uses __expf; partial
fusion retains expf. This is an operator-tolerance contract, not bitwise equality
of full-model logits or a guarantee of end-to-end generation throughput.

CTest targets naive_n05_dsa_fused_{bf16,fp8}, naive_n05_dsa_layer_{bf16,fp8}_
{local,mixed,streams,tp}, and naive_n05_dsa_topk cover BF16/FP8, dense/sparse
boundaries, empty requests, 32K/64K, independent permuted K/V pages, multiple
requests, live-length/pointer rebinding, independent streams/graphs and TP8.
The graph cases cross both sides of the one-tile histogram-reuse boundary.

Layer tests check projections, attention on identical projected operands and
complete residuals using per-row relative L2 <= 0.006 plus 1e-6 absolute L2.
On identical projected operands, the selected Top-K set/order must match exactly.
Exported pre-attention draft features must also match exactly.
Top-K also has an independent CPU ordered-float sort oracle covering signed
scores, threshold ties and candidate overflow, with ordinary/precomputed first
histograms and repeated graph replay. Tests skip unsupported devices with 77.
