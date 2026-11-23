# Naive MoE backends and weight ownership

These optional native CUDA backends specialize the Naive FP8 model (hidden 4096,
256 routed experts, top 8, intermediate 2048, BF16 activations, 128x128 FP8 scales).
They do not replace the ordinary FastLLM TP MoE implementation.

## Selection and fallback

`FASTLLM_NAIVE_FLEX_SHARED=1` requests the shared-layout Flex backend.
`FASTLLM_NAIVE_FLEX_MOE=1` requests the older TP-only Flex implementation. If both
are set, the shared-layout backend takes precedence; two packed copies are never
created. `FASTLLM_NAIVE_MOE_FUSION=0` selects the ordinary TP implementation before
weight placement, including when EP was requested. The admission log prints both
requested and actual EP and the selected backend.

| Request and capabilities | Selected execution | Persistent routed-weight owner |
| --- | --- | --- |
| TP8/EP8, shared Flex and Mega supported | Flex for 1–8 rows; Mega for larger rows | Mega owns packed weights; Flex borrows the same addresses |
| TP8/EP1, shared Flex supported | Flex, with ordered tiles of up to 16 rows when compact block128 routing is admitted, otherwise 8 | Shared-layout Flex owns one packed representation |
| TP8/EP1, older Flex supported | Older Flex, with ordered tiles of at most 8 rows | Older Flex owns one packed representation |
| EP4/EP8, Flex unavailable but Mega supported | Mega | Blackwell packed weights, or Hopper original FP8 weights |
| Optimized backend unavailable, unsupported geometry or format | Original TP MoE, actual EP1 | Original TP shards only |

Native admission requires GPU-resident routed experts and no expert cache.
Effective per-layer CPU, NUMA or disk placement rejects native admission, including
layered offload. Without `--tp`, Flex environment flags do not change the ordinary
serial/hybrid path or its source allocator. Existing Naive thread TP itself still
prepares CUDA shards: this change does not add TP attention with CPU/NUMA experts.

Capability checks run before placement. Actual source dtype, dimensions, scale
count, ownership and placement are checked before the first pack. Routing and
activation parameters are checked again after CLI setters have run. Changing
these parameters after preparation requires reloading the model and raises a
clear error on use; it cannot reinterpret a packed buffer with a different top-k. If an EP-loaded
checkpoint has an unsupported format, one expert at a time is staged through CPU
and resharded for ordinary TP. There is no retained full CPU checkpoint copy.
Malformed TP groups, corrupt/incomplete weights, allocation failures and CUDA
execution errors remain errors. They are not unsupported-shape signals. In
particular, a partially executed collective is never retried using retired weights.
A failed packed-weight preparation also poisons that model instance: subsequent
preparation/forward attempts require a reload, even when some layers were already
consumed before the first error.

All positive row counts have an execution path using the selected representation.
TP Flex prefill uses ordered row tiles; EP uses Mega's existing prefill path. Graph admission is separate: older TP Flex captures at most 8 rows;
shared-layout TP Flex can capture the ordered tile adapter for larger batches.
Only shapes prepared during startup replay a graph; unavailable shapes retain
eager execution and are never captured during requests. No raw-weight fallback
allocation is kept for graph-ineligible inputs.

The shared-layout TP single-row path optionally uses CUDA programmatic dependent
launch (PDL): gate/up triggers the secondary launch, and down prefetches routing
IDs and weights before waiting for the producer. Down reads activations/scales
and resets gate counters only after the dependency wait. Creation checks the
actual PDL cubin and complete peer-wait grid residency. Unsupported variants use
the ordinary launch path. The same dependency protocol extends to 2–6-row tiles:
`prepare` finishes membership, activation quantization and epoch initialization;
then gate/up triggers down, which prefetches weights before waiting for the
new intermediate activations/scales. Two/three rows retain the fused down/peer
collective only when its entire grid fits the measured PDL occupancy. Four–six
rows retain the separate `down_finish` collective. Seven/eight rows keep ordinary
launches because PDL can regress disjoint expert routing. Larger requests without
compact block128 admission retain the ordered eight-row tile adapter, including
this policy for their final tile.
All variants share the existing scratch and borrowed weights. Launch errors are
reported, never retried as a second partial collective. PDL does not require
concurrent execution to make progress.

For shared-layout TP, 2–8 rows can combine the existing exact TopK selector with
`prepare`. One warp owns each row's selector scratch; CTA 0 builds membership
from its own selected IDs while the other CTAs quantize independent activation
blocks. Kernel completion publishes both products before gate/up. There is no
cross-CTA spin wait, additional allocation, or host routing readback. Exact ties,
non-finite handling, score normalization, and route scaling use the same helper
as the standalone selector. Odd-pitched merge lists avoid shared-memory bank
conflicts without changing the legacy tie order.

Admission requires the compiled prepare kernel, dense FP32 256-expert router
probabilities, compatible correction bias, and top 8. One row, EP, and unsupported
routing storage retain standalone selection followed by the existing backend.
Larger routed TP inputs use ordered tiles; the compact block128 path below
reuses expert weights across up to thirty-two rows. The batched gate epilogue uses
two independent
64-thread groups for member rows, preserving each row's split-K accumulation,
BF16 rounding, and FP8 scale/quantization order. No persistent workspace or
weight layout changes are required. All decisions are made before startup
capture or launch; this does not add request-time graph capture.

Shared-layout TP's batched gate/up can assign two adjacent split-K tasks to one
CTA. It retains all eight original FP32 partitions and the original slot-order
reduction, BF16 rounding and FP8 quantization. Completion notification changes
from eight increments of one to four increments of two. The grouped schedule
requires at least 20 unique experts and enough CTAs for all grouped tasks (20–27
experts with the current 444-CTA launch). Smaller expert sets and sets requiring
a second task group per CTA retain the original schedule; disjoint routing keeps
the plain GEMM path. The decision uses existing device membership inside the
same kernel, without host readback or another Graph variant. One–four-row tiles
therefore retain the original paths, and larger batches keep ordered row tiling.
There are no new allocations, weight copies, dependencies or environment switches.
Backend/SM admission and startup-only Graph capture remain unchanged.

## Prepared router and routed gate

For admitted SM103 shared-layout TP8/EP1, seven/eight rows use two kernels from
post-attention normalization through gate/up/SiLU. When the grouped TMA consumer
is available, the first kernel preserves the exact BF16 normalization and FP32
router accumulation, prepares FP8/UE8M0 activations, and resets the existing gate
counters/epoch. The second kernel performs selection, expert membership,
gate/up and SiLU. Each CTA selects its own routes; CTA 0 publishes metadata for
the down/TP path. The producer needs no cooperative grid barrier.

If the grouped TMA descriptor is unavailable, eight rows may instead use the
existing cooperative router: a grid barrier publishes all probabilities, then
one CTA runs exact TopK and builds membership once. Its preselected consumer
retains the original gate arithmetic. Cooperative launch, kernel image, thread
limit and full-grid residency are checked on every rank before startup capture.
Seven rows, or an unavailable cooperative producer, keep the ordinary prepared
producer and routed gate. If preparation itself is unavailable, standalone
selection/preparation and the existing backend remain available. Launch failures
are reported rather than retried after a partially executed collective.

Both prepared paths remove the separate activation `prepare` node. Four
independent 8-lane router reductions share each warp while retaining the original
addition order. The cooperative selector reuses its probability storage for
membership masks only after all warps finish reading. Router/GEMM reductions,
BF16 rounding and intermediate FP8 materialization boundaries are preserved.

The prepared descriptor borrows existing per-rank scratch and the existing model
expert-index/score buffers; it owns no memory. Its optional routing fields are
validated for shape, alignment, address overflow, finite scaling, and pairwise
non-overlap, including read-only bias. Input/output/KV aliases also reject before
writes through the existing group rendezvous. A prepared consumer is selected
only after its producer succeeds on the same ordered stream. Graph nodes copy
the descriptor by value and reuse model-owned GPU addresses, never host stack
addresses. The original preparation and all fallback paths share the down launch.

One row, 2–6 rows, EP8, other backends, and unavailable producer/consumer images
keep their existing complete execution paths. The existing 2–6-row PDL path
remains preferred. The low-level prepared ABI supports
2–8 rows; model dispatch currently uses 7/8 rows. Larger aggregate batches use
the routed tile adapter, including compact sixteen-row admission described below. This baseline preparation adds no persistent
allocation, duplicate weights, dependencies, environment switches, or serving-time
graph captures. The optional grouped consumer is described below.

### Grouped TP gate on SM103

For 7/8 rows with more than four distinct experts per row on average, the gate
groups work by expert and keeps the full K reduction in one CTA. A TMA descriptor
reads the existing row-major FP8 activation image; expert membership is applied
only when scattering results. This avoids repeated weight loads for a shared
expert and per-page scalar activation gathers. UMMA K ordering, BF16 gate/up
rounding, SiLU, routing scale and FP8/UE8M0 output materialization are unchanged.
Small row counts and more concentrated routing retain the original split-K
variants. Positive row counts above eight retain the ordered chunk adapter.

The four shared-memory panes form an asynchronous ring: a helper warp stages
weight/input/scale pages, while a consumer warp issues UMMA in the original K
order. Separate ready and release barriers prevent reuse until the previous
UMMA read completes. This adapts the helper/consumer page protocol and evict-first
weight cache policy from the pinned NaiveRT rows-eight gate (see
`third_party/naivert_flex/UPSTREAM.json`, `LICENSE`, and `NOTICE`) to the existing
MegaMoE layout and block-scaled arithmetic. It adds no allocation or dependency.
All K pages remain individually ordered; the output rounding is unchanged.

When this consumer is available, model dispatch prefers selection in the routed
gate for both seven and eight rows. The ordinary norm/router/prepare producer
then needs no cooperative grid barrier. The existing cooperative producer and
preselected consumer remain usable, including when the new descriptor is absent.
Kernel image, thread/shared-memory limits and occupancy are checked at backend
creation. If admission or descriptor encoding fails, existing kernels remain
available; no partial launch is retried after an error.

The new descriptor costs 128 bytes per rank/state and addresses the existing
activation allocation. It is initialized before capture, borrowed by each layer,
and freed once by the owning rank. No weights or activations are duplicated and
no dependency, environment switch or serving-time capture is added. EP8 and other
architectures keep their established dispatch and fallback paths.

The ordinary prepared router has an optional SM103 path for two through eight
rows. Vector BF16 normalization and FP8 preparation stage the already-rounded
BF16 values in a two-row, 16 KiB CTA-local tile. The router dot then uses the
original per-thread K order and reduction tree, reusing each FP32 router weight
across the two rows. It preserves the normalized values, router probabilities,
FP8 values/scales, counter reset and epoch increment exactly; no FP32 reduction
reordering is enabled.

This selection is admitted for the entire device group before startup capture.
Absent kernel images, unsupported SMs or insufficient thread support retain the
original prepared router. Unaligned vector operands also use that original path.
The cooperative routed producer and unprepared router keep their existing
selection. Larger model inputs keep their ordered row adapter. No persistent
GPU allocation, weight representation, dependency, environment switch or
request-time graph capture is introduced.

## Prepared multi-row down projection

The SM103 TP8/EP1 prepared/preselected path optionally uses one CTA per
128-feature output tile for 4--8 rows. For dispersed routes it consumes the
existing packed FP8/UE8M0 weights, accumulates all eight expert outputs in the
original FP32 slot order, and publishes BF16 mailbox words directly. The
following kernel receives the peer contributions in the original rank order.
The producer never waits for a peer; it does not require the producer grid to
be simultaneously resident. It reuses the existing local-output scratch and
tensor map, without another weight representation or allocation.

Concentrated routing (unique experts <= 4 * rows) retains the grouped matrix
multiplication and original partial-sum order. The new pair is admitted for the
whole device group at startup, checking SM103 kernel images, block limits,
shared memory, producer occupancy, and full receiver-grid residency. A failed
optional check uses the previous down/collective pair. Rows 1--3, ordinary
unprepared calls, EP8 and other architectures keep their existing dispatch;
larger prepared batches retain ordered <=8-row chunks. Workspace ownership and
stream ordering are unchanged. No environment switch, dependency or serving-time
Graph capture is added.

`naive_n05_owner_down` is built only with the existing SM103 target. It checks
4--8 rows on both sides of the routing threshold against the original kernels,
including changing-input Graph replay, FP32 local sums, BF16 outputs, mailbox
words/guards and counter reset. It requires no model files. Distributed peer
execution and larger batches additionally require the TP8 integration tests.

## Ownership lifecycle

1. Select the group backend and validate all sources while original payloads exist.
2. TP sources use the framework's existing reference-counted weight slabs,
   explicitly grouped by layer with a 64 MiB minimum block. EP sources retain
   their direct allocations. Retirement returns the source slab after its last
   tensor is consumed; attention weights and ordinary weights with the same layer name cannot pin it. The original allocator
   policy is unchanged for tensors without the new cudaRepackSource flag.
3. Pack a layer/rank and synchronize the conversion. Only after success, release
   its original GPU payload and CPU scales. Retain the Data objects as metadata.
4. Blackwell Mega owns its packed buffers. EP Flex binds borrowed pointers and
   per-layer tensor maps; it never repacks Mega's weights. TP Flex owns its own
   packed TP shards, which have different dimensions from EP shards.
5. Forward and Graph capture use stable packed pointers. Shared scratch/mailboxes
   are temporary activations and communication state, not weight replicas.

For the 47-MoE-layer exp18 checkpoint, packed FP8 plus expanded UE8M0 scales total
36.3515625 GiB per GPU. Packing temporarily overlaps one layer/rank's old and new
storage; a second full-model representation is not retained after preparation.
`[Naive MoE prepared]` reports remaining source payload pointer count/bytes.
Hopper source allocation also retains its previous allocator policy.
Hopper Mega directly uses the original FP8 layout, so its original weights must
remain alive; it has no Blackwell-style packed duplicate to discard.

## Architecture boundaries

| Backend | Runtime requirements | Compiled code |
| --- | --- | --- |
| Older/shared Flex | 8 distinct SM103 GPUs, 148 SMs, sufficient opt-in shared memory, full peer access | Dedicated SM103a objects |
| Blackwell Mega | 4 or 8 matching SM100/SM103 GPUs, 148 SMs, required shared memory and peer access | Dedicated SM100a and/or SM103a objects |
| Hopper Mega | 4 or 8 matching SM90 GPUs with the existing 78-SM specialization and peer access | Dedicated SM90a object |
| Other devices/configurations | Ordinary TP operator support | Existing FastLLM kernels |

Native probes also query kernel attributes to reject absent cubins. Architecture-
specific code is not assumed to run on another SM. The build only adds objects for
architectures explicitly enabled in CMake and supported by its CUDA toolchain.
Only B300/SM103 hardware is exercised by the accompanying local integration run;
compile/admission tests cover missing specializations without claiming hardware
validation on SM90, SM100, SM80, SM89 or SM120.

## Third-party code

The CMake integration uses native OBJECT targets and private include paths. It
adds no Python, PyTorch, Triton, NaiveRT or DeepGEMM runtime library. CUDA, NCCL and
CUTLASS are the existing FastLLM toolchain/dependencies. The existing fastkv target
can link CUDA::cudart; this is not a new Flex/Mega linkage requirement.

- `third_party/deep_gemm_blackwell`: pinned DeepGEMM header subset, MIT license.
- `third_party/naivert_flex`: pinned NaiveRT header subset, Apache-2.0 license and
  upstream NOTICE / third-party notices.

Each directory records its upstream repository/commit, original and installed
SHA-256 hashes, complete adaptation diff, and license/notice files. Changes to
vendored code must update those records. The older Hopper import remains intact.

## Validation and limitations

The admission policy can be tested without CUDA:

```sh
c++ -std=c++17 -Iinclude tests/naive_moe_dispatch_test.cpp -o /tmp/naive-moe-dispatch-test
/tmp/naive-moe-dispatch-test
```

Integration validation must cover both EP modes with/without a draft, source
retirement and process GPU memory, Mega-only/older-Flex paths, forced ordinary-TP
fallback, unsupported activation format after EP placement, and cross-layer Graph
replay. Compare deterministic outputs with the same backend before ownership
changes. Top-p throughput alone is not a correctness test.

The experimental W8A8/requantized execution is not bitwise equivalent to generic
FP8 computation. An existing exp18 long greedy target-only/speculative comparison
can diverge; single ownership does not resolve that numerical limitation. Keep
these experimental backends opt-in and do not claim strict lossless speculation.

## Commit scope

The runtime sources and pinned header subsets are self-contained. The early
Python/JIT benchmark plugin is archived outside the repository; it is not a build
or runtime dependency. Runtime launch helpers have internal linkage and retain
only the production <=8-row legacy Flex path; larger model inputs use its ordered
tile adapter. Tests live under `tests/naive_moe_*`, separate from vendored code.

Suggested commit boundaries are (1) pinned third-party subsets and provenance,
(2) native Mega/Flex integration, ownership, admission, build integration and its
regression tests/documentation. The second group is one coherent behavior change;
keep its mutually dependent declarations, objects and callers together.


### Dense BF16 verification

The admitted SM103 shared-layout TP8/EP1 backend also handles dense verification
FFNs with H=4096 and I=2048 per rank. It uses two kernels per tile of up to eight
rows: RMSNorm + gate/up projections + SiLU/multiply, then down projection + the
rank-ordered BF16 TP collective. Rows 1–64 are handled by ordered tiles; request
identity is independent of row tiling. Both BF16 normalization materializations,
gate/up/SiLU/multiply rounding, and the deferred residual contract are retained.
The normalized intermediate stays in shared memory. The following attention
consumes the reduced branch using the original tensor dimensions.

Selection is automatic. Capability probes run on every rank before graph
capture and require the whole peer-wait grid to fit measured residency. Host
admission agrees on row count and all local dtype, layout, pointer, output
ownership and alias checks. A failed rank sends the entire TP group through the
existing linears/activation/reduction path. EP and other geometries retain that
path. A DSA launch can omit its post norm only after collective dense admission;
its original three-argument API keeps the existing normalization behavior.

Weights are borrowed in their existing row-major allocations. Outputs use the
existing dense workspace; mailbox/epoch ownership remains with the TP state.
The same state requires exclusive, ordered per-rank streams, as the MoE path
already does. Independent concurrent pipelines require independent state.
No new environment switch, packed weights or persistent GPU workspace is added.

`tests/naive_n05_dense_fused/validate.py` checks the new operator against an
independent BF16 cuBLAS reference on all eight devices, including row tails,
multiple tiles, guards, invalid inputs, group-wide rejection, dynamic graph
replay, finite outputs and rank agreement. Relative L2 is bounded per row by
0.006 at the intermediate and final outputs; this is not a claim of bitwise
agreement with cuBLAS. The unchanged routed-MoE regression is checked separately.


### Prepared router FP32 dot reuse

On the admitted SM103 prepared 7/8-row TP path, the router kernel shares eight
FP32 weight rows across two input rows and consumes each rounded normalization
vector directly in registers. Independent normalization reductions share one
block barrier. BF16 normalization, FP8/UE8M0 preparation, counter initialization
and epoch advancement retain their existing numerical/ownership contracts.
The FP32 router dot uses a different accumulation order: its operator regression
requires per-row relative L2 and absolute probability error below 1e-5, and
checks expert IDs for the covered fixtures; it does not promise identical
routing for every possible near-tie input. The existing vector and scalar
producers remain the automatic fallback. No new environment variable, weight
copy, request state or graph node is introduced. The kernel accepts ragged
two-row tiles; tests also exercise larger row counts for future batching.


### Prepared gate with block-128 scales

The owned TP8/EP1 SM103 backend optionally uses a 384-thread warp-MMA gate for
seven/eight-row prepared chunks. All ranks probe its image, dynamic shared
memory (184448 bytes), and occupancy before enabling the path. Each owned
layer additionally needs an encodable TMA map. Failed optional admission,
borrowed weights and preselected entry points retain their existing kernels.
Unprepared seven/eight-row inputs use the same gate after separate activation
preparation; larger routed inputs can use the compact variant described below.
No environment switch is added.

At load time, gate/up weights keep their original row-major layout and are
rebased in place to a common power-of-two scale per sixteen output channels
and K128, separately for gate and up. The scale is repeated in the original
MXFP8/32 scale slots, so fallback kernels can still consume these buffers.
This rebasing can round small FP8 subnormals; fallback does not undo that
load-time precision choice. Borrowed bind() buffers are never modified.
Only one 128-byte TMA descriptor per owned layer/rank is added; there is no
second weight image or new persistent activation allocation.

Prepared normalization/router producers keep their existing BF16 and
MXFP8/32 outputs, and optionally emit a rebased FP8/K128 image into 33920 bytes
of the rank's existing gate scratch. The descriptor validates its alignment,
capacity arithmetic and disjointness. The new gate retains FastLLM top-k,
bias, normalization and route-scale semantics, BF16 gate/up rounding, weighted
SwiGLU, and the original MXFP8/32 intermediate output contract. Down and the
collective are unchanged. Four TMA pages overlap with warp MMA; adjacent
groups16 share the output quantization, and CTA work is balanced in pairs.

The workspace remains exclusive to one prepared chunk on each rank's ordered
stream. Larger batches continue to be split into chunks; there is no global
request counter or single-request allocation. Tests cover 7/8-row ragged
tiles, 8--64 unique experts, exact-rebase numerical comparison, routing and
membership, changing-input Graph replay, optional-buffer guards/admission,
and the existing fallback. The standalone regression is
`naive_n05_block128_gate_test` (SM103 only).

Validation on 2026-11-13: three real layer fixtures had a maximum gate-output
relative L2 difference of 0.000300725 under their captured routes. Across
expanded artificial routes the maximum was 0.00918627; its dominant channel
is at a BF16 rounding boundary, and an independent FP64 dot/quantization
reference agreed with the new output, not the old one. Eight-rank full-MoE
tests across 1--33 rows and changing routes had maximum relative L2 0.00268049.
These are measured fixture results, not error bounds for arbitrary data.
Memcheck and synccheck reported zero errors for the recorded gate tests.

The block-128 page pipeline and warp fragments derive from NaiveRT commit
79d5704ac32ff5fa756e9fb3a76829c236444d86, Apache-2.0. The local implementation
changes the weight layout, routing, scale conversion, paired output epilogue,
work distribution, and accumulation order; provenance is recorded in
third_party/naivert_flex/UPSTREAM.json and NOTICE.


### Compact block128 gate routing and matrix loads

The prepared block128 gate specializes its internal selection for 256 experts
and top8. It removes generic bounds/owner handling, avoids intermediate key/ID
copies, and accumulates the probability denominator in the same FP32 order
while selecting. The full FP32 ordering key is retained; signed zero is
canonicalized only for the ordering reduction. Equal winning keys or a tie at
the cutoff rerun the existing selector, preserving its legacy list ordering.
Non-finite keys/probabilities and the normalization guard keep their original
semantics. This changes neither the router producer nor the gate arithmetic,
input/output format, workspace ownership, architecture admission or fallback.
The eight selection iterations deliberately remain rolled: fully unrolling
them improved repeated hot-kernel benchmarks but regressed complete Graphs.

The weight fragment uses one `ldmatrix.m8n8.x4.b16` instruction in place of
four scalar shared-memory loads. Each eight-lane address group supplies one
tile; the resulting four packed FP8 words match the original MMA operands.
The original weight layout, TMA swizzle and arithmetic are unchanged.

The gate regression includes 196608 routing cases covering signed zeros,
negative/very small/very large scores, adjacent FP32 values, equal keys,
cutoff ties, NaN/Inf, optional correction bias, normalization and route scale.
IDs and probability bits must match the generic selector exactly. Eight-rank
MoE comparisons across 1--33 rows, changing Graph inputs and varied routing
were also bitwise identical to the preceding block128 implementation.


### Compact expert reuse across sixteen rows

Owned SM103 TP8/EP1 routing can now process a sixteen-row tile as one group.
The model accepts dense FP32 256-expert router probabilities for any positive
multi-row count; the backend tiles them without assuming a request count or
speculative window. The seven/eight-row prepared layout remains unchanged.
For unprepared seven/eight-row tiles, one kernel preserves the
BF16 -> FP8/32 -> block128 rounding and initializes the epoch, followed by
selection inside the block128 gate and the existing prepared down projection.

For nine through sixteen rows, the block128 gate builds compact membership
lists per expert. N8 matrix columns contain only selected rows; an expert with
more than eight members uses two N8 calculations against the same staged
weight pages. Intermediate outputs scatter back to the original row/top-k-slot
positions. Each CTA converts its membership bitsets in place to byte job IDs
once after routing, reusing the now-dead bias half for counts. Output tiles and
the global down-dispatch publication reuse this map instead of rescanning the
bitsets or selected IDs; shared and global storage sizes are unchanged.
Down uses the existing FP8 weight map once per expert/output tile
and gathers the corresponding intermediate rows into an N16 matrix operation.
A final kernel sums FP32 partials in the original top-k-slot order, rounds to
BF16, publishes the TP mailbox and receives peers in the original rank order.
This retains the selector, intermediate quantization and collective contract.

No GPU allocation, descriptor, duplicate weight image or environment switch
is added. The existing 8 MiB gate arena contains transient wide-tile metadata,
activations and down partials in its lower half. The upper 4 MiB hold two
sixteen-row mailbox generations and are initialized once at creation; legacy
gate partials use at most the lower 4 MiB. Static assertions bound each region.
The state workspace remains 11,720,464 bytes per GPU. Ordered per-rank streams
and exclusive state ownership are still required; separate concurrent pipelines
use separate states, as before. Row batching itself does not add a new state.

All ranks must pass the wide kernel/shared-memory/occupancy probes, and every
rank in the layer must have its owned block128 map. Otherwise the entire group
uses eight-row tiles. A rank-local decision would disagree on mailbox shapes
and is deliberately prohibited. Borrowed weights, unsupported geometry,
backend/SM admission failures and single-row tails keep their established
paths. Without the thirty-two-row admission below, larger inputs use ordered
sixteen-row tiles followed by a supported tail. This does not change EP8 dispatch
or capture a graph during serving.

`naive_n05_moe_batch_test` compares routed execution with the existing prepared
eight-row path on all eight SM103 GPUs, without a model file. Its normal,
`borrowed`, and `mixed` modes cover 1--65 rows, repeated shape transitions,
shared/disjoint/changing routes, ties, NaN/Inf, zero inputs, graph replay,
exact IDs/probabilities, output guards, finite outputs and rank agreement.
The mixed mode removes one rank's map to exercise group-wide fallback.
The acceptance threshold is per-row relative L2 < 0.003; measured fixture
results and full-model timing belong in the benchmark record, not an arbitrary
input guarantee.

### Router-produced sixteen-row block128 input

For nine through sixteen rows, post-attention RMSNorm/router can also emit the
block128 gate input, removing the standalone input-preparation launch. The
producer retains the BF16 norm rounding and FP8/32-to-block128 conversion;
padded rows are zero. The descriptor uses the existing sixteen-row region in
the gate arena; legacy FP8/32 values/scales are absent for this layout. Range
and alias checks use the descriptor's eight- or sixteen-row capacity.

This path requires wide gate/down/finish support and owned block128 maps on all
ranks of every bound MoE layer. Capability is fixed before capture; unsupported
maps, routed metadata, shapes or devices retain the old preparation path.
The prepared consumer shares wide dispatch with the unprepared path and adds
no weight cache, workspace allocation or environment variable.

`naive_n05_prepared_router` covers all nine-to-sixteen-row shapes with changing
Graph inputs and bitwise block128/norm checks. The admission test checks wide
bounds, aliases and collective capability failures.
`naive_n05_moe_batch_test prepared` compares the fused producer's contract against unprepared execution,
including ragged tiles, multiple tiles and eight/sixteen-row transitions.

### Whole thirty-two-row routed MoE

Owned SM103 TP8/EP1 layers can process seventeen through thirty-two rows in one
prepare/gate/down/collective sequence. The prepare kernel selects Top-K in one
existing CTA per row while preserving FP8/32-to-block128 rounding. Gate consumes
the compact selections, distributes at most sixteen members per expert work
item, and keeps prepared inputs in L2. This is CTA work tiling inside one kernel;
it does not run two independent sixteen-row MoE pipelines. Down uses a uniform
N16 or N32 MMA specialization according to the maximum expert membership.

One resident finish kernel reduces the eight expert slots, rounds the local
partial to BF16, performs rank-ordered reduce-scatter, rounds the result to BF16,
and allgathers it. Each source sends its tagged partial to one output owner.
Owners publish the final result to the other ranks. Slot order, rank order and
rounding points match the existing path. Generation/payload pairs use the
existing release/acquire mailbox helpers. The entire receiver grid must fit
concurrently on every GPU to avoid cross-CTA peer-wait deadlock.

The original 1 MiB down arena holds transient intermediate/route/prepared data;
the lower 4 MiB of the gate arena holds FP32 down partials. The original 2 MiB
narrow mailbox holds two generations, each with a 512 KiB scatter and 512 KiB
gather region. Its 1 MiB generation stride deliberately matches the eight-row
collective: different shapes may share a state without overwriting the other
live generation. The upper wide-mailbox region remains reserved. Workspace
stays at 11,720,464 bytes/rank, with no new allocation, weight copy, descriptor
or environment variable.

Admission is group-wide and checks the kernel images, block/shared-memory
limits, occupancy and full finish-grid residency. Failure retains sixteen-row
or eight-row tiling; borrowed weights and unsupported maps keep their existing
paths. More than thirty-two rows use ordered thirty-two-row tiles and a tail.
The prepared producer descriptor remains limited to sixteen rows. Independent
concurrent streams still require separate states.

The routed MoE regression covers 1--65 rows including 24/48-row boundaries,
finite outputs, exact routing and guards, and per-row relative L2 < 0.003 against
the eight-row reference. Its `stress` mode compares with forced sixteen-row
tiling while mixing 8/16/32-row mailbox layouts and ragged tails within captured
graphs, changing inputs/routes, and delaying a different rank at each step.
