# CUDA expert-cache building blocks

The cache metadata, record transport, and routing reductions are separate from
the model and quantization adapter. They use ordinary CUDA operations, with no
dispatch based on an SM version. Runtime launch choices use cache capacity,
record size, and device resource limits.

## Enabling the cache

Use `--moe_cuda_cache 5g` (alias `--moe-cuda-cache`) with CUDA compute and
host/NUMA experts, or call `ftllm.llm.set_moe_cuda_cache(5 << 30)` before model
loading. Zero disables the cache by default. The budget covers expert records
on each CUDA device that runs the cache;
KV cache, common weights, workspaces and cache metadata are separate. Sizes
use binary units and must fit in uint64. No cache-specific environment
variables or experimental kernel switches are required. CUDA Graph uses the
existing application setting.

For unequal free memory, `FASTLLM_MOE_CUDA_CACHE_BYTES_<device>` overrides the
budget on that logical CUDA device (after `CUDA_VISIBLE_DEVICES` remapping).
Values are unsigned decimal **bytes**, without unit suffixes; unset or empty
inherits the CLI/API budget. Zero disables that device's cache; invalid or
overflowing values disable it with a warning. The global CLI/API budget must
still enable cache preparation and hold at least 16 expert records. Device
budgets are read when each cache is first used; change them before launching
the model. For example, 12 GiB on logical GPU 0 and 2 GiB on logical GPU 1:

```sh
FASTLLM_MOE_CUDA_CACHE_BYTES_0=12884901888 \
FASTLLM_MOE_CUDA_CACHE_BYTES_1=2147483648 \
ftllm server /path/to/model --device "{'cuda:0':1,'cuda:1':1}" \
  --moe_device numa --moe_cuda_cache 2g
```

The general adapters run SwiGLU experts for one to nine tokens with FP32, FP16
or BF16 activations. The V4.1 adapter has a separate BF16 contract:

| Expert weights | Host/cache record | Compute requirements |
| --- | --- | --- |
| GGUF `Q*`, `IQ*`, F32/F16/BF16 | Original ordinary GGUF blocks, including their embedded scales | Each projection width must contain complete quantization blocks; mixed gate/down types and different layer widths are supported |
| `NVFP4_BLOCK_16_E4M3` | Packed E2M1 weights, planar E4M3 block scales, global scales | Existing compact NVFP4 adapter |
| `FP8_E4M3` | Original E4M3 bytes and FP32 block-scale arrays for gate/up and down | Hidden/intermediate widths and column scale blocks divisible by 4 |
| `FP8_E4M3_BLOCK_128` | Original interleaved 128-byte weight blocks and FP32 scales | Hidden/intermediate widths divisible by 128 |
| DeepSeek V4.1 `NVFP4_BLOCK_32_E8M0` | 16 packed E2M1 bytes and one UE8M0 scale per 32 weights | BF16 decode and 2–8-row verification; NUMA weights and widths divisible by 32 |

The generic GGUF adapter accepts Q2_0, Q4_0/Q4_1, Q5_0/Q5_1,
Q8_0/Q8_1, Q2_K–Q6_K, IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S,
IQ3_XXS/IQ3_S, IQ4_NL/IQ4_XS, and F32/F16/BF16. Gate/up and down
may have different types; layers share an LRU pool with the largest record
stride, while each layer retains its own shapes, types and down offset.
Experts within a layer must have matching projection layouts. Gate/up rows
use ordinary concatenated `[gate, up]` order. CPU R4/interleaved layouts are
not admitted as ordinary GGUF blocks.

The GGUF fast path quantizes input and SwiGLU activations to Q8_1 and uses
DP4A integer dots for Q2_0, IQ1_M, IQ2_XXS, IQ2_XS and IQ2_S projections.
Each warp computes one output row; a block shares the activation and codebook.
Both projection types must be supported, widths must contain complete blocks,
and widths above 24,576 use the floating-point fallback. A per-device workspace
is allocated before graph capture and reused across layers and batch rows.
For hidden size 2,560 and intermediate size 640 it uses 353,600 bytes, separate
from the expert-record budget. Other formats retain the register-dequantized
floating-point kernels. Selection depends on projection types, dimensions and
available workspace, without an environment override. Q8_1 activation
quantization changes numerical results and can change generated tokens.

Both paths accumulate in FP32 and round projection/SwiGLU results to the
activation type. With registered NUMA weights, GGUF also uses the existing
hybrid decode scheduler: resident experts run on CUDA and CPU workers compute
cache misses concurrently. Tensor parallel decode assigns expert ownership
across its GPUs and reduces unweighted per-expert results once. Each GPU has
its own cache budget; for example, `--moe_cuda_cache 2g` with two GPUs allows
approximately 4 GiB of expert records in total.
The low-level cache hit/miss counters describe CUDA cache lookups/refills;
hybrid CPU-only routes do not all reach those lookups. Use the expert-parallel
CPU/GPU route counters to measure the fraction of all routes served by GPUs.

Eager expert-parallel planning reads a mapped host residency mirror updated
by the GPU LRU kernel, including during graph replay. Each rank drains its
stream before reading the mirror, so prefill and ordinary cache calls may
safely change residency between decode calls. Devices without a host mapping
retain the device-lookup fallback. This removes the per-layer lookup kernel
and residency readback without changing slot ownership or eviction policy.
For single-row decode, NUMA gate/up jobs run while the caller submits CUDA
expert work; down-projection jobs retain their gate/up completion dependency.
The independent shared branch is submitted before cache admissions, so a
future-token refill does not hold up that branch. Multi-row CPU subsets keep
their existing grouped/row-wise scheduling.

The CPU adapter accepts its existing Q8_0, Q8_1, Q8_K and Q8_K32 activation
formats. Q8_0 SwiGLU quantization is fused into complete 32-value blocks;
other formats use the ordinary whole-row quantizer. CPU shard boundaries and
GEMM task boundaries must meet the corresponding block/row alignment rules.
FP32 hybrid decode and small batches reuse the FP8/NVFP4 scheduling paths;
unsupported CPU layouts or input types retain the existing fallback.

The immutable mapped host snapshot is made before destructive NUMA repacking.
It adds host RAM usage, approximately the number of experts times the largest
record stride, in addition to CPU expert storage. Pure GPU expert execution
with on-demand refills remains available through the cache operator without
NUMA registration. Decode and verification batches of up to nine rows use
the cache; larger prefill batches retain the configured CPU/NUMA backend.

## GPU-resident GGUF experts

SwiGLU decode and small batches use these fused kernels when every routed
expert in a layer is already on the same GPU. This path is enabled automatically
for ordinary GGUF tensors with matching layouts within each layer and FP32,
FP16 or BF16 activations. It does not need an expert-cache budget. Larger batches
use grouped MMQ where supported. Shared experts inside the same operator,
repacked layouts and incompatible placements retain the existing operator paths.

A layer uploads only its gate/up and down pointers: 8 KiB for 512 experts.
The kernels read route IDs and scores directly on CUDA, without copying route
IDs back to the CPU or duplicating the weights. Q8-supported formats use the
same five-kernel DP4A sequence as the cache; other supported formats use the
floating-point fused kernels. Scratch storage belongs to the invocation, and
pointer tables are invalidated when any participating tensor moves, frees its
storage or is destroyed. Warm the layer before CUDA Graph capture so its
pointer table already exists.

Cached, resident and streamed experts select their supported kernels
automatically. Different reduction orders can change generated tokens;
the optimized and fallback paths are not guaranteed to be bitwise identical.

For a Qwen4-Exp target GGUF without embedded MTP weights, attach a
`qwen4exp-mtp` GGUF using `--draft /path/to/mtp.gguf --mtp 3`.
The auxiliary file uses HF `mtp.*` tensor names, raw HF norm weights, and
packed `gate_up_proj` / `down_proj` expert tensors. Loading splits the expert
axis while retaining ordinary quantized blocks and applies the norm offset
once. Target and draft host expert tables share the same GGUF LRU pool and
expert-record budget; draft dense weights and verifier state use additional
memory. The existing Qwen4 MTP generation constraints still apply.

After warmup, read or reset counters outside timed inference:

```python
from ftllm import llm
before = llm.get_moe_cuda_cache_stats(device=0, reset=True)
# Run the measured requests here.
after = llm.get_moe_cuda_cache_stats(device=0)
print(after["hit_rate"], after["payload_bytes"], after["slots"])
```

The query synchronizes the selected CUDA device and aggregates prepared
cache groups. Reset clears hits/misses while preserving cached weights and
LRU ages. Report actual `payload_bytes`, since free-memory reservation can
reduce the requested capacity. Route counters follow the shared LRU's
convention: each unique missing expert in a row counts one refill miss;
additional duplicate routes count as hits. Invalid expert IDs are excluded.
The API raises an error on builds without GPU cache statistics.

Records remain quantized; FP8 is not expanded to BF16 in the cache. FP8 slot
pointer tables share one allocation before capture: two weight tables for
packed block128, plus two scale tables for native E4M3. The existing indexed
gate/up/down kernels read them. Cache lookup and refill produce slot IDs on the GPU, so
changing routes do not require a per-token host pointer upload. All dependent
work runs on `cudaStreamPerThread`.

Model integration is available in DeepSeek V4.1 (described below) and the
Qwen4-Exp backbone used by Qwen3.8-Flash-Next. Qwen4-Exp snapshots host experts
before deferred NUMA registration can repack or release their original storage.
Prefill and unsupported layouts
retain the configured MoE backend. CPU-only and ROCm builds do not enable this
NVIDIA CUDA adapter. A model must prepare its expert tables and call the cache
dispatch/release interfaces to use the adapter; the CLI flag alone does not
add integration to other model implementations.

Qwen4-Exp / Qwen3.8-Flash-Next also supports the cache with thread-level TP:

```sh
ftllm server /path/to/Qwen3.8-Flash-Next --tp 2 \
  --moe_device numa --moe_cuda_cache 4g
```

For supported FP32 NUMA decode and verifier batches of up to nine rows,
individual experts are assigned across the TP ranks by expert ID modulo rank
count. Rank 0 supplies one
authoritative routing decision. Each GPU computes its resident expert subset
of the same layer; rank 0 computes the remaining routes once on NUMA. The
existing TP reduction combines these partial results and the shared-expert
slices. Shared experts launch after routing copies so their GPU work can
overlap the NUMA subset. Recurring cold routes are admitted at most one per
GPU per layer call, subject to measured copy cost; cold routes still execute
on NUMA for the current call. A device with no usable cache leaves its routes
on NUMA without disabling the other devices. A verifier snapshots cache
residency for all rows before admitting cold experts, so every occurrence of
an expert has the same owner. Supported NVFP4 NUMA subsets group rows by expert
to reuse weight reads; other CPU configurations retain row-wise computation.
GPU rows use the existing expert kernels, with admissions deferred until all
resident rows have completed. Rank contributions use fused FP32 multiply-adds
for weighted accumulation, including when all routes execute on NUMA.

The byte budget applies to each device. Ordinary TP decode retains its dense
CUDA Graph segments while expert dispatch runs outside capture. Cache warmup
precedes startup KV budgeting. Larger prefill batches and unsupported EP shapes retain
the rank-zero host-expert path and its existing cache eligibility.
`--moe_device_layers` can keep some expert layers on CUDA; those resident
layers are excluded from host-cache preparation. Disabling the cache preserves
the original rank-zero host-expert execution.

Measure throughput with the cache both enabled and disabled: refill and host
routing overhead can outweigh the saved CPU work. Qwen's existing NVFP4 hybrid
path uses FP32 CUDA activations and BF16 NUMA activations. Changing the measured
CPU/GPU split can therefore change logits and greedy tokens, including between
repeated requests; cache enablement does not promise bitwise CPU equivalence.
NUMA prefill also selects CPU/GPU experts by timing when the cache is disabled.
For numerical comparisons, set the existing `FT_EXPERT_LIMIT=0` in both runs
to assign prefill experts to GPU whenever GPU assistance is supported. Use
`FT_GPU_PREFILL=0` to hold these experts on CPU. These are diagnostic settings,
not recommended throughput settings.

To distinguish backend precision from implementation errors, compare expert
operators with identical inputs and routes before comparing whole-model logits.
Check the fused gate/SiLU and down projections against independently dequantized
weights, then check weighted rank contributions and the TP sum separately.
Small activation differences can cross a top-k routing boundary and replace an
expert, amplifying the difference in later layers.

Qwen4-Exp MTP verifier batches of up to nine tokens can use the cache and become
eligible for CUDA Graph when the remaining graph requirements are satisfied.
Larger batches retain the configured MoE backend. MTP draft expert tables are
not registered with this cache and retain their separately configured placement.
TP with host experts keeps its existing MTP graph fallback; enabling the cache
does not enable a whole-backbone MTP graph.

For supported compact NVFP4 host tables assigned entirely to NUMA, preparation
copies only original E4M3/global scales, invokes the model's NUMA registration
callback, then borrows its pinned block-16 shards. Refill restores compact GPU
records without rounding. With the cache enabled, complete 32-row tiles place
packed weights before their FP32 scale plane (`NVFP4_BLOCK_16_PLANAR`), using
exactly the same storage size. CPU kernels and CUDA hybrid-prefill Linear read
this layout directly with the existing arithmetic and scratch buffers. Shards
without complete tiles keep the inline layout; gate/up and down may differ.
The cache is published only after all shards validate. Failure or an exception
releases the temporary snapshot; unsupported layouts retain a full snapshot.
Repeated preparation reuses the group, preserving graph source addresses.
Borrowed weights must remain alive until the model releases the cache.

For Qwen3.8-Flash-Next-NVFP4 this removes about 56.25 GiB of duplicate weights,
retaining about 7.03 GiB of scales. No full snapshot is allocated during loading,
though individual tensor conversions still need temporary storage. Planar
weight ranges let refill use contiguous 16-byte loads without transferring
inline FP32 scales; narrow or inline shards use 4-byte loads. Cache capacity,
LRU replacement and configuration are unchanged. NUMA warmup also returns
freed allocator pages on the final pass when all weights are already registered;
this cleanup runs during warmup, outside request execution.

For a native FP8 checkpoint, keep `--dtype auto` and use CUDA compute with
`--moe_device numa --moe_cuda_cache 5g`. The same byte budget holds fewer FP8
experts than NVFP4 experts of the same shape. Capacity and performance should
be measured per format. The mapped host snapshot also occupies system RAM in
addition to storage used by the fallback backend.

## DeepSeek V4.1

Use CUDA for the main model and shared experts, and NUMA for the NVFP4 routed
experts. A dual-GPU configuration with 8 GiB of cache on each device is:

```sh
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device "{'cuda:0':1,'cuda:1':1}" --moe_device numa --threads 30 \
  --moe_cuda_cache 8g
```

The cache borrows registered, pinned NUMA block-32 weights, without a separate
host snapshot. The shared scheduler measures CPU/GPU compute and refill costs
to choose disjoint expert subsets, preferring resident experts. Resident experts
can remain on CPU when that gives a better split. A cache is not a guaranteed
speedup: route reuse, capacity and PCIe transfers matter.

Single-token decode and single-request DSpark verification with 2–8 BF16 rows
automatically use the hybrid cache. Verify groups all rows for each expert on
one backend, preserving CPU weight reuse, and offloads only resident groups.
It admits at most one recurring cold expert after GPU computation when estimated
CPU work can cover the copy. Verify has independent per-device buffers and cost
estimates while sharing the decode cache and LRU state. Ordinary prefill, CUDA
Graph outside TP and unsupported shapes retain the configured backend.
The draft model does not use this routed-expert cache.

TP single-token decode and single-request DSpark verification with 2–8 rows
also support the hybrid cache, including segmented CUDA Graph execution:

```sh
FT_NUMAS=1 FASTLLM_CUDA_GRAPH=1 numactl -C 0-31 -m 0 \
  ftllm server /path/to/DeepSeek-V4.1-Flash \
  --tp 2 --moe_device numa --threads 30 \
  --moe_cuda_cache 8g --dspark 5
```

Layers distribute their cache ownership across TP devices. Routing copies finish
before shared-expert work launches, allowing it to overlap the CPU expert subset.
Results for every verify row use CPU staging and replica upload without requiring
peer access. Cache work stays outside graph capture and preserves captured tensor
addresses. Unsupported inputs or unprofitable splits fall back before launching
shared work, so the shared expert runs exactly once on either path.

Eager TP DSpark verification with 2–8 rows reuses one workspace across candidate
counts and requests. Per-weight activation quantization buffers stay alive
through event-ordered device dispatch, and both devices finish before the
workspace lock is released. Long prefill keeps synchronous dispatch. Set
`FASTLLM_DSV41_DISABLE_TP_VERIFY_ASYNC=1` to compare with the synchronous eager
verification path without changing CUDA Graph or expert-cache settings.

Expert math preserves V4.1 block-32 FP8 activation quantization, SwiGLU clipping,
route weighting before down-input quantization and per-expert BF16 rounding.
Results accumulate in ascending expert-ID order in FP32 before final BF16
rounding. CPU/GPU reduction can still introduce floating-point differences.
The block-32 GPU prefill GEMM also accumulates into FP32 before its BF16 cast,
preventing reduced-precision cuBLAS partial reductions. Source NVFP4 experts
are registered before their first GPU prefill use, including when startup
warmup is skipped. Keeping the default warmup moves this one-time packing
cost out of the first request; `--moe_cuda_cache 0` disables the GPU cache.

Optional diagnostics and tuning (the defaults require no environment flags):

| Variable | Meaning |
| --- | --- |
| `FASTLLM_DSV41_MOE_CACHE_MODE=gpu` | Pure GPU cache for single-token decode; misses refill from NUMA. Default `hybrid`. Verify retains its NUMA fallback in pure mode. |
| `FASTLLM_DSV41_MOE_CACHE_PREFETCH=N` | Single-token admission interval, default 31; adjusted to be coprime with the layer count. Zero disables admission for both decode and verify. Positive values enable cost-controlled verify admission each call. |
| `FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS=N` | Override the measured split for diagnostics: 0–top-k decode routes, or up to 15 resident verify groups. Unset uses the scheduler. |
| `FASTLLM_DSV41_MOE_CACHE_TRACE=1` | Per-device route, residency and admission counters every 1024 calls, for decode and verify. |

Hybrid residency is resident routes divided by all routes; GPU-selected hits
alone do not measure the whole workload. Pure mode reports actual cache hits
and misses, synchronizing at entry and at reporting intervals. Pure mode can
be slower when demand refill dominates.

## Disk-backed multilevel cache

`MergeMOE` models can retain a budgeted subset of disk-backed experts in RAM
and VRAM:

```sh
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device "{'cuda:0':1,'cuda:1':1}" --moe_device disk --threads 30 \
  --cuda_shared_expert true --moe_cuda_cache 4g --moe_cpu_cache 32g
```

`--moe_cpu_cache` sets the process-wide RAM budget; `--moe_cuda_cache` sets
VRAM **per GPU**. Both default to zero. A GPU hit executes on that GPU, a RAM
hit executes on CPU, and a miss loads the expert from disk. Cold prefill can
stream missing experts through CUDA using the existing disk GPU-prefill
setting. NUMA-cache modes and per-device environment overrides do not apply
to this backend.

Gate/up and down weights are cached together. Replacement prefers the lowest
batch-normalized frequency score, then the least recently used entry. Scores
halve every 4096 expert lookups. RAM replacement requires a second visit and
a strictly hotter candidate; CUDA promotion also requires a second visit,
admits at most one expert per layer invocation, and requires a score advantage
greater than 16 for replacement. Promotion releases the RAM copy, increasing
combined coverage; another GPU can load it again if needed.

Budgets cover retained expert payload, excluding other model weights, KV/Engram
tables, metadata and temporary buffers. Computation stages at most 256 rows
per expert and prefetches one missing expert. Cache reads default to `O_DIRECT`
where available; `FASTLLM_DISK_DIRECT_IO=0` selects buffered reads. Shrinking or
disabling a cache releases excess entries, and model unload releases its
entries. On glibc, retired buffers are periodically returned to the OS.

CUDA residency supports compact NVFP4 with block-32 UE8M0 scales, block-128
FP8 (native or packed), FP32, FP16 and BF16. Other formats use CPU execution.
`KimiK3RoutedExperts` does not use this cache. V4.1 retains its quantization,
route-weight placement and ordered reduction; CPU/CUDA matrix products can
have floating-point rounding differences.

Python callers can set budgets with `llm.set_moe_cpu_cache(bytes)` and
`llm.set_moe_cuda_cache(bytes)` before loading. `llm.get_disk_moe_cache_stats()`
returns cumulative `cpu_hits`, `cuda_hits`, `misses`, `disk_bytes`, `uploads`,
`cpu_evictions` and `cuda_evictions`, plus current `cpu_bytes` and `cuda_bytes`
(summed across GPUs). Hits count token/expert routes, including disk-backed
shared experts; GPU hits are excluded from CPU hits. Use snapshot differences
for request statistics. `disk_bytes` counts checkpoint payload read on misses,
`uploads` counts retained GPU promotions, and evictions include explicit
capacity changes and model unload.

## Metadata: `fastllm-cuda-expert-cache.cuh`

`ExpertCacheView` owns no memory. The caller supplies a global key-to-slot map,
slot keys, 64-bit last-use timestamps and a step counter. Initialize both maps
to -1 and the other arrays to zero. Hit/miss counter pointers are optional.

`EnsureExpertCache<MaxQueries>` accepts an expert-ID list and a `keyBase` for its
table. It returns slots in the original route order, plus the unique missing
expert IDs and destination slots. Missing IDs are relative to `keyBase` and
sorted by ID. Repeated IDs copy once; invalid IDs return slot -1. A request
protects all its resident experts before selecting victims. LRU ties select
the lowest slot number, so results are deterministic. Counters count valid
routes minus unique misses as hits, including duplicate routes served by one
fill.

Choose the block size once using `ExpertCacheThreads`. Small caches keep ages
in registers. Larger caches scan metadata in tiles; they do not require shared
memory proportional to cache capacity. The default query bound is 64 and can
be changed by template instantiation. The NVFP4 adapter currently uses 16.
Keys and slots are signed 32-bit IDs; timestamps are unsigned 64-bit values.

The caller provides valid map/output allocations and enough slots for the
request. All updates and dependent copies/compute for one cache must be ordered
on a stream or through graph dependencies. This is not a concurrently writable
multi-stream cache. Resetting the timestamp requires resetting its metadata.

## Record transport: `fastllm-cuda-record-copy.cuh`

`RecordCopyView` describes source/destination pointers, independent record
pitches, and bytes to copy per record. `CopyRecords` reads its count on the GPU
and flattens work over records and byte units. It makes no allocations or host
readbacks, so changing miss counts work inside a captured graph.

Aligned records use 16-byte loads/stores. Other layouts use byte copies, with
no padding or tail overread. The ordinary path uses 32-bit division; large
batches use 64-bit indexing. Records are opaque: FP32, FP16, BF16, packed
quantized values, scale arrays and auxiliary data have identical copy semantics.
Sources may be device memory or mapped pinned host memory. Source storage must
stay alive until GPU work completes, destination IDs must be distinct, and
source/destination ranges must not overlap.

Call `RecordCopyConfiguration` once per layout/device. The concurrency target
is capped by device resource capacity, without naming an architecture. Layout
alignment remains an adapter choice: the current adapters align complete records
to 128 bytes to avoid repeated misalignment at record boundaries.

## Routing: `fastllm-cuda-ordered-reduce.cuh`

`OrderedBlockReduce<N>` preserves the descending-stride FP32 reduction tree,
using warp shuffles for its last five levels. All physical block threads must
participate; at least N threads are required. N is a power of two of at least
32, and scratch storage has N+1 floats. The distinct result cell permits
consecutive reductions to reuse scratch safely.

`WarpArgMax<Items>` separates value reduction from index reduction for finite
keys, preferring the lowest index on ties. All 32 lanes participate and own
contiguous slices of the row. Callers retain their NaN handling and any legacy
tie-resolution rules. Input types can convert to FP32 before these helpers;
changing the source precision does not silently change the accumulation type.

The first integrated router optimization is the existing 512-expert/top-10
shape specialization. Its dispatch checks shape and dtype, not model name.
It keeps the original softmax arithmetic, normalization, and legacy tie path.
Other router shapes retain their existing implementations. Reusing these
helpers in a new specialization requires checking that model's reduction and
tie semantics; the helpers do not automatically optimize every shape.

## Adding a model or quantization format

1. Define a record layout and host packing, including weights, scales and any
   format-specific metadata. Register separate groups for incompatible layouts.
2. Allocate cache metadata/records and use `keyBase` to distinguish expert tables.
3. Run `EnsureExpertCache`, then `CopyRecords`, on the compute stream. No dtype
   branch is needed in either building block.
4. Supply gate/up/down compute for the new record layout and activation type.
   This is the format-specific part. `fastllm-moe-cache.cu` manages the shared
   cache/record lifecycle and compact NVFP4 compute; FP8 slot compute reuses
   the indexed kernels in `fastllm-moe-fp8.cu`. Keep a new format's scales and
   layout checks in its adapter rather than in LRU or record transport.
5. Validate route IDs, weights and full generated sequences against that model's
   reference, then benchmark its shapes and target GPUs.

## Standalone checks

The three standalone kernel tests and an adapter integration test are available
through the existing `UNIT_TEST` CMake option:

```sh
cmake -S . -B build -DUSE_CUDA=ON -DUNIT_TEST=ON
cmake --build build --target cuda_expert_cache_test cuda_ordered_reduce_test cuda_record_copy_test cuda_moe_cache_test
ctest --test-dir build -L cuda --output-on-failure
```

From the repository root, build each test with the architecture of the test GPU:

```sh
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude/devices/cuda test/basic/test_cuda_expert_cache.cu -o /tmp/cache_test
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude/devices/cuda test/basic/test_cuda_ordered_reduce.cu -o /tmp/reduce_test
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude/devices/cuda test/basic/test_cuda_record_copy.cu -o /tmp/copy_test
/tmp/cache_test
/tmp/reduce_test
/tmp/copy_test
```

The cache test compares requests and complete state with an independent CPU
LRU reference, including duplicates, invalid IDs, multiple table bases, graph
replays, large capacities, and crossing the 32-bit timestamp boundary. It also
accepts a 16-int-per-row route trace and a slot count. `--quick` limits generated
request counts for sanitizer runs.

The reduction test compares FP32/FP16/BF16 softmax results bitwise with the
original shared-memory tree across row widths and logical block sizes. It also
checks argmax ties and partial rows. The copy test checks mapped-host and device
sources, different pitches, alignment, tails, guards, graph replay, changing
counts, and both index widths.

The adapter integration test links the real runtime and checks FP32/FP16/BF16,
known expert outputs, cache eviction, graph replay, disabled/insufficient
budgets, scale validation, hidden-width validation and release. NVFP4 covers
aligned and odd widths; FP8 covers its supported layouts. Random FP8 tests
compare gate and output bytes against the original all-resident GPU backend
for FP16/BF16, including two layer tables, repeated expert IDs, changing inputs
and scores, eviction and both eager execution and graph replay.
Layered model checks exclude CUDA-resident tables, initialize per-device caches,
and verify release/reprepare when the first registered host table is not layer 0.
On two GPUs, expert-parallel tests compare summed rank outputs against independent
CPU/GPU expert references, including duplicate routes, zero/negative scores,
asymmetric cache budgets, reversed device order and collective fallback.
An all-CPU routing case checks fused accumulation against a scalar reference.
Verifier checks include transitions between one, four and nine rows and
compare grouped NUMA values with independent single-row expert outputs.

`test/benchmark/qwen4_tp_short_requests.py` exercises a real Qwen checkpoint
with 512/2040-token inputs, 1/2/8-token outputs and repeated cache/MTP switches.
It checks the TP/NUMA prefill handoff without per-operator synchronization,
which would mask premature cross-thread reuse of CUDA temporary buffers.

Shared-NUMA NVFP4 cases compare gate/output bytes with a full snapshot across
NUMA shard counts, verifier sizes, eviction and eager/graph replay, including a
2560-by-640 expert. They cover distinct gate/up scales, invalid shards,
registration exceptions, retries and reuse of an already-active cache. Planar
and mixed-layout shards use the same full-snapshot oracle. CUDA Linear tests
compare FP32/FP16/BF16 outputs bitwise with the inline layout, including bias,
odd widths and GEMV/GEMM batch sizes. `nvfp4_planar_test` checks equal storage,
packing and CPU outputs with and without AVX512-BF16 across row-tile boundaries.

Architecture compilation and execution are distinct checks: compiling these
tests for another SM does not establish its runtime correctness or performance.

GPU-resident GGUF experts also accept 1–32 input rows with CUDA routing tensors.
The batched gate/up, down and reduction kernels retain each row's single-token
expert order and rounding. Q8 activations and partial results use caller-owned
scratch sized for the actual row and route counts; weights remain packed.
`cuda_gguf_moe_cache_test` checks all 24 admitted GGUF formats against a CPU
oracle and compares batched results bitwise with individual rows, including
four-row MTP verification, invalid/duplicate routes and graph replay.
With `USE_NUMAS`, its `--hybrid` mode checks CPU subsets against serial
GEMM/quantization and bounds the combined CPU/GPU result using independent
per-expert outputs. It covers mixed formats and widths across layers,
single-device and two-device execution, and duplicate/zero/negative routes.
For larger resident prefill batches, NVIDIA SM75+ can group routes on the GPU
and reuse packed expert tiles with MMQ. Gate/up supports Q2_0 and IQ2_XXS/XS/S;
IQ1_M gate/up retains its Q8 dot fallback. Down supports Q2_0 and IQ2_XXS/XS/S.
The path accepts up to 4096 rows subject to route and workspace limits, without
a persistent dequantized weight copy or host route synchronization. Q2_0 K
tails, including 320-column TP shards, are masked before loading weights.
Other types/shapes retain the ordinary prefill fallback. Large-batch MMQ is
checked against independent CPU gate/up and down oracles: changed accumulation
order can cross a later Q8 rounding boundary. It does not promise bitwise equality
to single-row inference; the 1–32-row decode/verifier path remains unchanged.
Set `--chunked_prefill_size` above 32 to use this path (for example, 512 or
1024, subject to available activation/workspace memory).
`cuda_gguf_moe_grouped` covers uneven groups, all routes selecting one expert,
empty groups, graph replay, three activation types and actual TP shards.
`cuda_data_device_test` requires two GPUs and checks source-device CPU reads
and CUDA clones while the caller is on the other GPU. These checks protect
CPU embedding handoffs and MTP checkpoints without per-operator synchronization.

Qwen4-Exp GGUF checkpoints support resident tensor parallel experts with
`--tp 2 --moe_device cuda:0,1 --moe_cuda_cache 0`. Gate/up rows and down columns
use the same intermediate slice, aligned to the down weight's GGUF block and
the 32-value Q8 activation block. A 640-wide Q2_0 expert therefore splits into
320 columns on each rank without dequantizing or requantizing its weights.
Packed GGUF output heads split by vocabulary row. CPU token embeddings share
one immutable host table across ranks; `--low_gpu_mem` keeps that policy explicit
when CUDA Graph is enabled. CUDA embedding operators honor `lockInCPU`.

Resident eager TP2 reductions use peer access for aligned FP32/FP16/BF16
tensors up to 288 KiB. Each thread reads its packed inputs before either rank
overwrites them, so the reduction supports in-place storage and changing
temporary addresses. Captured execution, larger messages and unsupported peer
topologies retain NCCL. Topology, tensor size, alignment and capture state
determine eligibility automatically.
`cuda_gguf_tp_shards_test` checks packed byte conservation and independent CPU
expert results on both ranks, including 320/320 splitting and four-row MTP.
`cuda_tp2_reduce_test` queues changing shapes, alternating buffers, in-place
and separate outputs, and delayed rank streams without intermediate host
synchronization. It also checks collective rejection and recovery for an
unaligned destination, and dtype-rounded residual additions with changing
inputs and delayed peer streams.

The V4.1 `cuda_dsv41_moe_cache_test --dual` integration test uses an independent
FP8/BF16 dense oracle for CPU subsets, hybrid/pure decode and 2–8-row verify.
It covers all 4096 FP4 code/scale combinations, repeated and zero-weight routes,
cold admission, eviction, CPU/GPU prefill, unregistered NVFP4 cold startup,
asymmetric budgets and GPU 0/1/0 output
movement. `--dual --verify-only` limits it to multirow verification.

## Measured GPU-resident GGUF configuration

The [Qwen3.8-Flash-Next benchmark](qwen3.8-flash-next/README_EN.md#gguf-resident-performance)
records the 2026-10-01 cleanup build on two 22 GiB RTX 2080 Ti cards:
70.87 token/s decode (512 input / 512 output, chunk 32) and 1205.13 token/s
for 4096-token prefill (chunk 1024), each the median of three measured runs
after warmup. MTP is off and CUDA Graph is on. Expert weights remain on GPU
with zero expert-cache budget; embedding remains on CPU and PLE uses disk.
Per-GPU logical weights are 19.36 GiB, used memory after loading is about
20.21 GiB, and sampled peaks are 21.06 / 20.84 GiB.
[Machine-readable results](benchmarks/qwen38_flash_next_iq2xs_2080ti_20261001.json)
include configuration, timing definitions, per-run data and validation limits.

## GGUF NUMA GPU-assisted prefill

NUMA prefill retains its CPU/GPU expert scheduler. For supported 33–4096-row
batches, each GPU worker uploads its selected expert subset into reusable
scratch and calls the grouped packed-weight MMQ kernels. CPU-selected experts
continue to run on NUMA. The GPU restores IQ2_XXS/XS/S R4 and cross-SwiGLU row
layouts without modifying host weights or retaining a second model snapshot.
Q2_0 gate/up weights are also supported; down weights use Q2_0 or
IQ2_XXS/XS/S. IQ1_M retains per-expert GEMM for host prefill: the current
grouped IQ1_M gate kernel uses per-route DP4A and loses batch weight reuse.
FP32, FP16 and BF16 activations are accepted on NVIDIA SM75 or newer.

Scratch includes the current worker's uploaded and restored packed weights,
route metadata, grouped MMQ products and activations. It is reused across
layers on the same device, separately from the expert-cache payload budget.
Admission checks the entire selected subset, workspace size and available
memory before uploading weights. Shared experts inside the operator,
incompatible shapes/types, small decode batches, CUDA Graph capture and
insufficient memory retain the existing per-expert implementation.
Q8 activation quantization can change logits and generated tokens.

The grouped path quantizes each gate/up input token once and gathers packed Q8
blocks into the expert route layout. Its compact input reuses the product
buffer before matrix multiplication, so the persistent workspace size does
not grow. Gather and down-input quantization skip rows beyond the device-side
active route count while explicitly zeroing padding within active expert
tiles. Q2_0 tiles use aligned 16-bit loads and register byte permutations;
their MMA accumulation order and Q8 scale rounding remain unchanged.

The ordinary Q2_0 Dense path also accepts 1–8-row Q8/DP4A MMVQ for
FP32/FP16/BF16. It requires 64-column alignment and at most 32 KiB of staged
Q8 activation per block; other shapes retain the existing fallback. This
Dense change adds activation quantization and can change generated tokens.
The [Q2_0 TP2 benchmark](qwen3.8-flash-next/README_EN.md#gguf-q2_0-expert-mmq-and-dense-decode-optimization-2026-10-01)
records 1354.38 token/s for 4096-token prefill (+19.83%) and 70.74 token/s
decode (+0.85%) on two 22 GiB RTX 2080 Ti cards, with MTP off and Graph on.

`cuda_gguf_moe_host` checks the CPU R4 repacker against an independent decoded
weight oracle, cross/non-cross gate/up layouts, selected subset changes,
NUMA row shards, both GPUs, three activation types and immutable host storage.
