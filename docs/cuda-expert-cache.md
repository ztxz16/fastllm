# CUDA expert-cache building blocks

The cache metadata, record transport, and routing reductions are separate from
the model and quantization adapter. They use ordinary CUDA operations, with no
dispatch based on an SM version. Runtime launch choices use cache capacity,
record size, and device resource limits.

## Enabling the cache

Use `--moe_cuda_cache 5g` (alias `--moe-cuda-cache`) with CUDA compute and
host/NUMA experts, or call `ftllm.llm.set_moe_cuda_cache(5 << 30)` before model
loading. Zero disables the cache by default. The budget covers expert records;
KV cache, common weights, workspaces and cache metadata are separate. Sizes
use binary units and must fit in uint64. No cache-specific environment
variables or experimental kernel switches are required. CUDA Graph uses the
existing application setting.

The adapter runs SwiGLU experts for one to nine tokens with FP32, FP16 or BF16
activations:

| Expert weights | Host/cache record | Compute requirements |
| --- | --- | --- |
| `NVFP4_BLOCK_16_E4M3` | Packed E2M1 weights, planar E4M3 block scales, global scales | Existing compact NVFP4 adapter |
| `FP8_E4M3` | Original E4M3 bytes and FP32 block-scale arrays for gate/up and down | Hidden/intermediate widths and column scale blocks divisible by 4 |
| `FP8_E4M3_BLOCK_128` | Original interleaved 128-byte weight blocks and FP32 scales | Hidden/intermediate widths divisible by 128 |

Records remain quantized; FP8 is not expanded to BF16 in the cache. FP8 slot
pointer tables share one allocation before capture: two weight tables for
packed block128, plus two scale tables for native E4M3. The existing indexed
gate/up/down kernels read them. Cache lookup and refill produce slot IDs on the GPU, so
changing routes do not require a per-token host pointer upload. All dependent
work runs on `cudaStreamPerThread`.

The model integration is currently in the Qwen4-Exp backbone used by
Qwen3.8-Flash-Next. It snapshots host experts before deferred NUMA registration
can repack or release their original storage. Prefill and unsupported layouts
retain the configured MoE backend. CPU-only and ROCm builds do not enable this
NVIDIA CUDA adapter. A model must prepare its expert tables and call the cache
dispatch/release interfaces to use the adapter; the CLI flag alone does not
add integration to other model implementations.

Qwen4-Exp MTP verifier batches of up to nine tokens can use the cache and become
eligible for CUDA Graph when the remaining graph requirements are satisfied.
Larger batches retain the configured MoE backend. MTP draft expert tables are
not registered with this cache and retain their separately configured placement.

For supported compact NVFP4 experts assigned entirely to NUMA, preparation
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
