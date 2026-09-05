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

The adapter currently runs single-token SwiGLU experts in compact NVFP4 format.
Prefill and unsupported layouts retain the configured MoE backend. CPU-only
and ROCm builds do not enable this NVIDIA CUDA adapter.

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
alignment remains an adapter choice: the NVFP4 adapter aligns complete records
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
   This is the format-specific part; the current adapter decodes NVFP4 records.
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
aligned and odd matrix dimensions, known expert outputs, cache eviction, graph
replay, disabled/insufficient budgets, hidden-width validation and release.

Architecture compilation and execution are distinct checks: compiling these
tests for another SM does not establish its runtime correctness or performance.
