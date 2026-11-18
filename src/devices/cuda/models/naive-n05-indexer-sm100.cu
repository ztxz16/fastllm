// Blackwell Indexer: FP8 UMMA, four-stage paged TMA and two accumulator stages.
// Inspired by the UMMA/TMA and paired FP32 epilogue in NaiveSglang naive_ops_lib.
// Reads existing standalone FP8 or BF16-plus-FP8 KV rows without repacking.
// K and scales share each pipeline stage; rebound bases use cp.async fallback.
#include "naive-n05-indexer-sm100.cuh"
#include <cuda.h>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <algorithm>
#include <map>
namespace naive_indexer_sm100 {
struct TmaBatch {
    Row rows[BatchLimit];
    CUtensorMap map, scaleMap;
    const void *base;
    int rowBytes;
    bool scaled;
};
static_assert(sizeof(TmaBatch) + 128 <= 32764);
__device__ unsigned address(const void *p) { return unsigned(__cvta_generic_to_shared(p)); }
__device__ uint64_t descriptor(void *p) {
    cute::UMMA::SmemDescriptor d{};
    d.version_ = 1;
    d.layout_type_ = uint8_t(cute::UMMA::LayoutType::SWIZZLE_128B);
    d.start_address_ = address(p) >> 4;
    d.stride_byte_offset_ = 1024 >> 4;
    return d;
}
__device__ unsigned OrderedScoreBits(float value) {
    unsigned bits = value == 0.f ? 0u : __float_as_uint(value);
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}
__device__ void LoadVector(void *dst, const void *src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address(dst)), "l"(src),
                 "r"(valid ? 16 : 0)
                 : "memory");
}
__device__ void load_tma(const Row &r, int first, unsigned char *out, const CUtensorMap *map,
                         const CUtensorMap *scaleMap, void *scales, uint64_t *barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        int physical = r.key.PhysicalRow(first);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(address(barrier)),
                     "r"(r.key.scaledIndexer ? 18432 : 16384)
                     : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, "
                     "{%2, %3}], [%4];" ::"r"(address(out)),
                     "l"(map), "r"(0), "r"(physical), "r"(address(barrier))
                     : "memory");
        if (r.key.scaledIndexer)
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, "
                "{%2, %3}], [%4];" ::"r"(address(scales)),
                "l"(scaleMap), "r"(0), "r"(physical), "r"(address(barrier))
                : "memory");
    }
#endif
}

__device__ void wait_bar(uint64_t *p, int phase) {
    asm volatile(
        "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(
            address(p)),
        "r"(phase)
        : "memory");
}
__device__ void arrive_bar(uint64_t *p) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(address(p)) : "memory");
}
template <int Stages = 4, int Outputs = 2>
__global__ __launch_bounds__(256, 2) void Pipeline(const __nv_bfloat16 *weights,
                                                   const __grid_constant__ TmaBatch batch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    auto r = batch.rows[blockIdx.y];
    r.key.Bind(r.cache);
    const int begin = r.Begin() + int(blockIdx.x) * 128, end = r.Begin() + r.Count(),
              step = int(gridDim.x) * 128;
    if (begin >= end) return;
    const int tiles = (end - begin + step - 1) / step, t = threadIdx.x, warp = t / 32, lane = t % 32;
    // 128-byte swizzle repeats every 1 KiB. Align both staging areas so TMA
    // and the cp.async fallback use the same relative XOR permutation.
    __shared__ __align__(1024) unsigned char q[16 * 128];
    extern __shared__ __align__(1024) unsigned char data[];
    auto k = reinterpret_cast<unsigned char(*)[128 * 128]>(data);
    // Stage the padded scale tail too: wide KV rows otherwise cause one
    // scattered global load per key in the epilogue.
    __shared__ __align__(128) unsigned char keyScales[Stages][128][16];
    __shared__ __align__(8) float w[16];
    __shared__ uint32_t tmem;
    __shared__ uint64_t fullK[Stages], emptyK[Stages], fullC[Outputs], emptyC[Outputs];
    bool tma = r.key.base == batch.base && r.key.scaledIndexer == batch.scaled &&
               r.stride * (r.key.fp8 ? 1 : 2) == batch.rowBytes && r.key.pages && r.key.pageLen % 128 == 0 &&
               r.Begin() % 128 == 0 && r.key.offset == 0;
    if (warp == 0) {
        cute::TMEM::Allocator1Sm a;
        a.allocate(32, &tmem);
        a.release_allocation_lock();
    }
    if (t == 0) {
        for (int i = 0; i < Stages; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(address(&fullK[i])),
                         "r"(tma ? 1 : 32));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address(&emptyK[i])));
            arrive_bar(&emptyK[i]);
        }
        for (int i = 0; i < Outputs; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address(&fullC[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address(&emptyC[i])));
            arrive_bar(&emptyC[i]);
        }
    }
    for (int i = t; i < 16 * 128; i += 256) {
        int row = i / 128, col = i % 128;
        q[row * 128 + (col ^ ((row & 7) * 16))] =
            __nv_cvt_float_to_fp8(float(r.query[i]), __NV_SATFINITE, __NV_E4M3);
    }
    if (t < 16) w[t] = r.scale[t] * float(weights[r.sequence * 16 + t]);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        for (int i = 0; i < tiles; ++i) {
            int stage = i % Stages, first = begin + i * step;
            if (lane == 0) wait_bar(&emptyK[stage], (i / Stages) & 1);
            __syncwarp();
            if (tma)
                load_tma(r, first, k[stage], &batch.map, &batch.scaleMap, keyScales[stage], &fullK[stage]);
            else {
                for (int item = lane; item < 128 * 8; item += 32) {
                    int row = item / 8, col = item % 8 * 16;
                    auto src = r.key.IndexRow(min(first + row, end - 1), r.stride);
                    LoadVector(k[stage] + row * 128 + (col ^ ((row & 7) * 16)), (const char *)src.data + col,
                               first + row < end);
                    if ((item / 32) % 8 == 7) asm volatile("cp.async.commit_group;" ::: "memory");
                }
                if (r.key.scaledIndexer)
                    for (int row = lane; row < 128; row += 32) {
                        auto src = r.key.IndexRow(min(first + row, end - 1), r.stride);
                        LoadVector(keyScales[stage][row], (const char *)src.data + 128, first + row < end);
                    }
                asm volatile("cp.async.commit_group;" ::: "memory");
                asm volatile("cp.async.wait_all;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                arrive_bar(&fullK[stage]);
            }
        }
    } else if (warp == 1) {
        const uint64_t id =
            cute::UMMA::make_runtime_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, 128, 16,
                                                cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto qdesc = descriptor(q);
        for (int i = 0; i < tiles; ++i) {
            int stage = i % Stages, out = i % Outputs;
            if (lane == 0) {
                wait_bar(&fullK[stage], (i / Stages) & 1);
                wait_bar(&emptyC[out], (i / Outputs) & 1);
            }
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            auto kd = descriptor(k[stage]);
#pragma unroll
            for (int d = 0; d < 128; d += 32)
                cute::SM100_MMA_F8F6F4_SS::fma(kd + d / 16, qdesc + d / 16, tmem + out * 16, d != 0, id);
            if (lane == 0) {
                asm volatile(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(
                        address(&fullC[out]))
                    : "memory");
            }
        }
    } else if (warp >= 4) {
        const int row = t - 128;
        uint64_t ww[8];
#pragma unroll
        for (int h = 0; h < 8; ++h)
            ww[h] = reinterpret_cast<uint64_t *>(w)[h];
        for (int i = 0; i < tiles; ++i) {
            int out = i % Outputs, first = begin + i * step;

            wait_bar(&fullC[out], (i / Outputs) & 1);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            float ks = r.key.scaledIndexer ? *(float *)keyScales[i % Stages][row] : 1.f;
            uint32_t v[16];
            cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b16x::copy(
                tmem + out * 16 + ((row / 32) << 21), v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("bar.sync 1, 128;" ::: "memory");
            // Release K only after all scale reads and TMEM loads have completed.
            if (row == 0) {
                arrive_bar(&emptyC[out]);
                arrive_bar(&emptyK[i % Stages]);
            }
            uint64_t values[8];
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                uint64_t x = uint64_t(v[h * 2]) | (uint64_t(v[h * 2 + 1]) << 32),
                         abs = x & 0x7fffffff7fffffffull;
                asm volatile("add.f32x2 %0, %1, %2;" : "=l"(values[h]) : "l"(x), "l"(abs));
                asm volatile("mul.f32x2 %0, %1, %2;" : "=l"(values[h]) : "l"(values[h]), "l"(ww[h]));
            }
#pragma unroll
            for (int d = 4; d; d /= 2)
#pragma unroll
                for (int h = 0; h < d; ++h)
                    asm volatile("add.f32x2 %0, %1, %2;"
                                 : "=l"(values[h])
                                 : "l"(values[h]), "l"(values[h + d]));
            float sum =
                (__uint_as_float(uint32_t(values[0])) + __uint_as_float(uint32_t(values[0] >> 32))) * 0.5f;
            if (first + row < end) r.scores[first + row - r.Begin()] = OrderedScoreBits(sum * ks);
        }
    }
    __syncthreads();
    if (warp == 0) {
        cute::TMEM::Allocator1Sm a;
        a.free(tmem, 32);
    }
#endif
}

} // namespace naive_indexer_sm100

bool FastllmCudaNaiveIndexerSm100(const naive_indexer_sm100::Row *rows, int count, int capacity,
                                  const __nv_bfloat16 *weights, cudaStream_t stream) {
    using namespace naive_indexer_sm100;
    if (!rows || !weights || count < 32 || count > BatchLimit || capacity < 4096) return false;
    for (int i = 0; i < count; ++i) {
        const auto &r = rows[i];
        const int rowBytes = r.stride * (r.key.fp8 ? 1 : 2);
        if ((!r.key.scaledIndexer && !r.key.fp8) || r.key.fp4 || r.key.offset || !r.key.pages ||
            r.key.pageLen % 128 || !r.key.base || !r.live || !r.query || !r.scale || !r.scores ||
            rowBytes < (r.key.scaledIndexer ? 144 : 128) || rowBytes % 16 || (uintptr_t)r.key.base % 16)
            return false;
    }
    constexpr int shared = 4 * 128 * 128;
    struct Support {
        bool ok;
        int sms;
        const void *base = nullptr;
        int rowBytes = 0;
        bool scaled = false;
        CUtensorMap map{}, scaleMap{};
    };
    static thread_local std::map<int, Support> devices;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    auto entry = devices.find(device);
    if (entry == devices.end()) {
        cudaDeviceProp prop{};
        cudaFuncAttributes attr{};
        bool ok = cudaGetDeviceProperties(&prop, device) == cudaSuccess && prop.major == 10 &&
                  (prop.minor == 0 || prop.minor == 3) &&
                  cudaFuncGetAttributes(&attr, Pipeline<>) == cudaSuccess && attr.binaryVersion >= 100 &&
                  cudaFuncSetAttribute(Pipeline<>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) ==
                      cudaSuccess;
        if (!ok) cudaGetLastError();
        entry = devices.emplace(device, Support{ok, prop.multiProcessorCount}).first;
    }
    if (!entry->second.ok) return false;
    // A descriptor is immutable in a graph. If its KV base is rebound at replay,
    // the loader uses cp.async with the live binding instead of the old TMA map.
    auto &cached = entry->second;
    const int rowBytes = rows[0].stride * (rows[0].key.fp8 ? 1 : 2);
    const bool scaled = rows[0].key.scaledIndexer;
    if (cached.base != rows[0].key.base || cached.rowBytes != rowBytes || cached.scaled != scaled) {
        CUtensorMap map{};
        cuuint64_t shape[] = {128, 2147483647}, strides[] = {(cuuint64_t)rowBytes};
        cuuint32_t box[] = {128, 128}, steps[] = {1, 1};
        void *base = (char *)rows[0].key.base + rowBytes - (scaled ? 144 : 128);
        if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, shape, strides, box, steps,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
            return false;
        CUtensorMap scaleMap{};
        shape[0] = 16;
        box[0] = 16;
        if (scaled && cuTensorMapEncodeTiled(&scaleMap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (char *)base + 128,
                                             shape, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                             CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
            return false;
        cached.base = rows[0].key.base;
        cached.rowBytes = rowBytes;
        cached.scaled = scaled;
        cached.map = map;
        cached.scaleMap = scaleMap;
    }
    TmaBatch batch{};
    std::copy(rows, rows + count, batch.rows);
    batch.map = cached.map;
    batch.scaleMap = cached.scaleMap;
    batch.base = rows[0].key.base;
    batch.rowBytes = rowBytes;
    batch.scaled = scaled;
    int blocks = std::min((capacity + 127) / 128, std::max(1, entry->second.sms * 2 / count));
    int aligned = 1;
    while (aligned * 2 <= blocks)
        aligned *= 2;
    blocks = aligned;
    Pipeline<><<<dim3(blocks, count), 256, shared, stream>>>(weights, batch);
    return cudaGetLastError() == cudaSuccess;
}
