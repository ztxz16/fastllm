#include "devices/cuda/fastllm-cuda-decode-dense.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "../low_latency/data_layout.h"
#include "devices/cuda/low_latency/execution.h"
#include "utils.h"
#include <cuda.h>
#include <cuda_bf16.h>

namespace fastllm {
namespace {
using Bf16 = __nv_bfloat16;
constexpr int rows = 16, hidden = 4096, intermediate = 2048;
constexpr int tileK = 64, features = 32, stages = 4;
constexpr int stageElements = (2 * features + rows) * tileK;
constexpr int sharedBytes = 2 * stages * stageElements * sizeof(Bf16);

struct GateUpMaps {
    CUtensorMap input, gate, up;
};

bool EncodeMap(CUtensorMap &map, void *data, int height, int tileRows) {
    cuuint64_t dims[] = {hidden, static_cast<cuuint64_t>(height)};
    cuuint64_t stride[] = {hidden * sizeof(Bf16)};
    cuuint32_t box[] = {tileK, static_cast<cuuint32_t>(tileRows)}, steps[] = {1, 1};
    return cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, data, dims, stride,
        box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

__device__ __forceinline__ unsigned SharedAddress(const void *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void LoadTile(Bf16 *dst, const CUtensorMap *map, int k, int row,
                                        uint64_t *barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2,%3}], [%4];" ::
                 "r"(SharedAddress(dst)), "l"(map), "r"(k), "r"(row), "r"(SharedAddress(barrier)) : "memory");
}

__device__ __forceinline__ uint64_t MatrixDescriptor(const Bf16 *ptr) {
    // 128-byte swizzle, 1024 bytes between adjacent groups of eight rows.
    return ((SharedAddress(ptr) >> 4) & 0x3fffull) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void Accumulate(float (&value)[8], uint64_t a, uint64_t b) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7},%8,%9,1,1,1,0,0;"
                 : "+f"(value[0]), "+f"(value[1]), "+f"(value[2]), "+f"(value[3]),
                   "+f"(value[4]), "+f"(value[5]), "+f"(value[6]), "+f"(value[7]) : "l"(a), "l"(b));
}

// Each warpgroup accumulates one contiguous half of K, matching the two FP32
// split-K partials of the SM90 verification projection. Its M tile holds 32
// gate and 32 up features, sharing the same 16 input rows.
__global__ __launch_bounds__(256) void VerifyDenseGateUp16(const __grid_constant__ GateUpMaps maps, Bf16 *output, int activeRows) {
    extern __shared__ __align__(1024) Bf16 storage[];
    __shared__ uint64_t barriers[2][stages];
    __shared__ float partial[2][rows][2 * features];
    const int part = threadIdx.x / 128, thread = threadIdx.x % 128;
    constexpr int tilesPerPart = hidden / (2 * tileK);
    auto *ring = storage + part * stages * stageElements;
    if (thread == 0) {
        for (int i = 0; i < stages; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(SharedAddress(&barriers[part][i])) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto prefetch = [&](int tile) {
        const int slot = tile % stages, k = (part * tilesPerPart + tile) * tileK;
        auto *dst = ring + slot * stageElements;
        auto *barrier = &barriers[part][slot];
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::
                     "r"(SharedAddress(barrier)), "r"(int(stageElements * sizeof(Bf16))) : "memory");
        LoadTile(dst, &maps.gate, k, blockIdx.x * features, barrier);
        LoadTile(dst + features * tileK, &maps.up, k, blockIdx.x * features, barrier);
        LoadTile(dst + 2 * features * tileK, &maps.input, k, blockIdx.y * rows, barrier);
    };
    if (thread == 0)
        for (int i = 0; i < stages; ++i) prefetch(i);
    float accum[8] = {};
    for (int i = 0; i < tilesPerPart; ++i) {
        const int slot = i % stages;
        asm volatile("{ .reg .pred p; wait_tile: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;"
                     " @!p bra wait_tile; }" :: "r"(SharedAddress(&barriers[part][slot])),
                     "r"((i / stages) & 1) : "memory");
        auto *weight = ring + slot * stageElements;
        auto *input = weight + 2 * features * tileK;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
        for (int k = 0; k < tileK; k += 16)
            Accumulate(accum, MatrixDescriptor(weight + k), MatrixDescriptor(input + k));
        asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
        // All four warps must retire their reads before the producer refills the slot.
        asm volatile("bar.sync %0,128;" :: "r"(part + 1) : "memory");
        if (thread == 0 && i + stages < tilesPerPart) prefetch(i + stages);
    }
    const int lane = thread % 32, warp = thread / 32;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int col = warp * 16 + lane / 4 + (i / 2 % 2) * 8;
        const int row = (i / 4) * 8 + (lane % 4) * 2 + i % 2;
        partial[part][row][col] = accum[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < min(rows, activeRows - int(blockIdx.y) * rows) * features; i += blockDim.x) {
        const int row = i / features, col = i % features;
        const float gate = __bfloat162float(__float2bfloat16_rn(partial[0][row][col] + partial[1][row][col]));
        const float up = __bfloat162float(__float2bfloat16_rn(partial[0][row][col + features] + partial[1][row][col + features]));
        const float silu = __bfloat162float(__float2bfloat16_rn(gate / (1.0f + expf(-gate))));
        output[(blockIdx.y * rows + row) * intermediate + blockIdx.x * features + col] = __float2bfloat16_rn(silu * up);
    }
}
} // namespace

bool FastllmCudaNaiveVerifyDenseGateUp(const Data &input, const Data &gate, const Data &up, Data &output) {
    namespace ll = cuda::low_latency;
    if (input.dims.empty() || input.dims.back() != hidden ||
        gate.dims != std::vector<int>{intermediate, hidden} || up.dims != gate.dims ||
        FastllmCudaRuntimeArch() != 90) return false;
    const auto active = input.Count(0) / hidden;
    if (!active || active > 128) return false;
    const int device = FastllmCudaGetDevice();
    if (output.isFake || output.cudaDataBorrowed || output.multiDeviceData ||
        (output.cudaData && output.dataDeviceIds != std::vector<int>{device})) return false;
    for (const Data *p : {&input, &gate, &up})
        if (p->dataType != BFLOAT16 || !ll::DenseData(*p, device) || ll::DataOverlap(*p, output)) return false;
    static thread_local int preparedDevice = -1;
    if (preparedDevice != device) {
        if (FastllmCudaGraphIsCapturingFast()) return false;
        auto error = cudaFuncSetAttribute(VerifyDenseGateUp16, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes);
        if (error != cudaSuccess) {
            if (ll::BackendUnavailable(error) || error == cudaErrorInvalidValue) { cudaGetLastError(); return false; }
            AssertInFastLLM(false, cudaGetErrorString(error));
        }
        preparedDevice = device;
    }
    GateUpMaps maps;
    if (!EncodeMap(maps.input, input.cudaData, active, rows) ||
        !EncodeMap(maps.gate, gate.cudaData, intermediate, features) ||
        !EncodeMap(maps.up, up.cudaData, intermediate, features)) return false;
    auto shape = input.dims;
    shape.back() = intermediate;
    output.dataType = BFLOAT16;
    output.UpdateUnitSize();
    output.Resize(shape);
    output.ToDevice(input.dataDevice, input.dataDeviceIds, false);
    output.Allocate(false);
    VerifyDenseGateUp16<<<dim3(intermediate / features, (active + rows - 1) / rows), 256, sharedBytes, cudaStreamPerThread>>>(
        maps, static_cast<Bf16 *>(output.cudaData), active);
    auto error = cudaGetLastError();
    AssertInFastLLM(error == cudaSuccess, cudaGetErrorString(error));
    return true;
}
} // namespace fastllm
