// Ordinary GGUF expert records: decode in registers, keeping cache entries packed.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "gguf.h"
#include "../fastllm-gguf-gemv.cuh"
#include "fastllm-cuda.cuh"
#include "fastllm-moe-stages.cuh"
#include "fastllm-moe-gguf-common.cuh"
#include "fastllm-moe-gguf-q8.cuh"
#include "fastllm-moe-deepseekv41-cache.cuh"
#include "fastllm-moe-v41-q8.cuh"
#include <algorithm>
#include <atomic>
#include <climits>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {
// Keep dispatch and admission on the same list. Cache slots hold ordinary GGUF
// blocks, either restored from NUMA shards or copied from a canonical snapshot.
#define GGUF_CACHE_TYPES(M) \
    M(Q2_0) M(Q4_0) M(Q4_1) M(Q5_0) M(Q5_1) M(Q8_0) M(Q8_1) \
    M(Q2_K) M(Q3_K) M(Q4_K) M(Q5_K) M(Q6_K) \
    M(IQ1_S) M(IQ1_M) M(IQ2_XXS) M(IQ2_XS) M(IQ2_S) \
    M(IQ3_XXS) M(IQ3_S) M(IQ4_NL) M(IQ4_XS) M(F32) M(F16) M(BF16)

#define GGUF_CACHE_Q8_LEGACY_TYPES(M) M(Q2_0) M(IQ1_M) M(IQ2_XXS) M(IQ2_XS) M(IQ2_S)
#define GGUF_CACHE_Q8_DECODE_TYPES(M) M(IQ3_XXS) M(IQ3_S) M(IQ4_NL) M(IQ4_XS)
#define GGUF_CACHE_Q8_TYPES(M) GGUF_CACHE_Q8_LEGACY_TYPES(M) GGUF_CACHE_Q8_DECODE_TYPES(M)

bool Q8TypeSupported(int type, bool legacy) {
    switch (static_cast<ggml_type>(type)) {
#define Q8_SUPPORTED(name) case GGML_TYPE_##name:
        GGUF_CACHE_Q8_LEGACY_TYPES(Q8_SUPPORTED)
            return true;
        GGUF_CACHE_Q8_DECODE_TYPES(Q8_SUPPORTED)
            return !legacy;
#undef Q8_SUPPORTED
        default: return false;
    }
}

// Bits 0/1 select Q8 gate/up and down independently for decode/verifier rows.
// Keep the existing large-batch dispatch; resident prefill uses grouped MMQ.
// A verifier row must use the same arithmetic as single-token decode.
int Q8Stages(int gateType, int downType, int hidden, int inter, int rows) {
    if (rows <= 0 || !FastllmCudaMoeGGUFCacheWorkspaceBytes(hidden, inter) ||
        !FastllmCudaMoeGGUFCacheSupported(gateType, hidden) ||
        !FastllmCudaMoeGGUFCacheSupported(downType, inter)) return 0;
    const bool gate = Q8TypeSupported(gateType, rows > 32);
    const bool down = Q8TypeSupported(downType, rows > 32);
    return rows <= 32 ? int(gate) | (int(down) << 1) : (gate && down ? 3 : 0);
}

int NumaOrdinary(int type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS_R4: return GGML_TYPE_IQ2_XXS;
        case GGML_TYPE_IQ2_XS_R4: return GGML_TYPE_IQ2_XS;
        case GGML_TYPE_IQ2_S_R4: return GGML_TYPE_IQ2_S;
        case GGML_TYPE_IQ3_XXS_R4: return GGML_TYPE_IQ3_XXS;
        default: return type;
    }
}

template<class View> bool NumaStagesSupported(const View &, int) { return true; }
bool NumaStagesSupported(const FastllmCudaMoeGGUFCacheView &view, int stages) {
    return (view.numaGateType < 0 || view.numaGateType == view.gateType || (stages & 1)) &&
           (view.numaDownType < 0 || view.numaDownType == view.downType || (stages & 2));
}

size_t Align16(size_t bytes) { return (bytes + 15) & ~size_t(15); }
size_t Q8Bytes(int rows, int columns) {
    return Align16(size_t(rows) * (columns / 32) * sizeof(block_q8_1));
}

size_t Q8WorkspaceBytes(int rows, int hidden, int inter, int topk) {
    const int routes = rows * topk;
    return Q8Bytes(rows, hidden) + Q8Bytes(routes, inter) +
        size_t(routes) * hidden * sizeof(float);
}

// The weight accessor is the only difference between cached records and
// resident tensors. Both views use identical projection and reduction kernels.
struct ResidentView {
    const uint8_t *const *weights;
    const int32_t *indices;
    int experts, gateType, downType, hidden, inter;
    void *workspace;
    size_t workspaceBytes;
};

template<class View>
__device__ __forceinline__ int OriginalRoute(const View &, int route) { return route; }
__device__ __forceinline__ int OriginalRoute(const FastllmCudaMoeGGUFCacheView &view, int route) {
    return view.routeMap ? view.routeMap[route] : route;
}
template<class View>
int ActiveRoutes(const View &, int routes) { return routes; }
int ActiveRoutes(const FastllmCudaMoeGGUFCacheView &view, int routes) {
    return view.routeMap ? view.routeCount : routes;
}

template<bool IsGate, class View>
__device__ __forceinline__ int NumaType(const View &) { return -1; }
template<bool IsGate>
__device__ __forceinline__ int NumaType(const FastllmCudaMoeGGUFCacheView &view) {
    return IsGate ? view.numaGateType : view.numaDownType;
}

template<ggml_type Type, int ThreadsPerRow>
__device__ __forceinline__ float ProjectionDot(const uint8_t *record,
        int row, size_t rowBytes, int storageType, const block_q8_1 *x,
        int columns, const uint64_t *grid) {
    if constexpr (Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ2_XS ||
                  Type == GGML_TYPE_IQ2_S || Type == GGML_TYPE_IQ3_XXS) {
        if (storageType >= 0 && storageType != Type)
            return gguf_cache_q8::RowDot<Type, ThreadsPerRow, true>(
                record + size_t(row/4)*4*rowBytes, x, columns, grid, row%4);
    }
    return gguf_cache_q8::RowDot<Type, ThreadsPerRow>(
        record + size_t(row)*rowBytes, x, columns, grid);
}

template<ggml_type Type>
__device__ __forceinline__ float2 ProjectionDotPair(const uint8_t *record,
        int gateRow, int upRow, size_t rowBytes, int storageType,
        const block_q8_1 *x, int columns, const uint64_t *grid) {
    // Keep the packing branch outside the accumulation loop, as in RowDot.
    if (storageType >= 0 && storageType != Type)
        return gguf_cache_q8::RowDotPair<Type, true>(
            record + size_t(gateRow/4)*4*rowBytes, record + size_t(upRow/4)*4*rowBytes,
            x, columns, grid, gateRow%4, upRow%4);
    return gguf_cache_q8::RowDotPair<Type>(record + size_t(gateRow)*rowBytes,
        record + size_t(upRow)*rowBytes, x, columns, grid, 0, 0);
}

template<bool IsGate>
__device__ __forceinline__ const uint8_t *ExpertWeight(
        const FastllmCudaMoeGGUFCacheView &view, int route) {
    const int slot = view.routeSlots[route];
    return slot < 0 ? nullptr : view.records +
        (view.slotOffsets ? view.slotOffsets[slot] : size_t(slot)*view.recordStride) +
        (IsGate ? 0 : view.downOffset);
}
template<bool IsGate>
__device__ __forceinline__ const uint8_t *ExpertWeight(
        const ResidentView &view, int route) {
    const int expert = view.indices[route];
    return expert < 0 || expert >= view.experts ? nullptr :
        view.weights[2*expert + (IsGate ? 0 : 1)];
}

template<typename T>
__global__ void QuantizeQ8(const T *input, block_q8_1 *output, int columns) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    // A complete warp owns a Q8_1 block. columns is divisible by 32.
    if (c >= columns) return;
    const float x = float(input[size_t(blockIdx.y)*columns + c]);
    float maximum = fabsf(x), sum = x;
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }
    const float scale = maximum / 127.0f;
    auto &q = output[size_t(blockIdx.y)*(columns/32) + c/32];
    q.qs[c%32] = maximum == 0 ? 0 : int8_t(roundf(x / scale));
    if (c%32 == 0) q.ds = __floats2half2_rn(scale, sum);
}

template<ggml_type Type, typename T, bool IsGate, typename View, int ThreadsPerRow = 32>
__global__ void Q8Projection(const block_q8_1 *input, T *gate, float *partial,
                            View view, int topk, size_t rowBytes) {
    using F = gguf_cache_q8::Format<Type>;
    __shared__ uint64_t grid[F::gridSize ? F::gridSize : 1];
    extern __shared__ uint32_t activation[];
    const int route = blockIdx.y;
    const int columns = IsGate ? view.hidden : view.inter;
    const auto *x = input + size_t(IsGate ? OriginalRoute(view, route)/topk : route)*(columns/32);
    gguf_cache_q8::StageGrid<Type>(grid);
    for (int i = threadIdx.x; i < columns/32*int(sizeof(block_q8_1))/4; i += blockDim.x)
        activation[i] = reinterpret_cast<const uint32_t *>(x)[i];
    __syncthreads();
    const int row = blockIdx.x*(256/ThreadsPerRow) + threadIdx.x/ThreadsPerRow;
    if (row >= (IsGate ? view.inter : view.hidden)) return;
    const uint8_t *record = ExpertWeight<IsGate>(view, route);
    float value = 0;
    if (record != nullptr) {
        const auto *sharedX = reinterpret_cast<const block_q8_1 *>(activation);
        const int storageType = NumaType<IsGate>(view);
        const int physicalRow = IsGate && storageType >= 0 ? 2*row : row;
        float up = 0;
        const int upRow = storageType >= 0 ? 2*row+1 : row+view.inter;
        if constexpr (IsGate && (Type == GGML_TYPE_IQ2_S || Type == GGML_TYPE_IQ3_XXS)) {
            const float2 values = ProjectionDotPair<Type>(record, physicalRow, upRow,
                rowBytes, storageType, sharedX, columns, grid);
            value = values.x;
            up = values.y;
        } else {
            value = ProjectionDot<Type, ThreadsPerRow>(record, physicalRow, rowBytes,
                storageType, sharedX, columns, grid);
            if constexpr (IsGate)
                up = ProjectionDot<Type, ThreadsPerRow>(record, upRow, rowBytes,
                    storageType, sharedX, columns, grid);
        }
        if constexpr (IsGate) {
            up = float(DequantizeCast<T>::cast(up));
            value = float(DequantizeCast<T>::cast(value));
            value = value / (1.0f + expf(-value)) * up;
        }
    }
    if (threadIdx.x%ThreadsPerRow == 0) {
        if constexpr (IsGate) gate[size_t(route)*view.inter + row] = DequantizeCast<T>::cast(value);
        else partial[size_t(OriginalRoute(view, route))*view.hidden + row] = float(DequantizeCast<T>::cast(value));
    }
}

template<ggml_type Type, typename T, typename View>
void LaunchQ8Down(const block_q8_1 *input, T *gate, float *partial,
                  const View &view, int topk, int routes, size_t rowBytes) {
    if constexpr (Type == GGML_TYPE_Q2_0) {
        // Eight lanes reproduce the original 32-lane sum for up to sixteen
        // Q8 blocks, and let one CTA cover 32 rather than eight output rows.
        // Select the smaller subgroup for prefill batches. Keep the existing
        // launch for decode/verifier batches, whose end-to-end gain is unstable.
        if (view.inter <= 512 && routes > 8*topk) {
            Q8Projection<Type, T, false, View, 8><<<dim3((view.hidden+31)/32, routes), 256,
                view.inter/32*sizeof(block_q8_1), cudaStreamPerThread>>>(
                    input, gate, partial, view, topk, rowBytes);
            return;
        }
    }
    Q8Projection<Type, T, false><<<dim3((view.hidden+7)/8, routes), 256,
        view.inter/32*sizeof(block_q8_1), cudaStreamPerThread>>>(
            input, gate, partial, view, topk, rowBytes);
}

template<typename T>
__global__ void Q8Reduce(const float *partial, T *output, const float *scores,
                         int hidden, int topk) {
    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= hidden) return;
    float sum = 0;
    // Keep the same expert order and activation/output rounding as fallback.
    const int firstRoute = blockIdx.y * topk;
    for (int route = firstRoute; route < firstRoute + topk; ++route)
        sum += partial[size_t(route)*hidden + row]*scores[route];
    output[size_t(blockIdx.y)*hidden + row] = DequantizeCast<T>::cast(sum);
}

template<ggml_type type, typename T>
__device__ __forceinline__ float Dot(const void *weight, const T *input,
                                      int columns, int warp, int warps) {
    const int lane = threadIdx.x % 32;
    float sum = 0;
    if constexpr (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        for (int c = warp * 32 + lane; c < columns; c += warps * 32) {
            float w;
            if constexpr (type == GGML_TYPE_F32) w = static_cast<const float *>(weight)[c];
            else if constexpr (type == GGML_TYPE_F16) w = __half2float(static_cast<const half *>(weight)[c]);
            else w = __bfloat162float(static_cast<const __nv_bfloat16 *>(weight)[c]);
            sum = fmaf(float(DequantizeCast<T>::cast(w)), float(input[c]), sum);
        }
    } else if constexpr (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_1) {
        for (int b = warp; b < columns / 32; b += warps) {
            float w;
            if constexpr (type == GGML_TYPE_Q8_1) {
                const auto &q = static_cast<const block_q8_1 *>(weight)[b];
                w = __low2float(q.ds) * q.qs[lane];
            } else {
                using Block = std::conditional_t<type == GGML_TYPE_Q5_0, block_q5_0, block_q5_1>;
                const auto &q = static_cast<const Block *>(weight)[b];
                const int low = (q.qs[lane % 16] >> (4 * (lane / 16))) & 15;
                const int high = (q.qh[lane / 8] >> (lane % 8)) & 1;
                if constexpr (type == GGML_TYPE_Q5_0) w = __half2float(q.d) * (low + 16 * high - 16);
                else w = __low2float(q.dm) * (low + 16 * high) + __high2float(q.dm);
            }
            sum = fmaf(float(DequantizeCast<T>::cast(w)), float(input[b * 32 + lane]), sum);
        }
    } else {
        FastllmGgufGemvDotOutput<T> dot{input, &sum, 0};
        for (int b = warp; b < (columns + 255) / 256; b += warps)
            FastllmGgufGemvBlock<type>(weight, dot, b, lane, columns);
    }
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_down_sync(0xffffffff, sum, mask);
    return sum;
}

template<ggml_type type, typename T, typename View>
__global__ void Gate(const T *input, T *gateOutput, View view,
                     int topk, size_t rowBytes) {
    const int row = blockIdx.x, route = blockIdx.y;
    const uint8_t *record = ExpertWeight<true>(view, route);
    if (record == nullptr) {
        if (threadIdx.x == 0) gateOutput[route * view.inter + row] = DequantizeCast<T>::cast(0);
        return;
    }
    input += size_t(OriginalRoute(view, route)/topk)*view.hidden;
    const int warp = threadIdx.x / 32;
    const bool cross = NumaType<true>(view) >= 0;
    float gate = Dot<type>(record + size_t(cross ? 2*row : row) * rowBytes, input, view.hidden, warp, 4);
    float up = Dot<type>(record + size_t(cross ? 2*row+1 : row+view.inter) * rowBytes, input, view.hidden, warp, 4);
    __shared__ float gates[4], ups[4];
    if (threadIdx.x % 32 == 0) { gates[warp] = gate; ups[warp] = up; }
    __syncthreads();
    if (threadIdx.x == 0) {
        gate = up = 0;
        for (int w = 0; w < 4; ++w) { gate += gates[w]; up += ups[w]; }
        gate = float(DequantizeCast<T>::cast(gate));
        up = float(DequantizeCast<T>::cast(up));
        gateOutput[route * view.inter + row] = DequantizeCast<T>::cast(gate / (1.0f + expf(-gate)) * up);
    }
}

template<ggml_type type, typename T, typename View>
__global__ void Down(const T *gateOutput, T *output, View view,
                     const float *scores, int topk, size_t rowBytes, float *perExpert, int routes) {
    const int row = blockIdx.x, localRoute = threadIdx.x / 32;
    const int route = blockIdx.y*topk + localRoute;
    // A compact last block can be partial. Compact calls always write
    // perExpert, so these inactive warps never participate in a block barrier.
    if (route >= routes) return;
    const uint8_t *record = ExpertWeight<false>(view, route);
    float value = 0;
    if (record != nullptr) {
        const uint8_t *weight = record + size_t(row) * rowBytes;
        value = Dot<type>(weight, gateOutput + route * view.inter, view.inter, 0, 1);
    }
    if (perExpert) {
        if (threadIdx.x % 32 == 0)
            perExpert[size_t(OriginalRoute(view, route))*view.hidden + row] = float(DequantizeCast<T>::cast(value));
        return;
    }
    extern __shared__ float partial[];
    if (threadIdx.x % 32 == 0) partial[localRoute] = float(DequantizeCast<T>::cast(value)) * scores[route];
    __syncthreads();
    if (threadIdx.x == 0) {
        float result = 0;
        for (int e = 0; e < topk; ++e) result += partial[e];
        output[size_t(blockIdx.y)*view.hidden + row] = DequantizeCast<T>::cast(result);
    }
}

template<typename T, typename View>
bool Compute(const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &output,
             const View &view, const float *scores, int topk, float *perExpert = nullptr,
             bool q8InputPrepared = false, const FastllmCudaMoeStageEvents *events = nullptr) {
    const int rows = input.dims[0], routes = ActiveRoutes(view, rows * topk);
    const int stages = view.workspace && view.workspaceBytes >= Q8WorkspaceBytes(rows, view.hidden, view.inter, topk)
        ? Q8Stages(view.gateType, view.downType, view.hidden, view.inter, rows) : 0;
    if (!NumaStagesSupported(view, stages)) return false;
    block_q8_1 *qInput = nullptr, *qGate = nullptr;
    float *partial = perExpert;
    if (stages) {
        qInput = static_cast<block_q8_1 *>(view.workspace);
        qGate = reinterpret_cast<block_q8_1 *>(static_cast<uint8_t *>(view.workspace) + Q8Bytes(rows, view.hidden));
        if (!partial)
            partial = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(qGate) + Q8Bytes(routes, view.inter));
    }
    const auto gateType = static_cast<ggml_type>(view.gateType);
    const auto downType = static_cast<ggml_type>(view.downType);
    if (stages & 1) {
        if (!q8InputPrepared)
            QuantizeQ8<<<dim3((view.hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
                static_cast<const T *>(input.cudaData), qInput, view.hidden);
        switch (gateType) {
#define Q8_GATE(name) case GGML_TYPE_##name: \
            Q8Projection<GGML_TYPE_##name, T, true><<<dim3((view.inter+7)/8, routes), 256, \
                view.hidden/32*sizeof(block_q8_1), cudaStreamPerThread>>>(qInput, \
                static_cast<T *>(gate.cudaData), partial, view, topk, ggml_row_size(gateType, view.hidden)); break;
            GGUF_CACHE_Q8_TYPES(Q8_GATE)
#undef Q8_GATE
            default: return false;
        }
    } else {
        switch (gateType) {
#define LAUNCH_GATE(name) case GGML_TYPE_##name: \
            Gate<GGML_TYPE_##name><<<dim3(view.inter, routes), 128, 0, cudaStreamPerThread>>>( \
                static_cast<const T *>(input.cudaData), static_cast<T *>(gate.cudaData), \
                view, topk, ggml_row_size(gateType, view.hidden)); break;
            GGUF_CACHE_TYPES(LAUNCH_GATE)
#undef LAUNCH_GATE
            default: return false;
        }
    }
    if (stages & 2)
        QuantizeQ8<<<dim3((view.inter+255)/256, routes), 256, 0, cudaStreamPerThread>>>(
            static_cast<const T *>(gate.cudaData), qGate, view.inter);
    // Gate/up and activation quantization do not read down weights. Run them
    // during the remaining DMA, and keep its wait out of scheduler timings.
    if (events && !FastllmMoeWaitDown(*events)) return false;
    if (stages & 2) {
        switch (downType) {
#define Q8_DOWN(name) case GGML_TYPE_##name: \
            LaunchQ8Down<GGML_TYPE_##name>(qGate, static_cast<T *>(gate.cudaData), \
                partial, view, topk, routes, ggml_row_size(downType, view.inter)); break;
            GGUF_CACHE_Q8_TYPES(Q8_DOWN)
#undef Q8_DOWN
            default: return false;
        }
        if (!perExpert)
            Q8Reduce<<<dim3((view.hidden+255)/256, rows), 256, 0, cudaStreamPerThread>>>(
                partial, static_cast<T *>(output.cudaData), scores, view.hidden, topk);
    } else {
        switch (downType) {
#define LAUNCH_DOWN(name) case GGML_TYPE_##name: \
            Down<GGML_TYPE_##name><<<dim3(view.hidden, (routes+topk-1)/topk), topk*32, topk*sizeof(float), cudaStreamPerThread>>>( \
                static_cast<const T *>(gate.cudaData), static_cast<T *>(output.cudaData), \
                view, scores, topk, ggml_row_size(downType, view.inter), perExpert, routes); break;
            GGUF_CACHE_TYPES(LAUNCH_DOWN)
#undef LAUNCH_DOWN
            default: return false;
        }
    }
    return cudaGetLastError() == cudaSuccess;
}

struct ResidentLayer {
    int device = -1, experts = 0, gateType = -1, downType = -1, hidden = 0, inter = 0;
    void *table = nullptr;
    std::vector<const fastllm::Data *> sources;
    std::atomic<bool> retired{false};
    ~ResidentLayer() {
        if (!table) return;
        int previous = -1;
        if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
            cudaFree(table);
            if (previous != device) cudaSetDevice(previous);
        }
    }
};
struct ResidentRegistry {
    std::mutex mutex;
    std::map<const fastllm::Data *, std::shared_ptr<ResidentLayer>> layers;
    std::unordered_map<const fastllm::Data *, const fastllm::Data *> owners;
};
ResidentRegistry &ResidentLayers() {
    // Data destructors retire every live entry, including during static teardown.
    static auto *registry = new ResidentRegistry;
    return *registry;
}
thread_local std::unordered_map<const fastllm::Data *, std::weak_ptr<ResidentLayer>> residentFront;

std::shared_ptr<ResidentLayer> GetResidentLayer(fastllm::Data **weights, int count, int device) {
    const auto *key = weights[2];
    auto front = residentFront.find(key);
    if (front != residentFront.end()) {
        auto layer = front->second.lock();
        if (layer && !layer->retired.load(std::memory_order_acquire) &&
            layer->device == device && (layer->experts + 1)*2 == count) return layer;
        residentFront.erase(front);
    }
    auto &registry = ResidentLayers();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.layers.find(key);
    if (found != registry.layers.end()) {
        auto layer = found->second;
        if (layer->device != device || (layer->experts + 1)*2 != count) return {};
        residentFront[key] = layer;
        return layer;
    }
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return {};
    auto layer = std::make_shared<ResidentLayer>();
    layer->device = device;
    layer->experts = count/2 - 1;
    layer->hidden = weights[2]->dims[1];
    layer->inter = weights[2]->dims[0]/2;
    layer->gateType = weights[2]->ggmlType;
    layer->downType = weights[3]->ggmlType;
    if (layer->inter <= 0 || !FastllmCudaMoeGGUFCacheSupported(layer->gateType, layer->hidden) ||
        !FastllmCudaMoeGGUFCacheSupported(layer->downType, layer->inter)) return {};
    std::vector<const uint8_t *> pointers(2*layer->experts, nullptr);
    for (int e = 0; e < layer->experts; ++e) {
        auto *gate = weights[2*(e+1)], *down = weights[2*(e+1)+1];
        if (!gate && !down) continue;
        if (!gate || !down) return {};
        for (int part = 0; part < 2; ++part) {
            auto *weight = part ? down : gate;
            const int rows = part ? layer->hidden : 2*layer->inter;
            const int cols = part ? layer->inter : layer->hidden;
            const int type = part ? layer->downType : layer->gateType;
            if (weight->dataType != fastllm::DATA_GGUF_FORMAT ||
                weight->dataDevice != fastllm::CUDA || !weight->cudaData ||
                weight->dataDeviceIds != std::vector<int>{device} ||
                weight->ggmlType != type || weight->dims != std::vector<int>({rows, cols})) return {};
            auto owner = registry.owners.find(weight);
            if (owner != registry.owners.end() && owner->second != key) return {};
            pointers[2*e+part] = static_cast<const uint8_t *>(weight->cudaData);
            layer->sources.push_back(weight);
        }
    }
    const size_t bytes = pointers.size()*sizeof(pointers[0]);
    if (cudaMalloc(&layer->table, bytes) != cudaSuccess) { cudaGetLastError(); return {}; }
    // Complete the one-time upload before publishing it to another stream.
    if (cudaMemcpy(layer->table, pointers.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) return {};
    for (const auto *source : layer->sources) registry.owners[source] = key;
    registry.layers[key] = layer;
    if (residentFront.size() > 256) residentFront.clear();
    residentFront[key] = layer;
    return layer;
}

}

size_t FastllmCudaMoeGGUFCacheWorkspaceBytes(int hidden, int inter) {
    // Bound shared Q8 input + the largest (IQ1_M) codebook below 48 KiB.
    if (hidden <= 0 || inter <= 0 || hidden > 24576 || inter > 24576 || hidden%32 || inter%32) return 0;
    const size_t v41 = Align16(size_t(8) * (hidden / 256) * sizeof(block_q8_K)) +
                       Align16(size_t(8 * 16) * (inter / 256) * sizeof(block_q8_K));
    return std::max(Q8WorkspaceBytes(1, hidden, inter, 32), v41);
}

size_t FastllmCudaMoeGGUFCacheBatchWorkspaceBytes(int hidden, int inter, int rows, int topk) {
    if (rows <= 0 || topk <= 0 || topk > 32 || rows > INT_MAX / topk ||
        !FastllmCudaMoeGGUFCacheWorkspaceBytes(hidden, inter)) return 0;
    return Q8WorkspaceBytes(rows, hidden, inter, topk);
}

bool FastllmCudaMoeGGUFCacheQ8Supported(int gateType, int downType, int hidden, int inter) {
    return Q8Stages(gateType, downType, hidden, inter, 1) == 3;
}

bool FastllmCudaMoeGGUFCacheNumaSupported(int gateType, int downType,
        int hidden, int inter, int rows) {
    const int gate = NumaOrdinary(gateType), down = NumaOrdinary(downType);
    if (rows <= 0 || !FastllmCudaMoeGGUFCacheSupported(gate, hidden) ||
        !FastllmCudaMoeGGUFCacheSupported(down, inter)) return false;
    const int stages = Q8Stages(gate, down, hidden, inter, rows);
    return (gateType == gate || ((stages & 1) && inter % 2 == 0)) &&
           (downType == down || ((stages & 2) && hidden % 4 == 0));
}

bool FastllmCudaMoeGGUFCacheSupported(int type, int columns) {
    if (columns <= 0) return false;
    switch (static_cast<ggml_type>(type)) {
#define GGUF_SUPPORTED(name) case GGML_TYPE_##name:
        GGUF_CACHE_TYPES(GGUF_SUPPORTED)
#undef GGUF_SUPPORTED
#undef GGUF_CACHE_TYPES
            return columns % ggml_blck_size(static_cast<ggml_type>(type)) == 0;
        default: return false;
    }
}

bool FastllmCudaMoeGGUFCacheCompute(const fastllm::Data &input, fastllm::Data &gate,
        fastllm::Data &output, const FastllmCudaMoeGGUFCacheView &view,
        const float *scores, int topk, float *perExpert) {
    return FastllmCudaMoeGGUFCacheComputeStaged(input, gate, output, view, scores, topk, perExpert, {});
}

bool FastllmCudaMoeGGUFCacheComputeStaged(const fastllm::Data &input, fastllm::Data &gate,
        fastllm::Data &output, const FastllmCudaMoeGGUFCacheView &view,
        const float *scores, int topk, float *perExpert,
        const FastllmCudaMoeStageEvents &events) {
    if (input.dims.size() != 2 || input.dims[0] <= 0 || input.dims[1] != view.hidden ||
        topk <= 0 || topk > 32 || input.dims[0] > INT_MAX / topk ||
        !scores || !view.records || !view.routeSlots ||
        (view.routeMap && (!perExpert || view.routeCount <= 0 || view.routeCount > input.dims[0]*topk)) ||
        !FastllmCudaMoeGGUFCacheSupported(view.gateType, view.hidden) ||
        !FastllmCudaMoeGGUFCacheSupported(view.downType, view.inter)) return false;
    if ((view.numaGateType >= 0 || view.numaDownType >= 0) &&
        (NumaOrdinary(view.numaGateType) != view.gateType ||
         NumaOrdinary(view.numaDownType) != view.downType ||
         !FastllmCudaMoeGGUFCacheNumaSupported(view.numaGateType, view.numaDownType,
             view.hidden, view.inter, input.dims[0]))) return false;
    switch (input.dataType) {
        case fastllm::FLOAT32: return Compute<float>(input, gate, output, view, scores, topk, perExpert, view.q8InputPrepared, &events);
        case fastllm::FLOAT16: return Compute<half>(input, gate, output, view, scores, topk, perExpert, view.q8InputPrepared, &events);
        case fastllm::BFLOAT16: return Compute<__nv_bfloat16>(input, gate, output, view, scores, topk, perExpert, view.q8InputPrepared, &events);
        default: return false;
    }
}


namespace glm5_gguf_cache {
// Match iqk_quantize_row_q8_K on the CPU: one FP32 positive scale per
// 256 elements and nearest-even integer rounding. Keep the dequantized
// activation in scratch; original expert weights remain packed in VRAM.
__global__ void Quantize(const __nv_bfloat16 *input, float *output, int columns) {
    __shared__ float maxima[8];
    const int c = blockIdx.x * 256 + threadIdx.x;
    const size_t offset = size_t(blockIdx.y) * columns + c;
    const float x = __bfloat162float(input[offset]);
    float amax = fabsf(x);
    for (int mask = 16; mask; mask >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
    if (threadIdx.x % 32 == 0) maxima[threadIdx.x / 32] = amax;
    __syncthreads();
    amax = 0;
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, maxima[i]);
    const float inverse = amax == 0 ? 0 : __fdiv_rn(127.f, amax);
    const int q = __float2int_rn(__fmul_rn(x, inverse));
    output[offset] = __fmul_rn(__fdiv_rn(amax, 127.f), float(q));
}

template<ggml_type Type>
__global__ void Gate(const float *input, __nv_bfloat16 *activation,
                    FastllmCudaMoeGGUFCacheView view, const float *scores, float limit) {
    const int row = blockIdx.x, route = blockIdx.y;
    const auto *record = ExpertWeight<true>(view, route);
    if (!record) {
        if (threadIdx.x == 0) activation[size_t(route) * view.inter + row] = __float2bfloat16_rn(0);
        return;
    }
    const size_t pitch = size_t(view.hidden / QK_K) * sizeof(typename std::conditional<
        Type == GGML_TYPE_IQ2_XXS, block_iq2_xxs, block_iq2_s>::type);
    const int warp = threadIdx.x / 32;
    float gate = Dot<Type>(record + size_t(row) * pitch, input, view.hidden, warp, 4);
    float up = Dot<Type>(record + size_t(row + view.inter) * pitch, input, view.hidden, warp, 4);
    __shared__ float gates[4], ups[4];
    if (threadIdx.x % 32 == 0) { gates[warp] = gate; ups[warp] = up; }
    __syncthreads();
    if (threadIdx.x == 0) {
        gate = up = 0;
        for (int i = 0; i < 4; ++i) { gate += gates[i]; up += ups[i]; }
        gate = __bfloat162float(__float2bfloat16_rn(gate));
        up = __bfloat162float(__float2bfloat16_rn(up));
        if (limit > 0) { gate = fminf(gate, limit); up = fmaxf(-limit, fminf(up, limit)); }
        const float value = __fmul_rn(__fmul_rn(gate / (1.f + expf(-gate)), up), scores[route]);
        activation[size_t(route) * view.inter + row] = __float2bfloat16_rn(value);
    }
}

template<ggml_type Type, typename T>
__global__ void Down(const T *activation, float *output, FastllmCudaMoeGGUFCacheView view) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32, route = blockIdx.y;
    if (row >= view.hidden) return;
    const auto *record = ExpertWeight<false>(view, route);
    const size_t pitch = size_t(view.inter / QK_K) * sizeof(typename std::conditional<
        Type == GGML_TYPE_IQ3_XXS, block_iq3_xxs, block_iq4_xs>::type);
    const float value = record ? Dot<Type>(record + size_t(row) * pitch,
        activation + size_t(route) * view.inter, view.inter, 0, 1) : 0;
    if (threadIdx.x % 32 == 0)
        output[size_t(route) * view.hidden + row] = __bfloat162float(__float2bfloat16_rn(value));
}
} // namespace glm5_gguf_cache

bool FastllmCudaMoeGlm5GGUFCacheSupported(int gateType, int downType, int hidden, int inter) {
    return (gateType == GGML_TYPE_IQ2_XXS || gateType == GGML_TYPE_IQ2_S) &&
           (downType == GGML_TYPE_IQ3_XXS || downType == GGML_TYPE_IQ4_XS) &&
           hidden > 0 && inter > 0 && hidden % QK_K == 0 && inter % QK_K == 0 &&
           FastllmCudaMoeGGUFCacheWorkspaceBytes(hidden, inter) >=
               (size_t(hidden) + 16 * size_t(inter)) * sizeof(float);
}

bool FastllmCudaMoeGlm5GGUFCacheCompute(const fastllm::Data &input, fastllm::Data &activation,
        const FastllmCudaMoeGGUFCacheView &view, const float *scores, int topk,
        float swigluLimit, float *perExpert) {
    if (input.dataDevice != fastllm::CUDA || input.dataType != fastllm::BFLOAT16 ||
        input.dims != std::vector<int>({1, view.hidden}) || !input.cudaData || topk < 1 || topk > 16 ||
        !FastllmCudaMoeGlm5GGUFCacheSupported(view.gateType, view.downType, view.hidden, view.inter) ||
        !view.workspace || view.workspaceBytes < (size_t(view.hidden) + size_t(topk) * view.inter) * sizeof(float) ||
        !scores || !perExpert || !view.records || !view.routeSlots) return false;
    auto *x = static_cast<float *>(view.workspace);
    auto *y = x + view.hidden;
    fastllm_gguf_moe::AllocateTensor(activation, fastllm::BFLOAT16,
        {topk, view.inter}, FastllmCudaGetDevice());
    glm5_gguf_cache::Quantize<<<view.hidden / QK_K, 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), x, view.hidden);
    if (view.gateType == GGML_TYPE_IQ2_XXS)
        glm5_gguf_cache::Gate<GGML_TYPE_IQ2_XXS><<<dim3(view.inter, topk), 128, 0, cudaStreamPerThread>>>(
            x, static_cast<__nv_bfloat16 *>(activation.cudaData), view, scores, swigluLimit);
    else
        glm5_gguf_cache::Gate<GGML_TYPE_IQ2_S><<<dim3(view.inter, topk), 128, 0, cudaStreamPerThread>>>(
            x, static_cast<__nv_bfloat16 *>(activation.cudaData), view, scores, swigluLimit);
    // IQ3 uses the CPU's Q8_K down-input boundary. IQ4_XS uses the BF16
    // fallback, including BF16 rounding of each decoded weight in Dot<T>.
    // GGUF applies neither NVFP4's block-128 nor V4.1's block-32 FP8 step.
    if (view.downType == GGML_TYPE_IQ3_XXS) {
        glm5_gguf_cache::Quantize<<<dim3(view.inter / QK_K, topk), 256, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(activation.cudaData), y, view.inter);
        glm5_gguf_cache::Down<GGML_TYPE_IQ3_XXS><<<dim3((view.hidden + 3) / 4, topk), 128, 0, cudaStreamPerThread>>>(
            y, perExpert, view);
    } else {
        glm5_gguf_cache::Down<GGML_TYPE_IQ4_XS><<<dim3((view.hidden + 3) / 4, topk), 128, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(activation.cudaData), perExpert, view);
    }
    return cudaGetLastError() == cudaSuccess;
}

void FastllmCudaReleaseMoeGGUFResident(const fastllm::Data *weight) {
    if (!weight) return;
    auto &registry = ResidentLayers();
    std::shared_ptr<ResidentLayer> retired;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        auto owner = registry.owners.find(weight);
        if (owner == registry.owners.end()) return;
        auto layer = registry.layers.find(owner->second);
        if (layer == registry.layers.end()) return;
        retired = std::move(layer->second);
        retired->retired.store(true, std::memory_order_release);
        registry.layers.erase(layer);
        for (const auto *source : retired->sources) registry.owners.erase(source);
    }
    // cudaFree runs outside the registry lock, on the allocation's own GPU.
}

bool FastllmCudaMergeMOEGGUFResidentIndexed(
        const fastllm::Data &input, fastllm::Data &gate,
        fastllm::Data &workspace, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores, int topk) {
    using fastllm_gguf_moe::AllocateTensor;
    if (topk <= 0 || topk > 32 || !indices || !scores ||
        input.dataDevice != fastllm::CUDA || !input.cudaData || input.dims.size() != 2 || input.dims[0] <= 0 || input.dims[0] > 4096 ||
        !fastllm_gguf_moe::IsActivationType(input.dataType) ||
        !weights || weightsBatch < 4 || weightsBatch%2 || !weights[2] || !weights[3] ||
        weights[2]->dataType != fastllm::DATA_GGUF_FORMAT || weights[2]->dims.size() != 2 ||
        weights[2]->dataDevice != fastllm::CUDA || weights[3]->dataDevice != fastllm::CUDA) return false;
    const int device = FastllmCudaGetDevice();
    if (input.dataDeviceIds != std::vector<int>{device} || input.dims[1] != weights[2]->dims[1]) return false;
    auto layer = GetResidentLayer(weights, weightsBatch, device);
    if (!layer) return false;
    const int rows = input.dims[0];
    if (rows > 32) {
        const size_t bytes = FastllmCudaMoeGGUFGroupedWorkspaceBytes(
            layer->gateType, layer->downType, rows, layer->hidden,
            layer->inter, layer->experts, topk);
        if (!bytes || bytes > size_t(INT32_MAX)) return false;
        AllocateTensor(gate, input.dataType, {rows*topk, layer->inter}, device);
        AllocateTensor(output, input.dataType, {rows, layer->hidden}, device);
        AllocateTensor(workspace, fastllm::INT8, {int(bytes)}, device);
        return FastllmCudaMoeGGUFGrouped(input, gate, output, layer->table,
            indices, scores, workspace.cudaData, layer->gateType,
            layer->downType, layer->hidden, layer->inter, layer->experts, topk);
    }
    const int routes = rows * topk;
    AllocateTensor(gate, input.dataType, {routes, layer->inter}, device);
    AllocateTensor(output, input.dataType, {rows, layer->hidden}, device);
    const size_t bytes = Q8Stages(
        layer->gateType, layer->downType, layer->hidden, layer->inter, rows)
        ? Q8WorkspaceBytes(rows, layer->hidden, layer->inter, topk) : 0;
    if (bytes) AllocateTensor(workspace, fastllm::INT8, {int(bytes)}, device);
    const ResidentView view{static_cast<const uint8_t *const *>(layer->table), indices,
        layer->experts, layer->gateType, layer->downType, layer->hidden, layer->inter,
        bytes ? workspace.cudaData : nullptr, bytes};
    switch (input.dataType) {
        case fastllm::FLOAT32: return Compute<float>(input, gate, output, view, scores, topk);
        case fastllm::FLOAT16: return Compute<half>(input, gate, output, view, scores, topk);
        case fastllm::BFLOAT16: return Compute<__nv_bfloat16>(input, gate, output, view, scores, topk);
        default: return false;
    }
}

namespace v41_gguf_cache {
// Match GGML Q8_K/K32's signed maximum and round-to-nearest-even quants.
// K32 only changes partial-sum metadata, which these integer dot kernels
// calculate directly. The input has already crossed the FP8 block-32 boundary.
__global__ void Quantize(const __nv_bfloat16 *input, block_q8_K *output, int columns) {
    const int c = threadIdx.x;
    const float x = __bfloat162float(input[size_t(blockIdx.y) * columns + blockIdx.x * 256 + c]);
    fastllm::cuda::v41_gguf::QuantizeQ8K(x,
        output[size_t(blockIdx.y) * (columns / 256) + blockIdx.x]);
}

template<bool Gate>
__device__ float Dot(const uint8_t *weight, const block_q8_K *input, int row, int columns) {
    const int lane = threadIdx.x % 32, r = row % 4, blocks = columns / 256;
    float result = 0;
    for (int b = 0; b < blocks; ++b) {
        int dot = 0, bias = 0;
        float scale, minimum;
        if constexpr (Gate) {
            const auto &w = reinterpret_cast<const block_q2_k_r4 *>(weight)[size_t(row / 4) * blocks + b];
            scale = __half2float(reinterpret_cast<const half *>(w.d)[r]);
            minimum = __half2float(reinterpret_cast<const half *>(w.d)[r + 4]);
            // Four adjacent activations share a scale. R4 stores their
            // packed weights in one aligned word; keep the integer sum
            // exact with DP4A before the original per-block FP32 FMAs.
            #pragma unroll
            for (int c = lane * 4; c < 256; c += 128) {
                const int pos = c % 32;
                const uint32_t packed = *reinterpret_cast<const uint32_t *>(
                    w.qs + 32 * (c / 32) + 4 * r + 16 * (pos / 16));
                const int q = (packed >> (2 * ((pos % 16) / 4))) & 0x03030303;
                const int s = w.scales[4 * (c / 16) + r];
                const int x = *reinterpret_cast<const int *>(input[b].qs + c);
                dot += (s & 15) * __dp4a(q, x, 0);
                bias += (s >> 4) * __dp4a(0x01010101, x, 0);
            }
        } else {
            const auto &w = reinterpret_cast<const block_q4_k_r4 *>(weight)[size_t(row / 4) * blocks + b];
            scale = __half2float(reinterpret_cast<const half *>(w.d)[r]);
            minimum = __half2float(reinterpret_cast<const half *>(w.d)[r + 4]);
            #pragma unroll
            for (int c = lane * 4; c < 256; c += 128) {
                const int pos = c % 32, index = 4 * (c / 32) + r;
                const int high = (w.scales_h[index % 16] >> (4 * (index / 16))) & 15;
                const int low = w.scales_l[index];
                const uint32_t packed = *reinterpret_cast<const uint32_t *>(
                    w.qs + 64 * (c / 32) + 4 * r + 32 * ((pos % 8) / 4) + 16 * (pos / 16));
                const int q = (packed >> (4 * ((pos % 16) / 8))) & 0x0f0f0f0f;
                const int x = *reinterpret_cast<const int *>(input[b].qs + c);
                dot += ((low & 15) + 16 * (high & 3)) * __dp4a(q, x, 0);
                bias += ((low >> 4) + 16 * (high >> 2)) * __dp4a(0x01010101, x, 0);
            }
        }
        for (int mask = 16; mask; mask >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, mask);
            bias += __shfl_down_sync(0xffffffff, bias, mask);
        }
        result = fmaf(input[b].d * scale, float(dot), result);
        result = fmaf(-input[b].d * minimum, float(bias), result);
    }
    return result;
}

__global__ void Gate(const block_q8_K *input, __nv_bfloat16 *activation,
                     FastllmCudaMoeGGUFCacheView view, const float *scores, int topk, float limit) {
    const int column = blockIdx.x * 4 + threadIdx.x / 32, route = blockIdx.y;
    if (column >= view.inter) return;
    if (view.routeSlots[route] < 0) {
        if (threadIdx.x % 32 == 0) activation[size_t(route) * view.inter + column] = __float2bfloat16_rn(0);
        return;
    }
    const uint8_t *record = view.records + size_t(view.routeSlots[route]) * view.recordStride;
    input += size_t(route / topk) * (view.hidden / 256);
    using fastllm::cuda::dsv41_cache::BFloat;
    // NUMA preserves the cross-interleaved gate/up row pairs in R4 blocks.
    float gate = BFloat(Dot<true>(record, input, 2 * column, view.hidden));
    float up = BFloat(Dot<true>(record, input, 2 * column + 1, view.hidden));
    if (threadIdx.x % 32 == 0) {
        if (limit > 0) { gate = fminf(gate, limit); up = fmaxf(-limit, fminf(up, limit)); }
        const float value = __fmul_rn(__fmul_rn(gate / (1.f + expf(-gate)), up), scores[route]);
        activation[size_t(route) * view.inter + column] = __float2bfloat16_rn(value);
    }
}

__global__ void Down(const block_q8_K *activation, float *output, FastllmCudaMoeGGUFCacheView view) {
    const int column = blockIdx.x * 4 + threadIdx.x / 32, route = blockIdx.y;
    if (column >= view.hidden || view.routeSlots[route] < 0) return;
    const uint8_t *weight = view.records + size_t(view.routeSlots[route]) * view.recordStride + view.downOffset;
    const float value = Dot<false>(weight, activation + size_t(route) * (view.inter / 256), column, view.inter);
    if (threadIdx.x % 32 == 0)
        output[size_t(route) * view.hidden + column] = fastllm::cuda::dsv41_cache::BFloat(value);
}
} // namespace v41_gguf_cache

bool FastllmCudaMoeV41GGUFCacheCompute(const fastllm::Data &input, fastllm::Data &activation,
        const FastllmCudaMoeGGUFCacheView &view, const float *scores, int topk,
        float swigluLimit, float *perExpert) {
    if (input.dataType != fastllm::BFLOAT16 || input.dims.size() != 2 || !input.cudaData ||
        input.dims[0] < 1 || input.dims[0] > 8 || input.dims[1] != view.hidden ||
        view.hidden <= 0 || view.inter <= 0 || view.hidden % 256 || view.inter % 256 || topk < 1 || topk > 16 ||
        view.gateType != GGML_TYPE_Q2_K_R4 || view.downType != GGML_TYPE_Q4_K_R4 ||
        !scores || !perExpert || !view.workspace || !view.routeSlots || !view.records) return false;
    const int rows = input.dims[0], routes = rows * topk;
    const size_t inputBytes = Align16(size_t(rows) * (view.hidden / 256) * sizeof(block_q8_K));
    const size_t gateBytes = Align16(size_t(routes) * (view.inter / 256) * sizeof(block_q8_K));
    if (view.workspaceBytes < inputBytes + gateBytes) return false;
    auto *qInput = static_cast<block_q8_K *>(view.workspace);
    auto *qGate = reinterpret_cast<block_q8_K *>(static_cast<uint8_t *>(view.workspace) + inputBytes);
    activation.dataType = fastllm::BFLOAT16;
    activation.Resize({routes, view.inter});
    activation.ToDevice(fastllm::DataDevice::CUDA, input.dataDeviceIds, false);
    activation.Allocate(false);
    v41_gguf_cache::Quantize<<<dim3(view.hidden / 256, rows), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(input.cudaData), qInput, view.hidden);
    v41_gguf_cache::Gate<<<dim3((view.inter + 3) / 4, routes), 128, 0, cudaStreamPerThread>>>(
        qInput, static_cast<__nv_bfloat16 *>(activation.cudaData), view, scores, topk, swigluLimit);
    if (!FastllmCudaDeepSeekV41QuantizeActivation(activation, activation)) return false;
    v41_gguf_cache::Quantize<<<dim3(view.inter / 256, routes), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(activation.cudaData), qGate, view.inter);
    v41_gguf_cache::Down<<<dim3((view.hidden + 3) / 4, routes), 128, 0, cudaStreamPerThread>>>(qGate, perExpert, view);
    return cudaGetLastError() == cudaSuccess;
}
