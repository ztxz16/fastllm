#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/naive-n05-kv.cuh"
#include "naive-n05-dsa-fused-attention.cuh"
#include "naivert/attn_dsa/body.cuh"
#include "utils/utils.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace fastllm_dsa_fused {
template <bool Fp8Key> __global__ __launch_bounds__(384, 1) void Verify(const __grid_constant__ Args args) {
    extern __shared__ __align__(1024) unsigned char memory[];
    Body<Fp8Key>::run(args, *reinterpret_cast<typename Body<Fp8Key>::SharedStorage *>(memory));
}
// Initialization is per host thread/device, never mutable request scratch.
template <bool Fp8Key> int GridSize() {
    if (FastllmCudaRuntimeArch() != 103)
        return 0;
    static thread_local std::unordered_map<int, int> grids;
    int device = FastllmCudaGetDevice();
    auto found = grids.find(device);
    if (found != grids.end())
        return found->second;
    if (FastllmCudaGraphIsCapturingFast())
        return 0;
    cudaDeviceProp prop{};
    int occupancy = 0;
    const int shared = sizeof(typename Body<Fp8Key>::SharedStorage);
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || !prop.cooperativeLaunch ||
        prop.multiProcessorCount < 128 || prop.sharedMemPerBlockOptin < shared ||
        cudaFuncSetAttribute(Verify<Fp8Key>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) !=
            cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, Verify<Fp8Key>, 384, shared) !=
            cudaSuccess ||
        occupancy < 1) {
        cudaGetLastError();
        grids[device] = 0;
        return 0;
    }
    return grids[device] = prop.multiProcessorCount;
}
} // namespace fastllm_dsa_fused

bool FastllmCudaNaiveFusedDSAVerify(const fastllm::Data &query, const fastllm::Data &indexQuery,
                                    const fastllm::Data &weights, const fastllm::Data &sink,
                                    const std::vector<FastllmNaiveVerifyIndexerItem> &items,
                                    const std::vector<FastllmNaiveDSARow> &rows, bool fp8Query, int heads,
                                    int kvHeads, int dim, int valueDim, FastllmNaiveDecodeScratch &scratch,
                                    fastllm::Data &output) {
    using namespace fastllm;
    using namespace fastllm_dsa_fused;
    const int count = rows.size();
    // The cooperative core processes at most eight rows at a time. Three
    // short/medium-context tiles benefit from the existing batch-wide pipeline.
    // Preserve the cooperative core for larger capacities: scoring a long
    // context dominates there, and its fused indexer remains faster. Capacity
    // is static across graph replays; no host readback of live lengths is needed.
    const bool batchPipeline = count >= 24 && count < 32 && std::all_of(rows.begin(), rows.end(),
        [](const auto &row) { return row.capacity <= 32768; });
    if (!fp8Query || heads != 8 || kvHeads != 1 || dim != 192 || valueDim != 128 || count < 1 ||
        count >= 32 || batchPipeline || items.empty() || query.dataType != BFLOAT16 || indexQuery.dataType != BFLOAT16 ||
        weights.dataType != BFLOAT16 || query.Count(0) != size_t(count) * 1536 ||
        indexQuery.Count(0) != size_t(count) * 2048 || weights.Count(0) != size_t(count) * 16 ||
        query.dataDevice != CUDA || indexQuery.dataDevice != CUDA || weights.dataDevice != CUDA ||
        !query.cudaData || !indexQuery.cudaData || !weights.cudaData || (uintptr_t)query.cudaData % 16 ||
        (uintptr_t)indexQuery.cudaData % 16 || (uintptr_t)weights.cudaData % 16 ||
        (!sink.dims.empty() &&
         (sink.dataType != FLOAT32 || sink.Count(0) != 8 || sink.dataDevice != CUDA || !sink.cudaData)))
        return false;
    // Validate every request before allocation or GPU writes: fallback is all-or-nothing.
    const bool fp8Key = items.front().key && items.front().key->kvCacheFp8Indexer;
    const int stride = fp8Key ? 264 : 320;
    int next = 0;
    for (const auto &item : items) {
        if (item.offset != next || item.rows <= 0 || next + item.rows > count || !item.key || !item.indices ||
            item.indices->dataType != INT32 || !item.indices->cudaData ||
            item.indices->Count(0) != size_t(item.rows) * 2048 || item.key->dataType != BFLOAT16 ||
            item.key->kvCacheFp8Indexer != fp8Key || item.key->dims.size() != 3 ||
            item.key->dims[2] != stride || item.key->dataDevice != CUDA)
            return false;
        for (int i = next; i < next + item.rows; ++i) {
            const auto &r = rows[i];
            if (r.fp8 || r.fp4 || r.keyStride != stride || r.verifyOffset >= 0 || r.capacity < 1 ||
                r.capacity > AttnDsaScratchLayout::kMaxSeqLen || r.length < 0 || r.length > r.capacity ||
                (uintptr_t)r.key % 16 || (uintptr_t)r.value % 16 || r.pageLen < 0 ||
                ((r.keyPages || r.valuePages) && !r.pageLen) || (!r.cache && (!r.key || !r.value)) ||
                (r.pageLen && (r.pageLen % 8 || (!r.cache && (!r.keyPages || !r.valuePages)))) ||
                (r.indices && r.selectedCount != 2048))
                return false;
            if (r.key != rows[next].key || r.value != rows[next].value || r.cache != rows[next].cache ||
                r.pageLen != rows[next].pageLen || r.keyPages != rows[next].keyPages ||
                r.valuePages != rows[next].valuePages)
                return false;
        }
        next += item.rows;
    }
    if (next != count)
        return false;
    int grid = fp8Key ? GridSize<true>() : GridSize<false>();
    if (!grid)
        return false;
    constexpr size_t bytes = AttnDsaScratchLayout::kBytes;
    if (FastllmCudaGraphIsCapturingFast() &&
        (scratch.dsaFused.expansionSize * scratch.dsaFused.unitSize < bytes || !scratch.dsaFused.cudaData ||
         output.expansionSize * output.unitSize < size_t(count) * 1024 * 2 || !output.cudaData))
        return false;
    scratch.dsaFused.dataType = INT32;
    scratch.dsaFused.UpdateUnitSize();
    scratch.dsaFused.Resize({int((bytes + 3) / 4)});
    scratch.dsaFused.ToDevice(CUDA, query.dataDeviceIds, false);
    scratch.dsaFused.Allocate(false);
    output.dataType = BFLOAT16;
    output.UpdateUnitSize();
    output.Resize({1, count, 1024});
    output.ToDevice(CUDA, query.dataDeviceIds, false);
    output.Allocate(false);
    for (const auto &item : items)
        for (int first = 0; first < item.rows; first += 8) {
            Args args{};
            const int offset = item.offset + first;
            args.query = (const __nv_bfloat16 *)query.cudaData + size_t(offset) * 1536;
            args.indexQuery = (const __nv_bfloat16 *)indexQuery.cudaData + size_t(offset) * 2048;
            args.weights = (const __nv_bfloat16 *)weights.cudaData + size_t(offset) * 16;
            args.sink = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
            args.indexKey = FastllmNaiveKV(*item.key);
            args.indexCache = item.cache;
            args.indexStride = item.key->dims[2];
            args.activeRows = std::min(8, item.rows - first);
            args.output = (__nv_bfloat16 *)output.cudaData + size_t(offset) * 1024;
            args.workspace = (unsigned char *)scratch.dsaFused.cudaData;
            for (int i = 0; i < args.activeRows; ++i)
                args.rows[i] = rows[offset + i];
            void *params[] = {&args};
            const void *kernel = fp8Key ? (const void *)Verify<true> : (const void *)Verify<false>;
            size_t shared = fp8Key ? sizeof(Body<true>::SharedStorage) : sizeof(Body<false>::SharedStorage);
            auto error = cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(384), params, shared,
                                                     cudaStreamPerThread);
            AssertInFastLLM(error == cudaSuccess,
                            std::string("Fused DSA launch failed: ") + cudaGetErrorString(error));
        }
    return true;
}
