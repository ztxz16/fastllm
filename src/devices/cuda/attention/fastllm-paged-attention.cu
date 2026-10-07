#include "fastllm-attention-workspace.cuh"
#include "fastllm-paged-attention-params.cuh"

#ifdef FASTLLM_ENABLE_FLASHINFER
static bool FastllmCanUseFlashInferPagedKV(const fastllm::Data &cache) {
    if (!FastllmCudaFlashInferSupported()) return false;
    if (cache.pagedKVCacheData != nullptr &&
        cache.pagedKVCacheData->dataType == fastllm::DataType::FP4_E2M1) {
#if CUDA_VERSION >= 12080
        return FastllmCudaRuntimeArch() >= 80;
#else
        return false;
#endif
    }
    return true;
}
#endif

bool FastllmCudaHalfPagedAttention(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &output, int group, float scale, bool inited) {
#ifndef FASTLLM_ENABLE_FLASHINFER
    return FastllmCudaHalfPagedAttentionFastllmFallback(q, k, v, output, group, scale);
#else
    if (!FastllmCanUseFlashInferPagedKV(k)) {
        return FastllmCudaHalfPagedAttentionFastllmFallback(q, k, v, output, group, scale);
    }
    using namespace flashinfer;
    FlashInferWorkSpaceManager& workspace = getFastllmFlashInferWorkSpace();
    // 检查是否是 paged KV cache
    if (!k.isPagedKVCache || !v.isPagedKVCache) {
        printf("DoCudaAttentionPaged: k and v must be paged KV cache.\n");
        exit(0);
    }

    // 获取基本参数
    int q0 = q.dims[0];  // num_qo_heads
    int q1 = q.dims[1];  // seq_len (query length)
    int q2 = q.dims[2];  // head_dim_qk
    int k0 = k.dims[0];  // num_kv_heads
    int v2 = v.dims[2];  // head_dim_vo
    // 计算 batch size: q0 = batch * num_qo_heads_per_batch
    uint32_t num_qo_heads_per_batch = group * k0;
    if (q0 % num_qo_heads_per_batch != 0) {
        printf("DoCudaAttentionPaged: q0 (%d) is not divisible by num_qo_heads_per_batch (%u)\n",
                       q0, num_qo_heads_per_batch);
        exit(0);
    }
    uint32_t batch_size = q0 / num_qo_heads_per_batch;

    // 获取 paged KV cache 信息（k 和 v 各自有独立的 cache）
    fastllm::Data *pagedKVCacheK = k.pagedKVCacheData;
    fastllm::Data *pagedKVCacheV = v.pagedKVCacheData;
    if (pagedKVCacheK == nullptr || pagedKVCacheV == nullptr) {
        printf("DoCudaAttentionPaged: pagedKVCacheData is nullptr\n");
        exit(0);
    }

    int pageLen = k.pageLen;
    int numHeads = pagedKVCacheK->dims[2];  // [maxPages, pageLen, numHeads, headDim]
    int headDim = pagedKVCacheK->dims[3];
    int valueHeadDim = pagedKVCacheV->dims[3];

    // 检查数据类型
    if (q.dataType != fastllm::DataType::FLOAT16 && q.dataType != fastllm::DataType::BFLOAT16) {
        printf("DoCudaAttentionPaged: Only FLOAT16 and BFLOAT16 are supported for paged attention, got q.dataType=%d\n", (int)q.dataType);
        return true;
    }
    if (output.dataType != q.dataType) {
        printf("DoCudaAttentionPaged: q/output dataType mismatch\n");
        return true;
    }
    if (pagedKVCacheK->dataType != pagedKVCacheV->dataType) {
        printf("DoCudaAttentionPaged: paged KV cache dataType mismatch\n");
        return true;
    }
    if (pagedKVCacheK->dataType != q.dataType && pagedKVCacheK->dataType != fastllm::DataType::FP8_E4M3 &&
        pagedKVCacheK->dataType != fastllm::DataType::FP4_E2M1) {
        printf("DoCudaAttentionPaged: unsupported KV cache dataType=%d\n", (int)pagedKVCacheK->dataType);
        return true;
    }

    if (q2 != headDim || v2 != valueHeadDim || headDim != valueHeadDim) {
        printf("DoCudaAttentionPaged: head_dim mismatch, got q2=%d, v2=%d, kHeadDim=%d, vHeadDim=%d\n",
                       q2, v2, headDim, valueHeadDim);
        exit(0);
    }

    // 检查指针有效性
    if (q.cudaData == nullptr || output.cudaData == nullptr) {
        printf("DoCudaAttentionPaged: q or output cudaData is nullptr\n");
        exit(0);
    }

    if (pagedKVCacheK->cudaData == nullptr || pagedKVCacheV->cudaData == nullptr) {
        printf("DoCudaAttentionPaged: pagedKVCacheData cudaData is nullptr\n");
        exit(0);
    }

    // 为每个 batch 构造 indptr、indices 和 last_page_len
    // 目前假设所有 batch 共享相同的 page 索引（单 batch 场景）
    std::vector<uint32_t> indptr_host(batch_size + 1);
    std::vector<uint32_t> indices_host;
    std::vector<uint32_t> last_page_len_host(batch_size);

    // 从 k 的 pageIndex 获取信息
    int numPages = k.pageIndex.size();
    if (numPages == 0) {
        printf("DoCudaAttentionPaged: No pages in cache (pageIndex is empty)\n");
        exit(0);
    }

    indptr_host[0] = 0;
    for (int b = 0; b < batch_size; b++) {
        // 假设每个 batch 使用相同的 pages（单 batch 场景）
        int pagesPerBatch = numPages;
        indptr_host[b + 1] = indptr_host[b] + pagesPerBatch;

        // 复制 page indices
        for (int i = 0; i < pagesPerBatch; i++) {
            if (k.pageIndex[i] < 0 || k.pageIndex[i] >= pagedKVCacheK->dims[0]) {
                printf("DoCudaAttentionPaged: Invalid page index %d (maxPages=%d)\n",
                       k.pageIndex[i], (int)pagedKVCacheK->dims[0]);
                exit(0);
            }
            indices_host.push_back((uint32_t)k.pageIndex[i]);
        }

        // 设置最后一个 page 的长度
        last_page_len_host[b] = (uint32_t)k.lastPageLen;
    }

    // 分配 GPU 内存并拷贝数据
    uint32_t *indptr_gpu = nullptr;
    uint32_t *indices_gpu = nullptr;
    uint32_t *last_page_len_gpu = nullptr;

    indptr_gpu = (uint32_t*)FastllmCudaMalloc((batch_size + 1) * sizeof(uint32_t));
    indices_gpu = (uint32_t*)FastllmCudaMalloc(indices_host.size() * sizeof(uint32_t));
    last_page_len_gpu = (uint32_t*)FastllmCudaMalloc(batch_size * sizeof(uint32_t));

    cudaMemcpy(indptr_gpu, indptr_host.data(), (batch_size + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(indices_gpu, indices_host.data(), indices_host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(last_page_len_gpu, last_page_len_host.data(), batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // 根据 q1 判断是 prefill 还是 decode
    cudaStream_t stream = nullptr;
    float *tmp_s = nullptr;
    cudaError_t status = cudaSuccess;

    // Decode 阶段需要的临时内存（在函数末尾统一释放）
    uint32_t *request_indices_gpu = nullptr;
    uint32_t *kv_tile_indices_gpu = nullptr;
    uint32_t *kv_chunk_size_ptr_gpu = nullptr;

    // 定义一个 lambda 来执行 prefill 逻辑，模板化 Q/KV 类型
    auto runPrefill = [&]<typename QType, typename KVType>() {
        QType *qd = (QType*)q.cudaData;
        QType *od = (QType*)output.cudaData;
        KVType *pagedKVCacheDataK = (KVType*)pagedKVCacheK->cudaData;
        KVType *pagedKVCacheDataV = (KVType*)pagedKVCacheV->cudaData;

        // 构造 paged_kv_t 结构
        // fastllm 的布局是 [maxPages, pageLen, numHeads, headDim]，这是 NHD 布局
        paged_kv_t<KVType, uint32_t> paged_kv(
            numHeads, pageLen, headDim, batch_size,
            QKVLayout::kNHD, pagedKVCacheDataK, pagedKVCacheDataV,
            indices_gpu, indptr_gpu, last_page_len_gpu, nullptr
        );

        QType *tmp_v = nullptr;
        // Prefill 阶段
        uint32_t *q_indptr_gpu = nullptr;
        std::vector<uint32_t> q_indptr_host(batch_size + 1);
        for (uint32_t i = 0; i <= batch_size; i++) {
            q_indptr_host[i] = i * q1;
        }

        q_indptr_gpu = (uint32_t*)FastllmCudaMalloc((batch_size + 1) * sizeof(uint32_t));
        cudaMemcpy(q_indptr_gpu, q_indptr_host.data(), (batch_size + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);

        uint32_t total_num_rows = q_indptr_host[batch_size];
        thread_local static std::map<int, PrefillPlanInfo> plan_info_map;
        thread_local static std::map<int, bool> plan_inited_map;
        int current_device_id = -1;
        cudaGetDevice(&current_device_id);
        if (!inited || !plan_inited_map[current_device_id]) {
            std::lock_guard<std::mutex> workspace_guard(workspace.plan_mutex);
            size_t floatBytes = 0, intBytes = 0;
            checkCudaErrors("FlashInfer prefill workspace size",
                PrefillPlanWorkspaceSize<uint32_t>(floatBytes, intBytes,
                    q_indptr_host.data(), indptr_host.data(), total_num_rows,
                    batch_size, num_qo_heads_per_batch, numHeads, headDim, headDim,
                    pageLen, false, sizeof(QType), -1, -1, false, 0, 0, stream));
            workspace.EnsureIntCapacity(intBytes);
            cudaError_t plan_status = PrefillPlan<uint32_t>(
                workspace.d_float_workspace, workspace.float_workspace_size, workspace.d_int_workspace, workspace.h_page_locked_int_workspace,
                workspace.int_workspace_size, plan_info_map[current_device_id], q_indptr_host.data(), indptr_host.data(),
                total_num_rows, batch_size, num_qo_heads_per_batch, numHeads, headDim, headDim,
                pageLen, /*enable_cuda_graph=*/false, /*sizeof_dtype_o=*/sizeof(QType),
                /*window_left=*/-1, /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
                /*num_colocated_ctas=*/0, /*uniform_q_len=*/0, stream);

            if (plan_status != cudaSuccess) {
                printf("DoCudaAttentionPaged: PrefillPlan failed: %s\n", cudaGetErrorString(plan_status));
                cudaStreamSynchronize(stream);
                FastllmCudaFree(q_indptr_gpu);
                exit(0);
            }
            plan_inited_map[current_device_id] = true;
        }
        PrefillPlanInfo &plan_info = plan_info_map[current_device_id];

        uint32_t q_stride_n = (q.dims.size() >= 2 && q.strides.size() >= 2) ? q.strides[1] : q2;
        uint32_t q_stride_h = (q.dims.size() >= 3 && q.strides.size() >= 1) ? q.strides[0] : (q1 * q2);

        FastllmPagedPrefillParams<QType, KVType> prefill_params(
            qd, paged_kv, nullptr, q_indptr_gpu, nullptr, nullptr,
            od, nullptr, nullptr,
            num_qo_heads_per_batch, q_stride_n, q_stride_h,
            -1, 0.0f, scale, 1.0f, 10000.0f
        );

        FastllmConfigureFP4PagedParams(prefill_params, pageLen, numHeads, headDim);
        prefill_params.request_indices = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(workspace.d_int_workspace) + plan_info.request_indices_offset);
        prefill_params.qo_tile_indices = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(workspace.d_int_workspace) + plan_info.qo_tile_indices_offset);
        prefill_params.kv_tile_indices = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(workspace.d_int_workspace) + plan_info.kv_tile_indices_offset);
        prefill_params.o_indptr = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(workspace.d_int_workspace) + plan_info.o_indptr_offset);
        prefill_params.kv_chunk_size_ptr = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(workspace.d_int_workspace) + plan_info.kv_chunk_size_ptr_offset);
        prefill_params.padded_batch_size = plan_info.padded_batch_size;
        prefill_params.max_total_num_rows = plan_info.total_num_rows;

        if (plan_info.split_kv) {
            prefill_params.merge_indptr = reinterpret_cast<uint32_t*>(
                static_cast<uint8_t*>(workspace.d_int_workspace) + plan_info.merge_indptr_offset);
            tmp_v = reinterpret_cast<QType*>(
                static_cast<uint8_t*>(workspace.d_float_workspace) + plan_info.v_offset);
            tmp_s = reinterpret_cast<float*>(
                static_cast<uint8_t*>(workspace.d_float_workspace) + plan_info.s_offset);
        }

        bool enable_pdl = false;
        status = FastllmDispatchPagedPrefillByHeadDim(
            headDim, (long)plan_info.cta_tile_q, prefill_params, tmp_v, tmp_s, enable_pdl, stream,
            "DoCudaAttentionPaged");

        FastllmCudaFree(q_indptr_gpu);
        ((fastllm::Data*)&output)->Resize({output.dims[1], output.dims[0], output.dims[2]});
        FastllmCudaPermute(*((fastllm::Data*)&output), {1, 0, 2});
    };

    if (q.dataType == fastllm::DataType::BFLOAT16) {
        if (pagedKVCacheK->dataType == fastllm::DataType::FP8_E4M3) {
            runPrefill.template operator()<__nv_bfloat16, __nv_fp8_e4m3>();
#if CUDA_VERSION >= 12080
        } else if (pagedKVCacheK->dataType == fastllm::DataType::FP4_E2M1) {
            runPrefill.template operator()<__nv_bfloat16, __nv_fp4x2_e2m1>();
#endif
        } else {
            runPrefill.template operator()<__nv_bfloat16, __nv_bfloat16>();
        }
    } else {
        if (pagedKVCacheK->dataType == fastllm::DataType::FP8_E4M3) {
            runPrefill.template operator()<half, __nv_fp8_e4m3>();
#if CUDA_VERSION >= 12080
        } else if (pagedKVCacheK->dataType == fastllm::DataType::FP4_E2M1) {
            runPrefill.template operator()<half, __nv_fp4x2_e2m1>();
#endif
        } else {
            runPrefill.template operator()<half, half>();
        }
    }

    // 清理 GPU 内存
    FastllmCudaFree(indptr_gpu);
    FastllmCudaFree(indices_gpu);
    FastllmCudaFree(last_page_len_gpu);

    // 清理 decode 阶段临时分配的内存
    if (request_indices_gpu != nullptr) {
        FastllmCudaFree(request_indices_gpu);
    }
    if (kv_tile_indices_gpu != nullptr) {
        FastllmCudaFree(kv_tile_indices_gpu);
    }
    if (kv_chunk_size_ptr_gpu != nullptr) {
        FastllmCudaFree(kv_chunk_size_ptr_gpu);
    }

    if (status != cudaSuccess) {
        printf("DoCudaAttentionPaged: FlashInfer error: %s\n", cudaGetErrorString(status));
        exit(0);
    }

    DeviceSync();
    return true;
#endif
}

#ifdef FASTLLM_ENABLE_FLASHINFER
namespace {

// FlashInfer's host PrefillPlan builds these arrays from qo_indptr and
// kv_indptr.  In CUDA graph decode both indptr buffers are stable, but their
// contents (most importantly the per-request page counts) can change between
// replays.  Keep the graph topology and workspace layout fixed and rebuild only
// the device-resident schedule before the attention nodes consume it.
struct FastllmFlashInferDecodePlanParams {
    const uint32_t *qoIndptr;
    const uint32_t *kvIndptr;
    uint32_t *requestIndices;
    uint32_t *qoTileIndices;
    uint32_t *kvTileIndices;
    uint32_t *mergeIndptr;
    uint32_t *oIndptr;
    uint32_t *kvChunkSize;
    uint32_t *totalNumRows;
    bool *blockValidMask;
    uint32_t *status;
    uint32_t batchSize;
    uint32_t groupSize;
    uint32_t ctaTileQ;
    uint32_t pageSize;
    uint32_t maxBatchSizeIfSplit;
    uint32_t paddedBatchSize;
    uint32_t maxTotalNumRows;
    int32_t windowLeft;
};

__device__ __forceinline__ uint64_t FastllmFlashInferCeilDivU64(
        uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

__global__ void FastllmFlashInferUpdateDecodePlanKernel(
        FastllmFlashInferDecodePlanParams params) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (params.status == nullptr) {
        return;
    }
    *params.status = 1;
    if (params.requestIndices == nullptr || params.qoTileIndices == nullptr ||
        params.kvTileIndices == nullptr || params.mergeIndptr == nullptr ||
        params.oIndptr == nullptr || params.kvChunkSize == nullptr ||
        params.totalNumRows == nullptr || params.blockValidMask == nullptr) {
        return;
    }

    // Start from a fail-closed schedule.  If metadata is malformed or the
    // fixed graph capacity is exceeded, every attention CTA is masked before
    // any request/tile index can be consumed.
    *params.totalNumRows = 0;
    *params.kvChunkSize = params.pageSize == 0 ? 1 : params.pageSize;
    for (uint32_t i = 0; i < params.paddedBatchSize; ++i) {
        params.blockValidMask[i] = false;
    }
    for (uint32_t i = 0; i <= params.batchSize; ++i) {
        params.oIndptr[i] = 0;
    }
    for (uint32_t i = 0; i <= params.maxTotalNumRows; ++i) {
        params.mergeIndptr[i] = 0;
    }

    if (params.qoIndptr == nullptr || params.kvIndptr == nullptr ||
        params.batchSize == 0 ||
        params.groupSize == 0 || params.ctaTileQ == 0 ||
        params.pageSize == 0 || params.paddedBatchSize == 0 ||
        params.maxBatchSizeIfSplit == 0 || params.qoIndptr[0] != 0 ||
        params.kvIndptr[0] != 0) {
        return;
    }

    uint32_t totalNumRows = params.qoIndptr[params.batchSize];
    if (totalNumRows > params.maxTotalNumRows) {
        return;
    }

    uint64_t maxEffectiveKvLen = 1;
    uint64_t windowPages = 0;
    if (params.windowLeft >= 0) {
        windowPages = FastllmFlashInferCeilDivU64(
            uint64_t(params.windowLeft) + params.ctaTileQ, params.pageSize);
    }
    for (uint32_t request = 0; request < params.batchSize; ++request) {
        uint32_t qoBegin = params.qoIndptr[request];
        uint32_t qoEnd = params.qoIndptr[request + 1];
        uint32_t kvBegin = params.kvIndptr[request];
        uint32_t kvEnd = params.kvIndptr[request + 1];
        if (qoEnd < qoBegin || kvEnd < kvBegin) {
            return;
        }
        uint64_t kvLen = uint64_t(kvEnd - kvBegin);
        uint64_t effectiveKvLen =
            params.windowLeft >= 0 ? min(kvLen, windowPages) : kvLen;
        maxEffectiveKvLen = max(maxEffectiveKvLen, effectiveKvLen);
    }

    // Match PrefillBinarySearchKVChunkSize exactly.  The graph-mode planner
    // always uses split-KV; paddedBatchSize is fixed by the captured shape.
    uint64_t low = max(uint64_t(128 / params.pageSize), uint64_t(1));
    uint64_t high = maxEffectiveKvLen;
    while (low < high) {
        uint64_t mid = (low + high) / 2;
        uint64_t newBatchSize = 0;
        for (uint32_t request = 0; request < params.batchSize; ++request) {
            uint64_t qoLen = uint64_t(params.qoIndptr[request + 1] -
                                       params.qoIndptr[request]);
            uint64_t packedQoLen = qoLen * params.groupSize;
            uint64_t numTilesQ = FastllmFlashInferCeilDivU64(
                packedQoLen, params.ctaTileQ);
            uint64_t kvLen = uint64_t(params.kvIndptr[request + 1] -
                                      params.kvIndptr[request]);
            uint64_t effectiveKvLen =
                params.windowLeft >= 0 ? min(kvLen, windowPages) : kvLen;
            uint64_t numChunksKv = FastllmFlashInferCeilDivU64(
                max(effectiveKvLen, uint64_t(1)), mid);
            newBatchSize += numTilesQ * numChunksKv;
        }
        if (newBatchSize > params.maxBatchSizeIfSplit) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    uint64_t chunkSizeTokens = low * params.pageSize;
    if (chunkSizeTokens > UINT32_MAX) {
        return;
    }

    uint64_t scheduleOffset = 0;
    uint64_t mergeOffset = 0;
    uint64_t outputOffset = 0;
    uint32_t rowOffset = 0;
    params.mergeIndptr[0] = 0;
    params.oIndptr[0] = 0;
    for (uint32_t request = 0; request < params.batchSize; ++request) {
        uint32_t qoLen = params.qoIndptr[request + 1] - params.qoIndptr[request];
        uint64_t packedQoLen = uint64_t(qoLen) * params.groupSize;
        uint64_t numTilesQ = FastllmFlashInferCeilDivU64(
            packedQoLen, params.ctaTileQ);
        uint64_t kvLen = uint64_t(params.kvIndptr[request + 1] -
                                  params.kvIndptr[request]);
        uint64_t effectiveKvLen =
            params.windowLeft >= 0 ? min(kvLen, windowPages) : kvLen;
        uint64_t numChunksKv = FastllmFlashInferCeilDivU64(
            max(effectiveKvLen, uint64_t(1)), low);

        if (scheduleOffset + numTilesQ * numChunksKv > params.paddedBatchSize ||
            uint64_t(rowOffset) + qoLen > params.maxTotalNumRows ||
            outputOffset + uint64_t(qoLen) * numChunksKv > UINT32_MAX) {
            // The arrays were initialized to a fully masked schedule above.
            // Clear any entries tentatively enabled by preceding requests.
            for (uint32_t i = 0; i < params.paddedBatchSize; ++i) {
                params.blockValidMask[i] = false;
            }
            return;
        }

        for (uint64_t qoTile = 0; qoTile < numTilesQ; ++qoTile) {
            for (uint64_t kvTile = 0; kvTile < numChunksKv; ++kvTile) {
                uint32_t index = uint32_t(scheduleOffset++);
                params.requestIndices[index] = request;
                params.qoTileIndices[index] = uint32_t(qoTile);
                params.kvTileIndices[index] = uint32_t(kvTile);
                params.blockValidMask[index] = true;
            }
        }

        for (uint32_t row = 0; row < qoLen; ++row) {
            mergeOffset += numChunksKv;
            if (mergeOffset > UINT32_MAX) {
                for (uint32_t i = 0; i < params.paddedBatchSize; ++i) {
                    params.blockValidMask[i] = false;
                }
                return;
            }
            params.mergeIndptr[++rowOffset] = uint32_t(mergeOffset);
        }
        outputOffset += uint64_t(qoLen) * numChunksKv;
        params.oIndptr[request + 1] = uint32_t(outputOffset);
    }

    *params.kvChunkSize = uint32_t(chunkSizeTokens);
    *params.totalNumRows = totalNumRows;
    *params.status = 0;
}

cudaError_t FastllmFlashInferUpdateDecodePlan(
        const FastllmFlashInferDecodePlanParams &params, cudaStream_t stream) {
    FastllmFlashInferUpdateDecodePlanKernel<<<1, 1, 0, stream>>>(params);
    return cudaGetLastError();
}

bool FastllmFlashInferCaptureId(cudaStream_t stream,
                               unsigned long long &captureId) {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaError_t state = cudaStreamGetCaptureInfo(stream, &captureStatus,
                                                 &captureId);
    if (state != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return captureStatus == cudaStreamCaptureStatusActive;
}

void FastllmFlashInferAppendPointerKey(std::vector<uint32_t> &key,
                                      const void *pointer) {
    uint64_t value = uint64_t(reinterpret_cast<uintptr_t>(pointer));
    key.push_back(uint32_t(value));
    key.push_back(uint32_t(value >> 32));
}

} // namespace
#endif

bool FastllmCudaHalfPagedAttentionBatch(fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches, fastllm::Data &qSizes, fastllm::Data &pageSizes, fastllm::Data &pageIndexs, fastllm::Data &lastPageLens, fastllm::Data &output, int group, float scale, int attentionType, bool inited, bool sync, bool enableCudaGraph, int flashInferCudaGraph, int windowLeft) {
#ifndef FASTLLM_ENABLE_FLASHINFER
    fastllm::AssertInFastLLM(windowLeft < 0,
                             "Sliding-window paged attention requires FlashInfer.\n");
    bool ok = FastllmCudaHalfPagedAttentionBatchFastllmFallback(
        q, kCaches, vCaches, qSizes, pageSizes, pageIndexs, lastPageLens, output, group, scale);
    if (sync) {
        DeviceSync();
    }
    return ok;
#else
    if (!FastllmCanUseFlashInferPagedKV(kCaches)) {
        fastllm::AssertInFastLLM(windowLeft < 0,
                                 "Sliding-window paged attention requires FlashInfer support on this GPU.\n");
        bool ok = FastllmCudaHalfPagedAttentionBatchFastllmFallback(
            q, kCaches, vCaches, qSizes, pageSizes, pageIndexs, lastPageLens, output, group, scale);
        if (sync) {
            DeviceSync();
        }
        return ok;
    }
    using namespace flashinfer;
    FlashInferWorkSpaceManager& workspace = getFastllmFlashInferWorkSpace();

    // 获取基本参数
    int q0 = q.dims[0];  // total num_qo_heads across all batches
    int q1 = q.dims[1];  // seq_len (should be 1 for decode)
    int q2 = q.dims[2];  // head_dim_qk
    int k0 = kCaches.dims[0];  // num_kv_heads
    int v2 = vCaches.dims[2];  // head_dim_vo

    // 计算每个 batch 的 Q heads 数
    uint32_t num_qo_heads_per_batch = group * k0;
    if (q0 % num_qo_heads_per_batch != 0) {
        printf("FastllmCudaHalfPagedAttentionBatch: q0 (%d) is not divisible by num_qo_heads_per_batch (%u)\n",
               q0, num_qo_heads_per_batch);
        exit(0);
    }

    // 从 qSizes 获取 batch size
    int32_t *qSizesData = (int32_t*)qSizes.cudaData;
    if (qSizesData == nullptr) {
        printf("FastllmCudaHalfPagedAttentionBatch: qSizes.cudaData is nullptr\n");
        exit(0);
    }

    if (qSizes.dims.empty() || qSizes.dims[0] <= 1) {
        printf("FastllmCudaHalfPagedAttentionBatch: invalid qSizes shape\n");
        return false;
    }

    // qSizes 的长度是 batch + 1，所以 batch_size = qSizes.dims[0] - 1
    uint32_t batch_size = qSizes.dims[0] - 1;
    if (batch_size == 0) {
        printf("FastllmCudaHalfPagedAttentionBatch: batch_size is 0\n");
        exit(0);
    }

    // 获取 paged KV cache 信息
    fastllm::Data *pagedKVCacheK = kCaches.pagedKVCacheData;
    fastllm::Data *pagedKVCacheV = vCaches.pagedKVCacheData;
    if (pagedKVCacheK == nullptr || pagedKVCacheV == nullptr) {
        printf("FastllmCudaHalfPagedAttentionBatch: pagedKVCacheData is nullptr\n");
        exit(0);
    }

    int pageLen = kCaches.pageLen;
    int numHeads = pagedKVCacheK->dims[2];  // [maxPages, pageLen, numHeads, headDim]
    int headDim = pagedKVCacheK->dims[3];
    int valueHeadDim = pagedKVCacheV->dims[3];
    bool useFlashInferCudaGraph = flashInferCudaGraph < 0 ? enableCudaGraph : (flashInferCudaGraph != 0);

    // 检查数据类型
    if (q.dataType != fastllm::DataType::FLOAT16 && q.dataType != fastllm::DataType::BFLOAT16) {
        printf("FastllmCudaHalfPagedAttentionBatch: Only FLOAT16 and BFLOAT16 are supported for paged attention, got q.dataType=%d\n", (int)q.dataType);
        return false;
    }
    if (output.dataType != q.dataType) {
        printf("FastllmCudaHalfPagedAttentionBatch: q/output dataType mismatch\n");
        return false;
    }
    if (pagedKVCacheK->dataType != pagedKVCacheV->dataType) {
        printf("FastllmCudaHalfPagedAttentionBatch: paged KV cache dataType mismatch\n");
        return false;
    }
    if (pagedKVCacheK->dataType != q.dataType && pagedKVCacheK->dataType != fastllm::DataType::FP8_E4M3 &&
        pagedKVCacheK->dataType != fastllm::DataType::FP4_E2M1) {
        printf("FastllmCudaHalfPagedAttentionBatch: unsupported KV cache dataType=%d\n", (int)pagedKVCacheK->dataType);
        return false;
    }

    if (q2 != headDim || v2 != valueHeadDim || headDim != valueHeadDim) {
        printf("FastllmCudaHalfPagedAttentionBatch: head_dim mismatch, got q2=%d, v2=%d, kHeadDim=%d, vHeadDim=%d\n",
               q2, v2, headDim, valueHeadDim);
        exit(0);
    }

    // 检查指针有效性
    if (q.cudaData == nullptr || output.cudaData == nullptr) {
        printf("FastllmCudaHalfPagedAttentionBatch: q or output cudaData is nullptr\n");
        exit(0);
    }

    if (pagedKVCacheK->cudaData == nullptr || pagedKVCacheV->cudaData == nullptr) {
        printf("FastllmCudaHalfPagedAttentionBatch: pagedKVCacheData cudaData is nullptr\n");
        exit(0);
    }

    // 从 pageSizes 和 pageIndexs 构造 indptr 和 indices
    int32_t *pageSizesData = (int32_t*)pageSizes.cudaData;
    int32_t *pageIndexsData = (int32_t*)pageIndexs.cudaData;
    int32_t *lastPageLensData = (int32_t*)lastPageLens.cudaData;

    if (pageSizesData == nullptr || pageIndexsData == nullptr || lastPageLensData == nullptr) {
        printf("FastllmCudaHalfPagedAttentionBatch: pageSizes, pageIndexs or lastPageLens cudaData is nullptr\n");
        exit(0);
    }

    cudaError_t status = cudaSuccess;

    // 定义一个 lambda 来执行 prefill 逻辑，模板化 Q/KV 类型
    auto runBatchPrefill = [&]<typename QType, typename KVType>() {
        QType *qd = (QType*)q.cudaData;
        QType *od = (QType*)output.cudaData;
        KVType *pagedKVCacheDataK = (KVType*)pagedKVCacheK->cudaData;
        KVType *pagedKVCacheDataV = (KVType*)pagedKVCacheV->cudaData;

        // 构造 paged_kv_t 结构
        paged_kv_t<KVType, uint32_t> paged_kv(
            numHeads, pageLen, headDim, batch_size,
            QKVLayout::kNHD, pagedKVCacheDataK, pagedKVCacheDataV,
            (uint32_t*)pageIndexsData, (uint32_t*)pageSizesData,
            (uint32_t*)lastPageLensData, nullptr
        );

        if (qSizes.cpuIntDatas.size() < batch_size + 1 ||
            pageSizes.cpuIntDatas.size() < batch_size + 1) {
            printf("FastllmCudaHalfPagedAttentionBatch: incomplete host indptr metadata\n");
            status = cudaErrorInvalidValue;
            return;
        }
        uint32_t total_num_rows = qSizes.cpuIntDatas[batch_size];
        // Graph-mode paged attention has a fixed query tensor shape, but both
        // query and KV indptr contents may change between replays.  The device
        // planner below already handles arbitrary qo lengths; restricting it
        // to one row per request left multi-token speculative verification with
        // a schedule frozen at capture time.  The product check validates that
        // the packed Q storage covers exactly the host-declared query rows,
        // independent of whether Q is laid out as [heads, total_rows, dim] or
        // [batch * heads, rows_per_request, dim].
        bool dynamic_decode_plan = enableCudaGraph && useFlashInferCudaGraph &&
                                   total_num_rows > 0 &&
                                   (uint64_t)q0 * (uint64_t)q1 ==
                                       (uint64_t)total_num_rows *
                                           num_qo_heads_per_batch &&
                                   qSizes.cpuIntDatas.size() >= batch_size + 1 &&
                                   pageSizes.cpuIntDatas.size() >= batch_size + 1;

        cudaStream_t stream = cudaStreamPerThread;
        static std::mutex plan_cache_mutex;
        int current_device_id = -1;
        cudaGetDevice(&current_device_id);

        std::vector<uint32_t> plan_key;
        plan_key.reserve(18 + (batch_size + 1) * 2);
        plan_key.push_back(total_num_rows);
        plan_key.push_back(batch_size);
        plan_key.push_back(num_qo_heads_per_batch);
        plan_key.push_back((uint32_t)numHeads);
        plan_key.push_back((uint32_t)headDim);
        plan_key.push_back((uint32_t)valueHeadDim);
        plan_key.push_back((uint32_t)pageLen);
        plan_key.push_back((uint32_t)sizeof(QType));
        plan_key.push_back((uint32_t)sizeof(KVType));
        plan_key.push_back(useFlashInferCudaGraph ? 1U : 0U);
        plan_key.push_back((uint32_t)q.dataType);
        plan_key.push_back((uint32_t)pagedKVCacheK->dataType);
        plan_key.push_back((uint32_t)windowLeft);
        plan_key.push_back(dynamic_decode_plan ? 1U : 0U);
        for (uint32_t i = 0; i <= batch_size; i++) {
            plan_key.push_back((uint32_t)qSizes.cpuIntDatas[i]);
        }
        if (dynamic_decode_plan) {
            // Mutable schedules must be private to the graph metadata buffers
            // that feed them.  Shape-only sharing would race when independent
            // models or batch states replay on the same device.
            FastllmFlashInferAppendPointerKey(plan_key, qSizesData);
            FastllmFlashInferAppendPointerKey(plan_key, pageSizesData);
        } else {
            for (uint32_t i = 0; i <= batch_size; i++) {
                plan_key.push_back((uint32_t)pageSizes.cpuIntDatas[i]);
            }
        }

        // Eager plans change when qo/kv indptr contents change, but keeping only
        // one entry per device makes mixed full/sliding models rebuild at every
        // layer boundary.  Use a stable signature for the cache slot and keep
        // the complete key above as that slot's current contents.
        std::vector<uint32_t> eager_slot_key = {
            batch_size,
            num_qo_heads_per_batch,
            (uint32_t)numHeads,
            (uint32_t)headDim,
            (uint32_t)valueHeadDim,
            (uint32_t)pageLen,
            (uint32_t)sizeof(QType),
            (uint32_t)sizeof(KVType),
            useFlashInferCudaGraph ? 1U : 0U,
            (uint32_t)q.dataType,
            (uint32_t)pagedKVCacheK->dataType,
            (uint32_t)windowLeft
        };

        // Different query lengths alternate during speculative decoding. Keep
        // each query distribution in its own bounded LRU slot; KV page changes
        // still rebuild that slot using the complete plan_key above.
        for (uint32_t i = 0; i <= batch_size; ++i) {
            eager_slot_key.push_back((uint32_t)qSizes.cpuIntDatas[i]);
        }

        struct PrefillPlanCacheEntry {
            PrefillPlanInfo plan_info;
            void *d_int_plan = nullptr;
            void *d_dynamic_status = nullptr;
            size_t int_plan_bytes = 0;
            uint32_t max_batch_size_if_split = 0;
            unsigned long long last_dynamic_capture_id = 0;
            bool has_dynamic_capture_id = false;
            bool dynamic_plan_validated = false;

            ~PrefillPlanCacheEntry() {
                if (d_int_plan != nullptr) {
                    FastllmCudaFree(d_int_plan);
                }
                if (d_dynamic_status != nullptr) {
                    FastllmCudaFree(d_dynamic_status);
                }
            }
        };

        auto planIntBytes = [&](const PrefillPlanInfo &info) -> size_t {
            auto endOf = [](int64_t offset, size_t bytes) -> size_t {
                return offset < 0 ? 0 : (size_t)offset + bytes;
            };
            size_t bytes = 0;
            bytes = std::max(bytes, endOf(info.request_indices_offset,
                                          (size_t)info.padded_batch_size * sizeof(uint32_t)));
            bytes = std::max(bytes, endOf(info.qo_tile_indices_offset,
                                          (size_t)info.padded_batch_size * sizeof(uint32_t)));
            bytes = std::max(bytes, endOf(info.kv_tile_indices_offset,
                                          (size_t)info.padded_batch_size * sizeof(uint32_t)));
            bytes = std::max(bytes, endOf(info.o_indptr_offset,
                                          (size_t)(batch_size + 1) * sizeof(uint32_t)));
            bytes = std::max(bytes, endOf(info.kv_chunk_size_ptr_offset, sizeof(uint32_t)));
            if (info.enable_cuda_graph) {
                bytes = std::max(bytes, endOf(info.total_num_rows_offset, sizeof(uint32_t)));
            }
            if (info.split_kv) {
                bytes = std::max(bytes, endOf(info.merge_indptr_offset,
                                              (size_t)(info.total_num_rows + 1) * sizeof(uint32_t)));
                bytes = std::max(bytes, endOf(info.block_valid_mask_offset,
                                              (size_t)info.padded_batch_size * sizeof(bool)));
            }
            return bytes;
        };

        PrefillPlanInfo plan_info;
        void *plan_int_base = workspace.d_int_workspace;
        std::shared_ptr<PrefillPlanCacheEntry> plan_entry;
        bool launch_dynamic_decode_plan = false;
        bool validate_dynamic_decode_plan = false;
        unsigned long long capture_id = 0;
        bool stream_is_capturing =
            FastllmFlashInferCaptureId(stream, capture_id);
        // PrefillPlan writes its integer schedule into the shared FlashInfer
        // workspace.  Different attention signatures overwrite that data.
        // Eager execution keeps one reusable private copy per stable signature
        // and host thread.  Graph execution retains one immutable copy per
        // captured signature because the graph stores its schedule pointers.
        struct EagerPlanCacheSlot {
            std::vector<uint32_t> plan_key;
            std::shared_ptr<PrefillPlanCacheEntry> entry;
            uint64_t last_use = 0;
        };
        struct EagerPlanDeviceCache {
            std::map<std::vector<uint32_t>, EagerPlanCacheSlot> slots;
            uint64_t use_counter = 0;
        };
        static thread_local std::map<int, EagerPlanDeviceCache> eager_plan_caches;
        static std::map<int, std::map<std::vector<uint32_t>, std::shared_ptr<PrefillPlanCacheEntry>>> graph_plan_cache;

        auto createPlan = [&](std::shared_ptr<PrefillPlanCacheEntry> entry)
                -> std::shared_ptr<PrefillPlanCacheEntry> {
            if (stream_is_capturing) {
                printf("FastllmCudaHalfPagedAttentionBatch: plan cache miss during CUDA stream capture.\n");
                status = cudaErrorStreamCaptureUnsupported;
                return nullptr;
            }
            // This mutex belongs to the per-device workspace, so TP ranks
            // on different GPUs never block each other while one rank waits
            // for its plan staging copies to finish.
            std::lock_guard<std::mutex> workspace_guard(workspace.plan_mutex);
            size_t floatBytes = 0, intBytes = 0;
            checkCudaErrors("FlashInfer batch prefill workspace size",
                PrefillPlanWorkspaceSize<uint32_t>(floatBytes, intBytes,
                    (uint32_t*)qSizes.cpuIntDatas.data(), (uint32_t*)pageSizes.cpuIntDatas.data(),
                    total_num_rows, batch_size, num_qo_heads_per_batch, numHeads, headDim, headDim,
                    pageLen, useFlashInferCudaGraph, sizeof(QType), windowLeft, -1, false, 0, 0, stream));
            workspace.EnsureIntCapacity(intBytes);
            PrefillPlanInfo created_plan_info;
            cudaError_t plan_status = PrefillPlan<uint32_t>(
                    workspace.d_float_workspace, workspace.float_workspace_size, workspace.d_int_workspace, workspace.h_page_locked_int_workspace,
                    workspace.int_workspace_size, created_plan_info,
                    (uint32_t*)qSizes.cpuIntDatas.data(), (uint32_t*)pageSizes.cpuIntDatas.data(),
                    total_num_rows, batch_size, num_qo_heads_per_batch, numHeads, headDim, headDim,
                    pageLen, useFlashInferCudaGraph, /*sizeof_dtype_o=*/sizeof(QType),
                    /*window_left=*/windowLeft, /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
                    /*num_colocated_ctas=*/0, /*uniform_q_len=*/0, stream);

            if (plan_status != cudaSuccess) {
                printf("FastllmCudaHalfPagedAttentionBatch: PrefillPlan failed: %s\n", cudaGetErrorString(plan_status));
                exit(0);
            }
            if (entry == nullptr) {
                entry = std::make_shared<PrefillPlanCacheEntry>();
            }
            entry->plan_info = created_plan_info;
            size_t required_bytes = planIntBytes(created_plan_info);
            if (entry->d_int_plan == nullptr || entry->int_plan_bytes < required_bytes) {
                if (entry->d_int_plan != nullptr) {
                    // This entry belongs to the current per-thread stream.
                    // Finish its previous consumer before returning the old
                    // allocation to FastLLM's process-wide CUDA pool.
                    cudaError_t prior_status = cudaStreamSynchronize(stream);
                    if (prior_status != cudaSuccess) {
                        printf("FastllmCudaHalfPagedAttentionBatch: plan resize synchronization failed: %s\n",
                               cudaGetErrorString(prior_status));
                        exit(0);
                    }
                    FastllmCudaFree(entry->d_int_plan);
                }
                entry->d_int_plan = FastllmCudaMalloc(required_bytes);
                entry->int_plan_bytes = required_bytes;
            }
            if (entry->d_int_plan == nullptr) {
                printf("FastllmCudaHalfPagedAttentionBatch: plan allocation failed (%zu bytes)\n",
                       required_bytes);
                exit(0);
            }
            cudaError_t copy_status = cudaMemcpyAsync(
                entry->d_int_plan, workspace.d_int_workspace,
                required_bytes, cudaMemcpyDeviceToDevice, stream);
            if (copy_status != cudaSuccess) {
                printf("FastllmCudaHalfPagedAttentionBatch: plan copy failed: %s\n",
                       cudaGetErrorString(copy_status));
                exit(0);
            }
            if (dynamic_decode_plan) {
                int num_sm = 0;
                cudaError_t attribute_status = cudaDeviceGetAttribute(
                    &num_sm, cudaDevAttrMultiProcessorCount, current_device_id);
                if (attribute_status != cudaSuccess || num_sm <= 0 || numHeads <= 0) {
                    printf("FastllmCudaHalfPagedAttentionBatch: failed to get dynamic plan capacity: %s\n",
                           cudaGetErrorString(attribute_status));
                    exit(0);
                }
                entry->max_batch_size_if_split =
                    uint32_t((2 * num_sm) / numHeads);
                if (entry->max_batch_size_if_split == 0 ||
                    !created_plan_info.enable_cuda_graph ||
                    !created_plan_info.split_kv ||
                    created_plan_info.total_num_rows != total_num_rows ||
                    created_plan_info.padded_batch_size <
                        entry->max_batch_size_if_split) {
                    printf("FastllmCudaHalfPagedAttentionBatch: unsupported dynamic graph plan layout.\n");
                    exit(0);
                }
                if (entry->d_dynamic_status == nullptr) {
                    entry->d_dynamic_status = FastllmCudaMalloc(sizeof(uint32_t));
                }
                if (entry->d_dynamic_status == nullptr) {
                    printf("FastllmCudaHalfPagedAttentionBatch: dynamic plan status allocation failed.\n");
                    exit(0);
                }
            }
            // Keep the workspace mutex until both its staging copy and the
            // private plan copy have completed. Otherwise another host
            // thread can rewrite the shared pinned/device buffers too soon.
            cudaError_t ready_status = cudaStreamSynchronize(stream);
            if (ready_status != cudaSuccess) {
                printf("FastllmCudaHalfPagedAttentionBatch: plan synchronization failed: %s\n",
                       cudaGetErrorString(ready_status));
                exit(0);
            }
            return entry;
        };

        std::shared_ptr<PrefillPlanCacheEntry> entry;
        if (enableCudaGraph) {
            {
                std::lock_guard<std::mutex> guard(plan_cache_mutex);
                auto &device_cache = graph_plan_cache[current_device_id];
                auto cache_it = device_cache.find(plan_key);
                if (cache_it != device_cache.end()) {
                    entry = cache_it->second;
                }
            }
            if (entry == nullptr) {
                std::shared_ptr<PrefillPlanCacheEntry> created = createPlan(nullptr);
                if (created == nullptr) {
                    return;
                }
                std::lock_guard<std::mutex> guard(plan_cache_mutex);
                auto &device_cache = graph_plan_cache[current_device_id];
                auto inserted = device_cache.emplace(plan_key, created);
                entry = inserted.first->second;
            }
        } else {
            constexpr size_t kMaxEagerPlanSlotsPerDevice = 32;
            auto &device_cache = eager_plan_caches[current_device_id];
            auto slot_it = device_cache.slots.find(eager_slot_key);
            if (slot_it == device_cache.slots.end()) {
                if (device_cache.slots.size() >= kMaxEagerPlanSlotsPerDevice) {
                    // Slots are private to this host thread and therefore to
                    // its per-thread CUDA stream. Synchronize once before
                    // evicting an entry that a previously launched kernel may
                    // still reference.
                    cudaError_t evict_status = cudaStreamSynchronize(stream);
                    if (evict_status != cudaSuccess) {
                        printf("FastllmCudaHalfPagedAttentionBatch: eager plan eviction synchronization failed: %s\n",
                               cudaGetErrorString(evict_status));
                        exit(0);
                    }
                    auto victim = device_cache.slots.begin();
                    for (auto it = device_cache.slots.begin();
                         it != device_cache.slots.end(); ++it) {
                        if (it->second.last_use < victim->second.last_use) {
                            victim = it;
                        }
                    }
                    device_cache.slots.erase(victim);
                }
                slot_it = device_cache.slots.emplace(
                    eager_slot_key, EagerPlanCacheSlot()).first;
            }
            EagerPlanCacheSlot &slot = slot_it->second;
            slot.last_use = ++device_cache.use_counter;
            if (slot.entry == nullptr || slot.plan_key != plan_key) {
                slot.entry = createPlan(slot.entry);
                if (slot.entry == nullptr) {
                    return;
                }
                slot.plan_key = plan_key;
            }
            entry = slot.entry;
        }
        plan_entry = entry;
        plan_info = entry->plan_info;
        plan_int_base = entry->d_int_plan;
        if (dynamic_decode_plan) {
            std::lock_guard<std::mutex> guard(plan_cache_mutex);
            launch_dynamic_decode_plan = true;
            if (stream_is_capturing && entry->has_dynamic_capture_id &&
                entry->last_dynamic_capture_id == capture_id) {
                launch_dynamic_decode_plan = false;
            } else if (stream_is_capturing) {
                entry->last_dynamic_capture_id = capture_id;
                entry->has_dynamic_capture_id = true;
            }
            validate_dynamic_decode_plan =
                !stream_is_capturing && !entry->dynamic_plan_validated;
        }

        uint32_t q_stride_n = (q.dims.size() >= 2 && q.strides.size() >= 2) ? q.strides[1] : q2;
        uint32_t q_stride_h = (q.dims.size() >= 3 && q.strides.size() >= 1) ? q.strides[0] : (q1 * q2);

        int32_t *qSizesData = (int32_t*)qSizes.cudaData;
        FastllmPagedPrefillParams<QType, KVType> prefill_params(
            qd, paged_kv, nullptr, (uint32_t*)qSizesData, nullptr, nullptr,
            od, nullptr, nullptr,
            num_qo_heads_per_batch, q_stride_n, q_stride_h,
            windowLeft, 0.0f, scale, 1.0f, 10000.0f
        );

        FastllmConfigureFP4PagedParams(prefill_params, pageLen, numHeads, headDim);
        prefill_params.request_indices = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(plan_int_base) + plan_info.request_indices_offset);
        prefill_params.qo_tile_indices = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(plan_int_base) + plan_info.qo_tile_indices_offset);
        prefill_params.kv_tile_indices = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(plan_int_base) + plan_info.kv_tile_indices_offset);
        prefill_params.o_indptr = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(plan_int_base) + plan_info.o_indptr_offset);
        prefill_params.kv_chunk_size_ptr = reinterpret_cast<uint32_t*>(
            static_cast<uint8_t*>(plan_int_base) + plan_info.kv_chunk_size_ptr_offset);
        prefill_params.padded_batch_size = plan_info.padded_batch_size;
        prefill_params.max_total_num_rows = plan_info.total_num_rows;
        if (plan_info.enable_cuda_graph) {
            prefill_params.total_num_rows = reinterpret_cast<uint32_t*>(
                static_cast<uint8_t*>(plan_int_base) + plan_info.total_num_rows_offset);
        }

        QType *tmp_v = nullptr;
        float *tmp_s = nullptr;
        if (plan_info.split_kv) {
            prefill_params.merge_indptr = reinterpret_cast<uint32_t*>(
                static_cast<uint8_t*>(plan_int_base) + plan_info.merge_indptr_offset);
            if (plan_info.enable_cuda_graph) {
                prefill_params.block_valid_mask = reinterpret_cast<bool*>(
                    static_cast<uint8_t*>(plan_int_base) + plan_info.block_valid_mask_offset);
            }
            tmp_v = reinterpret_cast<QType*>(
                static_cast<uint8_t*>(workspace.d_float_workspace) + plan_info.v_offset);
            tmp_s = reinterpret_cast<float*>(
                static_cast<uint8_t*>(workspace.d_float_workspace) + plan_info.s_offset);
        }

        if (launch_dynamic_decode_plan) {
            FastllmFlashInferDecodePlanParams dynamic_params;
            dynamic_params.qoIndptr = reinterpret_cast<uint32_t*>(qSizesData);
            dynamic_params.kvIndptr = reinterpret_cast<uint32_t*>(pageSizesData);
            dynamic_params.requestIndices = prefill_params.request_indices;
            dynamic_params.qoTileIndices = prefill_params.qo_tile_indices;
            dynamic_params.kvTileIndices = prefill_params.kv_tile_indices;
            dynamic_params.mergeIndptr = prefill_params.merge_indptr;
            dynamic_params.oIndptr = prefill_params.o_indptr;
            dynamic_params.kvChunkSize = prefill_params.kv_chunk_size_ptr;
            dynamic_params.totalNumRows = prefill_params.total_num_rows;
            dynamic_params.blockValidMask = prefill_params.block_valid_mask;
            dynamic_params.status = reinterpret_cast<uint32_t*>(
                plan_entry->d_dynamic_status);
            dynamic_params.batchSize = batch_size;
            dynamic_params.groupSize = num_qo_heads_per_batch / numHeads;
            dynamic_params.ctaTileQ = uint32_t(plan_info.cta_tile_q);
            dynamic_params.pageSize = uint32_t(pageLen);
            dynamic_params.maxBatchSizeIfSplit =
                plan_entry->max_batch_size_if_split;
            dynamic_params.paddedBatchSize =
                uint32_t(plan_info.padded_batch_size);
            dynamic_params.maxTotalNumRows =
                uint32_t(plan_info.total_num_rows);
            dynamic_params.windowLeft = windowLeft;
            cudaError_t dynamic_status = FastllmFlashInferUpdateDecodePlan(
                dynamic_params, stream);
            if (dynamic_status != cudaSuccess) {
                status = dynamic_status;
                return;
            }
            if (validate_dynamic_decode_plan) {
                uint32_t planner_status = 1;
                cudaError_t copy_status = cudaMemcpyAsync(
                    &planner_status, plan_entry->d_dynamic_status,
                    sizeof(planner_status), cudaMemcpyDeviceToHost, stream);
                if (copy_status == cudaSuccess) {
                    copy_status = cudaStreamSynchronize(stream);
                }
                if (copy_status != cudaSuccess || planner_status != 0) {
                    printf("FastllmCudaHalfPagedAttentionBatch: dynamic FlashInfer plan validation failed (%s, status=%u)\n",
                           cudaGetErrorString(copy_status), planner_status);
                    status = copy_status == cudaSuccess
                        ? cudaErrorInvalidValue : copy_status;
                    return;
                }
                std::lock_guard<std::mutex> guard(plan_cache_mutex);
                plan_entry->dynamic_plan_validated = true;
            }
        }

        bool enable_pdl = false;
        status = FastllmDispatchPagedPrefillByHeadDim(
            headDim, (long)plan_info.cta_tile_q, prefill_params, tmp_v, tmp_s, enable_pdl, stream,
            "FastllmCudaHalfPagedAttentionBatch");
    };

    if (q.dataType == fastllm::DataType::BFLOAT16) {
        if (pagedKVCacheK->dataType == fastllm::DataType::FP8_E4M3) {
            runBatchPrefill.template operator()<__nv_bfloat16, __nv_fp8_e4m3>();
#if CUDA_VERSION >= 12080
        } else if (pagedKVCacheK->dataType == fastllm::DataType::FP4_E2M1) {
            runBatchPrefill.template operator()<__nv_bfloat16, __nv_fp4x2_e2m1>();
#endif
        } else {
            runBatchPrefill.template operator()<__nv_bfloat16, __nv_bfloat16>();
        }
    } else {
        if (pagedKVCacheK->dataType == fastllm::DataType::FP8_E4M3) {
            runBatchPrefill.template operator()<half, __nv_fp8_e4m3>();
#if CUDA_VERSION >= 12080
        } else if (pagedKVCacheK->dataType == fastllm::DataType::FP4_E2M1) {
            runBatchPrefill.template operator()<half, __nv_fp4x2_e2m1>();
#endif
        } else {
            runBatchPrefill.template operator()<half, half>();
        }
    }

    if (status != cudaSuccess) {
        printf("FastllmCudaHalfPagedAttentionBatch: FlashInfer error: %s\n", cudaGetErrorString(status));
        exit(0);
    }

// 仅更新 output 的 shape 为 [seqlen, num_heads_total, head_dim]，与 FlashInfer 输出布局一致，
// 避免在 CUDA 侧再做 Permute，由调用方按需做一次 Reshape + Permute 即可得到 [bsz, seqlen, embed_dim]
((fastllm::Data*)&output)->Resize({output.dims[1], output.dims[0], output.dims[2]});

    if (sync) {
        DeviceSync();
    }
    return true;
#endif
}
