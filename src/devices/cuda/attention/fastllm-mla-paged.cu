#include "fastllm-attention-workspace.cuh"

static bool FastllmCudaMLAPagedImpl(const fastllm::Data &qNope, const fastllm::Data &qPe,
        const std::vector<const fastllm::Data*> &peCaches,
        const std::vector<const fastllm::Data*> &latentCaches,
        const std::vector<int> &requestedKvLengths,
        const std::vector<const fastllm::Data*> &physicalTokenIndices,
        fastllm::Data &output, float softmaxScale) {
#ifndef FASTLLM_ENABLE_FLASHINFER
    return false;
#else
    using namespace flashinfer;
    const int batch = (int)peCaches.size();
    if (batch == 0 || latentCaches.size() != peCaches.size() ||
        requestedKvLengths.size() != peCaches.size() || physicalTokenIndices.size() != peCaches.size() ||
        qPe.dims.size() != 4 || qNope.dims.size() != 3 ||
        qNope.dataDevice != fastllm::DataDevice::CUDA || !qNope.cudaData ||
        (qNope.dataType != fastllm::DataType::FLOAT16 && qNope.dataType != fastllm::DataType::BFLOAT16) ||
        qPe.dataType != qNope.dataType || output.dataType != qNope.dataType ||
        qPe.dataDeviceIds != qNope.dataDeviceIds || output.dataDeviceIds != qNope.dataDeviceIds ||
        !qPe.cudaData || !output.cudaData || output.dims != qNope.dims) return false;
    const int h = qPe.dims[2], head_dim_ckv = qNope.dims.back(), head_dim_kpe = qPe.dims[3];
    const int qoLen = qPe.dims[0] * qPe.dims[1];
    if (qoLen <= 0 || qoLen % batch || (batch > 1 && qoLen != batch) ||
        qNope.dims != std::vector<int>({h, qoLen, head_dim_ckv}) ||
        h <= 0 || head_dim_kpe != 64 || (head_dim_ckv != 128 && head_dim_ckv != 512)) return false;
    const bool causal = true;
    const int queryLength = qoLen / batch;
    const bool physicalPages = std::any_of(physicalTokenIndices.begin(), physicalTokenIndices.end(),
        [](const auto *indices) { return indices != nullptr; });
    if (!peCaches[0] || !latentCaches[0]) return false;
    const auto &peCache = *peCaches[0];
    const auto &latentCache = *latentCaches[0];
    const int pageLen = physicalPages ? 1 : peCache.pageLen;
    if (pageLen <= 0) return false;
    std::vector<int32_t> q_indptr_h(1, 0), kv_indptr_h(1, 0), kv_len_arr_h;
    std::vector<int32_t> hostIndices;
    bool allSparse = true;
    for (int request = 0; request < batch; ++request) {
        const auto *pe = peCaches[request], *latent = latentCaches[request];
        if (!pe || !latent || !pe->isPagedKVCache || !latent->isPagedKVCache ||
            !pe->pagedKVCacheData || !latent->pagedKVCacheData ||
            pe->pagedKVCacheData != peCache.pagedKVCacheData ||
            latent->pagedKVCacheData != latentCache.pagedKVCacheData ||
            pe->dataType != qNope.dataType || latent->dataType != qNope.dataType ||
            pe->dataDeviceIds != qNope.dataDeviceIds || latent->dataDeviceIds != qNope.dataDeviceIds ||
            pe->pageLen <= 0 || (!physicalPages && pe->pageLen != pageLen) ||
            pe->pageLen != latent->pageLen || pe->lastPageLen != latent->lastPageLen ||
            pe->pageIndex != latent->pageIndex || pe->pageIndex.empty()) return false;
        const int fullLength = ((int)pe->pageIndex.size() - 1) * pe->pageLen + pe->lastPageLen;
        const int length = requestedKvLengths[request] > 0 ? requestedKvLengths[request] : fullLength;
        if (length < queryLength || length > fullLength) return false;
        const auto *indices = physicalTokenIndices[request];
        if (indices && (queryLength != 1 || indices->dataDevice != fastllm::DataDevice::CUDA ||
            indices->dataType != fastllm::DataType::INT32 || !indices->cudaData ||
            indices->Count(0) < (uint64_t)length || indices->dataDeviceIds != qNope.dataDeviceIds)) return false;
        allSparse &= indices != nullptr;
        const int pages = (length + pageLen - 1) / pageLen;
        q_indptr_h.push_back(q_indptr_h.back() + queryLength);
        kv_indptr_h.push_back(kv_indptr_h.back() + pages);
        kv_len_arr_h.push_back(length);
        if (!indices) {
            hostIndices.resize(kv_indptr_h.back());
            int32_t *destination = hostIndices.data() + kv_indptr_h[request];
            if (physicalPages) {
                // A mixed dense/sparse batch uses physical tokens as size-one
                // pages, without gathering or modifying the shared KV pools.
                for (int token = 0; token < length; ++token)
                    destination[token] = pe->pageIndex[token / pe->pageLen] * pe->pageLen + token % pe->pageLen;
            } else std::copy_n(pe->pageIndex.begin(), pages, destination);
        }
    }
    const int numPages = kv_indptr_h.back();
    FlashInferWorkSpaceManager& workspace = getFastllmFlashInferWorkSpace();
    std::lock_guard<std::mutex> workspace_guard(workspace.plan_mutex);
    MLAPlanInfo plan_info;
    void *int_plan;
    // Keep capture on its existing path: captured graphs must never retain
    // pointers into this bounded, evictable eager cache.
    const bool capturing = FastllmCudaGraphIsCapturing();
    const bool cache_plan = allSparse && !capturing;
    std::vector<int> plan_key;
    if (cache_plan) {
        // Sparse decode has one query and page size one. Ordered lengths
        // determine the batch size and both indptr arrays.
        plan_key = {h, head_dim_ckv};
        plan_key.insert(plan_key.end(), kv_len_arr_h.begin(), kv_len_arr_h.end());
    }
    auto cached = cache_plan ? workspace.mla_decode_plans.find(plan_key) : workspace.mla_decode_plans.end();
    if (cached != workspace.mla_decode_plans.end()) {
        cached->second->last_used = ++workspace.mla_plan_clock;
        plan_info = cached->second->info;
        int_plan = cached->second->data;
    } else {
        // MLA uses a separate scheduler without a counting interface.
        workspace.EnsureIntCapacity(64ULL << 20);
        cudaError_t plan_status = MLAPlan<int32_t>(
            workspace.d_float_workspace, workspace.float_workspace_size,
            workspace.d_int_workspace, workspace.h_page_locked_int_workspace,
            workspace.int_workspace_size, plan_info, q_indptr_h.data(), kv_indptr_h.data(), kv_len_arr_h.data(),
            batch, (uint32_t)h, (uint32_t)head_dim_ckv, causal, 0, !capturing);
        if (plan_status != cudaSuccess) return false;
        int_plan = workspace.d_int_workspace;
        if (cache_plan) {
            constexpr size_t max_cached_plans = 16;
            if (workspace.mla_decode_plans.size() >= max_cached_plans) {
                // Entries can have consumers on other host threads/streams.
                checkCudaErrors("MLA plan eviction sync", cudaDeviceSynchronize());
                auto oldest = std::min_element(workspace.mla_decode_plans.begin(), workspace.mla_decode_plans.end(),
                    [](const auto &a, const auto &b) { return a.second->last_used < b.second->last_used; });
                workspace.mla_decode_plans.erase(oldest);
            }
            int device = -1;
            cudaGetDevice(&device);
            auto entry = std::make_unique<FlashInferWorkSpaceManager::MLADecodePlan>(device);
            entry->info = plan_info;
            entry->last_used = ++workspace.mla_plan_clock;
            // work_indptr is the last integer allocation in MLAPlan. Only
            // num_blks_y + 1 entries are read; preceding offsets stay intact.
            entry->size = plan_info.work_indptr_offset + (plan_info.num_blks_y + 1) * sizeof(int32_t);
            fastllm::AssertInFastLLM(entry->size <= workspace.int_workspace_size, "MLA plan exceeds workspace.\n");
            entry->data = FastllmCudaDirectMalloc(entry->size);
            checkCudaErrors("MLA plan cache copy", cudaMemcpyAsync(entry->data, workspace.d_int_workspace,
                entry->size, cudaMemcpyDeviceToDevice, 0));
            // Finish the upload before another planner reuses pinned staging,
            // and publish immutable plan data only after it is ready.
            checkCudaErrors("MLA plan cache ready", cudaStreamSynchronize(0));
            int_plan = entry->data;
            workspace.mla_decode_plans.emplace(plan_key, std::move(entry));
        }
    }

    const bool borrowIndices = batch == 1 && physicalTokenIndices[0] != nullptr;
    int32_t *d_kv_indices = borrowIndices ? (int32_t*)physicalTokenIndices[0]->cudaData
        : (int32_t*)FastllmCudaMalloc(numPages * sizeof(int32_t));
    if (!borrowIndices) {
        if (!physicalPages) {
            checkCudaErrors("MLA page indices upload", cudaMemcpy(d_kv_indices, hostIndices.data(),
                numPages * sizeof(int32_t), cudaMemcpyHostToDevice));
        } else {
            for (int request = 0; request < batch; ++request) {
                const auto *indices = physicalTokenIndices[request];
                const int offset = kv_indptr_h[request];
                const void *source = indices ? indices->cudaData : hostIndices.data() + offset;
                checkCudaErrors("MLA token indices gather", cudaMemcpyAsync(d_kv_indices + offset, source,
                    (kv_indptr_h[request + 1] - offset) * sizeof(int32_t),
                    indices ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, 0));
            }
        }
    }

    uint_fastdiv num_heads_div((uint32_t)h);
    uint_fastdiv block_size_div((uint32_t)pageLen);

    auto runAttention = [&](auto scalarTag) -> cudaError_t {
        using scalar_t = decltype(scalarTag);
        MLAParams<scalar_t, scalar_t, scalar_t, int32_t> params = {};
        params.q_nope = (scalar_t*)qNope.cudaData;
        params.q_pe = (scalar_t*)qPe.cudaData;
        params.ckv = (scalar_t*)latentCache.pagedKVCacheData->cudaData;
        params.kpe = (scalar_t*)peCache.pagedKVCacheData->cudaData;
        params.final_o = (scalar_t*)output.cudaData;
        params.final_lse = nullptr;
        params.q_indptr = (int32_t*)((uint8_t*)int_plan + plan_info.q_indptr_offset);
        params.kv_indptr = (int32_t*)((uint8_t*)int_plan + plan_info.kv_indptr_offset);
        params.partial_indptr = (int32_t*)((uint8_t*)int_plan + plan_info.partial_indptr_offset);
        params.kv_indices = d_kv_indices;
        params.q_len = (int32_t*)((uint8_t*)int_plan + plan_info.q_len_offset);
        params.kv_len = (int32_t*)((uint8_t*)int_plan + plan_info.kv_len_offset);
        params.q_start = (int32_t*)((uint8_t*)int_plan + plan_info.q_start_offset);
        params.kv_start = (int32_t*)((uint8_t*)int_plan + plan_info.kv_start_offset);
        params.kv_end = (int32_t*)((uint8_t*)int_plan + plan_info.kv_end_offset);
        params.work_indptr = (int32_t*)((uint8_t*)int_plan + plan_info.work_indptr_offset);
        params.merge_packed_offset_start = (int32_t*)((uint8_t*)int_plan + plan_info.merge_packed_offset_start_offset);
        params.merge_packed_offset_end = (int32_t*)((uint8_t*)int_plan + plan_info.merge_packed_offset_end_offset);
        params.merge_partial_packed_offset_start = (int32_t*)((uint8_t*)int_plan + plan_info.merge_partial_packed_offset_start_offset);
        params.merge_partial_packed_offset_end = (int32_t*)((uint8_t*)int_plan + plan_info.merge_partial_packed_offset_end_offset);
        params.merge_partial_stride = (int32_t*)((uint8_t*)int_plan + plan_info.merge_partial_stride_offset);
        params.partial_o = (scalar_t*)((uint8_t*)workspace.d_float_workspace + plan_info.partial_o_offset);
        params.partial_lse = (float*)((uint8_t*)workspace.d_float_workspace + plan_info.partial_lse_offset);
        params.num_heads = num_heads_div;
        params.block_size = block_size_div;
        // qNope / output layout is [h, b*s, c].
        params.q_nope_stride_n = head_dim_ckv;
        params.q_nope_stride_h = (uint32_t)qoLen * head_dim_ckv;
        params.q_pe_stride_n = h * head_dim_kpe;
        params.q_pe_stride_h = head_dim_kpe;
        params.ckv_stride_page = (uint32_t)(pageLen * head_dim_ckv);
        params.ckv_stride_n = (uint32_t)head_dim_ckv;
        params.kpe_stride_page = (uint32_t)(pageLen * head_dim_kpe);
        params.kpe_stride_n = (uint32_t)head_dim_kpe;
        params.o_stride_n = head_dim_ckv;
        params.o_stride_h = (uint32_t)qoLen * head_dim_ckv;
        params.sm_scale = softmaxScale;
        params.return_lse_base_on_e = false;

        if (head_dim_ckv == 128) {
            return mla::BatchMLAPagedAttention<MaskMode::kCausal, 128, 64>(
                params, (uint32_t)plan_info.num_blks_x,
                (uint32_t)plan_info.num_blks_y, 0);
        }
        return mla::BatchMLAPagedAttention<MaskMode::kCausal, 512, 64>(
            params, (uint32_t)plan_info.num_blks_x,
            (uint32_t)plan_info.num_blks_y, 0);
    };

    cudaError_t status = qNope.dataType == fastllm::DataType::BFLOAT16 ?
        runAttention(__nv_bfloat16()) : runAttention(half());

    if (!borrowIndices) FastllmCudaFree(d_kv_indices);

    if (status != cudaSuccess) return false;
    DeviceSync();
    return true;
#endif
}

bool FastllmCudaMLAPaged(const fastllm::Data &qNope, const fastllm::Data &qPe,
        const fastllm::Data &kvCachePaged, const fastllm::Data &peCachePaged,
        fastllm::Data &output, float softmaxScale, int requestedKvLen,
        const fastllm::Data *physicalTokenIndices) {
    return FastllmCudaMLAPagedImpl(qNope, qPe, {&kvCachePaged}, {&peCachePaged},
        {requestedKvLen}, {physicalTokenIndices}, output, softmaxScale);
}

bool FastllmCudaMLAPagedBatch(const fastllm::Data &qNope, const fastllm::Data &qPe,
        const std::vector<const fastllm::Data*> &peCaches,
        const std::vector<const fastllm::Data*> &latentCaches,
        const std::vector<int> &kvLengths,
        const std::vector<const fastllm::Data*> &physicalTokenIndices,
        fastllm::Data &output, float softmaxScale) {
    if (peCaches.size() <= 1) return false;
    return FastllmCudaMLAPagedImpl(qNope, qPe, peCaches, latentCaches,
        kvLengths, physicalTokenIndices, output, softmaxScale);
}
