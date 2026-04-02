//
// Created by huangyuyang on 8/2/24.
//

#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/multicudadevice.h"

#include "fastllm-multicuda.cuh"

#include "utils.h"

#include <mutex>

namespace fastllm {
    static void SyncCudaAndCheck(int device, const char *where);
    static bool NeedTpDebugInfo();

    MultiCudaDevice::MultiCudaDevice(CudaDevice *cudaDevice) {
        this->cudaDevice = cudaDevice;
        this->deviceType = "multicuda";

        this->ops["LinearAdd"] = (BaseOperator*)(new MultiCudaLinearAddOp());
        this->ops["LinearSwiglu"] = (BaseOperator*)(new MultiCudaLinearSwigluOp());
        this->ops["RMSNorm"] = (BaseOperator*)(new MultiCudaRMSNormOp());
        this->ops["AddTo"] = (BaseOperator*)(new MultiCudaAddToOp());
        this->ops["Split"] = (BaseOperator*)(new MultiCudaSplitOp());
        this->ops["Swiglu"] = (BaseOperator*)(new MultiCudaSwigluOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new MultiCudaPermuteSelfOp());
        this->ops["RopeEncoding"] = (BaseOperator*)(new MultiCudaRopeEncodingOp());
        this->ops["AppendPagedCache"] = (BaseOperator*)(new MultiCudaAppendPagedCacheOp());
        this->ops["AttentionPagedBatch"] = (BaseOperator*)(new MultiCudaAttentionPagedBatchOp());
        this->ops["QKVRMSNormRopeSplitAppendPagedCache"] =
                (BaseOperator*)(new MultiCudaQKVRMSNormRopeSplitAppendPagedCacheOp());
        this->ops["MLP"] = (BaseOperator*)(new MultiCudaMLPOp());
        this->ops["Linear"] = (BaseOperator*)(new MultiCudaLinearOp());
        this->ops["MergeMOE"] = (BaseOperator*)(new MultiCudaMergeMOE());
        this->ops["MergeAttention"] = (BaseOperator*)(new MultiCudaMergeAttention());
    }

    static void ResetMultiCudaTensor(fastllm::Data &data) {
        if (!data.multiDeviceData) {
            data.ClearTensorParallelLayout();
            return;
        }
        for (auto &it : data.multiDeviceDatas) {
            delete it.second;
        }
        data.multiDeviceDatas.clear();
        data.multiDeviceData = false;
        data.ClearTensorParallelLayout();
    }

    static bool HasReplicatedMultiCudaTensor(const fastllm::Data &data, const std::vector <int> &devices) {
        if (!data.multiDeviceData || !data.IsTensorParallelReplicated()) {
            return false;
        }
        for (int device : devices) {
            auto it = data.multiDeviceDatas.find(device);
            if (it == data.multiDeviceDatas.end() || it->second == nullptr) {
                return false;
            }
            if (it->second->dataType != data.dataType || it->second->dims != data.dims) {
                return false;
            }
            if (it->second->dataDevice == DataDevice::CUDA && it->second->cudaData == nullptr && data.Count(0) > 0) {
                return false;
            }
        }
        return true;
    }

    static void SyncReplicatedCpuIntData(fastllm::Data &data, const std::vector <int> &devices) {
        if (data.cpuIntDatas.empty()) {
            return;
        }
        for (int device : devices) {
            auto it = data.multiDeviceDatas.find(device);
            if (it == data.multiDeviceDatas.end() || it->second == nullptr) {
                continue;
            }
            it->second->cpuIntDatas = data.cpuIntDatas;
        }
    }

    // 确保 data 在指定设备上具有 REPLICATED 布局；已有完整副本时直接复用，否则重建并按需拷贝数据。
    static void EnsureReplicatedMultiCudaTensor(fastllm::Data &data, const std::vector <int> &devices, bool copyData) {
        if (HasReplicatedMultiCudaTensor(data, devices)) {
            SyncReplicatedCpuIntData(data, devices);
            return;
        }
        ResetMultiCudaTensor(data);
        PrepareMultiCudaReplicatedData(data, devices, copyData);
        SyncReplicatedCpuIntData(data, devices);
    }

    static void RefreshReplicatedMultiCudaTensor(fastllm::Data &data, const std::vector <int> &devices) {
        ResetMultiCudaTensor(data);
        PrepareMultiCudaReplicatedData(data, devices, true);
        SyncReplicatedCpuIntData(data, devices);
    }

    static void SyncReplicatedRootFromReplica(fastllm::Data &data, const std::vector <int> &devices) {
        if (devices.empty()) {
            return;
        }
        int rootDevice = devices[0];
        auto it = data.multiDeviceDatas.find(rootDevice);
        if (it == data.multiDeviceDatas.end() || it->second == nullptr) {
            return;
        }
        Data *replica = it->second;
        if (replica->cudaData != nullptr && data.Count(0) > 0) {
            data.dataDevice = DataDevice::CUDA;
            data.dataDeviceIds = {rootDevice};
            std::swap(data.cudaData, replica->cudaData);
            std::swap(data.expansionSize, replica->expansionSize);
            std::swap(data.expansionBytes, replica->expansionBytes);
            ResetMultiCudaTensor(data);
        }
    }

    static int NormalizeAxis(int axis, int dimsLen) {
        return (axis % dimsLen + dimsLen) % dimsLen;
    }

    static bool HasSuffix(const std::string &value, const std::string &suffix) {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    static bool IsTensorParallelRowWeight(const fastllm::Data &weight) {
        if (weight.tpLinearType == TP_LINEAR_ROW) {
            return true;
        }
        return HasSuffix(weight.name, ".self_attn.mergeqkv.weight") ||
               HasSuffix(weight.name, ".self_attn.W_pack.weight") ||
               HasSuffix(weight.name, ".self_attn.q_proj.weight") ||
               HasSuffix(weight.name, ".self_attn.k_proj.weight") ||
               HasSuffix(weight.name, ".self_attn.v_proj.weight") ||
               HasSuffix(weight.name, ".mlp.gateup_proj.weight") ||
               HasSuffix(weight.name, ".mlp.gate_proj.weight") ||
               HasSuffix(weight.name, ".mlp.up_proj.weight");
    }

    static bool IsTensorParallelColumnWeight(const fastllm::Data &weight) {
        if (weight.tpLinearType == TP_LINEAR_COLUMN) {
            return true;
        }
        return HasSuffix(weight.name, ".self_attn.o_proj.weight") ||
               HasSuffix(weight.name, ".mlp.down_proj.weight");
    }

    static void SyncReplicatedLocalShapeFromRoot(fastllm::Data &data, const std::vector <int> &devices) {
        if (!data.multiDeviceData || !data.IsTensorParallelReplicated()) {
            return;
        }
        for (int device : devices) {
            auto it = data.multiDeviceDatas.find(device);
            if (it == data.multiDeviceDatas.end() || it->second == nullptr) {
                continue;
            }
            if (it->second->dims != data.dims) {
                it->second->Resize(data.dims);
            }
        }
    }

    static void SyncShardedLocalShapeFromRoot(fastllm::Data &data, const std::vector <int> &devices) {
        if (!data.multiDeviceData || !data.IsTensorParallelSharded() || data.dims.empty()) {
            return;
        }
        int axis = NormalizeAxis(data.tpAxis, (int)data.dims.size());
        long long other = 1;
        for (int i = 0; i < (int)data.dims.size(); i++) {
            if (i != axis) {
                other *= data.dims[i];
            }
        }
        AssertInFastLLM(other > 0, "Tensor parallel local shape sync failed.\n");
        for (int device : devices) {
            auto it = data.multiDeviceDatas.find(device);
            if (it == data.multiDeviceDatas.end() || it->second == nullptr) {
                continue;
            }
            Data *local = it->second;
            long long localCount = local->Count(0);
            AssertInFastLLM(localCount % other == 0,
                            "Tensor parallel local shape sync failed: local count mismatch.\n");
            std::vector <int> localDims = data.dims;
            localDims[axis] = (int)(localCount / other);
            if (local->dims != localDims) {
                local->Resize(localDims);
            }
        }
    }

    static void CopyShardedLayout(fastllm::Data &output, const fastllm::Data &input, const std::vector <int> &devices) {
        output.multiDeviceData = true;
        output.tpLayout = TP_LAYOUT_SHARDED;
        output.tpAxis = input.tpAxis;
        output.tpGlobalDims = input.dims;
        output.tpRanges = input.tpRanges;
        for (int device : devices) {
            auto it = input.multiDeviceDatas.find(device);
            if (it == input.multiDeviceDatas.end() || it->second == nullptr) {
                continue;
            }
            Data *local = new Data(output.dataType, it->second->dims);
            local->dataDevice = DataDevice::CUDA;
            local->dataDeviceIds = {device};
            output.multiDeviceDatas[device] = local;
        }
        output.cudaData = nullptr;
    }

    static int GetLocalPagedCacheManagerIndex(PagedCacheManager *rootManager, int deviceId) {
        static std::mutex locker;
        static std::map <std::pair <uintptr_t, int>, int> mapping;
        static int nextIndex = 1000000;
        std::pair <uintptr_t, int> key = {(uintptr_t)rootManager, deviceId};
        std::lock_guard <std::mutex> guard(locker);
        auto it = mapping.find(key);
        if (it != mapping.end()) {
            return it->second;
        }
        int index = nextIndex++;
        mapping[key] = index;
        return index;
    }

    static PagedCacheManager* GetOrCreateLocalPagedCacheManager(PagedCacheManager &rootManager,
                                                                const fastllm::Data &localInput,
                                                                int deviceId) {
        Data localDesc(localInput.dataType, localInput.dims);
        localDesc.dataDevice = DataDevice::CUDA;
        localDesc.dataDeviceIds = {deviceId};
        int maxPages = rootManager.maxPages > 0 ? rootManager.maxPages : (rootManager.dims.empty() ? 0 : rootManager.dims[0]);
        AssertInFastLLM(maxPages > 0, "Local paged cache manager maxPages is invalid.\n");
        return AllocatePagedCacheManager(
            GetLocalPagedCacheManagerIndex(&rootManager, deviceId),
            rootManager.type,
            localDesc,
            rootManager.pageLen,
            maxPages
        );
    }

    static void AppendPagedCacheLocal(PagedCacheManager &pagedKVCache, fastllm::Data &cache, const fastllm::Data &input) {
        int numHeads = input.dims[0];
        int seqLen = input.dims[1];
        int headDim = input.dims[2];
        int pageLen = pagedKVCache.dims[1];
        auto directCopyPage = [&](int pageIdx, int inputOffset, int copyLen, int pageOffset) {
            if (input.dataType != pagedKVCache.dataType) {
                FastllmCudaPagedCacheCopy(
                    (uint8_t*)pagedKVCache.cudaData, pageIdx, pageLen,
                    numHeads, headDim, pagedKVCache.dataType,
                    (uint8_t*)input.cudaData, input.dataType, seqLen,
                    inputOffset, copyLen, pageOffset
                );
                return;
            }
            if (input.dataType != DataType::FLOAT16 &&
                input.dataType != DataType::FLOAT32 &&
                input.dataType != DataType::BFLOAT16) {
                FastllmCudaPagedCacheCopy(
                    (uint8_t*)pagedKVCache.cudaData, pageIdx, pageLen,
                    numHeads, headDim, pagedKVCache.dataType,
                    (uint8_t*)input.cudaData, input.dataType, seqLen,
                    inputOffset, copyLen, pageOffset
                );
                return;
            }

            size_t rowBytes = (size_t)headDim * input.unitSize;
            size_t srcPitch = rowBytes;
            size_t dstPitch = (size_t)numHeads * headDim * input.unitSize;
            uint8_t *pagedData = (uint8_t*)pagedKVCache.cudaData;
            uint8_t *inputData = (uint8_t*)input.cudaData;
            for (int head = 0; head < numHeads; head++) {
                uint8_t *src = inputData + ((size_t)head * seqLen + inputOffset) * rowBytes;
                uint8_t *dst = pagedData + (((size_t)pageIdx * pageLen + pageOffset) * numHeads + head) * rowBytes;
                FastllmCudaMemcpy2DDeviceToDeviceAuto(dst, dstPitch, src, srcPitch, rowBytes, copyLen,
                                                      input.dataDeviceIds.empty() ? 0 : input.dataDeviceIds[0],
                                                      input.dataDeviceIds.empty() ? 0 : input.dataDeviceIds[0]);
            }
        };
        if (cache.pagedKVCacheData == nullptr) {
            cache.pagedKVCacheData = &pagedKVCache;
        }
        cache.pageLen = pageLen;
        cache.isPagedKVCache = true;
        if (cache.dims.empty()) {
            cache.Resize(input.dims);
        } else {
            cache.Resize({cache.dims[0], cache.dims[1] + seqLen, cache.dims[2]});
        }

        uint8_t *pagedData = (uint8_t*)pagedKVCache.cudaData;
        uint8_t *inputData = (uint8_t*)input.cudaData;
        int tokensToAppend = seqLen;
        int inputOffset = 0;

        int remainingInCurrentPage = 0;
        if (!cache.pageIndex.empty()) {
            remainingInCurrentPage = pageLen - cache.lastPageLen;
        }

        if (remainingInCurrentPage > 0 && tokensToAppend > 0) {
            int currentPageIdx = cache.pageIndex.back();
            int copyLen = std::min(remainingInCurrentPage, tokensToAppend);
            static int appendCopyDebugCount = 0;
            if (NeedTpDebugInfo() && appendCopyDebugCount < 8) {
                printf("[tp append copy] device=%d page=%d inputOffset=%d copyLen=%d pageOffset=%d mgr=[%d,%d,%d,%d] paged=%p input=%p\n",
                       input.dataDeviceIds.empty() ? -1 : input.dataDeviceIds[0],
                       currentPageIdx, inputOffset, copyLen, cache.lastPageLen,
                       pagedKVCache.dims.size() > 0 ? pagedKVCache.dims[0] : -1,
                       pagedKVCache.dims.size() > 1 ? pagedKVCache.dims[1] : -1,
                       pagedKVCache.dims.size() > 2 ? pagedKVCache.dims[2] : -1,
                       pagedKVCache.dims.size() > 3 ? pagedKVCache.dims[3] : -1,
                       pagedData, inputData);
                fflush(stdout);
            }
            directCopyPage(currentPageIdx, inputOffset, copyLen, cache.lastPageLen);
            SyncCudaAndCheck(input.dataDeviceIds.empty() ? 0 : input.dataDeviceIds[0], "AppendPagedCacheLocal");
            cache.lastPageLen += copyLen;
            tokensToAppend -= copyLen;
            inputOffset += copyLen;
        }

        while (tokensToAppend > 0) {
            int newPageIdx = pagedKVCache.GetUnusedPageIndex(true);
            cache.pageIndex.push_back(newPageIdx);
            int copyLen = std::min(pageLen, tokensToAppend);
            static int appendCopyDebugCount = 0;
            if (NeedTpDebugInfo() && appendCopyDebugCount < 8) {
                printf("[tp append copy] device=%d page=%d inputOffset=%d copyLen=%d pageOffset=%d mgr=[%d,%d,%d,%d] paged=%p input=%p\n",
                       input.dataDeviceIds.empty() ? -1 : input.dataDeviceIds[0],
                       newPageIdx, inputOffset, copyLen, 0,
                       pagedKVCache.dims.size() > 0 ? pagedKVCache.dims[0] : -1,
                       pagedKVCache.dims.size() > 1 ? pagedKVCache.dims[1] : -1,
                       pagedKVCache.dims.size() > 2 ? pagedKVCache.dims[2] : -1,
                       pagedKVCache.dims.size() > 3 ? pagedKVCache.dims[3] : -1,
                       pagedData, inputData);
                fflush(stdout);
                appendCopyDebugCount++;
            }
            directCopyPage(newPageIdx, inputOffset, copyLen, 0);
            SyncCudaAndCheck(input.dataDeviceIds.empty() ? 0 : input.dataDeviceIds[0], "AppendPagedCacheLocal");
            cache.lastPageLen = copyLen;
            tokensToAppend -= copyLen;
            inputOffset += copyLen;
        }
    }

    static void SyncRootPagedCacheMeta(fastllm::Data &rootCache, const fastllm::Data &localCache,
                                       PagedCacheManager &rootManager) {
        if (rootCache.pageIndex != localCache.pageIndex) {
            if (!rootCache.pageIndex.empty()) {
                rootManager.ReleasePageIndices(rootCache.pageIndex);
            }
            rootCache.pageIndex = localCache.pageIndex;
            if (!rootCache.pageIndex.empty()) {
                rootManager.Pick(rootCache.pageIndex);
            }
        }
        rootCache.lastPageLen = localCache.lastPageLen;
        rootCache.pageLen = rootManager.pageLen;
        rootCache.pagedKVCacheData = &rootManager;
        rootCache.isPagedKVCache = true;
    }

    static void CopyPagedCacheSliceToDense(fastllm::Data &cache, int startSeq) {
        if (cache.pagedKVCacheData == nullptr || cache.dims.size() != 3 || cache.cudaData == nullptr) {
            return;
        }
        int numHeads = cache.dims[0];
        int seqLen = cache.dims[1];
        int headDim = cache.dims[2];
        int pageLen = cache.pageLen;
        if (numHeads <= 0 || seqLen <= 0 || headDim <= 0 || startSeq >= seqLen) {
            return;
        }
        startSeq = std::max(0, startSeq);

        fastllm::Data *pagedKVCache = cache.pagedKVCacheData;
        size_t rowBytes = (size_t)headDim * cache.unitSize;
        size_t srcPitch = (size_t)numHeads * rowBytes;
        size_t dstPitch = rowBytes;
        uint8_t *pagedData = (uint8_t*)pagedKVCache->cudaData;
        uint8_t *denseData = (uint8_t*)cache.cudaData;
        int seqOffset = 0;
        for (int pagePos = 0; pagePos < (int)cache.pageIndex.size(); pagePos++) {
            int pageIdx = cache.pageIndex[pagePos];
            int copyLen = (pagePos + 1 == (int)cache.pageIndex.size()) ? cache.lastPageLen : pageLen;
            if (copyLen <= 0) {
                continue;
            }
            int pageStart = seqOffset;
            int pageEnd = seqOffset + copyLen;
            if (pageEnd <= startSeq) {
                seqOffset = pageEnd;
                continue;
            }
            int localStart = std::max(startSeq, pageStart);
            int localLen = pageEnd - localStart;
            int pageOffset = localStart - pageStart;
            for (int head = 0; head < numHeads; head++) {
                uint8_t *src = pagedData + (((size_t)pageIdx * pageLen + pageOffset) * numHeads + head) * rowBytes;
                uint8_t *dst = denseData + ((size_t)head * cache.strides[0] + (size_t)localStart * cache.strides[1]) * cache.unitSize;
                FastllmCudaMemcpy2DDeviceToDeviceAuto(dst, dstPitch, src, srcPitch, rowBytes, localLen,
                                                      cache.dataDeviceIds.empty() ? 0 : cache.dataDeviceIds[0],
                                                      cache.dataDeviceIds.empty() ? 0 : cache.dataDeviceIds[0]);
            }
            seqOffset = pageEnd;
        }
    }

    static void EnsureDenseMirrorFromPagedCache(fastllm::Data &cache,
                                                const std::vector<int> &oldPageIndex,
                                                int oldSeqLen) {
        if (!cache.isPagedKVCache || cache.pagedKVCacheData == nullptr || cache.dims.size() != 3) {
            return;
        }
        int deviceId = cache.dataDeviceIds.empty() ? 0 : cache.dataDeviceIds[0];
        int seqLen = cache.dims[1];
        bool rebuild = (cache.cudaData == nullptr) ||
                       (oldSeqLen > seqLen) ||
                       (oldPageIndex.size() > cache.pageIndex.size()) ||
                       !std::equal(oldPageIndex.begin(), oldPageIndex.end(), cache.pageIndex.begin());
        int copyStart = rebuild ? 0 : oldSeqLen;

        FastllmCudaSetDevice(deviceId);
        cache.dataDevice = DataDevice::CUDA;
        cache.dataDeviceIds = {deviceId};
        if (seqLen > 0 && cache.cudaData == nullptr) {
            int reserveSeq = std::max(seqLen, ((seqLen + std::max(1, cache.pageLen) - 1) / std::max(1, cache.pageLen)) * std::max(1, cache.pageLen));
            cache.Expansion({cache.dims[0], reserveSeq, cache.dims[2]});
        } else if (seqLen > 0 && (!cache.expansionDims.empty() && cache.expansionDims[1] < seqLen)) {
            int reserveSeq = std::max(seqLen, ((seqLen + std::max(1, cache.pageLen) - 1) / std::max(1, cache.pageLen)) * std::max(1, cache.pageLen));
            std::vector<int> expandDims = cache.expansionDims;
            expandDims[0] = std::max(expandDims[0], cache.dims[0]);
            expandDims[1] = std::max(expandDims[1], reserveSeq);
            expandDims[2] = std::max(expandDims[2], cache.dims[2]);
            cache.Expansion(expandDims);
        }
        cache.Resize({cache.dims[0], seqLen, cache.dims[2]});
        if (seqLen > 0) {
            CopyPagedCacheSliceToDense(cache, copyStart);
        }
    }

    static int ReadDataIntValue(const fastllm::Data &data, int index) {
        if ((int)data.cpuIntDatas.size() > index) {
            return data.cpuIntDatas[index];
        }
        AssertInFastLLM(data.dataType == DataType::INT32PARAM || data.dataType == DataType::INT32,
                        "ReadDataIntValue: data type should be INT32PARAM or INT32.\n");
        if (data.dataDevice == DataDevice::CPU) {
            AssertInFastLLM(data.cpuData != nullptr, "ReadDataIntValue: cpuData is nullptr.\n");
            return ((int32_t*)data.cpuData)[index];
        }
        AssertInFastLLM(data.cudaData != nullptr, "ReadDataIntValue: cudaData is nullptr.\n");
        int deviceId = data.dataDeviceIds.empty() ? 0 : data.dataDeviceIds[0];
        int32_t value = 0;
        FastllmCudaSetDevice(deviceId);
        FastllmCudaCopyFromDeviceToHost(&value, (uint8_t*)data.cudaData + index * sizeof(int32_t), sizeof(int32_t));
        return (int)value;
    }

    static void CopyToPagedCachePosition(PagedCacheManager &pagedKVCache, const fastllm::Data &input,
                                         int pageIdx, int pageOffset) {
        AssertInFastLLM(input.dims.size() == 3, "CopyToPagedCachePosition: input should be 3D.\n");
        AssertInFastLLM(input.dataType == pagedKVCache.dataType,
                        "CopyToPagedCachePosition: input/cache datatype mismatch.\n");
        int numHeads = input.dims[0];
        int seqLen = input.dims[1];
        int headDim = input.dims[2];
        int pageLen = pagedKVCache.dims[1];
        AssertInFastLLM(pageIdx >= 0 && pageIdx < pagedKVCache.dims[0],
                        "CopyToPagedCachePosition: invalid page index.\n");
        AssertInFastLLM(pageOffset >= 0 && pageOffset + seqLen <= pageLen,
                        "CopyToPagedCachePosition: page range overflow.\n");
        AssertInFastLLM(pagedKVCache.dims[2] == numHeads && pagedKVCache.dims[3] == headDim,
                        "CopyToPagedCachePosition: paged cache shape mismatch.\n");

        int deviceId = input.dataDeviceIds.empty() ? 0 : input.dataDeviceIds[0];
        size_t rowBytes = (size_t)headDim * input.unitSize;
        size_t srcPitch = rowBytes;
        size_t dstPitch = (size_t)numHeads * headDim * input.unitSize;
        uint8_t *pagedData = (uint8_t*)pagedKVCache.cudaData;
        uint8_t *inputData = (uint8_t*)input.cudaData;
        for (int head = 0; head < numHeads; head++) {
            uint8_t *src = inputData + (size_t)head * seqLen * rowBytes;
            uint8_t *dst = pagedData + (((size_t)pageIdx * pageLen + pageOffset) * numHeads + head) * rowBytes;
            FastllmCudaMemcpy2DDeviceToDeviceAuto(dst, dstPitch, src, srcPitch, rowBytes, seqLen, deviceId, deviceId);
        }
    }

    static void SyncLocalPagedCacheMeta(fastllm::Data &localCache,
                                        const fastllm::Data &rootCache,
                                        PagedCacheManager &localManager) {
        std::vector<int> oldPageIndex = localCache.pageIndex;
        if (oldPageIndex != rootCache.pageIndex) {
            if (!oldPageIndex.empty()) {
                localManager.ReleasePageIndices(oldPageIndex);
            }
            localCache.pageIndex = rootCache.pageIndex;
            if (!localCache.pageIndex.empty()) {
                localManager.Pick(localCache.pageIndex);
            }
        }
        if (localCache.dims.size() == 3 && rootCache.dims.size() == 3) {
            std::vector<int> localDims = localCache.dims;
            int seqLen = 0;
            if (!rootCache.pageIndex.empty()) {
                seqLen = (int)(rootCache.pageIndex.size() - 1) * rootCache.pageLen + rootCache.lastPageLen;
            }
            localDims[1] = seqLen;
            localDims[2] = rootCache.dims[2];
            if (localCache.dims != localDims) {
                localCache.Resize(localDims);
            }
        }
        localCache.lastPageLen = rootCache.lastPageLen;
        localCache.pageLen = rootCache.pageLen;
        localCache.isPagedKVCache = true;
        localCache.pagedKVCacheData = &localManager;
    }

    static long long CountRange(const std::vector <int> &dims, int l, int r) {
        long long ret = 1;
        for (int i = l; i < r; i++) {
            ret *= dims[i];
        }
        return ret;
    }

    static size_t GetTensorSliceBytes(const fastllm::Data &data, int axis, int len) {
        long long elements = (long long)len * CountRange(data.dims, axis + 1, (int)data.dims.size());
        return (size_t)((elements * data.unitSize - 1) / data.unitSizeDiv + 1);
    }

    static size_t GetTensorRowBytes(const fastllm::Data &data, int axis) {
        long long elements = (long long)data.dims[axis] * CountRange(data.dims, axis + 1, (int)data.dims.size());
        return (size_t)((elements * data.unitSize - 1) / data.unitSizeDiv + 1);
    }

    static DivisionScheme BuildContiguousShardScheme(const std::vector <int> &devices,
                                                    std::map <int, int> &ratios,
                                                    int total, int unit) {
        DivisionScheme scheme;
        if (devices.empty() || total <= 0) {
            return scheme;
        }
        std::vector <int> points = FastllmMultiCudaGetSplitPoints((std::vector <int>&)devices, ratios, total, std::max(1, unit));
        for (int i = 0; i < (int)devices.size(); i++) {
            scheme[devices[i]];
            if (points[i] < points[i + 1]) {
                scheme[devices[i]].push_back({points[i], points[i + 1]});
            }
        }
        return scheme;
    }

    static DivisionScheme BuildPackedQKVShardScheme(const std::vector <int> &devices,
                                                    std::map <int, int> &ratios,
                                                    int qHeads, int kvHeads, int headDim) {
        DivisionScheme scheme;
        if (devices.empty() || kvHeads <= 0 || headDim <= 0) {
            return scheme;
        }
        std::vector <int> points = FastllmMultiCudaGetSplitPoints((std::vector <int>&)devices, ratios, kvHeads, 1);
        int group = qHeads / kvHeads;
        int qWidth = qHeads * headDim;
        int kvWidth = kvHeads * headDim;
        for (int i = 0; i < (int)devices.size(); i++) {
            int st = points[i], end = points[i + 1];
            scheme[devices[i]];
            if (st >= end) {
                continue;
            }
            int qSt = st * group * headDim, qEnd = end * group * headDim;
            int kSt = qWidth + st * headDim, kEnd = qWidth + end * headDim;
            int vSt = qWidth + kvWidth + st * headDim, vEnd = qWidth + kvWidth + end * headDim;
            scheme[devices[i]].push_back({qSt, qEnd});
            scheme[devices[i]].push_back({kSt, kEnd});
            scheme[devices[i]].push_back({vSt, vEnd});
        }
        return scheme;
    }

    static DivisionScheme BuildPairedHalfShardScheme(const std::vector <int> &devices,
                                                     std::map <int, int> &ratios,
                                                     int mid, int unit) {
        DivisionScheme scheme;
        if (devices.empty() || mid <= 0) {
            return scheme;
        }
        std::vector <int> points = FastllmMultiCudaGetSplitPoints((std::vector <int>&)devices, ratios, mid, std::max(1, unit));
        for (int i = 0; i < (int)devices.size(); i++) {
            int st = points[i], end = points[i + 1];
            scheme[devices[i]];
            if (st >= end) {
                continue;
            }
            scheme[devices[i]].push_back({st, end});
            scheme[devices[i]].push_back({mid + st, mid + end});
        }
        return scheme;
    }

    static bool IsSameDivisionScheme(const DivisionScheme &a, const DivisionScheme &b,
                                     const std::vector <int> &devices) {
        for (int device : devices) {
            auto ita = a.find(device), itb = b.find(device);
            const std::vector <std::pair <int, int> > *ra = ita == a.end() ? nullptr : &ita->second;
            const std::vector <std::pair <int, int> > *rb = itb == b.end() ? nullptr : &itb->second;
            if (ra == nullptr || ra->empty()) {
                if (rb != nullptr && !rb->empty()) {
                    return false;
                }
                continue;
            }
            if (rb == nullptr || *ra != *rb) {
                return false;
            }
        }
        return true;
    }

    static void RedistributeShardedTensor(fastllm::Data &data, const std::vector <int> &devices,
                                          const DivisionScheme &targetScheme) {
        if (!data.multiDeviceData || !data.IsTensorParallelSharded() || devices.empty()) {
            return;
        }
        if (IsSameDivisionScheme(data.tpRanges, targetScheme, devices)) {
            return;
        }

        int axis = NormalizeAxis(data.tpAxis, (int)data.dims.size());
        SyncShardedLocalShapeFromRoot(data, devices);

        auto oldLocalDatas = data.multiDeviceDatas;
        auto oldRanges = data.tpRanges;

        std::map <int, Data*> newLocalDatas;
        int oriDevice = FastllmCudaGetDevice();
        for (int device : devices) {
            int localLen = 0;
            for (auto &range : targetScheme.at(device)) {
                localLen += range.second - range.first;
            }
            std::vector <int> localDims = data.dims;
            localDims[axis] = localLen;
            Data *local = new Data(data.dataType, localDims);
            local->dataDevice = DataDevice::CUDA;
            local->dataDeviceIds = {device};
            local->name = data.name;
            local->tpLinearType = data.tpLinearType;
            local->tpPackType = data.tpPackType;
            local->tpQHeads = data.tpQHeads;
            local->tpKVHeads = data.tpKVHeads;
            local->tpHeadDim = data.tpHeadDim;
            if (local->Count(0) > 0) {
                FastllmCudaSetDevice(device);
                local->Allocate();
            }
            newLocalDatas[device] = local;
        }

        for (int targetDevice : devices) {
            Data *target = newLocalDatas[targetDevice];
            if (target == nullptr || target->Count(0) == 0) {
                continue;
            }
            int targetAxisBase = 0;
            for (auto &targetRange : targetScheme.at(targetDevice)) {
                int targetLen = targetRange.second - targetRange.first;
                int targetOffset = targetAxisBase;
                for (int sourceDevice : devices) {
                    Data *source = oldLocalDatas[sourceDevice];
                    if (source == nullptr || source->Count(0) == 0) {
                        continue;
                    }
                    int sourceAxisOffset = 0;
                    for (auto &sourceRange : oldRanges[sourceDevice]) {
                        int l = std::max(targetRange.first, sourceRange.first);
                        int r = std::min(targetRange.second, sourceRange.second);
                        if (l < r) {
                            int copyLen = r - l;
                            int sourceOffset = sourceAxisOffset + (l - sourceRange.first);
                            int destOffset = targetOffset + (l - targetRange.first);
                            size_t width = GetTensorSliceBytes(*source, axis, copyLen);
                            size_t spitch = GetTensorRowBytes(*source, axis);
                            size_t dpitch = GetTensorRowBytes(*target, axis);
                            size_t height = (size_t)CountRange(source->dims, 0, axis);
                            FastllmCudaMemcpy2DDeviceToDeviceAuto(
                                (uint8_t*)target->cudaData + GetTensorSliceBytes(*target, axis, destOffset), dpitch,
                                (uint8_t*)source->cudaData + GetTensorSliceBytes(*source, axis, sourceOffset), spitch,
                                width, height, targetDevice, sourceDevice
                            );
                        }
                        sourceAxisOffset += sourceRange.second - sourceRange.first;
                    }
                }
                targetAxisBase += targetLen;
            }
            SyncCudaAndCheck(targetDevice, "RedistributeShardedTensor");
        }

        FastllmCudaSetDevice(oriDevice);
        for (auto &it : oldLocalDatas) {
            delete it.second;
        }
        data.multiDeviceDatas = newLocalDatas;
        data.tpRanges = targetScheme;
        data.tpGlobalDims = data.dims;
        data.cudaData = nullptr;
    }

    static void EnsureEmptyPagedCacheCapacity(PagedCacheManager &manager, fastllm::Data &cache, int seqLen) {
        if (seqLen <= 0 || manager.pageLen <= 0) {
            return;
        }
        int currentUsedTokens = 0;
        if (!cache.pageIndex.empty()) {
            currentUsedTokens = (int)(cache.pageIndex.size() - 1) * cache.pageLen + cache.lastPageLen;
        }
        int neededPages = (currentUsedTokens + seqLen + manager.pageLen - 1) / manager.pageLen;
        if (neededPages <= manager.maxPages) {
            return;
        }
        if (cache.pageIndex.empty() && manager.FreePageCount() == manager.maxPages) {
            manager.SetMaxPages(neededPages);
            manager.Resize({neededPages, manager.pageLen, manager.dims[2], manager.dims[3]});
            manager.Allocate();
        }
    }

    static bool NeedTpCudaCheck() {
        static int cached = -1;
        if (cached == -1) {
            const char *env = getenv("FASTLLM_TP_CUDA_CHECK");
            cached = (env != nullptr && strcmp(env, "0") != 0) ? 1 : 0;
        }
        return cached != 0 || GetFastllmEnv().cudaSync;
    }

    static bool NeedTpDebugInfo() {
        return false;
    }

    static void SyncCudaAndCheck(int device, const char *where) {
        if (!NeedTpCudaCheck()) {
            return;
        }
        static const bool verbose = []() {
            const char *env = getenv("FASTLLM_TP_CUDA_CHECK_VERBOSE");
            return env != nullptr && strcmp(env, "0") != 0;
        }();
        if (verbose) {
            printf("[tp cuda check] device=%d where=%s\n", device, where == nullptr ? "unknown" : where);
            fflush(stdout);
        }
        FastllmCudaSyncDevice(device);
    }

    static void SyncCudaAndCheckAll(const std::vector <int> &devices, const char *where) {
        if (!NeedTpCudaCheck()) {
            return;
        }
        for (int device : devices) {
            SyncCudaAndCheck(device, where);
        }
    }

    struct MultiCudaDoRMSNormOp : MultiThreadBaseOp {
        Data *input, *weight, *output;
        float eps;
        int deviceId;

        MultiCudaDoRMSNormOp(Data *input, Data *weight, Data *output, float eps, int deviceId) :
                input(input), weight(weight), output(output), eps(eps), deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            output->Allocate();
            FastllmCudaRMSNorm(*input, *weight, *output, eps);
            SyncCudaAndCheck(deviceId, "MultiCudaRMSNormOp");
        }
    };

    struct MultiCudaDoAddToOp : MultiThreadBaseOp {
        Data *input0, *input1;
        float alpha;
        int deviceId;

        MultiCudaDoAddToOp(Data *input0, Data *input1, float alpha, int deviceId) :
                input0(input0), input1(input1), alpha(alpha), deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            FastllmCudaAddTo(*input0, *input1, alpha);
        }
    };

    bool MultiCudaDevice::Malloc(void **ret, size_t size) {
        *ret = FastllmCudaMalloc(size);
        return true;
    }

    bool MultiCudaDevice::Free(void *ret) {
        FastllmCudaFree(ret);
        return true;
    }

    bool MultiCudaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        FastllmCudaCopyFromHostToDevice(dst, src, size);
        return true;
    }

    bool MultiCudaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        FastllmCudaCopyFromDeviceToHost(dst, src, size);
        return true;
    }

    bool MultiCudaDevice::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            if (((BaseDevice*)this->cudaDevice)->ops.find(opType) == ((BaseDevice*)this->cudaDevice)->ops.end()) {
                return false;
            } else {
                return ((BaseDevice*)this->cudaDevice)->CanRun(opType, datas, floatParams, intParams);
            }
        } else {
            return this->ops[opType]->CanRun(opType, datas, floatParams, intParams);
        }
    }

    // 对某一个算子进行形状推理
    void MultiCudaDevice::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            ((BaseDevice*)this->cudaDevice)->Reshape(opType, datas, floatParams, intParams);
        } else {
            this->ops[opType]->Reshape(opType, datas, floatParams, intParams);
        }
    }

    // 对某一个算子进行推理
    void MultiCudaDevice::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            ((BaseDevice*)this->cudaDevice)->Run(opType, datas, floatParams, intParams);
        } else {
            this->ops[opType]->Run(opType, datas, floatParams, intParams);
        }
    }

    bool MultiCudaLinearAddOp::CanRun(const std::string &opType, const DataDict &datas,
                                      const FloatDict &floatParams, const IntDict &intParams) {
        return false;
    }

    void MultiCudaLinearAddOp::Reshape(const std::string &opType, const DataDict &datas,
                                       const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        AssertInFastLLM(weight.dims.size() == 2, "LinearAdd's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "LinearAdd's weight's shape error.\n");
        AssertInFastLLM(output.dims.back() == weight.dims[0], "LinearAdd's output's shape doesn't match weight.\n");
    }

    void MultiCudaLinearAddOp::Run(const std::string &opType, const DataDict &datas,
                                   const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
        Data &middle = *(datas.find("middle")->second);
        Data &output = *(datas.find("output")->second);
        Linear(input, weight, bias, middle);
        AddTo(output, middle);
    }

    bool MultiCudaLinearSwigluOp::CanRun(const std::string &opType, const DataDict &datas,
                                         const FloatDict &floatParams, const IntDict &intParams) {
        return true;
    }

    void MultiCudaLinearSwigluOp::Reshape(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &middle = *(datas.find("middle")->second);
        Data &output = *(datas.find("output")->second);
        AssertInFastLLM(weight.dims.size() == 2, "LinearSwiglu's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "LinearSwiglu's weight's shape error.\n");
        DoCudaLinearReshape(input, weight, middle);
        DoCudaSwigluReshape(middle, output);
    }

    void MultiCudaLinearSwigluOp::Run(const std::string &opType, const DataDict &datas,
                                      const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
        Data &middle = *(datas.find("middle")->second);
        Data &output = *(datas.find("output")->second);

        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);

        int n = input.Count(0) / input.dims.back();
        bool canFuse = devices.size() > 1 &&
                       !input.IsTensorParallelSharded() &&
                       weight.tpPackType == TP_PACK_GATEUP &&
                       input.dataType == DataType::FLOAT16 &&
                       weight.dataType == DataType::FLOAT16 &&
                       (bias.dataType == DataType::FLOAT32 || bias.dims.size() == 0) &&
                       n > 0 && n < 8;
        if (!canFuse) {
            /* printf("[MultiCuda LinearSwiglu fuse] disabled:\n");
            if (devices.size() <= 1) {
                printf("  - devices.size()=%zu (need >1)\n", devices.size());
            }
            if (input.IsTensorParallelSharded()) {
                printf("  - input is tensor-parallel sharded\n");
            }
            if (weight.tpPackType != TP_PACK_GATEUP) {
                printf("  - weight.tpPackType=%d (need TP_PACK_GATEUP=%d)\n",
                       (int) weight.tpPackType, (int) TP_PACK_GATEUP);
            }
            if (input.dataType != DataType::FLOAT16) {
                printf("  - input.dataType=%d (need FLOAT16=%d)\n",
                       (int) input.dataType, (int) DataType::FLOAT16);
            }
            if (weight.dataType != DataType::FLOAT16) {
                printf("  - weight.dataType=%d (need FLOAT16=%d)\n",
                       (int) weight.dataType, (int) DataType::FLOAT16);
            }
            if (!(bias.dataType == DataType::FLOAT32 || bias.dims.size() == 0)) {
                printf("  - bias: need FLOAT32 or empty (dims.size=%zu), got dataType=%d\n",
                       bias.dims.size(), (int) bias.dataType);
            }
            if (n <= 0 || n >= 8) {
                printf("  - batch token count n=%d (need 0 < n < 8)\n", n);
            } */
            Linear(input, weight, bias, middle);
            Swiglu(middle, output);
            return;
        }

        PrepareMultiCudaReplicatedData(input, devices, true);
        DivisionScheme divisionScheme = BuildMultiCudaRowSplitScheme(weight, devices, ratios);
        SplitMultiCudaWeight(weight, bias, devices, divisionScheme, 0);

        output.dataType = input.dataType;
        DoCudaSwigluReshape(middle, output);

        int mid = weight.dims[0] / 2;
        DivisionScheme outputScheme;
        for (int device : devices) {
            for (auto &range : divisionScheme[device]) {
                int l = std::max(0, range.first);
                int r = std::min(mid, range.second);
                if (l < r) {
                    outputScheme[device].push_back({l, r});
                }
            }
        }

        ResetMultiCudaTensor(output);
        output.multiDeviceData = true;
        output.tpLayout = TP_LAYOUT_SHARDED;
        output.tpAxis = (int)output.dims.size() - 1;
        output.tpGlobalDims = output.dims;
        output.tpRanges = outputScheme;
        output.cudaData = nullptr;
        for (int device : devices) {
            std::vector <int> localDims = output.dims;
            int localLen = 0;
            for (auto &range : outputScheme[device]) {
                localLen += range.second - range.first;
            }
            localDims.back() = localLen;
            Data *localOutput = new Data(output.dataType, localDims);
            localOutput->dataDevice = DataDevice::CUDA;
            localOutput->dataDeviceIds = {device};
            output.multiDeviceDatas[device] = localOutput;
        }

        struct MultiCudaDoLinearSwigluShardOp : MultiThreadBaseOp {
            Data *input, *weight, *bias, *output;
            int deviceId;

            MultiCudaDoLinearSwigluShardOp(Data *input, Data *weight, Data *bias, Data *output, int deviceId) :
                    input(input), weight(weight), bias(bias), output(output), deviceId(deviceId) {}

            void Run() {
                FastllmCudaSetDevice(deviceId);
                int n = input->Count(0) / input->dims.back();
                int m = input->dims.back();
                int k = output->dims.back();
                bool ok = FastllmCudaHalfMatMulFloat16Swiglu(*input, *weight, *bias, *output, n, m, k);
                AssertInFastLLM(ok, "MultiCudaLinearSwigluOp fused path failed.\n");
            }
        };

        auto *pool = fastllm::GetAlivePool();
        std::vector <fastllm::MultiThreadBaseOp*> ops;
        ops.reserve(devices.size());
        for (int device : devices) {
            ops.push_back(new MultiCudaDoLinearSwigluShardOp(
                input.multiDeviceDatas[device],
                weight.multiDeviceDatas[device],
                bias.multiDeviceDatas[device],
                output.multiDeviceDatas[device],
                device
            ));
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void MultiCudaRMSNormOp::Run(const std::string &opType, const DataDict &datas,
                                 const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5f;

        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1) {
            output.dataType = input.dataType;
            output.Resize(input.dims);
            output.Allocate();
            FastllmCudaRMSNorm(input, weight, output, eps);
            return;
        }

        EnsureReplicatedMultiCudaTensor(weight, devices, true);

        bool sharded = input.IsTensorParallelSharded();
        if (sharded) {
            SyncShardedLocalShapeFromRoot(input, devices);
            if (&input != &output) {
                output.dataType = input.dataType;
                output.Resize(input.dims);
                ResetMultiCudaTensor(output);
                CopyShardedLayout(output, input, devices);
                SyncShardedLocalShapeFromRoot(output, devices);
            }
        } else {
            EnsureReplicatedMultiCudaTensor(input, devices, true);
            SyncReplicatedLocalShapeFromRoot(input, devices);
            if (&input != &output) {
                output.dataType = input.dataType;
                output.Resize(input.dims);
                EnsureReplicatedMultiCudaTensor(output, devices, false);
                SyncReplicatedLocalShapeFromRoot(output, devices);
            }
        }

        auto *pool = fastllm::GetAlivePool();
        std::vector <fastllm::MultiThreadBaseOp*> ops;
        ops.reserve(devices.size());
        for (int device : devices) {
            ops.push_back(new MultiCudaDoRMSNormOp(
                input.multiDeviceDatas[device],
                weight.multiDeviceDatas[device],
                (&input == &output) ? input.multiDeviceDatas[device] : output.multiDeviceDatas[device],
                eps, device
            ));
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }

        if (!sharded) {
            // SyncReplicatedRootFromReplica((&input == &output) ? input : output, devices);
        }
    }

    void MultiCudaAddToOp::Run(const std::string &opType, const DataDict &datas,
                               const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;

        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1) {
            FastllmCudaAddTo(input0, input1, alpha);
            return;
        }

        EnsureReplicatedMultiCudaTensor(input0, devices, true);
        EnsureReplicatedMultiCudaTensor(input1, devices, true);

        auto *pool = fastllm::GetAlivePool();
        std::vector <fastllm::MultiThreadBaseOp*> ops;
        ops.reserve(devices.size());
        for (int device : devices) {
            ops.push_back(new MultiCudaDoAddToOp(
                input0.multiDeviceDatas[device],
                input1.multiDeviceDatas[device],
                alpha, device
            ));
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }

        // SyncReplicatedRootFromReplica(input0, devices);
    }

    void MultiCudaSplitOp::Reshape(const std::string &opType, const DataDict &datas,
                                   const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = NormalizeAxis(intParams.find("axis")->second, (int)input.dims.size());
        int start = intParams.find("start")->second;
        int end = intParams.find("end")->second;
        std::vector <int> dims = input.dims;
        dims[axis] = end - start;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void MultiCudaSplitOp::Run(const std::string &opType, const DataDict &datas,
                               const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = NormalizeAxis(intParams.find("axis")->second, (int)input.dims.size());
        int start = intParams.find("start")->second;
        int end = intParams.find("end")->second;

        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1 || !input.multiDeviceData) {
            DoCudaSplitReshape(input, axis, start, end, output);
            DoCudaSplit(input, axis, start, end, output);
            return;
        }

        std::vector <int> dims = input.dims;
        dims[axis] = end - start;
        output.dataType = input.dataType;
        output.Resize(dims);

        if (input.IsTensorParallelReplicated()) {
            EnsureReplicatedMultiCudaTensor(input, devices, true);
            SyncReplicatedLocalShapeFromRoot(input, devices);
            EnsureReplicatedMultiCudaTensor(output, devices, false);
            SyncReplicatedLocalShapeFromRoot(output, devices);
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                DoCudaSplitReshape(*input.multiDeviceDatas[device], axis, start, end, *output.multiDeviceDatas[device]);
                DoCudaSplit(*input.multiDeviceDatas[device], axis, start, end, *output.multiDeviceDatas[device]);
            }
            SyncReplicatedRootFromReplica(output, devices);
            return;
        }

        AssertInFastLLM(input.IsTensorParallelSharded(),
                        "MultiCudaSplit only supports tensor-parallel sharded or replicated input.\n");

        int tpAxisNorm = NormalizeAxis(input.tpAxis, (int)input.dims.size());

        if (axis != tpAxisNorm) {
            SyncShardedLocalShapeFromRoot(input, devices);
            ResetMultiCudaTensor(output);
            output.multiDeviceData = true;
            output.tpLayout = TP_LAYOUT_SHARDED;
            output.tpAxis = input.tpAxis;
            output.tpGlobalDims = output.dims;
            output.tpRanges = input.tpRanges;
            output.cudaData = nullptr;

            for (int device : devices) {
                Data *localInput = input.multiDeviceDatas[device];
                if (localInput == nullptr) continue;
                FastllmCudaSetDevice(device);
                Data *localOutput = new Data(output.dataType);
                localOutput->dataDevice = DataDevice::CUDA;
                localOutput->dataDeviceIds = {device};
                DoCudaSplitReshape(*localInput, axis, start, end, *localOutput);
                DoCudaSplit(*localInput, axis, start, end, *localOutput);
                output.multiDeviceDatas[device] = localOutput;
            }
            return;
        }

        SyncShardedLocalShapeFromRoot(input, devices);
        ResetMultiCudaTensor(output);
        output.multiDeviceData = true;
        output.tpLayout = TP_LAYOUT_SHARDED;
        output.tpAxis = axis;
        output.tpGlobalDims = output.dims;
        output.cudaData = nullptr;

        for (int device : devices) {
            Data *localInput = input.multiDeviceDatas[device];
            int localOffset = 0;
            int localStart = -1;
            int localLen = 0;
            for (auto &range : input.tpRanges[device]) {
                int l = std::max(start, range.first);
                int r = std::min(end, range.second);
                if (l < r) {
                    AssertInFastLLM(localStart == -1, "MultiCudaSplit only supports single-segment local split now.\n");
                    localStart = localOffset + (l - range.first);
                    localLen = r - l;
                    output.tpRanges[device].push_back({l - start, r - start});
                }
                localOffset += range.second - range.first;
            }

            std::vector <int> localDims = localInput->dims;
            localDims[axis] = localLen;
            Data *localOutput = new Data(output.dataType, localDims);
            localOutput->dataDevice = DataDevice::CUDA;
            localOutput->dataDeviceIds = {device};
            output.multiDeviceDatas[device] = localOutput;
            if (localLen > 0) {
                FastllmCudaSetDevice(device);
                DoCudaSplitReshape(*localInput, axis, localStart, localStart + localLen, *localOutput);
                DoCudaSplit(*localInput, axis, localStart, localStart + localLen, *localOutput);
            }
        }
    }

    void MultiCudaSwigluOp::Reshape(const std::string &opType, const DataDict &datas,
                                    const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        DoCudaSwigluReshape(input, output);
    }

    void MultiCudaSwigluOp::Run(const std::string &opType, const DataDict &datas,
                                const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1 || !input.multiDeviceData) {
            DoCudaSwigluReshape(input, output);
            DoCudaSwiglu(input, output);
            return;
        }

        if (input.IsTensorParallelReplicated()) {
            EnsureReplicatedMultiCudaTensor(input, devices, true);
            SyncReplicatedLocalShapeFromRoot(input, devices);
            output.dataType = input.dataType;
            DoCudaSwigluReshape(input, output);
            EnsureReplicatedMultiCudaTensor(output, devices, false);
            SyncReplicatedLocalShapeFromRoot(output, devices);
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                DoCudaSwigluReshape(*input.multiDeviceDatas[device], *output.multiDeviceDatas[device]);
                DoCudaSwiglu(*input.multiDeviceDatas[device], *output.multiDeviceDatas[device]);
            }
            SyncReplicatedRootFromReplica(output, devices);
            return;
        }

        AssertInFastLLM(input.IsTensorParallelSharded(),
                        "MultiCudaSwiglu only supports tensor-parallel sharded or replicated input.\n");
        SyncShardedLocalShapeFromRoot(input, devices);
        int axis = NormalizeAxis(input.tpAxis, (int)input.dims.size());
        AssertInFastLLM(axis == (int)input.dims.size() - 1,
                        "MultiCudaSwiglu currently requires the sharded axis to be the last axis.\n");
        int mid = input.dims[axis] / 2;
        int unit = 128;
        while (unit > 1 && mid % unit != 0) {
            unit >>= 1;
        }
        RedistributeShardedTensor(input, devices, BuildPairedHalfShardScheme(devices, ratios, mid, unit));
        SyncShardedLocalShapeFromRoot(input, devices);
        output.dataType = input.dataType;
        DoCudaSwigluReshape(input, output);
        ResetMultiCudaTensor(output);
        output.multiDeviceData = true;
        output.tpLayout = TP_LAYOUT_SHARDED;
        output.tpAxis = input.tpAxis;
        output.tpGlobalDims = output.dims;
        output.cudaData = nullptr;

        for (int device : devices) {
            Data *localInput = input.multiDeviceDatas[device];
            std::vector <int> localDims = localInput->dims;
            localDims[axis] /= 2;
            Data *localOutput = new Data(output.dataType, localDims);
            localOutput->dataDevice = DataDevice::CUDA;
            localOutput->dataDeviceIds = {device};
            output.multiDeviceDatas[device] = localOutput;
            for (auto &range : input.tpRanges[device]) {
                int l = std::max(0, range.first);
                int r = std::min(mid, range.second);
                if (l < r) {
                    output.tpRanges[device].push_back({l, r});
                }
            }
            FastllmCudaSetDevice(device);
            DoCudaSwigluReshape(*localInput, *localOutput);
            DoCudaSwiglu(*localInput, *localOutput);
        }
    }

    void MultiCudaPermuteSelfOp::Run(const std::string &opType, const DataDict &datas,
                                     const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t*)axisData.cpuData)[i]);
        }
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1 || !input.multiDeviceData) {
            DoCudaPermuteSelf(input, axis);
            return;
        }

        if (input.IsTensorParallelReplicated()) {
            SyncReplicatedLocalShapeFromRoot(input, devices);
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                DoCudaPermuteSelf(*input.multiDeviceDatas[device], axis);
            }
            SyncCudaAndCheckAll(devices, "MultiCudaPermuteSelfOp");
            std::vector <int> newDims(axis.size());
            for (int i = 0; i < (int)axis.size(); i++) {
                newDims[i] = input.dims[axis[i]];
            }
            input.Resize(newDims);
            SyncReplicatedRootFromReplica(input, devices);
            return;
        }

        SyncShardedLocalShapeFromRoot(input, devices);
        int oldTpAxis = NormalizeAxis(input.tpAxis, (int)input.dims.size());
        RedistributeShardedTensor(input, devices, BuildContiguousShardScheme(devices, ratios, input.dims[oldTpAxis], 1));
        SyncShardedLocalShapeFromRoot(input, devices);
        for (int device : devices) {
            FastllmCudaSetDevice(device);
            DoCudaPermuteSelf(*input.multiDeviceDatas[device], axis);
        }
        SyncCudaAndCheckAll(devices, "MultiCudaPermuteSelfOp");
        std::vector <int> newDims(axis.size());
        for (int i = 0; i < (int)axis.size(); i++) {
            newDims[i] = input.dims[axis[i]];
        }
        int oldAxis = NormalizeAxis(input.tpAxis, (int)input.dims.size());
        input.Resize(newDims);
        for (int i = 0; i < (int)axis.size(); i++) {
            if (axis[i] == oldAxis) {
                input.tpAxis = i;
                break;
            }
        }
        input.tpGlobalDims = input.dims;
    }

    void MultiCudaRopeEncodingOp::Run(const std::string &opType, const DataDict &datas,
                                      const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        float ropeTheta = floatParams.find("ropeTheta") != floatParams.end() ? floatParams.find("ropeTheta")->second : 10000.0f;
        float ropeScale = floatParams.find("ropeScale") != floatParams.end() ? floatParams.find("ropeScale")->second : 1.0f;

        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1 || !input.multiDeviceData) {
            FastllmCudaRopeEncoding(input, positionIds, rotaryDim, ropeTheta, ropeScale);
            return;
        }

        EnsureReplicatedMultiCudaTensor(positionIds, devices, true);
        if (input.IsTensorParallelReplicated()) {
            SyncReplicatedLocalShapeFromRoot(input, devices);
        } else {
            SyncShardedLocalShapeFromRoot(input, devices);
        }
        for (int device : devices) {
            FastllmCudaSetDevice(device);
            FastllmCudaRopeEncoding(*input.multiDeviceDatas[device], *positionIds.multiDeviceDatas[device], rotaryDim, ropeTheta, ropeScale);
        }
        SyncCudaAndCheckAll(devices, "MultiCudaRopeEncodingOp");
        if (input.IsTensorParallelReplicated()) {
            SyncReplicatedRootFromReplica(input, devices);
        }
    }

    void MultiCudaAppendPagedCacheOp::Reshape(const std::string &opType, const DataDict &datas,
                                              const FloatDict &floatParams, const IntDict &intParams) {
        Data &cache = *(datas.find("cache")->second);
        Data &input = *(datas.find("input")->second);
        PagedCacheManager &pagedCacheManager = *((PagedCacheManager*)datas.find("pagedCacheManager")->second);

        cache.isPagedKVCache = true;
        if (cache.pagedKVCacheData == nullptr) {
            cache.pagedKVCacheData = &pagedCacheManager;
        }
        cache.pageLen = pagedCacheManager.pageLen;
        if (cache.dims.empty()) {
            cache.Resize(input.dims);
        } else {
            cache.Resize({cache.dims[0], cache.dims[1] + input.dims[1], cache.dims[2]});
        }
    }

    void MultiCudaAppendPagedCacheOp::Run(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &cache = *(datas.find("cache")->second);
        Data &input = *(datas.find("input")->second);
        PagedCacheManager &rootManager = *((PagedCacheManager*)datas.find("pagedCacheManager")->second);

        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1 || !input.multiDeviceData) {
            AppendPagedCacheLocal(rootManager, cache, input);
            cache.pagedKVCacheData = &rootManager;
            cache.pageLen = rootManager.pageLen;
            cache.isPagedKVCache = true;
            return;
        }

        AssertInFastLLM(input.IsTensorParallelSharded(),
                        "MultiCudaAppendPagedCache only supports sharded input now.\n");
        SyncShardedLocalShapeFromRoot(input, devices);
        EnsureEmptyPagedCacheCapacity(rootManager, cache, input.dims[1]);

        if (!cache.multiDeviceData) {
            cache.multiDeviceData = true;
            cache.tpLayout = TP_LAYOUT_SHARDED;
            cache.tpAxis = 0;
            cache.tpRanges = input.tpRanges;
        }
        cache.tpGlobalDims = {input.dims[0], cache.dims.empty() ? input.dims[1] : cache.dims[1] + input.dims[1], input.dims[2]};

        for (int device : devices) {
            Data *localInput = input.multiDeviceDatas[device];
            Data *localCache = nullptr;
            auto it = cache.multiDeviceDatas.find(device);
            if (it == cache.multiDeviceDatas.end() || it->second == nullptr) {
                localCache = new Data(cache.dataType);
                localCache->dataDevice = DataDevice::CUDA;
                localCache->dataDeviceIds = {device};
                cache.multiDeviceDatas[device] = localCache;
            } else {
                localCache = it->second;
            }
            PagedCacheManager *localManager = GetOrCreateLocalPagedCacheManager(rootManager, *localInput, device);
            if (NeedTpDebugInfo()) {
                printf("[tp append mgr] device=%d mgr=[%d,%d,%d,%d] ptr=%p free=%d maxPages=%d input=[%d,%d,%d]\n",
                       device,
                       localManager->dims.size() > 0 ? localManager->dims[0] : -1,
                       localManager->dims.size() > 1 ? localManager->dims[1] : -1,
                       localManager->dims.size() > 2 ? localManager->dims[2] : -1,
                       localManager->dims.size() > 3 ? localManager->dims[3] : -1,
                       localManager->cudaData,
                       localManager->FreePageCount(),
                       localManager->maxPages,
                       localInput->dims.size() > 0 ? localInput->dims[0] : -1,
                       localInput->dims.size() > 1 ? localInput->dims[1] : -1,
                       localInput->dims.size() > 2 ? localInput->dims[2] : -1);
                fflush(stdout);
            }
            SyncCudaAndCheck(device, "MultiCudaAppendPagedCacheOp::localManagerReady");
            localCache->dataType = cache.dataType;
            localCache->UpdateUnitSize();
            localCache->pagedKVCacheData = localManager;
            localCache->pageLen = localManager->pageLen;
            localCache->isPagedKVCache = true;
            EnsureEmptyPagedCacheCapacity(*localManager, *localCache, localInput->dims[1]);
            SyncCudaAndCheck(device, "MultiCudaAppendPagedCacheOp::capacityReady");
            static int appendDebugCount = 0;
            if (NeedTpDebugInfo() && appendDebugCount < 12) {
                printf("[tp append] device=%d input=[%d,%d,%d] cache=[%d,%d,%d] maxPages=%d free=%zu trie=%zu pageLen=%d\n",
                       device,
                       localInput->dims.size() > 0 ? localInput->dims[0] : -1,
                       localInput->dims.size() > 1 ? localInput->dims[1] : -1,
                       localInput->dims.size() > 2 ? localInput->dims[2] : -1,
                       localCache->dims.size() > 0 ? localCache->dims[0] : -1,
                       localCache->dims.size() > 1 ? localCache->dims[1] : -1,
                       localCache->dims.size() > 2 ? localCache->dims[2] : -1,
                       localManager->maxPages,
                       localManager->freePages.size(),
                       localManager->triePages.size(),
                       localManager->pageLen);
                fflush(stdout);
                appendDebugCount++;
            }
            FastllmCudaSetDevice(device);
            AppendPagedCacheLocal(*localManager, *localCache, *localInput);
            SyncCudaAndCheck(device, "MultiCudaAppendPagedCacheOp");
        }

        int rootDevice = devices[0];
        Data *rootLocal = cache.multiDeviceDatas[rootDevice];
        SyncRootPagedCacheMeta(cache, *rootLocal, rootManager);
        cache.Resize({input.dims[0], rootLocal->dims[1], input.dims[2]});
    }

    void MultiCudaAttentionPagedBatchOp::Run(const std::string &opType, const DataDict &datas,
                                             const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &kCaches = *(datas.find("kCaches")->second);
        Data &vCaches = *(datas.find("vCaches")->second);
        Data &qSizes = *(datas.find("qSizes")->second);
        Data &pageSizes = *(datas.find("pageSizes")->second);
        Data &pageIndexs = *(datas.find("pageIndexs")->second);
        Data &lastPageLens = *(datas.find("lastPageLens")->second);
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / kCaches.dims[0];
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0f;
        int attentionType = intParams.find("attentionType") != intParams.end() ? intParams.find("attentionType")->second : 0;
        bool inited = intParams.find("inited") != intParams.end() ? (intParams.find("inited")->second != 0) : false;

        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1 || !q.multiDeviceData) {
            output.Allocate();
            FastllmCudaHalfPagedAttentionBatch(q, kCaches, vCaches, qSizes, pageSizes, pageIndexs, lastPageLens,
                                               output, group, scale, attentionType, inited);
            return;
        }

        AssertInFastLLM(q.IsTensorParallelSharded(), "MultiCudaAttentionPagedBatch only supports sharded q.\n");
        SyncShardedLocalShapeFromRoot(q, devices);
        EnsureReplicatedMultiCudaTensor(qSizes, devices, true);
        EnsureReplicatedMultiCudaTensor(pageSizes, devices, true);
        EnsureReplicatedMultiCudaTensor(pageIndexs, devices, true);
        EnsureReplicatedMultiCudaTensor(lastPageLens, devices, true);

        output.dataType = q.dataType;
        output.Resize(q.dims);
        if (!output.multiDeviceData || !output.IsTensorParallelSharded()) {
            ResetMultiCudaTensor(output);
            CopyShardedLayout(output, q, devices);
            SyncShardedLocalShapeFromRoot(output, devices);
        } else {
            output.tpAxis = q.tpAxis;
            output.tpGlobalDims = q.dims;
            output.tpRanges = q.tpRanges;
            output.cudaData = nullptr;
            for (int device : devices) {
                auto qIt = q.multiDeviceDatas.find(device);
                if (qIt == q.multiDeviceDatas.end() || qIt->second == nullptr) continue;
                auto oIt = output.multiDeviceDatas.find(device);
                if (oIt == output.multiDeviceDatas.end() || oIt->second == nullptr) {
                    Data *local = new Data(output.dataType, qIt->second->dims);
                    local->dataDevice = DataDevice::CUDA;
                    local->dataDeviceIds = {device};
                    output.multiDeviceDatas[device] = local;
                } else {
                    oIt->second->dataType = output.dataType;
                    oIt->second->Resize(qIt->second->dims);
                }
            }
        }

        if (!kCaches.multiDeviceData) {
            kCaches.multiDeviceData = true;
            kCaches.tpLayout = TP_LAYOUT_SHARDED;
            kCaches.tpAxis = 0;
        }
        if (!vCaches.multiDeviceData) {
            vCaches.multiDeviceData = true;
            vCaches.tpLayout = TP_LAYOUT_SHARDED;
            vCaches.tpAxis = 0;
        }

        PagedCacheManager &rootKManager = *(PagedCacheManager*)kCaches.pagedKVCacheData;
        PagedCacheManager &rootVManager = *(PagedCacheManager*)vCaches.pagedKVCacheData;

        for (int device : devices) {
            Data *localQ = q.multiDeviceDatas[device];
            Data *localKCache = nullptr;
            Data *localVCache = nullptr;
            if (kCaches.multiDeviceDatas.find(device) == kCaches.multiDeviceDatas.end() || kCaches.multiDeviceDatas[device] == nullptr) {
                int localKHeads = localQ->dims[0] / std::max(1, group);
                localKCache = new Data(kCaches.dataType, {localKHeads, kCaches.dims[1], kCaches.dims[2]});
                localKCache->dataDevice = DataDevice::CUDA;
                localKCache->dataDeviceIds = {device};
                kCaches.multiDeviceDatas[device] = localKCache;
            } else {
                localKCache = kCaches.multiDeviceDatas[device];
            }
            if (vCaches.multiDeviceDatas.find(device) == vCaches.multiDeviceDatas.end() || vCaches.multiDeviceDatas[device] == nullptr) {
                int localVHeads = localQ->dims[0] / std::max(1, group);
                localVCache = new Data(vCaches.dataType, {localVHeads, vCaches.dims[1], vCaches.dims[2]});
                localVCache->dataDevice = DataDevice::CUDA;
                localVCache->dataDeviceIds = {device};
                vCaches.multiDeviceDatas[device] = localVCache;
            } else {
                localVCache = vCaches.multiDeviceDatas[device];
            }
            PagedCacheManager *localKManager = GetOrCreateLocalPagedCacheManager(rootKManager, *localKCache, device);
            PagedCacheManager *localVManager = GetOrCreateLocalPagedCacheManager(rootVManager, *localVCache, device);
            SyncLocalPagedCacheMeta(*localKCache, kCaches, *localKManager);
            SyncLocalPagedCacheMeta(*localVCache, vCaches, *localVManager);
            FastllmCudaSetDevice(device);
            output.multiDeviceDatas[device]->Allocate();
            FastllmCudaHalfPagedAttentionBatch(
                *localQ, *localKCache, *localVCache,
                *qSizes.multiDeviceDatas[device], *pageSizes.multiDeviceDatas[device],
                *pageIndexs.multiDeviceDatas[device], *lastPageLens.multiDeviceDatas[device],
                *output.multiDeviceDatas[device],
                localQ->dims[0] / std::max(1, localKCache->dims[0]),
                scale, attentionType, inited, false
            );
        }
        SyncCudaAndCheckAll(devices, "MultiCudaAttentionPagedBatchOp");
    }

    void MultiCudaAttentionPagedBatchOp::Reshape(const std::string &opType, const DataDict &datas,
                                                 const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &vCaches = *(datas.find("vCaches")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(vCaches.pagedKVCacheData != nullptr,
                        "MultiCudaAttentionPagedBatchOp: vCaches.pagedKVCacheData should not be nullptr.\n");
        AssertInFastLLM(vCaches.pagedKVCacheData->dims.size() == 4,
                        "MultiCudaAttentionPagedBatchOp: paged KV cache dims should be 4.\n");

        output.dataType = q.dataType;
        output.Resize({q.dims[0], q.dims[1], vCaches.pagedKVCacheData->dims[3]});
    }

    void MultiCudaQKVRMSNormRopeSplitAppendPagedCacheOp::Run(const std::string &opType, const DataDict &datas,
                                                             const FloatDict &floatParams, const IntDict &intParams) {
        Data &qkv = *(datas.find("qkv")->second);
        Data &qNormWeight = *(datas.find("qNormWeight")->second);
        Data &kNormWeight = *(datas.find("kNormWeight")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &qOutput = *(datas.find("qOutput")->second);
        Data &pagedKCacheData = *(datas.find("pagedKCacheData")->second);
        Data &pagedVCacheData = *(datas.find("pagedVCacheData")->second);
        Data &insertIndexs = *(datas.find("insertIndexs")->second);
        Data &insertPositions = *(datas.find("insertPositions")->second);

        int q_heads = intParams.find("q_heads")->second;
        int k_heads = intParams.find("k_heads")->second;
        int head_dim = intParams.find("head_dim")->second;
        int rotateDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        int pageLen = intParams.find("pageLen")->second;
        int batch = intParams.find("batch")->second;
        float eps = floatParams.find("eps")->second;
        float ropeTheta = floatParams.find("ropeTheta") != floatParams.end() ? floatParams.find("ropeTheta")->second : 10000.0f;
        float ropeScale = floatParams.find("ropeScale") != floatParams.end() ? floatParams.find("ropeScale")->second : 1.0f;
        int doQKNorm = intParams.find("doQKNorm") != intParams.end() ? intParams.find("doQKNorm")->second : 1;

        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1 || !qkv.multiDeviceData) {
            qOutput.Allocate();
            FastllmCudaQKVRMSNormRopeSplitAppendPagedCache(
                qkv, qNormWeight, kNormWeight, positionIds,
                qOutput,
                (uint8_t*)pagedKCacheData.cudaData, (uint8_t*)pagedVCacheData.cudaData,
                (int32_t*)insertIndexs.cudaData, (int32_t*)insertPositions.cudaData,
                q_heads, k_heads, head_dim, rotateDim, eps, ropeTheta, ropeScale,
                pageLen, pagedKCacheData.dataType, batch, doQKNorm
            );
            return;
        }

        AssertInFastLLM(qkv.IsTensorParallelSharded(), "MultiCudaQKVRMSNormRopeSplitAppendPagedCache requires sharded qkv.\n");
        SyncShardedLocalShapeFromRoot(qkv, devices);
        RedistributeShardedTensor(qkv, devices, BuildPackedQKVShardScheme(devices, ratios, q_heads, k_heads, head_dim));
        SyncShardedLocalShapeFromRoot(qkv, devices);
        EnsureReplicatedMultiCudaTensor(qNormWeight, devices, true);
        EnsureReplicatedMultiCudaTensor(kNormWeight, devices, true);
        EnsureReplicatedMultiCudaTensor(positionIds, devices, true);
        EnsureReplicatedMultiCudaTensor(insertIndexs, devices, true);
        EnsureReplicatedMultiCudaTensor(insertPositions, devices, true);

        qOutput.dataType = qkv.dataType;
        qOutput.Resize({qkv.dims[0] * q_heads, qkv.dims[1], head_dim});
        ResetMultiCudaTensor(qOutput);
        qOutput.multiDeviceData = true;
        qOutput.tpLayout = TP_LAYOUT_SHARDED;
        qOutput.tpAxis = 0;
        qOutput.tpGlobalDims = qOutput.dims;
        qOutput.cudaData = nullptr;

        PagedCacheManager &rootKManager = *(PagedCacheManager*)&pagedKCacheData;
        PagedCacheManager &rootVManager = *(PagedCacheManager*)&pagedVCacheData;
        int group = q_heads / k_heads;
        int qHeadOffset = 0;

        for (int device : devices) {
            Data *localQKV = qkv.multiDeviceDatas[device];
            int per = localQKV->dims.back() / (group + 2);
            int localKHeads = per / head_dim;
            int localQHeads = localKHeads * group;
            int localQBatchHeads = localQHeads * localQKV->dims[0];

            std::vector <int> localQDims = {localQBatchHeads, localQKV->dims[1], head_dim};
            Data *localQ = new Data(qOutput.dataType, localQDims);
            localQ->dataDevice = DataDevice::CUDA;
            localQ->dataDeviceIds = {device};
            qOutput.multiDeviceDatas[device] = localQ;
            qOutput.tpRanges[device].push_back({qHeadOffset, qHeadOffset + localQBatchHeads});
            qHeadOffset += localQBatchHeads;

            Data localKDesc(qkv.dataType, {localKHeads, 1, head_dim});
            localKDesc.dataDevice = DataDevice::CUDA;
            localKDesc.dataDeviceIds = {device};
            PagedCacheManager *localKManager = GetOrCreateLocalPagedCacheManager(rootKManager, localKDesc, device);
            PagedCacheManager *localVManager = GetOrCreateLocalPagedCacheManager(rootVManager, localKDesc, device);

            FastllmCudaSetDevice(device);
            localQ->Allocate(false);
            AssertInFastLLM(localKManager->cudaData != nullptr && localVManager->cudaData != nullptr,
                            "MultiCudaQKVRMSNormRopeSplitAppendPagedCacheOp: local paged cache manager is not allocated.\n");

            FastllmCudaQKVRMSNormRopeSplitAppendPagedCache(
                *localQKV,
                *qNormWeight.multiDeviceDatas[device],
                *kNormWeight.multiDeviceDatas[device],
                *positionIds.multiDeviceDatas[device],
                *localQ,
                (uint8_t*)localKManager->cudaData,
                (uint8_t*)localVManager->cudaData,
                (int32_t*)insertIndexs.multiDeviceDatas[device]->cudaData,
                (int32_t*)insertPositions.multiDeviceDatas[device]->cudaData,
                localQHeads, localKHeads, head_dim, rotateDim, eps, ropeTheta, ropeScale,
                pageLen, localKManager->dataType, batch, doQKNorm
            );
        }
        SyncCudaAndCheckAll(devices, "MultiCudaQKVRMSNormRopeSplitAppendPagedCacheOp");
    }

    struct MultiCudaDoMergeMLPOp : MultiThreadBaseOp {
        uint8_t *oriCudaInput, *oriCpuInput; // 移除了 partOutput
        Data *input, *weight0, *bias0, *weight1, *bias1;
        Data *w1, *w2, *w3;
        Data *output;
        int deviceId;

        MultiCudaDoMergeMLPOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput, 
                            Data *input, Data *weight0, Data *bias0, Data *weight1, Data *bias1, 
                            Data *w1, Data *w2, Data *w3,
                            Data *output, int deviceId) : 
                oriCudaInput(oriCudaInput), oriCpuInput(oriCpuInput), // 移除了 partOutput 初始化
                input(input), weight0(weight0), bias0(bias0), weight1(weight1), bias1(bias1), 
                w1(w1), w2(w2), w3(w3), 
                output(output), deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            AssertInFastLLM(input->cudaData != nullptr, "MultiCudaDoMergeMLPOp: local input should be prepared.\n");

            DoCudaLinearReshape(*input, *weight0, *w3);
            if (bias0 == nullptr) {
                DoCudaLinear(*input, *weight0, *GetEmptyData(), *w3);
            } else {
                DoCudaLinear(*input, *weight0, *bias0, *w3);
            }

            DoCudaSwigluReshape(*w3, *w1);
            DoCudaSwiglu(*w3, *w1);

            DoCudaLinearReshape(*w1, *weight1, *output);
            output->Allocate();
            if (bias1 == nullptr) {
                DoCudaLinear(*w1, *weight1, *GetEmptyData(), *output);
            } else {
                DoCudaLinear(*w1, *weight1, *bias1, *output);
            }

            FastllmNcclAllReduce(output->cudaData, output->cudaData, output->Count(0), output->dataType, deviceId);
        }
    };

    struct MultiCudaCpuDoMergeMLPOp : MultiThreadBaseOp {
        uint8_t *oriCpuInput, *partOutput;
        Data *input, *weight0, *bias0, *weight1, *bias1;
        Data *w1, *w2, *w3;
        Data *output;
        int deviceId;

        MultiCudaCpuDoMergeMLPOp(uint8_t *oriCpuInput, uint8_t *partOutput,
                            Data *input, Data *weight0, Data *bias0, Data *weight1, Data *bias1, 
                            Data *w1, Data *w2, Data *w3,
                            Data *output, int deviceId) : 
                oriCpuInput(oriCpuInput), partOutput(partOutput),
                input(input), weight0(weight0), bias0(bias0), weight1(weight1), bias1(bias1), 
                w1(w1), w2(w2), w3(w3), 
                output(output), deviceId(deviceId) {}

        void Run() {
            input->Allocate();
            memcpy(input->cpuData, oriCpuInput, input->GetBytes());

            DoCpuLinearReshape(*input, *weight0, *w3);
            DoCpuLinear(*input, *weight0, bias0 == nullptr ? Data() : *bias0, *w3);

            DoCpuSwigluReshape(*w3, *w1);
            DoCpuSwiglu(*w3, *w1);

            DoCpuLinearReshape(*w1, *weight1, *output);
            DoCpuLinear(*w1, *weight1, bias1 == nullptr ? Data() : *bias1, *output);

            FastllmCudaCopyFromHostToDevice(partOutput, output->cpuData, output->GetBytes());
        }
    };

    void MultiCudaMLPOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight0 = *(datas.find("weight0")->second);
        Data &weight1 = *(datas.find("weight1")->second);

        AssertInFastLLM(weight0.dims.size() == 2 && weight1.dims.size() == 2, "MLP's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight0.dims[1], "MLP's weight's shape error.\n");
        AssertInFastLLM(weight0.dims[0] / 2 == weight1.dims[1], "MLP's weight's shape error.\n");
        AssertInFastLLM(weight0.dataType == weight1.dataType, "MLP's weight's data type error.\n");

        weight0.weightType = WeightType::LINEAR;
        weight1.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight1.dims[0];

        output.dataType = input.dataType;
        output.Resize(dims);
    }
    
    void DeviceGetInfos(int deviceId, std::string &specialId, int &mallocType) {
        static std::map <int, std::string> specialDeviceIds = {
            {99999, "cpu"}
        };
        specialId = "";
        if (specialDeviceIds.find(deviceId) != specialDeviceIds.end()) {
            specialId = specialDeviceIds[deviceId];
        }
        mallocType = 1;
        if (specialId == "cpu") {
            mallocType = 0;
        }
    }

    void MultiCudaMLPOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight0 = *(datas.find("weight0")->second);
        Data &bias0 = *(datas.find("bias0")->second);
        Data &weight1 = *(datas.find("weight1")->second);
        Data &bias1 = *(datas.find("bias1")->second);

        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);

        output.Allocate();
// auto st = std::chrono::system_clock::now();
        int mid = weight0.dims[0] / 2;
        int unit = weight0.groupCnt <= 0 ? 128 : weight0.groupCnt;
        if (weight0.dataType == fastllm::DataType::FP8_E4M3) {
            unit = weight0.blockM;
        }
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, false);
        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, mid, unit);

        FastllmInitNccl(devices);

        DivisionScheme divisionScheme, divisionSchemeO;
        for (int i = 0; i < devices.size(); i++) {
            int st = points[i], end = points[i + 1];
            int deviceId = devices[i];

            divisionScheme[deviceId].push_back(std::make_pair(st, end));
            divisionScheme[deviceId].push_back(std::make_pair(mid + st, mid + end));

            divisionSchemeO[deviceId].push_back(std::make_pair(st, end));
        }
        SplitMultiCudaWeight(weight0, bias0, devices, divisionScheme, 0);
        SplitMultiCudaWeight(weight1, bias1, devices, divisionSchemeO, 1);
        CopyToMultiDevices(w1, devices, false);
        CopyToMultiDevices(w2, devices, false);
        CopyToMultiDevices(w3, devices, false);

        EnsureReplicatedMultiCudaTensor(input, devices, true);
        output.dataDevice = input.dataDevice;
        EnsureReplicatedMultiCudaTensor(output, devices, false);
        FastllmCudaSetDevice(devices.empty() ? 0 : devices[0]);

        {
            auto *pool = fastllm::GetAlivePool();

            std::vector<fastllm::MultiThreadBaseOp*> ops;
            for (int i = 0; i < devices.size(); i++) {
                auto device = devices[i];
                std::string specialId = "";
                int mallocType;
                DeviceGetInfos(device, specialId, mallocType);

                if (specialId != "cpu") {
                    ops.push_back(new MultiCudaDoMergeMLPOp (
                        (uint8_t*)input.cudaData, (uint8_t*)input.cudaData, 
                        input.multiDeviceDatas[device], 
                        weight0.multiDeviceDatas[device], bias0.multiDeviceDatas[device], 
                        weight1.multiDeviceDatas[device], bias1.multiDeviceDatas[device], 
                        w1.multiDeviceDatas[device], w2.multiDeviceDatas[device], w3.multiDeviceDatas[device],
                        output.multiDeviceDatas[device], device));
                }
            }
            for (int i = 0; i < ops.size(); i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < ops.size(); i++) {
                pool->Wait(i);
                delete ops[i];
            }

            SyncReplicatedRootFromReplica(output, devices);
        }
/*
        uint8_t *partOutput = (uint8_t*)FastllmCudaMalloc(output.GetBytes() * devices.size());
        
        // Launch cuda op
        auto *pool = fastllm::GetAlivePool();

        std::vector<fastllm::MultiThreadBaseOp*> ops;
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            std::string specialId = "";
            int mallocType;
            DeviceGetInfos(device, specialId, mallocType);

            if (specialId != "cpu") {
                ops.push_back(new MultiCudaDoMergeMLPOp (
                    (uint8_t*)input.cudaData, (uint8_t*)cpuInput.data(), partOutput + output.GetBytes() * i,
                    input.multiDeviceDatas[device], 
                    weight0.multiDeviceDatas[device], bias0.multiDeviceDatas[device], 
                    weight1.multiDeviceDatas[device], bias1.multiDeviceDatas[device], 
                    w1.multiDeviceDatas[device], w2.multiDeviceDatas[device], w3.multiDeviceDatas[device],
                    curOutput.multiDeviceDatas[device], device));
            }
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }

        // run cpu op
        auto temp = pool->curActivateThreadInterval;
        pool->curActivateThreadInterval = std::make_pair(ops.size(), pool->threads.size());
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            std::string specialId = "";
            int mallocType;
            DeviceGetInfos(device, specialId, mallocType);

            if (specialId == "cpu") {
                MultiCudaCpuDoMergeMLPOp (
                    (uint8_t*)cpuInput.data(), partOutput + output.GetBytes() * i,
                    input.multiDeviceDatas[device], 
                    weight0.multiDeviceDatas[device], bias0.multiDeviceDatas[device], 
                    weight1.multiDeviceDatas[device], bias1.multiDeviceDatas[device], 
                    w1.multiDeviceDatas[device], w2.multiDeviceDatas[device], w3.multiDeviceDatas[device],
                    curOutput.multiDeviceDatas[device], device).Run();
            }
        }
        pool->curActivateThreadInterval = temp;

        // wait cuda op
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
// printf("calc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        FastllmReduce((uint8_t*)output.cudaData, partOutput, output.Count(0), devices.size(), output.dataType);
        FastllmCudaFree(partOutput);
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
*/
    }

    struct MultiCudaDoLinearOp : MultiThreadBaseOp {
        uint8_t *oriCudaInput, *oriCpuInput;
        Data *input, *weight, *bias;
        Data *output;
        int n, m, k, start, len;
        uint8_t *lastOutput;
        int deviceId;

        MultiCudaDoLinearOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput,
                            Data *input, Data *weight, Data *bias, Data *output, 
                            int n, int m, int k, int start, int len, uint8_t *lastOutput, int deviceId) : 
                oriCudaInput(oriCudaInput), oriCpuInput(oriCpuInput),
                input(input), weight(weight), bias(bias),
                output(output), 
                n(n), m(m), k(k), start(start), len(len), lastOutput(lastOutput),
                deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            AssertInFastLLM(input->cudaData != nullptr, "MultiCudaDoLinearOp: local input should be prepared.\n");
            DoCudaLinearReshape(*input, *weight, *output);
            if (deviceId == 0 && n == 1) {
                output->isFake = true;
                output->UpdateUnitSize();
                output->cudaData = lastOutput;
                output->expansionSize = output->Count(0);
                output->expansionBytes = (output->Count(0) * output->unitSize - 1) / output->unitSizeDiv + 1;
            }
            if (bias == nullptr) {
                DoCudaLinear(*input, *weight, *GetEmptyData(), *output);
            } else {
                DoCudaLinear(*input, *weight, *bias, *output);
            }

            if (deviceId != 0 || n > 1) {
                FastllmCudaMemcpy2DDeviceToDeviceAuto(lastOutput + start * output->unitSize, k * output->unitSize, output->cudaData, 
                    len * output->unitSize, len * output->unitSize, n, 0, deviceId);
            }
        }
    };

    struct MultiCudaDoLinearShardOp : MultiThreadBaseOp {
        Data *input, *weight, *bias, *output;
        int deviceId;

        MultiCudaDoLinearShardOp(Data *input, Data *weight, Data *bias, Data *output, int deviceId) :
                input(input), weight(weight), bias(bias), output(output), deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            DoCudaLinearReshape(*input, *weight, *output);
            if (bias == nullptr) {
                DoCudaLinear(*input, *weight, *GetEmptyData(), *output);
            } else {
                DoCudaLinear(*input, *weight, *bias, *output);
            }
        }
    };

    struct MultiCudaDoLinearReduceOp : MultiThreadBaseOp {
        Data *input, *weight, *bias, *output;
        int deviceId;
        bool doNcclReduce;

        MultiCudaDoLinearReduceOp(Data *input, Data *weight, Data *bias, Data *output, int deviceId, bool doNcclReduce) :
                input(input), weight(weight), bias(bias), output(output), deviceId(deviceId), doNcclReduce(doNcclReduce) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            DoCudaLinearReshape(*input, *weight, *output);
            if (bias == nullptr) {
                DoCudaLinear(*input, *weight, *GetEmptyData(), *output);
            } else {
                DoCudaLinear(*input, *weight, *bias, *output);
            }
            if (doNcclReduce) {
                FastllmNcclAllReduce(output->cudaData, output->cudaData, output->Count(0), output->dataType, deviceId);
            }
        }
    };

    static void ReduceReplicatedOutputOnRoot(Data &output, const std::vector <int> &devices) {
        if (devices.empty()) {
            return;
        }
        int rootDevice = devices[0];
        Data *root = output.multiDeviceDatas[rootDevice];
        if (root == nullptr || root->cudaData == nullptr) {
            return;
        }

        FastllmCudaSetDevice(rootDevice);
        Data temp(root->dataType, root->dims);
        temp.dataDevice = DataDevice::CUDA;
        temp.dataDeviceIds = {rootDevice};
        temp.Allocate();

        for (int i = 1; i < devices.size(); i++) {
            int device = devices[i];
            Data *part = output.multiDeviceDatas[device];
            if (part == nullptr || part->cudaData == nullptr) {
                continue;
            }
            FastllmCudaMemcpyBetweenDevices(rootDevice, temp.cudaData, device, part->cudaData, part->GetBytes());
            FastllmCudaAddTo(*root, temp, 1.0f);
        }

        for (int i = 1; i < devices.size(); i++) {
            int device = devices[i];
            Data *part = output.multiDeviceDatas[device];
            if (part == nullptr || part->cudaData == nullptr) {
                continue;
            }
            FastllmCudaMemcpyBetweenDevices(device, part->cudaData, rootDevice, root->cudaData, root->GetBytes());
        }
    }

    static void SyncReplicatedRootFromDevice0(Data &data, const std::vector <int> &devices) {
        if (devices.empty()) {
            return;
        }
        int rootDevice = devices[0];
        FastllmCudaSetDevice(rootDevice);
        data.ToDevice(DataDevice::CUDA, {rootDevice}, false);
        if (data.cudaData == nullptr && data.Count(0) > 0) {
            data.dataDevice = DataDevice::CUDA;
            data.dataDeviceIds = {rootDevice};
            data.expansionSize = 0;
            data.expansionBytes = 0;
            data.Allocate();
        }
        FastllmCudaCopyFromDeviceToDevice(data.cudaData, data.multiDeviceDatas[rootDevice]->cudaData, data.GetBytes());
    }

    static bool RunMultiCudaRowLinear(Data &input, Data &weight, Data &bias, Data &output) {
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1) {
            return false;
        }

        PrepareMultiCudaReplicatedData(input, devices, true);
        DivisionScheme divisionScheme = BuildMultiCudaRowSplitScheme(weight, devices, ratios);
        SplitMultiCudaWeight(weight, bias, devices, divisionScheme, 0);

        DoCudaLinearReshape(input, weight, output);
        PrepareMultiCudaShardedData(output, devices, output.dims, (int)output.dims.size() - 1, divisionScheme);

        auto *pool = fastllm::GetAlivePool();
        std::vector <fastllm::MultiThreadBaseOp*> ops;
        ops.reserve(devices.size());
        for (int device : devices) {
            ops.push_back(new MultiCudaDoLinearShardOp(
                input.multiDeviceDatas[device],
                weight.multiDeviceDatas[device],
                bias.multiDeviceDatas[device],
                output.multiDeviceDatas[device],
                device
            ));
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }

        output.tpLinearType = TP_LINEAR_NONE;
        return true;
    }

    static bool RunMultiCudaColumnLinear(Data &input, Data &weight, Data &bias, Data &output) {
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (devices.size() <= 1 || !input.IsTensorParallelSharded()) {
            return false;
        }
        bool useNccl = FastllmInitNccl(devices);
        if (const char *disableNccl = getenv("FASTLLM_DISABLE_NCCL")) {
            if (strcmp(disableNccl, "1") == 0 || strcasecmp(disableNccl, "true") == 0) {
                useNccl = false;
            }
        }

        DivisionScheme divisionScheme = input.tpRanges;
        SplitMultiCudaWeight(weight, bias, devices, divisionScheme, 1);

        DoCudaLinearReshape(input, weight, output);
        PrepareMultiCudaReplicatedData(output, devices, false);

        auto *pool = fastllm::GetAlivePool();
        std::vector <fastllm::MultiThreadBaseOp*> ops;
        ops.reserve(devices.size());
        for (int device : devices) {
            ops.push_back(new MultiCudaDoLinearReduceOp(
                input.multiDeviceDatas[device],
                weight.multiDeviceDatas[device],
                bias.multiDeviceDatas[device],
                output.multiDeviceDatas[device],
                device,
                useNccl
            ));
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }

        if (!useNccl) {
            ReduceReplicatedOutputOnRoot(output, devices);
        }
        // SyncReplicatedRootFromDevice0(output, devices);
        return true;
    }

    bool MultiCudaLinearRow(Data &input, Data &weight, Data &bias, Data &output) {
        return RunMultiCudaRowLinear(input, weight, bias, output);
    }

    bool MultiCudaLinearColumn(Data &input, Data &weight, Data &bias, Data &output) {
        return RunMultiCudaColumnLinear(input, weight, bias, output);
    }

    bool MultiCudaLinearOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        if (weight.tpLinearType != TP_LINEAR_NONE || input.IsTensorParallelSharded() ||
            IsTensorParallelRowWeight(weight) || IsTensorParallelColumnWeight(weight)) {
            return true;
        }
        return weight.dims[0] > 10000 || weight.dims[1] > 10000;
    }

    void MultiCudaLinearOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
/* printf("into multi linear\n");
auto st = std::chrono::system_clock::now();
{
    int n = input.Count(0) / input.dims.back();
    int m = input.dims.back();
    int k = output.dims.back();
    printf("n = %d, m = %d, k = %d\n", n, m, k);
} */
        if (weight.tpLinearType == TP_LINEAR_ROW || (!input.IsTensorParallelSharded() && IsTensorParallelRowWeight(weight))) {
            if (RunMultiCudaRowLinear(input, weight, bias, output)) {
// printf("row spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                return;
            }
        }
        if (weight.tpLinearType == TP_LINEAR_COLUMN || input.IsTensorParallelSharded() || IsTensorParallelColumnWeight(weight)) {
            if (RunMultiCudaColumnLinear(input, weight, bias, output)) {
// printf("column spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                return;
            }
        }

        output.Allocate();
// auto st = std::chrono::system_clock::now();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        int unit = weight.groupCnt <= 0 ? 128 : weight.groupCnt;
        if (weight.dataType == fastllm::DataType::FP8_E4M3) {
            unit = weight.blockM;
        }
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, weight.dims[0], unit);

        DivisionScheme divisionScheme;
        for (int i = 0; i < devices.size(); i++) {
            int st = points[i], end = points[i + 1];
            int deviceId = devices[i];
            divisionScheme[deviceId].push_back(std::make_pair(st, end));
        }
        SplitMultiCudaWeight(weight, bias, devices, divisionScheme, 0);
        Data curOutput;
        EnsureReplicatedMultiCudaTensor(input, devices, true);
        curOutput.dataDevice = input.dataDevice;
        CopyToMultiDevices(curOutput, devices, false);
        auto *pool = fastllm::GetAlivePool();
        std::vector<fastllm::MultiThreadBaseOp*> ops;
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            int start = points[i], len = points[i + 1] - points[i];
            ops.push_back(new MultiCudaDoLinearOp (
                (uint8_t*)input.cudaData, nullptr,
                input.multiDeviceDatas[device], 
                weight.multiDeviceDatas[device], bias.multiDeviceDatas[device], 
                curOutput.multiDeviceDatas[device], 
                n, m, k, start, len, (uint8_t*)output.cudaData, device));
        }
        for (int i = 0; i < devices.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        // ops[0]->Run();
        for (int i = 0; i < devices.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
// float spend = GetSpan(st, std::chrono::system_clock::now());
// float gops = (float)n * m * k / spend / 1e9;
// printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }
    
    void MultiCudaMergeAttention::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight1 = *(datas.find("weight1")->second);
        Data &output = *(datas.find("output")->second);
        std::vector <int> dims = input.dims;
        dims.back() = weight1.dims[0];
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    struct MultiCudaDoMergeAttentionOp : MultiThreadBaseOp {
        uint8_t *oriCudaInput;
        Data *input, *weight0, *bias0, *weight1, *bias1;
        Data *qNorm, *kNorm;
        Data *qkv, *q, *k, *v;
        int doQKNorm;
        int qNum, kvNum, headDim, rotDim;
        float attentionScale, eps;
        Data *positionIds, *sinData, *cosData;
        Data **keys, **values, **masks;
        Data *output;
        int batch;
        int deviceId;

        MultiCudaDoMergeAttentionOp(uint8_t *oriCudaInput,
                            Data *input, Data *weight0, Data *bias0, Data *weight1, Data *bias1, 
                            bool doQKNorm, Data *qNorm, Data *kNorm, float eps,
                            Data *qkv, Data *q, Data *k, Data *v,
                            int qNum, int kvNum, int headDim, int rotDim, float attentionScale,
                            Data *positionIds, Data *sinData, Data *cosData,
                            Data** keys, Data** values, Data** masks, 
                            Data *output, int batch, int deviceId) : 
                oriCudaInput(oriCudaInput),
                input(input), weight0(weight0), bias0(bias0), weight1(weight1), bias1(bias1), 
                doQKNorm(doQKNorm), qNorm(qNorm), kNorm(kNorm), eps(eps),
                qkv(qkv), q(q), k(k), v(v), 
                qNum(qNum), kvNum(kvNum), headDim(headDim), rotDim(rotDim), attentionScale(attentionScale),
                positionIds(positionIds), sinData(sinData), cosData(cosData),
                keys(keys), values(values), masks(masks), 
                output(output), batch(batch), deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            AssertInFastLLM(input->cudaData != nullptr,
                            "MultiCudaDoMergeAttentionOp: local input should be prepared.\n");

            if (batch > 1) {
                int bsz = batch, seqlen = input->dims[1];
                std::vector <Data*> vKeys, vValues, vMasks;
                std::vector <Data> curKs, curVs, curQs;
                Data curAttenOutput;
                curKs.resize(bsz);
                curVs.resize(bsz);
                curQs.resize(bsz);
                std::vector <Data*> pointersK, pointersV, pointersQ;
                pointersK.resize(bsz);
                pointersV.resize(bsz);
                pointersQ.resize(bsz);
                std::vector <Data*> qs, attns, contexts;
                qs.resize(bsz);
                attns.resize(bsz);
                contexts.resize(bsz);
                std::vector <Data> curContextLayer;
                curContextLayer.resize(bsz);

                vKeys.resize(bsz);
                vValues.resize(bsz);
                vMasks.resize(bsz);
                for (int i = 0; i < bsz; i++) {
                    vKeys[i] = keys[i];
                    vValues[i] = values[i];
                    vMasks[i] = masks[i];
                }
                DoCudaLinearReshape(*input, *weight0, *qkv);
                if (bias0 == nullptr) {
                    DoCudaLinear(*input, *weight0, *GetEmptyData(), *qkv);
                } else {
                    DoCudaLinear(*input, *weight0, *bias0, *qkv);
                }

                int per = qkv->dims.back() / (qNum / kvNum + 2);
                int qdim = per * (qNum / kvNum);
                DoCudaSplitReshape(*qkv, -1, 0, qdim, *q);
                DoCudaSplitReshape(*qkv, -1, qdim, qdim + per, *k);
                DoCudaSplitReshape(*qkv, -1, qdim + per, qdim + per * 2, *v);
                DoCudaSplit(*qkv, -1, 0, qdim, *q);
                DoCudaSplit(*qkv, -1, qdim, qdim + per, *k);
                DoCudaSplit(*qkv, -1, qdim + per, qdim + per * 2, *v);

                std::vector <int> qkvSize = {1, seqlen, -1, headDim};
                q->Reshape(qkvSize);
                k->Reshape(qkvSize);
                v->Reshape(qkvSize);

                if (doQKNorm) {
                    RMSNorm(*q, *qNorm, eps, *q);
                    RMSNorm(*k, *kNorm, eps, *k);                    
                }

                FastllmCudaLlamaRotatePosition2D(*q, *positionIds, *sinData, *cosData, rotDim);
                FastllmCudaLlamaRotatePosition2D(*k, *positionIds, *sinData, *cosData, rotDim);

                int total = 0;
                q->Reshape({-1, q->dims[2], q->dims[3]});
                k->Reshape({-1, k->dims[2], k->dims[3]});
                v->Reshape({-1, v->dims[2], v->dims[3]});

                std::vector <int> qdims = {q->dims[1], 1, q->dims[2]};
                std::vector <uint64_t> qstrides = {(uint64_t)q->dims[2], (uint64_t)q->dims[2], 1};
                std::vector <int> kdims = {k->dims[1], 1, k->dims[2]};
                std::vector <uint64_t> kstrides = {(uint64_t)k->dims[2], (uint64_t)k->dims[2], 1};
                std::vector <int> vdims = {v->dims[1], 1, v->dims[2]};
                std::vector <uint64_t> vstrides = {(uint64_t)v->dims[2], (uint64_t)v->dims[2], 1};
                for (int b = 0; b < bsz; b++) {
                    curQs[b].dims = qdims;
                    curQs[b].strides = qstrides;
                    curQs[b].FakeFrom(*q, b * q->strides[0] * q->unitSize);
                    curKs[b].dims = kdims;
                    curKs[b].strides = kstrides;
                    curKs[b].FakeFrom(*k, b * k->strides[0] * k->unitSize);
                    curVs[b].dims = vdims;
                    curVs[b].strides = vstrides;
                    curVs[b].FakeFrom(*v, b * v->strides[0] * v->unitSize);
                }

                for (int b = 0; b < bsz; b++) {
                    pointersK[b] = (&curKs[b]);
                    pointersV[b] = (&curVs[b]);
                }

                DoCudaCatDirectBatch(vKeys.data(), pointersK.data(), bsz, 1);
                DoCudaCatDirectBatch(vValues.data(), pointersV.data(), bsz, 1);

                int embed_dim = weight1->dims[1];
                Data &attenOutput = *qkv;
                attenOutput.ToDevice(q->dataDevice);
                attenOutput.Resize({1, bsz, embed_dim});
                attenOutput.Allocate();
                for (int b = 0; b < bsz; b++) {
                    qs[b] = (&curQs[b]);
                    curContextLayer[b].FakeFrom(attenOutput, b * embed_dim * attenOutput.unitSize);
                    contexts[b] = (&curContextLayer[b]);
                }

                DoCudaAttentionBatchReshape(qs.data(), vValues.data(), contexts.data(), bsz);
                DoCudaAttentionBatch(qs.data(), vKeys.data(), vValues.data(), vMasks.data(), contexts.data(), 
                                qs[0]->dims[0] / values[0]->dims[0], attentionScale, bsz);                

                DoCudaLinearReshape(*qkv, *weight1, *output);
                /* if (deviceId == 0) {
                    output->isFake = true;
                    output->UpdateUnitSize();
                    output->cudaData = partOutput;
                    output->expansionSize = output->Count(0);
                    output->expansionBytes = (output->Count(0) * output->unitSize - 1) / output->unitSizeDiv + 1;
                } */
                if (bias1 == nullptr) {
                    DoCudaLinear(*qkv, *weight1, *GetEmptyData(), *output);
                } else {
                    DoCudaLinear(*qkv, *weight1, *bias1, *output);
                }
                if (deviceId != 0) {
                    // FastllmCudaCopyFromDeviceToDevice(partOutput, output->cudaData, output->GetBytes());
                }

                FastllmNcclAllReduce(output->cudaData, output->cudaData, output->Count(0), output->dataType, deviceId);
            } else {
                int bsz = input->dims[0], seqlen = input->dims[1];

                DoCudaLinearReshape(*input, *weight0, *qkv);
                if (bias0 == nullptr) {
                    DoCudaLinear(*input, *weight0, *GetEmptyData(), *qkv);
                } else {
                    DoCudaLinear(*input, *weight0, *bias0, *qkv);
                }
                int per = qkv->dims.back() / (qNum / kvNum + 2);
                int qdim = per * (qNum / kvNum);
                DoCudaSplitReshape(*qkv, -1, 0, qdim, *q);
                DoCudaSplitReshape(*qkv, -1, qdim, qdim + per, *k);
                DoCudaSplitReshape(*qkv, -1, qdim + per, qdim + per * 2, *v);
                DoCudaSplit(*qkv, -1, 0, qdim, *q);
                DoCudaSplit(*qkv, -1, qdim, qdim + per, *k);
                DoCudaSplit(*qkv, -1, qdim + per, qdim + per * 2, *v);

                std::vector <int> qkvSize = {bsz, seqlen, -1, headDim};
                q->Reshape(qkvSize);
                k->Reshape(qkvSize);
                v->Reshape(qkvSize);
                if (doQKNorm) {
                    RMSNorm(*q, *qNorm, eps, *q);
                    RMSNorm(*k, *kNorm, eps, *k);                    
                }
                
                FastllmCudaLlamaRotatePosition2D(*q, *positionIds, *sinData, *cosData, rotDim);
                FastllmCudaLlamaRotatePosition2D(*k, *positionIds, *sinData, *cosData, rotDim);

                DoCudaPermuteSelf(*q, {0, 2, 1, 3});
                DoCudaPermuteSelf(*k, {0, 2, 1, 3});
                DoCudaPermuteSelf(*v, {0, 2, 1, 3});

                qkvSize = {-1, seqlen, headDim};
                q->Reshape(qkvSize);
                k->Reshape(qkvSize);
                v->Reshape(qkvSize);

                Data &pastKey = *keys[0];
                Data &pastValue = *values[0];

                DoCudaCatDirect(pastKey, *k, 1);
                DoCudaCatDirect(pastValue, *v, 1);
                DoCudaAttentionReshape(*q, pastValue, *qkv);
                DoCudaAttention(*q, pastKey, pastValue, *masks[0], *qkv, q->dims[0] / pastKey.dims[0], attentionScale, 1);
                DoCudaPermuteSelf(*qkv, {1, 0, 2});
                qkv->Reshape({seqlen, bsz, -1});
                DoCudaPermuteSelf(*qkv, {1, 0, 2});
                DoCudaLinearReshape(*qkv, *weight1, *output);

                /* if (deviceId == 0) {
                    output->isFake = true;
                    output->UpdateUnitSize();
                    output->cudaData = partOutput;
                    output->expansionSize = output->Count(0);
                    output->expansionBytes = (output->Count(0) * output->unitSize - 1) / output->unitSizeDiv + 1;
                } */
                if (bias1 == nullptr) {
                    DoCudaLinear(*qkv, *weight1, *GetEmptyData(), *output);
                } else {
                    DoCudaLinear(*qkv, *weight1, *bias1, *output);
                }
                
                if (deviceId != 0) {
                    // FastllmCudaCopyFromDeviceToDevice(partOutput, output->cudaData, output->GetBytes());
                }

                FastllmNcclAllReduce(output->cudaData, output->cudaData, output->Count(0), output->dataType, deviceId);
            }
        }
    };

    void MultiCudaMergeAttention::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight0 = *(datas.find("weight0")->second);
        Data &bias0 = *(datas.find("bias0")->second);
        Data &weight1 = *(datas.find("weight1")->second);
        Data &bias1 = *(datas.find("bias1")->second);
        Data &qNorm = *(datas.find("qNorm")->second);
        Data &kNorm = *(datas.find("kNorm")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sinData")->second);
        Data &cosData = *(datas.find("cosData")->second);
        Data &output = *(datas.find("output")->second);
        Data &qkv = *(datas.find("qkv")->second);
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        int qNum = intParams.find("qNum")->second;
        int kvNum = intParams.find("kvNum")->second;
        int headDim = intParams.find("headDim")->second;
        int rotDim = intParams.find("rotDim")->second;
        int doQKNorm = intParams.find("doQKNorm")->second;
        float attentionScale = floatParams.find("attentionScale")->second;
        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5;
        Data **keys = (Data**)(datas.find("keys")->second);
        Data **values = (Data**)(datas.find("values")->second);
        Data **masks = (Data**)(datas.find("masks")->second);
        int batch = intParams.find("keys___batch")->second;
        output.Allocate();
// auto st = std::chrono::system_clock::now();
        int group = qNum / kvNum;
        int vDim = weight1.dims[1] / qNum;
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, kvNum, 1);

        FastllmInitNccl(devices);
        
        DivisionScheme divisionScheme, divisionSchemeO;
        for (int i = 0; i < devices.size(); i++) {
            int st = points[i], end = points[i + 1];
            int deviceId = devices[i];
            int qgap = qNum * headDim, qkgap = (qNum + kvNum) * headDim;
            divisionScheme[deviceId].push_back(std::make_pair(st * group * headDim, end * group * headDim));
            divisionScheme[deviceId].push_back(std::make_pair(qgap + st * headDim, qgap + end * headDim));
            divisionScheme[deviceId].push_back(std::make_pair(qkgap + st * headDim, qkgap + end * headDim));

            divisionSchemeO[deviceId].push_back(std::make_pair(st * group * vDim, end * group * vDim));
        }
        SplitMultiCudaWeight(weight0, bias0, devices, divisionScheme, 0);
        SplitMultiCudaWeight(weight1, bias1, devices, divisionSchemeO, 1);
        if (doQKNorm) {
            EnsureReplicatedMultiCudaTensor(qNorm, devices, true);
            EnsureReplicatedMultiCudaTensor(kNorm, devices, true);
        }

        EnsureReplicatedMultiCudaTensor(qkv, devices, false);
        EnsureReplicatedMultiCudaTensor(q, devices, false);
        EnsureReplicatedMultiCudaTensor(k, devices, false);
        EnsureReplicatedMultiCudaTensor(v, devices, false);
        EnsureReplicatedMultiCudaTensor(positionIds, devices, true);
        EnsureReplicatedMultiCudaTensor(sinData, devices, true);
        EnsureReplicatedMultiCudaTensor(cosData, devices, true);
        for (int i = 0; i < batch; i++) {
            CopyToMultiDevices(*keys[i], devices, true);
            CopyToMultiDevices(*values[i], devices, true);
            if (masks[i] != nullptr) {
                RefreshReplicatedMultiCudaTensor(*masks[i], devices);
            }
        }
        std::map <int, std::vector <Data*> > curKeys, curValues, curMasks;
        for (int device : devices) {
            for (int i = 0; i < batch; i++) {
                curKeys[device].push_back(keys[i]->multiDeviceDatas[device]);
                curValues[device].push_back(values[i]->multiDeviceDatas[device]);
                curMasks[device].push_back(masks[i] == nullptr ? nullptr : masks[i]->multiDeviceDatas[device]);
            }
        }

        EnsureReplicatedMultiCudaTensor(input, devices, true);
        output.dataDevice = input.dataDevice;
        EnsureReplicatedMultiCudaTensor(output, devices, false);
        FastllmCudaSetDevice(devices.empty() ? 0 : devices[0]);
        auto *pool = fastllm::GetAlivePool();
        std::vector<fastllm::MultiThreadBaseOp*> ops;

        for (int i = 0; i < devices.size(); i++) {
            int device = devices[i];
            FastllmCudaSetDevice(device);
            int bsz = batch, seqlen = input.dims[1];
            if (bsz > 1) {
                seqlen = 1;
            }
            
            int unitLen = 128;
            for (int i = 0; i < bsz; i++) {
                Data &pastKey = *keys[i]->multiDeviceDatas[device];
                Data &pastValue = *values[i]->multiDeviceDatas[device];
                while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || seqlen > pastKey.expansionDims[1]))
                    || (pastKey.dims.size() > 0 && pastKey.dims[1] + seqlen > pastKey.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector <int> {kvNum, ((seqlen - 1) / unitLen + 1) * unitLen, headDim};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((seqlen - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || seqlen > pastValue.expansionDims[1]))
                    || (pastValue.dims.size() > 0 && pastValue.dims[1] + seqlen > pastValue.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector <int> {kvNum, ((seqlen - 1) / unitLen + 1) * unitLen, headDim};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((seqlen - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
            }
        }
        FastllmCudaSetDevice(devices.empty() ? 0 : devices[0]);
        
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            ops.push_back(new MultiCudaDoMergeAttentionOp (
                (uint8_t*)input.cudaData,
                input.multiDeviceDatas[device], 
                weight0.multiDeviceDatas[device], bias0.multiDeviceDatas[device], 
                weight1.multiDeviceDatas[device], bias1.multiDeviceDatas[device], 
                doQKNorm, qNorm.multiDeviceDatas[device], kNorm.multiDeviceDatas[device], eps,
                qkv.multiDeviceDatas[device], q.multiDeviceDatas[device], k.multiDeviceDatas[device], v.multiDeviceDatas[device], 
                qNum, kvNum, headDim, rotDim, attentionScale, 
                positionIds.multiDeviceDatas[device], sinData.multiDeviceDatas[device], cosData.multiDeviceDatas[device], 
                curKeys[device].data(), curValues[device].data(), curMasks[device].data(), 
                output.multiDeviceDatas[device], batch, device));
        }
        for (int i = 0; i < devices.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        // ops[0]->Run();
        for (int i = 0; i < devices.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
        SyncReplicatedRootFromReplica(output, devices);
        for (int i = 0; i < batch; i++) {
            keys[i]->dims = keys[i]->multiDeviceDatas[devices[0]]->dims;
            keys[i]->expansionDims = keys[i]->multiDeviceDatas[devices[0]]->expansionDims;
            values[i]->dims = values[i]->multiDeviceDatas[devices[0]]->dims;
            values[i]->expansionDims = values[i]->multiDeviceDatas[devices[0]]->expansionDims;
        }
    }

    struct MultiCudaDoMergeMOEOp : MultiThreadBaseOp {
        uint8_t *oriCudaInput, *oriCpuInput, *partOutput;
        Data *input;
        Data **weights;
        Data *index, *score;
        Data *w1, *w2, *w3;
        int wBatch;
        float sharedScale;
        Data *output;
        int deviceId;

        MultiCudaDoMergeMOEOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput, uint8_t *partOutput, 
                Data *input, Data **weights, Data *index, Data *score, 
                Data *w1, Data *w2, Data *w3, 
                int wBatch, float sharedScale,
                Data *output, int deviceId) : 
                oriCudaInput(oriCudaInput), oriCpuInput(oriCpuInput), partOutput(partOutput),
                input(input), weights(weights), index(index), score(score), 
                w1(w1), w2(w2), w3(w3),
                wBatch(wBatch), sharedScale(sharedScale),
                output(output), deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            if (deviceId == 0) {
                input->cudaData = oriCudaInput;
            } else {
                input->Allocate();
                FastllmCudaCopyFromHostToDevice(input->cudaData, oriCpuInput, input->GetBytes());
            }            
            if (deviceId == 0) {
                output->isFake = true;
                output->UpdateUnitSize();
                output->cudaData = partOutput;
                output->expansionSize = output->Count(0);
                output->expansionBytes = (output->Count(0) * output->unitSize - 1) / output->unitSizeDiv + 1;
            }
            
            std::vector <Data*> curWeights;
            curWeights.resize(wBatch);
            for (int i = 0; i < wBatch; i++) {
                if (weights[i] == nullptr) {
                    curWeights[i] = nullptr;
                } else {
                    curWeights[i] = weights[i]->multiDeviceDatas[deviceId];
                }
            }

            output->Resize(input->dims);
            DoCudaMergeMOE(*input, *output, *index, *score, *w1, *w2, *w3, curWeights.data(), nullptr, sharedScale);

            if (deviceId != 0) {
                FastllmCudaCopyFromDeviceToDevice(partOutput, output->cudaData, output->GetBytes());
            }
        }
    };

    struct MultiCudaCpuDoMergeMOEOp : MultiThreadBaseOp {
        uint8_t *oriCpuInput, *partOutput;
        Data *input;
        Data **weights;
        Data *index, *score;
        Data *w1, *w2, *w3;
        int wBatch;
        float sharedScale;
        Data *output;
        int deviceId;

        MultiCudaCpuDoMergeMOEOp(uint8_t *oriCpuInput, uint8_t *partOutput, 
                Data *input, Data **weights, Data *index, Data *score, 
                Data *w1, Data *w2, Data *w3, 
                int wBatch, float sharedScale,
                Data *output, int deviceId) : 
                oriCpuInput(oriCpuInput), partOutput(partOutput),
                input(input), weights(weights), index(index), score(score), 
                w1(w1), w2(w2), w3(w3),
                wBatch(wBatch), sharedScale(sharedScale),
                output(output), deviceId(deviceId) {}

        void Run() {
            // 注意weights里面的值，真正要使用的是weights[x]->multiDeviceDatas[deviceId]
            input->Allocate();
            memcpy(input->cpuData, oriCpuInput, input->GetBytes());

            int batch = input->dims[0];
            int n = index->dims[0];
            int topk = index->dims[1];
            
            index->ToDevice(DataDevice::CPU);
            score->ToDevice(DataDevice::CPU);
            ToDataType(*index, DataType::INT32PARAM);
            ToDataType(*score, DataType::FLOAT32);
            int32_t *indexData = (int32_t*)index->cpuData;
            float *scoreData = (float*)score->cpuData;

            if (batch == 1) {
                std::vector <std::pair <int, float> > v;
                v.resize(topk + 1);
                for (int j = 0; j < topk; j++) {
                    int expertIdx = indexData[j];
                    float expertScore = scoreData[j];
                    v[j] = std::make_pair(expertIdx + 1, expertScore);
                }
                v.back() = (std::make_pair(0, sharedScale));
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    float value = v[j].second;
                    if (weights[idx * 2] == nullptr) {
                        continue;
                    }

                    DoCpuLinearReshape(*input, *weights[idx * 2]->multiDeviceDatas[deviceId], *w3);
                    DoCpuLinear(*input, *weights[idx * 2]->multiDeviceDatas[deviceId], Data(), *w3);
                    
                    DoCpuSwigluReshape(*w3, *w1);
                    DoCpuSwiglu(*w3, *w1);

                    DoCpuLinearReshape(*w1, *weights[idx * 2 + 1]->multiDeviceDatas[deviceId], *w2);
                    DoCpuLinear(*w1, *weights[idx * 2 + 1]->multiDeviceDatas[deviceId], Data(), *w2);

                    if (j == 0) {
                        output->Resize(w2->dims);
                        output->Allocate();
                        for (int i = 0; i < output->Count(0); i++) {
                            ((float*)output->cpuData)[i] = ((float*)w2->cpuData)[i] * value;
                        }
                    } else {
                        for (int i = 0; i < output->Count(0); i++) {
                            ((float*)output->cpuData)[i] += ((float*)w2->cpuData)[i] * value;
                        }
                    }
                }
            } else {
                Data attenPart, moePart;
                Data moeFinal = Data();
                moeFinal.Resize({0, input->dims[1]});
                moeFinal.Expansion(input->dims);
                attenPart.ToDevice(input->dataDevice);
                moePart.ToDevice(input->dataDevice);
                moeFinal.ToDevice(input->dataDevice);
                
                for (int b = 0; b < batch; b++) {
                    Data *currentData = input;
                    if (batch != 1) {
                        attenPart.Resize({1, input->dims.back()});
                        attenPart.dataType = input->dataType;
                        attenPart.Allocate();
                        memcpy (
                            attenPart.cpuData,
                            input->cpuData + b * input->dims.back() * input->unitSize,
                            attenPart.GetBytes()
                        );

                        currentData = &attenPart;
                    }
                        
                    moePart.Resize(currentData->dims);
                    moePart.Allocate(0.0f);

                    std::vector <std::pair <int, float> > v;
                    for (int j = 0; j < topk; j++) {
                        int expertIdx = indexData[b * topk + j];
                        float expertScore = scoreData[b * topk + j];
                        v.push_back(std::make_pair(expertIdx + 1, expertScore));
                    }
                    v.push_back(std::make_pair(0, sharedScale));

                    for (int j = 0; j < v.size(); j++) {
                        int idx = v[j].first;
                        float value = v[j].second;
                        if (weights[idx * 2] == nullptr) {
                            continue;
                        }

                        DoCpuLinearReshape(*currentData, *weights[idx * 2]->multiDeviceDatas[deviceId], *w3);
                        DoCpuLinear(*currentData, *weights[idx * 2]->multiDeviceDatas[deviceId], Data(), *w3);
                        
                        DoCpuSwigluReshape(*w3, *w1);
                        DoCpuSwiglu(*w3, *w1);

                        DoCpuLinearReshape(*w1, *weights[idx * 2 + 1]->multiDeviceDatas[deviceId], *w2);
                        DoCpuLinear(*w1, *weights[idx * 2 + 1]->multiDeviceDatas[deviceId], Data(), *w2);

                        for (int i = 0; i < moePart.Count(0); i++) {
                            ((float*)moePart.cpuData)[i] += ((float*)w2->cpuData)[i] * value;
                        }
                    }

                    DoCpuCatDirect(moeFinal, moePart, 0);
                    moeFinal.expansionDims.clear();

                    output->Resize(moeFinal.dims);
                    output->Allocate();
                    memcpy(output->cpuData, moeFinal.cpuData, moeFinal.GetBytes());
                }
            }

            FastllmCudaCopyFromHostToDevice(partOutput, output->cpuData, output->GetBytes());
        }
    };

    void MultiCudaMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &index = *(datas.find("index")->second);
        Data &score = *(datas.find("score")->second);
        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);
        Data **weights = (Data**)(datas.find("weights")->second);
        Data **biass = (Data**)(datas.find("biass")->second);
        float sharedScale = floatParams.find("sharedScale") != floatParams.end() ? floatParams.find("sharedScale")->second : 1.0f;
        
        int n = index.dims[0];
        int topk = index.dims[1];        
        output.Allocate();
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, false);
        int wBatch = intParams.find("weights___batch") != intParams.end() ? intParams.find("weights___batch")->second : (topk + 1) * 2;
        if (!weights[2]->multiDeviceData) {
            // 这里需要保证已经warmup过了，如果weights[2]切好就代表所有weight都已经切好了
            Data empty;
            for (int i = 0; i < wBatch; i += 2) {
                if (weights[i] == nullptr) {
                    continue;
                }
                int k = weights[i]->dims[0];
                std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, k / 2, weights[i]->groupCnt <= 0 ? 128 : weights[i]->groupCnt);
                DivisionScheme divisionScheme, divisionSchemeO;
                int mid = weights[i]->dims[0] / 2;
                for (int i = 0; i < devices.size(); i++) {
                    int st = points[i], end = points[i + 1];
                    int deviceId = devices[i];
                    divisionScheme[deviceId].push_back(std::make_pair(st, end));

                    divisionSchemeO[deviceId].push_back(std::make_pair(st, end));
                    divisionSchemeO[deviceId].push_back(std::make_pair(mid + st, mid + end));
                }
                SplitMultiCudaWeight(*weights[i], empty, devices, divisionSchemeO, 0);
                SplitMultiCudaWeight(*weights[i + 1], empty, devices, divisionScheme, 1);
            }
        }

        index.ToDevice(DataDevice::CPU);
        score.ToDevice(DataDevice::CPU);
        ToDataType(index, DataType::INT32);
        ToDataType(score, DataType::FLOAT32);
        
        CopyToMultiDevices(w1, devices, false);
        CopyToMultiDevices(w2, devices, false);
        CopyToMultiDevices(w3, devices, false);

        Data &curInput = *(datas.find("curInput")->second);
        Data &curOutput = *(datas.find("curOutput")->second);

        CopyToMultiDevices(input, devices, false);
        curOutput.dataDevice = input.dataDevice;
        CopyToMultiDevices(curOutput, devices, false);
        std::vector <uint8_t> cpuInput;
        cpuInput.resize(input.GetBytes());
        FastllmCudaSetDevice(0);
        FastllmCudaCopyFromDeviceToHost(cpuInput.data(), input.cudaData, input.GetBytes());
        uint8_t *partOutput = (uint8_t*)FastllmCudaMalloc(output.GetBytes() * devices.size());

        auto *pool = fastllm::GetAlivePool();
        std::vector<fastllm::MultiThreadBaseOp*> ops;
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            std::string specialId = "";
            int mallocType;
            DeviceGetInfos(device, specialId, mallocType);

            if (specialId != "cpu") {
                ops.push_back(new MultiCudaDoMergeMOEOp(
                    (uint8_t*)input.cudaData, (uint8_t*)cpuInput.data(), partOutput + output.GetBytes() * i,
                    input.multiDeviceDatas[device], weights, &index, &score, 
                    w1.multiDeviceDatas[device], w2.multiDeviceDatas[device], w3.multiDeviceDatas[device], 
                    wBatch, sharedScale, 
                    curOutput.multiDeviceDatas[device], device
                ));
            }
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }

        // run cpu op
        auto temp = pool->curActivateThreadInterval;
        pool->curActivateThreadInterval = std::make_pair(ops.size(), pool->threads.size());
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            std::string specialId = "";
            int mallocType;
            DeviceGetInfos(device, specialId, mallocType);
            if (specialId == "cpu") {
                MultiCudaCpuDoMergeMOEOp (
                    (uint8_t*)cpuInput.data(), partOutput + output.GetBytes() * i,
                    input.multiDeviceDatas[device], weights, &index, &score, 
                    w1.multiDeviceDatas[device], w2.multiDeviceDatas[device], w3.multiDeviceDatas[device], 
                    wBatch, sharedScale, 
                    curOutput.multiDeviceDatas[device], device).Run();
            }
        }
        pool->curActivateThreadInterval = temp;

        // wait cuda op
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
        FastllmReduce((uint8_t*)output.cudaData, partOutput, output.Count(0), devices.size(), output.dataType);
        FastllmCudaFree(partOutput);
    }
}
