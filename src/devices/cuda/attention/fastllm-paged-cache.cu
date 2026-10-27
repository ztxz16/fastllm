#include "fastllm-attention-common.cuh"

template <typename Launch>
static bool FastllmDispatchPagedCacheCopyTypes(
        fastllm::DataType srcType, fastllm::DataType dstType, Launch launch) {
    auto dispatchDst = [&](auto *src) {
        switch (dstType) {
            case fastllm::DataType::FLOAT32: launch(src, (float*)nullptr); break;
            case fastllm::DataType::FLOAT16: launch(src, (half*)nullptr); break;
            case fastllm::DataType::BFLOAT16: launch(src, (__nv_bfloat16*)nullptr); break;
            case fastllm::DataType::FP8_E4M3: launch(src, (__nv_fp8_e4m3*)nullptr); break;
            case fastllm::DataType::FP4_E2M1: launch(src, (uint8_t*)nullptr); break;
            default: return false;
        }
        return true;
    };
    switch (srcType) {
        case fastllm::DataType::FLOAT32: return dispatchDst((float*)nullptr);
        case fastllm::DataType::FLOAT16: return dispatchDst((half*)nullptr);
        case fastllm::DataType::BFLOAT16: return dispatchDst((__nv_bfloat16*)nullptr);
        default: return false;
    }
}

// input: [numHeads, seqLen, headDim], pagedData: [maxPages, pageLen, numHeads, headDim]
template <typename SrcT, typename DstT, int THREAD_PER_BLOCK>
__global__ void FastllmPagedCacheCopyKernel(
    uint8_t *pagedData,      // dst: [maxPages, pageLen, numHeads, headDim]
    int pageIdx,             // target page index
    int pageLen,             // page length
    int numHeads,            // number of heads
    int headDim,             // head dimension
    uint8_t *inputData,      // src: [numHeads, seqLen, headDim]
    int seqLen,              // input sequence length
    int inputOffset,         // offset in input sequence
    int copyLen,             // number of tokens to copy
    int pageOffset           // offset in target page (where to start writing)
) {
    // Calculate the linear index for this thread
    // Each thread handles one element: (head, token, dim)
    int totalElements = numHeads * copyLen * headDim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalElements) {
        return;
    }

    // Decompose linear index into (head, token, dim)
    int head = idx / (copyLen * headDim);
    int remainder = idx % (copyLen * headDim);
    int token = remainder / headDim;
    int dim = remainder % headDim;

    // Calculate source address: input[head, inputOffset + token, dim]
    // input layout: [numHeads, seqLen, headDim]
    // For input[head, token_idx, dim] where token_idx = inputOffset + token:
    // srcOffset = head * (seqLen * headDim * unitSize) + token_idx * (headDim * unitSize) + dim * unitSize
    const SrcT *src = (const SrcT*)inputData;
    DstT *dst = (DstT*)pagedData;
    int srcOffset = head * seqLen * headDim + (inputOffset + token) * headDim + dim;

    // Calculate destination address: pagedData[pageIdx, pageOffset + token, head, dim]
    // pagedData layout: [maxPages, pageLen, numHeads, headDim]
    int pageStride = pageLen * numHeads * headDim;
    int tokenStride = numHeads * headDim;
    int headStride = headDim;
    int dstOffset = pageIdx * pageStride + (pageOffset + token) * tokenStride + head * headStride + dim;
    FastllmWritePagedKV(dst, src + srcOffset, dstOffset, pageStride);
}

template <typename SrcT, typename DstT>
static void FastllmCudaPagedCacheCopyTyped(
    uint8_t *pagedData,
    int pageIdx,
    int pageLen,
    int numHeads,
    int headDim,
    uint8_t *inputData,
    int seqLen,
    int inputOffset,
    int copyLen,
    int pageOffset) {
    int totalElements = numHeads * copyLen * headDim;
    if (totalElements == 0) {
        return;
    }

    const int THREAD_PER_BLOCK = 256;
    int numBlocks = (totalElements + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    FastllmPagedCacheCopyKernel<SrcT, DstT, THREAD_PER_BLOCK><<<numBlocks, THREAD_PER_BLOCK>>>(
        pagedData, pageIdx, pageLen, numHeads, headDim,
        inputData, seqLen, inputOffset, copyLen, pageOffset
    );

    DeviceSync();
}

// Host function to launch the kernel
void FastllmCudaPagedCacheCopy(
    uint8_t *pagedData,
    int pageIdx,
    int pageLen,
    int numHeads,
    int headDim,
    fastllm::DataType dstType,
    uint8_t *inputData,
    fastllm::DataType srcType,
    int seqLen,
    int inputOffset,
    int copyLen,
    int pageOffset) {
    bool supported = FastllmDispatchPagedCacheCopyTypes(srcType, dstType, [&](auto *src, auto *dst) {
        using SrcT = std::remove_pointer_t<decltype(src)>;
        using DstT = std::remove_pointer_t<decltype(dst)>;
        FastllmCudaPagedCacheCopyTyped<SrcT, DstT>(
            pagedData, pageIdx, pageLen, numHeads, headDim,
            inputData, seqLen, inputOffset, copyLen, pageOffset);
    });
    fastllm::AssertInFastLLM(supported, "FastllmCudaPagedCacheCopy: unsupported src/dst type.\n");
}

static constexpr int FASTLLM_PAGED_CACHE_COPY_MULTI_MAX_PAGES = 256;

struct FastllmPagedCacheCopyPageList {
    int pageIdx[FASTLLM_PAGED_CACHE_COPY_MULTI_MAX_PAGES];
};

// Copy a head-major sequence into multiple (possibly non-contiguous) cache
// pages in one launch.  The compact page list is passed as a kernel argument,
// avoiding a temporary H2D metadata copy for the common chunked-prefill path.
template <typename SrcT, typename DstT>
__global__ void FastllmPagedCacheCopyMultiPageKernel(
    uint8_t *pagedData,
    FastllmPagedCacheCopyPageList pageList,
    int pageCount,
    int firstPageOffset,
    int pageLen,
    int numHeads,
    int headDim,
    uint8_t *inputData,
    int seqLen) {
    int totalElements = numHeads * seqLen * headDim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) {
        return;
    }

    int head = idx / (seqLen * headDim);
    int remainder = idx % (seqLen * headDim);
    int token = remainder / headDim;
    int dim = remainder % headDim;

    int firstPageCapacity = pageLen - firstPageOffset;
    int pageSlot;
    int pageOffset;
    if (token < firstPageCapacity) {
        pageSlot = 0;
        pageOffset = firstPageOffset + token;
    } else {
        int tailToken = token - firstPageCapacity;
        pageSlot = 1 + tailToken / pageLen;
        pageOffset = tailToken % pageLen;
    }
    if (pageSlot >= pageCount) {
        return;
    }

    const SrcT *src = (const SrcT*)inputData;
    DstT *dst = (DstT*)pagedData;
    int srcOffset = head * seqLen * headDim + token * headDim + dim;
    int pageStride = pageLen * numHeads * headDim;
    int tokenStride = numHeads * headDim;
    int dstOffset = pageList.pageIdx[pageSlot] * pageStride +
                    pageOffset * tokenStride + head * headDim + dim;
    FastllmWritePagedKV(dst, src + srcOffset, dstOffset, pageStride);
}

template <typename SrcT, typename DstT>
static void FastllmCudaPagedCacheCopyMultiPageTyped(
    uint8_t *pagedData,
    const FastllmPagedCacheCopyPageList &pageList,
    int pageCount,
    int firstPageOffset,
    int pageLen,
    int numHeads,
    int headDim,
    uint8_t *inputData,
    int seqLen) {
    int totalElements = numHeads * seqLen * headDim;
    if (totalElements == 0) {
        return;
    }

    const int THREAD_PER_BLOCK = 256;
    int numBlocks = (totalElements + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    FastllmPagedCacheCopyMultiPageKernel<SrcT, DstT>
        <<<numBlocks, THREAD_PER_BLOCK>>>(
            pagedData, pageList, pageCount, firstPageOffset, pageLen,
            numHeads, headDim, inputData, seqLen);
    DeviceSync();
}

bool FastllmCudaPagedCacheCopyMultiPage(
    uint8_t *pagedData,
    const int *pageIdxHost,
    int pageCount,
    int firstPageOffset,
    int pageLen,
    int numHeads,
    int headDim,
    fastllm::DataType dstType,
    uint8_t *inputData,
    fastllm::DataType srcType,
    int seqLen) {
    if (pageIdxHost == nullptr || pageCount <= 0 ||
        pageCount > FASTLLM_PAGED_CACHE_COPY_MULTI_MAX_PAGES ||
        firstPageOffset < 0 || firstPageOffset >= pageLen) {
        return false;
    }

    FastllmPagedCacheCopyPageList pageList = {};
    for (int i = 0; i < pageCount; i++) {
        pageList.pageIdx[i] = pageIdxHost[i];
    }

    bool supported = FastllmDispatchPagedCacheCopyTypes(srcType, dstType, [&](auto *src, auto *dst) {
        using SrcT = std::remove_pointer_t<decltype(src)>;
        using DstT = std::remove_pointer_t<decltype(dst)>;
        FastllmCudaPagedCacheCopyMultiPageTyped<SrcT, DstT>(
            pagedData, pageList, pageCount, firstPageOffset, pageLen,
            numHeads, headDim, inputData, seqLen);
    });
    fastllm::AssertInFastLLM(supported, "FastllmCudaPagedCacheCopyMultiPage: unsupported src/dst type.\n");
    return true;
}

namespace {
    template <typename SrcT, typename DstT>
    __global__ void FastllmPagedCacheAppendPackedBatchKernel(
            DstT *pagedData,
            const int32_t *qSizes, const int32_t *pageSizes,
            const int32_t *pageIndexs, const int32_t *baseTokenLens,
            int batch, int totalTokens, int pageLen,
            int numHeads, int headDim, const SrcT *inputData) {
        int64_t totalElements =
            (int64_t)numHeads * totalTokens * headDim;
        int64_t index =
            (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= totalElements) {
            return;
        }

        int dim = index % headDim;
        int64_t headToken = index / headDim;
        int token = headToken % totalTokens;
        int head = headToken / totalTokens;

        // qSizes is monotonic and verify chains are short. Binary search keeps
        // the same kernel practical for larger scheduler batches as well.
        int left = 0;
        int right = batch;
        while (left + 1 < right) {
            int middle = (left + right) >> 1;
            if (token < qSizes[middle]) {
                right = middle;
            } else {
                left = middle;
            }
        }
        int request = left;
        int requestToken = token - qSizes[request];
        int absoluteToken = baseTokenLens[request] + requestToken;
        int logicalPage = absoluteToken / pageLen;
        int pageOffset = absoluteToken - logicalPage * pageLen;
        int page = pageIndexs[pageSizes[request] + logicalPage];

        int64_t pageStride = (int64_t)pageLen * numHeads * headDim;
        int64_t tokenStride = (int64_t)numHeads * headDim;
        int64_t dstIndex = (int64_t)page * pageStride +
            (int64_t)pageOffset * tokenStride +
            (int64_t)head * headDim + dim;
        FastllmWritePagedKV(pagedData, inputData + index, dstIndex, pageStride);
    }

    template <typename SrcT, typename DstT>
    static bool FastllmCudaPagedCacheAppendPackedBatchTyped(
            uint8_t *pagedData,
            const int32_t *qSizes, const int32_t *pageSizes,
            const int32_t *pageIndexs, const int32_t *baseTokenLens,
            int batch, int totalTokens, int pageLen,
            int numHeads, int headDim, const uint8_t *inputData) {
        int64_t totalElements =
            (int64_t)numHeads * totalTokens * headDim;
        if (totalElements <= 0) {
            return true;
        }
        constexpr int threads = 256;
        int blocks = (int)((totalElements + threads - 1) / threads);
        cudaError_t pendingState = cudaGetLastError();
        if (pendingState != cudaSuccess) {
            std::fprintf(stderr,
                         "[Fastllm] packed paged-cache append got a stale "
                         "CUDA error before launch: cuda=%d (%s), batch=%d "
                         "tokens=%d heads=%d dim=%d.\n",
                         (int)pendingState, cudaGetErrorString(pendingState),
                         batch, totalTokens, numHeads, headDim);
            std::fflush(stderr);
            FastllmCudaSetThreadError();
            return false;
        }
        FastllmPagedCacheAppendPackedBatchKernel<SrcT, DstT>
            <<<blocks, threads, 0, cudaStreamPerThread>>>(
                (DstT*)pagedData, qSizes, pageSizes, pageIndexs,
                baseTokenLens, batch, totalTokens, pageLen,
                numHeads, headDim, (const SrcT*)inputData);
        cudaError_t state = cudaGetLastError();
        if (state != cudaSuccess) {
            std::fprintf(stderr,
                         "[Fastllm] packed paged-cache append launch failed: "
                         "cuda=%d (%s), batch=%d tokens=%d heads=%d dim=%d.\n",
                         (int)state, cudaGetErrorString(state), batch,
                         totalTokens, numHeads, headDim);
            std::fflush(stderr);
            FastllmCudaSetThreadError();
            return false;
        }
        DeviceSync();
        return true;
    }
}

bool FastllmCudaPagedCacheAppendPackedBatch(
        uint8_t *pagedData,
        const int32_t *qSizes, const int32_t *pageSizes,
        const int32_t *pageIndexs, const int32_t *baseTokenLens,
        int batch, int totalTokens, int pageLen, int numHeads, int headDim,
        fastllm::DataType dstType, const uint8_t *inputData,
        fastllm::DataType srcType) {
    if (pagedData == nullptr || qSizes == nullptr || pageSizes == nullptr ||
        pageIndexs == nullptr || baseTokenLens == nullptr ||
        inputData == nullptr || batch <= 0 || totalTokens <= 0 ||
        pageLen <= 0 || numHeads <= 0 || headDim <= 0) {
        return false;
    }

    bool success = false;
    bool supported = FastllmDispatchPagedCacheCopyTypes(srcType, dstType, [&](auto *src, auto *dst) {
        using SrcT = std::remove_pointer_t<decltype(src)>;
        using DstT = std::remove_pointer_t<decltype(dst)>;
        success = FastllmCudaPagedCacheAppendPackedBatchTyped<SrcT, DstT>(
            pagedData, qSizes, pageSizes, pageIndexs, baseTokenLens,
            batch, totalTokens, pageLen, numHeads, headDim, inputData);
    });
    if (!supported) {
        std::fprintf(stderr,
                     "[Fastllm] packed paged-cache append unsupported dtype: src=%d dst=%d.\n",
                     (int)srcType, (int)dstType);
        std::fflush(stderr);
    }
    return success;
}

__global__ void FastllmPreparePagedBatchParamsSingleKernel(
    int32_t *qSizes,
    int32_t *pageSizes,
    int32_t *pageIndexs,
    int32_t *lastPageLens,
    FastllmPagedCacheCopyPageList pageList,
    int pageIndexCount,
    int totalPages,
    int qSize,
    int lastPageLen) {
    int idx = threadIdx.x;
    if (idx < pageIndexCount) {
        pageIndexs[idx] = pageList.pageIdx[idx];
    }
    if (idx == 0) {
        qSizes[0] = 0;
        qSizes[1] = qSize;
        pageSizes[0] = 0;
        pageSizes[1] = totalPages;
        lastPageLens[0] = lastPageLen;
    }
}

bool FastllmCudaPreparePagedBatchParamsSingle(
    int32_t *qSizes,
    int32_t *pageSizes,
    int32_t *pageIndexs,
    int32_t *lastPageLens,
    const int *pageIdxHost,
    int pageIndexCount,
    int totalPages,
    int qSize,
    int lastPageLen) {
    if (qSizes == nullptr || pageSizes == nullptr || pageIndexs == nullptr ||
        lastPageLens == nullptr || pageIdxHost == nullptr ||
        pageIndexCount <= 0 ||
        pageIndexCount > FASTLLM_PAGED_CACHE_COPY_MULTI_MAX_PAGES) {
        return false;
    }

    FastllmPagedCacheCopyPageList pageList = {};
    for (int i = 0; i < pageIndexCount; i++) {
        pageList.pageIdx[i] = pageIdxHost[i];
    }
    FastllmPreparePagedBatchParamsSingleKernel<<<1, 256>>>(
        qSizes, pageSizes, pageIndexs, lastPageLens, pageList,
        pageIndexCount, totalPages, qSize, lastPageLen);
    DeviceSync();
    return true;
}

// Upstream #722: the pageable H2D copies of the paged batch parameters hold
// the CUDA driver lock while the DMA completes; past the 256-page boundary
// that window overlaps the peer rank's allocation/collective submission and
// can deadlock a long-context prefill.  Carry the values in kernel parameters
// instead: no blocking copy, and the launch is capturable by a CUDA Graph.
constexpr int kFastllmPagedIntParamsMaxSmall = 64;
constexpr int kFastllmPagedIntParamsChunkPages = 512;
constexpr int kFastllmPagedIntParamsMaxPages = 4096;

template <int MaxPages>
struct FastllmPagedIntParamsList {
    int32_t qSizes[kFastllmPagedIntParamsMaxSmall];
    int32_t pageSizes[kFastllmPagedIntParamsMaxSmall];
    int32_t lastPageLens[kFastllmPagedIntParamsMaxSmall];
    int32_t pageIdx[MaxPages];
};

// Leave space for pointer/count arguments on pre-Volta and older toolchains.
// Bounded launches also work in fat binaries containing both sm_60 and sm_75.
static_assert(sizeof(FastllmPagedIntParamsList<kFastllmPagedIntParamsChunkPages>)
                  + 128 <= 4096, "Paged upload kernel exceeds legacy parameter space");

template <int MaxPages>
__global__ void FastllmUploadPagedIntParamsKernel(
        int32_t *qSizes, int qSizesCount,
        int32_t *pageSizes, int pageSizesCount,
        int32_t *pageIndexs, int pageIndexsCount,
        int32_t *lastPageLens, int lastPageLensCount,
        FastllmPagedIntParamsList<MaxPages> values) {
    const int total = qSizesCount + pageSizesCount + pageIndexsCount +
                      lastPageLensCount;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        if (i < qSizesCount) {
            qSizes[i] = values.qSizes[i];
        } else if (i < qSizesCount + pageSizesCount) {
            const int j = i - qSizesCount;
            pageSizes[j] = values.pageSizes[j];
        } else if (i < qSizesCount + pageSizesCount + pageIndexsCount) {
            const int j = i - qSizesCount - pageSizesCount;
            pageIndexs[j] = values.pageIdx[j];
        } else {
            const int j = i - qSizesCount - pageSizesCount - pageIndexsCount;
            lastPageLens[j] = values.lastPageLens[j];
        }
    }
}

template <int MaxPages>
static bool UploadPagedIntParamsKernel(
        int32_t *qSizes, int qSizesCount,
        int32_t *pageSizes, int pageSizesCount,
        int32_t *pageIndexs, int pageIndexsCount,
        int32_t *lastPageLens, int lastPageLensCount,
        const int *qSizesHost, const int *pageSizesHost,
        const int *pageIndexsHost, const int *lastPageLensHost) {
    FastllmPagedIntParamsList<MaxPages> values = {};
    for (int i = 0; i < qSizesCount; i++) {
        values.qSizes[i] = qSizesHost[i];
    }
    for (int i = 0; i < pageSizesCount; i++) {
        values.pageSizes[i] = pageSizesHost[i];
    }
    for (int i = 0; i < pageIndexsCount; i++) {
        values.pageIdx[i] = pageIndexsHost[i];
    }
    for (int i = 0; i < lastPageLensCount; i++) {
        values.lastPageLens[i] = lastPageLensHost[i];
    }
    FastllmUploadPagedIntParamsKernel<MaxPages><<<1, 256>>>(
        qSizes, qSizesCount, pageSizes, pageSizesCount,
        pageIndexs, pageIndexsCount, lastPageLens, lastPageLensCount,
        values);
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaUploadPagedIntParams(
        int32_t *qSizes, int qSizesCount,
        int32_t *pageSizes, int pageSizesCount,
        int32_t *pageIndexs, int pageIndexsCount,
        int32_t *lastPageLens, int lastPageLensCount,
        const int *qSizesHost, const int *pageSizesHost,
        const int *pageIndexsHost, const int *lastPageLensHost) {
    if (qSizes == nullptr || pageSizes == nullptr || pageIndexs == nullptr ||
        qSizesCount <= 0 ||
        qSizesCount > kFastllmPagedIntParamsMaxSmall ||
        pageSizesCount <= 0 ||
        pageSizesCount > kFastllmPagedIntParamsMaxSmall ||
        lastPageLensCount < 0 ||
        lastPageLensCount > kFastllmPagedIntParamsMaxSmall ||
        pageIndexsCount < 0 ||
        pageIndexsCount > kFastllmPagedIntParamsMaxPages ||
        (qSizesCount > 0 && qSizesHost == nullptr) ||
        (pageSizesCount > 0 && pageSizesHost == nullptr) ||
        (pageIndexsCount > 0 && pageIndexsHost == nullptr) ||
        (lastPageLensCount > 0 &&
         (lastPageLens == nullptr || lastPageLensHost == nullptr))) {
        return false;
    }
    // Launch only small by-value parameter lists. This preserves capture and
    // avoids pageable H2D copies without requiring large kernel-argument support.
    // Metadata is written once; following launches upload disjoint page chunks.
    bool uploaded = true;
    if (pageIndexsCount <= 256) {
        uploaded = UploadPagedIntParamsKernel<256>(
            qSizes, qSizesCount, pageSizes, pageSizesCount, pageIndexs,
            pageIndexsCount, lastPageLens, lastPageLensCount, qSizesHost,
            pageSizesHost, pageIndexsHost, lastPageLensHost);
    } else {
        for (int offset = 0; offset < pageIndexsCount;
             offset += kFastllmPagedIntParamsChunkPages) {
            const bool first = offset == 0;
            const int count = std::min(kFastllmPagedIntParamsChunkPages,
                                       pageIndexsCount - offset);
            uploaded = UploadPagedIntParamsKernel<kFastllmPagedIntParamsChunkPages>(
                qSizes, first ? qSizesCount : 0,
                pageSizes, first ? pageSizesCount : 0,
                pageIndexs + offset, count,
                lastPageLens, first ? lastPageLensCount : 0,
                qSizesHost, pageSizesHost, pageIndexsHost + offset,
                lastPageLensHost);
            if (!uploaded) {
                break;
            }
        }
    }
    if (!uploaded) {
        return false;
    }
    // Honor debug synchronization once after all launches, outside capture.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess) {
        return false;
    }
    if (capture == cudaStreamCaptureStatusNone) {
        DeviceSync();
    }
    return true;
}

// CUDA kernel for batch copying data from input to paged KV cache
// input: [batch, numHeads, headDim], pagedData: [maxPages, pageLen, numHeads, headDim]
// Each batch has 1 token, so we copy [numHeads, headDim] for each batch
template <typename SrcT, typename DstT, int THREAD_PER_BLOCK>
__global__ void FastllmPagedCacheCopyBatchKernel(
    uint8_t *pagedData,         // dst: [maxPages, pageLen, numHeads, headDim]
    int32_t *pageIdxArray,      // page index for each batch [batch]
    int32_t *pageOffsetArray,   // page offset for each batch [batch]
    int pageLen,                // page length
    int batch,                  // number of batches
    int numHeads,               // number of heads
    int headDim,                // head dimension
    uint8_t *inputData          // src: [batch, numHeads, headDim]
) {
    // Calculate the linear index for this thread
    // Each thread handles one element: (batch, head, dim)
    int totalElements = batch * numHeads * headDim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalElements) {
        return;
    }

    // Decompose linear index into (batch, head, dim)
    int b = idx / (numHeads * headDim);
    int remainder = idx % (numHeads * headDim);
    int h = remainder / headDim;
    int d = remainder % headDim;

    // Get page index and offset for this batch
    int pageIdx = pageIdxArray[b];
    int pageOffset = pageOffsetArray[b];

    // Calculate source address: input[b, h, d]
    // input layout: [batch, numHeads, headDim]
    const SrcT *src = (const SrcT*)inputData;
    DstT *dst = (DstT*)pagedData;
    int srcOffset = b * numHeads * headDim + h * headDim + d;

    // Calculate destination address: pagedData[pageIdx, pageOffset, h, d]
    // pagedData layout: [maxPages, pageLen, numHeads, headDim]
    int pageStride = pageLen * numHeads * headDim;
    int tokenStride = numHeads * headDim;
    int headStride = headDim;
    int dstOffset = pageIdx * pageStride + pageOffset * tokenStride + h * headStride + d;
    FastllmWritePagedKV(dst, src + srcOffset, dstOffset, pageStride);
}

template <typename SrcT, typename DstT>
static void FastllmCudaPagedCacheCopyBatchTyped(
    uint8_t *pagedData,
    int32_t *pageIdxArray,
    int32_t *pageOffsetArray,
    int pageLen,
    int batch,
    int numHeads,
    int headDim,
    uint8_t *inputData,
    bool sync) {
    int totalElements = batch * numHeads * headDim;
    if (totalElements == 0) {
        return;
    }

    const int THREAD_PER_BLOCK = 256;
    int numBlocks = (totalElements + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    FastllmPagedCacheCopyBatchKernel<SrcT, DstT, THREAD_PER_BLOCK><<<numBlocks, THREAD_PER_BLOCK>>>(
        pagedData, pageIdxArray, pageOffsetArray, pageLen, batch, numHeads, headDim, inputData
    );

    if (sync) {
        DeviceSync();
    }
}

// Host function to launch the batch kernel
void FastllmCudaPagedCacheCopyBatch(
    uint8_t *pagedData,
    int32_t *pageIdxArray,
    int32_t *pageOffsetArray,
    int pageLen,
    int batch,
    int numHeads,
    int headDim,
    fastllm::DataType dstType,
    uint8_t *inputData,
    fastllm::DataType srcType,
    bool sync) {
    bool supported = FastllmDispatchPagedCacheCopyTypes(srcType, dstType, [&](auto *src, auto *dst) {
        using SrcT = std::remove_pointer_t<decltype(src)>;
        using DstT = std::remove_pointer_t<decltype(dst)>;
        FastllmCudaPagedCacheCopyBatchTyped<SrcT, DstT>(
            pagedData, pageIdxArray, pageOffsetArray, pageLen,
            batch, numHeads, headDim, inputData, sync);
    });
    fastllm::AssertInFastLLM(supported, "FastllmCudaPagedCacheCopyBatch: unsupported src/dst type.\n");
}
