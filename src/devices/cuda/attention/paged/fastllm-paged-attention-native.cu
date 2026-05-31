//
// Native paged attention kernels (FlashInfer fallback, CUDA-graph capturable).
//
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "utils/utils.h"
#include "attention/fastllm-attention-dtype.cuh"
#include "attention/fastllm-paged-attention-native.cuh"

#include <cuda_fp8.h>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

// Gather paged KV [maxPages, pageLen, numHeads, headDim] -> contiguous HND [numHeads, kvLen, headDim].
template <typename SrcT, int THREAD_PER_BLOCK>
__global__ void FastllmPagedCacheGatherKernel(
    const uint8_t *pagedData,
    const int32_t *pageIndices,
    int numPages,
    int lastPageLen,
    int pageLen,
    int numHeads,
    int headDim,
    int kvLen,
    uint8_t *outData) {
    int totalElements = numHeads * kvLen * headDim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) {
        return;
    }
    int head = idx / (kvLen * headDim);
    int remainder = idx % (kvLen * headDim);
    int token = remainder / headDim;
    int dim = remainder % headDim;

    int remaining = token;
    int pageListIdx = 0;
    int offsetInPage = 0;
    for (int p = 0; p < numPages; p++) {
        int len = (p == numPages - 1) ? lastPageLen : pageLen;
        if (remaining < len) {
            pageListIdx = p;
            offsetInPage = remaining;
            break;
        }
        remaining -= len;
    }

    const SrcT *src = (const SrcT*)pagedData;
    half *dst = (half*)outData;
    int pageStride = pageLen * numHeads * headDim;
    int tokenStride = numHeads * headDim;
    int srcOffset = pageIndices[pageListIdx] * pageStride + offsetInPage * tokenStride + head * headDim + dim;
    dst[idx] = __float2half(FastllmAttentionValueToFloat<SrcT>(src[srcOffset]));
}

template <typename SrcT>
static void FastllmCudaPagedCacheGatherContiguous(
    uint8_t *pagedData,
    const int32_t *pageIndices,
    int numPages,
    int lastPageLen,
    int pageLen,
    int numHeads,
    int headDim,
    int kvLen,
    half *outData) {
    int totalElements = numHeads * kvLen * headDim;
    if (totalElements <= 0) {
        return;
    }
    const int THREAD_PER_BLOCK = 256;
    int numBlocks = (totalElements + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    FastllmPagedCacheGatherKernel<SrcT, THREAD_PER_BLOCK><<<numBlocks, THREAD_PER_BLOCK>>>(
        pagedData, pageIndices, numPages, lastPageLen, pageLen, numHeads, headDim, kvLen,
        (uint8_t*)outData);
}

static void FastllmCudaPagedCacheGatherToHalf(
    fastllm::Data *pagedKVCache,
    const std::vector<int32_t> &pageIndices,
    int lastPageLen,
    int pageLen,
    int numHeads,
    int headDim,
    int kvLen,
    half *outData) {
    int numPages = (int)pageIndices.size();
    if (numPages <= 0 || kvLen <= 0) {
        return;
    }
    int32_t *pageIndicesGpu = (int32_t*)FastllmCudaMalloc((size_t)numPages * sizeof(int32_t));
    cudaMemcpy(pageIndicesGpu, pageIndices.data(), (size_t)numPages * sizeof(int32_t), cudaMemcpyHostToDevice);
    uint8_t *pagedBytes = (uint8_t*)pagedKVCache->cudaData;
    if (pagedKVCache->dataType == fastllm::DataType::FLOAT16) {
        FastllmCudaPagedCacheGatherContiguous<half>(pagedBytes, pageIndicesGpu, numPages, lastPageLen,
                                                    pageLen, numHeads, headDim, kvLen, outData);
    } else if (pagedKVCache->dataType == fastllm::DataType::BFLOAT16) {
        FastllmCudaPagedCacheGatherContiguous<__nv_bfloat16>(pagedBytes, pageIndicesGpu, numPages, lastPageLen,
                                                             pageLen, numHeads, headDim, kvLen, outData);
    } else if (pagedKVCache->dataType == fastllm::DataType::FP8_E4M3) {
        FastllmCudaPagedCacheGatherContiguous<__nv_fp8_e4m3>(pagedBytes, pageIndicesGpu, numPages, lastPageLen,
                                                             pageLen, numHeads, headDim, kvLen, outData);
    } else {
        FastllmCudaFree(pageIndicesGpu);
        printf("FastllmCudaPagedCacheGatherToHalf: unsupported paged KV cache dataType=%d\n",
               (int)pagedKVCache->dataType);
        exit(0);
    }
    FastllmCudaFree(pageIndicesGpu);
}

static bool FastllmCudaHalfPagedAttentionNative(
    fastllm::Data &q,
    const std::vector<int32_t> &pageIndices,
    int lastPageLen,
    fastllm::Data *pagedKVCacheK,
    fastllm::Data *pagedKVCacheV,
    int pageLen,
    int numKvHeads,
    int headDim,
    fastllm::Data &output,
    int group,
    float scale,
    bool permuteOutput = true) {
    if (q.dataType != fastllm::DataType::FLOAT16) {
        printf("FastllmCudaHalfPagedAttentionNative: only FLOAT16 query is supported, got %d\n", (int)q.dataType);
        return false;
    }
    int numPages = (int)pageIndices.size();
    if (numPages <= 0) {
        printf("FastllmCudaHalfPagedAttentionNative: empty page list\n");
        return false;
    }
    int kvLen = (numPages - 1) * pageLen + lastPageLen;
    if (kvLen <= 0) {
        return false;
    }

    size_t kvBytes = (size_t)numKvHeads * kvLen * headDim * sizeof(half);
    half *kContig = (half*)FastllmCudaMalloc(kvBytes);
    half *vContig = (half*)FastllmCudaMalloc(kvBytes);
    FastllmCudaPagedCacheGatherToHalf(pagedKVCacheK, pageIndices, lastPageLen, pageLen, numKvHeads, headDim, kvLen, kContig);
    FastllmCudaPagedCacheGatherToHalf(pagedKVCacheV, pageIndices, lastPageLen, pageLen, numKvHeads, headDim, kvLen, vContig);

    fastllm::Data kData, vData, emptyMask;
    kData.dataType = fastllm::DataType::FLOAT16;
    kData.dataDevice = fastllm::DataDevice::CUDA;
    // 使用 Resize 正确填充 strides，否则 Data::Count 会越界访问 strides。
    kData.Resize({numKvHeads, kvLen, headDim});
    kData.cudaData = kContig;
    kData.isFake = true; // 视图，内存由本函数显式管理，析构时不要释放
    vData = kData;
    vData.cudaData = vContig;
    vData.isFake = true;

    bool ok = FastllmCudaHalfAttention(q, kData, vData, emptyMask, output, group, scale, 0);
    FastllmCudaFree(kContig);
    FastllmCudaFree(vContig);
    if (ok && permuteOutput) {
        // FastllmCudaHalfAttention 将结果写为 head-major 物理布局 [numHeads, qoLen, headDim]，
        // 这里显式按该物理布局设置 dims（不依赖入口 dims），再转置成与 FlashInfer 一致的
        // token-major 布局 [qoLen, numHeads, headDim]。
        int numHeads = q.dims[0];
        int qoLen = q.dims[1];
        int outHeadDim = output.dims[2];
        output.Resize({numHeads, qoLen, outHeadDim});
        FastllmCudaPermute(output, {1, 0, 2});
        DeviceSync();
    }
    return ok;
}

bool FastllmCudaHalfPagedAttentionFastllmFallback(
    fastllm::Data &q,
    fastllm::Data &k,
    fastllm::Data &v,
    fastllm::Data &output,
    int group,
    float scale) {
    if (!k.isPagedKVCache || !v.isPagedKVCache) {
        printf("FastllmCudaHalfPagedAttentionFastllmFallback: k/v must be paged KV cache.\n");
        exit(0);
    }
    fastllm::Data *pagedKVCacheK = k.pagedKVCacheData;
    fastllm::Data *pagedKVCacheV = v.pagedKVCacheData;
    if (pagedKVCacheK == nullptr || pagedKVCacheV == nullptr) {
        printf("FastllmCudaHalfPagedAttentionFastllmFallback: pagedKVCacheData is nullptr.\n");
        exit(0);
    }
    int numKvHeads = k.dims[0];
    int headDim = pagedKVCacheK->dims[3];
    return FastllmCudaHalfPagedAttentionNative(q, k.pageIndex, k.lastPageLen, pagedKVCacheK, pagedKVCacheV,
                                               k.pageLen, numKvHeads, headDim, output, group, scale);
}

// CUDA-graph 可捕获的分页注意力 kernel（前缀填充 + 解码统一实现）。
// 所有分页元数据均从 device 缓冲读取，kernel 内部按运行时长度循环，
// grid 维度仅依赖 batch_size 与 head 数（在每个捕获的 graph 中固定），
// 不做任何同步拷贝 / malloc / free，因此可安全用于 CUDA graph 流捕获与重放。
//
// 内存布局约定（与 FlashInfer 路径一致）：
//   q     : head-major [H, totalTokens, headDim]，token 维按 qSizes 在各 batch 间 ragged 拼接，
//           q(h, token, d) = qd[h*q_stride_h + token*q_stride_n + d]
//   paged : [maxPages, pageLen, numKvHeads, headDim]
//   output: token-major [totalTokens, H, headDim]，out(token, h, d) = od[token*H*headDim + h*headDim + d]
//   因果：batch 内第 tok 个 query（0 基）可见该 batch 的前 (kvLen - qoLen + tok + 1) 个 key。
__device__ __forceinline__ float FastllmWarpAllReduceSum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// pageIndex[i]==i 时 token j 的物理地址可线性化，省去每次 page 查表。
__device__ __forceinline__ size_t FastllmPagedKvTokenBase(
    int j, int pageStart, int pageLen, int numPages,
    const int32_t *pageIndexs, size_t pageStride, size_t tokenStride, size_t kvHeadOffset) {
    int pageListIdx = j / pageLen;
    int offsetInPage = j - pageListIdx * pageLen;
    if (pageStart == 0 && pageListIdx < numPages && pageIndexs[pageStart] == 0 &&
        pageIndexs[pageStart + pageListIdx] == pageListIdx) {
        return (size_t)j * tokenStride + kvHeadOffset;
    }
    int page = pageIndexs[pageStart + pageListIdx];
    return (size_t)page * pageStride + (size_t)offsetInPage * tokenStride + kvHeadOffset;
}

static bool FastllmPagedUseGqaDecode() {
    const char *env = std::getenv("FASTLLM_PAGED_GQA_DECODE");
    return env == nullptr || env[0] != '0';
}

// 微基准 / 对照：强制走非 split 的 BatchKernel（与部分 CUDA graph 捕获路径一致）。
static bool FastllmPagedForceNoSplit() {
    const char *env = std::getenv("FASTLLM_PAGED_NO_SPLIT");
    return env != nullptr && env[0] == '1';
}

static bool FastllmPagedUseGqaDecodeFor(int group, int headDim, int H, int numKvHeads) {
    return FastllmPagedUseGqaDecode() && group > 1 && headDim > 0 && headDim <= 128 &&
           group <= 4 && H == group * numKvHeads;
}

template <typename KVType>
__device__ __forceinline__ KVType FastllmKvLoad(const KVType *base, int d) {
    return __ldg(base + d);
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 FastllmKvLoad<__nv_fp8_e4m3>(const __nv_fp8_e4m3 *base, int d) {
    return __nv_fp8_e4m3(__ldg(reinterpret_cast<const unsigned char *>(base) + d));
}

template <typename KVType>
__device__ __forceinline__ float FastllmKvLoadFloat(const KVType *base, int d) {
    return FastllmAttentionValueToFloat(FastllmKvLoad(base, d));
}

// 向量化加载 4 个「连续」的 KV 元素到 out[0..3]（float）。
// 调用方保证 d0 % 4 == 0；当 d0+4 <= headDim 时走一次性向量化读取（warp 内 32 lane 各读 4 个连续元素，
// 合并成单条 128B/256B 事务，完全 coalesce），否则逐元素回退（仅 headDim 非 128 的尾部）。
// 这是 SM70(V100) 上把分页注意力从 ~30% 带宽拉高的关键：原实现按 d=lane+(i*32) 跨步逐元素 __ldg，
// 每个 key 需 4 条 64B 事务，向量化后降为 1 条指令。
template <typename KVType>
__device__ __forceinline__ void FastllmKvLoad4Contig(const KVType *base, int d0, int headDim, float out[4]);

template <>
__device__ __forceinline__ void FastllmKvLoad4Contig<half>(const half *base, int d0, int headDim, float out[4]) {
    if (d0 + 4 <= headDim) {
        uint2 raw = __ldg(reinterpret_cast<const uint2 *>(base + d0));   // 8 字节 = 4 个 half
        const half2 *h2 = reinterpret_cast<const half2 *>(&raw);
        float2 a = __half22float2(h2[0]);
        float2 b = __half22float2(h2[1]);
        out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            out[i] = (d0 + i < headDim) ? __half2float(__ldg(base + d0 + i)) : 0.0f;
        }
    }
}

template <>
__device__ __forceinline__ void FastllmKvLoad4Contig<__nv_bfloat16>(const __nv_bfloat16 *base, int d0, int headDim, float out[4]) {
    if (d0 + 4 <= headDim) {
        uint2 raw = __ldg(reinterpret_cast<const uint2 *>(base + d0));   // 8 字节 = 4 个 bf16
        const __nv_bfloat162 *b2 = reinterpret_cast<const __nv_bfloat162 *>(&raw);
        float2 a = __bfloat1622float2(b2[0]);
        float2 b = __bfloat1622float2(b2[1]);
        out[0] = a.x; out[1] = a.y; out[2] = b.x; out[3] = b.y;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            out[i] = (d0 + i < headDim) ? __bfloat162float(__ldg(base + d0 + i)) : 0.0f;
        }
    }
}

template <>
__device__ __forceinline__ void FastllmKvLoad4Contig<__nv_fp8_e4m3>(const __nv_fp8_e4m3 *base, int d0, int headDim, float out[4]) {
    // FP8 与标量路径保持逐元素一致的数值语义（不改变 FP8 行为），仅改为连续 d 排布。
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out[i] = (d0 + i < headDim) ? FastllmKvLoadFloat<__nv_fp8_e4m3>(base, d0 + i) : 0.0f;
    }
}

// flash-decoding 风格：每个 block 处理一个 (batch, head)，blockDim = MAX_WARPS*32。
//   - 多个 warp 并行处理不同的 key（warp 内用 shfl 归约点积，避免逐 key 的 __syncthreads）；
//   - 每个 warp 维护各自的在线 softmax 状态，最后一次性跨 warp 合并；
//   - 每个 token 仅做一次 block 同步（载入 q + 合并），而非每个 key 同步。
// 约束：headDim <= DIMS_PER_LANE*32（acc 每 lane 承载 DIMS_PER_LANE 维），numWarps <= 8。
// DIMS_PER_LANE=4 覆盖 headDim<=128，DIMS_PER_LANE=8 覆盖 headDim<=256。
template <typename QType, typename KVType, int DIMS_PER_LANE>
__global__ void FastllmPagedAttentionBatchKernel(
    const QType *qd,
    const KVType *pagedK,
    const KVType *pagedV,
    QType *od,
    const int32_t *qSizes,        // [batch+1] 前缀和
    const int32_t *pageSizes,     // [batch+1] 前缀和
    const int32_t *pageIndexs,    // [totalPages] 扁平页索引
    const int32_t *lastPageLens,  // [batch] 每个 batch 最后一页长度
    int H,                        // 每个 batch 的 query head 数 = group * numKvHeads
    int group,
    int numKvHeads,
    int headDim,
    int pageLen,
    int q_stride_h,
    int q_stride_n,
    float scale) {
    int b = blockIdx.x;   // batch 下标
    int h = blockIdx.y;   // query head 下标
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;
    int numWarps = blockDim.x >> 5;

    int tokenStart = qSizes[b];
    int qoLen = qSizes[b + 1] - tokenStart;
    int pageStart = pageSizes[b];
    int numPages = pageSizes[b + 1] - pageStart;
    int kvLen = (numPages > 0) ? ((numPages - 1) * pageLen + lastPageLens[b]) : 0;

    const int HD_PAD = DIMS_PER_LANE * 32;
    __shared__ float sQ[HD_PAD];
    __shared__ float sM[8];
    __shared__ float sL[8];
    __shared__ float sAcc[8 * HD_PAD];

    if (qoLen <= 0 || numPages <= 0 || kvLen <= 0) {
        return;   // 整个 block 条件一致，安全返回
    }

    int kvHead = h / group;
    size_t pageStride = (size_t)pageLen * numKvHeads * headDim;
    size_t tokenStride = (size_t)numKvHeads * headDim;
    size_t kvHeadOffset = (size_t)kvHead * headDim;

    for (int tok = 0; tok < qoLen; tok++) {
        int token = tokenStart + tok;
        for (int d = tid; d < headDim; d += blockDim.x) {
            sQ[d] = FastllmAttentionValueToFloat<QType>(
                qd[(size_t)h * q_stride_h + (size_t)token * q_stride_n + d]);
        }
        __syncthreads();

        int visible = kvLen - qoLen + tok + 1;   // 因果可见 key 数
        if (visible > kvLen) visible = kvLen;
        if (visible < 0) visible = 0;

        float m = -1e30f, l = 0.0f;
        float acc[DIMS_PER_LANE];
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; i++) {
            acc[i] = 0.0f;
        }

        for (int j = warpId; j < visible; j += numWarps) {
            size_t base = FastllmPagedKvTokenBase(j, pageStart, pageLen, numPages, pageIndexs,
                                                  pageStride, tokenStride, kvHeadOffset);

            float partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < DIMS_PER_LANE; i++) {
                int d = lane + (i << 5);
                if (d < headDim) {
                    partial += sQ[d] * FastllmKvLoadFloat<KVType>(pagedK + base, d);
                }
            }
            float score = FastllmWarpAllReduceSum(partial) * scale;

            float newM = fmaxf(m, score);
            float corr = __expf(m - newM);
            float p = __expf(score - newM);
            #pragma unroll
            for (int i = 0; i < DIMS_PER_LANE; i++) {
                int d = lane + (i << 5);
                if (d < headDim) {
                    acc[i] = acc[i] * corr + p * FastllmKvLoadFloat<KVType>(pagedV + base, d);
                }
            }
            l = l * corr + p;
            m = newM;
        }

        if (lane == 0) {
            sM[warpId] = m;
            sL[warpId] = l;
        }
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; i++) {
            int d = lane + (i << 5);
            if (d < headDim) {
                sAcc[warpId * headDim + d] = acc[i];
            }
        }
        __syncthreads();

        // 跨 warp 合并在线 softmax 状态
        float M = -1e30f;
        for (int w = 0; w < numWarps; w++) {
            M = fmaxf(M, sM[w]);
        }
        float L = 0.0f;
        for (int w = 0; w < numWarps; w++) {
            L += sL[w] * __expf(sM[w] - M);
        }
        for (int d = tid; d < headDim; d += blockDim.x) {
            float o = 0.0f;
            for (int w = 0; w < numWarps; w++) {
                o += sAcc[w * headDim + d] * __expf(sM[w] - M);
            }
            o = (L > 0.0f) ? (o / L) : 0.0f;
            od[(size_t)token * H * headDim + (size_t)h * headDim + d] =
                FastllmAttentionFloatToValue<QType>(o);
        }
        __syncthreads();   // 下一个 token 复用 shared 前同步
    }
}

template <typename QType, int DIMS_PER_LANE>
static bool FastllmPagedAttentionBatchKernelDispatchKV(
    dim3 grid, dim3 block, QType *qd, QType *od,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int32_t *qSizesData, int32_t *pageSizesData, int32_t *pageIndexsData, int32_t *lastPageLensData,
    int H, int group, int numKvHeads, int headDim, int pageLen,
    int q_stride_h, int q_stride_n, float scale) {
    if (pagedKVCacheK->dataType == fastllm::DataType::FP8_E4M3) {
        FastllmPagedAttentionBatchKernel<QType, __nv_fp8_e4m3, DIMS_PER_LANE><<<grid, block>>>(
            qd, (__nv_fp8_e4m3*)pagedKVCacheK->cudaData, (__nv_fp8_e4m3*)pagedKVCacheV->cudaData, od,
            qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
    } else if (pagedKVCacheK->dataType == fastllm::DataType::BFLOAT16) {
        FastllmPagedAttentionBatchKernel<QType, __nv_bfloat16, DIMS_PER_LANE><<<grid, block>>>(
            qd, (__nv_bfloat16*)pagedKVCacheK->cudaData, (__nv_bfloat16*)pagedKVCacheV->cudaData, od,
            qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
    } else if (pagedKVCacheK->dataType == fastllm::DataType::FLOAT16) {
        FastllmPagedAttentionBatchKernel<QType, half, DIMS_PER_LANE><<<grid, block>>>(
            qd, (half*)pagedKVCacheK->cudaData, (half*)pagedKVCacheV->cudaData, od,
            qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
    } else {
        printf("FastllmCudaPagedAttentionBatchKernelLaunch: unsupported KV dataType=%d\n",
               (int)pagedKVCacheK->dataType);
        return false;
    }
    return true;
}

template <typename QType>
static bool FastllmCudaPagedAttentionBatchKernelLaunch(
    fastllm::Data &q, fastllm::Data &output,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int32_t *qSizesData, int32_t *pageSizesData, int32_t *pageIndexsData, int32_t *lastPageLensData,
    uint32_t batch_size, int H, int group, int numKvHeads, int headDim, int pageLen,
    int q_stride_h, int q_stride_n, float scale) {
    // 该 kernel 中每个 lane 承载 DIMS_PER_LANE 个维度，DIMS_PER_LANE=4 覆盖 headDim<=128，
    // DIMS_PER_LANE=8 覆盖 headDim<=256；numWarps<=8。
    if (headDim <= 0 || headDim > 256) {
        printf("FastllmCudaPagedAttentionBatchKernelLaunch: unsupported headDim=%d\n", headDim);
        return false;
    }
    const unsigned int kBlockThreads = 256;   // 8 warps
    dim3 grid(batch_size, (unsigned int)H, 1);
    dim3 block(kBlockThreads, 1, 1);
    QType *qd = (QType*)q.cudaData;
    QType *od = (QType*)output.cudaData;
    if (headDim <= 128) {
        return FastllmPagedAttentionBatchKernelDispatchKV<QType, 4>(
            grid, block, qd, od, pagedKVCacheK, pagedKVCacheV,
            qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
    }
    return FastllmPagedAttentionBatchKernelDispatchKV<QType, 8>(
        grid, block, qd, od, pagedKVCacheK, pagedKVCacheV,
        qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
        H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
}

// ===================== split-KV flash-decoding（仅解码，qoLen==1）=====================
// 单序列解码时只有 batch*H 个 block，远不足以填满 GPU，导致随上下文增长 attention 线性变慢。
// 这里把每个 (batch, head) 的 KV 再切成 S 段，用 batch*H*S 个 block 并行（phase1 写部分 softmax 状态），
// 再用一个合并 kernel（phase2）跨段归并。S 仅依赖 batch 与 H（在每个捕获的 graph 中固定），
// 部分结果写入进程级持久 scratch（首次在非捕获的 warmup 前向中分配，之后固定地址复用），
// 因此整条路径对 CUDA graph 捕获 / 重放安全。

static const int FASTLLM_PAGED_MAX_SPLITS = 128;
static const int FASTLLM_PAGED_SPLIT_TARGET_BLOCKS = 1024;   // per-head split 期望并发 block 数
static const int FASTLLM_PAGED_SPLIT_TARGET_BLOCKS_GQA = 384; // GQA：在 SM 占用与 combine 开销间折中（batch=1 时 S≈48）

static int FastllmPagedSplitTargetBlocks(bool gqa) {
    const char *env = std::getenv("FASTLLM_PAGED_SPLIT_TARGET");
    if (env != nullptr && env[0] != '\0') {
        int t = atoi(env);
        return t < 32 ? 32 : (t > 4096 ? 4096 : t);
    }
    return gqa ? FASTLLM_PAGED_SPLIT_TARGET_BLOCKS_GQA : FASTLLM_PAGED_SPLIT_TARGET_BLOCKS;
}

// 根据 batch 与并行 head 数选择切分段数（host 端确定，CUDA graph 捕获与重放一致）。
static int FastllmChoosePagedSplits(uint32_t batch_size, int parallelHeads, bool gqa) {
    int bh = (int)batch_size * parallelHeads;
    if (bh <= 0) {
        return 1;
    }
    int target = FastllmPagedSplitTargetBlocks(gqa);
    int s = (target + bh - 1) / bh;
    if (s < 1) {
        s = 1;
    }
    if (s > FASTLLM_PAGED_MAX_SPLITS) {
        s = FASTLLM_PAGED_MAX_SPLITS;
    }
    return s;
}

// 进程级持久 scratch（按 device + H 缓存）。容量按最坏情况一次性分配，绝不再 realloc，
// 以免使已捕获的 graph 持有悬空指针。capturing 且尚未分配时返回 nullptr（调用方回退到非切分 kernel）。
static float *FastllmGetPagedSplitScratch(int device, int H, int maxBatch, int headDim,
                                          size_t &capacitySlots, bool capturing) {
    struct ScratchEntry {
        float *ptr = nullptr;
        size_t slots = 0;   // 槽位数，每槽 (headDim+2) 个 float
        int headDimPlus = 0;
    };
    static std::mutex mtx;
    static std::map<int64_t, ScratchEntry> cache;   // key = ((device<<20)|H)<<8 | (headDim)
    std::lock_guard<std::mutex> guard(mtx);
    int64_t key = (((int64_t)device << 20) | (int64_t)H) << 9 | (int64_t)headDim;
    auto &entry = cache[key];
    // 最坏槽位数：覆盖 batch*H*S 的最大值。
    //   - S=1（bh 很大）时 ≈ maxBatch*H；
    //   - S 受 MAX_SPLITS 限制（bh 很小）时 ≈ MAX_SPLITS*H；
    //   - bh*ceil(TARGET/bh) 的峰值可接近 2*TARGET。
    // 覆盖 GQA decode：S 可达 MAX_SPLITS，槽位按 batch*H*S 计。
    size_t worstSlots = (size_t)maxBatch * H * FASTLLM_PAGED_MAX_SPLITS;
    worstSlots = std::max(worstSlots, (size_t)2 * FASTLLM_PAGED_SPLIT_TARGET_BLOCKS);
    if (entry.ptr != nullptr && entry.slots >= worstSlots) {
        capacitySlots = entry.slots;
        return entry.ptr;
    }
    if (capturing) {
        // 捕获期间禁止分配；若已有（不足）缓冲则不可用，返回 nullptr 让调用方回退。
        capacitySlots = entry.ptr ? entry.slots : 0;
        return entry.ptr != nullptr && entry.slots >= worstSlots ? entry.ptr : nullptr;
    }
    // 非捕获：分配一次最坏容量。若之前分配过更小的（不应发生，因 worstSlots 固定），则保留旧的不动。
    if (entry.ptr == nullptr) {
        size_t bytes = worstSlots * (size_t)(headDim + 2) * sizeof(float);
        entry.ptr = (float*)FastllmCudaMalloc(bytes);
        entry.slots = worstSlots;
        entry.headDimPlus = headDim + 2;
    }
    capacitySlots = entry.slots;
    return entry.ptr;
}

// phase1：每个 block 处理 (b, h, split)，计算该 KV 段的部分在线 softmax 状态写入 scratch。
// DIMS_PER_LANE=4 覆盖 headDim<=128，DIMS_PER_LANE=8 覆盖 headDim<=256。
template <typename QType, typename KVType, int DIMS_PER_LANE>
__global__ void FastllmPagedAttentionSplitKernel(
    const QType *qd,
    const KVType *pagedK,
    const KVType *pagedV,
    float *scratch,               // [(b*H+h)*S + split] * (headDim+2)：[acc(headDim), M, L]
    const int32_t *qSizes,
    const int32_t *pageSizes,
    const int32_t *pageIndexs,
    const int32_t *lastPageLens,
    int H, int group, int numKvHeads, int headDim, int pageLen,
    int q_stride_h, int q_stride_n, float scale, int S) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int split = blockIdx.z;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;
    int numWarps = blockDim.x >> 5;

    int tokenStart = qSizes[b];
    int qoLen = qSizes[b + 1] - tokenStart;
    int pageStart = pageSizes[b];
    int numPages = pageSizes[b + 1] - pageStart;
    int kvLen = (numPages > 0) ? ((numPages - 1) * pageLen + lastPageLens[b]) : 0;

    int headDimPlus = headDim + 2;
    float *slot = scratch + ((size_t)(b * H + h) * S + split) * headDimPlus;

    const int HD_PAD = DIMS_PER_LANE * 32;
    __shared__ float sQ[HD_PAD];
    __shared__ float sM[8];
    __shared__ float sL[8];
    __shared__ float sAcc[8 * HD_PAD];

    // 计算该 split 的 KV 区间（仅解码：单 query token = tokenStart，可见全部 kvLen）。
    int chunk = (kvLen > 0) ? ((kvLen + S - 1) / S) : 0;
    int kvStart = split * chunk;
    int kvEnd = kvStart + chunk;
    if (kvEnd > kvLen) kvEnd = kvLen;

    if (qoLen <= 0 || numPages <= 0 || kvLen <= 0 || kvStart >= kvEnd) {
        // 空段：写中性值。
        for (int d = tid; d < headDim; d += blockDim.x) {
            slot[d] = 0.0f;
        }
        if (tid == 0) {
            slot[headDim] = -1e30f;   // M
            slot[headDim + 1] = 0.0f; // L
        }
        return;
    }

    int kvHead = h / group;
    size_t pageStride = (size_t)pageLen * numKvHeads * headDim;
    size_t tokenStride = (size_t)numKvHeads * headDim;
    size_t kvHeadOffset = (size_t)kvHead * headDim;
    int token = tokenStart;   // qoLen==1

    for (int d = tid; d < headDim; d += blockDim.x) {
        sQ[d] = FastllmAttentionValueToFloat<QType>(
            qd[(size_t)h * q_stride_h + (size_t)token * q_stride_n + d]);
    }
    __syncthreads();

    float m = -1e30f, l = 0.0f;
    float acc[DIMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; i++) {
        acc[i] = 0.0f;
    }
    for (int j = kvStart + warpId; j < kvEnd; j += numWarps) {
        size_t base = FastllmPagedKvTokenBase(j, pageStart, pageLen, numPages, pageIndexs,
                                              pageStride, tokenStride, kvHeadOffset);

        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; i++) {
            int d = lane + (i << 5);
            if (d < headDim) {
                partial += sQ[d] * FastllmKvLoadFloat<KVType>(pagedK + base, d);
            }
        }
        float score = FastllmWarpAllReduceSum(partial) * scale;
        float newM = fmaxf(m, score);
        float corr = __expf(m - newM);
        float p = __expf(score - newM);
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; i++) {
            int d = lane + (i << 5);
            if (d < headDim) {
                acc[i] = acc[i] * corr + p * FastllmKvLoadFloat<KVType>(pagedV + base, d);
            }
        }
        l = l * corr + p;
        m = newM;
    }

    if (lane == 0) {
        sM[warpId] = m;
        sL[warpId] = l;
    }
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; i++) {
        int d = lane + (i << 5);
        if (d < headDim) {
            sAcc[warpId * headDim + d] = acc[i];
        }
    }
    __syncthreads();

    float M = -1e30f;
    for (int w = 0; w < numWarps; w++) {
        M = fmaxf(M, sM[w]);
    }
    float L = 0.0f;
    for (int w = 0; w < numWarps; w++) {
        L += sL[w] * __expf(sM[w] - M);
    }
    // 写该 split 的未归一化 acc（相对 M）与 (M, L)。
    for (int d = tid; d < headDim; d += blockDim.x) {
        float o = 0.0f;
        for (int w = 0; w < numWarps; w++) {
            o += sAcc[w * headDim + d] * __expf(sM[w] - M);
        }
        slot[d] = o;
    }
    if (tid == 0) {
        slot[headDim] = M;
        slot[headDim + 1] = L;
    }
}

// phase1（GQA 共享版，直读流式）：grid = (batch, numKvHeads, S)，blockDim = 256（8 warp）。
// 8 个 warp 跨 key 条带分工，每个 warp 对它负责的每个 key 把 K/V 直接从显存读一次到寄存器
//（与 per-head 版本一样保持高带宽、load/compute 自然重叠），随后复用给组内 group 个 query head，
// 各自维护在线 softmax。HBM 上 KV 流量降为 per-head 版本的 1/group。
// 连续物理页（pageIndex 递增 1）时跳过逐 key 查表。
__device__ __forceinline__ bool FastllmPagedIsLinearPageLayout(
    int pageStart, int numPages, const int32_t *pageIndexs) {
    if (pageStart != 0 || numPages <= 0 || pageIndexs[pageStart] != 0) {
        return false;
    }
    for (int i = 1; i < numPages; i++) {
        if (pageIndexs[pageStart + i] != pageIndexs[pageStart + i - 1] + 1) {
            return false;
        }
    }
    return true;
}

// GQA decode 内层：对 [kvStart,kvEnd) 做在线 softmax，K/V 每个 key 只读一次。
//
// 性能要点：原实现对 group 个 query head 串行做点积，每个 head 的 warp 归约
// （5 次依赖 __shfl）都落在关键路径上，再夹着 exp/累加，导致每个 key 有
// group*5 次完全串行的 shuffle —— 这是 GQA 内核比 per-Q-head 内核还慢的根因。
// 这里改为：先把 group 个点积的 lane 局部和算到寄存器，再「按 offset 交错」对
// group 个值同时做 warp 归约（彼此独立可流水），最后统一做在线 softmax 更新。
// 这样把关键路径从 group 段 5-shuffle 链压缩为 1 段 5-shuffle 链。
template <typename KVType, int GROUP_MAX>
__device__ __forceinline__ void FastllmPagedGqaAttnKvRange(
    bool linearKv,
    int kvStart, int kvEnd,
    int warpId, int numWarps, int lane, int headDim,
    const KVType *pagedK, const KVType *pagedV,
    int pageStart, int pageLen, int numPages, const int32_t *pageIndexs,
    size_t pageStride, size_t tokenStride, size_t kvHeadOffset,
    const float *sQ, int group, float scale,
    float *m, float *l, float *acc) {
    int d0 = lane << 2;   // 每个 lane 负责 4 个「连续」维度 [4*lane, 4*lane+4)，使整 warp 读取完全合并。
    for (int j = kvStart + warpId; j < kvEnd; j += numWarps) {
        size_t base = linearKv
            ? ((size_t)j * tokenStride + kvHeadOffset)
            : FastllmPagedKvTokenBase(j, pageStart, pageLen, numPages, pageIndexs,
                                      pageStride, tokenStride, kvHeadOffset);
        float kreg[4], vreg[4];
        FastllmKvLoad4Contig<KVType>(pagedK + base, d0, headDim, kreg);
        FastllmKvLoad4Contig<KVType>(pagedV + base, d0, headDim, vreg);

        // 1) 各 group 的 lane 局部点积（彼此独立）。
        float partial[GROUP_MAX];
        #pragma unroll
        for (int g = 0; g < GROUP_MAX; g++) {
            float p = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int d = d0 + i;
                if (d < headDim) {
                    p += sQ[g * headDim + d] * kreg[i];
                }
            }
            partial[g] = p;
        }

        // 2) 交错 warp 归约：同一 offset 下对 GROUP_MAX 个值各做一次 shuffle，
        //    彼此无依赖，硬件可流水，关键路径只剩 5 次 shuffle 延迟。
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int g = 0; g < GROUP_MAX; g++) {
                partial[g] += __shfl_xor_sync(0xffffffffu, partial[g], offset);
            }
        }

        // 3) 在线 softmax 更新（各 group 独立）。
        // SM70 优化：在 log2 域用 exp2f（少一次乘法），并对 acc 的 rescale 加 warp-uniform 条件分支
        //（partial[g] 经全 warp 归约后各 lane 相同，分支不会 divergence）。最大值不变时跳过 corr-exp 与
        // acc rescale —— decode 时最大值很快稳定，绝大多数 key 走这条便宜路径。m[g]/sM 存的是 log2 域最大值。
        const float kLog2e = 1.4426950408889634f;
        float scaleLog2 = scale * kLog2e;
        #pragma unroll
        for (int g = 0; g < GROUP_MAX; g++) {
            float s2 = partial[g] * scaleLog2;
            if (s2 > m[g]) {
                float corr = exp2f(m[g] - s2);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    acc[g * 4 + i] *= corr;
                }
                l[g] *= corr;
                m[g] = s2;
            }
            float p = exp2f(s2 - m[g]);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc[g * 4 + i] += p * vreg[i];
            }
            l[g] += p;
        }
    }
}

// SM70(V100) 优化：默认编译该内核用到 ~91 寄存器，被寄存器数限制到每 SM 仅 2 个 block(25% 占用率)，
// 单 SM 在途访存请求不足以掩盖 HBM 延迟（实测带宽仅 ~31% 峰值）。用 __launch_bounds__ 强约束寄存器，
// 把每 SM 驻留 block 数提到 5（48 寄存器/线程、无 spill，受 18.7KB 共享内存约束 96KB/18.7KB≈5），
// 占用率 25% -> 62.5%，显著提升访存级并行与有效带宽（实测 split kernel 196us -> 176us，整体 attn 9.8ms -> ~7.5ms）。
#define FASTLLM_PAGED_GQA_MIN_BLOCKS_PER_SM 5
// 通过把切分段数 S 按 numKvHeads 选取，保持足够 SM 并行且控制 combine 段数。GROUP_MAX 为编译期上界。
template <typename QType, typename KVType, int GROUP_MAX>
__global__ void __launch_bounds__(256, FASTLLM_PAGED_GQA_MIN_BLOCKS_PER_SM)
FastllmPagedAttentionSplitGQAKernel(
    const QType *qd,
    const KVType *pagedK,
    const KVType *pagedV,
    float *scratch,
    const int32_t *qSizes,
    const int32_t *pageSizes,
    const int32_t *pageIndexs,
    const int32_t *lastPageLens,
    int H, int group, int numKvHeads, int headDim, int pageLen,
    int q_stride_h, int q_stride_n, float scale, int S) {
    int b = blockIdx.x;
    int kvh = blockIdx.y;
    int split = blockIdx.z;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;
    int numWarps = blockDim.x >> 5;   // 8
    int headDimPlus = headDim + 2;

    int tokenStart = qSizes[b];
    int qoLen = qSizes[b + 1] - tokenStart;
    int pageStart = pageSizes[b];
    int numPages = pageSizes[b + 1] - pageStart;
    int kvLen = (numPages > 0) ? ((numPages - 1) * pageLen + lastPageLens[b]) : 0;

    __shared__ float sQ[GROUP_MAX * 128];
    __shared__ float sM[GROUP_MAX * 8];
    __shared__ float sL[GROUP_MAX * 8];
    __shared__ float sAcc[GROUP_MAX * 8 * 128];

    int chunk = (kvLen > 0) ? ((kvLen + S - 1) / S) : 0;
    int kvStart = split * chunk;
    int kvEnd = kvStart + chunk;
    if (kvEnd > kvLen) kvEnd = kvLen;

    if (qoLen <= 0 || numPages <= 0 || kvLen <= 0 || kvStart >= kvEnd) {
        for (int g = 0; g < group; g++) {
            int h = kvh * group + g;
            float *slot = scratch + ((size_t)(b * H + h) * S + split) * headDimPlus;
            for (int d = tid; d < headDim; d += blockDim.x) {
                slot[d] = 0.0f;
            }
            if (tid == 0) {
                slot[headDim] = -1e30f;
                slot[headDim + 1] = 0.0f;
            }
        }
        return;
    }

    int token = tokenStart;   // qoLen==1
    for (int idx = tid; idx < group * headDim; idx += blockDim.x) {
        int gg = idx / headDim;
        int dd = idx - gg * headDim;
        int hh = kvh * group + gg;
        sQ[idx] = FastllmAttentionValueToFloat<QType>(
            qd[(size_t)hh * q_stride_h + (size_t)token * q_stride_n + dd]);
    }
    __syncthreads();

    size_t pageStride = (size_t)pageLen * numKvHeads * headDim;
    size_t tokenStride = (size_t)numKvHeads * headDim;
    size_t kvHeadOffset = (size_t)kvh * headDim;
    bool linearKv = FastllmPagedIsLinearPageLayout(pageStart, numPages, pageIndexs);

    float m[GROUP_MAX], l[GROUP_MAX];
    float acc[GROUP_MAX * 4];
    #pragma unroll
    for (int g = 0; g < GROUP_MAX; g++) {
        m[g] = -1e30f;
        l[g] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < GROUP_MAX * 4; i++) {
        acc[i] = 0.0f;
    }

    FastllmPagedGqaAttnKvRange<KVType, GROUP_MAX>(
        linearKv, kvStart, kvEnd, warpId, numWarps, lane, headDim,
        pagedK, pagedV, pageStart, pageLen, numPages, pageIndexs,
        pageStride, tokenStride, kvHeadOffset, sQ, group, scale, m, l, acc);

    for (int g = 0; g < group; g++) {
        if (lane == 0) {
            sM[g * numWarps + warpId] = m[g];
            sL[g * numWarps + warpId] = l[g];
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int d = (lane << 2) + i;
            if (d < headDim) {
                sAcc[(g * numWarps + warpId) * headDim + d] = acc[g * 4 + i];
            }
        }
    }
    __syncthreads();

    for (int g = 0; g < group; g++) {
        int h = kvh * group + g;
        float *slot = scratch + ((size_t)(b * H + h) * S + split) * headDimPlus;
        float M = -1e30f;
        for (int w = 0; w < numWarps; w++) {
            M = fmaxf(M, sM[g * numWarps + w]);
        }
        float L = 0.0f;
        for (int w = 0; w < numWarps; w++) {
            L += sL[g * numWarps + w] * exp2f(sM[g * numWarps + w] - M);
        }
        for (int d = tid; d < headDim; d += blockDim.x) {
            float o = 0.0f;
            for (int w = 0; w < numWarps; w++) {
                o += sAcc[(g * numWarps + w) * headDim + d] * exp2f(sM[g * numWarps + w] - M);
            }
            slot[d] = o;
        }
        if (tid == 0) {
            slot[headDim] = M;
            slot[headDim + 1] = L;
        }
    }
}

// 解码 GQA 非 split：grid = (batch, numKvHeads)，一次扫完整 KV，无 scratch/combine。
// 同样用 __launch_bounds__ 提高 V100 占用率（短上下文 decode 走此路径）。
template <typename QType, typename KVType, int GROUP_MAX>
__global__ void __launch_bounds__(256, FASTLLM_PAGED_GQA_MIN_BLOCKS_PER_SM)
FastllmPagedAttentionBatchGQAKernel(
    const QType *qd,
    const KVType *pagedK,
    const KVType *pagedV,
    QType *od,
    const int32_t *qSizes,
    const int32_t *pageSizes,
    const int32_t *pageIndexs,
    const int32_t *lastPageLens,
    int H, int group, int numKvHeads, int headDim, int pageLen,
    int q_stride_h, int q_stride_n, float scale) {
    int b = blockIdx.x;
    int kvh = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;
    int numWarps = blockDim.x >> 5;

    int tokenStart = qSizes[b];
    int qoLen = qSizes[b + 1] - tokenStart;
    int pageStart = pageSizes[b];
    int numPages = pageSizes[b + 1] - pageStart;
    int kvLen = (numPages > 0) ? ((numPages - 1) * pageLen + lastPageLens[b]) : 0;

    __shared__ float sQ[GROUP_MAX * 128];
    __shared__ float sM[GROUP_MAX * 8];
    __shared__ float sL[GROUP_MAX * 8];
    __shared__ float sAcc[GROUP_MAX * 8 * 128];

    if (qoLen <= 0 || numPages <= 0 || kvLen <= 0) {
        return;
    }

    int token = tokenStart;
    for (int idx = tid; idx < group * headDim; idx += blockDim.x) {
        int gg = idx / headDim;
        int dd = idx - gg * headDim;
        int hh = kvh * group + gg;
        sQ[idx] = FastllmAttentionValueToFloat<QType>(
            qd[(size_t)hh * q_stride_h + (size_t)token * q_stride_n + dd]);
    }
    __syncthreads();

    size_t pageStride = (size_t)pageLen * numKvHeads * headDim;
    size_t tokenStride = (size_t)numKvHeads * headDim;
    size_t kvHeadOffset = (size_t)kvh * headDim;
    bool linearKv = FastllmPagedIsLinearPageLayout(pageStart, numPages, pageIndexs);

    float m[GROUP_MAX], l[GROUP_MAX];
    float acc[GROUP_MAX * 4];
    #pragma unroll
    for (int g = 0; g < GROUP_MAX; g++) {
        m[g] = -1e30f;
        l[g] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < GROUP_MAX * 4; i++) {
        acc[i] = 0.0f;
    }

    FastllmPagedGqaAttnKvRange<KVType, GROUP_MAX>(
        linearKv, 0, kvLen, warpId, numWarps, lane, headDim,
        pagedK, pagedV, pageStart, pageLen, numPages, pageIndexs,
        pageStride, tokenStride, kvHeadOffset, sQ, group, scale, m, l, acc);

    for (int g = 0; g < group; g++) {
        if (lane == 0) {
            sM[g * numWarps + warpId] = m[g];
            sL[g * numWarps + warpId] = l[g];
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int d = (lane << 2) + i;
            if (d < headDim) {
                sAcc[(g * numWarps + warpId) * headDim + d] = acc[g * 4 + i];
            }
        }
    }
    __syncthreads();

    for (int g = 0; g < group; g++) {
        int h = kvh * group + g;
        float M = -1e30f;
        for (int w = 0; w < numWarps; w++) {
            M = fmaxf(M, sM[g * numWarps + w]);
        }
        float L = 0.0f;
        for (int w = 0; w < numWarps; w++) {
            L += sL[g * numWarps + w] * exp2f(sM[g * numWarps + w] - M);
        }
        for (int d = tid; d < headDim; d += blockDim.x) {
            float o = 0.0f;
            for (int w = 0; w < numWarps; w++) {
                o += sAcc[(g * numWarps + w) * headDim + d] * exp2f(sM[g * numWarps + w] - M);
            }
            o = (L > 0.0f) ? (o / L) : 0.0f;
            od[(size_t)token * H * headDim + (size_t)h * headDim + d] =
                FastllmAttentionFloatToValue<QType>(o);
        }
    }
}

// phase2：合并某个 (b, h) 的 S 段部分状态，写出最终 output（token-major）。
template <typename QType>
__global__ void FastllmPagedAttentionCombineKernel(
    const float *scratch,
    QType *od,
    const int32_t *qSizes,
    int H, int headDim, int S) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int token = qSizes[b];   // 解码：token-major 行号 = tokenStart

    int headDimPlus = headDim + 2;
    const float *base = scratch + ((size_t)(b * H + h) * S) * headDimPlus;

    __shared__ float sMs[FASTLLM_PAGED_MAX_SPLITS];
    __shared__ float sLs[FASTLLM_PAGED_MAX_SPLITS];
    for (int s = tid; s < S; s += blockDim.x) {
        sMs[s] = base[(size_t)s * headDimPlus + headDim];
        sLs[s] = base[(size_t)s * headDimPlus + headDim + 1];
    }
    __syncthreads();

    float M = -1e30f;
    for (int s = 0; s < S; s++) {
        M = fmaxf(M, sMs[s]);
    }
    float L = 0.0f;
    for (int s = 0; s < S; s++) {
        L += sLs[s] * __expf(sMs[s] - M);
    }
    for (int d = tid; d < headDim; d += blockDim.x) {
        float o = 0.0f;
        for (int s = 0; s < S; s++) {
            o += base[(size_t)s * headDimPlus + d] * __expf(sMs[s] - M);
        }
        o = (L > 0.0f) ? (o / L) : 0.0f;
        od[(size_t)token * H * headDim + (size_t)h * headDim + d] =
            FastllmAttentionFloatToValue<QType>(o);
    }
}

// phase2（GQA）：每个 block 合并一个 kv head 下 group 个 Q head 的 S 段，launch 数 H/group。
template <typename QType, int GROUP_MAX>
__global__ void FastllmPagedAttentionCombineGQAKernel(
    const float *scratch,
    QType *od,
    const int32_t *qSizes,
    int H, int group, int headDim, int S) {
    int b = blockIdx.x;
    int kvh = blockIdx.y;
    int tid = threadIdx.x;
    int token = qSizes[b];
    int headDimPlus = headDim + 2;

    __shared__ float sMs[FASTLLM_PAGED_MAX_SPLITS];
    __shared__ float sLs[FASTLLM_PAGED_MAX_SPLITS];

    for (int g = 0; g < group; g++) {
        int h = kvh * group + g;
        const float *base = scratch + ((size_t)(b * H + h) * S) * headDimPlus;
        for (int s = tid; s < S; s += blockDim.x) {
            sMs[s] = base[(size_t)s * headDimPlus + headDim];
            sLs[s] = base[(size_t)s * headDimPlus + headDim + 1];
        }
        __syncthreads();

        float M = -1e30f;
        for (int s = 0; s < S; s++) {
            M = fmaxf(M, sMs[s]);
        }
        float L = 0.0f;
        for (int s = 0; s < S; s++) {
            L += sLs[s] * exp2f(sMs[s] - M);
        }
        for (int d = tid; d < headDim; d += blockDim.x) {
            float o = 0.0f;
            for (int s = 0; s < S; s++) {
                o += base[(size_t)s * headDimPlus + d] * exp2f(sMs[s] - M);
            }
            o = (L > 0.0f) ? (o / L) : 0.0f;
            od[(size_t)token * H * headDim + (size_t)h * headDim + d] =
                FastllmAttentionFloatToValue<QType>(o);
        }
        __syncthreads();
    }
}

template <typename QType>
static void FastllmCudaPagedAttentionBatchGqaLaunch(
    fastllm::Data &q, fastllm::Data &output,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int32_t *qSizesData, int32_t *pageSizesData, int32_t *pageIndexsData, int32_t *lastPageLensData,
    uint32_t batch_size, int H, int group, int numKvHeads, int headDim, int pageLen,
    int q_stride_h, int q_stride_n, float scale) {
    const unsigned int kBlockThreads = 256;
    dim3 grid(batch_size, (unsigned int)numKvHeads, 1);
    dim3 block(kBlockThreads, 1, 1);
    QType *qd = (QType*)q.cudaData;
    QType *od = (QType*)output.cudaData;
    if (pagedKVCacheK->dataType == fastllm::DataType::FP8_E4M3) {
        if (group == 2) {
            FastllmPagedAttentionBatchGQAKernel<QType, __nv_fp8_e4m3, 2><<<grid, block>>>(
                qd, (__nv_fp8_e4m3*)pagedKVCacheK->cudaData, (__nv_fp8_e4m3*)pagedKVCacheV->cudaData, od,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
        } else if (group == 3) {
            FastllmPagedAttentionBatchGQAKernel<QType, __nv_fp8_e4m3, 3><<<grid, block>>>(
                qd, (__nv_fp8_e4m3*)pagedKVCacheK->cudaData, (__nv_fp8_e4m3*)pagedKVCacheV->cudaData, od,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
        } else {
            FastllmPagedAttentionBatchGQAKernel<QType, __nv_fp8_e4m3, 4><<<grid, block>>>(
                qd, (__nv_fp8_e4m3*)pagedKVCacheK->cudaData, (__nv_fp8_e4m3*)pagedKVCacheV->cudaData, od,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
        }
    } else if (pagedKVCacheK->dataType == fastllm::DataType::BFLOAT16) {
        if (group == 2) {
            FastllmPagedAttentionBatchGQAKernel<QType, __nv_bfloat16, 2><<<grid, block>>>(
                qd, (__nv_bfloat16*)pagedKVCacheK->cudaData, (__nv_bfloat16*)pagedKVCacheV->cudaData, od,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
        } else if (group == 3) {
            FastllmPagedAttentionBatchGQAKernel<QType, __nv_bfloat16, 3><<<grid, block>>>(
                qd, (__nv_bfloat16*)pagedKVCacheK->cudaData, (__nv_bfloat16*)pagedKVCacheV->cudaData, od,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
        } else {
            FastllmPagedAttentionBatchGQAKernel<QType, __nv_bfloat16, 4><<<grid, block>>>(
                qd, (__nv_bfloat16*)pagedKVCacheK->cudaData, (__nv_bfloat16*)pagedKVCacheV->cudaData, od,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
        }
    } else {
        if (group == 2) {
            FastllmPagedAttentionBatchGQAKernel<QType, half, 2><<<grid, block>>>(
                qd, (half*)pagedKVCacheK->cudaData, (half*)pagedKVCacheV->cudaData, od,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
        } else if (group == 3) {
            FastllmPagedAttentionBatchGQAKernel<QType, half, 3><<<grid, block>>>(
                qd, (half*)pagedKVCacheK->cudaData, (half*)pagedKVCacheV->cudaData, od,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
        } else {
            FastllmPagedAttentionBatchGQAKernel<QType, half, 4><<<grid, block>>>(
                qd, (half*)pagedKVCacheK->cudaData, (half*)pagedKVCacheV->cudaData, od,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
        }
    }
}

template <typename QType, typename KVType>
static void FastllmCudaPagedAttentionSplitLaunchGqa(
    QType *qd, KVType *pagedK, KVType *pagedV, float *scratch, QType *od,
    int32_t *qSizesData, int32_t *pageSizesData, int32_t *pageIndexsData, int32_t *lastPageLensData,
    uint32_t batch_size, int H, int group, int numKvHeads, int headDim, int pageLen,
    int q_stride_h, int q_stride_n, float scale, int S, dim3 block1) {
    dim3 grid1(batch_size, (unsigned int)numKvHeads, (unsigned int)S);
    if (group == 2) {
        FastllmPagedAttentionSplitGQAKernel<QType, KVType, 2><<<grid1, block1>>>(
            qd, pagedK, pagedV, scratch, qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
    } else if (group == 3) {
        FastllmPagedAttentionSplitGQAKernel<QType, KVType, 3><<<grid1, block1>>>(
            qd, pagedK, pagedV, scratch, qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
    } else {
        FastllmPagedAttentionSplitGQAKernel<QType, KVType, 4><<<grid1, block1>>>(
            qd, pagedK, pagedV, scratch, qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
    }
}

template <typename QType, int DIMS_PER_LANE>
static void FastllmPagedAttentionSplitKernelDispatchKV(
    dim3 grid1, dim3 block1, QType *qd, float *scratch,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int32_t *qSizesData, int32_t *pageSizesData, int32_t *pageIndexsData, int32_t *lastPageLensData,
    int H, int group, int numKvHeads, int headDim, int pageLen,
    int q_stride_h, int q_stride_n, float scale, int S) {
    if (pagedKVCacheK->dataType == fastllm::DataType::FP8_E4M3) {
        FastllmPagedAttentionSplitKernel<QType, __nv_fp8_e4m3, DIMS_PER_LANE><<<grid1, block1>>>(
            qd, (__nv_fp8_e4m3*)pagedKVCacheK->cudaData, (__nv_fp8_e4m3*)pagedKVCacheV->cudaData, scratch,
            qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
    } else if (pagedKVCacheK->dataType == fastllm::DataType::BFLOAT16) {
        FastllmPagedAttentionSplitKernel<QType, __nv_bfloat16, DIMS_PER_LANE><<<grid1, block1>>>(
            qd, (__nv_bfloat16*)pagedKVCacheK->cudaData, (__nv_bfloat16*)pagedKVCacheV->cudaData, scratch,
            qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
    } else {
        FastllmPagedAttentionSplitKernel<QType, half, DIMS_PER_LANE><<<grid1, block1>>>(
            qd, (half*)pagedKVCacheK->cudaData, (half*)pagedKVCacheV->cudaData, scratch,
            qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
    }
}

template <typename QType>
static void FastllmCudaPagedAttentionSplitLaunch(
    fastllm::Data &q, fastllm::Data &output, float *scratch,
    fastllm::Data *pagedKVCacheK, fastllm::Data *pagedKVCacheV,
    int32_t *qSizesData, int32_t *pageSizesData, int32_t *pageIndexsData, int32_t *lastPageLensData,
    uint32_t batch_size, int H, int group, int numKvHeads, int headDim, int pageLen,
    int q_stride_h, int q_stride_n, float scale, int S) {
    const unsigned int kBlockThreads = 256;   // 8 warps
    dim3 block1(kBlockThreads, 1, 1);
    QType *qd = (QType*)q.cudaData;
    QType *od = (QType*)output.cudaData;
    const bool useGqa = FastllmPagedUseGqaDecodeFor(group, headDim, H, numKvHeads);

    if (useGqa) {
        if (pagedKVCacheK->dataType == fastllm::DataType::FP8_E4M3) {
            FastllmCudaPagedAttentionSplitLaunchGqa<QType, __nv_fp8_e4m3>(
                qd, (__nv_fp8_e4m3*)pagedKVCacheK->cudaData, (__nv_fp8_e4m3*)pagedKVCacheV->cudaData,
                scratch, od, qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                batch_size, H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S, block1);
        } else if (pagedKVCacheK->dataType == fastllm::DataType::BFLOAT16) {
            FastllmCudaPagedAttentionSplitLaunchGqa<QType, __nv_bfloat16>(
                qd, (__nv_bfloat16*)pagedKVCacheK->cudaData, (__nv_bfloat16*)pagedKVCacheV->cudaData,
                scratch, od, qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                batch_size, H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S, block1);
        } else {
            FastllmCudaPagedAttentionSplitLaunchGqa<QType, half>(
                qd, (half*)pagedKVCacheK->cudaData, (half*)pagedKVCacheV->cudaData,
                scratch, od, qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                batch_size, H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S, block1);
        }
    } else {
        dim3 grid1(batch_size, (unsigned int)H, (unsigned int)S);
        if (headDim <= 128) {
            FastllmPagedAttentionSplitKernelDispatchKV<QType, 4>(
                grid1, block1, qd, scratch, pagedKVCacheK, pagedKVCacheV,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
        } else {
            FastllmPagedAttentionSplitKernelDispatchKV<QType, 8>(
                grid1, block1, qd, scratch, pagedKVCacheK, pagedKVCacheV,
                qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
        }
    }
    dim3 grid2(batch_size, (unsigned int)H, 1);
    dim3 block2((unsigned int)headDim, 1, 1);
    if (useGqa) {
        grid2.y = (unsigned int)numKvHeads;
        if (group == 2) {
            FastllmPagedAttentionCombineGQAKernel<QType, 2><<<grid2, block2>>>(
                scratch, od, qSizesData, H, group, headDim, S);
        } else if (group == 3) {
            FastllmPagedAttentionCombineGQAKernel<QType, 3><<<grid2, block2>>>(
                scratch, od, qSizesData, H, group, headDim, S);
        } else {
            FastllmPagedAttentionCombineGQAKernel<QType, 4><<<grid2, block2>>>(
                scratch, od, qSizesData, H, group, headDim, S);
        }
    } else {
        FastllmPagedAttentionCombineKernel<QType><<<grid2, block2>>>(
            scratch, od, qSizesData, H, headDim, S);
    }
}

// 使用可捕获 kernel 计算批量分页注意力。读取 device 端元数据，输出 token-major [totalTokens, H, headDim]。
static bool FastllmCudaHalfPagedAttentionBatchCapturable(
    fastllm::Data &q,
    fastllm::Data &kCaches,
    fastllm::Data &vCaches,
    fastllm::Data &qSizes,
    fastllm::Data &pageSizes,
    fastllm::Data &pageIndexs,
    fastllm::Data &lastPageLens,
    fastllm::Data &output,
    int group,
    float scale) {
    uint32_t batch_size = qSizes.dims[0] - 1;
    int numKvHeads = kCaches.dims[0];
    int H = group * numKvHeads;
    int q2 = q.dims[2];
    fastllm::Data *pagedKVCacheK = kCaches.pagedKVCacheData;
    fastllm::Data *pagedKVCacheV = vCaches.pagedKVCacheData;
    int pageLen = kCaches.pageLen;
    int headDim = pagedKVCacheK->dims[3];
    int totalTokens = q.dims[1];

    int q_stride_h = q.strides.size() >= 1 ? (int)q.strides[0] : (totalTokens * q2);
    int q_stride_n = q.strides.size() >= 2 ? (int)q.strides[1] : q2;

    // 解码（每 batch 1 个 query，totalTokens == batch_size）时优先用 split-KV，提升 SM 利用率，
    // 使长上下文下解码基本不掉速。
    bool isDecode = (totalTokens == (int)batch_size);
    int32_t *qSizesData = (int32_t*)qSizes.cudaData;
    int32_t *pageSizesData = (int32_t*)pageSizes.cudaData;
    int32_t *pageIndexsData = (int32_t*)pageIndexs.cudaData;
    int32_t *lastPageLensData = (int32_t*)lastPageLens.cudaData;

    if (!FastllmPagedForceNoSplit() && isDecode && headDim > 0 && headDim <= 256) {
        bool capturing = false;
        cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus) == cudaSuccess &&
            captureStatus != cudaStreamCaptureStatusNone) {
            capturing = true;
        } else {
            cudaGetLastError();
        }
        int device = -1;
        cudaGetDevice(&device);
        const int kMaxDecodeBatch = 32;   // 与 qwen3 的 maxCudaGraphDecodeBatch 一致
        size_t capacitySlots = 0;
        float *scratch = FastllmGetPagedSplitScratch(device, H, kMaxDecodeBatch, headDim,
                                                     capacitySlots, capturing);
        const bool useGqa = FastllmPagedUseGqaDecodeFor(group, headDim, H, numKvHeads);
        int S = useGqa ? FastllmChoosePagedSplits(batch_size, numKvHeads, true)
                       : FastllmChoosePagedSplits(batch_size, H, false);
        if (useGqa && S == 1) {
            if (q.dataType == fastllm::DataType::BFLOAT16) {
                FastllmCudaPagedAttentionBatchGqaLaunch<__nv_bfloat16>(
                    q, output, pagedKVCacheK, pagedKVCacheV,
                    qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                    batch_size, H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
            } else {
                FastllmCudaPagedAttentionBatchGqaLaunch<half>(
                    q, output, pagedKVCacheK, pagedKVCacheV,
                    qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                    batch_size, H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
            }
            output.Resize({totalTokens, H, q2});
            return true;
        }
        if (scratch != nullptr && capacitySlots >= (size_t)batch_size * H * S && S > 1) {
            if (q.dataType == fastllm::DataType::BFLOAT16) {
                FastllmCudaPagedAttentionSplitLaunch<__nv_bfloat16>(
                    q, output, scratch, pagedKVCacheK, pagedKVCacheV,
                    qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                    batch_size, H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
            } else {
                FastllmCudaPagedAttentionSplitLaunch<half>(
                    q, output, scratch, pagedKVCacheK, pagedKVCacheV,
                    qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
                    batch_size, H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale, S);
            }
            output.Resize({totalTokens, H, q2});
            return true;
        }
        // 否则落到下面的非切分 per-Q-head kernel（S==1 非 GQA 或 scratch 不可用）。
    }

    bool ok;
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        ok = FastllmCudaPagedAttentionBatchKernelLaunch<__nv_bfloat16>(
            q, output, pagedKVCacheK, pagedKVCacheV,
            qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            batch_size, H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
    } else {
        ok = FastllmCudaPagedAttentionBatchKernelLaunch<half>(
            q, output, pagedKVCacheK, pagedKVCacheV,
            qSizesData, pageSizesData, pageIndexsData, lastPageLensData,
            batch_size, H, group, numKvHeads, headDim, pageLen, q_stride_h, q_stride_n, scale);
    }
    if (ok) {
        // 输出物理布局已是 token-major [totalTokens, H, headDim]，仅更新 dims。
        output.Resize({totalTokens, H, q2});
    }
    return ok;
}

bool FastllmCudaHalfPagedAttentionBatchFastllmFallback(
    fastllm::Data &q,
    fastllm::Data &kCaches,
    fastllm::Data &vCaches,
    fastllm::Data &qSizes,
    fastllm::Data &pageSizes,
    fastllm::Data &pageIndexs,
    fastllm::Data &lastPageLens,
    fastllm::Data &output,
    int group,
    float scale) {
    uint32_t batch_size = qSizes.dims[0] - 1;
    if (batch_size == 0) {
        return false;
    }

    // 路由：
    //  - 解码阶段（每个 batch 仅 1 个 query，totalTokens == batch_size）或正处于 CUDA graph 流捕获时，
    //    使用完全从 device 元数据驱动的可捕获 kernel（无同步拷贝 / malloc / free）。这样 decode 的
    //    CUDA graph 捕获与重放才能正确工作（FlashInfer 不支持的 GPU 上）。
    //  - 否则（前缀填充且未在捕获）走基于 cublas 的收集路径，性能更好（前缀填充不会被 graph 捕获）。
    bool capturing = false;
    {
        cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus) == cudaSuccess &&
            captureStatus != cudaStreamCaptureStatusNone) {
            capturing = true;
        } else {
            cudaGetLastError();
        }
    }
    bool isDecode = (q.dims.size() >= 2 && (int)q.dims[1] == (int)batch_size);
    if ((capturing || isDecode) &&
        kCaches.pagedKVCacheData != nullptr && vCaches.pagedKVCacheData != nullptr &&
        (q.dataType == fastllm::DataType::FLOAT16 || q.dataType == fastllm::DataType::BFLOAT16)) {
        return FastllmCudaHalfPagedAttentionBatchCapturable(
            q, kCaches, vCaches, qSizes, pageSizes, pageIndexs, lastPageLens, output, group, scale);
    }

    if (batch_size == 1) {
        return FastllmCudaHalfPagedAttentionFastllmFallback(q, kCaches, vCaches, output, group, scale);
    }

    int q2 = q.dims[2];
    int numKvHeads = kCaches.dims[0];
    uint32_t num_qo_heads_per_batch = (uint32_t)group * (uint32_t)numKvHeads;
    if (num_qo_heads_per_batch == 0) {
        printf("FastllmCudaHalfPagedAttentionBatchFastllmFallback: invalid head layout.\n");
        return false;
    }
    fastllm::Data *pagedKVCacheK = kCaches.pagedKVCacheData;
    fastllm::Data *pagedKVCacheV = vCaches.pagedKVCacheData;
    if (pagedKVCacheK == nullptr || pagedKVCacheV == nullptr) {
        printf("FastllmCudaHalfPagedAttentionBatchFastllmFallback: pagedKVCacheData is nullptr.\n");
        exit(0);
    }
    int pageLen = kCaches.pageLen;
    int headDim = pagedKVCacheK->dims[3];

    if (q.dataType != fastllm::DataType::FLOAT16) {
        printf("FastllmCudaHalfPagedAttentionBatchFastllmFallback: only FLOAT16 query is supported.\n");
        return false;
    }

    // 分页元数据（qSizes/pageSizes/pageIndexs/lastPageLens）在 decode 阶段可能只在 GPU 上更新
    // （见 fillLastPageLensOnDevice 等机制），host 端的 cpuIntDatas 可能是过期值。
    // 这里统一从 GPU 拷贝最新值，与 FlashInfer 路径读取的缓冲保持一致。
    auto loadHostInts = [](fastllm::Data &data, int count, const std::vector<int> &fallbackHost) -> std::vector<int32_t> {
        std::vector<int32_t> host((size_t)std::max(count, 0));
        if (count <= 0) {
            return host;
        }
        if (data.dataDevice == fastllm::DataDevice::CUDA && data.cudaData != nullptr) {
            FastllmCudaCopyFromDeviceToHost(host.data(), data.cudaData, (size_t)count * sizeof(int32_t));
        } else {
            for (int i = 0; i < count && i < (int)fallbackHost.size(); i++) {
                host[i] = fallbackHost[i];
            }
        }
        return host;
    };

    std::vector<int32_t> qSizesHost = loadHostInts(qSizes, (int)batch_size + 1, qSizes.cpuIntDatas);
    std::vector<int32_t> pageSizesHost = loadHostInts(pageSizes, (int)batch_size + 1, pageSizes.cpuIntDatas);
    std::vector<int32_t> lastPageLensHost = loadHostInts(lastPageLens, (int)batch_size, lastPageLens.cpuIntDatas);
    int totalPages = pageSizesHost.empty() ? 0 : pageSizesHost[batch_size];
    std::vector<int32_t> pageIndexsHost = loadHostInts(pageIndexs, totalPages, pageIndexs.cpuIntDatas);

    // q 的物理布局与 FlashInfer 一致：head-major [num_qo_heads_per_batch, totalTokens, headDim]，
    // 各 batch 的 query 沿 token 维（dim1）按 qSizes 拼接（ragged）。注意 batch 内部 query 不是连续的，
    // 同一 head 内只取该 batch 对应的 token 段，head 之间的跨度仍是 totalTokens*headDim。
    int H = (int)num_qo_heads_per_batch;
    int totalTokens = qSizesHost.empty() ? 0 : qSizesHost[batch_size];
    size_t unit = sizeof(half);

    bool ok = true;
    for (uint32_t b = 0; b < batch_size; b++) {
        int qo_start = qSizesHost[b];
        int qo_len = qSizesHost[b + 1] - qo_start;
        if (qo_len <= 0) {
            continue;
        }
        int page_start = pageSizesHost[b];
        int page_end = pageSizesHost[b + 1];
        std::vector<int32_t> pageIndices((size_t)(page_end - page_start));
        for (int i = page_start; i < page_end; i++) {
            pageIndices[i - page_start] = pageIndexsHost[i];
        }
        int lastPageLen = lastPageLensHost[b];

        // 该 batch 的 query。q 有两种物理布局：
        //  - prefill：head-major [H, totalTokens, headDim]，各 batch 沿 token 维（dim1）拼接（ragged），
        //    某个 batch 的 token 段在每个 head 内并不连续（head 跨度=totalTokens*headDim）。
        //  - decode（q.dims[1]==1）：batch-major [bsz*H, 1, headDim]，batch b 占据连续的 H 行，
        //    该 batch 是一段连续的 [H, 1, headDim]。
        // 底层 FastllmCudaHalfAttention 的 cublas 批量 GEMM 路径假设同一 kv-head 下的 group 个 head
        // 在内存中按 head 跨度=q1*headDim 连续排布，无法处理 prefill 这种 head 跨度=totalTokens*headDim
        // 的跨步视图。因此 prefill 时必须先把该 batch 的 query 收集成连续的 [H, qo_len, headDim]。
        fastllm::Data qBatch;
        qBatch.dataType = fastllm::DataType::FLOAT16;
        qBatch.dataDevice = fastllm::DataDevice::CUDA;
        qBatch.isFake = true;
        qBatch.Resize({H, qo_len, q2});
        bool decodeLayout = (q.dims.size() >= 2 && q.dims[1] == 1);
        bool qBatchOwned = false;
        if (decodeLayout) {
            // batch b 是一段连续的 [H, 1, headDim]，可直接做视图。
            qBatch.cudaData = (uint8_t*)q.cudaData + (size_t)b * H * q2 * unit;
        } else {
            // prefill：把跨步的 token 段收集成连续缓冲。
            // 源中 head h、token t 位于偏移 (h*totalTokens + qo_start + t)*headDim。
            qBatch.cudaData = FastllmCudaMalloc((size_t)H * qo_len * q2 * unit);
            qBatchOwned = true;
            cudaMemcpy2D(qBatch.cudaData, (size_t)qo_len * q2 * unit,
                         (uint8_t*)q.cudaData + (size_t)qo_start * q.strides[1] * unit,
                         (size_t)q.strides[0] * unit,
                         (size_t)qo_len * q2 * unit, (size_t)H,
                         cudaMemcpyDeviceToDevice);
        }

        // Native 的输出是连续 head-major 写入，这里用一块连续临时缓冲承接，再让 Native 转置为 token-major。
        fastllm::Data tmpOut;
        tmpOut.dataType = output.dataType;
        tmpOut.dataDevice = fastllm::DataDevice::CUDA;
        tmpOut.Resize({H, qo_len, q2});
        tmpOut.cudaData = FastllmCudaMalloc((size_t)H * qo_len * q2 * unit);
        tmpOut.isFake = true; // 内存由本函数显式管理

        bool okBatch = FastllmCudaHalfPagedAttentionNative(qBatch, pageIndices, lastPageLen,
                                                           pagedKVCacheK, pagedKVCacheV,
                                                           pageLen, numKvHeads, headDim,
                                                           tmpOut, group, scale, /*permuteOutput=*/true);
        if (okBatch) {
            // tmpOut 现为 token-major [qo_len, H, headDim]，拷贝到整体输出对应的 token 偏移处。
            cudaMemcpy((uint8_t*)output.cudaData + (size_t)qo_start * H * q2 * unit,
                       tmpOut.cudaData, (size_t)qo_len * H * q2 * unit, cudaMemcpyDeviceToDevice);
        }
        FastllmCudaFree(tmpOut.cudaData);
        if (qBatchOwned) {
            FastllmCudaFree(qBatch.cudaData);
        }
        ok = okBatch && ok;
    }
    if (ok) {
        // 整体输出为 token-major [totalTokens, num_qo_heads_per_batch, headDim]，与 FlashInfer 输出布局一致。
        output.Resize({totalTokens, H, q2});
        DeviceSync();
    }
    return ok;
}
