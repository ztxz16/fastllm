//
// DeepSeek-V4.1 DSpark 草稿侧的 CUDA kernel。
//
// 只有一个算子：markov head 的整条链。
//
// 语义（与 src/models/deepseekv41_dspark.cpp 里的通用算子实现逐位一致）：
//
//   token[0] = anchor
//   for i in 0 .. block-1:
//       e          = embedWeight[token[i]]                  // [rank]
//       bias[v]    = dot(headWeight[v], e)                  // [vocab]
//       token[i+1] = argmax_v (logits[i][v] + bias[v])
//       embeds[i]  = e                                      // confidence head 的输入
//
// 这条链天然串行：每一步都要用上一步的 token 去查嵌入。用通用算子拼出来的话，
// 每一步都要把 token 取回主机（查嵌入 / 切片都需要主机侧的下标），于是每步一次
// 完整的设备同步；实测在迷你模型上 5 步就要 2.6 ms，比整个目标模型的一次前向还贵，
// 其中光 [1, rank] x [vocab, rank] 这个 GEMV 就占 0.39 ms（按带宽算只该 20 us，
// 通用 Linear 在这种「一行输入、又高又窄的权重」上效率很低）。
//
// 这里把整条链放到设备上跑：token 始终留在显存里，主机只在最后取回一次。
// 每一步两个 kernel：分块算分并做块内 argmax，再把各块的结果归约成一个 token。
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
    using fastllm::Data;
    using fastllm::DataType;
    using fastllm::DataDevice;

    constexpr int kDsparkMarkovThreads = 256;
    constexpr int kDsparkMarkovMaxRank = 512;
    // 分块数：够填满 GPU 就行，太多会让归约那一步变慢
    constexpr int kDsparkMarkovChunks = 128;

    __device__ __forceinline__ float DsLoad(const half *p, int i) {
        return __half2float(p[i]);
    }

    __device__ __forceinline__ float DsLoad(const __nv_bfloat16 *p, int i) {
        return __bfloat162float(p[i]);
    }

    __device__ __forceinline__ float DsLoad(const float *p, int i) {
        return p[i];
    }

    struct DsBest {
        float value;
        int index;
    };

    // 块内 argmax 归约：并列时取下标小的，与 CPU / TopK 的行为一致
    __device__ __forceinline__ void DsBlockArgmax(DsBest &best, DsBest *shared) {
        const int tid = threadIdx.x;
        shared[tid] = best;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const DsBest &other = shared[tid + stride];
                DsBest &self = shared[tid];
                if (other.value > self.value || (other.value == self.value && other.index < self.index)) {
                    self = other;
                }
            }
            __syncthreads();
        }
        best = shared[0];
    }

    // 一行 rank 个元素的点积。每个线程独吞一整行：行是连续的（rank * 2 字节，
    // 至少一个 cache line），所有取回来的字节都会被用到，所以不需要让一个 warp
    // 去拼一行——实测让 warp 协作反而更慢（rank 小的时候 shuffle 归约的开销盖过收益，
    // 而且同时在飞的行少了 32 倍）。这里按 16 字节（8 个 bf16 / half）向量化取数，
    // 把标量 2 字节读换成 uint4，rank 越大收益越明显。
    template <typename WT>
    __device__ __forceinline__ float DsRowDot(const WT *row, const float *e, int rank) {
        float acc = 0.0f;
        int d = 0;
        if ((((uintptr_t)row) & 15) == 0) {
            const int vecEnd = rank & ~7;
            for (; d < vecEnd; d += 8) {
                const uint4 packed = *(const uint4*)(row + d);
                const WT *chunk = (const WT*)&packed;
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    acc += DsLoad(chunk, k) * e[d + k];
                }
            }
        }
        for (; d < rank; d++) {
            acc += DsLoad(row, d) * e[d];
        }
        return acc;
    }

    // float 权重一行是 rank * 4 字节，按 4 个一组（uint4）取
    __device__ __forceinline__ float DsRowDot(const float *row, const float *e, int rank) {
        float acc = 0.0f;
        int d = 0;
        if ((((uintptr_t)row) & 15) == 0) {
            const int vecEnd = rank & ~3;
            for (; d < vecEnd; d += 4) {
                const float4 packed = *(const float4*)(row + d);
                acc += packed.x * e[d] + packed.y * e[d + 1] + packed.z * e[d + 2] + packed.w * e[d + 3];
            }
        }
        for (; d < rank; d++) {
            acc += row[d] * e[d];
        }
        return acc;
    }

    // 第一步：每个块负责词表的一段，算 logits + bias 并做块内 argmax
    template <typename WT>
    __global__ void DsparkMarkovScoreKernel(const float *logits, const WT *embedWeight, const WT *headWeight,
                                            const int *tokens, int step, int vocab, int rank,
                                            float *partialValue, int *partialIndex) {
        __shared__ float e[kDsparkMarkovMaxRank];
        __shared__ DsBest reduce[kDsparkMarkovThreads];

        const int token = tokens[step];
        const WT *embedRow = embedWeight + (int64_t)token * rank;
        for (int d = threadIdx.x; d < rank; d += blockDim.x) {
            e[d] = DsLoad(embedRow, d);
        }
        __syncthreads();

        const float *logitsRow = logits + (int64_t)step * vocab;
        const int perChunk = (vocab + gridDim.x - 1) / gridDim.x;
        const int begin = blockIdx.x * perChunk;
        const int end = min(vocab, begin + perChunk);

        DsBest best;
        best.value = -FLT_MAX;
        best.index = vocab;                     // 空块：归约时不会被选中
        for (int v = begin + threadIdx.x; v < end; v += blockDim.x) {
            const float score = logitsRow[v] + DsRowDot(headWeight + (int64_t)v * rank, e, rank);
            if (score > best.value || (score == best.value && v < best.index)) {
                best.value = score;
                best.index = v;
            }
        }
        DsBlockArgmax(best, reduce);
        if (threadIdx.x == 0) {
            partialValue[blockIdx.x] = best.value;
            partialIndex[blockIdx.x] = best.index;
        }
    }

    // 第二步：归约各块的结果，写下一个 token；顺便把本步用到的嵌入存给 confidence head
    template <typename WT>
    __global__ void DsparkMarkovReduceKernel(const float *partialValue, const int *partialIndex, int chunks,
                                             const WT *embedWeight, const int *tokens, int step, int rank,
                                             int *outTokens, float *outEmbeds) {
        __shared__ DsBest reduce[kDsparkMarkovThreads];
        DsBest best;
        best.value = -FLT_MAX;
        best.index = 0x7fffffff;
        for (int i = threadIdx.x; i < chunks; i += blockDim.x) {
            if (partialValue[i] > best.value ||
                (partialValue[i] == best.value && partialIndex[i] < best.index)) {
                best.value = partialValue[i];
                best.index = partialIndex[i];
            }
        }
        DsBlockArgmax(best, reduce);
        if (threadIdx.x == 0) {
            outTokens[step + 1] = best.index;
        }
        // embeds[step] 是本步输入 token 的嵌入（官方实现里 confidence head 用的就是它）
        const int token = tokens[step];
        const WT *embedRow = embedWeight + (int64_t)token * rank;
        for (int d = threadIdx.x; d < rank; d += blockDim.x) {
            outEmbeds[(int64_t)step * rank + d] = DsLoad(embedRow, d);
        }
    }

    // 在当前 CUDA 设备上准备输出张量（deepseekv41-kernels.cu 里的同名逻辑在匿名
    // 命名空间中，跨文件不可复用，这里按同样的方式重写一份）
    bool DsPrepareOutput(Data &output, DataType dataType, const std::vector<int> &dims) {
        output.dataType = dataType;
        output.Resize(dims);
        output.ToDevice(DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
        output.Allocate(false);
        return output.cudaData != nullptr;
    }

    bool DsOnCuda(const Data &d) {
        return d.dataDevice == DataDevice::CUDA && d.cudaData != nullptr && !d.multiDeviceData;
    }

    bool DsSupportedWeight(const Data &d) {
        return d.dataType == DataType::FLOAT16 || d.dataType == DataType::BFLOAT16 ||
               d.dataType == DataType::FLOAT32;
    }

    template <typename WT>
    void DsLaunchChain(const float *logits, const Data &embedWeight, const Data &headWeight,
                       int block, int vocab, int rank, int *tokens,
                       float *partialValue, int *partialIndex, float *outEmbeds) {
        const WT *embedData = (const WT*)embedWeight.cudaData;
        const WT *headData = (const WT*)headWeight.cudaData;
        for (int step = 0; step < block; step++) {
            DsparkMarkovScoreKernel<WT><<<kDsparkMarkovChunks, kDsparkMarkovThreads>>>(
                logits, embedData, headData, tokens, step, vocab, rank, partialValue, partialIndex);
            DsparkMarkovReduceKernel<WT><<<1, kDsparkMarkovThreads>>>(
                partialValue, partialIndex, kDsparkMarkovChunks, embedData, tokens, step, rank,
                tokens, outEmbeds);
        }
    }
}

// logits: [1, block, vocab] FLOAT32；embedWeight / headWeight: [vocab, rank] FP16 / BF16 / FP32。
// outTokens 写回 block 个 token（主机内存），outEmbeds 为设备上的 [1, block, rank] FLOAT32。
// 任何前提不满足时返回 false，调用方退回通用算子实现。
extern "C" bool FastllmCudaDeepSeekV41MarkovChain(const fastllm::Data &logits,
                                                  const fastllm::Data &embedWeight,
                                                  const fastllm::Data &headWeight,
                                                  int anchorToken, int block,
                                                  std::vector<int> &outTokens,
                                                  fastllm::Data &outEmbeds) {
    if (!DsOnCuda(logits) || !DsOnCuda(embedWeight) || !DsOnCuda(headWeight)) {
        return false;
    }
    if (logits.dataType != DataType::FLOAT32 || logits.dims.size() != 3 || logits.dims[0] != 1) {
        return false;
    }
    if (embedWeight.dims.size() != 2 || headWeight.dims.size() != 2) {
        return false;
    }
    if (!DsSupportedWeight(embedWeight) || embedWeight.dataType != headWeight.dataType) {
        return false;
    }
    const int vocab = logits.dims[2];
    const int rank = embedWeight.dims[1];
    if (block <= 0 || logits.dims[1] < block || rank <= 0 || rank > kDsparkMarkovMaxRank) {
        return false;
    }
    if (embedWeight.dims[0] < vocab || headWeight.dims[0] < vocab || headWeight.dims[1] != rank) {
        return false;
    }
    if (anchorToken < 0 || anchorToken >= vocab) {
        return false;
    }
    if (!DsPrepareOutput(outEmbeds, DataType::FLOAT32, {1, block, rank})) {
        return false;
    }

    // token[0] = anchor，之后每一步在设备上写下一个；主机只在最后取回一次
    int *tokens = (int*)FastllmCudaMalloc((size_t)(block + 1) * sizeof(int));
    float *partialValue = (float*)FastllmCudaMalloc((size_t)kDsparkMarkovChunks * sizeof(float));
    int *partialIndex = (int*)FastllmCudaMalloc((size_t)kDsparkMarkovChunks * sizeof(int));
    if (tokens == nullptr || partialValue == nullptr || partialIndex == nullptr) {
        if (tokens != nullptr) FastllmCudaFree(tokens);
        if (partialValue != nullptr) FastllmCudaFree(partialValue);
        if (partialIndex != nullptr) FastllmCudaFree(partialIndex);
        return false;
    }
    FastllmCudaCopyFromHostToDevice(tokens, &anchorToken, sizeof(int));

    const float *logitsData = (const float*)logits.cudaData;
    float *embedsData = (float*)outEmbeds.cudaData;
    if (embedWeight.dataType == DataType::FLOAT16) {
        DsLaunchChain<half>(logitsData, embedWeight, headWeight, block, vocab, rank,
                            tokens, partialValue, partialIndex, embedsData);
    } else if (embedWeight.dataType == DataType::BFLOAT16) {
        DsLaunchChain<__nv_bfloat16>(logitsData, embedWeight, headWeight, block, vocab, rank,
                                     tokens, partialValue, partialIndex, embedsData);
    } else {
        DsLaunchChain<float>(logitsData, embedWeight, headWeight, block, vocab, rank,
                             tokens, partialValue, partialIndex, embedsData);
    }

    outTokens.resize(block);
    FastllmCudaCopyFromDeviceToHost(outTokens.data(), tokens + 1, (size_t)block * sizeof(int));
    FastllmCudaFree(tokens);
    FastllmCudaFree(partialValue);
    FastllmCudaFree(partialIndex);

    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess) {
        printf("[Fastllm] DeepSeek-V4.1 DSpark markov chain kernel error: %s\n", cudaGetErrorString(state));
        return false;
    }
    for (int i = 0; i < block; i++) {
        if (outTokens[i] < 0 || outTokens[i] >= vocab) {
            return false;
        }
    }
    return true;
}
