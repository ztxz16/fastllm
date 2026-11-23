#pragma once
#include <cfloat>
#include <cstdint>
#ifndef USE_ROCM
#include "fastllm-cuda-router-topk.cuh"
#endif

namespace fastllm::cuda {
#ifndef USE_ROCM
// Fixed-shape selection with a legacy-compatible tie fallback.
// APPLY_SOFTMAX joins the established 256-thread FP32 softmax with
// selection while reproducing its max/sum/division order exactly. One warp
// then performs TopK; exact ties rebuild the original 64-thread lists and
// merge tree so all supported shapes remain compatible with the legacy path.
template <bool APPLY_SOFTMAX, int EXPERTS, int TOPK, bool APPLY_SIGMOID = false>
__device__ __forceinline__ void SelectExpertFixedRow(
        const float *logits, const float *bias, int32_t *index, float *score,
        int hasBias, int needNorm, float routeScale, int token = blockIdx.x) {
    __shared__ fastllm::cuda::RouterTopKScratch<TOPK> scratch;
    __shared__ float softmaxReduce[257];
    __shared__ __align__(16) float softmaxProbabilities[EXPERTS];

    static_assert((EXPERTS == 256 && TOPK == 8) ||
                  (EXPERTS == 512 && TOPK == 10) ||
                  (!APPLY_SOFTMAX && ((EXPERTS == 128 && TOPK == 8) ||
                                      (EXPERTS == 288 && TOPK == 8) ||
                                      (EXPERTS == 384 && TOPK == 6))));
    const float *tokenLogits = logits + (size_t)token * EXPERTS;
    index += (size_t)token * TOPK;
    score += (size_t)token * TOPK;
    int tid = threadIdx.x;
    const float *probabilityLogits = tokenLogits;
    if constexpr (APPLY_SOFTMAX) {
        float maxValue = -1e100;
        for (int expert = tid; expert < EXPERTS; expert += blockDim.x) {
            maxValue = max(maxValue, tokenLogits[expert]);
        }
        float softmaxMaximum = fastllm::cuda::OrderedBlockReduce<256>(
            maxValue, softmaxReduce, fastllm::cuda::OrderedMax{});

        float sum = 0.0f;
        for (int expert = tid; expert < EXPERTS; expert += blockDim.x) {
            softmaxProbabilities[expert] =
                exp(tokenLogits[expert] - softmaxMaximum);
            sum += softmaxProbabilities[expert];
        }
        float denominator = fastllm::cuda::OrderedBlockReduce<256>(
            sum, softmaxReduce, fastllm::cuda::OrderedSum{});
        if (fabs(denominator) < 1e-6) denominator = 0.0001f;
        for (int expert = tid; expert < EXPERTS; expert += blockDim.x) {
            softmaxProbabilities[expert] /= denominator;
        }
        __syncthreads();
        probabilityLogits = softmaxProbabilities;
    }
    if constexpr (APPLY_SIGMOID) {
        // Match the standalone FP32 sigmoid, including its double division.
        for (int expert = tid; expert < EXPERTS; expert += 32) {
            float x = tokenLogits[expert];
            softmaxProbabilities[expert] = 1.0 / (1.0 + expf(-x));
        }
        __syncwarp();
        probabilityLogits = softmaxProbabilities;
    }
    if (tid >= 32) {
        return;
    }
    fastllm::cuda::SelectRouterTopK<EXPERTS, TOPK>(probabilityLogits, bias,
        index, score, hasBias, needNorm, routeScale, scratch);
}

#endif

// CUDA kernel for SelectExpert
template <int THREAD_PER_BLOCK, int MAXK, bool Sigmoid = false>
__device__ __forceinline__ void SelectExpertRow(const float *logits, const float *bias, int32_t *index, float *score,
    int numExperts, int topk, int hasBias, bool needNorm, float routeScale) {
    __shared__ float idData[THREAD_PER_BLOCK][MAXK];
    __shared__ float maxData[THREAD_PER_BLOCK][MAXK];

    int tokenIdx = blockIdx.x;
    const float *inputData = logits + tokenIdx * numExperts;
    int32_t *outputIndex = index + tokenIdx * topk;
    float *outputScore = score + tokenIdx * topk;

    int tid = threadIdx.x;

    auto probability = [&](int id) {
        float x = inputData[id];
        if constexpr (Sigmoid) x = 1.0 / (1.0 + expf(-x));
        return x;
    };
    // The routing CTA may also be sized for a neighboring quantization tile.
    if (tid < THREAD_PER_BLOCK) {
        // Initialize
        for (int i = 0; i < topk; i++) {
            maxData[tid][i] = -1e100;
            idData[tid][i] = -1;
        }

        // Find topk experts
        for (int j = tid; j < numExperts; j += THREAD_PER_BLOCK) {
            float cur = probability(j);
            if (hasBias) {
                cur += bias[j];
            }
            // A non-finite router value used to leave idData at -1.  The writeback
            // below would then read inputData[-1], turning a numerical issue into an
            // illegal CUDA access and poisoning the entire device context.  Rank a
            // non-finite value last while retaining a valid expert id instead.
            if (!isfinite(cur)) {
                cur = -FLT_MAX;
            }

            for (int l = 0; l < topk; l++) {
                if (cur > maxData[tid][l]) {
                    for (int x = topk - 1; x > l; x--) {
                        maxData[tid][x] = maxData[tid][x - 1];
                        idData[tid][x] = idData[tid][x - 1];
                    }
                    maxData[tid][l] = cur;
                    idData[tid][l] = j;
                    break;
                }
            }
        }
    }
    __syncthreads();

    // Merge results from all threads
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int pos0 = 0, pos1 = 0;
            while (pos0 + pos1 < topk) {
                if (maxData[tid][pos0] > maxData[tid + s][pos1]) {
                    pos0++;
                } else {
                    pos1++;
                }
            }
            pos0--;
            pos1--;
            int pos = topk - 1;
            while (pos >= 0) {
                if (pos1 < 0 || (pos0 >= 0 && maxData[tid][pos0] < maxData[tid + s][pos1])) {
                    maxData[tid][pos] = maxData[tid][pos0];
                    idData[tid][pos] = idData[tid][pos0];
                    pos0--;
                } else {
                    maxData[tid][pos] = maxData[tid + s][pos1];
                    idData[tid][pos] = idData[tid + s][pos1];
                    pos1--;
                }
                pos--;
            }
        }
        __syncthreads();
    }

    // Write output
    if (tid == 0) {
        // Calculate sum for normalization
        float sum = 1.0f;
        if (needNorm) {
            sum = 0.0f;
            for (int i = 0; i < topk; i++) {
                int expertIdx = idData[0][i];
                if (expertIdx >= 0 && expertIdx < numExperts &&
                    isfinite(probability(expertIdx))) {
                    sum += probability(expertIdx);
                }
            }
            if (!isfinite(sum) || fabsf(sum) < 1.0e-20f) {
                sum = 1.0f;
            }
        }

        // Write index and score
        for (int i = 0; i < topk; i++) {
            int expertIdx = idData[0][i];
            if (expertIdx < 0 || expertIdx >= numExperts) {
                expertIdx = 0;
            }
            float expertScore = probability(expertIdx);
            if (!isfinite(expertScore)) {
                expertScore = 0.0f;
            }
            outputIndex[i] = expertIdx;
            outputScore[i] = expertScore / sum * routeScale;
        }
    }
}
} // namespace fastllm::cuda
