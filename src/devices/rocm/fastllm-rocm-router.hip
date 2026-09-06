#include "devices/rocm/fastllm-rocm-router.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cfloat>
#include <cmath>

namespace {
constexpr int Experts = 256;

template <typename T>
__device__ float RouterFloat(T value) {
    return static_cast<float>(value);
}

__device__ float RouterBias(const void *bias, int type, int expert) {
    if (type == 1) return RouterFloat(static_cast<const half *>(bias)[expert]);
    if (type == 2) return RouterFloat(static_cast<const hip_bfloat16 *>(bias)[expert]);
    return static_cast<const float *>(bias)[expert];
}

template <typename T, bool Sigmoid>
__global__ void FusedSelectExpert256(
        const T *logits, const void *bias, int biasType,
        int32_t *index, float *score, int topk, bool needNorm,
        float routeScale) {
    const int tid = threadIdx.x;
    const size_t row = blockIdx.x;
    const float logit = RouterFloat(logits[row * Experts + tid]);
    __shared__ float reduction[Experts];
    __shared__ float probabilities[Experts];
    __shared__ float keys[Experts];
    __shared__ int ids[Experts];

    float probability;
    if constexpr (Sigmoid) {
        probability = isnan(logit) ? 0.0f : 1.0f / (1.0f + expf(-logit));
    } else {
        reduction[tid] = isnan(logit) ? -INFINITY : logit;
        __syncthreads();
        for (int stride = Experts / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
            __syncthreads();
        }
        const float maximum = reduction[0];
        // Finish every read before reusing the reduction buffer. Block-wide
        // barriers make this independent of both wave32 and wave64 execution.
        __syncthreads();
        probability = maximum == INFINITY ? (logit == INFINITY ? 1.0f : 0.0f) :
            (isfinite(maximum) && !isnan(logit) ? expf(logit - maximum) : 0.0f);
        reduction[tid] = probability;
        __syncthreads();
        for (int stride = Experts / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            __syncthreads();
        }
        const float sum = reduction[0];
        probability = sum > 0.0f ? probability / sum : 0.0f;
    }
    probabilities[tid] = probability;
    // A correction bias changes selection only, never the returned weight.
    // Without it, rank softmax logits directly to avoid rounding close logits
    // to the same probability (or underflowing them to zero).
    float key = !Sigmoid && bias == nullptr ? logit : probability;
    if (bias != nullptr) key += RouterBias(bias, biasType, tid);
    keys[tid] = isnan(key) ? -INFINITY : key;
    ids[tid] = tid;
    __syncthreads();

    // Sort whole (key, id) pairs. Each disjoint pair has one writer, followed
    // by a block barrier; no CUDA warp shuffles or in-place list merges are
    // involved. Equal keys have a deterministic ascending expert-id order.
    for (int width = 2; width <= Experts; width <<= 1) {
        for (int stride = width / 2; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            if (other > tid) {
                const float a = keys[tid], b = keys[other];
                const int aId = ids[tid], bId = ids[other];
                const bool aFirst = a > b || (a == b && aId < bId);
                const bool descending = (tid & width) == 0;
                if (aFirst != descending) {
                    keys[tid] = b;
                    keys[other] = a;
                    ids[tid] = bId;
                    ids[other] = aId;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        float sum = 1.0f;
        if (needNorm) {
            sum = 0.0f;
            for (int rank = 0; rank < topk; ++rank) sum += probabilities[ids[rank]];
            if (!(sum > 0.0f)) sum = 1.0f;
        }
        for (int rank = 0; rank < topk; ++rank) {
            const int expert = ids[rank];
            index[row * topk + rank] = expert;
            score[row * topk + rank] = probabilities[expert] / sum * routeScale;
        }
    }
}

template <typename T>
void LaunchRouter(const void *logits, const void *bias, int biasType,
        int32_t *index, float *score, int tokens, int topk,
        bool sigmoid, bool needNorm, float routeScale) {
    if (sigmoid) {
        FusedSelectExpert256<T, true><<<tokens, Experts, 0, hipStreamPerThread>>>(
            static_cast<const T *>(logits), bias, biasType, index, score,
            topk, needNorm, routeScale);
    } else {
        FusedSelectExpert256<T, false><<<tokens, Experts, 0, hipStreamPerThread>>>(
            static_cast<const T *>(logits), bias, biasType, index, score,
            topk, needNorm, routeScale);
    }
}
} // namespace

void FastllmRocmFusedSelectExpert256(
        const void *logits, fastllm::DataType dataType, const void *bias,
        int biasType, int32_t *index, float *score, int tokens, int topk,
        bool sigmoid, bool needNorm, float routeScale) {
    if (dataType == fastllm::DataType::FLOAT16) {
        LaunchRouter<half>(logits, bias, biasType, index, score,
            tokens, topk, sigmoid, needNorm, routeScale);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        LaunchRouter<hip_bfloat16>(logits, bias, biasType, index, score,
            tokens, topk, sigmoid, needNorm, routeScale);
    } else {
        LaunchRouter<float>(logits, bias, biasType, index, score,
            tokens, topk, sigmoid, needNorm, routeScale);
    }
}
