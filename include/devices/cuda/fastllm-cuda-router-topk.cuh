#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include "fastllm-cuda-ordered-reduce.cuh"

namespace fastllm { namespace cuda {
// One warp owns this scratch. Exact ties preserve the legacy 64-list merge.
// Input probabilities and optional correction bias are not modified.
template<int TopK> struct RouterTopKScratch {
    float keys[TopK], probabilities[TopK];
    int ids[TopK];
    // An odd list stride avoids shared-memory bank conflicts in the merge.
    float legacyKeys[64][TopK + 1], legacyIds[64][TopK + 1];
};
__device__ __forceinline__ void LoadRouter8(const float *p, float (&values)[8]) {
    float4 a=*reinterpret_cast<const float4 *>(p), b=*reinterpret_cast<const float4 *>(p+4);
    values[0]=a.x; values[1]=a.y; values[2]=a.z; values[3]=a.w;
    values[4]=b.x; values[5]=b.y; values[6]=b.z; values[7]=b.w;
}
template<int EXPERTS, int TOPK>
__device__ __forceinline__ void SelectRouterTopK(const float *probabilityLogits,
        const float *bias, int32_t *index, float *score, int hasBias, int needNorm,
        float routeScale, RouterTopKScratch<TOPK> &scratch) {
    // Vector loads and register storage below cover these existing router widths.
    static_assert(EXPERTS == 256 || EXPERTS == 288 || EXPERTS == 384 || EXPERTS == 512);
    static_assert(TOPK > 0 && TOPK <= 32);
    constexpr int VALUES_PER_LANE=EXPERTS/32, LEGACY_THREADS=64;
    constexpr float NEG_INF=-FLT_MAX;
    constexpr unsigned FULL_WARP_MASK=0xffffffffu;
    auto &selectedKeys=scratch.keys;
    auto &selectedIds=scratch.ids;
    auto &selectedProbabilities=scratch.probabilities;
    auto &legacyKeys=scratch.legacyKeys;
    auto &legacyIds=scratch.legacyIds;
    int lane = threadIdx.x % 32;
    int firstExpert = lane * VALUES_PER_LANE;
    float choiceKeys[VALUES_PER_LANE];
    float firstProbabilities[8], secondProbabilities[8];
    if constexpr (EXPERTS != 288 && EXPERTS != 384) {
        LoadRouter8(probabilityLogits + firstExpert, firstProbabilities);
        if constexpr (VALUES_PER_LANE > 8)
            LoadRouter8(probabilityLogits + firstExpert + 8, secondProbabilities);
    }
#pragma unroll
    for (int part = 0; part < VALUES_PER_LANE; ++part) {
        int expert = firstExpert + part;
        float probability;
        if constexpr (EXPERTS == 288 || EXPERTS == 384) probability = probabilityLogits[expert];
        else probability = part < 8 ? firstProbabilities[part] : secondProbabilities[part - 8];
        float key = probability + (hasBias ? bias[expert] : 0.0f);
        choiceKeys[part] = isfinite(key) ? key : NEG_INF;
    }

#pragma unroll
    for (int rank = 0; rank < TOPK; ++rank) {
        auto winner = fastllm::cuda::WarpArgMax(choiceKeys, firstExpert, EXPERTS);

        int owner = winner.index / VALUES_PER_LANE;
        int ownerPart = winner.index % VALUES_PER_LANE;
        // The selected row is already available in shared/global storage.
        // A dynamic index into a per-thread array forces a local-memory stack.
        float probability = probabilityLogits[winner.index];
        if (lane == 0) {
            selectedKeys[rank] = winner.value;
            selectedIds[rank] = winner.index;
            selectedProbabilities[rank] =
                isfinite(probability) ? probability : 0.0f;
        }
        if (lane == owner) {
#pragma unroll
            for (int part = 0; part < VALUES_PER_LANE; ++part) {
                if (part == ownerPart) {
                    choiceKeys[part] = NEG_INF;
                }
            }
        }
    }

    __syncwarp(FULL_WARP_MASK);
    float cutoff = lane == 0 ? selectedKeys[TOPK - 1] : 0.0f;
    cutoff = __shfl_sync(FULL_WARP_MASK, cutoff, 0);
    int useLegacyMerge = 0;
#pragma unroll
    for (int part = 0; part < VALUES_PER_LANE; ++part) {
        useLegacyMerge |= choiceKeys[part] == cutoff;
    }
    useLegacyMerge = __any_sync(FULL_WARP_MASK, useLegacyMerge);
    if (lane == 0) {
#pragma unroll
        for (int rank = 1; rank < TOPK; ++rank) {
            useLegacyMerge |= selectedKeys[rank] == selectedKeys[rank - 1];
        }
    }
    useLegacyMerge = __shfl_sync(FULL_WARP_MASK, useLegacyMerge, 0);

    if (!useLegacyMerge) {
        float selectedSum = 1.0f;
        if (lane == 0 && needNorm) {
            selectedSum = 0.0f;
#pragma unroll
            for (int rank = 0; rank < TOPK; ++rank)
                selectedSum += selectedProbabilities[rank];
            if (!isfinite(selectedSum) || fabsf(selectedSum) < 1.0e-20f)
                selectedSum = 1.0f;
        }
        selectedSum = __shfl_sync(FULL_WARP_MASK, selectedSum, 0);
        if (lane < TOPK) {
            index[lane] = selectedIds[lane];
            score[lane] = selectedProbabilities[lane] / selectedSum * routeScale;
        }
        return;
    }

    // Restore winners consumed by the fast scan, then recreate the exact
    // virtual-thread lists used by FastllmSelectExpertKernel.
#pragma unroll
    for (int rank = 0; rank < TOPK; ++rank) {
        int expert = lane == 0 ? selectedIds[rank] : 0;
        float key = lane == 0 ? selectedKeys[rank] : 0.0f;
        expert = __shfl_sync(FULL_WARP_MASK, expert, 0);
        key = __shfl_sync(FULL_WARP_MASK, key, 0);
        int owner = expert / VALUES_PER_LANE;
        int ownerPart = expert % VALUES_PER_LANE;
        if (lane == owner) {
#pragma unroll
            for (int part = 0; part < VALUES_PER_LANE; ++part) {
                if (part == ownerPart) {
                    choiceKeys[part] = key;
                }
            }
        }
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        int virtualTid = lane + half * 32;
#pragma unroll
        for (int rank = 0; rank < TOPK; ++rank) {
            legacyKeys[virtualTid][rank] = -1.0e100;
            legacyIds[virtualTid][rank] = -1.0f;
        }
#pragma unroll
        for (int part = 0; part < (EXPERTS + LEGACY_THREADS - 1) / LEGACY_THREADS; ++part) {
            int expert = virtualTid + part * LEGACY_THREADS;
            int owner = expert / VALUES_PER_LANE;
            int ownerPart = expert % VALUES_PER_LANE;
            float key = NEG_INF;
#pragma unroll
            for (int sourcePart = 0; sourcePart < VALUES_PER_LANE;
                 ++sourcePart) {
                float candidate = __shfl_sync(
                    FULL_WARP_MASK, choiceKeys[sourcePart], owner);
                if (sourcePart == ownerPart) {
                    key = candidate;
                }
            }
#pragma unroll
            for (int rank = 0; rank < TOPK; ++rank) {
                if (expert < EXPERTS && key > legacyKeys[virtualTid][rank]) {
#pragma unroll
                    for (int shift = TOPK - 1; shift > rank; --shift) {
                        legacyKeys[virtualTid][shift] =
                            legacyKeys[virtualTid][shift - 1];
                        legacyIds[virtualTid][shift] =
                            legacyIds[virtualTid][shift - 1];
                    }
                    legacyKeys[virtualTid][rank] = key;
                    legacyIds[virtualTid][rank] = (float)expert;
                    break;
                }
            }
        }
    }
    __syncwarp(FULL_WARP_MASK);

    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            int pos0 = 0;
            int pos1 = 0;
            while (pos0 + pos1 < TOPK) {
                if (legacyKeys[lane][pos0] >
                    legacyKeys[lane + stride][pos1]) {
                    ++pos0;
                } else {
                    ++pos1;
                }
            }
            --pos0;
            --pos1;
            for (int position = TOPK - 1; position >= 0; --position) {
                if (pos1 < 0 ||
                    (pos0 >= 0 && legacyKeys[lane][pos0] <
                                      legacyKeys[lane + stride][pos1])) {
                    legacyKeys[lane][position] = legacyKeys[lane][pos0];
                    legacyIds[lane][position] = legacyIds[lane][pos0];
                    --pos0;
                } else {
                    legacyKeys[lane][position] =
                        legacyKeys[lane + stride][pos1];
                    legacyIds[lane][position] =
                        legacyIds[lane + stride][pos1];
                    --pos1;
                }
            }
        }
        __syncwarp(FULL_WARP_MASK);
    }

    if (lane == 0) {
        float selectedSum = 1.0f;
        if (needNorm) {
            selectedSum = 0.0f;
#pragma unroll
            for (int rank = 0; rank < TOPK; ++rank) {
                int expert = (int)legacyIds[0][rank];
                float probability = probabilityLogits[expert];
                if (isfinite(probability)) {
                    selectedSum += probability;
                }
            }
            if (!isfinite(selectedSum) || fabsf(selectedSum) < 1.0e-20f) {
                selectedSum = 1.0f;
            }
        }
#pragma unroll
        for (int rank = 0; rank < TOPK; ++rank) {
            int expert = (int)legacyIds[0][rank];
            float probability = probabilityLogits[expert];
            index[rank] = expert;
            score[rank] = (isfinite(probability) ? probability : 0.0f) /
                          selectedSum * routeScale;
        }
    }
}
}} // namespace fastllm::cuda
