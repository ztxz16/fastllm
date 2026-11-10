#pragma once
#include <unordered_map>
// Hopper streaming FP8 Indexer. The producer/consumer organization follows
// NaiveOpsLib's logits_prefill_sm90: resident Q, three K stages and four compute
// warp groups. This CUDA implementation uses the existing staged cache view
// and keeps FastLLM's candidate refinement and architecture fallback.
namespace native_fp8 {
constexpr int kStreamStages = 3, kStreamKeys = 256;
constexpr int kStreamSpan = 4096;
constexpr int kStreamQBytes = 128 * 128;
constexpr int kStreamKBytes = kStreamKeys * 128;
constexpr int kStreamShared = kStreamQBytes + kStreamStages * kStreamKBytes + 7 * 8;
__device__ __forceinline__ void StreamWait(unsigned long long *bar, int phase) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    unsigned p = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("{ .reg .pred ready; wait_stage: mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1; @!ready bra wait_stage; }" :: "r"(p), "r"(phase) : "memory");
#endif
}
__device__ __forceinline__ void StreamArrive(unsigned long long *bar) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    unsigned p = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(p) : "memory");
#endif
}
__device__ __forceinline__ void StreamLoad(void *dst, const CUtensorMap *map,
                                         unsigned long long *bar, int row, int bytes) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    unsigned d = (unsigned)__cvta_generic_to_shared(dst), b = (unsigned)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {0, %2}], [%3];" :: "r"(d), "l"(map), "r"(row), "r"(b) : "memory");
#endif
}
__device__ __forceinline__ void StreamMma128(float (&c)[64], unsigned long long k, unsigned long long q) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("{ .reg .pred p; .reg .b64 a, b; setp.ne.u32 p, 0, 0; "
        "add.u64 a, %64, 0; add.u64 b, %65, 0; "
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, a, b, p, 1, 1; "
        "setp.ne.u32 p, 1, 0; "
        "add.u64 a, %64, 2; add.u64 b, %65, 2; "
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, a, b, p, 1, 1; "
        "add.u64 a, %64, 4; add.u64 b, %65, 4; "
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, a, b, p, 1, 1; "
        "add.u64 a, %64, 6; add.u64 b, %65, 6; "
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, a, b, p, 1, 1; "
        "}"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]), "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7]),
          "=f"(c[8]), "=f"(c[9]), "=f"(c[10]), "=f"(c[11]), "=f"(c[12]), "=f"(c[13]), "=f"(c[14]), "=f"(c[15]),
          "=f"(c[16]), "=f"(c[17]), "=f"(c[18]), "=f"(c[19]), "=f"(c[20]), "=f"(c[21]), "=f"(c[22]), "=f"(c[23]),
          "=f"(c[24]), "=f"(c[25]), "=f"(c[26]), "=f"(c[27]), "=f"(c[28]), "=f"(c[29]), "=f"(c[30]), "=f"(c[31]),
          "=f"(c[32]), "=f"(c[33]), "=f"(c[34]), "=f"(c[35]), "=f"(c[36]), "=f"(c[37]), "=f"(c[38]), "=f"(c[39]),
          "=f"(c[40]), "=f"(c[41]), "=f"(c[42]), "=f"(c[43]), "=f"(c[44]), "=f"(c[45]), "=f"(c[46]), "=f"(c[47]),
          "=f"(c[48]), "=f"(c[49]), "=f"(c[50]), "=f"(c[51]), "=f"(c[52]), "=f"(c[53]), "=f"(c[54]), "=f"(c[55]),
          "=f"(c[56]), "=f"(c[57]), "=f"(c[58]), "=f"(c[59]), "=f"(c[60]), "=f"(c[61]), "=f"(c[62]), "=f"(c[63]) : "l"(k), "l"(q));
#endif
}
__global__ __launch_bounds__(640, 1) void IndexerScoresStream(naive_indexer::FP8Entry e,
        const __grid_constant__ CUtensorMap qmap, const __grid_constant__ CUtensorMap kmap) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    extern __shared__ __align__(1024) unsigned char storage[];
    unsigned char *sq = storage, *sk = sq + kStreamQBytes;
    auto *ready = (unsigned long long *)(sk + kStreamStages * kStreamKBytes);
    auto *released = ready + kStreamStages, *qready = released + kStreamStages;
    const int tid = threadIdx.x, wg = tid / 128, lane = tid % 32, warp = (tid % 128) / 32;
    constexpr int span = kStreamSpan;
    const int firstQ = blockIdx.y * 8, firstK = blockIdx.x * span;
    const int lastQ = min(e.rows - 1, firstQ + 7);
    const int end = e.past + lastQ + (e.part && lastQ >= e.part ? e.gap : 0);
    const int keys = min(span, e.keys - firstK);
    const int tiles = min((keys + 255) / 256, max(0, (end - firstK + 256) / 256));
    if (firstK > end) {
        for (int i = tid; i < 8 * keys; i += 640) {
            int row = firstQ + i / keys, col = firstK + i % keys;
            if (row < e.rows) e.scores[(size_t)row * e.keys + col] = -INFINITY;
        }
        return;
    }
    // Do not execute MMA on wholly future K tiles. Materialize their mask
    // once, including any caller padding beyond the current query interval.
    const int future = max(0, keys - tiles * kStreamKeys);
    for (int i = tid; i < 8 * future; i += 640) {
        const int row = firstQ + i / future, col = firstK + tiles * kStreamKeys + i % future;
        if (row < e.rows) e.scores[(size_t)row * e.keys + col] = -INFINITY;
    }
    if (tid == 0) {
        for (int s = 0; s < kStreamStages; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(ready + s)) : "memory");
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" :: "r"((unsigned)__cvta_generic_to_shared(released + s)) : "memory");
        }
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(qready)) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (wg == 4) {
        // Reserve the full producer warp group: all 128 threads release
        // registers before the four consumers request 112 registers each.
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
        if (tid == 512) {
            StreamLoad(sq, &qmap, qready, firstQ * 16, kStreamQBytes);
            for (int tile = 0; tile < tiles; ++tile) {
                int stage = tile % kStreamStages, phase = (tile / kStreamStages) & 1;
                if (tile >= kStreamStages) StreamWait(released + stage, phase ^ 1);
                StreamLoad(sk + stage * kStreamKBytes, &kmap, ready + stage,
                           firstK + tile * kStreamKeys, kStreamKBytes);
            }
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 112;" ::: "memory");
    // QuantizeIndexer supplies positive Q scales. Fold their constant scale
    // into the head weights once, as in the reference streaming formulation.
    float weight[32];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                int h = g * 8 + (lane % 4) * 2 + p, row = firstQ + q;
                weight[q * 4 + g * 2 + p] = row < e.rows ?
                    0.5f * e.scales[row * 16 + h] * (float)e.weights[row * 16 + h] : 0.f;
            }
        }
    }
    StreamWait(qready, 0);
    for (int tile = 0; tile < tiles; ++tile) {
        const int stage = tile % kStreamStages, phase = (tile / kStreamStages) & 1;
        StreamWait(ready + stage, phase);
        auto *keyTile = sk + stage * kStreamKBytes + wg * 64 * 128;
        float dot[64];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        StreamMma128(dot, descriptor(keyTile), descriptor(sq));
        asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
        // Every group has finished reading this K stage before the producer
        // can reuse it. Remaining work uses registers and the original scales.
        if (tid % 128 == 0) StreamArrive(released + stage);
        float keyScale[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int col = firstK + tile * kStreamKeys + wg * 64 + warp * 16 + lane / 4 + half * 8;
            keyScale[half] = col <= end ? *(const float *)((const unsigned char *)e.key.base + (size_t)col * 144 + 128) : 0.f;
        }
#pragma unroll
        for (int group = 0; group < 2; ++group) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                float output = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    constexpr unsigned mask = 0xffffffff;
                    const int q = group * 4 + i;
                    float sum = 0.f;
#pragma unroll
                    for (int g = 0; g < 2; ++g) {
#pragma unroll
                        for (int p = 0; p < 2; ++p) {
                            float value = dot[(q * 2 + g) * 4 + half * 2 + p];
                            sum += (value + fabsf(value)) * weight[q * 4 + g * 2 + p];
                        }
                    }
                    sum += __shfl_xor_sync(mask, sum, 1);
                    sum += __shfl_xor_sync(mask, sum, 2);
                    if ((lane & 3) == i) output = sum;
                }
                // All four head-reduction lanes have the same score. Assign
                // them different queries so every lane performs a useful store.
                const int row = firstQ + group * 4 + (lane & 3);
                const int col = firstK + tile * kStreamKeys + wg * 64 + warp * 16 + lane / 4 + half * 8;
                if (row < e.rows && col < e.keys) {
                    const int limit = e.past + row + (e.part && row >= e.part ? e.gap : 0);
                    e.scores[(size_t)row * e.keys + col] = col <= limit ? output * keyScale[half] : -INFINITY;
                }
            }
        }
    }
#endif
}
inline bool StreamMap(CUtensorMap &map, const void *data, int rows, int stride, int tile) {
    cuuint64_t dims[] = {128, (cuuint64_t)rows}, strides[] = {(cuuint64_t)stride};
    cuuint32_t box[] = {128, (cuuint32_t)tile}, steps[] = {1, 1};
    return cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void *>(data),
        dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
inline unsigned LaunchStream(const Batch &batch) {
    static thread_local std::unordered_map<int, int> initialized;
    const int device = FastllmCudaGetDevice();
    auto it = initialized.find(device);
    int sms = 0;
    if (it != initialized.end()) sms = it->second;
    else {
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            cudaFuncSetAttribute(IndexerScoresStream, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kStreamShared) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        initialized.emplace(device, sms);
    }
    unsigned launched = 0;
    for (int i = 0; i < batch.count; ++i) if (batch.mask & (1u << i)) {
        const auto &e = batch.entries[i].prepared;
        constexpr int span = kStreamSpan;
        const int blocks = ((e.keys + span - 1) / span) * ((e.rows + 7) / 8);
        if (e.keys < 8192 || blocks < (sms + 1) / 2) continue;
        CUtensorMap q, k;
        if (!StreamMap(q, e.q, e.rows * 16, 128, 128) ||
            !StreamMap(k, e.key.base, batch.entries[i].keyRows, 144, 256)) continue;
        IndexerScoresStream<<<dim3((e.keys + span - 1) / span, (e.rows + 7) / 8),
            640, kStreamShared>>>(e, q, k);
        launched |= 1u << i;
    }
    return launched;
}
} // namespace native_fp8
