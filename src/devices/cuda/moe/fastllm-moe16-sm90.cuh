// Native SM90 TMA staging for grouped BF16/FP16 experts. CUDA/PTX only.
// The generic WMMA path remains available on older architectures and small blocks.
#pragma once
#include <cuda.h>
#include <map>
#include <array>
#include <tuple>

#ifdef FASTLLM_ENABLE_MOE16_SM90
bool FastllmCudaMoe16WgmmaAvailable();
void FastllmCudaMoe16WgmmaExperts(const CUtensorMap *inputMap, const CUtensorMap *gateMap,
    const CUtensorMap *middleMap, const CUtensorMap *downMap, const float *scores,
    const int *workspace, void *middle, void *parts, void *packedMiddle,
    int rows, int hidden, int intermediate, int experts, int tiles, cudaStream_t stream);
#endif

#if CUDART_VERSION >= 12000
namespace {
struct Moe16ExpertMaps { CUtensorMap input, weight; };
__device__ __forceinline__ unsigned Moe16SharedAddress(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ int Moe16Swizzle128(int p) {
    return p ^ (((p >> 6) & 7) << 3);
}
__device__ __forceinline__ void Moe16TmaLoad(void *dst, const CUtensorMap *map,
        int k, int row, unsigned long long *barrier) {
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::
        "r"(Moe16SharedAddress(dst)), "l"(map), "r"(k), "r"(row),
        "r"(Moe16SharedAddress(barrier)) : "memory");
#endif
}
__device__ __forceinline__ void Moe16TmaWait(unsigned long long *barrier, int phase) {
#if __CUDA_ARCH__ >= 900
    asm volatile("{ .reg .pred p; wait_loop: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra wait_loop; }" ::
        "r"(Moe16SharedAddress(barrier)), "r"(phase) : "memory");
#endif
}
template<typename T>
__device__ __forceinline__ void Moe16Mma16816(float *c, const unsigned *a, const unsigned *b) {
#if __CUDA_ARCH__ >= 900
    if constexpr (std::is_same_v<T, __half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" :
            "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) :
            "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" :
            "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) :
            "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
#endif
}
template<typename T, int RowsPerBlock = 1, int TileRows = 16>
__global__ void Moe16PackExpertInput(const T *input, const int *workspace,
        T *packed, int rows, int hidden, int experts, int topk) {
    const int *tiles = workspace + (size_t)experts * (rows + 1);
    const int count = TileRows == 64 ? -tiles[0] : tiles[0];
    if (int(blockIdx.y) >= count) return;
    int index = tiles[blockIdx.y + 1], expert = index / (rows + 1), start = index % (rows + 1);
    // Decode uses two rows per CTA for the measured 2048-wide geometry;
    // prefill keeps four warp-sized rows. Other shapes retain one row.
    const int lanes = blockDim.x / RowsPerBlock;
    int row = blockIdx.x * RowsPerBlock + threadIdx.x / lanes;
    int task = start + row < workspace[expert * (rows + 1) + rows] ? workspace[index + row] : -1;
    auto *dst = reinterpret_cast<uint4 *>(packed + ((size_t)blockIdx.y * TileRows + row) * hidden);
    for (int d = threadIdx.x % lanes;
         d < hidden / 8; d += lanes) {
        uint4 value = make_uint4(0, 0, 0, 0);
        if (task >= 0) {
            const auto *src = input + (size_t)(task / topk) * hidden;
            if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) value = reinterpret_cast<const uint4 *>(src)[d];
            else {
                // Tensor views may begin at an unaligned 16-bit element. Copy
                // their bits without arithmetic conversions.
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    reinterpret_cast<uint16_t *>(&value)[j] = reinterpret_cast<const uint16_t *>(src)[d * 8 + j];
            }
        }
        dst[d] = value;
    }
}

template<typename T>
__device__ __forceinline__ void Moe16StoreExpertPair(T *p, float x, float y) {
    // Preserve the storage-precision rounding boundaries; tensor views may be unaligned.
    auto v = Moe16Cast2<T>(x, y);
    if ((reinterpret_cast<uintptr_t>(p) & 3) == 0)
        *(Moe16Pair<T> *)p = v;
    else {
        p[0] = Moe16Cast<T>(x);
        p[1] = Moe16Cast<T>(y);
    }
}

template<bool Gate, int K> constexpr int Moe16TmaExpertSharedBytes() {
    return 2 * (16 + 64 * (Gate ? 2 : 1)) * K * sizeof(uint16_t);
}
template<typename T, bool Gate, int K = 64, bool Gather = false, bool Strided = false>
__global__ void Moe16TmaExpertGemm(
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
        const __grid_constant__ Moe16ExpertMaps maps,
#else
        const Moe16ExpertMaps maps,
#endif
        const float *scores, const int *workspace, T *output,
        T *packedMiddle, int rows, int hidden, int intermediate, int experts, const T *input = nullptr, int topk = 1) {
#if __CUDA_ARCH__ >= 900
    constexpr int M = 16, N = 64, Stages = 2;
    constexpr int StageElements = (M + N * (Gate ? 2 : 1)) * K;
    const int *tiles = workspace + (size_t)experts * (rows + 1);
    if (int(blockIdx.y) >= tiles[0]) return;
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, column = blockIdx.x * N;
    int tile = blockIdx.y;
    do {
        int index = tiles[tile + 1], expert = index / (rows + 1), start = index % (rows + 1);
        int inputWidth = Gate ? hidden : intermediate, outputWidth = Gate ? intermediate : hidden;
        extern __shared__ __align__(1024) unsigned char rawStorage[];
        auto *storage = reinterpret_cast<T *>(rawStorage);
        __shared__ unsigned long long barriers[Stages];
        __shared__ int tasks[M];
        if (tid < M) tasks[tid] = start + tid < workspace[expert * (rows + 1) + rows] ? workspace[index + tid] : -1;
        if (tid == 0) {
#pragma unroll
            for (int i = 0; i < Stages; ++i)
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(Moe16SharedAddress(barriers + i)), "r"(Gather ? 129 : 1) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();
        // Gather the original token rows directly into the existing MMA layout.
        // The barrier tracks 128 per-thread copies plus the TMA weight issuer, so
        // all input/weight data is ready before consumers reuse this stage.
        static_assert(!Gather || (Gate && K == 64));
        const int gatherTask = Gather ? tasks[tid / 8] : 0;
        const T *gatherSource = Gather ? input + (size_t)(gatherTask < 0 ? 0 : gatherTask / topk) * hidden + (tid % 8) * 8 : nullptr;
        auto load = [&](int step) {
            bool valid = step * K < inputWidth;
            int stage = step % Stages;
            auto *ptr = storage + stage * StageElements;
            int weightRow = expert * outputWidth * (Gate ? 2 : 1) + column;
            if (tid == 0 && valid)
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                    "r"(Moe16SharedAddress(barriers + stage)), "r"((StageElements - (Gather ? M * K : 0)) * 2) : "memory");
            if constexpr (Gather) {
                if (valid) {
                    Moe16CopyAsync16(ptr + Moe16Swizzle128((tid / 8) * 64 + (tid % 8) * 8),
                        gatherSource + step * K, gatherTask >= 0);
                    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::
                        "r"(Moe16SharedAddress(barriers + stage)) : "memory");
                }
            }
            // K=128 assigns its two64-element TMA chunks to different warps.
            // Publish the total transaction count before either warp issues a load.
            // K=64 retains its single issuer and needs no extra CTA barrier.
            if constexpr (K > 64) __syncthreads();
            if ((K == 64 ? tid == 0 : lane == 0) && valid) {
                for (int chunk = K == 64 ? 0 : warp; chunk < K / 64; chunk += 4) {
                    if constexpr (!Gather) Moe16TmaLoad(ptr + chunk * M * 64, &maps.input,
                        step * K + chunk * 64, tile * M, barriers + stage);
                    Moe16TmaLoad(ptr + M * K + chunk * N * 64, &maps.weight,
                        step * K + chunk * 64, weightRow, barriers + stage);
                    if (Gate) Moe16TmaLoad(ptr + (M + N) * K + chunk * N * 64, &maps.weight,
                        step * K + chunk * 64, weightRow + outputWidth, barriers + stage);
                }
            }
        };
        for (int s = 0; s < Stages; ++s) load(s);
        float acc[2][4] = {}, up[Gate ? 2 : 1][4] = {};
        for (int step = 0; step * K < inputWidth; ++step) {
            int stage = step % Stages;
            Moe16TmaWait(barriers + stage, (step / Stages) & 1);
            auto *a = storage + stage * StageElements, *b = a + M * K;
#pragma unroll
            for (int j = 0; j < K; j += 16) {
                unsigned ar[4], br[2];
                unsigned ap = Moe16SharedAddress(a + (j / 64) * M * 64 + Moe16Swizzle128((lane % 16) * 64 + j % 64 + (lane / 16) * 8));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" :
                    "=r"(ar[0]), "=r"(ar[1]), "=r"(ar[2]), "=r"(ar[3]) : "r"(ap) : "memory");
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    int offset = (j / 64) * N * 64 + Moe16Swizzle128((warp * 16 + n * 8 + lane % 8) * 64 + j % 64 + ((lane / 8) % 2) * 8);
                    unsigned bp = Moe16SharedAddress(b + offset);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(br[0]), "=r"(br[1]) : "r"(bp) : "memory");
                    Moe16Mma16816<T>(acc[n], ar, br);
                    if (Gate) {
                        bp = Moe16SharedAddress(b + N * K + offset);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(br[0]), "=r"(br[1]) : "r"(bp) : "memory");
                        Moe16Mma16816<T>(up[n], ar, br);
                    }
                }
            }
            // Order every consumer's generic shared reads before the async proxy
            // overwrites this stage. A CTA barrier alone is not a proxy fence.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            load(step + Stages);
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                int row = lane / 4 + (j / 2) * 8;
                int col = column + warp * 16 + n * 8 + (lane % 4) * 2, task = tasks[row];
                float values[2] = {acc[n][j], acc[n][j + 1]};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if (Gate) {
                        float g = Moe16Float(Moe16Cast<T>(values[h]));
                        float u = Moe16Float(Moe16Cast<T>(up[n][j + h]));
                        values[h] = g / (1.f + expf(-g)) * u;
                    } else if (task >= 0)
                        values[h] *= scores[task];
                }
                if (task >= 0)
                    Moe16StoreExpertPair(output + (size_t)task * outputWidth + col, values[0], values[1]);
                if (Gate)
                    Moe16StoreExpertPair(packedMiddle + ((size_t)tile * M + row) * intermediate + col,
                                          task >= 0 ? values[0] : 0.f, task >= 0 ? values[1] : 0.f);
            }
        }
        if constexpr (!Strided) break;
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int i = 0; i < Stages; ++i)
                asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(Moe16SharedAddress(barriers + i)) : "memory");
        }
        tile += gridDim.y;
    } while (tile < tiles[0]);

#endif
}
template<typename T>
bool Moe16TmaAvailable() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    static thread_local int previous = -1;
    static thread_local bool available = false;
    if (device != previous) {
        cudaFuncAttributes attributes;
        if (cudaFuncGetAttributes(&attributes, Moe16TmaExpertGemm<T, true>) != cudaSuccess) return false;
        // SM80 PTX can be JIT-compiled on SM90, but its preprocessed kernel
        // contains no TMA instructions. Check the virtual architecture too.
        available = attributes.binaryVersion / 10 == 9 && attributes.ptxVersion >= 90;
        if (available && cudaFuncSetAttribute(Moe16TmaExpertGemm<T, true, 128>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                Moe16TmaExpertSharedBytes<true, 128>()) != cudaSuccess) return false;
        previous = device;
    }
    return available;
}
template<typename T>
bool Moe16MakeExpertMap(CUtensorMap &map, const void *ptr, int inner, int outer, int tileRows, int tileK = 64) {
    cuuint64_t shape[2] = {(cuuint64_t)inner, (cuuint64_t)outer}, stride[1] = {(cuuint64_t)inner * 2};
    cuuint32_t box[2] = {(cuuint32_t)tileK, (cuuint32_t)tileRows}, steps[2] = {1, 1};
    constexpr auto type = std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    return cuTensorMapEncodeTiled(&map, type, 2, (void *)ptr, shape, stride, box, steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE, tileK == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
template<typename T>
int Moe16ExpertTileRows(int rows, int experts, int topk) {
#ifdef FASTLLM_ENABLE_MOE16_SM90
    // Larger tiles amortize WGMMA setup only when enough routes populate them.
    if (std::is_same_v<T, __nv_bfloat16> && rows >= 1024 && (size_t)rows * topk >= (size_t)experts * 32 &&
        FastllmCudaMoe16WgmmaAvailable()) return 64;
#endif
    return 16;
}
// Return before launching anything if a tensor map cannot be prepared, so the
// caller can safely use the generic grouped implementation.
template<typename T>
bool Moe16ExpertsTma(const void *input, const void *gateUp, const void *down, const int *indices,
                      const float *scores, void *middle, void *parts, void *output, int rows, int hidden,
                      int intermediate, int experts, int topk, cudaStream_t stream, int *workspace,
                      void *packed, const void *shared, const void *sharedGate, int sharedGateStride) {
    int tileRows = Moe16ExpertTileRows<T>(rows, experts, topk);
    int tiles = min(experts, rows * topk) + (rows * topk + tileRows - 1) / tileRows, device;
    const bool adaptive = tileRows == 64 &&
        (rows < 2048 || (size_t)rows * topk < (size_t)experts * 64);
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    auto *packedInput = (T *)packed;
    auto *packedMiddle = packedInput + (size_t)tiles * tileRows * hidden;
    using Key = std::tuple<int, const void *, const void *, const void *, int, int, int, int, int, int>;
    static thread_local std::map<Key, std::array<Moe16ExpertMaps, 4>> cached;
    Key key{device, packed, gateUp, down, hidden, intermediate, tiles, experts, tileRows, rows};
    auto it = cached.find(key);
    if (it == cached.end()) {
        std::array<Moe16ExpertMaps, 4> maps{};
        auto &gateMaps = maps[0], &downMaps = maps[1];
        // Match the short-reduction WGMMA dispatch without first building
        // K64 descriptors that would immediately be overwritten.
        const int downK = tileRows == 64 && intermediate <= 256 ? 32 : 64;
        if (!Moe16MakeExpertMap<T>(gateMaps.input, packedInput, hidden, tiles * tileRows, tileRows) ||
            !Moe16MakeExpertMap<T>(gateMaps.weight, gateUp, hidden, experts * intermediate * 2, 64) ||
            !Moe16MakeExpertMap<T>(downMaps.input, packedMiddle, intermediate, tiles * tileRows, tileRows, downK) ||
            !Moe16MakeExpertMap<T>(downMaps.weight, down, intermediate, experts * hidden, 64, downK)) return false;
        if (adaptive) {
            int narrowTiles = min(experts, rows * topk) + (rows * topk + 15) / 16;
            maps[2].weight = gateMaps.weight;
            // Resolve both tensor-map layouts before launching anything, so
            // an unsupported map can safely fall back to the generic path.
            if (!Moe16MakeExpertMap<T>(maps[2].input, packedInput, hidden, narrowTiles * 16, 16) ||
                !Moe16MakeExpertMap<T>(maps[3].input, packedMiddle, intermediate, narrowTiles * 16, 16) ||
                !Moe16MakeExpertMap<T>(maps[3].weight, down, intermediate, experts * hidden, 64)) return false;
        }
        // Descriptors are copied into kernel parameters at launch. Eviction
        // does not invalidate a queued launch or a captured CUDA graph.
        if (cached.size() >= 256) cached.clear();
        it = cached.emplace(key, maps).first;
    }
#ifdef FASTLLM_ENABLE_MOE16_SM90
    if (tileRows == 64) {
        Moe16BuildExpertRows(indices, workspace, rows, topk, experts, stream);
        if (adaptive) Moe16ExpertTiles<0><<<1, 256, 0, stream>>>(workspace, rows, experts);
        else Moe16ExpertTiles<64><<<1, 256, 0, stream>>>(workspace, rows, experts);
        Moe16PackExpertInput<T, 4, 64><<<dim3(16, tiles), 128, 0, stream>>>(
            (const T *)input, workspace, packedInput, rows, hidden, experts, topk);
        FastllmCudaMoe16WgmmaExperts(&it->second[0].input, &it->second[0].weight,
            &it->second[1].input, &it->second[1].weight, scores, workspace,
            middle, parts, packedMiddle, rows, hidden, intermediate, experts, tiles, stream);
    }
    if (tileRows == 16 || adaptive) {
#endif
    const Moe16ExpertMaps *gateMaps = &it->second[adaptive ? 2 : 0];
    const Moe16ExpertMaps *downMaps = &it->second[adaptive ? 3 : 1];
    if (adaptive) tiles = min(experts, rows * topk) + (rows * topk + 15) / 16;
    else Moe16BuildExpertTiles(indices, workspace, rows, topk, experts, stream);
    const bool gather = std::is_same_v<T, __nv_bfloat16> && rows >= 256 && rows <= 512 &&
        (reinterpret_cast<uintptr_t>(input) & 15) == 0;
    if (!gather) {
        if (rows >= 1024 || (rows >= 256 && hidden >= 1024))
            Moe16PackExpertInput<T, 4><<<dim3(4, tiles), 128, 0, stream>>>(
                (const T *)input, workspace, packedInput, rows, hidden, experts, topk);
        else if (rows >= 16 && rows <= 64 && hidden == 2048)
            Moe16PackExpertInput<T, 2><<<dim3(8, tiles), 256, 0, stream>>>(
                (const T *)input, workspace, packedInput, rows, hidden, experts, topk);
        else Moe16PackExpertInput<T><<<dim3(16, tiles), min(256, hidden / 8), 0, stream>>>(
                (const T *)input, workspace, packedInput, rows, hidden, experts, topk);
    }
    // Wider K amortizes setup for small captured blocks. Large prefills and
    // widths not divisible by128 retain the established64-element kernel.
    if (gather)
        Moe16TmaExpertGemm<T, true, 64, true><<<dim3(intermediate / 64, tiles), 128,
            Moe16TmaExpertSharedBytes<true, 64>(), stream>>>(*gateMaps,
            scores, workspace, (T *)middle, packedMiddle, rows, hidden, intermediate, experts, (const T *)input, topk);
    else if (rows <= 32 && hidden % 128 == 0)
        Moe16TmaExpertGemm<T, true, 128><<<dim3(intermediate / 64, tiles), 128,
            Moe16TmaExpertSharedBytes<true, 128>(), stream>>>(*gateMaps,
            scores, workspace, (T *)middle, packedMiddle, rows, hidden, intermediate, experts);
    else
        Moe16TmaExpertGemm<T, true><<<dim3(intermediate / 64, tiles), 128,
            Moe16TmaExpertSharedBytes<true, 64>(), stream>>>(*gateMaps,
            scores, workspace, (T *)middle, packedMiddle, rows, hidden, intermediate, experts);
    if (adaptive) {
        // Limit empty CTAs when wide tiles are selected. Strided processing
        // still covers every narrow tile when the fallback is active.
        constexpr int MaxFallbackTiles = 256;
        Moe16TmaExpertGemm<T, false, 64, false, true><<<dim3(hidden / 64, min(tiles, MaxFallbackTiles)), 128,
            Moe16TmaExpertSharedBytes<false, 64>(), stream>>>(*downMaps,
            scores, workspace, (T *)parts, nullptr, rows, hidden, intermediate, experts);
    } else
        Moe16TmaExpertGemm<T, false><<<dim3(hidden / 64, tiles), 128,
            Moe16TmaExpertSharedBytes<false, 64>(), stream>>>(*downMaps,
            scores, workspace, (T *)parts, nullptr, rows, hidden, intermediate, experts);
#ifdef FASTLLM_ENABLE_MOE16_SM90
    }
#endif
    if (shared)
        Moe16SumExperts<T, true>((const T *)parts, indices, (T *)output, rows, hidden,
                               topk, stream, (const T *)shared,
                               (const T *)sharedGate, sharedGateStride);
    else
        Moe16SumExperts<T, false>((const T *)parts, indices, (T *)output, rows, hidden,
                                topk, stream, nullptr, nullptr, 1);
    return true;
}
}
#endif

std::size_t FastllmCudaMoe16PackedBytes(int rows, int hidden, int intermediate, int experts, int topk, bool fp16) {
#if CUDART_VERSION >= 12000
    // The caller chooses TMA for prefill or a captured small-block MoE.
    // Small eager blocks retain their existing low-latency path.
    if (rows < 4 || hidden < 512 || hidden % 64 || intermediate < 128 || intermediate % 64 ||
        experts < 1 || experts > 256 || topk < 1 || topk > 16 || topk > experts ||
        rows > 65535 / topk || (fp16 ? !Moe16TmaAvailable<__half>() : !Moe16TmaAvailable<__nv_bfloat16>())) return 0;
    int tileRows = fp16 ? Moe16ExpertTileRows<__half>(rows, experts, topk)
                        : Moe16ExpertTileRows<__nv_bfloat16>(rows, experts, topk);
    size_t tiles = min(experts, rows * topk) + (rows * topk + tileRows - 1) / tileRows;
    return tiles * tileRows * ((size_t)hidden + intermediate) * sizeof(uint16_t);
#else
    return 0;
#endif
}
