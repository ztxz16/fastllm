#include "devices/cuda/fastllm-nvfp4-prefill.h"
//
// BFloat16 Linear: 小规模时使用自定义 GEMV kernel，大规模时使用 cublas (仿照 fastllm-linear-fp16.cu)
//

#include "devices/cuda/fastllm-cuda-pdl.cuh"
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda-linear-output.h"
#include "fastllm-bf16-lt.cuh"
#ifndef USE_ROCM
#include "devices/cuda/fastllm-cuda-launch.cuh"
#endif
#include <stdexcept>
#include "../fastllm-hybrid-prefill.cuh"

#ifdef __CUDACC__
#include <cuda_bf16.h>
#endif

#ifndef FASTLLM_HAS_BF16_MMA_SM120
const FastllmBf16MmaProjections *FastllmCudaGetBf16MmaProjections() { return nullptr; }
#endif

bool FastllmCudaBf16ProjectionGemm(const void *input, const void *weight, void *output,
                                   int rows, int width, int columns, int groups) {
#if !defined(USE_ROCM) && CUBLAS_VERSION >= 11000
    if (!input || !weight || !output || rows <= 0 || width <= 0 || columns <= 0 || groups <= 0) return false;
    auto handle = getFastllmCublasFp32ReductionHandle();
    if (!handle) return false;
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t status;
    if (groups == 1) {
        status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, width,
            &alpha, weight, CUDA_R_16BF, width, input, CUDA_R_16BF, width, &beta,
            output, CUDA_R_16BF, columns, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    } else {
        status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, width,
            &alpha, weight, CUDA_R_16BF, width, int64_t(columns) * width,
            input, CUDA_R_16BF, width, int64_t(rows) * width, &beta,
            output, CUDA_R_16BF, columns, int64_t(rows) * columns, groups,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) return false;
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("BF16 projection GEMM failed (cuBLAS status " + std::to_string(status) + ")");
    return true;
#else
    return false;
#endif
}

bool FastllmCudaBf16HeuristicGemm(const void *input, const void *weight, void *output,
                                   int rows, int width, int columns) {
#if !defined(USE_ROCM) && CUBLAS_VERSION >= 11000
    if (fastllm_bf16_lt::Matmul(input, weight, output, rows, width, columns, CUDA_R_16BF, 0, true)) return true;
#endif
    return FastllmCudaBf16ProjectionGemm(input, weight, output, rows, width, columns, 1);
}

FastllmCudaLinearGraphScope::FastllmCudaLinearGraphScope(std::shared_ptr<void> &workspace) {
    if (!workspace) workspace = fastllm_bf16_lt::CreateSerialGraphWorkspace();
    auto &current = fastllm_bf16_lt::SerialGraphWorkspace();
    previous = std::move(current);
    current = std::static_pointer_cast<fastllm_bf16_lt::Workspace>(workspace);
    // Warmup can use eager Lt even if the graph lease pool is full. Capturing
    // a different GEMV/GEMM fallback would then change its reduction order.
    if (!current && FastllmCudaGraphIsCapturingFast()) FastllmCudaSetThreadError();
}

FastllmCudaLinearGraphScope::~FastllmCudaLinearGraphScope() {
    fastllm_bf16_lt::SerialGraphWorkspace() =
        std::static_pointer_cast<fastllm_bf16_lt::Workspace>(std::move(previous));
}

typedef union __align__(16) _union_bf16_4 {
    uint2 in;
    __nv_bfloat16 out[4];
    __nv_bfloat162 out2[2];
} union_bf16_4;

typedef union __align__(16) _union_bf16_8 {
    uint4 in;
    __nv_bfloat16 out[8];
    __nv_bfloat162 out2[4];
} union_bf16_8;

template <int THREAD_PER_BLOCK, int PART, bool GRID_BATCH = false, bool Pdl = false>
__global__ void FastllmGemvBf16Bf16Kernel2MultiRow(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, __nv_bfloat16 *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    // Large exact batches use grid.y for independent rows. Small batches
    // share each weight load across PART rows with the same per-row sums
    // and reduction tree as one-row decode.
    if constexpr (PART == 1 || GRID_BATCH) {
        const size_t gridRow = (size_t)blockIdx.y * PART;
        A += gridRow * m;
        C += gridRow * k;
    }
    union_bf16_8 regA;
    union_bf16_8 regB;

    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const __nv_bfloat16 *baseB = B + p * m;
    // Weight-first PDL scheduling follows Basalt v0.5.6 native_mmvq.cu (third_party/basalt/LICENSE).
    // A single decode row has at most two vector loads per lane in this variant.
    // Weights are immutable. Wait before reading the activation or writing output.
    uint4 weights[2];
    if constexpr (Pdl) {
        static_assert(!GRID_BATCH && (PART == 1 || PART == 7));
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
            const int col = (tile * THREAD_PER_BLOCK + tid) * 8;
            if (m % 8 == 0 && col < m) {
                if constexpr (PART == 1) weights[tile] = *reinterpret_cast<const uint4 *>(baseB + col);
                else weights[tile] = fastllm::cuda_pdl::ReadWeight(baseB + col);
            }
        }
#ifndef USE_ROCM
        if constexpr (PART == 1) fastllm::cuda_launch::DependencyStart();
        else fastllm::cuda_pdl::Wait();
#endif
    }
    if (m % 8 == 0) {
#pragma unroll
        for (int i = tid * 8; i < m; i += THREAD_PER_BLOCK * 8) {
            if constexpr (Pdl) {
                const int tile = i / (THREAD_PER_BLOCK * 8);
                regB.in = tile < 2 ? weights[tile] : *reinterpret_cast<const uint4 *>(baseB + i);
            } else regB.in = *reinterpret_cast<const uint4 *>(baseB + i);
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint4 *>(A + x * m + i);
                float sum = 0.0f;
                if (i < m)
                    sum += __bfloat162float(regA.out2[0].x) * __bfloat162float(regB.out2[0].x);
                if (i + 1 < m)
                    sum += __bfloat162float(regA.out2[0].y) * __bfloat162float(regB.out2[0].y);
                if (i + 2 < m)
                    sum += __bfloat162float(regA.out2[1].x) * __bfloat162float(regB.out2[1].x);
                if (i + 3 < m)
                    sum += __bfloat162float(regA.out2[1].y) * __bfloat162float(regB.out2[1].y);
                if (i + 4 < m)
                    sum += __bfloat162float(regA.out2[2].x) * __bfloat162float(regB.out2[2].x);
                if (i + 5 < m)
                    sum += __bfloat162float(regA.out2[2].y) * __bfloat162float(regB.out2[2].y);
                if (i + 6 < m)
                    sum += __bfloat162float(regA.out2[3].x) * __bfloat162float(regB.out2[3].x);
                if (i + 7 < m)
                    sum += __bfloat162float(regA.out2[3].y) * __bfloat162float(regB.out2[3].y);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += __bfloat162float(A[i + x * m]) * __bfloat162float(baseB[i]);
            }
        }
    }
    __syncthreads();
    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) diff[x] = 0.0f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
    // Hopper single-row decode can use the same compensated warp tree as
    // small batches; no product, reduction order or BF16 rounding changes.
    constexpr bool warpReduce = true;
#else
    constexpr bool warpReduce = PART > 1;
#endif
    for (unsigned int s = THREAD_PER_BLOCK / 2; s >= (warpReduce ? 32 : 1); s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff[x];
                float sumTmp = sdata[x][tid] + other;
                diff[x] = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if constexpr (warpReduce) {
        // Complete the same compensated binary tree within warp zero. The
        // 128/64/32 stages above preserve both each lane's sum and correction;
        // shuffles replace only the remaining shared-memory exchanges.
        if (tid < 32) {
            #pragma unroll
            for (int x = 0; x < PART; ++x) {
                float value = sdata[x][tid], correction = diff[x];
                #pragma unroll
                for (int step = 16; step; step >>= 1) {
                    float peer = __shfl_down_sync(0xffffffffu, value, step);
                    if (tid < step) {
                        float other = peer - correction;
                        float sum = value + other;
                        correction = (sum - value) - other;
                        value = sum;
                    }
                }
                if (tid == 0) sdata[x][0] = value;
            }
        }
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = __float2bfloat16_rn(sdata[x][0] + __bfloat162float(bias[p]));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = __float2bfloat16_rn(sdata[x][0]);
        }
    }
    __syncthreads();
}

#ifndef USE_ROCM
void *FastllmCudaV41IndexHeadGemvKernel() {
    return (void *)FastllmGemvBf16Bf16Kernel2MultiRow<256, 6>;
}
#endif

// Selected during model initialization, outside CUDA graph capture.
FastllmBf16MmaProjections::Project FastllmCudaGetPdlBf16Gemv() {
#if !defined(USE_ROCM) && CUDART_VERSION >= 12080
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, FastllmGemvBf16Bf16Kernel2MultiRow<256, 1, false, true>) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    if (attributes.ptxVersion >= 90 && attributes.binaryVersion >= 90)
        return [](const void *input, const void *weight, void *output, int rows, int width, int columns) {
            if (rows != 1 || width <= 256 || width > 4096 || width % 8 || columns < 1 ||
                ((reinterpret_cast<uintptr_t>(weight) | reinterpret_cast<uintptr_t>(input)) & 15)) return false;
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t config{};
            config.gridDim = columns;
            config.blockDim = 256;
            config.stream = cudaStreamPerThread;
            config.attrs = &attr;
            config.numAttrs = 1;
            auto status = cudaLaunchKernelEx(&config, FastllmGemvBf16Bf16Kernel2MultiRow<256, 1, false, true>,
                (__nv_bfloat16 *)input, (__nv_bfloat16 *)weight, (__nv_bfloat16 *)output, (__nv_bfloat16 *)nullptr, width, columns);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            return true;
        };
#endif
    return nullptr;
}

// The scalar reference has 256 leaves. Each leaf sums two eight-product
// groups, then a compensated binary tree reduces the 256 partials to BF16.
__device__ __forceinline__ void FastllmBf16HeadAdd(float &value, float &correction, float peer) {
    const float add = peer - correction;
    const float next = value + add;
    correction = (next - value) - add;
    value = next;
}

// One 64-thread group per vocabulary row. A lane owns four reference leaves
// spaced 64 apart. Keep their partials in registers and share every weight
// load across the input rows. Only the 64-to-32 stage needs shared memory.
template <int FixedWidth, int batch, bool Tail = false, bool Pdl = false, bool Capped = false>
__device__ __forceinline__ void FastllmBf16ExactTile(const __nv_bfloat16 *__restrict__ input,
        const __nv_bfloat16 *__restrict__ weight, std::conditional_t<Capped, float, __nv_bfloat16> *__restrict__ output, int vocab,
        int runtimeWidth, int firstColumn, int live = batch, float cap = 0.f) {
    const int width = FixedWidth ? FixedWidth : runtimeWidth;
    const int lane = threadIdx.x % 64, group = threadIdx.x / 64;
    const int column = firstColumn + group;
    const __nv_bfloat16 *row = weight + size_t(column) * width;
    uint4 staged[Pdl ? 8 : 1];
#ifndef USE_ROCM
    if constexpr (Pdl && batch == 1) {
        static_assert(batch == 1 && FixedWidth > 0 && FixedWidth <= 4096);
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int offset = (slot / 4) * 2048 + (lane + 64 * (slot % 4)) * 8;
            if (offset < FixedWidth) staged[slot] = *reinterpret_cast<const uint4 *>(row + offset);
        }
        fastllm::cuda_launch::DependencyStart();
    }
#endif
    uint4 firstWeight;
    if constexpr (Pdl && batch != 1) {
        if (lane * 8 < width) firstWeight = fastllm::cuda_pdl::ReadWeight(row + lane * 8);
        fastllm::cuda_pdl::Wait();
    }
    float sums[4][batch], corrections[4][batch] = {};
    #pragma unroll
    for (int leaf = 0; leaf < 4; ++leaf) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int offset = half * 2048 + (lane + 64 * leaf) * 8;
            if constexpr (FixedWidth != 4096)
                if (offset >= width) continue;
            union_bf16_8 w;
            if constexpr (Pdl && batch == 1) w.in = staged[half * 4 + leaf];
            else if constexpr (Pdl) {
                w.in = (leaf == 0 && half == 0) ? firstWeight : *reinterpret_cast<const uint4 *>(row + offset);
            } else w.in = *reinterpret_cast<const uint4 *>(row + offset);
            float wfp32[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) wfp32[i] = __bfloat162float(w.out[i]);
            #pragma unroll
            for (int b = 0; b < batch; ++b) {
                union_bf16_8 x;
                x.in = *reinterpret_cast<const uint4 *>(input + (Tail ? min(b, live - 1) : b) * width + offset);
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    partial = fmaf(__bfloat162float(x.out[i]), wfp32[i], partial);
                if (half == 0) sums[leaf][b] = partial;
                else sums[leaf][b] += partial;
            }
        }
    }
    __shared__ float upper[2][batch][32];
    #pragma unroll
    for (int b = 0; b < batch; ++b) {
        FastllmBf16HeadAdd(sums[0][b], corrections[0][b], sums[2][b]);
        FastllmBf16HeadAdd(sums[1][b], corrections[1][b], sums[3][b]);
        FastllmBf16HeadAdd(sums[0][b], corrections[0][b], sums[1][b]);
        if (lane >= 32) upper[group][b][lane - 32] = sums[0][b];
    }
    __syncthreads();
    if (lane < 32) {
        #pragma unroll
        for (int b = 0; b < batch; ++b) {
            float value = sums[0][b], correction = corrections[0][b];
            FastllmBf16HeadAdd(value, correction, upper[group][b][lane]);
            #pragma unroll
            for (int step = 16; step; step >>= 1)
                FastllmBf16HeadAdd(value, correction, __shfl_down_sync(0xffffffffu, value, step));
            if (lane == 0 && (!Tail || b < live)) {
                const auto rounded = __float2bfloat16_rn(value);
                if constexpr (Capped) output[column] = cap * tanhf(__bfloat162float(rounded) / cap);
                else output[size_t(b) * vocab + column] = rounded;
            }
        }
    }
}

template <int FixedWidth, int batch, bool Tail = false, bool Early = false>
__global__ void FastllmBf16ExactProjection(const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
        __nv_bfloat16 *output, int vocab, int width = FixedWidth, int live = batch) {
    FastllmBf16ExactTile<FixedWidth, batch, Tail, Early>(input + size_t(blockIdx.y) * batch * width, weight,
        output + size_t(blockIdx.y) * batch * vocab, vocab, width, blockIdx.x * 2, live);
}

// Packed weights can project separate Q, K and V activations in one grid.
// Each group retains its own row pitch, including when Q is wider than K/V.
struct FastllmBf16ProjectionInputs {
    const __nv_bfloat16 *x[3];
    int ends[3];
};

template <int Batch, int FixedWidth = 0>
__global__ void FastllmBf16GroupedProjection(FastllmBf16ProjectionInputs inputs,
        const __nv_bfloat16 *weight, __nv_bfloat16 *output, int rows, int width, int firstRow) {
    const int column = blockIdx.x * 2;
    const int group = (column >= inputs.ends[0]) + (column >= inputs.ends[1]);
    const int begin = group ? inputs.ends[group - 1] : 0;
    const int columns = inputs.ends[group] - begin;
    const int row = firstRow + blockIdx.y * Batch;
    FastllmBf16ExactTile<FixedWidth, Batch>(inputs.x[group] + size_t(row) * width, weight + size_t(begin) * width,
        output + size_t(begin) * rows + size_t(row) * columns, columns, width, column - begin);
}

#if !defined(USE_ROCM) && CUDART_VERSION >= 12080
// One-row QKV segments and the vocabulary head retain the BF16 GEMV reduction
// and its final BF16 rounding. Adjacent columns share one CTA.
template<bool Capped>
__global__ void FastllmBf16PdlProjection(FastllmBf16ProjectionInputs in, const __nv_bfloat16 *weight,
        std::conditional_t<Capped, float, __nv_bfloat16> *out, float cap) {
    const int column = blockIdx.x * 2;
    const int group = Capped ? 0 : (column >= in.ends[0]) + (column >= in.ends[1]);
    FastllmBf16ExactTile<2304, 1, false, true, Capped>(in.x[group], weight, out,
        in.ends[2], 2304, column, 1, cap);
}
#endif

bool FastllmCudaBf16PdlProjection(const void *q, const void *k, const void *v, const void *weights,
        void *out, int width, int qEnd, int kEnd, int columns, float cap) {
#if !defined(USE_ROCM) && CUDART_VERSION >= 12080
    if (width != 2304 || qEnd <= 0 || kEnd < qEnd || columns < kEnd ||
        ((qEnd | kEnd | columns) & 1) ||
        ((reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
          reinterpret_cast<uintptr_t>(v) | reinterpret_cast<uintptr_t>(weights)) & 15)) return false;
    const FastllmBf16ProjectionInputs in{{static_cast<const __nv_bfloat16 *>(q),
        static_cast<const __nv_bfloat16 *>(k), static_cast<const __nv_bfloat16 *>(v)}, {qEnd, kEnd, columns}};
    const auto *w = static_cast<const __nv_bfloat16 *>(weights);
    if (cap > 0.f)
        return fastllm::cuda_launch::Launch<FastllmBf16PdlProjection<true>>(
            columns / 2, 128, 0, cudaStreamPerThread, in, w, static_cast<float *>(out), cap);
    return fastllm::cuda_launch::Launch<FastllmBf16PdlProjection<false>>(
        columns / 2, 128, 0, cudaStreamPerThread, in, w, static_cast<__nv_bfloat16 *>(out), 0.f);
#else
    return false;
#endif
}

#include "devices/cuda/fastllm-bf16-warp-dot.cuh"

struct FastllmBf16ProjectionFork {
    cudaStream_t stream = nullptr;
    cudaEvent_t inputReady = nullptr, outputReady = nullptr;
    ~FastllmBf16ProjectionFork() {
        if (inputReady) cudaEventDestroy(inputReady);
        if (outputReady) cudaEventDestroy(outputReady);
        if (stream) cudaStreamDestroy(stream);
    }
};

bool FastllmCudaBf16ParallelProjection(const void *input, const void *weight, void *output,
        int rows, int width, int columns, std::shared_ptr<FastllmBf16ProjectionFork> &storage, const std::function<void()> &siblings) {
#if !defined(USE_ROCM) && CUBLAS_VERSION >= 11000
    using namespace fastllm_bf16_lt;
    int device;
    cudaStreamCaptureStatus capture;
    CheckCuda(cudaGetDevice(&device));
    CheckCuda(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
    auto *state = GetState(device, capture != cudaStreamCaptureStatusNone, false);
    if (!state) return false;
    const Key key{rows, columns, width, Alignment(weight), Alignment(input), Alignment(output), CUDA_R_16BF, 0};
    auto it = state->plans.find(key);
    if (it == state->plans.end() || !it->second->valid) return false;
    auto &p = *it->second;
    cublasLtMatmulHeuristicResult_t checked{};
    if (!CheckBlas(cublasLtMatmulAlgoCheck(state->handle, p.op, p.a, p.b, p.c, p.c, &p.algo, &checked)) ||
        checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize != 0) return false;
    if (!storage) {
        if (capture != cudaStreamCaptureStatusNone) return false;
        auto fork = std::make_shared<FastllmBf16ProjectionFork>();
        CheckCuda(cudaStreamCreateWithFlags(&fork->stream, cudaStreamNonBlocking));
        CheckCuda(cudaEventCreateWithFlags(&fork->inputReady, cudaEventDisableTiming));
        CheckCuda(cudaEventCreateWithFlags(&fork->outputReady, cudaEventDisableTiming));
        storage = fork;
    }
    auto &fork = *storage;
    CheckCuda(cudaEventRecord(fork.inputReady, cudaStreamPerThread));
    CheckCuda(cudaStreamWaitEvent(fork.stream, fork.inputReady));
    bool ok = false;
    try {
        ok = Run(*state, p, p.algo, weight, input, output, nullptr, fork.stream);
        if (ok) siblings();
    } catch (...) {
        cudaEventRecord(fork.outputReady, fork.stream);
        cudaStreamWaitEvent(cudaStreamPerThread, fork.outputReady);
        throw;
    }
    CheckCuda(cudaEventRecord(fork.outputReady, fork.stream));
    CheckCuda(cudaStreamWaitEvent(cudaStreamPerThread, fork.outputReady));
    return ok;
#else
    return false;
#endif
}

__device__ __forceinline__ float SmallKDot(const union_bf16_8 &a, const union_bf16_8 &b, int lane) {
    float value = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        value += __bfloat162float(a.out[i]) * __bfloat162float(b.out[i]);
    float correction = 0.0f;
    #pragma unroll
    for (int step = 16; step; step >>= 1) {
        float peer = __shfl_down_sync(0xffffffffu, value, step);
        if (lane < step) {
            float other = peer - correction;
            float sum = value + other;
            correction = (sum - value) - other;
            value = sum;
        }
    }
    return value;
}

// At K=256, one warp computes each dot with the original compensated
// reduction. Tiling output rows reuses its input vector without changing
// the per-lane products or BF16 rounding.
template <int RowsPerWarp = 1, bool Indexed = false>
__global__ void FastllmGemvBf16SmallK(const __nv_bfloat16 *input,
        const __nv_bfloat16 *weight, __nv_bfloat16 *output,
        const __nv_bfloat16 *bias, int outputs, const float *ids = nullptr) {
    if constexpr (Indexed) {
        const int row = __float2int_rz(*ids);
        input += (size_t)(row >= 0 && row < outputs ? row : 0) * 256;
    }
    FastllmBf16WarpDot256<RowsPerWarp>(input, weight, outputs, []{}, [&](int id, float value) {
        if (bias) value += __bfloat162float(bias[id]);
        output[id] = __float2bfloat16_rn(value);
    });
}

namespace {
constexpr int MarkovSlotBits = 8;
constexpr int MarkovSlots = 1 << MarkovSlotBits;
constexpr int MarkovTile = 256;

// One block owns each tile's tag and data. Launches use the same stream
// and lifetime as the proposal graph's IDs and BF16 output workspace.
struct MarkovCache {
    int device, vocab, tiles;
    const void *embedding, *projection;
    int *tags = nullptr;
    __nv_bfloat16 *values = nullptr;
    ~MarkovCache() {
        if (!tags) return;
        int previous;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cudaFree(tags);
        cudaSetDevice(previous);
    }
};

__global__ void DraftMarkovCached(const __nv_bfloat16 *table, const __nv_bfloat16 *weight,
        const float *ids, __nv_bfloat16 *out, int vocab, int tiles,
        int *tags, __nv_bfloat16 *cached) {
    int token = __float2int_rz(*ids);
    if (token < 0 || token >= vocab) token = 0;
    const unsigned slot = (unsigned(token) * 2654435761u) >> (32 - MarkovSlotBits);
    const int tile = blockIdx.x, first = tile * MarkovTile;
    auto *entry = cached + size_t(slot) * vocab;
    __shared__ int hit;
    if (threadIdx.x == 0) {
        int &tag = tags[slot * tiles + tile];
        hit = tag == token + 1;
        tag = token + 1;
    }
    __syncthreads();
    if (hit) {
        const int i = first + threadIdx.x;
        if (i < vocab) out[i] = entry[i];
        return;
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    union_bf16_8 a;
    a.in = *reinterpret_cast<const uint4 *>(table + size_t(token) * 256 + lane * 8);
    #pragma unroll
    for (int j = 0; j < MarkovTile; j += 8) {
        const int i = first + j + warp;
        if (i < vocab) {
            union_bf16_8 b;
            b.in = *reinterpret_cast<const uint4 *>(weight + size_t(i) * 256 + lane * 8);
            const float v = SmallKDot(a, b, lane);
            if (lane == 0) out[i] = entry[i] = __float2bfloat16_rn(v);
        }
    }
}
} // namespace

std::shared_ptr<void> FastllmCudaCreateDraftMarkovCache(const fastllm::Data &table, const fastllm::Data &w) {
    int device;
    if (table.dataType != fastllm::BFLOAT16 || w.dataType != fastllm::BFLOAT16 ||
        !table.cudaData || !w.cudaData || table.dims != w.dims || w.dims.size() != 2 || w.dims[1] != 256 ||
        w.dims[0] <= 0 || w.dims[0] > 262144 || FastllmCudaRuntimeArch() != 120 ||
        cudaGetDevice(&device) != cudaSuccess || FastllmCudaGraphIsCapturingFast()) return {};
    auto c = std::make_shared<MarkovCache>();
    c->device = device;
    c->embedding = table.cudaData;
    c->projection = w.cudaData;
    c->vocab = w.dims[0];
    c->tiles = (c->vocab + MarkovTile - 1) / MarkovTile;
    size_t tagBytes = size_t(MarkovSlots) * c->tiles * sizeof(int);
    size_t bytes = tagBytes + size_t(MarkovSlots) * c->vocab * sizeof(__nv_bfloat16);
    if (cudaMalloc(&c->tags, bytes) != cudaSuccess ||
        cudaMemsetAsync(c->tags, 0, tagBytes, cudaStreamPerThread) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    c->values = reinterpret_cast<__nv_bfloat16 *>(reinterpret_cast<char *>(c->tags) + tagBytes);
    return c;
}
bool FastllmCudaDraftIndexedBf16Gemv(const fastllm::Data &ids, const fastllm::Data &table,
        const fastllm::Data &weight, fastllm::Data &output, void *cache) {
    using namespace fastllm;
    if (ids.dataType != FLOAT32 || ids.dims != std::vector<int>{1, 1} ||
        table.dataType != BFLOAT16 || weight.dataType != BFLOAT16 || output.dataType != BFLOAT16 ||
        table.dims.size() != 2 || table.dims[1] != 256 || table.dims != weight.dims ||
        table.strides != std::vector<uint64_t>{256, 1} || weight.strides != table.strides ||
        table.dims[0] < 1 || output.dims != std::vector<int>{1, 1, table.dims[0]} ||
        output.strides.size() != 3 || output.strides.back() != 1 ||
        ids.dataDevice != CUDA || table.dataDevice != CUDA || weight.dataDevice != CUDA || output.dataDevice != CUDA ||
        ids.dataDeviceIds.size() != 1 || ids.dataDeviceIds.front() != FastllmCudaGetDevice() ||
        ids.dataDeviceIds != table.dataDeviceIds || ids.dataDeviceIds != weight.dataDeviceIds ||
        ids.dataDeviceIds != output.dataDeviceIds || !ids.cudaData || !table.cudaData || !weight.cudaData ||
        !output.cudaData || ((uintptr_t(table.cudaData) | uintptr_t(weight.cudaData)) & 15)) return false;

    const int rows = table.dims[0];
    auto *c = static_cast<MarkovCache *>(cache);
    if (c && c->vocab == rows && c->embedding == table.cudaData && c->projection == weight.cudaData &&
        c->device == table.dataDeviceIds.front()) {
        DraftMarkovCached<<<c->tiles, 256, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(table.cudaData), static_cast<const __nv_bfloat16 *>(weight.cudaData),
            static_cast<const float *>(ids.cudaData), static_cast<__nv_bfloat16 *>(output.cudaData),
            rows, c->tiles, c->tags, c->values);
        return true;
    }
    FastllmGemvBf16SmallK<1, true><<<(rows - 1) / 8 + 1, 256>>>(
        (const __nv_bfloat16 *)table.cudaData, (const __nv_bfloat16 *)weight.cudaData,
        (__nv_bfloat16 *)output.cudaData, nullptr, rows, (const float *)ids.cudaData);
    return true;
}

bool FastllmCudaIndexedBf16Gemv(const fastllm::Data &ids, const fastllm::Data &table,
        const fastllm::Data &weight, fastllm::Data &output) {
    return FastllmCudaDraftIndexedBf16Gemv(ids, table, weight, output, nullptr);
}

// FP16 input × BF16 weight -> FP16 output (用于 FastllmCudaHalfMatMulBFloat16)
template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Bf16Kernel2MultiRow(half *A, __nv_bfloat16 *B, half *C, half *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    union_half4 regA;
    union_bf16_4 regB;

    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const __nv_bfloat16 *baseB = B + p * m;
    if (m % 4 == 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint2 *>(A + i + x * m);
                regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += __low2float(regA.out2[0]) * __bfloat162float(regB.out2[0].x);
                if (i + 1 < m)
                    sum += __high2float(regA.out2[0]) * __bfloat162float(regB.out2[0].y);
                if (i + 2 < m)
                    sum += __low2float(regA.out2[1]) * __bfloat162float(regB.out2[1].x);
                if (i + 3 < m)
                    sum += __high2float(regA.out2[1]) * __bfloat162float(regB.out2[1].y);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += __half2float(A[i + x * m]) * __bfloat162float(baseB[i]);
            }
        }
    }
    __syncthreads();
    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) diff[x] = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff[x];
                float sumTmp = sdata[x][tid] + other;
                diff[x] = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = __float2half_rn(sdata[x][0] + __half2float(__ldg(bias + p)));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++)
                C[p + k * x] = __float2half_rn(sdata[x][0]);
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__device__ __forceinline__ void FastllmGemvFp32Bf16Rows(float *A, const __nv_bfloat16 *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    float4 regA;
    union_bf16_4 regB;

    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const __nv_bfloat16 *baseB = B + p * m;
    if (m % 4 == 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA = *reinterpret_cast<const float4 *>(A + i + x * m);
                regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += regA.x * __bfloat162float(regB.out2[0].x);
                if (i + 1 < m)
                    sum += regA.y * __bfloat162float(regB.out2[0].y);
                if (i + 2 < m)
                    sum += regA.z * __bfloat162float(regB.out2[1].x);
                if (i + 3 < m)
                    sum += regA.w * __bfloat162float(regB.out2[1].y);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += A[i + x * m] * __bfloat162float(baseB[i]);
            }
        }
    }
    __syncthreads();
    float diff[PART];
#pragma unroll
    for (int x = 0; x < PART; x++) diff[x] = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff[x];
                float sumTmp = sdata[x][tid] + other;
                diff[x] = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0];
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0] + __ldg(bias + p);
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Bf16Kernel2MultiRow(float *A, __nv_bfloat16 *B, float *C, float *bias, int m, int k) {
    // Exact speculative batches retain the ordinary one-row reduction tree.
    if constexpr (PART == 1) {
        A += (size_t)blockIdx.y * m;
        C += (size_t)blockIdx.y * k;
    }
    FastllmGemvFp32Bf16Rows<THREAD_PER_BLOCK, PART>(A, B, C, bias, m, k);
}

// By-value pointer chunks need no device registry or host route readback.
// They fit the 4 KiB kernel argument limit on older CUDA architectures and
// are owned by the graph node when captured. Only resident weights qualify.
struct FastllmBf16ExpertPointers {
    const __nv_bfloat16 *weights[256];
};

template <bool DOWN>
__global__ void FastllmMoeFp32Bf16IndexedKernel(
        float *input, float *output, const int32_t *indices,
        FastllmBf16ExpertPointers table, int first, int count,
        int topk, int m, int k) {
    const int task = blockIdx.y;
    const int expert = indices[task] - first;
    if (expert < 0 || expert >= count) return;
    const int row = DOWN ? task : task / topk;
    FastllmGemvFp32Bf16Rows<256, 1>(input + (size_t)row * m,
        table.weights[expert], output + (size_t)task * k, nullptr, m, k);
}

__device__ __forceinline__ float FastllmMoeFp32Bf16Dot4(
        const float *input, const __nv_bfloat16 *weight) {
    const float4 a = *reinterpret_cast<const float4 *>(input);
    union_bf16_4 b;
    b.in = *reinterpret_cast<const uint2 *>(weight);
    float sum = 0.0f;
    sum += a.x * __bfloat162float(b.out2[0].x);
    sum += a.y * __bfloat162float(b.out2[0].y);
    sum += a.z * __bfloat162float(b.out2[1].x);
    sum += a.w * __bfloat162float(b.out2[1].y);
    return sum;
}

template <int INTER>
__global__ void FastllmMoeFp32Bf16DownWarpKernel(
        const float *input, float *output, const int32_t *indices,
        FastllmBf16ExpertPointers table, int first, int count, int hidden) {
    const int task = blockIdx.y, expert = indices[task] - first;
    const int col = blockIdx.x * 8 + threadIdx.x / 32;
    if (expert < 0 || expert >= count || col >= hidden) return;
    const int lane = threadIdx.x % 32;
    input += (size_t)task * INTER;
    const __nv_bfloat16 *weight = table.weights[expert] + (size_t)col * INTER;
    float sum = FastllmMoeFp32Bf16Dot4(input + lane * 4, weight + lane * 4);
    float diff = 0.0f;
    // Only 32/64 lanes of the generic 256-thread reduction contain data.
    // Retain its compensated tree, including the upper 32 partial sums for
    // INTER=256, while assigning eight independent output columns per CTA.
    if constexpr (INTER == 256) {
        const float other = FastllmMoeFp32Bf16Dot4(
            input + (lane + 32) * 4, weight + (lane + 32) * 4);
        const float next = sum + other;
        diff = (next - sum) - other;
        sum = next;
    }
    for (int step = 16; step > 0; step >>= 1) {
        const float peer = __shfl_down_sync(0xffffffffu, sum, step);
        if (lane < step) {
            const float other = peer - diff;
            const float next = sum + other;
            diff = (next - sum) - other;
            sum = next;
        }
    }
    if (lane == 0) output[(size_t)task * hidden + col] = sum;
}

__global__ void FastllmMoeFp32Bf16ReduceKernel(
        const float *parts, const int32_t *indices, const float *scores,
        float *output, int hidden, int topk, int experts, bool singleRow) {
    __shared__ int order[16];
    const int row = blockIdx.y;
    indices += row * topk;
    scores += row * topk;
    if (threadIdx.x == 0) {
        for (int i = 0; i < topk; ++i) {
            int j = i;
            // The multi-row fallback visits experts in ascending id order;
            // single-row decode instead follows the router's top-k order.
            while (!singleRow && j > 0 && indices[order[j - 1]] > indices[i]) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = i;
        }
    }
    __syncthreads();
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden) return;
    float sum = 0.0f;
    bool first = true;
    for (int j = 0; j < topk; ++j) {
        const int route = order[j];
        if (indices[route] < 0 || indices[route] >= experts) continue;
        const float value = parts[((size_t)row * topk + route) * hidden + col];
        sum = singleRow && first ? value * scores[route]
            : fmaf(value, scores[route], sum);
        first = false;
    }
    output[(size_t)row * hidden + col] = sum;
}

bool FastllmCudaFloat32MergeMOEBFloat16Indexed(
        const fastllm::Data &input, const fastllm::Data &index,
        const fastllm::Data &score, fastllm::Data &gate,
        fastllm::Data &middle, fastllm::Data &parts, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch) {
    using namespace fastllm;
    if (input.dataType != FLOAT32 || input.dims.size() != 2 ||
        input.dims[0] < 1 || input.dims[0] > 4 || input.dims[1] <= 0 ||
        index.dataType != INT32 || score.dataType != FLOAT32 ||
        index.dims.size() != 2 || index.dims[0] != input.dims[0] ||
        index.dims[1] < 1 || index.dims[1] > 16 || score.dims != index.dims ||
        weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) ||
        weights[0] != nullptr || weights[1] != nullptr || weights[2] == nullptr ||
        weights[2]->dims.size() != 2 || weights[2]->dims[0] <= 0 ||
        (weights[2]->dims[0] & 1)) return false;
    const int device = FastllmCudaGetDevice();
    auto resident = [device](const Data &data) {
        return data.dataDevice == DataDevice::CUDA && data.cudaData != nullptr &&
            !data.multiDeviceData && !data.lockInCPU && !data.isDiskWeight &&
            (data.dataDeviceIds.empty() ||
             (data.dataDeviceIds.size() == 1 && data.dataDeviceIds[0] == device));
    };
    if (!resident(input) || !resident(index) || !resident(score)) return false;
    const int batch = input.dims[0], hidden = input.dims[1];
    const int topk = index.dims[1], inter = weights[2]->dims[0] / 2;
    const int experts = weightsBatch / 2 - 1;
    std::vector<FastllmBf16ExpertPointers> gateTables((experts + 255) / 256);
    std::vector<FastllmBf16ExpertPointers> downTables(gateTables.size());
    for (int e = 0; e < experts; ++e) {
        const Data *g = weights[2 * (e + 1)], *d = weights[2 * (e + 1) + 1];
        if (g == nullptr || d == nullptr || g->dataType != BFLOAT16 ||
            d->dataType != BFLOAT16 || g->dims.size() != 2 || d->dims.size() != 2 ||
            g->dims[0] != 2 * inter || g->dims[1] != hidden ||
            d->dims[0] != hidden || d->dims[1] != inter || !resident(*g) || !resident(*d)) return false;
        gateTables[e / 256].weights[e % 256] = (const __nv_bfloat16 *)g->cudaData;
        downTables[e / 256].weights[e % 256] = (const __nv_bfloat16 *)d->cudaData;
    }
    auto prepare = [&](Data &data, int rows, int cols) {
        data.dataType = FLOAT32;
        data.UpdateUnitSize();
        data.Resize({rows, cols});
        data.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
        data.Allocate(false);
    };
    prepare(gate, batch * topk, 2 * inter);
    prepare(middle, batch * topk, inter);
    prepare(parts, batch * topk, hidden);
    prepare(output, batch, hidden);
    for (int first = 0; first < experts; first += 256) {
        FastllmMoeFp32Bf16IndexedKernel<false><<<dim3(2 * inter, batch * topk), 256>>>(
            (float *)input.cudaData, (float *)gate.cudaData, (const int32_t *)index.cudaData,
            gateTables[first / 256], first, std::min(256, experts - first), topk, hidden, 2 * inter);
    }
    FastllmCudaSwiglu(gate, middle);
    for (int first = 0; first < experts; first += 256) {
        if (inter == 128) {
            FastllmMoeFp32Bf16DownWarpKernel<128><<<dim3((hidden + 7) / 8, batch * topk), 256>>>(
                (const float *)middle.cudaData, (float *)parts.cudaData, (const int32_t *)index.cudaData,
                downTables[first / 256], first, std::min(256, experts - first), hidden);
        } else if (inter == 256) {
            FastllmMoeFp32Bf16DownWarpKernel<256><<<dim3((hidden + 7) / 8, batch * topk), 256>>>(
                (const float *)middle.cudaData, (float *)parts.cudaData, (const int32_t *)index.cudaData,
                downTables[first / 256], first, std::min(256, experts - first), hidden);
        } else {
            FastllmMoeFp32Bf16IndexedKernel<true><<<dim3(hidden, batch * topk), 256>>>(
                (float *)middle.cudaData, (float *)parts.cudaData, (const int32_t *)index.cudaData,
                downTables[first / 256], first, std::min(256, experts - first), topk, inter, hidden);
        }
    }
    FastllmMoeFp32Bf16ReduceKernel<<<dim3((hidden + 255) / 256, batch), 256>>>(
        (const float *)parts.cudaData, (const int32_t *)index.cudaData,
        (const float *)score.cudaData, (float *)output.cudaData,
        hidden, topk, experts, batch == 1);
    return true;
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Fp32KernelForBf16(float *A, float *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * B[p * m + i];
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] + bias[p];
        }
        __syncthreads();
    }
}

static void FastllmCudaBF16EnsureBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        cudaError_t state = cudaSuccess;
        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void *)cudaBiasData);
    }
}

static void FastllmCudaBF16EnsureBiasBf16OnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() < 2) {
        __nv_bfloat16 *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(__nv_bfloat16));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2Bf16Kernel <<<(k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(__nv_bfloat16));
        }
        checkCudaErrors("Error: CUDA error when moving bias (bf16) to device!", state);
        if (weight.extraCudaData.size() < 2)
            weight.extraCudaData.push_back((void *)cudaBiasData);
        else
            weight.extraCudaData[1] = (void *)cudaBiasData;
    }
}

// Half (FP16) bias for FP16×BF16 matmul output
static void FastllmCudaBF16EnsureBiasHalfOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || (bias.dims.size() > 0 && weight.extraCudaHalfData.size() == 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<<(k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias (half for FP16×BF16) to device!", state);
        weight.extraCudaHalfData.push_back((void *)cudaBiasData);
    }
}

void LaunchFastllmGemmFp32Bf16(float *input, __nv_bfloat16 *weight, float *output, float *bias, int n, int m, int k) {
    if (n > 1 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold()) {
        if (n <= 65535) {
            FastllmGemvFp32Bf16Kernel2MultiRow<256, 1>
                <<<dim3(k, n), 256>>>(
                    input, weight, output, bias, m, k);
        } else {
            for (int i = 0; i < n; ++i) {
                FastllmGemvFp32Bf16Kernel2MultiRow<256, 1>
                    <<<k, 256>>>(input + (size_t)i * m, weight,
                                 output + (size_t)i * k, bias, m, k);
            }
        }
    } else if (n == 1) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 2> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 3> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 4> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 5> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 6> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp32Bf16Kernel2MultiRow<256, 7> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp32Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input + i * m, weight, output + i * k, bias, m, k);
        }
    }
}

// BF16 -> FP16 逐元素转换，供 cublas FP16 gemm 使用
__global__ void FastllmCudaBF162HalfKernel(const __nv_bfloat16 *src, half *dst, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        dst[idx] = __float2half_rn(__bfloat162float(src[idx]));
}

// FP16 -> BF16 逐元素转换，供 FP16 input 转 BF16 后走 cublas BF16 gemm
__global__ void FastllmCudaHalf2Bf16Kernel(const half *src, __nv_bfloat16 *dst, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        dst[idx] = __float2bfloat16_rn(__half2float(src[idx]));
}

// cublas 输出 C 为列主序 (k×n)：C[i,j] 在 src[i+j*k]，且 C[i,j]=output[j,i]；写入行主序 dst：dst[row*k+col]=output[row,col]=src[col+row*k]
__global__ void Bf16ToHalfTransposeKernel(const __nv_bfloat16 *src, half *dst, int n, int k) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n * k) return;
    int row = idx / k;
    int col = idx % k;
    dst[idx] = __float2half_rn(__bfloat162float(src[col + row * k]));
}

void LaunchFastllmGemmFp16Bf16(half *input, __nv_bfloat16 *weight, half *output, half *bias, int n, int m, int k) {
    if (n > 1 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold()) {
        for (int i = 0; i < n; ++i) {
            FastllmGemvFp16Bf16Kernel2MultiRow<256, 1>
                <<<k, 256>>>(input + (size_t)i * m, weight,
                             output + (size_t)i * k, bias, m, k);
        }
    } else if (n == 1) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 2> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 3> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 4> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 5> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 6> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp16Bf16Kernel2MultiRow<256, 7> <<<k, 256>>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp16Bf16Kernel2MultiRow<256, 1> <<<k, 256>>>(input + i * m, weight, output + i * k, bias, m, k);
        }
    }
}

template <int THREADS>
static void LaunchFastllmGemmBf16Bf16SmallBatch(__nv_bfloat16 *input, __nv_bfloat16 *weight, __nv_bfloat16 *output, __nv_bfloat16 *bias, int n, int m, int k) {
    // PART=2..8 reuses the weights without changing the per-row reduction.
    if (n == 1 && m == 256) {
        if (k >= 16384 && FastllmCudaRuntimeArch() == 90)
            FastllmGemvBf16SmallK<4><<<(k + 15) / 16, 128>>>(input, weight, output, bias, k);
        else
            FastllmGemvBf16SmallK<><<<(k + 7) / 8, 256>>>(input, weight, output, bias, k);
    } else if (n == 1) {
        FastllmGemvBf16Bf16Kernel2MultiRow<THREADS, 1> <<<k, THREADS>>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvBf16Bf16Kernel2MultiRow<THREADS, 2> <<<k, THREADS>>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvBf16Bf16Kernel2MultiRow<THREADS, 3> <<<k, THREADS>>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvBf16Bf16Kernel2MultiRow<THREADS, 4> <<<k, THREADS>>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvBf16Bf16Kernel2MultiRow<THREADS, 5> <<<k, THREADS>>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvBf16Bf16Kernel2MultiRow<THREADS, 6> <<<k, THREADS>>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        if (THREADS == 256 && fastllm::cuda_pdl::Prepare(fastllm::cuda_pdl::Producer::DraftNorm))
            FastllmGemvBf16Bf16Kernel2MultiRow<THREADS, 7, false, true><<<k, THREADS>>>(input, weight, output, bias, m, k);
        else
            FastllmGemvBf16Bf16Kernel2MultiRow<THREADS, 7><<<k, THREADS>>>(input, weight, output, bias, m, k);
    } else if (n == 8) {
        FastllmGemvBf16Bf16Kernel2MultiRow<THREADS, 8> <<<k, THREADS>>>(input, weight, output, bias, m, k);
    }
}

namespace {
// Naive Indexer layout: 16 query heads, one key head, and one score per head.
constexpr int kIndexerInput = 4096, kIndexerHeads = 16, kIndexerDim = 128;
constexpr int kIndexerQ = kIndexerHeads * kIndexerDim;
constexpr int kIndexerQK = kIndexerQ + kIndexerDim;
constexpr int kIndexerQKW = kIndexerQK + kIndexerHeads;

__device__ __forceinline__ void StoreIndexerProjection(__nv_bfloat16 *query,
        __nv_bfloat16 *key, __nv_bfloat16 *scores, int row, int column, __nv_bfloat16 value) {
    if (column < kIndexerQ) query[row * kIndexerQ + column] = value;
    else if (column < kIndexerQK) key[row * kIndexerDim + column - kIndexerQ] = value;
    // Preserve the BF16 projection round before multiplying by 1/sqrt(16).
    else scores[row * kIndexerHeads + column - kIndexerQK] =
        __float2bfloat16_rn(__bfloat162float(value) * 0.25f);
}
}

// Map the original 256 logical lanes onto one warp per output. Each lane
// keeps eight independent partials and reproduces the original 128/64/32
// compensated tree in registers before the warp tree. This packs four
// outputs per CTA without changing products, sum order or weight storage.
// Two input rows share weight loads and keep independent reduction state.
template <int K, int ROWS = 1, bool INDEXER = false>
__global__ void FastllmGemvBf16Bf16WarpRows(const __nv_bfloat16 *input,
        const __nv_bfloat16 *weight, __nv_bfloat16 *output,
        const __nv_bfloat16 *bias, int outputs,
        __nv_bfloat16 *indexKey = nullptr, __nv_bfloat16 *indexScores = nullptr) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= outputs) return;
    float values[ROWS][8] = {}, correction[ROWS][8] = {};
    #pragma unroll
    for (int part = 0; part < 8; ++part) {
        #pragma unroll
        for (int column = (lane + part * 32) * 8; column < K; column += 256 * 8) {
            union_bf16_8 b;
            b.in = *reinterpret_cast<const uint4 *>(weight + (size_t)row * K + column);
            #pragma unroll
            for (int batch = 0; batch < ROWS; ++batch) {
                union_bf16_8 a;
                a.in = *reinterpret_cast<const uint4 *>(input + batch * K + column);
                float sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    sum += __bfloat162float(a.out[i]) * __bfloat162float(b.out[i]);
                values[batch][part] += sum;
            }
        }
    }
    #pragma unroll
    for (int batch = 0; batch < ROWS; ++batch) {
        #pragma unroll
        for (int stride = 4; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int part = 0; part < stride; ++part) {
                float other = values[batch][part + stride] - correction[batch][part];
                float sum = values[batch][part] + other;
                correction[batch][part] = (sum - values[batch][part]) - other;
                values[batch][part] = sum;
            }
        }
        float value = values[batch][0], diff = correction[batch][0];
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            float peer = __shfl_down_sync(0xffffffffu, value, stride);
            if (lane < stride) {
                float other = peer - diff;
                float sum = value + other;
                diff = (sum - value) - other;
                value = sum;
            }
        }
        if (lane == 0) {
            const auto rounded = __float2bfloat16_rn(bias ? value + __bfloat162float(bias[row]) : value);
            if constexpr (INDEXER)
                StoreIndexerProjection(output, indexKey, indexScores, batch, row, rounded);
            else output[batch * outputs + row] = rounded;
        }
    }
}

namespace {
bool IndexerWeightLayout(const fastllm::Data &wq, const fastllm::Data &wk,
        const fastllm::Data &weights) {
    const fastllm::Data *parts[] = {&wq, &wk, &weights};
    const int widths[] = {kIndexerQ, kIndexerDim, kIndexerHeads};
    for (int i = 0; i < 3; ++i) {
        const auto &part = *parts[i];
        if (part.dataType != fastllm::BFLOAT16 || part.dataDevice != fastllm::DataDevice::CUDA ||
            !part.cudaData || part.dims.size() != 2 || part.dims[0] != widths[i] || part.dims[1] != kIndexerInput ||
            part.strides.size() != 2 || part.strides[0] != kIndexerInput || part.strides[1] != 1 ||
            part.dataDeviceIds != wq.dataDeviceIds) return false;
    }
    return FastllmCudaRuntimeArch() == 90;
}

bool IndexerWeightsPacked(const fastllm::Data &wq, const fastllm::Data &wk,
        const fastllm::Data &weights) {
    const auto *base = (const uint8_t *)wq.cudaData;
    return wq.expansionBytes >= (size_t)kIndexerQKW * kIndexerInput * 2 &&
        wk.cudaData == base + (size_t)kIndexerQ * kIndexerInput * 2 &&
        weights.cudaData == base + (size_t)kIndexerQK * kIndexerInput * 2;
}

__global__ void IndexerProjectionUnpack(const __nv_bfloat16 *packed,
        __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *scores, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * kIndexerQKW) return;
    const int row = i / kIndexerQKW, column = i % kIndexerQKW;
    StoreIndexerProjection(query, key, scores, row, column, packed[i]);
}
}

bool FastllmCudaPackIndexerWeights(fastllm::Data &wq, fastllm::Data &wk, fastllm::Data &weights) {
    if (!IndexerWeightLayout(wq, wk, weights)) return false;
    if (IndexerWeightsPacked(wq, wk, weights)) return true;
    if (FastllmCudaGraphIsCapturingFast()) return false;
    for (auto *part : {&wq, &wk, &weights})
        if (part->isFake || part->cudaDataBorrowed) return false;
    fastllm::Data packed(fastllm::BFLOAT16, {kIndexerQKW, kIndexerInput});
    packed.ToDevice(fastllm::DataDevice::CUDA, wq.dataDeviceIds, false);
    packed.Allocate();
    size_t offset = 0;
    for (auto *part : {&wq, &wk, &weights}) {
        FastllmCudaCopyFromDeviceToDevice((uint8_t *)packed.cudaData + offset, part->cudaData, part->GetBytes());
        offset += part->GetBytes();
    }
    // Replace the three allocations, retaining original shapes for K-only
    // short contexts and large-batch/unsupported projection fallbacks.
    offset = 0;
    for (auto *part : {&wq, &wk, &weights}) {
        part->FreeSpace();
        part->cudaData = (uint8_t *)packed.cudaData + offset;
        part->cudaDataBorrowed = part != &wq;
        part->directMemory = false;
        part->expansionSize = part == &wq ? packed.expansionSize : part->Count(0);
        part->expansionBytes = part == &wq ? packed.expansionBytes : part->GetBytes();
        offset += part->GetBytes();
    }
    packed.cudaData = nullptr;
    return true;
}

bool FastllmCudaIndexerProjection(const fastllm::Data &input,
        const fastllm::Data &wq, const fastllm::Data &wk, const fastllm::Data &weights,
        fastllm::Data &query, fastllm::Data &key, fastllm::Data &scores, fastllm::Data &scratch) {
    if (input.dataType != fastllm::BFLOAT16 || input.dataDevice != fastllm::DataDevice::CUDA ||
        !input.cudaData || input.dims.empty() || input.dims.back() != kIndexerInput ||
        input.strides.size() != input.dims.size() ||
        (reinterpret_cast<uintptr_t>(input.cudaData) & 15) ||
        (reinterpret_cast<uintptr_t>(wq.cudaData) & 15) ||
        input.dataDeviceIds != wq.dataDeviceIds ||
        !IndexerWeightLayout(wq, wk, weights) || !IndexerWeightsPacked(wq, wk, weights)) return false;
    size_t elements = 1;
    for (int i = (int)input.dims.size() - 1; i >= 0; --i) {
        if (input.strides[i] != elements || input.dims[i] <= 0) return false;
        elements *= input.dims[i];
    }
    const size_t rows = elements / kIndexerInput;
    // Limit fusion to measured decode batches; larger GEMMs keep the fallback.
    if (rows == 0 || rows > 256) return false;
    auto prepare = [&](fastllm::Data &output, int width) {
        auto dims = input.dims;
        dims.back() = width;
        output.dataType = fastllm::BFLOAT16;
        output.Resize(dims);
        output.ToDevice(input.dataDevice, input.dataDeviceIds, false);
        output.Allocate();
    };
    prepare(query, kIndexerQ); prepare(key, kIndexerDim); prepare(scores, kIndexerHeads);
    const auto *x = (const __nv_bfloat16 *)input.cudaData;
    const auto *w = (const __nv_bfloat16 *)wq.cudaData;
    auto *q = (__nv_bfloat16 *)query.cudaData;
    auto *k = (__nv_bfloat16 *)key.cudaData;
    auto *s = (__nv_bfloat16 *)scores.cudaData;
    if (rows == 1) {
        FastllmGemvBf16Bf16WarpRows<kIndexerInput, 1, true><<<(kIndexerQKW + 3) / 4, 128>>>(x, w, q, nullptr, kIndexerQKW, k, s);
    } else if (rows == 2) {
        FastllmGemvBf16Bf16WarpRows<kIndexerInput, 2, true><<<(kIndexerQKW + 3) / 4, 128>>>(x, w, q, nullptr, kIndexerQKW, k, s);
    } else {
        prepare(scratch, kIndexerQKW);
        auto handle = getFastllmCublasHandle();
        if (!fastllm_bf16_lt::Matmul(input.cudaData, wq.cudaData, scratch.cudaData, rows, kIndexerInput, kIndexerQKW)) {
            const float alpha = 1.0f, beta = 0.0f;
            const auto status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kIndexerQKW, rows, kIndexerInput,
                &alpha, w, CUDA_R_16BF, kIndexerInput, x, CUDA_R_16BF, kIndexerInput, &beta,
                scratch.cudaData, CUDA_R_16BF, kIndexerQKW, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Indexer BF16 projection failed.");
        }
        IndexerProjectionUnpack<<<(rows * kIndexerQKW + 255) / 256, 256>>>(
            (const __nv_bfloat16 *)scratch.cudaData, q, k, s, rows);
    }
    return true;
}

#include "fastllm-bf16-narrow.cuh"

// The narrow GDN gates are sensitive to a single BF16 rounding boundary.
// Use the single-row decode reduction for every prompt row in one launch.
void FastllmCudaBf16ExactRows(const void *input, const void *weight, void *output,
                              int rows, int columns, int outputs) {
    // In the narrow GDN prefill projection one CTA can share each weight
    // vector across four token rows with the same compensated reduction.
    auto *a = (__nv_bfloat16 *)input;
    auto *b = (__nv_bfloat16 *)weight;
    auto *c = (__nv_bfloat16 *)output;
    if (columns == 4608 && outputs % 8 == 0 && rows <= 4 * 65535 &&
        size_t(rows / 4) * (outputs / 8) >= 256 && FastllmCudaRuntimeArch() == 120) {
        FastllmBf16Narrow<4, 8, 4608><<<dim3(outputs / 8, (rows + 3) / 4), 256>>>(a, b, c, rows, outputs);
        return;
    }
    int begin = 0;
    if (columns == 4096 && outputs == 64) {
        while (rows - begin >= 4) {
            const int groups = std::min(65535, (rows - begin) / 4);
            FastllmGemvBf16Bf16Kernel2MultiRow<256, 4, true><<<dim3(outputs, groups), 256>>>(
                a + size_t(begin) * columns, b, c + size_t(begin) * outputs, nullptr, columns, outputs);
            begin += 4 * groups;
        }
    }
    if (columns == 4608 && outputs == 64 && FastllmCudaRuntimeArch() == 120) {
        // With only 64 output channels, pairs of queries share weights while
        // leaving enough independent blocks to fill the device.
        while (rows - begin >= 2) {
            const int groups = std::min(65535, (rows - begin) / 2);
            FastllmGemvBf16Bf16Kernel2MultiRow<256, 2, true><<<dim3(outputs, groups), 256>>>(
                a + size_t(begin) * columns, b, c + size_t(begin) * outputs, nullptr, columns, outputs);
            begin += 2 * groups;
        }
    }
    while (begin < rows) {
        const int count = std::min(65535, rows - begin);
        FastllmGemvBf16Bf16Kernel2MultiRow<256, 1><<<dim3(outputs, count), 256>>>(
            a + size_t(begin) * columns, b, c + size_t(begin) * outputs, nullptr, columns, outputs);
        begin += count;
    }
}

// Each warp owns four tokens and four output channels. Neighboring dots share
// both operand loads. A lane holds the eight leaves spaced 32 apart in the
// original 256-leaf GEMV tree, so its BF16 result is independent of the batch.
struct FastllmBf16WarpDots {
    float values[4][4];

    __device__ __forceinline__ void merge(FastllmBf16WarpDots &error, const FastllmBf16WarpDots &peer) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                FastllmBf16HeadAdd(values[r][c], error.values[r][c], peer.values[r][c]);
        }
    }
};

template <int FixedWidth>
__device__ __forceinline__ FastllmBf16WarpDots FastllmBf16WarpLeaf(
        const __nv_bfloat16 *input, const __nv_bfloat16 *weight, int width, int offset, int live) {
    if constexpr (FixedWidth) width = FixedWidth;
    FastllmBf16WarpDots dots{};
    #pragma unroll 2
    for (int half = 0; half < width; half += 2048) {
        if (half + offset >= width) break;
        union_bf16_8 x[4], w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            x[j].in = *reinterpret_cast<const uint4 *>(input + min(j, live - 1) * width + half + offset);
            w[j].in = *reinterpret_cast<const uint4 *>(weight + j * width + half + offset);
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                float sum = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    sum = fmaf(__bfloat162float(x[r].out[i]), __bfloat162float(w[c].out[i]), sum);
                dots.values[r][c] += sum;
            }
        }
    }
    return dots;
}

template <int FixedWidth, bool Grouped>
__global__ void FastllmBf16WarpProjection(FastllmBf16ProjectionInputs inputs, const __nv_bfloat16 *weight,
        __nv_bfloat16 *output, int rows, int width) {
    if constexpr (FixedWidth) width = FixedWidth;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.y * 8 + (warp / 4) * 4;
    const int column = blockIdx.x * 16 + (warp % 4) * 4;
    if (row >= rows || column >= inputs.ends[2]) return;
    const int group = Grouped ? (column >= inputs.ends[0]) + (column >= inputs.ends[1]) : 0;
    const int begin = group ? inputs.ends[group - 1] : 0;
    const int pitch = inputs.ends[group] - begin;
    const auto *x = inputs.x[group] + size_t(row) * width;
    const auto *w = weight + size_t(column) * width;
    const int live = rows - row;

    FastllmBf16WarpDots result{}, correction{};
    #pragma unroll
    for (int side = 0; side < 2; ++side) {
        FastllmBf16WarpDots subtree{}, carry{};
        #pragma unroll
        for (int child = 0; child < 2; ++child) {
            const int offset = (lane + 32 * (side + 2 * child)) * 8;
            auto left = FastllmBf16WarpLeaf<FixedWidth>(x, w, width, offset, live);
            auto right = FastllmBf16WarpLeaf<FixedWidth>(x, w, width, offset + 1024, live);
            FastllmBf16WarpDots error{};
            left.merge(error, right);
            if (child == 0) { subtree = left; carry = error; }
            else subtree.merge(carry, left);
        }
        if (side == 0) { result = subtree; correction = carry; }
        else result.merge(correction, subtree);
    }
    #pragma unroll
    for (int step = 16; step; step >>= 1) {
        FastllmBf16WarpDots peer;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                peer.values[r][c] = __shfl_down_sync(0xffffffff, result.values[r][c], step);
        }
        result.merge(correction, peer);
    }
    if (lane == 0) {
        auto *out = output + size_t(begin) * rows + size_t(row) * pitch + column - begin;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r >= live) break;
            #pragma unroll
            for (int c = 0; c < 4; ++c) out[r * pitch + c] = __float2bfloat16_rn(result.values[r][c]);
        }
    }
}

template <bool Grouped>
static bool LaunchBf16WarpProjection(const FastllmBf16ProjectionInputs &inputs, const __nv_bfloat16 *weight,
                                     __nv_bfloat16 *out, int rows, int width) {
    if (rows < 8 || rows > 8 * 65535 || width < 2048 || width > 4096 || width % 8 ||
        (reinterpret_cast<uintptr_t>(weight) & 15)) return false;
    for (int g = 0; g < (Grouped ? 3 : 1); ++g)
        if (inputs.ends[g] % 4 || (reinterpret_cast<uintptr_t>(inputs.x[g]) & 15)) return false;
    dim3 grid((inputs.ends[2] + 15) / 16, (rows + 7) / 8);
    if (width == 2304)
        FastllmBf16WarpProjection<2304, Grouped><<<grid, 256>>>(inputs, weight, out, rows, width);
    else if (width == 3072)
        FastllmBf16WarpProjection<3072, Grouped><<<grid, 256>>>(inputs, weight, out, rows, width);
    else
        FastllmBf16WarpProjection<0, Grouped><<<grid, 256>>>(inputs, weight, out, rows, width);
    return true;
}

bool FastllmCudaBf16ExactBatchProjection(const void *input, const void *weight, void *output,
                                        int rows, int width, int columns) {
    FastllmBf16ProjectionInputs inputs{{static_cast<const __nv_bfloat16 *>(input)}, {columns, columns, columns}};
    return LaunchBf16WarpProjection<false>(inputs, static_cast<const __nv_bfloat16 *>(weight),
                                           static_cast<__nv_bfloat16 *>(output), rows, width);
}

bool FastllmCudaBf16DecodeProjection(const void *input, const void *weight, void *output,
                                    int rows, int width, int columns) {
    return fastllm_bf16_lt::Matmul(input, weight, output, rows, width, columns,
                                   CUDA_R_16BF, fastllm_bf16_lt::prefillReferenceRows);
}

bool LaunchFastllmGroupedBf16Projection(const __nv_bfloat16 *const *x, const __nv_bfloat16 *weight,
        __nv_bfloat16 *out, int rows, int width, const int *columns) {
    if (width < 2048 || width > 4096 || width % 8 || (reinterpret_cast<uintptr_t>(weight) & 15)) return false;
    if (rows != 1 && (rows < 4 || rows % 4 || FastllmCudaRuntimeArch() != 120)) return false;
    FastllmBf16ProjectionInputs inputs{};
    int total = 0;
    for (int g = 0; g < 3; ++g) {
        if (columns[g] < 1024 || columns[g] % 2 || (reinterpret_cast<uintptr_t>(x[g]) & 15)) return false;
        inputs.x[g] = x[g];
        inputs.ends[g] = total += columns[g];
    }
    if (rows == 1) {
        // The same compensated dot product as three separate GEMVs. Each CTA
        // selects its input from its output channels, with no activation copy.
        if (width == 2304)
            FastllmBf16GroupedProjection<1, 2304><<<total / 2, 128>>>(inputs, weight, out, rows, width, 0);
        else
            FastllmBf16GroupedProjection<1><<<total / 2, 128>>>(inputs, weight, out, rows, width, 0);
        return true;
    }
    if (LaunchBf16WarpProjection<true>(inputs, weight, out, rows, width)) return true;
    int first = 0;
    while (rows - first >= 8) {
        const int tiles = std::min((rows - first) / 8, 65535);
        if (width == 2304)
            FastllmBf16GroupedProjection<8, 2304><<<dim3(total / 2, tiles), 128>>>(inputs, weight, out, rows, width, first);
        else
            FastllmBf16GroupedProjection<8><<<dim3(total / 2, tiles), 128>>>(inputs, weight, out, rows, width, first);
        first += tiles * 8;
    }
    if (first < rows)
        FastllmBf16GroupedProjection<4><<<total / 2, 128>>>(inputs, weight, out, rows, width, first);
    return true;
}

void LaunchFastllmGemmBf16Bf16(__nv_bfloat16 *input, __nv_bfloat16 *weight, __nv_bfloat16 *output, __nv_bfloat16 *bias, int n, int m, int k) {
    // Two banks cover 256 eight-value leaves. Keep the singleton FMA and compensated
    // reduction order while reusing each weight across a tile of independent rows.
    if (!bias && n >= 4 && m >= 2048 && m <= 4096 && m % 8 == 0 && k >= 1024 && k % 2 == 0 &&
        !(reinterpret_cast<uintptr_t>(input) & 15) && !(reinterpret_cast<uintptr_t>(weight) & 15) &&
        FastllmCudaRuntimeArch() == 120) {
        while (n >= 8) {
            int tiles = std::min(n / 8, 65535);
            if (m == 2304)
                FastllmBf16ExactProjection<2304, 8><<<dim3(k / 2, tiles), 128>>>(input, weight, output, k);
            else if (m == 3072)
                FastllmBf16ExactProjection<3072, 8><<<dim3(k / 2, tiles), 128>>>(input, weight, output, k);
            else
                FastllmBf16ExactProjection<0, 8><<<dim3(k / 2, tiles), 128>>>(input, weight, output, k, m);
            n -= tiles * 8;
            input += size_t(tiles) * 8 * m;
            output += size_t(tiles) * 8 * k;
        }
        if (n > 4) {
            // A partial tile shares each vocabulary weight across the live rows.
            if (n == 7 && fastllm::cuda_pdl::Prepare(fastllm::cuda_pdl::Producer::DraftNorm))
                FastllmBf16ExactProjection<0, 8, true, true><<<k / 2, 128>>>(input, weight, output, k, m, n);
            else
                FastllmBf16ExactProjection<0, 8, true><<<k / 2, 128>>>(input, weight, output, k, m, n);
            return;
        }
        if (n >= 4) {
            FastllmBf16ExactProjection<0, 4><<<k / 2, 128>>>(input, weight, output, k, m);
            n -= 4;
            input += size_t(4) * m;
            output += size_t(4) * k;
        }
        if (!n) return;
    }
    if ((n == 1 || n == 2) && k >= 256 && (m == 1024 || m == 2048 || m == 4096) &&
        (reinterpret_cast<uintptr_t>(input) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(weight) & 15) == 0 && FastllmCudaRuntimeArch() == 90) {
        if (n == 1) {
            if (m == 1024) FastllmGemvBf16Bf16WarpRows<1024><<<(k + 3) / 4, 128>>>(input, weight, output, bias, k);
            else if (m == 2048) FastllmGemvBf16Bf16WarpRows<2048><<<(k + 3) / 4, 128>>>(input, weight, output, bias, k);
            else FastllmGemvBf16Bf16WarpRows<4096><<<(k + 3) / 4, 128>>>(input, weight, output, bias, k);
        } else {
            if (m == 1024) FastllmGemvBf16Bf16WarpRows<1024, 2><<<(k + 3) / 4, 128>>>(input, weight, output, bias, k);
            else if (m == 2048) FastllmGemvBf16Bf16WarpRows<2048, 2><<<(k + 3) / 4, 128>>>(input, weight, output, bias, k);
            else FastllmGemvBf16Bf16WarpRows<4096, 2><<<(k + 3) / 4, 128>>>(input, weight, output, bias, k);
        }
        return;
    }
    if (n >= 1 && n <= 8) {
        if (m > 0 && m <= 256 && m % 8 == 0) {
            // Only one warp loads data in these short, aligned projections.
            LaunchFastllmGemmBf16Bf16SmallBatch<32>(input, weight, output, bias, n, m, k);
        } else {
            LaunchFastllmGemmBf16Bf16SmallBatch<256>(input, weight, output, bias, n, m, k);
        }
    } else if (n > 8 && n <= 65535 &&
               n < fastllm::FastllmCudaGetLinearExactBatchThreshold()) {
        // Larger exact batches use independent rows, within grid.y's limit.
        FastllmGemvBf16Bf16Kernel2MultiRow<256, 1>
            <<<dim3(k, n), 256>>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvBf16Bf16Kernel2MultiRow<256, 1>
                <<<k, 256>>>(input + (size_t)i * m, weight,
                             output + (size_t)i * k, bias, m, k);
        }
    }
}

bool FastllmCudaMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    // CPU MoE experts can be uploaded to a different GPU on the next request.
    // A cached zero bias belongs to the previous GPU; the bias-free kernel
    // needs no allocation and must not dereference that device's pointer.
    float *cudaBiasData = nullptr;
    if (!bias.dims.empty()) {
        FastllmCudaBF16EnsureBiasOnDevice(weight, bias, k);
        cudaBiasData = (float *)weight.extraCudaData[0];
    }
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    __nv_bfloat16 *weightPtr = (__nv_bfloat16 *)weight.cudaData;

    if (n < 8) {
        LaunchFastllmGemmFp32Bf16(cudaInput, weightPtr, cudaOutput, cudaBiasData, n, m, k);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        __nv_bfloat16 *cudaBf16Input = (__nv_bfloat16 *)FastllmCudaMalloc(n * m * sizeof(__nv_bfloat16));

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2Bf16Kernel <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaBf16Input, len);

        cublasStatus_t status;
        float h_alpha = 1.0f, h_beta = 0.0f;
        cudaDataType_t AType = CUDA_R_16BF, BType = CUDA_R_16BF, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weightPtr, AType,
                              m, cudaBf16Input, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error (MatMulBFloat16).\n");
            throw("cublas error");
            exit(0);
        }

        FastllmCudaFree(cudaBf16Input);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<<n, 256>>>(cudaOutput, (float *)weight.extraCudaData[0], k);
        }
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// FP16 input × BF16 weight -> FP16 output
bool FastllmCudaHalfMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    // Temporary weight views have no persistent bias cache. Bias-free
    // projections must not allocate zero buffers, especially during capture.
    if (!bias.dims.empty()) {
        FastllmCudaBF16EnsureBiasOnDevice(weight, bias, k);
        FastllmCudaBF16EnsureBiasHalfOnDevice(weight, bias, k);
    }

    half *cudaInput = (half *)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *)FastllmCudaPrepareOutput(output);
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *)weight.extraCudaHalfData[0];
    __nv_bfloat16 *weightPtr = (__nv_bfloat16 *)weight.cudaData;

    bool exactRows = n > 1 &&
        n < fastllm::FastllmCudaGetLinearExactBatchThreshold();
    if (n < 8 || exactRows) {
        LaunchFastllmGemmFp16Bf16(cudaInput, weightPtr, cudaOutput, cudaBiasData, n, m, k);
    } else if (false) {
        auto fastllmCublasHandle = getFastllmCublasHandle();

        size_t wsBytes = 0;
        bool ownScratch = false;
        half *cudaFp16Weight = (half *) FastllmBorrowDequantScratch((size_t)k * m * sizeof(half), &wsBytes, &ownScratch);
        size_t bytesPerRow = (size_t)m * sizeof(half);
        int maxRowsPerChunk = (int)std::min<size_t>((size_t)k, std::max<size_t>(1, wsBytes / bytesPerRow));

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *) FastllmCudaMalloc((size_t)n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
        int threadPerBlock = std::min(256, m);

        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            int chunkLen = kc * m;
            FastllmCudaBF162HalfKernel <<<(chunkLen - 1) / threadPerBlock + 1, threadPerBlock>>>(
                (__nv_bfloat16 *)weight.cudaData + (size_t)kOff * m,
                cudaFp16Weight, chunkLen);
#ifdef CUDA_NO_TENSOR_CORE
            status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    kc, n, m,
                                    &h_alpha, cudaFp16Weight, AType,
                                    m, cudaInput, BType,
                                    m, &h_beta,
                                    cudaFp32Output + kOff, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
            status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    kc, n, m,
                                    &h_alpha, cudaFp16Weight, AType,
                                    m, cudaInput, BType,
                                    m, &h_beta,
                                    cudaOutput + kOff, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif

            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error.\n");
                throw("cublas error");
                exit(0);
            }
        }

#ifdef CUDA_NO_TENSOR_CORE
        int len = n * k;
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmReleaseDequantScratch(cudaFp16Weight, ownScratch);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        __nv_bfloat16 *cudaBF16Input;
        __nv_bfloat16 *cudaBF16Output;
        cudaBF16Input = (__nv_bfloat16 *) FastllmCudaMalloc(n * m * sizeof(__nv_bfloat16));
        cudaBF16Output = (__nv_bfloat16 *) FastllmCudaMalloc(n * k * sizeof(__nv_bfloat16));
        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaHalf2Bf16Kernel <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaBF16Input, len);

        float h_alpha = 1.0f, h_beta = 0.0f;
        cudaDataType_t AType = CUDA_R_16BF, BType = CUDA_R_16BF, CType = CUDA_R_16BF, ComputeType = CUDA_R_32F;
        cublasStatus_t status;

        // Tiny output projections otherwise select a severely underfilled
        // BF16 GEMM at eight verification rows. Retain the GEMM path's BF16
        // input/output rounding and post-conversion bias semantics.
        if (n == 8 && k <= 1024) {
            LaunchFastllmGemmBf16Bf16(cudaBF16Input, weightPtr, cudaBF16Output,
                                     nullptr, n, m, k);
            status = CUBLAS_STATUS_SUCCESS;
        } else {
            status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    k, n, m,
                                    &h_alpha, weightPtr, AType,
                                    m, cudaBF16Input, BType,
                                    m, &h_beta,
                                    cudaBF16Output, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        }
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        threadPerBlock = std::min(256, len);
        FastllmCudaBF162HalfKernel <<<(len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaBF16Output, cudaOutput, len);

        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaBF16Input);
        FastllmCudaFree(cudaBF16Output);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// BF16 input × BF16 weight -> BF16 output
bool FastllmCudaBFloat16PrefillLinear(const fastllm::Data &input,
        const fastllm::Data &weight, fastllm::Data &output) {
    using namespace fastllm;
    if (input.dataType != BFLOAT16 || weight.dataType != BFLOAT16 ||
        input.dataDevice != CUDA || weight.dataDevice != CUDA || !input.cudaData || !weight.cudaData ||
        input.dataDeviceIds != weight.dataDeviceIds || input.multiDeviceData || weight.multiDeviceData ||
        input.dims.size() != 3 || input.dims[0] != 1 || input.dims[1] <= 0 || weight.dims.size() != 2 ||
        input.strides.size() != 3 || weight.strides.size() != 2 || weight.dims[0] <= 0 || weight.dims[1] <= 0 ||
        input.dims.back() != weight.dims[1] || input.strides[1] != (uint64_t)weight.dims[1] ||
        input.strides[2] != 1 || weight.strides[0] != (uint64_t)weight.dims[1] || weight.strides[1] != 1 ||
        FastllmCudaRuntimeArch() != 120) return false;
    output.dataType = BFLOAT16;
    output.UpdateUnitSize();
    output.Resize({1, input.dims[1], weight.dims[0]});
    output.ToDevice(CUDA, input.dataDeviceIds, false);
    output.Allocate(false);
    return fastllm_bf16_lt::Matmul(input.cudaData, weight.cudaData, output.cudaData,
        input.dims[1], weight.dims[1], weight.dims[0], CUDA_R_16BF, fastllm_bf16_lt::prefillReferenceRows);
}

bool FastllmCudaBFloat16LinearFixedRows(const fastllm::Data &input,
        const fastllm::Data &weight, fastllm::Data &output) {
    auto dense = [](const fastllm::Data &x) {
        return x.dataType == fastllm::BFLOAT16 && x.dataDevice == fastllm::CUDA &&
            x.cudaData && x.dims.size() == 2 && x.strides.size() == 2 &&
            x.dims[0] > 0 && x.dims[1] > 0 && x.strides[1] == 1 &&
            x.strides[0] == (uint64_t)x.dims[1] && !x.multiDeviceData;
    };
    if (!dense(input) || !dense(weight) || !dense(output) ||
        input.dataDeviceIds != weight.dataDeviceIds || input.dataDeviceIds != output.dataDeviceIds ||
        input.dims[1] != weight.dims[1] || output.dims[0] != input.dims[0] ||
        output.dims[1] != weight.dims[0] || input.dims[0] <= 32 || input.dims[0] % 32 ||
        input.dims[1] < 1024 || weight.dims[0] < 1024 || input.dims[1] % 4 || weight.dims[0] % 4 ||
        output.cudaData == input.cudaData || output.cudaData == weight.cudaData ||
        FastllmCudaRuntimeArch() != 90) return false;
    const int rows = input.dims[0], in = input.dims[1], out = weight.dims[0];
    // The shared Lt workspace is bounded at 8 MiB. Larger inputs are covered
    // in 256-row groups, without retaining another per-layer allocation.
    for (int row = 0; row < rows; row += 256)
        if (!fastllm_bf16_lt::Matmul((const uint16_t *)input.cudaData + size_t(row) * in,
                weight.cudaData, (uint16_t *)output.cudaData + size_t(row) * out,
                std::min(256, rows - row), in, out, CUDA_R_16BF, 32)) return false;
    return true;
}

bool FastllmCudaBFloat16MatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    // Temporary weight views have no persistent bias cache. Bias-free
    // projections must not allocate zero buffers, especially during capture.
    if (!bias.dims.empty()) {
        FastllmCudaBF16EnsureBiasOnDevice(weight, bias, k);
        FastllmCudaBF16EnsureBiasBf16OnDevice(weight, bias, k);
    }

    __nv_bfloat16 *cudaInput = (__nv_bfloat16 *)FastllmCudaPrepareInput(input);
    __nv_bfloat16 *cudaOutput = (__nv_bfloat16 *)FastllmCudaPrepareOutput(output);
    __nv_bfloat16 *cudaBiasData = bias.dims.size() == 0 ? nullptr : (__nv_bfloat16 *)weight.extraCudaData[1];
    __nv_bfloat16 *weightPtr = (__nv_bfloat16 *)weight.cudaData;

    if (n >= 8 && m >= 1024 && FastllmCudaNvfp4Scope::IsBatched() &&
        FastllmCudaRuntimeArch() == 120) {
        for (int row = 0; row < n; row += 8) {
            const int rows = std::min(8, n - row);
            auto *x = cudaInput + size_t(row) * m;
            auto *y = cudaOutput + size_t(row) * k;
            if (rows == 8 && m == 4096 && k >= 65536 && k % 2 == 0 && !cudaBiasData) {
                FastllmBf16ExactProjection<4096, 8><<<k / 2, 128>>>(x, weightPtr, y, k);
            } else {
                LaunchFastllmGemmBf16Bf16SmallBatch<256>(x, weightPtr, y, cudaBiasData, rows, m, k);
            }
        }
    } else if (n < 8) {
        // Wide Hopper BF16 projections use FP32-accumulating Lt, including batched
        // decode/verify. The exact-row policy still applies to FP32 routers;
        // BF16 results may differ in rounding from the GEMV reduction tree.
        // Narrow projections are faster with GEMV. Preserve its fused bias
        // rounding and the unwarmed/unsupported fallbacks as well.
        // A single 4096-wide row with at most 2048 outputs is faster with the
        // native GEMV; Lt's split-K launch/reduction overhead dominates here.
        // Wide SM103 heads also benefit from Lt at three through seven rows.
        const int arch = FastllmCudaRuntimeArch();
        const bool useLt = n > 0 && m >= 1024 && k >= 1024 &&
            !(n == 1 && m == 4096 && k <= 2048) && bias.dims.empty() &&
            (arch == 90 || (arch == 103 && n >= 3 && m >= 4096 && k >= 16384));
        if (!useLt || !fastllm_bf16_lt::Matmul(cudaInput, weightPtr, cudaOutput, n, m, k)) {
            LaunchFastllmGemmBf16Bf16(cudaInput, weightPtr, cudaOutput, cudaBiasData, n, m, k);
        }
    } else if (n == 8 && m > 0 && m <= 256 && m % 8 == 0) {
        LaunchFastllmGemmBf16Bf16(cudaInput, weightPtr, cudaOutput, nullptr, n, m, k);
        // Match the GEMM branch: round the dot product before adding bias.
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }
    } else {
        // Warm the fallback handle before a later graph capture can miss the
        // Lt cache. Handle/workspace creation during capture is not safe.
        auto fastllmCublasHandle = getFastllmCublasHandle();
        // Small batches use a cached Lt algorithm; unsupported shapes retain
        // GemmEx. Both paths accumulate in FP32 and round to BF16 before bias.
        if (!FastllmCudaHybridBf16Gemm(cudaInput, weightPtr, cudaOutput,
                n, m, k, k, fastllmCublasHandle) &&
            !fastllm_bf16_lt::Matmul(cudaInput, weightPtr, cudaOutput, n, m, k)) {
            float alpha = 1.0f, beta = 0.0f;
            cublasStatus_t status = cublasGemmEx(fastllmCublasHandle,
                CUBLAS_OP_T, CUBLAS_OP_N, k, n, m,
                &alpha, weightPtr, CUDA_R_16BF, m,
                cudaInput, CUDA_R_16BF, m, &beta,
                cudaOutput, CUDA_R_16BF, k, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error (BFloat16MatMulBFloat16).\n");
                throw("cublas error");
            }
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<<n, 256>>>(cudaOutput, cudaBiasData, k);
        }
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// BF16 input × FP32 weight -> BF16 output
bool FastllmCudaBFloat16MatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaBF16EnsureBiasOnDevice(weight, bias, k);

    float *cudaBiasData = (float *)weight.extraCudaData[0];
    float *cudaInput = (float *)FastllmCudaMalloc(input.Count(0) * sizeof(float));
    float *cudaOutput = (float *)FastllmCudaMalloc(output.Count(0) * sizeof(float));
    int inputLen = input.Count(0);
    FastllmCudaBF162FloatKernel <<< (inputLen - 1) / 256 + 1, 256 >>>((uint16_t *)input.cudaData, cudaInput, inputLen);

    if (n > 1) {
        float h_alpha = 1.0f, h_beta = 0.0f;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cudaDataType_t AType = CUDA_R_32F, BType = CUDA_R_32F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
        cublasStatus_t status;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weight.cudaData, AType,
                              m, cudaInput, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            FastllmCudaFree(cudaInput);
            FastllmCudaFree(cudaOutput);
            exit(0);
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float *)weight.extraCudaData[0], k);
        }
    } else {
        FastllmGemvFp32Fp32KernelForBf16<256, 1> <<< k, 256 >>>(cudaInput, (float *)weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

    int outputLen = output.Count(0);
    FastllmCudaFloat2Bf16Kernel <<< (outputLen - 1) / 256 + 1, 256 >>>(cudaOutput, (__nv_bfloat16 *)output.cudaData, outputLen);
    FastllmCudaFree(cudaInput);
    FastllmCudaFree(cudaOutput);
    DeviceSync();
    return true;
}

namespace {
__global__ void FastllmCompactBFloat16ExactKernel(const float *input, uint16_t *output,
                                                uint64_t count, int *inexact) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t bits = __float_as_uint(input[i]);
    output[i] = bits >> 16;
    if ((bits & 0xffffu) || !isfinite(input[i])) atomicExch(inexact, 1);
}
}

bool FastllmCudaTryCompactBFloat16Weight(const fastllm::Data &source, fastllm::Data &compact) {
    if (source.dataDevice != fastllm::DataDevice::CUDA || !source.cudaData || source.dims.size() != 2 ||
        source.dataType != fastllm::DataType::FLOAT32 || source.dims[0] <= 0 || source.dims[1] <= 0 || source.multiDeviceData ||
        FastllmCudaGraphIsCapturingFast() || FastllmCudaRuntimeArch() < 80) return false;
    compact.dataType = fastllm::DataType::BFLOAT16; compact.UpdateUnitSize(); compact.Resize(source.dims);
    compact.ToDevice(fastllm::DataDevice::CUDA, source.dataDeviceIds, false); compact.Allocate(false);
    int *flag = (int*)FastllmCudaMalloc(sizeof(int));
    FastllmCudaMemset0(flag, sizeof(int));
    FastllmCompactBFloat16ExactKernel<<<(source.Count(0) + 255) / 256, 256>>>(
        (const float*)source.cudaData, (uint16_t*)compact.cudaData, source.Count(0), flag);
    int inexact = 1;
    FastllmCudaCopyFromDeviceToHost(&inexact, flag, sizeof(int)); FastllmCudaFree(flag);
    return inexact == 0;
}

bool FastllmCudaBFloat16LinearFloat32(const fastllm::Data &input, const fastllm::Data &weight,
                                     fastllm::Data &output) {
    if (input.dataType != fastllm::DataType::BFLOAT16 || weight.dataType != fastllm::DataType::BFLOAT16 ||
        input.dataDevice != fastllm::DataDevice::CUDA || weight.dataDevice != fastllm::DataDevice::CUDA ||
        !input.cudaData || !weight.cudaData || input.multiDeviceData || weight.multiDeviceData ||
        input.dataDeviceIds != weight.dataDeviceIds || input.dims.empty() || weight.dims.size() != 2 ||
        input.dims.back() <= 0 ||
        input.dims.back() != weight.dims[1] || FastllmCudaRuntimeArch() < 80) return false;
    const int m = input.dims.back(), n = input.Count(0) / m, k = weight.dims[0];
    if (m <= 0 || n <= 0 || k <= 0) return false;
    auto dims = input.dims; dims.back() = k;
    output.dataType = fastllm::DataType::FLOAT32; output.UpdateUnitSize(); output.Resize(dims);
    output.ToDevice(fastllm::DataDevice::CUDA, input.dataDeviceIds, false); output.Allocate(false);
    float alpha = 1.f, beta = 0.f;
    const auto status = cublasGemmEx(getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, k, n, m,
        &alpha, weight.cudaData, CUDA_R_16BF, m, input.cudaData, CUDA_R_16BF, m,
        &beta, output.cudaData, CUDA_R_32F, k, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("BF16 linear with FP32 output failed");
    return true;
}

bool FastllmCudaTryPackBFloat16Weights(const fastllm::Data &first,
        const fastllm::Data *second, fastllm::Data &packed) {
    using namespace fastllm;
    if (FastllmCudaGraphIsCapturing() || FastllmCudaRuntimeArch() < 80 ||
        &packed == &first || &packed == second || packed.isFake || packed.multiDeviceData)
        return false;
    const int device = FastllmCudaGetDevice();
    for (const Data *w : {&first, second}) {
        if (!w) continue;
        if (w->dataType != FLOAT32 || w->dataDevice != CUDA || !w->cudaData ||
            w->multiDeviceData || w->dims.size() != 2 || w->dims[0] <= 0 || w->dims[1] <= 0 ||
            w->dataDeviceIds != std::vector<int>{device} || w->IsRepacked ||
            (packed.cudaData && packed.cudaData == w->cudaData) ||
            (second && w->dims[1] != first.dims[1])) return false;
    }
    const int64_t rows = (int64_t)first.dims[0] + (second ? second->dims[0] : 0);
    if (rows > std::numeric_limits<int>::max()) return false;
    packed.dataType = BFLOAT16; packed.UpdateUnitSize();
    packed.Resize({(int)rows, first.dims[1]});
    packed.ToDevice(CUDA, std::vector<int>{device}, false); packed.Allocate(false);
    int *flag = (int*)FastllmCudaMalloc(sizeof(int));
    FastllmCudaMemset0(flag, sizeof(int));
    uint64_t offset = 0;
    for (const Data *w : {&first, second}) {
        if (!w) continue;
        FastllmCompactBFloat16ExactKernel<<<(w->Count(0) + 255) / 256, 256>>>(
            (const float*)w->cudaData, (uint16_t*)packed.cudaData + offset, w->Count(0), flag);
        offset += w->Count(0);
    }
    int inexact = 1;
    FastllmCudaCopyFromDeviceToHost(&inexact, flag, sizeof(int));
    FastllmCudaFree(flag);
    return inexact == 0;
}

namespace {
__global__ void FastllmSplitLinearFloat32Kernel(const float *input, float *first,
        float *second, uint64_t count, int firstWidth, int width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int col = i % width;
    const uint64_t row = i / width;
    if (col < firstWidth) first[row * firstWidth + col] = input[i];
    else second[row * (width - firstWidth) + col - firstWidth] = input[i];
}
}

bool FastllmCudaBFloat16LinearPairFloat32(const fastllm::Data &input,
        const fastllm::Data &packed, int firstWidth, fastllm::Data &first,
        fastllm::Data &second, fastllm::Data &scratch) {
    using namespace fastllm;
    if (packed.dims.size() != 2 || firstWidth <= 0 || firstWidth >= packed.dims[0] ||
        input.dims.empty() || input.dataDeviceIds != std::vector<int>{FastllmCudaGetDevice()} ||
        &first == &second || &first == &scratch || &second == &scratch)
        return false;
    for (const Data *out : {&first, &second, &scratch})
        if (out == &input || out == &packed || out->multiDeviceData || out->isFake ||
            (out->cudaData && (out->cudaData == input.cudaData || out->cudaData == packed.cudaData)))
            return false;
    if ((first.cudaData && (first.cudaData == second.cudaData || first.cudaData == scratch.cudaData)) ||
        (second.cudaData && second.cudaData == scratch.cudaData)) return false;
    // Validate and submit the GEMM before the split. No intermediate BF16 rounding.
    if (!FastllmCudaBFloat16LinearFloat32(input, packed, scratch)) return false;
    const int width = packed.dims[0];
    for (Data *out : {&first, &second}) {
        auto dims = input.dims; dims.back() = out == &first ? firstWidth : width - firstWidth;
        out->dataType = FLOAT32; out->UpdateUnitSize(); out->Resize(dims);
        out->ToDevice(CUDA, input.dataDeviceIds, false); out->Allocate(false);
    }
    FastllmSplitLinearFloat32Kernel<<<(scratch.Count(0) + 255) / 256, 256>>>(
        (const float*)scratch.cudaData, (float*)first.cudaData, (float*)second.cudaData,
        scratch.Count(0), firstWidth, width);
    return true;
}
