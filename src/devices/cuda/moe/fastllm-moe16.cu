// Resident FP16/BF16 SwiGLU experts with device-side routing.
// Model-specific route selection stays outside this shared CUDA backend.
#include "devices/cuda/fastllm-moe.h"
#include "devices/cuda/fastllm-cuda-ordered-reduce.cuh"
#include "fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <type_traits>
namespace {
template<typename T> __device__ T Moe16Cast(float x);
template<> __device__ __nv_bfloat16 Moe16Cast<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template<> __device__ __half Moe16Cast<__half>(float x) { return __float2half_rn(x); }
__device__ float Moe16Float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ float Moe16Float(__half x) { return __half2float(x); }
template<typename T> using Moe16Pair = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
template<typename T> __device__ Moe16Pair<T> Moe16Cast2(float x, float y) {
    if constexpr (std::is_same_v<T, __half>) return __floats2half2_rn(x, y);
    else return __floats2bfloat162_rn(x, y);
}
__device__ float2 Moe16Float2(__half2 x) { return __half22float2(x); }
__device__ float2 Moe16Float2(__nv_bfloat162 x) { return __bfloat1622float2(x); }
using fastllm::cuda::WarpSum;
template<typename T>
__device__ void Moe16ExpertGateRow(const T *input, const T *weight,
        const int *indices, T *middle, int hidden, int intermediate,
        int experts, int topk, int task, int d) {
    int lane = threadIdx.x % 32;
    if (d >= intermediate) return;
    int expert = indices[task];
    if (expert < 0 || expert >= experts) return;
    const T *x = input + (size_t)(task / topk) * hidden;
    const T *w = weight + ((size_t)expert * intermediate * 2 + d) * hidden;
    float gate = 0.f, up = 0.f;
    if (hidden % 8 == 0 && ((reinterpret_cast<uintptr_t>(x) |
            reinterpret_cast<uintptr_t>(w)) & 15) == 0) {
        float ga[4] = {}, ua[4] = {};
        for (int k = lane * 8; k < hidden; k += 256) {
            uint4 xv = *(const uint4 *)(x + k);
            uint4 gv = *(const uint4 *)(w + k);
            uint4 uv = *(const uint4 *)(w + (size_t)intermediate * hidden + k);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 a = Moe16Float2(((const Moe16Pair<T> *)&xv)[j]);
                float2 g = Moe16Float2(((const Moe16Pair<T> *)&gv)[j]);
                float2 u = Moe16Float2(((const Moe16Pair<T> *)&uv)[j]);
                ga[j] += a.x * g.x; ga[j] += a.y * g.y;
                ua[j] += a.x * u.x; ua[j] += a.y * u.y;
            }
        }
        gate = (ga[0] + ga[1]) + (ga[2] + ga[3]);
        up = (ua[0] + ua[1]) + (ua[2] + ua[3]);
    } else {
        for (int k = lane; k < hidden; k += 32) {
            float a = Moe16Float(x[k]);
            gate += a * Moe16Float(w[k]);
            up += a * Moe16Float(w[(size_t)intermediate * hidden + k]);
        }
    }
    gate = WarpSum(gate); up = WarpSum(up);
    if (lane == 0) {
        gate = Moe16Float(Moe16Cast<T>(gate));
        up = Moe16Float(Moe16Cast<T>(up));
        middle[(size_t)task * intermediate + d] = Moe16Cast<T>(gate / (1.f + expf(-gate)) * up);
    }
}
template<typename T>
__device__ void Moe16ExpertDownRow(const T *middle, const T *weight,
        const int *indices, const float *scores, T *parts,
        int hidden, int intermediate, int experts, int task, int d) {
    int lane = threadIdx.x % 32;
    if (d >= hidden) return;
    int expert = indices[task];
    if (expert < 0 || expert >= experts) return;
    const T *x = middle + (size_t)task * intermediate;
    const T *w = weight + ((size_t)expert * hidden + d) * intermediate;
    float sum = 0.f;
    if (intermediate % 8 == 0 && ((reinterpret_cast<uintptr_t>(x) |
            reinterpret_cast<uintptr_t>(w)) & 15) == 0) {
        float partial[4] = {};
        for (int k = lane * 8; k < intermediate; k += 256) {
            uint4 xv = *(const uint4 *)(x + k), wv = *(const uint4 *)(w + k);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 a = Moe16Float2(((const Moe16Pair<T> *)&xv)[j]);
                float2 b = Moe16Float2(((const Moe16Pair<T> *)&wv)[j]);
                partial[j] += a.x * b.x; partial[j] += a.y * b.y;
            }
        }
        sum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
    } else {
        for (int k = lane; k < intermediate; k += 32)
            sum += Moe16Float(x[k]) * Moe16Float(w[k]);
    }
    sum = WarpSum(sum);
    if (lane == 0) parts[(size_t)task * hidden + d] = Moe16Cast<T>(sum * scores[task]);
}
template<typename T>
__global__ void Moe16ExpertGate(const T *input, const T *weight,
        const int *indices, T *middle, int hidden, int intermediate,
        int experts, int topk) {
    Moe16ExpertGateRow<T>(input, weight, indices, middle, hidden, intermediate, experts,
        topk, blockIdx.y, blockIdx.x * 8 + threadIdx.x / 32);
}
template<typename T>
__global__ void Moe16ExpertDown(const T *middle, const T *weight,
        const int *indices, const float *scores, T *parts,
        int hidden, int intermediate, int experts) {
    Moe16ExpertDownRow<T>(middle, weight, indices, scores, parts, hidden, intermediate,
        experts, blockIdx.y, blockIdx.x * 8 + threadIdx.x / 32);
}
// Group routes in their original token order. An expert can occur at most
// once per token. Each expert therefore needs only `rows` task slots.
__global__ void Moe16ExpertRows(const int *indices, int *workspace, int rows, int topk) {
    const int expert = blockIdx.x, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    __shared__ int offsets[4], base;
    if (tid == 0) base = 0;
    __syncthreads();
    for (int start = 0; start < rows * topk; start += 128) {
        int task = start + tid;
        bool match = task < rows * topk && indices[task] == expert;
        unsigned mask = __ballot_sync(0xffffffff, match);
        int rank = __popc(mask & ((1u << lane) - 1));
        if (lane == 0) offsets[warp] = __popc(mask);
        __syncthreads();
        int before = 0;
        for (int w = 0; w < warp; ++w) before += offsets[w];
        if (match) workspace[(size_t)expert * (rows + 1) + base + before + rank] = task;
        __syncthreads();
        if (tid == 0) base += offsets[0] + offsets[1] + offsets[2] + offsets[3];
        __syncthreads();
    }
    if (tid == 0) workspace[(size_t)expert * (rows + 1) + rows] = base;
}

__global__ void Moe16ClearExpertCounts(int *workspace, int rows, int experts) {
    int expert = threadIdx.x;
    if (expert < experts) workspace[(size_t)expert * (rows + 1) + rows] = 0;
}
__global__ void Moe16ScatterExpertRows(const int *indices, int *workspace, int rows, int topk) {
    int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= rows * topk) return;
    int expert = indices[task];
    int offset = atomicAdd(workspace + (size_t)expert * (rows + 1) + rows, 1);
    workspace[(size_t)expert * (rows + 1) + offset] = task;
}
void Moe16BuildExpertRows(const int *indices, int *workspace, int rows, int topk,
        int experts, cudaStream_t stream) {
    if (rows >= 256 && experts >= 32) {
        // Bucket independent token rows without rescanning every route for
        // every expert. Row order may vary; GEMMs scatter back by task id and
        // the final sum still follows ascending expert id.
        Moe16ClearExpertCounts<<<1, 256, 0, stream>>>(workspace, rows, experts);
        Moe16ScatterExpertRows<<<(rows * topk + 255) / 256, 256, 0, stream>>>(
            indices, workspace, rows, topk);
    } else Moe16ExpertRows<<<experts, 128, 0, stream>>>(indices, workspace, rows, topk);
}
template<int TileRows = 16>
__global__ void Moe16ExpertTiles(int *workspace, int rows, int experts) {
    int e = threadIdx.x, lane = e % 32, warp = e / 32;
    int rowsForExpert = e < experts ? workspace[(size_t)e * (rows + 1) + rows] : 0;
    int count = (rowsForExpert + TileRows - 1) / TileRows, inclusive = count;
    for (int d = 1; d < 32; d *= 2) {
        int v = __shfl_up_sync(0xffffffff, inclusive, d);
        if (lane >= d) inclusive += v;
    }
    __shared__ int totals[8];
    if (lane == 31) totals[warp] = inclusive;
    __syncthreads();
    int before = inclusive - count, total = 0;
    for (int w = 0; w < 8; ++w) {
        total += totals[w];
        if (w < warp) before += totals[w];
    }
    int *tiles = workspace + (size_t)experts * (rows + 1);
    if (e == 0) tiles[0] = total;
    for (int m = 0; m < count; ++m) tiles[1 + before + m] = e * (rows + 1) + m * TileRows;
}

// Small decode batches build stable expert rows and tile descriptors in one
// CTA. Warp peer masks give each route its rank; preceding warp/chunk counts
// preserve token order without atomically scattering the output list.
__global__ void Moe16ExpertRowsAndTiles(const int *indices, int *workspace,
        int rows, int topk, int experts) {
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    __shared__ int counts[8][256], totals[256], warpTotals[8];
    totals[tid] = 0;
    for (int start = 0; start < rows * topk; start += 256) {
#pragma unroll
        for (int w = 0; w < 8; ++w) counts[w][tid] = 0;
        __syncthreads();
        const int task = start + tid, expert = task < rows * topk ? indices[task] : -1;
        unsigned peers;
#if __CUDA_ARCH__ >= 700
        peers = __match_any_sync(0xffffffff, expert);
#else
        peers = 0;
        for (int i = 0; i < 32; ++i)
            if (__shfl_sync(0xffffffff, expert, i) == expert) peers |= 1u << i;
#endif
        if (expert >= 0 && lane == __ffs(peers) - 1) counts[warp][expert] = __popc(peers);
        __syncthreads();
        if (expert >= 0) {
            int rank = totals[expert] + __popc(peers & ((1u << lane) - 1));
            for (int w = 0; w < warp; ++w) rank += counts[w][expert];
            workspace[(size_t)expert * (rows + 1) + rank] = task;
        }
        __syncthreads();
#pragma unroll
        for (int w = 0; w < 8; ++w) totals[tid] += counts[w][tid];
        __syncthreads();
    }
    const int count = (totals[tid] + 15) / 16;
    if (tid < experts) workspace[(size_t)tid * (rows + 1) + rows] = totals[tid];
    int inclusive = count;
    for (int d = 1; d < 32; d *= 2) {
        int v = __shfl_up_sync(0xffffffff, inclusive, d);
        if (lane >= d) inclusive += v;
    }
    if (lane == 31) warpTotals[warp] = inclusive;
    __syncthreads();
    int before = inclusive - count, total = 0;
    for (int w = 0; w < 8; ++w) {
        total += warpTotals[w];
        if (w < warp) before += warpTotals[w];
    }
    int *tiles = workspace + (size_t)experts * (rows + 1);
    if (tid == 0) tiles[0] = total;
    for (int m = 0; m < count; ++m) tiles[1 + before + m] = tid * (rows + 1) + m * 16;
}

void Moe16BuildExpertTiles(const int *indices, int *workspace, int rows, int topk,
                           int experts, cudaStream_t stream) {
    if (rows * topk <= 512)
        Moe16ExpertRowsAndTiles<<<1, 256, 0, stream>>>(indices, workspace, rows, topk, experts);
    else {
        Moe16BuildExpertRows(indices, workspace, rows, topk, experts, stream);
        Moe16ExpertTiles<16><<<1, 256, 0, stream>>>(workspace, rows, experts);
    }
}

__device__ __forceinline__ void Moe16CopyAsync16(void *destination, const void *source, bool valid) {
#if __CUDA_ARCH__ >= 800
    unsigned address = (unsigned)__cvta_generic_to_shared(destination);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
        "r"(address), "l"(source), "r"(valid ? 16 : 0));
#else
    *(uint4 *)destination = valid ? *(const uint4 *)source : make_uint4(0, 0, 0, 0);
#endif
}

// Up to 16 token rows share a weight tile, including experts with only one
// active row. Compact descriptors avoid empty expert launches and separate
// sparse GEMVs that reread weights for each token. Tensor Core products
// accumulate in FP32; gate/up and routed-down boundaries round to storage
// precision. WMMA requires SM70+ for FP16 and SM80+ for BF16.
template<typename T, bool Gate, int N, int K>
__global__ void Moe16ExpertGemm(const T *input, const T *weight,
        const float *scores, const int *workspace, T *output,
        int rows, int hidden, int intermediate, int experts, int topk) {
#if __CUDA_ARCH__ >= 700
    if constexpr (std::is_same_v<T, __half> || __CUDA_ARCH__ >= 800) {
    using namespace nvcuda;
    constexpr int M = 16, LD = K + 8;
    const int *tiles = workspace + (size_t)experts * (rows + 1);
    if (blockIdx.y >= tiles[0]) return;
    const int index = tiles[1 + blockIdx.y], expert = index / (rows + 1);
    const int start = index % (rows + 1), count = workspace[expert * (rows + 1) + rows];
    const int inputWidth = Gate ? hidden : intermediate;
    const int outputWidth = Gate ? intermediate : hidden;
    const int column = blockIdx.x * N, tid = threadIdx.x, warp = tid / 32;
    __shared__ __align__(32) T a[2 * M * LD], b[2 * N * LD], u[Gate ? 2 * N * LD : 1];
    __shared__ int tasks[M], inputRows[M];
    if (tid < M) {
        int task = start + tid < count ? workspace[index + tid] : -1;
        tasks[tid] = task;
        inputRows[tid] = Gate && task >= 0 ? task / topk : task;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc, up;
    wmma::fill_fragment(acc, 0.f);
    if (Gate) wmma::fill_fragment(up, 0.f);
    auto load = [&](int k, int stage) {
        auto *pa = a + stage * M * LD;
        auto *pb = b + stage * N * LD;
        auto *pu = Gate ? u + stage * N * LD : nullptr;
        if (inputWidth % 8 == 0 && ((reinterpret_cast<uintptr_t>(input) |
            reinterpret_cast<uintptr_t>(weight)) & 15) == 0) {
            for (int i = tid * 8; i < M * K; i += N * 2 * 8) {
                int row = inputRows[i / K], d = k + i % K;
                bool valid = row >= 0 && d < inputWidth;
                Moe16CopyAsync16(pa + (i / K) * LD + i % K,
                    input + (valid ? (size_t)row * inputWidth + d : 0), valid);
            }
            for (int i = tid * 8; i < N * K; i += N * 2 * 8) {
                int n = column + i / K, d = k + i % K;
                size_t offset = ((size_t)expert * outputWidth * (Gate ? 2 : 1) + n) * inputWidth + d;
                bool valid = n < outputWidth && d < inputWidth;
                Moe16CopyAsync16(pb + (i / K) * LD + i % K, weight + (valid ? offset : 0), valid);
                if (Gate) Moe16CopyAsync16(pu + (i / K) * LD + i % K,
                    weight + (valid ? offset + (size_t)outputWidth * inputWidth : 0), valid);
            }
        } else {
            for (int i = tid; i < M * K; i += N * 2) {
                int row = inputRows[i / K], d = k + i % K;
                pa[(i / K) * LD + i % K] = row >= 0 && d < inputWidth
                    ? input[(size_t)row * inputWidth + d] : Moe16Cast<T>(0.f);
            }
            for (int i = tid; i < N * K; i += N * 2) {
                int n = column + i / K, d = k + i % K;
                size_t offset = ((size_t)expert * outputWidth * (Gate ? 2 : 1) + n) * inputWidth + d;
                pb[(i / K) * LD + i % K] = n < outputWidth && d < inputWidth ? weight[offset] : Moe16Cast<T>(0.f);
                if (Gate) pu[(i / K) * LD + i % K] = n < outputWidth && d < inputWidth
                    ? weight[offset + (size_t)outputWidth * inputWidth] : Moe16Cast<T>(0.f);
            }
        }
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;");
#endif
    };
    load(0, 0);
    for (int k = 0; k < inputWidth; k += K) {
        int stage = (k / K) % 2;
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;");
#endif
        __syncthreads();
        if (k + K < inputWidth) load(k + K, 1 - stage);
        auto *pa = a + stage * M * LD;
        auto *pb = b + stage * N * LD;
        auto *pu = Gate ? u + stage * N * LD : nullptr;
        for (int j = 0; j < K; j += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, pa + j, LD);
            wmma::load_matrix_sync(fb, pb + warp * 16 * LD + j, LD);
            wmma::mma_sync(acc, fa, fb, acc);
            if (Gate) {
                wmma::load_matrix_sync(fb, pu + warp * 16 * LD + j, LD);
                wmma::mma_sync(up, fa, fb, up);
            }
        }
        __syncthreads();
    }
    // Staging is finished; reuse it for the two FP32 epilogue tiles.
    float *result = (float *)b, *upResult = Gate ? (float *)u : nullptr;
    wmma::store_matrix_sync(result + warp * 16, acc, N, wmma::mem_row_major);
    if (Gate) wmma::store_matrix_sync(upResult + warp * 16, up, N, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < M * N; i += N * 2) {
        int task = tasks[i / N], n = column + i % N;
        if (task < 0 || n >= outputWidth) continue;
        float v = result[i];
        if (Gate) {
            v = Moe16Float(Moe16Cast<T>(v));
            float uv = Moe16Float(Moe16Cast<T>(upResult[i]));
            v = v / (1.f + expf(-v)) * uv;
        } else v *= scores[task];
        output[(size_t)task * outputWidth + n] = Moe16Cast<T>(v);
    }
    }
#endif
}

int Moe16CompiledArch() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    static thread_local int previous = -1, capability = 0;
    if (device != previous) {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes, Moe16ExpertGemm<__half, true, 32, 128>) != cudaSuccess) return 0;
        capability = std::min(attributes.binaryVersion, attributes.ptxVersion);
        previous = device;
    }
    return capability;
}
template<typename T>
bool Moe16TensorCores() {
    return Moe16CompiledArch() >= (std::is_same_v<T, __half> ? 70 : 80);
}
bool Moe16UseBatched(int rows, int hidden, int intermediate, bool fp16) {
    // Per-route GEMV avoids poorly occupied expert tiles for small blocks.
    // Larger blocks reuse weights better with the existing grouped GEMM.
    if (!fp16 || rows < 4 || rows > 32 || hidden < 512 || intermediate < 128 ||
        hidden % 8 != 0 || intermediate % 8 != 0 ||
        hidden > INT_MAX / (rows * 16) || intermediate > INT_MAX / (rows * 32)) return false;
    const int arch = Moe16CompiledArch();
    return arch >= 70 && arch < 80;
}
size_t Moe16MetadataBytes(int rows, int experts, int topk) {
    const size_t words = size_t(experts) * (rows + 1) + experts + (rows * topk + 15) / 16 + 1;
    return (words * sizeof(int) + 255) / 256 * 256;
}
// The expert table is rebuilt on the GPU on every graph replay. cuBLAS sees a
// fixed route count, with one row per route, and needs no host route readback.
__global__ void Moe16LinearPointers(const __half *input, const __half *gate,
        const __half *down, const int *indices, __half *gateOutput,
        __half *middle, float *downOutput, void **pointers,
        int tasks, int hidden, int intermediate, int topk) {
    const int task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= tasks) return;
    pointers[task] = (void *)(gate + size_t(indices[task]) * 2 * intermediate * hidden);
    pointers[task + tasks] = (void *)(input + size_t(task / topk) * hidden);
    pointers[task + 2 * tasks] = gateOutput + size_t(task) * 2 * intermediate;
    pointers[task + 3 * tasks] = (void *)(down + size_t(indices[task]) * hidden * intermediate);
    pointers[task + 4 * tasks] = middle + size_t(task) * intermediate;
    pointers[task + 5 * tasks] = downOutput + size_t(task) * hidden;
}
__global__ void Moe16ScaleLinearParts(const float *input, const float *scores,
        __half *output, int count, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half_rn(input[i] * scores[i / hidden]);
}
// Borrow scratch for the existing FLOAT32 SwiGLU operator. Its FP16 overload
// rounds intermediate sigmoid arithmetic, unlike the FP32 activation contract
// used here. Explicit storage conversions preserve gate/up and middle rounding.
bool Moe16BatchedLinear(const void *input, const void *gate, const void *down,
        const int *indices, const float *scores, void *middle, void *parts,
        int rows, int hidden, int intermediate, int experts, int topk, int *workspace) {
    const int tasks = rows * topk;
    auto **pointers = (void **)((char *)workspace + Moe16MetadataBytes(rows, experts, topk));
    auto *gateOutput = (__half *)(pointers + 6 * tasks);
    auto *scratch = (float *)(gateOutput + size_t(tasks) * 2 * intermediate);
    auto *floatMiddle = scratch + size_t(tasks) * 2 * intermediate;
    fastllm::Data gateView(fastllm::FLOAT32, {tasks, 2 * intermediate});
    fastllm::Data middleView(fastllm::FLOAT32, {tasks, intermediate});
    gateView.isFake = middleView.isFake = true;
    gateView.dataDevice = middleView.dataDevice = fastllm::DataDevice::CUDA;
    gateView.cudaData = scratch;
    middleView.cudaData = floatMiddle;
    Moe16LinearPointers<<<(tasks + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        (const __half *)input, (const __half *)gate, (const __half *)down, indices,
        gateOutput, (__half *)middle, scratch, pointers, tasks, hidden, intermediate, topk);
    const float one = 1.f, zero = 0.f;
    auto handle = getFastllmCublasHandle();
    if (cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2 * intermediate, 1, hidden,
            &one, (const void *const *)pointers, CUDA_R_16F, hidden,
            (const void *const *)(pointers + tasks), CUDA_R_16F, hidden,
            &zero, pointers + 2 * tasks, CUDA_R_16F, 2 * intermediate, tasks,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) return false;
    if (!FastllmHalfToFloat(gateOutput, scratch, tasks * 2 * intermediate) ||
        !FastllmCudaSwiglu(gateView, middleView) ||
        !FastllmFloatToHalf(floatMiddle, middle, tasks * intermediate)) return false;
    // Reuse activation scratch only after SwiGLU and its FP16 conversion.
    // Down stays FP32 until route weighting, then rounds once to FP16.
    if (cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1, intermediate,
            &one, (const void *const *)(pointers + 3 * tasks), CUDA_R_16F, intermediate,
            (const void *const *)(pointers + 4 * tasks), CUDA_R_16F, intermediate,
            &zero, pointers + 5 * tasks, CUDA_R_32F, hidden, tasks,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) return false;
    Moe16ScaleLinearParts<<<(tasks * hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        scratch, scores, (__half *)parts, tasks * hidden, hidden);
    return cudaGetLastError() == cudaSuccess;
}
template<typename T, bool Shared>
__global__ void Moe16ExpertSum(const T *parts, const int *indices, T *output,
                                int hidden, int topk, const T *shared,
                                const T *sharedGate, int sharedGateStride) {
    int d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    __shared__ int order[16];
    // Sort the routes once per CTA, preserving the reference accumulation order.
    if (threadIdx.x < topk) {
        int k = threadIdx.x, rank = 0;
        for (int j = 0; j < topk; ++j)
            rank += indices[row * topk + j] < indices[row * topk + k];
        order[rank] = k;
    }
    __syncthreads();
    if (d >= hidden) return;
    float sum = 0.f;
    for (int rank = 0; rank < topk; ++rank)
        sum += Moe16Float(parts[((size_t)row * topk + order[rank]) * hidden + d]);
    auto result = Moe16Cast<T>(sum);
    if constexpr (Shared) {
        // Preserve sigmoid, shared product, routed sum and final add rounding.
        auto probability = Moe16Cast<T>(1.f / (1.f + expf(-Moe16Float(sharedGate[(size_t)row * sharedGateStride]))));
        auto contribution = Moe16Cast<T>(Moe16Float(shared[(size_t)row * hidden + d]) *
                                             Moe16Float(probability));
        result = Moe16Cast<T>(Moe16Float(result) + Moe16Float(contribution));
    }
    output[(size_t)row * hidden + d] = result;
}

template<typename T, bool Shared>
__global__ void Moe16ExpertSumVector(const T *parts, const int *indices, T *output,
                                      int rows, int hidden, int topk, const T *shared,
                                      const T *sharedGate, int sharedGateStride) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    __shared__ int order[4][16];
    if (lane < topk) {
        int expert = indices[row * topk + lane], rank = 0;
        for (int j = 0; j < topk; ++j) rank += indices[row * topk + j] < expert;
        order[warp][rank] = lane;
    }
    __syncwarp();
    float probability = 0.f;
    if constexpr (Shared)
        probability =
            Moe16Float(Moe16Cast<T>(1.f / (1.f + expf(-Moe16Float(sharedGate[(size_t)row * sharedGateStride])))));
    for (int d = (blockIdx.y * 32 + lane) * 8; d < hidden; d += gridDim.y * 32 * 8) {
        float sums[8] = {};
        for (int k = 0; k < topk; ++k) {
            uint4 value = *reinterpret_cast<const uint4 *>(
                parts + ((size_t)row * topk + order[warp][k]) * hidden + d);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sums[j] += Moe16Float(reinterpret_cast<T *>(&value)[j]);
        }
        uint4 value;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            auto result = Moe16Cast<T>(sums[j]);
            if constexpr (Shared) {
                auto contribution =
                    Moe16Cast<T>(Moe16Float(shared[(size_t)row * hidden + d + j]) * probability);
                result = Moe16Cast<T>(Moe16Float(result) + Moe16Float(contribution));
            }
            reinterpret_cast<T *>(&value)[j] = result;
        }
        *reinterpret_cast<uint4 *>(output + (size_t)row * hidden + d) = value;
    }
}
template<typename T, bool Shared>
void Moe16SumExperts(const T *parts, const int *indices, T *output, int rows,
                      int hidden, int topk, cudaStream_t stream, const T *shared,
                      const T *sharedGate, int sharedGateStride) {
    // A warp per row amortizes route sorting and vectorizes expert reads.
    // Split wider rows across CTAs when row parallelism alone is insufficient;
    // long prefills retain one CTA per four rows. Small/unaligned views keep
    // the scalar path and every element retains its expert accumulation order.
    if (rows >= 128 && hidden >= 512 && hidden % 8 == 0 &&
        ((reinterpret_cast<uintptr_t>(parts) | reinterpret_cast<uintptr_t>(output)) & 15) == 0) {
        int stripes = std::min((hidden + 255) / 256, (2048 + rows - 1) / rows);
        Moe16ExpertSumVector<T, Shared><<<dim3((rows + 3) / 4, stripes), 128, 0, stream>>>(parts, indices, output, rows,
                                                                          hidden, topk, shared, sharedGate, sharedGateStride);
    } else
        Moe16ExpertSum<T, Shared><<<dim3((hidden + 255) / 256, rows), 256, 0, stream>>>(
            parts, indices, output, hidden, topk, shared, sharedGate, sharedGateStride);
}
}

#include "fastllm-moe16-sm90.cuh"

template<typename T>
bool FastllmCudaMoe16Impl(const void *input, const void *gateUp, const void *down, const int *indices,
    const float *scores, void *middle, void *parts, void *output, int rows, int hidden,
    int intermediate, int experts, int topk, void *stream, int *workspace, void *packed,
    const void *shared, const void *sharedGate, int sharedGateStride) {
    if ((shared == nullptr) != (sharedGate == nullptr) || sharedGateStride < 1)
        return false;
    if (!input || !gateUp || !down || !indices || !scores || !middle || !parts || !output ||
        rows < 1 || hidden < 1 || intermediate < 1 || experts < 1 || topk < 1 ||
        topk > experts || topk > 16 || rows > 65535 / topk) return false;
    cudaStream_t s = stream ? (cudaStream_t)stream : cudaStreamPerThread;
#if CUDART_VERSION >= 12000
    if (workspace && packed &&
        FastllmCudaMoe16PackedBytes(rows, hidden, intermediate, experts, topk, std::is_same_v<T, __half>) &&
        Moe16ExpertsTma<T>(input, gateUp, down, indices, scores, middle, parts, output, rows, hidden,
                         intermediate, experts, topk, s, workspace, packed, shared, sharedGate, sharedGateStride))
        return cudaGetLastError() == cudaSuccess;
#endif
    const bool batched = workspace && s == cudaStreamPerThread &&
        ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(gateUp) |
          reinterpret_cast<uintptr_t>(down) | reinterpret_cast<uintptr_t>(middle)) & 15) == 0 &&
        Moe16UseBatched(rows, hidden, intermediate, std::is_same_v<T, __half>);
    if (batched && !Moe16BatchedLinear(input, gateUp, down, indices, scores, middle, parts,
            rows, hidden, intermediate, experts, topk, workspace)) return false;
    bool grouped = !batched && workspace && rows >= 4 && hidden >= 512 && intermediate >= 128 &&
        experts <= 256 && Moe16TensorCores<T>();
    if (grouped) {
        Moe16BuildExpertTiles(indices, workspace, rows, topk, experts, s);
        int tiles = min(experts, rows * topk) + (rows * topk + 15) / 16;
        if (rows <= 32) {
            Moe16ExpertGemm<T, true, 32, 128><<<dim3((intermediate + 31) / 32, tiles), 64, 0, s>>>(
                (const T *)input, (const T *)gateUp, scores, workspace,
                (T *)middle, rows, hidden, intermediate, experts, topk);
            Moe16ExpertGemm<T, false, 32, 128><<<dim3((hidden + 31) / 32, tiles), 64, 0, s>>>(
                (const T *)middle, (const T *)down, scores, workspace,
                (T *)parts, rows, hidden, intermediate, experts, topk);
        } else {
            Moe16ExpertGemm<T, true, 64, 64><<<dim3((intermediate + 63) / 64, tiles), 128, 0, s>>>(
                (const T *)input, (const T *)gateUp, scores, workspace,
                (T *)middle, rows, hidden, intermediate, experts, topk);
            Moe16ExpertGemm<T, false, 64, 64><<<dim3((hidden + 63) / 64, tiles), 128, 0, s>>>(
                (const T *)middle, (const T *)down, scores, workspace,
                (T *)parts, rows, hidden, intermediate, experts, topk);
        }
    }
    if (!grouped && !batched) {
        Moe16ExpertGate<T><<<dim3((intermediate + 7) / 8, rows * topk), 256, 0, s>>>(
            (const T *)input, (const T *)gateUp, indices,
            (T *)middle, hidden, intermediate, experts, topk);
        Moe16ExpertDown<T><<<dim3((hidden + 7) / 8, rows * topk), 256, 0, s>>>(
            (const T *)middle, (const T *)down, indices, scores,
            (T *)parts, hidden, intermediate, experts);
    }
    if (shared)
        Moe16SumExperts<T, true>((const T *)parts, indices, (T *)output, rows, hidden,
                               topk, s, (const T *)shared, (const T *)sharedGate, sharedGateStride);
    else
        Moe16SumExperts<T, false>((const T *)parts, indices, (T *)output, rows, hidden,
                                topk, s, nullptr, nullptr, 1);
    return cudaGetLastError() == cudaSuccess;
}

std::size_t FastllmCudaMoe16WorkspaceBytes(int rows, int hidden, int intermediate,
        int experts, int topk, bool fp16) {
    if (rows < 1 || hidden < 1 || intermediate < 1 || experts < 1 || topk < 1 ||
        topk > experts || topk > 16 || rows > 65535 / topk) return 0;
    size_t bytes = Moe16MetadataBytes(rows, experts, topk);
    if (Moe16UseBatched(rows, hidden, intermediate, fp16)) {
        const size_t tasks = size_t(rows) * topk;
        bytes += tasks * (6 * sizeof(void *) + size_t(2) * intermediate * sizeof(__half) +
                         std::max(size_t(3) * intermediate, size_t(hidden)) * sizeof(float));
    }
    return bytes;
}

bool FastllmCudaMoe16(const void *input, const void *gateUp, const void *down, const int *indices,
    const float *scores, void *middle, void *parts, void *output, int rows, int hidden,
    int intermediate, int experts, int topk, void *stream, int *workspace, void *packed,
    const void *shared, const void *sharedGate, int sharedGateStride, bool fp16) {
    return fp16 ? FastllmCudaMoe16Impl<__half>(input, gateUp, down, indices, scores, middle, parts, output, rows, hidden, intermediate, experts, topk, stream, workspace, packed, shared, sharedGate, sharedGateStride)
                : FastllmCudaMoe16Impl<__nv_bfloat16>(input, gateUp, down, indices, scores, middle, parts, output, rows, hidden, intermediate, experts, topk, stream, workspace, packed, shared, sharedGate, sharedGateStride);
}
