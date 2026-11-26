// Large routed projections on Pascal and Volta. Keep the indexed layout for
// decode and bound temporary packing to a group of experts during prefill.
#pragma once

#include "devices/cuda/fastllm-awq-sm70.cuh"
#include <chrono>
#include <tuple>

namespace fastllm::legacy_experts {
namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA expert projection: ") + cudaGetErrorString(status));
}

struct Buffer {
    void *p = nullptr;
    bool allocate(size_t bytes) {
        return FastllmCudaTryMalloc(&p, bytes) == FASTLLM_CUDA_TRY_MALLOC_SUCCESS;
    }
    ~Buffer() {
        if (p && !FastllmCudaFreeAfterStream(p, cudaStreamPerThread)) FastllmCudaFree(p);
    }
};

__global__ void rankRoutes(const int *ids, int *counts, int *rank, int experts, int routes) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < routes && unsigned(ids[r]) < unsigned(experts)) rank[r] = atomicAdd(counts + ids[r], 1);
}

__global__ void prefixCounts(int *counts, int experts) {
    int base = 0;
    for (int e = 0; e < experts; ++e) {
        int n = counts[e];
        counts[e] = base;
        base += n;
    }
    counts[experts] = base;
}

__global__ void sortRoutes(const int *ids, const int *offsets, int *ranks, int *sorted, int routes,
                           const float *scores, float *sortedScores) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= routes) return;
    const int pos = offsets[ids[r]] + ranks[r];
    sorted[pos] = r;
    if (sortedScores) {
        ranks[r] = pos;
        sortedScores[pos] = scores[r];
    }
}

__global__ void gather(const float *src, half *dst, const int *order, int top, int width) {
    const int row = blockIdx.x;
    const float *input = src + size_t(order[row] / top) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        dst[size_t(row) * width + i] = __float2half(input[i]);
}

template<typename T>
__global__ void scatter(const T *src, float *dst, const int *order, int width) {
    const int row = blockIdx.x;
    float *output = dst + size_t(order ? order[row] : row) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        output[i] = float(src[size_t(row) * width + i]);
}

__global__ void sumRoutes(const float *routes, float *dst, int top, int width, const int *position) {
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float sum = 0;
        for (int j = 0; j < top; ++j) {
            const int route = row * top + j;
            sum += routes[size_t(position ? position[route] : route) * width + i];
        }
        dst[size_t(row) * width + i] = sum;
    }
}

// Two neighboring lanes expand one block-16 record. The four half2 results
// of each lane form an aligned 16-byte store.
struct alignas(16) EightHalves { half2 pairs[4]; };
__global__ void unpackExpertGroup(uint8_t *const *table, const int *offsets,
                                  EightHalves *dense, int words) {
    const int expert = blockIdx.y, word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= words || offsets[expert] == offsets[expert + 1]) return;
    const uint32_t *block = reinterpret_cast<const uint32_t *>(table[expert]) + size_t(word / 2) * 3;
    const unsigned codes = block[word & 1];
    const float scale = __uint_as_float(block[2]);
    EightHalves result;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned pair = codes >> (8 * i);
        const float lo = FastllmMoeNVFP4PseudoBFloat16ToFloat(pair & 15) * FastllmMoeNVFP4MagicScale();
        const float hi = FastllmMoeNVFP4PseudoBFloat16ToFloat((pair >> 4) & 15) * FastllmMoeNVFP4MagicScale();
        result.pairs[i] = __floats2half2_rn(lo * scale, hi * scale);
    }
    dense[size_t(expert) * words + word] = result;
}

__global__ void unpackE4M3(const uint8_t *codes, const float *scales, half *dense,
                           int K, int scaleRows, int scaleCols) {
    const int row = blockIdx.x;
    const int blocks = (K - 1) / scaleCols + 1;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const size_t pos = size_t(row) * K + k;
        const float value = FastllmMoeFp8Traits<float>::fp8ToFloat(codes[pos]) * 256.f;
        dense[pos] = __float2half_rn(value * scales[size_t(row / scaleRows) * blocks + k / scaleCols]);
    }
}

size_t aligned(size_t n) { return (n + 255) & ~size_t(255); }

bool run(const Data &input, Data &output, const Data &ids, const Data &scores, Data &pair, Data &middle,
         Data **weights, int count, const CudaIndexedMoeSlots &slots,
         const CudaMoeActivation &activation, bool exactHalfScales, int arch) {
    const auto type = weights[2]->dataType;
    const bool fp8 = type == FP8_E4M3 && exactHalfScales;
    if ((type != NVFP4_BLOCK_16 && !fp8) || ids.dims.size() != 2 ||
        input.dataType != FLOAT32 || pair.dataType != FLOAT32 || middle.dataType != FLOAT32 ||
        output.dataType != FLOAT32 || scores.dataType != FLOAT32 || ids.dataType != INT32) return false;
    const int experts = count / 2 - 1, rows = input.dims[0], hidden = input.dims[1];
    const int inter = weights[2]->dims[0] / 2, top = ids.dims[1];
    if (rows <= 0 || top <= 0 || rows > 65535 / top || ids.dims[0] != rows ||
        scores.dims != ids.dims || output.dims != input.dims) return false;
    const int routes = rows * top;
    if (routes / experts < 8 || hidden <= 0 || inter <= 0 || hidden % 16 || inter % 16 ||
        size_t(pair.Count(0)) < size_t(routes) * std::max(hidden, 2 * inter) ||
        size_t(middle.Count(0)) < size_t(routes) * inter) return false;
    auto stream = cudaStreamPerThread;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
        return false;
    for (int e = 2; e < count; ++e)
        if (!weights[e] || weights[e]->IsRepacked || weights[e]->dataType != type ||
            (fp8 && weights[e]->extraCudaData.empty())) return false;

    Buffer metadata;
    if (!metadata.allocate(size_t(experts + 1 + 2 * routes) * sizeof(int))) return false;
    auto *offsets = static_cast<int *>(metadata.p), *ranks = offsets + experts + 1, *sorted = ranks + routes;
    const auto *indices = static_cast<const int *>(ids.cudaData);
    check(cudaMemsetAsync(offsets, 0, experts * sizeof(int), stream));
    rankRoutes<<<(routes+255)/256,256,0,stream>>>(indices, offsets, ranks, experts, routes);
    prefixCounts<<<1,1,0,stream>>>(offsets, experts);
    // Only group sizes cross the device boundary. Route IDs, activations and
    // expert outputs stay on the GPU; reserve enough space for a skewed router.
    std::vector<int> starts(experts + 1);
    check(cudaMemcpyAsync(starts.data(), offsets,
                    starts.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    if (starts.back() != routes) throw std::invalid_argument("MoE route outside expert table");
    const bool ordered = arch == 70 && dynamic_cast<const CudaExpertOrderedFp16Activation *>(&activation) != nullptr;
    // The first half of the FP32 middle buffer holds the FP16 activations.
    // Its unused second half has room for one FP32 score per route.
    half *halfMiddle = static_cast<half *>(middle.cudaData);
    float *sortedScores = ordered ? reinterpret_cast<float *>(halfMiddle + size_t(routes) * inter) : nullptr;
    sortRoutes<<<(routes+255)/256,256,0,stream>>>(indices, offsets, ranks, sorted, routes,
                                                static_cast<const float *>(scores.cudaData), sortedScores);

    const bool volta = !fp8 && arch == 70 && hidden % 32 == 0 && inter % 32 == 0 && awq_sm70::Nvfp4Supported();
    const size_t largest = size_t(hidden) * 2 * inter;
    // Stage up to 32 MiB of packed expert weights. One very wide expert can
    // exceed the target, but its storage is independent of prompt length.
    const int group = std::max<size_t>(1, std::min<size_t>(experts, (32u << 20) / (largest / 2 + largest / 8)));
    int maxRows = 0;
    for (int e = 0; e < experts; e += group)
        maxRows = std::max(maxRows, starts[std::min(e + group, experts)] - starts[e]);
    const size_t inputBytes = aligned(size_t(maxRows) * std::max(hidden, inter) * sizeof(half));
    const size_t projectedBytes = aligned(size_t(maxRows) * std::max(hidden, 2 * inter) * sizeof(float));
    const size_t weightBytes = volta ? std::max(largest * sizeof(half), awq_sm70::GroupedNvfp4WorkspaceBytes(hidden, 2 * inter, group)) :
                                       largest * sizeof(half);
    Buffer storage;
    const size_t storageBytes = inputBytes + projectedBytes + weightBytes;
    if (!storage.allocate(storageBytes)) return false;
    auto *x = static_cast<half *>(storage.p);
    auto project = [&](bool down, const float *source) {
        const int K = down ? inter : hidden, N = down ? hidden : 2 * inter;
        const size_t elements = size_t(K) * N, denseBytes = elements * sizeof(half);
        const auto rowStorage = [&](int rows) {
            return aligned(size_t(rows) * K * sizeof(half)) + aligned(size_t(rows) * N * sizeof(float));
        };
        for (int cursor = 0; cursor < experts;) {
            const int e = cursor;
            int n = std::min(experts - e, group - e % group);
            // Keep the original workspace bound even for a heavily skewed
            // router. A smaller row group leaves room for more dense weights.
            if (!volta && !fp8) {
                while (n > 1 && rowStorage(starts[e+n] - starts[e]) + n * denseBytes > storageBytes)
                    n = (n + 1) / 2;
            }
            cursor += n;
            const int first = starts[e], m = starts[e + n] - first;
            if (!m) continue;
            auto *projected = static_cast<uint8_t *>(storage.p) + aligned(size_t(m) * K * sizeof(half));
            auto *packed = static_cast<uint8_t *>(storage.p) + rowStorage(m);
            auto *destination = static_cast<float *>(pair.cudaData) + size_t(first) * N;
            const half *operand = x;
            if (ordered && down) operand = halfMiddle + size_t(first) * K;
            else gather<<<m,256,0,stream>>>(source, x, sorted + first, down ? 1 : top, K);
            auto runCuBlas = [&] {
                auto *w = reinterpret_cast<half *>(packed);
                const int batch = fp8 ? 1 : std::min<size_t>(n, (storageBytes - rowStorage(m)) / denseBytes);
                const int words = elements / 8;
                const float one = 1, zero = 0;
                for (int j = 0; j < n; ++j) {
                    if (!fp8 && j % batch == 0) {
                        const dim3 grid((words + 255) / 256, std::min(batch, n - j));
                        unpackExpertGroup<<<grid,256,0,stream>>>((down ? slots.down : slots.gate) + e + j,
                            offsets + e + j, reinterpret_cast<EightHalves *>(w), words);
                    }
                    half *expertWeight = w + (j % batch) * elements;
                    const int offset = starts[e+j] - first, length = starts[e+j+1] - starts[e+j];
                    if (!length) continue;
                    if (fp8) {
                        const Data &weight = *weights[2 + 2 * (e + j) + int(down)];
                        unpackE4M3<<<N,256,0,stream>>>(static_cast<const uint8_t *>(weight.cudaData), static_cast<const float *>(weight.extraCudaData[0]),
                                                       w, K, weight.blockK, weight.blockM);
                    }
                    auto *y = (ordered ? destination : reinterpret_cast<float *>(projected)) + size_t(offset) * N;
                    if (cublasGemmEx(getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                     N, length, K, &one, expertWeight, CUDA_R_16F, K, operand + size_t(offset) * K,
                                     CUDA_R_16F, K, &zero, y, CUDA_R_32F, N, CUDA_R_32F,
                                     CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                        throw std::runtime_error("cuBLAS grouped expert projection failed");
                }
                return true;
            };
            auto runVolta = [&](bool measureTiles = false) {
                return awq_sm70::GroupedNvfp4((down ? slots.down : slots.gate) + e, n, offsets + e, first, m, K, N, operand,
                                              reinterpret_cast<half *>(projected), packed, stream, measureTiles);
            };
            bool useVolta = volta && n > 1;
            if (useVolta) {
                // Both providers accept the same canonical weights. Measure the
                // projection including transient packing once per density range.
                using Key = std::tuple<int,int,int,int>;
                static thread_local std::map<Key,bool> choices;
                int density = 0;
                for (int avg = routes / experts; avg > 1; avg /= 2) ++density;
                const Key key{FastllmCudaGetDevice(), K, N, density};
                auto it = choices.find(key);
                if (it == choices.end()) {
                    auto time = [&](auto &launch) {
                        check(cudaStreamSynchronize(stream));
                        const auto begin = std::chrono::steady_clock::now();
                        const bool ok = launch();
                        check(cudaStreamSynchronize(stream));
                        return ok ? std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()
                                  : std::numeric_limits<double>::infinity();
                    };
                    runCuBlas();
                    const bool available = runVolta(true);
                    const double blasTime = time(runCuBlas);
                    const bool faster = available && time(runVolta) < blasTime;
                    choices.emplace(key, faster);
                    useVolta = faster; // The measured run already produced the FP16 result.
                } else {
                    useVolta = it->second && runVolta();
                }
            }
            if (useVolta) {
                scatter<<<m,256,0,stream>>>(reinterpret_cast<const half *>(projected),
                                            ordered ? destination : static_cast<float *>(pair.cudaData), ordered ? nullptr : sorted + first, N);
            } else {
                runCuBlas();
                if (!ordered) scatter<<<m,256,0,stream>>>(reinterpret_cast<const float *>(projected),
                                            static_cast<float *>(pair.cudaData), sorted + first, N);
            }
        }
        check(cudaGetLastError());
    };
    project(false, static_cast<const float *>(input.cudaData));
    if (ordered) {
        Data reordered(FLOAT32, scores.dims, CUDA, sortedScores);
        Data narrow(FLOAT16, middle.dims, CUDA, halfMiddle);
        reordered.dataDeviceIds = narrow.dataDeviceIds = middle.dataDeviceIds;
        activation.Run(pair, reordered, narrow, routes, inter);
    } else {
        activation.Run(pair, scores, middle, routes, inter);
    }
    project(true, static_cast<const float *>(middle.cudaData));
    sumRoutes<<<rows,256,0,stream>>>(static_cast<const float *>(pair.cudaData),
                                    static_cast<float *>(output.cudaData), top, hidden, ordered ? ranks : nullptr);
    check(cudaGetLastError());
    return true;
}

} // namespace
} // namespace fastllm::legacy_experts
