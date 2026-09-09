#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {
void CheckCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << cudaGetErrorString(result) << '\n';
        // A failed rank cannot leave its peer waiting inside a collective.
        std::exit(2);
    }
}

template <typename T> T FromFloat(float value) { return (T)value; }
template <> half FromFloat<half>(float value) { return __float2half_rn(value); }
template <> __nv_bfloat16 FromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}
template <typename T> float ToFloat(T value) { return (float)value; }
template <> float ToFloat<half>(half value) { return __half2float(value); }
template <> float ToFloat<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T> bool Run(const std::vector<int> &devices, int dataType, int count) {
    std::atomic<int> errors{0};
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 2; ++rank) {
        workers.emplace_back([&, rank]() {
            const int device = devices[rank];
            CheckCuda(cudaSetDevice(device));
            std::vector<T> input(count), output(count);
            void *send = nullptr, *recv = nullptr;
            const size_t bytes = count * sizeof(T);
            CheckCuda(cudaMalloc(&send, bytes));
            CheckCuda(cudaMalloc(&recv, bytes));
            // Alternate generations, roots, and in-place/out-of-place calls.
            for (int iteration = 0; iteration < 24; ++iteration) {
                auto value = [&](int r, int i) {
                    return FromFloat<T>((float)((i * 13 + r * 31 + iteration * 17) % 101 - 50));
                };
                for (int i = 0; i < count; ++i)
                    input[i] = value(rank, i);
                CheckCuda(cudaMemcpyAsync(send, input.data(), bytes, cudaMemcpyHostToDevice,
                                          cudaStreamPerThread));
                const bool inPlace = (iteration / 4) % 2 == 0;
                void *destination = inPlace ? send : recv;
                const int operation = iteration % 4;
                const int rootRank = (iteration / 8) % 2;
                if (operation == 0) {
                    FastllmNcclAllReduce(send, destination, count, dataType, device);
                } else if (operation == 1) {
                    FastllmNcclAllReduceNoCustom(send, destination, count, dataType, device);
                } else if (operation == 2) {
                    FastllmNcclBroadcastFrom(send, destination, count, dataType, devices[rootRank],
                                             device);
                } else {
                    FastllmNcclReduce(send, destination, count, dataType, devices[rootRank],
                                      device);
                }
                if (operation == 3 && rank != rootRank)
                    continue;
                CheckCuda(cudaMemcpyAsync(output.data(), destination, bytes, cudaMemcpyDeviceToHost,
                                          cudaStreamPerThread));
                CheckCuda(cudaStreamSynchronize(cudaStreamPerThread));
                for (int i = 0; i < count; ++i) {
                    const T expected =
                        operation == 2 ? value(rootRank, i)
                                       : FromFloat<T>(ToFloat(value(0, i)) + ToFloat(value(1, i)));
                    if (ToFloat(output[i]) != ToFloat(expected)) {
                        if (errors.fetch_add(1) == 0) {
                            std::cerr << "Mismatch: type=" << dataType << " count=" << count
                                      << " rank=" << rank << " iteration=" << iteration
                                      << " index=" << i << " expected=" << ToFloat(expected)
                                      << " actual=" << ToFloat(output[i]) << '\n';
                        }
                    }
                }
            }
            CheckCuda(cudaFree(send));
            CheckCuda(cudaFree(recv));
        });
    }
    for (auto &worker : workers)
        worker.join();
    return errors.load() == 0;
}

template <typename T, typename U> T Bits(U value) {
    static_assert(sizeof(T) == sizeof(U), "Bit widths must match");
    T result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

template <typename T, int DataType> T RawValue(int rank, int i, int iteration) {
    if constexpr (DataType == fastllm::FLOAT16 || DataType == fastllm::BFLOAT16) {
        if (iteration < 2)
            return rank == 0 ? (T)i : (T)(i * 37 + 19);
        if (iteration < 4)
            return rank == 0 ? (T)i : (T)(i ^ 0x8000);
        return rank == 0 ? (T)(i * 13 + iteration * 997)
                         : (T)((i & 1) ? 0x1000 : 0x8000);
    } else {
        // Explicit zeros, subnormals, infinities, NaNs and integer extrema,
        // followed by broad full-width patterns and exact cancellation.
        const uint32_t edge[] = {0, 0x80000000U, 1, 0x80000001U, 0x007fffffU,
            0x00800000U, 0x7f7fffffU, 0xff7fffffU, 0x7f800000U, 0xff800000U,
            0x7fc00000U, 0x7f800001U, 0xffffffffU, 0x7fffffffU, 0x3f800000U,
            0x33800000U, 0x7f, 0x80, 0xff};
        uint32_t bits = i < 361 ? edge[(rank == 0 ? i : i / 19) % 19]
            : (uint32_t)i * 1664525U + 1013904223U + (uint32_t)iteration * 997U;
        if (i >= 361 && rank == 1)
            bits = iteration < 4 ? bits * 37U + 19U : bits ^ 0x80000000U;
        if constexpr (sizeof(T) == 1)
            return Bits<T>((uint8_t)bits);
        else
            return Bits<T>(bits);
    }
}

// Oracle follows the existing CPU SUM, including FP16's software conversion
// and integer narrowing. FP32/BF16 NaN payloads may differ across CPU/GPU.
template <typename T, int DataType> T ReferenceSum(T a, T b) {
    if constexpr (DataType == fastllm::INT8 || DataType == fastllm::INT32) {
        return (T)((int64_t)a + (int64_t)b);
    } else {
        auto toFloat = [](T value) {
            if constexpr (DataType == fastllm::FLOAT16)
                return fastllm::half_to_float(value);
            else if constexpr (DataType == fastllm::BFLOAT16)
                return Bits<float>((uint32_t)value << 16);
            else
                return (float)value;
        };
        float sum = 0.0f;
        sum += toFloat(a);
        sum += toFloat(b);
        if constexpr (DataType == fastllm::FLOAT16)
            return fastllm::float_to_half(sum);
        else if constexpr (DataType == fastllm::BFLOAT16) {
            uint32_t bits = Bits<uint32_t>(sum);
            bits += 0x7fffU + ((bits >> 16) & 1U);
            return (T)(bits >> 16);
        } else
            return sum;
    }
}

template <typename T, int DataType> bool SameResult(T actual, T expected) {
    if constexpr (DataType == fastllm::FLOAT32) {
        if ((Bits<uint32_t>(expected) & 0x7fffffffU) > 0x7f800000U)
            return (Bits<uint32_t>(actual) & 0x7fffffffU) > 0x7f800000U;
    } else if constexpr (DataType == fastllm::BFLOAT16) {
        if ((expected & 0x7fffU) > 0x7f80U)
            return (actual & 0x7fffU) > 0x7f80U;
    }
    return std::memcmp(&actual, &expected, sizeof(T)) == 0;
}

// Each dtype crosses the byte-based dispatch boundary and grows/reuses the
// same thread's pinned allocation up to 10 MiB, then returns to smaller sizes.
template <typename T, int DataType> bool RunBits(const std::vector<int> &devices) {
    std::atomic<int> errors{0};
    std::vector<std::thread> workers;
    for (int rank = 0; rank < 2; ++rank) {
        workers.emplace_back([&, rank]() {
            CheckCuda(cudaSetDevice(devices[rank]));
            T *send = nullptr, *recv = nullptr;
            const size_t maxBytes = 10 * 1024 * 1024;
            CheckCuda(cudaMalloc((void **)&send, maxBytes));
            CheckCuda(cudaMalloc((void **)&recv, maxBytes));
            const int boundary = 64 * 1024 / sizeof(T);
            for (int count : {boundary - 1, boundary, boundary + 1, 65536,
                              (int)(maxBytes / sizeof(T)), 65536, boundary - 1}) {
                const size_t bytes = (size_t)count * sizeof(T);
                std::vector<T> input(count), output(count);
                for (int iteration = 0; iteration < 8; ++iteration) {
                    auto value = [&](int r, int i) { return RawValue<T, DataType>(r, i, iteration); };
                    for (int i = 0; i < count; ++i)
                        input[i] = value(rank, i);
                    CheckCuda(cudaMemcpyAsync(send, input.data(), bytes, cudaMemcpyHostToDevice,
                                              cudaStreamPerThread));
                    void *destination = iteration % 2 == 0 ? send : recv;
                    const bool reduce = iteration >= 4;
                    const int root = (iteration / 2) % 2;
                    if (reduce)
                        FastllmNcclReduce(send, destination, count, DataType,
                                          devices[root], devices[rank]);
                    else
                        FastllmNcclAllReduceNoCustom(send, destination, count,
                                                     DataType, devices[rank]);
                    if (reduce && rank != root)
                        continue;
                    CheckCuda(cudaMemcpyAsync(output.data(), destination, bytes,
                                              cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    CheckCuda(cudaStreamSynchronize(cudaStreamPerThread));
                    for (int i = 0; i < count; ++i) {
                        T expected = ReferenceSum<T, DataType>(value(0, i), value(1, i));
                        if (!SameResult<T, DataType>(output[i], expected) && errors.fetch_add(1) == 0) {
                            std::cerr << "Bits mismatch: type=" << DataType
                                      << " count=" << count << " rank=" << rank
                                      << " iteration=" << iteration << " index=" << i
                                      << " expected=" << expected << " actual=" << output[i] << '\n';
                        }
                    }
                }
            }
            CheckCuda(cudaFree(send));
            CheckCuda(cudaFree(recv));
        });
    }
    for (auto &worker : workers)
        worker.join();
    return errors.load() == 0;
}
} // namespace

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 2) {
        std::cout << "SKIP: host collective regression requires two CUDA GPUs\n";
        return 77;
    }
    // Force host staging even on development machines with CUDA peer access.
#ifdef _WIN32
    _putenv_s("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "0");
#else
    setenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "0", 1);
#endif
    bool ok = true;
    for (int group = 0; group < 2; ++group) {
        // New threaded TP runners initialize their communicator independently
        // of the legacy MultiCuda operator. Neither an empty nor a stale
        // single-device operator list may replace this active group.
        FastllmMultiCudaSetDevice(group == 0 ? std::vector<int>{} : std::vector<int>{0});
        const std::vector<int> devices =
            group == 0 ? std::vector<int>{0, 1} : std::vector<int>{1, 0};
        if (!FastllmInitNccl(devices))
            return 2;
        ok &= RunBits<uint16_t, fastllm::FLOAT16>(devices);
        ok &= RunBits<uint16_t, fastllm::BFLOAT16>(devices);
        ok &= RunBits<float, fastllm::FLOAT32>(devices);
        ok &= RunBits<int8_t, fastllm::INT8>(devices);
        ok &= RunBits<int32_t, fastllm::INT32>(devices);
        for (int count : {1, 513, 5120, 112640}) {
            ok &= Run<half>(devices, fastllm::FLOAT16, count);
            ok &= Run<__nv_bfloat16>(devices, fastllm::BFLOAT16, count);
            ok &= Run<float>(devices, fastllm::FLOAT32, count);
            ok &= Run<int8_t>(devices, fastllm::INT8, count);
            ok &= Run<int32_t>(devices, fastllm::INT32, count);
        }
    }
    std::cout << "host collective initialized-group regression: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
