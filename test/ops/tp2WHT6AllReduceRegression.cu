#include "devices/multicuda/tp2wht6allreduce.cuh"
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>

#define CUDA_OK(call)                                                                                        \
    do {                                                                                                     \
        auto s = (call);                                                                                     \
        if (s != cudaSuccess) {                                                                              \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(s));                        \
            std::abort();                                                                                    \
        }                                                                                                    \
    } while (0)
#define NCCL_OK(call)                                                                                        \
    do {                                                                                                     \
        auto s = (call);                                                                                     \
        if (s != ncclSuccess) {                                                                              \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(s));                        \
            std::abort();                                                                                    \
        }                                                                                                    \
    } while (0)

__global__ void Fill(half *x, int n, int rank, int iteration) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        unsigned hash = (unsigned)i * 2654435761u + iteration * 2246822519u + rank * 3266489917u;
        hash ^= hash >> 13;
        float value = ((int)(hash % 65521u) - 32760) / 997.0f;
        if (rank < 2 && (hash & 255u) == 0) value *= 8;
        if (rank < 2 && iteration == 0) value = 0;
        if (rank == 2) value *= 4;
        x[i] = __float2half(value);
    }
}

__global__ void AddResidual(half *x, const half *residual, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = __hadd(x[i], residual[i]);
}

__global__ void ErrorMetrics(const half *value, const half *reference, int n, double *metrics) {
    float error = 0, energy = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float a = __half2float(value[i]), b = __half2float(reference[i]);
        error += (a - b) * (a - b);
        energy += b * b;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        error += __shfl_down_sync(0xffffffffu, error, offset);
        energy += __shfl_down_sync(0xffffffffu, energy, offset);
    }
    if (threadIdx.x % 32 == 0) {
        atomicAdd(metrics, (double)error);
        atomicAdd(metrics + 1, (double)energy);
    }
}

__global__ void Check(const half *a, const half *b, int n, unsigned *errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (__half_as_ushort(a[i]) != __half_as_ushort(b[i])) atomicAdd(errors, 1u);
    }
}

// Independent dense H-matrix CPU reference checks the wire format and inverse,
// including padding. It does not share the CUDA butterfly implementation.
static void CheckWHT6Codec() {
    constexpr int N = 65, Groups = (N + 31) / 32;
    half input[2][N], residual[N], result[N];
    fastllm::TP2HostWHT6Block expected[2][Groups] = {}, actual[Groups];
    float reconstructed[2][Groups][32] = {};
    half *gpuInput, *gpuResidual, *gpuOutput;
    fastllm::TP2HostWHT6Block *gpuPacked[2];
    CUDA_OK(cudaSetDevice(0));
    CUDA_OK(cudaMalloc(&gpuInput, sizeof(input[0])));
    CUDA_OK(cudaMalloc(&gpuResidual, sizeof(residual)));
    CUDA_OK(cudaMalloc(&gpuOutput, sizeof(result)));
    for (int i = 0; i < N; ++i) residual[i] = __float2half((i % 7 - 3) * .5f);
    for (int rank = 0; rank < 2; ++rank) {
        CUDA_OK(cudaMalloc(&gpuPacked[rank], sizeof(actual)));
        for (int i = 0; i < N; ++i) input[rank][i] = __float2half(((i * 17 + rank * 23) % 61 - 30) * .25f);
        for (int group = 0; group < Groups; ++group) {
            float rotated[32], maximum = 0;
            for (int row = 0; row < 32; ++row) {
                float sum = 0;
                for (int col = 0; col < 32; ++col) {
                    int index = group * 32 + col;
                    float value = index < N ? __half2float(input[rank][index]) : 0;
                    int sign = ((0xb5ad4ec9u >> col) & 1u) ? -1 : 1;
                    int matrix = (__builtin_popcount((unsigned)(row & col)) & 1) ? -1 : 1;
                    sum += value * (sign * matrix);
                }
                rotated[row] = sum * .1767766952966369f;
                maximum = std::max(maximum, std::fabs(rotated[row]));
            }
            auto &packet = expected[rank][group];
            packet.scale = __float2half(maximum == 0 ? 0.0f : std::max(maximum / 31.0f, 0x1p-24f));
            float scale = __half2float(packet.scale);
            for (int row = 0; row < 32; ++row) {
                int q = scale == 0 ? 0 : (int)std::nearbyint(rotated[row] / scale);
                q = std::max(-31, std::min(31, q));
                reconstructed[rank][group][row] = q * scale;
                // Generic bit writer: independent of the CUDA quartet packing.
                for (int bit = 0; bit < 6; ++bit) {
                    int position = row * 6 + bit;
                    packet.values[position / 8] |= (((unsigned)q >> bit) & 1u) << (position % 8);
                }
            }
        }
        CUDA_OK(cudaMemcpy(gpuInput, input[rank], sizeof(input[rank]), cudaMemcpyHostToDevice));
        fastllm::TP2HostWHT6PackKernel<<<1, 128>>>(gpuInput, gpuPacked[rank], N);
        CUDA_OK(cudaMemcpy(actual, gpuPacked[rank], sizeof(actual), cudaMemcpyDeviceToHost));
        if (std::memcmp(actual, expected[rank], sizeof(actual))) {
            fprintf(stderr, "WHT6 wire packet differs from dense CPU reference\n");
            std::abort();
        }
    }
    CUDA_OK(cudaMemcpy(gpuResidual, residual, sizeof(residual), cudaMemcpyHostToDevice));
    fastllm::TP2HostWHT6SumKernel<<<1, 128>>>(gpuPacked[0], gpuPacked[1], gpuResidual, gpuOutput, N);
    CUDA_OK(cudaMemcpy(result, gpuOutput, sizeof(result), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        int row = i % 32, group = i / 32;
        float sum = 0;
        for (int col = 0; col < 32; ++col) {
            int matrix = (__builtin_popcount((unsigned)(row & col)) & 1) ? -1 : 1;
            sum += matrix * (reconstructed[0][group][col] + reconstructed[1][group][col]);
        }
        float sign = ((0xb5ad4ec9u >> row) & 1u) ? -1.0f : 1.0f;
        float expectedValue = sum * .1767766952966369f * sign + __half2float(residual[i]);
        float error = std::fabs(__half2float(result[i]) - expectedValue);
        if (!(error <= .002f * std::max(1.0f, std::fabs(expectedValue)))) {
            fprintf(stderr, "WHT6 inverse differs from dense CPU reference at %d\n", i);
            std::abort();
        }
    }
    CUDA_OK(cudaFree(gpuInput));
    CUDA_OK(cudaFree(gpuResidual));
    CUDA_OK(cudaFree(gpuOutput));
    for (int rank = 0; rank < 2; ++rank) CUDA_OK(cudaFree(gpuPacked[rank]));
    printf("Dense CPU WHT6 wire/inverse reference: PASS (65 values, padded tail)\n");
}

int main() {
    constexpr int Count = fastllm::TP2WHT6AllReduce::Capacity / sizeof(half);
    constexpr size_t Bytes = (size_t)Count * sizeof(half);
    int ndev = 0;
    CUDA_OK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) return 77;
    CheckWHT6Codec();
    const int devices[2] = {0, 1};
    ncclComm_t comm[2];
    NCCL_OK(ncclCommInitAll(comm, 2, devices));
    fastllm::TP2WHT6AllReduce host;
    CUDA_OK(host.Init(0, 1));
    fastllm::NcclSubmitRendezvous gate(2, std::chrono::seconds(30));
    unsigned errors[2] = {};
    double errorSums[2][2] = {};
    auto worker = [&](int rank) {
        CUDA_OK(cudaSetDevice(rank));
        cudaStream_t stream;
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        half *input, *output, *reference, *residual, *rank0Result;
        double *metrics;
        unsigned *deviceErrors;
        CUDA_OK(cudaMalloc(&input, Bytes));
        CUDA_OK(cudaMalloc(&output, Bytes));
        CUDA_OK(cudaMalloc(&reference, Bytes));
        CUDA_OK(cudaMalloc(&residual, Bytes));
        CUDA_OK(cudaMalloc(&rank0Result, Bytes));
        CUDA_OK(cudaMalloc(&metrics, 2 * sizeof(double)));
        CUDA_OK(cudaMemsetAsync(metrics, 0, 2 * sizeof(double), stream));
        CUDA_OK(cudaMalloc(&deviceErrors, sizeof(unsigned)));
        CUDA_OK(cudaMemsetAsync(deviceErrors, 0, sizeof(unsigned), stream));
        auto nccl = [&](half *x, int n) {
            if (!gate.Wait(rank, fastllm::NcclSubmitRendezvous::Before, n, 0)) std::abort();
            NCCL_OK(ncclAllReduce(x, x, n, ncclHalf, ncclSum, comm[rank], stream));
            if (!gate.Wait(rank, fastllm::NcclSubmitRendezvous::After, n, 0)) std::abort();
        };
        // No GPU synchronization between iterations: exercise scratch/event
        // reuse, in/out-of-place operation, odd tails, and skewed host ranks.
        for (int iteration = 0; iteration < 24; ++iteration) {
            const int count = Count - (iteration % 3 == 0);
            Fill<<<1024, 256, 0, stream>>>(input, Count, rank, iteration);
            Fill<<<1024, 256, 0, stream>>>(residual, Count, 2, iteration);
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaMemcpyAsync(reference, input, Bytes, cudaMemcpyDeviceToDevice, stream));
            if (rank == 0) AddResidual<<<1024, 256, 0, stream>>>(reference, residual, count);
            if (rank == iteration % 2 && iteration % 4 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            half *result = iteration % 2 ? output : residual;
            CUDA_OK(host.Run(rank, input, result, count, stream, residual));
            nccl(reference, count);
            ErrorMetrics<<<1024, 256, 0, stream>>>(result, reference, count, metrics);
            if (!gate.Wait(rank, fastllm::NcclSubmitRendezvous::Before, count, 0)) std::abort();
            NCCL_OK(ncclBroadcast(result, rank0Result, count, ncclHalf, 0, comm[rank], stream));
            if (!gate.Wait(rank, fastllm::NcclSubmitRendezvous::After, count, 0)) std::abort();
            Check<<<1024, 256, 0, stream>>>(result, rank0Result, count, deviceErrors);
            // With all-zero contributions, the full-precision residual must
            // survive byte-for-byte, including when output aliases residual.
            if (iteration == 0) Check<<<1024, 256, 0, stream>>>(result, reference, count, deviceErrors);
            CUDA_OK(cudaGetLastError());
        }
        CUDA_OK(
            cudaMemcpyAsync(&errors[rank], deviceErrors, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        CUDA_OK(cudaMemcpy(errorSums[rank], metrics, 2 * sizeof(double), cudaMemcpyDeviceToHost));

        CUDA_OK(cudaFree(input));
        CUDA_OK(cudaFree(output));
        CUDA_OK(cudaFree(reference));
        CUDA_OK(cudaFree(residual));
        CUDA_OK(cudaFree(rank0Result));
        CUDA_OK(cudaFree(metrics));
        CUDA_OK(cudaFree(deviceErrors));
        CUDA_OK(cudaStreamDestroy(stream));
    };
    std::thread first(worker, 0), second(worker, 1);
    first.join();
    second.join();
    double rmse0 = sqrt(errorSums[0][0] / errorSums[0][1]);
    double rmse1 = sqrt(errorSums[1][0] / errorSums[1][1]);
    printf("WHT6 payload=%zu wire=%zu iterations=24 rank_mismatches=[%u,%u] relative_RMSE=[%.8f,%.8f]\n",
           Bytes, (size_t)Count / 32 * sizeof(fastllm::TP2HostWHT6Block), errors[0], errors[1], rmse0, rmse1);
    for (int r = 0; r < 2; ++r) NCCL_OK(ncclCommDestroy(comm[r]));
    return errors[0] || errors[1] || !(rmse0 < .015) || !(rmse1 < .015) ? 1 : 0;
}
