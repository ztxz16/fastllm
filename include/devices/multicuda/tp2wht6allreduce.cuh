#ifndef FASTLLM_TP2_WHT6_ALL_REDUCE_CUH
#define FASTLLM_TP2_WHT6_ALL_REDUCE_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "devices/multicuda/ncclsubmitrendezvous.h"

namespace fastllm {

struct TP2HostWHT6Block {
    half scale;
    uint8_t values[24]; // Four signed 6-bit coefficients per three bytes.
};
static_assert(sizeof(TP2HostWHT6Block) == 26, "WHT6 wire layout must stay packed");

// Fixed shared Rademacher signs require no per-packet metadata. Both ranks
// use R = H * D; the inverse is D * H, with H normalized in FP32.
static __device__ __forceinline__ float TP2HostWHTSign(int lane) {
    return ((0xb5ad4ec9u >> lane) & 1u) ? -1.0f : 1.0f;
}

static __device__ __forceinline__ float TP2HostWHT32(float value, int lane) {
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        float peer = __shfl_xor_sync(0xffffffffu, value, delta);
        value = (lane & delta) ? peer - value : value + peer;
    }
    return value * 0.1767766952966369f;
}

static __global__ void TP2HostWHT6PackKernel(const half *input,
        TP2HostWHT6Block *output, int count) {
    int lane = threadIdx.x % 32;
    int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int i = group * 32 + lane;
    if (group >= (count + 31) / 32) return;
    float value = i < count ? __half2float(input[i]) : 0.0f;
    value = TP2HostWHT32(value * TP2HostWHTSign(lane), lane);
    float maximum = fabsf(value);
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
    half scale = __float2half_rn(maximum == 0 ? 0.0f : fmaxf(maximum / 31.0f, 0x1p-24f));
    float d = __half2float(scale);
    int q = d == 0 ? 0 : __float2int_rn(value / d);
    unsigned bits = (unsigned)max(-31, min(31, q)) & 63u;
    unsigned b1 = __shfl_xor_sync(0xffffffffu, bits, 1);
    unsigned b2 = __shfl_xor_sync(0xffffffffu, bits, 2);
    unsigned b3 = __shfl_xor_sync(0xffffffffu, bits, 3);
    if (lane % 4 == 0) {
        unsigned word = bits | (b1 << 6) | (b2 << 12) | (b3 << 18);
        int offset = (lane / 4) * 3;
        output[group].values[offset] = word & 255u;
        output[group].values[offset + 1] = (word >> 8) & 255u;
        output[group].values[offset + 2] = (word >> 16) & 255u;
    }
    if (lane == 0) output[group].scale = scale;
}

static __device__ __forceinline__ float TP2HostWHT6Unpack(
        const TP2HostWHT6Block &block, int lane) {
    int offset = (lane / 4) * 3;
    unsigned word = (unsigned)block.values[offset] |
        ((unsigned)block.values[offset + 1] << 8) |
        ((unsigned)block.values[offset + 2] << 16);
    int q = (word >> ((lane % 4) * 6)) & 63u;
    if (q >= 32) q -= 64;
    return __half2float(block.scale) * (float)q;
}

static __global__ void TP2HostWHT6SumKernel(const TP2HostWHT6Block *rank0,
        const TP2HostWHT6Block *rank1, const half *residual, half *output, int count) {
    int lane = threadIdx.x % 32;
    int first = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int stride = gridDim.x * blockDim.x / 32;
    // Even the final padded block needs all 32 lanes for the inverse WHT.
    for (int group = first; group < (count + 31) / 32; group += stride) {
        float sum = TP2HostWHT6Unpack(rank0[group], lane) +
                    TP2HostWHT6Unpack(rank1[group], lane);
        sum = TP2HostWHT32(sum, lane) * TP2HostWHTSign(lane);
        int i = group * 32 + lane;
        if (i < count) {
            if (residual) sum += __half2float(residual[i]);
            output[i] = __float2half_rn(sum);
        }
    }
}

// Two host callers, one per GPU. Init/reset require an idle communicator.
// Large eager FP16 collectives only; the caller must exclude graph capture.
// Both ranks must call Run with the same count, in the same order.
class TP2WHT6AllReduce {
public:
    static constexpr size_t Capacity = 32ULL * 1024 * 1024;
    static constexpr size_t ChunkBytes = 2ULL * 1024 * 1024;
    static constexpr size_t WireCapacity = Capacity / (32 * sizeof(half)) * sizeof(TP2HostWHT6Block);
    static constexpr int MaxChunks = (WireCapacity + ChunkBytes - 1) / ChunkBytes;

    TP2WHT6AllReduce() : rendezvous(2, std::chrono::seconds(30)) {}
    TP2WHT6AllReduce(const TP2WHT6AllReduce &) = delete;
    TP2WHT6AllReduce &operator=(const TP2WHT6AllReduce &) = delete;
    ~TP2WHT6AllReduce() { Reset(); }

    cudaError_t Init(int first, int second) {
        if (devices[0] >= 0 || first < 0 || second < 0 || first == second) return cudaErrorInvalidValue;
        int original = -1;
        cudaGetDevice(&original);
        devices[0] = first;
        devices[1] = second;
        cudaError_t result = cudaSuccess;
        for (int r = 0; r < 2 && result == cudaSuccess; ++r) {
            auto check = [&](cudaError_t s) { result = s; return s == cudaSuccess; };
            if (!check(cudaSetDevice(devices[r])) ||
                !check(cudaHostAlloc((void **)&host[r], WireCapacity, cudaHostAllocPortable)) ||
                !check(cudaMalloc((void **)&scratch[r], WireCapacity)) ||
                !check(cudaStreamCreateWithFlags(&send[r], cudaStreamNonBlocking)) ||
                !check(cudaStreamCreateWithFlags(&recv[r], cudaStreamNonBlocking)) ||
                !check(cudaEventCreateWithFlags(&ready[r], cudaEventDisableTiming)) ||
                !check(cudaEventCreateWithFlags(&received[r], cudaEventDisableTiming)) ||
                !check(cudaEventCreateWithFlags(&finished[r], cudaEventDisableTiming))) {
                break;
            }
            if (!check(cudaMalloc((void **)&packed[r], WireCapacity))) break;
            for (int c = 0; c < MaxChunks; ++c) {
                if (!check(cudaEventCreateWithFlags(&sent[r][c], cudaEventDisableTiming))) break;
            }
        }
        if (original >= 0) cudaSetDevice(original);
        initialized = result == cudaSuccess;
        return result;
    }

    cudaError_t Run(int rank, const void *input, void *output, int count,
                    cudaStream_t compute, const void *residual = nullptr) {
        if (!initialized || rank < 0 || rank > 1 || !input || !output ||
            count <= 0 || (size_t)count * sizeof(half) > Capacity) {
            rendezvous.Abort("invalid TP2 host AllReduce arguments");
            return cudaErrorInvalidValue;
        }
        const int peer = 1 - rank;
        const size_t bytes = ((size_t)count + 31) / 32 * sizeof(TP2HostWHT6Block);
        const int chunks = (bytes + ChunkBytes - 1) / ChunkBytes;
        const char *src = packed[rank];
        // Abort the CPU rendezvous on any submission error. Never fall back
        // to NCCL after one rank has already entered this collective.
#define FASTLLM_HOST_AR_CUDA(call) do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            rendezvous.Abort(cudaGetErrorString(status)); \
            return status; \
        } \
    } while (0)
        TP2HostWHT6PackKernel<<<(count + 255) / 256, 256, 0, compute>>>(
            static_cast<const half *>(input), reinterpret_cast<TP2HostWHT6Block *>(packed[rank]), count);
        FASTLLM_HOST_AR_CUDA(cudaGetLastError());
        FASTLLM_HOST_AR_CUDA(cudaEventRecord(ready[rank], compute));
        FASTLLM_HOST_AR_CUDA(cudaStreamWaitEvent(send[rank], ready[rank], 0));
        if (used[rank]) {
            // Peer H2D from our pinned buffer must finish before reuse.
            FASTLLM_HOST_AR_CUDA(cudaStreamWaitEvent(send[rank], received[peer], 0));
        }
        for (int c = 0; c < chunks; ++c) {
            const size_t offset = c * ChunkBytes;
            const size_t size = std::min(ChunkBytes, bytes - offset);
            FASTLLM_HOST_AR_CUDA(cudaMemcpyAsync(host[rank] + offset, src + offset,
                size, cudaMemcpyDeviceToHost, send[rank]));
            FASTLLM_HOST_AR_CUDA(cudaEventRecord(sent[rank][c], send[rank]));
        }
        // cudaStreamWaitEvent captures the latest *recorded* event. Meet on
        // the CPU so neither rank accidentally waits on a previous iteration.
        const int mode = residual != nullptr;
        if (!rendezvous.Wait(rank, NcclSubmitRendezvous::Before, count, mode)) {
            return cudaErrorUnknown;
        }
        if (used[rank]) {
            // The previous sum kernel may still be reading our GPU scratch.
            FASTLLM_HOST_AR_CUDA(cudaStreamWaitEvent(recv[rank], finished[rank], 0));
        }
        for (int c = 0; c < chunks; ++c) {
            const size_t offset = c * ChunkBytes;
            const size_t size = std::min(ChunkBytes, bytes - offset);
            FASTLLM_HOST_AR_CUDA(cudaStreamWaitEvent(recv[rank], sent[peer][c], 0));
            FASTLLM_HOST_AR_CUDA(cudaMemcpyAsync(scratch[rank] + offset, host[peer] + offset,
                size, cudaMemcpyHostToDevice, recv[rank]));
        }
        FASTLLM_HOST_AR_CUDA(cudaEventRecord(received[rank], recv[rank]));
        FASTLLM_HOST_AR_CUDA(cudaStreamWaitEvent(compute, received[rank], 0));
        // In-place summation must also wait for OUR outgoing D2H, which runs
        // on a different stream and can lag behind the peer's transfers.
        FASTLLM_HOST_AR_CUDA(cudaStreamWaitEvent(compute, sent[rank][chunks - 1], 0));
        const int blocks = std::min(1024, (count + 255) / 256);
        const auto *local = reinterpret_cast<const TP2HostWHT6Block *>(packed[rank]);
        const auto *remote = reinterpret_cast<const TP2HostWHT6Block *>(scratch[rank]);
        TP2HostWHT6SumKernel<<<blocks, 256, 0, compute>>>(rank == 0 ? local : remote,
            rank == 0 ? remote : local, static_cast<const half *>(residual),
            static_cast<half *>(output), count);
        FASTLLM_HOST_AR_CUDA(cudaGetLastError());
        FASTLLM_HOST_AR_CUDA(cudaEventRecord(finished[rank], compute));
        used[rank] = true;
        // Also keeps the next rank's allocator/GEMM from racing peer CUDA
        // submission, matching the existing NCCL submission boundary.
        if (!rendezvous.Wait(rank, NcclSubmitRendezvous::After, count, mode)) {
            return cudaErrorUnknown;
        }
#undef FASTLLM_HOST_AR_CUDA
        return cudaSuccess;
    }

    std::string Error() { return rendezvous.Error(); }
    void Abort(const char *reason) { rendezvous.Abort(reason); }

private:
    void Reset() {
        int original = -1;
        cudaGetDevice(&original);
        for (int r = 0; r < 2; ++r) {
            if (devices[r] < 0) continue;
            cudaSetDevice(devices[r]);
            if (used[r]) cudaEventSynchronize(finished[r]);
            if (send[r]) cudaStreamSynchronize(send[r]);
            if (recv[r]) cudaStreamSynchronize(recv[r]);
        }
        for (int r = 0; r < 2; ++r) {
            if (devices[r] < 0) continue;
            cudaSetDevice(devices[r]);
            for (int c = 0; c < MaxChunks; ++c) if (sent[r][c]) cudaEventDestroy(sent[r][c]);
            if (ready[r]) cudaEventDestroy(ready[r]);
            if (received[r]) cudaEventDestroy(received[r]);
            if (finished[r]) cudaEventDestroy(finished[r]);
            if (send[r]) cudaStreamDestroy(send[r]);
            if (recv[r]) cudaStreamDestroy(recv[r]);
            if (scratch[r]) cudaFree(scratch[r]);
            if (packed[r]) cudaFree(packed[r]);
            if (host[r]) cudaFreeHost(host[r]);
        }
        if (original >= 0) cudaSetDevice(original);
    }

    int devices[2] = {-1, -1};
    char *host[2] = {}, *scratch[2] = {};
    char *packed[2] = {};
    cudaStream_t send[2] = {}, recv[2] = {};
    cudaEvent_t ready[2] = {}, received[2] = {}, finished[2] = {};
    cudaEvent_t sent[2][MaxChunks] = {};
    bool initialized = false;
    bool used[2] = {};
    NcclSubmitRendezvous rendezvous;
};

} // namespace fastllm
#endif
