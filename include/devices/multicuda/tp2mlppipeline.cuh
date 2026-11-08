#ifndef FASTLLM_TP2_MLP_PIPELINE_CUH
#define FASTLLM_TP2_MLP_PIPELINE_CUH

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>
#include "devices/multicuda/tp2wht6allreduce.cuh"

namespace fastllm {

// Native TP adds the replicated residual on rank 0 before its FP16 sum.
// Keep that rounding order when moving the reduction to another stream.
static __global__ void TP2MlpAddResidualKernel(half *partial, const half *residual, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count / 2) {
        reinterpret_cast<half2 *>(partial)[i] = __hadd2(
            reinterpret_cast<const half2 *>(partial)[i],
            reinterpret_cast<const half2 *>(residual)[i]);
    }
    if (i == count / 2 && (count & 1)) partial[count - 1] = __hadd(partial[count - 1], residual[count - 1]);
}

// Non-owning view of the existing communicator/transport. Both ranks choose
// the same backend before any work is submitted; errors never switch backend.
class TP2MlpReduction {
public:
    enum class Backend { Nccl, WHT6 };
    TP2MlpReduction(int rank, Backend backend, ncclComm_t comm,
                    NcclSubmitRendezvous *submit, TP2WHT6AllReduce *host, int ranks = 2)
        : rank(rank), ranks(ranks), backend(backend), comm(comm), submit(submit), host(host) {}

    const char *Name() const {
        switch (backend) {
            case Backend::WHT6: return "WHT6";
            default: return "NCCL FP16";
        }
    }

    void Abort(const char *reason) {
        if (submit) submit->Abort(reason);
        if (host) host->Abort(reason);
    }

    cudaError_t Run(half *partial, half *residual, int count, cudaStream_t stream) {
        if (rank < 0 || rank >= ranks || !partial || !residual || count <= 0 ||
            (backend == Backend::Nccl ? (!comm || !submit) : !host)) {
            Abort("invalid TP2 MLP reduction arguments");
            return cudaErrorInvalidValue;
        }
        if (backend == Backend::WHT6) {
            return host->Run(rank, partial, residual, count, stream, residual);
        }
        if (rank == 0) {
            TP2MlpAddResidualKernel<<<((count + 1) / 2 + 255) / 256, 256, 0, stream>>>(partial, residual, count);
            cudaError_t status = cudaGetLastError();
            if (status != cudaSuccess) { Abort(cudaGetErrorString(status)); return status; }
        }
        // Same pre/post CPU submission boundaries as ordinary NCCL. These do
        // not wait for GPU completion, so the next chunk can compute in parallel.
        if (!submit->Wait(rank, NcclSubmitRendezvous::Before, count, ncclHalf)) return cudaErrorUnknown;
        ncclResult_t status = ncclAllReduce(partial, residual, count, ncclHalf, ncclSum, comm, stream);
        if (status != ncclSuccess) {
            std::fprintf(stderr, "TP2 MLP ncclAllReduce failed: %s\n", ncclGetErrorString(status));
            Abort("TP2 MLP ncclAllReduce submission failed");
            return cudaErrorUnknown;
        }
        if (!submit->Wait(rank, NcclSubmitRendezvous::After, count, ncclHalf)) return cudaErrorUnknown;
        return cudaSuccess;
    }

    // A contribution already sitting in the residual buffer (rank 0 added its residual).
    cudaError_t RunInPlace(half *buffer, int count, cudaStream_t stream) {
        if (!comm || !submit) {
            Abort("invalid TP4 reduction arguments");
            return cudaErrorInvalidValue;
        }
        if (!submit->Wait(rank, NcclSubmitRendezvous::Before, count, ncclHalf)) return cudaErrorUnknown;
        ncclResult_t status = ncclAllReduce(buffer, buffer, count, ncclHalf, ncclSum, comm, stream);
        if (status != ncclSuccess) {
            std::fprintf(stderr, "TP4 ncclAllReduce failed: %s\n", ncclGetErrorString(status));
            Abort("TP4 ncclAllReduce submission failed");
            return cudaErrorUnknown;
        }
        if (!submit->Wait(rank, NcclSubmitRendezvous::After, count, ncclHalf)) return cudaErrorUnknown;
        return cudaSuccess;
    }

private:
    int rank;
    int ranks;
    Backend backend;
    ncclComm_t comm;
    NcclSubmitRendezvous *submit;
    TP2WHT6AllReduce *host;
};

// Match the eager FP16 SwiGLU arithmetic, including its FP16 rounding.
static __global__ void TP2MlpSwiGluKernel(const half *gateup, half *output,
                                       int count, int intermediate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int pos = (i / intermediate) * (2 * intermediate) + i % intermediate;
        half x = gateup[pos], y = gateup[pos + intermediate];
        output[i] = __hmul(__hdiv(x, __hadd(__float2half(1.0f), hexp(-x))), y);
    }
}

// One caller per GPU, matching TP2WHT6AllReduce. The caller dequantizes ONLY
// this layer's weights into GateUpWeight()/DownWeight() on the compute stream.
// They remain live through every token chunk; no shared dequant scratch aliases.
class TP2MlpPipeline {
public:
    static constexpr int TokenChunk = 512;
    TP2MlpPipeline() = default;
    TP2MlpPipeline(const TP2MlpPipeline &) = delete;
    TP2MlpPipeline &operator=(const TP2MlpPipeline &) = delete;
    ~TP2MlpPipeline() {
        if (device < 0) return;
        int previous = -1;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        if (communication) cudaStreamSynchronize(communication);
        for (cudaEvent_t event : attnDone) cudaEventDestroy(event);
        if (ready) cudaEventDestroy(ready);
        if (complete) cudaEventDestroy(complete);
        if (communication) cudaStreamDestroy(communication);
        if (storage) cudaFree(storage);
        if (previous >= 0) cudaSetDevice(previous);
    }

    cudaError_t Init(int tokens, int hidden, int intermediate, int tokenChunk = TokenChunk) {
        if (storage) return cudaErrorInvalidValue;
        if (tokens <= 0 || tokenChunk <= 0 || tokens % tokenChunk || hidden <= 0 || intermediate <= 0)
            return cudaErrorInvalidValue;
        n = tokens; h = hidden; k = intermediate; chunk = tokenChunk;
        cudaError_t status = cudaGetDevice(&device);
        if (status != cudaSuccess) return status;
        // 3*h*k weights, a one-chunk gate/up and SwiGLU workspace, and
        // disjoint per-token down outputs that communication can read safely.
        size_t elements = (size_t)3 * h * k + (size_t)chunk * 3 * k + (size_t)n * h;
        status = cudaMalloc(reinterpret_cast<void **>(&storage), elements * sizeof(half));
        if (status != cudaSuccess) return status;
        wgateup = storage;
        wdown = wgateup + (size_t)2 * h * k;
        gateup = wdown + (size_t)h * k;
        swiglu = gateup + (size_t)chunk * 2 * k;
        partial = swiglu + (size_t)chunk * k;
        int leastPriority = 0, greatestPriority = 0;
        status = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
        if (status != cudaSuccess) return status;
        status = cudaStreamCreateWithPriority(&communication, cudaStreamNonBlocking, greatestPriority);
        if (status != cudaSuccess) return status;
        status = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
        if (status != cudaSuccess) return status;
        for (int c = 0; c < n / chunk; ++c) {
            cudaEvent_t event = nullptr;
            status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            if (status != cudaSuccess) return status;
            attnDone.push_back(event);
        }
        return cudaEventCreateWithFlags(&complete, cudaEventDisableTiming);
    }

    half *GateUpWeight() const { return wgateup; }
    half *DownWeight() const { return wdown; }
    bool Matches(int tokens, int hidden, int intermediate) const {
        return storage && complete && n == tokens && h == hidden && k == intermediate;
    }

    cudaError_t Run(const half *input, half *residual, cublasHandle_t handle,
                    TP2MlpReduction &reduction, cudaStream_t compute) {
        return Run(input, residual, n, handle, reduction, compute);
    }

    int Capacity() const { return storage ? n : 0; }

    // The attention reduce leaves the compute stream before the weights are
    // dequantized, so it overlaps that work too.
    cudaError_t StartDeferred(half *hidden, int tokens, TP2MlpReduction &reduction, cudaStream_t compute) {
        cudaError_t status = cudaEventRecord(ready, compute);
        if (status == cudaSuccess) status = cudaStreamWaitEvent(communication, ready, 0);
        for (int c = 0; status == cudaSuccess && c < 2; ++c) {
            status = PostAttention(hidden, tokens, c, reduction);
        }
        if (status != cudaSuccess) reduction.Abort(cudaGetErrorString(status));
        return status;
    }

    // Each chunk: wait its attention reduce, norm, gate/up, SwiGLU, down, MLP reduce.
    cudaError_t RunDeferred(half *hidden, const half *input, int tokens, cublasHandle_t handle,
                            TP2MlpReduction &reduction, cudaStream_t compute,
                            const std::function<void(int, int)> &normChunk) {
#define TP4_MLP_CHECK(call) do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { reduction.Abort(cudaGetErrorString(status)); return status; } \
    } while (0)
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        auto gemm = [&](int out, int rows, int in, const half *weight,
                        const half *activation, half *output) -> cudaError_t {
            cublasStatus_t s = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out, rows, in, &alpha, weight, CUDA_R_16F, in,
                activation, CUDA_R_16F, in, &beta, output, CUDA_R_16F, out,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
            return s == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
        };
        if (cublasSetStream(handle, compute) != CUBLAS_STATUS_SUCCESS) {
            reduction.Abort("TP4 MLP cublasSetStream failed");
            return cudaErrorUnknown;
        }
        int index = 0;
        for (int offset = 0; offset < tokens; offset += chunk, ++index) {
            const int rows = std::min(chunk, tokens - offset);
            TP4_MLP_CHECK(cudaStreamWaitEvent(compute, attnDone[index], 0));
            normChunk(offset, rows);
            TP4_MLP_CHECK(cudaGetLastError());
            TP4_MLP_CHECK(gemm(2 * k, rows, h, wgateup, input + (size_t)offset * h, gateup));
            int count = rows * k;
            TP2MlpSwiGluKernel<<<(count + 255) / 256, 256, 0, compute>>>(gateup, swiglu, count, k);
            TP4_MLP_CHECK(cudaGetLastError());
            TP4_MLP_CHECK(gemm(h, rows, k, wdown, swiglu, partial + (size_t)offset * h));
            TP4_MLP_CHECK(cudaEventRecord(ready, compute));
            TP4_MLP_CHECK(cudaStreamWaitEvent(communication, ready, 0));
            TP4_MLP_CHECK(reduction.Run(partial + (size_t)offset * h,
                hidden + (size_t)offset * h, rows * h, communication));
            TP4_MLP_CHECK(PostAttention(hidden, tokens, index + 2, reduction));
        }
        TP4_MLP_CHECK(cudaEventRecord(complete, communication));
        TP4_MLP_CHECK(cudaStreamWaitEvent(compute, complete, 0));
#undef TP4_MLP_CHECK
        return cudaSuccess;
    }

    // Any row count up to Capacity(); the last chunk may be shorter.
    cudaError_t Run(const half *input, half *residual, int tokens, cublasHandle_t handle,
                    TP2MlpReduction &reduction, cudaStream_t compute) {
#define TP2_MLP_CHECK(call) do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { reduction.Abort(cudaGetErrorString(status)); return status; } \
    } while (0)
        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        auto gemm = [&](int out, int rows, int in, const half *weight,
                        const half *activation, half *output) -> cudaError_t {
            cublasStatus_t s = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                out, rows, in, &alpha, weight, CUDA_R_16F, in,
                activation, CUDA_R_16F, in, &beta, output, CUDA_R_16F, out,
                CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
            if (s != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "TP2 MLP cuBLAS failed: %d\n", static_cast<int>(s));
                return cudaErrorUnknown;
            }
            return cudaSuccess;
        };
        cublasStatus_t s = cublasSetStream(handle, compute);
        if (s != CUBLAS_STATUS_SUCCESS) {
            reduction.Abort("TP2 MLP cublasSetStream failed");
            return cudaErrorUnknown;
        }
        for (int offset = 0; offset < tokens; offset += chunk) {
            const int rows = std::min(chunk, tokens - offset);
            TP2_MLP_CHECK(gemm(2 * k, rows, h, wgateup,
                               input + (size_t)offset * h, gateup));
            int count = rows * k;
            TP2MlpSwiGluKernel<<<(count + 255) / 256, 256, 0, compute>>>(
                gateup, swiglu, count, k);
            TP2_MLP_CHECK(cudaGetLastError());
            TP2_MLP_CHECK(gemm(h, rows, k, wdown, swiglu,
                               partial + (size_t)offset * h));
            // Event reuse is safe: each wait captures the record submitted
            // immediately before it; neither peer host thread uses this event.
            TP2_MLP_CHECK(cudaEventRecord(ready, compute));
            TP2_MLP_CHECK(cudaStreamWaitEvent(communication, ready, 0));
            // All reduction work stays on the communication stream. Native
            // NCCL and WHT6 retain their own submission boundaries.
            TP2_MLP_CHECK(reduction.Run(partial + (size_t)offset * h,
                residual + (size_t)offset * h, rows * h, communication));
        }
        TP2_MLP_CHECK(cudaEventRecord(complete, communication));
        TP2_MLP_CHECK(cudaStreamWaitEvent(compute, complete, 0));
#undef TP2_MLP_CHECK
        return cudaSuccess;
    }

private:
    cudaError_t PostAttention(half *hidden, int tokens, int index, TP2MlpReduction &reduction) {
        const int offset = index * chunk;
        if (offset >= tokens) return cudaSuccess;
        cudaError_t status = reduction.RunInPlace(hidden + (size_t)offset * h,
            std::min(chunk, tokens - offset) * h, communication);
        return status == cudaSuccess ? cudaEventRecord(attnDone[index], communication) : status;
    }

    int device = -1, n = 0, h = 0, k = 0, chunk = TokenChunk;
    std::vector<cudaEvent_t> attnDone;
    half *storage = nullptr, *wgateup = nullptr, *wdown = nullptr;
    half *gateup = nullptr, *swiglu = nullptr, *partial = nullptr;
    cudaStream_t communication = nullptr;
    cudaEvent_t ready = nullptr, complete = nullptr;
};
} // namespace fastllm
#endif
