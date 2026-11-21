// Exercise production registration and startup self-test with delayed H2D
// completion and old pool writers on different per-thread streams.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

namespace {
void *staging[2] = {};
bool delayMetadata = false;
bool delaySelfTestInput = false;
int delayedSelfTestCopies = 0;
constexpr size_t kSelfTestBytes = 1024 * 1024;

void DelayCopy(void *) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

cudaError_t DelayedHostCopy(void *dest, const void *source, size_t bytes,
                           cudaMemcpyKind kind) {
    const bool metadata = delayMetadata && bytes == 8 * sizeof(void *);
    const bool selfTestInput = delaySelfTestInput && bytes == kSelfTestBytes;
    if ((!metadata && !selfTestInput) || kind != cudaMemcpyHostToDevice) {
        return cudaMemcpy(dest, source, bytes, kind);
    }
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess || device < 0 || device >= 2) return cudaErrorInvalidDevice;
    if (selfTestInput && device != 1) return cudaMemcpy(dest, source, bytes, kind);
    if (selfTestInput) ++delayedSelfTestCopies;
    // Start with a known incomplete upload; keep the staged source alive until
    // every queued DMA finishes. No CUDA API is called by the host callback.
    status = cudaMemsetAsync(dest, 0, bytes, cudaStreamPerThread);
    if (status != cudaSuccess) return status;
    status = cudaStreamSynchronize(cudaStreamPerThread);
    if (status != cudaSuccess) return status;
    std::memcpy(staging[device], source, bytes);
    status = cudaLaunchHostFunc(cudaStreamPerThread, DelayCopy, nullptr);
    if (status != cudaSuccess) return status;
    return cudaMemcpyAsync(dest, staging[device], bytes, kind, cudaStreamPerThread);
}
}  // namespace

// Compile the real implementation, replacing only the timing of its pageable
// host-to-device copy. The shared library supplies the allocator/TP helpers.
#ifdef cudaMemcpy
#undef cudaMemcpy
#endif
#define cudaMemcpy DelayedHostCopy
#ifndef CUSTOM_AR_SOURCE
#define CUSTOM_AR_SOURCE "../../src/devices/multicuda/fastllm-custom-allreduce.cu"
#endif
#include CUSTOM_AR_SOURCE
#undef cudaMemcpy

static void Require(cudaError_t status, const char *where) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(status));
        std::exit(1);
    }
}

static int RunSelfTestUploadRegression() {
    CustomArState &state = GetCustomArState();
    for (int device = 0; device < 2; ++device) {
        Require(cudaSetDevice(device), "staging device");
        Require(cudaHostAlloc(&staging[device], kSelfTestBytes, 0), "input staging allocation");
    }
    bool passed = true;
    const int types[] = {(int)fastllm::DataType::FLOAT16,
                         (int)fastllm::DataType::BFLOAT16,
                         (int)fastllm::DataType::FLOAT32};
    const char *names[] = {"FP16", "BF16", "FP32"};
    for (int type = 0; type < 3; ++type) {
        CustomArBenchBuffers buffers;
        if (!AllocateCustomArBenchBuffers(state.devices, kSelfTestBytes, buffers)) return 1;
        for (int device = 0; device < 2; ++device) {
            Require(cudaSetDevice(device), "warmup device");
            Require(cudaMemset(buffers.inputs[device], 0, kSelfTestBytes), "warmup input");
            Require(cudaStreamSynchronize(cudaStreamPerThread), "warmup input completion");
        }
        const int count = (int)(kSelfTestBytes / CustomArTypeBytes(types[type]));
        float ignoredUs = 0.0f;
        if (!RunCustomArRankOperation(state.devices, 1, [&](int rank) {
                return RunCustomArCandidate(buffers.inputs[rank], buffers.outputs[rank],
                    count, types[type], state.devices[rank]);
            }, ignoredUs)) return 1;
        const size_t registrations = state.registrations.size();
        const int copiesBefore = delayedSelfTestCopies;
        delaySelfTestInput = true;
        bool checked = CheckCustomArCorrectness(state, buffers, kSelfTestBytes, types[type]);
        delaySelfTestInput = false;
        const bool cached = registrations == state.registrations.size();
        const bool delayed = delayedSelfTestCopies == copiesBefore + 1;
        bool casePassed = checked && cached && delayed;
        passed = passed && casePassed;
        std::printf("%s: %s startup self-test waits for input upload (cached=%d, delayed=%d)\n",
                    casePassed ? "PASS" : "FAIL", names[type], cached, delayed);
        // The negative control can finish its worker reads before the upload.
        for (int device = 0; device < 2; ++device) {
            Require(cudaSetDevice(device), "cleanup device");
            Require(cudaStreamSynchronize(cudaStreamPerThread), "pending input upload");
        }
        FreeCustomArBenchBuffers(state.devices, buffers);
    }
    FastllmCudaCustomAllReduceReset();
    for (int device = 0; device < 2; ++device) {
        Require(cudaSetDevice(device), "staging cleanup device");
        Require(cudaFreeHost(staging[device]), "input staging cleanup");
    }
    return passed ? 0 : 1;
}

// Vary peer addresses on every launch. Queue multiple launches before reading
// back to detect a descriptor accidentally overwritten by the following round.
static int RunTransientInputsRegression() {
    auto &state = GetCustomArState();
    const auto &devices = state.devices;
    constexpr int rounds = 32, width = 128;
    CustomArBenchBuffers buffers;
    const size_t bytes = rounds * width * sizeof(float);
    if (!AllocateCustomArBenchBuffers(devices, bytes, buffers)) return 1;
    std::vector<float> values(rounds * width);
    for (int rank : devices) {
        Require(cudaSetDevice(rank), "input device");
        for (int i = 0; i < rounds * width; ++i) values[i] = (float)(rank + i);
        Require(cudaMemcpyAsync(buffers.inputs[rank], values.data(), bytes, cudaMemcpyHostToDevice, cudaStreamPerThread), "input upload");
        Require(cudaStreamSynchronize(cudaStreamPerThread), "input readiness");
    }
    float us = 0;
    auto reduce = [&](int rank, int row) {
        return RunCustomArCandidate(static_cast<float *>(buffers.inputs[rank]) + row * width,
                                    static_cast<float *>(buffers.outputs[rank]) + row * width,
                                    width, fastllm::FLOAT32, rank, false);
    };
    if (!RunCustomArRankOperation(devices, 1, [&](int rank) { return reduce(rank, 0); }, us)) return 1;
    const size_t registrations = state.registrations.size();
    bool ok = RunCustomArRankOperation(devices, 1, [&](int rank) {
        {
            // One transient peer makes the whole tuple transient.
            fastllm::TransientAllReduceInputs transient(rank == 0);
            for (int i = 1; i < rounds; ++i)
                if (!reduce(rank, i)) return false;
        }
        return reduce(rank, 0);
    }, us);
    for (int rank : devices) {
        Require(cudaSetDevice(rank), "result device");
        Require(cudaMemcpyAsync(values.data(), buffers.outputs[rank], bytes, cudaMemcpyDeviceToHost, cudaStreamPerThread), "result download");
        Require(cudaStreamSynchronize(cudaStreamPerThread), "result readiness");
        for (int i = 0; i < rounds * width; ++i) ok &= values[i] == (float)(2 * i + 1);
    }
    ok &= state.registrations.size() == registrations;
    FreeCustomArBenchBuffers(devices, buffers);
    std::printf("%s: rotating peer inputs preserve results and registration count\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    const bool reuse = argc > 1 && std::strcmp(argv[1], "reuse") == 0;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 77;
    const bool selfTest = argc > 1 && std::strcmp(argv[1], "selftest") == 0;
    const bool transient = argc > 1 && std::strcmp(argv[1], "transient") == 0;
    if (selfTest || transient) {
        for (int device = 0; device < 2; ++device) {
            int peer = 0;
            Require(cudaDeviceCanAccessPeer(&peer, device, 1 - device), "peer access query");
            if (!peer) return 77;
        }
        // Exercise the custom kernels even when the parent process disabled them.
        setenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "1", 1);
        if (!FastllmInitNccl({0, 1}) || !FastllmCudaCustomAllReduceEnabled()) return 1;
        return selfTest ? RunSelfTestUploadRegression() : RunTransientInputsRegression();
    }
    CustomArState state;
    state.devices = {0, 1};
    std::vector<void *> inputs(2);
    cudaStream_t consumer[2] = {};
    CustomArRankData *observed[2] = {};
    for (int device = 0; device < 2; ++device) {
        Require(cudaSetDevice(device), "set device");
        Require(cudaMalloc(&inputs[device], 16), "input allocation");
        Require(cudaHostAlloc(&staging[device], sizeof(CustomArRankData), 0), "staging allocation");
        Require(cudaHostAlloc(&observed[device], sizeof(CustomArRankData), 0), "result allocation");
        Require(cudaStreamCreateWithFlags(&consumer[device], cudaStreamNonBlocking), "consumer stream");
    }
    if (reuse) {
        // A peer worker may return a temporary to the pool before its stream
        // finishes writing it. Registration must drain that device before
        // borrowing this storage on the registering thread's own stream.
        for (int device = 0; device < 2; ++device) {
            Require(cudaSetDevice(device), "old writer device");
            void *temporary = FastllmCudaMalloc(sizeof(CustomArRankData));
            if (temporary == nullptr) return 1;
            Require(cudaLaunchHostFunc(consumer[device], DelayCopy, nullptr), "delay old writer");
            Require(cudaMemsetAsync(temporary, 0x5a, sizeof(CustomArRankData),
                                    consumer[device]), "old temporary write");
            FastllmCudaFree(temporary);
        }
    }
    delayMetadata = !reuse;
    std::vector<CustomArRankData *> tables;
    const bool registered = BuildCustomArRegistration(state, inputs, tables);
    delayMetadata = false;
    if (!registered) return 1;
    bool passed = true;
    if (reuse) {
        for (int device = 0; device < 2; ++device) {
            Require(cudaSetDevice(device), "old writer wait device");
            Require(cudaStreamSynchronize(consumer[device]), "old writer completion");
        }
    }
    // Read the last registered GPU first, before any cleanup can accidentally
    // synchronize the registration stream on that device.
    for (int device = 1; device >= 0; --device) {
        Require(cudaSetDevice(device), "consumer device");
        Require(cudaMemcpyAsync(observed[device], tables[device], sizeof(CustomArRankData),
                                cudaMemcpyDeviceToHost, consumer[device]), "consume metadata");
        Require(cudaStreamSynchronize(consumer[device]), "consumer completion");
        for (int peer = 0; peer < 2; ++peer) {
            if (observed[device]->ptrs[peer] != inputs[peer]) {
                std::fprintf(stderr, "GPU %d peer %d: observed %p, expected %p\n",
                             device, peer, observed[device]->ptrs[peer], inputs[peer]);
                passed = false;
            }
        }
    }
    for (int device = 0; device < 2; ++device) {
        Require(cudaSetDevice(device), "cleanup device");
        Require(cudaStreamSynchronize(cudaStreamPerThread), "pending DMA completion");
        FastllmCudaForceFree(tables[device]);
        Require(cudaFree(inputs[device]), "input cleanup");
        Require(cudaFreeHost(staging[device]), "staging cleanup");
        Require(cudaFreeHost(observed[device]), "result cleanup");
        Require(cudaStreamDestroy(consumer[device]), "stream cleanup");
    }
    std::printf("%s: custom all-reduce registration %s\n",
                passed ? "PASS" : "FAIL", reuse ? "waits for previous pool writers" : "publishes completed pointer tables");
    return passed ? 0 : 1;
}
