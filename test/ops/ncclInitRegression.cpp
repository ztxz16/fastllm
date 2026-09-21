#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <signal.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Interpose only in this executable. All non-injected calls still use the
// actual CUDA/NCCL libraries and FastllmInitNccl implementation.
static std::string scenario;
static std::atomic<bool> commReady{false};
static std::atomic<bool> submitted{false};
static std::atomic<int> selfTests{0};

template <class T> static T Real(const char *name) {
    void *symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) {
        std::fprintf(stderr, "Missing real symbol: %s\n", name);
        std::_Exit(90);
    }
    return reinterpret_cast<T>(symbol);
}

[[noreturn]] static void Hang() {
    for (;;) pause();
}

extern "C" ncclResult_t ncclCommInitAll(ncclComm_t *comms, int n, const int *devices) {
    if (scenario == "init_hang") Hang();
    if (scenario == "init_error") return ncclSystemError;
    auto status = Real<decltype(&ncclCommInitAll)>("ncclCommInitAll")(comms, n, devices);
    commReady = status == ncclSuccess;
    return status;
}

extern "C" ncclResult_t ncclGroupStart() {
    if (commReady) {
        ++selfTests;
        if (scenario == "group_start") return ncclSystemError;
    }
    return Real<decltype(&ncclGroupStart)>("ncclGroupStart")();
}

extern "C" ncclResult_t ncclGroupEnd() {
    if (commReady && scenario == "group_hang") Hang();
    auto status = Real<decltype(&ncclGroupEnd)>("ncclGroupEnd")();
    if (commReady) submitted = true;
    return commReady && scenario == "group_end" ? ncclSystemError : status;
}

extern "C" ncclResult_t ncclAllReduce(const void *send, void *recv, size_t count,
        ncclDataType_t type, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    if (commReady && (scenario == "launch_error" || scenario == "abort_hang")) {
        return ncclSystemError;
    }
    if (commReady && scenario == "partial_launch_error") {
        int device = -1;
        cudaGetDevice(&device);
        if (device == 1) return ncclSystemError;
    }
    return Real<decltype(&ncclAllReduce)>("ncclAllReduce")(
        send, recv, count, type, op, comm, stream);
}

extern "C" ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *error) {
    if (submitted && scenario == "async_error") {
        *error = ncclSystemError;
        return ncclSuccess;
    }
    return Real<decltype(&ncclCommGetAsyncError)>("ncclCommGetAsyncError")(comm, error);
}

extern "C" ncclResult_t ncclCommAbort(ncclComm_t comm) {
    std::fprintf(stderr, "TEST: abort communicator\n");
    if (scenario == "abort_hang") Hang();
    return Real<decltype(&ncclCommAbort)>("ncclCommAbort")(comm);
}

extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
    if (commReady && scenario == "stream_hang") return cudaErrorNotReady;
    return Real<decltype(&cudaStreamQuery)>("cudaStreamQuery")(stream);
}

extern "C" cudaError_t cudaMalloc(void **ptr, size_t bytes) {
    if (commReady && scenario == "allocation_error" && bytes == 1024 * sizeof(float)) {
        return cudaErrorMemoryAllocation;
    }
    using Allocate = cudaError_t (*)(void **, size_t);
    return Real<Allocate>("cudaMalloc")(ptr, bytes);
}

// Production CUDA translation units use --default-stream=per-thread; keep
// both symbol variants covered without changing their stream semantics.
extern "C" cudaError_t cudaStreamQuery_ptsz(cudaStream_t stream) {
    if (commReady && scenario == "stream_hang") return cudaErrorNotReady;
    return Real<decltype(&cudaStreamQuery)>("cudaStreamQuery_ptsz")(stream);
}

static cudaError_t Copy(const char *symbol, void *dest, const void *source,
                        size_t bytes, cudaMemcpyKind kind) {
    auto status = Real<decltype(&cudaMemcpy)>(symbol)(dest, source, bytes, kind);
    if (status == cudaSuccess && commReady && kind == cudaMemcpyDeviceToHost &&
        bytes == 1024 * sizeof(float)) {
        if (scenario == "nan") {
            static_cast<float *>(dest)[0] = std::numeric_limits<float>::quiet_NaN();
        } else if (scenario == "inf") {
            static_cast<float *>(dest)[0] = std::numeric_limits<float>::infinity();
        } else if (scenario == "wrong_value") {
            static_cast<float *>(dest)[0] = -1.0f;
        }
    }
    return status;
}

extern "C" cudaError_t cudaMemcpy(void *dest, const void *source, size_t bytes,
                                   cudaMemcpyKind kind) {
    return Copy("cudaMemcpy", dest, source, bytes, kind);
}

extern "C" cudaError_t cudaMemcpy_ptds(void *dest, const void *source, size_t bytes,
                                        cudaMemcpyKind kind) {
    return Copy("cudaMemcpy_ptds", dest, source, bytes, kind);
}

static int Child(const char *mode) {
    scenario = mode;
    setenv("NCCL_DEBUG", "WARN", 1);
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 77;
    setenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "0", 1);
    setenv("FASTLLM_NCCL_INIT_TIMEOUT_MS", "3000", 1);
    if (!FastllmInitNccl({0, 1})) return 91;
    if (scenario != "normal") {
        std::fprintf(stderr, "FAIL: injected fault escaped initialization\n");
        return 92;
    }
    auto generation = FastllmGetNcclGeneration();
    if (!FastllmInitNccl({0, 1}) || generation != FastllmGetNcclGeneration() ||
        selfTests != 1) return 93;
    // A successful/reused group must cancel its watchdog, not kill inference
    // once the original initialization deadline elapses.
    std::this_thread::sleep_for(std::chrono::milliseconds(3100));
    std::puts("PASS: initialization and ready-group reuse");
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) return Child(argv[2]);
    if (argc != 2) return 94;
    const std::string mode = argv[1];
    const bool normal = mode == "normal";
    const char *expected = nullptr;
    if (normal) expected = "PASS: initialization";
    else if (mode == "nan" || mode == "inf" || mode == "wrong_value") expected = "self-test mismatch";
    else if (mode == "launch_error" || mode == "partial_launch_error") expected = "all-reduce launch";
    else if (mode == "allocation_error") expected = "self-test allocate";
    else if (mode == "group_start") expected = "group start";
    else if (mode == "group_end") expected = "group end";
    else if (mode == "async_error") expected = "async error";
    else if (mode == "init_error") expected = "communicator initialization failed";
    else if (mode == "init_hang" || mode == "group_hang" ||
             mode == "stream_hang" || mode == "abort_hang") expected = "timed out";
    else return 94;

    int input[2], output[2];
    if (pipe(input) || pipe(output)) return 95;
    pid_t pid = fork();
    if (pid == -1) return 95;
    if (pid == 0) {
        dup2(input[0], STDIN_FILENO);
        dup2(output[1], STDOUT_FILENO);
        dup2(output[1], STDERR_FILENO);
        close(input[0]); close(input[1]); close(output[0]); close(output[1]);
        execl("/proc/self/exe", argv[0], "--child", mode.c_str(), (char *)nullptr);
        std::_Exit(96);
    }
    close(input[0]); close(output[1]);
    // Keep stdin open without supplying input: an accidental getchar() in a
    // fatal path must be caught, along with zero exits and unbounded waits.
    int status = 0;
    bool timeout = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    while (waitpid(pid, &status, WNOHANG) == 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            timeout = true;
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    close(input[1]);
    std::string log;
    char buffer[4096];
    ssize_t length;
    while ((length = read(output[0], buffer, sizeof(buffer))) > 0) log.append(buffer, length);
    close(output[0]);
    std::fputs(log.c_str(), stdout);
    if (!timeout && WIFEXITED(status) && WEXITSTATUS(status) == 77) return 77;
    const int expectedExit = normal ? 0 : EXIT_FAILURE;
    bool passed = !timeout && WIFEXITED(status) && WEXITSTATUS(status) == expectedExit &&
                  log.find(expected) != std::string::npos;
    if (!normal && mode != "init_error" && mode != "partial_launch_error" &&
        mode.find("hang") == std::string::npos) {
        auto first = log.find("TEST: abort communicator");
        passed = passed && first != std::string::npos &&
                 log.find("TEST: abort communicator", first + 1) != std::string::npos;
    }
    std::printf("%s: %s (wait status=%d, timeout=%d)\n",
                passed ? "PASS" : "FAIL", mode.c_str(), status, timeout);
    return passed ? 0 : 1;
}
