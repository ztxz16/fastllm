#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

static std::mutex observedMutex;
static std::map<unsigned long long, std::set<void *>> observed;
static cublasStatus_t ObserveMatmul(cublasLtHandle_t h, cublasLtMatmulDesc_t op, const void *alpha,
                                    const void *a, cublasLtMatrixLayout_t ad, const void *b,
                                    cublasLtMatrixLayout_t bd, const void *beta, const void *c,
                                    cublasLtMatrixLayout_t cd, void *d, cublasLtMatrixLayout_t dd,
                                    const cublasLtMatmulAlgo_t *algo, void *workspace, size_t bytes,
                                    cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    unsigned long long id = 0;
    if (cudaStreamGetCaptureInfo(stream, &status, &id) == cudaSuccess &&
        status == cudaStreamCaptureStatusActive) {
        std::lock_guard<std::mutex> lock(observedMutex);
        observed[id].insert(workspace);
    }
    return cublasLtMatmul(h, op, alpha, a, ad, b, bd, beta, c, cd, d, dd, algo, workspace, bytes,
                          stream);
}
#define cublasLtMatmul ObserveMatmul
#include "fastllm-bf16-lt.cuh"
#undef cublasLtMatmul
namespace lt = fastllm_bf16_lt;
static void need(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void cu(cudaError_t result) { need(result == cudaSuccess, cudaGetErrorString(result)); }
constexpr int M = 8, K = 4096, N = 1536;
struct Pack {
    void *input = nullptr, *weight = nullptr, *output = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::vector<unsigned short> reference;
    void init(int seed) {
        cu(cudaSetDevice(0));
        cu(cudaMalloc(&input, M * K * 2));
        cu(cudaMalloc(&weight, N * K * 2));
        cu(cudaMalloc(&output, M * N * 2));
        std::vector<__nv_bfloat16> data(N * K);
        unsigned rng = seed;
        for (auto &v : data) {
            rng = rng * 1664525 + 1013904223;
            v = __float2bfloat16((int(rng >> 16) - 32768) / 32768.f);
        }
        cu(cudaMemcpy(input, data.data(), M * K * 2, cudaMemcpyHostToDevice));
        cu(cudaMemcpy(weight, data.data(), N * K * 2, cudaMemcpyHostToDevice));
    }
    void warm() {
        need(lt::Matmul(input, weight, output, M, K, N), "Lt warm unavailable");
        reference.resize(M * N);
        cu(cudaMemcpy(reference.data(), output, M * N * 2, cudaMemcpyDeviceToHost));
    }
    void capture() {
        cu(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        need(lt::Matmul(input, weight, output, M, K, N), "Lt capture unavailable");
        cu(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        cu(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
    void check() {
        std::vector<unsigned short> data(M * N);
        cu(cudaMemcpy(data.data(), output, M * N * 2, cudaMemcpyDeviceToHost));
        need(data == reference, "concurrent graph/eager result corrupted");
    }
    void close() {
        if (exec)
            cu(cudaGraphExecDestroy(exec));
        if (graph)
            cu(cudaGraphDestroy(graph));
        exec = nullptr;
        graph = nullptr;
    }
    ~Pack() {
        close();
        cudaFree(input);
        cudaFree(weight);
        cudaFree(output);
    }
};
static size_t busy() {
    auto &p = lt::Pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    size_t n = 0;
    for (auto &b : p.devices[0])
        n += b->inUse;
    return n;
}
static void waitBusy(size_t expected) {
    for (int i = 0; i < 1000 && busy() != expected; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    need(busy() == expected, "workspace lease not returned");
}
int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
            return 77;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 8)
            return 77;
#if CUDART_VERSION < 11030
        return 77;
#endif
        cu(cudaSetDevice(0));
        // Capture two distinct graphs on ONE worker, then destroy the worker and
        // replay on independent streams while another worker performs eager GEMM.
        Pack p[3];
        for (int i = 0; i < 3; ++i)
            p[i].init(17 + i);
        std::exception_ptr error;
        std::thread owner([&]() {
            try {
                cu(cudaSetDevice(0));
                for (int i = 0; i < 2; ++i) {
                    p[i].warm();
                    p[i].capture();
                }
            } catch (...) {
                error = std::current_exception();
            }
        });
        owner.join();
        if (error)
            std::rethrow_exception(error);
        need(observed.size() == 2, "missing capture records");
        std::set<void *> scratch;
        for (auto &e : observed) {
            need(e.second.size() == 1, "capture changed scratch");
            scratch.insert(*e.second.begin());
        }
        bool separate = scratch.size() == 2;
        need(separate, "independent captures share workspace");
        waitBusy(2);
        cudaStream_t streams[2];
        for (auto &s : streams)
            cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        std::thread eager([&]() {
            try {
                cu(cudaSetDevice(0));
                p[2].warm();
                for (int i = 0; i < 256; ++i)
                    need(lt::Matmul(p[2].input, p[2].weight, p[2].output, M, K, N), "eager failed");
                cu(cudaStreamSynchronize(cudaStreamPerThread));
                p[2].check();
            } catch (...) {
                error = std::current_exception();
            }
        });
        for (int i = 0; i < 256; ++i)
            for (int j = 0; j < 2; ++j)
                cu(cudaGraphLaunch(p[j].exec, streams[j]));
        for (auto s : streams)
            cu(cudaStreamSynchronize(s));
        eager.join();
        if (error)
            std::rethrow_exception(error);
        p[0].check();
        p[1].check();
        // A CUDA clone/second instance retains the same capture resource until all
        // owners and asynchronous launches finish. Replay these dependent instances
        // sequentially, as their input/output buffers are shared too.
        cudaGraph_t clone = nullptr;
        cudaGraphExec_t copied = nullptr;
        cu(cudaGraphClone(&clone, p[0].graph));
        cu(cudaGraphInstantiate(&copied, clone, nullptr, nullptr, 0));
        p[0].close();
        p[1].close();
        waitBusy(1);
        cu(cudaGraphLaunch(copied, streams[0]));
        cu(cudaGraphExecDestroy(copied));
        cu(cudaGraphDestroy(clone));
        cu(cudaStreamSynchronize(streams[0]));
        p[0].check();
        waitBusy(0);
        // Recreating substantially more workers than the former eight lifetime
        // slots must keep using Lt. Source graph destruction releases its lease.
        for (int i = 0; i < 24; ++i) {
            std::thread worker([&]() {
                try {
                    cu(cudaSetDevice(0));
                    p[2].warm();
                    p[2].capture();
                    cu(cudaGraphLaunch(p[2].exec, cudaStreamPerThread));
                    cu(cudaStreamSynchronize(cudaStreamPerThread));
                    p[2].check();
                    p[2].close();
                } catch (...) {
                    error = std::current_exception();
                }
            });
            worker.join();
            if (error)
                std::rethrow_exception(error);
            waitBusy(0);
        }
        // Capture miss performs no search/allocation. Failed and abandoned captures
        // return leases once CUDA destroys the capture graph.
        std::thread failure([&]() {
            try {
                cu(cudaSetDevice(0));
                p[2].warm();
                auto *s = lt::GetState(0, false);
                size_t count = s->plans.size();
                cudaGraph_t g = nullptr;
                cu(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                need(!lt::Matmul(p[2].input, p[2].weight, p[2].output, 9, K, N),
                     "capture miss unexpectedly tuned");
                need(lt::Matmul(p[2].input, p[2].weight, p[2].output, M, K, N),
                     "capture unavailable");
                cu(cudaStreamEndCapture(cudaStreamPerThread, &g));
                cu(cudaGraphDestroy(g));
                need(s->plans.size() == count, "capture modified plan cache");
                p[2].warm();
                cu(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                need(lt::Matmul(p[2].input, p[2].weight, p[2].output, M, K, N),
                     "failed capture unavailable");
                need(cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess,
                     "capture not invalidated");
                cudaGetLastError();
                g = nullptr;
                need(cudaStreamEndCapture(cudaStreamPerThread, &g) != cudaSuccess,
                     "invalid capture succeeded");
                cudaGetLastError();
                if (g)
                    cu(cudaGraphDestroy(g));
            } catch (...) {
                error = std::current_exception();
            }
        });
        failure.join();
        if (error)
            std::rethrow_exception(error);
        waitBusy(0);
        // All leases, including graph-held leases, count towards one bounded pool;
        // exhaustion is temporary and immediately recovers after a return.
        std::vector<lt::Workspace *> leases;
        for (size_t i = 0; i < lt::maxWorkspacesPerDevice; ++i) {
            auto *w = lt::AcquireWorkspace(0);
            need(w, "pool exhausted early");
            leases.push_back(w);
        }
        need(!lt::AcquireWorkspace(0), "pool exceeded bound");
        auto *last = leases.back();
        lt::ReturnWorkspace(last);
        need(lt::AcquireWorkspace(0) == last, "returned slot not reusable");
        for (auto *w : leases)
            lt::ReturnWorkspace(w);
        waitBusy(0);
        for (auto s : streams)
            cu(cudaStreamDestroy(s));
        puts("PASS independent captures, cross-thread replay/eager, graph lifetime, 24 worker "
             "recreations, capture failure, bounded reusable leases");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
