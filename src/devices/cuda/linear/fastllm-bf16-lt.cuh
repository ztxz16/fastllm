#pragma once

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Tune small BF16 GEMMs using algorithms supported by the current GPU/library.
// Large prefill retains GemmEx. Calls use the per-thread default stream.
namespace fastllm_bf16_lt {
constexpr size_t workspaceBytes = 8ull << 20;
constexpr size_t maxPlans = 64;
constexpr size_t maxWorkspacesPerDevice = 16; // 128 MiB, including graph-held buffers
constexpr int maxCandidates = 8;
constexpr int timingRepeats = 8;
using Key = std::array<int, 7>; // M, N, K, alignment(A), alignment(B), alignment(C/D), storage type

inline void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("BF16 cuBLASLt: ") + cudaGetErrorString(status));
    }
}

inline bool CheckBlas(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH ||
        status == CUBLAS_STATUS_ALLOC_FAILED) return false;
    // Unsupported algorithms may fall back; execution/internal errors must
    // propagate instead of silently submitting another GEMM on a failed stream.
    throw std::runtime_error("BF16 cuBLASLt status " + std::to_string(static_cast<int>(status)));
}

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool valid = false;

    ~Plan() {
        if (op) cublasLtMatmulDescDestroy(op);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
    }
};

struct SearchResources {
    cublasLtMatmulPreference_t preference = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;

    ~SearchResources() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
    }
};

// Buffers are leased, not permanently assigned to host-thread identities.
// CUDA user-object callbacks only return a lease; they never call CUDA APIs.
struct Workspace {
    int device;
    void *data = nullptr;
    bool inUse = true;
    explicit Workspace(int device) : device(device) {}
};

struct WorkspacePool {
    std::mutex mutex;
    std::map<int, std::vector<std::unique_ptr<Workspace>>> devices;
};

inline WorkspacePool &Pool() {
    // User-object callbacks may arrive after host-thread/static destruction.
    // Keep this bounded cache alive until CUDA releases the process context.
    static auto *pool = new WorkspacePool;
    return *pool;
}

inline void ReturnWorkspace(Workspace *workspace) {
    if (!workspace) return;
    auto &pool = Pool();
    std::lock_guard<std::mutex> guard(pool.mutex);
    workspace->inUse = false;
}

inline void CUDART_CB ReturnGraphWorkspace(void *ptr) {
    ReturnWorkspace(static_cast<Workspace *>(ptr));
}

inline Workspace *AcquireWorkspace(int device) {
    Workspace *workspace = nullptr;
    auto &pool = Pool();
    {
        std::lock_guard<std::mutex> guard(pool.mutex);
        auto &buffers = pool.devices[device];
        for (auto &buffer : buffers) {
            if (!buffer->inUse) {
                workspace = buffer.get();
                workspace->inUse = true;
                break;
            }
        }
        if (!workspace && buffers.size() < maxWorkspacesPerDevice) {
            buffers.emplace_back(new Workspace(device));
            workspace = buffers.back().get();
        }
    }
    if (!workspace) return nullptr;
    // Never allocate/synchronize while holding a lock shared by ranks or a
    // CUDA destructor callback. Existing cached allocations need no CUDA call.
    try {
        if (!workspace->data) {
            CheckCuda(cudaPeekAtLastError());
            cudaError_t status = cudaMalloc(&workspace->data, workspaceBytes);
            if (status == cudaErrorMemoryAllocation) {
                cudaError_t pending = cudaGetLastError();
                if (pending != cudaErrorMemoryAllocation) CheckCuda(pending);
                ReturnWorkspace(workspace);
                return nullptr;
            }
            CheckCuda(status);
        }
    } catch (...) {
        ReturnWorkspace(workspace);
        throw;
    }
    return workspace;
}

struct State {
    int device;
    cublasLtHandle_t handle = nullptr;
    Workspace *eager = nullptr, *spare = nullptr;
    unsigned long long captureId = 0;
    Workspace *captured = nullptr; // borrowed only while captureId is active
    std::map<Key, std::unique_ptr<Plan>> plans;

    explicit State(int device) : device(device) {}
    ~State() {
        int previous = -1;
        cudaGetDevice(&previous);
        if (cudaSetDevice(device) == cudaSuccess) {
            // Only eager calls use this thread's scratch. Graphs own separate
            // leases and can outlive this state, handle and plan descriptors.
            if (!eager || cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess)
                ReturnWorkspace(eager);
            ReturnWorkspace(spare); // never used by GPU work
            plans.clear();
            if (handle) cublasLtDestroy(handle);
            if (previous >= 0 && previous != device) cudaSetDevice(previous);
        }
    }
};

inline State *GetState(int device, bool capture) {
    static thread_local std::map<int, std::unique_ptr<State>> local;
    auto found = local.find(device);
    if (found == local.end()) {
        if (capture) return nullptr;
        auto state = std::make_unique<State>(device);
        if (!CheckBlas(cublasLtCreate(&state->handle))) return nullptr;
        found = local.emplace(device, std::move(state)).first;
    }
    auto &state = *found->second;
    if (!capture) {
        state.captured = nullptr;
        state.captureId = 0;
        if (!state.eager) state.eager = AcquireWorkspace(device);
        if (!state.eager) return nullptr; // retry once other owners return leases
#if CUDART_VERSION >= 11030
        // Allocate before capture, never within it. One reserve is enough for
        // the next capture; another eager call can prepare the following one.
        if (!state.spare) state.spare = AcquireWorkspace(device);
#endif
    }
    return &state;
}

inline void *CaptureWorkspace(State &state) {
#if CUDART_VERSION >= 11030
    cudaStreamCaptureStatus status;
    unsigned long long id = 0;
    cudaGraph_t graph = nullptr;
#if CUDART_VERSION >= 12000
    CheckCuda(cudaStreamGetCaptureInfo(cudaStreamPerThread, &status, &id, &graph));
#else
    CheckCuda(cudaStreamGetCaptureInfo_v2(cudaStreamPerThread, &status, &id,
                                         &graph, nullptr, nullptr));
#endif
    if (status != cudaStreamCaptureStatusActive || !graph)
        throw std::runtime_error("BF16 cuBLASLt capture is not active");
    if (id != state.captureId) {
        if (!state.spare) return nullptr;
        Workspace *workspace = state.spare;
        state.spare = nullptr;
        cudaUserObject_t owner = nullptr;
        cudaError_t result = cudaUserObjectCreate(&owner, workspace,
            ReturnGraphWorkspace, 1, cudaUserObjectNoDestructorSync);
        if (result != cudaSuccess) {
            ReturnWorkspace(workspace);
            CheckCuda(result);
        }
        result = cudaGraphRetainUserObject(graph, owner, 1, cudaGraphUserObjectMove);
        if (result != cudaSuccess) {
            // The CUDA callback returns the lease, including failed captures.
            cudaUserObjectRelease(owner, 1);
            CheckCuda(result);
        }
        state.captureId = id;
        state.captured = workspace;
    }
    return state.captured->data;
#else
    return nullptr; // older runtimes keep the existing GemmEx capture path
#endif
}

inline int Alignment(const void *p) {
    uintptr_t value = reinterpret_cast<uintptr_t>(p);
    int alignment = 1;
    while (alignment < 256 && (value & (alignment * 2 - 1)) == 0) alignment *= 2;
    return alignment;
}

inline bool Run(State &s, Plan &p, const cublasLtMatmulAlgo_t &algo,
                const void *weight, const void *input, void *output, void *workspace) {
    float alpha = 1.0f, beta = 0.0f;
    return CheckBlas(cublasLtMatmul(s.handle, p.op, &alpha, weight, p.a, input, p.b,
        &beta, output, p.c, output, p.c, &algo, workspace, workspaceBytes, cudaStreamPerThread));
}

inline void MakePlan(State &s, Plan &p, const Key &key,
                     const void *weight, const void *input, void *output) {
    const int M = key[0], N = key[1], K = key[2];
    const auto dtype = static_cast<cudaDataType_t>(key[6]);
    cublasOperation_t trans = CUBLAS_OP_T;
    if (!CheckBlas(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) ||
        !CheckBlas(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans))) ||
        !CheckBlas(cublasLtMatrixLayoutCreate(&p.a, dtype, K, N, K)) ||
        !CheckBlas(cublasLtMatrixLayoutCreate(&p.b, dtype, K, M, K)) ||
        !CheckBlas(cublasLtMatrixLayoutCreate(&p.c, dtype, N, M, N))) return;

    SearchResources search;
    if (!CheckBlas(cublasLtMatmulPreferenceCreate(&search.preference))) return;
    // Allow unsplit GEMM or split-K with FP32 intermediate results. This avoids
    // BF16 partial-sum rounding, but does NOT reproduce the single-row GEMV tree.
    uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    if (!CheckBlas(cublasLtMatmulPreferenceSetAttribute(search.preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes))) ||
        !CheckBlas(cublasLtMatmulPreferenceSetAttribute(search.preference,
            CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction, sizeof(reduction)))) return;
    const cublasLtMatmulPreferenceAttributes_t attributes[] = {
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
    for (int i = 0; i < 4; ++i) {
        uint32_t alignment = key[3 + std::min(i, 2)];
        if (!CheckBlas(cublasLtMatmulPreferenceSetAttribute(search.preference,
                attributes[i], &alignment, sizeof(alignment)))) return;
    }
    cublasLtMatmulHeuristicResult_t candidates[maxCandidates]{};
    int count = 0;
    if (!CheckBlas(cublasLtMatmulAlgoGetHeuristic(s.handle, p.op, p.a, p.b, p.c, p.c,
            search.preference, maxCandidates, candidates, &count)) || count == 0) return;
    CheckCuda(cudaEventCreate(&search.begin));
    CheckCuda(cudaEventCreate(&search.end));
    float best = std::numeric_limits<float>::max();
    for (int i = 0; i < count; ++i) {
        if (candidates[i].state != CUBLAS_STATUS_SUCCESS ||
            candidates[i].workspaceSize > workspaceBytes) continue;
        const auto &algo = candidates[i].algo;
        if (!Run(s, p, algo, weight, input, output, s.eager->data)) continue;
        // First use only, outside capture. Inputs and weights must not alias
        // output; each trial overwrites the same Linear output with beta = 0.
        CheckCuda(cudaEventRecord(search.begin, cudaStreamPerThread));
        bool success = true;
        for (int repeat = 0; repeat < timingRepeats; ++repeat) {
            if (!Run(s, p, algo, weight, input, output, s.eager->data)) {
                success = false;
                break;
            }
        }
        CheckCuda(cudaEventRecord(search.end, cudaStreamPerThread));
        CheckCuda(cudaEventSynchronize(search.end));
        float ms = 0.0f;
        CheckCuda(cudaEventElapsedTime(&ms, search.begin, search.end));
        if (success && ms < best) {
            best = ms;
            p.algo = algo;
            p.valid = true;
        }
    }
}

inline bool Matmul(const void *input, const void *weight, void *output, int M, int K, int N,
                   cudaDataType_t storage = CUDA_R_16BF) {
    // This bounds tuning cost, not model/architecture-specific dispatch.
    if (N <= 0 || K <= 0) return false;
    if (storage == CUDA_R_16BF) {
        if (M < 8 || M > 32) return false;
    } else if (storage == CUDA_R_16F) {
        // Large decode heads amortize tuning and saturate Hopper bandwidth.
        // Other shapes retain their native GEMV and exact-row reduction tree.
        if (M < 1 || M >= 32 || N < 65536 || (N % 8) || (K % 8)) return false;
    } else return false;
    cudaStreamCaptureStatus capture;
    CheckCuda(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
    int device = 0;
    CheckCuda(cudaGetDevice(&device));
    State *s = GetState(device, capture != cudaStreamCaptureStatusNone);
    if (!s) return false;
    Key key{M, N, K, Alignment(weight), Alignment(input), Alignment(output), static_cast<int>(storage)};
    auto found = s->plans.find(key);
    if (found == s->plans.end()) {
        if (capture != cudaStreamCaptureStatusNone) return false;
        if (s->plans.size() >= maxPlans) s->plans.erase(s->plans.begin());
        auto plan = std::make_unique<Plan>();
        MakePlan(*s, *plan, key, weight, input, output);
        found = s->plans.emplace(key, std::move(plan)).first;
    }
    Plan &p = *found->second;
    if (!p.valid) return false;
    void *workspace = capture == cudaStreamCaptureStatusNone
        ? s->eager->data : CaptureWorkspace(*s);
    return workspace && Run(*s, p, p.algo, weight, input, output, workspace);
}
} // namespace fastllm_bf16_lt
