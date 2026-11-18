#pragma once

#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

// Host-only plans: operands and workspace belong to the caller's graph pool.
// Each worker/device owns its descriptors, including the mutable scale pointers.
namespace fastllm_mxfp8_lt {
inline bool Supported(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH ||
        status == CUBLAS_STATUS_INVALID_VALUE || status == CUBLAS_STATUS_ALLOC_FAILED) return false;
    throw std::runtime_error("MXFP8 cuBLASLt status " + std::to_string(int(status)));
}

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t workspaceBytes = 0;
    uint64_t lastUse = 0;
    bool valid = false;
    ~Plan() {
        if (op) cublasLtMatmulDescDestroy(op);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
    }
};

struct State {
    cublasLtHandle_t handle = nullptr;
    std::map<std::array<int, 3>, std::unique_ptr<Plan>> plans;
    uint64_t clock = 0;
    ~State() { plans.clear(); if (handle) cublasLtDestroy(handle); }
};

inline State *GetState(int device, bool capturing) {
    static thread_local std::map<int, std::unique_ptr<State>> states;
    auto found = states.find(device);
    if (found == states.end()) {
        if (capturing) return nullptr;
        auto state = std::make_unique<State>();
        if (!Supported(cublasLtCreate(&state->handle))) return nullptr;
        found = states.emplace(device, std::move(state)).first;
    }
    return found->second.get();
}

inline bool SetScales(Plan &p, const void *weightScales, const void *inputScales) {
    return Supported(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
               &weightScales, sizeof(weightScales))) &&
           Supported(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
               &inputScales, sizeof(inputScales)));
}

inline void Initialize(State &s, Plan &p, int m, int n, int k,
                       const void *weightScales, const void *inputScales) {
    // W[N,K] and X[M,K] are interpreted as column-major [K,N] and [K,M].
    // Their TN product is column-major [N,M], i.e. row-major output [M,N].
    if (!Supported(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F))) return;
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    if (!Supported(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans))) ||
        !Supported(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode))) ||
        !Supported(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode))) ||
        !SetScales(p, weightScales, inputScales) ||
        !Supported(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8F_E4M3, k, n, k)) ||
        !Supported(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8F_E4M3, k, m, k)) ||
        !Supported(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, n, m, n))) return;
    cublasLtMatmulPreference_t preference = nullptr;
    if (!Supported(cublasLtMatmulPreferenceCreate(&preference))) return;
    constexpr size_t limit = 32ull << 20;
    auto status = cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit));
    // All operands are checked/aligned to at least 16 bytes by the adapter.
    const uint32_t alignment = 16;
    for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                     CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                     CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                     CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulPreferenceSetAttribute(preference, attr, &alignment, sizeof(alignment));
    }
    cublasLtMatmulHeuristicResult_t candidates[8]{};
    int count = 0;
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulAlgoGetHeuristic(s.handle, p.op, p.a, p.b, p.c, p.c,
                                              preference, 8, candidates, &count);
    cublasLtMatmulPreferenceDestroy(preference);
    if (!Supported(status)) return;
    for (int i = 0; i < count; ++i) {
        if (candidates[i].state != CUBLAS_STATUS_SUCCESS || candidates[i].workspaceSize > limit) continue;
        p.algo = candidates[i].algo;
        p.workspaceBytes = candidates[i].workspaceSize;
        p.valid = true;
        break;
    }
}

inline Plan *GetPlan(State &s, int m, int n, int k, const void *weightScales,
                     const void *inputScales, bool capturing) {
    const std::array<int, 3> key{m, n, k};
    auto found = s.plans.find(key);
    if (found == s.plans.end()) {
        if (capturing) return nullptr;
        // Bound host metadata, never batch size. Evicting descriptors cannot
        // invalidate CUDA graphs, which retain their own kernel parameters.
        if (s.plans.size() >= 256) {
            auto oldest = std::min_element(s.plans.begin(), s.plans.end(),
                [](const auto &a, const auto &b) { return a.second->lastUse < b.second->lastUse; });
            s.plans.erase(oldest);
        }
        auto plan = std::make_unique<Plan>();
        Initialize(s, *plan, m, n, k, weightScales, inputScales);
        found = s.plans.emplace(key, std::move(plan)).first;
    }
    found->second->lastUse = ++s.clock;
    return found->second->valid ? found->second.get() : nullptr;
}

inline bool Run(State &s, Plan &p, const void *weight, const void *input, void *output,
                const void *weightScales, const void *inputScales, void *workspace,
                cudaStream_t stream) {
    if (!SetScales(p, weightScales, inputScales)) return false;
    const float alpha = 1.0f, beta = 0.0f;
    return Supported(cublasLtMatmul(s.handle, p.op, &alpha, weight, p.a, input, p.b,
        &beta, output, p.c, output, p.c, &p.algo, workspace, p.workspaceBytes, stream));
}
} // namespace fastllm_mxfp8_lt
