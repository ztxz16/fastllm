#include "fastllm-cuda-gguf-planar.h"
#include "fastllm-gguf-planar-mixed.cuh"

namespace fastllm_gguf_planar_mixed {
template <ggml_type Gate, ggml_type Up, int T>
void Launch(const void *gate, const void *up, const int8_t *qs, const half2 *ds, const short2 *sums,
            half *output, int k, int n, int stride, cudaStream_t stream) {
    static thread_local int cachedDevice = -1, limit = 0;
    constexpr int warps = BlockWarps<Gate, Up, T>;
    int device = -1, blocks = (n + warps - 1) / warps;
    if (cudaGetDevice(&device) == cudaSuccess) {
        if (device != cachedDevice) {
            int active = 0, sms = 0;
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, Project<Gate, Up, T>, warps * 32, 0) ==
                    cudaSuccess &&
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
                active > 0 && sms > 0) {
                cachedDevice = device;
                limit = active * sms;
            }
        }
        if (device == cachedDevice)
            blocks = std::min(blocks, limit);
    }
    Project<Gate, Up, T><<<blocks, warps * 32, 0, stream>>>(gate, up, qs, ds, sums, output, k, n, stride);
}

template <ggml_type Gate, ggml_type Up>
void DispatchRows(const void *gate, const void *up, const void *workspace, half *output, int tokens, int k,
                  int n, int stride, cudaStream_t stream) {
    // Same-format pairs reuse the single-codebook kernel already instantiated
    // by the projection API, avoiding duplicate device code.
    if constexpr (Gate == Up) {
        FastllmGgufProjectPlanar(Gate, 3, gate, up, workspace, output, tokens, k, n, stride, stream);
    } else {
        const auto *qs = static_cast<const int8_t *>(workspace);
        const auto *ds = reinterpret_cast<const half2 *>(qs + size_t(tokens) * k);
        const auto *sums = reinterpret_cast<const short2 *>(ds + size_t(tokens) * (k / 32));
        for (int first = 0; first < tokens; first += 8) {
#define ROWS(T)                                                                                              \
    case T:                                                                                                  \
        Launch<Gate, Up, T>(gate, up, qs + size_t(first) * k, ds + size_t(first) * (k / 32),                 \
                            sums + size_t(first) * (k / 32), output + size_t(first) * stride, k, n, stride,  \
                            stream);                                                                         \
        break
            switch (std::min(tokens - first, 8)) {
                ROWS(1);
                ROWS(2);
                ROWS(3);
                ROWS(4);
                ROWS(5);
                ROWS(6);
                ROWS(7);
                ROWS(8);
            }
#undef ROWS
        }
    }
}

template <ggml_type Gate>
void DispatchUp(int upType, const void *gate, const void *up, const void *workspace, half *output, int tokens,
                int k, int n, int stride, cudaStream_t stream) {
#define UP(U)                                                                                                \
    case U:                                                                                                  \
        DispatchRows<Gate, U>(gate, up, workspace, output, tokens, k, n, stride, stream);                    \
        break
    switch (static_cast<ggml_type>(upType)) {
        UP(GGML_TYPE_IQ3_S);
        UP(GGML_TYPE_IQ3_XXS);
        UP(GGML_TYPE_IQ4_XS);
        UP(GGML_TYPE_Q4_K);
        UP(GGML_TYPE_Q2_K);
        UP(GGML_TYPE_IQ2_S);
        UP(GGML_TYPE_IQ2_XS);
        UP(GGML_TYPE_IQ2_XXS);
        UP(GGML_TYPE_IQ1_M);
    default:
        break; // Both formats validated before dispatch.
    }
#undef UP
}
} // namespace fastllm_gguf_planar_mixed

bool FastllmGgufGateUpPlanar(int gateType, int upType, const void *gate, const void *up,
                             const void *workspace, void *output, int tokens, int columns, int outputRows,
                             int outputStride, void *stream) {
    if (!FastllmGgufPlanarSupported(gateType, tokens, columns, outputRows) ||
        !FastllmGgufPlanarSupported(upType, tokens, columns, outputRows) || !gate || !up || !workspace ||
        !output || (reinterpret_cast<uintptr_t>(workspace) & 15) || outputStride < outputRows)
        return false;
#define GATE(G)                                                                                              \
    case G:                                                                                                  \
        fastllm_gguf_planar_mixed::DispatchUp<G>(upType, gate, up, workspace, static_cast<half *>(output),   \
                                                 tokens, columns, outputRows, outputStride,                  \
                                                 static_cast<cudaStream_t>(stream));                         \
        return true
    switch (static_cast<ggml_type>(gateType)) {
        GATE(GGML_TYPE_IQ3_S);
        GATE(GGML_TYPE_IQ3_XXS);
        GATE(GGML_TYPE_IQ4_XS);
        GATE(GGML_TYPE_Q4_K);
        GATE(GGML_TYPE_Q2_K);
        GATE(GGML_TYPE_IQ2_S);
        GATE(GGML_TYPE_IQ2_XS);
        GATE(GGML_TYPE_IQ2_XXS);
        GATE(GGML_TYPE_IQ1_M);
    default:
        return false;
    }
#undef GATE
}
