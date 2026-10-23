#include "fastllm-gguf-mmvq-kernels.cuh"
#undef cudaMalloc
#include <vector>
#include <utility>
#include <cstring>
#include <cstdio>
#include <stdexcept>

static void Cuda(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static void Check(bool ok, const char *s) {
    if (!ok)
        throw std::runtime_error(s);
}
// Frozen legacy Q4_K decode reduction, independent of the new implementation.
template <typename T> __global__ void LegacyQ4(const void *w, const block_q8_1 *x, T *y, int columns) {
    constexpr int qi = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qi;
    constexpr int vdr = get_vdr_mmvq(GGML_TYPE_Q4_K);
    const int tid = 32 * threadIdx.y + threadIdx.x;
    float sum = 0;
    for (int b = tid / (qi / vdr); b < columns / 256; b += vdr * 128 / qi)
        sum += vec_dot_q4_K_q8_1(w, x + 8 * b, blockIdx.x * (columns / 256) + b, vdr * (tid % (qi / vdr)));
    __shared__ float tmp[3][32];
    if (threadIdx.y > 0)
        tmp[threadIdx.y - 1][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y > 0)
        return;
#pragma unroll
    for (int k = 0; k < 3; ++k)
        sum += tmp[k][threadIdx.x];
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0)
        FastllmGgufStore<0>(y + blockIdx.x, sum);
}

template <typename T> void Case(int rows, int columns, int pattern) {
    unsigned rng = 9919 + rows * 31 + columns;
    auto next = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return rng >> 24;
    };
    std::vector<block_q4_K> hw(size_t(rows) * columns / 256);
    for (auto &b : hw) {
        for (auto &q : b.qs)
            q = next();
        for (auto &v : b.scales)
            v = next();
        b.dm = __floats2half2_rn((int(next() % 7) - 3) * .00137f, (int(next() % 7) - 3) * .00091f);
    }
    std::vector<block_q8_1> hx(columns / 32);
    for (auto &b : hx) {
        int sum = 0;
        float d = pattern ? 0.f : (int(next() % 7) - 3) * .0137f;
        for (auto &q : b.qs) {
            q = pattern ? 0 : int(next()) - 128;
            sum += q;
        }
        b.ds = __floats2half2_rn(d, sum * d);
    }
    block_q4_K *w;
    block_q8_1 *x;
    T *actual, *ref;
    Cuda(cudaMalloc(&w, hw.size() * sizeof(block_q4_K)));
    Cuda(cudaMemcpy(w, hw.data(), hw.size() * sizeof(block_q4_K), cudaMemcpyHostToDevice));
    Cuda(cudaMalloc(&x, hx.size() * sizeof(block_q8_1)));
    Cuda(cudaMemcpy(x, hx.data(), hx.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
    Cuda(cudaMalloc(&actual, (rows + 16) * sizeof(T)));
    Cuda(cudaMalloc(&ref, rows * sizeof(T)));
    Check(!fastllm_gguf_q4_k::Supports(nullptr, x, 1280, 8192) &&
          !fastllm_gguf_q4_k::Supports(w, nullptr, 1280, 8192) &&
          !fastllm_gguf_q4_k::Supports(reinterpret_cast<const char *>(w) + 1, x, 1280, 8192) &&
          !fastllm_gguf_q4_k::Supports(w, reinterpret_cast<const char *>(x) + 1, 1280, 8192) &&
          !fastllm_gguf_q4_k::Supports(w, x, 1280, 8191) &&
          !fastllm_gguf_q4_k::Supports(w, x, 2304, 4095) &&
          !fastllm_gguf_q4_k::Supports(w, x, 5120, 32767) &&
          !fastllm_gguf_q4_k::Supports(w, x, 4096, 5120), "Q4 admission guard changed");
    auto launch = [&](int mode, T *out) {
        if (mode == 0)
            LegacyQ4<<<rows, dim3(32, 4), 0, cudaStreamPerThread>>>(w, x, out, columns);
        if (mode == 1)
            fastllm_gguf_q4_k::Gemv<T, 4><<<(rows + 3) / 4, 128, 0, cudaStreamPerThread>>>(w, x, out, columns, rows);
        if (mode == 2)
            fastllm_gguf_q4_k::PackedGemv<T, 2, 2>
                <<<(rows + 1) / 2, 128, 0, cudaStreamPerThread>>>(w, x, out, columns, rows);
        if (mode == 3)
            mul_mat_vec_q_cuda<GGML_TYPE_Q4_K, T, 0>(w, x, out, nullptr, columns, rows, columns, 1, rows, 1, 0, 0, 0, 0,
                                                     cudaStreamPerThread);
    };
    launch(0, ref);
    std::vector<T> expected(rows), observed(rows + 16);
    Cuda(cudaMemcpy(expected.data(), ref, rows * sizeof(T), cudaMemcpyDeviceToHost));
    for (int mode = 1; mode <= 3; ++mode) {
        Cuda(cudaMemset(actual, 0x5a, (rows + 16) * sizeof(T)));
        cudaGraph_t g;
        cudaGraphExec_t e;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        launch(mode, actual);
        Cuda(cudaStreamEndCapture(cudaStreamPerThread, &g));
        Cuda(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
        Cuda(cudaGraphLaunch(e, cudaStreamPerThread));
        Cuda(cudaMemcpy(observed.data(), actual, (rows + 16) * sizeof(T), cudaMemcpyDeviceToHost));
        Check(std::memcmp(expected.data(), observed.data(), rows * sizeof(T)) == 0, "Q4 bitwise mismatch");
        const auto *tail = reinterpret_cast<unsigned char *>(observed.data() + rows);
        for (size_t i = 0; i < 16 * sizeof(T); ++i)
            Check(tail[i] == 0x5a, "Q4 output tail changed");
        Cuda(cudaGraphExecDestroy(e));
        Cuda(cudaGraphDestroy(g));
    }
    printf("CHECK rows=%d cols=%d typebytes=%zu pattern=%d exact=1\n", rows, columns, sizeof(T), pattern);
    Cuda(cudaFree(w));
    Cuda(cudaFree(x));
    Cuda(cudaFree(actual));
    Cuda(cudaFree(ref));
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        Cuda(cudaSetDevice(0));
        for (int n : {1, 7, 8, 9, 127, 129})
            for (int k : {256, 768, 1280, 2304, 4096, 5120, 8192}) {
                Case<float>(n, k, 0);
                Case<half>(n, k, 0);
                Case<__nv_bfloat16>(n, k, 0);
            }
        Case<float>(129, 5120, 1);
        Case<half>(129, 5120, 1);
        Case<__nv_bfloat16>(129, 5120, 1);
        for (auto shape : std::vector<std::pair<int, int>>{{8191, 1280},
                                                           {8192, 1280},
                                                           {16385, 1280},
                                                           {4095, 2304},
                                                           {4096, 2304},
                                                           {5121, 2304},
                                                           {32767, 5120},
                                                           {32768, 5120},
                                                           {64769, 5120},
                                                           {1280, 5120},
                                                           {16384, 1280},
                                                           {512, 5120},
                                                           {5120, 4096},
                                                           {4608, 5120},
                                                           {5120, 2304},
                                                           {64768, 5120},
                                                           {64512, 5120}}) {
            Case<float>(shape.first, shape.second, 0);
            Case<half>(shape.first, shape.second, 0);
            Case<__nv_bfloat16>(shape.first, shape.second, 0);
        }
        puts("PASS: Q4 FP32/FP16/BF16 bitwise parity, tails, zero input and CUDA Graph");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
