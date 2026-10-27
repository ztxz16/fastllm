#include "naive-n05-attention-hopper.cuh"
namespace bulk = naive_attention_bulk;
#include <algorithm>
#include <vector>
static bool TensorSupported() {
    int d = 0;
    cudaDeviceProp p{};
    return cudaGetDevice(&d) == cudaSuccess && cudaGetDeviceProperties(&p, d) == cudaSuccess &&
           p.major >= 8;
}
template <int Max, bool Ring>
void launch(const __nv_bfloat16 *q, bulk::Batch b, const float *s, __nv_bfloat16 *o, int n, int h,
            int kh, cudaStream_t stream) {
    cudaFuncSetAttribute(bulk::Decode<Max, Ring>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sizeof(bulk::Storage<Max>));
    bulk::Decode<Max, Ring>
        <<<dim3(n, h / 8), 256, sizeof(bulk::Storage<Max>), stream>>>(q, b, s, o, h, kh);
}
extern "C" int run(const void *q, const FastllmNaiveDSARow *rows, const void *sink, void *out,
                   int n, int h, int kh, int cap, int ring, void *stream) {
    if (!TensorSupported())
        return cudaErrorNotSupported;
    if (n < 1 || h < 8 || kh < 1 || h % kh || (h / kh) % 8 || cap < 1 || cap > 2048 || !q || !out)
        return cudaErrorInvalidValue;
    for (int i = 0; i < n; i++)
        if (!rows[i].key || !rows[i].value || rows[i].keyStride % 8 ||
            (uintptr_t)rows[i].key % 16 || (uintptr_t)rows[i].value % 16)
            return cudaErrorNotSupported;

    for (int first = 0; first < n; first += bulk::BatchLimit) {
        bulk::Batch b{};
        int count = std::min(bulk::BatchLimit, n - first);
        std::copy_n(rows + first, count, b.rows);
        auto Q = (const __nv_bfloat16 *)q + (size_t)first * h * 192;
        auto O = (__nv_bfloat16 *)out + (size_t)first * h * 128;
        auto S = (const float *)sink;
        auto ST = (cudaStream_t)stream;
        if (ring)
            launch<128, true>(Q, b, S, O, count, h, kh, ST);
        else if (cap <= 1024)
            launch<1024, false>(Q, b, S, O, count, h, kh, ST);
        else
            launch<2048, false>(Q, b, S, O, count, h, kh, ST);
    }
    return cudaGetLastError();
}

#include "old_swa.cuh"
extern "C" int old(const void *q, const FastllmNaiveDSARow *rows, const void *sink, void *out,
                   void *scratch, int n, int h, int kh, int cap, int ring, void *stream) {
    auto st = (cudaStream_t)stream;
    if (ring) {
        for (int first = 0; first < n; first += 16) {
            int count = std::min(16, n - first);
            BatchSwaKeys b{};
            for (int i = 0; i < count; i++)
                b.rows[i] = {rows[first + i].key, rows[first + i].value};
            b.length = rows[first].liveKeys;
            AttentionShortDecode<128, BatchSwaKeys, false, 1024, 32, true>
                <<<dim3(h, count, 4), 1024, 0, st>>>(
                    (const BF16 *)q + (size_t)first * h * 192, nullptr, nullptr,
                    (const float *)sink, (BF16 *)out + (size_t)first * h * 128, h, kh, kh * 192, b);
        }
    } else {
        namespace d = naive_dsa_decode;
        cudaFuncSetAttribute(d::SoftmaxSparseValues<2048, 32>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sizeof(d::SparseStorage<2048, 32>));
        cudaFuncSetAttribute(d::GroupedValues<1024, 4, 16, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sizeof(d::GroupStorage<1024, 4, 16>));
        cudaFuncSetAttribute(d::GroupedValues<2048, 4, 16, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sizeof(d::GroupStorage<2048, 4, 16>));
        for (int first = 0; first < n; first += 32) {
            int count = std::min(32, n - first);
            FastllmNaiveDSABatch b{};
            std::copy_n(rows + first, count, b.rows);
            auto Q = (const BF16 *)q + (size_t)first * h * 192;
            auto O = (BF16 *)out + (size_t)first * h * 128;
            auto S = (float *)scratch;
            d::DirectScores<4, 4>
                <<<dim3(h, (cap + 15) / 16, count), 128, 0, st>>>(Q, b, S, h, kh, cap);
            if (rows[first].indices)
                d::SoftmaxSparseValues<2048, 32>
                    <<<dim3(h, 4, count), 1024, sizeof(d::SparseStorage<2048, 32>), st>>>(
                        S, (const float *)sink, b, O, h, kh, cap);
            else if (cap <= 1024)
                d::GroupedValues<1024, 4, 16, false>
                    <<<dim3(h / 4, 8, count), 1024, sizeof(d::GroupStorage<1024, 4, 16>), st>>>(
                        S, (const float *)sink, b, O, h, kh, cap);
            else
                d::GroupedValues<2048, 4, 16, false>
                    <<<dim3(h / 4, 8, count), 1024, sizeof(d::GroupStorage<2048, 4, 16>), st>>>(
                        S, (const float *)sink, b, O, h, kh, cap);
        }
    }
    return cudaGetLastError();
}

extern "C" int supports_tma() {
#if CUDART_VERSION >= 12010
    int dev = 0;
    cudaDeviceProp prop{};
    cudaFuncAttributes attr{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess ||
        prop.major != 9 || prop.minor != 0)
        return 0;
    if (cudaFuncGetAttributes(&attr, bulk::HopperDecode<1024>) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return attr.ptxVersion >= 90;
#else
    return 0;
#endif
}
extern "C" int run_tma(const void *q, const FastllmNaiveDSARow *rows, const void *sink, void *out,
                       int n, int h, int kh, int cap, void *stream) {
#if CUDART_VERSION >= 12010
    if (!supports_tma())
        return cudaErrorNotSupported;
    if (h < 8 || h % 8 || kh != 1 || cap < 1 || cap > 2048 || n < 1)
        return cudaErrorNotSupported;

    std::vector<bulk::TmaBatch> batches((n + bulk::TmaBatchLimit - 1) / bulk::TmaBatchLimit);
    for (int first = 0; first < n; first += bulk::TmaBatchLimit) {
        int count = std::min(bulk::TmaBatchLimit, n - first);
        if (!bulk::PrepareTmaBatch(batches[first / bulk::TmaBatchLimit], rows + first, count))
            return cudaErrorNotSupported;
    }
    auto st = (cudaStream_t)stream;
    for (int first = 0; first < n; first += bulk::TmaBatchLimit) {
        int count = std::min(bulk::TmaBatchLimit, n - first);
        const auto &b = batches[first / bulk::TmaBatchLimit];
        auto Q = (const BF16 *)q + (size_t)first * h * 192;
        auto O = (BF16 *)out + (size_t)first * h * 128;
        if (cap <= 1024) {
            cudaFuncSetAttribute(bulk::HopperDecode<1024>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 sizeof(bulk::TmaStorage<1024>));
            bulk::HopperDecode<1024>
                <<<dim3(count, h / 8), 256, sizeof(bulk::TmaStorage<1024>), st>>>(
                    Q, b, (const float *)sink, O, h);
        } else {
            cudaFuncSetAttribute(bulk::HopperDecode<2048>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 sizeof(bulk::TmaStorage<2048>));
            bulk::HopperDecode<2048>
                <<<dim3(count, h / 8), 256, sizeof(bulk::TmaStorage<2048>), st>>>(
                    Q, b, (const float *)sink, O, h);
        }
    }
    return cudaGetLastError();
#else
    return cudaErrorNotSupported;
#endif
}
