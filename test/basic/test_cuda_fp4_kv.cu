#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t state) {
    if (state != cudaSuccess) throw std::runtime_error(cudaGetErrorString(state));
}
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void AllocateGpu(fastllm::Data &data) {
    data.dataDevice = fastllm::DataDevice::CUDA;
    data.dataDeviceIds = {0};
    data.Allocate();
}
static void InitInt(fastllm::Data &data, const std::vector<int> &values) {
    data.dataType = fastllm::DataType::INT32;
    data.UpdateUnitSize();
    data.Resize({(int)values.size()});
    data.cpuIntDatas = values;
    AllocateGpu(data);
    Check(cudaMemcpy(data.cudaData, values.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice));
}

// Independent nearest-neighbour reference, with ties to the even E2M1 code.
static uint8_t EncodeReference(float value) {
    const float levels[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    int best = 0;
    float error = std::fabs(value);
    for (int i = 1; i < 8; ++i) {
        float next = std::fabs(std::fabs(value) - levels[i]);
        if (next < error || (next == error && i % 2 == 0)) {
            best = i;
            error = next;
        }
    }
    return best | (std::signbit(value) ? 8 : 0);
}

// The shared dispatch must preserve the existing non-FP4 storage formats too.
template <typename SrcT, typename DstT>
static void RunCopyCase(fastllm::DataType srcType, fastllm::DataType dstType) {
    using namespace fastllm;
    const int heads = 2, dim = 128, pageLen = 16, page = 1, offset = 3;
    Data input(srcType, {heads, 1, dim}), cache(dstType, {2, pageLen, heads, dim});
    AllocateGpu(input); AllocateGpu(cache);
    std::vector<SrcT> source(input.Count(0));
    std::vector<DstT> expected(cache.Count(0), DstT(0.0f));
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = SrcT(std::sin(float(i) * 0.17f) * 7.0f);
        expected[(page * pageLen + offset) * heads * dim + i] = DstT(float(source[i]));
    }
    Check(cudaMemcpy(input.cudaData, source.data(), input.GetBytes(), cudaMemcpyHostToDevice));
    Data indices, offsets, qs, pages;
    InitInt(indices, {page}); InitInt(offsets, {offset});
    InitInt(qs, {0, 1}); InitInt(pages, {0, 1});
    for (int mode = 0; mode < 4; ++mode) {
        Check(cudaMemset(cache.cudaData, 0, cache.GetBytes()));
        auto *dst = (uint8_t*)cache.cudaData;
        auto *src = (uint8_t*)input.cudaData;
        if (mode == 0) {
            FastllmCudaPagedCacheCopy(dst, page, pageLen, heads, dim, dstType,
                                     src, srcType, 1, 0, 1, offset);
        } else if (mode == 1) {
            Require(FastllmCudaPagedCacheCopyMultiPage(dst, &page, 1, offset,
                pageLen, heads, dim, dstType, src, srcType, 1), "multi-page copy failed");
        } else if (mode == 2) {
            FastllmCudaPagedCacheCopyBatch(dst, (int32_t*)indices.cudaData,
                (int32_t*)offsets.cudaData, pageLen, 1, heads, dim, dstType, src, srcType, true);
        } else {
            Require(FastllmCudaPagedCacheAppendPackedBatch(dst, (int32_t*)qs.cudaData,
                (int32_t*)pages.cudaData, (int32_t*)indices.cudaData, (int32_t*)offsets.cudaData,
                1, 1, pageLen, heads, dim, dstType, src, srcType), "packed append failed");
        }
        Check(cudaDeviceSynchronize());
        std::vector<uint8_t> actual(cache.GetBytes());
        Check(cudaMemcpy(actual.data(), dst, actual.size(), cudaMemcpyDeviceToHost));
        Require(std::memcmp(actual.data(), expected.data(), actual.size()) == 0,
                "paged copy changed a non-FP4 format");
    }
}

template <typename SrcT>
static void RunCopyFormats(fastllm::DataType srcType) {
    RunCopyCase<SrcT, float>(srcType, fastllm::DataType::FLOAT32);
    RunCopyCase<SrcT, half>(srcType, fastllm::DataType::FLOAT16);
    RunCopyCase<SrcT, __nv_bfloat16>(srcType, fastllm::DataType::BFLOAT16);
    RunCopyCase<SrcT, __nv_fp8_e4m3>(srcType, fastllm::DataType::FP8_E4M3);
}

template <typename T>
static void RunCase(fastllm::DataType dtype, int dim, int tokens, int queries, bool packedAppend) {
    using namespace fastllm;
    const int pageLen = 16, heads = 2, qHeads = 8;
    const int pages = (tokens + pageLen - 1) / pageLen;
    const size_t pageElements = pageLen * heads * dim;
    const size_t pageBytes = pageElements / 16 * 9;
    std::vector<int> indices(pages);
    for (int p = 0; p < pages; ++p) indices[p] = pages - p - 1;
    Data source(dtype, {heads, tokens, dim});
    AllocateGpu(source);
    std::vector<T> input(heads * tokens * dim);
    PagedCacheManager kPool, vPool, kRefPool, vRefPool;
    for (auto *pool : {&kPool, &vPool, &kRefPool, &vRefPool}) {
        pool->dataType = pool == &kPool || pool == &vPool ? DataType::FP4_E2M1 : dtype;
        pool->UpdateUnitSize();
        pool->Resize({pages, pageLen, heads, dim});
        AllocateGpu(*pool);
    }
    Require(kPool.GetBytes() == pages * pageBytes, "FP4 allocation omits scale bytes");
    Require(GetDataBytes(DataType::FP4_E2M1, pageLen, heads * dim) == pageBytes,
            "FP4 budget differs from physical storage");
    Data appendQs, pageSizes, pageIds, baseLens;
    InitInt(appendQs, {0, tokens}); InitInt(pageSizes, {0, pages});
    InitInt(pageIds, indices); InitInt(baseLens, {0});
    for (int kv = 0; kv < 2; ++kv) {
        auto &pool = kv == 0 ? kPool : vPool;
        auto &refPool = kv == 0 ? kRefPool : vRefPool;
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = T(std::sin(float(i) * 0.17f + kv) * (0.1f + float(i % 17)));
        }
        Check(cudaMemcpy(source.cudaData, input.data(), source.GetBytes(), cudaMemcpyHostToDevice));
        Check(cudaMemset(pool.cudaData, 0, pool.GetBytes()));
        if (packedAppend) {
            Require(FastllmCudaPagedCacheAppendPackedBatch(
                (uint8_t*)pool.cudaData, (int32_t*)appendQs.cudaData, (int32_t*)pageSizes.cudaData,
                (int32_t*)pageIds.cudaData, (int32_t*)baseLens.cudaData, 1, tokens, pageLen,
                heads, dim, DataType::FP4_E2M1, (const uint8_t*)source.cudaData, dtype),
                "packed FP4 append failed");
        } else if (!FastllmCudaPagedCacheCopyMultiPage(
                (uint8_t*)pool.cudaData, indices.data(), pages, 0, pageLen, heads, dim,
                DataType::FP4_E2M1, (uint8_t*)source.cudaData, dtype, tokens)) {
            for (int p = 0; p < pages; ++p) {
                FastllmCudaPagedCacheCopy((uint8_t*)pool.cudaData, indices[p], pageLen,
                    heads, dim, DataType::FP4_E2M1, (uint8_t*)source.cudaData, dtype,
                    tokens, p * pageLen, std::min(pageLen, tokens - p * pageLen), 0);
            }
        }
        Check(cudaDeviceSynchronize());
        std::vector<uint8_t> actual(pool.GetBytes()), expected(pool.GetBytes(), 0);
        std::vector<T> dequant(pages * pageElements, T(0));
        const float levels[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
        for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < dim; d += 16) {
                size_t src = (h * tokens + t) * dim + d;
                size_t row = ((t % pageLen) * heads + h) * dim + d;
                int page = indices[t / pageLen];
                float amax = 0;
                for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(float(input[src + j])));
                __nv_fp8_e4m3 sf(amax / 6);
                expected[page * pageBytes + pageElements / 2 + row / 16] = sf.__x;
                const float invScale = float(sf) > 0 ? 1.0f / float(sf) : 0.0f;
                for (int j = 0; j < 16; ++j) {
                    uint8_t code = EncodeReference(float(input[src + j]) * invScale);
                    expected[page * pageBytes + (row + j) / 2] |= code << ((j % 2) * 4);
                    dequant[page * pageElements + row + j] = T(levels[code & 7] * (code & 8 ? -1 : 1) * float(sf));
                }
            }
        }
        Check(cudaMemcpy(actual.data(), pool.cudaData, actual.size(), cudaMemcpyDeviceToHost));
        if (actual != expected) {
            int shown = 0;
            for (size_t i = 0; i < actual.size() && shown < 8; ++i) {
                if (actual[i] != expected[i]) {
                    std::fprintf(stderr, "dtype=%d kv=%d byte=%zu actual=%02x expected=%02x\n",
                                 int(dtype), kv, i, actual[i], expected[i]);
                    ++shown;
                }
            }
        }
        Require(actual == expected, "packed FP4 values/scales differ from CPU reference");
        Check(cudaMemcpy(refPool.cudaData, dequant.data(), refPool.GetBytes(), cudaMemcpyHostToDevice));
    }
    Data q(dtype, {qHeads, queries, dim}), out(dtype, {qHeads, queries, dim});
    AllocateGpu(q); AllocateGpu(out);
    std::vector<T> qHost(q.Count(0));
    for (size_t i = 0; i < qHost.size(); ++i) qHost[i] = T(std::cos(float(i) * 0.013f) * 0.1f);
    Check(cudaMemcpy(q.cudaData, qHost.data(), q.GetBytes(), cudaMemcpyHostToDevice));
    Data qs, lastLens;
    InitInt(qs, {0, queries}); InitInt(lastLens, {(tokens - 1) % pageLen + 1});
    auto attention = [&](PagedCacheManager &kp, PagedCacheManager &vp) {
        Data k(kp.dataType), v(vp.dataType);
        for (auto *cache : {&k, &v}) {
            cache->Resize({heads, tokens, dim});
            cache->isPagedKVCache = true;
            cache->pageLen = pageLen;
        }
        k.pagedKVCacheData = &kp; v.pagedKVCacheData = &vp;
        out.Resize({qHeads, queries, dim});
        Require(FastllmCudaHalfPagedAttentionBatch(q, k, v, qs, pageSizes, pageIds, lastLens,
            out, qHeads / heads, 1 / std::sqrt(float(dim)), 1, false, true), "attention failed");
        Check(cudaDeviceSynchronize());
        std::vector<T> values(out.Count(0));
        Check(cudaMemcpy(values.data(), out.cudaData, out.GetBytes(), cudaMemcpyDeviceToHost));
        return values;
    };
    auto actual = attention(kPool, vPool), reference = attention(kRefPool, vRefPool);
    float maxError = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        Require(std::isfinite(float(actual[i])), "non-finite FP4 attention output");
        maxError = std::max(maxError, std::fabs(float(actual[i]) - float(reference[i])));
    }
    std::printf("dtype=%d dim=%d kv=%d q=%d packed=%d max_error=%g\n",
                int(dtype), dim, tokens, queries, int(packedAppend), maxError);
    Require(maxError < (dtype == DataType::FLOAT16 ? 0.01f : 0.08f), "FP4 attention differs from dequantized reference");
}

template <typename T>
static void RunFusedCase(fastllm::DataType dtype) {
    using namespace fastllm;
    const int batch = 2, dim = 256, qHeads = 8, heads = 2, pageLen = 16;
    const int width = (qHeads + heads) * dim * 2;
    Data input(dtype, {1, batch, width}), norm(DataType::FLOAT32, {dim});
    Data positions(DataType::FLOAT32, {1, batch});
    Data q(dtype, {qHeads, batch, dim}), gate(dtype, {1, batch, qHeads * dim});
    Data indices, offsets, lastLens;
    for (auto *data : {&input, &norm, &positions, &q, &gate}) AllocateGpu(*data);
    InitInt(indices, {1, 0}); InitInt(offsets, {15, 3}); InitInt(lastLens, {0, 0});
    std::vector<T> hostInput(input.Count(0));
    for (size_t i = 0; i < hostInput.size(); ++i) hostInput[i] = T(std::sin(float(i) * 0.11f));
    std::vector<float> weights(dim, 1), hostPositions{31, 7};
    Check(cudaMemcpy(norm.cudaData, weights.data(), norm.GetBytes(), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(positions.cudaData, hostPositions.data(), positions.GetBytes(), cudaMemcpyHostToDevice));
    Data kRef(dtype, {2, pageLen, heads, dim}), vRef(dtype, {2, pageLen, heads, dim});
    Data k(DataType::FP4_E2M1, {2, pageLen, heads, dim}), v(DataType::FP4_E2M1, {2, pageLen, heads, dim});
    Data expected(DataType::FP4_E2M1, {2, pageLen, heads, dim});
    for (auto *data : {&kRef, &vRef, &k, &v, &expected}) {
        AllocateGpu(*data);
        Check(cudaMemset(data->cudaData, 0, data->GetBytes()));
    }
    auto launch = [&](Data &kp, Data &vp) {
        Require(FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache(
            input, norm, norm, positions, q, gate, (uint8_t*)kp.cudaData, (uint8_t*)vp.cudaData,
            (int32_t*)indices.cudaData, (int32_t*)offsets.cudaData, (int32_t*)lastLens.cudaData,
            qHeads, heads, dim, 64, 0, 0, 0, 1e-6f, 10000.0f, 1.0f,
            pageLen, kp.dataType, batch, 1), "fused Qwen3.5 append failed");
    };
    auto download = [&](Data &data) {
        Check(cudaDeviceSynchronize());
        std::vector<uint8_t> bytes(data.GetBytes());
        Check(cudaMemcpy(bytes.data(), data.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
        return bytes;
    };
    Check(cudaMemcpy(input.cudaData, hostInput.data(), input.GetBytes(), cudaMemcpyHostToDevice));
    launch(kRef, vRef);
    auto qRef = download(q), gateRef = download(gate);
    // Exercise graph capture and replay of fused FP4 writes into existing pages.
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    launch(k, v);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    Check(cudaMemcpy(input.cudaData, hostInput.data(), input.GetBytes(), cudaMemcpyHostToDevice));
    Check(cudaGraphLaunch(exec, cudaStreamPerThread));
    Require(download(q) == qRef && download(gate) == gateRef, "FP4 changed Q or gate outputs");
    Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
    for (auto pair : {std::make_pair(&kRef, &k), std::make_pair(&vRef, &v)}) {
        std::vector<T> raw(pair.first->Count(0));
        Check(cudaMemcpy(raw.data(), pair.first->cudaData, pair.first->GetBytes(), cudaMemcpyDeviceToHost));
        std::vector<T> rows(batch * heads * dim);
        const int pages[] = {1, 0}, slots[] = {15, 3};
        for (int b = 0; b < batch; ++b) {
            std::copy_n(raw.data() + (pages[b] * pageLen + slots[b]) * heads * dim,
                        heads * dim, rows.data() + b * heads * dim);
        }
        Data append(dtype, {batch, heads, dim});
        AllocateGpu(append);
        Check(cudaMemcpy(append.cudaData, rows.data(), append.GetBytes(), cudaMemcpyHostToDevice));
        Check(cudaMemset(expected.cudaData, 0, expected.GetBytes()));
        FastllmCudaPagedCacheCopyBatch((uint8_t*)expected.cudaData,
            (int32_t*)indices.cudaData, (int32_t*)offsets.cudaData, pageLen, batch, heads, dim,
            DataType::FP4_E2M1, (uint8_t*)append.cudaData, dtype, true);
        Require(download(expected) == download(*pair.second), "fused FP4 differs from standalone quantization");
    }
    std::printf("fused FP4 append and graph replay passed: dtype=%d\n", int(dtype));
}

int main() {
    try {
#if CUDA_VERSION < 12080
        return 77;
#endif
        Check(cudaSetDevice(0));
        cudaDeviceProp prop;
        Check(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 8 || !FastllmCudaFlashInferSupported()) return 77;
        fastllm::Data empty(fastllm::DataType::FP4_E2M1);
        Require(empty.GetBytes() == 0 && empty.expansionBytes == 0,
                "empty FP4 cache must not reserve storage");
        empty.ToDevice(fastllm::DataDevice::CUDA);
        Require(empty.cudaData == nullptr, "empty FP4 cache allocated CUDA storage");
        RunCopyFormats<float>(fastllm::DataType::FLOAT32);
        RunCopyFormats<half>(fastllm::DataType::FLOAT16);
        RunCopyFormats<__nv_bfloat16>(fastllm::DataType::BFLOAT16);
        std::puts("paged KV copy dispatch passed: 48 non-FP4 combinations");
        for (int dim : {128, 256}) for (int tokens : {5, 65, 1025, 4097}) {
            RunCase<half>(fastllm::DataType::FLOAT16, dim, tokens, 1, false);
            RunCase<__nv_bfloat16>(fastllm::DataType::BFLOAT16, dim, tokens, std::min(tokens, 7), true);
        }
        RunFusedCase<half>(fastllm::DataType::FLOAT16);
        RunFusedCase<__nv_bfloat16>(fastllm::DataType::BFLOAT16);
        std::puts("FP4 KV cache tests passed");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
