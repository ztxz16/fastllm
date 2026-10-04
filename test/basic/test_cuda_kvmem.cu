#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#include "fastllm.h"
#include "kvmem.h"
#include "models/qwen3.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace fastllm;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
#define CUDA_OK(x) CHECK((x) == cudaSuccess)

template<class T> void Upload(Data &data, const std::vector<int> &shape, const std::vector<T> &values) {
    data.Resize(shape);
    data.Allocate();
    std::memcpy(data.cpuData, values.data(), values.size() * sizeof(T));
    data.ToDevice(DataDevice::CUDA, {0}, true);
}

void Ints(Data &data, const std::vector<int> &values) {
    data.cpuIntDatas = values;
    Upload(data, {(int)values.size()}, values);
}

template<class T> std::vector<T> Read(const Data &data) {
    std::vector<T> out(data.Count(0));
    CUDA_OK(cudaMemcpy(out.data(), data.cudaData, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return out;
}

// A model adapter can reject a candidate configuration before any state changes.
void CheckModelConfiguration() {
    class ModelWithChunkLimit final : public Qwen3Model {
    public:
        bool CanUseGPUForward() const override { return true; }
        void ValidateKvMemModel(const KvMemConfig &config) const override {
            if (config.prefillTokens < 256) {
                throw std::invalid_argument("adapter requires prefill_tokens >= 256");
            }
        }
    } model;
    model.deviceMap = {{"cuda:0", 1}};
    model.dataType = model.kvCacheDataType = FLOAT16;
    model.basellm::head_dim = 128;
    model.block_cnt = 1;
    model.max_positions = 2048;
    model.maxBatch = 1;
    model.saveHistoryChat = true;
    KvMemConfig config;
    config.enabled = true;
    config.maxTokens = 2048;
    config.prefillTokens = 128;
    bool rejected = false;
    try {
        model.ConfigureKvMem(config);
    } catch (const std::invalid_argument &error) {
        rejected = std::string(error.what()) == "adapter requires prefill_tokens >= 256";
    }
    CHECK(rejected && !model.kvMemConfig.enabled && model.saveHistoryChat);
    config.prefillTokens = 256;
    model.ConfigureKvMem(config);
    CHECK(model.kvMemConfig.prefillTokens == 256 && !model.saveHistoryChat);
    printf("PASS KVMem model configuration validation\n");
}

template<class T> void Run(DataType type, int group, int dim) {
    constexpr int pageLen = 16, kvHeads = 2, total = 417;
    int heads = kvHeads * group;
    auto config = std::make_shared<KvMemConfig>();
    config->enabled = true;
    config->maxTokens = 512;
    config->residentTokens = 160;
    config->sinkTokens = config->recentTokens = 16;
    config->retrievalTokens = 32;
    config->prefillTokens = 64;
    Data kc(type), vc(type);
    kc.kvMemConfig = vc.kvMemConfig = config;
    auto key = [&](int h, int t, int d) -> T {
        return T(d == 0 ? float(t / pageLen) : d == 1 ? float(t % pageLen) :
                 float((h * 7 + t * 11 + d * 3) % 67 - 33) / 100);
    };
    auto value = [&](int h, int t, int d) -> T { return T(float((h * 13 + t * 7 + d * 11) % 97 - 48) / 64); };
    auto query = [&](int h, int t, int d) -> T { return T(d < 2 ? 0.0f : float((h * 7 + t * 5 + d * 13) % 31 - 15) / 40); };
    std::vector<int> chunks{1, 3, 16, 31, 64, 2, 17};
    int iteration = 0;
    for (int old = 0; old < total;) {
        int n = std::min(total - old, chunks[iteration % chunks.size()]);
        Data q(type), k(type), v(type), rawQ(type), rawK(type), output(type);
        std::vector<T> qh((size_t)heads * n * dim), kh((size_t)kvHeads * n * dim), vh(kh.size());
        std::vector<T> qr(qh.size(), T(0.0f)), kr(kh.size(), T(0.0f));
        int targetPage = iteration % 2 ? 1 : 3;
        for (int t = 0; t < n; ++t) {
            for (int h = 0; h < heads; ++h) {
                qr[((size_t)t * heads + h) * dim + targetPage] = T(4.0f);
                // Later query rows prefer another page. Selection must only
                // inspect the first row and already committed keys.
                if (t > 0) qr[((size_t)t * heads + h) * dim + 5] = T(100.0f);
                for (int d = 0; d < dim; ++d) qh[((size_t)h * n + t) * dim + d] = query(h, old + t, d);
            }
            for (int h = 0; h < kvHeads; ++h) {
                kr[((size_t)t * kvHeads + h) * dim + (old + t) / pageLen] = T(4.0f);
                for (int d = 0; d < dim; ++d) {
                    kh[((size_t)h * n + t) * dim + d] = key(h, old + t, d);
                    vh[((size_t)h * n + t) * dim + d] = value(h, old + t, d);
                }
            }
        }
        Upload(q, {heads, n, dim}, qh); Upload(k, {kvHeads, n, dim}, kh); Upload(v, {kvHeads, n, dim}, vh);
        Upload(rawQ, {1, n, heads, dim}, qr); Upload(rawK, {1, n, kvHeads, dim}, kr);
        KvMemAppend(rawQ, rawK, k, v, kc, vc);
        CHECK(kc.dims[1] == old + n && kc.pageIndex == vc.pageIndex);
        auto poolK = Read<T>(*kc.pagedKVCacheData), poolV = Read<T>(*vc.pagedKVCacheData);
        std::vector<int> logicalPages, positions;
        for (int slot : kc.pageIndex) {
            int logical = (int)(float)poolK[(size_t)slot * pageLen * kvHeads * dim];
            logicalPages.push_back(logical);
            for (int t = logical * pageLen; t < std::min(old + n, (logical + 1) * pageLen); ++t) {
                positions.push_back(t);
                for (int h = 0; h < kvHeads; ++h) for (int d = 0; d < dim; ++d) {
                    size_t index = (((size_t)slot * pageLen + t % pageLen) * kvHeads + h) * dim + d;
                    CHECK((float)poolK[index] == (float)key(h, t, d));
                    CHECK((float)poolV[index] == (float)value(h, t, d));
                }
            }
        }
        CHECK(std::is_sorted(logicalPages.begin(), logicalPages.end()));
        if (old + n > config->residentTokens && n > 1)
            CHECK(std::find(logicalPages.begin(), logicalPages.end(), targetPage) != logicalPages.end());
        Data qs(INT32), ps(INT32), pi(INT32), last(INT32);
        Ints(qs, {0, n}); Ints(ps, {0, (int)kc.pageIndex.size()});
        Ints(pi, kc.pageIndex); Ints(last, {kc.lastPageLen});
        output.Resize({heads, n, dim}); output.dataDevice = DataDevice::CUDA; output.dataDeviceIds = {0}; output.Allocate();
        CHECK(FastllmCudaHalfPagedAttentionBatch(q, kc, vc, qs, ps, pi, last, output,
                                                group, 1.0f / std::sqrt((float)dim), 1));
        CUDA_OK(cudaDeviceSynchronize());
        auto actual = Read<T>(output);
        for (int row : {0, n - 1}) for (int h = 0; h < heads; ++h) {
            std::vector<double> weights;
            double sum = 0;
            for (int t : positions) {
                if (t > old + row) break;
                double dot = 0;
                for (int d = 0; d < dim; ++d) dot += (float)query(h, old + row, d) * (float)key(h / group, t, d);
                double w = std::exp(dot / std::sqrt((double)dim));
                weights.push_back(w); sum += w;
            }
            for (int d = 0; d < dim; ++d) {
                double expected = 0;
                for (size_t t = 0; t < weights.size(); ++t) expected += weights[t] * (float)value(h / group, positions[t], d);
                float got = (float)actual[((size_t)row * heads + h) * dim + d];
                CHECK(std::isfinite(got) && std::abs(got - expected / sum) < (type == BFLOAT16 ? .006 : .001));
            }
        }
        old += n; ++iteration;
    }
    CHECK(kc.kvMemCache->Stats().evictedPages > 0 && kc.kvMemCache->Stats().restoredPages > 0);
    bool rejected = false;
    try { Data copy(kc); } catch (const std::exception&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { Data view; view.FakeFrom(kc, 0); } catch (const std::exception&) { rejected = true; }
    CHECK(rejected);
    printf("PASS KVMem dtype=%d group=%d dim=%d evicted=%llu restored=%llu\n", (int)type, group, dim,
           (unsigned long long)kc.kvMemCache->Stats().evictedPages, (unsigned long long)kc.kvMemCache->Stats().restoredPages);
    // Key/value views share ownership: releasing either view must leave the
    // other valid, and releasing both must free the private pool exactly once.
    std::weak_ptr<KvMemCache> owner = kc.kvMemCache;
    CHECK(ReleaseKvMemCache(kc) && !owner.expired());
    CHECK(!kc.pagedKVCacheData && kc.pageIndex.empty());
    CHECK(vc.kvMemCache->Stats().tokens == total);
    CHECK(!ReleaseKvMemCache(kc));
    CHECK(ReleaseKvMemCache(vc) && owner.expired());
}

// Compare a speculative cache against an append-only reference. Rejected raw
// keys are deliberately huge: indexing even one would change later retrieval.
template<class T> void RunTransactions(DataType type, int group, int dim,
                                      int interval = KvMemConfig().retrievalInterval, int pageLen = 16) {
    constexpr int kvHeads = 2;
    auto config = std::make_shared<KvMemConfig>();
    config->enabled = true;
    config->maxTokens = 64 * pageLen;
    config->residentTokens = 10 * pageLen;
    config->sinkTokens = config->recentTokens = pageLen;
    config->retrievalTokens = 2 * pageLen;
    config->prefillTokens = 4 * pageLen;
    config->retrievalInterval = interval;
    Data key(type), value(type), refKey(type), refValue(type);
    key.kvMemConfig = value.kvMemConfig = refKey.kvMemConfig = refValue.kvMemConfig = config;
    auto append = [&](Data &kc, Data &vc, int n, int keep, int preferred) {
        const int old = kc.kvMemCache ? kc.kvMemCache->Stats().tokens : 0;
        const int heads = kvHeads * group;
        Data rq(type), rk(type), k(type), v(type);
        std::vector<T> qr((size_t)n * heads * dim, T(0.f));
        std::vector<T> kr((size_t)n * kvHeads * dim, T(0.f)), kh(kr.size()), vh(kr.size());
        for (int t = 0; t < n; ++t) {
            for (int h = 0; h < heads; ++h)
                qr[((size_t)t * heads + h) * dim + preferred] = T(4.f);
            for (int h = 0; h < kvHeads; ++h) for (int d = 0; d < dim; ++d) {
                kr[((size_t)t * kvHeads + h) * dim + d] = T(t < keep ? (d == (old + t) / pageLen ? 4.f : 0.f) : 1000.f);
                const size_t i = ((size_t)h * n + t) * dim + d;
                kh[i] = T(t < keep ? (d == 0 ? float((old + t) / pageLen) :
                    float(((old + t) * 7 + d + h) % 31) / 32) : 999.f);
                vh[i] = T(t < keep ? float(((old + t) + d * 3 + h) % 37) / 32 : 999.f);
            }
        }
        Upload(rq, {1, n, heads, dim}, qr); Upload(rk, {1, n, kvHeads, dim}, kr);
        Upload(k, {kvHeads, n, dim}, kh); Upload(v, {kvHeads, n, dim}, vh);
        KvMemAppend(rq, rk, k, v, kc, vc);
    };
    for (int i = 0; i < 4; ++i) {
        append(key, value, 4 * pageLen, 4 * pageLen, 1);
        append(refKey, refValue, 4 * pageLen, 4 * pageLen, 1);
    }
    auto checkSelection = [&](int expected) {
        auto pool = Read<T>(*key.pagedKVCacheData);
        bool found = false;
        for (int slot : key.pageIndex) {
            const int page = (int)(float)pool[(size_t)slot * pageLen * kvHeads * dim];
            if (page == expected) found = true;
            CHECK(page != (expected == 3 ? 5 : 3));
        }
        CHECK(found);
    };
    // Query changes within an interval reuse scores. Accepted positions, not
    // provisional rows, determine the next refresh boundary.
    auto verify = [&](int keep, int preferred, int expected) {
        const int old = key.kvMemCache->Stats().tokens;
        key.kvMemCache->BeginTransaction(8);
        append(key, value, 8, keep, preferred);
        checkSelection(expected);
        key.kvMemCache->FinishTransaction(keep, key, value);
        if (keep) append(refKey, refValue, keep, keep, preferred);
        CHECK(key.kvMemCache->Stats().tokens == old + keep);
    };
    verify(3, 3, 3);
    if (interval == 1) {
        verify(7, 5, 5); // Explicit per-round mode follows each changed query.
        verify(8, 3, 3);
        verify(0, 5, 5);
        verify(1, 3, 3);
    } else {
        const int boundary = (16 * pageLen / interval + 1) * interval;
        while (key.kvMemCache->Stats().tokens < boundary - 2) {
            int keep = std::min(7, boundary - 2 - (int)key.kvMemCache->Stats().tokens);
            verify(keep, 5, 3);
        }
        verify(0, 5, 3); // Rejected lookahead crosses the boundary; position stays.
        verify(2, 3, 3); // Cancellation invalidated scores, then commit to boundary.
        verify(3, 5, 5); // Committed boundary refreshes even inside a physical page.
        verify(0, 3, 5);
        verify(1, 3, 3); // Same-position retry must discard the cancelled query.
    }
    // Switching back to ordinary decode must not reuse a speculative bucket.
    if (interval != pageLen) {
        append(key, value, 1, 1, 5);
        append(refKey, refValue, 1, 1, 5);
        checkSelection(5);
    }
    for (int keep : {0, 1, 3, 7, 12, 16, 31, 64, 0, 3, 7, 16, 31}) {
        const int old = key.kvMemCache->Stats().tokens;
        const int preferred = (old / 7) % (old / pageLen);
        key.kvMemCache->BeginTransaction(64);
        append(key, value, 64, keep, preferred);
        key.kvMemCache->FinishTransaction(keep, key, value);
        if (keep) append(refKey, refValue, keep, keep, preferred);
        CHECK(key.dims[1] == old + keep && key.kvMemCache->Stats().tokens == old + keep);
        CHECK(key.pageIndex == value.pageIndex && key.lastPageLen == (old + keep - 1) % pageLen + 1);
        // Force fresh selection against the committed index, then compare all
        // visible KV bytes (physical slot assignments may legitimately differ).
        append(key, value, 2, 2, preferred);
        append(refKey, refValue, 2, 2, preferred);
        auto k = Read<T>(*key.pagedKVCacheData), v = Read<T>(*value.pagedKVCacheData);
        auto rk = Read<T>(*refKey.pagedKVCacheData), rv = Read<T>(*refValue.pagedKVCacheData);
        CHECK(key.pageIndex.size() == refKey.pageIndex.size());
        for (size_t p = 0; p < key.pageIndex.size(); ++p) {
            const size_t a = (size_t)key.pageIndex[p] * pageLen * kvHeads * dim;
            const size_t b = (size_t)refKey.pageIndex[p] * pageLen * kvHeads * dim;
            CHECK((float)k[a] == (float)rk[b]);
            const int n = p + 1 == key.pageIndex.size() ? key.lastPageLen : pageLen;
            for (int j = 0; j < n * kvHeads * dim; ++j) {
                CHECK((float)k[a + j] == (float)rk[b + j]);
                CHECK((float)v[a + j] == (float)rv[b + j]);
            }
        }
        key.kvMemCache->BeginTransaction(1);
        key.kvMemCache->FinishTransaction(0, key, value);
    }
    CHECK(key.kvMemCache->Stats().evictedPages && key.kvMemCache->Stats().restoredPages);
    printf("PASS KVMem transactions dtype=%d group=%d dim=%d interval=%d page=%d\n",
           (int)type, group, dim, interval, pageLen);
}

// Alternate two disjoint 64-page selections (32 MiB each) to exercise large
// transfer batches, cold backing lifetimes, and reused physical slots.
template<class T> void RunLargeTransfers(DataType type) {
    constexpr int pageLen = 128, kvHeads = 4, heads = 8, dim = 256;
    auto config = std::make_shared<KvMemConfig>();
    config->enabled = true;
    config->maxTokens = 32768;
    config->residentTokens = 12288;
    config->sinkTokens = config->recentTokens = pageLen;
    config->retrievalTokens = 8192;
    config->prefillTokens = 512;
    Data key(type), value(type);
    key.kvMemConfig = value.kvMemConfig = config;
    auto expectedK = [](int h, int t, int d) -> T {
        return T(d == 0 ? float(t / pageLen) : float((h * 7 + t * 11 + d * 3) % 67 - 33) / 64);
    };
    auto expectedV = [](int h, int t, int d) -> T { return T(float((h * 13 + t * 7 + d * 11) % 97 - 48) / 64); };
    int old = 0;
    auto append = [&](int n, int preferred) {
        Data rq(type), rk(type), k(type), v(type);
        std::vector<T> qr((size_t)n * heads * dim, T(0.f));
        std::vector<T> kr((size_t)n * kvHeads * dim, T(0.f)), kh(kr.size()), vh(kr.size());
        for (int h = 0; h < heads; ++h)
            for (int p = preferred; p < preferred + 64; ++p) qr[(size_t)h * dim + p] = T(4.f);
        for (int t = 0; t < n; ++t) for (int h = 0; h < kvHeads; ++h) {
            kr[((size_t)t * kvHeads + h) * dim + (old + t) / pageLen] = T(4.f);
            for (int d = 0; d < dim; ++d) {
                const size_t i = ((size_t)h * n + t) * dim + d;
                kh[i] = expectedK(h, old + t, d);
                vh[i] = expectedV(h, old + t, d);
            }
        }
        Upload(rq, {1, n, heads, dim}, qr); Upload(rk, {1, n, kvHeads, dim}, kr);
        Upload(k, {kvHeads, n, dim}, kh); Upload(v, {kvHeads, n, dim}, vh);
        KvMemAppend(rq, rk, k, v, key, value);
        old += n;
    };
    while (old < 24576) append(512, 1);
    int round = 0;
    for (int preferred : {65, 1, 65, 1}) {
        const auto before = key.kvMemCache->Stats().restoredPages;
        append(8, preferred);
        const auto restored = key.kvMemCache->Stats().restoredPages - before;
        CHECK(restored > 0 && restored <= 64);
        CHECK(round++ == 0 ? restored == 64 : restored < 64);
        auto k = Read<T>(*key.pagedKVCacheData), v = Read<T>(*value.pagedKVCacheData);
        for (int slot : key.pageIndex) {
            const size_t offset = (size_t)slot * pageLen * kvHeads * dim;
            const int page = (int)(float)k[offset];
            for (int t = page * pageLen; t < std::min(old, (page + 1) * pageLen); ++t)
                for (int h = 0; h < kvHeads; ++h) for (int d = 0; d < dim; ++d) {
                    const size_t i = offset + ((size_t)(t % pageLen) * kvHeads + h) * dim + d;
                    CHECK((float)k[i] == (float)expectedK(h, t, d));
                    CHECK((float)v[i] == (float)expectedV(h, t, d));
                }
        }
    }
    printf("PASS KVMem batched 32 MiB restore dtype=%d\n", (int)type);
}

int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        CUDA_OK(cudaSetDevice(0));
        SetPageLen(128);
        CheckModelConfiguration();
        SetPageLen(16);
        for (int group : {1, 4}) for (int dim : {128, 256}) {
            Run<half>(FLOAT16, group, dim);
            Run<__nv_bfloat16>(BFLOAT16, group, dim);
            RunTransactions<half>(FLOAT16, group, dim);
            RunTransactions<__nv_bfloat16>(BFLOAT16, group, dim);
        }
        SetPageLen(128);
        for (int interval : {1, 32, 64}) {
            RunTransactions<half>(FLOAT16, 4, 128, interval, 128);
            RunTransactions<__nv_bfloat16>(BFLOAT16, 4, 128, interval, 128);
        }
        RunLargeTransfers<half>(FLOAT16);
        RunLargeTransfers<__nv_bfloat16>(BFLOAT16);
    } catch (const std::exception &error) {
        fprintf(stderr, "FAIL KVMem: %s\n", error.what()); return 1;
    }
    printf("PASS KVMem all 8 cases\n");
    return 0;
}
