#pragma once
#include "utils.h"
#include "devices/cuda/fastllm-cuda-moe-prefill.h"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace fastllm {
void DoCudaMergeMOEFromCPU(Data &, Data &, Data &, Data &, Data &, Data &, Data &,
    Data **, Data **, float, bool, const std::unordered_set<int> &, bool,
    MoeGateType, bool, float, int, bool, bool);
}
namespace fastllm_prefill_test {
inline void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
inline void Check(cudaError_t error) { Require(error == cudaSuccess, cudaGetErrorString(error)); }

// Identical unmodified arithmetic with cache bypassed is the reference. Check
// raw native NUMA bytes too, so a cache conversion cannot hide behind a loose
// end-to-end tolerance. Small cache + multiple layers exercises reservations,
// eviction, overlapping expert sets and cold/warm transitions.
inline void Run(FastllmCudaMoeCacheLayer *layers, std::vector<fastllm::Data *> *tables,
                int count, bool deepSeek, int quantBlock,
                fastllm::DataType activation = fastllm::BFLOAT16) {
    using namespace fastllm;
    const int hidden = tables[0][2]->dims[1], experts = tables[0].size() / 2 - 1, topk = 4;
    uint64_t first[8], last[8];
    Require(fastllm_moe_cuda_cache_prefill_stats(0, first), "initial prefill stats unavailable");
    int comparisons = 0, byteChecks = 0;
    for (int rows : {2, 5, 33, 129, 513}) {
        Data input(activation, {rows, hidden}), ids(INT32, {rows, topk}), scores(FLOAT32, {rows, topk});
        input.Allocate(); ids.Allocate(); scores.Allocate();
        for (int j = 0; j < rows * hidden; ++j) {
            const float value = (j % 37 - 18) / 8.f;
            if (activation == FLOAT32) reinterpret_cast<float *>(input.cpuData)[j] = value;
            else reinterpret_cast<uint16_t *>(input.cpuData)[j] = Float32ToBFloat16RNEBits(value);
        }
        std::unordered_set<int> selected;
        for (int row = 0; row < rows; ++row) for (int k = 0; k < topk; ++k) {
            const int expert = (row * 7 + k + rows) % experts;
            reinterpret_cast<int *>(ids.cpuData)[row * topk + k] = expert;
            reinterpret_cast<float *>(scores.cpuData)[row * topk + k] = k == 0 ? 0.f : (k + 1) / 8.f;
            selected.insert(expert + 1);
        }
        input.ToDevice(CUDA, {0}, true);
        std::vector<std::vector<uint8_t>> reference(count);
        for (int pass = 0; pass < 5; ++pass) {
            if (pass == 0) setenv("FASTLLM_MOE_PREFILL_CACHE", "0", 1);
            else unsetenv("FASTLLM_MOE_PREFILL_CACHE");
            uint64_t before[8], after[8];
            Require(fastllm_moe_cuda_cache_prefill_stats(0, before), "prefill stats before failed");
            for (int table = 0; table < count; ++table) {
                auto **weights = tables[table].data();
                Data output(activation, {rows, hidden}), w1, w2, w3;
                std::vector<Data *> biases(tables[table].size(), nullptr);
                DoCudaMergeMOEFromCPU(input, output, ids, scores, w1, w2, w3,
                    weights, biases.data(), 0, true, selected, true, MoeGateSwiglu,
                    deepSeek, 10, quantBlock, false, pass != 1);
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                std::vector<uint8_t> actual(output.GetBytes());
                Check(cudaMemcpy(actual.data(), output.cudaData, actual.size(), cudaMemcpyDeviceToHost));
                if (pass == 0) reference[table] = actual;
                else { Require(actual == reference[table], "prefill cache changed projection/output bits"); ++comparisons; }
            }
            Require(fastllm_moe_cuda_cache_prefill_stats(0, after), "prefill stats after failed");
            if (pass <= 1) Require(after[7] == before[7], "bypass/assist call admitted experts");
        }
        // Exercise transport directly after warm computation, including its
        // bounded staging allocations and auxiliary-buffer ownership.
        for (int table = 0; table < count; ++table) {
            FastllmCudaMoePrefillCache cache;
            Require(cache.Begin(tables[table].data(), experts,
                reinterpret_cast<int32_t *>(ids.cpuData), reinterpret_cast<float *>(scores.cpuData),
                rows, topk, selected, true), "common prefill transport unavailable");
            auto **weights = cache.Weights();
            for (int expert : selected) {
                for (int part = 0; part < 2; ++part) {
                    const int slot = 2 * expert + part;
                    cache.Upload(slot);
                    Check(cudaStreamSynchronize(cudaStreamPerThread));
                    const auto &source = *tables[table][slot];
                    std::vector<uint8_t> bytes(source.GetBytes());
                    Check(cudaMemcpy(bytes.data(), weights[slot]->cudaData, bytes.size(), cudaMemcpyDeviceToHost));
                    const size_t shardBytes = bytes.size() / source.numasData.size();
                    for (size_t node = 0; node < source.numasData.size(); ++node)
                        Require(std::memcmp(bytes.data() + node * shardBytes, source.numasData[node], shardBytes) == 0,
                            "prefill cached layout differs from original NUMA bytes");
                    ++byteChecks;
                }
                cache.Store(expert - 1);
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                cache.Release(2 * expert); cache.Release(2 * expert + 1);
            }
            cache.Finish();
        }
    }
    unsetenv("FASTLLM_MOE_PREFILL_CACHE");
    // Once decode has established its policy, a long prompt may reuse
    // residents but cannot flush this full cache or reset decode heat.
    void *decode = FastllmCudaBeginMoeDecode(tables[0].data(), tables[0].size(), topk);
    // Generic transport-only fixtures need not register a CPU decode backend;
    // the model-family fixtures must exercise the live decode transition.
    if (deepSeek) Require(decode != nullptr, "decode scope unavailable for warm prefill check");
    if (decode) {
        FastllmCudaEndMoeDecode(decode);
        const int warmRows = 513;
        std::vector<int32_t> warmIds(warmRows * topk);
        std::vector<float> warmScores(warmIds.size(), 1.f);
        std::unordered_set<int> allExperts;
        for (int e = 0; e < experts; ++e) allExperts.insert(e + 1);
        for (size_t r = 0; r < warmIds.size(); ++r) warmIds[r] = r % experts;
        for (int table = 0; table < count; ++table) {
            FastllmCudaMoePrefillPlan plan;
            FastllmCudaPlanMoePrefill(tables[table].data(), experts, warmIds.data(), warmScores.data(),
                warmRows, topk, allExperts, plan);
            Require(plan.keys.empty(), "warm prefill replaced decode cache residents");
        }
        std::puts("WARM_PREFILL_PROTECT_PASS");
    }
    Require(fastllm_moe_cuda_cache_prefill_stats(0, last), "final prefill stats unavailable");
    Require(last[3] > first[3] && last[5] > first[5], "prefill never reused resident payloads");
    Require(last[6] > first[6] && last[7] > first[7], "prefill never uploaded/admitted misses");
    std::printf("COMMON_PREFILL_PASS comparisons=%d byte_checks=%d reused=%llu uploaded=%llu admitted=%llu\n",
        comparisons, byteChecks, (unsigned long long)(last[5]-first[5]),
        (unsigned long long)(last[6]-first[6]), (unsigned long long)(last[7]-first[7]));
}
} // namespace fastllm_prefill_test
