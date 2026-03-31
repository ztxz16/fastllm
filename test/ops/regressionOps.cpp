#include "executor.h"
#include "fastllm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
    class ScopedFirstDevice {
    public:
        explicit ScopedFirstDevice(const std::string &device) {
            executor = (fastllm::Executor*) fastllm::GetExecutor();
            previous = executor->firstDevice;
            executor->SetFirstDevice(device);
        }

        ~ScopedFirstDevice() {
            if (!previous.empty()) {
                executor->SetFirstDevice(previous);
            }
        }

    private:
        fastllm::Executor *executor = nullptr;
        std::string previous;
    };

    void Expect(bool condition, const std::string &message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    fastllm::Data MakeTensor(fastllm::DataType dataType, const std::vector<int> &dims, float seed = 0.0f) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        std::vector<float> values(count);
        for (int i = 0; i < count; i++) {
            values[i] = std::sin((i + 1) * 0.17f + seed) + std::cos((i + 3) * 0.11f + seed * 0.5f);
        }
        return fastllm::Data(dataType, dims, values);
    }

    fastllm::Data MakeFloatTensor(const std::vector<int> &dims, float seed = 0.0f) {
        return MakeTensor(fastllm::DataType::FLOAT32, dims, seed);
    }

    fastllm::Data MakeIntTensor(const std::vector<int> &dims, const std::vector<int32_t> &values) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        Expect(count == (int) values.size(), "INT32 tensor element count mismatch.");
        fastllm::Data data(fastllm::DataType::INT32, dims);
        data.Allocate();
        if (count > 0) {
            std::memcpy(data.cpuData, values.data(), (size_t) count * sizeof(int32_t));
        }
        return data;
    }

    std::vector<int32_t> ToIntVector(fastllm::Data data, int logicalCount = -1) {
        data.ToDevice(fastllm::DataDevice::CPU);
        int count = logicalCount >= 0 ? logicalCount : (int) data.Count(0);
        std::vector<int32_t> values(count);
        if (count > 0) {
            Expect(data.cpuData != nullptr, "INT32 tensor has no CPU buffer.");
            std::memcpy(values.data(), data.cpuData, (size_t) count * sizeof(int32_t));
        }
        return values;
    }

    std::vector<float> ToFloatVector(fastllm::Data data) {
        data.ToDevice(fastllm::DataDevice::CPU);
        int count = (int) data.Count(0);
        std::vector<float> values(count);
        Expect(data.dataType == fastllm::DataType::FLOAT32, "Only FLOAT32 tensors are supported here.");
        if (count > 0) {
            Expect(data.cpuData != nullptr, "FLOAT32 tensor has no CPU buffer.");
            std::memcpy(values.data(), data.cpuData, (size_t) count * sizeof(float));
        }
        return values;
    }

    void ExpectIntEqual(const std::vector<int32_t> &expected, const std::vector<int32_t> &actual,
                        const std::string &name) {
        Expect(expected.size() == actual.size(), name + " size mismatch.");
        for (size_t i = 0; i < expected.size(); i++) {
            if (expected[i] != actual[i]) {
                throw std::runtime_error(name + " mismatch at index " + std::to_string(i) +
                                         ": expected " + std::to_string(expected[i]) +
                                         ", got " + std::to_string(actual[i]));
            }
        }
    }

    void ExpectFloatNear(const std::vector<float> &expected, const std::vector<float> &actual,
                         float atol, float rtol, const std::string &name) {
        Expect(expected.size() == actual.size(), name + " size mismatch.");
        for (size_t i = 0; i < expected.size(); i++) {
            float diff = std::fabs(expected[i] - actual[i]);
            float limit = atol + rtol * std::fabs(expected[i]);
            if (diff > limit) {
                throw std::runtime_error(name + " mismatch at index " + std::to_string(i) +
                                         ": expected " + std::to_string(expected[i]) +
                                         ", got " + std::to_string(actual[i]));
            }
        }
    }

    struct PastKeyBatch {
        std::vector<fastllm::Data> keys;
        std::vector<fastllm::Data*> keyPtrs;
        std::vector<int> seqLens;
        int totalPages = 0;
        int totalSeq = 0;
    };

    PastKeyBatch BuildPastKeysForPagedRegression(int batch, int pageLen, fastllm::PagedCacheManager *manager) {
        PastKeyBatch result;
        result.keys.reserve(batch);
        result.keyPtrs.reserve(batch);
        result.seqLens.reserve(batch);

        for (int b = 0; b < batch; b++) {
            result.keys.emplace_back();
            fastllm::Data &key = result.keys.back();
            key.isKVCache = true;
            key.isPagedKVCache = true;
            key.pageLen = pageLen;
            key.pagedKVCacheData = manager;

            int mode = b % 4;
            int pageCount = mode;
            if (pageCount > 0) {
                key.pageIndex.reserve(pageCount);
                for (int i = 0; i < pageCount; i++) {
                    key.pageIndex.push_back(manager->GetUnusedPageIndex(true));
                }
            }
            if (pageCount == 0) {
                key.lastPageLen = 0;
            } else if (mode == 1) {
                key.lastPageLen = pageLen / 2;
            } else if (mode == 2) {
                key.lastPageLen = pageLen;
            } else {
                key.lastPageLen = pageLen - 3;
            }

            result.totalPages += pageCount;
            int seqLen = 1 + (b % 5);
            result.seqLens.push_back(seqLen);
            result.totalSeq += seqLen;

            result.keyPtrs.push_back(&key);
        }

        return result;
    }

    fastllm::PagedCacheManager* CreateManager(int layerIndex, int pageLen, int maxPages) {
        fastllm::Data cache = MakeFloatTensor({4, 1, 8}, 0.2f);
        return fastllm::AllocatePagedCacheManager(
            layerIndex,
            fastllm::PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
            cache,
            pageLen,
            maxPages
        );
    }

    void RunGenerateAppendPagedCacheBatchParams(const std::string &device, int batch) {
        const int pageLen = 128;
        fastllm::ClearAllPagedCacheManagers();
        {
            fastllm::PagedCacheManager *cpuManager = CreateManager(0, pageLen, batch * 4);
            fastllm::PagedCacheManager *deviceManager = CreateManager(1, pageLen, batch * 4);

            PastKeyBatch cpuPast = BuildPastKeysForPagedRegression(batch, pageLen, cpuManager);
            PastKeyBatch devicePast = BuildPastKeysForPagedRegression(batch, pageLen, deviceManager);

            fastllm::Data cpuInsertIndexs, cpuInsertPositions;
            {
                ScopedFirstDevice guard("cpu");
                fastllm::GenerateAppendPagedCacheBatchParams(
                    *cpuManager, cpuPast.keyPtrs, batch, cpuInsertIndexs, cpuInsertPositions);
            }

            fastllm::Data deviceInsertIndexs, deviceInsertPositions;
            {
                ScopedFirstDevice guard(device);
                fastllm::GenerateAppendPagedCacheBatchParams(
                    *deviceManager, devicePast.keyPtrs, batch, deviceInsertIndexs, deviceInsertPositions);
            }

            ExpectIntEqual(ToIntVector(cpuInsertIndexs), ToIntVector(deviceInsertIndexs), "insertIndexs");
            ExpectIntEqual(ToIntVector(cpuInsertPositions), ToIntVector(deviceInsertPositions), "insertPositions");
        }
        fastllm::ClearAllPagedCacheManagers();
    }

    void RunGeneratePagedBatchParams(const std::string &device, int batch, bool zeroPages) {
        const int pageLen = 128;
        fastllm::ClearAllPagedCacheManagers();
        {
            fastllm::PagedCacheManager *manager = CreateManager(2, pageLen, std::max(batch * 4, 16));
            PastKeyBatch past = zeroPages ? PastKeyBatch() : BuildPastKeysForPagedRegression(batch, pageLen, manager);
            if (zeroPages) {
                past.keys.reserve(batch);
                past.keyPtrs.reserve(batch);
                past.seqLens.reserve(batch);
                for (int b = 0; b < batch; b++) {
                    past.keys.emplace_back();
                    fastllm::Data &key = past.keys.back();
                    key.isKVCache = true;
                    key.isPagedKVCache = true;
                    key.pageLen = pageLen;
                    key.pagedKVCacheData = manager;
                    key.lastPageLen = 0;
                    past.keyPtrs.push_back(&key);
                    int seqLen = 1 + (b % 3);
                    past.seqLens.push_back(seqLen);
                    past.totalSeq += seqLen;
                }
                past.totalPages = 0;
            }

            fastllm::Data q = MakeFloatTensor({4, past.totalSeq, 8}, 0.3f);

            fastllm::Data cpuQSizes, cpuPageSizes, cpuPageIndexs, cpuLastPageLens;
            {
                ScopedFirstDevice guard("cpu");
                fastllm::GeneratePagedBatchParams(
                    q, past.keyPtrs, batch, cpuQSizes, cpuPageSizes, cpuPageIndexs, cpuLastPageLens, past.seqLens);
            }

            fastllm::Data deviceQSizes, devicePageSizes, devicePageIndexs, deviceLastPageLens;
            {
                ScopedFirstDevice guard(device);
                fastllm::GeneratePagedBatchParams(
                    q, past.keyPtrs, batch, deviceQSizes, devicePageSizes, devicePageIndexs, deviceLastPageLens, past.seqLens);
            }

            std::vector<int32_t> cpuPageSizesVec = ToIntVector(cpuPageSizes);
            std::vector<int32_t> devicePageSizesVec = ToIntVector(devicePageSizes);
            int logicalPages = cpuPageSizesVec.empty() ? 0 : cpuPageSizesVec.back();

            ExpectIntEqual(ToIntVector(cpuQSizes), ToIntVector(deviceQSizes), "qSizes");
            ExpectIntEqual(cpuPageSizesVec, devicePageSizesVec, "pageSizes");
            ExpectIntEqual(ToIntVector(cpuLastPageLens), ToIntVector(deviceLastPageLens), "lastPageLens");
            ExpectIntEqual(
                ToIntVector(cpuPageIndexs, logicalPages),
                ToIntVector(devicePageIndexs, logicalPages),
                "pageIndexs"
            );
            Expect(devicePageIndexs.dims.empty() || devicePageIndexs.dims[0] >= logicalPages,
                   "device pageIndexs shape is smaller than the logical page count.");
        }
        fastllm::ClearAllPagedCacheManagers();
    }

    struct MoeWeights {
        fastllm::Data routedGate;
        fastllm::Data routedDown;
    };

    MoeWeights MakeMoeWeights(int inputDim, int interDim, int outputDim, float seed) {
        MoeWeights weights {
            MakeTensor(fastllm::DataType::FLOAT16, {interDim * 2, inputDim}, seed),
            MakeTensor(fastllm::DataType::FLOAT16, {outputDim, interDim}, seed + 1.0f)
        };
        weights.routedGate.name = "test.routed_gate";
        weights.routedDown.name = "test.routed_down";
        return weights;
    }

    std::vector<float> RunMergeMoeOnDevice(const std::string &device, MoeWeights &weights) {
        const int batch = 32;
        const int inputDim = 64;
        const int outputDim = 64;

        fastllm::Data input = MakeFloatTensor({batch, inputDim}, 0.7f);
        fastllm::Data output(fastllm::DataType::FLOAT32, {batch, outputDim});
        fastllm::Data index = MakeIntTensor({batch, 1}, std::vector<int32_t>(batch, 0));
        fastllm::Data score(fastllm::DataType::FLOAT32, {batch, 1}, std::vector<float>(batch, 1.0f));
        fastllm::Data w1, w2, w3, curInput, curOutput;

        std::vector<fastllm::Data*> weightPtrs = {
            nullptr, nullptr, &weights.routedGate, &weights.routedDown
        };
        std::vector<fastllm::Data*> biasPtrs(4, nullptr);

        {
            ScopedFirstDevice guard(device);
            fastllm::MergeMOE(
                input, index, score, weightPtrs, biasPtrs,
                w1, w2, w3, curInput, curOutput,
                0.0f, output, 0
            );
        }

        return ToFloatVector(output);
    }

    void RunNumasMergeMoeRegression() {
        const int inputDim = 64;
        const int interDim = 128;
        const int outputDim = 64;

        MoeWeights cpuWeights = MakeMoeWeights(inputDim, interDim, outputDim, 1.1f);
        MoeWeights numasWeights = MakeMoeWeights(inputDim, interDim, outputDim, 1.1f);

        std::vector<float> expected = RunMergeMoeOnDevice("cpu", cpuWeights);
        std::vector<float> actual = RunMergeMoeOnDevice("numa", numasWeights);

        ExpectFloatNear(expected, actual, 1e-3f, 1e-4f, "numas MergeMOE output");
        Expect(!numasWeights.routedGate.numasData.empty(), "routed gate weight was not registered to NUMA shards.");
        Expect(!numasWeights.routedDown.numasData.empty(), "routed down weight was not registered to NUMA shards.");
        Expect(numasWeights.routedGate.cpuData == nullptr, "routed gate CPU buffer should be released after NUMA registration.");
        Expect(numasWeights.routedDown.cpuData == nullptr, "routed down CPU buffer should be released after NUMA registration.");
    }
}

int main() {
    try {
        bool ranAny = false;

        if (fastllm::HasDeviceType("cuda")) {
            RunGenerateAppendPagedCacheBatchParams("cuda:0", 1536);
            RunGeneratePagedBatchParams("cuda:0", 1536, false);
            RunGeneratePagedBatchParams("cuda:0", 64, true);
            std::cout << "cuda paged-batch regressions: PASS\n";
            ranAny = true;
        } else {
            std::cout << "cuda paged-batch regressions: SKIP (cuda unavailable)\n";
        }

        if (fastllm::HasDeviceType("numa") && !fastllm::GetFastllmEnv().activateNuma) {
            RunNumasMergeMoeRegression();
            std::cout << "numa MergeMOE regression: PASS\n";
            ranAny = true;
        } else if (fastllm::HasDeviceType("numa")) {
            std::cout << "numa MergeMOE regression: SKIP (legacy numa device is active)\n";
        } else {
            std::cout << "numa MergeMOE regression: SKIP (numa unavailable)\n";
        }

        if (!ranAny) {
            std::cout << "no matching regression device paths available\n";
        }
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "regressionOps failed: " << ex.what() << "\n";
    } catch (...) {
        std::cerr << "regressionOps failed: unknown error\n";
    }
    return 1;
}
