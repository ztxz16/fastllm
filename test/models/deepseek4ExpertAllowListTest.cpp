#include "models/deepseekv4.h"
#include "executor.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    using fastllm::DeepSeekV4ExpertAllowList;
    using fastllm::DeepSeekV4HashRemapResult;

    class ScopedFirstDevice {
    public:
        explicit ScopedFirstDevice(const std::string &device) {
            executor = (fastllm::Executor*)fastllm::GetExecutor();
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

    class TestableDeepSeekV4Model : public fastllm::DeepSeekV4Model {
    public:
        void SetExpertPolicies(
                int expertCount,
                const DeepSeekV4ExpertAllowList &backbone,
                const DeepSeekV4ExpertAllowList &mtp) {
            num_experts = expertCount;
            backboneExpertAllowList = backbone;
            mtpExpertAllowList = mtp;
        }

        const DeepSeekV4ExpertAllowList &PolicyFor(
                const std::string &prefix) const {
            return GetExpertAllowListForPrefix(prefix);
        }

        void SetHashCoverage(const std::string &prefix,
                             uint64_t entries, uint64_t remapped) {
            hashRemapCoverage[prefix] = std::make_pair(entries, remapped);
            UpdateExpertAllowListMetadata();
        }

        const std::map<std::string, std::string> &Metadata() const {
            return weight.dicts;
        }
    };

    void ExpectFailure(const std::function<void()> &operation,
                       const std::string &messagePart) {
        std::string message;
        try {
            operation();
        } catch (const std::runtime_error &error) {
            message = error.what();
        }
        assert(!message.empty());
        assert(message.find(messagePart) != std::string::npos);
    }

    void AssertUniqueAllowedRows(const std::vector<int> &routes, int topk,
                                 const DeepSeekV4ExpertAllowList &policy) {
        assert(routes.size() % (size_t)topk == 0);
        for (size_t row = 0; row < routes.size(); row += topk) {
            std::set<int> unique;
            for (int slot = 0; slot < topk; ++slot) {
                const int expert = routes[row + slot];
                assert(policy.Allows(expert));
                unique.insert(expert);
            }
            assert((int)unique.size() == topk);
        }
    }

    void TestStrictParsingAndCanonicalIds() {
        const DeepSeekV4ExpertAllowList unrestricted =
            fastllm::ParseDeepSeekV4ExpertAllowList("", 12, 6, "backbone");
        assert(!unrestricted.configured);
        assert(unrestricted.experts.size() == 12);
        for (int expert = 0; expert < 12; ++expert) {
            assert(unrestricted.experts[expert] == expert);
            assert(unrestricted.Allows(expert));
        }

        const DeepSeekV4ExpertAllowList restricted =
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[11,1,9,3,7,5]", 12, 6, "backbone");
        assert(restricted.configured);
        assert(restricted.experts == std::vector<int>({1, 3, 5, 7, 9, 11}));
        assert(fastllm::DeepSeekV4ExpertAllowListJson(restricted) ==
               "[1,3,5,7,9,11]");

        ExpectFailure([] {
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[0,1,2,3,4]", 12, 6, "backbone");
        }, "at least 6");
        ExpectFailure([] {
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[0,1,2,3,4,4]", 12, 6, "backbone");
        }, "duplicate");
        ExpectFailure([] {
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[0,1,2,3,4,12]", 12, 6, "mtp");
        }, "out-of-range");
        ExpectFailure([] {
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[0,1,2,3,4,5.5]", 12, 6, "mtp");
        }, "non-integer");
        ExpectFailure([] {
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "0,1,2,3,4,5", 12, 6, "backbone");
        }, "JSON array");
    }

    void TestLearnedRouterMasksBeforeTopK() {
        const DeepSeekV4ExpertAllowList policy =
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[1,3,5,7,9,11]", 12, 6, "backbone");
        std::vector<float> rawScores(24);
        for (int token = 0; token < 2; ++token) {
            for (int expert = 0; expert < 12; ++expert) {
                rawScores[token * 12 + expert] = policy.Allows(expert)
                    ? (float)(expert + token)
                    : (float)(1000 + expert);
            }
        }
        std::vector<float> gateBias(12, 0.0f);
        for (int expert = 0; expert < 12; expert += 2) {
            gateBias[expert] = 10000.0f;
        }

        std::vector<int> indices;
        std::vector<float> scores;
        fastllm::SelectDeepSeekV4AllowedExperts(
            rawScores, 2, 12, 6, "sqrtsoftplus", 1.5f,
            &gateBias, &policy, indices, scores);
        AssertUniqueAllowedRows(indices, 6, policy);
        for (int token = 0; token < 2; ++token) {
            float sum = 0.0f;
            for (int slot = 0; slot < 6; ++slot) {
                assert(scores[token * 6 + slot] > 0.0f);
                sum += scores[token * 6 + slot];
            }
            assert(std::fabs(sum - 1.5f) < 1e-5f);
        }

        fastllm::SelectDeepSeekV4AllowedExperts(
            rawScores, 2, 12, 6, "softmax", 0.75f,
            &gateBias, &policy, indices, scores);
        AssertUniqueAllowedRows(indices, 6, policy);
        for (int token = 0; token < 2; ++token) {
            float sum = 0.0f;
            for (int slot = 0; slot < 6; ++slot) {
                sum += scores[token * 6 + slot];
            }
            assert(std::fabs(sum - 0.75f) < 1e-5f);
        }

        std::vector<float> oneRow(12);
        for (int expert = 0; expert < 12; ++expert) {
            oneRow[expert] = (float)expert;
        }
        fastllm::SelectDeepSeekV4AllowedExperts(
            oneRow, 1, 12, 6, "softmax", 1.0f,
            nullptr, nullptr, indices, scores);
        assert(indices == std::vector<int>({11, 10, 9, 8, 7, 6}));
    }

    std::vector<float> BuildRouterRows(int expertCount, int rowWidth) {
        std::vector<float> rows((size_t)expertCount * rowWidth, 0.0f);
        for (int expert = 0; expert < expertCount; ++expert) {
            const float angle = 0.23f * expert;
            rows[(size_t)expert * rowWidth] = std::cos(angle);
            rows[(size_t)expert * rowWidth + 1] = std::sin(angle);
            if (rowWidth > 2) {
                rows[(size_t)expert * rowWidth + 2] = 0.01f * expert;
            }
        }
        return rows;
    }

    void TestHashRouterSimilarityRemap() {
        const DeepSeekV4ExpertAllowList policy =
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[1,3,5,7,9,11]", 12, 6, "backbone");
        const std::vector<float> routerRows = BuildRouterRows(12, 3);
        const std::vector<int> sourceRoutes = {
            0, 3, 4, 7, 8, 11,
            10, 9, 6, 5, 2, 1,
        };

        const DeepSeekV4HashRemapResult first =
            fastllm::RemapDeepSeekV4HashRoutes(
                routerRows, sourceRoutes, 12, 3, 6, policy);
        const DeepSeekV4HashRemapResult second =
            fastllm::RemapDeepSeekV4HashRoutes(
                routerRows, sourceRoutes, 12, 3, 6, policy);
        assert(first.routes == second.routes);
        assert(first.entries == sourceRoutes.size());
        assert(first.remapped > 0);
        AssertUniqueAllowedRows(first.routes, 6, policy);
        assert(first.routes[1] == 3);
        assert(first.routes[3] == 7);
        assert(first.routes[5] == 11);
        assert(first.routes[7] == 9);
        assert(first.routes[9] == 5);
        assert(first.routes[11] == 1);

        const DeepSeekV4ExpertAllowList explicitFull =
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[0,1,2,3,4,5,6,7,8,9,10,11]",
                12, 6, "backbone");
        const DeepSeekV4HashRemapResult unchanged =
            fastllm::RemapDeepSeekV4HashRoutes(
                routerRows, sourceRoutes, 12, 3, 6, explicitFull);
        assert(unchanged.routes == sourceRoutes);
        assert(unchanged.remapped == 0);

        std::vector<int> invalidRoutes = sourceRoutes;
        invalidRoutes[0] = 12;
        ExpectFailure([&] {
            fastllm::RemapDeepSeekV4HashRoutes(
                routerRows, invalidRoutes, 12, 3, 6, policy);
        }, "out-of-range");

        std::vector<float> nonFiniteRows = routerRows;
        nonFiniteRows[0] = std::numeric_limits<float>::quiet_NaN();
        ExpectFailure([&] {
            fastllm::RemapDeepSeekV4HashRoutes(
                nonFiniteRows, sourceRoutes, 12, 3, 6, policy);
        }, "non-finite router row");
    }

    void TestScopeAndMetadata() {
        const DeepSeekV4ExpertAllowList backbone =
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[1,3,5,7,9,11]", 12, 6, "backbone");
        const DeepSeekV4ExpertAllowList mtp =
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[0,2,4,6,8,10]", 12, 6, "mtp");

        TestableDeepSeekV4Model model;
        model.SetExpertPolicies(12, backbone, mtp);
        assert(model.PolicyFor("layers.0.ffn").experts == backbone.experts);
        assert(model.PolicyFor("mtp.0.ffn").experts == mtp.experts);

        model.SetHashCoverage("layers.0.ffn", 72, 31);
        const auto &metadata = model.Metadata();
        assert(metadata.at("expert_allow_list.physical_expert_count") == "12");
        assert(metadata.at("expert_allow_list.backbone.configured") == "true");
        assert(metadata.at("expert_allow_list.backbone.effective") ==
               "[1,3,5,7,9,11]");
        assert(metadata.at("expert_allow_list.mtp.configured") == "true");
        assert(metadata.at("expert_allow_list.mtp.effective") ==
               "[0,2,4,6,8,10]");
        assert(metadata.at("expert_allow_list.hash_remap_entries") == "72");
        assert(metadata.at("expert_allow_list.hash_remapped_entries") == "31");
        const std::string coverage =
            metadata.at("expert_allow_list.hash_remap_coverage");
        assert(coverage.find("layers.0.ffn") != std::string::npos);
        assert(coverage.find("\"complete\": true") != std::string::npos);
    }

    void TestDiskMoeRoutingBoundary() {
        const DeepSeekV4ExpertAllowList policy =
            fastllm::ParseDeepSeekV4ExpertAllowList(
                "[1,3,5,7,9,11]", 12, 6, "backbone");
        const std::vector<int> routed = {1, 3, 5, 7, 9, 11};
        fastllm::ValidateDeepSeekV4AllowedExpertIndices(routed, &policy);

        std::vector<int> disallowed = routed;
        disallowed[4] = 8;
        ExpectFailure([&] {
            fastllm::ValidateDeepSeekV4AllowedExpertIndices(
                disallowed, &policy);
        }, "disallowed physical expert");

        constexpr int expertCount = 12;
        constexpr int hidden = 2;
        constexpr int intermediate = 2;
        const std::vector<float> gateValues = {
            0.25f, -0.5f, 0.75f, 0.125f,
            -0.25f, 0.5f, 0.375f, -0.625f,
        };
        const std::vector<float> downValues = {
            0.5f, -0.25f, 0.125f, 0.75f,
        };
        const auto nonce = std::chrono::steady_clock::now()
                               .time_since_epoch().count();
        const std::filesystem::path validPath =
            std::filesystem::temp_directory_path() /
            ("fastllm-dsv4-allowed-experts-" + std::to_string(nonce) + ".bin");
        const std::filesystem::path deniedPath = validPath.string() + ".denied";
        std::filesystem::remove(validPath);
        std::filesystem::remove(deniedPath);
        {
            std::ofstream output(validPath, std::ios::binary);
            assert(output.good());
            output.write(reinterpret_cast<const char*>(gateValues.data()),
                         (std::streamsize)(gateValues.size() * sizeof(float)));
            output.write(reinterpret_cast<const char*>(downValues.data()),
                         (std::streamsize)(downValues.size() * sizeof(float)));
            assert(output.good());
        }

        std::vector<std::unique_ptr<fastllm::Data> > ownedWeights;
        std::vector<fastllm::Data*> weights(2 + expertCount * 2, nullptr);
        auto addDiskWeight = [&](int slot, const std::vector<int> &dims,
                                 uint64_t bytes, long long fileOffset,
                                 bool allowed) {
            auto weight = std::make_unique<fastllm::Data>(
                fastllm::DataType::FLOAT32, dims);
            weight->name = "dsv4.allowlist.expert." + std::to_string(slot);
            weight->weightType = fastllm::WeightType::LINEAR;
            weight->isDiskWeight = true;
            fastllm::DiskWeightPart part;
            part.fileName = allowed ? validPath.string() : deniedPath.string();
            part.fileOffset = fileOffset;
            part.bytes = bytes;
            part.sourceDataType = fastllm::DataType::FLOAT32;
            part.dims = dims;
            weight->diskWeightParts.push_back(part);
            weights[slot] = weight.get();
            ownedWeights.push_back(std::move(weight));
        };
        const uint64_t gateBytes = gateValues.size() * sizeof(float);
        const uint64_t downBytes = downValues.size() * sizeof(float);
        for (int expert = 0; expert < expertCount; ++expert) {
            const bool allowed = policy.Allows(expert);
            addDiskWeight(2 + expert * 2,
                          {intermediate * 2, hidden}, gateBytes, 0, allowed);
            addDiskWeight(3 + expert * 2,
                          {hidden, intermediate}, downBytes,
                          (long long)gateBytes, allowed);
        }

        fastllm::Data input(fastllm::DataType::FLOAT32, {1, hidden},
                            {0.5f, -0.75f});
        fastllm::Data expertIndex;
        expertIndex.dataType = fastllm::DataType::INT32;
        expertIndex.Resize({1, 6});
        expertIndex.Allocate(false);
        std::memcpy(expertIndex.cpuData, routed.data(),
                    routed.size() * sizeof(int32_t));
        fastllm::Data expertScore(
            fastllm::DataType::FLOAT32, {1, 6},
            std::vector<float>(6, 1.0f / 6.0f));
        fastllm::Data allowedExpertMask;
        allowedExpertMask.dataType = fastllm::DataType::INT32;
        allowedExpertMask.Resize({expertCount});
        allowedExpertMask.Allocate(false);
        std::vector<int32_t> allowedMaskValues(
            policy.mask.begin(), policy.mask.end());
        std::memcpy(allowedExpertMask.cpuData, allowedMaskValues.data(),
                    allowedMaskValues.size() * sizeof(int32_t));
        std::vector<fastllm::Data*> biass(weights.size(), nullptr);
        fastllm::Data w1, w2, w3, tempInput, tempOutput;
        fastllm::Data output(fastllm::DataType::FLOAT32, {1, hidden});
        {
            ScopedFirstDevice device("disk");
            fastllm::MergeMOE(
                input, expertIndex, expertScore, weights, biass,
                w1, w2, w3, tempInput, tempOutput, 0.0f, output,
                0, fastllm::MoeGateSwiglu, false, 0.0f, false,
                &allowedExpertMask);
        }
        const float *outputData = reinterpret_cast<const float*>(output.cpuData);
        assert(outputData != nullptr);
        for (int i = 0; i < hidden; ++i) {
            assert(std::isfinite(outputData[i]));
        }
        for (const auto &weight : ownedWeights) {
            assert(weight->cpuData == nullptr);
        }

        ((int32_t*)expertIndex.cpuData)[0] = 8;
        ExpectFailure([&] {
            ScopedFirstDevice device("disk");
            fastllm::MergeMOE(
                input, expertIndex, expertScore, weights, biass,
                w1, w2, w3, tempInput, tempOutput, 0.0f, output,
                0, fastllm::MoeGateSwiglu, false, 0.0f, false,
                &allowedExpertMask);
        }, "disallowed physical expert");
        std::filesystem::remove(validPath);

        // Every denied expert points at a deliberately missing file. A
        // successful Disk MergeMOE therefore proves that only the physical IDs
        // published in expertIndex were staged at the disk/cache boundary.
        assert(!std::filesystem::exists(deniedPath));
    }
}

int main() {
    TestStrictParsingAndCanonicalIds();
    TestLearnedRouterMasksBeforeTopK();
    TestHashRouterSimilarityRemap();
    TestScopeAndMetadata();
    TestDiskMoeRoutingBoundary();
    std::cout << "DeepSeek V4 expert allow-list tests passed.\n";
    return 0;
}
