#include "models/deepseekv4.h"
#include "executor.h"
#include "fastllm.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// Exercises the file-local BuildMoERoutingData through the
// DeepSeekV4BuildMoERoutingDataForTesting seam, with tiny in-memory weights and
// no checkpoint. The two CPU cases run unconditionally (they never launch a
// CUDA kernel). The CUDA case is opt-in via FASTLLM_DSV4_TEST_CUDA_ROUTE=1 so a
// default run never creates GPU work on a machine whose GPU may be serving the
// frozen production endpoint.

namespace {
    using fastllm::Data;
    using fastllm::DataType;
    using fastllm::DataDevice;
    using fastllm::DeepSeekV4ExpertAllowList;
    using fastllm::WeightMap;

    constexpr int kExperts = 4;
    constexpr int kTopk = 2;
    constexpr int kHidden = 3;
    const std::string kPrefix = "layers.0.ffn";

    DeepSeekV4ExpertAllowList MakePolicyDenyingExpert3() {
        DeepSeekV4ExpertAllowList policy;
        policy.configured = true;
        policy.experts = {0, 1, 2};
        policy.mask = {1, 1, 1, 0}; // experts 0,1,2 allowed; 3 denied
        return policy;
    }

    // Gate rows chosen so that, for an all-ones input, the router logit of the
    // denied expert 3 (sum 10) is highest, then expert 2 (3), 1 (2), 0 (1).
    // Materialize the weight directly inside the map slot: Data's copy
    // assignment is shallow (only its copy constructor is deep), so assigning a
    // local Data into the map would leave a dangling cpuData pointer.
    void FillGateWeight(WeightMap &weight) {
        const std::vector<float> gate = {
            0.3f, 0.3f, 0.4f, // expert 0 -> logit 1
            0.6f, 0.7f, 0.7f, // expert 1 -> logit 2
            1.0f, 1.0f, 1.0f, // expert 2 -> logit 3
            3.0f, 3.0f, 4.0f, // expert 3 (denied) -> logit 10
        };
        Data &gateWeight = weight.weight[kPrefix + ".gate.weight"];
        gateWeight.dataType = DataType::FLOAT32;
        gateWeight.Resize({kExperts, kHidden});
        gateWeight.Allocate();
        std::memcpy(gateWeight.cpuData, gate.data(),
                    gate.size() * sizeof(float));
        gateWeight.weightType = fastllm::WeightType::LINEAR;
    }

    void FillRouteTable(WeightMap &weight, const std::vector<float> &tid2eid) {
        Data &routeTable = weight.weight[kPrefix + ".gate.tid2eid"];
        routeTable.dataType = DataType::FLOAT32;
        routeTable.Resize({(int)tid2eid.size()});
        routeTable.Allocate();
        std::memcpy(routeTable.cpuData, tid2eid.data(),
                    tid2eid.size() * sizeof(float));
    }

    Data MakeAllOnesInput(DataDevice device) {
        Data x(DataType::FLOAT32, {1, kHidden},
               std::vector<float>(kHidden, 1.0f));
        if (device == DataDevice::CUDA) {
            x.ToDevice(DataDevice::CUDA);
        }
        return x;
    }

    std::vector<int> ReadIndices(Data &expertIndex, int count) {
        expertIndex.ToDevice(DataDevice::CPU);
        const int32_t *raw = reinterpret_cast<const int32_t*>(expertIndex.cpuData);
        assert(raw != nullptr);
        return std::vector<int>(raw, raw + count);
    }

    void AssertAllowedAndDenies3(const std::vector<int> &indices,
                                 const DeepSeekV4ExpertAllowList &policy) {
        assert((int)indices.size() == kTopk);
        std::set<int> unique(indices.begin(), indices.end());
        assert((int)unique.size() == kTopk);
        for (int expert : indices) {
            assert(policy.Allows(expert));
            assert(expert != 3);
        }
        fastllm::ValidateDeepSeekV4AllowedExpertIndices(indices, &policy);
    }

    // Learned routing on CPU: no tid2eid. The denied expert owns the highest
    // logit, so an unrestricted selector would pick it; the policy must mask it
    // before top-k, leaving experts {2, 1}.
    void TestLearnedCpuDeniesHighest() {
        WeightMap weight;
        FillGateWeight(weight);
        const DeepSeekV4ExpertAllowList policy = MakePolicyDenyingExpert3();
        const Data x = MakeAllOnesInput(DataDevice::CPU);
        const std::vector<int> inputIds = {0};

        Data expertIndex, expertScore;
        fastllm::DeepSeekV4BuildMoERoutingDataForTesting(
            weight, kPrefix, x, inputIds, kExperts, kTopk, "softmax", 1.0f,
            expertIndex, expertScore, nullptr, &policy, nullptr);

        const std::vector<int> indices = ReadIndices(expertIndex, kTopk);
        assert(indices[0] == 2);
        assert(indices[1] == 1);
        AssertAllowedAndDenies3(indices, policy);
    }

    // Hash routing on CPU: a remapped tid2eid table directly publishes the
    // selected physical experts for each input token. CUDA routing is disabled
    // (see main) so the CPU hash publication is the path under test.
    void TestHashCpuRemappedTable() {
        WeightMap weight;
        FillGateWeight(weight);
        // Row for inputId 0 = {0,1}; row for inputId 1 = {2,0}. All allowed.
        FillRouteTable(weight, {0.0f, 1.0f, 2.0f, 0.0f});

        const DeepSeekV4ExpertAllowList policy = MakePolicyDenyingExpert3();
        const Data x = MakeAllOnesInput(DataDevice::CPU);
        const std::vector<int> inputIds = {1}; // selects row {2,0}

        Data expertIndex, expertScore;
        fastllm::DeepSeekV4BuildMoERoutingDataForTesting(
            weight, kPrefix, x, inputIds, kExperts, kTopk, "softmax", 1.0f,
            expertIndex, expertScore, nullptr, &policy, nullptr);

        const std::vector<int> indices = ReadIndices(expertIndex, kTopk);
        assert(indices[0] == 2);
        assert(indices[1] == 0);
        AssertAllowedAndDenies3(indices, policy);
    }

#ifdef USE_CUDA
    // Hash routing on CUDA: same table and input, but the router logits live on
    // the device so the CUDA early-return path publishes the indices. Opt-in.
    void TestHashCuda() {
        WeightMap weight;
        FillGateWeight(weight);
        FillRouteTable(weight, {0.0f, 1.0f, 2.0f, 0.0f});
        weight.weight[kPrefix + ".gate.tid2eid"].ToDevice(DataDevice::CUDA);
        weight.weight[kPrefix + ".gate.weight"].ToDevice(DataDevice::CUDA);

        const DeepSeekV4ExpertAllowList policy = MakePolicyDenyingExpert3();
        const Data x = MakeAllOnesInput(DataDevice::CUDA);
        const std::vector<int> inputIds = {1};

        Data expertIndex, expertScore;
        fastllm::DeepSeekV4BuildMoERoutingDataForTesting(
            weight, kPrefix, x, inputIds, kExperts, kTopk, "softmax", 1.0f,
            expertIndex, expertScore, nullptr, &policy, nullptr);

        const std::vector<int> indices = ReadIndices(expertIndex, kTopk);
        AssertAllowedAndDenies3(indices, policy);
    }
#endif
}

int main() {
    // Force the deterministic CPU routing publications for the always-on cases.
    setenv("FASTLLM_DSV4_DISABLE_CUDA_ROUTE", "1", 1);

    TestLearnedCpuDeniesHighest();
    TestHashCpuRemappedTable();

#ifdef USE_CUDA
    if (std::getenv("FASTLLM_DSV4_TEST_CUDA_ROUTE") != nullptr) {
        unsetenv("FASTLLM_DSV4_DISABLE_CUDA_ROUTE");
        TestHashCuda();
        std::cout << "CUDA route seam exercised.\n";
    } else {
        std::cout << "Skipping CUDA route seam "
                     "(set FASTLLM_DSV4_TEST_CUDA_ROUTE=1 on a free GPU).\n";
    }
#endif

    std::cout << "DeepSeek V4 MoE routing seam tests passed.\n";
    return 0;
}
