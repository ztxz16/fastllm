#include "models/qwen3_5.h"
#include "executor.h"
#include "utils/p0_snapshot_observation.h"
#include <iostream>
#include <stdexcept>
using namespace fastllm;
class PackedCpuModel : public Qwen3_5Model {
public:
    using Qwen3_5Model::ForwardGPUWithHiddenStates;
    using Qwen3_5Model::BuildFlattenedPositionIds;
    using Qwen3_5Model::SplitMultimodalPositionIdsCPU;
};
// A preferred non-CPU Split candidate makes accidental automatic dispatch fail
// without CUDA initialization. The production CPU helper must bypass it.
class PositionDispatchProbe : public BaseDevice {
public:
    int splitCalls = 0;
    PositionDispatchProbe() { deviceType = "position-dispatch-probe"; }
    bool Malloc(void **, size_t) override { return false; }
    bool Free(void *) override { return false; }
    bool CopyDataToCPU(void *, void *, size_t) override { return false; }
    bool CopyDataFromCPU(void *, void *, size_t) override { return false; }
    bool CanRun(const std::string &op, const DataDict &, const FloatDict &,
                const IntDict &) override { return op == "Split"; }
    void Reshape(const std::string &, const DataDict &, const FloatDict &,
                 const IntDict &) override {}
    void Run(const std::string &, const DataDict &, const FloatDict &,
             const IntDict &) override {
        ++splitCalls;
        throw std::runtime_error("position dispatch negative control");
    }
};
int main() {
    ResponseContextDict functionalIdentity;
    const int firstHandle = functionalIdentity.CreateHandle();
    const bool validZeroHandle = firstHandle == 0 &&
        functionalIdentity.GetHandle(firstHandle)->nativeRequestGeneration != 0;
    PackedCpuModel model;
    model.block_cnt = 1;
    Data ids(DataType::FLOAT32, {1, 5}, {1, 2, 3, 4, 5});
    Data pos0(DataType::FLOAT32, {3, 2}, {10, 11, 20, 21, 30, 31});
    Data pos1(DataType::FLOAT32, {3, 3}, {100, 101, 102, 200, 201, 202, 300, 301, 302});
    Data hidden(DataType::FLOAT32, {1, 5, 2}, {1,2,3,4,5,6,7,8,9,10});
    Data key0(DataType::FLOAT32, {1, 1, 1}, {101});
    Data value0(DataType::FLOAT32, {1, 1, 1}, {102});
    Data key1(DataType::FLOAT32, {1, 1, 1}, {201});
    Data value1(DataType::FLOAT32, {1, 1, 1}, {202});
    std::vector<std::pair<Data*, Data*>> kv{{&key0,&value0},{&key1,&value1}};
    std::vector<Data*> positions{&pos0,&pos1}, masks{nullptr,nullptr};
    std::vector<GenerationConfig> configs(2);
    Data packedPositions = model.BuildFlattenedPositionIds(positions, {2,3}, false);
    const std::vector<float> expected{10,11,100,101,102,20,21,200,201,202,30,31,300,301,302};
    bool positionsCorrect = packedPositions.dims == std::vector<int>({3,5});
    for (size_t i = 0; positionsCorrect && i < expected.size(); ++i)
        positionsCorrect = ((float*)packedPositions.cpuData)[i] == expected[i];
    positionsCorrect = positionsCorrect && ((float*)pos0.cpuData)[0] == 10 &&
        ((float*)pos1.cpuData)[0] == 100;
    bool rejected = false;
    try {
        model.ForwardGPUWithHiddenStates(2, ids, masks, positions, {2,3}, kv,
            configs, LastTokensManager(), nullptr, &hidden);
    } catch (const std::invalid_argument &) { rejected = true; }
    const bool isolated = key0.dims == std::vector<int>({1,1,1}) &&
        key1.dims == key0.dims && value0.dims == key0.dims && value1.dims == key0.dims &&
        ((float*)key0.cpuData)[0] == 101 && ((float*)value0.cpuData)[0] == 102 &&
        ((float*)key1.cpuData)[0] == 201 && ((float*)value1.cpuData)[0] == 202;
    auto *executor = (Executor*)GetExecutor();
    auto *probe = new PositionDispatchProbe();
    executor->AddDevice(probe);
    executor->SetFirstDevice("position-dispatch-probe");
    bool negativeControl = false;
    try { Data accidental; Split(pos1, 1, 1, 3, accidental); }
    catch (const std::runtime_error &) { negativeControl = probe->splitCalls == 1; }
    Data positionSlice, nestedSlice;
    model.SplitMultimodalPositionIdsCPU(pos1, 1, 3, positionSlice);
    model.SplitMultimodalPositionIdsCPU(positionSlice, 0, 2, nestedSlice);
    const std::vector<float> expectedSlice{101,102,201,202,301,302};
    bool cpuPositionOwnership = positionSlice.dataDevice == DataDevice::CPU &&
        nestedSlice.dataDevice == DataDevice::CPU && positionSlice.cpuData != nullptr &&
        nestedSlice.cpuData != nullptr && positionSlice.dims == std::vector<int>({3,2}) &&
        nestedSlice.dims == positionSlice.dims && probe->splitCalls == 1 &&
        pos1.dataDevice == DataDevice::CPU && ((float*)pos1.cpuData)[0] == 100;
    for (size_t i = 0; cpuPositionOwnership && i < expectedSlice.size(); ++i)
        cpuPositionOwnership = ((float*)positionSlice.cpuData)[i] == expectedSlice[i] &&
            ((float*)nestedSlice.cpuData)[i] == expectedSlice[i];
    const bool readableDigest = cpuPositionOwnership &&
        p0snapshot::ContentDigest(nestedSlice.cpuData, nestedSlice.GetBytes()) ==
        p0snapshot::ContentDigest(expectedSlice.data(), expectedSlice.size()*sizeof(float));
    std::cout << "{\"kind\":\"actual_Data_CPU_guard\",\"rejected\":"
        << (rejected ? "true" : "false") << ",\"both_row_kv_unchanged\":"
        << (isolated ? "true" : "false") << ",\"actual_Data_axis_major_correct\":"
        << (positionsCorrect ? "true" : "false")
        << ",\"automatic_split_negative_control\":" << (negativeControl ? "true" : "false")
        << ",\"production_position_slice_cpu_owned\":" << (cpuPositionOwnership ? "true" : "false")
        << ",\"position_digest_matches_values\":" << (readableDigest ? "true" : "false") << "}\n";
    return rejected && isolated && positionsCorrect && validZeroHandle &&
        negativeControl && cpuPositionOwnership && readableDigest ? 0 : 1;
}
