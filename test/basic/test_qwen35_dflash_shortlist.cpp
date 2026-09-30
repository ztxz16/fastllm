#include "models/qwen3_5.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"
#include <iostream>
#include <set>
#include <stdexcept>
using namespace fastllm;
struct Probe : Qwen3_5Model {
    using Qwen3_5Model::dflashDraftTokenIds;
    using Qwen3_5Model::dflashNvfp4TpLmHeads;
    using Qwen3_5Model::threadTpLmHeadScheme;
    using Qwen3_5Model::RunDFlashLmHead;
    using Qwen3_5Model::MapDFlashShortlistCandidates;
};
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    if (FastllmCudaGetDeviceCount() == 0) return 77;
    try {
        FastllmCudaSetDevice(0);
        Probe model;
        model.weight["lm_head.weight"].multiDeviceData = true;
        model.threadTpLmHeadScheme[0] = {{0,512}};
        model.threadTpLmHeadScheme[1] = {{512,1024}};
        std::vector<int> rows;
        for (int i = 0; i < 129; ++i) rows.push_back(2*i);
        model.dflashDraftTokenIds[0] = rows;
        rows.resize(256, rows.back());
        std::vector<float> values(512*128);
        for (int row=0; row<512; ++row)
            for (int k=0; k<128; ++k)
                values[row*128+k] = row == 256 ? 2.0f : 0.005f*(row%13+1);
        Data original(FLOAT32, {512,128}, values), empty;
        original.ToDevice(CUDA, std::vector<int>{0});
        ToDataType(original, FLOAT16);
        Data &head = model.dflashNvfp4TpLmHeads[0];
        Check(FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(original, head, rows), "quantize rows");
        head.weightType = WeightType::LINEAR;
        head.isModelWeight = true;
        for (DataType type : {FLOAT16, BFLOAT16}) {
            Data input(FLOAT32, {1,8,128}, std::vector<float>(8*128, 0.01f)), output;
            input.ToDevice(CUDA, std::vector<int>{0});
            ToDataType(input, type);
            model.RunDFlashLmHead(0, input, original, empty, output, true);
            ForceDeviceSync();
            Check(output.dims == std::vector<int>({1,8,129}), "padding was not removed");
            ToDataType(output, FLOAT32);
            Data top;
            TopK(output, top, 16);
            top.ToDevice(CPU);
            model.MapDFlashShortlistCandidates(top);
            const float *pairs = reinterpret_cast<const float*>(top.cpuData);
            for (int row=0; row<8; ++row) {
                Check(int(pairs[row*32]) == 256, "highest candidate changed");
                std::set<int> ids;
                for (int i=0; i<16; ++i) {
                    int id = int(pairs[row*32+2*i]);
                    Check(id >= 0 && id <= 256 && id%2 == 0, "invalid global token ID");
                    ids.insert(id);
                }
                Check(ids.size() == 16, "padding created duplicate candidates");
            }
            Data full;
            model.RunDFlashLmHead(0, input, original, empty, full, false);
            Check(full.dims.back() == 512, "sampling fallback used compact head");
        }
        model.dflashDraftTokenIds[1] = {700,900};
        Data candidates(FLOAT32, {2,4}, {0,1,128,3,512,5,513,6});
        model.MapDFlashShortlistCandidates(candidates);
        const float *p = reinterpret_cast<const float*>(candidates.cpuData);
        const float expected[] = {0,1,256,3,700,5,900,6};
        for (int i=0; i<8; ++i) Check(p[i] == expected[i], "cross-shard mapping or score changed");
        std::cout << "PASS: padded rows excluded, unique top-k, global IDs, full-head fallback\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
