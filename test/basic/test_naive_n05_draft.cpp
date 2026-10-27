#include "models/naive_n05_flash.h"
#include "json11.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime_api.h>

using namespace fastllm;

// Checkpoint-backed oracle test. Fixtures are produced by the original Torch
// draft implementation, independently of FastLLM's attention/cache code.
class NaiveDraftFixture : public NaiveN05FlashModel {
public:
    double referenceRounding = 0;
    void Read(const std::string &directory) {
        std::ifstream stream(directory + "/manifest.json");
        std::string text((std::istreambuf_iterator<char>(stream)), {}), error;
        auto manifest = json11::Json::parse(text, error);
        if (!error.empty()) throw std::runtime_error(error);
        embed_dim = manifest["hidden_size"].int_value();
        draftLayers = manifest["layers"].int_value();
        draftHeads = manifest["heads"].int_value();
        draftKvHeads = manifest["kv_heads"].int_value();
        draftHeadDim = manifest["head_dim"].int_value();
        draftWindow = manifest["window"].int_value();
        draftBlock = manifest["block"].int_value();
        draftEps = manifest["eps"].number_value();
        draftTheta = manifest["theta"].number_value();
        referenceRounding = manifest["torch_bf16_vs_fp32_rmse"].number_value();
        deviceMap = {{"cuda:0", 1}};
        block_cnt = 48;
        for (const auto &entry : manifest["tensors"].array_items()) {
            std::string name = entry["name"].string_value();
            std::vector<int> shape;
            for (auto &d : entry["shape"].array_items()) shape.push_back(d.int_value());
            auto type = entry["dtype"].string_value() == "float32" ? FLOAT32 : BFLOAT16;
            weight.AddEmptyWeight(name, shape, type);
            Data &data = weight[name];
            data.Allocate();
            std::ifstream input(directory + "/" + entry["file"].string_value(), std::ios::binary);
            if (!input.read((char *)data.cpuData, data.GetBytes()))
                throw std::runtime_error("Cannot read tensor: " + name);
        }
    }
    void Run() {
        Data &hidden = weight["test.context"];
        const int length = hidden.dims[1], split = length / 2;
        DraftContext context;
        Data first, last;
        Split(hidden, 1, 0, split, first);
        Split(hidden, 1, split, length, last);
        AppendDraftContext(first, 0, context);
        AppendDraftContext(last, split, context);
        if (context.committed != length || context.kv[0].first.dims[1] != std::min(length, draftWindow - 1))
            throw std::runtime_error("Draft context append/trim failed");
        Data actual = RunDraft(0, context);
        ToDataType(actual, FLOAT32);
        actual.ToDevice(DataDevice::CPU);
        Data &expected = weight["test.expected"];
        double squared = 0, dot = 0, aa = 0, bb = 0, maximum = 0;
        for (int i = 0; i < actual.Count(0); ++i) {
            double a = ((float *)actual.cpuData)[i], b = ((float *)expected.cpuData)[i];
            maximum = std::max(maximum, std::abs(a - b));
            squared += (a - b) * (a - b); dot += a * b; aa += a * a; bb += b * b;
        }
        double rmse = std::sqrt(squared / actual.Count(0)), cosine = dot / std::sqrt(aa * bb);
        std::cout << "draft_hidden_rmse=" << rmse << " cosine=" << cosine << " max_abs=" << maximum << '\n';
        // Quantify the BF16 error with the independent FP32 pass, rather than
        // imposing an absolute threshold on checkpoint-specific norm scales.
        if (!std::isfinite(rmse) || rmse > std::max(.02, 1.5 * referenceRounding) || cosine < .998)
            throw std::runtime_error("Draft differs from original Torch implementation");
        Data repeat = RunDraft(0, context);
        ToDataType(repeat, FLOAT32); repeat.ToDevice(DataDevice::CPU);
        if (std::memcmp(actual.cpuData, repeat.cpuData, actual.GetBytes()) != 0)
            throw std::runtime_error("Draft proposal contaminated the committed KV cache");
    }
};

int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "Usage: naive_n05_draft_test FIXTURE_DIRECTORY\n"; return 2; }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        NaiveDraftFixture model;
        model.Read(argv[1]);
        model.Run();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
