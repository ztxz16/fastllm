#include "models/qwen3_5.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-vision.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace fastllm;

// End-to-end synthetic tower: real head width and non-divisible MLP width,
// nonzero row-projection biases, image/video ordering and cache identities.
class VisionFixture : public Qwen3_5Model {
public:
    VisionFixture() {
        dataType = DataType::FLOAT32;
        vision_depth = 2;
        vision_hidden_size = 1152;
        vision_num_heads = 16;
        vision_head_dim = 72;
        vision_intermediate_size = 4304;
        vision_out_hidden_size = 80;
        vision_patch_size = 2;
        vision_temporal_patch_size = 2;
        vision_spatial_merge_size = 2;
        vision_num_grid_per_side = 4;
        vision_num_position_embeddings = 16;
        std::mt19937 rng(8179);
        auto add = [&](const std::string &name, std::vector<int> dims, bool norm = false) {
            size_t count = 1;
            for (int dim : dims) count *= dim;
            std::vector<float> values(count);
            std::uniform_real_distribution<float> distribution(-0.015f, 0.015f);
            for (float &value : values) value = (norm ? 1.0f : 0.0f) + distribution(rng);
            weight[visual_prefix + name].CopyFrom(Data(DataType::FLOAT32, dims, values));
        };
        auto linear = [&](const std::string &name, int out, int in) {
            add(name + ".weight", {out, in});
            add(name + ".bias", {out});
        };
        auto norm = [&](const std::string &name) {
            add(name + ".weight", {vision_hidden_size}, true);
            add(name + ".bias", {vision_hidden_size});
        };
        linear("patch_embed.proj", 1152, 24);
        add("pos_embed.weight", {16, 1152});
        for (int i = 0; i < vision_depth; ++i) {
            std::string prefix = "blocks." + std::to_string(i);
            norm(prefix + ".norm1");
            norm(prefix + ".norm2");
            linear(prefix + ".attn.qkv", 3456, 1152);
            linear(prefix + ".attn.proj", 1152, 1152);
            linear(prefix + ".mlp.linear_fc1", 4304, 1152);
            linear(prefix + ".mlp.linear_fc2", 1152, 4304);
        }
        norm("merger.norm");
        linear("merger.linear_fc1", 4608, 4608);
        linear("merger.linear_fc2", vision_out_hidden_size, 4608);
    }
    void CheckUnsupportedStorage(bool gguf) {
        const std::string name = visual_prefix + "blocks.0.attn.qkv.weight";
        Data &tensor = weight[name];
        const auto type = tensor.dataType;
        tensor.dataType = gguf ? type : DataType::INT8;
        tensor.isGGUFData = gguf;
        bool rejected = false;
        try {
            PrepareVision();
        } catch (const std::runtime_error &error) {
            rejected = std::string(error.what()).find(name) != std::string::npos;
        }
        tensor.dataType = type;
        tensor.isGGUFData = false;
        if (!rejected) throw std::runtime_error("Packed vision TP weights were not explicitly rejected");
    }
    std::vector<float> Encode(bool video, bool reverse = false, bool cache = false) {
        const int h = 8, w = 12;
        std::vector<float> pixels((video ? 4 : 1) * h * w * 3);
        for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = ((i * 17 + 29) % 127) / 127.0f;
        std::vector<float> second = pixels;
        for (float &value : second) value = 1.0f - value;
        const auto shape = video ? std::vector<int>{4, h, w, 3} : std::vector<int>{h, w, 3};
        Data first(DataType::FLOAT32, shape, pixels), other(DataType::FLOAT32, shape, second);
        Data grid(DataType::FLOAT32, {2, 3}, {(float)(video ? 2 : 1), 4, 6, (float)(video ? 2 : 1), 4, 6});
        Data keys(DataType::INT32, {2, 8});
        keys.Allocate();
        for (int i = 0; i < 16; ++i) ((int*)keys.cpuData)[i] = (i < 8) != reverse ? 17 + i % 8 : 35 + i % 8;
        Data output;
        std::vector<std::vector<int>> grids;
        std::vector<Data*> inputs = reverse ? std::vector<Data*>{&other, &first} : std::vector<Data*>{&first, &other};
        EncodeVisualItems(inputs, &grid, video, output, grids, cache ? &keys : nullptr);
        if (output.dataDevice != DataDevice::CPU || output.dataType != DataType::FLOAT32 || grids.size() != 2)
            throw std::runtime_error("Vision output staging contract failed");
        return {(float*)output.cpuData, (float*)output.cpuData + output.Count(0)};
    }
};

static void Compare(const std::vector<float> &actual, const std::vector<float> &reference, const char *name) {
    if (actual.size() != reference.size()) throw std::runtime_error("Feature shape mismatch");
    double squared = 0, energy = 0, maxError = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("Non-finite vision feature");
        const double delta = actual[i] - reference[i];
        squared += delta * delta;
        energy += (double)reference[i] * reference[i];
        maxError = std::max(maxError, std::abs(delta));
    }
    const double nrmse = std::sqrt(squared / std::max(energy, 1e-20));
    std::cout << name << " nrmse=" << nrmse << " max_abs=" << maxError << std::endl;
    if (nrmse > 1e-4 || maxError > 1e-4) throw std::runtime_error("Vision TP differs from unsharded FP32 reference");
}

static void CheckUnroundedPartial(int device) {
    // 1024.25 cannot be represented in half. A partial GEMM that rounds to
    // half before returning FP32 would silently lose the .25 contribution.
    Data input(DataType::FLOAT16, {2, 16}, std::vector<float>(32, 1.0f));
    std::vector<float> values(16, 0.0f);
    values[0] = 1024.0f;
    values[15] = 0.25f;
    Data weight(DataType::FLOAT16, {1, 16}, values);
    Data output(DataType::FLOAT32, {2, 1});
    input.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    weight.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    output.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
    output.Allocate(false);
    if (!FastllmCudaVisionLinearFloat32(input, weight, output))
        throw std::runtime_error("Vision FP32 partial GEMM rejected valid operands");
    FastllmCudaSyncCurrentThreadStream();
    output.ToDevice(DataDevice::CPU);
    for (int row = 0; row < 2; ++row) {
        if (((float*)output.cpuData)[row] != 1024.25f)
            throw std::runtime_error("Vision partial GEMM rounded before the full sum");
    }
    std::cout << "unrounded_partial PASS" << std::endl;
}



static void CheckBf16ColumnAccumulation(int device) {
    // TP8 QKV has 24 rows, 1152 input channels, and 432 local output
    // channels. Products are exact multiples of 1/512 and every partial
    // sum fits FP32 exactly, so the CPU double dot is a bit-exact oracle.
    // BF16 output would lose low bits in many of these completed dots.
    constexpr int rows = 24, inputs = 1152, outputs = 432;
    std::vector<float> inputValues(rows * inputs), weightValues(outputs * inputs);
    for (size_t i = 0; i < inputValues.size(); ++i)
        inputValues[i] = ((int)((i * 17 + 29) % 31) - 15) / 16.0f;
    for (size_t i = 0; i < weightValues.size(); ++i)
        weightValues[i] = ((int)((i * 13 + 7) % 29) - 14) / 32.0f;
    Data input(DataType::BFLOAT16, {rows, inputs}, inputValues);
    Data weight(DataType::BFLOAT16, {outputs, inputs}, weightValues);
    Data output(DataType::FLOAT32, {rows, outputs});
    input.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    weight.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    output.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
    output.Allocate(false);
    if (!FastllmCudaVisionLinearFloat32(input, weight, output))
        throw std::runtime_error("Vision BF16 column GEMM rejected valid operands");
    FastllmCudaSyncCurrentThreadStream();
    output.ToDevice(DataDevice::CPU);
    size_t nonBf16Results = 0;
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < outputs; ++column) {
            double exact = 0.0;
            for (int i = 0; i < inputs; ++i)
                exact += (double)inputValues[row * inputs + i] * weightValues[column * inputs + i];
            const float expected = (float)exact;
            uint32_t bits;
            std::memcpy(&bits, &expected, sizeof(bits));
            nonBf16Results += (bits & 0xffff) != 0;
            if (((float*)output.cpuData)[row * outputs + column] != expected)
                throw std::runtime_error("Vision BF16 column GEMM rounded before the complete FP32 dot");
        }
    }
    if (nonBf16Results == 0)
        throw std::runtime_error("Vision BF16 column fixture cannot distinguish a low-precision result");
    std::cout << "bf16_column_accumulation PASS" << std::endl;
}

static void CheckProjectionRounding(int device) {
    // The two midpoint ties exercise even low/high mantissas; neighboring
    // values and both signs distinguish nearest-even from toward-zero.
    const float firstMid = 1.0f + std::ldexp(1.0f, -11);
    const float secondMid = 1.0f + std::ldexp(3.0f, -11);
    const float subnormalMid = std::ldexp(1.0f, -25);
    const std::vector<float> positive = {
        std::nextafter(firstMid, 0.0f), firstMid,
        std::nextafter(firstMid, 2.0f), secondMid,
        subnormalMid, std::nextafter(subnormalMid, 1.0f)
    };
    const std::vector<uint16_t> positiveBits = {0x3c00, 0x3c00, 0x3c01, 0x3c02, 0x0000, 0x0001};
    std::vector<float> values = positive;
    std::vector<uint16_t> expected = positiveBits;
    for (size_t i = 0; i < positive.size(); ++i) {
        values.push_back(-positive[i]);
        expected.push_back(positiveBits[i] | 0x8000);
    }
    // These are real first-layer complete FP32 sums from the FP64 audit.
    values.push_back(-0.1543915867805481f);
    expected.push_back(0xb0f1); // -0.1544189453125
    values.push_back(-2.1093242168426514f);
    expected.push_back(0xc038); // -2.109375
    Data input(DataType::FLOAT32, {(int)values.size()}, values);
    Data output(DataType::FLOAT16, input.dims);
    input.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    output.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
    output.Allocate(false);
    if (!FastllmCudaVisionFloat32ToHalf(input, output))
        throw std::runtime_error("Vision nearest-even conversion rejected valid operands");
    FastllmCudaSyncCurrentThreadStream();
    output.ToDevice(DataDevice::CPU);
    const auto *actual = (const uint16_t*)output.cpuData;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i])
            throw std::runtime_error("Vision projection cast did not round to nearest even");
    }
    std::cout << "projection_rounding PASS" << std::endl;
}

int main(int argc, char **argv) {
    try {
        if (FastllmCudaGetDeviceCount() < 2) return 77;
        if (argc != 1 && argc != 3) throw std::runtime_error("usage: vision_tp_test [REFERENCE_DEVICE TP_SPEC]");
        const char *referenceDevice = argc == 3 ? argv[1] : "cuda:0";
        const char *tpSpec = argc == 3 ? argv[2] : (FastllmCudaGetDeviceCount() >= 3 ? "0,1,2" : "0,1");
        const std::string referenceSpec(referenceDevice);
        const int referenceIndex = std::stoi(referenceSpec.rfind("cuda:", 0) == 0
            ? referenceSpec.substr(5) : referenceSpec);
        CheckUnroundedPartial(referenceIndex);
        CheckProjectionRounding(referenceIndex);
        CheckBf16ColumnAccumulation(referenceIndex);
        setenv("FASTLLM_QWEN35_VISION_DEVICE", "auto", 1);
        unsetenv("FASTLLM_QWEN35_THREAD_TP");
        setenv("FASTLLM_TP", referenceDevice, 1);
        std::vector<float> image, video;
        {
            VisionFixture model;
            image = model.Encode(false);
            video = model.Encode(true);
        }
        setenv("FASTLLM_TP", tpSpec, 1);
        VisionFixture model;
        if (std::string(tpSpec).find(',') != std::string::npos) {
            model.CheckUnsupportedStorage(false);
            model.CheckUnsupportedStorage(true);
        }
        // The prefill reservation must also succeed for uneven/empty ranks.
        setenv("FASTLLM_QWEN35_MM_MAX_PATCHES", "48", 1);
        model.Prepare();
        auto first = model.Encode(false, false, true);
        Compare(first, image, "image");
        Compare(model.Encode(false, false, true), first, "cache_hit");
        auto reversed = model.Encode(false, true, true);
        auto expected = first;
        std::rotate(expected.begin(), expected.begin() + expected.size() / 2, expected.end());
        Compare(reversed, expected, "cache_reordered");
        Compare(model.Encode(true), video, "video");
        std::cout << "PASS vision TP " << tpSpec << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
