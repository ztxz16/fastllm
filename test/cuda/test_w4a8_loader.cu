#include "model.h"
#include "test_utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "cutlass/float8.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using fastllm::cuda_test::Expect;

float BFloat16ToFloat(uint16_t bits) {
    __nv_bfloat16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return __bfloat162float(value);
}

float RoundBFloat16(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

float RoundFp8(float value) {
    return (float)cutlass::float_e4m3_t(value);
}

std::vector<float> MakeInput(int inFeatures) {
    std::vector<float> input(inFeatures);
    for (int i = 0; i < inFeatures; ++i) {
        input[i] = std::sin((float)i * 0.013f) * 0.75f +
                   std::cos((float)i * 0.007f) * 0.25f;
    }
    return input;
}

std::vector<float> CpuReference(const fastllm::CompressedW4A8LinearProbe &probe,
                                const std::vector<float> &input) {
    int outFeatures = probe.logicalShape[0];
    int inFeatures = probe.logicalShape[1];
    int groups = inFeatures / fastllm::COMPRESSED_W4A8_GROUP_SIZE;

    std::vector<float> quantizedInput(inFeatures);
    float inputAbsMax = 0.0f;
    for (int i = 0; i < inFeatures; ++i) {
        inputAbsMax = std::max(inputAbsMax, std::fabs(RoundBFloat16(input[i])));
    }
    float tokenScale = std::max(inputAbsMax, 1.0e-10f) / 448.0f;
    for (int i = 0; i < inFeatures; ++i) {
        float q = RoundBFloat16(input[i]) / tokenScale;
        q = std::max(-448.0f, std::min(448.0f, q));
        quantizedInput[i] = RoundFp8(q) * tokenScale;
    }

    std::vector<float> decodedScales(probe.groupScales.size());
    for (int out = 0; out < outFeatures; ++out) {
        float scaleAbsMax = 0.0f;
        for (int group = 0; group < groups; ++group) {
            scaleAbsMax = std::max(
                scaleAbsMax,
                std::fabs(BFloat16ToFloat(probe.groupScales[(size_t)out * groups + group])));
        }
        float channelScale = std::max(scaleAbsMax, 1.0e-10f) / 56.0f;
        for (int group = 0; group < groups; ++group) {
            size_t index = (size_t)out * groups + group;
            decodedScales[index] =
                RoundFp8(BFloat16ToFloat(probe.groupScales[index]) / channelScale) *
                channelScale;
        }
    }

    std::vector<float> output(outFeatures, 0.0f);
    size_t rowBytes = (size_t)inFeatures / 2;
    for (int out = 0; out < outFeatures; ++out) {
        float acc = 0.0f;
        for (int in = 0; in < inFeatures; ++in) {
            uint8_t packed = probe.packedWeight[(size_t)out * rowBytes + in / 2];
            int q = ((packed >> ((in & 1) * 4)) & 0xF) - 8;
            float scale = decodedScales[(size_t)out * groups +
                                        in / fastllm::COMPRESSED_W4A8_GROUP_SIZE];
            acc += quantizedInput[in] * q * scale;
        }
        output[out] = RoundBFloat16(acc);
    }
    return output;
}

std::vector<float> DataToFloat(fastllm::Data &data) {
    data.ToDevice(fastllm::DataDevice::CPU);
    const uint16_t *src = reinterpret_cast<const uint16_t*>(data.cpuData);
    std::vector<float> output(data.Count(0));
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = BFloat16ToFloat(src[i]);
    }
    return output;
}

bool RunMissingCompanionTest() {
    std::string path = "/tmp/fastllm_w4a8_loader_" + std::to_string((long long)getpid());
    mkdir(path.c_str(), 0700);
    std::string config = R"({"quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{"group_0":{"targets":["Linear"],"weights":{"type":"int","num_bits":4,"strategy":"group","group_size":128,"symmetric":true,"dynamic":false,"actorder":"weight"},"input_activations":{"type":"float","num_bits":8,"strategy":"token","symmetric":true,"dynamic":true,"actorder":null}}}}})";
    std::ofstream(path + "/config.json") << config;

    std::string header = R"({"layer.weight_packed":{"dtype":"I32","shape":[128,16],"data_offsets":[0,8192]}})";
    std::ofstream weights(path + "/model.safetensors", std::ios::binary);
    uint64_t headerSize = header.size();
    weights.write(reinterpret_cast<const char*>(&headerSize), sizeof(headerSize));
    weights.write(header.data(), header.size());
    weights.close();

    fastllm::CompressedW4A8LinearProbe probe;
    std::string error;
    bool accepted = fastllm::InspectCompressedW4A8Linear(path, "layer", probe, error);
    bool ok = !accepted && error.find("missing companion scale") != std::string::npos;
    std::remove((path + "/model.safetensors").c_str());
    std::remove((path + "/config.json").c_str());
    rmdir(path.c_str());
    std::printf("[W4A8 loader] missing companion %s: %s\n",
                ok ? "PASS" : "FAIL", error.c_str());
    return ok;
}

bool RunRealCheckpointTest(const std::string &modelPath,
                           const std::string &tensorPrefix) {
    fastllm::CompressedW4A8LinearProbe probe;
    std::string error;
    Expect(fastllm::InspectCompressedW4A8Linear(
               modelPath, tensorPrefix, probe, error), error);
    Expect(probe.logicalShape.size() == 2 && probe.packedShape.size() == 2 &&
           probe.scaleShape.size() == 2,
           "W4A8 loader did not return three valid tensor shapes.");

    int outFeatures = probe.logicalShape[0];
    int inFeatures = probe.logicalShape[1];
    Expect(probe.packedShape[0] == outFeatures && probe.packedShape[1] * 8 == inFeatures,
           "weight_packed shape disagrees with weight_shape.");
    Expect(probe.scaleShape[0] == outFeatures && probe.scaleShape[1] * 128 == inFeatures,
           "weight_scale shape disagrees with weight_shape.");
    Expect(probe.packedWeight.size() == (size_t)outFeatures * inFeatures / 2,
           "weight_packed byte count is invalid.");
    Expect(probe.groupScales.size() == (size_t)outFeatures * (inFeatures / 128),
           "weight_scale element count is invalid.");
    Expect(probe.lmHeadBfloat16, "lm_head is not stored as BF16.");
    std::printf("[W4A8 loader] config/tensors PASS logical=[%d,%d] packed=[%d,%d] scale=[%d,%d]\n",
                outFeatures, inFeatures, probe.packedShape[0], probe.packedShape[1],
                probe.scaleShape[0], probe.scaleShape[1]);

    if (FastllmCudaRuntimeArch() != 90) {
        std::printf("[W4A8 loader] CUDA preprocess skipped: runtime GPU is not SM90\n");
        return true;
    }

    std::vector<float> input = MakeInput(inFeatures);
    fastllm::Data inputData(fastllm::DataType::BFLOAT16, {1, inFeatures}, input);
    inputData.ToDevice(fastllm::DataDevice::CUDA);
    fastllm::Data weight(fastllm::DataType::INT4_W4A8, {outFeatures, inFeatures});
    weight.InitW4A8Weight(
        fastllm::W4A8WeightEncoding::COMPRESSED_TENSORS_UINT4B8);
    weight.SetW4A8GroupScales(probe.groupScales.data(), probe.groupScales.size());
    weight.Allocate(false);
    std::memcpy(weight.cpuData, probe.packedWeight.data(), probe.packedWeight.size());
    weight.ToDevice(fastllm::DataDevice::CUDA);
    fastllm::Data output(fastllm::DataType::BFLOAT16, {1, outFeatures});
    output.Allocate(false);
    fastllm::Data bias;

    bool used = TryCudaCutlassW4A8(
        inputData, weight, bias, output, 1, inFeatures, outFeatures);
    Expect(used, "Real checkpoint did not enter the W4A8 production path.");
    cudaError_t state = cudaDeviceSynchronize();
    Expect(state == cudaSuccess, cudaGetErrorString(state));

    std::vector<float> expected = CpuReference(probe, input);
    std::vector<float> actual = DataToFloat(output);
    auto result = fastllm::cuda_test::CompareVectors(
        expected, actual, 2.5e-1f, 3.0e-2f, 5.0f);
    fastllm::cuda_test::PrintCompareResult(
        "W4A8 loader real checkpoint", result, 2.5e-1f, 3.0e-2f, 5.0f);
    return result.passed;
}

} // namespace

int main(int argc, char **argv) {
    try {
        bool ok = RunMissingCompanionTest();
        const char *envPath = std::getenv("FASTLLM_W4A8_TEST_MODEL");
        std::string modelPath = argc > 1 ? argv[1] : (envPath == nullptr ? "" : envPath);
        const char *envPrefix = std::getenv("FASTLLM_W4A8_TEST_TENSOR");
        std::string tensorPrefix = envPrefix == nullptr
            ? "model.layers.0.self_attn.q_proj" : envPrefix;
        if (modelPath.empty()) {
            std::printf("[W4A8 loader] real checkpoint skipped: set FASTLLM_W4A8_TEST_MODEL\n");
        } else {
            ok = RunRealCheckpointTest(modelPath, tensorPrefix) && ok;
        }
        std::printf("[W4A8 loader] %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception &exception) {
        std::printf("[W4A8 loader] exception: %s\n", exception.what());
        return 1;
    }
}
