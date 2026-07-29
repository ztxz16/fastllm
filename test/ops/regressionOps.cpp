#include "executor.h"
#include "fastllm.h"
#include "model.h"
#include "models/basellm.h"
#include "devices/cpu/computeutils.h"
#include "gguf.h"

#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif

#if defined(USE_CUDA) && !defined(USE_ROCM)
#include <cuda_runtime_api.h>
#endif

#ifdef USE_CUDA
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
    void Expect(bool condition, const std::string &message);

    class ScopedTempDirectory {
    public:
        explicit ScopedTempDirectory(const std::string &prefix) {
            uint64_t nonce = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            for (int attempt = 0; attempt < 100; attempt++) {
                std::error_code error;
                auto candidate = std::filesystem::temp_directory_path() /
                    (prefix + std::to_string(nonce) + "_" + std::to_string(attempt));
                if (std::filesystem::create_directory(candidate, error)) {
                    path = std::move(candidate);
                    return;
                }
            }
            throw std::runtime_error("failed to create unique regression temp directory.");
        }

        ~ScopedTempDirectory() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }

        const std::filesystem::path &Path() const {
            return path;
        }

    private:
        std::filesystem::path path;
    };

    void RunBFloat16Q8KConversionRegression() {
        constexpr size_t rows = 41;
        constexpr size_t columns = QK_K * 3;
        std::vector<uint16_t> input(rows * columns);
        std::vector<float> referenceInput(rows * columns);
        uint32_t state = 0x4b1d2a39u;
        for (size_t index = 0; index < input.size(); index++) {
            state = state * 1664525u + 1013904223u;
            float value =
                ((int32_t)(state % 400003u) - 200001) / 2048.0f;
            if (index < QK_K) {
                value = 0.0f;
            } else if (index % 257 == 0) {
                value = 0.0f;
            } else if (index % 263 == 0) {
                value = -0.0f;
            } else if (index % 269 == 0) {
                value = 1.0f / 65536.0f;
            } else if (index % 271 == 0) {
                value = -127.5f;
            }
            input[index] = fastllm::Float32ToBFloat16RNEBits(value);
            referenceInput[index] =
                fastllm::BFloat16BitsToFloat32(input[index]);
        }

        const std::vector<ggml_type> types = {
            GGML_TYPE_Q8_K, GGML_TYPE_Q8_K32
        };
        for (ggml_type type : types) {
            const fastllm::DataType dataType = (fastllm::DataType)(
                (int)fastllm::DataType::DATA_GGUF_FORMAT + (int)type);
            const size_t bytes =
                fastllm::GetDataBytes(dataType, rows, columns);
            std::vector<uint8_t> expected(bytes, 0xa5);
            std::vector<uint8_t> actual(bytes, 0x5a);
            std::vector<uint8_t> threaded(bytes, 0x3c);
            fastllm::ConvertFromFloat32(
                expected.data(), dataType, referenceInput.data(),
                rows, columns);
            fastllm::ConvertFromBFloat16(
                actual.data(), dataType, input.data(), rows, columns);
            fastllm::RunMultiThreadConvertFromBFloat16(
                threaded.data(), dataType, input.data(), rows, columns,
                fastllm::GetAlivePool());
            const std::string typeName = ggml_type_name(type);
            Expect(memcmp(expected.data(), actual.data(), bytes) == 0,
                   "direct BF16 to " + typeName +
                   " conversion differs from FLOAT32 reference");
            Expect(memcmp(expected.data(), threaded.data(), bytes) == 0,
                   "multi-thread BF16 to " + typeName +
                   " conversion differs from FLOAT32 reference");
        }
    }

    class MoeAtypeConfigTestModel final : public fastllm::basellm {
    public:
        std::string MakeInput(const std::string &, int, const std::string &) override {
            return "";
        }

        std::string MakeHistory(const std::string &, int, const std::string &,
                                const std::string &) override {
            return "";
        }
    };

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

    void ExpectFloatNear(const std::vector<float> &expected,
                         const std::vector<float> &actual,
                         float atol, float rtol, const std::string &name);

    void ExpectMinOutputLengthLogits(const float *logits, int batch, int stride,
                                     int eosTokenId, const std::vector<int> &resetLengths,
                                     const std::string &device) {
        constexpr float suppressedLogitUpperBound = -1.0e20f;
        for (int b = 0; b < batch; b++) {
            float eosLogit = logits[b * stride + eosTokenId];
            if (resetLengths[b] > 0) {
                Expect(eosLogit <= suppressedLogitUpperBound,
                       device + " EOS was not suppressed for a request below min_tokens.");
            } else {
                Expect(eosLogit == 7.0f,
                       device + " EOS was suppressed for a request that met min_tokens.");
            }
            for (int token = 0; token < stride; token++) {
                if (token != eosTokenId) {
                    Expect(logits[b * stride + token] == -3.0f,
                           device + " reset changed a non-EOS logit.");
                }
            }
        }
    }

    void RunMoeAtypeConfigRegression() {
        MoeAtypeConfigTestModel model;
        Expect(!model.useCustomMoeAtype, "moe_atype should default to auto.");

        model.SetMoeAtype(fastllm::DataType::FLOAT32);
        Expect(model.useCustomMoeAtype && model.moeAtype == fastllm::DataType::FLOAT32,
               "explicit float32 moe_atype was not preserved.");

        model.SetMoeAtype(fastllm::DataType::DATA_AUTO_NONE);
        Expect(!model.useCustomMoeAtype,
               "auto moe_atype did not clear the explicit configuration.");
        Expect(model.moeAtype == fastllm::DataType::FLOAT32,
               "auto moe_atype did not restore the compatibility sentinel.");

        model.SetMoeAtype(fastllm::DataType::BFLOAT16);
        Expect(model.useCustomMoeAtype && model.moeAtype == fastllm::DataType::BFLOAT16,
               "explicit bfloat16 moe_atype was not preserved.");
        model.SetMoeAtype(fastllm::DataType::DATA_AUTO_NONE);
        Expect(!model.useCustomMoeAtype,
               "auto moe_atype did not reset a previous bfloat16 configuration.");
    }

    void WriteLagunaAutoDtypeFixture(const std::filesystem::path &path) {
        const std::string config = R"JSON({
            "model_type": "laguna",
            "architectures": ["LagunaForCausalLM"],
            "eos_token_id": 2,
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "format": "nvfp4-pack-quantized",
                "quant_method": "compressed-tensors"
            },
            "num_hidden_layers": 1,
            "hidden_size": 4,
            "head_dim": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_attention_heads_per_layer": [2],
            "layer_types": ["full_attention"],
            "intermediate_size": 4,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "num_experts": 1,
            "num_experts_per_tok": 1,
            "max_position_embeddings": 8,
            "sliding_window": 4
        })JSON";
        {
            std::ofstream output(path / "config.json");
            Expect(output.good(), "failed to create Laguna dtype fixture config.");
            output << config;
        }

        const std::string packedName =
            "model.layers.0.mlp.experts.0.down_proj.weight_packed";
        const std::string scaleName =
            "model.layers.0.mlp.experts.0.down_proj.weight_scale";
        std::string header =
            "{\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[0,16]},"
            "\"" + packedName + "\":{\"dtype\":\"U8\",\"shape\":[4,2],\"data_offsets\":[16,24]},"
            "\"" + scaleName + "\":{\"dtype\":\"F8_E8M0\",\"shape\":[1,1],\"data_offsets\":[24,25]}}";
        while (header.size() % 8 != 0) {
            header.push_back(' ');
        }

        std::ofstream output(path / "model.safetensors", std::ios::binary);
        Expect(output.good(), "failed to create Laguna dtype fixture weights.");
        uint64_t headerSize = header.size();
        output.write(reinterpret_cast<const char*>(&headerSize), sizeof(headerSize));
        output.write(header.data(), header.size());
        const std::vector<uint8_t> payload(25, 0);
        output.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        Expect(output.good(), "failed to write Laguna dtype fixture weights.");
    }

    void RunLagunaNVFP4AutoDtypeRegression() {
        ScopedTempDirectory temp("fastllm_laguna_nvfp4_auto_");
        WriteLagunaAutoDtypeFixture(temp.Path());

        auto model = fastllm::CreateLLMModelFromHF(
            temp.Path().string(), fastllm::DataType::DATA_AUTO_SOURCE,
            -1, true);
        Expect(model != nullptr && model->model_type == "laguna",
               "Laguna dtype fixture did not create a Laguna model.");

        const std::string packedWeight =
            "model.layers.0.moe.experts.0.down_proj.weight";
        auto packed = model->weight.weight.find(packedWeight);
        Expect(packed != model->weight.weight.end(),
               "packed Laguna expert weight was not loaded.");
        Expect(packed->second.dataType == fastllm::DataType::NVFP4,
               "Laguna auto dtype did not preserve packed NVFP4 expert weight.");
        Expect(packed->second.blockK == 4 && packed->second.blockM == 4 &&
                   packed->second.scales.empty(),
               "Laguna auto dtype did not preserve compact NVFP4 metadata.");

        auto unquantized = model->weight.weight.find("lm_head.weight");
        Expect(unquantized != model->weight.weight.end(),
               "unquantized Laguna linear weight was not loaded.");
        Expect(unquantized->second.dataType == fastllm::DataType::FLOAT16,
               "Laguna auto dtype unexpectedly overrode the standard linear dtype.");
    }

    void WriteLagunaPackedInt4Fixture(const std::filesystem::path &path) {
        const std::string config = R"JSON({
            "model_type": "laguna",
            "architectures": ["LagunaForCausalLM"],
            "eos_token_id": 2,
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "format": "pack-quantized",
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "format": "pack-quantized",
                        "weights": {
                            "group_size": 32,
                            "num_bits": 4,
                            "strategy": "group",
                            "symmetric": true,
                            "type": "int"
                        }
                    }
                }
            },
            "num_hidden_layers": 1,
            "hidden_size": 32,
            "head_dim": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "num_attention_heads_per_layer": [4],
            "layer_types": ["full_attention"],
            "intermediate_size": 32,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 32,
            "num_experts": 1,
            "num_experts_per_tok": 1,
            "max_position_embeddings": 32,
            "sliding_window": 16
        })JSON";
        {
            std::ofstream output(path / "config.json");
            Expect(output.good(), "failed to create Laguna packed INT4 fixture config.");
            output << config;
        }

        constexpr int rows = 32;
        constexpr int columns = 64;
        constexpr int groups = columns / 32;
        constexpr int packedColumns = columns / 8;
        const std::string packedName =
            "model.layers.0.mlp.experts.0.down_proj.weight_packed";
        const std::string scaleName =
            "model.layers.0.mlp.experts.0.down_proj.weight_scale";
        constexpr uint64_t lmHeadBytes = 2 * 32 * sizeof(uint16_t);
        constexpr uint64_t packedBytes = rows * packedColumns * sizeof(uint32_t);
        constexpr uint64_t scaleBytes = rows * groups * sizeof(uint16_t);
        std::string header =
            "{\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[2,32],\"data_offsets\":[0," +
            std::to_string(lmHeadBytes) + "]},\"" + packedName +
            "\":{\"dtype\":\"I32\",\"shape\":[32,8],\"data_offsets\":[" +
            std::to_string(lmHeadBytes) + "," + std::to_string(lmHeadBytes + packedBytes) +
            "]},\"" + scaleName +
            "\":{\"dtype\":\"BF16\",\"shape\":[32,2],\"data_offsets\":[" +
            std::to_string(lmHeadBytes + packedBytes) + "," +
            std::to_string(lmHeadBytes + packedBytes + scaleBytes) + "]}}";
        while (header.size() % 8 != 0) {
            header.push_back(' ');
        }

        std::vector<uint16_t> lmHead(2 * 32, 0);
        std::vector<uint32_t> packed(rows * packedColumns, 0);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                int signedValue = (row * 3 + column) % 16 - 8;
                uint32_t storedValue = (uint32_t)(signedValue + 8);
                packed[row * packedColumns + column / 8] |=
                    storedValue << ((column % 8) * 4);
            }
        }
        std::vector<uint16_t> scales(rows * groups);
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < groups; group++) {
                float value = 0.125f * (float)((row + group) % 7 + 1);
                uint32_t bits;
                memcpy(&bits, &value, sizeof(bits));
                scales[row * groups + group] = (uint16_t)(bits >> 16);
            }
        }

        std::ofstream output(path / "model.safetensors", std::ios::binary);
        Expect(output.good(), "failed to create Laguna packed INT4 fixture weights.");
        uint64_t headerSize = header.size();
        output.write(reinterpret_cast<const char*>(&headerSize), sizeof(headerSize));
        output.write(header.data(), header.size());
        output.write(reinterpret_cast<const char*>(lmHead.data()), lmHeadBytes);
        output.write(reinterpret_cast<const char*>(packed.data()), packedBytes);
        output.write(reinterpret_cast<const char*>(scales.data()), scaleBytes);
        Expect(output.good(), "failed to write Laguna packed INT4 fixture weights.");
    }

    void RunLagunaPackedInt4AutoDtypeRegression() {
        ScopedTempDirectory temp("fastllm_laguna_packed_int4_auto_");
        WriteLagunaPackedInt4Fixture(temp.Path());

        auto model = fastllm::CreateLLMModelFromHF(
            temp.Path().string(), fastllm::DataType::DATA_AUTO_SOURCE,
            -1, true);
        Expect(model != nullptr && model->model_type == "laguna",
               "Laguna packed INT4 fixture did not create a Laguna model.");

        const std::string weightName =
            "model.layers.0.moe.experts.0.down_proj.weight";
        auto packed = model->weight.weight.find(weightName);
        Expect(packed != model->weight.weight.end(),
               "packed Laguna INT4 expert weight was not loaded.");
        const fastllm::Data &weight = packed->second;
        Expect(weight.dataType == fastllm::DataType::INT4_GROUP32,
               "Laguna auto dtype did not preserve compressed-tensors INT4 weight.");
        Expect(weight.dims == std::vector<int>({32, 64}) &&
                   weight.groupCnt == 32 && weight.group == 2,
               "Laguna packed INT4 logical shape or group metadata is incorrect.");
        Expect(weight.cpuData != nullptr && weight.GetBytes() == 32 * 36 &&
                   weight.scales.empty() && weight.mins.empty() &&
                   weight.zeros.empty() && weight.weightSum.empty(),
               "Laguna packed INT4 data is not using compact inline scales.");

        const uint8_t *data = (const uint8_t*)weight.cpuData;
        for (int row = 0; row < 32; row++) {
            const uint8_t *rowData = data + row * 36;
            for (int group = 0; group < 2; group++) {
                const uint8_t *block = rowData +
                    fastllm::GetInt4Group32DataOffset(group, 2);
                float expectedScale =
                    0.125f * (float)((row + group) % 7 + 1);
                uint16_t scaleBits;
                memcpy(&scaleBits, rowData +
                           fastllm::GetInt4Group32ScaleOffset(group, 2),
                       sizeof(scaleBits));
                Expect(std::fabs(fastllm::BFloat16BitsToFloat32(scaleBits) -
                                 expectedScale) < 1e-6f,
                       "Laguna packed INT4 inline BF16 scale is incorrect.");
                for (int localColumn = 0; localColumn < 32; localColumn++) {
                    const int column = group * 32 + localColumn;
                    uint8_t byte = block[localColumn / 2];
                    int storedValue = (localColumn & 1) ?
                        (byte & 0xF) : (byte >> 4);
                    int expectedValue = (row * 3 + column) % 16;
                    Expect(storedValue == expectedValue,
                           "Laguna packed INT4 nibble order is incorrect.");
                }
            }
        }
    }

    void RunPerRequestMinOutputLengthRegression() {
        MoeAtypeConfigTestModel model;
        model.block_cnt = 3;
        model.kvCacheId = 1;
        model.eos_token_id = 2;

        const int batch = 4;
        const int generatedTokens[batch] = {2, 4, 6, 0};
        const int inputTokens[batch] = {10, 20, 30, 40};
        std::vector<fastllm::Data> keys;
        std::vector<fastllm::Data> values;
        std::vector<std::pair<fastllm::Data*, fastllm::Data*> > pastKeyValues;
        keys.reserve(batch * model.block_cnt);
        values.reserve(batch * model.block_cnt);
        pastKeyValues.reserve(batch * model.block_cnt);
        for (int b = 0; b < batch; b++) {
            int cacheLen = inputTokens[b] + generatedTokens[b];
            for (int layer = 0; layer < model.block_cnt; layer++) {
                keys.emplace_back(fastllm::DataType::FLOAT32,
                                  std::vector<int>{1, cacheLen, 1});
                values.emplace_back(fastllm::DataType::FLOAT32,
                                    std::vector<int>{1, cacheLen, 1});
                pastKeyValues.push_back({&keys.back(), &values.back()});
            }
        }

        std::vector<fastllm::GenerationConfig> configs(batch);
        for (int b = 0; b < batch; b++) {
            configs[b].input_token_length = inputTokens[b];
            configs[b].output_token_least = b == batch - 1 ? 0 : 5;
        }
        std::vector<int> resetLengths = model.GetMinOutputResetLengths(
            batch, pastKeyValues, configs);
        Expect(resetLengths == std::vector<int>({3, 1, -1, 0}),
               "minimum output length did not use each request's KV cache length.");

        const int stride = 5;
        std::vector<float> logits(batch * stride, -3.0f);
        for (int b = 0; b < batch; b++) {
            logits[b * stride + model.eos_token_id] = 7.0f;
        }
        fastllm::Data logitsData(fastllm::DataType::FLOAT32,
                                 {batch, stride}, logits);
        model.ResetLogitsOfEOS(batch, &logitsData, pastKeyValues, configs);
        ExpectMinOutputLengthLogits((float*)logitsData.cpuData, batch, stride,
                                    model.eos_token_id, resetLengths, "CPU");

#ifdef USE_CUDA
        if (fastllm::HasDeviceType("cuda")) {
            fastllm::Data cudaLogitsData(fastllm::DataType::FLOAT32,
                                         {batch, stride}, logits);
            cudaLogitsData.ToDevice(fastllm::DataDevice::CUDA);
            model.ResetLogitsOfEOS(batch, &cudaLogitsData, pastKeyValues, configs);
            cudaLogitsData.ToDevice(fastllm::DataDevice::CPU);
            ExpectMinOutputLengthLogits((float*)cudaLogitsData.cpuData, batch, stride,
                                        model.eos_token_id, resetLengths, "CUDA");
        }
#endif
    }

#ifdef USE_CUDA
    void RunCudaLinearDataTypeCapabilityRegression() {
        using fastllm::DataType;
        using fastllm::IsCudaLinearDataTypeSupported;

        Expect(IsCudaLinearDataTypeSupported(DataType::FLOAT16, DataType::INT4_GROUP128,
                                             DataType::FLOAT32),
               "FLOAT16 + INT4_GROUP128 should be supported by CUDA Linear.");
        Expect(IsCudaLinearDataTypeSupported(DataType::FLOAT16, DataType::INT4_GROUP32,
                                             DataType::FLOAT32) &&
                   IsCudaLinearDataTypeSupported(DataType::BFLOAT16, DataType::INT4_GROUP32,
                                                 DataType::FLOAT32) &&
                   IsCudaLinearDataTypeSupported(DataType::FLOAT32, DataType::INT4_GROUP32,
                                                 DataType::FLOAT32),
               "compact INT4_GROUP32 should support FP16/BF16/FP32 CUDA Linear.");
        Expect(!IsCudaLinearDataTypeSupported(DataType::FLOAT32, DataType::INT4_GROUP128,
                                              DataType::FLOAT32),
               "FLOAT32 + INT4_GROUP128 should be rejected by CUDA Linear.");
        Expect(!IsCudaLinearDataTypeSupported(DataType::BFLOAT16, DataType::INT8,
                                              DataType::FLOAT32),
               "BFLOAT16 + INT8 should be rejected by CUDA Linear.");
        Expect(IsCudaLinearDataTypeSupported(DataType::BFLOAT16, DataType::FLOAT16,
                                             DataType::FLOAT32),
               "BFLOAT16 + FLOAT16 should be supported by CUDA Linear.");
        Expect(IsCudaLinearDataTypeSupported(DataType::FLOAT16, DataType::FP8_E4M3_PERCHANNEL,
                                             DataType::FLOAT32),
               "FLOAT16 + FP8_E4M3_PERCHANNEL should be supported by CUDA Linear.");
        Expect(IsCudaLinearDataTypeSupported(DataType::FLOAT32, DataType::FP8_E4M3_PERCHANNEL,
                                             DataType::FLOAT32),
               "FLOAT32 + FP8_E4M3_PERCHANNEL should be supported by CUDA Linear.");
        Expect(IsCudaLinearDataTypeSupported(DataType::BFLOAT16, DataType::FP8_E4M3_PERCHANNEL,
                                             DataType::FLOAT32),
               "BFLOAT16 + FP8_E4M3_PERCHANNEL should be supported by CUDA Linear.");
        Expect(!IsCudaLinearDataTypeSupported(DataType::FLOAT16, DataType::INT8_PERCHANNEL,
                                              DataType::FLOAT32),
               "internal per-channel weights should be rejected by CUDA Linear.");
        Expect(!IsCudaLinearDataTypeSupported(DataType::FLOAT16, DataType::FLOAT16,
                                              DataType::FLOAT16),
               "non-FLOAT32 bias should be rejected by CUDA Linear.");
    }
#endif

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

    void RunCpuPackedInt4Group32KernelRegression() {
        constexpr int n = 31;
        constexpr int m = 128;
        constexpr int k = 5;
        constexpr int groupCnt = 32;
        constexpr int groups = m / groupCnt;
        const size_t inputRowBytes = fastllm::GetDataBytes(
            fastllm::DataType::INF_INT8_GROUP32, 1, m);
        const size_t weightRowBytes = fastllm::GetDataBytes(
            fastllm::DataType::INT4_GROUP32, 1, m);
        constexpr size_t inputGroupBytes = groupCnt + sizeof(float) + sizeof(int);

        std::vector<uint8_t> input(n * inputRowBytes, 0);
        std::vector<uint8_t> weight(k * weightRowBytes, 0);
        std::vector<float> expected(n * k, 0.0f);
        std::vector<float> actual(n * k, 0.0f);

        for (int row = 0; row < n; row++) {
            for (int group = 0; group < groups; group++) {
                uint8_t *block = input.data() + row * inputRowBytes +
                                 group * inputGroupBytes;
                int8_t *values = (int8_t*)block;
                int sum = 0;
                for (int column = 0; column < groupCnt; column++) {
                    values[column] = (int8_t)(((row * 29 + group * 17 + column * 7) % 255) - 127);
                    sum += values[column];
                }
                float scale = 0.01f * (float)(row + group + 1);
                memcpy(block + groupCnt, &scale, sizeof(scale));
                memcpy(block + groupCnt + sizeof(float), &sum, sizeof(sum));
            }
        }
        for (int row = 0; row < k; row++) {
            for (int group = 0; group < groups; group++) {
                uint8_t *rowData = weight.data() + row * weightRowBytes;
                uint8_t *block = rowData +
                    fastllm::GetInt4Group32DataOffset(group, groups);
                for (int column = 0; column < groupCnt; column += 2) {
                    int high = (row * 5 + group * 3 + column) % 16;
                    int low = (row * 5 + group * 3 + column + 1) % 16;
                    block[column / 2] = (uint8_t)((high << 4) | low);
                }
                float scale = 0.025f * (float)(row + group + 1);
                uint16_t scaleBits = fastllm::Float32ToBFloat16RNEBits(scale);
                memcpy(rowData +
                           fastllm::GetInt4Group32ScaleOffset(group, groups),
                       &scaleBits, sizeof(scaleBits));
            }
        }

        for (int inputRow = 0; inputRow < n; inputRow++) {
            for (int weightRow = 0; weightRow < k; weightRow++) {
                float total = 0.0f;
                for (int group = 0; group < groups; group++) {
                    const uint8_t *inputBlock = input.data() + inputRow * inputRowBytes +
                                                group * inputGroupBytes;
                    const int8_t *inputValues = (const int8_t*)inputBlock;
                    float inputScale;
                    memcpy(&inputScale, inputBlock + groupCnt, sizeof(inputScale));
                    const uint8_t *weightRowData = weight.data() + weightRow * weightRowBytes;
                    const uint8_t *weightBlock = weightRowData +
                        fastllm::GetInt4Group32DataOffset(group, groups);
                    uint16_t weightScaleBits;
                    memcpy(&weightScaleBits, weightRowData +
                               fastllm::GetInt4Group32ScaleOffset(group, groups),
                           sizeof(weightScaleBits));
                    float weightScale = fastllm::BFloat16BitsToFloat32(weightScaleBits);
                    int dot = 0;
                    for (int column = 0; column < groupCnt; column++) {
                        uint8_t packed = weightBlock[column / 2];
                        int value = (column & 1) ? (packed & 0xF) : (packed >> 4);
                        dot += inputValues[column] * (value - 8);
                    }
                    total += dot * inputScale * weightScale;
                }
                expected[inputRow * k + weightRow] = total;
            }
        }

        // Exercise the dedicated n=1 decode path, every templated batched
        // remainder, and multiple row blocks. All of them share the optimized
        // compact pair-unpack primitive.
        for (int batch : {1, 2, 3, 4, 5, 6, 7, 8,
                          9, 15, 16, 17, 23, 24, 25, 31}) {
            std::fill(actual.begin(), actual.end(), 0.0f);
            Expect(fastllm::LinearINT8GROUP32_INT4GROUP32_Kernel(
                       input.data(), weight.data(), nullptr, actual.data(),
                       batch, m, k, 0, k),
                   "packed INT4_GROUP(32) kernel rejected a valid small batch.");
            std::vector<float> batchExpected(
                expected.begin(), expected.begin() + (size_t)batch * k);
            std::vector<float> batchActual(
                actual.begin(), actual.begin() + (size_t)batch * k);
            ExpectFloatNear(batchExpected, batchActual, 1e-4f, 1e-5f,
                            "packed INT4_GROUP(32) small-batch kernel");
        }

        // Cover the parallel FLOAT32 -> INF_INT8_GROUP32 conversion used by
        // larger prefill batches before the compact kernel runs.
        constexpr int parallelRows = 105;
        std::vector<float> floatInput((size_t)parallelRows * m);
        for (int row = 0; row < parallelRows; row++) {
            for (int column = 0; column < m; column++) {
                floatInput[(size_t)row * m + column] =
                    (float)((row * 31 + column * 11) % 113 - 56) / 37.0f;
            }
        }
        std::vector<uint8_t> referenceInput(
            fastllm::GetDataBytes(fastllm::DataType::INF_INT8_GROUP32,
                                  parallelRows, m));
        fastllm::ConvertFromFloat32(
            referenceInput.data(), fastllm::DataType::INF_INT8_GROUP32,
            floatInput.data(), parallelRows, m);
        std::vector<float> parallelExpected((size_t)parallelRows * k);
        Expect(fastllm::LinearINT8GROUP32_INT4GROUP32_Kernel(
                   referenceInput.data(), weight.data(), nullptr,
                   parallelExpected.data(), parallelRows, m, k, 0, k),
               "packed INT4_GROUP(32) reference kernel rejected a valid shape.");

        fastllm::Data compactWeight(fastllm::DataType::INT4_GROUP32, {k, m});
        compactWeight.Allocate();
        memcpy(compactWeight.cpuData, weight.data(), weight.size());
        std::vector<float> parallelActual((size_t)parallelRows * k);
        fastllm::AliveThreadPool *pool = fastllm::GetAlivePool();
        fastllm::RunLinearFloat32Int4Group32(
            floatInput.data(), compactWeight, parallelActual.data(), nullptr,
            parallelRows, m, k, pool, 0, (int)pool->threads.size());
        ExpectFloatNear(parallelExpected, parallelActual, 1e-4f, 1e-5f,
                        "parallel compact INT4_GROUP(32) linear");
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
        if (data.dataType != fastllm::DataType::FLOAT32) {
            fastllm::ToDataTypeForceCPU(data, fastllm::DataType::FLOAT32);
        }
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
            Expect(std::isfinite(expected[i]) && std::isfinite(actual[i]),
                   name + " contains a non-finite value at index " + std::to_string(i));
            float diff = std::fabs(expected[i] - actual[i]);
            float limit = atol + rtol * std::fabs(expected[i]);
            if (diff > limit) {
                throw std::runtime_error(name + " mismatch at index " + std::to_string(i) +
                                         ": expected " + std::to_string(expected[i]) +
                                         ", got " + std::to_string(actual[i]));
            }
        }
    }

#ifdef USE_CUDA
    bool RegressionEnvFlagDefaultEnabled(const char *name, bool fallback) {
        const char *value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return fallback;
        }
        return std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0 &&
               std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "OFF") != 0;
    }

    fastllm::Data MakeCudaTensor(fastllm::DataType dataType, const std::vector<int> &dims,
                                 const std::vector<float> &values) {
        fastllm::Data data(dataType, dims, values);
        data.ToDevice(fastllm::DataDevice::CUDA);
        return data;
    }

    void RunCudaBFloat16SigmoidMulToRegression() {
        const std::vector<float> sigmoidValues = {
            -9.0f, -2.5f, -0.1f, 0.0f, 0.1f, 2.5f, 9.0f
        };
        fastllm::Data cpuSigmoidInput(
            fastllm::DataType::BFLOAT16, {(int)sigmoidValues.size()},
            sigmoidValues);
        fastllm::Data cpuSigmoidOutput;
        {
            ScopedFirstDevice device("cpu");
            fastllm::Sigmoid(cpuSigmoidInput, cpuSigmoidOutput);
        }
        fastllm::Data cudaSigmoidInput = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {(int)sigmoidValues.size()},
            sigmoidValues);
        fastllm::Data cudaSigmoidOutput;
        {
            ScopedFirstDevice device("cuda:0");
            fastllm::Sigmoid(cudaSigmoidInput, cudaSigmoidOutput);
        }
        ExpectFloatNear(ToFloatVector(cpuSigmoidOutput),
                        ToFloatVector(cudaSigmoidOutput),
                        1.0f / 256.0f, 0.0f,
                        "CUDA BF16 Sigmoid output");

        const std::vector<float> inputValues = {
            -1.5f, -0.75f, -0.25f, 0.25f, 0.75f, 1.5f
        };
        const std::vector<std::pair<std::vector<int>, std::vector<float>>>
            multipliers = {
                {{6}, {0.5f, -0.25f, 1.25f, -1.0f, 0.75f, 2.0f}},
                {{1}, {-0.625f}},
                {{2}, {0.5f, -1.25f}},
            };
        for (const auto &multiplier : multipliers) {
            fastllm::Data cpuInput(
                fastllm::DataType::BFLOAT16, {2, 3}, inputValues);
            fastllm::Data cpuMultiplier(
                fastllm::DataType::BFLOAT16, multiplier.first,
                multiplier.second);
            {
                ScopedFirstDevice device("cpu");
                fastllm::MulTo(cpuInput, cpuMultiplier);
            }

            fastllm::Data cudaInput = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, {2, 3}, inputValues);
            fastllm::Data cudaMultiplier = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, multiplier.first,
                multiplier.second);
            {
                ScopedFirstDevice device("cuda:0");
                fastllm::MulTo(cudaInput, cudaMultiplier);
            }
            ExpectFloatNear(ToFloatVector(cpuInput), ToFloatVector(cudaInput),
                            0.0f, 0.0f, "CUDA BF16 MulTo output");
        }
    }

    void ExpectCudaTensorMeta(const fastllm::Data &data, fastllm::DataType dataType,
                              const std::vector<int> &dims, const std::string &name) {
        Expect(data.dataType == dataType, name + " dtype mismatch");
        Expect(data.dataDevice == fastllm::DataDevice::CUDA, name + " device mismatch");
        Expect(data.dims == dims, name + " shape mismatch");
        Expect(data.strides.size() == dims.size(), name + " stride rank mismatch");
        uint64_t expectedStride = 1;
        for (int i = (int)dims.size() - 1; i >= 0; i--) {
            Expect(data.strides[i] == expectedStride,
                   name + " is not dense at axis " + std::to_string(i));
            expectedStride *= (uint64_t)dims[i];
        }
        Expect(data.cudaData != nullptr, name + " CUDA buffer is null");
    }

    std::vector<float> MakeRegressionValues(int count, float seed, float scale = 1.0f) {
        std::vector<float> values(count);
        for (int i = 0; i < count; i++) {
            values[i] = scale * (std::sin((i + 1) * 0.071f + seed) +
                                 0.5f * std::cos((i + 3) * 0.043f - seed));
        }
        return values;
    }

    void RunCudaKimiK3PackedConvCacheRegression() {
        FastllmCudaSetDevice(0);
        constexpr int sequence = 8;
        constexpr int channels = 17;
        constexpr int history = 3;
        for (int batch : {1, 2}) {
            const std::vector<int> inputDims = {batch, sequence, channels};
            const std::vector<int> cacheDims = batch == 1 ?
                std::vector<int>({3, history, channels}) :
                std::vector<int>({3, batch, history, channels});
            const int inputItems = batch * sequence * channels;
            const int cacheItems = 3 * batch * history * channels;
            const std::vector<float> qValues =
                MakeRegressionValues(inputItems, 0.19f + batch, 0.7f);
            const std::vector<float> kValues =
                MakeRegressionValues(inputItems, 0.53f + batch, 0.6f);
            const std::vector<float> vValues =
                MakeRegressionValues(inputItems, 0.97f + batch, 0.5f);
            const std::vector<float> initialValues =
                MakeRegressionValues(cacheItems, 1.43f + batch, 0.4f);

            fastllm::Data cpuQ(
                fastllm::DataType::BFLOAT16, inputDims, qValues);
            fastllm::Data cpuK(
                fastllm::DataType::BFLOAT16, inputDims, kValues);
            fastllm::Data cpuV(
                fastllm::DataType::BFLOAT16, inputDims, vValues);
            fastllm::Data initialCache(
                fastllm::DataType::BFLOAT16, cacheDims, initialValues);
            const std::vector<std::vector<float>> roundedInputs = {
                ToFloatVector(cpuQ), ToFloatVector(cpuK),
                ToFloatVector(cpuV)};
            const std::vector<float> roundedInitial =
                ToFloatVector(initialCache);
            fastllm::Data cudaQ = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, inputDims, qValues);
            fastllm::Data cudaK = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, inputDims, kValues);
            fastllm::Data cudaV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, inputDims, vValues);

            for (int tokens = 1; tokens <= sequence; tokens++) {
                std::vector<float> expected = roundedInitial;
                for (int stream = 0; stream < 3; stream++) {
                    for (int batchIndex = 0; batchIndex < batch;
                         batchIndex++) {
                        for (int channel = 0; channel < channels;
                             channel++) {
                            const size_t cacheBase =
                                ((size_t)stream * batch + batchIndex) *
                                history * channels;
                            for (int slot = 0; slot < history; slot++) {
                                const int combined = tokens + slot;
                                if (combined < history) {
                                    expected[cacheBase +
                                             (size_t)slot * channels +
                                             channel] =
                                        roundedInitial[
                                            cacheBase +
                                            (size_t)combined * channels +
                                            channel];
                                } else {
                                    const int inputToken =
                                        combined - history;
                                    expected[cacheBase +
                                             (size_t)slot * channels +
                                             channel] =
                                        roundedInputs[stream][
                                            ((size_t)batchIndex * sequence +
                                             inputToken) * channels +
                                            channel];
                                }
                            }
                        }
                    }
                }

                fastllm::Data cpuCache(
                    fastllm::DataType::BFLOAT16, cacheDims, initialValues);
                {
                    ScopedFirstDevice device("cpu");
                    fastllm::KimiK3UpdatePackedConvCache(
                        cpuQ, cpuK, cpuV, history, tokens, cpuCache);
                }
                fastllm::Data cudaCache = MakeCudaTensor(
                    fastllm::DataType::BFLOAT16, cacheDims, initialValues);
                {
                    ScopedFirstDevice device("cuda:0");
                    fastllm::KimiK3UpdatePackedConvCache(
                        cudaQ, cudaK, cudaV, history, tokens, cudaCache);
                }
                FastllmCudaSyncCurrentThreadStream();
                const std::string suffix =
                    " batch=" + std::to_string(batch) +
                    " tokens=" + std::to_string(tokens);
                ExpectFloatNear(expected, ToFloatVector(cpuCache),
                                0.0f, 0.0f,
                                "CPU Kimi-K3 packed conv cache" + suffix);
                ExpectFloatNear(expected, ToFloatVector(cudaCache),
                                0.0f, 0.0f,
                                "CUDA Kimi-K3 packed conv cache" + suffix);
            }
        }
        std::cout << "CUDA Kimi-K3 packed conv cache regression: PASS\n";
    }

    void RunCudaKimiK3RecurrentKdaRegression() {
        FastllmCudaSetDevice(0);
        constexpr int batch = 1;
        constexpr int sequence = 8;
        constexpr int heads = 3;
        constexpr int dimension = 128;
        const std::vector<int> vectorDims = {
            batch, sequence, heads, dimension};
        const std::vector<int> betaDims = {batch, sequence, heads};
        const std::vector<int> stateDims = {
            batch, heads, dimension, dimension};

        const std::vector<float> qValues = MakeRegressionValues(
            batch * sequence * heads * dimension, 0.13f, 0.035f);
        const std::vector<float> kValues = MakeRegressionValues(
            batch * sequence * heads * dimension, 0.41f, 0.03f);
        const std::vector<float> vValues = MakeRegressionValues(
            batch * sequence * heads * dimension, 0.79f, 0.2f);
        const std::vector<float> gateValues = MakeRegressionValues(
            batch * sequence * heads * dimension, 1.17f, 0.25f);
        const std::vector<float> rawBetaValues = MakeRegressionValues(
            batch * sequence * heads, 1.61f, 0.4f);
        const std::vector<float> aLogValues = {-0.7f, -0.45f, -0.2f};
        const std::vector<float> dtBiasValues = MakeRegressionValues(
            heads * dimension, 2.03f, 0.08f);
        const std::vector<float> initialStateValues = MakeRegressionValues(
            batch * heads * dimension * dimension, 2.47f, 0.015f);

        fastllm::Data cpuQ(fastllm::DataType::BFLOAT16, vectorDims, qValues);
        fastllm::Data cpuK(fastllm::DataType::BFLOAT16, vectorDims, kValues);
        fastllm::Data cpuV(fastllm::DataType::BFLOAT16, vectorDims, vValues);
        fastllm::Data cpuGate(
            fastllm::DataType::BFLOAT16, vectorDims, gateValues);
        fastllm::Data cpuRawBeta(
            fastllm::DataType::FLOAT32, betaDims, rawBetaValues);
        fastllm::Data cpuALog(
            fastllm::DataType::FLOAT32, {heads}, aLogValues);
        fastllm::Data cpuDtBias(
            fastllm::DataType::FLOAT32, {heads, dimension}, dtBiasValues);
        fastllm::Data cpuState(
            fastllm::DataType::FLOAT32, stateDims, initialStateValues);
        fastllm::Data cpuOutput, cpuDecay, cpuBeta;
        {
            ScopedFirstDevice device("cpu");
            fastllm::KimiK3RecurrentKDA(
                cpuQ, cpuK, cpuV, cpuGate, cpuRawBeta, cpuALog, cpuDtBias,
                -5.0f, cpuState, cpuOutput, cpuDecay, cpuBeta);
        }

        fastllm::Data cudaQ = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, vectorDims, qValues);
        fastllm::Data cudaK = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, vectorDims, kValues);
        fastllm::Data cudaV = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, vectorDims, vValues);
        fastllm::Data cudaGate = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, vectorDims, gateValues);
        fastllm::Data cudaRawBeta = MakeCudaTensor(
            fastllm::DataType::FLOAT32, betaDims, rawBetaValues);
        fastllm::Data cudaALog = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {heads}, aLogValues);
        fastllm::Data cudaDtBias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {heads, dimension}, dtBiasValues);
        fastllm::Data cudaState = MakeCudaTensor(
            fastllm::DataType::FLOAT32, stateDims, initialStateValues);
        fastllm::Data cudaOutput, cudaDecay, cudaBeta;
        {
            ScopedFirstDevice device("cuda:0");
            fastllm::KimiK3RecurrentKDA(
                cudaQ, cudaK, cudaV, cudaGate, cudaRawBeta, cudaALog,
                cudaDtBias, -5.0f, cudaState, cudaOutput, cudaDecay,
                cudaBeta);
        }
        FastllmCudaSyncCurrentThreadStream();

        ExpectFloatNear(ToFloatVector(cpuOutput), ToFloatVector(cudaOutput),
                        1.0f / 128.0f, 0.0f,
                        "CUDA Kimi-K3 recurrent KDA output");
        ExpectFloatNear(ToFloatVector(cpuDecay), ToFloatVector(cudaDecay),
                        2e-6f, 2e-6f,
                        "CUDA Kimi-K3 recurrent KDA decay");
        ExpectFloatNear(ToFloatVector(cpuBeta), ToFloatVector(cudaBeta),
                        2e-6f, 2e-6f,
                        "CUDA Kimi-K3 recurrent KDA beta");
        ExpectFloatNear(ToFloatVector(cpuState), ToFloatVector(cudaState),
                        2e-5f, 2e-5f,
                        "CUDA Kimi-K3 recurrent KDA state");

        constexpr int replayTokens = 5;
        const int replayVectorItems = replayTokens * heads * dimension;
        const int replayBetaItems = replayTokens * heads;
        const std::vector<int> replayVectorDims = {
            batch, replayTokens, heads, dimension};
        const std::vector<int> replayBetaDims = {
            batch, replayTokens, heads};
        fastllm::Data replayQ(
            fastllm::DataType::BFLOAT16, replayVectorDims,
            std::vector<float>(qValues.begin(),
                               qValues.begin() + replayVectorItems));
        fastllm::Data replayK(
            fastllm::DataType::BFLOAT16, replayVectorDims,
            std::vector<float>(kValues.begin(),
                               kValues.begin() + replayVectorItems));
        fastllm::Data replayV(
            fastllm::DataType::BFLOAT16, replayVectorDims,
            std::vector<float>(vValues.begin(),
                               vValues.begin() + replayVectorItems));
        fastllm::Data replayGate(
            fastllm::DataType::BFLOAT16, replayVectorDims,
            std::vector<float>(gateValues.begin(),
                               gateValues.begin() + replayVectorItems));
        fastllm::Data replayRawBeta(
            fastllm::DataType::FLOAT32, replayBetaDims,
            std::vector<float>(rawBetaValues.begin(),
                               rawBetaValues.begin() + replayBetaItems));
        fastllm::Data referenceReplayState(
            fastllm::DataType::FLOAT32, stateDims, initialStateValues);
        fastllm::Data replayOutput, replayDecay, replayBeta;
        {
            ScopedFirstDevice device("cpu");
            fastllm::KimiK3RecurrentKDA(
                replayQ, replayK, replayV, replayGate, replayRawBeta,
                cpuALog, cpuDtBias, -5.0f, referenceReplayState,
                replayOutput, replayDecay, replayBeta);
        }

        fastllm::Data cpuReplayState(
            fastllm::DataType::FLOAT32, stateDims, initialStateValues);
        {
            ScopedFirstDevice device("cpu");
            fastllm::KimiK3RecurrentKDAUpdateState(
                cpuK, cpuV, cpuGate, cpuRawBeta,
                cpuALog, cpuDtBias, -5.0f, replayTokens,
                cpuReplayState);
        }
        ExpectFloatNear(ToFloatVector(referenceReplayState),
                        ToFloatVector(cpuReplayState), 0.0f, 0.0f,
                        "CPU Kimi-K3 state-only KDA replay");

        fastllm::Data cudaReplayState = MakeCudaTensor(
            fastllm::DataType::FLOAT32, stateDims, initialStateValues);
        {
            ScopedFirstDevice device("cuda:0");
            fastllm::KimiK3RecurrentKDAUpdateState(
                cudaK, cudaV, cudaGate, cudaRawBeta,
                cudaALog, cudaDtBias, -5.0f, replayTokens,
                cudaReplayState);
        }
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(ToFloatVector(referenceReplayState),
                        ToFloatVector(cudaReplayState), 2e-5f, 2e-5f,
                        "CUDA Kimi-K3 state-only KDA replay");

        std::cout << "CUDA Kimi-K3 recurrent KDA regression: PASS\n";
    }

    void RunCudaTritonChunkGdnPrefillRegression() {
        bool tritonEnabled = fastllm::GetFastllmEnv().cudaTriton &&
            RegressionEnvFlagDefaultEnabled(
                "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL", true);
        FastllmCudaSetDevice(0);
        int arch = FastllmCudaRuntimeArch();
        if (!tritonEnabled || arch < 80) {
            std::cout << "Triton chunk GDN prefill regression: SKIP\n";
            return;
        }

        constexpr int batch = 8;
        constexpr int heads = 1;
        constexpr int chunks = 2;
        constexpr int chunkSize = 64;
        constexpr int kDim = 128;
        constexpr int vDim = 128;
        const std::vector<int> qDims = {batch, heads, chunks, chunkSize, kDim};
        const std::vector<int> vDims = {batch, heads, chunks, chunkSize, vDim};
        const std::vector<int> gDims = {batch, heads, chunks, chunkSize};
        const std::vector<int> attnDims = {
            batch, heads, chunks, chunkSize, chunkSize};
        const std::vector<int> stateDims = {batch, heads, kDim, vDim};

        fastllm::Data q = MakeCudaTensor(
            fastllm::DataType::FLOAT16, qDims,
            MakeRegressionValues(
                batch * heads * chunks * chunkSize * kDim, 0.11f, 0.008f));
        fastllm::Data k = MakeCudaTensor(
            fastllm::DataType::FLOAT16, qDims,
            MakeRegressionValues(
                batch * heads * chunks * chunkSize * kDim, 0.37f, 0.007f));
        fastllm::Data v = MakeCudaTensor(
            fastllm::DataType::FLOAT16, vDims,
            MakeRegressionValues(
                batch * heads * chunks * chunkSize * vDim, 0.59f, 0.012f));
        fastllm::Data g = MakeCudaTensor(
            fastllm::DataType::FLOAT16, gDims,
            MakeRegressionValues(
                batch * heads * chunks * chunkSize, 0.83f, 0.003f));
        fastllm::Data attn = MakeCudaTensor(
            fastllm::DataType::FLOAT16, attnDims,
            MakeRegressionValues(
                batch * heads * chunks * chunkSize * chunkSize,
                1.07f, 0.002f));
        fastllm::Data kCumdecay = MakeCudaTensor(
            fastllm::DataType::FLOAT16, qDims,
            MakeRegressionValues(
                batch * heads * chunks * chunkSize * kDim, 1.31f, 0.006f));
        const std::vector<float> initialState = MakeRegressionValues(
            batch * heads * kDim * vDim, 1.73f, 0.015f);
        fastllm::Data referenceState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, initialState);
        fastllm::Data tritonState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, initialState);

        fastllm::Data referenceOutput;
        FastllmChunkGatedDeltaRulePrefill(
            q, k, v, g, attn, kCumdecay, referenceState, referenceOutput);
        FastllmCudaSyncCurrentThreadStream();
        const std::vector<float> referenceStateValues = ToFloatVector(referenceState);
        const std::vector<float> referenceOutputValues = ToFloatVector(referenceOutput);

        fastllm::Data tritonOutput;
        {
            ScopedFirstDevice device("cuda:0");
            fastllm::ChunkGatedDeltaRulePrefill(
                q, k, v, g, attn, kCumdecay, tritonState, tritonOutput);
        }
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(referenceStateValues, ToFloatVector(tritonState),
                        1e-3f, 1e-3f,
                        "Triton chunk GDN prefill recurrent state");
        ExpectFloatNear(referenceOutputValues, ToFloatVector(tritonOutput),
                        1e-3f, 1e-3f,
                        "Triton chunk GDN prefill output");

        auto envIntRange = [](const char *name, int fallback,
                              int minValue, int maxValue) {
            const char *text = std::getenv(name);
            if (text == nullptr || text[0] == '\0') {
                return fallback;
            }
            char *end = nullptr;
            long value = std::strtol(text, &end, 10);
            return end == text || value < minValue || value > maxValue
                ? fallback : (int)value;
        };
        int blockV = envIntRange(
            "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_BLOCK_V", 64, 32, 64);
        if (blockV != 32 && blockV != 64) {
            blockV = 64;
        }
        int numWarps = envIntRange(
            "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_NUM_WARPS", 4, 1, 32);
        int numStages = envIntRange(
            "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_NUM_STAGES", 2, 1, 8);

        std::string cacheDir;
        const char *cacheEnv = std::getenv("FASTLLM_CUDA_TRITON_CACHE_DIR");
        if (cacheEnv != nullptr && cacheEnv[0] != '\0') {
            cacheDir = cacheEnv;
        } else {
            const char *xdg = std::getenv("XDG_CACHE_HOME");
            const char *home = std::getenv("HOME");
            if (xdg != nullptr && xdg[0] != '\0') {
                cacheDir = std::string(xdg) + "/fastllm/triton";
            } else if (home != nullptr && home[0] != '\0') {
                cacheDir = std::string(home) + "/.cache/fastllm/triton";
            } else {
                cacheDir = "/tmp/fastllm-triton";
            }
        }
        if (cacheDir.size() > 1 && cacheDir[0] == '~' && cacheDir[1] == '/') {
            const char *home = std::getenv("HOME");
            if (home != nullptr && home[0] != '\0') {
                cacheDir = std::string(home) + cacheDir.substr(1);
            }
        }
        std::string separator = cacheDir.empty() || cacheDir.back() == '/'
            ? "" : "/";
        std::string base = cacheDir + separator +
            "chunk_gdn_prefill_v2_fp16_sm" + std::to_string(arch) +
            "_c2_t64_k128_v128_bv" + std::to_string(blockV) +
            "_nw" + std::to_string(numWarps) +
            "_ns" + std::to_string(numStages);
        std::ifstream metaFile(base + ".json", std::ios::binary);
        Expect(metaFile.good(),
               "Triton chunk GDN prefill metadata was not generated");
        std::string metaText(
            (std::istreambuf_iterator<char>(metaFile)),
            std::istreambuf_iterator<char>());
        std::string jsonError;
        json11::Json meta = json11::Json::parse(metaText, jsonError);
        Expect(jsonError.empty() && meta["ok"].bool_value(),
               "Triton chunk GDN prefill metadata is invalid");
        json11::Json hMeta = meta["kernels"]["h"];
        json11::Json oMeta = meta["kernels"]["o"];
        std::string hCubin = hMeta["cubin"].string_value();
        std::string hKernel = hMeta["kernel"].string_value();
        int hNumWarps = hMeta["num_warps"].int_value();
        int hShared = hMeta["shared"].int_value();
        std::string oCubin = oMeta["cubin"].string_value();
        std::string oKernel = oMeta["kernel"].string_value();
        int oNumWarps = oMeta["num_warps"].int_value();
        int oShared = oMeta["shared"].int_value();
        Expect(!hCubin.empty() && !hKernel.empty() && hNumWarps > 0 &&
                   !oCubin.empty() && !oKernel.empty() && oNumWarps > 0,
               "Triton chunk GDN prefill kernel metadata is incomplete");

        fastllm::Data directState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, initialState);
        fastllm::Data directOutput;
        Expect(FastllmCudaTritonChunkGatedDeltaRulePrefill(
                   hCubin.c_str(), hKernel.c_str(), hNumWarps, hShared,
                   oCubin.c_str(), oKernel.c_str(), oNumWarps, oShared,
                   chunks, chunkSize, kDim, vDim, blockV,
                   q, k, v, g, attn, kCumdecay, directState, directOutput),
               "direct Triton chunk GDN prefill launch failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(referenceStateValues, ToFloatVector(directState),
                        1e-3f, 1e-3f,
                        "direct Triton chunk GDN recurrent state");
        ExpectFloatNear(referenceOutputValues, ToFloatVector(directOutput),
                        1e-3f, 1e-3f,
                        "direct Triton chunk GDN output");

        fastllm::Data failedState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, initialState);
        const std::vector<float> initialFp16 = ToFloatVector(failedState);
        fastllm::Data failedOutput;
        bool accepted = FastllmCudaTritonChunkGatedDeltaRulePrefill(
            hCubin.c_str(), hKernel.c_str(), hNumWarps, hShared,
            oCubin.c_str(), oKernel.c_str(), 1024, oShared,
            chunks, chunkSize, kDim, vDim, blockV,
            q, k, v, g, attn, kCumdecay, failedState, failedOutput);
        Expect(!accepted,
               "fault-injected Triton chunk GDN O launch unexpectedly succeeded");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(initialFp16, ToFloatVector(failedState), 0.0f, 0.0f,
                        "failed Triton chunk GDN launch state transaction");

        FastllmChunkGatedDeltaRulePrefill(
            q, k, v, g, attn, kCumdecay, failedState, failedOutput);
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(referenceStateValues, ToFloatVector(failedState),
                        0.0f, 0.0f,
                        "Triton chunk GDN native fallback recurrent state");
        ExpectFloatNear(referenceOutputValues, ToFloatVector(failedOutput),
                        0.0f, 0.0f,
                        "Triton chunk GDN native fallback output");
        std::cout << "Triton chunk GDN prefill transaction regression: PASS\n";
    }

    void RunCudaDeepSeekV4TritonWoARegression() {
        bool tritonEnabled = fastllm::GetFastllmEnv().cudaTriton &&
            RegressionEnvFlagDefaultEnabled(
                "FASTLLM_CUDA_TRITON_DEEPSEEK_V4_FP8_WOA", true);
        FastllmCudaSetDevice(0);
        constexpr int groups = 2;
        constexpr int heads = 4;
        constexpr int headDim = 64;
        constexpr int hidden = (heads / groups) * headDim;
        constexpr int outRank = 128;

        std::vector<float> inputValues(heads * headDim, 1.0f);
        fastllm::Data input = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, heads, headDim}, inputValues);

        fastllm::Data weight;
        weight.dataType = fastllm::DataType::FP8_E4M3;
        weight.UpdateUnitSize();
        weight.Resize({groups * outRank, hidden});
        weight.weightType = fastllm::WeightType::LINEAR;
        weight.blockK = 128;
        weight.blockM = 128;
        weight.Allocate(false);
        uint8_t *weightBytes = reinterpret_cast<uint8_t*>(weight.cpuData);
        for (uint64_t i = 0; i < weight.GetBytes(); i++) {
            weightBytes[i] = static_cast<uint8_t>(0x20 + (i & 0x1f));
        }
        weight.scales = {1.0f / 64.0f, 1.0f / 32.0f};
        weight.ToDevice(fastllm::DataDevice::CUDA);

        fastllm::Data reference;
        Expect(FastllmCudaDeepSeekV4WoA(
                   input, weight, groups, outRank, reference, false),
               "built-in DeepSeek-V4 WoA reference rejected its test input");

        fastllm::Data actual = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, groups * outRank},
            std::vector<float>(groups * outRank, 0.0f));
        bool usedTriton = fastllm::FastllmCudaTryTritonDeepSeekV4WoA(
            input, weight, groups, outRank, actual);
        if (tritonEnabled) {
            Expect(usedTriton,
                   "Triton DeepSeek-V4 WoA rejected its supported test input");
        } else {
            Expect(!usedTriton,
                   "DeepSeek-V4 WoA ignored its disabled Triton gate");
            Expect(FastllmCudaDeepSeekV4WoA(
                       input, weight, groups, outRank, actual),
                   "built-in DeepSeek-V4 WoA fallback rejected its test input");
        }
        ExpectFloatNear(ToFloatVector(reference), ToFloatVector(actual),
                        2e-2f, 2e-3f, "DeepSeek-V4 Triton WoA output");
        std::cout << "DeepSeek-V4 Triton WoA regression: PASS ("
                  << (tritonEnabled ? "Triton" : "disabled-gate fallback")
                  << ")\n";
    }

    void RunCudaDeepSeekV4TritonSparseDecodeRegression() {
        bool tritonEnabled = fastllm::GetFastllmEnv().cudaTriton &&
            RegressionEnvFlagDefaultEnabled(
                "FASTLLM_CUDA_TRITON_DEEPSEEK_V4_SPARSE_DECODE", true);
        FastllmCudaSetDevice(0);
        constexpr int heads = 4;
        constexpr int headDim = 64;
        constexpr int windowSize = 8;
        constexpr int compressedCapacity = 4;
        constexpr int startPos = 13;
        constexpr int ropeDim = 16;

        fastllm::Data q = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, heads, headDim},
            MakeRegressionValues(heads * headDim, 0.31f, 0.16f));
        fastllm::Data windowKV = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, windowSize, headDim},
            MakeRegressionValues(windowSize * headDim, 0.73f, 0.12f));
        fastllm::Data compressedKV = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, compressedCapacity, headDim},
            MakeRegressionValues(compressedCapacity * headDim, 1.07f, 0.10f));
        fastllm::Data sink = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {heads},
            MakeRegressionValues(heads, 1.41f, 0.08f));
        fastllm::Data decodeMeta = MakeIntTensor({2}, {startPos, 123});
        decodeMeta.ToDevice(fastllm::DataDevice::CUDA);
        const int32_t *decodeMetaPtr =
            reinterpret_cast<const int32_t*>(decodeMeta.cudaData);
        float softmaxScale = 1.0f / std::sqrt((float)headDim);

        const int compressRatios[] = {0, 4, 128};
        for (int compressRatio : compressRatios) {
            fastllm::Data reference;
            Expect(FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraph(
                       q, windowKV, compressedKV, sink, windowSize, compressRatio,
                       decodeMetaPtr, ropeDim, 10000.0f, 4096, 8.0f, 32, 1,
                       softmaxScale, reference, false),
                   "built-in DeepSeek-V4 sparse decode rejected ratio=" +
                       std::to_string(compressRatio));

            fastllm::Data directOutput = MakeCudaTensor(
                fastllm::DataType::FLOAT32, {1, 1, heads, headDim},
                std::vector<float>(heads * headDim, 0.0f));
            bool usedTriton =
                fastllm::FastllmCudaTryTritonDeepSeekV4SparseAttentionDecodeGraph(
                    q, windowKV, compressedKV, sink, windowSize, compressRatio,
                    decodeMetaPtr, softmaxScale,
                    reinterpret_cast<float*>(directOutput.cudaData));
            fastllm::Data actual;
            if (tritonEnabled) {
                Expect(usedTriton,
                       "Triton DeepSeek-V4 sparse decode rejected ratio=" +
                           std::to_string(compressRatio));
            } else {
                Expect(!usedTriton,
                       "DeepSeek-V4 sparse decode ignored its disabled Triton gate");
            }
            // The trial entry writes the pre-RoPE FLOAT32 attention result.
            // Compare final outputs through the production wrapper, which
            // applies the rotary cast after either Triton or native fallback.
            Expect(FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraph(
                       q, windowKV, compressedKV, sink, windowSize, compressRatio,
                       decodeMetaPtr, ropeDim, 10000.0f, 4096, 8.0f, 32, 1,
                       softmaxScale, actual),
                   "DeepSeek-V4 sparse decode wrapper rejected ratio=" +
                       std::to_string(compressRatio));
            ExpectFloatNear(ToFloatVector(reference), ToFloatVector(actual),
                            3e-2f, 3e-3f,
                            "DeepSeek-V4 Triton sparse decode output ratio=" +
                                std::to_string(compressRatio));
        }
        std::cout << "DeepSeek-V4 Triton sparse decode regression: PASS ("
                  << (tritonEnabled ? "Triton" : "disabled-gate fallback")
                  << ")\n";
    }

    void RunCudaDeepSeekV4HashRouteCacheRegression() {
        FastllmCudaSetDevice(0);
        constexpr int topk = 2;
        fastllm::Data logits = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, 4},
            {0.3f, -0.1f, 0.8f, 0.2f});
        fastllm::Data routeTable = MakeIntTensor(
            {2, topk}, {0, 1, 2, 3});
        int inputId = 0;
        fastllm::Data expertIndex, expertScore;
        Expect(FastllmCudaDeepSeekV4HashRouteScore(
                   logits, routeTable, &inputId, 1, topk, 0, 1.0f,
                   expertIndex, expertScore),
               "DeepSeek-V4 eager hash route rejected its cache test input");
        ExpectIntEqual({0, 1}, ToIntVector(expertIndex),
                       "DeepSeek-V4 eager hash route initial table");

        // Mutable non-model tensors must refresh even when the CPU address,
        // shape and element count stay unchanged.
        int32_t *routeValues = reinterpret_cast<int32_t*>(routeTable.cpuData);
        routeValues[0] = 3;
        routeValues[1] = 2;
        fastllm::Data decodeMeta = MakeIntTensor({2}, {0, inputId});
        decodeMeta.ToDevice(fastllm::DataDevice::CUDA);
        Expect(FastllmCudaDeepSeekV4HashRouteScoreGraph(
                   logits, routeTable,
                   reinterpret_cast<const int32_t*>(decodeMeta.cudaData),
                   1, topk, 0, 1.0f, expertIndex, expertScore),
               "DeepSeek-V4 graph hash route rejected its refreshed table");
        ExpectIntEqual({3, 2}, ToIntVector(expertIndex),
                       "DeepSeek-V4 graph hash route refreshed table");

        // Model weights skip per-token hashing because they are immutable.
        // Explicit retirement must nevertheless make a reloaded table upload
        // fresh contents, even if the same Data object is reused by a test.
        routeTable.isModelWeight = true;
        FastllmCudaReleaseDeepSeekV4RouteTableCache(&routeTable);
        routeValues[0] = 1;
        routeValues[1] = 3;
        Expect(FastllmCudaDeepSeekV4HashRouteScore(
                   logits, routeTable, &inputId, 1, topk, 0, 1.0f,
                   expertIndex, expertScore),
               "DeepSeek-V4 eager hash route rejected its retired cache");
        ExpectIntEqual({1, 3}, ToIntVector(expertIndex),
                       "DeepSeek-V4 eager hash route after cache retirement");

        std::cout << "DeepSeek-V4 hash route cache regression: PASS\n";
    }

    void RunCudaDeepSeekV4FusedHcPreNormRegression() {
        FastllmCudaSetDevice(0);
        constexpr int hcMult = 4;
        constexpr int hidden = 4096;
        constexpr int mixHc = (2 + hcMult) * hcMult;
        constexpr int flat = hcMult * hidden;
        constexpr int sinkhornIters = 20;
        constexpr float eps = 1e-6f;

        fastllm::Data input = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, hcMult, hidden},
            MakeRegressionValues(flat, 0.23f, 0.06f));
        fastllm::Data hcFn = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {mixHc, flat},
            MakeRegressionValues(mixHc * flat, 0.67f, 0.008f));
        fastllm::Data hcScale = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {3}, {0.71f, 0.83f, 0.57f});
        fastllm::Data hcBase = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {mixHc},
            MakeRegressionValues(mixHc, 1.11f, 0.12f));
        std::vector<float> normValues(hidden);
        for (int i = 0; i < hidden; i++) {
            normValues[i] = 1.0f + 0.08f * std::sin((i + 1) * 0.013f);
        }
        fastllm::Data normWeight = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {hidden}, normValues);

        fastllm::Data preOutput, referencePost, referenceComb;
        Expect(FastllmCudaDeepSeekV4HcPre(
                   input, hcFn, hcScale, hcBase, hcMult, sinkhornIters,
                   eps, eps, preOutput, referencePost, referenceComb),
               "built-in DeepSeek-V4 HcPre rejected fused-norm regression input");
        fastllm::Data referenceNorm = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, hidden},
            std::vector<float>(hidden, 0.0f));
        Expect(FastllmCudaRMSNorm(preOutput, normWeight, referenceNorm, eps),
               "built-in RMSNorm rejected fused HcPre regression input");

        fastllm::Data actualNorm, actualPost, actualComb;
        Expect(FastllmCudaDeepSeekV4HcPreNorm(
                   input, hcFn, hcScale, hcBase, normWeight, hcMult,
                   sinkhornIters, eps, eps, actualNorm, actualPost, actualComb),
               "fused DeepSeek-V4 HcPreNorm rejected its decode shape");
        ExpectFloatNear(ToFloatVector(referenceNorm), ToFloatVector(actualNorm),
                        2e-2f, 2e-3f, "DeepSeek-V4 fused HcPreNorm output");
        ExpectFloatNear(ToFloatVector(referencePost), ToFloatVector(actualPost),
                        2e-5f, 2e-5f, "DeepSeek-V4 fused HcPreNorm post mix");
        ExpectFloatNear(ToFloatVector(referenceComb), ToFloatVector(actualComb),
                        2e-5f, 2e-5f, "DeepSeek-V4 fused HcPreNorm comb mix");

        fastllm::Data layerOutput = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, hidden},
            MakeRegressionValues(hidden, 0.91f, 0.09f));
        fastllm::Data previousPost = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, 1, hcMult},
            {0.62f, 0.48f, 0.71f, 0.55f});
        fastllm::Data previousComb = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, 1, hcMult, hcMult},
            {0.70f, 0.12f, 0.09f, 0.09f,
             0.10f, 0.72f, 0.08f, 0.10f,
             0.08f, 0.11f, 0.73f, 0.08f,
             0.12f, 0.07f, 0.10f, 0.71f});
        fastllm::Data referenceResidual;
        Expect(FastllmCudaDeepSeekV4HcPostCudaMix(
                   layerOutput, input, previousPost, previousComb,
                   1, 1, hcMult, hidden, referenceResidual),
               "built-in DeepSeek-V4 HcPost rejected transition regression input");
        fastllm::Data transitionReferenceNorm;
        fastllm::Data transitionReferencePost;
        fastllm::Data transitionReferenceComb;
        Expect(FastllmCudaDeepSeekV4HcPreNorm(
                   referenceResidual, hcFn, hcScale, hcBase, normWeight,
                   hcMult, sinkhornIters, eps, eps, transitionReferenceNorm,
                   transitionReferencePost, transitionReferenceComb),
               "built-in DeepSeek-V4 HcPreNorm rejected transition reference");

        fastllm::Data transitionResidual;
        fastllm::Data transitionNorm;
        fastllm::Data transitionPost;
        fastllm::Data transitionComb;
        Expect(FastllmCudaDeepSeekV4HcPostPreNorm(
                   layerOutput, input, previousPost, previousComb, hcFn,
                   hcScale, hcBase, normWeight, hcMult, sinkhornIters,
                   eps, eps, transitionResidual, transitionNorm,
                   transitionPost, transitionComb),
               "fused DeepSeek-V4 HcPostPreNorm rejected its decode shape");
        ExpectFloatNear(ToFloatVector(referenceResidual),
                        ToFloatVector(transitionResidual), 4e-3f, 2e-3f,
                        "DeepSeek-V4 fused HcPostPreNorm residual");
        ExpectFloatNear(ToFloatVector(transitionReferenceNorm),
                        ToFloatVector(transitionNorm), 3e-2f, 4e-3f,
                        "DeepSeek-V4 fused HcPostPreNorm norm output");
        ExpectFloatNear(ToFloatVector(transitionReferencePost),
                        ToFloatVector(transitionPost), 2e-3f, 2e-3f,
                        "DeepSeek-V4 fused HcPostPreNorm post mix");
        ExpectFloatNear(ToFloatVector(transitionReferenceComb),
                        ToFloatVector(transitionComb), 2e-3f, 2e-3f,
                        "DeepSeek-V4 fused HcPostPreNorm comb mix");

        std::cout << "DeepSeek-V4 fused HcPreNorm regression: PASS\n";
    }

    void RunCudaDeepSeekV4FusedQKVRopeCacheRegression() {
        FastllmCudaSetDevice(0);
        constexpr int heads = 64;
        constexpr int headDim = 512;
        constexpr int ropeDim = 64;
        constexpr int windowSize = 128;
        constexpr int position = 173;
        constexpr float eps = 1e-6f;

        std::vector<float> qValues =
            MakeRegressionValues(heads * headDim, 0.47f, 0.65f);
        std::vector<float> kvValues =
            MakeRegressionValues(headDim, 1.31f, 0.72f);
        std::vector<float> weightValues(headDim);
        for (int i = 0; i < headDim; i++) {
            weightValues[i] = 1.0f + 0.09f * std::sin((i + 1) * 0.019f);
        }
        std::vector<float> cacheValues(windowSize * headDim);
        for (int i = 0; i < (int)cacheValues.size(); i++) {
            cacheValues[i] = -0.4f + 0.03f * std::sin((i + 1) * 0.007f);
        }

        fastllm::Data decodeMeta = MakeIntTensor({2}, {position, 123});
        decodeMeta.ToDevice(fastllm::DataDevice::CUDA);
        const int32_t *decodeMetaPtr =
            reinterpret_cast<const int32_t*>(decodeMeta.cudaData);
        fastllm::Data kvNormWeight = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {headDim}, weightValues);

        fastllm::Data referenceQ = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, heads, headDim}, qValues);
        fastllm::Data referenceKVInput = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, 1, headDim}, kvValues);
        fastllm::Data referenceKV = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, 1, headDim},
            std::vector<float>(headDim, 0.0f));
        Expect(FastllmCudaRMSNorm(referenceKVInput, kvNormWeight, referenceKV, eps),
               "built-in RMSNorm rejected fused QKV reference input");
        Expect(FastllmCudaDeepSeekV4ScaleQRotaryGraph(
                   referenceQ, ropeDim, 160000.0f, decodeMetaPtr,
                   4096, 4.0f, 32, 1, eps),
               "built-in Q rotary rejected fused QKV reference input");
        Expect(FastllmCudaDeepSeekV4RotaryQuantGraph(
                   referenceKV, ropeDim, 160000.0f, decodeMetaPtr,
                   4096, 4.0f, 32, 1, headDim - ropeDim, 64, 1),
               "built-in KV rotary rejected fused QKV reference input");
        referenceKV.Reshape({1, 1, headDim});
        fastllm::Data referenceCache = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, windowSize, headDim}, cacheValues);
        Expect(FastllmCudaDeepSeekV4UpdateWindowKVCacheGraph(
                   referenceKV, decodeMetaPtr, windowSize, referenceCache),
               "built-in cache update rejected fused QKV reference input");

        fastllm::Data actualQ = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, heads, headDim}, qValues);
        fastllm::Data actualKV = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, 1, headDim}, kvValues);
        fastllm::Data actualCache = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, windowSize, headDim}, cacheValues);
        Expect(FastllmCudaDeepSeekV4FusedQKVRopeCacheGraph(
                   actualQ, actualKV, kvNormWeight, decodeMetaPtr,
                   ropeDim, 160000.0f, 4096, 4.0f, 32, 1, eps,
                   headDim - ropeDim, 64, windowSize, actualCache),
               "fused DeepSeek-V4 QKV rope/cache rejected its decode shape");

        ExpectFloatNear(ToFloatVector(referenceQ), ToFloatVector(actualQ),
                        3e-2f, 3e-3f, "DeepSeek-V4 fused Q output");
        ExpectFloatNear(ToFloatVector(referenceKV), ToFloatVector(actualKV),
                        3e-2f, 3e-3f, "DeepSeek-V4 fused KV output");
        ExpectFloatNear(ToFloatVector(referenceCache), ToFloatVector(actualCache),
                        3e-2f, 3e-3f, "DeepSeek-V4 fused window cache");

        // TP8 shards the model's 64 global query heads into eight local heads.
        // Keep this shape covered so the fused path cannot silently fall back to
        // the three-kernel reference sequence in tensor-parallel decode.
        {
            constexpr int localHeads = 8;
            std::vector<float> localQValues =
                MakeRegressionValues(localHeads * headDim, 0.91f, 0.58f);
            fastllm::Data localReferenceQ = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, 1, localHeads, headDim}, localQValues);
            fastllm::Data localReferenceKVInput = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, {1, 1, 1, headDim}, kvValues);
            fastllm::Data localReferenceKV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, {1, 1, 1, headDim},
                std::vector<float>(headDim, 0.0f));
            Expect(FastllmCudaRMSNorm(
                       localReferenceKVInput, kvNormWeight,
                       localReferenceKV, eps),
                   "built-in RMSNorm rejected TP8 fused QKV reference input");
            Expect(FastllmCudaDeepSeekV4ScaleQRotaryGraph(
                       localReferenceQ, ropeDim, 160000.0f, decodeMetaPtr,
                       4096, 4.0f, 32, 1, eps),
                   "built-in Q rotary rejected TP8 fused QKV reference input");
            Expect(FastllmCudaDeepSeekV4RotaryQuantGraph(
                       localReferenceKV, ropeDim, 160000.0f, decodeMetaPtr,
                       4096, 4.0f, 32, 1, headDim - ropeDim, 64, 1),
                   "built-in KV rotary rejected TP8 fused QKV reference input");
            localReferenceKV.Reshape({1, 1, headDim});
            fastllm::Data localReferenceCache = MakeCudaTensor(
                fastllm::DataType::FLOAT32,
                {1, windowSize, headDim}, cacheValues);
            Expect(FastllmCudaDeepSeekV4UpdateWindowKVCacheGraph(
                       localReferenceKV, decodeMetaPtr,
                       windowSize, localReferenceCache),
                   "built-in cache update rejected TP8 fused QKV reference input");

            fastllm::Data localActualQ = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, 1, localHeads, headDim}, localQValues);
            fastllm::Data localActualKV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, {1, 1, 1, headDim}, kvValues);
            fastllm::Data localActualCache = MakeCudaTensor(
                fastllm::DataType::FLOAT32,
                {1, windowSize, headDim}, cacheValues);
            Expect(FastllmCudaDeepSeekV4FusedQKVRopeCacheGraph(
                       localActualQ, localActualKV, kvNormWeight, decodeMetaPtr,
                       ropeDim, 160000.0f, 4096, 4.0f, 32, 1, eps,
                       headDim - ropeDim, 64, windowSize, localActualCache),
                   "fused DeepSeek-V4 QKV rope/cache rejected TP8 local heads");
            ExpectFloatNear(ToFloatVector(localReferenceQ),
                            ToFloatVector(localActualQ),
                            3e-2f, 3e-3f,
                            "DeepSeek-V4 TP8 fused Q output");
            ExpectFloatNear(ToFloatVector(localReferenceKV),
                            ToFloatVector(localActualKV),
                            3e-2f, 3e-3f,
                            "DeepSeek-V4 TP8 fused KV output");
            ExpectFloatNear(ToFloatVector(localReferenceCache),
                            ToFloatVector(localActualCache),
                            3e-2f, 3e-3f,
                            "DeepSeek-V4 TP8 fused window cache");
        }

        std::cout << "DeepSeek-V4 fused QKV rope/cache regression: PASS\n";
    }

    void RunCudaGraphMemoryPoolOwnershipRegression() {
        FastllmCudaSetDevice(0);

        // A pointer released by a participating capture stream must not be
        // handed to an unrelated allocator thread before finalization pins it.
        constexpr size_t concurrentBytes = 19 * 1024 * 1024 + 4096;
        void *warm = FastllmCudaMalloc(concurrentBytes);
        Expect(warm != nullptr, "graph pool concurrency warmup allocation failed");
        FastllmCudaFree(warm);

        Expect(FastllmCudaGraphMemoryPoolBegin(),
               "graph pool concurrency capture begin failed");
        Expect(FastllmCudaGraphBeginCapture(),
               "graph pool concurrency stream capture begin failed");
        void *captured = FastllmCudaMalloc(concurrentBytes);
        Expect(captured != nullptr,
               "graph pool concurrency capture allocation failed");
        Expect(cudaMemsetAsync(captured, 0, concurrentBytes,
                               cudaStreamPerThread) == cudaSuccess,
               "graph pool concurrency captured memset failed");
        FastllmCudaFree(captured);

        void *external = nullptr;
        std::thread allocator([&]() {
            FastllmCudaSetDevice(0);
            external = FastllmCudaMalloc(concurrentBytes);
        });
        allocator.join();
        Expect(external != nullptr,
               "graph pool external allocation failed");
        Expect(external != captured,
               "external allocator reused a pointer owned by active capture");

        void *graph = nullptr;
        Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
               "graph pool concurrency stream capture end failed");
        std::vector<void*> reserved;
        Expect(FastllmCudaGraphMemoryPoolEnd(reserved),
               "graph pool concurrency finalization failed");
        Expect(std::find(reserved.begin(), reserved.end(), captured) !=
                   reserved.end(),
               "idle captured pointer was not pinned");
        FastllmCudaGraphDestroy(graph);
        FastllmCudaGraphMemoryPoolRelease(reserved);
        FastllmCudaFree(external);

        // If a released temporary is immediately reused by a persistent owner
        // on the capture stream, the graph and Data must hold independent
        // references so releasing the graph first cannot expose the Data buffer.
        constexpr size_t persistentBytes = 23 * 1024 * 1024 + 8192;
        warm = FastllmCudaMalloc(persistentBytes);
        Expect(warm != nullptr, "graph pool persistent warmup allocation failed");
        FastllmCudaFree(warm);

        Expect(FastllmCudaGraphMemoryPoolBegin(),
               "graph pool persistent capture begin failed");
        Expect(FastllmCudaGraphBeginCapture(),
               "graph pool persistent stream capture begin failed");
        void *temporary = FastllmCudaMalloc(persistentBytes);
        Expect(temporary != nullptr,
               "graph pool persistent temporary allocation failed");
        Expect(cudaMemsetAsync(temporary, 0, persistentBytes,
                               cudaStreamPerThread) == cudaSuccess,
               "graph pool persistent captured memset failed");
        FastllmCudaFree(temporary);
        void *persistentOwner = FastllmCudaMalloc(persistentBytes);
        Expect(persistentOwner == temporary,
               "capture stream did not reuse its released temporary");

        graph = nullptr;
        Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
               "graph pool persistent stream capture end failed");
        reserved.clear();
        Expect(FastllmCudaGraphMemoryPoolEnd(reserved),
               "graph pool persistent finalization failed");
        Expect(std::find(reserved.begin(), reserved.end(), persistentOwner) !=
                   reserved.end(),
               "persistent captured owner did not receive an independent graph pin");
        FastllmCudaGraphDestroy(graph);
        FastllmCudaGraphMemoryPoolRelease(reserved);

        void *probe = FastllmCudaMalloc(persistentBytes);
        Expect(probe != nullptr && probe != persistentOwner,
               "graph release exposed a buffer still owned by persistent Data");
        FastllmCudaFree(probe);
        FastllmCudaFree(persistentOwner);

        // Exercise the opposite release order: a Data owner may disappear while
        // the graph is alive, but its address must remain unavailable until the
        // graph pin is released.
        constexpr size_t ownerFirstBytes = 29 * 1024 * 1024 + 12288;
        warm = FastllmCudaMalloc(ownerFirstBytes);
        Expect(warm != nullptr, "graph pool owner-first warmup allocation failed");
        FastllmCudaFree(warm);
        Expect(FastllmCudaGraphMemoryPoolBegin(),
               "graph pool owner-first capture begin failed");
        Expect(FastllmCudaGraphBeginCapture(),
               "graph pool owner-first stream capture begin failed");
        temporary = FastllmCudaMalloc(ownerFirstBytes);
        Expect(temporary != nullptr,
               "graph pool owner-first temporary allocation failed");
        Expect(cudaMemsetAsync(temporary, 0, ownerFirstBytes,
                               cudaStreamPerThread) == cudaSuccess,
               "graph pool owner-first captured memset failed");
        FastllmCudaFree(temporary);
        persistentOwner = FastllmCudaMalloc(ownerFirstBytes);
        Expect(persistentOwner == temporary,
               "owner-first capture did not reuse its temporary");
        graph = nullptr;
        Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
               "graph pool owner-first stream capture end failed");
        reserved.clear();
        Expect(FastllmCudaGraphMemoryPoolEnd(reserved),
               "graph pool owner-first finalization failed");
        FastllmCudaFree(persistentOwner);
        probe = FastllmCudaMalloc(ownerFirstBytes);
        Expect(probe != nullptr && probe != persistentOwner,
               "Data release exposed an address still pinned by the graph");
        FastllmCudaFree(probe);
        FastllmCudaGraphDestroy(graph);
        FastllmCudaClearThreadError();
        FastllmCudaGraphMemoryPoolRelease(reserved);
        Expect(!FastllmCudaGetThreadError(),
               "graph pin release lost its owner-first pool entry");
        void *afterRelease = FastllmCudaMalloc(ownerFirstBytes);
        Expect(afterRelease != nullptr,
               "allocation failed after releasing the owner-first graph pin");
        FastllmCudaFree(afterRelease);
        std::cout << "CUDA graph memory-pool ownership regression: PASS\n";
    }

    std::vector<float> ExtractLastAxisToken(const std::vector<float> &values,
                                            int rows, int tokens, int token) {
        std::vector<float> result(rows);
        for (int row = 0; row < rows; row++) {
            result[row] = values[(size_t) row * tokens + token];
        }
        return result;
    }

    void RunCudaConvMultiTokenSnapshotsRegression() {
        FastllmCudaSetDevice(0);
        const int batch = 2;
        const int channels = 5;
        const int rows = batch * channels;

        std::vector<float> initialCacheValues = MakeRegressionValues(rows * 4, 0.2f, 0.35f);
        std::vector<float> weightValues = MakeRegressionValues(channels * 4, 0.7f, 0.25f);
        std::vector<float> biasValues = MakeRegressionValues(channels, 1.1f, 0.08f);
        fastllm::Data weight = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                              {channels, 4}, weightValues);
        fastllm::Data bias = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                            {channels}, biasValues);

        for (int tokenCount = 1; tokenCount <= 6; tokenCount++) {
            std::vector<float> tokenValues =
                MakeRegressionValues(rows * tokenCount, 1.7f + tokenCount, 0.4f);
            fastllm::Data allTokens = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                     {batch, channels, tokenCount}, tokenValues);
            fastllm::Data sequentialCache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                           {batch, channels, 4}, initialCacheValues);
            fastllm::Data multiCache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                      {batch, channels, 4}, initialCacheValues);

            std::vector<std::vector<float> > expectedOutputs(tokenCount);
            std::vector<std::vector<float> > expectedCaches(tokenCount);
            for (int token = 0; token < tokenCount; token++) {
                std::vector<float> singleTokenValues(rows);
                for (int row = 0; row < rows; row++) {
                    singleTokenValues[row] = tokenValues[(size_t) row * tokenCount + token];
                }
                fastllm::Data singleToken = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                           {batch, channels, 1}, singleTokenValues);
                fastllm::Data singleOutput;
                Expect(FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                           sequentialCache, singleToken, weight, bias, singleOutput),
                       "single-token conv reference rejected N=" + std::to_string(tokenCount) +
                       ", token=" + std::to_string(token));
                expectedOutputs[token] = ToFloatVector(singleOutput);
                expectedCaches[token] = ToFloatVector(sequentialCache);
            }

            std::vector<fastllm::Data> snapshots(tokenCount);
            std::vector<fastllm::Data*> snapshotPtrs(tokenCount);
            for (int token = 0; token < tokenCount; token++) {
                snapshotPtrs[token] = &snapshots[token];
            }
            fastllm::Data multiOutput;
            Expect(FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       multiCache, allTokens, weight, bias, multiOutput,
                       snapshotPtrs.data(), tokenCount),
                   "multi-token conv rejected N=" + std::to_string(tokenCount));
            ExpectCudaTensorMeta(multiOutput, fastllm::DataType::FLOAT16,
                                 {batch, channels, tokenCount},
                                 "multi-token conv output metadata");

            std::vector<float> actualOutput = ToFloatVector(multiOutput);
            for (int token = 0; token < tokenCount; token++) {
                std::string suffix = " N=" + std::to_string(tokenCount) +
                                     ", token=" + std::to_string(token);
                ExpectFloatNear(expectedOutputs[token],
                                ExtractLastAxisToken(actualOutput, rows, tokenCount, token),
                                1e-3f, 1e-3f, "multi-token conv output" + suffix);
                ExpectFloatNear(expectedCaches[token], ToFloatVector(snapshots[token]),
                                1e-3f, 1e-3f, "multi-token conv snapshot" + suffix);
                ExpectCudaTensorMeta(snapshots[token], fastllm::DataType::FLOAT16,
                                     {batch, channels, 4},
                                     "multi-token conv snapshot metadata" + suffix);
            }
            ExpectFloatNear(ToFloatVector(sequentialCache), ToFloatVector(multiCache),
                            1e-3f, 1e-3f,
                            "multi-token conv final cache N=" + std::to_string(tokenCount));
            if (tokenCount == 6) {
                fastllm::Data partialCache = MakeCudaTensor(
                    fastllm::DataType::FLOAT16, {batch, channels, 4}, initialCacheValues);
                std::vector<fastllm::Data> partialSnapshots(5);
                std::vector<fastllm::Data*> partialSnapshotPtrs(5);
                for (int token = 0; token < 5; token++) {
                    partialSnapshotPtrs[token] = &partialSnapshots[token];
                }
                fastllm::Data partialOutput;
                Expect(FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                           partialCache, allTokens, weight, bias, partialOutput,
                           partialSnapshotPtrs.data(), 5),
                       "multi-token conv rejected N=6 with five prefix snapshots");
                ExpectFloatNear(ToFloatVector(multiOutput), ToFloatVector(partialOutput),
                                1e-3f, 1e-3f,
                                "multi-token conv partial-snapshot output");
                ExpectFloatNear(ToFloatVector(multiCache), ToFloatVector(partialCache),
                                1e-3f, 1e-3f,
                                "multi-token conv partial-snapshot final cache");
                for (int token = 0; token < 5; token++) {
                    ExpectFloatNear(expectedCaches[token], ToFloatVector(partialSnapshots[token]),
                                    1e-3f, 1e-3f,
                                    "multi-token conv partial snapshot token=" +
                                    std::to_string(token));
                }
            }
        }

        const int tokenCount = 2;
        std::vector<float> tokenValues = MakeRegressionValues(rows * tokenCount, 3.4f, 0.4f);
        fastllm::Data allTokens = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, tokenCount}, tokenValues);
        {
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data output;
            Expect(FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, allTokens, weight, bias, output, nullptr, 0),
                   "multi-token conv should accept nullptr snapshots when count is zero");
        }
        {
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, allTokens, weight, bias, output, nullptr, 1),
                   "multi-token conv accepted nullptr snapshots with a positive count");
        }
        {
            fastllm::Data badCache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                    {batch, channels * 4}, initialCacheValues);
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       badCache, allTokens, weight, bias, output, nullptr, 0),
                   "multi-token conv accepted a rank-2 cache");
        }
        {
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data badTokens = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                     {batch, channels * tokenCount}, tokenValues);
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, badTokens, weight, bias, output, nullptr, 0),
                   "multi-token conv accepted rank-2 new tokens");
        }
        {
            // Keep valid backing buffers so this case specifically exercises
            // the zero-channel/grid guard rather than failing on null data.
            fastllm::Data emptyCache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                      {batch, channels, 4}, initialCacheValues);
            fastllm::Data emptyTokens(allTokens);
            fastllm::Data emptyWeight(weight);
            fastllm::Data emptyBias(bias);
            emptyCache.Resize({batch, 0, 4});
            emptyTokens.Resize({batch, 0, 1});
            emptyWeight.Resize({0, 4});
            emptyBias.Resize({0});
            fastllm::Data output = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {batch, channels, 1},
                MakeRegressionValues(rows, 6.1f, 0.1f));
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       emptyCache, emptyTokens, emptyWeight, emptyBias, output, nullptr, 0),
                   "multi-token conv accepted zero channels");
        }
        {
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data snapshots[3];
            fastllm::Data *snapshotPtrs[3] = {&snapshots[0], &snapshots[1], &snapshots[2]};
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, allTokens, weight, bias, output, snapshotPtrs, 3),
                       "multi-token conv accepted a snapshot count larger than N");
        }
        {
            const int tooManyTokens = 7;
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data tokens = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {batch, channels, tooManyTokens},
                MakeRegressionValues(rows * tooManyTokens, 6.7f, 0.4f));
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, tokens, weight, bias, output, nullptr, 0),
                   "multi-token conv accepted N=7");
        }
    }

    bool RunCudaCrossDeviceViewRejectionRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "cross-device CUDA view regression: SKIP (two GPUs required)\n";
            return false;
        }

        const int originalDevice = FastllmCudaGetDevice();
        const int batch = 1;
        const int channels = 2;
        FastllmCudaSetDevice(0);
        fastllm::Data cache = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 4},
            MakeRegressionValues(batch * channels * 4, 7.3f, 0.2f));
        fastllm::Data weight = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {channels, 4},
            MakeRegressionValues(channels * 4, 7.7f, 0.1f));
        fastllm::Data bias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {channels},
            MakeRegressionValues(channels, 8.1f, 0.05f));

        FastllmCudaSetDevice(1);
        fastllm::Data remoteTokens = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 2},
            MakeRegressionValues(batch * channels * 2, 8.5f, 0.2f));
        fastllm::Data remoteView;
        remoteView.FakeFrom(remoteTokens, 0);
        remoteView.dims = remoteTokens.dims;
        remoteView.strides = remoteTokens.strides;
        Expect(remoteView.dataDeviceIds.empty(),
               "cross-device fake view unexpectedly inherited device IDs");

        FastllmCudaSetDevice(0);
        fastllm::Data output;
        Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                   cache, remoteView, weight, bias, output, nullptr, 0),
               "multi-token conv accepted a CUDA view owned by another device");
        remoteView.dataDeviceIds = {0};
        Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                   cache, remoteView, weight, bias, output, nullptr, 0),
               "multi-token conv trusted a stale CUDA view device ID");

        // A legal view may omit metadata. Its actual pointer device must still
        // control where a reusable destination is migrated and allocated.
        FastllmCudaSetDevice(1);
        fastllm::Data localCache = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 4},
            MakeRegressionValues(batch * channels * 4, 8.9f, 0.2f));
        fastllm::Data localTokens = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 2},
            MakeRegressionValues(batch * channels * 2, 9.3f, 0.2f));
        fastllm::Data localWeight = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {channels, 4},
            MakeRegressionValues(channels * 4, 9.7f, 0.1f));
        fastllm::Data localBias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {channels},
            MakeRegressionValues(channels, 10.1f, 0.05f));
        localCache.dataDeviceIds.clear();

        FastllmCudaSetDevice(0);
        fastllm::Data reusableOutput = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 2},
            MakeRegressionValues(batch * channels * 2, 10.5f, 0.1f));
        Expect(GetPointerDeviceId(reusableOutput.cudaData) == 0,
               "cross-device output fixture was not allocated on GPU 0");
        FastllmCudaSetDevice(1);
        Expect(FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                   localCache, localTokens, localWeight, localBias,
                   reusableOutput, nullptr, 0),
               "multi-token conv could not migrate a reusable output to the cache device");
        Expect(GetPointerDeviceId(reusableOutput.cudaData) == 1 &&
                   reusableOutput.dataDeviceIds == std::vector<int>({1}),
               "multi-token conv left its output on the wrong CUDA device");
        FastllmCudaSetDevice(originalDevice);
        return true;
    }

    void RunMultiCudaReplicatedExpansionRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "multi-CUDA replicated expansion regression: SKIP (two GPUs required)\n";
            return;
        }

        const int originalDevice = FastllmCudaGetDevice();
        FastllmCudaSetDevice(0);
        fastllm::Data data = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, 3, 8}, MakeRegressionValues(24, 10.9f, 0.2f));
        data.Expansion({1, 16, 8});
        const size_t bytes = data.GetBytes();
        Expect(bytes == data.expansionBytes && bytes > 24 * sizeof(float),
               "expanded replication fixture did not retain padded backing storage");

        std::vector<uint8_t> expected(bytes);
        FastllmCudaCopyFromDeviceToHost(expected.data(), data.cudaData, bytes);
        std::vector<int> devices = {0, 1};
        PrepareMultiCudaReplicatedData(data, devices, true);
        Expect(data.IsTensorParallelReplicated() && data.multiDeviceData,
               "expanded tensor did not become a replicated multi-CUDA tensor");

        for (int device : devices) {
            auto it = data.multiDeviceDatas.find(device);
            Expect(it != data.multiDeviceDatas.end() && it->second != nullptr,
                   "expanded tensor is missing a device replica");
            fastllm::Data *local = it->second;
            Expect(local->dims == data.dims && local->strides == data.strides &&
                       local->expansionDims == data.expansionDims &&
                       local->expansionBytes >= bytes,
                   "expanded tensor replica lost its backing layout");
            Expect(GetPointerDeviceId(local->cudaData) == device,
                   "expanded tensor replica was allocated on the wrong device");
            FastllmCudaSetDevice(device);
            std::vector<uint8_t> actual(bytes);
            FastllmCudaCopyFromDeviceToHost(actual.data(), local->cudaData, bytes);
            Expect(actual == expected, "expanded tensor replica data mismatch");
        }
        FastllmCudaSetDevice(originalDevice);
    }

#ifndef USE_ROCM
    void RunMultiCudaInt4GroupColumnSplitRegression() {
        constexpr int rows = 3;
        constexpr int columns = 128;
        constexpr int groupCnt = 32;
        constexpr int globalGroup = columns / groupCnt;
        constexpr int splitBegin = 32;
        constexpr int splitEnd = 96;
        constexpr int localColumns = splitEnd - splitBegin;
        constexpr int localGroup = localColumns / groupCnt;

        const int originalDevice = FastllmCudaGetDevice();
        FastllmCudaSetDevice(0);
        fastllm::Data weight(fastllm::DataType::INT4_GROUP, {rows, columns});
        weight.name = "regression.int4_group_column_split.weight";
        weight.group = globalGroup;
        weight.groupCnt = groupCnt;
        weight.perChannelAxis = 0;
        weight.scales.resize(rows * globalGroup);
        weight.mins.resize(rows * globalGroup);
        weight.zeros.resize(rows * globalGroup);
        weight.Allocate(true);
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < globalGroup; group++) {
                size_t index = static_cast<size_t>(row) * globalGroup + group;
                weight.scales[index] = 100.0f * row + group + 0.25f;
                weight.mins[index] = -100.0f * row - group - 0.5f;
                weight.zeros[index] = row * globalGroup + group + 1;
            }
        }
        weight.ToDevice(fastllm::DataDevice::CUDA, {0});

        fastllm::Data bias;
        std::vector<int> devices = {0};
        DivisionScheme scheme;
        scheme[0] = {{splitBegin, splitEnd}};
        Expect(SplitMultiCudaWeight(weight, bias, devices, scheme, 1, true),
               "failed to split INT4_GROUP weight by columns");

        auto localIt = weight.multiDeviceDatas.find(0);
        Expect(localIt != weight.multiDeviceDatas.end() && localIt->second != nullptr,
               "INT4_GROUP column split did not create the local tensor");
        fastllm::Data *local = localIt->second;
        Expect(local->dims == std::vector<int>({rows, localColumns}),
               "INT4_GROUP column split produced the wrong local shape");
        Expect(local->group == localGroup && local->groupCnt == groupCnt,
               "INT4_GROUP column split kept the global group count");
        Expect(local->scales.size() == static_cast<size_t>(rows * localGroup) &&
                   local->mins.size() == static_cast<size_t>(rows * localGroup) &&
                   local->zeros.size() == static_cast<size_t>(rows * localGroup),
               "INT4_GROUP column split produced the wrong quantization metadata shape");
        for (int row = 0; row < rows; row++) {
            for (int group = 0; group < localGroup; group++) {
                size_t localIndex = static_cast<size_t>(row) * localGroup + group;
                size_t sourceIndex = static_cast<size_t>(row) * globalGroup +
                                     splitBegin / groupCnt + group;
                Expect(local->scales[localIndex] == weight.scales[sourceIndex] &&
                           local->mins[localIndex] == weight.mins[sourceIndex] &&
                           local->zeros[localIndex] == weight.zeros[sourceIndex],
                       "INT4_GROUP column split copied quantization metadata from the wrong row/group");
            }
        }

        FastllmCudaSetDevice(originalDevice);
    }

    bool RunMultiCudaLargeWeightOffsetRegression() {
        if (FastllmCudaGetDeviceCount() < 1) {
            std::cout << "multi-CUDA large-weight offset regression: SKIP (CUDA unavailable)\n";
            return false;
        }

        constexpr int rows = 248320;
        constexpr int columns = 5120;
        constexpr int lateRow = 217280;
        const size_t rowBytes = static_cast<size_t>(columns) * sizeof(uint16_t);
        const size_t lateOffset = static_cast<size_t>(lateRow) * rowBytes;
        const size_t totalBytes = static_cast<size_t>(rows) * rowBytes;
        Expect(lateOffset > 0x7fffffffULL,
               "large-weight regression does not cross the int32 byte-offset boundary");

        const int originalDevice = FastllmCudaGetDevice();
        FastllmCudaSetDevice(0);
        void *managed = nullptr;
        cudaError_t state = cudaMallocManaged(&managed, totalBytes);
        if (state != cudaSuccess) {
            cudaGetLastError();
            FastllmCudaSetDevice(originalDevice);
            std::cout << "multi-CUDA large-weight offset regression: SKIP (managed allocation unavailable)\n";
            return false;
        }

        fastllm::Data weight(fastllm::DataType::BFLOAT16, {rows, columns});
        weight.name = "regression.large_lm_head.weight";
        weight.dataDevice = fastllm::DataDevice::CUDA;
        weight.dataDeviceIds = {0};
        weight.cudaData = managed;
        // SplitMultiCudaWeight consumes and releases the root CUDA storage.
        // Treat the managed allocation as owned so its normal cleanup path is
        // exercised without reserving the full logical tensor physically.
        weight.cudaDataBorrowed = false;
        fastllm::Data bias;
        std::vector<int> devices = {0};
        DivisionScheme scheme;
        scheme[0] = {{0, 1}, {lateRow, lateRow + 1}};

        state = cudaMemset(managed, 0x11, rowBytes);
        if (state == cudaSuccess) {
            state = cudaMemset(static_cast<uint8_t*>(managed) + lateOffset,
                               0xa5, rowBytes);
        }
        Expect(state == cudaSuccess, "failed to initialize managed large-weight rows");
        Expect(SplitMultiCudaWeight(weight, bias, devices, scheme, 0, true),
               "failed to split a weight whose source offset exceeds INT_MAX");

        auto localIt = weight.multiDeviceDatas.find(0);
        Expect(localIt != weight.multiDeviceDatas.end() && localIt->second != nullptr,
               "large-weight split did not create the local tensor");
        fastllm::Data *local = localIt->second;
        Expect(local->dims == std::vector<int>({2, columns}),
               "large-weight split produced the wrong local shape");
        std::vector<uint8_t> actual(2 * rowBytes);
        state = cudaMemcpy(actual.data(), local->cudaData, actual.size(),
                           cudaMemcpyDeviceToHost);
        Expect(state == cudaSuccess, "failed to copy the split large-weight rows");
        Expect(std::all_of(actual.begin(), actual.begin() + rowBytes,
                           [](uint8_t value) { return value == 0x11; }),
               "large-weight split corrupted the first source row");
        Expect(std::all_of(actual.begin() + rowBytes, actual.end(),
                           [](uint8_t value) { return value == 0xa5; }),
               "large-weight split used a truncated source offset");

        FastllmCudaSetDevice(originalDevice);
        return true;
    }
#endif

    void RunCudaRecurrentSnapshotsRegression() {
        FastllmCudaSetDevice(0);
        const int numKHeads = 1;
        const int numVHeads = 2;
        const int headKDim = 128;
        const int headVDim = 9;
        const int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
        const float eps = 1e-6f;
        const float qScale = 1.0f / std::sqrt((float) headKDim);

        std::vector<float> normValues(headKDim);
        for (int i = 0; i < headKDim; i++) {
            normValues[i] = 0.85f + 0.08f * std::cos((i + 1) * 0.031f);
        }
        std::vector<float> initialStateValues =
            MakeRegressionValues(numVHeads * headVDim * headKDim, 0.9f, 0.025f);
        fastllm::Data normWeight = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                                  {headKDim}, normValues);
        fastllm::Data aLog = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                            {numVHeads}, {-0.7f, -0.55f});
        fastllm::Data dtBias = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                              {numVHeads}, {0.15f, -0.08f});

        for (int tokenCount = 2; tokenCount <= 6; tokenCount++) {
            std::vector<float> convValues =
                MakeRegressionValues(tokenCount * qkvDim, 2.1f + tokenCount, 0.12f);
            std::vector<float> baValues(tokenCount * numVHeads * 2);
            for (int token = 0; token < tokenCount; token++) {
                for (int head = 0; head < numVHeads; head++) {
                    baValues[(size_t)token * numVHeads * 2 + head] =
                        -4.5f + 0.04f * token - 0.03f * head;
                    baValues[(size_t)token * numVHeads * 2 + numVHeads + head] =
                        -0.35f + 0.03f * token + 0.02f * head;
                }
            }

            fastllm::Data convSequence = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                        {1, tokenCount, qkvDim}, convValues);
            fastllm::Data baSequence = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                      {1, tokenCount, numVHeads * 2}, baValues);
            fastllm::Data sequentialState = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                           {1, numVHeads, headKDim, headVDim},
                                                           initialStateValues);
            fastllm::Data sequenceState = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                         {1, numVHeads, headKDim, headVDim},
                                                         initialStateValues);
            sequentialState.isLinearAttentionTransposed = true;
            sequenceState.isLinearAttentionTransposed = true;

            std::vector<std::vector<float> > expectedOutputs(tokenCount);
            std::vector<std::vector<float> > expectedStates(tokenCount);
            for (int token = 0; token < tokenCount; token++) {
                std::vector<float> singleConv(qkvDim);
                std::copy(convValues.begin() + (size_t) token * qkvDim,
                          convValues.begin() + (size_t) (token + 1) * qkvDim,
                          singleConv.begin());
                std::vector<float> singleBa(numVHeads * 2);
                std::copy(baValues.begin() + (size_t) token * numVHeads * 2,
                          baValues.begin() + (size_t) (token + 1) * numVHeads * 2,
                          singleBa.begin());
                fastllm::Data convToken = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                         {1, 1, qkvDim}, singleConv);
                fastllm::Data baToken = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                       {1, 1, numVHeads * 2}, singleBa);
                fastllm::Data singleOutput;
                Expect(FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
                           convToken, baToken, normWeight, aLog, dtBias,
                           sequentialState, singleOutput,
                           numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                       "single-token recurrent reference rejected N=" +
                       std::to_string(tokenCount) + ", token=" + std::to_string(token));
                expectedOutputs[token] = ToFloatVector(singleOutput);
                expectedStates[token] = ToFloatVector(sequentialState);
            }

            int snapshotCount = std::min(tokenCount, 5);
            std::vector<fastllm::Data> snapshots(snapshotCount);
            std::vector<fastllm::Data*> snapshotPtrs(snapshotCount);
            for (int token = 0; token < snapshotCount; token++) {
                snapshotPtrs[token] = &snapshots[token];
            }
            fastllm::Data sequenceOutput;
            Expect(FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
                       convSequence, baSequence, normWeight, aLog, dtBias,
                       sequenceState, sequenceOutput, snapshotPtrs.data(), snapshotCount,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                   "recurrent snapshot sequence rejected N=" + std::to_string(tokenCount));
            ExpectCudaTensorMeta(sequenceOutput, fastllm::DataType::FLOAT16,
                                 {1, tokenCount, numVHeads, headVDim},
                                 "recurrent sequence output metadata");

            fastllm::Data noSnapshotState = MakeCudaTensor(
                fastllm::DataType::FLOAT16,
                {1, numVHeads, headKDim, headVDim}, initialStateValues);
            noSnapshotState.isLinearAttentionTransposed = true;
            fastllm::Data noSnapshotOutput;
            Expect(FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16(
                       convSequence, baSequence, normWeight, aLog, dtBias,
                       noSnapshotState, noSnapshotOutput,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                   "recurrent no-snapshot sequence rejected N=" +
                   std::to_string(tokenCount));
            ExpectCudaTensorMeta(noSnapshotOutput, fastllm::DataType::FLOAT16,
                                 {1, tokenCount, numVHeads, headVDim},
                                 "recurrent no-snapshot output metadata");
            ExpectFloatNear(ToFloatVector(sequenceOutput), ToFloatVector(noSnapshotOutput),
                            1e-3f, 1e-3f,
                            "recurrent no-snapshot output N=" + std::to_string(tokenCount));
            ExpectFloatNear(ToFloatVector(sequenceState), ToFloatVector(noSnapshotState),
                            1e-3f, 1e-3f,
                            "recurrent no-snapshot final state N=" + std::to_string(tokenCount));

            std::vector<float> actualOutput = ToFloatVector(sequenceOutput);
            const int outputRows = numVHeads * headVDim;
            for (int token = 0; token < tokenCount; token++) {
                std::vector<float> actualTokenOutput(
                    actualOutput.begin() + (size_t) token * outputRows,
                    actualOutput.begin() + (size_t) (token + 1) * outputRows);
                std::string suffix = " N=" + std::to_string(tokenCount) +
                                     ", token=" + std::to_string(token);
                ExpectFloatNear(expectedOutputs[token], actualTokenOutput,
                                2e-3f, 2e-3f, "recurrent sequence output" + suffix);
                if (token < snapshotCount) {
                    ExpectFloatNear(expectedStates[token], ToFloatVector(snapshots[token]),
                                    2e-3f, 2e-3f, "recurrent state snapshot" + suffix);
                    Expect(snapshots[token].isLinearAttentionTransposed,
                           "recurrent snapshot lost transposed layout marker" + suffix);
                    ExpectCudaTensorMeta(snapshots[token], fastllm::DataType::FLOAT16,
                                         {1, numVHeads, headKDim, headVDim},
                                         "recurrent snapshot metadata" + suffix);
                }
            }
            ExpectFloatNear(ToFloatVector(sequentialState), ToFloatVector(sequenceState),
                            2e-3f, 2e-3f,
                            "recurrent final state N=" + std::to_string(tokenCount));
            if (tokenCount == 6) {
                fastllm::Data rejectedState = MakeCudaTensor(
                    fastllm::DataType::FLOAT16,
                    {1, numVHeads, headKDim, headVDim}, initialStateValues);
                rejectedState.isLinearAttentionTransposed = true;
                fastllm::Data rejectedSnapshots[6];
                fastllm::Data *rejectedSnapshotPtrs[6];
                for (int token = 0; token < 6; token++) {
                    rejectedSnapshotPtrs[token] = &rejectedSnapshots[token];
                }
                fastllm::Data rejectedOutput;
                Expect(!FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
                           convSequence, baSequence, normWeight, aLog, dtBias,
                           rejectedState, rejectedOutput, rejectedSnapshotPtrs, 6,
                           numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                       "recurrent sequence accepted N=6 with six snapshots");
            }
        }

        std::vector<float> oneConvValues = MakeRegressionValues(qkvDim, 5.2f, 0.12f);
        std::vector<float> oneBaValues = {-4.5f, -4.53f, -0.35f, -0.33f};
        {
            fastllm::Data conv = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                {1, 1, qkvDim}, oneConvValues);
            fastllm::Data ba = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                              {1, 1, numVHeads * 2}, oneBaValues);
            fastllm::Data state = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {1, numVHeads, headKDim, headVDim},
                                                 initialStateValues);
            state.isLinearAttentionTransposed = true;
            fastllm::Data snapshot;
            fastllm::Data *snapshotPtr = &snapshot;
            fastllm::Data output;
            Expect(!FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
                       conv, ba, normWeight, aLog, dtBias, state, output, &snapshotPtr, 1,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                   "recurrent snapshot sequence accepted N=1");
        }
        {
            std::vector<float> twoConvValues = MakeRegressionValues(2 * qkvDim, 5.8f, 0.12f);
            std::vector<float> twoBaValues = {
                -4.5f, -4.53f, -0.35f, -0.33f,
                -4.46f, -4.49f, -0.32f, -0.30f
            };
            fastllm::Data conv = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                {1, 2, qkvDim}, twoConvValues);
            fastllm::Data ba = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                              {1, 2, numVHeads * 2}, twoBaValues);
            fastllm::Data state = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {1, numVHeads, headKDim, headVDim},
                                                 initialStateValues);
            state.isLinearAttentionTransposed = true;
            fastllm::Data output;
            Expect(!FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
                       conv, ba, normWeight, aLog, dtBias, state, output, nullptr, 2,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                       "recurrent snapshot sequence accepted nullptr tokenStates");
        }
        {
            const int tooManyTokens = 7;
            fastllm::Data conv = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {1, tooManyTokens, qkvDim},
                MakeRegressionValues(tooManyTokens * qkvDim, 6.4f, 0.12f));
            fastllm::Data ba = MakeCudaTensor(
                fastllm::DataType::FLOAT16,
                {1, tooManyTokens, numVHeads * 2},
                MakeRegressionValues(tooManyTokens * numVHeads * 2, 6.9f, 0.08f));
            fastllm::Data state = MakeCudaTensor(
                fastllm::DataType::FLOAT16,
                {1, numVHeads, headKDim, headVDim}, initialStateValues);
            state.isLinearAttentionTransposed = true;
            fastllm::Data output;
            Expect(!FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16(
                       conv, ba, normWeight, aLog, dtBias, state, output,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                   "recurrent sequence accepted N=7");
        }
    }
#endif

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

    float DecodeFP8E4M3(uint8_t value) {
        const float sign = (value & 0x80) ? -1.0f : 1.0f;
        const int exponent = (value >> 3) & 0x0f;
        const int mantissa = value & 0x07;
        if (exponent == 0) {
            return sign * std::ldexp((float)mantissa, -9);
        }
        return sign * std::ldexp(1.0f + (float)mantissa / 8.0f, exponent - 7);
    }

    void RunDiskOperatorsRegression() {
        // Exercise multiple decode chunks without creating a 64 MiB fixture.
        setenv("FASTLLM_DISK_LINEAR_CHUNK_MB", "1", 1);
        constexpr int inputDim = 256;
        constexpr int outputDim = 8704;
        constexpr int splitRows = 4352;
        constexpr int blockK = 128;
        constexpr int blockM = 128;
        constexpr int batch = 2;
        const int scaleColumns = (inputDim + blockM - 1) / blockM;

        ScopedTempDirectory temp("fastllm_disk_linear_");
        const std::filesystem::path weightPath = temp.Path() / "weights.bin";
        std::vector<uint8_t> fp8Weights((size_t)outputDim * inputDim);
        for (int row = 0; row < outputDim; row++) {
            for (int column = 0; column < inputDim; column++) {
                uint8_t magnitude = (uint8_t)((row * 17 + column * 29 + 3) % 127);
                fp8Weights[(size_t)row * inputDim + column] =
                    magnitude | (((row + column) & 1) ? 0x80 : 0x00);
            }
        }
        std::vector<float> scales((size_t)((outputDim + blockK - 1) / blockK) *
                                  scaleColumns);
        for (size_t i = 0; i < scales.size(); i++) {
            scales[i] = 0.0005f * (float)(1 + (i % 13));
        }
        std::vector<float> biasValues(outputDim);
        for (int row = 0; row < outputDim; row++) {
            biasValues[row] = (float)((row * 7) % 31 - 15) / 32.0f;
        }

        constexpr int vocabSize = 9;
        constexpr int embeddingDim = 17;
        std::vector<uint16_t> sourceEmbedding((size_t)vocabSize * embeddingDim);
        std::vector<float> roundedEmbedding(sourceEmbedding.size());
        for (size_t i = 0; i < sourceEmbedding.size(); i++) {
            float value = (float)((int)(i * 11 % 101) - 50) / 19.0f;
            sourceEmbedding[i] = fastllm::Float32ToBFloat16RNEBits(value);
            roundedEmbedding[i] = fastllm::BFloat16BitsToFloat32(sourceEmbedding[i]);
        }
        const uint64_t embeddingOffset = fp8Weights.size();
        {
            std::ofstream output(weightPath, std::ios::binary);
            Expect(output.good(), "failed to create disk operator fixture.");
            output.write(reinterpret_cast<const char*>(fp8Weights.data()),
                         fp8Weights.size());
            output.write(reinterpret_cast<const char*>(sourceEmbedding.data()),
                         sourceEmbedding.size() * sizeof(uint16_t));
            Expect(output.good(), "failed to write disk operator fixture.");
        }

        fastllm::Data residentWeight(fastllm::DataType::FP8_E4M3,
                                     {outputDim, inputDim});
        residentWeight.blockK = blockK;
        residentWeight.blockM = blockM;
        residentWeight.scales = scales;
        residentWeight.Allocate();
        memcpy(residentWeight.cpuData, fp8Weights.data(), fp8Weights.size());

        fastllm::Data diskWeight(fastllm::DataType::FP8_E4M3,
                                 {outputDim, inputDim});
        diskWeight.name = "regression.disk.fp8.weight";
        diskWeight.blockK = blockK;
        diskWeight.blockM = blockM;
        diskWeight.scales = scales;
        diskWeight.weightType = fastllm::WeightType::LINEAR;
        diskWeight.isDiskWeight = true;
        for (const auto &range : {std::pair<int, int>{0, splitRows},
                                  std::pair<int, int>{splitRows, outputDim}}) {
            fastllm::DiskWeightPart part;
            part.fileName = weightPath.string();
            part.fileOffset = (long long)range.first * inputDim;
            part.bytes = (uint64_t)(range.second - range.first) * inputDim;
            part.sourceDataType = fastllm::DataType::FP8_E4M3;
            part.dims = {range.second - range.first, inputDim};
            diskWeight.diskWeightParts.push_back(part);
        }

        std::vector<float> inputValues((size_t)batch * inputDim);
        for (int b = 0; b < batch; b++) {
            for (int column = 0; column < inputDim; column++) {
                inputValues[(size_t)b * inputDim + column] =
                    (float)((column * (b + 5) + 13 * b) % 97 - 48) / 37.0f;
            }
        }
        fastllm::Data input(fastllm::DataType::FLOAT32,
                            {batch, inputDim}, inputValues);
        fastllm::Data bias(fastllm::DataType::FLOAT32, {outputDim}, biasValues);
        fastllm::Data cpuOutput, diskOutput;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(input, residentWeight, bias, cpuOutput);
        }
        {
            ScopedFirstDevice guard("disk");
            fastllm::Linear(input, diskWeight, bias, diskOutput);
        }
        ExpectFloatNear(ToFloatVector(cpuOutput), ToFloatVector(diskOutput),
                        2e-4f, 2e-4f, "disk transient-prefill FP8 linear batch 2");
        Expect(diskWeight.cpuData == nullptr,
               "disk prefill retained the Linear weight after the operator finished.");

        std::vector<float> manual(outputDim);
        for (int row = 0; row < outputDim; row++) {
            float sum = biasValues[row];
            for (int column = 0; column < inputDim; column++) {
                float roundedInput = fastllm::BFloat16BitsToFloat32(
                    fastllm::Float32ToBFloat16RNEBits(inputValues[column]));
                float scale = scales[(size_t)(row / blockK) * scaleColumns +
                                     column / blockM];
                sum += roundedInput *
                       DecodeFP8E4M3(fp8Weights[(size_t)row * inputDim + column]) *
                       scale;
            }
            manual[row] = sum;
        }
        std::vector<float> cpuValues = ToFloatVector(cpuOutput);
        cpuValues.resize(outputDim);
        ExpectFloatNear(manual, cpuValues, 3e-3f, 3e-3f,
                        "ARM/vector FP8 linear numerical reference");

        fastllm::Data singleInput(fastllm::DataType::FLOAT32, {1, inputDim},
                                  std::vector<float>(inputValues.begin(),
                                                     inputValues.begin() + inputDim));
        std::vector<fastllm::Data*> mergeWeights = {
            nullptr, nullptr, &diskWeight, &diskWeight
        };
        std::vector<fastllm::Data*> mergeBiass(mergeWeights.size(), nullptr);
        {
            ScopedFirstDevice guard("disk");
            Expect(fastllm::CanRunMergeMOE(singleInput, mergeWeights, mergeBiass),
                   "disk MergeMOE capability probe did not receive disk weights.");
        }

        // A multi-token MergeMOE uses w2 as a gathered-input scratch buffer.
        // Reproduce the model warmup case where that reusable tensor still has
        // a BF16 allocation but the current MoE input is FLOAT32.
        constexpr int moeHidden = 16;
        constexpr int moeInter = 8;
        constexpr int moeBatch = 32;
        constexpr int moeBlock = 8;
        const std::filesystem::path moePath = temp.Path() / "moe.bin";
        std::vector<uint8_t> moeGateBytes((size_t)moeInter * 2 * moeHidden);
        std::vector<uint8_t> moeDownBytes((size_t)moeHidden * moeInter);
        for (size_t i = 0; i < moeGateBytes.size(); i++) {
            moeGateBytes[i] = (uint8_t)(0x20 + i % 16) |
                              ((i & 7) == 0 ? 0x80 : 0x00);
        }
        for (size_t i = 0; i < moeDownBytes.size(); i++) {
            moeDownBytes[i] = (uint8_t)(0x18 + i % 16) |
                              ((i & 5) == 0 ? 0x80 : 0x00);
        }
        {
            std::ofstream output(moePath, std::ios::binary);
            Expect(output.good(), "failed to create disk MergeMOE fixture.");
            output.write(reinterpret_cast<const char*>(moeGateBytes.data()),
                         moeGateBytes.size());
            output.write(reinterpret_cast<const char*>(moeDownBytes.data()),
                         moeDownBytes.size());
            Expect(output.good(), "failed to write disk MergeMOE fixture.");
        }

        auto initMoeWeight = [&](fastllm::Data &weight,
                                 const std::vector<int> &dims,
                                 const std::vector<uint8_t> &bytes,
                                 long long fileOffset, bool disk) {
            weight.dataType = fastllm::DataType::FP8_E4M3;
            weight.Resize(dims);
            weight.blockK = moeBlock;
            weight.blockM = moeBlock;
            int scaleRows = (dims[0] + moeBlock - 1) / moeBlock;
            int scaleColumns = (dims[1] + moeBlock - 1) / moeBlock;
            weight.scales.resize((size_t)scaleRows * scaleColumns);
            for (size_t i = 0; i < weight.scales.size(); i++) {
                weight.scales[i] = 0.01f * (float)(i + 1);
            }
            weight.weightType = fastllm::WeightType::LINEAR;
            if (disk) {
                weight.isDiskWeight = true;
                fastllm::DiskWeightPart part;
                part.fileName = moePath.string();
                part.fileOffset = fileOffset;
                part.bytes = bytes.size();
                part.sourceDataType = fastllm::DataType::FP8_E4M3;
                part.dims = dims;
                weight.diskWeightParts.push_back(part);
            } else {
                weight.Allocate(false);
                memcpy(weight.cpuData, bytes.data(), bytes.size());
            }
        };
        fastllm::Data residentMoeGate, residentMoeDown;
        fastllm::Data diskMoeGate, diskMoeDown;
        initMoeWeight(residentMoeGate, {moeInter * 2, moeHidden},
                      moeGateBytes, 0, false);
        initMoeWeight(residentMoeDown, {moeHidden, moeInter},
                      moeDownBytes, moeGateBytes.size(), false);
        initMoeWeight(diskMoeGate, {moeInter * 2, moeHidden},
                      moeGateBytes, 0, true);
        initMoeWeight(diskMoeDown, {moeHidden, moeInter},
                      moeDownBytes, moeGateBytes.size(), true);

        std::vector<float> moeInputValues((size_t)moeBatch * moeHidden);
        for (size_t i = 0; i < moeInputValues.size(); i++) {
            moeInputValues[i] = (float)((int)(i * 13 % 67) - 33) / 29.0f;
        }
        fastllm::Data moeInput(fastllm::DataType::FLOAT32,
                               {moeBatch, moeHidden}, moeInputValues);
        fastllm::Data moeIndex = MakeIntTensor(
            {moeBatch, 1}, std::vector<int32_t>(moeBatch, 0));
        fastllm::Data moeScore(fastllm::DataType::FLOAT32, {moeBatch, 1},
                               std::vector<float>(moeBatch, 1.0f));
        std::vector<fastllm::Data*> residentMoeWeights = {
            nullptr, nullptr, &residentMoeGate, &residentMoeDown
        };
        std::vector<fastllm::Data*> diskMoeWeights = {
            nullptr, nullptr, &diskMoeGate, &diskMoeDown
        };
        std::vector<fastllm::Data*> moeBiass(4, nullptr);
        fastllm::Data cpuMoeW1, cpuMoeW2, cpuMoeW3;
        fastllm::Data cpuMoeInput, cpuMoeIntermediate;
        fastllm::Data cpuMoeOutput(fastllm::DataType::FLOAT32,
                                   {moeBatch, moeHidden});
        {
            ScopedFirstDevice guard("cpu");
            fastllm::MergeMOE(
                moeInput, moeIndex, moeScore, residentMoeWeights, moeBiass,
                cpuMoeW1, cpuMoeW2, cpuMoeW3,
                cpuMoeInput, cpuMoeIntermediate, 0.0f, cpuMoeOutput);
        }

        fastllm::Data diskMoeW1(fastllm::DataType::BFLOAT16,
                                {moeBatch, moeInter});
        diskMoeW1.Allocate(false);
        fastllm::Data diskMoeW3(fastllm::DataType::BFLOAT16,
                                {moeBatch, moeInter * 2});
        diskMoeW3.Allocate(false);
        fastllm::Data diskMoeW2(fastllm::DataType::BFLOAT16,
                                {moeBatch, moeHidden});
        diskMoeW2.Allocate(false);
        fastllm::Data diskMoeInput, diskMoeIntermediate;
        fastllm::Data diskMoeOutput(fastllm::DataType::FLOAT32,
                                    {moeBatch, moeHidden});
        {
            ScopedFirstDevice guard("disk");
            fastllm::MergeMOE(
                moeInput, moeIndex, moeScore, diskMoeWeights, moeBiass,
                diskMoeW1, diskMoeW2, diskMoeW3,
                diskMoeInput, diskMoeIntermediate, 0.0f, diskMoeOutput);
        }
        ExpectFloatNear(ToFloatVector(cpuMoeOutput), ToFloatVector(diskMoeOutput),
                        2e-4f, 2e-4f,
                        "disk MergeMOE transient weights and scratch dtype reuse");
        Expect(diskMoeGate.cpuData == nullptr && diskMoeDown.cpuData == nullptr,
               "disk MergeMOE retained an expert weight after the operator finished.");

        fastllm::Data singleCpuOutput, singleDiskOutput;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(singleInput, residentWeight, bias, singleCpuOutput);
        }
        {
            ScopedFirstDevice guard("disk");
            fastllm::Linear(singleInput, diskWeight, bias, singleDiskOutput);
        }
        ExpectFloatNear(ToFloatVector(singleCpuOutput), ToFloatVector(singleDiskOutput),
                        2e-4f, 2e-4f, "disk chunked FP8 linear batch 1");
        Expect(diskWeight.cpuData == nullptr,
               "disk Linear materialized the full resident weight.");

        fastllm::Data residentEmbedding(fastllm::DataType::FLOAT16,
                                        {vocabSize, embeddingDim}, roundedEmbedding);
        fastllm::Data diskEmbedding(fastllm::DataType::FLOAT16,
                                    {vocabSize, embeddingDim});
        diskEmbedding.name = "regression.disk.embedding.weight";
        diskEmbedding.weightType = fastllm::WeightType::EMBEDDING;
        diskEmbedding.isDiskWeight = true;
        fastllm::DiskWeightPart embeddingPart;
        embeddingPart.fileName = weightPath.string();
        embeddingPart.fileOffset = embeddingOffset;
        embeddingPart.bytes = sourceEmbedding.size() * sizeof(uint16_t);
        embeddingPart.sourceDataType = fastllm::DataType::BFLOAT16;
        embeddingPart.dims = {vocabSize, embeddingDim};
        diskEmbedding.diskWeightParts.push_back(embeddingPart);

        const std::vector<float> tokenValues = {5, 1, 5, 0, 8, 1};
        fastllm::Data tokenIds(fastllm::DataType::FLOAT32, {2, 3}, tokenValues);
        fastllm::Data cpuEmbedding, diskEmbeddingOutput;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Embedding(tokenIds, residentEmbedding, cpuEmbedding);
        }
        {
            ScopedFirstDevice guard("disk");
            fastllm::Embedding(tokenIds, diskEmbedding, diskEmbeddingOutput);
        }
        ExpectFloatNear(ToFloatVector(cpuEmbedding), ToFloatVector(diskEmbeddingOutput),
                        0.0f, 0.0f, "disk row-wise BF16 embedding");
        Expect(diskEmbedding.cpuData == nullptr,
               "disk Embedding materialized the full resident weight.");
    }

    void RunCpuInt4GroupAwqLinearRegressionCase(int inputDim, int outputDim,
                                                int groupCnt,
                                                const std::string &caseName) {
        const int batch = 3;
        Expect(inputDim % groupCnt == 0,
               "CPU AWQ regression requires complete quantization groups.");
        const int group = inputDim / groupCnt;

        std::vector<float> inputValues((size_t)batch * inputDim);
        for (int b = 0; b < batch; b++) {
            for (int x = 0; x < inputDim; x++) {
                inputValues[(size_t)b * inputDim + x] =
                    (float)((x * (b + 3) + 11 * b) % 251 - 125) / 64.0f;
            }
        }

        fastllm::Data quantWeight(fastllm::DataType::INT4_GROUP, {outputDim, inputDim});
        quantWeight.group = group;
        quantWeight.groupCnt = groupCnt;
        quantWeight.perChannelAxis = 0;
        quantWeight.scales.resize((size_t)outputDim * group);
        quantWeight.mins.resize((size_t)outputDim * group);
        quantWeight.zeros.resize((size_t)outputDim * group);
        quantWeight.Allocate(true);

        std::vector<float> floatWeightValues((size_t)outputDim * inputDim);
        for (int y = 0; y < outputDim; y++) {
            for (int g = 0; g < group; g++) {
                int zero = (y * 3 + g * 5 + 1) & 15;
                float scale = 0.0025f * (1 + ((y + 2 * g) % 7));
                int gid = y * group + g;
                quantWeight.scales[gid] = scale;
                quantWeight.mins[gid] = -scale * zero;
                quantWeight.zeros[gid] = zero;
                for (int x = g * groupCnt; x < (g + 1) * groupCnt; x++) {
                    int q = (y * 7 + x * 5 + g * 3) & 15;
                    size_t byteId = ((size_t)y * inputDim + x) / 2;
                    if ((x & 1) == 0) {
                        quantWeight.cpuData[byteId] |= (uint8_t)(q << 4);
                    } else {
                        quantWeight.cpuData[byteId] |= (uint8_t)q;
                    }
                    floatWeightValues[(size_t)y * inputDim + x] = (q - zero) * scale;
                }
            }
        }

        fastllm::Data input(fastllm::DataType::FLOAT32, {batch, inputDim}, inputValues);
        fastllm::Data floatWeight(fastllm::DataType::FLOAT32,
                                  {outputDim, inputDim}, floatWeightValues);
        fastllm::Data emptyBias, expected, actual;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(input, floatWeight, emptyBias, expected);
            fastllm::Linear(input, quantWeight, emptyBias, actual);
        }

        ExpectFloatNear(ToFloatVector(expected), ToFloatVector(actual),
                        3e-2f, 2e-2f,
                        "CPU AWQ-style INT4_GROUP linear " + caseName);
    }

    void RunCpuInt4GroupAwqLinearRegression() {
        RunCpuInt4GroupAwqLinearRegressionCase(2048, 1024, 32, "group32");
        RunCpuInt4GroupAwqLinearRegressionCase(384, 64, 64, "group64");
        RunCpuInt4GroupAwqLinearRegressionCase(384, 64, 96, "group96");
        RunCpuInt4GroupAwqLinearRegressionCase(384, 64, 128, "group128");
        RunCpuInt4GroupAwqLinearRegressionCase(72, 64, 24, "scalar-group24");
    }

#ifdef USE_CUDA
    void RunCudaGgufMmvqBatch8Regression() {
        constexpr int inputDim = QK_K;
        constexpr int outputDim = 64;
        constexpr int batch = 8;

        std::vector<float> weightValues((size_t)outputDim * inputDim);
        for (int row = 0; row < outputDim; row++) {
            for (int column = 0; column < inputDim; column++) {
                weightValues[(size_t)row * inputDim + column] =
                    std::sin((float)(row * 17 + column * 5 + 3) * 0.03125f) *
                    (0.25f + (float)((row + column) % 11) / 16.0f);
            }
        }

        std::vector<float> inputValues((size_t)batch * inputDim);
        for (int row = 0; row < batch; row++) {
            for (int column = 0; column < inputDim; column++) {
                inputValues[(size_t)row * inputDim + column] =
                    std::cos((float)(row * 13 + column * 7 + 1) * 0.01953125f) *
                    (0.5f + (float)((row * 3 + column) % 9) / 16.0f);
            }
        }

        for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q4_K}) {
            fastllm::Data weight(
                fastllm::DataType::DATA_GGUF_FORMAT, (int)type,
                {outputDim, inputDim});
            weight.name = "regression.cuda_gguf_mmvq_batch8." +
                std::string(ggml_type_name(type));
            weight.CreateFromOriData(
                fastllm::WeightType::LINEAR, fastllm::DataType::FLOAT32,
                (uint8_t*)weightValues.data(), nullptr, nullptr);
            weight.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, true);

            fastllm::Data emptyBias;
            fastllm::Data input8(
                fastllm::DataType::BFLOAT16,
                {batch, inputDim}, inputValues);
            input8.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
            fastllm::Data output8;
            {
                ScopedFirstDevice guard("cuda");
                fastllm::Linear(input8, weight, emptyBias, output8);
            }

            std::vector<float> firstSevenValues(
                inputValues.begin(),
                inputValues.begin() + (batch - 1) * inputDim);
            fastllm::Data input7(
                fastllm::DataType::BFLOAT16,
                {batch - 1, inputDim}, firstSevenValues);
            input7.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
            fastllm::Data output7;
            {
                ScopedFirstDevice guard("cuda");
                fastllm::Linear(input7, weight, emptyBias, output7);
            }

            std::vector<float> lastValue(
                inputValues.begin() + (batch - 1) * inputDim,
                inputValues.end());
            fastllm::Data input1(
                fastllm::DataType::BFLOAT16, {1, inputDim}, lastValue);
            input1.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
            fastllm::Data output1;
            {
                ScopedFirstDevice guard("cuda");
                fastllm::Linear(input1, weight, emptyBias, output1);
            }

            std::vector<float> expected = ToFloatVector(output7);
            std::vector<float> lastOutput = ToFloatVector(output1);
            expected.insert(expected.end(), lastOutput.begin(), lastOutput.end());
            ExpectFloatNear(
                expected, ToFloatVector(output8), 0.0f, 0.0f,
                "CUDA GGUF MMVQ batch 8 " +
                    std::string(ggml_type_name(type)));
        }
    }

    void RunCudaInt4Group32AwqLinearRegression() {
        const int inputDim = 128;
        const int outputDim = 64;
        const int groupCnt = 32;
        const int group = inputDim / groupCnt;

        fastllm::Data quantWeight(fastllm::DataType::INT4_GROUP, {outputDim, inputDim});
        quantWeight.name = "regression.cuda_int4_group32.weight";
        quantWeight.group = group;
        quantWeight.groupCnt = groupCnt;
        quantWeight.perChannelAxis = 0;
        quantWeight.scales.resize((size_t)outputDim * group);
        quantWeight.mins.resize((size_t)outputDim * group);
        quantWeight.zeros.resize((size_t)outputDim * group);
        quantWeight.Allocate(true);

        std::vector<float> floatWeightValues((size_t)outputDim * inputDim);
        for (int y = 0; y < outputDim; y++) {
            for (int g = 0; g < group; g++) {
                int zero = (y * 3 + g * 5 + 1) & 15;
                float scale = 0.0025f * (1 + ((y + 2 * g) % 7));
                int gid = y * group + g;
                quantWeight.scales[gid] = scale;
                quantWeight.mins[gid] = -scale * zero;
                quantWeight.zeros[gid] = zero;
                for (int x = g * groupCnt; x < (g + 1) * groupCnt; x++) {
                    int q = (y * 7 + x * 5 + g * 3) & 15;
                    size_t byteId = ((size_t)y * inputDim + x) / 2;
                    if ((x & 1) == 0) {
                        quantWeight.cpuData[byteId] |= (uint8_t)(q << 4);
                    } else {
                        quantWeight.cpuData[byteId] |= (uint8_t)q;
                    }
                    floatWeightValues[(size_t)y * inputDim + x] = (q - zero) * scale;
                }
            }
        }

        fastllm::Data floatWeight(fastllm::DataType::FLOAT32,
                                  {outputDim, inputDim}, floatWeightValues);
        quantWeight.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data emptyBias;

        for (int batch : {1, 3, 33}) {
            std::vector<float> inputValues((size_t)batch * inputDim);
            for (int b = 0; b < batch; b++) {
                for (int x = 0; x < inputDim; x++) {
                    inputValues[(size_t)b * inputDim + x] =
                        (float)((x * (b + 3) + 11 * b) % 97 - 48) / 64.0f;
                }
            }

            fastllm::Data inputFloat(fastllm::DataType::FLOAT32,
                                     {batch, inputDim}, inputValues);
            fastllm::Data expected;
            {
                ScopedFirstDevice guard("cpu");
                fastllm::Linear(inputFloat, floatWeight, emptyBias, expected);
            }

            fastllm::Data inputHalf(fastllm::DataType::FLOAT16,
                                    {batch, inputDim}, inputValues);
            inputHalf.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
            fastllm::Data actual;
            {
                ScopedFirstDevice guard("cuda");
                fastllm::Linear(inputHalf, quantWeight, emptyBias, actual);
            }
#if !defined(USE_ROCM) && !defined(CUDA_NO_TENSOR_CORE)
            int cudaDevice = 0, major = 0, minor = 0;
            if (cudaGetDevice(&cudaDevice) == cudaSuccess &&
                cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, cudaDevice) == cudaSuccess &&
                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, cudaDevice) == cudaSuccess &&
                major * 10 + minor >= 75) {
                Expect(quantWeight.cudaData == nullptr,
                       "CUDA INT4_GROUP(32) regression did not select Marlin on SM75+.");
            }
#endif
            ExpectFloatNear(ToFloatVector(expected), ToFloatVector(actual),
                            2e-2f, 2e-2f,
                            "CUDA AWQ-style INT4_GROUP(32) linear batch " + std::to_string(batch));
        }
    }

    void RunCudaCompactInt4Group32LinearRegression() {
        // Five groups exercise the non-padded final compact block as well as
        // the less-aligned CUDA load path used when groups % 4 != 0.
        constexpr int inputDim = 160;
        constexpr int outputDim = 64;
        constexpr int groups = inputDim / 32;
        const size_t rowBytes = fastllm::GetDataBytes(
            fastllm::DataType::INT4_GROUP32, 1, inputDim);

        fastllm::Data compactWeight(fastllm::DataType::INT4_GROUP32,
                                    {outputDim, inputDim});
        compactWeight.name = "regression.cuda_compact_int4_group32.weight";
        compactWeight.group = groups;
        compactWeight.groupCnt = 32;
        compactWeight.perChannelAxis = 0;
        compactWeight.Allocate(true);
        std::vector<float> floatWeightValues((size_t)outputDim * inputDim);
        for (int row = 0; row < outputDim; row++) {
            uint8_t *weightRow = compactWeight.cpuData + (size_t)row * rowBytes;
            for (int group = 0; group < groups; group++) {
                uint8_t *weightBlock = weightRow +
                    fastllm::GetInt4Group32DataOffset(group, groups);
                float sourceScale = 0.0025f * (float)(1 + ((row + group * 2) % 7));
                uint16_t scaleBits = fastllm::Float32ToBFloat16RNEBits(sourceScale);
                memcpy(weightRow +
                           fastllm::GetInt4Group32ScaleOffset(group, groups),
                       &scaleBits, sizeof(scaleBits));
                float scale = fastllm::BFloat16BitsToFloat32(scaleBits);
                for (int column = group * 32; column < (group + 1) * 32; column++) {
                    int q = (row * 7 + column * 5 + group * 3) & 15;
                    const int localColumn = column - group * 32;
                    if ((column & 1) == 0) {
                        weightBlock[localColumn / 2] |= (uint8_t)(q << 4);
                    } else {
                        weightBlock[localColumn / 2] |= (uint8_t)q;
                    }
                    floatWeightValues[(size_t)row * inputDim + column] =
                        (float)(q - 8) * scale;
                }
            }
        }
        fastllm::Data floatWeight(fastllm::DataType::FLOAT32,
                                  {outputDim, inputDim}, floatWeightValues);
        compactWeight.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data emptyBias;

        for (fastllm::DataType inputType : {fastllm::DataType::FLOAT16,
                                            fastllm::DataType::BFLOAT16,
                                            fastllm::DataType::FLOAT32}) {
            for (int batch : {1, 2, 4, 8, 9, 16}) {
                std::vector<float> values((size_t)batch * inputDim);
                for (int b = 0; b < batch; b++) {
                    for (int x = 0; x < inputDim; x++) {
                        values[(size_t)b * inputDim + x] =
                            (float)((x * (b + 5) + 13 * b) % 97 - 48) / 64.0f;
                    }
                }
                fastllm::Data referenceInput(inputType, {batch, inputDim}, values);
                fastllm::Data expected;
                {
                    ScopedFirstDevice guard("cpu");
                    fastllm::Data referenceFloat;
                    fastllm::ToDataType(referenceInput, referenceFloat,
                                        fastllm::DataType::FLOAT32);
                    fastllm::Linear(referenceFloat, floatWeight, emptyBias, expected);
                }
                fastllm::Data cudaInput(inputType, {batch, inputDim}, values);
                cudaInput.ToDevice(fastllm::DataDevice::CUDA,
                                   std::vector<int>{0}, true);
                fastllm::Data actual;
                {
                    ScopedFirstDevice guard("cuda");
                    fastllm::Linear(cudaInput, compactWeight, emptyBias, actual);
                }
                const bool directFloatPath =
                    inputType == fastllm::DataType::FLOAT32 && batch <= 9;
                ExpectFloatNear(
                    ToFloatVector(expected), ToFloatVector(actual),
                    directFloatPath ? 2e-4f : 5e-2f,
                    directFloatPath ? 2e-4f : 2e-2f,
                    "CUDA compact INT4_GROUP32 linear " +
                        fastllm::GetDataTypeName(inputType) + " batch " +
                        std::to_string(batch));
            }
        }
    }

    void RunCudaInt4GroupHalfWeightRoundingRegression() {
        const int inputDim = 160;
        const int outputDim = 1;
        const int groupCnt = 32;
        const int group = inputDim / groupCnt;
        const int zero = 4;
        const int q = 7;
        const float sourceScale = 0.0005f;

        fastllm::Data weight(fastllm::DataType::INT4_GROUP, {outputDim, inputDim});
        // Routed-expert weights intentionally retain the source AWQ layout and
        // exercise the small-batch GEMV rather than the Marlin repack path.
        weight.name = "regression.model.layers.0.mlp.experts.0.gate_up_proj.weight";
        weight.group = group;
        weight.groupCnt = groupCnt;
        weight.perChannelAxis = 0;
        weight.scales.assign(group, sourceScale);
        weight.mins.assign(group, -sourceScale * zero);
        weight.zeros.assign(group, zero);
        weight.Allocate(true);
        for (int x = 0; x < inputDim; x += 2) {
            weight.cpuData[x / 2] = (uint8_t)((q << 4) | q);
        }

        std::vector<float> inputValues(inputDim, 1.0f);
        fastllm::Data input(fastllm::DataType::FLOAT16, {1, inputDim}, inputValues);
        input.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        weight.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data output;
        fastllm::Data emptyBias;
        {
            ScopedFirstDevice guard("cuda");
            fastllm::Linear(input, weight, emptyBias, output);
        }

        float halfScale = fastllm::half_to_float(fastllm::float_to_half(sourceScale));
        float halfWeight = fastllm::half_to_float(
            fastllm::float_to_half(halfScale * (q - zero)));
        float expected = fastllm::half_to_float(
            fastllm::float_to_half(halfWeight * inputDim));
        float unrounded = fastllm::half_to_float(
            fastllm::float_to_half(halfScale * (q - zero) * inputDim));
        Expect(expected != unrounded,
               "INT4_GROUP rounding regression fixture does not distinguish A16 semantics.");
        std::vector<float> actual = ToFloatVector(output);
        Expect(actual.size() == 1 && actual[0] == expected,
               "CUDA INT4_GROUP GEMV did not round dequantized weights to FP16 before accumulation: expected " +
               std::to_string(expected) + ", got " +
               (actual.empty() ? std::string("<empty>") : std::to_string(actual[0])));
    }

    void RunCudaInt4GroupBatch1MoeRegression() {
        const int expertCount = 4;
        const int hidden = 128;
        const int inter = 32;
        const int topk = 2;
        const int groupCnt = 32;

        std::vector<float> inputValues(hidden);
        for (int x = 0; x < hidden; x++) {
            inputValues[x] = (float)((x * 13 + 7) % 67 - 33) / 48.0f;
        }

        std::vector<std::unique_ptr<fastllm::Data>> ownedWeights;
        std::vector<fastllm::Data*> weights((size_t)expertCount * 2 + 2, nullptr);
        std::vector<std::vector<float>> gateupFloat(expertCount);
        std::vector<std::vector<float>> downFloat(expertCount);

        auto makeWeight = [&](int rows, int cols, int expert, int salt,
                              std::vector<float> &dequantized) {
            auto weight = std::make_unique<fastllm::Data>(
                fastllm::DataType::INT4_GROUP, std::vector<int>{rows, cols});
            weight->name = "regression.cuda_int4_group32.moe." + std::to_string(expert) +
                           "." + std::to_string(salt);
            weight->groupCnt = groupCnt;
            weight->group = cols / groupCnt;
            weight->perChannelAxis = 0;
            weight->scales.resize((size_t)rows * weight->group);
            weight->mins.resize((size_t)rows * weight->group);
            weight->zeros.resize((size_t)rows * weight->group);
            weight->Allocate(true);
            dequantized.resize((size_t)rows * cols);
            for (int row = 0; row < rows; row++) {
                for (int g = 0; g < weight->group; g++) {
                    int zero = (expert * 5 + row * 3 + g * 7 + salt) & 15;
                    float scale = 0.002f * (1 + ((expert + row + g + salt) % 5));
                    size_t meta = (size_t)row * weight->group + g;
                    weight->scales[meta] = scale;
                    weight->mins[meta] = -scale * zero;
                    weight->zeros[meta] = zero;
                    for (int col = g * groupCnt; col < (g + 1) * groupCnt; col++) {
                        int q = (expert * 11 + row * 7 + col * 3 + salt) & 15;
                        size_t byte = ((size_t)row * cols + col) / 2;
                        if ((col & 1) == 0) {
                            weight->cpuData[byte] |= (uint8_t)(q << 4);
                        } else {
                            weight->cpuData[byte] |= (uint8_t)q;
                        }
                        dequantized[(size_t)row * cols + col] = (q - zero) * scale;
                    }
                }
            }
            weight->ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
            return weight;
        };

        for (int expert = 0; expert < expertCount; expert++) {
            auto gateup = makeWeight(inter * 2, hidden, expert, 1, gateupFloat[expert]);
            auto down = makeWeight(hidden, inter, expert, 9, downFloat[expert]);
            weights[(expert + 1) * 2] = gateup.get();
            weights[(expert + 1) * 2 + 1] = down.get();
            ownedWeights.push_back(std::move(gateup));
            ownedWeights.push_back(std::move(down));
        }

        const std::vector<int32_t> routeIndices = {1, 3};
        const std::vector<float> routeScores = {0.625f, 0.375f};
        std::vector<float> expected(hidden, 0.0f);
        for (int route = 0; route < topk; route++) {
            int expert = routeIndices[route];
            std::vector<float> middle(inter);
            for (int j = 0; j < inter; j++) {
                float gate = 0.0f, up = 0.0f;
                for (int x = 0; x < hidden; x++) {
                    gate += inputValues[x] * gateupFloat[expert][(size_t)j * hidden + x];
                    up += inputValues[x] * gateupFloat[expert][(size_t)(inter + j) * hidden + x];
                }
                middle[j] = (gate / (1.0f + std::exp(-gate))) * up;
            }
            for (int out = 0; out < hidden; out++) {
                float value = 0.0f;
                for (int j = 0; j < inter; j++) {
                    value += middle[j] * downFloat[expert][(size_t)out * inter + j];
                }
                expected[out] += routeScores[route] * value;
            }
        }

        fastllm::Data input(fastllm::DataType::FLOAT16, {1, hidden}, inputValues);
        input.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data index = MakeIntTensor({1, topk}, routeIndices);
        index.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data score(fastllm::DataType::FLOAT32, {1, topk}, routeScores);
        score.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data scratch;
        scratch.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
        fastllm::Data output(fastllm::DataType::FLOAT16);
        output.Resize({1, hidden});
        output.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
        output.Allocate(false);

        bool ok = FastllmCudaHalfMergeMOEInt4GroupBatch1Indexed(
            input, scratch, output, weights.data(), (int)weights.size(),
            (const int32_t*)index.cudaData, (const float*)score.cudaData, topk);
        Expect(ok, "CUDA INT4_GROUP batch-1 fused MoE path was not selected.");
        ExpectFloatNear(expected, ToFloatVector(output), 3e-2f, 3e-2f,
                        "CUDA INT4_GROUP batch-1 fused MoE");

        fastllm::Data graphOutput(fastllm::DataType::FLOAT16);
        graphOutput.Resize({1, hidden});
        graphOutput.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
        graphOutput.Allocate(false);
        FastllmCudaMemset0(output.cudaData, output.GetBytes());
        FastllmCudaMemset0(graphOutput.cudaData, graphOutput.GetBytes());

        void *graph = nullptr;
        void *graphExec = nullptr;
        Expect(FastllmCudaGraphBeginCapture(),
               "CUDA INT4_GROUP batch-1 fused MoE graph capture did not start.");
        ok = FastllmCudaHalfMergeMOEInt4GroupBatch1Indexed(
            input, scratch, output, weights.data(), (int)weights.size(),
            (const int32_t*)index.cudaData, (const float*)score.cudaData, topk);
        Expect(ok, "CUDA INT4_GROUP batch-1 fused MoE path failed during graph capture.");
        // Also verify that a downstream operation captured after the fused MoE
        // observes its completed result during graph replay.
        FastllmCudaCopyFromDeviceToDevice(graphOutput.cudaData, output.cudaData,
                                          output.GetBytes());
        Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
               "CUDA INT4_GROUP batch-1 fused MoE graph capture failed.");
        Expect(FastllmCudaGraphInstantiate(graph, &graphExec) && graphExec != nullptr,
               "CUDA INT4_GROUP batch-1 fused MoE graph instantiate failed.");
        Expect(FastllmCudaGraphLaunch(graphExec),
               "CUDA INT4_GROUP batch-1 fused MoE graph replay failed.");
        ExpectFloatNear(expected, ToFloatVector(graphOutput), 3e-2f, 3e-2f,
                        "CUDA INT4_GROUP batch-1 fused MoE graph downstream consumer");
        FastllmCudaGraphExecDestroy(graphExec);
        FastllmCudaGraphDestroy(graph);

        const int smallBatch = 4;
        std::vector<float> batchInputValues((size_t)smallBatch * hidden);
        for (int b = 0; b < smallBatch; b++) {
            for (int x = 0; x < hidden; x++) {
                batchInputValues[(size_t)b * hidden + x] =
                    inputValues[x] * (0.75f + 0.125f * b) +
                    (float)(b - 1) / 64.0f;
            }
        }
        const std::vector<int32_t> batchRouteIndices = {
            1, 3,
            0, 2,
            3, 1,
            2, 0
        };
        const std::vector<float> batchRouteScores = {
            0.625f, 0.375f,
            0.55f, 0.45f,
            0.2f, 0.8f,
            0.7f, 0.3f
        };
        std::vector<float> batchExpected((size_t)smallBatch * hidden, 0.0f);
        for (int b = 0; b < smallBatch; b++) {
            const float *rowInput = batchInputValues.data() + (size_t)b * hidden;
            for (int route = 0; route < topk; route++) {
                int routed = b * topk + route;
                int expert = batchRouteIndices[routed];
                std::vector<float> middle(inter);
                for (int j = 0; j < inter; j++) {
                    float gate = 0.0f, up = 0.0f;
                    for (int x = 0; x < hidden; x++) {
                        gate += rowInput[x] *
                                gateupFloat[expert][(size_t)j * hidden + x];
                        up += rowInput[x] *
                              gateupFloat[expert][(size_t)(inter + j) * hidden + x];
                    }
                    middle[j] = (gate / (1.0f + std::exp(-gate))) * up;
                }
                for (int out = 0; out < hidden; out++) {
                    float value = 0.0f;
                    for (int j = 0; j < inter; j++) {
                        value += middle[j] *
                                 downFloat[expert][(size_t)out * inter + j];
                    }
                    batchExpected[(size_t)b * hidden + out] +=
                        batchRouteScores[routed] * value;
                }
            }
        }

        fastllm::Data batchInput(fastllm::DataType::FLOAT16,
                                 {smallBatch, hidden}, batchInputValues);
        batchInput.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data batchIndex = MakeIntTensor(
            {smallBatch, topk}, batchRouteIndices);
        batchIndex.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data batchScore(fastllm::DataType::FLOAT32,
                                 {smallBatch, topk}, batchRouteScores);
        batchScore.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data batchScratch;
        batchScratch.ToDevice(fastllm::DataDevice::CUDA,
                              std::vector<int>{0}, false);
        fastllm::Data batchOutput(fastllm::DataType::FLOAT16);
        batchOutput.Resize({smallBatch, hidden});
        batchOutput.ToDevice(fastllm::DataDevice::CUDA,
                             std::vector<int>{0}, false);
        batchOutput.Allocate(false);
        ok = FastllmCudaHalfMergeMOEInt4GroupSmallBatchIndexed(
            batchInput, batchScratch, batchOutput,
            weights.data(), (int)weights.size(),
            (const int32_t*)batchIndex.cudaData,
            (const float*)batchScore.cudaData, smallBatch, topk);
        Expect(ok, "CUDA INT4_GROUP small-batch fused MoE path was not selected.");
        ExpectFloatNear(batchExpected, ToFloatVector(batchOutput),
                        3e-2f, 3e-2f,
                        "CUDA INT4_GROUP small-batch fused MoE");

        std::vector<float> rowWiseOutput;
        rowWiseOutput.reserve((size_t)smallBatch * hidden);
        for (int b = 0; b < smallBatch; b++) {
            std::vector<float> rowInputValues(
                batchInputValues.begin() + (size_t)b * hidden,
                batchInputValues.begin() + (size_t)(b + 1) * hidden);
            std::vector<int32_t> rowIndices(
                batchRouteIndices.begin() + b * topk,
                batchRouteIndices.begin() + (b + 1) * topk);
            std::vector<float> rowScores(
                batchRouteScores.begin() + b * topk,
                batchRouteScores.begin() + (b + 1) * topk);
            fastllm::Data rowInput(fastllm::DataType::FLOAT16,
                                   {1, hidden}, rowInputValues);
            rowInput.ToDevice(fastllm::DataDevice::CUDA,
                              std::vector<int>{0}, true);
            fastllm::Data rowIndex = MakeIntTensor({1, topk}, rowIndices);
            rowIndex.ToDevice(fastllm::DataDevice::CUDA,
                              std::vector<int>{0}, true);
            fastllm::Data rowScore(fastllm::DataType::FLOAT32,
                                   {1, topk}, rowScores);
            rowScore.ToDevice(fastllm::DataDevice::CUDA,
                              std::vector<int>{0}, true);
            fastllm::Data rowScratch;
            rowScratch.ToDevice(fastllm::DataDevice::CUDA,
                                std::vector<int>{0}, false);
            fastllm::Data rowOutput(fastllm::DataType::FLOAT16);
            rowOutput.Resize({1, hidden});
            rowOutput.ToDevice(fastllm::DataDevice::CUDA,
                               std::vector<int>{0}, false);
            rowOutput.Allocate(false);
            ok = FastllmCudaHalfMergeMOEInt4GroupBatch1Indexed(
                rowInput, rowScratch, rowOutput,
                weights.data(), (int)weights.size(),
                (const int32_t*)rowIndex.cudaData,
                (const float*)rowScore.cudaData, topk);
            Expect(ok, "CUDA INT4_GROUP row-wise fused MoE path was not selected.");
            std::vector<float> actualRow = ToFloatVector(rowOutput);
            rowWiseOutput.insert(rowWiseOutput.end(),
                                 actualRow.begin(), actualRow.end());
        }
        ExpectFloatNear(rowWiseOutput, ToFloatVector(batchOutput),
                        1e-3f, 1e-3f,
                        "CUDA INT4_GROUP small-batch versus row-wise fused MoE");

        fastllm::Data fallbackIndex = MakeIntTensor(
            {smallBatch, topk}, batchRouteIndices);
        fallbackIndex.ToDevice(fastllm::DataDevice::CUDA,
                               std::vector<int>{0}, true);
        fastllm::Data fallbackScore(fastllm::DataType::FLOAT32,
                                    {smallBatch, topk}, batchRouteScores);
        fallbackScore.ToDevice(fastllm::DataDevice::CUDA,
                               std::vector<int>{0}, true);
        fastllm::Data fallbackW1, fallbackW2, fallbackW3;
        fastllm::Data fallbackInput, fallbackIntermediate;
        fastllm::Data fallbackOutput(fastllm::DataType::FLOAT16);
        fallbackOutput.Resize({smallBatch, hidden});
        fallbackOutput.ToDevice(fastllm::DataDevice::CUDA,
                                std::vector<int>{0}, false);
        std::vector<fastllm::Data*> biases(weights.size(), nullptr);

        const char *oldSmallBatchEnv =
            std::getenv("FASTLLM_CUDA_MOE_INT4_GROUP_SMALL_BATCH");
        const bool hadSmallBatchEnv = oldSmallBatchEnv != nullptr;
        const std::string oldSmallBatchEnvValue =
            hadSmallBatchEnv ? oldSmallBatchEnv : "";
        setenv("FASTLLM_CUDA_MOE_INT4_GROUP_SMALL_BATCH", "0", 1);
        {
            ScopedFirstDevice guard("cuda");
            fastllm::MergeMOE(
                batchInput, fallbackIndex, fallbackScore, weights, biases,
                fallbackW1, fallbackW2, fallbackW3,
                fallbackInput, fallbackIntermediate,
                0.0f, fallbackOutput);
        }
        if (hadSmallBatchEnv) {
            setenv("FASTLLM_CUDA_MOE_INT4_GROUP_SMALL_BATCH",
                   oldSmallBatchEnvValue.c_str(), 1);
        } else {
            unsetenv("FASTLLM_CUDA_MOE_INT4_GROUP_SMALL_BATCH");
        }
        ExpectFloatNear(ToFloatVector(fallbackOutput), ToFloatVector(batchOutput),
                        0.0f, 0.0f,
                        "CUDA INT4_GROUP small-batch fused versus generic MergeMOE");
    }

    void RunCudaInt8Batch1MoeRegression() {
        const int expertCount = 4;
        const int hidden = 128;
        const int inter = 32;
        const int topk = 2;
        std::vector<float> inputValues(hidden);
        for (int x = 0; x < hidden; x++) {
            inputValues[x] = (float)((x * 17 + 5) % 73 - 36) / 56.0f;
        }

        std::vector<std::unique_ptr<fastllm::Data>> ownedWeights;
        std::vector<fastllm::Data*> weights((size_t)expertCount * 2 + 2, nullptr);
        std::vector<std::vector<float>> gateupFloat(expertCount), downFloat(expertCount);
        auto makeWeight = [&](int rows, int cols, int expert, int salt,
                              std::vector<float> &dequantized) {
            auto weight = std::make_unique<fastllm::Data>(
                fastllm::DataType::INT8, std::vector<int>{rows, cols});
            weight->name = "regression.cuda_int8.moe." + std::to_string(expert) +
                           "." + std::to_string(salt);
            weight->perChannelAxis = 0;
            weight->scales.resize(rows);
            weight->zeros.resize(rows);
            weight->Allocate(true);
            dequantized.resize((size_t)rows * cols);
            for (int row = 0; row < rows; row++) {
                int zero = 96 + ((expert * 11 + row * 3 + salt) % 65);
                float scale = 0.0008f * (1 + ((expert + row + salt) % 7));
                weight->scales[row] = scale;
                weight->zeros[row] = zero;
                for (int col = 0; col < cols; col++) {
                    int delta = ((expert * 13 + row * 7 + col * 5 + salt) % 61) - 30;
                    int q = std::max(0, std::min(255, zero + delta));
                    weight->cpuData[(size_t)row * cols + col] = (uint8_t)q;
                    dequantized[(size_t)row * cols + col] = (q - zero) * scale;
                }
            }
            weight->ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
            return weight;
        };

        for (int expert = 0; expert < expertCount; expert++) {
            auto gateup = makeWeight(inter * 2, hidden, expert, 3, gateupFloat[expert]);
            auto down = makeWeight(hidden, inter, expert, 7, downFloat[expert]);
            weights[(expert + 1) * 2] = gateup.get();
            weights[(expert + 1) * 2 + 1] = down.get();
            ownedWeights.push_back(std::move(gateup));
            ownedWeights.push_back(std::move(down));
        }

        const std::vector<int32_t> routeIndices = {0, 3};
        const std::vector<float> routeScores = {0.45f, 0.55f};
        std::vector<float> expected(hidden, 0.0f);
        for (int route = 0; route < topk; route++) {
            int expert = routeIndices[route];
            std::vector<float> middle(inter);
            for (int j = 0; j < inter; j++) {
                float gate = 0.0f, up = 0.0f;
                for (int x = 0; x < hidden; x++) {
                    gate += inputValues[x] * gateupFloat[expert][(size_t)j * hidden + x];
                    up += inputValues[x] * gateupFloat[expert][(size_t)(inter + j) * hidden + x];
                }
                middle[j] = (gate / (1.0f + std::exp(-gate))) * up;
            }
            for (int out = 0; out < hidden; out++) {
                float value = 0.0f;
                for (int j = 0; j < inter; j++) {
                    value += middle[j] * downFloat[expert][(size_t)out * inter + j];
                }
                expected[out] += routeScores[route] * value;
            }
        }

        fastllm::Data input(fastllm::DataType::FLOAT16, {1, hidden}, inputValues);
        input.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data index = MakeIntTensor({1, topk}, routeIndices);
        index.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data score(fastllm::DataType::FLOAT32, {1, topk}, routeScores);
        score.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data scratch;
        scratch.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
        fastllm::Data output(fastllm::DataType::FLOAT16);
        output.Resize({1, hidden});
        output.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
        output.Allocate(false);
        bool ok = FastllmCudaHalfMergeMOEInt8Batch1Indexed(
            input, scratch, output, weights.data(), (int)weights.size(),
            (const int32_t*)index.cudaData, (const float*)score.cudaData, topk);
        Expect(ok, "CUDA INT8 batch-1 fused MoE path was not selected.");
        ExpectFloatNear(expected, ToFloatVector(output), 2e-2f, 2e-2f,
                        "CUDA INT8 batch-1 fused MoE");

        fastllm::Data graphOutput(fastllm::DataType::FLOAT16);
        graphOutput.Resize({1, hidden});
        graphOutput.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
        graphOutput.Allocate(false);
        FastllmCudaMemset0(output.cudaData, output.GetBytes());
        FastllmCudaMemset0(graphOutput.cudaData, graphOutput.GetBytes());

        void *graph = nullptr;
        void *graphExec = nullptr;
        Expect(FastllmCudaGraphBeginCapture(),
               "CUDA INT8 batch-1 fused MoE graph capture did not start.");
        ok = FastllmCudaHalfMergeMOEInt8Batch1Indexed(
            input, scratch, output, weights.data(), (int)weights.size(),
            (const int32_t*)index.cudaData, (const float*)score.cudaData, topk);
        Expect(ok, "CUDA INT8 batch-1 fused MoE path failed during graph capture.");
        FastllmCudaCopyFromDeviceToDevice(graphOutput.cudaData, output.cudaData,
                                          output.GetBytes());
        Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
               "CUDA INT8 batch-1 fused MoE graph capture failed.");
        Expect(FastllmCudaGraphInstantiate(graph, &graphExec) && graphExec != nullptr,
               "CUDA INT8 batch-1 fused MoE graph instantiate failed.");
        Expect(FastllmCudaGraphLaunch(graphExec),
               "CUDA INT8 batch-1 fused MoE graph replay failed.");
        ExpectFloatNear(expected, ToFloatVector(graphOutput), 2e-2f, 2e-2f,
                        "CUDA INT8 batch-1 fused MoE graph downstream consumer");
        FastllmCudaGraphExecDestroy(graphExec);
        FastllmCudaGraphDestroy(graph);
    }
#endif

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

    fastllm::Data MakeCompactInt4Group32Weight(int outputDim, int inputDim,
                                               float seed) {
        constexpr int groupCnt = 32;
        Expect(inputDim % groupCnt == 0,
               "compact INT4_GROUP32 test weight has an invalid input dimension.");
        fastllm::Data weight(fastllm::DataType::INT4_GROUP32,
                             {outputDim, inputDim});
        weight.Allocate(true);
        const size_t rowBytes = fastllm::GetDataBytes(
            fastllm::DataType::INT4_GROUP32, 1, inputDim);
        const int groups = inputDim / groupCnt;
        const int seedValue = (int)std::lround(seed * 100.0f);
        for (int row = 0; row < outputDim; row++) {
            uint8_t *rowData = weight.cpuData + (size_t)row * rowBytes;
            for (int group = 0; group < groups; group++) {
                uint8_t *block = rowData +
                    fastllm::GetInt4Group32DataOffset(group, groups);
                const float sourceScale =
                    0.005f * (float)(1 + ((row + group + seedValue) % 11));
                const uint16_t scale =
                    fastllm::Float32ToBFloat16RNEBits(sourceScale);
                memcpy(rowData +
                           fastllm::GetInt4Group32ScaleOffset(group, groups),
                       &scale, sizeof(scale));
                for (int column = 0; column < groupCnt; column += 2) {
                    const int high =
                        (row * 7 + group * 5 + column * 3 + seedValue) & 15;
                    const int low =
                        (row * 7 + group * 5 + (column + 1) * 3 + seedValue) & 15;
                    block[column / 2] = (uint8_t)((high << 4) | low);
                }
            }
        }
        return weight;
    }

    fastllm::Data MakeInt4GroupWeight(int outputDim, int inputDim, float seed,
                                      int groupCnt = 128) {
        fastllm::Data source = MakeFloatTensor({outputDim, inputDim}, seed);
        fastllm::Data weight(fastllm::DataType::INT4_GROUP, {outputDim, inputDim});
        weight.CreateFromOriData(fastllm::WeightType::LINEAR, fastllm::DataType::FLOAT32,
                                 source.cpuData, nullptr, nullptr, groupCnt);
        return weight;
    }

    MoeWeights MakeCompactInt4Group32MoeWeights(
            int inputDim, int interDim, int outputDim, float seed) {
        MoeWeights weights {
            MakeCompactInt4Group32Weight(interDim * 2, inputDim, seed),
            MakeCompactInt4Group32Weight(outputDim, interDim, seed + 1.0f)
        };
        weights.routedGate.name = "test.int4g32_routed_gate";
        weights.routedDown.name = "test.int4g32_routed_down";
        return weights;
    }

    MoeWeights MakeInt4GroupMoeWeights(int inputDim, int interDim, int outputDim,
                                       float seed, int groupCnt = 128) {
        MoeWeights weights {
            MakeInt4GroupWeight(interDim * 2, inputDim, seed, groupCnt),
            MakeInt4GroupWeight(outputDim, interDim, seed + 1.0f, groupCnt)
        };
        weights.routedGate.name = "test.int4g_routed_gate";
        weights.routedDown.name = "test.int4g_routed_down";
        return weights;
    }

#ifdef USE_NUMAS
    fastllm::DataType KimiNumaTestActType(const fastllm::Data &weight) {
        if (weight.dataType == fastllm::DataType::FLOAT32) {
            return fastllm::DataType::FLOAT32;
        }
        if (weight.dataType == fastllm::DataType::BFLOAT16) {
            return fastllm::DataType::BFLOAT16;
        }
        if (weight.dataType == fastllm::DataType::INT4_GROUP32) {
            return fastllm::DataType::INF_INT8_GROUP32;
        }
        throw std::runtime_error(
            "unsupported Kimi NUMA regression weight type");
    }

    std::vector<float> RunKimiRoutedExpertsReference(
            const fastllm::Data &input,
            const std::vector<int32_t> &indices,
            const std::vector<float> &scores,
            int topk, std::vector<fastllm::Data> &w1s,
            std::vector<fastllm::Data> &w2s,
            std::vector<fastllm::Data> &w3s,
            float beta, float linearBeta) {
        const int tokens = input.dims[0];
        const int inputDim = input.dims[1];
        const int interDim = w1s[0].dims[0];
        const int outputDim = w2s[0].dims[0];
        const uint16_t *inputData = (const uint16_t*)input.cpuData;
        std::vector<uint16_t> routed(
            (size_t)tokens * topk * outputDim);

        for (int token = 0; token < tokens; token++) {
            std::vector<float> inputFloat(inputDim);
            for (int channel = 0; channel < inputDim; channel++) {
                inputFloat[channel] = fastllm::BFloat16BitsToFloat32(
                    inputData[(size_t)token * inputDim + channel]);
            }
            for (int slot = 0; slot < topk; slot++) {
                int expert = indices[(size_t)token * topk + slot];
                fastllm::Data &w1 = w1s[expert];
                fastllm::Data &w2 = w2s[expert];
                fastllm::Data &w3 = w3s[expert];
                fastllm::DataType gateType = KimiNumaTestActType(w1);
                fastllm::DataType downType = KimiNumaTestActType(w2);
                std::vector<uint8_t> gateInput(
                    fastllm::GetDataBytes(gateType, 1, inputDim));
                fastllm::ConvertFromFloat32(
                    gateInput.data(), gateType, inputFloat.data(), 1,
                    inputDim);
                std::vector<float> gate(interDim), up(interDim);
                fastllm::FastllmGemm(
                    1, inputDim, interDim,
                    gateInput.data(),
                    fastllm::GetDataBytes(gateType, 1, inputDim),
                    w1.cpuData,
                    fastllm::GetDataBytes(w1.GetDataType(), 1, inputDim),
                    gate.data(), sizeof(float) * interDim,
                    0, interDim, gateType, w1.GetDataType(),
                    fastllm::DataType::FLOAT32);
                fastllm::FastllmGemm(
                    1, inputDim, interDim,
                    gateInput.data(),
                    fastllm::GetDataBytes(gateType, 1, inputDim),
                    w3.cpuData,
                    fastllm::GetDataBytes(w3.GetDataType(), 1, inputDim),
                    up.data(), sizeof(float) * interDim,
                    0, interDim, gateType, w3.GetDataType(),
                    fastllm::DataType::FLOAT32);

                std::vector<float> activated(interDim);
                for (int channel = 0; channel < interDim; channel++) {
                    float gateValue =
                        fastllm::RoundFloat32ToBFloat16RNE(gate[channel]);
                    float upValue =
                        fastllm::RoundFloat32ToBFloat16RNE(up[channel]);
                    float sigmoid =
                        1.0f / (1.0f + std::exp(-gateValue));
                    float situ = beta * std::tanh(gateValue / beta) *
                                 sigmoid;
                    float boundedUp = linearBeta > 0.0f ?
                        linearBeta * std::tanh(upValue / linearBeta) :
                        upValue;
                    uint16_t bits = fastllm::Float32ToBFloat16RNEBits(
                        situ * boundedUp);
                    activated[channel] =
                        fastllm::BFloat16BitsToFloat32(bits);
                }
                std::vector<uint8_t> downInput(
                    fastllm::GetDataBytes(downType, 1, interDim));
                fastllm::ConvertFromFloat32(
                    downInput.data(), downType, activated.data(), 1,
                    interDim);
                std::vector<float> down(outputDim);
                fastllm::FastllmGemm(
                    1, interDim, outputDim,
                    downInput.data(),
                    fastllm::GetDataBytes(downType, 1, interDim),
                    w2.cpuData,
                    fastllm::GetDataBytes(w2.GetDataType(), 1, interDim),
                    down.data(), sizeof(float) * outputDim,
                    0, outputDim, downType, w2.GetDataType(),
                    fastllm::DataType::FLOAT32);
                uint16_t *destination = routed.data() +
                    ((size_t)token * topk + slot) * outputDim;
                for (int channel = 0; channel < outputDim; channel++) {
                    destination[channel] =
                        fastllm::Float32ToBFloat16RNEBits(down[channel]);
                }
            }
        }

        std::vector<float> output((size_t)tokens * outputDim);
        for (int token = 0; token < tokens; token++) {
            for (int channel = 0; channel < outputDim; channel++) {
                float sum = 0.0f;
                for (int slot = 0; slot < topk; slot++) {
                    sum += fastllm::BFloat16BitsToFloat32(
                               routed[((size_t)token * topk + slot) *
                                      outputDim + channel]) *
                           scores[(size_t)token * topk + slot];
                }
                output[(size_t)token * outputDim + channel] =
                    fastllm::RoundFloat32ToBFloat16RNE(sum);
            }
        }
        return output;
    }

    fastllm::Data MakeKimiNumaTestWeight(
            fastllm::DataType dataType, int outputDim, int inputDim,
            float seed) {
        if (dataType == fastllm::DataType::INT4_GROUP32) {
            return MakeCompactInt4Group32Weight(
                outputDim, inputDim, seed);
        }
        return MakeTensor(dataType, {outputDim, inputDim}, seed);
    }

    void RunNumasKimiRoutedExpertsFormatCase(
            fastllm::DataType dataType, const std::string &formatName) {
        constexpr int tokens = 2;
        constexpr int topk = 2;
        constexpr int expertCount = 3;
        constexpr int inputDim = 64;
        constexpr int interDim = 64;
        constexpr int outputDim = 64;
        constexpr float beta = 1.7f;
        constexpr float linearBeta = 2.5f;
        fastllm::Data input = MakeTensor(
            fastllm::DataType::BFLOAT16, {tokens, inputDim}, 0.4f);
        std::vector<int32_t> indices = {0, 1, 1, 2};
        std::vector<float> scores = {0.65f, 0.35f, 0.4f, 0.6f};
        fastllm::Data index = MakeIntTensor({tokens, topk}, indices);
        fastllm::Data score(
            fastllm::DataType::FLOAT32, {tokens, topk}, scores);

        std::vector<fastllm::Data> w1s;
        std::vector<fastllm::Data> w2s;
        std::vector<fastllm::Data> w3s;
        w1s.reserve(expertCount);
        w2s.reserve(expertCount);
        w3s.reserve(expertCount);
        for (int expert = 0; expert < expertCount; expert++) {
            w1s.push_back(MakeKimiNumaTestWeight(
                dataType, interDim, inputDim, 1.0f + expert));
            w2s.push_back(MakeKimiNumaTestWeight(
                dataType, outputDim, interDim, 4.0f + expert));
            w3s.push_back(MakeKimiNumaTestWeight(
                dataType, interDim, inputDim, 7.0f + expert));
        }
        std::vector<float> expected = RunKimiRoutedExpertsReference(
            input, indices, scores, topk, w1s, w2s, w3s,
            beta, linearBeta);

        std::vector<fastllm::Data*> w1Ptrs, w2Ptrs, w3Ptrs, allWeights;
        for (int expert = 0; expert < expertCount; expert++) {
            w1Ptrs.push_back(&w1s[expert]);
            w2Ptrs.push_back(&w2s[expert]);
            w3Ptrs.push_back(&w3s[expert]);
            allWeights.push_back(&w1s[expert]);
            allWeights.push_back(&w2s[expert]);
            allWeights.push_back(&w3s[expert]);
        }
        fastllm::RegisterNumasLinearWeightBatch(allWeights);
        for (fastllm::Data *weight : allWeights) {
            Expect(fastllm::IsNumasLinearWeightRegistered(weight),
                   formatName + " Kimi weight was not registered.");
            Expect(weight->cpuData == nullptr,
                   formatName + " Kimi source weight was not released.");
        }

        // Run twice so the second invocation exercises the persistent
        // per-expert GEMM task plan, including refreshed input/output
        // pointers and route counts.
        for (int invocation = 0; invocation < 2; invocation++) {
            fastllm::Data output;
            {
                ScopedFirstDevice guard("numa");
                fastllm::KimiK3RoutedExperts(
                    input, index, score, w1Ptrs, w2Ptrs, w3Ptrs,
                    beta, linearBeta, output);
            }
            ExpectFloatNear(
                expected, ToFloatVector(output), 0.0f, 0.0f,
                "NUMA KimiK3RoutedExperts " + formatName +
                    " invocation " + std::to_string(invocation));
        }
    }

    void RunNumasKimiRoutedExpertsFormatRegression() {
        RunNumasKimiRoutedExpertsFormatCase(
            fastllm::DataType::FLOAT32, "FLOAT32");
        RunNumasKimiRoutedExpertsFormatCase(
            fastllm::DataType::BFLOAT16, "BFLOAT16");
        RunNumasKimiRoutedExpertsFormatCase(
            fastllm::DataType::INT4_GROUP32, "INT4_GROUP32");
    }
#endif

    std::vector<float> RunMergeMoeOnDevice(const std::string &device, MoeWeights &weights,
                                           int batch = 32, bool keepCudaInputMirror = false) {
        const int inputDim = weights.routedGate.dims[1];
        const int outputDim = weights.routedDown.dims[0];

        fastllm::Data input = MakeFloatTensor({batch, inputDim}, 0.7f);
        if (keepCudaInputMirror) {
#ifdef USE_CUDA
            input.ToDevice(fastllm::DataDevice::CUDA);
            input.ToDevice(fastllm::DataDevice::CPU);
            Expect(input.cudaData != nullptr,
                   "failed to retain a CUDA input mirror for NUMA GPU-prefill regression.");
#else
            throw std::runtime_error("CUDA input mirror requested in a non-CUDA build.");
#endif
        }
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

        Expect(output.dataType == fastllm::DataType::FLOAT32,
               "MergeMOE output dtype mismatch.");
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

    void RunNumaMoeWarmupRegistrationRegression() {
        MoeAtypeConfigTestModel model;
        model.block_cnt = 1;
        model.moeDeviceMap = {{"numa", 1}};

        const std::string gateName =
            "model.layers.0.moe.experts.0.gate_proj.weight";
        const std::string upName =
            "model.layers.0.moe.experts.0.up_proj.weight";
        const std::string expertName =
            "model.layers.0.moe.experts.0.gateup_proj.weight";
        const std::string ordinaryName =
            "model.layers.0.self_attn.o_proj.weight";
        constexpr int rows = 720720; // divisible by every practical NUMA count <= 16
        std::vector<float> expertValues(rows, 0.25f);
        std::vector<float> ordinaryValues(rows, 0.5f);
        model.weight.weight[expertName].CopyFrom(
            fastllm::Data(fastllm::DataType::FLOAT32, {rows, 1}, expertValues));
        model.weight.weight[ordinaryName].CopyFrom(
            fastllm::Data(fastllm::DataType::FLOAT32, {rows, 1}, ordinaryValues));
        model.moeLinears.insert(gateName);
        model.moeLinears.insert(upName);
        model.weightMergeRules.push_back(fastllm::WeightMergeRule({
            fastllm::WeightMergeRuleSingle(
                {gateName, upName}, expertName, "linearSwiglu")
        }));
        model.AddSpecialWeight(expertName, "linearSwiglu", 0);
        model.AddSpecialWeight(ordinaryName, "linearColumn", 0);

        model.WarmupNumaMoeWeights();

        const fastllm::Data &expert = model.weight.weight[expertName];
        const fastllm::Data &ordinary = model.weight.weight[ordinaryName];
        Expect(!expert.numasData.empty() && expert.cpuData == nullptr,
               "AutoWarmup did not register the NUMA MoE expert weight.");
        Expect(ordinary.numasData.empty() && ordinary.cpuData != nullptr,
               "AutoWarmup incorrectly registered a non-MoE special weight.");
    }

    void RunNumasInt4Group32MergeMoeRegression() {
        const int inputDim = 64;
        const int interDim = 64;
        const int outputDim = 64;

        MoeWeights cpuWeights = MakeCompactInt4Group32MoeWeights(
            inputDim, interDim, outputDim, 1.9f);
        MoeWeights numasWeights = MakeCompactInt4Group32MoeWeights(
            inputDim, interDim, outputDim, 1.9f);

        std::vector<float> expected = RunMergeMoeOnDevice("cpu", cpuWeights, 8);
        std::vector<float> actual = RunMergeMoeOnDevice("numa", numasWeights, 8);
        ExpectFloatNear(expected, actual, 1e-4f, 1e-5f,
                        "NUMA INT4_GROUP(32) MergeMOE output");
        Expect(numasWeights.routedGate.dataType == fastllm::DataType::INT4_GROUP32,
               "NUMA gate weight did not preserve INT4_GROUP32.");
        Expect(numasWeights.routedDown.dataType == fastllm::DataType::INT4_GROUP32,
               "NUMA down weight did not preserve INT4_GROUP32.");
        Expect(numasWeights.routedGate.cpuData == nullptr &&
                   numasWeights.routedDown.cpuData == nullptr,
               "NUMA INT4_GROUP(32) source buffers were not released.");

        // Batch 32 takes the expert-batched NUMA path. It validates that the
        // same fused SwiGLU -> group32 preparation works for multiple rows and
        // that the paired VNNI row templates remain numerically equivalent.
        expected = RunMergeMoeOnDevice("cpu", cpuWeights, 32);
        actual = RunMergeMoeOnDevice("numa", numasWeights, 32);
        ExpectFloatNear(expected, actual, 1e-4f, 1e-5f,
                        "NUMA batched INT4_GROUP(32) MergeMOE output");
    }

#ifdef USE_CUDA
    void RunNumasCudaPrefillFallbackRegression() {
        const int inputDim = 128;
        const int interDim = 128;
        const int outputDim = 128;

        MoeWeights numasWeights = MakeInt4GroupMoeWeights(inputDim, interDim, outputDim, 2.3f);

        // Prime NUMA registration with a small batch. INT4_GROUP(128) becomes the
        // internal INT4_GROUP128 format, which FLOAT32 CUDA Linear cannot consume.
        RunMergeMoeOnDevice("numa", numasWeights, 1);
        Expect(numasWeights.routedGate.dataType == fastllm::DataType::INT4_GROUP128,
               "NUMA gate weight was not converted to INT4_GROUP128.");
        Expect(numasWeights.routedDown.dataType == fastllm::DataType::INT4_GROUP128,
               "NUMA down weight was not converted to INT4_GROUP128.");

        // With no CUDA mirror, NUMA must use its CPU implementation. Repeating the
        // same input with a CUDA mirror should hit the compatibility check and
        // produce the same CPU fallback result.
        std::vector<float> expected = RunMergeMoeOnDevice("numa", numasWeights);
        std::vector<float> actual = RunMergeMoeOnDevice("numa", numasWeights, 32, true);
        ExpectFloatNear(expected, actual, 1e-4f, 1e-5f,
                        "NUMA unsupported CUDA-prefill fallback output");
    }
#endif
}

int main() {
    try {
        bool ranAny = false;
        bool ranCrossDeviceViewRegression = false;
#ifndef USE_ROCM
        bool ranLargeWeightOffsetRegression = false;
#endif

        RunMoeAtypeConfigRegression();
        std::cout << "moe_atype auto/explicit configuration regression: PASS\n";
        ranAny = true;

        RunLagunaNVFP4AutoDtypeRegression();
        std::cout << "Laguna NVFP4 auto dtype regression: PASS\n";

        RunLagunaPackedInt4AutoDtypeRegression();
        std::cout << "Laguna compressed-tensors INT4 auto dtype regression: PASS\n";

        RunPerRequestMinOutputLengthRegression();
        std::cout << "per-request minimum output length regression: PASS\n";

        if (fastllm::HasDeviceType("cpu")) {
            RunBFloat16Q8KConversionRegression();
            std::cout << "BF16 to Q8_K/Q8_K32 bytewise regression: PASS\n";
            RunCpuInt4GroupAwqLinearRegression();
            std::cout << "cpu AWQ-style INT4_GROUP linear regression: PASS\n";
            RunCpuPackedInt4Group32KernelRegression();
            std::cout << "cpu packed INT4_GROUP(32) kernel regression: PASS\n";
            ranAny = true;
        }

        if (fastllm::HasDeviceType("disk")) {
            RunDiskOperatorsRegression();
            std::cout << "disk Linear, Embedding and MergeMOE regression: PASS\n";
            ranAny = true;
        }

        if (fastllm::HasDeviceType("cuda")) {
#ifdef USE_CUDA
            RunCudaBFloat16SigmoidMulToRegression();
            std::cout << "cuda BF16 Sigmoid/MulTo regression: PASS\n";
            Expect(FastllmCudaGraphQwen35MoeSelfTest(),
                   "Qwen3.5 CUDA graph shared/routed MoE parallelization/fallback self-test failed");
            RunCudaTritonChunkGdnPrefillRegression();
            RunCudaDeepSeekV4TritonWoARegression();
            RunCudaDeepSeekV4TritonSparseDecodeRegression();
            RunCudaDeepSeekV4HashRouteCacheRegression();
            RunCudaDeepSeekV4FusedHcPreNormRegression();
            RunCudaDeepSeekV4FusedQKVRopeCacheRegression();
            RunCudaGraphMemoryPoolOwnershipRegression();
            RunCudaLinearDataTypeCapabilityRegression();
            RunCudaGgufMmvqBatch8Regression();
            RunCudaKimiK3PackedConvCacheRegression();
            RunCudaKimiK3RecurrentKdaRegression();
            RunCudaInt4Group32AwqLinearRegression();
            RunCudaCompactInt4Group32LinearRegression();
            RunCudaInt4GroupHalfWeightRoundingRegression();
            RunCudaInt4GroupBatch1MoeRegression();
            RunCudaInt8Batch1MoeRegression();
            RunCudaConvMultiTokenSnapshotsRegression();
            ranCrossDeviceViewRegression = RunCudaCrossDeviceViewRejectionRegression();
            RunMultiCudaReplicatedExpansionRegression();
#ifndef USE_ROCM
            RunMultiCudaInt4GroupColumnSplitRegression();
            ranLargeWeightOffsetRegression = RunMultiCudaLargeWeightOffsetRegression();
#endif
            RunCudaRecurrentSnapshotsRegression();
#endif
            RunGenerateAppendPagedCacheBatchParams("cuda:0", 1536);
            RunGeneratePagedBatchParams("cuda:0", 1536, false);
            RunGeneratePagedBatchParams("cuda:0", 64, true);
            std::cout << "cuda snapshot and paged-batch regressions: PASS";
            if (!ranCrossDeviceViewRegression) {
                std::cout << " (cross-device view SKIPPED)";
            }
#ifndef USE_ROCM
            if (!ranLargeWeightOffsetRegression) {
                std::cout << " (large-weight offset SKIPPED)";
            }
#endif
            std::cout << "\n";
            ranAny = true;
        } else {
            std::cout << "cuda snapshot and paged-batch regressions: SKIP (cuda unavailable)\n";
        }

        if (fastllm::HasDeviceType("numa") && !fastllm::GetFastllmEnv().activateNuma) {
            RunNumaMoeWarmupRegistrationRegression();
            std::cout << "numa MoE full warmup registration regression: PASS\n";
            RunNumasMergeMoeRegression();
            std::cout << "numa MergeMOE regression: PASS\n";
            RunNumasInt4Group32MergeMoeRegression();
            std::cout << "numa INT4_GROUP(32) MergeMOE regression: PASS\n";
#ifdef USE_NUMAS
            RunNumasKimiRoutedExpertsFormatRegression();
            std::cout << "numa Kimi routed-expert multi-format regression: PASS\n";
#endif
#ifdef USE_CUDA
            if (fastllm::HasDeviceType("cuda")) {
                RunNumasCudaPrefillFallbackRegression();
                std::cout << "numa CUDA-prefill fallback regression: PASS\n";
            }
#endif
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
