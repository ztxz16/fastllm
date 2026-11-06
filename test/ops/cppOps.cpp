#include "fastllm.h"
#include "executor.h"
#include "utils/utils.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace {
    using Clock = std::chrono::high_resolution_clock;

    struct ParamSpec {
        std::string name;
        std::string defaultValue;
        std::string description;
    };

    class OpTestParams {
    public:
        void Add(const std::string &name, const std::string &defaultValue, const std::string &description) {
            if (values.find(name) == values.end()) {
                values[name] = defaultValue;
            }
            specs.push_back({name, defaultValue, description});
        }

        void Override(const std::string &name, const std::string &value) {
            values[name] = value;
        }

        bool Has(const std::string &name) const {
            return values.find(name) != values.end();
        }

        std::string GetString(const std::string &name) const {
            auto it = values.find(name);
            if (it == values.end()) {
                throw std::runtime_error("missing param: " + name);
            }
            return it->second;
        }

        int GetInt(const std::string &name) const {
            return std::stoi(GetString(name));
        }

        float GetFloat(const std::string &name) const {
            return std::stof(GetString(name));
        }

        std::vector<int> GetInts(const std::string &name) const {
            std::vector<int> result;
            std::stringstream ss(GetString(name));
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) {
                    result.push_back(std::stoi(token));
                }
            }
            return result;
        }

        void Print(std::ostream &os) const {
            for (const auto &spec : specs) {
                auto it = values.find(spec.name);
                os << "  " << spec.name << "=" << (it == values.end() ? spec.defaultValue : it->second);
                if (!spec.description.empty()) {
                    os << "  # " << spec.description;
                }
                os << "\n";
            }
        }

    private:
        std::vector<ParamSpec> specs;
        std::map<std::string, std::string> values;
    };

    struct CliConfig {
        std::string op = "all";
        std::vector<std::string> deviceFilters;
        std::vector<std::pair<std::string, std::string>> paramOverrides;
        int warmup = 2;
        int iters = 10;
        float atol = 1e-4f;
        float rtol = 1e-4f;
        bool listOps = false;
        bool help = false;
    };

    struct ComparisonStats {
        bool passed = true;
        float maxAbsDiff = 0.0f;
        float meanAbsDiff = 0.0f;
        float maxRelDiff = 0.0f;
        float maxToleranceExcess = 0.0f;
        size_t failedElements = 0;
        size_t mismatchIndex = 0;
        float expected = 0.0f;
        float actual = 0.0f;
    };

    struct BenchmarkResult {
        double avgMs = 0.0;
        double bytesMoved = 0.0;
        double flops = 0.0;
        double bandwidthGBps = -1.0;
        double computeTFlops = -1.0;
    };

    static std::string Trim(const std::string &s) {
        size_t l = 0, r = s.size();
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            l++;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            r--;
        }
        return s.substr(l, r - l);
    }

    static std::vector<std::string> Split(const std::string &s, char sep) {
        std::vector<std::string> result;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, sep)) {
            item = Trim(item);
            if (!item.empty()) {
                result.push_back(item);
            }
        }
        return result;
    }

    static fastllm::Data MakeTensor(const std::vector<int> &dims, float seed = 0.0f, float scale = 1.0f) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        std::vector<float> data(count);
        for (int i = 0; i < count; i++) {
            float v = std::sin((i + 1) * 0.37f + seed) + std::cos((i + 3) * 0.19f + seed * 0.5f);
            data[i] = v * scale;
        }
        return fastllm::Data(fastllm::DataType::FLOAT32, dims, data);
    }

    static fastllm::Data MakeRampTensor(const std::vector<int> &dims, float seed = 0.0f) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        std::vector<float> data(count);
        for (int i = 0; i < count; i++) {
            data[i] = seed + (float) i / std::max(1, count);
        }
        return fastllm::Data(fastllm::DataType::FLOAT32, dims, data);
    }

    static void InitLinearWeight(const OpTestParams &params, int out, int in, fastllm::Data &weight) {
        fastllm::Data fp32Weight = MakeTensor({out, in}, 0.4f);
        std::string weightType = params.Has("weight_type") ? params.GetString("weight_type") : "float32";
        if (weightType == "float32") {
            weight.dataType = fastllm::DataType::FLOAT32;
            weight.Resize({out, in});
            weight.Allocate();
            std::memcpy(weight.cpuData, fp32Weight.cpuData, weight.GetBytes());
            return;
        }
        if (weightType == "int4group") {
            int groupCnt = params.Has("group_cnt") ? params.GetInt("group_cnt") : 128;
            weight.dataType = fastllm::DataType::INT4_GROUP;
            weight.Resize({out, in});
            weight.CreateFromOriData(fastllm::WeightType::LINEAR, fastllm::DataType::FLOAT32,
                                     (uint8_t *) fp32Weight.cpuData, nullptr, nullptr, groupCnt);
            return;
        }
        if (weightType == "int4group32") {
            constexpr int groupCnt = 32;
            if (in % groupCnt != 0) {
                throw std::runtime_error("int4group32 input features must be divisible by 32");
            }
            weight.dataType = fastllm::DataType::INT4_GROUP32;
            weight.Resize({out, in});
            weight.Allocate(true);
            const int groups = in / groupCnt;
            const size_t rowBytes = fastllm::GetDataBytes(
                fastllm::DataType::INT4_GROUP32, 1, in);
            const float *source = (const float*)fp32Weight.cpuData;
            for (int row = 0; row < out; row++) {
                uint8_t *rowData = weight.cpuData + (size_t)row * rowBytes;
                for (int group = 0; group < groups; group++) {
                    const float *sourceBlock = source +
                        (size_t)row * in + group * groupCnt;
                    float absMax = 0.0f;
                    for (int column = 0; column < groupCnt; column++) {
                        absMax = std::max(absMax, std::fabs(sourceBlock[column]));
                    }
                    uint8_t *block = rowData +
                        fastllm::GetInt4Group32DataOffset(group, groups);
                    const uint16_t scaleBits = fastllm::Float32ToBFloat16RNEBits(
                        absMax == 0.0f ? 0.0f : absMax / 7.0f);
                    std::memcpy(rowData +
                                    fastllm::GetInt4Group32ScaleOffset(group, groups),
                                &scaleBits, sizeof(scaleBits));
                    const float scale = fastllm::BFloat16BitsToFloat32(scaleBits);
                    for (int column = 0; column < groupCnt; column += 2) {
                        auto quantize = [scale](float value) {
                            if (scale == 0.0f) {
                                return 8;
                            }
                            return std::max(0, std::min(15,
                                (int)std::lround(value / scale) + 8));
                        };
                        const int high = quantize(sourceBlock[column]);
                        const int low = quantize(sourceBlock[column + 1]);
                        block[column / 2] = (uint8_t)((high << 4) | low);
                    }
                }
            }
            return;
        }
        throw std::runtime_error("unsupported linear weight_type: " + weightType);
    }

    static uint16_t FloatToBFloat16(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return uint16_t(bits >> 16);
    }

    static uint16_t FloatToBFloat16RoundToNearest(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        uint32_t roundingBias = 0x7fffU + ((bits >> 16) & 1U);
        return uint16_t((bits + roundingBias) >> 16);
    }

    static float BFloat16ToFloat(uint16_t value) {
        uint32_t bits = uint32_t(value) << 16;
        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    static float RoundFp8E4M3(float value) {
        if (!std::isfinite(value)) {
            return std::copysign(448.0f, value);
        }
        bool negative = std::signbit(value);
        float absolute = std::min(std::fabs(value), 448.0f);
        float best = 0.0f;
        float bestError = absolute;
        int bestEncoding = 0;
        for (int encoding = 1; encoding <= 0x7e; ++encoding) {
            int exponent = encoding >> 3;
            int mantissa = encoding & 7;
            float candidate;
            if (exponent == 0) {
                candidate = std::ldexp(float(mantissa), -9);
            } else {
                candidate = std::ldexp(1.0f + float(mantissa) / 8.0f,
                                       exponent - 7);
            }
            float error = std::fabs(candidate - absolute);
            if (error < bestError ||
                (error == bestError && (encoding & 1) == 0 &&
                 (bestEncoding & 1) != 0)) {
                best = candidate;
                bestError = error;
                bestEncoding = encoding;
            }
        }
        return negative ? -best : best;
    }

    static float DecodeFp8E4M3(uint8_t value) {
        bool negative = (value & 0x80U) != 0;
        int exponent = (value >> 3) & 0xf;
        int mantissa = value & 7;
        float decoded;
        if (exponent == 0) {
            decoded = std::ldexp(float(mantissa), -9);
        } else {
            decoded = std::ldexp(1.0f + float(mantissa) / 8.0f,
                                 exponent - 7);
        }
        return negative ? -decoded : decoded;
    }

    static constexpr float W4A8Fp8Max = 448.0f;
    static constexpr float W4A8Fp8MinScale =
        1.0f / (W4A8Fp8Max * 512.0f);

    struct W4A8ActivationQuantFixture {
        int tokens = 0;
        int hidden = 0;
        fastllm::DataType inputType = fastllm::DataType::BFLOAT16;
        std::vector<float> inputValues;

        explicit W4A8ActivationQuantFixture(const OpTestParams &params) {
            tokens = params.GetInt("tokens");
            hidden = params.GetInt("hidden");
            if (tokens <= 0 || hidden <= 0) {
                throw std::runtime_error(
                    "w4a8_activation_quant requires positive tokens and hidden");
            }

            std::string dtype = params.GetString("dtype");
            if (dtype == "bfloat16") {
                inputType = fastllm::DataType::BFLOAT16;
            } else if (dtype == "float16") {
                inputType = fastllm::DataType::FLOAT16;
            } else {
                throw std::runtime_error(
                    "w4a8_activation_quant dtype must be bfloat16 or float16");
            }

            std::vector<float> source((size_t)tokens * hidden);
            for (int token = 0; token < tokens; ++token) {
                for (int col = 0; col < hidden; ++col) {
                    float value = 0.0f;
                    if (token != 0) {
                        value =
                            0.75f * std::sin(float(token * hidden + col + 1) * 0.013f) +
                            0.25f * std::cos(float(col + 3) * 0.031f);
                        if (token == 1) {
                            value *= 1.0e-5f;
                        }
                    }
                    source[(size_t)token * hidden + col] = value;
                }
            }

            fastllm::Data typed(inputType, {tokens, hidden}, source);
            fastllm::Data normalized;
            fastllm::ToDataType(typed, normalized, fastllm::DataType::FLOAT32);
            normalized.ToDevice(fastllm::DataDevice::CPU);
            const float *ptr = (const float*)normalized.cpuData;
            inputValues.assign(ptr, ptr + source.size());
        }

        fastllm::Data MakeInput() const {
            return fastllm::Data(inputType, {tokens, hidden}, inputValues);
        }

        fastllm::Data CpuReference() const {
            std::vector<float> result((size_t)tokens * (hidden + 1));
            for (int token = 0; token < tokens; ++token) {
                size_t inputOffset = (size_t)token * hidden;
                size_t outputOffset = (size_t)token * (hidden + 1);
                float absoluteMax = 0.0f;
                for (int col = 0; col < hidden; ++col) {
                    absoluteMax = std::max(
                        absoluteMax, std::fabs(inputValues[inputOffset + col]));
                }
                float scale = std::max(
                    absoluteMax / W4A8Fp8Max, W4A8Fp8MinScale);
                result[outputOffset] = scale;
                for (int col = 0; col < hidden; ++col) {
                    float value = inputValues[inputOffset + col] / scale;
                    value = std::max(
                        -W4A8Fp8Max, std::min(W4A8Fp8Max, value));
                    result[outputOffset + col + 1] =
                        RoundFp8E4M3(value) * scale;
                }
            }
            return fastllm::Data(
                fastllm::DataType::FLOAT32, {tokens, hidden + 1}, result);
        }

#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
        fastllm::Data CudaResult() const {
            fastllm::Data input = MakeInput();
            input.ToDevice(fastllm::DataDevice::CUDA);
            std::vector<uint8_t> fp8Bytes;
            std::vector<float> tokenScales;
            if (!FastllmCudaInspectW4A8Activation(
                    input, tokens, hidden, fp8Bytes, tokenScales)) {
                throw std::runtime_error(
                    "FastllmCudaInspectW4A8Activation failed");
            }

            std::vector<float> result((size_t)tokens * (hidden + 1));
            for (int token = 0; token < tokens; ++token) {
                size_t inputOffset = (size_t)token * hidden;
                size_t outputOffset = (size_t)token * (hidden + 1);
                float scale = tokenScales[token];
                result[outputOffset] = scale;
                for (int col = 0; col < hidden; ++col) {
                    result[outputOffset + col + 1] =
                        DecodeFp8E4M3(fp8Bytes[inputOffset + col]) * scale;
                }
            }
            return fastllm::Data(
                fastllm::DataType::FLOAT32, {tokens, hidden + 1}, result);
        }
#endif
    };

#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
    struct W4A8ActivationQuantBenchmarkState {
        fastllm::Data input;
        int tokens = 0;
        int hidden = 0;
        void *fp8 = nullptr;
        float *scales = nullptr;

        explicit W4A8ActivationQuantBenchmarkState(
            const W4A8ActivationQuantFixture &fixture)
            : input(fixture.MakeInput()),
              tokens(fixture.tokens),
              hidden(fixture.hidden) {
            input.ToDevice(fastllm::DataDevice::CUDA);
            fp8 = FastllmCudaMalloc((size_t)tokens * hidden);
            scales = (float*)FastllmCudaMalloc((size_t)tokens * sizeof(float));
            if (fp8 == nullptr || scales == nullptr) {
                if (fp8 != nullptr) {
                    FastllmCudaFree(fp8);
                }
                if (scales != nullptr) {
                    FastllmCudaFree(scales);
                }
                throw std::runtime_error(
                    "failed to allocate W4A8 activation quant buffers");
            }
        }

        ~W4A8ActivationQuantBenchmarkState() {
            FastllmCudaFree(fp8);
            FastllmCudaFree(scales);
        }

        void Run() {
            if (!FastllmCudaW4A8QuantizeActivationPerToken(
                    input, tokens, hidden, fp8, scales)) {
                throw std::runtime_error(
                    "W4A8 activation quantization launch failed");
            }
        }
    };
#endif

    struct LinearW4A8Fixture {
        int batch = 0;
        int in = 0;
        int out = 0;
        int groups = 0;
        int seed = 0;
        std::vector<float> inputValues;
        std::vector<uint8_t> packedWeights;
        std::vector<uint16_t> bf16Scales;
        std::vector<float> sourceScales;
        std::vector<float> biasValues;

        explicit LinearW4A8Fixture(const OpTestParams &params) {
            batch = params.GetInt("batch");
            in = params.GetInt("in");
            out = params.GetInt("out");
            seed = params.Has("seed") ? params.GetInt("seed") : 0;
            if (batch <= 0 || in <= 0 || out <= 0 ||
                in % fastllm::COMPRESSED_W4A8_GROUP_SIZE != 0 ||
                out % fastllm::COMPRESSED_W4A8_GROUP_SIZE != 0) {
                throw std::runtime_error(
                    "linear weight_type=int4_w4a8 requires positive batch "
                    "and 128-aligned in/out");
            }

            groups = in / fastllm::COMPRESSED_W4A8_GROUP_SIZE;
            inputValues.resize((size_t)batch * in);
            for (size_t i = 0; i < inputValues.size(); ++i) {
                float value = 0.5f * std::sin(float(i + 1) * 0.013f) +
                              0.25f * std::cos(float(i + 3) * 0.007f);
                inputValues[i] = BFloat16ToFloat(FloatToBFloat16(value));
            }

            packedWeights.assign((size_t)out * in / 2, 0);
            sourceScales.resize((size_t)out * groups);
            bf16Scales.resize(sourceScales.size());
            for (int row = 0; row < out; ++row) {
                for (int group = 0; group < groups; ++group) {
                    float scale =
                        0.001f * float(
                            1 + ((row * 7 + group * 3 + seed * 5) % 19));
                    size_t index = (size_t)row * groups + group;
                    bf16Scales[index] = FloatToBFloat16(scale);
                    sourceScales[index] =
                        BFloat16ToFloat(bf16Scales[index]);
                }
                for (int col = 0; col < in; ++col) {
                    int signedValue =
                        ((row * 11 + col * 5 + seed * 7 + 3) % 16) - 8;
                    uint8_t nibble = uint8_t(signedValue + 8);
                    size_t index = (size_t)row * (in / 2) + col / 2;
                    if ((col & 1) == 0) {
                        packedWeights[index] =
                            (packedWeights[index] & 0xf0U) | nibble;
                    } else {
                        packedWeights[index] =
                            (packedWeights[index] & 0x0fU) |
                            uint8_t(nibble << 4);
                    }
                }
            }

            biasValues.resize(out);
            for (int i = 0; i < out; ++i) {
                biasValues[i] = float((i % 17) - 8) * 0.001f;
            }
        }

        std::vector<float> ReferenceValues(
                const std::vector<float> &values, int tokens,
                bool addBias = false) const {
            if (values.size() != (size_t)tokens * in) {
                throw std::runtime_error("invalid W4A8 reference input size");
            }
            std::vector<float> quantizedInput(values.size());
            for (int token = 0; token < tokens; ++token) {
                size_t offset = (size_t)token * in;
                float absoluteMax = 0.0f;
                for (int col = 0; col < in; ++col) {
                    absoluteMax = std::max(
                        absoluteMax, std::fabs(values[offset + col]));
                }
                float tokenScale = std::max(
                    absoluteMax / W4A8Fp8Max, W4A8Fp8MinScale);
                for (int col = 0; col < in; ++col) {
                    float value = values[offset + col] / tokenScale;
                    value = std::max(-448.0f, std::min(448.0f, value));
                    quantizedInput[offset + col] =
                        RoundFp8E4M3(value) * tokenScale;
                }
            }

            std::vector<float> decodedScales(sourceScales.size());
            for (int row = 0; row < out; ++row) {
                size_t offset = (size_t)row * groups;
                float absoluteMax = 0.0f;
                for (int group = 0; group < groups; ++group) {
                    absoluteMax = std::max(
                        absoluteMax,
                        std::fabs(sourceScales[offset + group]));
                }
                float channelScale =
                    std::max(absoluteMax, 1.0e-10f) / 56.0f;
                for (int group = 0; group < groups; ++group) {
                    decodedScales[offset + group] =
                        RoundFp8E4M3(
                            sourceScales[offset + group] / channelScale) *
                        channelScale;
                }
            }

            std::vector<float> result((size_t)tokens * out);
            for (int token = 0; token < tokens; ++token) {
                for (int row = 0; row < out; ++row) {
                    float value = 0.0f;
                    for (int col = 0; col < in; ++col) {
                        uint8_t packed =
                            packedWeights[(size_t)row * (in / 2) + col / 2];
                        int q =
                            int((packed >> ((col & 1) * 4)) & 0xfU) - 8;
                        value +=
                            quantizedInput[(size_t)token * in + col] *
                            float(q) *
                            decodedScales[(size_t)row * groups +
                                          col /
                                              fastllm::COMPRESSED_W4A8_GROUP_SIZE];
                    }
                    value = BFloat16ToFloat(
                        FloatToBFloat16RoundToNearest(value));
                    if (addBias) {
                        value = BFloat16ToFloat(
                            FloatToBFloat16RoundToNearest(
                                value + biasValues[row]));
                    }
                    result[(size_t)token * out + row] = value;
                }
            }
            return result;
        }

        fastllm::Data CpuReference() const {
            std::vector<float> quantizedInput(inputValues.size());
            for (int token = 0; token < batch; ++token) {
                size_t offset = (size_t)token * in;
                float absoluteMax = 0.0f;
                for (int col = 0; col < in; ++col) {
                    absoluteMax = std::max(
                        absoluteMax, std::fabs(inputValues[offset + col]));
                }
                float tokenScale = std::max(
                    absoluteMax / W4A8Fp8Max, W4A8Fp8MinScale);
                for (int col = 0; col < in; ++col) {
                    float value = inputValues[offset + col] / tokenScale;
                    value = std::max(-448.0f, std::min(448.0f, value));
                    quantizedInput[offset + col] =
                        RoundFp8E4M3(value) * tokenScale;
                }
            }

            std::vector<float> decodedScales(sourceScales.size());
            for (int row = 0; row < out; ++row) {
                size_t offset = (size_t)row * groups;
                float absoluteMax = 0.0f;
                for (int group = 0; group < groups; ++group) {
                    absoluteMax = std::max(
                        absoluteMax,
                        std::fabs(sourceScales[offset + group]));
                }
                float channelScale =
                    std::max(absoluteMax, 1.0e-10f) / 56.0f;
                for (int group = 0; group < groups; ++group) {
                    decodedScales[offset + group] =
                        RoundFp8E4M3(
                            sourceScales[offset + group] / channelScale) *
                        channelScale;
                }
            }

            std::vector<float> expected((size_t)batch * out);
            for (int token = 0; token < batch; ++token) {
                for (int row = 0; row < out; ++row) {
                    float value = 0.0f;
                    for (int col = 0; col < in; ++col) {
                        uint8_t packed =
                            packedWeights[(size_t)row * (in / 2) + col / 2];
                        int q =
                            int((packed >> ((col & 1) * 4)) & 0xfU) - 8;
                        value +=
                            quantizedInput[(size_t)token * in + col] *
                            float(q) *
                            decodedScales[(size_t)row * groups +
                                          col /
                                              fastllm::COMPRESSED_W4A8_GROUP_SIZE];
                    }
                    value = BFloat16ToFloat(
                        FloatToBFloat16RoundToNearest(value));
                    expected[(size_t)token * out + row] =
                        BFloat16ToFloat(FloatToBFloat16RoundToNearest(
                            value + biasValues[row]));
                }
            }
            return fastllm::Data(
                fastllm::DataType::FLOAT32, {batch, out}, expected);
        }

        void InitDeviceWeight(fastllm::Data &weight) const {
            weight.dataType = fastllm::DataType::INT4_W4A8;
            weight.UpdateUnitSize();
            weight.Resize({out, in});
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.InitW4A8Weight(
                fastllm::W4A8WeightEncoding::COMPRESSED_TENSORS_UINT4B8);
            weight.SetW4A8GroupScales(
                bf16Scales.data(), bf16Scales.size());
            weight.Allocate(false);
            std::memcpy(
                weight.cpuData, packedWeights.data(), packedWeights.size());
            weight.ToDevice(fastllm::DataDevice::CUDA);
        }

        void InitDeviceData(fastllm::Data &input,
                            fastllm::Data &weight,
                            fastllm::Data &bias,
                            fastllm::Data &output) const {
            fastllm::Data inputSource(
                fastllm::DataType::BFLOAT16, {batch, in}, inputValues);
            input.CopyFrom(inputSource);
            input.ToDevice(fastllm::DataDevice::CUDA);

            InitDeviceWeight(weight);

            fastllm::Data biasSource(
                fastllm::DataType::FLOAT32, {out}, biasValues);
            bias.CopyFrom(biasSource);
            bias.ToDevice(fastllm::DataDevice::CUDA);

            output.dataType = fastllm::DataType::BFLOAT16;
            output.UpdateUnitSize();
            output.Resize({batch, out});
            output.Allocate(false);
            output.ToDevice(fastllm::DataDevice::CUDA, false);
        }
    };

    struct MergeMoeW4A8Fixture {
        int batch = 0;
        int hidden = 0;
        int inter = 0;
        int experts = 0;
        int topk = 0;
        std::vector<float> inputValues;
        std::vector<int> routeRows;
        std::vector<float> routeScales;
        std::vector<int> routePositions;
        std::vector<int> expertStarts;
        std::vector<int> expertCounts;
        std::vector<std::shared_ptr<LinearW4A8Fixture>> gateFixtures;
        std::vector<std::shared_ptr<LinearW4A8Fixture>> downFixtures;

        explicit MergeMoeW4A8Fixture(const OpTestParams &params) {
            batch = params.GetInt("batch");
            hidden = params.GetInt("hidden");
            inter = params.GetInt("inter");
            experts = params.GetInt("experts");
            topk = params.GetInt("topk");
            if (batch <= 0 || hidden <= 0 || inter <= 0 || experts <= 0 ||
                topk <= 0 || topk > experts ||
                hidden % 256 != 0 || inter % 256 != 0) {
                throw std::runtime_error(
                    "mergemoe_w4a8 requires positive batch/experts, "
                    "topk <= experts and 256-aligned hidden/inter");
            }

            inputValues.resize((size_t)batch * hidden);
            for (size_t i = 0; i < inputValues.size(); ++i) {
                float value =
                    0.35f * std::sin(float(i + 1) * 0.017f) +
                    0.15f * std::cos(float(i + 5) * 0.011f);
                inputValues[i] =
                    BFloat16ToFloat(FloatToBFloat16RoundToNearest(value));
            }

            std::vector<int> routeExperts((size_t)batch * topk);
            std::vector<float> originalScales((size_t)batch * topk);
            expertCounts.assign(experts, 0);
            for (int token = 0; token < batch; ++token) {
                float denominator = float(topk * (topk + 1)) * 0.5f;
                for (int slot = 0; slot < topk; ++slot) {
                    int route = token * topk + slot;
                    int expert = (token * 3 + slot) % experts;
                    routeExperts[route] = expert;
                    originalScales[route] = float(topk - slot) / denominator;
                    expertCounts[expert]++;
                }
            }
            expertStarts.resize(experts);
            int totalTasks = 0;
            for (int expert = 0; expert < experts; ++expert) {
                expertStarts[expert] = totalTasks;
                totalTasks += expertCounts[expert];
            }
            std::vector<int> offsets = expertStarts;
            routeRows.resize(totalTasks);
            routeScales.resize(totalTasks);
            routePositions.resize(totalTasks);
            for (int token = 0; token < batch; ++token) {
                for (int slot = 0; slot < topk; ++slot) {
                    int route = token * topk + slot;
                    int position = offsets[routeExperts[route]]++;
                    routeRows[position] = token;
                    routeScales[position] = originalScales[route];
                    routePositions[route] = position;
                }
            }

            for (int expert = 0; expert < experts; ++expert) {
                OpTestParams gateParams;
                gateParams.Override("batch", std::to_string(batch * topk));
                gateParams.Override("in", std::to_string(hidden));
                gateParams.Override("out", std::to_string(inter * 2));
                gateParams.Override("seed", std::to_string(expert * 2 + 1));
                gateFixtures.push_back(
                    std::make_shared<LinearW4A8Fixture>(gateParams));

                OpTestParams downParams;
                downParams.Override("batch", std::to_string(batch * topk));
                downParams.Override("in", std::to_string(inter));
                downParams.Override("out", std::to_string(hidden));
                downParams.Override("seed", std::to_string(expert * 2 + 2));
                downFixtures.push_back(
                    std::make_shared<LinearW4A8Fixture>(downParams));
            }
        }

        fastllm::Data CpuReference() const {
            std::vector<float> result((size_t)batch * hidden, 0.0f);
            for (int expert = 0; expert < experts; ++expert) {
                int count = expertCounts[expert];
                if (count == 0) {
                    continue;
                }
                std::vector<float> expertInput((size_t)count * hidden);
                for (int local = 0; local < count; ++local) {
                    int route = expertStarts[expert] + local;
                    int token = routeRows[route];
                    std::copy_n(
                        inputValues.data() + (size_t)token * hidden, hidden,
                        expertInput.data() + (size_t)local * hidden);
                }
                std::vector<float> gate =
                    gateFixtures[expert]->ReferenceValues(
                        expertInput, count, false);
                std::vector<float> activated((size_t)count * inter);
                for (int row = 0; row < count; ++row) {
                    for (int col = 0; col < inter; ++col) {
                        float x = gate[(size_t)row * inter * 2 + col];
                        float y =
                            gate[(size_t)row * inter * 2 + inter + col];
                        activated[(size_t)row * inter + col] =
                            BFloat16ToFloat(FloatToBFloat16RoundToNearest(
                                (x / (1.0f + std::exp(-x))) * y));
                    }
                }
                std::vector<float> down =
                    downFixtures[expert]->ReferenceValues(
                        activated, count, false);
                for (int local = 0; local < count; ++local) {
                    int route = expertStarts[expert] + local;
                    int token = routeRows[route];
                    float scale = routeScales[route];
                    for (int col = 0; col < hidden; ++col) {
                        result[(size_t)token * hidden + col] +=
                            down[(size_t)local * hidden + col] * scale;
                    }
                }
            }
            for (float &value : result) {
                value = BFloat16ToFloat(
                    FloatToBFloat16RoundToNearest(value));
            }
            return fastllm::Data(
                fastllm::DataType::FLOAT32, {batch, hidden}, result);
        }
    };

#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
    struct MergeMoeW4A8BenchmarkState {
        std::shared_ptr<MergeMoeW4A8Fixture> fixture;
        fastllm::Data input, w1, w2, output;
        std::vector<std::shared_ptr<fastllm::Data>> ownedWeights;
        std::vector<fastllm::Data*> weights;

        explicit MergeMoeW4A8BenchmarkState(
                std::shared_ptr<MergeMoeW4A8Fixture> fixture)
            : fixture(std::move(fixture)) {
            fastllm::Data inputSource(
                fastllm::DataType::BFLOAT16,
                {this->fixture->batch, this->fixture->hidden},
                this->fixture->inputValues);
            input.CopyFrom(inputSource);
            input.ToDevice(fastllm::DataDevice::CUDA);

            weights.assign((size_t)(this->fixture->experts + 1) * 2, nullptr);
            for (int expert = 0; expert < this->fixture->experts; ++expert) {
                auto gateWeight = std::make_shared<fastllm::Data>();
                auto downWeight = std::make_shared<fastllm::Data>();
                fastllm::Data unusedInput, unusedBias, unusedOutput;
                this->fixture->gateFixtures[expert]->InitDeviceData(
                    unusedInput, *gateWeight, unusedBias, unusedOutput);
                this->fixture->downFixtures[expert]->InitDeviceData(
                    unusedInput, *downWeight, unusedBias, unusedOutput);
                weights[(expert + 1) * 2] = gateWeight.get();
                weights[(expert + 1) * 2 + 1] = downWeight.get();
                ownedWeights.push_back(std::move(gateWeight));
                ownedWeights.push_back(std::move(downWeight));
            }
        }

        void Run() {
            if (!FastllmCudaBFloat16MergeMOEW4A8GroupedIndexed(
                    input, w1, w2, output, weights.data(),
                    static_cast<int>(weights.size()),
                    fixture->routeRows.data(), fixture->routeScales.data(),
                    fixture->routePositions.data(),
                    fixture->expertStarts.data(),
                    fixture->expertCounts.data(), fixture->batch,
                    fixture->topk, fixture->batch * fixture->topk,
                    fixture->hidden, fixture->inter)) {
                throw std::runtime_error("W4A8 grouped MoE launch failed");
            }
        }
    };
#endif

    static int CountElements(const std::vector<int> &dims) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        return count;
    }

    static double FloatBytes(const std::vector<int> &dims) {
        return (double) CountElements(dims) * (double) sizeof(float);
    }

    static std::string FormatComputeSpeed(double computeTops) {
        std::ostringstream os;
        if (computeTops < 1.0) {
            os << std::fixed << std::setprecision(4) << computeTops * 1000.0 << " GOPS";
        } else {
            os << std::fixed << std::setprecision(4) << computeTops << " TOPS";
        }
        return os.str();
    }

    static std::string FormatIOSpeed(double gbps) {
        std::ostringstream os;
        if (gbps < 1.0) {
            os << std::fixed << std::setprecision(4) << gbps * 1000.0 << " MB/s";
        } else {
            os << std::fixed << std::setprecision(4) << gbps << " GB/s";
        }
        return os.str();
    }

    static std::vector<float> ToFloatVector(fastllm::Data data) {
        data.ToDevice(fastllm::DataDevice::CPU);
        size_t count = data.Count(0);
        std::vector<float> result(count);
        if (data.dataType == fastllm::DataType::FLOAT32) {
            const float *ptr = reinterpret_cast<const float*>(data.cpuData);
            for (size_t i = 0; i < count; i++) {
                result[i] = ptr[i];
            }
            return result;
        }
        throw std::runtime_error("only FLOAT32 outputs are supported in optest currently");
    }

    static std::vector<int32_t> ToInt32Vector(fastllm::Data data) {
        data.ToDevice(fastllm::DataDevice::CPU);
        size_t count = data.Count(0);
        if (data.dataType != fastllm::DataType::INT32) {
            throw std::runtime_error("only INT32 index outputs are supported");
        }
        const int32_t *ptr = reinterpret_cast<const int32_t*>(data.cpuData);
        return std::vector<int32_t>(ptr, ptr + count);
    }

    static fastllm::Data ConvertToFloat32Data(const fastllm::Data &data) {
        fastllm::Data output;
        fastllm::ToDataType(data, output, fastllm::DataType::FLOAT32);
        output.ToDevice(fastllm::DataDevice::CPU);
        return output;
    }

#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
    static fastllm::Data MergeMoeW4A8DenseReference(
            const MergeMoeW4A8Fixture &fixture) {
        const int totalTasks = fixture.batch * fixture.topk;
        std::vector<float> routedValues(
            (size_t)totalTasks * fixture.hidden, 0.0f);
        fastllm::Data emptyBias;

        for (int expert = 0; expert < fixture.experts; ++expert) {
            int count = fixture.expertCounts[expert];
            if (count == 0) {
                continue;
            }

            std::vector<float> expertValues((size_t)count * fixture.hidden);
            for (int local = 0; local < count; ++local) {
                int route = fixture.expertStarts[expert] + local;
                int token = fixture.routeRows[route];
                std::copy_n(
                    fixture.inputValues.data() + (size_t)token * fixture.hidden,
                    fixture.hidden,
                    expertValues.data() + (size_t)local * fixture.hidden);
            }

            fastllm::Data expertSource(
                fastllm::DataType::BFLOAT16, {count, fixture.hidden},
                expertValues);
            fastllm::Data expertInput;
            expertInput.CopyFrom(expertSource);
            expertInput.ToDevice(fastllm::DataDevice::CUDA);

            fastllm::Data gateWeight, gateOutput, activated;
            fixture.gateFixtures[expert]->InitDeviceWeight(gateWeight);
            gateOutput.dataType = fastllm::DataType::BFLOAT16;
            gateOutput.UpdateUnitSize();
            gateOutput.Resize({count, fixture.inter * 2});
            gateOutput.Allocate(false);
            gateOutput.ToDevice(fastllm::DataDevice::CUDA, false);
            if (!TryCudaCutlassW4A8(
                    expertInput, gateWeight, emptyBias, gateOutput,
                    count, fixture.hidden, fixture.inter * 2)) {
                throw std::runtime_error(
                    "W4A8 dense gate reference launch failed");
            }

            activated.dataType = fastllm::DataType::BFLOAT16;
            activated.UpdateUnitSize();
            activated.Resize({count, fixture.inter});
            activated.Allocate(false);
            activated.ToDevice(fastllm::DataDevice::CUDA, false);
            if (!FastllmCudaSwiglu(gateOutput, activated)) {
                throw std::runtime_error(
                    "W4A8 dense reference SwiGLU launch failed");
            }

            fastllm::Data downWeight, routeOutput;
            fixture.downFixtures[expert]->InitDeviceWeight(downWeight);
            routeOutput.dataType = fastllm::DataType::BFLOAT16;
            routeOutput.UpdateUnitSize();
            routeOutput.Resize({count, fixture.hidden});
            routeOutput.Allocate(false);
            routeOutput.ToDevice(fastllm::DataDevice::CUDA, false);
            if (!TryCudaCutlassW4A8(
                    activated, downWeight, emptyBias, routeOutput,
                    count, fixture.inter, fixture.hidden)) {
                throw std::runtime_error(
                    "W4A8 dense down reference launch failed");
            }

            std::vector<float> routeValues =
                ToFloatVector(ConvertToFloat32Data(routeOutput));
            for (int local = 0; local < count; ++local) {
                int route = fixture.expertStarts[expert] + local;
                std::copy_n(
                    routeValues.data() + (size_t)local * fixture.hidden,
                    fixture.hidden,
                    routedValues.data() + (size_t)route * fixture.hidden);
            }
        }

        std::vector<float> result(
            (size_t)fixture.batch * fixture.hidden, 0.0f);
        for (int token = 0; token < fixture.batch; ++token) {
            for (int col = 0; col < fixture.hidden; ++col) {
                float value = 0.0f;
                for (int slot = 0; slot < fixture.topk; ++slot) {
                    int route =
                        fixture.routePositions[token * fixture.topk + slot];
                    value +=
                        routedValues[(size_t)route * fixture.hidden + col] *
                        fixture.routeScales[route];
                }
                result[(size_t)token * fixture.hidden + col] =
                    BFloat16ToFloat(
                        FloatToBFloat16RoundToNearest(value));
            }
        }
        return fastllm::Data(
            fastllm::DataType::FLOAT32,
            {fixture.batch, fixture.hidden}, result);
    }
#endif

    static ComparisonStats CompareData(const fastllm::Data &expectedData, const fastllm::Data &actualData,
                                       float atol, float rtol) {
        fastllm::Data expected(expectedData);
        fastllm::Data actual(actualData);
        expected.ToDevice(fastllm::DataDevice::CPU);
        actual.ToDevice(fastllm::DataDevice::CPU);
        if (expected.dims != actual.dims) {
            throw std::runtime_error("output shape mismatch");
        }
        std::vector<float> expectedVec = ToFloatVector(expected);
        std::vector<float> actualVec = ToFloatVector(actual);

        ComparisonStats stats;
        double absDiffSum = 0.0;
        for (size_t i = 0; i < expectedVec.size(); i++) {
            float absDiff = std::fabs(expectedVec[i] - actualVec[i]);
            absDiffSum += absDiff;
            float relDiff = absDiff / std::max(std::fabs(expectedVec[i]), 1e-6f);
            stats.maxAbsDiff = std::max(stats.maxAbsDiff, absDiff);
            stats.maxRelDiff = std::max(stats.maxRelDiff, relDiff);
            float toleranceExcess =
                absDiff - (atol + rtol * std::fabs(expectedVec[i]));
            if (toleranceExcess > stats.maxToleranceExcess) {
                stats.passed = false;
                stats.maxToleranceExcess = toleranceExcess;
                stats.mismatchIndex = i;
                stats.expected = expectedVec[i];
                stats.actual = actualVec[i];
            }
            if (toleranceExcess > 0.0f) {
                stats.failedElements++;
            }
        }
        stats.meanAbsDiff = expectedVec.empty()
            ? 0.0f
            : static_cast<float>(absDiffSum / expectedVec.size());
        return stats;
    }

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

    struct OpCase {
        std::string name;
        std::string description;
        std::function<OpTestParams()> makeDefaultParams;
        std::function<bool(const OpTestParams&, const std::string&)> canRun;
        std::function<fastllm::Data(const OpTestParams&, const std::string&)> run;
        std::function<std::function<void()>(const OpTestParams&, const std::string&)> makeBenchmarkRun = nullptr;
        std::function<BenchmarkResult(const OpTestParams&, const std::string&, int, int)> benchmarkOverride = nullptr;
        std::function<double(const OpTestParams&)> GetIOBytes = nullptr;
        std::function<double(const OpTestParams&)> GetComputeOps = nullptr;
        std::function<fastllm::Data(
            const OpTestParams&, const std::string&)> runReference = nullptr;
        bool benchmarkOnly = false;

        OpCase() = default;

        OpCase(std::string name, std::string description,
               std::function<OpTestParams()> makeDefaultParams,
               std::function<bool(const OpTestParams&, const std::string&)> canRun,
               std::function<fastllm::Data(const OpTestParams&, const std::string&)> run,
               std::function<double(const OpTestParams&)> GetIOBytes,
               std::function<double(const OpTestParams&)> GetComputeOps)
            : name(std::move(name)), description(std::move(description)),
              makeDefaultParams(std::move(makeDefaultParams)), canRun(std::move(canRun)),
              run(std::move(run)), GetIOBytes(std::move(GetIOBytes)),
              GetComputeOps(std::move(GetComputeOps)) {}

        OpCase(std::string name, std::string description,
               std::function<OpTestParams()> makeDefaultParams,
               std::function<bool(const OpTestParams&, const std::string&)> canRun,
               std::function<fastllm::Data(const OpTestParams&, const std::string&)> run,
               std::function<std::function<void()>(const OpTestParams&, const std::string&)> makeBenchmarkRun,
               std::function<double(const OpTestParams&)> GetIOBytes,
               std::function<double(const OpTestParams&)> GetComputeOps)
            : name(std::move(name)), description(std::move(description)),
              makeDefaultParams(std::move(makeDefaultParams)), canRun(std::move(canRun)),
              run(std::move(run)), makeBenchmarkRun(std::move(makeBenchmarkRun)),
              GetIOBytes(std::move(GetIOBytes)), GetComputeOps(std::move(GetComputeOps)) {}

        OpCase(std::string name, std::string description,
               std::function<OpTestParams()> makeDefaultParams,
               std::function<bool(const OpTestParams&, const std::string&)> canRun,
               std::function<fastllm::Data(const OpTestParams&, const std::string&)> run,
               std::function<BenchmarkResult(const OpTestParams&, const std::string&, int, int)> benchmarkOverride,
               std::function<double(const OpTestParams&)> GetIOBytes,
               std::function<double(const OpTestParams&)> GetComputeOps,
               bool benchmarkOnly)
            : name(std::move(name)), description(std::move(description)),
              makeDefaultParams(std::move(makeDefaultParams)), canRun(std::move(canRun)),
              run(std::move(run)), benchmarkOverride(std::move(benchmarkOverride)),
              GetIOBytes(std::move(GetIOBytes)), GetComputeOps(std::move(GetComputeOps)),
              benchmarkOnly(benchmarkOnly) {}
    };

    static bool CanRunOnDevice(const std::string &device, const std::string &opType,
                               const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams,
                               const fastllm::IntDict &intParams) {
        ScopedFirstDevice guard(device);
        return ((fastllm::Executor*) fastllm::GetExecutor())->CanRunOnFirstDevice(opType, datas, floatParams, intParams);
    }

    static void SyncBenchmarkDevice(const std::string &device) {
#ifdef USE_CUDA
        if (device.rfind("cuda", 0) == 0 || device.rfind("multicuda", 0) == 0) {
            ForceDeviceSync();
        }
#endif
    }

    static BenchmarkResult Benchmark(const OpCase &opCase, const OpTestParams &params,
                                     const std::string &device, int warmup, int iters) {
        if (opCase.benchmarkOverride) {
            return opCase.benchmarkOverride(params, device, warmup, iters);
        }
        std::function<void()> benchRun;
        if (opCase.makeBenchmarkRun) {
            benchRun = opCase.makeBenchmarkRun(params, device);
        }
        if (benchRun) {
            for (int i = 0; i < warmup; i++) {
                benchRun();
                SyncBenchmarkDevice(device);
            }
            auto begin = Clock::now();
            for (int i = 0; i < iters; i++) {
                benchRun();
                SyncBenchmarkDevice(device);
            }
            auto end = Clock::now();
            double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();
            BenchmarkResult result;
            result.avgMs = totalMs / std::max(iters, 1);
            if (opCase.GetIOBytes) {
                result.bytesMoved = opCase.GetIOBytes(params);
            }
            if (opCase.GetComputeOps) {
                result.flops = opCase.GetComputeOps(params);
            }
            double seconds = result.avgMs / 1000.0;
            if (seconds > 0.0 && result.bytesMoved > 0.0) {
                result.bandwidthGBps = result.bytesMoved / seconds / 1e9;
            }
            if (seconds > 0.0 && result.flops > 0.0) {
                result.computeTFlops = result.flops / seconds / 1e12;
            }
            return result;
        }

        for (int i = 0; i < warmup; i++) {
            fastllm::Data output = opCase.run(params, device);
            output.ToDevice(fastllm::DataDevice::CPU);
        }
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            fastllm::Data output = opCase.run(params, device);
            output.ToDevice(fastllm::DataDevice::CPU);
        }
        auto end = Clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();
        BenchmarkResult result;
        result.avgMs = totalMs / std::max(iters, 1);
        if (opCase.GetIOBytes) {
            result.bytesMoved = opCase.GetIOBytes(params);
        }
        if (opCase.GetComputeOps) {
            result.flops = opCase.GetComputeOps(params);
        }
        double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0 && result.bytesMoved > 0.0) {
            result.bandwidthGBps = result.bytesMoved / seconds / 1e9;
        }
        if (seconds > 0.0 && result.flops > 0.0) {
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
    }

    static std::vector<std::string> GetAvailableDevices() {
        std::vector<std::string> devices = {"cpu"};

        if (fastllm::HasDeviceType("cuda")) {
#ifdef USE_CUDA
            int cudaDeviceCount = FastllmCudaGetDeviceCount();
            if (cudaDeviceCount > 0) {
                devices.push_back("cuda:0");
            }
            if (fastllm::HasDeviceType("multicuda") && cudaDeviceCount > 1) {
                std::string spec = "multicuda:";
                for (int i = 0; i < cudaDeviceCount; i++) {
                    if (i > 0) {
                        spec += ",";
                    }
                    spec += std::to_string(i);
                }
                devices.push_back(spec);
            }
#else
            devices.push_back("cuda");
#endif
        }
        if (fastllm::HasDeviceType("numa")) {
            devices.push_back("numa");
        }
        if (fastllm::HasDeviceType("numas")) {
            devices.push_back("numas");
        }
        if (fastllm::HasDeviceType("tfacc")) {
            devices.push_back("tfacc");
        }
        if (fastllm::HasDeviceType("tops")) {
            devices.push_back("tops");
        }
        return devices;
    }

    static bool DeviceSelected(const std::vector<std::string> &filters, const std::string &device) {
        if (filters.empty()) {
            return true;
        }
        for (const auto &filter : filters) {
            if (filter == "all" || filter == device) {
                return true;
            }
            if (device.rfind(filter, 0) == 0) {
                return true;
            }
        }
        return false;
    }

    static OpCase MakeAddToCase() {
        return {
            "addto",
            "input0 += input1 * alpha",
            []() {
                OpTestParams params;
                params.Add("dims", "4,8", "input tensor shape");
                params.Add("alpha", "1.5", "scale applied to input1");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input0 = MakeTensor(params.GetInts("dims"), 0.1f);
                fastllm::Data input1 = MakeTensor(params.GetInts("dims"), 0.7f);
                return CanRunOnDevice(device, "AddTo", {{"input0", &input0}, {"input1", &input1}},
                                      {{"alpha", params.GetFloat("alpha")}}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input0 = MakeTensor(params.GetInts("dims"), 0.1f);
                fastllm::Data input1 = MakeTensor(params.GetInts("dims"), 0.7f);
                ScopedFirstDevice guard(device);
                fastllm::AddTo(input0, input1, params.GetFloat("alpha"));
                input0.ToDevice(fastllm::DataDevice::CPU);
                return input0;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 3.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims")) * 2.0;
            }
        };
    }

    static OpCase MakeCatCase() {
        return {
            "cat",
            "concatenate two tensors",
            []() {
                OpTestParams params;
                params.Add("dims0", "2,3", "left input shape");
                params.Add("dims1", "2,5", "right input shape");
                params.Add("axis", "1", "concat axis");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input0 = MakeTensor(params.GetInts("dims0"), 0.2f);
                fastllm::Data input1 = MakeTensor(params.GetInts("dims1"), 0.5f);
                return CanRunOnDevice(device, "Cat", {{"input0", &input0}, {"input1", &input1}},
                                      {}, {{"axis", params.GetInt("axis")}});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input0 = MakeTensor(params.GetInts("dims0"), 0.2f);
                fastllm::Data input1 = MakeTensor(params.GetInts("dims1"), 0.5f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Cat(input0, input1, params.GetInt("axis"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                std::vector<int> dims0 = params.GetInts("dims0");
                std::vector<int> dims1 = params.GetInts("dims1");
                std::vector<int> outDims = dims0;
                outDims[params.GetInt("axis")] += dims1[params.GetInt("axis")];
                return FloatBytes(dims0) + FloatBytes(dims1) + FloatBytes(outDims);
            },
            [](const OpTestParams&) {
                return 0.0;
            }
        };
    }

    static OpCase MakeMulCase() {
        return {
            "mul",
            "multiply tensor by scalar",
            []() {
                OpTestParams params;
                params.Add("dims", "4,8", "input tensor shape");
                params.Add("value", "2.0", "scalar multiplier");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.3f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Mul", {{"input", &input}, {"output", &output}},
                                      {{"v", params.GetFloat("value")}}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.3f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Mul(input, params.GetFloat("value"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims"));
            }
        };
    }

    static OpCase MakePermuteCase() {
        return {
            "permute",
            "permute tensor axes",
            []() {
                OpTestParams params;
                params.Add("dims", "2,3,4", "input tensor shape");
                params.Add("axis", "1,2,0", "permutation");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.4f);
                fastllm::Data output;
                fastllm::Data axisData(fastllm::DataType::INT32PARAM, {(int) params.GetInts("axis").size()});
                axisData.Allocate();
                std::vector<int> axis = params.GetInts("axis");
                for (int i = 0; i < (int) axis.size(); i++) {
                    ((int32_t*) axisData.cpuData)[i] = axis[i];
                }
                return CanRunOnDevice(device, "Permute", {{"input", &input}, {"axis", &axisData}, {"output", &output}}, {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.4f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Permute(input, params.GetInts("axis"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams&) {
                return 0.0;
            }
        };
    }

    static OpCase MakeSplitCase() {
        return {
            "split",
            "slice tensor on one axis",
            []() {
                OpTestParams params;
                params.Add("dims", "3,8", "input tensor shape");
                params.Add("axis", "1", "split axis");
                params.Add("start", "2", "inclusive start");
                params.Add("end", "6", "exclusive end");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.9f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Split", {{"input", &input}, {"output", &output}},
                                      {}, {{"axis", params.GetInt("axis")}, {"start", params.GetInt("start")}, {"end", params.GetInt("end")}});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.9f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Split(input, params.GetInt("axis"), params.GetInt("start"), params.GetInt("end"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                std::vector<int> outDims = params.GetInts("dims");
                outDims[params.GetInt("axis")] = params.GetInt("end") - params.GetInt("start");
                return FloatBytes(params.GetInts("dims")) + FloatBytes(outDims);
            },
            [](const OpTestParams&) {
                return 0.0;
            }
        };
    }

    static OpCase MakeMatMulCase() {
        return {
            "matmul",
            "matrix multiplication",
            []() {
                OpTestParams params;
                params.Add("m", "4", "left rows");
                params.Add("k", "8", "shared dimension");
                params.Add("n", "5", "right cols");
                params.Add("alpha", "1.0", "output scale");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int m = params.GetInt("m"), k = params.GetInt("k"), n = params.GetInt("n");
                fastllm::Data input0 = MakeTensor({m, k}, 0.2f);
                fastllm::Data input1 = MakeTensor({k, n}, 0.8f);
                fastllm::Data output;
                return CanRunOnDevice(device, "MatMul", {{"input0", &input0}, {"input1", &input1}, {"output", &output}},
                                      {{"alpha", params.GetFloat("alpha")}}, {{"group", 1}});
            },
            [](const OpTestParams &params, const std::string &device) {
                int m = params.GetInt("m"), k = params.GetInt("k"), n = params.GetInt("n");
                fastllm::Data input0 = MakeTensor({m, k}, 0.2f);
                fastllm::Data input1 = MakeTensor({k, n}, 0.8f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::MatMul(input0, input1, output, params.GetFloat("alpha"), 1);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int m = params.GetInt("m"), k = params.GetInt("k"), n = params.GetInt("n");
                return FloatBytes({m, k}) + FloatBytes({k, n}) + FloatBytes({m, n});
            },
            [](const OpTestParams &params) {
                return 2.0 * params.GetInt("m") * params.GetInt("k") * params.GetInt("n");
            }
        };
    }

    static OpCase MakeLayerNormCase() {
        return {
            "layernorm",
            "layer normalization",
            []() {
                OpTestParams params;
                params.Add("batch", "3", "batch size");
                params.Add("hidden", "8", "hidden size");
                params.Add("axis", "-1", "norm axis");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden}, 0.1f);
                fastllm::Data gamma = MakeRampTensor({hidden}, 0.8f);
                fastllm::Data beta = MakeRampTensor({hidden}, -0.3f);
                fastllm::Data output;
                return CanRunOnDevice(device, "LayerNorm", {{"input", &input}, {"gamma", &gamma}, {"beta", &beta}, {"output", &output}},
                                      {}, {{"axis", params.GetInt("axis")}});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden}, 0.1f);
                fastllm::Data gamma = MakeRampTensor({hidden}, 0.8f);
                fastllm::Data beta = MakeRampTensor({hidden}, -0.3f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::LayerNorm(input, gamma, beta, params.GetInt("axis"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                return FloatBytes({batch, hidden}) * 2.0 + FloatBytes({hidden}) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) params.GetInt("batch") * params.GetInt("hidden") * 8.0;
            }
        };
    }

    static OpCase MakeRmsNormCase() {
        return {
            "rmsnorm",
            "rms normalization",
            []() {
                OpTestParams params;
                params.Add("batch", "3", "batch size");
                params.Add("hidden", "8", "hidden size");
                params.Add("eps", "1e-5", "epsilon");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden}, 0.6f);
                fastllm::Data weight = MakeRampTensor({hidden}, 1.0f);
                fastllm::Data output;
                return CanRunOnDevice(device, "RMSNorm", {{"input", &input}, {"weight", &weight}, {"output", &output}},
                                      {{"eps", params.GetFloat("eps")}}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden}, 0.6f);
                fastllm::Data weight = MakeRampTensor({hidden}, 1.0f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::RMSNorm(input, weight, params.GetFloat("eps"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                return FloatBytes({batch, hidden}) * 2.0 + FloatBytes({hidden});
            },
            [](const OpTestParams &params) {
                return (double) params.GetInt("batch") * params.GetInt("hidden") * 6.0;
            }
        };
    }

#ifdef USE_CUDA
    struct RmsNormSpecializedBenchState {
        std::string kind;
        std::string path;
        fastllm::DataType dataType = fastllm::DataType::FLOAT16;
        int threads = 0;
        float eps = 1.0e-6f;
        fastllm::Data input, weight, gate;
        fastllm::Data legacyOutput, fastOutput;

        static void PrepareTensor(fastllm::Data &data, const std::vector<int> &dims,
                                  float seed, fastllm::DataType dataType) {
            data.CopyFrom(MakeTensor(dims, seed, 0.25f));
            fastllm::ToDataType(data, dataType);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        static void PrepareOutput(fastllm::Data &data, const std::vector<int> &dims,
                                  fastllm::DataType dataType) {
            data.dataType = dataType;
            data.UpdateUnitSize();
            data.Resize(dims);
            data.Allocate(false);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        void RunOne(const fastllm::Data &source, fastllm::Data &output, int threadCount) {
            bool ok = false;
            if (dataType == fastllm::DataType::BFLOAT16 && kind == "rmsnorm") {
                ok = FastllmCudaRMSNormBFloat16WithThreadCount(
                    source, weight, output, eps, threadCount);
            } else if (dataType == fastllm::DataType::FLOAT16 && kind == "rmsnorm") {
                ok = FastllmCudaRMSNormFloat16WithThreadCount(
                    source, weight, output, eps, threadCount);
            } else if (dataType == fastllm::DataType::FLOAT16 &&
                       kind == "rmsnorm_silu_mul") {
                ok = FastllmCudaRMSNormSiluMulFloat16WithThreadCount(
                    source, weight, gate, output, eps, threadCount);
            } else {
                throw std::runtime_error(
                    "BF16 supports rmsnorm; FP16 supports rmsnorm and rmsnorm_silu_mul");
            }
            if (!ok) {
                throw std::runtime_error("RMSNorm specialized launch failed");
            }
        }

        static void ExpectExact(const fastllm::Data &expectedData,
                                const fastllm::Data &actualData,
                                const char *label) {
            std::vector<float> expected = ToFloatVector(ConvertToFloat32Data(expectedData));
            std::vector<float> actual = ToFloatVector(ConvertToFloat32Data(actualData));
            if (expected == actual) {
                return;
            }
            for (size_t i = 0; i < expected.size(); i++) {
                if (expected[i] != actual[i]) {
                    std::ostringstream os;
                    os << label << " mismatch at " << i
                       << ": expected=" << expected[i]
                       << " actual=" << actual[i];
                    throw std::runtime_error(os.str());
                }
            }
        }

        void Check(const std::vector<int> &dims) {
            RunOne(input, legacyOutput, 0);
            RunOne(input, fastOutput, threads);
            ForceDeviceSync();
            ExpectExact(legacyOutput, fastOutput, "out-of-place RMSNorm");

            fastllm::Data legacyInplace, fastInplace;
            PrepareTensor(legacyInplace, dims, 0.413f, dataType);
            PrepareTensor(fastInplace, dims, 0.413f, dataType);
            RunOne(legacyInplace, legacyInplace, 0);
            RunOne(fastInplace, fastInplace, threads);
            ForceDeviceSync();
            ExpectExact(legacyInplace, fastInplace, "in-place RMSNorm");
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            FastllmCudaSetDevice(0);
            kind = params.GetString("kind");
            path = params.GetString("path");
            std::string dtype = params.GetString("dtype");
            if (dtype == "fp16") {
                dataType = fastllm::DataType::FLOAT16;
            } else if (dtype == "bf16") {
                dataType = fastllm::DataType::BFLOAT16;
            } else {
                throw std::runtime_error("dtype must be fp16 or bf16");
            }
            threads = params.GetInt("threads");
            eps = params.GetFloat("eps");
            int outer = params.GetInt("outer");
            int hidden = params.GetInt("hidden");
            std::vector<int> dims = {outer, hidden};
            PrepareTensor(input, dims, 0.413f, dataType);
            PrepareTensor(gate, dims, 0.927f, dataType);
            weight.CopyFrom(MakeRampTensor({hidden}, 0.731f));
            weight.ToDevice(fastllm::DataDevice::CUDA);
            PrepareOutput(legacyOutput, dims, dataType);
            PrepareOutput(fastOutput, dims, dataType);
            Check(dims);
#else
            (void)params;
            throw std::runtime_error("RMSNorm specialized benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (path == "legacy") {
                RunOne(input, legacyOutput, 0);
            } else if (path == "fast") {
                RunOne(input, fastOutput, threads);
            } else {
                throw std::runtime_error("path must be legacy or fast");
            }
        }
    };
#endif

    static BenchmarkResult BenchmarkRmsNormSpecializedCuda(
            const OpTestParams &params, const std::string &device, int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<RmsNormSpecializedBenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("RMSNorm specialized benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeRmsNormSpecializedCase() {
        return {
            "rmsnorm_fp16_specialized",
            "benchmark exact FP16/BF16 RMSNorm and FP16 RMSNorm+SiLU+mul specializations",
            []() {
                OpTestParams params;
                params.Add("kind", "rmsnorm", "rmsnorm or rmsnorm_silu_mul");
                params.Add("path", "fast", "legacy or fast");
                params.Add("dtype", "fp16", "fp16 or bf16");
                params.Add("outer", "16", "number of rows");
                params.Add("hidden", "128", "channels per row");
                params.Add("threads", "32", "fast thread count");
                params.Add("eps", "1e-6", "epsilon");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkRmsNormSpecializedCuda,
            [](const OpTestParams &params) {
                int outer = params.GetInt("outer"), hidden = params.GetInt("hidden");
                return (double)(outer * hidden * 2 * 2 + hidden * 4);
            },
            [](const OpTestParams &params) {
                return (double)params.GetInt("outer") * params.GetInt("hidden") * 6.0;
            },
            true
        };
    }

#ifdef USE_CUDA
    struct RecurrentFromConvFp16BenchState {
        std::string path;
        int tileV = 8;
        bool exactNorm = false;
        int numKHeads = 8;
        int numVHeads = 16;
        int headKDim = 128;
        int headVDim = 128;
        float eps = 1.0e-6f;
        float qScale = 1.0f / std::sqrt(128.0f);
        fastllm::Data conv, ba, normWeight, aLog, dtBias;
        fastllm::Data workState, workOutput;

        static void PrepareFp16Tensor(fastllm::Data &data,
                                      const std::vector<int> &dims,
                                      float seed, float scale) {
            data.CopyFrom(MakeTensor(dims, seed, scale));
            fastllm::ToDataType(data, fastllm::DataType::FLOAT16);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        void PrepareState(fastllm::Data &state) const {
            PrepareFp16Tensor(state,
                              {1, numVHeads, headKDim, headVDim},
                              0.517f, 0.035f);
            state.isLinearAttentionTransposed = true;
        }

        void RunOne(fastllm::Data &state, fastllm::Data &output,
                    int selectedTileV, bool selectedExactNorm) {
            if (!FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16WithConfig(
                    conv, ba, normWeight, aLog, dtBias,
                    state, output,
                    numKHeads, numVHeads, headKDim, headVDim,
                    eps, qScale, selectedTileV, selectedExactNorm)) {
                throw std::runtime_error("configured recurrent kernel launch failed");
            }
        }

        void Check() {
            fastllm::Data legacyState, candidateState;
            fastllm::Data legacyOutput, candidateOutput;
            PrepareState(legacyState);
            PrepareState(candidateState);
            RunOne(legacyState, legacyOutput, 8, false);
            RunOne(candidateState, candidateOutput, tileV, exactNorm);
            ForceDeviceSync();
            RmsNormSpecializedBenchState::ExpectExact(
                legacyOutput, candidateOutput, "recurrent output");
            RmsNormSpecializedBenchState::ExpectExact(
                legacyState, candidateState, "recurrent state");
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            FastllmCudaSetDevice(0);
            path = params.GetString("path");
            tileV = params.GetInt("tile_v");
            exactNorm = params.GetInt("exact_norm") != 0;
            numKHeads = params.GetInt("k_heads");
            numVHeads = params.GetInt("v_heads");
            headVDim = params.GetInt("v_dim");
            eps = params.GetFloat("eps");
            headKDim = 128;
            qScale = 1.0f / std::sqrt((float)headKDim);
            int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;

            PrepareFp16Tensor(conv, {1, 1, qkvDim}, 0.271f, 0.08f);
            PrepareFp16Tensor(ba, {1, 1, numVHeads * 2}, 0.913f, 0.10f);
            normWeight.CopyFrom(MakeRampTensor({headKDim}, 0.85f));
            normWeight.ToDevice(fastllm::DataDevice::CUDA);
            aLog.CopyFrom(MakeTensor({numVHeads}, 0.419f, 0.15f));
            aLog.ToDevice(fastllm::DataDevice::CUDA);
            dtBias.CopyFrom(MakeTensor({numVHeads}, 0.731f, 0.10f));
            dtBias.ToDevice(fastllm::DataDevice::CUDA);

            Check();
            PrepareState(workState);
#else
            (void)params;
            throw std::runtime_error("recurrent_from_conv_fp16 benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (path == "legacy") {
                RunOne(workState, workOutput, 8, false);
            } else if (path == "fast") {
                RunOne(workState, workOutput, tileV, exactNorm);
            } else {
                throw std::runtime_error("path must be legacy or fast");
            }
        }
    };
#endif

    static BenchmarkResult BenchmarkRecurrentFromConvFp16Cuda(
            const OpTestParams &params, const std::string &device,
            int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<RecurrentFromConvFp16BenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("recurrent_from_conv_fp16 benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeRecurrentFromConvFp16Case() {
        return {
            "recurrent_from_conv_fp16",
            "benchmark the exact Qwen3.5 recurrent-from-conv normalization path",
            []() {
                OpTestParams params;
                params.Add("path", "fast", "legacy or fast");
                params.Add("tile_v", "8", "value columns per CTA (currently 8)");
                params.Add("exact_norm", "1", "use exact single-warp q/k norm");
                params.Add("k_heads", "8", "local key heads");
                params.Add("v_heads", "16", "local value heads");
                params.Add("v_dim", "128", "value head dimension");
                params.Add("eps", "1e-6", "RMSNorm epsilon");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkRecurrentFromConvFp16Cuda,
            [](const OpTestParams &params) {
                double stateElements = (double)params.GetInt("v_heads") * 128.0 *
                                       params.GetInt("v_dim");
                return stateElements * 4.0;
            },
            [](const OpTestParams &params) {
                double stateElements = (double)params.GetInt("v_heads") * 128.0 *
                                       params.GetInt("v_dim");
                return stateElements * 8.0;
            },
            true
        };
    }

#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_NVFP4)
    struct Nvfp4SwiGLUQuantState {
        int rows = 0, hidden = 0;
        fastllm::DataType inputType = fastllm::DataType::FLOAT16;
        bool useProduction = true;
        FastllmCudaNvfp4SiluMulVersion implementation =
            FastllmCudaNvfp4SiluMulVersion::Cached;
        fastllm::Data input;
        uint8_t *packed = nullptr;
        uint8_t *scales = nullptr;
        size_t packedBytes = 0, scaleBytes = 0;

        explicit Nvfp4SwiGLUQuantState(const OpTestParams &params) {
            rows = params.GetInt("rows");
            hidden = params.GetInt("hidden");
            inputType = params.GetString("input_type") == "bf16"
                ? fastllm::DataType::BFLOAT16 : fastllm::DataType::FLOAT16;
            const std::string implementationName = params.GetString("implementation");
            if (implementationName == "production") {
                // 常规功能和性能测试必须经过正式推理入口，避免测试路径
                // 与模型主流程选择不同的融合实现。
                useProduction = true;
            } else if (implementationName == "baseline") {
                useProduction = false;
                implementation = FastllmCudaNvfp4SiluMulVersion::Baseline;
            } else if (implementationName == "optimized") {
                useProduction = false;
                implementation = FastllmCudaNvfp4SiluMulVersion::Optimized;
            } else if (implementationName == "cached") {
                useProduction = false;
                implementation = FastllmCudaNvfp4SiluMulVersion::Cached;
            } else {
                throw std::runtime_error(
                    "nvfp4_swiglu_quant implementation must be production, baseline, optimized or cached");
            }
            if (rows <= 0 || hidden <= 0 || hidden % 32 != 0 ||
                (params.GetString("input_type") != "fp16" &&
                 params.GetString("input_type") != "bf16")) {
                throw std::runtime_error(
                    "nvfp4_swiglu_quant requires rows>0, 32-aligned hidden and fp16/bf16 input");
            }
            input.CopyFrom(MakeTensor({rows, hidden * 2}, 0.619f, 0.5f));
            fastllm::ToDataType(input, inputType);
            input.ToDevice(fastllm::DataDevice::CUDA);
            packedBytes = (size_t)rows * hidden / 2;
            scaleBytes = FastllmCudaNvfp4SwizzledScaleBytes(rows, hidden);
            packed = static_cast<uint8_t *>(FastllmCudaMalloc(packedBytes));
            scales = static_cast<uint8_t *>(FastllmCudaMalloc(scaleBytes));
            if (packed == nullptr || scales == nullptr) {
                throw std::runtime_error("NVFP4 SwiGLU quant scratch allocation failed");
            }
        }

        ~Nvfp4SwiGLUQuantState() {
            FastllmCudaFree(packed);
            FastllmCudaFree(scales);
        }

        void Run() {
            const bool launched = useProduction
                ? FastllmCudaSiluMulNvfp4Quantize(
                      input.cudaData, inputType, packed, scales,
                      rows, hidden, 1.0f, nullptr)
                : FastllmCudaSiluMulNvfp4QuantizeVersion(
                      input.cudaData, inputType, packed, scales,
                      rows, hidden, 1.0f, implementation, nullptr);
            if (!launched) {
                throw std::runtime_error("NVFP4 fused SwiGLU quant launch failed");
            }
        }

        fastllm::Data DequantizedBuffers(uint8_t *sourcePacked,
                                         uint8_t *sourceScales) const {
            ForceDeviceSync();
            std::vector<uint8_t> hostPacked(packedBytes), hostScales(scaleBytes);
            FastllmCudaCopyFromDeviceToHost(hostPacked.data(), sourcePacked, packedBytes);
            FastllmCudaCopyFromDeviceToHost(hostScales.data(), sourceScales, scaleBytes);
            static const float e2m1[16] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
            std::vector<float> values((size_t)rows * hidden);
            const int kTiles = (hidden + 63) / 64;
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < hidden; ++col) {
                    int pack = col / 16;
                    int mTile = row >> 7;
                    int outerM = row & 31;
                    int innerM = (row >> 5) & 3;
                    int kTile = pack >> 2;
                    int innerK = pack & 3;
                    size_t scaleOffset = ((size_t)mTile * kTiles + kTile) * 512 +
                                         outerM * 16 + innerM * 4 + innerK;
                    uint8_t byte = hostPacked[(size_t)row * (hidden / 2) + col / 2];
                    uint8_t q = uint8_t((byte >> ((col & 1) * 4)) & 0xf);
                    values[(size_t)row * hidden + col] =
                        e2m1[q] * DecodeFp8E4M3(hostScales[scaleOffset]);
                }
            }
            return fastllm::Data(fastllm::DataType::FLOAT32, {rows, hidden}, values);
        }

        fastllm::Data Dequantized() {
            Run();
            return DequantizedBuffers(packed, scales);
        }

        fastllm::Data Reference() const {
            // 对照路径与 vLLM 测试一致：普通 SwiGLU，再单独做 NVFP4 动态量化。
            fastllm::Data ordinary;
            ordinary.dataDevice = fastllm::DataDevice::CUDA;
            ordinary.dataType = inputType;
            ordinary.UpdateUnitSize();
            ordinary.Resize({rows, hidden});
            ordinary.Allocate(false);
            if (!FastllmCudaSwiglu(input, ordinary)) {
                throw std::runtime_error("ordinary SwiGLU launch failed");
            }
            uint8_t *referencePacked = static_cast<uint8_t *>(FastllmCudaMalloc(packedBytes));
            uint8_t *referenceScales = static_cast<uint8_t *>(FastllmCudaMalloc(scaleBytes));
            if (!referencePacked || !referenceScales ||
                !FastllmCudaNvfp4QuantizeActivation(
                    ordinary.cudaData, inputType, referencePacked, referenceScales,
                    rows, hidden, 1.0f, nullptr)) {
                FastllmCudaFree(referencePacked);
                FastllmCudaFree(referenceScales);
                throw std::runtime_error("separate NVFP4 activation quantization failed");
            }
            fastllm::Data result = DequantizedBuffers(referencePacked, referenceScales);
            FastllmCudaFree(referencePacked);
            FastllmCudaFree(referenceScales);
            return result;
        }
    };
#endif

    /**
     * 按正式主流程测量SwiGLU与NVFP4融合量化的平均延迟。
     *
     * 测试状态只初始化一次，循环内持续调用Nvfp4SwiGLUQuantState::Run；
     * production配置会进一步进入FastllmCudaSiluMulNvfp4Quantize正式入口。
     * warmup循环结束后同步一次，全部计时迭代结束后再同步一次，不在每次
     * kernel启动后强制同步，从而与正式异步执行方式及vLLM基准保持一致。
     * 显式选择baseline、optimized或cached时仍复用相同计时协议，仅用于
     * 独立多版本微基准。
     *
     * @param params 算子参数，包含rows、hidden、input_type和implementation。
     * @param device 目标CUDA设备，例如cuda:0。
     * @param warmup 计时前连续执行的预热次数。
     * @param iters  纳入计时的连续执行次数。
     * @return 平均延迟、逻辑访存字节数和估算计算量；CUDA不可用时抛出异常。
     */
    static BenchmarkResult BenchmarkNvfp4SwiGLUQuantCuda(
            const OpTestParams &params, const std::string &device,
            int warmup, int iters) {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_NVFP4)
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<Nvfp4SwiGLUQuantState>(params);

        // 预热与正式推理一样连续提交，边界同步确保计时区间不包含预热。
        for (int i = 0; i < warmup; ++i) state->Run();
        ForceDeviceSync();

        const auto begin = Clock::now();
        for (int i = 0; i < iters; ++i) state->Run();
        ForceDeviceSync();
        const auto end = Clock::now();

        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        result.bytesMoved = (double)params.GetInt("rows") *
                            params.GetInt("hidden") * 4.5;
        result.flops = (double)params.GetInt("rows") *
                       params.GetInt("hidden") * 5.0;
        const double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0) {
            result.bandwidthGBps = result.bytesMoved / seconds / 1e9;
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("NVFP4 SwiGLU benchmark requires CUDA SM100+ build");
#endif
    }

    static OpCase MakeNvfp4SwiGLUQuantCase() {
        OpCase result{
            "nvfp4_swiglu_quant",
            "fused SwiGLU plus dynamic NVFP4 activation quantization",
            []() {
                OpTestParams params;
                params.Add("rows", "32", "flattened token rows");
                params.Add("hidden", "1024", "post-SwiGLU width, multiple of 32");
                params.Add("input_type", "fp16", "fp16 or bf16");
                params.Add("implementation", "production",
                           "production main path, baseline, optimized or cached CUDA kernel");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_NVFP4)
                return device.rfind("cuda:", 0) == 0 &&
                       FastllmCudaRuntimeArch() >= 100 && FastllmCudaRuntimeArch() < 130;
#else
                (void)device;
                return false;
#endif
            },
            [](const OpTestParams &params, const std::string &device) -> fastllm::Data {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_NVFP4)
                ScopedFirstDevice guard(device);
                Nvfp4SwiGLUQuantState state(params);
                return state.Dequantized();
#else
                (void)params; (void)device;
                throw std::runtime_error("nvfp4_swiglu_quant requires SM100+ CUDA build");
#endif
            },
            [](const OpTestParams &params, const std::string &device) -> std::function<void()> {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_NVFP4)
                ScopedFirstDevice guard(device);
                auto state = std::make_shared<Nvfp4SwiGLUQuantState>(params);
                return [state]() { state->Run(); };
#else
                (void)params; (void)device;
                return {};
#endif
            },
            [](const OpTestParams &params) {
                return (double)params.GetInt("rows") * params.GetInt("hidden") * 4.5;
            },
            [](const OpTestParams &params) {
                return (double)params.GetInt("rows") * params.GetInt("hidden") * 5.0;
            }};
        result.runReference = [](const OpTestParams &params, const std::string&) {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_NVFP4)
            Nvfp4SwiGLUQuantState state(params);
            return state.Reference();
#else
            (void)params;
            return fastllm::Data();
#endif
        };
        result.benchmarkOverride = BenchmarkNvfp4SwiGLUQuantCuda;
        return result;
    }

    struct LinearNvfp4Fixture {
        int batch = 0, in = 0, out = 0;
        bool useBias = false;
        bool checkRelease = false;
        bool checkFixedBackend = false;
        bool checkWarmupFallback = false;
        bool checkMarlinFallback = false;
        bool checkDeviceMove = false;
        bool checkFusionSeparation = false;
        fastllm::LinearQuantScheme quantScheme =
            fastllm::LinearQuantScheme::LEGACY_AUTO;
        fastllm::DataType inputType = fastllm::DataType::FLOAT16;
        std::vector<float> inputValues;
        std::vector<float> biasValues;
        std::vector<uint8_t> weightBytes;
        std::vector<uint8_t> groupScales;
        // 故意使用不能被 E4M3 精确表示的 global scale，
        // 防止测试意外放过“合并后再量化 group scale”的旧实现。
        float globalScale = 0.01f;

        explicit LinearNvfp4Fixture(const OpTestParams &params) {
            batch = params.GetInt("batch");
            in = params.GetInt("in");
            out = params.GetInt("out");
            useBias = params.GetInt("bias") != 0;
            checkRelease = params.GetInt("check_release") != 0;
            checkFixedBackend = params.GetInt("check_fixed_backend") != 0;
            checkWarmupFallback = params.GetInt("check_warmup_fallback") != 0;
            checkMarlinFallback = params.GetInt("check_marlin_fallback") != 0;
            checkDeviceMove = params.GetInt("check_device_move") != 0;
            checkFusionSeparation = params.GetInt("check_fusion_separation") != 0;
            const std::string scheme = params.GetString("quant_scheme");
            if (scheme == "w4a4") {
                quantScheme = fastllm::LinearQuantScheme::NVFP4_W4A4;
            } else if (scheme == "w4a16") {
                quantScheme = fastllm::LinearQuantScheme::NVFP4_W4A16;
            } else if (scheme != "auto") {
                throw std::runtime_error(
                    "linear_nvfp4 quant_scheme must be auto, w4a4 or w4a16");
            }
            const std::string dtype = params.GetString("input_type");
            inputType = dtype == "bf16" ? fastllm::DataType::BFLOAT16
                                         : fastllm::DataType::FLOAT16;
            if (batch <= 0 || in <= 0 || out <= 0 || in % 16 != 0 ||
                (dtype != "fp16" && dtype != "bf16")) {
                throw std::runtime_error(
                    "linear_nvfp4 requires positive batch/out, 16-aligned in, and fp16/bf16 input");
            }

            fastllm::Data typed(inputType, {batch, in},
                                ToFloatVector(MakeTensor({batch, in}, 0.271f, 0.35f)));
            fastllm::Data normalized;
            fastllm::ToDataType(typed, normalized, fastllm::DataType::FLOAT32);
            normalized.ToDevice(fastllm::DataDevice::CPU);
            inputValues = ToFloatVector(normalized);
            biasValues.resize(out);
            for (int i = 0; i < out; ++i) biasValues[i] = 0.001f * float((i % 17) - 8);

            const size_t rowBytes = fastllm::GetDataBytes(
                fastllm::DataType::NVFP4_BLOCK_16, 1, in);
            weightBytes.resize((size_t)out * rowBytes);
            groupScales.resize((size_t)out * (in / 16));
            for (int row = 0; row < out; ++row) {
                for (int group = 0; group < in / 16; ++group) {
                    uint8_t *block = weightBytes.data() + (size_t)row * rowBytes +
                                     (size_t)group * (8 + sizeof(float));
                    for (int pair = 0; pair < 8; ++pair) {
                        // Valid E2M1 encodings.  Avoid signed zero.
                        uint8_t low = uint8_t(1 + (row * 3 + group + pair * 5) % 7);
                        uint8_t high = uint8_t(1 + (row + group * 3 + pair * 2) % 7);
                        if ((row + group + pair) & 1) low |= 8;
                        if ((row + group + pair * 2) & 2) high |= 8;
                        block[pair] = uint8_t(low | (high << 4));
                    }
                    const uint8_t scaleByte = uint8_t(0x28 + ((row * 7 + group * 3) % 16));
                    groupScales[(size_t)row * (in / 16) + group] = scaleByte;
                    float scale = DecodeFp8E4M3(scaleByte) * globalScale;
                    std::memcpy(block + 8, &scale, sizeof(scale));
                }
            }
        }

        fastllm::Data MakeInput() const {
            return fastllm::Data(inputType, {batch, in}, inputValues);
        }

        void MakeWeight(fastllm::Data &weight) const {
            weight.dataType = fastllm::DataType::NVFP4_BLOCK_16;
            weight.blockM = 16;
            weight.blockK = 1;
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.linearQuantScheme = quantScheme;
            weight.UpdateUnitSize();
            weight.Resize({out, in});
            weight.Allocate(true);
            std::memcpy(weight.cpuData, weightBytes.data(), weightBytes.size());
            weight.nvfp4GroupScales = groupScales;
            weight.nvfp4GlobalScale = globalScale;
        }

        fastllm::Data MakeBias() const {
            return useBias
                ? fastllm::Data(fastllm::DataType::FLOAT32, {out}, biasValues)
                : fastllm::Data();
        }

        fastllm::Data Reference() const {
            static const float e2m1[16] = {
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
            const size_t rowBytes = fastllm::GetDataBytes(
                fastllm::DataType::NVFP4_BLOCK_16, 1, in);
            std::vector<float> result((size_t)batch * out, 0.0f);
            for (int token = 0; token < batch; ++token) {
                for (int row = 0; row < out; ++row) {
                    float sum = 0.0f;
                    for (int col = 0; col < in; ++col) {
                        const uint8_t *block = weightBytes.data() + (size_t)row * rowBytes +
                                               (size_t)(col / 16) * (8 + sizeof(float));
                        float scale;
                        std::memcpy(&scale, block + 8, sizeof(scale));
                        uint8_t packed = block[(col % 16) / 2];
                        uint8_t q = uint8_t((packed >> ((col & 1) * 4)) & 0xf);
                        sum += inputValues[(size_t)token * in + col] * e2m1[q] * scale;
                    }
                    result[(size_t)token * out + row] = sum +
                        (useBias ? biasValues[row] : 0.0f);
                }
            }
            return fastllm::Data(fastllm::DataType::FLOAT32, {batch, out}, result);
        }

        fastllm::Data RunCuda() const {
            fastllm::Data input = MakeInput(), weight, bias = MakeBias(), output;
            MakeWeight(weight);
            input.ToDevice(fastllm::DataDevice::CUDA);
            weight.ToDevice(fastllm::DataDevice::CUDA);
            if (checkRelease && FastllmCudaRuntimeArch() >= 100 &&
                FastllmCudaRuntimeArch() < 130 && weight.cudaData != nullptr) {
                throw std::runtime_error(
                    "NVFP4 upload-time prepack retained the original CUDA weight");
            }
            if (checkFusionSeparation) {
                fastllm::Data malformed(
                    inputType, {batch, in * 2 - 16},
                    std::vector<float>((size_t)batch * (in * 2 - 16), 0.25f));
                malformed.ToDevice(fastllm::DataDevice::CUDA);
                fastllm::Data fusedOutput;
                fusedOutput.dataType = inputType;
                fusedOutput.UpdateUnitSize();
                fusedOutput.Resize({batch, out});
                if (FastllmCudaCutlassNvfp4W4A4FromSwiglu(
                        malformed, weight, bias, fusedOutput, batch, in, out)) {
                    throw std::runtime_error(
                        "malformed NVFP4 SwiGLU unexpectedly used the fused backend");
                }
            }
            const char *oldValue = std::getenv("FASTLLM_CUDA_NVFP4_W4A4");
            const bool hadOldValue = oldValue != nullptr;
            const std::string savedValue = hadOldValue ? oldValue : "";
            if (checkWarmupFallback) {
                // 权重已在ToDevice阶段预重排。用错误的激活宽度触发首次
                // GEMM验证失败，检查cache销毁、原始权重恢复和Legacy固定。
                fastllm::Data malformed(
                    inputType, {batch, in - 16},
                    std::vector<float>((size_t)batch * (in - 16), 0.25f));
                malformed.ToDevice(fastllm::DataDevice::CUDA);
                fastllm::Data rejectedOutput;
                rejectedOutput.dataType = inputType;
                rejectedOutput.UpdateUnitSize();
                rejectedOutput.Resize({batch, out});
                if (TryCudaCutlassNvfp4W4A4(
                        malformed, weight, bias, rejectedOutput,
                        batch, out, in)) {
                    throw std::runtime_error(
                        "malformed NVFP4 warmup unexpectedly used CUTLASS");
                }
            }
            fastllm::Linear(input, weight, bias, output);
            if (checkMarlinFallback &&
                FastllmCudaGetNvfp4W4A4BackendState(
                    weight, FastllmCudaGetDevice()) !=
                    FastllmCudaNvfp4BackendState::MarlinW4A16Fallback) {
                throw std::runtime_error(
                    "NVFP4 W4A4 selector did not fix the Marlin W4A16 fallback");
            }
            if (checkRelease && FastllmCudaRuntimeArch() >= 100 &&
                FastllmCudaRuntimeArch() < 130 && weight.cudaData != nullptr) {
                throw std::runtime_error(
                    "NVFP4 W4A4 retained the original CUDA weight after repack");
            }
            fastllm::Data nativeOutput = ConvertToFloat32Data(output);
            if (checkFixedBackend && FastllmCudaRuntimeArch() >= 100 &&
                FastllmCudaRuntimeArch() < 130) {
                setenv("FASTLLM_CUDA_NVFP4_W4A4", "0", 1);
                fastllm::Data fixedOutput;
                fastllm::Linear(input, weight, bias, fixedOutput);
                if (hadOldValue) setenv("FASTLLM_CUDA_NVFP4_W4A4", savedValue.c_str(), 1);
                else unsetenv("FASTLLM_CUDA_NVFP4_W4A4");
                if (weight.cudaData != nullptr) {
                    throw std::runtime_error(
                        "NVFP4 changed a fixed CUTLASS backend during inference");
                }
            }
            if (checkWarmupFallback && weight.cudaData == nullptr) {
                throw std::runtime_error(
                    "NVFP4 warmup fallback did not retain the original CUDA weight");
            }
            if (checkWarmupFallback) {
                fastllm::Data fixedLegacyOutput;
                fastllm::Linear(input, weight, bias, fixedLegacyOutput);
                if (weight.cudaData == nullptr) {
                    throw std::runtime_error(
                        "NVFP4 warmup fallback did not retain its original CUDA weight");
                }
                if (FastllmCudaGetNvfp4W4A4BackendState(
                        weight, FastllmCudaGetDevice()) !=
                    FastllmCudaNvfp4BackendState::NativeW4A16Fallback) {
                    throw std::runtime_error(
                        "NVFP4 W4A4 selector did not fix the native W4A16 fallback");
                }
            }
            if (checkDeviceMove) {
                int deviceCount = FastllmCudaGetDeviceCount();
                if (deviceCount >= 2 && FastllmCudaRuntimeArch() >= 100 &&
                    FastllmCudaRuntimeArch() < 130) {
                    input.ToDevice(fastllm::DataDevice::CUDA, {1});
                    weight.ToDevice(fastllm::DataDevice::CUDA, {1});
                    if (useBias) bias.ToDevice(fastllm::DataDevice::CUDA, {1});
                    if (weight.cudaData != nullptr) {
                        throw std::runtime_error(
                            "NVFP4 device move retained the destination original CUDA weight");
                    }
                    FastllmCudaSetDevice(1);
                    fastllm::Data movedOutput;
                    fastllm::Linear(input, weight, bias, movedOutput);
                    if (weight.cudaData != nullptr) {
                        throw std::runtime_error(
                            "NVFP4 destination warmup retained the original CUDA weight");
                    }
                    ComparisonStats movedStats = CompareData(
                        nativeOutput, ConvertToFloat32Data(movedOutput), 0.20f, 0.20f);
                    if (!movedStats.passed) {
                        throw std::runtime_error(
                            "NVFP4 GPU 1 output differs from GPU 0 output");
                    }
                    FastllmCudaSetDevice(0);
                }
            }
            return nativeOutput;
        }
    };

    struct LinearNvfp4BenchState {
        int batch = 0, in = 0, out = 0;
        bool requireCutlass = false;
        fastllm::Data input, weight, bias, output;

        void Init(const OpTestParams &params) {
            LinearNvfp4Fixture fixture(params);
            batch = fixture.batch;
            in = fixture.in;
            out = fixture.out;
            requireCutlass = fixture.checkRelease;
            fastllm::Data hostInput = fixture.MakeInput();
            input.CopyFrom(hostInput);
            fixture.MakeWeight(weight);
            if (fixture.useBias) {
                fastllm::Data hostBias = fixture.MakeBias();
                bias.CopyFrom(hostBias);
                bias.ToDevice(fastllm::DataDevice::CUDA);
            }
            input.ToDevice(fastllm::DataDevice::CUDA);
            weight.ToDevice(fastllm::DataDevice::CUDA);
        }

        void Run() {
            fastllm::Linear(input, weight, bias, output);
        }

        void CheckRequiredBackend() const {
            if (requireCutlass && weight.cudaData != nullptr) {
                throw std::runtime_error(
                    "strict NVFP4 dense benchmark fell back from CUTLASS");
            }
        }
    };

    static BenchmarkResult BenchmarkLinearNvfp4Cuda(
            const OpTestParams &params, const std::string &device,
            int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<LinearNvfp4BenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; ++i) state->Run();
        ForceDeviceSync();
        state->CheckRequiredBackend();
        auto begin = Clock::now();
        for (int i = 0; i < iters; ++i) state->Run();
        ForceDeviceSync();
        auto end = Clock::now();

        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        result.bytesMoved = (double)state->batch * state->in * 2.0 +
                            (double)fastllm::GetDataBytes(
                                fastllm::DataType::NVFP4_BLOCK_16,
                                state->out, state->in) +
                            (double)state->batch * state->out * 2.0;
        result.flops = 2.0 * state->batch * state->in * state->out;
        const double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0) {
            result.bandwidthGBps = result.bytesMoved / seconds / 1e9;
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("linear_nvfp4 benchmark requires CUDA");
#endif
    }

    static OpCase MakeLinearNvfp4Case() {
        OpCase result{
            "linear_nvfp4",
            "NVFP4 dense: native W4A4 CUTLASS or weight-only Marlin W4A16",
            []() {
                OpTestParams params;
                params.Add("batch", "1", "rows; W4A4 and Marlin support decode M=1");
                params.Add("in", "1008", "input features, multiple of 16; CUTLASS pads to 32");
                params.Add("out", "1000", "output features; CUTLASS pads to 32 and slices");
                params.Add("bias", "1", "add FP32 bias after W4A4 GEMM");
                params.Add("input_type", "fp16", "fp16 or bf16; Marlin accepts fp16 only");
                params.Add("quant_scheme", "auto", "auto, w4a4 or w4a16");
                params.Add("check_release", "0", "require W4A4 to release the original CUDA weight");
                params.Add("check_fixed_backend", "0", "verify inference cannot switch a warmed CUTLASS backend");
                params.Add("check_warmup_fallback", "0", "force first GEMM failure and verify native W4A16 fallback remains fixed");
                params.Add("check_marlin_fallback", "0", "disable CUTLASS and verify Marlin W4A16 fallback remains fixed");
                params.Add("check_device_move", "0", "on 2+ GPUs, rebuild the fixed backend on GPU 1");
                params.Add("check_fusion_separation", "0", "verify fused SwiGLU rejection does not reject Linear CUTLASS");
                params.Add("performance_only", "0", "skip the O(MNK) CPU reference for large benchmark shapes");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
#ifdef USE_CUDA
                if (device.rfind("cuda:", 0) != 0) return false;
                LinearNvfp4Fixture fixture(params);
                int arch = FastllmCudaRuntimeArch();
#ifdef FASTLLM_ENABLE_CUTLASS_NVFP4
                if (arch >= 100 && arch < 130) return true;
#endif
                return arch >= 75 && arch < 100 && fixture.inputType == fastllm::DataType::FLOAT16 &&
                       fixture.batch >= 2 && fixture.in % 64 == 0 && fixture.out % 64 == 0;
#else
                (void)params; (void)device;
                return false;
#endif
            },
            [](const OpTestParams &params, const std::string &device) {
#ifdef USE_CUDA
                ScopedFirstDevice guard(device);
                if (params.GetInt("performance_only") != 0) {
                    fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                    marker.Allocate(0.0f);
                    return marker;
                }
                return LinearNvfp4Fixture(params).RunCuda();
#else
                (void)params; (void)device;
                throw std::runtime_error("linear_nvfp4 requires CUDA");
#endif
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                return (double)batch * in * 2.0 +
                       (double)fastllm::GetDataBytes(fastllm::DataType::NVFP4_BLOCK_16, out, in) +
                       (double)batch * out * 2.0;
            },
            [](const OpTestParams &params) {
                return 2.0 * params.GetInt("batch") * params.GetInt("in") * params.GetInt("out");
            }};
        result.runReference = [](const OpTestParams &params, const std::string&) {
            if (params.GetInt("performance_only") != 0) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            }
            return LinearNvfp4Fixture(params).Reference();
        };
        result.benchmarkOverride = BenchmarkLinearNvfp4Cuda;
        return result;
    }

    static OpCase MakeLinearCase() {
        return {
            "linear",
            "fully connected layer",
            []() {
                OpTestParams params;
                params.Add("batch", "4", "batch size");
                params.Add("in", "8", "input features");
                params.Add("out", "6", "output features");
                params.Add("weight_type", "float32",
                           "weight datatype: float32, int4group, int4group32 or int4_w4a8");
                params.Add("group_cnt", "128", "group size used by int4group quantization");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                if (params.GetString("weight_type") == "int4_w4a8") {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
                    int batch = params.GetInt("batch");
                    int in = params.GetInt("in");
                    int out = params.GetInt("out");
                    if (batch <= 0 || in <= 0 || out <= 0 ||
                        in % fastllm::COMPRESSED_W4A8_GROUP_SIZE != 0 ||
                        out % fastllm::COMPRESSED_W4A8_GROUP_SIZE != 0) {
                        throw std::runtime_error(
                            "linear weight_type=int4_w4a8 requires positive "
                            "batch and 128-aligned in/out");
                    }
                    if (device.rfind("cuda:", 0) != 0) {
                        return false;
                    }
                    ScopedFirstDevice guard(device);
                    return FastllmCudaRuntimeArch() == 90;
#else
                    (void)params;
                    (void)device;
                    return false;
#endif
                }
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                fastllm::Data input = MakeTensor({batch, in}, 0.2f);
                fastllm::Data weight;
                InitLinearWeight(params, out, in, weight);
                fastllm::Data bias = MakeRampTensor({out}, -0.1f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Linear", {{"input", &input}, {"weight", &weight}, {"bias", &bias}, {"output", &output}},
                                      {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                if (params.GetString("weight_type") == "int4_w4a8") {
                    LinearW4A8Fixture fixture(params);
                    if (device == "cpu") {
                        return fixture.CpuReference();
                    }
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
                    ScopedFirstDevice guard(device);
                    fastllm::Data input, weight, bias, output;
                    fixture.InitDeviceData(input, weight, bias, output);
                    fastllm::Linear(input, weight, bias, output);
                    return ConvertToFloat32Data(output);
#else
                    throw std::runtime_error(
                        "int4_w4a8 linear requires CUDA SM90 CUTLASS build");
#endif
                }
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                fastllm::Data input = MakeTensor({batch, in}, 0.2f);
                fastllm::Data weight;
                InitLinearWeight(params, out, in, weight);
                fastllm::Data bias = MakeRampTensor({out}, -0.1f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Linear(input, weight, bias, output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params, const std::string &device)
                -> std::function<void()> {
                if (params.GetString("weight_type") == "int4_w4a8") {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
                    auto fixture =
                        std::make_shared<LinearW4A8Fixture>(params);
                    auto input = std::make_shared<fastllm::Data>();
                    auto weight = std::make_shared<fastllm::Data>();
                    auto bias = std::make_shared<fastllm::Data>();
                    auto output = std::make_shared<fastllm::Data>();
                    {
                        ScopedFirstDevice guard(device);
                        fixture->InitDeviceData(
                            *input, *weight, *bias, *output);
                    }
                    return [device, input, weight, bias, output]() {
                        ScopedFirstDevice guard(device);
                        fastllm::Linear(*input, *weight, *bias, *output);
                    };
#else
                    throw std::runtime_error(
                        "int4_w4a8 linear requires CUDA SM90 CUTLASS build");
#endif
                }
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                auto input = std::make_shared<fastllm::Data>(MakeTensor({batch, in}, 0.2f));
                auto weight = std::make_shared<fastllm::Data>();
                InitLinearWeight(params, out, in, *weight);
                auto bias = std::make_shared<fastllm::Data>(MakeRampTensor({out}, -0.1f));
                auto output = std::make_shared<fastllm::Data>();
                return [device, input, weight, bias, output]() {
                    ScopedFirstDevice guard(device);
                    fastllm::Linear(*input, *weight, *bias, *output);
                };
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                double weightBytes = FloatBytes({out, in});
                if (params.GetString("weight_type") == "int4_w4a8") {
                    int groups = in / fastllm::COMPRESSED_W4A8_GROUP_SIZE;
                    return (double)batch * in * 2.0 +
                           (double)out * in * 0.5 +
                           (double)out * groups * 2.0 +
                           (double)out * sizeof(float) +
                           (double)batch * out * 2.0;
                }
                if (params.GetString("weight_type") == "int4group32") {
                    weightBytes = (double)fastllm::GetDataBytes(
                        fastllm::DataType::INT4_GROUP32, out, in);
                }
                return FloatBytes({batch, in}) + weightBytes +
                       FloatBytes({out}) + FloatBytes({batch, out});
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                return 2.0 * batch * in * out + (double) batch * out;
            }
        };
    }

    static OpCase MakeW4A8ActivationQuantCase() {
        return {
            "w4a8_activation_quant",
            "dynamic per-token FP8 E4M3 activation quantization for W4A8",
            []() {
                OpTestParams params;
                params.Add("tokens", "16", "number of flattened tokens");
                params.Add("hidden", "4096", "hidden size per token");
                params.Add("dtype", "bfloat16",
                           "input datatype: bfloat16 or float16");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                W4A8ActivationQuantFixture fixture(params);
                (void)fixture;
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
                if (device.rfind("cuda:", 0) != 0) {
                    return false;
                }
                ScopedFirstDevice guard(device);
                return FastllmCudaRuntimeArch() == 90;
#else
                (void)device;
                return false;
#endif
            },
            [](const OpTestParams &params, const std::string &device) {
                W4A8ActivationQuantFixture fixture(params);
                if (device == "cpu") {
                    return fixture.CpuReference();
                }
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
                ScopedFirstDevice guard(device);
                return fixture.CudaResult();
#else
                throw std::runtime_error(
                    "w4a8_activation_quant requires CUDA SM90 CUTLASS build");
#endif
            },
            [](const OpTestParams &params, const std::string &device)
                -> std::function<void()> {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
                ScopedFirstDevice guard(device);
                W4A8ActivationQuantFixture fixture(params);
                auto state =
                    std::make_shared<W4A8ActivationQuantBenchmarkState>(fixture);
                return [device, state]() {
                    ScopedFirstDevice runGuard(device);
                    state->Run();
                };
#else
                (void)params;
                (void)device;
                throw std::runtime_error(
                    "w4a8_activation_quant requires CUDA SM90 CUTLASS build");
#endif
            },
            [](const OpTestParams &params) {
                double tokens = params.GetInt("tokens");
                double hidden = params.GetInt("hidden");
                return tokens * hidden * 2.0 +
                       tokens * hidden +
                       tokens * sizeof(float);
            },
            [](const OpTestParams&) {
                return 0.0;
            }
        };
    }

    static OpCase MakeMergeMoeW4A8Case() {
        OpCase result = {
            "mergemoe_w4a8",
            "CUTLASS grouped W4A8 MoE with routed SwiGLU experts",
            []() {
                OpTestParams params;
                params.Add("batch", "16", "token batch size");
                params.Add("hidden", "256", "model hidden size");
                params.Add("inter", "256", "expert intermediate size");
                params.Add("experts", "4", "number of routed experts");
                params.Add("topk", "2", "experts selected per token");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                MergeMoeW4A8Fixture fixture(params);
                (void)fixture;
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
                if (device.rfind("cuda:", 0) != 0) {
                    return false;
                }
                ScopedFirstDevice guard(device);
                return FastllmCudaRuntimeArch() == 90;
#else
                (void)device;
                return false;
#endif
            },
            [](const OpTestParams &params, const std::string &device) {
                auto fixture =
                    std::make_shared<MergeMoeW4A8Fixture>(params);
                if (device == "cpu") {
                    return fixture->CpuReference();
                }
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
                ScopedFirstDevice guard(device);
                MergeMoeW4A8BenchmarkState state(fixture);
                state.Run();
                return ConvertToFloat32Data(state.output);
#else
                throw std::runtime_error(
                    "mergemoe_w4a8 requires CUDA SM90 CUTLASS build");
#endif
            },
            [](const OpTestParams &params, const std::string &device)
                -> std::function<void()> {
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
                ScopedFirstDevice guard(device);
                auto fixture =
                    std::make_shared<MergeMoeW4A8Fixture>(params);
                auto state =
                    std::make_shared<MergeMoeW4A8BenchmarkState>(fixture);
                return [device, state]() {
                    ScopedFirstDevice runGuard(device);
                    state->Run();
                };
#else
                (void)params;
                (void)device;
                throw std::runtime_error(
                    "mergemoe_w4a8 requires CUDA SM90 CUTLASS build");
#endif
            },
            [](const OpTestParams &params) {
                double batch = params.GetInt("batch");
                double hidden = params.GetInt("hidden");
                double inter = params.GetInt("inter");
                double experts = params.GetInt("experts");
                double topk = params.GetInt("topk");
                return batch * hidden * 2.0 +
                       experts * (2.0 * inter * hidden +
                                  hidden * inter) * 0.5 +
                       batch * topk * (2.0 * inter + hidden) * 2.0;
            },
            [](const OpTestParams &params) {
                double batch = params.GetInt("batch");
                double hidden = params.GetInt("hidden");
                double inter = params.GetInt("inter");
                double topk = params.GetInt("topk");
                return batch * topk * 6.0 * hidden * inter;
            }
        };
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_W4A8)
        result.runReference = [](
                const OpTestParams &params, const std::string &device) {
            ScopedFirstDevice guard(device);
            MergeMoeW4A8Fixture fixture(params);
            return MergeMoeW4A8DenseReference(fixture);
        };
#endif
        return result;
    }

    static OpCase MakeSiluCase() {
        return {
            "silu",
            "SiLU activation",
            []() {
                OpTestParams params;
                params.Add("dims", "4,8", "input tensor shape");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.4f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Silu", {{"input", &input}, {"output", &output}}, {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.4f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Silu(input, output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims")) * 4.0;
            }
        };
    }

    static OpCase MakeSoftmaxCase() {
        return {
            "softmax",
            "softmax on one axis",
            []() {
                OpTestParams params;
                params.Add("dims", "3,7", "input tensor shape");
                params.Add("axis", "-1", "softmax axis");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.1f);
                fastllm::Data output;
                return CanRunOnDevice(device, "SoftMax", {{"input", &input}, {"output", &output}},
                                      {}, {{"axis", params.GetInt("axis")}});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.1f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Softmax(input, output, params.GetInt("axis"));
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims")) * 5.0;
            }
        };
    }

    static OpCase MakeGeluNewCase() {
        return {
            "gelunew",
            "GELU New activation",
            []() {
                OpTestParams params;
                params.Add("dims", "4,8", "input tensor shape");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.5f);
                fastllm::Data output;
                return CanRunOnDevice(device, "GeluNew", {{"input", &input}, {"output", &output}}, {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.5f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::GeluNew(input, output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims")) * 8.0;
            }
        };
    }

    static OpCase MakeSwigluCase() {
        return {
            "swiglu",
            "SwiGLU activation",
            []() {
                OpTestParams params;
                params.Add("batch", "4", "batch size");
                params.Add("hidden", "8", "output hidden size, input will be hidden * 2");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden * 2}, 0.7f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Swiglu", {{"input", &input}, {"output", &output}}, {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden * 2}, 0.7f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Swiglu(input, output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                return FloatBytes({batch, hidden * 2}) + FloatBytes({batch, hidden});
            },
            [](const OpTestParams &params) {
                return (double) params.GetInt("batch") * params.GetInt("hidden") * 6.0;
            }
        };
    }

    static OpCase MakeAttentionCase() {
        return {
            "attention",
            "scaled dot-product attention",
            []() {
                OpTestParams params;
                params.Add("batch", "1", "batch size");
                params.Add("seq", "4", "sequence length");
                params.Add("dim", "8", "head dimension total");
                params.Add("group", "1", "attention group");
                params.Add("attention_type", "1", "1 normal, 2 no mask");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), seq = params.GetInt("seq"), dim = params.GetInt("dim");
                fastllm::Data q = MakeTensor({batch, seq, dim}, 0.1f);
                fastllm::Data k = MakeTensor({batch, seq, dim}, 0.5f);
                fastllm::Data v = MakeTensor({batch, seq, dim}, 0.9f);
                fastllm::Data mask;
                fastllm::Data output;
                return CanRunOnDevice(device, "Attention", {{"q", &q}, {"k", &k}, {"v", &v}, {"mask", &mask}, {"output", &output}},
                                      {{"scale", 1.0f / std::sqrt((float) dim)}},
                                      {{"group", params.GetInt("group")}, {"maskType", params.GetInt("attention_type") == 2 ? 2 : 0}});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), seq = params.GetInt("seq"), dim = params.GetInt("dim");
                fastllm::Data q = MakeTensor({batch, seq, dim}, 0.1f);
                fastllm::Data k = MakeTensor({batch, seq, dim}, 0.5f);
                fastllm::Data v = MakeTensor({batch, seq, dim}, 0.9f);
                fastllm::Data mask;
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Attention(q, k, v, mask, output, params.GetInt("group"),
                                   1.0f / std::sqrt((float) dim), params.GetInt("attention_type"));
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), seq = params.GetInt("seq"), dim = params.GetInt("dim");
                return FloatBytes({batch, seq, dim}) * 4.0;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), seq = params.GetInt("seq"), dim = params.GetInt("dim");
                return 4.0 * batch * seq * seq * dim;
            }
        };
    }

#ifdef USE_CUDA
    struct LinearFp8Block128BenchState {
        int batch = 16;
        int in = 4096;
        int out = 4096;
        int block = 128;
        fastllm::Data input, weight, bias, output;

        static void InitPackedWeight(fastllm::Data &weight, int out, int in, int block, float seed) {
            if (block != 128 || (in % block) != 0) {
                throw std::runtime_error("linear_fp8_block128 requires block=128 and input features aligned to 128");
            }
            weight.dataType = fastllm::DataType::FP8_E4M3_BLOCK_128;
            weight.UpdateUnitSize();
            weight.Resize({out, in});
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = block;
            weight.blockM = block;
            weight.Allocate(false);

            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            size_t perRow = fastllm::GetDataBytes(fastllm::DataType::FP8_E4M3_BLOCK_128, 1, in);
            int blocks = (in + block - 1) / block;
            for (int r = 0; r < out; r++) {
                uint8_t *row = ptr + (size_t)r * perRow;
                for (int b = 0; b < blocks; b++) {
                    uint8_t *blockPtr = row + b * (block + (int)sizeof(float));
                    for (int c = 0; c < block; c++) {
                        blockPtr[c] = static_cast<uint8_t>(0x20 + ((r * 131 + b * 17 + c + (int)(seed * 11.0f)) & 0x1f));
                    }
                    float scale = 0.0125f + 0.0001f *
                        (float)(((r / block) + b + (int)seed) % 11);
                    memcpy(blockPtr + block, &scale, sizeof(float));
                }
            }
        }

        static void InitSeparateScaleWeight(fastllm::Data &weight, int out, int in, int block, float seed) {
            if (block != 128 || (in % block) != 0 || (out % block) != 0) {
                throw std::runtime_error("linear_fp8_block128 separate layout requires in/out aligned to 128");
            }
            weight.dataType = fastllm::DataType::FP8_E4M3;
            weight.UpdateUnitSize();
            weight.Resize({out, in});
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = block;
            weight.blockM = block;
            weight.Allocate(false);

            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            for (uint64_t i = 0; i < weight.GetBytes(); i++) {
                ptr[i] = static_cast<uint8_t>(0x20 + ((i + (uint64_t)(seed * 11.0f)) & 0x1f));
            }
            int scaleRows = (out + block - 1) / block;
            int scaleCols = (in + block - 1) / block;
            weight.scales.resize((size_t)scaleRows * scaleCols);
            for (int r = 0; r < scaleRows; r++) {
                for (int c = 0; c < scaleCols; c++) {
                    float mild = 0.0125f + 0.0001f * (float)((r * scaleCols + c + (int)seed) % 11);
                    float blocky = 0.002f * (1.0f + (float)((r * 17 + c * 29 + (int)seed) % 31));
                    weight.scales[(size_t)r * scaleCols + c] = blocky + mild * 0.1f;
                }
            }
        }

        static void InitPerChannelWeight(
                fastllm::Data &weight, int out, int in, float seed) {
            if ((in % 16) != 0 || (out % 4) != 0) {
                throw std::runtime_error(
                    "linear_fp8 per-channel layout requires in aligned to 16 and out aligned to 4");
            }
            weight.dataType = fastllm::DataType::FP8_E4M3;
            weight.UpdateUnitSize();
            weight.Resize({out, in});
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = 1;
            weight.blockM = in;
            weight.perChannelAxis = 0;
            weight.Allocate(false);
            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            for (uint64_t i = 0; i < weight.GetBytes(); ++i) {
                ptr[i] = static_cast<uint8_t>(
                    0x20 + ((i * 17 + (uint64_t)(seed * 11.0f)) & 0x1f));
            }
            weight.scales.resize(out);
            for (int row = 0; row < out; ++row) {
                weight.scales[row] =
                    0.006f + 0.00015f * (float)((row + (int)seed) % 17);
            }
        }

        static void MakeBlockyInput(fastllm::Data &input, int batch, int in, int block) {
            float *ptr = reinterpret_cast<float*>(input.cpuData);
            int groups = (in + block - 1) / block;
            for (int r = 0; r < batch; r++) {
                for (int g = 0; g < groups; g++) {
                    float gain = 0.15f + 0.11f * (float)((r * 13 + g * 7) % 23);
                    for (int c = 0; c < block && g * block + c < in; c++) {
                        int col = g * block + c;
                        ptr[(size_t)r * in + col] *= gain;
                    }
                }
            }
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            batch = params.GetInt("batch");
            in = params.GetInt("in");
            out = params.GetInt("out");
            block = params.GetInt("block");
            FastllmCudaSetDevice(0);

            fastllm::Data fp32Input = MakeTensor({batch, in}, 0.23f, 0.02f);
            if (params.GetString("input_pattern") == "blocky") {
                MakeBlockyInput(fp32Input, batch, in, block);
            }
            input.CopyFrom(fp32Input);
            std::string inputType = params.GetString("input_type");
            if (inputType == "fp16") {
                fastllm::ToDataType(input, fastllm::DataType::FLOAT16);
            } else if (inputType == "bf16") {
                fastllm::ToDataType(input, fastllm::DataType::BFLOAT16);
            } else {
                throw std::runtime_error("input_type must be fp16 or bf16");
            }
            input.ToDevice(fastllm::DataDevice::CUDA);

            std::string weightLayout = params.GetString("weight_layout");
            if (weightLayout == "packed") {
                InitPackedWeight(weight, out, in, block, 0.7f);
            } else if (weightLayout == "separate") {
                InitSeparateScaleWeight(weight, out, in, block, 0.7f);
                weight.linearQuantScheme =
                    fastllm::LinearQuantScheme::FP8_W8A8_BLOCK128;
            } else if (weightLayout == "perchannel") {
                InitPerChannelWeight(weight, out, in, 0.7f);
                weight.linearQuantScheme = fastllm::LinearQuantScheme::FP8_W8A8;
            } else if (weightLayout == "tensorwise") {
                InitPerChannelWeight(weight, out, in, 0.7f);
                weight.scales.assign(1, 0.0075f);
                weight.perChannelAxis = -1;
                weight.linearQuantScheme = fastllm::LinearQuantScheme::FP8_W8A8;
            } else {
                throw std::runtime_error(
                    "weight_layout must be packed, separate, perchannel or tensorwise");
            }
            weight.ToDevice(fastllm::DataDevice::CUDA);

            if (params.GetInt("has_bias") != 0) {
                fastllm::Data fp32Bias = MakeRampTensor({out}, -0.1f);
                bias.CopyFrom(fp32Bias);
                bias.ToDevice(fastllm::DataDevice::CUDA);
            }
            output.dataType = input.dataType;
            output.UpdateUnitSize();
            output.Resize({batch, out});
            output.Allocate(false);
            output.ToDevice(fastllm::DataDevice::CUDA, false);
            ForceDeviceSync();
#else
            (void)params;
            throw std::runtime_error("linear FP8 benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            fastllm::Linear(input, weight, bias, output);
        }

        void ConvertWeightToMarlin() {
#ifdef USE_CUDA
            fastllm::Data fp32Warmup = MakeTensor({2, in}, 0.31f, 0.02f);
            fastllm::Data warmupInput;
            warmupInput.CopyFrom(fp32Warmup);
            fastllm::ToDataType(warmupInput, fastllm::DataType::FLOAT16);
            warmupInput.ToDevice(fastllm::DataDevice::CUDA);
            fastllm::Data warmupOutput;
            warmupOutput.dataType = fastllm::DataType::FLOAT16;
            warmupOutput.UpdateUnitSize();
            warmupOutput.Resize({2, out});
            warmupOutput.Allocate(false);
            warmupOutput.ToDevice(fastllm::DataDevice::CUDA, false);
            FastllmCudaSetNcclForceSync(true);
            bool ok = FastllmCudaHalfMatMulFloatFP8E4M3(
                warmupInput, weight, bias, warmupOutput, 2, in, out);
            if (!ok) {
                throw std::runtime_error("FP8 Marlin benchmark conversion failed");
            }
            ForceDeviceSync();
#endif
        }
    };

    static fastllm::Data MakeCudaOutputLike(fastllm::DataType dataType, int batch, int out) {
        fastllm::Data output;
        output.dataType = dataType;
        output.UpdateUnitSize();
        output.Resize({batch, out});
        output.Allocate(false);
        output.ToDevice(fastllm::DataDevice::CUDA, false);
        return output;
    }

    static void PrintLinearFp8CheckStats(const std::vector<float> &expected,
                                         const std::vector<float> &actual,
                                         int batch, int out) {
        if (expected.empty() || expected.size() != actual.size()) {
            throw std::runtime_error(
                "linear FP8 check received empty or mismatched output vectors");
        }
        float maxAbsDiff = 0.0f;
        float maxRelDiff = 0.0f;
        size_t maxIndex = 0;
        int over1e2 = 0;
        int over1e1 = 0;
        int over1e0 = 0;
        double sumAbsDiff = 0.0;
        for (size_t i = 0; i < expected.size(); i++) {
            float absDiff = std::fabs(expected[i] - actual[i]);
            float relDiff = absDiff / std::max(std::fabs(expected[i]), 1e-6f);
            sumAbsDiff += absDiff;
            if (absDiff > 1e-2f) {
                over1e2++;
            }
            if (absDiff > 1e-1f) {
                over1e1++;
            }
            if (absDiff > 1.0f) {
                over1e0++;
            }
            if (absDiff > maxAbsDiff) {
                maxAbsDiff = absDiff;
                maxRelDiff = relDiff;
                maxIndex = i;
            }
        }
        int row = out == 0 ? 0 : (int)(maxIndex / (size_t)out);
        int col = out == 0 ? 0 : (int)(maxIndex % (size_t)out);
        std::cout << "linear_fp8 check: batch=" << batch << ", out=" << out
                  << ", count=" << expected.size() << "\n";
        std::cout << "  max_abs_diff=" << maxAbsDiff
                  << ", max_rel_diff=" << maxRelDiff
                  << ", mean_abs_diff=" << (expected.empty() ? 0.0 : sumAbsDiff / expected.size())
                  << ", idx=" << maxIndex << " (" << row << ", " << col << ")"
                  << ", ref=" << expected[maxIndex] << ", actual=" << actual[maxIndex] << "\n";
        std::cout << "  abs_diff>1e-2: " << over1e2
                  << ", >1e-1: " << over1e1
                  << ", >1: " << over1e0 << "\n";
    }

    static void CheckLinearFp8Near(const std::vector<float> &expected,
                                   const std::vector<float> &actual,
                                   float atol, float rtol,
                                   const std::string &label) {
        if (expected.size() != actual.size()) {
            throw std::runtime_error(
                label + " output size mismatch: " +
                std::to_string(expected.size()) + " vs " +
                std::to_string(actual.size()));
        }
        size_t mismatches = 0;
        float maxExcess = 0.0f;
        size_t maxIndex = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            float tolerance = atol + rtol * std::fabs(expected[i]);
            float diff = std::fabs(expected[i] - actual[i]);
            if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]) ||
                diff > tolerance) {
                mismatches++;
                float excess = std::isfinite(diff) ? diff - tolerance : INFINITY;
                if (excess > maxExcess) {
                    maxExcess = excess;
                    maxIndex = i;
                }
            }
        }
        if (mismatches != 0) {
            throw std::runtime_error(
                label + " mismatch count: " + std::to_string(mismatches) +
                ", worst index: " + std::to_string(maxIndex));
        }
    }

    static void CheckLinearFp8Block128Cuda(const OpTestParams &params) {
        if (params.GetString("weight_layout") != "separate") {
            throw std::runtime_error("linear_fp8_block128 check currently targets separate FP8_E4M3 scale layout");
        }
        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);

        fastllm::Data ref = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);
        fastllm::Data cutlass = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);

        bool ok = false;
        if (state->input.dataType == fastllm::DataType::FLOAT16) {
            ok = FastllmCudaHalfMatMulFloatFP8E4M3(state->input, state->weight, state->bias,
                                                   ref, state->batch, state->in, state->out);
        } else if (state->input.dataType == fastllm::DataType::BFLOAT16) {
            ok = FastllmCudaBFloat16MatMulFP8E4M3(state->input, state->weight, state->bias,
                                                  ref, state->batch, state->in, state->out);
        } else {
            throw std::runtime_error("linear_fp8_block128 check requires fp16 or bf16 input");
        }
        if (!ok) {
            throw std::runtime_error("reference FP8 linear path failed");
        }
        ForceDeviceSync();

        // 被测结果走正式Linear入口；权重在Init阶段已经声明为Block128
        // W8A8，严格模式下目标CUTLASS后端未接管会直接失败。
        fastllm::Linear(state->input, state->weight, state->bias, cutlass);
        ForceDeviceSync();

        fastllm::Data ref32 = ConvertToFloat32Data(ref);
        fastllm::Data cutlass32 = ConvertToFloat32Data(cutlass);
        std::vector<float> refVec = ToFloatVector(ref32);
        std::vector<float> cutlassVec = ToFloatVector(cutlass32);
        PrintLinearFp8CheckStats(refVec, cutlassVec, state->batch, state->out);

        ComparisonStats stats = CompareData(ref32, cutlass32, 0.5f, 0.1f);
        if (!stats.passed) {
            throw std::runtime_error(
                "Block128 FP8 W8A8 output mismatch");
        }

        if (params.GetInt("print") != 0) {
            fastllm::Data input32 = ConvertToFloat32Data(state->input);
            fastllm::Data scales(fastllm::DataType::FLOAT32,
                                  {state->out / state->block, state->in / state->block},
                                  state->weight.scales);
            input32.Print("linear_fp8_check.input.float32");
            scales.Print("linear_fp8_check.weight_scales");
            ref32.Print("linear_fp8_check.ref.float32");
            cutlass32.Print("linear_fp8_check.cutlass.float32");
        }
    }

    static void CheckLinearFp8StandardCuda(const OpTestParams &params) {
        const std::string weightLayout = params.GetString("weight_layout");
        if (weightLayout != "perchannel" && weightLayout != "tensorwise") {
            throw std::runtime_error(
                "standard FP8 check requires perchannel or tensorwise weight scale");
        }
        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);
        fastllm::Data reference = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);
        fastllm::Data actual = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);
        // 旧参考kernel只实现per-channel scale。tensorwise case仅在参考
        // 计算期间把同一个标量广播成N份；正式Linear调用前恢复单元素，
        // 确保被测CUTLASS路径真正执行标量广播分支。
        const bool tensorwise = weightLayout == "tensorwise";
        const float tensorScale = tensorwise ? state->weight.scales[0] : 0.0f;
        if (tensorwise) {
            state->weight.scales.assign(state->out, tensorScale);
        }
        bool ok = state->input.dataType == fastllm::DataType::FLOAT16
            ? FastllmCudaHalfMatMulFloatFP8E4M3(state->input, state->weight, state->bias,
                                                reference, state->batch, state->in, state->out)
            : FastllmCudaBFloat16MatMulFP8E4M3(state->input, state->weight, state->bias,
                                                reference, state->batch, state->in, state->out);
        if (tensorwise) {
            state->weight.scales.assign(1, tensorScale);
        }
        if (!ok) throw std::runtime_error("standard FP8 reference failed");
        // 正确性case走正式Linear入口；严格模式保证该标准FP8权重必须由
        // Dense FP8 W8A8 CUTLASS完成，避免测试绕过生产后端选择和缓存流程。
        fastllm::Linear(
            state->input, state->weight, state->bias, actual);
        ForceDeviceSync();
        fastllm::Data reference32 = ConvertToFloat32Data(reference);
        fastllm::Data actual32 = ConvertToFloat32Data(actual);
        PrintLinearFp8CheckStats(
            ToFloatVector(reference32), ToFloatVector(actual32),
            state->batch, state->out);
        ComparisonStats stats = CompareData(reference32, actual32, 0.5f, 0.1f);
        if (!stats.passed)
            throw std::runtime_error("standard FP8 W8A8 output mismatch");
    }

    static void CheckLinearFp8StandardLifecycleCuda(
            const OpTestParams &params) {
        auto restoreEnv = [](const char *name, const std::string &value,
                             bool existed) {
            if (existed) setenv(name, value.c_str(), 1);
            else unsetenv(name);
        };
        const char *enabledValue = std::getenv("FASTLLM_CUDA_W8A8");
        const char *strictValue = std::getenv("FASTLLM_CUDA_W8A8_STRICT");
        const bool enabledExisted = enabledValue != nullptr;
        const bool strictExisted = strictValue != nullptr;
        const std::string oldEnabled = enabledValue == nullptr ? "" : enabledValue;
        const std::string oldStrict = strictValue == nullptr ? "" : strictValue;

        try {
            setenv("FASTLLM_CUDA_W8A8", "1", 1);
            setenv("FASTLLM_CUDA_W8A8_STRICT", "1", 1);
            auto fixed = std::make_shared<LinearFp8Block128BenchState>();
            fixed->Init(params);
            bool ok = FastllmCudaCutlassLinearFp8W8A8(
                fixed->input, fixed->weight, fixed->bias, fixed->output,
                fixed->batch, fixed->in, fixed->out);
            if (!ok) throw std::runtime_error("W8A8 first backend selection failed");

            // 后端固定为CUTLASS后，运行期开关变化不能令同一权重静默改走
            // Legacy；第二次调用仍必须由已经固定的CUTLASS完成。
            setenv("FASTLLM_CUDA_W8A8", "0", 1);
            ok = FastllmCudaCutlassLinearFp8W8A8(
                fixed->input, fixed->weight, fixed->bias, fixed->output,
                fixed->batch, fixed->in, fixed->out);
            if (!ok) throw std::runtime_error("fixed W8A8 backend was not retained");

            // 新权重在开关关闭时只能固定为Rejected。关闭strict以便直接
            // 观察false返回；后续主流程会据此选择既有Legacy后端。
            unsetenv("FASTLLM_CUDA_W8A8_STRICT");
            auto rejected = std::make_shared<LinearFp8Block128BenchState>();
            rejected->Init(params);
            ok = FastllmCudaCutlassLinearFp8W8A8(
                rejected->input, rejected->weight, rejected->bias,
                rejected->output, rejected->batch, rejected->in, rejected->out);
            if (ok) throw std::runtime_error("disabled W8A8 backend was not rejected");
        } catch (...) {
            restoreEnv("FASTLLM_CUDA_W8A8", oldEnabled, enabledExisted);
            restoreEnv("FASTLLM_CUDA_W8A8_STRICT", oldStrict, strictExisted);
            throw;
        }
        restoreEnv("FASTLLM_CUDA_W8A8", oldEnabled, enabledExisted);
        restoreEnv("FASTLLM_CUDA_W8A8_STRICT", oldStrict, strictExisted);
        std::cout << "Dense FP8 W8A8 fixed backend lifecycle: PASS\n";
    }

    static void CheckLinearFp8W8A8FirstFailureRejectedCuda(
            const OpTestParams &params) {
        const char *enabledValue = std::getenv("FASTLLM_CUDA_W8A8");
        const char *strictValue = std::getenv("FASTLLM_CUDA_W8A8_STRICT");
        const char *failureValue = std::getenv(
            "FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
        const std::optional<std::string> oldEnabled = enabledValue == nullptr
            ? std::nullopt : std::optional<std::string>(enabledValue);
        const std::optional<std::string> oldStrict = strictValue == nullptr
            ? std::nullopt : std::optional<std::string>(strictValue);
        const std::optional<std::string> oldFailure = failureValue == nullptr
            ? std::nullopt : std::optional<std::string>(failureValue);
        auto restore = [](const char *name,
                          const std::optional<std::string> &value) {
            if (value.has_value()) setenv(name, value->c_str(), 1);
            else unsetenv(name);
        };

        try {
            setenv("FASTLLM_CUDA_W8A8", "1", 1);
            unsetenv("FASTLLM_CUDA_W8A8_STRICT");
            setenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE", "1", 1);
            LinearFp8Block128BenchState state;
            state.Init(params);
            bool ok = FastllmCudaCutlassLinearFp8W8A8(
                state.input, state.weight, state.bias, state.output,
                state.batch, state.in, state.out);
            if (ok || FastllmCudaGetFp8W8A8BackendState(
                    state.weight, FastllmCudaGetDevice()) !=
                    FastllmCudaFp8W8A8BackendState::Rejected) {
                throw std::runtime_error(
                    "first W8A8 GEMM failure did not fix the backend to Rejected");
            }

            // 清除故障后同一权重仍必须保持Rejected，不能重新探测CUTLASS。
            unsetenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
            ok = FastllmCudaCutlassLinearFp8W8A8(
                state.input, state.weight, state.bias, state.output,
                state.batch, state.in, state.out);
            if (ok) {
                throw std::runtime_error(
                    "Rejected W8A8 weight unexpectedly retried CUTLASS");
            }
        } catch (...) {
            restore("FASTLLM_CUDA_W8A8", oldEnabled);
            restore("FASTLLM_CUDA_W8A8_STRICT", oldStrict);
            restore("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE", oldFailure);
            throw;
        }
        restore("FASTLLM_CUDA_W8A8", oldEnabled);
        restore("FASTLLM_CUDA_W8A8_STRICT", oldStrict);
        restore("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE", oldFailure);
        std::cout << "Dense FP8 W8A8 first GEMM failure -> Rejected: PASS\n";
    }

    static void CheckLinearFp8W8A8FixedFailureThrowsCuda(
            const OpTestParams &params) {
        const char *failureValue = std::getenv(
            "FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
        const std::optional<std::string> oldFailure = failureValue == nullptr
            ? std::nullopt : std::optional<std::string>(failureValue);
        auto restoreFailure = [&]() {
            if (oldFailure.has_value()) {
                setenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE",
                       oldFailure->c_str(), 1);
            } else {
                unsetenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
            }
        };

        try {
            unsetenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
            LinearFp8Block128BenchState state;
            state.Init(params);
            bool ok = FastllmCudaCutlassLinearFp8W8A8(
                state.input, state.weight, state.bias, state.output,
                state.batch, state.in, state.out);
            if (!ok || FastllmCudaGetFp8W8A8BackendState(
                    state.weight, FastllmCudaGetDevice()) !=
                    FastllmCudaFp8W8A8BackendState::Cutlass) {
                throw std::runtime_error("W8A8 backend was not fixed to CUTLASS");
            }

            setenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE", "1", 1);
            bool threw = false;
            try {
                FastllmCudaCutlassLinearFp8W8A8(
                    state.input, state.weight, state.bias, state.output,
                    state.batch, state.in, state.out);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            if (!threw || FastllmCudaGetFp8W8A8BackendState(
                    state.weight, FastllmCudaGetDevice()) !=
                    FastllmCudaFp8W8A8BackendState::Cutlass) {
                throw std::runtime_error(
                    "fixed W8A8 failure did not throw or changed backend");
            }

            // 故障消失后仍使用原来固定的CUTLASS，不允许中途切到Legacy。
            unsetenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
            if (!FastllmCudaCutlassLinearFp8W8A8(
                    state.input, state.weight, state.bias, state.output,
                    state.batch, state.in, state.out)) {
                throw std::runtime_error(
                    "fixed W8A8 backend did not recover after injected failure");
            }
        } catch (...) {
            restoreFailure();
            throw;
        }
        restoreFailure();
        std::cout << "Dense FP8 W8A8 fixed CUTLASS failure forbids fallback: PASS\n";
    }

    static void CheckLinearFp8W8A8DestructorClearsStateCuda(
            const OpTestParams &params) {
        std::optional<LinearFp8Block128BenchState> holder;
        holder.emplace();
        holder->Init(params);
        if (!FastllmCudaCutlassLinearFp8W8A8(
                holder->input, holder->weight, holder->bias, holder->output,
                holder->batch, holder->in, holder->out)) {
            throw std::runtime_error("W8A8 destructor test selection failed");
        }
        const fastllm::Data *oldAddress = &holder->weight;
        if (FastllmCudaGetFp8W8A8BackendState(
                holder->weight, FastllmCudaGetDevice()) !=
                FastllmCudaFp8W8A8BackendState::Cutlass) {
            throw std::runtime_error("W8A8 destructor test did not reach CUTLASS");
        }

        // optional在同一内嵌存储上析构并重建对象。新Data地址相同，若析构
        // 未清状态，它会错误继承旧权重的Cutlass记录。
        holder.reset();
        holder.emplace();
        if (&holder->weight != oldAddress ||
            FastllmCudaGetFp8W8A8BackendState(
                holder->weight, FastllmCudaGetDevice()) !=
                FastllmCudaFp8W8A8BackendState::Uninitialized) {
            throw std::runtime_error(
                "destroyed W8A8 weight left a stale backend state");
        }
        std::cout << "Dense FP8 W8A8 Data destructor clears backend state: PASS\n";
    }

    static void CheckLinearFp8W8A8DeviceMoveCuda(
            const OpTestParams &params) {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "Dense FP8 W8A8 GPU migration: SKIP (requires 2 GPUs)\n";
            return;
        }
        LinearFp8Block128BenchState state;
        state.Init(params);
        if (!FastllmCudaCutlassLinearFp8W8A8(
                state.input, state.weight, state.bias, state.output,
                state.batch, state.in, state.out)) {
            throw std::runtime_error("W8A8 GPU0 backend selection failed");
        }
        ForceDeviceSync();
        fastllm::Data gpu0 = ConvertToFloat32Data(state.output);

        // Data::ToDevice必须先销毁GPU0的scale/bias cache和后端状态，再从
        // Data::scales主机源在GPU1创建新cache，不能沿用extraCudaData地址。
        state.input.ToDevice(fastllm::DataDevice::CUDA, {1});
        state.weight.ToDevice(fastllm::DataDevice::CUDA, {1});
        if (!state.bias.dims.empty()) {
            state.bias.ToDevice(fastllm::DataDevice::CUDA, {1});
        }
        state.output.ToDevice(fastllm::DataDevice::CUDA, {1}, false);
        FastllmCudaSetDevice(1);
        if (FastllmCudaGetFp8W8A8BackendState(state.weight, 0) !=
                FastllmCudaFp8W8A8BackendState::Uninitialized ||
            FastllmCudaGetFp8W8A8BackendState(state.weight, 1) !=
                FastllmCudaFp8W8A8BackendState::Prepared) {
            throw std::runtime_error("W8A8 GPU migration retained stale backend state");
        }
        if (!FastllmCudaCutlassLinearFp8W8A8(
                state.input, state.weight, state.bias, state.output,
                state.batch, state.in, state.out)) {
            throw std::runtime_error("W8A8 GPU1 backend selection failed");
        }
        ForceDeviceSync();
        fastllm::Data gpu1 = ConvertToFloat32Data(state.output);
        ComparisonStats stats = CompareData(gpu0, gpu1, 0.02f, 0.02f);
        if (!stats.passed) {
            throw std::runtime_error("W8A8 GPU migration output mismatch");
        }
        FastllmCudaSetDevice(0);
        std::cout << "Dense FP8 W8A8 GPU migration rebuilds cache: PASS\n";
    }

    static void CheckLinearFp8Block128LifecycleCuda(
            const OpTestParams &params, int mode) {
        const char *enabledValue = std::getenv("FASTLLM_CUDA_W8A8");
        const char *strictValue = std::getenv("FASTLLM_CUDA_W8A8_STRICT");
        const char *failureValue = std::getenv(
            "FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
        const std::optional<std::string> oldEnabled = enabledValue == nullptr
            ? std::nullopt : std::optional<std::string>(enabledValue);
        const std::optional<std::string> oldStrict = strictValue == nullptr
            ? std::nullopt : std::optional<std::string>(strictValue);
        const std::optional<std::string> oldFailure = failureValue == nullptr
            ? std::nullopt : std::optional<std::string>(failureValue);
        auto restore = [](const char *name,
                          const std::optional<std::string> &value) {
            if (value.has_value()) setenv(name, value->c_str(), 1);
            else unsetenv(name);
        };
        auto restoreAll = [&]() {
            restore("FASTLLM_CUDA_W8A8", oldEnabled);
            restore("FASTLLM_CUDA_W8A8_STRICT", oldStrict);
            restore("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE", oldFailure);
        };

        try {
            setenv("FASTLLM_CUDA_W8A8", "1", 1);
            unsetenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");

            if (mode == 0) {
                setenv("FASTLLM_CUDA_W8A8_STRICT", "1", 1);
                LinearFp8Block128BenchState state;
                state.Init(params);
                if (!FastllmCudaCutlassLinearFp8W8A8Block128(
                        state.input, state.weight, state.bias, state.output,
                        state.batch, state.in, state.out)) {
                    throw std::runtime_error(
                        "Block128 first backend selection failed");
                }
                setenv("FASTLLM_CUDA_W8A8", "0", 1);
                if (!FastllmCudaCutlassLinearFp8W8A8Block128(
                        state.input, state.weight, state.bias, state.output,
                        state.batch, state.in, state.out)) {
                    throw std::runtime_error(
                        "fixed Block128 backend was not retained");
                }
            } else if (mode == 1) {
                unsetenv("FASTLLM_CUDA_W8A8_STRICT");
                setenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE", "1", 1);
                LinearFp8Block128BenchState state;
                state.Init(params);
                bool ok = FastllmCudaCutlassLinearFp8W8A8Block128(
                    state.input, state.weight, state.bias, state.output,
                    state.batch, state.in, state.out);
                if (ok || FastllmCudaGetFp8W8A8Block128BackendState(
                        state.weight, FastllmCudaGetDevice()) !=
                        FastllmCudaFp8W8A8BackendState::Rejected) {
                    throw std::runtime_error(
                        "Block128 first failure did not fix Rejected");
                }
                unsetenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
                if (FastllmCudaCutlassLinearFp8W8A8Block128(
                        state.input, state.weight, state.bias, state.output,
                        state.batch, state.in, state.out)) {
                    throw std::runtime_error(
                        "Rejected Block128 weight retried CUTLASS");
                }
            } else if (mode == 2) {
                setenv("FASTLLM_CUDA_W8A8_STRICT", "1", 1);
                LinearFp8Block128BenchState state;
                state.Init(params);
                if (!FastllmCudaCutlassLinearFp8W8A8Block128(
                        state.input, state.weight, state.bias, state.output,
                        state.batch, state.in, state.out)) {
                    throw std::runtime_error(
                        "Block128 backend was not fixed to CUTLASS");
                }
                setenv("FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE", "1", 1);
                bool threw = false;
                try {
                    FastllmCudaCutlassLinearFp8W8A8Block128(
                        state.input, state.weight, state.bias, state.output,
                        state.batch, state.in, state.out);
                } catch (const std::runtime_error &) {
                    threw = true;
                }
                if (!threw || FastllmCudaGetFp8W8A8Block128BackendState(
                        state.weight, FastllmCudaGetDevice()) !=
                        FastllmCudaFp8W8A8BackendState::Cutlass) {
                    throw std::runtime_error(
                        "fixed Block128 failure did not forbid fallback");
                }
            } else if (mode == 3) {
                std::optional<LinearFp8Block128BenchState> holder;
                holder.emplace();
                holder->Init(params);
                if (!FastllmCudaCutlassLinearFp8W8A8Block128(
                        holder->input, holder->weight, holder->bias,
                        holder->output, holder->batch, holder->in,
                        holder->out)) {
                    throw std::runtime_error(
                        "Block128 destructor test selection failed");
                }
                const fastllm::Data *address = &holder->weight;
                holder.reset();
                holder.emplace();
                if (&holder->weight != address ||
                    FastllmCudaGetFp8W8A8Block128BackendState(
                        holder->weight, FastllmCudaGetDevice()) !=
                        FastllmCudaFp8W8A8BackendState::Uninitialized) {
                    throw std::runtime_error(
                        "destroyed Block128 weight left stale state");
                }
            } else if (mode == 4) {
                if (FastllmCudaGetDeviceCount() < 2) {
                    std::cout << "Block128 W8A8 GPU migration: SKIP "
                              << "(requires 2 GPUs)\n";
                    restoreAll();
                    return;
                }

                setenv("FASTLLM_CUDA_W8A8_STRICT", "1", 1);
                LinearFp8Block128BenchState state;
                state.Init(params);

                if (!FastllmCudaCutlassLinearFp8W8A8Block128(
                        state.input, state.weight, state.bias, state.output,
                        state.batch, state.in, state.out)) {
                    throw std::runtime_error(
                        "Block128 GPU0 backend selection failed");
                }
                ForceDeviceSync();
                fastllm::Data gpu0 = ConvertToFloat32Data(state.output);

                state.input.ToDevice(fastllm::DataDevice::CUDA, {1});
                state.weight.ToDevice(fastllm::DataDevice::CUDA, {1});
                if (!state.bias.dims.empty()) {
                    state.bias.ToDevice(fastllm::DataDevice::CUDA, {1});
                }
                state.output.ToDevice(
                    fastllm::DataDevice::CUDA, {1}, false);
                FastllmCudaSetDevice(1);

                if (FastllmCudaGetFp8W8A8Block128BackendState(
                        state.weight, 0) !=
                            FastllmCudaFp8W8A8BackendState::Uninitialized ||
                    FastllmCudaGetFp8W8A8Block128BackendState(
                        state.weight, 1) !=
                            FastllmCudaFp8W8A8BackendState::Prepared) {
                    throw std::runtime_error(
                        "Block128 GPU migration retained stale backend state");
                }

                if (!FastllmCudaCutlassLinearFp8W8A8Block128(
                        state.input, state.weight, state.bias, state.output,
                        state.batch, state.in, state.out)) {
                    throw std::runtime_error(
                        "Block128 GPU1 backend selection failed");
                }
                ForceDeviceSync();
                fastllm::Data gpu1 = ConvertToFloat32Data(state.output);

                ComparisonStats stats = CompareData(gpu0, gpu1, 0.02f, 0.02f);
                if (!stats.passed) {
                    throw std::runtime_error(
                        "Block128 GPU migration output mismatch");
                }
                FastllmCudaSetDevice(0);
            }
        } catch (...) {
            FastllmCudaSetDevice(0);
            restoreAll();
            throw;
        }

        restoreAll();
        std::cout << "Block128 W8A8 lifecycle check " << mode
                  << ": PASS\n";
    }

    static void CheckLinearFp8PerChannelCuda(const OpTestParams &params) {
        if (params.GetString("weight_layout") != "perchannel") {
            throw std::runtime_error(
                "per-channel CUTLASS check requires weight_layout=perchannel");
        }
        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);
        fastllm::Data reference = MakeCudaOutputLike(
            state->input.dataType, state->batch, state->out);
        fastllm::Data cutlass = MakeCudaOutputLike(
            state->input.dataType, state->batch, state->out);

        bool ok = state->input.dataType == fastllm::DataType::FLOAT16
            ? FastllmCudaHalfMatMulFloatFP8E4M3(
                  state->input, state->weight, state->bias, reference,
                  state->batch, state->in, state->out)
            : FastllmCudaBFloat16MatMulFP8E4M3(
                  state->input, state->weight, state->bias, reference,
                  state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("native per-channel FP8 linear failed");
        }
        ForceDeviceSync();
        ok = FastllmCudaCutlassLinearFP8E4M3PerChannel(
            state->input, state->weight, state->bias, cutlass,
            state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("CUTLASS per-channel FP8 linear failed");
        }
        ForceDeviceSync();

        std::vector<float> refVec = ToFloatVector(ConvertToFloat32Data(reference));
        std::vector<float> cutlassVec = ToFloatVector(ConvertToFloat32Data(cutlass));
        PrintLinearFp8CheckStats(
            refVec, cutlassVec, state->batch, state->out);
        const float atol = state->input.dataType == fastllm::DataType::FLOAT16
            ? 2.0e-3f : 8.0e-3f;
        CheckLinearFp8Near(refVec, cutlassVec, atol, 2.0e-3f,
                           "CUTLASS per-channel FP8 linear");
    }

    static void CheckLinearFp8RawBatchOneCuda(const OpTestParams &params) {
        if (params.GetString("weight_layout") != "separate" ||
            params.GetString("input_type") != "fp16" ||
            params.GetInt("batch") < 2) {
            throw std::runtime_error(
                "raw FP8 batch-one check requires separate fp16 weight and batch >= 2");
        }

        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);

        // Keep the weight in its original row-major layout through the public
        // dispatch policy. This exercises the raw batch-one GEMV without
        // adding test-only state to Data or production operator metadata.
        struct ScopedNcclForceSync {
            bool previous = FastllmCudaGetNcclForceSync();
            ~ScopedNcclForceSync() {
                FastllmCudaSetNcclForceSync(previous);
            }
        } restoreForceSync;
        FastllmCudaSetNcclForceSync(false);

        fastllm::Data reference = MakeCudaOutputLike(
            state->input.dataType, state->batch, state->out);
        fastllm::Data single = MakeCudaOutputLike(
            state->input.dataType, 1, state->out);
        bool ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, state->weight, state->bias, reference,
            state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("raw FP8 multi-row reference path failed");
        }
        ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, state->weight, state->bias, single,
            1, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("raw FP8 batch-one path failed");
        }
        ForceDeviceSync();

        std::vector<float> expected =
            ToFloatVector(ConvertToFloat32Data(reference));
        expected.resize(state->out);
        std::vector<float> observed =
            ToFloatVector(ConvertToFloat32Data(single));
        PrintLinearFp8CheckStats(
            expected, observed, 1, state->out);

        int mismatches = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            float diff = std::fabs(expected[i] - observed[i]);
            float tolerance = 0.1f + 0.05f * std::fabs(expected[i]);
            if (!std::isfinite(observed[i]) || diff > tolerance) {
                mismatches++;
            }
        }
        if (mismatches != 0) {
            throw std::runtime_error(
                "raw FP8 batch-one mismatch count: " +
                std::to_string(mismatches));
        }
    }

    static void CheckLinearFp8MarlinCuda(const OpTestParams &params) {
        if (params.GetString("weight_layout") != "separate" ||
            params.GetString("input_type") != "fp16") {
            throw std::runtime_error(
                "FP8 Marlin check requires weight_layout=separate and input_type=fp16");
        }

        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);
        if (state->batch <= 8) {
            throw std::runtime_error("FP8 Marlin tail check requires batch > 8");
        }

        fastllm::Data referenceWeight;
        LinearFp8Block128BenchState::InitSeparateScaleWeight(
            referenceWeight, state->out, state->in, state->block, 0.7f);
        referenceWeight.ToDevice(fastllm::DataDevice::CUDA);

        fastllm::Data reference = MakeCudaOutputLike(
            state->input.dataType, state->batch, state->out);
        fastllm::Data actual = MakeCudaOutputLike(
            state->input.dataType, state->batch, state->out);
        fastllm::Data warmup = MakeCudaOutputLike(
            state->input.dataType, 2, state->out);
        fastllm::Data actualSingle = MakeCudaOutputLike(
            state->input.dataType, 1, state->out);

        // A fresh weight with M > 8 stays on the legacy FP8 path and provides
        // the reference. A two-row call converts the second weight to Marlin;
        // the following calls exercise its large bulk/tail split and its
        // dedicated batch-one GEMV.
        bool ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, referenceWeight, state->bias, reference,
            state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("legacy FP8 reference path failed");
        }
        ForceDeviceSync();

        ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, state->weight, state->bias, warmup,
            2, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("FP8 Marlin warmup conversion failed");
        }
        ForceDeviceSync();

        ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, state->weight, state->bias, actual,
            state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("FP8 Marlin bulk/tail path failed");
        }
        ForceDeviceSync();

        ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, state->weight, state->bias, actualSingle,
            1, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("FP8 Marlin-layout batch-one GEMV failed");
        }
        ForceDeviceSync();

        std::vector<float> expected = ToFloatVector(ConvertToFloat32Data(reference));
        std::vector<float> observed = ToFloatVector(ConvertToFloat32Data(actual));
        PrintLinearFp8CheckStats(
            expected, observed, state->batch, state->out);

        int mismatches = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            float diff = std::fabs(expected[i] - observed[i]);
            float tolerance = 0.1f + 0.05f * std::fabs(expected[i]);
            if (!std::isfinite(observed[i]) || diff > tolerance) {
                mismatches++;
            }
        }
        if (mismatches != 0) {
            throw std::runtime_error(
                "FP8 Marlin output mismatch count: " + std::to_string(mismatches));
        }

        expected.resize(state->out);
        observed = ToFloatVector(ConvertToFloat32Data(actualSingle));
        PrintLinearFp8CheckStats(expected, observed, 1, state->out);
        mismatches = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            float diff = std::fabs(expected[i] - observed[i]);
            float tolerance = 0.1f + 0.05f * std::fabs(expected[i]);
            if (!std::isfinite(observed[i]) || diff > tolerance) {
                mismatches++;
            }
        }
        if (mismatches != 0) {
            throw std::runtime_error(
                "FP8 Marlin-layout batch-one GEMV mismatch count: " +
                std::to_string(mismatches));
        }
    }

    static void CheckLinearFp8LayoutSafetyCuda(const OpTestParams &params) {
        if (params.GetString("weight_layout") != "separate" ||
            params.GetString("input_type") != "fp16") {
            throw std::runtime_error(
                "FP8 layout safety check requires weight_layout=separate and input_type=fp16");
        }
        int arch = FastllmCudaRuntimeArch();
        if (arch != 120 && arch != 121) {
            throw std::runtime_error("FP8 layout safety check targets SM120/SM121");
        }
        const char *marlinEnv = std::getenv("FASTLLM_CUDA_FP8_MARLIN");
        if (marlinEnv != nullptr && marlinEnv[0] != '\0') {
            throw std::runtime_error(
                "FP8 layout safety check requires FASTLLM_CUDA_FP8_MARLIN to be unset");
        }

        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);

        struct ScopedNcclForceSync {
            bool previous = FastllmCudaGetNcclForceSync();
            ScopedNcclForceSync() {
                FastllmCudaSetNcclForceSync(true);
            }
            ~ScopedNcclForceSync() {
                FastllmCudaSetNcclForceSync(previous);
            }
        } forceSync;

        fastllm::Data warmup = MakeCudaOutputLike(
            state->input.dataType, 2, state->out);
        bool ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, state->weight, state->bias, warmup,
            2, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("FP8 layout safety warmup failed");
        }
        ForceDeviceSync();
        if (FastllmCudaHasFp8MarlinLayout(state->weight)) {
            throw std::runtime_error(
                "SM120/SM121 warmup must preserve row-major FP8 weights");
        }

        // Also verify the defensive CUTLASS check independently of the SM120
        // default policy. The fake non-null metadata is restored before Data
        // destruction and is never dereferenced by the expected reject path.
        struct ScopedFakeMarlinLayout {
            fastllm::Data &weight;
            std::vector<void*> oldExtraCudaData;
            std::vector<void*> oldExtraCudaHalfData;

            explicit ScopedFakeMarlinLayout(fastllm::Data &weight)
                : weight(weight),
                  oldExtraCudaData(weight.extraCudaData),
                  oldExtraCudaHalfData(weight.extraCudaHalfData) {
                weight.extraCudaData.resize(4, nullptr);
                weight.extraCudaHalfData.resize(2, nullptr);
                weight.extraCudaData[3] = weight.cudaData;
                weight.extraCudaHalfData[1] = weight.cudaData;
            }

            ~ScopedFakeMarlinLayout() {
                weight.extraCudaData = std::move(oldExtraCudaData);
                weight.extraCudaHalfData = std::move(oldExtraCudaHalfData);
            }
        } fakeLayout(state->weight);

        if (!FastllmCudaHasFp8MarlinLayout(state->weight)) {
            throw std::runtime_error("FP8 layout safety test metadata setup failed");
        }
        fastllm::Data cutlass = MakeCudaOutputLike(
            state->input.dataType, state->batch, state->out);
        ok = FastllmCudaCutlassLinearFP8E4M3Block128(
            state->input, state->weight, state->bias, cutlass,
            state->batch, state->in, state->out);
        if (ok) {
            ForceDeviceSync();
            throw std::runtime_error(
                "CUTLASS must reject FP8 weights already repacked for Marlin");
        }
    }

    static void CheckLinearFp8Block128SwigluCuda(const OpTestParams &params) {
        if (params.GetString("weight_layout") != "separate") {
            throw std::runtime_error("linear_fp8_block128 swiglu check targets separate FP8_E4M3 scale layout");
        }
        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);

        fastllm::Data gateupFp32 = MakeTensor({state->batch, state->in * 2}, 0.17f, 0.03f);
        if (params.GetString("input_pattern") == "blocky") {
            LinearFp8Block128BenchState::MakeBlockyInput(gateupFp32, state->batch, state->in * 2, state->block);
        }
        fastllm::Data gateup;
        gateup.CopyFrom(gateupFp32);
        if (state->input.dataType == fastllm::DataType::FLOAT16) {
            fastllm::ToDataType(gateup, fastllm::DataType::FLOAT16);
        } else if (state->input.dataType == fastllm::DataType::BFLOAT16) {
            fastllm::ToDataType(gateup, fastllm::DataType::BFLOAT16);
        } else {
            throw std::runtime_error("linear_fp8_block128 swiglu check requires fp16 or bf16 input");
        }
        gateup.ToDevice(fastllm::DataDevice::CUDA);

        fastllm::Data swiglu = MakeCudaOutputLike(state->input.dataType, state->batch, state->in);
        fastllm::Data ref = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);
        fastllm::Data fused = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);

        fastllm::Swiglu(gateup, swiglu);
        ForceDeviceSync();

        bool ok = FastllmCudaCutlassLinearFP8E4M3Block128(
            swiglu, state->weight, state->bias, ref, state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("CUTLASS FP8 swiglu reference linear path failed");
        }
        ForceDeviceSync();

        ok = FastllmCudaCutlassLinearFP8E4M3Block128FromSwiglu(
            gateup, state->weight, state->bias, fused, state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("CUTLASS FP8 fused swiglu path failed");
        }
        ForceDeviceSync();

        fastllm::Data ref32 = ConvertToFloat32Data(ref);
        fastllm::Data fused32 = ConvertToFloat32Data(fused);
        std::vector<float> refVec = ToFloatVector(ref32);
        std::vector<float> fusedVec = ToFloatVector(fused32);
        PrintLinearFp8CheckStats(refVec, fusedVec, state->batch, state->out);

        float maxAbsDiff = 0.0f;
        for (size_t i = 0; i < refVec.size(); i++) {
            maxAbsDiff = std::max(maxAbsDiff, std::fabs(refVec[i] - fusedVec[i]));
        }
        if (maxAbsDiff > 1.0e-3f) {
            throw std::runtime_error("CUTLASS FP8 fused swiglu output mismatch");
        }

        if (params.GetInt("print") != 0) {
            fastllm::Data gateup32 = ConvertToFloat32Data(gateup);
            fastllm::Data swiglu32 = ConvertToFloat32Data(swiglu);
            gateup32.Print("linear_fp8_swiglu_check.gateup.float32");
            swiglu32.Print("linear_fp8_swiglu_check.swiglu.float32");
            ref32.Print("linear_fp8_swiglu_check.ref.float32");
            fused32.Print("linear_fp8_swiglu_check.fused.float32");
        }
    }
#endif

    static BenchmarkResult BenchmarkLinearFp8Block128Cuda(const OpTestParams &params,
                                                          const std::string &device,
                                                          int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        if (params.GetInt("check") == 1) {
            CheckLinearFp8Block128Cuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 2) {
            CheckLinearFp8Block128SwigluCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 3) {
            CheckLinearFp8MarlinCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 4) {
            CheckLinearFp8RawBatchOneCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 5) {
            CheckLinearFp8LayoutSafetyCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 6) {
            CheckLinearFp8PerChannelCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 7) {
            CheckLinearFp8StandardCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 8) {
            CheckLinearFp8StandardLifecycleCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 9) {
            CheckLinearFp8W8A8FirstFailureRejectedCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 10) {
            CheckLinearFp8W8A8FixedFailureThrowsCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 11) {
            CheckLinearFp8W8A8DestructorClearsStateCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 12) {
            CheckLinearFp8W8A8DeviceMoveCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") >= 13 &&
                   params.GetInt("check") <= 17) {
            CheckLinearFp8Block128LifecycleCuda(
                params, params.GetInt("check") - 13);
            return BenchmarkResult();
        }

        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);

        bool oldForceSync = FastllmCudaGetNcclForceSync();
        std::string kernel = params.GetString("kernel");
        if (kernel == "legacy") {
            FastllmCudaSetNcclForceSync(false);
        } else if (kernel == "marlin_batch1") {
            if (state->batch != 1) {
                throw std::runtime_error("marlin_batch1 benchmark requires batch=1");
            }
            state->ConvertWeightToMarlin();
            FastllmCudaSetNcclForceSync(false);
        } else if (kernel != "auto") {
            throw std::runtime_error("kernel must be auto, legacy, or marlin_batch1");
        }

        for (int i = 0; i < warmup; i++) {
            state->Run();
            ForceDeviceSync();
        }

        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        FastllmCudaSetNcclForceSync(oldForceSync);
        double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();

        int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
        BenchmarkResult result;
        result.avgMs = totalMs / std::max(iters, 1);
        const double weightBytes =
            (double)state->weight.GetBytes() +
            (double)state->weight.scales.size() * sizeof(float);
        result.bytesMoved = (double)batch * in * 2.0 +
                            (double)batch * out * 2.0 +
                            weightBytes +
                            (params.GetInt("has_bias") != 0
                                ? (double)out * sizeof(float)
                                : 0.0);
        result.flops = 2.0 * (double)batch * in * out;
        double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0 && result.bytesMoved > 0.0) {
            result.bandwidthGBps = result.bytesMoved / seconds / 1e9;
        }
        if (seconds > 0.0 && result.flops > 0.0) {
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("linear FP8 benchmark requires USE_CUDA");
#endif
    }

    /**
     * 执行标准Dense FP8 W8A8功能或性能测试。
     *
     * 本入口只接受per-channel或tensorwise权重scale，并把对外连续编号的
     * 检查项映射到已有公共实现；实际计算仍通过正式Linear入口完成。
     *
     * @param params  测试参数；check=1..6依次表示正确性、固定后端、首次
     *                失败拒绝、固定后端失败、析构清理和设备迁移。
     * @param device  CUDA设备名称。
     * @param warmup  性能测试预热次数。
     * @param iters   性能测试正式迭代次数。
     * @return 功能检查结果或性能统计。
     */
    static BenchmarkResult BenchmarkLinearFp8W8A8Cuda(
            const OpTestParams &params, const std::string &device,
            int warmup, int iters) {
        const std::string layout = params.GetString("weight_layout");
        if (layout != "perchannel" && layout != "tensorwise") {
            throw std::runtime_error(
                "linear_fp8_w8a8 requires perchannel or tensorwise "
                "weight_layout");
        }
        if (params.GetString("kernel") != "auto") {
            throw std::runtime_error(
                "linear_fp8_w8a8 only supports the production auto path");
        }

        const int check = params.GetInt("check");
        if (check < 0 || check > 6) {
            throw std::runtime_error("linear_fp8_w8a8 check must be 0..6");
        }

        OpTestParams forwarded = params;
        if (check != 0) {
            forwarded.Override("check", std::to_string(check + 6));
        }
        return BenchmarkLinearFp8Block128Cuda(
            forwarded, device, warmup, iters);
    }

    /**
     * 执行Block128 FP8 W8A8功能或性能测试。
     *
     * 本入口只接受分离存放的(128,128)权重scale；check=2..6复用已有
     * Block128后端生命周期检查，实际计算仍通过正式Linear入口完成。
     *
     * @param params  测试参数；check=1表示正确性，2..6表示后端生命周期。
     * @param device  CUDA设备名称。
     * @param warmup  性能测试预热次数。
     * @param iters   性能测试正式迭代次数。
     * @return 功能检查结果或性能统计。
     */
    static BenchmarkResult BenchmarkLinearFp8Block128OnlyCuda(
            const OpTestParams &params, const std::string &device,
            int warmup, int iters) {
        if (params.GetString("weight_layout") != "separate") {
            throw std::runtime_error(
                "linear_fp8_block128 requires separate weight_layout");
        }
        if (params.GetString("kernel") != "auto") {
            throw std::runtime_error(
                "linear_fp8_block128 only supports the production auto path");
        }

        const int check = params.GetInt("check");
        if (check < 0 || check > 6) {
            throw std::runtime_error(
                "linear_fp8_block128 check must be 0..6");
        }

        OpTestParams forwarded = params;
        if (check >= 2) {
            forwarded.Override("check", std::to_string(check + 11));
        }
        return BenchmarkLinearFp8Block128Cuda(
            forwarded, device, warmup, iters);
    }

    /**
     * 构造标准Dense FP8 W8A8测试入口。
     *
     * 该入口仅暴露per-channel和tensorwise scale语义，并统一调用正式
     * Linear路径；Block128和历史实验检查由其他入口承载。
     *
     * @return 标准Dense FP8 W8A8算子测试定义。
     */
    static OpCase MakeLinearFp8W8A8Case() {
        return {
            "linear_fp8_w8a8",
            "dense FP8 W8A8 linear functional and performance test",
            []() {
                OpTestParams params;
                params.Add("batch", "16", "token batch size");
                params.Add("in", "4096", "input features");
                params.Add("out", "4096", "output features");
                params.Add("block", "128", "input pattern group size");
                params.Add("input_type", "bf16", "fp16 or bf16");
                params.Add("input_pattern", "blocky", "smooth or blocky");
                params.Add("weight_layout", "perchannel",
                           "perchannel or tensorwise");
                params.Add("has_bias", "1",
                           "1 to include a float32 bias, 0 for no bias");
                params.Add("kernel", "auto", "auto only for production path");
                params.Add("check", "0",
                           "1 correctness, 2 fixed backend lifecycle, "
                           "3 first failure rejection, 4 fixed backend failure, "
                           "5 destructor cleanup, 6 GPU migration cache rebuild");
                params.Add("print", "0", "1 to print debug tensors");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkLinearFp8W8A8Cuda,
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch");
                int in = params.GetInt("in");
                int out = params.GetInt("out");
                double inputBytes = (double)batch * in * 2.0;
                double outputBytes = (double)batch * out * 2.0;
                double scaleBytes =
                    params.GetString("weight_layout") == "tensorwise"
                        ? sizeof(float)
                        : (double)out * sizeof(float);
                double biasBytes = params.GetInt("has_bias") != 0
                    ? (double)out * sizeof(float)
                    : 0.0;
                return inputBytes + outputBytes +
                       (double)out * in + scaleBytes + biasBytes;
            },
            [](const OpTestParams &params) {
                return 2.0 * (double)params.GetInt("batch") *
                       params.GetInt("in") * params.GetInt("out");
            },
            true
        };
    }

    /**
     * 构造Block128 FP8 W8A8测试入口。
     *
     * 该入口只允许独立存放的(128,128)权重scale，并通过正式Linear
     * 路径验证Block128后端及其生命周期。
     *
     * @return Block128 FP8 W8A8算子测试定义。
     */
    static OpCase MakeLinearFp8Block128Case() {
        return {
            "linear_fp8_block128",
            "Block128 FP8 W8A8 linear functional and performance test",
            []() {
                OpTestParams params;
                params.Add("batch", "16", "token batch size");
                params.Add("in", "4096", "input features");
                params.Add("out", "4096", "output features");
                params.Add("block", "128", "FP8 scale block size");
                params.Add("input_type", "bf16", "fp16 or bf16");
                params.Add("input_pattern", "blocky", "smooth or blocky");
                params.Add("weight_layout", "separate",
                           "separate Block128 scales only");
                params.Add("has_bias", "1", "1 to include a float32 bias, 0 for no bias");
                params.Add("kernel", "auto", "auto only for production path");
                params.Add("check", "0",
                           "1 correctness, 2 fixed backend lifecycle, "
                           "3 first failure rejection, 4 fixed backend failure, "
                           "5 destructor cleanup, 6 GPU migration cache rebuild");
                params.Add("print", "0", "1 to print debug tensors when check=1");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkLinearFp8Block128OnlyCuda,
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                double inputBytes = (double)batch * in * 2.0;
                double outputBytes = (double)batch * out * 2.0;
                double weightBytes = (double)out * in;
                double scaleBytes = (double)(out / 128) * (in / 128) *
                                    sizeof(float);
                double biasBytes = params.GetInt("has_bias") != 0
                    ? (double)out * sizeof(float)
                    : 0.0;
                return inputBytes + outputBytes + weightBytes +
                       scaleBytes + biasBytes;
            },
            [](const OpTestParams &params) {
                return 2.0 * (double)params.GetInt("batch") * params.GetInt("in") * params.GetInt("out");
            },
            true
        };
    }

    /**
     * 构造历史FP8专项检查入口。
     *
     * 该入口保留融合SwiGLU、Marlin、原始Batch-1和布局安全等既有测试，
     * 避免Dense与Block128入口拆分后丢失原有专项验证能力。
     *
     * @return 历史FP8专项测试定义。
     */
    static OpCase MakeLinearFp8LegacyCase() {
        return {
            "linear_fp8_legacy",
            "legacy FP8 fused, Marlin and layout safety tests",
            []() {
                OpTestParams params;
                params.Add("batch", "16", "token batch size");
                params.Add("in", "4096", "input features");
                params.Add("out", "4096", "output features");
                params.Add("block", "128", "FP8 scale block size");
                params.Add("input_type", "bf16", "fp16 or bf16");
                params.Add("input_pattern", "blocky", "smooth or blocky");
                params.Add("weight_layout", "packed",
                           "packed, separate, perchannel or tensorwise");
                params.Add("has_bias", "1",
                           "1 to include a float32 bias, 0 for no bias");
                params.Add("kernel", "auto",
                           "auto, legacy, or marlin_batch1");
                params.Add("check", "0",
                           "2 fused swiglu+quant, 3 Marlin bulk/tail, "
                           "4 raw batch-one, 5 Marlin/CUTLASS layout safety, "
                           "6 per-channel CUTLASS scaled-mm");
                params.Add("print", "0", "1 to print debug tensors");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkLinearFp8Block128Cuda,
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch");
                int in = params.GetInt("in");
                int out = params.GetInt("out");
                return (double)batch * in * 2.0 +
                       (double)batch * out * 2.0 +
                       (double)fastllm::GetDataBytes(
                           fastllm::DataType::FP8_E4M3_BLOCK_128, out, in) +
                       (double)out * sizeof(float);
            },
            [](const OpTestParams &params) {
                return 2.0 * (double)params.GetInt("batch") *
                       params.GetInt("in") * params.GetInt("out");
            },
            true
        };
    }

    struct LinearInt8W8A8State {
        int batch, in, out, check;
        std::string activationScheme;
        fastllm::Data input, weight, bias, output;
        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            batch = params.GetInt("batch"); in = params.GetInt("in"); out = params.GetInt("out");
            check = params.GetInt("check");
            std::vector<float> inputValues((size_t)batch * in);
            for (size_t i = 0; i < inputValues.size(); ++i) {
                inputValues[i] = ((int)(i * 37 % 501) - 100) * 0.001f;
            }
            fastllm::Data fp32(
                fastllm::DataType::FLOAT32, {batch, in}, inputValues);
            input.CopyFrom(fp32);
            fastllm::ToDataType(input, params.GetString("input_type") == "fp16"
                ? fastllm::DataType::FLOAT16 : fastllm::DataType::BFLOAT16);
            input.ToDevice(fastllm::DataDevice::CUDA);
            weight.dataType = fastllm::DataType::INT8_W8A8;
            weight.UpdateUnitSize(); weight.Resize({out, in}); weight.Allocate(false);
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.linearQuantScheme = fastllm::LinearQuantScheme::INT8_W8A8;
            activationScheme = params.GetString("activation_scheme");
            if (activationScheme != "symmetric" &&
                activationScheme != "symmetric_static" &&
                activationScheme != "azp_dynamic" &&
                activationScheme != "azp_static") {
                throw std::runtime_error(
                    "activation_scheme must be symmetric, symmetric_static, "
                    "azp_dynamic or azp_static");
            }
            weight.w8a8InputDynamic = activationScheme != "symmetric_static" &&
                                      activationScheme != "azp_static";
            weight.w8a8InputSymmetric =
                activationScheme == "symmetric" ||
                activationScheme == "symmetric_static";
            weight.w8a8InputScale = 0.01f;
            weight.w8a8InputZeroPoint = weight.w8a8InputSymmetric ? 0 : -3;
            weight.perChannelAxis = 0; weight.scales.resize(out);
            int8_t *q = (int8_t*)weight.cpuData;
            for (int row = 0; row < out; ++row) {
                float scale = 0.0025f + 0.00001f * (row % 31);
                weight.scales[row] = scale;
                for (int col = 0; col < in; ++col)
                    q[(size_t)row * in + col] = (int8_t)(((row * 17 + col * 13) % 255) - 127);
            }
            weight.ToDevice(fastllm::DataDevice::CUDA);
            if (params.GetInt("has_bias")) {
                bias.CopyFrom(MakeRampTensor({out}, -0.1f));
                bias.ToDevice(fastllm::DataDevice::CUDA);
            }
            output.dataType = input.dataType; output.UpdateUnitSize();
            output.Resize({batch, out}); output.Allocate(false);
            output.ToDevice(fastllm::DataDevice::CUDA, false);
#else
            (void)params;
#endif
        }
        void Run() {
            if (!FastllmCudaCutlassLinearInt8W8A8Sm90(
                    input, weight, bias, output, batch, in, out))
                throw std::runtime_error("SM90 INT8 W8A8 path rejected the test case");
        }
        void Check() {
            if (check >= 2 && check <= 4) {
                const char *reason = nullptr;
                if (check == 2) {
                    weight.zeros.assign(out, 1);
                    reason = "asymmetric weight zero-points";
                } else if (check == 3) {
                    weight.w8a8InputDynamic = false;
                    weight.w8a8InputScale = 0.0f;
                    reason = "non-positive static activation scale";
                } else {
                    weight.w8a8InputSymmetric = true;
                    weight.w8a8InputZeroPoint = 1;
                    reason = "non-zero symmetric activation zero-point";
                }
                if (FastllmCudaCutlassLinearInt8W8A8Sm90(
                        input, weight, bias, output, batch, in, out)) {
                    throw std::runtime_error(
                        std::string("SM90 INT8 W8A8 accepted ") + reason);
                }
                std::cout << "linear_int8_w8a8 rejected " << reason << "\n";
                return;
            }
            Run(); ForceDeviceSync();
            std::vector<float> x = ToFloatVector(ConvertToFloat32Data(input));
            std::vector<float> y = ToFloatVector(ConvertToFloat32Data(output));
            const int8_t *q = (const int8_t*)weight.cpuData;
            const float *b = bias.dims.empty() ? nullptr : (const float*)bias.cpuData;
            float maxDiff = 0.0f, sumDiff = 0.0f;
            int failures = 0;
            for (int r = 0; r < batch; ++r) {
                float tokenScale = weight.w8a8InputScale;
                int zeroPoint = weight.w8a8InputZeroPoint;
                if (weight.w8a8InputDynamic && weight.w8a8InputSymmetric) {
                    float maxInput = 0.0f;
                    for (int p = 0; p < in; ++p)
                        maxInput = std::max(maxInput, std::fabs(x[(size_t)r * in + p]));
                    tokenScale = std::max(maxInput / 127.0f, 1.1920928955078125e-7f);
                    zeroPoint = 0;
                } else if (weight.w8a8InputDynamic) {
                    float minimum = std::numeric_limits<float>::max();
                    float maximum = std::numeric_limits<float>::lowest();
                    for (int p = 0; p < in; ++p) {
                        float value = x[(size_t)r * in + p];
                        minimum = std::min(minimum, value);
                        maximum = std::max(maximum, value);
                    }
                    tokenScale = (maximum - minimum) / 255.0f;
                    zeroPoint = (int)std::nearbyint(
                        -128.0f - minimum / tokenScale);
                }
                for (int c = 0; c < out; ++c) {
                    int32_t dot = 0, weightSum = 0;
                    for (int p = 0; p < in; ++p) {
                        int value = (int)std::nearbyint(
                            x[(size_t)r * in + p] / tokenScale) + zeroPoint;
                        const int qmin = weight.w8a8InputDynamic &&
                                         weight.w8a8InputSymmetric ? -127 : -128;
                        value = std::max(qmin, std::min(127, value));
                        const int weightValue = (int)q[(size_t)c * in + p];
                        dot += value * weightValue;
                        weightSum += weightValue;
                    }
                    int32_t corrected = dot - zeroPoint * weightSum;
                    float ref = (float)corrected * tokenScale * weight.scales[c] +
                                (b ? b[c] : 0.0f);
                    float diff = std::fabs(ref - y[(size_t)r * out + c]);
                    maxDiff = std::max(maxDiff, diff);
                    sumDiff += diff;
                    failures += diff > 0.75f;
                }
            }
            const float meanDiff = sumDiff / std::max(1, batch * out);
            std::cout << "linear_int8_w8a8 check max_abs_diff=" << maxDiff
                      << " mean_abs_diff=" << meanDiff
                      << " failed_elements=" << failures
                      << " activation_scheme="
                      << activationScheme
                      << "\n";
            if (!std::isfinite(maxDiff) || failures != 0)
                throw std::runtime_error("SM90 INT8 W8A8 output mismatch");
        }
    };

    static BenchmarkResult BenchmarkLinearInt8W8A8(
        const OpTestParams &params, const std::string &device, int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        LinearInt8W8A8State state; state.Init(params);
        if (params.GetInt("check")) { state.Check(); return {}; }

        for (int i = 0; i < warmup; ++i) state.Run();
        ForceDeviceSync();

        cudaEvent_t begin = nullptr;
        cudaEvent_t end = nullptr;
        if (cudaEventCreate(&begin) != cudaSuccess ||
            cudaEventCreate(&end) != cudaSuccess) {
            if (begin != nullptr) cudaEventDestroy(begin);
            if (end != nullptr) cudaEventDestroy(end);
            throw std::runtime_error(
                "linear_int8_w8a8 failed to create CUDA timing events");
        }
        if (cudaEventRecord(begin, 0) != cudaSuccess) {
            cudaEventDestroy(begin);
            cudaEventDestroy(end);
            throw std::runtime_error(
                "linear_int8_w8a8 failed to record the start CUDA event");
        }

        for (int i = 0; i < iters; ++i) state.Run();
        if (cudaEventRecord(end, 0) != cudaSuccess ||
            cudaEventSynchronize(end) != cudaSuccess) {
            cudaEventDestroy(begin);
            cudaEventDestroy(end);
            throw std::runtime_error(
                "linear_int8_w8a8 failed to record or synchronize the end CUDA event");
        }

        float elapsedMs = 0.0f;
        if (cudaEventElapsedTime(&elapsedMs, begin, end) != cudaSuccess) {
            cudaEventDestroy(begin);
            cudaEventDestroy(end);
            throw std::runtime_error(
                "linear_int8_w8a8 failed to read CUDA event timing");
        }
        cudaEventDestroy(begin);
        cudaEventDestroy(end);

        BenchmarkResult result;
        result.avgMs = elapsedMs / std::max(iters, 1);
        result.bytesMoved = (double)state.batch * state.in * 2 +
            (double)state.out * state.in + state.out * sizeof(float) +
            (double)state.batch * state.out * 2;
        result.flops = 2.0 * state.batch * state.in * state.out;
        double seconds = result.avgMs / 1000.0;
        if (seconds > 0) result.computeTFlops = result.flops / seconds / 1e12;
        return result;
#else
        (void)params; (void)device; (void)warmup; (void)iters;
        throw std::runtime_error("linear_int8_w8a8 requires CUDA");
#endif
    }

    static OpCase MakeLinearInt8W8A8Case() {
        return {"linear_int8_w8a8", "SM90 symmetric/asymmetric INT8 W8A8 dense",
            [](){ OpTestParams p; p.Add("batch","16","rows"); p.Add("in","4096","K");
                  p.Add("out","4096","N"); p.Add("input_type","bf16","fp16 or bf16");
                  p.Add("has_bias","1","FP32 bias");
                  p.Add("activation_scheme","symmetric","symmetric, symmetric_static, azp_dynamic or azp_static");
                  p.Add("check","0","1 functional, 2 reject weight zero-points, 3 reject static scale, 4 reject symmetric zero-point"); return p; },
            [](const OpTestParams&, const std::string &device) {
#ifdef USE_CUDA
                if (device.rfind("cuda",0) != 0) return false;
                ScopedFirstDevice guard(device); return FastllmCudaRuntimeArch() == 90;
#else
                return false;
#endif
            },
            [](const OpTestParams&, const std::string&) { fastllm::Data x(fastllm::DataType::FLOAT32,{1}); x.Allocate(0.0f); return x; },
            BenchmarkLinearInt8W8A8,
            [](const OpTestParams &p) { return (double)p.GetInt("batch")*p.GetInt("in")*2 +
                (double)p.GetInt("out")*p.GetInt("in") + (double)p.GetInt("batch")*p.GetInt("out")*2; },
            [](const OpTestParams &p) { return 2.0*p.GetInt("batch")*p.GetInt("in")*p.GetInt("out"); }, true};
    }

    struct MergeMoeFp8BenchState {
        int batch = 1;
        int topk = 8;
        int hidden = 2048;
        int inter = 768;
        int experts = 82;
        int block = 128;
        std::string path = "operator";
        std::string weightType = "fp8";
        std::string scaleLayout = "block128";
        float sharedScale = 0.0f;
        fastllm::MoeGateType gateType = fastllm::MoeGateSwiglu;
        fastllm::Data input, index, score, output;
        fastllm::Data w1, w2, w3, curInput, curOutput;
        fastllm::Data referenceW1, referenceOutput;
        std::vector<std::unique_ptr<fastllm::Data>> ownedWeights;
        std::vector<fastllm::Data*> weights;
        std::vector<std::unique_ptr<fastllm::Data>> referenceOwnedWeights;
        std::vector<fastllm::Data*> referenceWeights;
        std::vector<fastllm::Data*> biass;

        static void InitFp8Weight(fastllm::Data &weight, int rows, int cols, int block, float seed) {
            weight.dataType = fastllm::DataType::FP8_E4M3;
            weight.UpdateUnitSize();
            weight.Resize({rows, cols});
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = block;
            weight.blockM = block;
            weight.Allocate(false);
            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            for (uint64_t i = 0; i < weight.GetBytes(); i++) {
                ptr[i] = static_cast<uint8_t>(0x20 + ((i + (uint64_t)(seed * 17.0f)) & 0x1f));
            }
            int scaleRows = (rows + block - 1) / block;
            int scaleCols = (cols + block - 1) / block;
            weight.scales.resize((size_t)scaleRows * scaleCols);
            for (size_t i = 0; i < weight.scales.size(); i++) {
                weight.scales[i] = 0.015f + 0.0001f * (float)((i + (int)seed) % 7);
            }
            weight.ToDevice(fastllm::DataDevice::CUDA);
        }

        /**
         * 构造与vLLM 128x128 scale语义一致的交错FP8测试权重。
         *
         * @param weight 返回CPU初始化后迁移至CUDA的Block128权重。
         * @param rows 逻辑输出行数。
         * @param cols 逻辑输入列数，必须按128对齐。
         * @param block scale块大小，当前必须为128。
         * @param seed 固定测试数据的确定性种子。
         */
        static void InitFp8PackedBlock128Weight(
            fastllm::Data &weight, int rows, int cols, int block, float seed) {
            LinearFp8Block128BenchState::InitPackedWeight(
                weight, rows, cols, block, seed);
            weight.ToDevice(fastllm::DataDevice::CUDA);
        }

        static void InitFp8PerChannelWeight(fastllm::Data &weight, int rows,
                                             int cols, float seed) {
            weight.dataType = fastllm::DataType::FP8_E4M3;
            weight.UpdateUnitSize();
            weight.Resize({rows, cols});
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = 1;
            weight.blockM = cols;
            weight.Allocate(false);
            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            for (uint64_t i = 0; i < weight.GetBytes(); i++)
                ptr[i] = static_cast<uint8_t>(0x20 + ((i + (uint64_t)(seed * 17.0f)) & 0x1f));
            weight.scales.resize(rows);
            for (int i = 0; i < rows; ++i)
                weight.scales[i] = 0.015f + 0.0001f * (float)((i + (int)seed) % 7);
            weight.ToDevice(fastllm::DataDevice::CUDA);
        }

        static void InitFp8TensorwiseWeight(fastllm::Data &weight, int rows,
                                             int cols, float seed) {
            InitFp8PerChannelWeight(weight, rows, cols, seed);
            weight.scales.resize(1);
            weight.scales[0] = 0.015f +
                0.0001f * static_cast<float>(static_cast<int>(seed) % 7);
        }

        static void InitNvfp4Weight(fastllm::Data &weight, int rows, int cols, int seed) {
            weight.dataType = fastllm::DataType::NVFP4_BLOCK_16;
            weight.blockM = 16;
            weight.blockK = 1;
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.UpdateUnitSize();
            weight.Resize({rows, cols});
            weight.Allocate(false);
            const size_t rowBytes = fastllm::GetDataBytes(
                fastllm::DataType::NVFP4_BLOCK_16, 1, cols);
            weight.nvfp4GlobalScale = 0.01f;
            weight.nvfp4GroupScales.resize((size_t)rows * (cols / 16));
            for (int row = 0; row < rows; ++row) {
                for (int group = 0; group < cols / 16; ++group) {
                    uint8_t *dst = weight.cpuData + (size_t)row * rowBytes +
                                   (size_t)group * (8 + sizeof(float));
                    for (int pair = 0; pair < 8; ++pair) {
                        uint8_t low = uint8_t(1 + (row + group * 3 + pair + seed) % 7);
                        uint8_t high = uint8_t(1 + (row * 3 + group + pair * 5 + seed) % 7);
                        if ((row + pair + seed) & 1) low |= 8;
                        if ((group + pair + seed) & 1) high |= 8;
                        dst[pair] = uint8_t(low | (high << 4));
                    }
                    const uint8_t scaleByte = uint8_t(0x28 + ((row + group + seed) % 16));
                    weight.nvfp4GroupScales[(size_t)row * (cols / 16) + group] = scaleByte;
                    float scale = DecodeFp8E4M3(scaleByte) * weight.nvfp4GlobalScale;
                    std::memcpy(dst + 8, &scale, sizeof(scale));
                }
            }
            weight.ToDevice(fastllm::DataDevice::CUDA);
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            batch = params.GetInt("batch");
            topk = params.GetInt("topk");
            hidden = params.GetInt("hidden");
            inter = params.GetInt("inter");
            experts = params.GetInt("experts");
            block = params.GetInt("block");
            path = params.GetString("path");
            weightType = params.GetString("weight_type");
            scaleLayout = params.GetString("scale_layout");
            sharedScale = params.GetInt("shared_expert") ? 0.5f : 0.0f;
            const std::string gateTypeName = params.GetString("gate_type");
            if (gateTypeName == "swiglu") {
                gateType = fastllm::MoeGateSwiglu;
            } else if (gateTypeName == "geglu") {
                gateType = fastllm::MoeGateGeglu;
            } else {
                throw std::runtime_error("gate_type must be swiglu or geglu");
            }
            FastllmCudaSetDevice(0);

            fastllm::Data fp32Input = MakeTensor({batch, hidden}, 0.11f, 0.02f);
            input.CopyFrom(fp32Input);
            std::string inputType = params.GetString("input_type");
            if (inputType == "fp16") {
                fastllm::ToDataType(input, fastllm::DataType::FLOAT16);
            } else if (inputType == "bf16") {
                fastllm::ToDataType(input, fastllm::DataType::BFLOAT16);
            } else {
                throw std::runtime_error("input_type must be fp16 or bf16");
            }
            input.ToDevice(fastllm::DataDevice::CUDA);

            index.dataType = fastllm::DataType::INT32;
            index.UpdateUnitSize();
            index.Resize({batch, topk});
            index.Allocate(false);
            score.dataType = fastllm::DataType::FLOAT32;
            score.UpdateUnitSize();
            score.Resize({batch, topk});
            score.Allocate(false);
            int32_t *indexPtr = reinterpret_cast<int32_t*>(index.cpuData);
            float *scorePtr = reinterpret_cast<float*>(score.cpuData);
            for (int b = 0; b < batch; b++) {
                for (int j = 0; j < topk; j++) {
                    indexPtr[b * topk + j] = (b * topk + j) % experts;
                    scorePtr[b * topk + j] = 1.0f / std::max(topk, 1);
                }
            }
            index.ToDevice(fastllm::DataDevice::CUDA);
            score.ToDevice(fastllm::DataDevice::CUDA);

            ownedWeights.resize((size_t)(experts + 1) * 2);
            weights.assign((size_t)(experts + 1) * 2, nullptr);
            biass.assign((size_t)(experts + 1) * 2, nullptr);
            if (sharedScale != 0.0f) {
                ownedWeights[0] = std::make_unique<fastllm::Data>();
                ownedWeights[1] = std::make_unique<fastllm::Data>();
                if (weightType == "nvfp4") {
                    InitNvfp4Weight(*ownedWeights[0], inter * 2, hidden, 101);
                    InitNvfp4Weight(*ownedWeights[1], hidden, inter, 117);
                } else if (scaleLayout == "perchannel") {
                    InitFp8PerChannelWeight(
                        *ownedWeights[0], inter * 2, hidden, 101.0f);
                    InitFp8PerChannelWeight(
                        *ownedWeights[1], hidden, inter, 117.0f);
                } else if (scaleLayout == "tensorwise") {
                    InitFp8TensorwiseWeight(
                        *ownedWeights[0], inter * 2, hidden, 101.0f);
                    InitFp8TensorwiseWeight(
                        *ownedWeights[1], hidden, inter, 117.0f);
                } else if (scaleLayout == "block128") {
                    InitFp8PackedBlock128Weight(
                        *ownedWeights[0], inter * 2, hidden, block, 101.0f);
                    InitFp8PackedBlock128Weight(
                        *ownedWeights[1], hidden, inter, block, 117.0f);
                } else {
                    throw std::runtime_error(
                        "scale_layout must be block128, perchannel or tensorwise");
                }
                weights[0] = ownedWeights[0].get();
                weights[1] = ownedWeights[1].get();
            }
            for (int e = 0; e < experts; e++) {
                int idx = (e + 1) * 2;
                ownedWeights[idx] = std::make_unique<fastllm::Data>();
                ownedWeights[idx + 1] = std::make_unique<fastllm::Data>();
                if (weightType == "nvfp4") {
                    InitNvfp4Weight(*ownedWeights[idx], inter * 2, hidden, e + 3);
                    InitNvfp4Weight(*ownedWeights[idx + 1], hidden, inter, e + 17);
                } else if (weightType == "fp8") {
                    if (scaleLayout == "perchannel") {
                        InitFp8PerChannelWeight(*ownedWeights[idx], inter * 2, hidden, (float)e + 0.3f);
                        InitFp8PerChannelWeight(*ownedWeights[idx + 1], hidden, inter, (float)e + 13.7f);
                    } else if (scaleLayout == "tensorwise") {
                        InitFp8TensorwiseWeight(*ownedWeights[idx], inter * 2, hidden, (float)e + 0.3f);
                        InitFp8TensorwiseWeight(*ownedWeights[idx + 1], hidden, inter, (float)e + 13.7f);
                    } else if (scaleLayout == "block128") {
                        InitFp8PackedBlock128Weight(
                            *ownedWeights[idx], inter * 2, hidden, block,
                            (float)e + 0.3f);
                        InitFp8PackedBlock128Weight(
                            *ownedWeights[idx + 1], hidden, inter, block,
                            (float)e + 13.7f);
                    } else {
                        throw std::runtime_error(
                            "scale_layout must be block128, perchannel or tensorwise");
                    }
                } else {
                    throw std::runtime_error("weight_type must be fp8 or nvfp4");
                }
                weights[idx] = ownedWeights[idx].get();
                weights[idx + 1] = ownedWeights[idx + 1].get();
            }

            // grouped正确性检查使用独立权重对象生成非grouped参考值，避免
            // 参考计算提前改变正式被测权重的Prepared/Cutlass生命周期。
            if (weightType == "nvfp4" && path == "check_nvfp4") {
                referenceOwnedWeights.resize(weights.size());
                referenceWeights.assign(weights.size(), nullptr);
                if (sharedScale != 0.0f) {
                    referenceOwnedWeights[0] = std::make_unique<fastllm::Data>();
                    referenceOwnedWeights[1] = std::make_unique<fastllm::Data>();
                    InitNvfp4Weight(*referenceOwnedWeights[0], inter * 2, hidden, 101);
                    InitNvfp4Weight(*referenceOwnedWeights[1], hidden, inter, 117);
                    referenceWeights[0] = referenceOwnedWeights[0].get();
                    referenceWeights[1] = referenceOwnedWeights[1].get();
                }
                for (int e = 0; e < experts; ++e) {
                    const int idx = (e + 1) * 2;
                    referenceOwnedWeights[idx] = std::make_unique<fastllm::Data>();
                    referenceOwnedWeights[idx + 1] = std::make_unique<fastllm::Data>();
                    InitNvfp4Weight(*referenceOwnedWeights[idx], inter * 2, hidden, e + 3);
                    InitNvfp4Weight(*referenceOwnedWeights[idx + 1], hidden, inter, e + 17);
                    referenceWeights[idx] = referenceOwnedWeights[idx].get();
                    referenceWeights[idx + 1] = referenceOwnedWeights[idx + 1].get();
                }
            }

            output.dataType = input.dataType;
            output.UpdateUnitSize();
            output.Resize({batch, hidden});
            output.ToDevice(fastllm::DataDevice::CUDA);
            output.Allocate(false);
            ForceDeviceSync();
            CheckWarpSpecialization();
#else
            (void)params;
            throw std::runtime_error("mergemoe_fp8 benchmark requires USE_CUDA");
#endif
        }

        bool RunIndexed(fastllm::Data &work, fastllm::Data &result,
                        bool allowWarpSpecialization) {
#ifdef USE_CUDA
            const int32_t *indices = reinterpret_cast<const int32_t*>(index.cudaData);
            const float *scores = reinterpret_cast<const float*>(score.cudaData);
            if (input.dataType == fastllm::DataType::FLOAT16) {
                return FastllmCudaHalfMergeMOEFP8E4M3Batch1Indexed(
                    input, work, result, weights.data(), (int)weights.size(),
                    indices, scores, topk, hidden, inter, allowWarpSpecialization);
            }
            return FastllmCudaBFloat16MergeMOEFP8E4M3Batch1Indexed(
                input, work, result, weights.data(), (int)weights.size(),
                indices, scores, topk, hidden, inter, allowWarpSpecialization);
#else
            (void)work;
            (void)result;
            (void)allowWarpSpecialization;
            return false;
#endif
        }

        void CheckWarpSpecialization() {
#ifdef USE_CUDA
            if (weightType != "fp8" || scaleLayout == "block128" ||
                batch != 1 || topk != 8 || hidden != 2048 ||
                inter != 256 || block != 128) {
                return;
            }
            if (!RunIndexed(referenceW1, referenceOutput, false)) {
                throw std::runtime_error("legacy MergeMOE FP8 launch failed");
            }
            ForceDeviceSync();
            std::vector<float> expected = ToFloatVector(ConvertToFloat32Data(referenceOutput));
            if (!RunIndexed(w1, output, true)) {
                throw std::runtime_error("warp MergeMOE FP8 launch failed");
            }
            ForceDeviceSync();
            std::vector<float> actual = ToFloatVector(ConvertToFloat32Data(output));
            if (expected.size() != actual.size()) {
                throw std::runtime_error("MergeMOE FP8 output size mismatch");
            }
            for (size_t i = 0; i < expected.size(); i++) {
                if (expected[i] != actual[i]) {
                    std::ostringstream os;
                    os << "MergeMOE FP8 warp mismatch at " << i
                       << ": expected=" << expected[i] << ", actual=" << actual[i];
                    throw std::runtime_error(os.str());
                }
            }
#endif
        }

        void Run() {
            if (path == "operator") {
                fastllm::MergeMOE(input, index, score, weights, biass,
                                  w1, w2, w3, curInput, curOutput,
                                  sharedScale, output, 0, gateType);
            } else if (path == "check_nvfp4") {
                if (weightType != "nvfp4") {
                    throw std::runtime_error("check_nvfp4 requires weight_type=nvfp4");
                }
                // 先用正式权重走grouped入口完成首次后端选择；随后关闭MoE
                // grouped，使用数值相同但生命周期独立的权重计算参考值。
                setenv("FASTLLM_CUDA_MOE_NVFP4_W4A4", "1", 1);
                fastllm::MergeMOE(input, index, score, weights, biass,
                                  w1, w2, w3, curInput, curOutput,
                                  sharedScale, output, 0, gateType);
                ForceDeviceSync();
                if (index.dataDevice != fastllm::DataDevice::CUDA ||
                    score.dataDevice != fastllm::DataDevice::CUDA) {
                    throw std::runtime_error(
                        "grouped NVFP4 W4A4 moved routing tensors off CUDA");
                }
                setenv("FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT", "0", 1);
                setenv("FASTLLM_CUDA_MOE_NVFP4_W4A4", "0", 1);
                fastllm::MergeMOE(input, index, score, referenceWeights, biass,
                                  referenceW1, w2, w3, curInput, curOutput,
                                  sharedScale, referenceOutput, 0, gateType);
                ForceDeviceSync();
                setenv("FASTLLM_CUDA_MOE_NVFP4_W4A4", "1", 1);
                setenv("FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT", "1", 1);
                ComparisonStats stats = CompareData(
                    ConvertToFloat32Data(referenceOutput),
                    ConvertToFloat32Data(output), 0.25f, 0.25f);
                if (!stats.passed) {
                    throw std::runtime_error(
                        "grouped NVFP4 W4A4 differs from non-grouped reference");
                }
            } else if (path == "check_nvfp4_lifecycle") {
                if (weightType != "nvfp4") {
                    throw std::runtime_error(
                        "check_nvfp4_lifecycle requires weight_type=nvfp4");
                }
                // 人为让首次grouped选择失败；主流程应整层清cache、恢复原始
                // 权重并固定Rejected，随后即使条件恢复也不得再次尝试CUTLASS。
                setenv("FASTLLM_CUDA_MOE_NVFP4_W4A4", "0", 1);
                fastllm::MergeMOE(input, index, score, weights, biass,
                                  referenceW1, w2, w3, curInput, curOutput,
                                  sharedScale, referenceOutput, 0, gateType);
                ForceDeviceSync();

                const int device = FastllmCudaGetDevice();
                for (size_t i = 2; i < weights.size(); ++i) {
                    if (weights[i] == nullptr || weights[i]->cudaData == nullptr ||
                        FastllmCudaGetNvfp4W4A4BackendState(*weights[i], device) !=
                            FastllmCudaNvfp4BackendState::Rejected) {
                        throw std::runtime_error(
                            "grouped NVFP4 fallback did not restore and reject all routed weights");
                    }
                }

                setenv("FASTLLM_CUDA_MOE_NVFP4_W4A4", "1", 1);
                fastllm::MergeMOE(input, index, score, weights, biass,
                                  w1, w2, w3, curInput, curOutput,
                                  sharedScale, output, 0, gateType);
                ForceDeviceSync();
                if (FastllmCudaGetNvfp4W4A4BackendState(*weights[2], device) !=
                    FastllmCudaNvfp4BackendState::Rejected) {
                    throw std::runtime_error(
                        "grouped NVFP4 fallback retried after backend rejection");
                }
                ComparisonStats stats = CompareData(
                    ConvertToFloat32Data(referenceOutput),
                    ConvertToFloat32Data(output), 0.25f, 0.25f);
                if (!stats.passed) {
                    throw std::runtime_error(
                        "fixed NVFP4 Legacy output changed after retry conditions recovered");
                }
            } else if (path == "check_fp8") {
                if (weightType != "fp8" ||
                    (scaleLayout != "perchannel" &&
                     scaleLayout != "tensorwise") ||
                    (input.dataType != fastllm::DataType::FLOAT16 &&
                     input.dataType != fastllm::DataType::BFLOAT16)) {
                    throw std::runtime_error(
                        "check_fp8 requires fp16/bf16 + fp8 + "
                        "perchannel/tensorwise scales");
                }
                setenv("FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90", "0", 1);
                fastllm::MergeMOE(input, index, score, weights, biass,
                                  referenceW1, w2, w3, curInput, curOutput,
                                  sharedScale, referenceOutput, 0,
                                  fastllm::MoeGateSwiglu);
                ForceDeviceSync();
                setenv("FASTLLM_CUDA_MOE_CUTLASS_FP8_SM90", "1", 1);
                fastllm::MergeMOE(input, index, score, weights, biass,
                                  w1, w2, w3, curInput, curOutput,
                                  sharedScale, output, 0,
                                  fastllm::MoeGateSwiglu);
                ForceDeviceSync();
                ComparisonStats stats = CompareData(
                    ConvertToFloat32Data(referenceOutput),
                    ConvertToFloat32Data(output), 0.25f, 0.25f);
                std::printf(
                    "SM90 FP8 W8A8 grouped MoE correctness: "
                    "max_abs_diff=%.6f mean_abs_diff=%.6f "
                    "failed_elements=%zu result=%s\n",
                    stats.maxAbsDiff, stats.meanAbsDiff,
                    stats.failedElements, stats.passed ? "PASS" : "FAIL");
                if (!stats.passed) {
                    throw std::runtime_error(
                        "SM90 grouped FP8 CUTLASS differs from CUDA fallback");
                }
            } else if (path == "check_block128") {
                if (weightType != "fp8" || scaleLayout != "block128" ||
                    (input.dataType != fastllm::DataType::FLOAT16 &&
                     input.dataType != fastllm::DataType::BFLOAT16)) {
                    throw std::runtime_error(
                        "check_block128 requires fp16/bf16 + packed Block128 FP8");
                }
                setenv("FASTLLM_CUDA_TRITON", "1", 1);
                setenv("FASTLLM_CUDA_TRITON_MERGE_MOE", "1", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90", "1", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT", "1", 1);
                fastllm::MergeMOE(input, index, score, weights, biass,
                                  w1, w2, w3, curInput, curOutput,
                                  sharedScale, output, 0,
                                  fastllm::MoeGateSwiglu);
                ForceDeviceSync();

                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT", "0", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90", "0", 1);
                setenv("FASTLLM_CUDA_TRITON_MERGE_MOE", "0", 1);
                fastllm::MergeMOE(input, index, score, weights, biass,
                                  referenceW1, w2, w3, curInput, curOutput,
                                  sharedScale, referenceOutput, 0,
                                  fastllm::MoeGateSwiglu);
                ForceDeviceSync();
                setenv("FASTLLM_CUDA_TRITON_MERGE_MOE", "1", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90", "1", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT", "1", 1);

                ComparisonStats stats = CompareData(
                    ConvertToFloat32Data(referenceOutput),
                    ConvertToFloat32Data(output), 0.25f, 0.25f);
                std::printf(
                    "SM90 Block128 FP8 W8A8 MoE correctness: "
                    "max_abs_diff=%.6f mean_abs_diff=%.6f "
                    "failed_elements=%zu result=%s\n",
                    stats.maxAbsDiff, stats.meanAbsDiff,
                    stats.failedElements, stats.passed ? "PASS" : "FAIL");
                if (!stats.passed) {
                    throw std::runtime_error(
                        "SM90 Block128 Triton MoE differs from CUDA fallback");
                }
            } else if (path == "legacy") {
                if (!RunIndexed(w1, output, false)) {
                    throw std::runtime_error("legacy MergeMOE FP8 launch failed");
                }
            } else if (path == "warp") {
                if (!RunIndexed(w1, output, true)) {
                    throw std::runtime_error("warp MergeMOE FP8 launch failed");
                }
            } else {
                throw std::runtime_error(
                    "path must be operator, check_fp8, check_block128, check_nvfp4, "
                    "check_nvfp4_lifecycle, legacy or warp");
            }
        }
    };

    static BenchmarkResult BenchmarkMergeMoeFp8Cuda(const OpTestParams &params,
                                                    const std::string &device,
                                                    int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<MergeMoeFp8BenchState>();
        state->Init(params);

        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        // 与正式异步主流程及vLLM基准一致：整段warmup结束后只同步一次。
        ForceDeviceSync();

        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();

        BenchmarkResult result;
        result.avgMs = totalMs / std::max(iters, 1);
        if (result.avgMs <= 0.0) {
            result.avgMs = 0.0;
        }
        result.bytesMoved = 0.0;
        const int expertCalls = params.GetInt("topk") +
                                (params.GetInt("shared_expert") ? 1 : 0);
        result.flops = (double)params.GetInt("batch") * expertCalls *
                       6.0 * (double)params.GetInt("hidden") * params.GetInt("inter");
        double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0 && result.flops > 0.0) {
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("mergemoe_fp8 benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeMergeMoeFp8Case() {
        return {
            "mergemoe_fp8",
            "benchmark/check Qwen3-style FP8 or NVFP4 grouped MergeMOE",
            []() {
                OpTestParams params;
                params.Add("batch", "1", "token batch size");
                params.Add("hidden", "2048", "Qwen3-20B-A3B hidden size");
                params.Add("inter", "768", "Qwen3-20B-A3B MoE intermediate size");
                params.Add("experts", "82", "number of routed experts");
                params.Add("topk", "8", "experts per token");
                params.Add("block", "128", "FP8 scale block size");
                params.Add("input_type", "fp16", "fp16 or bf16");
                params.Add("path", "operator",
                           "operator, check_fp8, check_nvfp4, check_nvfp4_lifecycle, legacy or warp");
                params.Add("weight_type", "fp8", "fp8 or nvfp4");
                params.Add("scale_layout", "block128",
                           "FP8 scale layout: block128, perchannel or tensorwise");
                params.Add("gate_type", "swiglu", "swiglu or geglu");
                params.Add("shared_expert", "0", "1 enables one separately computed shared expert");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                if (device.rfind("cuda", 0) != 0) return false;
#ifdef USE_CUDA
                if (params.GetString("weight_type") == "nvfp4") {
                    int arch = FastllmCudaRuntimeArch();
                    return arch >= 100 && arch < 130 &&
                           (arch < 120 || params.GetString("input_type") == "bf16");
                }
#endif
                return true;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkMergeMoeFp8Cuda,
            [](const OpTestParams&) {
                return 0.0;
            },
            [](const OpTestParams &params) {
                return (double)params.GetInt("batch") * params.GetInt("topk") *
                       6.0 * (double)params.GetInt("hidden") * params.GetInt("inter");
            },
            true
        };
    }

#ifdef USE_CUDA
    struct FusedMoeFp8BenchState {
        int batch = 1;
        int topk = 8;
        int hidden = 2048;
        int inter = 256;
        int experts = 16;
        int block = 128;
        std::string path = "warp";
        std::string scaleLayout = "separate";
        fastllm::Data input, index, score;
        fastllm::Data gate, up, down;
        fastllm::Data w1, output, referenceW1, referenceOutput;

        static void InitWeight(fastllm::Data &weight, const std::vector<int> &dims,
                               int block, int seed, const std::string &scaleLayout) {
            if (scaleLayout == "block128") {
                LinearFp8Block128BenchState::InitPackedWeight(
                    weight, dims[0] * dims[1], dims[2], block, (float)seed);
                weight.Resize(dims);
                weight.ToDevice(fastllm::DataDevice::CUDA);
                return;
            }
            weight.dataType = fastllm::DataType::FP8_E4M3;
            weight.UpdateUnitSize();
            weight.Resize(dims);
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = block;
            weight.blockM = block;
            weight.Allocate(false);
            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            for (uint64_t i = 0; i < weight.GetBytes(); i++) {
                ptr[i] = static_cast<uint8_t>(0x20 + ((i + (uint64_t)seed * 17) & 0x1f));
            }
            int scaleRows = (dims[dims.size() - 2] + block - 1) / block;
            int scaleCols = (dims.back() + block - 1) / block;
            weight.scales.resize((size_t)dims[0] * scaleRows * scaleCols);
            for (size_t i = 0; i < weight.scales.size(); i++) {
                weight.scales[i] = 0.015f + 0.0001f * (float)((i + seed) % 7);
            }
            weight.ToDevice(fastllm::DataDevice::CUDA);
            fastllm::Data emptyBias;
            FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, emptyBias, 1);
        }

        bool RunDirect(fastllm::Data &work, fastllm::Data &result,
                       bool allowWarpSpecialization) {
#ifdef USE_CUDA
            if (scaleLayout == "block128") {
                return input.dataType == fastllm::DataType::FLOAT16 ?
                    FastllmCudaHalfFusedMOEFP8E4M3Block128(
                        input, gate, up, down, index, score, work, result,
                        batch, topk, hidden, inter, experts, 0.0f) :
                    FastllmCudaBFloat16FusedMOEFP8E4M3Block128(
                        input, gate, up, down, index, score, work, result,
                        batch, topk, hidden, inter, experts, 0.0f);
            }
            if (input.dataType == fastllm::DataType::FLOAT16) {
                return FastllmCudaHalfFusedMOEFP8E4M3(
                    input, gate, up, down, index, score, work, result,
                    batch, topk, hidden, inter, experts, 0.0f,
                    allowWarpSpecialization);
            }
            return FastllmCudaBFloat16FusedMOEFP8E4M3(
                input, gate, up, down, index, score, work, result,
                batch, topk, hidden, inter, experts, 0.0f,
                allowWarpSpecialization);
#else
            (void)work;
            (void)result;
            (void)allowWarpSpecialization;
            return false;
#endif
        }

        void CheckWarpSpecialization() {
#ifdef USE_CUDA
            if (scaleLayout == "block128" || batch != 1 || topk != 8 ||
                hidden != 2048 || inter != 256 || block != 128) {
                return;
            }
            if (!RunDirect(referenceW1, referenceOutput, false)) {
                throw std::runtime_error("legacy FusedMOE FP8 launch failed");
            }
            ForceDeviceSync();
            std::vector<float> expected = ToFloatVector(ConvertToFloat32Data(referenceOutput));
            if (!RunDirect(w1, output, true)) {
                throw std::runtime_error("warp FusedMOE FP8 launch failed");
            }
            ForceDeviceSync();
            std::vector<float> actual = ToFloatVector(ConvertToFloat32Data(output));
            if (expected.size() != actual.size()) {
                throw std::runtime_error("FusedMOE FP8 output size mismatch");
            }
            for (size_t i = 0; i < expected.size(); i++) {
                if (expected[i] != actual[i]) {
                    std::ostringstream os;
                    os << "FusedMOE FP8 warp mismatch at " << i
                       << ": expected=" << expected[i] << ", actual=" << actual[i];
                    throw std::runtime_error(os.str());
                }
            }
#endif
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            batch = params.GetInt("batch");
            topk = params.GetInt("topk");
            hidden = params.GetInt("hidden");
            inter = params.GetInt("inter");
            experts = params.GetInt("experts");
            block = params.GetInt("block");
            path = params.GetString("path");
            scaleLayout = params.GetString("scale_layout");
            FastllmCudaSetDevice(0);

            fastllm::Data fp32Input = MakeTensor({batch, hidden}, 0.11f, 0.02f);
            input.CopyFrom(fp32Input);
            std::string inputType = params.GetString("input_type");
            if (inputType == "fp16") {
                fastllm::ToDataType(input, fastllm::DataType::FLOAT16);
            } else if (inputType == "bf16") {
                fastllm::ToDataType(input, fastllm::DataType::BFLOAT16);
            } else {
                throw std::runtime_error("input_type must be fp16 or bf16");
            }
            input.ToDevice(fastllm::DataDevice::CUDA);

            index.dataType = fastllm::DataType::INT32;
            index.UpdateUnitSize();
            index.Resize({batch, topk});
            index.Allocate(false);
            score.dataType = fastllm::DataType::FLOAT32;
            score.UpdateUnitSize();
            score.Resize({batch, topk});
            score.Allocate(false);
            int32_t *indexPtr = reinterpret_cast<int32_t*>(index.cpuData);
            float *scorePtr = reinterpret_cast<float*>(score.cpuData);
            for (int b = 0; b < batch; b++) {
                for (int j = 0; j < topk; j++) {
                    indexPtr[b * topk + j] = (b * topk + j) % experts;
                    scorePtr[b * topk + j] = 1.0f / std::max(topk, 1);
                }
            }
            index.ToDevice(fastllm::DataDevice::CUDA);
            score.ToDevice(fastllm::DataDevice::CUDA);

            InitWeight(gate, {experts, inter, hidden}, block, 3, scaleLayout);
            InitWeight(up, {experts, inter, hidden}, block, 17, scaleLayout);
            InitWeight(down, {experts, hidden, inter}, block, 29, scaleLayout);
            ForceDeviceSync();
            CheckWarpSpecialization();
#else
            (void)params;
            throw std::runtime_error("fusedmoe_fp8 benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (path == "operator") {
                fastllm::FusedMOE(input, index, score, gate, up, down, w1,
                                  output, 0, fastllm::MoeGateSwiglu, 0.0f);
            } else if (path == "check_block128") {
                if (scaleLayout != "block128") {
                    throw std::runtime_error(
                        "check_block128 requires packed Block128 FP8 weights");
                }
                setenv("FASTLLM_CUDA_TRITON", "1", 1);
                setenv("FASTLLM_CUDA_TRITON_MERGE_MOE", "1", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90", "1", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT", "1", 1);
                fastllm::FusedMOE(input, index, score, gate, up, down, w1,
                                  output, 0, fastllm::MoeGateSwiglu, 0.0f);
                ForceDeviceSync();

                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT", "0", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90", "0", 1);
                setenv("FASTLLM_CUDA_TRITON_MERGE_MOE", "0", 1);
                fastllm::FusedMOE(
                    input, index, score, gate, up, down, referenceW1,
                    referenceOutput, 0, fastllm::MoeGateSwiglu, 0.0f);
                ForceDeviceSync();
                setenv("FASTLLM_CUDA_TRITON_MERGE_MOE", "1", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90", "1", 1);
                setenv("FASTLLM_CUDA_MOE_BLOCK128_SM90_STRICT", "1", 1);

                ComparisonStats stats = CompareData(
                    ConvertToFloat32Data(referenceOutput),
                    ConvertToFloat32Data(output), 0.25f, 0.25f);
                std::printf(
                    "SM90 Block128 FP8 W8A8 FusedMOE correctness: "
                    "max_abs_diff=%.6f mean_abs_diff=%.6f "
                    "failed_elements=%zu result=%s\n",
                    stats.maxAbsDiff, stats.meanAbsDiff,
                    stats.failedElements, stats.passed ? "PASS" : "FAIL");
                if (!stats.passed) {
                    throw std::runtime_error(
                        "SM90 Block128 Triton FusedMOE differs from CUDA fallback");
                }
            } else if (path == "legacy") {
                if (!RunDirect(w1, output, false)) {
                    throw std::runtime_error("legacy FusedMOE FP8 launch failed");
                }
            } else if (path == "warp") {
                if (!RunDirect(w1, output, true)) {
                    throw std::runtime_error("warp FusedMOE FP8 launch failed");
                }
            } else {
                throw std::runtime_error(
                    "path must be operator, check_block128, legacy or warp");
            }
        }
    };
#endif

    static BenchmarkResult BenchmarkFusedMoeFp8Cuda(
            const OpTestParams &params, const std::string &device, int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<FusedMoeFp8BenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        result.flops = (double)params.GetInt("batch") * params.GetInt("topk") *
                       6.0 * (double)params.GetInt("hidden") * params.GetInt("inter");
        double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0) {
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("fusedmoe_fp8 benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeFusedMoeFp8Case() {
        return {
            "fusedmoe_fp8",
            "benchmark and validate Qwen3.5 fused-weight FP8 MoE",
            []() {
                OpTestParams params;
                params.Add("batch", "1", "token batch size");
                params.Add("hidden", "2048", "Qwen3.5 hidden size");
                params.Add("inter", "256", "TP-local MoE intermediate size");
                params.Add("experts", "16", "number of routed experts in the test table");
                params.Add("topk", "8", "experts per token");
                params.Add("block", "128", "FP8 scale block size");
                params.Add("input_type", "fp16", "fp16 or bf16");
                params.Add("path", "warp",
                           "operator, check_block128, legacy or warp");
                params.Add("scale_layout", "separate",
                           "separate or block128");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkFusedMoeFp8Cuda,
            [](const OpTestParams&) { return 0.0; },
            [](const OpTestParams &params) {
                return (double)params.GetInt("batch") * params.GetInt("topk") *
                       6.0 * (double)params.GetInt("hidden") * params.GetInt("inter");
            },
            true
        };
    }

#ifdef USE_CUDA
    struct RouterLinearFp16BenchState {
        int batch = 8;
        int inputDim = 5120;
        int outputDim = 48;
        bool withBias = false;
        std::string path;
        FastllmCudaLinearFp16Path selectedPath =
            FASTLLM_CUDA_LINEAR_FP16_PATH_AUTO;
        FastllmCudaLinearFp16Path resolvedAutoPath =
            FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS;
        fastllm::Data input, weight, bias;
        fastllm::Data autoOutput, nativeOutput, cublasOutput;

        void PrepareOutput(fastllm::Data &data) const {
            data.dataType = fastllm::DataType::FLOAT16;
            data.UpdateUnitSize();
            data.Resize({batch, outputDim});
            data.Allocate(false);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        static FastllmCudaLinearFp16Path ParsePath(
                const std::string &value) {
            if (value == "auto" || value == "fast") {
                return FASTLLM_CUDA_LINEAR_FP16_PATH_AUTO;
            }
            if (value == "native") {
                return FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE;
            }
            if (value == "cublas" || value == "legacy") {
                return FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS;
            }
            throw std::runtime_error(
                "path must be auto, native, or cublas");
        }

        static void CheckAutoPath(
                int n, int m, int k, bool addTo, bool hasBias,
                FastllmCudaLinearFp16Path expected) {
            if (FastllmCudaResolveLinearFp16AutoPath(
                    n, m, k, addTo, hasBias) != expected) {
                throw std::runtime_error(
                    "FP16 AUTO dispatch policy regression");
            }
        }

        static void CheckAutoDispatchPolicy() {
            CheckAutoPath(7, 2048, 256, false, false,
                          FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE);
            CheckAutoPath(8, 5120, 48, false, false,
                          FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE);
            CheckAutoPath(8, 5120, 48, true, false,
                          FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS);
            CheckAutoPath(8, 5120, 48, false, true,
                          FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS);
            CheckAutoPath(8, 2048, 256, false, false,
                          FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS);
            CheckAutoPath(9, 5120, 48, false, false,
                          FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS);
        }

        void RunOne(fastllm::Data &output,
                    FastllmCudaLinearFp16Path selected) {
            bool ok = FastllmCudaHalfMatMulFloat16WithPath(
                input, weight, bias, output,
                batch, inputDim, outputDim, false,
                true, selected);
            if (!ok) {
                throw std::runtime_error(
                    "FP16 small-batch linear path rejected the test shape");
            }
        }

        void Check() {
            RunOne(cublasOutput, FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS);
            RunOne(autoOutput, FASTLLM_CUDA_LINEAR_FP16_PATH_AUTO);
            if (batch <= 8) {
                RunOne(nativeOutput, FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE);
            }
            ForceDeviceSync();

            const fastllm::Data &expectedAutoOutput =
                resolvedAutoPath == FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE
                    ? nativeOutput
                    : cublasOutput;
            std::vector<float> autoValues =
                ToFloatVector(ConvertToFloat32Data(autoOutput));
            std::vector<float> expectedAutoValues =
                ToFloatVector(ConvertToFloat32Data(expectedAutoOutput));
            if (autoValues != expectedAutoValues) {
                throw std::runtime_error(
                    "FP16 AUTO output does not match its resolved path");
            }

            // Native and cuBLAS do not use the same accumulation order. Check
            // every selectable path independently against an FP64 host
            // reference instead of comparing two aliases of AUTO.
            int64_t referenceOps =
                (int64_t)batch * inputDim * outputDim;
            if (referenceOps <= 100LL * 1024 * 1024) {
                std::vector<float> inputValues =
                    ToFloatVector(ConvertToFloat32Data(input));
                std::vector<float> weightValues =
                    ToFloatVector(ConvertToFloat32Data(weight));
                std::vector<float> biasValues;
                if (withBias) {
                    biasValues = ToFloatVector(ConvertToFloat32Data(bias));
                }
                std::vector<double> reference(
                    (size_t)batch * outputDim, 0.0);
                for (int row = 0; row < batch; ++row) {
                    for (int column = 0; column < outputDim; ++column) {
                        double value =
                            withBias ? (double)biasValues[column] : 0.0;
                        for (int inner = 0; inner < inputDim; ++inner) {
                            value +=
                                (double)inputValues[(size_t)row * inputDim + inner] *
                                (double)weightValues[(size_t)column * inputDim + inner];
                        }
                        reference[(size_t)row * outputDim + column] = value;
                    }
                }

                auto validate = [&](const fastllm::Data &output,
                                    const char *label) {
                    std::vector<float> actual =
                        ToFloatVector(ConvertToFloat32Data(output));
                    for (size_t index = 0; index < actual.size(); ++index) {
                        // Both paths accumulate in FP16 but use different
                        // reduction trees. Scale the bound with the reduction
                        // length rather than the number of output columns.
                        float relativeTolerance =
                            inputDim >= 4096 ? 5.0e-3f : 2.0e-3f;
                        float absoluteTolerance =
                            inputDim >= 4096 ? 0.25f : 0.125f;
                        float tolerance =
                            absoluteTolerance + relativeTolerance *
                                                    std::fabs(
                                                        (float)reference[index]);
                        if (!std::isfinite(actual[index]) ||
                            std::fabs(actual[index] -
                                      (float)reference[index]) >
                                tolerance) {
                            std::ostringstream os;
                            os << "FP16 " << label << " path mismatch at "
                               << index << ": reference=" << reference[index]
                               << " actual=" << actual[index]
                               << " tolerance=" << tolerance;
                            throw std::runtime_error(os.str());
                        }
                    }
                };
                validate(cublasOutput, "cublas");
                validate(autoOutput, "auto");
                if (batch <= 8) {
                    validate(nativeOutput, "native");
                }
            }
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            FastllmCudaSetDevice(0);
            CheckAutoDispatchPolicy();
            batch = params.GetInt("batch");
            inputDim = params.GetInt("in");
            outputDim = params.GetInt("out");
            withBias = params.GetInt("bias") != 0;
            if (batch <= 0 || inputDim <= 0 || outputDim <= 0) {
                throw std::runtime_error(
                    "batch, in and out must all be positive");
            }
            path = params.GetString("path");
            selectedPath = ParsePath(path);
            resolvedAutoPath = FastllmCudaResolveLinearFp16AutoPath(
                batch, inputDim, outputDim, false, withBias);
            const bool qwen36GdnB8 =
                batch == 8 && inputDim == 5120 && outputDim == 48 &&
                !withBias;
            FastllmCudaLinearFp16Path expectedAutoPath =
                (batch >= 1 && batch < 8) || qwen36GdnB8
                    ? FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE
                    : FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS;
            if (resolvedAutoPath != expectedAutoPath) {
                throw std::runtime_error(
                    "FP16 AUTO resolved an unexpected production path");
            }
            if (selectedPath == FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE &&
                batch > 8) {
                throw std::runtime_error(
                    "native FP16 path supports batch 1 through 8");
            }
            input.CopyFrom(MakeTensor({batch, inputDim}, 0.271f, 0.25f));
            fastllm::ToDataType(input, fastllm::DataType::FLOAT16);
            input.ToDevice(fastllm::DataDevice::CUDA);
            weight.CopyFrom(
                MakeTensor({outputDim, inputDim}, 0.619f, 0.125f));
            fastllm::ToDataType(weight, fastllm::DataType::FLOAT16);
            weight.ToDevice(fastllm::DataDevice::CUDA);
            if (withBias) {
                bias.CopyFrom(MakeTensor({outputDim}, 0.887f, 0.03125f));
                bias.ToDevice(fastllm::DataDevice::CUDA);
            }
            PrepareOutput(autoOutput);
            PrepareOutput(nativeOutput);
            PrepareOutput(cublasOutput);
            Check();
#else
            (void)params;
            throw std::runtime_error("router_linear_fp16 benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (selectedPath == FASTLLM_CUDA_LINEAR_FP16_PATH_AUTO) {
                RunOne(autoOutput, selectedPath);
            } else if (selectedPath ==
                       FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE) {
                RunOne(nativeOutput, selectedPath);
            } else {
                RunOne(cublasOutput, selectedPath);
            }
        }
    };
#endif
    static BenchmarkResult BenchmarkRouterLinearFp16Cuda(
            const OpTestParams &params, const std::string &device, int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<RouterLinearFp16BenchState>();
        state->Init(params);
        bool graphMode = params.GetInt("graph") != 0;
        void *graph = nullptr;
        void *graphExec = nullptr;
        if (graphMode) {
            if (!FastllmCudaGraphBeginCapture()) {
                throw std::runtime_error(
                    "FP16 small-batch linear graph capture did not start");
            }
            state->Run();
            if (!FastllmCudaGraphEndCapture(&graph) || graph == nullptr ||
                !FastllmCudaGraphInstantiate(graph, &graphExec) ||
                graphExec == nullptr) {
                if (graph != nullptr) {
                    FastllmCudaGraphDestroy(graph);
                }
                throw std::runtime_error(
                    "FP16 small-batch linear graph creation failed");
            }
        }
        auto run = [&]() {
            if (graphMode) {
                if (!FastllmCudaGraphLaunch(graphExec)) {
                    throw std::runtime_error(
                        "FP16 small-batch linear graph replay failed");
                }
            } else {
                state->Run();
            }
        };
        for (int i = 0; i < warmup; i++) {
            run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        if (graphExec != nullptr) {
            FastllmCudaGraphExecDestroy(graphExec);
        }
        if (graph != nullptr) {
            FastllmCudaGraphDestroy(graph);
        }
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("router_linear_fp16 benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeRouterLinearFp16Case() {
        return {
            "router_linear_fp16",
            "benchmark and validate Qwen3.6 B8 GDN FP16 linear",
            []() {
                OpTestParams params;
                params.Add("path", "auto", "auto, native, or cublas");
                params.Add("batch", "8", "token batch size");
                params.Add("in", "5120", "input features");
                params.Add("out", "48", "output features");
                params.Add("bias", "0", "use an FP32 bias (0 or 1)");
                params.Add("graph", "0", "benchmark CUDA Graph replay (0 or 1)");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkRouterLinearFp16Cuda,
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch");
                int in = params.GetInt("in");
                int out = params.GetInt("out");
                return ((double)batch * in + (double)out * in +
                        (double)batch * out) * 2.0;
            },
            [](const OpTestParams &params) {
                return 2.0 * params.GetInt("batch") * params.GetInt("in") *
                       params.GetInt("out");
            },
            true
        };
    }

#ifdef USE_CUDA
    struct FusedRouterTopKBenchState {
        int batch = 1;
        int topk = 8;
        bool withBias = true;
        bool needNorm = true;
        bool sigmoid = false;
        float routeScale = 1.0f;
        std::string path;
        fastllm::Data logits, bias;
        fastllm::Data fusedIndex, fusedScore;
        fastllm::Data referenceLogits, referenceProb, referenceBias;
        fastllm::Data referenceIndex, referenceScore;

        static void PrepareOutput(fastllm::Data &data, fastllm::DataType type,
                                  const std::vector<int> &dims) {
            data.dataType = type;
            data.UpdateUnitSize();
            data.Resize(dims);
            data.Allocate(false);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        void RunReference() {
            fastllm::ToDataType(logits, referenceLogits, fastllm::DataType::FLOAT32);
            if (sigmoid) {
                fastllm::Sigmoid(referenceLogits, referenceProb);
            } else {
                fastllm::Softmax(referenceLogits, referenceProb, -1);
            }
            fastllm::SelectExpert(referenceProb, referenceIndex, referenceScore, topk,
                                  needNorm, routeScale,
                                  withBias ? &referenceBias : nullptr);
        }

        void RunFused() {
            bool ok = sigmoid
                ? FastllmCudaFusedSigmoidSelectExpert(
                    logits, withBias ? &bias : nullptr, fusedIndex, fusedScore,
                    topk, needNorm, routeScale)
                : FastllmCudaFusedSoftmaxSelectExpert(
                    logits, withBias ? &bias : nullptr, fusedIndex, fusedScore,
                    topk, needNorm, routeScale);
            if (!ok) {
                throw std::runtime_error("fused router top-k launch failed");
            }
        }

        void Check() {
            RunReference();
            RunFused();
            ForceDeviceSync();
            std::vector<int32_t> expectedIndex = ToInt32Vector(referenceIndex);
            std::vector<int32_t> actualIndex = ToInt32Vector(fusedIndex);
            if (expectedIndex != actualIndex) {
                std::ostringstream os;
                os << "fused router index mismatch:";
                size_t mismatch = 0;
                for (size_t i = 0; i < expectedIndex.size(); i++) {
                    if (expectedIndex[i] != actualIndex[i]) {
                        os << " pos=" << i << " expected=" << expectedIndex[i]
                           << " actual=" << actualIndex[i];
                        mismatch = i;
                        break;
                    }
                }
                size_t tokenStart = (mismatch / topk) * topk;
                os << " expected_topk=";
                for (size_t i = tokenStart; i < tokenStart + topk; i++) {
                    os << (i == tokenStart ? "[" : ",") << expectedIndex[i];
                }
                os << "] actual_topk=";
                for (size_t i = tokenStart; i < tokenStart + topk; i++) {
                    os << (i == tokenStart ? "[" : ",") << actualIndex[i];
                }
                os << "] expected_prob=";
                std::vector<float> referenceProbValues = ToFloatVector(referenceProb);
                size_t probStart = (mismatch / topk) * 256;
                for (size_t i = tokenStart; i < tokenStart + topk; i++) {
                    os << (i == tokenStart ? "[" : ",")
                       << referenceProbValues[probStart + expectedIndex[i]];
                }
                os << "]";
                throw std::runtime_error(os.str());
            }
            std::vector<float> expectedScore = ToFloatVector(referenceScore);
            std::vector<float> actualScore = ToFloatVector(fusedScore);
            float maxAbsDiff = 0.0f;
            for (size_t i = 0; i < expectedScore.size(); i++) {
                maxAbsDiff = std::max(maxAbsDiff, std::fabs(expectedScore[i] - actualScore[i]));
            }
            const bool selectedLogitFastPath =
                !sigmoid && !withBias && needNorm;
            const float scoreTolerance = sigmoid ? 1.0e-6f :
                (selectedLogitFastPath ? 2.0e-6f : 0.0f);
            if (maxAbsDiff > scoreTolerance) {
                std::ostringstream os;
                os << "fused router score mismatch: max_abs_diff=" << maxAbsDiff;
                throw std::runtime_error(os.str());
            }
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            FastllmCudaSetDevice(0);
            batch = params.GetInt("batch");
            sigmoid = params.GetString("activation") == "sigmoid";
            topk = sigmoid ? 10 : 8;
            withBias = params.GetInt("bias") != 0;
            needNorm = params.GetInt("norm") != 0;
            routeScale = params.GetFloat("route_scale");
            path = params.GetString("path");

            std::string pattern = params.GetString("pattern");
            fastllm::Data fp32Logits;
            if (pattern == "unique") {
                fastllm::Data generated = MakeTensor({batch, 256}, 0.731f, 1.7f);
                fp32Logits.CopyFrom(generated);
            } else if (pattern == "tied") {
                std::vector<float> values((size_t)batch * 256);
                for (int token = 0; token < batch; token++) {
                    for (int expert = 0; expert < 256; expert++) {
                        values[(size_t)token * 256 + expert] =
                            (float)((expert * 17 + token * 13) & 7) * 0.25f;
                    }
                }
                fastllm::Data generated(fastllm::DataType::FLOAT32,
                                        {batch, 256}, values);
                fp32Logits.CopyFrom(generated);
            } else {
                throw std::runtime_error("pattern must be unique or tied");
            }
            logits.CopyFrom(fp32Logits);
            std::string inputType = params.GetString("input_type");
            if (inputType == "fp16") {
                fastllm::ToDataType(logits, fastllm::DataType::FLOAT16);
            } else if (inputType == "bf16") {
                fastllm::ToDataType(logits, fastllm::DataType::BFLOAT16);
            } else if (inputType != "fp32") {
                throw std::runtime_error("input_type must be fp16, bf16 or fp32");
            }
            logits.ToDevice(fastllm::DataDevice::CUDA);

            fastllm::Data fp32Bias;
            if (pattern == "unique") {
                fastllm::Data generated = MakeTensor({256}, 1.217f, 0.015f);
                fp32Bias.CopyFrom(generated);
            } else {
                std::vector<float> values(256);
                for (int expert = 0; expert < 256; expert++) {
                    values[expert] = (float)(expert & 3) * 0.001f;
                }
                fastllm::Data generated(fastllm::DataType::FLOAT32,
                                        {256}, values);
                fp32Bias.CopyFrom(generated);
            }
            bias.CopyFrom(fp32Bias);
            std::string biasType = params.GetString("bias_type");
            if (biasType == "fp16") {
                fastllm::ToDataType(bias, fastllm::DataType::FLOAT16);
            } else if (biasType == "bf16") {
                fastllm::ToDataType(bias, fastllm::DataType::BFLOAT16);
            } else if (biasType != "fp32") {
                throw std::runtime_error("bias_type must be fp16, bf16 or fp32");
            }
            referenceBias.CopyFrom(bias);
            fastllm::ToDataType(referenceBias, fastllm::DataType::FLOAT32);
            referenceBias.ToDevice(fastllm::DataDevice::CUDA);
            bias.ToDevice(fastllm::DataDevice::CUDA);
            PrepareOutput(fusedIndex, fastllm::DataType::INT32, {batch, topk});
            PrepareOutput(fusedScore, fastllm::DataType::FLOAT32, {batch, topk});
            Check();
#else
            (void)params;
            throw std::runtime_error("fused_router_topk benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (path == "fused") {
                RunFused();
            } else if (path == "reference") {
                RunReference();
            } else {
                throw std::runtime_error("path must be fused or reference");
            }
        }
    };
#endif

    static BenchmarkResult BenchmarkFusedRouterTopKCuda(
            const OpTestParams &params, const std::string &device, int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<FusedRouterTopKBenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("fused_router_topk benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeFusedRouterTopKCase() {
        return {
            "fused_router_topk",
            "benchmark and validate fused softmax/sigmoid + SelectExpert for 256 experts",
            []() {
                OpTestParams params;
                params.Add("batch", "1", "token batch size");
                params.Add("input_type", "fp16", "fp16, bf16 or fp32");
                params.Add("bias_type", "fp32", "fp16, bf16 or fp32");
                params.Add("activation", "softmax", "softmax/top8 or sigmoid/top10");
                params.Add("bias", "1", "whether correction bias is present");
                params.Add("norm", "1", "normalize selected probabilities");
                params.Add("route_scale", "1.0", "router score scale");
                params.Add("path", "fused", "fused or reference");
                params.Add("pattern", "unique", "unique or tied ranking keys");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkFusedRouterTopKCuda,
            [](const OpTestParams&) { return 0.0; },
            [](const OpTestParams&) { return 0.0; },
            true
        };
    }

    static std::vector<OpCase> BuildRegistry() {
        return {
            MakeAddToCase(),
            MakeCatCase(),
            MakeMulCase(),
            MakePermuteCase(),
            MakeSplitCase(),
            MakeMatMulCase(),
            MakeLayerNormCase(),
            MakeRmsNormCase(),
            MakeW4A8ActivationQuantCase(),
            MakeNvfp4SwiGLUQuantCase(),
            MakeLinearNvfp4Case(),
            MakeLinearCase(),
            MakeMergeMoeW4A8Case(),
            MakeSiluCase(),
            MakeSoftmaxCase(),
            MakeGeluNewCase(),
            MakeSwigluCase(),
            MakeAttentionCase(),
            MakeRmsNormSpecializedCase(),
            MakeRecurrentFromConvFp16Case(),
            MakeLinearFp8LegacyCase(),
            MakeLinearFp8W8A8Case(),
            MakeLinearFp8Block128Case(),
            MakeLinearInt8W8A8Case(),
            MakeMergeMoeFp8Case(),
            MakeFusedMoeFp8Case(),
            MakeRouterLinearFp16Case(),
            MakeFusedRouterTopKCase()
        };
    }

    static const OpCase* FindCase(const std::vector<OpCase> &registry, const std::string &name) {
        for (const auto &opCase : registry) {
            if (opCase.name == name) {
                return &opCase;
            }
        }
        return nullptr;
    }

    static void PrintHelp(const std::vector<OpCase> &registry) {
        std::cout
            << "Usage: ./optest --op <name|all> [--device cpu,cuda:0] [--param key=value]\n"
            << "                [--iters N] [--warmup N] [--atol X] [--rtol X] [--list]\n\n"
            << "Examples:\n"
            << "  ./optest --list\n"
            << "  ./optest --op matmul\n"
            << "  ./optest --op matmul --param m=64 --param k=128 --param n=256 --device cpu,cuda:0\n"
            << "  ./optest --op attention --param seq=128 --param dim=128 --iters 50\n\n"
            << "Available ops:\n";
        for (const auto &opCase : registry) {
            std::cout << "  " << opCase.name << "  # " << opCase.description << "\n";
        }
    }

    static CliConfig ParseArgs(int argc, char **argv) {
        CliConfig config;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            auto requireValue = [&](const std::string &name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for " + name);
                }
                return argv[++i];
            };

            if (arg == "-h" || arg == "--help") {
                config.help = true;
            } else if (arg == "--list") {
                config.listOps = true;
            } else if (arg == "--op") {
                config.op = requireValue(arg);
            } else if (arg == "--device") {
                config.deviceFilters = Split(requireValue(arg), ',');
            } else if (arg == "--param") {
                std::string kv = requireValue(arg);
                size_t pos = kv.find('=');
                if (pos == std::string::npos) {
                    throw std::runtime_error("--param expects key=value");
                }
                config.paramOverrides.push_back({Trim(kv.substr(0, pos)), Trim(kv.substr(pos + 1))});
            } else if (arg == "--iters") {
                config.iters = std::stoi(requireValue(arg));
            } else if (arg == "--warmup") {
                config.warmup = std::stoi(requireValue(arg));
            } else if (arg == "--atol") {
                config.atol = std::stof(requireValue(arg));
            } else if (arg == "--rtol") {
                config.rtol = std::stof(requireValue(arg));
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        return config;
    }

    static OpTestParams ResolveParams(const OpCase &opCase, const CliConfig &config) {
        OpTestParams params = opCase.makeDefaultParams();
        for (const auto &it : config.paramOverrides) {
            params.Override(it.first, it.second);
        }
        return params;
    }

    static bool RunSingleCase(const OpCase &opCase, const CliConfig &config) {
        OpTestParams params = ResolveParams(opCase, config);
        std::vector<std::string> devices = GetAvailableDevices();

        std::cout << "\n== op: " << opCase.name << " ==\n";
        std::cout << opCase.description << "\n";
        params.Print(std::cout);

        if (opCase.benchmarkOnly) {
            bool ran = false;
            for (const auto &device : devices) {
                if (!DeviceSelected(config.deviceFilters, device)) {
                    continue;
                }
                if (!opCase.canRun(params, device)) {
                    std::cout << "  [" << device << "] skipped: op not supported on this device\n";
                    continue;
                }

                BenchmarkResult bench = Benchmark(opCase, params, device, config.warmup, config.iters);
                std::cout << "  [" << device << "] BENCHMARK\n";
                std::cout << "    latency:"
                          << " avg_ms=" << std::fixed << std::setprecision(4) << bench.avgMs << "\n";
                std::cout << "    throughput:";
                if (bench.bandwidthGBps >= 0.0) {
                    std::cout << " io_speed=" << FormatIOSpeed(bench.bandwidthGBps);
                } else {
                    std::cout << " io_speed=n/a";
                }
                if (bench.computeTFlops >= 0.0) {
                    std::cout << ", compute_speed=" << FormatComputeSpeed(bench.computeTFlops);
                } else {
                    std::cout << ", compute_speed=n/a";
                }
                std::cout << "\n";
                ran = true;
            }
            return ran;
        }

        bool ok = true;

        for (const auto &device : devices) {
            if (!DeviceSelected(config.deviceFilters, device)) {
                continue;
            }
            if (!opCase.canRun(params, device)) {
                std::cout << "  [" << device << "] skipped: op not supported on this device\n";
                continue;
            }

            fastllm::Data baseline = opCase.runReference
                ? opCase.runReference(params, device)
                : opCase.run(params, "cpu");
            baseline.ToDevice(fastllm::DataDevice::CPU);
            fastllm::Data output = opCase.run(params, device);
            output.ToDevice(fastllm::DataDevice::CPU);
            ComparisonStats stats = CompareData(baseline, output, config.atol, config.rtol);
            bool pass = stats.passed;
            BenchmarkResult bench = Benchmark(opCase, params, device, config.warmup, config.iters);

            std::cout << "  [" << device << "] " << (pass ? "PASS" : "FAIL") << "\n";
            std::cout << "    accuracy:"
                      << " max_abs_diff=" << std::scientific << stats.maxAbsDiff
                      << ", max_rel_diff=" << stats.maxRelDiff << "\n";
            std::cout << "    latency:"
                      << " avg_ms=" << std::fixed << std::setprecision(4) << bench.avgMs << "\n";
            std::cout << "    throughput:";
            if (bench.bandwidthGBps >= 0.0) {
                std::cout << " io_speed=" << FormatIOSpeed(bench.bandwidthGBps);
            } else {
                std::cout << " io_speed=n/a";
            }
            if (bench.computeTFlops >= 0.0) {
                std::cout << ", compute_speed=" << FormatComputeSpeed(bench.computeTFlops);
            } else {
                std::cout << ", compute_speed=n/a";
            }
            std::cout << "\n";
            if (!pass) {
                std::cout << "    mismatch index=" << stats.mismatchIndex
                          << ", cpu=" << stats.expected
                          << ", device=" << stats.actual << "\n";
                ok = false;
            }
        }
        return ok;
    }
}

int main(int argc, char **argv) {
    try {
        std::vector<OpCase> registry = BuildRegistry();
        CliConfig config = ParseArgs(argc, argv);

        if (config.help) {
            PrintHelp(registry);
            return 0;
        }

        if (config.listOps) {
            for (const auto &opCase : registry) {
                std::cout << opCase.name << "\n";
            }
            return 0;
        }

        bool allOk = true;
        if (config.op == "all") {
            for (const auto &opCase : registry) {
                allOk = RunSingleCase(opCase, config) && allOk;
            }
        } else {
            const OpCase *opCase = FindCase(registry, config.op);
            if (opCase == nullptr) {
                throw std::runtime_error("unknown op: " + config.op);
            }
            allOk = RunSingleCase(*opCase, config);
        }

        std::cout << "\nSummary: " << (allOk ? "PASS" : "FAIL") << "\n";
        return allOk ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "optest error: " << e.what() << "\n";
        return 1;
    }
}
