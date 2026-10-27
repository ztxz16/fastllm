// Generate weights.bin with prepareGgufMmqFixtures.py. Build with the same
// include paths and libfastllm_tools as cudaGgufFastPathRegression.cu.
// Run input rows 9, 16, 32, 33, 65, 129 and 1024; 16 includes model widths.
// Small-row MMVQ is covered separately by cudaGgufFastPathRegression.cu.
#include "fastllm-gguf-dequant.cuh"
#include "fastllm-cuda-gguf-projections.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

// CPU GGUF decoders are independent of the CUDA tile loaders.
void Decode(int type, const void *source, float *output, int64_t count) {
#define DECODE(Type, Block, Function) \
    case GGML_TYPE_##Type: \
        dequantize_row_##Function(static_cast<const Block *>(source), output, count); \
        break
    switch (type) {
        DECODE(Q4_0, block_q4_0, q4_0);
        DECODE(Q4_1, block_q4_1, q4_1);
        DECODE(Q5_0, block_q5_0, q5_0);
        DECODE(Q5_1, block_q5_1, q5_1);
        DECODE(Q8_0, block_q8_0, q8_0);
        DECODE(Q2_K, block_q2_K, q2_K);
        DECODE(Q3_K, block_q3_K, q3_K);
        DECODE(Q4_K, block_q4_K, q4_K);
        DECODE(Q5_K, block_q5_K, q5_K);
        DECODE(Q6_K, block_q6_K, q6_K);
        DECODE(IQ2_XXS, block_iq2_xxs, iq2_xxs);
        DECODE(IQ2_XS, block_iq2_xs, iq2_xs);
        DECODE(IQ2_S, block_iq2_s, iq2_s);
        DECODE(IQ3_XXS, block_iq3_xxs, iq3_xxs);
        DECODE(IQ3_S, block_iq3_s, iq3_s);
        DECODE(IQ1_S, block_iq1_s, iq1_s);
        DECODE(IQ1_M, block_iq1_m, iq1_m);
        DECODE(IQ4_NL, block_iq4_nl, iq4_nl);
        DECODE(IQ4_XS, block_iq4_xs, iq4_xs);
        default:
            throw std::runtime_error("Unsupported fixture type");
    }
#undef DECODE
}

float RoundInput(float value, bool bfloat16) {
    return bfloat16 ? __bfloat162float(__float2bfloat16_rn(value))
                    : __half2float(__float2half_rn(value));
}

bool UsesFloatScales(int type) {
    return type != GGML_TYPE_Q2_K && type != GGML_TYPE_Q4_0 &&
           type != GGML_TYPE_Q4_1 && type != GGML_TYPE_Q5_1 &&
           type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q5_K &&
           type != GGML_TYPE_IQ1_S;
}

std::vector<float> QuantizeD4(const std::vector<float> &input) {
    // D4 stores FP32 activation scales per 32 elements, unlike MMVQ's FP16
    // scales. Reconstruct these inputs to isolate CUDA arithmetic/layout errors.
    std::vector<float> quantized(input.size());
    for (size_t offset = 0; offset < input.size(); offset += 32) {
        float maxAbs = 0.0f;
        for (int index = 0; index < 32; ++index) {
            maxAbs = std::max(maxAbs, std::abs(input[offset + index]));
        }
        const float scale = maxAbs / 127.0f;
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int index = 0; index < 32; ++index) {
            quantized[offset + index] = std::round(input[offset + index] * inverse) * scale;
        }
    }
    return quantized;
}

std::vector<float> ReadOutput(fastllm::Data &output) {
    FastllmCudaSyncCurrentThreadStream();
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("CUDA error");
    }
    output.ToDevice(fastllm::DataDevice::CPU);
    fastllm::ToDataTypeForceCPU(output, fastllm::DataType::FLOAT32);
    const float *values = reinterpret_cast<const float *>(output.cpuData);
    return {values, values + output.Count(0)};
}

void CheckUnsupportedShapes() {
    for (int columns : {128, 384}) {
        if (FastllmCudaHalfMatMulGGUFMMQ(
                nullptr, nullptr, nullptr, GGML_TYPE_Q4_0,
                16, columns, 1, cudaStreamPerThread) ||
            FastllmCudaBFloat16MatMulGGUFMMQ(
                nullptr, nullptr, nullptr, GGML_TYPE_Q4_0,
                16, columns, 1, cudaStreamPerThread)) {
            throw std::runtime_error("Partial K tile accepted");
        }
    }
}

} // namespace

int main(int argc, char **argv) try {
    if (argc < 2 || argc > 3) {
        throw std::runtime_error("Usage: test-mmq weights.bin [input-rows=16]");
    }
    const int tokens = argc == 3 ? std::stoi(argv[2]) : 16;
    if (tokens < 9 || tokens > 1024) {
        throw std::runtime_error("MMQ input rows must be in [9, 1024]");
    }
    FastllmCudaSetDevice(0);
    CheckUnsupportedShapes();

    std::ifstream file(argv[1], std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    uint32_t caseCount;
    file.read(reinterpret_cast<char *>(&caseCount), sizeof(caseCount));
    int checks = 0, failures = 0, dispatchChecks = 0;
    std::map<int, double> worst;
    for (uint32_t caseIndex = 0; caseIndex < caseCount; ++caseIndex) {
        uint32_t header[4];
        file.read(reinterpret_cast<char *>(header), sizeof(header));
        const int type = header[0], sourceRows = header[1], columns = header[2];
        if (sourceRows <= 0 || columns <= 0 || columns % 256 != 0 ||
            header[3] != size_t(sourceRows) * ggml_row_size(ggml_type(type), columns)) {
            throw std::runtime_error("Invalid fixture dimensions or packed size");
        }
        std::vector<char> packed(header[3]);
        file.read(packed.data(), packed.size());
        // Boundary sweeps use narrow matrices; 16 rows covers real model widths.
        if (tokens != 16 && columns != 256) {
            continue;
        }
        std::vector<float> weights(size_t(sourceRows) * columns);
        Decode(type, packed.data(), weights.data(), weights.size());
        if (!std::all_of(weights.begin(), weights.end(), [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("Nonfinite CPU-decoded fixture weights");
        }
        fastllm::Data weight(fastllm::DataType::DATA_GGUF_FORMAT, type, {sourceRows, columns});
        weight.disableGGUFRepack = true;
        weight.Allocate();
        std::memcpy(weight.cpuData, packed.data(), packed.size());
        weight.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        for (bool bfloat16 : {false, true}) {
            const auto dtype = bfloat16 ? fastllm::DataType::BFLOAT16 : fastllm::DataType::FLOAT16;
            std::mt19937 rng(711 + columns);
            std::normal_distribution<float> normal(0.0f, 0.2f);
            for (int pattern = 0; pattern < 3; ++pattern) {
                std::vector<float> input(size_t(tokens) * columns);
                for (size_t index = 0; index < input.size(); ++index) {
                    const float value = pattern == 0 ? normal(rng) :
                                        pattern == 1 ? (int(index % 3) - 1) * 0.125f : 0.0f;
                    input[index] = RoundInput(value, bfloat16);
                }
                const bool d4 = UsesFloatScales(type);
                const auto quantized = d4 ? QuantizeD4(input) : std::vector<float>();
                fastllm::Data x(dtype, {tokens, columns}, input);
                x.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
                for (int rows : {1, 7, sourceRows}) {
                    const int count = tokens * rows;
                    fastllm::Data output(dtype, {1, count + 2}, std::vector<float>(count + 2, -17.0f));
                    output.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
                    const bool usedMmq = bfloat16 ? FastllmCudaBFloat16MatMulGGUFMMQ(
                        x.cudaData, weight.cudaData, output.cudaData, type,
                        tokens, columns, rows, cudaStreamPerThread) : FastllmCudaHalfMatMulGGUFMMQ(
                        x.cudaData, weight.cudaData, output.cudaData, type,
                        tokens, columns, rows, cudaStreamPerThread);
                    if (!usedMmq) {
                        throw std::runtime_error("MMQ refused supported fixture");
                    }
                    const auto actual = ReadOutput(output);
                    if (actual[count] != -17.0f || actual[count + 1] != -17.0f) {
                        throw std::runtime_error("Output guard overwritten");
                    }
                    double error2 = 0, reference2 = 0, absolute2 = 0;
                    double quantError2 = 0, quantReference2 = 0;
                    for (int token = 0; token < tokens; ++token) {
                        for (int row = 0; row < rows; ++row) {
                            double reference = 0, absolute = 0, quantReference = 0;
                            for (int column = 0; column < columns; ++column) {
                                const size_t inputIndex = size_t(token) * columns + column;
                                const float decodedWeight = weights[size_t(row) * columns + column];
                                const double term = double(input[inputIndex]) * decodedWeight;
                                reference += term;
                                absolute += std::abs(term);
                                if (d4) {
                                    quantReference += double(quantized[inputIndex]) * decodedWeight;
                                }
                            }
                            const float value = actual[token * rows + row];
                            if (!std::isfinite(value)) {
                                throw std::runtime_error("Nonfinite CUDA output");
                            }
                            error2 += std::pow(value - reference, 2);
                            reference2 += reference * reference;
                            absolute2 += absolute * absolute;
                            if (d4) {
                                quantError2 += std::pow(value - quantReference, 2);
                                quantReference2 += quantReference * quantReference;
                            }
                        }
                    }
                    // Dense-reference tolerance includes intentional activation
                    // quantization. The absolute term handles cancelling sums.
                    const double bound = 0.025 * std::sqrt(reference2) +
                                         0.00005 * std::sqrt(absolute2) + 1e-6;
                    const double relative = std::sqrt(error2 / std::max(reference2, 1e-30));
                    if (pattern == 0 && rows == sourceRows) {
                        worst[type] = std::max(worst[type], relative);
                    }
                    auto checkError = [&](const char *label, double error, double limit) {
                        ++checks;
                        if (error > limit) {
                            ++failures;
                            std::cerr << label << " type=" << type << " bf16=" << bfloat16
                                      << " tokens=" << tokens << " columns=" << columns
                                      << " rows=" << rows << " pattern=" << pattern
                                      << " error=" << error << " bound=" << limit << "\n";
                        }
                    };
                    checkError("DENSE_FAIL", std::sqrt(error2), bound);
                    if (d4) {
                        const double quantBound = (bfloat16 ? 0.008 : 0.001) * std::sqrt(quantReference2) +
                                                  0.000002 * std::sqrt(absolute2) + 1e-6;
                        checkError("QUANT_FAIL", std::sqrt(quantError2), quantBound);
                    }

                    // Also check the public linear dispatcher, so a silently
                    // bypassed MMQ path cannot pass only the backend tests.
                    if (tokens == 16 && pattern == 0 && rows == sourceRows) {
                        fastllm::Data integrated(dtype, {tokens, rows});
                        integrated.Allocate();
                        integrated.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
                        fastllm::Data bias;
                        const bool ok = bfloat16 ? FastllmCudaBFloat16MatMulGGUF(
                            x, weight, bias, integrated, tokens, columns, rows) : FastllmCudaHalfMatMulGGUF(
                            x, weight, bias, integrated, tokens, columns, rows);
                        const auto dispatched = ReadOutput(integrated);
                        if (!ok || !std::equal(dispatched.begin(), dispatched.end(), actual.begin())) {
                            throw std::runtime_error("Public linear dispatch differs from MMQ");
                        }
                        ++dispatchChecks;
                    }
                }
            }
        }
    }
    if (checks == 0) {
        throw std::runtime_error("No applicable test fixtures");
    }
    for (auto [type, relative] : worst) {
        std::cout << "TYPE type=" << type << " max_random_relative_L2=" << relative << "\n";
    }
    std::cout << "RESULT tokens=" << tokens << " checks=" << checks
              << " dispatch_checks=" << dispatchChecks << " failures=" << failures << "\n";
    return failures ? 1 : 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
}
