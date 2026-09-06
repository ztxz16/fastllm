#include "fastllm-gguf-dequant.cuh"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
    void Expect(bool condition, const std::string &message) {
        if (!condition) throw std::runtime_error(message);
    }

    std::vector<float> ToFloatVector(fastllm::Data data) {
        data.ToDevice(fastllm::DataDevice::CPU);
        fastllm::ToDataTypeForceCPU(data, fastllm::DataType::FLOAT32);
        const float *values = (const float*)data.cpuData;
        return {values, values + data.Count(0)};
    }

    void ExpectFloatNear(const std::vector<float> &expected,
                         const std::vector<float> &actual,
                         float atol, float rtol, const std::string &label) {
        Expect(expected.size() == actual.size(), label + " size mismatch");
        for (size_t i = 0; i < expected.size(); ++i) {
            Expect(std::isfinite(actual[i]) &&
                   std::fabs(actual[i] - expected[i]) <= atol + rtol * std::fabs(expected[i]),
                   label + " mismatch at " + std::to_string(i) +
                   ": expected=" + std::to_string(expected[i]) +
                   ", actual=" + std::to_string(actual[i]));
        }
    }

    void RunCudaGgufChunkedDequantRegression() {
        FastllmCudaSetDevice(0);
        cudaDeviceProp properties;
        Expect(cudaGetDeviceProperties(&properties, 0) == cudaSuccess,
               "chunked dequant device query failed");
        // R4 rounds a 67-row workspace down to 64 rows. The final chunk
        // must keep the full output stride.
        constexpr int inputDim = 512;
        fastllm::Data workspace(fastllm::DataType::FLOAT16, {67, inputDim});
        workspace.ToDevice(fastllm::DataDevice::CUDA);
        workspace.Allocate();
        cublasHandle_t handle;
        Expect(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS,
               "chunked dequant reference handle creation failed");
        Expect(cublasSetStream(handle, cudaStreamPerThread) == CUBLAS_STATUS_SUCCESS,
               "chunked dequant reference stream setup failed");

        struct QuantCase { ggml_type type, sourceType; };
        for (const QuantCase test : {
                 QuantCase{GGML_TYPE_Q5_K, GGML_TYPE_Q5_K},
                 QuantCase{GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS},
                 QuantCase{GGML_TYPE_Q2_K_R4, GGML_TYPE_Q2_K},
                 QuantCase{GGML_TYPE_Q4_K_R4, GGML_TYPE_Q4_K},
                 QuantCase{GGML_TYPE_Q5_K_R4, GGML_TYPE_Q5_K},
                 QuantCase{GGML_TYPE_Q6_K_R4, GGML_TYPE_Q6_K}}) {
            const ggml_type type = test.type, sourceType = test.sourceType;
            const bool repacked = type != sourceType;
            for (int outputDim : {12, repacked ? 64 : 67, repacked ? 148 : 149}) {
                std::vector<float> source((size_t)outputDim * inputDim);
                for (size_t i = 0; i < source.size(); ++i) {
                    source[i] = 0.1f * std::sin((float)i * 0.03125f);
                }
                fastllm::Data weight(fastllm::DataType::DATA_GGUF_FORMAT,
                                     (int)type, {outputDim, inputDim});
                weight.name = "regression.chunked_dequant." +
                    std::string(ggml_type_name(type));
                weight.disableGGUFRepack = true;
                weight.forceGGUFFp32Dequant = true;
                weight.Allocate();
                std::vector<uint8_t> packed(ggml_row_size(sourceType, inputDim) * outputDim);
                switch (sourceType) {
                    case GGML_TYPE_Q2_K:
                        quantize_row_q2_K_ref(source.data(), (block_q2_K*)packed.data(), source.size());
                        break;
                    case GGML_TYPE_Q4_K:
                        quantize_row_q4_K_ref(source.data(), (block_q4_K*)packed.data(), source.size());
                        break;
                    case GGML_TYPE_Q5_K:
                        quantize_row_q5_K_ref(source.data(), (block_q5_K*)packed.data(), source.size());
                        break;
                    case GGML_TYPE_Q6_K:
                        quantize_row_q6_K_ref(source.data(), (block_q6_K*)packed.data(), source.size());
                        break;
                    case GGML_TYPE_IQ4_XS: {
                        auto *blocks = (block_iq4_xs*)packed.data();
                        for (size_t b = 0; b < source.size() / QK_K; ++b) {
                            blocks[b].d = fastllm::float_to_half(0.000025f * (1 + b % 7));
                            blocks[b].scales_h = (uint16_t)(b * 137);
                            for (size_t i = 0; i < sizeof(blocks[b].scales_l); ++i) {
                                blocks[b].scales_l[i] = (uint8_t)(b * 19 + i * 31);
                            }
                            for (size_t i = 0; i < sizeof(blocks[b].qs); ++i) {
                                blocks[b].qs[i] = (uint8_t)(b * 13 + i * 29 + 7);
                            }
                        }
                        break;
                    }
                    default: throw std::runtime_error("unsupported chunked dequant test type");
                }
                if (repacked) {
                    const Repack *info = get_repack_info(sourceType);
                    Expect(info != nullptr && info->new_type == type, "missing R4 repack");
                    info->repack(outputDim, inputDim, (const char*)packed.data(),
                                 (char*)weight.cpuData, false);
                } else {
                    std::memcpy(weight.cpuData, packed.data(), packed.size());
                }
                weight.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);

                for (auto dtype : {fastllm::DataType::FLOAT16, fastllm::DataType::BFLOAT16}) {
                    const bool bf16 = dtype == fastllm::DataType::BFLOAT16;
                    if (bf16 && properties.major < 8) continue;
                    // Match the old full-matrix dequant + cuBLAS computation.
                    fastllm::Data fullWeight(dtype, {outputDim, inputDim});
                    fullWeight.ToDevice(fastllm::DataDevice::CUDA);
                    fullWeight.Allocate();
                    if (bf16) {
                        auto dequant = ggml_get_to_bf16_cuda(type);
                        Expect(dequant != nullptr, "missing BF16 dequant dispatch");
                        dequant(weight.cudaData, (__nv_bfloat16*)fullWeight.cudaData,
                                outputDim, inputDim, cudaStreamPerThread);
                    } else {
                        auto dequant = ggml_get_to_fp16_cuda(type);
                        Expect(dequant != nullptr, "missing FP16 dequant dispatch");
                        dequant(weight.cudaData, (half*)fullWeight.cudaData,
                                outputDim, inputDim, cudaStreamPerThread);
                    }
                    for (int batch : {1, 3, 9}) {
                        std::vector<float> values((size_t)batch * inputDim);
                        for (size_t i = 0; i < values.size(); ++i) {
                            values[i] = 0.25f * std::cos((float)i * 0.01953125f);
                        }
                        fastllm::Data input(dtype, {batch, inputDim}, values);
                        input.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
                        fastllm::Data expected(dtype, {batch, outputDim});
                        fastllm::Data actual(dtype, {batch, outputDim});
                        expected.ToDevice(fastllm::DataDevice::CUDA);
                        actual.ToDevice(fastllm::DataDevice::CUDA);
                        expected.Allocate();
                        actual.Allocate();
                        fastllm::Data emptyBias;
                        uint16_t halfOne = fastllm::float_to_half(1.0f), halfZero = 0;
                        float one = 1.0f, zero = 0.0f;
                        const cudaDataType_t cudaType = bf16 ? CUDA_R_16BF : CUDA_R_16F;
                        Expect(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            outputDim, batch, inputDim,
                            bf16 ? (void*)&one : (void*)&halfOne,
                            fullWeight.cudaData, cudaType, inputDim,
                            input.cudaData, cudaType, inputDim,
                            bf16 ? (void*)&zero : (void*)&halfZero,
                            expected.cudaData, cudaType, outputDim,
                            bf16 ? CUDA_R_32F : CUDA_R_16F,
                            CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS,
                            "full-matrix reference GEMM failed");
                        auto run = [&]() {
                            auto handle = getFastllmCublasHandle();
                            if (bf16) {
                                FastllmGGUFDequantGemm(
                                    (__nv_bfloat16*)input.cudaData, weight,
                                    (__nv_bfloat16*)actual.cudaData, batch, inputDim, outputDim,
                                    workspace.cudaData, workspace.GetBytes(),
                                    ggml_get_to_bf16_cuda(type), handle, cudaStreamPerThread);
                            } else {
                                FastllmGGUFDequantGemm(
                                    (half*)input.cudaData, weight,
                                    (half*)actual.cudaData, batch, inputDim, outputDim,
                                    workspace.cudaData, workspace.GetBytes(),
                                    ggml_get_to_fp16_cuda(type), handle, cudaStreamPerThread);
                            }
                        };
                        run();
                        FastllmCudaSyncCurrentThreadStream();
                        const auto reference = ToFloatVector(expected);
                        const std::string label = weight.name + (bf16 ? " BF16" : " FP16") +
                            " rows=" + std::to_string(outputDim) + " batch=" + std::to_string(batch);
                        const float tolerance = bf16 ? 4e-3f : 5e-4f;
                        ExpectFloatNear(reference, ToFloatVector(actual), tolerance, tolerance, label);
                        if (batch == 9) {
                            Expect(bf16 ? FastllmCudaBFloat16MatMulGGUF(
                                input, weight, emptyBias, actual, batch, inputDim, outputDim) :
                                FastllmCudaHalfMatMulGGUF(
                                input, weight, emptyBias, actual, batch, inputDim, outputDim),
                                label + " public GGUF GEMM failed");
                            FastllmCudaSyncCurrentThreadStream();
                            ExpectFloatNear(reference, ToFloatVector(actual),
                                            tolerance, tolerance, label + " public GGUF GEMM");
                        }
                        if (batch == 3 && outputDim > 67) {
                            FastllmCudaSetNcclForceSync(false);
                            void *graph = nullptr, *graphExec = nullptr;
                            Expect(FastllmCudaGraphBeginCapture(), label + " graph begin failed");
                            run();
                            Expect(FastllmCudaGraphEndCapture(&graph) && graph,
                                   label + " graph capture failed");
                            Expect(FastllmCudaGraphInstantiate(graph, &graphExec) && graphExec,
                                   label + " graph instantiate failed");
                            for (int replay = 0; replay < 2; ++replay) {
                                FastllmCudaMemset0(actual.cudaData, actual.GetBytes());
                                Expect(FastllmCudaGraphLaunch(graphExec), label + " graph replay failed");
                                FastllmCudaSyncCurrentThreadStream();
                                ExpectFloatNear(reference, ToFloatVector(actual),
                                                tolerance, tolerance, label + " graph");
                            }
                            FastllmCudaGraphExecDestroy(graphExec);
                            FastllmCudaGraphDestroy(graph);
                        }
                    }
                }
            }
        }
        cublasDestroy(handle);
    }
}

int main() {
    try {
        if (FastllmCudaGetDeviceCount() == 0) return 77;
        RunCudaGgufChunkedDequantRegression();
        std::cout << "CUDA GGUF chunked dequant regressions: PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
