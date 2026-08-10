#include "executor.h"
#include "fastllm.h"
#include "model.h"
#include "models/basellm.h"
#include "models/deepseekv4.h"
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
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fastllm {
    bool FastllmGemmBFloat16NVFP4Block16E8M0_AVX512BF16(
        const void *A, long lda, const void *B, long ldb,
        void *C, long ldc, int n, int m, int k, int st, int end);
    bool FastllmGemmBFloat16NVFP4Block32E8M0_AVX512BF16(
        const void *A, long lda, const void *B, long ldb,
        void *C, long ldc, int n, int m, int k, int st, int end);
}

#ifdef USE_CUDA
namespace fastllm {
    bool DeepSeekV4CopyCudaTensorToCpuForTest(
        const Data &source, Data &destination);
    bool DeepSeekV4AddQuantizedCudaReplicaForTest(
        Data &activation, int device);
    int DeepSeekV4BuildWindowKVPrefixForTest(
        const Data &windowKV, int bsz, int headDim, int startPos,
        int windowSize, Data &output);
    void DeepSeekV4UpdateWindowKVCacheForTest(
        const Data &kv, int bsz, int headDim, int startPos,
        int windowSize, Data &windowKV);
    void DeepSeekV4AppendCompressorRawForTest(
        const Data &kv, const Data &score, int bsz, int seqlen,
        int wideDim, Data &allKV, Data &allScore);
    void DeepSeekV4TrimCompressorRawForTest(
        int bsz, int totalLen, int compressRatio, int wideDim,
        int compressedBlocks, Data &allKV, Data &allScore,
        int &rawTokenBase);
}
#endif

namespace {
    void Expect(bool condition, const std::string &message);
    void ExpectIntEqual(const std::vector<int32_t> &expected,
                        const std::vector<int32_t> &actual,
                        const std::string &name);
    void RunCpuDeepSeekV4IndexerBenchmark(int sequence);

    void RunCpuDeepSeekV4Nvfp4Block32Benchmark() {
        constexpr int rows = 8;
        constexpr int inputDim = 4096;
        constexpr int outputDim = 4096;
        constexpr int block16Count = inputDim / 16;
        constexpr int block16Stride = block16Count * 9;
        constexpr int block32Count = inputDim / 32;
        constexpr int block32Stride = block32Count * 17;
        constexpr int iterations = 40;

        uint32_t state = 0x9e3779b9u;
        auto nextRandom = [&state]() {
            state = state * 1664525u + 1013904223u;
            return state;
        };
        std::vector<uint16_t> input((size_t)rows * inputDim);
        for (uint16_t &bits : input) {
            const int value = (int)(nextRandom() >> 24) - 128;
            bits = fastllm::Float32ToBFloat16RNEBits(value / 32.0f);
        }
        std::vector<uint8_t> block16Weight(
            (size_t)outputDim * block16Stride);
        std::vector<uint8_t> block32Weight(
            (size_t)outputDim * block32Stride);
        constexpr uint8_t scaleValues[] = {
            0, 1, 63, 64, 119, 120, 121, 190, 191
        };
        for (int outputChannel = 0;
             outputChannel < outputDim; outputChannel++) {
            uint8_t *row16 = block16Weight.data() +
                (size_t)outputChannel * block16Stride;
            for (int block = 0; block < block16Count; block++) {
                uint8_t *packed = row16 + block * 9;
                for (int byte = 0; byte < 8; byte++) {
                    packed[byte] = (uint8_t)(nextRandom() >> 24);
                }
                // The model's blockM=32 source scale is duplicated across
                // each adjacent pair of repacked blockM=16 blocks.
                packed[8] = scaleValues[
                    (outputChannel * 17 + block / 2) %
                    (sizeof(scaleValues) / sizeof(scaleValues[0]))];
            }
            uint8_t *row32 = block32Weight.data() +
                (size_t)outputChannel * block32Stride;
            for (int block = 0; block < block32Count; block++) {
                const uint8_t *first = row16 + block * 18;
                const uint8_t *second = first + 9;
                Expect(first[8] == second[8],
                       "block16 scale pair is not compactable.");
                uint8_t *packed = row32 + block * 17;
                std::memcpy(packed, first, 8);
                std::memcpy(packed + 8, second, 8);
                packed[16] = first[8];
            }
        }
        std::vector<float> output((size_t)rows * outputDim);
        auto run = [&](bool block32, int activeRows) {
            std::fill(output.begin(), output.end(), 0.0f);
            bool used = block32 ? fastllm::
                FastllmGemmBFloat16NVFP4Block32E8M0_AVX512BF16(
                    input.data(), inputDim * (long)sizeof(uint16_t),
                    block32Weight.data(), block32Stride,
                    output.data(), outputDim * (long)sizeof(float),
                    activeRows, inputDim, outputDim, 0, outputDim) :
                fastllm::
                FastllmGemmBFloat16NVFP4Block16E8M0_AVX512BF16(
                    input.data(), inputDim * (long)sizeof(uint16_t),
                    block16Weight.data(), block16Stride,
                    output.data(), outputDim * (long)sizeof(float),
                    activeRows, inputDim, outputDim, 0, outputDim);
            Expect(used, "AVX512-BF16 NVFP4 E8M0 kernel is unavailable.");
            uint64_t hash = 1469598103934665603ull;
            for (size_t index = 0;
                 index < (size_t)activeRows * outputDim; index++) {
                uint32_t bits;
                std::memcpy(&bits, &output[index], sizeof(bits));
                hash ^= bits;
                hash *= 1099511628211ull;
            }
            return hash;
        };

        for (int activeRows = 1; activeRows <= rows; activeRows++) {
            const uint64_t block16Hash = run(false, activeRows);
            const uint64_t block32Hash = run(true, activeRows);
            Expect(block16Hash == block32Hash,
                   "NVFP4 block32 kernel changed FP32 output bits at rows=" +
                       std::to_string(activeRows) + ".");
            std::cout << "rows=" << activeRows << " hash="
                      << std::hex << block32Hash << std::dec << "\n";
        }
        constexpr int prefillRows = 64;
        std::vector<uint16_t> prefillInput(
            (size_t)prefillRows * inputDim);
        for (int row = 0; row < prefillRows; row++) {
            std::memcpy(
                prefillInput.data() + (size_t)row * inputDim,
                input.data() + (size_t)(row % rows) * inputDim,
                (size_t)inputDim * sizeof(uint16_t));
        }
        std::vector<float> prefillOutput16(
            (size_t)prefillRows * outputDim);
        std::vector<float> prefillOutput32(
            (size_t)prefillRows * outputDim);
        fastllm::FastllmGemm(
            prefillRows, inputDim, outputDim,
            prefillInput.data(), inputDim * (long)sizeof(uint16_t),
            block16Weight.data(), block16Stride,
            prefillOutput16.data(), outputDim * (long)sizeof(float),
            0, outputDim, fastllm::DataType::BFLOAT16,
            fastllm::DataType::NVFP4_BLOCK_16_E8M0,
            fastllm::DataType::FLOAT32);
        fastllm::FastllmGemm(
            prefillRows, inputDim, outputDim,
            prefillInput.data(), inputDim * (long)sizeof(uint16_t),
            block32Weight.data(), block32Stride,
            prefillOutput32.data(), outputDim * (long)sizeof(float),
            0, outputDim, fastllm::DataType::BFLOAT16,
            fastllm::DataType::NVFP4_BLOCK_32_E8M0,
            fastllm::DataType::FLOAT32);
        Expect(std::memcmp(
                   prefillOutput16.data(), prefillOutput32.data(),
                   prefillOutput16.size() * sizeof(float)) == 0,
               "CPU NVFP4 block32 prefill changed FP32 output bits.");
        std::cout << "cpu block32 prefill_rows=" << prefillRows
                  << " hash=bitwise-match\n";
#ifdef USE_CUDA
        int cudaDeviceCount = 0;
        if (cudaGetDeviceCount(&cudaDeviceCount) == cudaSuccess &&
            cudaDeviceCount > 0) {
            constexpr int cudaRows = 64;
            std::vector<uint16_t> cudaInputValues(
                (size_t)cudaRows * inputDim);
            for (int row = 0; row < cudaRows; row++) {
                std::memcpy(
                    cudaInputValues.data() + (size_t)row * inputDim,
                    input.data() + (size_t)(row % rows) * inputDim,
                    (size_t)inputDim * sizeof(uint16_t));
            }
            fastllm::Data cudaInput(
                fastllm::DataType::BFLOAT16, {cudaRows, inputDim});
            cudaInput.Allocate(false);
            std::memcpy(cudaInput.cpuData, cudaInputValues.data(),
                        cudaInput.GetBytes());
            cudaInput.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, true);

            fastllm::Data cudaWeight16(
                fastllm::DataType::NVFP4_BLOCK_16_E8M0,
                {outputDim, inputDim});
            cudaWeight16.Allocate(false);
            std::memcpy(cudaWeight16.cpuData, block16Weight.data(),
                        block16Weight.size());
            cudaWeight16.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
            fastllm::Data cudaWeight32(
                fastllm::DataType::NVFP4_BLOCK_32_E8M0,
                {outputDim, inputDim});
            cudaWeight32.Allocate(false);
            std::memcpy(cudaWeight32.cpuData, block32Weight.data(),
                        block32Weight.size());
            cudaWeight32.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, true);

            fastllm::Data cudaOutput16(
                fastllm::DataType::BFLOAT16, {cudaRows, outputDim});
            fastllm::Data cudaOutput32(
                fastllm::DataType::BFLOAT16, {cudaRows, outputDim});
            cudaOutput16.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
            cudaOutput32.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
            cudaOutput16.Allocate(false);
            cudaOutput32.Allocate(false);
            fastllm::Data emptyBias;
            Expect(fastllm::IsCudaLinearDataTypeSupported(
                       fastllm::DataType::BFLOAT16,
                       fastllm::DataType::NVFP4_BLOCK_32_E8M0,
                       fastllm::DataType::FLOAT32),
                   "CUDA Linear does not advertise NVFP4 block32 support.");
            Expect(FastllmCudaBFloat16MatMulNVFP4Block16E8M0(
                       cudaInput, cudaWeight16, emptyBias, cudaOutput16,
                       cudaRows, inputDim, outputDim),
                   "CUDA NVFP4 block16 E8M0 reference failed.");
            Expect(FastllmCudaBFloat16MatMulNVFP4Block16E8M0(
                       cudaInput, cudaWeight32, emptyBias, cudaOutput32,
                       cudaRows, inputDim, outputDim),
                   "CUDA NVFP4 block32 E8M0 path failed.");
            FastllmCudaSyncCurrentThreadStream();
            cudaOutput16.ToDevice(fastllm::DataDevice::CPU);
            cudaOutput32.ToDevice(fastllm::DataDevice::CPU);
            Expect(std::memcmp(
                       cudaOutput16.cpuData, cudaOutput32.cpuData,
                       cudaOutput16.GetBytes()) == 0,
                   "CUDA NVFP4 block32 changed BF16 output bits.");
            for (int gemvRows : {1, 7, 8, 31}) {
                fastllm::Data halfInput(
                    fastllm::DataType::FLOAT16,
                    {gemvRows, inputDim});
                halfInput.Allocate(false);
                uint16_t *halfBits = (uint16_t*)halfInput.cpuData;
                for (int index = 0; index < gemvRows * inputDim;
                     index++) {
                    const float value = fastllm::BFloat16BitsToFloat32(
                        input[(size_t)(index % (rows * inputDim))]);
                    halfBits[index] = fastllm::float_to_half(value);
                }
                halfInput.ToDevice(
                    fastllm::DataDevice::CUDA,
                    std::vector<int>{0}, true);
                fastllm::Data halfOutput16(
                    fastllm::DataType::FLOAT16,
                    {gemvRows, outputDim});
                fastllm::Data halfOutput32(
                    fastllm::DataType::FLOAT16,
                    {gemvRows, outputDim});
                halfOutput16.ToDevice(
                    fastllm::DataDevice::CUDA,
                    std::vector<int>{0}, false);
                halfOutput32.ToDevice(
                    fastllm::DataDevice::CUDA,
                    std::vector<int>{0}, false);
                halfOutput16.Allocate(false);
                halfOutput32.Allocate(false);
                Expect(FastllmCudaHalfMatMulFloatNVFP4Block16E8M0(
                           halfInput, cudaWeight16, emptyBias,
                           halfOutput16, gemvRows, inputDim, outputDim),
                       "CUDA NVFP4 block16 FP16 GEMV reference failed.");
                Expect(FastllmCudaHalfMatMulFloatNVFP4Block16E8M0(
                           halfInput, cudaWeight32, emptyBias,
                           halfOutput32, gemvRows, inputDim, outputDim),
                       "CUDA NVFP4 block32 FP16 GEMV failed.");
                FastllmCudaSyncCurrentThreadStream();
                halfOutput16.ToDevice(fastllm::DataDevice::CPU);
                halfOutput32.ToDevice(fastllm::DataDevice::CPU);
                Expect(std::memcmp(
                           halfOutput16.cpuData, halfOutput32.cpuData,
                           halfOutput16.GetBytes()) == 0,
                       "CUDA NVFP4 block32 FP16 GEMV changed output bits at "
                       "rows=" + std::to_string(gemvRows) + ".");
            }
            std::cout << "cuda block32 hash=bitwise-match "
                      << "gemv_rows=1,7,8,31\n";

            if (cudaDeviceCount >= 2) {
                const int originalDevice = FastllmCudaGetDevice();
                const std::vector<int> devices = {0, 1};
                auto makeCpuWeight = [&]() {
                    fastllm::Data weight(
                        fastllm::DataType::NVFP4_BLOCK_32_E8M0,
                        {outputDim, inputDim});
                    weight.name =
                        "regression.nvfp4_block32_cpu_source.weight";
                    weight.Allocate(false);
                    std::memcpy(
                        weight.cpuData, block32Weight.data(),
                        block32Weight.size());
                    return weight;
                };
                auto copyLocalBytes = [](fastllm::Data *local) {
                    Expect(
                        local != nullptr && local->cudaData != nullptr,
                        "NVFP4 block32 TP shard is missing CUDA storage.");
                    FastllmCudaSetDevice(
                        GetPointerDeviceId(local->cudaData));
                    std::vector<uint8_t> bytes(local->GetBytes());
                    FastllmCudaCopyFromDeviceToHost(
                        bytes.data(), local->cudaData, bytes.size());
                    return bytes;
                };

                {
                    fastllm::Data weight = makeCpuWeight();
                    fastllm::Data bias;
                    DivisionScheme scheme = {
                        {0, {{0, outputDim / 2}}},
                        {1, {{outputDim / 2, outputDim}}}
                    };
                    std::vector<int> mutableDevices = devices;
                    Expect(
                        SplitMultiCudaWeight(
                            weight, bias, mutableDevices, scheme, 0, true),
                        "NVFP4 block32 CPU-source TP row split failed.");
                    for (int device : devices) {
                        fastllm::Data *local =
                            weight.multiDeviceDatas.at(device);
                        std::vector<uint8_t> actual =
                            copyLocalBytes(local);
                        const int rowBegin = scheme[device][0].first;
                        const uint8_t *expected = block32Weight.data() +
                            (size_t)rowBegin * block32Stride;
                        Expect(
                            std::memcmp(
                                actual.data(), expected,
                                actual.size()) == 0,
                            "NVFP4 block32 TP row split copied the wrong "
                            "CPU-source bytes.");
                    }
                }

                {
                    fastllm::Data weight = makeCpuWeight();
                    fastllm::Data bias;
                    DivisionScheme scheme = {
                        {0, {{0, inputDim / 2}}},
                        {1, {{inputDim / 2, inputDim}}}
                    };
                    std::vector<int> mutableDevices = devices;
                    Expect(
                        SplitMultiCudaWeight(
                            weight, bias, mutableDevices, scheme, 1, true),
                        "NVFP4 block32 CPU-source TP column split failed.");
                    const size_t localRowBytes =
                        fastllm::GetDataBytes(
                            fastllm::DataType::NVFP4_BLOCK_32_E8M0,
                            1, inputDim / 2);
                    for (int device : devices) {
                        fastllm::Data *local =
                            weight.multiDeviceDatas.at(device);
                        std::vector<uint8_t> actual =
                            copyLocalBytes(local);
                        const size_t sourceOffset =
                            (size_t)(scheme[device][0].first / 32) * 17;
                        for (int row = 0; row < outputDim; row++) {
                            const uint8_t *expected =
                                block32Weight.data() +
                                (size_t)row * block32Stride + sourceOffset;
                            Expect(
                                std::memcmp(
                                    actual.data() +
                                        (size_t)row * localRowBytes,
                                    expected, localRowBytes) == 0,
                                "NVFP4 block32 TP column split copied the "
                                "wrong CPU-source bytes.");
                        }
                    }
                }
                FastllmCudaSetDevice(originalDevice);
                std::cout << "cuda block32 TP CPU-source row/column "
                          << "split=bytewise-match\n";
            }
        }
#endif
        for (int timedRows = 2; timedRows <= rows; timedRows++) {
            for (bool block32 : {false, true}) {
                for (int warmup = 0; warmup < 3; warmup++) {
                    run(block32, timedRows);
                }
                const auto begin = std::chrono::steady_clock::now();
                uint64_t hash = 0;
                for (int iteration = 0; iteration < iterations;
                     iteration++) {
                    hash ^= run(block32, timedRows);
                }
                const double elapsedMs =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin).count();
                std::cout << (block32 ? "block32" : "block16")
                          << " rows=" << timedRows
                          << " iterations=" << iterations
                          << " average_ms=" << elapsedMs / iterations
                          << " guard=" << std::hex << hash << std::dec
                          << "\n";
            }
        }
    }

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

    void RunDeepSeekV4DsparkPrefixSelectionRegression() {
        fastllm::DeepSeekV4HistoryCacheManager manager;
        std::vector<int> input(700);
        for (int i = 0; i < (int)input.size(); ++i) {
            input[i] = (i * 37 + 11) % 32000;
        }
        auto prefixHash = [&](int tokens) {
            uint64_t hash = 1469598103934665603ULL;
            auto mix = [&](uint64_t value) {
                hash ^= value + 0x9e3779b97f4a7c15ULL +
                        (hash << 6) + (hash >> 2);
                hash *= 1099511628211ULL;
            };
            mix((uint64_t)manager.logicalBlockSize);
            for (int i = 0; i < tokens; ++i) {
                mix((uint64_t)(uint32_t)input[i]);
                if ((i + 1) % manager.logicalBlockSize == 0) {
                    mix(0xff51afd7ed558ccdULL ^
                        (uint64_t)((i + 1) / manager.logicalBlockSize));
                }
            }
            return hash;
        };

        auto makeMemory = [&](int tokens, bool dsparkValid) {
            fastllm::DeepSeekV4HistoryCacheMemory memory;
            memory.tokens = tokens;
            memory.blockCount = tokens / manager.logicalBlockSize;
            memory.inputToken.assign(input.begin(), input.begin() + tokens);
            memory.blockHash = prefixHash(tokens);
            memory.layers.resize(1);
            memory.dsparkValid = dsparkValid;
            memory.dsparkCommittedTokens = dsparkValid ? tokens : 0;
            return memory;
        };

        manager.Record(makeMemory(256, true));
        manager.Record(makeMemory(512, false));
        fastllm::DeepSeekV4HistoryCacheMemory hit;
        int hitLen = 0;
        Expect(manager.Get(input, hit, hitLen, false) && hitLen == 512,
               "DeepSeek-V4 target prefix lookup did not choose the longest snapshot.");
        Expect(manager.Get(input, hit, hitLen, true) && hitLen == 256 &&
                   hit.dsparkValid,
               "DeepSeek-V4 DSpark prefix lookup reused a target-only snapshot.");

        struct TestDeepSeekV4Model : fastllm::DeepSeekV4Model {
            using fastllm::DeepSeekV4Model::dsparkConfidenceThreshold;
            using fastllm::DeepSeekV4Model::dsparkDraftBestUs;
            using fastllm::DeepSeekV4Model::dsparkTokens;
            using fastllm::DeepSeekV4Model::dsparkValidationCount;
            using fastllm::DeepSeekV4Model::dsparkVerifyBestUs;
            using fastllm::DeepSeekV4Model::SelectDsparkVerifyDrafts;
        } model;
        model.dsparkTokens = 7;
        model.dsparkConfidenceThreshold = 0.5f;
        fastllm::DeepSeekV4DsparkProposal proposal;
        proposal.tokens.assign(7, 11);
        proposal.confidence = {
            0.96f, 0.83f, 0.61f, 0.49f, 0.80f, 0.77f, 0.75f};
        fastllm::DeepSeekV4DsparkContext context;
        Expect(model.SelectDsparkVerifyDrafts(proposal, context) == 7,
               "DeepSeek-V4 DSpark scheduler skipped calibration probes.");
        model.dsparkValidationCount.store(6);
        Expect(model.SelectDsparkVerifyDrafts(proposal, context) == 3,
               "DeepSeek-V4 DSpark confidence scheduler did not prune at "
               "the first low-confidence draft.");
        model.dsparkDraftBestUs.store(15000);
        model.dsparkVerifyBestUs[0].store(50000);
        model.dsparkVerifyBestUs[6].store(180000);
        model.dsparkVerifyBestUs[7].store(205000);
        Expect(model.SelectDsparkVerifyDrafts(proposal, context) == 2,
               "DeepSeek-V4 DSpark hardware-aware scheduler ignored the "
               "profiled verifier curve.");
        model.dsparkConfidenceThreshold = 0.0f;
        Expect(model.SelectDsparkVerifyDrafts(proposal, context) == 7,
               "DeepSeek-V4 DSpark fixed-block compatibility mode was pruned.");
        proposal = fastllm::DeepSeekV4DsparkProposal();
        proposal.gpuDeferred = true;
        model.dsparkConfidenceThreshold = 0.5f;
        Expect(model.SelectDsparkVerifyDrafts(proposal, context) == 7,
               "DeepSeek-V4 DSpark deferred GPU proposal lost fixed shape.");
    }

#ifdef USE_CUDA
    void RunCudaGreedyTieBreakRegression() {
        constexpr int vocabSize = 32768;
        const size_t logitsBytes = (size_t)vocabSize * sizeof(float);
        FastllmCudaSetDevice(0);

        float *cudaLogits = (float*)FastllmCudaMalloc(logitsBytes);
        int *cudaOutput = (int*)FastllmCudaMalloc(sizeof(int));
        float *cudaFloatOutput = (float*)FastllmCudaMalloc(sizeof(float));
        Expect(cudaLogits != nullptr && cudaOutput != nullptr &&
                   cudaFloatOutput != nullptr,
               "failed to allocate CUDA greedy tie-break buffers");

        auto runCase = [&](const std::string &name,
                           const std::vector<int> &maxIds,
                           int expectedId) {
            std::vector<float> logits(vocabSize, -1000.0f);
            for (int id : maxIds) {
                Expect(id >= 0 && id < vocabSize,
                       name + " has an invalid maximum token ID");
                logits[id] = 10.0f;
            }
            FastllmCudaCopyFromHostToDevice(cudaLogits, logits.data(),
                                            logitsBytes);

            Expect(FastllmCudaGreedySampling(
                       cudaLogits, cudaOutput, 1, vocabSize),
                   name + " host-output greedy launch failed");
            int hostOutput = -1;
            FastllmCudaCopyFromDeviceToHost(&hostOutput, cudaOutput,
                                            sizeof(hostOutput));
            Expect(hostOutput == expectedId,
                   name + " host-output greedy selected token " +
                       std::to_string(hostOutput) + " instead of " +
                       std::to_string(expectedId));

            Expect(FastllmCudaGreedySamplingWithFloatOutput(
                       cudaLogits, cudaOutput, cudaFloatOutput,
                       1, vocabSize),
                   name + " GPU-handoff greedy launch failed");
            int handoffOutput = -1;
            float handoffFloatOutput = -1.0f;
            FastllmCudaCopyFromDeviceToHost(&handoffOutput, cudaOutput,
                                            sizeof(handoffOutput));
            FastllmCudaCopyFromDeviceToHost(
                &handoffFloatOutput, cudaFloatOutput,
                sizeof(handoffFloatOutput));
            Expect(handoffOutput == expectedId,
                   name + " GPU-handoff greedy selected token " +
                       std::to_string(handoffOutput) + " instead of " +
                       std::to_string(expectedId));
            Expect(handoffFloatOutput == (float)expectedId,
                   name + " GPU-handoff float token differs from int output");
        };

        runCase("unique maximum", {1234}, 1234);
        runCase("same-CTA tied maximum", {1, 2}, 1);
        runCase("cross-partition tied maximum", {100, 20000}, 100);

        FastllmCudaFree(cudaFloatOutput);
        FastllmCudaFree(cudaOutput);
        FastllmCudaFree(cudaLogits);
    }

    void RunCudaHandoffSamplingRegression() {
        constexpr int batch = 2;
        constexpr int vocabSize = 128;
        constexpr int penaltyTokens = 1;
        const size_t logitsBytes =
            (size_t)batch * vocabSize * sizeof(float);
        FastllmCudaSetDevice(0);

        std::vector<float> logits(batch * vocabSize, -20.0f);
        logits[7] = 10.0f;
        logits[9] = 9.0f;
        logits[vocabSize + 5] = 8.0f;
        logits[vocabSize + 6] = 7.0f;
        std::vector<float> temperatures(batch, 1.0f);
        std::vector<int> topKs(batch, 2);
        std::vector<float> topPs(batch, 0.01f);
        std::vector<int> penaltyIds = {-1, 5};
        std::vector<float> penaltyFactors = {1.0f, 100.0f};

        float *cudaLogits = (float*)FastllmCudaMalloc(logitsBytes);
        float *cudaProbs = (float*)FastllmCudaMalloc(logitsBytes);
        float *cudaTemperatures =
            (float*)FastllmCudaMalloc(batch * sizeof(float));
        int *cudaTopKs = (int*)FastllmCudaMalloc(batch * sizeof(int));
        float *cudaTopPs =
            (float*)FastllmCudaMalloc(batch * sizeof(float));
        int *cudaPenaltyIds =
            (int*)FastllmCudaMalloc(batch * penaltyTokens * sizeof(int));
        float *cudaPenaltyFactors = (float*)FastllmCudaMalloc(
            batch * penaltyTokens * sizeof(float));
        int *cudaOutput = (int*)FastllmCudaMalloc(batch * sizeof(int));
        float *cudaFloatOutput =
            (float*)FastllmCudaMalloc(batch * sizeof(float));
        Expect(cudaLogits != nullptr && cudaProbs != nullptr &&
                   cudaTemperatures != nullptr && cudaTopKs != nullptr &&
                   cudaTopPs != nullptr && cudaPenaltyIds != nullptr &&
                   cudaPenaltyFactors != nullptr && cudaOutput != nullptr &&
                   cudaFloatOutput != nullptr,
               "failed to allocate CUDA handoff sampling buffers");

        FastllmCudaCopyFromHostToDevice(
            cudaLogits, logits.data(), logitsBytes);
        FastllmCudaCopyFromHostToDevice(
            cudaTemperatures, temperatures.data(),
            batch * sizeof(float));
        FastllmCudaCopyFromHostToDevice(
            cudaTopKs, topKs.data(), batch * sizeof(int));
        FastllmCudaCopyFromHostToDevice(
            cudaTopPs, topPs.data(), batch * sizeof(float));
        FastllmCudaCopyFromHostToDevice(
            cudaPenaltyIds, penaltyIds.data(),
            batch * penaltyTokens * sizeof(int));
        FastllmCudaCopyFromHostToDevice(
            cudaPenaltyFactors, penaltyFactors.data(),
            batch * penaltyTokens * sizeof(float));

        Expect(FastllmCudaTopKTopPSamplingToDevice(
                   cudaLogits, cudaProbs,
                   cudaTemperatures, cudaTopKs, cudaTopPs,
                   cudaPenaltyIds, cudaPenaltyFactors, penaltyTokens,
                   cudaOutput, cudaFloatOutput, batch, vocabSize),
               "CUDA handoff top-k/top-p launch failed");
        std::vector<int> output(batch, -1);
        std::vector<float> floatOutput(batch, -1.0f);
        std::vector<float> penalizedLogits(batch * vocabSize);
        FastllmCudaCopyFromDeviceToHost(
            output.data(), cudaOutput, batch * sizeof(int));
        FastllmCudaCopyFromDeviceToHost(
            floatOutput.data(), cudaFloatOutput, batch * sizeof(float));
        FastllmCudaCopyFromDeviceToHost(
            penalizedLogits.data(), cudaLogits, logitsBytes);
        Expect(output[0] == 7 && output[1] == 6,
               "CUDA handoff sampling ignored top-p or repetition penalty");
        Expect(floatOutput[0] == 7.0f && floatOutput[1] == 6.0f,
               "CUDA handoff sampling float tokens differ from int output");
        Expect(std::fabs(penalizedLogits[vocabSize + 5] - 0.08f) <
                   1.0e-6f,
               "CUDA handoff repetition factor was not applied exactly once");

        FastllmCudaFree(cudaFloatOutput);
        FastllmCudaFree(cudaOutput);
        FastllmCudaFree(cudaPenaltyFactors);
        FastllmCudaFree(cudaPenaltyIds);
        FastllmCudaFree(cudaTopPs);
        FastllmCudaFree(cudaTopKs);
        FastllmCudaFree(cudaTemperatures);
        FastllmCudaFree(cudaProbs);
        FastllmCudaFree(cudaLogits);
    }

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

    void RunCpuDeepSeekV4HcPreRegressionCase(int tokens) {
        constexpr int hcMult = 4;
        constexpr int dim = 64;
        constexpr int flatDim = hcMult * dim;
        constexpr int mixHc = (2 + hcMult) * hcMult;
        std::vector<float> inputValues((size_t)tokens * flatDim);
        std::vector<float> fnValues((size_t)mixHc * flatDim);
        std::vector<float> baseValues(mixHc);
        for (size_t i = 0; i < inputValues.size(); i++) {
            inputValues[i] =
                (float)((int)((i * 17 + i / 11) % 97) - 48) / 64.0f;
        }
        for (size_t i = 0; i < fnValues.size(); i++) {
            fnValues[i] =
                (float)((int)((i * 13 + i / 7) % 89) - 44) / 4096.0f;
        }
        for (int i = 0; i < mixHc; i++) {
            baseValues[i] = (float)((i * 5) % 17 - 8) / 32.0f;
        }
        fastllm::Data input(
            fastllm::DataType::BFLOAT16,
            {1, tokens, hcMult, dim});
        input.Allocate();
        uint16_t *inputBits = (uint16_t*)input.cpuData;
        for (size_t i = 0; i < inputValues.size(); i++) {
            inputBits[i] =
                fastllm::Float32ToBFloat16RNEBits(inputValues[i]);
        }
        fastllm::Data hcFn(
            fastllm::DataType::FLOAT32, {mixHc, flatDim}, fnValues);
        fastllm::Data hcScale(
            fastllm::DataType::FLOAT32, {3},
            std::vector<float>({0.75f, -0.5f, 1.25f}));
        fastllm::Data hcBase(
            fastllm::DataType::FLOAT32, {mixHc}, baseValues);

        std::vector<float> roundedInputValues(inputValues.size());
        for (size_t i = 0; i < inputValues.size(); i++) {
            roundedInputValues[i] = fastllm::BFloat16BitsToFloat32(
                inputBits[i]);
        }
        fastllm::Data referenceInput(
            fastllm::DataType::FLOAT32,
            {1, tokens, hcMult, dim}, roundedInputValues);
        fastllm::Data referenceOutput, referencePost, referenceComb;
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4HcPre(
                referenceInput, hcFn, hcScale, hcBase, hcMult, 20,
                1e-6f, 1e-6f, referenceOutput, referencePost,
                referenceComb);
        }
        fastllm::Data parallelOutput, parallelPost, parallelComb;
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4HcPre(
                input, hcFn, hcScale, hcBase, hcMult, 20,
                1e-6f, 1e-6f, parallelOutput, parallelPost,
                parallelComb);
        }
        std::vector<float> referenceOutputValues =
            ToFloatVector(referenceOutput);
        fastllm::Data roundedReferenceOutput(
            fastllm::DataType::BFLOAT16, referenceOutput.dims);
        roundedReferenceOutput.Allocate();
        uint16_t *referenceOutputBits =
            (uint16_t*)roundedReferenceOutput.cpuData;
        for (size_t i = 0; i < referenceOutputValues.size(); i++) {
            referenceOutputBits[i] = fastllm::Float32ToBFloat16RNEBits(
                referenceOutputValues[i]);
        }
        ExpectFloatNear(
            ToFloatVector(roundedReferenceOutput),
            ToFloatVector(parallelOutput),
            0.0f, 0.0f,
            "DeepSeek-V4 CPU HcPre BF16 output fast versus reference");
        ExpectFloatNear(
            ToFloatVector(referencePost), ToFloatVector(parallelPost),
            0.0f, 0.0f,
            "DeepSeek-V4 CPU HcPre post fast versus reference");
        ExpectFloatNear(
            ToFloatVector(referenceComb), ToFloatVector(parallelComb),
            0.0f, 0.0f,
            "DeepSeek-V4 CPU HcPre comb fast versus reference");
    }

    void RunCpuDeepSeekV4HcPreRegression() {
        // Cover both the mix-parallel one-token decode path and the
        // token-partitioned prefill path.
        RunCpuDeepSeekV4HcPreRegressionCase(1);
        RunCpuDeepSeekV4HcPreRegressionCase(73);
    }

    void RunCpuDeepSeekV4HcPreDecodeBenchmark() {
        constexpr int hcMult = 4;
        constexpr int dim = 4096;
        constexpr int flatDim = hcMult * dim;
        constexpr int mixHc = (2 + hcMult) * hcMult;
        constexpr int repeats = 200;
        fastllm::SetThreads(40);

        fastllm::Data input(
            fastllm::DataType::BFLOAT16, {1, 1, hcMult, dim});
        input.Allocate();
        for (int i = 0; i < flatDim; i++) {
            ((uint16_t*)input.cpuData)[i] =
                fastllm::Float32ToBFloat16RNEBits(
                    (float)((i * 17) % 257 - 128) / 256.0f);
        }
        std::vector<float> fnValues((size_t)mixHc * flatDim);
        for (size_t i = 0; i < fnValues.size(); i++) {
            fnValues[i] =
                (float)((int)((i * 13 + i / 7) % 251) - 125) /
                16384.0f;
        }
        fastllm::Data hcFn(
            fastllm::DataType::FLOAT32,
            {mixHc, flatDim}, fnValues);
        fastllm::Data hcScale(
            fastllm::DataType::FLOAT32, {3},
            std::vector<float>({0.75f, -0.5f, 1.25f}));
        fastllm::Data hcBase(
            fastllm::DataType::FLOAT32, {mixHc},
            std::vector<float>(mixHc, 0.125f));
        fastllm::Data output, post, comb;
        {
            ScopedFirstDevice device("cpu");
            for (int i = 0; i < 5; i++) {
                fastllm::DeepSeekV4HcPre(
                    input, hcFn, hcScale, hcBase, hcMult, 20,
                    1e-6f, 1e-6f, output, post, comb);
            }
            auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; i++) {
                fastllm::DeepSeekV4HcPre(
                    input, hcFn, hcScale, hcBase, hcMult, 20,
                    1e-6f, 1e-6f, output, post, comb);
            }
            auto end = std::chrono::steady_clock::now();
            double milliseconds = std::chrono::duration<double, std::milli>(
                end - begin).count() / repeats;
            std::cout << "DeepSeek-V4 CPU HcPre decode benchmark: "
                      << milliseconds << " ms/call, checksum="
                      << ((const uint16_t*)output.cpuData)[dim / 2]
                      << "\n";
        }
    }

    void RunCpuDeepSeekV4WindowKVUpdateRegressionCase(
            fastllm::DataType inputType) {
        constexpr int batch = 2;
        constexpr int sequence = 8;
        constexpr int window = 5;
        constexpr int headDim = 7;
        constexpr int startPos = 7;
        std::vector<float> initialValues(batch * window * headDim);
        std::vector<float> sourceValues(batch * sequence * headDim);
        for (size_t i = 0; i < initialValues.size(); i++) {
            initialValues[i] = (float)((int)(i % 31) - 15) / 8.0f;
        }
        for (size_t i = 0; i < sourceValues.size(); i++) {
            sourceValues[i] = (float)((int)((i * 13 + 5) % 67) - 33) /
                              16.0f;
        }

        fastllm::Data input(inputType, {batch, sequence, headDim});
        input.Allocate();
        std::vector<float> roundedSource(sourceValues.size());
        if (inputType == fastllm::DataType::FLOAT32) {
            memcpy(input.cpuData, sourceValues.data(),
                   sourceValues.size() * sizeof(float));
            roundedSource = sourceValues;
        } else if (inputType == fastllm::DataType::FLOAT16) {
            uint16_t *bits = (uint16_t*)input.cpuData;
            for (size_t i = 0; i < sourceValues.size(); i++) {
                bits[i] = fastllm::float_to_half(sourceValues[i]);
                roundedSource[i] = fastllm::half_to_float(bits[i]);
            }
        } else {
            uint16_t *bits = (uint16_t*)input.cpuData;
            for (size_t i = 0; i < sourceValues.size(); i++) {
                bits[i] = fastllm::Float32ToBFloat16RNEBits(sourceValues[i]);
                roundedSource[i] =
                    fastllm::BFloat16BitsToFloat32(bits[i]);
            }
        }

        fastllm::Data cache(
            fastllm::DataType::FLOAT32,
            {batch, window, headDim}, initialValues);
        std::vector<float> expected = initialValues;
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < sequence; s++) {
                int slot = (startPos + s) % window;
                memcpy(expected.data() +
                           ((uint64_t)b * window + slot) * headDim,
                       roundedSource.data() +
                           ((uint64_t)b * sequence + s) * headDim,
                       headDim * sizeof(float));
            }
        }

        {
            ScopedFirstDevice device("cpu");
            auto *executor =
                (fastllm::Executor*)fastllm::GetExecutor();
            executor->Run(
                "DeepSeekV4UpdateWindowKVCache",
                {{"input", &input}, {"cache", &cache}}, {},
                {{"startPos", startPos}, {"windowSize", window}});
        }
        Expect(cache.dataType == fastllm::DataType::FLOAT32,
               "DeepSeek-V4 CPU window update changed cache dtype.");
        Expect(cache.dims == std::vector<int>({batch, window, headDim}),
               "DeepSeek-V4 CPU window update changed cache shape.");
        Expect(memcmp(cache.cpuData, expected.data(),
                      expected.size() * sizeof(float)) == 0,
               "DeepSeek-V4 CPU window update is not bitwise aligned.");
    }

    void RunCpuDeepSeekV4WindowKVUpdateRegression() {
        RunCpuDeepSeekV4WindowKVUpdateRegressionCase(
            fastllm::DataType::FLOAT32);
        RunCpuDeepSeekV4WindowKVUpdateRegressionCase(
            fastllm::DataType::FLOAT16);
        RunCpuDeepSeekV4WindowKVUpdateRegressionCase(
            fastllm::DataType::BFLOAT16);
    }

    void RunCpuDeepSeekV4ScaleQRatoryRegressionCase(int tokens) {
        constexpr int heads = 7;
        constexpr int dim = 96;
        constexpr int ropeDim = 32;
        std::vector<uint16_t> inputBits((size_t)tokens * heads * dim);
        std::vector<float> referenceValues(inputBits.size());
        uint32_t state = 0x91e10da5u;
        for (size_t i = 0; i < inputBits.size(); i++) {
            state = state * 1664525u + 1013904223u;
            float value = (float)((int32_t)(state % 32749u) - 16374) /
                          2048.0f;
            inputBits[i] = fastllm::Float32ToBFloat16RNEBits(value);
            referenceValues[i] =
                fastllm::BFloat16BitsToFloat32(inputBits[i]);
        }

        fastllm::Data reference(
            fastllm::DataType::FLOAT32,
            {1, tokens, heads, dim}, referenceValues);
        fastllm::Data actual(
            fastllm::DataType::BFLOAT16,
            {1, tokens, heads, dim});
        actual.Allocate();
        memcpy(actual.cpuData, inputBits.data(),
               inputBits.size() * sizeof(uint16_t));

        {
            ScopedFirstDevice device("cpu");
            fastllm::ScaleQRatory(
                reference, 1e-6f, ropeDim, 10000.0f, 137, 4096,
                8.0f, 32, 1);
            fastllm::ScaleQRatory(
                actual, 1e-6f, ropeDim, 10000.0f, 137, 4096,
                8.0f, 32, 1);
        }

        Expect(reference.dataType == fastllm::DataType::BFLOAT16,
               "ScaleQRatory reference output should be BF16.");
        Expect(actual.dataType == fastllm::DataType::BFLOAT16,
               "ScaleQRatory optimized output should be BF16.");
        Expect(reference.Count(0) == actual.Count(0),
               "ScaleQRatory reference and optimized output sizes differ.");
        Expect(memcmp(reference.cpuData, actual.cpuData,
                      inputBits.size() * sizeof(uint16_t)) == 0,
               "DeepSeek-V4 CPU ScaleQRatory BF16 output is not bitwise aligned with the reference path.");
    }

    void RunCpuDeepSeekV4ScaleQRatoryRegression() {
        // One token covers the inline decode path; 373 tokens create enough
        // rows to exercise all active prefill workers.
        RunCpuDeepSeekV4ScaleQRatoryRegressionCase(1);
        RunCpuDeepSeekV4ScaleQRatoryRegressionCase(373);
    }

    void RunCpuDeepSeekV4PreprocessRegression() {
        fastllm::SetThreads(40);

        // More than 4K independent blocks exercise the parallel activation
        // quantizer.  Periodic very large blocks also cover the non-lookup
        // fallback without introducing infinities or NaNs.
        constexpr int quantRows = 129;
        constexpr int quantDim = 4096;
        constexpr uint64_t quantCount =
            (uint64_t)quantRows * quantDim;
        std::vector<uint16_t> quantInput(quantCount);
        uint32_t state = 0x6a09e667u;
        for (uint64_t i = 0; i < quantCount; i++) {
            state = state * 1664525u + 1013904223u;
            uint64_t block = i / 128;
            int exponent = (int)((block * 13) % 31) - 15;
            if (block % 257 == 0) {
                exponent = 100;
            }
            float mantissa =
                (float)((int32_t)(state % 2047u) - 1023) / 1024.0f;
            quantInput[i] = fastllm::Float32ToBFloat16RNEBits(
                std::ldexp(mantissa, exponent));
        }

        std::vector<uint16_t> quantReference(quantCount);
        static const fastllm::FP8E4M3ToFP32Manager fp8;
        for (uint64_t start = 0; start < quantCount; start += 128) {
            float amax = 1e-4f;
            for (uint64_t i = start; i < start + 128; i++) {
                amax = std::max(
                    amax, std::fabs(
                        fastllm::BFloat16BitsToFloat32(quantInput[i])));
            }
            float normalized = amax / 448.0f;
            uint32_t bits;
            memcpy(&bits, &normalized, sizeof(bits));
            int exponent = (int)((bits >> 23) & 0xFF) - 127 +
                           ((bits & ((1u << 23) - 1)) != 0);
            float scale = std::ldexp(1.0f, exponent);
            const uint16_t *lookupRow =
                fastllm::GetFP8E4M3BFloat16QuantizeLookupRow(exponent);
            for (uint64_t i = start; i < start + 128; i++) {
                if (lookupRow != nullptr) {
                    quantReference[i] = lookupRow[quantInput[i]];
                } else {
                    float value = fastllm::BFloat16BitsToFloat32(
                        quantInput[i]);
                    float q = std::max(
                        -448.0f, std::min(448.0f, value / scale));
                    quantReference[i] =
                        fastllm::Float32ToBFloat16RNEBits(
                            fp8.quantizeDequantize(q) * scale);
                }
            }
        }
        std::vector<uint16_t> quantActual(quantCount);
        fastllm::RunCpuDeepSeekV4ActivationQuantization(
            quantInput.data(), quantActual.data(), quantCount);
        Expect(memcmp(
                   quantReference.data(), quantActual.data(),
                   quantCount * sizeof(uint16_t)) == 0,
               "DeepSeek-V4 parallel activation quantization changed BF16 bits.");

        // Match the model's hidden-size norm and cover both separate output
        // storage and the in-place KV normalization call.
        constexpr int normRows = 257;
        constexpr int normChannels = 7168;
        constexpr float normEps = 1e-6f;
        constexpr uint64_t normCount =
            (uint64_t)normRows * normChannels;
        std::vector<uint16_t> normInput(normCount);
        std::vector<float> normWeight(normChannels);
        state = 0xbb67ae85u;
        for (uint64_t i = 0; i < normCount; i++) {
            state = state * 1103515245u + 12345u;
            float value =
                (float)((int32_t)(state % 8191u) - 4095) / 1024.0f;
            normInput[i] =
                fastllm::Float32ToBFloat16RNEBits(value);
        }
        for (int channel = 0; channel < normChannels; channel++) {
            normWeight[channel] =
                0.75f + (float)((channel * 37) % 257) / 512.0f;
        }
        std::vector<uint16_t> normReference(normCount);
        for (int row = 0; row < normRows; row++) {
            const uint16_t *source =
                normInput.data() + (uint64_t)row * normChannels;
            uint16_t *destination =
                normReference.data() + (uint64_t)row * normChannels;
            double sumSquares = 0.0;
            for (int channel = 0; channel < normChannels; channel++) {
                float value =
                    fastllm::BFloat16BitsToFloat32(source[channel]);
                sumSquares += (double)value * value;
            }
            float scale = 1.0f / std::sqrt(
                (float)(sumSquares / normChannels) + normEps);
            for (int channel = 0; channel < normChannels; channel++) {
                float value =
                    fastllm::BFloat16BitsToFloat32(source[channel]);
                destination[channel] =
                    fastllm::Float32ToBFloat16RNEBits(
                        value * scale * normWeight[channel]);
            }
        }

        std::vector<uint16_t> normActual(normCount);
        fastllm::RunCpuDeepSeekV4RMSNormBFloat16(
            normInput.data(), normWeight.data(), normActual.data(),
            normRows, normChannels, normEps);
        Expect(memcmp(
                   normReference.data(), normActual.data(),
                   normCount * sizeof(uint16_t)) == 0,
               "DeepSeek-V4 parallel BF16 RMSNorm changed output bits.");

        std::vector<uint16_t> normInPlace = normInput;
        fastllm::RunCpuDeepSeekV4RMSNormBFloat16(
            normInPlace.data(), normWeight.data(), normInPlace.data(),
            normRows, normChannels, normEps);
        Expect(memcmp(
                   normReference.data(), normInPlace.data(),
                   normCount * sizeof(uint16_t)) == 0,
               "DeepSeek-V4 in-place BF16 RMSNorm changed output bits.");
    }

    void RunCpuDeepSeekV4WoARegression() {
        constexpr int tokens = 37;
        constexpr int groups = 8;
        constexpr int heads = 16;
        constexpr int headDim = 64;
        constexpr int groupDim = heads / groups * headDim;
        constexpr int oRank = 257;
        constexpr int inputStride = heads * headDim;
        constexpr int outputStride = groups * oRank;

        fastllm::SetThreads(40);
        fastllm::Data input(
            fastllm::DataType::BFLOAT16,
            {1, tokens, heads, headDim});
        input.Allocate();
        uint16_t *inputBits = (uint16_t*)input.cpuData;
        uint32_t inputState = 0x7f4a7c15u;
        for (uint64_t i = 0; i < input.Count(0); i++) {
            inputState = inputState * 1664525u + 1013904223u;
            float value =
                (float)((int32_t)(inputState % 4093u) - 2046) / 512.0f;
            inputBits[i] = fastllm::Float32ToBFloat16RNEBits(value);
        }

        fastllm::Data weight(
            fastllm::DataType::FLOAT16,
            {groups, oRank, groupDim});
        weight.Allocate();
        uint16_t *weightBits = (uint16_t*)weight.cpuData;
        uint32_t weightState = 0x243f6a88u;
        for (uint64_t i = 0; i < weight.Count(0); i++) {
            weightState = weightState * 1103515245u + 12345u;
            float value =
                (float)((int32_t)(weightState % 2039u) - 1019) / 4096.0f;
            weightBits[i] = fastllm::float_to_half(value);
        }

        // Reproduce the old prefill path exactly: materialize token-major FP32,
        // run each group through the same Linear kernel sequentially, scatter to
        // token-major FP32, then round to BF16.
        std::vector<float> inputFloat(input.Count(0));
        for (uint64_t i = 0; i < input.Count(0); i++) {
            inputFloat[i] = fastllm::BFloat16BitsToFloat32(inputBits[i]);
        }
        std::vector<float> referenceFloat(
            (uint64_t)tokens * outputStride, 0.0f);
        std::vector<float> groupInput((uint64_t)tokens * groupDim);
        std::vector<float> groupOutput((uint64_t)tokens * oRank);
        fastllm::AliveThreadPool *pool = fastllm::GetAlivePool();
        int threadCount = (int)pool->threads.size();
        for (int group = 0; group < groups; group++) {
            for (int token = 0; token < tokens; token++) {
                memcpy(
                    groupInput.data() + (uint64_t)token * groupDim,
                    inputFloat.data() + (uint64_t)token * inputStride +
                        (uint64_t)group * groupDim,
                    (uint64_t)groupDim * sizeof(float));
            }
            fastllm::RunLinearFloat32Float16(
                groupInput.data(),
                weightBits + (uint64_t)group * oRank * groupDim,
                groupOutput.data(), nullptr,
                tokens, groupDim, oRank, pool, 0, threadCount);
            for (int token = 0; token < tokens; token++) {
                memcpy(
                    referenceFloat.data() +
                        (uint64_t)token * outputStride +
                        (uint64_t)group * oRank,
                    groupOutput.data() + (uint64_t)token * oRank,
                    (uint64_t)oRank * sizeof(float));
            }
        }
        std::vector<uint16_t> referenceBits(referenceFloat.size());
        for (size_t i = 0; i < referenceFloat.size(); i++) {
            referenceBits[i] =
                fastllm::Float32ToBFloat16RNEBits(referenceFloat[i]);
        }

        fastllm::Data output;
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4WoA(
                input, weight, groups, oRank, output);
        }
        Expect(output.dataType == fastllm::DataType::BFLOAT16,
               "DeepSeek-V4 CPU WoA output should be BF16.");
        Expect(output.dims == std::vector<int>({1, tokens, outputStride}),
               "DeepSeek-V4 CPU WoA output shape mismatch.");
        Expect(output.Count(0) == referenceBits.size(),
               "DeepSeek-V4 CPU WoA output size mismatch.");
        Expect(memcmp(output.cpuData, referenceBits.data(),
                      referenceBits.size() * sizeof(uint16_t)) == 0,
               "DeepSeek-V4 CPU WoA global Linear queue changed BF16 output bits.");

#ifdef USE_NUMAS
        if (fastllm::HasDeviceType("numa")) {
            fastllm::Data numaOutput;
            {
                ScopedFirstDevice device("numa");
                fastllm::DeepSeekV4WoA(
                    input, weight, groups, oRank, numaOutput);
            }
            Expect(
                numaOutput.dataType == fastllm::DataType::BFLOAT16 &&
                    numaOutput.dims ==
                        std::vector<int>({1, tokens, outputStride}),
                "DeepSeek-V4 NUMA WoA prefill output metadata mismatch.");
            Expect(memcmp(
                       numaOutput.cpuData, referenceBits.data(),
                       referenceBits.size() * sizeof(uint16_t)) == 0,
                   "DeepSeek-V4 NUMA WoA prefill changed BF16 output bits.");
            Expect(weight.cpuData == nullptr,
                   "DeepSeek-V4 NUMA WoA did not release the source weight.");
            Expect(!weight.numasData.empty() &&
                       std::all_of(
                           weight.numasData.begin(), weight.numasData.end(),
                           [](const uint8_t *shard) {
                               return shard != nullptr;
                           }),
                   "DeepSeek-V4 NUMA WoA did not create every weight shard.");

            fastllm::Data decodeInput(
                fastllm::DataType::BFLOAT16,
                {1, 1, heads, headDim});
            decodeInput.Allocate();
            memcpy(
                decodeInput.cpuData, input.cpuData,
                (size_t)inputStride * sizeof(uint16_t));
            fastllm::Data decodeOutput;
            {
                ScopedFirstDevice device("numa");
                fastllm::DeepSeekV4WoA(
                    decodeInput, weight, groups, oRank, decodeOutput);
            }
            Expect(
                decodeOutput.dataType == fastllm::DataType::BFLOAT16 &&
                    decodeOutput.dims ==
                        std::vector<int>({1, 1, outputStride}),
                "DeepSeek-V4 NUMA WoA decode output metadata mismatch.");
            Expect(memcmp(
                       decodeOutput.cpuData, referenceBits.data(),
                       (size_t)outputStride * sizeof(uint16_t)) == 0,
                   "DeepSeek-V4 NUMA WoA decode changed BF16 output bits.");
        }
#endif
    }

    void RunCpuDeepSeekV4SparseAttentionRegression() {
        fastllm::Data q(
            fastllm::DataType::FLOAT32, {1, 1, 1, 4},
            std::vector<float>({1.0f, 1.0f, 1.0f, 1.0f}));
        fastllm::Data kv(
            fastllm::DataType::FLOAT32, {1, 1, 4},
            std::vector<float>({
                std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f, 3.0f
            }));
        fastllm::Data sink(
            fastllm::DataType::FLOAT32, {1}, std::vector<float>({0.0f}));
        fastllm::Data output;
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4SparseAttention(
                q, kv, sink, 1, 2, 10000.0f, 0, 0.5f, output,
                0, 0, 1.0f, 32, 1, 0);
        }
        for (float value : ToFloatVector(output)) {
            Expect(std::isfinite(value) && value == 0.0f,
                   "DeepSeek-V4 CPU sparse attention did not skip a "
                   "non-finite score.");
        }

        fastllm::Data fusedQ(
            fastllm::DataType::BFLOAT16, {1, 2, 1, 4},
            std::vector<float>(8, 0.0f));
        fastllm::Data fusedKv(
            fastllm::DataType::BFLOAT16, {1, 2, 4},
            std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f,
                                5.0f, 6.0f, 7.0f, 8.0f}));
        fastllm::Data fusedSink(
            fastllm::DataType::FLOAT32, {1},
            std::vector<float>({
                -std::numeric_limits<float>::infinity()
            }));
        fastllm::Data fusedOutput;
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4SparseAttention(
                fusedQ, fusedKv, fusedSink, 2, 2, 10000.0f, 0, 1.0f,
                fusedOutput, 0, 0, 1.0f, 32, 1, 0);
        }
        float c = std::cos(1.0f), sn = std::sin(1.0f);
        std::vector<float> fusedExpected = {
            1.0f, 2.0f, 3.0f, 4.0f,
            3.0f, 4.0f, 5.0f * c + 6.0f * sn,
            -5.0f * sn + 6.0f * c
        };
        Expect(fusedOutput.dataType == fastllm::DataType::BFLOAT16 &&
                   fusedOutput.cpuData != nullptr,
               "DeepSeek-V4 fused sparse output should be CPU BF16.");
        const uint16_t *fusedBits =
            reinterpret_cast<const uint16_t *>(fusedOutput.cpuData);
        for (int i = 0; i < (int)fusedExpected.size(); i++) {
            Expect(fusedBits[i] ==
                       fastllm::Float32ToBFloat16RNEBits(fusedExpected[i]),
                   "DeepSeek-V4 fused sparse output mismatch at index " +
                       std::to_string(i));
        }

        constexpr int batch = 2;
        constexpr int sequence = 4;
        constexpr int heads = 3;
        constexpr int dimension = 8;
        std::vector<float> qValues(batch * sequence * heads * dimension);
        std::vector<float> kvValues(batch * sequence * dimension);
        for (int i = 0; i < (int)qValues.size(); i++) {
            qValues[i] = (float)((i * 17) % 41 - 20) / 23.0f;
        }
        for (int i = 0; i < (int)kvValues.size(); i++) {
            kvValues[i] = (float)((i * 13) % 37 - 18) / 19.0f;
        }
        fastllm::Data finiteQ(
            fastllm::DataType::FLOAT32,
            {batch, sequence, heads, dimension}, qValues);
        fastllm::Data finiteKv(
            fastllm::DataType::FLOAT32,
            {batch, sequence, dimension}, kvValues);
        fastllm::Data finiteSink(
            fastllm::DataType::FLOAT32, {heads},
            std::vector<float>({-0.25f, 0.0f, 0.25f}));
        fastllm::Data serialOutput, parallelOutput;
        const char *oldDisable = std::getenv(
            "FASTLLM_DSV4_DISABLE_CPU_SPARSE_PREFILL_PARALLEL");
        std::string oldDisableValue = oldDisable == nullptr ? "" : oldDisable;
        setenv("FASTLLM_DSV4_DISABLE_CPU_SPARSE_PREFILL_PARALLEL", "1", 1);
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4SparseAttention(
                finiteQ, finiteKv, finiteSink, sequence, 4, 10000.0f,
                0, 0.5f, serialOutput, 0, 0, 1.0f, 32, 1, 0);
        }
        unsetenv("FASTLLM_DSV4_DISABLE_CPU_SPARSE_PREFILL_PARALLEL");
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4SparseAttention(
                finiteQ, finiteKv, finiteSink, sequence, 4, 10000.0f,
                0, 0.5f, parallelOutput, 0, 0, 1.0f, 32, 1, 0);
        }
        if (oldDisable != nullptr) {
            setenv("FASTLLM_DSV4_DISABLE_CPU_SPARSE_PREFILL_PARALLEL",
                   oldDisableValue.c_str(), 1);
        }
        ExpectFloatNear(
            ToFloatVector(serialOutput), ToFloatVector(parallelOutput),
            0.0f, 0.0f,
            "DeepSeek-V4 CPU sparse attention serial versus parallel");

        // The compressed index tensor must be consumed as a real gather, not
        // merely computed and then ignored.  With zero queries and no sink the
        // selected live/compressed values have uniform attention weights.
        fastllm::Data gatherQ(
            fastllm::DataType::FLOAT32, {1, 4, 1, 4},
            std::vector<float>(16, 0.0f));
        fastllm::Data gatherKv(
            fastllm::DataType::FLOAT32, {1, 6, 4},
            std::vector<float>({
                1.0f, 0.0f, 0.0f, 0.0f,
                2.0f, 0.0f, 0.0f, 0.0f,
                3.0f, 0.0f, 0.0f, 0.0f,
                4.0f, 0.0f, 0.0f, 0.0f,
                10.0f, 0.0f, 0.0f, 0.0f,
                20.0f, 0.0f, 0.0f, 0.0f
            }));
        fastllm::Data gatherTopK = MakeIntTensor(
            {1, 4, 1}, {-1, 0, 0, 1});
        fastllm::Data gatherOutput;
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4SparseAttention(
                gatherQ, gatherKv, fusedSink, 1, 2, 10000.0f, 0,
                1.0f, gatherOutput, 2, 0, 1.0f, 32, 1, 0,
                &gatherTopK);
        }
        ExpectFloatNear(
            {1.0f, 0.0f, 0.0f, 0.0f,
             6.0f, 0.0f, 0.0f, 0.0f,
             6.5f, 0.0f, 0.0f, 0.0f,
             12.0f, 0.0f, 0.0f, 0.0f},
            ToFloatVector(gatherOutput), 0.0f, 0.0f,
            "DeepSeek-V4 CPU sparse attention compressed top-k gather");

        // Exercise the production top-k=512 limit beyond the 2048-token
        // crossover (compression ratio 4).  The monotonically increasing
        // prepared keys also verify score ordering, not just output shape.
        constexpr int indexSequence = 2080;
        constexpr int indexHeads = 1;
        constexpr int indexDimension = 32;
        constexpr int indexBlocks = indexSequence / 4;
        constexpr int indexTopK = 512;
        std::vector<float> indexQ(
            (size_t)indexSequence * indexHeads * indexDimension, 0.0f);
        for (int token = 0; token < indexSequence; token++) {
            indexQ[(size_t)token * indexDimension] = 1.0f;
        }
        std::vector<float> indexKv((size_t)indexBlocks * indexDimension);
        for (int block = 0; block < indexBlocks; block++) {
            std::fill(indexKv.begin() + (size_t)block * indexDimension,
                      indexKv.begin() + (size_t)(block + 1) * indexDimension,
                      (float)(block + 1));
        }
        fastllm::Data indexQData(
            fastllm::DataType::FLOAT32,
            {1, indexSequence, indexHeads, indexDimension}, indexQ);
        fastllm::Data indexWeights(
            fastllm::DataType::FLOAT32,
            {1, indexSequence, indexHeads},
            std::vector<float>(indexSequence * indexHeads, 1.0f));
        fastllm::Data indexKvData(
            fastllm::DataType::FLOAT32,
            {1, indexBlocks, indexDimension}, indexKv);
        fastllm::Data indexTopKData;
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                indexQData, indexWeights, indexKvData, indexTopK, 4,
                2, 10000.0f, 0, 0, 1.0f, 32, 1, indexTopKData);
        }
        Expect(indexTopKData.dims ==
                   std::vector<int>({1, indexSequence, indexTopK}),
               "DeepSeek-V4 CPU indexer top-k output shape mismatch.");
        std::vector<int32_t> indexValues = ToIntVector(indexTopKData);
        for (int token = 0; token < indexSequence; token++) {
            int available = (token + 1) / 4;
            int keep = std::min(available, indexTopK);
            for (int i = 0; i < indexTopK; i++) {
                int32_t expected = i < keep ? available - 1 - i : -1;
                int32_t actual =
                    indexValues[(size_t)token * indexTopK + i];
                Expect(actual == expected,
                       "DeepSeek-V4 CPU indexer causal top-k mismatch at "
                       "token " + std::to_string(token) + ", rank " +
                       std::to_string(i) + ": expected " +
                       std::to_string(expected) + ", got " +
                       std::to_string(actual));
            }
        }

        // Hit the production 64-head x 128-dim SIMD scorer and compare its
        // selected indices with the generic dot-product fallback.
        constexpr int simdSequence = 40;
        constexpr int simdHeads = 64;
        constexpr int simdDimension = 128;
        constexpr int simdBlocks = simdSequence / 4;
        constexpr int simdTopK = 5;
        std::vector<float> simdQ(
            (size_t)simdSequence * simdHeads * simdDimension, 0.0f);
        for (int token = 0; token < simdSequence; token++) {
            for (int head = 0; head < simdHeads; head++) {
                simdQ[((size_t)token * simdHeads + head) *
                      simdDimension] = 1.0f;
            }
        }
        std::vector<float> simdKv((size_t)simdBlocks * simdDimension);
        for (int block = 0; block < simdBlocks; block++) {
            std::fill(simdKv.begin() + (size_t)block * simdDimension,
                      simdKv.begin() + (size_t)(block + 1) * simdDimension,
                      (float)(block + 1));
        }
        fastllm::Data simdQData(
            fastllm::DataType::FLOAT32,
            {1, simdSequence, simdHeads, simdDimension}, simdQ);
        fastllm::Data simdWeights(
            fastllm::DataType::FLOAT32,
            {1, simdSequence, simdHeads},
            std::vector<float>(simdSequence * simdHeads, 1.0f));
        fastllm::Data simdKvData(
            fastllm::DataType::FLOAT32,
            {1, simdBlocks, simdDimension}, simdKv);
        fastllm::Data fallbackTopK, simdTopKData;
        const char *oldHeadSimd = std::getenv(
            "FASTLLM_DSV4_DISABLE_CPU_INDEXER_HEAD_SIMD");
        std::string oldHeadSimdValue =
            oldHeadSimd == nullptr ? "" : oldHeadSimd;
        setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_HEAD_SIMD", "1", 1);
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                simdQData, simdWeights, simdKvData, simdTopK, 4,
                64, 10000.0f, 0, 0, 1.0f, 32, 1, fallbackTopK);
        }
        unsetenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_HEAD_SIMD");
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                simdQData, simdWeights, simdKvData, simdTopK, 4,
                64, 10000.0f, 0, 0, 1.0f, 32, 1, simdTopKData);
        }
        if (oldHeadSimd != nullptr) {
            setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_HEAD_SIMD",
                   oldHeadSimdValue.c_str(), 1);
        }
        ExpectIntEqual(
            ToIntVector(fallbackTopK), ToIntVector(simdTopKData),
            "DeepSeek-V4 CPU indexer 64-head SIMD versus fallback");
        std::vector<int32_t> simdIndices = ToIntVector(simdTopKData);
        for (int token = 0; token < simdSequence; token++) {
            int available = (token + 1) / 4;
            int keep = std::min(available, simdTopK);
            for (int rank = 0; rank < simdTopK; rank++) {
                int32_t expected = rank < keep ?
                    available - 1 - rank : -1;
                Expect(simdIndices[(size_t)token * simdTopK + rank] ==
                           expected,
                       "DeepSeek-V4 CPU indexer 64-head SIMD ranking "
                       "mismatch at token " + std::to_string(token) +
                       ", rank " + std::to_string(rank));
            }
        }

        std::vector<float> mixedQ(simdQ.size());
        std::vector<float> mixedWeights(
            (size_t)simdSequence * simdHeads);
        std::vector<float> mixedKv(simdKv.size());
        for (int token = 0; token < simdSequence; token++) {
            for (int head = 0; head < simdHeads; head++) {
                mixedWeights[(size_t)token * simdHeads + head] =
                    (float)((token * 3 + head * 5) % 17 - 8) / 9.0f;
                for (int d = 0; d < simdDimension; d++) {
                    mixedQ[((size_t)token * simdHeads + head) *
                           simdDimension + d] =
                        (float)((token * 3 + head * 7 + d * 11) % 29 - 14) /
                        17.0f;
                }
            }
        }
        for (int block = 0; block < simdBlocks; block++) {
            for (int d = 0; d < simdDimension; d++) {
                mixedKv[(size_t)block * simdDimension + d] =
                    (float)((block * 13 + d * 7) % 31 - 15) / 16.0f;
            }
        }
        fastllm::Data mixedQData(
            fastllm::DataType::FLOAT32,
            {1, simdSequence, simdHeads, simdDimension}, mixedQ);
        fastllm::Data mixedWeightData(
            fastllm::DataType::FLOAT32,
            {1, simdSequence, simdHeads}, mixedWeights);
        fastllm::Data mixedKvData(
            fastllm::DataType::FLOAT32,
            {1, simdBlocks, simdDimension}, mixedKv);
        fastllm::Data mixedFallbackTopK, mixedSimdTopK;
        setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_HEAD_SIMD", "1", 1);
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                mixedQData, mixedWeightData, mixedKvData, simdTopK, 4,
                64, 10000.0f, 0, 0, 1.0f, 32, 1,
                mixedFallbackTopK);
        }
        unsetenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_HEAD_SIMD");
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                mixedQData, mixedWeightData, mixedKvData, simdTopK, 4,
                64, 10000.0f, 0, 0, 1.0f, 32, 1,
                mixedSimdTopK);
        }
        if (oldHeadSimd != nullptr) {
            setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_HEAD_SIMD",
                   oldHeadSimdValue.c_str(), 1);
        }
        ExpectIntEqual(
            ToIntVector(mixedFallbackTopK), ToIntVector(mixedSimdTopK),
            "DeepSeek-V4 CPU indexer mixed 64-head SIMD versus fallback");

        // The fused path converts and preprocesses each token in the scoring
        // worker instead of materializing a full FP32 Q tensor first.  It must
        // preserve the exact selected indices of the original two-pass path.
        const char *oldFusedPrep = std::getenv(
            "FASTLLM_DSV4_DISABLE_CPU_INDEXER_FUSED_PREP");
        std::string oldFusedPrepValue =
            oldFusedPrep == nullptr ? "" : oldFusedPrep;
        fastllm::Data mixedTwoPassTopK, mixedFusedTopK;
        setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_FUSED_PREP", "1", 1);
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                mixedQData, mixedWeightData, mixedKvData, simdTopK, 4,
                64, 10000.0f, 0, 0, 1.0f, 32, 1,
                mixedTwoPassTopK);
        }
        unsetenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_FUSED_PREP");
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                mixedQData, mixedWeightData, mixedKvData, simdTopK, 4,
                64, 10000.0f, 0, 0, 1.0f, 32, 1,
                mixedFusedTopK);
        }
        if (oldFusedPrep != nullptr) {
            setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_FUSED_PREP",
                   oldFusedPrepValue.c_str(), 1);
        }
        ExpectIntEqual(
            ToIntVector(mixedTwoPassTopK), ToIntVector(mixedFusedTopK),
            "DeepSeek-V4 CPU indexer fused versus two-pass preprocessing");

        // Production indexer activations are BF16.  Exercise the vectorized
        // raw-row loader as well as the fused preprocessing/scoring handoff
        // with all three inputs in their production dtype.
        auto makeBFloat16Tensor = [](
                const std::vector<int> &dims,
                const std::vector<float> &values) {
            fastllm::Data tensor(fastllm::DataType::BFLOAT16, dims);
            tensor.Allocate();
            uint16_t *bits = (uint16_t*)tensor.cpuData;
            for (size_t i = 0; i < values.size(); i++) {
                bits[i] = fastllm::Float32ToBFloat16RNEBits(values[i]);
            }
            return tensor;
        };
        fastllm::Data mixedQBFloat = makeBFloat16Tensor(
            {1, simdSequence, simdHeads, simdDimension}, mixedQ);
        fastllm::Data mixedWeightBFloat = makeBFloat16Tensor(
            {1, simdSequence, simdHeads}, mixedWeights);
        fastllm::Data mixedKvBFloat = makeBFloat16Tensor(
            {1, simdBlocks, simdDimension}, mixedKv);
        fastllm::Data mixedBFloatTwoPassTopK, mixedBFloatFusedTopK;
        setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_FUSED_PREP", "1", 1);
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                mixedQBFloat, mixedWeightBFloat, mixedKvBFloat,
                simdTopK, 4, 64, 10000.0f, 0, 0, 1.0f, 32, 1,
                mixedBFloatTwoPassTopK);
        }
        unsetenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_FUSED_PREP");
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                mixedQBFloat, mixedWeightBFloat, mixedKvBFloat,
                simdTopK, 4, 64, 10000.0f, 0, 0, 1.0f, 32, 1,
                mixedBFloatFusedTopK);
        }
        if (oldFusedPrep != nullptr) {
            setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_FUSED_PREP",
                   oldFusedPrepValue.c_str(), 1);
        }
        ExpectIntEqual(
            ToIntVector(mixedBFloatTwoPassTopK),
            ToIntVector(mixedBFloatFusedTopK),
            "DeepSeek-V4 CPU BF16 indexer fused versus two-pass preprocessing");

        // Compare the AVX-512 FP4 preprocessing path with the original scalar
        // log2/pow plus exhaustive E2M1 nearest-value implementation.  The
        // mixed signs and non-power-of-two inputs exercise scale selection,
        // midpoint direction and signed dequantization before top-k scoring.
        const char *oldFp4Simd = std::getenv(
            "FASTLLM_DSV4_DISABLE_CPU_INDEXER_FP4_SIMD");
        std::string oldFp4SimdValue =
            oldFp4Simd == nullptr ? "" : oldFp4Simd;
        fastllm::Data mixedFp4ReferenceTopK, mixedFp4SimdTopK;
        setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_FP4_SIMD", "1", 1);
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                mixedQData, mixedWeightData, mixedKvData, simdTopK, 4,
                64, 10000.0f, 0, 0, 1.0f, 32, 1,
                mixedFp4ReferenceTopK);
        }
        unsetenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_FP4_SIMD");
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4IndexerTopK(
                mixedQData, mixedWeightData, mixedKvData, simdTopK, 4,
                64, 10000.0f, 0, 0, 1.0f, 32, 1,
                mixedFp4SimdTopK);
        }
        if (oldFp4Simd != nullptr) {
            setenv("FASTLLM_DSV4_DISABLE_CPU_INDEXER_FP4_SIMD",
                   oldFp4SimdValue.c_str(), 1);
        }
        ExpectIntEqual(
            ToIntVector(mixedFp4ReferenceTopK),
            ToIntVector(mixedFp4SimdTopK),
            "DeepSeek-V4 CPU indexer FP4 SIMD versus scalar reference");
    }

    void RunCpuDeepSeekV4SparseDecodeCachedRegression() {
        constexpr int batch = 1;
        constexpr int heads = 64;
        constexpr int dimension = 192;
        constexpr int windowSize = 128;
        constexpr int compressedCount = 520;
        constexpr int startPos = 2048;
        constexpr int ropeDim = 64;
        constexpr float ropeBase = 10000.0f;
        constexpr float softmaxScale = 0.07216878f;
        constexpr int originalSeqLen = 4096;
        constexpr float ropeFactor = 4.0f;
        constexpr int betaFast = 32;
        constexpr int betaSlow = 1;

        std::vector<float> qInput((size_t)heads * dimension);
        std::vector<float> windowValues(
            (size_t)windowSize * dimension);
        std::vector<float> compressedValues(
            (size_t)compressedCount * dimension);
        std::vector<float> sinkValues(heads);
        for (size_t i = 0; i < qInput.size(); i++) {
            qInput[i] = (float)((int)(i * 17 % 73) - 36) / 29.0f;
        }
        for (size_t i = 0; i < windowValues.size(); i++) {
            windowValues[i] =
                (float)((int)(i * 13 % 61) - 30) / 31.0f;
        }
        for (size_t i = 0; i < compressedValues.size(); i++) {
            compressedValues[i] =
                (float)((int)(i * 19 % 67) - 33) / 37.0f;
        }
        for (int head = 0; head < heads; head++) {
            sinkValues[head] =
                (float)((head * 7) % 19 - 9) / 11.0f;
        }

        fastllm::Data q(
            fastllm::DataType::BFLOAT16,
            {batch, 1, heads, dimension}, qInput);
        fastllm::Data windowKV(
            fastllm::DataType::BFLOAT16,
            {batch, windowSize, dimension}, windowValues);
        windowValues = ToFloatVector(windowKV);
        fastllm::Data compressedKV(
            fastllm::DataType::BFLOAT16,
            {batch, compressedCount, dimension}, compressedValues);
        compressedValues = ToFloatVector(compressedKV);
        fastllm::Data sink(
            fastllm::DataType::FLOAT32, {heads}, sinkValues);
        std::vector<int32_t> topKValues = {-1, compressedCount};
        for (int i = 0; i < 512; i++) {
            topKValues.push_back(compressedCount - 1 - i);
        }
        fastllm::Data topK = MakeIntTensor(
            {batch, 1, (int)topKValues.size()}, topKValues);

        std::vector<float> qValues = ToFloatVector(q);
        std::vector<int> indices;
        int ringPosition = startPos % windowSize;
        for (int i = ringPosition + 1; i < windowSize; i++) {
            indices.push_back(i);
        }
        for (int i = 0; i <= ringPosition; i++) {
            indices.push_back(i);
        }
        for (int32_t index : topKValues) {
            if (index >= 0 && index < compressedCount) {
                indices.push_back(windowSize + index);
            }
        }

        std::vector<float> reference((size_t)heads * dimension, 0.0f);
        std::vector<float> scores(indices.size());
        for (int head = 0; head < heads; head++) {
            const float *qrow =
                qValues.data() + (size_t)head * dimension;
            float maxScore =
                -std::numeric_limits<float>::infinity();
            for (int k = 0; k < (int)indices.size(); k++) {
                int index = indices[k];
                const float *kvrow = index < windowSize ?
                    windowValues.data() + (size_t)index * dimension :
                    compressedValues.data() +
                        (size_t)(index - windowSize) * dimension;
                double dot = 0.0;
                for (int d = 0; d < dimension; d++) {
                    dot += (double)qrow[d] * kvrow[d];
                }
                scores[k] = (float)dot * softmaxScale;
                maxScore = std::max(maxScore, scores[k]);
            }
            float safeMax =
                std::isfinite(maxScore) ? maxScore : 0.0f;
            double denominator =
                std::exp((double)sinkValues[head] - safeMax);
            for (float score : scores) {
                denominator += std::exp((double)score - safeMax);
            }
            float *outputRow =
                reference.data() + (size_t)head * dimension;
            for (int k = 0; k < (int)indices.size(); k++) {
                float weight = (float)(
                    std::exp((double)scores[k] - safeMax) /
                    std::max(denominator, 1e-30));
                int index = indices[k];
                const float *kvrow = index < windowSize ?
                    windowValues.data() + (size_t)index * dimension :
                    compressedValues.data() +
                        (size_t)(index - windowSize) * dimension;
                for (int d = 0; d < dimension; d++) {
                    outputRow[d] += weight * kvrow[d];
                }
            }
        }

        std::vector<float> invFreq;
        for (int i = 0; i < ropeDim; i += 2) {
            invFreq.push_back(
                1.0f / std::pow(ropeBase, (float)i / ropeDim));
        }
        float lowF = ropeDim * std::log(
            (float)originalSeqLen /
            (betaFast * 2.0f * (float)M_PI)) /
            (2.0f * std::log(ropeBase));
        float highF = ropeDim * std::log(
            (float)originalSeqLen /
            (betaSlow * 2.0f * (float)M_PI)) /
            (2.0f * std::log(ropeBase));
        int low = std::max((int)std::floor(lowF), 0);
        int high = std::min((int)std::ceil(highF), ropeDim - 1);
        if (low == high) {
            high++;
        }
        for (int i = 0; i < (int)invFreq.size(); i++) {
            float ramp = std::max(
                0.0f,
                std::min(1.0f, ((float)i - low) / (high - low)));
            float smooth = 1.0f - ramp;
            invFreq[i] = invFreq[i] / ropeFactor * (1.0f - smooth) +
                         invFreq[i] * smooth;
        }
        int rotaryOffset = dimension - ropeDim;
        for (int head = 0; head < heads; head++) {
            float *row = reference.data() +
                (size_t)head * dimension + rotaryOffset;
            for (int i = 0; i < ropeDim; i += 2) {
                float angle = startPos * invFreq[i / 2];
                float cosine = std::cos(angle);
                float sine = -std::sin(angle);
                float a = row[i], b = row[i + 1];
                row[i] = a * cosine - b * sine;
                row[i + 1] = a * sine + b * cosine;
            }
        }
        std::vector<uint16_t> referenceBits(reference.size());
        for (size_t i = 0; i < reference.size(); i++) {
            referenceBits[i] =
                fastllm::Float32ToBFloat16RNEBits(reference[i]);
        }

        fastllm::Data output;
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4SparseAttentionDecodeCached(
                q, windowKV, compressedKV, sink, windowSize, startPos,
                compressedCount, ropeDim, ropeBase, softmaxScale, output,
                originalSeqLen, ropeFactor, betaFast, betaSlow, &topK);
        }
        Expect(output.dataType == fastllm::DataType::BFLOAT16 &&
                   output.dims ==
                       std::vector<int>({batch, 1, heads, dimension}) &&
                   output.cpuData != nullptr,
               "DeepSeek-V4 CPU cached sparse decode output metadata "
               "mismatch.");
        Expect(memcmp(output.cpuData, referenceBits.data(),
                      referenceBits.size() * sizeof(uint16_t)) == 0,
               "DeepSeek-V4 CPU cached sparse decode changed BF16 output "
               "bits versus the scalar reference.");
    }

    void RunCpuDeepSeekV4SparseDecodeCachedBenchmarkCase(
            int compressedCount) {
        constexpr int batch = 1;
        constexpr int heads = 64;
        constexpr int dimension = 512;
        constexpr int windowSize = 128;
        constexpr int startPos = 2048;
        constexpr int ropeDim = 64;
        constexpr int repeats = 100;
        fastllm::Data q(
            fastllm::DataType::BFLOAT16,
            {batch, 1, heads, dimension});
        fastllm::Data windowKV(
            fastllm::DataType::BFLOAT16,
            {batch, windowSize, dimension});
        fastllm::Data compressedKV(
            fastllm::DataType::BFLOAT16,
            {batch, compressedCount, dimension});
        q.Allocate();
        windowKV.Allocate();
        compressedKV.Allocate();
        auto fillBFloat = [](fastllm::Data &data, uint32_t state) {
            uint16_t *values = (uint16_t*)data.cpuData;
            for (uint64_t i = 0; i < data.Count(0); i++) {
                state = state * 1664525u + 1013904223u;
                values[i] = fastllm::Float32ToBFloat16RNEBits(
                    (float)((int32_t)(state % 4093u) - 2046) / 4096.0f);
            }
        };
        fillBFloat(q, 0x243f6a88u);
        fillBFloat(windowKV, 0x85a308d3u);
        fillBFloat(compressedKV, 0x13198a2eu);
        std::vector<float> sinkValues(heads);
        for (int head = 0; head < heads; head++) {
            sinkValues[head] =
                (float)((head * 7) % 19 - 9) / 11.0f;
        }
        fastllm::Data sink(
            fastllm::DataType::FLOAT32, {heads}, sinkValues);
        const int topKCount = std::min(512, compressedCount);
        std::vector<int32_t> topKValues(topKCount);
        std::iota(topKValues.begin(), topKValues.end(), 0);
        fastllm::Data topK = MakeIntTensor(
            {batch, 1, topKCount}, topKValues);
        fastllm::Data output;
        {
            ScopedFirstDevice device("cpu");
            for (int repeat = 0; repeat < 5; repeat++) {
                fastllm::DeepSeekV4SparseAttentionDecodeCached(
                    q, windowKV, compressedKV, sink, windowSize,
                    startPos, compressedCount, ropeDim, 160000.0f,
                    0.04419417f, output, 65536, 16.0f, 32, 1,
                    &topK);
            }
            auto begin = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < repeats; repeat++) {
                fastllm::DeepSeekV4SparseAttentionDecodeCached(
                    q, windowKV, compressedKV, sink, windowSize,
                    startPos, compressedCount, ropeDim, 160000.0f,
                    0.04419417f, output, 65536, 16.0f, 32, 1,
                    &topK);
            }
            auto end = std::chrono::steady_clock::now();
            double milliseconds =
                std::chrono::duration<double, std::milli>(
                    end - begin).count() / repeats;
            std::cout
                << "DeepSeek-V4 CPU sparse decode benchmark: compressed="
                << compressedCount << ", candidates="
                << windowSize + topKCount << ", " << milliseconds
                << " ms/call, checksum="
                << ((const uint16_t*)output.cpuData)[dimension / 2]
                << "\n";
        }
    }

    void RunCpuDeepSeekV4SparseDecodeCachedBenchmark() {
        fastllm::SetThreads(40);
        RunCpuDeepSeekV4SparseDecodeCachedBenchmarkCase(512);
        RunCpuDeepSeekV4SparseDecodeCachedBenchmarkCase(16);
    }

    void RunCpuDeepSeekV4IndexerBenchmark(int sequence) {
        constexpr int heads = 64;
        constexpr int dim = 128;
        constexpr int topK = 512;
        int blocks = sequence / 4;
        Expect(sequence >= 4 && sequence % 4 == 0,
               "DeepSeek-V4 indexer benchmark length must be divisible by 4.");

        fastllm::SetThreads(40);
        fastllm::Data q(
            fastllm::DataType::BFLOAT16, {1, sequence, heads, dim});
        fastllm::Data weights(
            fastllm::DataType::BFLOAT16, {1, sequence, heads});
        fastllm::Data kv(
            fastllm::DataType::BFLOAT16, {1, blocks, dim});
        q.Allocate();
        weights.Allocate();
        kv.Allocate();
        std::fill_n(
            (uint16_t*)q.cpuData, q.Count(0),
            fastllm::Float32ToBFloat16RNEBits(0.375f));
        std::fill_n(
            (uint16_t*)weights.cpuData, weights.Count(0),
            fastllm::Float32ToBFloat16RNEBits(1.0f / heads));
        uint16_t *kvBits = (uint16_t*)kv.cpuData;
        for (int block = 0; block < blocks; block++) {
            for (int d = 0; d < dim; d++) {
                float value = (float)((block * 13 + d * 7) % 31 - 15) /
                    16.0f;
                kvBits[(uint64_t)block * dim + d] =
                    fastllm::Float32ToBFloat16RNEBits(value);
            }
        }

        for (int repeat = 0; repeat < 2; repeat++) {
            fastllm::Data output;
            auto begin = std::chrono::steady_clock::now();
            {
                ScopedFirstDevice device("cpu");
                fastllm::DeepSeekV4IndexerTopK(
                    q, weights, kv, topK, 4, 64, 10000.0f, 0, 0,
                    1.0f, 32, 1, output);
            }
            auto end = std::chrono::steady_clock::now();
            const int32_t *indices = (const int32_t*)output.cpuData;
            int64_t checksum = 0;
            for (uint64_t i = 0; i < output.Count(0); i += 257) {
                checksum += indices[i];
            }
            double seconds = std::chrono::duration<double>(end - begin).count();
            std::cout << "DeepSeek-V4 CPU IndexerTopK benchmark: sequence="
                      << sequence << ", repeat=" << repeat + 1
                      << ", seconds=" << seconds
                      << ", checksum=" << checksum << "\n";
        }
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

    std::vector<float> ApplyCachedYarnReference(
            const std::vector<float> &input, const std::vector<int> &dims,
            const std::vector<float> &positions, int rotaryDim,
            float ropeTheta, float factor, float originalMaxPosition,
            float betaFast, float betaSlow, float attentionFactor) {
        std::vector<float> output = input;
        const int bs = dims[0], len = dims[1], heads = dims[2], headDim = dims[3];
        const int half = rotaryDim / 2;
        auto findCorrectionDim = [&](float rotations) {
            return (rotaryDim * std::log(originalMaxPosition /
                    (rotations * 2.0f * (float)M_PI))) /
                   (2.0f * std::log(ropeTheta));
        };
        float low = std::max(0.0f, std::floor(findCorrectionDim(betaFast)));
        float high = std::min((float)rotaryDim - 1.0f,
                              std::ceil(findCorrectionDim(betaSlow)));
        if (low == high) {
            high += 0.001f;
        }

        std::vector<float> invFreq(half);
        for (int dim = 0; dim < half; dim++) {
            float posFreq = std::pow(ropeTheta, (float)(2 * dim) / rotaryDim);
            float extrapolation = 1.0f / posFreq;
            float interpolation = 1.0f / (factor * posFreq);
            float ramp = std::max(0.0f, std::min(1.0f,
                (dim - low) / (high - low)));
            float extrapolationFactor = 1.0f - ramp;
            invFreq[dim] = interpolation * (1.0f - extrapolationFactor) +
                           extrapolation * extrapolationFactor;
        }

        for (int batch = 0; batch < bs; batch++) {
            for (int token = 0; token < len; token++) {
                int position = (int)positions[batch * len + token];
                for (int dim = 0; dim < half; dim++) {
                    float angle = position * invFreq[dim];
                    float curSin = std::sin(angle) * attentionFactor;
                    float curCos = std::cos(angle) * attentionFactor;
                    for (int head = 0; head < heads; head++) {
                        size_t offset = (((size_t)batch * len + token) * heads + head) * headDim;
                        float a = input[offset + dim];
                        float b = input[offset + dim + half];
                        output[offset + dim] = a * curCos - b * curSin;
                        output[offset + dim + half] = a * curSin + b * curCos;
                    }
                }
            }
        }
        return output;
    }

    void RunYarnRopeEncodingRegression() {
        const std::vector<int> dims = {2, 4, 3, 128};
        const std::vector<float> positions = {
            0.0f, 1.0f, 8191.0f, 8192.0f,
            32768.0f, 131071.0f, 262143.0f, 17.75f
        };
        constexpr int rotaryDim = 64;
        constexpr float ropeTheta = 500000.0f;
        constexpr float factor = 32.0f;
        constexpr float originalMaxPosition = 8192.0f;
        constexpr float betaFast = 32.0f;
        constexpr float betaSlow = 1.0f;
        constexpr float attentionFactor = 1.3465735902799727f;

        fastllm::Data initialData = MakeFloatTensor(dims, 0.37f);
        std::vector<float> initial = ToFloatVector(initialData);
        std::vector<float> expected = ApplyCachedYarnReference(
            initial, dims, positions, rotaryDim, ropeTheta, factor,
            originalMaxPosition, betaFast, betaSlow, attentionFactor);
        fastllm::Data positionIds(fastllm::DataType::FLOAT32, {2, 4}, positions);

        fastllm::Data cpuInput(fastllm::DataType::FLOAT32, dims, initial);
        {
            ScopedFirstDevice guard("cpu");
            fastllm::YarnRopeEncoding(
                cpuInput, positionIds, rotaryDim, ropeTheta, factor,
                originalMaxPosition, betaFast, betaSlow, attentionFactor);
        }
        ExpectFloatNear(expected, ToFloatVector(cpuInput), 2e-6f, 2e-6f,
                        "CPU direct YaRN versus cached reference");

#ifdef USE_CUDA
        if (fastllm::HasDeviceType("cuda")) {
            for (fastllm::DataType dataType : {
                     fastllm::DataType::FLOAT32,
                     fastllm::DataType::FLOAT16,
                     fastllm::DataType::BFLOAT16}) {
                fastllm::Data typedCpu(dataType, dims, initial);
                fastllm::Data typedCuda(dataType, dims, initial);
                {
                    ScopedFirstDevice guard("cpu");
                    fastllm::YarnRopeEncoding(
                        typedCpu, positionIds, rotaryDim, ropeTheta, factor,
                        originalMaxPosition, betaFast, betaSlow, attentionFactor);
                }
                typedCuda.ToDevice(fastllm::DataDevice::CUDA);
                {
                    ScopedFirstDevice guard("cuda");
                    fastllm::YarnRopeEncoding(
                        typedCuda, positionIds, rotaryDim, ropeTheta, factor,
                        originalMaxPosition, betaFast, betaSlow, attentionFactor);
                }
                float atol = dataType == fastllm::DataType::FLOAT32 ? 5e-4f :
                    (dataType == fastllm::DataType::FLOAT16 ? 1.5e-3f : 1e-2f);
                float rtol = dataType == fastllm::DataType::FLOAT32 ? 2e-4f :
                    (dataType == fastllm::DataType::FLOAT16 ? 1e-3f : 5e-3f);
                ExpectFloatNear(ToFloatVector(typedCpu), ToFloatVector(typedCuda),
                                atol, rtol, "CUDA direct YaRN versus CPU");
            }
        }
#endif
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

    class ScopedEnvOverride {
    public:
        ScopedEnvOverride(const char *name, const char *value)
            : name(name) {
            const char *current = std::getenv(name);
            hadValue = current != nullptr;
            if (hadValue) {
                oldValue = current;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnvOverride() {
            if (hadValue) {
                setenv(name.c_str(), oldValue.c_str(), 1);
            } else {
                unsetenv(name.c_str());
            }
        }

    private:
        std::string name;
        std::string oldValue;
        bool hadValue = false;
    };

    bool RunCudaVarlenChunkGdnSelected(
            fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
            fastllm::Data &g, fastllm::Data &attn,
            fastllm::Data &decayMask, fastllm::Data &kCumdecay,
            fastllm::Data &lastRecurrentState,
            bool fuseDecayMask, bool directOutputQk,
            const std::vector<int> &seqLens, fastllm::Data &coreAttnOut,
            const std::string &label) {
        bool requireTriton = fastllm::GetFastllmEnv().cudaTriton &&
            RegressionEnvFlagDefaultEnabled(
                "FASTLLM_CUDA_TRITON_CHUNK_GDN_VARLEN_PREFILL", true);
        bool ok = requireTriton
            ? fastllm::FastllmCudaTryTritonChunkGdnVarlenPrefill(
                  q, k, v, g, attn, decayMask, kCumdecay,
                  lastRecurrentState, fuseDecayMask, directOutputQk,
                  seqLens, coreAttnOut)
            : fastllm::FastllmCudaChunkGatedDeltaRuleVarlenPrefill(
                  q, k, v, g, attn, decayMask, kCumdecay,
                  lastRecurrentState, fuseDecayMask, directOutputQk,
                  seqLens, coreAttnOut);
        if (requireTriton) {
            Expect(ok, label + " did not execute the enabled Triton path");
        }
        return ok;
    }

    fastllm::Data MakeCudaTensor(fastllm::DataType dataType, const std::vector<int> &dims,
                                 const std::vector<float> &values) {
        fastllm::Data data(dataType, dims, values);
        data.ToDevice(fastllm::DataDevice::CUDA);
        return data;
    }

    std::vector<float> MakeRegressionValues(int count, float seed, float scale);

#ifndef USE_ROCM
    fastllm::Data MakeNvfp4Block16Weight(
            int outputDim, int inputDim, float globalScale) {
        Expect(outputDim > 0 && inputDim > 0 && inputDim % 16 == 0,
               "NVFP4_BLOCK_16 regression weight shape is invalid");
        fastllm::Data weight;
        weight.dataType = fastllm::DataType::NVFP4_BLOCK_16;
        weight.UpdateUnitSize();
        weight.Resize({outputDim, inputDim});
        weight.weightType = fastllm::WeightType::LINEAR;
        weight.blockK = 1;
        weight.blockM = 16;
        weight.Allocate(false);

        const int groups = inputDim / 16;
        const size_t rowBytes = fastllm::GetDataBytes(
            fastllm::DataType::NVFP4_BLOCK_16, 1, inputDim);
        auto *bytes = reinterpret_cast<uint8_t *>(weight.cpuData);
        for (int row = 0; row < outputDim; row++) {
            for (int group = 0; group < groups; group++) {
                uint8_t *block = bytes + (size_t)row * rowBytes + group * 12;
                for (int packed = 0; packed < 8; packed++) {
                    uint8_t low = static_cast<uint8_t>(
                        (row * 3 + group * 5 + packed * 7 + 1) & 0xf);
                    uint8_t high = static_cast<uint8_t>(
                        (row * 11 + group * 13 + packed * 3 + 2) & 0xf);
                    block[packed] = static_cast<uint8_t>((high << 4) | low);
                }
                float effectiveScale = globalScale *
                    static_cast<float>(1 << ((row + group) & 3));
                std::memcpy(block + 8, &effectiveScale, sizeof(float));
            }
        }
        // Multiple entries model merged weights whose partitions retained
        // different tensor-level scales.  Marlin must select the common minimum
        // without losing the effective inline block scales.
        weight.scales = {globalScale * 2.0f, globalScale};
        weight.ToDevice(
            fastllm::DataDevice::CUDA, std::vector<int>{0});
        return weight;
    }

    void RunCudaNVFP4MarlinRegression() {
        FastllmCudaSetDevice(0);
        constexpr int inputDim = 128;
        constexpr int outputDim = 256;
        constexpr int batch = 31;
        if (!FastllmCudaMarlinNVFP4Supported(outputDim, inputDim)) {
            std::cout << "CUDA NVFP4 Marlin regression: SKIP\n";
            return;
        }

        ScopedEnvOverride forceMarlin("FASTLLM_CUDA_NVFP4_MARLIN", "1");
        struct ScopedNcclForceSyncRestore {
            bool previous = FastllmCudaGetNcclForceSync();
            ~ScopedNcclForceSyncRestore() {
                FastllmCudaSetNcclForceSync(previous);
            }
        } restoreForceSync;

        constexpr float globalScale = 1.0f / 256.0f;
        fastllm::Data referenceWeight = MakeNvfp4Block16Weight(
            outputDim, inputDim, globalScale);
        fastllm::Data marlinWeight = MakeNvfp4Block16Weight(
            outputDim, inputDim, globalScale);
        std::vector<float> inputValues = MakeRegressionValues(
            batch * inputDim, 0.43f, 0.08f);
        fastllm::Data input = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, inputDim}, inputValues);
        fastllm::Data warmupInput = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {2, inputDim},
            std::vector<float>(inputValues.begin(),
                               inputValues.begin() + 2 * inputDim));
        fastllm::Data bias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {outputDim},
            MakeRegressionValues(outputDim, 0.79f, 0.015f));
        fastllm::Data noBias;
        fastllm::Data reference = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, outputDim},
            std::vector<float>((size_t)batch * outputDim, 0.0f));
        fastllm::Data actual = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, outputDim},
            std::vector<float>((size_t)batch * outputDim, 0.0f));
        fastllm::Data warmup = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {2, outputDim},
            std::vector<float>(2 * outputDim, 0.0f));

        FastllmCudaSetNcclForceSync(false);
        Expect(FastllmCudaHalfMatMulFloatNVFP4Block16(
                   input, referenceWeight, bias, reference,
                   batch, inputDim, outputDim),
               "native NVFP4_BLOCK_16 reference failed");
        FastllmCudaSyncCurrentThreadStream();
        Expect(!FastllmCudaHasNVFP4MarlinLayout(referenceWeight),
               "large-M NVFP4 reference unexpectedly repacked its weight");

        FastllmCudaSetNcclForceSync(true);
        Expect(FastllmCudaHalfMatMulFloatNVFP4Block16(
                   warmupInput, marlinWeight, bias, warmup,
                   2, inputDim, outputDim),
               "NVFP4 Marlin warmup conversion failed");
        FastllmCudaSyncCurrentThreadStream();
        Expect(FastllmCudaHasNVFP4MarlinLayout(marlinWeight),
               "NVFP4 Marlin warmup did not mark the in-place layout");

        Expect(FastllmCudaHalfMatMulFloatNVFP4Block16(
                   input, marlinWeight, bias, actual,
                   batch, inputDim, outputDim),
               "NVFP4 Marlin large-M path failed after repack");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(ToFloatVector(reference), ToFloatVector(actual),
                        2.0e-2f, 2.0e-3f,
                        "NVFP4 Marlin large-M output with bias");

        fastllm::Data referenceSingle = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, outputDim},
            std::vector<float>(outputDim, 0.0f));
        fastllm::Data actualSingle = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, outputDim},
            std::vector<float>(outputDim, 0.0f));
        FastllmCudaSetNcclForceSync(false);
        Expect(FastllmCudaHalfMatMulFloatNVFP4Block16(
                   input, referenceWeight, noBias, referenceSingle,
                   1, inputDim, outputDim),
               "native NVFP4_BLOCK_16 batch-one reference failed");
        Expect(FastllmCudaHalfMatMulFloatNVFP4Block16(
                   input, marlinWeight, noBias, actualSingle,
                   1, inputDim, outputDim),
               "NVFP4 Marlin batch-one path failed after repack");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(ToFloatVector(referenceSingle),
                        ToFloatVector(actualSingle),
                        2.0e-2f, 2.0e-3f,
                        "NVFP4 Marlin batch-one output without bias");
        std::cout << "CUDA NVFP4 Marlin regression: PASS\n";
    }

    void RunCudaBFloat16Hidden3072RMSNormRegression() {
        FastllmCudaSetDevice(0);
        constexpr int outer = 7;
        constexpr int hidden = 3072;
        constexpr float eps = 1.0e-6f;
        const std::vector<int> dims = {outer, hidden};
        std::vector<float> inputValues =
            MakeRegressionValues(outer * hidden, 0.413f, 0.19f);
        std::vector<float> weightValues(hidden);
        for (int i = 0; i < hidden; i++) {
            weightValues[i] = 0.85f + 0.2f * std::sin((i + 1) * 0.017f);
        }
        fastllm::Data weight = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {hidden}, weightValues);

        fastllm::Data input = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, dims, inputValues);
        fastllm::Data legacyOutput = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, dims,
            std::vector<float>(outer * hidden, 0.0f));
        fastllm::Data specializedOutput = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, dims,
            std::vector<float>(outer * hidden, 0.0f));
        Expect(FastllmCudaRMSNormBFloat16WithThreadCount(
                   input, weight, legacyOutput, eps, 0),
               "legacy BF16 hidden-3072 RMSNorm launch failed");
        Expect(FastllmCudaRMSNormBFloat16WithThreadCount(
                   input, weight, specializedOutput, eps, 256),
               "specialized BF16 hidden-3072 RMSNorm launch failed");
        ExpectFloatNear(ToFloatVector(legacyOutput),
                        ToFloatVector(specializedOutput), 0.0f, 0.0f,
                        "out-of-place BF16 hidden-3072 RMSNorm");

        fastllm::Data legacyInplace = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, dims, inputValues);
        fastllm::Data specializedInplace = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, dims, inputValues);
        Expect(FastllmCudaRMSNormBFloat16WithThreadCount(
                   legacyInplace, weight, legacyInplace, eps, 0),
               "legacy in-place BF16 hidden-3072 RMSNorm launch failed");
        Expect(FastllmCudaRMSNormBFloat16WithThreadCount(
                   specializedInplace, weight, specializedInplace, eps, 256),
               "specialized in-place BF16 hidden-3072 RMSNorm launch failed");
        ExpectFloatNear(ToFloatVector(legacyInplace),
                        ToFloatVector(specializedInplace), 0.0f, 0.0f,
                        "in-place BF16 hidden-3072 RMSNorm");
    }
#endif

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

    void RunCudaLocalExpertRangeMaskRegression() {
        const std::vector<int32_t> routeIndices = {
            -1, 0, 31, 32, 47, 63, 64, 255
        };
        const std::vector<float> routeScores = {
            0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f
        };
        fastllm::Data index = MakeIntTensor({1, (int)routeIndices.size()}, routeIndices);
        fastllm::Data score(
            fastllm::DataType::FLOAT32,
            {1, (int)routeScores.size()}, routeScores);
        index.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        score.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);

        Expect(FastllmCudaMaskAndRemapExpertsForLocalRange(index, score, 32, 64),
               "CUDA local expert range mask rejected valid inputs.");
        const std::vector<int32_t> expectedIndices = {
            -1, -1, -1, 0, 15, 31, -1, -1
        };
        const std::vector<float> expectedScores = {
            0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.30f, 0.0f, 0.0f
        };
        Expect(ToIntVector(index) == expectedIndices,
               "CUDA local expert range mask produced incorrect local ids.");
        ExpectFloatNear(expectedScores, ToFloatVector(score), 0.0f, 0.0f,
                        "CUDA local expert range mask scores");
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

        fastllm::Data cpuOutputOnlyState(
            fastllm::DataType::FLOAT32, stateDims, initialStateValues);
        fastllm::Data cpuOutputOnly;
        {
            ScopedFirstDevice device("cpu");
            fastllm::KimiK3RecurrentKDAOutputOnly(
                cpuQ, cpuK, cpuV, cpuGate, cpuRawBeta, cpuALog, cpuDtBias,
                -5.0f, cpuOutputOnlyState, cpuOutputOnly);
        }
        ExpectFloatNear(ToFloatVector(cpuOutput),
                        ToFloatVector(cpuOutputOnly), 0.0f, 0.0f,
                        "CPU Kimi-K3 output-only KDA output");
        ExpectFloatNear(ToFloatVector(cpuState),
                        ToFloatVector(cpuOutputOnlyState), 0.0f, 0.0f,
                        "CPU Kimi-K3 output-only KDA state");

        fastllm::Data cudaChunkedState = MakeCudaTensor(
            fastllm::DataType::FLOAT32, stateDims, initialStateValues);
        fastllm::Data cudaChunkedOutput;
        {
            ScopedFirstDevice device("cuda:0");
            constexpr int recurrentChunkSize = 3;
            for (int start = 0; start < sequence;
                 start += recurrentChunkSize) {
                const int end = std::min(
                    sequence, start + recurrentChunkSize);
                fastllm::Data qChunk, kChunk, vChunk, gateChunk;
                fastllm::Data betaChunk, outputChunk;
                fastllm::Split(cudaQ, 1, start, end, qChunk);
                fastllm::Split(cudaK, 1, start, end, kChunk);
                fastllm::Split(cudaV, 1, start, end, vChunk);
                fastllm::Split(cudaGate, 1, start, end, gateChunk);
                fastllm::Split(
                    cudaRawBeta, 1, start, end, betaChunk);
                fastllm::KimiK3RecurrentKDAOutputOnly(
                    qChunk, kChunk, vChunk, gateChunk, betaChunk,
                    cudaALog, cudaDtBias, -5.0f,
                    cudaChunkedState, outputChunk);
                if (start == 0) {
                    fastllm::Copy(outputChunk, cudaChunkedOutput);
                    cudaChunkedOutput.Expansion(vectorDims);
                } else {
                    fastllm::CatDirect(
                        cudaChunkedOutput, outputChunk, 1);
                }
            }
            cudaChunkedOutput.expansionDims.clear();
        }
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(ToFloatVector(cudaOutput),
                        ToFloatVector(cudaChunkedOutput), 0.0f, 0.0f,
                        "CUDA Kimi-K3 chunked output-only KDA output");
        ExpectFloatNear(ToFloatVector(cudaState),
                        ToFloatVector(cudaChunkedState), 2e-5f, 2e-5f,
                        "CUDA Kimi-K3 chunked output-only KDA state");

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

    void RunCudaMergeMlaPagedChunkRegression() {
        FastllmCudaSetDevice(0);
        constexpr int pageLen = 4;
        constexpr int kvLength = 11;
        constexpr int queryLength = 6;
        constexpr int historyLength = kvLength - queryLength;
        constexpr int heads = 2;
        constexpr int ckvDimension = 512;
        constexpr int kpeDimension = 64;

        fastllm::ClearAllPagedCacheManagers();
        {
        fastllm::Data kpe = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {1, kvLength, kpeDimension},
            MakeRegressionValues(
                kvLength * kpeDimension, 3.11f, 0.015f));
        fastllm::Data ckv = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {1, kvLength, ckvDimension},
            MakeRegressionValues(
                kvLength * ckvDimension, 3.47f, 0.012f));
        fastllm::Data qNope = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {heads, queryLength, ckvDimension},
            MakeRegressionValues(
                heads * queryLength * ckvDimension, 3.83f, 0.01f));
        fastllm::Data qPe = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {1, queryLength, heads, kpeDimension},
            MakeRegressionValues(
                queryLength * heads * kpeDimension, 4.19f, 0.014f));

        fastllm::Data kpeCache, ckvCache;
        auto prepareDescriptor = [](fastllm::Data &cache,
                                    const fastllm::Data &current) {
            cache.dataType = current.dataType;
            cache.UpdateUnitSize();
            cache.dataDevice = current.dataDevice;
            cache.dataDeviceIds = current.dataDeviceIds;
            cache.isKVCache = true;
        };
        prepareDescriptor(kpeCache, kpe);
        prepareDescriptor(ckvCache, ckv);
        fastllm::PagedCacheManager *kpeManager =
            fastllm::AllocatePagedCacheManager(
                10000,
                fastllm::PagedCacheManager::
                    PAGED_CACHE_MANAGER_TYPE_MLP_CACHE,
                kpe, pageLen, 4);
        fastllm::PagedCacheManager *ckvManager =
            fastllm::AllocatePagedCacheManager(
                10001,
                fastllm::PagedCacheManager::
                    PAGED_CACHE_MANAGER_TYPE_MLP_CACHE,
                ckv, pageLen, 4);
        kpeCache.ToDevice(kpe.dataDevice, kpe.dataDeviceIds, false);
        ckvCache.ToDevice(ckv.dataDevice, ckv.dataDeviceIds, false);

        fastllm::Data fullOutput, explicitFullOutput, chunkedOutput;
        {
            ScopedFirstDevice device("cuda:0");
            fastllm::AppendPagedCache(*kpeManager, kpeCache, kpe);
            fastllm::AppendPagedCache(*ckvManager, ckvCache, ckv);
            fastllm::MergeMLAPaged(
                qNope, qPe, kpeCache, ckvCache,
                fullOutput, 1.0f / std::sqrt(576.0f));
            fastllm::MergeMLAPaged(
                qNope, qPe, kpeCache, ckvCache,
                explicitFullOutput, 1.0f / std::sqrt(576.0f),
                kvLength);

            constexpr int mlaChunkSize = 3;
            for (int start = 0; start < queryLength;
                 start += mlaChunkSize) {
                const int end = std::min(
                    queryLength, start + mlaChunkSize);
                fastllm::Data qNopeChunk, qPeChunk, outputChunk;
                fastllm::Split(
                    qNope, 1, start, end, qNopeChunk);
                fastllm::Split(qPe, 1, start, end, qPeChunk);
                fastllm::MergeMLAPaged(
                    qNopeChunk, qPeChunk,
                    kpeCache, ckvCache, outputChunk,
                    1.0f / std::sqrt(576.0f),
                    historyLength + end);
                if (start == 0) {
                    fastllm::Copy(outputChunk, chunkedOutput);
                    chunkedOutput.Expansion(
                        {heads, queryLength, ckvDimension});
                } else {
                    fastllm::CatDirect(
                        chunkedOutput, outputChunk, 1);
                }
            }
            chunkedOutput.expansionDims.clear();
        }
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(
            ToFloatVector(fullOutput), ToFloatVector(explicitFullOutput),
            0.0f, 0.0f,
            "CUDA length-limited full paged MLA output");
        ExpectFloatNear(
            ToFloatVector(fullOutput), ToFloatVector(chunkedOutput),
            0.0f, 0.0f,
            "CUDA chunked paged MLA output");
        }
        fastllm::ClearAllPagedCacheManagers();
        std::cout << "CUDA chunked paged MLA regression: PASS\n";
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
            "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_BLOCK_V", 32, 32, 64);
        if (blockV != 32 && blockV != 64) {
            blockV = 32;
        }
        int numWarps = envIntRange(
            "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_NUM_WARPS", 4, 1, 32);
        constexpr const char *commonStagesKey =
            "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_NUM_STAGES";
        const char *commonStagesValue = std::getenv(commonStagesKey);
        bool commonStagesSet =
            commonStagesValue != nullptr && commonStagesValue[0] != '\0';
        int commonNumStages =
            envIntRange(commonStagesKey, 3, 1, 8);
        int hNumStages = envIntRange(
            "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_H_NUM_STAGES",
            commonStagesSet ? commonNumStages : 2, 1, 8);
        int oNumStages = envIntRange(
            "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_O_NUM_STAGES",
            commonStagesSet ? commonNumStages : 3, 1, 8);
        int oBlockV = blockV;
        if (blockV == 32 && RegressionEnvFlagDefaultEnabled(
                "FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_SPLIT_BLOCK_V",
                true)) {
            oBlockV = 64;
        }

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
        auto metadataBase = [&](int selectedBlockV, int selectedNumStages) {
            return cacheDir + separator +
            "chunk_gdn_prefill_v6_fp16_sm" + std::to_string(arch) +
            "_c2_t64_k128_v128_bv" +
            std::to_string(selectedBlockV) +
            "_nw" + std::to_string(numWarps) +
            "_ns" + std::to_string(selectedNumStages);
        };
        auto readMetadata = [&](const std::string &base) {
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
            return meta;
        };
        json11::Json hMetadata = readMetadata(
            metadataBase(blockV, hNumStages));
        json11::Json oMetadata =
            oBlockV == blockV && oNumStages == hNumStages
                ? hMetadata
                : readMetadata(metadataBase(oBlockV, oNumStages));
        json11::Json hMeta = hMetadata["kernels"]["h"];
        json11::Json oMeta = oMetadata["kernels"]["o"];
        json11::Json oFusedDecayMeta =
            oMetadata["kernels"]["o_fused_decay_mask"];
        json11::Json hPrecomputedScaleMeta =
            hMetadata["kernels"]["h_precomputed_scale"];
        std::string hCubin = hMeta["cubin"].string_value();
        std::string hKernel = hMeta["kernel"].string_value();
        int hNumWarps = hMeta["num_warps"].int_value();
        int hShared = hMeta["shared"].int_value();
        std::string oCubin = oMeta["cubin"].string_value();
        std::string oKernel = oMeta["kernel"].string_value();
        int oNumWarps = oMeta["num_warps"].int_value();
        int oShared = oMeta["shared"].int_value();
        std::string oFusedDecayCubin =
            oFusedDecayMeta["cubin"].string_value();
        std::string oFusedDecayKernel =
            oFusedDecayMeta["kernel"].string_value();
        int oFusedDecayNumWarps =
            oFusedDecayMeta["num_warps"].int_value();
        int oFusedDecayShared =
            oFusedDecayMeta["shared"].int_value();
        std::string hPrecomputedScaleCubin =
            hPrecomputedScaleMeta["cubin"].string_value();
        std::string hPrecomputedScaleKernel =
            hPrecomputedScaleMeta["kernel"].string_value();
        int hPrecomputedScaleNumWarps =
            hPrecomputedScaleMeta["num_warps"].int_value();
        int hPrecomputedScaleShared =
            hPrecomputedScaleMeta["shared"].int_value();
        Expect(!hCubin.empty() && !hKernel.empty() && hNumWarps > 0 &&
                   !oCubin.empty() && !oKernel.empty() && oNumWarps > 0 &&
                   !oFusedDecayCubin.empty() &&
                   !oFusedDecayKernel.empty() &&
                   oFusedDecayNumWarps > 0 &&
                   !hPrecomputedScaleCubin.empty() &&
                   !hPrecomputedScaleKernel.empty() &&
                   hPrecomputedScaleNumWarps > 0,
               "Triton chunk GDN prefill kernel metadata is incomplete");

        fastllm::Data directState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, initialState);
        fastllm::Data directOutput;
        Expect(FastllmCudaTritonChunkGatedDeltaRulePrefill(
                   hCubin.c_str(), hKernel.c_str(), hNumWarps, hShared,
                   oCubin.c_str(), oKernel.c_str(), oNumWarps, oShared,
                   oFusedDecayCubin.c_str(),
                   oFusedDecayKernel.c_str(),
                   oFusedDecayNumWarps, oFusedDecayShared,
                   hPrecomputedScaleCubin.c_str(),
                   hPrecomputedScaleKernel.c_str(),
                   hPrecomputedScaleNumWarps, hPrecomputedScaleShared,
                   false, false,
                   chunks, chunkSize, kDim, vDim, blockV, oBlockV,
                   q, k, v, g, attn, attn,
                   kCumdecay, directState, directOutput),
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
            oFusedDecayCubin.c_str(),
            oFusedDecayKernel.c_str(),
            oFusedDecayNumWarps, oFusedDecayShared,
            hPrecomputedScaleCubin.c_str(),
            hPrecomputedScaleKernel.c_str(),
            hPrecomputedScaleNumWarps, hPrecomputedScaleShared,
            false, false,
            chunks, chunkSize, kDim, vDim, blockV, oBlockV,
            q, k, v, g, attn, attn,
            kCumdecay, failedState, failedOutput);
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

    void RunCudaRaggedGdnLogicalPrepRegression() {
        FastllmCudaSetDevice(0);
        constexpr int chunkSize = 64;
        constexpr int kDim = 128;
        constexpr int vDim = 128;
        constexpr int keyHeads = 2;
        constexpr int valueHeads = 6;
        constexpr int headGroup = valueHeads / keyHeads;
        constexpr int baOffset = 5;
        constexpr int baChannels = baOffset + valueHeads * 2 + 3;
        constexpr float eps = 1.0e-6f;
        const float qScale = 1.0f / std::sqrt((float)kDim);
        const std::vector<int> seqLens = {65, 7};
        const int totalTokens =
            std::accumulate(seqLens.begin(), seqLens.end(), 0);
        int totalChunks = 0;
        for (int len : seqLens) {
            totalChunks += (len + chunkSize - 1) / chunkSize;
        }
        const int packedTokens = totalChunks * chunkSize;
        const int qkvChannels =
            keyHeads * kDim * 2 + valueHeads * vDim;

        std::vector<float> qkvValues = MakeRegressionValues(
            (size_t)totalTokens * qkvChannels, 0.23f, 0.013f);
        std::vector<float> baValues = MakeRegressionValues(
            (size_t)totalTokens * baChannels, 0.41f, 0.017f);
        std::vector<float> normValues = MakeRegressionValues(
            kDim, 0.59f, 0.004f);
        std::vector<float> aLogValues = MakeRegressionValues(
            valueHeads, -1.7f, 0.03f);
        std::vector<float> dtBiasValues = MakeRegressionValues(
            valueHeads, -0.2f, 0.02f);
        fastllm::Data qkv = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, qkvChannels}, qkvValues);
        fastllm::Data combinedBa = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, baChannels}, baValues);
        fastllm::Data normWeight = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {kDim}, normValues);
        fastllm::Data aLog = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {valueHeads}, aLogValues);
        fastllm::Data dtBias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {valueHeads}, dtBiasValues);

        fastllm::Data logicalQ, logicalK, packedG, kBeta, vBeta;
        Expect(FastllmCudaQwen35GdnPostConvRaggedExactFloat16(
                   qkv, normWeight, combinedBa, aLog, dtBias,
                   baOffset, seqLens, chunkSize,
                   keyHeads, valueHeads, kDim, vDim,
                   eps, qScale, logicalQ, logicalK,
                   packedG, kBeta, vBeta),
               "ragged exact GDN post-conv rejected valid input");
        FastllmCudaSyncCurrentThreadStream();
        ExpectCudaTensorMeta(
            logicalQ, fastllm::DataType::FLOAT16,
            {1, keyHeads, packedTokens, kDim},
            "ragged logical Q");
        ExpectCudaTensorMeta(
            logicalK, fastllm::DataType::FLOAT16,
            {1, keyHeads, packedTokens, kDim},
            "ragged logical K");
        ExpectCudaTensorMeta(
            packedG, fastllm::DataType::FLOAT16,
            {1, valueHeads, packedTokens},
            "ragged packed G");

        fastllm::Data normalizedQ, normalizedK;
        Expect(FastllmCudaRMSNormCombinedQKFloat16(
                   qkv, normWeight, 1, totalTokens,
                   keyHeads, valueHeads, kDim, vDim, eps,
                   normalizedQ, normalizedK),
               "combined Q/K RMSNorm reference failed");
        fastllm::Data tokenBeta, tokenG;
        Expect(FastllmCudaSigmoidMambaSoftplusCombinedFloat16(
                   combinedBa, aLog, dtBias, 1, totalTokens,
                   baChannels, baOffset, valueHeads,
                   tokenBeta, tokenG),
               "combined BA reference failed");
        FastllmCudaSyncCurrentThreadStream();

        std::vector<float> normalizedQValues = ToFloatVector(normalizedQ);
        std::vector<float> normalizedKValues = ToFloatVector(normalizedK);
        std::vector<float> repeatedQValues(
            (size_t)totalTokens * valueHeads * kDim);
        std::vector<float> repeatedKValues(repeatedQValues.size());
        std::vector<float> tokenVValues(
            (size_t)totalTokens * valueHeads * vDim);
        for (int token = 0; token < totalTokens; token++) {
            for (int keyHead = 0; keyHead < keyHeads; keyHead++) {
                for (int group = 0; group < headGroup; group++) {
                    int valueHead = keyHead * headGroup + group;
                    for (int d = 0; d < kDim; d++) {
                        size_t source =
                            ((size_t)token * keyHeads + keyHead) * kDim + d;
                        size_t target =
                            ((size_t)token * valueHeads + valueHead) * kDim + d;
                        repeatedQValues[target] = normalizedQValues[source];
                        repeatedKValues[target] = normalizedKValues[source];
                    }
                }
            }
            for (int valueHead = 0; valueHead < valueHeads; valueHead++) {
                for (int d = 0; d < vDim; d++) {
                    tokenVValues[
                        ((size_t)token * valueHeads + valueHead) * vDim + d] =
                        qkvValues[(size_t)token * qkvChannels +
                                  keyHeads * kDim * 2 +
                                  valueHead * vDim + d];
                }
            }
        }
        fastllm::Data repeatedQ = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, valueHeads, kDim}, repeatedQValues);
        fastllm::Data repeatedK = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, valueHeads, kDim}, repeatedKValues);
        fastllm::Data tokenV = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, valueHeads, vDim}, tokenVValues);
        fastllm::Data referenceQRepeated, referenceKRepeated;
        fastllm::Data referenceV, referenceBeta, referenceG;
        Expect(FastllmCudaPackRaggedGdnPrefillChunksFloat16(
                   repeatedQ, repeatedK, tokenV, tokenBeta, tokenG,
                   seqLens, chunkSize, qScale,
                   referenceQRepeated, referenceKRepeated,
                   referenceV, referenceBeta, referenceG),
               "legacy repeated-head ragged pack reference failed");

        std::vector<float> zeros(
            (size_t)totalTokens * keyHeads * kDim, 0.0f);
        std::vector<float> scalarZeros(
            (size_t)totalTokens * keyHeads, 0.0f);
        fastllm::Data dummyV = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, keyHeads, vDim}, zeros);
        fastllm::Data dummyB = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, keyHeads}, scalarZeros);
        fastllm::Data dummyG = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, keyHeads}, scalarZeros);
        fastllm::Data referenceLogicalQ, referenceLogicalK;
        fastllm::Data unusedV, unusedB, unusedG;
        Expect(FastllmCudaPackRaggedGdnPrefillChunksFloat16(
                   normalizedQ, normalizedK, dummyV, dummyB, dummyG,
                   seqLens, chunkSize, qScale,
                   referenceLogicalQ, referenceLogicalK,
                   unusedV, unusedB, unusedG),
               "legacy logical-head ragged pack reference failed");

        referenceBeta.Reshape(
            {1, valueHeads, packedTokens, 1});
        fastllm::Data referenceKBeta = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            referenceKRepeated.dims, ToFloatVector(referenceKRepeated));
        fastllm::Data referenceVBeta = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            referenceV.dims, ToFloatVector(referenceV));
        Expect(FastllmCudaMulTo(referenceKBeta, referenceBeta, 1.0f),
               "legacy K beta reference failed");
        Expect(FastllmCudaMulTo(referenceVBeta, referenceBeta, 1.0f),
               "legacy V beta reference failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(ToFloatVector(referenceLogicalQ),
                        ToFloatVector(logicalQ), 0.0f, 0.0f,
                        "ragged fused logical Q bitwise output");
        ExpectFloatNear(ToFloatVector(referenceLogicalK),
                        ToFloatVector(logicalK), 0.0f, 0.0f,
                        "ragged fused logical K bitwise output");
        ExpectFloatNear(ToFloatVector(referenceG), ToFloatVector(packedG),
                        0.0f, 0.0f,
                        "ragged fused G bitwise output");
        ExpectFloatNear(ToFloatVector(referenceKBeta), ToFloatVector(kBeta),
                        0.0f, 0.0f,
                        "ragged fused K-beta bitwise output");
        ExpectFloatNear(ToFloatVector(referenceVBeta), ToFloatVector(vBeta),
                        0.0f, 0.0f,
                        "ragged fused V-beta bitwise output");

        logicalK.Reshape(
            {1, keyHeads, totalChunks, chunkSize, kDim});
        kBeta.Reshape(
            {1, valueHeads, totalChunks, chunkSize, kDim});
        referenceKRepeated.Reshape(
            {1, valueHeads, totalChunks, chunkSize, kDim});
        referenceKBeta.Reshape(
            {1, valueHeads, totalChunks, chunkSize, kDim});
        fastllm::Data mappedAt;
        Expect(FastllmCudaBatchMatMulTransBHeadMapped(
                   kBeta, logicalK, mappedAt, headGroup, 1.0f),
               "mapped ragged GDN KKT failed");
        fastllm::Data referenceAt;
        referenceAt.dataType = fastllm::DataType::FLOAT16;
        referenceAt.Resize(
            {1, valueHeads, totalChunks, chunkSize, chunkSize});
        referenceAt.ToDevice(
            fastllm::DataDevice::CUDA, std::vector<int>{0});
        referenceAt.Allocate(false);
        Expect(FastllmCudaBatchMatMulTransB(
                   referenceKBeta, referenceKRepeated, referenceAt,
                   chunkSize * kDim, chunkSize * kDim,
                   chunkSize * chunkSize, kDim, kDim,
                   valueHeads * totalChunks,
                   chunkSize, kDim, chunkSize, 1.0f),
               "repeated-head GDN KKT reference failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(ToFloatVector(referenceAt), ToFloatVector(mappedAt),
                        0.0f, 0.0f,
                        "mapped ragged GDN KKT bitwise output");
        {
            ScopedEnvOverride forceRepeatedHead(
                "FASTLLM_CUDA_GDN_MAPPED_KKT_BATCHED_POINTERS", "0");
            fastllm::Data fallbackAt;
            Expect(FastllmCudaBatchMatMulTransBHeadMapped(
                       kBeta, logicalK, fallbackAt, headGroup, 1.0f),
                   "repeated-head GDN KKT fallback failed");
            FastllmCudaSyncCurrentThreadStream();
            ExpectFloatNear(
                ToFloatVector(referenceAt), ToFloatVector(fallbackAt),
                0.0f, 0.0f,
                "repeated-head GDN KKT fallback bitwise output");
        }
        std::cout << "CUDA repeated-head GDN KKT fallback regression: PASS\n";
        bool mappedKktSelected = fastllm::GetFastllmEnv().cudaTriton &&
            RegressionEnvFlagDefaultEnabled(
                "FASTLLM_CUDA_TRITON_CHUNK_GDN_MAPPED_KKT", true);
        if (mappedKktSelected) {
            fastllm::Data selectedAt;
            Expect(fastllm::FastllmCudaMappedGdnKkt(
                       kBeta, logicalK, headGroup, selectedAt),
                   "selected mapped ragged GDN KKT failed");
            FastllmCudaSyncCurrentThreadStream();
            bool tritonKkt = RegressionEnvFlagDefaultEnabled(
                "FASTLLM_CUDA_TRITON_CHUNK_GDN_KKT", false);
            float tolerance = tritonKkt ? 2e-3f : 0.0f;
            ExpectFloatNear(
                ToFloatVector(referenceAt), ToFloatVector(selectedAt),
                tolerance, tolerance,
                tritonKkt ? "Triton mapped ragged GDN KKT output"
                          : "default mapped ragged GDN KKT output");
        }
        std::cout << "CUDA ragged GDN logical prep regression: PASS\n";
    }

    void RunCudaVarlenChunkGdnPrefillRegression() {
        FastllmCudaSetDevice(0);
        constexpr int heads = 1;
        constexpr int chunkSize = 64;
        constexpr int kDim = 128;
        constexpr int vDim = 128;
        const std::vector<int> seqLens = {65, 129, 7};
        const int batch = (int)seqLens.size();
        std::vector<int> tokenOffsets(batch + 1, 0);
        std::vector<int> chunkOffsets(batch + 1, 0);
        for (int request = 0; request < batch; request++) {
            tokenOffsets[request + 1] =
                tokenOffsets[request] + seqLens[request];
            chunkOffsets[request + 1] = chunkOffsets[request] +
                (seqLens[request] + chunkSize - 1) / chunkSize;
        }
        const int totalTokens = tokenOffsets.back();
        const int totalChunks = chunkOffsets.back();

        const std::vector<float> tokenQValues = MakeRegressionValues(
            totalTokens * heads * kDim, 0.17f, 0.01f);
        const std::vector<float> tokenKValues = MakeRegressionValues(
            totalTokens * heads * kDim, 0.31f, 0.009f);
        const std::vector<float> tokenVValues = MakeRegressionValues(
            totalTokens * heads * vDim, 0.47f, 0.012f);
        const std::vector<float> tokenBValues = MakeRegressionValues(
            totalTokens * heads, 0.61f, 0.02f);
        const std::vector<float> tokenGValues = MakeRegressionValues(
            totalTokens * heads, 0.79f, 0.001f);
        fastllm::Data tokenQ = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, heads, kDim}, tokenQValues);
        fastllm::Data tokenK = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, heads, kDim}, tokenKValues);
        fastllm::Data tokenV = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, heads, vDim}, tokenVValues);
        fastllm::Data tokenB = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, heads}, tokenBValues);
        fastllm::Data tokenG = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            {1, totalTokens, heads}, tokenGValues);
        fastllm::Data packedQ, packedK, packedV, packedB, packedG;
        Expect(FastllmCudaPackRaggedGdnPrefillChunksFloat16(
                   tokenQ, tokenK, tokenV, tokenB, tokenG,
                   seqLens, chunkSize, 1.0f,
                   packedQ, packedK, packedV, packedB, packedG),
               "packed varlen GDN input packing failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectCudaTensorMeta(
            packedQ, fastllm::DataType::FLOAT16,
            {1, heads, totalChunks * chunkSize, kDim},
            "packed varlen GDN Q");
        fastllm::Data unpackedQ;
        Expect(FastllmCudaUnpackRaggedGdnPrefillChunksFloat16(
                   packedQ, seqLens, chunkSize, unpackedQ),
               "packed varlen GDN output unpacking failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(ToFloatVector(tokenQ), ToFloatVector(unpackedQ),
                        0.0f, 0.0f,
                        "packed varlen GDN pack/unpack round trip");

        const std::vector<int> qDims =
            {1, heads, totalChunks, chunkSize, kDim};
        const std::vector<int> vDims =
            {1, heads, totalChunks, chunkSize, vDim};
        const std::vector<int> gDims =
            {1, heads, totalChunks, chunkSize};
        const std::vector<int> attnDims =
            {1, heads, totalChunks, chunkSize, chunkSize};
        const std::vector<int> stateDims =
            {batch, heads, kDim, vDim};
        const std::vector<float> qValues = MakeRegressionValues(
            totalChunks * chunkSize * kDim, 1.01f, 0.004f);
        const std::vector<float> kValues = MakeRegressionValues(
            totalChunks * chunkSize * kDim, 1.19f, 0.003f);
        const std::vector<float> vValues = MakeRegressionValues(
            totalChunks * chunkSize * vDim, 1.37f, 0.006f);
        const std::vector<float> gValues = MakeRegressionValues(
            totalChunks * chunkSize, 1.53f, 0.0004f);
        const std::vector<float> attnValues = MakeRegressionValues(
            totalChunks * chunkSize * chunkSize, 1.71f, 0.001f);
        const std::vector<float> kCumValues = MakeRegressionValues(
            totalChunks * chunkSize * kDim, 1.89f, 0.003f);
        const std::vector<float> initialState = MakeRegressionValues(
            batch * kDim * vDim, 2.07f, 0.002f);

        auto makeChunkSlice = [](
            const std::vector<float> &values, size_t begin,
            size_t count) {
            return std::vector<float>(
                values.begin() + begin, values.begin() + begin + count);
        };
        std::vector<float> referenceOutputValues;
        std::vector<float> referenceStateValues;
        for (int request = 0; request < batch; request++) {
            int requestChunks =
                chunkOffsets[request + 1] - chunkOffsets[request];
            size_t qBegin =
                (size_t)chunkOffsets[request] * chunkSize * kDim;
            size_t vBegin =
                (size_t)chunkOffsets[request] * chunkSize * vDim;
            size_t gBegin =
                (size_t)chunkOffsets[request] * chunkSize;
            size_t attnBegin =
                (size_t)chunkOffsets[request] * chunkSize * chunkSize;
            std::vector<int> requestQDims =
                {1, heads, requestChunks, chunkSize, kDim};
            std::vector<int> requestVDims =
                {1, heads, requestChunks, chunkSize, vDim};
            std::vector<int> requestGDims =
                {1, heads, requestChunks, chunkSize};
            std::vector<int> requestAttnDims =
                {1, heads, requestChunks, chunkSize, chunkSize};
            fastllm::Data requestQ = MakeCudaTensor(
                fastllm::DataType::FLOAT16, requestQDims,
                makeChunkSlice(qValues, qBegin,
                               (size_t)requestChunks * chunkSize * kDim));
            fastllm::Data requestK = MakeCudaTensor(
                fastllm::DataType::FLOAT16, requestQDims,
                makeChunkSlice(kValues, qBegin,
                               (size_t)requestChunks * chunkSize * kDim));
            fastllm::Data requestV = MakeCudaTensor(
                fastllm::DataType::FLOAT16, requestVDims,
                makeChunkSlice(vValues, vBegin,
                               (size_t)requestChunks * chunkSize * vDim));
            fastllm::Data requestG = MakeCudaTensor(
                fastllm::DataType::FLOAT16, requestGDims,
                makeChunkSlice(gValues, gBegin,
                               (size_t)requestChunks * chunkSize));
            fastllm::Data requestAttn = MakeCudaTensor(
                fastllm::DataType::FLOAT16, requestAttnDims,
                makeChunkSlice(
                    attnValues, attnBegin,
                    (size_t)requestChunks * chunkSize * chunkSize));
            fastllm::Data requestKCum = MakeCudaTensor(
                fastllm::DataType::FLOAT16, requestQDims,
                makeChunkSlice(kCumValues, qBegin,
                               (size_t)requestChunks * chunkSize * kDim));
            fastllm::Data requestState = MakeCudaTensor(
                fastllm::DataType::FLOAT16,
                {1, heads, kDim, vDim},
                makeChunkSlice(
                    initialState, (size_t)request * kDim * vDim,
                    (size_t)kDim * vDim));
            fastllm::Data requestOutput;
            FastllmChunkGatedDeltaRulePrefill(
                requestQ, requestK, requestV, requestG, requestAttn,
                requestKCum, requestState, requestOutput);
            FastllmCudaSyncCurrentThreadStream();
            std::vector<float> output = ToFloatVector(requestOutput);
            std::vector<float> state = ToFloatVector(requestState);
            referenceOutputValues.insert(
                referenceOutputValues.end(), output.begin(),
                output.begin() + (size_t)seqLens[request] * vDim);
            referenceStateValues.insert(
                referenceStateValues.end(), state.begin(), state.end());
        }

        fastllm::Data q = MakeCudaTensor(
            fastllm::DataType::FLOAT16, qDims, qValues);
        fastllm::Data k = MakeCudaTensor(
            fastllm::DataType::FLOAT16, qDims, kValues);
        fastllm::Data v = MakeCudaTensor(
            fastllm::DataType::FLOAT16, vDims, vValues);
        fastllm::Data g = MakeCudaTensor(
            fastllm::DataType::FLOAT16, gDims, gValues);
        fastllm::Data attn = MakeCudaTensor(
            fastllm::DataType::FLOAT16, attnDims, attnValues);
        fastllm::Data kCum = MakeCudaTensor(
            fastllm::DataType::FLOAT16, qDims, kCumValues);

        fastllm::Data nativeState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, initialState);
        fastllm::Data nativeOutput;
        Expect(FastllmChunkGatedDeltaRuleVarlenPrefillNative(
                   q, k, v, g, attn, kCum, nativeState,
                   seqLens, nativeOutput),
               "native packed varlen GDN prefill failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectCudaTensorMeta(
            nativeOutput, fastllm::DataType::FLOAT16,
            {1, totalTokens, heads, vDim},
            "native direct ragged GDN output");
        ExpectFloatNear(referenceStateValues, ToFloatVector(nativeState),
                        2e-3f, 2e-3f,
                        "native packed varlen GDN recurrent state");
        ExpectFloatNear(referenceOutputValues, ToFloatVector(nativeOutput),
                        2e-3f, 2e-3f,
                        "native packed varlen GDN output");

        fastllm::Data selectedState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, initialState);
        fastllm::Data selectedOutput;
        Expect(RunCudaVarlenChunkGdnSelected(
                   q, k, v, g, attn, attn, kCum, selectedState,
                   false, false, seqLens, selectedOutput,
                   "packed varlen GDN"),
               "selected packed varlen GDN prefill failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectCudaTensorMeta(
            selectedOutput, fastllm::DataType::FLOAT16,
            {1, totalTokens, heads, vDim},
            "selected direct ragged GDN output");
        ExpectFloatNear(referenceStateValues, ToFloatVector(selectedState),
                        2e-3f, 2e-3f,
                        "selected packed varlen GDN recurrent state");
        ExpectFloatNear(referenceOutputValues, ToFloatVector(selectedOutput),
                        2e-3f, 2e-3f,
                        "selected packed varlen GDN output");
        std::cout << "CUDA packed varlen chunk GDN regression: PASS\n";
    }

    void RunCudaVarlenChunkGdnHeadMappingRegression() {
        FastllmCudaSetDevice(0);
        constexpr int keyHeads = 2;
        constexpr int valueHeads = 6;
        constexpr int headGroup = valueHeads / keyHeads;
        constexpr int chunkSize = 64;
        constexpr int kDim = 128;
        constexpr int vDim = 128;
        const std::vector<int> seqLens = {65, 7};
        const int batch = (int)seqLens.size();
        const int totalTokens =
            std::accumulate(seqLens.begin(), seqLens.end(), 0);
        int totalChunks = 0;
        for (int len : seqLens) {
            totalChunks += (len + chunkSize - 1) / chunkSize;
        }
        const size_t keyMatrixElements =
            (size_t)totalChunks * chunkSize * kDim;
        const size_t attnElements =
            (size_t)totalChunks * chunkSize * chunkSize;
        std::vector<float> logicalQValues = MakeRegressionValues(
            keyHeads * keyMatrixElements, 0.29f, 0.003f);
        std::vector<float> logicalKValues = MakeRegressionValues(
            keyHeads * keyMatrixElements, 0.47f, 0.002f);
        std::vector<float> logicalAttnValues = MakeRegressionValues(
            keyHeads * attnElements, 0.61f, 0.0007f);
        std::vector<float> physicalQValues(
            valueHeads * keyMatrixElements);
        std::vector<float> physicalKValues(
            valueHeads * keyMatrixElements);
        std::vector<float> physicalAttnValues(
            valueHeads * attnElements);
        for (int valueHead = 0; valueHead < valueHeads; valueHead++) {
            int keyHead = valueHead / headGroup;
            std::copy_n(
                logicalQValues.begin() + keyHead * keyMatrixElements,
                keyMatrixElements,
                physicalQValues.begin() + valueHead * keyMatrixElements);
            std::copy_n(
                logicalKValues.begin() + keyHead * keyMatrixElements,
                keyMatrixElements,
                physicalKValues.begin() + valueHead * keyMatrixElements);
            std::copy_n(
                logicalAttnValues.begin() + keyHead * attnElements,
                attnElements,
                physicalAttnValues.begin() + valueHead * attnElements);
        }
        const std::vector<int> logicalQDims =
            {1, keyHeads, totalChunks, chunkSize, kDim};
        const std::vector<int> physicalQDims =
            {1, valueHeads, totalChunks, chunkSize, kDim};
        const std::vector<int> vDims =
            {1, valueHeads, totalChunks, chunkSize, vDim};
        const std::vector<int> gDims =
            {1, valueHeads, totalChunks, chunkSize};
        const std::vector<int> logicalAttnDims =
            {1, keyHeads, totalChunks, chunkSize, chunkSize};
        const std::vector<int> physicalAttnDims =
            {1, valueHeads, totalChunks, chunkSize, chunkSize};
        const std::vector<int> stateDims =
            {batch, valueHeads, kDim, vDim};
        std::vector<float> vValues = MakeRegressionValues(
            (size_t)valueHeads * totalChunks * chunkSize * vDim,
            0.83f, 0.004f);
        std::vector<float> gValues = MakeRegressionValues(
            (size_t)valueHeads * totalChunks * chunkSize,
            -0.019f, 0.00003f);
        std::vector<float> decayValues = MakeRegressionValues(
            valueHeads * attnElements, 0.91f, 0.0002f);
        std::vector<float> kCumValues = MakeRegressionValues(
            (size_t)valueHeads * totalChunks * chunkSize * kDim,
            1.07f, 0.002f);
        std::vector<float> stateValues = MakeRegressionValues(
            (size_t)batch * valueHeads * kDim * vDim,
            1.23f, 0.001f);

        fastllm::Data physicalQ = MakeCudaTensor(
            fastllm::DataType::FLOAT16, physicalQDims, physicalQValues);
        fastllm::Data physicalK = MakeCudaTensor(
            fastllm::DataType::FLOAT16, physicalQDims, physicalKValues);
        fastllm::Data logicalQ = MakeCudaTensor(
            fastllm::DataType::FLOAT16, logicalQDims, logicalQValues);
        fastllm::Data logicalK = MakeCudaTensor(
            fastllm::DataType::FLOAT16, logicalQDims, logicalKValues);
        fastllm::Data v = MakeCudaTensor(
            fastllm::DataType::FLOAT16, vDims, vValues);
        fastllm::Data g = MakeCudaTensor(
            fastllm::DataType::FLOAT16, gDims, gValues);
        fastllm::Data physicalAttn = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            physicalAttnDims, physicalAttnValues);
        fastllm::Data logicalAttn = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            logicalAttnDims, logicalAttnValues);
        fastllm::Data decay = MakeCudaTensor(
            fastllm::DataType::FLOAT16,
            physicalAttnDims, decayValues);
        fastllm::Data kCum = MakeCudaTensor(
            fastllm::DataType::FLOAT16, physicalQDims, kCumValues);

        fastllm::Data referenceState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, stateValues);
        fastllm::Data referenceOutput;
        Expect(FastllmChunkGatedDeltaRuleVarlenPrefillNative(
                   physicalQ, physicalK, v, g, physicalAttn,
                   kCum, referenceState, seqLens, referenceOutput,
                   &decay, true),
               "physical-head varlen GDN reference failed");
        fastllm::Data mappedNativeState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, stateValues);
        fastllm::Data mappedNativeOutput;
        Expect(FastllmChunkGatedDeltaRuleVarlenPrefillNative(
                   logicalQ, logicalK, v, g, logicalAttn,
                   kCum, mappedNativeState, seqLens, mappedNativeOutput,
                   &decay, true),
               "logical-head native varlen GDN failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(ToFloatVector(referenceState),
                        ToFloatVector(mappedNativeState),
                        0.0f, 0.0f,
                        "logical-head native varlen GDN state");
        ExpectFloatNear(ToFloatVector(referenceOutput),
                        ToFloatVector(mappedNativeOutput),
                        0.0f, 0.0f,
                        "logical-head native varlen GDN output");

        fastllm::Data selectedState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, stateValues);
        fastllm::Data selectedOutput;
        Expect(RunCudaVarlenChunkGdnSelected(
                   logicalQ, logicalK, v, g, logicalAttn, decay,
                   kCum, selectedState, true, false,
                   seqLens, selectedOutput,
                   "logical-head varlen GDN"),
               "selected logical-head varlen GDN failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectCudaTensorMeta(
            selectedOutput, fastllm::DataType::FLOAT16,
            {1, totalTokens, valueHeads, vDim},
            "selected logical-head direct varlen output");
        ExpectFloatNear(ToFloatVector(referenceState),
                        ToFloatVector(selectedState),
                        2e-3f, 2e-3f,
                        "selected logical-head varlen GDN state");
        ExpectFloatNear(ToFloatVector(referenceOutput),
                        ToFloatVector(selectedOutput),
                        2e-3f, 2e-3f,
                        "selected logical-head varlen GDN output");

        // Exercise the direct-QK O variant against the legacy FP16
        // materialization boundary. This comparison keeps QK and the decay
        // mask explicit only on the reference side.
        fastllm::Data directAttn;
        directAttn.dataType = fastllm::DataType::FLOAT16;
        directAttn.Resize(logicalAttnDims);
        directAttn.ToDevice(
            fastllm::DataDevice::CUDA, std::vector<int>{0});
        directAttn.Allocate(false);
        Expect(FastllmCudaBatchMatMulTransB(
                   logicalQ, logicalK, directAttn,
                   chunkSize * kDim, chunkSize * kDim,
                   chunkSize * chunkSize, kDim, kDim,
                   keyHeads * totalChunks,
                   chunkSize, kDim, chunkSize, 1.0f),
               "direct-QK legacy attention reference failed");
        fastllm::Data directDecay;
        directDecay.dataType = fastllm::DataType::FLOAT16;
        directDecay.Resize(physicalAttnDims);
        directDecay.ToDevice(
            fastllm::DataDevice::CUDA, std::vector<int>{0});
        directDecay.Allocate(false);
        Expect(FastllmCudaMakeDecayMask(g, directDecay),
               "direct-QK decay reference failed");

        fastllm::Data directReferenceState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, stateValues);
        fastllm::Data directReferenceOutput;
        Expect(FastllmChunkGatedDeltaRuleVarlenPrefillNative(
                   logicalQ, logicalK, v, g, directAttn,
                   kCum, directReferenceState, seqLens,
                   directReferenceOutput, &directDecay, true),
               "direct-QK native reference failed");
        fastllm::Data directSelectedState = MakeCudaTensor(
            fastllm::DataType::FLOAT16, stateDims, stateValues);
        fastllm::Data directSelectedOutput;
        Expect(RunCudaVarlenChunkGdnSelected(
                   logicalQ, logicalK, v, g, directAttn, directDecay,
                   kCum, directSelectedState, true, true,
                   seqLens, directSelectedOutput,
                   "direct-QK logical-head varlen GDN"),
               "selected direct-QK logical-head varlen GDN failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(ToFloatVector(directReferenceState),
                        ToFloatVector(directSelectedState),
                        2e-3f, 2e-3f,
                        "direct-QK logical-head varlen GDN state");
        ExpectFloatNear(ToFloatVector(directReferenceOutput),
                        ToFloatVector(directSelectedOutput),
                        3e-3f, 3e-3f,
                        "direct-QK logical-head varlen GDN output");
        std::cout << "CUDA logical-head varlen chunk GDN regression: PASS\n";
    }

    void RunCudaDeepSeekV4TokenTiledWoARegression() {
        FastllmCudaSetDevice(0);
        constexpr int groups = 2;
        constexpr int heads = 8;
        constexpr int headDim = 1024;
        constexpr int hidden = (heads / groups) * headDim;
        constexpr int outRank = 128;

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
            weightBytes[i] = static_cast<uint8_t>(
                ((i * 29 + i / 97 + 0x11) & 0x7e) |
                (((i / 43) & 1) << 7));
        }
        const int scaleRows = groups * outRank / weight.blockK;
        const int scaleCols = hidden / weight.blockM;
        weight.scales.resize((size_t)scaleRows * scaleCols);
        for (size_t i = 0; i < weight.scales.size(); i++) {
            weight.scales[i] = 1.0f / (float)(16 << (i % 4));
        }
        weight.ToDevice(fastllm::DataDevice::CUDA);

        const char *savedDisable = std::getenv(
            "FASTLLM_DSV4_DISABLE_CUDA_WOA_TOKEN_TILE");
        const bool hadDisable = savedDisable != nullptr;
        const std::string disableValue = hadDisable ? savedDisable : "";
        const char *savedTile = std::getenv(
            "FASTLLM_DSV4_CUDA_WOA_TOKENS_PER_BLOCK");
        const bool hadTile = savedTile != nullptr;
        const std::string tileValue = hadTile ? savedTile : "";
        const char *savedRowTile = std::getenv(
            "FASTLLM_DSV4_CUDA_WOA_ROWS_PER_BLOCK");
        const bool hadRowTile = savedRowTile != nullptr;
        const std::string rowTileValue = hadRowTile ? savedRowTile : "";

        const std::vector<std::pair<int, int>> tiles = {
            {2, 4}, {4, 4}, {8, 4}, {2, 8}, {4, 8}, {8, 8}};
        const std::vector<fastllm::DataType> inputTypes = {
            fastllm::DataType::BFLOAT16,
            fastllm::DataType::FLOAT16,
            fastllm::DataType::FLOAT32};

        for (const std::pair<int, int> shape : {
                 std::pair<int, int>{1, 2}, {1, 7}, {1, 8}, {1, 9}, {2, 5}}) {
            const int bsz = shape.first;
            const int seqlen = shape.second;
            const int tokens = bsz * seqlen;
            std::vector<float> inputValues(
                (size_t)tokens * heads * headDim);
            for (size_t i = 0; i < inputValues.size(); i++) {
                inputValues[i] =
                    (float)((int)((i * 17 + i / 31) % 257) - 128) /
                    64.0f;
            }
            for (const fastllm::DataType inputType : inputTypes) {
                fastllm::Data input = MakeCudaTensor(
                    inputType, {bsz, seqlen, heads, headDim}, inputValues);

                setenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_TOKEN_TILE", "1", 1);
                fastllm::Data reference;
                Expect(FastllmCudaDeepSeekV4WoA(
                           input, weight, groups, outRank, reference, false),
                       "one-token DeepSeek-V4 WoA reference rejected input");
                reference.ToDevice(fastllm::DataDevice::CPU);
                for (const auto &tile : tiles) {
                    unsetenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_TOKEN_TILE");
                    setenv("FASTLLM_DSV4_CUDA_WOA_TOKENS_PER_BLOCK",
                           std::to_string(tile.first).c_str(), 1);
                    setenv("FASTLLM_DSV4_CUDA_WOA_ROWS_PER_BLOCK",
                           std::to_string(tile.second).c_str(), 1);
                    fastllm::Data actual;
                    Expect(FastllmCudaDeepSeekV4WoA(
                               input, weight, groups, outRank, actual, false),
                           "token-tiled DeepSeek-V4 WoA rejected input");
                    actual.ToDevice(fastllm::DataDevice::CPU);
                    Expect(reference.dataType == fastllm::DataType::BFLOAT16 &&
                               actual.dataType == fastllm::DataType::BFLOAT16 &&
                               reference.GetBytes() == actual.GetBytes(),
                           "token-tiled DeepSeek-V4 WoA output metadata mismatch");
                    Expect(std::memcmp(reference.cpuData, actual.cpuData,
                                       reference.GetBytes()) == 0,
                           "token-tiled DeepSeek-V4 WoA changed BF16 output bits at "
                           "input_type=" + std::to_string((int)inputType) +
                           " batch=" + std::to_string(bsz) +
                           " tokens=" + std::to_string(seqlen) +
                           " tile=" + std::to_string(tile.first) + "x" +
                           std::to_string(tile.second));
                }
            }
        }

        if (hadDisable) {
            setenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_TOKEN_TILE",
                   disableValue.c_str(), 1);
        } else {
            unsetenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_TOKEN_TILE");
        }
        if (hadTile) {
            setenv("FASTLLM_DSV4_CUDA_WOA_TOKENS_PER_BLOCK",
                   tileValue.c_str(), 1);
        } else {
            unsetenv("FASTLLM_DSV4_CUDA_WOA_TOKENS_PER_BLOCK");
        }
        if (hadRowTile) {
            setenv("FASTLLM_DSV4_CUDA_WOA_ROWS_PER_BLOCK",
                   rowTileValue.c_str(), 1);
        } else {
            unsetenv("FASTLLM_DSV4_CUDA_WOA_ROWS_PER_BLOCK");
        }
        std::cout << "DeepSeek-V4 CUDA token-tiled WoA regression: PASS "
                  << "(BF16 output bitwise, BF16/FP16/FP32 inputs, K=4096, "
                     "tiles=2/4/8 x rows=4/8)\n";
    }

    void RunCudaDeepSeekV4TokenTiledWoABenchmark() {
        FastllmCudaSetDevice(0);
        constexpr int groups = 8;
        constexpr int heads = 64;
        constexpr int headDim = 512;
        constexpr int hidden = (heads / groups) * headDim;
        constexpr int outRank = 1024;

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
            weightBytes[i] = static_cast<uint8_t>(
                0x18 + ((i * 13 + i / 127) & 0x3f));
        }
        const int scaleRows = groups * outRank / weight.blockK;
        const int scaleCols = hidden / weight.blockM;
        weight.scales.resize((size_t)scaleRows * scaleCols);
        for (size_t i = 0; i < weight.scales.size(); i++) {
            weight.scales[i] = 1.0f / (float)(32 << (i % 3));
        }
        weight.ToDevice(fastllm::DataDevice::CUDA);

        const char *benchTokens = std::getenv(
            "FASTLLM_DSV4_WOA_BENCH_TOKENS");
        const int tokens = benchTokens == nullptr ? 2048 :
            std::max(1, std::atoi(benchTokens));
        fastllm::Data input(
            fastllm::DataType::BFLOAT16,
            {1, tokens, heads, headDim});
        input.Allocate(false);
        uint16_t *inputBits = reinterpret_cast<uint16_t*>(input.cpuData);
        for (uint64_t i = 0; i < input.Count(0); i++) {
            const float value =
                (float)((int)((i * 17 + i / 31) % 257) - 128) / 64.0f;
            inputBits[i] = fastllm::Float32ToBFloat16RNEBits(value);
        }
        input.ToDevice(fastllm::DataDevice::CUDA);

        auto measure = [&](const char *label, int tokenTile, int rowTile,
                           bool disabled) {
            if (disabled) {
                setenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_TOKEN_TILE", "1", 1);
            } else {
                unsetenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_TOKEN_TILE");
                setenv("FASTLLM_DSV4_CUDA_WOA_TOKENS_PER_BLOCK",
                       std::to_string(tokenTile).c_str(), 1);
                setenv("FASTLLM_DSV4_CUDA_WOA_ROWS_PER_BLOCK",
                       std::to_string(rowTile).c_str(), 1);
            }
            fastllm::Data output;
            for (int i = 0; i < 2; i++) {
                Expect(FastllmCudaDeepSeekV4WoA(
                           input, weight, groups, outRank, output, false),
                       "DeepSeek-V4 WoA benchmark launch failed");
            }
            ForceDeviceSync();
            constexpr int iterations = 5;
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; i++) {
                Expect(FastllmCudaDeepSeekV4WoA(
                           input, weight, groups, outRank, output, false),
                       "DeepSeek-V4 WoA benchmark launch failed");
            }
            ForceDeviceSync();
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count() / iterations;
            std::cout << "woa tokens=" << tokens << " " << label
                      << " ms=" << milliseconds << "\n";
        };

        measure("one-token", 1, 4, true);
        measure("tile=2x4", 2, 4, false);
        measure("tile=4x4", 4, 4, false);
        measure("tile=8x4", 8, 4, false);
        measure("tile=2x8", 2, 8, false);
        measure("tile=4x8", 4, 8, false);
        measure("tile=8x8", 8, 8, false);
        unsetenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_TOKEN_TILE");
        unsetenv("FASTLLM_DSV4_CUDA_WOA_TOKENS_PER_BLOCK");
        unsetenv("FASTLLM_DSV4_CUDA_WOA_ROWS_PER_BLOCK");
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

        for (int tokens : {1, 8}) {
            std::vector<float> inputValues(
                (size_t)tokens * heads * headDim, 1.0f);
            fastllm::Data input = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, tokens, heads, headDim}, inputValues);

            fastllm::Data reference;
            Expect(FastllmCudaDeepSeekV4WoA(
                       input, weight, groups, outRank, reference, false),
                   "built-in DeepSeek-V4 WoA reference rejected its test input");

            fastllm::Data actual = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, tokens, groups * outRank},
                std::vector<float>((size_t)tokens * groups * outRank, 0.0f));
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
            ExpectFloatNear(
                ToFloatVector(reference), ToFloatVector(actual),
                2e-2f, 2e-3f,
                "DeepSeek-V4 Triton WoA output rows=" +
                    std::to_string(tokens));
        }
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
                       q, windowKV, compressedKV, nullptr, nullptr, sink,
                       windowSize, compressRatio,
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
                       q, windowKV, compressedKV, nullptr, nullptr, sink,
                       windowSize, compressRatio,
                       decodeMetaPtr, ropeDim, 10000.0f, 4096, 8.0f, 32, 1,
                       softmaxScale, actual),
                   "DeepSeek-V4 sparse decode wrapper rejected ratio=" +
                       std::to_string(compressRatio));
            ExpectFloatNear(ToFloatVector(reference), ToFloatVector(actual),
                            3e-2f, 3e-3f,
                            "DeepSeek-V4 Triton sparse decode output ratio=" +
                                std::to_string(compressRatio));
        }

        // Exercise the production SM12x tile: TP8 has eight local heads and
        // DSV4 uses a 512-wide latent.  Other architectures transparently run
        // the generic Triton variant with the same inputs.
        constexpr int smHeads = 8;
        constexpr int smHeadDim = 512;
        constexpr int smWindowSize = 128;
        constexpr int smCompressedCapacity = 64;
        constexpr int smStartPos = 127;
        fastllm::Data smQ = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, smHeads, smHeadDim},
            MakeRegressionValues(smHeads * smHeadDim, 0.19f, 0.11f));
        fastllm::Data smWindowKV = MakeCudaTensor(
            fastllm::DataType::FLOAT32,
            {1, smWindowSize, smHeadDim},
            MakeRegressionValues(smWindowSize * smHeadDim, 0.47f, 0.09f));
        fastllm::Data smCompressedKV = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {1, smCompressedCapacity, smHeadDim},
            MakeRegressionValues(
                smCompressedCapacity * smHeadDim, 0.83f, 0.08f));
        fastllm::Data smSink = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {smHeads},
            MakeRegressionValues(smHeads, 1.29f, 0.07f));
        fastllm::Data smDecodeMeta =
            MakeIntTensor({2}, {smStartPos, 456});
        smDecodeMeta.ToDevice(fastllm::DataDevice::CUDA);
        const int32_t *smDecodeMetaPtr =
            reinterpret_cast<const int32_t*>(smDecodeMeta.cudaData);
        float smSoftmaxScale = 1.0f / std::sqrt((float)smHeadDim);
        fastllm::Data smReference, smActual;
        Expect(FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraph(
                   smQ, smWindowKV, smCompressedKV, nullptr, nullptr, smSink,
                   smWindowSize, 4,
                   smDecodeMetaPtr, 64, 10000.0f, 4096, 8.0f, 32, 1,
                   smSoftmaxScale, smReference, false),
               "built-in DeepSeek-V4 sparse decode rejected SM120 shape");
        Expect(FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraph(
                   smQ, smWindowKV, smCompressedKV, nullptr, nullptr, smSink,
                   smWindowSize, 4,
                   smDecodeMetaPtr, 64, 10000.0f, 4096, 8.0f, 32, 1,
                   smSoftmaxScale, smActual),
               "DeepSeek-V4 Triton sparse decode rejected SM120 shape");
        ExpectFloatNear(ToFloatVector(smReference), ToFloatVector(smActual),
                        5e-2f, 5e-3f,
                        "DeepSeek-V4 Triton sparse decode SM120 tile output");
        std::cout << "DeepSeek-V4 Triton sparse decode regression: PASS ("
                  << (tritonEnabled ? "Triton" : "disabled-gate fallback")
                  << ")\n";
    }

    void RunCudaDeepSeekV4IndexerRegression() {
        FastllmCudaSetDevice(0);
        constexpr int heads = 64;
        constexpr int headDim = 128;
        constexpr int topk = 512;
        constexpr int compressedRows = 640;
        constexpr int startPos = compressedRows * 4 - 1;

        fastllm::Data q = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, heads, headDim},
            MakeRegressionValues(heads * headDim, 0.271f, 0.37f));
        fastllm::Data weights = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, 1, heads},
            MakeRegressionValues(heads, 0.613f, 0.29f));
        fastllm::Data compressedKV = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {1, compressedRows, headDim},
            MakeRegressionValues(compressedRows * headDim, 0.947f, 0.41f));
        fastllm::Data decodeMeta =
            MakeIntTensor({2}, {startPos, 123});
        decodeMeta.ToDevice(fastllm::DataDevice::CUDA);
        const int32_t *decodeMetaPtr =
            reinterpret_cast<const int32_t *>(decodeMeta.cudaData);

        fastllm::Data optimizedIndices, optimizedLengths;
        Expect(FastllmCudaDeepSeekV4BuildIndexerTopKGraph(
                   q, weights, compressedKV, decodeMetaPtr, 4,
                   160000.0f, 65536, 16.0f, 32, 1,
                   optimizedIndices, optimizedLengths),
               "DeepSeek-V4 optimized indexer rejected a valid long row");

        ExpectIntEqual({topk}, ToIntVector(optimizedLengths),
                       "DeepSeek-V4 optimized indexer length");
        std::vector<int32_t> optimized = ToIntVector(optimizedIndices);
        auto validateSelection = [&](const std::vector<int32_t> &selection,
                                     const std::string &name) {
            Expect((int)selection.size() == topk,
                   name + " returned the wrong number of indices");
            std::vector<int32_t> sorted = selection;
            std::sort(sorted.begin(), sorted.end());
            Expect(sorted.front() >= 0 && sorted.back() < compressedRows,
                   name + " returned an out-of-range index");
            Expect(std::adjacent_find(sorted.begin(), sorted.end()) ==
                       sorted.end(),
                   name + " returned duplicate indices");
            return sorted;
        };
        validateSelection(optimized, "DeepSeek-V4 optimized indexer");

        // vLLM bypasses q/score/top-k entirely while the compressed row has no
        // more than 512 candidates.  The public helper must preserve that exact
        // ascending-order contract as well.
        fastllm::Data shortMeta =
            MakeIntTensor({2}, {127 * 4 - 1, 456});
        shortMeta.ToDevice(fastllm::DataDevice::CUDA);
        fastllm::Data shortIndices, shortLengths;
        Expect(FastllmCudaDeepSeekV4BuildIndexerTopKGraph(
                   q, weights, compressedKV,
                   reinterpret_cast<const int32_t *>(shortMeta.cudaData), 4,
                   160000.0f, 65536, 16.0f, 32, 1,
                   shortIndices, shortLengths),
               "DeepSeek-V4 indexer rejected a valid short row");
        ExpectIntEqual({127}, ToIntVector(shortLengths),
                       "DeepSeek-V4 short indexer length");
        std::vector<int32_t> shortSelection = ToIntVector(shortIndices);
        for (int index = 0; index < topk; index++) {
            int32_t expected = index < 127 ? index : -1;
            Expect(shortSelection[index] == expected,
                   "DeepSeek-V4 short indexer is not ascending");
        }
        std::cout << "DeepSeek-V4 learned indexer regression: PASS\n";
    }

    void RunCudaDeepSeekV4FullWindowAppendRegression() {
        const int originalDevice = FastllmCudaGetDevice();
        FastllmCudaSetDevice(0);
        constexpr int bsz = 1;
        constexpr int windowSize = 8;
        constexpr int appendTokens = 13;
        constexpr int headDim = 5;

        fastllm::Data source = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {bsz, appendTokens, headDim},
            MakeRegressionValues(appendTokens * headDim, 0.37f, 0.17f));
        const std::vector<float> sourceValues = ToFloatVector(source);

        fastllm::Data longWindow = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {bsz, windowSize, headDim},
            MakeRegressionValues(windowSize * headDim, 0.71f, 0.13f));
        Expect(FastllmCudaDeepSeekV4AppendFullWindowKVCache(
                   source, appendTokens, longWindow),
               "DeepSeek-V4 full-window append rejected a long prefill chunk");
        ForceDeviceSync();
        const std::vector<float> expectedLong(
            sourceValues.begin() +
                (appendTokens - windowSize) * headDim,
            sourceValues.end());
        ExpectFloatNear(expectedLong, ToFloatVector(longWindow), 0.0f, 0.0f,
                        "DeepSeek-V4 long-prefill full-window append");

        constexpr int shortAppend = 3;
        fastllm::Data shortWindow = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {bsz, windowSize, headDim},
            MakeRegressionValues(windowSize * headDim, 0.91f, 0.11f));
        const std::vector<float> shortWindowValues = ToFloatVector(shortWindow);
        Expect(FastllmCudaDeepSeekV4AppendFullWindowKVCache(
                   source, shortAppend, shortWindow),
               "DeepSeek-V4 full-window append rejected a decode-sized suffix");
        ForceDeviceSync();
        std::vector<float> expectedShort(
            shortWindowValues.begin() + shortAppend * headDim,
            shortWindowValues.end());
        expectedShort.insert(
            expectedShort.end(), sourceValues.begin(),
            sourceValues.begin() + shortAppend * headDim);
        ExpectFloatNear(expectedShort, ToFloatVector(shortWindow), 0.0f, 0.0f,
                        "DeepSeek-V4 decode-sized full-window append");
        FastllmCudaSetDevice(originalDevice);
        std::cout << "DeepSeek-V4 full-window append regression: PASS\n";
    }

    void RunCudaPeerAccessInitRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "CUDA peer-access initialization regression: "
                         "SKIP (two GPUs required)\n";
            return;
        }

        int canAccess01 = 0;
        int canAccess10 = 0;
        cudaError_t state01 = cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
        cudaError_t state10 = cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
        if (state01 != cudaSuccess || state10 != cudaSuccess ||
            !canAccess01 || !canAccess10) {
            cudaGetLastError();
            std::cout << "CUDA peer-access initialization regression: "
                         "SKIP (bidirectional P2P unavailable)\n";
            return;
        }

        const int originalDevice = FastllmCudaGetDevice();
        Expect(FastllmCudaPeerAccessInit({0, 1}),
               "CUDA peer access incorrectly depended on custom all-reduce");
        Expect(FastllmCudaGetDevice() == originalDevice,
               "CUDA peer-access initialization changed the current device");
        Expect(FastllmCudaPeerAccessInit({1, 0, 1}),
               "CUDA peer-access initialization did not reuse a canonical topology");
        Expect(FastllmCudaGetDevice() == originalDevice,
               "cached CUDA peer-access initialization changed the current device");
        std::cout << "CUDA peer-access initialization regression: PASS\n";
    }

    void RunMultiCudaDeepSeekV4SparsePrefillRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "DeepSeek-V4 multi-CUDA sparse prefill regression: "
                         "SKIP (two GPUs required)\n";
            return;
        }

        const int originalDevice = FastllmCudaGetDevice();
        const std::vector<int> devices = {0, 1};
        constexpr int batch = 1;
        constexpr int seqlen = 17;
        constexpr int heads = 32;
        constexpr int headDim = 512;
        constexpr int windowSize = 128;
        constexpr int ropeDim = 64;
        const std::vector<int> qDims = {batch, seqlen, heads, headDim};
        const std::vector<int> kvDims = {batch, seqlen, headDim};
        const std::vector<float> qValues = MakeRegressionValues(
            batch * seqlen * heads * headDim, 0.27f, 0.011f);
        const std::vector<float> kvValues = MakeRegressionValues(
            batch * seqlen * headDim, 0.83f, 0.017f);
        const std::vector<float> sinkValues = MakeRegressionValues(
            heads, 1.31f, 0.09f);

        std::vector<std::unique_ptr<fastllm::Data>> queries;
        std::vector<std::unique_ptr<fastllm::Data>> keys;
        std::vector<std::unique_ptr<fastllm::Data>> sinks;
        std::vector<std::unique_ptr<fastllm::Data>> outputs;
        for (int device : devices) {
            FastllmCudaSetDevice(device);
            queries.emplace_back(std::make_unique<fastllm::Data>(
                fastllm::DataType::BFLOAT16, qDims, qValues));
            keys.emplace_back(std::make_unique<fastllm::Data>(
                fastllm::DataType::BFLOAT16, kvDims, kvValues));
            sinks.emplace_back(std::make_unique<fastllm::Data>(
                fastllm::DataType::FLOAT32, std::vector<int>{heads}, sinkValues));
            outputs.emplace_back(std::make_unique<fastllm::Data>());
            const std::vector<int> targetDevice = {device};
            queries.back()->ToDevice(fastllm::DataDevice::CUDA, targetDevice);
            keys.back()->ToDevice(fastllm::DataDevice::CUDA, targetDevice);
            sinks.back()->ToDevice(fastllm::DataDevice::CUDA, targetDevice);
            Expect(GetPointerDeviceId(queries.back()->cudaData) == device &&
                       GetPointerDeviceId(keys.back()->cudaData) == device &&
                       GetPointerDeviceId(sinks.back()->cudaData) == device,
                   "DeepSeek-V4 sparse prefill fixture was allocated on the wrong GPU");
        }

        std::vector<char> accepted(devices.size(), 0);
        bool dispatched = fastllm::MultiCudaRunDeviceCallbacks(
            devices, [&](int rank, int device) {
                accepted[rank] = FastllmCudaDeepSeekV4SparseAttentionPrefill(
                    *queries[rank], *keys[rank], *sinks[rank],
                    windowSize, 0, 0, ropeDim, 10000.0f, 65536, 16.0f,
                    32, 1, 1.0f / std::sqrt((float)headDim),
                    *outputs[rank], 0);
                if (accepted[rank]) {
                    accepted[rank] =
                        outputs[rank]->dims == qDims &&
                        GetPointerDeviceId(outputs[rank]->cudaData) == device;
                }
            });
        Expect(dispatched,
               "DeepSeek-V4 sparse prefill did not dispatch on multi-CUDA workers");
        for (int rank = 0; rank < (int)devices.size(); rank++) {
            Expect(accepted[rank] != 0,
                   "DeepSeek-V4 sparse prefill rejected GPU " +
                       std::to_string(devices[rank]));
        }

        FastllmCudaSetDevice(devices[0]);
        const std::vector<float> reference = ToFloatVector(*outputs[0]);
        FastllmCudaSetDevice(devices[1]);
        const std::vector<float> actual = ToFloatVector(*outputs[1]);
        ExpectFloatNear(reference, actual, 0.0f, 0.0f,
                        "DeepSeek-V4 multi-CUDA sparse prefill output");

        // Prefix-cache hits resume from a 256-token boundary.  Cover a suffix
        // larger than the 128-token local window, including the compressed KV
        // rows present at a 2K-token restore point.  This is the geometry that
        // previously deferred an illegal CUDA access until the following MoE
        // host handoff.
        constexpr int cachedStartPos = 2048;
        constexpr int cachedSeqlen = 156;
        constexpr int cachedPrefixLen = windowSize;
        constexpr int cachedCompressedCount = 551;
        constexpr int cachedKvLen =
            cachedPrefixLen + cachedSeqlen + cachedCompressedCount;
        const std::vector<int> cachedQDims =
            {batch, cachedSeqlen, heads, headDim};
        const std::vector<int> cachedKvDims =
            {batch, cachedKvLen, headDim};
        const std::vector<float> cachedQValues = MakeRegressionValues(
            batch * cachedSeqlen * heads * headDim, 0.43f, 4.0f);
        const std::vector<float> cachedKvValues = MakeRegressionValues(
            batch * cachedKvLen * headDim, 0.91f, 1.5f);
        queries.clear();
        keys.clear();
        sinks.clear();
        outputs.clear();
        for (int device : devices) {
            FastllmCudaSetDevice(device);
            queries.emplace_back(std::make_unique<fastllm::Data>(
                fastllm::DataType::BFLOAT16, cachedQDims, cachedQValues));
            keys.emplace_back(std::make_unique<fastllm::Data>(
                fastllm::DataType::BFLOAT16, cachedKvDims, cachedKvValues));
            sinks.emplace_back(std::make_unique<fastllm::Data>(
                fastllm::DataType::FLOAT32, std::vector<int>{heads}, sinkValues));
            outputs.emplace_back(std::make_unique<fastllm::Data>());
            const std::vector<int> targetDevice = {device};
            queries.back()->ToDevice(fastllm::DataDevice::CUDA, targetDevice);
            keys.back()->ToDevice(fastllm::DataDevice::CUDA, targetDevice);
            sinks.back()->ToDevice(fastllm::DataDevice::CUDA, targetDevice);
        }
        accepted.assign(devices.size(), 0);
        dispatched = fastllm::MultiCudaRunDeviceCallbacks(
            devices, [&](int rank, int device) {
                accepted[rank] = FastllmCudaDeepSeekV4SparseAttentionPrefill(
                    *queries[rank], *keys[rank], *sinks[rank],
                    windowSize, cachedStartPos, 4, ropeDim, 160000.0f,
                    65536, 16.0f, 32, 1,
                    1.0f / std::sqrt((float)headDim),
                    *outputs[rank], cachedPrefixLen);
                if (accepted[rank]) {
                    accepted[rank] =
                        outputs[rank]->dims == cachedQDims &&
                        GetPointerDeviceId(outputs[rank]->cudaData) == device;
                }
            });
        Expect(dispatched,
               "DeepSeek-V4 cache-hit sparse prefill did not dispatch on multi-CUDA workers");
        for (int rank = 0; rank < (int)devices.size(); rank++) {
            Expect(accepted[rank] != 0,
                   "DeepSeek-V4 cache-hit sparse prefill rejected GPU " +
                       std::to_string(devices[rank]));
        }
        FastllmCudaSetDevice(devices[0]);
        const std::vector<float> cachedReference = ToFloatVector(*outputs[0]);
        FastllmCudaSetDevice(devices[1]);
        const std::vector<float> cachedActual = ToFloatVector(*outputs[1]);
        ExpectFloatNear(cachedReference, cachedActual, 4e-2f, 4e-3f,
                        "DeepSeek-V4 cache-hit multi-CUDA sparse prefill output");
        FastllmCudaSetDevice(originalDevice);
        std::cout << "DeepSeek-V4 multi-CUDA sparse prefill regression: PASS\n";
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

    void RunCudaDeepSeekV4FusedRouterRegression() {
        FastllmCudaSetDevice(0);
        constexpr int experts = 256;
        constexpr int topk = 6;
        constexpr float routeScale = 1.5f;

        std::vector<float> biasValues(experts);
        for (int expert = 0; expert < experts; expert++) {
            biasValues[expert] =
                ((expert * 11) % 17 - 8) * 4.0e-4f + expert * 1.0e-7f;
        }
        fastllm::Data gateBias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {experts}, biasValues);

        bool tritonGate = fastllm::GetFastllmEnv().cudaTriton &&
            RegressionEnvFlagDefaultEnabled(
                "FASTLLM_CUDA_TRITON_DEEPSEEK_V4_ROUTER", true) &&
            RegressionEnvFlagDefaultEnabled(
                "FASTLLM_CUDA_TRITON_DSV4_ROUTER_SM120", true);
        int major = 0, minor = 0;
#ifndef USE_ROCM
        int device = 0;
        bool haveArch = cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(
                &major, cudaDevAttrComputeCapabilityMajor, device) ==
                cudaSuccess &&
            cudaDeviceGetAttribute(
                &minor, cudaDevAttrComputeCapabilityMinor, device) ==
                cudaSuccess;
#else
        bool haveArch = false;
#endif
        bool expectSm120Triton = tritonGate && haveArch &&
            (major * 10 + minor == 120 || major * 10 + minor == 121);
        bool expectSm120Production = haveArch &&
            (major * 10 + minor == 120 || major * 10 + minor == 121);
        float productionTolerance = expectSm120Triton ? 2.0e-5f : 0.0f;

        for (int tokens : {1, 7, 127}) {
            std::vector<float> logitsValues((size_t)tokens * experts);
            for (int token = 0; token < tokens; token++) {
                for (int expert = 0; expert < experts; expert++) {
                    int permuted = (expert * 37 + token * 17) % 257;
                    logitsValues[(size_t)token * experts + expert] =
                        (permuted - 128) / 24.0f + expert * 1.0e-5f;
                }
                logitsValues[(size_t)token * experts] = -25.0f - token;
                logitsValues[(size_t)token * experts + 1] = 25.0f + token;
            }

            fastllm::Data referenceLogits = MakeCudaTensor(
                fastllm::DataType::FLOAT32, {tokens, experts}, logitsValues);
            fastllm::Data fusedLogits = MakeCudaTensor(
                fastllm::DataType::FLOAT32, {tokens, experts}, logitsValues);
            fastllm::Data referenceIndex = MakeIntTensor(
                {tokens, topk}, std::vector<int32_t>(tokens * topk, -1));
            referenceIndex.ToDevice(fastllm::DataDevice::CUDA);
            fastllm::Data referenceScore = MakeCudaTensor(
                fastllm::DataType::FLOAT32, {tokens, topk},
                std::vector<float>(tokens * topk, 0.0f));
            Expect(FastllmCudaDeepSeekV4RouteScoreTransform(
                       referenceLogits, 2),
                   "DeepSeek-V4 legacy sqrt-softplus transform failed");
            Expect(FastllmCudaSelectExpert(
                       referenceLogits, &gateBias,
                       referenceIndex, referenceScore,
                       topk, true, routeScale),
                   "DeepSeek-V4 legacy top-6 reference failed");
            FastllmCudaSyncCurrentThreadStream();
            const std::vector<int32_t> expectedIndex =
                ToIntVector(referenceIndex);
            const std::vector<float> expectedScore =
                ToFloatVector(referenceScore);

            fastllm::Data genericIndex, genericScore;
            Expect(FastllmCudaDeepSeekV4SqrtSoftplusRouter(
                       fusedLogits, gateBias, routeScale,
                       genericIndex, genericScore, false),
                   "generic DeepSeek-V4 fused router rejected a valid input");
            ExpectIntEqual(expectedIndex, ToIntVector(genericIndex),
                           "generic DeepSeek-V4 fused router indices");
            ExpectFloatNear(expectedScore, ToFloatVector(genericScore),
                            2.0e-6f, 2.0e-6f,
                            "generic DeepSeek-V4 fused router scores");
            ExpectFloatNear(logitsValues, ToFloatVector(fusedLogits),
                            0.0f, 0.0f,
                            "generic DeepSeek-V4 fused router input");

            // Exercise the production dispatch as well. With Triton disabled
            // this selects the native one-warp-per-row SM120 kernel; with it
            // enabled it continues to validate the preferred Triton kernel.
            fastllm::Data productionIndex, productionScore;
            Expect(FastllmCudaDeepSeekV4SqrtSoftplusRouter(
                       fusedLogits, gateBias, routeScale,
                       productionIndex, productionScore, true),
                   "production DeepSeek-V4 fused router rejected a valid input");
            ExpectIntEqual(expectedIndex, ToIntVector(productionIndex),
                           "production DeepSeek-V4 fused router indices");
            ExpectFloatNear(expectedScore, ToFloatVector(productionScore),
                            productionTolerance, productionTolerance,
                            "production DeepSeek-V4 fused router scores");

            for (int token = 0; token < tokens; token++) {
                float sum = std::accumulate(
                    expectedScore.begin() + token * topk,
                    expectedScore.begin() + (token + 1) * topk, 0.0f);
                Expect(std::fabs(sum - routeScale) < 2.0e-5f,
                       "DeepSeek-V4 fused router score sum mismatch");
            }

            fastllm::Data tritonIndex = MakeIntTensor(
                {tokens, topk}, std::vector<int32_t>(tokens * topk, -1));
            tritonIndex.ToDevice(fastllm::DataDevice::CUDA);
            fastllm::Data tritonScore = MakeCudaTensor(
                fastllm::DataType::FLOAT32, {tokens, topk},
                std::vector<float>(tokens * topk, 0.0f));
            bool usedTriton =
                fastllm::FastllmCudaTryTritonDeepSeekV4SqrtSoftplusRouter(
                    fusedLogits, gateBias, routeScale,
                    tritonIndex, tritonScore);
            if (expectSm120Triton) {
                Expect(usedTriton,
                       "SM120 DeepSeek-V4 fused router was not selected");
                FastllmCudaSyncCurrentThreadStream();
                ExpectIntEqual(expectedIndex, ToIntVector(tritonIndex),
                               "SM120 DeepSeek-V4 fused router indices");
                ExpectFloatNear(expectedScore, ToFloatVector(tritonScore),
                                2.0e-5f, 2.0e-5f,
                                "SM120 DeepSeek-V4 fused router scores");
            } else {
                Expect(!usedTriton,
                       "DeepSeek-V4 SM120 router ignored its architecture/gate");
            }

            if (tokens == 1) {
                void *graph = nullptr;
                void *graphExec = nullptr;
                Expect(FastllmCudaGraphBeginCapture(),
                       "generic DeepSeek-V4 router graph capture did not start");
                Expect(FastllmCudaDeepSeekV4SqrtSoftplusRouter(
                           fusedLogits, gateBias, routeScale,
                           genericIndex, genericScore, false),
                       "generic DeepSeek-V4 router failed during graph capture");
                Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
                       "generic DeepSeek-V4 router graph capture failed");
                Expect(FastllmCudaGraphInstantiate(graph, &graphExec) &&
                           graphExec != nullptr,
                       "generic DeepSeek-V4 router graph instantiate failed");
                Expect(FastllmCudaGraphLaunch(graphExec),
                       "generic DeepSeek-V4 router graph replay failed");
                FastllmCudaSyncCurrentThreadStream();
                ExpectIntEqual(expectedIndex, ToIntVector(genericIndex),
                               "generic DeepSeek-V4 router graph indices");
                ExpectFloatNear(expectedScore, ToFloatVector(genericScore),
                                2.0e-6f, 2.0e-6f,
                                "generic DeepSeek-V4 router graph scores");
                FastllmCudaGraphExecDestroy(graphExec);
                FastllmCudaGraphDestroy(graph);

                graph = nullptr;
                graphExec = nullptr;
                Expect(FastllmCudaGraphBeginCapture(),
                       "production DeepSeek-V4 router graph capture did not start");
                Expect(FastllmCudaDeepSeekV4SqrtSoftplusRouter(
                           fusedLogits, gateBias, routeScale,
                           productionIndex, productionScore, true),
                       "production DeepSeek-V4 router failed during graph capture");
                Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
                       "production DeepSeek-V4 router graph capture failed");
                Expect(FastllmCudaGraphInstantiate(graph, &graphExec) &&
                           graphExec != nullptr,
                       "production DeepSeek-V4 router graph instantiate failed");
                Expect(FastllmCudaGraphLaunch(graphExec),
                       "production DeepSeek-V4 router graph replay failed");
                FastllmCudaSyncCurrentThreadStream();
                ExpectIntEqual(expectedIndex, ToIntVector(productionIndex),
                               "production DeepSeek-V4 router graph indices");
                ExpectFloatNear(expectedScore, ToFloatVector(productionScore),
                                productionTolerance, productionTolerance,
                                "production DeepSeek-V4 router graph scores");
                FastllmCudaGraphExecDestroy(graphExec);
                FastllmCudaGraphDestroy(graph);

                if (expectSm120Triton) {
                    graph = nullptr;
                    graphExec = nullptr;
                    Expect(FastllmCudaGraphBeginCapture(),
                           "SM120 DeepSeek-V4 router graph capture did not start");
                    Expect(fastllm::
                               FastllmCudaTryTritonDeepSeekV4SqrtSoftplusRouter(
                                   fusedLogits, gateBias, routeScale,
                                   tritonIndex, tritonScore),
                           "SM120 DeepSeek-V4 router failed during graph capture");
                    Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
                           "SM120 DeepSeek-V4 router graph capture failed");
                    Expect(FastllmCudaGraphInstantiate(graph, &graphExec) &&
                               graphExec != nullptr,
                           "SM120 DeepSeek-V4 router graph instantiate failed");
                    Expect(FastllmCudaGraphLaunch(graphExec),
                           "SM120 DeepSeek-V4 router graph replay failed");
                    FastllmCudaSyncCurrentThreadStream();
                    ExpectIntEqual(expectedIndex, ToIntVector(tritonIndex),
                                   "SM120 DeepSeek-V4 router graph indices");
                    ExpectFloatNear(expectedScore, ToFloatVector(tritonScore),
                                    2.0e-5f, 2.0e-5f,
                                    "SM120 DeepSeek-V4 router graph scores");
                    FastllmCudaGraphExecDestroy(graphExec);
                    FastllmCudaGraphDestroy(graph);
                }
            }
        }

        std::vector<float> nonFiniteValues(
            2 * experts, std::numeric_limits<float>::quiet_NaN());
        fastllm::Data nonFiniteLogits = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {2, experts}, nonFiniteValues);
        fastllm::Data nonFiniteIndex, nonFiniteScore;
        Expect(FastllmCudaDeepSeekV4SqrtSoftplusRouter(
                   nonFiniteLogits, gateBias, routeScale,
                   nonFiniteIndex, nonFiniteScore, false),
               "generic DeepSeek-V4 router rejected non-finite input");
        for (int expert : ToIntVector(nonFiniteIndex)) {
            Expect(expert >= 0 && expert < experts,
                   "generic DeepSeek-V4 router returned an invalid expert");
        }
        for (float score : ToFloatVector(nonFiniteScore)) {
            Expect(std::isfinite(score) && score == 0.0f,
                   "generic DeepSeek-V4 router returned a non-finite score");
        }
        fastllm::Data productionNonFiniteIndex, productionNonFiniteScore;
        Expect(FastllmCudaDeepSeekV4SqrtSoftplusRouter(
                   nonFiniteLogits, gateBias, routeScale,
                   productionNonFiniteIndex, productionNonFiniteScore, true),
               "production DeepSeek-V4 router rejected non-finite input");
        for (int expert : ToIntVector(productionNonFiniteIndex)) {
            Expect(expert >= 0 && expert < experts,
                   "production DeepSeek-V4 router returned an invalid expert");
        }
        for (float score : ToFloatVector(productionNonFiniteScore)) {
            Expect(std::isfinite(score) && score == 0.0f,
                   "production DeepSeek-V4 router returned a non-finite score");
        }
        if (expectSm120Triton) {
            fastllm::Data tritonIndex = MakeIntTensor(
                {2, topk}, std::vector<int32_t>(2 * topk, -1));
            tritonIndex.ToDevice(fastllm::DataDevice::CUDA);
            fastllm::Data tritonScore = MakeCudaTensor(
                fastllm::DataType::FLOAT32, {2, topk},
                std::vector<float>(2 * topk, -1.0f));
            Expect(fastllm::
                       FastllmCudaTryTritonDeepSeekV4SqrtSoftplusRouter(
                           nonFiniteLogits, gateBias, routeScale,
                           tritonIndex, tritonScore),
                   "SM120 DeepSeek-V4 router rejected non-finite input");
            FastllmCudaSyncCurrentThreadStream();
            for (int expert : ToIntVector(tritonIndex)) {
                Expect(expert >= 0 && expert < experts,
                       "SM120 DeepSeek-V4 router returned an invalid expert");
            }
            for (float score : ToFloatVector(tritonScore)) {
                Expect(std::isfinite(score) && score == 0.0f,
                       "SM120 DeepSeek-V4 router returned a non-finite score");
            }
        }

        std::cout << "DeepSeek-V4 fused sqrt-softplus router regression: PASS ("
                  << (expectSm120Triton ? "generic + SM120 Triton" :
                      expectSm120Production ? "generic + SM120 native" :
                                              "generic fallback")
                  << ")\n";
    }

    void RunCudaDeepSeekV4FusedHcPreNormRegression() {
        FastllmCudaSetDevice(0);
        constexpr int hcMult = 4;
        constexpr int hidden = 4096;
        constexpr int mixHc = (2 + hcMult) * hcMult;
        constexpr int flat = hcMult * hidden;
        constexpr int tokens = 1;
        constexpr int sinkhornIters = 20;
        constexpr float eps = 1e-6f;

        fastllm::Data input = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, tokens, hcMult, hidden},
            MakeRegressionValues(tokens * flat, 0.23f, 0.06f));
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
            fastllm::DataType::BFLOAT16, {1, tokens, hidden},
            std::vector<float>(tokens * hidden, 0.0f));
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
            fastllm::DataType::BFLOAT16, {1, tokens, hidden},
            MakeRegressionValues(tokens * hidden, 0.91f, 0.09f));
        fastllm::Data previousPost = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, tokens, hcMult},
            MakeRegressionValues(tokens * hcMult, 0.62f, 0.08f));
        fastllm::Data previousComb = MakeCudaTensor(
            fastllm::DataType::FLOAT32,
            {1, tokens, hcMult, hcMult},
            MakeRegressionValues(tokens * hcMult * hcMult, 0.70f, 0.05f));
        fastllm::Data referenceResidual;
        Expect(FastllmCudaDeepSeekV4HcPostCudaMix(
                   layerOutput, input, previousPost, previousComb,
                   1, tokens, hcMult, hidden, referenceResidual),
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

        // DSpark-7 verifies eight target rows in one graph replay.  Exercise
        // that production shape against the established operator-composed
        // path; every row must preserve the same BF16 tensor boundaries.
        constexpr int dsparkRows = 8;
        fastllm::Data dsparkInput = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {1, dsparkRows, hcMult, hidden},
            MakeRegressionValues(dsparkRows * flat, 0.29f, 0.055f));
        fastllm::Data dsparkPre, dsparkReferencePost, dsparkReferenceComb;
        Expect(FastllmCudaDeepSeekV4HcPre(
                   dsparkInput, hcFn, hcScale, hcBase, hcMult,
                   sinkhornIters, eps, eps, dsparkPre,
                   dsparkReferencePost, dsparkReferenceComb),
               "built-in DeepSeek-V4 HcPre rejected DSpark rows");
        fastllm::Data dsparkReferenceNorm = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, dsparkRows, hidden},
            std::vector<float>(dsparkRows * hidden, 0.0f));
        Expect(FastllmCudaRMSNorm(
                   dsparkPre, normWeight, dsparkReferenceNorm, eps),
               "built-in RMSNorm rejected DSpark HcPre rows");

        fastllm::Data dsparkNorm, dsparkPost, dsparkComb;
        Expect(FastllmCudaDeepSeekV4HcPreNorm(
                   dsparkInput, hcFn, hcScale, hcBase, normWeight,
                   hcMult, sinkhornIters, eps, eps, dsparkNorm,
                   dsparkPost, dsparkComb),
               "fused DeepSeek-V4 HcPreNorm rejected DSpark rows");
        ExpectFloatNear(ToFloatVector(dsparkReferenceNorm),
                        ToFloatVector(dsparkNorm), 0.0f, 0.0f,
                        "DeepSeek-V4 DSpark fused HcPreNorm output");
        ExpectFloatNear(ToFloatVector(dsparkReferencePost),
                        ToFloatVector(dsparkPost), 0.0f, 0.0f,
                        "DeepSeek-V4 DSpark fused HcPreNorm post mix");
        ExpectFloatNear(ToFloatVector(dsparkReferenceComb),
                        ToFloatVector(dsparkComb), 0.0f, 0.0f,
                        "DeepSeek-V4 DSpark fused HcPreNorm comb mix");

        fastllm::Data dsparkLayerOutput = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, dsparkRows, hidden},
            MakeRegressionValues(dsparkRows * hidden, 0.87f, 0.085f));
        fastllm::Data dsparkPreviousPost = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, dsparkRows, hcMult},
            MakeRegressionValues(dsparkRows * hcMult, 0.59f, 0.075f));
        fastllm::Data dsparkPreviousComb = MakeCudaTensor(
            fastllm::DataType::FLOAT32,
            {1, dsparkRows, hcMult, hcMult},
            MakeRegressionValues(dsparkRows * hcMult * hcMult,
                                 0.73f, 0.045f));
        fastllm::Data dsparkReferenceResidual;
        Expect(FastllmCudaDeepSeekV4HcPostCudaMix(
                   dsparkLayerOutput, dsparkInput, dsparkPreviousPost,
                   dsparkPreviousComb, 1, dsparkRows, hcMult, hidden,
                   dsparkReferenceResidual),
               "built-in DeepSeek-V4 HcPost rejected DSpark rows");
        fastllm::Data dsparkTransitionPre;
        fastllm::Data dsparkTransitionReferencePost;
        fastllm::Data dsparkTransitionReferenceComb;
        Expect(FastllmCudaDeepSeekV4HcPre(
                   dsparkReferenceResidual, hcFn, hcScale, hcBase,
                   hcMult, sinkhornIters, eps, eps, dsparkTransitionPre,
                   dsparkTransitionReferencePost,
                   dsparkTransitionReferenceComb),
               "built-in DeepSeek-V4 HcPre rejected DSpark transition");
        fastllm::Data dsparkTransitionReferenceNorm = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {1, dsparkRows, hidden},
            std::vector<float>(dsparkRows * hidden, 0.0f));
        Expect(FastllmCudaRMSNorm(
                   dsparkTransitionPre, normWeight,
                   dsparkTransitionReferenceNorm, eps),
               "built-in RMSNorm rejected DSpark transition");

        fastllm::Data dsparkTransitionResidual;
        fastllm::Data dsparkTransitionNorm;
        fastllm::Data dsparkTransitionPost;
        fastllm::Data dsparkTransitionComb;
        Expect(FastllmCudaDeepSeekV4HcPostPreNorm(
                   dsparkLayerOutput, dsparkInput, dsparkPreviousPost,
                   dsparkPreviousComb, hcFn, hcScale, hcBase, normWeight,
                   hcMult, sinkhornIters, eps, eps,
                   dsparkTransitionResidual, dsparkTransitionNorm,
                   dsparkTransitionPost, dsparkTransitionComb),
               "fused DeepSeek-V4 HcPostPreNorm rejected DSpark rows");
        // SM120 automatically selects vLLM's float-transition, split-K4
        // schedule.  Older architectures retain the bit-exact generic path.
        const bool sm120 = FastllmCudaRuntimeArch() >= 120;
        ExpectFloatNear(ToFloatVector(dsparkReferenceResidual),
                        ToFloatVector(dsparkTransitionResidual),
                        sm120 ? 4e-3f : 0.0f, sm120 ? 2e-3f : 0.0f,
                        "DeepSeek-V4 DSpark fused HcPostPreNorm residual");
        ExpectFloatNear(ToFloatVector(dsparkTransitionReferenceNorm),
                        ToFloatVector(dsparkTransitionNorm),
                        sm120 ? 3e-2f : 0.0f, sm120 ? 4e-3f : 0.0f,
                        "DeepSeek-V4 DSpark fused HcPostPreNorm norm output");
        ExpectFloatNear(ToFloatVector(dsparkTransitionReferencePost),
                        ToFloatVector(dsparkTransitionPost),
                        sm120 ? 2e-3f : 0.0f, sm120 ? 2e-3f : 0.0f,
                        "DeepSeek-V4 DSpark fused HcPostPreNorm post mix");
        ExpectFloatNear(ToFloatVector(dsparkTransitionReferenceComb),
                        ToFloatVector(dsparkTransitionComb),
                        sm120 ? 2e-3f : 0.0f, sm120 ? 2e-3f : 0.0f,
                        "DeepSeek-V4 DSpark fused HcPostPreNorm comb mix");

        std::cout << "DeepSeek-V4 fused HcPreNorm regression: PASS\n";
    }

    void RunMultiCudaDeepSeekV4HcPrePrefillRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "DeepSeek-V4 multi-CUDA HcPre prefill regression: "
                         "SKIP (two GPUs required)\n";
            return;
        }

        constexpr int batch = 1;
        // Prefix-cache chunking feeds a full 256-token block through this
        // path before the short residual suffix.  Exercise that production
        // shape instead of only covering a small prefill.
        constexpr int seqlen = 256;
        constexpr int hcMult = 4;
        constexpr int hidden = 4096;
        constexpr int mixHc = (2 + hcMult) * hcMult;
        constexpr int flat = hcMult * hidden;
        constexpr int sinkhornIters = 20;
        constexpr float eps = 1e-6f;
        FastllmCudaSetDevice(0);
        fastllm::Data warmupInput = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {batch, 1, 1, hidden},
            MakeRegressionValues(batch * hidden, 0.17f, 0.019f));
        fastllm::Data requestInput = MakeCudaTensor(
            fastllm::DataType::BFLOAT16,
            {batch, seqlen, 1, hidden},
            MakeRegressionValues(batch * seqlen * hidden, 0.19f, 0.021f));
        fastllm::Data hcFn = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {mixHc, flat},
            MakeRegressionValues(mixHc * flat, 0.53f, 0.006f));
        fastllm::Data hcScale = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {3}, {0.71f, 0.83f, 0.57f});
        fastllm::Data hcBase = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {mixHc},
            MakeRegressionValues(mixHc, 1.07f, 0.09f));

        fastllm::Data repeated, output, post, comb;
        {
            ScopedFirstDevice device("multicuda:0,1");
            fastllm::Repeat(warmupInput, 2, hcMult, repeated);
            fastllm::DeepSeekV4HcPre(
                repeated, hcFn, hcScale, hcBase, hcMult, sinkhornIters,
                eps, eps, output, post, comb);

            // Reuse the exact workspaces created by a one-token model warmup,
            // then grow them for the first multi-token server request.  This
            // covers stale replicated metadata around the generic CUDA Repeat
            // fallback immediately before the dedicated MultiCuda HcPre op.
            FastllmCudaSetDevice(1);
            fastllm::Repeat(requestInput, 2, hcMult, repeated);
            fastllm::DeepSeekV4HcPre(
                repeated, hcFn, hcScale, hcBase, hcMult, sinkhornIters,
                eps, eps, output, post, comb);
        }
        const std::vector<fastllm::Data*> outputs = {&output, &post, &comb};
        const std::vector<std::vector<int>> expectedDims = {
            {batch, seqlen, hidden},
            {batch, seqlen, hcMult},
            {batch, seqlen, hcMult, hcMult}
        };
        for (int tensor = 0; tensor < (int)outputs.size(); tensor++) {
            fastllm::Data &value = *outputs[tensor];
            Expect(value.IsTensorParallelReplicated() && value.multiDeviceData &&
                       value.dims == expectedDims[tensor],
                   "DeepSeek-V4 HcPre prefill output " + std::to_string(tensor) +
                       " lost its replicated layout: layout=" +
                       std::to_string((int)value.tpLayout) + ", multi=" +
                       std::to_string((int)value.multiDeviceData) +
                       ", rank=" + std::to_string(value.dims.size()));
            std::vector<std::vector<float>> localValues;
            for (int device : {0, 1}) {
                auto it = value.multiDeviceDatas.find(device);
                Expect(it != value.multiDeviceDatas.end() && it->second != nullptr &&
                           it->second->dims == expectedDims[tensor] &&
                           GetPointerDeviceId(it->second->cudaData) == device,
                       "DeepSeek-V4 HcPre prefill output is missing a local replica");
                FastllmCudaSetDevice(device);
                localValues.push_back(ToFloatVector(*it->second));
            }
            ExpectFloatNear(localValues[0], localValues[1], 0.0f, 0.0f,
                            "DeepSeek-V4 multi-CUDA HcPre prefill output");
        }

        // Request-final prefix snapshots split metadata-only replicated cache
        // tensors. Reproduce a stale layout with one missing rank: recovery
        // must preserve a healthy local payload before rebuilding both replicas,
        // rather than attempting to copy from the null root descriptor.
        Expect(output.cudaData == nullptr && output.cpuData == nullptr,
               "DeepSeek-V4 replicated HcPre output unexpectedly owns root storage");
        std::vector<float> fullOutput;
        {
            auto source = output.multiDeviceDatas.find(0);
            Expect(source != output.multiDeviceDatas.end() && source->second != nullptr,
                   "DeepSeek-V4 replicated split recovery has no source rank");
            fullOutput = ToFloatVector(*source->second);
        }
        auto missing = output.multiDeviceDatas.find(1);
        Expect(missing != output.multiDeviceDatas.end() && missing->second != nullptr,
               "DeepSeek-V4 replicated split recovery cannot remove rank 1");
        delete missing->second;
        output.multiDeviceDatas.erase(missing);

        fastllm::Data tail;
        {
            ScopedFirstDevice device("multicuda:0,1");
            fastllm::Split(output, 1, seqlen - 2, seqlen, tail);
        }
        Expect(output.IsTensorParallelReplicated() && output.multiDeviceData &&
                   output.multiDeviceDatas.size() == 2,
               "DeepSeek-V4 replicated split did not rebuild the missing rank");
        for (int device : {0, 1}) {
            auto it = output.multiDeviceDatas.find(device);
            Expect(it != output.multiDeviceDatas.end() && it->second != nullptr &&
                       GetPointerDeviceId(it->second->cudaData) == device,
                   "DeepSeek-V4 replicated split rebuilt a rank on the wrong GPU");
        }
        std::vector<float> expectedTail(
            fullOutput.end() - 2 * hidden, fullOutput.end());
        Expect(tail.dims == std::vector<int>({batch, 2, hidden}),
               "DeepSeek-V4 replicated split returned the wrong tail shape");
        ExpectFloatNear(expectedTail, ToFloatVector(tail), 0.0f, 0.0f,
                        "DeepSeek-V4 replicated prefix-cache tail split");
        FastllmCudaSetDevice(0);
        std::cout << "DeepSeek-V4 multi-CUDA HcPre prefill regression: PASS\n";
    }

    void RunMultiCudaDeepSeekV4RouterLinearResizeRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "DeepSeek-V4 multi-CUDA router linear resize regression: "
                         "SKIP (two GPUs required)\n";
            return;
        }

        constexpr int hidden = 4096;
        constexpr int experts = 256;
        const int originalDevice = FastllmCudaGetDevice();
        FastllmCudaSetDevice(0);
        fastllm::Data weight = MakeCudaTensor(
            fastllm::DataType::BFLOAT16, {experts, hidden},
            MakeRegressionValues(experts * hidden, 0.37f, 0.003f));
        weight.name = "layers.0.ffn.gate.weight";
        fastllm::Data gateBias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {experts},
            MakeRegressionValues(experts, 8.7f, 0.001f));

        for (int tokens : {121, 137, 1024}) {
            fastllm::Data input = MakeCudaTensor(
                fastllm::DataType::FLOAT32, {tokens, hidden},
                MakeRegressionValues(tokens * hidden,
                                     0.61f + tokens * 0.001f, 0.005f));
            PrepareMultiCudaReplicatedData(input, {0, 1}, true);
            fastllm::Data output;
            {
                ScopedFirstDevice device("multicuda:0,1");
                fastllm::Linear(input, weight, fastllm::Data(), output, true);
            }
            Expect(output.multiDeviceData &&
                       output.IsTensorParallelReplicated() &&
                       output.dims == std::vector<int>({tokens, experts}),
                   "DeepSeek-V4 router linear did not return replicated logits");
            std::vector<std::vector<float>> localValues;
            for (int device : {0, 1}) {
                auto it = output.multiDeviceDatas.find(device);
                Expect(it != output.multiDeviceDatas.end() &&
                           it->second != nullptr &&
                           it->second->dims == output.dims &&
                           it->second->expansionSize >= it->second->Count(0) &&
                           GetPointerDeviceId(it->second->cudaData) == device,
                       "DeepSeek-V4 router linear has an invalid local output");
                FastllmCudaSetDevice(device);
                ForceDeviceSync();
                localValues.push_back(ToFloatVector(*it->second));
            }
            ExpectFloatNear(localValues[0], localValues[1], 2e-6f, 2e-6f,
                            "DeepSeek-V4 replicated router logits");

            std::vector<int> devices = {0, 1};
            std::vector<char> transformed(devices.size(), 0);
            Expect(fastllm::MultiCudaRunDeviceCallbacks(
                       devices, [&](int rank, int device) {
                           transformed[rank] =
                               FastllmCudaDeepSeekV4RouteScoreTransform(
                                   *output.multiDeviceDatas.at(device), 2);
                       }),
                   "DeepSeek-V4 router score transform did not dispatch on both GPUs");
            Expect(std::all_of(transformed.begin(), transformed.end(),
                               [](char state) { return state != 0; }),
                   "DeepSeek-V4 router score transform rejected a local replica");

            constexpr int topk = 6;
            constexpr float routeScale = 1.5f;
            fastllm::Data expertIndex, expertScore;
            {
                ScopedFirstDevice device("multicuda:0,1");
                fastllm::SelectExpert(output, expertIndex, expertScore,
                                      topk, true, routeScale, &gateBias);
            }
            Expect(expertIndex.multiDeviceData && expertScore.multiDeviceData &&
                       expertIndex.IsTensorParallelReplicated() &&
                       expertScore.IsTensorParallelReplicated(),
                   "DeepSeek-V4 topk=6 routing outputs are not replicated");

            std::vector<std::vector<int32_t>> localIndices;
            std::vector<std::vector<float>> localScores;
            for (int device : devices) {
                auto indexIt = expertIndex.multiDeviceDatas.find(device);
                auto scoreIt = expertScore.multiDeviceDatas.find(device);
                Expect(indexIt != expertIndex.multiDeviceDatas.end() &&
                           indexIt->second != nullptr &&
                           scoreIt != expertScore.multiDeviceDatas.end() &&
                           scoreIt->second != nullptr &&
                           indexIt->second->dims ==
                               std::vector<int>({tokens, topk}) &&
                           scoreIt->second->dims ==
                               std::vector<int>({tokens, topk}) &&
                           indexIt->second->expansionSize >=
                               indexIt->second->Count(0) &&
                           scoreIt->second->expansionSize >=
                               scoreIt->second->Count(0),
                       "DeepSeek-V4 topk=6 routing output has invalid local storage");
                FastllmCudaSetDevice(device);
                ForceDeviceSync();
                localIndices.push_back(ToIntVector(*indexIt->second));
                localScores.push_back(ToFloatVector(*scoreIt->second));
                for (int token = 0; token < tokens; token++) {
                    float sum = 0.0f;
                    for (int rank = 0; rank < topk; rank++) {
                        int expert = localIndices.back()[token * topk + rank];
                        float score = localScores.back()[token * topk + rank];
                        Expect(expert >= 0 && expert < experts &&
                                   std::isfinite(score),
                               "DeepSeek-V4 topk=6 routing returned an invalid expert");
                        sum += score;
                    }
                    Expect(std::fabs(sum - routeScale) < 2e-5f,
                           "DeepSeek-V4 normalized route scores have the wrong sum");
                }
            }
            Expect(localIndices[0] == localIndices[1],
                   "DeepSeek-V4 topk=6 expert ids differ between TP ranks");
            ExpectFloatNear(localScores[0], localScores[1], 2e-6f, 2e-6f,
                            "DeepSeek-V4 replicated topk=6 route scores");
        }

        // Restored long-context state can expose a numerical failure in an
        // upstream attention row.  SelectExpert must never turn NaN logits into
        // inputData[-1] and an illegal CUDA access; it should return bounded ids
        // and finite zero scores so the request can fail safely downstream.
        std::vector<float> nonFiniteValues(2 * experts,
                                           std::numeric_limits<float>::quiet_NaN());
        fastllm::Data nonFiniteLogits = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {2, experts}, nonFiniteValues);
        PrepareMultiCudaReplicatedData(nonFiniteLogits, {0, 1}, true);
        fastllm::Data nonFiniteIndex, nonFiniteScore;
        {
            ScopedFirstDevice device("multicuda:0,1");
            fastllm::SelectExpert(nonFiniteLogits, nonFiniteIndex,
                                  nonFiniteScore, 6, true, 1.5f, &gateBias);
        }
        for (int device : {0, 1}) {
            FastllmCudaSetDevice(device);
            ForceDeviceSync();
            std::vector<int32_t> indices =
                ToIntVector(*nonFiniteIndex.multiDeviceDatas.at(device));
            std::vector<float> scores =
                ToFloatVector(*nonFiniteScore.multiDeviceDatas.at(device));
            for (int expert : indices) {
                Expect(expert >= 0 && expert < experts,
                       "DeepSeek-V4 non-finite route returned an invalid expert id");
            }
            for (float score : scores) {
                Expect(std::isfinite(score) && score == 0.0f,
                       "DeepSeek-V4 non-finite route returned an invalid score");
            }
        }

        FastllmCudaSetDevice(originalDevice);
        std::cout << "DeepSeek-V4 multi-CUDA router linear resize regression: PASS\n";
    }

    void RunMultiCudaDeepSeekV4RawCacheAppendRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "DeepSeek-V4 restored raw-cache append regression: "
                         "SKIP (two GPUs required)\n";
            return;
        }

        constexpr int bsz = 1;
        constexpr int oldLen = 4;
        constexpr int suffixLen = 156;
        constexpr int totalLen = oldLen + suffixLen;
        constexpr int wideDim = 1024;
        const std::vector<float> oldKV = MakeRegressionValues(
            (uint64_t)oldLen * wideDim, 0.37f, 0.17f);
        const std::vector<float> oldScore = MakeRegressionValues(
            (uint64_t)oldLen * wideDim, 0.61f, 0.13f);
        const std::vector<float> suffixKV = MakeRegressionValues(
            (uint64_t)suffixLen * wideDim, 0.83f, 0.11f);
        const std::vector<float> suffixScore = MakeRegressionValues(
            (uint64_t)suffixLen * wideDim, 1.07f, 0.09f);

        // History keeps one expanded CPU payload: four live ratio-4 tail rows
        // inside a 128-row allocation.  The resumed projections are replicated
        // CUDA tensors and push the destination past that original capacity.
        fastllm::Data allKV(fastllm::DataType::FLOAT32,
                            {bsz, oldLen, wideDim}, oldKV);
        fastllm::Data allScore(fastllm::DataType::FLOAT32,
                               {bsz, oldLen, wideDim}, oldScore);
        allKV.Expansion({bsz, 128, wideDim});
        allScore.Expansion({bsz, 128, wideDim});
        allKV.lockInCPU = true;
        allScore.lockInCPU = true;

        fastllm::Data newKV(fastllm::DataType::FLOAT32,
                            {bsz, suffixLen, wideDim}, suffixKV);
        fastllm::Data newScore(fastllm::DataType::FLOAT32,
                               {bsz, suffixLen, wideDim}, suffixScore);
        FastllmCudaSetDevice(0);
        newKV.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
        newScore.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
        PrepareMultiCudaReplicatedData(newKV, {0, 1}, true);
        PrepareMultiCudaReplicatedData(newScore, {0, 1}, true);
        {
            ScopedFirstDevice device("multicuda:0,1");
            fastllm::DeepSeekV4AppendCompressorRawForTest(
                newKV, newScore, bsz, suffixLen, wideDim, allKV, allScore);
        }

        Expect(allKV.multiDeviceData && allScore.multiDeviceData &&
                   allKV.IsTensorParallelReplicated() &&
                   allScore.IsTensorParallelReplicated() &&
                   allKV.dims == std::vector<int>({bsz, totalLen, wideDim}) &&
                   allScore.dims == allKV.dims &&
                   allKV.expansionDims ==
                       std::vector<int>({bsz, 256, wideDim}) &&
                   allScore.expansionDims == allKV.expansionDims &&
                   allKV.cpuData == nullptr && allKV.cudaData == nullptr &&
                   allScore.cpuData == nullptr && allScore.cudaData == nullptr,
               "DeepSeek-V4 restored raw cache has stale root storage or layout");

        std::vector<float> expectedKV = oldKV;
        expectedKV.insert(expectedKV.end(), suffixKV.begin(), suffixKV.end());
        std::vector<float> expectedScore = oldScore;
        expectedScore.insert(expectedScore.end(), suffixScore.begin(), suffixScore.end());
        for (int device : {0, 1}) {
            FastllmCudaSetDevice(device);
            ForceDeviceSync();
            fastllm::Data *localKV = allKV.multiDeviceDatas.at(device);
            fastllm::Data *localScore = allScore.multiDeviceDatas.at(device);
            Expect(localKV->dims == allKV.dims &&
                       localKV->strides == allKV.strides &&
                       localKV->expansionDims == allKV.expansionDims &&
                       localKV->expansionSize == allKV.expansionSize &&
                       localKV->expansionBytes == allKV.expansionBytes &&
                       localScore->dims == allScore.dims &&
                       localScore->strides == allScore.strides &&
                       localScore->expansionDims == allScore.expansionDims &&
                       localScore->expansionSize == allScore.expansionSize &&
                       localScore->expansionBytes == allScore.expansionBytes,
                   "DeepSeek-V4 restored raw-cache root/local metadata mismatch");
            std::vector<float> actualKV = ToFloatVector(*localKV);
            std::vector<float> actualScore = ToFloatVector(*localScore);
            actualKV.resize(expectedKV.size());
            actualScore.resize(expectedScore.size());
            ExpectFloatNear(expectedKV, actualKV, 0.0f, 0.0f,
                            "DeepSeek-V4 restored raw KV append");
            ExpectFloatNear(expectedScore, actualScore, 0.0f, 0.0f,
                            "DeepSeek-V4 restored raw score append");
        }
        FastllmCudaSetDevice(0);
        std::cout << "DeepSeek-V4 restored raw-cache append regression: PASS\n";
    }

    void RunCudaDeepSeekV4CompressedKvMaintenanceRegression() {
        auto toLogicalBFloat16Bits = [](fastllm::Data data) {
            const int count = std::accumulate(
                data.dims.begin(), data.dims.end(), 1,
                std::multiplies<int>());
            Expect(data.dataType == fastllm::DataType::BFLOAT16,
                   "Expected a BFLOAT16 tensor.");
            data.ToDevice(fastllm::DataDevice::CPU);
            std::vector<uint16_t> values(count);
            if (count > 0) {
                Expect(data.cpuData != nullptr,
                       "BFLOAT16 tensor has no CPU buffer.");
                std::memcpy(values.data(), data.cpuData,
                            (size_t) count * sizeof(uint16_t));
            }
            return values;
        };
        auto toLogicalFloatVector = [](fastllm::Data data) {
            const int count = std::accumulate(
                data.dims.begin(), data.dims.end(), 1,
                std::multiplies<int>());
            Expect(data.dataType == fastllm::DataType::FLOAT32,
                   "Expected a FLOAT32 tensor.");
            data.ToDevice(fastllm::DataDevice::CPU);
            std::vector<float> values(count);
            if (count > 0) {
                Expect(data.cpuData != nullptr,
                       "FLOAT32 tensor has no CPU buffer.");
                std::memcpy(values.data(), data.cpuData,
                            (size_t) count * sizeof(float));
            }
            return values;
        };
        const char *oldDisable = std::getenv(
            "FASTLLM_DSV4_DISABLE_FUSED_COMPRESSED_KV");
        const bool hadOldDisable = oldDisable != nullptr;
        const std::string oldDisableValue =
            oldDisable == nullptr ? std::string() : std::string(oldDisable);

        auto runFinalizeCase = [&](int headDim, int ropeDim, bool indexer) {
            constexpr int bsz = 1;
            constexpr int compressRatio = 4;
            constexpr int blockStart = 2;
            constexpr int blockCount = 2;
            constexpr int rawTokenBase = 4;
            constexpr int rawLen = 24;
            const int wideDim = 2 * headDim;

            std::vector<float> kvValues = MakeRegressionValues(
                (uint64_t)bsz * rawLen * wideDim, 0.43f, 0.013f);
            std::vector<float> scoreValues = MakeRegressionValues(
                (uint64_t)bsz * rawLen * wideDim, 0.19f, 0.007f);
            std::vector<float> apeValues = MakeRegressionValues(
                (uint64_t)compressRatio * wideDim, 0.11f, 0.009f);
            std::vector<float> normValues = MakeRegressionValues(
                headDim, 0.03f, 0.001f);
            for (float &value : normValues) {
                value += 0.75f;
            }
            std::vector<float> cacheValues = MakeRegressionValues(
                (uint64_t)bsz * blockStart * headDim, 0.71f, 0.005f);

            fastllm::Data kv(fastllm::DataType::FLOAT32,
                              {bsz, rawLen, wideDim}, kvValues);
            fastllm::Data score(fastllm::DataType::FLOAT32,
                                 {bsz, rawLen, wideDim}, scoreValues);
            fastllm::Data ape(fastllm::DataType::FLOAT32,
                               {compressRatio, wideDim}, apeValues);
            fastllm::Data norm(fastllm::DataType::FLOAT32,
                                {headDim}, normValues);
            fastllm::Data referenceCache(fastllm::DataType::BFLOAT16,
                                          {bsz, blockStart, headDim},
                                          cacheValues);
            fastllm::Data fusedCache(fastllm::DataType::BFLOAT16,
                                      {bsz, blockStart, headDim},
                                      cacheValues);
            FastllmCudaSetDevice(0);
            for (fastllm::Data *data :
                 {&kv, &score, &ape, &norm, &referenceCache, &fusedCache}) {
                data->ToDevice(fastllm::DataDevice::CUDA, {0}, true);
            }

            setenv("FASTLLM_DSV4_DISABLE_FUSED_COMPRESSED_KV", "1", 1);
            {
                ScopedFirstDevice device("cuda:0");
                fastllm::DeepSeekV4BuildCompressedKVFromRaw(
                    kv, score, ape, norm, rawTokenBase, rawLen,
                    blockStart, blockCount, compressRatio, headDim,
                    ropeDim, 10000.0f, 1.0f, 32, 1, 4096,
                    true, true, referenceCache, indexer);
            }
            unsetenv("FASTLLM_DSV4_DISABLE_FUSED_COMPRESSED_KV");
            {
                ScopedFirstDevice device("cuda:0");
                fastllm::DeepSeekV4BuildCompressedKVFromRaw(
                    kv, score, ape, norm, rawTokenBase, rawLen,
                    blockStart, blockCount, compressRatio, headDim,
                    ropeDim, 10000.0f, 1.0f, 32, 1, 4096,
                    true, true, fusedCache, indexer);
            }
            ForceDeviceSync();

            Expect(referenceCache.dims == fusedCache.dims &&
                       fusedCache.dims == std::vector<int>(
                           {bsz, blockStart + blockCount, headDim}),
                   "DeepSeek-V4 fused compressed finalize returned the wrong shape");
            Expect(fusedCache.strides.size() == 3 &&
                       fusedCache.strides[0] ==
                           fusedCache.dims[1] * headDim &&
                       fusedCache.expansionDims.empty() &&
                       fusedCache.expansionSize >=
                           (uint64_t)bsz * 64 * headDim,
                   "DeepSeek-V4 fused compressed finalize exposed reserve rows");
            const std::vector<uint16_t> referenceBits =
                toLogicalBFloat16Bits(referenceCache);
            const std::vector<uint16_t> fusedBits =
                toLogicalBFloat16Bits(fusedCache);
            if (referenceBits != fusedBits) {
                size_t first = 0;
                while (first < referenceBits.size() &&
                       referenceBits[first] == fusedBits[first]) {
                    first++;
                }
                std::cerr << "DeepSeek-V4 finalize mismatch: headDim="
                          << headDim << " first=" << first
                          << " reference="
                          << (first < referenceBits.size()
                                  ? referenceBits[first] : 0)
                          << " fused="
                          << (first < fusedBits.size()
                                  ? fusedBits[first] : 0)
                          << "\n";
            }
            Expect(referenceBits == fusedBits,
                   "DeepSeek-V4 fused compressed finalize changed BF16 bits");

            void *reservedPointer = fusedCache.cudaData;
            setenv("FASTLLM_DSV4_DISABLE_FUSED_COMPRESSED_KV", "1", 1);
            {
                ScopedFirstDevice device("cuda:0");
                fastllm::DeepSeekV4BuildCompressedKVFromRaw(
                    kv, score, ape, norm, rawTokenBase, rawLen,
                    blockStart + blockCount, blockCount, compressRatio,
                    headDim, ropeDim, 10000.0f, 1.0f, 32, 1, 4096,
                    true, true, referenceCache, indexer);
            }
            unsetenv("FASTLLM_DSV4_DISABLE_FUSED_COMPRESSED_KV");
            {
                ScopedFirstDevice device("cuda:0");
                fastllm::DeepSeekV4BuildCompressedKVFromRaw(
                    kv, score, ape, norm, rawTokenBase, rawLen,
                    blockStart + blockCount, blockCount, compressRatio,
                    headDim, ropeDim, 10000.0f, 1.0f, 32, 1, 4096,
                    true, true, fusedCache, indexer);
            }
            ForceDeviceSync();
            Expect(fusedCache.cudaData == reservedPointer &&
                       fusedCache.dims == std::vector<int>(
                           {bsz, blockStart + 2 * blockCount, headDim}) &&
                       fusedCache.strides[0] ==
                           fusedCache.dims[1] * headDim &&
                       fusedCache.expansionDims.empty(),
                   "DeepSeek-V4 fused compressed append did not reuse its hidden reserve");
            Expect(toLogicalBFloat16Bits(referenceCache) ==
                       toLogicalBFloat16Bits(fusedCache),
                   "DeepSeek-V4 repeated fused append changed BF16 bits");
        };

        runFinalizeCase(512, 64, false);
        runFinalizeCase(128, 64, true);

        constexpr int bsz = 1;
        constexpr int oldLen = 8;
        constexpr int wideDim = 32;
        std::vector<float> rawKvValues = MakeRegressionValues(
            (uint64_t)bsz * oldLen * wideDim, 0.37f, 0.017f);
        std::vector<float> rawScoreValues = MakeRegressionValues(
            (uint64_t)bsz * oldLen * wideDim, 0.61f, 0.011f);
        std::vector<float> expectedKv(
            rawKvValues.begin() + 4 * wideDim, rawKvValues.end());
        std::vector<float> expectedScore(
            rawScoreValues.begin() + 4 * wideDim, rawScoreValues.end());
        fastllm::Data rawKv(fastllm::DataType::FLOAT32,
                             {bsz, oldLen, wideDim}, rawKvValues);
        fastllm::Data rawScore(fastllm::DataType::FLOAT32,
                                {bsz, oldLen, wideDim}, rawScoreValues);
        rawKv.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
        rawScore.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
        rawKv.Expansion({bsz, 128, wideDim});
        rawScore.Expansion({bsz, 128, wideDim});
        void *kvPointer = rawKv.cudaData;
        void *scorePointer = rawScore.cudaData;
        int rawTokenBase = 0;
        fastllm::DeepSeekV4TrimCompressorRawForTest(
            bsz, oldLen, 4, wideDim, 2,
            rawKv, rawScore, rawTokenBase);
        ForceDeviceSync();
        Expect(rawTokenBase == 4 &&
                   rawKv.dims == std::vector<int>({bsz, 4, wideDim}) &&
                   rawScore.dims == rawKv.dims &&
                   rawKv.expansionDims ==
                       std::vector<int>({bsz, 128, wideDim}) &&
                   rawScore.expansionDims == rawKv.expansionDims &&
                   rawKv.cudaData == kvPointer &&
                   rawScore.cudaData == scorePointer,
               "DeepSeek-V4 raw compressor compaction changed its reserve");
        ExpectFloatNear(expectedKv, toLogicalFloatVector(rawKv), 0.0f, 0.0f,
                        "DeepSeek-V4 raw KV in-place compaction");
        ExpectFloatNear(expectedScore, toLogicalFloatVector(rawScore), 0.0f, 0.0f,
                        "DeepSeek-V4 raw score in-place compaction");

        if (hadOldDisable) {
            setenv("FASTLLM_DSV4_DISABLE_FUSED_COMPRESSED_KV",
                   oldDisableValue.c_str(), 1);
        } else {
            unsetenv("FASTLLM_DSV4_DISABLE_FUSED_COMPRESSED_KV");
        }
        std::cout << "DeepSeek-V4 compressed-KV maintenance regression: PASS\n";
    }

    void RunMultiCudaDeepSeekV4CompressedCacheAppendRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "DeepSeek-V4 restored compressed-cache append regression: "
                         "SKIP (two GPUs required)\n";
            return;
        }

        constexpr int bsz = 1;
        constexpr int headDim = 512;
        constexpr int ropeDim = 64;
        constexpr int compressRatio = 4;
        constexpr int wideDim = 2 * headDim;
        constexpr int rawTokenBase = 2044;
        constexpr int rawLen = 160;
        constexpr int blockStart = 512;
        constexpr int blockCount = 39;
        constexpr int totalBlocks = blockStart + blockCount;

        const std::vector<float> kvValues = MakeRegressionValues(
            (uint64_t)bsz * rawLen * wideDim, 0.43f, 0.013f);
        const std::vector<float> scoreValues = MakeRegressionValues(
            (uint64_t)bsz * rawLen * wideDim, 0.19f, 0.007f);
        const std::vector<float> apeValues = MakeRegressionValues(
            (uint64_t)compressRatio * wideDim, 0.11f, 0.009f);
        std::vector<float> normValues(headDim, 1.0f);
        const std::vector<float> cacheValues = MakeRegressionValues(
            (uint64_t)bsz * blockStart * headDim, 0.71f, 0.005f);

        fastllm::Data cpuKV(fastllm::DataType::BFLOAT16,
                            {bsz, rawLen, wideDim}, kvValues);
        fastllm::Data cpuScore(fastllm::DataType::BFLOAT16,
                               {bsz, rawLen, wideDim}, scoreValues);
        fastllm::Data cpuApe(fastllm::DataType::FLOAT32,
                             {compressRatio, wideDim}, apeValues);
        fastllm::Data cpuNorm(fastllm::DataType::BFLOAT16,
                              {headDim}, normValues);
        fastllm::Data cpuCache(fastllm::DataType::BFLOAT16,
                               {bsz, blockStart, headDim}, cacheValues);
        {
            ScopedFirstDevice device("cpu");
            fastllm::DeepSeekV4BuildCompressedKVFromRaw(
                cpuKV, cpuScore, cpuApe, cpuNorm,
                rawTokenBase, rawLen, blockStart, blockCount,
                compressRatio, headDim, ropeDim, 10000.0f,
                1.0f, 32, 1, 4096, true, false, cpuCache);
        }
        Expect(cpuCache.dims ==
                   std::vector<int>({bsz, totalBlocks, headDim}),
               "DeepSeek-V4 CPU compressed-cache append returned the wrong shape");

        fastllm::Data actualKV(fastllm::DataType::BFLOAT16,
                               {bsz, rawLen, wideDim}, kvValues);
        fastllm::Data actualScore(fastllm::DataType::BFLOAT16,
                                  {bsz, rawLen, wideDim}, scoreValues);
        fastllm::Data actualApe(fastllm::DataType::FLOAT32,
                                {compressRatio, wideDim}, apeValues);
        fastllm::Data actualNorm(fastllm::DataType::BFLOAT16,
                                 {headDim}, normValues);
        fastllm::Data residentCache(fastllm::DataType::BFLOAT16,
                                    {bsz, blockStart, headDim}, cacheValues);
        // The cold suffix starts from an already replicated compressed cache.
        // Use it as the CUDA-path reference; the CPU implementation is also
        // exercised above, but CUDA's RMS/quant rounding is intentionally not
        // required to be bit-identical to it.
        PrepareMultiCudaReplicatedData(residentCache, {0, 1}, true);
        {
            ScopedFirstDevice device("multicuda:0,1");
            fastllm::DeepSeekV4BuildCompressedKVFromRaw(
                actualKV, actualScore, actualApe, actualNorm,
                rawTokenBase, rawLen, blockStart, blockCount,
                compressRatio, headDim, ropeDim, 10000.0f,
                1.0f, 32, 1, 4096, true, true, residentCache);
        }
        fastllm::Data actualCache(fastllm::DataType::BFLOAT16,
                                  {bsz, blockStart, headDim}, cacheValues);
        // Match a history snapshot: one CPU cache payload is restored to the
        // root GPU immediately before the MultiCUDA operator recreates both
        // physical replicas and appends the suffix-compressed rows.
        actualCache.lockInCPU = false;
        FastllmCudaSetDevice(0);
        actualCache.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
        {
            ScopedFirstDevice device("multicuda:0,1");
            fastllm::DeepSeekV4BuildCompressedKVFromRaw(
                actualKV, actualScore, actualApe, actualNorm,
                rawTokenBase, rawLen, blockStart, blockCount,
                compressRatio, headDim, ropeDim, 10000.0f,
                1.0f, 32, 1, 4096, true, true, actualCache);
        }
        Expect(actualCache.multiDeviceData &&
                   actualCache.IsTensorParallelReplicated() &&
                   actualCache.dims ==
                       std::vector<int>({bsz, totalBlocks, headDim}),
               "DeepSeek-V4 restored compressed cache was not replicated");
        for (int device : {0, 1}) {
            FastllmCudaSetDevice(device);
            ForceDeviceSync();
            auto it = actualCache.multiDeviceDatas.find(device);
            Expect(it != actualCache.multiDeviceDatas.end() &&
                       it->second != nullptr && it->second->cudaData != nullptr,
                   "DeepSeek-V4 restored compressed cache is missing a TP replica");
            ExpectFloatNear(
                            ToFloatVector(*residentCache.multiDeviceDatas.at(device)),
                            ToFloatVector(*it->second), 0.0f, 0.0f,
                            "DeepSeek-V4 restored compressed-cache append");
        }
        FastllmCudaSetDevice(0);
        std::cout << "DeepSeek-V4 restored compressed-cache append regression: PASS\n";
    }

    void RunMultiCudaDeepSeekV4ExpandedSnapshotRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "DeepSeek-V4 expanded prefix snapshot regression: "
                         "SKIP (two GPUs required)\n";
            return;
        }
        ScopedFirstDevice firstDevice("multicuda:0,1");

        constexpr int logicalTokens = 3;
        constexpr int capacityTokens = 8;
        constexpr int dim = 16;
        fastllm::DeepSeekV4HistoryLayerCache source;
        fastllm::Data &cache = source.windowKV;
        cache.dataType = fastllm::DataType::FLOAT32;
        cache.UpdateUnitSize();
        cache.Resize({1, logicalTokens, dim});
        cache.dataDevice = fastllm::DataDevice::CUDA;
        cache.dataDeviceIds = {0};
        FastllmCudaSetDevice(0);
        cache.Allocate(false);
        std::vector<float> values =
            MakeRegressionValues(logicalTokens * dim, 0.73f, 0.14f);
        fastllm::Data initial = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, logicalTokens, dim}, values);
        FastllmCudaCopyFromDeviceToDevice(
            cache.cudaData, initial.cudaData, initial.GetBytes());
        cache.Expansion({1, capacityTokens, dim});
        PrepareMultiCudaReplicatedData(cache, {0, 1}, true);

        FastllmCudaSetDevice(0);
        fastllm::DeepSeekV4HistoryLayerCache snapshot(source);
        Expect(FastllmCudaGetDevice() == 0,
               "DeepSeek-V4 prefix snapshot leaked its last CUDA device");
        Expect(snapshot.windowKV.dataDevice == fastllm::DataDevice::CPU &&
                   !snapshot.windowKV.multiDeviceData &&
                   snapshot.windowKV.expansionDims.empty() &&
                   snapshot.windowKV.expansionBytes >=
                       snapshot.windowKV.GetBytes() &&
                   snapshot.windowKV.expansionBytes <
                       source.windowKV.multiDeviceDatas.at(0)->expansionBytes,
               "DeepSeek-V4 prefix snapshot retained CUDA graph reserve capacity");
        FastllmCudaSetDevice(0);
        ExpectFloatNear(values, ToFloatVector(snapshot.windowKV),
                        0.0f, 0.0f,
                        "DeepSeek-V4 expanded prefix snapshot payload");

        // A restored CPU snapshot has one physical GPU allocation but retains
        // the logical {0,1} device list.  Launch the direct prefix builder while
        // GPU 1 is current and verify that it follows the source pointer to GPU 0.
        snapshot.windowKV.ToDevice(fastllm::DataDevice::CUDA, {0, 1}, true);
        Expect(GetPointerDeviceId(snapshot.windowKV.cudaData) == 0,
               "DeepSeek-V4 restored prefix snapshot is not resident on GPU 0");
        FastllmCudaSetDevice(1);
        fastllm::Data prefix;
        Expect(FastllmCudaDeepSeekV4BuildWindowKVPrefix(
                   snapshot.windowKV, logicalTokens, logicalTokens, 2, prefix),
               "DeepSeek-V4 prefix builder rejected a restored snapshot");
        ForceDeviceSync();
        Expect(GetPointerDeviceId(prefix.cudaData) == 0,
               "DeepSeek-V4 prefix builder launched on the stale current GPU");
        std::vector<float> expectedPrefix(values.begin() + dim, values.end());
        ExpectFloatNear(expectedPrefix, ToFloatVector(prefix), 0.0f, 0.0f,
                        "DeepSeek-V4 restored snapshot prefix payload");

        // Model inference promotes the restored cache to one physical replica
        // per TP rank before building the prefix.  Exercise that exact path so
        // the model helper cannot accidentally launch through the metadata-only
        // root tensor or return a single-rank prefix.
        fastllm::DeepSeekV4HistoryLayerCache replicatedSnapshot(source);
        replicatedSnapshot.windowKV.lockInCPU = false;
        replicatedSnapshot.windowKV.ToDevice(
            fastllm::DataDevice::CUDA, {0}, true);
        PrepareMultiCudaReplicatedData(
            replicatedSnapshot.windowKV, {0, 1}, true);
        fastllm::Data replicatedPrefix;
        Expect(fastllm::DeepSeekV4BuildWindowKVPrefixForTest(
                   replicatedSnapshot.windowKV, 1, dim, logicalTokens,
                   logicalTokens, replicatedPrefix) == logicalTokens,
               "DeepSeek-V4 replicated prefix builder rejected a restored snapshot");
        Expect(replicatedPrefix.multiDeviceData &&
                   replicatedPrefix.IsTensorParallelReplicated(),
               "DeepSeek-V4 replicated prefix builder returned a single-rank tensor");
        for (int device : {0, 1}) {
            auto it = replicatedPrefix.multiDeviceDatas.find(device);
            Expect(it != replicatedPrefix.multiDeviceDatas.end() &&
                       it->second != nullptr && it->second->cudaData != nullptr &&
                       GetPointerDeviceId(it->second->cudaData) == device,
                   "DeepSeek-V4 replicated prefix builder used the wrong GPU");
            ExpectFloatNear(values, ToFloatVector(*it->second),
                            0.0f, 0.0f,
                            "DeepSeek-V4 replicated restored prefix payload");
        }

        // Match the full model's cache-hit geometry: a 128-token float window,
        // 512-wide KV head, followed by a 36-token suffix update on both ranks.
        constexpr int modelWindow = 128;
        constexpr int modelHeadDim = 512;
        constexpr int suffixTokens = 137;
        std::vector<float> modelWindowValues = MakeRegressionValues(
            modelWindow * modelHeadDim, 0.41f, 0.07f);
        fastllm::DeepSeekV4HistoryLayerCache modelSource;
        fastllm::Data modelWindowInitial = MakeCudaTensor(
            fastllm::DataType::FLOAT32,
            {1, modelWindow, modelHeadDim}, modelWindowValues);
        modelSource.windowKV.dataType = fastllm::DataType::FLOAT32;
        modelSource.windowKV.UpdateUnitSize();
        modelSource.windowKV.Resize({1, modelWindow, modelHeadDim});
        modelSource.windowKV.dataDevice = fastllm::DataDevice::CUDA;
        modelSource.windowKV.dataDeviceIds = {0};
        FastllmCudaSetDevice(0);
        modelSource.windowKV.Allocate(false);
        FastllmCudaCopyFromDeviceToDevice(
            modelSource.windowKV.cudaData, modelWindowInitial.cudaData,
            modelWindowInitial.GetBytes());
        PrepareMultiCudaReplicatedData(modelSource.windowKV, {0, 1}, true);
        fastllm::DeepSeekV4HistoryLayerCache modelSnapshot(modelSource);
        modelSnapshot.windowKV.lockInCPU = false;
        modelSnapshot.windowKV.ToDevice(
            fastllm::DataDevice::CUDA, {0}, true);
        PrepareMultiCudaReplicatedData(
            modelSnapshot.windowKV, {0, 1}, true);

        fastllm::Data modelPrefix;
        Expect(fastllm::DeepSeekV4BuildWindowKVPrefixForTest(
                   modelSnapshot.windowKV, 1, modelHeadDim, 256,
                   modelWindow, modelPrefix) == modelWindow,
               "DeepSeek-V4 model-shape prefix builder rejected the restored window");
        std::vector<float> suffixValues = MakeRegressionValues(
            suffixTokens * modelHeadDim, 0.83f, 0.11f);
        fastllm::Data suffix = MakeCudaTensor(
            fastllm::DataType::FLOAT32,
            {1, suffixTokens, modelHeadDim}, suffixValues);
        PrepareMultiCudaReplicatedData(suffix, {0, 1}, true);
        fastllm::DeepSeekV4UpdateWindowKVCacheForTest(
            suffix, 1, modelHeadDim, 256, modelWindow,
            modelSnapshot.windowKV);

        std::vector<float> expectedWindow = modelWindowValues;
        for (int s = 0; s < suffixTokens; ++s) {
            int slot = (256 + s) % modelWindow;
            std::copy(suffixValues.begin() + (uint64_t)s * modelHeadDim,
                      suffixValues.begin() + (uint64_t)(s + 1) * modelHeadDim,
                      expectedWindow.begin() + (uint64_t)slot * modelHeadDim);
        }
        for (int device : {0, 1}) {
            ExpectFloatNear(
                modelWindowValues,
                ToFloatVector(*modelPrefix.multiDeviceDatas.at(device)),
                0.0f, 0.0f,
                "DeepSeek-V4 model-shape replicated prefix payload");
            ExpectFloatNear(
                expectedWindow,
                ToFloatVector(*modelSnapshot.windowKV.multiDeviceDatas.at(device)),
                0.0f, 0.0f,
                "DeepSeek-V4 model-shape replicated window update");
        }
        FastllmCudaSetDevice(0);
        std::cout << "DeepSeek-V4 expanded prefix snapshot regression: PASS\n";
    }

    void RunCudaDeepSeekV4FusedQKVRopeCacheRegression() {
        FastllmCudaSetDevice(0);
        constexpr int heads = 64;
        constexpr int headDim = 512;
        constexpr int ropeDim = 64;
        constexpr int windowSize = 128;
        constexpr int position = 173;
        constexpr float eps = 1e-6f;

        // Internal prefix-cache splitting continues a prefill with a multi-token
        // suffix.  Cover the exact 256 + 28 shape used by the full-model cache
        // regression on every TP rank.
        for (int device = 0; device < std::min(2, FastllmCudaGetDeviceCount()); device++) {
            FastllmCudaSetDevice(device);
            fastllm::Data chunkKV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, {1, 28, 1, headDim},
                MakeRegressionValues(28 * headDim, 0.37f + device, 0.61f));
            Expect(FastllmCudaDeepSeekV4RotaryQuant(
                       chunkKV, ropeDim, 160000.0f, 256,
                       65536, 16.0f, 32, 1,
                       headDim - ropeDim, 64, 1),
                   "DeepSeek-V4 chunk-continuation KV rotary rejected its input");
            ForceDeviceSync();
        }
        FastllmCudaSetDevice(0);

        if (FastllmCudaGetDeviceCount() >= 2) {
            fastllm::Data chunkKV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16, {1, 28, 1, headDim},
                MakeRegressionValues(28 * headDim, 0.83f, 0.61f));
            {
                ScopedFirstDevice device("multicuda:0,1");
                bool previousAsync = fastllm::MultiCudaSetPersistentAsyncDispatch(true);
                fastllm::DeepSeekV4RotaryQuant(
                    chunkKV, ropeDim, 10000.0f, 256,
                    65536, 16.0f, 32, 1,
                    headDim - ropeDim, 64, 1);
                fastllm::MultiCudaSetPersistentAsyncDispatch(previousAsync);
            }
            Expect(chunkKV.IsTensorParallelReplicated() &&
                       chunkKV.multiDeviceData &&
                       chunkKV.multiDeviceDatas.size() == 2,
                   "DeepSeek-V4 chunk-continuation rotary lost its replicated TP layout");
            for (int device : {0, 1}) {
                auto it = chunkKV.multiDeviceDatas.find(device);
                Expect(it != chunkKV.multiDeviceDatas.end() &&
                           it->second != nullptr &&
                           GetPointerDeviceId(it->second->cudaData) == device,
                       "DeepSeek-V4 chunk-continuation rotary is missing a TP replica");
                FastllmCudaSetDevice(device);
                ForceDeviceSync();
            }
            FastllmCudaSetDevice(0);
        }

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

        // CUDA graphs keep the decode position in a stable device-side metadata
        // buffer.  Its dynamic-position kernels must remain bit-identical to
        // the ordinary scalar-position path for every speculative row, not just
        // the first row.  In particular, DSpark KV RoPE advances once per draft
        // token; using a zero position step makes later draft logits diverge.
        {
            constexpr int draftTokens = 7;
            constexpr int localHeads = 8;
            const std::vector<float> draftQValues = MakeRegressionValues(
                draftTokens * localHeads * headDim, 0.73f, 0.41f);
            fastllm::Data staticQ = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, draftTokens, localHeads, headDim}, draftQValues);
            fastllm::Data dynamicQ = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, draftTokens, localHeads, headDim}, draftQValues);
            Expect(FastllmCudaDeepSeekV4ScaleQRotary(
                       staticQ, ropeDim, 160000.0f, position,
                       0, 16.0f, 32, 1, eps),
                   "static DeepSeek-V4 draft Q rotary rejected its input");
            Expect(FastllmCudaDeepSeekV4ScaleQRotaryGraph(
                       dynamicQ, ropeDim, 160000.0f, decodeMetaPtr,
                       0, 16.0f, 32, 1, eps),
                   "dynamic DeepSeek-V4 draft Q rotary rejected its input");
            ExpectFloatNear(ToFloatVector(staticQ), ToFloatVector(dynamicQ),
                            0.0f, 0.0f,
                            "DeepSeek-V4 dynamic draft Q rotary");

            const std::vector<float> draftKVValues = MakeRegressionValues(
                draftTokens * headDim, 1.17f, 0.37f);
            fastllm::Data staticKV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, draftTokens, 1, headDim}, draftKVValues);
            fastllm::Data dynamicKV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, draftTokens, 1, headDim}, draftKVValues);
            Expect(FastllmCudaDeepSeekV4RotaryQuant(
                       staticKV, ropeDim, 160000.0f, position,
                       0, 16.0f, 32, 1, headDim - ropeDim, 64, 1),
                   "static DeepSeek-V4 draft KV rotary rejected its input");
            Expect(FastllmCudaDeepSeekV4RotaryQuantGraph(
                       dynamicKV, ropeDim, 160000.0f, decodeMetaPtr,
                       0, 16.0f, 32, 1, headDim - ropeDim, 64, 1),
                   "dynamic DeepSeek-V4 draft KV rotary rejected its input");
            ExpectFloatNear(ToFloatVector(staticKV), ToFloatVector(dynamicKV),
                            0.0f, 0.0f,
                            "DeepSeek-V4 dynamic draft KV rotary");

            const std::vector<float> attentionQValues = MakeRegressionValues(
                draftTokens * localHeads * headDim, 0.39f, 0.53f);
            const std::vector<float> attentionKVValues = MakeRegressionValues(
                (windowSize + draftTokens) * headDim, 0.97f, 0.43f);
            const std::vector<float> attentionSinkValues =
                MakeRegressionValues(localHeads, 1.23f, 0.17f);
            fastllm::Data attentionQ = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, draftTokens, localHeads, headDim}, attentionQValues);
            fastllm::Data attentionKV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, windowSize + draftTokens, headDim}, attentionKVValues);
            fastllm::Data attentionSink = MakeCudaTensor(
                fastllm::DataType::FLOAT32, {localHeads},
                attentionSinkValues);
            fastllm::Data staticAttention, dynamicAttention;
            Expect(FastllmCudaDeepSeekV4SparseAttentionPrefill(
                       attentionQ, attentionKV, attentionSink,
                       windowSize, position, 0, ropeDim, 160000.0f,
                       0, 16.0f, 32, 1,
                       1.0f / std::sqrt((float)headDim), staticAttention,
                       windowSize, true, nullptr),
                   "static DeepSeek-V4 draft sparse attention rejected its input");
            Expect(FastllmCudaDeepSeekV4SparseAttentionPrefill(
                       attentionQ, attentionKV, attentionSink,
                       windowSize, position, 0, ropeDim, 160000.0f,
                       0, 16.0f, 32, 1,
                       1.0f / std::sqrt((float)headDim), dynamicAttention,
                       windowSize, true, decodeMetaPtr),
                   "dynamic DeepSeek-V4 draft sparse attention rejected its input");
            ExpectFloatNear(ToFloatVector(staticAttention),
                            ToFloatVector(dynamicAttention), 0.0f, 0.0f,
                            "DeepSeek-V4 dynamic draft sparse attention");
        }

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
                        0.0f, 0.0f, "DeepSeek-V4 fused Q output");
        ExpectFloatNear(ToFloatVector(referenceKV), ToFloatVector(actualKV),
                        0.0f, 0.0f, "DeepSeek-V4 fused KV output");
        ExpectFloatNear(ToFloatVector(referenceCache), ToFloatVector(actualCache),
                        0.0f, 0.0f, "DeepSeek-V4 fused window cache");

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
                            0.0f, 0.0f,
                            "DeepSeek-V4 TP8 fused Q output");
            ExpectFloatNear(ToFloatVector(localReferenceKV),
                            ToFloatVector(localActualKV),
                            0.0f, 0.0f,
                            "DeepSeek-V4 TP8 fused KV output");
            ExpectFloatNear(ToFloatVector(localReferenceCache),
                            ToFloatVector(localActualCache),
                            0.0f, 0.0f,
                            "DeepSeek-V4 TP8 fused window cache");
        }

        // DSpark-7 verifies eight target rows at once.  This is the production
        // TP8 shape that used to miss the single-row fusion and execute four
        // separate Q/KV preparation kernels per layer.
        if (FastllmCudaRuntimeArch() >= 120) {
            constexpr int targetTokens = 8;
            constexpr int localHeads = 8;
            std::vector<float> targetQValues = MakeRegressionValues(
                targetTokens * localHeads * headDim, 1.07f, 0.54f);
            std::vector<float> targetKVValues = MakeRegressionValues(
                targetTokens * headDim, 1.41f, 0.69f);

            fastllm::Data targetReferenceQ = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, targetTokens, localHeads, headDim}, targetQValues);
            fastllm::Data targetReferenceKVInput = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, targetTokens, 1, headDim}, targetKVValues);
            fastllm::Data targetReferenceKV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, targetTokens, 1, headDim},
                std::vector<float>(targetTokens * headDim, 0.0f));
            Expect(FastllmCudaRMSNorm(
                       targetReferenceKVInput, kvNormWeight,
                       targetReferenceKV, eps),
                   "built-in RMSNorm rejected DSpark target QKV input");
            Expect(FastllmCudaDeepSeekV4ScaleQRotaryGraph(
                       targetReferenceQ, ropeDim, 160000.0f, decodeMetaPtr,
                       4096, 4.0f, 32, 1, eps),
                   "built-in Q rotary rejected DSpark target QKV input");
            Expect(FastllmCudaDeepSeekV4RotaryQuantGraph(
                       targetReferenceKV, ropeDim, 160000.0f, decodeMetaPtr,
                       4096, 4.0f, 32, 1, headDim - ropeDim, 64, 1),
                   "built-in KV rotary rejected DSpark target QKV input");
            targetReferenceKV.Reshape({1, targetTokens, headDim});
            fastllm::Data targetReferenceCache = MakeCudaTensor(
                fastllm::DataType::FLOAT32,
                {1, windowSize, headDim}, cacheValues);
            Expect(FastllmCudaDeepSeekV4UpdateWindowKVCacheGraph(
                       targetReferenceKV, decodeMetaPtr,
                       windowSize, targetReferenceCache),
                   "built-in cache update rejected DSpark target QKV input");

            fastllm::Data targetActualQ = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, targetTokens, localHeads, headDim}, targetQValues);
            fastllm::Data targetActualKV = MakeCudaTensor(
                fastllm::DataType::BFLOAT16,
                {1, targetTokens, 1, headDim}, targetKVValues);
            fastllm::Data targetActualCache = MakeCudaTensor(
                fastllm::DataType::FLOAT32,
                {1, windowSize, headDim}, cacheValues);
            Expect(FastllmCudaDeepSeekV4FusedQKVRopeCacheGraph(
                       targetActualQ, targetActualKV, kvNormWeight,
                       decodeMetaPtr, ropeDim, 160000.0f, 4096, 4.0f,
                       32, 1, eps, headDim - ropeDim, 64, windowSize,
                       targetActualCache),
                   "fused DeepSeek-V4 QKV rope/cache rejected DSpark target rows");
            ExpectFloatNear(ToFloatVector(targetReferenceQ),
                            ToFloatVector(targetActualQ), 0.0f, 0.0f,
                            "DeepSeek-V4 DSpark target fused Q output");
            ExpectFloatNear(ToFloatVector(targetReferenceKV),
                            ToFloatVector(targetActualKV), 0.0f, 0.0f,
                            "DeepSeek-V4 DSpark target fused KV output");
            ExpectFloatNear(ToFloatVector(targetReferenceCache),
                            ToFloatVector(targetActualCache), 0.0f, 0.0f,
                            "DeepSeek-V4 DSpark target fused window cache");
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
    void RunMultiCudaDeepSeekV4AttentionSplitUnitRegression() {
        fastllm::Data weight(
            fastllm::DataType::FP8_E4M3_BLOCK_128, {64 * 512, 1024});
        weight.name = "layers.0.attn.wq_b.weight";
        weight.tpSplitUnit = 8 * 512;
        std::vector<int> devices = {0, 1};
        std::map<int, int> ratios = {{0, 64}, {1, 57}};
        DivisionScheme scheme =
            BuildMultiCudaRowSplitScheme(weight, devices, ratios);

        Expect(scheme[0] == DivisionScheme::mapped_type({{0, 32 * 512}}) &&
                   scheme[1] == DivisionScheme::mapped_type({{32 * 512, 64 * 512}}),
               "DeepSeek-V4 heterogeneous attention split did not preserve wo_a groups");
        for (int device : devices) {
            for (const auto &range : scheme[device]) {
                Expect(range.first % weight.tpSplitUnit == 0 &&
                           range.second % weight.tpSplitUnit == 0,
                       "DeepSeek-V4 attention split is not group aligned");
            }
        }

        fastllm::Data woA(fastllm::DataType::BFLOAT16, {8 * 1024, 4096});
        woA.name = "layers.0.attn.wo_a.weight";
        woA.tpSplitUnit = 1024;
        DivisionScheme woAScheme =
            BuildMultiCudaRowSplitScheme(woA, devices, ratios);
        Expect(woAScheme[0] == DivisionScheme::mapped_type({{0, 4 * 1024}}) &&
                   woAScheme[1] == DivisionScheme::mapped_type({{4 * 1024, 8 * 1024}}),
               "DeepSeek-V4 wo_a split did not follow attention output groups");

        fastllm::Data woB(fastllm::DataType::BFLOAT16, {6144, 8 * 1024});
        woB.name = "layers.0.attn.wo_b.weight";
        woB.tpSplitUnit = 1024;
        DivisionScheme woBScheme =
            BuildMultiCudaColumnSplitScheme(woB, devices, ratios);
        Expect(woBScheme[0] == DivisionScheme::mapped_type({{0, 4 * 1024}}) &&
                   woBScheme[1] == DivisionScheme::mapped_type({{4 * 1024, 8 * 1024}}),
               "DeepSeek-V4 wo_b split did not follow attention output groups");
        std::cout << "DeepSeek-V4 heterogeneous attention split-unit regression: PASS\n";
    }

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

        constexpr int kimiTokens = 2;
        constexpr int kimiTopk = 2;
        constexpr int kimiExperts = 3;
        constexpr int kimiHidden = 256;
        constexpr int kimiInter = 256;
        constexpr float kimiBeta = 1.7f;
        constexpr float kimiLinearBeta = 2.5f;
        const std::filesystem::path kimiPath = temp.Path() / "kimi_q2k.bin";
        std::ofstream kimiFile(kimiPath, std::ios::binary);
        Expect(kimiFile.good(), "failed to create disk Kimi fixture.");
        long long kimiFileOffset = 0;
        std::vector<std::unique_ptr<fastllm::Data>> kimiResidentWeights;
        std::vector<std::unique_ptr<fastllm::Data>> kimiDiskWeights;
        std::vector<fastllm::Data*> residentW1s, residentW2s, residentW3s;
        std::vector<fastllm::Data*> diskW1s, diskW2s, diskW3s;

        auto makeKimiWeight = [&](const std::vector<int> &dims, float seed,
                                  const std::string &name) {
            fastllm::Data source = MakeFloatTensor(dims, seed);
            auto resident = std::make_unique<fastllm::Data>(
                fastllm::DataType::DATA_GGUF_FORMAT,
                (int)GGML_TYPE_Q2_K, dims);
            resident->name = name + ".resident";
            resident->disableGGUFRepack = true;
            resident->CreateFromOriData(
                fastllm::WeightType::LINEAR,
                fastllm::DataType::FLOAT32,
                source.cpuData, nullptr, nullptr);
            const uint64_t bytes = resident->GetBytes();
            kimiFile.write(
                reinterpret_cast<const char*>(resident->cpuData), bytes);
            Expect(kimiFile.good(), "failed to write disk Kimi fixture.");

            auto disk = std::make_unique<fastllm::Data>(
                fastllm::DataType::DATA_GGUF_FORMAT,
                (int)GGML_TYPE_Q2_K, dims);
            disk->name = name + ".disk";
            disk->weightType = fastllm::WeightType::LINEAR;
            disk->isDiskWeight = true;
            disk->disableGGUFRepack = true;
            fastllm::DiskWeightPart part;
            part.fileName = kimiPath.string();
            part.fileOffset = kimiFileOffset;
            part.bytes = bytes;
            part.sourceDataType = fastllm::DataType::DATA_GGUF_FORMAT;
            part.dims = dims;
            disk->diskWeightParts.push_back(part);
            kimiFileOffset += bytes;
            return std::make_pair(std::move(resident), std::move(disk));
        };

        for (int expert = 0; expert < kimiExperts; expert++) {
            auto w1 = makeKimiWeight(
                {kimiInter, kimiHidden}, 1.0f + expert,
                "regression.disk.kimi." + std::to_string(expert) + ".w1");
            auto w2 = makeKimiWeight(
                {kimiHidden, kimiInter}, 4.0f + expert,
                "regression.disk.kimi." + std::to_string(expert) + ".w2");
            auto w3 = makeKimiWeight(
                {kimiInter, kimiHidden}, 7.0f + expert,
                "regression.disk.kimi." + std::to_string(expert) + ".w3");
            residentW1s.push_back(w1.first.get());
            diskW1s.push_back(w1.second.get());
            residentW2s.push_back(w2.first.get());
            diskW2s.push_back(w2.second.get());
            residentW3s.push_back(w3.first.get());
            diskW3s.push_back(w3.second.get());
            kimiResidentWeights.push_back(std::move(w1.first));
            kimiDiskWeights.push_back(std::move(w1.second));
            kimiResidentWeights.push_back(std::move(w2.first));
            kimiDiskWeights.push_back(std::move(w2.second));
            kimiResidentWeights.push_back(std::move(w3.first));
            kimiDiskWeights.push_back(std::move(w3.second));
        }
        kimiFile.close();

        // Expert 1 is never routed below. Point its lazy metadata at a file
        // that does not exist so the regression also proves the disk operator
        // materializes selected experts only.
        const std::string unusedKimiPath =
            (temp.Path() / "unused_kimi_expert.bin").string();
        for (fastllm::Data *weight :
             {diskW1s[1], diskW2s[1], diskW3s[1]}) {
            weight->diskWeightParts[0].fileName = unusedKimiPath;
        }

        fastllm::Data kimiInput = MakeTensor(
            fastllm::DataType::BFLOAT16,
            {kimiTokens, kimiHidden}, 0.4f);
        fastllm::Data kimiIndex = MakeIntTensor(
            {kimiTokens, kimiTopk}, {0, 2, 2, 0});
        fastllm::Data kimiScore(
            fastllm::DataType::FLOAT32,
            {kimiTokens, kimiTopk}, {0.65f, 0.35f, 0.4f, 0.6f});
        fastllm::Data cpuKimiOutput, diskKimiOutput;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::KimiK3RoutedExperts(
                kimiInput, kimiIndex, kimiScore,
                residentW1s, residentW2s, residentW3s,
                kimiBeta, kimiLinearBeta, cpuKimiOutput);
        }
        {
            ScopedFirstDevice guard("disk");
            fastllm::KimiK3RoutedExperts(
                kimiInput, kimiIndex, kimiScore,
                diskW1s, diskW2s, diskW3s,
                kimiBeta, kimiLinearBeta, diskKimiOutput);
        }
        ExpectFloatNear(
            ToFloatVector(cpuKimiOutput), ToFloatVector(diskKimiOutput),
            0.0f, 0.0f, "disk Kimi Q2_K routed experts");
        for (const auto &weight : kimiDiskWeights) {
            Expect(weight->cpuData == nullptr,
                   "disk Kimi retained a routed-expert weight.");
        }
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

    void RunCudaFp8LinearAddRegression() {
        FastllmCudaSetDevice(0);
        constexpr int inputDim = 256;
        constexpr int outputDim = 256;
        constexpr int block = 128;

        fastllm::Data weight;
        weight.dataType = fastllm::DataType::FP8_E4M3_BLOCK_128;
        weight.UpdateUnitSize();
        weight.Resize({outputDim, inputDim});
        weight.weightType = fastllm::WeightType::LINEAR;
        weight.blockK = block;
        weight.blockM = block;
        weight.Allocate(false);
        const size_t perRow = fastllm::GetDataBytes(
            fastllm::DataType::FP8_E4M3_BLOCK_128, 1, inputDim);
        uint8_t *weightData = (uint8_t *)weight.cpuData;
        for (int row = 0; row < outputDim; ++row) {
            uint8_t *rowData = weightData + (size_t)row * perRow;
            for (int group = 0; group < inputDim / block; ++group) {
                uint8_t *groupData =
                    rowData + group * (block + (int)sizeof(float));
                for (int column = 0; column < block; ++column) {
                    groupData[column] = (uint8_t)(
                        0x20 + ((row * 17 + group * 13 + column) & 0x1f));
                }
                const float scale =
                    0.009f + 0.0002f * (float)((row + group) % 11);
                std::memcpy(groupData + block, &scale, sizeof(scale));
            }
        }
        weight.ToDevice(fastllm::DataDevice::CUDA);

        fastllm::Data emptyBias;
        for (fastllm::DataType dataType : {
                 fastllm::DataType::FLOAT16,
                 fastllm::DataType::BFLOAT16}) {
            const std::string typeName =
                dataType == fastllm::DataType::FLOAT16 ? "FP16" : "BF16";
            fastllm::Data input = MakeCudaTensor(
                dataType, {1, inputDim},
                MakeRegressionValues(inputDim, 0.47f, 0.19f));
            const std::vector<float> residualValues =
                MakeRegressionValues(outputDim, 1.13f, 0.23f);
            fastllm::Data reference = MakeCudaTensor(
                dataType, {1, outputDim}, residualValues);
            fastllm::Data fused = MakeCudaTensor(
                dataType, {1, outputDim}, residualValues);
            fastllm::Data projection;
            fastllm::Data middle;
            middle.isFake = true;

            {
                ScopedFirstDevice guard("cuda");
                fastllm::Linear(
                    input, weight, emptyBias, projection);
                fastllm::AddTo(reference, projection);
                Expect(fastllm::CanRunLinearAdd(
                           input, weight, emptyBias, fused),
                       typeName + " FP8 LinearAdd capability check failed");
                fastllm::LinearAdd(
                    input, weight, emptyBias, middle, fused);
            }
            FastllmCudaSyncCurrentThreadStream();
            Expect(middle.dims.empty() && middle.cudaData == nullptr,
                   typeName + " FP8 LinearAdd used the unfused fallback");
            const float tolerance =
                dataType == fastllm::DataType::FLOAT16 ? 2.0e-3f : 2.0e-2f;
            ExpectFloatNear(
                ToFloatVector(reference), ToFloatVector(fused),
                tolerance, tolerance, typeName + " FP8 LinearAdd output");
        }
    }

    void RunCudaFp16WarpRowsGemvRegression() {
        FastllmCudaSetDevice(0);
        fastllm::Data emptyBias;

        auto runCase = [&](int inputDim, int outputDim, bool withBias,
                           bool addTo, bool graphTimingCase) {
            fastllm::Data input = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {1, inputDim},
                MakeRegressionValues(inputDim, 0.31f, 0.27f));
            fastllm::Data weight = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {outputDim, inputDim},
                MakeRegressionValues(
                    (size_t)outputDim * inputDim, 0.79f, 0.019f));
            fastllm::Data bias = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {outputDim},
                MakeRegressionValues(outputDim, 1.23f, 0.043f));
            std::vector<float> initial =
                MakeRegressionValues(outputDim, 1.71f, 0.13f);
            fastllm::Data reference = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {1, outputDim}, initial);
            fastllm::Data actual = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {1, outputDim}, initial);
            const fastllm::Data &caseBias = withBias ? bias : emptyBias;

            Expect(FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
                       input, weight, caseBias, reference,
                       1, inputDim, outputDim, addTo, false),
                   "legacy FP16 GEMV rejected warp-row regression input");
            Expect(FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
                       input, weight, caseBias, actual,
                       1, inputDim, outputDim, addTo, true),
                   "warp-row FP16 GEMV rejected valid input");
            FastllmCudaSyncCurrentThreadStream();
            ExpectFloatNear(
                ToFloatVector(reference), ToFloatVector(actual),
                0.0f, 0.0f,
                "warp-row FP16 GEMV exact output m=" +
                    std::to_string(inputDim) + ", k=" +
                    std::to_string(outputDim));

            if (!graphTimingCase) {
                return;
            }

            auto replayGraph = [&](bool specialized) {
                void *graph = nullptr;
                void *graphExec = nullptr;
                Expect(FastllmCudaGraphBeginCapture(),
                       "FP16 GEMV timing graph capture did not start");
                for (int repeat = 0; repeat < 64; repeat++) {
                    Expect(FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
                               input, weight, caseBias,
                               specialized ? actual : reference,
                               1, inputDim, outputDim, addTo, specialized),
                           "FP16 GEMV failed during timing graph capture");
                }
                Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
                       "FP16 GEMV timing graph capture failed");
                Expect(FastllmCudaGraphInstantiate(graph, &graphExec) &&
                           graphExec != nullptr,
                       "FP16 GEMV timing graph instantiate failed");
                Expect(FastllmCudaGraphLaunch(graphExec),
                       "FP16 GEMV timing graph replay failed");
                FastllmCudaSyncCurrentThreadStream();
                FastllmCudaGraphExecDestroy(graphExec);
                FastllmCudaGraphDestroy(graph);
            };
            replayGraph(false);
            replayGraph(true);
        };

        runCase(2048, 2048, true, true, true);
        runCase(256, 2048, false, false, true);
    }

    void RunCudaFusedRouterSelectionRegression() {
        FastllmCudaSetDevice(0);
        constexpr int experts = 256;
        constexpr int topk = 8;
        constexpr float routeScale = 1.25f;
        std::vector<float> logitValues(experts);
        std::vector<float> biasValues(experts);
        for (int i = 0; i < experts; i++) {
            logitValues[i] =
                (float)(((i * 37 + 11) % 257) - 128) * 0.013f +
                (float)(i % 5) * 0.0007f;
            biasValues[i] =
                (float)(((i * 53 + 7) % 251) - 125) * 0.00031f;
        }
        fastllm::Data logits = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, experts}, logitValues);
        fastllm::Data gateBias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {experts}, biasValues);
        fastllm::Data index = MakeIntTensor(
            {1, topk}, std::vector<int32_t>(topk, -1));
        index.ToDevice(fastllm::DataDevice::CUDA);
        fastllm::Data score = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, topk},
            std::vector<float>(topk, 0.0f));

        Expect(FastllmCudaFusedSoftmaxSelectExpert(
                   logits, &gateBias, index, score,
                   topk, true, routeScale),
               "fused router selection rejected valid Qwen3.5 input");
        FastllmCudaSyncCurrentThreadStream();
        std::vector<int32_t> eagerIndex = ToIntVector(index);
        std::vector<float> eagerScore = ToFloatVector(score);

        std::vector<float> roundedLogits = ToFloatVector(logits);
        float maxLogit =
            *std::max_element(roundedLogits.begin(), roundedLogits.end());
        std::vector<float> probabilities(experts);
        float probabilitySum = 0.0f;
        for (int i = 0; i < experts; i++) {
            probabilities[i] = std::exp(roundedLogits[i] - maxLogit);
            probabilitySum += probabilities[i];
        }
        for (float &probability : probabilities) {
            probability /= probabilitySum;
        }
        std::vector<int32_t> expectedIndex(experts);
        std::iota(expectedIndex.begin(), expectedIndex.end(), 0);
        std::sort(
            expectedIndex.begin(), expectedIndex.end(),
            [&](int32_t a, int32_t b) {
                float keyA = probabilities[a] + biasValues[a];
                float keyB = probabilities[b] + biasValues[b];
                return keyA != keyB ? keyA > keyB : a < b;
            });
        expectedIndex.resize(topk);
        ExpectIntEqual(expectedIndex, eagerIndex,
                       "fused router selection top-8");
        float eagerScoreSum =
            std::accumulate(eagerScore.begin(), eagerScore.end(), 0.0f);
        Expect(std::fabs(eagerScoreSum - routeScale) < 2.0e-5f,
               "fused router normalized scores do not sum to route scale");

        fastllm::Data noBiasIndex = MakeIntTensor(
            {1, topk}, std::vector<int32_t>(topk, -1));
        noBiasIndex.ToDevice(fastllm::DataDevice::CUDA);
        fastllm::Data noBiasScore = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, topk},
            std::vector<float>(topk, 0.0f));
        Expect(FastllmCudaFusedSoftmaxSelectExpert(
                   logits, nullptr, noBiasIndex, noBiasScore,
                   topk, true, routeScale),
               "bias-free fused router selection rejected valid input");
        FastllmCudaSyncCurrentThreadStream();
        std::vector<int32_t> expectedNoBiasIndex(experts);
        std::iota(
            expectedNoBiasIndex.begin(), expectedNoBiasIndex.end(), 0);
        std::sort(
            expectedNoBiasIndex.begin(), expectedNoBiasIndex.end(),
            [&](int32_t a, int32_t b) {
                return roundedLogits[a] != roundedLogits[b]
                    ? roundedLogits[a] > roundedLogits[b]
                    : a < b;
            });
        expectedNoBiasIndex.resize(topk);
        std::vector<int32_t> eagerNoBiasIndex =
            ToIntVector(noBiasIndex);
        std::vector<float> eagerNoBiasScore =
            ToFloatVector(noBiasScore);
        ExpectIntEqual(expectedNoBiasIndex, eagerNoBiasIndex,
                       "bias-free fused router selection top-8");
        float eagerNoBiasScoreSum = std::accumulate(
            eagerNoBiasScore.begin(), eagerNoBiasScore.end(), 0.0f);
        Expect(std::fabs(eagerNoBiasScoreSum - routeScale) < 2.0e-5f,
               "bias-free fused router scores do not sum to route scale");

        fastllm::Data zeroBias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {experts},
            std::vector<float>(experts, 0.0f));
        fastllm::Data fullSoftmaxIndex = MakeIntTensor(
            {1, topk}, std::vector<int32_t>(topk, -1));
        fullSoftmaxIndex.ToDevice(fastllm::DataDevice::CUDA);
        fastllm::Data fullSoftmaxScore = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, topk},
            std::vector<float>(topk, 0.0f));
        Expect(FastllmCudaFusedSoftmaxSelectExpert(
                   logits, &zeroBias,
                   fullSoftmaxIndex, fullSoftmaxScore,
                   topk, true, routeScale),
               "full-softmax router reference rejected zero bias");
        FastllmCudaSyncCurrentThreadStream();
        ExpectIntEqual(
            eagerNoBiasIndex, ToIntVector(fullSoftmaxIndex),
            "selected-logit and full-softmax router indices");
        ExpectFloatNear(
            ToFloatVector(fullSoftmaxScore), eagerNoBiasScore,
            2.0e-6f, 2.0e-6f,
            "selected-logit and full-softmax router scores");

        fastllm::Data tiedLogits = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, experts},
            std::vector<float>(experts, 0.0f));
        fastllm::Data tiedIndex = MakeIntTensor(
            {1, topk}, std::vector<int32_t>(topk, -1));
        tiedIndex.ToDevice(fastllm::DataDevice::CUDA);
        fastllm::Data tiedScore = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {1, topk},
            std::vector<float>(topk, 0.0f));
        Expect(FastllmCudaFusedSoftmaxSelectExpert(
                   tiedLogits, nullptr, tiedIndex, tiedScore,
                   topk, true, routeScale),
               "bias-free fused router rejected tied logits");
        FastllmCudaSyncCurrentThreadStream();
        ExpectIntEqual(
            {31, 95, 159, 223, 63, 127, 191, 255},
            ToIntVector(tiedIndex),
            "fused router legacy tie ordering");
        std::vector<float> tiedScores = ToFloatVector(tiedScore);
        for (float value : tiedScores) {
            Expect(std::fabs(value - routeScale / topk) < 1.0e-6f,
                   "fused router tied score is not uniform");
        }

        constexpr int multiTokens = 4;
        std::vector<float> multiLogitValues(multiTokens * experts);
        for (int token = 0; token < multiTokens; token++) {
            for (int expert = 0; expert < experts; expert++) {
                multiLogitValues[token * experts + expert] =
                    (float)((((expert * (37 + token * 2) +
                               token * 41 + 11) % 257) - 128)) *
                        (0.009f + token * 0.001f) +
                    (float)((expert + token) % 7) * 0.00031f;
            }
        }
        fastllm::Data multiLogits = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {multiTokens, experts},
            multiLogitValues);
        fastllm::Data multiReferenceIndex = MakeIntTensor(
            {multiTokens, topk},
            std::vector<int32_t>(multiTokens * topk, -1));
        multiReferenceIndex.ToDevice(fastllm::DataDevice::CUDA);
        fastllm::Data multiReferenceScore = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {multiTokens, topk},
            std::vector<float>(multiTokens * topk, 0.0f));
        fastllm::Data multiPackedIndex = MakeIntTensor(
            {multiTokens, topk},
            std::vector<int32_t>(multiTokens * topk, -1));
        multiPackedIndex.ToDevice(fastllm::DataDevice::CUDA);
        fastllm::Data multiPackedScore = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {multiTokens, topk},
            std::vector<float>(multiTokens * topk, 0.0f));
        Expect(FastllmCudaFusedSoftmaxSelectExpert(
                   multiLogits, &zeroBias,
                   multiReferenceIndex, multiReferenceScore,
                   topk, true, routeScale),
               "multi-token full-softmax router reference failed");
        Expect(FastllmCudaFusedSoftmaxSelectExpert(
                   multiLogits, nullptr,
                   multiPackedIndex, multiPackedScore,
                   topk, true, routeScale),
               "multi-token packed router selection failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectIntEqual(
            ToIntVector(multiReferenceIndex),
            ToIntVector(multiPackedIndex),
            "multi-token packed router indices");
        ExpectFloatNear(
            ToFloatVector(multiReferenceScore),
            ToFloatVector(multiPackedScore),
            2.0e-6f, 2.0e-6f,
            "multi-token packed router scores");

        void *graph = nullptr;
        void *graphExec = nullptr;
        Expect(FastllmCudaGraphBeginCapture(),
               "fused router timing graph capture did not start");
        for (int repeat = 0; repeat < 64; repeat++) {
            Expect(FastllmCudaFusedSoftmaxSelectExpert(
                       logits, nullptr, noBiasIndex, noBiasScore,
                       topk, true, routeScale),
                   "fused router selection failed during graph capture");
        }
        Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
               "fused router timing graph capture failed");
        Expect(FastllmCudaGraphInstantiate(graph, &graphExec) &&
                   graphExec != nullptr,
               "fused router timing graph instantiate failed");
        Expect(FastllmCudaGraphLaunch(graphExec),
               "fused router timing graph replay failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectIntEqual(eagerNoBiasIndex, ToIntVector(noBiasIndex),
                       "bias-free fused router graph top-8");
        ExpectFloatNear(
            eagerNoBiasScore, ToFloatVector(noBiasScore),
            0.0f, 0.0f, "bias-free fused router graph scores");
        FastllmCudaGraphExecDestroy(graphExec);
        FastllmCudaGraphDestroy(graph);

        graph = nullptr;
        graphExec = nullptr;
        Expect(FastllmCudaGraphBeginCapture(),
               "multi-token packed router graph capture did not start");
        Expect(FastllmCudaFusedSoftmaxSelectExpert(
                   multiLogits, nullptr,
                   multiPackedIndex, multiPackedScore,
                   topk, true, routeScale),
               "multi-token packed router failed during graph capture");
        Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
               "multi-token packed router graph capture failed");
        Expect(FastllmCudaGraphInstantiate(graph, &graphExec) &&
                   graphExec != nullptr,
               "multi-token packed router graph instantiate failed");
        Expect(FastllmCudaGraphLaunch(graphExec),
               "multi-token packed router graph replay failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectIntEqual(
            ToIntVector(multiReferenceIndex),
            ToIntVector(multiPackedIndex),
            "multi-token packed router graph indices");
        ExpectFloatNear(
            ToFloatVector(multiReferenceScore),
            ToFloatVector(multiPackedScore),
            2.0e-6f, 2.0e-6f,
            "multi-token packed router graph scores");
        FastllmCudaGraphExecDestroy(graphExec);
        FastllmCudaGraphDestroy(graph);
    }

    void RunCudaQwen35RouterSharedGateFusionRegression() {
        FastllmCudaSetDevice(0);
        constexpr int hidden = 2048;
        constexpr int experts = 256;
        fastllm::Data input = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, hidden},
            MakeRegressionValues(hidden, 0.29f, 0.37f));
        fastllm::Data routerWeight = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {experts, hidden},
            MakeRegressionValues(experts * hidden, 0.83f, 0.051f));
        fastllm::Data sharedGateWeight = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, hidden},
            MakeRegressionValues(hidden, 1.37f, 0.063f));
        fastllm::Data emptyBias;

        fastllm::Data referenceRouter = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, experts},
            std::vector<float>(experts, 0.0f));
        fastllm::Data referenceSharedGate = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, 1},
            std::vector<float>(1, 0.0f));
        Expect(FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
                   input, routerWeight, emptyBias, referenceRouter,
                   1, hidden, experts, false, false),
               "Qwen3.5 router reference GEMV failed");
        Expect(FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
                   input, sharedGateWeight, emptyBias, referenceSharedGate,
                   1, hidden, 1, false, true),
               "Qwen3.5 shared-gate reference GEMV failed");
        Expect(FastllmCudaSigmoid(
                   referenceSharedGate, referenceSharedGate),
               "Qwen3.5 shared-gate reference sigmoid failed");

        fastllm::Data fusedRouter = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, experts},
            std::vector<float>(experts, 0.0f));
        fastllm::Data fusedSharedGate = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, 1},
            std::vector<float>(1, 0.0f));
        Expect(FastllmCudaQwen35RouterSharedGateFloat16(
                   input, routerWeight, sharedGateWeight,
                   fusedRouter, fusedSharedGate, true),
               "Qwen3.5 fused router/shared-gate GEMV rejected valid tensors");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(
            ToFloatVector(referenceRouter), ToFloatVector(fusedRouter),
            0.0f, 0.0f,
            "Qwen3.5 fused router GEMV exact FP16 result");
        ExpectFloatNear(
            ToFloatVector(referenceSharedGate),
            ToFloatVector(fusedSharedGate),
            0.0f, 0.0f,
            "Qwen3.5 fused shared-gate GEMV exact FP16 result");

        constexpr int multiRows = 7;
        fastllm::Data multiInput = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {multiRows, hidden},
            MakeRegressionValues(multiRows * hidden, 1.73f, 0.41f));
        fastllm::Data multiReferenceRouter = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {multiRows, experts},
            std::vector<float>(multiRows * experts, 0.0f));
        fastllm::Data multiReferenceSharedGate = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {multiRows, 1},
            std::vector<float>(multiRows, 0.0f));
        Expect(FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
                   multiInput, routerWeight, emptyBias,
                   multiReferenceRouter,
                   multiRows, hidden, experts, false, false),
               "Qwen3.5 multi-row router reference GEMV failed");
        Expect(FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
                   multiInput, sharedGateWeight, emptyBias,
                   multiReferenceSharedGate,
                   multiRows, hidden, 1, false, true),
               "Qwen3.5 multi-row shared-gate reference GEMV failed");
        Expect(FastllmCudaSigmoid(
                   multiReferenceSharedGate, multiReferenceSharedGate),
               "Qwen3.5 multi-row shared-gate reference sigmoid failed");

        fastllm::Data multiFusedRouter = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {multiRows, experts},
            std::vector<float>(multiRows * experts, 0.0f));
        fastllm::Data multiFusedSharedGate = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {multiRows, 1},
            std::vector<float>(multiRows, 0.0f));
        Expect(FastllmCudaQwen35RouterSharedGateFloat16(
                   multiInput, routerWeight, sharedGateWeight,
                   multiFusedRouter, multiFusedSharedGate, true),
               "Qwen3.5 fused router/shared-gate GEMV rejected multi-row tensors");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(
            ToFloatVector(multiReferenceRouter),
            ToFloatVector(multiFusedRouter),
            0.0f, 0.0f,
            "Qwen3.5 fused multi-row router exact FP16 result");
        ExpectFloatNear(
            ToFloatVector(multiReferenceSharedGate),
            ToFloatVector(multiFusedSharedGate),
            0.0f, 0.0f,
            "Qwen3.5 fused multi-row shared-gate exact FP16 result");

        fastllm::Data graphRouter = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {multiRows, experts},
            std::vector<float>(multiRows * experts, 0.0f));
        fastllm::Data graphSharedGate = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {multiRows, 1},
            std::vector<float>(multiRows, 0.0f));
        void *graph = nullptr;
        void *graphExec = nullptr;
        Expect(FastllmCudaGraphBeginCapture(),
               "Qwen3.5 fused router/shared-gate graph capture did not start");
        for (int repeat = 0; repeat < 64; repeat++) {
            Expect(FastllmCudaQwen35RouterSharedGateFloat16(
                       multiInput, routerWeight, sharedGateWeight,
                       graphRouter, graphSharedGate, true),
                   "Qwen3.5 fused router/shared-gate GEMV failed during graph capture");
        }
        Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
               "Qwen3.5 fused router/shared-gate graph capture failed");
        Expect(FastllmCudaGraphInstantiate(graph, &graphExec) &&
                   graphExec != nullptr,
               "Qwen3.5 fused router/shared-gate graph instantiate failed");
        Expect(FastllmCudaGraphLaunch(graphExec),
               "Qwen3.5 fused router/shared-gate graph replay failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(
            ToFloatVector(multiReferenceRouter), ToFloatVector(graphRouter),
            0.0f, 0.0f,
            "Qwen3.5 fused multi-row router graph exact FP16 result");
        ExpectFloatNear(
            ToFloatVector(multiReferenceSharedGate),
            ToFloatVector(graphSharedGate),
            0.0f, 0.0f,
            "Qwen3.5 fused multi-row shared-gate graph exact FP16 result");
        FastllmCudaGraphExecDestroy(graphExec);
        FastllmCudaGraphDestroy(graph);

    }

    void RunCudaQwen35FusedMoeJoinRegression() {
        FastllmCudaSetDevice(0);
        constexpr int hidden = 2048;

        auto runCase = [&](int rows, bool hasGate, bool addResidual,
                           bool gateAlreadySigmoid) {
            const int count = rows * hidden;
            const std::vector<float> residualValues =
                MakeRegressionValues(count, 0.17f, 0.45f);
            const std::vector<float> routedValues =
                MakeRegressionValues(count, 0.73f, 0.31f);
            const std::vector<float> sharedValues =
                MakeRegressionValues(count, 1.19f, 0.27f);
            const std::vector<float> gateValues =
                MakeRegressionValues(rows, 1.67f, 1.4f);

            fastllm::Data referenceResidual = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {rows, hidden}, residualValues);
            fastllm::Data referenceRouted = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {rows, hidden}, routedValues);
            fastllm::Data referenceShared = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {rows, hidden}, sharedValues);
            std::unique_ptr<fastllm::Data> referenceGate;
            if (hasGate) {
                referenceGate = std::make_unique<fastllm::Data>(
                    fastllm::DataType::FLOAT16,
                    std::vector<int>{rows}, gateValues);
                referenceGate->ToDevice(fastllm::DataDevice::CUDA);
                Expect(FastllmCudaSigmoid(*referenceGate, *referenceGate),
                       "Qwen3.5 fused MoE join reference sigmoid failed");
                Expect(FastllmCudaMulTo(
                           referenceShared, *referenceGate, 1.0f),
                       "Qwen3.5 fused MoE join reference scale failed");
            }
            Expect(FastllmCudaAddTo(
                       referenceRouted, referenceShared, 1.0f),
                   "Qwen3.5 fused MoE join reference branch add failed");
            fastllm::Data *referenceOutput = &referenceRouted;
            if (addResidual) {
                Expect(FastllmCudaAddTo(
                           referenceResidual, referenceRouted, 1.0f),
                       "Qwen3.5 fused MoE join reference residual add failed");
                referenceOutput = &referenceResidual;
            }

            fastllm::Data fusedResidual = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {rows, hidden}, residualValues);
            fastllm::Data fusedRouted = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {rows, hidden}, routedValues);
            fastllm::Data fusedShared = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {rows, hidden}, sharedValues);
            std::unique_ptr<fastllm::Data> fusedGate;
            const fastllm::Data *fusedGatePointer = nullptr;
            if (hasGate) {
                fusedGate = std::make_unique<fastllm::Data>(
                    fastllm::DataType::FLOAT16,
                    std::vector<int>{rows}, gateValues);
                fusedGate->ToDevice(fastllm::DataDevice::CUDA);
                if (gateAlreadySigmoid) {
                    Expect(FastllmCudaSigmoid(*fusedGate, *fusedGate),
                           "Qwen3.5 fused MoE pre-sigmoid gate setup failed");
                }
                fusedGatePointer = fusedGate.get();
            }
            fastllm::Data &fusedOutput =
                addResidual ? fusedResidual : fusedRouted;
            Expect(FastllmCudaQwen35FusedMoeJoin(
                       fusedOutput, fusedRouted, fusedShared,
                       fusedGatePointer, addResidual,
                       gateAlreadySigmoid),
                   "Qwen3.5 fused MoE join rejected a valid FLOAT16 case");

            FastllmCudaSyncCurrentThreadStream();
            ExpectFloatNear(
                ToFloatVector(*referenceOutput), ToFloatVector(fusedOutput),
                0.0f, 0.0f,
                "Qwen3.5 fused MoE join exact FP16 result rows=" +
                    std::to_string(rows) + " gate=" +
                    std::to_string(hasGate) + " residual=" +
                    std::to_string(addResidual) + " pre_sigmoid=" +
                    std::to_string(gateAlreadySigmoid));
        };

        runCase(1, false, true, false);
        runCase(1, true, true, false);
        runCase(1, true, true, true);
        runCase(1, true, false, false);
        runCase(3, true, true, false);
        runCase(3, false, true, false);

        const std::vector<float> residualValues =
            MakeRegressionValues(hidden, 2.03f, 0.41f);
        const std::vector<float> routedValues =
            MakeRegressionValues(hidden, 2.41f, 0.29f);
        const std::vector<float> sharedValues =
            MakeRegressionValues(hidden, 2.79f, 0.23f);
        const std::vector<float> gateValues = {-0.625f};
        fastllm::Data referenceResidual = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, hidden}, residualValues);
        fastllm::Data referenceRouted = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, hidden}, routedValues);
        fastllm::Data referenceShared = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, hidden}, sharedValues);
        fastllm::Data referenceGate = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1}, gateValues);
        Expect(FastllmCudaSigmoid(referenceGate, referenceGate) &&
                   FastllmCudaMulTo(referenceShared, referenceGate, 1.0f) &&
                   FastllmCudaAddTo(referenceRouted, referenceShared, 1.0f) &&
                   FastllmCudaAddTo(referenceResidual, referenceRouted, 1.0f),
               "Qwen3.5 fused MoE graph reference failed");

        fastllm::Data graphResidual = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, hidden}, residualValues);
        fastllm::Data graphRouted = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, hidden}, routedValues);
        fastllm::Data graphShared = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1, hidden}, sharedValues);
        fastllm::Data graphGate = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {1}, gateValues);
        void *graph = nullptr;
        void *graphExec = nullptr;
        Expect(FastllmCudaGraphBeginCapture(),
               "Qwen3.5 fused MoE join graph capture did not start");
        Expect(FastllmCudaQwen35FusedMoeJoin(
                   graphResidual, graphRouted, graphShared,
                   &graphGate, true),
               "Qwen3.5 fused MoE join failed during graph capture");
        Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
               "Qwen3.5 fused MoE join graph capture failed");
        Expect(FastllmCudaGraphInstantiate(graph, &graphExec) &&
                   graphExec != nullptr,
               "Qwen3.5 fused MoE join graph instantiate failed");
        Expect(FastllmCudaGraphLaunch(graphExec),
               "Qwen3.5 fused MoE join graph replay failed");
        FastllmCudaSyncCurrentThreadStream();
        ExpectFloatNear(
            ToFloatVector(referenceResidual), ToFloatVector(graphResidual),
            0.0f, 0.0f,
            "Qwen3.5 fused MoE join graph exact FP16 result");
        FastllmCudaGraphExecDestroy(graphExec);
        FastllmCudaGraphDestroy(graph);
    }

    void RunCudaInt4GroupBatch1MoeRegressionCase(int hidden, int inter,
                                                  int expertCount, int topk,
                                                  bool runSmallBatchChecks) {
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
            weight->name =
                "regression.model.layers.0.mlp.experts." +
                std::to_string(expert) + "." + std::to_string(salt);
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
            // Production AWQ expert shards use direct allocations so the
            // compact representation can be returned to CUDA immediately
            // after grouped-Marlin repacking.
            weight->directMemory = true;
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

        std::vector<int32_t> routeIndices(topk);
        std::vector<float> routeScores(topk);
        float scoreSum = 0.0f;
        for (int route = 0; route < topk; route++) {
            routeIndices[route] = (route * 3 + 1) % expertCount;
            routeScores[route] = (float)(topk - route);
            scoreSum += routeScores[route];
        }
        for (float &score : routeScores) {
            score /= scoreSum;
        }
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

        const bool productionMarlinShape =
            hidden == 2048 && inter == 256 &&
            expertCount == 8 && topk == 8;
        bool ok = false;
        if (!productionMarlinShape) {
            ok = FastllmCudaHalfMergeMOEInt4GroupBatch1Indexed(
                input, scratch, output, weights.data(), (int)weights.size(),
                (const int32_t*)index.cudaData,
                (const float*)score.cudaData, topk);
            Expect(ok,
                   "CUDA INT4_GROUP batch-1 fused MoE path was not selected.");
            ExpectFloatNear(expected, ToFloatVector(output), 3e-2f, 3e-2f,
                            "CUDA INT4_GROUP batch-1 fused MoE");

            fastllm::Data graphOutput(fastllm::DataType::FLOAT16);
            graphOutput.Resize({1, hidden});
            graphOutput.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
            graphOutput.Allocate(false);
            FastllmCudaMemset0(output.cudaData, output.GetBytes());
            FastllmCudaMemset0(
                graphOutput.cudaData, graphOutput.GetBytes());

            void *graph = nullptr;
            void *graphExec = nullptr;
            Expect(
                FastllmCudaGraphBeginCapture(),
                "CUDA INT4_GROUP batch-1 fused MoE graph capture did not "
                "start.");
            ok = FastllmCudaHalfMergeMOEInt4GroupBatch1Indexed(
                input, scratch, output, weights.data(), (int)weights.size(),
                (const int32_t*)index.cudaData,
                (const float*)score.cudaData, topk);
            Expect(
                ok,
                "CUDA INT4_GROUP batch-1 fused MoE path failed during graph "
                "capture.");
            // Also verify that a downstream operation captured after the
            // fused MoE observes its completed result during graph replay.
            FastllmCudaCopyFromDeviceToDevice(
                graphOutput.cudaData, output.cudaData, output.GetBytes());
            Expect(FastllmCudaGraphEndCapture(&graph) && graph != nullptr,
                   "CUDA INT4_GROUP batch-1 fused MoE graph capture failed.");
            Expect(
                FastllmCudaGraphInstantiate(graph, &graphExec) &&
                    graphExec != nullptr,
                "CUDA INT4_GROUP batch-1 fused MoE graph instantiate failed.");
            Expect(FastllmCudaGraphLaunch(graphExec),
                   "CUDA INT4_GROUP batch-1 fused MoE graph replay failed.");
            ExpectFloatNear(
                expected, ToFloatVector(graphOutput), 3e-2f, 3e-2f,
                "CUDA INT4_GROUP batch-1 fused MoE graph downstream "
                "consumer");
            FastllmCudaGraphExecDestroy(graphExec);
            FastllmCudaGraphDestroy(graph);
        }

        if ((hidden == 128 && inter == 256 &&
             expertCount == 4 && topk == 2) ||
            (hidden == 2048 && inter == 256 &&
             expertCount == 8 && topk == 8)) {
            // Reproduce the lifecycle that originally exposed the repack
            // double-free: an earlier legacy linear invocation creates
            // extraCudaHalfData[0/1] aliases to extraCudaData[0/1], then the
            // grouped-Marlin builder replaces and releases the compact AWQ
            // representation. Production routed-expert names deliberately
            // keep this invocation on the legacy layout.
            fastllm::Data legacyGateOutput(fastllm::DataType::FLOAT16);
            legacyGateOutput.Resize({1, inter * 2});
            legacyGateOutput.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
            legacyGateOutput.Allocate(false);
            fastllm::Data emptyBias;
            Expect(
                FastllmCudaHalfMatMulFloatInt4Group(
                    input, *weights[2], emptyBias, legacyGateOutput,
                    1, hidden, inter * 2),
                "CUDA INT4_GROUP legacy metadata priming failed before "
                "grouped-Marlin repack.");
            FastllmCudaSyncCurrentThreadStream();
            Expect(
                weights[2]->extraCudaData.size() >= 2 &&
                    weights[2]->extraCudaHalfData.size() >= 2 &&
                    weights[2]->extraCudaData[0] != nullptr &&
                    weights[2]->extraCudaData[1] != nullptr &&
                    weights[2]->extraCudaHalfData[0] ==
                        weights[2]->extraCudaData[0] &&
                    weights[2]->extraCudaHalfData[1] ==
                        weights[2]->extraCudaData[1],
                "CUDA INT4_GROUP legacy metadata did not create the expected "
                "scale/min pointer aliases.");

            fastllm::Data marlinDecodeGate, marlinDecodeActivation;
            fastllm::Data marlinDecodeOutput(fastllm::DataType::FLOAT16);
            marlinDecodeOutput.Resize({1, hidden});
            marlinDecodeOutput.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
            cudaGetLastError();
            ok = FastllmCudaHalfMergeMOEInt4GroupMarlinIndexed(
                input, marlinDecodeGate, marlinDecodeActivation,
                marlinDecodeOutput, weights.data(), (int)weights.size(),
                (const int32_t*)index.cudaData,
                (const float*)score.cudaData, 1, topk);
            cudaError_t marlinLaunchError = cudaGetLastError();
            Expect(
                ok,
                std::string(
                    "CUDA INT4_GROUP grouped Marlin decode path was not "
                    "selected: ") +
                    cudaGetErrorString(marlinLaunchError));
            FastllmCudaSyncCurrentThreadStream();
            ExpectFloatNear(
                expected, ToFloatVector(marlinDecodeOutput),
                5e-2f, 5e-2f,
                "CUDA INT4_GROUP grouped Marlin decode");
            for (int expert = 0; expert < expertCount; expert++) {
                fastllm::Data *gateup = weights[2 + expert * 2];
                fastllm::Data *down = weights[3 + expert * 2];
                Expect(gateup->cudaData == nullptr &&
                           down->cudaData == nullptr,
                       "CUDA INT4_GROUP grouped Marlin repack retained "
                       "compact expert weights.");
                for (void *pointer : gateup->extraCudaData) {
                    Expect(pointer == nullptr,
                           "CUDA INT4_GROUP grouped Marlin repack retained "
                           "gate quant metadata.");
                }
                for (void *pointer : down->extraCudaData) {
                    Expect(pointer == nullptr,
                           "CUDA INT4_GROUP grouped Marlin repack retained "
                           "down quant metadata.");
                }
                for (void *pointer : gateup->extraCudaHalfData) {
                    Expect(pointer == nullptr,
                           "CUDA INT4_GROUP grouped Marlin repack retained "
                           "gate half metadata.");
                }
                for (void *pointer : down->extraCudaHalfData) {
                    Expect(pointer == nullptr,
                           "CUDA INT4_GROUP grouped Marlin repack retained "
                           "down half metadata.");
                }
            }

            // Once repacking has released the compact weights, a runtime
            // failure is no longer recoverable through the legacy INT4 path.
            // Make the activation scratch deliberately non-allocating and
            // verify that the operator fails explicitly instead of returning
            // false to the dispatcher's fallback chain.
            {
                fastllm::Data failedGateOutput;
                fastllm::Data failedActivation;
                fastllm::Data failedOutput(fastllm::DataType::FLOAT16);
                failedActivation.isFake = true;
                bool rejectedAfterRepack = false;
                try {
                    FastllmCudaHalfMergeMOEInt4GroupMarlinIndexed(
                        input, failedGateOutput, failedActivation,
                        failedOutput, weights.data(), (int)weights.size(),
                        (const int32_t*)index.cudaData,
                        (const float*)score.cudaData, 1, topk);
                } catch (const std::runtime_error &error) {
                    rejectedAfterRepack =
                        std::string(error.what()).find(
                            "legacy INT4_GROUP fallback is unavailable") !=
                        std::string::npos;
                }
                Expect(
                    rejectedAfterRepack,
                    "CUDA INT4_GROUP grouped Marlin failure returned to "
                    "the released-weight fallback path.");
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
            }

            fastllm::Data marlinGraphOutput(fastllm::DataType::FLOAT16);
            marlinGraphOutput.Resize({1, hidden});
            marlinGraphOutput.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
            marlinGraphOutput.Allocate(false);
            void *marlinGraph = nullptr;
            void *marlinGraphExec = nullptr;
            Expect(FastllmCudaGraphBeginCapture(),
                   "CUDA INT4_GROUP grouped Marlin decode graph capture "
                   "did not start.");
            ok = FastllmCudaHalfMergeMOEInt4GroupMarlinIndexed(
                input, marlinDecodeGate, marlinDecodeActivation,
                marlinDecodeOutput, weights.data(), (int)weights.size(),
                (const int32_t*)index.cudaData,
                (const float*)score.cudaData, 1, topk);
            Expect(ok,
                   "CUDA INT4_GROUP grouped Marlin decode failed during "
                   "graph capture.");
            FastllmCudaCopyFromDeviceToDevice(
                marlinGraphOutput.cudaData, marlinDecodeOutput.cudaData,
                marlinDecodeOutput.GetBytes());
            Expect(FastllmCudaGraphEndCapture(&marlinGraph) &&
                       marlinGraph != nullptr,
                   "CUDA INT4_GROUP grouped Marlin decode graph capture "
                   "failed.");
            Expect(FastllmCudaGraphInstantiate(
                       marlinGraph, &marlinGraphExec) &&
                       marlinGraphExec != nullptr,
                   "CUDA INT4_GROUP grouped Marlin decode graph "
                   "instantiate failed.");
            const int marlinGraphReplayCount =
                productionMarlinShape ? 64 : 1;
            for (int replay = 0; replay < marlinGraphReplayCount; ++replay) {
                Expect(FastllmCudaGraphLaunch(marlinGraphExec),
                       "CUDA INT4_GROUP grouped Marlin decode graph replay "
                       "failed.");
            }
            ExpectFloatNear(
                expected, ToFloatVector(marlinGraphOutput),
                5e-2f, 5e-2f,
                "CUDA INT4_GROUP grouped Marlin decode graph");

            if (hidden == 128 && expertCount == 4 && topk == 2) {
                constexpr int marlinBatch = 65;
                std::vector<float> marlinInputValues(
                    (size_t)marlinBatch * hidden);
                std::vector<int32_t> marlinRouteIndices(
                    (size_t)marlinBatch * topk);
                std::vector<float> marlinRouteScores(
                    (size_t)marlinBatch * topk);
                std::vector<float> marlinExpected(
                    (size_t)marlinBatch * hidden, 0.0f);
                for (int b = 0; b < marlinBatch; b++) {
                    for (int x = 0; x < hidden; x++) {
                        marlinInputValues[(size_t)b * hidden + x] =
                            inputValues[x] * (0.625f + 0.003f * b) +
                            (float)((b + x) % 11 - 5) / 256.0f;
                    }
                    marlinRouteIndices[(size_t)b * topk] =
                        (b * 3 + 1) % expertCount;
                    marlinRouteIndices[(size_t)b * topk + 1] =
                        (b * 5 + 2) % expertCount;
                    marlinRouteScores[(size_t)b * topk] =
                        0.35f + 0.05f * (b % 4);
                    marlinRouteScores[(size_t)b * topk + 1] =
                        1.0f - marlinRouteScores[(size_t)b * topk];
                }
                for (int b = 0; b < marlinBatch; b++) {
                    const float *rowInput =
                        marlinInputValues.data() + (size_t)b * hidden;
                    for (int route = 0; route < topk; route++) {
                        int routed = b * topk + route;
                        int expert = marlinRouteIndices[routed];
                        std::vector<float> middle(inter);
                        for (int j = 0; j < inter; j++) {
                            float gate = 0.0f, up = 0.0f;
                            for (int x = 0; x < hidden; x++) {
                                gate += rowInput[x] *
                                    gateupFloat[expert][
                                        (size_t)j * hidden + x];
                                up += rowInput[x] *
                                    gateupFloat[expert][
                                        (size_t)(inter + j) * hidden + x];
                            }
                            middle[j] =
                                (gate / (1.0f + std::exp(-gate))) * up;
                        }
                        for (int out = 0; out < hidden; out++) {
                            float value = 0.0f;
                            for (int j = 0; j < inter; j++) {
                                value += middle[j] *
                                    downFloat[expert][
                                        (size_t)out * inter + j];
                            }
                            marlinExpected[(size_t)b * hidden + out] +=
                                marlinRouteScores[routed] * value;
                        }
                    }
                }

                fastllm::Data marlinInput(
                    fastllm::DataType::FLOAT16,
                    {marlinBatch, hidden}, marlinInputValues);
                marlinInput.ToDevice(
                    fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
                fastllm::Data marlinIndex = MakeIntTensor(
                    {marlinBatch, topk}, marlinRouteIndices);
                marlinIndex.ToDevice(
                    fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
                fastllm::Data marlinScore(
                    fastllm::DataType::FLOAT32,
                    {marlinBatch, topk}, marlinRouteScores);
                marlinScore.ToDevice(
                    fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
                fastllm::Data marlinGateOutput, marlinActivation;
                fastllm::Data marlinOutput(fastllm::DataType::FLOAT16);
                marlinOutput.Resize({marlinBatch, hidden});
                marlinOutput.ToDevice(
                    fastllm::DataDevice::CUDA, std::vector<int>{0}, false);
                ok = FastllmCudaHalfMergeMOEInt4GroupMarlinIndexed(
                    marlinInput, marlinGateOutput, marlinActivation,
                    marlinOutput, weights.data(), (int)weights.size(),
                    (const int32_t*)marlinIndex.cudaData,
                    (const float*)marlinScore.cudaData,
                    marlinBatch, topk);
                Expect(
                    ok,
                    "CUDA INT4_GROUP grouped Marlin MoE path was not "
                    "selected.");
                FastllmCudaSyncCurrentThreadStream();
                ExpectFloatNear(
                    marlinExpected, ToFloatVector(marlinOutput),
                    5e-2f, 5e-2f,
                    "CUDA INT4_GROUP grouped Marlin MoE");

                // Large-prefill metadata lives in separate storage. Replaying
                // the previously captured decode graph proves its route
                // pointers did not move during prefill growth.
                FastllmCudaMemset0(
                    marlinGraphOutput.cudaData,
                    marlinGraphOutput.GetBytes());
                Expect(FastllmCudaGraphLaunch(marlinGraphExec),
                       "CUDA INT4_GROUP grouped Marlin decode graph replay "
                       "failed after prefill.");
                ExpectFloatNear(
                    expected, ToFloatVector(marlinGraphOutput),
                    5e-2f, 5e-2f,
                    "CUDA INT4_GROUP grouped Marlin decode graph after "
                    "prefill");
            }
            FastllmCudaGraphExecDestroy(marlinGraphExec);
            FastllmCudaGraphDestroy(marlinGraph);
            FastllmCudaReleaseMergeMOEVllmMarlinCache(weights[2]);
        }

        if (!runSmallBatchChecks) {
            return;
        }

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

    void RunCudaInt4GroupBatch1MoeRegression() {
        // Retain the original extended small-batch coverage, then exercise the
        // Qwen3.6 local expert shape that selects the paired-output gate/up
        // and down kernels under TP=2.
        RunCudaInt4GroupBatch1MoeRegressionCase(2048, 256, 8, 8, false);
        RunCudaInt4GroupBatch1MoeRegressionCase(128, 32, 4, 2, true);
        RunCudaInt4GroupBatch1MoeRegressionCase(128, 256, 4, 2, false);
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

#ifdef USE_CUDA
    void RunNumasKimiHybridPrefillRegression() {
        constexpr int tokens = 1030;
        constexpr int topk = 2;
        constexpr int expertCount = 11;
        constexpr int inputDim = 32;
        constexpr int interDim = 32;
        constexpr float beta = 1.7f;
        constexpr float linearBeta = 2.5f;

        fastllm::Data input = MakeTensor(
            fastllm::DataType::BFLOAT16, {tokens, inputDim}, 0.4f);
        std::vector<int32_t> indices((size_t)tokens * topk);
        std::vector<float> scores((size_t)tokens * topk);
        for (int token = 0; token < tokens; token++) {
            indices[(size_t)token * topk] = 0;
            indices[(size_t)token * topk + 1] =
                1 + token % (expertCount - 1);
            scores[(size_t)token * topk] = 0.6f;
            scores[(size_t)token * topk + 1] = 0.4f;
        }
        fastllm::Data index = MakeIntTensor({tokens, topk}, indices);
        fastllm::Data score(
            fastllm::DataType::FLOAT32, {tokens, topk}, scores);

        std::vector<fastllm::Data> w1s, w2s, w3s;
        w1s.reserve(expertCount);
        w2s.reserve(expertCount);
        w3s.reserve(expertCount);
        for (int expert = 0; expert < expertCount; expert++) {
            w1s.push_back(MakeTensor(
                fastllm::DataType::BFLOAT16,
                {interDim, inputDim}, 1.0f + expert));
            w2s.push_back(MakeTensor(
                fastllm::DataType::BFLOAT16,
                {inputDim, interDim}, 4.0f + expert));
            w3s.push_back(MakeTensor(
                fastllm::DataType::BFLOAT16,
                {interDim, inputDim}, 7.0f + expert));
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
        input.ToDevice(fastllm::DataDevice::CUDA);
        index.ToDevice(fastllm::DataDevice::CUDA);
        score.ToDevice(fastllm::DataDevice::CUDA);

        fastllm::Data output;
        {
            ScopedFirstDevice guard("numa");
            fastllm::KimiK3RoutedExperts(
                input, index, score, w1Ptrs, w2Ptrs, w3Ptrs,
                beta, linearBeta, output);
        }
        Expect(output.dataDevice == fastllm::DataDevice::CUDA,
               "NUMA Kimi hybrid prefill did not return a CUDA tensor.");
        ExpectFloatNear(
            expected, ToFloatVector(output), 0.25f, 0.03f,
            "NUMA Kimi hybrid CUDA prefill output");
    }
#endif
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

    fastllm::Data MakeDeepSeekV4Fp8MoeWeight(
            int outputDim, int inputDim, float seed,
            const std::string &name) {
        constexpr int block = 128;
        Expect(outputDim % block == 0 && inputDim % block == 0,
               "DeepSeek-V4 FP8 NUMA regression requires block-aligned weights.");
        fastllm::Data weight(
            fastllm::DataType::FP8_E4M3, {outputDim, inputDim});
        weight.name = name;
        weight.blockK = block;
        weight.blockM = block;
        weight.scales.resize(
            (size_t)(outputDim / block) * (inputDim / block));
        for (size_t i = 0; i < weight.scales.size(); i++) {
            weight.scales[i] =
                0.0125f * (float)(1 + ((int)i + (int)(seed * 10)) % 5);
        }
        weight.Allocate(true);
        int seedValue = (int)std::lround(seed * 100.0f);
        for (size_t i = 0; i < (size_t)outputDim * inputDim; i++) {
            int exponent = 3 + (int)((i + seedValue) % 6);
            int mantissa = (int)((i * 7 + seedValue) & 7);
            uint8_t value = (uint8_t)((exponent << 3) | mantissa);
            if ((i + seedValue) & 1) {
                value |= 0x80;
            }
            weight.cpuData[i] = value;
        }
        return weight;
    }

    fastllm::Data MakeDeepSeekV4Nvfp4MoeWeight(
            int outputDim, int inputDim, int seed,
            const std::string &name) {
        constexpr int block = 32;
        Expect(outputDim > 0 && inputDim % block == 0,
               "DeepSeek-V4 NVFP4 NUMA regression requires block-aligned weights.");
        fastllm::Data weight(
            fastllm::DataType::NVFP4, {outputDim, inputDim});
        weight.name = name;
        weight.blockK = 1;
        weight.blockM = block;
        weight.Allocate(false);

        const size_t packedBytes = fastllm::GetNVFP4WeightBytes(
            outputDim, inputDim);
        for (size_t i = 0; i < packedBytes; i++) {
            const uint8_t low = (uint8_t)((i * 5 + seed) & 0xf);
            const uint8_t high =
                (uint8_t)((i * 11 + seed * 3 + 1) & 0xf);
            weight.cpuData[i] = low | (high << 4);
        }
        uint8_t *scales = fastllm::GetNVFP4ScaleData(weight);
        Expect(scales != nullptr,
               "DeepSeek-V4 NVFP4 NUMA regression scale storage is missing.");
        const size_t scaleBytes = fastllm::GetNVFP4ScaleBytes(
            outputDim, inputDim, weight.blockK, weight.blockM);
        for (size_t i = 0; i < scaleBytes; i++) {
            scales[i] = (uint8_t)(119 + ((i + seed) % 7));
        }
        return weight;
    }

#ifdef USE_NUMAS
    void RunNumasDeepSeekV4Nvfp4MoeBenchmark() {
        auto readPositiveEnv = [](const char *name, int fallback) {
            const char *value = std::getenv(name);
            int parsed = value == nullptr ? 0 : std::atoi(value);
            return parsed > 0 ? parsed : fallback;
        };
        const int batch = readPositiveEnv(
            "FASTLLM_DSV4_NUMAS_MOE_BENCH_BATCH", 8);
        const int threads = readPositiveEnv(
            "FASTLLM_DSV4_NUMAS_MOE_BENCH_THREADS", 24);
        const int iterations = readPositiveEnv(
            "FASTLLM_DSV4_NUMAS_MOE_BENCH_ITERATIONS", 40);
        constexpr int topk = 6;
        constexpr int expertCount = 24;
        constexpr int inputDim = 4096;
        constexpr int interDim = 2048;
        constexpr int outputDim = 4096;

        fastllm::SetThreads(threads);
        fastllm::Data input = MakeTensor(
            fastllm::DataType::BFLOAT16, {batch, inputDim}, 0.73f);
        std::vector<int32_t> indices((size_t)batch * topk);
        std::vector<float> scores((size_t)batch * topk);
        for (int row = 0; row < batch; row++) {
            for (int slot = 0; slot < topk; slot++) {
                indices[(size_t)row * topk + slot] =
                    (row * 3 + slot * 4) % expertCount;
                scores[(size_t)row * topk + slot] =
                    0.125f + 0.03125f * (float)slot;
            }
        }
        fastllm::Data index = MakeIntTensor({batch, topk}, indices);
        fastllm::Data score(
            fastllm::DataType::FLOAT32, {batch, topk}, scores);

        std::vector<fastllm::Data> gateWeights;
        std::vector<fastllm::Data> downWeights;
        gateWeights.reserve(expertCount);
        downWeights.reserve(expertCount);
        auto initializeWeight = [](fastllm::Data &weight,
                                   int outputChannels, int inputChannels,
                                   int seed, const std::string &name) {
            weight.name = name;
            weight.blockK = 1;
            weight.blockM = 32;
            weight.Allocate(false);
            const size_t packedBytes = fastllm::GetNVFP4WeightBytes(
                outputChannels, inputChannels);
            for (size_t i = 0; i < packedBytes; i++) {
                const uint8_t low = (uint8_t)((i * 5 + seed) & 0xf);
                const uint8_t high =
                    (uint8_t)((i * 11 + seed * 3 + 1) & 0xf);
                weight.cpuData[i] = low | (high << 4);
            }
            uint8_t *scaleData = fastllm::GetNVFP4ScaleData(weight);
            Expect(scaleData != nullptr,
                   "NUMA MoE benchmark NVFP4 scale storage is missing.");
            const size_t scaleBytes = fastllm::GetNVFP4ScaleBytes(
                outputChannels, inputChannels,
                weight.blockK, weight.blockM);
            for (size_t i = 0; i < scaleBytes; i++) {
                scaleData[i] = (uint8_t)(120 + (((i + seed) % 3) != 0));
            }
        };
        for (int expert = 0; expert < expertCount; expert++) {
            gateWeights.emplace_back(
                fastllm::DataType::NVFP4,
                std::vector<int>{interDim * 2, inputDim});
            initializeWeight(
                gateWeights.back(), interDim * 2, inputDim,
                101 + expert * 7,
                "bench.dsv4_nvfp4_gate." + std::to_string(expert));
            downWeights.emplace_back(
                fastllm::DataType::NVFP4,
                std::vector<int>{outputDim, interDim});
            initializeWeight(
                downWeights.back(), outputDim, interDim,
                211 + expert * 11,
                "bench.dsv4_nvfp4_down." + std::to_string(expert));
        }
        std::vector<fastllm::Data*> weights(
            (expertCount + 1) * 2, nullptr);
        for (int expert = 0; expert < expertCount; expert++) {
            weights[(expert + 1) * 2] = &gateWeights[expert];
            weights[(expert + 1) * 2 + 1] = &downWeights[expert];
        }
        std::vector<fastllm::Data*> biass(weights.size(), nullptr);
        fastllm::Data output(
            fastllm::DataType::BFLOAT16, {batch, outputDim});
        fastllm::Data w1, w2, w3, curInput, curOutput;
        auto run = [&]() {
            fastllm::MergeMOE(
                input, index, score, weights, biass,
                w1, w2, w3, curInput, curOutput,
                0.0f, output, 0, fastllm::MoeGateSwiglu,
                false, 7.0f, true);
        };

        std::vector<double> samples;
        samples.reserve(iterations);
        {
            ScopedFirstDevice guard("numa");
            for (int warmup = 0; warmup < 4; warmup++) {
                run();
            }
            for (int iteration = 0; iteration < iterations; iteration++) {
                const auto begin = std::chrono::steady_clock::now();
                run();
                samples.push_back(
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin).count());
            }
        }
        uint64_t hash = 1469598103934665603ull;
        const uint16_t *outputBits = (const uint16_t*)output.cpuData;
        for (size_t i = 0; i < (size_t)batch * outputDim; i++) {
            hash ^= outputBits[i];
            hash *= 1099511628211ull;
        }
        std::sort(samples.begin(), samples.end());
        const double mean = std::accumulate(
            samples.begin(), samples.end(), 0.0) / samples.size();
        std::cout << "threads=" << threads << " rows=" << batch << " routes="
                  << batch * topk << " unique_experts=" << expertCount
                  << " iterations=" << iterations
                  << " min_ms=" << samples.front()
                  << " median_ms=" << samples[samples.size() / 2]
                  << " mean_ms=" << mean
                  << " hash=" << std::hex << hash << std::dec << "\n";
    }
#endif

    void RunNumasLinearRegression() {
        constexpr int prefillTokens = 37;
        constexpr int inputDim = 256;
        constexpr int outputDim = 256;

        fastllm::Data input = MakeTensor(
            fastllm::DataType::BFLOAT16,
            {prefillTokens, inputDim}, 0.37f);
        fastllm::Data cpuWeight = MakeDeepSeekV4Fp8MoeWeight(
            outputDim, inputDim, 1.9f, "test.cpu_linear_fp8");
        fastllm::Data numaWeight = MakeDeepSeekV4Fp8MoeWeight(
            outputDim, inputDim, 1.9f, "test.numa_linear_fp8");
        fastllm::Data emptyBias;
        fastllm::Data expected, actual;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(input, cpuWeight, emptyBias, expected);
        }
        {
            ScopedFirstDevice guard("numa");
            fastllm::Linear(input, numaWeight, emptyBias, actual);
        }
        Expect(expected.dataType == fastllm::DataType::BFLOAT16 &&
                   actual.dataType == fastllm::DataType::BFLOAT16,
               "NUMA FP8 Linear output dtype mismatch.");
        Expect(expected.GetBytes() == actual.GetBytes() &&
                   memcmp(expected.cpuData, actual.cpuData,
                          expected.GetBytes()) == 0,
               "NUMA FP8 Linear prefill differs bitwise from CPU Linear.");
        Expect(!numaWeight.numasData.empty() &&
                   numaWeight.cpuData == nullptr,
               "NUMA FP8 Linear weight was not moved to node-local shards.");

        // A registered prefill weight is reused by subsequent one-token
        // decode calls, so cover the packed small-batch kernel as well.
        fastllm::Data decodeInput = MakeTensor(
            fastllm::DataType::BFLOAT16, {1, inputDim}, 0.53f);
        fastllm::Data decodeExpected, decodeActual;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(
                decodeInput, cpuWeight, emptyBias, decodeExpected);
        }
        {
            ScopedFirstDevice guard("numa");
            fastllm::Linear(
                decodeInput, numaWeight, emptyBias, decodeActual);
        }
        Expect(decodeExpected.GetBytes() == decodeActual.GetBytes() &&
                   memcmp(decodeExpected.cpuData, decodeActual.cpuData,
                          decodeExpected.GetBytes()) == 0,
               "NUMA FP8 Linear decode differs bitwise from CPU Linear.");

        // Compressor, router and LM-head projections enter Linear as FLOAT32
        // activations with BF16 weights.  The CPU path first rounds the input
        // to BF16, then accumulates to FP32; NUMA must preserve that boundary.
        fastllm::Data floatInput = MakeTensor(
            fastllm::DataType::FLOAT32,
            {prefillTokens, inputDim}, 0.71f);
        fastllm::Data cpuBfloatWeight = MakeTensor(
            fastllm::DataType::BFLOAT16,
            {outputDim, inputDim}, 0.83f);
        fastllm::Data numaBfloatWeight = MakeTensor(
            fastllm::DataType::BFLOAT16,
            {outputDim, inputDim}, 0.83f);
        fastllm::Data floatExpected, floatActual;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(
                floatInput, cpuBfloatWeight, emptyBias, floatExpected);
        }
        {
            ScopedFirstDevice guard("numa");
            fastllm::Linear(
                floatInput, numaBfloatWeight, emptyBias, floatActual);
        }
        Expect(floatExpected.dataType == fastllm::DataType::FLOAT32 &&
                   floatActual.dataType == fastllm::DataType::FLOAT32 &&
                   floatExpected.GetBytes() == floatActual.GetBytes() &&
                   memcmp(floatExpected.cpuData, floatActual.cpuData,
                          floatExpected.GetBytes()) == 0,
               "NUMA FLOAT32 x BF16 Linear differs bitwise from CPU Linear.");

        // DeepSeek-V4 keeps compressor and router projections as FP16.  Their
        // FLOAT32-input path must retain the original FP32 x FP16 accumulation
        // order (including bias), while BF16 input is expanded to FP32 first.
        fastllm::Data cpuHalfWeight = MakeTensor(
            fastllm::DataType::FLOAT16,
            {outputDim, inputDim}, 0.91f);
        fastllm::Data numaHalfWeight = MakeTensor(
            fastllm::DataType::FLOAT16,
            {outputDim, inputDim}, 0.91f);
        fastllm::Data cpuBias = MakeTensor(
            fastllm::DataType::FLOAT32, {outputDim}, 0.11f);
        fastllm::Data numaBias = MakeTensor(
            fastllm::DataType::FLOAT32, {outputDim}, 0.11f);
        fastllm::Data halfFloatExpected, halfFloatActual;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(
                floatInput, cpuHalfWeight, cpuBias, halfFloatExpected);
        }
        {
            ScopedFirstDevice guard("numa");
            fastllm::Linear(
                floatInput, numaHalfWeight, numaBias, halfFloatActual);
        }
        Expect(halfFloatExpected.GetBytes() == halfFloatActual.GetBytes() &&
                   memcmp(halfFloatExpected.cpuData, halfFloatActual.cpuData,
                          halfFloatExpected.GetBytes()) == 0,
               "NUMA FLOAT32 x FLOAT16 Linear differs bitwise from CPU "
               "Linear.");

        fastllm::Data halfBfloatExpected, halfBfloatActual;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(
                input, cpuHalfWeight, cpuBias, halfBfloatExpected);
        }
        {
            ScopedFirstDevice guard("numa");
            fastllm::Linear(
                input, numaHalfWeight, numaBias, halfBfloatActual);
        }
        Expect(halfBfloatExpected.GetBytes() == halfBfloatActual.GetBytes() &&
                   memcmp(halfBfloatExpected.cpuData, halfBfloatActual.cpuData,
                          halfBfloatExpected.GetBytes()) == 0,
               "NUMA BFLOAT16 x FLOAT16 Linear differs bitwise from CPU "
               "Linear.");

        // The one-token path expands BF16 directly in the AVX512 dot-product
        // kernel instead of materializing an FP32 input row.  Its FMA and
        // reduction order must remain identical to the CPU reference.
        fastllm::Data halfDecodeExpected, halfDecodeActual;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(
                decodeInput, cpuHalfWeight, cpuBias,
                halfDecodeExpected);
        }
        {
            ScopedFirstDevice guard("numa");
            fastllm::Linear(
                decodeInput, numaHalfWeight, numaBias,
                halfDecodeActual);
        }
        Expect(halfDecodeExpected.GetBytes() ==
                   halfDecodeActual.GetBytes() &&
                   memcmp(halfDecodeExpected.cpuData,
                          halfDecodeActual.cpuData,
                          halfDecodeExpected.GetBytes()) == 0,
               "NUMA BFLOAT16 x FLOAT16 decode differs bitwise from CPU "
               "Linear.");

        // Also cover native FP32 weights and their bias accumulation order.
        fastllm::Data cpuFloatWeight = MakeTensor(
            fastllm::DataType::FLOAT32,
            {outputDim, inputDim}, 0.97f);
        fastllm::Data numaFloatWeight = MakeTensor(
            fastllm::DataType::FLOAT32,
            {outputDim, inputDim}, 0.97f);
        fastllm::Data nativeExpected, nativeActual;
        {
            ScopedFirstDevice guard("cpu");
            fastllm::Linear(
                floatInput, cpuFloatWeight, cpuBias, nativeExpected);
        }
        {
            ScopedFirstDevice guard("numa");
            fastllm::Linear(
                floatInput, numaFloatWeight, numaBias, nativeActual);
        }
        Expect(nativeExpected.GetBytes() == nativeActual.GetBytes() &&
                   memcmp(nativeExpected.cpuData, nativeActual.cpuData,
                          nativeExpected.GetBytes()) == 0,
               "NUMA FLOAT32 x FLOAT32 Linear differs bitwise from CPU "
               "Linear.");
    }

    std::vector<uint16_t> RunNumasDeepSeekV4LargeMoeCase(
            MoeWeights &weights, int batch,
            fastllm::DataDevice *mergeOutputDevice = nullptr,
            bool keepCudaInputMirror = false) {
        const int inputDim = weights.routedGate.dims[1];
        const int outputDim = weights.routedDown.dims[0];
        fastllm::Data input = MakeTensor(
            fastllm::DataType::BFLOAT16, {batch, inputDim}, 0.73f);
#ifdef USE_CUDA
        if (keepCudaInputMirror) {
            input.ToDevice(
                fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
            input.ToDevice(
                fastllm::DataDevice::CPU, std::vector<int>{0}, true);
            Expect(input.cudaData != nullptr &&
                       input.dataDevice == fastllm::DataDevice::CPU,
                   "failed to retain the DeepSeek-V4 mixed-inference CUDA mirror.");
        }
#else
        Expect(!keepCudaInputMirror,
               "CUDA input mirror requested in a non-CUDA build.");
#endif
        fastllm::Data index = MakeIntTensor(
            {batch, 1}, std::vector<int32_t>(batch, 0));
        std::vector<float> routeScores(batch);
        for (int i = 0; i < batch; i++) {
            routeScores[i] = 0.25f + (float)(i % 7) * 0.0625f;
        }
        fastllm::Data score(
            fastllm::DataType::FLOAT32, {batch, 1}, routeScores);
        fastllm::Data output(
            fastllm::DataType::BFLOAT16, {batch, outputDim});
        fastllm::Data w1, w2, w3, curInput, curOutput;
        std::vector<fastllm::Data*> weightPtrs = {
            nullptr, nullptr, &weights.routedGate, &weights.routedDown
        };
        std::vector<fastllm::Data*> biasPtrs(4, nullptr);
        {
            ScopedFirstDevice guard("numa");
            fastllm::MergeMOE(
                input, index, score, weightPtrs, biasPtrs,
                w1, w2, w3, curInput, curOutput,
                0.0f, output, 0, fastllm::MoeGateSwiglu,
                false, 7.0f, true);
        }
        Expect(output.dataType == fastllm::DataType::BFLOAT16,
               "DeepSeek-V4 NUMA MergeMOE output dtype mismatch.");
#ifdef USE_CUDA
        if (mergeOutputDevice != nullptr) {
            *mergeOutputDevice = output.dataDevice;
        }
        if (output.dataDevice == fastllm::DataDevice::CUDA) {
            output.ToDevice(
                fastllm::DataDevice::CPU, output.dataDeviceIds, true);
        }
#else
        if (mergeOutputDevice != nullptr) {
            *mergeOutputDevice = output.dataDevice;
        }
#endif
        const uint16_t *data = (const uint16_t*)output.cpuData;
        return std::vector<uint16_t>(
            data, data + (size_t)batch * outputDim);
    }

    void RunNumasDeepSeekV4LargeMoeRegression() {
        constexpr int batch = 32;
        constexpr int inputDim = 128;
        constexpr int interDim = 256;
        constexpr int outputDim = 128;
        auto makeWeights = []() {
            return MoeWeights {
                MakeDeepSeekV4Fp8MoeWeight(
                    interDim * 2, inputDim, 1.3f,
                    "test.dsv4_fp8_routed_gate"),
                MakeDeepSeekV4Fp8MoeWeight(
                    outputDim, interDim, 2.1f,
                    "test.dsv4_fp8_routed_down")
            };
        };
        auto makeNvfp4Weights = []() {
            return MoeWeights {
                MakeDeepSeekV4Nvfp4MoeWeight(
                    interDim * 2, inputDim, 13,
                    "test.dsv4_nvfp4_routed_gate"),
                MakeDeepSeekV4Nvfp4MoeWeight(
                    outputDim, interDim, 29,
                    "test.dsv4_nvfp4_routed_down")
            };
        };

        // DSpark verification groups repeated expert routes into 2-8 row
        // NVFP4 GEMMs.  Compare every fused row count against the established
        // row-at-a-time path before exercising the large prefill path below.
        const char *savedGroupedDecodeDisable = std::getenv(
            "FASTLLM_DSV4_DISABLE_NUMAS_MOE_GROUPED_DECODE");
        const bool hadGroupedDecodeDisable =
            savedGroupedDecodeDisable != nullptr;
        const std::string savedGroupedDecodeDisableValue =
            hadGroupedDecodeDisable ? savedGroupedDecodeDisable : "";
        MoeWeights groupedDecodeWeights = makeNvfp4Weights();
        for (int smallBatch = 2; smallBatch <= 8; smallBatch++) {
            setenv(
                "FASTLLM_DSV4_DISABLE_NUMAS_MOE_GROUPED_DECODE", "1", 1);
            std::vector<uint16_t> rowMajor =
                RunNumasDeepSeekV4LargeMoeCase(
                    groupedDecodeWeights, smallBatch);
            unsetenv("FASTLLM_DSV4_DISABLE_NUMAS_MOE_GROUPED_DECODE");
            std::vector<uint16_t> expertMajor =
                RunNumasDeepSeekV4LargeMoeCase(
                    groupedDecodeWeights, smallBatch);
            Expect(rowMajor == expertMajor,
                   "DeepSeek-V4 NUMA grouped decode changed BF16 output "
                   "bits at batch " + std::to_string(smallBatch) + ".");
        }
        if (hadGroupedDecodeDisable) {
            setenv("FASTLLM_DSV4_DISABLE_NUMAS_MOE_GROUPED_DECODE",
                   savedGroupedDecodeDisableValue.c_str(), 1);
        } else {
            unsetenv("FASTLLM_DSV4_DISABLE_NUMAS_MOE_GROUPED_DECODE");
        }

        MoeWeights referenceWeights = makeWeights();
        setenv("FASTLLM_DSV4_DISABLE_NUMAS_MOE_LARGE_FAST", "1", 1);
        std::vector<uint16_t> expected =
            RunNumasDeepSeekV4LargeMoeCase(referenceWeights, batch);
        unsetenv("FASTLLM_DSV4_DISABLE_NUMAS_MOE_LARGE_FAST");

        MoeWeights optimizedWeights = makeWeights();
        std::vector<uint16_t> actual =
            RunNumasDeepSeekV4LargeMoeCase(optimizedWeights, batch);
        Expect(expected == actual,
               "DeepSeek-V4 NUMA large-batch MergeMOE fast path changed BF16 output bits.");

#ifdef USE_CUDA
        if (FastllmCudaGetDeviceCount() > 0) {
            std::vector<int> savedDevices;
            std::map<int, int> savedRatios;
            FastllmGetMulticudaDeviceAndRatio(
                savedDevices, savedRatios, false);
            FastllmMultiCudaSetDevice({0});

            MoeWeights cpuOnlyWeights = makeWeights();
            fastllm::DataDevice mergeOutputDevice =
                fastllm::DataDevice::CUDA;
            std::vector<uint16_t> cpuOnlyOutput =
                RunNumasDeepSeekV4LargeMoeCase(
                    cpuOnlyWeights, 128, &mergeOutputDevice);
            Expect(mergeOutputDevice == fastllm::DataDevice::CPU,
                   "CPU-only DeepSeek-V4 NUMA input was staged to a stale "
                   "MultiCUDA device.");

            MoeWeights mixedWeights = makeWeights();
            fastllm::DataDevice mixedOutputDevice =
                fastllm::DataDevice::CPU;
            std::vector<uint16_t> mixedOutput =
                RunNumasDeepSeekV4LargeMoeCase(
                    mixedWeights, 128, &mixedOutputDevice, true);
            Expect(mixedOutputDevice == fastllm::DataDevice::CUDA,
                   "DeepSeek-V4 mixed input with a valid CUDA mirror did not "
                   "use NUMA GPU-prefill.");
            std::vector<float> cpuOnlyFloat(cpuOnlyOutput.size());
            std::vector<float> mixedFloat(mixedOutput.size());
            for (size_t i = 0; i < cpuOnlyOutput.size(); i++) {
                cpuOnlyFloat[i] = fastllm::BFloat16BitsToFloat32(
                    cpuOnlyOutput[i]);
                mixedFloat[i] = fastllm::BFloat16BitsToFloat32(
                    mixedOutput[i]);
            }
            ExpectFloatNear(
                cpuOnlyFloat, mixedFloat, 2.0f, 0.05f,
                "DeepSeek-V4 NUMA mixed CPU/GPU MergeMOE output");

            FastllmMultiCudaSetDevice(savedDevices);
        }
#endif
    }

#ifdef USE_CUDA
    void RunDeepSeekV4CudaToCpuMirrorRegression() {
        if (FastllmCudaGetDeviceCount() < 1) {
            return;
        }
        std::vector<float> values = {
            0.25f, -1.5f, 3.0f, 0.0625f,
            -0.75f, 2.25f, 0.0f, -4.0f
        };
        fastllm::Data source(
            fastllm::DataType::BFLOAT16, {2, 4});
        source.Allocate();
        uint16_t *sourceBits = (uint16_t*)source.cpuData;
        std::vector<uint16_t> expected(values.size());
        for (size_t i = 0; i < values.size(); i++) {
            expected[i] = fastllm::Float32ToBFloat16RNEBits(values[i]);
            sourceBits[i] = expected[i];
        }
        source.ToDevice(
            fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        void *sourceCudaData = source.cudaData;

        fastllm::Data destination;
        Expect(fastllm::DeepSeekV4CopyCudaTensorToCpuForTest(
                   source, destination),
               "DeepSeek-V4 mixed-inference CUDA-to-CPU copy failed.");
        Expect(destination.dataDevice == fastllm::DataDevice::CPU &&
                   destination.cpuData != nullptr,
               "DeepSeek-V4 mixed-inference copy did not produce CPU data.");
        Expect(destination.cudaData == sourceCudaData &&
                   destination.cudaDataBorrowed,
               "DeepSeek-V4 mixed-inference copy lost its borrowed CUDA mirror.");
        Expect(destination.dataDeviceIds == std::vector<int>{0},
               "DeepSeek-V4 mixed-inference copy recorded the wrong CUDA device.");
        const uint16_t *actual = (const uint16_t*)destination.cpuData;
        Expect(std::equal(expected.begin(), expected.end(), actual),
               "DeepSeek-V4 mixed-inference CUDA-to-CPU copy changed BF16 data.");

        constexpr int quantizedElements = 256;
        std::vector<uint16_t> rawQuantizedInput(quantizedElements);
        std::vector<uint16_t> quantizedExpected(quantizedElements);
        for (int i = 0; i < quantizedElements; i++) {
            float value = ((i * 37) % 257 - 128) * 0.03125f;
            rawQuantizedInput[i] =
                fastllm::Float32ToBFloat16RNEBits(value);
        }
        fastllm::RunCpuDeepSeekV4ActivationQuantization(
            rawQuantizedInput.data(), quantizedExpected.data(),
            quantizedElements);
        fastllm::Data quantizedActivation(
            fastllm::DataType::BFLOAT16, {2, 128});
        quantizedActivation.Allocate(false);
        memcpy(quantizedActivation.cpuData, quantizedExpected.data(),
               quantizedActivation.GetBytes());
        Expect(fastllm::DeepSeekV4AddQuantizedCudaReplicaForTest(
                   quantizedActivation, 0),
               "DeepSeek-V4 failed to add a quantized CUDA activation replica.");
        Expect(quantizedActivation.dataDevice == fastllm::DataDevice::CPU &&
                   quantizedActivation.cpuData != nullptr &&
                   quantizedActivation.cudaData != nullptr &&
                   !quantizedActivation.cudaDataBorrowed,
               "DeepSeek-V4 quantized activation is not CPU/CUDA dual-resident.");
        std::vector<uint16_t> quantizedRoundTrip(quantizedElements);
        FastllmCudaCopyFromDeviceToHost(
            quantizedRoundTrip.data(), quantizedActivation.cudaData,
            quantizedActivation.GetBytes());
        Expect(quantizedExpected == quantizedRoundTrip,
               "DeepSeek-V4 quantized CUDA activation replica changed BF16 bits.");

        constexpr int moeRows = 2;
        constexpr int moeIntermediate = 256;
        std::vector<uint16_t> gateUpBits(
            (size_t)moeRows * moeIntermediate * 2);
        for (int row = 0; row < moeRows; row++) {
            for (int d = 0; d < moeIntermediate; d++) {
                float gate =
                    (((d * 13 + row * 5) % 41) - 20) * 0.625f;
                float up =
                    (((d * 17 + row * 3) % 37) - 18) * 0.75f;
                gateUpBits[((size_t)row * moeIntermediate + d) * 2] =
                    fastllm::Float32ToBFloat16RNEBits(gate);
                gateUpBits[((size_t)row * moeIntermediate + d) * 2 + 1] =
                    fastllm::Float32ToBFloat16RNEBits(up);
            }
        }
        // This pair lands on opposite sides of a BF16 tie if route, SiLU and
        // up are reassociated.  It locks the official (SiLU * up) * route
        // boundary used by the CPU expert path.
        gateUpBits[0] = fastllm::Float32ToBFloat16RNEBits(-4.546875f);
        gateUpBits[1] = fastllm::Float32ToBFloat16RNEBits(-1.453125f);
        fastllm::Data gateUp(
            fastllm::DataType::BFLOAT16,
            {moeRows, moeIntermediate * 2});
        gateUp.Allocate(false);
        memcpy(gateUp.cpuData, gateUpBits.data(), gateUp.GetBytes());
        gateUp.ToDevice(
            fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        std::vector<float> routeScaleValues = {1.85298490524292f, 1.25f};
        fastllm::Data routeScales(
            fastllm::DataType::FLOAT32, {moeRows}, routeScaleValues);
        routeScales.ToDevice(
            fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        fastllm::Data cudaDownInput;
        Expect(FastllmCudaDeepSeekV4PrepareMoeDownInput(
                   gateUp, cudaDownInput,
                   (const float*)routeScales.cudaData, 10.0f, true),
               "DeepSeek-V4 CUDA MoE down-input preparation failed.");

        std::vector<uint16_t> preQuantized(
            (size_t)moeRows * moeIntermediate);
        for (int row = 0; row < moeRows; row++) {
            for (int d = 0; d < moeIntermediate; d++) {
                size_t gateUpOffset =
                    ((size_t)row * moeIntermediate + d) * 2;
                float gate = fastllm::BFloat16BitsToFloat32(
                    gateUpBits[gateUpOffset]);
                float up = fastllm::BFloat16BitsToFloat32(
                    gateUpBits[gateUpOffset + 1]);
                gate = std::min(gate, 10.0f);
                up = std::max(-10.0f, std::min(up, 10.0f));
                float h = (gate / (1.0f + std::exp(-gate))) * up;
                float value = routeScaleValues[row] * h;
                preQuantized[(size_t)row * moeIntermediate + d] =
                    fastllm::Float32ToBFloat16RNEBits(value);
            }
        }
        std::vector<uint16_t> expectedDownInput(preQuantized.size());
        fastllm::RunCpuDeepSeekV4ActivationQuantization(
            preQuantized.data(), expectedDownInput.data(),
            expectedDownInput.size());
        std::vector<uint16_t> actualDownInput(expectedDownInput.size());
        FastllmCudaCopyFromDeviceToHost(
            actualDownInput.data(), cudaDownInput.cudaData,
            cudaDownInput.GetBytes());
        Expect(expectedDownInput == actualDownInput,
               "DeepSeek-V4 CUDA MoE down input differs from the CPU "
               "clip/SwiGLU/route/UE8M0 reference.");
    }
#endif

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

int main(int argc, char **argv) {
    try {
#ifdef USE_CUDA
        if (argc == 2 &&
            std::string(argv[1]) == "--cuda-dsv4-compressed-kv") {
            Expect(FastllmCudaGetDeviceCount() > 0,
                   "DeepSeek-V4 compressed-KV regression requires CUDA.");
            RunCudaDeepSeekV4CompressedKvMaintenanceRegression();
            RunMultiCudaDeepSeekV4CompressedCacheAppendRegression();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--cuda-dsv4-woa") {
            Expect(FastllmCudaGetDeviceCount() > 0,
                   "DeepSeek-V4 WoA regression requires CUDA.");
            RunCudaDeepSeekV4TokenTiledWoARegression();
            return 0;
        }
        if (argc == 2 &&
            std::string(argv[1]) == "--bench-cuda-dsv4-woa") {
            Expect(FastllmCudaGetDeviceCount() > 0,
                   "DeepSeek-V4 WoA benchmark requires CUDA.");
            RunCudaDeepSeekV4TokenTiledWoABenchmark();
            return 0;
        }
#endif
        if (argc == 2 &&
            std::string(argv[1]) == "--cpu-dsv4-preprocess") {
            RunCpuDeepSeekV4PreprocessRegression();
            std::cout << "DeepSeek-V4 CPU preprocessing bitwise regressions: PASS\n";
            return 0;
        }
        if (argc == 2 &&
            std::string(argv[1]) == "--cpu-dsv4-scale-qratory") {
            RunCpuDeepSeekV4ScaleQRatoryRegression();
            std::cout << "DeepSeek-V4 CPU ScaleQRatory bitwise regression: PASS\n";
            return 0;
        }
        if (argc == 2 &&
            (std::string(argv[1]) == "--cpu-dsv4-woa" ||
             std::string(argv[1]) == "--numas-dsv4-woa")) {
            RunCpuDeepSeekV4WoARegression();
            std::cout << "DeepSeek-V4 CPU/NUMA WoA bitwise regression: PASS\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--cpu-dsv4-sparse") {
            RunCpuDeepSeekV4SparseAttentionRegression();
            RunCpuDeepSeekV4SparseDecodeCachedRegression();
            std::cout << "DeepSeek-V4 CPU sparse bitwise regressions: PASS\n";
            return 0;
        }
        if (argc == 2 &&
            std::string(argv[1]) == "--bench-cpu-dsv4-sparse") {
            RunCpuDeepSeekV4SparseDecodeCachedBenchmark();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--cpu-dsv4-hcpre") {
            RunCpuDeepSeekV4HcPreRegression();
            std::cout << "DeepSeek-V4 CPU HcPre bitwise regressions: PASS\n";
            return 0;
        }
        if (argc == 2 &&
            std::string(argv[1]) == "--bench-cpu-dsv4-hcpre") {
            RunCpuDeepSeekV4HcPreDecodeBenchmark();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--numas-linear") {
            Expect(fastllm::HasDeviceType("numa"),
                   "NUMA Linear regression requires a NUMA device.");
            RunNumasLinearRegression();
            std::cout << "NUMA Linear bitwise regression: PASS\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--numas-dsv4-moe") {
            Expect(fastllm::HasDeviceType("numa"),
                   "DeepSeek-V4 MoE regression requires a NUMA device.");
            RunNumasDeepSeekV4LargeMoeRegression();
            std::cout << "DeepSeek-V4 NUMA MoE bitwise regressions: PASS\n";
            return 0;
        }
        if (argc == 2 &&
            std::string(argv[1]) == "--bench-cpu-dsv4-nvfp4-block32") {
            RunCpuDeepSeekV4Nvfp4Block32Benchmark();
            return 0;
        }
#ifdef USE_NUMAS
        if (argc == 2 &&
            std::string(argv[1]) == "--bench-numas-dsv4-nvfp4-moe") {
            RunNumasDeepSeekV4Nvfp4MoeBenchmark();
            return 0;
        }
#endif
        const char *indexerBenchmark = std::getenv(
            "FASTLLM_DSV4_CPU_INDEXER_BENCH_LENGTH");
        if (indexerBenchmark != nullptr) {
            RunCpuDeepSeekV4IndexerBenchmark(std::atoi(indexerBenchmark));
            return 0;
        }
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

        RunDeepSeekV4DsparkPrefixSelectionRegression();
        std::cout << "DeepSeek-V4 DSpark prefix selection regression: PASS\n";

        if (fastllm::HasDeviceType("cpu")) {
            RunCpuDeepSeekV4HcPreRegression();
            std::cout << "DeepSeek-V4 CPU HcPre token-parallel regression: PASS\n";
            RunCpuDeepSeekV4WindowKVUpdateRegression();
            std::cout << "DeepSeek-V4 CPU window KV update regression: PASS\n";
            RunCpuDeepSeekV4SparseAttentionRegression();
            std::cout << "DeepSeek-V4 CPU sparse attention regression: PASS\n";
            RunCpuDeepSeekV4SparseDecodeCachedRegression();
            std::cout << "DeepSeek-V4 CPU cached sparse decode regression: PASS\n";
        }

#ifdef USE_CUDA
        if (fastllm::HasDeviceType("cuda")) {
            RunDeepSeekV4CudaToCpuMirrorRegression();
            std::cout << "DeepSeek-V4 mixed CUDA/CPU mirror regression: PASS\n";
        }
#endif

        if (fastllm::HasDeviceType("numa") &&
            !fastllm::GetFastllmEnv().activateNuma) {
            RunNumasDeepSeekV4LargeMoeRegression();
            std::cout << "DeepSeek-V4 NUMA large-batch MergeMOE regression: PASS\n";
        }

        RunYarnRopeEncodingRegression();
        std::cout << "direct YaRN RoPE cached-reference regression: PASS\n";

#ifdef USE_CUDA
        if (fastllm::HasDeviceType("cuda")) {
            RunCudaLocalExpertRangeMaskRegression();
            std::cout << "cuda local expert range mask regression: PASS\n";
            RunCudaGreedyTieBreakRegression();
            std::cout << "cuda greedy tie-break regression: PASS\n";
            RunCudaHandoffSamplingRegression();
            std::cout << "cuda handoff sampling regression: PASS\n";
        }
#endif

        if (fastllm::HasDeviceType("cpu")) {
            RunBFloat16Q8KConversionRegression();
            std::cout << "BF16 to Q8_K/Q8_K32 bytewise regression: PASS\n";
            RunCpuInt4GroupAwqLinearRegression();
            std::cout << "cpu AWQ-style INT4_GROUP linear regression: PASS\n";
            RunCpuPackedInt4Group32KernelRegression();
            std::cout << "cpu packed INT4_GROUP(32) kernel regression: PASS\n";
            RunCpuDeepSeekV4ScaleQRatoryRegression();
            std::cout << "DeepSeek-V4 CPU ScaleQRatory bitwise regression: PASS\n";
            RunCpuDeepSeekV4PreprocessRegression();
            std::cout << "DeepSeek-V4 CPU preprocessing bitwise regressions: PASS\n";
            RunCpuDeepSeekV4WoARegression();
            std::cout << "DeepSeek-V4 CPU WoA bitwise regression: PASS\n";
            ranAny = true;
        }

        if (fastllm::HasDeviceType("disk")) {
            RunDiskOperatorsRegression();
            std::cout << "disk Linear, Embedding and MergeMOE regression: PASS\n";
            ranAny = true;
        }

        if (fastllm::HasDeviceType("cuda")) {
#ifdef USE_CUDA
#ifndef USE_ROCM
            RunCudaNVFP4MarlinRegression();
            RunCudaBFloat16Hidden3072RMSNormRegression();
            std::cout << "cuda BF16 hidden-3072 RMSNorm regression: PASS\n";
#endif
            RunCudaBFloat16SigmoidMulToRegression();
            std::cout << "cuda BF16 Sigmoid/MulTo regression: PASS\n";
            Expect(FastllmCudaGraphQwen35MoeSelfTest(),
                   "Qwen3.5 CUDA graph shared/routed MoE parallelization/fallback self-test failed");
            RunCudaTritonChunkGdnPrefillRegression();
            RunCudaRaggedGdnLogicalPrepRegression();
            RunCudaVarlenChunkGdnPrefillRegression();
            RunCudaVarlenChunkGdnHeadMappingRegression();
            RunCudaDeepSeekV4TokenTiledWoARegression();
            RunCudaDeepSeekV4TritonWoARegression();
            RunCudaDeepSeekV4TritonSparseDecodeRegression();
            RunCudaDeepSeekV4IndexerRegression();
            RunCudaDeepSeekV4FullWindowAppendRegression();
            RunCudaPeerAccessInitRegression();
            RunMultiCudaDeepSeekV4SparsePrefillRegression();
            RunCudaDeepSeekV4HashRouteCacheRegression();
            RunCudaDeepSeekV4FusedRouterRegression();
            RunCudaDeepSeekV4FusedHcPreNormRegression();
            RunMultiCudaDeepSeekV4HcPrePrefillRegression();
            RunMultiCudaDeepSeekV4RouterLinearResizeRegression();
            RunMultiCudaDeepSeekV4RawCacheAppendRegression();
            RunCudaDeepSeekV4CompressedKvMaintenanceRegression();
            RunMultiCudaDeepSeekV4CompressedCacheAppendRegression();
            RunMultiCudaDeepSeekV4ExpandedSnapshotRegression();
            RunCudaDeepSeekV4FusedQKVRopeCacheRegression();
            RunCudaGraphMemoryPoolOwnershipRegression();
            RunCudaLinearDataTypeCapabilityRegression();
            RunCudaGgufMmvqBatch8Regression();
            RunCudaKimiK3PackedConvCacheRegression();
            RunCudaKimiK3RecurrentKdaRegression();
            RunCudaMergeMlaPagedChunkRegression();
            RunCudaInt4Group32AwqLinearRegression();
            RunCudaCompactInt4Group32LinearRegression();
            RunCudaInt4GroupHalfWeightRoundingRegression();
            RunCudaFp8LinearAddRegression();
            RunCudaFp16WarpRowsGemvRegression();
            RunCudaFusedRouterSelectionRegression();
            RunCudaQwen35RouterSharedGateFusionRegression();
            RunCudaQwen35FusedMoeJoinRegression();
            RunCudaInt4GroupBatch1MoeRegression();
            RunCudaInt8Batch1MoeRegression();
            RunCudaConvMultiTokenSnapshotsRegression();
            ranCrossDeviceViewRegression = RunCudaCrossDeviceViewRejectionRegression();
            RunMultiCudaReplicatedExpansionRegression();
#ifndef USE_ROCM
            RunMultiCudaDeepSeekV4AttentionSplitUnitRegression();
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
            RunNumasLinearRegression();
            std::cout << "numa Linear bitwise regression: PASS\n";
            RunNumaMoeWarmupRegistrationRegression();
            std::cout << "numa MoE full warmup registration regression: PASS\n";
            RunNumasMergeMoeRegression();
            std::cout << "numa MergeMOE regression: PASS\n";
            RunNumasInt4Group32MergeMoeRegression();
            std::cout << "numa INT4_GROUP(32) MergeMOE regression: PASS\n";
#ifdef USE_NUMAS
            RunNumasKimiRoutedExpertsFormatRegression();
            std::cout << "numa Kimi routed-expert multi-format regression: PASS\n";
#ifdef USE_CUDA
            RunNumasKimiHybridPrefillRegression();
            std::cout << "numa Kimi hybrid CUDA-prefill regression: PASS\n";
#endif
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
