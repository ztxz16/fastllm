#include "fastllm.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"
#include "blocks/baseblock.h"
#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char **environ;

using namespace fastllm;

namespace {
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int CheckTritonDependency() {
    const char *configured = std::getenv("FASTLLM_CUDA_TRITON_PYTHON");
    std::string python = configured != nullptr && configured[0] != '\0' ? configured : "";
    if (python.empty()) {
        // Match the compiler's interpreter selection for a native executable.
        for (const char *name : {"VIRTUAL_ENV", "CONDA_PREFIX"}) {
            const char *prefix = std::getenv(name);
            if (prefix == nullptr || prefix[0] == '\0') continue;
            std::string candidate = std::string(prefix) + "/bin/python";
            if (access(candidate.c_str(), X_OK) == 0) {
                python = candidate;
                break;
            }
        }
        if (python.empty()) python = "python3";
    }
    // Only an absent Python/Triton dependency is a skip. Broken installations,
    // compiler failures and numerical mismatches must still fail the test.
    const char *probe =
        "import importlib.util, sys\n"
        "if importlib.util.find_spec('triton') is None:\n"
        "    print('SKIP: Triton is not installed in ' + sys.executable)\n"
        "    sys.exit(77)\n"
        "import triton.language\n"
        "from triton.language.extra import libdevice\n"
        "from triton.compiler.compiler import ASTSource\n"
        "from triton.backends.compiler import GPUTarget\n";
    char *args[] = {const_cast<char *>(python.c_str()), const_cast<char *>("-c"),
                    const_cast<char *>(probe), nullptr};
    pid_t child;
    int error = posix_spawnp(&child, python.c_str(), nullptr, nullptr, args, environ);
    if (error == ENOENT) {
        std::printf("SKIP: Python interpreter is unavailable: %s\n", python.c_str());
        return 77;
    }
    Require(error == 0, "could not start Triton dependency check");
    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited == -1 && errno == EINTR);
    Require(waited == child && WIFEXITED(status), "Triton dependency check did not exit normally");
    int result = WEXITSTATUS(status);
    if (result == 0) {
        Require(setenv("FASTLLM_CUDA_TRITON_PYTHON", python.c_str(), 1) == 0,
                "could not select Triton compiler interpreter");
    }
    return result;
}

std::vector<uint16_t> Read(const Data &data) {
    std::vector<uint16_t> result(data.Count(0));
    Require(cudaMemcpy(result.data(), data.cudaData, result.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess, "copy failed");
    return result;
}

void RunMlpDispatchCase(int device) {
    constexpr int rows = 128, hidden = 128, inter = 256;
    Data input(FLOAT16, {1, rows, hidden}, std::vector<float>(rows * hidden, 0.125f));
    Data gate(FP8_E4M3, {2 * inter, hidden}), down(FP8_E4M3, {hidden, inter});
    for (Data *weight : {&gate, &down}) {
        weight->blockK = weight->blockM = 128;
        weight->scales.assign(weight->dims[0] / 128 * (weight->dims[1] / 128), 0.01f);
        weight->Allocate();
        for (uint64_t i = 0; i < weight->Count(0); ++i) {
            weight->cpuData[i] = uint8_t((i % 64) | ((i / 17 % 2) << 7));
        }
        weight->ToDevice(DataDevice::CUDA, {device}, true);
    }
    input.ToDevice(DataDevice::CUDA, {device}, true);
    static_cast<Executor *>(GetExecutor())->SetFirstDevice("cuda:" + std::to_string(device));
    Data gateShape(FLOAT16, {1, rows, 2 * inter});
    Data bias, gateResult, swigluResult, output(FLOAT16, {1, rows, hidden});
    output.dataDevice = DataDevice::CUDA;
    output.dataDeviceIds = {device};
    output.Allocate();
    auto run = [&]() {
        Require(cudaMemset(output.cudaData, 0, output.GetBytes()) == cudaSuccess, "zero failed");
        MLPBlock(&input, &gate, &down, &gateResult, &swigluResult, &output);
        return Read(output);
    };
    // CUTLASS knobs must not control Triton's generic MLP entry point.
    setenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_SWIGLU_QUANT", "0", 1);
    setenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_SWIGLU_QUANT_TP", "0", 1);
    setenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH", "4096", 1);
    Require(CanRunSwigluLinearAdd(gateShape, down, bias, output), "CUTLASS disabled Triton MLP");
    auto fused = run();
    setenv("FASTLLM_CUDA_TRITON_LINEAR_FP8_SWIGLU_QUANT", "0", 1);
    Require(!CanRunSwigluLinearAdd(gateShape, down, bias, output), "disabled backends accepted MLP");
    Require(run() == fused, "MLP fallback differs bitwise");
    setenv("FASTLLM_CUDA_TRITON_LINEAR_FP8_SWIGLU_QUANT", "1", 1);
    unsetenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_SWIGLU_QUANT");
    unsetenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_SWIGLU_QUANT_TP");
    unsetenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH");
    std::printf("device=%d MLP backend isolation/fallback bitwise PASS\n", device);
}

void RunCase(int device, DataType dtype, int rows, int inter, int hidden, bool withBias,
             bool packed = false) {
    std::mt19937 random(123 + rows + inter + hidden);
    std::vector<float> values((size_t)rows * inter * 2);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < inter; ++c) {
            // Include zero groups, underflow, exp overflow and ordinary values.
            values[(size_t)r * inter * 2 + c] = r == 0 ? 0.0f :
                (r == 1 ? -16.0f : (r == 2 ? 0.000001f :
                    float(int(random() % 65537) - 32768) / 2048.0f));
            values[(size_t)r * inter * 2 + inter + c] =
                float(int(random() % 65537) - 32768) / 16384.0f;
        }
    }
    Data input(dtype, {1, rows, inter * 2}, values);
    input.ToDevice(DataDevice::CUDA, {device}, true);
    Data weight(packed ? FP8_E4M3_BLOCK_128 : FP8_E4M3, {hidden, inter});
    weight.blockK = weight.blockM = 128;
    weight.scales.resize(((hidden + 127) / 128) * (inter / 128));
    for (auto &scale : weight.scales) scale = float(1 + random() % 100) * 0.00001f;
    weight.Allocate();
    for (uint64_t i = 0; i < weight.Count(0); ++i) {
        weight.cpuData[i] = uint8_t((random() % 120) | ((random() & 1) << 7));
    }
    if (packed) {
        for (int r = 0; r < hidden; ++r) {
            for (int c = 0; c < inter / 128; ++c) {
                auto *block = weight.cpuData + ((size_t)r * (inter / 128) + c) * 132;
                for (int i = 0; i < 128; ++i) {
                    block[i] = uint8_t((random() % 120) | ((random() & 1) << 7));
                }
                float scale = float(1 + random() % 100) * 0.00001f;
                std::memcpy(block + 128, &scale, sizeof(scale));
            }
        }
    }
    weight.ToDevice(DataDevice::CUDA, {device}, true);
    Data bias(FLOAT32);
    if (withBias) {
        bias.Resize({hidden});
        bias.Allocate();
        for (int i = 0; i < hidden; ++i) {
            reinterpret_cast<float *>(bias.cpuData)[i] = float(i % 17 - 8) / 16.0f;
        }
        bias.ToDevice(DataDevice::CUDA, {device}, true);
    }
    Data swiglu(dtype), reference(dtype), output(dtype);
    for (Data *data : {&swiglu, &reference, &output}) {
        data->dataDevice = DataDevice::CUDA;
        data->dataDeviceIds = {device};
    }
    DoCudaSwigluReshape(input, swiglu);
    DoCudaSwiglu(input, swiglu);
    DoCudaLinearReshape(swiglu, weight, reference);
    DoCudaLinear(swiglu, weight, bias, reference);
    output.Resize(reference.dims);
    output.Allocate();
    Require(DoCudaTritonSwigluLinear(input, weight, bias, output), "fusion was not used");
    Require(cudaDeviceSynchronize() == cudaSuccess, "kernel failed");
    Require(Read(output) == Read(reference), "fused result differs bitwise from SwiGLU + Linear");

    const auto expected = Read(output);
    for (const char *flag : {"FASTLLM_CUDA_TRITON", "FASTLLM_CUDA_TRITON_LINEAR_FP8",
                            "FASTLLM_CUDA_TRITON_LINEAR_FP8_SWIGLU_QUANT",
                            "FASTLLM_CUDA_TRITON_LINEAR_FP8_NATIVE_QUANT"}) {
        setenv(flag, "0", 1);
        Require(!DoCudaTritonSwigluLinear(input, weight, bias, output), "disabled fusion ran");
        Require(Read(output) == expected, "disabled fusion modified output");
        setenv(flag, "1", 1);
    }
    input.strides[0] += 128;
    Require(!DoCudaTritonSwigluLinear(input, weight, bias, output), "noncontiguous input accepted");
    input.strides[0] -= 128;
    output.strides[0] += 128;
    Require(!DoCudaTritonSwigluLinear(input, weight, bias, output), "noncontiguous output accepted");
    output.strides[0] -= 128;
    output.dims.back() += 1;
    Require(!DoCudaTritonSwigluLinear(input, weight, bias, output), "wrong output shape accepted");
    output.dims.back() -= 1;
    auto weightType = weight.dataType;
    weight.dataType = FLOAT16;
    Require(!DoCudaTritonSwigluLinear(input, weight, bias, output), "non-FP8 weight accepted");
    weight.dataType = weightType;
    input.dataDevice = DataDevice::CPU;
    Require(!DoCudaTritonSwigluLinear(input, weight, bias, output), "CPU input accepted");
    input.dataDevice = DataDevice::CUDA;
    setenv("FASTLLM_CUDA_TRITON_LINEAR_FP8_MIN_BATCH", "4096", 1);
    Require(!DoCudaTritonSwigluLinear(input, weight, bias, output), "min batch override ignored");
    unsetenv("FASTLLM_CUDA_TRITON_LINEAR_FP8_MIN_BATCH");
    setenv("FASTLLM_CUDA_TRITON_LINEAR_FP8_MAX_BATCH", "64", 1);
    Require(!CanUseCudaTritonSwigluLinear(rows, withBias, packed), "max batch probe ignored");
    Require(!DoCudaTritonSwigluLinear(input, weight, bias, output), "max batch override ignored");
    setenv("FASTLLM_CUDA_TRITON_LINEAR_FP8_MAX_BATCH", "0", 1);
    // Small decode batches must retain the original GEMV path.
    input.Resize({1, 4, inter * 2});
    output.Resize({1, 4, hidden});
    Require(!DoCudaTritonSwigluLinear(input, weight, bias, output), "decode dispatch changed");
    std::printf("device=%d dtype=%d M=%d K=%d N=%d bias=%d packed=%d bitwise/fallback PASS\n",
                device, (int)dtype, rows, inter, hidden, withBias, packed);
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    // An isolated cache can be supplied by the caller. Compilation is excluded
    // from any performance measurement; this test checks only correctness.
    setenv("FASTLLM_CUDA_TRITON", "1", 1);
    setenv("FASTLLM_CUDA_TRITON_LINEAR_FP8", "1", 1);
    setenv("FASTLLM_CUDA_TRITON_LINEAR_FP8_SWIGLU_QUANT", "1", 1);
    setenv("FASTLLM_CUDA_TRITON_LINEAR_FP8_NATIVE_QUANT", "1", 1);
    setenv("FASTLLM_CUDA_TRITON_LINEAR_FP8_MAX_BATCH", "0", 1);
    setenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8", "0", 1);
    try {
        bool tested = false;
        for (int device = 0; device < devices; ++device) {
            Require(cudaSetDevice(device) == cudaSuccess, "set device failed");
            if (!CanUseCudaTritonSwigluLinear()) continue;
            if (!tested) {
                int dependency = CheckTritonDependency();
                if (dependency != 0) return dependency;
            }
            tested = true;
            RunMlpDispatchCase(device);
            for (DataType dtype : {FLOAT16, BFLOAT16}) {
                for (auto shape : std::vector<std::vector<int>>{
                        {128, 128, 129}, {129, 384, 255}, {511, 1024, 768},
                        {2048, 8704, 5120}}) {
                    for (bool bias : {false, true}) {
                        RunCase(device, dtype, shape[0], shape[1], shape[2], bias);
                    }
                }
                for (bool bias : {false, true}) {
                    RunCase(device, dtype, 129, 384, 255, bias, true);
                }
            }
        }
        return tested ? 0 : 77;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
