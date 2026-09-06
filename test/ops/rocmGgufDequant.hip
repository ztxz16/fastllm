// Standalone ROCm regression linked against the built libfastllm_tools.so.
// Exercises the 256 MiB boundary, a short final chunk, prefill output strides,
// bias, and Graph replay with new inputs. No model files are required.
// From the repository root, after building with USE_ROCM=ON:
// hipcc -O2 -std=c++17 --offload-arch=gfx1100 -fgpu-default-stream=per-thread
//   -DUSE_ROCM -DUSE_CUDA -Iinclude -Iinclude/utils -Iinclude/models
//   -Iinclude/blocks -Iinclude/devices/cpu -Ithird_party/json11 -Ithird_party/gguf
//   test/ops/rocmGgufDequant.hip -Lbuild-rocm-dequant/tools/ftllm
//   -lfastllm_tools -Wl,-rpath,$PWD/build-rocm-dequant/tools/ftllm
//   -o rocmGgufDequant
// ./rocmGgufDequant
#include <hip/hip_runtime.h>
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr int Columns = 5120;
// Keep the row period different from both the capacity and aligned chunk
// boundaries, so accidentally restarting a chunk at row zero is detectable.
constexpr int Patterns = 19;
constexpr size_t Limit = 256ULL * 1024 * 1024;
void Hip(hipError_t s) {
    if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s));
}
void Check(bool b, const char *s) { if (!b) throw std::runtime_error(s); }
float Half(float x) { return fastllm::half_to_float(fastllm::float_to_half(x)); }

struct Weights {
    ggml_type type;
    size_t rowBytes;
    std::vector<uint8_t> packed;
    std::vector<float> reference;
    explicit Weights(ggml_type t) : type(t), rowBytes(ggml_row_size(t, Columns)),
            packed(Patterns * rowBytes, 0), reference(Patterns * Columns) {
        if (t == GGML_TYPE_Q6_K) {
            std::vector<float> values(reference.size());
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = std::sin(float(i * 13 + 7) * 0.03125f) * 0.25f;
            quantize_row_q6_K_ref(values.data(),
                    reinterpret_cast<block_q6_K *>(packed.data()), values.size());
            dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(packed.data()),
                    reference.data(), reference.size());
        } else {
            auto *blocks = reinterpret_cast<block_iq4_xs *>(packed.data());
            for (size_t b = 0; b < reference.size() / QK_K; ++b) {
                blocks[b].d = fastllm::float_to_half(0.00025f * (1 + b % 7));
                for (int j = 0; j < 8; ++j) {
                    const int scale = 1 + (b * 19 + j * 9) % 63;
                    blocks[b].scales_l[j / 2] |= (scale & 15) << (4 * (j % 2));
                    blocks[b].scales_h |= ((scale >> 4) & 3) << (2 * j);
                }
                for (int j = 0; j < QK_K / 2; ++j)
                    blocks[b].qs[j] = ((b * 13 + j * 5 + 2) & 15) |
                        (((b * 3 + j * 7 + 12) & 15) << 4);
            }
            dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(packed.data()),
                    reference.data(), reference.size());
        }
        for (float &v : reference) v = Half(v);
    }
};

void Run(const Weights &patterns, int rows, int batch, bool withBias, bool graph) {
    fastllm::Data weight(fastllm::DataType::DATA_GGUF_FORMAT,
            int(patterns.type), {rows, Columns});
    weight.name = "regression.rocm_gguf_dequant";
    weight.disableGGUFRepack = true;
    weight.forceGGUFFp32Dequant = true;
    weight.Allocate();
    for (int r = 0; r < rows; ++r)
        std::memcpy(weight.cpuData + size_t(r) * patterns.rowBytes,
                patterns.packed.data() + (r % Patterns) * patterns.rowBytes,
                patterns.rowBytes);
    weight.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
    std::vector<float> zeros(size_t(batch) * Columns, 0);
    fastllm::Data input(fastllm::DataType::FLOAT16, {batch, Columns}, zeros);
    input.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
    fastllm::Data output(fastllm::DataType::FLOAT16, {batch, rows});
    output.ToDevice(fastllm::DataDevice::CUDA, {0}, false);
    output.Allocate();
    std::vector<float> biasValues(rows, 0);
    for (int r = 0; r < rows; ++r) biasValues[r] = Half((r % 11 - 5) / 64.0f);
    fastllm::Data bias;
    if (withBias) {
        fastllm::Data source(fastllm::DataType::FLOAT32, {rows}, biasValues);
        bias.CopyFrom(source);
        bias.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
    }
    auto run = [&] {
        Check(FastllmCudaHalfMatMulGGUF(input, weight, bias, output,
                batch, Columns, rows), "GGUF GEMM failed");
    };
    run();
    Hip(hipDeviceSynchronize());
    hipGraph_t captured = nullptr;
    hipGraphExec_t executable = nullptr;
    if (graph) {
        Hip(hipStreamBeginCapture(hipStreamPerThread, hipStreamCaptureModeThreadLocal));
        run();
        Hip(hipStreamEndCapture(hipStreamPerThread, &captured));
        Hip(hipGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    }
    double maxError = 0;
    for (int repeat = 0; repeat < (graph ? 3 : 1); ++repeat) {
        std::vector<uint16_t> values(zeros.size(), 0);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < 8; ++j) {
                const int column = (b * 71 + j * 613 + repeat * 197) % Columns;
                values[size_t(b) * Columns + column] =
                    fastllm::float_to_half((j % 3 - 1) / 4.0f);
            }
        Hip(hipMemcpy(input.cudaData, values.data(), values.size() * 2, hipMemcpyHostToDevice));
        if (graph) Hip(hipGraphLaunch(executable, hipStreamPerThread));
        else run();
        Hip(hipDeviceSynchronize());
        std::vector<uint16_t> actual(size_t(batch) * rows);
        Hip(hipMemcpy(actual.data(), output.cudaData, actual.size() * 2, hipMemcpyDeviceToHost));
        // CPU scalar reference uses dequantized pattern rows and only the
        // nonzero input columns; it does not reuse GPU indexing or GEMM.
        for (int b = 0; b < batch; ++b) {
            for (int r = 0; r < rows; ++r) {
                float sum = 0;
                for (int j = 0; j < 8; ++j) {
                    const int column = (b * 71 + j * 613 + repeat * 197) % Columns;
                    sum += patterns.reference[(r % Patterns) * Columns + column] *
                        ((j % 3 - 1) / 4.0f);
                }
                const float expected = Half(Half(sum) + (withBias ? biasValues[r] : 0));
                const float got = fastllm::half_to_float(actual[size_t(b) * rows + r]);
                const double error = std::fabs(got - expected);
                maxError = std::max(maxError, error);
                Check(std::isfinite(got) && error <= 0.002f + 0.002f * std::fabs(expected),
                        "CPU reference mismatch");
            }
        }
    }
    if (graph) {
        Hip(hipGraphExecDestroy(executable));
        Hip(hipGraphDestroy(captured));
    }
    size_t scratchBytes = 0;
    void *scratch = FastllmCudaGetFlashInferFloatWorkspace(&scratchBytes);
    Check(scratch && scratchBytes == Limit, "unexpected shared dequant workspace size");
    std::cout << "PASS type=" << ggml_type_name(patterns.type)
              << " rows=" << rows << " batch=" << batch << " bias=" << withBias
              << " graph=" << graph << " max_error=" << maxError
              << " scratch_bytes=" << scratchBytes << std::endl;
}
}

int main() {
    try {
        Hip(hipSetDevice(0));
        const int boundary = Limit / (Columns * 2);
        int cases = 0;
        for (ggml_type type : {GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS}) {
            Weights patterns(type);
            for (int rows : {37, boundary, boundary + 5}) {
                for (int batch : {1, 3, 64}) {
                    Run(patterns, rows, batch, batch != 1, false);
                    ++cases;
                }
            }
            // Exact multiple of the shared-workspace chunks (no short final chunk).
            Run(patterns, 2 * boundary, 3, true, false);
            ++cases;
            Run(patterns, boundary + 5, 3, true, true);
            ++cases;
        }
        std::cout << "PASS cases=" << cases << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    } catch (const char *e) {
        std::cerr << "FAIL: " << e << std::endl;
        return 1;
    }
}
