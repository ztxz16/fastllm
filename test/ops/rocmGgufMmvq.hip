// Verify ROCm GGUF matvec against independent CPU dequantization and dot products.
// Fixtures contain unmodified rows from the model, including odd output sizes.
// Generate fixtures with: python3 test/ops/rocmGgufMmvqFixtures.py MODEL.gguf FIXTURES
// Build against a USE_ROCM library, with the same includes as rocmGgufDequant.hip.
// Run: ./rocmGgufMmvq FIXTURES
#include <hip/hip_runtime.h>
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Hip(hipError_t s) {
    if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s));
}
static float Half(float f) { return fastllm::half_to_float(fastllm::float_to_half(f)); }

static void Run(const std::string &file, int type, int m, int k, int n, bool graph) {
    const size_t bytes = ggml_row_size(ggml_type(type), m) * k;
    std::vector<char> packed(bytes);
    std::ifstream in(file, std::ios::binary);
    in.read(packed.data(), bytes);
    if (size_t(in.gcount()) != bytes) throw std::runtime_error("fixture read failed");
    std::vector<float> w(size_t(k) * m);
    switch (type) {
        case GGML_TYPE_IQ2_XS: dequantize_row_iq2_xs(reinterpret_cast<const block_iq2_xs *>(packed.data()), w.data(), w.size()); break;
        case GGML_TYPE_IQ2_S: dequantize_row_iq2_s(reinterpret_cast<const block_iq2_s *>(packed.data()), w.data(), w.size()); break;
        case GGML_TYPE_IQ3_XXS: dequantize_row_iq3_xxs(reinterpret_cast<const block_iq3_xxs *>(packed.data()), w.data(), w.size()); break;
        case GGML_TYPE_IQ3_S: dequantize_row_iq3_s(reinterpret_cast<const block_iq3_s *>(packed.data()), w.data(), w.size()); break;
        case GGML_TYPE_IQ4_NL: dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(packed.data()), w.data(), w.size()); break;
        case GGML_TYPE_IQ4_XS: dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(packed.data()), w.data(), w.size()); break;
        default: {
            auto dequant = ggml_type_to_float(ggml_type(type));
            if (!dequant) throw std::runtime_error("no CPU reference for type");
            dequant(packed.data(), w.data(), w.size());
        }
    }
    fastllm::Data weight(fastllm::DataType::DATA_GGUF_FORMAT, type, {k, m});
    weight.disableGGUFRepack = true;
    weight.Allocate();
    std::memcpy(weight.cpuData, packed.data(), bytes);
    weight.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
    std::vector<float> zero(size_t(n)*m, 0), biasValues(k);
    for (int r = 0; r < k; ++r) biasValues[r] = Half((r % 11 - 5) * .01f);
    fastllm::Data input(fastllm::DataType::FLOAT16, {n, m}, zero);
    input.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
    fastllm::Data bias(fastllm::DataType::FLOAT32, {k}, biasValues);
    bias.ToDevice(fastllm::DataDevice::CUDA, {0}, true);
    fastllm::Data output(fastllm::DataType::FLOAT16, {n, k});
    output.ToDevice(fastllm::DataDevice::CUDA, {0}, false);
    output.Allocate();
    auto run = [&] { FastllmCudaHalfMatMulGGUF(input, weight, bias, output, n, m, k); };
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
    double maxRelativeRms = 0, maxQuantizedRms = 0;
    for (int repeat = 0; repeat < 3; ++repeat) {
        std::vector<uint16_t> h(zero.size());
        std::vector<float> x(zero.size()), xq(zero.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = Half(repeat == 0 ? 0 : std::sin((i * 73 + repeat * 197) * .037f) * .5f);
            h[i] = fastllm::float_to_half(x[i]);
        }
        for (size_t i = 0; i < x.size(); i += 32) {
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[i+j]));
            const float d = amax / 127.0f;
            for (int j = 0; j < 32; ++j) xq[i+j] = amax ? std::round(x[i+j]/d) * Half(d) : 0;
        }
        Hip(hipMemcpy(input.cudaData, h.data(), h.size()*2, hipMemcpyHostToDevice));
        if (graph) Hip(hipGraphLaunch(executable, hipStreamPerThread));
        else run();
        Hip(hipDeviceSynchronize());
        std::vector<uint16_t> out(size_t(n)*k);
        Hip(hipMemcpy(out.data(), output.cudaData, out.size()*2, hipMemcpyDeviceToHost));
        double error = 0, quantizedError = 0, norm = 0;
        for (int b = 0; b < n; ++b) for (int r = 0; r < k; ++r) {
            double sum = 0, qsum = 0;
            for (int c = 0; c < m; ++c) {
                sum += double(w[size_t(r)*m+c]) * x[size_t(b)*m+c];
                qsum += double(w[size_t(r)*m+c]) * xq[size_t(b)*m+c];
            }
            const float ref = Half(Half(sum) + biasValues[r]);
            const float qref = Half(Half(qsum) + biasValues[r]);
            const float actual = fastllm::half_to_float(out[size_t(b)*k+r]);
            if (!std::isfinite(actual)) throw std::runtime_error("non-finite output");
            error += double(actual-ref)*(actual-ref);
            quantizedError += double(actual-qref)*(actual-qref);
            norm += double(ref)*ref;
        }
        maxRelativeRms = std::max(maxRelativeRms, std::sqrt(error/std::max(norm, 1e-12)));
        maxQuantizedRms = std::max(maxQuantizedRms, std::sqrt(quantizedError/std::max(norm, 1e-12)));
    }
    if (graph) { Hip(hipGraphExecDestroy(executable)); Hip(hipGraphDestroy(captured)); }
    std::cout << "CASE type=" << ggml_type_name(ggml_type(type)) << " m=" << m << " k=" << k
              << " n=" << n << " graph=" << graph << " rel_rms=" << maxRelativeRms
              << " quantized_ref_rel_rms=" << maxQuantizedRms << std::endl;
    if (maxRelativeRms > .02 || maxQuantizedRms > .003)
        throw std::runtime_error("CPU reference mismatch");
}

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected fixtures directory");
        Hip(hipSetDevice(0));
        std::string root = argv[1], file;
        std::ifstream manifest(root + "/manifest.txt");
        int type, m, k, cases = 0;
        while (manifest >> type >> m >> k >> file) {
            for (int n : {1, 3, 4, 8}) {
                Run(root + "/" + file, type, m, k, n, n == 3);
                ++cases;
            }
        }
        if (!cases) throw std::runtime_error("empty manifest");
        std::cout << "PASS cases=" << cases << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}
