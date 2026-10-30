// Differential checks against the unfused FP32/BF16 operators. No model weights needed.
#include "models/minimax_h3/runtime.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3;

static std::mt19937 rng(1701);
static Tensor randomTensor(Runtime &rt, int rows, int cols) {
    std::normal_distribution<float> normal(0.f, 0.4f);
    std::vector<float> data(size_t(rows) * cols);
    for (auto &v : data)
        v = normal(rng);
    return rt.upload(data.data(), rows, cols);
}
static void equal(Runtime &rt, const Tensor &a, const Tensor &b) {
    if (a.rows != b.rows || a.cols != b.cols || a.dtype != b.dtype)
        throw std::runtime_error("shape/dtype mismatch");
    auto x = rt.download(a), y = rt.download(b);
    size_t different = 0;
    double squared = 0, reference = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
            throw std::runtime_error("non-finite output");
        different += x[i] != y[i];
        squared += double(x[i] - y[i]) * (x[i] - y[i]);
        reference += double(x[i]) * x[i];
    }
    std::cout << " different=" << different << " relative_l2=" << std::sqrt(squared / std::max(reference, 1e-30));
    if (different)
        throw std::runtime_error("fused operator changed rounding");
}
int main() {
    try {
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess && status != cudaErrorNoDevice)
            throw std::runtime_error(cudaGetErrorString(status));
        if (count != 1) {
            std::cerr << "Expose only the permitted GPU" << std::endl;
            return 77;
        }
        Runtime rt(0, false), full(0, true);
        for (bool fp32 : {false, true}) {
            Runtime &r = fp32 ? full : rt;
            for (int rows : {1, 17, 513, 12847}) {
                if (fp32 && rows == 12847)
                    continue;
                int heads = rows == 12847 ? 56 : 3;
                for (int dim : {64, 128}) {
                    if (rows == 12847 && dim != 128)
                        continue;
                    auto q = randomTensor(r, rows, heads * dim), k = randomTensor(r, rows, heads * dim);
                    auto qw = randomTensor(r, 1, dim), kw = randomTensor(r, 1, dim);
                    for (int rd : {dim / 2, dim == 128 ? 96 : 48, dim}) {
                        auto cs = randomTensor(full, rows, rd), sn = randomTensor(full, rows, rd);
                        auto qRef = r.rotary(r.rms(q, qw, 1e-5f, dim), cs, sn, dim);
                        auto kRef = r.rotary(r.rms(k, kw, 1e-5f, dim), cs, sn, dim);
                        auto fused = r.qkNormRotary(q, k, qw, kw, 1e-5f, cs, sn, dim);
                        std::cout << "qk fp32=" << fp32 << " rows=" << rows << " dim=" << dim << " rope=" << rd;
                        equal(r, qRef, fused.first);
                        equal(r, kRef, fused.second);
                        std::cout << std::endl;
                    }
                }
            }
        }
        for (bool fp32 : {false, true}) {
            Runtime &r = fp32 ? full : rt;
            for (int rows : {1, 47, 513, 12847}) {
                int width = rows == 12847 ? 14336 : 67;
                auto x = randomTensor(r, rows, width * 2);
                auto reference = r.multiply(r.columns(x, 0, width), r.activation(r.columns(x, width, width)));
                std::cout << "silu fp32=" << fp32 << " rows=" << rows;
                equal(r, reference, r.siluMultiply(x));
                std::cout << std::endl;
            }
        }
        for (bool fp32 : {false, true}) {
            Runtime &r = fp32 ? full : rt;
            for (int rows : {1, 17, 513, 12847}) {
                std::vector<int> indices(rows);
                for (int i = 0; i < rows; ++i)
                    indices[i] = (i * 7) % 9;
                auto idx = r.indices(indices);
                for (int width : {127, 5376}) {
                    auto x = randomTensor(r, rows, width), w = randomTensor(r, 1, width);
                    auto mod = randomTensor(r, 9, 6 * width);
                    for (int shift : {0, 3}) {
                        auto reference = r.modulateRows(r.rms(x, w, 1e-5f), mod, idx, shift, shift + 1);
                        std::cout << "rms_mod fp32=" << fp32 << " rows=" << rows << " width=" << width
                                  << " shift=" << shift;
                        equal(r, reference, r.rmsModulateRows(x, w, 1e-5f, mod, idx, shift, shift + 1));
                        std::cout << std::endl;
                    }
                }
            }
        }
        std::cout << "PASS" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
