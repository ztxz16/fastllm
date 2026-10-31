// Fused-operator and shared linear-output regression checks. No model weights needed.
#include "models/minimax_h3/runtime.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
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
template <class F> static void rejects(F fn) {
    try {
        fn();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid linear output was accepted");
}
static void linearOutputs() {
    for (auto dtype : {DType::FP32, DType::FP16, DType::BF16}) {
        Runtime rt(0, dtype);
        for (int rows : {1, 47, 512, 513}) {
            const int width = 128;
            // Dyadic values make the FP32 dot products exact, so this CPU oracle
            // checks every dtype without depending on the GEMM under test.
            std::vector<float> x(size_t(rows) * width), w(width * width), b(width);
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = (int(i % 31) - 15) / 16.f;
            for (size_t i = 0; i < w.size(); ++i)
                w[i] = (int((i * 7) % 31) - 15) / 16.f;
            for (int i = 0; i < width; ++i)
                b[i] = (i % 7 - 3) / 16.f;
            auto input = rt.upload(x.data(), rows, width), weight = rt.upload(w.data(), width, width);
            auto bias = rt.upload(b.data(), 1, width);
            for (bool biased : {false, true}) {
                std::vector<float> expected(size_t(rows) * width);
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < width; ++c) {
                        float sum = 0;
                        for (int k = 0; k < width; ++k)
                            sum += x[size_t(r) * width + k] * w[c * width + k];
                        expected[size_t(r) * width + c] = sum + (biased ? b[c] : 0.f);
                    }
                std::vector<float> initial(size_t(rows + 2) * width, -17.f);
                auto storage = rt.upload(initial.data(), rows + 2, width);
                auto target = storage.rowSlice(1, rows);
                rt.linearInto(input, weight, target, biased ? &bias : nullptr);
                std::cout << "linear_into dtype=" << int(dtype) << " rows=" << rows << " bias=" << biased;
                equal(rt, rt.upload(expected.data(), rows, width), target);
                equal(rt, rt.linear(input, weight, biased ? &bias : nullptr), target);
                auto result = rt.download(storage);
                for (int i = 0; i < width; ++i)
                    if (result[i] != -17.f || result[size_t(rows + 1) * width + i] != -17.f)
                        throw std::runtime_error("linear wrote outside the output slice");
                std::cout << std::endl;
            }
            auto wrongShape = randomTensor(rt, rows, width + 1);
            rejects([&] { rt.linearInto(input, weight, wrongShape); });
            rejects([&] { rt.linearInto(input, weight, input); });
            Tensor wrongType(rows, width, dtype == DType::FP32 ? DType::BF16 : DType::FP32);
            rejects([&] { rt.linearInto(input, weight, wrongType); });
        }
        if (dtype != DType::FP32) {
            auto input = randomTensor(rt, 1, 128), weight = randomTensor(rt, 127, 128);
            auto bias = randomTensor(rt, 1, 127), storage = randomTensor(rt, 3, 127);
            auto unaligned = storage.rowSlice(1, 1);
            rejects([&] { rt.linearInto(input, weight, unaligned, &bias); });
        }
    }
}
static void residuals(Runtime &rt) {
    for (int rows : {1, 17, 513, 12847}) {
        std::vector<int> indices(rows);
        for (int i = 0; i < rows; ++i)
            indices[i] = (i * 7) % 9;
        auto idx = rt.indices(indices);
        for (int width : {127, 512, 5376}) {
            auto x = randomTensor(rt, rows, width), y = randomTensor(rt, rows, width);
            // Exercise an odd table stride as well as the packed BF16 path.
            for (int extra : {0, 1}) {
                if (extra && rows != 17)
                    continue;
                auto mod = randomTensor(rt, 9, 6 * width + extra);
                for (int gate : {2, 5}) {
                    auto gates = rt.gather(rt.columns(mod, gate * width, width), idx);
                    auto reference = rt.add(x, rt.multiply(y, gates));
                    std::cout << "residual fp32=" << rt.isFloat32() << " rows=" << rows
                              << " width=" << width << " gate=" << gate << " extra=" << extra;
                    equal(rt, reference, rt.residualRows(x, y, mod, idx, gate));
                    std::cout << std::endl;
                }
            }
        }
    }
    const int width = 512;
    std::vector<float> edges = {0.f, -0.f, 1.f, -1.f, std::ldexp(1.f, -133),
                               -std::ldexp(1.f, -133), std::ldexp(1.f, -126),
                               -std::ldexp(1.f, -126), std::ldexp(1.f, -60), std::ldexp(1.f, 60)};
    std::vector<float> a(width), b(width), m(6 * width);
    for (int i = 0; i < width; ++i) {
        a[i] = edges[i % edges.size()];
        b[i] = edges[(i / edges.size()) % edges.size()];
        for (int g = 0; g < 6; ++g)
            m[g * width + i] = edges[(i / (edges.size() * edges.size()) + g) % edges.size()];
    }
    auto x = rt.upload(a.data(), 1, width), y = rt.upload(b.data(), 1, width);
    auto mod = rt.upload(m.data(), 1, 6 * width), idx = rt.indices({0});
    for (int gate : {2, 5}) {
        auto reference = rt.add(x, rt.multiply(y, rt.columns(mod, gate * width, width)));
        std::cout << "residual edges fp32=" << rt.isFloat32() << " gate=" << gate;
        equal(rt, reference, rt.residualRows(x, y, mod, idx, gate));
        std::cout << std::endl;
    }
    if (!rt.isFloat32()) {
        auto storage = randomTensor(rt, 1, width + 1);
        auto unaligned = storage;
        ++unaligned.data;
        unaligned.cols = width;
        auto reference = rt.add(unaligned, rt.multiply(y, rt.columns(mod, 2 * width, width)));
        std::cout << "residual unaligned BF16";
        equal(rt, reference, rt.residualRows(unaligned, y, mod, idx, 2));
        std::cout << std::endl;
    }
}
static void hostWeightCache() {
    auto path = std::filesystem::temp_directory_path() /
                ("fastllm-h3-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove_all(path, error); }
    } cleanup{path};
    std::filesystem::create_directory(path);
    for (bool bf16 : {false, true}) {
        std::string name = bf16 ? "a" : "b";
        std::string header = "{\"" + name + "\":{\"dtype\":\"" + (bf16 ? "BF16" : "F32") +
                             "\",\"shape\":[2,4],\"data_offsets\":[0," + (bf16 ? "16" : "32") + "]}}";
        uint64_t length = header.size();
        float floats[] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint16_t halves[] = {0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100};
        std::ofstream file(path / (name + ".safetensors"), std::ios::binary);
        file.write(reinterpret_cast<const char *>(&length), 8);
        file.write(header.data(), header.size());
        file.write(bf16 ? reinterpret_cast<const char *>(halves) : reinterpret_cast<const char *>(floats), bf16 ? 16 : 32);
    }
    std::ofstream(path / "model.safetensors.index.json") <<
        "{\"weight_map\":{\"a\":\"a.safetensors\",\"b\":\"b.safetensors\"}}";
    for (auto dtype : {DType::FP32, DType::BF16, DType::FP16}) {
        Runtime rt(0, dtype);
        for (size_t budget : {size_t(16), size_t(1024)}) {
            Weights weights(rt, path.string());
            weights.enableHostCache(budget);
            for (int round = 0; round < 3; ++round) {
                for (auto name : {"a", "b"}) {
                    auto result = rt.download(weights.get(name));
                    for (int i = 0; i < 8; ++i)
                        if (result[i] != float(i + 1))
                            throw std::runtime_error("host weight cache changed values");
                }
                weights.erasePrefix("a");
                weights.clear();
                if (round == 1)
                    weights.enableHostCache(0);
            }
        }
    }
    Runtime rt(0, false);
    Tensor first, second;
    {
        Weights weights(rt, path.string());
        weights.enableHostCache(1024);
        std::exception_ptr error;
        std::thread worker([&] {
            try { first = weights.get("a"); } catch (...) { error = std::current_exception(); }
        });
        worker.join();
        if (error)
            std::rethrow_exception(error);
        weights.erasePrefix("a");
        second = weights.get("a");
        // Destroy the host cache immediately after enqueueing its last upload.
        weights.clear();
    }
    equal(rt, first, second);
    std::cout << "host cache: dtype/shard/budget/clear/disable/worker/lifetime PASS" << std::endl;
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
        hostWeightCache();
        linearOutputs();
        Runtime rt(0, false), full(0, true);
        residuals(rt);
        residuals(full);
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
