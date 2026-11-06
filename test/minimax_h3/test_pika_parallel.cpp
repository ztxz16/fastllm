// Requires the real H3/Pika checkpoints and exactly two CUDA-visible GPUs.
// Compare fixed-input blocks with the single-GPU path, including packed FFN
// partitions, both row-reduction policies, odd tails and cache reloads.
#include "models/minimax_h3/runtime.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

using namespace fastllm::minimax_h3;
using fastllm::nomas::ParallelContext;
using fastllm::nomas::ParallelGroup;
using fastllm::nomas::RowReduction;

static double relativeL2(const std::vector<float> &expected, const std::vector<float> &actual) {
    if (expected.size() != actual.size()) throw std::runtime_error("shape mismatch");
    double error = 0, norm = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        double delta = double(actual[i]) - expected[i];
        if (!std::isfinite(delta)) throw std::runtime_error("nonfinite block output");
        error += delta * delta;
        norm += double(expected[i]) * expected[i];
    }
    return std::sqrt(error / std::max(norm, 1e-30));
}

int main(int argc, char **argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices != 2)
            throw std::runtime_error("Expose exactly two permitted GPUs");
        if (argc != 3) throw std::invalid_argument("usage: minimax_h3_pika_parallel MODEL PIKA");
        ParallelGroup group({0, 1});
        for (bool reference : {false, true}) {
            std::string model = std::string(argv[1]) + (reference ? "/transformer_ref" : "/transformer");
            std::string adapter = std::string(argv[2]) + (reference ? "/ref2va" : "/fl2va");
            Runtime root(0, false), full(0, true);
            Transformer single(root, full, model);
            single.loadPikaAdapter(adapter, reference);
            const auto &cfg = single.config();
            std::map<std::pair<int, int>, std::vector<float>> fp32Sums;
            for (auto reduction : {RowReduction::FP32, RowReduction::Activation}) {
                std::vector<std::unique_ptr<Runtime>> rt(2), fp(2);
                std::vector<std::unique_ptr<Transformer>> dit(2);
                group.run([&](int rank) {
                    rt[rank] = std::make_unique<Runtime>(rank, false);
                    fp[rank] = std::make_unique<Runtime>(rank, true);
                    dit[rank] = std::make_unique<Transformer>(*rt[rank], *fp[rank], model,
                        ParallelContext{&group, rank, reduction});
                    dit[rank]->loadPikaAdapter(adapter, reference);
                });
                for (int rows : {67, 513, 4097}) for (int layer : {0, 25, 49}) {
                    if (rows != 67 && layer != 0) continue;
                    root.activate();
                    std::mt19937 rng(42);
                    std::normal_distribution<float> normal(0.f, .4f);
                    std::vector<float> values(size_t(rows) * cfg.hidden), positions(rows * 3);
                    std::vector<int> ids(rows);
                    for (auto &v : values) v = normal(rng);
                    for (size_t i = 0; i < positions.size(); ++i) positions[i] = float(i % 41);
                    for (int i = 0; i < rows; ++i) ids[i] = i % 6;
                    auto x = root.upload(values.data(), rows, cfg.hidden);
                    auto input = root.cast(full.activation(single.timeEmbedding({.3f, .7f})), root.dtype());
                    auto mod = single.projectModulation(layer, input);
                    auto rope = root.ropeTable(positions, cfg.ropeFreq, cfg.ropeTheta);
                    auto expected = root.download(single.block(layer, x, mod, root.indices(ids), rope.first, rope.second));
                    std::vector<float> actual[2];
                    group.run([&](int rank) {
                        auto &r = *rt[rank];
                        auto xx = group.broadcast(x, rank), ii = group.broadcast(input, rank);
                        auto m = dit[rank]->projectModulation(layer, ii);
                        auto rr = r.ropeTable(positions, cfg.ropeFreq, cfg.ropeTheta);
                        actual[rank] = r.download(dit[rank]->block(layer, xx, m, r.indices(ids), rr.first, rr.second));
                        dit[rank]->unload();
                    });
                    if (actual[0] != actual[1]) throw std::runtime_error("ranks differ");
                    auto key = std::make_pair(rows, layer);
                    if (reduction == RowReduction::FP32) fp32Sums[key] = actual[0];
                    else if (fp32Sums.at(key) != actual[0])
                        throw std::runtime_error("Pika must retain FP32 branch sums for either caller policy");
                    double error = relativeL2(expected, actual[0]);
                    std::cout << (reference ? "ref2va" : "fl2va") << " reduction="
                        << (reduction == RowReduction::FP32 ? "fp32" : "activation")
                        << " layer=" << layer << " rows=" << rows << " relative_l2=" << error << std::endl;
                    if (error >= .01) throw std::runtime_error("BF16 block relative L2 exceeds 0.01");
                    root.activate();
                    single.unload();
                }
                group.run([&](int rank) { dit[rank].reset(); fp[rank].reset(); rt[rank].reset(); });
            }
        }
        std::cout << "PASS Pika TP2 blocks and identical rank outputs" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
