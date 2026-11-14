// Real-checkpoint test. Restore each oracle activation to isolate TP errors
// from cumulative quantization drift; also report a complete uncorrected pass.
#include "models/minimax_h3_image/model.h"
#include "models/minimax_h3_image/source.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3_image;
namespace nomas = fastllm::nomas;

static double relativeL2(const std::vector<float> &expected, const std::vector<float> &actual) {
    if (expected.size() != actual.size()) throw std::runtime_error("parallel output shape mismatch");
    double error = 0, norm = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]))
            throw std::runtime_error("nonfinite parallel output");
        error += std::pow(double(expected[i]) - actual[i], 2);
        norm += double(expected[i]) * expected[i];
    }
    return std::sqrt(error / std::max(norm, 1e-30));
}
struct StopAfterTwoBlocks {};
struct Case {
    int height, width;
    bool tail;
    Conditioning condition;
    std::vector<float> latent, text;
    std::map<std::string, std::vector<float>> oracle;
};

int main(int argc, char **argv) {
    try {
        if (argc != 5) throw std::invalid_argument(
            "usage: minimax_h3_image_quantized_parallel MODEL tp|sp fp32|activation resident|stream");
        int count = 0;
        cudaGetDeviceCount(&count);
        if (count != 2) throw std::runtime_error("expose exactly two permitted GPUs");
        const std::string mode = argv[2], reductionName = argv[3], memoryName = argv[4];
        if ((mode != "tp" && mode != "sp") || (reductionName != "fp32" && reductionName != "activation") ||
            (memoryName != "resident" && memoryName != "stream"))
            throw std::invalid_argument("invalid parallel test option");
        const auto source = resolveModelSource(argv[1]);
        if (source.quantizedWeights.empty()) throw std::invalid_argument("expected a quantized image checkpoint");
        const bool sp = mode == "sp", streaming = memoryName == "stream";
        const auto reduction = reductionName == "fp32" ? nomas::RowReduction::FP32 : nomas::RowReduction::Activation;
        Runtime rt(0), full(0, true);
        MemoryPolicy memory(rt, 0, 2);
        memory.layerStreaming = streaming;
        std::mt19937 random(42);
        std::normal_distribution<float> noise(0, .4f);
        std::vector<Case> cases{{512, 512, false, {}, {}, {}, {}},
                                {64, 96, false, {}, {}, {}, {}},
                                {2048, 2048, true, {}, {}, {}, {}}};
        for (size_t i = 0; i < cases.size(); ++i) {
            auto &c = cases[i];
            c.text.resize(17 * 5120);
            c.condition.textTags.assign(17, 1);
            int referenceRows = 0;
            if (i != 0) {
                // Two references with different grids and visual text tokens.
                c.condition.grids = {{8, 16}, {4, 8}};
                referenceRows = 40;
                std::fill(c.condition.textTags.begin() + 3, c.condition.textTags.begin() + 11, 0);
            }
            c.latent.resize(size_t(c.height / 32 * (c.width / 32) + referenceRows) * 96);
            for (auto &x : c.text) x = noise(random);
            for (auto &x : c.latent) x = noise(random);
        }
        {
            Transformer single(rt, full, source.directory, {}, false, source.quantizedWeights);
            single.setMemoryPolicy(memory);
            single.prepare();
            for (auto &c : cases) {
                try {
                    auto result = single.forward(full.upload(c.latent.data(), c.latent.size() / 96, 96),
                        rt.upload(c.text.data(), c.text.size() / 5120, 5120), c.height, c.width, 2,
                        [&](const std::string &name, const Tensor &x) {
                            c.oracle[name] = rt.download(x);
                            if (c.tail && name == "block.1") throw StopAfterTwoBlocks{};
                        }, &c.condition);
                    c.oracle["velocity"] = full.download(result);
                } catch (const StopAfterTwoBlocks &) {}
                if (c.oracle.size() != (c.tail ? 3u : 52u)) throw std::runtime_error("incomplete oracle trace");
            }
        }
        nomas::ParallelGroup group({0, 1});
        struct Rank {
            Runtime rt, full;
            Transformer model;
            Rank(int rank, nomas::ParallelGroup &group, const ModelSource &source,
                 nomas::RowReduction reduction, bool sp, const MemoryPolicy &memory)
                : rt(rank), full(rank, true), model(rt, full, source.directory,
                      {&group, rank, reduction}, sp, source.quantizedWeights) {
                model.setMemoryPolicy(memory);
            }
        };
        std::vector<std::unique_ptr<Rank>> ranks(2);
        group.run([&](int rank) {
            ranks[rank] = std::make_unique<Rank>(rank, group, source, reduction, sp, memory);
            ranks[rank]->model.prepare();
        });
        double maximum = 0;
        for (auto &c : cases) {
            std::vector<float> outputs[2];
            group.run([&](int rank) {
                auto &r = *ranks[rank];
                size_t stages = 0;
                try {
                    auto result = r.model.forward(r.full.upload(c.latent.data(), c.latent.size() / 96, 96),
                        r.rt.upload(c.text.data(), c.text.size() / 5120, 5120), c.height, c.width, 2,
                        [&](const std::string &name, const Tensor &x) {
                            const auto &expected = c.oracle.at(name);
                            if (rank == 0) {
                                const auto error = relativeL2(expected, r.rt.download(x));
                                maximum = std::max(maximum, error);
                                std::cout << c.height << 'x' << c.width << '/' << name << " l2=" << error << std::endl;
                            }
                            auto oracle = r.rt.cast(r.full.upload(expected.data(), x.rows, x.cols), x.dtype);
                            const auto status = cudaMemcpyAsync(x.data, oracle.data, x.bytes(),
                                                                 cudaMemcpyDeviceToDevice, cudaStreamPerThread);
                            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
                            ++stages;
                            if (c.tail && name == "block.1") throw StopAfterTwoBlocks{};
                        }, &c.condition);
                    outputs[rank] = r.full.download(result);
                } catch (const StopAfterTwoBlocks &) {}
                if (stages != (c.tail ? 3u : 51u)) throw std::runtime_error("incomplete parallel trace");
            });
            if (!c.tail) {
                if (outputs[0] != outputs[1]) throw std::runtime_error("rank outputs differ");
                const double error = relativeL2(c.oracle.at("velocity"), outputs[0]);
                std::cout << "velocity_l2=" << error << std::endl;
                if (error > .01) throw std::runtime_error("parallel output head L2 exceeds 0.01");
            }
        }
        auto &c = cases.front();
        std::vector<float> accumulated[2];
        group.run([&](int rank) {
            auto &r = *ranks[rank];
            auto out = r.model.forward(r.full.upload(c.latent.data(), c.latent.size() / 96, 96),
                r.rt.upload(c.text.data(), c.text.size() / 5120, 5120), c.height, c.width, 2, {}, &c.condition);
            accumulated[rank] = r.full.download(out);
        });
        if (accumulated[0] != accumulated[1]) throw std::runtime_error("uncorrected rank outputs differ");
        std::cout << "accumulated_velocity_l2=" << relativeL2(c.oracle.at("velocity"), accumulated[0]) << std::endl;
        group.run([&](int rank) { ranks[rank].reset(); });
        std::cout << weightFormatName(source.format) << ' ' << mode << ' ' << reductionName << ' ' << memoryName
                  << " fixed_block_inputs=1 block_max_l2=" << maximum << std::endl;
        if (maximum > .05) throw std::runtime_error("quantized parallel block L2 exceeds 0.05");
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
