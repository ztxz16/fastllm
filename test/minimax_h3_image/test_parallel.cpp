#include "models/minimax_h3_image/parallel.h"
#include "models/minimax_h3_image/ops.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3_image;
namespace nomas = fastllm::nomas;

static void compare(const std::string &name, const std::vector<float> &expected,
                    const std::vector<float> &actual, bool enforce = true) {
    if (expected.size() != actual.size())
        throw std::runtime_error("TP shape mismatch: " + name);
    double delta = 0, norm = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            throw std::runtime_error("nonfinite TP output: " + name);
        double difference = double(actual[i]) - expected[i];
        delta += difference * difference;
        norm += double(expected[i]) * expected[i];
    }
    double error = std::sqrt(delta / std::max(norm, 1e-30));
    std::cout << name << " relative_l2=" << error << std::endl;
    if (enforce && error > 1e-2)
        throw std::runtime_error("TP exceeds relative L2 tolerance: " + name);
}

// Exercise actual side-stream dependencies, uneven shards and allocator reuse.
// The oracle executes each head group serially with exactly the same inputs.
static void checkAttentionPipeline() {
    nomas::ParallelGroup group({0, 1});
    for (int totalRows : {1023, 1025, 4105, 8191, 9243, 16385}) {
        const int perRank = (totalRows + 1) / 2;
        for (int requestedGroups : {4, 7, 0}) {
            group.run([&](int rank) {
                Runtime rt(rank);
                nomas::TensorPool pool(rank, 256u << 20);
                nomas::TensorPool::Scope scope(pool);
                SequenceAttentionPipeline pipeline;
                const int groups = requestedGroups ? requestedGroups : pipeline.groups(perRank, 2);
                const int rows = std::min(perRank, totalRows - rank * perRank);
                const int width = 56 / groups;
                std::vector<float> values(size_t(rows) * 56 * 384), weights(128, 1), positions(rows * 3);
                for (size_t j = 0; j < values.size(); ++j)
                    values[j] = std::sin(float(j % 65537) * .019f + rank) * .5f;
                for (size_t j = 0; j < positions.size(); ++j) positions[j] = float(j % 113) / 7;
                auto qkv = rt.upload(values.data(), rows, 56 * 384);
                auto w = rt.upload(weights.data(), 1, 128);
                auto rope = rt.ropeTable(positions, 16, 10000);
                auto kv = prepareSequenceQkv(qkv, w, w, rope.first, rope.second, groups,
                                              perRank * 2, rank * perRank);
                Tensor expected(rows, 56 * 128);
                group.gatherChunks(kv, groups, rank, [&](int g, const Tensor &all) {
                    auto part = sequenceAttention(qkv, all.rowSlice(0, totalRows), 56, g * width, width);
                    copyAttentionHeads(part, expected, g * width);
                }, true);
                auto oracle = rt.download(expected);
                Tensor actual;
                for (int repeat = 0; repeat < 8; ++repeat) {
                    actual = pipeline.forward(qkv, kv, totalRows, groups, {&group, rank});
                    // Queue consumers and recycle their temporary storage while
                    // the next forward starts; no host sync between iterations.
                    auto scratch = rt.add(actual, actual);
                    if (repeat == 3) pipeline.release();
                }
                const auto result = rt.download(actual);
                compare("pipeline/rows" + std::to_string(totalRows) + "/groups" +
                            std::to_string(groups) + "/rank" + std::to_string(rank), oracle, result);
                if (oracle != result) throw std::runtime_error("concurrent attention must be bitwise equal");
                rt.synchronize();
            });
        }
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 2 && argc != 3)
            throw std::invalid_argument("usage: minimax_h3_image_parallel MODEL [fp32|activation|sp]");
        const std::string mode = argc == 3 ? argv[2] : "fp32";
        if (mode != "fp32" && mode != "activation" && mode != "sp")
            throw std::invalid_argument("parallel mode must be fp32, activation or sp");
        const bool sequence = mode == "sp";
        if (sequence) checkAttentionPipeline();
        auto reduction =
            mode == "fp32" ? fastllm::nomas::RowReduction::FP32 : fastllm::nomas::RowReduction::Activation;
        Runtime rt(0), full(0, true);
        MemoryPolicy memory(rt);
        fastllm::nomas::QwenTokenizer tokenizer(std::string(argv[1]) + "/tokenizer");
        auto tokens = tokenizer.encode("A red panda beside a mountain stream, autumn forest");
        std::vector<float> expectedText;
        {
            fastllm::minimax_h3::TextEncoder text(rt, std::string(argv[1]) + "/text_encoder");
            text.setHostMemoryBudget(memory.host);
            expectedText = full.download(text.encode(tokens));
        }
        struct Case {
            int height, width;
            Conditioning condition;
            std::vector<float> latent, text;
            std::map<std::string, std::vector<float>> stages;
        };
        std::vector<Case> cases{{64, 96, {}, {}, expectedText, {}}, {2048, 2048, {}, {}, expectedText, {}}};
        cases.push_back({2048, 2048, {}, {}, {}, {}});
        auto &edit = cases.back();
        edit.text.resize(size_t(1052) * 5120);
        for (size_t i = 0; i < edit.text.size(); ++i)
            edit.text[i] = expectedText[i % expectedText.size()];
        std::mt19937 random(42);
        std::normal_distribution<float> noise(0, .5f);
        for (auto &c : cases) {
            const bool editing = c.text.size() / 5120 == 1052;
            c.condition.textTags.assign(c.text.size() / 5120, 1);
            if (editing)
                std::fill(c.condition.textTags.begin() + 4, c.condition.textTags.begin() + 1028, 0);
            c.condition.grids = editing ? std::vector<std::pair<int, int>>{{128, 128}}
                                        : std::vector<std::pair<int, int>>{{8, 16}};
            c.latent.resize(size_t(c.height / 32 * (c.width / 32) + (editing ? 4096 : 32)) * 96);
            for (auto &x : c.latent)
                x = noise(random);
        }
        {
            Transformer transformer(rt, full, argv[1]);
            transformer.setMemoryPolicy(memory);
            transformer.prepare();
            for (auto &c : cases) {
                auto z = full.upload(c.latent.data(), c.latent.size() / 96, 96);
                auto text = rt.upload(c.text.data(), c.text.size() / 5120, 5120);
                auto out = transformer.forward(
                    z, text, c.height, c.width, 2,
                    [&](const std::string &name, const Tensor &value) {
                        c.stages[name] = full.download(value);
                    },
                    &c.condition);
                c.stages["velocity"] = full.download(out);
            }
        }
        {
            ParallelModel model(argv[1], {0, 1}, memory, 0, reduction, sequence);
            compare("text", expectedText, full.download(model.encode(tokens, {}, {})));
            rt.activate();
            VAE serial(rt, full, argv[1]);
            serial.setMemoryPolicy(memory);
            for (int batch : {1, 3, 121}) {
                std::vector<float> data(size_t(batch) * 16 * 16 * 24);
                for (auto &value : data)
                    value = noise(random);
                auto input = full.upload(data.data(), batch * 16 * 16, 24);
                auto expected = full.download(serial.decodeTiles(input, 16, 16, batch));
                compare("vae/batch" + std::to_string(batch), expected,
                        full.download(model.decodeTiles(input, 16, 16, batch)));
            }
        }
        rt.activate();
        struct Rank {
            Runtime rt, full;
            Transformer model;
            Rank(int rank, fastllm::nomas::ParallelGroup &group, const char *directory,
                 const MemoryPolicy &memory, fastllm::nomas::RowReduction reduction, bool sequence)
                : rt(rank), full(rank, true),
                  model(rt, full, directory, {&group, rank, reduction}, sequence) {
                model.setMemoryPolicy(memory);
                rt.setLinearAutotuning(sequence);
            }
        };
        {
            fastllm::nomas::ParallelGroup group({0, 1});
            std::vector<std::unique_ptr<Rank>> ranks(2);
            group.run([&](int rank) {
                ranks[rank] = std::make_unique<Rank>(rank, group, argv[1], memory, reduction, sequence);
                ranks[rank]->model.prepare();
            });
            for (auto &c : cases) {
                auto z = full.upload(c.latent.data(), c.latent.size() / 96, 96);
                auto text = rt.upload(c.text.data(), c.text.size() / 5120, 5120);
                auto prefix = mode + "/" + std::to_string(c.height) + "x" + std::to_string(c.width) +
                              "/tokens" + std::to_string(c.text.size() / 5120);
                for (bool sameInput : {true, false}) {
                    group.run([&](int rank) {
                        auto &r = *ranks[rank];
                        auto latent = group.broadcast(z, rank), prompt = group.broadcast(text, rank);
                        auto out = r.model.forward(
                            latent, prompt, c.height, c.width, 2,
                            [&](const std::string &name, const Tensor &value) {
                                const auto &expected = c.stages.at(name);
                                if (rank == 0)
                                    compare(prefix + (sameInput ? "/same_input/" : "/accumulated/") + name,
                                            expected, r.full.download(value), sameInput);
                                if (sameInput) {
                                    // Compare every block with identical inputs on every rank.
                                    // Full 50-block accumulated drift is reported separately.
                                    auto oracle = r.full.cast(
                                        r.full.upload(expected.data(), value.rows, value.cols), value.dtype);
                                    auto status =
                                        cudaMemcpyAsync(value.data, oracle.data, value.bytes(),
                                                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
                                    if (status != cudaSuccess)
                                        throw std::runtime_error(cudaGetErrorString(status));
                                }
                            },
                            &c.condition);
                        if (rank == 0)
                            compare(prefix + (sameInput ? "/same_input/velocity" : "/accumulated/velocity"),
                                    c.stages.at("velocity"), r.full.download(out), sameInput);
                    });
                }
            }
            group.run([&](int rank) { ranks[rank].reset(); });
        }
        rt.activate();
        std::cout << "image tensor parallel comparison passed" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
