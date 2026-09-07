#include "fastllm.h"
#include "devices/moe_decode_scheduler.h"
#include "fastllm-cuda.cuh"
#include "fastllm-cuda-shared-weight.cuh"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#include "devices/numas/numas.h"
#include "executor.h"
namespace fastllm { NumaConfig *GetNumaConfig(); }
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

static void Check(cudaError_t state) {
    if (state != cudaSuccess) throw std::runtime_error(cudaGetErrorString(state));
}
static void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static void AllocateGpu(fastllm::Data &data) {
    data.dataDevice = fastllm::DataDevice::CUDA;
    data.dataDeviceIds = {0};
    data.Allocate(false);
}

static size_t ExpertRecordBytes(const fastllm::Data &gate, const fastllm::Data &down) {
    const size_t downOffset = (gate.GetBytes() + 15) / 16 * 16;
    const size_t scalesOffset = (downOffset + down.GetBytes() + 15) / 16 * 16;
    const size_t scaleBytes = gate.dataType == fastllm::DataType::FP8_E4M3_BLOCK_128 ? 0 :
        (gate.scales.size() + down.scales.size()) * sizeof(float);
    return (scalesOffset + scaleBytes + 127) / 128 * 128;
}

template<class T>
static void Run(fastllm::DataType dtype, int hidden, int inter,
                fastllm::DataType weightType = fastllm::DataType::NVFP4_BLOCK_16_E4M3,
                int batch = 1) {
    constexpr int experts = 32, topk = 10;
    std::vector<std::unique_ptr<fastllm::Data>> owned;
    std::vector<fastllm::Data *> weights(2 * (experts + 1), nullptr);
    for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
        auto weight = std::make_unique<fastllm::Data>(weightType);
        int rows = part == 0 ? 2 * inter : hidden;
        int cols = part == 0 ? hidden : inter;
        weight->blockK = weightType == fastllm::DataType::FP8_E4M3 ? 128 : 1;
        weight->blockM = weightType == fastllm::DataType::NVFP4_BLOCK_16_E4M3 ? 16 : 128;
        weight->Resize({rows, cols}); weight->Allocate(false);
        // Constant decoded weights give an independent closed-form oracle.
        // Distinct down scales expose wrong table/slot selection.
        const float scale = part == 0 ? 1 : float(e + 1) / 32;
        if (weightType == fastllm::DataType::NVFP4_BLOCK_16_E4M3) {
            size_t packed = size_t(rows) * ((cols + 1) / 2);
            std::memset(weight->cpuData, 0x22, packed);
            std::memset(weight->cpuData + packed, 0x38, weight->GetBytes() - packed);
            weight->scales = part == 0 ? std::vector<float>{1, 1}
                                      : std::vector<float>{scale};
        } else if (weightType == fastllm::DataType::FP8_E4M3) {
            std::memset(weight->cpuData, 0x38, weight->GetBytes());
            weight->scales.assign(size_t((rows + 127) / 128) * ((cols + 127) / 128), scale);
        } else {
            const int scaleCols = (cols + 127) / 128;
            const size_t pitch = cols + scaleCols * sizeof(float);
            for (int row = 0; row < rows; ++row) {
                uint8_t *p = weight->cpuData + row * pitch;
                for (int col = 0; col < scaleCols; ++col) {
                    uint8_t *block = p + col * (128 + sizeof(float));
                    const int count = std::min(128, cols - col * 128);
                    std::memset(block, 0x38, count);
                    std::memcpy(block + count, &scale, sizeof(float));
                }
            }
        }
        weights[2 * (e + 1) + part] = weight.get();
        owned.push_back(std::move(weight));
    }
    FastllmCudaMoeCacheLayer layer{weights.data(), int(weights.size())};
    fastllm::SetMoeCudaCacheBytes(0);
    Require(!FastllmCudaPrepareMoeCache(&layer, 1), "disabled cache prepared");
    fastllm::SetMoeCudaCacheBytes(1);
    Require(!FastllmCudaPrepareMoeCache(&layer, 1), "insufficient budget accepted");
    const size_t stride = ExpertRecordBytes(*weights[2], *weights[3]);
    fastllm::SetMoeCudaCacheBytes(16 * stride);
    if (weightType == fastllm::DataType::FP8_E4M3) {
        const float saved = weights[2]->scales.back();
        weights[2]->scales.pop_back();
        Require(!FastllmCudaPrepareMoeCache(&layer, 1), "missing FP8 scale accepted");
        weights[2]->scales.push_back(saved);
        weights[2]->blockM = 127;
        Require(!FastllmCudaPrepareMoeCache(&layer, 1), "unaligned FP8 scale block accepted");
        weights[2]->blockM = 128;
    }
    Require(FastllmCudaPrepareMoeCache(&layer, 1), "valid table rejected");

    fastllm::Data input(dtype, {batch, hidden}), index(fastllm::DataType::INT32, {batch, topk});
    fastllm::Data score(fastllm::DataType::FLOAT32, {batch, topk}), gate, output;
    AllocateGpu(input); AllocateGpu(index); AllocateGpu(score);
    std::vector<T> activation(batch * hidden);
    for (int row = 0; row < batch; ++row)
        std::fill_n(activation.begin() + row * hidden, hidden, T((1 + row % 4) / 128.0f));
    std::vector<float> scores(batch * topk, 1.0f / topk);
    Check(cudaMemcpy(input.cudaData, activation.data(), activation.size() * sizeof(T), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(score.cudaData, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
    auto supported = [&] {
        return FastllmCudaCanRunMoeCacheSmallBatch(input, index, score, weights.data(),
                                              weights.size(), fastllm::MoeGateSwiglu);
    };
    Require(supported(), "valid decode rejected");
    input.dims[1] = hidden + 1;
    Require(!supported(), "mismatched hidden width accepted");
    input.dims[1] = hidden;
    index.dims[0] = batch + 1;
    Require(!supported(), "mismatched route rows accepted");
    index.dims[0] = batch;
    input.dims[0] = index.dims[0] = score.dims[0] = FASTLLM_CUDA_MOE_CACHE_MAX_BATCH + 1;
    Require(!supported(), "prefill batch accepted by decode cache");
    input.dims[0] = index.dims[0] = score.dims[0] = batch;
    for (auto *data : {&input, &index, &score}) {
        data->strides[1]++;
        Require(!supported(), "noncontiguous columns accepted");
        data->strides[1]--;
        if (batch > 1) {
            data->strides[0]++;
            Require(!supported(), "padded cache rows accepted");
            data->strides[0]--;
        }
    }

    auto launch = [&] {
        Require(FastllmCudaMergeMOECache(input, gate, output, weights.data(),
                    weights.size(), static_cast<int32_t *>(index.cudaData),
                    static_cast<float *>(score.cudaData), topk), "decode failed");
    };
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    for (int pass = 0; pass < 5; ++pass) {
        std::vector<int32_t> ids(batch * topk);
        for (int row = 0; row < batch; ++row) for (int k = 0; k < topk; ++k)
            ids[row * topk + k] = (pass * 9 + row * 11 + k) % experts;
        Check(cudaMemcpy(index.cudaData, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        if (pass == 0) {
            launch(); Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            launch();
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        }
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<T> actual(batch * hidden);
        Check(cudaMemcpy(actual.data(), output.cudaData, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
        for (int row = 0; row < batch; ++row) {
            float g = float(T(hidden * (1 + row % 4) / 128.0f));
            float activated = float(T((g / (1 + std::exp(-g))) * g));
            float expected = 0;
            for (int k = 0; k < topk; ++k)
                expected += float(T(inter * activated * ((ids[row * topk + k] + 1) / 32.0f))) / topk;
            expected = float(T(expected));
            float tolerance = dtype == fastllm::DataType::FLOAT32 ? 1e-5f :
                              dtype == fastllm::DataType::FLOAT16 ? 0.002f : 0.012f;
            for (int col = 0; col < hidden; ++col)
                Require(std::fabs(float(actual[row * hidden + col]) - expected) <= tolerance * std::max(1.0f, std::fabs(expected)),
                        "expert output mismatch");
        }
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    FastllmCudaReleaseMoeCache(weights.data(), weights.size());
    Require(!supported(), "released table still registered");
    FastllmCudaReleaseMoeCache(weights.data(), weights.size());
    fastllm::SetMoeCudaCacheBytes(0);
    std::printf("PASS adapter weight=%d dtype=%d hidden=%d inter=%d batch=%d, graph/eviction/release/validation\n",
                int(weightType), int(dtype), hidden, inter, batch);
}

template<class T>
static void CompareFP8(fastllm::DataType dtype, fastllm::DataType weightType,
                       std::vector<std::unique_ptr<fastllm::Data>> &keepAlive, int batch = 1) {
    constexpr int hidden = 256, inter = 128, experts = 24, topk = 7, tables = 2;
    const auto computeE4M3 = std::is_same_v<T, half>
        ? FastllmCudaHalfMergeMOEFP8E4M3Batch1Indexed
        : FastllmCudaBFloat16MergeMOEFP8E4M3Batch1Indexed;
    const auto computeBlock128 = std::is_same_v<T, half>
        ? FastllmCudaHalfMergeMOEFP8E4M3Block128Batch1Indexed
        : FastllmCudaBFloat16MergeMOEFP8E4M3Block128Batch1Indexed;
    std::mt19937 rng(42);
    std::vector<std::unique_ptr<fastllm::Data>> owned;
    std::vector<fastllm::Data *> weights[tables];
    FastllmCudaMoeCacheLayer layers[tables];
    for (int table = 0; table < tables; ++table) {
        weights[table].resize(2 * (experts + 1), nullptr);
        for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
            auto w = std::make_unique<fastllm::Data>(weightType);
            const int rows = part == 0 ? inter * 2 : hidden;
            const int cols = part == 0 ? hidden : inter;
            w->blockM = w->blockK = 128;
            w->Resize({rows, cols}); w->Allocate(false);
            if (weightType == fastllm::DataType::FP8_E4M3) {
                for (size_t i = 0; i < w->GetBytes(); ++i)
                    w->cpuData[i] = uint8_t(0x20 + rng() % 32) | ((rng() & 1) ? 0x80 : 0);
                w->scales.resize(size_t((rows + 127) / 128) * ((cols + 127) / 128));
                for (float &scale : w->scales) scale = float(1 + rng() % 7) / 16;
            } else {
                size_t offset = 0;
                for (int row = 0; row < rows; ++row) for (int col = 0; col < cols; col += 128) {
                    for (int i = 0; i < 128; ++i)
                        w->cpuData[offset++] = uint8_t(0x20 + rng() % 32) | ((rng() & 1) ? 0x80 : 0);
                    float scale = float(1 + rng() % 7) / 16;
                    std::memcpy(w->cpuData + offset, &scale, sizeof(float)); offset += sizeof(float);
                }
                Require(offset == w->GetBytes(), "packed FP8 test layout mismatch");
            }
            weights[table][2 * (e + 1) + part] = w.get();
            owned.push_back(std::move(w));
        }
        layers[table] = {weights[table].data(), int(weights[table].size())};
    }
    const size_t stride = ExpertRecordBytes(*weights[0][2], *weights[0][3]);
    fastllm::SetMoeCudaCacheBytes(16 * stride);
    Require(FastllmCudaPrepareMoeCache(layers, tables), "FP8 reference cache prepare failed");
    // The cache owns a compact host snapshot; the original tables can now
    // serve as an independent all-resident GPU backend reference.
    for (auto &w : owned) w->ToDevice(fastllm::DataDevice::CUDA);
    fastllm::Data input(dtype, {batch, hidden}), ids(fastllm::DataType::INT32, {batch, topk});
    fastllm::Data scores(fastllm::DataType::FLOAT32, {batch, topk});
    fastllm::Data gate[tables], output[tables], refGate[tables], refOutput[tables];
    AllocateGpu(input); AllocateGpu(ids); AllocateGpu(scores);
    std::vector<T> activation(batch * hidden);
    std::vector<float> score(batch * topk);
    cudaGraph_t graph[tables]{}; cudaGraphExec_t exec[tables]{};
    for (int pass = 0; pass < 50; ++pass) {
        const int table = pass % tables;
        std::vector<int32_t> indices(batch * topk);
        for (auto &v : activation) v = T((int(rng() % 31) - 15) / 64.0f);
        for (int k = 0; k < batch * topk; ++k) {
            indices[k] = pass < 4 ? k % experts : rng() % experts;
            score[k] = float(1 + rng() % 8) / 32;
        }
        Check(cudaMemcpy(input.cudaData, activation.data(), activation.size() * sizeof(T), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(ids.cudaData, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(scores.cudaData, score.data(), score.size() * sizeof(float), cudaMemcpyHostToDevice));
        auto launch = [&] {
            Require(FastllmCudaMergeMOECache(input, gate[table], output[table],
                weights[table].data(), weights[table].size(), static_cast<int32_t *>(ids.cudaData),
                static_cast<float *>(scores.cudaData), topk), "FP8 cached compute failed");
        };
        if (!exec[table]) {
            launch(); Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            launch(); Check(cudaStreamEndCapture(cudaStreamPerThread, &graph[table]));
            Check(cudaGraphInstantiate(&exec[table], graph[table], nullptr, nullptr, 0));
        }
        if (pass % 3 == 0) launch();
        else Check(cudaGraphLaunch(exec[table], cudaStreamPerThread));
        refGate[table].dataType = refOutput[table].dataType = dtype;
        refGate[table].Resize({batch * topk, inter});
        refOutput[table].Resize({batch, hidden});
        AllocateGpu(refGate[table]); AllocateGpu(refOutput[table]);
        for (int row = 0; row < batch; ++row) {
            fastllm::Data rowInput, rowGate, rowOutput;
            rowInput.FakeFrom(input, size_t(row) * hidden * sizeof(T));
            rowInput.Resize({1, hidden});
            rowGate.FakeFrom(refGate[table], size_t(row) * topk * inter * sizeof(T));
            rowGate.Resize({topk, inter});
            rowOutput.FakeFrom(refOutput[table], size_t(row) * hidden * sizeof(T));
            rowOutput.Resize({1, hidden});
            rowInput.dataDeviceIds = rowGate.dataDeviceIds = rowOutput.dataDeviceIds = {0};
            bool ok;
            auto *index = static_cast<int32_t *>(ids.cudaData) + row * topk;
            auto *s = static_cast<float *>(scores.cudaData) + row * topk;
            if (weightType == fastllm::DataType::FP8_E4M3) {
                ok = computeE4M3(rowInput, rowGate, rowOutput,
                    weights[table].data(), weights[table].size(), index, s, topk, hidden, inter, false);
            } else {
                ok = computeBlock128(rowInput, rowGate, rowOutput,
                    weights[table].data(), weights[table].size(), index, s, topk, hidden, inter);
            }
            Require(ok, "resident FP8 reference failed");
        }
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        for (bool intermediate : {false, true}) {
            auto &actual = intermediate ? gate[table] : output[table];
            auto &expected = intermediate ? refGate[table] : refOutput[table];
            std::vector<uint8_t> a(actual.GetBytes()), b(expected.GetBytes());
            Check(cudaMemcpy(a.data(), actual.cudaData, a.size(), cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(b.data(), expected.cudaData, b.size(), cudaMemcpyDeviceToHost));
            Require(a == b, "cached FP8 differs bitwise from resident GPU reference");
        }
    }
    for (int t = 0; t < tables; ++t) {
        Check(cudaGraphExecDestroy(exec[t])); Check(cudaGraphDestroy(graph[t]));
        FastllmCudaReleaseMoeCache(weights[t].data(), weights[t].size());
    }
    // Existing resident-table registries key by Data address. Keep those
    // reference weights alive until all cases finish so addresses cannot alias.
    for (auto &w : owned) keepAlive.push_back(std::move(w));
    fastllm::SetMoeCudaCacheBytes(0);
    std::printf("PASS FP8 resident-reference weight=%d dtype=%d batch=%d: 50 steps, two tables, duplicates, eviction, eager/graph, bitwise gate/output\n",
                int(weightType), int(dtype), batch);
}

template<class T>
static void CompareNumaCache(fastllm::DataType dtype, int nodes, int batch,
                            int hidden = 128, int inter = 64, bool planar = false,
                            bool hybrid = false,
                            fastllm::DataType weightType = fastllm::DataType::NVFP4_BLOCK_16_E4M3,
                            bool perchannel = false) {
    constexpr int experts = 24, topk = 7, tables = 2;
    std::mt19937 rng(171);
    std::vector<std::unique_ptr<fastllm::Data>> owned[3];
    std::vector<fastllm::Data *> weights[3][tables];
    FastllmCudaMoeCacheLayer layers[3][tables];
    std::vector<void *> shards;
    for (int backend = 0; backend < 3; ++backend) {
        // Reference, shared, and failed-registration groups start identically.
        rng.seed(171);
        for (int table = 0; table < tables; ++table) {
            weights[backend][table].resize(2 * (experts + 1), nullptr);
            for (int expert = 0; expert < experts; ++expert) {
                for (int part = 0; part < 2; ++part) {
                    const int rows = part == 0 ? 2 * inter : hidden;
                    const int cols = part == 0 ? hidden : inter;
                    const bool nvfp4 = weightType == fastllm::DataType::NVFP4_BLOCK_16_E4M3;
                    auto w = std::make_unique<fastllm::Data>(weightType);
                    w->blockK = nvfp4 ? 1 : 128; w->blockM = nvfp4 ? 16 : (perchannel ? cols : 128);
                    w->Resize({rows, cols}); w->Allocate(false);
                    if (nvfp4) {
                        for (size_t b = 0; b < fastllm::GetNVFP4WeightBytes(rows, cols); ++b) w->cpuData[b] = rng() % 256;
                        auto *scale = fastllm::GetNVFP4ScaleData(*w);
                        for (int b = 0; b < rows * ((cols + 15) / 16); ++b) scale[b] = 0x28 + rng() % 24;
                        w->scales = part == 0 ? std::vector<float>{0.73f, 1.31f} : std::vector<float>{0.39f};
                    } else if (weightType == fastllm::DataType::FP8_E4M3) {
                        for (size_t b = 0; b < size_t(rows) * cols; ++b)
                            w->cpuData[b] = (rng() % 2 ? 128 : 0) | (0x28 + rng() % 24);
                        w->scales.resize(((rows + 127) / 128) * ((cols + w->blockM - 1) / w->blockM));
                        for (float &scale : w->scales) scale = (1 + rng() % 16) / 64.0f;
                    } else {
                        const size_t pitch = fastllm::GetDataBytes(weightType, 1, cols);
                        for (int row = 0; row < rows; ++row) for (int c = 0; c < cols; c += 128) {
                            auto *block = w->cpuData + row * pitch + (c / 128) * 132;
                            for (int j = 0; j < 128; ++j) block[j] = (rng() % 2 ? 128 : 0) | (0x28 + rng() % 24);
                            const float scale = (1 + rng() % 16) / 64.0f;
                            memcpy(block + 128, &scale, sizeof(scale));
                        }
                    }
                    weights[backend][table][2 * (expert + 1) + part] = w.get();
                    owned[backend].push_back(std::move(w));
                }
            }
            layers[backend][table] = {weights[backend][table].data(), 2 * (experts + 1)};
        }
    }
    fastllm::SetMoeCudaCacheBytes(16 * ExpertRecordBytes(*weights[0][0][2], *weights[0][0][3]));
    for (int backend = 0; backend < 3; ++backend) {
        auto registerWeights = [&] {
            for (size_t i = 0; i < owned[backend].size(); ++i) {
                auto &w = *owned[backend][i];
                const int rows = w.dims[0], cols = w.dims[1];
                const bool nvfp4 = weightType == fastllm::DataType::NVFP4_BLOCK_16_E4M3;
                const bool usePlanar = nvfp4 && planar && (rows / nodes) % fastllm::NVFP4_PLANAR_TILE_ROWS == 0;
                const auto storageType = nvfp4 ? fastllm::DataType::NVFP4_BLOCK_16 :
                    (perchannel && cols != 128 ? fastllm::DataType::FP8_E4M3_PERCHANNEL : fastllm::DataType::FP8_E4M3_BLOCK_128);
                const size_t pitch = fastllm::GetDataBytes(storageType, 1, cols);
                const size_t bytes = pitch * (rows / nodes);
                for (int node = 0; node < nodes; ++node) {
                    void *ptr = nullptr;
                    Check(cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
                    if (nvfp4) {
                        fastllm::PackCompactE4M3NVFP4Block16Rows(rows, cols, w.cpuData,
                            fastllm::GetNVFP4ScaleData(w), w.scales, 1, 16,
                            static_cast<uint8_t *>(ptr), node * (rows / nodes), rows / nodes, i % 2 == 0, usePlanar);
                    } else {
                        auto *dest = static_cast<uint8_t *>(ptr);
                        for (int r = 0; r < rows / nodes; ++r) {
                            const int row = node * (rows / nodes) + r;
                            const int originalRow = i % 2 == 0 ? row / 2 + (row % 2) * (rows / 2) : row;
                            if (weightType == fastllm::DataType::FP8_E4M3_BLOCK_128) {
                                memcpy(dest + r * pitch, w.cpuData + originalRow * pitch, pitch);
                            } else for (int c = 0; c < cols; c += w.blockM) {
                                const int bytes = std::min(w.blockM, cols - c);
                                auto *block = dest + r * pitch + (c / w.blockM) * (w.blockM + sizeof(float));
                                memcpy(block, w.cpuData + originalRow * cols + c, bytes);
                                const float scale = w.scales[(originalRow / w.blockK) *
                                    ((cols + w.blockM - 1) / w.blockM) + c / w.blockM];
                                memcpy(block + bytes, &scale, sizeof(scale));
                            }
                        }
                    }
                    shards.push_back(ptr);
                    w.numasData.push_back(static_cast<uint8_t *>(ptr));
                }
                w.FreeSpace();
                w.dataType = usePlanar ? fastllm::DataType::NVFP4_BLOCK_16_PLANAR
                                    : storageType;
                w.isPinned = true;
                w.UpdateUnitSize();
            }
        };
        if (backend == 0) {
            Require(FastllmCudaPrepareMoeCache(layers[0], tables), "cache snapshot failed");
            registerWeights();
        } else if (backend == 1) {
            bool called = false;
            Require(!FastllmCudaPrepareMoeCache(layers[1], tables, [&] { called = true; }),
                    "unregistered NUMA source accepted");
            Require(called, "registration callback skipped");
            struct RegistrationError {};
            bool caught = false;
            try {
                FastllmCudaPrepareMoeCache(layers[1], tables, [] { throw RegistrationError{}; });
            } catch (const RegistrationError &) { caught = true; }
            Require(caught, "registration exception lost");
            for (int table = 0; table < tables; ++table)
                Require(!FastllmCudaCanRunMoeCache(weights[1][table].data(), 2 * (experts + 1)),
                        "failed or interrupted preparation published a cache");
            Require(FastllmCudaPrepareMoeCache(layers[1], tables, registerWeights),
                    "valid NUMA registration rejected");
        } else {
            Require(!FastllmCudaPrepareMoeCache(layers[2], tables, [&] {
                registerWeights();
                if (weightType == fastllm::DataType::NVFP4_BLOCK_16_E4M3)
                    owned[2][0]->isPinned = false;
                else owned[2][0]->numasData[0] = nullptr;
            }), "unpinned NUMA source accepted");
            for (int table = 0; table < tables; ++table)
                Require(!FastllmCudaCanRunMoeCache(weights[2][table].data(), 2 * (experts + 1)),
                        "invalid NUMA source published a cache");
            continue;
        }
        Require(FastllmCudaCanRunMoeCache(weights[backend][0].data(), 2 * (experts + 1)),
                "prepared cache unavailable");
        Require(FastllmCudaPrepareMoeCache(layers[backend], tables, [] {
            throw std::runtime_error("active cache invoked NUMA registration");
        }),
                "active cache was not reused");
    }
    fastllm::Data input(dtype, {batch, hidden}), ids(fastllm::DataType::INT32, {batch, topk});
    fastllm::Data scores(fastllm::DataType::FLOAT32, {batch, topk});
    AllocateGpu(input); AllocateGpu(ids); AllocateGpu(scores);
    fastllm::Data gate[2][tables], output[2][tables];
    cudaGraph_t graph[2][tables]{};
    cudaGraphExec_t exec[2][tables]{};
    for (int step = 0; step < 20; ++step) {
        const int table = step % tables;
        std::vector<T> activation(batch * hidden);
        std::vector<int32_t> indices(batch * topk);
        std::vector<float> routes(batch * topk);
        for (auto &v : activation) v = T((int(rng() % 31) - 15) / 64.0f);
        for (size_t i = 0; i < routes.size(); ++i) {
            indices[i] = rng() % experts;
            routes[i] = (1 + rng() % 7) / 32.0f;
        }
        Check(cudaMemcpy(input.cudaData, activation.data(), activation.size() * sizeof(T), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(ids.cudaData, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(scores.cudaData, routes.data(), routes.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (int backend = 0; backend < 2; ++backend) {
            auto launch = [&] {
                Require(FastllmCudaMergeMOECache(input, gate[backend][table], output[backend][table],
                    weights[backend][table].data(), 2 * (experts + 1),
                    static_cast<int32_t *>(ids.cudaData), static_cast<float *>(scores.cudaData), topk),
                    "cache compute failed");
            };
            if (exec[backend][table] == nullptr) {
                launch(); Check(cudaStreamSynchronize(cudaStreamPerThread));
                Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                launch();
                Check(cudaStreamEndCapture(cudaStreamPerThread, &graph[backend][table]));
                Check(cudaGraphInstantiate(&exec[backend][table], graph[backend][table], nullptr, nullptr, 0));
            }
            if (step % 3 == 0) launch();
            else Check(cudaGraphLaunch(exec[backend][table], cudaStreamPerThread));
        }
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        for (bool intermediate : {false, true}) {
            auto &expected = intermediate ? gate[0][table] : output[0][table];
            auto &actual = intermediate ? gate[1][table] : output[1][table];
            std::vector<uint8_t> a(expected.GetBytes()), b(actual.GetBytes());
            Check(cudaMemcpy(a.data(), expected.cudaData, a.size(), cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(b.data(), actual.cudaData, b.size(), cudaMemcpyDeviceToHost));
            Require(a == b, "NUMA refill changed gate/output bits");
        }
#ifdef USE_NUMAS
        if (hybrid) {
            using namespace fastllm;
            Require(dtype == FLOAT32 && batch == 1, "invalid hybrid test shape");
            // Independent existing NUMA and CUDA paths provide each expert's
            // reference. Mixed execution must choose exactly one of them per
            // route; FP32 CUDA and BF16 NUMA activations can differ slightly.
            std::vector<float> lower(hidden, 0), upper(hidden, 0), exact(hidden, 0), subset(topk * hidden);
            std::vector<int32_t> mask(topk, -1);
            Data hostInput(FLOAT32, {1, hidden}); hostInput.Allocate(false);
            memcpy(hostInput.cpuData, activation.data(), hidden * sizeof(float));
            Data oneId(INT32, {1, 1}), oneScore(FLOAT32, {1, 1});
            oneId.Allocate(false); oneScore.Allocate(false);
            *reinterpret_cast<int32_t *>(oneId.cpuData) = 0;
            *reinterpret_cast<float *>(oneScore.cpuData) = 1;
            NumasMoeDecodeExperts(reinterpret_cast<float *>(hostInput.cpuData), subset.data(),
                weights[1][table].data(), indices.data(), mask.data(), topk, table);
            Data w1, w2, w3, currentInput, currentOutput, cpuResult;
            for (int r = 0; r < topk; ++r) {
                Data *single[]{nullptr, nullptr, weights[1][table][2 * (indices[r] + 1)],
                                                weights[1][table][2 * (indices[r] + 1) + 1]};
                Data *bias[4]{};
                reinterpret_cast<Executor *>(GetExecutor())->RunOnDevice("numa", "MergeMOE",
                    {{"input", &hostInput}, {"index", &oneId}, {"score", &oneScore},
                     {"weights", reinterpret_cast<Data *>(single)}, {"biass", reinterpret_cast<Data *>(bias)},
                     {"w1", &w1}, {"w2", &w2}, {"w3", &w3}, {"curInput", &currentInput},
                     {"curOutput", &currentOutput}, {"output", &cpuResult}},
                    {{"sharedScale", 1}}, {{"weights___batch", 4}, {"biass___batch", 4}, {"layer", table}});
                std::vector<float> only(topk, 0), gpuResult(hidden);
                only[r] = 1;
                Check(cudaMemcpy(scores.cudaData, only.data(), topk * sizeof(float), cudaMemcpyHostToDevice));
                Require(FastllmCudaMergeMOECache(input, gate[0][table], output[0][table],
                    weights[0][table].data(), 2 * (experts + 1),
                    static_cast<int32_t *>(ids.cudaData), static_cast<float *>(scores.cudaData), topk),
                    "hybrid CUDA reference failed");
                Check(cudaMemcpy(gpuResult.data(), output[0][table].cudaData, hidden * sizeof(float), cudaMemcpyDeviceToHost));
                for (int c = 0; c < hidden; ++c) {
                    const float cpu = reinterpret_cast<float *>(cpuResult.cpuData)[c];
                    Require(std::abs(cpu - subset[r * hidden + c]) <= 1e-5f * (1 + std::abs(cpu)),
                            "NUMA subset changed expert arithmetic");
                    lower[c] += std::min(cpu, gpuResult[c]) * routes[r];
                    upper[c] += std::max(cpu, gpuResult[c]) * routes[r];
                    // Warmup populates all layers, then samples each split. Even steps have all
                    // routes resident; odd steps evict all of this table.
                    // Both cases keep original route order when admitting.
                    exact[c] += (r < (step < tables ? topk : topk - (step - tables) % (topk + 1)) ? gpuResult[c] : cpu) * routes[r];
                }
            }
            if (step % 2) {
                // Evict this table between lookup/compute test calls. This
                // exercises mixed misses as well as the fully resident case.
                for (int pass = 0; pass < 3; ++pass) {
                    std::vector<int32_t> other(topk);
                    for (int r = 0; r < topk; ++r) other[r] = pass * topk + r;
                    Check(cudaMemcpy(ids.cudaData, other.data(), topk * sizeof(int32_t), cudaMemcpyHostToDevice));
                    Require(FastllmCudaMergeMOECache(input, gate[1][1 - table], output[1][1 - table],
                        weights[1][1 - table].data(), 2 * (experts + 1),
                        static_cast<int32_t *>(ids.cudaData), static_cast<float *>(scores.cudaData), topk),
                        "hybrid eviction setup failed");
                }
                Check(cudaMemcpy(ids.cudaData, indices.data(), topk * sizeof(int32_t), cudaMemcpyHostToDevice));
            }
            Check(cudaMemcpy(scores.cudaData, routes.data(), topk * sizeof(float), cudaMemcpyHostToDevice));
            Data result;
            Require(FastllmCudaMergeMOEHybrid(input, ids, scores, result,
                weights[1][table].data(), 2 * (experts + 1), table), "hybrid execution rejected");
            std::vector<float> actual(hidden);
            Check(cudaMemcpy(actual.data(), result.cudaData, hidden * sizeof(float), cudaMemcpyDeviceToHost));
            for (int c = 0; c < hidden; ++c) {
                const float tolerance = 2e-5f * (1 + std::max(std::abs(lower[c]), std::abs(upper[c])));
                Require(std::isfinite(actual[c]) && actual[c] >= lower[c] - tolerance &&
                        actual[c] <= upper[c] + tolerance, "hybrid lost or duplicated an expert contribution");
                if (step < tables + 2 * (topk + 1))
                    Require(std::abs(actual[c] - exact[c]) <= tolerance,
                            "hybrid result disagrees with the assigned expert subsets");
            }
        }
#endif
    }
    for (int backend = 0; backend < 2; ++backend) {
        for (int table = 0; table < tables; ++table) {
            Check(cudaGraphExecDestroy(exec[backend][table]));
            Check(cudaGraphDestroy(graph[backend][table]));
        }
        FastllmCudaReleaseMoeCache(weights[backend][0].data(), 2 * (experts + 1));
    }
    for (void *ptr : shards) Check(cudaFreeHost(ptr));
    fastllm::SetMoeCudaCacheBytes(0);
    std::printf("PASS NUMA cache weight=%d dtype=%d nodes=%d batch=%d hidden=%d inter=%d planar=%d: bitwise gate/output, graph, eviction, validation\n",
                int(weightType), int(dtype), nodes, batch, hidden, inter, planar);
}

template <class T>
static void ComparePlanarLinear(fastllm::DataType dtype, int columns, int batch) {
    using namespace fastllm;
    constexpr int rows = 64;
    std::mt19937 rng(819);
    std::vector<uint8_t> packed(GetNVFP4WeightBytes(rows, columns));
    std::vector<uint8_t> scales(rows * ((columns + 15) / 16));
    for (auto &v : packed) v = rng();
    for (auto &v : scales) v = 0x28 + rng() % 24;
    const std::vector<float> globals{0.73f, 1.31f};
    Data weights[2]{{NVFP4_BLOCK_16, {rows, columns}},
                    {NVFP4_BLOCK_16_PLANAR, {rows, columns}}};
    Data input(dtype, {batch, columns}), bias(FLOAT32, {rows});
    AllocateGpu(input); AllocateGpu(bias);
    std::vector<T> activation(batch * columns);
    for (auto &v : activation) v = T((int(rng() % 31) - 15) / 32.0f);
    std::vector<float> biasValues(rows);
    for (auto &v : biasValues) v = (int(rng() % 17) - 8) / 32.0f;
    Check(cudaMemcpy(input.cudaData, activation.data(), activation.size() * sizeof(T), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(bias.cudaData, biasValues.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
    Data output[2]{{dtype, {batch, rows}}, {dtype, {batch, rows}}};
    for (int layout = 0; layout < 2; ++layout) {
        auto &weight = weights[layout];
        Require(IsCudaLinearDataTypeSupported(dtype, weight.dataType, FLOAT32),
                "planar CUDA Linear eligibility rejected");
        std::vector<uint8_t> bytes(weight.GetBytes());
        PackCompactE4M3NVFP4Block16Rows(rows, columns, packed.data(), scales.data(), globals,
            1, 16, bytes.data(), 0, rows, true, layout != 0);
        AllocateGpu(weight); AllocateGpu(output[layout]);
        Check(cudaMemcpy(weight.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        DoCudaLinear(input, weight, bias, output[layout]);
    }
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<uint8_t> expected(output[0].GetBytes()), actual(expected.size());
    Check(cudaMemcpy(expected.data(), output[0].cudaData, expected.size(), cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(actual.data(), output[1].cudaData, actual.size(), cudaMemcpyDeviceToHost));
    Require(expected == actual, "planar CUDA Linear changed output bits");
    std::printf("PASS planar CUDA Linear dtype=%d columns=%d batch=%d: bitwise outputs with bias\n",
                int(dtype), columns, batch);
}

// Raw byte payloads deliberately have no DataType. Exercise ordinary packed
// rows, inline block metadata, row tiles, odd tails and row interleaving, so
// sharing does not depend on the three currently implemented cache kernels.
static void CheckSharedByteViews() {
    using namespace fastllm::cuda;
    for (int mode = 0; mode < 6; ++mode) {
        SharedExpertLayout layout;
        layout.shards = 3;
        layout.auxiliaryBytes = 16;
        const uint32_t rowBytes = mode == 3 ? 37 : mode == 4 ? 128 : mode == 5 ? 132 : 64;
        for (int part = 0; part < 2; ++part) {
            auto &v = layout.weights[part];
            v = SharedWeightView::Rows(24, rowBytes, 3, part ? 1 : 2);
            if (mode == 1 || mode == 3) {
                v.blockBytes = mode == 3 ? 7 : 8;
                v.blockStride = v.blockBytes + 4;
                v.rowStride = v.tileStride = ((rowBytes + v.blockBytes - 1) / v.blockBytes) * v.blockStride;
            } else if (mode == 2) {
                v.tileRows = 4;
                v.tileStride = 4 * v.rowStride + 128;
            } else if (mode == 4) {
                v.blockBytes = 128; v.blockStride = 132;
                v.tileStride = v.rowStride = 132;
            }
            v.destinationOffset = part * 24 * rowBytes;
        }
        layout.recordBytes = 48 * rowBytes + 16;
        layout.auxiliary[0] = {0, 48 * rowBytes, 16};
        Require(layout.Prepare(), "valid shared byte view rejected");
        Require(layout.scatter == (mode >= 4), "shared copy chose an unexpected host access pattern");
        auto invalid = layout; invalid.weights[0].rowsPerShard++;
        Require(!invalid.Prepare(), "inconsistent row shards accepted");
        std::vector<void *> hostPointers, devicePointers;
        std::vector<uint8_t> expected(4 * layout.recordBytes, 0), actual(expected.size());
        uint8_t *metadata = nullptr, *deviceMetadata = nullptr, *records = nullptr;
        void **table = nullptr;
        int32_t *ids = nullptr;
        Check(cudaHostAlloc(&metadata, 4 * layout.auxiliaryBytes, cudaHostAllocMapped));
        Check(cudaHostGetDevicePointer(&deviceMetadata, metadata, 0));
        for (int e = 0; e < 4; ++e) {
            for (int part = 0; part < 2; ++part) {
                const auto &v = layout.weights[part];
                for (uint32_t n = 0; n < layout.shards; ++n) {
                    void *host = nullptr, *mapped = nullptr;
                    Check(cudaHostAlloc(&host, v.rowsPerShard / v.tileRows * v.tileStride, cudaHostAllocMapped));
                    memset(host, 0xa5, v.rowsPerShard / v.tileRows * v.tileStride);
                    Check(cudaHostGetDevicePointer(&mapped, host, 0));
                    hostPointers.push_back(host); devicePointers.push_back(mapped);
                }
                // Populate physical rows, deriving the independent logical
                // reference in the inverse direction from the gather kernel.
                for (uint32_t physical = 0; physical < v.rows; ++physical) {
                    const uint32_t logical = physical / v.rowGroups +
                        (physical % v.rowGroups) * (v.rows / v.rowGroups);
                    const uint32_t local = physical % v.rowsPerShard;
                    auto *dst = static_cast<uint8_t *>(hostPointers[(e * 2 + part) * layout.shards + physical / v.rowsPerShard]);
                    for (uint32_t c = 0; c < v.rowBytes; ++c) {
                        const uint8_t value = (e * 71 + part * 43 + physical * 19 + c) % 251;
                        dst[(local / v.tileRows) * v.tileStride + (local % v.tileRows) * v.rowStride +
                            (c / v.blockBytes) * v.blockStride + c % v.blockBytes] = value;
                        expected[e * layout.recordBytes + v.destinationOffset + logical * v.rowBytes + c] = value;
                    }
                }
            }
            for (uint32_t c = 0; c < layout.auxiliaryBytes; ++c)
                expected[(e + 1) * layout.recordBytes - 16 + c] = metadata[e * 16 + c] = e * 13 + c;
        }
        Check(cudaMalloc(&table, devicePointers.size() * sizeof(void *)));
        Check(cudaMemcpy(table, devicePointers.data(), devicePointers.size() * sizeof(void *), cudaMemcpyHostToDevice));
        Check(cudaMalloc(&records, expected.size()));
        Check(cudaMalloc(&ids, 9 * sizeof(int32_t)));
        const int32_t hostIds[]{3, 1, 0, 2, 3, 1, 0, 2, 4};
        Check(cudaMemcpy(ids, hostIds, sizeof(hostIds), cudaMemcpyHostToDevice));
        auto copy = [&] { Require(CopySharedExpertRecords(layout, table, deviceMetadata, records,
            ids, ids + 4, ids + 8, 4, {4, 128}, cudaStreamPerThread), "shared byte gather failed"); };
        cudaGraph_t graph; cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        copy();
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int pass = 0; pass < 2; ++pass) {
            if (pass) {
                // The captured copy must read the owner's current allocation,
                // rather than a hidden snapshot of its contents.
                static_cast<uint8_t *>(hostPointers[0])[0] ^= 0x5a;
                expected[0] ^= 0x5a;
            }
            Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            Check(cudaMemcpy(actual.data(), records, actual.size(), cudaMemcpyDeviceToHost));
            Require(actual == expected, "shared byte gather changed payload or retained a snapshot");
        }
        Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
        Check(cudaFree(ids)); Check(cudaFree(records)); Check(cudaFree(table));
        for (void *p : hostPointers) Check(cudaFreeHost(p));
        Check(cudaFreeHost(metadata));
    }
    std::puts("PASS shared byte layouts: packed rows, blocks, tiles, tails, borrowed updates, graph replay");
}

static void CheckDecodeScheduler() {
    fastllm::MoeDecodeScheduler scheduler;
    for (int call = 0; call < 48 + 2 * 11; ++call) {
        scheduler.calls = call;
        Require(scheduler.SelectGpuCount(10, 0, 48) == (call < 48 ? 10 : 10 - (call - 48) % 11),
                "scheduler did not populate layers before calibrating splits");
    }
    std::vector<bool> visited(48, false);
    for (int probe = 2; probe < 50; ++probe) {
        scheduler.calls = probe * 1021;
        Require(scheduler.SelectGpuCount(10, 0, 48) == 10 - probe % 11,
                "scheduler did not refresh a sparse alternative");
        visited[scheduler.calls % 48] = true;
    }
    Require(std::all_of(visited.begin(), visited.end(), [](bool v) { return v; }),
            "scheduler probes were locked to a subset of layers");
    for (int n = 1; n <= 10; ++n) {
        scheduler.cpu[n].Observe(20 + n * 10);
        scheduler.compute[n].Observe(5);
    }
    scheduler.dispatch.Observe(5);
    scheduler.refill.Observe(40);
    scheduler.calls = 33;
    Require(scheduler.SelectGpuCount(10, 0) == 2, "scheduler did not balance CPU and refill");
    Require(scheduler.SelectGpuCount(10, 5) == 6, "scheduler ignored resident experts");
    Require(scheduler.SelectGpuCount(10, 10) == 10, "scheduler rejected a resident GPU batch");
    for (int n = 0; n < 100; ++n) scheduler.refill.Observe(300);
    Require(scheduler.SelectGpuCount(10, 0) == 0, "scheduler cannot fall back after transfer slowdown");
    fastllm::MoeDecodeScheduler::Estimate cold;
    cold.Observe(15000); cold.Observe(40);
    Require(cold.us == 40, "cold-start cost poisoned estimate");
    cold.Observe(15000);
    Require(cold.us < 100, "isolated stall poisoned estimate");
}

static void CheckDecodePolicy() {
    auto routeStep = [](fastllm::MoeDecodePolicy &policy, int step) {
        int ids[10];
        for (int i = 0; i < 10; ++i) ids[i] = (step % 2) * 10 + i;
        for (int layer = 0; layer < 4; ++layer) policy.ObserveRoutes(layer * 128, ids, 10);
        policy.ObserveStep(1000, 20, 20, 4, 10);
    };
    // The same access stream fits in the large cache and thrashes the small
    // one. CPU-selected routes must contribute to the reuse estimate too.
    fastllm::MoeDecodePolicy small(512, 16), large(512, 128);
    for (int step = 0; step < 256; ++step) routeStep(small, step);
    Require(!small.UseGpu() && small.gpuSteps == 0,
            "small cache paid for an unprofitable GPU trial");
    for (int step = 0; step < 32; ++step) routeStep(large, step);
    Require(large.UseGpu() && large.MissRate() < 0.1,
            "large cache did not explore reuse of CPU-selected experts");
    for (int step = 0; step < 32; ++step) large.ObserveStep(10000, 20, 20, 4, 10);
    Require(large.GetMode() == fastllm::MoeDecodePolicy::Mode::FillGpu,
            "a still-cold GPU cache was judged before the bounded warmup");
    for (int step = 0; step < 32; ++step) large.ObserveStep(600, 20, 20, 4, 10);
    Require(large.GetMode() == fastllm::MoeDecodePolicy::Mode::Gpu,
            "cold cache/graph creation hid a faster steady GPU path");
    for (int step = 0; step < 96 && large.UseGpu(); ++step)
        large.ObserveStep(2000, 20, 20, 4, 10);
    Require(!large.UseGpu(), "GPU slowdown did not restore hybrid scheduling");

    fastllm::MoeDecodePolicy slower(512, 128);
    for (int step = 0; step < 32; ++step) routeStep(slower, step);
    for (int step = 0; step < 160; ++step) slower.ObserveStep(1100, 20, 20, 4, 10);
    Require(!slower.UseGpu(), "a high hit rate overrode slower whole-step timing");
    for (int step = 0; step < 256; ++step) routeStep(slower, step);
    Require(!slower.UseGpu(), "failed trial was immediately repeated");
}

int main() {
    try {
        CheckSharedByteViews();
        CheckDecodeScheduler();
        CheckDecodePolicy();
        for (int columns : {17, 128, 640, 4096}) {
            for (int batch : {1, 4, 31, 32, 65}) {
                ComparePlanarLinear<float>(fastllm::DataType::FLOAT32, columns, batch);
                ComparePlanarLinear<half>(fastllm::DataType::FLOAT16, columns, batch);
                ComparePlanarLinear<__nv_bfloat16>(fastllm::DataType::BFLOAT16, columns, batch);
            }
        }
        for (bool wide : {false, true}) {
            int hidden = wide ? 128 : 19, inter = wide ? 128 : 23;
            Run<float>(fastllm::DataType::FLOAT32, hidden, inter);
            Run<half>(fastllm::DataType::FLOAT16, hidden, inter);
            Run<__nv_bfloat16>(fastllm::DataType::BFLOAT16, hidden, inter);
        }
        for (auto weightType : {fastllm::DataType::FP8_E4M3,
                                fastllm::DataType::FP8_E4M3_BLOCK_128}) {
            const int widerHidden = weightType == fastllm::DataType::FP8_E4M3_BLOCK_128 ? 256 : 260;
            for (int hidden : {128, widerHidden}) {
                Run<float>(fastllm::DataType::FLOAT32, hidden, 128, weightType);
                Run<half>(fastllm::DataType::FLOAT16, hidden, 128, weightType);
                Run<__nv_bfloat16>(fastllm::DataType::BFLOAT16, hidden, 128, weightType);
            }
        }
        for (int batch : {2, 4, 6, FASTLLM_CUDA_MOE_CACHE_MAX_BATCH}) {
            for (auto weightType : {fastllm::DataType::NVFP4_BLOCK_16_E4M3,
                                   fastllm::DataType::FP8_E4M3,
                                   fastllm::DataType::FP8_E4M3_BLOCK_128}) {
                Run<float>(fastllm::DataType::FLOAT32, 128, 128, weightType, batch);
                Run<half>(fastllm::DataType::FLOAT16, 128, 128, weightType, batch);
                Run<__nv_bfloat16>(fastllm::DataType::BFLOAT16, 128, 128, weightType, batch);
            }
        }
        std::vector<std::unique_ptr<fastllm::Data>> references;
        for (auto weightType : {fastllm::DataType::FP8_E4M3, fastllm::DataType::FP8_E4M3_BLOCK_128}) {
            for (int batch : {1, 2, 4, 6, FASTLLM_CUDA_MOE_CACHE_MAX_BATCH}) {
                CompareFP8<half>(fastllm::DataType::FLOAT16, weightType, references, batch);
                CompareFP8<__nv_bfloat16>(fastllm::DataType::BFLOAT16, weightType, references, batch);
            }
        }
        for (int nodes : {1, 2, 4}) {
            for (int batch : {1, 4, FASTLLM_CUDA_MOE_CACHE_MAX_BATCH}) {
                CompareNumaCache<float>(fastllm::DataType::FLOAT32, nodes, batch);
                CompareNumaCache<half>(fastllm::DataType::FLOAT16, nodes, batch);
                CompareNumaCache<__nv_bfloat16>(fastllm::DataType::BFLOAT16, nodes, batch);
                CompareNumaCache<float>(fastllm::DataType::FLOAT32, nodes, batch, 128, 64, true);
                CompareNumaCache<half>(fastllm::DataType::FLOAT16, nodes, batch, 128, 64, true);
                CompareNumaCache<__nv_bfloat16>(fastllm::DataType::BFLOAT16, nodes, batch, 128, 64, true);
            }
        }
        for (auto type : {fastllm::DataType::FP8_E4M3, fastllm::DataType::FP8_E4M3_BLOCK_128})
            for (int nodes : {1, 2, 4}) for (int batch : {1, 4}) {
                CompareNumaCache<float>(fastllm::DataType::FLOAT32, nodes, batch, 256, 128, false, false, type);
                CompareNumaCache<half>(fastllm::DataType::FLOAT16, nodes, batch, 256, 128, false, false, type);
                CompareNumaCache<__nv_bfloat16>(fastllm::DataType::BFLOAT16, nodes, batch, 256, 128, false, false, type);
            }
        CompareNumaCache<half>(fastllm::DataType::FLOAT16, 2, 4, 260, 132, false, false,
                              fastllm::DataType::FP8_E4M3, true);
        CompareNumaCache<half>(fastllm::DataType::FLOAT16, 2, 4, 260, 128, false, false,
                              fastllm::DataType::FP8_E4M3);
        CompareNumaCache<half>(fastllm::DataType::FLOAT16, 1, 4, 32, 17);
        CompareNumaCache<half>(fastllm::DataType::FLOAT16, 1, 4, 32, 17, true);
        CompareNumaCache<half>(fastllm::DataType::FLOAT16, 4, 4, 2560, 640);
        CompareNumaCache<half>(fastllm::DataType::FLOAT16, 4, 4, 2560, 640, true);
        CompareNumaCache<half>(fastllm::DataType::FLOAT16, 1, 4, 128, 16, true);
        CompareNumaCache<half>(fastllm::DataType::FLOAT16, 2, 4, 32, 64, true);
        CompareNumaCache<half>(fastllm::DataType::FLOAT16, 2, 4, 128, 16, true);
#ifdef USE_NUMAS
        for (bool planar : {false, true})
            CompareNumaCache<float>(fastllm::DataType::FLOAT32,
                fastllm::GetNumaConfig()->numaCnt, 1, 2560, 640, planar, true);
        for (auto type : {fastllm::DataType::FP8_E4M3, fastllm::DataType::FP8_E4M3_BLOCK_128})
            CompareNumaCache<float>(fastllm::DataType::FLOAT32,
                fastllm::GetNumaConfig()->numaCnt, 1, 2560, 640, false, true, type);
        fastllm::ClearNumasMoeRuntimeCache();
#endif
        std::puts("ALL_PASS"); return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
