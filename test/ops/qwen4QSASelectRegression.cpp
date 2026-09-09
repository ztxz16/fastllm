#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace fastllm;

static void CheckCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

static void ToGpu(Data &data) {
    if (data.cpuData == nullptr) data.Allocate();
    data.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}

static uint32_t Ordered(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}

static int CheckPrefill() {
    // The metadata-driven path deliberately retains the generic scoring
    // kernel. Compare full indices with it, including ties, causal tails,
    // partial tiles, all activation types and unsupported tile shapes.
    struct Shape { int rows, blocks, heads, dim; };
    const Shape shapes[] = {{15, 513, 4, 128}, {16, 513, 4, 128},
        {19, 2049, 4, 128}, {128, 12801, 4, 128}, {1024, 32768, 4, 128},
        {19, 2049, 3, 128}, {19, 2049, 8, 64}};
    constexpr int budget = 2048, ratio = 4, width = budget + ratio - 1;
    std::mt19937 generator(73);
    std::uniform_real_distribution<float> random(-1, 1);
    int checks = 0;
    for (DataType type : {FLOAT32, FLOAT16, BFLOAT16}) {
        for (const Shape &shape : shapes) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                std::vector<float> queries(shape.rows * shape.heads * shape.dim);
                std::vector<float> keys(shape.blocks * shape.dim);
                for (float &v : queries) v = random(generator);
                for (size_t i = 0; i < keys.size(); ++i) {
                    keys[i] = pattern == 0 ? random(generator) : pattern == 1 ? 0.0f
                        : i < (size_t)17 * shape.dim ? random(generator)
                        : keys[i % (17 * shape.dim)];
                }
                Data query(type, {shape.rows, 1, shape.heads, shape.dim}, queries);
                Data compressed(FLOAT32, {shape.blocks, shape.dim}, keys);
                Data scores(FLOAT32, {shape.rows, shape.blocks});
                Data selected(INT32, {shape.rows, budget / ratio});
                Data reference(INT32, {shape.rows, width});
                Data output(INT32, {shape.rows, width});
                Data meta(INT32, {1});
                for (Data *d : {&query, &compressed, &scores, &selected,
                                &reference, &output, &meta}) ToGpu(*d);
                for (int start : {0, shape.blocks * ratio - shape.rows}) {
                    CheckCuda(cudaMemcpy(meta.cudaData, &start, sizeof(start), cudaMemcpyHostToDevice));
                    if (!FastllmCudaQwen4QSASelectGraph(query, compressed,
                            (const int32_t*)meta.cudaData, scores, selected, reference,
                            shape.heads, shape.dim, budget, ratio) ||
                        !FastllmCudaQwen4QSASelect(query, compressed, output,
                            shape.blocks * ratio, shape.heads, shape.dim,
                            budget, ratio, start)) {
                        throw std::runtime_error("prefill QSA launch rejected");
                    }
                    CheckCuda(cudaStreamSynchronize(cudaStreamPerThread));
                    std::vector<int32_t> expected(shape.rows * width), actual(expected.size());
                    CheckCuda(cudaMemcpy(expected.data(), reference.cudaData,
                        expected.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                    CheckCuda(cudaMemcpy(actual.data(), output.cudaData,
                        actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                    if (actual != expected) {
                        throw std::runtime_error("prefill QSA indices differ from generic scoring");
                    }
                    checks += shape.rows;
                }
            }
        }
    }
    return checks;
}

// Dense TP verification uses ascending indices, unlike QSA's already-causal
// selection. Check the actual gathered values and mask across graph replays,
// including a shrinking cache and a subset of query rows.
static int CheckDenseVerifierGather() {
    int checks = 0;
    constexpr int capacity = 16, queryHeads = 4, keyHeads = 2;
    for (DataType type : {FLOAT32, FLOAT16, BFLOAT16}) {
        for (int dim : {7, 128}) { // Scalar and vectorized FP16 gather.
            for (int sequence : {1, 2, 3, 4, 9}) {
                std::vector<float> values(keyHeads * capacity * dim);
                for (size_t i = 0; i < values.size(); ++i) values[i] = (i % 31 + 1) / 8.0f;
                Data query(type, {queryHeads, sequence, dim},
                    std::vector<float>(queryHeads * sequence * dim, 1));
                Data key(type, {keyHeads, capacity, dim}, values);
                Data value(type, {keyHeads, capacity, dim}, values);
                Data indices(INT32, {sequence, capacity}), meta(INT32, {1});
                indices.Allocate();
                auto ids = reinterpret_cast<int32_t *>(indices.cpuData);
                for (int i = 0; i < sequence * capacity; ++i)
                    ids[i] = i % capacity == capacity - 1 ? -1 : i % capacity;
                for (Data *d : {&query, &key, &value, &indices, &meta}) ToGpu(*d);
                for (int start : {0, sequence > 1 ? 1 : 0}) {
                    const int rows = sequence - start;
                    Data packed(type, {rows * queryHeads, 1, dim});
                    Data compactKey(type, {rows * keyHeads, capacity, dim});
                    Data compactValue(type, {rows * keyHeads, capacity, dim});
                    Data mask(type, {rows, 1, capacity});
                    for (Data *d : {&packed, &compactKey, &compactValue, &mask}) ToGpu(*d);
                    int previous = 0;
                    CheckCuda(cudaMemcpy(meta.cudaData, &previous, sizeof(previous), cudaMemcpyHostToDevice));
                    auto launch = [&] {
                        if (!FastllmCudaQwen4PrepareSparseBatchGraph(query, key, value,
                                indices, (const int32_t *)meta.cudaData, sequence,
                                packed, compactKey, compactValue, mask, start, rows))
                            throw std::runtime_error("dense verifier gather rejected");
                    };
                    launch();
                    CheckCuda(cudaStreamSynchronize(cudaStreamPerThread));
                    cudaGraph_t graph;
                    cudaGraphExec_t executable;
                    CheckCuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                    launch();
                    CheckCuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    CheckCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                    for (int base : {0, capacity - sequence - 1, 1}) {
                        previous = base;
                        CheckCuda(cudaMemcpyAsync(meta.cudaData, &previous, sizeof(previous),
                            cudaMemcpyHostToDevice, cudaStreamPerThread));
                        CheckCuda(cudaGraphLaunch(executable, cudaStreamPerThread));
                        CheckCuda(cudaStreamSynchronize(cudaStreamPerThread));
                        std::vector<float> expectedMask(rows * capacity, 1);
                        std::vector<float> expectedKV(rows * keyHeads * capacity * dim, 0);
                        for (int row = 0; row < rows; ++row) {
                            for (int token = 0; token < base + start + row + 1; ++token) {
                                expectedMask[row * capacity + token] = 0;
                                for (int head = 0; head < keyHeads; ++head)
                                    for (int c = 0; c < dim; ++c)
                                        expectedKV[((row * keyHeads + head) * capacity + token) * dim + c] =
                                            values[(head * capacity + token) * dim + c];
                            }
                        }
                        auto compare = [&](const Data &actual, const std::vector<float> &expected) {
                            Data reference(type, actual.dims, expected);
                            std::vector<unsigned char> bytes(actual.GetBytes());
                            CheckCuda(cudaMemcpy(bytes.data(), actual.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
                            if (std::memcmp(bytes.data(), reference.cpuData, bytes.size()))
                                throw std::runtime_error("dense verifier sees a future/padded token or loses a valid KV");
                        };
                        compare(mask, expectedMask);
                        compare(compactKey, expectedKV);
                        compare(compactValue, expectedKV);
                        checks += rows;
                    }
                    CheckCuda(cudaGraphExecDestroy(executable));
                    CheckCuda(cudaGraphDestroy(graph));
                }
            }
        }
    }
    return checks;
}

// Use the GPU's scores as the reference input, so CPU/GPU dot-product
// rounding cannot obscure a selection or tie-breaking error. Reuse each
// captured graph while advancing and shrinking its visible context.
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: this regression requires a CUDA GPU\n";
        return 77;
    }
    try {
        FastllmCudaSetDevice(0);
        const int gatherChecks = CheckDenseVerifierGather();
        std::cout << "PASS: " << gatherChecks << " dense verifier rows preserve causal KV and padding across replays\n";
        constexpr int heads = 4, dim = 128, budget = 2048, ratio = 4;
        constexpr int selectedK = budget / ratio, width = budget + ratio - 1;
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> random(-1.0f, 1.0f);
        int checks = 0;
        for (int capacity : {512, 1281, 2048, 2049, 4096, 4097, 8192, 8193,
                             16384, 16385, 32768, 32769, 36864, 36865,
                             65536, 65537, 69632, 69633}) {
            for (int rows : {1, 4}) {
                for (int pattern = 0; pattern < 3; pattern++) {
                    std::vector<float> queries(rows * heads * dim);
                    std::vector<float> keys(capacity * dim);
                    for (float &value : queries) value = random(generator);
                    for (int block = 0; block < capacity; block++) {
                        for (int column = 0; column < dim; column++) {
                            keys[block * dim + column] = pattern == 0
                                ? random(generator) : pattern == 1 ? 0.0f
                                : keys[(block % 17) * dim + column];
                            if (pattern == 2 && block < 17) {
                                keys[block * dim + column] = random(generator);
                            }
                        }
                    }
                    Data query(FLOAT16, {rows, 1, heads, dim}, queries);
                    Data compressed(FLOAT32, {capacity, dim}, keys);
                    Data scores(FLOAT32, {rows, capacity});
                    Data selected(INT32, {rows, selectedK});
                    Data indices(INT32, {rows, width});
                    Data meta(INT32, {1});
                    for (Data *data : {&query, &compressed, &scores,
                                       &selected, &indices, &meta}) ToGpu(*data);
                    int previous = capacity * ratio - rows;
                    CheckCuda(cudaMemcpy(meta.cudaData, &previous, sizeof(previous),
                                         cudaMemcpyHostToDevice));
                    auto launch = [&] {
                        if (!FastllmCudaQwen4QSASelectGraph(query, compressed,
                                (const int32_t *)meta.cudaData, scores, selected,
                                indices, heads, dim, budget, ratio)) {
                            throw std::runtime_error("QSA launch rejected");
                        }
                    };
                    launch();
                    CheckCuda(cudaStreamSynchronize(cudaStreamPerThread));
                    cudaGraph_t graph;
                    cudaGraphExec_t executable;
                    CheckCuda(cudaStreamBeginCapture(cudaStreamPerThread,
                                                     cudaStreamCaptureModeThreadLocal));
                    launch();
                    CheckCuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    CheckCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                    std::vector<float> hostScores(rows * capacity);
                    std::vector<int32_t> hostSelected(rows * selectedK);
                    std::vector<int32_t> hostIndices(rows * width);
                    for (int base : {budget - ratio, capacity * ratio / 2,
                                     capacity * ratio - rows - ratio}) {
                        for (int tail = 0; tail < ratio; tail++) {
                            previous = base + tail;
                            CheckCuda(cudaMemcpyAsync(meta.cudaData, &previous,
                                sizeof(previous), cudaMemcpyHostToDevice, cudaStreamPerThread));
                            CheckCuda(cudaGraphLaunch(executable, cudaStreamPerThread));
                            CheckCuda(cudaStreamSynchronize(cudaStreamPerThread));
                            CheckCuda(cudaMemcpy(hostScores.data(), scores.cudaData,
                                hostScores.size() * sizeof(float), cudaMemcpyDeviceToHost));
                            CheckCuda(cudaMemcpy(hostSelected.data(), selected.cudaData,
                                hostSelected.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                            CheckCuda(cudaMemcpy(hostIndices.data(), indices.cudaData,
                                hostIndices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
                            for (int row = 0; row < rows; row++) {
                                const int length = previous + row + 1;
                                const int blocks = length / ratio;
                                std::vector<int32_t> expected(blocks);
                                std::iota(expected.begin(), expected.end(), 0);
                                std::stable_sort(expected.begin(), expected.end(),
                                    [&](int a, int b) {
                                        return Ordered(hostScores[row * capacity + a]) >
                                               Ordered(hostScores[row * capacity + b]);
                                    });
                                expected.resize(std::min(blocks, selectedK));
                                std::sort(expected.begin(), expected.end());
                                expected.resize(selectedK, -1);
                                if (!std::equal(expected.begin(), expected.end(),
                                                hostSelected.begin() + row * selectedK)) {
                                    throw std::runtime_error("selected block ids differ from stable Top-K");
                                }
                                const int selectedTokens = std::min(blocks, selectedK) * ratio;
                                for (int column = 0; column < width; column++) {
                                    const int tailIndex = column - selectedTokens;
                                    const int wanted = column < selectedTokens
                                        ? expected[column / ratio] * ratio + column % ratio
                                        : tailIndex < length % ratio
                                            ? blocks * ratio + tailIndex : -1;
                                    if (hostIndices[row * width + column] != wanted) {
                                        throw std::runtime_error("expanded indices or tail padding differ");
                                    }
                                }
                                checks++;
                            }
                        }
                    }
                    CheckCuda(cudaGraphExecDestroy(executable));
                    CheckCuda(cudaGraphDestroy(graph));
                }
            }
        }
        const int prefillChecks = CheckPrefill();
        std::cout << "PASS: " << prefillChecks
                  << " prefill QSA rows match the generic scoring path\n";
        std::cout << "PASS: " << checks
                  << " QSA rows match stable Top-K and tail indices across graph replays\n";
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
