#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-dsa-layer.cuh"
#include "devices/cuda/naive-n05-kv.cuh"
#include "naive-n05-dsa-fused-attention.cuh"
#include "naivert/attn_dsa/body.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

using Body = fastllm_dsa_fused::Body<false, false>;
using Layout = Body::Scratch;

__global__ __launch_bounds__(384, 1) void Select(unsigned char *workspace, const unsigned *first_hist) {
    extern __shared__ __align__(1024) unsigned char bytes[];
    auto &storage = *reinterpret_cast<Body::SharedStorage *>(bytes);
    auto *grid = reinterpret_cast<unsigned *>(workspace + Layout::kTopKGridOffset);
    Body::zero_topk_grid(grid);
    cooperative_groups::this_grid().sync();
    if (first_hist) {
        for (int i = blockIdx.x * 384 + threadIdx.x; i < 8 * 256; i += gridDim.x * 384)
            Body::topk_grid_hist(grid, i / 256, 0)[i % 256] = first_hist[i];
        cooperative_groups::this_grid().sync();
    }
    Body::select_per_query_topk_grid(reinterpret_cast<float *>(workspace + Layout::kIndexLogitsOffset),
                                     reinterpret_cast<int *>(workspace + Layout::kIndexSelectsOffset), grid,
                                     reinterpret_cast<int *>(workspace + Layout::kRadixControlOffset),
                                     reinterpret_cast<unsigned *>(&storage), &storage.topk_sort,
                                     first_hist != nullptr);
}

void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
unsigned Ordered(float value) {
    unsigned bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

int main() {
    try {
        cudaDeviceProp prop{};
        Check(cudaGetDeviceProperties(&prop, 0));
        if (prop.major != 10 || prop.minor != 3 || !prop.cooperativeLaunch)
            return 77;
        Check(cudaFuncSetAttribute(Select, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   sizeof(Body::SharedStorage)));
        unsigned char *workspace = nullptr;
        Check(cudaMalloc(&workspace, Layout::kBytes));
        Check(cudaMemset(workspace, 0, Layout::kBytes));
        unsigned *cpu_histogram = nullptr;
        Check(cudaMalloc(&cpu_histogram, 8 * 256 * sizeof(unsigned)));
        const unsigned *first_hist = nullptr;
        void *args[] = {&workspace, &first_hist};
        auto launch = [&] {
            Check(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(Select),
                                              dim3(prop.multiProcessorCount), dim3(384), args,
                                              sizeof(Body::SharedStorage), cudaStreamPerThread));
        };
        cudaGraph_t graphs[2] = {};
        cudaGraphExec_t executables[2] = {};
        int cases = 0;
        for (int visible : {2049, 8192, 16384, 16391, 32768, 65536}) {
            for (int pattern = 0; pattern < 4; ++pattern) {
                std::vector<float> logits(8 * Layout::kMaxSeqLen, 0.f);
                std::vector<unsigned> histogram(8 * 256);
                int lengths[9];
                lengths[8] = visible;
                std::vector<int> expected(8 * 2048, -1);
                for (int query = 0; query < 8; ++query) {
                    lengths[query] = visible - query;
                    float *row = logits.data() + query * Layout::kMaxSeqLen;
                    for (int token = 0; token < lengths[query]; ++token) {
                        unsigned hash = unsigned(token + 17 * query) * 1664525u + 1013904223u;
                        int order = (uint64_t(token) * 65537u) % lengths[query];
                        if (pattern == 0)
                            row[token] = float(int(hash >> 16) - 32768) / 127.f;
                        if (pattern == 1)
                            row[token] = 0.f; // candidate overflow / full-pass fallback
                        if (pattern == 2)     // compact path, many threshold ties, scattered token order
                            row[token] = order < 1200 + 100 * query   ? 2.f
                                         : order < 7200 + 100 * query ? 1.f
                                                                      : 0.f;
                        if (pattern == 3)
                            row[token] = float(hash & 4095) * 0.00001f - 0.02f;
                    }
                    for (int token = 0; token < lengths[query]; ++token)
                        ++histogram[query * 256 + (Ordered(row[token]) >> 24)];
                    std::vector<int> tokens(lengths[query]);
                    std::iota(tokens.begin(), tokens.end(), 0);
                    const int count = std::min(lengths[query], 2048);
                    std::partial_sort(tokens.begin(), tokens.begin() + count, tokens.end(),
                                      [&](int a, int b) {
                                          unsigned x = Ordered(row[a]), y = Ordered(row[b]);
                                          return x == y ? a < b : x > y;
                                      });
                    std::sort(tokens.begin(), tokens.begin() + count);
                    std::copy_n(tokens.begin(), count, expected.begin() + query * 2048);
                }
                Check(cudaMemcpy(workspace + Layout::kIndexLogitsOffset, logits.data(),
                                 logits.size() * sizeof(float), cudaMemcpyHostToDevice));
                Check(cudaMemcpy(workspace + Layout::kRadixControlOffset, lengths, sizeof(lengths),
                                 cudaMemcpyHostToDevice));
                Check(cudaMemcpy(cpu_histogram, histogram.data(), 8 * 256 * sizeof(unsigned),
                                 cudaMemcpyHostToDevice));
                for (int ready = 0; ready < 2; ++ready) {
                    first_hist = ready ? cpu_histogram : nullptr;
                    auto &graph = graphs[ready];
                    auto &executable = executables[ready];
                    if (!executable) {
                        launch();
                        Check(cudaStreamSynchronize(cudaStreamPerThread));
                        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                        launch();
                        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                        Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                    }
                    for (int repeat = 0; repeat < 2; ++repeat) {
                        Check(cudaMemset(workspace + Layout::kIndexSelectsOffset, 0xff,
                                         8 * 2048 * sizeof(int)));
                        Check(cudaGraphLaunch(executable, cudaStreamPerThread));
                        Check(cudaStreamSynchronize(cudaStreamPerThread));
                        std::vector<int> actual(8 * 2048);
                        Check(cudaMemcpy(actual.data(), workspace + Layout::kIndexSelectsOffset,
                                         actual.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        for (int query = 0; query < 8; ++query)
                            for (int i = 0; i < std::min(lengths[query], 2048); ++i)
                                if (actual[query * 2048 + i] != expected[query * 2048 + i]) {
                                    std::fprintf(stderr,
                                                 "Mismatch visible=%d pattern=%d first_ready=%d query=%d "
                                                 "slot=%d got=%d expected=%d\n",
                                                 visible, pattern, ready, query, i, actual[query * 2048 + i],
                                                 expected[query * 2048 + i]);
                                    throw std::runtime_error("exact Top-K mismatch");
                                }
                    }
                    ++cases;
                    std::printf("TOPK visible=%d pattern=%d first_ready=%d graph_replays=2 PASS\n", visible,
                                pattern, ready);
                }
            }
        }
        for (int ready = 0; ready < 2; ++ready) {
            Check(cudaGraphExecDestroy(executables[ready]));
            Check(cudaGraphDestroy(graphs[ready]));
        }
        Check(cudaFree(cpu_histogram));
        Check(cudaFree(workspace));
        std::printf("DSA TOPK PASS cases=%d\n", cases);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
