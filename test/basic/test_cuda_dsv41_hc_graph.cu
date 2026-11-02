#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cstdint>
using namespace fastllm;
static void Require(bool x, const char *m) {
    if (!x)
        throw std::runtime_error(m);
}
static void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static void Fill(Data &d, int seed, float factor) {
    d.ToDevice(CPU);
    d.Allocate(false);
    for (size_t i = 0; i < d.Count(0); ++i) {
        float f = std::sin((i % 100003 + seed) * .1723f) * factor;
        if (d.dataType == FLOAT32)
            ((float *)d.cpuData)[i] = f;
        else {
            f = RoundFloat32ToBFloat16RNE(f);
            uint32_t b;
            memcpy(&b, &f, 4);
            ((uint16_t *)d.cpuData)[i] = b >> 16;
        }
    }
    d.ToDevice(CUDA);
}
__global__ void Work(const __nv_bfloat16 *x, float *out) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= 5120)
        return;
    float v = __bfloat162float(x[i]);
    for (int j = 0; j < 256; ++j)
        v = __fmaf_rn(v, .9999f, .00001f);
    out[i] = v;
}
__global__ void Consume(const float *p, const float *post, const float *c, const float *mid, float *out) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < 5120)
        out[i] = (mid[i] + p[i % 4]) + post[i % 4] + c[i % 16];
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE");
            return 77;
        }
        cudaDeviceProp prop{};
        Check(cudaGetDeviceProperties(&prop, 0));
        if (prop.major != 12 || prop.minor != 0) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: mHC graph optimization requires SM120");
            return 77;
        }
        FastllmCudaSetDevice(0);
        int comparisons = 0;
        for (bool quant : {false, true}) {
            Data x(BFLOAT16, {1, 1, 4, 5120}), fn(FLOAT32, {24, 20480}), prev(FLOAT32, {1, 1, 4}),
                scale(FLOAT32, {3}), base(FLOAT32, {24}), norm(FLOAT32, {5120});
            Data p, o, c, y, q, mid(FLOAT32, {5120}), out(FLOAT32, {5120}), saved(FLOAT32, {32});
            Fill(x, 0, 1);
            Fill(fn, 11, .1);
            Fill(prev, 3, .8);
            Fill(scale, 5, .3);
            Fill(base, 7, .5);
            Fill(norm, 9, 1);
            Fill(mid, 0, 0);
            Fill(out, 0, 0);
            Fill(saved, 0, 0);
            auto run = [&] {
                Require(FastllmCudaDeepSeekV41HcMixPreNorm(x, prev, fn, scale, base, norm, 20, 1e-6, 1e-6, p,
                                                           o, c, y, quant ? &q : nullptr),
                        "mix");
            };
            run();
            Check(cudaDeviceSynchronize());
            Require(FastllmCudaGraphMemoryPoolBegin(), "pool begin");
            std::vector<void *> reserved;
            void *children[3] = {};
            Require(FastllmCudaGraphBeginCapture(), "capture");
            run();
            Require(FastllmCudaGraphEndCapture(&children[0]), "capture end");
            // Add a deliberate overwrite after Finish in the original segment. This
            // catches accidental borrowing of temporary mix storage by the side branch.
            auto child = (cudaGraph_t)children[0];
            size_t n = 0;
            Check(cudaGraphGetNodes(child, nullptr, &n));
            std::vector<cudaGraphNode_t> nodes(n);
            Check(cudaGraphGetNodes(child, nodes.data(), &n));
            cudaGraphNode_t finish = nullptr;
            const float *mix = nullptr;
            for (auto node : nodes) {
                cudaGraphNodeType type;
                Check(cudaGraphNodeGetType(node, &type));
                if (type != cudaGraphNodeTypeKernel)
                    continue;
                cudaKernelNodeParams a{};
                Check(cudaGraphKernelNodeGetParams(node, &a));
                if (a.blockDim.x == 32 && a.gridDim.x == 1) {
                    Require(!finish, "unique finish");
                    finish = node;
                    mix = *(const float **)a.kernelParams[0];
                }
            }
            Require(finish && mix, "finish present");
            cudaMemsetParams mp{};
            mp.dst = (void *)mix;
            mp.value = 0x7fc00000;
            mp.elementSize = 4;
            mp.width = 25;
            mp.height = 1;
            cudaGraphNode_t overwrite;
            Check(cudaGraphAddMemsetNode(&overwrite, child, &finish, 1, &mp));
            Require(FastllmCudaGraphBeginCapture(), "work capture");
            Work<<<20, 256>>>((__nv_bfloat16 *)y.cudaData, (float *)mid.cudaData);
            Require(FastllmCudaGraphEndCapture(&children[1]), "work end");
            Require(FastllmCudaGraphBeginCapture(), "consume capture");
            Consume<<<20, 256>>>((float *)p.cudaData, (float *)o.cudaData, (float *)c.cudaData,
                                 (float *)mid.cudaData, (float *)out.cudaData);
            Require(FastllmCudaGraphEndCapture(&children[2]), "consume end");
            Require(FastllmCudaGraphMemoryPoolEnd(reserved), "pool end");
            void *serial = nullptr, *serialExec = nullptr, *parallel = nullptr, *parallelExec = nullptr;
            int start = 0, join = 2;
            Require(FastllmCudaGraphCompose(children, 3, &serial, &serialExec), "serial compose");
            Require(FastllmCudaV41ComposeHcGraph(children, 3, &start, &join, 1, (float *)saved.cudaData,
                                                 &parallel, &parallelExec),
                    "parallel compose");
            for (int seed = 0; seed < 32; ++seed) {
                // Copy dynamic inputs without moving the graph's allocation.
                std::vector<uint16_t> host(20480);
                for (int i = 0; i < 20480; ++i) {
                    float v = RoundFloat32ToBFloat16RNE(std::sin((i + seed * 71) * .13f) *
                                                        (seed % 3 ? 1.f : 1e-3f));
                    uint32_t bits;
                    memcpy(&bits, &v, 4);
                    host[i] = bits >> 16;
                }
                Check(cudaMemcpy(x.cudaData, host.data(), host.size() * 2, cudaMemcpyHostToDevice));
                Require(FastllmCudaGraphLaunch(serialExec), "serial launch");
                Check(cudaDeviceSynchronize());
                std::vector<unsigned char> reference;
                std::vector<Data *> outputs = {&p, &o, &c, &y, &out};
                if (quant)
                    outputs.push_back(&q);
                for (Data *a : outputs) {
                    size_t offset = reference.size();
                    reference.resize(offset + a->GetBytes());
                    Check(cudaMemcpy(reference.data() + offset, a->cudaData, a->GetBytes(),
                                     cudaMemcpyDeviceToHost));
                }
                for (int replay = 0; replay < 16; ++replay) {
                    Require(FastllmCudaGraphLaunch(parallelExec), "parallel launch");
                    Check(cudaDeviceSynchronize());
                    size_t offset = 0;
                    for (Data *a : outputs) {
                        std::vector<unsigned char> actual(a->GetBytes());
                        Check(cudaMemcpy(actual.data(), a->cudaData, a->GetBytes(), cudaMemcpyDeviceToHost));
                        Require(!memcmp(actual.data(), reference.data() + offset, actual.size()),
                                "deferred mHC mismatch");
                        offset += actual.size();
                    }
                    ++comparisons;
                }
            }
            // Rejection must not mutate the original graphs or return partial graphs.
            void *bad = nullptr, *badExec = nullptr;
            int invalidJoin = 0;
            Require(!FastllmCudaV41ComposeHcGraph(children, 3, &start, &invalidJoin, 1,
                                                  (float *)saved.cudaData, &bad, &badExec) &&
                        !bad && !badExec,
                    "bad join rejection");
            int missingStart = 1;
            Require(!FastllmCudaV41ComposeHcGraph(children, 3, &missingStart, &join, 1,
                                                  (float *)saved.cudaData, &bad, &badExec) &&
                        !bad && !badExec,
                    "missing finish rejection");
            Require(FastllmCudaGraphLaunch(serialExec), "fallback serial");
            Check(cudaDeviceSynchronize());
            FastllmCudaGraphExecDestroy(parallelExec);
            FastllmCudaGraphDestroy(parallel);
            FastllmCudaGraphExecDestroy(serialExec);
            FastllmCudaGraphDestroy(serial);
            for (auto child : children)
                FastllmCudaGraphDestroy(child);
            FastllmCudaGraphMemoryPoolRelease(reserved);
        }
        std::printf("HC_GRAPH_PASS %d dynamic-input replays, pre/post/comb/normalization/FP8/consumer "
                    "byte-exact, scratch overwrite and fallback covered\n",
                    comparisons);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
