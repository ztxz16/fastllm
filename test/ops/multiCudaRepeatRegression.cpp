#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "executor.h"
#include "fastllm.h"
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <vector>
#include <stdexcept>
using namespace fastllm;
static void Require(bool ok, const char *m) {
    if (!ok)
        throw std::runtime_error(m);
}
static void Cuda(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
struct Graph {
    std::vector<int> devices;
    std::vector<void *> graphs, execs, starts, ends, pinned;
    Graph(const std::vector<int> &d, const std::function<void()> &body) : devices(d) {
        graphs.resize(d.size());
        execs.resize(d.size());
        starts.resize(d.size());
        ends.resize(d.size());
        for (size_t i = 0; i < d.size(); ++i) {
            FastllmCudaSetDevice(d[i]);
            Cuda(cudaDeviceSynchronize());
            Require(FastllmCudaGraphPrepareCaptureDevice(), "prepare capture");
            starts[i] = FastllmCudaEventCreate();
            ends[i] = FastllmCudaEventCreate();
        }
        Require(FastllmCudaGraphMemoryPoolBegin(), "pool begin");
        for (size_t i = 0; i < d.size(); ++i) {
            FastllmCudaSetDevice(d[i]);
            Require(FastllmCudaGraphBeginCapture(), "capture begin");
            FastllmCudaEventRecordCurrentThread(starts[i]);
        }
        Require(MultiCudaGraphWorkersWaitEvents(d, starts), "worker join");
        FastllmCudaSetDevice(d[0]);
        body();
        Require(MultiCudaGraphWorkersRecordEvents(d, ends), "worker end");
        for (size_t i = 0; i < d.size(); ++i) {
            FastllmCudaSetDevice(d[i]);
            FastllmCudaCurrentThreadStreamWaitEvent(ends[i]);
            Require(FastllmCudaGraphEndCapture(&graphs[i]), "capture end");
        }
        Require(FastllmCudaGraphMemoryPoolEnd(pinned), "pool end");
        for (size_t i = 0; i < d.size(); ++i) {
            FastllmCudaSetDevice(d[i]);
            Require(FastllmCudaGraphInstantiate(graphs[i], &execs[i]), "instantiate");
        }
        FastllmCudaSetDevice(d[0]);
    }
    void Run() {
        Require(MultiCudaRunDeviceCallbacksEnqueueOnly(devices,
                                                       [&](int rank, int device) {
                                                           Require(FastllmCudaGraphLaunch(execs[rank]),
                                                                   "parallel graph launch");
                                                       }),
                "parallel graph dispatch");
        FastllmCudaSetDevice(devices[0]);
        Require(MultiCudaCurrentThreadWaitForWorker(devices[0]), "wait graph root");
    }
    ~Graph() {
        for (size_t i = 0; i < devices.size(); ++i) {
            FastllmCudaSetDevice(devices[i]);
            cudaDeviceSynchronize();
            FastllmCudaGraphExecDestroy(execs[i]);
            FastllmCudaGraphDestroy(graphs[i]);
            FastllmCudaEventDestroy(starts[i]);
            FastllmCudaEventDestroy(ends[i]);
        }
        FastllmCudaGraphMemoryPoolRelease(pinned);
        FastllmCudaSetDevice(devices[0]);
    }
};

static int CheckUnsupportedDtypes() {
    int cases = 0;
    for (auto type : {INT16, FP8_E4M3})
        for (int root : {0, 3, 7}) {
            const std::map<std::string, int> deviceMap{{"cuda:" + std::to_string(root), 1}};
            ApplyDeviceMap(deviceMap, 0, 1);
            for (const auto &dims : std::vector<std::vector<int>>{{1, 1, 4, 7}, {2, 3, 4, 7}})
                for (int axis : {0, 1, 2, 3})
                    for (int repeat : {1, 4}) {
                        Data input(type, dims), output(FLOAT32, {5}), fallback;
                        input.ToDevice(CUDA, std::vector<int>{root});
                        output.ToDevice(CUDA, std::vector<int>{root});
                        FastllmCudaSetDevice(root);
                        input.Allocate();
                        output.Allocate();
                        Require(GetPointerDeviceId(input.cudaData) == root, "wrong input device");
                        const auto inputStrides = input.strides;
                        const auto outputStrides = output.strides;
                        void *inputPtr = input.cudaData, *outputPtr = output.cudaData;
                        std::vector<unsigned char> src(input.GetBytes()), sentinel(output.GetBytes(), 0xa5);
                        FastllmCudaSetDevice(root);
                        Cuda(cudaMemcpy(outputPtr, sentinel.data(), sentinel.size(), cudaMemcpyHostToDevice));
                        for (int seed = 0; seed < 3; ++seed) {
                            for (size_t j = 0; j < src.size(); ++j)
                                src[j] = (j * 37 + 51 + int(type) * 11 + seed * 43) % 251 + 1;
                            Cuda(cudaMemcpy(inputPtr, src.data(), src.size(), cudaMemcpyHostToDevice));
                            Require(!MultiCudaRepeatToReplicated(input, axis, repeat, output),
                                    "unsupported repeat dtype accepted");
                            Require(input.cudaData == inputPtr && input.dataType == type && input.dims == dims &&
                                        input.strides == inputStrides && input.dataDevice == CUDA &&
                                        input.dataDeviceIds == std::vector<int>{root} && !input.multiDeviceData &&
                                        input.multiDeviceDatas.empty() && input.tpLayout == TP_LAYOUT_NONE,
                                    "rejected repeat changed input metadata");
                            Require(output.cudaData == outputPtr && output.dataType == FLOAT32 &&
                                        output.dims == std::vector<int>{5} && output.strides == outputStrides &&
                                        output.dataDevice == CUDA && output.dataDeviceIds == std::vector<int>{root} &&
                                        output.unitSize == 4 && output.unitSizeDiv == 1 && !output.multiDeviceData &&
                                        output.multiDeviceDatas.empty() && output.tpLayout == TP_LAYOUT_NONE,
                                    "rejected repeat changed output metadata");
                            std::vector<unsigned char> actual(sentinel.size()), inputCheck(src.size());
                            Cuda(cudaMemcpy(actual.data(), outputPtr, actual.size(), cudaMemcpyDeviceToHost));
                            Cuda(cudaMemcpy(inputCheck.data(), inputPtr, inputCheck.size(), cudaMemcpyDeviceToHost));
                            Require(actual == sentinel && inputCheck == src, "rejected repeat changed tensor bytes");

                            // Exercise the ordinary CUDA Repeat used by the model's fallback.
                            Repeat(input, axis, repeat, fallback);
                            auto expectedDims = dims;
                            expectedDims[axis] *= repeat;
                            Require(fallback.dims == expectedDims && fallback.dataType == type, "fallback metadata");
                            size_t slab = input.unitSize;
                            for (size_t j = axis; j < dims.size(); ++j) slab *= dims[j];
                            std::vector<unsigned char> expected(src.size() * repeat);
                            for (size_t o = 0; o < src.size() / slab; ++o)
                                for (int k = 0; k < repeat; ++k)
                                    std::memcpy(expected.data() + (o * repeat + k) * slab, src.data() + o * slab, slab);
                            actual.resize(expected.size());
                            Cuda(cudaMemcpy(actual.data(), fallback.cudaData, actual.size(), cudaMemcpyDeviceToHost));
                            Require(actual == expected, "fallback bytes differ from CPU oracle");
                            ++cases;
                        }
                    }
        }
    return cases;
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8)
        return 77;
    try {
        const std::vector<int> ranks{0, 1, 2, 3, 4, 5, 6, 7};
        SetThreads(8);
        SetCudaGraph(true);
        FastllmCudaSetNcclForceSync(false);
        MultiCudaSetPersistentAsyncDispatch(true);
        SetDeviceMap({{"multicuda:0,1,2,3,4,5,6,7", 1}});
        ApplyDeviceMap({{"multicuda:0,1,2,3,4,5,6,7", 1}}, 0, 1);
        FastllmMultiCudaSetDevice(ranks);
        const int rejectedCases = CheckUnsupportedDtypes();
        ApplyDeviceMap({{"multicuda:0,1,2,3,4,5,6,7", 1}}, 0, 1);
        int cases = 0;
        for (auto type : {BFLOAT16, FLOAT16, FLOAT32, INT8, INT32})
            for (int root : {0, 3, 7}) {
                std::vector<std::vector<int>> shapes{{1, 1, 1, 5120}, {1, 2, 1, 5120}, {1, 5, 1, 5120},
                                                     {1, 6, 1, 5120}, {1, 8, 1, 5120}, {2, 3, 4, 7}};
                for (auto dims : shapes)
                    for (int axis : {0, 1, 2, 3})
                        for (int repeat : {1, 4}) {
                            Data input(type, dims), output;
                            input.ToDevice(CUDA, std::vector<int>{root});
                            FastllmCudaSetDevice(root);
                            input.Allocate();
                            Require(GetPointerDeviceId(input.cudaData) == root, "wrong input device");
                            size_t bytes = input.GetBytes();
                            std::vector<unsigned char> src(bytes);
                            auto write = [&](int seed) {
                                for (size_t j = 0; j < bytes; ++j)
                                    src[j] = (j * 13 + j / 7 + seed * 43 + int(type) * 11) & 255;
                                FastllmCudaSetDevice(root);
                                Cuda(cudaMemcpy(input.cudaData, src.data(), bytes, cudaMemcpyHostToDevice));
                                FastllmCudaSetDevice(0);
                            };
                            auto op = [&] {
                                Require(MultiCudaRepeatToReplicated(input, axis, repeat, output),
                                        "repeat rejected");
                            };
                            write(0);
                            op();
                            for (int d : ranks) {
                                FastllmCudaSetDevice(d);
                                Cuda(cudaDeviceSynchronize());
                            }
                            FastllmCudaSetDevice(0);
                            Graph graph(ranks, op);
                            auto check = [&] {
                                auto expectedDims = dims;
                                expectedDims[axis] *= repeat;
                                Require(output.dims == expectedDims, "repeat shape");
                                size_t inner = 1;
                                for (size_t j = axis + 1; j < dims.size(); ++j)
                                    inner *= dims[j];
                                size_t slab = inner * dims[axis] * input.unitSize;
                                size_t outer = bytes / slab;
                                std::vector<unsigned char> expected(bytes * repeat), actual(bytes * repeat);
                                for (size_t o = 0; o < outer; ++o)
                                    for (int k = 0; k < repeat; ++k)
                                        std::memcpy(expected.data() + (o * repeat + k) * slab,
                                                    src.data() + o * slab, slab);
                                for (int d : ranks) {
                                    FastllmCudaSetDevice(d);
                                    Cuda(cudaDeviceSynchronize());
                                    auto *p = output.multiDeviceDatas.at(d);
                                    Cuda(cudaMemcpy(actual.data(), p->cudaData, actual.size(),
                                                    cudaMemcpyDeviceToHost));
                                    Require(actual == expected, "repeat bytes differ from CPU oracle");
                                }
                                FastllmCudaSetDevice(root);
                                std::vector<unsigned char> rootCheck(bytes);
                                Cuda(cudaMemcpy(rootCheck.data(), input.cudaData, bytes,
                                                cudaMemcpyDeviceToHost));
                                Require(rootCheck == src, "repeat modified input");
                                FastllmCudaSetDevice(0);
                                ++cases;
                            };
                            for (int seed = 1; seed <= 3; ++seed) {
                                write(seed);
                                op();
                                check();
                                write(seed + 3);
                                graph.Run();
                                check();
                            }
                        }
            }
        std::cout << "PASS " << cases
                  << " repeat/broadcast CPU-byte-oracle checks: 5 dtypes, roots 0/3/7, all axes, repeat 1/4, "
                     "changed-input eager/graph, production and unaligned shapes"
                  << std::endl;
        std::cout << "PASS " << rejectedCases
                  << " INT16/FP8_E4M3 rejection and CUDA fallback checks: unchanged input/output, "
                     "roots 0/3/7, both repeat paths, all axes, repeat 1/4, changed inputs"
                  << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
