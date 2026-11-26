#pragma once

namespace fastllm {
struct InklingWindowKernel {
    void *handle = nullptr;
    int shared = 0;
};
InklingWindowKernel PrepareInklingWindowKernel(int heads, int group, int window, int extent, int page,
                                              bool decode = false, int splits = 1, bool mappedPages = false);
}

extern "C" void *FastllmCudaLoadWindowVerify(const char *cubin, const char *name, int shared);
extern "C" bool FastllmCudaLaunchWindowVerify(fastllm::InklingWindowKernel kernel, int tiles, int kvHeads,
    int requests, const void *visits, const void *q, const void *bias, const void *sink, const void *gate,
    void *output, int visitBytes, int pastOffset, int chain, int sinkStride, int gateStride);
