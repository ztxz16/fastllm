#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
__global__ void forceDraft(const int *ctrl, int *out, int stride, int i) {
    if (blockIdx.x < ctrl[0]) out[blockIdx.x * stride] = 128 + i * 13 + (i >= ctrl[1] ? ctrl[2] : 0);
}
__global__ void forcePrefix(int *ctrl, const int *tokens, int *result, int batch, int drafts, const int *live) {
    const int b = blockIdx.x;
    if (!b && batch > 1) atomicAdd(ctrl + 7, 1);
    if (ctrl[0] != batch) return;
    const int prefix = min(b < 2 && ctrl[4 + b] >= 0 ? ctrl[4 + b] : ctrl[1], live[b]);
    for (int i = 0; i < prefix; ++i) result[b * (drafts + 1) + i] = tokens[b * drafts + i];
    result[b * (drafts + 1) + prefix] = 512;
    result[batch * (drafts + 1) + b] = prefix;
}
} // namespace

void forceInklingDraft(const int *ctrl, int *out, int stride, int step, int capacity) {
    forceDraft<<<capacity, 1, 0, cudaStreamPerThread>>>(ctrl, out, stride, step);
}
void forceInklingPrefix(int *ctrl, const int *tokens, int *result, int batch, int drafts, const int *live) {
    forcePrefix<<<batch, 1, 0, cudaStreamPerThread>>>(ctrl, tokens, result, batch, drafts, live);
}

namespace {
template<class T> __global__ void snapshot(const int *ctrl, const T *base, int baseStride, const T *adjust, const T *bias,
                                          float cap, const float *q, int stride, float *saved, int step,
                         int vocab) {
    const int row = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!ctrl[3] || row >= ctrl[0] || i >= vocab) return;
    size_t offset = (size_t(row) * 7 + step) * 2 * vocab + i;
    const T corr = T(float(adjust[size_t(row) * vocab + i]) + float(bias[i]));
    const T sum = T(float(base[size_t(row) * baseStride + i]) + float(corr));
    saved[offset] = float(T(cap * tanhf(float(sum) / cap)));
    saved[offset + vocab] = q[size_t(row) * stride + i];
}
} // namespace
void snapshotInklingDraft(const int *ctrl, const void *base, int baseStride, const void *adjust, const void *bias,
                          float cap, bool bf16, const float *q, int stride, float *saved, int step, int vocab, int capacity) {
    const dim3 grid((vocab + 255) / 256, capacity);
    auto launch = [&](auto typed) {
        using T = decltype(typed);
        snapshot<<<grid, 256, 0, cudaStreamPerThread>>>(ctrl, static_cast<T>(base), baseStride, typed,
            static_cast<T>(bias), cap, q, stride, saved, step, vocab);
    };
    if (bf16) launch(static_cast<const __nv_bfloat16 *>(adjust));
    else launch(static_cast<const half *>(adjust));
}
