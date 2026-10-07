#pragma once
// Included inside grouped_moe. GLM keeps one positive FP32 Q8_K scale per
// 256 values, nearest-even quants, and no V4.1 block-32 FP8 conversion.
__global__ void QuantizeGlm5(const __nv_bfloat16 *input, block_q8_K *output, int columns) {
    __shared__ float maxima[8];
    const int c = threadIdx.x;
    const float x = __bfloat162float(input[size_t(blockIdx.y)*columns + blockIdx.x*256 + c]);
    float amax = fabsf(x);
    for (int mask = 16; mask; mask >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
    if (c%32 == 0) maxima[c/32] = amax;
    __syncthreads();
    amax = 0;
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, maxima[i]);
    auto &q = output[size_t(blockIdx.y)*(columns/256) + blockIdx.x];
    const float inverse = amax == 0 ? 0 : __fdiv_rn(127.f, amax);
    q.qs[c] = __float2int_rn(__fmul_rn(x, inverse));
    if (c == 0) q.d = __fdiv_rn(amax, 127.f);
    // Gather consumes only d/qs; bsums are not used by these symmetric IQ tiles.
}

static void RunGlm5(const __nv_bfloat16 *input, __nv_bfloat16 *gate, __nv_bfloat16 *output,
                    const uint8_t *const *weights, const int *indices, const float *scores,
                    Workspace &w, int gateType, int downType, int rows, int hidden, int inter,
                    int experts, int topk, float limit) {
    const auto stream = cudaStreamPerThread;
    const int routes = rows*topk;
    auto *q = reinterpret_cast<block_q8_K *>(w.products);
    QuantizeGlm5<<<dim3(hidden/256, rows), 256, 0, stream>>>(input, q, hidden);
    GatherV41<<<dim3((w.capacity+7)/8, hidden/128), 256, 0, stream>>>(
        q, w.quantized, w.groupRoutes, w.offsets+experts, hidden, w.capacity, topk);
    Matrix(gateType, weights, 0, w, experts, hidden, 2*inter, stream);
    // GLM and V4.1 share BF16 projection rounding, asymmetric clamp,
    // score-before-down and ascending-expert output reduction.
    ActivateV41<<<(routes*inter+255)/256, 256, 0, stream>>>(w.products, gate,
        w.routeGroups, scores, routes, inter, limit);
    QuantizeGlm5<<<dim3(inter/256, routes), 256, 0, stream>>>(gate, q, inter);
    GatherV41<<<dim3((w.capacity+7)/8, inter/128), 256, 0, stream>>>(
        q, w.quantized, w.groupRoutes, w.offsets+experts, inter, w.capacity, 1);
    Matrix(downType, weights, 1, w, experts, inter, hidden, stream);
    ReduceV41<<<(rows*hidden+255)/256, 256, 0, stream>>>(w.products, output,
        w.routeGroups, indices, rows, hidden, topk);
}
