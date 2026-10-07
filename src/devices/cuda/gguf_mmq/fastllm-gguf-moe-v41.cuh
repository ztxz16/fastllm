#pragma once
// Included inside grouped_moe. V4.1 uses signed-max GGML Q8_K activations,
// preceded by a BF16 / E4M3 block-32 boundary at both expert projections.
// Keep those quants and FP32 scales when gathering into MMQ tiles.

__global__ void QuantizeV41(const __nv_bfloat16 *input, block_q8_K *output, int columns) {
    const int c = threadIdx.x;
    float x = __bfloat162float(input[size_t(blockIdx.y)*columns+blockIdx.x*256+c]);
    float amax = fmaxf(1e-4f, fabsf(x));
    for (int m = 16; m; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
    const unsigned bits = __float_as_uint(amax/448.f);
    const int exponent = int((bits>>23)&255)-127+((bits&0x7fffff) != 0);
    const float scale = exp2f(float(exponent));
    x = float(__nv_fp8_e4m3(fmaxf(-448.f, fminf(448.f, x/scale))))*scale;
    fastllm::cuda::v41_gguf::QuantizeQ8K(x,
        output[size_t(blockIdx.y) * (columns / 256) + blockIdx.x]);
}

__global__ void GatherV41(const block_q8_K *input, block_q8_1_mmq *output,
                          const int *groupRoutes, const int *activeRows,
                          int columns, int capacity, int topk) {
    const int row = blockIdx.x*(blockDim.x/32)+threadIdx.x/32, lane = threadIdx.x%32;
    if (row >= *activeRows) return;
    const int route = groupRoutes[row], halfBlock = blockIdx.y%2;
    const block_q8_K *q = route < 0 ? nullptr : input+size_t(route/topk)*(columns/256)+blockIdx.y/2;
    auto &dst = output[size_t(blockIdx.y)*capacity+row];
    for (int j = 0; j < 4; ++j) {
        dst.qs[j*32+lane] = q ? q->qs[halfBlock*128+j*32+lane] : 0;
    }
    const float scale = q ? q->d : 0;
    // V4.1 tiles consume FP32 Q8_K scales and form exact integer sums by
    // MMA. Do not round scale/sum metadata to the ordinary D2S6/DS4 halves.
    if (lane < 4) dst.d4[lane] = scale;
}

__global__ void ActivateV41(const float *products, __nv_bfloat16 *gate,
                            const int *routeGroups, const float *scores,
                            int routes, int inter, float limit) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= routes*inter) return;
    const int row = routeGroups[i/inter], col = i%inter;
    float g = 0, u = 0;
    if (row >= 0) {
        g = __bfloat162float(__float2bfloat16_rn(products[size_t(row)*2*inter+col]));
        u = __bfloat162float(__float2bfloat16_rn(products[size_t(row)*2*inter+inter+col]));
    }
    if (limit > 0) { g = fminf(g, limit); u = fmaxf(-limit, fminf(u, limit)); }
    const float h = __fmul_rn(g/(1.f+expf(-g)), u);
    gate[i] = __float2bfloat16_rn(row < 0 ? 0.f : __fmul_rn(scores[i/inter], h));
}

__global__ void ReduceV41(const float *products, __nv_bfloat16 *output,
                          const int *routeGroups, const int *indices,
                          int rows, int hidden, int topk) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= rows*hidden) return;
    const int base = (i/hidden)*topk;
    // NUMA sums in ascending expert-ID order; stable on repeated routes.
    int order[32];
    for (int k = 0; k < topk; ++k) {
        int pos = k;
        while (pos && indices[base+order[pos-1]] > indices[base+k]) {
            order[pos] = order[pos-1]; --pos;
        }
        order[pos] = k;
    }
    float sum = 0;
    for (int k = 0; k < topk; ++k) {
        const int route = base+order[k];
        const int group = routeGroups ? routeGroups[route] : route;
        if (group >= 0) sum = __fadd_rn(sum, __bfloat162float(__float2bfloat16_rn(
            products[size_t(group)*hidden+i%hidden])));
    }
    output[i] = __float2bfloat16_rn(sum);
}

static void RunV41(const __nv_bfloat16 *input, __nv_bfloat16 *gate, __nv_bfloat16 *output,
                   const uint8_t *const *weights, const int *indices, const float *scores,
                   Workspace &w, int rows, int hidden, int inter, int experts, int topk,
                   float limit, cudaEvent_t downWeightsReady) {
    const auto stream = cudaStreamPerThread;
    const int routes = rows*topk;
    // The product workspace is unused until Matrix, so it can hold packed
    // activations without retaining a second large temporary allocation.
    auto *q = reinterpret_cast<block_q8_K *>(w.products);
    QuantizeV41<<<dim3(hidden/256, rows), 256, 0, stream>>>(input, q, hidden);
    GatherV41<<<dim3((w.capacity+7)/8, hidden/128), 256, 0, stream>>>(
        q, w.quantized, w.groupRoutes, w.offsets+experts, hidden, w.capacity, topk);
    Matrix(GGML_TYPE_Q2_K, weights, 0, w, experts, hidden, 2*inter, stream);
    ActivateV41<<<(routes*inter+255)/256, 256, 0, stream>>>(w.products, gate,
        w.routeGroups, scores, routes, inter, limit);
    QuantizeV41<<<dim3(inter/256, routes), 256, 0, stream>>>(gate, q, inter);
    GatherV41<<<dim3((w.capacity+7)/8, inter/128), 256, 0, stream>>>(
        q, w.quantized, w.groupRoutes, w.offsets+experts, inter, w.capacity, 1);
    if (downWeightsReady) CUDA_CHECK(cudaStreamWaitEvent(stream, downWeightsReady, 0));
    Matrix(GGML_TYPE_Q4_K, weights, 1, w, experts, inter, hidden, stream);
    ReduceV41<<<(rows*hidden+255)/256, 256, 0, stream>>>(w.products, output,
        w.routeGroups, indices, rows, hidden, topk);
}
