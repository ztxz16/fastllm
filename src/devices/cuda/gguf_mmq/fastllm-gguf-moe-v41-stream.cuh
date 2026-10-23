#pragma once
// Included inside grouped_moe. Native Q8_1 preserves the CUDA per-expert
// FP8 block-32 / BF16 boundaries, including IQ1_S's FP16 activation sums.
__device__ float NativeV41Fp8(float x) {
    const float maximum = fmaxf(warp_reduce_max(fabsf(x)), 1e-4f);
    const unsigned bits = __float_as_uint(maximum * (1.0f / 448.0f));
    const int exponent = int((bits >> 23) & 255) - 127 + int((bits & 0x7fffff) != 0);
    const float scale = exp2f(float(exponent));
    return __bfloat162float(__float2bfloat16_rn(
        float(__nv_fp8_e4m3(fminf(448.f, fmaxf(-448.f, x / scale)))) * scale));
}

__global__ void QuantizeNativeV41Streamed(const __nv_bfloat16 *input,
        block_q8_1 *output, int columns) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= columns) return;
    const float x = NativeV41Fp8(__bfloat162float(input[size_t(blockIdx.y) * columns + c]));
    const float maximum = warp_reduce_max(fabsf(x)), sum = warp_reduce_sum(x);
    const float scale = maximum / 127.f;
    auto &q = output[size_t(blockIdx.y) * (columns / 32) + c / 32];
    q.qs[c % 32] = maximum == 0 ? 0 : int8_t(roundf(x / scale));
    if (c % 32 == 0) q.ds = __floats2half2_rn(scale, sum);
}

template<bool DS4>
__global__ void ActivateQuantizeNativeV41Streamed(const float *products,
        block_q8_1_mmq *output, const int *routes,
        const float *scores, int inter, int capacity, float limit) {
    const int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    const int route = routes[row];
    float x = 0;
    if (route >= 0) {
        float g = __bfloat162float(__float2bfloat16_rn(products[size_t(row) * 2 * inter + col]));
        float u = __bfloat162float(__float2bfloat16_rn(products[size_t(row) * 2 * inter + inter + col]));
        if (limit > 0) { g = fminf(g, limit); u = fmaxf(-limit, fminf(u, limit)); }
        x = __bfloat162float(__float2bfloat16_rn(
            __fmul_rn(__fmul_rn(g / (1.f + expf(-g)), u), scores[route])));
    }
    x = NativeV41Fp8(x);
    const float maximum = warp_reduce_max(fabsf(x)), sum = warp_reduce_sum(x);
    const float scale = maximum / 127.f;
    auto &q = output[size_t(col / 128) * capacity + row];
    q.qs[col % 128] = maximum == 0 ? 0 : int8_t(roundf(x / scale));
    if (col % 32 == 0) {
        if constexpr (DS4) q.ds4[(col % 128) / 32] = __floats2half2_rn(scale, sum);
        else q.d4[(col % 128) / 32] = __half2float(__float2half_rn(scale));
    }
}

__global__ void ScatterNativeV41Streamed(const float *products, __nv_bfloat16 *output,
        const int *routes, int rows, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    const int route = routes[i / hidden];
    if (route >= 0) output[size_t(route) * hidden + i % hidden] = __float2bfloat16_rn(products[i]);
}

static bool RunNativeV41Streamed(StreamedMoePhase phase, const __nv_bfloat16 *input,
        __nv_bfloat16 *output, void *workspace, int capacity,
        int rows, int hidden, int inter, int topk, int gt, int dt,
        const StreamedMoeBatch &batch, const float *scores, const int *indices, float limit) {
    const auto stream = cudaStreamPerThread;
    StreamedWorkspace s(workspace, rows, hidden, inter, topk, capacity, true);
    auto *partials = reinterpret_cast<__nv_bfloat16 *>(s.routes);
    if (phase == StreamedMoePhase::Prepare) {
        CUDA_CHECK(cudaMemsetAsync(partials, 0, size_t(rows) * topk * hidden * sizeof(__nv_bfloat16), stream));
        QuantizeNativeV41Streamed<<<dim3(hidden / 256, rows), 256, 0, stream>>>(input, s.input, hidden);
    } else if (phase == StreamedMoePhase::Finish) {
        ReduceV41<<<(rows * hidden + 255) / 256, 256, 0, stream>>>(
            partials, output, nullptr, indices, rows, hidden, topk);
    } else {
        Workspace w(nullptr, rows, hidden, inter, batch.experts, topk);
        w.capacity = capacity; w.activeRows = batch.rows;
        w.counts = const_cast<int *>(batch.counts); w.offsets = const_cast<int *>(batch.offsets);
        w.tileExperts = const_cast<int *>(batch.tileExperts); w.groupRoutes = const_cast<int *>(batch.routes);
        w.quantized = s.quantized; w.products = s.products;
        const dim3 gather((batch.rows + 7) / 8, hidden / 128);
        if (gt == GGML_TYPE_IQ1_S) GatherNativeV41<true><<<gather, 256, 0, stream>>>(
            s.input, s.quantized, batch.routes, batch.offsets + batch.experts, hidden, capacity, topk);
        else GatherNativeV41<false><<<gather, 256, 0, stream>>>(
            s.input, s.quantized, batch.routes, batch.offsets + batch.experts, hidden, capacity, topk);
        if (!NativeV41Matrix(gt, batch.weights, 0, w, batch.experts, hidden, 2 * inter, stream)) return false;
        const dim3 activation(inter / 256, batch.rows);
        if (dt == GGML_TYPE_IQ1_S) ActivateQuantizeNativeV41Streamed<true><<<activation, 256, 0, stream>>>(
            s.products, s.quantized, batch.routes, scores, inter, capacity, limit);
        else ActivateQuantizeNativeV41Streamed<false><<<activation, 256, 0, stream>>>(
            s.products, s.quantized, batch.routes, scores, inter, capacity, limit);
        if (!NativeV41Matrix(dt, batch.weights, 1, w, batch.experts, inter, hidden, stream)) return false;
        ScatterNativeV41Streamed<<<(batch.rows * hidden + 255) / 256, 256, 0, stream>>>(
            s.products, partials, batch.routes, batch.rows, hidden);
    }
    return cudaGetLastError() == cudaSuccess;
}
