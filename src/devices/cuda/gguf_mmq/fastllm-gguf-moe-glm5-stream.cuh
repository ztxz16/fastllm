#pragma once
// Included inside grouped_moe after StreamedWorkspace and ScatterStreamed.
// Preserve GLM's BF16 projection boundary, asymmetric clamp, score-before-down
// and positive block-256 Q8_K quantization while using bounded expert groups.
__global__ void ActivateQuantizeGlm5Streamed(const float *products,
        block_q8_1_mmq *output, __nv_bfloat16 *gate, const int *routes,
        const float *scores, int inter, int capacity, float limit) {
    __shared__ float maxima[8];
    const int row = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
    const int route = routes[row];
    float x = 0;
    if (route >= 0) {
        float g = __bfloat162float(__float2bfloat16_rn(products[size_t(row) * 2 * inter + col]));
        float u = __bfloat162float(__float2bfloat16_rn(products[size_t(row) * 2 * inter + inter + col]));
        if (limit > 0) { g = fminf(g, limit); u = fmaxf(-limit, fminf(u, limit)); }
        const float h = __fmul_rn(g / (1.f + expf(-g)), u);
        const auto value = __float2bfloat16_rn(__fmul_rn(scores[route], h));
        gate[size_t(route) * inter + col] = value;
        x = __bfloat162float(value);
    }
    float maximum = fabsf(x);
    for (int mask = 16; mask; mask >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
    if (threadIdx.x % 32 == 0) maxima[threadIdx.x / 32] = maximum;
    __syncthreads();
    maximum = 0;
    for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, maxima[i]);
    const float inverse = maximum == 0 ? 0 : __fdiv_rn(127.f, maximum);
    auto &q = output[size_t(col / 128) * capacity + row];
    q.qs[col % 128] = __float2int_rn(__fmul_rn(x, inverse));
    if (col % 32 == 0) q.d4[(col % 128) / 32] = __fdiv_rn(maximum, 127.f);
}

static bool RunGlm5Streamed(StreamedMoePhase phase, const __nv_bfloat16 *input,
        __nv_bfloat16 *gate, __nv_bfloat16 *output, void *workspace, int capacity,
        int rows, int hidden, int inter, int topk, int gt, int dt,
        const StreamedMoeBatch &batch, const float *scores, const int *indices, float limit) {
    const auto stream = cudaStreamPerThread;
    StreamedWorkspace s(workspace, rows, hidden, inter, topk, capacity);
    if (phase == StreamedMoePhase::Prepare) {
        CUDA_CHECK(cudaMemsetAsync(s.routes, 0, size_t(rows) * topk * hidden * sizeof(float), stream));
        QuantizeGlm5<<<dim3(hidden / 256, rows), 256, 0, stream>>>(
            input, reinterpret_cast<block_q8_K *>(s.input), hidden);
    } else if (phase == StreamedMoePhase::Finish) {
        // ScatterStreamed already placed BF16-rounded products in route order.
        // Scores were applied before down; only ascending-expert summation remains.
        ReduceV41<<<(rows * hidden + 255) / 256, 256, 0, stream>>>(
            s.routes, output, nullptr, indices, rows, hidden, topk);
    } else {
        Workspace w(nullptr, rows, hidden, inter, batch.experts, topk);
        w.capacity = capacity; w.activeRows = batch.rows;
        w.counts = const_cast<int *>(batch.counts); w.offsets = const_cast<int *>(batch.offsets);
        w.tileExperts = const_cast<int *>(batch.tileExperts); w.groupRoutes = const_cast<int *>(batch.routes);
        w.quantized = s.quantized; w.products = s.products;
        GatherV41<<<dim3((batch.rows + 7) / 8, hidden / 128), 256, 0, stream>>>(
            reinterpret_cast<const block_q8_K *>(s.input), s.quantized, batch.routes,
            batch.offsets + batch.experts, hidden, capacity, topk);
        Matrix(gt, batch.weights, 0, w, batch.experts, hidden, 2 * inter, stream);
        ActivateQuantizeGlm5Streamed<<<dim3(inter / 256, batch.rows), 256, 0, stream>>>(
            s.products, s.quantized, gate, batch.routes, scores, inter, capacity, limit);
        Matrix(dt, batch.weights, 1, w, batch.experts, inter, hidden, stream);
        ScatterStreamed<__nv_bfloat16><<<(batch.rows * hidden + 255) / 256, 256, 0, stream>>>(
            s.products, s.routes, batch.routes, batch.rows, hidden);
    }
    return cudaGetLastError() == cudaSuccess;
}
