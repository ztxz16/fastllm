// Use batched FP16 GEMMs with FP32 accumulation when tensor-core FlashInfer
// prefill is unavailable. Each KV head serves all of its query heads. Stream
// the keys in panels so temporary storage is independent of context length.
namespace legacy_prefill {
constexpr int queryTile = 128, keyTile = 1024;

__global__ void packQuery(AttentionRow a, half *packed, int count) {
    const int group = a.heads / a.kvHeads;
    const int local = blockIdx.x % count, kv = blockIdx.x / count;
    const size_t src = (size_t(a.first + local) * a.heads + kv * group) * a.width;
    for (int i = threadIdx.x; i < group * a.width; i += blockDim.x)
        packed[size_t(blockIdx.x) * group * a.width + i] = __float2half_rn(a.q[src + i]);
}

__global__ void gatherPrefixPanel(AttentionRow a, int begin, half *key, half *value) {
    const int t = begin + blockIdx.x, stride = a.kvHeads * a.width;
    const auto *k = t < a.past ? a.pages.row<half>(t, false, stride) :
                                static_cast<const half *>(a.key) + size_t(t - a.past) * stride;
    const auto *v = t < a.past ? a.pages.row<half>(t, true, stride) :
                                static_cast<const half *>(a.value) + size_t(t - a.past) * stride;
    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
        key[size_t(blockIdx.x) * stride + i] = k[i];
        value[size_t(blockIdx.x) * stride + i] = v[i];
    }
}

__global__ void softmaxPanel(AttentionRow a, float *scores, half *weights, float4 *state,
                             int rows, int begin, int keys, int pitch, bool firstPanel) {
    constexpr int threads = 128;
    using Reduce = cub::BlockReduce<float, threads>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float maximum;
    const int r = blockIdx.x, group = a.heads / a.kvHeads;
    const int q = (r / group) % rows, head = (r / (group * rows)) * group + r % group;
    const int pos = a.past + a.first + q, tokenHead = (a.first + q) * a.heads + head;
    const float tau = a.window ? 1.f : 1.f + a.alpha * logf(fmaxf(1.f, (pos + 1.f) / a.floor));
    const float4 prev = firstPanel ? make_float4(a.sink[tokenHead], 1.f, 0.f, 0.f) : state[r];
    scores += size_t(r) * pitch;
    weights += size_t(r) * pitch;
    const float *bias = a.relative + (size_t(q) * a.heads + head) * a.distances;
    float mx = prev.x;
    for (int j = threadIdx.x; j < keys; j += threads) {
        const int distance = pos - (begin + j);
        const bool valid = distance >= 0 && (!a.window || distance < a.window);
        const float v = valid ? (scores[j] + (distance < a.distances ? bias[distance] : 0.f)) * tau : -CUDART_INF_F;
        scores[j] = v;
        mx = fmaxf(mx, v);
    }
    const float blockMax = Reduce(temp).Reduce(mx, [](float x, float y) { return fmaxf(x, y); });
    if (!threadIdx.x) maximum = blockMax;
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < keys; j += threads) {
        const float w = __expf(scores[j] - maximum);
        sum += w;
        weights[j] = __float2half_rn(w);
    }
    const float total = Reduce(temp).Sum(sum);
    if (!threadIdx.x) {
        const float scale = __expf(prev.x - maximum);
        state[r] = make_float4(maximum, fmaf(prev.y, scale, total), scale, 0.f);
    }
}

__global__ void combinePanel(AttentionRow a, const float *partial, const float4 *state,
                             int rows, bool firstPanel, bool lastPanel) {
    const int r = blockIdx.x, group = a.heads / a.kvHeads;
    const int q = (r / group) % rows, head = (r / (group * rows)) * group + r % group;
    const float4 s = state[r];
    for (int i = threadIdx.x; i < a.width; i += blockDim.x) {
        const size_t j = size_t(r) * a.width + i;
        float &out = a.output[(size_t(a.first + q) * a.heads + head) * a.width + i];
        const float value = firstPanel ? partial[j] : fmaf(out, s.z, partial[j]);
        out = lastPanel ? value / s.y : value;
    }
}

struct Workspace {
    int panelKeys;
    Buffer<half> query, weights, prefix;
    Buffer<float> scores, partial;
    Buffer<float4> state;
    Workspace(int rows, int keys, int heads, int kvHeads, int width, bool hasPrefix)
        : panelKeys(keys) {
        const size_t n = size_t(rows) * heads;
        auto arena = CurrentPrefillWorkspace();
        query = Buffer<half>(n * width, arena);
        weights = Buffer<half>(n * keys, arena);
        scores = Buffer<float>(n * keys, arena);
        partial = Buffer<float>(n * width, arena);
        state = Buffer<float4>(n, arena);
        if (hasPrefix) prefix = Buffer<half>(size_t(2) * keys * kvHeads * width, arena);
    }

    bool run(AttentionRow a, int rows) const {
        auto handle = getFastllmCublasHandle();
        const int d = a.width, g = a.heads / a.kvHeads, m = rows * g, stride = a.kvHeads * d;
        const int end = a.past + a.first + rows;
        const int start = a.window ? std::max(0, a.past + a.first + 1 - a.window) : 0;
        const float scale = 1.f / std::sqrt(float(d)), one = 1.f, zero = 0.f;
        const dim3 grid(rows * a.heads);
        packQuery<<<rows * a.kvHeads,128,0,cudaStreamPerThread>>>(a, query.p, rows);
        for (int begin = start; begin < end;) {
            const int keys = std::min(panelKeys, end - begin), pitch = (keys + 7) & ~7;
            const half *k, *v;
            if (begin < a.past) {
                auto *pv = prefix.p + size_t(panelKeys) * stride;
                gatherPrefixPanel<<<keys,128,0,cudaStreamPerThread>>>(a, begin, prefix.p, pv);
                k = prefix.p; v = pv;
            } else {
                const size_t offset = size_t(begin - a.past) * stride;
                k = static_cast<const half *>(a.key) + offset;
                v = static_cast<const half *>(a.value) + offset;
            }
            const long long scoreStride = static_cast<long long>(m) * pitch;
            auto status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                keys, m, d, &scale, k, CUDA_R_16F, stride, d, query.p, CUDA_R_16F, d, m*d,
                &zero, scores.p, CUDA_R_32F, pitch, scoreStride, a.kvHeads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) return false;
            if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Inkling legacy prefill QK GEMM failed");
            softmaxPanel<<<grid,128,0,cudaStreamPerThread>>>(a, scores.p, weights.p, state.p, rows, begin, keys, pitch, begin == start);
            status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                d, m, keys, &one, v, CUDA_R_16F, stride, d, weights.p, CUDA_R_16F, pitch, scoreStride,
                &zero, partial.p, CUDA_R_32F, d, m*d, a.kvHeads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) return false;
            if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Inkling legacy prefill PV GEMM failed");
            combinePanel<<<grid,128,0,cudaStreamPerThread>>>(a, partial.p, state.p, rows,
                                                           begin == start, begin + keys == end);
            begin += keys;
        }
        check(cudaGetLastError());
        return true;
    }
};
} // namespace legacy_prefill
