// Singleton Q8_0 fold. Keep the window MMA's four K accumulators and reduction
// order, and publish final FP32 scores directly to the radix selector.
__global__ void q8FoldScores(const InklingQ80 *weights, const InklingQ81 *x, const float *base, const float *bias,
                             const float *temperature, float cap, int vocab, float *q, float *scores,
                             FastllmFoldSelection selection, const int *topK, const int *complete, bool first) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    fastllm::cuda_launch::DependencyStart();
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned i = linear; i < selection.radixWords; i += gridDim.x * blockDim.x) selection.radix[i] = 0;
    if (!linear) *selection.count = 0;
    if (*topK > selection.maxCandidates || (!first && !*complete)) return;

    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, group = lane / 4, word = lane % 4;
    const int id = blockIdx.x * 16 + group;
    const auto *a = weights + size_t(min(id, vocab-1)) * 8;
    const auto *b = weights + size_t(min(id+8, vocab-1)) * 8;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int k = warp; k < 8; k += 4) {
        const auto *codes = reinterpret_cast<const unsigned *>(x[k].qs);
        int dot[4];
        q8WindowDot(q8WeightWord(a[k], word), q8WeightWord(b[k], word),
                    q8WeightWord(a[k], word+4), q8WeightWord(b[k], word+4), codes[word], codes[word+4], dot);
        const float scale = __low2float(x[k].ds);
        s0 = fmaf(float(dot[0]), scale * float(a[k].scale), s0);
        s1 = fmaf(float(dot[2]), scale * float(b[k].scale), s1);
    }
    __shared__ float partial[4][16];
    if (!word) {
        partial[warp][group] = s0;
        partial[warp][group+8] = s1;
    }
    __syncthreads();
    const int col = threadIdx.x, token = blockIdx.x * 16 + col;
    if (col < 16 && token < vocab) {
        const float fold = (partial[0][col] + partial[2][col]) + (partial[1][col] + partial[3][col]);
        scores[token] = (cap * tanhf(((base[token] + fold) + bias[token]) / cap)) * (1.f / *temperature);
        q[token] = 0.f;
    }
#endif
}

// Fetch immutable BF16 weights before waiting for the Markov state. The score
// selector consumes the rounded projection with contiguous base and bias loads.
template<bool Hot>
__global__ void bf16FoldProjection(const __nv_bfloat16 *weights, const __nv_bfloat16 *latent, int vocab,
        __nv_bfloat16 *adjust, FastllmFoldSelection selection) {
    FastllmBf16WarpDot256<1>(latent, weights, vocab, [&] {
        fastllm::cuda_launch::DependencyStart();
        if constexpr (Hot) {
            const unsigned linear = blockIdx.x * blockDim.x + threadIdx.x;
            for (unsigned i = linear; i < selection.radixWords; i += gridDim.x * blockDim.x)
                selection.radix[i] = 0;
            if (!linear) *selection.count = 0;
        }
    }, [&](int id, float value) { adjust[id] = __float2bfloat16_rn(value); });
}
