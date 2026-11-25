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
