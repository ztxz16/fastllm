// One warp evaluates one (query, head) from a paged KV tile shared by the block.
// Keep the scalar decode dot products and six-token softmax updates in order.
template <int QueriesPerTile> __global__ void verifyWindowTile(AttentionRow a, float *partial, int chain) {
    constexpr int D = 128, G = 3, lanes = 16, vectorSize = 8, channels = 4, tile = 2 * G;
    constexpr int span = windowTileRows + QueriesPerTile - 1;
    __shared__ __align__(16) half keys[span][D], values[span][D];
    __shared__ float scores[QueriesPerTile * G][windowTileRows];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, split = blockIdx.z;
    const int tiles = (chain + QueriesPerTile - 1) / QueriesPerTile;
    const int first = blockIdx.x / tiles * chain + blockIdx.x % tiles * QueriesPerTile;
    const int queryCount = min(QueriesPerTile, chain - int(blockIdx.x % tiles) * QueriesPerTile);
    const int kvHead = blockIdx.y, head = kvHead * G + warp % G;
    const int queryIndex = warp / G, row = first + min(queryIndex, queryCount - 1);
    const int past = a.visit[row].past, len = min(a.window, past + 1);
    const int splits = gridDim.z, chunk = max(16, (len + splits - 1) / splits);
    const int queryStart = past + 1 - len + split * chunk, tokenCount = max(0, min(chunk, len - split * chunk));
    const float *query = a.q + (size_t(row) * a.heads + head) * D;
    const float *bias = a.relative + (size_t(row) * a.heads + head) * a.distances;
    flashinfer::vec_t<float, vectorSize> queryVec;
    queryVec.cast_load(query + (lane % lanes) * vectorSize);
    // Adjacent queries can cross a split-size boundary while filling the window.
    // Split that small group before staging KV so its tiles always fit in shared memory.
    for (int groupBegin = 0; groupBegin < queryCount;) {
        const int basePast = a.visit[first + groupBegin].past, length = min(a.window, basePast + 1);
        const int groupChunk = max(16, (length + splits - 1) / splits);
        int groupEnd = groupBegin + 1;
        while (groupEnd < queryCount &&
               max(16, (min(a.window, a.visit[first + groupEnd].past + 1) + splits - 1) / splits) == groupChunk)
            ++groupEnd;
        const int firstToken = basePast + 1 - length + split * groupChunk;
        const int endPast = a.visit[first + groupEnd - 1].past;
        const int endToken = min(endPast + 1, endPast + 1 - min(a.window, endPast + 1) + (split + 1) * groupChunk);
        for (int i = threadIdx.x; i < (endToken - firstToken) * lanes; i += blockDim.x) {
            const int pos = i / lanes, col = i % lanes * vectorSize;
            loadRelativeKV<half, true>(a.visit[first].pages, firstToken + pos, a.kvHeads * D, kvHead * D + col,
                                       *reinterpret_cast<uint4 *>(&keys[pos][col]), *reinterpret_cast<uint4 *>(&values[pos][col]));
        }
        flashinfer::cp_async::commit_group();
        flashinfer::cp_async::wait_group<0>();
        __syncthreads();
        if (queryIndex >= groupBegin && queryIndex < groupEnd && tokenCount > 0) {
            for (int i = lane / lanes; i < (tokenCount + tile - 1) / tile * tile; i += 32 / lanes) {
                float score = -flashinfer::math::inf;
                if (i < tokenCount) {
                    flashinfer::vec_t<float, vectorSize> k;
                    k.cast_load(&keys[queryStart - firstToken + i][(lane % lanes) * vectorSize]);
                    float dot = 0.f;
#pragma unroll
                    for (int j = 0; j < vectorSize; ++j) dot += queryVec[j] * k[j];
#pragma unroll
                    for (int delta = lanes / 2; delta; delta /= 2)
                        dot += __shfl_xor_sync((0xffffu << (lane & 16)), dot, delta, lanes);
                    const int dist = past - queryStart - i;
                    score = (dot * rsqrtf(float(a.width)) + (dist < a.distances ? bias[dist] : 0.f)) *
                            RelativeAttention::sm_scale_log2;
                }
                if (!(lane % lanes))
                    scores[warp][i] = score;
            }
            __syncwarp();
            flashinfer::state_t<channels> state;
            accumulateRelativeValues<G, false, true>(state, scores[warp], values + (queryStart - firstToken),
                                                     lane * channels, tokenCount);
            const size_t partialIndex = size_t(head) * splits + split;
            float *out = partial + size_t(row) * splits * a.heads * (D + 1);
#pragma unroll
            for (int c = 0; c < channels; ++c) state.o[c] /= state.d;
            state.o.store(out + partialIndex * D + lane * channels);
            if (!lane)
                out[size_t(splits) * a.heads * D + partialIndex] = state.get_lse();
        }
        groupBegin = groupEnd;
        if (groupBegin < queryCount)
            __syncthreads();
    }
}
