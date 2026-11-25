// Exchange only the local candidates that can survive a bounded global top-k.
// The original sampler still sees raw logits at their original token IDs.
constexpr int shardCandidates = 64;
using ShardKey = unsigned long long;
struct ShardCandidate { float logit; int token; };

__device__ ShardKey shardKey(float value, int token) {
    const unsigned bits = __float_as_uint(value == 0.f ? 0.f : value);
    const unsigned order = bits >> 31 ? ~bits : bits ^ 0x80000000u;
    return (ShardKey(order) << 32) | ~unsigned(token);
}

// The kth score of any subset is an exact lower bound on the local kth score.
template<class T>
__global__ void sampleShardPivot(const T *input, ShardKey *pivot, int stride, int valid, int first,
                                 const float *temperature) {
    using Sort = cub::BlockRadixSort<ShardKey,128,8>;
    __shared__ typename Sort::TempStorage scratch;
    ShardKey sample[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int i = j * 128 + threadIdx.x;
        const int col = int(size_t(i / 32) * valid / 32) + i % 32;
        sample[j] = valid >= 1024 ? shardKey(float(input[size_t(blockIdx.x) * stride + col]) *
                                            (1.f / temperature[blockIdx.x]), first + col) : 0;
    }
    Sort(scratch).SortDescending(sample);
    if (threadIdx.x == (shardCandidates - 1) / 8) pivot[blockIdx.x] = sample[(shardCandidates-1) % 8];
}

template<class T, bool Merge>
__global__ void selectShardCandidates(const T *input, const ShardKey *keys, ShardKey *selected,
                                      int stride, int valid, int first, const float *temperature, const ShardKey *pivot) {
    constexpr int lanes = 128, items = 4;
    using Sort = cub::BlockRadixSort<ShardKey, lanes, items>;
    __shared__ typename Sort::TempStorage scratch;
    const int row = blockIdx.y, tile = blockIdx.x;
    ShardKey values[items];
#pragma unroll
    for (int j = 0; j < items; ++j) {
        const int col = tile * lanes * items + j * lanes + threadIdx.x;
        ShardKey key = 0;
        if (col < valid) {
            if constexpr (Merge) key = keys[size_t(row) * stride + col];
            else {
                const float value = float(input[size_t(row) * stride + col]) * (1.f / temperature[row]);
                key = shardKey(value, first + col);
            }
        }
        values[j] = key;
    }
    if constexpr (!Merge) {
        __shared__ int count;
        __shared__ ShardKey shortlist[shardCandidates];
        if (!threadIdx.x) count = 0;
        __syncthreads();
        const ShardKey cutoff = pivot[row];
        const int lane = threadIdx.x % 32;
#pragma unroll
        for (int j = 0; j < items; ++j) {
            const bool keep = values[j] && values[j] >= cutoff;
            const unsigned mask = __ballot_sync(0xffffffff, keep);
            int offset = 0;
            if (!lane && mask) offset = atomicAdd(&count, __popc(mask));
            offset = __shfl_sync(0xffffffff, offset, 0) + __popc(mask & ((1u << lane)-1));
            if (keep && offset < shardCandidates) shortlist[offset] = values[j];
        }
        __syncthreads();
        if (count <= shardCandidates) {
            if (threadIdx.x < shardCandidates)
                selected[(size_t(row)*gridDim.x+tile)*shardCandidates+threadIdx.x] =
                    int(threadIdx.x) < count ? shortlist[threadIdx.x] : 0;
            return;
        }
    }
    Sort(scratch).SortDescending(values);
    if (threadIdx.x < shardCandidates / items)
        reinterpret_cast<ulonglong4 *>(selected + (size_t(row) * gridDim.x + tile) * shardCandidates)[threadIdx.x] =
            make_ulonglong4(values[0], values[1], values[2], values[3]);
}

template<class T>
__global__ void packShardCandidates(const T *input, const ShardKey *keys, ShardCandidate *out, int stride, int first) {
    const int row = blockIdx.x, i = row * shardCandidates + threadIdx.x;
    const ShardKey key = keys[i];
    const int token = int(~unsigned(key));
    out[i] = key ? ShardCandidate{float(input[size_t(row) * stride + token - first]), token} :
                   ShardCandidate{-CUDART_INF_F, -1};
}

template<class T> __global__ void emptyShardLogits(T *out, size_t size) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) out[i] = T(-CUDART_INF_F);
}
template<class T> __global__ void scatterShardCandidates(const ShardCandidate *input, T *out, int width, int rows) {
    const int row = blockIdx.x % rows;
    const auto c = input[blockIdx.x * shardCandidates + threadIdx.x];
    if (c.token >= 0) out[size_t(row) * width + c.token] = T(c.logit);
}

__global__ void unpackCandidateRows(const ShardCandidate *input, float *logits, int *tokens, int rows, int width) {
    const int row = blockIdx.x, i = threadIdx.x;
    ShardCandidate value{-CUDART_INF_F, -1};
    if (i < width) value = input[((i / shardCandidates) * rows + row) * shardCandidates + i % shardCandidates];
    int id[1] = {value.token >= 0 ? value.token : INT_MAX};
    float score[1] = {value.logit};
    using Sort = cub::BlockRadixSort<int, 512, 1, float>;
    __shared__ typename Sort::TempStorage scratch;
    Sort(scratch).Sort(id, score);
    if (i < width) {
        logits[row * width + i] = score[0];
        tokens[row * width + i] = id[0] == INT_MAX ? -1 : id[0];
    }
}
__global__ void remapBatchSamples(const int *tokens, int *sampled, int rows, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) sampled[row] = tokens[row * width + sampled[row]];
}
__global__ void publishCandidateProbabilities(const float *compact, const int *tokens, float *probs,
                                               int vocab, int width) {
    const int i = blockIdx.x * width + threadIdx.x, token = tokens[i];
    if (token >= 0) probs[size_t(blockIdx.x) * vocab + token] = compact[i];
}

template<class T> void gatherBoundedLogits(const Rows<T> &local, Rows<T> &output, const Placement &placement,
                                           const float *temperature, int vocab, int *tokenMap = nullptr) {
    const int rows = local.rows, first = placement.rank * local.cols;
    int count = std::clamp(vocab - first, 0, local.cols);
    const auto tiles = [](int n) { return std::max(1, (n + 511) / 512); };
    int blocks = tiles(count);
    Buffer<ShardKey> large(size_t(rows) * blocks * shardCandidates);
    Buffer<ShardKey> small(size_t(rows) * tiles(blocks * shardCandidates) * shardCandidates);
    Buffer<ShardKey> pivot(rows);
    sampleShardPivot<<<rows,128,0,cudaStreamPerThread>>>(local.p,pivot.p,local.cols,count,first,temperature);
    selectShardCandidates<T,false><<<dim3(blocks, rows),128,0,cudaStreamPerThread>>>(
        local.p, nullptr, large.p, local.cols, count, first, temperature, pivot.p);
    auto *a = large.p, *b = small.p;
    while (blocks > 1) {
        count = blocks * shardCandidates;
        blocks = tiles(count);
        selectShardCandidates<T,true><<<dim3(blocks, rows),128,0,cudaStreamPerThread>>>(
            nullptr, a, b, count, count, 0, nullptr, nullptr);
        std::swap(a,b);
    }
    Buffer<ShardCandidate> send(size_t(rows) * shardCandidates), receive(size_t(rows) * shardCandidates * placement.world);
    packShardCandidates<<<rows, shardCandidates, 0, cudaStreamPerThread>>>(local.p, a, send.p, local.cols, first);
    static_assert(sizeof(ShardCandidate) == 2 * sizeof(float));
    if (!FastllmNcclAllGather(send.p, receive.p, rows * shardCandidates * 2, FLOAT32, placement.gpu))
        throw std::runtime_error("Inkling TP candidate all-gather failed");
    if constexpr (std::is_same_v<T,float>) {
        if (tokenMap) {
            unpackCandidateRows<<<rows,512,0,cudaStreamPerThread>>>(receive.p,output.p,tokenMap,rows,output.cols);
            return;
        }
    }
    const size_t size = size_t(rows) * output.cols;
    emptyShardLogits<<<(size+255)/256,256,0,cudaStreamPerThread>>>(output.p,size);
    scatterShardCandidates<<<rows * placement.world, shardCandidates, 0, cudaStreamPerThread>>>(receive.p,output.p,output.cols,rows);
}
