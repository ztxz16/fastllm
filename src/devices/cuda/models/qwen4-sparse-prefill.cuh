// Indexed FP16 attention. A tile shares one KV head among up to 16 query
// heads. KV remains in its original cache; only the scores are materialized.
__global__ void Qwen4SparseQK(const half *query, const half *key, const int32_t *indices,
                              half *scores, int sequence, int queryHeads, int groups, int length,
                              int dim, int width, int rowStart, uint64_t keyStride, float scale,
                              Qwen4CacheAddress address) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char shared[];
    auto *q = reinterpret_cast<half *>(shared);
    const int pitch = dim + 8;
    auto *k = q + 16 * pitch;
    auto *dots = reinterpret_cast<float *>(k + 64 * pitch);
    const int row = blockIdx.y + rowStart;
    const int headTiles = (groups + 15) / 16;
    const int kvHead = blockIdx.z / headTiles;
    const int firstHead = kvHead * groups + (blockIdx.z % headTiles) * 16;
    const int heads = min(16, kvHead * groups + groups - firstHead);
    for (int i = threadIdx.x; i < 16 * (dim / 8); i += blockDim.x) {
        const int h = i / (dim / 8), col = (i % (dim / 8)) * 8;
        *reinterpret_cast<uint4 *>(q + h * pitch + col) =
            h < heads ? *reinterpret_cast<const uint4 *>(
                            query + (uint64_t(firstHead + h) * sequence + row) * dim + col)
                      : make_uint4(0, 0, 0, 0);
    }
    for (int i = threadIdx.x; i < 64 * (dim / 8); i += blockDim.x) {
        const int tileRow = i / (dim / 8), col = (i % (dim / 8)) * 8;
        const int selected = blockIdx.x * 64 + tileRow;
        const int token = selected < width ? indices[uint64_t(row) * width + selected] : -1;
        *reinterpret_cast<uint4 *>(k + tileRow * pitch + col) =
            token >= 0 && token < length
                ? *reinterpret_cast<const uint4 *>(
                      key + address.Offset(false, kvHead, token, keyStride) + col)
                : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int col = 0; col < dim; col += 16) {
        wmma::load_matrix_sync(a, q + col, pitch);
        wmma::load_matrix_sync(b, k + warp * 16 * pitch + col, pitch);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(dots + warp * 256, c, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        const int h = i / 16, selected = blockIdx.x * 64 + warp * 16 + i % 16;
        if (h < heads && selected < width) {
            const int token = indices[uint64_t(row) * width + selected];
            scores[(uint64_t(blockIdx.y) * queryHeads + firstHead + h) * width + selected] =
                __float2half_rn(token >= 0 && token < length ? dots[warp * 256 + i] * scale
                                                             : -10000.0f);
        }
    }
#endif
}

__global__ void Qwen4SparsePV(const half *probabilities, const half *value, const int32_t *indices,
                              half *output, int sequence, int queryHeads, int groups, int length,
                              int dim, int width, int rowStart, uint64_t valueStride,
                              Qwen4CacheAddress address) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    __shared__ __align__(32) half p[16 * 24], v[16 * 72];
    __shared__ __align__(32) float result[4 * 16 * 16];
    const int row = blockIdx.y + rowStart;
    const int headTiles = (groups + 15) / 16;
    const int kvHead = blockIdx.z / headTiles;
    const int firstHead = kvHead * groups + (blockIdx.z % headTiles) * 16;
    const int heads = min(16, kvHead * groups + groups - firstHead);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int selected = 0; selected < width; selected += 16) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            const int h = i / 16, column = selected + i % 16;
            p[h * 24 + i % 16] =
                h < heads && column < width
                    ? probabilities[(uint64_t(blockIdx.y) * queryHeads + firstHead + h) * width +
                                    column]
                    : half(0);
        }
        for (int i = threadIdx.x; i < 16 * 8; i += blockDim.x) {
            const int column = blockIdx.x * 64 + (i % 8) * 8;
            const int index = selected + i / 8;
            const int token = index < width ? indices[uint64_t(row) * width + index] : -1;
            *reinterpret_cast<uint4 *>(v + (i / 8) * 72 + (i % 8) * 8) =
                column < dim && token >= 0 && token < length
                    ? *reinterpret_cast<const uint4 *>(
                          value + address.Offset(true, kvHead, token, valueStride) + column)
                    : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        wmma::load_matrix_sync(a, p, 24);
        wmma::load_matrix_sync(b, v + warp * 16, 72);
        wmma::mma_sync(c, a, b, c);
        __syncthreads();
    }
    wmma::store_matrix_sync(result + warp * 256, c, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        const int h = i / 16, col = blockIdx.x * 64 + warp * 16 + i % 16;
        if (h < heads && col < dim)
            output[(uint64_t(firstHead + h) * sequence + row) * dim + col] =
                __float2half_rn(result[warp * 256 + i]);
    }
#endif
}
