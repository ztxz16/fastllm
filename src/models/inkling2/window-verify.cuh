// The two query halves multiply the same FP16 key. Stage that key once and
// keep both query fragments in registers across the canonical 32-key tiles.
// FlashInfer owns the MMA layout, online softmax and compensated PV arithmetic.
using VerifyTileMemory = flashinfer::smem_t<flashinfer::SwizzleMode::k128B>;
constexpr int verifyTileVectors = 32 * 128 / 8;

__device__ __forceinline__ void stageVerifyKV(const PageView &pages, int begin, int end,
        int kvHead, int kvHeads, VerifyTileMemory key, VerifyTileMemory value) {
    const int thread = threadIdx.y * 32 + threadIdx.x, col = thread % 16;
    const size_t pageStride = size_t(pages.pageLen) * kvHeads * 128;
    auto issue = [&](int row, int2 id, size_t column) {
        const unsigned offset = VerifyTileMemory::get_permuted_offset<16>(row, col);
        key.load_128b_async<flashinfer::SharedMemFillMode::kNoFill>(offset,
            pages.pageBase<half>(id, false, pageStride) + column, begin + row < end);
        value.load_128b_async<flashinfer::SharedMemFillMode::kFillZero>(offset,
            pages.pageBase<half>(id, true, pageStride) + column, begin + row < end);
    };
    // Canonical 32-key tiles begin on a page boundary with 16-token pages.
    // Fetch the two page IDs once for all eight transfers issued by each lane.
    if (pages.pageLen == 16) {
        const int page = (begin >> 4) - pages.first;
        const int2 first = pages.ids[page], second = begin + 16 < end ? pages.ids[page + 1] : first;
#pragma unroll
        for (int row = thread / 16; row < 32; row += 4) {
            const int2 ids = row < 16 ? first : second;
            const size_t column = ((row % 16) * kvHeads + kvHead) * 128 + col * 8;
            issue(row, ids, column);
        }
    } else {
#pragma unroll
        for (int row = thread / 16; row < 32; row += 4) {
            const int token = min(begin + row, end - 1);
            const int page = pages.shift >= 0 ? token >> pages.shift : token / pages.pageLen;
            const int2 id = pages.ids[page - pages.first];
            const size_t column = (size_t(token - page * pages.pageLen) * kvHeads + kvHead) * 128 + col * 8;
            issue(row, id, column);
        }
    }
    flashinfer::cp_async::commit_group();
}

template<class Traits>
__device__ __forceinline__ void scanVerifyTile(VerifyPrefillParams p, const DecodeVisit &visit,
        int first, int rows, int tile, int split, const VerifyPartition &part, int kvHead, int kvHeads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(Traits::CTA_TILE_Q == 32 && Traits::CTA_TILE_KV == 32 && Traits::NUM_WARPS_Q == 2);
    const int begin = split * part.chunk, past = visit.past - part.origin, length = past + rows;
    const int end = min(length, begin + part.chunk);
    if (tile * 32 >= rows * unsigned(p.group_size) || begin >= end) return;
    p.rows = rows;
    p.past = past;
    p.paged_kv.length = length;
    p.relative += size_t(first) * p.num_qo_heads * p.distances;
    p.q += size_t(first) * p.q_stride_n;
    p.qo_tile_indices = &tile;
    extern __shared__ uint4 verifyStorage[];
    VerifyTileMemory key(verifyStorage), value(verifyStorage + verifyTileVectors);
    typename Traits::AttentionVariant relative(p, 0, reinterpret_cast<uint8_t *>(verifyStorage));
    const unsigned lane = threadIdx.x;
    unsigned query[16][4], queryRow[2], queryHead[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        unsigned head;
        p.group_size.divmod(tile * 32 + threadIdx.y * 16 + lane / 4 + j * 8, queryRow[j], head);
        queryHead[j] = kvHead * unsigned(p.group_size) + head;
        const half *src = p.q + size_t(min(int(queryRow[j]), rows - 1)) * p.q_stride_n +
                          queryHead[j] * p.q_stride_h + (lane % 4) * 2;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            query[i][j] = *reinterpret_cast<const unsigned *>(src + i * 16);
            query[i][j + 2] = *reinterpret_cast<const unsigned *>(src + i * 16 + 8);
        }
    }

    float output[1][8][8], maximum[1][2], denominator[1][2];
    flashinfer::init_states<Traits>(relative, output, maximum, denominator);
    for (int token = begin; token < end; token += 32) {
        stageVerifyKV(visit.pages, part.origin + token, part.origin + end, kvHead, kvHeads, key, value);
        flashinfer::cp_async::wait_group<0>();
        __syncthreads();
        float score[1][2][8]{};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                unsigned k[4];
                key.ldmatrix_m8n8x4(VerifyTileMemory::get_permuted_offset<16>(
                    n * 16 + 8 * (lane / 16) + lane % 8, 2 * (i % 8) + (lane % 16) / 8), k);
                flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<half>(score[0][n], query[i], k);
            }
        }
        relative.template TransformScoreTile<Traits>(p, token, score);
        if (token + 31 > past) {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const unsigned index = token + n * 16 + (lane % 4) * 2 + (i / 4) * 8 + i % 2;
                    if (index >= end || index > past + queryRow[(i / 2) % 2]) score[0][n][i] = Traits::MaskFillValue;
                }
            }
        }
        flashinfer::update_mdo_states<Traits, true>(relative, score, output, maximum, denominator);
        unsigned offset = VerifyTileMemory::get_permuted_offset<16>(lane % 16, lane / 16);
        flashinfer::compute_sfm_v<Traits>(&value, &offset, nullptr, lane, score, output, denominator);
        __syncthreads();
    }
    flashinfer::finalize_m<Traits>(relative, maximum);
    const int splits = (length - 1) / part.chunk + 1;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        if (queryRow[j] >= rows) continue;
        const size_t index = (size_t(first) * p.maxSplits + queryRow[j] * splits + split) * p.num_qo_heads + queryHead[j];
        const float d = denominator[0][j];
        if (lane % 4 == 0) p.lse[index] = flashinfer::math::ptx_log2(d) + maximum[0][j];
#pragma unroll
        for (int col = 0; col < 8; ++col) {
#pragma unroll
            for (int side = 0; side < 2; ++side) {
                const float *src = output[0][col] + side * 4 + j * 2;
                const float2 pair = d > 0.f ? make_float2(src[0] / d, src[1] / d) : make_float2(0.f, 0.f);
                *reinterpret_cast<float2 *>(p.o + index * 128 + col * 16 + side * 8 + (lane % 4) * 2) = pair;
            }
        }
    }
#endif
}
