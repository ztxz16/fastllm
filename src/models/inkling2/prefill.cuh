// Adapt the existing FlashInfer paged prefill operator to Inkling's independent
// K/V pages and uncommitted suffix. No historical KV is copied or repacked.
// Q = high + low. The QK operand addresses the same K head twice, so
// retaining the residual costs no additional cache storage or page commits.
template<class KV> struct PrefillSegment : KVSegment<KV> {
    __device__ const KV *operator+(size_t address) const {
        return this->at(int(address >> 32), unsigned(address));
    }
};
template<class KV, int D> struct PrefillKey : PrefillSegment<KV> {
    struct Vector {
        const KV *ptr;
        __device__ const KV *operator+(unsigned col) const { return ptr + col % D; }
    };
    __device__ Vector operator+(size_t element) const { return {PrefillSegment<KV>::operator+(element)}; }
};
template<class KV, int D> struct PrefillPages {
    typename KVView<KV>::UnitPage page_size;
    int batch_size = 1;
    const int *indptr = nullptr, *indices = nullptr, *rope_pos_offset = nullptr;
    PrefillKey<KV, D> k_data;
    PrefillSegment<KV> v_data;
    int length;
    __device__ size_t protective_get_kv_offset(unsigned token, unsigned head, unsigned, unsigned col, int) const {
        // Masked prefetch lanes still form a pointer. Use the first suffix row,
        // which is valid even after the sliding prefix has been evicted.
        if (token >= length) token = k_data.past;
        return (size_t(token) << 32) | (head * D + col);
    }
};

template<class KV, int D, class Out = float, class Bias = float> struct PrefillParams : flashinfer::BatchPrefillPagedParams<KV, KV, Out, int> {
    PrefillPages<KV,D> paged_kv;
    int rows, past, distances, floor;
    float alpha, qkScale;
    const Bias *relative;
    const float *sink;
    __host__ __device__ unsigned get_qo_len(unsigned) const { return rows; }
    __host__ __device__ unsigned get_kv_len(unsigned) const { return paged_kv.length; }
};

struct PrefillRelative : flashinfer::DefaultAttention<false, true, false, false> {
    using DefaultAttention::DefaultAttention;
    static constexpr bool use_fp32_softmax_sum = true, use_pv_residual = true;
    REGISTER_LOGITS_TRANSFORM(p, score, batch, row, key, head, kvHead, {
        const unsigned pos = p.past + row, distance = pos - key;
        const float bias = row < p.rows && distance < p.distances
            ? float(p.relative[(size_t(row) * p.num_qo_heads + head) * p.distances + distance]) : 0.f;
        const float tau = p.window_left >= 0 ? 1.f : 1.f + p.alpha * logf(fmaxf(1.f, (pos + 1.f) / p.floor));
        return (score * p.qkScale + bias) * tau;
    })
    REGISTER_OUTPUT_TRANSFORM(p, out, batch, row, head, m, d, scale, {
        if (row >= p.rows) return 0.f;
        const float sink = p.sink[size_t(row) * p.num_qo_heads + head] * flashinfer::math::log2e;
        const float maximum = fmaxf(float(m), sink), a = exp2f(float(m) - maximum);
        return out * a / (d * a + exp2f(sink - maximum));
    })
};

template<class Traits, class Params>
__global__ __launch_bounds__(Traits::NUM_THREADS) void pagedRelativePrefill(Params p) {
    extern __shared__ uint8_t smem[];
    int zero = 0, tile = blockIdx.x, keys[] = {0, p.paged_kv.length};
    p.request_indices = p.kv_tile_indices = p.q_indptr = p.o_indptr = &zero;
    p.qo_tile_indices = &tile;
    p.kv_chunk_size_ptr = &keys[1];
    p.paged_kv.indptr = keys;
    auto &storage = *reinterpret_cast<typename Traits::SharedStoragePaged *>(smem);
    flashinfer::BatchPrefillWithPagedKVCacheDevice<Traits>(p, storage, threadIdx, 0, blockIdx.z, gridDim.z);
}

template<int D, class KV> using PrefillTraits = flashinfer::KernelTraits<flashinfer::MaskMode::kCausal, 64, 1, 2, 2*D/16, D/16, 4, 1,
        flashinfer::PosEncodingMode::kNone, KV, KV, float, float, int, PrefillRelative, false>;

template<class Traits, class Params> bool configurePrefillKernel() {
    auto kernel = pagedRelativePrefill<Traits, Params>;
    cudaFuncAttributes attr{};
    auto status = cudaFuncGetAttributes(&attr, kernel);
    if (status == cudaErrorInvalidDeviceFunction || status == cudaErrorNoKernelImageForDevice) {
        cudaGetLastError();
        return false;
    }
    check(status);
    // A legacy PTX image JIT-compiled on a newer GPU still contains the
    // pre-SM80 implementation selected at compile time.
    if (attr.ptxVersion < 80 || attr.binaryVersion < 80) return false;
    constexpr size_t bytes = sizeof(typename Traits::SharedStoragePaged);
    if constexpr (bytes <= 48 * 1024) return true;
    int device, limit;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (bytes > size_t(limit)) return false;
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    return true;
}

template<int D, class KV> bool configurePrefill() {
    return configurePrefillKernel<PrefillTraits<D, KV>, PrefillParams<KV, D>>();
}

template<class Traits, int D, class KV, class Params>
void runPrefill(Params &p, const AttentionRow &a, KV *query, int rows) {
    p.q = query;
    p.q_stride_n = a.heads * Traits::HEAD_DIM_QK;
    p.q_stride_h = Traits::HEAD_DIM_QK;
    p.num_qo_heads = a.heads;
    p.group_size = a.heads / a.kvHeads;
    p.sm_scale = 1.f;
    p.qkScale = 1.f / sqrtf(float(D));
    p.sink = a.sink + size_t(a.first) * a.heads;
    p.rows = rows;
    p.past = a.past + a.first;
    p.distances = a.distances;
    p.window_left = a.window ? a.window - 1 : -1;
    p.floor = a.floor;
    p.alpha = a.alpha;
    auto &kv = p.paged_kv;
    kv.length = p.past + rows;
    kv.k_data = {a.pages, static_cast<const KV *>(a.key), 0, a.past, a.kvHeads * D, false};
    kv.v_data = {a.pages, static_cast<const KV *>(a.value), 0, a.past, a.kvHeads * D, true};
    constexpr size_t shared = sizeof(typename Traits::SharedStoragePaged);
    dim3 blocks((rows * (a.heads / a.kvHeads) + Traits::CTA_TILE_Q - 1) / Traits::CTA_TILE_Q, 1, a.kvHeads);
    pagedRelativePrefill<Traits><<<blocks, dim3(32, Traits::NUM_WARPS_Q), shared, cudaStreamPerThread>>>(p);
}

template<int D, class KV> void launchPrefill(const AttentionRow &a, KV *query, int rows) {
    PrefillParams<KV, D> p;
    p.relative = a.relative;
    p.o = a.output + size_t(a.first) * a.heads * D;
    runPrefill<PrefillTraits<D, KV>, D>(p, a, query, rows);
}

// One warp owns 16 packed (query, head) rows. Each lane visits the same two
// rows throughout the KV scan, so its two position scales live in registers.
template<bool Windowed, class Bias> struct GatedPrefillRelative : PrefillRelative {
    static constexpr bool use_pv_residual = false;
    static constexpr bool defer_softmax_rescale = true;
    static constexpr bool custom_score_tile = true;
    float tau[2];
    unsigned pos[2];
    const Bias *bias[2];
    template<class P> __device__ GatedPrefillRelative(const P &p, unsigned batch, uint8_t *smem)
        : PrefillRelative(p, batch, smem) {
        const unsigned first = (blockIdx.x * blockDim.y + threadIdx.y) * 16 + threadIdx.x / 4;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            unsigned row, head;
            p.group_size.divmod(first + 8 * j, row, head);
            pos[j] = p.past + row;
            bias[j] = row < p.rows ? p.relative +
                (size_t(row) * p.num_qo_heads + blockIdx.z * unsigned(p.group_size) + head) * p.distances : nullptr;
            if constexpr (!Windowed) tau[j] = 1.f + p.alpha * logf(fmaxf(1.f, (p.past + row + 1.f) / p.floor));
        }
    }
    template<class Traits, class Params>
    __device__ void TransformScoreTile(const Params &p, unsigned keyBase, float (*score)[Traits::NUM_MMA_KV][8]) const {
        // Beyond the learned bias extent, the whole score tile only needs a row scale.
        if constexpr (!Windowed) {
            if (pos[0] >= keyBase + Traits::CTA_TILE_KV - 1u &&
                pos[0] - (keyBase + Traits::CTA_TILE_KV - 1u) >= unsigned(p.distances)) {
#pragma unroll
                for (int k = 0; k < Traits::NUM_MMA_KV; ++k) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) score[0][k][i] *= tau[(i >> 1) & 1] * p.qkScale;
                }
                return;
            }
        }
#pragma unroll
        for (int k = 0; k < Traits::NUM_MMA_KV; ++k) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = (i >> 1) & 1;
                const unsigned key = keyBase + 16 * k + 2 * (threadIdx.x % 4) + 8 * (i / 4) + i % 2;
                const unsigned distance = pos[j] - key;
                const float relative = distance < p.distances && bias[j] ? float(bias[j][distance]) : 0.f;
                float value = score[0][k][i] * p.qkScale + relative;
                if constexpr (!Windowed) value *= tau[j];
                score[0][k][i] = value;
            }
        }
    }
};

template<class T, class Bias> struct GatedPrefillParams : PrefillParams<half,128,T,Bias> {
    const T *output_gate;
    int gateStride;
    __device__ float GateOutput(unsigned row, unsigned head, unsigned col, float value) const {
        if (row >= this->rows) return 0.f;
        const float g = float(output_gate[size_t(row) * gateStride + head * 128 + col]);
        return value * __frcp_rn(1.f + __expf(-g));
    }
};

// Each MMA lane owns pairs of adjacent channels in two rows. Normalize once
// per row, apply the gate in FP32 and store each rounded pair directly.
template<class T, class Bias> struct WindowPrefillParams : GatedPrefillParams<T,Bias> {
    static constexpr bool custom_output_store = true;
    template<class Traits>
    __device__ void StoreOutputTile(float (*output)[Traits::NUM_MMA_D_VO_TILE][8],
                                    float (*maxLogit)[2], float (*denominator)[2],
                                    unsigned packedRow, unsigned kvHead, unsigned lane) const {
        static_assert(Traits::NUM_MMA_Q == 1 && Traits::NUM_D_VO_TILES == 1 && Traits::NUM_WARPS_KV == 1);
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            unsigned row, head;
            this->group_size.divmod(packedRow + lane / 4 + 8 * j, row, head);
            if (row >= this->rows) continue;
            head += kvHead * unsigned(this->group_size);
            const size_t index = size_t(row) * this->num_qo_heads + head;
            const float sink = this->sink[index] * flashinfer::math::log2e;
            const float maximum = fmaxf(maxLogit[0][j], sink);
            const float rescale = exp2f(maxLogit[0][j] - maximum);
            const float norm = __fdividef(rescale, denominator[0][j] * rescale + exp2f(sink - maximum));
            const T *gate = this->output_gate + size_t(row) * this->gateStride + head * 128;
#pragma unroll
            for (int tile = 0; tile < Traits::NUM_MMA_D_VO_TILE; ++tile) {
#pragma unroll
                for (int halfTile = 0; halfTile < 2; ++halfTile) {
                    const int col = 16 * tile + 8 * halfTile + 2 * (lane % 4);
                    flashinfer::vec_t<T, 2> result;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const float value = output[0][tile][4 * halfTile + 2 * j + i] * norm;
                        result[i] = T(value * __fdividef(1.f, 1.f + __expf(-float(gate[col + i]))));
                    }
                    result.store(this->o + index * 128 + col);
                }
            }
        }
    }
};

template<bool Windowed, class T, class Bias = float>
using GatedParams = std::conditional_t<Windowed, WindowPrefillParams<T,Bias>, GatedPrefillParams<T,Bias>>;

template<bool Windowed, class T, class Bias = float> using GatedPrefillTraits = flashinfer::KernelTraits<flashinfer::MaskMode::kCausal,
    64, 1, 4, 8, 8, 4, 1, flashinfer::PosEncodingMode::kNone, half, half, T, float, int, GatedPrefillRelative<Windowed,Bias>>;

template<bool Windowed, class T, class Bias = float> bool configureGatedPrefill() {
    return configurePrefillKernel<GatedPrefillTraits<Windowed, T, Bias>, GatedParams<Windowed, T, Bias>>();
}

template<bool Windowed, class T, class Bias = float>
void launchGatedPrefill(const AttentionRow &a, half *query, T *out, const T *gate,
                        int gateStride, int rows, const Bias *bias) {
    using Traits = GatedPrefillTraits<Windowed, T, Bias>;
    // Each CTA owns disjoint output heads, stages its entire Q tile before
    // scanning KV and writes it once. This permits Q and output to alias.
    static_assert(Traits::NUM_D_VO_TILES == 1 && Traits::NUM_MMA_Q == 1 && sizeof(T) == sizeof(half));
    GatedParams<Windowed, T, Bias> p;
    p.relative = bias;
    p.o = out;
    p.output_gate = gate;
    p.gateStride = gateStride;
    runPrefill<Traits, 128>(p, a, query, rows);
}
