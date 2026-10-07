#pragma once
// Packed MoE prefill on the existing MMQ tile machinery. Included inside
// fastllm_gguf_mmq after mmq_io, the Q8 helpers and imported MMQ definitions.

#include "fastllm-gguf-moe-v41-mmq.cuh"

namespace grouped_moe {
constexpr int kTile = 16;

// Q2_0 stores four consecutive {-1,0,1,2} codes per byte, with one FP16
// scale per 64 values. Expand only the current weight tile into MMA shared
// memory. Guard the K tail: TP shards may have 320 columns, not 256*k.
template<int Y, int Warps, bool Check>
__device__ void LoadQ2(const char *x, int *tile, const int &kb0,
                       const int &imax, const int &stride) {
#ifdef INT8_MMA_AVAILABLE
    constexpr int Pitch = MMQ_MMA_TILE_X_K_Q8_0;
    auto *scales = reinterpret_cast<float *>(tile + 2*WARP_SIZE);
    for (int row0 = 0; row0 < Y; row0 += Warps) {
        const int row = row0 + threadIdx.y;
        const int srcRow = Check ? min(row, imax) : row;
        const auto *weight = reinterpret_cast<const block_q2_0 *>(x + srcRow*stride);
        // One aligned 16-bit load supplies eight values. Decode both byte
        // vectors in registers instead of issuing two dependent byte loads.
        const int block = kb0 + threadIdx.x/8;
        int2 values = make_int2(0, 0);
        if (block < stride/int(sizeof(block_q2_0))) {
            const uint16_t codes = reinterpret_cast<const uint16_t *>(
                weight[block].qs)[threadIdx.x%8];
            values = gguf_cache_q8::UnpackQ2(codes);
        }
        tile[row*Pitch + 2*threadIdx.x] = values.x;
        tile[row*Pitch + 2*threadIdx.x+1] = values.y;
        if (threadIdx.x < 8) {
            const int block = kb0 + threadIdx.x/2;
            scales[row*Pitch + threadIdx.x] = block < stride/int(sizeof(block_q2_0))
                ? __half2float(weight[block].d) : 0.0f;
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}
} // namespace grouped_moe

template<int X, int Y, int Warps, bool Check>
struct mmq_type_traits<X, Y, Warps, Check, GGML_TYPE_Q2_0> {
    static constexpr load_tiles_mmq_t load_tiles = grouped_moe::LoadQ2<Y, Warps, Check>;
    static constexpr vec_dot_mmq_t vec_dot_mma =
        vec_dot_q8_0_q8_1_mma<X, Y, Warps, MMQ_Q8_1_DS_LAYOUT_D4>;
    // Admission requires NVIDIA SM75+, so the DP4A instantiation is unreachable.
    static constexpr vec_dot_mmq_t vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<X, Y, Warps>;
};

namespace grouped_moe {
// MTP and TP shards can have a Q8_0 K dimension that ends inside a 256-value
// MMA tile. The ordinary loader assumes complete tiles; zero the tail here.
template<int Y, int Warps, bool Check>
__device__ void LoadQ8(const char *x, int *tile, const int &kb0,
                       const int &imax, const int &stride) {
#ifdef INT8_MMA_AVAILABLE
    constexpr int Pitch = MMQ_MMA_TILE_X_K_Q8_0;
    auto *scales = reinterpret_cast<float *>(tile+2*WARP_SIZE);
    const int lane = threadIdx.x, blocks = stride/int(sizeof(block_q8_0));
    for (int row0 = 0; row0 < Y; row0 += Warps) {
        const int row = row0+threadIdx.y;
        const int srcRow = Check ? min(row, imax) : row;
        const auto *weight = reinterpret_cast<const block_q8_0 *>(x+srcRow*stride);
        for (int half = 0; half < 2; ++half) {
            const int block = kb0+lane/8+half*4;
            tile[row*Pitch+lane+half*32] = block < blocks
                ? get_int_b2(weight[block].qs, lane%8) : 0;
        }
        if (lane < 8) scales[row*Pitch+lane] = kb0+lane < blocks
            ? __half2float(weight[kb0+lane].d) : 0.0f;
    }
#else
    NO_DEVICE_CODE;
#endif
}
template<int X, int Y, int Warps, bool Check, ggml_type Type>
struct StreamTraits : mmq_type_traits<X, Y, Warps, Check, Type> {};
template<int X, int Y, int Warps, bool Check>
struct StreamTraits<X, Y, Warps, Check, GGML_TYPE_Q8_0>
        : mmq_type_traits<X, Y, Warps, Check, GGML_TYPE_Q8_0> {
    static constexpr load_tiles_mmq_t load_tiles = LoadQ8<Y, Warps, Check>;
};

static bool MatrixType(int type, int columns) {
    if (columns <= 0) return false;
    if (type == GGML_TYPE_Q2_0) return columns%64 == 0;
    if (type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q8_0) return columns%32 == 0;
    return columns%256 == 0 && (type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
        type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S ||
        type == GGML_TYPE_IQ4_XS);
}
static size_t Align(size_t x) { return (x+255)&~size_t(255); }
struct Workspace {
    int capacity, inputRows, activeRows;
    int *counts, *offsets, *cursors, *tileExperts, *groupRoutes, *routeGroups;
    block_q8_1_mmq *quantized;
    float *products;
    size_t bytes;
    Workspace(void *base, int rows, int hidden, int inter, int experts, int topk) : inputRows(rows) {
        const int routes = rows*topk;
        capacity = ((routes+experts*(kTile-1)+kTile-1)/kTile)*kTile;
        activeRows = capacity;
        size_t used = 0;
        auto take = [&](size_t n) -> void * {
            void *p = base ? static_cast<char *>(base)+used : nullptr;
            used += Align(n);
            return p;
        };
        counts = static_cast<int *>(take(experts*sizeof(int)));
        offsets = static_cast<int *>(take((experts+1)*sizeof(int)));
        cursors = static_cast<int *>(take(experts*sizeof(int)));
        tileExperts = static_cast<int *>(take((capacity/kTile)*sizeof(int)));
        groupRoutes = static_cast<int *>(take(capacity*sizeof(int)));
        routeGroups = static_cast<int *>(take(routes*sizeof(int)));
        const int padded = ((std::max(hidden, inter)+255)/256)*256;
        quantized = static_cast<block_q8_1_mmq *>(take(
            size_t(capacity)*(padded/128)*sizeof(block_q8_1_mmq)));
        // Gate/up and down are sequential: reuse one product buffer.
        products = static_cast<float *>(take(
            size_t(capacity)*std::max(2*inter, hidden)*sizeof(float)));
        bytes = used;
    }
};

__global__ void Count(const int *indices, const uint8_t *const *weights,
                      int *counts, int routes, int experts) {
    const int r = blockIdx.x*blockDim.x+threadIdx.x;
    if (r >= routes) return;
    const int e = indices[r];
    if (e >= 0 && e < experts && weights[2*e] && weights[2*e+1]) atomicAdd(counts+e, 1);
}
__global__ void Prefix(const int *counts, int *offsets, int *tileExperts, int experts) {
    __shared__ int sums[1024];
    const int e = threadIdx.x;
    const int count = e < experts ? ((counts[e]+kTile-1)/kTile)*kTile : 0;
    sums[e] = count;
    __syncthreads();
    for (int step = 1; step < blockDim.x; step *= 2) {
        const int previous = e >= step ? sums[e-step] : 0;
        __syncthreads();
        sums[e] += previous;
        __syncthreads();
    }
    if (e < experts) {
        const int begin = sums[e]-count;
        offsets[e] = begin;
        for (int tile = begin/kTile; tile < sums[e]/kTile; ++tile) tileExperts[tile] = e;
        if (e == experts-1) offsets[experts] = sums[e];
    }
}
__global__ void Scatter(const int *indices, const uint8_t *const *weights,
                        const int *offsets, int *cursors, int *groupRoutes,
                        int *routeGroups, int routes, int experts) {
    const int r = blockIdx.x*blockDim.x+threadIdx.x;
    if (r >= routes) return;
    const int e = indices[r];
    int position = -1;
    if (e >= 0 && e < experts && weights[2*e] && weights[2*e+1]) {
        position = offsets[e] + atomicAdd(cursors+e, 1);
        groupRoutes[position] = r;
    }
    routeGroups[r] = position;
}

// Use the same 32-value activation quantizer as the small-batch path. D4
// stores its FP16-rounded scale in float; this avoids changing the Q8 oracle.
template<class T>
__global__ void Quantize(const T *input, block_q8_1_mmq *output,
                         const int *groupRoutes, const int *activeRows,
                         int columns, int capacity) {
    if (blockIdx.y >= *activeRows) return;
    const int col = blockIdx.x*blockDim.x+threadIdx.x;
    const int padded = ((columns+255)/256)*256;
    if (col >= padded) return;
    const int row = blockIdx.y, route = groupRoutes ? groupRoutes[row] : row;
    const float x = route >= 0 && col < columns ? mmq_io<T>::to_float(input[size_t(route)*columns+col]) : 0.0f;
    float maximum = fabsf(x);
#pragma unroll
    for (int m = 16; m; m >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, m));
    const float scale = maximum/127.0f;
    auto &q = output[size_t(col/128)*capacity + row];
    q.qs[col%128] = maximum == 0 ? 0 : int8_t(roundf(x/scale));
    if (col%32 == 0) q.d4[(col%128)/32] = __half2float(__float2half_rn(scale));
}

template<ggml_type Type, int Tile = kTile>
__global__ void Matmul(const uint8_t *const *weights, int part,
                       const block_q8_1_mmq *input, float *output,
                       const int *counts, const int *offsets, const int *tileExperts,
                       int experts, int columns, int width, int capacity, int stride) {
    if (blockIdx.y*kTile >= offsets[experts]) return;
    const int e = tileExperts[blockIdx.y], begin = offsets[e];
    const int packedTile = blockIdx.y-begin/kTile;
    // Keep the 16-row routing/workspace layout. Only the first CTA of each
    // wider tile computes; this avoids padding every expert to 64 rows.
    if (packedTile%(Tile/kTile)) return;
    const int localTile = packedTile/(Tile/kTile);
    const int padded = ((columns+255)/256)*256;
    if constexpr (Type == GGML_TYPE_Q2_K || Type == GGML_TYPE_Q4_K) {
        mul_mat_q_process_tile<Type, Tile, MMQ_NWARPS, true, false, v41_mmq_type_traits, (Tile > kTile)>(
            reinterpret_cast<const char *>(weights[2*e+part]),
            reinterpret_cast<const char *>(input+begin), output+size_t(begin)*width,
            nullptr, padded, width, stride, padded, counts[e], capacity, width,
            blockIdx.x, localTile, 0, padded/256);
    } else mul_mat_q_process_tile<Type, Tile, MMQ_NWARPS, true, false, StreamTraits, (Tile > kTile)>(
        reinterpret_cast<const char *>(weights[2*e+part]),
        reinterpret_cast<const char *>(input+begin), output+size_t(begin)*width,
        nullptr, padded, width, stride, padded, counts[e], capacity, width,
        blockIdx.x, localTile, 0, padded/ggml_cuda_type_traits<Type>::qk);
}

template<class T>
__global__ void Activate(const float *products, T *gate, const int *routeGroups,
                         int routes, int inter) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= routes*inter) return;
    const int row = routeGroups[i/inter], col = i%inter;
    float g = 0, u = 0;
    if (row >= 0) {
        g = mmq_io<T>::to_float(mmq_io<T>::from_float(products[size_t(row)*2*inter+col]));
        u = mmq_io<T>::to_float(mmq_io<T>::from_float(products[size_t(row)*2*inter+inter+col]));
    }
    gate[i] = mmq_io<T>::from_float(g/(1.0f+expf(-g))*u);
}
template<class T>
__global__ void Reduce(const float *products, T *output, const int *routeGroups,
                       const float *scores, int rows, int hidden, int topk) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= rows*hidden) return;
    float sum = 0;
    for (int k = 0; k < topk; ++k) {
        const int route = (i/hidden)*topk+k, group = routeGroups[route];
        const float v = group < 0 ? 0.0f : mmq_io<T>::to_float(
            mmq_io<T>::from_float(products[size_t(group)*hidden+i%hidden]));
        sum += v*scores[route];
    }
    output[i] = mmq_io<T>::from_float(sum);
}

// Quantize each token once, then gather its packed Q8 values for each expert.
// A warp copies one 128-value D4 block; padded expert rows are explicitly zero.
__global__ void GatherQuantized(const block_q8_1 *input, block_q8_1_mmq *output,
                                const int *groupRoutes, const int *activeRows,
                                int columns, int capacity, int topk) {
    const int row = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (row >= *activeRows) return;
    const int route = groupRoutes[row], firstBlock = blockIdx.y*4;
    auto *destination = reinterpret_cast<uint32_t *>(output+size_t(blockIdx.y)*capacity+row);
    for (int word = threadIdx.x%32; word < sizeof(block_q8_1_mmq)/sizeof(uint32_t); word += 32) {
        const int block = firstBlock+(word < 4 ? word : (word-4)/8);
        uint32_t value = 0;
        if (route >= 0 && block < columns/32) {
            const auto &q = input[size_t(route/topk)*(columns/32)+block];
            value = word < 4 ? __float_as_uint(__low2float(q.ds))
                            : reinterpret_cast<const uint32_t *>(q.qs)[(word-4)%8];
        }
        destination[word] = value;
    }
}
// IQ1_M retains its existing Q8 dot arithmetic until an MMQ tile loader is
// available. This is only the gate/up fallback; its down still uses MMQ.
template<class T>
__global__ void IQ1Gate(const block_q8_1 *input, T *gate,
                        const uint8_t *const *weights, const int *indices,
                        int experts, int hidden, int inter, int topk, int stride) {
    __shared__ uint64_t grid[2048];
    extern __shared__ uint32_t activation[];
    gguf_cache_q8::StageGrid<GGML_TYPE_IQ1_M>(grid);
    const int route = blockIdx.y;
    const auto *x = input + size_t(route/topk)*(hidden/32);
    for (int i = threadIdx.x; i < hidden/32*int(sizeof(block_q8_1))/4; i += blockDim.x)
        activation[i] = reinterpret_cast<const uint32_t *>(x)[i];
    __syncthreads();
    const int row = blockIdx.x*8+threadIdx.x/32;
    if (row >= inter) return;
    const int e = indices[route];
    float value = 0;
    if (e >= 0 && e < experts && weights[2*e] && weights[2*e+1]) {
        const auto *w = weights[2*e]+size_t(row)*stride;
        const auto *qx = reinterpret_cast<const block_q8_1 *>(activation);
        const float g = mmq_io<T>::to_float(mmq_io<T>::from_float(
            gguf_cache_q8::RowDot<GGML_TYPE_IQ1_M>(w, qx, hidden, grid)));
        const float u = mmq_io<T>::to_float(mmq_io<T>::from_float(
            gguf_cache_q8::RowDot<GGML_TYPE_IQ1_M>(w+size_t(inter)*stride, qx, hidden, grid)));
        value = g/(1.0f+expf(-g))*u;
    }
    if (threadIdx.x%32 == 0) gate[size_t(route)*inter+row] = mmq_io<T>::from_float(value);
}

template<ggml_type Type, int Tile = kTile>
static void LaunchMatrix(const uint8_t *const *weights, int part, Workspace &w,
                          int experts, int columns, int width, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    constexpr ggml_type SharedType = Type == GGML_TYPE_Q2_0 ? GGML_TYPE_Q8_0 :
        Type == GGML_TYPE_Q4_K ? GGML_TYPE_Q2_K : Type;
    const int shared = mmq_get_shmem<SharedType>(Tile, get_mmq_y_host(cc), cc);
    static std::once_flag initialized[GGML_CUDA_MAX_DEVICES];
    std::call_once(initialized[device], [shared]() {
        CUDA_CHECK(cudaFuncSetAttribute(Matmul<Type, Tile>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    });
    Matmul<Type, Tile><<<dim3((width+get_mmq_y_host(cc)-1)/get_mmq_y_host(cc), w.activeRows/kTile),
        dim3(32, MMQ_NWARPS), shared, stream>>>(weights, part, w.quantized, w.products,
            w.counts, w.offsets, w.tileExperts, experts, columns, width, w.capacity,
            int(ggml_row_size(Type, columns)));
}
static void Matrix(int type, const uint8_t *const *weights, int part, Workspace &w,
                    int experts, int columns, int width, cudaStream_t stream) {
    // Short expert batches waste half of a 64-row tile on Turing. Use a
    // narrower tile based on the routed batch, independent of model dimensions.
    const bool compactTuring = w.inputRows >= 1024 &&
        int64_t(w.activeRows) <= int64_t(experts) * 48 &&
        ggml_cuda_info().devices[ggml_cuda_get_device()].cc == CC_TURING;
    switch (type) {
#define GROUPED_CASE(T) case GGML_TYPE_##T: \
        if (compactTuring) LaunchMatrix<GGML_TYPE_##T, 32>(weights, part, w, experts, columns, width, stream); \
        else if (w.inputRows >= 1024) LaunchMatrix<GGML_TYPE_##T, 64>(weights, part, w, experts, columns, width, stream); \
        else LaunchMatrix<GGML_TYPE_##T>(weights, part, w, experts, columns, width, stream); break;
        GROUPED_CASE(Q2_0) GROUPED_CASE(IQ2_XXS) GROUPED_CASE(IQ2_XS) GROUPED_CASE(IQ2_S)
        GROUPED_CASE(IQ3_XXS) GROUPED_CASE(IQ3_S) GROUPED_CASE(IQ4_NL) GROUPED_CASE(IQ4_XS)
        GROUPED_CASE(Q8_0) GROUPED_CASE(Q2_K) GROUPED_CASE(Q4_K)
#undef GROUPED_CASE
    }
}
#include "fastllm-gguf-moe-v41.cuh"
template<class T>
static bool Run(const T *input, T *gate, T *output, const uint8_t *const *weights,
                 const int *indices, const float *scores, void *workspace,
                 int gt, int dt, int rows, int hidden, int inter, int experts, int topk,
                 bool deepSeekV41, float swigluLimit, cudaEvent_t downWeightsReady) {
    const auto stream = cudaStreamPerThread;
    Workspace w(workspace, rows, hidden, inter, experts, topk);
    const int routes = rows*topk;
    CUDA_CHECK(cudaMemsetAsync(w.counts, 0, experts*sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(w.cursors, 0, experts*sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(w.groupRoutes, 0xff, w.capacity*sizeof(int), stream));
    Count<<<(routes+255)/256, 256, 0, stream>>>(indices, weights, w.counts, routes, experts);
    int threads = 32;
    while (threads < experts) threads *= 2;
    Prefix<<<1, threads, 0, stream>>>(w.counts, w.offsets, w.tileExperts, experts);
    Scatter<<<(routes+255)/256, 256, 0, stream>>>(indices, weights, w.offsets, w.cursors,
        w.groupRoutes, w.routeGroups, routes, experts);
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        if (deepSeekV41) {
            RunV41(input, gate, output, weights, indices, scores, w,
                rows, hidden, inter, experts, topk, swigluLimit, downWeightsReady);
            return cudaGetLastError() == cudaSuccess;
        }
    }
    // Gate input uses the same Q8 quantizer as Dense MMVQ. Matrix is the
    // first consumer of products, so this scratch reuse adds no allocation.
    auto *q = reinterpret_cast<block_q8_1 *>(w.products);
    quantize_mmvq_q8_1<<<dim3((hidden+255)/256, rows), 256, 0, stream>>>(input, q, hidden);
    if (gt == GGML_TYPE_IQ1_M) {
        IQ1Gate<<<dim3((inter+7)/8, routes), 256, hidden/32*sizeof(block_q8_1), stream>>>(
            q, gate, weights, indices, experts, hidden, inter, topk,
            int(ggml_row_size(GGML_TYPE_IQ1_M, hidden)));
    } else {
        GatherQuantized<<<dim3((w.capacity+7)/8, ((hidden+255)/256)*2), 256, 0, stream>>>(
            q, w.quantized, w.groupRoutes, w.offsets+experts, hidden, w.capacity, topk);
        Matrix(gt, weights, 0, w, experts, hidden, 2*inter, stream);
        Activate<<<(routes*inter+255)/256, 256, 0, stream>>>(w.products, gate, w.routeGroups, routes, inter);
    }
    Quantize<<<dim3((inter+255)/256, w.capacity), 256, 0, stream>>>(
        gate, w.quantized, w.groupRoutes, w.offsets+experts, inter, w.capacity);
    if (downWeightsReady) CUDA_CHECK(cudaStreamWaitEvent(stream, downWeightsReady, 0));
    Matrix(dt, weights, 1, w, experts, inter, hidden, stream);
    Reduce<<<(rows*hidden+255)/256, 256, 0, stream>>>(w.products, output,
        w.routeGroups, scores, rows, hidden, topk);
    return cudaGetLastError() == cudaSuccess;
}

// A bounded compute workspace is shared by all streamed groups. Only the
// per-route products survive between groups, so final reduction keeps the
// original top-k order and activation rounding without atomic accumulation.
struct StreamedWorkspace {
    block_q8_1 *input;
    block_q8_1_mmq *quantized;
    float *products, *routes;
    size_t bytes;
    StreamedWorkspace(void *base, int rows, int hidden, int inter, int topk, int capacity) {
        size_t used = 0;
        auto take = [&](size_t n) -> void * {
            void *p = base ? static_cast<char *>(base) + used : nullptr;
            used += Align(n); return p;
        };
        input = static_cast<block_q8_1 *>(take(size_t(rows) * (hidden / 32) * sizeof(block_q8_1)));
        quantized = static_cast<block_q8_1_mmq *>(take(size_t(capacity) *
            (((std::max(hidden, inter) + 255) / 256) * 2) * sizeof(block_q8_1_mmq)));
        products = static_cast<float *>(take(size_t(capacity) * std::max(2 * inter, hidden) * sizeof(float)));
        routes = static_cast<float *>(take(size_t(rows) * topk * hidden * sizeof(float)));
        bytes = used;
    }
};

// Preserve the projection and activation rounding, then quantize the value
// still in registers. Padding participates as zero in the same 32-value max.
template<class T>
__global__ void ActivateQuantizeStreamed(const float *products,
        block_q8_1_mmq *output, T *gate, const int *routes,
        int inter, int capacity) {
    const int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ((inter + 255) / 256) * 256) return;
    const int route = routes[row];
    T value = mmq_io<T>::from_float(0);
    if (route >= 0 && col < inter) {
        const float g = mmq_io<T>::to_float(mmq_io<T>::from_float(products[size_t(row) * 2 * inter + col]));
        const float u = mmq_io<T>::to_float(mmq_io<T>::from_float(products[size_t(row) * 2 * inter + inter + col]));
        value = mmq_io<T>::from_float(g / (1 + expf(-g)) * u);
        gate[size_t(route) * inter + col] = value;
    }
    const float x = mmq_io<T>::to_float(value);
    float maximum = fabsf(x);
#pragma unroll
    for (int m = 16; m; m >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, m));
    const float scale = maximum / 127.0f;
    auto &q = output[size_t(col / 128) * capacity + row];
    q.qs[col % 128] = maximum == 0 ? 0 : int8_t(roundf(x / scale));
    if (col % 32 == 0) q.d4[(col % 128) / 32] = __half2float(__float2half_rn(scale));
}

template<class T>
__global__ void ScatterStreamed(const float *products, float *output,
        const int *routes, int rows, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    const int route = routes[i / hidden];
    if (route >= 0) output[size_t(route) * hidden + i % hidden] =
        mmq_io<T>::to_float(mmq_io<T>::from_float(products[i]));
}

template<class T>
__global__ void ReduceStreamed(const float *products, T *output,
        const float *scores, int rows, int hidden, int topk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    float sum = 0;
    for (int k = 0; k < topk; ++k) {
        const int route = (i / hidden) * topk + k;
        sum += products[size_t(route) * hidden + i % hidden] * scores[route];
    }
    output[i] = mmq_io<T>::from_float(sum);
}

template<class T>
static bool RunStreamed(StreamedMoePhase phase, const T *input, T *gate, T *output,
        void *workspace, int capacity, int rows, int hidden, int inter, int topk,
        int gt, int dt, const StreamedMoeBatch &batch, const float *scores) {
    const auto stream = cudaStreamPerThread;
    StreamedWorkspace s(workspace, rows, hidden, inter, topk, capacity);
    if (phase == StreamedMoePhase::Prepare) {
        CUDA_CHECK(cudaMemsetAsync(s.routes, 0, size_t(rows) * topk * hidden * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(gate, 0, size_t(rows) * topk * inter * sizeof(T), stream));
        quantize_mmvq_q8_1<<<dim3((hidden + 255) / 256, rows), 256, 0, stream>>>(input, s.input, hidden);
    } else if (phase == StreamedMoePhase::Finish) {
        ReduceStreamed<<<(rows * hidden + 255) / 256, 256, 0, stream>>>(s.routes, output, scores, rows, hidden, topk);
    } else {
        Workspace w(nullptr, rows, hidden, inter, batch.experts, topk);
        w.capacity = capacity; w.activeRows = batch.rows;
        w.counts = const_cast<int *>(batch.counts); w.offsets = const_cast<int *>(batch.offsets);
        w.tileExperts = const_cast<int *>(batch.tileExperts); w.groupRoutes = const_cast<int *>(batch.routes);
        w.quantized = s.quantized; w.products = s.products;
        GatherQuantized<<<dim3((batch.rows + 7) / 8, ((hidden + 255) / 256) * 2), 256, 0, stream>>>(
            s.input, s.quantized, batch.routes, batch.offsets + batch.experts, hidden, capacity, topk);
        Matrix(gt, batch.weights, 0, w, batch.experts, hidden, 2 * inter, stream);
        ActivateQuantizeStreamed<<<dim3((inter + 255) / 256, batch.rows), 256, 0, stream>>>(
            s.products, s.quantized, gate, batch.routes, inter, capacity);
        Matrix(dt, batch.weights, 1, w, batch.experts, inter, hidden, stream);
        ScatterStreamed<T><<<(batch.rows * hidden + 255) / 256, 256, 0, stream>>>(
            s.products, s.routes, batch.routes, batch.rows, hidden);
    }
    return cudaGetLastError() == cudaSuccess;
}
} // namespace grouped_moe
