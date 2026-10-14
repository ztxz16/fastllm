// Included inside glm5_gguf_cache. Dispatch checks SM75, canonical weights,
// hidden=4096, inter=2048 and enough aligned scratch before using these kernels.
__global__ void PackIQActivation(const float *input, float *output, int columns) {
    const int t = threadIdx.x, lane = (t % 32 / 8) * 8 + t / 32;
    const size_t block = size_t(blockIdx.y) * columns + blockIdx.x * 256;
    output[block + (t % 8 / 4) * 128 + lane * 4 + t % 4] = input[block + t];
}

struct IQ2XXSBlock {
    uint64_t grid;
    unsigned signs;
    float d;
    __device__ __forceinline__ IQ2XXSBlock(const block_iq2_xxs &q, int lane, const uint64_t *table) {
        const int il = lane / 8, ib = lane % 8;
        const uint32_t part = __ldg(q.qs + lane);
        const uint32_t low = __shfl_sync(0xffffffff, part, 4 * ib) |
            (__shfl_sync(0xffffffff, part, 4 * ib + 1) << 16);
        const uint32_t aux = __shfl_sync(0xffffffff, part, 4 * ib + 2) |
            (__shfl_sync(0xffffffff, part, 4 * ib + 3) << 16);
        grid = table[(low >> (8 * il)) & 255];
        signs = (aux >> (7 * il)) & 127;
        signs |= (__popc(signs) & 1) << 7;
        d = __half2float(q.d) * (0.5f + (aux >> 28)) * 0.25f;
    }
    __device__ __forceinline__ float value(int j) const {
        const int q = int((grid >> (8u * j)) & 255);
        return d * q * ((signs & (1u << j)) ? -1.f : 1.f);
    }
};

template<class View, bool Tp = false>
__global__ void GateIQ2XXSSm75(const float *__restrict__ input, __nv_bfloat16 *activation,
        View view, const float *scores, float limit, int topk, int rank = 0) {
    const int localRow = threadIdx.x / 64, row = blockIdx.x * 2 + localRow;
    const int outputRow = row + (Tp ? rank * 1024 : 0);
    const int route = blockIdx.y, lane = threadIdx.x % 32, warp = threadIdx.x / 32 % 2;
    const int original = OriginalRoute(view, route);
    const auto *record = ExpertWeight<true>(view, route);
    if (!record) {
        if (threadIdx.x % 64 == 0) activation[size_t(route) * 2048 + outputRow] = __float2bfloat16_rn(0);
        return;
    }
    __shared__ uint64_t table[256];
    for (int i = threadIdx.x; i < 256; i += 128) table[i] = iq2xxs_grid[i];
    __syncthreads();
    input += size_t(original / topk) * 4096;
    const auto *g = reinterpret_cast<const block_iq2_xxs *>(record) + row * 16;
    const auto *u = g + (Tp ? 1024 : 2048) * 16;
    // Two physical warps retain the original four logical accumulators and
    // their FMA/reduction order, including the final BF16 rounding boundary.
    float gs[2] = {}, us[2] = {};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int b = warp + p * 2 + 4 * k;
            const IQ2XXSBlock gg(g[b], lane, table), uu(u[b], lane, table);
            const float4 x0 = __ldg(reinterpret_cast<const float4 *>(input + b * 256 + lane * 4));
            const float4 x1 = __ldg(reinterpret_cast<const float4 *>(input + b * 256 + 128 + lane * 4));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 v = j < 4 ? x0 : x1;
                const float x = j % 4 == 0 ? v.x : j % 4 == 1 ? v.y : j % 4 == 2 ? v.z : v.w;
                gs[p] = fmaf(gg.value(j), x, gs[p]);
                us[p] = fmaf(uu.value(j), x, us[p]);
            }
        }
    }
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            gs[p] += __shfl_down_sync(0xffffffff, gs[p], mask);
            us[p] += __shfl_down_sync(0xffffffff, us[p], mask);
        }
    }
    __shared__ float gparts[8], uparts[8];
    if (lane == 0) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            gparts[localRow * 4 + warp + p * 2] = gs[p];
            uparts[localRow * 4 + warp + p * 2] = us[p];
        }
    }
    __syncthreads();
    if (threadIdx.x % 64 == 0) {
        float gate = 0, up = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            gate += gparts[localRow * 4 + p];
            up += uparts[localRow * 4 + p];
        }
        gate = __bfloat162float(__float2bfloat16_rn(gate));
        up = __bfloat162float(__float2bfloat16_rn(up));
        if (limit > 0) { gate = fminf(gate, limit); up = fmaxf(-limit, fminf(up, limit)); }
        const float value = __fmul_rn(__fmul_rn(gate / (1.f + expf(-gate)), up), scores[original]);
        activation[size_t(route) * 2048 + outputRow] = __float2bfloat16_rn(value);
    }
}

template<class View, bool Tp = false>
__global__ __launch_bounds__(128, 8) void DownIQ3XXSSm75(const float *__restrict__ input, float *output, View view,
        int rank = 0) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32, route = blockIdx.y, lane = threadIdx.x % 32;
    const auto *record = ExpertWeight<false>(view, route);
    float sum = 0;
    __shared__ uint32_t table[256];
    if (record) {
        for (int i = threadIdx.x; i < 256; i += 128) table[i] = iq3xxs_grid[i];
        __syncthreads();
        const auto *w = reinterpret_cast<const block_iq3_xxs *>(record) + row * 8;
        input += size_t(route) * 2048;
        const int il = lane / 8, ib = lane % 8;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const float4 x0 = __ldg(reinterpret_cast<const float4 *>(input + b * 256 + lane * 4));
            const float4 x1 = __ldg(reinterpret_cast<const float4 *>(input + b * 256 + 128 + lane * 4));
            const auto &q = w[b];
            const uint32_t codes = __ldg(reinterpret_cast<const uint16_t *>(q.qs) + lane);
            const uint32_t pair = __shfl_sync(0xffffffff, codes, 4 * ib + il);
            const uint32_t part = __ldg(reinterpret_cast<const uint16_t *>(q.qs + 64) + lane % 16);
            const uint32_t aux = __shfl_sync(0xffffffff, part, 2 * ib) |
                (__shfl_sync(0xffffffff, part, 2 * ib + 1) << 16);
            const uint32_t g0 = table[pair & 255], g1 = table[pair >> 8];
            unsigned signs = (aux >> (7 * il)) & 127;
            signs |= (__popc(signs) & 1) << 7;
            const float d = __half2float(q.d) * (0.5f + (aux >> 28)) * 0.5f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int q0 = int((g0 >> (8u * j)) & 255), q1 = int((g1 >> (8u * j)) & 255);
                const float s0 = (signs & (1u << j)) ? -1.f : 1.f;
                const float s1 = (signs & (1u << (j + 4))) ? -1.f : 1.f;
                sum = fmaf(d * q0 * s0, j == 0 ? x0.x : j == 1 ? x0.y : j == 2 ? x0.z : x0.w, sum);
                sum = fmaf(d * q1 * s1, j == 0 ? x1.x : j == 1 ? x1.y : j == 2 ? x1.z : x1.w, sum);
            }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) sum += __shfl_down_sync(0xffffffff, sum, mask);
    }
    if (lane == 0)
        output[size_t(OriginalRoute(view, route)) * 4096 + row + (Tp ? rank * 2048 : 0)] =
            __bfloat162float(__float2bfloat16_rn(sum));
}
