// Singleton BF16 controller shared by native and Q8 decode. The fused normalization
// and PDL prologues follow Basalt v0.5.6 (MIT, third_party/basalt/LICENSE).
// https://github.com/jesdga95/basalt/blob/v0.5.6/src/models/inkling2/kernels/glue.cu
// Normalize the two dot products before adding them, avoiding the packed
// destination/message temporary. Hidden coefficients remain FP32 until blended.
__global__ void singletonControllerDown(LoopStreams streams, const __nv_bfloat16 *weights,
                                         float *hidden, int rank, float eps) {
    constexpr int width = 2304, slice = width / 4, pairs = slice / 64;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int stream = blockIdx.y, column = blockIdx.x, base = warp % 4 * slice;
    const auto &inputs = streams.items[stream];
    const auto *w = reinterpret_cast<const __nv_bfloat162 *>(weights +
        (size_t(stream) * rank + column) * (2 * width) + warp / 4 * width + base);
    __nv_bfloat162 coefficients[pairs];
#pragma unroll
    for (int i = 0; i < pairs; ++i) coefficients[i] = w[lane + i * 32];
    cuda_launch::DependencyStart();
    const auto *x = reinterpret_cast<const float2 *>((warp < 4 ? inputs.dest : inputs.msg) + base);
    float square = 0.f, dot = 0.f;
#pragma unroll
    for (int i = 0; i < pairs; ++i) {
        const auto value = x[lane + i * 32];
        const auto weight = __bfloat1622float2(coefficients[i]);
        square = fmaf(value.y, value.y, fmaf(value.x, value.x, square));
        dot = fmaf(value.y, weight.y, fmaf(value.x, weight.x, dot));
    }
    dot = cuda::WarpSum(dot);
    square = cuda::WarpSum(square);
    __shared__ float2 partial[8];
    if (!lane) partial[warp] = make_float2(dot, square);
    __syncthreads();
    if (threadIdx.x == 0) {
        float v[2]{};
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            float d = 0.f, s = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d += partial[half * 4 + i].x;
                s += partial[half * 4 + i].y;
            }
            v[half] = d * rsqrtf(s / width + eps);
        }
        hidden[stream * rank + column] = v[0] + v[1];
    }
}

__global__ void singletonControllerGates(const float *hidden, const __nv_bfloat16 *weights,
                                          float *gates, int width) {
    constexpr int rank = 64, channels = 32, group = 8;
    const int lane = threadIdx.x % group, channel = blockIdx.x * channels + threadIdx.x / group;
    const size_t row = size_t(blockIdx.y) * (2 * width) + channel;
    const auto *w = reinterpret_cast<const uint4 *>(weights + row * rank + lane * group);
    const uint4 coeff = *w;
    cuda_launch::DependencyStart();
    const auto *x = hidden + blockIdx.y * rank + lane * group;
    unsigned bits[4] = {coeff.x, coeff.y, coeff.z, coeff.w};
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = __uint_as_float(bits[i] << 16), b = __uint_as_float(bits[i] & 0xffff0000u);
        v = fmaf(b, x[2*i+1], fmaf(a, x[2*i], v));
    }
#pragma unroll
    for (int d = 1; d < group; d *= 2) v += __shfl_xor_sync(0xffffffff, v, d, group);
    if (!lane) gates[row] = v;
}
