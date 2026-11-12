// Derived from Strata src/kernels/cuda/fused_gdn.cu at
// 9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e. See third_party/strata.
/*
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once
namespace fastllm_strata_gdn {
constexpr int S = 128;
struct Request {
    const float *qkv;
    const half *alpha, *beta, *gate;
    const float *state;
    float *nextState;
    half *output;
};
struct Requests {
    static constexpr int CAPACITY = 32;
    Request items[CAPACITY];
};
// FastLLM adapter: [head,K,V] state and grouped value heads, explicit
// state input/output for safe tuning, original Q/K normalization and FP16
// output boundaries. Independent requests occupy grid.y; no logical row cap.
template<int RG>
__global__ __launch_bounds__(S * RG) void step_norm_kernel(Requests requests,
        const float *aLog, const float *dtBias, const float *gamma,
        int keyHeads, int valueHeads, float recurrentEps, float normEps,
        float inverseHead, bool tiledOutput) {
    constexpr int RPG = S / RG;
    static_assert(RG == 1 || RG == 2 || RG == 4);
    const Request &a = requests.items[blockIdx.y];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y;
    const int tid = rg * S + col, repeat = valueHeads / keyHeads;
    const int qh = head / repeat;
    const float *q = a.qkv + qh * S;
    const float *k = a.qkv + (keyHeads + qh) * S;
    const float *v = a.qkv + (2 * keyHeads + head) * S;
    __shared__ float sk[S], sq[S], scales[2], gates[2];
    __shared__ float red_kv[RG][S], red_o[RG][S];
    __shared__ __align__(8) float core[S];
    float s[RPG];
    const size_t offset = (size_t)head * S * S + rg * RPG * S + col;
    const float *base = a.state + offset;
    float *next = a.nextState + offset;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * S];
    if (tid < 32) {
        const float4 qv = reinterpret_cast<const float4 *>(q)[tid];
        const float4 kv = reinterpret_cast<const float4 *>(k)[tid];
        float qs = qv.x*qv.x + qv.y*qv.y + qv.z*qv.z + qv.w*qv.w;
        float ks = kv.x*kv.x + kv.y*kv.y + kv.z*kv.z + kv.w*kv.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            qs += __shfl_down_sync(0xffffffffu, qs, o);
            ks += __shfl_down_sync(0xffffffffu, ks, o);
        }
        if (tid == 0) {
            scales[0] = rsqrtf(qs / S + recurrentEps);
            scales[1] = rsqrtf(ks / S + recurrentEps);
        }
    }
    if (tid == 0) {
        const float alpha = __half2float(a.alpha[head]) + dtBias[head];
        const float beta = __half2float(a.beta[head]);
        const float softplus = alpha > 20.f ? alpha : log1pf(expf(alpha));
        gates[0] = 1.f / (1.f + expf(-beta));
        gates[1] = expf(-expf(aLog[head]) * softplus);
    }
    __syncthreads();
    if (rg == 0) {
        sq[col] = q[col] * scales[0] * inverseHead;
        sk[col] = k[col] * scales[1] * inverseHead;
    }
    __syncthreads();
    if (rg == 0) sq[col] *= inverseHead;
    __syncthreads();
    const float g = gates[1];
    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
    red_kv[rg][col] = kv;
    __syncthreads();
    float kv_col = red_kv[0][col];
#pragma unroll
    for (int group = 1; group < RG; ++group) kv_col += red_kv[group][col];
    const float delta = (v[col] - g * kv_col) * gates[0];
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
        o = fmaf(s[r], sq[rg * RPG + r], o);
        next[r * S] = s[r];
    }
    red_o[rg][col] = o;
    __syncthreads();
    if (rg == 0) {
        float oc = red_o[0][col];
#pragma unroll
        for (int group = 1; group < RG; ++group) oc += red_o[group][col];
        core[col] = oc;
    }
    __syncthreads();
    if (tid < 32) {
        const int outputHead = tiledOutput ? (head % repeat) * keyHeads + qh : head;
        Qwen4GdnOutputGateExactDevice(core, gamma, a.gate + head * S,
            a.output + outputHead * S, normEps);
    }
}
} // namespace fastllm_strata_gdn
