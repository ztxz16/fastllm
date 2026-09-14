#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <vector>

namespace {
constexpr int Threads = 256, Tile = 1024;
struct Partial { float max, sum, best; int token; };
struct Workspace {
    int device = 0;
    Partial *partials = nullptr;
    int *results = nullptr, *hostResults = nullptr;
    float *temperatures = nullptr;
    unsigned long long *counter = nullptr;
    size_t partialCapacity = 0, resultCapacity = 0, temperatureCapacity = 0;
    std::vector<float> savedTemperatures;
    unsigned long long seed = std::mt19937_64(std::random_device{}())();
    Workspace() { cudaGetDevice(&device); }
    ~Workspace() {
        int previous = device;
        cudaGetDevice(&previous); cudaSetDevice(device);
        cudaFree(partials); cudaFree(results); cudaFreeHost(hostResults);
        cudaFree(temperatures); cudaFree(counter);
        cudaSetDevice(previous);
    }
    bool Prepare(size_t partialCount, size_t resultCount) {
        if (!counter) {
            if (cudaMalloc(&counter, sizeof(*counter)) != cudaSuccess ||
                cudaMemsetAsync(counter, 0, sizeof(*counter), cudaStreamPerThread) != cudaSuccess)
                return false;
        }
        if (partialCount > partialCapacity) {
            cudaFree(partials); partials = nullptr; partialCapacity = 0;
            if (cudaMalloc(&partials, partialCount * sizeof(Partial)) != cudaSuccess) return false;
            partialCapacity = partialCount;
        }
        if (resultCount > resultCapacity) {
            cudaFree(results); cudaFreeHost(hostResults);
            results = nullptr; hostResults = nullptr; resultCapacity = 0;
            if (cudaMalloc(&results, resultCount * sizeof(int)) != cudaSuccess ||
                cudaMallocHost(&hostResults, resultCount * sizeof(int)) != cudaSuccess) return false;
            resultCapacity = resultCount;
        }
        return true;
    }
    bool SetTemperatures(const float *t, int batch, bool &uniform) {
        uniform = true;
        for (int b = 0; b < batch; ++b) {
            if (!std::isfinite(t[b]) || t[b] <= 0) return false;
            uniform &= t[b] == t[0];
        }
        if (uniform) return true;
        if ((size_t)batch > temperatureCapacity) {
            cudaFree(temperatures); temperatures = nullptr; temperatureCapacity = 0;
            if (cudaMalloc(&temperatures, batch * sizeof(float)) != cudaSuccess) return false;
            temperatureCapacity = batch; savedTemperatures.clear();
        }
        if (savedTemperatures.size() != (size_t)batch ||
            !std::equal(savedTemperatures.begin(), savedTemperatures.end(), t)) {
            savedTemperatures.assign(t, t + batch);
            // This only uploads on a changed, nonuniform batch. Complete the
            // staging before the next call can resize/reuse the host vector.
            if (cudaMemcpy(temperatures, savedTemperatures.data(), batch * sizeof(float),
                    cudaMemcpyHostToDevice) != cudaSuccess) return false;
        }
        return true;
    }
};
Workspace &GetWorkspace() {
    int device = 0; cudaGetDevice(&device);
    static thread_local std::map<int, std::unique_ptr<Workspace>> workspaces;
    auto &ws = workspaces[device];
    if (!ws) ws.reset(new Workspace);
    return *ws;
}
__device__ float Uniform(unsigned int x) {
    // Put the well-resolved Gumbel tail at u -> 0; avoid either endpoint.
    return fminf(fmaxf((float)x * 0x1p-32f, 0x1p-32f), 0x1.fffffep-1f);
}
__device__ float Gumbel(unsigned int x) { return -logf(-log1pf(-Uniform(x))); }
__device__ bool Better(float x, int id, float y, int other) {
    return x > y || (x == y && id < other);
}
__device__ Partial Reduce(Partial v, Partial *shared) {
    int tid = threadIdx.x;
    shared[tid] = v; __syncthreads();
    for (int stride = Threads / 2; stride; stride >>= 1) {
        if (tid < stride) {
            Partial a = shared[tid], b = shared[tid + stride];
            float m = fmaxf(a.max, b.max);
            a.sum = isfinite(m) ? a.sum * expf(a.max - m) + b.sum * expf(b.max - m) : 0;
            a.max = m;
            if (Better(b.best, b.token, a.best, a.token)) { a.best = b.best; a.token = b.token; }
            shared[tid] = a;
        }
        __syncthreads();
    }
    return shared[0];
}
__global__ void DraftKernel(const float *logits, float *saved, Partial *partials,
        const float *temperatures, float temperature, int vocab, int blocks,
        unsigned long long seed, const unsigned long long *counter) {
    int row = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    float t = temperatures ? temperatures[row] : temperature;
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, ((unsigned long long)row * blocks + tile) * Threads + tid,
                *counter * 4, &rng);
    uint4 noise = curand4(&rng);
    unsigned int randoms[4] = {noise.x, noise.y, noise.z, noise.w};
    Partial local{-INFINITY, 0, -INFINITY, 0x7fffffff};
    float values[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int id = tile * Tile + tid + i * Threads;
        float x = id < vocab ? logits[(size_t)row * vocab + id] / t : -INFINITY;
        values[i] = x;
        if (id < vocab) saved[(size_t)row * vocab + id] = x;
        local.max = fmaxf(local.max, x);
        float score = x + Gumbel(randoms[i]);
        if (id < vocab && Better(score, id, local.best, local.token)) {
            local.best = score; local.token = id;
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (isfinite(local.max)) local.sum += expf(values[i] - local.max);
    __shared__ Partial shared[Threads];
    Partial reduced = Reduce(local, shared);
    if (tid == 0) partials[(size_t)row * blocks + tile] = reduced;
}
__global__ void FinishDraftKernel(const Partial *partials, float *lse, int *output,
        float *floatOutput, int blocks, unsigned long long *counter) {
    int row = blockIdx.x, tid = threadIdx.x;
    Partial v{-INFINITY, 0, -INFINITY, 0x7fffffff};
    for (int b = tid; b < blocks; b += Threads) {
        Partial p = partials[(size_t)row * blocks + b];
        float m = fmaxf(v.max, p.max);
        v.sum = isfinite(m) ? v.sum * expf(v.max - m) + p.sum * expf(p.max - m) : 0;
        v.max = m;
        if (Better(p.best, p.token, v.best, v.token)) { v.best = p.best; v.token = p.token; }
    }
    __shared__ Partial shared[Threads];
    Partial reduced = Reduce(v, shared);
    if (tid == 0) {
        lse[row] = reduced.max + logf(reduced.sum);
        output[row] = isfinite(reduced.best) ? reduced.token : -1;
        if (floatOutput) floatOutput[row] = (float)output[row];
        // Increment on device so graph replays never repeat the same noise.
        if (row == 0) ++*counter;
    }
}
__global__ void AcceptKernel(const float *p, const float *logits, const float *lse,
        const int *draftIds, int *result, int drafts, int vocab, int batch,
        unsigned long long seed, const unsigned long long *counter) {
    if (threadIdx.x) return;
    int b = blockIdx.x, offset = b * (drafts + 1), accepted = 0;
    for (int i = 0; i <= drafts; ++i) result[offset + i] = -1;
    curandStatePhilox4_32_10_t rng;
    // Keep acceptance uniforms disjoint from both draft and residual Gumbels.
    curand_init(seed ^ 0xd2b74407b1ce6e93ULL, b, *counter * 16, &rng);
    for (int i = 0; i < drafts; ++i) {
        int token = draftIds[b * drafts + i];
        if (token < 0 || token >= vocab) break;
        float target = p[(size_t)(offset + i) * vocab + token];
        float logq = logits[(size_t)(b * drafts + i) * vocab + token] - lse[b * drafts + i];
        float logp = logf(target);
        float logu = logf(Uniform(curand(&rng)));
        // Subtract first: adding a tiny log(u) to a large negative log(q)
        // can round back to log(q), spuriously rejecting p == q.
        if (!(target > 0 && (logp >= logq || logu < logp - logq))) break;
        result[offset + i] = token; ++accepted;
    }
    result[batch * (drafts + 1) + b] = accepted;
}
__global__ void ResidualKernel(const float *p, const float *logits, const float *lse,
        const int *result, Partial *partials, int drafts, int vocab, int batch,
        int blocks, unsigned long long seed, const unsigned long long *counter) {
    int b = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    int accepted = result[batch * (drafts + 1) + b];
    const float *target = p + (size_t)(b * (drafts + 1) + accepted) * vocab;
    bool bonus = accepted == drafts;
    const float *proposal = logits + (size_t)(b * drafts + accepted) * vocab;
    float normalizer = bonus ? 0 : lse[b * drafts + accepted];
    curandStatePhilox4_32_10_t rng;
    curand_init(seed ^ 0x9e3779b97f4a7c15ULL,
        ((unsigned long long)b * blocks + tile) * Threads + tid, *counter * 4, &rng);
    uint4 noise = curand4(&rng);
    unsigned int randoms[4] = {noise.x, noise.y, noise.z, noise.w};
    Partial local{-INFINITY, 0, -INFINITY, 0x7fffffff};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int token = tile * Tile + tid + i * Threads;
        if (token < vocab) {
            float logp = logf(target[token]), score = logp;
            if (!bonus) {
                float delta = proposal[token] - normalizer - logp;
                // expm1 retains small positive residuals when p and q are close.
                score = delta < 0 ? logp + logf(-expm1f(delta)) : -INFINITY;
            }
            score += Gumbel(randoms[i]);
            if (Better(score, token, local.best, local.token)) {
                local.best = score; local.token = token;
            }
        }
    }
    __shared__ Partial shared[Threads];
    Partial reduced = Reduce(local, shared);
    if (tid == 0) partials[(size_t)b * blocks + tile] = reduced;
}
__global__ void FinishResidualKernel(const Partial *partials, int *result,
        int drafts, int batch, int blocks, unsigned long long *counter) {
    int b = blockIdx.x, tid = threadIdx.x;
    Partial local{-INFINITY, 0, -INFINITY, 0x7fffffff};
    for (int i = tid; i < blocks; i += Threads) {
        Partial p = partials[(size_t)b * blocks + i];
        if (Better(p.best, p.token, local.best, local.token)) { local.best = p.best; local.token = p.token; }
    }
    __shared__ Partial shared[Threads];
    Partial reduced = Reduce(local, shared);
    if (tid == 0) {
        int accepted = result[batch * (drafts + 1) + b];
        result[b * (drafts + 1) + accepted] = isfinite(reduced.best) ? reduced.token : -1;
        if (b == 0) ++*counter;
    }
}
} // namespace

bool FastllmCudaMtpSampleDraftLogits(const float *logits, float *saved, float *lse,
        int *output, float *floatOutput, const float *temperatures, int batch, int vocab) {
    if (!logits || !saved || !lse || !output || !temperatures || batch <= 0 || vocab <= 0) return false;
    auto &ws = GetWorkspace();
    int blocks = (vocab + Tile - 1) / Tile;
    bool uniform = false;
    if (!ws.Prepare((size_t)batch * blocks, 0) ||
        !ws.SetTemperatures(temperatures, batch, uniform)) return false;
    DraftKernel<<<dim3(blocks, batch), Threads, 0, cudaStreamPerThread>>>(
        logits, saved, ws.partials, uniform ? nullptr : ws.temperatures,
        temperatures[0], vocab, blocks, ws.seed, ws.counter);
    FinishDraftKernel<<<batch, Threads, 0, cudaStreamPerThread>>>(
        ws.partials, lse, output, floatOutput, blocks, ws.counter);
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaMtpRejectionFromProbs(const float *p, const float *logits,
        const float *lse, const int *draftIds, int *output, int *accepted,
        int batch, int drafts, int vocab) {
    if (!p || !logits || !lse || !draftIds || !output || !accepted ||
        batch <= 0 || drafts <= 0 || drafts > 8 || vocab <= 0) return false;
    auto &ws = GetWorkspace();
    int blocks = (vocab + Tile - 1) / Tile, count = batch * (drafts + 2);
    if (!ws.Prepare((size_t)batch * blocks, count)) return false;
    AcceptKernel<<<batch, 32, 0, cudaStreamPerThread>>>(p, logits, lse, draftIds,
        ws.results, drafts, vocab, batch, ws.seed, ws.counter);
    ResidualKernel<<<dim3(blocks, batch), Threads, 0, cudaStreamPerThread>>>(
        p, logits, lse, ws.results, ws.partials, drafts, vocab, batch, blocks, ws.seed, ws.counter);
    FinishResidualKernel<<<batch, Threads, 0, cudaStreamPerThread>>>(
        ws.partials, ws.results, drafts, batch, blocks, ws.counter);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(ws.hostResults, ws.results, count * sizeof(int), cudaMemcpyDeviceToHost,
            cudaStreamPerThread) != cudaSuccess || cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess)
        return false;
    std::copy_n(ws.hostResults, batch * (drafts + 1), output);
    std::copy_n(ws.hostResults + batch * (drafts + 1), batch, accepted);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i <= accepted[b]; ++i)
            if (output[b * (drafts + 1) + i] < 0 || output[b * (drafts + 1) + i] >= vocab) return false;
    return true;
}
