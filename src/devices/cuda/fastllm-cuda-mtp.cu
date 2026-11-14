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
#include <tuple>
#include <climits>
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
    void *descriptors = nullptr;
    size_t descriptorCapacity = 0;
    size_t partialCapacity = 0, resultCapacity = 0, temperatureCapacity = 0;
    std::vector<float> savedTemperatures;
    unsigned long long seed = std::mt19937_64(std::random_device{}())();
    Workspace() { cudaGetDevice(&device); }
    ~Workspace() {
        int previous = device;
        cudaGetDevice(&previous); cudaSetDevice(device);
        cudaFree(partials); cudaFree(results); cudaFreeHost(hostResults);
        cudaFree(temperatures); cudaFree(counter); cudaFree(descriptors);
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
    bool SetDescriptors(const void *host, size_t bytes) {
        if (bytes > descriptorCapacity) {
            cudaFree(descriptors); descriptors = nullptr; descriptorCapacity = 0;
            if (cudaMalloc(&descriptors, bytes) != cudaSuccess) return false;
            descriptorCapacity = bytes;
        }
        // Complete staging before the caller reuses its host descriptor array.
        return cudaMemcpy(descriptors, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
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
template<bool Indirect = false>
__global__ void DraftKernel(const float *logits, float *saved, Partial *partials,
        const float *temperatures, float temperature, int vocab, int blocks,
        unsigned long long seed, const unsigned long long *counter,
        const FastllmMtpDraftOutput *outputs = nullptr) {
    int row = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    float *savedRow;
    if constexpr (Indirect) savedRow = outputs[row].logits;
    else savedRow = saved + (size_t)row * vocab;
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
        if (id < vocab) savedRow[id] = x;
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
template<bool Indirect = false>
__global__ void FinishDraftKernel(const Partial *partials, float *lse, int *output,
        float *floatOutput, int blocks, unsigned long long *counter,
        const FastllmMtpDraftOutput *outputs = nullptr) {
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
        const float normalizer = reduced.max + logf(reduced.sum);
        const int token = isfinite(reduced.best) ? reduced.token : -1;
        if constexpr (Indirect) {
            *outputs[row].logsumexp = normalizer;
            *outputs[row].token = token;
            if (outputs[row].floatToken) *outputs[row].floatToken = (float)token;
            if (output) output[row] = token;
        } else {
            lse[row] = normalizer;
            output[row] = token;
            if (floatOutput) floatOutput[row] = (float)token;
        }
        // Increment on device so graph replays never repeat the same noise.
        if (row == 0) ++*counter;
    }
}
template<bool Indirect = false, bool Probability = false>
__global__ void AcceptKernel(const float *p, const float *logits, const float *lse,
        const int *draftIds, int *result, int drafts, int vocab, int batch,
        unsigned long long seed, const unsigned long long *counter,
        const FastllmMtpProposalView *proposals = nullptr) {
    if (threadIdx.x) return;
    int b = blockIdx.x, offset = b * (drafts + 1), accepted = 0;
    const float *qLogits, *qLse;
    const int *ids;
    if constexpr (Indirect) {
        qLogits = proposals[b].logits; qLse = proposals[b].logsumexp;
        ids = proposals[b].tokens;
    } else {
        qLogits = logits + (size_t)b * drafts * vocab;
        qLse = Probability ? nullptr : lse + b * drafts; ids = draftIds + b * drafts;
    }
    for (int i = 0; i <= drafts; ++i) result[offset + i] = -1;
    curandStatePhilox4_32_10_t rng;
    // Keep acceptance uniforms disjoint from both draft and residual Gumbels.
    curand_init(seed ^ 0xd2b74407b1ce6e93ULL, b, *counter * (((uint64_t)drafts + 15) / 16 * 16), &rng);
    for (int i = 0; i < drafts; ++i) {
        int token = ids[i];
        if (token < 0 || token >= vocab) break;
        float target = p[(size_t)(offset + i) * vocab + token];
        if constexpr (Probability) {
            float proposal = qLogits[(size_t)i * vocab + token];
            float u = Uniform(curand(&rng));
            if (!(target > 0 && (target >= proposal || u * proposal < target))) break;
        } else {
            float logq = qLogits[(size_t)i * vocab + token] - qLse[i];
            float logp = logf(target);
            float logu = logf(Uniform(curand(&rng)));
            // Subtract first: adding a tiny log(u) to a large negative log(q)
            // can round back to log(q), spuriously rejecting p == q.
            if (!(target > 0 && (logp >= logq || logu < logp - logq))) break;
        }
        result[offset + i] = token; ++accepted;
    }
    result[batch * (drafts + 1) + b] = accepted;
}
template<bool Indirect = false, bool Probability = false>
__global__ void ResidualKernel(const float *p, const float *logits, const float *lse,
        const int *result, Partial *partials, int drafts, int vocab, int batch,
        int blocks, unsigned long long seed, const unsigned long long *counter,
        const FastllmMtpProposalView *proposals = nullptr) {
    int b = blockIdx.y, tile = blockIdx.x, tid = threadIdx.x;
    int accepted = result[batch * (drafts + 1) + b];
    const float *target = p + (size_t)(b * (drafts + 1) + accepted) * vocab;
    bool bonus = accepted == drafts;
    const float *proposal = nullptr;
    float normalizer = 0;
    if (!bonus) {
        if constexpr (Indirect) {
            proposal = proposals[b].logits + (size_t)accepted * vocab;
            if constexpr (!Probability) normalizer = proposals[b].logsumexp[accepted];
        } else {
            proposal = logits + (size_t)(b * drafts + accepted) * vocab;
            if constexpr (!Probability) normalizer = lse[b * drafts + accepted];
        }
    }
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
                if constexpr (Probability) score = logf(fmaxf(target[token] - proposal[token], 0.0f));
                else {
                    float delta = proposal[token] - normalizer - logp;
                    // expm1 retains small positive residuals when p and q are close.
                    score = delta < 0 ? logp + logf(-expm1f(delta)) : -INFINITY;
                }
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
    DraftKernel<false><<<dim3(blocks, batch), Threads, 0, cudaStreamPerThread>>>(
        logits, saved, ws.partials, uniform ? nullptr : ws.temperatures,
        temperatures[0], vocab, blocks, ws.seed, ws.counter);
    FinishDraftKernel<false><<<batch, Threads, 0, cudaStreamPerThread>>>(
        ws.partials, lse, output, floatOutput, blocks, ws.counter);
    return cudaGetLastError() == cudaSuccess;
}

template<bool Probability>
static bool FastllmMtpReject(const float *p, const float *logits,
        const float *lse, const int *draftIds, int *output, int *accepted,
        int batch, int drafts, int vocab) {
    if (!p || !logits || (!Probability && !lse) || !draftIds || !output || !accepted ||
        batch <= 0 || drafts <= 0 || drafts > 8 || vocab <= 0) return false;
    auto &ws = GetWorkspace();
    int blocks = (vocab + Tile - 1) / Tile, count = batch * (drafts + 2);
    if (!ws.Prepare((size_t)batch * blocks, count)) return false;
    AcceptKernel<false, Probability><<<batch, 32, 0, cudaStreamPerThread>>>(p, logits, lse, draftIds,
        ws.results, drafts, vocab, batch, ws.seed, ws.counter);
    ResidualKernel<false, Probability><<<dim3(blocks, batch), Threads, 0, cudaStreamPerThread>>>(
        p, logits, lse, ws.results, ws.partials, drafts, vocab, batch, blocks, ws.seed, ws.counter);
    FinishResidualKernel<<<batch, Threads, 0, cudaStreamPerThread>>>(
        ws.partials, ws.results, drafts, batch, blocks, ws.counter);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(ws.hostResults, ws.results, count * sizeof(int), cudaMemcpyDeviceToHost,
            cudaStreamPerThread) != cudaSuccess || cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess)
        return false;
    std::copy_n(ws.hostResults, batch * (drafts + 1), output);
    std::copy_n(ws.hostResults + batch * (drafts + 1), batch, accepted);
    for (int b = 0; b < batch; ++b) {
        if (accepted[b] < 0 || accepted[b] > drafts) return false;
        for (int i = 0; i <= accepted[b]; ++i)
            if (output[b * (drafts + 1) + i] < 0 || output[b * (drafts + 1) + i] >= vocab) return false;
    }
    return true;
}

bool FastllmCudaMtpRejectionFromProbs(const float *p, const float *logits,
        const float *lse, const int *draftIds, int *output, int *accepted,
        int batch, int drafts, int vocab) {
    return FastllmMtpReject<false>(p, logits, lse, draftIds, output, accepted, batch, drafts, vocab);
}

bool FastllmCudaMtpRejectionFromProbabilities(const float *p, const float *q,
        const int *draftIds, int *output, int *accepted, int batch, int drafts, int vocab) {
    return FastllmMtpReject<true>(p, q, nullptr, draftIds, output, accepted, batch, drafts, vocab);
}

bool FastllmCudaMtpSampleDraftLogitsBatch(const float *logits,
        const FastllmMtpDraftOutput *outputs, const float *temperatures,
        int *hostTokens, int batch, int vocab) {
    if (!logits || !outputs || !temperatures || batch <= 0 || vocab <= 0) return false;
    for (int b = 0; b < batch; ++b)
        if (!outputs[b].logits || !outputs[b].logsumexp || !outputs[b].token) return false;
    auto &ws = GetWorkspace();
    int blocks = (vocab + Tile - 1) / Tile;
    bool uniform = false;
    if (!ws.Prepare((size_t)batch * blocks, hostTokens ? batch : 0) ||
        !ws.SetTemperatures(temperatures, batch, uniform) ||
        !ws.SetDescriptors(outputs, (size_t)batch * sizeof(*outputs))) return false;
    auto *deviceOutputs = static_cast<const FastllmMtpDraftOutput*>(ws.descriptors);
    DraftKernel<true><<<dim3(blocks, batch), Threads, 0, cudaStreamPerThread>>>(
        logits, nullptr, ws.partials, uniform ? nullptr : ws.temperatures,
        temperatures[0], vocab, blocks, ws.seed, ws.counter, deviceOutputs);
    FinishDraftKernel<true><<<batch, Threads, 0, cudaStreamPerThread>>>(
        ws.partials, nullptr, hostTokens ? ws.results : nullptr, nullptr,
        blocks, ws.counter, deviceOutputs);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (hostTokens) {
        if (cudaMemcpyAsync(ws.hostResults, ws.results, batch * sizeof(int),
                cudaMemcpyDeviceToHost, cudaStreamPerThread) != cudaSuccess ||
            cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) return false;
        std::copy_n(ws.hostResults, batch, hostTokens);
    }
    return true;
}

bool FastllmCudaMtpRejectionFromProbsBatch(const float *p,
        const FastllmMtpProposalView *proposals, int *output, int *accepted,
        int batch, int drafts, int vocab) {
    if (!p || !proposals || !output || !accepted || batch <= 0 ||
        drafts <= 0 || drafts > 8 || vocab <= 0) return false;
    for (int b = 0; b < batch; ++b)
        if (!proposals[b].logits || !proposals[b].logsumexp || !proposals[b].tokens) return false;
    auto &ws = GetWorkspace();
    int blocks = (vocab + Tile - 1) / Tile, count = batch * (drafts + 2);
    if (!ws.Prepare((size_t)batch * blocks, count) ||
        !ws.SetDescriptors(proposals, (size_t)batch * sizeof(*proposals))) return false;
    auto *deviceProposals = static_cast<const FastllmMtpProposalView*>(ws.descriptors);
    AcceptKernel<true><<<batch, 32, 0, cudaStreamPerThread>>>(
        p, nullptr, nullptr, nullptr, ws.results, drafts, vocab, batch,
        ws.seed, ws.counter, deviceProposals);
    ResidualKernel<true><<<dim3(blocks, batch), Threads, 0, cudaStreamPerThread>>>(
        p, nullptr, nullptr, ws.results, ws.partials, drafts, vocab, batch,
        blocks, ws.seed, ws.counter, deviceProposals);
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

namespace {
struct VerifierShape {
    int device, batch, drafts, vocab;
    bool probabilities;
    bool operator<(const VerifierShape &o) const {
        return std::tie(device, batch, drafts, vocab, probabilities) <
            std::tie(o.device, o.batch, o.drafts, o.vocab, o.probabilities);
    }
    uint64_t Weight() const { return uint64_t(batch) * (uint64_t(drafts) + 1) * vocab; }
};
struct VerifierState {
    VerifierShape shape;
    Workspace ws;
    void *distribution = nullptr;
    cudaEvent_t done = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    float *boundLogits = nullptr;
    bool pending = false, graphUnavailable = false, descriptorsReady = false;
    FastllmMtpProposalView *pinnedProposals = nullptr;
    uint64_t lastUse = 0;
    explicit VerifierState(VerifierShape shape) : shape(shape) {}
    void ResetGraph() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        exec = nullptr; graph = nullptr; graphUnavailable = false;
    }
    ~VerifierState() {
        int previous; cudaGetDevice(&previous); cudaSetDevice(shape.device);
        if (pending) cudaEventSynchronize(done);
        ResetGraph();
        if (done) cudaEventDestroy(done);
        cudaFreeHost(pinnedProposals);
        FastllmCudaMtpDestroyTargetDistribution(distribution);
        cudaSetDevice(previous);
    }
    bool Prepare(const FastllmMtpProposalView *views) {
        const int blocks = (shape.vocab + Tile - 1) / Tile;
        const size_t descriptorBytes = size_t(shape.batch) * sizeof(*views);
        if (!distribution) distribution = FastllmCudaMtpCreateTargetDistribution(
            shape.batch * (shape.drafts + 1), shape.vocab);
        if (!distribution || !ws.Prepare(size_t(shape.batch) * blocks,
                size_t(shape.batch) * (shape.drafts + 2))) return false;
        if (!done && cudaEventCreateWithFlags(&done, cudaEventDisableTiming) != cudaSuccess) return false;
        if (!pinnedProposals && cudaMallocHost(&pinnedProposals, descriptorBytes) != cudaSuccess) return false;
        if (!ws.descriptors && cudaMalloc(&ws.descriptors, descriptorBytes) != cudaSuccess) return false;
        ws.descriptorCapacity = descriptorBytes;
        bool changed = !descriptorsReady;
        if (!changed) for (int b = 0; b < shape.batch; ++b)
            changed |= views[b].logits != pinnedProposals[b].logits ||
                views[b].logsumexp != pinnedProposals[b].logsumexp ||
                views[b].tokens != pinnedProposals[b].tokens;
        if (changed) {
            // The resolved request owns this pinned cache; a second host copy
            // is unnecessary. Publish it only after successful submission.
            descriptorsReady = false;
            std::copy_n(views, shape.batch, pinnedProposals);
            if (cudaMemcpyAsync(ws.descriptors, pinnedProposals, descriptorBytes,
                    cudaMemcpyHostToDevice, cudaStreamPerThread) != cudaSuccess) return false;
            descriptorsReady = true;
        }
        return true;
    }
    bool Run(float *logits) {
        float *p = nullptr;
        if (!FastllmCudaMtpNormalizeTargetDistribution(distribution, logits, &p)) return false;
        const int batch = shape.batch, drafts = shape.drafts, vocab = shape.vocab;
        const int blocks = (vocab + Tile - 1) / Tile;
        auto *views = static_cast<const FastllmMtpProposalView *>(ws.descriptors);
        if (shape.probabilities) {
            AcceptKernel<true, true><<<batch, 32, 0, cudaStreamPerThread>>>(p, nullptr, nullptr, nullptr,
                ws.results, drafts, vocab, batch, ws.seed, ws.counter, views);
            ResidualKernel<true, true><<<dim3(blocks, batch), Threads, 0, cudaStreamPerThread>>>(
                p, nullptr, nullptr, ws.results, ws.partials, drafts, vocab, batch, blocks, ws.seed, ws.counter, views);
        } else {
            AcceptKernel<true><<<batch, 32, 0, cudaStreamPerThread>>>(p, nullptr, nullptr, nullptr,
                ws.results, drafts, vocab, batch, ws.seed, ws.counter, views);
            ResidualKernel<true><<<dim3(blocks, batch), Threads, 0, cudaStreamPerThread>>>(
                p, nullptr, nullptr, ws.results, ws.partials, drafts, vocab, batch, blocks, ws.seed, ws.counter, views);
        }
        FinishResidualKernel<<<batch, Threads, 0, cudaStreamPerThread>>>(
            ws.partials, ws.results, drafts, batch, blocks, ws.counter);
        return cudaGetLastError() == cudaSuccess && cudaMemcpyAsync(ws.hostResults,
            ws.results, size_t(batch) * (drafts + 2) * sizeof(int), cudaMemcpyDeviceToHost,
            cudaStreamPerThread) == cudaSuccess;
    }
};
struct VerifierPlan {
    std::map<VerifierShape, std::unique_ptr<VerifierState>> shapes;
    VerifierState *active = nullptr;
    uint64_t peak = 0, clock = 0;
    void Trim(const VerifierShape &incoming) {
        peak = std::max(peak, incoming.Weight());
        uint64_t retained = incoming.Weight();
        for (const auto &s : shapes) retained += s.first.Weight();
        // Geometric batches sum to less than two peak working sets. Leave
        // room for a non-geometric maximum and a drain shape as well; otherwise
        // 4 -> 3 -> 2 -> 1 can evict the hot 4-request graph before the next wave.
        // Bound storage independently of shape history and discard transient
        // shapes first. Different draft/vocabulary families still share the cap.
        auto retainedShape = [&](const VerifierShape &s) {
            return (s.batch & (s.batch - 1)) == 0 || s.Weight() == peak;
        };
        while (retained > 4 * peak && !shapes.empty()) {
            auto oldest = std::min_element(shapes.begin(), shapes.end(), [&](const auto &a, const auto &b) {
                const bool keepA = retainedShape(a.first), keepB = retainedShape(b.first);
                return keepA != keepB ? !keepA : a.second->lastUse < b.second->lastUse;
            });
            retained -= oldest->first.Weight(); shapes.erase(oldest);
        }
    }
};
}

void *FastllmCudaMtpCreateVerifier() { return new VerifierPlan; }
void FastllmCudaMtpDestroyVerifier(void *plan) { delete static_cast<VerifierPlan *>(plan); }

static bool MtpVerifierEnqueue(void *opaque, float *logits,
        const FastllmMtpProposalView *proposals, const float *temperatures,
        const int *topKs, const float *topPs, int batch, int drafts, int vocab, bool useGraph, bool probabilities) {
    if (!opaque || !logits || !proposals || !temperatures || !topKs || !topPs ||
        batch <= 0 || drafts <= 0 || vocab <= 0 ||
        uint64_t(batch) * (uint64_t(drafts) + 2) > INT_MAX) return false;
    auto &plan = *static_cast<VerifierPlan *>(opaque);
    if (plan.active) return false;
    for (int b = 0; b < batch; ++b)
        if (!proposals[b].logits || (!probabilities && !proposals[b].logsumexp) || !proposals[b].tokens) return false;
    for (int row = 0; row < batch * (drafts + 1); ++row)
        if (!std::isfinite(temperatures[row]) || temperatures[row] <= 0 ||
            !std::isfinite(topPs[row]) || topPs[row] <= 0 || topPs[row] > 1) return false;
    int device; if (cudaGetDevice(&device) != cudaSuccess) return false;
    VerifierShape shape{device, batch, drafts, vocab, probabilities};
    if (!plan.shapes.count(shape)) plan.Trim(shape);
    auto &owned = plan.shapes[shape];
    if (!owned) owned.reset(new VerifierState(shape));
    auto &state = *owned;
    state.lastUse = ++plan.clock;
    bool changed = false;
    auto fail = [&] { cudaStreamSynchronize(cudaStreamPerThread); return false; };
    if (!state.Prepare(proposals) || !FastllmCudaMtpPrepareTargetDistribution(
            state.distribution, temperatures, topKs, topPs, &changed)) return fail();
    if (changed) state.ResetGraph();
    if (state.boundLogits != logits) {
        if (state.exec && !FastllmCudaMtpRebindTargetDistribution(state.graph, state.exec, logits))
            state.ResetGraph();
        state.boundLogits = logits;
    }
    if (useGraph && !state.exec && !state.graphUnavailable) {
        bool captured = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (captured) {
            bool ran = state.Run(logits);
            cudaError_t ended = cudaStreamEndCapture(cudaStreamPerThread, &state.graph);
            captured = ran && ended == cudaSuccess && state.graph &&
                cudaGraphInstantiate(&state.exec, state.graph, nullptr, nullptr, 0) == cudaSuccess;
        }
        if (!captured) {
            state.ResetGraph(); state.graphUnavailable = true;
            cudaGetLastError(); // Failed capture submitted no sampling/RNG work.
        }
    }
    const bool launched = useGraph && state.exec
        ? cudaGraphLaunch(state.exec, cudaStreamPerThread) == cudaSuccess : state.Run(logits);
    if (!launched) return fail();
    if (cudaEventRecord(state.done, cudaStreamPerThread) != cudaSuccess) return fail();
    state.pending = true; plan.active = &state;
    return true;
}

bool FastllmCudaMtpVerifierEnqueue(void *opaque, float *logits,
        const FastllmMtpProposalView *proposals, const float *temperatures,
        const int *topKs, const float *topPs, int batch, int drafts, int vocab, bool useGraph) {
    return MtpVerifierEnqueue(opaque, logits, proposals, temperatures, topKs, topPs,
        batch, drafts, vocab, useGraph, false);
}

bool FastllmCudaMtpVerifierEnqueueProbabilities(void *opaque, float *logits,
        const FastllmMtpProposalView *proposals, const float *temperatures,
        const int *topKs, const float *topPs, int batch, int drafts, int vocab, bool useGraph) {
    return MtpVerifierEnqueue(opaque, logits, proposals, temperatures, topKs, topPs,
        batch, drafts, vocab, useGraph, true);
}

bool FastllmCudaMtpVerifierResolve(void *opaque, int *output, int *accepted) {
    if (!opaque || !output || !accepted) return false;
    auto &plan = *static_cast<VerifierPlan *>(opaque);
    if (!plan.active) return false;
    auto &state = *plan.active;
    int previous; cudaGetDevice(&previous);
    cudaSetDevice(state.shape.device);
    const cudaError_t completed = cudaEventSynchronize(state.done);
    cudaSetDevice(previous);
    if (completed != cudaSuccess) return false;
    state.pending = false; plan.active = nullptr;
    const int batch = state.shape.batch, drafts = state.shape.drafts;
    std::copy_n(state.ws.hostResults, size_t(batch) * (drafts + 1), output);
    std::copy_n(state.ws.hostResults + size_t(batch) * (drafts + 1), batch, accepted);
    for (int b = 0; b < batch; ++b) {
        if (accepted[b] < 0 || accepted[b] > drafts) return false;
        for (int i = 0; i <= accepted[b]; ++i)
            if (output[b * (drafts + 1) + i] < 0 || output[b * (drafts + 1) + i] >= state.shape.vocab) return false;
    }
    return true;
}
