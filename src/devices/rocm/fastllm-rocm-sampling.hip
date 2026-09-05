// Native HIP sampling. Keep probabilities and sampled tokens on the GPU so
// the device-output API also works during stream capture and graph replay.
#include "fastllm-hip.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {
constexpr int Threads = 256;
__device__ unsigned long long samplingCounter = 0;

__device__ unsigned long long NextRandom(unsigned long long &state) {
    unsigned long long x = (state += 0x9e3779b97f4a7c15ULL);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__global__ void TemperatureSoftmax(float *logits, float *probs,
        const float *temperatures, const int *penaltyIds,
        const float *penaltyFactors, int penaltyTokens, int vocab) {
    const int row = blockIdx.x, tid = threadIdx.x;
    float *input = logits + (size_t)row * vocab;
    float *output = probs + (size_t)row * vocab;
    // A single writer also handles duplicate ids without racing.
    if (tid == 0) {
        for (int j = 0; j < penaltyTokens; ++j) {
            int id = penaltyIds[(size_t)row * penaltyTokens + j];
            if (id < 0 || id >= vocab) continue;
            float f = penaltyFactors[(size_t)row * penaltyTokens + j];
            input[id] = f > 0 && isfinite(f) ?
                (input[id] < 0 ? input[id] * f : input[id] / f) : NAN;
        }
    }
    __syncthreads();
    const float temp = temperatures[row];
    const float invTemp = temp > 0 ? 1.0f / temp : 1.0f;
    __shared__ float values[Threads];
    __shared__ int invalid[Threads];
    float maximum = -INFINITY;
    int bad = !isfinite(temp);
    for (int i = tid; i < vocab; i += Threads) {
        const float v = input[i] * invTemp;
        maximum = fmaxf(maximum, v);
        bad |= isnan(v);
    }
    values[tid] = maximum;
    invalid[tid] = bad;
    __syncthreads();
    for (int step = Threads / 2; step; step >>= 1) {
        if (tid < step) {
            values[tid] = fmaxf(values[tid], values[tid + step]);
            invalid[tid] |= invalid[tid + step];
        }
        __syncthreads();
    }
    maximum = values[0];
    bad = invalid[0] || maximum == -INFINITY;
    __syncthreads();
    float sum = 0;
    for (int i = tid; i < vocab; i += Threads) {
        float v = input[i] * invTemp;
        float p = bad ? NAN : (maximum == INFINITY ? (v == INFINITY ? 1.0f : 0.0f) : expf(v - maximum));
        output[i] = p;
        sum += p;
    }
    values[tid] = sum;
    __syncthreads();
    for (int step = Threads / 2; step; step >>= 1) {
        if (tid < step) values[tid] += values[tid + step];
        __syncthreads();
    }
    sum = values[0];
    for (int i = tid; i < vocab; i += Threads) output[i] /= sum;
}

__global__ void SampleProbabilities(const float *probs, const float *temperatures,
        const int *topKs, const float *topPs, int *output,
        float *floatOutput, int vocab, unsigned long long seed) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const float *p = probs + (size_t)row * vocab;
    if (!isfinite(topPs[row])) {
        if (tid == 0) {
            output[row] = -1;
            if (floatOutput) floatOutput[row] = -1.0f;
        }
        return;
    }
    const int k = topKs[row] <= 0 ? vocab : min(topKs[row], vocab);
    const float topP = fminf(fmaxf(topPs[row], 0.0f), 1.0f);
    __shared__ float masses[Threads];
    __shared__ int ids[Threads];
    __shared__ int picked, owner, done;
    __shared__ float target, pivot, lower;
    __shared__ unsigned long long randomState;
    if (tid == 0) {
        // Advance on device, not at graph construction time: replay must draw
        // fresh samples. Atomic allocation also separates concurrent streams.
        randomState = seed ^ (atomicAdd(&samplingCounter, 1ULL) + 1) * 0xd1342543de82ef95ULL;
        lower = -1.0f;
        picked = -1;
    }
    __syncthreads();
    if (k == 1 || temperatures[row] <= 0 || topP <= 0) {
        float best = -1;
        int id = -1;
        for (int i = tid; i < vocab; i += Threads) {
            if (p[i] > best) { best = p[i]; id = i; }
        }
        masses[tid] = best;
        ids[tid] = id;
        __syncthreads();
        for (int step = Threads / 2; step; step >>= 1) {
            if (tid < step && (masses[tid + step] > masses[tid] ||
                    (masses[tid + step] == masses[tid] && ids[tid + step] < ids[tid]))) {
                masses[tid] = masses[tid + step];
                ids[tid] = ids[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0) picked = masses[0] > 0 ? ids[0] : -1;
    } else {
        // Match FlashInfer's joint filter on the original softmax: a token
        // is eligible iff fewer than k tokens have strictly greater mass,
        // and their summed mass is below top-p. Equal probabilities are kept
        // together. Rejecting a pivot removes only ineligible probabilities.
        for (int iteration = 0; iteration < vocab; ++iteration) {
            float mass = 0;
            for (int i = tid; i < vocab; i += Threads) if (p[i] > lower) mass += p[i];
            masses[tid] = mass;
            __syncthreads();
            if (tid == 0) {
                float total = 0;
                for (int t = 0; t < Threads; ++t) total += masses[t];
                done = !(total > 0 && isfinite(total)) || !isfinite(topPs[row]);
                owner = -1;
                if (!done) {
                    double u = ((NextRandom(randomState) >> 32) + 1.0) / 4294967297.0;
                    double remaining = u * total;
                    for (int t = 0; t < Threads; ++t) {
                        if (masses[t] > 0) {
                            owner = t;
                            target = (float)remaining;
                        }
                        if (remaining < masses[t]) break;
                        remaining -= masses[t];
                    }
                }
            }
            __syncthreads();
            if (done) break;
            if (tid == owner) {
                float cumulative = 0;
                for (int i = tid; i < vocab; i += Threads) {
                    if (p[i] <= lower || !(p[i] > 0)) continue;
                    cumulative += p[i];
                    picked = i;
                    if (cumulative > target) break;
                }
                pivot = p[picked];
            }
            __syncthreads();
            float greaterMass = 0;
            int greaterCount = 0;
            for (int i = tid; i < vocab; i += Threads) if (p[i] > pivot) {
                greaterMass += p[i];
                ++greaterCount;
            }
            masses[tid] = greaterMass;
            ids[tid] = greaterCount;
            __syncthreads();
            for (int step = Threads / 2; step; step >>= 1) {
                if (tid < step) { masses[tid] += masses[tid + step]; ids[tid] += ids[tid + step]; }
                __syncthreads();
            }
            if (tid == 0) {
                done = ids[0] < k && (topP >= 1 || masses[0] < topP);
                if (!done) { lower = pivot; picked = -1; }
            }
            __syncthreads();
            if (done) break;
        }
    }
    __syncthreads();
    if (tid == 0) {
        output[row] = picked;
        if (floatOutput) floatOutput[row] = (float)picked;
    }
}

unsigned long long HostSeed() {
    static thread_local std::mt19937_64 generator(std::random_device{}());
    return generator();
}

bool Launch(float *logits, float *probs, float *temperatures, int *topKs,
        float *topPs, int *penaltyIds, float *penaltyFactors, int penaltyTokens,
        int *output, float *floatOutput, int batch, int vocab) {
    TemperatureSoftmax<<<batch, Threads, 0, hipStreamPerThread>>>(logits, probs,
        temperatures, penaltyIds, penaltyFactors, penaltyTokens, vocab);
    if (hipGetLastError() != hipSuccess) return false;
    SampleProbabilities<<<batch, Threads, 0, hipStreamPerThread>>>(probs,
        temperatures, topKs, topPs, output, floatOutput, vocab, HostSeed());
    return hipGetLastError() == hipSuccess;
}
} // namespace

bool FastllmCudaTopKTopPSamplingToDevice(float *logits, float *probs,
        float *temperatures, int *topKArr, float *topPArr,
        int *penaltyIds, float *penaltyFactors, int penaltyTokens,
        int *output, float *floatOutput, int batch, int vocabSize) {
    if (!logits || !probs || !temperatures || !topKArr || !topPArr ||
        !output || !floatOutput || batch <= 0 || vocabSize <= 0 ||
        penaltyTokens < 0 || (penaltyTokens && (!penaltyIds || !penaltyFactors))) return false;
    return Launch(logits, probs, temperatures, topKArr, topPArr, penaltyIds,
        penaltyFactors, penaltyTokens, output, floatOutput, batch, vocabSize);
}

bool FastllmCudaTopKTopPSamplingWithTypicalAcceptance(float *logits,
        float *temperatures, int *topKArr, float *topPArr, int *output,
        int batch, int vocabSize, const int *typicalCandidateIds,
        const int *typicalCandidateRows, unsigned char *typicalAccepted,
        int *typicalRecoveredIds, int typicalCount,
        float typicalPosteriorThreshold, float typicalPosteriorAlpha) {
    if (!logits || !temperatures || !topKArr || !topPArr || !output || batch <= 0 || vocabSize <= 0) return false;
    for (int b = 0; b < batch; ++b) if (!std::isfinite(temperatures[b]) || !std::isfinite(topPArr[b])) return false;
    const int count = typicalCandidateIds && typicalAccepted && typicalRecoveredIds ?
        std::max(0, std::min(typicalCount, batch)) : 0;
    for (int i = 0; i < count; ++i) if (typicalCandidateRows && (typicalCandidateRows[i] < 0 || typicalCandidateRows[i] >= batch)) return false;
    const size_t probBytes = (size_t)batch * vocabSize * sizeof(float);
    const size_t need = probBytes + (size_t)batch * 4 * sizeof(float);
    size_t available = 0;
    bool own = false;
    void *scratch = FastllmBorrowDequantScratch(need, &available, &own);
    struct Release { void *p; bool own; ~Release() { FastllmReleaseDequantScratch(p, own); } } release{scratch, own};
    if (!scratch || available < need) return false;
    float *probs = (float*)scratch;
    float *temps = (float*)((char*)scratch + probBytes);
    int *ks = (int*)(temps + batch);
    float *ps = (float*)(ks + batch);
    int *out = (int*)(ps + batch);
    const size_t bytes = (size_t)batch * sizeof(float);
    if (hipMemcpyAsync(temps, temperatures, bytes, hipMemcpyHostToDevice, hipStreamPerThread) != hipSuccess ||
        hipMemcpyAsync(ks, topKArr, bytes, hipMemcpyHostToDevice, hipStreamPerThread) != hipSuccess ||
        hipMemcpyAsync(ps, topPArr, bytes, hipMemcpyHostToDevice, hipStreamPerThread) != hipSuccess ||
        !Launch(logits, probs, temps, ks, ps, nullptr, nullptr, 0, out, nullptr, batch, vocabSize) ||
        hipMemcpyAsync(output, out, bytes, hipMemcpyDeviceToHost, hipStreamPerThread) != hipSuccess ||
        hipStreamSynchronize(hipStreamPerThread) != hipSuccess) return false;
    for (int b = 0; b < batch; ++b) if (output[b] < 0 || output[b] >= vocabSize) return false;
    // The typical-acceptance API already returns host results. Only speculative
    // callers need this copy; ordinary decoding stays entirely on the GPU.
    std::vector<float> row(count ? vocabSize : 0);
    for (int c = 0; c < count; ++c) {
        int b = typicalCandidateRows ? typicalCandidateRows[c] : c;
        if (hipMemcpyAsync(row.data(), probs + (size_t)b * vocabSize, (size_t)vocabSize * sizeof(float), hipMemcpyDeviceToHost, hipStreamPerThread) != hipSuccess ||
            hipStreamSynchronize(hipStreamPerThread) != hipSuccess) return false;
        int id = typicalCandidateIds[c];
        float candidate = id >= 0 && id < vocabSize ? row[id] : 0;
        double entropy = 0, greaterMass = 0;
        int greaterCount = 0, best = 0;
        for (int j = 0; j < vocabSize; ++j) {
            float p = row[j];
            entropy -= p * std::log(p + 1.0e-5f);
            if (p > candidate) { greaterMass += p; ++greaterCount; }
            if (p > row[best]) best = j;
        }
        float threshold = std::min(typicalPosteriorThreshold, (float)std::exp(-entropy) * typicalPosteriorAlpha);
        typicalAccepted[c] = (topKArr[b] <= 0 || greaterCount < topKArr[b]) &&
            (topPArr[b] >= 1 || greaterMass < topPArr[b]) && candidate > threshold;
        typicalRecoveredIds[c] = best;
    }
    return true;
}
