#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>

#ifdef FASTLLM_ENABLE_DSV4_SPARSE_MLA_SM120
extern "C" bool FastllmCudaDeepSeekV4SparseMlaSm120Raw(
    const void *q, const uint8_t *kvCache, const int32_t *indices,
    void *midOut, float *midLse, const int *topkLength, void *output,
    const float *attnSink, const uint8_t *extraKvCache,
    const int32_t *extraIndices, const int *extraTopkLength, int numTokens,
    int numHeads, int mainTopk, int extraTopk, int numSplits, float smScale,
    size_t strideKvBlock, size_t strideExtraKvBlock, void *stream);
extern "C" bool FastllmCudaDeepSeekV4IndexerLogitsSm120Raw(
    const uint8_t *q, const uint8_t *kv, const float *kvScales,
    const float *weights, uint32_t *cuSeqLenKStart,
    uint32_t *cuSeqLenKEnd, float *logits, int seqLen,
    int kvCapacity, int logitsStride, void *stream);
#endif

namespace {

__device__ __forceinline__ bool DeepSeekV4DsparkGreedyIsBetter(
        float value, int id, float bestValue, int bestId) {
    return value > bestValue ||
           (value == bestValue && id < bestId);
}

__device__ __forceinline__ void DeepSeekV4DsparkStoreRelease(
        uint32_t *pointer, uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::
                 "r"(value), "l"(pointer));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::
                 "r"(value), "l"(pointer));
#endif
}

__device__ __forceinline__ uint32_t DeepSeekV4DsparkLoadAcquire(
        const uint32_t *pointer) {
    uint32_t value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" :
                 "=r"(value) : "l"(pointer));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" :
                 "=r"(value) : "l"(pointer));
#endif
    return value;
}

__global__ void DeepSeekV4DsparkMarkovSignalKernel(
        uint32_t *signal, int step) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t next = DeepSeekV4DsparkLoadAcquire(signal + step) + 1;
        DeepSeekV4DsparkStoreRelease(signal + step, next);
    }
}

__global__ void DeepSeekV4DsparkMarkovWaitPeerKernel(
        const uint32_t *peerSignal, uint32_t *localSeen, int step) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t previous = localSeen[step];
        uint32_t current;
        do {
            current = DeepSeekV4DsparkLoadAcquire(peerSignal + step);
        } while (current == previous);
        localSeen[step] = current;
    }
}

template <int THREADS>
__global__ void DeepSeekV4DsparkMarkovCopyPeerKernel(
        const uint32_t *peerSignal, uint32_t *localSeen, int step,
        const float *peerLatent, float *localLatent, int hiddenSize) {
    __shared__ uint32_t readyEpoch;
    if (threadIdx.x == 0) {
        const uint32_t previous = localSeen[step];
        uint32_t current;
        do {
            current = DeepSeekV4DsparkLoadAcquire(peerSignal + step);
        } while (current == previous);
        readyEpoch = current;
    }
    __syncthreads();

    // Acquire on every copying lane.  The root publishes the epoch only after
    // its embedding kernel has written the 1 KiB latent, so subsequent peer
    // loads observe that complete row.  Copying it once per rank avoids having
    // every vocab-row CTA issue the same peer reads again.
    (void)DeepSeekV4DsparkLoadAcquire(peerSignal + step);
    for (int index = threadIdx.x; index < hiddenSize; index += THREADS) {
        localLatent[index] = peerLatent[index];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        localSeen[step] = readyEpoch;
    }
}

template <int THREADS>
__global__ void DeepSeekV4DsparkPrepareTargetPeerKernel(
        const uint32_t *peerSignal, uint32_t *localSeen,
        const int *peerProposalIds, int proposalCount,
        int anchorToken, int startPos, int32_t *decodeMeta,
        float *inputIds) {
    __shared__ uint32_t readyEpoch;
    if (threadIdx.x == 0) {
        const uint32_t previous = localSeen[0];
        uint32_t current;
        do {
            current = DeepSeekV4DsparkLoadAcquire(peerSignal);
        } while (current == previous);
        readyEpoch = current;
        decodeMeta[0] = startPos;
        decodeMeta[1] = anchorToken;
        if (inputIds != nullptr) {
            inputIds[0] = (float)anchorToken;
        }
    }
    __syncthreads();

    // Acquire on each copying lane before reading the root rank's proposal.
    // The root publishes its epoch after the final Markov select kernel, so
    // the peer loads below observe all seven IDs from that draft round.
    (void)DeepSeekV4DsparkLoadAcquire(peerSignal);
    for (int index = threadIdx.x; index < proposalCount;
         index += THREADS) {
        const int token = peerProposalIds[index];
        decodeMeta[index + 2] = token;
        if (inputIds != nullptr) {
            inputIds[index + 1] = (float)token;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        localSeen[0] = readyEpoch;
    }
}

template <int THREADS>
__global__ void DeepSeekV4DsparkAcceptPeerKernel(
        const int *candidateIds, const float *candidateScores,
        const int *globalOffsets, const int *proposalIds,
        int ranks, int rows, int *result, uint32_t *readySignal) {
    __shared__ int targetIds[THREADS];
    const int row = threadIdx.x;
    if (row < rows) {
        float bestScore = -FLT_MAX;
        int bestId = 0;
        for (int rank = 0; rank < ranks; ++rank) {
            const int globalId = globalOffsets[rank] +
                candidateIds[rank * rows + row];
            const float score = candidateScores[rank * rows + row];
            if (DeepSeekV4DsparkGreedyIsBetter(
                    score, globalId, bestScore, bestId)) {
                bestScore = score;
                bestId = globalId;
            }
        }
        targetIds[row] = bestId;
        result[3 + row] = bestId;
    }
    __syncthreads();
    if (row == 0) {
        const int proposalCount = rows - 1;
        int accepted = 0;
        while (accepted < proposalCount &&
               proposalIds[accepted] == targetIds[accepted]) {
            ++accepted;
        }
        result[0] = accepted;
        result[1] = accepted + 1;
        result[2] = targetIds[accepted];
        for (int index = 0; index < proposalCount; ++index) {
            result[3 + rows + index] = proposalIds[index];
        }
        const uint32_t next =
            DeepSeekV4DsparkLoadAcquire(readySignal) + 1;
        DeepSeekV4DsparkStoreRelease(readySignal, next);
    }
}

template <int THREADS>
__global__ void DeepSeekV4DsparkPrepareDraftPeerKernel(
        const uint32_t *peerSignal, uint32_t *localSeen,
        const int *peerResult, int baseCommittedTokens,
        const __nv_bfloat16 *stageKv0, __nv_bfloat16 *windowKv0,
        const __nv_bfloat16 *stageKv1, __nv_bfloat16 *windowKv1,
        const __nv_bfloat16 *stageKv2, __nv_bfloat16 *windowKv2,
        int rows, int windowSize, int headDim,
        int noiseTokenId, int proposalCount,
        int32_t *decodeMeta, float *inputIds) {
    const int operation = blockIdx.x;
    __shared__ uint32_t readyEpoch;
    if (threadIdx.x == 0) {
        const uint32_t previous = localSeen[operation];
        uint32_t current;
        do {
            current = DeepSeekV4DsparkLoadAcquire(peerSignal);
        } while (current == previous);
        readyEpoch = current;
    }
    __syncthreads();
    (void)DeepSeekV4DsparkLoadAcquire(peerSignal);

    const int appendTokens = peerResult[1];
    const int nextToken = peerResult[2];
    const bool valid = appendTokens > 0 && appendTokens <= rows &&
        appendTokens <= windowSize;
    if (valid && operation < 3) {
        const __nv_bfloat16 *source = operation == 0 ? stageKv0 :
            (operation == 1 ? stageKv1 : stageKv2);
        __nv_bfloat16 *window = operation == 0 ? windowKv0 :
            (operation == 1 ? windowKv1 : windowKv2);
        const int retained = windowSize - appendTokens;
        for (int d = threadIdx.x; d < headDim; d += THREADS) {
            for (int token = 0; token < retained; ++token) {
                window[(uint64_t)token * headDim + d] =
                    window[(uint64_t)(token + appendTokens) * headDim + d];
            }
            for (int token = 0; token < appendTokens; ++token) {
                window[(uint64_t)(retained + token) * headDim + d] =
                    source[(uint64_t)token * headDim + d];
            }
        }
    } else if (valid) {
        if (threadIdx.x == 0) {
            decodeMeta[0] = baseCommittedTokens + appendTokens;
        }
        for (int token = threadIdx.x; token < proposalCount;
             token += THREADS) {
            const int id = token == 0 ? nextToken : noiseTokenId;
            decodeMeta[token + 1] = id;
            if (inputIds != nullptr) {
                inputIds[token] = (float)id;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        localSeen[operation] = readyEpoch;
    }
}

template <int THREADS>
__global__ void DeepSeekV4DsparkMarkovLocalArgmaxKernel(
        const float *baseLogits, const float *markovBias,
        int *packedCandidate, int vocabSize) {
    const int tid = threadIdx.x;
    float localMax = -FLT_MAX;
    int localId = 0;
    for (int index = tid; index < vocabSize; index += THREADS) {
        const float value = baseLogits[index] + markovBias[index];
        // Indices increase monotonically in each lane, matching FastLLM's
        // ordinary greedy sampler's lowest-token tie break.
        if (value > localMax) {
            localMax = value;
            localId = index;
        }
    }

    __shared__ float maxValues[THREADS];
    __shared__ int maxIds[THREADS];
    maxValues[tid] = localMax;
    maxIds[tid] = localId;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride && DeepSeekV4DsparkGreedyIsBetter(
                maxValues[tid + stride], maxIds[tid + stride],
                maxValues[tid], maxIds[tid])) {
            maxValues[tid] = maxValues[tid + stride];
            maxIds[tid] = maxIds[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        packedCandidate[0] = maxIds[0];
        packedCandidate[1] = __float_as_int(maxValues[0]);
    }
}

__global__ void DeepSeekV4DsparkMarkovSelectKernel(
        const int *packedCandidates, const int *globalOffsets,
        int ranks, int *proposalIds, float *previousId, int step) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float bestScore = -FLT_MAX;
    int bestId = 0;
    for (int rank = 0; rank < ranks; rank++) {
        const int globalId = globalOffsets[rank] +
            packedCandidates[rank * 2];
        const float score = __int_as_float(
            packedCandidates[rank * 2 + 1]);
        if (DeepSeekV4DsparkGreedyIsBetter(
                score, globalId, bestScore, bestId)) {
            bestScore = score;
            bestId = globalId;
        }
    }
    proposalIds[step] = bestId;
    previousId[0] = (float)bestId;
}

// This intentionally mirrors FastllmGemvFp32Fp16Kernel2MultiRow<PART=1>
// instruction-for-instruction.  Its input is the per-rank replica populated by
// DeepSeekV4DsparkMarkovCopyPeerKernel.  Keeping the hot GEMV's 256 FP32 values
// local avoids repeating the same peer reads in every vocabulary-row CTA.
template <int THREADS>
__global__ void DeepSeekV4DsparkMarkovLinearPeerKernel(
        const float *peerLatent, const half *weight, float *output,
        int hiddenSize, int localVocabSize) {
    __shared__ float values[THREADS];
    const unsigned int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= localVocabSize) {
        return;
    }
    values[tid] = 0.0f;
    const half *rowWeight = weight + (size_t)row * hiddenSize;
    if (hiddenSize % 4 == 0) {
        for (int index = tid * 4; index + 3 < hiddenSize;
             index += THREADS * 4) {
            const float4 latent = *reinterpret_cast<const float4 *>(
                peerLatent + index);
            const uint2 packed = *reinterpret_cast<const uint2 *>(
                rowWeight + index);
            const half2 *weightPairs =
                reinterpret_cast<const half2 *>(&packed);
            values[tid] += latent.x * __low2float(weightPairs[0]);
            values[tid] += latent.y * __high2float(weightPairs[0]);
            values[tid] += latent.z * __low2float(weightPairs[1]);
            values[tid] += latent.w * __high2float(weightPairs[1]);
        }
    } else {
        for (int index = tid; index < hiddenSize; index += THREADS) {
            values[tid] += peerLatent[index] * (float)rowWeight[index];
        }
    }
    __syncthreads();
    float diff = 0.0f;
    for (unsigned int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = values[tid + stride] - diff;
            const float sum = values[tid] + other;
            diff = (sum - values[tid]) - other;
            values[tid] = sum;
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[row] = values[0];
    }
}

__global__ void DeepSeekV4DsparkMarkovSelectPeerKernel(
        const uint64_t *peerCandidatePointers,
        const uint64_t *peerSignalPointers, uint32_t *localSeen,
        const int *globalOffsets, int ranks, int steps,
        int *proposalIds, float *previousId, int step) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    float bestScore = -FLT_MAX;
    int bestId = 0;
    for (int rank = 0; rank < ranks; rank++) {
        const uint32_t *peerSignal =
            reinterpret_cast<const uint32_t *>(
                (uintptr_t)peerSignalPointers[rank]);
        const int seenIndex = rank * steps + step;
        const uint32_t previous = localSeen[seenIndex];
        uint32_t current;
        do {
            current = DeepSeekV4DsparkLoadAcquire(peerSignal + step);
        } while (current == previous);
        localSeen[seenIndex] = current;
        const int *candidate = reinterpret_cast<const int *>(
            (uintptr_t)peerCandidatePointers[rank]);
        const int globalId = globalOffsets[rank] + candidate[0];
        const float score = __int_as_float(candidate[1]);
        if (DeepSeekV4DsparkGreedyIsBetter(
                score, globalId, bestScore, bestId)) {
            bestScore = score;
            bestId = globalId;
        }
    }
    proposalIds[step] = bestId;
    previousId[0] = (float)bestId;
}

constexpr int kDeepSeekV4Sparse1MCompressedKeys = 256 * 1024;
constexpr int kDeepSeekV4SparseMaxKeys = kDeepSeekV4Sparse1MCompressedKeys + 64 * 1024;
constexpr int kDeepSeekV4SparseDecodeMaxKeys = kDeepSeekV4SparseMaxKeys;
constexpr int kDeepSeekV4SparsePrefillMaxKeys = kDeepSeekV4SparseMaxKeys;
constexpr size_t kDeepSeekV4SparsePrefillDefaultTempBytes = 256ULL * 1024ULL * 1024ULL;
constexpr int kDeepSeekV4SparseMlaPageSize = 64;
constexpr int kDeepSeekV4SparseMlaNopeDim = 448;
constexpr int kDeepSeekV4SparseMlaRopeDim = 64;
constexpr int kDeepSeekV4SparseMlaHeadDim = 512;
constexpr int kDeepSeekV4SparseMlaTokenDataBytes = 576;
constexpr int kDeepSeekV4SparseMlaScaleBytes = 8;
constexpr int kDeepSeekV4SparseMlaBytesPerToken = 584;
constexpr int kDeepSeekV4SparseMlaPageBytes =
    kDeepSeekV4SparseMlaPageSize * kDeepSeekV4SparseMlaBytesPerToken;

struct DeepSeekV4RouteTableCacheEntry {
    void *cudaData = nullptr;
    const void *sourceData = nullptr;
    uint64_t count = 0;
    fastllm::DataType dataType = fastllm::DataType::FLOAT32;
    bool verifyContent = false;
    uint64_t fingerprint = 0;
    // A changed non-model table can still have an in-flight kernel referring
    // to the old address. Keep superseded replicas alive until the owning Data
    // is retired instead of making them immediately reusable.
    std::vector<void*> retiredCudaData;
};

using DeepSeekV4RouteTableDeviceCache =
    std::map<int, DeepSeekV4RouteTableCacheEntry>;
using DeepSeekV4RouteTableCache =
    std::map<const fastllm::Data*, DeepSeekV4RouteTableDeviceCache,
             std::less<const fastllm::Data*> >;

std::mutex &DeepSeekV4RouteTableCacheMutex() {
    // Data objects can be destroyed by language bindings during shared-library
    // teardown. Keep the registry primitives alive until process exit so a
    // late Data destructor never touches an already-destroyed static mutex.
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

DeepSeekV4RouteTableCache &DeepSeekV4RouteTableCaches() {
    // Entries are explicitly retired by Data::~Data. Intentionally retain the
    // empty registry object itself to avoid cross-translation-unit static
    // destruction ordering hazards in Python extension shutdown.
    static DeepSeekV4RouteTableCache *cache = new DeepSeekV4RouteTableCache();
    return *cache;
}

uint64_t DeepSeekV4RouteTableFingerprint(const void *data, size_t bytes) {
    const uint8_t *p = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

const void *DeepSeekV4GetCudaRouteTable(fastllm::Data &routeTable) {
    if (routeTable.dataDevice == fastllm::DataDevice::CUDA &&
        routeTable.cudaData != nullptr) {
        return routeTable.cudaData;
    }
    if (routeTable.dataDevice != fastllm::DataDevice::CPU ||
        routeTable.cpuData == nullptr) {
        return nullptr;
    }

    uint64_t count = routeTable.Count(0);
    if (count == 0 || count > SIZE_MAX / sizeof(int32_t)) {
        return nullptr;
    }
    size_t bytes = (size_t)count * sizeof(int32_t);
    // Loaded model weights are immutable after construction, so hashing a
    // multi-megabyte route table on every decode token would be pure overhead.
    // Non-model tensors remain mutable and are fingerprinted on every lookup.
    bool verifyContent = !routeTable.isModelWeight;
    uint64_t fingerprint = verifyContent ?
        DeepSeekV4RouteTableFingerprint(routeTable.cpuData, bytes) : 0;
    int device = FastllmCudaGetDevice();

    std::lock_guard<std::mutex> guard(DeepSeekV4RouteTableCacheMutex());
    DeepSeekV4RouteTableCacheEntry &entry =
        DeepSeekV4RouteTableCaches()[&routeTable][device];
    bool matches = entry.cudaData != nullptr &&
                   entry.sourceData == routeTable.cpuData &&
                   entry.count == count &&
                   entry.dataType == routeTable.dataType &&
                   entry.verifyContent == verifyContent &&
                   (!verifyContent || entry.fingerprint == fingerprint);
    if (matches) {
        return entry.cudaData;
    }

    void *next = FastllmCudaMalloc(bytes);
    if (next == nullptr) {
        return nullptr;
    }
    FastllmCudaCopyFromHostToDevice(next, routeTable.cpuData, bytes);
    if (entry.cudaData != nullptr) {
        entry.retiredCudaData.push_back(entry.cudaData);
    }
    entry.cudaData = next;
    entry.sourceData = routeTable.cpuData;
    entry.count = count;
    entry.dataType = routeTable.dataType;
    entry.verifyContent = verifyContent;
    entry.fingerprint = fingerprint;
    return entry.cudaData;
}

size_t DeepSeekV4SparsePrefillTempBytesLimit() {
    size_t ret = kDeepSeekV4SparsePrefillDefaultTempBytes;
    if (const char *env = std::getenv("FASTLLM_DSV4_SPARSE_PREFILL_TEMP_MB")) {
        int mb = std::atoi(env);
        if (mb > 0) {
            ret = (size_t)mb * 1024ULL * 1024ULL;
        }
    }
    return ret;
}

bool DeepSeekV4Sm120RuntimeEnabled() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
        cudaGetLastError();
        return false;
    }
    static thread_local int cachedDevice = -1;
    static thread_local bool cachedSupported = false;
    if (cachedDevice != device) {
        int major = 0, minor = 0;
        cudaError_t majorState = cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, device);
        cudaError_t minorState = cudaDeviceGetAttribute(
            &minor, cudaDevAttrComputeCapabilityMinor, device);
        cachedSupported = majorState == cudaSuccess && minorState == cudaSuccess &&
                          major == 12 && minor == 0;
        if (majorState != cudaSuccess || minorState != cudaSuccess) {
            cudaGetLastError();
        }
        cachedDevice = device;
    }
    return cachedSupported;
}

bool DeepSeekV4DsparkMarkovPeerRuntimeEnabled() {
    return DeepSeekV4Sm120RuntimeEnabled();
}

bool DeepSeekV4SparseMlaSm120RuntimeEnabled() {
#ifdef FASTLLM_ENABLE_DSV4_SPARSE_MLA_SM120
    return DeepSeekV4Sm120RuntimeEnabled();
#else
    return false;
#endif
}

inline int DeepSeekV4HcPreDotParts(int flatDim, int threads) {
    return std::min(16, std::max(1, (flatDim + threads * 4 - 1) / (threads * 4)));
}

inline int DeepSeekV4HcPreFinishParts(int dim, int threads) {
    return std::min(16, std::max(1, (dim + threads - 1) / threads));
}

template <typename Kernel>
bool DeepSeekV4EnsureDynamicSharedMemory(Kernel kernel, size_t sharedBytes) {
    constexpr size_t kDefaultDynamicSharedLimit = 48 * 1024;
    if (sharedBytes <= kDefaultDynamicSharedLimit) {
        return true;
    }
    int device = 0;
    cudaError_t state = cudaGetDevice(&device);
    if (state != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    int maxOptinShared = 0;
    state = cudaDeviceGetAttribute(&maxOptinShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (state != cudaSuccess || sharedBytes > (size_t)maxOptinShared) {
        cudaGetLastError();
        return false;
    }
    state = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);
    if (state != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

__device__ __forceinline__ float Dsv4ToFloat(float v) {
    return v;
}

__device__ __forceinline__ float Dsv4ToFloat(half v) {
    return __half2float(v);
}

__device__ __forceinline__ float Dsv4ToFloat(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename XT, typename WT>
__device__ __forceinline__ float Dsv4PairDot(const XT *x, const WT *w, int k) {
    return Dsv4ToFloat(x[k]) * Dsv4ToFloat(w[k]) +
           Dsv4ToFloat(x[k + 1]) * Dsv4ToFloat(w[k + 1]);
}

__device__ __forceinline__ float Dsv4PairDot(const __nv_bfloat16 *x, const __nv_bfloat16 *w, int k) {
    __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162 *>(x + k);
    __nv_bfloat162 wv = *reinterpret_cast<const __nv_bfloat162 *>(w + k);
    float2 xf = __bfloat1622float2(xv);
    float2 wf = __bfloat1622float2(wv);
    return xf.x * wf.x + xf.y * wf.y;
}

__device__ __forceinline__ float Dsv4PairDot(const half *x, const half *w, int k) {
    half2 xv = *reinterpret_cast<const half2 *>(x + k);
    half2 wv = *reinterpret_cast<const half2 *>(w + k);
    float2 xf = __half22float2(xv);
    float2 wf = __half22float2(wv);
    return xf.x * wf.x + xf.y * wf.y;
}

__device__ __forceinline__ float Dsv4PairDot(const float *x, const float *w, int k) {
    float2 xv = *reinterpret_cast<const float2 *>(x + k);
    float2 wv = *reinterpret_cast<const float2 *>(w + k);
    return xv.x * wv.x + xv.y * wv.y;
}

template <typename T>
__device__ __forceinline__ T Dsv4FromFloat(float v);

__device__ __forceinline__ float Dsv4Fp8E4M3RoundTrip(float value,
                                                      float scale) {
    float scaled = fminf(448.0f, fmaxf(-448.0f, value / scale));
    __nv_fp8_e4m3 quantized(scaled);
    return static_cast<float>(quantized) * scale;
}

template <>
__device__ __forceinline__ float Dsv4FromFloat<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ half Dsv4FromFloat<half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 Dsv4FromFloat<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

__global__ void DeepSeekV4HcMeanBf16Kernel(
        const __nv_bfloat16 *__restrict__ input,
        __nv_bfloat16 *__restrict__ output, int tokens, int hc, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * dim;
    if (index >= total) {
        return;
    }
    int token = index / dim;
    int channel = index - token * dim;
    const __nv_bfloat16 *row = input + (uint64_t)token * hc * dim + channel;
    float invHc = 1.0f / (float)hc;
    float value = Dsv4ToFloat(row[0]) * invHc;
    for (int branch = 1; branch < hc; ++branch) {
        value = fmaf(Dsv4ToFloat(row[(uint64_t)branch * dim]), invHc, value);
    }
    output[index] = Dsv4FromFloat<__nv_bfloat16>(value);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                    int bsz, int seqlen, int heads, int headDim,
                                    int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    double v = 0.0;
    for (int d = 0; d < groupDim; d++) {
        v += (double)Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
    }
    output[idx] = __float2bfloat16_rn((float)v);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoABlockReduceKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                               int bsz, int seqlen, int heads, int headDim,
                                               int groups, int oRank) {
    extern __shared__ float partial[];
    int idx = blockIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < groupDim; d += blockDim.x) {
        sum += Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[idx] = __float2bfloat16_rn(partial[0]);
    }
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAPairBlockReduceKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                                   int bsz, int seqlen, int heads, int headDim,
                                                   int groups, int oRank) {
    extern __shared__ float partial[];
    float *partial0 = partial;
    float *partial1 = partial + blockDim.x;
    int pairsPerGroup = oRank / 2;
    int idx = blockIdx.x;
    int total = bsz * seqlen * groups * pairsPerGroup;
    if (idx >= total) {
        return;
    }

    int pair = idx % pairsPerGroup;
    int r0 = pair * 2;
    int r1 = r0 + 1;
    int tmp = idx / pairsPerGroup;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow0 = w + ((uint64_t)g * oRank + r0) * groupDim;
    const WT *wrow1 = wrow0 + groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int d = threadIdx.x; d < groupDim; d += blockDim.x) {
        float x = Dsv4ToFloat(src[d]);
        sum0 += x * Dsv4ToFloat(wrow0[d]);
        sum1 += x * Dsv4ToFloat(wrow1[d]);
    }
    partial0[threadIdx.x] = sum0;
    partial1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint64_t outBase = (((uint64_t)b * seqlen + s) * groups + g) * oRank;
        output[outBase + r0] = __float2bfloat16_rn(partial0[0]);
        output[outBase + r1] = __float2bfloat16_rn(partial1[0]);
    }
}

// The checkpoint stores wo_a as block-scaled FP8 E4M3, but the legacy path
// expands it to FP16 while loading.  Decode is bandwidth-bound on this 64 MiB
// matrix.  Read the original 8-bit payload and reproduce the legacy FP16
// dequantization in registers so the dot-product order and rounded weight
// values stay unchanged while global weight traffic is halved.
__device__ __forceinline__ float DeepSeekV4Fp8E4M3ScaledHalfToFloat(uint8_t value,
                                                                    float scale) {
    uint16_t pseudoHalfBits = ((value & 0x80u) << 8) | ((value & 0x7fu) << 7);
    half pseudoHalf = __ushort_as_half(pseudoHalfBits);
    half rounded = __float2half_rn(__half2float(pseudoHalf) * scale * 0x1p8f);
    return __half2float(rounded);
}

// vLLM quantizes the inverse-RoPE attention output to block-scaled E4M3
// before wo_a.  A power-of-two scale makes the dequantized value exactly
// representable in BF16, so materializing that value here preserves the FP8
// operand semantics while allowing both the generic CUDA kernel and the
// optional Triton kernel to consume the existing BF16 ABI.
bool DeepSeekV4PrepareCudaOutput(fastllm::Data &output,
                                 fastllm::DataType dataType,
                                 const std::vector<int> &dims);

template <typename InT>
__global__ void DeepSeekV4WoAQuantizeInputE4M3Kernel(
        const InT *input, __nv_bfloat16 *output) {
    __shared__ float warpMax[4];
    __shared__ float quantScale;
    int block = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    uint64_t offset = (uint64_t)block * 128 + threadIdx.x;
    float value = Dsv4ToFloat(input[offset]);
    float maximum = fabsf(value);
    for (int delta = 16; delta > 0; delta >>= 1) {
        maximum = fmaxf(maximum,
                        __shfl_down_sync(0xffffffff, maximum, delta));
    }
    if (lane == 0) {
        warpMax[warp] = maximum;
    }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < 4 ? warpMax[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            maximum = fmaxf(maximum,
                            __shfl_down_sync(0xffffffff, maximum, delta));
        }
        if (lane == 0) {
            float rawScale = fmaxf(maximum, 1.0e-10f) / 448.0f;
            quantScale = exp2f(ceilf(log2f(rawScale)));
        }
    }
    __syncthreads();
    output[offset] = __float2bfloat16_rn(
        Dsv4Fp8E4M3RoundTrip(value, quantScale));
}

bool DeepSeekV4PrepareWoAQuantizedInput(const fastllm::Data &input,
                                       fastllm::Data &output) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || input.dims.size() != 4 ||
        input.Count(0) == 0 || input.Count(0) % 128 != 0 ||
        !DeepSeekV4PrepareCudaOutput(
            output, fastllm::DataType::BFLOAT16, input.dims)) {
        return false;
    }
    int blocks = (int)(input.Count(0) / 128);
    if (input.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4WoAQuantizeInputE4M3Kernel<<<blocks, 128>>>(
            (const __nv_bfloat16 *)input.cudaData,
            (__nv_bfloat16 *)output.cudaData);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4WoAQuantizeInputE4M3Kernel<<<blocks, 128>>>(
            (const half *)input.cudaData,
            (__nv_bfloat16 *)output.cudaData);
    } else if (input.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4WoAQuantizeInputE4M3Kernel<<<blocks, 128>>>(
            (const float *)input.cudaData,
            (__nv_bfloat16 *)output.cudaData);
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

// Reproduce inference/model.py::Expert.forward before the down projection.
// DeepSeek-V4 applies the routed score before its second dynamic activation
// quantization; moving the score after the GEMM is not equivalent once the
// UE8M0 scale and E4M3 rounding are observable.
__global__ void DeepSeekV4PrepareMoeDownInputKernel(
        const __nv_bfloat16 *__restrict__ gateUp,
        __nv_bfloat16 *__restrict__ downInput,
        const float *__restrict__ routeScales,
        int intermediateDimension, float swigluLimit, bool quantize) {
    __shared__ float warpMax[4];
    __shared__ float quantScale;
    const int blocksPerRow = intermediateDimension / 128;
    const int row = blockIdx.x / blocksPerRow;
    const int blockInRow = blockIdx.x - row * blocksPerRow;
    const int dimension = blockInRow * 128 + threadIdx.x;
    const uint64_t gateUpOffset =
        (uint64_t)row * intermediateDimension * 2 + dimension * 2;
    const uint64_t outputOffset =
        (uint64_t)row * intermediateDimension + dimension;

    float gate = __bfloat162float(gateUp[gateUpOffset]);
    float up = __bfloat162float(gateUp[gateUpOffset + 1]);
    if (swigluLimit > 0.0f) {
        gate = fminf(gate, swigluLimit);
        up = fminf(swigluLimit, fmaxf(-swigluLimit, up));
    }
    // Keep the expert's observable multiplication order: SwiGLU forms h
    // first, then the routed score is applied before the BF16 boundary.
    // Reassociating this as (score * silu) * up can change the final BF16 bit.
    float h = (gate / (1.0f + expf(-gate))) * up;
    float value = routeScales[row] * h;
    // The expert contract has a BF16 boundary before act_quant.
    value = __bfloat162float(__float2bfloat16_rn(value));
    if (!quantize) {
        downInput[outputOffset] = __float2bfloat16_rn(value);
        return;
    }

    float maximum = fmaxf(1.0e-4f, fabsf(value));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int delta = 16; delta > 0; delta >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffffu, maximum, delta));
    }
    if (lane == 0) {
        warpMax[warp] = maximum;
    }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < 4 ? warpMax[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffu, maximum, delta));
        }
        if (lane == 0) {
            quantScale = exp2f(ceilf(log2f(maximum / 448.0f)));
        }
    }
    __syncthreads();
    downInput[outputOffset] = __float2bfloat16_rn(
        Dsv4Fp8E4M3RoundTrip(value, quantScale));
}

bool DeepSeekV4PrepareMoeDownInputImpl(
        const fastllm::Data &gateUp, fastllm::Data &downInput,
        const float *routeScales, float swigluLimit, bool quantize) {
    if (gateUp.dataDevice != fastllm::DataDevice::CUDA ||
        gateUp.dataType != fastllm::DataType::BFLOAT16 ||
        gateUp.cudaData == nullptr || routeScales == nullptr ||
        gateUp.dims.size() != 2 || gateUp.dims[0] <= 0 ||
        gateUp.dims[1] <= 0 || (gateUp.dims[1] & 1) != 0) {
        return false;
    }
    const int rows = gateUp.dims[0];
    const int intermediateDimension = gateUp.dims[1] / 2;
    if ((intermediateDimension & 127) != 0 ||
        !DeepSeekV4PrepareCudaOutput(
            downInput, fastllm::DataType::BFLOAT16,
            {rows, intermediateDimension})) {
        return false;
    }
    const int blocks = rows * intermediateDimension / 128;
    DeepSeekV4PrepareMoeDownInputKernel<<<blocks, 128>>>(
        (const __nv_bfloat16*)gateUp.cudaData,
        (__nv_bfloat16*)downInput.cudaData,
        routeScales, intermediateDimension, swigluLimit, quantize);
    return cudaGetLastError() == cudaSuccess;
}

template <typename InT>
__global__ void DeepSeekV4WoAFp8PairBlockReduceKernel(
        const InT *o, const uint8_t *w, const float *scales, __nv_bfloat16 *output,
        int bsz, int seqlen, int heads, int headDim, int groups, int oRank,
        int blockK, int blockM, int scaleCols) {
    extern __shared__ float partial[];
    float *partial0 = partial;
    float *partial1 = partial + blockDim.x;
    int pairsPerGroup = oRank / 2;
    int idx = blockIdx.x;
    int total = bsz * seqlen * groups * pairsPerGroup;
    if (idx >= total) {
        return;
    }

    int pair = idx % pairsPerGroup;
    int r0 = pair * 2;
    int r1 = r0 + 1;
    int tmp = idx / pairsPerGroup;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    int weightRow0 = g * oRank + r0;
    const uint8_t *wrow0 = w + (uint64_t)weightRow0 * groupDim;
    const uint8_t *wrow1 = wrow0 + groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;
    const float *rowScales = scales + (weightRow0 / blockK) * scaleCols;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int d = threadIdx.x; d < groupDim; d += blockDim.x) {
        float x = Dsv4ToFloat(src[d]);
        float scale = rowScales[d / blockM];
        sum0 += x * DeepSeekV4Fp8E4M3ScaledHalfToFloat(wrow0[d], scale);
        sum1 += x * DeepSeekV4Fp8E4M3ScaledHalfToFloat(wrow1[d], scale);
    }
    partial0[threadIdx.x] = sum0;
    partial1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint64_t outBase = (((uint64_t)b * seqlen + s) * groups + g) * oRank;
        output[outBase + r0] = __float2bfloat16_rn(partial0[0]);
        output[outBase + r1] = __float2bfloat16_rn(partial1[0]);
    }
}

template <typename InT, int RowsPerBlock>
__global__ void DeepSeekV4WoAFp8RowsBlockReduceKernel(
        const InT *o, const uint8_t *w, const float *scales, __nv_bfloat16 *output,
        int bsz, int seqlen, int heads, int headDim, int groups, int oRank,
        int blockK, int blockM, int scaleCols) {
    extern __shared__ float partial[];
    constexpr int threads = 256;
    int rowBlocksPerGroup = oRank / RowsPerBlock;
    int idx = blockIdx.x;
    int total = bsz * seqlen * groups * rowBlocksPerGroup;
    if (idx >= total) {
        return;
    }

    int rowBlock = idx % rowBlocksPerGroup;
    int rowStart = rowBlock * RowsPerBlock;
    int tmp = idx / rowBlocksPerGroup;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    int weightRowStart = g * oRank + rowStart;
    const uint8_t *weightRows =
        w + (uint64_t)weightRowStart * groupDim;
    const InT *src =
        o + (((uint64_t)b * seqlen + s) * heads +
             g * headsPerGroup) * headDim;
    const float *rowScales =
        scales + (weightRowStart / blockK) * scaleCols;

    float sums[RowsPerBlock] = {};
    for (int d = threadIdx.x; d < groupDim; d += threads) {
        float x = Dsv4ToFloat(src[d]);
        float scale = rowScales[d / blockM];
#pragma unroll
        for (int row = 0; row < RowsPerBlock; row++) {
            sums[row] +=
                x * DeepSeekV4Fp8E4M3ScaledHalfToFloat(
                        weightRows[(uint64_t)row * groupDim + d], scale);
        }
    }
#pragma unroll
    for (int row = 0; row < RowsPerBlock; row++) {
        partial[row * threads + threadIdx.x] = sums[row];
    }
    __syncthreads();

    for (int stride = threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int row = 0; row < RowsPerBlock; row++) {
                partial[row * threads + threadIdx.x] +=
                    partial[row * threads + threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint64_t outBase =
            (((uint64_t)b * seqlen + s) * groups + g) * oRank +
            rowStart;
#pragma unroll
        for (int row = 0; row < RowsPerBlock; row++) {
            output[outBase + row] =
                __float2bfloat16_rn(partial[row * threads]);
        }
    }
}

// Prefill has many query tokens but reuses the same wo_a matrix for every
// token.  The one-token kernel above reloads each FP8 weight once per token.
// Tile the token dimension as well as the output-row dimension so one decoded
// weight feeds several independent dot products.  Every (token, row) sum still
// visits d in exactly the same order and uses the same 256-way reduction tree
// as DeepSeekV4WoAFp8RowsBlockReduceKernel.  Consequently this changes neither
// the FP8 -> FP16 weight rounding nor the FP32 accumulation semantics.
template <typename InT, int TokensPerBlock, int RowsPerBlock>
__global__ void DeepSeekV4WoAFp8TokenRowsBlockReduceKernel(
        const InT *o, const uint8_t *w, const float *scales,
        __nv_bfloat16 *output,
        int bsz, int seqlen, int heads, int headDim, int groups, int oRank,
        int blockK, int blockM, int scaleCols) {
    extern __shared__ float partial[];
    constexpr int threads = 256;

    const int totalTokens = bsz * seqlen;
    const int tokenStart = blockIdx.x * TokensPerBlock;
    const int rowStart = blockIdx.y * RowsPerBlock;
    const int g = blockIdx.z;
    if (tokenStart >= totalTokens || rowStart >= oRank || g >= groups) {
        return;
    }

    const int headsPerGroup = heads / groups;
    const int groupDim = headsPerGroup * headDim;
    const int weightRowStart = g * oRank + rowStart;
    const uint8_t *weightRows =
        w + (uint64_t)weightRowStart * groupDim;
    const float *rowScales =
        scales + (weightRowStart / blockK) * scaleCols;

    float sums[TokensPerBlock][RowsPerBlock] = {};
    for (int d = threadIdx.x; d < groupDim; d += threads) {
        const float scale = rowScales[d / blockM];
        float decodedWeights[RowsPerBlock];
#pragma unroll
        for (int row = 0; row < RowsPerBlock; row++) {
            decodedWeights[row] = DeepSeekV4Fp8E4M3ScaledHalfToFloat(
                weightRows[(uint64_t)row * groupDim + d], scale);
        }
#pragma unroll
        for (int token = 0; token < TokensPerBlock; token++) {
            const int flatToken = tokenStart + token;
            float x = 0.0f;
            if (flatToken < totalTokens) {
                const InT *src =
                    o + ((uint64_t)flatToken * heads +
                         g * headsPerGroup) * headDim;
                x = Dsv4ToFloat(src[d]);
            }
#pragma unroll
            for (int row = 0; row < RowsPerBlock; row++) {
                sums[token][row] += x * decodedWeights[row];
            }
        }
    }

#pragma unroll
    for (int token = 0; token < TokensPerBlock; token++) {
#pragma unroll
        for (int row = 0; row < RowsPerBlock; row++) {
            partial[(token * RowsPerBlock + row) * threads + threadIdx.x] =
                sums[token][row];
        }
    }
    __syncthreads();

    // The first three levels cross warp boundaries and therefore stay in
    // shared memory.  After stride 32, warp 0 owns the same 32 partials that
    // the original block-wide tree would reduce; shuffles reproduce the
    // remaining 16/8/4/2/1 additions without four more CTA barriers.
    for (int stride = threads >> 1; stride >= 32; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int token = 0; token < TokensPerBlock; token++) {
#pragma unroll
                for (int row = 0; row < RowsPerBlock; row++) {
                    const int base =
                        (token * RowsPerBlock + row) * threads + threadIdx.x;
                    partial[base] += partial[base + stride];
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
#pragma unroll
        for (int token = 0; token < TokensPerBlock; token++) {
            const int flatToken = tokenStart + token;
#pragma unroll
            for (int row = 0; row < RowsPerBlock; row++) {
                float value = partial[
                    (token * RowsPerBlock + row) * threads + threadIdx.x];
#pragma unroll
                for (int stride = 16; stride > 0; stride >>= 1) {
                    value += __shfl_down_sync(0xffffffffu, value, stride);
                }
                if (threadIdx.x == 0 && flatToken < totalTokens) {
                    const uint64_t outBase =
                        ((uint64_t)flatToken * groups + g) * oRank + rowStart;
                    output[outBase + row] = __float2bfloat16_rn(
                        value);
                }
            }
        }
    }
}

template <typename InT, int TokensPerBlock, int RowsPerBlock>
bool DeepSeekV4LaunchWoAFp8TokenRowsBlockReduce(
        const InT *o, const uint8_t *w, const float *scales,
        __nv_bfloat16 *output,
        int bsz, int seqlen, int heads, int headDim, int groups, int oRank,
        int blockK, int blockM, int scaleCols) {
    constexpr int sharedBytes =
        TokensPerBlock * RowsPerBlock * 256 * sizeof(float);
    if (sharedBytes > 48 * 1024) {
        cudaError_t state = cudaFuncSetAttribute(
            DeepSeekV4WoAFp8TokenRowsBlockReduceKernel<
                InT, TokensPerBlock, RowsPerBlock>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes);
        if (state != cudaSuccess) {
            return false;
        }
    }
    const int totalTokens = bsz * seqlen;
    const dim3 grid(
        (totalTokens + TokensPerBlock - 1) / TokensPerBlock,
        oRank / RowsPerBlock, groups);
    DeepSeekV4WoAFp8TokenRowsBlockReduceKernel<
        InT, TokensPerBlock, RowsPerBlock>
        <<<grid, 256, sharedBytes>>>(
            o, w, scales, output, bsz, seqlen, heads, headDim,
            groups, oRank, blockK, blockM, scaleCols);
    return cudaGetLastError() == cudaSuccess;
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAFloatAccKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                            int bsz, int seqlen, int heads, int headDim,
                                            int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float v = 0.0f;
    for (int d = 0; d < groupDim; d++) {
        v += Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
    }
    output[idx] = __float2bfloat16_rn(v);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAKahanAccKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                            int bsz, int seqlen, int heads, int headDim,
                                            int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float sum = 0.0f;
    float c = 0.0f;
    for (int d = 0; d < groupDim; d++) {
        float prod = Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
        float y = prod - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    output[idx] = __float2bfloat16_rn(sum);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAPairKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                        int bsz, int seqlen, int heads, int headDim,
                                        int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairsPerGroup = oRank / 2;
    int total = bsz * seqlen * groups * pairsPerGroup;
    if (idx >= total) {
        return;
    }

    int pair = idx % pairsPerGroup;
    int r0 = pair * 2;
    int r1 = r0 + 1;
    int tmp = idx / pairsPerGroup;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow0 = w + ((uint64_t)g * oRank + r0) * groupDim;
    const WT *wrow1 = wrow0 + groupDim;

    double v0 = 0.0;
    double v1 = 0.0;
    int d = 0;
    for (int hh = 0; hh < headsPerGroup; hh++) {
        const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup + hh) * headDim;
        for (int localD = 0; localD < headDim; localD++, d++) {
            double x = (double)Dsv4ToFloat(src[localD]);
            v0 += x * Dsv4ToFloat(wrow0[d]);
            v1 += x * Dsv4ToFloat(wrow1[d]);
        }
    }
    uint64_t outBase = (((uint64_t)b * seqlen + s) * groups + g) * oRank;
    output[outBase + r0] = __float2bfloat16_rn((float)v0);
    output[outBase + r1] = __float2bfloat16_rn((float)v1);
}

__device__ __forceinline__ float DeepSeekV4InvFreq(int idx, int ropeDim, float base,
                                                   int originalSeqLen, float factor,
                                                   int betaFast, int betaSlow) {
    float inv = 1.0f / powf(base, (float)(idx * 2) / (float)ropeDim);
    if (originalSeqLen > 0) {
        float lowF = ropeDim * logf((float)originalSeqLen / (betaFast * 2.0f * 3.14159265358979323846f)) /
                     (2.0f * logf(base));
        float highF = ropeDim * logf((float)originalSeqLen / (betaSlow * 2.0f * 3.14159265358979323846f)) /
                      (2.0f * logf(base));
        int low = max((int)floorf(lowF), 0);
        int high = min((int)ceilf(highF), ropeDim - 1);
        if (low == high) {
            high++;
        }
        float ramp = fminf(1.0f, fmaxf(0.0f, ((float)idx - low) / (float)(high - low)));
        float smooth = 1.0f - ramp;
        inv = inv / factor * (1.0f - smooth) + inv * smooth;
    }
    return inv;
}

template <typename T>
__global__ void DeepSeekV4ScaleQRotaryKernel(T *q, int rows, int seqlen, int heads, int dim,
                                             int ropeDim, float ropeBase, int startPos,
                                             int originalSeqLen, float ropeFactor,
                                             int betaFast, int betaSlow, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = q + (uint64_t)row * dim;
    double ss = 0.0;
    for (int d = 0; d < dim; d++) {
        double v = (double)Dsv4ToFloat(ptr[d]);
        ss += v * v;
    }
    float scale = 1.0f / sqrtf((float)(ss / dim) + eps);
    for (int d = 0; d < dim; d++) {
        ptr[d] = Dsv4FromFloat<T>(Dsv4ToFloat(ptr[d]) * scale);
    }

    int s = (row / heads) % seqlen;
    int pos = startPos + s;
    int off = dim - ropeDim;
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }
}

template <typename T>
__global__ void DeepSeekV4ScaleQRotaryBlockKernel(T *q, int rows, int seqlen, int heads, int dim,
                                                  int ropeDim, float ropeBase, int startPos,
                                                  int originalSeqLen, float ropeFactor,
                                                  int betaFast, int betaSlow, float eps,
                                                  const int32_t *decodeMeta) {
    extern __shared__ float partial[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = q + (uint64_t)row * dim;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = Dsv4ToFloat(ptr[d]);
        ss += v * v;
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[0] = rsqrtf(partial[0] / dim + eps);
    }
    __syncthreads();
    float scale = partial[0];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        ptr[d] = Dsv4FromFloat<T>(Dsv4ToFloat(ptr[d]) * scale);
    }
    __syncthreads();

    int s = (row / heads) % seqlen;
    int dynamicStartPos = decodeMeta == nullptr ? startPos : decodeMeta[0];
    int pos = dynamicStartPos + s;
    int off = dim - ropeDim;
    for (int i = threadIdx.x * 2; i < ropeDim; i += blockDim.x * 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }
}

template <typename T>
__global__ void DeepSeekV4RotaryQuantKernel(T *x, int rows, int seqlen, int heads, int dim,
                                            int ropeDim, float ropeBase, int startPos,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow, int quantDim,
                                            int blockSize, int posStep) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = x + (uint64_t)row * dim;
    int s = (row / heads) % seqlen;
    int pos = startPos + s * posStep;
    int off = dim - ropeDim;
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }

    for (int start = 0; start < quantDim; start += blockSize) {
        int end = min(start + blockSize, quantDim);
        float amax = 1e-4f;
        for (int d = start; d < end; d++) {
            amax = fmaxf(amax, fabsf(Dsv4ToFloat(ptr[d])));
        }
        float qScale = powf(2.0f, ceilf(log2f(amax / 448.0f)));
        for (int d = start; d < end; d++) {
            float qv = fminf(448.0f, fmaxf(-448.0f, Dsv4ToFloat(ptr[d]) / qScale));
            float rounded = __bfloat162float(__float2bfloat16_rn(qv)) * qScale;
            ptr[d] = Dsv4FromFloat<T>(rounded);
        }
    }
}

template <typename T>
__global__ void DeepSeekV4RotaryQuantBlockKernel(T *x, int rows, int seqlen, int heads, int dim,
                                                 int ropeDim, float ropeBase, int startPos,
                                                 int originalSeqLen, float ropeFactor,
                                                 int betaFast, int betaSlow, int quantDim,
                                                 int blockSize, int posStep,
                                                 const int32_t *decodeMeta,
                                                 bool realFp8) {
    extern __shared__ float partial[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = x + (uint64_t)row * dim;
    int s = (row / heads) % seqlen;
    int dynamicStartPos = decodeMeta == nullptr ? startPos : decodeMeta[0];
    int pos = dynamicStartPos + s * posStep;
    int off = dim - ropeDim;
    for (int i = threadIdx.x * 2; i < ropeDim; i += blockDim.x * 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }
    __syncthreads();

    for (int start = 0; start < quantDim; start += blockSize) {
        int end = min(start + blockSize, quantDim);
        float amax = 1e-4f;
        for (int d = start + threadIdx.x; d < end; d += blockDim.x) {
            amax = fmaxf(amax, fabsf(Dsv4ToFloat(ptr[d])));
        }
        partial[threadIdx.x] = amax;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partial[0] = powf(2.0f, ceilf(log2f(partial[0] / 448.0f)));
        }
        __syncthreads();
        float qScale = partial[0];
        for (int d = start + threadIdx.x; d < end; d += blockDim.x) {
            float qv = fminf(448.0f, fmaxf(-448.0f, Dsv4ToFloat(ptr[d]) / qScale));
            float rounded = realFp8 ?
                Dsv4Fp8E4M3RoundTrip(Dsv4ToFloat(ptr[d]), qScale) :
                __bfloat162float(__float2bfloat16_rn(qv)) * qScale;
            ptr[d] = Dsv4FromFloat<T>(rounded);
        }
        __syncthreads();
    }
}

template <typename T>
__device__ __forceinline__ void DeepSeekV4PackSparseMlaRow(
        const T *source, uint8_t *packedCache, int logicalPosition) {
    constexpr int kElemsPerLane = kDeepSeekV4SparseMlaHeadDim / 32;
    int lane = threadIdx.x & 31;
    int dimBase = lane * kElemsPerLane;
    float values[kElemsPerLane];
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        values[i] = Dsv4ToFloat(source[dimBase + i]);
    }

    int page = logicalPosition / kDeepSeekV4SparseMlaPageSize;
    int pageOffset = logicalPosition % kDeepSeekV4SparseMlaPageSize;
    uint8_t *pageBase = packedCache +
        (uint64_t)page * kDeepSeekV4SparseMlaPageBytes;
    uint8_t *tokenData = pageBase +
        (uint64_t)pageOffset * kDeepSeekV4SparseMlaTokenDataBytes;
    uint8_t *tokenScales = pageBase +
        kDeepSeekV4SparseMlaPageSize * kDeepSeekV4SparseMlaTokenDataBytes +
        pageOffset * kDeepSeekV4SparseMlaScaleBytes;

    if (dimBase < kDeepSeekV4SparseMlaNopeDim) {
        float amax = 1.0e-4f;
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
            amax = fmaxf(amax, fabsf(values[i]));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        float exponent = ceilf(log2f(amax / 448.0f));
        float invScale = exp2f(-exponent);
        uint4 packedBytes;
        uint8_t *bytes = reinterpret_cast<uint8_t *>(&packedBytes);
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
            float scaled = fminf(448.0f,
                                 fmaxf(-448.0f, values[i] * invScale));
            __nv_fp8_e4m3 quantized(scaled);
            bytes[i] = quantized.__x;
        }
        *reinterpret_cast<uint4 *>(tokenData + dimBase) = packedBytes;
        if ((lane & 3) == 0) {
            int scaleIndex = lane >> 2;
            int encoded = max(0, min(255, (int)exponent + 127));
            tokenScales[scaleIndex] = (uint8_t)encoded;
        }
        if (lane == 0) {
            tokenScales[7] = 0;
        }
    } else {
        uint4 rope0, rope1;
        __nv_bfloat16 *ropeFirst =
            reinterpret_cast<__nv_bfloat16 *>(&rope0);
        __nv_bfloat16 *ropeSecond =
            reinterpret_cast<__nv_bfloat16 *>(&rope1);
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
            if (i < 8) {
                ropeFirst[i] = __float2bfloat16_rn(values[i]);
            } else {
                ropeSecond[i - 8] = __float2bfloat16_rn(values[i]);
            }
        }
        *reinterpret_cast<uint4 *>(tokenData + kDeepSeekV4SparseMlaNopeDim +
                                    (dimBase - kDeepSeekV4SparseMlaNopeDim) *
                                        (int)sizeof(__nv_bfloat16)) =
            rope0;
        *reinterpret_cast<uint4 *>(tokenData + kDeepSeekV4SparseMlaNopeDim +
                                    (dimBase - kDeepSeekV4SparseMlaNopeDim + 8) *
                                        (int)sizeof(__nv_bfloat16)) =
            rope1;
    }
}

template <typename T>
__global__ void DeepSeekV4PackSparseMlaWindowInitialKernel(
        const T *windowKV, uint8_t *packedCache, int totalLen,
        int windowSize) {
    int liveWindow = min(totalLen, windowSize);
    int local = blockIdx.x;
    if (local >= liveWindow || threadIdx.x >= 32) {
        return;
    }
    int logicalPosition = totalLen - liveWindow + local;
    int sourceSlot = logicalPosition % windowSize;
    DeepSeekV4PackSparseMlaRow(
        windowKV + (uint64_t)sourceSlot * kDeepSeekV4SparseMlaHeadDim,
        packedCache, logicalPosition);
}

template <typename T>
__global__ void DeepSeekV4PackSparseMlaLinearInitialKernel(
        const T *source, uint8_t *packedCache, int count) {
    int logicalPosition = blockIdx.x;
    if (logicalPosition >= count || threadIdx.x >= 32) {
        return;
    }
    DeepSeekV4PackSparseMlaRow(
        source + (uint64_t)logicalPosition * kDeepSeekV4SparseMlaHeadDim,
        packedCache, logicalPosition);
}

__global__ void DeepSeekV4PackSparseMlaWindowUpdatesKernel(
        const float *windowKV, uint8_t *packedCache,
        const int32_t *decodeMeta, int seqlen, int windowSize,
        int packedCapacity) {
    int s = blockIdx.x;
    if (s >= seqlen || threadIdx.x >= 32) {
        return;
    }
    int logicalPosition = decodeMeta[0] + s;
    if (logicalPosition < 0 || logicalPosition >= packedCapacity) {
        return;
    }
    int sourceSlot = logicalPosition % windowSize;
    DeepSeekV4PackSparseMlaRow(
        windowKV + (uint64_t)sourceSlot * kDeepSeekV4SparseMlaHeadDim,
        packedCache, logicalPosition);
}

__global__ void DeepSeekV4PackSparseMlaCompressedUpdatesKernel(
        const __nv_bfloat16 *compressedKV, uint8_t *packedCache,
        const int32_t *decodeMeta, int seqlen, int compressRatio,
        int compressedCapacity) {
    int s = blockIdx.x;
    if (s >= seqlen || threadIdx.x >= 32) {
        return;
    }
    int totalLen = decodeMeta[0] + s + 1;
    if (totalLen <= 0 || totalLen % compressRatio != 0) {
        return;
    }
    int logicalPosition = totalLen / compressRatio - 1;
    if (logicalPosition < 0 || logicalPosition >= compressedCapacity) {
        return;
    }
    DeepSeekV4PackSparseMlaRow(
        compressedKV +
            (uint64_t)logicalPosition * kDeepSeekV4SparseMlaHeadDim,
        packedCache, logicalPosition);
}

__global__ void DeepSeekV4BuildSparseMlaDecodeIndicesKernel(
        const int32_t *decodeMeta, int seqlen, int compressRatio,
        int32_t *windowIndices, int *windowLengths,
        int32_t *compressedIndices, int *compressedLengths) {
    int s = blockIdx.x;
    if (s >= seqlen) {
        return;
    }
    int position = decodeMeta[0] + s;
    int windowLength = min(128, position + 1);
    int windowStart = position + 1 - windowLength;
    for (int k = threadIdx.x; k < 128; k += blockDim.x) {
        windowIndices[(uint64_t)s * 128 + k] =
            k < windowLength ? windowStart + k : 0;
    }
    if (threadIdx.x == 0) {
        windowLengths[s] = windowLength;
    }
    if (compressRatio > 0 && compressedIndices != nullptr &&
        compressedLengths != nullptr) {
        int compressedCount = (position + 1) / compressRatio;
        int compressedLength = min(512, compressedCount);
        int compressedStart = compressedCount - compressedLength;
        for (int k = threadIdx.x; k < 512; k += blockDim.x) {
            compressedIndices[(uint64_t)s * 512 + k] =
                k < compressedLength ? compressedStart + k : 0;
        }
        if (threadIdx.x == 0) {
            compressedLengths[s] = compressedLength;
        }
    }
}

constexpr int kDeepSeekV4IndexerHeads = 64;
constexpr int kDeepSeekV4IndexerHeadDim = 128;
constexpr int kDeepSeekV4IndexerTopK = 512;

template <typename WT>
__global__ void DeepSeekV4IndexerQuantQKernel(
        const __nv_bfloat16 *q, const WT *weights,
        const int32_t *decodeMeta, uint8_t *qFp8,
        float *foldedWeights, int seqlen, float ropeBase,
        int originalSeqLen, float ropeFactor, int betaFast,
        int betaSlow) {
    __shared__ float reduction[kDeepSeekV4IndexerHeadDim];
    int row = blockIdx.x;
    int totalRows = seqlen * kDeepSeekV4IndexerHeads;
    if (row >= totalRows || threadIdx.x >= kDeepSeekV4IndexerHeadDim) {
        return;
    }
    int token = row / kDeepSeekV4IndexerHeads;
    int head = row % kDeepSeekV4IndexerHeads;
    int d = threadIdx.x;
    const __nv_bfloat16 *src =
        q + (uint64_t)row * kDeepSeekV4IndexerHeadDim;
    float value = __bfloat162float(src[d]);
    constexpr int ropeOffset = kDeepSeekV4IndexerHeadDim -
                               kDeepSeekV4SparseMlaRopeDim;
    if (d >= ropeOffset) {
        int local = d - ropeOffset;
        int pair = local >> 1;
        float even = __bfloat162float(src[ropeOffset + pair * 2]);
        float odd = __bfloat162float(src[ropeOffset + pair * 2 + 1]);
        float inv = DeepSeekV4InvFreq(
            pair, kDeepSeekV4SparseMlaRopeDim, ropeBase,
            originalSeqLen, ropeFactor, betaFast, betaSlow);
        float angle = (decodeMeta[0] + token) * inv;
        float c = cosf(angle), s = sinf(angle);
        value = (local & 1) ? odd * c + even * s
                            : even * c - odd * s;
        // vLLM deliberately inserts this BF16 boundary before computing the
        // per-head UE8M0 scale.
        value = __bfloat162float(__float2bfloat16_rn(value));
    }
    reduction[d] = fabsf(value);
    __syncthreads();
    for (int stride = kDeepSeekV4IndexerHeadDim >> 1;
         stride > 0; stride >>= 1) {
        if (d < stride) {
            reduction[d] = fmaxf(reduction[d], reduction[d + stride]);
        }
        __syncthreads();
    }
    float scale = powf(
        2.0f, ceilf(log2f(fmaxf(reduction[0], 1.0e-4f) / 448.0f)));
    __nv_fp8_e4m3 quantized(
        fminf(448.0f, fmaxf(-448.0f, value / scale)));
    qFp8[(uint64_t)row * kDeepSeekV4IndexerHeadDim + d] = quantized.__x;
    if (d == 0) {
        constexpr float softmaxScale = 0.08838834764831845f; // 128^-0.5
        constexpr float headScale = 0.125f;                 // 64^-0.5
        foldedWeights[(uint64_t)token * kDeepSeekV4IndexerHeads + head] =
            Dsv4ToFloat(weights[(uint64_t)token *
                                kDeepSeekV4IndexerHeads + head]) *
            scale * softmaxScale * headScale;
    }
}

__global__ void DeepSeekV4IndexerQuantKKernel(
        const __nv_bfloat16 *kv, uint8_t *kvFp8, float *kvScales,
        int rows) {
    __shared__ float reduction[kDeepSeekV4IndexerHeadDim];
    int row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= kDeepSeekV4IndexerHeadDim) {
        return;
    }
    float value = __bfloat162float(
        kv[(uint64_t)row * kDeepSeekV4IndexerHeadDim + d]);
    reduction[d] = fabsf(value);
    __syncthreads();
    for (int stride = kDeepSeekV4IndexerHeadDim >> 1;
         stride > 0; stride >>= 1) {
        if (d < stride) {
            reduction[d] = fmaxf(reduction[d], reduction[d + stride]);
        }
        __syncthreads();
    }
    float scale = powf(
        2.0f, ceilf(log2f(fmaxf(reduction[0], 1.0e-4f) / 448.0f)));
    __nv_fp8_e4m3 quantized(
        fminf(448.0f, fmaxf(-448.0f, value / scale)));
    kvFp8[(uint64_t)row * kDeepSeekV4IndexerHeadDim + d] = quantized.__x;
    if (d == 0) {
        kvScales[row] = scale;
    }
}

__global__ void DeepSeekV4IndexerSeqMetaKernel(
        const int32_t *decodeMeta, uint32_t *starts, uint32_t *ends,
        int *lengths, int32_t *indices, int seqlen, int compressRatio,
        int kvCapacity) {
    int token = blockIdx.x;
    if (token >= seqlen) {
        return;
    }
    int count = min(kvCapacity,
                    max(0, (decodeMeta[0] + token + 1) / compressRatio));
    if (threadIdx.x == 0) {
        starts[token] = 0;
        ends[token] = (uint32_t)count;
        lengths[token] = min(kDeepSeekV4IndexerTopK, count);
    }
    // This is also the exact vLLM top-k shortcut for rowLen <= topK.  The
    // persistent selector below overwrites the row only when count > 512.
    if (count <= kDeepSeekV4IndexerTopK) {
        for (int i = threadIdx.x; i < kDeepSeekV4IndexerTopK;
             i += blockDim.x) {
            indices[(uint64_t)token * kDeepSeekV4IndexerTopK + i] =
                i < count ? i : -1;
        }
    }
}

__device__ __forceinline__ float DeepSeekV4IndexerFp8ToFloat(uint8_t raw) {
    __nv_fp8_e4m3 value;
    value.__x = raw;
    return static_cast<float>(value);
}

__global__ void DeepSeekV4IndexerLogitsGenericKernel(
        const uint8_t *q, const uint8_t *kv, const float *kvScales,
        const float *weights, const uint32_t *ends, float *logits,
        int seqlen, int kvCapacity) {
    int token = blockIdx.x;
    int key = blockIdx.y;
    if (token >= seqlen || key >= (int)ends[token]) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float warpContribution = 0.0f;
    for (int head = warp; head < kDeepSeekV4IndexerHeads; head += 8) {
        const uint8_t *qRow = q +
            ((uint64_t)token * kDeepSeekV4IndexerHeads + head) *
            kDeepSeekV4IndexerHeadDim;
        const uint8_t *kRow = kv +
            (uint64_t)key * kDeepSeekV4IndexerHeadDim;
        float dot = 0.0f;
#pragma unroll
        for (int d = lane; d < kDeepSeekV4IndexerHeadDim; d += 32) {
            dot += DeepSeekV4IndexerFp8ToFloat(qRow[d]) *
                   DeepSeekV4IndexerFp8ToFloat(kRow[d]);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            warpContribution += fmaxf(dot, 0.0f) *
                weights[(uint64_t)token * kDeepSeekV4IndexerHeads + head];
        }
    }
    __shared__ float warpSums[8];
    if (lane == 0) {
        warpSums[warp] = warpContribution;
    }
    __syncthreads();
    if (warp == 0) {
        float sum = lane < 8 ? warpSums[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            logits[(uint64_t)token * kvCapacity + key] =
                sum * kvScales[key];
        }
    }
}

__device__ __forceinline__ uint32_t DeepSeekV4IndexerBin(float value,
                                                         int step) {
    if (step == 0) {
        __half halfValue = __float2half(value);
        uint16_t bits = __half_as_ushort(halfValue);
        bits = (bits & 0x8000u) ? bits : (~bits & 0x7fffu);
        return bits >> 5;
    }
    uint32_t bits = __float_as_uint(value);
    bits = (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
    if (step == 1) {
        return bits >> 21;
    }
    if (step == 2) {
        return (bits >> 10) & 0x7ffu;
    }
    return bits & 0x3ffu;
}

template <int SHIFT>
__device__ __forceinline__ bool DeepSeekV4IndexerPatternMatches(
        float value, uint32_t pattern) {
    if (SHIFT == 0) {
        return true;
    }
    uint32_t bits = __float_as_uint(value);
    bits = (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
    return ((bits ^ pattern) >> SHIFT) == 0;
}

template <typename Func>
__device__ void DeepSeekV4IndexerVectorizedProcess(
        const float *input, int length, Func fn) {
    constexpr int valuesPerLoad = 4;
    union Float4Values {
        float4 wide;
        float values[valuesPerLoad];
    } loaded;
    int prefix = reinterpret_cast<uintptr_t>(input) % sizeof(float4) == 0 ?
                 0 : (sizeof(float4) -
                      reinterpret_cast<uintptr_t>(input) % sizeof(float4)) /
                     sizeof(float);
    prefix = min(prefix, length);
    if (threadIdx.x < prefix) {
        fn(input[threadIdx.x], threadIdx.x);
    }
    const float4 *wideInput = reinterpret_cast<const float4 *>(input + prefix);
    int wideLength = (length - prefix) / valuesPerLoad;
    for (int i = threadIdx.x; i < wideLength; i += blockDim.x) {
        loaded.wide = wideInput[i];
        int base = prefix + i * valuesPerLoad;
#pragma unroll
        for (int j = 0; j < valuesPerLoad; ++j) {
            fn(loaded.values[j], base + j);
        }
    }
    int tail = prefix + wideLength * valuesPerLoad + threadIdx.x;
    if (tail < length) {
        fn(input[tail], tail);
    }
}

template <int STEP, typename SharedState>
__device__ bool DeepSeekV4IndexerHistogramStep(
        const float *logits, int rowEnd, uint32_t &pattern,
        int &thresholdBin, int32_t *selected, SharedState &shared) {
    constexpr int threads = 512;
    constexpr int bins = 2048;
    constexpr int finalItems = 2048;
    for (int i = threadIdx.x; i < bins; i += threads) {
        shared.histogram[i] = 0;
    }
    __syncthreads();
    constexpr int shift = STEP < 2 ? 0 : (STEP == 2 ? 21 : 10);
    if (STEP == 2) {
        pattern = (uint32_t)(thresholdBin & 0x7ff) << shift;
    } else if (STEP == 3) {
        pattern |= (uint32_t)(thresholdBin & 0x7ff) << shift;
    }
    DeepSeekV4IndexerVectorizedProcess(logits, rowEnd,
        [&](float value, int) {
            if (DeepSeekV4IndexerPatternMatches<shift>(value, pattern)) {
                atomicAdd(&shared.histogram[
                    DeepSeekV4IndexerBin(value, STEP)], 1);
            }
        });
    __syncthreads();

    int previous = shared.found;
    for (int round = 0; round < bins / threads; ++round) {
        int idx = threadIdx.x + round * threads;
        int count = shared.histogram[idx];
        __syncthreads();
        int prefix = 0, total = 0;
        using Scan = cub::BlockScan<int, threads>;
        Scan(shared.scan).ExclusiveSum(count, prefix, total);
        prefix += previous;
        total += previous;
        shared.histogram[idx] = prefix;
        __syncthreads();
        bool found = false;
        if (prefix < kDeepSeekV4IndexerTopK) {
            int next = threadIdx.x == threads - 1 ? total :
                       shared.histogram[idx + 1];
            if (next >= kDeepSeekV4IndexerTopK) {
                shared.threshold = idx;
                shared.finalBinSize = next - prefix;
                found = true;
            }
        }
        if (__syncthreads_or(found)) {
            break;
        }
        previous = total;
    }
    __syncthreads();
    thresholdBin = shared.threshold;
    DeepSeekV4IndexerVectorizedProcess(logits, rowEnd,
        [&](float value, int idx) {
            if (!DeepSeekV4IndexerPatternMatches<shift>(value, pattern)) {
                return;
            }
            uint32_t bin = DeepSeekV4IndexerBin(value, STEP);
            bool direct = (STEP == 0 && shared.finalBinSize <= finalItems) ||
                          STEP >= 1;
            if (bin < (uint32_t)thresholdBin && direct) {
                int dst = atomicAdd(&shared.found, 1);
                if (dst < kDeepSeekV4IndexerTopK) {
                    selected[dst] = idx;
                }
            }
            if (STEP < 3 && bin == (uint32_t)thresholdBin &&
                shared.finalBinSize <= finalItems) {
                int dst = atomicAdd(&shared.finalCount, 1);
                if (dst < finalItems) {
                    shared.finalLogits[dst] = value;
                    shared.finalIndices[dst] = idx;
                }
            } else if (STEP == 3 && bin == (uint32_t)thresholdBin) {
                int dst = atomicAdd(&shared.histogram[bin], 1);
                if (dst < kDeepSeekV4IndexerTopK) {
                    selected[dst] = idx;
                }
            }
        });
    __syncthreads();
    return shared.finalBinSize > finalItems;
}

__global__ __launch_bounds__(512) void DeepSeekV4IndexerTopKKernel(
        const float *logits, const uint32_t *ends, int32_t *indices,
        int seqlen, int stride) {
    int row = blockIdx.x;
    if (row >= seqlen) {
        return;
    }
    int rowEnd = (int)ends[row];
    int32_t *output = indices + (uint64_t)row * kDeepSeekV4IndexerTopK;
    if (rowEnd <= kDeepSeekV4IndexerTopK) {
        return;
    }
    constexpr int bins = 2048;
    constexpr int finalItems = 2048;
    struct SharedState {
        typename cub::BlockScan<int, 512>::TempStorage scan;
        int histogram[bins];
        int finalIndices[finalItems];
        float finalLogits[finalItems];
        int threshold;
        int finalCount;
        int finalBinSize;
        int found;
    };
    __shared__ SharedState shared;
    if (threadIdx.x == 0) {
        shared.finalCount = 0;
        shared.found = 0;
    }
    __syncthreads();
    uint32_t pattern = 0;
    int threshold = -1;
    const float *rowLogits = logits + (uint64_t)row * stride;
    bool next = DeepSeekV4IndexerHistogramStep<0>(
        rowLogits, rowEnd, pattern, threshold, output, shared);
    if (next) {
        next = DeepSeekV4IndexerHistogramStep<1>(
            rowLogits, rowEnd, pattern, threshold, output, shared);
    }
    if (next) {
        next = DeepSeekV4IndexerHistogramStep<2>(
            rowLogits, rowEnd, pattern, threshold, output, shared);
    }
    if (next) {
        DeepSeekV4IndexerHistogramStep<3>(
            rowLogits, rowEnd, pattern, threshold, output, shared);
    }
    if (!next) {
        int base = shared.found;
        for (int i = threadIdx.x; i < shared.finalCount;
             i += blockDim.x) {
            float value = shared.finalLogits[i];
            int rank = 0;
            for (int j = 0; j < shared.finalCount; ++j) {
                float other = shared.finalLogits[j];
                if (value < other || (value == other && i < j)) {
                    rank++;
                }
            }
            if (base + rank < kDeepSeekV4IndexerTopK) {
                output[base + rank] = shared.finalIndices[i];
            }
        }
    }
}

__global__ void DeepSeekV4SparseMlaInverseRotaryBf16Kernel(
        __nv_bfloat16 *output, const int32_t *decodeMeta, int rows,
        int seqlen, int heads, float ropeBase, int originalSeqLen,
        float ropeFactor, int betaFast, int betaSlow) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPairs = rows * (kDeepSeekV4SparseMlaRopeDim / 2);
    if (pair >= totalPairs) {
        return;
    }
    int row = pair / (kDeepSeekV4SparseMlaRopeDim / 2);
    int localPair = pair % (kDeepSeekV4SparseMlaRopeDim / 2);
    int s = (row / heads) % seqlen;
    int position = decodeMeta[0] + s;
    float inv = DeepSeekV4InvFreq(
        localPair, kDeepSeekV4SparseMlaRopeDim, ropeBase,
        originalSeqLen, ropeFactor, betaFast, betaSlow);
    float angle = position * inv;
    float c = cosf(angle);
    float sn = -sinf(angle);
    int d = kDeepSeekV4SparseMlaNopeDim + localPair * 2;
    __nv_bfloat16 *rowData =
        output + (uint64_t)row * kDeepSeekV4SparseMlaHeadDim;
    float a = __bfloat162float(rowData[d]);
    float b = __bfloat162float(rowData[d + 1]);
    rowData[d] = __float2bfloat16_rn(a * c - b * sn);
    rowData[d + 1] = __float2bfloat16_rn(a * sn + b * c);
}

// Fuse the Q and KV post-projection work into one launch.  One CTA owns each
// row, so speculative target/draft rows keep independent RoPE positions and
// sliding-window slots.  The arithmetic deliberately preserves FastLLM's two
// observable BF16 boundaries and its existing reduction order:
//   Q RMSNorm -> BF16 -> RoPE -> BF16
//   KV weighted RMSNorm -> BF16 -> (RoPE / quant round-trip) -> BF16
// The KV CTA also writes the FLOAT32 sliding-window cache.
__global__ void DeepSeekV4FusedQKVRopeCache512Kernel(
        __nv_bfloat16 *q, __nv_bfloat16 *kv, const float *kvNormWeight,
        float *windowKV, const int32_t *decodeMeta, int windowSize,
        int seqlen, int qHeads,
        float ropeBase, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, float eps, bool realFp8) {
    constexpr int kHeadDim = 512;
    constexpr int kRopeDim = 64;
    constexpr int kNopeDim = kHeadDim - kRopeDim;
    constexpr float kQuantMax = 448.0f;

    // One CTA owns one Q or KV row.  Q deliberately emulates the established
    // 256-thread ScaleQRotary reduction; KV emulates the established
    // 512-thread RMSNorm followed by the 256-thread rotary/quant reduction.
    // Keeping those reduction trees and BF16 stores exact avoids speculative
    // accept/reject changes from small normalization-order differences.
    int rowsPerToken = qHeads + 1;
    int token = blockIdx.x / rowsPerToken;
    int slot = blockIdx.x % rowsPerToken;
    if (token >= seqlen) {
        return;
    }
    bool isKV = slot == qHeads;
    __nv_bfloat16 *row = isKV ?
        kv + (uint64_t)token * kHeadDim :
        q + ((uint64_t)token * qHeads + slot) * kHeadDim;

    __shared__ float partial[256];
    __shared__ float warpSums[16];
    __shared__ float normScale;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (!isKV) {
        if (tid < 256) {
            float ss = 0.0f;
            for (int d = tid; d < kHeadDim; d += 256) {
                float value = __bfloat162float(row[d]);
                ss += value * value;
            }
            partial[tid] = ss;
        }
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            partial[0] = rsqrtf(partial[0] / kHeadDim + eps);
        }
        __syncthreads();
        float scale = partial[0];
        if (tid < 256) {
            for (int d = tid; d < kHeadDim; d += 256) {
                row[d] = __float2bfloat16_rn(
                    __bfloat162float(row[d]) * scale);
            }
        }
        __syncthreads();
    } else {
        const int pairs = kHeadDim / 2;
        const __nv_bfloat162 *inputPairs =
            reinterpret_cast<const __nv_bfloat162 *>(row);
        float sum = 0.0f;
        for (int pair = tid; pair < pairs; pair += 512) {
            __nv_bfloat162 value = inputPairs[pair];
            float lo = __bfloat162float(value.x);
            float hi = __bfloat162float(value.y);
            sum += lo * lo + hi * hi;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            warpSums[warp] = sum;
        }
        __syncthreads();
        if (warp == 0) {
            float value = lane < 16 ? warpSums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffffu, value, offset);
            }
            if (lane == 0) {
                normScale = rsqrtf(value / kHeadDim + eps);
            }
        }
        __syncthreads();

        __nv_bfloat162 *outputPairs =
            reinterpret_cast<__nv_bfloat162 *>(row);
        for (int pair = tid; pair < pairs; pair += 512) {
            __nv_bfloat162 value = inputPairs[pair];
            float lo = __bfloat162float(value.x);
            float hi = __bfloat162float(value.y);
            __nv_bfloat162 output;
            output.x = __float2bfloat16_rn(
                lo * normScale * __ldg(&kvNormWeight[pair * 2]));
            output.y = __float2bfloat16_rn(
                hi * normScale * __ldg(&kvNormWeight[pair * 2 + 1]));
            outputPairs[pair] = output;
        }
        __syncthreads();
    }

    int position = decodeMeta[0] + token;
    for (int i = tid * 2; i < kRopeDim; i += 256 * 2) {
        float inv = DeepSeekV4InvFreq(
            i / 2, kRopeDim, ropeBase, originalSeqLen,
            ropeFactor, betaFast, betaSlow);
        float angle = position * inv;
        float c = cosf(angle);
        float sn = sinf(angle);
        float a = __bfloat162float(row[kNopeDim + i]);
        float b = __bfloat162float(row[kNopeDim + i + 1]);
        row[kNopeDim + i] = __float2bfloat16_rn(a * c - b * sn);
        row[kNopeDim + i + 1] = __float2bfloat16_rn(a * sn + b * c);
    }
    __syncthreads();

    if (isKV) {
        for (int start = 0; start < kNopeDim; start += 64) {
            // A quant block is exactly 64 values. Two warp reductions preserve
            // the max/scale result while replacing the former 256-thread
            // shared-memory tree and its eight block-wide barriers.
            if (tid < 64) {
                float amax = fmaxf(
                    1.0e-4f,
                    fabsf(__bfloat162float(row[start + tid])));
                for (int offset = 16; offset > 0; offset >>= 1) {
                    amax = fmaxf(
                        amax,
                        __shfl_down_sync(0xffffffffu, amax, offset));
                }
                if (lane == 0) {
                    partial[warp] = amax;
                }
            }
            __syncthreads();
            if (tid == 0) {
                partial[0] = powf(
                    2.0f, ceilf(log2f(
                        fmaxf(partial[0], partial[1]) / kQuantMax)));
            }
            __syncthreads();
            float quantScale = partial[0];
            if (tid < 64) {
                int d = start + tid;
                float value = __bfloat162float(row[d]);
                float quantValue = fminf(
                    kQuantMax, fmaxf(-kQuantMax, value / quantScale));
                float rounded = realFp8 ?
                    Dsv4Fp8E4M3RoundTrip(value, quantScale) :
                    __bfloat162float(__float2bfloat16_rn(quantValue)) *
                        quantScale;
                row[d] = __float2bfloat16_rn(rounded);
            }
            __syncthreads();
        }

        int cacheSlot = position % windowSize;
        float *cache = windowKV + (uint64_t)cacheSlot * kHeadDim;
        if (tid < kHeadDim) {
            cache[tid] = __bfloat162float(row[tid]);
        }
    }
}

template <typename T>
__global__ void DeepSeekV4UpdateWindowKVCacheKernel(const T *kv, float *windowKV,
                                                    int bsz, int seqlen, int headDim, int startPos,
                                                    int windowSize, const int32_t *decodeMeta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * headDim;
    if (idx >= total) {
        return;
    }
    int d = idx % headDim;
    int tmp = idx / headDim;
    int s = tmp % seqlen;
    int b = tmp / seqlen;
    // Only the newest window survives when a cache-hit suffix is longer than
    // the ring.  Skipping older rows avoids multiple CUDA threads racing to
    // write the same slot and makes the result match the sequential CPU path.
    if (s < seqlen - windowSize) {
        return;
    }
    int dynamicStartPos = decodeMeta == nullptr ? startPos : decodeMeta[0];
    windowKV[((uint64_t)b * windowSize + ((dynamicStartPos + s) % windowSize)) * headDim + d] =
        Dsv4ToFloat(kv[((uint64_t)b * seqlen + s) * headDim + d]);
}

// DSpark consumes its main-model prefix as a chronological tensor, rather
// than the position-addressed ring used by the target sparse-attention cache.
// Assign one thread to each feature column so the overlapping left shift is
// ordered within that thread and does not require a temporary allocation.
template <typename T>
__global__ void DeepSeekV4AppendFullWindowKVCacheKernel(
        const T *kv, T *windowKV, int bsz, int seqlen, int headDim,
        int windowSize, int appendTokens) {
    int b = blockIdx.x;
    if (b >= bsz) {
        return;
    }
    // A long-prefill chunk can be much larger than the fixed DSpark window.
    // In that case none of the old window survives and only the trailing
    // windowSize rows from the committed prefix need to be copied.
    const int copied = appendTokens < windowSize ? appendTokens : windowSize;
    const int retained = windowSize - copied;
    const int sourceStart = appendTokens - copied;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        T *window = windowKV + (uint64_t)b * windowSize * headDim;
        const T *source = kv + (uint64_t)b * seqlen * headDim;
        for (int s = 0; s < retained; ++s) {
            window[(uint64_t)s * headDim + d] =
                window[(uint64_t)(s + copied) * headDim + d];
        }
        for (int s = 0; s < copied; ++s) {
            window[(uint64_t)(retained + s) * headDim + d] =
                source[(uint64_t)(sourceStart + s) * headDim + d];
        }
    }
}

template <typename T>
__global__ void DeepSeekV4StoreWindowKVCacheKernel(const T *kv, float *windowKV,
                                                   int bsz, int seqlen, int headDim,
                                                   int startPos, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * windowSize * headDim;
    if (idx >= total) {
        return;
    }
    int d = idx % headDim;
    int slot = (idx / headDim) % windowSize;
    int b = idx / (windowSize * headDim);
    int srcToken = -1;
    if (startPos == 0) {
        if (seqlen <= windowSize) {
            srcToken = slot < seqlen ? slot : -1;
        } else {
            int cutoff = seqlen % windowSize;
            srcToken = seqlen - windowSize + ((slot - cutoff + windowSize) % windowSize);
        }
    } else {
        srcToken = slot == (startPos % windowSize) ? 0 : -1;
    }
    windowKV[((uint64_t)b * windowSize + slot) * headDim + d] =
        srcToken >= 0 ? Dsv4ToFloat(kv[((uint64_t)b * seqlen + srcToken) * headDim + d]) : 0.0f;
}

__global__ void DeepSeekV4BuildWindowKVPrefixKernel(const float *windowKV, float *output,
                                                    int bsz, int prefixLen, int headDim,
                                                    int startPos, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * prefixLen * headDim;
    if (idx >= total) {
        return;
    }
    int d = idx % headDim;
    int tmp = idx / headDim;
    int s = tmp % prefixLen;
    int b = tmp / prefixLen;
    int firstPos = startPos - prefixLen;
    int srcSlot = (firstPos + s) % windowSize;
    output[((uint64_t)b * prefixLen + s) * headDim + d] =
        windowKV[((uint64_t)b * windowSize + srcSlot) * headDim + d];
}

template <typename T>
__global__ void DeepSeekV4BuildCompressedKVKernel(const T *kv, const T *score, const float *ape,
                                                  float *compressed, int bsz, int rawTokenBase,
                                                  int rawLen, int blockStart, int blockCount,
                                                  int compressRatio, int headDim, int wideDim,
                                                  bool overlap) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)bsz * blockCount * headDim;
    if (idx >= total) {
        return;
    }
    int d = (int)(idx % headDim);
    uint64_t tmp = idx / headDim;
    int localBlock = (int)(tmp % blockCount);
    int b = (int)(tmp / blockCount);
    int block = blockStart + localBlock;
    float mx = -1.0e30f;
    if (overlap) {
        if (block > 0) {
            for (int r = 0; r < compressRatio; r++) {
                int tok = (block - 1) * compressRatio + r;
                int localToken = tok - rawTokenBase;
                uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + d;
                mx = fmaxf(mx, Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + d]);
            }
        }
        for (int r = 0; r < compressRatio; r++) {
            int tok = block * compressRatio + r;
            int localToken = tok - rawTokenBase;
            uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + headDim + d;
            mx = fmaxf(mx, Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + headDim + d]);
        }
    } else {
        for (int r = 0; r < compressRatio; r++) {
            int tok = block * compressRatio + r;
            int localToken = tok - rawTokenBase;
            uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + d;
            mx = fmaxf(mx, Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + d]);
        }
    }

    float sum = 0.0f, value = 0.0f;
    if (overlap) {
        if (block > 0) {
            for (int r = 0; r < compressRatio; r++) {
                int tok = (block - 1) * compressRatio + r;
                int localToken = tok - rawTokenBase;
                uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + d;
                float e = expf(Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + d] - mx);
                sum += e;
                value += e * Dsv4ToFloat(kv[off]);
            }
        }
        for (int r = 0; r < compressRatio; r++) {
            int tok = block * compressRatio + r;
            int localToken = tok - rawTokenBase;
            uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + headDim + d;
            float e = expf(Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + headDim + d] - mx);
            sum += e;
            value += e * Dsv4ToFloat(kv[off]);
        }
    } else {
        for (int r = 0; r < compressRatio; r++) {
            int tok = block * compressRatio + r;
            int localToken = tok - rawTokenBase;
            uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + d;
            float e = expf(Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + d] - mx);
            sum += e;
            value += e * Dsv4ToFloat(kv[off]);
        }
    }
    compressed[((uint64_t)b * blockCount + localBlock) * headDim + d] = value / fmaxf(sum, 1e-30f);
}

// Finish the eager compressed-KV pipeline without materializing its two BF16
// temporaries.  The numerical boundaries deliberately mirror the standalone
// kernels:
//   FP32 build -> BF16 -> RMSNorm -> BF16 -> RoPE -> BF16 -> quant -> BF16.
// RMSNorm also preserves the exact virtual-thread/reduction layout used by
// FastllmRMSNormKernelInner1<64/512>, which is required for bitwise equality.
__global__ void DeepSeekV4FinalizeCompressedKVKernel(
        const float *compressed, const float *normWeight,
        __nv_bfloat16 *cache, int bsz, int blockCount, int blockStart,
        int cacheStride, int compressRatio, int headDim, int ropeDim,
        float ropeBase, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, bool realFp8) {
    extern __shared__ float shared[];
    float *values = shared;
    float *warpSums = values + headDim;
    float *quantPartial = warpSums + 16;
    __shared__ float normScale;
    __shared__ float quantScale;

    int row = blockIdx.x;
    int localBlock = row % blockCount;
    int b = row / blockCount;
    if (b >= bsz) {
        return;
    }

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float value = compressed[((uint64_t)b * blockCount + localBlock) *
                                 headDim + d];
        values[d] = __bfloat162float(__float2bfloat16_rn(value));
    }
    __syncthreads();

    const int rmsThreads = headDim == 128 ? 64 : 512;
    if (threadIdx.x < rmsThreads) {
        float sum2 = 0.0f;
        int bf2Channels = headDim / 2;
        for (int i = threadIdx.x; i < bf2Channels; i += rmsThreads) {
            float lo = values[i * 2];
            float hi = values[i * 2 + 1];
            sum2 += lo * lo + hi * hi;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
        }
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        if (lane == 0) {
            warpSums[warp] = sum2;
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        int rmsWarps = rmsThreads / 32;
        float value = threadIdx.x < rmsWarps ? warpSums[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (threadIdx.x == 0) {
            normScale = rsqrtf(value / headDim + 1.0e-6f);
        }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        values[d] = __bfloat162float(__float2bfloat16_rn(
            values[d] * normScale * __ldg(normWeight + d)));
    }
    __syncthreads();

    int ropeOffset = headDim - ropeDim;
    int rotaryPos = (blockStart + localBlock) * compressRatio;
    for (int i = threadIdx.x * 2; i < ropeDim; i += blockDim.x * 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase,
                                      originalSeqLen, ropeFactor,
                                      betaFast, betaSlow);
        float angle = rotaryPos * inv;
        float c = cosf(angle);
        float s = sinf(angle);
        float a = values[ropeOffset + i];
        float bb = values[ropeOffset + i + 1];
        values[ropeOffset + i] = __bfloat162float(
            __float2bfloat16_rn(a * c - bb * s));
        values[ropeOffset + i + 1] = __bfloat162float(
            __float2bfloat16_rn(a * s + bb * c));
    }
    __syncthreads();

    const int quantDim = headDim == 128 ? headDim : ropeOffset;
    const int quantBlock = headDim == 128 ? 128 : 64;
    for (int start = 0; start < quantDim; start += quantBlock) {
        int end = min(start + quantBlock, quantDim);
        if (threadIdx.x < 256) {
            float amax = 1.0e-4f;
            for (int d = start + threadIdx.x; d < end; d += 256) {
                amax = fmaxf(amax, fabsf(values[d]));
            }
            quantPartial[threadIdx.x] = amax;
        }
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                quantPartial[threadIdx.x] = fmaxf(
                    quantPartial[threadIdx.x],
                    quantPartial[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            quantScale = powf(
                2.0f, ceilf(log2f(quantPartial[0] / 448.0f)));
        }
        __syncthreads();
        if (threadIdx.x < 256) {
            for (int d = start + threadIdx.x; d < end; d += 256) {
                float value = values[d];
                float qv = fminf(
                    448.0f, fmaxf(-448.0f, value / quantScale));
                float rounded = realFp8 ?
                    Dsv4Fp8E4M3RoundTrip(value, quantScale) :
                    __bfloat162float(__float2bfloat16_rn(qv)) * quantScale;
                values[d] = __bfloat162float(
                    __float2bfloat16_rn(rounded));
            }
        }
        __syncthreads();
    }

    int block = blockStart + localBlock;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        cache[((uint64_t)b * cacheStride + block) * headDim + d] =
            __float2bfloat16_rn(values[d]);
    }
}

template <typename T>
__global__ void DeepSeekV4CompactCompressorRawKernel(
        T *kv, T *score, int bsz, int newLen,
        int wideDim, int rowCapacity, int dropLen) {
    uint64_t tensorElems = (uint64_t)bsz * newLen * wideDim;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tensorElems * 2) {
        return;
    }
    T *data = idx < tensorElems ? kv : score;
    uint64_t local = idx < tensorElems ? idx : idx - tensorElems;
    int d = local % wideDim;
    uint64_t row = local / wideDim;
    int s = row % newLen;
    int b = row / newLen;
    uint64_t dst = ((uint64_t)b * rowCapacity + s) * wideDim + d;
    uint64_t src = ((uint64_t)b * rowCapacity + dropLen + s) * wideDim + d;
    data[dst] = data[src];
}

template <typename T>
__global__ void DeepSeekV4InitGraphRawRingKernel(const T *raw, T *ring,
                                                 int bsz, int rawLen, int wideDim,
                                                 int rawTokenBase, int ringCapacity) {
    uint64_t total = (uint64_t)bsz * rawLen * wideDim;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int d = idx % wideDim;
    uint64_t tmp = idx / wideDim;
    int s = tmp % rawLen;
    int b = tmp / rawLen;
    int slot = (rawTokenBase + s) % ringCapacity;
    ring[((uint64_t)b * ringCapacity + slot) * wideDim + d] = raw[idx];
}

template <typename T>
__global__ void DeepSeekV4StoreGraphCompressorRawKernel(const T *kv, const T *score,
                                                        T *kvRing, T *scoreRing,
                                                        const int32_t *decodeMeta,
                                                        int bsz, int seqlen,
                                                        int wideDim,
                                                        int ringCapacity) {
    uint64_t total = (uint64_t)bsz * seqlen * wideDim;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int d = idx % wideDim;
    uint64_t row = idx / wideDim;
    int s = row % seqlen;
    int b = row / seqlen;
    int slot = (decodeMeta[0] + s) % ringCapacity;
    uint64_t dst = ((uint64_t)b * ringCapacity + slot) * wideDim + d;
    kvRing[dst] = kv[idx];
    scoreRing[dst] = score[idx];
}

template <typename T, typename WT>
__global__ void DeepSeekV4UpdateCompressedKVGraphKernel(
                                                        const T *kvRing, const T *scoreRing,
                                                        const float *ape, const WT *normWeight,
                                                        const int32_t *decodeMeta,
                                                        __nv_bfloat16 *compressedKV,
                                                        int bsz, int ringCapacity,
                                                        int compressedCapacity,
                                                        int seqlen,
                                                        int compressRatio, int headDim,
                                                        int wideDim, int ropeDim,
                                                        float ropeBase, int originalSeqLen,
                                                        float ropeFactor, int betaFast,
                                                        int betaSlow, bool realFp8) {
    extern __shared__ float shared[];
    float *values = shared;
    float *red = values + headDim;
    int s = blockIdx.x % seqlen;
    int b = blockIdx.x / seqlen;
    if (b >= bsz) {
        return;
    }
    int startPos = decodeMeta[0] + s;
    int totalLen = startPos + 1;
    if (totalLen <= 0 || totalLen % compressRatio != 0) {
        return;
    }
    int block = totalLen / compressRatio - 1;
    if (block < 0 || block >= compressedCapacity) {
        return;
    }
    bool overlap = compressRatio == 4;

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float mx = -INFINITY;
        if (overlap && block > 0) {
            for (int r = 0; r < compressRatio; r++) {
                int token = (block - 1) * compressRatio + r;
                int slot = token % ringCapacity;
                uint64_t off = ((uint64_t)b * ringCapacity + slot) * wideDim + d;
                mx = fmaxf(mx, Dsv4ToFloat(scoreRing[off]) + ape[(uint64_t)r * wideDim + d]);
            }
        }
        int currentBase = block * compressRatio;
        int currentDimOffset = overlap ? headDim : 0;
        for (int r = 0; r < compressRatio; r++) {
            int token = currentBase + r;
            int slot = token % ringCapacity;
            uint64_t off = ((uint64_t)b * ringCapacity + slot) * wideDim + currentDimOffset + d;
            mx = fmaxf(mx, Dsv4ToFloat(scoreRing[off]) +
                            ape[(uint64_t)r * wideDim + currentDimOffset + d]);
        }

        float sum = 0.0f, value = 0.0f;
        if (overlap && block > 0) {
            for (int r = 0; r < compressRatio; r++) {
                int token = (block - 1) * compressRatio + r;
                int slot = token % ringCapacity;
                uint64_t off = ((uint64_t)b * ringCapacity + slot) * wideDim + d;
                float e = expf(Dsv4ToFloat(scoreRing[off]) +
                               ape[(uint64_t)r * wideDim + d] - mx);
                sum += e;
                value += e * Dsv4ToFloat(kvRing[off]);
            }
        }
        for (int r = 0; r < compressRatio; r++) {
            int token = currentBase + r;
            int slot = token % ringCapacity;
            uint64_t off = ((uint64_t)b * ringCapacity + slot) * wideDim + currentDimOffset + d;
            float e = expf(Dsv4ToFloat(scoreRing[off]) +
                           ape[(uint64_t)r * wideDim + currentDimOffset + d] - mx);
            sum += e;
            value += e * Dsv4ToFloat(kvRing[off]);
        }
        // Match the existing pipeline's FP32 -> BF16 conversion before RMSNorm.
        values[d] = __bfloat162float(__float2bfloat16_rn(value / fmaxf(sum, 1e-30f)));
    }
    __syncthreads();

    float ss = 0.0f;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        ss += values[d] * values[d];
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float normScale = rsqrtf(red[0] / headDim + 1.0e-6f);
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        values[d] = __bfloat162float(__float2bfloat16_rn(
            values[d] * normScale * Dsv4ToFloat(normWeight[d])));
    }
    __syncthreads();

    int ropeOffset = headDim - ropeDim;
    int rotaryPos = block * compressRatio;
    for (int i = threadIdx.x * 2; i < ropeDim; i += blockDim.x * 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen,
                                      ropeFactor, betaFast, betaSlow);
        float angle = rotaryPos * inv;
        float c = cosf(angle), s = sinf(angle);
        float a = values[ropeOffset + i];
        float bb = values[ropeOffset + i + 1];
        values[ropeOffset + i] = __bfloat162float(__float2bfloat16_rn(a * c - bb * s));
        values[ropeOffset + i + 1] = __bfloat162float(__float2bfloat16_rn(a * s + bb * c));
    }
    __syncthreads();

    // Main MLA rows use seven 64-value NoPE scales and leave the BF16 RoPE
    // tail unquantized.  Indexer rows (headDim=128) use one scale across the
    // complete post-RoPE row, matching indexer_k_quant_and_cache/DeepGEMM.
    int quantDim = headDim == 128 ? headDim : ropeOffset;
    int quantBlock = headDim == 128 ? 128 : 64;
    for (int groupStart = 0; groupStart < quantDim; groupStart += quantBlock) {
        int groupEnd = min(groupStart + quantBlock, quantDim);
        float amax = 1.0e-4f;
        for (int d = groupStart + threadIdx.x; d < groupEnd; d += blockDim.x) {
            amax = fmaxf(amax, fabsf(values[d]));
        }
        red[threadIdx.x] = amax;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        float qScale = powf(2.0f, ceilf(log2f(red[0] / 448.0f)));
        for (int d = groupStart + threadIdx.x; d < groupEnd; d += blockDim.x) {
            float qv = fminf(448.0f, fmaxf(-448.0f, values[d] / qScale));
            values[d] = realFp8 ?
                Dsv4Fp8E4M3RoundTrip(values[d], qScale) :
                __bfloat162float(__float2bfloat16_rn(qv)) * qScale;
        }
        __syncthreads();
    }

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        compressedKV[((uint64_t)b * compressedCapacity + block) * headDim + d] =
            __float2bfloat16_rn(values[d]);
    }
}

__device__ __forceinline__ float DeepSeekV4Sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ float DeepSeekV4Softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

__global__ void DeepSeekV4RouteScoreTransformKernel(float *logits, int rows, int experts, int mode) {
    int row = blockIdx.x;
    float *rowData = logits + (uint64_t)row * experts;
    if (mode == 0) {
        float mx = -INFINITY;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            mx = fmaxf(mx, rowData[e]);
        }
        __shared__ float red[256];
        red[threadIdx.x] = mx;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        mx = red[0];
        double partial = 0.0;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            partial += (double)expf(rowData[e] - mx);
        }
        red[threadIdx.x] = (float)partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float sum = red[0];
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            rowData[e] = expf(rowData[e] - mx) / sum;
        }
    } else {
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            float raw = rowData[e];
            rowData[e] = mode == 1 ? DeepSeekV4Sigmoid(raw) : sqrtf(DeepSeekV4Softplus(raw));
        }
    }
}

struct DeepSeekV4RouterCandidate {
    float key;
    int id;
};

__device__ __forceinline__ bool DeepSeekV4RouterCandidateBetter(
        const DeepSeekV4RouterCandidate &a,
        const DeepSeekV4RouterCandidate &b) {
    if (a.id < 0) {
        return false;
    }
    if (b.id < 0) {
        return true;
    }
    return a.key != b.key ? a.key > b.key : a.id < b.id;
}

__device__ __forceinline__ DeepSeekV4RouterCandidate
DeepSeekV4RouterWarpBest(DeepSeekV4RouterCandidate value) {
    int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset >>= 1) {
        DeepSeekV4RouterCandidate other;
        other.key = __shfl_down_sync(0xffffffffu, value.key, offset);
        other.id = __shfl_down_sync(0xffffffffu, value.id, offset);
        if (lane + offset < 32 &&
            DeepSeekV4RouterCandidateBetter(other, value)) {
            value = other;
        }
    }
    return value;
}

// Return the same best candidate to every lane in a warp. A DeepSeek-V4
// routing row has exactly 256 values, so one warp can keep eight candidates
// per lane in registers and perform all six selections without shared memory.
// The explicit id tie-break matches the established 256-thread kernel.
__device__ __forceinline__ DeepSeekV4RouterCandidate
DeepSeekV4RouterWarpConsensusBest(DeepSeekV4RouterCandidate value) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        DeepSeekV4RouterCandidate other;
        other.key = __shfl_xor_sync(0xffffffffu, value.key, mask);
        other.id = __shfl_xor_sync(0xffffffffu, value.id, mask);
        if (DeepSeekV4RouterCandidateBetter(other, value)) {
            value = other;
        }
    }
    return value;
}

// Architecture-generic fused router. One CTA owns one token; every lane
// transforms one of the 256 logits and six block reductions select the routed
// experts. A winner writes its unbiased route weight before proceeding; the
// next round's first barrier also completes that write, avoiding a dedicated
// third barrier per round. This removes the intermediate transformed-logit
// write and the generic SelectExpert<64, 50> shared-memory merge tree while
// remaining usable on pre-SM120 CUDA devices.
__global__ __launch_bounds__(256)
void DeepSeekV4SqrtSoftplusTop6GenericKernel(
        const float *logits, const float *bias,
        int32_t *index, float *score, int tokens, float routeScale) {
    constexpr int experts = 256;
    constexpr int topk = 6;
    constexpr int warps = 8;
    constexpr float invalidKey = -FLT_MAX;

    __shared__ DeepSeekV4RouterCandidate warpBest[warps];
    __shared__ int selectedIds[topk];
    __shared__ float selectedWeights[topk];

    int token = blockIdx.x;
    int expert = threadIdx.x;
    if (token >= tokens || expert >= experts) {
        return;
    }
    int lane = expert & 31;
    int warp = expert >> 5;
    float raw = logits[(uint64_t)token * experts + expert];
    float weight = sqrtf(DeepSeekV4Softplus(raw));
    float key = weight + bias[expert];
    if (!isfinite(key)) {
        key = invalidKey;
    }
    if (!isfinite(weight)) {
        weight = 0.0f;
    }
    bool active = true;

#pragma unroll
    for (int rank = 0; rank < topk; rank++) {
        DeepSeekV4RouterCandidate local = {
            active ? key : invalidKey,
            active ? expert : -1,
        };
        local = DeepSeekV4RouterWarpBest(local);
        if (lane == 0) {
            warpBest[warp] = local;
        }
        __syncthreads();

        if (warp == 0) {
            DeepSeekV4RouterCandidate block = lane < warps ?
                warpBest[lane] :
                DeepSeekV4RouterCandidate{invalidKey, -1};
            block = DeepSeekV4RouterWarpBest(block);
            if (lane == 0) {
                selectedIds[rank] = block.id;
            }
        }
        __syncthreads();

        if (expert == selectedIds[rank]) {
            selectedWeights[rank] = weight;
            active = false;
        }
    }
    __syncthreads();

    if (expert == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int rank = 0; rank < topk; rank++) {
            sum += selectedWeights[rank];
        }
        if (!isfinite(sum) || fabsf(sum) < 1.0e-20f) {
            sum = 1.0f;
        }
        int32_t *tokenIndex = index + (uint64_t)token * topk;
        float *tokenScore = score + (uint64_t)token * topk;
#pragma unroll
        for (int rank = 0; rank < topk; rank++) {
            tokenIndex[rank] = selectedIds[rank];
            tokenScore[rank] = selectedWeights[rank] / sum * routeScale;
        }
    }
}

#ifndef USE_ROCM
// SM120 decode/verification specialization modeled after vLLM's
// topkGatingSoftplusSqrt schedule. Four independent routing rows share one
// CTA, while each warp owns one row and each lane keeps two aligned float4
// chunks. There are no block barriers or shared-memory round trips.
__global__ __launch_bounds__(128)
void DeepSeekV4SqrtSoftplusTop6Warp4Kernel(
        const float *logits, const float *bias,
        int32_t *index, float *score, int tokens, float routeScale) {
    constexpr int experts = 256;
    constexpr int topk = 6;
    constexpr int rowsPerBlock = 4;
    constexpr float invalidKey = -FLT_MAX;

    int lane = threadIdx.x;
    int rowInBlock = threadIdx.y;
    int token = blockIdx.x * rowsPerBlock + rowInBlock;
    if (lane >= 32 || rowInBlock >= rowsPerBlock || token >= tokens) {
        return;
    }

    const float4 *row4 = reinterpret_cast<const float4 *>(
        logits + (uint64_t)token * experts);
    const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
    float4 raw0 = row4[lane];
    float4 raw1 = row4[32 + lane];
    float4 bias0 = bias4[lane];
    float4 bias1 = bias4[32 + lane];
    float rawValues[8] = {
        raw0.x, raw0.y, raw0.z, raw0.w,
        raw1.x, raw1.y, raw1.z, raw1.w,
    };
    float biasValues[8] = {
        bias0.x, bias0.y, bias0.z, bias0.w,
        bias1.x, bias1.y, bias1.z, bias1.w,
    };
    float weights[8];
    float keys[8];
    int ids[8];
    bool active[8];

#pragma unroll
    for (int item = 0; item < 8; ++item) {
        int half = item >> 2;
        int component = item & 3;
        int expert = half * 128 + lane * 4 + component;
        float weight = sqrtf(DeepSeekV4Softplus(rawValues[item]));
        float key = weight + biasValues[item];
        if (!isfinite(key)) {
            key = invalidKey;
        }
        if (!isfinite(weight)) {
            weight = 0.0f;
        }
        weights[item] = weight;
        keys[item] = key;
        ids[item] = expert;
        active[item] = true;
    }

    int selectedIds[topk];
    float selectedWeights[topk];
#pragma unroll
    for (int rank = 0; rank < topk; ++rank) {
        DeepSeekV4RouterCandidate local = {invalidKey, -1};
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            DeepSeekV4RouterCandidate candidate = {
                active[item] ? keys[item] : invalidKey,
                active[item] ? ids[item] : -1,
            };
            if (DeepSeekV4RouterCandidateBetter(candidate, local)) {
                local = candidate;
            }
        }
        DeepSeekV4RouterCandidate best =
            DeepSeekV4RouterWarpConsensusBest(local);

        float selectedWeight = 0.0f;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            if (ids[item] == best.id) {
                selectedWeight = weights[item];
                active[item] = false;
            }
        }
        // Both 128-expert halves use the same lane mapping.
        int ownerLane = (best.id & 127) >> 2;
        selectedWeight = __shfl_sync(
            0xffffffffu, selectedWeight, ownerLane);
        if (lane == 0) {
            selectedIds[rank] = best.id;
            selectedWeights[rank] = selectedWeight;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int rank = 0; rank < topk; ++rank) {
            sum += selectedWeights[rank];
        }
        if (!isfinite(sum) || fabsf(sum) < 1.0e-20f) {
            sum = 1.0f;
        }
        int32_t *tokenIndex = index + (uint64_t)token * topk;
        float *tokenScore = score + (uint64_t)token * topk;
#pragma unroll
        for (int rank = 0; rank < topk; ++rank) {
            tokenIndex[rank] = selectedIds[rank];
            tokenScore[rank] =
                selectedWeights[rank] / sum * routeScale;
        }
    }
}
#endif

template <typename RouteT>
__global__ void DeepSeekV4HashRouteScoreKernel(float *logits, const RouteT *tid2eid,
                                               const int *inputIds, int singleTokenId,
                                               int32_t *index,
                                               float *score, int tokens, int experts,
                                               int topk, int mode, float routeScale,
                                               int routeRows) {
    int row = blockIdx.x;
    if (row >= tokens) {
        return;
    }
    float *rowData = logits + (uint64_t)row * experts;
    int32_t *outIndex = index + (uint64_t)row * topk;
    float *outScore = score + (uint64_t)row * topk;
    int tokenId = inputIds != nullptr ? inputIds[row] : singleTokenId;
    tokenId = max(0, min(tokenId, routeRows - 1));
    const RouteT *routeRow = tid2eid + (uint64_t)tokenId * topk;

    __shared__ float red[256];
    if (mode == 0) {
        float mx = -INFINITY;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            mx = fmaxf(mx, rowData[e]);
        }
        red[threadIdx.x] = mx;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        mx = red[0];

        float partial = 0.0f;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            partial += expf(rowData[e] - mx);
        }
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float denom = fmaxf(red[0], 1e-30f);
        for (int k = threadIdx.x; k < topk; k += blockDim.x) {
            int expert = (int)routeRow[k];
            expert = max(0, min(expert, experts - 1));
            outIndex[k] = expert;
            outScore[k] = expf(rowData[expert] - mx) / denom * routeScale;
        }
        return;
    }

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int k = 0; k < topk; k++) {
            int expert = (int)routeRow[k];
            expert = max(0, min(expert, experts - 1));
            float raw = rowData[expert];
            float v = mode == 1 ? DeepSeekV4Sigmoid(raw) : sqrtf(DeepSeekV4Softplus(raw));
            outIndex[k] = expert;
            outScore[k] = v;
            sum += v;
        }
        float invSum = 1.0f / fmaxf(sum, 1e-30f);
        for (int k = 0; k < topk; k++) {
            outScore[k] = outScore[k] * invSum * routeScale;
        }
    }
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcPreDotsKernel(const XT *x, const WT *w, float *dots,
                                          int tokens, int flatDim, int mixHc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * mixHc;
    if (idx >= total) {
        return;
    }
    int m = idx % mixHc;
    int t = idx / mixHc;
    const XT *xrow = x + (uint64_t)t * flatDim;
    const WT *wrow = w + (uint64_t)m * flatDim;
    double v = 0.0;
    for (int k = 0; k < flatDim; k++) {
        v += (double)Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
    }
    dots[idx] = (float)v;
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcPreDotsBlockKernel(const XT *x, const WT *w, float *dots,
                                               int tokens, int flatDim, int mixHc, int dotsStride,
                                               int dotParts) {
    extern __shared__ float red[];
    int idx = blockIdx.x / dotParts;
    int part = blockIdx.x - idx * dotParts;
    if (idx >= tokens * dotsStride) {
        return;
    }
    int m = idx % dotsStride;
    int t = idx / dotsStride;
    int chunkStart = (int)(((uint64_t)flatDim * part) / dotParts);
    int chunkEnd = (int)(((uint64_t)flatDim * (part + 1)) / dotParts);
    const XT *xrow = x + (uint64_t)t * flatDim;
    float partial = 0.0f;
    if (m == mixHc) {
        for (int k = chunkStart + threadIdx.x; k < chunkEnd; k += blockDim.x) {
            float v = Dsv4ToFloat(xrow[k]);
            partial += v * v;
        }
    } else {
        const WT *wrow = w + (uint64_t)m * flatDim;
        int pairStart = chunkStart;
        if (pairStart & 1) {
            if (threadIdx.x == 0) {
                partial += Dsv4ToFloat(xrow[pairStart]) * Dsv4ToFloat(wrow[pairStart]);
            }
            pairStart++;
        }
        for (int k = pairStart + threadIdx.x * 2; k + 1 < chunkEnd; k += blockDim.x * 2) {
            partial += Dsv4PairDot(xrow, wrow, k);
        }
        if (((chunkEnd - pairStart) & 1) && threadIdx.x == 0) {
            int k = chunkEnd - 1;
            partial += Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
        }
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dots[((uint64_t)t * dotsStride + m) * dotParts + part] = red[0];
    }
}

template <typename XT>
__global__ void DeepSeekV4HcPreSqSumKernel(const XT *x, float *dots,
                                           int tokens, int flatDim, int dotsStride) {
    extern __shared__ float red[];
    int t = blockIdx.x;
    if (t >= tokens) {
        return;
    }
    const XT *xrow = x + (uint64_t)t * flatDim;
    float partial = 0.0f;
    for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
        float v = Dsv4ToFloat(xrow[k]);
        partial += v * v;
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dots[(uint64_t)t * dotsStride + (dotsStride - 1)] = red[0];
    }
}

template <typename XT>
__global__ void DeepSeekV4HcPreFinishKernel(const XT *x, const float *dots, const float *scale,
                                            const float *base, XT *y, float *post, float *comb,
                                            int tokens, int dim, int hcMult, int sinkhornIters,
                                            float eps, float normEps, int dotsStride, int dotParts) {
    extern __shared__ float shared[];
    float *mixes = shared;
    float *pre = mixes + (2 + hcMult) * hcMult;
    float *combLocal = pre + hcMult;
    float *rowStats = combLocal + hcMult * hcMult;
    float *colStats = rowStats + hcMult;

    int t = blockIdx.x;
    int finishPart = blockIdx.y;
    int finishParts = gridDim.y;
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    const XT *xrow = x + (uint64_t)t * flatDim;

    if (threadIdx.x == 0) {
        float sqSum = 0.0f;
        uint64_t sqBase = ((uint64_t)t * dotsStride + mixHc) * dotParts;
        for (int part = 0; part < dotParts; part++) {
            sqSum += dots[sqBase + part];
        }
        rowStats[0] = rsqrtf(sqSum / flatDim + normEps);
    }
    __syncthreads();
    float rsqrt = rowStats[0];
    for (int m = threadIdx.x; m < mixHc; m += blockDim.x) {
        float dotSum = 0.0f;
        uint64_t dotBase = ((uint64_t)t * dotsStride + m) * dotParts;
        for (int part = 0; part < dotParts; part++) {
            dotSum += dots[dotBase + part];
        }
        mixes[m] = dotSum * rsqrt;
    }
    __syncthreads();

    for (int h = threadIdx.x; h < hcMult; h += blockDim.x) {
        pre[h] = DeepSeekV4Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
        if (finishPart == 0) {
            post[(uint64_t)t * hcMult + h] =
                2.0f * DeepSeekV4Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
        }
    }
    __syncthreads();

    const int combThreads = 32;
    int hcSq = hcMult * hcMult;
    if (finishPart == 0 && threadIdx.x < combThreads) {
        int lane = threadIdx.x;
        for (int idx = lane; idx < hcSq; idx += combThreads) {
            int mixIdx = idx + 2 * hcMult;
            combLocal[idx] = mixes[mixIdx] * scale[2] + base[mixIdx];
        }
        __syncwarp();

        for (int r = lane; r < hcMult; r += combThreads) {
            float rowMax = -INFINITY;
            for (int c = 0; c < hcMult; c++) {
                rowMax = fmaxf(rowMax, combLocal[r * hcMult + c]);
            }
            rowStats[r] = rowMax;
        }
        __syncwarp();
        for (int idx = lane; idx < hcSq; idx += combThreads) {
            int r = idx / hcMult;
            combLocal[idx] = expf(combLocal[idx] - rowStats[r]);
        }
        __syncwarp();
        for (int r = lane; r < hcMult; r += combThreads) {
            float rowSum = 0.0f;
            for (int c = 0; c < hcMult; c++) {
                rowSum += combLocal[r * hcMult + c];
            }
            rowStats[r] = rowSum;
        }
        __syncwarp();
        for (int idx = lane; idx < hcSq; idx += combThreads) {
            int r = idx / hcMult;
            combLocal[idx] = combLocal[idx] / rowStats[r] + eps;
        }
        __syncwarp();

        for (int c = lane; c < hcMult; c += combThreads) {
            float colSum = 0.0f;
            for (int r = 0; r < hcMult; r++) {
                colSum += combLocal[r * hcMult + c];
            }
            colStats[c] = colSum;
        }
        __syncwarp();
        for (int idx = lane; idx < hcSq; idx += combThreads) {
            int c = idx % hcMult;
            combLocal[idx] /= (colStats[c] + eps);
        }
        __syncwarp();

        for (int it = 1; it < sinkhornIters; it++) {
            for (int r = lane; r < hcMult; r += combThreads) {
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    rowSum += combLocal[r * hcMult + c];
                }
                rowStats[r] = rowSum;
            }
            __syncwarp();
            for (int idx = lane; idx < hcSq; idx += combThreads) {
                int r = idx / hcMult;
                combLocal[idx] /= (rowStats[r] + eps);
            }
            __syncwarp();

            for (int c = lane; c < hcMult; c += combThreads) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combLocal[r * hcMult + c];
                }
                colStats[c] = colSum;
            }
            __syncwarp();
            for (int idx = lane; idx < hcSq; idx += combThreads) {
                int c = idx % hcMult;
                combLocal[idx] /= (colStats[c] + eps);
            }
            __syncwarp();
        }
        for (int i = lane; i < hcSq; i += combThreads) {
            comb[(uint64_t)t * hcMult * hcMult + i] = combLocal[i];
        }
    } else {
        int yStart = (int)(((uint64_t)dim * finishPart) / finishParts);
        int yEnd = (int)(((uint64_t)dim * (finishPart + 1)) / finishParts);
        int yThreads = finishPart == 0 ? blockDim.x - combThreads : blockDim.x;
        int yThread = finishPart == 0 ? threadIdx.x - combThreads : threadIdx.x;
        XT *yrow = y + (uint64_t)t * dim;
        for (int d = yStart + yThread; d < yEnd; d += yThreads) {
            float v = 0.0f;
            for (int h = 0; h < hcMult; h++) {
                v += pre[h] * Dsv4ToFloat(xrow[(uint64_t)h * dim + d]);
            }
            yrow[d] = Dsv4FromFloat<XT>(v);
        }
    }
}

// Build one adjacent pair of the rounded HcPre output.  The two accumulators
// intentionally use the same h=0..3 statement order as the scalar fallback;
// only the memory transaction is widened to BF16x2.
struct DeepSeekV4HcPreRoundedPair {
    __nv_bfloat162 value;
    float square;
};

__device__ __forceinline__ DeepSeekV4HcPreRoundedPair
DeepSeekV4HcPreRoundedPair4x4096(
        const __nv_bfloat162 *xPairs, int pair,
        float pre0, float pre1, float pre2, float pre3) {
    constexpr int pairCount = 4096 / 2;
    __nv_bfloat162 x0 = xPairs[pair];
    __nv_bfloat162 x1 = xPairs[pairCount + pair];
    __nv_bfloat162 x2 = xPairs[2 * pairCount + pair];
    __nv_bfloat162 x3 = xPairs[3 * pairCount + pair];

    float lo = 0.0f;
    lo += pre0 * Dsv4ToFloat(x0.x);
    lo += pre1 * Dsv4ToFloat(x1.x);
    lo += pre2 * Dsv4ToFloat(x2.x);
    lo += pre3 * Dsv4ToFloat(x3.x);
    float hi = 0.0f;
    hi += pre0 * Dsv4ToFloat(x0.y);
    hi += pre1 * Dsv4ToFloat(x1.y);
    hi += pre2 * Dsv4ToFloat(x2.y);
    hi += pre3 * Dsv4ToFloat(x3.y);

    __nv_bfloat162 rounded;
    rounded.x = __float2bfloat16_rn(lo);
    rounded.y = __float2bfloat16_rn(hi);
    lo = Dsv4ToFloat(rounded.x);
    hi = Dsv4ToFloat(rounded.y);
    return {rounded, lo * lo + hi * hi};
}

// Decode specialization for hc_mult=4 and hidden_size=4096.  The generic
// variant keeps the established scalar write followed by RMSNorm unchanged.
// On SM120, the optimized variant builds adjacent BF16 pairs and accumulates
// their rounded square sums in one pass.  It still launches 1024 threads so
// the 32-warps reduction tree and the latency-hiding behavior stay identical
// to the generic path.
template <int finishThreads, bool optimizedSm120>
__global__ void DeepSeekV4HcPreFinishNorm4x4096Kernel(
        const __nv_bfloat16 *x, const float *dots, const float *scale,
        const float *base, const float *normWeight, __nv_bfloat16 *normOutput,
        float *post, float *comb, int tokens, int sinkhornIters,
        float eps, float normEps, int dotsStride, int dotParts) {
    constexpr int hcMult = 4;
    constexpr int dim = 4096;
    constexpr int flatDim = hcMult * dim;
    constexpr int mixHc = (2 + hcMult) * hcMult;
    constexpr int hcSq = hcMult * hcMult;
    constexpr int pairCount = dim / 2;

    __shared__ float mixes[mixHc];
    __shared__ float pre[hcMult];
    __shared__ float combLocal[hcSq];
    __shared__ float rowStats[hcMult];
    __shared__ float colStats[hcMult];
    __shared__ float warpSums[32];
    __shared__ float normScale;
    __shared__ __nv_bfloat162 yShared[pairCount];

    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    const __nv_bfloat16 *xrow = x + (uint64_t)token * flatDim;

    if (threadIdx.x == 0) {
        float sqSum = 0.0f;
        uint64_t sqBase = ((uint64_t)token * dotsStride + mixHc) * dotParts;
        for (int part = 0; part < dotParts; part++) {
            sqSum += dots[sqBase + part];
        }
        normScale = rsqrtf(sqSum / flatDim + normEps);
    }
    __syncthreads();

    if (threadIdx.x < mixHc) {
        int m = threadIdx.x;
        float dotSum = 0.0f;
        uint64_t dotBase = ((uint64_t)token * dotsStride + m) * dotParts;
        for (int part = 0; part < dotParts; part++) {
            dotSum += dots[dotBase + part];
        }
        mixes[m] = dotSum * normScale;
    }
    __syncthreads();

    if (threadIdx.x < hcMult) {
        int h = threadIdx.x;
        pre[h] = DeepSeekV4Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
        post[(uint64_t)token * hcMult + h] =
            2.0f * DeepSeekV4Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
    }
    __syncthreads();

    // Match the established HcPre kernel's summation order exactly.  The
    // SM120 path keeps one matrix value in each of lanes 0..15 and fetches
    // the other row/column members with ordered shuffles.  This removes the
    // shared-memory round trips and roughly 80 warp barriers without changing
    // the scalar association used by greedy decode.
    if constexpr (optimizedSm120) {
        if (threadIdx.x < hcSq) {
            constexpr unsigned activeMask = 0x0000ffffu;
            int lane = threadIdx.x;
            int rowBase = lane & ~3;
            int col = lane & 3;
            int mixIdx = lane + 2 * hcMult;
            float cm = mixes[mixIdx] * scale[2] + base[mixIdx];

            float rowMax = -INFINITY;
            rowMax = fmaxf(rowMax,
                           __shfl_sync(activeMask, cm, rowBase));
            rowMax = fmaxf(rowMax,
                           __shfl_sync(activeMask, cm, rowBase + 1));
            rowMax = fmaxf(rowMax,
                           __shfl_sync(activeMask, cm, rowBase + 2));
            rowMax = fmaxf(rowMax,
                           __shfl_sync(activeMask, cm, rowBase + 3));
            cm = expf(cm - rowMax);

            float rowSum = 0.0f;
            rowSum += __shfl_sync(activeMask, cm, rowBase);
            rowSum += __shfl_sync(activeMask, cm, rowBase + 1);
            rowSum += __shfl_sync(activeMask, cm, rowBase + 2);
            rowSum += __shfl_sync(activeMask, cm, rowBase + 3);
            cm = cm / rowSum + eps;

            float colSum = 0.0f;
            colSum += __shfl_sync(activeMask, cm, col);
            colSum += __shfl_sync(activeMask, cm, col + hcMult);
            colSum += __shfl_sync(activeMask, cm, col + 2 * hcMult);
            colSum += __shfl_sync(activeMask, cm, col + 3 * hcMult);
            cm /= colSum + eps;

            for (int it = 1; it < sinkhornIters; it++) {
                rowSum = 0.0f;
                rowSum += __shfl_sync(activeMask, cm, rowBase);
                rowSum += __shfl_sync(activeMask, cm, rowBase + 1);
                rowSum += __shfl_sync(activeMask, cm, rowBase + 2);
                rowSum += __shfl_sync(activeMask, cm, rowBase + 3);
                cm /= rowSum + eps;

                colSum = 0.0f;
                colSum += __shfl_sync(activeMask, cm, col);
                colSum += __shfl_sync(activeMask, cm, col + hcMult);
                colSum += __shfl_sync(activeMask, cm, col + 2 * hcMult);
                colSum += __shfl_sync(activeMask, cm, col + 3 * hcMult);
                cm /= colSum + eps;
            }
            comb[(uint64_t)token * hcSq + lane] = cm;
        }
    } else if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        for (int idx = lane; idx < hcSq; idx += 32) {
            int mixIdx = idx + 2 * hcMult;
            combLocal[idx] = mixes[mixIdx] * scale[2] + base[mixIdx];
        }
        __syncwarp();

        for (int r = lane; r < hcMult; r += 32) {
            float rowMax = -INFINITY;
            for (int c = 0; c < hcMult; c++) {
                rowMax = fmaxf(rowMax, combLocal[r * hcMult + c]);
            }
            rowStats[r] = rowMax;
        }
        __syncwarp();
        for (int idx = lane; idx < hcSq; idx += 32) {
            int r = idx / hcMult;
            combLocal[idx] = expf(combLocal[idx] - rowStats[r]);
        }
        __syncwarp();
        for (int r = lane; r < hcMult; r += 32) {
            float rowSum = 0.0f;
            for (int c = 0; c < hcMult; c++) {
                rowSum += combLocal[r * hcMult + c];
            }
            rowStats[r] = rowSum;
        }
        __syncwarp();
        for (int idx = lane; idx < hcSq; idx += 32) {
            int r = idx / hcMult;
            combLocal[idx] = combLocal[idx] / rowStats[r] + eps;
        }
        __syncwarp();

        for (int c = lane; c < hcMult; c += 32) {
            float colSum = 0.0f;
            for (int r = 0; r < hcMult; r++) {
                colSum += combLocal[r * hcMult + c];
            }
            colStats[c] = colSum;
        }
        __syncwarp();
        for (int idx = lane; idx < hcSq; idx += 32) {
            int c = idx % hcMult;
            combLocal[idx] /= colStats[c] + eps;
        }
        __syncwarp();

        for (int it = 1; it < sinkhornIters; it++) {
            for (int r = lane; r < hcMult; r += 32) {
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    rowSum += combLocal[r * hcMult + c];
                }
                rowStats[r] = rowSum;
            }
            __syncwarp();
            for (int idx = lane; idx < hcSq; idx += 32) {
                int r = idx / hcMult;
                combLocal[idx] /= rowStats[r] + eps;
            }
            __syncwarp();

            for (int c = lane; c < hcMult; c += 32) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combLocal[r * hcMult + c];
                }
                colStats[c] = colSum;
            }
            __syncwarp();
            for (int idx = lane; idx < hcSq; idx += 32) {
                int c = idx % hcMult;
                combLocal[idx] /= colStats[c] + eps;
            }
            __syncwarp();
        }
        for (int idx = lane; idx < hcSq; idx += 32) {
            comb[(uint64_t)token * hcSq + idx] = combLocal[idx];
        }
    }

    if constexpr (optimizedSm120) {
        static_assert(finishThreads == 1024,
                      "optimized HcPre finish preserves the 32-warp reduction tree");
        int warp = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;

        const __nv_bfloat162 *xPairs =
            reinterpret_cast<const __nv_bfloat162 *>(xrow);
        float pre0 = pre[0];
        float pre1 = pre[1];
        float pre2 = pre[2];
        float pre3 = pre[3];
        int pair = threadIdx.x;
        DeepSeekV4HcPreRoundedPair rounded0 =
            DeepSeekV4HcPreRoundedPair4x4096(
                xPairs, pair, pre0, pre1, pre2, pre3);
        DeepSeekV4HcPreRoundedPair rounded1 =
            DeepSeekV4HcPreRoundedPair4x4096(
                xPairs, pair + finishThreads,
                pre0, pre1, pre2, pre3);
        float sum2 = rounded0.square + rounded1.square;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
        }
        if (lane == 0) {
            warpSums[warp] = sum2;
        }
        __syncthreads();

        if (warp == 0) {
            float value = warpSums[lane];
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffff, value, offset);
            }
            if (lane == 0) {
                normScale = rsqrtf(value / dim + normEps);
            }
        }
        __syncthreads();

        __nv_bfloat162 *outPairs = reinterpret_cast<__nv_bfloat162 *>(
            normOutput + (uint64_t)token * dim);
        const float2 *weightPairs =
            reinterpret_cast<const float2 *>(normWeight);
        float finalScale = normScale;
        DeepSeekV4HcPreRoundedPair roundedPairs[2] = {rounded0, rounded1};
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            int i = pair + item * finishThreads;
            __nv_bfloat162 rounded = roundedPairs[item].value;
            float lo = Dsv4ToFloat(rounded.x);
            float hi = Dsv4ToFloat(rounded.y);
            float2 weights = __ldg(weightPairs + i);
            __nv_bfloat162 normalized;
            normalized.x = __float2bfloat16_rn(
                lo * finalScale * weights.x);
            normalized.y = __float2bfloat16_rn(
                hi * finalScale * weights.y);
            outPairs[i] = normalized;
        }
    } else {
        // Compute each scalar with the same statement order as the established
        // HcPre finish kernel, then make the BF16 boundary visible to every
        // thread before starting the exact RMSNorm reduction.
        __nv_bfloat16 *yScalars =
            reinterpret_cast<__nv_bfloat16 *>(yShared);
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            float value = 0.0f;
#pragma unroll
            for (int h = 0; h < hcMult; h++) {
                value += pre[h] *
                         Dsv4ToFloat(xrow[(uint64_t)h * dim + d]);
            }
            yScalars[d] = Dsv4FromFloat<__nv_bfloat16>(value);
        }
        __syncthreads();

        // Use the same 1024-thread element assignment and two-level reduction
        // as FastllmRMSNormKernelInner1<1024>.
        float sum2 = 0.0f;
        for (int i = threadIdx.x; i < pairCount; i += blockDim.x) {
            __nv_bfloat162 rounded = yShared[i];
            float lo = __bfloat162float(rounded.x);
            float hi = __bfloat162float(rounded.y);
            sum2 += lo * lo + hi * hi;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
        }
        int warp = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        if (lane == 0) {
            warpSums[warp] = sum2;
        }
        __syncthreads();
        if (warp == 0) {
            int numWarps = blockDim.x >> 5;
            float value = lane < numWarps ? warpSums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffff, value, offset);
            }
            if (lane == 0) {
                normScale = rsqrtf(value / dim + normEps);
            }
        }
        __syncthreads();

        __nv_bfloat16 *out = normOutput + (uint64_t)token * dim;
        float finalScale = normScale;
        __nv_bfloat162 *outPairs = reinterpret_cast<__nv_bfloat162 *>(out);
        const float2 *weightPairs = reinterpret_cast<const float2 *>(normWeight);
        for (int i = threadIdx.x; i < pairCount; i += blockDim.x) {
            __nv_bfloat162 rounded = yShared[i];
            float lo = __bfloat162float(rounded.x);
            float hi = __bfloat162float(rounded.y);
            float2 weights = __ldg(weightPairs + i);
            __nv_bfloat162 normalized;
            normalized.x = __float2bfloat16_rn(
                lo * finalScale * weights.x);
            normalized.y = __float2bfloat16_rn(
                hi * finalScale * weights.y);
            outPairs[i] = normalized;
        }
    }
}

void DeepSeekV4LaunchHcPreFinishNorm4x4096(
        const __nv_bfloat16 *x, const float *dots, const float *scale,
        const float *base, const float *normWeight,
        __nv_bfloat16 *normOutput, float *post, float *comb,
        int tokens, int sinkhornIters, float eps, float normEps,
        int dotsStride, int dotParts) {
    bool useSm120 = FastllmCudaRuntimeArch() >= 120 &&
        std::getenv("FASTLLM_DSV4_REFERENCE_HC_PRE_FINISH") == nullptr;
    if (!useSm120) {
        constexpr int finishThreads = 1024;
        DeepSeekV4HcPreFinishNorm4x4096Kernel<finishThreads, false>
            <<<tokens, finishThreads>>>(
                x, dots, scale, base, normWeight, normOutput, post, comb,
                tokens, sinkhornIters, eps, normEps, dotsStride, dotParts);
        return;
    }

    constexpr int finishThreads = 1024;
    DeepSeekV4HcPreFinishNorm4x4096Kernel<finishThreads, true>
        <<<tokens, finishThreads>>>(
            x, dots, scale, base, normWeight, normOutput, post, comb,
            tokens, sinkhornIters, eps, normEps, dotsStride, dotParts);
}

template <typename XT>
__global__ void DeepSeekV4HcHeadFinishKernel(const XT *x, const float *dots,
                                             const float *scale, const float *base,
                                             XT *output, int tokens, int dim,
                                             int hcMult, float eps, float normEps) {
    extern __shared__ float shared[];
    float *pre = shared;
    float *rsqrtValue = pre + hcMult;
    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    int flatDim = hcMult * dim;
    int dotsStride = hcMult + 1;
    if (threadIdx.x == 0) {
        float sqSum = dots[(uint64_t)token * dotsStride + hcMult];
        rsqrtValue[0] = rsqrtf(sqSum / flatDim + normEps);
        for (int h = 0; h < hcMult; h++) {
            pre[h] = DeepSeekV4Sigmoid(
                dots[(uint64_t)token * dotsStride + h] * rsqrtValue[0] * scale[0] + base[h]) + eps;
        }
    }
    __syncthreads();
    const XT *xrow = x + (uint64_t)token * flatDim;
    XT *out = output + (uint64_t)token * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = 0.0f;
        for (int h = 0; h < hcMult; h++) {
            value += pre[h] * Dsv4ToFloat(xrow[(uint64_t)h * dim + d]);
        }
        out[d] = Dsv4FromFloat<XT>(value);
    }
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcHeadDotsKernel(const XT *x, const WT *weight,
                                           float *dots, int tokens,
                                           int flatDim, int hcMult,
                                           int dotsStride) {
    extern __shared__ double hcHeadRed[];
    int index = blockIdx.x;
    int token = index / hcMult;
    int h = index - token * hcMult;
    if (token >= tokens) {
        return;
    }
    const XT *xrow = x + (uint64_t)token * flatDim;
    const WT *wrow = weight + (uint64_t)h * flatDim;
    double sum = 0.0;
    for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
        sum += (double)Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
    }
    hcHeadRed[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            hcHeadRed[threadIdx.x] += hcHeadRed[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dots[(uint64_t)token * dotsStride + h] = (float)hcHeadRed[0];
    }
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcPreKernel(const XT *x, const WT *fn, const float *scale,
                                      const float *base, XT *y, float *post, float *comb,
                                      int tokens, int dim, int hcMult, int sinkhornIters,
                                      float eps, float normEps) {
    extern __shared__ float shared[];
    int t = blockIdx.x;
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    float *red = shared;
    float *mixes = red + blockDim.x;
    float *pre = mixes + mixHc;
    float *combLocal = pre + hcMult;

    const XT *xrow = x + (uint64_t)t * flatDim;
    float partial = 0.0f;
    for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
        float v = Dsv4ToFloat(xrow[k]);
        partial += v * v;
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float rsqrt = rsqrtf(red[0] / flatDim + normEps);

    for (int m = 0; m < mixHc; m++) {
        const WT *wrow = fn + (uint64_t)m * flatDim;
        partial = 0.0f;
        for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
            partial += Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
        }
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            mixes[m] = red[0] * rsqrt;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int h = 0; h < hcMult; h++) {
            pre[h] = DeepSeekV4Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
            post[(uint64_t)t * hcMult + h] =
                2.0f * DeepSeekV4Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
        }

        for (int r = 0; r < hcMult; r++) {
            float rowMax = -INFINITY;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                int mixIdx = idx + 2 * hcMult;
                combLocal[idx] = mixes[mixIdx] * scale[2] + base[mixIdx];
                rowMax = fmaxf(rowMax, combLocal[idx]);
            }
            float rowSum = 0.0f;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                float v = expf(combLocal[idx] - rowMax);
                combLocal[idx] = v;
                rowSum += v;
            }
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                combLocal[idx] = combLocal[idx] / rowSum + eps;
            }
        }
        for (int c = 0; c < hcMult; c++) {
            float colSum = 0.0f;
            for (int r = 0; r < hcMult; r++) {
                colSum += combLocal[r * hcMult + c];
            }
            for (int r = 0; r < hcMult; r++) {
                combLocal[r * hcMult + c] /= (colSum + eps);
            }
        }
        for (int it = 1; it < sinkhornIters; it++) {
            for (int r = 0; r < hcMult; r++) {
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    rowSum += combLocal[r * hcMult + c];
                }
                for (int c = 0; c < hcMult; c++) {
                    combLocal[r * hcMult + c] /= (rowSum + eps);
                }
            }
            for (int c = 0; c < hcMult; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combLocal[r * hcMult + c];
                }
                for (int r = 0; r < hcMult; r++) {
                    combLocal[r * hcMult + c] /= (colSum + eps);
                }
            }
        }
        for (int i = 0; i < hcMult * hcMult; i++) {
            comb[(uint64_t)t * hcMult * hcMult + i] = combLocal[i];
        }
    }
    __syncthreads();

    XT *yrow = y + (uint64_t)t * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int h = 0; h < hcMult; h++) {
            v += pre[h] * Dsv4ToFloat(xrow[(uint64_t)h * dim + d]);
        }
        yrow[d] = Dsv4FromFloat<XT>(v);
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedKernel(const QT *q, const float *windowKV,
                                                            const CT *compressedKV, const float *sink,
                                                            float *output, int bsz, int heads, int dim,
                                                            int windowSize, int startPos, int compressedCount,
                                                            float softmaxScale) {
    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    const QT *qrow = q + ((uint64_t)b * heads + h) * dim;
    float *orow = output + ((uint64_t)b * heads + h) * dim;

    int idxCount = 0;
    int pos = startPos % windowSize;
    int idxs[512];
    if (startPos >= windowSize - 1) {
        for (int i = pos + 1; i < windowSize; i++) {
            idxs[idxCount++] = i;
        }
        for (int i = 0; i <= pos; i++) {
            idxs[idxCount++] = i;
        }
    } else {
        for (int i = 0; i <= startPos; i++) {
            idxs[idxCount++] = i;
        }
    }
    for (int i = 0; i < compressedCount; i++) {
        idxs[idxCount++] = windowSize + i;
    }

    double scores[512];
    float mx = -INFINITY;
    for (int k = 0; k < idxCount; k++) {
        const float *windowRow = nullptr;
        const CT *compressedRow = nullptr;
        int idx = idxs[k];
        if (idx < windowSize) {
            windowRow = windowKV + ((uint64_t)b * windowSize + idx) * dim;
        } else {
            compressedRow = compressedKV + ((uint64_t)b * compressedCount + (idx - windowSize)) * dim;
        }
        double dot = 0.0;
        for (int d = 0; d < dim; d++) {
            float kv = windowRow != nullptr ? windowRow[d] : Dsv4ToFloat(compressedRow[d]);
            dot += (double)Dsv4ToFloat(qrow[d]) * kv;
        }
        float score = (float)dot * softmaxScale;
        scores[k] = (double)score;
        mx = fmaxf(mx, score);
    }

    float safeMx = isfinite(mx) ? mx : 0.0f;
    double denom = exp((double)sink[h] - safeMx);
    for (int k = 0; k < idxCount; k++) {
        denom += exp(scores[k] - safeMx);
    }

    for (int d = 0; d < dim; d++) {
        double v = 0.0;
        for (int k = 0; k < idxCount; k++) {
            int idx = idxs[k];
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            double w = exp(scores[k] - safeMx) / fmax(denom, 1e-30);
            v += w * kv;
        }
        orow[d] = (float)v;
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedBlockKernel(const QT *q, const float *windowKV,
                                                                 const CT *compressedKV, const float *sink,
                                                                 float *output, int bsz, int heads, int dim,
                                                                 int windowSize, int startPos, int compressedCount,
                                                                 int keyCapacity, float softmaxScale) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *red = shared + keyCapacity;

    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    const QT *qrow = q + ((uint64_t)b * heads + h) * dim;
    float *orow = output + ((uint64_t)b * heads + h) * dim;

    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int idxCount = liveWindow + compressedCount;
    if (idxCount <= 0 || idxCount > keyCapacity || idxCount > kDeepSeekV4SparseDecodeMaxKeys) {
        return;
    }
    int pos = startPos % windowSize;

    __shared__ float mxShared;
    __shared__ float denomShared;
    if (threadIdx.x == 0) {
        mxShared = -INFINITY;
    }
    __syncthreads();

    for (int k = 0; k < idxCount; k++) {
        int idx;
        if (k < liveWindow) {
            idx = (startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k;
        } else {
            idx = windowSize + (k - liveWindow);
        }

        float partial = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            partial += Dsv4ToFloat(qrow[d]) * kv;
        }

        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = red[0] * softmaxScale;
            scores[k] = score;
            mxShared = fmaxf(mxShared, score);
        }
        __syncthreads();
    }

    float safeMx = isfinite(mxShared) ? mxShared : 0.0f;
    float partialDenom = 0.0f;
    for (int k = threadIdx.x; k < idxCount; k += blockDim.x) {
        partialDenom += expf(scores[k] - safeMx);
    }
    red[threadIdx.x] = partialDenom;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        denomShared = red[0] + expf(sink[h] - safeMx);
    }
    __syncthreads();

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < idxCount; k++) {
            int idx;
            if (k < liveWindow) {
                idx = (startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k;
            } else {
                idx = windowSize + (k - liveWindow);
            }
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            float w = expf(scores[k] - safeMx) / fmaxf(denomShared, 1e-30f);
            v += w * kv;
        }
        orow[d] = v;
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedOnlineKernel(const QT *q, const float *windowKV,
                                                                  const CT *compressedKV, const float *sink,
                                                                  float *output, int bsz, int seqlen,
                                                                  int heads, int dim,
                                                                  int windowSize, int startPos, int compressedCount,
                                                                  int compressedStride,
                                                                  const int32_t *compressedIndices,
                                                                  const int *compressedLengths,
                                                                  float softmaxScale,
                                                                  const int32_t *decodeMeta,
                                                                  int compressRatio) {
    constexpr int kMaxDimsPerThread = 4;
    __shared__ float red[256];
    __shared__ float mxShared;
    __shared__ float denomShared;
    __shared__ float alphaShared;
    __shared__ float betaShared;

    int bsh = blockIdx.x;
    int h = bsh % heads;
    int bs = bsh / heads;
    int s = bs % seqlen;
    int b = bs / seqlen;
    const QT *qrow = q + (((uint64_t)b * seqlen + s) * heads + h) * dim;
    float *orow = output + (((uint64_t)b * seqlen + s) * heads + h) * dim;

    if (decodeMeta != nullptr) {
        startPos = decodeMeta[0] + s;
        compressedCount = compressRatio > 0 ? (startPos + 1) / compressRatio : 0;
    } else {
        startPos += s;
    }
    int selectedCompressedCount = compressedIndices != nullptr &&
                                  compressedLengths != nullptr ?
                                  compressedLengths[s] : compressedCount;
    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int idxCount = liveWindow + selectedCompressedCount;
    if (idxCount <= 0 || idxCount > kDeepSeekV4SparseDecodeMaxKeys ||
        dim > blockDim.x * kMaxDimsPerThread) {
        return;
    }
    int pos = startPos % windowSize;

    int localDims = 0;
    int localOffsets[kMaxDimsPerThread];
    float localAcc[kMaxDimsPerThread];
    float localKV[kMaxDimsPerThread];
    for (int d = threadIdx.x; d < dim && localDims < kMaxDimsPerThread; d += blockDim.x) {
        localOffsets[localDims] = d;
        localAcc[localDims] = 0.0f;
        localKV[localDims] = 0.0f;
        localDims++;
    }

    if (threadIdx.x == 0) {
        mxShared = -INFINITY;
        denomShared = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < idxCount; k++) {
        int idx = k < liveWindow
                      ? ((startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k)
                      : (windowSize + (compressedIndices != nullptr ?
                          compressedIndices[(uint64_t)s *
                              kDeepSeekV4IndexerTopK + (k - liveWindow)] :
                          (k - liveWindow)));

        float partial = 0.0f;
        for (int i = 0; i < localDims; i++) {
            int d = localOffsets[i];
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedStride +
                    (idx - windowSize)) * dim + d]);
            }
            localKV[i] = kv;
            partial += Dsv4ToFloat(qrow[d]) * kv;
        }

        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = red[0] * softmaxScale;
            float oldMx = mxShared;
            float newMx = fmaxf(oldMx, score);
            float alpha = isfinite(oldMx) ? expf(oldMx - newMx) : 0.0f;
            float beta = expf(score - newMx);
            mxShared = newMx;
            denomShared = denomShared * alpha + beta;
            alphaShared = alpha;
            betaShared = beta;
        }
        __syncthreads();
        for (int i = 0; i < localDims; i++) {
            localAcc[i] = localAcc[i] * alphaShared + betaShared * localKV[i];
        }
    }

    float finalDenom = denomShared + expf(sink[h] - mxShared);
    finalDenom = fmaxf(finalDenom, 1e-30f);
    for (int i = 0; i < localDims; i++) {
        orow[localOffsets[i]] = localAcc[i] / finalDenom;
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedBatchOnlineKernel(
        const void * const *qPtrs, const float * const *windowPtrs,
        const void * const *compressedPtrs, const float *sink,
        const int *startPositions, const int *compressedCounts,
        float *output, int batch, int heads, int dim, int windowSize,
        float softmaxScale) {
    constexpr int kMaxDimsPerThread = 4;
    __shared__ float red[256];
    __shared__ float mxShared;
    __shared__ float denomShared;
    __shared__ float alphaShared;
    __shared__ float betaShared;

    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch) {
        return;
    }

    const QT *qBase = reinterpret_cast<const QT *>(qPtrs[b]);
    const float *windowKV = windowPtrs[b];
    const CT *compressedKV = reinterpret_cast<const CT *>(compressedPtrs[b]);
    int startPos = startPositions[b];
    int compressedCount = compressedCounts[b];
    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int idxCount = liveWindow + compressedCount;
    if (qBase == nullptr || windowKV == nullptr || startPos < 0 ||
        idxCount <= 0 || idxCount > kDeepSeekV4SparseDecodeMaxKeys ||
        dim > blockDim.x * kMaxDimsPerThread ||
        (compressedCount > 0 && compressedKV == nullptr)) {
        return;
    }

    int pos = startPos % windowSize;
    const QT *qrow = qBase + (uint64_t)h * dim;
    float *orow = output + (uint64_t)bh * dim;

    int localDims = 0;
    int localOffsets[kMaxDimsPerThread];
    float localAcc[kMaxDimsPerThread];
    float localKV[kMaxDimsPerThread];
    for (int d = threadIdx.x; d < dim && localDims < kMaxDimsPerThread; d += blockDim.x) {
        localOffsets[localDims] = d;
        localAcc[localDims] = 0.0f;
        localKV[localDims] = 0.0f;
        localDims++;
    }

    if (threadIdx.x == 0) {
        mxShared = -INFINITY;
        denomShared = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < idxCount; k++) {
        int idx = k < liveWindow
                      ? ((startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k)
                      : (windowSize + (k - liveWindow));

        float partial = 0.0f;
        for (int i = 0; i < localDims; i++) {
            int d = localOffsets[i];
            float kv;
            if (idx < windowSize) {
                kv = windowKV[(uint64_t)idx * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)(idx - windowSize)) * dim + d]);
            }
            localKV[i] = kv;
            partial += Dsv4ToFloat(qrow[d]) * kv;
        }

        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = red[0] * softmaxScale;
            float oldMx = mxShared;
            float newMx = fmaxf(oldMx, score);
            float alpha = isfinite(oldMx) ? expf(oldMx - newMx) : 0.0f;
            float beta = expf(score - newMx);
            mxShared = newMx;
            denomShared = denomShared * alpha + beta;
            alphaShared = alpha;
            betaShared = beta;
        }
        __syncthreads();
        for (int i = 0; i < localDims; i++) {
            localAcc[i] = localAcc[i] * alphaShared + betaShared * localKV[i];
        }
    }

    float finalDenom = denomShared + expf(sink[h] - mxShared);
    finalDenom = fmaxf(finalDenom, 1e-30f);
    for (int i = 0; i < localDims; i++) {
        orow[localOffsets[i]] = localAcc[i] / finalDenom;
    }
}

template <typename QT>
__global__ void DeepSeekV4SparseDecodeConvertQKernel(const QT *q, float *qFloat, uint64_t total) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        qFloat[idx] = Dsv4ToFloat(q[idx]);
    }
}

template <typename CT>
__global__ void DeepSeekV4SparseDecodeGatherKVKernel(const float *windowKV, const CT *compressedKV,
                                                     float *kvFloat, int bsz, int dim, int windowSize,
                                                     int startPos, int compressedCount, int liveWindow,
                                                     int keyCount) {
    uint64_t total = (uint64_t)bsz * keyCount * dim;
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    int d = (int)(linear % dim);
    uint64_t tmp = linear / dim;
    int k = (int)(tmp % keyCount);
    int b = (int)(tmp / keyCount);
    int pos = startPos % windowSize;

    float kv;
    if (k < liveWindow) {
        int idx = (startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k;
        kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
    } else {
        int ck = k - liveWindow;
        kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + ck) * dim + d]);
    }
    kvFloat[((uint64_t)b * keyCount + k) * dim + d] = kv;
}

template <typename KT>
__global__ void DeepSeekV4SparsePrefillCastCompressedKVKernel(const KT *kv, float *kvFloat,
                                                              int bsz, int kvLen, int compressedStart,
                                                              int compressedCount, int dim) {
    uint64_t total = (uint64_t)bsz * compressedCount * dim;
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    int d = (int)(linear % dim);
    uint64_t tmp = linear / dim;
    int k = (int)(tmp % compressedCount);
    int b = (int)(tmp / compressedCount);
    kvFloat[((uint64_t)b * compressedCount + k) * dim + d] =
        Dsv4ToFloat(kv[((uint64_t)b * kvLen + compressedStart + k) * dim + d]);
}

__global__ void DeepSeekV4SparseDecodeSoftmaxSinkKernel(float *scores, const float *sink,
                                                        int bsz, int heads, int keyCount) {
    extern __shared__ float red[];
    int row = blockIdx.x;
    int h = row % heads;
    float *rowScores = scores + (uint64_t)row * keyCount;

    float mx = -INFINITY;
    for (int k = threadIdx.x; k < keyCount; k += blockDim.x) {
        mx = fmaxf(mx, rowScores[k]);
    }
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float safeMx = isfinite(red[0]) ? red[0] : 0.0f;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < keyCount; k += blockDim.x) {
        sum += expf(rowScores[k] - safeMx);
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denom = red[0] + expf(sink[h] - safeMx);
    denom = fmaxf(denom, 1e-30f);

    for (int k = threadIdx.x; k < keyCount; k += blockDim.x) {
        rowScores[k] = expf(rowScores[k] - safeMx) / denom;
    }
}

template <typename QT, typename KT>
__global__ void DeepSeekV4SparseAttentionPrefillBlockKernel(const QT *q, const KT *kv,
                                                            const float *sink, float *output,
                                                            int bsz, int seqlen, int heads, int dim,
                                                            int kvLen, int windowSize, int compressRatio,
                                                            int startPos, int prefixLen, float softmaxScale,
                                                            int rowOffset, bool nonCausalBlock) {
    extern __shared__ float scores[];

    int localRow = blockIdx.x;
    int row = rowOffset + localRow;
    int h = row % heads;
    int tmp = row / heads;
    int s = tmp % seqlen;
    int b = tmp / seqlen;
    if (b >= bsz) {
        return;
    }

    int realPrefixLen = max(0, min(prefixLen, kvLen - seqlen));
    int compressedStart = realPrefixLen + seqlen;
    int compressedCount = max(0, kvLen - compressedStart);
    int liveWindow = nonCausalBlock ? realPrefixLen + seqlen :
        min(windowSize, realPrefixLen + s + 1);
    int beginPos = startPos + s - liveWindow + 1;
    int prefixStartPos = startPos - realPrefixLen;
    int liveCompressed = 0;
    if (compressRatio > 0 && compressedCount > 0) {
        liveCompressed = min(compressedCount, (startPos + s + 1) / compressRatio);
    }
    int idxCount = liveWindow + liveCompressed;
    if (idxCount <= 0 || idxCount > kDeepSeekV4SparsePrefillMaxKeys) {
        return;
    }

    __shared__ float mxShared;
    __shared__ float denomShared;
    const QT *qrow = q + (((uint64_t)b * seqlen + s) * heads + h) * dim;

    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    for (int base = 0; base < idxCount; base += warps) {
        int k = base + warpId;
        float dot = 0.0f;
        if (k < idxCount) {
            int idx;
            if (k < liveWindow) {
                if (nonCausalBlock) {
                    idx = k;
                } else {
                    int pos = beginPos + k;
                    idx = (pos < startPos) ? (pos - prefixStartPos) :
                        (realPrefixLen + pos - startPos);
                }
            } else {
                idx = compressedStart + k - liveWindow;
            }
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            for (int d = lane; d < dim; d += 32) {
                dot += Dsv4ToFloat(qrow[d]) * Dsv4ToFloat(kvrow[d]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0 && k < idxCount) {
            scores[k] = dot * softmaxScale;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int k = 0; k < idxCount; k++) {
            mx = fmaxf(mx, scores[k]);
        }
        mxShared = isfinite(mx) ? mx : 0.0f;
        float denom = expf(sink[h] - mxShared);
        for (int k = 0; k < idxCount; k++) {
            denom += expf(scores[k] - mxShared);
        }
        denomShared = fmaxf(denom, 1e-30f);
    }
    __syncthreads();

    for (int k = threadIdx.x; k < idxCount; k += blockDim.x) {
        scores[k] = expf(scores[k] - mxShared) / denomShared;
    }
    __syncthreads();

    float *orow = output + (uint64_t)localRow * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < idxCount; k++) {
            int idx;
            if (k < liveWindow) {
                if (nonCausalBlock) {
                    idx = k;
                } else {
                    int pos = beginPos + k;
                    idx = (pos < startPos) ? (pos - prefixStartPos) :
                        (realPrefixLen + pos - startPos);
                }
            } else {
                idx = compressedStart + k - liveWindow;
            }
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            v += scores[k] * Dsv4ToFloat(kvrow[d]);
        }
        orow[d] = v;
    }
}

template <typename QT, typename KT>
__global__ void DeepSeekV4SparseAttentionPrefillCublasCompressedKernel(
        const QT *q, const KT *kv, const float *sink, float *compressedScores, float *output,
        int bsz, int seqlen, int heads, int dim, int kvLen, int windowSize, int compressRatio,
        int startPos, int prefixLen, int compressedStart, int compressedCount, float softmaxScale,
        int rowOffset) {
    extern __shared__ float localScores[];

    int localRow = blockIdx.x;
    int row = rowOffset + localRow;
    int h = row % heads;
    int tmp = row / heads;
    int s = tmp % seqlen;
    int b = tmp / seqlen;
    if (b >= bsz) {
        return;
    }

    int realPrefixLen = max(0, min(prefixLen, kvLen - seqlen));
    int liveWindow = min(windowSize, realPrefixLen + s + 1);
    int beginPos = startPos + s - liveWindow + 1;
    int prefixStartPos = startPos - realPrefixLen;
    int liveCompressed = 0;
    if (compressRatio > 0 && compressedCount > 0) {
        liveCompressed = min(compressedCount, (startPos + s + 1) / compressRatio);
    }
    if (liveWindow + liveCompressed <= 0) {
        return;
    }

    __shared__ float mxShared;
    __shared__ float denomShared;
    const QT *qrow = q + (((uint64_t)b * seqlen + s) * heads + h) * dim;
    float *crow = compressedScores + (uint64_t)localRow * compressedCount;

    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    for (int base = 0; base < liveWindow; base += warps) {
        int k = base + warpId;
        float dot = 0.0f;
        if (k < liveWindow) {
            int pos = beginPos + k;
            int idx = (pos < startPos) ? (pos - prefixStartPos) : (realPrefixLen + pos - startPos);
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            for (int d = lane; d < dim; d += 32) {
                dot += Dsv4ToFloat(qrow[d]) * Dsv4ToFloat(kvrow[d]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0 && k < liveWindow) {
            localScores[k] = dot * softmaxScale;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int k = 0; k < liveWindow; k++) {
            mx = fmaxf(mx, localScores[k]);
        }
        for (int k = 0; k < liveCompressed; k++) {
            mx = fmaxf(mx, crow[k]);
        }
        mxShared = isfinite(mx) ? mx : 0.0f;
        float denom = expf(sink[h] - mxShared);
        for (int k = 0; k < liveWindow; k++) {
            denom += expf(localScores[k] - mxShared);
        }
        for (int k = 0; k < liveCompressed; k++) {
            denom += expf(crow[k] - mxShared);
        }
        denomShared = fmaxf(denom, 1e-30f);
    }
    __syncthreads();

    for (int k = threadIdx.x; k < liveWindow; k += blockDim.x) {
        localScores[k] = expf(localScores[k] - mxShared) / denomShared;
    }
    for (int k = threadIdx.x; k < liveCompressed; k += blockDim.x) {
        crow[k] = expf(crow[k] - mxShared) / denomShared;
    }
    for (int k = liveCompressed + threadIdx.x; k < compressedCount; k += blockDim.x) {
        crow[k] = 0.0f;
    }
    __syncthreads();

    float *orow = output + (uint64_t)localRow * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < liveWindow; k++) {
            int pos = beginPos + k;
            int idx = (pos < startPos) ? (pos - prefixStartPos) : (realPrefixLen + pos - startPos);
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            v += localScores[k] * Dsv4ToFloat(kvrow[d]);
        }
        orow[d] = v;
    }
}

__global__ void DeepSeekV4SparsePrefillCublasLocalSoftmaxKernel(
        float *localScores, float *compressedScores,
        __nv_bfloat16 *localScoresBf16, __nv_bfloat16 *compressedScoresBf16,
        const float *sink,
        int bsz, int seqlen, int heads, int rawStart, int rawKeyCount,
        int compressedCount, int windowSize, int compressRatio, int startPos,
        int prefixLen, int kvLen, int rowOffset, int rows) {
    extern __shared__ float red[];
    int localRow = blockIdx.x;
    if (localRow >= rows) {
        return;
    }

    int row = rowOffset + localRow;
    int h = row % heads;
    int tmp = row / heads;
    int s = tmp % seqlen;
    int b = tmp / seqlen;
    if (b >= bsz) {
        return;
    }

    int realPrefixLen = max(0, min(prefixLen, kvLen - seqlen));
    int liveWindow = min(windowSize, realPrefixLen + s + 1);
    int highRaw = realPrefixLen + s;
    int lowRaw = highRaw - liveWindow + 1;
    int liveCompressed = 0;
    if (compressRatio > 0 && compressedCount > 0) {
        liveCompressed = min(compressedCount, (startPos + s + 1) / compressRatio);
    }

    float *lrow = localScores + (uint64_t)localRow * rawKeyCount;
    float *crow = compressedScores == nullptr ? nullptr : compressedScores + (uint64_t)localRow * compressedCount;
    __nv_bfloat16 *lbrow = localScoresBf16 == nullptr ? nullptr :
        localScoresBf16 + (uint64_t)localRow * rawKeyCount;
    __nv_bfloat16 *cbrow = compressedScoresBf16 == nullptr ? nullptr :
        compressedScoresBf16 + (uint64_t)localRow * compressedCount;

    float mx = -INFINITY;
    for (int k = threadIdx.x; k < rawKeyCount; k += blockDim.x) {
        int rawIdx = rawStart + k;
        bool visible = rawIdx >= lowRaw && rawIdx <= highRaw;
        float v = visible ? lrow[k] : -INFINITY;
        lrow[k] = v;
        mx = fmaxf(mx, v);
    }
    if (crow != nullptr) {
        for (int k = threadIdx.x; k < compressedCount; k += blockDim.x) {
            bool visible = k < liveCompressed;
            float v = visible ? crow[k] : -INFINITY;
            crow[k] = v;
            mx = fmaxf(mx, v);
        }
    }
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float safeMx = isfinite(red[0]) ? red[0] : 0.0f;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < rawKeyCount; k += blockDim.x) {
        if (isfinite(lrow[k])) {
            sum += expf(lrow[k] - safeMx);
        }
    }
    if (crow != nullptr) {
        for (int k = threadIdx.x; k < compressedCount; k += blockDim.x) {
            if (isfinite(crow[k])) {
                sum += expf(crow[k] - safeMx);
            }
        }
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denom = fmaxf(red[0] + expf(sink[h] - safeMx), 1e-30f);

    for (int k = threadIdx.x; k < rawKeyCount; k += blockDim.x) {
        float p = isfinite(lrow[k]) ? expf(lrow[k] - safeMx) / denom : 0.0f;
        lrow[k] = p;
        if (lbrow != nullptr) {
            lbrow[k] = __float2bfloat16_rn(p);
        }
    }
    if (crow != nullptr) {
        for (int k = threadIdx.x; k < compressedCount; k += blockDim.x) {
            float p = isfinite(crow[k]) ? expf(crow[k] - safeMx) / denom : 0.0f;
            crow[k] = p;
            if (cbrow != nullptr) {
                cbrow[k] = __float2bfloat16_rn(p);
            }
        }
    }
}

__global__ void DeepSeekV4SparsePrefillRotaryCastKernel(const float *input, __nv_bfloat16 *output,
                                                        int rows, int rowOffset,
                                                        int seqlen, int heads, int dim,
                                                        int ropeDim, float ropeBase, int startPos,
                                                        int originalSeqLen, float ropeFactor,
                                                        int betaFast, int betaSlow) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int globalRow = rowOffset + row;
    int s = (globalRow / heads) % seqlen;
    int pos = startPos + s;
    const float *src = input + (uint64_t)row * dim;
    __nv_bfloat16 *dst = output + (uint64_t)globalRow * dim;
    int off = dim - ropeDim;
    for (int d = 0; d < off; d++) {
        dst[d] = __float2bfloat16_rn(src[d]);
    }
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = -sinf(ang);
        float a = src[off + i];
        float b = src[off + i + 1];
        dst[off + i] = __float2bfloat16_rn(a * c - b * sn);
        dst[off + i + 1] = __float2bfloat16_rn(a * sn + b * c);
    }
}

__global__ void DeepSeekV4SparseDecodeRotaryCastKernel(const float *input, __nv_bfloat16 *output,
                                                       int rows, int rowOffset,
                                                       int seqlen, int heads,
                                                       int dim, int ropeDim,
                                                       float ropeBase, int startPos,
                                                       int originalSeqLen, float ropeFactor,
                                                       int betaFast, int betaSlow,
                                                       const int32_t *decodeMeta) {
    int off = dim - ropeDim;
    int ropePairs = ropeDim >> 1;
    int workPerRow = off + ropePairs;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)rows * workPerRow;
    if (idx >= total) {
        return;
    }
    int item = idx % workPerRow;
    int row = idx / workPerRow;
    int globalRow = rowOffset + row;
    const float *src = input + (uint64_t)row * dim;
    __nv_bfloat16 *dst = output + (uint64_t)globalRow * dim;
    if (item < off) {
        dst[item] = __float2bfloat16_rn(src[item]);
        return;
    }
    int i = (item - off) << 1;
    float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    int s = (globalRow / heads) % seqlen;
    int dynamicStartPos =
        (decodeMeta == nullptr ? startPos : decodeMeta[0]) + s;
    float ang = dynamicStartPos * inv;
    float c = cosf(ang);
    float sn = -sinf(ang);
    float a = src[off + i];
    float b = src[off + i + 1];
    dst[off + i] = __float2bfloat16_rn(a * c - b * sn);
    dst[off + i + 1] = __float2bfloat16_rn(a * sn + b * c);
}

__global__ void DeepSeekV4SparseDecodeRotaryCastBatchKernel(const float *input, __nv_bfloat16 *output,
                                                            const int *startPositions,
                                                            int rows, int heads, int dim, int ropeDim,
                                                            float ropeBase, int originalSeqLen,
                                                            float ropeFactor, int betaFast,
                                                            int betaSlow) {
    int off = dim - ropeDim;
    int ropePairs = ropeDim >> 1;
    int workPerRow = off + ropePairs;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)rows * workPerRow;
    if (idx >= total) {
        return;
    }
    int item = idx % workPerRow;
    int row = idx / workPerRow;
    int b = row / heads;
    int startPos = startPositions[b];
    const float *src = input + (uint64_t)row * dim;
    __nv_bfloat16 *dst = output + (uint64_t)row * dim;
    if (item < off) {
        dst[item] = __float2bfloat16_rn(src[item]);
        return;
    }
    int i = (item - off) << 1;
    float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    float ang = startPos * inv;
    float c = cosf(ang);
    float sn = -sinf(ang);
    float a = src[off + i];
    float bval = src[off + i + 1];
    dst[off + i] = __float2bfloat16_rn(a * c - bval * sn);
    dst[off + i + 1] = __float2bfloat16_rn(a * sn + bval * c);
}

template <typename XT, typename RT, typename OT>
__global__ void DeepSeekV4HcPostKernel(const XT *x, const RT *residual, const float *post,
                                       const float *comb, OT *output, int tokens,
                                       int hcMult, int dim) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)tokens * hcMult * dim;
    if (idx >= total) {
        return;
    }

    int d = idx % dim;
    uint64_t tmp = idx / dim;
    int target = tmp % hcMult;
    int t = tmp / hcMult;

    double v = (double)post[(uint64_t)t * hcMult + target] * Dsv4ToFloat(x[(uint64_t)t * dim + d]);
    const float *combRow = comb + (uint64_t)t * hcMult * hcMult;
    const RT *resRow = residual + (uint64_t)t * hcMult * dim;
    for (int src = 0; src < hcMult; src++) {
        v += (double)combRow[src * hcMult + target] * Dsv4ToFloat(resRow[(uint64_t)src * dim + d]);
    }
    output[idx] = Dsv4FromFloat<OT>((float)v);
}

template <typename XT, typename RT, typename OT>
__global__ void DeepSeekV4HcPost4Kernel(const XT *x, const RT *residual, const float *post,
                                        const float *comb, OT *output, int tokens, int dim) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)tokens * dim;
    if (idx >= total) {
        return;
    }

    int d = idx % dim;
    int t = idx / dim;
    const RT *resRow = residual + (uint64_t)t * 4 * dim;
    double r0 = (double)Dsv4ToFloat(resRow[d]);
    double r1 = (double)Dsv4ToFloat(resRow[(uint64_t)dim + d]);
    double r2 = (double)Dsv4ToFloat(resRow[(uint64_t)2 * dim + d]);
    double r3 = (double)Dsv4ToFloat(resRow[(uint64_t)3 * dim + d]);
    double xv = (double)Dsv4ToFloat(x[(uint64_t)t * dim + d]);
    const float *postRow = post + (uint64_t)t * 4;
    const float *combRow = comb + (uint64_t)t * 16;
    OT *outRow = output + (uint64_t)t * 4 * dim + d;

    double v0 = (double)postRow[0] * xv +
                (double)combRow[0] * r0 + (double)combRow[4] * r1 +
                (double)combRow[8] * r2 + (double)combRow[12] * r3;
    double v1 = (double)postRow[1] * xv +
                (double)combRow[1] * r0 + (double)combRow[5] * r1 +
                (double)combRow[9] * r2 + (double)combRow[13] * r3;
    double v2 = (double)postRow[2] * xv +
                (double)combRow[2] * r0 + (double)combRow[6] * r1 +
                (double)combRow[10] * r2 + (double)combRow[14] * r3;
    double v3 = (double)postRow[3] * xv +
                (double)combRow[3] * r0 + (double)combRow[7] * r1 +
                (double)combRow[11] * r2 + (double)combRow[15] * r3;
    outRow[0] = Dsv4FromFloat<OT>((float)v0);
    outRow[(uint64_t)dim] = Dsv4FromFloat<OT>((float)v1);
    outRow[(uint64_t)2 * dim] = Dsv4FromFloat<OT>((float)v2);
    outRow[(uint64_t)3 * dim] = Dsv4FromFloat<OT>((float)v3);
}

// SM120 DSpark-7 specialization matching vLLM's small-FMA TileLang schedule:
// eight target rows, three output projections per CTA and four hidden-dimension
// splits.  A CTA computes all four new residual streams together, so post/comb
// inputs and the layer output are reused before accumulating the three
// projections.  Projections and the pre-normalization square sum deliberately
// consume the float transition; only residual_out is rounded to BF16.  This is
// the arithmetic boundary used by vLLM's mhc_fused_tilelang kernel.
//
// The established split-K16 kernel below remains the generic, bit-reproducible
// fallback for lower architectures, non-DSpark decode and explicit opt-out.
__global__ __launch_bounds__(256, 1)
void DeepSeekV4HcPostPreDots4x4096Sm120Kernel(
        const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
        const float *post, const float *comb, const float *nextFn,
        float *dots, __nv_bfloat16 *residualOutput, int tokens) {
    constexpr int hcMult = 4;
    constexpr int dim = 4096;
    constexpr int flatDim = hcMult * dim;
    constexpr int mixHc = (2 + hcMult) * hcMult;
    constexpr int dotsStride = mixHc + 1;
    constexpr int tileN = 3;
    constexpr int splitK = 4;
    constexpr int hiddenPerSplit = dim / splitK;
    constexpr int warps = 256 / 32;

    __shared__ float sharedPost[hcMult];
    __shared__ float sharedComb[hcMult * hcMult];
    __shared__ float warpSums[warps][tileN + 1];

    const int tile = blockIdx.x;
    const int part = blockIdx.y;
    const int token = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (token >= tokens || tile >= mixHc / tileN || part >= splitK) {
        return;
    }

    const __nv_bfloat16 *xRow = x + (uint64_t)token * dim;
    const __nv_bfloat16 *residualRow =
        residual + (uint64_t)token * flatDim;
    __nv_bfloat16 *outputRow =
        residualOutput + (uint64_t)token * flatDim;
    const float *postRow = post + (uint64_t)token * hcMult;
    const float *combRow = comb + (uint64_t)token * hcMult * hcMult;
    if (tid < hcMult) {
        sharedPost[tid] = postRow[tid];
    }
    if (tid < hcMult * hcMult) {
        sharedComb[tid] = combRow[tid];
    }
    __syncthreads();

    const int n0 = tile * tileN;
    const float *weight0 = nextFn + (uint64_t)n0 * flatDim;
    const float *weight1 = weight0 + flatDim;
    const float *weight2 = weight1 + flatDim;
    const int hiddenStart = part * hiddenPerSplit;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float square = 0.0f;

#pragma unroll
    for (int iteration = 0; iteration < hiddenPerSplit / 256;
         ++iteration) {
        const int d = hiddenStart + iteration * 256 + tid;
        const float xv = __bfloat162float(xRow[d]);
        const float r0 = __bfloat162float(residualRow[d]);
        const float r1 = __bfloat162float(residualRow[dim + d]);
        const float r2 = __bfloat162float(residualRow[2 * dim + d]);
        const float r3 = __bfloat162float(residualRow[3 * dim + d]);

        float newR0 = sharedPost[0] * xv;
        newR0 += sharedComb[0] * r0;
        newR0 += sharedComb[4] * r1;
        newR0 += sharedComb[8] * r2;
        newR0 += sharedComb[12] * r3;
        float newR1 = sharedPost[1] * xv;
        newR1 += sharedComb[1] * r0;
        newR1 += sharedComb[5] * r1;
        newR1 += sharedComb[9] * r2;
        newR1 += sharedComb[13] * r3;
        float newR2 = sharedPost[2] * xv;
        newR2 += sharedComb[2] * r0;
        newR2 += sharedComb[6] * r1;
        newR2 += sharedComb[10] * r2;
        newR2 += sharedComb[14] * r3;
        float newR3 = sharedPost[3] * xv;
        newR3 += sharedComb[3] * r0;
        newR3 += sharedComb[7] * r1;
        newR3 += sharedComb[11] * r2;
        newR3 += sharedComb[15] * r3;

        if (tile == 0) {
            outputRow[d] = __float2bfloat16_rn(newR0);
            outputRow[dim + d] = __float2bfloat16_rn(newR1);
            outputRow[2 * dim + d] = __float2bfloat16_rn(newR2);
            outputRow[3 * dim + d] = __float2bfloat16_rn(newR3);
            square += newR0 * newR0;
            square += newR1 * newR1;
            square += newR2 * newR2;
            square += newR3 * newR3;
        }

        const float *w0 = weight0 + d;
        const float *w1 = weight1 + d;
        const float *w2 = weight2 + d;
        acc0 += __ldg(w0) * newR0;
        acc0 += __ldg(w0 + dim) * newR1;
        acc0 += __ldg(w0 + 2 * dim) * newR2;
        acc0 += __ldg(w0 + 3 * dim) * newR3;
        acc1 += __ldg(w1) * newR0;
        acc1 += __ldg(w1 + dim) * newR1;
        acc1 += __ldg(w1 + 2 * dim) * newR2;
        acc1 += __ldg(w1 + 3 * dim) * newR3;
        acc2 += __ldg(w2) * newR0;
        acc2 += __ldg(w2 + dim) * newR1;
        acc2 += __ldg(w2 + 2 * dim) * newR2;
        acc2 += __ldg(w2 + 3 * dim) * newR3;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        square += __shfl_down_sync(0xffffffffu, square, offset);
    }
    if (lane == 0) {
        warpSums[warp][0] = acc0;
        warpSums[warp][1] = acc1;
        warpSums[warp][2] = acc2;
        if (tile == 0) {
            warpSums[warp][3] = square;
        }
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < tileN) {
            float value = 0.0f;
#pragma unroll
            for (int w = 0; w < warps; ++w) {
                value += warpSums[w][lane];
            }
            dots[((uint64_t)token * dotsStride + n0 + lane) * splitK +
                 part] = value;
        }
        if (tile == 0 && lane == 0) {
            float value = 0.0f;
#pragma unroll
            for (int w = 0; w < warps; ++w) {
                value += warpSums[w][3];
            }
            dots[((uint64_t)token * dotsStride + mixHc) * splitK +
                 part] = value;
        }
    }
}

// Decode specialization combining the previous block's post mapping with the
// next block's 24 pre-norm projections.  The transition is rounded to BF16
// before either a projection or the square sum consumes it, exactly like the
// established HcPost + HcPre operator sequence.  Each split also covers the
// same contiguous 2048-element slice and uses the same 256-thread reduction
// tree as DeepSeekV4HcPreDotsBlockKernel.  These details preserve the numerical
// behavior of the unfused path for DSpark's eight-row target verification.
template <bool materializeTransitionOnce>
__global__ void DeepSeekV4HcPostPreDots4x4096Kernel(
        const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
        const float *post, const float *comb, const float *nextFn,
        float *dots, __nv_bfloat16 *residualOutput, int tokens) {
    constexpr int hcMult = 4;
    constexpr int dim = 4096;
    constexpr int flatDim = hcMult * dim;
    constexpr int mixHc = (2 + hcMult) * hcMult;
    constexpr int dotsStride = mixHc + 1;
    // Match the production TileLang schedule's three output columns per CTA.
    // Every column keeps its original per-thread accumulation and reduction
    // order; sharing the rounded transition only removes duplicate work.
    constexpr int tileN = 3;
    // Keep the same split count selected by DeepSeekV4HcPreDotParts for the
    // 4x4096 production shape (ceil(16384 / (256 * 4)), capped at 16).
    constexpr int splitK = 16;
    constexpr int flatChunk = flatDim / splitK;
    constexpr int partsPerStream = splitK / hcMult;

    __shared__ float reductions[tileN + 1][256];

    int tile = blockIdx.x;
    int part = blockIdx.y;
    int token = blockIdx.z;
    if constexpr (!materializeTransitionOnce) {
        if (token >= tokens || tile >= mixHc / tileN || part >= splitK) {
            return;
        }
    } else {
        // Ordinary SM120 decode launches the complete 8 x 16 CTA grid
        // cooperatively.  Materialize the double -> BF16 transition exactly
        // once, then make it visible to every projection CTA without adding
        // a second CUDA Graph node.
        uint64_t linearBlock =
            ((uint64_t)blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x +
            blockIdx.x;
        uint64_t gridThreads =
            (uint64_t)gridDim.x * gridDim.y * gridDim.z * blockDim.x;
        uint64_t transitionTotal = (uint64_t)tokens * flatDim;
        for (uint64_t idx = linearBlock * blockDim.x + threadIdx.x;
             idx < transitionTotal; idx += gridThreads) {
            int d = idx % dim;
            uint64_t tmp = idx / dim;
            int target = tmp % hcMult;
            int transitionToken = tmp / hcMult;
            const __nv_bfloat16 *transitionX =
                x + (uint64_t)transitionToken * dim;
            const __nv_bfloat16 *transitionResidual =
                residual + (uint64_t)transitionToken * flatDim;
            const float *transitionPost =
                post + (uint64_t)transitionToken * hcMult;
            const float *transitionComb =
                comb + (uint64_t)transitionToken * hcMult * hcMult;
            double value =
                (double)transitionPost[target] *
                (double)__bfloat162float(transitionX[d]);
#pragma unroll
            for (int source = 0; source < hcMult; ++source) {
                value +=
                    (double)transitionComb[source * hcMult + target] *
                    (double)__bfloat162float(
                        transitionResidual[(uint64_t)source * dim + d]);
            }
            residualOutput[idx] = __float2bfloat16_rn((float)value);
        }
        cooperative_groups::this_grid().sync();
    }

    const __nv_bfloat16 *xRow = x + (uint64_t)token * dim;
    const __nv_bfloat16 *residualRow =
        residual + (uint64_t)token * flatDim;
    __nv_bfloat16 *outputRow =
        residualOutput + (uint64_t)token * flatDim;
    const float *postRow = post + (uint64_t)token * hcMult;
    const float *combRow = comb + (uint64_t)token * hcMult * hcMult;

    int n0 = tile * tileN;
    const float *weight0 = nextFn + (uint64_t)n0 * flatDim;
    const float *weight1 = weight0 + flatDim;
    const float *weight2 = weight1 + flatDim;
    // A generic HcPre split is a contiguous flat-tensor chunk.  With H=4,
    // D=4096 and sixteen splits this is one quarter of one residual stream.
    int target = part / partsPerStream;
    int hiddenStart = (part % partsPerStream) * flatChunk;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    const float *targetWeight0 = weight0 + (uint64_t)target * dim;
    const float *targetWeight1 = weight1 + (uint64_t)target * dim;
    const float *targetWeight2 = weight2 + (uint64_t)target * dim;

    auto roundedTransition = [&](int d) {
        if constexpr (materializeTransitionOnce) {
            return outputRow[(uint64_t)target * dim + d];
        }
        double value =
            (double)postRow[target] * (double)__bfloat162float(xRow[d]);
#pragma unroll
        for (int source = 0; source < hcMult; ++source) {
            value += (double)combRow[source * hcMult + target] *
                     (double)__bfloat162float(
                         residualRow[(uint64_t)source * dim + d]);
        }
        return __float2bfloat16_rn((float)value);
    };

    // Match the pair assignment in DeepSeekV4HcPreDotsBlockKernel: every
    // thread owns four pairs separated by blockDim.x * 2.
#pragma unroll
    for (int iteration = 0; iteration < flatChunk / (256 * 2);
         ++iteration) {
        int d = hiddenStart + threadIdx.x * 2 + iteration * 256 * 2;
        __nv_bfloat16 rounded0 = roundedTransition(d);
        __nv_bfloat16 rounded1 = roundedTransition(d + 1);
        float value0 = __bfloat162float(rounded0);
        float value1 = __bfloat162float(rounded1);
        acc0 += value0 * targetWeight0[d] +
                value1 * targetWeight0[d + 1];
        acc1 += value0 * targetWeight1[d] +
                value1 * targetWeight1[d + 1];
        acc2 += value0 * targetWeight2[d] +
                value1 * targetWeight2[d + 1];
    }

    float square = 0.0f;
    if (tile == 0) {
        // The generic square-sum branch assigns scalar elements rather than
        // pairs.  Recompute this inexpensive four-FMA transition using that
        // exact assignment so its reduction is bit-for-bit reproducible.
#pragma unroll
        for (int iteration = 0; iteration < flatChunk / 256;
             ++iteration) {
            int d = hiddenStart + threadIdx.x + iteration * 256;
            __nv_bfloat16 rounded = roundedTransition(d);
            outputRow[(uint64_t)target * dim + d] = rounded;
            float value = __bfloat162float(rounded);
            square += value * value;
        }
    }

    reductions[0][threadIdx.x] = acc0;
    reductions[1][threadIdx.x] = acc1;
    reductions[2][threadIdx.x] = acc2;
    reductions[3][threadIdx.x] = square;
    __syncthreads();
    // Preserve the established 256-lane binary tree.  Once only one warp is
    // live, shuffle-down performs the same pairings as shared memory while
    // removing five block-wide barriers.
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            reductions[0][threadIdx.x] +=
                reductions[0][threadIdx.x + stride];
            reductions[1][threadIdx.x] +=
                reductions[1][threadIdx.x + stride];
            reductions[2][threadIdx.x] +=
                reductions[2][threadIdx.x + stride];
            reductions[3][threadIdx.x] +=
                reductions[3][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float reduced0 = reductions[0][threadIdx.x];
        float reduced1 = reductions[1][threadIdx.x];
        float reduced2 = reductions[2][threadIdx.x];
        float reducedSquare = reductions[3][threadIdx.x];
#pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            reduced0 += __shfl_down_sync(0xffffffff, reduced0, stride);
            reduced1 += __shfl_down_sync(0xffffffff, reduced1, stride);
            reduced2 += __shfl_down_sync(0xffffffff, reduced2, stride);
            reducedSquare +=
                __shfl_down_sync(0xffffffff, reducedSquare, stride);
        }
        if (threadIdx.x != 0) {
            return;
        }
        dots[((uint64_t)token * dotsStride + n0) * splitK + part] =
            reduced0;
        dots[((uint64_t)token * dotsStride + n0 + 1) * splitK + part] =
            reduced1;
        dots[((uint64_t)token * dotsStride + n0 + 2) * splitK + part] =
            reduced2;
        if (tile == 0) {
            dots[((uint64_t)token * dotsStride + mixHc) * splitK + part] =
                reducedSquare;
        }
    }
}

bool DeepSeekV4CanLaunchHcPostPreCooperative() {
    struct CooperativeCapacity {
        int device = -1;
        bool supported = false;
    };
    static thread_local CooperativeCapacity capacity;

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    if (capacity.device == device) {
        return capacity.supported;
    }

    int cooperative = 0;
    int multiprocessors = 0;
    int blocksPerMultiprocessor = 0;
    cudaError_t cooperativeState = cudaDeviceGetAttribute(
        &cooperative, cudaDevAttrCooperativeLaunch, device);
    cudaError_t multiprocessorState = cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, device);
    cudaError_t occupancyState =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerMultiprocessor,
            DeepSeekV4HcPostPreDots4x4096Kernel<true>, 256, 0);
    constexpr int requiredBlocks = (24 / 3) * 16;
    capacity.device = device;
    capacity.supported =
        cooperativeState == cudaSuccess &&
        multiprocessorState == cudaSuccess &&
        occupancyState == cudaSuccess && cooperative != 0 &&
        multiprocessors * blocksPerMultiprocessor >= requiredBlocks;
    return capacity.supported;
}

bool DeepSeekV4PrepareCudaOutput(fastllm::Data &output, fastllm::DataType dataType,
                                 const std::vector<int> &dims) {
    output.dataType = dataType;
    output.Resize(dims);
    // MultiCuda delegates this CUDA operator from one worker per device.  The
    // generic ToDevice(CUDA) overload resolves the executor's default CUDA
    // device (normally device 0), which would switch non-root workers back to
    // GPU 0 before launching the kernel.  Keep temporary/output tensors on the
    // worker's current CUDA device instead.
    output.ToDevice(fastllm::DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
    output.Allocate(false);
    return output.cudaData != nullptr;
}

extern "C" bool FastllmCudaDeepSeekV4HcMean(
        const fastllm::Data &x, fastllm::Data &output) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || x.cudaData == nullptr ||
        x.dataType != fastllm::DataType::BFLOAT16 || x.dims.size() != 4 ||
        x.dims[0] <= 0 || x.dims[1] <= 0 || x.dims[2] <= 0 ||
        x.dims[3] <= 0) {
        return false;
    }
    int tokens = x.dims[0] * x.dims[1];
    int hc = x.dims[2];
    int dim = x.dims[3];
    if (!DeepSeekV4PrepareCudaOutput(
            output, fastllm::DataType::BFLOAT16,
            {x.dims[0], x.dims[1], dim})) {
        return false;
    }
    constexpr int threads = 256;
    int total = tokens * dim;
    DeepSeekV4HcMeanBf16Kernel<<<(total + threads - 1) / threads, threads>>>(
        (const __nv_bfloat16 *)x.cudaData,
        (__nv_bfloat16 *)output.cudaData, tokens, hc, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool DeepSeekV4CublasDataType(fastllm::DataType dataType, cudaDataType_t &cudaType) {
    if (dataType == fastllm::DataType::BFLOAT16) {
        cudaType = CUDA_R_16BF;
        return true;
    }
    if (dataType == fastllm::DataType::FLOAT16) {
        cudaType = CUDA_R_16F;
        return true;
    }
    if (dataType == fastllm::DataType::FLOAT32) {
        cudaType = CUDA_R_32F;
        return true;
    }
    return false;
}

cublasStatus_t DeepSeekV4CublasSgemmStrict(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const float *alpha,
        const float *A, int lda, const float *B, int ldb,
        const float *beta, float *C, int ldc) {
#if CUBLAS_VERSION >= 11000
    return cublasGemmEx(
        handle, transa, transb, m, n, k,
        alpha,
        A, CUDA_R_32F, lda,
        B, CUDA_R_32F, ldb,
        beta,
        C, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT);
#else
    return cublasSgemm(
        handle, transa, transb, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

cublasStatus_t DeepSeekV4CublasBf16GemmToFloat(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const float *alpha,
        const __nv_bfloat16 *A, int lda, const __nv_bfloat16 *B, int ldb,
        const float *beta, float *C, int ldc) {
#if CUBLAS_VERSION >= 11000
    return cublasGemmEx(
        handle, transa, transb, m, n, k,
        alpha,
        A, CUDA_R_16BF, lda,
        B, CUDA_R_16BF, ldb,
        beta,
        C, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    return CUBLAS_STATUS_NOT_SUPPORTED;
#endif
}

bool DeepSeekV4LaunchWoAGemm(const fastllm::Data &o, const fastllm::Data &woA,
                             fastllm::Data &output, int bsz, int seqlen,
                             int heads, int headDim, int groups, int oRank) {
    int tokens = bsz * seqlen;
    if (tokens < 16) {
        return false;
    }
    cudaDataType_t oType, wType;
    if (!DeepSeekV4CublasDataType(o.dataType, oType) ||
        !DeepSeekV4CublasDataType(woA.dataType, wType) ||
        output.dataType != fastllm::DataType::BFLOAT16 ||
        heads % groups != 0) {
        return false;
    }

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    int fullDim = heads * headDim;
    float alpha = 1.0f, beta = 0.0f;
    auto handle = getFastllmCublasHandle();

    if (o.dataType == fastllm::DataType::BFLOAT16 && woA.dataType == fastllm::DataType::FLOAT16) {
        half *halfO = (half *)FastllmCudaMalloc(o.Count(0) * sizeof(half));
        half *halfOut = (half *)FastllmCudaMalloc(output.Count(0) * sizeof(half));
        if (halfO != nullptr && halfOut != nullptr) {
            FastllmBF16ToHalf(o.cudaData, halfO, (int)o.Count(0));
            half hAlpha = __float2half(1.0f), hBeta = __float2half(0.0f);
            cublasStatus_t halfStatus = cublasHgemmStridedBatched(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                oRank, tokens, groupDim,
                &hAlpha,
                (const half *)woA.cudaData, groupDim, (long long)oRank * groupDim,
                halfO, fullDim, groupDim,
                &hBeta,
                halfOut, groups * oRank, oRank,
                groups);
            if (halfStatus == CUBLAS_STATUS_SUCCESS) {
                FastllmHalfToBF16(halfOut, output.cudaData, (int)output.Count(0));
                FastllmCudaFree(halfO);
                FastllmCudaFree(halfOut);
                return true;
            }
            if (std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
                printf("DeepSeekV4WoA half cuBLAS fallback: status=%d tokens=%d groupDim=%d groups=%d oRank=%d\n",
                       (int)halfStatus, tokens, groupDim, groups, oRank);
            }
        }
        if (halfO != nullptr) {
            FastllmCudaFree(halfO);
        }
        if (halfOut != nullptr) {
            FastllmCudaFree(halfOut);
        }
        return false;
    }

    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        oRank, tokens, groupDim,
        &alpha,
        woA.cudaData, wType, groupDim, (long long)oRank * groupDim,
        o.cudaData, oType, fullDim, groupDim,
        &beta,
        output.cudaData, CUDA_R_16BF, groups * oRank, oRank,
        groups,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }

    if (status != CUBLAS_STATUS_SUCCESS && std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
        printf("DeepSeekV4WoA cuBLAS fallback: status=%d oType=%d wType=%d tokens=%d groupDim=%d groups=%d oRank=%d\n",
               (int)status, (int)o.dataType, (int)woA.dataType, tokens, groupDim, groups, oRank);
    }
    return false;
}

template <typename InT>
bool DeepSeekV4LaunchWoAByWeight(const fastllm::Data &o, const fastllm::Data &woA, int groups,
                                 int oRank, fastllm::Data &output, int bsz, int seqlen,
                                 int heads, int headDim) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_GEMM") == nullptr &&
        DeepSeekV4LaunchWoAGemm(o, woA, output, bsz, seqlen, heads, headDim, groups, oRank)) {
        return true;
    }
    bool usePair = std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_PAIR") != nullptr && (oRank % 2 == 0);
    bool useFloatAcc = !usePair && std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_FLOAT_ACC") != nullptr;
    bool useKahanAcc = !usePair && !useFloatAcc && std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_KAHAN_ACC") != nullptr;
    bool useBlockReduce = !usePair && !useFloatAcc && !useKahanAcc &&
                          std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_BLOCK") == nullptr;
    bool usePairBlockReduce = useBlockReduce && seqlen == 1 && (oRank % 2 == 0) &&
                              std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_PAIR_BLOCK") == nullptr;
    int total = bsz * seqlen * groups * oRank;
    int threads = std::min(256, std::max(1, total));
    int pairTotal = bsz * seqlen * groups * (oRank / 2);
    int launchTotal = (usePair || usePairBlockReduce) ? pairTotal : total;
    int blocks = (launchTotal + threads - 1) / threads;
    const InT *oData = (const InT *)o.cudaData;
    __nv_bfloat16 *outData = (__nv_bfloat16 *)output.cudaData;

    if (woA.dataType == fastllm::DataType::FP8_E4M3) {
        int headsPerGroup = heads / groups;
        int groupDim = headsPerGroup * headDim;
        int weightRows = groups * oRank;
        int scaleRows = woA.blockK > 0 ? (weightRows + woA.blockK - 1) / woA.blockK : 0;
        int scaleCols = woA.blockM > 0 ? (groupDim + woA.blockM - 1) / woA.blockM : 0;
        if ((oRank & 1) != 0 || woA.blockK <= 0 || woA.blockM <= 0 ||
            woA.scales.size() != (size_t)scaleRows * scaleCols) {
            return false;
        }
        fastllm::Data &mutableWeight = const_cast<fastllm::Data&>(woA);
        fastllm::Data emptyBias;
        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(mutableWeight, emptyBias, weightRows);
        if (mutableWeight.extraCudaData.empty() || mutableWeight.extraCudaData[0] == nullptr) {
            return false;
        }
        const float *scaleData =
            (const float*)mutableWeight.extraCudaData[0];
        constexpr int fallbackRowsPerBlock = 4;
        const bool fallbackRowsCompatible =
            oRank % fallbackRowsPerBlock == 0 &&
            woA.blockK % fallbackRowsPerBlock == 0 &&
            oRank % woA.blockK == 0;
        if (fallbackRowsCompatible) {
            const int totalTokens = bsz * seqlen;
            int tokensPerBlock = totalTokens == 2 ? 2 : 4;
            const char *tokenTile =
                std::getenv("FASTLLM_DSV4_CUDA_WOA_TOKENS_PER_BLOCK");
            if (tokenTile != nullptr) {
                tokensPerBlock = std::atoi(tokenTile);
            }
            int rowsPerBlock =
                (oRank % 8 == 0 && woA.blockK % 8 == 0) ? 8 : 4;
            const char *rowTile =
                std::getenv("FASTLLM_DSV4_CUDA_WOA_ROWS_PER_BLOCK");
            if (rowTile != nullptr) {
                rowsPerBlock = std::atoi(rowTile);
            }
            const bool useTokenTile =
                totalTokens >= 2 &&
                std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_TOKEN_TILE") ==
                    nullptr && rowsPerBlock > 0 &&
                oRank % rowsPerBlock == 0 &&
                woA.blockK % rowsPerBlock == 0;
            bool launched = false;
            const uint8_t *weightData = (const uint8_t*)woA.cudaData;
            if (useTokenTile && tokensPerBlock == 2 &&
                rowsPerBlock == 4) {
                launched = DeepSeekV4LaunchWoAFp8TokenRowsBlockReduce<
                    InT, 2, 4>(
                        oData, weightData, scaleData, outData,
                        bsz, seqlen, heads, headDim, groups, oRank,
                        woA.blockK, woA.blockM, scaleCols);
            } else if (useTokenTile && tokensPerBlock == 4 &&
                       rowsPerBlock == 4) {
                launched = DeepSeekV4LaunchWoAFp8TokenRowsBlockReduce<
                    InT, 4, 4>(
                        oData, weightData, scaleData, outData,
                        bsz, seqlen, heads, headDim, groups, oRank,
                        woA.blockK, woA.blockM, scaleCols);
            } else if (useTokenTile && tokensPerBlock == 8 &&
                       rowsPerBlock == 4) {
                launched = DeepSeekV4LaunchWoAFp8TokenRowsBlockReduce<
                    InT, 8, 4>(
                        oData, weightData, scaleData, outData,
                        bsz, seqlen, heads, headDim, groups, oRank,
                        woA.blockK, woA.blockM, scaleCols);
            } else if (useTokenTile && tokensPerBlock == 2 &&
                       rowsPerBlock == 8) {
                launched = DeepSeekV4LaunchWoAFp8TokenRowsBlockReduce<
                    InT, 2, 8>(
                        oData, weightData, scaleData, outData,
                        bsz, seqlen, heads, headDim, groups, oRank,
                        woA.blockK, woA.blockM, scaleCols);
            } else if (useTokenTile && tokensPerBlock == 4 &&
                       rowsPerBlock == 8) {
                launched = DeepSeekV4LaunchWoAFp8TokenRowsBlockReduce<
                    InT, 4, 8>(
                        oData, weightData, scaleData, outData,
                        bsz, seqlen, heads, headDim, groups, oRank,
                        woA.blockK, woA.blockM, scaleCols);
            } else if (useTokenTile && tokensPerBlock == 8 &&
                       rowsPerBlock == 8) {
                launched = DeepSeekV4LaunchWoAFp8TokenRowsBlockReduce<
                    InT, 8, 8>(
                        oData, weightData, scaleData, outData,
                        bsz, seqlen, heads, headDim, groups, oRank,
                        woA.blockK, woA.blockM, scaleCols);
            }
            if (!launched) {
                int rowBlocks = totalTokens * groups * (oRank / 4);
                DeepSeekV4WoAFp8RowsBlockReduceKernel<InT, 4>
                    <<<rowBlocks, 256, 4 * 256 * sizeof(float)>>>(
                        oData, weightData, scaleData, outData,
                        bsz, seqlen, heads, headDim, groups, oRank,
                        woA.blockK, woA.blockM, scaleCols);
            }
        } else {
            DeepSeekV4WoAFp8PairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const uint8_t*)woA.cudaData,
                (const float*)mutableWeight.extraCudaData[0], outData,
                bsz, seqlen, heads, headDim, groups, oRank,
                woA.blockK, woA.blockM, scaleCols);
        }
        return true;
    }

    if (woA.dataType == fastllm::DataType::BFLOAT16) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else if (usePairBlockReduce) {
            DeepSeekV4WoAPairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useBlockReduce) {
            DeepSeekV4WoABlockReduceKernel<<<total, 256, 256 * sizeof(float)>>>(
                oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useFloatAcc) {
            DeepSeekV4WoAFloatAccKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useKahanAcc) {
            DeepSeekV4WoAKahanAccKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else if (woA.dataType == fastllm::DataType::FLOAT16) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else if (usePairBlockReduce) {
            DeepSeekV4WoAPairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const half *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useBlockReduce) {
            DeepSeekV4WoABlockReduceKernel<<<total, 256, 256 * sizeof(float)>>>(
                oData, (const half *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useFloatAcc) {
            DeepSeekV4WoAFloatAccKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useKahanAcc) {
            DeepSeekV4WoAKahanAccKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else if (woA.dataType == fastllm::DataType::FLOAT32) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else if (usePairBlockReduce) {
            DeepSeekV4WoAPairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const float *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useBlockReduce) {
            DeepSeekV4WoABlockReduceKernel<<<total, 256, 256 * sizeof(float)>>>(
                oData, (const float *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useFloatAcc) {
            DeepSeekV4WoAFloatAccKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useKahanAcc) {
            DeepSeekV4WoAKahanAccKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else {
        return false;
    }
    return true;
}

template <typename XT, typename RT>
bool DeepSeekV4LaunchHcPostByOutput(const fastllm::Data &x, const fastllm::Data &residual,
                                    const float *cudaPost, const float *cudaComb,
                                    fastllm::Data &output, int tokens, int hcMult, int dim) {
    int threads = 256;
    const XT *xData = (const XT *)x.cudaData;
    const RT *resData = (const RT *)residual.cudaData;

    int hcPost4MinTokens = 16384;
    if (const char *env = std::getenv("FASTLLM_DSV4_HCPOST4_MIN_TOKENS")) {
        hcPost4MinTokens = std::max(1, std::atoi(env));
    }
    bool useHcPost4 = hcMult == 4 &&
                      std::getenv("FASTLLM_DSV4_DISABLE_CUDA_HCPOST4") == nullptr &&
                      (std::getenv("FASTLLM_DSV4_ENABLE_CUDA_HCPOST4") != nullptr ||
                       tokens >= hcPost4MinTokens);
    if (useHcPost4) {
        uint64_t total = (uint64_t)tokens * dim;
        int blocks = (int)((total + threads - 1) / threads);
        if (output.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4HcPost4Kernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                         (__nv_bfloat16 *)output.cudaData, tokens, dim);
            return true;
        } else if (output.dataType == fastllm::DataType::FLOAT16) {
            DeepSeekV4HcPost4Kernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                         (half *)output.cudaData, tokens, dim);
            return true;
        } else if (output.dataType == fastllm::DataType::FLOAT32) {
            DeepSeekV4HcPost4Kernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                         (float *)output.cudaData, tokens, dim);
            return true;
        }
    }

    uint64_t total = (uint64_t)tokens * hcMult * dim;
    int blocks = (int)((total + threads - 1) / threads);
    if (output.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPostKernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                    (__nv_bfloat16 *)output.cudaData, tokens, hcMult, dim);
    } else if (output.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPostKernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                    (half *)output.cudaData, tokens, hcMult, dim);
    } else if (output.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPostKernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                    (float *)output.cudaData, tokens, hcMult, dim);
    } else {
        return false;
    }
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPostByResidual(const fastllm::Data &x, const fastllm::Data &residual,
                                      const float *cudaPost, const float *cudaComb,
                                      fastllm::Data &output, int tokens, int hcMult, int dim) {
    if (residual.dataType == fastllm::DataType::BFLOAT16) {
        return DeepSeekV4LaunchHcPostByOutput<XT, __nv_bfloat16>(x, residual, cudaPost, cudaComb,
                                                                output, tokens, hcMult, dim);
    }
    if (residual.dataType == fastllm::DataType::FLOAT16) {
        return DeepSeekV4LaunchHcPostByOutput<XT, half>(x, residual, cudaPost, cudaComb,
                                                       output, tokens, hcMult, dim);
    }
    if (residual.dataType == fastllm::DataType::FLOAT32) {
        return DeepSeekV4LaunchHcPostByOutput<XT, float>(x, residual, cudaPost, cudaComb,
                                                        output, tokens, hcMult, dim);
    }
    return false;
}

template <typename QT>
bool DeepSeekV4LaunchSparseDecodeCublas(const void *cudaQ, const void *cudaCompressed,
                                        fastllm::DataType compressedType, const float *cudaWindow,
                                        const float *cudaSink, float *outData,
                                        int bsz, int heads, int dim, int windowSize,
                                        int startPos, int compressedCount, int liveWindow,
                                        int keyCount, float softmaxScale) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUBLAS_SPARSE_DECODE") != nullptr) {
        return false;
    }
    // A TP shard can contain only a handful of heads.  For small decode
    // problems, converting/gathering into FP32 and launching two GEMMs costs
    // more than the fused online kernel.  Keep cuBLAS for sufficiently large
    // head-key products, where reusing the gathered KV starts to pay off.
    constexpr int minCublasWork = 1024;
    if ((uint64_t)bsz * heads * keyCount < (uint64_t)minCublasWork) {
        return false;
    }
    int maxCublasKeys = kDeepSeekV4SparseDecodeMaxKeys;
    if (const char *env = std::getenv("FASTLLM_DSV4_CUBLAS_SPARSE_DECODE_MAX_KEYS")) {
        maxCublasKeys = std::max(1, std::atoi(env));
    }
    if (keyCount <= 0 || keyCount > maxCublasKeys || dim <= 0 || heads <= 0 || bsz <= 0) {
        return false;
    }

    uint64_t qElems = (uint64_t)bsz * heads * dim;
    uint64_t kvElems = (uint64_t)bsz * keyCount * dim;
    uint64_t scoreElems = (uint64_t)bsz * heads * keyCount;
    float *qFloat = (float *)FastllmCudaMalloc(qElems * sizeof(float));
    float *kvFloat = (float *)FastllmCudaMalloc(kvElems * sizeof(float));
    float *scores = (float *)FastllmCudaMalloc(scoreElems * sizeof(float));
    if (qFloat == nullptr || kvFloat == nullptr || scores == nullptr) {
        if (qFloat != nullptr) {
            FastllmCudaFree(qFloat);
        }
        if (kvFloat != nullptr) {
            FastllmCudaFree(kvFloat);
        }
        if (scores != nullptr) {
            FastllmCudaFree(scores);
        }
        return false;
    }

    int threads = 256;
    DeepSeekV4SparseDecodeConvertQKernel<QT><<<
        (int)((qElems + threads - 1) / threads), threads>>>((const QT *)cudaQ, qFloat, qElems);

    bool gathered = true;
    if (compressedCount == 0 || cudaCompressed == nullptr) {
        DeepSeekV4SparseDecodeGatherKVKernel<float><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const float *)nullptr, kvFloat, bsz, dim, windowSize,
                startPos, 0, liveWindow, keyCount);
    } else if (compressedType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4SparseDecodeGatherKVKernel<__nv_bfloat16><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const __nv_bfloat16 *)cudaCompressed, kvFloat, bsz, dim, windowSize,
                startPos, compressedCount, liveWindow, keyCount);
    } else if (compressedType == fastllm::DataType::FLOAT16) {
        DeepSeekV4SparseDecodeGatherKVKernel<half><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const half *)cudaCompressed, kvFloat, bsz, dim, windowSize,
                startPos, compressedCount, liveWindow, keyCount);
    } else if (compressedType == fastllm::DataType::FLOAT32) {
        DeepSeekV4SparseDecodeGatherKVKernel<float><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const float *)cudaCompressed, kvFloat, bsz, dim, windowSize,
                startPos, compressedCount, liveWindow, keyCount);
    } else {
        gathered = false;
    }

    bool ok = false;
    if (gathered) {
        float alpha = softmaxScale;
        float beta = 0.0f;
        auto handle = getFastllmCublasHandle();
        cublasStatus_t status = cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            keyCount, heads, dim,
            &alpha,
            kvFloat, dim, (long long)keyCount * dim,
            qFloat, dim, (long long)heads * dim,
            &beta,
            scores, keyCount, (long long)heads * keyCount,
            bsz);

        if (status == CUBLAS_STATUS_SUCCESS) {
            DeepSeekV4SparseDecodeSoftmaxSinkKernel<<<bsz * heads, 256, 256 * sizeof(float)>>>(
                scores, cudaSink, bsz, heads, keyCount);

            alpha = 1.0f;
            status = cublasSgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dim, heads, keyCount,
                &alpha,
                kvFloat, dim, (long long)keyCount * dim,
                scores, keyCount, (long long)heads * keyCount,
                &beta,
                outData, dim, (long long)heads * dim,
                bsz);
        }
        ok = status == CUBLAS_STATUS_SUCCESS;
        if (!ok && std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_DECODE") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparseDecode cuBLAS fallback: status=%d bsz=%d heads=%d dim=%d keys=%d compressed=%d\n",
                         (int)status, bsz, heads, dim, keyCount, compressedCount);
        }
    }

    FastllmCudaFree(qFloat);
    FastllmCudaFree(kvFloat);
    FastllmCudaFree(scores);
    return ok;
}

template <typename QT>
bool DeepSeekV4LaunchSparseDecodeByCompressed(const void *cudaQ, const void *cudaCompressed,
                                              fastllm::DataType compressedType, const float *cudaWindow,
                                              const float *cudaSink, float *outData,
                                              int bsz, int heads, int dim, int windowSize,
                                              int startPos, int compressedCount, float softmaxScale,
                                              int *launchMode = nullptr) {
    const QT *qData = (const QT *)cudaQ;
    int blocks = bsz * heads;
    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int maxKeys = liveWindow + compressedCount;
    if (maxKeys <= 0 || maxKeys > kDeepSeekV4SparseDecodeMaxKeys) {
        return false;
    }
    if (launchMode != nullptr) {
        *launchMode = 0;
    }
    if (DeepSeekV4LaunchSparseDecodeCublas<QT>(
            cudaQ, cudaCompressed, compressedType, cudaWindow, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, liveWindow,
            maxKeys, softmaxScale)) {
        if (launchMode != nullptr) {
            *launchMode = 2;
        }
        return true;
    }
    bool useOnlineDecode = dim <= 256 * 4 &&
                           std::getenv("FASTLLM_DSV4_DISABLE_ONLINE_SPARSE_DECODE") == nullptr;
    if (useOnlineDecode) {
        if (compressedCount == 0 || cudaCompressed == nullptr) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, float><<<blocks, 256>>>(
                qData, cudaWindow, (const float *)nullptr, cudaSink, outData,
                bsz, 1, heads, dim, windowSize, startPos, 0, 0,
                nullptr, nullptr, softmaxScale, nullptr, 0);
        } else if (compressedType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, __nv_bfloat16><<<blocks, 256>>>(
                qData, cudaWindow, (const __nv_bfloat16 *)cudaCompressed, cudaSink, outData,
                bsz, 1, heads, dim, windowSize, startPos, compressedCount,
                compressedCount, nullptr, nullptr, softmaxScale, nullptr, 0);
        } else if (compressedType == fastllm::DataType::FLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, half><<<blocks, 256>>>(
                qData, cudaWindow, (const half *)cudaCompressed, cudaSink, outData,
                bsz, 1, heads, dim, windowSize, startPos, compressedCount,
                compressedCount, nullptr, nullptr, softmaxScale, nullptr, 0);
        } else if (compressedType == fastllm::DataType::FLOAT32) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, float><<<blocks, 256>>>(
                qData, cudaWindow, (const float *)cudaCompressed, cudaSink, outData,
                bsz, 1, heads, dim, windowSize, startPos, compressedCount,
                compressedCount, nullptr, nullptr, softmaxScale, nullptr, 0);
        } else {
            return false;
        }
        if (launchMode != nullptr) {
            *launchMode = 1;
        }
        return true;
    }
    size_t sharedBytes = ((size_t)maxKeys + 256) * sizeof(float);
    if (compressedCount == 0 || cudaCompressed == nullptr) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, float>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const float *)nullptr, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, 0, maxKeys, softmaxScale);
    } else if (compressedType == fastllm::DataType::BFLOAT16) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, __nv_bfloat16>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const __nv_bfloat16 *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, maxKeys, softmaxScale);
    } else if (compressedType == fastllm::DataType::FLOAT16) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, half>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const half *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, maxKeys, softmaxScale);
    } else if (compressedType == fastllm::DataType::FLOAT32) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, float>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const float *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, maxKeys, softmaxScale);
    } else {
        return false;
    }
    return true;
}

template <typename QT, typename CT>
bool DeepSeekV4LaunchSparseDecodeBatchOnline(const void * const *cudaQPtrs,
                                             const float * const *cudaWindowPtrs,
                                             const void * const *cudaCompressedPtrs,
                                             const float *cudaSink,
                                             const int *cudaStartPositions,
                                             const int *cudaCompressedCounts,
                                             float *outData, int batch, int heads,
                                             int dim, int windowSize,
                                             float softmaxScale) {
    if (dim <= 0 || dim > 256 * 4 || batch <= 0 || heads <= 0 || windowSize <= 0) {
        return false;
    }
    DeepSeekV4SparseAttentionDecodeCachedBatchOnlineKernel<QT, CT><<<batch * heads, 256>>>(
        cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, cudaSink,
        cudaStartPositions, cudaCompressedCounts, outData,
        batch, heads, dim, windowSize, softmaxScale);
    return true;
}

template <typename QT>
bool DeepSeekV4LaunchSparseDecodeBatchByCompressed(const void * const *cudaQPtrs,
                                                   const float * const *cudaWindowPtrs,
                                                   const void * const *cudaCompressedPtrs,
                                                   fastllm::DataType compressedType,
                                                   const float *cudaSink,
                                                   const int *cudaStartPositions,
                                                   const int *cudaCompressedCounts,
                                                   float *outData, int batch, int heads,
                                                   int dim, int windowSize,
                                                   float softmaxScale) {
    if (compressedType == fastllm::DataType::BFLOAT16) {
        return DeepSeekV4LaunchSparseDecodeBatchOnline<QT, __nv_bfloat16>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, cudaSink,
            cudaStartPositions, cudaCompressedCounts, outData,
            batch, heads, dim, windowSize, softmaxScale);
    }
    if (compressedType == fastllm::DataType::FLOAT16) {
        return DeepSeekV4LaunchSparseDecodeBatchOnline<QT, half>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, cudaSink,
            cudaStartPositions, cudaCompressedCounts, outData,
            batch, heads, dim, windowSize, softmaxScale);
    }
    if (compressedType == fastllm::DataType::FLOAT32) {
        return DeepSeekV4LaunchSparseDecodeBatchOnline<QT, float>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, cudaSink,
            cudaStartPositions, cudaCompressedCounts, outData,
            batch, heads, dim, windowSize, softmaxScale);
    }
    return false;
}

template <typename QT, typename KT>
bool DeepSeekV4LaunchSparsePrefillCompressedCublasSegment(
        const fastllm::Data &q, const fastllm::Data &kv, const float *cudaSink,
        float *compressedScores, float *compressedKVFloat, float *outData,
        int bsz, int seqlen, int heads, int dim, int kvLen,
        int windowSize, int compressRatio, int startPos, int prefixLen,
        int compressedStart, int compressedCount, float softmaxScale,
        int rowOffset, int rows) {
    if (rows <= 0 || compressedCount <= 0 || q.dataType != kv.dataType) {
        return false;
    }
    cudaDataType_t inputType;
    if (!DeepSeekV4CublasDataType(q.dataType, inputType)) {
        return false;
    }

    int rowsPerBatch = seqlen * heads;
    int b = rowOffset / rowsPerBatch;
    if (b < 0 || b >= bsz || rowOffset + rows > (b + 1) * rowsPerBatch) {
        return false;
    }

    const QT *qPtr = (const QT *)q.cudaData + (uint64_t)rowOffset * dim;
    const KT *kvPtr = (const KT *)kv.cudaData + ((uint64_t)b * kvLen + compressedStart) * dim;
    float alpha = softmaxScale;
    float beta = 0.0f;
    auto handle = getFastllmCublasHandle();
    cublasStatus_t status;
    if (q.dataType == fastllm::DataType::FLOAT32) {
        status = cublasSgemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            compressedCount, rows, dim,
            &alpha,
            (const float *)kvPtr, dim,
            (const float *)qPtr, dim,
            &beta,
            compressedScores, compressedCount);
    } else {
        status = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            compressedCount, rows, dim,
            &alpha,
            kvPtr, inputType, dim,
            qPtr, inputType, dim,
            &beta,
            compressedScores, CUDA_R_32F, compressedCount,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparsePrefill compressed cuBLAS QK failed: status=%d rows=%d dim=%d compressed=%d type=%d\n",
                         (int)status, rows, dim, compressedCount, (int)q.dataType);
        }
        return false;
    }

    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    int maxLocalKeys = std::min(windowSize, realPrefixLen + seqlen);
    if (maxLocalKeys <= 0 || maxLocalKeys > kDeepSeekV4SparsePrefillMaxKeys) {
        return false;
    }
    size_t sharedBytes = (size_t)maxLocalKeys * sizeof(float);
    auto kernel = DeepSeekV4SparseAttentionPrefillCublasCompressedKernel<QT, KT>;
    if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
        return false;
    }
    kernel<<<rows, 256, sharedBytes>>>(
        (const QT *)q.cudaData, (const KT *)kv.cudaData, cudaSink, compressedScores, outData,
        bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
        compressedStart, compressedCount, softmaxScale, rowOffset);

    const float *compressedKVForGemm = nullptr;
    if (kv.dataType == fastllm::DataType::FLOAT32) {
        compressedKVForGemm = (const float *)kvPtr;
    } else {
        if (compressedKVFloat == nullptr) {
            return false;
        }
        uint64_t kvElems = (uint64_t)compressedCount * dim;
        int threads = 256;
        DeepSeekV4SparseDecodeGatherKVKernel<KT><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                (const float *)nullptr, kvPtr, compressedKVFloat, 1, dim, 1,
                0, compressedCount, 0, compressedCount);
        compressedKVForGemm = compressedKVFloat;
    }

    float avAlpha = 1.0f;
    float avBeta = 1.0f;
    status = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        dim, rows, compressedCount,
        &avAlpha,
        compressedKVForGemm, dim,
        compressedScores, compressedCount,
        &avBeta,
        outData, dim);
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparsePrefill compressed cuBLAS AV failed: status=%d rows=%d dim=%d compressed=%d type=%d\n",
                         (int)status, rows, dim, compressedCount, (int)kv.dataType);
        }
        return false;
    }
    return true;
}

bool DeepSeekV4RunSparsePrefillCompressedCublas(
        const fastllm::Data &q, const fastllm::Data &kv, const float *cudaSink,
        fastllm::Data &output,
        int bsz, int seqlen, int heads, int dim, int kvLen,
        int windowSize, int compressRatio, int startPos, int prefixLen,
        int ropeDim, float ropeBase, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, float softmaxScale) {
    if (std::getenv("FASTLLM_DSV4_ENABLE_CUBLAS_SPARSE_PREFILL_COMPRESSED") == nullptr) {
        return false;
    }
    if (q.dataType != kv.dataType || compressRatio <= 0 || windowSize <= 0 ||
        dim <= 0 || heads <= 0 || seqlen <= 0) {
        return false;
    }
    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    int compressedStart = realPrefixLen + seqlen;
    int compressedCount = std::max(0, kvLen - compressedStart);
    if (compressedCount <= 0) {
        return false;
    }
    int maxCompressed = kDeepSeekV4Sparse1MCompressedKeys;
    if (const char *env = std::getenv("FASTLLM_DSV4_CUBLAS_SPARSE_PREFILL_MAX_COMPRESSED")) {
        maxCompressed = std::max(1, std::atoi(env));
    }
    if (compressedCount > maxCompressed) {
        return false;
    }

    int rows = bsz * seqlen * heads;
    size_t rowBytes = (size_t)dim * sizeof(float);
    size_t scoreRowBytes = (size_t)compressedCount * sizeof(float);
    size_t perRowBytes = rowBytes + scoreRowBytes;
    size_t maxTempBytes = std::max(perRowBytes, DeepSeekV4SparsePrefillTempBytesLimit());
    int rowsPerChunk = (int)std::max<size_t>(1, maxTempBytes / perRowBytes);
    rowsPerChunk = std::min(rowsPerChunk, rows);
    if (rowsPerChunk <= 0) {
        return false;
    }
    size_t tempBytes = (size_t)rowsPerChunk * rowBytes;
    size_t scoreBytes = (size_t)rowsPerChunk * scoreRowBytes;
    size_t kvFloatBytes = q.dataType == fastllm::DataType::FLOAT32 ?
        0 : (size_t)compressedCount * dim * sizeof(float);
    uint8_t *scratch = (uint8_t *)FastllmCudaMalloc(tempBytes + scoreBytes + kvFloatBytes);
    if (scratch == nullptr) {
        return false;
    }
    float *cudaTemp = (float *)scratch;
    float *compressedScores = (float *)(scratch + tempBytes);
    float *compressedKVFloat = kvFloatBytes == 0 ? nullptr : (float *)(scratch + tempBytes + scoreBytes);

    bool ok = true;
    int rowsPerBatch = seqlen * heads;
    for (int rowOffset = 0; rowOffset < rows && ok; rowOffset += rowsPerChunk) {
        int chunkRows = std::min(rowsPerChunk, rows - rowOffset);
        for (int localStart = 0; localStart < chunkRows && ok;) {
            int globalRow = rowOffset + localStart;
            int b = globalRow / rowsPerBatch;
            int batchEnd = std::min(rowOffset + chunkRows, (b + 1) * rowsPerBatch);
            int segmentRows = batchEnd - globalRow;
            float *segmentScores = compressedScores + (uint64_t)localStart * compressedCount;
            float *segmentTemp = cudaTemp + (uint64_t)localStart * dim;
            if (q.dataType == fastllm::DataType::BFLOAT16) {
                ok = DeepSeekV4LaunchSparsePrefillCompressedCublasSegment<__nv_bfloat16, __nv_bfloat16>(
                    q, kv, cudaSink, segmentScores, compressedKVFloat, segmentTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos,
                    realPrefixLen, compressedStart, compressedCount, softmaxScale,
                    globalRow, segmentRows);
            } else if (q.dataType == fastllm::DataType::FLOAT16) {
                ok = DeepSeekV4LaunchSparsePrefillCompressedCublasSegment<half, half>(
                    q, kv, cudaSink, segmentScores, compressedKVFloat, segmentTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos,
                    realPrefixLen, compressedStart, compressedCount, softmaxScale,
                    globalRow, segmentRows);
            } else if (q.dataType == fastllm::DataType::FLOAT32) {
                ok = DeepSeekV4LaunchSparsePrefillCompressedCublasSegment<float, float>(
                    q, kv, cudaSink, segmentScores, compressedKVFloat, segmentTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos,
                    realPrefixLen, compressedStart, compressedCount, softmaxScale,
                    globalRow, segmentRows);
            } else {
                ok = false;
            }
            localStart += segmentRows;
        }
        if (!ok) {
            break;
        }
        int threads = 128;
        int blocks = (chunkRows + threads - 1) / threads;
        DeepSeekV4SparsePrefillRotaryCastKernel<<<blocks, threads>>>(
            cudaTemp, (__nv_bfloat16 *)output.cudaData, chunkRows, rowOffset, seqlen, heads, dim,
            ropeDim, ropeBase, startPos, originalSeqLen, ropeFactor, betaFast, betaSlow);
    }
    if (ok) {
        DeviceSync();
    } else if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
        std::fprintf(stderr, "DeepSeekV4SparsePrefill compressed cuBLAS path fell back to row-wise kernel.\n");
    }

    FastllmCudaFree(scratch);
    return ok;
}

template <typename QT, typename KT>
bool DeepSeekV4LaunchSparsePrefillLocalCublasSegment(
        const fastllm::Data &q, const fastllm::Data &kv, const float *cudaSink,
        float *localScores, float *compressedScores,
        __nv_bfloat16 *localScoresBf16, __nv_bfloat16 *compressedScoresBf16,
        float *rawKVFloat,
        float *compressedKVFloat, float *outData,
        int bsz, int seqlen, int heads, int dim, int kvLen,
        int windowSize, int compressRatio, int startPos, int prefixLen,
        int rawStart, int rawKeyCount, int compressedStart, int compressedCount,
        int compressedKVStride, float softmaxScale, int rowOffset, int rows,
        bool compressedKVFloatReady) {
    if (rows <= 0 || rawKeyCount <= 0 || q.dataType != kv.dataType) {
        return false;
    }
    if (compressedCount < 0 || compressedCount > compressedKVStride) {
        return false;
    }

    int rowsPerBatch = seqlen * heads;
    int b = rowOffset / rowsPerBatch;
    if (b < 0 || b >= bsz || rowOffset + rows > (b + 1) * rowsPerBatch) {
        return false;
    }

    const QT *qPtr = (const QT *)q.cudaData + (uint64_t)rowOffset * dim;
    const KT *rawKVPtr = (const KT *)kv.cudaData + ((uint64_t)b * kvLen + rawStart) * dim;
    const KT *compressedKVPtr = compressedCount > 0 ?
        (const KT *)kv.cudaData + ((uint64_t)b * kvLen + compressedStart) * dim : nullptr;
    bool useBf16TensorGemm = q.dataType == fastllm::DataType::BFLOAT16 &&
                             localScoresBf16 != nullptr &&
                             (compressedCount == 0 || compressedScoresBf16 != nullptr);

    const float *qForGemm = nullptr;
    const float *rawKVForGemm = nullptr;
    const float *compressedKVForGemm = nullptr;
    if (useBf16TensorGemm) {
        // BF16 Tensor Core path reads q/kv directly and keeps FP32 scores for softmax.
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        qForGemm = (const float *)qPtr;
        rawKVForGemm = (const float *)rawKVPtr;
        compressedKVForGemm = (const float *)compressedKVPtr;
    } else {
        if (rawKVFloat == nullptr) {
            return false;
        }
        int threads = 256;
        uint64_t qElems = (uint64_t)rows * dim;
        DeepSeekV4SparseDecodeConvertQKernel<QT><<<
            (int)((qElems + threads - 1) / threads), threads>>>(qPtr, outData, qElems);
        qForGemm = outData;

        uint64_t rawElems = (uint64_t)rawKeyCount * dim;
        DeepSeekV4SparseDecodeGatherKVKernel<KT><<<
            (int)((rawElems + threads - 1) / threads), threads>>>(
                (const float *)nullptr, rawKVPtr, rawKVFloat, 1, dim, 1,
                0, rawKeyCount, 0, rawKeyCount);
        rawKVForGemm = rawKVFloat;

        if (compressedCount > 0) {
            if (compressedKVFloat == nullptr) {
                return false;
            }
            if (compressedKVFloatReady) {
                compressedKVForGemm = compressedKVFloat + (uint64_t)b * compressedKVStride * dim;
            } else {
                uint64_t kvElems = (uint64_t)compressedCount * dim;
                DeepSeekV4SparseDecodeGatherKVKernel<KT><<<
                    (int)((kvElems + threads - 1) / threads), threads>>>(
                        (const float *)nullptr, compressedKVPtr, compressedKVFloat, 1, dim, 1,
                        0, compressedCount, 0, compressedCount);
                compressedKVForGemm = compressedKVFloat;
            }
        }
    }

    float alpha = softmaxScale;
    float beta = 0.0f;
    auto handle = getFastllmCublasHandle();
    cublasStatus_t status;
    if (useBf16TensorGemm) {
        status = DeepSeekV4CublasBf16GemmToFloat(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            rawKeyCount, rows, dim,
            &alpha,
            (const __nv_bfloat16 *)rawKVPtr, dim,
            (const __nv_bfloat16 *)qPtr, dim,
            &beta,
            localScores, rawKeyCount);
    } else {
        status = DeepSeekV4CublasSgemmStrict(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            rawKeyCount, rows, dim,
            &alpha,
            rawKVForGemm, dim,
            qForGemm, dim,
            &beta,
            localScores, rawKeyCount);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparsePrefill local cuBLAS QK failed: status=%d rows=%d dim=%d rawKeys=%d type=%d\n",
                         (int)status, rows, dim, rawKeyCount, (int)q.dataType);
        }
        return false;
    }

    if (compressedCount > 0) {
        if (useBf16TensorGemm) {
            status = DeepSeekV4CublasBf16GemmToFloat(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                compressedCount, rows, dim,
                &alpha,
                (const __nv_bfloat16 *)compressedKVPtr, dim,
                (const __nv_bfloat16 *)qPtr, dim,
                &beta,
                compressedScores, compressedCount);
        } else {
            status = DeepSeekV4CublasSgemmStrict(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                compressedCount, rows, dim,
                &alpha,
                compressedKVForGemm, dim,
                qForGemm, dim,
                &beta,
                compressedScores, compressedCount);
        }
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
                std::fprintf(stderr,
                             "DeepSeekV4SparsePrefill local cuBLAS compressed QK failed: status=%d rows=%d dim=%d compressed=%d type=%d\n",
                             (int)status, rows, dim, compressedCount, (int)q.dataType);
            }
            return false;
        }
    }

    DeepSeekV4SparsePrefillCublasLocalSoftmaxKernel<<<rows, 256, 256 * sizeof(float)>>>(
        localScores, compressedCount > 0 ? compressedScores : nullptr,
        useBf16TensorGemm ? localScoresBf16 : nullptr,
        useBf16TensorGemm && compressedCount > 0 ? compressedScoresBf16 : nullptr,
        cudaSink,
        bsz, seqlen, heads, rawStart, rawKeyCount, compressedCount, windowSize, compressRatio,
        startPos, prefixLen, kvLen, rowOffset, rows);

    float avAlpha = 1.0f;
    float avBeta = 0.0f;
    if (useBf16TensorGemm) {
        status = DeepSeekV4CublasBf16GemmToFloat(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            dim, rows, rawKeyCount,
            &avAlpha,
            (const __nv_bfloat16 *)rawKVPtr, dim,
            localScoresBf16, rawKeyCount,
            &avBeta,
            outData, dim);
    } else {
        status = DeepSeekV4CublasSgemmStrict(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            dim, rows, rawKeyCount,
            &avAlpha,
            rawKVForGemm, dim,
            localScores, rawKeyCount,
            &avBeta,
            outData, dim);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparsePrefill local cuBLAS AV failed: status=%d rows=%d dim=%d rawKeys=%d type=%d\n",
                         (int)status, rows, dim, rawKeyCount, (int)kv.dataType);
        }
        return false;
    }

    if (compressedCount > 0) {
        avBeta = 1.0f;
        if (useBf16TensorGemm) {
            status = DeepSeekV4CublasBf16GemmToFloat(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dim, rows, compressedCount,
                &avAlpha,
                (const __nv_bfloat16 *)compressedKVPtr, dim,
                compressedScoresBf16, compressedCount,
                &avBeta,
                outData, dim);
        } else {
            status = DeepSeekV4CublasSgemmStrict(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dim, rows, compressedCount,
                &avAlpha,
                compressedKVForGemm, dim,
                compressedScores, compressedCount,
                &avBeta,
                outData, dim);
        }
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
                std::fprintf(stderr,
                             "DeepSeekV4SparsePrefill local cuBLAS compressed AV failed: status=%d rows=%d dim=%d compressed=%d type=%d\n",
                             (int)status, rows, dim, compressedCount, (int)kv.dataType);
            }
            return false;
        }
    }
    return true;
}

bool DeepSeekV4RunSparsePrefillLocalCublas(
        const fastllm::Data &q, const fastllm::Data &kv, const float *cudaSink,
        fastllm::Data &output,
        int bsz, int seqlen, int heads, int dim, int kvLen,
        int windowSize, int compressRatio, int startPos, int prefixLen,
        int ropeDim, float ropeBase, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, float softmaxScale) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUBLAS_SPARSE_PREFILL_LOCAL") != nullptr) {
        return false;
    }
    if (q.dataType != kv.dataType || windowSize <= 0 || dim <= 0 || heads <= 0 || seqlen <= 0) {
        return false;
    }

    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    int rawTotal = realPrefixLen + seqlen;
    int compressedStart = realPrefixLen + seqlen;
    int compressedCount = std::max(0, kvLen - compressedStart);
    int maxCompressed = kDeepSeekV4Sparse1MCompressedKeys;
    if (const char *env = std::getenv("FASTLLM_DSV4_CUBLAS_SPARSE_PREFILL_MAX_COMPRESSED")) {
        maxCompressed = std::max(1, std::atoi(env));
    }
    if (compressedCount > maxCompressed || rawTotal <= 0) {
        return false;
    }
    bool useBf16TensorGemm = q.dataType == fastllm::DataType::BFLOAT16 &&
                             std::getenv("FASTLLM_DSV4_DISABLE_BF16_SPARSE_PREFILL_GEMM") == nullptr;

    int tokenBlock = 64;
    if (const char *env = std::getenv("FASTLLM_DSV4_CUBLAS_SPARSE_PREFILL_LOCAL_TOKENS")) {
        tokenBlock = std::max(1, std::atoi(env));
    }
    tokenBlock = std::min(tokenBlock, seqlen);
    size_t tempLimit = DeepSeekV4SparsePrefillTempBytesLimit();
    auto calcScratchBytes = [&](int tb) {
        int rowsMax = tb * heads;
        int maxRawKeys = std::min(rawTotal, windowSize + tb - 1);
        uint64_t tempFloats = (uint64_t)rowsMax * dim;
        uint64_t localScoreFloats = (uint64_t)rowsMax * maxRawKeys;
        uint64_t compressedScoreFloats = (uint64_t)rowsMax * compressedCount;
        uint64_t rawKVFloats = (q.dataType == fastllm::DataType::FLOAT32 || useBf16TensorGemm) ?
            0ULL : (uint64_t)maxRawKeys * dim;
        uint64_t compressedKVFloats = (q.dataType == fastllm::DataType::FLOAT32 ||
                                       useBf16TensorGemm || compressedCount <= 0) ?
            0ULL : (uint64_t)bsz * compressedCount * dim;
        uint64_t localScoreBf16Elems = useBf16TensorGemm ? (uint64_t)rowsMax * maxRawKeys : 0ULL;
        uint64_t compressedScoreBf16Elems = (useBf16TensorGemm && compressedCount > 0) ?
            (uint64_t)rowsMax * compressedCount : 0ULL;
        return (tempFloats + localScoreFloats + compressedScoreFloats + rawKVFloats + compressedKVFloats) *
                   sizeof(float) +
               (localScoreBf16Elems + compressedScoreBf16Elems) * sizeof(__nv_bfloat16);
    };
    while (tokenBlock > 1 && calcScratchBytes(tokenBlock) > tempLimit) {
        tokenBlock = std::max(1, tokenBlock / 2);
    }

    int rowsMax = tokenBlock * heads;
    int maxRawKeys = std::min(rawTotal, windowSize + tokenBlock - 1);
    size_t tempBytes = (size_t)rowsMax * dim * sizeof(float);
    size_t localScoreBytes = (size_t)rowsMax * maxRawKeys * sizeof(float);
    size_t compressedScoreBytes = (size_t)rowsMax * compressedCount * sizeof(float);
    size_t rawKVBytes = (q.dataType == fastllm::DataType::FLOAT32 || useBf16TensorGemm) ?
        0 : (size_t)maxRawKeys * dim * sizeof(float);
    size_t compressedKVBytes = (q.dataType == fastllm::DataType::FLOAT32 ||
                                useBf16TensorGemm || compressedCount <= 0) ?
        0 : (size_t)bsz * compressedCount * dim * sizeof(float);
    size_t localScoreBf16Bytes = useBf16TensorGemm ?
        (size_t)rowsMax * maxRawKeys * sizeof(__nv_bfloat16) : 0;
    size_t compressedScoreBf16Bytes = (useBf16TensorGemm && compressedCount > 0) ?
        (size_t)rowsMax * compressedCount * sizeof(__nv_bfloat16) : 0;
    size_t scratchBytes = tempBytes + localScoreBytes + compressedScoreBytes +
                          rawKVBytes + compressedKVBytes +
                          localScoreBf16Bytes + compressedScoreBf16Bytes;
    if (scratchBytes == 0) {
        return false;
    }
    uint8_t *scratch = (uint8_t *)FastllmCudaMalloc(scratchBytes);
    if (scratch == nullptr) {
        return false;
    }
    float *cudaTemp = (float *)scratch;
    float *localScores = (float *)(scratch + tempBytes);
    float *compressedScores = compressedCount > 0 ? (float *)(scratch + tempBytes + localScoreBytes) : nullptr;
    float *rawKVFloat = rawKVBytes == 0 ? nullptr :
        (float *)(scratch + tempBytes + localScoreBytes + compressedScoreBytes);
    float *compressedKVFloat = compressedKVBytes == 0 ? nullptr :
        (float *)(scratch + tempBytes + localScoreBytes + compressedScoreBytes + rawKVBytes);
    __nv_bfloat16 *localScoresBf16 = localScoreBf16Bytes == 0 ? nullptr :
        (__nv_bfloat16 *)(scratch + tempBytes + localScoreBytes + compressedScoreBytes +
                          rawKVBytes + compressedKVBytes);
    __nv_bfloat16 *compressedScoresBf16 = compressedScoreBf16Bytes == 0 ? nullptr :
        (__nv_bfloat16 *)(scratch + tempBytes + localScoreBytes + compressedScoreBytes +
                          rawKVBytes + compressedKVBytes + localScoreBf16Bytes);
    bool compressedKVFloatReady = false;
    if (compressedKVFloat != nullptr && compressedCount > 0) {
        uint64_t compressedElems = (uint64_t)bsz * compressedCount * dim;
        int threads = 256;
        if (kv.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparsePrefillCastCompressedKVKernel<<<
                (int)((compressedElems + threads - 1) / threads), threads>>>(
                    (const __nv_bfloat16 *)kv.cudaData, compressedKVFloat,
                    bsz, kvLen, compressedStart, compressedCount, dim);
            compressedKVFloatReady = true;
        } else if (kv.dataType == fastllm::DataType::FLOAT16) {
            DeepSeekV4SparsePrefillCastCompressedKVKernel<<<
                (int)((compressedElems + threads - 1) / threads), threads>>>(
                    (const half *)kv.cudaData, compressedKVFloat,
                    bsz, kvLen, compressedStart, compressedCount, dim);
            compressedKVFloatReady = true;
        }
    }

    bool ok = true;
    for (int b = 0; b < bsz && ok; b++) {
        for (int tokenStart = 0; tokenStart < seqlen && ok; tokenStart += tokenBlock) {
            int tokenCount = std::min(tokenBlock, seqlen - tokenStart);
            int rows = tokenCount * heads;
            int rowOffset = (b * seqlen + tokenStart) * heads;
            int rawStart = std::max(0, realPrefixLen + tokenStart - windowSize + 1);
            int rawEnd = std::min(rawTotal, realPrefixLen + tokenStart + tokenCount);
            int rawKeyCount = rawEnd - rawStart;
            if (rawKeyCount <= 0 || rawKeyCount > maxRawKeys) {
                ok = false;
                break;
            }
            int activeCompressedCount = 0;
            if (compressRatio > 0 && compressedCount > 0) {
                activeCompressedCount = std::min(compressedCount,
                                                 (startPos + tokenStart + tokenCount) / compressRatio);
            }
            if (q.dataType == fastllm::DataType::BFLOAT16) {
                ok = DeepSeekV4LaunchSparsePrefillLocalCublasSegment<__nv_bfloat16, __nv_bfloat16>(
                    q, kv, cudaSink, localScores, compressedScores, localScoresBf16, compressedScoresBf16,
                    rawKVFloat, compressedKVFloat, cudaTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                    rawStart, rawKeyCount, compressedStart, activeCompressedCount, compressedCount,
                    softmaxScale, rowOffset, rows, compressedKVFloatReady);
            } else if (q.dataType == fastllm::DataType::FLOAT16) {
                ok = DeepSeekV4LaunchSparsePrefillLocalCublasSegment<half, half>(
                    q, kv, cudaSink, localScores, compressedScores, nullptr, nullptr,
                    rawKVFloat, compressedKVFloat, cudaTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                    rawStart, rawKeyCount, compressedStart, activeCompressedCount, compressedCount,
                    softmaxScale, rowOffset, rows, compressedKVFloatReady);
            } else if (q.dataType == fastllm::DataType::FLOAT32) {
                ok = DeepSeekV4LaunchSparsePrefillLocalCublasSegment<float, float>(
                    q, kv, cudaSink, localScores, compressedScores, nullptr, nullptr,
                    rawKVFloat, compressedKVFloat, cudaTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                    rawStart, rawKeyCount, compressedStart, activeCompressedCount, compressedCount,
                    softmaxScale, rowOffset, rows, compressedKVFloatReady);
            } else {
                ok = false;
            }
            if (!ok) {
                break;
            }
            int threads = 128;
            int blocks = (rows + threads - 1) / threads;
            DeepSeekV4SparsePrefillRotaryCastKernel<<<blocks, threads>>>(
                cudaTemp, (__nv_bfloat16 *)output.cudaData, rows, rowOffset, seqlen, heads, dim,
                ropeDim, ropeBase, startPos, originalSeqLen, ropeFactor, betaFast, betaSlow);
        }
    }
    if (ok) {
        DeviceSync();
    } else if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
        std::fprintf(stderr, "DeepSeekV4SparsePrefill local cuBLAS path fell back to hybrid kernel.\n");
    }

    FastllmCudaFree(scratch);
    return ok;
}

template <typename QT, typename KT>
bool DeepSeekV4LaunchSparsePrefillByKV(const fastllm::Data &q, const fastllm::Data &kv,
                                       const float *cudaSink, float *outData,
                                       int bsz, int seqlen, int heads, int dim, int kvLen,
                                       int windowSize, int compressRatio, int startPos, int prefixLen,
                                       float softmaxScale, int rowOffset, int rows,
                                       bool nonCausalBlock) {
    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    int compressedCount = std::max(0, kvLen - realPrefixLen - seqlen);
    int maxKeys = (nonCausalBlock ? realPrefixLen + seqlen :
        std::min(windowSize, realPrefixLen + seqlen)) +
        (compressRatio > 0 ? compressedCount : 0);
    if (maxKeys <= 0 || maxKeys > kDeepSeekV4SparsePrefillMaxKeys || rows <= 0) {
        return false;
    }
    size_t sharedBytes = (size_t)maxKeys * sizeof(float);
    auto kernel = DeepSeekV4SparseAttentionPrefillBlockKernel<QT, KT>;
    if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
        return false;
    }
    kernel<<<rows, 256, sharedBytes>>>(
        (const QT *)q.cudaData, (const KT *)kv.cudaData, cudaSink, outData,
        bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen, softmaxScale,
        rowOffset, nonCausalBlock);
    return true;
}

template <typename QT>
bool DeepSeekV4LaunchSparsePrefillByQ(const fastllm::Data &q, const fastllm::Data &kv,
                                      const float *cudaSink, float *outData,
                                      int bsz, int seqlen, int heads, int dim, int kvLen,
                                      int windowSize, int compressRatio, int startPos, int prefixLen,
                                      float softmaxScale, int rowOffset, int rows,
                                      bool nonCausalBlock) {
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        return DeepSeekV4LaunchSparsePrefillByKV<QT, __nv_bfloat16>(
            q, kv, cudaSink, outData, bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio,
            startPos, prefixLen, softmaxScale, rowOffset, rows,
            nonCausalBlock);
    }
    if (kv.dataType == fastllm::DataType::FLOAT16) {
        return DeepSeekV4LaunchSparsePrefillByKV<QT, half>(
            q, kv, cudaSink, outData, bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio,
            startPos, prefixLen, softmaxScale, rowOffset, rows,
            nonCausalBlock);
    }
    if (kv.dataType == fastllm::DataType::FLOAT32) {
        return DeepSeekV4LaunchSparsePrefillByKV<QT, float>(
            q, kv, cudaSink, outData, bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio,
            startPos, prefixLen, softmaxScale, rowOffset, rows,
            nonCausalBlock);
    }
    return false;
}

bool DeepSeekV4LaunchHcPreCublasDots(const fastllm::Data &x, const fastllm::Data &hcFn,
                                     float *dotsData, int tokens, int flatDim,
                                     int mixHc, int dotsStride) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUDA_HCPRE_CUBLAS") != nullptr) {
        return false;
    }
    cudaDataType_t xType, wType;
    if (!DeepSeekV4CublasDataType(x.dataType, xType) ||
        !DeepSeekV4CublasDataType(hcFn.dataType, wType)) {
        return false;
    }
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        getFastllmCublasHandle(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        mixHc, tokens, flatDim,
        &alpha,
        hcFn.cudaData, wType, flatDim,
        x.cudaData, xType, flatDim,
        &beta,
        dotsData, CUDA_R_32F, dotsStride,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUDA_HCPRE_CUBLAS") != nullptr) {
            std::fprintf(stderr, "DeepSeekV4HcPre cuBLAS dots failed: status=%d tokens=%d flatDim=%d mixHc=%d xType=%d wType=%d\n",
                         (int)status, tokens, flatDim, mixHc, (int)x.dataType, (int)hcFn.dataType);
        }
        return false;
    }
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPreByWeight(const fastllm::Data &x, const fastllm::Data &hcFn,
                                   const fastllm::Data &hcScale, const fastllm::Data &hcBase,
                                   fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb,
                                   float *dotsData,
                                   int tokens, int dim, int hcMult, int sinkhornIters,
                                   float eps, float normEps) {
    int threads = 256;
    int mixHc = (2 + hcMult) * hcMult;
    int flatDim = hcMult * dim;
    int dotsStride = mixHc + 1;
    int dotParts = DeepSeekV4HcPreDotParts(flatDim, threads);
    int dotBlocks = tokens * dotsStride * dotParts;
    int finishParts = DeepSeekV4HcPreFinishParts(dim, threads);
    dim3 finishGrid(tokens, finishParts);
    size_t dotSharedBytes = (size_t)threads * sizeof(float);
    size_t finishSharedBytes = (size_t)(mixHc + hcMult + hcMult * hcMult + hcMult * 2) * sizeof(float);
    const XT *xData = (const XT *)x.cudaData;
    XT *yData = (XT *)y.cudaData;
    float *postData = (float *)post.cudaData;
    float *combData = (float *)comb.cudaData;
    const float *scaleData = (const float *)hcScale.cudaData;
    const float *baseData = (const float *)hcBase.cudaData;
    if (hcFn.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, threads, dotSharedBytes>>>(
            xData, (const __nv_bfloat16 *)hcFn.cudaData, dotsData, tokens, flatDim, mixHc, dotsStride, dotParts);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, threads, dotSharedBytes>>>(
            xData, (const half *)hcFn.cudaData, dotsData, tokens, flatDim, mixHc, dotsStride, dotParts);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, threads, dotSharedBytes>>>(
            xData, (const float *)hcFn.cudaData, dotsData, tokens, flatDim, mixHc, dotsStride, dotParts);
    } else {
        return false;
    }
    DeepSeekV4HcPreFinishKernel<<<finishGrid, threads, finishSharedBytes>>>(
        xData, dotsData, scaleData, baseData, yData, postData, combData,
        tokens, dim, hcMult, sinkhornIters, eps, normEps, dotsStride, dotParts);
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPreCublasByWeight(const fastllm::Data &x, const fastllm::Data &hcFn,
                                         const fastllm::Data &hcScale, const fastllm::Data &hcBase,
                                         fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb,
                                         float *dotsData,
                                         int tokens, int dim, int hcMult, int sinkhornIters,
                                         float eps, float normEps) {
    int threads = 256;
    int mixHc = (2 + hcMult) * hcMult;
    int flatDim = hcMult * dim;
    int dotsStride = mixHc + 1;
    if (!DeepSeekV4LaunchHcPreCublasDots(x, hcFn, dotsData, tokens, flatDim, mixHc, dotsStride)) {
        return false;
    }

    DeepSeekV4HcPreSqSumKernel<<<tokens, threads, threads * sizeof(float)>>>(
        (const XT *)x.cudaData, dotsData, tokens, flatDim, dotsStride);

    int finishParts = DeepSeekV4HcPreFinishParts(dim, threads);
    dim3 finishGrid(tokens, finishParts);
    size_t finishSharedBytes = (size_t)(mixHc + hcMult + hcMult * hcMult + hcMult * 2) * sizeof(float);
    DeepSeekV4HcPreFinishKernel<<<finishGrid, threads, finishSharedBytes>>>(
        (const XT *)x.cudaData, dotsData,
        (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
        (XT *)y.cudaData, (float *)post.cudaData, (float *)comb.cudaData,
        tokens, dim, hcMult, sinkhornIters, eps, normEps, dotsStride, 1);
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPreDotsByWeight(const fastllm::Data &x, const fastllm::Data &hcFn,
                                       fastllm::Data &dotsFloat, int tokens, int flatDim, int mixHc) {
    int total = tokens * mixHc;
    int threads = std::min(256, std::max(1, total));
    int blocks = (total + threads - 1) / threads;
    const XT *xData = (const XT *)x.cudaData;
    float *dotsData = (float *)dotsFloat.cudaData;
    if (hcFn.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPreDotsKernel<<<blocks, threads>>>(xData, (const __nv_bfloat16 *)hcFn.cudaData,
                                                       dotsData, tokens, flatDim, mixHc);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPreDotsKernel<<<blocks, threads>>>(xData, (const half *)hcFn.cudaData,
                                                       dotsData, tokens, flatDim, mixHc);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPreDotsKernel<<<blocks, threads>>>(xData, (const float *)hcFn.cudaData,
                                                       dotsData, tokens, flatDim, mixHc);
    } else {
        return false;
    }
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcHeadDotsByWeight(const fastllm::Data &x,
                                        const fastllm::Data &hcFn,
                                        float *dots, int tokens, int flatDim,
                                        int hcMult, int dotsStride) {
    constexpr int threads = 256;
    int blocks = tokens * hcMult;
    size_t sharedBytes = threads * sizeof(double);
    if (hcFn.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcHeadDotsKernel<<<blocks, threads, sharedBytes>>>(
            (const XT *)x.cudaData, (const __nv_bfloat16 *)hcFn.cudaData,
            dots, tokens, flatDim, hcMult, dotsStride);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcHeadDotsKernel<<<blocks, threads, sharedBytes>>>(
            (const XT *)x.cudaData, (const half *)hcFn.cudaData,
            dots, tokens, flatDim, hcMult, dotsStride);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcHeadDotsKernel<<<blocks, threads, sharedBytes>>>(
            (const XT *)x.cudaData, (const float *)hcFn.cudaData,
            dots, tokens, flatDim, hcMult, dotsStride);
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

extern "C" bool FastllmCudaDeepSeekV4PrepareMoeDownInput(
        const fastllm::Data &gateUp, fastllm::Data &downInput,
        const float *routeScales, float swigluLimit, bool quantize) {
    return DeepSeekV4PrepareMoeDownInputImpl(
        gateUp, downInput, routeScales, swigluLimit, quantize);
}

extern "C" bool FastllmCudaDeepSeekV4DsparkMarkovLocalArgmax(
        const float *baseLogits, const float *markovBias,
        int *packedCandidate, int vocabSize) {
    if (baseLogits == nullptr || markovBias == nullptr ||
        packedCandidate == nullptr || vocabSize <= 0) {
        return false;
    }
    DeepSeekV4DsparkMarkovLocalArgmaxKernel<256><<<1, 256>>>(
        baseLogits, markovBias, packedCandidate, vocabSize);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4DsparkMarkovPeerAvailable() {
    return DeepSeekV4DsparkMarkovPeerRuntimeEnabled();
}

extern "C" bool FastllmCudaDeepSeekV4DsparkMarkovLinearPeer(
        const float *peerLatent, const fastllm::Data &localWeight,
        float *localOutput, int hiddenSize, int localVocabSize) {
    if (!DeepSeekV4DsparkMarkovPeerRuntimeEnabled() ||
        peerLatent == nullptr || localOutput == nullptr ||
        localWeight.dataDevice != fastllm::DataDevice::CUDA ||
        localWeight.dataType != fastllm::DataType::FLOAT16 ||
        localWeight.cudaData == nullptr || localWeight.dims.size() != 2 ||
        localWeight.dims[0] != localVocabSize ||
        localWeight.dims[1] != hiddenSize || hiddenSize <= 0 ||
        localVocabSize <= 0) {
        return false;
    }
    DeepSeekV4DsparkMarkovLinearPeerKernel<256>
        <<<localVocabSize, 256>>>(
            peerLatent, (const half*)localWeight.cudaData, localOutput,
            hiddenSize, localVocabSize);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4DsparkMarkovSignal(
        uint32_t *signal, int step) {
    if (!DeepSeekV4DsparkMarkovPeerRuntimeEnabled() ||
        signal == nullptr || step < 0) {
        return false;
    }
    DeepSeekV4DsparkMarkovSignalKernel<<<1, 1>>>(signal, step);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4DsparkMarkovWaitPeer(
        const uint32_t *peerSignal, uint32_t *localSeen, int step) {
    if (!DeepSeekV4DsparkMarkovPeerRuntimeEnabled() || peerSignal == nullptr ||
        localSeen == nullptr || step < 0) {
        return false;
    }
    DeepSeekV4DsparkMarkovWaitPeerKernel<<<1, 1>>>(
        peerSignal, localSeen, step);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4DsparkMarkovCopyPeer(
        const uint32_t *peerSignal, uint32_t *localSeen, int step,
        const float *peerLatent, float *localLatent, int hiddenSize) {
    if (!DeepSeekV4DsparkMarkovPeerRuntimeEnabled() || peerSignal == nullptr ||
        localSeen == nullptr || step < 0 || peerLatent == nullptr ||
        localLatent == nullptr || hiddenSize <= 0) {
        return false;
    }
    DeepSeekV4DsparkMarkovCopyPeerKernel<256><<<1, 256>>>(
        peerSignal, localSeen, step, peerLatent, localLatent, hiddenSize);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4DsparkMarkovSelect(
        const int *packedCandidates, const int *globalOffsets,
        int ranks, int *proposalIds, float *previousId, int step) {
    if (packedCandidates == nullptr || globalOffsets == nullptr ||
        ranks <= 0 || proposalIds == nullptr || previousId == nullptr ||
        step < 0) {
        return false;
    }
    DeepSeekV4DsparkMarkovSelectKernel<<<1, 1>>>(
        packedCandidates, globalOffsets, ranks,
        proposalIds, previousId, step);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4DsparkMarkovSelectPeer(
        const uint64_t *peerCandidatePointers,
        const uint64_t *peerSignalPointers, uint32_t *localSeen,
        const int *globalOffsets, int ranks, int steps,
        int *proposalIds, float *previousId, int step) {
    if (!DeepSeekV4DsparkMarkovPeerRuntimeEnabled() ||
        peerCandidatePointers == nullptr || peerSignalPointers == nullptr ||
        localSeen == nullptr || globalOffsets == nullptr || ranks <= 0 ||
        steps <= 0 || proposalIds == nullptr || previousId == nullptr ||
        step < 0 || step >= steps) {
        return false;
    }
    DeepSeekV4DsparkMarkovSelectPeerKernel<<<1, 1>>>(
        peerCandidatePointers, peerSignalPointers, localSeen,
        globalOffsets, ranks, steps, proposalIds, previousId, step);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4DsparkPrepareTargetPeer(
        const uint32_t *peerSignal, uint32_t *localSeen,
        const int *peerProposalIds, int proposalCount,
        int anchorToken, int startPos, int32_t *decodeMeta,
        float *inputIds) {
    if (!DeepSeekV4DsparkMarkovPeerRuntimeEnabled() ||
        peerSignal == nullptr || localSeen == nullptr ||
        peerProposalIds == nullptr || proposalCount <= 0 ||
        decodeMeta == nullptr) {
        return false;
    }
    DeepSeekV4DsparkPrepareTargetPeerKernel<32><<<1, 32>>>(
        peerSignal, localSeen, peerProposalIds, proposalCount,
        anchorToken, startPos, decodeMeta, inputIds);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4DsparkAcceptPeer(
        const int *candidateIds, const float *candidateScores,
        const int *globalOffsets, const int *proposalIds,
        int ranks, int rows, int *result, uint32_t *readySignal) {
    if (!DeepSeekV4DsparkMarkovPeerRuntimeEnabled() ||
        candidateIds == nullptr || candidateScores == nullptr ||
        globalOffsets == nullptr || proposalIds == nullptr ||
        ranks <= 0 || ranks > 32 || rows <= 1 || rows > 32 ||
        result == nullptr || readySignal == nullptr) {
        return false;
    }
    DeepSeekV4DsparkAcceptPeerKernel<32><<<1, 32>>>(
        candidateIds, candidateScores, globalOffsets, proposalIds,
        ranks, rows, result, readySignal);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4DsparkPrepareDraftPeer(
        const uint32_t *peerSignal, uint32_t *localSeen,
        const int *peerResult, int baseCommittedTokens,
        const void *stageKv0, void *windowKv0,
        const void *stageKv1, void *windowKv1,
        const void *stageKv2, void *windowKv2,
        int rows, int windowSize, int headDim,
        int noiseTokenId, int proposalCount,
        int32_t *decodeMeta, float *inputIds) {
    if (!DeepSeekV4DsparkMarkovPeerRuntimeEnabled() ||
        peerSignal == nullptr || localSeen == nullptr ||
        peerResult == nullptr || stageKv0 == nullptr ||
        windowKv0 == nullptr || stageKv1 == nullptr ||
        windowKv1 == nullptr || stageKv2 == nullptr ||
        windowKv2 == nullptr || rows <= 1 || windowSize <= 0 ||
        headDim <= 0 || proposalCount <= 0 || decodeMeta == nullptr) {
        return false;
    }
    DeepSeekV4DsparkPrepareDraftPeerKernel<256><<<4, 256>>>(
        peerSignal, localSeen, peerResult, baseCommittedTokens,
        (const __nv_bfloat16*)stageKv0,
        (__nv_bfloat16*)windowKv0,
        (const __nv_bfloat16*)stageKv1,
        (__nv_bfloat16*)windowKv1,
        (const __nv_bfloat16*)stageKv2,
        (__nv_bfloat16*)windowKv2,
        rows, windowSize, headDim, noiseTokenId, proposalCount,
        decodeMeta, inputIds);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4HcPre(const fastllm::Data &x, fastllm::Data &hcFn,
                                           fastllm::Data &hcScale, fastllm::Data &hcBase,
                                           int hcMult, int sinkhornIters, float eps, float normEps,
                                           fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        hcScale.dataDevice != fastllm::DataDevice::CUDA || hcBase.dataDevice != fastllm::DataDevice::CUDA ||
        x.dims.size() != 4 || hcMult <= 0 || sinkhornIters <= 0 ||
        hcScale.dataType != fastllm::DataType::FLOAT32 ||
        hcBase.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    int tokens = bsz * seqlen;
    if (x.dims[2] != hcMult || hcFn.Count(0) != (uint64_t)mixHc * flatDim ||
        hcScale.Count(0) < 3 || hcBase.Count(0) < (uint64_t)mixHc) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(y, x.dataType, {bsz, seqlen, dim}) ||
        !DeepSeekV4PrepareCudaOutput(post, fastllm::DataType::FLOAT32, {bsz, seqlen, hcMult}) ||
        !DeepSeekV4PrepareCudaOutput(comb, fastllm::DataType::FLOAT32, {bsz, seqlen, hcMult, hcMult})) {
        return false;
    }

    int cublasMinTokens = 128;
    if (const char *env = std::getenv("FASTLLM_DSV4_HCPRE_CUBLAS_MIN_TOKENS")) {
        cublasMinTokens = std::max(1, std::atoi(env));
    }
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUDA_HCPRE_CUBLAS") == nullptr &&
        tokens >= cublasMinTokens) {
        float *cublasDots = (float *)FastllmCudaMalloc((size_t)tokens * (mixHc + 1) * sizeof(float));
        if (cublasDots != nullptr) {
            bool cublasOk = false;
            if (x.dataType == fastllm::DataType::BFLOAT16) {
                cublasOk = DeepSeekV4LaunchHcPreCublasByWeight<__nv_bfloat16>(
                    x, hcFn, hcScale, hcBase, y, post, comb, cublasDots,
                    tokens, dim, hcMult, sinkhornIters, eps, normEps);
            } else if (x.dataType == fastllm::DataType::FLOAT16) {
                cublasOk = DeepSeekV4LaunchHcPreCublasByWeight<half>(
                    x, hcFn, hcScale, hcBase, y, post, comb, cublasDots,
                    tokens, dim, hcMult, sinkhornIters, eps, normEps);
            } else if (x.dataType == fastllm::DataType::FLOAT32) {
                cublasOk = DeepSeekV4LaunchHcPreCublasByWeight<float>(
                    x, hcFn, hcScale, hcBase, y, post, comb, cublasDots,
                    tokens, dim, hcMult, sinkhornIters, eps, normEps);
            }
            DeviceSync();
            FastllmCudaFree(cublasDots);
            if (cublasOk) {
                return true;
            }
        }
    }

    int dotParts = DeepSeekV4HcPreDotParts(flatDim, 256);
    float *dotsData = (float *)FastllmCudaMalloc((size_t)tokens * (mixHc + 1) * dotParts * sizeof(float));
    if (dotsData == nullptr) {
        return false;
    }

    bool ok = false;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPreByWeight<__nv_bfloat16>(
            x, hcFn, hcScale, hcBase, y, post, comb, dotsData,
            tokens, dim, hcMult, sinkhornIters, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPreByWeight<half>(
            x, hcFn, hcScale, hcBase, y, post, comb, dotsData,
            tokens, dim, hcMult, sinkhornIters, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPreByWeight<float>(
            x, hcFn, hcScale, hcBase, y, post, comb, dotsData,
            tokens, dim, hcMult, sinkhornIters, eps, normEps);
    }
    DeviceSync();
    FastllmCudaFree(dotsData);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcPreNorm(const fastllm::Data &x,
                                                fastllm::Data &hcFn,
                                                fastllm::Data &hcScale,
                                                fastllm::Data &hcBase,
                                                fastllm::Data &normWeight,
                                                int hcMult, int sinkhornIters,
                                                float eps, float normEps,
                                                fastllm::Data &normOutput,
                                                fastllm::Data &post,
                                                fastllm::Data &comb) {
    // Cover ordinary one-token decode and DSpark-7's eight-row target
    // verification.  Other multi-token execution retains the established
    // graph-safe generic HcPre + RMSNorm path.
    const bool supportedRows =
        x.dims.size() == 4 && x.dims[0] == 1 &&
        (x.dims[1] == 1 || x.dims[1] == 8);
    if (x.dataDevice != fastllm::DataDevice::CUDA ||
        hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        hcScale.dataDevice != fastllm::DataDevice::CUDA ||
        hcBase.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        x.dataType != fastllm::DataType::BFLOAT16 ||
        hcFn.dataType != fastllm::DataType::FLOAT32 ||
        hcScale.dataType != fastllm::DataType::FLOAT32 ||
        hcBase.dataType != fastllm::DataType::FLOAT32 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        !supportedRows || x.dims[2] != 4 || x.dims[3] != 4096 ||
        hcMult != 4 || sinkhornIters <= 0) {
        return false;
    }

    int bsz = x.dims[0];
    int seqlen = x.dims[1];
    int tokens = bsz * seqlen;
    constexpr int dim = 4096;
    constexpr int flatDim = 4 * dim;
    constexpr int mixHc = 24;
    constexpr int dotsStride = mixHc + 1;
    int dotParts = DeepSeekV4HcPreDotParts(flatDim, 256);
    if (tokens <= 0 || hcFn.Count(0) != (uint64_t)mixHc * flatDim ||
        hcScale.Count(0) < 3 || hcBase.Count(0) < mixHc ||
        normWeight.Count(0) < dim) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(normOutput, fastllm::DataType::BFLOAT16,
                                     {bsz, seqlen, dim}) ||
        !DeepSeekV4PrepareCudaOutput(post, fastllm::DataType::FLOAT32,
                                     {bsz, seqlen, hcMult}) ||
        !DeepSeekV4PrepareCudaOutput(comb, fastllm::DataType::FLOAT32,
                                     {bsz, seqlen, hcMult, hcMult})) {
        return false;
    }

    size_t dotsBytes = (size_t)tokens * dotsStride * dotParts * sizeof(float);
    float *dots = (float *)FastllmCudaMalloc(dotsBytes);
    if (dots == nullptr) {
        return false;
    }
    int dotThreads = 256;
    int dotBlocks = tokens * dotsStride * dotParts;
    DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, dotThreads,
                                     dotThreads * sizeof(float)>>>(
        (const __nv_bfloat16 *)x.cudaData,
        (const float *)hcFn.cudaData, dots, tokens, flatDim,
        mixHc, dotsStride, dotParts);

    DeepSeekV4LaunchHcPreFinishNorm4x4096(
        (const __nv_bfloat16 *)x.cudaData, dots,
        (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
        (const float *)normWeight.cudaData,
        (__nv_bfloat16 *)normOutput.cudaData,
        (float *)post.cudaData, (float *)comb.cudaData,
        tokens, sinkhornIters, eps, normEps, dotsStride, dotParts);
    DeviceSync();
    FastllmCudaFree(dots);
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4HcPostPreNorm(
        const fastllm::Data &x, const fastllm::Data &residual,
        const fastllm::Data &previousPost,
        const fastllm::Data &previousComb, fastllm::Data &nextHcFn,
        fastllm::Data &nextHcScale, fastllm::Data &nextHcBase,
        fastllm::Data &nextNormWeight, int hcMult, int sinkhornIters,
        float eps, float normEps, fastllm::Data &residualOutput,
        fastllm::Data &normOutput, fastllm::Data &nextPost,
        fastllm::Data &nextComb) {
    // Keep this path deliberately narrow: ordinary one-token decode and the
    // eight-row DSpark-7 target verification use it; prefill and unrelated
    // multi-token shapes retain the established generic transition path.
    constexpr int dim = 4096;
    constexpr int flatDim = 4 * dim;
    constexpr int mixHc = 24;
    constexpr int dotsStride = mixHc + 1;
    const bool supportedRows =
        residual.dims.size() == 4 && residual.dims[0] == 1 &&
        (residual.dims[1] == 1 || residual.dims[1] == 8);
    if (x.dataDevice != fastllm::DataDevice::CUDA ||
        residual.dataDevice != fastllm::DataDevice::CUDA ||
        previousPost.dataDevice != fastllm::DataDevice::CUDA ||
        previousComb.dataDevice != fastllm::DataDevice::CUDA ||
        nextHcFn.dataDevice != fastllm::DataDevice::CUDA ||
        nextHcScale.dataDevice != fastllm::DataDevice::CUDA ||
        nextHcBase.dataDevice != fastllm::DataDevice::CUDA ||
        nextNormWeight.dataDevice != fastllm::DataDevice::CUDA ||
        x.dataType != fastllm::DataType::BFLOAT16 ||
        residual.dataType != fastllm::DataType::BFLOAT16 ||
        previousPost.dataType != fastllm::DataType::FLOAT32 ||
        previousComb.dataType != fastllm::DataType::FLOAT32 ||
        nextHcFn.dataType != fastllm::DataType::FLOAT32 ||
        nextHcScale.dataType != fastllm::DataType::FLOAT32 ||
        nextHcBase.dataType != fastllm::DataType::FLOAT32 ||
        nextNormWeight.dataType != fastllm::DataType::FLOAT32 ||
        !supportedRows || residual.dims[2] != 4 ||
        residual.dims[3] != dim ||
        hcMult != 4 || sinkhornIters <= 0) {
        return false;
    }

    int bsz = residual.dims[0];
    int seqlen = residual.dims[1];
    int tokens = bsz * seqlen;
    const bool useSm120PreciseDecode =
        tokens == 1 && FastllmCudaRuntimeArch() >= 120 &&
        DeepSeekV4CanLaunchHcPostPreCooperative() &&
        std::getenv("FASTLLM_DSV4_DISABLE_SM120_HC_PRECISE_DECODE") == nullptr;
    const bool useSm120VllmSchedule =
        tokens == 8 && FastllmCudaRuntimeArch() >= 120;
    const int dotParts = useSm120VllmSchedule ? 4 : 16;
    if (tokens <= 0 || x.Count(0) != (uint64_t)tokens * dim ||
        residual.Count(0) != (uint64_t)tokens * flatDim ||
        previousPost.Count(0) != (uint64_t)tokens * hcMult ||
        previousComb.Count(0) != (uint64_t)tokens * hcMult * hcMult ||
        nextHcFn.Count(0) != (uint64_t)mixHc * flatDim ||
        nextHcScale.Count(0) < 3 || nextHcBase.Count(0) < mixHc ||
        nextNormWeight.Count(0) < dim) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(
            residualOutput, fastllm::DataType::BFLOAT16,
            {bsz, seqlen, hcMult, dim}) ||
        !DeepSeekV4PrepareCudaOutput(
            normOutput, fastllm::DataType::BFLOAT16,
            {bsz, seqlen, dim}) ||
        !DeepSeekV4PrepareCudaOutput(
            nextPost, fastllm::DataType::FLOAT32,
            {bsz, seqlen, hcMult}) ||
        !DeepSeekV4PrepareCudaOutput(
            nextComb, fastllm::DataType::FLOAT32,
            {bsz, seqlen, hcMult, hcMult})) {
        return false;
    }

    size_t dotsBytes =
        (size_t)tokens * dotsStride * dotParts * sizeof(float);
    float *dots = (float *)FastllmCudaMalloc(dotsBytes);
    if (dots == nullptr) {
        return false;
    }

    dim3 transitionGrid(mixHc / 3, dotParts, tokens);
    if (useSm120PreciseDecode) {
        // The DSpark-oriented SM120 kernel intentionally follows vLLM's
        // float transition boundary.  Ordinary decode instead requires the
        // established double -> BF16 transition and split-K16 reduction
        // contract.  A cooperative grid materializes that rounded transition
        // once, synchronizes on-device, and then executes the established
        // reduction tree.  Keeping both phases in one kernel avoids adding a
        // CUDA Graph node for every transformer layer.
        const __nv_bfloat16 *xData =
            (const __nv_bfloat16 *)x.cudaData;
        const __nv_bfloat16 *residualData =
            (const __nv_bfloat16 *)residual.cudaData;
        const float *postData = (const float *)previousPost.cudaData;
        const float *combData = (const float *)previousComb.cudaData;
        const float *fnData = (const float *)nextHcFn.cudaData;
        __nv_bfloat16 *residualOutputData =
            (__nv_bfloat16 *)residualOutput.cudaData;
        void *kernelArgs[] = {
            &xData, &residualData, &postData, &combData, &fnData,
            &dots, &residualOutputData, &tokens};
        cudaError_t launchError = cudaLaunchCooperativeKernel(
            (const void *)DeepSeekV4HcPostPreDots4x4096Kernel<true>,
            transitionGrid, dim3(256), kernelArgs, 0, cudaStreamPerThread);
        if (launchError != cudaSuccess) {
            FastllmCudaFree(dots);
            return false;
        }
    } else if (useSm120VllmSchedule) {
        DeepSeekV4HcPostPreDots4x4096Sm120Kernel<<<transitionGrid, 256>>>(
            (const __nv_bfloat16 *)x.cudaData,
            (const __nv_bfloat16 *)residual.cudaData,
            (const float *)previousPost.cudaData,
            (const float *)previousComb.cudaData,
            (const float *)nextHcFn.cudaData, dots,
            (__nv_bfloat16 *)residualOutput.cudaData, tokens);
    } else {
        DeepSeekV4HcPostPreDots4x4096Kernel<false><<<transitionGrid, 256>>>(
            (const __nv_bfloat16 *)x.cudaData,
            (const __nv_bfloat16 *)residual.cudaData,
            (const float *)previousPost.cudaData,
            (const float *)previousComb.cudaData,
            (const float *)nextHcFn.cudaData, dots,
            (__nv_bfloat16 *)residualOutput.cudaData, tokens);
    }

    DeepSeekV4LaunchHcPreFinishNorm4x4096(
        (const __nv_bfloat16 *)residualOutput.cudaData, dots,
        (const float *)nextHcScale.cudaData,
        (const float *)nextHcBase.cudaData,
        (const float *)nextNormWeight.cudaData,
        (__nv_bfloat16 *)normOutput.cudaData,
        (float *)nextPost.cudaData, (float *)nextComb.cudaData,
        tokens, sinkhornIters, eps, normEps, dotsStride, dotParts);
    bool ok = cudaGetLastError() == cudaSuccess;
    DeviceSync();
    FastllmCudaFree(dots);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcPreDots(const fastllm::Data &x, const fastllm::Data &hcFn,
                                               int hcMult, fastllm::Data &dotsFloat) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        x.dims.size() != 4 || hcMult <= 0) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    int tokens = bsz * seqlen;
    if (x.dims[2] != hcMult || hcFn.Count(0) != (uint64_t)mixHc * flatDim) {
        return false;
    }
    dotsFloat.dataType = fastllm::DataType::FLOAT32;
    dotsFloat.Resize({tokens, mixHc});
    dotsFloat.Allocate(false);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(dotsFloat);
    dotsFloat.cudaData = cudaOutput;

    bool ok = false;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPreDotsByWeight<__nv_bfloat16>(x, hcFn, dotsFloat, tokens, flatDim, mixHc);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPreDotsByWeight<half>(x, hcFn, dotsFloat, tokens, flatDim, mixHc);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPreDotsByWeight<float>(x, hcFn, dotsFloat, tokens, flatDim, mixHc);
    }

    FastllmCudaFinishOutput(dotsFloat, cudaOutput);
    dotsFloat.cudaData = nullptr;
    dotsFloat.dataDevice = fastllm::DataDevice::CPU;
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcHead(const fastllm::Data &x,
                                             const fastllm::Data &hcFn,
                                             const fastllm::Data &hcScale,
                                             const fastllm::Data &hcBase,
                                             int hcMult, float eps, float normEps,
                                             fastllm::Data &output) {
    if (x.dataDevice != fastllm::DataDevice::CUDA ||
        hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        hcScale.dataDevice != fastllm::DataDevice::CUDA ||
        hcBase.dataDevice != fastllm::DataDevice::CUDA ||
        x.dims.size() != 4 || hcMult <= 0 || x.dims[2] != hcMult ||
        hcScale.dataType != fastllm::DataType::FLOAT32 ||
        hcBase.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int tokens = bsz * seqlen;
    int flatDim = hcMult * dim;
    if (hcFn.Count(0) != (uint64_t)hcMult * flatDim ||
        hcScale.Count(0) < 1 || hcBase.Count(0) < (uint64_t)hcMult ||
        !DeepSeekV4PrepareCudaOutput(output, x.dataType, {bsz, seqlen, dim})) {
        return false;
    }

    int dotsStride = hcMult + 1;
    float *dots = (float *)FastllmCudaMalloc((size_t)tokens * dotsStride * sizeof(float));
    if (dots == nullptr) {
        return false;
    }
    bool dotsOk = false;
    if (x.dataType == hcFn.dataType) {
        dotsOk = DeepSeekV4LaunchHcPreCublasDots(
            x, hcFn, dots, tokens, flatDim, hcMult, dotsStride);
    }
    if (!dotsOk && x.dataType == fastllm::DataType::BFLOAT16) {
        dotsOk = DeepSeekV4LaunchHcHeadDotsByWeight<__nv_bfloat16>(
            x, hcFn, dots, tokens, flatDim, hcMult, dotsStride);
    } else if (!dotsOk && x.dataType == fastllm::DataType::FLOAT16) {
        dotsOk = DeepSeekV4LaunchHcHeadDotsByWeight<half>(
            x, hcFn, dots, tokens, flatDim, hcMult, dotsStride);
    } else if (!dotsOk && x.dataType == fastllm::DataType::FLOAT32) {
        dotsOk = DeepSeekV4LaunchHcHeadDotsByWeight<float>(
            x, hcFn, dots, tokens, flatDim, hcMult, dotsStride);
    }
    if (!dotsOk) {
        FastllmCudaFree(dots);
        return false;
    }
    int threads = 256;
    bool ok = true;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPreSqSumKernel<<<tokens, threads, threads * sizeof(float)>>>(
            (const __nv_bfloat16 *)x.cudaData, dots, tokens, flatDim, dotsStride);
        DeepSeekV4HcHeadFinishKernel<<<tokens, threads, (hcMult + 1) * sizeof(float)>>>(
            (const __nv_bfloat16 *)x.cudaData, dots,
            (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
            (__nv_bfloat16 *)output.cudaData, tokens, dim, hcMult, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPreSqSumKernel<<<tokens, threads, threads * sizeof(float)>>>(
            (const half *)x.cudaData, dots, tokens, flatDim, dotsStride);
        DeepSeekV4HcHeadFinishKernel<<<tokens, threads, (hcMult + 1) * sizeof(float)>>>(
            (const half *)x.cudaData, dots,
            (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
            (half *)output.cudaData, tokens, dim, hcMult, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPreSqSumKernel<<<tokens, threads, threads * sizeof(float)>>>(
            (const float *)x.cudaData, dots, tokens, flatDim, dotsStride);
        DeepSeekV4HcHeadFinishKernel<<<tokens, threads, (hcMult + 1) * sizeof(float)>>>(
            (const float *)x.cudaData, dots,
            (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
            (float *)output.cudaData, tokens, dim, hcMult, eps, normEps);
    } else {
        ok = false;
    }
    DeviceSync();
    FastllmCudaFree(dots);
    return ok;
}

static bool FastllmCudaDeepSeekV4ScaleQRotaryImpl(fastllm::Data &q, int ropeDim, float ropeBase,
                                                  int startPos, const int32_t *decodeMeta,
                                                  int originalSeqLen, float ropeFactor,
                                                  int betaFast, int betaSlow, float eps) {
    if (q.dataDevice != fastllm::DataDevice::CUDA || q.dims.size() != 4 ||
        ropeDim <= 0 || ropeDim > q.dims[3]) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    int rows = bsz * seqlen * heads;
    int threads = 256;
    int blocks = rows;
    size_t sharedBytes = (size_t)threads * sizeof(float);
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4ScaleQRotaryBlockKernel<<<blocks, threads, sharedBytes>>>(
            (__nv_bfloat16 *)q.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, eps, decodeMeta);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4ScaleQRotaryBlockKernel<<<blocks, threads, sharedBytes>>>(
            (half *)q.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, eps, decodeMeta);
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4ScaleQRotaryBlockKernel<<<blocks, threads, sharedBytes>>>(
            (float *)q.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, eps, decodeMeta);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4ScaleQRotary(fastllm::Data &q, int ropeDim, float ropeBase,
                                                    int startPos, int originalSeqLen,
                                                    float ropeFactor, int betaFast, int betaSlow,
                                                    float eps) {
    return FastllmCudaDeepSeekV4ScaleQRotaryImpl(q, ropeDim, ropeBase, startPos, nullptr,
                                                 originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
}

extern "C" bool FastllmCudaDeepSeekV4ScaleQRotaryGraph(
                                                    fastllm::Data &q, int ropeDim, float ropeBase,
                                                    const int32_t *decodeMeta, int originalSeqLen,
                                                    float ropeFactor, int betaFast, int betaSlow,
                                                    float eps) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaDeepSeekV4ScaleQRotaryImpl(q, ropeDim, ropeBase, 0, decodeMeta,
                                                 originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
}

static bool FastllmCudaDeepSeekV4RotaryQuantImpl(fastllm::Data &x, int ropeDim, float ropeBase,
                                                 int startPos, const int32_t *decodeMeta,
                                                 int originalSeqLen, float ropeFactor,
                                                 int betaFast, int betaSlow, int quantDim,
                                                 int blockSize, int posStep) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || x.dims.size() < 3 || x.dims.size() > 4 ||
        ropeDim <= 0 || blockSize <= 0) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1];
    int heads = x.dims.size() == 4 ? x.dims[2] : 1;
    int dim = x.dims.size() == 4 ? x.dims[3] : x.dims[2];
    if (ropeDim > dim || quantDim < 0 || quantDim > dim) {
        return false;
    }
    int rows = bsz * seqlen * heads;
    int threads = 256;
    int blocks = rows;
    size_t sharedBytes = (size_t)threads * sizeof(float);
    bool realFp8 = DeepSeekV4SparseMlaSm120RuntimeEnabled();
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4RotaryQuantBlockKernel<<<blocks, threads, sharedBytes>>>(
            (__nv_bfloat16 *)x.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep,
            decodeMeta, realFp8);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4RotaryQuantBlockKernel<<<blocks, threads, sharedBytes>>>(
            (half *)x.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep,
            decodeMeta, realFp8);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4RotaryQuantBlockKernel<<<blocks, threads, sharedBytes>>>(
            (float *)x.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep,
            decodeMeta, realFp8);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4RotaryQuant(fastllm::Data &x, int ropeDim, float ropeBase,
                                                   int startPos, int originalSeqLen,
                                                   float ropeFactor, int betaFast, int betaSlow,
                                                   int quantDim, int blockSize, int posStep) {
    return FastllmCudaDeepSeekV4RotaryQuantImpl(x, ropeDim, ropeBase, startPos, nullptr,
                                                originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                quantDim, blockSize, posStep);
}

extern "C" bool FastllmCudaDeepSeekV4RotaryQuantGraph(
                                                   fastllm::Data &x, int ropeDim, float ropeBase,
                                                   const int32_t *decodeMeta, int originalSeqLen,
                                                   float ropeFactor, int betaFast, int betaSlow,
                                                   int quantDim, int blockSize, int posStep) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaDeepSeekV4RotaryQuantImpl(x, ropeDim, ropeBase, 0, decodeMeta,
                                                originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                quantDim, blockSize, posStep);
}

extern "C" bool FastllmCudaDeepSeekV4FusedQKVRopeCacheGraph(
                                                   fastllm::Data &q, fastllm::Data &kv,
                                                   fastllm::Data &kvNormWeight,
                                                   const int32_t *decodeMeta,
                                                   int ropeDim, float ropeBase,
                                                   int originalSeqLen, float ropeFactor,
                                                   int betaFast, int betaSlow, float eps,
                                                   int quantDim, int quantBlockSize,
                                                   int windowSize, fastllm::Data &windowKV) {
    // Keep this specialization intentionally narrow.  The caller retains the
    // established RMSNorm/rotary/cache-update sequence as the fallback.
    if (decodeMeta == nullptr || q.dataDevice != fastllm::DataDevice::CUDA ||
        kv.dataDevice != fastllm::DataDevice::CUDA ||
        kvNormWeight.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::BFLOAT16 ||
        kv.dataType != fastllm::DataType::BFLOAT16 ||
        kvNormWeight.dataType != fastllm::DataType::FLOAT32 ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        q.cudaData == nullptr || kv.cudaData == nullptr ||
        kvNormWeight.cudaData == nullptr || windowKV.cudaData == nullptr ||
        q.dims.size() != 4 || q.dims[0] != 1 || q.dims[1] <= 0 ||
        q.dims[1] > 64 ||
        q.dims[2] <= 0 || q.dims[2] > 64 || q.dims[3] != 512 ||
        kv.dims != std::vector<int>({1, q.dims[1], 1, 512}) ||
        kvNormWeight.Count(0) != 512 ||
        windowKV.dims != std::vector<int>({1, windowSize, 512}) ||
        ropeDim != 64 || quantDim != 448 || quantBlockSize != 64 ||
        windowSize <= 0 || eps <= 0.0f) {
        return false;
    }
    // Multi-row fusion is tuned and validated for the SM120 DSpark verifier;
    // older architectures retain the operator-composed compatibility path.
    if (q.dims[1] > 1 && FastllmCudaRuntimeArch() < 120) {
        return false;
    }
    int qDevice = GetPointerDeviceId(q.cudaData);
    int metaDevice = GetPointerDeviceId((void*)decodeMeta);
    if (qDevice < 0 || (metaDevice >= 0 && metaDevice != qDevice)) {
        return false;
    }

    constexpr int threads = 512;
    int seqlen = q.dims[1];
    int qHeads = q.dims[2];
    int blocks = seqlen * (qHeads + 1);
    DeepSeekV4FusedQKVRopeCache512Kernel<<<blocks, threads>>>(
        (__nv_bfloat16 *)q.cudaData, (__nv_bfloat16 *)kv.cudaData,
        (const float *)kvNormWeight.cudaData, (float *)windowKV.cudaData,
        decodeMeta, windowSize, seqlen, qHeads, ropeBase, originalSeqLen, ropeFactor,
        betaFast, betaSlow, eps, DeepSeekV4SparseMlaSm120RuntimeEnabled());
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4StoreWindowKVCache(const fastllm::Data &kv, int startPos,
                                                        int windowSize, fastllm::Data &windowKV) {
    if (kv.dataDevice != fastllm::DataDevice::CUDA || kv.dims.size() != 3 || kv.dims[1] <= 0 ||
        startPos < 0 || windowSize <= 0) {
        return false;
    }
    int bsz = kv.dims[0], seqlen = kv.dims[1], headDim = kv.dims[2];
    if (!DeepSeekV4PrepareCudaOutput(windowKV, fastllm::DataType::FLOAT32, {bsz, windowSize, headDim})) {
        return false;
    }
    int total = bsz * windowSize * headDim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4StoreWindowKVCacheKernel<<<blocks, threads>>>(
            (__nv_bfloat16 *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim, startPos, windowSize);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4StoreWindowKVCacheKernel<<<blocks, threads>>>(
            (half *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim, startPos, windowSize);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4StoreWindowKVCacheKernel<<<blocks, threads>>>(
            (float *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim, startPos, windowSize);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

static bool FastllmCudaDeepSeekV4UpdateWindowKVCacheImpl(const fastllm::Data &kv, int startPos,
                                                         const int32_t *decodeMeta, int windowSize,
                                                         fastllm::Data &windowKV) {
    if (kv.dataDevice != fastllm::DataDevice::CUDA || kv.dims.size() != 3 || kv.dims[1] <= 0 ||
        startPos < 0 || windowSize <= 0) {
        return false;
    }
    int bsz = kv.dims[0], seqlen = kv.dims[1], headDim = kv.dims[2];
    if (!DeepSeekV4PrepareCudaOutput(windowKV, fastllm::DataType::FLOAT32, {bsz, windowSize, headDim})) {
        return false;
    }
    int total = bsz * seqlen * headDim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4UpdateWindowKVCacheKernel
            <<<blocks, threads, 0, cudaStreamPerThread>>>(
            (__nv_bfloat16 *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim,
            startPos, windowSize, decodeMeta);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4UpdateWindowKVCacheKernel
            <<<blocks, threads, 0, cudaStreamPerThread>>>(
            (half *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim,
            startPos, windowSize, decodeMeta);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4UpdateWindowKVCacheKernel
            <<<blocks, threads, 0, cudaStreamPerThread>>>(
            (float *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim,
            startPos, windowSize, decodeMeta);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4UpdateWindowKVCache(const fastllm::Data &kv, int startPos,
                                                           int windowSize, fastllm::Data &windowKV) {
    return FastllmCudaDeepSeekV4UpdateWindowKVCacheImpl(kv, startPos, nullptr, windowSize, windowKV);
}

extern "C" bool FastllmCudaDeepSeekV4UpdateWindowKVCacheGraph(
                                                           const fastllm::Data &kv,
                                                           const int32_t *decodeMeta,
                                                           int windowSize, fastllm::Data &windowKV) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaDeepSeekV4UpdateWindowKVCacheImpl(kv, 0, decodeMeta, windowSize, windowKV);
}

extern "C" bool FastllmCudaDeepSeekV4AppendFullWindowKVCache(
        const fastllm::Data &kv, int appendTokens,
        fastllm::Data &windowKV) {
    if (kv.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        kv.cudaData == nullptr || windowKV.cudaData == nullptr ||
        kv.dims.size() != 3 || windowKV.dims.size() != 3 ||
        kv.dims[0] <= 0 || kv.dims[0] != windowKV.dims[0] ||
        kv.dims[2] <= 0 || kv.dims[2] != windowKV.dims[2] ||
        appendTokens <= 0 || appendTokens > kv.dims[1] ||
        kv.dataType != windowKV.dataType) {
        return false;
    }
    const int bsz = kv.dims[0];
    const int seqlen = kv.dims[1];
    const int headDim = kv.dims[2];
    const int windowSize = windowKV.dims[1];
    constexpr int threads = 256;
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4AppendFullWindowKVCacheKernel<<<
            bsz, threads, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)kv.cudaData,
            (__nv_bfloat16*)windowKV.cudaData, bsz, seqlen,
            headDim, windowSize, appendTokens);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4AppendFullWindowKVCacheKernel<<<
            bsz, threads, 0, cudaStreamPerThread>>>(
            (const half*)kv.cudaData, (half*)windowKV.cudaData,
            bsz, seqlen, headDim, windowSize, appendTokens);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4AppendFullWindowKVCacheKernel<<<
            bsz, threads, 0, cudaStreamPerThread>>>(
            (const float*)kv.cudaData, (float*)windowKV.cudaData,
            bsz, seqlen, headDim, windowSize, appendTokens);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4BuildWindowKVPrefix(const fastllm::Data &windowKV, int startPos,
                                                         int windowSize, int prefixLen,
                                                         fastllm::Data &output) {
    if (windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        windowKV.dims.size() != 3 || prefixLen <= 0 || startPos < prefixLen ||
        windowSize <= 0 || windowKV.dims[1] != windowSize) {
        return false;
    }
    int bsz = windowKV.dims[0], headDim = windowKV.dims[2];
    int sourceDevice = GetPointerDeviceId(windowKV.cudaData);
    if (sourceDevice < 0) {
        return false;
    }
    // Prefix snapshots can restore a replicated MultiCUDA cache as a single
    // physical CUDA tensor.  dataDeviceIds still describes the logical replica
    // set, so select the device from the actual pointer before launching this
    // direct (non-executor) CUDA operator.
    FastllmCudaSetDevice(sourceDevice);
    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::FLOAT32, {bsz, prefixLen, headDim})) {
        return false;
    }
    int total = bsz * prefixLen * headDim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    DeepSeekV4BuildWindowKVPrefixKernel<<<blocks, threads>>>(
        (const float *)windowKV.cudaData, (float *)output.cudaData,
        bsz, prefixLen, headDim, startPos, windowSize);
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4BuildCompressedKV(const fastllm::Data &kv,
                                                       const fastllm::Data &score,
                                                       const fastllm::Data &ape,
                                                       int rawTokenBase, int rawLen,
                                                       int blockStart, int blockCount,
                                                       int compressRatio, int headDim,
                                                       int wideDim, bool overlap,
                                                       fastllm::Data &output) {
    if (kv.dataDevice != fastllm::DataDevice::CUDA ||
        score.dataDevice != fastllm::DataDevice::CUDA ||
        ape.dataDevice != fastllm::DataDevice::CUDA ||
        ape.dataType != fastllm::DataType::FLOAT32 ||
        score.dataType != kv.dataType ||
        kv.dims.size() != 3 || score.dims != kv.dims ||
        rawLen <= 0 || blockCount <= 0 || compressRatio <= 0 ||
        headDim <= 0 || wideDim <= 0 || kv.dims[1] < rawLen ||
        kv.dims[2] != wideDim || rawTokenBase < 0) {
        return false;
    }
    int bsz = kv.dims[0];
    if (ape.Count(0) < (uint64_t)compressRatio * wideDim) {
        return false;
    }
    int firstNeededToken = blockStart * compressRatio;
    if (overlap && blockStart > 0) {
        firstNeededToken = (blockStart - 1) * compressRatio;
    }
    int lastNeededToken = (blockStart + blockCount) * compressRatio;
    if (rawTokenBase > firstNeededToken || rawTokenBase + rawLen < lastNeededToken) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::FLOAT32, {bsz, blockCount, headDim})) {
        return false;
    }
    uint64_t total = (uint64_t)bsz * blockCount * headDim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4BuildCompressedKVKernel<<<blocks, threads>>>(
            (const __nv_bfloat16 *)kv.cudaData, (const __nv_bfloat16 *)score.cudaData,
            (const float *)ape.cudaData, (float *)output.cudaData, bsz, rawTokenBase, rawLen,
            blockStart, blockCount, compressRatio, headDim, wideDim, overlap);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4BuildCompressedKVKernel<<<blocks, threads>>>(
            (const half *)kv.cudaData, (const half *)score.cudaData,
            (const float *)ape.cudaData, (float *)output.cudaData, bsz, rawTokenBase, rawLen,
            blockStart, blockCount, compressRatio, headDim, wideDim, overlap);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4BuildCompressedKVKernel<<<blocks, threads>>>(
            (const float *)kv.cudaData, (const float *)score.cudaData,
            (const float *)ape.cudaData, (float *)output.cudaData, bsz, rawTokenBase, rawLen,
            blockStart, blockCount, compressRatio, headDim, wideDim, overlap);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4FinalizeCompressedKV(
                                            const fastllm::Data &compressed,
                                            const fastllm::Data &normWeight,
                                            int blockStart, int compressRatio,
                                            int ropeDim, float ropeBase,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow,
                                            fastllm::Data &cache) {
    if (compressed.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        cache.dataDevice != fastllm::DataDevice::CUDA ||
        compressed.dataType != fastllm::DataType::FLOAT32 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        cache.dataType != fastllm::DataType::BFLOAT16 ||
        compressed.cudaData == nullptr || normWeight.cudaData == nullptr ||
        cache.cudaData == nullptr || compressed.dims.size() != 3 ||
        cache.dims.size() != 3 || blockStart < 0 || compressRatio <= 0) {
        return false;
    }
    int bsz = compressed.dims[0];
    int blockCount = compressed.dims[1];
    int headDim = compressed.dims[2];
    int totalBlocks = blockStart + blockCount;
    int cacheStride = cache.dims[1];
    if (cache.expansionDims.size() == 3) {
        cacheStride = cache.expansionDims[1];
    }
    if (bsz != 1 || blockCount <= 0 ||
        (headDim != 128 && headDim != 512) ||
        ropeDim <= 0 || ropeDim > headDim ||
        normWeight.Count(0) < (uint64_t)headDim ||
        cache.dims[0] != bsz || cache.dims[1] != totalBlocks ||
        cache.dims[2] != headDim || cacheStride < totalBlocks) {
        return false;
    }

    int threads = 512;
    size_t sharedBytes = (size_t)(headDim + 16 + 256) * sizeof(float);
    bool realFp8 = DeepSeekV4SparseMlaSm120RuntimeEnabled();
    DeepSeekV4FinalizeCompressedKVKernel<<<
        bsz * blockCount, threads, sharedBytes>>>(
            (const float*)compressed.cudaData,
            (const float*)normWeight.cudaData,
            (__nv_bfloat16*)cache.cudaData,
            bsz, blockCount, blockStart, cacheStride, compressRatio,
            headDim, ropeDim, ropeBase, originalSeqLen, ropeFactor,
            betaFast, betaSlow, realFp8);
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4CompactCompressorRaw(
                                            fastllm::Data &kv,
                                            fastllm::Data &score,
                                            int dropLen) {
    if (kv.dataDevice != fastllm::DataDevice::CUDA ||
        score.dataDevice != fastllm::DataDevice::CUDA ||
        kv.cudaData == nullptr || score.cudaData == nullptr ||
        kv.dataType != score.dataType || kv.dims.size() != 3 ||
        score.dims != kv.dims || kv.strides.size() != 3 ||
        score.strides != kv.strides || dropLen <= 0) {
        return false;
    }
    int bsz = kv.dims[0];
    int oldLen = kv.dims[1];
    int wideDim = kv.dims[2];
    int newLen = oldLen - dropLen;
    if (bsz <= 0 || wideDim <= 0 || newLen <= 0 || dropLen < newLen ||
        kv.strides[0] % wideDim != 0) {
        return false;
    }
    int rowCapacity = kv.strides[0] / wideDim;
    if (rowCapacity < oldLen) {
        return false;
    }

    uint64_t total = (uint64_t)bsz * newLen * wideDim * 2;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (kv.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4CompactCompressorRawKernel<<<blocks, threads>>>(
            (float*)kv.cudaData, (float*)score.cudaData, bsz, newLen,
            wideDim, rowCapacity, dropLen);
    } else if (kv.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4CompactCompressorRawKernel<<<blocks, threads>>>(
            (__nv_bfloat16*)kv.cudaData, (__nv_bfloat16*)score.cudaData,
            bsz, newLen, wideDim, rowCapacity, dropLen);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4CompactCompressorRawKernel<<<blocks, threads>>>(
            (half*)kv.cudaData, (half*)score.cudaData, bsz, newLen,
            wideDim, rowCapacity, dropLen);
    } else {
        return false;
    }
    DeviceSync();
    std::vector<int> dims = kv.dims;
    dims[1] = newLen;
    kv.Resize(dims);
    score.Resize(dims);
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4InitGraphRawRing(const fastllm::Data &raw,
                                                       int rawTokenBase,
                                                       fastllm::Data &ring) {
    if (raw.dataDevice != fastllm::DataDevice::CUDA ||
        ring.dataDevice != fastllm::DataDevice::CUDA ||
        raw.cudaData == nullptr || ring.cudaData == nullptr ||
        raw.dataType != ring.dataType || raw.dims.size() != 3 ||
        ring.dims.size() != 3 || raw.dims[0] != ring.dims[0] ||
        raw.dims[2] != ring.dims[2] || rawTokenBase < 0 || ring.dims[1] <= 0) {
        return false;
    }
    int bsz = raw.dims[0], rawLen = raw.dims[1], wideDim = raw.dims[2];
    int ringCapacity = ring.dims[1];
    uint64_t total = (uint64_t)bsz * rawLen * wideDim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (raw.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4InitGraphRawRingKernel<<<blocks, threads>>>(
            (const __nv_bfloat16 *)raw.cudaData, (__nv_bfloat16 *)ring.cudaData,
            bsz, rawLen, wideDim, rawTokenBase, ringCapacity);
    } else if (raw.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4InitGraphRawRingKernel<<<blocks, threads>>>(
            (const half *)raw.cudaData, (half *)ring.cudaData,
            bsz, rawLen, wideDim, rawTokenBase, ringCapacity);
    } else if (raw.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4InitGraphRawRingKernel<<<blocks, threads>>>(
            (const float *)raw.cudaData, (float *)ring.cudaData,
            bsz, rawLen, wideDim, rawTokenBase, ringCapacity);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

template <typename T>
static bool DeepSeekV4LaunchUpdateCompressedKVGraph(
                                            const fastllm::Data &kv,
                                            const fastllm::Data &score,
                                            const fastllm::Data &ape,
                                            const fastllm::Data &normWeight,
                                            const int32_t *decodeMeta,
                                            int compressRatio, int headDim,
                                            int ropeDim, float ropeBase,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow,
                                            fastllm::Data &kvRing,
                                            fastllm::Data &scoreRing,
                                            fastllm::Data &compressedKV,
                                            int bsz, int ringCapacity,
                                            int compressedCapacity, int wideDim,
                                            int seqlen) {
    uint64_t rawTotal = (uint64_t)bsz * seqlen * wideDim;
    int threads = 256;
    DeepSeekV4StoreGraphCompressorRawKernel<<<
        (int)((rawTotal + threads - 1) / threads), threads>>>(
            (const T *)kv.cudaData, (const T *)score.cudaData,
            (T *)kvRing.cudaData, (T *)scoreRing.cudaData,
            decodeMeta, bsz, seqlen, wideDim, ringCapacity);

    size_t sharedBytes = (size_t)(headDim + threads) * sizeof(float);
    bool realFp8 = DeepSeekV4SparseMlaSm120RuntimeEnabled();
    if (normWeight.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4UpdateCompressedKVGraphKernel<<<bsz * seqlen, threads, sharedBytes>>>(
            (const T *)kvRing.cudaData, (const T *)scoreRing.cudaData,
            (const float *)ape.cudaData, (const __nv_bfloat16 *)normWeight.cudaData,
            decodeMeta, (__nv_bfloat16 *)compressedKV.cudaData,
            bsz, ringCapacity, compressedCapacity, seqlen, compressRatio, headDim,
            wideDim, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast,
            betaSlow, realFp8);
    } else if (normWeight.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4UpdateCompressedKVGraphKernel<<<bsz * seqlen, threads, sharedBytes>>>(
            (const T *)kvRing.cudaData, (const T *)scoreRing.cudaData,
            (const float *)ape.cudaData, (const half *)normWeight.cudaData,
            decodeMeta, (__nv_bfloat16 *)compressedKV.cudaData,
            bsz, ringCapacity, compressedCapacity, seqlen, compressRatio, headDim,
            wideDim, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast,
            betaSlow, realFp8);
    } else if (normWeight.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4UpdateCompressedKVGraphKernel<<<bsz * seqlen, threads, sharedBytes>>>(
            (const T *)kvRing.cudaData, (const T *)scoreRing.cudaData,
            (const float *)ape.cudaData, (const float *)normWeight.cudaData,
            decodeMeta, (__nv_bfloat16 *)compressedKV.cudaData,
            bsz, ringCapacity, compressedCapacity, seqlen, compressRatio, headDim,
            wideDim, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast,
            betaSlow, realFp8);
    } else {
        return false;
    }
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4UpdateCompressedKVGraph(
                                            const fastllm::Data &kv,
                                            const fastllm::Data &score,
                                            const fastllm::Data &ape,
                                            const fastllm::Data &normWeight,
                                            const int32_t *decodeMeta,
                                            int compressRatio, int headDim,
                                            int ropeDim, float ropeBase,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow,
                                            fastllm::Data &kvRing,
                                            fastllm::Data &scoreRing,
                                            fastllm::Data &compressedKV) {
    if (decodeMeta == nullptr || compressRatio <= 0 || headDim <= 0 ||
        ropeDim <= 0 || ropeDim > headDim || kv.dataDevice != fastllm::DataDevice::CUDA ||
        score.dataDevice != fastllm::DataDevice::CUDA ||
        ape.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        kvRing.dataDevice != fastllm::DataDevice::CUDA ||
        scoreRing.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKV.dataDevice != fastllm::DataDevice::CUDA ||
        kv.cudaData == nullptr || score.cudaData == nullptr || ape.cudaData == nullptr ||
        normWeight.cudaData == nullptr || kvRing.cudaData == nullptr ||
        scoreRing.cudaData == nullptr || compressedKV.cudaData == nullptr ||
        kv.dataType != score.dataType || kv.dataType != kvRing.dataType ||
        kv.dataType != scoreRing.dataType || ape.dataType != fastllm::DataType::FLOAT32 ||
        compressedKV.dataType != fastllm::DataType::BFLOAT16 ||
        kv.dims.size() != 3 || score.dims != kv.dims ||
        kvRing.dims.size() != 3 || scoreRing.dims != kvRing.dims ||
        compressedKV.dims.size() != 3) {
        return false;
    }
    int bsz = kv.dims[0];
    int seqlen = kv.dims[1];
    int wideDim = (compressRatio == 4 ? 2 : 1) * headDim;
    int ringCapacity = kvRing.dims[1];
    int compressedCapacity = compressedKV.dims[1];
    if (compressedKV.expansionDims.size() >= 3) {
        compressedCapacity = std::max(compressedCapacity, compressedKV.expansionDims[1]);
    }
    if (seqlen <= 0 || kv.dims[2] != wideDim ||
        kvRing.dims[0] != bsz || kvRing.dims[2] != wideDim ||
        ringCapacity < (compressRatio == 4 ? 2 * compressRatio : compressRatio) ||
        compressedCapacity <= 0 || compressedKV.dims[0] != bsz ||
        compressedKV.dims[2] != headDim ||
        ape.Count(0) < (uint64_t)compressRatio * wideDim ||
        normWeight.Count(0) < (uint64_t)headDim) {
        return false;
    }
    bool ok = false;
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchUpdateCompressedKVGraph<__nv_bfloat16>(
            kv, score, ape, normWeight, decodeMeta, compressRatio, headDim,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow,
            kvRing, scoreRing, compressedKV, bsz, ringCapacity,
            compressedCapacity, wideDim, seqlen);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchUpdateCompressedKVGraph<half>(
            kv, score, ape, normWeight, decodeMeta, compressRatio, headDim,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow,
            kvRing, scoreRing, compressedKV, bsz, ringCapacity,
            compressedCapacity, wideDim, seqlen);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchUpdateCompressedKVGraph<float>(
            kv, score, ape, normWeight, decodeMeta, compressRatio, headDim,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow,
            kvRing, scoreRing, compressedKV, bsz, ringCapacity,
            compressedCapacity, wideDim, seqlen);
    }
    DeviceSync();
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4RouteScoreTransform(fastllm::Data &logits, int scoreFuncMode) {
    if (logits.dataDevice != fastllm::DataDevice::CUDA || logits.dataType != fastllm::DataType::FLOAT32 ||
        logits.dims.empty() || scoreFuncMode < 0 || scoreFuncMode > 2) {
        return false;
    }
    int experts = logits.dims.back();
    int rows = (int)(logits.Count(0) / experts);
    if (experts > 256) {
        return false;
    }
    DeepSeekV4RouteScoreTransformKernel<<<rows, 256>>>((float *)logits.cudaData, rows, experts, scoreFuncMode);
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4SqrtSoftplusRouter(
        const fastllm::Data &logits, const fastllm::Data &gateBias,
        float routeScale, fastllm::Data &expertIndex,
        fastllm::Data &expertScore, bool allowTriton) {
    constexpr int experts = 256;
    constexpr int topk = 6;
    if (logits.dataDevice != fastllm::DataDevice::CUDA ||
        logits.dataType != fastllm::DataType::FLOAT32 ||
        logits.cudaData == nullptr || logits.dims.empty() ||
        logits.dims.back() != experts || logits.Count(0) == 0 ||
        logits.Count(0) % experts != 0 ||
        gateBias.dataDevice != fastllm::DataDevice::CUDA ||
        gateBias.dataType != fastllm::DataType::FLOAT32 ||
        gateBias.cudaData == nullptr || gateBias.Count(0) != experts ||
        !isfinite(routeScale)) {
        return false;
    }
    int tokens = (int)(logits.Count(0) / experts);
    if (!DeepSeekV4PrepareCudaOutput(
            expertIndex, fastllm::DataType::INT32, {tokens, topk}) ||
        !DeepSeekV4PrepareCudaOutput(
            expertScore, fastllm::DataType::FLOAT32, {tokens, topk})) {
        return false;
    }
#ifndef USE_ROCM
    if (allowTriton &&
        fastllm::FastllmCudaTryTritonDeepSeekV4SqrtSoftplusRouter(
            logits, gateBias, routeScale, expertIndex, expertScore)) {
        return true;
    }
    if (allowTriton && FastllmCudaRuntimeArch() >= 120 &&
        std::getenv(
            "FASTLLM_DSV4_REFERENCE_SQRTSOFTPLUS_ROUTER") == nullptr) {
        constexpr int rowsPerBlock = 4;
        dim3 threads(32, rowsPerBlock);
        DeepSeekV4SqrtSoftplusTop6Warp4Kernel<<<
            (tokens + rowsPerBlock - 1) / rowsPerBlock, threads>>>(
                (const float *)logits.cudaData,
                (const float *)gateBias.cudaData,
                (int32_t *)expertIndex.cudaData,
                (float *)expertScore.cudaData, tokens, routeScale);
        bool ok = cudaGetLastError() == cudaSuccess;
        DeviceSync();
        return ok;
    }
#else
    (void)allowTriton;
#endif
    DeepSeekV4SqrtSoftplusTop6GenericKernel<<<tokens, experts>>>(
        (const float *)logits.cudaData,
        (const float *)gateBias.cudaData,
        (int32_t *)expertIndex.cudaData,
        (float *)expertScore.cudaData, tokens, routeScale);
    bool ok = cudaGetLastError() == cudaSuccess;
    DeviceSync();
    return ok;
}

void FastllmCudaReleaseDeepSeekV4RouteTableCache(
        const fastllm::Data *routeTable) {
    if (routeTable == nullptr) {
        return;
    }

    DeepSeekV4RouteTableDeviceCache retired;
    {
        std::lock_guard<std::mutex> guard(DeepSeekV4RouteTableCacheMutex());
        DeepSeekV4RouteTableCache &cache = DeepSeekV4RouteTableCaches();
        auto it = cache.find(routeTable);
        if (it == cache.end()) {
            return;
        }
        retired = std::move(it->second);
        cache.erase(it);
    }

    int originalDevice = FastllmCudaGetDevice();
    for (auto &deviceEntry : retired) {
        FastllmCudaSetDevice(deviceEntry.first);
        DeepSeekV4RouteTableCacheEntry &entry = deviceEntry.second;
        FastllmCudaFree(entry.cudaData);
        for (void *ptr : entry.retiredCudaData) {
            FastllmCudaFree(ptr);
        }
    }
    FastllmCudaSetDevice(originalDevice);
}

extern "C" bool FastllmCudaDeepSeekV4HashRouteScore(const fastllm::Data &logits,
                                                    fastllm::Data &tid2eid,
                                                    const int *inputIds, int tokens,
                                                    int topk, int scoreFuncMode,
                                                    float routeScale,
                                                    fastllm::Data &expertIndex,
                                                    fastllm::Data &expertScore) {
    if (logits.dataDevice != fastllm::DataDevice::CUDA ||
        logits.dataType != fastllm::DataType::FLOAT32 ||
        logits.dims.empty() || inputIds == nullptr ||
        tokens <= 0 || topk <= 0 || scoreFuncMode < 0 || scoreFuncMode > 2 ||
        (tid2eid.dataType != fastllm::DataType::FLOAT32 &&
         tid2eid.dataType != fastllm::DataType::INT32 &&
         tid2eid.dataType != fastllm::DataType::INT32PARAM)) {
        return false;
    }
    int experts = logits.dims.back();
    int rows = (int)(logits.Count(0) / experts);
    if (rows != tokens || experts <= 0 || experts > 256) {
        return false;
    }
    int maxInputId = -1;
    for (int i = 0; i < tokens; i++) {
        if (inputIds[i] < 0) {
            return false;
        }
        maxInputId = std::max(maxInputId, inputIds[i]);
    }
    if (tid2eid.Count(0) < (uint64_t)(maxInputId + 1) * topk) {
        return false;
    }
    const void *cudaRouteTable = DeepSeekV4GetCudaRouteTable(tid2eid);
    bool intRouteTable = tid2eid.dataType == fastllm::DataType::INT32 ||
                         tid2eid.dataType == fastllm::DataType::INT32PARAM;
    if (cudaRouteTable == nullptr ||
        !DeepSeekV4PrepareCudaOutput(expertIndex, fastllm::DataType::INT32, {tokens, topk}) ||
        !DeepSeekV4PrepareCudaOutput(expertScore, fastllm::DataType::FLOAT32, {tokens, topk})) {
        return false;
    }

    int *cudaInputIds = nullptr;
    int singleTokenId = tokens == 1 ? inputIds[0] : -1;
    if (tokens > 1) {
        size_t inputBytes = (size_t)tokens * sizeof(int);
        cudaInputIds = (int *)FastllmCudaMalloc(inputBytes);
        if (cudaInputIds == nullptr) {
            return false;
        }
        FastllmCudaCopyFromHostToDevice(cudaInputIds, (void *)inputIds, inputBytes);
    }
    if (intRouteTable) {
        DeepSeekV4HashRouteScoreKernel<<<tokens, 256>>>(
            (float *)logits.cudaData, (const int32_t *)cudaRouteTable,
            cudaInputIds, singleTokenId,
            (int32_t *)expertIndex.cudaData, (float *)expertScore.cudaData,
            tokens, experts, topk, scoreFuncMode, routeScale,
            (int)(tid2eid.Count(0) / topk));
    } else {
        DeepSeekV4HashRouteScoreKernel<<<tokens, 256>>>(
            (float *)logits.cudaData, (const float *)cudaRouteTable,
            cudaInputIds, singleTokenId,
            (int32_t *)expertIndex.cudaData, (float *)expertScore.cudaData,
            tokens, experts, topk, scoreFuncMode, routeScale,
            (int)(tid2eid.Count(0) / topk));
    }
    DeviceSync();
    if (cudaInputIds != nullptr) {
        FastllmCudaFree(cudaInputIds);
    }
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4HashRouteScoreGraph(
                                                    const fastllm::Data &logits,
                                                    fastllm::Data &tid2eid,
                                                    const int32_t *decodeMeta,
                                                    int tokens, int topk,
                                                    int scoreFuncMode,
                                                    float routeScale,
                                                    fastllm::Data &expertIndex,
                                                    fastllm::Data &expertScore) {
    if (decodeMeta == nullptr || tokens <= 0 ||
        logits.dataDevice != fastllm::DataDevice::CUDA ||
        logits.dataType != fastllm::DataType::FLOAT32 || logits.dims.empty() ||
        topk <= 0 || scoreFuncMode < 0 || scoreFuncMode > 2 ||
        tid2eid.Count(0) < (uint64_t)topk ||
        (tid2eid.dataType != fastllm::DataType::FLOAT32 &&
         tid2eid.dataType != fastllm::DataType::INT32 &&
         tid2eid.dataType != fastllm::DataType::INT32PARAM)) {
        return false;
    }
    int experts = logits.dims.back();
    if ((int)(logits.Count(0) / experts) != tokens || experts <= 0 || experts > 256) {
        return false;
    }

    const void *cudaRouteTable = DeepSeekV4GetCudaRouteTable(tid2eid);
    bool intRouteTable = tid2eid.dataType == fastllm::DataType::INT32 ||
                         tid2eid.dataType == fastllm::DataType::INT32PARAM;
    if (cudaRouteTable == nullptr ||
        !DeepSeekV4PrepareCudaOutput(expertIndex, fastllm::DataType::INT32, {tokens, topk}) ||
        !DeepSeekV4PrepareCudaOutput(expertScore, fastllm::DataType::FLOAT32, {tokens, topk})) {
        return false;
    }

    int routeRows = (int)(tid2eid.Count(0) / topk);
    const int *cudaInputIds = (const int *)(decodeMeta + 1);
    if (intRouteTable) {
        DeepSeekV4HashRouteScoreKernel<<<tokens, 256>>>(
            (float *)logits.cudaData, (const int32_t *)cudaRouteTable,
            cudaInputIds, -1, (int32_t *)expertIndex.cudaData,
            (float *)expertScore.cudaData, tokens, experts, topk,
            scoreFuncMode, routeScale, routeRows);
    } else {
        DeepSeekV4HashRouteScoreKernel<<<tokens, 256>>>(
            (float *)logits.cudaData, (const float *)cudaRouteTable,
            cudaInputIds, -1, (int32_t *)expertIndex.cudaData,
            (float *)expertScore.cudaData, tokens, experts, topk,
            scoreFuncMode, routeScale, routeRows);
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionPrefill(const fastllm::Data &q,
                                                            const fastllm::Data &kv,
                                                            fastllm::Data &attnSink,
                                                            int windowSize, int startPos,
                                                            int compressRatio,
                                                            int ropeDim, float ropeBase,
                                                            int originalSeqLen,
                                                            float ropeFactor, int betaFast,
                                                            int betaSlow, float softmaxScale,
                                                            fastllm::Data &output, int prefixLen,
                                                            bool nonCausalBlock,
                                                            const int32_t *decodeMeta) {
    if (q.dataDevice != fastllm::DataDevice::CUDA ||
        kv.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        q.dims.size() != 4 || kv.dims.size() != 3 ||
        windowSize <= 0 || ropeDim <= 0 || ropeDim > q.dims[3]) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    int kvLen = kv.dims[1];
    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    if (seqlen <= 0 || kv.dims[0] != bsz || kv.dims[2] != dim ||
        kvLen < realPrefixLen + seqlen || attnSink.Count(0) != (uint64_t)heads) {
        return false;
    }
    int compressedCount = std::max(0, kvLen - realPrefixLen - seqlen);
    if (nonCausalBlock && compressRatio != 0) {
        return false;
    }
    int maxKeys = (nonCausalBlock ? realPrefixLen + seqlen :
        std::min(windowSize, realPrefixLen + seqlen)) +
        (compressRatio > 0 ? compressedCount : 0);
    if (maxKeys <= 0 || maxKeys > kDeepSeekV4SparsePrefillMaxKeys) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16,
                                     {bsz, seqlen, heads, dim})) {
        return false;
    }
    if (!nonCausalBlock && DeepSeekV4RunSparsePrefillLocalCublas(
            q, kv, (const float *)attnSink.cudaData, output,
            bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow, softmaxScale)) {
        return true;
    }
    if (!nonCausalBlock && DeepSeekV4RunSparsePrefillCompressedCublas(
            q, kv, (const float *)attnSink.cudaData, output,
            bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow, softmaxScale)) {
        return true;
    }

    int rows = bsz * seqlen * heads;
    size_t rowBytes = (size_t)dim * sizeof(float);
    size_t maxTempBytes = std::max(rowBytes, DeepSeekV4SparsePrefillTempBytesLimit());
    int rowsPerChunk = (int)std::max<size_t>(1, maxTempBytes / rowBytes);
    if (rowsPerChunk >= heads) {
        rowsPerChunk = std::max(heads, (rowsPerChunk / heads) * heads);
    }
    rowsPerChunk = std::min(rowsPerChunk, rows);
    size_t tempBytes = (size_t)rowsPerChunk * rowBytes;
    float *cudaTemp = (float *)FastllmCudaMalloc(tempBytes);
    if (cudaTemp == nullptr) {
        return false;
    }

    bool ok = true;
    for (int rowOffset = 0; rowOffset < rows; rowOffset += rowsPerChunk) {
        int chunkRows = std::min(rowsPerChunk, rows - rowOffset);
        bool launched = false;
        if (q.dataType == fastllm::DataType::BFLOAT16) {
            launched = DeepSeekV4LaunchSparsePrefillByQ<__nv_bfloat16>(
                q, kv, (const float *)attnSink.cudaData, cudaTemp,
                bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                softmaxScale, rowOffset, chunkRows, nonCausalBlock);
        } else if (q.dataType == fastllm::DataType::FLOAT16) {
            launched = DeepSeekV4LaunchSparsePrefillByQ<half>(
                q, kv, (const float *)attnSink.cudaData, cudaTemp,
                bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                softmaxScale, rowOffset, chunkRows, nonCausalBlock);
        } else if (q.dataType == fastllm::DataType::FLOAT32) {
            launched = DeepSeekV4LaunchSparsePrefillByQ<float>(
                q, kv, (const float *)attnSink.cudaData, cudaTemp,
                bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                softmaxScale, rowOffset, chunkRows, nonCausalBlock);
        }
        if (!launched) {
            ok = false;
            break;
        }
        int threads = 128;
        int blocks = (chunkRows + threads - 1) / threads;
        if (decodeMeta != nullptr) {
            int rotaryWorkPerRow = dim - ropeDim + (ropeDim >> 1);
            int rotaryBlocks =
                (chunkRows * rotaryWorkPerRow + threads - 1) / threads;
            DeepSeekV4SparseDecodeRotaryCastKernel<<<rotaryBlocks, threads>>>(
                cudaTemp, (__nv_bfloat16 *)output.cudaData,
                chunkRows, rowOffset, seqlen, heads, dim, ropeDim, ropeBase,
                startPos, originalSeqLen, ropeFactor, betaFast, betaSlow,
                decodeMeta);
        } else {
            DeepSeekV4SparsePrefillRotaryCastKernel<<<blocks, threads>>>(
                cudaTemp, (__nv_bfloat16 *)output.cudaData, chunkRows,
                rowOffset, seqlen, heads, dim, ropeDim, ropeBase, startPos,
                originalSeqLen, ropeFactor, betaFast, betaSlow);
        }
    }
    if (ok) {
        DeviceSync();
    }

    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4SparseMlaSm120Available() {
    return DeepSeekV4SparseMlaSm120RuntimeEnabled();
}

extern "C" bool FastllmCudaDeepSeekV4PrepareSparseMlaSm120Cache(
                                                        const fastllm::Data &windowKV,
                                                        int totalLen, int windowSize,
                                                        const fastllm::Data &compressedKV,
                                                        int compressedCount,
                                                        fastllm::Data &packedWindowKV,
                                                        fastllm::Data &packedCompressedKV) {
    if (!DeepSeekV4SparseMlaSm120RuntimeEnabled() || totalLen < 0 ||
        windowSize != 128 || compressedCount < 0 ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        windowKV.cudaData == nullptr || windowKV.dims.size() != 3 ||
        windowKV.dims[0] != 1 || windowKV.dims[1] != windowSize ||
        windowKV.dims[2] != kDeepSeekV4SparseMlaHeadDim ||
        packedWindowKV.dataDevice != fastllm::DataDevice::CUDA ||
        packedWindowKV.dataType != fastllm::DataType::INT8 ||
        packedWindowKV.cudaData == nullptr || packedWindowKV.dims.size() != 2 ||
        packedWindowKV.dims[1] != kDeepSeekV4SparseMlaPageBytes ||
        packedCompressedKV.dataDevice != fastllm::DataDevice::CUDA ||
        packedCompressedKV.dataType != fastllm::DataType::INT8 ||
        packedCompressedKV.cudaData == nullptr ||
        packedCompressedKV.dims.size() != 2 ||
        packedCompressedKV.dims[1] != kDeepSeekV4SparseMlaPageBytes) {
        return false;
    }
    int windowCapacity = packedWindowKV.dims[0] * kDeepSeekV4SparseMlaPageSize;
    int compressedCapacity =
        packedCompressedKV.dims[0] * kDeepSeekV4SparseMlaPageSize;
    if (totalLen > windowCapacity || compressedCount > compressedCapacity) {
        return false;
    }
    int sourceDevice = GetPointerDeviceId(windowKV.cudaData);
    if (sourceDevice < 0 ||
        GetPointerDeviceId(packedWindowKV.cudaData) != sourceDevice ||
        GetPointerDeviceId(packedCompressedKV.cudaData) != sourceDevice) {
        return false;
    }
    FastllmCudaSetDevice(sourceDevice);
    int liveWindow = std::min(totalLen, windowSize);
    if (liveWindow > 0) {
        DeepSeekV4PackSparseMlaWindowInitialKernel<<<liveWindow, 32>>>(
            (const float *)windowKV.cudaData,
            (uint8_t *)packedWindowKV.cudaData, totalLen, windowSize);
    }
    if (compressedCount > 0) {
        if (compressedKV.dataDevice != fastllm::DataDevice::CUDA ||
            compressedKV.dataType != fastllm::DataType::BFLOAT16 ||
            compressedKV.cudaData == nullptr || compressedKV.dims.size() != 3 ||
            compressedKV.dims[0] != 1 ||
            compressedKV.dims[2] != kDeepSeekV4SparseMlaHeadDim ||
            GetPointerDeviceId(compressedKV.cudaData) != sourceDevice) {
            return false;
        }
        DeepSeekV4PackSparseMlaLinearInitialKernel<<<compressedCount, 32>>>(
            (const __nv_bfloat16 *)compressedKV.cudaData,
            (uint8_t *)packedCompressedKV.cudaData, compressedCount);
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool FastllmCudaDeepSeekV4BuildIndexerTopKGraph(
                                                        const fastllm::Data &q,
                                                        const fastllm::Data &weights,
                                                        const fastllm::Data &compressedKV,
                                                        const int32_t *decodeMeta,
                                                        int compressRatio,
                                                        float ropeBase,
                                                        int originalSeqLen,
                                                        float ropeFactor,
                                                        int betaFast,
                                                        int betaSlow,
                                                        fastllm::Data &indices,
                                                        fastllm::Data &lengths) {
    if (decodeMeta == nullptr || compressRatio != 4 ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::BFLOAT16 ||
        q.cudaData == nullptr || q.dims.size() != 4 || q.dims[0] != 1 ||
        q.dims[2] != kDeepSeekV4IndexerHeads ||
        q.dims[3] != kDeepSeekV4IndexerHeadDim ||
        weights.dataDevice != fastllm::DataDevice::CUDA ||
        weights.cudaData == nullptr ||
        weights.Count(0) != (uint64_t)q.dims[1] *
                             kDeepSeekV4IndexerHeads ||
        compressedKV.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKV.dataType != fastllm::DataType::BFLOAT16 ||
        compressedKV.cudaData == nullptr || compressedKV.dims.size() != 3 ||
        compressedKV.dims[0] != 1 ||
        compressedKV.dims[2] != kDeepSeekV4IndexerHeadDim) {
        return false;
    }
    int device = GetPointerDeviceId(q.cudaData);
    if (device < 0 || GetPointerDeviceId(weights.cudaData) != device ||
        GetPointerDeviceId(compressedKV.cudaData) != device ||
        GetPointerDeviceId((void *)decodeMeta) != device) {
        return false;
    }
    FastllmCudaSetDevice(device);
    int seqlen = q.dims[1];
    int sourceCapacity = compressedKV.dims[1];
    if (compressedKV.expansionDims.size() >= 3) {
        sourceCapacity = max(sourceCapacity,
                             compressedKV.expansionDims[1]);
    }
    if (seqlen <= 0 || seqlen > 64 || sourceCapacity <= 0) {
        return false;
    }
    int kvCapacity = (sourceCapacity + 127) / 128 * 128;
    if (!DeepSeekV4PrepareCudaOutput(
            indices, fastllm::DataType::INT32,
            {seqlen, kDeepSeekV4IndexerTopK}) ||
        !DeepSeekV4PrepareCudaOutput(
            lengths, fastllm::DataType::INT32, {seqlen})) {
        return false;
    }

    auto align256 = [](size_t value) {
        return (value + 255) & ~(size_t)255;
    };
    size_t qBytes = (size_t)seqlen * kDeepSeekV4IndexerHeads *
                    kDeepSeekV4IndexerHeadDim;
    size_t kvBytes = (size_t)kvCapacity * kDeepSeekV4IndexerHeadDim;
    size_t scaleBytes = (size_t)kvCapacity * sizeof(float);
    size_t weightBytes = (size_t)seqlen * kDeepSeekV4IndexerHeads *
                         sizeof(float);
    size_t metaBytes = (size_t)seqlen * sizeof(uint32_t);
    size_t logitsBytes = (size_t)seqlen * kvCapacity * sizeof(float);
    size_t qOffset = 0;
    size_t kvOffset = align256(qOffset + qBytes);
    size_t scaleOffset = align256(kvOffset + kvBytes);
    size_t weightOffset = align256(scaleOffset + scaleBytes);
    size_t startsOffset = align256(weightOffset + weightBytes);
    size_t endsOffset = align256(startsOffset + metaBytes);
    size_t logitsOffset = align256(endsOffset + metaBytes);
    size_t scratchBytes = align256(logitsOffset + logitsBytes);
    uint8_t *scratch = (uint8_t *)FastllmCudaMalloc(scratchBytes);
    if (scratch == nullptr) {
        return false;
    }
    uint8_t *qFp8 = scratch + qOffset;
    uint8_t *kvFp8 = scratch + kvOffset;
    float *kvScales = (float *)(scratch + scaleOffset);
    float *foldedWeights = (float *)(scratch + weightOffset);
    uint32_t *starts = (uint32_t *)(scratch + startsOffset);
    uint32_t *ends = (uint32_t *)(scratch + endsOffset);
    float *logits = (float *)(scratch + logitsOffset);

    int qRows = seqlen * kDeepSeekV4IndexerHeads;
    if (weights.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4IndexerQuantQKernel<<<
            qRows, kDeepSeekV4IndexerHeadDim>>>(
            (const __nv_bfloat16 *)q.cudaData,
            (const __nv_bfloat16 *)weights.cudaData, decodeMeta,
            qFp8, foldedWeights, seqlen, ropeBase, originalSeqLen,
            ropeFactor, betaFast, betaSlow);
    } else if (weights.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4IndexerQuantQKernel<<<
            qRows, kDeepSeekV4IndexerHeadDim>>>(
            (const __nv_bfloat16 *)q.cudaData,
            (const half *)weights.cudaData, decodeMeta,
            qFp8, foldedWeights, seqlen, ropeBase, originalSeqLen,
            ropeFactor, betaFast, betaSlow);
    } else if (weights.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4IndexerQuantQKernel<<<
            qRows, kDeepSeekV4IndexerHeadDim>>>(
            (const __nv_bfloat16 *)q.cudaData,
            (const float *)weights.cudaData, decodeMeta,
            qFp8, foldedWeights, seqlen, ropeBase, originalSeqLen,
            ropeFactor, betaFast, betaSlow);
    } else {
        FastllmCudaFree(scratch);
        return false;
    }
    DeepSeekV4IndexerQuantKKernel<<<
        sourceCapacity, kDeepSeekV4IndexerHeadDim>>>(
        (const __nv_bfloat16 *)compressedKV.cudaData,
        kvFp8, kvScales, sourceCapacity);
    DeepSeekV4IndexerSeqMetaKernel<<<seqlen, 256>>>(
        decodeMeta, starts, ends, (int *)lengths.cudaData,
        (int32_t *)indices.cudaData, seqlen, compressRatio,
        sourceCapacity);

    bool sm120 = false;
#ifdef FASTLLM_ENABLE_DSV4_SPARSE_MLA_SM120
    sm120 = FastllmCudaDeepSeekV4IndexerLogitsSm120Raw(
        qFp8, kvFp8, kvScales, foldedWeights, starts, ends,
        logits, seqlen, kvCapacity, kvCapacity,
        (void *)cudaStreamPerThread);
#endif
    if (!sm120) {
        DeepSeekV4IndexerLogitsGenericKernel<<<
            dim3(seqlen, sourceCapacity), 256>>>(
            qFp8, kvFp8, kvScales, foldedWeights, ends, logits,
            seqlen, kvCapacity);
    }
    DeepSeekV4IndexerTopKKernel<<<
        seqlen, 512, kDeepSeekV4IndexerTopK * sizeof(int32_t)>>>(
        logits, ends, (int32_t *)indices.cudaData, seqlen, kvCapacity);
    bool ok = cudaGetLastError() == cudaSuccess;
    DeviceSync();
    FastllmCudaFree(scratch);
    return ok;
}

extern "C" size_t
FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraphSm120ScratchBytes(
        int seqlen, int heads, int compressRatio) {
#ifndef FASTLLM_ENABLE_DSV4_SPARSE_MLA_SM120
    (void)seqlen;
    (void)heads;
    (void)compressRatio;
    return 0;
#else
    if (seqlen <= 0 || seqlen > 64 || heads <= 0 || heads > 128 ||
        (compressRatio != 0 && compressRatio != 4 &&
         compressRatio != 128)) {
        return 0;
    }
    constexpr int mainTopk = 128;
    const int extraTopk = compressRatio > 0 ? 512 : 0;
    const int numSplits = 2 + (extraTopk > 0 ? 8 : 0);
    auto align256 = [](size_t value) {
        return (value + 255) & ~(size_t)255;
    };
    const size_t mainIndicesBytes =
        (size_t)seqlen * mainTopk * sizeof(int32_t);
    const size_t mainLengthsBytes = (size_t)seqlen * sizeof(int);
    const size_t extraIndicesBytes =
        (size_t)seqlen * extraTopk * sizeof(int32_t);
    const size_t extraLengthsBytes = extraTopk > 0 ?
        (size_t)seqlen * sizeof(int) : 0;
    const size_t midOutBytes =
        (size_t)seqlen * heads * numSplits *
        kDeepSeekV4SparseMlaHeadDim * sizeof(__nv_bfloat16);
    const size_t midLseBytes =
        (size_t)seqlen * heads * numSplits * sizeof(float);
    const size_t mainLengthsOffset = align256(mainIndicesBytes);
    const size_t extraIndicesOffset =
        align256(mainLengthsOffset + mainLengthsBytes);
    const size_t extraLengthsOffset =
        align256(extraIndicesOffset + extraIndicesBytes);
    const size_t midOutOffset =
        align256(extraLengthsOffset + extraLengthsBytes);
    const size_t midLseOffset = align256(midOutOffset + midOutBytes);
    return align256(midLseOffset + midLseBytes);
#endif
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraphSm120(
                                                        const fastllm::Data &q,
                                                        const fastllm::Data &windowKV,
                                                        const fastllm::Data &compressedKV,
                                                        const fastllm::Data *compressedIndices,
                                                        const fastllm::Data *compressedLengths,
                                                        fastllm::Data &packedWindowKV,
                                                        fastllm::Data &packedCompressedKV,
                                                        fastllm::Data *scratchWorkspace,
                                                        fastllm::Data &attnSink,
                                                        int windowSize, int compressRatio,
                                                        const int32_t *decodeMeta,
                                                        int ropeDim, float ropeBase,
                                                        int originalSeqLen,
                                                        float ropeFactor, int betaFast,
                                                        int betaSlow, float softmaxScale,
                                                        fastllm::Data &output) {
#ifndef FASTLLM_ENABLE_DSV4_SPARSE_MLA_SM120
    (void)q; (void)windowKV; (void)compressedKV; (void)packedWindowKV;
    (void)compressedIndices; (void)compressedLengths; (void)scratchWorkspace;
    (void)packedCompressedKV; (void)attnSink; (void)windowSize;
    (void)compressRatio; (void)decodeMeta; (void)ropeDim; (void)ropeBase;
    (void)originalSeqLen; (void)ropeFactor; (void)betaFast; (void)betaSlow;
    (void)softmaxScale; (void)output;
    return false;
#else
    if (!DeepSeekV4SparseMlaSm120RuntimeEnabled() || decodeMeta == nullptr ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::BFLOAT16 || q.cudaData == nullptr ||
        q.dims.size() != 4 || q.dims[0] != 1 || q.dims[1] <= 0 ||
        q.dims[1] > 64 || q.dims[3] != kDeepSeekV4SparseMlaHeadDim ||
        (q.dims[2] != 8 && q.dims[2] != 16 && q.dims[2] != 32 &&
         q.dims[2] != 64 &&
         q.dims[2] != 128) || windowSize != 128 ||
        (compressRatio != 0 && compressRatio != 4 && compressRatio != 128) ||
        ropeDim != kDeepSeekV4SparseMlaRopeDim ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        windowKV.cudaData == nullptr ||
        compressedKV.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKV.dataType != fastllm::DataType::BFLOAT16 ||
        compressedKV.cudaData == nullptr || compressedKV.dims.size() != 3 ||
        compressedKV.dims[0] != 1 ||
        compressedKV.dims[2] != kDeepSeekV4SparseMlaHeadDim ||
        packedWindowKV.dataDevice != fastllm::DataDevice::CUDA ||
        packedWindowKV.dataType != fastllm::DataType::INT8 ||
        packedWindowKV.cudaData == nullptr || packedWindowKV.dims.size() != 2 ||
        packedWindowKV.dims[1] != kDeepSeekV4SparseMlaPageBytes ||
        packedCompressedKV.dataDevice != fastllm::DataDevice::CUDA ||
        packedCompressedKV.dataType != fastllm::DataType::INT8 ||
        packedCompressedKV.cudaData == nullptr ||
        packedCompressedKV.dims.size() != 2 ||
        packedCompressedKV.dims[1] != kDeepSeekV4SparseMlaPageBytes ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        attnSink.cudaData == nullptr ||
        attnSink.Count(0) != (uint64_t)q.dims[2]) {
        return false;
    }
    int device = GetPointerDeviceId(q.cudaData);
    if (device < 0 || GetPointerDeviceId(windowKV.cudaData) != device ||
        GetPointerDeviceId(compressedKV.cudaData) != device ||
        GetPointerDeviceId(packedWindowKV.cudaData) != device ||
        GetPointerDeviceId(packedCompressedKV.cudaData) != device ||
        GetPointerDeviceId(attnSink.cudaData) != device ||
        GetPointerDeviceId((void *)decodeMeta) != device) {
        return false;
    }
    FastllmCudaSetDevice(device);
    int seqlen = q.dims[1];
    int heads = q.dims[2];
    const bool hasIndexerIndices = compressedIndices != nullptr ||
                                   compressedLengths != nullptr;
    if (hasIndexerIndices &&
        (compressedIndices == nullptr || compressedLengths == nullptr ||
         compressedIndices->dataDevice != fastllm::DataDevice::CUDA ||
         compressedIndices->dataType != fastllm::DataType::INT32 ||
         compressedIndices->cudaData == nullptr ||
         compressedIndices->dims != std::vector<int>({seqlen, 512}) ||
         compressedLengths->dataDevice != fastllm::DataDevice::CUDA ||
         compressedLengths->dataType != fastllm::DataType::INT32 ||
         compressedLengths->cudaData == nullptr ||
         compressedLengths->Count(0) != (uint64_t)seqlen ||
         GetPointerDeviceId(compressedIndices->cudaData) != device ||
         GetPointerDeviceId(compressedLengths->cudaData) != device)) {
        return false;
    }
    int compressedCapacity = compressedKV.dims[1];
    if (compressedKV.expansionDims.size() >= 3) {
        compressedCapacity = std::max(compressedCapacity,
                                      compressedKV.expansionDims[1]);
    }
    int packedWindowCapacity =
        packedWindowKV.dims[0] * kDeepSeekV4SparseMlaPageSize;
    int packedCompressedCapacity =
        packedCompressedKV.dims[0] * kDeepSeekV4SparseMlaPageSize;
    if (packedWindowCapacity <= 0 || packedCompressedCapacity <= 0 ||
        compressedCapacity > packedCompressedCapacity) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(
            output, fastllm::DataType::BFLOAT16,
            {1, seqlen, heads, kDeepSeekV4SparseMlaHeadDim})) {
        return false;
    }

    DeepSeekV4PackSparseMlaWindowUpdatesKernel<<<seqlen, 32>>>(
        (const float *)windowKV.cudaData,
        (uint8_t *)packedWindowKV.cudaData, decodeMeta, seqlen, windowSize,
        packedWindowCapacity);
    if (compressRatio > 0) {
        DeepSeekV4PackSparseMlaCompressedUpdatesKernel<<<seqlen, 32>>>(
            (const __nv_bfloat16 *)compressedKV.cudaData,
            (uint8_t *)packedCompressedKV.cudaData, decodeMeta, seqlen,
            compressRatio, packedCompressedCapacity);
    }

    constexpr int mainTopk = 128;
    int extraTopk = compressRatio > 0 ? 512 : 0;
    int numSplits = 2 + (extraTopk > 0 ? 8 : 0);
    size_t mainIndicesBytes = (size_t)seqlen * mainTopk * sizeof(int32_t);
    size_t mainLengthsBytes = (size_t)seqlen * sizeof(int);
    size_t extraIndicesBytes =
        (size_t)seqlen * extraTopk * sizeof(int32_t);
    size_t extraLengthsBytes = extraTopk > 0 ?
        (size_t)seqlen * sizeof(int) : 0;
    size_t midOutBytes = (size_t)seqlen * heads * numSplits *
                         kDeepSeekV4SparseMlaHeadDim *
                         sizeof(__nv_bfloat16);
    size_t midLseBytes =
        (size_t)seqlen * heads * numSplits * sizeof(float);
    auto align256 = [](size_t value) {
        return (value + 255) & ~(size_t)255;
    };
    size_t mainIndicesOffset = 0;
    size_t mainLengthsOffset = align256(mainIndicesBytes);
    size_t extraIndicesOffset =
        align256(mainLengthsOffset + mainLengthsBytes);
    size_t extraLengthsOffset =
        align256(extraIndicesOffset + extraIndicesBytes);
    size_t midOutOffset =
        align256(extraLengthsOffset + extraLengthsBytes);
    size_t midLseOffset = align256(midOutOffset + midOutBytes);
    size_t scratchBytes =
        FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraphSm120ScratchBytes(
            seqlen, heads, compressRatio);
    if (scratchBytes == 0 ||
        scratchBytes != align256(midLseOffset + midLseBytes)) {
        return false;
    }
    const bool ownsScratch = scratchWorkspace == nullptr;
    if (!ownsScratch &&
        (scratchWorkspace->dataDevice != fastllm::DataDevice::CUDA ||
         scratchWorkspace->dataType != fastllm::DataType::INT8 ||
         scratchWorkspace->cudaData == nullptr ||
         scratchWorkspace->Count(0) < scratchBytes ||
         GetPointerDeviceId(scratchWorkspace->cudaData) != device)) {
        return false;
    }
    uint8_t *scratch = ownsScratch ?
        (uint8_t *)FastllmCudaMalloc(scratchBytes) :
        (uint8_t *)scratchWorkspace->cudaData;
    if (scratch == nullptr) {
        return false;
    }
    int32_t *mainIndices =
        (int32_t *)(scratch + mainIndicesOffset);
    int *mainLengths = (int *)(scratch + mainLengthsOffset);
    int32_t *extraIndices = extraTopk > 0 ?
        (int32_t *)(scratch + extraIndicesOffset) : nullptr;
    int *extraLengths = extraTopk > 0 ?
        (int *)(scratch + extraLengthsOffset) : nullptr;
    __nv_bfloat16 *midOut =
        (__nv_bfloat16 *)(scratch + midOutOffset);
    float *midLse = (float *)(scratch + midLseOffset);
    DeepSeekV4BuildSparseMlaDecodeIndicesKernel<<<seqlen, 128>>>(
        decodeMeta, seqlen, compressRatio, mainIndices, mainLengths,
        hasIndexerIndices ? nullptr : extraIndices,
        hasIndexerIndices ? nullptr : extraLengths);

    const int32_t *activeExtraIndices = hasIndexerIndices ?
        (const int32_t *)compressedIndices->cudaData : extraIndices;
    const int *activeExtraLengths = hasIndexerIndices ?
        (const int *)compressedLengths->cudaData : extraLengths;

    bool ok = FastllmCudaDeepSeekV4SparseMlaSm120Raw(
        q.cudaData, (const uint8_t *)packedWindowKV.cudaData, mainIndices,
        midOut, midLse, mainLengths, output.cudaData,
        (const float *)attnSink.cudaData,
        extraTopk > 0 ? (const uint8_t *)packedCompressedKV.cudaData : nullptr,
        activeExtraIndices, activeExtraLengths, seqlen, heads,
        mainTopk, extraTopk,
        numSplits, softmaxScale, kDeepSeekV4SparseMlaPageBytes,
        extraTopk > 0 ? kDeepSeekV4SparseMlaPageBytes : 0,
        (void *)cudaStreamPerThread);
    if (ok) {
        int rows = seqlen * heads;
        int totalPairs = rows * (kDeepSeekV4SparseMlaRopeDim / 2);
        int threads = 128;
        DeepSeekV4SparseMlaInverseRotaryBf16Kernel<<<
            (totalPairs + threads - 1) / threads, threads>>>(
            (__nv_bfloat16 *)output.cudaData, decodeMeta, rows, seqlen,
            heads, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        DeviceSync();
    }
    if (ownsScratch) {
        FastllmCudaFree(scratch);
    }
    return ok;
#endif
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionDecodeCached(const fastllm::Data &q,
                                                                 const fastllm::Data &windowKV,
                                                                 const fastllm::Data &compressedKV,
                                                                 fastllm::Data &attnSink,
                                                                 int windowSize, int startPos,
                                                                 int compressedCount,
                                                                 int ropeDim, float ropeBase,
                                                                 int originalSeqLen,
                                                                 float ropeFactor, int betaFast,
                                                                 int betaSlow, float softmaxScale,
                                                                 fastllm::Data &output) {
    if (q.dims.size() != 4 || q.dims[1] != 1 ||
        windowSize <= 0 || compressedCount < 0 ||
        windowSize + compressedCount > kDeepSeekV4SparseDecodeMaxKeys ||
        ropeDim <= 0 || ropeDim > q.dims[3] ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        (compressedCount > 0 && compressedKV.dataDevice != fastllm::DataDevice::CUDA)) {
        return false;
    }
    int bsz = q.dims[0], heads = q.dims[2], dim = q.dims[3];
    if (attnSink.Count(0) != (uint64_t)heads ||
        windowKV.Count(0) != (uint64_t)bsz * windowSize * dim ||
        (compressedCount > 0 && compressedKV.Count(0) != (uint64_t)bsz * compressedCount * dim)) {
        return false;
    }
    int curDevice = FastllmCudaGetDevice();
    int qDevice = q.cudaData == nullptr ? -1 : GetPointerDeviceId(q.cudaData);
    int windowDevice = windowKV.cudaData == nullptr ? -1 : GetPointerDeviceId(windowKV.cudaData);
    int compressedDevice = compressedCount > 0 && compressedKV.cudaData != nullptr ?
                           GetPointerDeviceId(compressedKV.cudaData) : qDevice;
    int sinkDevice = attnSink.cudaData == nullptr ? -1 : GetPointerDeviceId(attnSink.cudaData);
    int kernelDevice = qDevice >= 0 ? qDevice : curDevice;
    if ((qDevice >= 0 && qDevice != curDevice) ||
        (windowDevice >= 0 && windowDevice != kernelDevice) ||
        (compressedCount > 0 && compressedDevice >= 0 && compressedDevice != kernelDevice) ||
        (sinkDevice >= 0 && sinkDevice != kernelDevice)) {
        return false;
    }
    const float *cudaWindow = (const float *)windowKV.cudaData;

    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16, {bsz, 1, heads, dim})) {
        return false;
    }

    size_t tempBytes = (size_t)bsz * heads * dim * sizeof(float);
    float *cudaTemp = (float *)FastllmCudaMalloc(tempBytes);

    bool ok = false;
    bool sparseDecodeStats = std::getenv("FASTLLM_DSV4_SPARSE_DECODE_STATS") != nullptr;
    auto sparseDecodeStart = std::chrono::steady_clock::now();
    int sparseDecodeMode = 0;
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<__nv_bfloat16>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, cudaTemp,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, &sparseDecodeMode);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<half>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, cudaTemp,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, &sparseDecodeMode);
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<float>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, cudaTemp,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, &sparseDecodeMode);
    }
    if (ok) {
        int rows = bsz * heads;
        int threads = 128;
        int rotaryWorkPerRow = dim - ropeDim + (ropeDim >> 1);
        int blocks = (rows * rotaryWorkPerRow + threads - 1) / threads;
        DeepSeekV4SparseDecodeRotaryCastKernel<<<blocks, threads>>>(
            cudaTemp, (__nv_bfloat16 *)output.cudaData, rows, 0, 1, heads,
            dim, ropeDim, ropeBase,
            startPos, originalSeqLen, ropeFactor, betaFast, betaSlow, nullptr);
        DeviceSync();
    }
    if (sparseDecodeStats) {
        auto sparseDecodeEnd = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(sparseDecodeEnd - sparseDecodeStart).count();
        int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
        int kvPasses = sparseDecodeMode == 0 ? 2 : 1;
        double windowBytes = (double)bsz * heads * dim * liveWindow * sizeof(float) * kvPasses;
        double compressedBytes = compressedCount > 0
                                     ? (double)bsz * heads * fastllm::GetDataBytes(compressedKV.dataType,
                                                                                   compressedCount, dim) * kvPasses
                                     : 0.0;
        double kvGB = (windowBytes + compressedBytes) / 1.0e9;
        static std::mutex statMutex;
        static unsigned long long statCalls = 0;
        static double statMs = 0.0;
        static double statKvGB = 0.0;
        std::lock_guard<std::mutex> statLock(statMutex);
        statCalls++;
        statMs += ms;
        statKvGB += kvGB;
        int printEvery = 43;
        if (const char *envEvery = std::getenv("FASTLLM_DSV4_SPARSE_DECODE_STATS_EVERY")) {
            printEvery = std::max(1, std::atoi(envEvery));
        }
        if (statCalls % (unsigned long long)printEvery == 0) {
            double avgMs = statMs / printEvery;
            double avgKvGB = statKvGB / printEvery;
            double avgGBps = avgMs > 0.0 ? avgKvGB / (avgMs / 1000.0) : 0.0;
            std::fprintf(stderr,
                         "DeepSeekV4SparseDecodeCachedStats calls=%llu last_ms=%.4f avg_ms=%.4f "
                         "last_kv_gb=%.6f avg_kv_gb=%.6f avg_kv_gbps=%.2f startPos=%d "
                         "liveWindow=%d compressedCount=%d dim=%d heads=%d kvPasses=%d mode=%s\n",
                         statCalls, ms, avgMs, kvGB, avgKvGB, avgGBps, startPos,
                         liveWindow, compressedCount, dim, heads, kvPasses,
                         sparseDecodeMode == 2 ? "cublas" : (sparseDecodeMode == 1 ? "online" : "two-pass"));
            std::fflush(stderr);
            statMs = 0.0;
            statKvGB = 0.0;
        }
    }

    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraph(
                                                                 const fastllm::Data &q,
                                                                 const fastllm::Data &windowKV,
                                                                 const fastllm::Data &compressedKV,
                                                                 const fastllm::Data *compressedIndices,
                                                                 const fastllm::Data *compressedLengths,
                                                                 fastllm::Data &attnSink,
                                                                 int windowSize, int compressRatio,
                                                                 const int32_t *decodeMeta,
                                                                 int ropeDim, float ropeBase,
                                                                 int originalSeqLen,
                                                                 float ropeFactor, int betaFast,
                                                                 int betaSlow, float softmaxScale,
                                                                 fastllm::Data &output,
                                                                 bool allowTriton) {
    if (decodeMeta == nullptr || q.dims.size() != 4 || q.dims[1] <= 0 ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        compressedKV.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKV.cudaData == nullptr ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        windowSize <= 0 || compressRatio < 0 || ropeDim <= 0 ||
        ropeDim > q.dims[3]) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1];
    int heads = q.dims[2], dim = q.dims[3];
    int compressedCapacity = compressedKV.dims.size() >= 3 ? compressedKV.dims[1] : 0;
    if (compressedKV.expansionDims.size() >= 3) {
        compressedCapacity = std::max(compressedCapacity, compressedKV.expansionDims[1]);
    }
    if (attnSink.Count(0) != (uint64_t)heads ||
        windowKV.Count(0) != (uint64_t)bsz * windowSize * dim ||
        compressedCapacity <= 0 || compressedKV.dims.size() < 3 ||
        compressedKV.dims[0] != bsz || compressedKV.dims[2] != dim ||
        windowSize + compressedCapacity > kDeepSeekV4SparseDecodeMaxKeys) {
        return false;
    }
    const bool hasIndexerIndices = compressedIndices != nullptr ||
                                   compressedLengths != nullptr;
    if (hasIndexerIndices &&
        (compressedIndices == nullptr || compressedLengths == nullptr ||
         compressedIndices->dataDevice != fastllm::DataDevice::CUDA ||
         compressedIndices->dataType != fastllm::DataType::INT32 ||
         compressedIndices->cudaData == nullptr ||
         compressedIndices->dims != std::vector<int>({seqlen, 512}) ||
         compressedLengths->dataDevice != fastllm::DataDevice::CUDA ||
         compressedLengths->dataType != fastllm::DataType::INT32 ||
         compressedLengths->cudaData == nullptr ||
         compressedLengths->Count(0) != (uint64_t)seqlen)) {
        return false;
    }

    int curDevice = FastllmCudaGetDevice();
    int qDevice = q.cudaData == nullptr ? -1 : GetPointerDeviceId(q.cudaData);
    int windowDevice = windowKV.cudaData == nullptr ? -1 : GetPointerDeviceId(windowKV.cudaData);
    int compressedDevice = GetPointerDeviceId(compressedKV.cudaData);
    int sinkDevice = attnSink.cudaData == nullptr ? -1 : GetPointerDeviceId(attnSink.cudaData);
    int metaDevice = GetPointerDeviceId((void*)decodeMeta);
    int indicesDevice = hasIndexerIndices ?
        GetPointerDeviceId(compressedIndices->cudaData) : qDevice;
    int lengthsDevice = hasIndexerIndices ?
        GetPointerDeviceId(compressedLengths->cudaData) : qDevice;
    int kernelDevice = qDevice >= 0 ? qDevice : curDevice;
    if ((qDevice >= 0 && qDevice != curDevice) ||
        (windowDevice >= 0 && windowDevice != kernelDevice) ||
        (compressedDevice >= 0 && compressedDevice != kernelDevice) ||
        (sinkDevice >= 0 && sinkDevice != kernelDevice) ||
        (metaDevice >= 0 && metaDevice != kernelDevice) ||
        (indicesDevice >= 0 && indicesDevice != kernelDevice) ||
        (lengthsDevice >= 0 && lengthsDevice != kernelDevice)) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16,
                                     {bsz, seqlen, heads, dim})) {
        return false;
    }

    size_t tempBytes = (size_t)bsz * seqlen * heads * dim * sizeof(float);
    float *cudaTemp = (float *)FastllmCudaMalloc(tempBytes);
    if (cudaTemp == nullptr) {
        return false;
    }
    bool ok = !hasIndexerIndices && seqlen == 1 && allowTriton &&
        fastllm::FastllmCudaTryTritonDeepSeekV4SparseAttentionDecodeGraph(
            q, windowKV, compressedKV, attnSink, windowSize, compressRatio,
            decodeMeta, softmaxScale, cudaTemp);
    int blocks = bsz * seqlen * heads;
    if (!ok) {
        if (q.dataType == fastllm::DataType::BFLOAT16 &&
            compressedKV.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<__nv_bfloat16, __nv_bfloat16>
                <<<blocks, 256>>>((const __nv_bfloat16 *)q.cudaData,
                                  (const float *)windowKV.cudaData,
                                  (const __nv_bfloat16 *)compressedKV.cudaData,
                                  (const float *)attnSink.cudaData, cudaTemp,
                                  bsz, seqlen, heads, dim, windowSize, 0, 0,
                                  compressedCapacity,
                                  hasIndexerIndices ? (const int32_t *)compressedIndices->cudaData : nullptr,
                                  hasIndexerIndices ? (const int *)compressedLengths->cudaData : nullptr,
                                  softmaxScale,
                                  decodeMeta, compressRatio);
            ok = true;
        } else if (q.dataType == fastllm::DataType::FLOAT16 &&
                   compressedKV.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<half, __nv_bfloat16>
                <<<blocks, 256>>>((const half *)q.cudaData,
                                  (const float *)windowKV.cudaData,
                                  (const __nv_bfloat16 *)compressedKV.cudaData,
                                  (const float *)attnSink.cudaData, cudaTemp,
                                  bsz, seqlen, heads, dim, windowSize, 0, 0,
                                  compressedCapacity,
                                  hasIndexerIndices ? (const int32_t *)compressedIndices->cudaData : nullptr,
                                  hasIndexerIndices ? (const int *)compressedLengths->cudaData : nullptr,
                                  softmaxScale,
                                  decodeMeta, compressRatio);
            ok = true;
        } else if (q.dataType == fastllm::DataType::FLOAT32 &&
                   compressedKV.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<float, __nv_bfloat16>
                <<<blocks, 256>>>((const float *)q.cudaData,
                                  (const float *)windowKV.cudaData,
                                  (const __nv_bfloat16 *)compressedKV.cudaData,
                                  (const float *)attnSink.cudaData, cudaTemp,
                                  bsz, seqlen, heads, dim, windowSize, 0, 0,
                                  compressedCapacity,
                                  hasIndexerIndices ? (const int32_t *)compressedIndices->cudaData : nullptr,
                                  hasIndexerIndices ? (const int *)compressedLengths->cudaData : nullptr,
                                  softmaxScale,
                                  decodeMeta, compressRatio);
            ok = true;
        }
    }
    if (ok) {
        int rows = bsz * seqlen * heads;
        int threads = 128;
        int rotaryWorkPerRow = dim - ropeDim + (ropeDim >> 1);
        int rotaryBlocks = (rows * rotaryWorkPerRow + threads - 1) / threads;
        DeepSeekV4SparseDecodeRotaryCastKernel<<<rotaryBlocks, threads>>>(
            cudaTemp, (__nv_bfloat16 *)output.cudaData, rows, 0, seqlen, heads,
            dim, ropeDim, ropeBase,
            0, originalSeqLen, ropeFactor, betaFast, betaSlow, decodeMeta);
        DeviceSync();
    }
    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionDecodeCachedBatch(
                                                                 const std::vector<fastllm::Data*> &qParts,
                                                                 const std::vector<fastllm::Data*> &windowKVs,
                                                                 const std::vector<fastllm::Data*> &compressedKVs,
                                                                 fastllm::Data &attnSink,
                                                                 int windowSize,
                                                                 const std::vector<int> &startPositions,
                                                                 const std::vector<int> &compressedCounts,
                                                                 int ropeDim, float ropeBase,
                                                                 int originalSeqLen,
                                                                 float ropeFactor, int betaFast,
                                                                 int betaSlow, float softmaxScale,
                                                                 fastllm::Data &output) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_BATCH_SPARSE_DECODE") != nullptr) {
        return false;
    }
    int batch = (int)qParts.size();
    if (batch <= 0 || (int)windowKVs.size() != batch || (int)compressedKVs.size() != batch ||
        (int)startPositions.size() != batch || (int)compressedCounts.size() != batch ||
        windowSize <= 0 || ropeDim <= 0) {
        return false;
    }
    if (qParts[0] == nullptr || qParts[0]->dims.size() != 4 || qParts[0]->dims[0] != 1 ||
        qParts[0]->dims[1] != 1 || qParts[0]->dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    int heads = qParts[0]->dims[2], dim = qParts[0]->dims[3];
    if (heads <= 0 || dim <= 0 || ropeDim > dim || dim > 256 * 4 ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        attnSink.Count(0) != (uint64_t)heads) {
        return false;
    }

    fastllm::DataType qType = qParts[0]->dataType;
    fastllm::DataType compressedType = fastllm::DataType::FLOAT32;
    bool hasCompressed = false;
    std::vector<const void*> qPtrs(batch, nullptr);
    std::vector<const float*> windowPtrs(batch, nullptr);
    std::vector<const void*> compressedPtrs(batch, nullptr);
    for (int b = 0; b < batch; b++) {
        const fastllm::Data *q = qParts[b];
        const fastllm::Data *windowKV = windowKVs[b];
        const fastllm::Data *compressedKV = compressedKVs[b];
        int startPos = startPositions[b];
        int compressedCount = compressedCounts[b];
        int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
        if (q == nullptr || windowKV == nullptr || startPos < 0 || compressedCount < 0 ||
            liveWindow + compressedCount <= 0 ||
            liveWindow + compressedCount > kDeepSeekV4SparseDecodeMaxKeys ||
            q->dims.size() != 4 || q->dims[0] != 1 || q->dims[1] != 1 ||
            q->dims[2] != heads || q->dims[3] != dim ||
            q->dataType != qType || q->dataDevice != fastllm::DataDevice::CUDA ||
            windowKV->dataDevice != fastllm::DataDevice::CUDA ||
            windowKV->dataType != fastllm::DataType::FLOAT32 ||
            windowKV->Count(0) != (uint64_t)windowSize * dim) {
            return false;
        }
        qPtrs[b] = q->cudaData;
        windowPtrs[b] = (const float*)windowKV->cudaData;
        if (compressedCount > 0) {
            if (compressedKV == nullptr ||
                compressedKV->dataDevice != fastllm::DataDevice::CUDA ||
                compressedKV->Count(0) < (uint64_t)compressedCount * dim) {
                return false;
            }
            if (!hasCompressed) {
                compressedType = compressedKV->dataType;
                hasCompressed = true;
            } else if (compressedKV->dataType != compressedType) {
                return false;
            }
            if (compressedType != fastllm::DataType::BFLOAT16 &&
                compressedType != fastllm::DataType::FLOAT16 &&
                compressedType != fastllm::DataType::FLOAT32) {
                return false;
            }
            compressedPtrs[b] = compressedKV->cudaData;
        }
    }
    if (!hasCompressed) {
        compressedType = fastllm::DataType::FLOAT32;
    }
    if (qType != fastllm::DataType::BFLOAT16 &&
        qType != fastllm::DataType::FLOAT16 &&
        qType != fastllm::DataType::FLOAT32) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16, {batch, 1, heads, dim})) {
        return false;
    }

    const void **cudaQPtrs = (const void**)FastllmCudaMalloc((size_t)batch * sizeof(void*));
    const float **cudaWindowPtrs = (const float**)FastllmCudaMalloc((size_t)batch * sizeof(float*));
    const void **cudaCompressedPtrs = (const void**)FastllmCudaMalloc((size_t)batch * sizeof(void*));
    int *cudaStartPositions = (int*)FastllmCudaMalloc((size_t)batch * sizeof(int));
    int *cudaCompressedCounts = (int*)FastllmCudaMalloc((size_t)batch * sizeof(int));
    float *cudaTemp = (float*)FastllmCudaMalloc((size_t)batch * heads * dim * sizeof(float));
    if (cudaQPtrs == nullptr || cudaWindowPtrs == nullptr || cudaCompressedPtrs == nullptr ||
        cudaStartPositions == nullptr || cudaCompressedCounts == nullptr || cudaTemp == nullptr) {
        if (cudaQPtrs != nullptr) {
            FastllmCudaFree((void*)cudaQPtrs);
        }
        if (cudaWindowPtrs != nullptr) {
            FastllmCudaFree((void*)cudaWindowPtrs);
        }
        if (cudaCompressedPtrs != nullptr) {
            FastllmCudaFree((void*)cudaCompressedPtrs);
        }
        if (cudaStartPositions != nullptr) {
            FastllmCudaFree(cudaStartPositions);
        }
        if (cudaCompressedCounts != nullptr) {
            FastllmCudaFree(cudaCompressedCounts);
        }
        if (cudaTemp != nullptr) {
            FastllmCudaFree(cudaTemp);
        }
        return false;
    }

    cudaMemcpy((void*)cudaQPtrs, qPtrs.data(), (size_t)batch * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)cudaWindowPtrs, windowPtrs.data(), (size_t)batch * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)cudaCompressedPtrs, compressedPtrs.data(), (size_t)batch * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaStartPositions, startPositions.data(), (size_t)batch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaCompressedCounts, compressedCounts.data(), (size_t)batch * sizeof(int), cudaMemcpyHostToDevice);

    bool ok = false;
    if (qType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeBatchByCompressed<__nv_bfloat16>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, compressedType,
            (const float*)attnSink.cudaData, cudaStartPositions, cudaCompressedCounts,
            cudaTemp, batch, heads, dim, windowSize, softmaxScale);
    } else if (qType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeBatchByCompressed<half>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, compressedType,
            (const float*)attnSink.cudaData, cudaStartPositions, cudaCompressedCounts,
            cudaTemp, batch, heads, dim, windowSize, softmaxScale);
    } else if (qType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchSparseDecodeBatchByCompressed<float>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, compressedType,
            (const float*)attnSink.cudaData, cudaStartPositions, cudaCompressedCounts,
            cudaTemp, batch, heads, dim, windowSize, softmaxScale);
    }

    if (ok) {
        int rows = batch * heads;
        int threads = 128;
        int rotaryWorkPerRow = dim - ropeDim + (ropeDim >> 1);
        int blocks = (rows * rotaryWorkPerRow + threads - 1) / threads;
        DeepSeekV4SparseDecodeRotaryCastBatchKernel<<<blocks, threads>>>(
            cudaTemp, (__nv_bfloat16*)output.cudaData, cudaStartPositions,
            rows, heads, dim, ropeDim, ropeBase, originalSeqLen,
            ropeFactor, betaFast, betaSlow);
        DeviceSync();
    }

    FastllmCudaFree((void*)cudaQPtrs);
    FastllmCudaFree((void*)cudaWindowPtrs);
    FastllmCudaFree((void*)cudaCompressedPtrs);
    FastllmCudaFree(cudaStartPositions);
    FastllmCudaFree(cudaCompressedCounts);
    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4WoA(const fastllm::Data &o, const fastllm::Data &woA,
                                         int groups, int oRank, fastllm::Data &output,
                                         bool allowTriton) {
    if (o.dataDevice != fastllm::DataDevice::CUDA || woA.dataDevice != fastllm::DataDevice::CUDA ||
        o.dims.size() != 4 || groups <= 0 || oRank <= 0) {
        return false;
    }

    int bsz = o.dims[0], seqlen = o.dims[1], heads = o.dims[2], headDim = o.dims[3];
    if (heads % groups != 0) {
        return false;
    }
    int groupDim = (heads / groups) * headDim;
    if (woA.Count(0) != (uint64_t)groups * oRank * groupDim) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16,
                                     {bsz, seqlen, groups * oRank})) {
        return false;
    }

#ifdef FASTLLM_ENABLE_DSV4_WOA_DEEPGEMM_SM120
    // The SM120 DeepGEMM specialization produces its FP8 operand and packed
    // UE8M0 scales directly from the inverse-RoPE BF16 tensor.  Calling it
    // before the compatibility round-trip avoids quantizing the same values
    // twice; all unsupported shapes continue through the generic path below.
    if (woA.dataType == fastllm::DataType::FP8_E4M3 &&
        FastllmCudaRuntimeArch() >= 120 &&
        FastllmCudaDeepSeekV4WoADeepGemmSm120(
            o, woA, groups, oRank, output)) {
        DeviceSync();
        return true;
    }
#endif

    const fastllm::Data *projectionInput = &o;
    fastllm::Data quantizedInput;
    bool useVllmFp8Input =
        woA.dataType == fastllm::DataType::FP8_E4M3 &&
        FastllmCudaRuntimeArch() >= 120;
    if (useVllmFp8Input) {
        if (!DeepSeekV4PrepareWoAQuantizedInput(o, quantizedInput)) {
            return false;
        }
        projectionInput = &quantizedInput;
    }

    fastllm::Data &mutableWoA = const_cast<fastllm::Data&>(woA);
    if (allowTriton && fastllm::FastllmCudaTryTritonDeepSeekV4WoA(
            *projectionInput, mutableWoA, groups, oRank, output)) {
        DeviceSync();
        return true;
    }

    bool ok = false;
    if (projectionInput->dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchWoAByWeight<__nv_bfloat16>(
            *projectionInput, woA, groups, oRank, output,
            bsz, seqlen, heads, headDim);
    } else if (projectionInput->dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchWoAByWeight<half>(
            *projectionInput, woA, groups, oRank, output,
            bsz, seqlen, heads, headDim);
    } else if (projectionInput->dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchWoAByWeight<float>(
            *projectionInput, woA, groups, oRank, output,
            bsz, seqlen, heads, headDim);
    }
    if (!ok) {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4HcPost(const fastllm::Data &x, const fastllm::Data &residual,
                                            const float *post, const float *comb, int bsz,
                                            int seqlen, int hcMult, int dim, fastllm::Data &output) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || residual.dataDevice != fastllm::DataDevice::CUDA ||
        post == nullptr || comb == nullptr || bsz <= 0 || seqlen <= 0 || hcMult <= 0 || dim <= 0 ||
        x.Count(0) != (uint64_t)bsz * seqlen * dim ||
        residual.Count(0) != (uint64_t)bsz * seqlen * hcMult * dim) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, x.dataType, {bsz, seqlen, hcMult, dim})) {
        return false;
    }

    int tokens = bsz * seqlen;
    size_t postBytes = (size_t)tokens * hcMult * sizeof(float);
    size_t combBytes = (size_t)tokens * hcMult * hcMult * sizeof(float);
    float *cudaPost = (float *)FastllmCudaMalloc(postBytes);
    float *cudaComb = (float *)FastllmCudaMalloc(combBytes);
    FastllmCudaCopyFromHostToDevice(cudaPost, (void *)post, postBytes);
    FastllmCudaCopyFromHostToDevice(cudaComb, (void *)comb, combBytes);

    bool ok = false;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<__nv_bfloat16>(x, residual, cudaPost, cudaComb,
                                                            output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<half>(x, residual, cudaPost, cudaComb,
                                                   output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPostByResidual<float>(x, residual, cudaPost, cudaComb,
                                                    output, tokens, hcMult, dim);
    }

    DeviceSync();
    FastllmCudaFree(cudaPost);
    FastllmCudaFree(cudaComb);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcPostCudaMix(const fastllm::Data &x, const fastllm::Data &residual,
                                                   const fastllm::Data &post, const fastllm::Data &comb,
                                                   int bsz, int seqlen, int hcMult, int dim,
                                                   fastllm::Data &output) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || residual.dataDevice != fastllm::DataDevice::CUDA ||
        post.dataDevice != fastllm::DataDevice::CUDA || comb.dataDevice != fastllm::DataDevice::CUDA ||
        post.dataType != fastllm::DataType::FLOAT32 || comb.dataType != fastllm::DataType::FLOAT32 ||
        bsz <= 0 || seqlen <= 0 || hcMult <= 0 || dim <= 0 ||
        x.Count(0) != (uint64_t)bsz * seqlen * dim ||
        residual.Count(0) != (uint64_t)bsz * seqlen * hcMult * dim ||
        post.Count(0) != (uint64_t)bsz * seqlen * hcMult ||
        comb.Count(0) != (uint64_t)bsz * seqlen * hcMult * hcMult) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, x.dataType, {bsz, seqlen, hcMult, dim})) {
        return false;
    }

    int tokens = bsz * seqlen;
    bool ok = false;
    const float *cudaPost = (const float *)post.cudaData;
    const float *cudaComb = (const float *)comb.cudaData;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<__nv_bfloat16>(x, residual, cudaPost, cudaComb,
                                                            output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<half>(x, residual, cudaPost, cudaComb,
                                                   output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPostByResidual<float>(x, residual, cudaPost, cudaComb,
                                                    output, tokens, hcMult, dim);
    }
    DeviceSync();
    return ok;
}
