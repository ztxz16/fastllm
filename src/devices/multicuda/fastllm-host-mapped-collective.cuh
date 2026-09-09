// Included by fastllm-multicuda.cu after the host-collective kind declaration.
#if defined(_WIN32) && !defined(USE_ROCM)
namespace {
constexpr size_t kHostMappedMaxBytes = 64 * 1024;
constexpr uint64_t kHostMappedTimeoutNs = 1000000000ULL;
struct alignas(128) HostMappedFlag {
    uint32_t value;
};
struct HostMappedShared {
    HostMappedFlag flags[2][2][2]; // ready/done, alternating generation, rank
    HostMappedFlag failed[2];
    alignas(128) int parameters[2][4]; // count, dtype, operation, root rank
    alignas(128) uint8_t inputs[2][kHostMappedMaxBytes];
};
struct HostMappedState {
    int devices[2] = {-1, -1};
    HostMappedShared *host = nullptr;
    HostMappedShared *mapped[2] = {nullptr, nullptr};
    uint32_t *sequences[2] = {nullptr, nullptr};
    ~HostMappedState();
};
static HostMappedState *hostMappedCurrent = nullptr;

__device__ __forceinline__ void HostMappedRelease(uint32_t *p, uint32_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
#endif
}
__device__ __forceinline__ uint32_t HostMappedAcquire(const uint32_t *p) {
    uint32_t v = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
#endif
    return v;
}
__device__ __forceinline__ uint64_t HostMappedTime() {
    uint64_t v;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
    return v;
}
template <bool FailStop>
__device__ bool HostMappedWait(HostMappedShared *s, int phase, int slot, int rank,
                               uint32_t sequence) {
    const uint64_t start = HostMappedTime();
    while (HostMappedAcquire(&s->flags[phase][slot][1 - rank].value) != sequence) {
        if (HostMappedAcquire(&s->failed[1 - rank].value) ||
            HostMappedTime() - start > kHostMappedTimeoutNs) {
            HostMappedRelease(&s->failed[rank].value, 1);
            if (FailStop) {
                printf("Error: mapped-host CUDA collective timed out on rank %d (phase %d).\n",
                       rank, phase);
                asm volatile("trap;");
            }
            return false;
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
    return true;
}
template <typename T> __device__ T HostMappedSum(T a, T b) { return (T)((int64_t)a + (int64_t)b); }
template <> __device__ float HostMappedSum(float a, float b) { return a + b; }
template <> __device__ half HostMappedSum(half a, half b) {
    return __float2half_rn(__half2float(a) + __half2float(b));
}
template <> __device__ __nv_bfloat16 HostMappedSum(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16_rn(__bfloat162float(a) + __bfloat162float(b));
}

// One CTA per rank owns the complete small message. No remote atomic RMW is
// used: each naturally aligned 32-bit flag has exactly one GPU writer. System
// release/acquire publishes mapped-memory payloads across PCIe; every writer
// fences before the CTA publishes readiness. The second rendezvous prevents
// reuse of either payload until both readers finish. Alternating flag slots
// prevent an early next invocation from overwriting an unobserved completion.
// Sequence numbers live on the GPU and advance on execution, not capture, so
// repeated launches and recaptures cannot replay constant barrier generations.
template <typename T, bool FailStop = true>
__global__ void FastllmHostMappedCollectiveKernel(const T *send, T *recv, int count, int dataType,
                                                  int kind, int root, int rank, HostMappedShared *s,
                                                  uint32_t *counter) {
    __shared__ uint32_t sequence;
    __shared__ int ok;
    if (threadIdx.x == 0) {
        sequence = ++*counter;
        ok = !HostMappedAcquire(&s->failed[rank].value) &&
             !HostMappedAcquire(&s->failed[1 - rank].value);
        s->parameters[rank][0] = count;
        s->parameters[rank][1] = dataType;
        s->parameters[rank][2] = kind;
        s->parameters[rank][3] = root;
    }
    __syncthreads();
    if (!ok) {
        if (FailStop && threadIdx.x == 0)
            asm volatile("trap;");
        return;
    }
    T *local = reinterpret_cast<T *>(s->inputs[rank]);
    const T *peer = reinterpret_cast<const T *>(s->inputs[1 - rank]);
    if (kind != (int)FastllmHostCollectiveKind::Broadcast || rank == root) {
        for (int i = threadIdx.x; i < count; i += blockDim.x)
            local[i] = send[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        HostMappedRelease(&s->flags[0][sequence & 1][rank].value, sequence);
        ok = HostMappedWait<FailStop>(s, 0, sequence & 1, rank, sequence);
        if (ok) {
            for (int i = 0; i < 4; ++i)
                ok &= s->parameters[0][i] == s->parameters[1][i];
            if (!ok) {
                HostMappedRelease(&s->failed[rank].value, 2);
                if (FailStop) {
                    printf("Error: mapped-host CUDA collective parameters differ across ranks.\n");
                    asm volatile("trap;");
                }
            }
        }
    }
    __syncthreads();
    if (!ok)
        return;
    if (kind != (int)FastllmHostCollectiveKind::Reduce || rank == root) {
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            if (kind == (int)FastllmHostCollectiveKind::Broadcast) {
                recv[i] = rank == root ? send[i] : peer[i];
            } else {
                // Preserve rank order, including conversion/rounding to T.
                T first = rank == 0 ? send[i] : peer[i];
                T second = rank == 0 ? peer[i] : send[i];
                recv[i] = HostMappedSum(first, second);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        HostMappedRelease(&s->flags[1][sequence & 1][rank].value, sequence);
        HostMappedWait<FailStop>(s, 1, sequence & 1, rank, sequence);
    }
}

HostMappedState::~HostMappedState() {
    int old = 0;
    cudaGetDevice(&old);
    for (int r = 0; r < 2; ++r) {
        if (sequences[r]) {
            cudaSetDevice(devices[r]);
            cudaFree(sequences[r]);
        }
    }
    if (host)
        cudaFreeHost(host);
    cudaSetDevice(old);
}

static bool HostMappedSelfTest(HostMappedState &s) {
    constexpr int count = 513;
    constexpr int replays = 32;
    float *inputs[2] = {nullptr, nullptr};
    cudaGraph_t graphs[2] = {nullptr, nullptr};
    cudaGraphExec_t execs[2] = {nullptr, nullptr};
    std::atomic<bool> passed{true};
    // Prepare every allocation before either rank enters capture.
    for (int r = 0; r < 2; ++r) {
        std::vector<float> values(count);
        for (int i = 0; i < count; ++i)
            values[i] = (float)((i * 7 + r * 19) % 101 - 50);
        if (cudaSetDevice(s.devices[r]) != cudaSuccess ||
            cudaMalloc((void **)&inputs[r], count * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(inputs[r], values.data(), count * sizeof(float), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            passed = false;
            break;
        }
    }
    auto runRanks = [](const auto &run) {
        std::thread peer(run, 1);
        run(0);
        peer.join();
    };
    if (passed) {
        runRanks([&](int r) {
            bool ok = cudaSetDevice(s.devices[r]) == cudaSuccess;
            bool began =
                ok && cudaStreamBeginCapture(cudaStreamPerThread,
                                             cudaStreamCaptureModeThreadLocal) == cudaSuccess;
            if (began) {
                FastllmHostMappedCollectiveKernel<float, false><<<1, 256, 0, cudaStreamPerThread>>>(
                    inputs[r], inputs[r], count, fastllm::DataType::FLOAT32,
                    (int)FastllmHostCollectiveKind::AllReduce, -1, r, s.mapped[r], s.sequences[r]);
                ok = cudaStreamEndCapture(cudaStreamPerThread, &graphs[r]) == cudaSuccess;
            } else
                ok = false;
            if (ok)
                ok = cudaGraphInstantiate(&execs[r], graphs[r], nullptr, nullptr, 0) == cudaSuccess;
            if (!ok)
                passed = false;
        });
    }
    if (passed) {
        runRanks([&](int r) {
            bool ok = cudaSetDevice(s.devices[r]) == cudaSuccess;
            for (int i = 0; ok && i < replays; ++i) {
                ok = cudaGraphLaunch(execs[r], cudaStreamPerThread) == cudaSuccess &&
                     cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
                if (s.host->failed[r].value)
                    ok = false;
            }
            std::vector<float> values(count);
            if (ok)
                ok = cudaMemcpy(values.data(), inputs[r], count * sizeof(float),
                                cudaMemcpyDeviceToHost) == cudaSuccess;
            for (int i = 0; ok && i < count; ++i) {
                float expected = (float)((i * 7) % 101 - 50) + (float)((i * 7 + 19) % 101 - 50);
                // In-place replay doubles the preceding sum. A stale payload
                // or a skipped replay cannot pass by returning a fixed answer.
                expected *= (float)(uint64_t{1} << (replays - 1));
                ok = values[i] == expected;
            }
            if (!ok)
                passed = false;
        });
    }
    for (int r = 0; r < 2; ++r) {
        cudaSetDevice(s.devices[r]);
        cudaDeviceSynchronize();
        if (execs[r])
            cudaGraphExecDestroy(execs[r]);
        if (graphs[r])
            cudaGraphDestroy(graphs[r]);
        if (inputs[r])
            cudaFree(inputs[r]);
    }
    return passed && !s.host->failed[0].value && !s.host->failed[1].value;
}

static void FastllmInitHostMappedCollectives(const std::vector<int> &devices) {
    hostMappedCurrent = nullptr;
    if (devices.size() != 2)
        return;
    // A bounded, process-lifetime cache per ordered physical GPU pair keeps
    // captured addresses valid across communicator changes and graph teardown.
    // Each pair owns only 128 KiB of pinned payload plus flags and two device
    // counters; recapture/model reload does not allocate another transport.
    static auto *cache = new std::map<std::vector<int>, std::unique_ptr<HostMappedState>>;
    auto existing = cache->find(devices);
    if (existing != cache->end()) {
        hostMappedCurrent = existing->second.get();
        return;
    }
    int original = 0;
    cudaGetDevice(&original);
    std::unique_ptr<HostMappedState> owned(new HostMappedState);
    HostMappedState &s = *owned;
    bool ok = true;
    for (int r = 0; r < 2 && ok; ++r) {
        s.devices[r] = devices[r];
        cudaDeviceProp prop{};
        ok = cudaGetDeviceProperties(&prop, devices[r]) == cudaSuccess && prop.canMapHostMemory &&
             prop.major >= 7;
    }
    if (ok)
        ok = cudaSetDevice(devices[0]) == cudaSuccess &&
             cudaHostAlloc((void **)&s.host, sizeof(HostMappedShared),
                           cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess;
    if (ok)
        std::memset(s.host, 0, sizeof(HostMappedShared));
    for (int r = 0; r < 2 && ok; ++r) {
        ok = cudaSetDevice(devices[r]) == cudaSuccess &&
             cudaHostGetDevicePointer((void **)&s.mapped[r], s.host, 0) == cudaSuccess &&
             cudaMalloc((void **)&s.sequences[r], sizeof(uint32_t)) == cudaSuccess &&
             cudaMemset(s.sequences[r], 0, sizeof(uint32_t)) == cudaSuccess &&
             cudaDeviceSynchronize() == cudaSuccess;
        // Resolve lazily loaded functions before model stream capture.
        cudaFuncAttributes attributes{};
#define HOST_MAPPED_PREPARE(T)                                                                     \
    if (ok)                                                                                        \
    ok = cudaFuncGetAttributes(&attributes, FastllmHostMappedCollectiveKernel<T>) == cudaSuccess
        HOST_MAPPED_PREPARE(half);
        HOST_MAPPED_PREPARE(__nv_bfloat16);
        HOST_MAPPED_PREPARE(float);
        HOST_MAPPED_PREPARE(int32_t);
        HOST_MAPPED_PREPARE(int8_t);
#undef HOST_MAPPED_PREPARE
        if (ok)
            ok = cudaFuncGetAttributes(
                     &attributes, FastllmHostMappedCollectiveKernel<float, false>) == cudaSuccess;
    }
    if (ok)
        ok = HostMappedSelfTest(s);
    if (ok)
        hostMappedCurrent = &s;
    else {
        // Remember an unavailable topology without retaining failed state.
        owned.reset();
        cudaGetLastError();
    }
    (*cache)[devices] = std::move(owned);
    cudaSetDevice(original);
    std::fprintf(
        stderr, "[Fastllm] mapped-host TP2 CUDA Graph collectives: %s (up to %zu bytes/rank).\n",
        ok ? "self-test passed" : "unavailable; keeping host fallback", kHostMappedMaxBytes);
    std::fflush(stderr);
}

static bool FastllmTryHostMappedCollective(FastllmHostCollectiveKind kind, const void *send,
                                           void *recv, int count, int dataType, int root,
                                           int device) {
    HostMappedState *s = hostMappedCurrent;
    const size_t bytes = (size_t)count * FastllmNcclDataTypeBytes(dataType);
    // Eager execution keeps its synchronous fallback and allocation ordering.
    // Only graph capture substitutes the GPU-coordinated mapped transport.
    if (!s || count <= 0 || bytes == 0 || bytes > kHostMappedMaxBytes)
        return false;
    int rank = s->devices[0] == device ? 0 : (s->devices[1] == device ? 1 : -1);
    if (rank < 0)
        return false;
    if (cudaSetDevice(device) != cudaSuccess) {
        FastllmCudaSetThreadError();
        return true;
    }
    if (!FastllmCudaGraphIsCapturingFast())
        return false;
#define HOST_MAPPED_LAUNCH(T)                                                                      \
    FastllmHostMappedCollectiveKernel<T><<<1, 256, 0, cudaStreamPerThread>>>(                      \
        (const T *)send, (T *)recv, count, dataType, (int)kind, root, rank, s->mapped[rank],       \
        s->sequences[rank])
    if (dataType == fastllm::DataType::FLOAT16) {
        HOST_MAPPED_LAUNCH(half);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        HOST_MAPPED_LAUNCH(__nv_bfloat16);
    } else if (dataType == fastllm::DataType::FLOAT32) {
        HOST_MAPPED_LAUNCH(float);
    } else if (dataType == fastllm::DataType::INT32) {
        HOST_MAPPED_LAUNCH(int32_t);
    } else if (dataType == fastllm::DataType::INT8) {
        HOST_MAPPED_LAUNCH(int8_t);
    } else
        return false;
#undef HOST_MAPPED_LAUNCH
    if (cudaPeekAtLastError() != cudaSuccess)
        FastllmCudaSetThreadError();
    return true;
}
} // namespace

#else
static void FastllmInitHostMappedCollectives(const std::vector<int> &) {}
static bool FastllmTryHostMappedCollective(FastllmHostCollectiveKind, const void *, void *, int,
                                           int, int, int) {
    return false;
}
#endif
