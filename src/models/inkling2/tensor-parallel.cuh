// Each rank owns contiguous output-channel tiles of every NVFP4 expert.
// Event edges join the rank streams before activation quantization or reduction.
// The rank stream also owns its transfers. Ordinary CUDA kernels can be
// captured in a multi-device graph, including pitched writes into the result.
static __global__ void copyRankRows(const char *src, char *dst, size_t width, size_t pitch, size_t rows) {
    const size_t packs = width / sizeof(uint4);
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * packs; i += size_t(gridDim.x) * blockDim.x) {
        const size_t row = i / packs, col = i % packs;
        reinterpret_cast<uint4 *>(dst + row * pitch)[col] = reinterpret_cast<const uint4 *>(src + row * width)[col];
    }
    const size_t tail = width % sizeof(uint4);
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * tail; i += size_t(gridDim.x) * blockDim.x) {
        const size_t row = i / tail, col = width - tail + i % tail;
        dst[row * pitch + col] = src[row * width + col];
    }
}

struct TensorRank {
    int device, primary, smCount = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr, done = nullptr;
    explicit TensorRank(int device, int primary) : device(device), primary(primary) {}
    struct DeviceScope {
        int previous;
        explicit DeviceScope(int device) { check(cudaGetDevice(&previous)); check(cudaSetDevice(device)); }
        ~DeviceScope() { cudaSetDevice(previous); }
    };
    // Decode graphs keep their scratch addresses through replay. Preserve earlier
    // allocations when a larger batch is warmed on the same rank stream.
    std::vector<std::unique_ptr<uint8_t, decltype(&cudaFree)>> scratch;
    size_t capacity = 0, cursor = 0;
    void reserve(size_t required) {
        cursor = 0;
        if (required <= capacity) return;
        const size_t bytes = std::max(required, capacity * 2);
        uint8_t *ptr = nullptr;
        check(cudaMalloc(&ptr, bytes));
        std::unique_ptr<uint8_t, decltype(&cudaFree)> owner(ptr, cudaFree);
        scratch.push_back(std::move(owner));
        capacity = bytes;
    }
    template<class T> T *allocate(size_t count) {
        cursor = (cursor + 255) & ~size_t(255);
        if (cursor > capacity || count > (capacity - cursor) / sizeof(T))
            throw std::runtime_error("Inkling TP staging buffer overflow");
        auto *p = reinterpret_cast<T *>(scratch.back().get() + cursor);
        cursor += count * sizeof(T);
        return p;
    }
    void transfer(const void *src, void *dst, size_t width, size_t pitch, size_t rows) const {
        const int blocks = int(std::min<size_t>(1024, (rows * ((width + 15) / 16) + 127) / 128));
        copyRankRows<<<blocks, 128, 0, stream>>>(static_cast<const char *>(src), static_cast<char *>(dst), width, pitch, rows);
        check(cudaGetLastError());
    }
    template<class T> T *receive(const T *source, size_t count) {
        if (!source || !count) return nullptr;
        auto *local = allocate<T>(count);
        const size_t bytes = count * sizeof(T);
        if (bytes >= 16384 && !DecodeGraphScratch::active)
            check(cudaMemcpyPeerAsync(local, device, source, primary, bytes, stream));
        else transfer(source, local, bytes, bytes, 1);
        return local;
    }
    template<class T> void gather(const T *local, T *dst, int rows, int cols, int pitch) {
        transfer(local, dst, size_t(cols) * sizeof(T), size_t(pitch) * sizeof(T), rows);
    }
    void initialize() {
        if (device == primary) return;
        DeviceScope scope(device);
        check(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
        int peer;
        check(cudaDeviceCanAccessPeer(&peer, device, primary));
        if (!peer) throw std::runtime_error("Inkling TP requires CUDA peer access to the backbone GPU");
        auto status = cudaDeviceEnablePeerAccess(primary, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else check(status);

#if CUDART_VERSION >= 11020
        cudaMemPool_t pool;
        check(cudaDeviceGetMemPool(&pool, primary));
        cudaMemAccessDesc access{};
        access.location = {cudaMemLocationTypeDevice, device};
        access.flags = cudaMemAccessFlagsProtReadWrite;
        check(cudaMemPoolSetAccess(pool, &access, 1));
#endif
        int image = 0;
        check(cudaMemcpyFromSymbol(&image, native::nativeImage, sizeof(image)));
        if (!image) throw std::runtime_error("Inkling TP needs native NVFP4 kernels on every CUDA device");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        check(cudaSetDevice(primary));
        check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    }
    ~TensorRank() {
        int previous;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (ready) { cudaSetDevice(primary); cudaEventDestroy(ready); }
        if (stream || done) {
            cudaSetDevice(device);
            if (stream) cudaStreamDestroy(stream);
            scratch.clear();
            if (done) cudaEventDestroy(done);
        }
        cudaSetDevice(previous);
    }
};

struct WeightShard {
    std::shared_ptr<TensorRank> rank;
    int first, rows;
    Buffer<uint4> codes;
    Buffer<unsigned> scales;
    Buffer<float> outer;
#ifdef FASTLLM_INKLING_GROUPED_SM120
    native::GroupedWeightMaps groupedMaps{};
#endif
    native::Weights view() const { return {codes.p, scales.p, outer.p}; }
};

template<class T>
Buffer<T> copyWeightShard(const T *src, size_t stride, size_t offset, size_t count, int experts,
                          const TensorRank &rank) {
    TensorRank::DeviceScope scope(rank.device);
    Buffer<T> dst;
    check(cudaMalloc(&dst.p, count * experts * sizeof(T)));
    dst.owner = std::shared_ptr<void>(dst.p, [device = rank.device](void *ptr) {
        int previous;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        cudaSetDevice(device);
        cudaFree(ptr);
        cudaSetDevice(previous);
    });
    for (int e = 0; e < experts; ++e) {
        check(cudaMemcpyPeerAsync(dst.p + e * count, rank.device, src + e * stride + offset, rank.primary,
                                  count * sizeof(T), cudaStreamPerThread));
    }
    check(cudaStreamSynchronize(cudaStreamPerThread));
    return dst;
}
