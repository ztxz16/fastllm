#include "fastllm-attention-workspace.cuh"

static size_t ParseSizeFromEnv(const char* env_name, size_t default_size) {
    const char* val = std::getenv(env_name);
    if (!val || val[0] == '\0') return default_size;
    char* end = nullptr;
    double num = std::strtod(val, &end);
    if (end == val) return default_size;
    size_t result = default_size;
    if (*end == 'G' || *end == 'g') result = (size_t)(num * 1024 * 1024 * 1024);
    else if (*end == 'M' || *end == 'm') result = (size_t)(num * 1024 * 1024);
    else if (*end == 'K' || *end == 'k') result = (size_t)(num * 1024);
    else if (*end == '\0') result = (size_t)num;
    else return default_size;
    printf("[Info] %s = %s (%.2f MB)\n", env_name, val, result / (1024.0 * 1024.0));
    return result;
}

FastllmCudaTempDeviceBuffer::~FastllmCudaTempDeviceBuffer() {
    if (data == nullptr) {
        return;
    }
    int oldDevice = -1;
    cudaGetDevice(&oldDevice);
    cudaSetDevice(device);
    FastllmCudaDirectFree(data);
    if (oldDevice >= 0) {
        cudaSetDevice(oldDevice);
    }
}

void FlashInferWorkSpaceManager::EnsureIntCapacity(size_t required) {
    if (required <= int_workspace_size) return;
    size_t capacity = ((required + (1 << 20) - 1) >> 20) << 20;
    checkCudaErrors("FlashInfer integer workspace resize sync", cudaDeviceSynchronize());
    void *device = FastllmCudaDirectMalloc(capacity);
    void *host = nullptr;
    checkCudaErrors("FlashInfer integer host workspace resize", cudaMallocHost(&host, capacity));
    FastllmCudaDirectFree(d_int_workspace);
    checkCudaErrors("FlashInfer integer host workspace release", cudaFreeHost(h_page_locked_int_workspace));
    d_int_workspace = device;
    h_page_locked_int_workspace = host;
    int_workspace_size = capacity;
}

FlashInferWorkSpaceManager::FlashInferWorkSpaceManager()
    : float_workspace_size(ParseSizeFromEnv("FT_FLOAT_WORKSPACE_SIZE", 256 * 1024 * 1024)) {
    d_float_workspace = FastllmCudaMalloc(float_workspace_size);
    d_int_workspace = FastllmCudaDirectMalloc(int_workspace_size);
    cudaError_t err = cudaMallocHost(&h_page_locked_int_workspace, int_workspace_size);
    if (err != cudaSuccess || h_page_locked_int_workspace == nullptr) {
        printf("FlashInferWorkSpaceManager: Failed to allocate h_page_locked_int_workspace: %s\n", cudaGetErrorString(err));
        exit(0);
    }
}

FlashInferWorkSpaceManager::~FlashInferWorkSpaceManager() {
    FastllmCudaFree(d_float_workspace);
    FastllmCudaDirectFree(d_int_workspace);
    cudaFreeHost(h_page_locked_int_workspace);
}

static std::map<int, std::unique_ptr<FlashInferWorkSpaceManager>> s_fastllmFlashInferWorkSpaceMap;
static std::mutex s_fastllmFlashInferWorkSpaceMapLock;

FlashInferWorkSpaceManager& getFastllmFlashInferWorkSpace() {
    int id = -1;
    cudaGetDevice(&id);
    std::lock_guard<std::mutex> guard(s_fastllmFlashInferWorkSpaceMapLock);
    auto it = s_fastllmFlashInferWorkSpaceMap.find(id);
    if (it != s_fastllmFlashInferWorkSpaceMap.end()) {
        return *it->second;
    }
    auto manager = std::make_unique<FlashInferWorkSpaceManager>();
    FlashInferWorkSpaceManager* ptr = manager.get();
    s_fastllmFlashInferWorkSpaceMap[id] = std::move(manager);
    return *ptr;
}

static FlashInferWorkSpaceManager *tryGetFastllmFlashInferWorkSpace(int id) {
    std::lock_guard<std::mutex> guard(s_fastllmFlashInferWorkSpaceMapLock);
    auto it = s_fastllmFlashInferWorkSpaceMap.find(id);
    return it == s_fastllmFlashInferWorkSpaceMap.end() ? nullptr : it->second.get();
}

void *FastllmCudaGetFlashInferFloatWorkspace(size_t *outSize) {
    FlashInferWorkSpaceManager &workspace = getFastllmFlashInferWorkSpace();
    if (outSize != nullptr) {
        *outSize = workspace.float_workspace_size;
    }
    return workspace.d_float_workspace;
}

static std::map<int, std::unique_ptr<FastllmCudaTempDeviceBuffer>> s_fastllmCudaTempBuffers;
// s_fastllmCudaTempBuffersMapLock 仅保护 map 结构本身（查找/插入），
// 持锁期间绝不调用任何 CUDA API，避免跨设备线程互相阻塞。
static std::mutex s_fastllmCudaTempBuffersMapLock;
// 每个设备一把锁：缓冲区的 malloc/free（会触发本设备同步）只在对应设备锁内进行，
// 不会阻塞其它设备线程。这是张量并行下避免死锁的关键：
// 否则一个 rank 在持有全局锁时执行 cudaFree（隐式同步本设备），
// 而本设备 stream 上挂着需要其它 rank 共同完成的 NCCL 集合通信，
// 其它 rank 又卡在等待这把全局锁，从而形成跨 rank 死锁。
static std::map<int, std::unique_ptr<std::mutex>> s_fastllmCudaTempBufferDeviceLocks;

static FastllmCudaTempDeviceBuffer *FastllmGetCudaTempBufferHolder(int id, std::mutex **outDeviceLock) {
    std::lock_guard<std::mutex> guard(s_fastllmCudaTempBuffersMapLock);
    auto &holder = s_fastllmCudaTempBuffers[id];
    if (holder == nullptr) {
        holder = std::make_unique<FastllmCudaTempDeviceBuffer>(id);
    }
    auto &deviceLock = s_fastllmCudaTempBufferDeviceLocks[id];
    if (deviceLock == nullptr) {
        deviceLock = std::make_unique<std::mutex>();
    }
    if (outDeviceLock != nullptr) {
        *outDeviceLock = deviceLock.get();
    }
    return holder.get();
}

static size_t FastllmCudaTempAlignBytes(size_t size) {
    const size_t align = 256;
    return ((size + align - 1) / align) * align;
}

void *FastllmBorrowCudaTempBuffer(size_t needBytes, size_t *outBytes, bool *outOwn) {
    if (outOwn != nullptr) {
        *outOwn = false;
    }
    if (needBytes == 0) {
        needBytes = 1;
    }

    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    checkCudaErrors("Error: CUDA error when find device!", state);

    FlashInferWorkSpaceManager *workspace = tryGetFastllmFlashInferWorkSpace(id);
    if (workspace != nullptr && workspace->d_float_workspace != nullptr &&
        workspace->float_workspace_size >= needBytes) {
        if (outBytes != nullptr) {
            *outBytes = workspace->float_workspace_size;
        }
        return workspace->d_float_workspace;
    }

    std::mutex *deviceLock = nullptr;
    FastllmCudaTempDeviceBuffer *holderPtr = FastllmGetCudaTempBufferHolder(id, &deviceLock);
    std::lock_guard<std::mutex> guard(*deviceLock);
    FastllmCudaTempDeviceBuffer &holder = *holderPtr;

    size_t allocBytes = FastllmCudaTempAlignBytes(needBytes);
    if (holder.size < allocBytes) {
        if (holder.data != nullptr) {
            FastllmCudaDirectFree(holder.data);
            holder.data = nullptr;
            holder.size = 0;
        }
        holder.data = FastllmCudaDirectMalloc(allocBytes);
        if (holder.data == nullptr) {
            holder.size = 0;
            if (outBytes != nullptr) {
                *outBytes = 0;
            }
            return nullptr;
        }
        holder.size = allocBytes;
    }

    if (outBytes != nullptr) {
        *outBytes = holder.size;
    }
    return holder.data;
}

void FastllmReleaseCudaTempBuffer(void *ptr, bool own) {
    if (own && ptr != nullptr) {
        FastllmCudaFree(ptr);
    }
}

void *FastllmBorrowDequantScratch(size_t needBytes, size_t *outBytes, bool *outOwn) {
    return FastllmBorrowCudaTempBuffer(needBytes, outBytes, outOwn);
}

void FastllmReleaseDequantScratch(void *ptr, bool own) {
    FastllmReleaseCudaTempBuffer(ptr, own);
}
