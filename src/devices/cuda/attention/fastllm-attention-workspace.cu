#include "fastllm-attention-workspace.cuh"
#include <thread>

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

// Registry operations must not allocate on CUDA: an allocation can wait for
// a collective whose other device still needs to look up its workspace.
struct FlashInferWorkSpaceEntry {
    std::mutex mutex;
    std::unique_ptr<FlashInferWorkSpaceManager> workspace;
};
static std::map<int, std::unique_ptr<FlashInferWorkSpaceEntry>> s_fastllmFlashInferWorkSpaceMap;
static std::mutex s_fastllmFlashInferWorkSpaceMapLock;

static FlashInferWorkSpaceEntry *FindFlashInferWorkSpace(int id, bool create) {
    std::lock_guard<std::mutex> guard(s_fastllmFlashInferWorkSpaceMapLock);
    auto it = s_fastllmFlashInferWorkSpaceMap.find(id);
    if (it != s_fastllmFlashInferWorkSpaceMap.end()) return it->second.get();
    if (!create) return nullptr;
    auto entry = std::make_unique<FlashInferWorkSpaceEntry>();
    auto *result = entry.get();
    s_fastllmFlashInferWorkSpaceMap.emplace(id, std::move(entry));
    return result;
}

FlashInferWorkSpaceManager& getFastllmFlashInferWorkSpace(bool allocateFloat) {
    int id = -1;
    checkCudaErrors("Error: CUDA error when find device!", cudaGetDevice(&id));
    auto *entry = FindFlashInferWorkSpace(id, true);
    std::lock_guard<std::mutex> guard(entry->mutex);
    auto &workspace = entry->workspace;
    if (workspace == nullptr) {
        workspace = std::make_unique<FlashInferWorkSpaceManager>();
    }
    if (allocateFloat && workspace->d_float_workspace == nullptr) {
        workspace->d_float_workspace = FastllmCudaMalloc(workspace->float_workspace_size);
    }
    return *workspace;
}

static void *TryGetFlashInferFloatWorkspace(int id, size_t *outSize) {
    if (outSize != nullptr) *outSize = 0;
    auto *entry = FindFlashInferWorkSpace(id, false);
    if (entry == nullptr) return nullptr;
    std::lock_guard<std::mutex> guard(entry->mutex);
    auto *workspace = entry->workspace.get();
    void *data = workspace ? workspace->d_float_workspace : nullptr;
    if (outSize != nullptr && data != nullptr) *outSize = workspace->float_workspace_size;
    return data;
}

void *FastllmCudaTryGetFlashInferFloatWorkspace(size_t *outSize) {
    int id = -1;
    checkCudaErrors("Error: CUDA error when find device!", cudaGetDevice(&id));
    return TryGetFlashInferFloatWorkspace(id, outSize);
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

// The shared temp storage (FlashInfer float arena or holder) of one device is
// used by the scheduler thread and TP worker threads, each on its own
// cudaStreamPerThread. One user at a time; the next user from another stream
// waits on the previous user's release event. Guarded by the device lock.
struct FastllmCudaTempLease {
    void *leasedPtr = nullptr;
    cudaEvent_t released = nullptr;
    bool releasePending = false;
    std::thread::id releaser;
};
static std::map<int, FastllmCudaTempLease> s_fastllmCudaTempLeases;

static FastllmCudaTempDeviceBuffer *FastllmGetCudaTempBufferHolder(
        int id, std::mutex **outDeviceLock, FastllmCudaTempLease **outLease = nullptr) {
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
    if (outLease != nullptr) {
        *outLease = &s_fastllmCudaTempLeases[id];
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

    std::mutex *deviceLock = nullptr;
    FastllmCudaTempLease *lease = nullptr;
    FastllmCudaTempDeviceBuffer *holderPtr = FastllmGetCudaTempBufferHolder(id, &deviceLock, &lease);
    void *ret = nullptr;
    size_t retBytes = 0;
    cudaEvent_t waitEvent = nullptr;
    {
        std::lock_guard<std::mutex> guard(*deviceLock);
        if (lease->leasedPtr == nullptr) {
            size_t workspaceBytes = 0;
            void *workspace = TryGetFlashInferFloatWorkspace(id, &workspaceBytes);
            FastllmCudaTempDeviceBuffer &holder = *holderPtr;
            size_t allocBytes = FastllmCudaTempAlignBytes(needBytes);
            if (workspace != nullptr && workspaceBytes >= needBytes) {
                ret = workspace;
                retBytes = workspaceBytes;
            } else {
                if (holder.size < allocBytes) {
                    if (holder.data != nullptr) {
                        FastllmCudaDirectFree(holder.data);
                        holder.data = nullptr;
                        holder.size = 0;
                    }
                    holder.data = FastllmCudaDirectMalloc(allocBytes);
                    holder.size = holder.data == nullptr ? 0 : allocBytes;
                }
                ret = holder.data;
                retBytes = holder.size;
            }
            if (ret == nullptr) {
                if (outBytes != nullptr) {
                    *outBytes = 0;
                }
                return nullptr;
            }
            lease->leasedPtr = ret;
            if (lease->releasePending && lease->releaser != std::this_thread::get_id()) {
                waitEvent = lease->released;
            }
        }
    }

    if (ret == nullptr) {
        // Leased by another stream or an outer borrow on this thread: private pool memory.
        ret = FastllmCudaMalloc(needBytes);
        retBytes = ret == nullptr ? 0 : needBytes;
        if (ret != nullptr && outOwn != nullptr) {
            *outOwn = true;
        }
    } else if (waitEvent != nullptr && !FastllmCudaGraphIsCapturingFast()) {
        checkCudaErrors("Error: CUDA error when waiting for temp buffer release!",
                        cudaStreamWaitEvent(cudaStreamPerThread, waitEvent, 0));
    }
    if (outBytes != nullptr) {
        *outBytes = retBytes;
    }
    return ret;
}

void FastllmReleaseCudaTempBuffer(void *ptr, bool own) {
    if (ptr == nullptr) {
        return;
    }
    if (own) {
        if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr);
        return;
    }

    int id = -1;
    checkCudaErrors("Error: CUDA error when find device!", cudaGetDevice(&id));
    std::mutex *deviceLock = nullptr;
    FastllmCudaTempLease *lease = nullptr;
    FastllmGetCudaTempBufferHolder(id, &deviceLock, &lease);
    std::lock_guard<std::mutex> guard(*deviceLock);
    if (ptr != lease->leasedPtr) {
        return;
    }
    // cudaEventRecord/cudaStreamWaitEvent across a capture boundary would invalidate it.
    if (!FastllmCudaGraphIsCapturingFast()) {
        if (lease->released == nullptr) {
            checkCudaErrors("Error: CUDA error when creating temp buffer event!",
                            cudaEventCreateWithFlags(&lease->released, cudaEventDisableTiming));
        }
        checkCudaErrors("Error: CUDA error when recording temp buffer release!",
                        cudaEventRecord(lease->released, cudaStreamPerThread));
        lease->releasePending = true;
        lease->releaser = std::this_thread::get_id();
    }
    lease->leasedPtr = nullptr;
}

void *FastllmBorrowDequantScratch(size_t needBytes, size_t *outBytes, bool *outOwn) {
    return FastllmBorrowCudaTempBuffer(needBytes, outBytes, outOwn);
}

void FastllmReleaseDequantScratch(void *ptr, bool own) {
    FastllmReleaseCudaTempBuffer(ptr, own);
}
