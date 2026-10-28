#pragma once

// Included only by diskdevice.cpp in the CUDA (non-ROCm) build.
namespace fastllm {
struct DiskMoeHandoffCounters {
    inline static std::atomic<uint64_t> downloads{0}, downloadBytes{0}, waitUs{0};
    inline static std::atomic<uint64_t> uploads{0}, uploadBytes{0};
    inline static std::atomic<uint64_t> readbacks{0}, readbackBytes{0}, readbackWaitUs{0};
};

class DiskMoeHandoffWorkspace {
    int device;
    void *download = nullptr, *upload = nullptr;
    size_t downloadCapacity = 0, uploadCapacity = 0;
    void *downloadReady = nullptr, *uploadDone = nullptr;
    bool uploadPending = false;
    struct DeviceGuard {
        int previous = FastllmCudaGetDevice();
        explicit DeviceGuard(int device) { FastllmCudaSetDevice(device); }
        ~DeviceGuard() { FastllmCudaSetDevice(previous); }
    };
    static bool Ensure(void *&host, size_t &capacity, size_t bytes) {
        if (bytes <= capacity) return true;
        if (host) FastllmCudaHostFree(host);
        host = FastllmCudaHostMalloc(bytes);
        capacity = host ? bytes : 0;
        if (!host) FastllmCudaClearThreadError();
        return host != nullptr;
    }
public:
    Data cpuInput, cpuIndex, cpuScore, cpuOutput{BFLOAT16}, cpuOrdinaryOutput{FLOAT32};
    explicit DiskMoeHandoffWorkspace(int device) : device(device) {}
    ~DiskMoeHandoffWorkspace() {
        DeviceGuard guard(device);
        if (uploadPending) FastllmCudaEventSynchronize(uploadDone);
        if (downloadReady) FastllmCudaEventDestroy(downloadReady);
        if (uploadDone) FastllmCudaEventDestroy(uploadDone);
        if (download) FastllmCudaHostFree(download);
        if (upload) FastllmCudaHostFree(upload);
    }
    static DiskMoeHandoffWorkspace &Get(int device) {
        // Per submitting thread: another request cannot overwrite an in-flight
        // pinned upload. CPU tensors remain ordinary owned Data allocations.
        static thread_local std::map<int, std::unique_ptr<DiskMoeHandoffWorkspace>> items;
        auto &item = items[device];
        if (!item) item = std::make_unique<DiskMoeHandoffWorkspace>(device);
        return *item;
    }
    bool Stage(const Data &input, const Data &index, const Data &score,
               Data &hostInput, Data &hostIndex, Data &hostScore) {
        if ((input.dataType != BFLOAT16 && input.dataType != FLOAT16 && input.dataType != FLOAT32) ||
                input.dims.size() != 2 || input.dims[0] != 1 ||
                input.dims[1] <= 0 || input.dims[1] > 24576 ||
                (index.dataType != INT32 && index.dataType != INT32PARAM) ||
                score.dataType != FLOAT32 || index.dims != score.dims ||
                index.dims.size() != 2 || index.dims[0] != 1 ||
                index.dims[1] <= 0 || index.dims[1] > 16) return false;
        const Data *src[] = {&input, &index, &score};
        Data *dst[] = {&hostInput, &hostIndex, &hostScore};
        size_t offsets[3], bytes = 0;
        int copies = 0;
        for (int i = 0; i < 3; ++i) {
            if (src[i]->multiDeviceData || dst[i]->multiDeviceData ||
                    dst[i]->isFake || dst[i]->cudaDataBorrowed) return false;
            for (int j = 0; j < 3; ++j) if (dst[i] == src[j] || (i != j && dst[i] == dst[j])) return false;
            if (src[i]->dataDevice == CUDA) {
                if (!src[i]->cudaData) return false;
                const int owner = src[i]->dataDeviceIds.empty()
                    ? GetPointerDeviceId(src[i]->cudaData) : src[i]->dataDeviceIds[0];
                if (owner != device) return false;
                ++copies;
            } else if (src[i]->dataDevice != CPU || !src[i]->cpuData) return false;
            offsets[i] = bytes;
            bytes += (src[i]->GetBytes() + 255) & ~size_t(255);
        }
        DeviceGuard guard(device);
        if (copies && !Ensure(download, downloadCapacity, bytes)) return false;
        for (int i = 0; i < 3; ++i) {
            if (dst[i]->dataDevice != CPU || dst[i]->dataType != src[i]->dataType) {
                dst[i]->FreeSpace(); dst[i]->dataDevice = CPU;
                dst[i]->dataDeviceIds.clear();
                dst[i]->dataType = src[i]->dataType; dst[i]->UpdateUnitSize();
            }
            dst[i]->Resize(src[i]->dims); dst[i]->Allocate(false);
        }
        struct Drain {
            bool pending = false;
            ~Drain() { if (pending) FastllmCudaSyncCurrentThreadStream(); }
        } drain;
        uint64_t transferred = 0;
        for (int i = 0; i < 3; ++i) if (src[i]->dataDevice == CUDA) {
            drain.pending = true;
            if (!FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(
                    static_cast<uint8_t *>(download) + offsets[i], src[i]->cudaData, src[i]->GetBytes())) {
                FastllmCudaSyncCurrentThreadStream(); drain.pending = false;
                FastllmCudaClearThreadError(); return false;
            }
            transferred += src[i]->GetBytes();
        }
        if (copies) {
            if (!downloadReady) downloadReady = FastllmCudaEventCreate();
            FastllmCudaEventRecordCurrentThread(downloadReady);
            const auto begin = std::chrono::steady_clock::now();
            FastllmCudaEventSynchronize(downloadReady);
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - begin).count();
            drain.pending = false;
            DiskMoeHandoffCounters::downloads.fetch_add(copies, std::memory_order_relaxed);
            DiskMoeHandoffCounters::downloadBytes.fetch_add(transferred, std::memory_order_relaxed);
            DiskMoeHandoffCounters::waitUs.fetch_add(us, std::memory_order_relaxed);
        }
        for (int i = 0; i < 3; ++i)
            memcpy(dst[i]->cpuData, src[i]->dataDevice == CUDA
                ? static_cast<uint8_t *>(download) + offsets[i] : src[i]->cpuData, src[i]->GetBytes());
        return true;
    }
    void Upload(const Data &source, Data &output) {
        DeviceGuard guard(device);
        if (uploadPending) {
            FastllmCudaEventSynchronize(uploadDone); uploadPending = false;
        }
        const size_t bytes = source.GetBytes();
        const bool pinned = Ensure(upload, uploadCapacity, bytes);
        if (pinned) memcpy(upload, source.cpuData, bytes);
        // Output is write-only, as in Executor. Submit from stable pinned
        // storage on the producer stream without a synchronous H2D handoff.
        if (output.dataDevice != CUDA || output.dataType != source.dataType) {
            output.FreeSpace(); output.dataDevice = CUDA;
            output.dataType = source.dataType; output.UpdateUnitSize();
        }
        output.dataDeviceIds = {device}; output.Resize(source.dims); output.Allocate(false);
        if (pinned && FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(output.cudaData, upload, bytes)) {
            if (!uploadDone) uploadDone = FastllmCudaEventCreate();
            FastllmCudaEventRecordCurrentThread(uploadDone);
            uploadPending = true;
        } else {
            FastllmCudaSyncCurrentThreadStream(); FastllmCudaClearThreadError();
            FastllmCudaCopyFromHostToDevice(output.cudaData, source.cpuData, bytes);
        }
        DiskMoeHandoffCounters::uploads.fetch_add(1, std::memory_order_relaxed);
        DiskMoeHandoffCounters::uploadBytes.fetch_add(bytes, std::memory_order_relaxed);
    }
};
} // namespace fastllm
