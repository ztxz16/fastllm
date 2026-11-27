#include "pd_bindings.h"
#include <cstring>
#include <memory>
#include <thread>
#ifdef FASTLLM_PD_CUDA
#include <cuda_runtime.h>
#endif

namespace fastllm {
namespace pd {

struct PendingCopy {
    int device = 0;
    bool done = false;
    std::string error;
    uint64_t bytes = 0;
    double gpuMs = 0;
#ifdef FASTLLM_PD_CUDA
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr, start = nullptr;
#endif
    std::vector<std::pair<int, std::string>> mappings;
};

/**
 * 单机 IPC 映射和异步拷贝；只在所属传输线程提交/查询。
 *
 * 源/目标是已校验的 FP16/BF16/FP32 [token,head,dim] 区域，dim 连续。
 * 在部分提交失败后仍保留 stream 与映射；实际排空前禁止关闭映射。
 */
class CudaBackend {
    std::thread::id owner;
    bool started = false;
    std::map<std::string, int> devices;
    std::map<std::pair<int, std::string>, std::pair<void *, size_t>> mappings;
    std::vector<std::shared_ptr<PendingCopy>> pending;
    std::vector<std::shared_ptr<PendingCopy>> quarantined;
#ifdef FASTLLM_PD_CUDA
    static void Check(cudaError_t result) {
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("CUDA PD: ") + cudaGetErrorString(result));
    }
    void Dispose(PendingCopy &task) {
        if (task.start) {
            Check(cudaEventDestroy(task.start));
            task.start = nullptr;
        }
        if (task.event) {
            Check(cudaEventDestroy(task.event));
            task.event = nullptr;
        }
        if (task.stream) {
            Check(cudaStreamDestroy(task.stream));
            task.stream = nullptr;
        }
        while (!task.mappings.empty()) {
            auto key = task.mappings.back();
            auto found = mappings.find(key);
            if (found == mappings.end())
                throw std::runtime_error("missing IPC mapping reference");
            if (found->second.second == 1) {
                Check(cudaIpcCloseMemHandle(found->second.first));
                mappings.erase(found);
            } else
                --found->second.second;
            task.mappings.pop_back();
        }
    }
#endif
  public:
    /**
     * 查询本进程可见的 UUID 和有向 P2P 能力，不打开 IPC 或绑定传输线程。
     *
     * access 中每项为 [源 UUID, 目标 UUID]。当前 pull/push 都在目标 GPU
     * 提交拷贝，因此检查目标访问源的能力；设备编号仅在此进程内使用。
     * @return devices 按本地 CUDA 编号排列；access 仅包含可执行的传输边。
     */
    D Topology() const {
#ifdef FASTLLM_PD_CUDA
        int count = 0;
        Check(cudaGetDeviceCount(&count));
        L visible, access;
        for (int device = 0; device < count; ++device) {
            cudaDeviceProp property{};
            Check(cudaGetDeviceProperties(&property, device));
            std::string uuid;
            const char hex[] = "0123456789abcdef";
            for (unsigned char byte : property.uuid.bytes) {
                uuid += hex[byte >> 4];
                uuid += hex[byte & 15];
            }
            visible.append(uuid);
        }
        for (int source = 0; source < count; ++source)
            for (int target = 0; target < count; ++target) {
                int supported = source == target;
                if (!supported)
                    Check(cudaDeviceCanAccessPeer(&supported, target, source));
                if (supported)
                    access.append(list(py::make_tuple(visible[source], visible[target])));
            }
        return D("devices"_a = visible, "access"_a = access);
#else
        throw std::runtime_error("CUDA topology requires USE_CUDA=ON");
#endif
    }

    void Start() {
        if (started) {
            if (owner != std::this_thread::get_id())
                throw std::runtime_error("CUDA IPC must use its owner thread");
            return;
        }
#ifdef FASTLLM_PD_CUDA
        int count = 0;
        Check(cudaGetDeviceCount(&count));
        for (int device = 0; device < count; ++device) {
            cudaDeviceProp property{};
            Check(cudaGetDeviceProperties(&property, device));
            std::string uuid;
            const char hex[] = "0123456789abcdef";
            for (unsigned char byte : property.uuid.bytes) {
                uuid += hex[byte >> 4];
                uuid += hex[byte & 15];
            }
            devices.emplace(uuid, device);
        }
        owner = std::this_thread::get_id();
        started = true;
#else
        throw std::runtime_error("native PD module was built without CUDA; build with USE_CUDA=ON");
#endif
    }

    std::shared_ptr<PendingCopy> Submit(O copies) {
        Start();
#ifdef FASTLLM_PD_CUDA
        if (!py::len(copies))
            throw std::runtime_error("empty copy batch");
        auto task = std::make_shared<PendingCopy>();
        std::string target;
        for (auto c : copies) {
            auto destination = str(get(get(c, "target"), "device"));
            if (target.empty())
                target = destination;
            if (destination != target || !devices.count(destination))
                throw std::runtime_error("target GPU not visible or mixed batch");
            auto source = str(get(get(c, "source"), "device"));
            if (!devices.count(source))
                throw std::runtime_error("source GPU not visible");
            task->device = devices.at(target);
            if (devices.at(source) != task->device) {
                int supported = 0;
                Check(cudaDeviceCanAccessPeer(&supported, task->device, devices.at(source)));
                if (!supported)
                    throw std::runtime_error("GPU pair lacks P2P; CPU fallback disabled");
            }
        }
        Check(cudaSetDevice(task->device));
        std::map<std::string, char *> addresses;
        try {
            for (auto c : copies)
                for (const char *side : {"source", "target"}) {
                    O region = get(c, side);
                    auto memory = str(get(region, "memory"));
                    if (addresses.count(memory))
                        continue;
                    auto key = std::make_pair(task->device, memory);
                    auto found = mappings.find(key);
                    if (found == mappings.end()) {
                        auto encoded =
                            cls("protocol", "ipc_handle")(get(region, "memory")).cast<std::string>();
                        cudaIpcMemHandle_t handle;
                        std::memcpy(&handle, encoded.data(), sizeof(handle));
                        void *address = nullptr;
                        Check(cudaIpcOpenMemHandle(&address, handle, cudaIpcMemLazyEnablePeerAccess));
                        found = mappings.emplace(key, std::make_pair(address, size_t(0))).first;
                    }
                    ++found->second.second;
                    task->mappings.push_back(key);
                    addresses[memory] = static_cast<char *>(found->second.first);
                }
            Check(cudaStreamCreateWithFlags(&task->stream, cudaStreamNonBlocking));
            Check(cudaEventCreate(&task->event));
            Check(cudaEventCreate(&task->start));
        } catch (...) {
            // 此时尚未提交拷贝；若关闭映射失败也保留任务以便显式排空/重试收尾。
            try {
                Dispose(*task);
            } catch (...) {
                task->done = true;
                quarantined.push_back(task);
            }
            throw;
        }
        pending.push_back(task);
        try {
            Check(cudaEventRecord(task->start, task->stream));
            for (auto entry : copies) {
                O c = py::reinterpret_borrow<O>(entry), src = get(c, "source"), dst = get(c, "target");
                I begin = num(get(c, "token_begin")), end = num(get(c, "token_end")),
                  hb = num(get(c, "head_begin")), he = num(get(c, "head_end"));
                I vector = num(get(c, "vector_bytes"));
                bool merged =
                    num(get(src, "head_stride")) == vector && num(get(dst, "head_stride")) == vector;
                for (I head = hb; head < he; head += merged ? he - hb : 1) {
                    size_t width = size_t(vector) * (merged ? size_t(he - hb) : 1),
                           height = size_t(end - begin);
                    char *from =
                        addresses.at(str(get(src, "memory"))) + num(src.attr("byte_offset")(begin, head));
                    char *to =
                        addresses.at(str(get(dst, "memory"))) + num(dst.attr("byte_offset")(begin, head));
                    Check(cudaMemcpy2DAsync(to, num(get(dst, "token_stride")), from,
                                            num(get(src, "token_stride")), width, height, cudaMemcpyDefault,
                                            task->stream));
                    task->bytes += width * height;
                }
            }
            Check(cudaEventRecord(task->event, task->stream));
        } catch (const std::exception &error) {
            task->error = error.what();
        }
        return task;
#else
        return {};
#endif
    }

    bool Query(const std::shared_ptr<PendingCopy> &task) {
        Start();
#ifdef FASTLLM_PD_CUDA
        Check(cudaSetDevice(task->device));
        if (task->done)
            return true;
        if (!task->error.empty()) {
            py::gil_scoped_release release;
            Check(cudaStreamSynchronize(task->stream));
            task->done = true;
            return true;
        }
        auto status = cudaEventQuery(task->event);
        if (status == cudaErrorNotReady)
            return false;
        if (status != cudaSuccess) {
            task->error = cudaGetErrorString(status);
            return false;
        }
        float elapsed = 0;
        Check(cudaEventElapsedTime(&elapsed, task->start, task->event));
        task->gpuMs = elapsed;
        task->done = true;
        return true;
#else
        return false;
#endif
    }

    void Retire(const std::shared_ptr<PendingCopy> &task) {
        Start();
        if (!task->done)
            throw std::runtime_error("cannot retire in-flight CUDA access");
#ifdef FASTLLM_PD_CUDA
        Check(cudaSetDevice(task->device));
        Dispose(*task);
        auto found = std::find(pending.begin(), pending.end(), task);
        if (found == pending.end())
            throw std::runtime_error("unknown CUDA transfer handle");
        pending.erase(found);
#endif
    }
    void Close() {
#ifdef FASTLLM_PD_CUDA
        // 预提交失败没有在途访问；重试关闭残留映射，不能把不可达任务永久留在 pending。
        while (!quarantined.empty()) {
            auto task = quarantined.back();
            Check(cudaSetDevice(task->device));
            Dispose(*task);
            quarantined.pop_back();
        }
#endif
        if (!pending.empty() || !mappings.empty())
            throw std::runtime_error("IPC mappings still have outstanding transfers");
    }
};

void BindCuda(py::module_ &m) {
#ifdef FASTLLM_PD_CUDA
    m.attr("cuda_enabled") = true;
    m.attr("cuda_runtime_headers") = CUDART_VERSION;
#else
    m.attr("cuda_enabled") = false;
    m.attr("cuda_runtime_headers") = 0;
#endif
    py::class_<PendingCopy, std::shared_ptr<PendingCopy>>(m, "Pending")
        .def_readonly("done", &PendingCopy::done)
        .def_readonly("error", &PendingCopy::error)
        .def_readonly("bytes", &PendingCopy::bytes)
        .def_readonly("gpu_ms", &PendingCopy::gpuMs);
    py::class_<CudaBackend, std::shared_ptr<CudaBackend>>(m, "CudaIPC")
        .def(py::init<>())
        .def("topology", &CudaBackend::Topology)
        .def("submit", &CudaBackend::Submit)
        .def("query", &CudaBackend::Query)
        .def("retire", &CudaBackend::Retire)
        .def("close", &CudaBackend::Close);
}
} // namespace pd
} // namespace fastllm
