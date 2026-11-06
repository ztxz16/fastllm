#include "distributed/process_protocol.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <iostream>
#include <map>
#include <poll.h>
#include <stdexcept>
#include <thread>
#include <unistd.h>

#ifdef FASTLLM_PROCESS_TP_CUDA
#include <cuda_runtime_api.h>
#include <nccl.h>
#endif

using namespace fastllm::distributed::protocol;

namespace {
#ifdef FASTLLM_PROCESS_TP_CUDA
void CheckCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(result));
    }
}

void CheckNccl(ncclResult_t result) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("NCCL: ") + ncclGetErrorString(result));
    }
}
#endif

/**
 * 持有单个 worker 的 CUDA/NCCL 资源，不创建模型或 KV。
 *
 * 失败时由进程退出释放 CUDA 上下文；正常退出必须先同步，再销毁通信组和缓冲。
 * 不在异常析构路径进入可能阻塞的 NCCL 清理，由父进程负责超时终止。
 */
struct DeviceRuntime {
    bool controlOnly;
#ifdef FASTLLM_PROCESS_TP_CUDA
    cudaStream_t stream = nullptr;
    ncclComm_t comm = nullptr;
    float *buffer = nullptr;
#endif

    void Bind(int device) {
        if (controlOnly) {
            return;
        }
#ifdef FASTLLM_PROCESS_TP_CUDA
        CheckCuda(cudaSetDevice(device));
        CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&buffer), sizeof(float)));
#else
        throw std::runtime_error("worker built without CUDA/NCCL; use control-only for transport checks");
#endif
    }

    std::vector<uint8_t> CreateId() {
#ifdef FASTLLM_PROCESS_TP_CUDA
        ncclUniqueId id;
        CheckNccl(ncclGetUniqueId(&id));
        const auto *bytes = reinterpret_cast<const uint8_t*>(&id);
        return {bytes, bytes + sizeof(id)};
#else
        throw std::runtime_error("NCCL unavailable");
#endif
    }

    void Initialize(int world, int rank, const std::vector<uint8_t> &payload) {
        if (controlOnly) {
            if (!payload.empty()) {
                throw std::runtime_error("control-only initialization must have empty payload");
            }
            return;
        }
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (payload.size() != sizeof(ncclUniqueId)) {
            throw std::runtime_error("NCCL unique ID size mismatch");
        }
        ncclUniqueId id;
        memcpy(&id, payload.data(), sizeof(id));
        CheckNccl(ncclCommInitRank(&comm, world, id, rank));
#endif
    }

    std::vector<uint8_t> AllReduce(int rank) {
        if (controlOnly) {
            throw std::runtime_error("control-only worker cannot run AllReduce");
        }
#ifdef FASTLLM_PROCESS_TP_CUDA
        float value = static_cast<float>(rank + 1);
        CheckCuda(cudaMemcpyAsync(buffer, &value, sizeof(value), cudaMemcpyHostToDevice, stream));
        CheckNccl(ncclAllReduce(buffer, buffer, 1, ncclFloat, ncclSum, comm, stream));
        CheckCuda(cudaMemcpyAsync(&value, buffer, sizeof(value), cudaMemcpyDeviceToHost, stream));
        CheckCuda(cudaStreamSynchronize(stream));
        ncclResult_t asynchronous;
        CheckNccl(ncclCommGetAsyncError(comm, &asynchronous));
        CheckNccl(asynchronous);
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        std::vector<uint8_t> result;
        Put32(result, bits);
        return result;
#else
        throw std::runtime_error("NCCL unavailable");
#endif
    }

    void Shutdown() {
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (!controlOnly) {
            CheckCuda(cudaStreamSynchronize(stream));
            CheckNccl(ncclCommDestroy(comm));
            comm = nullptr;
            CheckCuda(cudaFree(buffer));
            buffer = nullptr;
            CheckCuda(cudaStreamDestroy(stream));
            stream = nullptr;
        }
#endif
    }

    void CheckHealth() {
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (!controlOnly) {
            ncclResult_t asynchronous;
            CheckNccl(ncclCommGetAsyncError(comm, &asynchronous));
            CheckNccl(asynchronous);
        }
#endif
    }
};

uint64_t ParseUnsigned(const std::string &text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("worker expects unsigned decimal arguments");
    }
    return std::stoull(text);
}
} // namespace

int main(int argc, char **argv) {
    uint64_t session = 0;
    uint64_t sequence = 0;
    bool canReport = false;
    try {
        std::map<std::string, uint64_t> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || !args.emplace(argv[i], ParseUnsigned(argv[i + 1])).second) {
                throw std::runtime_error("invalid/duplicate worker arguments");
            }
        }
        if (args.size() != 7 || args.at("--control-fd") != uint64_t(kWorkerFd) ||
            args.at("--world-size") == 0 || args.at("--world-size") > 1024 ||
            args.at("--rank") >= args.at("--world-size") || args.at("--device") > INT_MAX ||
            args.at("--control-only") > 1 || args.at("--parent-pid") == 0 ||
            args.at("--parent-pid") > INT_MAX || args.at("--session") == 0) {
            throw std::runtime_error("invalid worker identity");
        }
        session = args.at("--session");
        pid_t parentPid = static_cast<pid_t>(args.at("--parent-pid"));
        if (getppid() != parentPid) {
            throw std::runtime_error("worker parent is unavailable");
        }
        canReport = true;
        // 不用 PR_SET_PDEATHSIG：它跟随创建线程退出，可能误杀仍有效的进程组。
        // 独立监控可在主线程阻塞于 CUDA/NCCL 时响应父进程死亡或控制连接关闭。
        std::thread([parentPid]() {
            while (true) {
                pollfd item{kWorkerFd, POLLRDHUP, 0};
                int result = poll(&item, 1, 250);
                if (getppid() != parentPid ||
                    (result > 0 && (item.revents & (POLLRDHUP | POLLHUP | POLLERR | POLLNVAL)))) {
                    _exit(1);
                }
            }
        }).detach();
        int rank = static_cast<int>(args.at("--rank"));
        int world = static_cast<int>(args.at("--world-size"));
        int device = static_cast<int>(args.at("--device"));
        DeviceRuntime runtime{args.at("--control-only") != 0};
        runtime.Bind(device);

        std::vector<uint8_t> hello;
        for (uint32_t value : {uint32_t(rank), uint32_t(device), uint32_t(getpid()), uint32_t(!runtime.controlOnly)}) {
            Put32(hello, value);
        }
        int cudaVersion = 0;
        int ncclVersion = 0;
        std::string deviceName = "control-only (no GPU)";
        std::vector<uint8_t> deviceUuid(16, 0);
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (!runtime.controlOnly) {
            cudaDeviceProp properties;
            CheckCuda(cudaGetDeviceProperties(&properties, device));
            CheckCuda(cudaRuntimeGetVersion(&cudaVersion));
            CheckNccl(ncclGetVersion(&ncclVersion));
            deviceName = properties.name;
            memcpy(deviceUuid.data(), properties.uuid.bytes, deviceUuid.size());
        }
#endif
        Put32(hello, static_cast<uint32_t>(cudaVersion));
        Put32(hello, static_cast<uint32_t>(ncclVersion));
        hello.insert(hello.end(), deviceUuid.begin(), deviceUuid.end());
        hello.insert(hello.end(), deviceName.begin(), deviceName.end());
        Send(kWorkerFd, {Kind::Hello, session, 0, hello}, Clock::now() + std::chrono::seconds(10));

        bool initialized = false;
        bool idCreated = false;
        while (true) {
            auto message = Receive(kWorkerFd, Deadline::max());
            if (message.session != session || message.sequence <= sequence) {
                throw std::runtime_error("worker session/sequence mismatch");
            }
            sequence = message.sequence;
            Kind response;
            std::vector<uint8_t> payload;
            if (message.kind != Kind::Initialize && !message.payload.empty()) {
                throw std::runtime_error("unexpected worker command payload");
            }
            switch (message.kind) {
                case Kind::CreateId:
                    if (rank != 0 || runtime.controlOnly || initialized || idCreated) {
                        throw std::runtime_error("invalid CreateId state");
                    }
                    payload = runtime.CreateId();
                    idCreated = true;
                    response = Kind::UniqueId;
                    break;
                case Kind::Initialize:
                    if (initialized || (!runtime.controlOnly && rank == 0 && !idCreated)) {
                        throw std::runtime_error("invalid Initialize state");
                    }
                    runtime.Initialize(world, rank, message.payload);
                    initialized = true;
                    response = Kind::Ready;
                    break;
                case Kind::Ping:
                    if (!initialized) {
                        throw std::runtime_error("Ping before Initialize");
                    }
                    runtime.CheckHealth();
                    response = Kind::Pong;
                    break;
                case Kind::AllReduce:
                    if (!initialized) {
                        throw std::runtime_error("AllReduce before Initialize");
                    }
                    payload = runtime.AllReduce(rank);
                    response = Kind::AllReduceResult;
                    break;
                case Kind::Shutdown:
                    if (!initialized) {
                        throw std::runtime_error("Shutdown before Initialize");
                    }
                    runtime.Shutdown();
                    Send(kWorkerFd, {Kind::Bye, session, sequence, {}}, Clock::now() + std::chrono::seconds(10));
                    _exit(0);
                default:
                    throw std::runtime_error("unknown worker command");
            }
            Send(kWorkerFd, {response, session, sequence, payload}, Clock::now() + std::chrono::seconds(10));
        }
    } catch (const std::exception &e) {
        if (canReport) {
            try {
                std::string reason = e.what();
                reason.resize(std::min<size_t>(reason.size(), 4096));
                Send(kWorkerFd, {Kind::Error, session, sequence, {reason.begin(), reason.end()}},
                     Clock::now() + std::chrono::seconds(1));
            } catch (...) {
            }
        }
        std::cerr << "fastllm_tp_worker: " << e.what() << std::endl;
        // 避免 CUDA/NCCL 异常退出时静态析构再次阻塞；父进程负责回收。
        _exit(1);
    }
}
