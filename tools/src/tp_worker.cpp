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
#include "distributed/qwen3_device_weights.h"
#include "distributed/qwen3_layer.h"
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
 * 持有单个 worker 的 CUDA/NCCL、可选 Qwen3 权重和管理请求 KV/Graph 的执行器。
 *
 * 失败时由进程退出释放 CUDA 上下文；正常退出必须先同步，再销毁通信组和缓冲。
 * 不在异常析构路径进入可能阻塞的 NCCL 清理，由父进程负责超时终止。
 */
struct DeviceRuntime {
    bool controlOnly;
    bool cudaGraph;
#ifdef FASTLLM_PROCESS_TP_CUDA
    cudaStream_t stream = nullptr;
    ncclComm_t comm = nullptr;
    float *buffer = nullptr;
    std::unique_ptr<fastllm::distributed::Qwen3DeviceWeights> weights;
    std::unique_ptr<fastllm::distributed::Qwen3LayerExecutor> layerExecutor;
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
            UnloadWeights();
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

    /**
     * 按控制消息创建本 rank 的 GPU 权重，任何失败交由主循环终止进程。
     * @param rank 本 worker 的 rank。
     * @param world TP 进程数。
     * @param payload 两个 uint64 配额及不含 NUL 的模型路径。
     * @return 张量数、GPU 字节数、源读取字节数；不是前向就绪标志。
     */
    std::vector<uint8_t> LoadWeights(int rank, int world, const std::vector<uint8_t> &payload) {
        if (controlOnly) {
            throw std::runtime_error("Qwen3 weights require CUDA");
        }
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (weights) {
            throw std::runtime_error("Qwen3 weights already loaded; unload first");
        }
        if (payload.size() <= 16) {
            throw std::runtime_error("invalid Qwen3 load payload");
        }
        std::string directory(payload.begin() + 16, payload.end());
        if (directory.find('\0') != std::string::npos) {
            throw std::runtime_error("invalid Qwen3 path");
        }
        fastllm::distributed::Qwen3ShardLoader loader(directory, rank, world);
        weights = std::make_unique<fastllm::distributed::Qwen3DeviceWeights>(
            loader, stream, Get64(payload, 0), Get64(payload, 8));
        std::vector<uint8_t> result;
        Put32(result, weights->TensorCount());
        Put64(result, weights->DeviceBytes());
        Put64(result, weights->SourceBytesRead());
        return result;
#else
        throw std::runtime_error("worker built without CUDA");
#endif
    }

    /**
     * 仅目标 rank 在 worker 执行 stream 上读回 GPU 权重，其他 rank 返回空响应。
     * @param rank 当前 rank。
     * @param world 组大小，用于验证目标 rank。
     * @param payload 目标 rank、张量序号、字节偏移及读取长度。
     * @return GPU 读回的 FP16 字节；未加载、越界或 CUDA 错误抛异常。
     */
    std::vector<uint8_t> ReadWeights(int rank, int world, const std::vector<uint8_t> &payload) {
        if (controlOnly || payload.size() != 20 || Get32(payload, 0) >= uint32_t(world)) {
            throw std::runtime_error("invalid Qwen3 read payload/state");
        }
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (!weights) {
            throw std::runtime_error("Qwen3 weights not loaded");
        }
        if (Get32(payload, 0) != uint32_t(rank)) {
            return {};
        }
        return weights->Read(Get32(payload, 4), Get64(payload, 8), Get32(payload, 16));
#else
        throw std::runtime_error("worker built without CUDA");
#endif
    }

    /** 同步并卸载可选权重，重复调用无副作用；CUDA 失败交由主循环退出回收。 */
    void UnloadWeights() {
        if (controlOnly) {
            throw std::runtime_error("Qwen3 weights require CUDA");
        }
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (layerExecutor) {
            layerExecutor->Clear();
            layerExecutor.reset();
        }
        if (weights) {
            weights->Clear();
            weights.reset();
        }
#endif
    }

    /**
     * 解码有界单层任务并复用当前权重、stream 和 NCCL 通信组执行。
     * @param payload 六个 uint32 头字段及行主序 FP32 激活，禁止传递进程指针。
     * @return 同形状 FP32 输出字节；失败终止 worker，不能复用部分状态。
     */
    std::vector<uint8_t> ForwardLayer(const std::vector<uint8_t> &payload) {
        if (controlOnly || payload.size() < 24 || (payload.size() - 24) % 4 ||
            !Get32(payload, 20) || Get32(payload, 20) > INT_MAX) {
            throw std::runtime_error("invalid Qwen3 layer payload/state");
        }
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (!weights) {
            throw std::runtime_error("Qwen3 weights not loaded");
        }
        std::vector<float> input((payload.size() - 24) / 4);
        for (size_t i = 0; i < input.size(); ++i) {
            uint32_t bits = Get32(payload, 24 + i * 4);
            memcpy(&input[i], &bits, sizeof(bits));
        }
        if (!layerExecutor) {
            layerExecutor = std::make_unique<fastllm::distributed::Qwen3LayerExecutor>(*weights, stream, comm, cudaGraph);
        }
        auto output = layerExecutor->Forward(Get32(payload, 0), Get32(payload, 4), Get32(payload, 8),
            Get32(payload, 12), static_cast<fastllm::distributed::Qwen3LayerStage>(Get32(payload, 16)), input);
        std::vector<uint8_t> result;
        for (float value : output) {
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            Put32(result, bits);
        }
        return result;
#else
        throw std::runtime_error("worker built without CUDA");
#endif
    }

    /**
     * 解码无状态 Prefill、请求初始化或单 token Decode，统一打包词表分片。
     * @param payload 前两项为 position/token、timeout；Begin 另有 capacity，之后为 Prompt ID。
     * @param kind PrefillQwen3、BeginQwen3Request 或 DecodeQwen3。
     * @return 三个 uint64（词表起点、长度、总长）及 FP32 logits；失败由 worker 终止整组。
     */
    std::vector<uint8_t> Prefill(const std::vector<uint8_t> &payload, Kind kind) {
        const bool decode = kind == Kind::DecodeQwen3, begin = kind == Kind::BeginQwen3Request;
        const size_t header = begin ? 12 : 8;
        const bool valid = decode ? payload.size() == 8 :
            (payload.size() >= header + 4 && payload.size() <= header + 128 * 4 && (payload.size() - header) % 4 == 0);
        if (controlOnly || !valid || !Get32(payload, 4) || Get32(payload, 4) > INT_MAX) {
            throw std::runtime_error("invalid Qwen3 prefill payload/state");
        }
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (!weights) {
            throw std::runtime_error("Qwen3 weights not loaded");
        }
        std::vector<uint32_t> ids;
        for (size_t offset = header; offset < payload.size(); offset += 4) {
            ids.push_back(Get32(payload, offset));
        }
        if (!layerExecutor) {
            layerExecutor = std::make_unique<fastllm::distributed::Qwen3LayerExecutor>(*weights, stream, comm, cudaGraph);
        }
        auto logits = decode ? layerExecutor->Decode(Get32(payload, 0)) :
            begin ? layerExecutor->BeginRequest(ids, Get32(payload, 8), Get32(payload, 0)) :
                    layerExecutor->Prefill(ids, Get32(payload, 0));
        std::vector<uint8_t> result;
        Put64(result, weights->Plan().vocabBegin);
        Put64(result, logits.size());
        Put64(result, weights->Plan().vocabSize);
        for (float value : logits) {
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            Put32(result, bits);
        }
        return result;
#else
        throw std::runtime_error("worker built without CUDA");
#endif
    }

    /** 查询或同步释放唯一请求 KV；未加载模型也返回全零状态，禁止 control-only。 */
    std::vector<uint8_t> RequestState(bool end) {
        if (controlOnly) {
            throw std::runtime_error("Qwen3 request requires CUDA");
        }
#ifdef FASTLLM_PROCESS_TP_CUDA
        if (end && layerExecutor) {
            layerExecutor->EndRequest();
        }
        auto info = layerExecutor ? layerExecutor->RequestInfo() : fastllm::distributed::Qwen3RequestInfo{};
        if (!layerExecutor) info.graph.enabled = cudaGraph;
        std::vector<uint8_t> result;
        Put32(result, info.active);
        Put32(result, info.tokens);
        Put32(result, info.capacity);
        Put32(result, info.positionBegin);
        Put64(result, info.deviceBytes);
        Put32(result, info.graph.enabled);
        Put32(result, info.graph.active);
        Put64(result, info.graph.captures);
        Put64(result, info.graph.replays);
        Put64(result, info.graph.warmups);
        Put64(result, info.graph.eagerDecodes);
        Put64(result, info.graph.workspaceBytes);
        return result;
#else
        throw std::runtime_error("worker built without CUDA");
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
        if (args.size() != 8 || args.at("--control-fd") != uint64_t(kWorkerFd) ||
            args.at("--world-size") == 0 || args.at("--world-size") > 1024 ||
            args.at("--rank") >= args.at("--world-size") || args.at("--device") > INT_MAX ||
            args.at("--control-only") > 1 || args.at("--cuda-graph") > 1 ||
            (args.at("--control-only") && args.at("--cuda-graph")) || args.at("--parent-pid") == 0 ||
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
        DeviceRuntime runtime{args.at("--control-only") != 0, args.at("--cuda-graph") != 0};
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
            if (message.kind != Kind::Initialize && message.kind != Kind::LoadQwen3 &&
                message.kind != Kind::ReadWeight && message.kind != Kind::ForwardLayer &&
                message.kind != Kind::PrefillQwen3 && message.kind != Kind::BeginQwen3Request &&
                message.kind != Kind::DecodeQwen3 && !message.payload.empty()) {
                throw std::runtime_error("unexpected worker command payload");
            }
            switch (message.kind) {
                case Kind::EndQwen3Request:
                case Kind::QueryQwen3Request:
                    if (!initialized) {
                        throw std::runtime_error("Qwen3 request state before Initialize");
                    }
                    payload = runtime.RequestState(message.kind == Kind::EndQwen3Request);
                    response = message.kind == Kind::EndQwen3Request ? Kind::Qwen3RequestEnded : Kind::Qwen3RequestState;
                    break;
                case Kind::BeginQwen3Request:
                case Kind::DecodeQwen3:
                case Kind::PrefillQwen3:
                    if (!initialized) {
                        throw std::runtime_error("PrefillQwen3 before Initialize");
                    }
                    payload = runtime.Prefill(message.payload, message.kind);
                    response = Kind::PrefillLogits;
                    break;
                case Kind::ForwardLayer:
                    if (!initialized) {
                        throw std::runtime_error("ForwardLayer before Initialize");
                    }
                    payload = runtime.ForwardLayer(message.payload);
                    response = Kind::LayerOutput;
                    break;
                case Kind::LoadQwen3:
                    if (!initialized) {
                        throw std::runtime_error("LoadQwen3 before Initialize");
                    }
                    payload = runtime.LoadWeights(rank, world, message.payload);
                    response = Kind::ModelLoaded;
                    break;
                case Kind::ReadWeight:
                    if (!initialized) {
                        throw std::runtime_error("ReadWeight before Initialize");
                    }
                    payload = runtime.ReadWeights(rank, world, message.payload);
                    response = Kind::WeightData;
                    break;
                case Kind::UnloadQwen3:
                    if (!initialized) {
                        throw std::runtime_error("UnloadQwen3 before Initialize");
                    }
                    runtime.UnloadWeights();
                    response = Kind::ModelUnloaded;
                    break;
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
            auto responseTimeout = message.kind == Kind::ForwardLayer ?
                std::chrono::milliseconds(Get32(message.payload, 20)) :
                (message.kind == Kind::PrefillQwen3 || message.kind == Kind::BeginQwen3Request || message.kind == Kind::DecodeQwen3) ?
                std::chrono::milliseconds(Get32(message.payload, 4)) :
                std::chrono::milliseconds(10000);
            Send(kWorkerFd, {response, session, sequence, payload}, Clock::now() + responseTimeout);
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
