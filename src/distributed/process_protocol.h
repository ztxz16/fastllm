#ifndef FASTLLM_PROCESS_PROTOCOL_H
#define FASTLLM_PROCESS_PROTOCOL_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastllm {
namespace distributed {
namespace protocol {

using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
constexpr int kWorkerFd = 3;
constexpr uint32_t kVersion = 4;
constexpr uint32_t kMaxPayload = 1024 * 1024;
enum class Kind : uint32_t {
    Hello = 1, CreateId, UniqueId, Initialize, Ready,
    Ping, Pong, Shutdown, Bye, Error, AllReduce, AllReduceResult,
    LoadQwen3, ModelLoaded, ReadWeight, WeightData, UnloadQwen3, ModelUnloaded,
    ForwardLayer, LayerOutput, PrefillQwen3, PrefillLogits,
    BeginQwen3Request, DecodeQwen3, EndQwen3Request, Qwen3RequestEnded,
    QueryQwen3Request, Qwen3RequestState, ClearQwen3Prefix, Qwen3PrefixCleared,
    LoadQwenMoe, ForwardMoeLayer, MoeLayerOutput
};
struct Message {
    Kind kind;
    uint64_t session;
    uint64_t sequence;
    std::vector<uint8_t> payload;
};

/** 父进程借用的 worker 句柄；仅在进程组互斥锁保护下使用，不负责关闭或回收。 */
struct WorkerWatch {
    int rank;
    int fd;
    int pidfd;
};

/** 单次控制事务的退出监控上下文；session/sequence 用于识别退出前留下的错误回复。 */
struct WaitContext {
    std::vector<WorkerWatch> workers;
    uint64_t session = 0;
    uint64_t sequence = 0;
};

/** 携带实际故障 rank 的退出异常；上层不得改写成当前正在收发的 rank。 */
class WorkerExited : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void Put32(std::vector<uint8_t> &bytes, uint32_t value);
uint32_t Get32(const std::vector<uint8_t> &bytes, size_t offset);
void Put64(std::vector<uint8_t> &bytes, uint64_t value);
uint64_t Get64(const std::vector<uint8_t> &bytes, size_t offset);
void Send(int fd, const Message &message, Deadline deadline, const WaitContext *context = nullptr);
Message Receive(int fd, Deadline deadline, const WaitContext *context = nullptr);
void CheckWorkerExits(const WaitContext &context);
void Validate(const Message &message, Kind kind, uint64_t session, uint64_t sequence);
std::string ErrorText(const Message &message);

} // namespace protocol
} // namespace distributed
} // namespace fastllm
#endif
