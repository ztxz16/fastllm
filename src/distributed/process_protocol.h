#ifndef FASTLLM_PROCESS_PROTOCOL_H
#define FASTLLM_PROCESS_PROTOCOL_H

#include <chrono>
#include <cstddef>
#include <cstdint>
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
    QueryQwen3Request, Qwen3RequestState, ClearQwen3Prefix, Qwen3PrefixCleared
};
struct Message {
    Kind kind;
    uint64_t session;
    uint64_t sequence;
    std::vector<uint8_t> payload;
};

void Put32(std::vector<uint8_t> &bytes, uint32_t value);
uint32_t Get32(const std::vector<uint8_t> &bytes, size_t offset);
void Put64(std::vector<uint8_t> &bytes, uint64_t value);
uint64_t Get64(const std::vector<uint8_t> &bytes, size_t offset);
void Send(int fd, const Message &message, Deadline deadline);
Message Receive(int fd, Deadline deadline);
void Validate(const Message &message, Kind kind, uint64_t session, uint64_t sequence);
std::string ErrorText(const Message &message);

} // namespace protocol
} // namespace distributed
} // namespace fastllm
#endif
