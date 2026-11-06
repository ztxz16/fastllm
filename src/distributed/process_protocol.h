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
constexpr uint32_t kVersion = 1;
constexpr uint32_t kMaxPayload = 1024 * 1024;
enum class Kind : uint32_t {
    Hello = 1, CreateId, UniqueId, Initialize, Ready,
    Ping, Pong, Shutdown, Bye, Error, AllReduce, AllReduceResult
};
struct Message {
    Kind kind;
    uint64_t session;
    uint64_t sequence;
    std::vector<uint8_t> payload;
};

void Put32(std::vector<uint8_t> &bytes, uint32_t value);
uint32_t Get32(const std::vector<uint8_t> &bytes, size_t offset);
void Send(int fd, const Message &message, Deadline deadline);
Message Receive(int fd, Deadline deadline);
void Validate(const Message &message, Kind kind, uint64_t session, uint64_t sequence);
std::string ErrorText(const Message &message);

} // namespace protocol
} // namespace distributed
} // namespace fastllm
#endif
