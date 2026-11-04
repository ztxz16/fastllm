#include "fastkv/fastkv.h"

// Keep ordinary FastLLM builds available on platforms without the POSIX
// transport implementation. Selecting PD fails explicitly at configuration.
namespace fastkv {
struct Service::Impl {};
Service::Service(Options) {
    throw std::runtime_error("FastKV networking currently requires a POSIX platform");
}
Service::~Service() = default;
void Service::Start(const std::string &, Handler) {
    throw std::runtime_error("FastKV transport unavailable");
}
void Service::Stop() {}
std::string Service::Address() const {
    return {};
}
Message Service::Call(const std::string &, const Message &) {
    throw std::runtime_error("FastKV transport unavailable");
}
std::future<Message> Service::Submit(const std::string &, Message) {
    throw std::runtime_error("FastKV transport unavailable");
}
Metrics Service::Stats() const {
    return {};
}
bool Service::HasRdma() {
    return false;
}
} // namespace fastkv
