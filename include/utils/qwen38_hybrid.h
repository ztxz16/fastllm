#pragma once

namespace fastllm {
// Model dispatch context, restored on exit (including exceptions). A process
// may load another model while retaining Qwen's automatic environment values.
inline thread_local bool qwen38ForwardActive = false;
class Qwen38ForwardScope {
    bool previous = qwen38ForwardActive;
public:
    Qwen38ForwardScope() { qwen38ForwardActive = true; }
    ~Qwen38ForwardScope() { qwen38ForwardActive = previous; }
    Qwen38ForwardScope(const Qwen38ForwardScope &) = delete;
    Qwen38ForwardScope &operator=(const Qwen38ForwardScope &) = delete;
};
} // namespace fastllm
