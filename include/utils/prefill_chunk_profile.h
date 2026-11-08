#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace fastllm {

// Host call boundaries only. Never adds a CUDA event, synchronization, lock,
// allocation, or scheduling decision. Disabled collection never reads a clock.
class PrefillChunkProfile {
public:
    enum class Phase : std::size_t {
        Begin, TargetBegin, TargetEnd, DraftBegin, DraftEnd,
        CheckpointBegin, CheckpointEnd, Count
    };
    using Clock = uint64_t (*)();

    PrefillChunkProfile(bool enabled, Clock clock)
        : enabled_(enabled), clock_(clock) {}

    void Mark(Phase phase) {
        if (!enabled_ || !valid_) return;
        if (clock_ == nullptr) {
            valid_ = false;
            return;
        }
        MarkAt(phase, clock_());
    }

    void MarkAt(Phase phase, uint64_t timestamp) {
        if (!enabled_ || !valid_) return;
        const std::size_t index = static_cast<std::size_t>(phase);
        if (index >= times_.size() || index != next_ || timestamp == 0 ||
            (next_ > 0 && timestamp < times_[next_ - 1])) {
            valid_ = false;
            return;
        }
        times_[next_++] = timestamp;
    }

    bool Enabled() const { return enabled_; }
    bool Complete() const {
        return enabled_ && valid_ && next_ == times_.size();
    }
    uint64_t At(Phase phase) const {
        const std::size_t index = static_cast<std::size_t>(phase);
        return Complete() && index < times_.size() ? times_[index] : 0;
    }
    uint64_t Duration(Phase begin, Phase end) const {
        const std::size_t first = static_cast<std::size_t>(begin);
        const std::size_t last = static_cast<std::size_t>(end);
        return Complete() && first <= last && last < times_.size()
            ? times_[last] - times_[first] : 0;
    }

private:
    bool enabled_ = false, valid_ = true;
    Clock clock_ = nullptr;
    std::size_t next_ = 0;
    std::array<uint64_t, static_cast<std::size_t>(Phase::Count)> times_{};
};

} // namespace fastllm
