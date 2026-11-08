#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
namespace fastllm { namespace p0snapshot {
// All digests cover actual valid bytes, never a descriptor or pointer value.
static_assert(sizeof(int)==4 && sizeof(float)==4,"snapshot digest scalar width");
inline uint64_t ContentDigest(const void *data, size_t bytes) {
    uint64_t hash=14695981039346656037ull;
    const auto *p=static_cast<const unsigned char *>(data);
    for (size_t i=0;i<bytes;++i) hash=(hash^p[i])*1099511628211ull;
    return hash;
}
struct Observation {
    std::string event;
    uint64_t generation=0, monoNs=0;
    size_t seq=0, tensor=0, head=0, tokens=0, width=0, strideBytes=0;
    size_t sourceOffset=0, destinationOffset=0, bytes=0;
    size_t tensorHeads=0, sourceExtent=0, sourceAllocationBytes=0;
    size_t deviceAccepted=0, deviceTerminal=0, devicePending=0;
    int device=-1, dtype=0;
    uint64_t digest=0, producerId=0;
    size_t accepted=0, terminal=0, pending=0, sourceRefs=0;
    size_t currentBytes=0, currentLeases=0, peakBytes=0, peakLeases=0;
    bool success=true;
};
using Observer=std::function<void(const Observation &)>;
// Reporting is optional and must never alter the operation being reported.
// The error sink is separately protected; callers can use allocation-free I/O.
template<class F,class E> bool ObserveNoThrow(F &&observe,E &&error) noexcept {
    try { observe(); return true; }
    catch (...) { try { error(); } catch (...) {} return false; }
}

template<class F> class ScopeExit {
    F f; bool active=true;
public:
    explicit ScopeExit(F fn):f(std::move(fn)) {}
    ~ScopeExit() noexcept { if (active) { try { f(); } catch (...) {} } }
    ScopeExit(const ScopeExit &)=delete;
    ScopeExit &operator=(const ScopeExit &)=delete;
};
template<class F> ScopeExit(F) -> ScopeExit<F>;
}} // namespace fastllm::p0snapshot
