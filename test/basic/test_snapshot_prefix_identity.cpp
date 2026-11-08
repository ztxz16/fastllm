#include "utils/multimodal_prefix_cache.h"
#include "utils/snapshot_prefix_identity.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

namespace {
std::atomic<bool> countAllocations{false};
std::atomic<size_t> allocationCount{0};

void *Allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
    if (countAllocations.load(std::memory_order_relaxed)) ++allocationCount;
    if (alignment <= alignof(std::max_align_t)) {
        if (void *ptr = std::malloc(size == 0 ? 1 : size)) return ptr;
    } else {
        const std::size_t rounded = (size + alignment - 1) / alignment * alignment;
        if (void *ptr = std::aligned_alloc(alignment, rounded == 0 ? alignment : rounded)) return ptr;
    }
    throw std::bad_alloc();
}
}

void *operator new(std::size_t size) { return Allocate(size); }
void *operator new[](std::size_t size) { return Allocate(size); }
void *operator new(std::size_t size, std::align_val_t alignment) {
    return Allocate(size, static_cast<std::size_t>(alignment));
}
void *operator new[](std::size_t size, std::align_val_t alignment) {
    return Allocate(size, static_cast<std::size_t>(alignment));
}
void operator delete(void *ptr) noexcept { std::free(ptr); }
void operator delete[](void *ptr) noexcept { std::free(ptr); }
void operator delete(void *ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void *ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete(void *ptr, std::align_val_t) noexcept { std::free(ptr); }
void operator delete[](void *ptr, std::align_val_t) noexcept { std::free(ptr); }
void operator delete(void *ptr, std::size_t, std::align_val_t) noexcept { std::free(ptr); }
void operator delete[](void *ptr, std::size_t, std::align_val_t) noexcept { std::free(ptr); }

namespace {
using namespace fastllm;

void Check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "snapshot prefix identity check failed: %s\n", message);
        std::exit(1);
    }
}

std::vector<int> TextPositions(int tokens) {
    std::vector<int> positions;
    positions.reserve(static_cast<size_t>(tokens) * 3);
    for (int axis = 0; axis < 3; ++axis)
        for (int token = 0; token < tokens; ++token) positions.push_back(token);
    return positions;
}

std::vector<int> Tokens(int tokens, int seed = 0) {
    std::vector<int> result;
    result.reserve(tokens);
    for (int token = 0; token < tokens; ++token) result.push_back(seed + token + 1000);
    return result;
}

MultimodalPrefixSpan Span(int begin, int end, uint8_t digest) {
    MultimodalPrefixSpan span;
    span.begin = begin;
    span.end = end;
    span.digest.fill(digest);
    return span;
}

std::string WithProvenance(const std::string &base, const std::string &provenance) {
    std::string result = base;
    const uint32_t size = static_cast<uint32_t>(provenance.size());
    for (int shift = 0; shift < 32; shift += 8)
        result.push_back(static_cast<char>((size >> shift) & 255));
    result += provenance;
    return result;
}

void TestPageKeysPreserveTextPrefixBeforeImage() {
    const auto tokens = Tokens(384);
    const auto positions = TextPositions(384);
    const auto keys = BuildMultimodalPrefixPageKeys(tokens, positions, {Span(256, 320, 7)}, 128);
    Check(keys.size() == 3, "three complete pages expected");
    Check(keys[0].empty() && keys[1].empty(), "pages before appended image remain text keys");
    Check(!keys[2].empty(), "image page receives multimodal key");

    // The image is after the candidate's two pages. A text snapshot must match
    // exactly that prefix, while the page containing the image is out of scope.
    SnapshotPrefixIdentity text;
    Check(BuildSnapshotPrefixIdentity({"text-page-0", "text-page-1"}, nullptr,
                                      "", 2, text), "text snapshot identity builds");
    std::vector<std::string> queryBase(keys.begin(), keys.begin() + 2);
    Check(SnapshotPrefixIdentityMatches(text, {"text-page-0", "text-page-1"},
                                       &queryBase, ""),
          "text snapshot matches multimodal query before trailing image");
}

void TestImageInsideRequiresIdentity() {
    const auto tokens = Tokens(256);
    const auto positions = TextPositions(256);
    const auto keys = BuildMultimodalPrefixPageKeys(tokens, positions, {Span(64, 80, 7)}, 128);
    Check(keys.size() == 2 && !keys[0].empty() && !keys[1].empty(),
          "image inside first page makes both affected pages multimodal");

    SnapshotPrefixIdentity textIdentity;
    Check(BuildSnapshotPrefixIdentity({"text-0", "text-1"}, nullptr, "", 2, textIdentity),
          "text identity builds before in-prefix-image rejection");
    Check(!SnapshotPrefixIdentityMatches(textIdentity, {"text-0", "text-1"}, &keys, ""),
          "text snapshot cannot cover an image inside the candidate prefix");

    // For an in-memory multimodal snapshot the raw key equals its base key.
    const std::vector<std::string> base = {keys[0], keys[1]};
    SnapshotPrefixIdentity identity;
    Check(BuildSnapshotPrefixIdentity(base, &base, "", 2, identity),
          "multimodal snapshot identity builds");
    Check(identity.textPages == 0 && identity.mediaBaseSizes.size() == 2,
          "image inside has no text-only pages");
    Check(SnapshotPrefixIdentityMatches(identity, base, &base, ""),
          "same image and positions match");

    auto changedDigest = BuildMultimodalPrefixPageKeys(
        tokens, positions, {Span(64, 80, 8)}, 128);
    Check(!SnapshotPrefixIdentityMatches(identity, base, &changedDigest, ""),
          "different image digest does not match");

    auto changedPositions = positions;
    changedPositions[64] = 65;
    auto changedMrope = BuildMultimodalPrefixPageKeys(
        tokens, changedPositions, {Span(64, 80, 7)}, 128);
    Check(!SnapshotPrefixIdentityMatches(identity, base, &changedMrope, ""),
          "different MRoPE positions do not match");

    SnapshotPrefixIdentity imageIdentity;
    const std::string provenance = "encoder-provenance-v1";
    std::vector<std::string> persisted = {WithProvenance(keys[0], provenance),
                                          WithProvenance(keys[1], provenance)};
    Check(BuildSnapshotPrefixIdentity(persisted, &base, provenance, 2, imageIdentity),
          "provenance-bearing image identity builds");
    Check(SnapshotPrefixIdentityMatches(imageIdentity, persisted, &base, provenance),
          "same image provenance matches");
    Check(!SnapshotPrefixIdentityMatches(imageIdentity, persisted, &base, "other-provenance"),
          "different provenance does not match");
    Check(!SnapshotPrefixIdentityMatches(imageIdentity, persisted, &base, ""),
          "missing provenance does not match");
    Check(!SnapshotPrefixIdentityMatches(imageIdentity, persisted, nullptr, provenance),
          "multimodal image snapshot does not match text query");
}

void TestMalformedAndIncompleteIdentityFailsClosed() {
    SnapshotPrefixIdentity identity;
    Check(!BuildSnapshotPrefixIdentity({}, nullptr, "", 0, identity), "zero-page identity fails");
    const std::vector<std::string> twoBases = {"base", "base2"};
    const std::vector<std::string> oneBase = {"base"};
    Check(!BuildSnapshotPrefixIdentity({"raw"}, &twoBases, "", 2, identity),
          "raw vector shorter than page count fails");
    Check(!BuildSnapshotPrefixIdentity({"raw", "raw2"}, &oneBase, "", 2, identity),
          "base vector shorter than page count fails");
    const std::vector<std::string> base = {"media-base"};
    Check(!BuildSnapshotPrefixIdentity({"wrong"}, &base, "", 1, identity),
          "raw/base mismatch fails");
    Check(!BuildSnapshotPrefixIdentity({"media-base\x01"}, &base, "prov", 1, identity),
          "truncated provenance suffix fails");
    auto malformed = WithProvenance(base[0], "prov");
    malformed[base[0].size()] = 3;
    Check(!BuildSnapshotPrefixIdentity({malformed}, &base, "prov", 1, identity),
          "incorrect encoded provenance length fails");
    malformed = WithProvenance(base[0], "prov");
    malformed.back() = 'x';
    Check(!BuildSnapshotPrefixIdentity({malformed}, &base, "prov", 1, identity),
          "incorrect provenance bytes fail");

    Check(BuildSnapshotPrefixIdentity({"plain"}, nullptr, "", 1, identity),
          "plain text identity builds");
    const std::vector<std::string> incomplete;
    Check(!SnapshotPrefixIdentityMatches(identity, {}, &incomplete, ""),
          "incomplete query base fails closed");
}

void TestTrailingImageAndTextDirection() {
    const auto tokens = Tokens(256);
    const auto positions = TextPositions(256);
    const auto keys = BuildMultimodalPrefixPageKeys(tokens, positions, {Span(200, 240, 3)}, 128);
    Check(keys.size() == 2 && keys[0].empty() && !keys[1].empty(),
          "trailing image starts after first text page");

    SnapshotPrefixIdentity text;
    Check(BuildSnapshotPrefixIdentity({"text-page-0"}, nullptr, "", 1, text),
          "text identity builds for trailing-image query");
    std::vector<std::string> firstBase = {keys[0]};
    Check(SnapshotPrefixIdentityMatches(text, {"text-page-0"}, &firstBase, ""),
          "trailing image is ignored outside candidate prefix");

    SnapshotPrefixIdentity image;
    const std::vector<std::string> imageBase = {keys[0], keys[1]};
    Check(BuildSnapshotPrefixIdentity(imageBase, &imageBase, "", 2, image),
          "image identity builds");
    Check(!SnapshotPrefixIdentityMatches(image, {"text-page-0", "text-page-1"}, nullptr, ""),
          "image snapshot does not falsely match text query");
}

void TestTextIdentityMatcherDoesNotAllocate() {
    SnapshotPrefixIdentity identity;
    Check(BuildSnapshotPrefixIdentity({"raw-page-0", "raw-page-1"}, nullptr, "", 2, identity),
          "text identity builds for allocation test");
    std::vector<std::string> emptyBase(2);
    const std::vector<std::string> raw = {"snapshot-page-0", "snapshot-page-1"};
    const std::string emptyProvenance;
    countAllocations.store(false, std::memory_order_relaxed);
    allocationCount.store(0, std::memory_order_relaxed);
    countAllocations.store(true, std::memory_order_relaxed);
    for (int iteration = 0; iteration < 1024; ++iteration) {
        Check(SnapshotPrefixIdentityMatches(identity, raw, &emptyBase, emptyProvenance),
              "empty text-base representation remains matched");
        Check(SnapshotPrefixIdentityMatches(identity, raw, nullptr, emptyProvenance),
              "text-only query remains matched through constant-time path");
    }
    countAllocations.store(false, std::memory_order_relaxed);
    Check(allocationCount.load(std::memory_order_relaxed) == 0,
          "text-only matcher performs no heap allocations");
}
} // namespace

int main() {
    TestPageKeysPreserveTextPrefixBeforeImage();
    TestImageInsideRequiresIdentity();
    TestMalformedAndIncompleteIdentityFailsClosed();
    TestTrailingImageAndTextDirection();
    TestTextIdentityMatcherDoesNotAllocate();
    return 0;
}
