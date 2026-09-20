#ifdef NDEBUG
#undef NDEBUG
#endif
#include "utils/multimodal_prefix_cache.h"
#include <cassert>
#include <iostream>
#include <numeric>

using namespace fastllm;

static std::vector<int> Positions(int length) {
    std::vector<int> result(length * 3);
    for (int axis = 0; axis < 3; ++axis) {
        std::iota(result.begin() + axis * length, result.begin() + (axis + 1) * length, 0);
    }
    return result;
}

int main() {
    std::vector<int> tokens(1024, 42);
    MultimodalPrefixSpan a, b;
    a.begin = 140; a.end = 350; a.digest.fill(1);
    b.begin = 600; b.end = 710; b.digest.fill(2);
    auto positions = Positions(tokens.size());
    auto original = BuildMultimodalPrefixPageKeys(tokens, positions, {a}, 128);
    auto changedA = a;
    changedA.digest[31] ^= 1; // All 256 digest bits participate in equality.
    auto changed = BuildMultimodalPrefixPageKeys(tokens, positions, {changedA}, 128);
    assert(original[0] == changed[0]);
    assert(original[1] != changed[1] && original[2] != changed[2]);
    auto appended = BuildMultimodalPrefixPageKeys(tokens, positions, {a, b}, 128);
    assert(std::equal(original.begin(), original.begin() + 4, appended.begin()));
    assert(original[4] != appended[4]);
    positions[1024 + 130]++;
    auto moved = BuildMultimodalPrefixPageKeys(tokens, positions, {a}, 128);
    assert(original[1] != moved[1]);
    assert(BuildMultimodalPrefixPageKeys(tokens, {}, {a}, 128).empty());
    for (int size : {127, 128, 129, 256, 2048, 8192, 8193, 10101}) {
        const auto pages = BuildMultimodalPrefixPageKeys(
            std::vector<int>(size, 42), Positions(size), {}, 128);
        assert(pages.size() == (size_t)size / 128);
    }
    std::cout << "Multimodal prefix identity and complete-page keys: PASS\n";
}
