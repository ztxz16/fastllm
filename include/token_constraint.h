#ifndef FASTLLM_TOKEN_CONSTRAINT_H
#define FASTLLM_TOKEN_CONSTRAINT_H
#include <cstddef>
#include <cstdint>
#include <vector>
namespace fastllm {
// Immutable byte-DFA/token-mask program. Tokenizers and grammar frontends compile
// this representation on CPU; inference backends need no parser or schema code.
struct TokenConstraintProgram {
    int vocab = 0;
    std::vector<int> transitions;  // state * 256 + byte; -1 rejects
    std::vector<int> maskIds;      // identical masks are interned
    std::vector<uint32_t> masks;
    std::vector<int> tokenOffsets;
    std::vector<uint8_t> tokenBytes, stopTokens;
    size_t Bytes() const {
        return sizeof(int) * (transitions.size() + maskIds.size() + tokenOffsets.size()) +
            sizeof(uint32_t) * masks.size() + tokenBytes.size() + stopTokens.size();
    }
};
} // namespace fastllm
#endif
