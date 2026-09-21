#ifndef FASTLLM_MULTIMODAL_PREFIX_CACHE_H
#define FASTLLM_MULTIMODAL_PREFIX_CACHE_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace fastllm {
    struct MultimodalPrefixSpan {
        int begin = 0;
        int end = 0;
        std::array<uint8_t, 32> digest{};
    };

    inline void AppendPrefixKeyInt(std::string &key, uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) {
            key.push_back(static_cast<char>((value >> shift) & 255));
        }
    }

    // Keep the complete canonical identity. The trie hash only selects a bucket;
    // equality checks all bytes. Later images cannot change an earlier page key.
    inline std::vector<std::string> BuildMultimodalPrefixPageKeys(
            const std::vector<int> &tokens, const std::vector<int> &positions,
            const std::vector<MultimodalPrefixSpan> &spans, int pageLen) {
        std::vector<std::string> keys;
        if (pageLen <= 0 || positions.size() != tokens.size() * 3) return keys;
        const int length = static_cast<int>(tokens.size());
        for (int start = 0; start + pageLen <= length; start += pageLen) {
            std::string key("qwen35-image-prefix-v1");
            for (int i = start; i < start + pageLen; ++i) {
                AppendPrefixKeyInt(key, tokens[i]);
                for (int axis = 0; axis < 3; ++axis) {
                    AppendPrefixKeyInt(key, positions[axis * length + i]);
                }
            }
            for (size_t i = 0; i < spans.size(); ++i) {
                const auto &span = spans[i];
                if (span.begin >= start + pageLen || span.end <= start) continue;
                AppendPrefixKeyInt(key, static_cast<uint32_t>(i));
                AppendPrefixKeyInt(key, span.begin);
                AppendPrefixKeyInt(key, span.end);
                key.append(reinterpret_cast<const char*>(span.digest.data()), span.digest.size());
            }
            keys.push_back(std::move(key));
        }
        return keys;
    }

}
#endif
