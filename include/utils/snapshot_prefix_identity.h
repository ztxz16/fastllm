#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fastllm {

// Snapshot-owned metadata only: the large image/position key bytes remain in
// the existing immutable raw-key vector. Matching allocates and hashes nothing.
struct SnapshotPrefixIdentity {
    int pages = 0;
    int textPages = 0;
    std::vector<size_t> mediaBaseSizes;
    std::string provenance;
};

inline bool BuildSnapshotPrefixIdentity(
        const std::vector<std::string> &rawKeys,
        const std::vector<std::string> *baseKeys,
        const std::string &provenance,
        int pages,
        SnapshotPrefixIdentity &identity) {
    identity.pages = 0; // Failed construction cannot produce a match.
    identity.mediaBaseSizes.clear();
    if (pages <= 0) return false;
    identity.provenance = provenance;
    identity.textPages = pages;
    if (baseKeys == nullptr) {
        // Text uses no extra paged-KV identity. SSD's historical text hash
        // chain does not change this; token equality is checked by the caller.
        identity.pages = pages;
        return true;
    }
    if (baseKeys->size() < size_t(pages) || rawKeys.size() < size_t(pages)) return false;
    for (int page = 0; page < pages; ++page) {
        const auto &base = (*baseKeys)[page];
        const auto &raw = rawKeys[page];
        const size_t suffix = provenance.empty() ? 0 : 4 + provenance.size();
        if (raw.size() != base.size() + suffix ||
            raw.compare(0, base.size(), base) != 0) return false;
        // Validate the known producer format once at capture. The matcher
        // never parses or guesses a suffix in an arbitrary persisted key.
        if (!provenance.empty()) {
            uint32_t length = 0;
            for (int byte = 0; byte < 4; ++byte)
                length |= uint32_t(static_cast<unsigned char>(raw[base.size() + byte])) << (byte * 8);
            if (length != provenance.size() ||
                raw.compare(base.size() + 4, provenance.size(), provenance) != 0) return false;
        }
        if (base.empty() && identity.textPages == pages) continue;
        if (identity.textPages == pages) {
            identity.textPages = page;
            identity.mediaBaseSizes.reserve(pages - page);
        }
        identity.mediaBaseSizes.push_back(base.size());
    }
    identity.pages = pages;
    return true;
}

inline bool SnapshotPrefixIdentityMatches(
        const SnapshotPrefixIdentity &identity,
        const std::vector<std::string> &snapshotRawKeys,
        const std::vector<std::string> *queryBaseKeys,
        const std::string &queryProvenance) {
    if (identity.pages <= 0 || identity.provenance != queryProvenance) return false;
    if (queryBaseKeys == nullptr) return identity.textPages == identity.pages;
    if (queryBaseKeys->size() < size_t(identity.pages)) return false;
    for (int page = 0; page < identity.textPages; ++page)
        if (!(*queryBaseKeys)[page].empty()) return false;
    if (identity.textPages == identity.pages) return true;
    if (snapshotRawKeys.size() < size_t(identity.pages) ||
        identity.mediaBaseSizes.size() != size_t(identity.pages - identity.textPages)) return false;
    for (int page = identity.textPages; page < identity.pages; ++page) {
        const auto &query = (*queryBaseKeys)[page];
        const size_t baseSize = identity.mediaBaseSizes[page - identity.textPages];
        const auto &raw = snapshotRawKeys[page];
        if (query.size() != baseSize || raw.size() < baseSize ||
            raw.compare(0, baseSize, query) != 0) return false;
    }
    return true;
}

} // namespace fastllm
