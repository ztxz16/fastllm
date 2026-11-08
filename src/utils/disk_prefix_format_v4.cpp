#include "utils/disk_prefix_format_v4.h"

#include <openssl/evp.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>

namespace fastllm {
namespace prefix_v4 {
namespace {

// Fixed offsets inside the 128-byte record header. These are the frozen byte
// offsets the tests pin as golden bytes; changing any of them is a format break.
constexpr size_t kOffMagic = 0;
constexpr size_t kOffType = 8;
constexpr size_t kOffFlags = 12;
constexpr size_t kOffHeaderBytes = 16;
constexpr size_t kOffDigestAlgo = 20;
constexpr size_t kOffCoverage = 24;
constexpr size_t kOffReserved0 = 28;
constexpr size_t kOffPayloadBytes = 32;
constexpr size_t kOffPhysicalBytes = 40;
constexpr size_t kOffSequence = 48;
constexpr size_t kOffReserved1 = 56;
constexpr size_t kOffHeaderCrc = 60;
constexpr size_t kOffDigest = 64;
constexpr size_t kCrcCovered = 60;

void PutU32(char *destination, uint32_t value) {
    for (int i = 0; i < 4; ++i) destination[i] = char(value >> (i * 8));
}
void PutU64(char *destination, uint64_t value) {
    for (int i = 0; i < 8; ++i) destination[i] = char(value >> (i * 8));
}
uint32_t GetU32(const char *source) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= uint32_t(uint8_t(source[i])) << (i * 8);
    return value;
}
uint64_t GetU64(const char *source) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(uint8_t(source[i])) << (i * 8);
    return value;
}

bool Fail(std::string &error, const char *message) {
    error = message;
    return false;
}

bool ValidRecordType(uint32_t type) {
    return type >= kRecordObjectBatch && type <= kRecordMetadata;
}
} // namespace

const char kContainerMagic[8] = {'F', 'L', 'K', 'V', '4', 'P', '0', '1'};
const char kRecordMagic[8] = {'F', 'L', 'K', 'V', '4', 'R', '0', '1'};

uint32_t Crc32(const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    uint32_t crc = ~uint32_t(0);
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1)));
    }
    return ~crc;
}

bool IsDigest(const std::string &value) {
    if (value.size() != 64) return false;
    for (const char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

std::string DomainDigest(const char *domain, const void *data, size_t size) {
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int count = 0;
    const std::string prefix = std::string("fastllm-prefix-v4:") + domain + ":";
    const bool ok = context != nullptr &&
        EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(context, prefix.data(), prefix.size()) == 1 &&
        EVP_DigestUpdate(context, data, size) == 1 &&
        EVP_DigestFinal_ex(context, digest, &count) == 1 && count == 32;
    if (context) EVP_MD_CTX_free(context);
    if (!ok) return "";
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (unsigned int i = 0; i < count; ++i) {
        result += hex[digest[i] >> 4];
        result += hex[digest[i] & 15];
    }
    return result;
}

std::string DomainDigest(const char *domain, const std::string &value) {
    return DomainDigest(domain, value.data(), value.size());
}

// Route B：块大小可被 env 覆盖；下限 64（再小会让块对象/索引条数爆炸），
// 上限 1M。解析失败/非正数一律回落到默认 256。
uint32_t BlockTokens() {
    static const uint32_t value = [] {
        const char *text = std::getenv("FASTLLM_PREFIX_CACHE_BLOCK_TOKENS");
        const long long parsed = text != nullptr ? std::atoll(text) : 0;
        long long chosen = parsed > 0 ? parsed : kDefaultBlockTokens;
        if (chosen < 64) chosen = 64;
        if (chosen > (1ll << 20)) chosen = 1ll << 20;
        return uint32_t(chosen);
    }();
    return value;
}

std::string BlockGenesisKey() {
    return DomainDigest(kBlockChainDomain, "fastllm-block-genesis");
}

std::string BlockChainKey(const std::string &parentKey, uint64_t tokens,
                          const std::string &contentDigest) {
    std::string preimage = parentKey;
    preimage += '|';
    preimage += std::to_string(tokens);
    preimage += '|';
    preimage += contentDigest;
    return DomainDigest(kBlockChainDomain, preimage);
}

uint64_t AlignUp(uint64_t value, std::string &error) {
    if (value > std::numeric_limits<uint64_t>::max() - (kAlignment - 1)) {
        error = "v4_size_overflow";
        return 0;
    }
    return (value + kAlignment - 1) / kAlignment * kAlignment;
}

std::vector<ExportWindow> SplitExportWindows(uint64_t pageCount, uint64_t bytesPerPage,
                                             uint64_t budgetBytes, uint64_t maxWindowBytes) {
    std::vector<ExportWindow> windows;
    if (pageCount == 0) return windows;
    uint64_t pagesPerWindow = pageCount;
    if (bytesPerPage > 0) {
        // 两个上界都按"每窗口字节"理解：0 表示该上界不存在。
        uint64_t limit = budgetBytes;
        if (maxWindowBytes != 0) limit = (limit == 0) ? maxWindowBytes : std::min(limit, maxWindowBytes);
        if (limit == 0) {
            // 预算被明确压到 0（或未给出任何字节上界）：逐页推进，保证每窗口 ≥1 页、必有进展。
            pagesPerWindow = 1;
        } else {
            pagesPerWindow = limit / bytesPerPage;
            if (pagesPerWindow == 0) pagesPerWindow = 1;   // 预算不足一页 ⇒ 仍取 1 页
            if (pagesPerWindow > pageCount) pagesPerWindow = pageCount;
        }
    }
    for (uint64_t begin = 0; begin < pageCount; begin += pagesPerWindow) {
        ExportWindow window;
        window.beginPage = begin;
        window.endPage = std::min(begin + pagesPerWindow, pageCount);
        windows.push_back(window);
    }
    return windows;
}

std::vector<char> EncodeRecordHeader(const RecordHeader &header) {
    std::vector<char> bytes(size_t(kRecordHeaderBytes), 0);
    memcpy(bytes.data() + kOffMagic, kRecordMagic, 8);
    PutU32(bytes.data() + kOffType, header.type);
    PutU32(bytes.data() + kOffFlags, header.flags);
    PutU32(bytes.data() + kOffHeaderBytes, header.headerBytes);
    PutU32(bytes.data() + kOffDigestAlgo, header.digestAlgo);
    PutU32(bytes.data() + kOffCoverage, header.coverageMode);
    PutU32(bytes.data() + kOffReserved0, 0);
    PutU64(bytes.data() + kOffPayloadBytes, header.payloadBytes);
    PutU64(bytes.data() + kOffPhysicalBytes, header.physicalBytes);
    PutU64(bytes.data() + kOffSequence, header.sequence);
    PutU32(bytes.data() + kOffReserved1, 0);
    PutU32(bytes.data() + kOffHeaderCrc, Crc32(bytes.data(), kCrcCovered));
    memcpy(bytes.data() + kOffDigest, header.digest.data(),
           std::min<size_t>(header.digest.size(), 64));
    return bytes;
}

bool DecodeRecordHeader(const char *data, size_t size, RecordHeader &out, std::string &error) {
    if (data == nullptr || size < kRecordHeaderBytes) return Fail(error, "v4_record_header_short");
    if (memcmp(data + kOffMagic, kRecordMagic, 8) != 0) return Fail(error, "v4_record_magic");
    if (GetU32(data + kOffHeaderCrc) != Crc32(data, kCrcCovered)) {
        return Fail(error, "v4_record_header_crc");
    }
    RecordHeader header;
    header.type = GetU32(data + kOffType);
    header.flags = GetU32(data + kOffFlags);
    header.headerBytes = GetU32(data + kOffHeaderBytes);
    header.digestAlgo = GetU32(data + kOffDigestAlgo);
    header.coverageMode = GetU32(data + kOffCoverage);
    header.payloadBytes = GetU64(data + kOffPayloadBytes);
    header.physicalBytes = GetU64(data + kOffPhysicalBytes);
    header.sequence = GetU64(data + kOffSequence);
    header.digest.assign(data + kOffDigest, 64);
    if (!ValidRecordType(header.type)) return Fail(error, "v4_record_type");
    // Unknown critical flags must be rejected, never ignored: a reader that
    // ignores them could publish data whose meaning it does not know.
    if ((header.flags & ~kKnownRecordFlags) != 0) return Fail(error, "v4_record_unknown_flag");
    if (header.headerBytes != kRecordHeaderBytes) return Fail(error, "v4_record_header_size");
    if (header.digestAlgo != 1) return Fail(error, "v4_record_digest_algo");
    if (header.coverageMode != kCoverageObjectTableOnly &&
        header.coverageMode != kCoverageFullRecord) {
        return Fail(error, "v4_record_coverage");
    }
    if (header.payloadBytes == 0 || header.payloadBytes > kMaxPackedPayloadBytes) {
        return Fail(error, "v4_record_payload_bounds");
    }
    std::string alignError;
    const uint64_t expected = AlignUp(kRecordHeaderBytes + header.payloadBytes, alignError);
    if (!alignError.empty()) return Fail(error, "v4_size_overflow");
    if (header.physicalBytes != expected) return Fail(error, "v4_record_physical");
    if (!IsDigest(header.digest)) return Fail(error, "v4_record_digest");
    if (header.type == kRecordObjectBatch) {
        if (header.coverageMode != kCoverageObjectTableOnly) {
            return Fail(error, "v4_record_coverage");
        }
    } else if (header.coverageMode != kCoverageFullRecord) {
        // PREPARE/COMMIT/ERASE/METADATA are small records and are always hashed
        // in full; object batches are the only record with a table-only digest.
        return Fail(error, "v4_record_coverage");
    }
    out = header;
    return true;
}

std::vector<char> EncodeObjectTable(const std::vector<ObjectEntry> &entries,
                                    uint64_t payloadRegionBytes) {
    std::vector<char> table(16 + entries.size() * 88, 0);
    PutU32(table.data(), uint32_t(entries.size()));
    PutU32(table.data() + 4, 0);
    PutU64(table.data() + 8, payloadRegionBytes);
    for (size_t i = 0; i < entries.size(); ++i) {
        char *entry = table.data() + 16 + i * 88;
        PutU32(entry, entries[i].kind);
        PutU32(entry + 4, 0);
        PutU64(entry + 8, entries[i].bytes);
        PutU64(entry + 16, entries[i].offset);
        memcpy(entry + 24, entries[i].hash.data(),
               std::min<size_t>(entries[i].hash.size(), 64));
    }
    return table;
}

bool DecodeObjectTable(const char *data, size_t size, std::vector<ObjectEntry> &out,
                       uint64_t &payloadRegionBytes, std::string &error) {
    out.clear();
    payloadRegionBytes = 0;
    if (data == nullptr || size < 16) return Fail(error, "v4_object_table_short");
    const uint32_t count = GetU32(data);
    if (GetU32(data + 4) != 0) return Fail(error, "v4_object_table_reserved");
    payloadRegionBytes = GetU64(data + 8);
    // Validate the declared count against the bytes actually present BEFORE
    // allocating anything from it.
    if (size < 16ull + uint64_t(count) * 88ull) return Fail(error, "v4_object_table_count");
    if (payloadRegionBytes > kMaxPackedPayloadBytes) {
        return Fail(error, "v4_object_table_region_bounds");
    }
    std::vector<ObjectEntry> entries;
    entries.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const char *entry = data + 16 + size_t(i) * 88;
        ObjectEntry value;
        value.kind = GetU32(entry);
        if (GetU32(entry + 4) != 0) return Fail(error, "v4_object_entry_reserved");
        value.bytes = GetU64(entry + 8);
        value.offset = GetU64(entry + 16);
        value.hash.assign(entry + 24, 64);
        if (value.kind < kObjectData || value.kind > kObjectBlock) {
            return Fail(error, "v4_object_kind");
        }
        if (value.bytes == 0 || value.bytes > kMaxObjectBytes) {
            return Fail(error, "v4_object_bytes");
        }
        if (value.offset % 8 != 0) return Fail(error, "v4_object_alignment");
        if (value.offset > payloadRegionBytes || value.bytes > payloadRegionBytes - value.offset) {
            return Fail(error, "v4_object_bounds");
        }
        if (!IsDigest(value.hash)) return Fail(error, "v4_object_digest");
        entries.push_back(std::move(value));
    }
    // Reject overlapping objects and duplicate hashes: either would make a
    // later "verify by hash" step ambiguous about which bytes are the object.
    std::vector<size_t> order(entries.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&entries](size_t a, size_t b) {
        if (entries[a].offset != entries[b].offset) return entries[a].offset < entries[b].offset;
        return entries[a].bytes < entries[b].bytes;
    });
    for (size_t i = 1; i < order.size(); ++i) {
        const auto &previous = entries[order[i - 1]];
        const auto &current = entries[order[i]];
        if (current.offset < previous.offset + previous.bytes) {
            return Fail(error, "v4_object_overlap");
        }
    }
    std::set<std::string> hashes;
    for (const auto &entry : entries) {
        if (!hashes.insert(entry.hash).second) return Fail(error, "v4_object_duplicate");
    }
    out = std::move(entries);
    return true;
}

std::vector<char> EncodeManifestNode(const ManifestNode &node) {
    const size_t entryBytes = node.leaf ? 96 : 88;
    const size_t count = node.leaf ? node.bundles.size() : node.children.size();
    std::vector<char> bytes(24 + count * entryBytes, 0);
    bytes[0] = node.leaf ? 1 : 2;
    PutU32(bytes.data() + 4, uint32_t(count));
    PutU64(bytes.data() + 8, node.tokenStart);
    PutU64(bytes.data() + 16, node.tokenEnd);
    for (size_t i = 0; i < count; ++i) {
        char *entry = bytes.data() + 24 + i * entryBytes;
        if (node.leaf) {
            PutU32(entry, node.bundles[i].rank);
            PutU32(entry + 4, node.bundles[i].layoutId);
            PutU64(entry + 8, node.bundles[i].tokenStart);
            PutU64(entry + 16, node.bundles[i].tokenEnd);
            PutU64(entry + 24, node.bundles[i].bytes);
            memcpy(entry + 32, node.bundles[i].hash.data(),
                   std::min<size_t>(node.bundles[i].hash.size(), 64));
        } else {
            PutU64(entry, node.children[i].tokenStart);
            PutU64(entry + 8, node.children[i].tokenEnd);
            PutU64(entry + 16, node.children[i].nodeBytes);
            memcpy(entry + 24, node.children[i].hash.data(),
                   std::min<size_t>(node.children[i].hash.size(), 64));
        }
    }
    return bytes;
}

bool DecodeManifestNode(const char *data, size_t size, uint64_t depth,
                        bool hasParentRange, uint64_t parentStart, uint64_t parentEnd,
                        ManifestNode &out, std::string &error) {
    if (data == nullptr || size < 24) return Fail(error, "v4_node_short");
    if (depth > kMaxManifestDepth) return Fail(error, "v4_node_depth");
    const uint8_t type = uint8_t(data[0]);
    if (type != 1 && type != 2) return Fail(error, "v4_node_type");
    if (data[1] != 0 || data[2] != 0 || data[3] != 0) return Fail(error, "v4_node_reserved");
    const uint32_t count = GetU32(data + 4);
    if (count == 0 || count > kMaxManifestFanout) return Fail(error, "v4_node_count");
    const uint64_t tokenStart = GetU64(data + 8);
    const uint64_t tokenEnd = GetU64(data + 16);
    if (tokenStart >= tokenEnd) return Fail(error, "v4_node_range");
    if (hasParentRange && (tokenStart < parentStart || tokenEnd > parentEnd)) {
        return Fail(error, "v4_node_range_escape");
    }
    const size_t entryBytes = type == 1 ? 96 : 88;
    if (size != 24ull + uint64_t(count) * entryBytes) return Fail(error, "v4_node_size");
    ManifestNode node;
    node.leaf = type == 1;
    node.tokenStart = tokenStart;
    node.tokenEnd = tokenEnd;
    uint64_t previousEnd = tokenStart;
    for (uint32_t i = 0; i < count; ++i) {
        const char *entry = data + 24 + size_t(i) * entryBytes;
        if (node.leaf) {
            BundleRef bundle;
            bundle.rank = GetU32(entry);
            bundle.layoutId = GetU32(entry + 4);
            bundle.tokenStart = GetU64(entry + 8);
            bundle.tokenEnd = GetU64(entry + 16);
            bundle.bytes = GetU64(entry + 24);
            bundle.hash.assign(entry + 32, 64);
            if (bundle.tokenStart >= bundle.tokenEnd || bundle.tokenStart < tokenStart ||
                bundle.tokenEnd > tokenEnd) {
                return Fail(error, "v4_bundle_range");
            }
            if (bundle.bytes == 0 || bundle.bytes > kMaxObjectBytes) {
                return Fail(error, "v4_bundle_bytes");
            }
            if (!IsDigest(bundle.hash)) return Fail(error, "v4_bundle_digest");
            // Bundles must be contiguous and non-overlapping: a gap or an
            // overlap here would silently change which bytes a token maps to.
            if (bundle.tokenStart != previousEnd) return Fail(error, "v4_bundle_sequence");
            previousEnd = bundle.tokenEnd;
            node.bundles.push_back(std::move(bundle));
        } else {
            ChildRef child;
            child.tokenStart = GetU64(entry);
            child.tokenEnd = GetU64(entry + 8);
            child.nodeBytes = GetU64(entry + 16);
            child.hash.assign(entry + 24, 64);
            if (child.tokenStart >= child.tokenEnd || child.tokenStart < tokenStart ||
                child.tokenEnd > tokenEnd) {
                return Fail(error, "v4_child_range");
            }
            if (child.nodeBytes == 0 || child.nodeBytes > kMaxNodeBytes) {
                return Fail(error, "v4_child_bytes");
            }
            if (!IsDigest(child.hash)) return Fail(error, "v4_child_digest");
            if (child.tokenStart != previousEnd) return Fail(error, "v4_child_sequence");
            previousEnd = child.tokenEnd;
            node.children.push_back(std::move(child));
        }
    }
    // Children/bundles must cover the node range without a gap, otherwise a
    // reader would silently miss tokens.
    if (previousEnd != tokenEnd) return Fail(error, "v4_node_gap");
    out = std::move(node);
    return true;
}

namespace {

uint64_t AlignTo8(uint64_t value) { return (value + 7) / 8 * 8; }

void PutBytes(std::vector<char> &out, const std::string &value) {
    out.insert(out.end(), value.begin(), value.end());
}
void PutU32To(std::vector<char> &out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.push_back(char(value >> (i * 8)));
}
void PutU64To(std::vector<char> &out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.push_back(char(value >> (i * 8)));
}

// OBJECT_BATCH digests bind only the object table; every other record is small
// and is digested in full. Append and scan must agree exactly.
namespace {
std::atomic<uint64_t> g_recordHashBytes{0}, g_recordHashNs{0};
uint64_t RecordHashNowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
} // namespace

std::string RecordDigestForImpl(uint32_t type, const char *payload, size_t size);
std::string RecordDigestFor(uint32_t type, const char *payload, size_t size) {
    const uint64_t hashStart = RecordHashNowNs();
    std::string result = RecordDigestForImpl(type, payload, size);
    g_recordHashBytes.fetch_add(size, std::memory_order_relaxed);
    g_recordHashNs.fetch_add(RecordHashNowNs() - hashStart, std::memory_order_relaxed);
    return result;
}

std::string RecordDigestForImpl(uint32_t type, const char *payload, size_t size) {
    if (type == kRecordObjectBatch) {
        if (size < 16) return "";
        const uint32_t count = GetU32(payload);
        const uint64_t tableBytes = 16ull + uint64_t(count) * 88ull;
        if (tableBytes > size) return "";
        return DomainDigest("batch", payload, size_t(tableBytes));
    }
    return DomainDigest(RecordDomain(type), payload, size);
}

bool DecodeKeyValue(const char *data, size_t size, size_t &cursor, std::string &out,
                    uint32_t maximum, std::string &error) {
    if (cursor + 4 > size) return Fail(error, "v4_record_short");
    const uint32_t length = GetU32(data + cursor);
    cursor += 4;
    if (length == 0 || length > maximum) return Fail(error, "v4_record_length");
    if (cursor + length > size) return Fail(error, "v4_record_short");
    out.assign(data + cursor, length);
    cursor += length;
    return true;
}

} // namespace

const char *RecordDomain(uint32_t type) {
    switch (type) {
        case kRecordObjectBatch: return "batch";
        case kRecordManifestPrepare: return "prepare";
        case kRecordManifestCommit: return "commit";
        case kRecordErase: return "erase";
        case kRecordMetadata: return "metadata";
        default: return "unknown";
    }
}

std::string RecordDigest(uint32_t type, const std::string &payload) {
    return RecordDigestFor(type, payload.data(), payload.size());
}

std::vector<char> EncodeObjectList(const std::vector<ObjectListEntry> &entries) {
    std::vector<char> out(8 + entries.size() * 80, 0);
    PutU32(out.data(), uint32_t(entries.size()));
    PutU32(out.data() + 4, 0);
    for (size_t i = 0; i < entries.size(); ++i) {
        char *entry = out.data() + 8 + i * 80;
        PutU32(entry, entries[i].kind);
        PutU32(entry + 4, 0);
        PutU64(entry + 8, entries[i].bytes);
        memcpy(entry + 16, entries[i].hash.data(),
               std::min<size_t>(entries[i].hash.size(), 64));
    }
    return out;
}

bool DecodeObjectList(const char *data, size_t size, std::vector<ObjectListEntry> &out,
                      std::string &error) {
    out.clear();
    if (data == nullptr || size < 8) return Fail(error, "v4_object_list_short");
    const uint32_t count = GetU32(data);
    if (GetU32(data + 4) != 0) return Fail(error, "v4_object_list_reserved");
    // Validate the count against the bytes present before allocating.
    if (count > kMaxObjectListEntries) return Fail(error, "v4_object_list_count");
    if (size != 8ull + uint64_t(count) * 80ull) return Fail(error, "v4_object_list_size");
    std::set<std::string> hashes;
    for (uint32_t i = 0; i < count; ++i) {
        const char *entry = data + 8 + size_t(i) * 80;
        ObjectListEntry value;
        value.kind = GetU32(entry);
        if (GetU32(entry + 4) != 0) return Fail(error, "v4_object_list_entry_reserved");
        value.bytes = GetU64(entry + 8);
        value.hash.assign(entry + 16, 64);
        if (value.kind < kObjectData || value.kind > kObjectStateNode) {
            return Fail(error, "v4_object_list_kind");
        }
        if (value.bytes == 0 || value.bytes > kMaxObjectBytes) {
            return Fail(error, "v4_object_list_bytes");
        }
        if (!IsDigest(value.hash)) return Fail(error, "v4_object_list_digest");
        if (!hashes.insert(value.hash).second) return Fail(error, "v4_object_list_duplicate");
        out.push_back(std::move(value));
    }
    return true;
}

std::vector<char> EncodeObjectBatchPayload(const std::vector<ObjectEntry> &entries,
                                           const std::string &payload) {
    const std::vector<char> table = EncodeObjectTable(entries, payload.size());
    std::vector<char> out = table;
    out.resize(size_t(AlignTo8(table.size())), 0);
    PutBytes(out, payload);
    return out;
}

bool DecodeObjectBatchPayloadView(const char *data, size_t size,
                                  std::vector<ObjectEntry> &entries,
                                  uint64_t &regionStart, uint64_t &regionBytes,
                                  std::string &error) {
    entries.clear();
    regionStart = 0;
    regionBytes = 0;
    uint64_t region = 0;
    if (!DecodeObjectTable(data, size, entries, region, error)) return false;
    const uint64_t tableBytes = 16ull + uint64_t(entries.size()) * 88ull;
    regionStart = AlignTo8(tableBytes);
    if (regionStart > size || region > size - regionStart) {
        return Fail(error, "v4_batch_payload_bounds");
    }
    // The declared region must be exactly the bytes present: a trailing gap
    // would let a later reader attribute unknown bytes to an object.
    if (regionStart + region != size) return Fail(error, "v4_batch_payload_size");
    regionBytes = region;
    return true;
}

bool DecodeObjectBatchPayload(const char *data, size_t size,
                              std::vector<ObjectEntry> &entries, std::string &payload,
                              std::string &error) {
    entries.clear();
    payload.clear();
    uint64_t regionStart = 0, region = 0;
    // 与 View 变体共用同一套校验（错误串不变）；本函数只是额外把区域拷出来。
    if (!DecodeObjectBatchPayloadView(data, size, entries, regionStart, region, error)) return false;
    payload.assign(data + regionStart, size_t(region));
    return true;
}

std::vector<char> EncodePrepare(const PrepareRecord &record) {
    std::vector<char> out;
    PutU64To(out, record.sequence);
    PutU32To(out, uint32_t(record.key.size()));
    PutBytes(out, record.key);
    PutBytes(out, record.identity);
    PutBytes(out, record.rootHash);
    PutU64To(out, record.firstBatch);
    PutU64To(out, record.lastBatch);
    return out;
}

bool DecodePrepare(const char *data, size_t size, PrepareRecord &out, std::string &error) {
    if (data == nullptr || size < 8 + 4 + 64 + 64 + 8 + 8) return Fail(error, "v4_record_short");
    size_t cursor = 0;
    PrepareRecord record;
    record.sequence = GetU64(data + cursor);
    cursor += 8;
    if (!DecodeKeyValue(data, size, cursor, record.key, kMaxKeyBytes, error)) return false;
    if (cursor + 128 > size) return Fail(error, "v4_record_short");
    record.identity.assign(data + cursor, 64);
    record.rootHash.assign(data + cursor + 64, 64);
    cursor += 128;
    if (!IsDigest(record.identity) || !IsDigest(record.rootHash)) {
        return Fail(error, "v4_record_digest");
    }
    record.firstBatch = GetU64(data + cursor);
    record.lastBatch = GetU64(data + cursor + 8);
    cursor += 16;
    if (cursor != size) return Fail(error, "v4_record_size");
    out = std::move(record);
    return true;
}

std::vector<char> EncodeCommit(const CommitRecord &record) {
    std::vector<char> out;
    PutU64To(out, record.sequence);
    PutU64To(out, record.prepareOffset);
    PutU64To(out, record.prepareBytes);
    PutBytes(out, record.prepareDigest);
    PutBytes(out, record.rootHash);
    return out;
}

bool DecodeCommit(const char *data, size_t size, CommitRecord &out, std::string &error) {
    if (data == nullptr || size != 8 + 8 + 8 + 64 + 64) return Fail(error, "v4_record_size");
    CommitRecord record;
    record.sequence = GetU64(data);
    record.prepareOffset = GetU64(data + 8);
    record.prepareBytes = GetU64(data + 16);
    record.prepareDigest.assign(data + 24, 64);
    record.rootHash.assign(data + 88, 64);
    if (record.prepareBytes == 0 || record.prepareBytes > kMaxPackedPayloadBytes) {
        return Fail(error, "v4_record_bounds");
    }
    if (!IsDigest(record.prepareDigest) || !IsDigest(record.rootHash)) {
        return Fail(error, "v4_record_digest");
    }
    out = std::move(record);
    return true;
}

std::vector<char> EncodeErase(const EraseRecord &record) {
    std::vector<char> out;
    PutU64To(out, record.sequence);
    PutU32To(out, uint32_t(record.key.size()));
    PutBytes(out, record.key);
    PutBytes(out, record.identity);
    return out;
}

bool DecodeErase(const char *data, size_t size, EraseRecord &out, std::string &error) {
    if (data == nullptr || size < 8 + 4 + 64) return Fail(error, "v4_record_short");
    size_t cursor = 0;
    EraseRecord record;
    record.sequence = GetU64(data + cursor);
    cursor += 8;
    if (!DecodeKeyValue(data, size, cursor, record.key, kMaxKeyBytes, error)) return false;
    if (cursor + 64 != size) return Fail(error, "v4_record_size");
    record.identity.assign(data + cursor, 64);
    if (!IsDigest(record.identity)) return Fail(error, "v4_record_digest");
    out = std::move(record);
    return true;
}

std::vector<char> EncodeMetadata(const MetadataRecord &record) {
    std::vector<char> out;
    PutU64To(out, record.sequence);
    PutU32To(out, uint32_t(record.key.size()));
    PutBytes(out, record.key);
    PutU32To(out, uint32_t(record.value.size()));
    PutBytes(out, record.value);
    return out;
}

bool DecodeMetadata(const char *data, size_t size, MetadataRecord &out, std::string &error) {
    if (data == nullptr || size < 8 + 4 + 4) return Fail(error, "v4_record_short");
    size_t cursor = 0;
    MetadataRecord record;
    record.sequence = GetU64(data + cursor);
    cursor += 8;
    if (!DecodeKeyValue(data, size, cursor, record.key, kMaxKeyBytes, error)) return false;
    if (!DecodeKeyValue(data, size, cursor, record.value, kMaxMetadataValueBytes, error)) {
        return false;
    }
    if (cursor != size) return Fail(error, "v4_record_size");
    out = std::move(record);
    return true;
}

LogClassification ScanControlLog(const char *data, size_t size,
                                 std::vector<ScannedRecord> &out, std::string &error) {
    out.clear();
    error.clear();
    uint64_t previous = 0;
    bool hasPrevious = false;
    size_t cursor = 0;
    while (cursor < size) {
        const size_t remaining = size - cursor;
        if (remaining < kRecordHeaderBytes) return LogClassification::UncommittedTail;
        RecordHeader header;
        std::string decodeError;
        if (!DecodeRecordHeader(data + cursor, remaining, header, decodeError)) {
            if (decodeError == "v4_record_header_short") return LogClassification::UncommittedTail;
            if (decodeError == "v4_record_magic") {
                // A zero-filled tail is a preallocated/short-written region, not
                // corruption; anything else is ambiguous and fails closed.
                bool allZero = true;
                for (size_t i = cursor; i < size && allZero; ++i) allZero = data[i] == 0;
                if (allZero) return LogClassification::UncommittedTail;
            }
            error = decodeError;
            return LogClassification::CorruptControl;
        }
        if (header.physicalBytes > remaining) return LogClassification::UncommittedTail;
        const std::string digest =
            RecordDigestFor(header.type, data + cursor + kRecordHeaderBytes, size_t(header.payloadBytes));
        if (digest.empty() || digest != header.digest) {
            error = "v4_control_record_digest";
            return LogClassification::CorruptControl;
        }
        if (hasPrevious && header.sequence <= previous) {
            error = "v4_control_sequence";
            return LogClassification::CorruptControl;
        }
        previous = header.sequence;
        hasPrevious = true;
        ScannedRecord record;
        record.header = header;
        record.offset = cursor;
        out.push_back(std::move(record));
        cursor += size_t(header.physicalBytes);
    }
    return LogClassification::Complete;
}


ControlLogReader::~ControlLogReader() {
    if (fd_ >= 0) close(fd_);
}

bool ControlLogReader::Open(const std::string &path, std::string &error) {
    error.clear();
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    size_ = 0;
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        // A missing control log is an empty log, not an error.
        if (errno == ENOENT) return true;
        return Fail(error, "v4_control_read");
    }
    struct stat info {};
    if (fstat(fd_, &info) != 0) {
        close(fd_);
        fd_ = -1;
        return Fail(error, "v4_control_stat");
    }
    size_ = uint64_t(info.st_size);
    return true;
}

bool ControlLogReader::ReadAt(uint64_t offset, void *out, size_t bytes,
                              std::string &error) const {
    if (bytes == 0) return true;
    char *destination = static_cast<char *>(out);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t got = pread(fd_, destination + done, bytes - done, off_t(offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            return Fail(error, "v4_control_read");
        }
        if (got == 0) return Fail(error, "v4_control_read_short");
        done += size_t(got);
    }
    return true;
}

bool ControlLogReader::ReadPayload(const ScannedRecord &record, std::string &out,
                                   std::string &error) const {
    out.assign(size_t(record.header.payloadBytes), '\0');
    return ReadAt(record.offset + kRecordHeaderBytes, out.data(), out.size(), error);
}

namespace {
std::string HexDigestString(const unsigned char *digest, unsigned int count) {
    static const char hex[] = "0123456789abcdef";
    std::string out(size_t(count) * 2, '0');
    for (unsigned int i = 0; i < count; ++i) {
        out[2 * i] = hex[(digest[i] >> 4) & 0xF];
        out[2 * i + 1] = hex[digest[i] & 0xF];
    }
    return out;
}

// Byte-for-byte equivalent of RecordDigestFor/domain digest, computed over a
// record payload read in bounded chunks instead of a resident whole-log buffer.
std::string StreamRecordDigest(const ControlLogReader &reader, const RecordHeader &header,
                               uint64_t payloadOffset, std::string &error) {
    const char *domain = header.type == kRecordObjectBatch ? "batch" : RecordDomain(header.type);
    if (domain == nullptr) return "";
    uint64_t covered = header.payloadBytes;
    if (header.type == kRecordObjectBatch) {
        char table[16];
        if (!reader.ReadAt(payloadOffset, table, sizeof(table), error)) return "";
        const uint32_t count = GetU32(table);
        const uint64_t tableBytes = 16ull + uint64_t(count) * 88ull;
        if (tableBytes > header.payloadBytes) return "";
        covered = tableBytes;
    }
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    if (context == nullptr) return "";
    const std::string prefix = std::string("fastllm-prefix-v4:") + domain + ":";
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digestBytes = 0;
    bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(context, prefix.data(), prefix.size()) == 1;
    std::vector<char> chunk(1u << 20);
    uint64_t done = 0;
    while (ok && done < covered) {
        const size_t want = size_t(std::min<uint64_t>(chunk.size(), covered - done));
        ok = reader.ReadAt(payloadOffset + done, chunk.data(), want, error);
        if (!ok) break;
        ok = EVP_DigestUpdate(context, chunk.data(), want) == 1;
        done += want;
    }
    ok = ok && EVP_DigestFinal_ex(context, digest, &digestBytes) == 1 && digestBytes == 32;
    EVP_MD_CTX_free(context);
    if (!ok) return "";
    return HexDigestString(digest, digestBytes);
}
} // namespace

LogClassification ControlLogReader::Scan(std::vector<ScannedRecord> &out, std::string &error) {
    out.clear();
    error.clear();
    uint64_t previous = 0;
    bool hasPrevious = false;
    uint64_t cursor = 0;
    std::vector<char> headerBytes(kRecordHeaderBytes);
    std::vector<char> tailChunk(1u << 20);
    while (cursor < size_) {
        const uint64_t remaining = size_ - cursor;
        if (remaining < kRecordHeaderBytes) return LogClassification::UncommittedTail;
        std::string readError;
        if (!ReadAt(cursor, headerBytes.data(), kRecordHeaderBytes, readError)) {
            return LogClassification::UncommittedTail;
        }
        RecordHeader header;
        std::string decodeError;
        if (!DecodeRecordHeader(headerBytes.data(), kRecordHeaderBytes, header, decodeError)) {
            if (decodeError == "v4_record_header_short") return LogClassification::UncommittedTail;
            if (decodeError == "v4_record_magic") {
                // A zero-filled tail is a preallocated/short-written region, not
                // corruption; anything else is ambiguous and fails closed.
                bool allZero = true;
                uint64_t scan = cursor;
                while (allZero && scan < size_) {
                    const size_t want = size_t(std::min<uint64_t>(tailChunk.size(), size_ - scan));
                    if (!ReadAt(scan, tailChunk.data(), want, readError)) {
                        return LogClassification::UncommittedTail;
                    }
                    for (size_t i = 0; i < want && allZero; ++i) allZero = tailChunk[i] == 0;
                    scan += want;
                }
                if (allZero) return LogClassification::UncommittedTail;
            }
            error = decodeError;
            return LogClassification::CorruptControl;
        }
        if (header.physicalBytes > remaining) return LogClassification::UncommittedTail;
        const std::string digest =
            StreamRecordDigest(*this, header, cursor + kRecordHeaderBytes, readError);
        if (digest.empty() || digest != header.digest) {
            error = "v4_control_record_digest";
            return LogClassification::CorruptControl;
        }
        if (hasPrevious && header.sequence <= previous) {
            error = "v4_control_sequence";
            return LogClassification::CorruptControl;
        }
        previous = header.sequence;
        hasPrevious = true;
        ScannedRecord record;
        record.header = header;
        record.offset = cursor;
        out.push_back(std::move(record));
        cursor += header.physicalBytes;
    }
    return LogClassification::Complete;
}

namespace {
std::atomic<uint64_t> g_journalCalls{0}, g_journalBytes{0}, g_journalWriteNs{0}, g_journalSyncNs{0};
std::atomic<uint64_t> g_journalDirSyncs{0}, g_journalDirSyncNs{0};
std::atomic<uint64_t> g_journalDataSyncs{0}, g_journalDataSyncNs{0};
std::atomic<uint64_t> g_journalControlSyncs{0}, g_journalControlSyncNs{0};
uint64_t JournalNowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
} // namespace

JournalStats JournalStatsSnapshot() {
    JournalStats stats;
    stats.calls = g_journalCalls.load();
    stats.bytes = g_journalBytes.load();
    stats.writeNs = g_journalWriteNs.load();
    stats.syncNs = g_journalSyncNs.load();
    stats.dirSyncs = g_journalDirSyncs.load();
    stats.dirSyncNs = g_journalDirSyncNs.load();
    stats.dataSyncs = g_journalDataSyncs.load();
    stats.dataSyncNs = g_journalDataSyncNs.load();
    stats.controlSyncs = g_journalControlSyncs.load();
    stats.controlSyncNs = g_journalControlSyncNs.load();
    stats.recordHashBytes = g_recordHashBytes.load();
    stats.recordHashNs = g_recordHashNs.load();
    return stats;
}

namespace {
// §10.3 第 4 条：新文件创建后还必须同步其父目录，否则目录项本身可能未落盘。
bool SyncParentDirectory(const std::string &path) {
    const std::string::size_type slash = path.find_last_of('/');
    const std::string parent = slash == std::string::npos ? std::string(".")
                              : (slash == 0 ? std::string("/") : path.substr(0, slash));
    const int fd = open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return false;
    const int result = fsync(fd);
    close(fd);
    return result == 0;
}
} // namespace

bool AppendRecordToFile(const std::string &path, uint32_t type, uint64_t sequence,
                        const std::string &payload, uint64_t *outOffset, std::string &error) {
    if (payload.empty() || payload.size() > kMaxPackedPayloadBytes) {
        return Fail(error, "v4_record_payload_bounds");
    }
#ifdef FASTLLM_DISK_PREFIX_CACHE_TESTING
    // CPU-only regression hook: descriptor append failure must not publish a
    // checkpoint; this hook has no effect in product builds.
    const char *failType = std::getenv("FASTLLM_TEST_FAIL_RECORD_TYPE");
    if (failType != nullptr && std::strcmp(failType, "metadata") == 0 &&
        type == kRecordMetadata) {
        return Fail(error, "v4_control_test_injected");
    }
#endif
    RecordHeader header;
    header.type = type;
    header.coverageMode = type == kRecordObjectBatch ? kCoverageObjectTableOnly
                                                     : kCoverageFullRecord;
    header.sequence = sequence;
    header.payloadBytes = payload.size();
    header.physicalBytes = AlignUp(kRecordHeaderBytes + payload.size(), error);
    if (!error.empty()) return false;
    header.digest = RecordDigestFor(type, payload.data(), payload.size());
    if (header.digest.empty()) return Fail(error, "v4_record_digest");
    const std::vector<char> headerBytes = EncodeRecordHeader(header);
    std::vector<char> physical(size_t(header.physicalBytes), 0);
    memcpy(physical.data(), headerBytes.data(), headerBytes.size());
    memcpy(physical.data() + kRecordHeaderBytes, payload.data(), payload.size());

    struct stat pre {};
    const bool existed = stat(path.c_str(), &pre) == 0;
    const int fd = open(path.c_str(), O_CREAT | O_WRONLY | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) return Fail(error, "v4_control_open");
    struct stat info {};
    if (fstat(fd, &info) != 0) { close(fd); return Fail(error, "v4_control_stat"); }
    // §17.1（2026-09-30 修复①，故障注入实测）：**写入一旦开始，任何失败路径都必须把文件截回
    // 写前长度**（info.st_size 是本次 append 之前的文件长度）。原因：fdatasync / 父目录 fsync
    // 失败时字节其实已经完整留在 control.log 里并占用了序号，而调用方在失败时会把
    // impl->sequence 回退 → 下一笔提交复用同一序号 → ScanControlLog 的严格递增判定
    // （header.sequence <= previous）把**整份日志**判成 CorruptControl → 此后 RebuildIndex
    // 永久失败（对象字节其实完好）。保证"报告失败 ⇒ 文件里没有多出任何字节"后，调用方的
    // 序号回退才是自洽的；这条同时覆盖"短写/半截记录（torn record）"（现成注入器造不出）。
    auto rollbackUnwrittenBytes = [&]() {
        // 尽力而为：截断只释放空间，通常不会因 ENOSPC 失败；失败则报更明确的错误码。
        if (ftruncate(fd, off_t(info.st_size)) != 0) {
            error = "v4_control_rollback";
            return false;
        }
        (void)fdatasync(fd);
        return true;
    };
    const uint64_t writeStart = JournalNowNs();
    size_t written = 0;
    while (written < physical.size()) {
        const ssize_t count = write(fd, physical.data() + written, physical.size() - written);
        if (count < 0) {
            if (errno == EINTR) continue;
            const bool rolledBack = rollbackUnwrittenBytes();
            close(fd);
            if (!rolledBack) return false;
            return Fail(error, "v4_control_write");
        }
        written += size_t(count);
    }
    const uint64_t writeEnd = JournalNowNs();
    if (fdatasync(fd) != 0) {
        const bool rolledBack = rollbackUnwrittenBytes();
        close(fd);
        if (!rolledBack) return false;
        return Fail(error, "v4_control_sync");
    }
    g_journalCalls.fetch_add(1, std::memory_order_relaxed);
    g_journalBytes.fetch_add(physical.size(), std::memory_order_relaxed);
    g_journalWriteNs.fetch_add(writeEnd - writeStart, std::memory_order_relaxed);
    const uint64_t syncNs = JournalNowNs() - writeEnd;
    g_journalSyncNs.fetch_add(syncNs, std::memory_order_relaxed);
    if (type == kRecordObjectBatch) {
        g_journalDataSyncs.fetch_add(1, std::memory_order_relaxed);
        g_journalDataSyncNs.fetch_add(syncNs, std::memory_order_relaxed);
    } else {
        g_journalControlSyncs.fetch_add(1, std::memory_order_relaxed);
        g_journalControlSyncNs.fetch_add(syncNs, std::memory_order_relaxed);
    }
    // 首次创建控制日志时同步父目录，保证目录项也持久。
    if (!existed) {
        const uint64_t dirStart = JournalNowNs();
        const bool synced = SyncParentDirectory(path);
        g_journalDirSyncs.fetch_add(1, std::memory_order_relaxed);
        g_journalDirSyncNs.fetch_add(JournalNowNs() - dirStart, std::memory_order_relaxed);
        if (!synced) {
            const bool rolledBack = rollbackUnwrittenBytes();   // 首次建日志：截回 0，调用方回退到 sequence 0 才自洽
            close(fd);
            if (!rolledBack) return false;
            return Fail(error, "v4_control_dir_sync");
        }
    }
    if (outOffset != nullptr) *outOffset = uint64_t(info.st_size);
    close(fd);
    return true;
}

bool ReadWholeFile(const std::string &path, std::string &out, std::string &error) {
    out.clear();
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        // A missing control log is an empty log, not an error: a fresh root has
        // nothing to recover.
        struct stat info {};
        if (stat(path.c_str(), &info) != 0 && errno == ENOENT) return true;
        return Fail(error, "v4_control_read");
    }
    out.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return true;
}

// ---------------------------------------------------------------------------
// Incremental manifest tree (plan §9).

namespace {

struct TreeBuildNode {
    std::string hash, bytes;
    uint64_t start = 0, end = 0;
    bool leaf = false;
};

void RecordNode(const TreeBuildNode &node, std::vector<TreeNode> &sink) {
    TreeNode out;
    out.hash = node.hash;
    out.bytes = node.bytes;
    out.tokenStart = node.start;
    out.tokenEnd = node.end;
    out.leaf = node.leaf;
    sink.push_back(std::move(out));
}

TreeBuildNode MakeNode(const std::vector<char> &bytes, bool leaf, uint64_t start, uint64_t end) {
    TreeBuildNode node;
    node.bytes.assign(bytes.data(), bytes.size());
    node.hash = DomainDigest("node", node.bytes);
    node.start = start;
    node.end = end;
    node.leaf = leaf;
    return node;
}

TreeBuildNode MakeLeaf(const std::vector<BundleRef> &bundles, std::vector<TreeNode> &sink) {
    ManifestNode node;
    node.leaf = true;
    node.tokenStart = bundles.front().tokenStart;
    node.tokenEnd = bundles.back().tokenEnd;
    node.bundles = bundles;
    const std::vector<char> bytes = EncodeManifestNode(node);
    TreeBuildNode built = MakeNode(bytes, true, node.tokenStart, node.tokenEnd);
    RecordNode(built, sink);
    return built;
}

// Groups a contiguous level into parents of at most kMaxManifestFanout children.
std::vector<TreeBuildNode> BuildLevel(const std::vector<TreeBuildNode> &children,
                                      std::vector<TreeNode> &sink) {
    std::vector<TreeBuildNode> parents;
    for (size_t i = 0; i < children.size(); i += kMaxManifestFanout) {
        const size_t count = std::min<size_t>(kMaxManifestFanout, children.size() - i);
        ManifestNode node;
        node.leaf = false;
        node.tokenStart = children[i].start;
        node.tokenEnd = children[i + count - 1].end;
        for (size_t j = 0; j < count; ++j) {
            node.children.push_back({children[i + j].start, children[i + j].end,
                                     children[i + j].bytes.size(), children[i + j].hash});
        }
        const std::vector<char> bytes = EncodeManifestNode(node);
        TreeBuildNode built = MakeNode(bytes, false, node.tokenStart, node.tokenEnd);
        RecordNode(built, sink);
        parents.push_back(std::move(built));
    }
    return parents;
}

// Splits an ordered bundle run into leaf nodes of at most kMaxManifestFanout.
std::vector<TreeBuildNode> BuildLeaves(const std::vector<BundleRef> &bundles,
                                       std::vector<TreeNode> &sink) {
    std::vector<TreeBuildNode> leaves;
    for (size_t i = 0; i < bundles.size(); i += kMaxManifestFanout) {
        const size_t count = std::min<size_t>(kMaxManifestFanout, bundles.size() - i);
        std::vector<BundleRef> group(bundles.begin() + i, bundles.begin() + i + count);
        leaves.push_back(MakeLeaf(group, sink));
    }
    return leaves;
}

// Fetches an existing node. Re-fetched siblings are never recorded in the sink,
// which is what keeps untouched subtrees shared instead of rewritten.
bool FetchNode(const NodeResolver &resolver, const std::string &hash, TreeBuildNode &out,
               std::string &error) {
    std::string bytes;
    if (!resolver || !resolver(hash, bytes)) {
        error = "v4_tree_node_missing";
        return false;
    }
    out.bytes = std::move(bytes);
    out.hash = hash;
    return true;
}

} // namespace

bool AppendManifestDelta(const std::string &existingRoot, uint64_t existingTokens,
                         const std::vector<BundleRef> &delta, const NodeResolver &resolver,
                         std::vector<TreeNode> &outNodes, std::string &outRootHash,
                         uint64_t &outTokens, std::string &error) {
    outNodes.clear();
    outRootHash.clear();
    outTokens = existingTokens;
    error.clear();
    if (delta.empty()) return Fail(error, "v4_tree_empty_delta");
    uint64_t expected = existingTokens;
    for (const auto &bundle : delta) {
        // The delta must continue exactly where the tree ends: a gap or an
        // overlap would silently change what a token position means.
        if (bundle.tokenStart != expected || bundle.tokenEnd <= bundle.tokenStart) {
            return Fail(error, "v4_tree_delta_gap");
        }
        if (bundle.bytes == 0 || bundle.bytes > kMaxObjectBytes) {
            return Fail(error, "v4_tree_bundle_bytes");
        }
        if (!IsDigest(bundle.hash)) return Fail(error, "v4_tree_bundle_digest");
        expected = bundle.tokenEnd;
    }

    std::vector<TreeBuildNode> level;
    if (existingRoot.empty()) {
        if (existingTokens != 0) return Fail(error, "v4_tree_missing_root");
        level = BuildLeaves(delta, outNodes);
    } else {
        std::vector<ManifestNode> spine;
        std::vector<TreeBuildNode> spineNodes;
        std::string hash = existingRoot;
        uint64_t depth = 0;
        while (true) {
            TreeBuildNode raw;
            if (!FetchNode(resolver, hash, raw, error)) return false;
            ManifestNode node;
            if (!DecodeManifestNode(raw.bytes.data(), raw.bytes.size(), depth, false, 0, 0, node,
                                    error)) {
                return false;
            }
            raw.leaf = node.leaf;
            raw.start = node.tokenStart;
            raw.end = node.tokenEnd;
            spine.push_back(node);
            spineNodes.push_back(std::move(raw));
            if (node.leaf) break;
            hash = node.children.back().hash;
            if (++depth > kMaxManifestDepth) return Fail(error, "v4_tree_depth");
        }
        if (!spine.back().leaf) return Fail(error, "v4_tree_shape");
        if (spineNodes.back().end != existingTokens) return Fail(error, "v4_tree_token_mismatch");
        std::vector<BundleRef> tail = spine.back().bundles;
        tail.insert(tail.end(), delta.begin(), delta.end());
        std::vector<TreeBuildNode> replacement = BuildLeaves(tail, outNodes);
        for (size_t levelIndex = spine.size(); levelIndex-- > 1;) {
            const ManifestNode &parent = spine[levelIndex - 1];
            std::vector<TreeBuildNode> children;
            for (size_t i = 0; i + 1 < parent.children.size(); ++i) {
                const ChildRef &child = parent.children[i];
                TreeBuildNode raw;
                if (!FetchNode(resolver, child.hash, raw, error)) return false;
                raw.start = child.tokenStart;
                raw.end = child.tokenEnd;
                children.push_back(std::move(raw));
            }
            children.insert(children.end(), replacement.begin(), replacement.end());
            replacement = BuildLevel(children, outNodes);
        }
        level = replacement;
    }
    while (level.size() > 1) level = BuildLevel(level, outNodes);
    if (level.size() != 1) return Fail(error, "v4_tree_root");
    outRootHash = level.front().hash;
    outTokens = level.front().end;
    return true;
}

bool FlattenManifest(const std::string &root, const NodeResolver &resolver,
                     std::vector<BundleRef> &out, std::string &error) {
    out.clear();
    error.clear();
    if (!IsDigest(root)) return Fail(error, "v4_tree_root_digest");
    struct Frame {
        std::string hash;
        uint64_t depth = 0;
    };
    std::set<std::string> visited;
    // Explicit stack: a hostile deep tree cannot overflow the C++ stack here.
    std::vector<Frame> stack;
    stack.push_back({root, 0});
    while (!stack.empty()) {
        const Frame frame = stack.back();
        stack.pop_back();
        // A node reached twice means the structure is not a tree.
        if (!visited.insert(frame.hash).second) return Fail(error, "v4_tree_cycle");
        TreeBuildNode raw;
        if (!FetchNode(resolver, frame.hash, raw, error)) return false;
        ManifestNode node;
        if (!DecodeManifestNode(raw.bytes.data(), raw.bytes.size(), frame.depth, false, 0, 0, node,
                                error)) {
            return false;
        }
        if (node.leaf) {
            for (const auto &bundle : node.bundles) out.push_back(bundle);
            continue;
        }
        // Push in reverse so the leftmost child is processed first.
        for (size_t i = node.children.size(); i-- > 0;) {
            stack.push_back({node.children[i].hash, frame.depth + 1});
        }
    }
    // The flattened bundles must be contiguous: a gap means a lost token range.
    for (size_t i = 1; i < out.size(); ++i) {
        if (out[i].tokenStart != out[i - 1].tokenEnd) {
            return Fail(error, "v4_tree_flatten_gap");
        }
    }
    return true;
}


} // namespace prefix_v4
} // namespace fastllm
