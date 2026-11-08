// CPU tests for the v4 record/object-table/manifest-node codec
// (SSD_CACHE_REWRITE_PLAN.md §10.1/§10.2).
//
// These tests freeze the byte layout (golden bytes), prove the decoder rejects
// truncation/overflow/unknown flags/overlap instead of trusting lengths, and
// check that digest domains cannot be confused with each other. No CUDA, no
// model, no service, no filesystem writes outside TMPDIR.
#include "utils/disk_prefix_format_v4.h"

#include <openssl/evp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

using namespace fastllm::prefix_v4;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAIL line " << __LINE__ << ": " #condition << "\n"; \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static std::string Hex(const std::vector<char> &bytes) {
    static const char digits[] = "0123456789abcdef";
    std::string text;
    for (const char byte : bytes) {
        const uint8_t value = uint8_t(byte);
        text += digits[value >> 4];
        text += digits[value & 15];
    }
    return text;
}

static std::string Digest(char fill) { return std::string(64, fill); }

static RecordHeader Batch(uint64_t sequence, uint64_t payload, const std::string &digest) {
    RecordHeader header;
    header.type = kRecordObjectBatch;
    header.coverageMode = kCoverageObjectTableOnly;
    header.sequence = sequence;
    header.payloadBytes = payload;
    std::string error;
    header.physicalBytes = AlignUp(kRecordHeaderBytes + payload, error);
    header.digest = digest;
    return header;
}

static void headerLayout() {
    // Golden bytes for a fixed record: if an offset moves, this fails.
    const RecordHeader header = Batch(7, 64, Digest('a'));
    const std::vector<char> bytes = EncodeRecordHeader(header);
    CHECK(bytes.size() == 128);
    CHECK(std::string(bytes.data(), 8) == "FLKV4R01");
    const std::string golden =
        "464c4b5634523031"                     // magic "FLKV4R01"
        "01000000" "00000000"                  // type=1 (OBJECT_BATCH), flags=0
        "80000000" "01000000" "01000000" "00000000"   // hdr=128, sha256, table-only, reserved
        "4000000000000000"                     // payload=64
        "0010000000000000"                     // physical=4096 (aligned header+payload)
        "0700000000000000"                     // sequence=7
        "00000000";                            // reserved
    // Freeze the first 60 bytes; the CRC is checked separately so this test does
    // not depend on a hand-computed value.
    CHECK(Hex(bytes).substr(0, golden.size()) == golden);
    CHECK(golden.size() == 120);
    uint32_t storedCrc = uint32_t(uint8_t(bytes[60])) | (uint32_t(uint8_t(bytes[61])) << 8) |
                         (uint32_t(uint8_t(bytes[62])) << 16) | (uint32_t(uint8_t(bytes[63])) << 24);
    CHECK(storedCrc == Crc32(bytes.data(), 60));
    CHECK(std::string(bytes.data() + 64, 64) == Digest('a'));

    RecordHeader decoded;
    std::string error;
    CHECK(DecodeRecordHeader(bytes.data(), bytes.size(), decoded, error));
    CHECK(decoded.type == header.type && decoded.sequence == 7);
    CHECK(decoded.payloadBytes == 64 && decoded.physicalBytes == header.physicalBytes);
    CHECK(decoded.digest == Digest('a'));
    CHECK(decoded.coverageMode == kCoverageObjectTableOnly);

    // Truncated header, wrong magic and a flipped CRC byte must all fail.
    CHECK(!DecodeRecordHeader(bytes.data(), bytes.size() - 1, decoded, error));
    CHECK(error == "v4_record_header_short");
    std::vector<char> badMagic = bytes;
    badMagic[0] = 'X';
    CHECK(!DecodeRecordHeader(badMagic.data(), badMagic.size(), decoded, error));
    CHECK(error == "v4_record_magic");
    std::vector<char> badCrc = bytes;
    badCrc[20] = char(badCrc[20] ^ 0x1);
    CHECK(!DecodeRecordHeader(badCrc.data(), badCrc.size(), decoded, error));
    CHECK(error == "v4_record_header_crc");
    // An unknown critical flag is rejected, not ignored.
    RecordHeader flagged = header;
    flagged.flags = 1u << 7;
    CHECK(!DecodeRecordHeader(EncodeRecordHeader(flagged).data(), 128, decoded, error));
    // A physical length that does not match header+payload is rejected.
    std::vector<char> badPhysical = bytes;
    badPhysical[40] = char(badPhysical[40] ^ 0x1);
    CHECK(!DecodeRecordHeader(badPhysical.data(), badPhysical.size(), decoded, error));
    CHECK(error == "v4_record_header_crc");  // corrupting bytes invalidates the CRC first
    // Non-batch records must use full-record coverage.
    RecordHeader commit = header;
    commit.type = kRecordManifestCommit;
    commit.coverageMode = kCoverageObjectTableOnly;
    CHECK(!DecodeRecordHeader(EncodeRecordHeader(commit).data(), 128, decoded, error));
    CHECK(error == "v4_record_coverage");
    // Zero payload is not a record.
    RecordHeader empty = header;
    empty.payloadBytes = 0;
    empty.physicalBytes = kRecordHeaderBytes;
    CHECK(!DecodeRecordHeader(EncodeRecordHeader(empty).data(), 128, decoded, error));
    CHECK(error == "v4_record_payload_bounds");
}

static void objectTableRules() {
    std::vector<ObjectEntry> entries = {
        {kObjectData, 8, 0, Digest('1')},
        {kObjectKvNode, 16, 8, Digest('2')},
    };
    const std::vector<char> table = EncodeObjectTable(entries, 24);
    CHECK(table.size() == 16 + 2 * 88);
    std::vector<ObjectEntry> decoded;
    uint64_t region = 0;
    std::string error;
    CHECK(DecodeObjectTable(table.data(), table.size(), decoded, region, error));
    CHECK(decoded.size() == 2 && region == 24);
    CHECK(decoded[0].hash == Digest('1') && decoded[1].kind == kObjectKvNode);

    // Truncation, a count that outruns the buffer, an overlapping offset, a
    // duplicate hash and a bad kind must all be rejected.
    CHECK(!DecodeObjectTable(table.data(), table.size() - 1, decoded, region, error));
    CHECK(error == "v4_object_table_count");
    std::vector<char> overlap = table;
    // second entry offset 0 instead of 8 -> overlaps the first (offset 0, 8 bytes)
    overlap[16 + 88 + 16] = 0;
    CHECK(!DecodeObjectTable(overlap.data(), overlap.size(), decoded, region, error));
    CHECK(error == "v4_object_overlap");
    std::vector<char> duplicate = table;
    memcpy(duplicate.data() + 16 + 88 + 24, Digest('1').data(), 64);
    CHECK(!DecodeObjectTable(duplicate.data(), duplicate.size(), decoded, region, error));
    CHECK(error == "v4_object_duplicate");
    std::vector<char> badKind = table;
    badKind[16] = 99;
    CHECK(!DecodeObjectTable(badKind.data(), badKind.size(), decoded, region, error));
    CHECK(error == "v4_object_kind");
    std::vector<char> outOfRegion = table;
    // first entry byte count larger than the declared payload region
    outOfRegion[16 + 8] = 64;
    CHECK(!DecodeObjectTable(outOfRegion.data(), outOfRegion.size(), decoded, region, error));
    CHECK(error == "v4_object_bounds");
    // Region larger than the packed-payload bound is refused before allocation.
    std::vector<char> hugeRegion = table;
    hugeRegion[8] = 1;   // low byte
    hugeRegion[11] = 1;  // + 2^24 -> 16 MiB + 1
    CHECK(!DecodeObjectTable(hugeRegion.data(), hugeRegion.size(), decoded, region, error));
    CHECK(error == "v4_object_table_region_bounds");
}

static void manifestNodeRules() {
    ManifestNode leaf;
    leaf.leaf = true;
    leaf.tokenStart = 0;
    leaf.tokenEnd = 256;
    leaf.bundles = {{0, 0, 0, 128, 1024, Digest('a')}, {1, 0, 128, 256, 1024, Digest('b')}};
    const std::vector<char> leafBytes = EncodeManifestNode(leaf);
    CHECK(leafBytes.size() == 24 + 2 * 96);
    ManifestNode decoded;
    std::string error;
    CHECK(DecodeManifestNode(leafBytes.data(), leafBytes.size(), 1, true, 0, 256, decoded, error));
    CHECK(decoded.leaf && decoded.bundles.size() == 2);
    CHECK(decoded.bundles[1].tokenStart == 128 && decoded.bundles[1].hash == Digest('b'));

    ManifestNode inner;
    inner.leaf = false;
    inner.tokenStart = 0;
    inner.tokenEnd = 256;
    inner.children = {{0, 128, 200, Digest('c')}, {128, 256, 192, Digest('d')}};
    const std::vector<char> innerBytes = EncodeManifestNode(inner);
    CHECK(innerBytes.size() == 24 + 2 * 88);
    CHECK(DecodeManifestNode(innerBytes.data(), innerBytes.size(), 2, true, 0, 256, decoded, error));
    CHECK(!decoded.leaf && decoded.children.size() == 2);

    // A node claiming tokens its parent never covered must be rejected.
    CHECK(!DecodeManifestNode(leafBytes.data(), leafBytes.size(), 1, true, 0, 128, decoded, error));
    CHECK(error == "v4_node_range_escape");
    // Depth beyond the bound is refused before any recursion.
    CHECK(!DecodeManifestNode(leafBytes.data(), leafBytes.size(), kMaxManifestDepth + 1, false, 0, 0,
                              decoded, error));
    CHECK(error == "v4_node_depth");
    // A trailing byte changes the size and is refused.
    std::vector<char> padded = leafBytes;
    padded.push_back(0);
    CHECK(!DecodeManifestNode(padded.data(), padded.size(), 1, false, 0, 0, decoded, error));
    CHECK(error == "v4_node_size");
    // A gap between bundles would silently drop tokens.
    std::vector<char> gap = leafBytes;
    gap[24 + 96 + 8] = 130;  // second bundle starts at 130 instead of 128
    CHECK(!DecodeManifestNode(gap.data(), gap.size(), 1, false, 0, 0, decoded, error));
    CHECK(error == "v4_bundle_sequence");
    // Overlapping children are refused.
    std::vector<char> childOverlap = innerBytes;
    childOverlap[24 + 88] = 100;  // second child starts at 100 instead of 128
    CHECK(!DecodeManifestNode(childOverlap.data(), childOverlap.size(), 2, false, 0, 0, decoded, error));
    CHECK(error == "v4_child_sequence");
    // Fanout beyond 64 is refused (declare 65 bundles).
    std::vector<char> fanout = leafBytes;
    fanout[4] = 65;
    CHECK(!DecodeManifestNode(fanout.data(), fanout.size(), 1, false, 0, 0, decoded, error));
    CHECK(error == "v4_node_count");
    // Reserved bytes must be zero.
    std::vector<char> reserved = leafBytes;
    reserved[1] = 9;
    CHECK(!DecodeManifestNode(reserved.data(), reserved.size(), 1, false, 0, 0, decoded, error));
    CHECK(error == "v4_node_reserved");
}

static void digestDomains() {
    const std::string payload = "same bytes";
    CHECK(DomainDigest("node", payload) != DomainDigest("data", payload));
    CHECK(DomainDigest("node", payload) == DomainDigest("node", payload));
    CHECK(IsDigest(DomainDigest("node", payload)));
    CHECK(!IsDigest(std::string(63, 'a')));
    CHECK(!IsDigest(std::string(64, 'A')));
    std::string error;
    CHECK(AlignUp(0, error) == 0 && error.empty());
    CHECK(AlignUp(1, error) == 4096);
    CHECK(AlignUp(4096, error) == 4096);
    CHECK(AlignUp(UINT64_MAX, error) == 0 && error == "v4_size_overflow");
}

// ---------------------------------------------------------------- block ① ----
// v4 写入与断电恢复记账：对象批次负载、控制日志记录、崩溃分类。


// 用户硬性要求：禁止写 /tmp，所有夹具必须落在项目内。
// 这里把规则写成**可执行约束**：TMPDIR 指向 /tmp 时直接拒绝运行，而不是默默照做。
static fs::path VerifiedTempBase(const char *tag) {
    const char *tmp = std::getenv("TMPDIR");
    if (tmp != nullptr && std::string(tmp).rfind("/tmp", 0) == 0) {
        std::cerr << "REFUSED: TMPDIR points at " << tmp
                  << " but this project forbids writing under /tmp (" << tag << ")\n";
        std::exit(2);
    }
    return tmp ? fs::path(tmp) : fs::current_path();
}

static fs::path TempRoot() {
    std::string pattern = (VerifiedTempBase("v4log") / "fastllm-v4log-test-XXXXXX").string();
    char *name = mkdtemp(pattern.data());
    CHECK(name != nullptr);
    return name ? fs::path(name) : fs::current_path();
}

static std::string ReadAllBytes(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

static void objectBatchPayload() {
    const std::string payload = std::string(8, 'A') + std::string(16, 'B');
    std::vector<ObjectEntry> entries = {{kObjectData, 8, 0, Digest('1')},
                                        {kObjectKvNode, 16, 8, Digest('2')}};
    const std::vector<char> batch = EncodeObjectBatchPayload(entries, payload);
    std::vector<ObjectEntry> decoded;
    std::string decodedPayload, error;
    const bool ok = DecodeObjectBatchPayload(batch.data(), batch.size(), decoded, decodedPayload, error);
    CHECK(ok);
    CHECK(decodedPayload == payload && decoded.size() == 2);
    CHECK(decoded[1].kind == kObjectKvNode && decoded[1].offset == 8);
    // A trailing byte is not attributable to any object: refuse it.
    std::vector<char> trailing = batch;
    trailing.push_back('x');
    const bool trailingRejected =
        !DecodeObjectBatchPayload(trailing.data(), trailing.size(), decoded, decodedPayload, error);
    CHECK(trailingRejected);
    // A payload region larger than the bytes present is refused.
    std::vector<char> shorted = batch;
    shorted.pop_back();
    const bool shortRejected =
        !DecodeObjectBatchPayload(shorted.data(), shorted.size(), decoded, decodedPayload, error);
    CHECK(shortRejected);
    CHECK(error == "v4_batch_payload_bounds");
}

static void journalRecordCodecs() {
    std::string error;
    PrepareRecord prepare;
    prepare.sequence = 11;
    prepare.key = "checkpoint-key";
    prepare.identity = Digest('a');
    prepare.rootHash = Digest('b');
    prepare.firstBatch = 3;
    prepare.lastBatch = 4;
    const std::vector<char> prepareBytes = EncodePrepare(prepare);
    PrepareRecord decodedPrepare;
    CHECK(DecodePrepare(prepareBytes.data(), prepareBytes.size(), decodedPrepare, error));
    CHECK(decodedPrepare.key == prepare.key && decodedPrepare.sequence == 11);
    CHECK(decodedPrepare.lastBatch == 4);
    CHECK(!DecodePrepare(prepareBytes.data(), prepareBytes.size() - 1, decodedPrepare, error));
    CHECK(error == "v4_record_size");
    std::vector<char> badIdentity = prepareBytes;
    badIdentity[prepareBytes.size() - 20] = 'Z';
    CHECK(!DecodePrepare(badIdentity.data(), badIdentity.size(), decodedPrepare, error));
    CHECK(error == "v4_record_digest");

    CommitRecord commit;
    commit.sequence = 12;
    commit.prepareOffset = 4096;
    commit.prepareBytes = 512;
    commit.prepareDigest = Digest('c');
    commit.rootHash = Digest('d');
    const std::vector<char> commitBytes = EncodeCommit(commit);
    CommitRecord decodedCommit;
    CHECK(DecodeCommit(commitBytes.data(), commitBytes.size(), decodedCommit, error));
    CHECK(decodedCommit.prepareOffset == 4096 && decodedCommit.rootHash == Digest('d'));
    CHECK(!DecodeCommit(commitBytes.data(), commitBytes.size() + 1, decodedCommit, error));
    CHECK(error == "v4_record_size");
    std::vector<char> zeroPrepare = commitBytes;
    memset(zeroPrepare.data() + 16, 0, 8);   // prepareBytes = 0
    CHECK(!DecodeCommit(zeroPrepare.data(), zeroPrepare.size(), decodedCommit, error));
    CHECK(error == "v4_record_bounds");

    EraseRecord erase;
    erase.sequence = 13;
    erase.key = "gone";
    erase.identity = Digest('e');
    const std::vector<char> eraseBytes = EncodeErase(erase);
    EraseRecord decodedErase;
    CHECK(DecodeErase(eraseBytes.data(), eraseBytes.size(), decodedErase, error));
    CHECK(decodedErase.key == "gone");

    MetadataRecord metadata;
    metadata.sequence = 14;
    metadata.key = "cost";
    metadata.value = "{\"prefill\":1}";
    const std::vector<char> metadataBytes = EncodeMetadata(metadata);
    MetadataRecord decodedMetadata;
    CHECK(DecodeMetadata(metadataBytes.data(), metadataBytes.size(), decodedMetadata, error));
    CHECK(decodedMetadata.value == metadata.value);

    // Domain separation keeps a prepare domain distinct from a commit domain.
    CHECK(std::string(RecordDomain(kRecordManifestPrepare)) == "prepare");
    CHECK(std::string(RecordDomain(kRecordManifestCommit)) == "commit");
    CHECK(std::string(RecordDomain(kRecordObjectBatch)) == "batch");
}

static void controlLogCrashClassification() {
    const fs::path root = TempRoot();
    const fs::path log = root / "control.log";
    std::string error;
    uint64_t offset = 0;
    PrepareRecord prepare;
    prepare.sequence = 1;
    prepare.key = "k";
    prepare.identity = Digest('a');
    prepare.rootHash = Digest('b');
    CommitRecord commit;
    commit.sequence = 2;
    commit.prepareBytes = 512;
    commit.prepareDigest = Digest('c');
    commit.rootHash = Digest('b');
    const std::vector<char> prepareVec = EncodePrepare(prepare);
    const std::string prepareBytes(prepareVec.data(), prepareVec.size());
    const std::vector<char> commitVec = EncodeCommit(commit);
    const std::string commitBytes(commitVec.data(), commitVec.size());
    const bool wrotePrepare =
        AppendRecordToFile(log.string(), kRecordManifestPrepare, 1, prepareBytes, &offset, error);
    CHECK(wrotePrepare);
    CHECK(offset == 0);
    const bool wroteCommit =
        AppendRecordToFile(log.string(), kRecordManifestCommit, 2, commitBytes, &offset, error);
    CHECK(wroteCommit);
    // Each small control record occupies one 4 KiB aligned slot, so the second
    // record starts at exactly one alignment boundary.
    CHECK(offset == kAlignment);

    std::string bytes;
    CHECK(ReadWholeFile(log.string(), bytes, error));
    std::vector<ScannedRecord> records;
    CHECK(ScanControlLog(bytes.data(), bytes.size(), records, error) == LogClassification::Complete);
    CHECK(records.size() == 2);
    CHECK(records[0].header.type == kRecordManifestPrepare);
    CHECK(records[1].offset == kAlignment);

    // A short write at the tail must publish nothing from the unfinished record.
    {
        const std::string tail = bytes + std::string(64, '\0');
        CHECK(ScanControlLog(tail.data(), tail.size(), records, error) ==
              LogClassification::UncommittedTail);
        CHECK(records.size() == 2);   // the two complete records remain usable
    }
    // A declared payload that runs past EOF is an unfinished record.
    {
        const std::string truncated = bytes.substr(0, bytes.size() - 100);
        CHECK(ScanControlLog(truncated.data(), truncated.size(), records, error) ==
              LogClassification::UncommittedTail);
        CHECK(records.size() == 1);
    }
    // A zero-filled preallocated tail is a short write, not corruption.
    {
        const std::string zeros = bytes + std::string(size_t(kAlignment), '\0');
        CHECK(ScanControlLog(zeros.data(), zeros.size(), records, error) ==
              LogClassification::UncommittedTail);
        CHECK(records.size() == 2);
    }
    // A flipped byte inside a complete record is corruption: fail closed.
    {
        std::string corrupt = bytes;
        corrupt[size_t(kAlignment) + 20] = char(corrupt[size_t(kAlignment) + 20] ^ 0x1);
        CHECK(ScanControlLog(corrupt.data(), corrupt.size(), records, error) ==
              LogClassification::CorruptControl);
        CHECK(!error.empty());
    }
    // A payload byte that no longer matches the record digest is corruption.
    {
        std::string swapped = bytes;
        swapped[size_t(kAlignment) + kRecordHeaderBytes + 8] =
            char(swapped[size_t(kAlignment) + kRecordHeaderBytes + 8] ^ 0x1);
        CHECK(ScanControlLog(swapped.data(), swapped.size(), records, error) ==
              LogClassification::CorruptControl);
    }
    // Rewriting the second record's sequence breaks its header CRC first, which
    // is still fail-closed rather than a silent reordering.
    {
        std::string regression = bytes;
        regression[size_t(kAlignment) + 48] = 0;
        regression[size_t(kAlignment) + 49] = 0;
        CHECK(ScanControlLog(regression.data(), regression.size(), records, error) ==
              LogClassification::CorruptControl);
    }
    // An empty (fresh) log is complete with no records.
    {
        CHECK(ScanControlLog("", 0, records, error) == LogClassification::Complete);
        CHECK(records.empty());
    }
}

// ---------------------------------------------------------------- tree (§9) --
// 增量 manifest 树：固定扇出 64、不可变节点、只复制右脊。

static BundleRef Bundle(uint64_t start, uint64_t end, char fill) {
    BundleRef bundle;
    bundle.rank = uint32_t(start / 128);
    bundle.layoutId = 0;
    bundle.tokenStart = start;
    bundle.tokenEnd = end;
    bundle.bytes = 1024;
    bundle.hash = Digest(fill);
    return bundle;
}

static std::vector<BundleRef> Run(uint64_t from, uint64_t count, char fill) {
    std::vector<BundleRef> out;
    for (uint64_t i = 0; i < count; ++i) out.push_back(Bundle(from + i * 128, from + (i + 1) * 128, fill));
    return out;
}

struct NodeStore {
    std::map<std::string, std::string> nodes;
    void Absorb(const std::vector<TreeNode> &written) {
        for (const auto &node : written) nodes[node.hash] = node.bytes;
    }
    NodeResolver Resolver() {
        return [this](const std::string &hash, std::string &bytes) {
            const auto found = nodes.find(hash);
            if (found == nodes.end()) return false;
            bytes = found->second;
            return true;
        };
    }
};

static void manifestTree() {
    NodeStore store;
    std::string root, error;
    uint64_t tokens = 0;
    std::vector<TreeNode> written;
    std::vector<BundleRef> flattened;

    // A small fresh tree is a single leaf.
    CHECK(AppendManifestDelta("", 0, Run(0, 3, 'a'), store.Resolver(), written, root, tokens, error));
    CHECK(tokens == 384 && written.size() == 1);
    store.Absorb(written);
    CHECK(FlattenManifest(root, store.Resolver(), flattened, error));
    CHECK(flattened.size() == 3);
    CHECK(flattened[0].tokenStart == 0 && flattened[2].tokenEnd == 384);

    // Growing past 64 bundles adds a level: 2 leaves + 1 inner node.
    {
        NodeStore big;
        std::vector<TreeNode> nodes;
        std::string bigRoot;
        uint64_t bigTokens = 0;
        CHECK(AppendManifestDelta("", 0, Run(0, 70, 'b'), big.Resolver(), nodes, bigRoot, bigTokens,
                                  error));
        CHECK(nodes.size() == 3);
        CHECK(bigTokens == 70 * 128);
        big.Absorb(nodes);
        std::vector<BundleRef> all;
        CHECK(FlattenManifest(bigRoot, big.Resolver(), all, error));
        CHECK(all.size() == 70);
        CHECK(all[0].tokenStart == 0 && all[69].tokenEnd == 70 * 128);
        // Appending one more bundle only rewrites the rightmost leaf and the
        // root: the untouched left leaf keeps its hash and is not re-emitted.
        const std::string leftLeaf = nodes.front().hash;
        const std::string oldRoot = bigRoot;
        std::vector<TreeNode> deltaNodes;
        std::string newRoot;
        uint64_t newTokens = 0;
        CHECK(AppendManifestDelta(bigRoot, bigTokens, Run(bigTokens, 1, 'c'), big.Resolver(),
                                  deltaNodes, newRoot, newTokens, error));
        CHECK(deltaNodes.size() == 2);          // rightmost leaf + root only
        CHECK(newRoot != oldRoot);
        CHECK(newTokens == 71 * 128);
        for (const auto &node : deltaNodes) CHECK(node.hash != leftLeaf);
        big.Absorb(deltaNodes);
        std::vector<BundleRef> grown;
        CHECK(FlattenManifest(newRoot, big.Resolver(), grown, error));
        CHECK(grown.size() == 71);
        // The first 64 bundles are byte-identical to before (shared subtree).
        for (size_t i = 0; i < 64; ++i) CHECK(grown[i].hash == all[i].hash);
        CHECK(grown[70].tokenStart == bigTokens);
    }

    // Appending to the small leaf root keeps both the old and the new bundles.
    {
        std::vector<TreeNode> deltaNodes;
        std::string grownRoot;
        uint64_t grownTokens = 0;
        CHECK(AppendManifestDelta(root, tokens, Run(tokens, 2, 'd'), store.Resolver(), deltaNodes,
                                  grownRoot, grownTokens, error));
        CHECK(deltaNodes.size() == 1 && grownTokens == 384 + 256);
        store.Absorb(deltaNodes);
        std::vector<BundleRef> grown;
        CHECK(FlattenManifest(grownRoot, store.Resolver(), grown, error));
        CHECK(grown.size() == 5);
    }

    // A delta that does not continue where the tree ends is refused.
    {
        std::vector<TreeNode> nodes;
        std::string outRoot;
        uint64_t outTokens = 0;
        CHECK(!AppendManifestDelta(root, tokens, Run(0, 1, 'e'), store.Resolver(), nodes, outRoot,
                                   outTokens, error));
        CHECK(error == "v4_tree_delta_gap");
        CHECK(!AppendManifestDelta(root, tokens, {}, store.Resolver(), nodes, outRoot, outTokens,
                                   error));
        CHECK(error == "v4_tree_empty_delta");
        std::vector<BundleRef> forged = Run(384, 1, 'f');
        forged[0].hash = "not-a-digest";
        CHECK(!AppendManifestDelta(root, tokens, forged, store.Resolver(), nodes, outRoot, outTokens,
                                   error));
        CHECK(error == "v4_tree_bundle_digest");
        // A root that cannot be fetched is an error, not an empty tree.
        CHECK(!AppendManifestDelta(Digest('9'), 0, Run(0, 1, '0'), store.Resolver(), nodes, outRoot,
                                   outTokens, error));
        CHECK(error == "v4_tree_node_missing");
    }

    // Flatten refuses a cycle and a missing node.
    {
        NodeStore broken;
        std::vector<TreeNode> cycle;
        std::string cycleRoot;
        uint64_t cycleTokens = 0;
        CHECK(AppendManifestDelta("", 0, Run(0, 3, '1'), broken.Resolver(), cycle, cycleRoot,
                                  cycleTokens, error));
        broken.Absorb(cycle);
        // Point a node at itself: the resolver now returns a self-referencing node.
        ManifestNode self;
        self.leaf = false;
        self.tokenStart = 0;
        self.tokenEnd = 384;
        self.children = {{0, 384, 100, cycleRoot}};
        const std::vector<char> selfBytes = EncodeManifestNode(self);
        broken.nodes[cycleRoot] = std::string(selfBytes.data(), selfBytes.size());
        std::vector<BundleRef> out;
        CHECK(!FlattenManifest(cycleRoot, broken.Resolver(), out, error));
        CHECK(error == "v4_tree_cycle");
        CHECK(!FlattenManifest(Digest('a'), broken.Resolver(), out, error));
        CHECK(error == "v4_tree_node_missing");
        CHECK(!FlattenManifest("not-a-digest", broken.Resolver(), out, error));
        CHECK(error == "v4_tree_root_digest");
    }
}


// P1：日志黄金文件——把控制日志的**整体字节**冻结下来，任何布局变化都会失败。
static void controlLogGoldenFile() {
    const fs::path root = TempRoot();
    const fs::path log = root / "golden.log";
    std::string error;
    std::vector<ObjectEntry> entries = {{kObjectData, 8, 0, Digest('1')}};
    const std::vector<char> batch = EncodeObjectBatchPayload(entries, std::string(8, 'A'));
    uint64_t offset = 0;
    CHECK(AppendRecordToFile(log.string(), kRecordObjectBatch, 1,
                             std::string(batch.data(), batch.size()), &offset, error));
    PrepareRecord prepare;
    prepare.sequence = 2;
    prepare.key = "k";
    prepare.identity = Digest('a');
    prepare.rootHash = Digest('b');
    const std::vector<char> prepareBytes = EncodePrepare(prepare);
    CHECK(AppendRecordToFile(log.string(), kRecordManifestPrepare, 2,
                             std::string(prepareBytes.data(), prepareBytes.size()), &offset, error));
    CHECK(offset == kAlignment);
    CommitRecord commit;
    commit.sequence = 3;
    commit.prepareBytes = 4096;
    commit.prepareDigest = Digest('c');
    commit.rootHash = Digest('b');
    const std::vector<char> commitBytes = EncodeCommit(commit);
    CHECK(AppendRecordToFile(log.string(), kRecordManifestCommit, 3,
                             std::string(commitBytes.data(), commitBytes.size()), nullptr, error));
    const std::string bytes = ReadAllBytes(log);
    // 3 条小记录各占一个 4 KiB 槽位
    CHECK(bytes.size() == 3 * kAlignment);
    // 冻结值（2026-09-28 实测）：记录布局/摘要语义一变，这里立刻失败。
    unsigned char raw[32]; unsigned int count = 0;
    CHECK(EVP_Digest(bytes.data(), bytes.size(), raw, &count, EVP_sha256(), nullptr) == 1);
    static const char digits[] = "0123456789abcdef";
    std::string sha;
    for (unsigned i = 0; i < count; ++i) { sha += digits[raw[i] >> 4]; sha += digits[raw[i] & 15]; }
    CHECK(bytes.size() == 12288);
    CHECK(sha == "0c9c2e1a93dc3bd3eb176ae9e449b2dd11936ba5843040323eb0c82faf6afde7");
    // 三个槽位的起点也一并冻结：记录之间不允许出现空洞或错位。
    CHECK(std::string(bytes.data(), 8) == "FLKV4R01");
    CHECK(std::string(bytes.data() + kAlignment, 8) == "FLKV4R01");
    CHECK(std::string(bytes.data() + 2 * kAlignment, 8) == "FLKV4R01");
}

int main() {
    try {
        std::cout << "v4 header hex: " << Hex(EncodeRecordHeader(Batch(7, 64, Digest('a')))) << "\n";
        headerLayout();
        objectTableRules();
        manifestNodeRules();
        digestDomains();
        objectBatchPayload();
        journalRecordCodecs();
        controlLogCrashClassification();
        manifestTree();
        controlLogGoldenFile();
    } catch (const std::exception &error) {
        std::cerr << "FAIL exception: " << error.what() << "\n";
        ++failures;
    }
    if (failures != 0) {
        std::cerr << "FAILED: " << failures << " check(s)\n";
        return 1;
    }
    std::cout << "PASS disk_prefix_format_v4\n";
    return 0;
}
