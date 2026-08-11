#include "devices/disk/disk_expert_cache.h"
#include "fastllm.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace fastllm;

namespace {
    // Fake GGUF disk weight: isFake=true so the cache's release path only
    // deletes the Data (no owned descriptor, no CUDA extras). Once a weight is
    // handed to the cache the test no longer owns it.
    Data *MakeWeight(const std::string &name, int ggmlType,
                     const std::string &file, long long offset) {
        Data *weight = new Data();
        weight->name = name;
        weight->dataType = DataType::DATA_GGUF_FORMAT;
        weight->ggmlType = ggmlType;
        weight->dims = {1, 2048, 4096};
        weight->isFake = true;
        DiskWeightPart part;
        part.fileName = file;
        part.fileOffset = offset;
        part.isScalePart = false;
        weight->diskWeightParts.push_back(part);
        return weight;
    }

    DiskExpertWeightCache &Cache() {
        return DiskExpertWeightCache::Get();
    }

    void TestKeyIdentity() {
        Data *same1 = MakeWeight("layers.5.ffn.experts.9.w1.weight", 20, "s1.gguf", 4096);
        Data *same2 = MakeWeight("layers.5.ffn.experts.9.w1.weight", 20, "s1.gguf", 4096);
        Data *otherSplit = MakeWeight("layers.5.ffn.experts.9.w1.weight", 20, "s2.gguf", 8192);
        Data *otherType = MakeWeight("layers.5.ffn.experts.9.w1.weight", 21, "s1.gguf", 4096);

        assert(MakeDiskWeightCacheKey(same1) == MakeDiskWeightCacheKey(same2));
        // Same logical expert living in a different split must not collide.
        assert(MakeDiskWeightCacheKey(same1) != MakeDiskWeightCacheKey(otherSplit));
        // A different physical quant type must not collide either.
        assert(MakeDiskWeightCacheKey(same1) != MakeDiskWeightCacheKey(otherType));

        delete same1;
        delete same2;
        delete otherSplit;
        delete otherType;
    }

    void TestHitMissAndStats() {
        DiskExpertWeightCache &cache = Cache();
        cache.Clear();
        cache.ConfigureForTesting(1000);
        assert(cache.Enabled());

        DiskExpertCacheStats base = cache.GetStats();
        assert(cache.Lookup("x") == nullptr); // miss

        bool cached = false;
        Data *weight = MakeWeight("x", 20, "f.gguf", 0);
        Data *inserted = cache.Insert("x", weight, 100, cached);
        assert(cached);
        assert(inserted == weight);

        assert(cache.Lookup("x") == weight); // hit, canonical pointer

        DiskExpertCacheStats after = cache.GetStats();
        assert(after.hits == base.hits + 1);
        assert(after.misses == base.misses + 1);
        assert(after.entries == 1);
        assert(after.usedBytes == 100);
        cache.Clear();
    }

    void TestLruEvictionAndRecency() {
        DiskExpertWeightCache &cache = Cache();
        cache.Clear();
        cache.ConfigureForTesting(250);
        DiskExpertCacheStats base = cache.GetStats();

        bool cached = false;
        Data *a = MakeWeight("a", 20, "f.gguf", 0);
        Data *b = MakeWeight("b", 20, "f.gguf", 100);
        Data *c = MakeWeight("c", 20, "f.gguf", 200);
        cache.Insert("a", a, 100, cached);
        assert(cached);
        cache.Insert("b", b, 100, cached);
        assert(cached);

        // Promote a to MRU so b becomes the eviction candidate.
        assert(cache.Lookup("a") == a);

        cache.Insert("c", c, 100, cached);
        assert(cached);

        DiskExpertCacheStats after = cache.GetStats();
        assert(after.entries == 2);
        assert(after.evictions == base.evictions + 1);
        assert(after.usedBytes == 200);
        assert(cache.Lookup("b") == nullptr); // evicted LRU
        assert(cache.Lookup("a") == a);
        assert(cache.Lookup("c") == c);
        cache.Clear();
    }

    void TestOversizedNotCached() {
        DiskExpertWeightCache &cache = Cache();
        cache.Clear();
        cache.ConfigureForTesting(100);

        bool cached = true;
        Data *big = MakeWeight("big", 20, "f.gguf", 0);
        Data *result = cache.Insert("big", big, 500, cached);
        assert(!cached);
        assert(result == big); // returned unchanged, caller keeps ownership

        DiskExpertCacheStats stats = cache.GetStats();
        assert(stats.entries == 0);
        assert(stats.usedBytes == 0);
        delete big;
        cache.Clear();
    }

    void TestDuplicateInsertFreesIncoming() {
        DiskExpertWeightCache &cache = Cache();
        cache.Clear();
        cache.ConfigureForTesting(1000);

        bool cached = false;
        Data *first = MakeWeight("n", 20, "f.gguf", 0);
        Data *second = MakeWeight("n", 20, "f.gguf", 0);
        Data *p1 = cache.Insert("k", first, 100, cached);
        assert(cached && p1 == first);

        // Redundant incoming weight is freed by the cache; existing pointer wins.
        Data *p2 = cache.Insert("k", second, 100, cached);
        assert(cached && p2 == first);

        DiskExpertCacheStats stats = cache.GetStats();
        assert(stats.entries == 1);
        assert(stats.usedBytes == 100); // not double counted
        cache.Clear();
    }

    void TestDisabledCache() {
        DiskExpertWeightCache &cache = Cache();
        cache.Clear();
        cache.ConfigureForTesting(0);
        assert(!cache.Enabled());

        bool cached = true;
        Data *weight = MakeWeight("a", 20, "f.gguf", 0);
        Data *result = cache.Insert("a", weight, 100, cached);
        assert(!cached);
        assert(result == weight);
        assert(cache.Lookup("a") == nullptr);
        delete weight;
        cache.Clear();
    }
}

int main() {
    TestKeyIdentity();
    TestHitMissAndStats();
    TestLruEvictionAndRecency();
    TestOversizedNotCached();
    TestDuplicateInsertFreesIncoming();
    TestDisabledCache();
    std::cout << "DeepSeek V4 disk expert cache tests passed.\n";
    return 0;
}
