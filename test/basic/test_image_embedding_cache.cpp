// g++ -std=c++17 -pthread -Iinclude test/basic/test_image_embedding_cache.cpp -o /tmp/test_image_embedding_cache
#include "utils/image_embedding_cache.h"
#include <cassert>
#include <thread>

static void SetCapacityEnv(const char *value) {
#ifdef _WIN32
    _putenv_s("FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES", value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv("FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES");
    } else {
        setenv("FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES", value, 1);
    }
#endif
}

static void TestCapacityEnv() {
    using fastllm::ImageEmbeddingCache;
    const char *previous = std::getenv("FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES");
    const bool hadPrevious = previous != nullptr;
    const std::string saved = hadPrevious ? previous : "";
    const size_t fallback = size_t(512) << 20;
    SetCapacityEnv(nullptr);
    assert(ImageEmbeddingCache::CapacityFromEnv() == fallback);
    for (const char *value : {"", "-1", "+1", "invalid", "512m"}) {
        SetCapacityEnv(value);
        assert(ImageEmbeddingCache::CapacityFromEnv() == fallback);
    }
    SetCapacityEnv("0");
    assert(ImageEmbeddingCache::CapacityFromEnv() == 0);
    SetCapacityEnv("16");
    assert(ImageEmbeddingCache::CapacityFromEnv() == 16);
    const std::string maximum = std::to_string(std::numeric_limits<size_t>::max());
    SetCapacityEnv(maximum.c_str());
    assert(ImageEmbeddingCache::CapacityFromEnv() == std::numeric_limits<size_t>::max());
    SetCapacityEnv((maximum + "0").c_str());
    assert(ImageEmbeddingCache::CapacityFromEnv() == fallback);
    SetCapacityEnv(hadPrevious ? saved.c_str() : nullptr);
}

int main() {
    using fastllm::ImageEmbeddingCache;
    TestCapacityEnv();
    const float a[] = {1, 2}, b[] = {3, 4}, c[] = {5, 6};
    ImageEmbeddingCache cache(16);
    assert(cache.Put("a", a, 2));
    assert(cache.Put("b", b, 2));
    std::vector<float> output = {9};
    assert(cache.Append("a", output) == 2);
    assert(output == std::vector<float>({9, 1, 2}));
    assert(cache.Put("c", c, 2)); // a was touched: b is the victim.
    assert(cache.Append("b", output) == 0);
    assert(cache.SizeBytes() == 16);
    assert(!cache.Put("too-big", a, 5)); // Must reject before reading/evicting.
    assert(cache.Append("a", output) == 2);
    assert(cache.Put("a", b, 2)); // Existing immutable result stays unchanged.
    output.clear();
    assert(cache.Append("a", output) == 2);
    assert(output == std::vector<float>({1, 2}));
    cache.Clear();
    assert(cache.SizeBytes() == 0 && cache.Append("a", output) == 0);
    assert(output == std::vector<float>({1, 2})); // Output owns its copy after eviction.

    ImageEmbeddingCache disabled(0), other(16);
    assert(!disabled.Put("a", a, 2));
    assert(disabled.Append("a", output) == 0);
    assert(other.Append("a", output) == 0);
    std::string binaryKey(32, '\0');
    assert(other.Put(binaryKey, a, 2));
    assert(other.Append(binaryKey, output) == 2);
    binaryKey[31] = 1;
    assert(other.Append(binaryKey, output) == 0);

    ImageEmbeddingCache entryLimited(4096);
    for (int i = 0; i < 129; ++i) {
        assert(entryLimited.Put(std::to_string(i), a, 1));
    }
    assert(entryLimited.SizeBytes() == 128 * sizeof(float));
    assert(entryLimited.Append("0", output) == 0);
    assert(entryLimited.Append("1", output) == 1);
    assert(entryLimited.Append("128", output) == 1);

    ImageEmbeddingCache concurrent(1024);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 1000; ++i) {
                std::string key = std::to_string(t * 1000 + i);
                concurrent.Put(key, a, 2);
                std::vector<float> local;
                if (concurrent.Append(key, local)) {
                    assert(local == std::vector<float>({1, 2}));
                }
                assert(concurrent.SizeBytes() <= 1024);
            }
        });
    }
    for (auto &thread : threads) thread.join();
    concurrent.Clear();
    assert(concurrent.SizeBytes() == 0);
}
