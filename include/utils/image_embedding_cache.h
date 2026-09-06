#ifndef FASTLLM_IMAGE_EMBEDDING_CACHE_H
#define FASTLLM_IMAGE_EMBEDDING_CACHE_H

#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fastllm {

// CPU-only, model-owned storage. Copies into the request output so cache eviction
// cannot invalidate a Data tensor or leave shared GPU allocations alive.
class ImageEmbeddingCache {
public:
    static size_t CapacityFromEnv() {
        const size_t fallback = size_t(512) << 20;
        const char *value = std::getenv("FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES");
        if (value == nullptr || *value == '\0') {
            return fallback;
        }
        size_t result = 0;
        for (const char *p = value; *p; ++p) {
            if (*p < '0' || *p > '9' ||
                result > (std::numeric_limits<size_t>::max() - (*p - '0')) / 10) {
                return fallback;
            }
            result = result * 10 + (*p - '0');
        }
        return result;
    }

    explicit ImageEmbeddingCache(size_t capacity = CapacityFromEnv()) : capacity(capacity) {}

    size_t Capacity() const { return capacity; }

    size_t SizeBytes() const {
        std::lock_guard<std::mutex> guard(mutex);
        return sizeBytes;
    }

    // Returns the number of floats appended, or zero on a miss.
    size_t Append(const std::string &key, std::vector<float> &output) {
        std::lock_guard<std::mutex> guard(mutex);
        auto it = index.find(key);
        if (it == index.end()) {
            return 0;
        }
        auto entry = it->second;
        output.insert(output.end(), entry->values.get(), entry->values.get() + entry->count);
        entries.splice(entries.begin(), entries, entry);
        return entry->count;
    }

    bool Put(const std::string &key, const float *values, size_t count) {
        if (key.empty() || count == 0 || count > capacity / sizeof(float)) {
            return false;
        }
        std::lock_guard<std::mutex> guard(mutex);
        auto found = index.find(key);
        if (found != index.end()) {
            entries.splice(entries.begin(), entries, found->second);
            return true;
        }
        const size_t bytes = count * sizeof(float);
        // A count limit also bounds metadata for very small embeddings.
        while (!entries.empty() && (sizeBytes > capacity - bytes || entries.size() >= 128)) {
            sizeBytes -= entries.back().count * sizeof(float);
            index.erase(entries.back().key);
            entries.pop_back();
        }
        bool inserted = false;
        try {
            std::unique_ptr<float[]> copy(new float[count]);
            std::memcpy(copy.get(), values, bytes);
            entries.push_front(Entry{key, std::move(copy), count});
            inserted = true;
            index.emplace(key, entries.begin());
        } catch (const std::bad_alloc &) {
            if (inserted) {
                entries.pop_front();
            }
            return false; // Retaining an optional cache must not fail generation.
        }
        sizeBytes += bytes;
        return true;
    }

    void Clear() {
        std::lock_guard<std::mutex> guard(mutex);
        index.clear();
        entries.clear();
        sizeBytes = 0;
    }

private:
    struct Entry {
        std::string key;
        std::unique_ptr<float[]> values;
        size_t count;
    };
    const size_t capacity;
    size_t sizeBytes = 0;
    std::list<Entry> entries;
    std::unordered_map<std::string, std::list<Entry>::iterator> index;
    mutable std::mutex mutex;
};

} // namespace fastllm
#endif
