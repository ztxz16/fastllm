#include "internal.h"
#include <algorithm>
#include <cstring>
#include <limits>
#ifdef FASTKV_CUDA
#include <cuda_runtime_api.h>
#endif

namespace fastkv {
void Require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
size_t CheckedAdd(size_t a, size_t b) {
    Require(b <= std::numeric_limits<size_t>::max() - a, "FastKV: size overflow");
    return a + b;
}
size_t CheckedMul(size_t a, size_t b) {
    Require(!a || b <= std::numeric_limits<size_t>::max() / a, "FastKV: size overflow");
    return a * b;
}
void Writer::U32(uint32_t v) {
    for (int i = 0; i < 4; ++i)
        bytes.push_back(uint8_t(v >> (8 * i)));
}
void Writer::U64(uint64_t v) {
    for (int i = 0; i < 8; ++i)
        bytes.push_back(uint8_t(v >> (8 * i)));
}
void Writer::Blob(const void *p, size_t n) {
    Require(n <= UINT32_MAX, "FastKV: control field too large");
    U32(uint32_t(n));
    if (n) {
        const auto *b = static_cast<const uint8_t *>(p);
        bytes.insert(bytes.end(), b, b + n);
    }
}
void Writer::String(const std::string &s) {
    Blob(s.data(), s.size());
}
void Writer::Ints(const std::vector<int> &v) {
    Require(v.size() <= UINT32_MAX, "FastKV: too many integers");
    U32(uint32_t(v.size()));
    for (int i : v)
        U32(uint32_t(i));
}
Reader::Reader(const Bytes &b) : Reader(b.data(), b.size()) {}
Reader::Reader(const void *p, size_t n) : data(static_cast<const uint8_t *>(p)), size(n) {}
const uint8_t *Reader::Take(size_t n) {
    Require(n <= Remaining(), "FastKV: truncated control message");
    const uint8_t *p = data ? data + position : nullptr;
    position += n;
    return p;
}
uint32_t Reader::U32() {
    auto *p = Take(4);
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= uint32_t(p[i]) << (8 * i);
    return v;
}
uint64_t Reader::U64() {
    auto *p = Take(8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= uint64_t(p[i]) << (8 * i);
    return v;
}
Bytes Reader::Blob(size_t limit) {
    size_t n = U32();
    Require(n <= limit, "FastKV: field limit");
    auto *p = Take(n);
    return n ? Bytes(p, p + n) : Bytes{};
}
std::string Reader::String(size_t limit) {
    auto b = Blob(limit);
    return std::string(b.begin(), b.end());
}
std::vector<int> Reader::Ints(size_t limit) {
    size_t n = U32();
    Require(n <= limit && n <= Remaining() / 4, "FastKV: integer array limit");
    std::vector<int> v(n);
    for (auto &i : v)
        i = int32_t(U32());
    return v;
}
void Reader::End() const {
    Require(!Remaining(), "FastKV: trailing control bytes");
}

#ifdef FASTKV_CUDA
static void CudaCheck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("FastKV CUDA: ") + cudaGetErrorString(e));
}
struct DeviceScope {
    int previous;
    explicit DeviceScope(int device) {
        CudaCheck(cudaGetDevice(&previous));
        CudaCheck(cudaSetDevice(device));
    }
    ~DeviceScope() { cudaSetDevice(previous); }
};
#endif
struct Buffer::Storage {
    void *data = nullptr;
    size_t size = 0;
    MemoryKind kind = MemoryKind::Host;
    int device = -1;
    uint64_t id = 0;
    std::shared_ptr<void> owner;
    mutable std::mutex mutex;
    mutable std::map<std::string, std::shared_ptr<void>> attachments;
    ~Storage() {
        attachments.clear();
        owner.reset();
    }
};
Buffer Buffer::Borrow(void *p, size_t n, MemoryKind kind, int device, std::shared_ptr<void> owner) {
    Require((!n || p) && owner, "FastKV: buffer requires an allocation lease");
    static std::atomic<uint64_t> next{1};
    Buffer b;
    b.storage = std::make_shared<Storage>();
    b.length = n;
    b.storage->data = p;
    b.storage->size = n;
    b.storage->kind = kind;
    b.storage->device = device;
    b.storage->owner = std::move(owner);
    b.storage->id = next.fetch_add(1);
    return b;
}
Buffer Buffer::Allocate(size_t n, int device) {
    if (device < 0) {
        auto owner =
            std::shared_ptr<uint8_t>(new uint8_t[std::max(size_t(1), n)], std::default_delete<uint8_t[]>());
        return Borrow(owner.get(), n, MemoryKind::Host, -1, owner);
    }
#ifdef FASTKV_CUDA
    DeviceScope scope(device);
    void *p = nullptr;
    CudaCheck(cudaMalloc(&p, std::max(size_t(1), n)));
    auto owner = std::shared_ptr<void>(p, [device](void *p) {
        int old = 0;
        cudaGetDevice(&old);
        cudaSetDevice(device);
        cudaFree(p);
        cudaSetDevice(old);
    });
    return Borrow(p, n, MemoryKind::Cuda, device, std::move(owner));
#else
    throw std::runtime_error("FastKV was built without CUDA memory support");
#endif
}
Buffer Buffer::Slice(size_t at, size_t n) const {
    Require(at <= length && n <= length - at, "FastKV: buffer slice out of range");
    Buffer b = *this;
    b.offset += at;
    b.length = n;
    return b;
}
const void *Buffer::Data() const {
    return storage ? static_cast<uint8_t *>(storage->data) + offset : nullptr;
}
void *Buffer::MutableData() const {
    Require(!readOnly, "FastKV: immutable buffer");
    return const_cast<void *>(Data());
}
Buffer Buffer::ReadOnly() const {
    auto copy = *this;
    copy.readOnly = true;
    return copy;
}
MemoryKind Buffer::Kind() const {
    return storage ? storage->kind : MemoryKind::Host;
}
int Buffer::Device() const {
    return storage ? storage->device : -1;
}
uint64_t Buffer::Id() const {
    return storage ? storage->id : 0;
}
void Buffer::Read(void *host, size_t at, size_t n) const {
    auto b = Slice(at, n);
    if (!n)
        return;
    if (Kind() == MemoryKind::Host) {
        std::memcpy(host, b.Data(), n);
        return;
    }
#ifdef FASTKV_CUDA
    DeviceScope scope(Device());
    CudaCheck(cudaMemcpy(host, b.Data(), n, cudaMemcpyDeviceToHost));
#else
    throw std::runtime_error("FastKV: CUDA unavailable");
#endif
}
void Buffer::Write(const void *host, size_t at, size_t n) const {
    auto b = Slice(at, n);
    if (!n)
        return;
    if (Kind() == MemoryKind::Host) {
        std::memcpy(b.MutableData(), host, n);
        return;
    }
#ifdef FASTKV_CUDA
    DeviceScope scope(Device());
    CudaCheck(cudaMemcpy(b.MutableData(), host, n, cudaMemcpyHostToDevice));
    CudaCheck(cudaStreamSynchronize(nullptr));
#else
    throw std::runtime_error("FastKV: CUDA unavailable");
#endif
}
void Buffer::CopyFrom(const Buffer &source, size_t src, size_t dst, size_t n) const {
    auto a = source.Slice(src, n), b = Slice(dst, n);
    if (!n)
        return;
    if (a.Kind() == MemoryKind::Host) {
        b.Write(a.Data(), 0, n);
        return;
    }
    if (b.Kind() == MemoryKind::Host) {
        a.Read(b.MutableData(), 0, n);
        return;
    }
#ifdef FASTKV_CUDA
    DeviceScope scope(Device());
    if (a.Device() == b.Device())
        CudaCheck(cudaMemcpy(b.MutableData(), a.Data(), n, cudaMemcpyDeviceToDevice));
    else
        CudaCheck(cudaMemcpyPeer(b.MutableData(), b.Device(), a.Data(), a.Device(), n));
    CudaCheck(cudaStreamSynchronize(nullptr));
#else
    throw std::runtime_error("FastKV: CUDA unavailable");
#endif
}
void Buffer::MakeRemoteWritesVisible() const {
#ifdef FASTKV_CUDA
    if (Kind() == MemoryKind::Cuda) {
        DeviceScope scope(Device());
#if CUDART_VERSION >= 11030
        int ordering = 0, flush = 0;
        CudaCheck(cudaDeviceGetAttribute(&ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, Device()));
        CudaCheck(cudaDeviceGetAttribute(&flush, cudaDevAttrGPUDirectRDMAFlushWritesOptions, Device()));
        if (ordering < cudaGPUDirectRDMAWritesOrderingAllDevices &&
            (flush & cudaFlushGPUDirectRDMAWritesOptionHost))
            CudaCheck(cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                         cudaFlushGPUDirectRDMAWritesToAllDevices));
#endif
        // This submission follows the transport completion notification. No
        // consumer is allowed to run concurrently with the remote writes.
        CudaCheck(cudaDeviceSynchronize());
    }
#endif
}
std::shared_ptr<void>
Buffer::Attachment(const std::string &key,
                   const std::function<std::shared_ptr<void>(void *, size_t)> &create) const {
    Require(bool(storage), "FastKV: empty allocation");
    std::lock_guard<std::mutex> lock(storage->mutex);
    auto &v = storage->attachments[key];
    if (!v)
        v = create(storage->data, storage->size);
    return v;
}
bool Service::HasCuda() {
#ifdef FASTKV_CUDA
    return true;
#else
    return false;
#endif
}

Pool::Pool(int n, int cap) : capacity(cap) {
    Require(n > 0 && cap > 0, "FastKV: invalid worker pool");
    for (int i = 0; i < n; ++i)
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> f;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&] { return stop || !queue.empty(); });
                    if (queue.empty())
                        return;
                    f = std::move(queue.front());
                    queue.pop_front();
                }
                try {
                    f();
                } catch (...) { /* packaged tasks carry errors to their owner */
                }
            }
        });
}
Pool::~Pool() {
    Stop();
}
void Pool::Enqueue(std::function<void()> f) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        Require(!stop, "FastKV: service stopped");
        Require(queue.size() < capacity, "FastKV: queue is full");
        queue.push_back(std::move(f));
    }
    cv.notify_one();
}
void Pool::Stop() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cv.notify_all();
    for (auto &t : workers)
        if (t.joinable())
            t.join();
    workers.clear();
}
struct Budget::State {
    mutable std::mutex mutex;
    size_t capacity, used = 0;
    explicit State(size_t c) : capacity(c) {}
};
Budget::Budget(size_t c) : state(std::make_shared<State>(c)) {}
std::shared_ptr<void> Budget::Reserve(size_t n) {
    struct Reservation {
        std::shared_ptr<State> state;
        size_t bytes = 0;
        ~Reservation() {
            if (state) {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->used -= bytes;
            }
        }
    };
    auto lease = std::make_shared<Reservation>();
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        Require(n <= state->capacity - state->used, "FastKV: receiver byte budget exhausted");
        state->used += n;
        lease->state = state;
        lease->bytes = n;
    }
    return lease;
}
size_t Budget::Used() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->used;
}
ObjectStore::ObjectStore(size_t n, int device) : budget(n), storageDevice(device) {}
Message Snapshot(const Message &message, int storageDevice, size_t arenaBytes) {
    Require(storageDevice >= -2 && arenaBytes > 0, "FastKV: invalid snapshot options");
    Message result;
    result.metadata = message.metadata;
    result.buffers.resize(message.buffers.size());
    std::map<int, std::vector<size_t>> groups;
    for (size_t i = 0; i < message.buffers.size(); ++i) {
        const auto &buffer = message.buffers[i];
        const int device =
            storageDevice == -2 ? (buffer.Kind() == MemoryKind::Cuda ? buffer.Device() : -1) : storageDevice;
        groups[device].push_back(i);
    }
    for (const auto &group : groups) {
        const auto &indices = group.second;
        for (size_t start = 0; start < indices.size();) {
            size_t end = start, bytes = 0;
            while (end < indices.size()) {
                const size_t next = message.buffers[indices[end]].Size();
                if (end > start && (bytes >= arenaBytes || next > arenaBytes - bytes))
                    break;
                bytes = CheckedAdd(bytes, next);
                ++end;
            }
            auto arena = Buffer::Allocate(bytes, group.first);
            size_t offset = 0;
            for (size_t i = start; i < end; ++i) {
                const auto &source = message.buffers[indices[i]];
                auto slice = arena.Slice(offset, source.Size());
                slice.CopyFrom(source, 0, 0, source.Size());
                result.buffers[indices[i]] = slice.ReadOnly();
                offset += source.Size();
            }
            start = end;
        }
    }
    return result;
}
void ObjectStore::Put(const std::string &key, Message value) {
    size_t n = value.metadata.size();
    for (auto &b : value.buffers)
        n = CheckedAdd(n, b.Size());
    std::lock_guard<std::mutex> lock(mutex);
    // Publication is immutable. Mutating a caller-owned Buffer after Put is
    // prevented by taking independent copies, rather than trusting aliasing.
    Require(!objects.count(key), "FastKV: immutable object already exists");
    std::shared_ptr<void> lease;
    for (;;) {
        try {
            lease = budget.Reserve(n);
            break;
        } catch (const std::runtime_error &) {
            auto victim = objects.end();
            for (auto it = objects.begin(); it != objects.end(); ++it)
                if (it->second.value.use_count() == 1 &&
                    (victim == objects.end() || it->second.access < victim->second.access))
                    victim = it;
            if (victim == objects.end())
                throw;
            objects.erase(victim);
        }
    }
    value = Snapshot(value, storageDevice);
    for (auto &buffer : value.buffers)
        buffer.Attachment("object-credit", [lease](void *, size_t) { return lease; });
    auto result = std::shared_ptr<const Message>(new Message(std::move(value)),
                                                 [lease](const Message *m) { delete m; });
    objects.emplace(key, Entry{std::move(result), ++clock});
}
std::shared_ptr<const Message> ObjectStore::Get(const std::string &key) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = objects.find(key);
    if (it == objects.end())
        return {};
    it->second.access = ++clock;
    return it->second.value;
}
void ObjectStore::Remove(const std::string &key) {
    std::lock_guard<std::mutex> lock(mutex);
    objects.erase(key);
}
size_t ObjectStore::BytesUsed() const {
    return budget.Used();
}

static uint64_t PrefixHash(uint64_t hash, int token) {
    return (hash ^ uint32_t(token)) * 1099511628211ULL;
}
PrefixCache::PrefixCache(size_t bytes, size_t limit, int device) : store(bytes, device), maxEntries(limit) {
    Require(limit > 0, "FastKV: prefix index needs an entry bound");
}
void PrefixCache::Publish(const std::string &nameSpace, const std::vector<int> &tokens, Message value) {
    Require(!nameSpace.empty() && !tokens.empty(), "FastKV: empty prefix identity");
    Writer wrapper;
    wrapper.String(nameSpace);
    wrapper.Ints(tokens);
    wrapper.Blob(value.metadata.data(), value.metadata.size());
    value.metadata = std::move(wrapper.bytes);
    uint64_t hash = 1469598103934665603ULL;
    for (int token : tokens)
        hash = PrefixHash(hash, token);
    std::lock_guard<std::mutex> lock(mutex);
    if (entries.size() >= maxEntries) {
        auto oldest = std::min_element(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
            return a.accessed < b.accessed;
        });
        store.Remove(oldest->key);
        entries.erase(oldest);
    }
    const auto id = ++clock;
    const std::string key = "prefix:" + std::to_string(id);
    store.Put(key, std::move(value));
    entries.push_back({nameSpace, key, tokens.size(), hash, id});
}
PrefixCache::Hit PrefixCache::Find(const std::string &nameSpace, const std::vector<int> &tokens,
                                   size_t maxTokens) {
    maxTokens = std::min(tokens.size(), maxTokens);
    std::lock_guard<std::mutex> lock(mutex);
    std::map<size_t, uint64_t> hashes;
    for (const auto &entry : entries)
        if (entry.nameSpace == nameSpace && entry.tokens <= maxTokens)
            hashes[entry.tokens] = 0;
    uint64_t hash = 1469598103934665603ULL;
    size_t previous = 0;
    for (auto &entry : hashes) {
        for (; previous < entry.first; ++previous)
            hash = PrefixHash(hash, tokens[previous]);
        entry.second = hash;
    }
    Hit result;
    for (auto it = entries.begin(); it != entries.end();) {
        if (it->nameSpace != nameSpace || it->tokens > maxTokens || it->tokens <= result.tokens ||
            hashes[it->tokens] != it->hash) {
            ++it;
            continue;
        }
        auto lease = store.Get(it->key);
        if (!lease) {
            it = entries.erase(it);
            continue;
        }
        Reader reader(lease->metadata);
        const auto storedNamespace = reader.String();
        const auto storedTokens = reader.Ints();
        auto metadata = reader.Blob();
        reader.End();
        if (storedNamespace == nameSpace && storedTokens.size() == it->tokens &&
            std::equal(storedTokens.begin(), storedTokens.end(), tokens.begin())) {
            auto view = new Message{std::move(metadata), lease->buffers};
            result = {it->tokens,
                      std::shared_ptr<const Message>(view, [lease](const Message *m) { delete m; })};
            it->accessed = ++clock;
        }
        ++it;
    }
    return result;
}
size_t PrefixCache::BytesUsed() const {
    return store.BytesUsed();
}

static uint64_t Elements(const std::vector<uint64_t> &v) {
    size_t n = 1;
    for (auto d : v)
        n = CheckedMul(n, d);
    return n;
}
static void ValidateShard(const Shard &s) {
    Require(!s.shape.empty() && s.shape.size() <= 8 && s.begin.size() == s.shape.size() &&
                s.fullShape.size() == s.shape.size() && s.elementBytes,
            "FastKV: invalid shard");
    for (size_t i = 0; i < s.shape.size(); ++i)
        Require(s.begin[i] <= s.fullShape[i] && s.shape[i] <= s.fullShape[i] - s.begin[i],
                "FastKV: shard outside logical tensor");
    Require(CheckedMul(Elements(s.shape), s.elementBytes) <= s.buffer.Size(),
            "FastKV: shard allocation too small");
}
std::vector<CopyRegion> PlanCopies(const std::vector<Shard> &src, const std::vector<Shard> &dst) {
    for (auto &s : src)
        ValidateShard(s);
    for (auto &d : dst)
        ValidateShard(d);
    std::vector<CopyRegion> result;
    for (size_t j = 0; j < dst.size(); ++j) {
        const auto &d = dst[j];
        std::vector<std::pair<uint64_t, uint64_t>> covered;
        for (size_t i = 0; i < src.size(); ++i) {
            const auto &s = src[i];
            if (s.component != d.component)
                continue;
            Require(s.representation == d.representation && s.fullShape == d.fullShape &&
                        s.elementBytes == d.elementBytes,
                    "FastKV: incompatible wire representation");
            bool duplicate = false;
            for (size_t k = 0; k < i; ++k)
                if (src[k].component == s.component && src[k].begin == s.begin && src[k].shape == s.shape) {
                    Require(s.replica && src[k].replica == s.replica,
                            "FastKV: ambiguous overlapping sources");
                    duplicate = true;
                    break;
                }
            if (duplicate)
                continue;
            size_t axes = d.shape.size();
            std::vector<uint64_t> start(axes), shape(axes), ss(axes, 1), ds(axes, 1);
            bool empty = false;
            for (size_t a = 0; a < axes; ++a) {
                start[a] = std::max(s.begin[a], d.begin[a]);
                uint64_t end = std::min(s.begin[a] + s.shape[a], d.begin[a] + d.shape[a]);
                if (end <= start[a]) {
                    empty = true;
                    break;
                }
                shape[a] = end - start[a];
            }
            if (empty)
                continue;
            for (size_t a = axes - 1; a > 0; --a) {
                ss[a - 1] = CheckedMul(ss[a], s.shape[a]);
                ds[a - 1] = CheckedMul(ds[a], d.shape[a]);
            }
            const uint64_t rows = Elements(shape) / shape.back();
            Require(rows <= 10000000, "FastKV: excessive fragmented transfer; use a pack codec");
            for (uint64_t row = 0; row < rows; ++row) {
                uint64_t v = row, so = 0, dd = 0;
                for (size_t a = axes; a-- > 0;) {
                    uint64_t pos = start[a];
                    if (a + 1 < axes) {
                        pos += v % shape[a];
                        v /= shape[a];
                    }
                    so = CheckedAdd(so, CheckedMul(pos - s.begin[a], ss[a]));
                    dd = CheckedAdd(dd, CheckedMul(pos - d.begin[a], ds[a]));
                }
                so = CheckedMul(so, s.elementBytes);
                dd = CheckedMul(dd, d.elementBytes);
                uint64_t bytes = CheckedMul(shape.back(), s.elementBytes);
                covered.emplace_back(dd, dd + bytes);
                if (!result.empty() && result.back().source == i && result.back().target == j &&
                    result.back().sourceOffset + result.back().bytes == so &&
                    result.back().targetOffset + result.back().bytes == dd)
                    result.back().bytes += bytes;
                else
                    result.push_back({i, j, so, dd, bytes});
            }
        }
        std::sort(covered.begin(), covered.end());
        uint64_t end = 0;
        for (auto interval : covered) {
            Require(interval.first == end, "FastKV: missing or overlapping source coverage");
            end = interval.second;
        }
        Require(end == CheckedMul(Elements(d.shape), d.elementBytes),
                "FastKV: incomplete destination coverage");
    }
    return result;
}
} // namespace fastkv
