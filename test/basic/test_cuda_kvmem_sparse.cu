#include "devices/cuda/fastllm-cuda-kvmem.cuh"
// Exercise the production DSA kernels, not a new attention implementation.
#if CUDART_VERSION >= 12000
#include "naive-n05-dsa-decode.cuh"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void C(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
template<class T> struct Buffer {
    T *p = nullptr; size_t size;
    explicit Buffer(size_t n) : size(n) { C(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer &) = delete;
    void Put(const std::vector<T> &v) {
        Require(v.size() == size, "upload size");
        C(cudaMemcpy(p, v.data(), size * sizeof(T), cudaMemcpyHostToDevice));
        // Pageable H2D may return after staging. The graph uses a nonblocking
        // stream, so explicitly finish these test uploads before replay.
        C(cudaStreamSynchronize(nullptr));
    }
    std::vector<T> Get() const {
        std::vector<T> v(size); C(cudaMemcpy(v.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
};
template<class T> static std::vector<T> Read(const T *p, size_t n) {
    std::vector<T> v(n); C(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
struct Stream {
    cudaStream_t s{};
    Stream() { C(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(s); }
    void Sync() { C(cudaStreamSynchronize(s)); }
};
struct Graph {
    cudaGraph_t g{}; cudaGraphExec_t e{};
    ~Graph() { if (e) cudaGraphExecDestroy(e); if (g) cudaGraphDestroy(g); }
    template<class F> void Capture(cudaStream_t s, F body) {
        C(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)); body();
        C(cudaStreamEndCapture(s, &g)); C(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
        size_t n = 0; C(cudaGraphGetNodes(g, nullptr, &n));
        std::vector<cudaGraphNode_t> nodes(n); C(cudaGraphGetNodes(g, nodes.data(), &n));
        for (auto node : nodes) {
            cudaGraphNodeType type; C(cudaGraphNodeGetType(node, &type));
            Require(type == cudaGraphNodeTypeKernel, "cache graph contains host/memcpy/allocation node");
        }
    }
    void Run(cudaStream_t s) { C(cudaGraphLaunch(e, s)); }
};
static unsigned char Byte(int owner, int token, size_t column, int salt) {
    return (owner * 37 + token * 73 + column * 13 + salt) % 251;
}
static void Fill(KvMemSparseCache &cache, const KvMemSparseConfig &c, int salt = 0) {
    const int owners = c.pageLen ? 1 : c.maxRequests;
    const int tokens = c.pageLen ? (c.totalTokens ? c.totalTokens : c.maxRequests * c.maxTokens) : c.maxTokens;
    for (int r = 0; r < owners; ++r) {
        auto *k = (unsigned char *)cache.HostKey(r), *v = (unsigned char *)cache.HostValue(r);
        for (int token = 0; token < tokens; ++token) {
            for (size_t d = 0; d < c.keyRowBytes; ++d) k[size_t(token) * c.keyRowBytes + d] = Byte(r, token, d, salt);
            for (size_t d = 0; d < c.valueRowBytes; ++d) v[size_t(token) * c.valueRowBytes + d] = Byte(r, token, d, salt + 19);
        }
    }
}
static void Verify(KvMemSparseCache &cache, const KvMemSparseConfig &c,
                   const std::vector<int> &requests, const std::vector<int> &lengths,
                   const std::vector<int> &selected, const std::vector<int> &counts) {
    const auto mapping = Read(cache.Indices(), size_t(c.maxBatch) * c.maxSelected);
    const auto bounds = Read(cache.Lengths(), c.maxBatch);
    const auto pointers = Read(cache.CachePointers(), c.maxBatch * 2);
    Require(Read(cache.DeviceErrors(), 1)[0] == 0, "unexpected device error");
    for (int b = 0; b < c.maxBatch; ++b) {
        const int request = requests[b];
        Require(bounds[b] == (request < 0 || !lengths[b] ? 0 : c.residentTokens), "incorrect physical bound");
        std::vector<unsigned char> k, v;
        if (request >= 0) {
            k = Read((const unsigned char *)pointers[2 * b], size_t(c.residentTokens) * c.keyRowBytes);
            v = Read((const unsigned char *)pointers[2 * b + 1], size_t(c.residentTokens) * c.valueRowBytes);
        }
        for (int i = 0; i < c.maxSelected; ++i) {
            const int token = selected[size_t(b) * c.maxSelected + i];
            const int slot = mapping[size_t(b) * c.maxSelected + i];
            bool valid = request >= 0 && i < counts[b] && token >= 0 && token < lengths[b];
            if (!valid) { Require(slot == -1, "invalid TopK entry was not masked"); continue; }
            Require(slot >= 0 && slot < c.residentTokens, "missing selected GPU row");
            const auto kp = c.pageLen ? Read(cache.KeyPages(request), c.maxTokens / c.pageLen) : std::vector<int>();
            const auto vp = c.pageLen ? Read(cache.ValuePages(request), c.maxTokens / c.pageLen) : std::vector<int>();
            size_t kr = c.pageLen ? kp[token / c.pageLen] * c.pageLen + token % c.pageLen : token;
            size_t vr = c.pageLen ? vp[token / c.pageLen] * c.pageLen + token % c.pageLen : token;
            Require(!std::memcmp(k.data() + size_t(slot) * c.keyRowBytes,
                (unsigned char *)cache.HostKey(c.pageLen ? 0 : request) + kr * c.keyRowBytes, c.keyRowBytes), "key bytes differ");
            Require(!std::memcmp(v.data() + size_t(slot) * c.valueRowBytes,
                (unsigned char *)cache.HostValue(c.pageLen ? 0 : request) + vr * c.valueRowBytes, c.valueRowBytes), "value bytes differ");
        }
    }
}
static void PointerBindingTest() {
    KvMemSparseConfig c;
    c.maxRequests = 3; c.maxBatch = 4; c.maxTokens = 259;
    c.residentTokens = 16; c.maxSelected = 8;
    c.keyRowBytes = 7; c.valueRowBytes = 3;
    KvMemSparseCache cache(c); Stream stream; Fill(cache, c);
    Buffer<const void *> table(c.maxBatch * 4);
    Buffer<int> lengths(c.maxBatch);
    Graph graph;
    graph.Capture(stream.s, [&] {
        cache.ResolveFromPointers(table.p, 4, lengths.p, nullptr, nullptr, c.maxBatch, stream.s);
    });
    for (int round = 0; round < 5; ++round) {
        std::vector<const void *> pointers(c.maxBatch * 4, nullptr);
        std::vector<int> owners(c.maxBatch, -1), live(c.maxBatch, 0);
        std::vector<int> selected(c.maxBatch * c.maxSelected), counts(c.maxBatch, c.maxSelected);
        for (int b = 0; b < c.maxBatch; ++b) {
            for (int k = 0; k < c.maxSelected; ++k) selected[b * c.maxSelected + k] = k;
            if (b == round % c.maxBatch) continue;
            const int owner = (b - (b > round % c.maxBatch) + round) % c.maxRequests;
            owners[b] = owner; live[b] = 2 + b + round;
            pointers[b * 4] = cache.DeviceHostKey(owner);
            pointers[b * 4 + 1] = cache.DeviceHostValue(owner);
        }
        table.Put(pointers); lengths.Put(live);
        graph.Run(stream.s); stream.Sync();
        Verify(cache, c, owners, live, selected, counts);
    }
    std::vector<const void *> invalid(c.maxBatch * 4, nullptr);
    invalid[0] = static_cast<unsigned char *>(cache.DeviceHostKey(0)) + 1;
    table.Put(invalid); lengths.Put({1, 0, 0, 0});
    graph.Run(stream.s); stream.Sync();
    Require(Read(cache.DeviceErrors(), 1)[0] == 1, "misaligned owner address accepted");
    Require(Read(cache.Lengths(), c.maxBatch)[0] == 0, "invalid owner produced attention rows");
    std::puts("PASS graph pointer binding, implicit selection and request reorder");
}

static void PagedSharingTest(int totalTokens = 0) {
    KvMemSparseConfig c;
    c.maxRequests = 2; c.maxBatch = 3; c.maxTokens = 320; c.pageLen = 64;
    c.totalTokens = totalTokens;
    c.residentTokens = 16; c.maxSelected = 8; c.keyRowBytes = 16; c.valueRowBytes = 7;
    KvMemSparseCache cache(c); Stream stream; Fill(cache, c);
    Require(cache.HostBytes() == size_t(totalTokens ? totalTokens : 640) *
        (c.keyRowBytes + c.valueRowBytes), "physical pool used per-request capacity");
    std::vector<int> kp{5, 2, 7, 0, 4, 5, 2, 1, 6, 3}, vp{4, 7, 1, 0, 5, 4, 7, 2, 6, 3};
    for (int owner = 0; owner < 2; ++owner) {
        Require(uintptr_t(cache.KeyPages(owner)) % 16 == 0 && uintptr_t(cache.ValuePages(owner)) % 16 == 0,
            "odd page count broke table alignment");
        C(cudaMemcpy(cache.KeyPages(owner), kp.data() + owner * 5, 5 * sizeof(int), cudaMemcpyHostToDevice));
        C(cudaMemcpy(cache.ValuePages(owner), vp.data() + owner * 5, 5 * sizeof(int), cudaMemcpyHostToDevice));
    }
    C(cudaDeviceSynchronize());
    Buffer<const void *> table(3);
    Buffer<int> lengths(3), topk(24);
    std::vector<int> selected{0, 63, 64, 127, 128, 191, 200, 200};
    std::vector<int> ht;
    for (int i = 0; i < 3; ++i) ht.insert(ht.end(), selected.begin(), selected.end());
    topk.Put(ht);
    Graph graph;
    graph.Capture(stream.s, [&] { cache.ResolveFromPointers(table.p, 1, lengths.p, topk.p, nullptr, 3, stream.s); });
    for (int round = 0; round < 4; ++round) {
        int first = round % 2;
        table.Put({cache.KeyPages(first), cache.KeyPages(1 - first), nullptr});
        lengths.Put({256, 192, 0}); graph.Run(stream.s); stream.Sync();
        Verify(cache, c, {first, 1 - first, -1}, {256, 192, 0}, ht, {8, 8, 8});
        // Retire one hot owner while the other still refers to the shared prefix.
        cache.Reset(first, stream.s); stream.Sync();
    }
    // Append to one private suffix while another owner retains the prefix.
    Buffer<int> owner(1), position(1);
    Buffer<unsigned char> writeK(c.keyRowBytes), writeV(c.valueRowBytes);
    owner.Put({1}); position.Put({200});
    writeK.Put(std::vector<unsigned char>(c.keyRowBytes, 231));
    writeV.Put(std::vector<unsigned char>(c.valueRowBytes, 123));
    const auto *hostK = (const unsigned char *)cache.HostKey(0);
    const auto *hostV = (const unsigned char *)cache.HostValue(0);
    std::vector<unsigned char> prefixK(hostK + kp[0] * 64 * c.keyRowBytes,
        hostK + (kp[0] + 1) * 64 * c.keyRowBytes);
    std::vector<unsigned char> prefixV(hostV + vp[0] * 64 * c.valueRowBytes,
        hostV + (vp[0] + 1) * 64 * c.valueRowBytes);
    Graph append;
    append.Capture(stream.s, [&] { cache.Write(owner.p, position.p, writeK.p, writeV.p, 1, stream.s); });
    append.Run(stream.s); stream.Sync();
    Require(!std::memcmp(prefixK.data(), hostK + kp[0] * 64 * c.keyRowBytes, prefixK.size()), "append changed shared K prefix");
    Require(!std::memcmp(prefixV.data(), hostV + vp[0] * 64 * c.valueRowBytes, prefixV.size()), "append changed shared V prefix");
    for (size_t d = 0; d < c.keyRowBytes; ++d)
        Require(hostK[(kp[8] * 64 + 8) * c.keyRowBytes + d] == 231, "paged append K location");
    for (size_t d = 0; d < c.valueRowBytes; ++d)
        Require(hostV[(vp[8] * 64 + 8) * c.valueRowBytes + d] == 123, "paged append V location");
    table.Put({cache.KeyPages(0), cache.KeyPages(1), nullptr}); lengths.Put({256, 256, 0});
    graph.Run(stream.s); stream.Sync();
    Verify(cache, c, {0, 1, -1}, {256, 256, 0}, ht, {8, 8, 8});
    Buffer<unsigned char> key(256 * c.keyRowBytes), value(256 * c.valueRowBytes);
    cache.Gather(cache.KeyPages(1), cache.ValuePages(1), 256, key.p, value.p, stream.s); stream.Sync();
    auto k = key.Get(), v = value.Get();
    for (int t = 0; t < 256; ++t) {
        size_t kr = kp[5 + t / 64] * 64 + t % 64, vr = vp[5 + t / 64] * 64 + t % 64;
        Require(!std::memcmp(k.data() + t * c.keyRowBytes, (char *)cache.HostKey(0) + kr * c.keyRowBytes, c.keyRowBytes), "paged prefill K");
        Require(!std::memcmp(v.data() + t * c.valueRowBytes, (char *)cache.HostValue(0) + vr * c.valueRowBytes, c.valueRowBytes), "paged prefill V");
    }
    std::puts("PASS shared paged prefixes, private hot buffers, reordered graph and exact prefill gather");
}

static void StorageTest(int requests, int padded, int tokens, int selected, int resident, size_t kb, size_t vb) {
    KvMemSparseConfig c;
    c.maxRequests = requests; c.maxBatch = padded; c.maxTokens = tokens;
    c.maxSelected = selected; c.residentTokens = resident; c.keyRowBytes = kb; c.valueRowBytes = vb;
    KvMemSparseCache cache(c); Stream stream;
    Fill(cache, c);
    Buffer<int> ids(padded), lengths(padded), counts(padded), topk(size_t(padded) * selected), positions(padded);
    Buffer<unsigned char> writeK(size_t(padded) * kb), writeV(size_t(padded) * vb);
    std::vector<int> hi(padded), hl(padded), hc(padded), ht(size_t(padded) * selected), hp(padded);
    std::vector<unsigned char> wk(size_t(padded) * kb), wv(size_t(padded) * vb);
    for (size_t i = 0; i < wk.size(); ++i) wk[i] = (i * 17 + 21) % 253;
    for (size_t i = 0; i < wv.size(); ++i) wv[i] = (i * 29 + 11) % 247;
    writeK.Put(wk); writeV.Put(wv);
    auto resolve = [&] { cache.Resolve(ids.p, lengths.p, topk.p, counts.p, padded, stream.s); };
    Graph graph, append;
    graph.Capture(stream.s, resolve);
    append.Capture(stream.s, [&] {
        cache.Write(ids.p, positions.p, writeK.p, writeV.p, padded, stream.s); resolve();
    });
    const void *address = cache.Indices(); const size_t memory = cache.DeviceBytes();
    unsigned long long totalMiss = 0, totalEvicted = 0;
    std::mt19937 rng(197);
    for (int round = 0; round < 14; ++round) {
        std::iota(hi.begin(), hi.end(), 0);
        std::shuffle(hi.begin(), hi.begin() + requests, rng);
        for (int b = 0; b < padded; ++b) {
            if (b >= requests || (round == 5 && b % 3 == 0)) hi[b] = -1;
            hl[b] = round == 6 ? std::max(0, tokens / 2 - b * 13) : tokens;
            hc[b] = round == 7 ? b % (selected + 1) : selected;
            hp[b] = std::max(0, hl[b] - 1);
            for (int i = 0; i < selected; ++i) {
                int token = (i + round * selected + b * 37) % std::max(1, hl[b]);
                if (i % 31 == 9) token = -1;
                if (i % 31 == 10) token = hl[b];
                if (i % 31 == 12) token = ht[size_t(b) * selected + i - 1];
                ht[size_t(b) * selected + i] = token;
            }
            ht[size_t(b) * selected] = hp[b]; // appended and hot overwrite cases
        }
        ids.Put(hi); lengths.Put(hl); counts.Put(hc); topk.Put(ht); positions.Put(hp);
        if (round == 10) {
            for (int r = 0; r < requests; ++r) cache.Reset(r, stream.s);
            stream.Sync(); Fill(cache, c, 101);
        }
        if (round % 3 == 0) {
            for (size_t i = 0; i < wk.size(); ++i) wk[i] = (i * 17 + round * 23 + 21) % 253;
            for (size_t i = 0; i < wv.size(); ++i) wv[i] = (i * 29 + round * 31 + 11) % 247;
            writeK.Put(wk); writeV.Put(wv);
            append.Run(stream.s);
        }
        else if (round % 3 == 1) graph.Run(stream.s);
        else resolve();
        stream.Sync(); Verify(cache, c, hi, hl, ht, hc);
        if (round % 3 == 0) {
            for (int b = 0; b < padded; ++b) if (hi[b] >= 0) {
                Require(!std::memcmp((unsigned char *)cache.HostKey(hi[b]) + size_t(hp[b]) * kb,
                                    wk.data() + size_t(b) * kb, kb), "DDR key append missing");
                Require(!std::memcmp((unsigned char *)cache.HostValue(hi[b]) + size_t(hp[b]) * vb,
                                    wv.data() + size_t(b) * vb, vb), "DDR value append missing");
            }
        }
        auto before = Read(cache.DeviceStats(), requests);
        // Identical requests must be all hits, including duplicates. The same
        // captured executable is reused after changing owners and lengths.
        graph.Run(stream.s); stream.Sync(); Verify(cache, c, hi, hl, ht, hc);
        auto after = Read(cache.DeviceStats(), requests);
        for (int r = 0; r < requests; ++r) {
            Require(before[r].misses == after[r].misses && before[r].copiedBytes == after[r].copiedBytes,
                    "all-hit replay copied DDR rows");
            Require(after[r].copiedBytes == after[r].misses * (kb + vb), "transfer byte accounting");
            totalMiss += after[r].misses; totalEvicted += after[r].evictions;
        }
        Require(cache.Indices() == address && cache.DeviceBytes() == memory, "graph workspace grew or moved");
    }
    Require(totalMiss > 0 && totalEvicted > 0, "eviction path was not exercised");
    std::printf("PASS storage requests=%d padded=%d context=%d topk=%d resident=%d row=%zu+%zu host=%zu device=%zu\n",
                requests, padded, tokens, selected, resident, kb, vb, cache.HostBytes(), cache.DeviceBytes());
}

// Use exactly the same production score/softmax/PV kernels and rounding for
// full GPU storage and remapped hot storage. Only addresses/indices differ.
#if CUDART_VERSION >= 12000
static void Attention(const __nv_bfloat16 *q, FastllmNaiveDSABatch rows, int batch,
                      int count, float *scores, __nv_bfloat16 *out, cudaStream_t stream) {
    constexpr int heads = 4, dim = 192, vd = 128;
    naive_dsa_decode::Scores<dim><<<dim3(heads, (count + 63) / 64, batch), 256, 0, stream>>>(
        q, rows, scores, heads, 1, dim, count);
    naive_dsa_decode::Softmax<<<batch * heads, 256, 0, stream>>>(scores, nullptr, rows, heads, count);
    naive_dsa_decode::Values<vd><<<dim3(heads, vd / 8, batch), 128, 0, stream>>>(
        scores, rows, out, heads, 1, count);
    C(cudaGetLastError());
}
static void AttentionTest(bool fp8, int batch, int active = 0) {
    if (!active) active = batch;
    constexpr int tokens = 8193, count = 2048, resident = 4096, heads = 4, dim = 192, vd = 128;
    int unit = fp8 ? 1 : 2;
    KvMemSparseConfig c;
    c.maxRequests = active; c.maxBatch = batch; c.maxTokens = tokens; c.maxSelected = count;
    c.residentTokens = resident; c.keyRowBytes = dim * unit; c.valueRowBytes = vd * unit;
    KvMemSparseCache cache(c); Stream stream;
    Buffer<unsigned char> fullK(size_t(batch) * tokens * dim * unit), fullV(size_t(batch) * tokens * vd * unit);
    std::vector<unsigned char> hk(fullK.size), hv(fullV.size);
    auto generate = [&](std::vector<unsigned char> &v) {
        for (size_t i = 0; i < v.size() / unit; ++i) {
            float x = float(int((i * 17 + 3) % 251) - 125) / 256;
            if (fp8) reinterpret_cast<__nv_fp8_e4m3 *>(v.data())[i] = __nv_fp8_e4m3(x);
            else reinterpret_cast<__nv_bfloat16 *>(v.data())[i] = __float2bfloat16(x);
        }
    };
    generate(hk); generate(hv); fullK.Put(hk); fullV.Put(hv);
    for (int r = 0; r < active; ++r) {
        std::memcpy(cache.HostKey(r), hk.data() + size_t(r) * tokens * dim * unit, size_t(tokens) * dim * unit);
        std::memcpy(cache.HostValue(r), hv.data() + size_t(r) * tokens * vd * unit, size_t(tokens) * vd * unit);
    }
    Buffer<__nv_bfloat16> q(batch * heads * dim), expected(batch * heads * vd), actual(expected.size);
    Buffer<float> scores(batch * heads * count);
    std::vector<__nv_bfloat16> hq(q.size);
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = __float2bfloat16(float(int(i % 31) - 15) / 64);
    q.Put(hq);
    Buffer<int> requests(batch), lengths(batch), topk(batch * count);
    Buffer<const void *> fullPointers(batch * 2);
    std::vector<int> hi(batch), hl(batch), ht(batch * count);
    std::iota(hi.begin(), hi.end(), 0);
    std::fill(hi.begin() + active, hi.end(), -1); requests.Put(hi);
    std::vector<FastllmNaiveDSABatch> dense((batch + 31) / 32), hot(dense.size());
    for (int b = 0; b < batch; ++b) {
        auto &r = dense[b / 32].rows[b % 32];
        r.key = fullK.p + size_t(b) * tokens * dim * unit;
        r.value = fullV.p + size_t(b) * tokens * vd * unit;
        r.keyStride = dim; r.capacity = tokens; r.liveKeys = lengths.p + b;
        r.indices = topk.p + b * count; r.selectedCount = count; r.fp8 = fp8;
        r.cache = fullPointers.p + 2 * b;
        auto &h = hot[b / 32].rows[b % 32]; h = r;
        h.cache = cache.CachePointers() + b * 2; h.capacity = resident;
        h.indices = cache.Indices() + b * count; h.liveKeys = cache.Lengths() + b;
    }
    auto attention = [&](const std::vector<FastllmNaiveDSABatch> &rows, __nv_bfloat16 *out) {
        for (int first = 0; first < batch; first += 32)
            Attention(q.p + size_t(first) * heads * dim, rows[first / 32], std::min(32, batch - first), count,
                      scores.p + size_t(first) * heads * count, out + size_t(first) * heads * vd, stream.s);
    };
    auto body = [&] {
        cache.Resolve(requests.p, lengths.p, topk.p, nullptr, batch, stream.s);
        attention(hot, actual.p);
    };
    Graph graph; graph.Capture(stream.s, body);
    for (int round = 0; round < 7; ++round) {
        std::rotate(hi.begin(), hi.begin() + 1, hi.begin() + active);
        requests.Put(hi);
        std::vector<const void *> pointers(batch * 2);
        for (int b = 0; b < batch; ++b) {
            pointers[2 * b] = fullK.p + size_t(std::max(0, hi[b])) * tokens * dim * unit;
            pointers[2 * b + 1] = fullV.p + size_t(std::max(0, hi[b])) * tokens * vd * unit;
            hl[b] = round == 5 || hi[b] < 0 ? 0 : std::max(1, tokens - round * 311 - b * 17);
            for (int i = 0; i < count; ++i)
                ht[b * count + i] = i % 29 == 0 ? -1 : (i * 37 + round * 997 + b) % std::max(1, hl[b]);
        }
        lengths.Put(hl); topk.Put(ht); fullPointers.Put(pointers);
        graph.Run(stream.s);
        attention(dense, expected.p);
        stream.Sync();
        auto a = actual.Get(), e = expected.Get();
        Require(!std::memcmp(a.data(), e.data(), a.size() * sizeof(a[0])), "DSA output differs from full GPU reference");
        for (auto x : a) Require(std::isfinite(__bfloat162float(x)), "nonfinite DSA output");
    }
    std::printf("PASS production DSA %s batch=%d active=%d graph output bitwise equal\n", fp8 ? "FP8" : "BF16", batch, active);
}
#endif

static void InvalidInputs() {
    KvMemSparseConfig c; c.keyRowBytes = 16; c.valueRowBytes = 32;
    auto rejected = [&](KvMemSparseConfig bad) {
        bool ok = false; try { KvMemSparseCache cache(bad); } catch (const std::invalid_argument &) { ok = true; }
        Require(ok, "invalid configuration accepted");
    };
    auto bad = c; bad.residentTokens = bad.maxSelected - 1; rejected(bad);
    bad = c; bad.hostBudgetBytes = 1; rejected(bad);
    bad = c; bad.keyRowBytes = size_t(-1); rejected(bad);
    KvMemSparseCache cache(c); Stream stream;
    Buffer<int> ids(1), lengths(1), counts(1), selected(c.maxSelected), pos(1);
    Buffer<unsigned char> key(c.keyRowBytes), value(c.valueRowBytes);
    selected.Put(std::vector<int>(c.maxSelected, -1)); ids.Put({0}); lengths.Put({c.maxTokens + 1}); counts.Put({1});
    cache.Resolve(ids.p, lengths.p, selected.p, counts.p, 1, stream.s); stream.Sync();
    Require(Read(cache.DeviceErrors(), 1)[0] == 2 && Read(cache.Lengths(), 1)[0] == 0, "invalid length handling");
    cache.ClearErrors(stream.s); ids.Put({1}); lengths.Put({0});
    cache.Resolve(ids.p, lengths.p, selected.p, counts.p, 1, stream.s); stream.Sync();
    Require(Read(cache.DeviceErrors(), 1)[0] == 1, "invalid owner handling");
    cache.ClearErrors(stream.s); ids.Put({0}); pos.Put({c.maxTokens});
    cache.Write(ids.p, pos.p, key.p, value.p, 1, stream.s); stream.Sync();
    Require(Read(cache.DeviceErrors(), 1)[0] == 4, "invalid append handling");
    std::puts("PASS invalid configuration and device metadata guards");
}

static void PackedAppendTest() {
    KvMemSparseConfig c;
    c.maxRequests = c.maxBatch = 2; c.maxTokens = 8193;
    c.keyRowBytes = 192; c.valueRowBytes = 128;
    KvMemSparseCache cache(c); Stream stream;
    const int rows = c.maxTokens * 2;
    Buffer<int> owners(rows), positions(rows), requests(2), lengths(2), indices(2 * c.maxSelected);
    Buffer<unsigned char> keys(size_t(rows) * c.keyRowBytes), values(size_t(rows) * c.valueRowBytes);
    std::vector<int> ho(rows), hp(rows), hi(indices.size);
    for (int i = 0; i < rows; ++i) { ho[i] = i % 2; hp[i] = i / 2; }
    for (int i = 0; i < 2 * c.maxSelected; ++i) hi[i] = (i * 4) % c.maxTokens;
    owners.Put(ho); positions.Put(hp); requests.Put({0, 1}); lengths.Put({c.maxTokens, c.maxTokens}); indices.Put(hi);
    Graph graph; graph.Capture(stream.s, [&] {
        cache.Write(owners.p, positions.p, keys.p, values.p, rows, stream.s);
        cache.Resolve(requests.p, lengths.p, indices.p, nullptr, 2, stream.s);
    });
    for (int round = 0; round < 3; ++round) {
        std::vector<unsigned char> hk(keys.size), hv(values.size);
        for (size_t i = 0; i < hk.size(); ++i) hk[i] = (i * 13 + round * 19) % 251;
        for (size_t i = 0; i < hv.size(); ++i) hv[i] = (i * 7 + round * 23) % 251;
        keys.Put(hk); values.Put(hv); graph.Run(stream.s); stream.Sync();
        for (int row = 0; row < rows; ++row) {
            Require(!std::memcmp((unsigned char *)cache.HostKey(ho[row]) + size_t(hp[row]) * c.keyRowBytes,
                                hk.data() + size_t(row) * c.keyRowBytes, c.keyRowBytes), "packed DDR key write");
            Require(!std::memcmp((unsigned char *)cache.HostValue(ho[row]) + size_t(hp[row]) * c.valueRowBytes,
                                hv.data() + size_t(row) * c.valueRowBytes, c.valueRowBytes), "packed DDR value write");
        }
        Verify(cache, c, {0, 1}, {c.maxTokens, c.maxTokens}, hi, {c.maxSelected, c.maxSelected});
    }
    std::puts("PASS packed 16386-row graph append and hot overwrites");
}

static void Benchmark(int batch) {
    KvMemSparseConfig c;
    c.maxRequests = batch; c.maxBatch = batch <= 32 ? batch : (batch + 7) / 8 * 8; c.maxTokens = 100000;
    c.maxSelected = 2048; c.residentTokens = 4096;
    c.keyRowBytes = 384; c.valueRowBytes = 256; c.hostBudgetBytes = size_t(32) << 30;
    KvMemSparseCache cache(c); Stream stream;
    for (int b = 0; b < batch; ++b) {
        std::memset(cache.HostKey(b), 0x3c, size_t(c.maxTokens) * c.keyRowBytes);
        std::memset(cache.HostValue(b), 0x38, size_t(c.maxTokens) * c.valueRowBytes);
    }
    Buffer<int> owners(c.maxBatch), lengths(c.maxBatch), indices(size_t(c.maxBatch) * c.maxSelected);
    std::vector<int> ho(c.maxBatch, -1), hl(c.maxBatch, c.maxTokens), hi(indices.size, -1);
    for (int b = 0; b < batch; ++b) ho[b] = b;
    owners.Put(ho); lengths.Put(hl);
    Graph graph; graph.Capture(stream.s, [&] {
        cache.Resolve(owners.p, lengths.p, indices.p, nullptr, c.maxBatch, stream.s);
    });
    cudaEvent_t begin, end; C(cudaEventCreate(&begin)); C(cudaEventCreate(&end));
    for (int hitPercent : {100, 90, 0}) {
        for (int b = 0; b < batch; ++b) cache.Reset(b, stream.s);
        stream.Sync();
        for (int b = 0; b < batch; ++b) for (int i = 0; i < c.maxSelected; ++i)
            hi[size_t(b) * c.maxSelected + i] = int(size_t(i) * 7919 % c.maxTokens);
        indices.Put(hi); graph.Run(stream.s); stream.Sync();
        auto before = Read(cache.DeviceStats(), batch);
        std::vector<float> times;
        const int common = c.maxSelected * hitPercent / 100;
        for (int step = 1; step <= 20; ++step) {
            for (int b = 0; b < batch; ++b) for (int i = 0; i < c.maxSelected; ++i) {
                // A bijection spreads reads across the whole context instead
                // of measuring only a contiguous DDR slice.
                int ordinal = i < common ? i : (step + 1) * c.maxSelected + i;
                hi[size_t(b) * c.maxSelected + i] = int(size_t(ordinal) * 7919 % c.maxTokens);
            }
            indices.Put(hi);
            C(cudaEventRecord(begin, stream.s)); graph.Run(stream.s); C(cudaEventRecord(end, stream.s));
            C(cudaEventSynchronize(end)); float ms; C(cudaEventElapsedTime(&ms, begin, end)); times.push_back(ms);
        }
        auto after = Read(cache.DeviceStats(), batch);
        unsigned long long bytes = 0, hits = 0, misses = 0;
        for (int b = 0; b < batch; ++b) {
            bytes += after[b].copiedBytes - before[b].copiedBytes;
            hits += after[b].hits - before[b].hits; misses += after[b].misses - before[b].misses;
        }
        Require(Read(cache.DeviceErrors(), 1)[0] == 0, "benchmark metadata error");
        Require(misses == size_t(20) * batch * (c.maxSelected - common), "benchmark unexpected miss count");
        std::sort(times.begin(), times.end()); double totalMs = std::accumulate(times.begin(), times.end(), 0.0);
        std::printf("BENCH batch=%d graph_batch=%d context=%d hit_pct=%.4f median_ms=%.6f mean_ms=%.6f copied_bytes_per_step=%llu transfer_GBps=%.4f host_bytes=%zu device_bytes=%zu\n",
                    batch, c.maxBatch, c.maxTokens, 100. * hits / (hits + misses), times[10], totalMs / times.size(),
                    bytes / 20, bytes / totalMs / 1e6, cache.HostBytes(), cache.DeviceBytes());
    }
    C(cudaEventDestroy(begin)); C(cudaEventDestroy(end));
}

int main(int argc, char **argv) {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || !n) { std::puts("FASTLLM_TEST_SKIP_NO_DEVICE"); return 77; }
        setvbuf(stdout, nullptr, _IONBF, 0);
        if (argc == 2 && std::string(argv[1]) == "--benchmark") {
            for (int batch : {1, 32, 220}) Benchmark(batch);
            return 0;
        }
        bool quick = argc == 2 && std::string(argv[1]) == "--quick";
        Require(argc == 1 || quick, "unknown argument");
        InvalidInputs();
        PointerBindingTest();
        PagedSharingTest();
        PagedSharingTest(512); // Two 320-token histories share a smaller physical pool.
        PackedAppendTest();
        StorageTest(1, 1, 257, 1, 1, 3, 7);
        StorageTest(3, 4, 4099, 33, 65, 385, 259);
        StorageTest(1, 1, 200003, 2048, 4096, 384, 256);
        StorageTest(32, 32, 4099, 257, 512, 192, 128);
        StorageTest(220, 224, 4099, 64, 128, 64, 32);
#if CUDART_VERSION >= 12000
        for (bool fp8 : {false, true}) {
            for (int b : {1, 3, 32}) AttentionTest(fp8, b);
            if (!quick) AttentionTest(fp8, 224, 220);
        }
#else
        std::puts("SKIP DSA numerical subtest: requires CUDA 12+; storage tests completed");
#endif
        std::puts("PASS KVMem sparse storage, graph replay and request lifecycle");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
