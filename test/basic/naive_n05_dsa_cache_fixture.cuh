#pragma once
namespace {
void Require(bool ok, const char *why) {
    if (!ok)
        throw std::runtime_error(why);
}
void Cuda(cudaError_t status) {
    Require(status == cudaSuccess, cudaGetErrorString(status));
}
void Init(Data &d, DataType type, std::vector<int> shape) {
    d.dataType = type;
    d.Resize(shape);
    d.ToDevice(CUDA, {FastllmCudaGetDevice()}, false);
    d.Allocate(false);
    d.expansionDims = d.dims;
}
void Fill(Data &d, unsigned seed, float scale = 1, float bias = 0) {
    std::vector<unsigned char> bytes(d.GetBytes());
    for (size_t i = 0; i < d.Count(0); ++i) {
        seed = seed * 1664525u + 1013904223u;
        float x = (int(seed >> 16) - 32768) / 32768.f * scale + bias;
        if (d.dataType == BFLOAT16) {
            B y = __float2bfloat16_rn(x);
            std::memcpy(bytes.data() + i * 2, &y, 2);
        } else if (d.dataType == FP8_E4M3) {
            bytes[i] = __nv_fp8_e4m3(x).__x;
        } else std::memcpy(bytes.data() + i * 4, &x, 4);
    }
    Cuda(cudaMemcpy(d.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}
void CopyDevice(Data &to, const Data &from) {
    Cuda(cudaMemcpy(to.cudaData, from.cudaData, from.GetBytes(), cudaMemcpyDeviceToDevice));
}
std::vector<unsigned char> Read(const Data &d) {
    std::vector<unsigned char> out(d.GetBytes());
    Cuda(cudaMemcpy(out.data(), d.cudaData, out.size(), cudaMemcpyDeviceToHost));
    return out;
}
void View(Data &d, const Data &source, std::vector<int> shape, size_t offset = 0) {
    d.Resize(shape);
    d.FakeFrom(source, offset);
    d.dataType = BFLOAT16;
    d.UpdateUnitSize();
}
struct Cache {
    Data key, value, kp, vp, keyView, valueView, table, indexView;
    PagedCacheManager keyPool, valuePool, indexPool;
    int capacity, stride;
    bool paged, fp8;
    Cache(int capacity, bool paged, bool fp8, int seed, int kvHeads = 1, int bits = 16)
        : capacity(capacity), stride(192 * kvHeads + (bits == 4 ? 0 : fp8 ? 72 : 128)), paged(paged), fp8(fp8) {
        const auto type = bits == 4 ? FP4_E2M1 : bits == 8 ? FP8_E4M3 : BFLOAT16;
        Init(key, type, {1, capacity, stride});
        Init(value, type, {1, capacity, kvHeads * 128});
        auto fillCache = [&](Data &d, unsigned rng) {
            if (d.dataType == BFLOAT16) { Fill(d, rng, .3f); return; }
            std::vector<uint8_t> bytes(d.GetBytes());
            for (auto &v : bytes) {
                rng = 1664525u * rng + 1013904223u;
                v = d.dataType == FP4_E2M1 ? uint8_t(rng >> 24) :
                    __nv_fp8_e4m3(float(int(rng >> 22) - 512) / 1024.f).__x;
            }
            if (d.dataType == FP4_E2M1) {
                const int cols = d.dims.back(), stride = cols / 16 * 9;
                for (int row = 0; row < capacity; ++row)
                    for (int k = 0; k < cols / 16; ++k)
                        bytes[size_t(row) * stride + cols / 2 + k] = 10 + ((k + row) % 24);
            }
            Cuda(cudaMemcpy(d.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        };
        fillCache(key, seed); fillCache(value, seed + 2);
        key.kvCacheFp8Indexer = fp8;
        if (fp8) {
            auto bytes = Read(key);
            unsigned rng = seed + 5;
            for (int row = 0; row < capacity; ++row) {
                auto dst = bytes.data() + size_t(row) * stride * 2 + 192 * 2;
                for (int d = 0; d < 128; ++d) {
                    rng = rng * 1664525u + 1013904223u;
                    __nv_fp8_e4m3 x((int(rng >> 16) - 32768) / 32768.f * 4.f);
                    dst[d] = x.__x;
                }
                float scale = .03f;
                std::memcpy(dst + 128, &scale, 4);
                std::memset(dst + 132, 0, 12);
            }
            Cuda(cudaMemcpy(key.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        }
        if (paged) {
            int pages = capacity / 128;
            Init(kp, INT32, {pages});
            Init(vp, INT32, {pages});
            std::vector<int> k(pages), v(pages);
            for (int i = 0; i < pages; ++i) {
                k[i] = pages - 1 - i;
                v[i] = (i + 3) % pages;
            }
            Cuda(cudaMemcpy(kp.cudaData, k.data(), pages * 4, cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(vp.cudaData, v.data(), pages * 4, cudaMemcpyHostToDevice));
            keyPool.Resize(key.dims);
            keyPool.FakeFrom(key, 0);
            keyPool.pageLen = 128;
            keyPool.maxPages = pages;
            valuePool.Resize(value.dims);
            valuePool.FakeFrom(value, 0);
            valuePool.pageLen = 128;
            valuePool.maxPages = pages;
            View(keyView, kp, key.dims);
            View(valueView, vp, value.dims);
            keyView.dataType = valueView.dataType = type;
            keyView.UpdateUnitSize(); valueView.UpdateUnitSize();
            if (bits == 4) {
                Init(indexPool, FP8_E4M3, {pages, 128, 1, 128});
                indexPool.pageLen = 128; indexPool.maxPages = pages;
                fillCache(indexPool, seed + 3);
                // The serving Indexer pool shares K's page IDs.
                View(indexView, kp, {1, capacity, 128});
                indexView.dataType = FP8_E4M3; indexView.UpdateUnitSize();
                indexView.isPagedKVCache = true; indexView.pageLen = 128;
                indexView.expansionDims = indexView.dims;
                indexView.pagedKVCacheData = &indexPool;
            }
            keyView.isPagedKVCache = valueView.isPagedKVCache = true;
            keyView.pageLen = valueView.pageLen = 128;
            keyView.pagedKVCacheData = &keyPool;
            valueView.pagedKVCacheData = &valuePool;
            keyView.kvCacheFp8Indexer = fp8;
            keyView.expansionDims = key.dims;
            valueView.expansionDims = value.dims;
        }
        Init(table, INT32, {bits == 4 ? 16 : 8});
        Bind(*this);
    }
    Cache(int capacity, bool paged, bool fp8, int seed, int kvHeads, DataType storage)
        : Cache(capacity, paged, fp8, seed, kvHeads, storage == FP8_E4M3 ? 8 : storage == FP4_E2M1 ? 4 : 16) {}
    void Bind(Cache &payload) {
        const void *ptrs[8] = {payload.key.cudaData, payload.value.cudaData,
                               paged ? payload.kp.cudaData : nullptr, paged ? payload.vp.cudaData : nullptr,
                               payload.indexPool.cudaData, nullptr, paged ? payload.kp.cudaData : nullptr, nullptr};
        Cuda(cudaMemcpy(table.cudaData, ptrs, table.GetBytes(), cudaMemcpyHostToDevice));
    }
    Data *K() { return paged ? &keyView : &key; }
    Data *V() { return paged ? &valueView : &value; }
};
} // namespace
