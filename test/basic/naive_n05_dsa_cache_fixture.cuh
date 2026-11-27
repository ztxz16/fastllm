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
    Data key, value, kp, vp, keyView, valueView, table;
    PagedCacheManager keyPool, valuePool;
    int capacity, stride;
    bool paged, fp8;
    Cache(int capacity, bool paged, bool fp8, int seed, int kvHeads = 1, DataType storage = BFLOAT16)
        : capacity(capacity), stride(192 * kvHeads + (fp8 ? 72 : 128)), paged(paged), fp8(fp8) {
        Init(key, storage, {1, capacity, stride});
        Init(value, storage, {1, capacity, kvHeads * 128});
        Fill(key, seed, .3f);
        Fill(value, seed + 2, .3f);
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
            keyView.dataType = valueView.dataType = storage;
            keyView.UpdateUnitSize(); valueView.UpdateUnitSize();
            keyView.isPagedKVCache = valueView.isPagedKVCache = true;
            keyView.pageLen = valueView.pageLen = 128;
            keyView.pagedKVCacheData = &keyPool;
            valueView.pagedKVCacheData = &valuePool;
            keyView.kvCacheFp8Indexer = fp8;
            keyView.expansionDims = key.dims;
            valueView.expansionDims = value.dims;
        }
        Init(table, INT32, {8});
        Bind(*this);
    }
    void Bind(Cache &payload) {
        const void *ptrs[4] = {payload.key.cudaData, payload.value.cudaData,
                               paged ? payload.kp.cudaData : nullptr, paged ? payload.vp.cudaData : nullptr};
        Cuda(cudaMemcpy(table.cudaData, ptrs, sizeof(ptrs), cudaMemcpyHostToDevice));
    }
    Data *K() { return paged ? &keyView : &key; }
    Data *V() { return paged ? &valueView : &value; }
};
} // namespace
