#pragma once
bool FastllmCudaNaiveDSALayerPrepare(const FastllmNaiveDSALayerInput &x, int device,
                                     FastllmNaiveDSALayerArgs &a) {
    using namespace fastllm;
    using B = __nv_bfloat16;
    using L = FastllmNaiveDSALayerLayout;
    const int rows = int(x.caches.size());
    if (rows < 1 || rows > 16 || x.requests.empty() || !x.fp8Query || x.capacity < 1 || x.capacity > 65536 ||
        !x.scratch || !std::isfinite(x.eps) || x.eps <= 0 || !std::isfinite(x.theta) || x.theta <= 0 ||
        !std::isfinite(x.valueScale))
        return false;
    auto tensor = [&](const Data *d, int type, size_t count, size_t align = 16) {
        if (!d || d->dataType != type || d->dataDevice != CUDA || !d->cudaData ||
            uintptr_t(d->cudaData) % align || d->dataDeviceIds != std::vector<int>{device} ||
            d->dims.empty() || d->strides.size() != d->dims.size())
            return false;
        size_t stride = 1;
        for (int i = d->dims.size() - 1; i >= 0; --i) {
            if (d->dims[i] <= 0 || d->strides[i] != stride || stride > count / size_t(d->dims[i]))
                return false;
            stride *= d->dims[i];
        }
        return stride == count;
    };
    const size_t elements = size_t(rows) * 4096;
    if (!tensor(x.hidden, BFLOAT16, elements) || (x.branch && !tensor(x.branch, BFLOAT16, elements)) ||
        (x.feature && !tensor(x.feature, BFLOAT16, elements)) || !tensor(x.inputNorm, FLOAT32, 4096) ||
        !tensor(x.postNorm, FLOAT32, 4096) || !tensor(x.qkvWeight, BFLOAT16, 1856 * 4096) ||
        (x.qkvBias && !x.qkvBias->dims.empty()) || !tensor(x.indexKeyWeight, BFLOAT16, 128 * 4096) ||
        !tensor(x.indexQueryWeight, BFLOAT16, 2048 * 4096) || !tensor(x.indexWeight, BFLOAT16, 16 * 4096) ||
        !tensor(x.indexNorm, FLOAT32, 128) || !tensor(x.indexBias, FLOAT32, 128) ||
        !tensor(x.outputWeight, BFLOAT16, 4096 * 1024) ||
        (x.sink && !x.sink->dims.empty() && !tensor(x.sink, FLOAT32, 8, 4)) ||
        !tensor(x.positions, FLOAT32, rows, 4) || !tensor(x.live, INT32, rows, 4))
        return false;
    if (x.hidden->dims.back() != 4096 || x.qkvWeight->dims != std::vector<int>({1856, 4096}) ||
        x.indexKeyWeight->dims != std::vector<int>({128, 4096}) ||
        x.indexQueryWeight->dims != std::vector<int>({2048, 4096}) ||
        x.indexWeight->dims != std::vector<int>({16, 4096}) ||
        x.outputWeight->dims != std::vector<int>({4096, 1024}))
        return false;
    int sum = 0;
    for (int count : x.requests) {
        if (count < 1 || count > 8)
            return false;
        sum += count;
    }
    if (sum != rows)
        return false;
    a = {};
    bool fp8Key = x.caches.front().key && x.caches.front().key->kvCacheFp8Indexer;
    int stride = fp8Key ? 264 : 320;
    std::vector<const Data *> sources{x.hidden,      x.branch,    x.feature,        x.inputNorm,
                                      x.postNorm,    x.qkvWeight, x.indexKeyWeight, x.indexQueryWeight,
                                      x.indexWeight, x.indexNorm, x.indexBias,      x.outputWeight,
                                      x.sink,        x.positions, x.live,           x.routerWeight};
    for (int row = 0; row < rows; ++row) {
        const auto &c = x.caches[row];
        if (!c.key || !c.value || c.indexCache || c.indexerContextRank || c.verifyOffset >= 0 ||
            c.cacheOffset >= 0 || (c.cache && uintptr_t(c.cache) % alignof(void *)))
            return false;
        const Data &k = *c.key, &v = *c.value;
        if (k.dataType != BFLOAT16 || v.dataType != BFLOAT16 || k.dataDevice != CUDA ||
            v.dataDevice != CUDA || k.dataDeviceIds != std::vector<int>{device} ||
            v.dataDeviceIds != std::vector<int>{device} || k.dims.size() != 3 || v.dims.size() != 3 ||
            k.dims[0] != 1 || v.dims[0] != 1 || k.dims[2] != stride || v.dims[2] != 128 ||
            k.kvCacheFp8Indexer != fp8Key || k.isPagedKVCache != v.isPagedKVCache || !k.cudaData ||
            !v.cudaData ||
            (k.isPagedKVCache && (!k.pagedKVCacheData || !v.pagedKVCacheData || k.pageLen <= 0 ||
                                  k.pageLen % 8 || k.pageLen != v.pageLen ||
                                  k.pagedKVCacheData->maxPages <= 0 || v.pagedKVCacheData->maxPages <= 0)))
            return false;
        FastllmNaiveDSARow r;
        r.key = k.isPagedKVCache ? k.pagedKVCacheData->cudaData : k.cudaData;
        r.value = v.isPagedKVCache ? v.pagedKVCacheData->cudaData : v.cudaData;
        if (!r.key || !r.value || uintptr_t(r.key) % 16 || uintptr_t(r.value) % 16)
            return false;
        if (k.isPagedKVCache) {
            r.keyPages = (const int *)k.cudaData;
            r.valuePages = (const int *)v.cudaData;
            r.pageLen = k.pageLen;
            r.poolTokens = k.pagedKVCacheData->maxPages * k.pageLen;
            sources.push_back(k.pagedKVCacheData);
            sources.push_back(v.pagedKVCacheData);
        } else if (k.expansionSize * k.unitSize < size_t(x.capacity) * stride * 2 ||
                   v.expansionSize * v.unitSize < size_t(x.capacity) * 128 * 2)
            return false;
        r.cache = c.cache;
        r.capacity = x.capacity;
        r.keyStride = stride;
        r.liveKeys = (const int *)x.live->cudaData + row;
        r.selectedCount = 2048;
        a.bindings[row] = r;
        sources.push_back(&k);
        sources.push_back(&v);
    }
    int offset = 0;
    for (int count : x.requests) {
        const auto &first = a.bindings[offset];
        for (int i = offset; i < offset + count; ++i) {
            const auto &r = a.bindings[i];
            if (r.key != first.key || r.value != first.value || r.cache != first.cache ||
                r.keyPages != first.keyPages || r.valuePages != first.valuePages)
                return false;
        }
        // Each live request must own independent storage/binding; duplicate requests could race their append.
        for (int i = 0; i < offset; ++i)
            if (first.key == a.bindings[i].key || first.value == a.bindings[i].value ||
                (first.cache && first.cache == a.bindings[i].cache)) {
                // A paged pool is shared, while its page table/binding is per request.
                if (!first.pageLen || first.keyPages == a.bindings[i].keyPages ||
                    first.valuePages == a.bindings[i].valuePages ||
                    (first.cache && first.cache == a.bindings[i].cache))
                    return false;
            }
        offset += count;
    }
    if (!(fp8Key ? fastllm_dsa_layer::GridSize<true>() : fastllm_dsa_layer::GridSize<false>()))
        return false;
    std::vector<Data *> outputs{x.query,  x.attention,          x.projected,
                                x.normed, &x.scratch->dsaLayer, &x.scratch->dsaFused};
    const int widths[] = {1536, 1024, 4096, 4096};
    const size_t bytes[] = {size_t(rows) * 1536 * 2,
                            size_t(rows) * 1024 * 2,
                            elements * 2,
                            elements * 2,
                            L::Bytes,
                            fastllm_dsa_fused::AttnDsaScratchLayout::kBytes};
    bool capture = FastllmCudaGraphIsCapturingFast();
    for (size_t i = 0; i < outputs.size(); ++i) {
        Data *d = outputs[i];
        if (!d)
            return false;
        for (auto source : sources)
            if (d == source)
                return false;
        for (size_t j = 0; j < i; ++j)
            if (d == outputs[j])
                return false;
        if (capture && (!d->cudaData || d->expansionSize * d->unitSize < bytes[i]))
            return false;
        if ((d->isFake || d->cudaDataBorrowed) && (!d->cudaData || d->Count(0) * d->unitSize < bytes[i]))
            return false;
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto &d = *outputs[i];
        d.dataType = i < 4 ? BFLOAT16 : INT32;
        d.UpdateUnitSize();
        if (i < 4) {
            auto dims = x.hidden->dims;
            dims.back() = widths[i];
            d.Resize(dims);
        } else
            d.Resize({int((bytes[i] + 3) / 4)});
        d.ToDevice(CUDA, {device}, false);
        d.Allocate(false);
    }
    struct Range {
        const void *p;
        size_t n;
    };
    auto disjoint = [](Range x, Range y) {
        if (!x.p || !y.p || !x.n || !y.n)
            return true;
        uintptr_t a = uintptr_t(x.p), b = uintptr_t(y.p);
        return a <= b ? b - a >= x.n : a - b >= y.n;
    };
    std::vector<Range> writes{{x.hidden->cudaData, elements * 2}};
    if (x.feature)
        writes.push_back({x.feature->cudaData, elements * 2});
    for (size_t i = 0; i < outputs.size(); ++i)
        writes.push_back({outputs[i]->cudaData, bytes[i]});
    for (size_t i = 0; i < writes.size(); ++i) {
        for (size_t j = 0; j < i; ++j)
            if (!disjoint(writes[i], writes[j]))
                return false;
        for (const Data *s : sources)
            if (s && s != x.hidden && s != x.feature && !s->isPagedKVCache)
                if (!disjoint(writes[i], {s->cudaData, s->Count(0) * s->unitSize}))
                    return false;
    }
    a.rows = rows;
    a.requestCount = x.requests.size();
    a.rank = -1;
    a.fp8Key = fp8Key;
    for (size_t i = 0; i < x.requests.size(); ++i)
        a.requests[i] = x.requests[i];
    a.branch = x.branch ? (const B *)x.branch->cudaData : nullptr;
    a.hidden = (B *)x.hidden->cudaData;
    a.feature = x.feature ? (B *)x.feature->cudaData : nullptr;
    a.inputNorm = (const float *)x.inputNorm->cudaData;
    a.indexNorm = (const float *)x.indexNorm->cudaData;
    a.indexBias = (const float *)x.indexBias->cudaData;
    a.qkvWeight = (const B *)x.qkvWeight->cudaData;
    a.indexKeyWeight = (const B *)x.indexKeyWeight->cudaData;
    a.indexQueryWeight = (const B *)x.indexQueryWeight->cudaData;
    a.indexWeight = (const B *)x.indexWeight->cudaData;
    a.outputWeight = (const B *)x.outputWeight->cudaData;
    a.sink = x.sink && !x.sink->dims.empty() ? (const float *)x.sink->cudaData : nullptr;
    a.positions = (const float *)x.positions->cudaData;
    a.query = (B *)x.query->cudaData;
    a.attention = (B *)x.attention->cudaData;
    a.projected = (B *)x.projected->cudaData;
    a.normed = (B *)x.normed->cudaData;
    a.workspace = (unsigned char *)x.scratch->dsaLayer.cudaData;
    a.coreWorkspace = (unsigned char *)x.scratch->dsaFused.cudaData;
    a.eps = x.eps;
    a.theta = x.theta;
    a.valueScale = x.valueScale;
    for (int i = 0; i < rows; ++i)
        a.bindings[i].indices = (const int *)(a.workspace + L::Indices) + i * 2048;
    return true;
}
