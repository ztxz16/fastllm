#include "kvmem.h"
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm {
namespace {
    void Check(cudaError_t error) {
        if (error != cudaSuccess) throw std::runtime_error(std::string("KVMem CUDA: ") + cudaGetErrorString(error));
    }

    template<class T>
    __global__ void IndexKeys(const T *keys, float *sums, int oldTokens, int tokens,
                              int pageLen, int width, int firstPage) {
        int feature = blockIdx.x * blockDim.x + threadIdx.x;
        int page = firstPage + blockIdx.y;
        if (feature >= width) return;
        int start = max(oldTokens, page * pageLen);
        int end = min(oldTokens + tokens, (page + 1) * pageLen);
        float value = 0;
        for (int t = start; t < end; ++t) value += (float)keys[(t - oldTokens) * width + feature];
        sums[(size_t)page * width + feature] += value;
    }

    template<class T>
    __global__ void ScorePages(const T *query, const float *sums, float *scores,
                               int pages, int oldTokens, int pageLen, int heads,
                               int kvHeads, int dim) {
        int page = blockIdx.x, head = blockIdx.y;
        int kvHead = head / (heads / kvHeads);
        float value = 0;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            value += (float)query[head * dim + d] * sums[((size_t)page * kvHeads + kvHead) * dim + d];
        }
        __shared__ float partial[256];
        partial[threadIdx.x] = value;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            int count = min(pageLen, oldTokens - page * pageLen);
            scores[(size_t)head * pages + page] = partial[0] / (count * sqrtf((float)dim));
        }
    }

    template<class T>
    __global__ void AppendKV(const T *k, const T *v, T *cacheK, T *cacheV,
                             const int *slots, int oldTokens, int tokens,
                             int pageLen, int heads, int dim) {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= (size_t)heads * tokens * dim) return;
        int d = i % dim, t = (i / dim) % tokens, h = i / ((size_t)tokens * dim);
        int logical = oldTokens + t;
        int slot = slots[logical / pageLen - oldTokens / pageLen];
        size_t dst = (((size_t)slot * pageLen + logical % pageLen) * heads + h) * dim + d;
        cacheK[dst] = k[i];
        cacheV[dst] = v[i];
    }

    class CudaKvMemCache final : public KvMemCache {
        KvMemConfig config;
        int device, pageLen, heads, dim;
        int previousAppendTokens = 0;
        int scoredQueryBucket = -1, scoredQueryInterval = 0;
        bool failed = false;
        DataType type;
        size_t onePageBytes;
        KvMemPages pages;
        PagedCacheManager poolK, poolV;
        Data sums, scoreBuffer, writeSlots, provisionalKeys;
        std::vector<float> scores;

        void Publish(Data &keyCache, Data &valueCache) {
            for (Data *cache : {&keyCache, &valueCache}) {
                cache->dataDevice = DataDevice::CUDA;
                cache->dataDeviceIds = {device};
                cache->isPagedKVCache = true;
                cache->pageLen = pageLen;
                cache->pagedKVCacheData = cache == &keyCache ? &poolK : &poolV;
                cache->Resize({heads, pages.Tokens(), dim});
                cache->lastPageLen = pages.Tokens() ? (pages.Tokens() - 1) % pageLen + 1 : 0;
                cache->pageIndex.clear();
                for (int p : pages.Visible()) cache->pageIndex.push_back(pages.Slot(p));
            }
        }

        template<class T> void Index(const Data &keys, int oldTokens, int tokens) {
            if (!tokens) return;
            int first = oldTokens / pageLen, last = (oldTokens + tokens - 1) / pageLen;
            IndexKeys<T><<<dim3((heads * dim + 255) / 256, last - first + 1), 256, 0, cudaStreamPerThread>>>(
                (const T*)keys.cudaData, (float*)sums.cudaData, oldTokens, tokens, pageLen, heads * dim, first);
            Check(cudaGetLastError());
        }

        void Allocate(Data &data, DataType dtype, const std::vector<int> &shape) {
            data.dataType = dtype;
            data.UpdateUnitSize();
            data.dataDevice = DataDevice::CUDA;
            data.dataDeviceIds = {device};
            data.Resize(shape);
            data.Allocate();
        }

        template<bool restore, class Page>
        void TransferPages(const std::vector<Page> &batch) {
            if (batch.empty()) return;
            try {
                for (const auto &page : batch) {
                    uint8_t *key = (uint8_t*)poolK.cudaData + (size_t)page.slot * onePageBytes;
                    uint8_t *value = (uint8_t*)poolV.cudaData + (size_t)page.slot * onePageBytes;
                    if constexpr (restore) {
                        Check(cudaMemcpyAsync(key, page.data, onePageBytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
                        Check(cudaMemcpyAsync(value, page.data + onePageBytes, onePageBytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
                    } else {
                        Check(cudaMemcpyAsync(page.data, key, onePageBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                        Check(cudaMemcpyAsync(page.data + onePageBytes, value, onePageBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    }
                }
                Check(cudaStreamSynchronize(cudaStreamPerThread));
            } catch (...) {
                // Callback buffers must outlive every submitted transfer,
                // including when a later submission in the batch fails.
                cudaStreamSynchronize(cudaStreamPerThread);
                throw;
            }
        }

        template<class T>
        void Run(const Data &rawQ, const Data &rawK, const Data &k, const Data &v,
                 Data &keyCache, Data &valueCache) {
            int oldTokens = pages.Tokens(), tokens = k.dims[1];
            if (tokens <= 0 || tokens > config.prefillTokens || tokens > config.maxTokens - oldTokens) {
                throw std::invalid_argument("KVMem: append exceeds configured context/chunk size");
            }
            int oldPages = (oldTokens + pageLen - 1) / pageLen;
            // A shared prefill selection must not see a later query row. Only
            // query[0] and already committed keys participate in retrieval.
            // Ordinary decode reuses retrieval within a committed query page.
            // Speculative verification uses a configurable committed-token
            // interval. Provisional/rejected rows never advance this interval.
            // A policy change also invalidates the previous query bucket.
            const int queryInterval = pages.InTransaction() ? config.retrievalInterval : pageLen;
            const bool prefillRefresh = !pages.InTransaction() && (tokens > 1 || previousAppendTokens > 1);
            if (oldTokens + tokens > config.residentTokens && oldPages > 0 &&
                (scores.empty() || scoredQueryBucket != oldTokens / queryInterval ||
                 scoredQueryInterval != queryInterval || prefillRefresh)) {
                int qHeads = rawQ.dims[2];
                // Growing by a few pages per chunk leaves every smaller
                // allocation in the CUDA pool. Reserve the configured bound
                // once; ScorePages still packs only oldPages scores per head.
                Allocate(scoreBuffer, DataType::FLOAT32,
                         {qHeads, (config.maxTokens + pageLen - 1) / pageLen});
                ScorePages<T><<<dim3(oldPages, qHeads), 256, 0, cudaStreamPerThread>>>(
                    (const T*)rawQ.cudaData, (const float*)sums.cudaData, (float*)scoreBuffer.cudaData,
                    oldPages, oldTokens, pageLen, qHeads, heads, dim);
                Check(cudaGetLastError());
                std::vector<float> logits((size_t)qHeads * oldPages);
                Check(cudaMemcpyAsync(logits.data(), scoreBuffer.cudaData, logits.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost, cudaStreamPerThread));
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                scores.assign(oldPages, 0.0f);
                for (int h = 0; h < qHeads; ++h) {
                    float *row = logits.data() + (size_t)h * oldPages;
                    float peak = *std::max_element(row, row + oldPages);
                    double total = 0;
                    for (int p = 0; p < oldPages; ++p) { row[p] = std::exp(row[p] - peak); total += row[p]; }
                    if (!(total > 0) || !std::isfinite(total)) throw std::runtime_error("KVMem: non-finite retrieval scores");
                    for (int p = 0; p < oldPages; ++p) scores[p] += row[p] / (total * qHeads);
                }
                scoredQueryBucket = oldTokens / queryInterval;
                scoredQueryInterval = queryInterval;
            }
            pages.Append(tokens, scores,
                [&](const std::vector<KvMemPages::ReadPage> &batch) { TransferPages<false>(batch); },
                [&](const std::vector<KvMemPages::WritePage> &batch) { TransferPages<true>(batch); });
            int first = oldTokens / pageLen, last = (oldTokens + tokens - 1) / pageLen;
            std::vector<int> appendSlots;
            for (int p = first; p <= last; ++p) appendSlots.push_back(pages.Slot(p));
            Allocate(writeSlots, DataType::INT32, {(int)appendSlots.size()});
            Check(cudaMemcpyAsync(writeSlots.cudaData, appendSlots.data(), appendSlots.size() * sizeof(int),
                                  cudaMemcpyHostToDevice, cudaStreamPerThread));
            if (pages.InTransaction()) provisionalKeys.CopyFrom(rawK);
            else Index<T>(rawK, oldTokens, tokens);
            AppendKV<T><<<((size_t)heads * tokens * dim + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                (const T*)k.cudaData, (const T*)v.cudaData, (T*)poolK.cudaData, (T*)poolV.cudaData,
                (const int*)writeSlots.cudaData, oldTokens, tokens, pageLen, heads, dim);
            Check(cudaGetLastError());
            // The pageable host metadata must remain valid until its copy ends.
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            previousAppendTokens = tokens;
            Publish(keyCache, valueCache);
        }

    public:
        CudaKvMemCache(const KvMemConfig &config, const Data &k)
            : config(config), device(FastllmCudaGetDevice()), pageLen(GetPageLen()),
              heads(k.dims[0]), dim(k.dims[2]), type(k.dataType),
              onePageBytes((size_t)pageLen * heads * dim * 2), pages(config, pageLen, onePageBytes * 2) {
            for (auto *pool : {&poolK, &poolV}) {
                pool->type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
                pool->pageLen = pageLen;
                pool->maxPages = config.residentTokens / pageLen;
                Allocate(*pool, type, {pool->maxPages, pageLen, heads, dim});
            }
            Allocate(sums, DataType::FLOAT32, {(config.maxTokens + pageLen - 1) / pageLen, heads, dim});
            Check(cudaMemsetAsync(sums.cudaData, 0, sums.GetBytes(), cudaStreamPerThread));
        }
        ~CudaKvMemCache() override {
            int previous = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            cudaStreamSynchronize(cudaStreamPerThread);
            FastllmCudaSetDevice(previous);
        }
        const KvMemStats& Stats() const override { return pages.Stats(); }
        void BeginTransaction(int tokens) override {
            if (failed || FastllmCudaGetDevice() != device) {
                throw std::runtime_error("KVMem: transaction on an unusable cache or wrong device");
            }
            pages.BeginTransaction(tokens);
        }
        void FinishTransaction(int acceptedTokens, Data &keyCache, Data &valueCache) override {
            if (failed || FastllmCudaGetDevice() != device ||
                keyCache.kvMemCache.get() != this || valueCache.kvMemCache.get() != this) {
                throw std::runtime_error("KVMem: invalid transaction owner/device or failed cache");
            }
            // Validate the prefix before launching an irreversible index update.
            pages.FinishTransaction(acceptedTokens);
            try {
                const int committedBase = pages.Tokens() - acceptedTokens;
                if (type == DataType::FLOAT16) {
                    Index<half>(provisionalKeys, committedBase, acceptedTokens);
                } else {
                    Index<__nv_bfloat16>(provisionalKeys, committedBase, acceptedTokens);
                }
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                provisionalKeys.FreeSpace();
                // A fully cancelled round must not retain a selection derived
                // from its uncommitted first query. Partial commits keep row 0.
                if (!acceptedTokens) {
                    scores.clear();
                    scoredQueryBucket = -1;
                    scoredQueryInterval = 0;
                }
                previousAppendTokens = 0;
                Publish(keyCache, valueCache);
            } catch (...) {
                failed = true;
                throw;
            }
        }
        void Append(const Data &rawQ, const Data &rawK, const Data &k, const Data &v,
                    Data &keyCache, Data &valueCache) override {
            if (FastllmCudaGetDevice() != device || k.dataType != type || v.dataType != type ||
                rawQ.dataType != type || rawK.dataType != type || keyCache.dataType != type || valueCache.dataType != type ||
                k.dims.size() != 3 || k.dims != v.dims || k.dims[0] != heads || k.dims[2] != dim ||
                rawK.dims != std::vector<int>({1, k.dims[1], heads, dim}) || rawQ.dims.size() != 4 ||
                rawQ.dims[0] != 1 || rawQ.dims[1] != k.dims[1] || rawQ.dims[2] <= 0 ||
                rawQ.dims[2] % heads != 0 || rawQ.dims[3] != dim) {
                throw std::invalid_argument("KVMem: unsupported device, dtype or Q/K/V shape");
            }
            for (const Data *data : {&rawQ, &rawK, &k, &v}) {
                if (data->dataDevice != DataDevice::CUDA || !data->cudaData || data->multiDeviceData) {
                    throw std::invalid_argument("KVMem: Q/K/V must be contiguous single-device CUDA tensors");
                }
            }
            if (failed) throw std::runtime_error("KVMem: cache unusable after a failed CUDA append");
            try {
                if (type == DataType::FLOAT16) Run<half>(rawQ, rawK, k, v, keyCache, valueCache);
                else if (type == DataType::BFLOAT16) Run<__nv_bfloat16>(rawQ, rawK, k, v, keyCache, valueCache);
                else throw std::invalid_argument("KVMem: only FP16/BF16 KV is supported");
            } catch (...) {
                failed = true;
                throw;
            }
        }
    };
}

void KvMemAppend(const Data &rawQ, const Data &rawK, const Data &k, const Data &v,
                 Data &keyCache, Data &valueCache) {
    if (!keyCache.kvMemConfig || !keyCache.kvMemConfig->enabled || k.dims.size() != 3) {
        throw std::invalid_argument("KVMem: cache is not configured");
    }
    if (!keyCache.kvMemCache) {
        if (!keyCache.pageIndex.empty() || !valueCache.pageIndex.empty()) {
            throw std::invalid_argument("KVMem: cannot attach to an existing KV cache");
        }
        keyCache.kvMemCache = std::make_shared<CudaKvMemCache>(*keyCache.kvMemConfig, k);
        valueCache.kvMemCache = keyCache.kvMemCache;
    }
    if (keyCache.kvMemCache != valueCache.kvMemCache) {
        throw std::invalid_argument("KVMem: key/value cache owners differ");
    }
    keyCache.kvMemCache->Append(rawQ, rawK, k, v, keyCache, valueCache);
}
}
