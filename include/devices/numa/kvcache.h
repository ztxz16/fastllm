//
// Created by huangyuyang on 24-4-19.
//

#ifndef TFACCCOMPUTESERVER_KVCACHE_H
#define TFACCCOMPUTESERVER_KVCACHE_H

#include "fastllm.h"

namespace fastllm {
    struct KVCache {
        std::chrono::system_clock::time_point lastFlushTime;

        DataType dataType;
        int unitSize;

        int len;
        int head, dim; // 尺寸为[head, len, dim]
        int currentCap; // 预分配[head, currentCap, dim]的空间，当middle超出时扩容
        int unitLen = 64; // 扩容单位
        uint8_t *data = nullptr;

        KVCache (DataType dataType, int head, int dim);
        ~KVCache();

        void Append(int len, uint8_t *data);
    };

    struct KVCacheManager {
        std::unordered_map <long long, KVCache*> caches;

        KVCache *Get(long long uid);
        KVCache *Get(long long uid, DataType dataType, int head, int dim);
        void Delete(long long uid);
    };
}

#endif //TFACCCOMPUTESERVER_KVCACHE_H
