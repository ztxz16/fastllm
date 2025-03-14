//
// Created by huangyuyang on 24-4-19.
//

#include "kvcache.h"

namespace fastllm {
    KVCache::KVCache(fastllm::DataType dataType, int head, int dim)  {
        this->dataType = dataType;
        if (dataType == DataType::FLOAT32) {
            this->unitSize = 4;
        } else if (dataType == DataType::FLOAT16 || dataType == DataType::BFLOAT16) {
            this->unitSize = 2;
        } else if (dataType == DataType::INT8) {
            this->unitSize = 1;
        }

        this->head = head;
        this->dim = dim;
        this->currentCap = 0;
        this->len = 0;
    }

    KVCache::~KVCache() {
        delete this->data;
    }

    void KVCache::Append(int len, uint8_t *data)  {
        this->lastFlushTime = std::chrono::system_clock::now();
        if (this->len + len > this->currentCap) {
            int newCap = ((this->len + len - 1) / unitLen + 1) * unitLen;
            if (this->currentCap != 0) {
                uint8_t *old = this->data;
                this->data = new uint8_t [head * newCap * dim * unitSize];
                for (int h = 0; h < head; h++) {
                    memcpy(this->data + h * newCap * dim * unitSize,
                           old + h * this->currentCap * dim * unitSize,
                           this->currentCap * dim * unitSize);
                }
                delete old;
            } else {
                this->data = new uint8_t [head * newCap * dim * unitSize];
            }
            this->currentCap = newCap;
        }
        for (int h = 0; h < head; h++) {
            memcpy(this->data + (h * this->currentCap + this->len) * dim * unitSize,
                   data + h * len * dim * unitSize,
                   len * dim * unitSize);
        }
        this->len += len;
    }

    KVCache *KVCacheManager::Get(long long uid) {
        if (this->caches.find(uid) == this->caches.end()) {
            return nullptr;
        }
        return this->caches[uid];
    }

    KVCache *KVCacheManager::Get(long long uid, fastllm::DataType dataType, int head, int dim) {
        if (this->caches.find(uid) == this->caches.end()) {
            this->caches[uid] = new KVCache(dataType, head, dim);
        }
        return this->caches[uid];
    }

    void KVCacheManager::Delete(long long uid) {
        if (this->caches.find(uid) != this->caches.end()) {
            delete this->caches[uid];
            this->caches.erase(uid);
        }
    }
}