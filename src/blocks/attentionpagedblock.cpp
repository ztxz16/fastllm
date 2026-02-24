#include "baseblock.h"
#include "basellm.h"
#include "fastllm.h"

namespace fastllm {
    /*
    Paged Attention Block:
    Decode 路径 (isPrefill=false):
        1. Linear QKV projection
        2. QKVRMSNormRope + Split Q/K/V + AppendPagedCacheBatch(K,V)（融合算子）
        3. AttentionPagedBatch
        4. Reshape + Permute
        5. output += Linear(attenOutput, oWeight, oBias)
    Prefill 路径 (isPrefill=true):
        1. Linear QKV projection
        2. Split Q/K/V + Reshape + RMSNorm + RopeEncoding + Permute + Reshape
        3. 逐 batch: AppendPagedCache(K,V)
        4. AttentionPagedBatch（批量 attention）
        5. Reshape
        6. output += Linear(attenOutput, oWeight, oBias)
    */
    void AttentionPagedBlock (
        Data *attenInput,
        Data *mergeQkvWeight, Data *mergeQkvBias,
        Data *preQNormWeight, Data *preKNormWeight,
        Data *qNormWeight, Data *kNormWeight,
        Data *oWeight, Data *oBias,
        Data *allPositionIds,
        std::vector<std::pair<Data*, Data*>> *pastKeyValues,
        std::vector<Data*> *batchPastKeys,
        std::vector<Data*> *batchPastValues,
        Data *qkv, Data *q, Data *attenOutput, Data *attenLastOutput,
        Data *insertIndexs, Data *insertPositions,
        Data *qSizes, Data *pageSizes, Data *pageIndexs, Data *lastPageLens,
        bool *generatedAppendParams, bool *generatedDecodeParams,
        int batch, int block_cnt, int layerIdx,
        const std::vector<int> &seqLens,
        int num_attention_heads, int num_key_value_heads, int head_dim,
        int rotary_dim, float rms_norm_eps,
        float rope_base, float rope_factor, int max_positions,
        int rope_type,
        bool kvCacheInCPU,
        bool isPrefill,
        Data *hiddenStates,
        bool doQKNorm,
        bool doPostQKNorm
    ) {
        // 1. Linear QKV projection
        Linear(*attenInput, *mergeQkvWeight, *mergeQkvBias, *qkv);

        if (doPostQKNorm) {
            int per = qkv->dims.back() / (num_attention_heads / num_key_value_heads + 2);
            int qdim = per * (num_attention_heads / num_key_value_heads);
            RMSNormPart(*qkv, *preQNormWeight, rms_norm_eps, 0, qdim, *qkv);
            RMSNormPart(*qkv, *preKNormWeight, rms_norm_eps, qdim, qdim + per, *qkv);
        }

        // 2. 计算 targetSeqLength 和 rope 参数
        int targetSeqLength = 0;
        for (int b = 0; b < batch; b++) {
            Data &pastKey = *(*pastKeyValues)[b * block_cnt + layerIdx].first;
            Data &pastValue = *(*pastKeyValues)[b * block_cnt + layerIdx].second;
            if (kvCacheInCPU) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(qkv->dataDevice);
                pastValue.ToDevice(qkv->dataDevice);
            }
            targetSeqLength = std::max(targetSeqLength, 
                (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqLens[b] : seqLens[b]);
        }

        float curRopeTheta = rope_base;
        if (targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
            float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), 
                rotary_dim / (rotary_dim - 2));
            curRopeTheta = rope_base * scale;
        }
        float ropeScale = (rope_type == RoPEType::LINEAR_SCALE) ? rope_factor : 1.0f;

        // 3. 准备batch的pastKeyValues列表
        for (int b = 0; b < batch; b++) {
            (*batchPastKeys)[b] = (*pastKeyValues)[b * block_cnt + layerIdx].first;
            (*batchPastValues)[b] = (*pastKeyValues)[b * block_cnt + layerIdx].second;
        }

        int bsz = attenInput->dims[0], seqlen = attenInput->dims[1];

        if (!isPrefill && (*batchPastKeys)[0]->pagedKVCacheData == nullptr) {
            isPrefill = true;
        }

        if (isPrefill) {
            // ===== Prefill 路径：逐 batch AppendPagedCache + 批量 AttentionPagedBatch =====
            Data k, v;

            // 2.1 Split QKV
            int per = qkv->dims.back() / (num_attention_heads / num_key_value_heads + 2);
            int qdim = per * (num_attention_heads / num_key_value_heads);
            Split(*qkv, -1, 0, qdim, *q);
            Split(*qkv, -1, qdim, qdim + per, k);
            Split(*qkv, -1, qdim + per, qdim + per * 2, v);

            // 2.2 Reshape to [bsz, seqlen, num_heads, head_dim]
            std::vector<int> qkvSize = {bsz, seqlen, -1, head_dim};
            q->Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            // 2.3 RMSNorm + RopeEncoding（对 q 和 k）
            if (doQKNorm) {
                RMSNorm(*q, *qNormWeight, rms_norm_eps, *q);
                RMSNorm(k, *kNormWeight, rms_norm_eps, k);
            }
            RopeEncoding(*q, *allPositionIds, rotary_dim, curRopeTheta, ropeScale);
            RopeEncoding(k, *allPositionIds, rotary_dim, curRopeTheta, ropeScale);

            // 2.4 Permute [bsz, seqlen, num_heads, head_dim] -> [bsz, num_heads, seqlen, head_dim]
            PermuteSelf(*q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            // 2.5 Reshape to [num_kv_heads, seqlen, head_dim]（k/v 用 kv_heads，q 用 q_heads）
            k.Reshape({-1, seqlen, head_dim});
            v.Reshape({-1, seqlen, head_dim});
            q->Reshape({-1, seqlen, head_dim});

            // 2.6 逐 batch 做 AppendPagedCache
            // k/v 形状为 [num_kv_heads, totalSeqLen, head_dim]
            if (batch == 1) {
                PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                    layerIdx * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, k);
                PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                    layerIdx * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, v);
                AppendPagedCache(*pagedCacheKManager, *(*batchPastKeys)[0], k);
                AppendPagedCache(*pagedCacheVManager, *(*batchPastValues)[0], v);
            } else  {
                int total = 0;
                Data curK, curV;

                for (int b = 0; b < batch; b++) {
                    Data &pastKey = *(*batchPastKeys)[b];
                    Data &pastValue = *(*batchPastValues)[b];

                    Split(k, 1, total, total + seqLens[b], curK);
                    Split(v, 1, total, total + seqLens[b], curV);

                    PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                        layerIdx * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, curK);
                    PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                        layerIdx * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, curV);
                    AppendPagedCache(*pagedCacheKManager, pastKey, curK);
                    AppendPagedCache(*pagedCacheVManager, pastValue, curV);

                    total += seqLens[b];
                }
            }

            // 2.7 使用 AttentionPagedBatch 批量做 attention
            {
                Data &kCaches = *(*batchPastKeys)[0];
                Data &vCaches = *(*batchPastValues)[0];

                // 生成 PagedBatch 参数，传入 seqLens 使 qSizes 按各 batch 的 seqLen 前缀和生成
                GeneratePagedBatchParams(*q, *batchPastKeys, batch,
                    *qSizes, *pageSizes, *pageIndexs, *lastPageLens, seqLens);

                AttentionPagedBatch(*q,
                    kCaches, vCaches,
                    *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                    *attenOutput, q->dims[0] / kCaches.dims[0], 1.0 / sqrt(head_dim), 1, layerIdx > 0);
            }

            // 2.8 AttentionPagedBatch 输出形状为 [seqlen, num_heads, head_dim]
            // Reshape 为 [1, seqlen, num_heads * head_dim]
            attenOutput->Reshape({1, seqlen, -1});
            // 2.9 output += Linear(attenOutput, oWeight, oBias)
            LinearAddBlock(attenOutput, oWeight, oBias, attenLastOutput, hiddenStates);
        } else {
            // ===== Decode 路径：使用融合算子批量处理 =====

            // 4. 获取第一个batch的pastKey和pastValue（所有batch共享同一个PagedCacheManager）
            Data &kCaches = *(*batchPastKeys)[0];
            Data &vCaches = *(*batchPastValues)[0];
            PagedCacheManager *pagedCacheKManager = kCaches.pagedKVCacheData;
            PagedCacheManager *pagedCacheVManager = vCaches.pagedKVCacheData;

            // 5. 生成分页批量参数（insertIndexs/insertPositions 在所有层共享）
            if (!(*generatedAppendParams)) {
                GenerateAppendPagedCacheBatchParams(*pagedCacheKManager, *batchPastKeys, batch, 
                    *insertIndexs, *insertPositions);
                *generatedAppendParams = true;
            }

            // 6. 融合操作：QKVRMSNormRope + Split Q/K/V + AppendPagedCacheBatch(K,V)
            // q 输出为 [bsz * q_heads, seqlen, head_dim]（已做Permute）
            // K/V 直接写入 paged cache
            q->dataType = qkv->dataType;
            q->Resize({bsz * num_attention_heads, seqlen, head_dim});
            int curPageLen = kCaches.pageLen;
            QKVRMSNormRopeSplitAppendPagedCache(*qkv,
                *qNormWeight, *kNormWeight,
                *allPositionIds,
                *q,
                *(Data*)pagedCacheKManager, *(Data*)pagedCacheVManager,
                *insertIndexs, *insertPositions,
                num_attention_heads, num_key_value_heads, head_dim,
                rotary_dim, rms_norm_eps, curRopeTheta, ropeScale,
                curPageLen, batch, doQKNorm);

            // 7. 更新 pastKey/pastValue 的 pageIndex 和 lastPageLen
            for (int b = 0; b < batch; b++) {
                auto updatePageMeta = [](Data *cache, PagedCacheManager *mgr) {
                    if (cache->lastPageLen < cache->pageLen) {
                        cache->lastPageLen++;
                    } else {
                        cache->lastPageLen = 0;
                        cache->pageIndex.push_back(mgr->GetUnusedPageIndex(true));
                    }
                };
                updatePageMeta((*batchPastKeys)[b], pagedCacheKManager);
                updatePageMeta((*batchPastValues)[b], pagedCacheVManager);
            }

            // 8. 生成分页批量参数并执行 AttentionPagedBatch
            if (!(*generatedDecodeParams)) {
                GeneratePagedBatchParams(*q, *batchPastKeys, batch, 
                    *qSizes, *pageSizes, *pageIndexs, *lastPageLens);
                *generatedDecodeParams = true;
            }
            AttentionPagedBatch(*q, 
                kCaches, vCaches, 
                *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                *attenOutput, q->dims[0] / kCaches.dims[0], 1.0 / sqrt(head_dim), 1, layerIdx > 0);

            // 9. Reshape + Permute
            attenOutput->Reshape({seqlen, bsz, -1});
            PermuteSelf(*attenOutput, {1, 0, 2});

            // 10. output += Linear(attenOutput, oWeight, oBias)
            LinearAddBlock(attenOutput, oWeight, oBias, attenLastOutput, hiddenStates);
        }
    }
}
