#include "baseblock.h"
#include "basellm.h"
#include "fastllm.h"

namespace fastllm {
    /*
    Paged Attention Block:
    1. Linear QKV projection
    2. QKVRMSNormRope + Split Q/K/V + AppendPagedCache(K,V)
    3. AttentionPagedBatch
    4. Reshape + Permute
    5. output += Linear(attenOutput, oWeight, oBias)
    */
    void AttentionPagedBlock (
        Data *attenInput,
        Data *mergeQkvWeight, Data *mergeQkvBias,
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
        Data *hiddenStates
    ) {
        // 1. Linear QKV projection
        Linear(*attenInput, *mergeQkvWeight, *mergeQkvBias, *qkv);

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
        int bsz = attenInput->dims[0], seqlen = attenInput->dims[1];
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
            curPageLen, batch);

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
