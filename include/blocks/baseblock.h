#ifndef FASTLLM_BASEBLOCK_H
#define FASTLLM_BASEBLOCK_H

#include "fastllm.h"

namespace fastllm {
    /*
    output += Linear(input, weight, bias)
    */
    void LinearAddBlock (
        Data *input, 
        Data *weight, Data *bias,
        Data *middle, Data *output
    );

    /*
    output = Swiglu(Linear(input, weight, bias))
    */
    void LinearSwigluBlock (
        Data *input, 
        Data *weight, Data *bias,
        Data *middle, Data *output
    );

    /*
    Embedding with optional type conversion:
    if weight->dataType != outputType: Embedding -> ToDataType
    else: Embedding directly into output
    */
    void EmbeddingBlock (
        Data *input,
        Data *weight,
        Data *output,
        DataType outputType
    );

    /*
    gateUpResult = Linear(input, gateUp)
    swigluResult = Swiglu(gateUpResult)
    output += Linear(swigluResult, down)
    */
    void MLPBlock (
        Data *input, 
        Data *gateUp, Data *down, 
        Data *gateUpResult, 
        Data *swigluResult,
        Data *output
    );

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

    参数说明:
    - attenInput: attention的输入 [bsz, seqlen, hidden]
    - mergeQkvWeight/Bias: 合并的QKV投影权重/偏置
    - qNormWeight/kNormWeight: Q/K的RMSNorm权重
    - oWeight/oBias: output projection权重/偏置
    - allPositionIds: 位置编码
    - pastKeyValues: batch * block_cnt 个 (pastKey, pastValue) 对
    - batchPastKeys/Values: batch个pastKey/Value指针的列表
    - qkv/q/attenOutput/attenLastOutput: 中间变量
    - insertIndexs/insertPositions: append paged cache 批量参数（跨层共享）
    - qSizes/pageSizes/pageIndexs/lastPageLens: paged batch decode 参数（跨层共享）
    - generatedAppendParams/generatedDecodeParams: 是否已生成跨层共享参数
    - hiddenStates: 残差连接的输出
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
        int rope_type, // RoPEType enum value
        bool kvCacheInCPU,
        bool isPrefill,
        Data *hiddenStates,
        bool doQKNorm,
        bool doPostQKNorm
    );
    /*
    MergeMOE with optional activation type conversion:
    if dataType != moeAtype: ToDataType(input) -> MergeMOE -> ToDataType(output)
    else: MergeMOE directly
    */
    void MergeMOEBlock (
        Data *input, Data *expertIndex, Data *expertScore,
        std::vector <Data*> *weights, std::vector <Data*> *biass,
        Data *w1, Data *w2, Data *w3, Data *tempInput, Data *tempOutput,
        float sharedScale, Data *output, int layer,
        DataType dataType, DataType moeAtype,
        Data *moeInputTemp, Data *moeOutputTemp
    );

    class basellm;

    /*
    LLM Sampling Block:
    1. (可选) 提取 last token (当 all1=false 时)
    2. RMSNorm(hiddenStates, normWeight)
    3. Linear(hiddenStates, lmHeadWeight) -> logits
    4. ResetLogitsOfEOS
    5. Sampling (allSimple / batch CUDA sampling / per-sample fallback)
    输入: hiddenStates
    输出: lastRet (采样得到的 token id)
    */
    void LLMSamplingBlock (
        basellm *model,
        Data *hiddenStates,
        Data *normWeight,
        Data *lmHeadWeight,
        float rms_norm_eps,
        int batch,
        bool all1,
        const std::vector<int> &seqLens,
        std::vector<std::pair<Data*, Data*>> &pastKeyValues,
        const std::vector<GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector<std::vector<float>*> *retLogits,
        std::vector<int> &lastRet
    );
}

#endif //FASTLLM_BASEBLOCK_H