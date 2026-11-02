#pragma once

namespace fastllm {
class Data;
}

// Optional SM12x draft-only W8A8 head. Target weights stay BF16; logits are FP32.
// Defaults on when eligible; FASTLLM_DSPARK_FP8_HEAD=0 restores the original head.
// Unsupported layouts/backends and graph capture return false without launching.
extern "C" bool FastllmCudaTryFp8DraftHead(const fastllm::Data &input,
    const fastllm::Data &weight, fastllm::Data &output);
