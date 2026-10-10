#ifndef FASTLLM_GGUF_DIRECT_MERGE_H
#define FASTLLM_GGUF_DIRECT_MERGE_H

#include "gguf.h"

namespace fastllm {
    // Concatenation uses the source storage layout, independent of quantization
    // type. Conversion, padding/strides and incompatible shapes use the loader's
    // existing per-tensor import/merge path instead.
    bool GetGGUFDirectMergeLayout(const std::vector<const ReadGGUFTask*> &inputs,
                                  std::vector<int> &dims, uint64_t &bytes);
    void ReadGGUFTensorBytes(const std::string &fileName, uint64_t offset,
                             uint8_t *destination, uint64_t bytes);

}

#endif
