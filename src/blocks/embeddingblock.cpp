#include "baseblock.h"
#include "fastllm.h"

namespace fastllm {
    void EmbeddingBlock (
        Data *input,
        Data *weight,
        Data *output,
        DataType outputType
    ) {
        if (weight->dataType != outputType) {
            ToDataType(*weight, outputType);
        }
        EmbeddingDirect(*input, *weight, *output);
    }
}
