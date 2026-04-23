#include "baseblock.h"
#include "fastllm.h"

namespace fastllm {
    void EmbeddingBlock (
        Data *input,
        Data *weight,
        Data *output,
        DataType outputType
    ) {
        // 不开 cuda_embedding 时强制把权重保留在 CPU 上，避免 ToDataType / EmbeddingDirect
        bool keepOnCpu = !GetCudaEmbedding() || GetLowMemMode();
        if (keepOnCpu) {
            if (weight->dataDevice != DataDevice::CPU) {
                weight->ToDevice(DataDevice::CPU);
            }
            if (weight->dataType != outputType) {
                ToDataTypeForceCPU(*weight, outputType);
            }
        } else {
            if (weight->dataType != outputType) {
                ToDataType(*weight, outputType);
            }
        }
        EmbeddingDirect(*input, *weight, *output);
    }
}
