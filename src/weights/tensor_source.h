#pragma once
#include "fastllm.h"
#include <memory>

namespace fastllm {
// Logical weights assembled from several checkpoint tensors. Ordinary tensors
// retain the direct file path; compound sources own format validation/readback.
struct TensorSource {
    virtual ~TensorSource() = default;
    // DATA_AUTO_NONE leaves destination precision to the ordinary loader.
    virtual DataType StoredType() const = 0;
    virtual uint8_t *Read(DataType destination) const = 0;
    virtual void Configure(Data &weight) const = 0;
};
}
