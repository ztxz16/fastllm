#pragma once
#include "fastllm.h"

namespace fastllm {
// Packed weights may encode block geometry or a sequence of independent
// projections. Keep those layouts out of the checkpoint merge scheduler.
inline bool CanConcatWeightRows(const Data &part, const Data &first) {
    if (part.dims.size() != 2 || first.dims.size() != 2 ||
        part.dataType != first.dataType || part.ggmlType != first.ggmlType ||
        part.dims[1] != first.dims[1]) return false;
    if (part.dataType == EXL3)
        return part.blockK == first.blockK && part.blockM > 0 && first.blockM > 0 &&
            part.dims[0] % part.blockM == 0 && first.dims[0] % first.blockM == 0 &&
            part.dims[0] / part.blockM == first.dims[0] / first.blockM;
    return (part.groupCnt == -1 || (part.groupCnt > 0 && part.dims[1] % part.groupCnt == 0)) &&
        (part.blockK == -1 || (part.blockK > 0 && part.dims[0] % part.blockK == 0)) &&
        (part.blockM == -1 || (part.blockM > 0 && part.dims[1] % part.blockM == 0));
}

inline void ConfigureWeightRowConcat(Data &merged, const std::vector<const Data *> &parts) {
    if (merged.dataType != EXL3) return;
    AssertInFastLLM(!parts.empty(), "Cannot concatenate an empty weight list.");
    merged.blockM = 0;
    for (const Data *part : parts) {
        AssertInFastLLM(part && CanConcatWeightRows(*part, *parts.front()),
                       "Incompatible packed weight projections.");
        merged.blockM += part->blockM;
    }
    merged.UpdateUnitSize();
}
}
