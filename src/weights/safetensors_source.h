#pragma once
#include "exl3_source.h"

namespace fastllm {
template<class Item>
void NormalizeSafeTensorSources(const std::string &quantization,
                               std::map<std::string, Item> &items) {
    if (quantization == "exl3") NormalizeExl3Sources(items);
}
}
