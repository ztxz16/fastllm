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
}

#endif //FASTLLM_BASEBLOCK_H