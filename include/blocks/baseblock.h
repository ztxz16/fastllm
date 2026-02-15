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
}

#endif //FASTLLM_BASEBLOCK_H