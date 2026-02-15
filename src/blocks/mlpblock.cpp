#include "baseblock.h"
#include "fastllm.h"

namespace fastllm {
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
    ) {
        /* if (CanRunMLP()) {
            Data w3;
            Data mlpOutput;
            MLP(*input, *gateUp, *GetEmptyData(), *down, *GetEmptyData(), *gateUpResult, *swigluResult, w3, mlpOutput);
            AddTo(*output, mlpOutput);
        } else */ {
            LinearSwigluBlock(input, gateUp, GetEmptyData(), gateUpResult, swigluResult);
            LinearAddBlock(swigluResult, down, GetEmptyData(), gateUpResult, output);
        }
    }
}
