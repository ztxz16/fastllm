#include "baseblock.h"
#include "fastllm.h"

namespace fastllm {
    /*
    output = Swiglu(Linear(input, weight, bias))
    */
    void LinearSwigluBlock (
        Data *input, 
        Data *weight, Data *bias,
        Data *middle, Data *output
    ) {
        if (CanRunLinearSwiglu(*input, *weight)) {
            LinearSwiglu(*input, *weight, *bias, *middle, *output);
        } else {
            Linear(*input, *weight, *bias, *middle);
            Swiglu(*middle, *output);
        }
    }
}
