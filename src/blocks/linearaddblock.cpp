#include "baseblock.h"
#include "fastllm.h"

namespace fastllm {
    /*
    output += Linear(input, weight, bias)
    */
    void LinearAddBlock (
        Data *input, 
        Data *weight, Data *bias,
        Data *middle, Data *output
    ) {
        if (CanRunLinearAdd(*input, *weight, *bias, *output)) {
            LinearAdd(*input, *weight, *bias, *middle, *output);
        } else {
            Linear(*input, *weight, *bias, *middle)
            if (middle->dataType != output->dataType) {
                Data tempMiddle;
                ToDataType(*middle, tempMiddle, output->dataType);
                AddTo(*output, tempMiddle);
            } else {
                AddTo(*output, *middle);
            }
        }
    }
}
