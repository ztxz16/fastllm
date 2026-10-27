#include "baseblock.h"
#include "fastllm.h"

namespace fastllm {
    namespace {
        bool ShouldTrySwigluLinearAdd(Data *input, Data *gateUp, Data *down, Data *output) {
            if (input == nullptr || gateUp == nullptr || down == nullptr || output == nullptr ||
                input->dims.empty() || gateUp->dims.size() != 2 || down->dims.size() != 2 ||
                output->dims.empty() || input->dims.back() <= 0) {
                return false;
            }
            int inter = down->dims[1];
            int hidden = output->dims.back();
            bool shapeSupported = (input->dataType == DataType::FLOAT16 || input->dataType == DataType::BFLOAT16) &&
                   output->dataType == input->dataType &&
                   down->dataType == DataType::FP8_E4M3 &&
                   down->blockM == 128 && down->blockK == 128 && !down->scales.empty() &&
                   gateUp->dims[0] == inter * 2 && down->dims[0] == hidden &&
                   (inter % 128) == 0 && (hidden % 128) == 0;
            if (!shapeSupported) {
                return false;
            }
            // Query the selected backend with shape metadata before computing
            // gate/up. Backend-specific switches belong to CanRun, not MLPBlock.
            std::vector<int> dims = input->dims;
            dims.back() = gateUp->dims[0];
            Data gateUpShape(input->dataType, dims);
            return CanRunSwigluLinearAdd(gateUpShape, *down, *GetEmptyData(), *output);
        }
    }

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
        gateUp->tpPackType = TP_PACK_GATEUP;
        /* if (CanRunMLP()) {
            Data w3;
            Data mlpOutput;
            MLP(*input, *gateUp, *GetEmptyData(), *down, *GetEmptyData(), *gateUpResult, *swigluResult, w3, mlpOutput);
            AddTo(*output, mlpOutput);
        } else */ {
            if (ShouldTrySwigluLinearAdd(input, gateUp, down, output)) {
                Linear(*input, *gateUp, *GetEmptyData(), *gateUpResult);
                if (CanRunSwigluLinearAdd(*gateUpResult, *down, *GetEmptyData(), *output)) {
                    SwigluLinearAdd(*gateUpResult, *down, *GetEmptyData(), *swigluResult, *output);
                    return;
                }
                Swiglu(*gateUpResult, *swigluResult);
                LinearAddBlock(swigluResult, down, GetEmptyData(), gateUpResult, output);
                return;
            }
            LinearSwigluBlock(input, gateUp, GetEmptyData(), gateUpResult, swigluResult);
            LinearAddBlock(swigluResult, down, GetEmptyData(), gateUpResult, output);
        }
    }
}
