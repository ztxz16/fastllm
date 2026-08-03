#include "baseblock.h"
#include "fastllm.h"
#include "utils.h"

namespace fastllm {
    static void ClearStaleReplicatedView(Data *output) {
        if (output == nullptr || !output->multiDeviceData || !output->IsTensorParallelReplicated()) {
            return;
        }

        bool stale = (!output->tpGlobalDims.empty() && output->tpGlobalDims != output->dims);
        if (!stale) {
            for (auto &it : output->multiDeviceDatas) {
                if (it.second != nullptr && it.second->dims != output->dims) {
                    stale = true;
                    break;
                }
            }
        }
        if (!stale) {
            return;
        }

        for (auto &it : output->multiDeviceDatas) {
            delete it.second;
        }
        output->multiDeviceDatas.clear();
        output->multiDeviceData = false;
        output->ClearTensorParallelLayout();
    }

    void MergeMOEBlock (
        Data *input, Data *expertIndex, Data *expertScore,
        std::vector <Data*> *weights, std::vector <Data*> *biass,
        Data *w1, Data *w2, Data *w3, Data *tempInput, Data *tempOutput,
        float sharedScale, Data *output, int layer,
        DataType dataType, DataType moeAtype,
        Data *moeInputTemp, Data *moeOutputTemp,
        MoeGateType gateType, bool expertParallel,
        float swigluLimit, bool deepSeekV4Mode,
        Data *pairedReduceInput
    ) {
        if (dataType == moeAtype) {
            MergeMOE(*input, *expertIndex, *expertScore,
                     *weights, *biass,
                     *w1, *w2, *w3, *tempInput, *tempOutput,
                     sharedScale, *output, layer, gateType, expertParallel,
                     swigluLimit, deepSeekV4Mode, pairedReduceInput);
        } else {
            AssertInFastLLM(
                pairedReduceInput == nullptr,
                "Paired MoE reduction requires matching model/MoE activation types.\n");
            ToDataType(*input, *moeInputTemp, moeAtype);
            MergeMOE(*moeInputTemp, *expertIndex, *expertScore,
                     *weights, *biass,
                     *w1, *w2, *w3, *tempInput, *tempOutput,
                     sharedScale, *moeOutputTemp, layer, gateType, expertParallel,
                     swigluLimit, deepSeekV4Mode, nullptr);
            ToDataType(*moeOutputTemp, *output, dataType);
        }
        ClearStaleReplicatedView(output);
    }
}
