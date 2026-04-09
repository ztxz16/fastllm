#include "baseblock.h"
#include "fastllm.h"

namespace fastllm {
    void MergeMOEBlock (
        Data *input, Data *expertIndex, Data *expertScore,
        std::vector <Data*> *weights, std::vector <Data*> *biass,
        Data *w1, Data *w2, Data *w3, Data *tempInput, Data *tempOutput,
        float sharedScale, Data *output, int layer,
        DataType dataType, DataType moeAtype,
        Data *moeInputTemp, Data *moeOutputTemp,
        MoeGateType gateType
    ) {
        if (dataType == moeAtype) {
            MergeMOE(*input, *expertIndex, *expertScore,
                     *weights, *biass,
                     *w1, *w2, *w3, *tempInput, *tempOutput,
                     sharedScale, *output, layer, gateType);
        } else {
            ToDataType(*input, *moeInputTemp, moeAtype);
            MergeMOE(*moeInputTemp, *expertIndex, *expertScore,
                     *weights, *biass,
                     *w1, *w2, *w3, *tempInput, *tempOutput,
                     sharedScale, *moeOutputTemp, layer, gateType);
            ToDataType(*moeOutputTemp, *output, dataType);
        }
    }
}
