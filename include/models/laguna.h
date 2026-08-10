//
// Laguna text model support.
//

#ifndef FASTLLM_LAGUNA_H
#define FASTLLM_LAGUNA_H

#include "step3p5.h"

#include <string>
#include <vector>

namespace fastllm {
    class LagunaModel : public Step3p5Model {
    public:
        LagunaModel();
        ~LagunaModel() override;

        void InitParams() override;

        std::map<std::string, std::vector<std::pair<std::string, DataType>>>
                GetTensorMap(const std::vector<std::string> &tensorNames) override;

        bool CanUseGPUForward() const override;
        bool NeedAttentionMask(int qlen, int klen) override;
        int GetKVCacheRetainedTokens(int layer) const override;
        bool BoundedKVCacheUsesTokenGrowingStorage() const override;
        bool TryRecordPagedPrefixCacheExtra(ResponseContext *context) override;
        int QueryPagedPrefixCacheExtra(ResponseContext *context,
                                       int maxCachedLen) const override;
        bool RestorePagedPrefixCacheExtra(ResponseContext *context,
                                          int cachedLen) const override;
        std::string ApplyChatTemplate(const ChatMessages &messages) override;

    protected:
        void PrepareMoeWeights() override;
        void PrepareRuntimeWeights() override;
        DataType GPUForwardComputeType() const override;
        DataType GPUForwardCacheType(int layer, DataType requestedType,
                                     DataType computeType) const override;
        bool GPUForwardUseYarnRope(int layer) const override;
        void GPUForwardYarnRopeParams(int layer, float &factor,
                                      float &attentionFactor,
                                      float &correctionLow,
                                      float &correctionHigh) const override;
        int GPUForwardAttentionWindowLeft(int layer) const override;
        int GPUForwardPagedCacheMaxPages(int layer) const override;
        bool GPUForwardUseMambaSoftplusGate(int layer) const override;
        Data *GPUForwardMambaSoftplusALog() override;
        Data *GPUForwardMambaSoftplusDtBias() override;
        void ApplyStepRotary(Data &input, const Data &positionIds, int layer) override;
        void ApplyAttentionGateActivation(Data &gate, int layer) override;
        bool UsePagedAttention(int layer) const override;
        DataType NonPagedAttentionDataType(DataType inputType) const override;
        bool UseHostMergeMoe() const override;
        Data *PrepareAttentionMask(int layer, int pastLen, int qLen,
                                   DataType attentionType, Data *inputMask,
                                   Data &generatedMask) override;

    private:
        float fullRopeTheta = 500000.0f;
        float fullRopeFactor = 128.0f;
        float fullRopeAttentionFactor = 1.0f;
        float fullRopeBetaFast = 32.0f;
        float fullRopeBetaSlow = 1.0f;
        int fullRopeOriginalMaxPosition = 8192;
        int fullRotaryDim = 64;
        float slidingRopeTheta = 10000.0f;
        int slidingRotaryDim = 128;

        Data softplusALog;
        Data softplusDtBias;

        std::string MapTensorName(const std::string &name) const;
    };
}

#endif
