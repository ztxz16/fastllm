#ifndef FASTLLM_CONTEXTCONFIG_H
#define FASTLLM_CONTEXTCONFIG_H

#include "fastllm.h"
#include "device.h"
#include <string>
#include <vector>

namespace fastllm {
    struct ContextOptions {
        int maxLength = -1;
        std::string ropeScaling;
    };

    // Model classes opt in to a tested layout, rather than matching checkpoint names.
    struct ModelContextSpec {
        bool supported = false;
        bool interleavedMRope = false;
        std::string configPrefix;
    };

    struct RopeConfig {
        std::string type = "default";
        float theta = 10000.0f, factor = 1.0f;
        int originalMaxPosition = 0, rotaryDim = 0;
        float betaFast = 32.0f, betaSlow = 1.0f, attentionFactor = 1.0f;
        float correctionLow = 0.0f, correctionHigh = 1.0f;
        std::vector<int> mropeSections;

        bool IsYarn() const { return type == "yarn"; }
        FloatDict Params() const;
        IntDict PositionParams(bool multimodal) const;
        void AddFusedParams(FloatDict &floats, IntDict &ints) const;
    };

    struct ContextPlan {
        bool configured = false;
        int declaredLength = 0, requestedLength = -1, effectiveLength = 0;
        RopeConfig rope;
        std::string ToJson() const;
    };

    ContextPlan ResolveContextOptions(const std::map<std::string, std::string> &dict, const ModelContextSpec &spec,
                                      const ContextOptions &options);
    void ApplyYarnRope(Data &input, const Data &positions, const RopeConfig &rope);
} // namespace fastllm
#endif
