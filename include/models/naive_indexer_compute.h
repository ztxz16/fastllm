#pragma once

namespace fastllm {
    enum class NaiveIndexerCompute : unsigned char { Auto, FP8, FP8Direct };

    inline NaiveIndexerCompute NaiveIndexerComputeFromFP8(bool enabled) {
        return enabled ? NaiveIndexerCompute::FP8 : NaiveIndexerCompute::Auto;
    }
}
