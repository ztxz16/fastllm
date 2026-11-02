#include "executor.h"
#include "fastllm.h"
#include "models/deepseekv41.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
struct HeadModel : DeepSeekV41Model {
    using DeepSeekV41Model::DsparkProjectHead;
    using DeepSeekV41Model::TryBFloat16HeadLinear;
};

int main() {
    try {
        SetThreads(4);
        Executor executor;
        executor.SetFirstDevice("cpu");
        SetCurrentThreadExecutor(&executor);
        HeadModel model;
        constexpr int width = 32, vocab = 17;
        Data weights(FLOAT32, {vocab, width});
        weights.Allocate();
        auto *w = reinterpret_cast<float *>(weights.cpuData);
        for (int i = 0; i < vocab * width; ++i)
            w[i] = float(i % 23 - 11) / 32;
        ToDataType(weights, model.weight["head.weight"], BFLOAT16);
        for (int rows : {1, 8})
            for (DataType dtype : {BFLOAT16, FLOAT32}) {
                Data values(FLOAT32, {rows, width}), input, output, draft;
                values.Allocate();
                auto *x = reinterpret_cast<float *>(values.cpuData);
                // Non-BF16 values expose accidental down-conversion on the FP32 path.
                for (int i = 0; i < rows * width; ++i)
                    x[i] = float(i % 19 - 9) / 37;
                ToDataType(values, input, dtype);
                if (!model.TryBFloat16HeadLinear(input, output))
                    throw std::runtime_error("compact head fallback rejected");
                model.DsparkProjectHead(input, draft);
                Data actualInput;
                ToDataType(input, actualInput, FLOAT32);
                const auto *a = reinterpret_cast<const float *>(actualInput.cpuData);
                for (Data *result : {&output, &draft}) {
                    if (result->dataType != FLOAT32 || result->dims != std::vector<int>{rows, vocab})
                        throw std::runtime_error("head/draft logits must remain FP32");
                    const auto *y = reinterpret_cast<const float *>(result->cpuData);
                    for (int r = 0; r < rows; ++r)
                        for (int n = 0; n < vocab; ++n) {
                            double expected = 0;
                            for (int k = 0; k < width; ++k)
                                expected += double(a[r * width + k]) * w[n * width + k];
                            if (!std::isfinite(y[r * vocab + n]) ||
                                std::abs(y[r * vocab + n] - expected) > 1e-6)
                                throw std::runtime_error("compact head fallback numerical mismatch");
                        }
                }
            }
        SetCurrentThreadExecutor(nullptr);
        std::cout << "PASS: CPU compact head and draft preserve FP32 inputs, accumulation and logits\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
