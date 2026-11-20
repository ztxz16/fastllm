#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

namespace fastllm { class Data; }

struct FastllmNvfp4ScaleRun {
    int endRow;
    float scale;
};
using FastllmNvfp4ScaleRuns = std::vector<FastllmNvfp4ScaleRun>;
using FastllmNvfp4PrefillPlan = std::unordered_map<const fastllm::Data *, FastllmNvfp4ScaleRuns>;

class FastllmCompactMarlinCache;
std::shared_ptr<FastllmCompactMarlinCache> FastllmCreateCompactMarlinCache();
bool FastllmCudaCompactMarlinLinear(FastllmCompactMarlinCache &cache, const FastllmNvfp4PrefillPlan &plan,
        const fastllm::Data &weight, const void *input, void *output, int rows, int columns, int inner);

// The original layout can be converted in place; compact BF16 keeps a second
// copy for decode. Callers supply their default when the shared switch is unset.
bool FastllmCudaNvfp4MarlinEnabled(bool defaultEnabled);
void FastllmCudaPlanNvfp4Prefill(FastllmNvfp4PrefillPlan &plan, const fastllm::Data &weight);

// Forward-local compact NVFP4 policy. Nested target verification restores the
// draft's policy on return. Equal-length prompts share each unpacked panel.
class FastllmCudaNvfp4Scope {
    const FastllmCudaNvfp4Scope *previous;
public:
    const FastllmNvfp4PrefillPlan &plan;
    const bool batched;
    const int promptRows;
    FastllmCompactMarlinCache *const marlin;

    explicit FastllmCudaNvfp4Scope(const FastllmNvfp4PrefillPlan &plan, bool batched = false,
                                   int promptRows = 0, FastllmCompactMarlinCache *marlin = nullptr);
    ~FastllmCudaNvfp4Scope();
    static bool IsBatched();
    FastllmCudaNvfp4Scope(const FastllmCudaNvfp4Scope &) = delete;
    FastllmCudaNvfp4Scope &operator=(const FastllmCudaNvfp4Scope &) = delete;
};
