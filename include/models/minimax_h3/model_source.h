#pragma once
#include <string>
#include <cstdint>

namespace fastllm { namespace minimax_h3 {
struct ModelSource {
    std::string baseDirectory;
    std::string pikaDirectory;
    // Explicit mode binding avoids guessing FL2VA versus Ref2VA from shapes
    // or filenames. Empty entries retain the original BF16 base projections.
    std::string transformerWeights, referenceWeights;
    std::string transformerPrecision = "bf16", referencePrecision = "bf16";
    uint64_t transformerBytes = 0, referenceBytes = 0;
};
struct VideoWeightInfo {
    std::string precision;
    uint64_t bytes = 0;
};
// Bounded local metadata/shape checks; this function does not initialize CUDA.
VideoWeightInfo inspectVideoWeights(const std::string &);
// Resolve local paths only. Pika is identified by its release manifests, never
// by the directory name. An empty pikaDirectory selects the original H3 model.
ModelSource resolveModelSource(const std::string &model, const std::string &base = {},
                               const std::string &legacyPika = {});
// Shared by source discovery and the weight loader; does not initialize CUDA.
void validatePikaManifest(const std::string &directory, bool reference);
}} // namespace fastllm::minimax_h3
