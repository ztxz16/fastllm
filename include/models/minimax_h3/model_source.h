#pragma once
#include <string>

namespace fastllm { namespace minimax_h3 {
struct ModelSource {
    std::string baseDirectory;
    std::string pikaDirectory;
};
// Resolve local paths only. Pika is identified by its release manifests, never
// by the directory name. An empty pikaDirectory selects the original H3 model.
ModelSource resolveModelSource(const std::string &model, const std::string &base = {},
                               const std::string &legacyPika = {});
// Shared by source discovery and the weight loader; does not initialize CUDA.
void validatePikaManifest(const std::string &directory, bool reference);
}} // namespace fastllm::minimax_h3
