#pragma once
#include <cstdlib>
#include <cstring>

namespace fastllm_swa_settings {
inline bool Enabled(const char *name) {
    const char *value = std::getenv(name);
    return !value || std::strcmp(value, "1") == 0;
}
inline bool QkvEnabled() { return Enabled("FASTLLM_NAIVE_SWA_QKV_FUSION"); }
inline bool CoreEnabled() { return Enabled("FASTLLM_NAIVE_SWA_CORE"); }
}
