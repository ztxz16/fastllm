#pragma once

// Opt in on the model's hybrid execution thread. Nested scopes restore the
// caller's policy; GPU-only forwards and decode retain their existing kernels.
bool FastllmCudaSetHybridPrefill(bool enabled);

class FastllmCudaHybridPrefillScope {
    bool previous;
public:
    explicit FastllmCudaHybridPrefillScope(bool enabled)
        : previous(FastllmCudaSetHybridPrefill(enabled)) {}
    ~FastllmCudaHybridPrefillScope() { FastllmCudaSetHybridPrefill(previous); }
    FastllmCudaHybridPrefillScope(const FastllmCudaHybridPrefillScope &) = delete;
    FastllmCudaHybridPrefillScope &operator=(const FastllmCudaHybridPrefillScope &) = delete;
};
