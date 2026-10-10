#pragma once

// Immutable cache addresses are captured with the graph. Lengths are read on
// the device at replay time; eager callers can instead supply a host length.
struct FastllmNaiveDSARow {
    const void *key = nullptr, *value = nullptr;
    const int *indices = nullptr, *liveKeys = nullptr;
    int length = 0, capacity = 0, keyStride = 0, selectedCount = 0;
};

constexpr int FastllmNaiveDSABatchLimit = 32;
struct FastllmNaiveDSABatch {
    FastllmNaiveDSARow rows[FastllmNaiveDSABatchLimit];
};
