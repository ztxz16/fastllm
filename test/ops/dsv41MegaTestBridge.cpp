// Test-only ABI for an independent PyTorch numerical reference. This file is
// not part of libfastllm and adds no production Python/PyTorch dependency.
#include "devices/cuda/fastllm-dsv41-mega.h"
extern "C" {
size_t mega_test_workspace_bytes() { return FastllmDsv41MegaWorkspaceBytes(); }
void mega_test_init(void *p, const int *devices, int rank) { FastllmDsv41MegaInitWorkspace(p, devices, rank); }
void *mega_test_create(const void *const *gate, const void *const *down, void *const *workspaces, int rank) {
    return FastllmDsv41MegaCreate(gate, down, workspaces, rank);
}
void mega_test_run(FastllmDsv41MegaLayer *p, const void *x, const int32_t *ids,
                   const float *scores, float *out, int rows, void *stream) {
    FastllmDsv41MegaRun(p, x, ids, scores, out, rows, stream);
}
void mega_test_run_rows(FastllmDsv41MegaLayer *p, const void *x, const int32_t *ids,
                        const float *scores, float *out, int rows, void *stream) {
    FastllmDsv41MegaRun(p, x, ids, scores, out, rows, stream, true);
}
void mega_test_shared(FastllmDsv41MegaLayer *p, const void *gate, const uint8_t *gateScales,
                      const void *down, const uint8_t *downScales) {
    FastllmDsv41MegaEnableShared(p, gate, gateScales, down, downScales);
}
void mega_test_run_shared(FastllmDsv41MegaLayer *p, const void *x, const int32_t *ids,
                          const float *scores, void *out, int rows, void *stream) {
    FastllmDsv41MegaRunShared(p, x, ids, scores, out, rows, stream);
}
void mega_test_run_local(FastllmDsv41MegaLayer *p, const void *x, const int32_t *ids,
                         const float *scores, void *out, int rows, void *stream) {
    FastllmDsv41MegaRunLocal(p, x, ids, scores, out, rows, stream);
}
void mega_test_local_post(FastllmDsv41MegaLayer *p, const void *x, const int32_t *ids,
    const float *scores, const void *residual, const float *post, const float *comb,
    void *hidden, int rows, void *stream) {
    FastllmDsv41MegaRunLocalPost(p, x, ids, scores, residual, post, comb, hidden, rows, stream);
}
void mega_test_destroy(FastllmDsv41MegaLayer *p) { FastllmDsv41MegaDestroy(p); }
}
