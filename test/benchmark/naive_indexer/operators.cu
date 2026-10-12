#include "naive-n05-indexer-batch.cuh"
#include "baseline-dsa-mma.cuh"
#include "baseline-topk.cuh"
#include <algorithm>
using namespace naive_indexer_batch;
struct Length {
    const int *p;
    __device__ operator int() const { return *p; }
};
extern "C" size_t workspace_size() { return sizeof(naive_topk::Workspace); }
// baseline: 0 = runtime selection, 1 = committed per-row reference,
// 2 = batched fallback scoring with the same quantization and TopK stages.
extern "C" int run(const BF16 *query, const BF16 *weights, const Row *rows, int count,
        int capacity, int baseline, int stage, cudaStream_t stream) {
    if (baseline == 1) {
        for (int i = 0; i < count; ++i) {
            Row r = rows[i];
            if (!stage || stage == 1)
                baseline_naive_dsa_mma::QuantizeIndexer<<<16, 128, 0, stream>>>(
                    query + (size_t)r.sequence * 2048, r.query, r.scale, 128, 0);
            if (!stage || stage == 2)
                baseline_naive_dsa_mma::IndexerDecodeScores<<<(capacity + 63) / 64, 256, 0, stream>>>(
                    r.query, r.key, r.scale, weights + (size_t)r.sequence * 16,
                    ScoreOutput{r.scores}, r.stride, Length{r.live}, capacity - 1);
            if (!stage || stage == 3) {
                auto *ws = reinterpret_cast<baseline_naive_topk::Workspace*>(r.workspace);
                auto *scores = r.scores;
                Length length{r.live};
                auto *hist = ws->partials; auto *ties = ws->ties;
                auto *state = &ws->state; auto *candidates = &ws->candidates;
                void *args[] = {&scores, &length, &hist, &ties, &state, &candidates};
                auto err = cudaLaunchCooperativeKernel((void*)baseline_naive_topk::Select<unsigned, Length>,
                    64, 256, args, 0, stream);
                if (err) return err;
                baseline_naive_topk::Sort<<<1, 256, 0, stream>>>(candidates, state, r.indices);
            }
        }
    } else {
        int device, major, minor;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        bool hopper = baseline == 0 && major == 9 && minor == 0;
        for (int i = 0; i < count; ++i)
            hopper = hopper && rows[i].stride % 8 == 0 && (uintptr_t)rows[i].key % 16 == 0;
        for (int first = 0; first < count; first += BatchLimit) {
            int n = std::min(count - first, BatchLimit);
            Batch batch{};
            std::copy_n(rows + first, n, batch.rows);
            if (!stage || stage == 1) Quantize<<<dim3(16, n), 128, 0, stream>>>(query, batch);
            if (!stage || stage == 2) {
                const dim3 grid((capacity + 63) / 64, n);
                if (hopper) HopperScores<<<grid, 128, 0, stream>>>(weights, batch);
                else Scores<<<grid, 256, 0, stream>>>(weights, batch);
            }
            if (!stage || stage == 3) {
                Select<<<n, 256, 0, stream>>>(batch);
                Sort<<<n, 256, 0, stream>>>(batch);
            }
        }
    }
    return cudaGetLastError();
}
