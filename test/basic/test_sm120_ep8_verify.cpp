#include "devices/cuda/fastllm-mxfp4-ep-generic.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> T *Upload(const std::vector<T> &host) {
    T *p;
    Check(cudaMalloc(reinterpret_cast<void **>(&p), host.size() * sizeof(T)));
    Check(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
constexpr int Width = 4096, Rows = 8, K = 8;
using Output = std::vector<uint16_t>;
struct RankInputs {
    uint16_t *input, *output;
    int *ids, *mask;
    float *scores;
};
} // namespace

int main() {
    try {
        std::vector<int> devices{0, 1, 2, 3, 4, 5, 6, 7};
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8 || !FastllmGenericEpTunedEp8(devices)) return 77;
        for (int src : devices) for (int dst : devices) if (src != dst) {
            int peer;
            Check(cudaDeviceCanAccessPeer(&peer, src, dst));
            if (!peer) return 77;
        }
        auto ep = FastllmCreateSharedEp8Decode(devices);
        std::array<RankInputs, 8> ranks;
        for (int rank : devices) {
            Check(cudaSetDevice(rank));
            std::vector<uint64_t> weights(64);
            for (int part : {0, 1}) {
                const size_t elements = size_t(Width) * (part ? 2048 : 4096);
                std::vector<uint8_t> image(elements / 2 + elements / 32, 118);
                std::fill_n(image.begin(), elements / 2, 0x22 + rank % 2);
                auto *w = Upload(image);
                for (int e = 0; e < 32; ++e) weights[2 * e + part] = reinterpret_cast<uint64_t>(w);
            }
            FastllmBindGenericEp(*ep, rank, 0, weights);
            Check(cudaFree(reinterpret_cast<void *>(weights[0])));
            Check(cudaFree(reinterpret_cast<void *>(weights[1])));
            std::vector<int> ids(Rows * K);
            for (int i = 0; i < Rows * K; ++i) ids[i] = ((rank + i) % 8) * 32 + i / K;
            ranks[rank] = {Upload(Output(Rows * Width, 0x3e80 + rank)), Upload(Output(Rows * Width)),
                Upload(ids), Upload(std::vector<int>{0, 1, 2, 0, 4, 5, 0, 7}), Upload(std::vector<float>(Rows * K, .125f))};
        }
        for (bool masked : {false, true}) {
            auto run = [&](int capacity) {
                std::array<Output, 8> result;
                for (int rank : devices) {
                    Check(cudaSetDevice(rank));
                    auto &r = ranks[rank];
                    const int rows = rank == 7 ? 8 : rank;
                    Check(cudaMemsetAsync(r.output, 0xa5, Rows * Width * 2, cudaStreamPerThread));
                    FastllmRunGenericEp(*ep, rank, 0, rows ? r.input : nullptr, rows ? r.ids : nullptr,
                        rows ? r.scores : nullptr, rows ? r.output : nullptr, rows, capacity,
                        rows && masked ? r.mask : nullptr);
                }
                for (int rank : devices) {
                    Check(cudaSetDevice(rank));
                    Check(cudaStreamSynchronize(cudaStreamPerThread));
                    result[rank].resize(Rows * Width);
                    Check(cudaMemcpy(result[rank].data(), ranks[rank].output, Rows * Width * 2, cudaMemcpyDeviceToHost));
                }
                return result;
            };
            // Nine slots use the multi-kernel path with the same eight slots of real work.
            const auto reference = run(9), actual = run(8);
            if (actual != reference || !actual[7][Width] || actual[7][Width] == 0xa5a5 ||
                (masked && actual[7][0]) || actual[0][0] != 0xa5a5)
                throw std::runtime_error("EP8 verify disagrees with the unfused path");
        }
        for (int rank : devices) {
            Check(cudaSetDevice(rank));
            const auto &r = ranks[rank];
            for (void *p : std::array<void *, 5>{r.input, r.output, r.ids, r.mask, r.scores}) Check(cudaFree(p));
        }
        std::puts("EP8_VERIFY_PASS masked and unmasked, empty and uneven ranks");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
