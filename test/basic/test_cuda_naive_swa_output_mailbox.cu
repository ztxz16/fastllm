#include "../../src/devices/multicuda/fastllm-attention-output-rows.cuh"
#include "../../src/devices/multicuda/fastllm-swa-layer.cuh"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace fastllm_swa_output;
void Check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class F> void AllRanks(F fn) {
    std::array<std::exception_ptr, Ranks> errors{};
    std::vector<std::thread> threads;
    for (int r = 0; r < Ranks; ++r) threads.emplace_back([&, r] {
        try { Check(cudaSetDevice(r)); fn(r); } catch (...) { errors[r] = std::current_exception(); }
    });
    for (auto &t : threads) t.join();
    for (auto &e : errors) if (e) std::rethrow_exception(e);
}
struct Storage {
    BF16 *input{}, *weight{}, *output{}, *hidden{};
    uint64_t *epochs{};
    float *normWeight{}, *routerWeight{}, *router{};
    BF16 *normed{};
};
}

// Exercise the real communication kernels near generation wrap, including
// dormant rows carrying a tag from a much older invocation. No model required.
void TestAttentionOutputMailboxWrap(bool mma) {
    Peers peers;
    std::array<Storage, Ranks> data{};
    constexpr size_t bytes = Frames * FrameBytes;
    constexpr size_t epochBytes = MaxRows * MaxBlocks * sizeof(uint64_t);
    auto release = [&] {
        for (int r = 0; r < Ranks; ++r) {
            cudaSetDevice(r); auto &d = data[r];
            cudaFree(d.input); cudaFree(d.weight); cudaFree(d.output); cudaFree(d.hidden);
            cudaFree(d.epochs); cudaFree(peers.data[r]);
            cudaFree(d.normWeight); cudaFree(d.routerWeight); cudaFree(d.router); cudaFree(d.normed);
        }
    };
    try {
        AllRanks([&](int r) {
            auto &d = data[r];
            Check(cudaMalloc(&d.input, MaxRows * Inner * sizeof(BF16)));
            Check(cudaMalloc(&d.weight, Width * Inner * sizeof(BF16)));
            Check(cudaMalloc(&d.output, MaxRows * Width * sizeof(BF16)));
            Check(cudaMalloc(&d.hidden, MaxRows * Width * sizeof(BF16)));
            Check(cudaMalloc(&d.normWeight, Width * sizeof(float)));
            Check(cudaMalloc(&d.routerWeight, 256 * Width * sizeof(float)));
            Check(cudaMalloc(&d.router, MaxRows * 256 * sizeof(float)));
            Check(cudaMalloc(&d.normed, MaxRows * Width * sizeof(BF16)));
            std::vector<float> norm(Width, 1.f), router(256 * Width, 1.f / Width);
            Check(cudaMemcpy(d.normWeight, norm.data(), norm.size() * sizeof(float), cudaMemcpyHostToDevice));
            Check(cudaMemcpy(d.routerWeight, router.data(), router.size() * sizeof(float), cudaMemcpyHostToDevice));
            Check(cudaMalloc(&d.epochs, epochBytes)); Check(cudaMalloc(&peers.data[r], bytes));
            Check(cudaMemset(d.epochs, 0, epochBytes)); Check(cudaMemset(peers.data[r], 0, bytes));
            std::vector<BF16> w(Width * Inner, __float2bfloat16_rn(float(r + 1) / Inner));
            Check(cudaMemcpy(d.weight, w.data(), w.size() * sizeof(BF16), cudaMemcpyHostToDevice));
            Check(cudaDeviceSynchronize());
        });
        auto run = [&](int rows, int seed) {
            AllRanks([&](int r) {
                auto &d = data[r]; std::vector<BF16> x(MaxRows * Inner);
                for (int i = 0; i < MaxRows; ++i)
                    std::fill(x.begin() + i * Inner, x.begin() + (i + 1) * Inner,
                        __float2bfloat16_rn(float((i + seed) % 8 + 1) / 16));
                Check(cudaMemcpy(d.input, x.data(), x.size() * sizeof(BF16), cudaMemcpyHostToDevice));
                Check(cudaMemset(d.hidden, 0, MaxRows * Width * sizeof(BF16)));
                if (rows == 8 && !mma) {
                    // Production keeps the original eight-row GEMM. Its local
                    // result is exact for these constant weights/inputs.
                    std::vector<BF16> local(rows * Width);
                    for (int i = 0; i < rows * Width; ++i)
                        local[i] = __float2bfloat16_rn((r + 1) * float(x[i / Width * Inner]));
                    Check(cudaMemcpy(d.output, local.data(), local.size() * sizeof(BF16), cudaMemcpyHostToDevice));
                }
                if (r == Ranks - 1) std::this_thread::sleep_for(std::chrono::milliseconds(20));
                void *args[] = {&d.input, &d.weight, &d.output, &d.hidden, &peers, &d.epochs, &r, &rows};
                const void *kernel = rows == 1 ? (const void *)fastllm_swa_layer::OutputResidual
                                               : OutputRowsKernel(rows, mma);
                Check(cudaLaunchCooperativeKernel(kernel, dim3(MaxBlocks), dim3(RowThreads), args, 0, cudaStreamPerThread));
                NormRows<<<rows, 256, 0, cudaStreamPerThread>>>(d.hidden, d.normWeight, d.normed, 1e-6f);
                NormRowsRouter<<<dim3(256, (rows + 1) / 2), 256, 0, cudaStreamPerThread>>>(
                    d.hidden, d.routerWeight, d.router, d.normWeight, d.normed, 1e-6f, rows);
                Check(cudaGetLastError()); Check(cudaDeviceSynchronize());
                std::vector<BF16> out(rows * Width);
                Check(cudaMemcpy(out.data(), d.output, out.size() * sizeof(BF16), cudaMemcpyDeviceToHost));
                for (int i = 0; i < rows * Width; ++i) {
                    const float expected = 36.f * float((i / Width + seed) % 8 + 1) / 16;
                    if (float(out[i]) != expected) throw std::runtime_error("mailbox consumed a stale row at generation wrap");
                }
                Check(cudaMemcpy(out.data(), d.normed, out.size() * sizeof(BF16), cudaMemcpyDeviceToHost));
                for (BF16 value : out)
                    if (float(value) != 1.f) throw std::runtime_error("row norm mismatch");
                std::vector<float> router(rows * 256);
                Check(cudaMemcpy(router.data(), d.router, router.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (float value : router)
                    if (!std::isfinite(value) || std::abs(value - 1.f / (1.f + std::exp(-1.f))) > 1e-6f)
                        throw std::runtime_error("row router mismatch");
            });
        };
        for (int rows = 2; rows <= 8; ++rows) run(rows, rows);
        AllRanks([&](int r) {
            Check(cudaMemset(data[r].epochs, 0, epochBytes));
            Check(cudaMemset(peers.data[r], 0, bytes)); Check(cudaDeviceSynchronize());
        });
        run(8, 0); // Dormant rows retain tag 1 in frame 0.
        auto nearWrap = [&](bool allRows) {
            AllRanks([&](int r) {
                auto &d = data[r]; std::vector<uint64_t> epochs(MaxRows * MaxBlocks);
                Check(cudaMemcpy(epochs.data(), d.epochs, epochBytes, cudaMemcpyDeviceToHost));
                const int rows = allRows ? MaxRows : 1;
                std::fill(epochs.begin(), epochs.begin() + rows * MaxBlocks, Period - 2);
                Check(cudaMemcpy(d.epochs, epochs.data(), epochBytes, cudaMemcpyHostToDevice));
                // Active rows' recent frames would have been overwritten while
                // reaching this generation. Keep dormant rows' old publications.
                for (int f = 0; f < Frames; ++f)
                    Check(cudaMemset(peers.data[r] + f * FrameBytes, 0, rows * Ranks * SlotBytes));
                Check(cudaDeviceSynchronize());
            });
        };
        nearWrap(false); run(1, 1); run(1, 2); run(8, 3);
        AllRanks([&](int r) {
            std::vector<uint64_t> epochs(MaxRows * MaxBlocks);
            Check(cudaMemcpy(epochs.data(), data[r].epochs, epochBytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < MaxRows * MaxBlocks; ++i)
                if (uint32_t(epochs[i]) != (i < MaxBlocks ? 1u : 2u))
                    throw std::runtime_error("row generations did not advance independently");
        });
        nearWrap(true); run(8, 4); run(8, 5); run(8, 6);
        release();
        std::printf("Attention output PASS: dormant-row and all-row generation wrap\n");
    } catch (...) { release(); throw; }
}
