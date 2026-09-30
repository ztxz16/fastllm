// Run against the baseline and candidate libraries in separate processes.
// Baseline output files are independent of the candidate dispatch and also
// verify that untuned row counts and shapes keep their existing results.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-nvfp4-fused.h"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Run(int gpu, int n, int k, bool fused, bool writeReference,
                const std::string &folder, bool quick, bool timing) {
    const float global = 0.000371f;
    const int groups = k / 16;
    std::vector<uint8_t> raw(size_t(n) * groups * 12);
    for (int row = 0; row < n; ++row) {
        for (int group = 0; group < groups; ++group) {
            auto *p = raw.data() + (size_t(row) * groups + group) * 12;
            for (int j = 0; j < 8; ++j) {
                p[j] = uint8_t((row * 31 + group * 43 + j * 59 + (row * group) % 113) & 255);
            }
            __nv_fp8_e4m3 scale;
            scale.__x = uint8_t((row + group * 7) % 127);
            float effective = global * float(scale);
            memcpy(p + 8, &effective, sizeof(effective));
        }
    }
    Data weight(NVFP4_BLOCK_16, {n, k});
    weight.blockM = 16;
    weight.blockK = 1;
    weight.weightType = WeightType::LINEAR;
    weight.scales = {global};
    weight.Allocate();
    memcpy(weight.cpuData, raw.data(), raw.size());
    weight.ToDevice(CUDA, std::vector<int>{gpu});
    Data empty;
    const std::vector<int> rows = quick ? std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8} :
        std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 65, 66, 67, 68, 69, 70, 71, 72};
    for (int m : rows) {
        for (int seed = 0; seed < (quick ? 1 : 3); ++seed) {
            std::vector<float> values(size_t(m) * k);
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = 0.15f * sinf(float(i) * 0.017f + seed * 0.7f);
            }
            const int out = fused ? n / 2 : n;
            Data input(FLOAT16, {m, k}, values);
            Data output(FLOAT16, {m, out}, std::vector<float>(size_t(m) * out, 0));
            Data middle(FLOAT16);
            middle.dataDevice = CUDA;
            middle.dataDeviceIds = {gpu};
            input.ToDevice(CUDA, std::vector<int>{gpu});
            output.ToDevice(CUDA, std::vector<int>{gpu});
            FastllmCudaSetNcclForceSync(true);
            auto launch = [&]() {
                if (fused) {
                    CudaNvfp4LinearSwigluBlock(input, weight, empty, middle, output);
                } else {
                    Require(FastllmCudaTryMarlinHalfMatMulFloatNVFP4Block16(
                        input, weight, empty, output, m, k, n), "NVFP4 projection unavailable");
                }
            };
            launch();
            Check(cudaDeviceSynchronize());
            FastllmCudaSetNcclForceSync(false);
            launch();
            Check(cudaGetLastError());
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<half> actual(size_t(m) * out), reference(actual.size());
            Check(cudaMemcpy(actual.data(), output.cudaData, actual.size() * sizeof(half), cudaMemcpyDeviceToHost));
            std::string name = folder + "/g" + std::to_string(gpu) + "-m" + std::to_string(m) +
                "-n" + std::to_string(n) + "-k" + std::to_string(k) + "-f" + std::to_string(fused) +
                "-s" + std::to_string(seed) + ".bin";
            int different = 0, failures = 0;
            float maxAbs = 0;
            if (writeReference) {
                std::ofstream file(name, std::ios::binary);
                file.write(reinterpret_cast<const char *>(actual.data()), actual.size() * sizeof(half));
                Require(bool(file), "Cannot write reference");
            } else {
                std::ifstream file(name, std::ios::binary);
                file.read(reinterpret_cast<char *>(reference.data()), reference.size() * sizeof(half));
                Require(file.gcount() == std::streamsize(reference.size() * sizeof(half)), "Missing/truncated reference");
                const char *mode = std::getenv("FASTLLM_CUDA_NVFP4_SM75_DECODE_TUNE");
                const bool linear = !mode || !std::strcmp(mode, "1") || !std::strcmp(mode, "linear");
                const bool swiglu = !mode || !std::strcmp(mode, "1") || !std::strcmp(mode, "swiglu");
                const bool tunedReduction =
                    (linear && m >= 1 && m <= 8 && !fused && n == 5120 && (k == 3072 || k == 8704)) ||
                    (swiglu && m == 1 && fused && n == 17408 && k == 5120);
                for (size_t i = 0; i < actual.size(); ++i) {
                    float a = __half2float(actual[i]), b = __half2float(reference[i]);
                    float error = fabsf(a - b);
                    bool same = memcmp(&actual[i], &reference[i], sizeof(half)) == 0;
                    different += !same;
                    maxAbs = fmaxf(maxAbs, error);
                    failures += !std::isfinite(a) || (tunedReduction ? error > 0.002f + 0.002f * fabsf(b) : !same);
                }
            }
            printf("NVFP4_CASE gpu=%d M=%d N=%d K=%d fused=%d seed=%d reference=%d different=%d max_abs=%.8g failures=%d\n",
                   gpu, m, n, k, int(fused), seed, int(writeReference), different, maxAbs, failures);
            fflush(stdout);
            Require(failures == 0, "NVFP4 output mismatch");
            if (timing) {
                cudaEvent_t start, end;
                Check(cudaEventCreate(&start));
                Check(cudaEventCreate(&end));
                for (int pass = 0; pass < 3; ++pass) {
                    for (int i = 0; i < (pass == 0 ? 1000 : 20); ++i) launch();
                    Check(cudaEventRecord(start, cudaStreamPerThread));
                    for (int i = 0; i < 200; ++i) launch();
                    Check(cudaEventRecord(end, cudaStreamPerThread));
                    Check(cudaEventSynchronize(end));
                    float ms = 0;
                    Check(cudaEventElapsedTime(&ms, start, end));
                    printf("NVFP4_BENCH gpu=%d M=%d N=%d K=%d fused=%d pass=%d us=%.4f\n",
                           gpu, m, n, k, int(fused), pass, ms * 1000 / 200);
                    fflush(stdout);
                }
                Check(cudaEventDestroy(start));
                Check(cudaEventDestroy(end));
            }
        }
    }
}
int main(int argc, char **argv) {
    try {
        Require(argc >= 4, "usage: test GPU write|check REFERENCE_DIR [quick|timing]");
        int gpu = std::atoi(argv[1]);
        bool writeReference = !std::strcmp(argv[2], "write");
        bool timing = argc > 4 && !std::strcmp(argv[4], "timing");
        bool quick = timing || (argc > 4 && !std::strcmp(argv[4], "quick"));
        std::filesystem::create_directories(argv[3]);
        Check(cudaSetDevice(gpu));
        Run(gpu, 17408, 5120, true, writeReference, argv[3], quick, timing);
        Run(gpu, 5120, 8704, false, writeReference, argv[3], quick, timing);
        Run(gpu, 5120, 3072, false, writeReference, argv[3], quick, timing);
        Run(gpu, 8192, 5120, false, writeReference, argv[3], quick, timing);
        puts("NVFP4_DECODE_TEST_PASS");
        return 0;  // This entry is also renamed for the CUDA-free sanitizer loader.
    } catch (const std::exception &error) {
        fprintf(stderr, "NVFP4_DECODE_TEST_FAIL %s\n", error.what());
        return 1;
    } catch (const char *error) {
        fprintf(stderr, "NVFP4_DECODE_TEST_FAIL %s\n", error);
        return 1;
    }
}
