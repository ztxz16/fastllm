#include "cuda_gguf_data_test.cuh"

static void Linear(Data &x, Data &w, Data &y, int t, int k, int n) {
    Data bias(FLOAT32);
    Check(x.dataType == FLOAT32 ? FastllmCudaMatMulFloatGGUF(x, w, bias, y, t, k, n)
                              : FastllmCudaHalfMatMulGGUF(x, w, bias, y, t, k, n), "linear rejected");
}
template <typename T> static void Test(int t, int k, int n) {
    const DataType dtype = sizeof(T) == 4 ? FLOAT32 : FLOAT16;
    Data x(dtype, {t, k}), w(DATA_GGUF_FORMAT, int(GGML_TYPE_Q6_K), {n, k});
    Data storage(dtype, {t * n + 8}), y(dtype), reference(dtype, {t, n});
    for (Data *d : {&x, &w, &storage, &reference}) Allocate(*d);
    w.strides = {1}; w.forceGGUFFp32Dequant = true;
    Upload(w, Weights(GGML_TYPE_Q6_K, n, k));
    y.FakeFrom(storage, 0); y.Resize({t, n}); y.dataDeviceIds = {0};
    auto prepare = [&](int seed) {
        std::vector<T> input(t * k);
        for (int i = 0; i < t * k; ++i)
            input[i] = T((i % k < 32 || (seed == 0 && t > 1 && i / k == t - 1)) ? 0.f :
                         std::sin(float(i + seed) * .117f) * (1 + i % 7));
        Upload(x, input); Upload(storage, std::vector<T>(t * n + 8, T(42.f)));
        for (int row = 0; row < t; ++row) {
            Data xr(dtype), yr(dtype);
            xr.FakeFrom(x, size_t(row) * k * sizeof(T)); xr.Resize({1, k});
            yr.FakeFrom(reference, size_t(row) * n * sizeof(T)); yr.Resize({1, n});
            xr.dataDeviceIds = yr.dataDeviceIds = {0};
            Linear(xr, w, yr, 1, k, n);
        }
    };
    auto check = [&] {
        std::vector<T> got(t * n + 8), want(t * n);
        Cuda(cudaMemcpy(got.data(), storage.cudaData, got.size() * sizeof(T), cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(want.data(), reference.cudaData, want.size() * sizeof(T), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < want.size(); ++i) {
            // The pre-existing 5..8-row path can round differently from T1.
            // The optimized MTP 2..4-row path must remain bitwise equal.
            if (t <= 4 ? std::memcmp(&got[i], &want[i], sizeof(T)) != 0 :
                std::abs(float(got[i]) - float(want[i])) >
                    (sizeof(T) == 4 ? 2e-5f : .02f) * (1 + std::abs(float(want[i])))) {
                std::cerr << "T=" << t << " K=" << k << " N=" << n << " bytes=" << sizeof(T)
                          << " index=" << i << " got=" << float(got[i]) << " reference=" << float(want[i]) << '\n';
                throw std::runtime_error("multi-row differs from single-row arithmetic");
            }
        }
        for (size_t i = want.size(); i < got.size(); ++i) Check(float(got[i]) == 42, "tail overwritten");
    };
    prepare(0); Linear(x, w, y, t, k, n); Cuda(cudaDeviceSynchronize()); check();
    cudaGraph_t graph; cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Linear(x, w, y, t, k, n);
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int seed : {71, 159}) {
        prepare(seed); Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaDeviceSynchronize()); check();
    }
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));

}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::cout << "SKIP no CUDA device\n"; return 77; }
    Cuda(cudaSetDevice(0)); FastllmCudaSetDevice(0);
    try {
        int cases = 0;
        for (int t = 1; t <= 8; ++t)
            for (int k : {256, 768, 2560, 5120})
                for (int n : {1, 17, 257}) {
                    Test<float>(t, k, n);
                    Test<half>(t, k, n); cases += 2;
                }
        std::cout << "PASS Q6 multi-row cases=" << cases << " with changed-input graph replays\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
