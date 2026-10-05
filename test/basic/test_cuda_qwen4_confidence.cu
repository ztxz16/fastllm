#include "cuda_gguf_data_test.cuh"
#include "models/qwen4_tp_sampling.h"
#include <limits>
#include <random>

static double maxError = 0, maxOldError = 0;
static int cases = 0;
static float CheckRow(const std::vector<float> &v) {
    const int n = v.size();
    const float maximum = *std::max_element(v.begin(), v.end());
    Data x(FLOAT32, {1, n}), y(FLOAT32, {1}), softmax(FLOAT32, {1, n}), top(FLOAT32, {1, 2});
    for (Data *d : {&x, &y, &softmax, &top}) Allocate(*d);
    Upload(x, v);
    Check(FastllmCudaQwen4TopProbability((float *)x.cudaData, (float *)y.cudaData, n, maximum), "confidence rejected");
    Check(FastllmCudaSoftmax(x, softmax, -1), "softmax rejected");
    Check(FastllmCudaTopK(softmax, top, 1), "topk rejected");
    float got, old[2];
    Cuda(cudaMemcpy(&got, y.cudaData, sizeof(float), cudaMemcpyDeviceToHost));
    Cuda(cudaMemcpy(old, top.cudaData, sizeof(old), cudaMemcpyDeviceToHost));
    double sum = 0;
    for (float x : v) sum += std::exp(double(x) - maximum);
    const double expected = std::isfinite(maximum) && std::isfinite(1 / sum) ? 1 / sum : 0;
    Check(std::isfinite(got) && got >= 0 && got <= 1, "invalid confidence probability");
    maxError = std::max(maxError, std::abs(got - expected));
    if (std::isfinite(old[1])) maxOldError = std::max(maxOldError, double(std::abs(old[1] - got)));
    if (std::abs(got - expected) > 2e-6 * expected + 2e-7) {
        std::cerr << "vocabulary=" << n << " maximum=" << maximum << " got=" << got
                  << " expected=" << expected << " old=" << old[1] << '\n';
        throw std::runtime_error("confidence differs from FP64 reference");
    }
    if (std::abs(expected - .4) > 2e-6) Check((got >= .4f) == (expected >= .4), "threshold decision differs");
    if (std::isfinite(old[1]) && std::abs(expected - .4) > 2e-6)
        Check((got >= .4f) == (old[1] >= .4f), "old threshold decision differs");
    ++cases; return got;
}
static void CheckGraphReplay() {
    const int n = 248320;
    Data x(FLOAT32, {1, n}), y(FLOAT32, {1});
    for (Data *d : {&x, &y}) Allocate(*d);
    auto run = [&] { Check(FastllmCudaQwen4TopProbability((float *)x.cudaData, (float *)y.cudaData, n, 0), "confidence rejected"); };
    cudaGraph_t graph; cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); run();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (float p : {.2f, .6f, .9f}) {
        std::vector<float> v(n, std::log((1.0 / p - 1) / (n - 1))); v[0] = 0; Upload(x, v);
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread)); Cuda(cudaDeviceSynchronize());
        float got; Cuda(cudaMemcpy(&got, y.cudaData, sizeof(float), cudaMemcpyDeviceToHost));
        Check(std::abs(got - p) < 2e-6, "graph failed to reread logits");
    }
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));

}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::cout << "SKIP no CUDA device\n"; return 77; }
    Cuda(cudaSetDevice(0)); FastllmCudaSetDevice(0);
    try {
        std::mt19937 random(19381); std::uniform_real_distribution<float> dist(-30, 0);
        for (int n : {1, 31, 32, 33, 255, 1024, 4097, 124160, 248320}) {
            for (float shift : {-1000.f, 0.f, 1000.f}) {
                std::vector<float> v(n, shift); CheckRow(v);
                for (auto &x : v) x += dist(random);
                CheckRow(v); v[n / 2] += 40; CheckRow(v);
                v[0] = -std::numeric_limits<float>::infinity(); CheckRow(v);
            }
            if (n > 2) for (float p : {.1f, .39999f, .40001f, .8f, .9999f}) {
                std::vector<float> v(n, std::log((1.0 / p - 1) / (n - 1))); v[0] = 0; CheckRow(v);
            }
        }
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        for (const auto &v : std::vector<std::vector<float>>{{-inf, -inf}, {inf, 1}, {0, nan}}) CheckRow(v);
        std::vector<float> a(124160), b(124159);
        for (auto &v : a) v = dist(random);
        for (auto &v : b) v = dist(random) + 3;
        const float pa = CheckRow(a), pb = CheckRow(b);
        const float ma = *std::max_element(a.begin(), a.end()), mb = *std::max_element(b.begin(), b.end());
        a.insert(a.end(), b.begin(), b.end());
        Check(std::abs(qwen4_tp::MergeTopProbability({{ma, pa}, {mb, pb}}) - CheckRow(a)) < 2e-7, "TP merge mismatch");
        Check(!FastllmCudaQwen4TopProbability(nullptr, nullptr, 0, 0), "invalid arguments accepted");
        CheckGraphReplay();
        std::cout << "PASS MTP confidence cases=" << cases << " max_fp64_error=" << maxError << " max_old_error=" << maxOldError << '\n';
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
