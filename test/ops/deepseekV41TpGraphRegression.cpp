// Run with test/basic/test_deepseek_v41_tp_graph.py (two CUDA devices).
#include "model.h"
#include "models/deepseekv41.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
struct Access : DeepSeekV41Model { using DeepSeekV41Model::ForwardSegments; };

static std::vector<float> RunRequests(DeepSeekV41Model &model, const std::vector<int> &lengths) {
    std::vector<float> result;
    for (int length : lengths) {
        auto state = std::make_shared<DeepSeekV41RequestState>();
        state->layers.resize(model.block_cnt);
        std::vector<Data> dummy(model.block_cnt * 2);
        std::vector<std::pair<Data *, Data *>> past;
        for (size_t i = 0; i < dummy.size(); i += 2) past.emplace_back(&dummy[i], &dummy[i + 1]);
        for (int step = 0; step <= 12; ++step) {
            DeepSeekV41Segment seg;
            seg.state = state; seg.startPos = state->totalLen;
            seg.seqlen = step == 0 ? length : 1; seg.offset = 0;
            std::vector<float> tokens;
            for (int i = 0; i < seg.seqlen; ++i) tokens.push_back((seg.startPos + i) * 17 % 123 + 3);
            Data ids(DataType::FLOAT32, {1, seg.seqlen}, tokens);
            GenerationConfig config; config.output_logits = true;
            LastTokensManager last(1, 64);
            std::vector<float> logits;
            std::vector<std::vector<float> *> outputs{&logits};
            (model.*(&Access::ForwardSegments))({seg}, ids, nullptr, nullptr, {config}, last, &outputs, past);
            if (logits.size() != 256) throw std::runtime_error("fixture vocabulary mismatch");
            result.insert(result.end(), logits.begin(), logits.end());
        }
    }
    return result;
}

static void Compare(const std::vector<float> &expected, const std::vector<float> &actual) {
    if (actual.size() != expected.size()) throw std::runtime_error("missing request logits");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 1e-5f) {
            std::cerr << "logit " << i << ": " << expected[i] << " vs " << actual[i] << '\n';
            throw std::runtime_error("TP decode differs from eager after warmup");
        }
    }
}

int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) return 77;
    try {
        if (argc != 2) throw std::runtime_error("usage: deepseekV41TpGraphRegression FIXTURE_DIR");
        unsetenv("FASTLLM_DSV41_CUDA_GRAPH");
        setenv("FASTLLM_DSV41_DISABLE_SHARED_OVERLAP", "1", 1);
        SetCudaGraph(false);
        SetThreads(2);
        SetCudaSharedExpert(true);
        SetDeviceMap({{"multicuda:0,1", 1}});
        SetMoeDeviceMap({{"cpu", 1}});
        auto base = CreateLLMModelFromHF(argv[1], DataType::FLOAT16);
        auto &model = dynamic_cast<DeepSeekV41Model &>(*base);
        // Initialize weight shards and kernel plans before comparing dispatch.
        RunRequests(model, {7});
        const auto expected = RunRequests(model, {7, 17, 33, 7});
        unsetenv("FASTLLM_DSV41_DISABLE_SHARED_OVERLAP");
        Compare(expected, RunRequests(model, {7, 17, 33, 7}));
        SetCudaGraph(true);
        Compare(expected, RunRequests(model, {7, 17, 33, 7}));
        SetCudaGraph(false);
        Compare(expected, RunRequests(model, {7, 17, 33, 7}));
        SetCudaGraph(true);
        Compare(expected, RunRequests(model, {7, 17, 33, 7}));
        std::cout << "PASS: eager/graph switching and shared overlap match across 4 requests, 52 steps each\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
