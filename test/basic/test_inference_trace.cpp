#include "utils/inference_trace.h"
#include <cassert>
#include <stdexcept>
#include <thread>
#include <limits>

using namespace fastllm::inference_trace;

template <typename T> static void CheckArray(const std::vector<T> &values) {
    std::ostringstream expected, actual;
    expected << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) expected << ',';
        expected << values[i];
    }
    expected << ']';
    Array(actual, values);
    if (actual.str() != expected.str()) throw std::runtime_error("trace array format changed");
}

int main() {
    CheckArray<int>({});
    CheckArray<int>({0, -1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()});
    CheckArray<uint64_t>({0, std::numeric_limits<uint64_t>::max()});
    CheckArray<char>({'a', 'b'});
    CheckArray<bool>({false, true});
    CheckArray<float>({-1.25f, 0.f, 1.5f});
    std::vector<int> longArray(4096);
    for (size_t i = 0; i < longArray.size(); ++i) longArray[i] = int(i) * 137 - 200000;
    CheckArray(longArray);
    if (!Enabled()) return 0;
    {
        Step step;
        step.phase = "prefill";
        step.handles = {9};
        step.lengths = {24};
        step.contextBefore = {100};
        step.cachedPrefix = {100};
        step.Mark("prepare_ms");
        for (int chunk = 0; chunk < 2; ++chunk) {
            Forward forward(step, {chunk ? 16 : 8}, {chunk ? 108 : 100});
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            forward.Mark("execute_ms");
        }
        step.Mark("forward_dispatch_ms");
    }
    assert(CurrentStep() == nullptr && CurrentForward() == nullptr);
    {
        Step step;
        step.phase = "decode";
        step.handles = {9, 10};
        step.lengths = {1, 1};
        step.contextBefore = {124, 250};
        step.cachedPrefix = {100, 200};
        {
            Forward forward(step, {1, 1}, {124, 250});
            forward.mode = "cuda_graph";
            forward.ranks = {{0, .2, .01, .3, .001}, {1, .3, .02, .4, .001}};
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            forward.Mark("execute_ms");
        }
        step.Mark("forward_dispatch_ms");
    }
    {
        Step step;
        step.phase = "prefill";
        step.handles = {11};
        step.lengths = {1};
        step.contextBefore = {0};
        step.cachedPrefix = {0};
        try {
            Forward forward(step, {1}, {0});
            throw std::runtime_error("expected failure");
        } catch (const std::runtime_error &) {
            step.failed = true;
        }
    }
    assert(CurrentStep() == nullptr && CurrentForward() == nullptr);
}
