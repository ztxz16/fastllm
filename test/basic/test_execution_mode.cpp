#include "utils/execution_mode.h"
#include <iostream>
#include <stdexcept>

using namespace fastllm;

static void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void Reject(const char *value, const std::vector<std::string> &devices) {
    try {
        ResolveExecutionMode(value, devices);
    } catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("invalid mode/device combination accepted");
}

int main() {
    Check(ResolveExecutionMode(nullptr, {"cuda:0", "cuda:0"}) == ExecutionMode::Gpu, "single GPU");
    Check(ResolveExecutionMode("auto", {"multicuda:0,1", "multicuda:0,1"}) == ExecutionMode::Gpu, "TP GPU");
    Check(ResolveExecutionMode(" GPU ", {"cuda:0", "cuda:1"}) == ExecutionMode::Gpu, "explicit GPU");
    Check(ResolveExecutionMode("auto", {"cuda", "numa"}) == ExecutionMode::Hybrid, "NUMA experts");
    Check(ResolveExecutionMode("hybrid", {"cuda", "cpu"}) == ExecutionMode::Hybrid, "CPU experts");
    Check(ResolveExecutionMode("auto", {"cuda", "disk"}) == ExecutionMode::Hybrid, "disk experts");
    Check(ResolveExecutionMode("hybrid", {"multicuda:0,1", "disk"}) == ExecutionMode::Hybrid, "TP disk experts");
    Check(ResolveExecutionMode("auto", {"cuda", "cuda", "cuda", "numa"}) == ExecutionMode::Hybrid, "layered experts");
    Check(ResolveExecutionMode("auto", {"cpu", "cuda"}) == ExecutionMode::Hybrid, "CPU main");
    Check(ResolveExecutionMode("", {"cpu", "numa"}) == ExecutionMode::Cpu, "CPU fallback");
    Check(ResolveExecutionMode("cpu", {"cpu", "cpu"}) == ExecutionMode::Cpu, "explicit CPU");
    Check(ResolveExecutionMode("auto", {"cpu", "disk"}) == ExecutionMode::Cpu, "CPU disk experts");
    Reject("gup", {"cuda"});
    Reject("gpu", {"cuda", "numa"});
    Reject("gpu", {"cuda", "disk"});
    Reject("hybrid", {"cuda", "cuda"});
    Reject("cpu", {"cuda", "cpu"});
    Reject("auto", {});
    Reject("auto", {"unexpected"});
    // Repeated model construction must not cache an earlier model's mode.
    Check(ResolveExecutionMode("auto", {"cuda", "numa"}) == ExecutionMode::Hybrid, "first model");
    Check(ResolveExecutionMode("auto", {"cuda", "cuda"}) == ExecutionMode::Gpu, "second model");
    std::cout << "ALL_PASS: execution mode selection and validation\n";
}
