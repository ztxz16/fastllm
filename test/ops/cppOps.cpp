#include "fastllm.h"
#include "executor.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace {
    using Clock = std::chrono::high_resolution_clock;

    struct ParamSpec {
        std::string name;
        std::string defaultValue;
        std::string description;
    };

    class OpTestParams {
    public:
        void Add(const std::string &name, const std::string &defaultValue, const std::string &description) {
            if (values.find(name) == values.end()) {
                values[name] = defaultValue;
            }
            specs.push_back({name, defaultValue, description});
        }

        void Override(const std::string &name, const std::string &value) {
            values[name] = value;
        }

        bool Has(const std::string &name) const {
            return values.find(name) != values.end();
        }

        std::string GetString(const std::string &name) const {
            auto it = values.find(name);
            if (it == values.end()) {
                throw std::runtime_error("missing param: " + name);
            }
            return it->second;
        }

        int GetInt(const std::string &name) const {
            return std::stoi(GetString(name));
        }

        float GetFloat(const std::string &name) const {
            return std::stof(GetString(name));
        }

        std::vector<int> GetInts(const std::string &name) const {
            std::vector<int> result;
            std::stringstream ss(GetString(name));
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) {
                    result.push_back(std::stoi(token));
                }
            }
            return result;
        }

        void Print(std::ostream &os) const {
            for (const auto &spec : specs) {
                auto it = values.find(spec.name);
                os << "  " << spec.name << "=" << (it == values.end() ? spec.defaultValue : it->second);
                if (!spec.description.empty()) {
                    os << "  # " << spec.description;
                }
                os << "\n";
            }
        }

    private:
        std::vector<ParamSpec> specs;
        std::map<std::string, std::string> values;
    };

    struct CliConfig {
        std::string op = "all";
        std::vector<std::string> deviceFilters;
        std::vector<std::pair<std::string, std::string>> paramOverrides;
        int warmup = 2;
        int iters = 10;
        float atol = 1e-4f;
        float rtol = 1e-4f;
        bool listOps = false;
        bool help = false;
    };

    struct ComparisonStats {
        float maxAbsDiff = 0.0f;
        float maxRelDiff = 0.0f;
        size_t mismatchIndex = 0;
        float expected = 0.0f;
        float actual = 0.0f;
    };

    struct BenchmarkResult {
        double avgMs = 0.0;
        double bytesMoved = 0.0;
        double flops = 0.0;
        double bandwidthGBps = -1.0;
        double computeTFlops = -1.0;
    };

    static std::string Trim(const std::string &s) {
        size_t l = 0, r = s.size();
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            l++;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            r--;
        }
        return s.substr(l, r - l);
    }

    static std::vector<std::string> Split(const std::string &s, char sep) {
        std::vector<std::string> result;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, sep)) {
            item = Trim(item);
            if (!item.empty()) {
                result.push_back(item);
            }
        }
        return result;
    }

    static fastllm::Data MakeTensor(const std::vector<int> &dims, float seed = 0.0f, float scale = 1.0f) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        std::vector<float> data(count);
        for (int i = 0; i < count; i++) {
            float v = std::sin((i + 1) * 0.37f + seed) + std::cos((i + 3) * 0.19f + seed * 0.5f);
            data[i] = v * scale;
        }
        return fastllm::Data(fastllm::DataType::FLOAT32, dims, data);
    }

    static fastllm::Data MakeRampTensor(const std::vector<int> &dims, float seed = 0.0f) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        std::vector<float> data(count);
        for (int i = 0; i < count; i++) {
            data[i] = seed + (float) i / std::max(1, count);
        }
        return fastllm::Data(fastllm::DataType::FLOAT32, dims, data);
    }

    static int CountElements(const std::vector<int> &dims) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        return count;
    }

    static double FloatBytes(const std::vector<int> &dims) {
        return (double) CountElements(dims) * (double) sizeof(float);
    }

    static std::string FormatComputeSpeed(double computeTops) {
        std::ostringstream os;
        if (computeTops < 1.0) {
            os << std::fixed << std::setprecision(4) << computeTops * 1000.0 << " GOPS";
        } else {
            os << std::fixed << std::setprecision(4) << computeTops << " TOPS";
        }
        return os.str();
    }

    static std::string FormatIOSpeed(double gbps) {
        std::ostringstream os;
        if (gbps < 1.0) {
            os << std::fixed << std::setprecision(4) << gbps * 1000.0 << " MB/s";
        } else {
            os << std::fixed << std::setprecision(4) << gbps << " GB/s";
        }
        return os.str();
    }

    static std::vector<float> ToFloatVector(fastllm::Data data) {
        data.ToDevice(fastllm::DataDevice::CPU);
        size_t count = data.Count(0);
        std::vector<float> result(count);
        if (data.dataType == fastllm::DataType::FLOAT32) {
            const float *ptr = reinterpret_cast<const float*>(data.cpuData);
            for (size_t i = 0; i < count; i++) {
                result[i] = ptr[i];
            }
            return result;
        }
        throw std::runtime_error("only FLOAT32 outputs are supported in optest currently");
    }

    static ComparisonStats CompareData(const fastllm::Data &expectedData, const fastllm::Data &actualData,
                                       float atol, float rtol) {
        fastllm::Data expected(expectedData);
        fastllm::Data actual(actualData);
        expected.ToDevice(fastllm::DataDevice::CPU);
        actual.ToDevice(fastllm::DataDevice::CPU);
        if (expected.dims != actual.dims) {
            throw std::runtime_error("output shape mismatch");
        }
        std::vector<float> expectedVec = ToFloatVector(expected);
        std::vector<float> actualVec = ToFloatVector(actual);

        ComparisonStats stats;
        for (size_t i = 0; i < expectedVec.size(); i++) {
            float absDiff = std::fabs(expectedVec[i] - actualVec[i]);
            float relDiff = absDiff / std::max(std::fabs(expectedVec[i]), 1e-6f);
            if (absDiff > stats.maxAbsDiff) {
                stats.maxAbsDiff = absDiff;
                stats.maxRelDiff = relDiff;
                stats.mismatchIndex = i;
                stats.expected = expectedVec[i];
                stats.actual = actualVec[i];
            }
            if (absDiff > atol + rtol * std::fabs(expectedVec[i])) {
                return stats;
            }
        }
        return stats;
    }

    class ScopedFirstDevice {
    public:
        explicit ScopedFirstDevice(const std::string &device) {
            executor = (fastllm::Executor*) fastllm::GetExecutor();
            previous = executor->firstDevice;
            executor->SetFirstDevice(device);
        }

        ~ScopedFirstDevice() {
            if (!previous.empty()) {
                executor->SetFirstDevice(previous);
            }
        }

    private:
        fastllm::Executor *executor = nullptr;
        std::string previous;
    };

    struct OpCase {
        std::string name;
        std::string description;
        std::function<OpTestParams()> makeDefaultParams;
        std::function<bool(const OpTestParams&, const std::string&)> canRun;
        std::function<fastllm::Data(const OpTestParams&, const std::string&)> run;
        std::function<std::function<void()>(const OpTestParams&, const std::string&)> makeBenchmarkRun = nullptr;
        std::function<double(const OpTestParams&)> GetIOBytes = nullptr;
        std::function<double(const OpTestParams&)> GetComputeOps = nullptr;

        OpCase() = default;

        OpCase(std::string name, std::string description,
               std::function<OpTestParams()> makeDefaultParams,
               std::function<bool(const OpTestParams&, const std::string&)> canRun,
               std::function<fastllm::Data(const OpTestParams&, const std::string&)> run,
               std::function<double(const OpTestParams&)> GetIOBytes,
               std::function<double(const OpTestParams&)> GetComputeOps)
            : name(std::move(name)), description(std::move(description)),
              makeDefaultParams(std::move(makeDefaultParams)), canRun(std::move(canRun)),
              run(std::move(run)), GetIOBytes(std::move(GetIOBytes)),
              GetComputeOps(std::move(GetComputeOps)) {}

        OpCase(std::string name, std::string description,
               std::function<OpTestParams()> makeDefaultParams,
               std::function<bool(const OpTestParams&, const std::string&)> canRun,
               std::function<fastllm::Data(const OpTestParams&, const std::string&)> run,
               std::function<std::function<void()>(const OpTestParams&, const std::string&)> makeBenchmarkRun,
               std::function<double(const OpTestParams&)> GetIOBytes,
               std::function<double(const OpTestParams&)> GetComputeOps)
            : name(std::move(name)), description(std::move(description)),
              makeDefaultParams(std::move(makeDefaultParams)), canRun(std::move(canRun)),
              run(std::move(run)), makeBenchmarkRun(std::move(makeBenchmarkRun)),
              GetIOBytes(std::move(GetIOBytes)), GetComputeOps(std::move(GetComputeOps)) {}
    };

    static bool CanRunOnDevice(const std::string &device, const std::string &opType,
                               const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams,
                               const fastllm::IntDict &intParams) {
        ScopedFirstDevice guard(device);
        return ((fastllm::Executor*) fastllm::GetExecutor())->CanRunOnFirstDevice(opType, datas, floatParams, intParams);
    }

    static void SyncBenchmarkDevice(const std::string &device) {
#ifdef USE_CUDA
        if (device.rfind("cuda", 0) == 0 || device.rfind("multicuda", 0) == 0) {
            ForceDeviceSync();
        }
#endif
    }

    static BenchmarkResult Benchmark(const OpCase &opCase, const OpTestParams &params,
                                     const std::string &device, int warmup, int iters) {
        std::function<void()> benchRun;
        if (opCase.makeBenchmarkRun) {
            benchRun = opCase.makeBenchmarkRun(params, device);
        }
        if (benchRun) {
            for (int i = 0; i < warmup; i++) {
                benchRun();
                SyncBenchmarkDevice(device);
            }
            auto begin = Clock::now();
            for (int i = 0; i < iters; i++) {
                benchRun();
                SyncBenchmarkDevice(device);
            }
            auto end = Clock::now();
            double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();
            BenchmarkResult result;
            result.avgMs = totalMs / std::max(iters, 1);
            if (opCase.GetIOBytes) {
                result.bytesMoved = opCase.GetIOBytes(params);
            }
            if (opCase.GetComputeOps) {
                result.flops = opCase.GetComputeOps(params);
            }
            double seconds = result.avgMs / 1000.0;
            if (seconds > 0.0 && result.bytesMoved > 0.0) {
                result.bandwidthGBps = result.bytesMoved / seconds / 1e9;
            }
            if (seconds > 0.0 && result.flops > 0.0) {
                result.computeTFlops = result.flops / seconds / 1e12;
            }
            return result;
        }

        for (int i = 0; i < warmup; i++) {
            fastllm::Data output = opCase.run(params, device);
            output.ToDevice(fastllm::DataDevice::CPU);
        }
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            fastllm::Data output = opCase.run(params, device);
            output.ToDevice(fastllm::DataDevice::CPU);
        }
        auto end = Clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();
        BenchmarkResult result;
        result.avgMs = totalMs / std::max(iters, 1);
        if (opCase.GetIOBytes) {
            result.bytesMoved = opCase.GetIOBytes(params);
        }
        if (opCase.GetComputeOps) {
            result.flops = opCase.GetComputeOps(params);
        }
        double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0 && result.bytesMoved > 0.0) {
            result.bandwidthGBps = result.bytesMoved / seconds / 1e9;
        }
        if (seconds > 0.0 && result.flops > 0.0) {
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
    }

    static std::vector<std::string> GetAvailableDevices() {
        std::vector<std::string> devices = {"cpu"};

        if (fastllm::HasDeviceType("cuda")) {
#ifdef USE_CUDA
            int cudaDeviceCount = FastllmCudaGetDeviceCount();
            if (cudaDeviceCount > 0) {
                devices.push_back("cuda:0");
            }
            if (fastllm::HasDeviceType("multicuda") && cudaDeviceCount > 1) {
                std::string spec = "multicuda:";
                for (int i = 0; i < cudaDeviceCount; i++) {
                    if (i > 0) {
                        spec += ",";
                    }
                    spec += std::to_string(i);
                }
                devices.push_back(spec);
            }
#else
            devices.push_back("cuda");
#endif
        }
        if (fastllm::HasDeviceType("numa")) {
            devices.push_back("numa");
        }
        if (fastllm::HasDeviceType("numas")) {
            devices.push_back("numas");
        }
        if (fastllm::HasDeviceType("tfacc")) {
            devices.push_back("tfacc");
        }
        if (fastllm::HasDeviceType("tops")) {
            devices.push_back("tops");
        }
        return devices;
    }

    static bool DeviceSelected(const std::vector<std::string> &filters, const std::string &device) {
        if (filters.empty()) {
            return true;
        }
        for (const auto &filter : filters) {
            if (filter == "all" || filter == device) {
                return true;
            }
            if (device.rfind(filter, 0) == 0) {
                return true;
            }
        }
        return false;
    }

    static OpCase MakeAddToCase() {
        return {
            "addto",
            "input0 += input1 * alpha",
            []() {
                OpTestParams params;
                params.Add("dims", "4,8", "input tensor shape");
                params.Add("alpha", "1.5", "scale applied to input1");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input0 = MakeTensor(params.GetInts("dims"), 0.1f);
                fastllm::Data input1 = MakeTensor(params.GetInts("dims"), 0.7f);
                return CanRunOnDevice(device, "AddTo", {{"input0", &input0}, {"input1", &input1}},
                                      {{"alpha", params.GetFloat("alpha")}}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input0 = MakeTensor(params.GetInts("dims"), 0.1f);
                fastllm::Data input1 = MakeTensor(params.GetInts("dims"), 0.7f);
                ScopedFirstDevice guard(device);
                fastllm::AddTo(input0, input1, params.GetFloat("alpha"));
                input0.ToDevice(fastllm::DataDevice::CPU);
                return input0;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 3.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims")) * 2.0;
            }
        };
    }

    static OpCase MakeCatCase() {
        return {
            "cat",
            "concatenate two tensors",
            []() {
                OpTestParams params;
                params.Add("dims0", "2,3", "left input shape");
                params.Add("dims1", "2,5", "right input shape");
                params.Add("axis", "1", "concat axis");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input0 = MakeTensor(params.GetInts("dims0"), 0.2f);
                fastllm::Data input1 = MakeTensor(params.GetInts("dims1"), 0.5f);
                return CanRunOnDevice(device, "Cat", {{"input0", &input0}, {"input1", &input1}},
                                      {}, {{"axis", params.GetInt("axis")}});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input0 = MakeTensor(params.GetInts("dims0"), 0.2f);
                fastllm::Data input1 = MakeTensor(params.GetInts("dims1"), 0.5f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Cat(input0, input1, params.GetInt("axis"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                std::vector<int> dims0 = params.GetInts("dims0");
                std::vector<int> dims1 = params.GetInts("dims1");
                std::vector<int> outDims = dims0;
                outDims[params.GetInt("axis")] += dims1[params.GetInt("axis")];
                return FloatBytes(dims0) + FloatBytes(dims1) + FloatBytes(outDims);
            },
            [](const OpTestParams&) {
                return 0.0;
            }
        };
    }

    static OpCase MakeMulCase() {
        return {
            "mul",
            "multiply tensor by scalar",
            []() {
                OpTestParams params;
                params.Add("dims", "4,8", "input tensor shape");
                params.Add("value", "2.0", "scalar multiplier");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.3f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Mul", {{"input", &input}, {"output", &output}},
                                      {{"v", params.GetFloat("value")}}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.3f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Mul(input, params.GetFloat("value"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims"));
            }
        };
    }

    static OpCase MakePermuteCase() {
        return {
            "permute",
            "permute tensor axes",
            []() {
                OpTestParams params;
                params.Add("dims", "2,3,4", "input tensor shape");
                params.Add("axis", "1,2,0", "permutation");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.4f);
                fastllm::Data output;
                fastllm::Data axisData(fastllm::DataType::INT32PARAM, {(int) params.GetInts("axis").size()});
                axisData.Allocate();
                std::vector<int> axis = params.GetInts("axis");
                for (int i = 0; i < (int) axis.size(); i++) {
                    ((int32_t*) axisData.cpuData)[i] = axis[i];
                }
                return CanRunOnDevice(device, "Permute", {{"input", &input}, {"axis", &axisData}, {"output", &output}}, {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.4f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Permute(input, params.GetInts("axis"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams&) {
                return 0.0;
            }
        };
    }

    static OpCase MakeSplitCase() {
        return {
            "split",
            "slice tensor on one axis",
            []() {
                OpTestParams params;
                params.Add("dims", "3,8", "input tensor shape");
                params.Add("axis", "1", "split axis");
                params.Add("start", "2", "inclusive start");
                params.Add("end", "6", "exclusive end");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.9f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Split", {{"input", &input}, {"output", &output}},
                                      {}, {{"axis", params.GetInt("axis")}, {"start", params.GetInt("start")}, {"end", params.GetInt("end")}});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.9f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Split(input, params.GetInt("axis"), params.GetInt("start"), params.GetInt("end"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                std::vector<int> outDims = params.GetInts("dims");
                outDims[params.GetInt("axis")] = params.GetInt("end") - params.GetInt("start");
                return FloatBytes(params.GetInts("dims")) + FloatBytes(outDims);
            },
            [](const OpTestParams&) {
                return 0.0;
            }
        };
    }

    static OpCase MakeMatMulCase() {
        return {
            "matmul",
            "matrix multiplication",
            []() {
                OpTestParams params;
                params.Add("m", "4", "left rows");
                params.Add("k", "8", "shared dimension");
                params.Add("n", "5", "right cols");
                params.Add("alpha", "1.0", "output scale");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int m = params.GetInt("m"), k = params.GetInt("k"), n = params.GetInt("n");
                fastllm::Data input0 = MakeTensor({m, k}, 0.2f);
                fastllm::Data input1 = MakeTensor({k, n}, 0.8f);
                fastllm::Data output;
                return CanRunOnDevice(device, "MatMul", {{"input0", &input0}, {"input1", &input1}, {"output", &output}},
                                      {{"alpha", params.GetFloat("alpha")}}, {{"group", 1}});
            },
            [](const OpTestParams &params, const std::string &device) {
                int m = params.GetInt("m"), k = params.GetInt("k"), n = params.GetInt("n");
                fastllm::Data input0 = MakeTensor({m, k}, 0.2f);
                fastllm::Data input1 = MakeTensor({k, n}, 0.8f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::MatMul(input0, input1, output, params.GetFloat("alpha"), 1);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int m = params.GetInt("m"), k = params.GetInt("k"), n = params.GetInt("n");
                return FloatBytes({m, k}) + FloatBytes({k, n}) + FloatBytes({m, n});
            },
            [](const OpTestParams &params) {
                return 2.0 * params.GetInt("m") * params.GetInt("k") * params.GetInt("n");
            }
        };
    }

    static OpCase MakeLayerNormCase() {
        return {
            "layernorm",
            "layer normalization",
            []() {
                OpTestParams params;
                params.Add("batch", "3", "batch size");
                params.Add("hidden", "8", "hidden size");
                params.Add("axis", "-1", "norm axis");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden}, 0.1f);
                fastllm::Data gamma = MakeRampTensor({hidden}, 0.8f);
                fastllm::Data beta = MakeRampTensor({hidden}, -0.3f);
                fastllm::Data output;
                return CanRunOnDevice(device, "LayerNorm", {{"input", &input}, {"gamma", &gamma}, {"beta", &beta}, {"output", &output}},
                                      {}, {{"axis", params.GetInt("axis")}});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden}, 0.1f);
                fastllm::Data gamma = MakeRampTensor({hidden}, 0.8f);
                fastllm::Data beta = MakeRampTensor({hidden}, -0.3f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::LayerNorm(input, gamma, beta, params.GetInt("axis"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                return FloatBytes({batch, hidden}) * 2.0 + FloatBytes({hidden}) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) params.GetInt("batch") * params.GetInt("hidden") * 8.0;
            }
        };
    }

    static OpCase MakeRmsNormCase() {
        return {
            "rmsnorm",
            "rms normalization",
            []() {
                OpTestParams params;
                params.Add("batch", "3", "batch size");
                params.Add("hidden", "8", "hidden size");
                params.Add("eps", "1e-5", "epsilon");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden}, 0.6f);
                fastllm::Data weight = MakeRampTensor({hidden}, 1.0f);
                fastllm::Data output;
                return CanRunOnDevice(device, "RMSNorm", {{"input", &input}, {"weight", &weight}, {"output", &output}},
                                      {{"eps", params.GetFloat("eps")}}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden}, 0.6f);
                fastllm::Data weight = MakeRampTensor({hidden}, 1.0f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::RMSNorm(input, weight, params.GetFloat("eps"), output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                return FloatBytes({batch, hidden}) * 2.0 + FloatBytes({hidden});
            },
            [](const OpTestParams &params) {
                return (double) params.GetInt("batch") * params.GetInt("hidden") * 6.0;
            }
        };
    }

    static OpCase MakeLinearCase() {
        return {
            "linear",
            "fully connected layer",
            []() {
                OpTestParams params;
                params.Add("batch", "4", "batch size");
                params.Add("in", "8", "input features");
                params.Add("out", "6", "output features");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                fastllm::Data input = MakeTensor({batch, in}, 0.2f);
                fastllm::Data weight = MakeTensor({out, in}, 0.4f);
                fastllm::Data bias = MakeRampTensor({out}, -0.1f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Linear", {{"input", &input}, {"weight", &weight}, {"bias", &bias}, {"output", &output}},
                                      {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                fastllm::Data input = MakeTensor({batch, in}, 0.2f);
                fastllm::Data weight = MakeTensor({out, in}, 0.4f);
                fastllm::Data bias = MakeRampTensor({out}, -0.1f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Linear(input, weight, bias, output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                auto input = std::make_shared<fastllm::Data>(MakeTensor({batch, in}, 0.2f));
                auto weight = std::make_shared<fastllm::Data>(MakeTensor({out, in}, 0.4f));
                auto bias = std::make_shared<fastllm::Data>(MakeRampTensor({out}, -0.1f));
                auto output = std::make_shared<fastllm::Data>();
                return [device, input, weight, bias, output]() {
                    ScopedFirstDevice guard(device);
                    fastllm::Linear(*input, *weight, *bias, *output);
                };
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                return FloatBytes({batch, in}) + FloatBytes({out, in}) + FloatBytes({out}) + FloatBytes({batch, out});
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                return 2.0 * batch * in * out + (double) batch * out;
            }
        };
    }

    static OpCase MakeSiluCase() {
        return {
            "silu",
            "SiLU activation",
            []() {
                OpTestParams params;
                params.Add("dims", "4,8", "input tensor shape");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.4f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Silu", {{"input", &input}, {"output", &output}}, {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.4f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Silu(input, output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims")) * 4.0;
            }
        };
    }

    static OpCase MakeSoftmaxCase() {
        return {
            "softmax",
            "softmax on one axis",
            []() {
                OpTestParams params;
                params.Add("dims", "3,7", "input tensor shape");
                params.Add("axis", "-1", "softmax axis");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.1f);
                fastllm::Data output;
                return CanRunOnDevice(device, "SoftMax", {{"input", &input}, {"output", &output}},
                                      {}, {{"axis", params.GetInt("axis")}});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.1f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Softmax(input, output, params.GetInt("axis"));
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims")) * 5.0;
            }
        };
    }

    static OpCase MakeGeluNewCase() {
        return {
            "gelunew",
            "GELU New activation",
            []() {
                OpTestParams params;
                params.Add("dims", "4,8", "input tensor shape");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.5f);
                fastllm::Data output;
                return CanRunOnDevice(device, "GeluNew", {{"input", &input}, {"output", &output}}, {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                fastllm::Data input = MakeTensor(params.GetInts("dims"), 0.5f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::GeluNew(input, output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                return FloatBytes(params.GetInts("dims")) * 2.0;
            },
            [](const OpTestParams &params) {
                return (double) CountElements(params.GetInts("dims")) * 8.0;
            }
        };
    }

    static OpCase MakeSwigluCase() {
        return {
            "swiglu",
            "SwiGLU activation",
            []() {
                OpTestParams params;
                params.Add("batch", "4", "batch size");
                params.Add("hidden", "8", "output hidden size, input will be hidden * 2");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden * 2}, 0.7f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Swiglu", {{"input", &input}, {"output", &output}}, {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                fastllm::Data input = MakeTensor({batch, hidden * 2}, 0.7f);
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Swiglu(input, output);
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), hidden = params.GetInt("hidden");
                return FloatBytes({batch, hidden * 2}) + FloatBytes({batch, hidden});
            },
            [](const OpTestParams &params) {
                return (double) params.GetInt("batch") * params.GetInt("hidden") * 6.0;
            }
        };
    }

    static OpCase MakeAttentionCase() {
        return {
            "attention",
            "scaled dot-product attention",
            []() {
                OpTestParams params;
                params.Add("batch", "1", "batch size");
                params.Add("seq", "4", "sequence length");
                params.Add("dim", "8", "head dimension total");
                params.Add("group", "1", "attention group");
                params.Add("attention_type", "1", "1 normal, 2 no mask");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), seq = params.GetInt("seq"), dim = params.GetInt("dim");
                fastllm::Data q = MakeTensor({batch, seq, dim}, 0.1f);
                fastllm::Data k = MakeTensor({batch, seq, dim}, 0.5f);
                fastllm::Data v = MakeTensor({batch, seq, dim}, 0.9f);
                fastllm::Data mask;
                fastllm::Data output;
                return CanRunOnDevice(device, "Attention", {{"q", &q}, {"k", &k}, {"v", &v}, {"mask", &mask}, {"output", &output}},
                                      {{"scale", 1.0f / std::sqrt((float) dim)}},
                                      {{"group", params.GetInt("group")}, {"maskType", params.GetInt("attention_type") == 2 ? 2 : 0}});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), seq = params.GetInt("seq"), dim = params.GetInt("dim");
                fastllm::Data q = MakeTensor({batch, seq, dim}, 0.1f);
                fastllm::Data k = MakeTensor({batch, seq, dim}, 0.5f);
                fastllm::Data v = MakeTensor({batch, seq, dim}, 0.9f);
                fastllm::Data mask;
                fastllm::Data output;
                ScopedFirstDevice guard(device);
                fastllm::Attention(q, k, v, mask, output, params.GetInt("group"),
                                   1.0f / std::sqrt((float) dim), params.GetInt("attention_type"));
                output.ToDevice(fastllm::DataDevice::CPU);
                return output;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), seq = params.GetInt("seq"), dim = params.GetInt("dim");
                return FloatBytes({batch, seq, dim}) * 4.0;
            },
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), seq = params.GetInt("seq"), dim = params.GetInt("dim");
                return 4.0 * batch * seq * seq * dim;
            }
        };
    }

    static std::vector<OpCase> BuildRegistry() {
        return {
            MakeAddToCase(),
            MakeCatCase(),
            MakeMulCase(),
            MakePermuteCase(),
            MakeSplitCase(),
            MakeMatMulCase(),
            MakeLayerNormCase(),
            MakeRmsNormCase(),
            MakeLinearCase(),
            MakeSiluCase(),
            MakeSoftmaxCase(),
            MakeGeluNewCase(),
            MakeSwigluCase(),
            MakeAttentionCase()
        };
    }

    static const OpCase* FindCase(const std::vector<OpCase> &registry, const std::string &name) {
        for (const auto &opCase : registry) {
            if (opCase.name == name) {
                return &opCase;
            }
        }
        return nullptr;
    }

    static void PrintHelp(const std::vector<OpCase> &registry) {
        std::cout
            << "Usage: ./optest --op <name|all> [--device cpu,cuda:0] [--param key=value]\n"
            << "                [--iters N] [--warmup N] [--atol X] [--rtol X] [--list]\n\n"
            << "Examples:\n"
            << "  ./optest --list\n"
            << "  ./optest --op matmul\n"
            << "  ./optest --op matmul --param m=64 --param k=128 --param n=256 --device cpu,cuda:0\n"
            << "  ./optest --op attention --param seq=128 --param dim=128 --iters 50\n\n"
            << "Available ops:\n";
        for (const auto &opCase : registry) {
            std::cout << "  " << opCase.name << "  # " << opCase.description << "\n";
        }
    }

    static CliConfig ParseArgs(int argc, char **argv) {
        CliConfig config;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            auto requireValue = [&](const std::string &name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for " + name);
                }
                return argv[++i];
            };

            if (arg == "-h" || arg == "--help") {
                config.help = true;
            } else if (arg == "--list") {
                config.listOps = true;
            } else if (arg == "--op") {
                config.op = requireValue(arg);
            } else if (arg == "--device") {
                config.deviceFilters = Split(requireValue(arg), ',');
            } else if (arg == "--param") {
                std::string kv = requireValue(arg);
                size_t pos = kv.find('=');
                if (pos == std::string::npos) {
                    throw std::runtime_error("--param expects key=value");
                }
                config.paramOverrides.push_back({Trim(kv.substr(0, pos)), Trim(kv.substr(pos + 1))});
            } else if (arg == "--iters") {
                config.iters = std::stoi(requireValue(arg));
            } else if (arg == "--warmup") {
                config.warmup = std::stoi(requireValue(arg));
            } else if (arg == "--atol") {
                config.atol = std::stof(requireValue(arg));
            } else if (arg == "--rtol") {
                config.rtol = std::stof(requireValue(arg));
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        return config;
    }

    static OpTestParams ResolveParams(const OpCase &opCase, const CliConfig &config) {
        OpTestParams params = opCase.makeDefaultParams();
        for (const auto &it : config.paramOverrides) {
            params.Override(it.first, it.second);
        }
        return params;
    }

    static bool RunSingleCase(const OpCase &opCase, const CliConfig &config) {
        OpTestParams params = ResolveParams(opCase, config);
        std::vector<std::string> devices = GetAvailableDevices();

        std::cout << "\n== op: " << opCase.name << " ==\n";
        std::cout << opCase.description << "\n";
        params.Print(std::cout);

        fastllm::Data baseline = opCase.run(params, "cpu");
        baseline.ToDevice(fastllm::DataDevice::CPU);
        bool ok = true;

        for (const auto &device : devices) {
            if (!DeviceSelected(config.deviceFilters, device)) {
                continue;
            }
            if (!opCase.canRun(params, device)) {
                std::cout << "  [" << device << "] skipped: op not supported on this device\n";
                continue;
            }

            fastllm::Data output = opCase.run(params, device);
            output.ToDevice(fastllm::DataDevice::CPU);
            ComparisonStats stats = CompareData(baseline, output, config.atol, config.rtol);
            bool pass = stats.maxAbsDiff <= config.atol + config.rtol * std::fabs(stats.expected);
            BenchmarkResult bench = Benchmark(opCase, params, device, config.warmup, config.iters);

            std::cout << "  [" << device << "] " << (pass ? "PASS" : "FAIL") << "\n";
            std::cout << "    accuracy:"
                      << " max_abs_diff=" << std::scientific << stats.maxAbsDiff
                      << ", max_rel_diff=" << stats.maxRelDiff << "\n";
            std::cout << "    latency:"
                      << " avg_ms=" << std::fixed << std::setprecision(4) << bench.avgMs << "\n";
            std::cout << "    throughput:";
            if (bench.bandwidthGBps >= 0.0) {
                std::cout << " io_speed=" << FormatIOSpeed(bench.bandwidthGBps);
            } else {
                std::cout << " io_speed=n/a";
            }
            if (bench.computeTFlops >= 0.0) {
                std::cout << ", compute_speed=" << FormatComputeSpeed(bench.computeTFlops);
            } else {
                std::cout << ", compute_speed=n/a";
            }
            std::cout << "\n";
            if (!pass) {
                std::cout << "    mismatch index=" << stats.mismatchIndex
                          << ", cpu=" << stats.expected
                          << ", device=" << stats.actual << "\n";
                ok = false;
            }
        }
        return ok;
    }
}

int main(int argc, char **argv) {
    try {
        std::vector<OpCase> registry = BuildRegistry();
        CliConfig config = ParseArgs(argc, argv);

        if (config.help) {
            PrintHelp(registry);
            return 0;
        }

        if (config.listOps) {
            for (const auto &opCase : registry) {
                std::cout << opCase.name << "\n";
            }
            return 0;
        }

        bool allOk = true;
        if (config.op == "all") {
            for (const auto &opCase : registry) {
                allOk = RunSingleCase(opCase, config) && allOk;
            }
        } else {
            const OpCase *opCase = FindCase(registry, config.op);
            if (opCase == nullptr) {
                throw std::runtime_error("unknown op: " + config.op);
            }
            allOk = RunSingleCase(*opCase, config);
        }

        std::cout << "\nSummary: " << (allOk ? "PASS" : "FAIL") << "\n";
        return allOk ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "optest error: " << e.what() << "\n";
        return 1;
    }
}
