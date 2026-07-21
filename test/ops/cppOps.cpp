#include "fastllm.h"
#include "executor.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
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

    static void InitLinearWeight(const OpTestParams &params, int out, int in, fastllm::Data &weight) {
        fastllm::Data fp32Weight = MakeTensor({out, in}, 0.4f);
        std::string weightType = params.Has("weight_type") ? params.GetString("weight_type") : "float32";
        if (weightType == "float32") {
            weight.dataType = fastllm::DataType::FLOAT32;
            weight.Resize({out, in});
            weight.Allocate();
            std::memcpy(weight.cpuData, fp32Weight.cpuData, weight.GetBytes());
            return;
        }
        if (weightType == "int4group") {
            int groupCnt = params.Has("group_cnt") ? params.GetInt("group_cnt") : 128;
            weight.dataType = fastllm::DataType::INT4_GROUP;
            weight.Resize({out, in});
            weight.CreateFromOriData(fastllm::WeightType::LINEAR, fastllm::DataType::FLOAT32,
                                     (uint8_t *) fp32Weight.cpuData, nullptr, nullptr, groupCnt);
            return;
        }
        throw std::runtime_error("unsupported linear weight_type: " + weightType);
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

    static std::vector<int32_t> ToInt32Vector(fastllm::Data data) {
        data.ToDevice(fastllm::DataDevice::CPU);
        size_t count = data.Count(0);
        if (data.dataType != fastllm::DataType::INT32) {
            throw std::runtime_error("only INT32 index outputs are supported");
        }
        const int32_t *ptr = reinterpret_cast<const int32_t*>(data.cpuData);
        return std::vector<int32_t>(ptr, ptr + count);
    }

    static fastllm::Data ConvertToFloat32Data(const fastllm::Data &data) {
        fastllm::Data output;
        fastllm::ToDataType(data, output, fastllm::DataType::FLOAT32);
        output.ToDevice(fastllm::DataDevice::CPU);
        return output;
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
        std::function<BenchmarkResult(const OpTestParams&, const std::string&, int, int)> benchmarkOverride = nullptr;
        std::function<double(const OpTestParams&)> GetIOBytes = nullptr;
        std::function<double(const OpTestParams&)> GetComputeOps = nullptr;
        bool benchmarkOnly = false;

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

        OpCase(std::string name, std::string description,
               std::function<OpTestParams()> makeDefaultParams,
               std::function<bool(const OpTestParams&, const std::string&)> canRun,
               std::function<fastllm::Data(const OpTestParams&, const std::string&)> run,
               std::function<BenchmarkResult(const OpTestParams&, const std::string&, int, int)> benchmarkOverride,
               std::function<double(const OpTestParams&)> GetIOBytes,
               std::function<double(const OpTestParams&)> GetComputeOps,
               bool benchmarkOnly)
            : name(std::move(name)), description(std::move(description)),
              makeDefaultParams(std::move(makeDefaultParams)), canRun(std::move(canRun)),
              run(std::move(run)), benchmarkOverride(std::move(benchmarkOverride)),
              GetIOBytes(std::move(GetIOBytes)), GetComputeOps(std::move(GetComputeOps)),
              benchmarkOnly(benchmarkOnly) {}
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
        if (opCase.benchmarkOverride) {
            return opCase.benchmarkOverride(params, device, warmup, iters);
        }
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

    struct RmsNormFp16BenchState {
        std::string kind;
        std::string path;
        int threads = 0;
        float eps = 1.0e-6f;
        fastllm::Data input, weight, gate;
        fastllm::Data legacyOutput, fastOutput;

        static void PrepareFp16Tensor(fastllm::Data &data, const std::vector<int> &dims, float seed) {
            data.CopyFrom(MakeTensor(dims, seed, 0.25f));
            fastllm::ToDataType(data, fastllm::DataType::FLOAT16);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        static void PrepareOutput(fastllm::Data &data, const std::vector<int> &dims) {
            data.dataType = fastllm::DataType::FLOAT16;
            data.UpdateUnitSize();
            data.Resize(dims);
            data.Allocate(false);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        void RunOne(const fastllm::Data &source, fastllm::Data &output, int threadCount) {
            bool ok = false;
            if (kind == "rmsnorm") {
                ok = FastllmCudaRMSNormFloat16WithThreadCount(
                    source, weight, output, eps, threadCount);
            } else if (kind == "rmsnorm_silu_mul") {
                ok = FastllmCudaRMSNormSiluMulFloat16WithThreadCount(
                    source, weight, gate, output, eps, threadCount);
            } else {
                throw std::runtime_error("kind must be rmsnorm or rmsnorm_silu_mul");
            }
            if (!ok) {
                throw std::runtime_error("FP16 RMSNorm specialized launch failed");
            }
        }

        static void ExpectExact(const fastllm::Data &expectedData,
                                const fastllm::Data &actualData,
                                const char *label) {
            std::vector<float> expected = ToFloatVector(ConvertToFloat32Data(expectedData));
            std::vector<float> actual = ToFloatVector(ConvertToFloat32Data(actualData));
            if (expected == actual) {
                return;
            }
            for (size_t i = 0; i < expected.size(); i++) {
                if (expected[i] != actual[i]) {
                    std::ostringstream os;
                    os << label << " mismatch at " << i
                       << ": expected=" << expected[i]
                       << " actual=" << actual[i];
                    throw std::runtime_error(os.str());
                }
            }
        }

        void Check(const std::vector<int> &dims) {
            RunOne(input, legacyOutput, 0);
            RunOne(input, fastOutput, threads);
            ForceDeviceSync();
            ExpectExact(legacyOutput, fastOutput, "out-of-place FP16 RMSNorm");

            fastllm::Data legacyInplace, fastInplace;
            PrepareFp16Tensor(legacyInplace, dims, 0.413f);
            PrepareFp16Tensor(fastInplace, dims, 0.413f);
            RunOne(legacyInplace, legacyInplace, 0);
            RunOne(fastInplace, fastInplace, threads);
            ForceDeviceSync();
            ExpectExact(legacyInplace, fastInplace, "in-place FP16 RMSNorm");
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            FastllmCudaSetDevice(0);
            kind = params.GetString("kind");
            path = params.GetString("path");
            threads = params.GetInt("threads");
            eps = params.GetFloat("eps");
            int outer = params.GetInt("outer");
            int hidden = params.GetInt("hidden");
            std::vector<int> dims = {outer, hidden};
            PrepareFp16Tensor(input, dims, 0.413f);
            PrepareFp16Tensor(gate, dims, 0.927f);
            weight.CopyFrom(MakeRampTensor({hidden}, 0.731f));
            weight.ToDevice(fastllm::DataDevice::CUDA);
            PrepareOutput(legacyOutput, dims);
            PrepareOutput(fastOutput, dims);
            Check(dims);
#else
            (void)params;
            throw std::runtime_error("rmsnorm_fp16_specialized benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (path == "legacy") {
                RunOne(input, legacyOutput, 0);
            } else if (path == "fast") {
                RunOne(input, fastOutput, threads);
            } else {
                throw std::runtime_error("path must be legacy or fast");
            }
        }
    };

    static BenchmarkResult BenchmarkRmsNormFp16Cuda(
            const OpTestParams &params, const std::string &device, int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<RmsNormFp16BenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("rmsnorm_fp16_specialized benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeRmsNormFp16SpecializedCase() {
        return {
            "rmsnorm_fp16_specialized",
            "benchmark exact FP16 RMSNorm and RMSNorm+SiLU+mul specializations",
            []() {
                OpTestParams params;
                params.Add("kind", "rmsnorm", "rmsnorm or rmsnorm_silu_mul");
                params.Add("path", "fast", "legacy or fast");
                params.Add("outer", "16", "number of rows");
                params.Add("hidden", "128", "channels per row");
                params.Add("threads", "32", "fast thread count");
                params.Add("eps", "1e-6", "epsilon");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkRmsNormFp16Cuda,
            [](const OpTestParams &params) {
                int outer = params.GetInt("outer"), hidden = params.GetInt("hidden");
                return (double)(outer * hidden * 2 * 2 + hidden * 4);
            },
            [](const OpTestParams &params) {
                return (double)params.GetInt("outer") * params.GetInt("hidden") * 6.0;
            },
            true
        };
    }

    struct RecurrentFromConvFp16BenchState {
        std::string path;
        int tileV = 8;
        bool exactNorm = false;
        int numKHeads = 8;
        int numVHeads = 16;
        int headKDim = 128;
        int headVDim = 128;
        float eps = 1.0e-6f;
        float qScale = 1.0f / std::sqrt(128.0f);
        fastllm::Data conv, ba, normWeight, aLog, dtBias;
        fastllm::Data workState, workOutput;

        static void PrepareFp16Tensor(fastllm::Data &data,
                                      const std::vector<int> &dims,
                                      float seed, float scale) {
            data.CopyFrom(MakeTensor(dims, seed, scale));
            fastllm::ToDataType(data, fastllm::DataType::FLOAT16);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        void PrepareState(fastllm::Data &state) const {
            PrepareFp16Tensor(state,
                              {1, numVHeads, headKDim, headVDim},
                              0.517f, 0.035f);
            state.isLinearAttentionTransposed = true;
        }

        void RunOne(fastllm::Data &state, fastllm::Data &output,
                    int selectedTileV, bool selectedExactNorm) {
            if (!FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16WithConfig(
                    conv, ba, normWeight, aLog, dtBias,
                    state, output,
                    numKHeads, numVHeads, headKDim, headVDim,
                    eps, qScale, selectedTileV, selectedExactNorm)) {
                throw std::runtime_error("configured recurrent kernel launch failed");
            }
        }

        void Check() {
            fastllm::Data legacyState, candidateState;
            fastllm::Data legacyOutput, candidateOutput;
            PrepareState(legacyState);
            PrepareState(candidateState);
            RunOne(legacyState, legacyOutput, 8, false);
            RunOne(candidateState, candidateOutput, tileV, exactNorm);
            ForceDeviceSync();
            RmsNormFp16BenchState::ExpectExact(
                legacyOutput, candidateOutput, "recurrent output");
            RmsNormFp16BenchState::ExpectExact(
                legacyState, candidateState, "recurrent state");
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            FastllmCudaSetDevice(0);
            path = params.GetString("path");
            tileV = params.GetInt("tile_v");
            exactNorm = params.GetInt("exact_norm") != 0;
            numKHeads = params.GetInt("k_heads");
            numVHeads = params.GetInt("v_heads");
            headVDim = params.GetInt("v_dim");
            eps = params.GetFloat("eps");
            headKDim = 128;
            qScale = 1.0f / std::sqrt((float)headKDim);
            int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;

            PrepareFp16Tensor(conv, {1, 1, qkvDim}, 0.271f, 0.08f);
            PrepareFp16Tensor(ba, {1, 1, numVHeads * 2}, 0.913f, 0.10f);
            normWeight.CopyFrom(MakeRampTensor({headKDim}, 0.85f));
            normWeight.ToDevice(fastllm::DataDevice::CUDA);
            aLog.CopyFrom(MakeTensor({numVHeads}, 0.419f, 0.15f));
            aLog.ToDevice(fastllm::DataDevice::CUDA);
            dtBias.CopyFrom(MakeTensor({numVHeads}, 0.731f, 0.10f));
            dtBias.ToDevice(fastllm::DataDevice::CUDA);

            Check();
            PrepareState(workState);
#else
            (void)params;
            throw std::runtime_error("recurrent_from_conv_fp16 benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (path == "legacy") {
                RunOne(workState, workOutput, 8, false);
            } else if (path == "fast") {
                RunOne(workState, workOutput, tileV, exactNorm);
            } else {
                throw std::runtime_error("path must be legacy or fast");
            }
        }
    };

    static BenchmarkResult BenchmarkRecurrentFromConvFp16Cuda(
            const OpTestParams &params, const std::string &device,
            int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<RecurrentFromConvFp16BenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("recurrent_from_conv_fp16 benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeRecurrentFromConvFp16Case() {
        return {
            "recurrent_from_conv_fp16",
            "benchmark the exact Qwen3.5 recurrent-from-conv normalization path",
            []() {
                OpTestParams params;
                params.Add("path", "fast", "legacy or fast");
                params.Add("tile_v", "8", "value columns per CTA (currently 8)");
                params.Add("exact_norm", "1", "use exact single-warp q/k norm");
                params.Add("k_heads", "8", "local key heads");
                params.Add("v_heads", "16", "local value heads");
                params.Add("v_dim", "128", "value head dimension");
                params.Add("eps", "1e-6", "RMSNorm epsilon");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkRecurrentFromConvFp16Cuda,
            [](const OpTestParams &params) {
                double stateElements = (double)params.GetInt("v_heads") * 128.0 *
                                       params.GetInt("v_dim");
                return stateElements * 4.0;
            },
            [](const OpTestParams &params) {
                double stateElements = (double)params.GetInt("v_heads") * 128.0 *
                                       params.GetInt("v_dim");
                return stateElements * 8.0;
            },
            true
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
                params.Add("weight_type", "float32", "weight datatype: float32 or int4group");
                params.Add("group_cnt", "128", "group size used by int4group quantization");
                return params;
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                fastllm::Data input = MakeTensor({batch, in}, 0.2f);
                fastllm::Data weight;
                InitLinearWeight(params, out, in, weight);
                fastllm::Data bias = MakeRampTensor({out}, -0.1f);
                fastllm::Data output;
                return CanRunOnDevice(device, "Linear", {{"input", &input}, {"weight", &weight}, {"bias", &bias}, {"output", &output}},
                                      {}, {});
            },
            [](const OpTestParams &params, const std::string &device) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                fastllm::Data input = MakeTensor({batch, in}, 0.2f);
                fastllm::Data weight;
                InitLinearWeight(params, out, in, weight);
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
                auto weight = std::make_shared<fastllm::Data>();
                InitLinearWeight(params, out, in, *weight);
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

    struct LinearFp8Block128BenchState {
        int batch = 16;
        int in = 4096;
        int out = 4096;
        int block = 128;
        fastllm::Data input, weight, bias, output;

        static void InitPackedWeight(fastllm::Data &weight, int out, int in, int block, float seed) {
            if (block != 128 || (in % block) != 0) {
                throw std::runtime_error("linear_fp8_block128 requires block=128 and input features aligned to 128");
            }
            weight.dataType = fastllm::DataType::FP8_E4M3_BLOCK_128;
            weight.UpdateUnitSize();
            weight.Resize({out, in});
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = block;
            weight.blockM = block;
            weight.Allocate(false);

            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            size_t perRow = fastllm::GetDataBytes(fastllm::DataType::FP8_E4M3_BLOCK_128, 1, in);
            int blocks = (in + block - 1) / block;
            for (int r = 0; r < out; r++) {
                uint8_t *row = ptr + (size_t)r * perRow;
                for (int b = 0; b < blocks; b++) {
                    uint8_t *blockPtr = row + b * (block + (int)sizeof(float));
                    for (int c = 0; c < block; c++) {
                        blockPtr[c] = static_cast<uint8_t>(0x20 + ((r * 131 + b * 17 + c + (int)(seed * 11.0f)) & 0x1f));
                    }
                    float scale = 0.0125f + 0.0001f * (float)((r + b + (int)seed) % 11);
                    memcpy(blockPtr + block, &scale, sizeof(float));
                }
            }
        }

        static void InitSeparateScaleWeight(fastllm::Data &weight, int out, int in, int block, float seed) {
            if (block != 128 || (in % block) != 0 || (out % block) != 0) {
                throw std::runtime_error("linear_fp8_block128 separate layout requires in/out aligned to 128");
            }
            weight.dataType = fastllm::DataType::FP8_E4M3;
            weight.UpdateUnitSize();
            weight.Resize({out, in});
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = block;
            weight.blockM = block;
            weight.Allocate(false);

            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            for (uint64_t i = 0; i < weight.GetBytes(); i++) {
                ptr[i] = static_cast<uint8_t>(0x20 + ((i + (uint64_t)(seed * 11.0f)) & 0x1f));
            }
            int scaleRows = (out + block - 1) / block;
            int scaleCols = (in + block - 1) / block;
            weight.scales.resize((size_t)scaleRows * scaleCols);
            for (int r = 0; r < scaleRows; r++) {
                for (int c = 0; c < scaleCols; c++) {
                    float mild = 0.0125f + 0.0001f * (float)((r * scaleCols + c + (int)seed) % 11);
                    float blocky = 0.002f * (1.0f + (float)((r * 17 + c * 29 + (int)seed) % 31));
                    weight.scales[(size_t)r * scaleCols + c] = blocky + mild * 0.1f;
                }
            }
        }

        static void MakeBlockyInput(fastllm::Data &input, int batch, int in, int block) {
            float *ptr = reinterpret_cast<float*>(input.cpuData);
            int groups = (in + block - 1) / block;
            for (int r = 0; r < batch; r++) {
                for (int g = 0; g < groups; g++) {
                    float gain = 0.15f + 0.11f * (float)((r * 13 + g * 7) % 23);
                    for (int c = 0; c < block && g * block + c < in; c++) {
                        int col = g * block + c;
                        ptr[(size_t)r * in + col] *= gain;
                    }
                }
            }
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            batch = params.GetInt("batch");
            in = params.GetInt("in");
            out = params.GetInt("out");
            block = params.GetInt("block");
            FastllmCudaSetDevice(0);

            fastllm::Data fp32Input = MakeTensor({batch, in}, 0.23f, 0.02f);
            if (params.GetString("input_pattern") == "blocky") {
                MakeBlockyInput(fp32Input, batch, in, block);
            }
            input.CopyFrom(fp32Input);
            std::string inputType = params.GetString("input_type");
            if (inputType == "fp16") {
                fastllm::ToDataType(input, fastllm::DataType::FLOAT16);
            } else if (inputType == "bf16") {
                fastllm::ToDataType(input, fastllm::DataType::BFLOAT16);
            } else {
                throw std::runtime_error("input_type must be fp16 or bf16");
            }
            input.ToDevice(fastllm::DataDevice::CUDA);

            std::string weightLayout = params.GetString("weight_layout");
            if (weightLayout == "packed") {
                InitPackedWeight(weight, out, in, block, 0.7f);
            } else if (weightLayout == "separate") {
                InitSeparateScaleWeight(weight, out, in, block, 0.7f);
            } else {
                throw std::runtime_error("weight_layout must be packed or separate");
            }
            weight.ToDevice(fastllm::DataDevice::CUDA);

            if (params.GetInt("has_bias") != 0) {
                fastllm::Data fp32Bias = MakeRampTensor({out}, -0.1f);
                bias.CopyFrom(fp32Bias);
                bias.ToDevice(fastllm::DataDevice::CUDA);
            }
            output.dataType = input.dataType;
            output.UpdateUnitSize();
            output.Resize({batch, out});
            output.Allocate(false);
            output.ToDevice(fastllm::DataDevice::CUDA, false);
            ForceDeviceSync();
#else
            (void)params;
            throw std::runtime_error("linear_fp8_block128 benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            fastllm::Linear(input, weight, bias, output);
        }

        void ConvertWeightToMarlin() {
#ifdef USE_CUDA
            fastllm::Data fp32Warmup = MakeTensor({2, in}, 0.31f, 0.02f);
            fastllm::Data warmupInput;
            warmupInput.CopyFrom(fp32Warmup);
            fastllm::ToDataType(warmupInput, fastllm::DataType::FLOAT16);
            warmupInput.ToDevice(fastllm::DataDevice::CUDA);
            fastllm::Data warmupOutput;
            warmupOutput.dataType = fastllm::DataType::FLOAT16;
            warmupOutput.UpdateUnitSize();
            warmupOutput.Resize({2, out});
            warmupOutput.Allocate(false);
            warmupOutput.ToDevice(fastllm::DataDevice::CUDA, false);
            FastllmCudaSetNcclForceSync(true);
            bool ok = FastllmCudaHalfMatMulFloatFP8E4M3(
                warmupInput, weight, bias, warmupOutput, 2, in, out);
            if (!ok) {
                throw std::runtime_error("FP8 Marlin benchmark conversion failed");
            }
            ForceDeviceSync();
#endif
        }
    };

    static fastllm::Data MakeCudaOutputLike(fastllm::DataType dataType, int batch, int out) {
        fastllm::Data output;
        output.dataType = dataType;
        output.UpdateUnitSize();
        output.Resize({batch, out});
        output.Allocate(false);
        output.ToDevice(fastllm::DataDevice::CUDA, false);
        return output;
    }

    static void PrintLinearFp8Block128CheckStats(const std::vector<float> &expected,
                                                 const std::vector<float> &actual,
                                                 int batch, int out) {
        float maxAbsDiff = 0.0f;
        float maxRelDiff = 0.0f;
        size_t maxIndex = 0;
        int over1e2 = 0;
        int over1e1 = 0;
        int over1e0 = 0;
        double sumAbsDiff = 0.0;
        for (size_t i = 0; i < expected.size(); i++) {
            float absDiff = std::fabs(expected[i] - actual[i]);
            float relDiff = absDiff / std::max(std::fabs(expected[i]), 1e-6f);
            sumAbsDiff += absDiff;
            if (absDiff > 1e-2f) {
                over1e2++;
            }
            if (absDiff > 1e-1f) {
                over1e1++;
            }
            if (absDiff > 1.0f) {
                over1e0++;
            }
            if (absDiff > maxAbsDiff) {
                maxAbsDiff = absDiff;
                maxRelDiff = relDiff;
                maxIndex = i;
            }
        }
        int row = out == 0 ? 0 : (int)(maxIndex / (size_t)out);
        int col = out == 0 ? 0 : (int)(maxIndex % (size_t)out);
        std::cout << "linear_fp8_block128 check: batch=" << batch << ", out=" << out
                  << ", count=" << expected.size() << "\n";
        std::cout << "  max_abs_diff=" << maxAbsDiff
                  << ", max_rel_diff=" << maxRelDiff
                  << ", mean_abs_diff=" << (expected.empty() ? 0.0 : sumAbsDiff / expected.size())
                  << ", idx=" << maxIndex << " (" << row << ", " << col << ")"
                  << ", ref=" << expected[maxIndex] << ", actual=" << actual[maxIndex] << "\n";
        std::cout << "  abs_diff>1e-2: " << over1e2
                  << ", >1e-1: " << over1e1
                  << ", >1: " << over1e0 << "\n";
    }

    static void CheckLinearFp8Block128Cuda(const OpTestParams &params) {
        if (params.GetString("weight_layout") != "separate") {
            throw std::runtime_error("linear_fp8_block128 check currently targets separate FP8_E4M3 scale layout");
        }
        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);

        fastllm::Data ref = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);
        fastllm::Data cutlass = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);

        bool ok = false;
        if (state->input.dataType == fastllm::DataType::FLOAT16) {
            ok = FastllmCudaHalfMatMulFloatFP8E4M3(state->input, state->weight, state->bias,
                                                   ref, state->batch, state->in, state->out);
        } else if (state->input.dataType == fastllm::DataType::BFLOAT16) {
            ok = FastllmCudaBFloat16MatMulFP8E4M3(state->input, state->weight, state->bias,
                                                  ref, state->batch, state->in, state->out);
        } else {
            throw std::runtime_error("linear_fp8_block128 check requires fp16 or bf16 input");
        }
        if (!ok) {
            throw std::runtime_error("reference FP8 linear path failed");
        }
        ForceDeviceSync();

        ok = FastllmCudaCutlassLinearFP8E4M3Block128(state->input, state->weight, state->bias,
                                                     cutlass, state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("CUTLASS FP8 linear path failed");
        }
        ForceDeviceSync();

        fastllm::Data ref32 = ConvertToFloat32Data(ref);
        fastllm::Data cutlass32 = ConvertToFloat32Data(cutlass);
        std::vector<float> refVec = ToFloatVector(ref32);
        std::vector<float> cutlassVec = ToFloatVector(cutlass32);
        PrintLinearFp8Block128CheckStats(refVec, cutlassVec, state->batch, state->out);

        if (params.GetInt("print") != 0) {
            fastllm::Data input32 = ConvertToFloat32Data(state->input);
            fastllm::Data scales(fastllm::DataType::FLOAT32,
                                  {state->out / state->block, state->in / state->block},
                                  state->weight.scales);
            input32.Print("linear_fp8_check.input.float32");
            scales.Print("linear_fp8_check.weight_scales");
            ref32.Print("linear_fp8_check.ref.float32");
            cutlass32.Print("linear_fp8_check.cutlass.float32");
        }
    }

    static void CheckLinearFp8MarlinCuda(const OpTestParams &params) {
        if (params.GetString("weight_layout") != "separate" ||
            params.GetString("input_type") != "fp16") {
            throw std::runtime_error(
                "FP8 Marlin check requires weight_layout=separate and input_type=fp16");
        }

        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);
        if (state->batch <= 8) {
            throw std::runtime_error("FP8 Marlin tail check requires batch > 8");
        }

        fastllm::Data referenceWeight;
        LinearFp8Block128BenchState::InitSeparateScaleWeight(
            referenceWeight, state->out, state->in, state->block, 0.7f);
        referenceWeight.ToDevice(fastllm::DataDevice::CUDA);

        fastllm::Data reference = MakeCudaOutputLike(
            state->input.dataType, state->batch, state->out);
        fastllm::Data actual = MakeCudaOutputLike(
            state->input.dataType, state->batch, state->out);
        fastllm::Data warmup = MakeCudaOutputLike(
            state->input.dataType, 2, state->out);
        fastllm::Data actualSingle = MakeCudaOutputLike(
            state->input.dataType, 1, state->out);

        // A fresh weight with M > 8 stays on the legacy FP8 path and provides
        // the reference. A two-row call converts the second weight to Marlin;
        // the following calls exercise its large bulk/tail split and its
        // dedicated batch-one GEMV.
        bool ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, referenceWeight, state->bias, reference,
            state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("legacy FP8 reference path failed");
        }
        ForceDeviceSync();

        ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, state->weight, state->bias, warmup,
            2, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("FP8 Marlin warmup conversion failed");
        }
        ForceDeviceSync();

        ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, state->weight, state->bias, actual,
            state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("FP8 Marlin bulk/tail path failed");
        }
        ForceDeviceSync();

        ok = FastllmCudaHalfMatMulFloatFP8E4M3(
            state->input, state->weight, state->bias, actualSingle,
            1, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("FP8 Marlin-layout batch-one GEMV failed");
        }
        ForceDeviceSync();

        std::vector<float> expected = ToFloatVector(ConvertToFloat32Data(reference));
        std::vector<float> observed = ToFloatVector(ConvertToFloat32Data(actual));
        PrintLinearFp8Block128CheckStats(
            expected, observed, state->batch, state->out);

        int mismatches = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            float diff = std::fabs(expected[i] - observed[i]);
            float tolerance = 0.1f + 0.05f * std::fabs(expected[i]);
            if (!std::isfinite(observed[i]) || diff > tolerance) {
                mismatches++;
            }
        }
        if (mismatches != 0) {
            throw std::runtime_error(
                "FP8 Marlin output mismatch count: " + std::to_string(mismatches));
        }

        expected.resize(state->out);
        observed = ToFloatVector(ConvertToFloat32Data(actualSingle));
        PrintLinearFp8Block128CheckStats(expected, observed, 1, state->out);
        mismatches = 0;
        for (size_t i = 0; i < expected.size(); i++) {
            float diff = std::fabs(expected[i] - observed[i]);
            float tolerance = 0.1f + 0.05f * std::fabs(expected[i]);
            if (!std::isfinite(observed[i]) || diff > tolerance) {
                mismatches++;
            }
        }
        if (mismatches != 0) {
            throw std::runtime_error(
                "FP8 Marlin-layout batch-one GEMV mismatch count: " +
                std::to_string(mismatches));
        }
    }

    static void CheckLinearFp8Block128SwigluCuda(const OpTestParams &params) {
        if (params.GetString("weight_layout") != "separate") {
            throw std::runtime_error("linear_fp8_block128 swiglu check targets separate FP8_E4M3 scale layout");
        }
        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);

        fastllm::Data gateupFp32 = MakeTensor({state->batch, state->in * 2}, 0.17f, 0.03f);
        if (params.GetString("input_pattern") == "blocky") {
            LinearFp8Block128BenchState::MakeBlockyInput(gateupFp32, state->batch, state->in * 2, state->block);
        }
        fastllm::Data gateup;
        gateup.CopyFrom(gateupFp32);
        if (state->input.dataType == fastllm::DataType::FLOAT16) {
            fastllm::ToDataType(gateup, fastllm::DataType::FLOAT16);
        } else if (state->input.dataType == fastllm::DataType::BFLOAT16) {
            fastllm::ToDataType(gateup, fastllm::DataType::BFLOAT16);
        } else {
            throw std::runtime_error("linear_fp8_block128 swiglu check requires fp16 or bf16 input");
        }
        gateup.ToDevice(fastllm::DataDevice::CUDA);

        fastllm::Data swiglu = MakeCudaOutputLike(state->input.dataType, state->batch, state->in);
        fastllm::Data ref = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);
        fastllm::Data fused = MakeCudaOutputLike(state->input.dataType, state->batch, state->out);

        fastllm::Swiglu(gateup, swiglu);
        ForceDeviceSync();

        bool ok = FastllmCudaCutlassLinearFP8E4M3Block128(
            swiglu, state->weight, state->bias, ref, state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("CUTLASS FP8 swiglu reference linear path failed");
        }
        ForceDeviceSync();

        ok = FastllmCudaCutlassLinearFP8E4M3Block128FromSwiglu(
            gateup, state->weight, state->bias, fused, state->batch, state->in, state->out);
        if (!ok) {
            throw std::runtime_error("CUTLASS FP8 fused swiglu path failed");
        }
        ForceDeviceSync();

        fastllm::Data ref32 = ConvertToFloat32Data(ref);
        fastllm::Data fused32 = ConvertToFloat32Data(fused);
        std::vector<float> refVec = ToFloatVector(ref32);
        std::vector<float> fusedVec = ToFloatVector(fused32);
        PrintLinearFp8Block128CheckStats(refVec, fusedVec, state->batch, state->out);

        float maxAbsDiff = 0.0f;
        for (size_t i = 0; i < refVec.size(); i++) {
            maxAbsDiff = std::max(maxAbsDiff, std::fabs(refVec[i] - fusedVec[i]));
        }
        if (maxAbsDiff > 1.0e-3f) {
            throw std::runtime_error("CUTLASS FP8 fused swiglu output mismatch");
        }

        if (params.GetInt("print") != 0) {
            fastllm::Data gateup32 = ConvertToFloat32Data(gateup);
            fastllm::Data swiglu32 = ConvertToFloat32Data(swiglu);
            gateup32.Print("linear_fp8_swiglu_check.gateup.float32");
            swiglu32.Print("linear_fp8_swiglu_check.swiglu.float32");
            ref32.Print("linear_fp8_swiglu_check.ref.float32");
            fused32.Print("linear_fp8_swiglu_check.fused.float32");
        }
    }

    static BenchmarkResult BenchmarkLinearFp8Block128Cuda(const OpTestParams &params,
                                                          const std::string &device,
                                                          int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        if (params.GetInt("check") == 1) {
            CheckLinearFp8Block128Cuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 2) {
            CheckLinearFp8Block128SwigluCuda(params);
            return BenchmarkResult();
        } else if (params.GetInt("check") == 3) {
            CheckLinearFp8MarlinCuda(params);
            return BenchmarkResult();
        }

        auto state = std::make_shared<LinearFp8Block128BenchState>();
        state->Init(params);

        bool oldForceSync = FastllmCudaGetNcclForceSync();
        std::string kernel = params.GetString("kernel");
        if (kernel == "legacy") {
            FastllmCudaSetNcclForceSync(false);
        } else if (kernel == "marlin_batch1") {
            if (state->batch != 1) {
                throw std::runtime_error("marlin_batch1 benchmark requires batch=1");
            }
            state->ConvertWeightToMarlin();
            FastllmCudaSetNcclForceSync(false);
        } else if (kernel != "auto") {
            throw std::runtime_error("kernel must be auto, legacy, or marlin_batch1");
        }

        for (int i = 0; i < warmup; i++) {
            state->Run();
            ForceDeviceSync();
        }

        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        FastllmCudaSetNcclForceSync(oldForceSync);
        double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();

        int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
        BenchmarkResult result;
        result.avgMs = totalMs / std::max(iters, 1);
        result.bytesMoved = (double)batch * in * 2.0 +
                            (double)batch * out * 2.0 +
                            (double)fastllm::GetDataBytes(fastllm::DataType::FP8_E4M3_BLOCK_128, out, in) +
                            (double)out * sizeof(float);
        result.flops = 2.0 * (double)batch * in * out;
        double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0 && result.bytesMoved > 0.0) {
            result.bandwidthGBps = result.bytesMoved / seconds / 1e9;
        }
        if (seconds > 0.0 && result.flops > 0.0) {
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("linear_fp8_block128 benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeLinearFp8Block128Case() {
        return {
            "linear_fp8_block128",
            "benchmark-only dense FP8 block128 linear",
            []() {
                OpTestParams params;
                params.Add("batch", "16", "token batch size");
                params.Add("in", "4096", "input features");
                params.Add("out", "4096", "output features");
                params.Add("block", "128", "FP8 scale block size");
                params.Add("input_type", "bf16", "fp16 or bf16");
                params.Add("input_pattern", "blocky", "smooth or blocky");
                params.Add("weight_layout", "packed", "packed or separate");
                params.Add("has_bias", "1", "1 to include a float32 bias, 0 for no bias");
                params.Add("kernel", "auto", "auto, legacy, or marlin_batch1");
                params.Add("check", "0", "1 linear, 2 fused swiglu+quant, 3 Marlin bulk/tail check");
                params.Add("print", "0", "1 to print debug tensors when check=1");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkLinearFp8Block128Cuda,
            [](const OpTestParams &params) {
                int batch = params.GetInt("batch"), in = params.GetInt("in"), out = params.GetInt("out");
                double inputBytes = (double)batch * in * 2.0;
                double outputBytes = (double)batch * out * 2.0;
                double weightBytes = (double)fastllm::GetDataBytes(fastllm::DataType::FP8_E4M3_BLOCK_128, out, in);
                return inputBytes + outputBytes + weightBytes + (double)out * sizeof(float);
            },
            [](const OpTestParams &params) {
                return 2.0 * (double)params.GetInt("batch") * params.GetInt("in") * params.GetInt("out");
            },
            true
        };
    }

    struct MergeMoeFp8BenchState {
        int batch = 1;
        int topk = 8;
        int hidden = 2048;
        int inter = 768;
        int experts = 82;
        int block = 128;
        std::string path = "operator";
        fastllm::Data input, index, score, output;
        fastllm::Data w1, w2, w3, curInput, curOutput;
        fastllm::Data referenceW1, referenceOutput;
        std::vector<std::unique_ptr<fastllm::Data>> ownedWeights;
        std::vector<fastllm::Data*> weights;
        std::vector<fastllm::Data*> biass;

        static void InitFp8Weight(fastllm::Data &weight, int rows, int cols, int block, float seed) {
            weight.dataType = fastllm::DataType::FP8_E4M3;
            weight.UpdateUnitSize();
            weight.Resize({rows, cols});
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = block;
            weight.blockM = block;
            weight.Allocate(false);
            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            for (uint64_t i = 0; i < weight.GetBytes(); i++) {
                ptr[i] = static_cast<uint8_t>(0x20 + ((i + (uint64_t)(seed * 17.0f)) & 0x1f));
            }
            int scaleRows = (rows + block - 1) / block;
            int scaleCols = (cols + block - 1) / block;
            weight.scales.resize((size_t)scaleRows * scaleCols);
            for (size_t i = 0; i < weight.scales.size(); i++) {
                weight.scales[i] = 0.015f + 0.0001f * (float)((i + (int)seed) % 7);
            }
            weight.ToDevice(fastllm::DataDevice::CUDA);
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            batch = params.GetInt("batch");
            topk = params.GetInt("topk");
            hidden = params.GetInt("hidden");
            inter = params.GetInt("inter");
            experts = params.GetInt("experts");
            block = params.GetInt("block");
            path = params.GetString("path");
            FastllmCudaSetDevice(0);

            fastllm::Data fp32Input = MakeTensor({batch, hidden}, 0.11f, 0.02f);
            input.CopyFrom(fp32Input);
            std::string inputType = params.GetString("input_type");
            if (inputType == "fp16") {
                fastllm::ToDataType(input, fastllm::DataType::FLOAT16);
            } else if (inputType == "bf16") {
                fastllm::ToDataType(input, fastllm::DataType::BFLOAT16);
            } else {
                throw std::runtime_error("input_type must be fp16 or bf16");
            }
            input.ToDevice(fastllm::DataDevice::CUDA);

            index.dataType = fastllm::DataType::INT32;
            index.UpdateUnitSize();
            index.Resize({batch, topk});
            index.Allocate(false);
            score.dataType = fastllm::DataType::FLOAT32;
            score.UpdateUnitSize();
            score.Resize({batch, topk});
            score.Allocate(false);
            int32_t *indexPtr = reinterpret_cast<int32_t*>(index.cpuData);
            float *scorePtr = reinterpret_cast<float*>(score.cpuData);
            for (int b = 0; b < batch; b++) {
                for (int j = 0; j < topk; j++) {
                    indexPtr[b * topk + j] = (b * topk + j) % experts;
                    scorePtr[b * topk + j] = 1.0f / std::max(topk, 1);
                }
            }
            index.ToDevice(fastllm::DataDevice::CUDA);
            score.ToDevice(fastllm::DataDevice::CUDA);

            ownedWeights.resize((size_t)(experts + 1) * 2);
            weights.assign((size_t)(experts + 1) * 2, nullptr);
            biass.assign((size_t)(experts + 1) * 2, nullptr);
            for (int e = 0; e < experts; e++) {
                int idx = (e + 1) * 2;
                ownedWeights[idx] = std::make_unique<fastllm::Data>();
                ownedWeights[idx + 1] = std::make_unique<fastllm::Data>();
                InitFp8Weight(*ownedWeights[idx], inter * 2, hidden, block, (float)e + 0.3f);
                InitFp8Weight(*ownedWeights[idx + 1], hidden, inter, block, (float)e + 13.7f);
                weights[idx] = ownedWeights[idx].get();
                weights[idx + 1] = ownedWeights[idx + 1].get();
            }

            output.dataType = input.dataType;
            output.UpdateUnitSize();
            output.Resize({batch, hidden});
            output.ToDevice(fastllm::DataDevice::CUDA);
            output.Allocate(false);
            ForceDeviceSync();
            CheckWarpSpecialization();
#else
            (void)params;
            throw std::runtime_error("mergemoe_fp8 benchmark requires USE_CUDA");
#endif
        }

        bool RunIndexed(fastllm::Data &work, fastllm::Data &result,
                        bool allowWarpSpecialization) {
#ifdef USE_CUDA
            const int32_t *indices = reinterpret_cast<const int32_t*>(index.cudaData);
            const float *scores = reinterpret_cast<const float*>(score.cudaData);
            if (input.dataType == fastllm::DataType::FLOAT16) {
                return FastllmCudaHalfMergeMOEFP8E4M3Batch1Indexed(
                    input, work, result, weights.data(), (int)weights.size(),
                    indices, scores, topk, hidden, inter, allowWarpSpecialization);
            }
            return FastllmCudaBFloat16MergeMOEFP8E4M3Batch1Indexed(
                input, work, result, weights.data(), (int)weights.size(),
                indices, scores, topk, hidden, inter, allowWarpSpecialization);
#else
            (void)work;
            (void)result;
            (void)allowWarpSpecialization;
            return false;
#endif
        }

        void CheckWarpSpecialization() {
#ifdef USE_CUDA
            if (batch != 1 || topk != 8 || hidden != 2048 || inter != 256 || block != 128) {
                return;
            }
            if (!RunIndexed(referenceW1, referenceOutput, false)) {
                throw std::runtime_error("legacy MergeMOE FP8 launch failed");
            }
            ForceDeviceSync();
            std::vector<float> expected = ToFloatVector(ConvertToFloat32Data(referenceOutput));
            if (!RunIndexed(w1, output, true)) {
                throw std::runtime_error("warp MergeMOE FP8 launch failed");
            }
            ForceDeviceSync();
            std::vector<float> actual = ToFloatVector(ConvertToFloat32Data(output));
            if (expected.size() != actual.size()) {
                throw std::runtime_error("MergeMOE FP8 output size mismatch");
            }
            for (size_t i = 0; i < expected.size(); i++) {
                if (expected[i] != actual[i]) {
                    std::ostringstream os;
                    os << "MergeMOE FP8 warp mismatch at " << i
                       << ": expected=" << expected[i] << ", actual=" << actual[i];
                    throw std::runtime_error(os.str());
                }
            }
#endif
        }

        void Run() {
            if (path == "operator") {
                fastllm::MergeMOE(input, index, score, weights, biass,
                                  w1, w2, w3, curInput, curOutput,
                                  0.0f, output, 0, fastllm::MoeGateSwiglu);
            } else if (path == "legacy") {
                if (!RunIndexed(w1, output, false)) {
                    throw std::runtime_error("legacy MergeMOE FP8 launch failed");
                }
            } else if (path == "warp") {
                if (!RunIndexed(w1, output, true)) {
                    throw std::runtime_error("warp MergeMOE FP8 launch failed");
                }
            } else {
                throw std::runtime_error("path must be operator, legacy or warp");
            }
        }
    };

    static BenchmarkResult BenchmarkMergeMoeFp8Cuda(const OpTestParams &params,
                                                    const std::string &device,
                                                    int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<MergeMoeFp8BenchState>();
        state->Init(params);

        for (int i = 0; i < warmup; i++) {
            state->Run();
            ForceDeviceSync();
        }

        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(end - begin).count();

        BenchmarkResult result;
        result.avgMs = totalMs / std::max(iters, 1);
        if (result.avgMs <= 0.0) {
            result.avgMs = 0.0;
        }
        result.bytesMoved = 0.0;
        result.flops = (double)params.GetInt("batch") * params.GetInt("topk") *
                       6.0 * (double)params.GetInt("hidden") * params.GetInt("inter");
        double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0 && result.flops > 0.0) {
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("mergemoe_fp8 benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeMergeMoeFp8Case() {
        return {
            "mergemoe_fp8",
            "benchmark-only Qwen3-style FP8 block-scaled MergeMOE",
            []() {
                OpTestParams params;
                params.Add("batch", "1", "token batch size");
                params.Add("hidden", "2048", "Qwen3-20B-A3B hidden size");
                params.Add("inter", "768", "Qwen3-20B-A3B MoE intermediate size");
                params.Add("experts", "82", "number of routed experts");
                params.Add("topk", "8", "experts per token");
                params.Add("block", "128", "FP8 scale block size");
                params.Add("input_type", "fp16", "fp16 or bf16");
                params.Add("path", "operator", "operator, legacy or warp");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkMergeMoeFp8Cuda,
            [](const OpTestParams&) {
                return 0.0;
            },
            [](const OpTestParams &params) {
                return (double)params.GetInt("batch") * params.GetInt("topk") *
                       6.0 * (double)params.GetInt("hidden") * params.GetInt("inter");
            },
            true
        };
    }

    struct FusedMoeFp8BenchState {
        int batch = 1;
        int topk = 8;
        int hidden = 2048;
        int inter = 256;
        int experts = 16;
        int block = 128;
        std::string path = "warp";
        fastllm::Data input, index, score;
        fastllm::Data gate, up, down;
        fastllm::Data w1, output, referenceW1, referenceOutput;

        static void InitWeight(fastllm::Data &weight, const std::vector<int> &dims,
                               int block, int seed) {
            weight.dataType = fastllm::DataType::FP8_E4M3;
            weight.UpdateUnitSize();
            weight.Resize(dims);
            weight.weightType = fastllm::WeightType::LINEAR;
            weight.blockK = block;
            weight.blockM = block;
            weight.Allocate(false);
            uint8_t *ptr = reinterpret_cast<uint8_t*>(weight.cpuData);
            for (uint64_t i = 0; i < weight.GetBytes(); i++) {
                ptr[i] = static_cast<uint8_t>(0x20 + ((i + (uint64_t)seed * 17) & 0x1f));
            }
            int scaleRows = (dims[dims.size() - 2] + block - 1) / block;
            int scaleCols = (dims.back() + block - 1) / block;
            weight.scales.resize((size_t)dims[0] * scaleRows * scaleCols);
            for (size_t i = 0; i < weight.scales.size(); i++) {
                weight.scales[i] = 0.015f + 0.0001f * (float)((i + seed) % 7);
            }
            weight.ToDevice(fastllm::DataDevice::CUDA);
            fastllm::Data emptyBias;
            FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, emptyBias, 1);
        }

        bool RunDirect(fastllm::Data &work, fastllm::Data &result,
                       bool allowWarpSpecialization) {
#ifdef USE_CUDA
            if (input.dataType == fastllm::DataType::FLOAT16) {
                return FastllmCudaHalfFusedMOEFP8E4M3(
                    input, gate, up, down, index, score, work, result,
                    batch, topk, hidden, inter, experts, 0.0f,
                    allowWarpSpecialization);
            }
            return FastllmCudaBFloat16FusedMOEFP8E4M3(
                input, gate, up, down, index, score, work, result,
                batch, topk, hidden, inter, experts, 0.0f,
                allowWarpSpecialization);
#else
            (void)work;
            (void)result;
            (void)allowWarpSpecialization;
            return false;
#endif
        }

        void CheckWarpSpecialization() {
#ifdef USE_CUDA
            if (batch != 1 || topk != 8 || hidden != 2048 || inter != 256 || block != 128) {
                return;
            }
            if (!RunDirect(referenceW1, referenceOutput, false)) {
                throw std::runtime_error("legacy FusedMOE FP8 launch failed");
            }
            ForceDeviceSync();
            std::vector<float> expected = ToFloatVector(ConvertToFloat32Data(referenceOutput));
            if (!RunDirect(w1, output, true)) {
                throw std::runtime_error("warp FusedMOE FP8 launch failed");
            }
            ForceDeviceSync();
            std::vector<float> actual = ToFloatVector(ConvertToFloat32Data(output));
            if (expected.size() != actual.size()) {
                throw std::runtime_error("FusedMOE FP8 output size mismatch");
            }
            for (size_t i = 0; i < expected.size(); i++) {
                if (expected[i] != actual[i]) {
                    std::ostringstream os;
                    os << "FusedMOE FP8 warp mismatch at " << i
                       << ": expected=" << expected[i] << ", actual=" << actual[i];
                    throw std::runtime_error(os.str());
                }
            }
#endif
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            batch = params.GetInt("batch");
            topk = params.GetInt("topk");
            hidden = params.GetInt("hidden");
            inter = params.GetInt("inter");
            experts = params.GetInt("experts");
            block = params.GetInt("block");
            path = params.GetString("path");
            FastllmCudaSetDevice(0);

            fastllm::Data fp32Input = MakeTensor({batch, hidden}, 0.11f, 0.02f);
            input.CopyFrom(fp32Input);
            std::string inputType = params.GetString("input_type");
            if (inputType == "fp16") {
                fastllm::ToDataType(input, fastllm::DataType::FLOAT16);
            } else if (inputType == "bf16") {
                fastllm::ToDataType(input, fastllm::DataType::BFLOAT16);
            } else {
                throw std::runtime_error("input_type must be fp16 or bf16");
            }
            input.ToDevice(fastllm::DataDevice::CUDA);

            index.dataType = fastllm::DataType::INT32;
            index.UpdateUnitSize();
            index.Resize({batch, topk});
            index.Allocate(false);
            score.dataType = fastllm::DataType::FLOAT32;
            score.UpdateUnitSize();
            score.Resize({batch, topk});
            score.Allocate(false);
            int32_t *indexPtr = reinterpret_cast<int32_t*>(index.cpuData);
            float *scorePtr = reinterpret_cast<float*>(score.cpuData);
            for (int b = 0; b < batch; b++) {
                for (int j = 0; j < topk; j++) {
                    indexPtr[b * topk + j] = (b * topk + j) % experts;
                    scorePtr[b * topk + j] = 1.0f / std::max(topk, 1);
                }
            }
            index.ToDevice(fastllm::DataDevice::CUDA);
            score.ToDevice(fastllm::DataDevice::CUDA);

            InitWeight(gate, {experts, inter, hidden}, block, 3);
            InitWeight(up, {experts, inter, hidden}, block, 17);
            InitWeight(down, {experts, hidden, inter}, block, 29);
            ForceDeviceSync();
            CheckWarpSpecialization();
#else
            (void)params;
            throw std::runtime_error("fusedmoe_fp8 benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (path == "legacy") {
                if (!RunDirect(w1, output, false)) {
                    throw std::runtime_error("legacy FusedMOE FP8 launch failed");
                }
            } else if (path == "warp") {
                if (!RunDirect(w1, output, true)) {
                    throw std::runtime_error("warp FusedMOE FP8 launch failed");
                }
            } else {
                throw std::runtime_error("path must be legacy or warp");
            }
        }
    };

    static BenchmarkResult BenchmarkFusedMoeFp8Cuda(
            const OpTestParams &params, const std::string &device, int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<FusedMoeFp8BenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        result.flops = (double)params.GetInt("batch") * params.GetInt("topk") *
                       6.0 * (double)params.GetInt("hidden") * params.GetInt("inter");
        double seconds = result.avgMs / 1000.0;
        if (seconds > 0.0) {
            result.computeTFlops = result.flops / seconds / 1e12;
        }
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("fusedmoe_fp8 benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeFusedMoeFp8Case() {
        return {
            "fusedmoe_fp8",
            "benchmark and validate Qwen3.5 fused-weight FP8 MoE",
            []() {
                OpTestParams params;
                params.Add("batch", "1", "token batch size");
                params.Add("hidden", "2048", "Qwen3.5 hidden size");
                params.Add("inter", "256", "TP-local MoE intermediate size");
                params.Add("experts", "16", "number of routed experts in the test table");
                params.Add("topk", "8", "experts per token");
                params.Add("block", "128", "FP8 scale block size");
                params.Add("input_type", "fp16", "fp16 or bf16");
                params.Add("path", "warp", "legacy or warp");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkFusedMoeFp8Cuda,
            [](const OpTestParams&) { return 0.0; },
            [](const OpTestParams &params) {
                return (double)params.GetInt("batch") * params.GetInt("topk") *
                       6.0 * (double)params.GetInt("hidden") * params.GetInt("inter");
            },
            true
        };
    }

    struct RouterLinearFp16BenchState {
        std::string path;
        fastllm::Data input, weight, bias;
        fastllm::Data legacyOutput, fastOutput;

        static void PrepareOutput(fastllm::Data &data) {
            data.dataType = fastllm::DataType::FLOAT16;
            data.UpdateUnitSize();
            data.Resize({1, 256});
            data.Allocate(false);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        void RunOne(fastllm::Data &output, bool allowSpecialization) {
            bool ok = FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
                input, weight, bias, output, 1, 2048, 256, false,
                allowSpecialization);
            if (!ok) {
                throw std::runtime_error("FP16 router GEMV launch failed");
            }
        }

        void Check() {
            RunOne(legacyOutput, false);
            RunOne(fastOutput, true);
            ForceDeviceSync();
            std::vector<float> expected = ToFloatVector(ConvertToFloat32Data(legacyOutput));
            std::vector<float> actual = ToFloatVector(ConvertToFloat32Data(fastOutput));
            if (expected != actual) {
                for (size_t i = 0; i < expected.size(); i++) {
                    if (expected[i] != actual[i]) {
                        std::ostringstream os;
                        os << "FP16 router GEMV mismatch at " << i
                           << ": expected=" << expected[i]
                           << " actual=" << actual[i];
                        throw std::runtime_error(os.str());
                    }
                }
            }
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            FastllmCudaSetDevice(0);
            path = params.GetString("path");
            input.CopyFrom(MakeTensor({1, 2048}, 0.271f, 0.25f));
            fastllm::ToDataType(input, fastllm::DataType::FLOAT16);
            input.ToDevice(fastllm::DataDevice::CUDA);
            weight.CopyFrom(MakeTensor({256, 2048}, 0.619f, 0.125f));
            fastllm::ToDataType(weight, fastllm::DataType::FLOAT16);
            weight.ToDevice(fastllm::DataDevice::CUDA);
            PrepareOutput(legacyOutput);
            PrepareOutput(fastOutput);
            Check();
#else
            (void)params;
            throw std::runtime_error("router_linear_fp16 benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (path == "fast") {
                RunOne(fastOutput, true);
            } else if (path == "legacy") {
                RunOne(legacyOutput, false);
            } else {
                throw std::runtime_error("path must be legacy or fast");
            }
        }
    };
    static BenchmarkResult BenchmarkRouterLinearFp16Cuda(
            const OpTestParams &params, const std::string &device, int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<RouterLinearFp16BenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("router_linear_fp16 benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeRouterLinearFp16Case() {
        return {
            "router_linear_fp16",
            "benchmark and validate Qwen3.5 batch-1 FP16 router GEMV",
            []() {
                OpTestParams params;
                params.Add("path", "fast", "legacy or fast");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkRouterLinearFp16Cuda,
            [](const OpTestParams&) {
                return (double)(2048 + 256 * 2048 + 256) * 2.0;
            },
            [](const OpTestParams&) {
                return 2.0 * 2048.0 * 256.0;
            },
            true
        };
    }

    struct FusedRouterTopKBenchState {
        int batch = 1;
        bool withBias = true;
        bool needNorm = true;
        float routeScale = 1.0f;
        std::string path;
        fastllm::Data logits, bias;
        fastllm::Data fusedIndex, fusedScore;
        fastllm::Data referenceLogits, referenceProb, referenceIndex, referenceScore;

        static void PrepareOutput(fastllm::Data &data, fastllm::DataType type,
                                  const std::vector<int> &dims) {
            data.dataType = type;
            data.UpdateUnitSize();
            data.Resize(dims);
            data.Allocate(false);
            data.ToDevice(fastllm::DataDevice::CUDA);
        }

        void RunReference() {
            fastllm::ToDataType(logits, referenceLogits, fastllm::DataType::FLOAT32);
            fastllm::Softmax(referenceLogits, referenceProb, -1);
            fastllm::SelectExpert(referenceProb, referenceIndex, referenceScore, 8,
                                  needNorm, routeScale, withBias ? &bias : nullptr);
        }

        void RunFused() {
            bool ok = FastllmCudaFusedSoftmaxSelectExpert(
                logits, withBias ? &bias : nullptr, fusedIndex, fusedScore,
                8, needNorm, routeScale);
            if (!ok) {
                throw std::runtime_error("fused router top-k launch failed");
            }
        }

        void Check() {
            RunReference();
            RunFused();
            ForceDeviceSync();
            std::vector<int32_t> expectedIndex = ToInt32Vector(referenceIndex);
            std::vector<int32_t> actualIndex = ToInt32Vector(fusedIndex);
            if (expectedIndex != actualIndex) {
                std::ostringstream os;
                os << "fused router index mismatch:";
                size_t mismatch = 0;
                for (size_t i = 0; i < expectedIndex.size(); i++) {
                    if (expectedIndex[i] != actualIndex[i]) {
                        os << " pos=" << i << " expected=" << expectedIndex[i]
                           << " actual=" << actualIndex[i];
                        mismatch = i;
                        break;
                    }
                }
                size_t tokenStart = (mismatch / 8) * 8;
                os << " expected_top8=";
                for (size_t i = tokenStart; i < tokenStart + 8; i++) {
                    os << (i == tokenStart ? "[" : ",") << expectedIndex[i];
                }
                os << "] actual_top8=";
                for (size_t i = tokenStart; i < tokenStart + 8; i++) {
                    os << (i == tokenStart ? "[" : ",") << actualIndex[i];
                }
                os << "] expected_prob=";
                std::vector<float> referenceProbValues = ToFloatVector(referenceProb);
                size_t probStart = (mismatch / 8) * 256;
                for (size_t i = tokenStart; i < tokenStart + 8; i++) {
                    os << (i == tokenStart ? "[" : ",")
                       << referenceProbValues[probStart + expectedIndex[i]];
                }
                os << "]";
                throw std::runtime_error(os.str());
            }
            std::vector<float> expectedScore = ToFloatVector(referenceScore);
            std::vector<float> actualScore = ToFloatVector(fusedScore);
            float maxAbsDiff = 0.0f;
            for (size_t i = 0; i < expectedScore.size(); i++) {
                maxAbsDiff = std::max(maxAbsDiff, std::fabs(expectedScore[i] - actualScore[i]));
            }
            if (maxAbsDiff > 0.0f) {
                std::ostringstream os;
                os << "fused router score mismatch: max_abs_diff=" << maxAbsDiff;
                throw std::runtime_error(os.str());
            }
        }

        void Init(const OpTestParams &params) {
#ifdef USE_CUDA
            FastllmCudaSetDevice(0);
            batch = params.GetInt("batch");
            withBias = params.GetInt("bias") != 0;
            needNorm = params.GetInt("norm") != 0;
            routeScale = params.GetFloat("route_scale");
            path = params.GetString("path");

            std::string pattern = params.GetString("pattern");
            fastllm::Data fp32Logits;
            if (pattern == "unique") {
                fastllm::Data generated = MakeTensor({batch, 256}, 0.731f, 1.7f);
                fp32Logits.CopyFrom(generated);
            } else if (pattern == "tied") {
                std::vector<float> values((size_t)batch * 256);
                for (int token = 0; token < batch; token++) {
                    for (int expert = 0; expert < 256; expert++) {
                        values[(size_t)token * 256 + expert] =
                            (float)((expert * 17 + token * 13) & 7) * 0.25f;
                    }
                }
                fastllm::Data generated(fastllm::DataType::FLOAT32,
                                        {batch, 256}, values);
                fp32Logits.CopyFrom(generated);
            } else {
                throw std::runtime_error("pattern must be unique or tied");
            }
            logits.CopyFrom(fp32Logits);
            std::string inputType = params.GetString("input_type");
            if (inputType == "fp16") {
                fastllm::ToDataType(logits, fastllm::DataType::FLOAT16);
            } else if (inputType == "bf16") {
                fastllm::ToDataType(logits, fastllm::DataType::BFLOAT16);
            } else if (inputType != "fp32") {
                throw std::runtime_error("input_type must be fp16, bf16 or fp32");
            }
            logits.ToDevice(fastllm::DataDevice::CUDA);

            fastllm::Data fp32Bias;
            if (pattern == "unique") {
                fastllm::Data generated = MakeTensor({256}, 1.217f, 0.015f);
                fp32Bias.CopyFrom(generated);
            } else {
                std::vector<float> values(256);
                for (int expert = 0; expert < 256; expert++) {
                    values[expert] = (float)(expert & 3) * 0.001f;
                }
                fastllm::Data generated(fastllm::DataType::FLOAT32,
                                        {256}, values);
                fp32Bias.CopyFrom(generated);
            }
            bias.CopyFrom(fp32Bias);
            bias.ToDevice(fastllm::DataDevice::CUDA);
            PrepareOutput(fusedIndex, fastllm::DataType::INT32, {batch, 8});
            PrepareOutput(fusedScore, fastllm::DataType::FLOAT32, {batch, 8});
            Check();
#else
            (void)params;
            throw std::runtime_error("fused_router_topk benchmark requires USE_CUDA");
#endif
        }

        void Run() {
            if (path == "fused") {
                RunFused();
            } else if (path == "reference") {
                RunReference();
            } else {
                throw std::runtime_error("path must be fused or reference");
            }
        }
    };

    static BenchmarkResult BenchmarkFusedRouterTopKCuda(
            const OpTestParams &params, const std::string &device, int warmup, int iters) {
#ifdef USE_CUDA
        ScopedFirstDevice guard(device);
        auto state = std::make_shared<FusedRouterTopKBenchState>();
        state->Init(params);
        for (int i = 0; i < warmup; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto begin = Clock::now();
        for (int i = 0; i < iters; i++) {
            state->Run();
        }
        ForceDeviceSync();
        auto end = Clock::now();
        BenchmarkResult result;
        result.avgMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                       std::max(iters, 1);
        return result;
#else
        (void)params;
        (void)device;
        (void)warmup;
        (void)iters;
        throw std::runtime_error("fused_router_topk benchmark requires USE_CUDA");
#endif
    }

    static OpCase MakeFusedRouterTopKCase() {
        return {
            "fused_router_topk",
            "benchmark and validate fused softmax + SelectExpert top8/256",
            []() {
                OpTestParams params;
                params.Add("batch", "1", "token batch size");
                params.Add("input_type", "fp16", "fp16, bf16 or fp32");
                params.Add("bias", "1", "whether correction bias is present");
                params.Add("norm", "1", "normalize selected probabilities");
                params.Add("route_scale", "1.0", "router score scale");
                params.Add("path", "fused", "fused or reference");
                params.Add("pattern", "unique", "unique or tied ranking keys");
                return params;
            },
            [](const OpTestParams&, const std::string &device) {
                return device.rfind("cuda", 0) == 0;
            },
            [](const OpTestParams&, const std::string&) {
                fastllm::Data marker(fastllm::DataType::FLOAT32, {1});
                marker.Allocate(0.0f);
                return marker;
            },
            BenchmarkFusedRouterTopKCuda,
            [](const OpTestParams&) { return 0.0; },
            [](const OpTestParams&) { return 0.0; },
            true
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
            MakeAttentionCase(),
            MakeRmsNormFp16SpecializedCase(),
            MakeRecurrentFromConvFp16Case(),
            MakeLinearFp8Block128Case(),
            MakeMergeMoeFp8Case(),
            MakeFusedMoeFp8Case(),
            MakeRouterLinearFp16Case(),
            MakeFusedRouterTopKCase()
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

        if (opCase.benchmarkOnly) {
            bool ran = false;
            for (const auto &device : devices) {
                if (!DeviceSelected(config.deviceFilters, device)) {
                    continue;
                }
                if (!opCase.canRun(params, device)) {
                    std::cout << "  [" << device << "] skipped: op not supported on this device\n";
                    continue;
                }

                BenchmarkResult bench = Benchmark(opCase, params, device, config.warmup, config.iters);
                std::cout << "  [" << device << "] BENCHMARK\n";
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
                ran = true;
            }
            return ran;
        }

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
