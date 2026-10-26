#include "utils/autotune.h"
#include "devices/disk/diskdevice.h"
#include "json11.hpp"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include <chrono>
#include <cstdio>
#include <mutex>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

namespace fastllm {
std::atomic<bool> autotuneRecording{false};
namespace {
using Json = json11::Json;
using Clock = std::chrono::steady_clock;
std::mutex mutex;
FILE *file = nullptr;
size_t written = 0;
bool failed = false;
thread_local int depth = 0;
constexpr size_t byteLimit = 64 * 1024 * 1024;
double Now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
void FailCapture() {
    std::lock_guard<std::mutex> guard(mutex);
    failed = true;
    autotuneRecording.store(false, std::memory_order_relaxed);
}
void Compact(const Json &value, std::string &out) {
    // This repository's json11::dump pretty-prints containers across lines.
    // Keep records independently parseable without changing the shared serializer.
    if (value.is_array()) {
        out += '[';
        bool first = true;
        for (const auto &item : value.array_items()) {
            if (!first) out += ',';
            first = false; Compact(item, out);
        }
        out += ']';
    } else if (value.is_object()) {
        out += '{';
        bool first = true;
        for (const auto &item : value.object_items()) {
            if (!first) out += ',';
            first = false; out += Json(item.first).dump(); out += ':'; Compact(item.second, out);
        }
        out += '}';
    } else out += value.dump();
}
void Write(const Json &value) {
    std::string line;
    Compact(value, line); line += '\n';
    std::lock_guard<std::mutex> guard(mutex);
    if (!file || failed) return;
    if (line.size() > byteLimit - std::min(written, byteLimit) ||
        fwrite(line.data(), 1, line.size(), file) != line.size()) {
        failed = true;
        autotuneRecording.store(false, std::memory_order_relaxed);
    } else written += line.size();
}
Json Ints(const std::vector<int> &values) {
    Json::array out;
    out.reserve(values.size());
    for (int value : values) out.emplace_back(value);
    return out;
}
}
bool StartAutotuneRecording(const std::string &path) {
    std::lock_guard<std::mutex> guard(mutex);
    if (file || path.empty()) return false;
    file = fopen(path.c_str(), "wb");
    if (!file) return false;
    written = 0; failed = false;
    autotuneRecording.store(true, std::memory_order_relaxed);
    return true;
}
bool StopAutotuneRecording() {
    autotuneRecording.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> guard(mutex);
    if (!file) return false;
    bool ok = !failed;
    // A complete footer distinguishes a valid capture from a killed worker.
    if (ok && fputs("{\"kind\":\"complete\",\"schema\":1}\n", file) < 0) ok = false;
    if (fclose(file)) ok = false;
    file = nullptr;
    return ok;
}
void RecordAutotuneSample(const std::string &kind, const std::vector<int> &key,
                          double units, double ms, int device) noexcept {
    if (!AutotuneRecording() || !std::isfinite(units) || units <= 0 || !std::isfinite(ms) || ms <= 0) return;
    try {
        Write(Json::object{{"kind", kind}, {"key", Ints(key)}, {"units", units},
                       {"ms", ms}, {"device", device}, {"time", Now()}});
    } catch (...) { FailCapture(); }
}
AutotuneSampleScope::AutotuneSampleScope(std::string kind, std::vector<int> key, double units)
    : kind(std::move(kind)), key(std::move(key)), units(units), start(Now()) {}
AutotuneSampleScope::~AutotuneSampleScope() {
    RecordAutotuneSample(kind, key, units, (Now() - start) * 1000);
}
struct AutotuneOpScope::Impl {
    Json::object record;
    double start;
    bool disk = false;
    DiskMoeCacheStats before;
    int cuda = -1;
};
AutotuneOpScope::AutotuneOpScope(const std::string &op, const std::string &backend,
                                const DataDict &datas, const IntDict &params) {
    if (!AutotuneRecording()) return;
    entered = true;
    ++depth;
    if (depth != 1) return;
    try {
        impl.reset(new Impl);
        auto &r = impl->record;
        r = {{"kind", "op"}, {"op", op}, {"backend", backend}};
        auto input = datas.find("input");
        if (input != datas.end() && input->second) r["shape"] = Ints(input->second->dims);
#ifdef USE_CUDA
        impl->cuda = FastllmCudaGetDevice();
        FastllmCudaSyncCurrentThreadStream();
#endif
        if (op == "MergeMOE" && datas.count("index") && datas.count("weights")) {
            const auto &ids = *datas.at("index");
            const size_t count = ids.Count(0);
            // Prevent a malformed or unexpected operator from allocating an unbounded trace.
            if (ids.dataType == INT32 && ids.dims.size() == 2 && count <= 1024 * 1024) {
                std::vector<int> values(count);
#ifdef USE_CUDA
                if (ids.dataDevice == CUDA && ids.cudaData)
                    FastllmCudaCopyFromDeviceToHost(values.data(), ids.cudaData, count * sizeof(int));
                else
#endif
                if (ids.cpuData) memcpy(values.data(), ids.cpuData, count * sizeof(int));
                else values.clear();
                r["ids"] = Ints(values);
                r["rows"] = ids.dims[0]; r["topk"] = ids.dims[1];
            }
            r["layer"] = params.count("layer") ? params.at("layer") : -1;
            const int n = params.count("weights___batch") ? params.at("weights___batch") : 0;
            auto weights = reinterpret_cast<Data **>(datas.at("weights"));
            if (n >= 4 && weights[2] && weights[3]) {
                Json::array sizes;
                for (int e = 2; e + 1 < n; e += 2)
                    sizes.emplace_back(double(weights[e] && weights[e+1] ?
                        weights[e]->GetBytes() + weights[e+1]->GetBytes() : 0));
                r["bytes"] = sizes;
                r["format"] = Ints({int(weights[2]->dataType), weights[2]->ggmlType,
                                       int(weights[3]->dataType), weights[3]->ggmlType});
                r["weight_shape"] = Ints(weights[2]->dims);
            }
            impl->disk = backend == "disk";
            if (impl->disk) impl->before = GetDiskMoeCacheStats();
        }
        r["device"] = impl->cuda;
        impl->start = Now();
        r["start"] = impl->start;
    } catch (...) {
        impl.reset();
        FailCapture();
    }
}
AutotuneOpScope::~AutotuneOpScope() {
    // A nested scope must unwind even if the byte limit disabled recording.
    if (!impl) { if (entered) --depth; return; }
    try {
#ifdef USE_CUDA
        if (impl->cuda >= 0) {
            FastllmCudaSetDevice(impl->cuda);
            FastllmCudaSyncCurrentThreadStream();
        }
#endif
        impl->record["ms"] = (Now() - impl->start) * 1000;
        if (impl->disk) {
            auto after = GetDiskMoeCacheStats();
            impl->record["read_bytes"] = double(after.diskBytes - impl->before.diskBytes);
            impl->record["cpu_hits"] = double(after.cpuHits - impl->before.cpuHits);
            impl->record["gpu_hits"] = double(after.cudaHits - impl->before.cudaHits);
            impl->record["misses"] = double(after.misses - impl->before.misses);
        }
        Write(impl->record);
    } catch (...) { FailCapture(); }
    --depth;
}
}
