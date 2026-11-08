"""Run the actual snapshot copy planner on CPU-only tensor and worker stubs.

Checks content/metadata, all-rank completion before publication, one submission
per device, unavailable-dispatch fallback, invalid sources, and worker failures.
The source extractor does not import FastLLM or touch CUDA/model weights.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent


def record(action, **fields):
    row = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "action": action, **fields}
    with (PROJECT / "logs/actions.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def extract(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
        end += 1
    return source[start:end]


STUBS = r"""
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
enum class DataDevice { CPU, CUDA };
enum class DataType { FLOAT16, BFLOAT16 };
struct Data {
    std::vector<int> dims{1, 8}, dataDeviceIds, tpRanges, tpGlobalDims;
    int tpLayout = 7, tpAxis = 2;
    DataType dataType = DataType::BFLOAT16;
    bool multiDeviceData = false, isLinearAttentionTransposed = true;
    DataDevice dataDevice = DataDevice::CUDA;
    std::map<int, Data*> multiDeviceDatas;
    std::vector<unsigned char> bytes;
    void *cudaData = nullptr;
    unsigned char *cpuData = nullptr;
    bool fail = false, raise = false;
    int pointerDevice = -1;
    std::atomic<int> reads{0};
    explicit Data(int device = -1) : dataDeviceIds(device < 0 ? std::vector<int>{} : std::vector<int>{device}),
                                    bytes(16, static_cast<unsigned char>(device + 31)), pointerDevice(device) {
        if (device < 0) { dataDevice = DataDevice::CPU; cpuData = bytes.data(); }
        else cudaData = this;
        tpRanges = {2, 5}; tpGlobalDims = {3, 9};
    }
};
struct Qwen35LinearPrefixSnapshotCache {
    bool valid = false, multiDevice = false, transposed = false;
    Data single;
    std::map<int, Data> locals;
    std::vector<int> dataDeviceIds, tpRanges, tpGlobalDims, dims;
    int tpLayout = 0, tpAxis = -1;
    DataType dataType = DataType::FLOAT16;
};
std::vector<Qwen35LinearPrefixSnapshotCache*> destinations;
std::atomic<int> active{0}, peak{0}, finished{0}, callbackCount{0};
std::vector<int> dispatchDevices;
bool dispatcherAvailable = true;
thread_local int currentDevice = -1;
int GetPointerDeviceId(void *p) { return static_cast<Data*>(p)->pointerDevice; }
bool Qwen35SnapshotCopyTensor(const Data &source, Data &dst) {
    for (auto *cache : destinations) assert(!cache->valid);
    auto &src = const_cast<Data&>(source); ++src.reads;
    const int now = ++active;
    int before = peak.load();
    while (before < now && !peak.compare_exchange_weak(before, now)) {}
    struct Exit { ~Exit() { --active; ++finished; } } exit;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (src.raise) throw std::runtime_error("injected rank copy exception");
    if (src.fail) return false;
    dst.bytes = src.bytes; dst.cpuData = dst.bytes.data(); dst.cudaData = nullptr;
    dst.dataDevice = DataDevice::CPU;
    return true;
}
bool MultiCudaRunDeviceCallbacks(const std::vector<int> &devices,
        const std::function<void(int, int)> &callback) {
    if (!dispatcherAvailable) return false;
    dispatchDevices = devices;
    std::vector<std::exception_ptr> errors(devices.size());
    std::vector<std::thread> workers;
    for (int rank = 0; rank < static_cast<int>(devices.size()); ++rank) {
        workers.emplace_back([&, rank] {
            currentDevice = devices[rank]; ++callbackCount;
            try { callback(rank, devices[rank]); }
            catch (...) { errors[rank] = std::current_exception(); }
        });
    }
    for (auto &worker : workers) worker.join();
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
    return true;
}
void reset() {
    assert(active == 0);
    destinations.clear(); dispatchDevices.clear();
    peak = 0; finished = 0; callbackCount = 0; dispatcherAvailable = true;
}
"""

CASES = r"""
using Entry = std::pair<const Data*, Qwen35LinearPrefixSnapshotCache*>;
void metadata(const Data &src, const Qwen35LinearPrefixSnapshotCache &dst) {
    assert(dst.dataDeviceIds == src.dataDeviceIds && dst.tpLayout == src.tpLayout);
    assert(dst.tpAxis == src.tpAxis && dst.tpRanges == src.tpRanges);
    assert(dst.tpGlobalDims == src.tpGlobalDims && dst.dims == src.dims);
    assert(dst.dataType == src.dataType && dst.transposed == src.isLinearAttentionTransposed);
}
struct Fixture {
    std::vector<std::unique_ptr<Data>> roots, locals;
    std::vector<std::unique_ptr<Qwen35LinearPrefixSnapshotCache>> output;
    std::vector<Entry> entries;
    explicit Fixture(int cacheCount = 4) {
        for (int n = 0; n < cacheCount; ++n) {
            roots.emplace_back(new Data());
            auto &root = *roots.back(); root.multiDeviceData = true;
            root.dataDeviceIds = {5, 2, 9, 1};
            for (int device : root.dataDeviceIds) {
                locals.emplace_back(new Data(device));
                locals.back()->bytes[0] = static_cast<unsigned char>(n + 60);
                root.multiDeviceDatas.emplace(device, locals.back().get());
            }
            output.emplace_back(new Qwen35LinearPrefixSnapshotCache());
            entries.emplace_back(&root, output.back().get());
            destinations.push_back(output.back().get());
        }
    }
    void validate(bool expected) {
        for (size_t n = 0; n < roots.size(); ++n) {
            const auto &src = *roots[n]; const auto &dst = *output[n];
            assert(dst.valid == expected);
            if (!expected) continue;
            metadata(src, dst); assert(dst.multiDevice && dst.locals.size() == 4);
            for (const auto &local : src.multiDeviceDatas) {
                assert(dst.locals.at(local.first).bytes == local.second->bytes);
                assert(dst.locals.at(local.first).cpuData != local.second->cpuData);
                assert(local.second->reads == 1);
            }
        }
        assert(active == 0);
    }
};
int main() {
    reset(); {
        Data host; Qwen35LinearPrefixSnapshotCache dst; destinations = {&dst};
        assert(Qwen35SnapshotCopyCaches({{&host, &dst}}));
        assert(dst.valid && dst.single.bytes == host.bytes && callbackCount == 0);
        metadata(host, dst);
    }
    reset(); {
        Data a(2), b(2); Qwen35LinearPrefixSnapshotCache x, y; destinations = {&x, &y};
        assert(Qwen35SnapshotCopyCaches({{&a, &x}, {&b, &y}}));
        assert(x.valid && y.valid && peak == 1 && callbackCount == 0);
        assert(a.reads == 1 && b.reads == 1);
    }
    reset(); {
        Fixture fixture; assert(Qwen35SnapshotCopyCaches(fixture.entries)); fixture.validate(true);
        assert(callbackCount == 4 && peak > 1 && peak <= 4);
        assert((dispatchDevices == std::vector<int>{1, 2, 5, 9}));
    }
    reset(); {
        dispatcherAvailable = false; Fixture fixture;
        assert(Qwen35SnapshotCopyCaches(fixture.entries)); fixture.validate(true);
        assert(callbackCount == 0 && peak == 1 && finished == 16);
    }
    reset(); {
        Fixture fixture; fixture.locals[0]->fail = true;
        assert(!Qwen35SnapshotCopyCaches(fixture.entries)); fixture.validate(false);
        assert(callbackCount == 4 && active == 0);
    }
    reset(); {
        Fixture fixture; fixture.locals[0]->raise = true; bool caught = false;
        try { Qwen35SnapshotCopyCaches(fixture.entries); }
        catch (const std::runtime_error&) { caught = true; }
        assert(caught); fixture.validate(false); assert(callbackCount == 4 && active == 0);
    }
    reset(); {
        Data invalid(2); Qwen35LinearPrefixSnapshotCache dst; invalid.cudaData = nullptr;
        destinations = {&dst}; assert(!Qwen35SnapshotCopyCaches({{&invalid, &dst}}));
        assert(!dst.valid && callbackCount == 0 && invalid.reads == 0);
        assert(!Qwen35SnapshotCopyCaches({{nullptr, &dst}}));
        assert(!Qwen35SnapshotCopyCaches({{&invalid, nullptr}}));
    }
    reset(); {
        Data invalid(2); Qwen35LinearPrefixSnapshotCache dst; invalid.dims.clear();
        destinations = {&dst}; assert(!Qwen35SnapshotCopyCaches({{&invalid, &dst}}));
        assert(!dst.valid && callbackCount == 0 && invalid.reads == 0);
    }
    reset(); {
        Data source(2); source.dataDeviceIds.clear(); Qwen35LinearPrefixSnapshotCache dst;
        destinations = {&dst}; assert(Qwen35SnapshotCopyCaches({{&source, &dst}}));
        assert(dst.valid && source.reads == 1);
    }
    reset(); {
        Data invalid(2); invalid.dataDeviceIds.clear(); invalid.pointerDevice = -1;
        Qwen35LinearPrefixSnapshotCache dst; destinations = {&dst};
        assert(!Qwen35SnapshotCopyCaches({{&invalid, &dst}})); assert(!dst.valid);
    }
    reset(); {
        Data root; root.multiDeviceData = true; root.multiDeviceDatas[2] = nullptr;
        Qwen35LinearPrefixSnapshotCache dst; destinations = {&dst};
        assert(!Qwen35SnapshotCopyCaches({{&root, &dst}})); assert(!dst.valid);
    }
    reset(); {
        Data root; root.multiDeviceData = true; Data local(2);
        root.multiDeviceDatas[1] = nullptr; root.multiDeviceDatas[2] = &local;
        Qwen35LinearPrefixSnapshotCache dst; destinations = {&dst};
        assert(Qwen35SnapshotCopyCaches({{&root, &dst}}));
        assert(dst.valid && dst.locals.size() == 1 && dst.locals.at(2).bytes == local.bytes);
    }
    reset(); std::cout << "PASS: actual snapshot planner content, metadata, parallel/join, fallback, invalid, failure and exception gates\n";
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--log-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    build, logs = args.build_dir.resolve(), args.log_dir.resolve()
    assert build.is_relative_to(PROJECT / "build")
    assert logs.is_relative_to(PROJECT / "logs")
    source_path = ROOT / "src/models/qwen3_5.cpp"
    record("parallel_snapshot_read", phase="before", path=str(source_path))
    source = source_path.read_text()
    helper = extract(source, "        static bool Qwen35SnapshotCopyCaches(")
    record("parallel_snapshot_read", phase="after", sha256=hashlib.sha256(source.encode()).hexdigest())
    code = STUBS + "\n" + helper + "\n" + CASES
    flags = "-std=c++17 -O2 -pthread -Wall -Wextra -Werror"
    if args.sanitize:
        flags += " -fsanitize=address,undefined -fno-omit-frame-pointer"
    record("parallel_snapshot_stage", phase="before", build=str(build), logs=str(logs))
    build.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    files = {"probe.cpp": code, "Makefile":
             "CXX := /usr/bin/c++\nFLAGS := " + flags +
             "\nall: probe\nprobe: probe.cpp\n\t$(CXX) $(FLAGS) $< -o $@\n"}
    for name, content in files.items():
        with (build / name).open("x") as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
    record("parallel_snapshot_stage", phase="after", helper_sha256=hashlib.sha256(helper.encode()).hexdigest())
    for label, command in (("compile", ["make", "-j30"]), ("run", ["./probe"])):
        target = logs / (label + ".log")
        record("parallel_snapshot_" + label, phase="before", argv=command,
               cwd=str(build), log=str(target), gpu_requests=0)
        with target.open("xb") as output:
            process = subprocess.run(command, cwd=build, stdout=output, stderr=subprocess.STDOUT)
            output.flush()
            os.fsync(output.fileno())
        record("parallel_snapshot_" + label, phase="after", exit_code=process.returncode)
        if process.returncode:
            raise SystemExit(process.returncode)
    result = {"passed": True, "scope": "actual production planner with CPU tensor/CUDA-worker stubs; no GPU performance acceptance",
              "sanitize": args.sanitize, "build_jobs": 30, "model_requests": 0,
              "helper_sha256": hashlib.sha256(helper.encode()).hexdigest()}
    record("parallel_snapshot_result", phase="before", path=str(logs / "RESULT.json"))
    with (logs / "RESULT.json").open("x") as out:
        json.dump(result, out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    record("parallel_snapshot_result", phase="after", result=result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
