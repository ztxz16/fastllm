#include "distributed/process_group.h"
#include "distributed/qwen3_shard_loader.h"
#include "json11.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

using namespace fastllm::distributed;
using json11::Json;
namespace fs = std::filesystem;

namespace {
uint64_t Number(const std::string &text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("expected unsigned integer: " + text);
    }
    return std::stoull(text);
}

/**
 * 通过公开 ProcessGroup 接口分块读回实际驻留的权重，协调器不加载源权重 payload。
 * @param group 已加载模型的进程组。
 * @param model 本机 HF 目录，仅用于独立取得张量名称/形状和预期内存大小。
 * @param infos worker 返回的驻留统计，必须与元数据计划一致。
 * @param directory 新的输出目录，不覆盖历史数据。
 * @return 每个 rank 的元数据及读回文件，数值对比由 Python 独立参考完成。
 */
Json Dump(ProcessGroup &group, const std::string &model, const std::vector<ProcessModelInfo> &infos,
          const fs::path &directory) {
    if (!fs::create_directory(directory)) {
        throw std::runtime_error("dump directory already exists");
    }
    Json::array ranks;
    for (size_t rank = 0; rank < infos.size(); ++rank) {
        Qwen3ShardLoader loader(model, static_cast<int>(rank), static_cast<int>(infos.size()));
        const auto &plan = loader.Plan();
        if (infos[rank].tensorCount != plan.tensors.size()) {
            throw std::runtime_error("worker tensor count differs from plan");
        }
        uint64_t expectedBytes = 0, expectedRead = 0;
        Json::array entries;
        for (size_t index = 0; index < plan.tensors.size(); ++index) {
            const auto &tensor = plan.tensors[index];
            uint64_t bytes = tensor.bytes / (tensor.dtype == "F32" ? 4 : 2) * 2;
            expectedBytes += bytes;
            expectedRead += tensor.zeroFill ? 0 : tensor.bytes;
            const std::string filename = "rank" + std::to_string(rank) + "-tensor" + std::to_string(index) + ".bin";
            std::ofstream file(directory / filename, std::ios::binary);
            if (!file) {
                throw std::runtime_error("cannot open GPU dump: " + filename);
            }
            for (uint64_t offset = 0; offset < bytes;) {
                uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(1024 * 1024, bytes - offset));
                auto data = group.ReadQwen3Weight(static_cast<int>(rank), static_cast<uint32_t>(index), offset, count);
                file.write(reinterpret_cast<const char*>(data.data()), data.size());
                if (!file) {
                    throw std::runtime_error("cannot write GPU dump: " + filename);
                }
                offset += count;
            }
            file.close();
            if (!file) {
                throw std::runtime_error("cannot close GPU dump: " + filename);
            }
            Json::array shape;
            for (auto dimension : tensor.shape) {
                shape.push_back(static_cast<double>(dimension));
            }
            entries.push_back(Json::object{{"name", tensor.name}, {"shape", shape}, {"dtype", "F16"},
                {"source_dtype", tensor.dtype}, {"bytes", static_cast<double>(bytes)}, {"file", filename}});
        }
        if (infos[rank].deviceBytes != expectedBytes || infos[rank].sourceBytesRead != expectedRead) {
            throw std::runtime_error("worker weight byte accounting differs from plan");
        }
        ranks.push_back(Json::object{{"rank", int(rank)}, {"device_bytes", static_cast<double>(expectedBytes)},
            {"source_bytes_read", static_cast<double>(expectedRead)}, {"tensors", entries}});
    }
    return ranks;
}
} // namespace

/**
 * 驱动真实 worker 加载、读回及释放权重；不执行 Qwen3 前向，也不报告性能。
 * @param argc/argv 必填 --worker、--model、--output；可选 --devices、--case、内存配额。
 * @return 0 表示流程完成，1 表示失败；数值正确性须由外部参考验证读回文件。
 */
int main(int argc, char **argv) {
    Json::object report{{"status", "FAIL"}, {"backend", "qwen3_process_cuda_fp16"}};
    ProcessGroup group;
    try {
        std::map<std::string, std::string> args;
        const std::vector<std::string> allowed = {"--worker", "--model", "--reload-model", "--output", "--devices", "--case",
            "--max-tensor-bytes", "--max-device-bytes"};
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || std::find(allowed.begin(), allowed.end(), argv[i]) == allowed.end() ||
                !args.emplace(argv[i], argv[i + 1]).second) {
                throw std::invalid_argument("unknown/duplicate/incomplete checker argument");
            }
        }
        ProcessGroupOptions options;
        options.workerPath = fs::absolute(args.at("--worker")).string();
        options.operationTimeoutMs = 30000;
        options.shutdownTimeoutMs = 30000;
        std::string devices = args.count("--devices") ? args.at("--devices") : "0,1";
        if (devices.empty() || devices.back() == ',') {
            throw std::invalid_argument("invalid devices");
        }
        std::istringstream input(devices);
        for (std::string item; std::getline(input, item, ',');) {
            uint64_t value = Number(item);
            if (value > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("device index overflow");
            }
            options.devices.push_back(static_cast<int>(value));
        }
        const std::string model = fs::absolute(args.at("--model")).string();
        const fs::path output = fs::absolute(args.at("--output"));
        if (!fs::create_directory(output)) {
            throw std::runtime_error("output directory must not exist");
        }
        const std::string scenario = args.count("--case") ? args.at("--case") : "load";
        if (scenario != "load" && scenario != "reload" && scenario != "duplicate" &&
            scenario != "unloaded_read" && scenario != "invalid_index" && scenario != "invalid_range" &&
            scenario != "stop_loaded") {
            throw std::invalid_argument("unknown model checker case");
        }
        if (args.count("--reload-model") && scenario != "reload") {
            throw std::invalid_argument("--reload-model requires --case reload");
        }
        const std::string reloadModel = args.count("--reload-model") ?
            fs::absolute(args.at("--reload-model")).string() : model;
        uint64_t maxTensor = args.count("--max-tensor-bytes") ? Number(args.at("--max-tensor-bytes")) : 1073741824ULL;
        uint64_t maxDevice = args.count("--max-device-bytes") ? Number(args.at("--max-device-bytes")) : 8589934592ULL;
        group.Start(options);
        Json::array workers;
        for (const auto &worker : group.Workers()) {
            workers.push_back(Json::object{{"rank", worker.rank}, {"device", worker.device}, {"pid", worker.pid},
                {"gpu", worker.deviceName}, {"cuda_runtime_version", worker.cudaRuntimeVersion},
                {"nccl_version", worker.ncclVersion}});
        }
        report["workers"] = workers;
        Json::array cycles;
        for (int iteration = 0; iteration < (scenario == "reload" ? 2 : 1); ++iteration) {
            const std::string &activeModel = iteration == 0 ? model : reloadModel;
            auto infos = group.LoadQwen3(activeModel, maxTensor, maxDevice);
            if (scenario == "duplicate") {
                group.LoadQwen3(model, maxTensor, maxDevice);
                throw std::runtime_error("duplicate load unexpectedly accepted");
            }
            if (scenario == "invalid_index") {
                group.ReadQwen3Weight(0, infos[0].tensorCount, 0, 2);
                throw std::runtime_error("invalid index unexpectedly accepted");
            }
            if (scenario == "invalid_range") {
                group.ReadQwen3Weight(0, 0, std::numeric_limits<uint64_t>::max() - 1, 2);
                throw std::runtime_error("invalid range unexpectedly accepted");
            }
            auto sums = group.CheckAllReduce();
            const float expected = float(options.devices.size() * (options.devices.size() + 1) / 2);
            if (std::any_of(sums.begin(), sums.end(), [expected](float value) { return value != expected; })) {
                throw std::runtime_error("NCCL result changed after weight loading");
            }
            std::string cycle = "cycle-" + std::to_string(iteration);
            cycles.push_back(Json::object{{"directory", cycle}, {"model", activeModel},
                {"ranks", Dump(group, activeModel, infos, output / cycle)}});
            report["cycles"] = cycles;
            if (scenario != "stop_loaded") {
                group.UnloadQwen3();
                group.UnloadQwen3();
            }
            if (scenario == "unloaded_read") {
                group.ReadQwen3Weight(0, 0, 0, 2);
                throw std::runtime_error("unloaded read unexpectedly accepted");
            }
            group.Ping();
        }
        group.Stop();
        if (!group.LastError().empty() || !group.Workers().empty()) {
            throw std::runtime_error("model shutdown failed: " + group.LastError());
        }
        report["status"] = "PASS";
    } catch (const std::exception &error) {
        report["error"] = error.what();
        report["group_failed"] = group.State() == ProcessGroupState::Failed;
        report["workers_reaped"] = group.Workers().empty();
        try {
            group.Stop();
        } catch (const std::exception &cleanup) {
            report["cleanup_error"] = cleanup.what();
        }
    }
    std::cout << Json(report).dump() << std::endl;
    return report["status"].string_value() == "PASS" ? 0 : 1;
}
