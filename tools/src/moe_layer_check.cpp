#include "distributed/process_group.h"
#include "distributed/qwen3_shard_loader.h"
#include "json11.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

using namespace fastllm::distributed;
using json11::Json;
namespace fs = std::filesystem;
namespace {
/** 解析任务中的非负 uint32，拒绝缺失、小数、布尔及溢出。 */
uint32_t Integer(const Json &value) {
    double n = value.number_value();
    if (!value.is_number() || !std::isfinite(n) || n < 0 || n > 4294967295.0 || std::floor(n) != n) {
        throw std::runtime_error("invalid MoE task integer");
    }
    return static_cast<uint32_t>(n);
}

/** 转换小模型数值数组，整数仍按整数语义保存，不在 JSON 截断浮点精度。 */
template<class T> Json Values(const std::vector<T> &values) {
    Json::array result;
    for (auto value : values) result.push_back(static_cast<double>(value));
    return result;
}

/**
 * 经公开权重读回接口保存本 rank 的全部 GPU 权重；只用于有界小模型验收。
 * @param group 已加载的进程组。
 * @param model 本地 HF 目录，独立读取元数据取得张量名称与序号。
 * @param directory 新建报告目录。
 * @param infos 实际加载字节计数。
 * @return 各 rank 实际权重清单和二进制文件名；超过 32 MiB 拒绝诊断导出。
 */
Json DumpWeights(ProcessGroup &group, const std::string &model, const fs::path &directory,
                 const std::vector<ProcessModelInfo> &infos) {
    Json::array ranks;
    for (size_t rank = 0; rank < infos.size(); ++rank) {
        Qwen3ShardLoader loader(model, rank, infos.size(), ShardModel::Qwen2Moe);
        const auto &plan = loader.Plan();
        if (infos[rank].deviceBytes > 32ULL * 1024 * 1024) throw std::runtime_error("small-fixture checker weight limit");
        Json::array entries;
        uint64_t sourceBytes = 0, deviceBytes = 0;
        for (size_t index = 0; index < plan.tensors.size(); ++index) {
            const auto &tensor = plan.tensors[index];
            uint64_t bytes = tensor.bytes / (tensor.dtype == "F32" ? 4 : 2) * 2;
            sourceBytes += tensor.bytes;
            deviceBytes += bytes;
            std::string filename = "rank" + std::to_string(rank) + "_" + std::to_string(index) + ".bin";
            std::ofstream file(directory / filename, std::ios::binary);
            for (uint64_t offset = 0; offset < bytes;) {
                auto count = static_cast<uint32_t>(std::min<uint64_t>(1024 * 1024, bytes - offset));
                auto data = group.ReadQwen3Weight(rank, index, offset, count);
                file.write(reinterpret_cast<const char*>(data.data()), data.size());
                offset += count;
            }
            file.close();
            if (!file) throw std::runtime_error("failed GPU weight dump: " + filename);
            entries.push_back(Json::object{{"name", tensor.name}, {"shape", Values(tensor.shape)},
                {"source_dtype", tensor.dtype}, {"file", filename}});
        }
        if (infos[rank].tensorCount != plan.tensors.size() || infos[rank].sourceBytesRead != sourceBytes ||
            infos[rank].deviceBytes != deviceBytes) throw std::runtime_error("MoE weight accounting mismatch");
        ranks.push_back(Json::object{{"rank", int(rank)}, {"tensors", entries},
            {"device_bytes", double(deviceBytes)}, {"source_bytes_read", double(sourceBytes)}});
    }
    return ranks;
}
} // namespace

/**
 * 通过公开 ProcessGroup MoE 接口验证权重和单层；不执行完整模型生成或性能计时。
 * @param argc 参数数量。
 * @param argv 必填 --worker、--model、--tasks、--output、--devices；--case 用于拒绝路径。
 * @return 成功 0，任一失败 1；stdout JSON 保留已完成任务及通信组回收证据。
 */
int main(int argc, char **argv) {
    Json::object report{{"status", "FAIL"}, {"backend", "qwen2_moe_attention_dp_ep_fp16_nccl"}};
    ProcessGroup group;
    try {
        std::map<std::string, std::string> args;
        const std::vector<std::string> allowed = {"--worker", "--model", "--tasks", "--output", "--devices", "--case"};
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc || std::find(allowed.begin(), allowed.end(), argv[i]) == allowed.end() ||
                !args.emplace(argv[i], argv[i + 1]).second) throw std::runtime_error("invalid checker argument");
        }
        const auto taskPath = fs::absolute(args.at("--tasks"));
        if (fs::file_size(taskPath) > 16 * 1024 * 1024) throw std::runtime_error("tasks exceed 16 MiB");
        std::ifstream file(taskPath);
        std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()), error;
        auto tasks = Json::parse(source, error);
        if (!file || !error.empty() || !tasks.is_array() || tasks.array_items().empty() || tasks.array_items().size() > 64) {
            throw std::runtime_error("invalid MoE task JSON");
        }
        ProcessGroupOptions options;
        options.workerPath = fs::absolute(args.at("--worker")).string();
        options.operationTimeoutMs = options.shutdownTimeoutMs = 30000;
        std::string devices = args.at("--devices");
        if (devices.empty() || devices.back() == ',') throw std::runtime_error("invalid devices");
        std::istringstream stream(devices);
        for (std::string item; std::getline(stream, item, ',');) {
            if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("invalid device");
            options.devices.push_back(std::stoi(item));
        }
        std::string scenario = args.count("--case") ? args.at("--case") : "forward";
        const std::vector<std::string> scenarios = {"forward", "unloaded", "double_load", "dense_api", "graph", "prefix", "source_limit", "device_limit"};
        if (std::find(scenarios.begin(), scenarios.end(), scenario) == scenarios.end()) throw std::runtime_error("invalid scenario");
        options.cudaGraph = scenario == "graph";
        options.prefixCache = scenario == "prefix";
        group.Start(options);
        Json::array workers;
        for (const auto &w : group.Workers()) {
            workers.push_back(Json::object{{"rank", w.rank}, {"device", w.device}, {"pid", w.pid},
                {"gpu", w.deviceName}, {"cuda_runtime_version", w.cudaRuntimeVersion}, {"nccl_version", w.ncclVersion}});
        }
        report["workers"] = workers;
        const auto model = fs::absolute(args.at("--model")).string();
        const auto output = fs::absolute(args.at("--output"));
        if (!fs::create_directory(output)) throw std::runtime_error("output directory must be new");
        auto infos = group.LoadQwenMoe(model, scenario == "source_limit" ? 1 : 1073741824ULL,
                                      scenario == "device_limit" ? 1 : 12884901888ULL);
        report["weights"] = DumpWeights(group, model, output, infos);
        if (scenario == "unloaded") group.UnloadQwen3();
        if (scenario == "double_load") group.LoadQwenMoe(model);
        Json::array results;
        for (const auto &task : tasks.array_items()) {
            if (!task["inputs"].is_array()) throw std::runtime_error("missing rank inputs");
            std::vector<QwenMoeLayerInput> inputs;
            for (const auto &item : task["inputs"].array_items()) {
                QwenMoeLayerInput input;
                input.positionBegin = Integer(item["position_begin"]);
                if (!item["hidden"].is_array()) throw std::runtime_error("missing hidden array");
                for (const auto &value : item["hidden"].array_items()) {
                    if (!value.is_number()) throw std::runtime_error("invalid hidden value");
                    input.hidden.push_back(static_cast<float>(value.number_value()));
                }
                inputs.push_back(std::move(input));
            }
            auto stage = static_cast<Qwen3LayerStage>(Integer(task["stage"]));
            if (scenario == "dense_api") {
                group.ForwardQwen3Layer(0, inputs[0].hidden.size() / Integer(task["hidden"]),
                    Integer(task["hidden"]), 0, stage, inputs[0].hidden);
            }
            auto outputs = group.ForwardQwenMoeLayer(Integer(task["layer"]), Integer(task["hidden"]), stage, inputs);
            Json::array ranks;
            for (const auto &r : outputs) {
                ranks.push_back(Json::object{{"output", Values(r.output)}, {"router_logits", Values(r.routerLogits)},
                    {"routing_weights", Values(r.routingWeights)}, {"selected_experts", Values(r.selectedExperts)},
                    {"routed_output", Values(r.routedOutput)}, {"shared_output", Values(r.sharedOutput)},
                    {"expert_tokens", Values(r.expertTokens)}, {"allgather_calls", int(r.allGatherCalls)},
                    {"reduce_scatter_calls", int(r.reduceScatterCalls)}, {"indexed_decode", r.indexedDecode}});
            }
            results.push_back(Json::object{{"task", task["name"]}, {"ranks", ranks}});
            report["results"] = results;
        }
        group.UnloadQwen3();
        group.UnloadQwen3();
        group.Ping();
        auto reloaded = group.LoadQwenMoe(model);
        for (size_t rank = 0; rank < infos.size(); ++rank) {
            if (infos[rank].deviceBytes != reloaded[rank].deviceBytes || infos[rank].sourceBytesRead != reloaded[rank].sourceBytesRead) {
                throw std::runtime_error("reload accounting mismatch");
            }
        }
        report["reload_passed"] = true;
        group.UnloadQwen3();
        group.Stop();
        if (!group.LastError().empty() || !group.Workers().empty()) throw std::runtime_error("MoE shutdown failed");
        report["workers_reaped"] = true;
        report["status"] = "PASS";
    } catch (const std::exception &error) {
        report["error"] = error.what();
        report["group_failed"] = group.State() == ProcessGroupState::Failed;
        report["workers_reaped"] = group.Workers().empty();
        const std::string previous = group.LastError();
        try {
            group.Stop();
            if (group.State() != ProcessGroupState::Stopped || !group.Workers().empty() || group.LastError() != previous) {
                throw std::runtime_error("cleanup incomplete: " + group.LastError());
            }
            report["shutdown_complete"] = true;
        } catch (const std::exception &cleanup) {
            report["shutdown_complete"] = false;
            report["cleanup_error"] = cleanup.what();
        }
    }
    std::cout << Json(report).dump() << std::endl;
    return report["status"].string_value() == "PASS" ? 0 : 1;
}
