#include "distributed/process_group.h"
#include "json11.hpp"
#include "tp_graph_report.h"

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
constexpr uint64_t kMaxLogitElements = 8ULL * 1024 * 1024;

/** 读取有界整数，拒绝负数、小数和超出 uint32 的任务字段。 */
uint32_t Integer(const Json &value, const std::string &name) {
    double number = value.number_value();
    if (!value.is_number() || !std::isfinite(number) || number < 0 || number > 4294967295.0 ||
        std::floor(number) != number) {
        throw std::runtime_error("invalid task integer: " + name);
    }
    return static_cast<uint32_t>(number);
}

/** 从公开入口读取各 rank KV 状态，保存真实分配字节和有效长度。 */
Json::array CacheState(ProcessGroup &group) {
    Json::array states;
    for (const auto &info : group.Qwen3Requests()) {
        states.push_back(Json::object{{"active", info.active}, {"tokens", double(info.tokens)},
            {"capacity", double(info.capacity)}, {"position_begin", double(info.positionBegin)},
            {"device_bytes", double(info.deviceBytes)}});
    }
    return states;
}
}

/**
 * 经公开进程接口执行请求生命周期任务，保存逐步 logits 和 KV 状态，由 Python 验证数值。
 * @param argc/argv --worker、--model、--tasks 必填；--devices 默认 0,1；--case 仅 forward。
 *                 load 可用 --reload-model；--max-logit-elements 可降低报告的 8M 元素上限。
 * @return 0 表示全部任务执行完成，1 表示失败；失败输出整组状态和清理证据。
 */
int main(int argc, char **argv) {
    Json::object report{{"status", "FAIL"}, {"backend", "qwen3_process_decode_cublas_nccl"}};
    ProcessGroup group;
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            std::string key = argv[i];
            if (i + 1 == argc || (key != "--worker" && key != "--model" && key != "--tasks" &&
                key != "--devices" && key != "--case" && key != "--reload-model" &&
                key != "--max-logit-elements" && key != "--cuda-graph" && key != "--custom-allreduce" && key != "--prefix-cache") || !args.emplace(key, argv[i + 1]).second) {
                throw std::runtime_error("invalid/duplicate decode checker argument");
            }
        }
        auto taskPath = fs::path(args.at("--tasks"));
        if (fs::file_size(taskPath) > 16 * 1024 * 1024) {
            throw std::runtime_error("decode tasks file exceeds 16 MiB");
        }
        std::ifstream file(taskPath);
        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::string error;
        auto tasks = Json::parse(text, error);
        if (!file || !error.empty() || !tasks.is_array() || tasks.array_items().empty() || tasks.array_items().size() > 8192) {
            throw std::runtime_error("invalid decode tasks JSON: " + error);
        }
        ProcessGroupOptions options;
        options.cudaGraph = args.count("--cuda-graph") && ParseCudaGraph(args.at("--cuda-graph"));
        options.customAllReduce = args.count("--custom-allreduce") && ParseCustomAllReduce(args.at("--custom-allreduce"));
        options.prefixCache = args.count("--prefix-cache") && ParsePrefixCache(args.at("--prefix-cache"));
        report["execution"] = options.cudaGraph ? "decode_cuda_graph" : "eager";
        options.workerPath = fs::absolute(args.at("--worker")).string();
        options.operationTimeoutMs = options.shutdownTimeoutMs = 30000;
        std::string devices = args.count("--devices") ? args.at("--devices") : "0,1";
        if (devices.empty() || devices.back() == ',') {
            throw std::runtime_error("invalid devices");
        }
        std::istringstream stream(devices);
        for (std::string item; std::getline(stream, item, ',');) {
            if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos) {
                throw std::runtime_error("invalid device index");
            }
            options.devices.push_back(std::stoi(item));
        }
        std::string scenario = args.count("--case") ? args.at("--case") : "forward";
        if (scenario != "forward") {
            throw std::runtime_error("unknown decode checker case");
        }
        uint64_t logitLimit = kMaxLogitElements, recordedLogits = 0;
        if (args.count("--max-logit-elements")) {
            const auto &value = args.at("--max-logit-elements");
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
                throw std::runtime_error("invalid decode report logit limit");
            }
            logitLimit = std::stoull(value);
            if (!logitLimit || logitLimit > kMaxLogitElements) {
                throw std::runtime_error("invalid decode report logit limit");
            }
        }
        report["logit_limit"] = double(logitLimit);
        report["recorded_logits"] = 0;

        group.Start(options);
        Json::array workers;
        for (const auto &worker : group.Workers()) {
            workers.push_back(Json::object{{"rank", worker.rank}, {"device", worker.device}, {"pid", worker.pid},
                {"gpu", worker.deviceName}, {"cuda_runtime_version", worker.cudaRuntimeVersion}, {"nccl_version", worker.ncclVersion}});
        }
        report["workers"] = workers;
        group.LoadQwen3(fs::absolute(args.at("--model")).string());
        Json::array results;
        for (const auto &task : tasks.array_items()) {
            report["current_task"] = task["name"];
            if (!task["name"].is_string() || task["name"].string_value().empty()) {
                throw std::runtime_error("decode task requires a name");
            }
            const auto op = task["op"].string_value();
            std::vector<float> logits;
            if (op == "begin" || op == "prefill") {
                if (!task["token_ids"].is_array()) {
                    throw std::runtime_error("decode token_ids must be an array");
                }
                std::vector<uint32_t> ids;
                for (const auto &value : task["token_ids"].array_items()) {
                    ids.push_back(Integer(value, "token ID"));
                }
                auto position = Integer(task["position_begin"], "position_begin");
                logits = op == "begin" ? group.BeginQwen3Request(ids, Integer(task["capacity"], "capacity"), position) :
                                        group.PrefillQwen3(ids, position);
            } else if (op == "decode") {
                logits = group.DecodeQwen3(Integer(task["token_id"], "token ID"));
            } else if (op == "end") {
                group.EndQwen3Request();
            } else if (op == "clear_prefix") {
                group.ClearQwen3PrefixCache();
            } else if (op == "unload") {
                group.UnloadQwen3();
            } else if (op == "load") {
                group.LoadQwen3(fs::absolute(args.at(task["reload"].bool_value() ? "--reload-model" : "--model")).string());
            } else if (op != "query") {
                throw std::runtime_error("unknown decode task operation");
            }
            Json::object result{{"task", task["name"]}, {"op", op}, {"cache", CacheState(group)}, {"graph", GraphState(group)}};
            if (!logits.empty()) {
                // Json 的每个数字都有对象开销；仅限制任务数不足以约束长序列报告的内存。
                if (logits.size() > logitLimit - recordedLogits) {
                    throw std::runtime_error("decode report logit limit exceeded");
                }
                Json::array values;
                values.reserve(logits.size());
                for (float value : logits) {
                    values.push_back(value);
                }
                result["logits"] = values;
                result["argmax"] = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
                recordedLogits += logits.size();
                report["recorded_logits"] = double(recordedLogits);
            }
            results.push_back(result);
            report["results"] = results;
        }
        group.UnloadQwen3();
        report["after_unload"] = CacheState(group);
        report["graph_after_unload"] = GraphState(group);
        group.Ping();
        group.Stop();
        if (!group.LastError().empty() || !group.Workers().empty()) {
            throw std::runtime_error("decode checker shutdown failed: " + group.LastError());
        }
        report["status"] = "PASS";
    } catch (const std::exception &error) {
        report["error"] = error.what();
        report["group_failed"] = group.State() == ProcessGroupState::Failed;
        report["workers_reaped"] = group.Workers().empty();
        const std::string previousError = group.LastError();
        try {
            group.Stop();
            // Stop 在强制回收成功时可能只写 LastError 而不抛异常，不能据此判定正常退出。
            if (group.State() != ProcessGroupState::Stopped || !group.Workers().empty() ||
                group.LastError() != previousError) {
                throw std::runtime_error("decode checker cleanup failed: " + group.LastError());
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
