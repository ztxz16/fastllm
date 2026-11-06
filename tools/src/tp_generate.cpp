#include "distributed/qwen3_generation.h"
#include "json11.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace fastllm::distributed;
using json11::Json;
namespace fs = std::filesystem;
namespace {
/** 将 JSON 字段读取为 uint32；拒绝隐式类型转换和截断。 */
uint32_t Integer(const Json &value) {
    double x = value.number_value();
    if (!value.is_number() || !std::isfinite(x) || x < 0 || x > 4294967295.0 || x != std::floor(x)) {
        throw std::runtime_error("generation field must be uint32");
    }
    return uint32_t(x);
}

/** 读取有限浮点采样参数；范围由公开采样接口统一验证。 */
double Number(const Json &value) {
    if (!value.is_number() || !std::isfinite(value.number_value())) {
        throw std::runtime_error("generation field must be finite number");
    }
    return value.number_value();
}

/** 读取 token ID 数组；只负责 JSON 类型，词表范围由模型接口检查。 */
std::vector<uint32_t> Ids(const Json &value) {
    if (!value.is_array()) {
        throw std::runtime_error("generation IDs must be array");
    }
    std::vector<uint32_t> ids;
    for (const auto &item : value.array_items()) {
        ids.push_back(Integer(item));
    }
    return ids;
}

/** 读取真实 KV 状态，生成完成和卸载后都要求完全释放。 */
Json::array EmptyCache(ProcessGroup &group) {
    Json::array states;
    for (const auto &info : group.Qwen3Requests()) {
        if (info.active || info.tokens || info.capacity || info.positionBegin || info.deviceBytes) {
            throw std::runtime_error("generation leaked request KV");
        }
        states.push_back(Json::object{{"active", info.active}, {"tokens", int(info.tokens)},
            {"capacity", int(info.capacity)}, {"device_bytes", double(info.deviceBytes)}});
    }
    return states;
}
}

/**
 * 从 token ID 请求文件执行一次真实生成；文本和聊天模板由 Python 前端负责。
 * @param argc/argv --worker、--model、--request 必填，--devices 默认 0,1；
 *                 --trace-logits 1 开启有界诊断，默认不保留词表历史。
 * @return 0 表示生成、KV 释放和 worker 回收均成功；否则 JSON 含错误及部分输出。
 */
int main(int argc, char **argv) {
    ProcessGroup group;
    Json::object report{{"status", "FAIL"}, {"backend", "qwen3_process_generate_cublas_nccl"}};
    Json::array steps;
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            const std::string key = argv[i];
            if (i + 1 == argc || (key != "--worker" && key != "--model" && key != "--request" &&
                key != "--devices" && key != "--trace-logits") || !args.emplace(key, argv[i+1]).second) {
                throw std::runtime_error("invalid/duplicate generation argument");
            }
        }
        bool trace = false;
        if (args.count("--trace-logits")) {
            if (args.at("--trace-logits") != "1" && args.at("--trace-logits") != "0") {
                throw std::runtime_error("trace-logits must be 0 or 1");
            }
            trace = args.at("--trace-logits") == "1";
        }
        auto path = fs::path(args.at("--request"));
        if (fs::file_size(path) > 1024 * 1024) {
            throw std::runtime_error("generation request exceeds 1 MiB");
        }
        std::ifstream file(path);
        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()), error;
        auto request = Json::parse(text, error);
        if (!file || !error.empty() || !request.is_object()) {
            throw std::runtime_error("invalid generation JSON: " + error);
        }
        const std::set<std::string> fields{"token_ids", "max_new_tokens", "temperature", "top_k", "top_p", "seed", "eos_token_ids"};
        for (const auto &item : request.object_items()) {
            if (!fields.count(item.first)) {
                throw std::runtime_error("unknown generation field: " + item.first);
            }
        }
        auto prompt = Ids(request["token_ids"]);
        Qwen3GenerationOptions sampling;
        const auto &object = request.object_items();
        if (object.count("max_new_tokens")) sampling.maxNewTokens = Integer(request["max_new_tokens"]);
        if (object.count("temperature")) sampling.temperature = Number(request["temperature"]);
        if (object.count("top_k")) sampling.topK = Integer(request["top_k"]);
        if (object.count("top_p")) sampling.topP = Number(request["top_p"]);
        if (object.count("seed")) sampling.seed = Integer(request["seed"]);
        if (object.count("eos_token_ids")) sampling.eosTokenIds = Ids(request["eos_token_ids"]);
        report["request"] = request;
        report["trace_logits"] = trace;

        ProcessGroupOptions options;
        options.workerPath = fs::absolute(args.at("--worker")).string();
        options.operationTimeoutMs = options.shutdownTimeoutMs = 30000;
        auto devices = args.count("--devices") ? args.at("--devices") : "0,1";
        if (devices.empty() || devices.back() == ',') throw std::runtime_error("invalid devices");
        std::istringstream stream(devices);
        for (std::string item; std::getline(stream, item, ',');) {
            if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos) {
                throw std::runtime_error("invalid device index");
            }
            options.devices.push_back(std::stoi(item));
        }
        group.Start(options);
        Json::array workers;
        for (const auto &worker : group.Workers()) {
            workers.push_back(Json::object{{"rank", worker.rank}, {"device", worker.device}, {"pid", worker.pid},
                {"gpu", worker.deviceName}, {"cuda_runtime_version", worker.cudaRuntimeVersion}, {"nccl_version", worker.ncclVersion}});
        }
        report["workers"] = workers;
        group.LoadQwen3(fs::absolute(args.at("--model")).string());
        uint64_t recorded = 0;
        auto result = GenerateQwen3(group, prompt, sampling, [&](uint32_t token, const std::vector<float> &logits) {
            // JSON 的离散 ID 必须走整数序列化；double 会写成小数并被 Python 读为 float。
            Json::object step{{"token_id", static_cast<long long>(token)}};
            if (trace) {
                if (logits.size() > 8388608ULL - recorded) {
                    throw std::runtime_error("generation trace exceeds 8M logit elements");
                }
                Json::array values;
                values.reserve(logits.size());
                for (auto value : logits) values.push_back(value);
                step["logits"] = std::move(values);
                recorded += logits.size();
            }
            steps.push_back(std::move(step));
        });
        Json::array ids;
        for (auto token : result.tokenIds) ids.push_back(static_cast<long long>(token));
        report["token_ids"] = std::move(ids);
        report["finish_reason"] = result.finishReason;
        report["cache_after_generate"] = EmptyCache(group);
        group.UnloadQwen3();
        report["cache_after_unload"] = EmptyCache(group);
        group.Stop();
        if (!group.LastError().empty() || !group.Workers().empty()) {
            throw std::runtime_error("generation shutdown failed: " + group.LastError());
        }
        report["shutdown_complete"] = true;
        report["status"] = "PASS";
    } catch (const std::exception &error) {
        report["error"] = error.what();
        report["group_failed"] = group.State() == ProcessGroupState::Failed;
        auto previous = group.LastError();
        try {
            bool cacheClean = true;
            if (group.State() == ProcessGroupState::Ready) {
                try {
                    report["cache_after_error"] = EmptyCache(group);
                } catch (const std::exception &cleanup) {
                    cacheClean = false;
                    report["request_cleanup_error"] = cleanup.what();
                }
            }
            group.Stop();
            if (!cacheClean || group.State() != ProcessGroupState::Stopped || !group.Workers().empty() || group.LastError() != previous) {
                throw std::runtime_error("generation cleanup failed: " + group.LastError());
            }
            report["shutdown_complete"] = true;
        } catch (const std::exception &cleanup) {
            report["shutdown_complete"] = false;
            report["cleanup_error"] = cleanup.what();
        }
    }
    report["steps"] = std::move(steps);
    std::cout << Json(report).dump() << std::endl;
    return report["status"].string_value() == "PASS" ? 0 : 1;
}
