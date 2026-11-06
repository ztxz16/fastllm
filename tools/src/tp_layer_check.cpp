#include "distributed/process_group.h"
#include "json11.hpp"

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
/** 读取有界整数，拒绝负数、小数和超出 uint32 的任务字段。 */
uint32_t Integer(const Json &value, const std::string &name) {
    double number = value.number_value();
    if (!value.is_number() || !std::isfinite(number) || number < 0 || number > 4294967295.0 ||
        std::floor(number) != number) {
        throw std::runtime_error("invalid task integer: " + name);
    }
    return static_cast<uint32_t>(number);
}
}

/**
 * 经公开进程接口运行单层任务，保存全部 rank 的实际输出，由 Python 独立验证数值。
 * @param argc/argv --worker、--model、--tasks 必填；--devices 默认 0,1；--case 可选 unloaded。
 * @return 0 表示全部任务执行完成，1 表示失败；失败输出整组状态和清理证据。
 */
int main(int argc, char **argv) {
    Json::object report{{"status", "FAIL"}, {"backend", "qwen3_process_layer_cublas_nccl"}};
    ProcessGroup group;
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            std::string key = argv[i];
            if (i + 1 == argc || (key != "--worker" && key != "--model" && key != "--tasks" &&
                key != "--devices" && key != "--case") || !args.emplace(key, argv[i + 1]).second) {
                throw std::runtime_error("invalid/duplicate layer checker argument");
            }
        }
        auto taskPath = fs::path(args.at("--tasks"));
        if (fs::file_size(taskPath) > 16 * 1024 * 1024) {
            throw std::runtime_error("layer tasks file exceeds 16 MiB");
        }
        std::ifstream file(taskPath);
        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        std::string error;
        auto tasks = Json::parse(text, error);
        if (!file || !error.empty() || !tasks.is_array() || tasks.array_items().empty() || tasks.array_items().size() > 32) {
            throw std::runtime_error("invalid layer tasks JSON: " + error);
        }
        ProcessGroupOptions options;
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
        if (scenario != "forward" && scenario != "unloaded") {
            throw std::runtime_error("unknown layer checker case");
        }
        group.Start(options);
        Json::array workers;
        for (const auto &worker : group.Workers()) {
            workers.push_back(Json::object{{"rank", worker.rank}, {"device", worker.device}, {"pid", worker.pid},
                {"gpu", worker.deviceName}, {"cuda_runtime_version", worker.cudaRuntimeVersion}, {"nccl_version", worker.ncclVersion}});
        }
        report["workers"] = workers;
        group.LoadQwen3(fs::absolute(args.at("--model")).string());
        if (scenario == "unloaded") {
            group.UnloadQwen3();
        }
        Json::array results;
        for (const auto &task : tasks.array_items()) {
            if (!task["input"].is_array()) {
                throw std::runtime_error("layer input must be an array");
            }
            std::vector<float> input;
            for (const auto &value : task["input"].array_items()) {
                if (!value.is_number() || !std::isfinite(value.number_value()) || std::fabs(value.number_value()) > 65504) {
                    throw std::runtime_error("layer task input outside finite FP16 range");
                }
                input.push_back(static_cast<float>(value.number_value()));
            }
            auto outputs = group.ForwardQwen3Layer(Integer(task["layer"], "layer"), Integer(task["tokens"], "tokens"),
                Integer(task["hidden"], "hidden"), Integer(task["position_begin"], "position_begin"),
                static_cast<Qwen3LayerStage>(Integer(task["stage"], "stage")), input);
            Json::array ranks;
            for (const auto &output : outputs) {
                Json::array values;
                for (float value : output) {
                    values.push_back(value);
                }
                ranks.push_back(values);
            }
            results.push_back(Json::object{{"task", task["name"]}, {"outputs", ranks}});
            report["results"] = results;
        }
        group.UnloadQwen3();
        group.Ping();
        group.Stop();
        if (!group.LastError().empty() || !group.Workers().empty()) {
            throw std::runtime_error("layer checker shutdown failed: " + group.LastError());
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
                throw std::runtime_error("layer checker cleanup failed: " + group.LastError());
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
