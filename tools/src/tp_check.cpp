#include "distributed/process_group.h"
#include "json11.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace fastllm::distributed;

namespace {
void Require(bool condition, const std::string &reason) {
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

int ParseInt(const std::string &text) {
    Require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
            "expected a nonnegative integer: " + text);
    return std::stoi(text);
}

/**
 * 通过公开 ProcessGroup 接口验证结果及生命周期，不加载模型。
 * @param options 与正式进程底座共用的启动参数。
 * @param caseName 正常运行、重启或故障注入场景名。
 * @param report 保存实际返回值、误差和实际 worker 元数据，异常时保留部分结果。
 */
void Run(ProcessGroupOptions options, const std::string &caseName, json11::Json::object &report) {
    ProcessGroup group;
    if (caseName == "invalid_devices") {
        options.devices = {0, 0};
        bool rejected = false;
        try {
            group.Start(options);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        Require(rejected && group.State() == ProcessGroupState::Stopped && group.Workers().empty(),
                "duplicate devices were not rejected before spawn");
        return;
    }
    if (caseName == "missing_worker" || caseName == "startup_exit" || caseName == "startup_timeout" ||
        caseName == "protocol_version" || caseName == "protocol_session") {
        if (caseName == "missing_worker") {
            options.workerPath += ".does-not-exist";
        } else if (caseName == "startup_exit") {
            options.workerPath = "/bin/false";
        }
        bool rejected = false;
        try {
            group.Start(options);
        } catch (const std::exception &e) {
            rejected = true;
            report["observed_error"] = e.what();
        }
        Require(rejected && group.State() == ProcessGroupState::Failed && group.Workers().empty(),
                "startup failure did not clean up the group");
        if (caseName == "startup_timeout") {
            Require(group.LastError().find("timeout") != std::string::npos,
                    "startup_timeout did not reach the deadline path");
        }
        if (caseName == "protocol_version" || caseName == "protocol_session") {
            std::string expected = caseName == "protocol_version" ? "magic/version" : "session/sequence";
            Require(group.LastError().find(expected) != std::string::npos,
                    "wrong protocol rejection reason: " + group.LastError());
        }
        group.Stop();
        return;
    }
    Require(caseName == "smoke" || caseName == "pid1" || caseName == "restart" || caseName == "crash" ||
            caseName == "stall" || caseName == "hold", "unknown case: " + caseName);
    if (caseName == "pid1") {
        Require(getpid() == 1, "pid1 case must run as the PID namespace init process");
    }

    int repeats = caseName == "restart" ? 3 : 1;
    for (int iteration = 0; iteration < repeats; iteration++) {
        // 启动线程退出后 worker 必须仍存活，避免误用基于线程死亡的 PDEATHSIG。
        std::exception_ptr startError;
        std::thread starter([&]() {
            try { group.Start(options); } catch (...) { startError = std::current_exception(); }
        });
        starter.join();
        if (startError) {
            std::rethrow_exception(startError);
        }
        auto workers = group.Workers();
        Require(workers.size() == options.devices.size(), "wrong worker count");
        json11::Json::array workerReport;
        for (const auto &worker : workers) {
            workerReport.push_back(json11::Json::object{
                {"rank", worker.rank}, {"device", worker.device}, {"pid", worker.pid},
                {"gpu", worker.deviceName}, {"cuda_runtime_version", worker.cudaRuntimeVersion},
                {"nccl_version", worker.ncclVersion}});
        }
        report["workers"] = workerReport;
        group.Ping();

        if (caseName == "hold") {
            std::cout << json11::Json(report).dump() << std::endl;
            while (group.State() == ProcessGroupState::Ready) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            throw std::runtime_error("hold group failed: " + group.LastError());
        }
        if (caseName == "crash" || caseName == "stall") {
            Require(kill(workers.back().pid, caseName == "crash" ? SIGKILL : SIGSTOP) == 0,
                    "could not inject worker failure");
            auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(
                options.operationTimeoutMs + options.heartbeatIntervalMs + options.shutdownTimeoutMs + 2000);
            while (group.State() == ProcessGroupState::Ready && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            Require(group.State() == ProcessGroupState::Failed && group.Workers().empty(),
                    "monitor did not fail and reap the entire group");
            Require(!group.LastError().empty(), "failure has no diagnostic");
            report["observed_error"] = group.LastError();
            bool rejected = false;
            try { group.Ping(); } catch (const std::exception &) { rejected = true; }
            Require(rejected, "failed group accepted a command");
        } else if (!options.controlOnly) {
            auto values = group.CheckAllReduce();
            double expected = double(workers.size()) * (workers.size() + 1) / 2;
            double maximum = 0, sum = 0;
            int failed = 0;
            json11::Json::array actual;
            for (float value : values) {
                actual.push_back(double(value));
                double error = std::isfinite(value) ? std::abs(value - expected) : expected + 1;
                maximum = std::max(maximum, error);
                sum += error;
                failed += !std::isfinite(value) || error != 0;
            }
            report["actual"] = actual;
            report["expected"] = expected;
            report["max_abs_error"] = maximum;
            report["mean_abs_error"] = sum / values.size();
            report["failed_elements"] = failed;
            Require(values.size() == workers.size() && failed == 0, "NCCL SUM output mismatch");
        }
        group.Stop();
        Require(group.State() == ProcessGroupState::Stopped && group.Workers().empty(), "Stop left workers");
        if (caseName != "crash" && caseName != "stall") {
            Require(group.LastError().empty(), "graceful shutdown failed: " + group.LastError());
        }
        group.Stop();
    }
}
} // namespace

int main(int argc, char **argv) {
    ProcessGroupOptions options;
    options.heartbeatIntervalMs = 100;
    std::string caseName = "smoke";
    json11::Json::object report{{"status", "FAIL"}, {"coordinator_pid", int(getpid())}, {"max_abs_error", nullptr},
        {"mean_abs_error", nullptr}, {"failed_elements", nullptr}, {"shape", "one FP32 scalar per rank"}};
    try {
        for (int i = 1; i < argc; i++) {
            std::string key = argv[i];
            if (key == "--control-only") {
                options.controlOnly = true;
                continue;
            }
            Require(i + 1 < argc, "missing argument for " + key);
            std::string value = argv[++i];
            if (key == "--worker") {
                options.workerPath = value;
            } else if (key == "--devices") {
                Require(!value.empty() && value.back() != ',', "invalid device list");
                std::istringstream input(value);
                std::string item;
                while (std::getline(input, item, ',')) {
                    options.devices.push_back(ParseInt(item));
                }
            } else if (key == "--case") {
                caseName = value;
            } else if (key == "--startup-timeout-ms") {
                options.startupTimeoutMs = ParseInt(value);
            } else if (key == "--operation-timeout-ms") {
                options.operationTimeoutMs = ParseInt(value);
            } else {
                throw std::runtime_error("unknown argument: " + key);
            }
        }
        report["case"] = caseName;
        report["backend"] = options.controlOnly ? "process/control-only" : "process/NCCL";
        Run(options, caseName, report);
        report["status"] = "PASS";
    } catch (const std::exception &e) {
        report["reason"] = e.what();
    }
    std::cout << json11::Json(report).dump() << std::endl;
    return report.at("status").string_value() == "PASS" ? 0 : 1;
}
