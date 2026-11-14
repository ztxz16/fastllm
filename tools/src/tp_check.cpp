#include "distributed/process_group.h"
#include "json11.hpp"
#include "tp_graph_report.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <exception>
#include <filesystem>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <sys/syscall.h>
#include <sys/wait.h>
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
 * 通过公开控制接口验证等待其他 rank 时的退出唤醒、诊断和正常关闭。
 *
 * 专用 Python worker 通过文件屏障决定退出顺序，避免用随机睡眠猜测执行阶段。
 * 返回耗时包含故障发现和整组回收，不将其标记为纯检测延迟或推理性能。
 * @param options 必须使用 controlOnly 和两个逻辑设备，不初始化 CUDA。
 * @param caseName watch_ 前缀的故障夹具场景。
 * @param report 记录任务截止时间、错误原因、返回耗时及实际回收状态。
 */
void RunWatchCase(ProcessGroupOptions options, const std::string &caseName, json11::Json::object &report) {
    Require(options.controlOnly && options.devices.size() == 2, "watch cases require control-only TP=2");
    const std::string mode = caseName.substr(6);
    const bool startup = mode == "startup_peer_exit", stall = mode == "stall";
    const bool shutdown = mode == "shutdown_order";
    options.startupTimeoutMs = 30000;
    options.operationTimeoutMs = stall ? 1000 : 30000;
    options.shutdownTimeoutMs = 1000;
    // 禁止空闲心跳抢先发 Ping；退出监听始终有效，协议故障由夹具收到命令后触发。
    options.heartbeatIntervalMs = 60000;
    report["operation_timeout_ms"] = startup ? options.startupTimeoutMs : options.operationTimeoutMs;
    report["shape"] = "control-only TP=2; protocol fixture; no tensor output";

    ProcessGroup group;
    auto started = std::chrono::steady_clock::now();
    bool rejected = false;
    bool startupComplete = false;
    try {
        group.Start(options);
        startupComplete = true;
        started = std::chrono::steady_clock::now();
        group.Ping();
    } catch (const std::exception &e) {
        rejected = true;
        report["observed_error"] = e.what();
    }
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    report["failure_return_ms"] = elapsed;
    report["failure_phase"] = startupComplete ? "ping" : "startup";
    Require(startupComplete != startup, "failure occurred outside the intended phase");

    if (shutdown) {
        Require(!rejected, "normal fixture failed before shutdown: " + group.LastError());
        group.Stop();
        Require(group.LastError().empty(), "normal peer exit was treated as a crash: " + group.LastError());
        group.Stop();
        // 验证上下文未持有上一次启动的失效句柄。
        group.Start(options);
        group.Ping();
        group.Stop();
        Require(group.LastError().empty(), "restart shutdown failed: " + group.LastError());
    } else {
        Require(rejected && group.State() == ProcessGroupState::Failed, "watch case did not fail the group");
        const std::string error = group.LastError();
        if (stall) {
            Require(error.find("timeout") != std::string::npos && elapsed >= 900,
                    "live stalled worker did not follow the original deadline: " + error);
        } else {
            int rank = (mode == "current_error_exit" || mode == "reply_then_exit") ? 0 : 1;
            report["fault_rank"] = rank;
            Require(error.rfind("rank " + std::to_string(rank), 0) == 0,
                    "reported waiting rank instead of failed rank: " + error);
            Require(error.find("timeout") == std::string::npos, "worker exit fell through to timeout: " + error);
            if (mode == "buffered_error_exit" || mode == "current_error_exit") {
                Require(error.find("injected worker failure before exit") != std::string::npos,
                        "lost complete worker error: " + error);
            }
            if (mode == "stale_error_exit" || mode == "partial_error_exit") {
                Require(error.find("injected worker failure") == std::string::npos,
                        "accepted stale or incomplete diagnostic: " + error);
            }
        }
        Require(elapsed < 5000, "failure response waited too long (includes cleanup)");
        Require(group.Workers().empty(), "failed group left unreaped workers");
        bool commandRejected = false;
        try { group.Ping(); } catch (const std::exception &) { commandRejected = true; }
        Require(commandRejected, "failed group accepted another command");
        auto stopping = std::chrono::steady_clock::now();
        group.Stop();
        report["stop_return_ms"] = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - stopping).count();
        Require(report["stop_return_ms"].number_value() < 5000, "failed monitor did not wake for Stop");
        group.Stop();
        Require(group.LastError() == error, "cleanup overwrote the original failure");
    }
    Require(group.State() == ProcessGroupState::Stopped && group.Workers().empty(), "watch case leaked workers");
    report["workers_reaped"] = true;
}

/**
 * 统计协调器的打开句柄，检查重复启停是否遗留监控管道或 pidfd 副本。
 * @return 当前 /proc/self/fd 条目数；目录打开失败时抛出异常。
 */
int OpenFdCount() {
    DIR *directory = opendir("/proc/self/fd");
    Require(directory != nullptr, "cannot inspect coordinator file descriptors");
    int count = 0;
    while (auto *entry = readdir(directory)) {
        if (entry->d_name[0] != '.') {
            ++count;
        }
    }
    closedir(directory);
    return count;
}

/**
 * 验证 worker 的退出状态已被协调器回收，而不仅是从内部列表移除。
 * @param workers 故障或关闭前从公开接口取得的 worker 身份。
 */
void RequireReaped(const std::vector<ProcessWorkerInfo> &workers) {
    for (const auto &worker : workers) {
        int status = 0;
        errno = 0;
        auto result = waitpid(worker.pid, &status, WNOHANG);
        Require(result == -1 && errno == ECHILD, "worker is still alive or was not reaped");
    }
}

/**
 * 使用稳定的进程句柄注入故障，避免停止竞态中裸 PID 被复用。
 * @param worker 仍由当前测试进程组管理的 worker。
 * @param signal 要发送的 SIGKILL 或 SIGSTOP。
 */
void SignalWorker(const ProcessWorkerInfo &worker, int signal) {
    int fd = static_cast<int>(syscall(SYS_pidfd_open, worker.pid, 0));
    Require(fd >= 0, "cannot open fault-injection pidfd");
    int result = static_cast<int>(syscall(SYS_pidfd_send_signal, fd, signal, nullptr, 0));
    close(fd);
    Require(result == 0, "cannot inject worker signal");
}

/**
 * 通过正式接口验证控制通信；CUDA 场景同时验证真实 FP32 SUM 输出。
 * @param group 已就绪的进程组。
 * @param options 本次启动配置，决定是否验证 GPU 数值。
 * @param report 保存最近一次求和输出、误差及实际 worker 元数据。
 */
void CheckMonitorWork(ProcessGroup &group, const ProcessGroupOptions &options, json11::Json::object &report) {
    group.Ping();
    json11::Json::array identities;
    for (const auto &worker : group.Workers()) {
        identities.push_back(json11::Json::object{{"rank", worker.rank}, {"pid", worker.pid},
            {"device", worker.device}, {"gpu", worker.deviceName},
            {"cuda_runtime_version", worker.cudaRuntimeVersion}, {"nccl_version", worker.ncclVersion}});
    }
    report["workers"] = identities;
    if (options.controlOnly) {
        return;
    }

    auto values = group.CheckAllReduce();
    double expected = double(options.devices.size()) * (options.devices.size() + 1) / 2;
    double maximum = 0, sum = 0;
    int failed = 0;
    json11::Json::array actual;
    for (float value : values) {
        actual.push_back(double(value));
        double error = std::isfinite(value) ? std::abs(value - expected) : expected + 1;
        maximum = std::max(maximum, error);
        sum += error;
        failed += error != 0;
    }
    report["actual"] = actual;
    report["expected"] = expected;
    report["max_abs_error"] = maximum;
    report["mean_abs_error"] = values.empty() ? 0 : sum / values.size();
    report["failed_elements"] = failed;
    Require(values.size() == options.devices.size() && failed == 0, "monitor lifecycle SUM output mismatch");

    auto states = group.Qwen3Requests();
    Require(states.size() == options.devices.size(), "missing AllReduce path diagnostics");
    for (const auto &state : states) {
        const auto &info = state.allReduce;
        Require(options.customAllReduce ? (info.active && info.customCalls > 0 && info.ncclCalls == 0) :
                                         (!info.enabled && info.ncclCalls > 0 && info.customCalls == 0),
                "monitor lifecycle used unexpected AllReduce path");
    }
    report["custom_calls"] = static_cast<long long>(states.front().allReduce.customCalls);
    report["nccl_calls"] = static_cast<long long>(states.front().allReduce.ncclCalls);
}

/**
 * 验证空闲退出唤醒、心跳及停止/重启生命周期，避免只验证操作中的退出监听。
 *
 * 退出与停止场景使用 60 秒心跳和 5 秒完成上限，旧的定时检查不能通过。
 * 耗时包含调度、整组清理及回收，仅用于功能断言，不代表纯检测延迟或性能结果。
 * @param options 正式 worker 配置；夹具场景仅允许 controlOnly TP=2。
 * @param caseName monitor_ 前缀的场景名。
 * @param report 保存配置、实际诊断、生命周期耗时和数值验证结果。
 */
void RunMonitorCase(ProcessGroupOptions options, const std::string &caseName, json11::Json::object &report) {
    using Clock = std::chrono::steady_clock;
    const std::string mode = caseName.substr(8);
    const bool heartbeat = mode == "heartbeat_timeout" || mode == "heartbeat_healthy" || mode == "stop_heartbeat";
    const bool fixture = mode == "heartbeat_healthy" || mode == "stop_heartbeat";
    Require(mode == "idle_exit_first" || mode == "idle_exit_last" || mode == "heartbeat_timeout" ||
            mode == "stop_restart" || mode == "destructor" || mode == "group_isolation" || fixture,
            "unknown monitor case: " + mode);
    Require(!fixture || (options.controlOnly && options.devices.size() == 2), "monitor fixture requires control-only TP=2");
    options.heartbeatIntervalMs = heartbeat ? 100 : 60000;
    options.operationTimeoutMs = heartbeat ? 1000 : 30000;
    options.shutdownTimeoutMs = 3000;
    report["heartbeat_interval_ms"] = options.heartbeatIntervalMs;
    report["operation_timeout_ms"] = options.operationTimeoutMs;
    report["shape"] = options.controlOnly ? "process lifecycle; no tensor output" : "one FP32 scalar per rank";
    const int baselineFds = OpenFdCount();

    if (mode == "destructor") {
        std::vector<ProcessWorkerInfo> workers;
        auto stopping = Clock::now();
        {
            ProcessGroup scoped;
            scoped.Start(options);
            CheckMonitorWork(scoped, options, report);
            workers = scoped.Workers();
            // 留出后台线程进入空闲等待的调度机会，耗时断言从真正开始析构时计算。
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            stopping = Clock::now();
        }
        report["stop_return_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - stopping).count();
        Require(report["stop_return_ms"].number_value() < 5000, "destructor waited for long heartbeat");
        RequireReaped(workers);
    } else {
        ProcessGroup group;
        group.Start(options);
        if (!fixture) {
            CheckMonitorWork(group, options, report);
            if (!heartbeat) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        auto workers = group.Workers();
        if (mode == "stop_restart") {
            double maximum = 0;
            for (int iteration = 0; iteration < 8; ++iteration) {
                if (iteration != 0) {
                    group.Start(options);
                    workers = group.Workers();
                    // 交替覆盖启动后立即 Stop，以及正常通信后的 Stop。
                    if (iteration % 2 == 0) {
                        CheckMonitorWork(group, options, report);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                }
                auto stopping = Clock::now();
                group.Stop();
                maximum = std::max(maximum, std::chrono::duration<double, std::milli>(Clock::now() - stopping).count());
                Require(maximum < 5000, "Stop waited for long heartbeat");
                Require(group.LastError().empty(), "normal shutdown misclassified as crash: " + group.LastError());
                RequireReaped(workers);
                group.Stop();
                Require(OpenFdCount() == baselineFds, "monitor descriptors leaked across restart");
            }
            report["stop_return_ms"] = maximum;
            report["restarts"] = 7;
        } else if (fixture) {
            const char *directory = std::getenv("FASTLLM_TEST_EXIT_DIR");
            Require(directory && *directory, "missing monitor fixture directory");
            const auto root = std::filesystem::path(directory);
            const auto prefix = mode == "heartbeat_healthy" ? "heartbeat.3." : "heartbeat.1.";
            auto deadline = Clock::now() + std::chrono::seconds(5);
            while (!std::filesystem::exists(root / (std::string(prefix) + "0")) ||
                   !std::filesystem::exists(root / (std::string(prefix) + "1"))) {
                Require(Clock::now() < deadline, "background heartbeat did not reach both workers");
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            if (mode == "heartbeat_healthy") {
                CheckMonitorWork(group, options, report);
            }
            auto stopping = Clock::now();
            group.Stop();
            report["stop_return_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - stopping).count();
            Require(report["stop_return_ms"].number_value() < 5000, "Stop hung during heartbeat");
            Require(mode == "heartbeat_healthy" ? group.LastError().empty() :
                    group.LastError().find("timeout") != std::string::npos, "wrong heartbeat outcome: " + group.LastError());
            report["observed_error"] = group.LastError();
            RequireReaped(workers);
        } else {
            ProcessGroup survivor;
            if (mode == "group_isolation") {
                survivor.Start(options);
                CheckMonitorWork(survivor, options, report);
            }
            const auto &victim = mode == "idle_exit_first" ? workers.front() : workers.back();
            report["fault_rank"] = victim.rank;
            auto started = Clock::now();
            SignalWorker(victim, mode == "heartbeat_timeout" ? SIGSTOP : SIGKILL);
            // 不发送前台命令：故障必须由后台监控独立发现。
            while (group.State() == ProcessGroupState::Ready && Clock::now() - started < std::chrono::seconds(5)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            report["failure_return_ms"] = elapsed;
            report["failure_phase"] = "idle_monitor";
            report["observed_error"] = group.LastError();
            Require(elapsed < 5000 && group.State() == ProcessGroupState::Failed && group.Workers().empty(),
                    "idle monitor did not fail and reap the group before heartbeat deadline");
            const auto error = group.LastError();
            if (mode == "heartbeat_timeout") {
                Require(error.find("timeout") != std::string::npos && elapsed >= 900,
                        "live stalled worker did not follow heartbeat timeout: " + error);
            } else {
                Require(error.rfind("rank " + std::to_string(victim.rank), 0) == 0 && error.find("timeout") == std::string::npos,
                        "idle exit lost fault rank or fell through to timeout: " + error);
            }
            RequireReaped(workers);
            bool rejected = false;
            try { group.Ping(); } catch (const std::exception &) { rejected = true; }
            Require(rejected, "failed group accepted a command");
            auto stopping = Clock::now();
            group.Stop();
            report["stop_return_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - stopping).count();
            Require(report["stop_return_ms"].number_value() < 5000 && group.LastError() == error,
                    "failure cleanup hung or overwrote diagnostic");
            group.Stop();
            group.Start(options);
            CheckMonitorWork(group, options, report);
            workers = group.Workers();
            group.Stop();
            Require(group.LastError().empty(), "restart inherited failed monitor state");
            RequireReaped(workers);
            if (mode == "group_isolation") {
                CheckMonitorWork(survivor, options, report);
                auto survivors = survivor.Workers();
                survivor.Stop();
                Require(survivor.LastError().empty(), "one group's failure affected another group");
                RequireReaped(survivors);
            }
        }
        group.Stop();
        Require(group.State() == ProcessGroupState::Stopped && group.Workers().empty(), "monitor case leaked workers");
    }
    report["workers_reaped"] = true;
    report["fd_count_before"] = baselineFds;
    report["fd_count_after"] = OpenFdCount();
    Require(report["fd_count_after"].int_value() == baselineFds, "monitor handles were not released");
}

/**
 * 通过公开 ProcessGroup 接口验证结果及生命周期，不加载模型。
 * @param options 与正式进程底座共用的启动参数。
 * @param caseName 正常运行、重启或故障注入场景名。
 * @param report 保存实际返回值、误差和实际 worker 元数据，异常时保留部分结果。
 */
void Run(ProcessGroupOptions options, const std::string &caseName, json11::Json::object &report) {
    if (caseName.rfind("monitor_", 0) == 0) {
        RunMonitorCase(options, caseName, report);
        return;
    }
    if (caseName.rfind("watch_", 0) == 0) {
        RunWatchCase(options, caseName, report);
        return;
    }
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
        const int baselineFds = OpenFdCount();
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
        report["fd_count_before"] = baselineFds;
        report["fd_count_after"] = OpenFdCount();
        Require(report["fd_count_after"].int_value() == baselineFds, "startup failure leaked monitor handles");
        return;
    }
    if (caseName == "custom_invalid_world") {
        options.customAllReduce = true;
        options.devices.resize(1);
        bool rejected = false;
        try { group.Start(options); } catch (const std::invalid_argument &) { rejected = true; }
        Require(rejected && group.Workers().empty(), "custom AllReduce accepted TP=1");
        return;
    }
    Require(caseName == "smoke" || caseName == "pid1" || caseName == "restart" || caseName == "crash" ||
            caseName == "stall" || caseName == "hold" || caseName == "allreduce_boundaries" ||
            caseName == "collective_crash", "unknown case: " + caseName);
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
        if (caseName == "collective_crash") {
            Require(!options.controlOnly, "collective_crash needs CUDA");
            Require(kill(workers.back().pid, SIGKILL) == 0, "could not kill peer");
            bool rejected = false;
            try { group.CheckAllReduce(4096, 3); } catch (const std::exception &) { rejected = true; }
            Require(rejected && group.State() == ProcessGroupState::Failed && group.Workers().empty(),
                    "collective failure left peers spinning");
            report["observed_error"] = group.LastError();
        } else if (caseName == "crash" || caseName == "stall") {
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
        } else if (caseName == "allreduce_boundaries") {
            Require(!options.controlOnly, "AllReduce boundaries need CUDA");
            double maximum = 0, sum = 0;
            long long elements = 0, failures = 0;
            uint64_t custom = 0, nccl = 0;
            json11::Json::array checks;
            for (uint32_t salt : {1U, 19U, 2U}) {
                for (uint32_t count : {1U, 31U, 255U, 256U, 4095U, 4096U, 65535U, 65536U, 65537U}) {
                    auto actual = group.CheckAllReduce(count, salt);
                    Require(actual.size() == workers.size() * count, "wrong AllReduce result shape");
                    double localMax = 0, localSum = 0;
                    long long localFailed = 0;
                    for (size_t i = 0; i < actual.size(); ++i) {
                        double expected = workers.size() * (workers.size() + 1) / 2.0 *
                                          (1.0 + ((i % count) % 257) * 0.125 + salt * 0.25);
                        double error = std::isfinite(actual[i]) ? std::abs(actual[i] - expected) : expected + 1;
                        localMax = std::max(localMax, error);
                        localSum += error;
                        localFailed += error != 0;
                    }
                    maximum = std::max(maximum, localMax);
                    sum += localSum;
                    failures += localFailed;
                    elements += actual.size();
                    bool selected = options.customAllReduce && count <= 65536;
                    selected ? ++custom : ++nccl;
                    checks.push_back(json11::Json::object{{"elements", int(count)}, {"salt", int(salt)},
                        {"path", selected ? "cuda_ipc_fp32_tp2" : "nccl"}, {"max_abs_error", localMax},
                        {"mean_abs_error", localSum / actual.size()}, {"failed_elements", localFailed}});
                    report["checks"] = checks;
                    report["max_abs_error"] = maximum;
                    report["mean_abs_error"] = sum / elements;
                    report["failed_elements"] = failures;
                    Require(!localFailed, "AllReduce output mismatch at count=" + std::to_string(count));
                    auto states = group.Qwen3Requests();
                    report["custom_calls"] = static_cast<long long>(states.front().allReduce.customCalls);
                    report["nccl_calls"] = static_cast<long long>(states.front().allReduce.ncclCalls);
                    for (const auto &state : states) {
                        Require(state.allReduce.customCalls == custom && state.allReduce.ncclCalls == nccl,
                                "AllReduce path mismatch at count=" + std::to_string(count));
                    }
                }
            }
            for (auto count : {0U, 131073U}) {
                bool rejected = false;
                try { group.CheckAllReduce(count); } catch (const std::invalid_argument &) { rejected = true; }
                Require(rejected && group.State() == ProcessGroupState::Ready, "invalid shape poisoned group");
            }
            report["shape"] = "FP32[count], 27 varying-message collectives, including 65536 boundary";
            report["graph"] = GraphState(group);
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
        if (caseName != "crash" && caseName != "stall" && caseName != "collective_crash") {
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
            } else if (key == "--custom-allreduce") {
                options.customAllReduce = ParseCustomAllReduce(value);
            } else {
                throw std::runtime_error("unknown argument: " + key);
            }
        }
        report["case"] = caseName;
        report["backend"] = options.controlOnly ? "process/control-only" :
            options.customAllReduce ? "process/custom-ipc+NCCL-large" : "process/NCCL";
        Run(options, caseName, report);
        report["status"] = "PASS";
    } catch (const std::exception &e) {
        report["reason"] = e.what();
    }
    std::cout << json11::Json(report).dump() << std::endl;
    return report.at("status").string_value() == "PASS" ? 0 : 1;
}
