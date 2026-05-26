#ifndef FASTLLM_PERSISTENT_WORKER_GROUP_H
#define FASTLLM_PERSISTENT_WORKER_GROUP_H

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace fastllm {
    class PersistentWorkerGroup {
    public:
        PersistentWorkerGroup() = default;

        ~PersistentWorkerGroup() {
            Stop();
        }

        PersistentWorkerGroup(const PersistentWorkerGroup&) = delete;
        PersistentWorkerGroup &operator = (const PersistentWorkerGroup&) = delete;

        bool HasWorkers() {
            std::lock_guard<std::mutex> lock(mutex);
            return !workers.empty();
        }

        void Stop() {
            std::lock_guard<std::mutex> submitGuard(submitMutex);
            StopLocked();
        }

        void Run(const std::vector<int> &workerKeys,
                 const std::function<void(int)> &task,
                 std::vector<std::exception_ptr> &errors) {
            if (workerKeys.empty()) {
                return;
            }

            std::lock_guard<std::mutex> submitGuard(submitMutex);
            EnsureLocked(workerKeys);
            {
                std::lock_guard<std::mutex> lock(mutex);
                currentTask = task;
                currentErrors = &errors;
                activeCount = (int)workerKeys.size();
                finishedCount = 0;
                taskId++;
            }
            cv.notify_all();
            {
                std::unique_lock<std::mutex> lock(mutex);
                doneCv.wait(lock, [&]() {
                    return finishedCount >= activeCount;
                });
                currentTask = nullptr;
                currentErrors = nullptr;
                activeCount = 0;
                finishedCount = 0;
            }
        }

    private:
        void EnsureLocked(const std::vector<int> &workerKeys) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!stop && keys == workerKeys && workers.size() == workerKeys.size()) {
                    return;
                }
            }

            StopLocked();
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = false;
                keys = workerKeys;
                currentTask = nullptr;
                currentErrors = nullptr;
                taskId = 0;
                activeCount = 0;
                finishedCount = 0;
            }

            std::vector<std::thread> nextWorkers;
            nextWorkers.reserve(workerKeys.size());
            try {
                for (int i = 0; i < (int)workerKeys.size(); i++) {
                    nextWorkers.emplace_back([this, i]() { WorkerLoop(i); });
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    stop = true;
                }
                cv.notify_all();
                for (auto &worker : nextWorkers) {
                    if (worker.joinable()) {
                        worker.join();
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    stop = false;
                    keys.clear();
                }
                throw;
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                workers.swap(nextWorkers);
            }
        }

        void StopLocked() {
            std::vector<std::thread> stoppingWorkers;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (workers.empty()) {
                    stop = false;
                    keys.clear();
                    currentTask = nullptr;
                    currentErrors = nullptr;
                    taskId = 0;
                    activeCount = 0;
                    finishedCount = 0;
                    return;
                }
                stop = true;
                keys.clear();
                stoppingWorkers.swap(workers);
            }
            cv.notify_all();
            for (auto &worker : stoppingWorkers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = false;
                currentTask = nullptr;
                currentErrors = nullptr;
                taskId = 0;
                activeCount = 0;
                finishedCount = 0;
            }
        }

        void WorkerLoop(int rank) {
            uint64_t lastTaskId = 0;
            while (true) {
                std::function<void(int)> task;
                std::vector<std::exception_ptr> *errors = nullptr;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&]() {
                        return stop || taskId != lastTaskId;
                    });
                    if (stop) {
                        return;
                    }
                    lastTaskId = taskId;
                    task = currentTask;
                    errors = currentErrors;
                }

                std::exception_ptr error;
                try {
                    task(rank);
                } catch (...) {
                    error = std::current_exception();
                }

                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (errors != nullptr && rank >= 0 && rank < (int)errors->size()) {
                        (*errors)[rank] = error;
                    }
                    finishedCount++;
                    if (finishedCount >= activeCount) {
                        doneCv.notify_one();
                    }
                }
            }
        }

        std::mutex submitMutex;
        std::mutex mutex;
        std::condition_variable cv;
        std::condition_variable doneCv;
        std::vector<std::thread> workers;
        std::vector<int> keys;
        std::function<void(int)> currentTask;
        std::vector<std::exception_ptr> *currentErrors = nullptr;
        uint64_t taskId = 0;
        int activeCount = 0;
        int finishedCount = 0;
        bool stop = false;
    };
}

#endif
