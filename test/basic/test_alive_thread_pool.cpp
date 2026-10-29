#include "devices/cpu/alivethreadpool.h"
#include <cstdlib>
#include <iostream>
#include <memory>

static void Check(bool ok, const char *message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

struct Work : fastllm::MultiThreadBaseOp {
    std::atomic<int> &completed;
    explicit Work(std::atomic<int> &completed) : completed(completed) {}
    void Run() override {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        completed.fetch_add(1);
    }
};

int main() {
    fastllm::AliveThreadPool pool(8);
    Check(pool.threads.size() == 8, "initial worker count");
    for (int round = 0; round < 10; ++round) {
        for (int count : {8, 3, 12, 1, 9, 2}) {
            pool.ResizeThreads(count);
            Check((int)pool.threads.size() == count, "allocated count after resize");
            Check((int)pool.loops.size() == count, "task storage count after resize");
            Check(pool.curActivateThreadInterval == std::make_pair(0, count), "active count");
            std::atomic<int> completed{0};
            std::vector<std::unique_ptr<Work>> work;
            for (int i = 0; i < count; ++i) {
                work.emplace_back(new Work(completed));
                pool.PushOp(i, work.back().get());
            }
            // No explicit Wait for retired workers: shrinking must drain them.
            const int retained = count / 2;
            pool.ResizeThreads(retained);
            for (int i = 0; i < retained; ++i) pool.Wait(i);
            Check(completed.load() == count, "resize lost or duplicated queued work");
            Check((int)pool.threads.size() == retained, "retired workers still allocated");
        }
        pool.Shutdown();
        pool.Shutdown();
        Check(pool.threads.empty() && pool.loops.empty(), "idempotent shutdown");
        Check(pool.curActivateThreadInterval == std::make_pair(0, 0), "shutdown active range");
    }
    std::atomic<int> completed{0};
    Work pending(completed);
    {
        fastllm::AliveThreadPool temporary(1);
        temporary.PushOp(0, &pending);
    }
    Check(completed.load() == 1, "destructor did not finish queued work");
    std::cout << "PASS: worker resize, pending work, restart and shutdown" << std::endl;
}
