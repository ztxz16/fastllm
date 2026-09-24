#include "devices/cpu/alivethreadpool.h"

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

// Thread-pool tasks are always deleted through MultiThreadBaseOp*. Without a
// virtual destructor the derived destructor never ran, leaking every owning
// member (MultiCuda dispatch parameter maps leaked on each operator).
static_assert(std::has_virtual_destructor<fastllm::MultiThreadBaseOp>::value,
              "MultiThreadBaseOp tasks are deleted through the base pointer");

namespace {
struct CountingOp : fastllm::MultiThreadBaseOp {
    explicit CountingOp(int &live) : live(live), name("owning-member"), params{{"k", 1}} { ++live; }
    ~CountingOp() override { --live; }
    void Run() override {}

    int &live;
    std::string name;
    std::map<std::string, int> params;
};

void Check(bool ok, const char *message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}
}

int main() {
    try {
        int live = 0;
        for (int i = 0; i < 1000; ++i) {
            fastllm::MultiThreadBaseOp *op = new CountingOp(live);
            op->Run();
            delete op;
        }
        Check(live == 0, "deleting a task through the base pointer skipped its destructor");

        // MultiThreadMultiOps owns single-object children and must release them.
        for (int i = 0; i < 1000; ++i) {
            auto *group = new fastllm::MultiThreadMultiOps();
            group->ops.push_back(new CountingOp(live));
            group->ops.push_back(new CountingOp(live));
            fastllm::MultiThreadBaseOp *base = group;
            base->Run();
            delete base;
        }
        Check(live == 0, "MultiThreadMultiOps did not release its child tasks");
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "multithread op lifetime: PASS" << std::endl;
    return 0;
}
