#include "utils/streaming_load_budget.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using fastllm::ReadFileRange;
using fastllm::StreamingLoadBudget;

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        const std::filesystem::path file =
            std::filesystem::path("p2-range-fixture-20261001.bin");
        std::vector<uint8_t> source(8192);
        for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>(i * 17 + 3);
        { std::ofstream out(file, std::ios::binary | std::ios::trunc); out.write(
                reinterpret_cast<const char *>(source.data()), source.size()); }
        std::array<uint8_t, 257> part{};
        std::string error;
        Check(ReadFileRange(file.string(), 128, 4096, 0, part.data(), part.size(), part.size(), &error),
              "first range read failed");
        Check(std::equal(part.begin(), part.end(), source.begin() + 128), "first range changed bytes");
        Check(ReadFileRange(file.string(), 128, 4096, 1900, part.data(), part.size(), part.size(), &error),
              "middle range read failed");
        Check(std::equal(part.begin(), part.end(), source.begin() + 2028), "middle range changed bytes");
        Check(ReadFileRange(file.string(), 128, 4096, 3839, part.data(), part.size(), part.size(), &error),
              "last range read failed");
        Check(std::equal(part.begin(), part.end(), source.begin() + 3967), "last range changed bytes");
        Check(!ReadFileRange(file.string(), 128, 4096, 4000, part.data(), part.size(), part.size(), &error),
              "out of range read was accepted");
        Check(!ReadFileRange(file.string(), 128, 4096, 0, part.data(), part.size(), 16, &error),
              "destination overflow was accepted");
        Check(!ReadFileRange(file.string(), 128, 4096, 0, part.data(), 4097, part.size(), &error),
              "count overflow was accepted");
        Check(!ReadFileRange(file.string() + ".missing", 0, 1, 0, part.data(), 1, 1, &error),
              "missing file was accepted");

        StreamingLoadBudget budget(4096);
        auto first = budget.Acquire(3000);
        Check(budget.InFlight() == 3000 && budget.Peak() == 3000, "budget acquire accounting failed");
        StreamingLoadBudget::Lease rejected;
        Check(!budget.TryAcquire(1200, rejected), "budget allowed an over-commit");
        first.Release();
        Check(budget.InFlight() == 0, "budget release accounting failed");
        Check(budget.TryAcquire(4096, rejected), "budget rejected exact capacity");
        Check(!budget.TryAcquire(1, first), "budget allowed capacity overflow");
        rejected.Release();
        bool oversized = false;
        try { budget.Acquire(4097); } catch (const std::invalid_argument &) { oversized = true; }
        Check(oversized, "oversized request did not fail closed");

        StreamingLoadBudget concurrent(4096);
        std::atomic<bool> acquired{false};
        auto held = concurrent.Acquire(3000);
        std::thread waiter([&] {
            auto lease = concurrent.Acquire(2000);
            acquired = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        Check(!acquired.load(), "budget waiter passed before release");
        held.Release();
        waiter.join();
        Check(concurrent.Peak() == 3000, "budget peak exceeded the hard limit");
        std::cout << "PASS bounded Safetensors range reads and 4GiB-style budget gate\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
