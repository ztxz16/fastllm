// A-5：D2 流水"一张单只搬一次"的离线判据（纯 CPU，无 GPU）。
// 判据：copy_ 抛异常后，该 ticket 必须**确定性终结** —— 不得被再次投递、槽位必须归还、
// Pending() 归零、Failed() 为真。
// 旧实现（HEAD disk_prefix_write_coordinator.cpp）在 copy_ 抛异常时会留下 pending_ 条目与槽位，
// 下一次 PumpCopy 再次调用 copy_（真机表现为 duplicate_or_unadmitted_copy），因此本测试在旧实现上必失败。
#include "utils/disk_prefix_write_coordinator.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm::prefix_write;

static int failures = 0;
#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAIL line " << __LINE__ << ": " #condition "\n";  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

int main() {
    const size_t oneMiB = size_t(1) << 20;

    // 用例 1：copy_ 抛异常（真机 stale_page_generation / bytes_mismatch 的等价形态）。
    {
        std::map<uint64_t, bool> ready{{0, true}};
        int copyCalls = 0;
        auto stream = CreateD2Stream(
            [&](uint64_t eventId) {
                const auto found = ready.find(eventId);
                return found != ready.end() && found->second;
            },
            [&](int, const TicketId &, size_t) -> std::string {
                ++copyCalls;
                throw std::runtime_error("injected copy failure");
            },
            size_t(32) << 20);
        CHECK(stream != nullptr);
        if (!stream) return 1;
        CHECK(stream->Submit({7, 0}, 0, oneMiB));
        bool threw = false;
        try {
            stream->PumpCopy();
        } catch (...) {
            threw = true;
        }
        CHECK(threw);
        CHECK(copyCalls == 1);              // 旧实现=1（第一次），随后会被再调
        CHECK(stream->Pending() == 0);      // 旧实现=1：条目留在队首
        CHECK(stream->SlotsInUse() == 0);   // 旧实现=1：槽位与预登记都未归还
        CHECK(stream->Failed({7, 0}));
        stream->PumpCopy();                 // 旧实现在这里会再调一次 copy_
        CHECK(copyCalls == 1);              // 旧实现=2
        stream->Stop();
    }

    // 用例 2：正常路径不被破坏 —— 每个 ticket 只交付一次。
    {
        std::map<uint64_t, bool> ready{{0, true}, {1, true}};
        std::vector<uint64_t> copied;
        auto stream = CreateD2Stream(
            [&](uint64_t eventId) {
                const auto found = ready.find(eventId);
                return found != ready.end() && found->second;
            },
            [&](int, const TicketId &ticket, size_t bytes) {
                copied.push_back(ticket.sequence);
                return std::string(bytes, char('a' + int(ticket.sequence)));
            },
            size_t(32) << 20);
        CHECK(stream != nullptr);
        if (!stream) return 1;
        CHECK(stream->Submit({8, 0}, 0, oneMiB));
        CHECK(stream->Submit({8, 1}, 1, oneMiB));
        stream->PumpCopy();
        CHECK(copied.size() == 2 && copied[0] == 0 && copied[1] == 1);
        for (uint64_t i = 0; i < 2; ++i) {
            TicketId ticket;
            HashedChunk chunk;
            CHECK(stream->TakeOrderedChunk(ticket, chunk, 5000));
            CHECK(ticket.requestId == 8 && ticket.sequence == i);
        }
        stream->PumpCopy();
        CHECK(copied.size() == 2);          // 不得重复投递
        CHECK(stream->Pending() == 0);
        CHECK(stream->SlotsInUse() == 0);
        stream->Stop();
    }

    // 用例 3：已拷贝完成但未取回的结果，Reset 必须归还槽位并记为失败（残留清理）。
    {
        std::map<uint64_t, bool> ready{{0, true}, {1, true}};
        auto stream = CreateD2Stream(
            [&](uint64_t eventId) {
                const auto found = ready.find(eventId);
                return found != ready.end() && found->second;
            },
            [&](int, const TicketId &ticket, size_t bytes) {
                return std::string(bytes, char('a' + int(ticket.sequence)));
            },
            size_t(32) << 20);
        CHECK(stream != nullptr);
        if (!stream) return 1;
        CHECK(stream->Submit({9, 0}, 0, oneMiB));
        CHECK(stream->Submit({9, 1}, 1, oneMiB));
        stream->PumpCopy();
        CHECK(stream->Pending() == 0);
        CHECK(stream->SlotsInUse() == 2);
        stream->Reset();
        CHECK(stream->SlotsInUse() == 0);
        CHECK(stream->Failed({9, 0}) && stream->Failed({9, 1}));
        stream->Stop();
    }

    // 用例 4：队列里还有未拷贝的条目时，Reset 也必须清空并记失败。
    {
        std::map<uint64_t, bool> ready{{0, false}};
        auto stream = CreateD2Stream(
            [&](uint64_t eventId) {
                const auto found = ready.find(eventId);
                return found != ready.end() && found->second;
            },
            [&](int, const TicketId &ticket, size_t bytes) {
                return std::string(bytes, char('a' + int(ticket.sequence)));
            },
            size_t(32) << 20);
        CHECK(stream != nullptr);
        if (!stream) return 1;
        CHECK(stream->Submit({10, 0}, 0, oneMiB));
        stream->PumpCopy();                 // 事件未完成：不拷贝
        CHECK(stream->Pending() == 1);
        stream->Reset();
        CHECK(stream->Pending() == 0);
        CHECK(stream->SlotsInUse() == 0);
        CHECK(stream->Failed({10, 0}));
        stream->Stop();
    }

    if (failures != 0) {
        std::cout << "FAILED " << failures << "\n";
        return 1;
    }
    std::cout << "PASS: d2 copy terminal (one ticket, one copy)\n";
    return 0;
}
