// Standalone: c++ -std=c++17 -Iinclude src/kvmem.cpp test/basic/test_kvmem_pages.cpp -o /tmp/test-kvmem
#include "kvmem.h"
#include <algorithm>

#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace fastllm;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)

template<class F> void Throws(F fn) {
    bool caught = false;
    try { fn(); } catch (const std::exception&) { caught = true; }
    CHECK(caught);
}

int main() {
    KvMemConfig config;
    config.enabled = true;
    config.maxTokens = 160;
    config.residentTokens = 24;
    config.sinkTokens = config.recentTokens = config.retrievalTokens = config.prefillTokens = 4;
    config.hostBytes = 320;
    CHECK(config.retrievalInterval == 64);
    for (int interval : {1, 17, 32, 64}) {
        auto other = config;
        other.retrievalInterval = interval;
        other.Validate(4); // Refresh intervals need not be page aligned.
    }
    for (int interval : {0, -1}) {
        auto other = config;
        other.retrievalInterval = interval;
        Throws([&] { other.Validate(4); });
    }
    for (int seed = 0; seed < 20; ++seed) {
        KvMemPages pages(config, 4, 8);
        std::vector<std::vector<uint8_t>> device(6, std::vector<uint8_t>(8, 0));
        std::vector<int> reads(40, 0);
        std::mt19937 rng(seed);
        auto read = [&](const std::vector<KvMemPages::ReadPage> &batch) {
            for (auto page : batch) {
                int logical = (device[page.slot][0] - 1) / 4;
                CHECK(++reads[logical] == 1); // immutable backing is reused
                std::memcpy(page.data, device[page.slot].data(), 8);
            }
        };
        auto write = [&](const std::vector<KvMemPages::WritePage> &batch) {
            for (auto page : batch) std::memcpy(device[page.slot].data(), page.data, 8);
        };
        for (int old = 0; old < 160;) {
            int n = std::min(160 - old, 1 + (int)(rng() % 4));
            std::vector<float> scores(40, 0);
            scores[(rng() % 20)] = 100;
            auto visible = pages.Append(n, scores, read, write);
            CHECK(std::is_sorted(visible.begin(), visible.end()));
            for (int t = old; t < old + n; ++t) {
                int slot = pages.Slot(t / 4);
                device[slot][t % 4] = t + 1;
                device[slot][4 + t % 4] = 255 - t;
            }
            std::vector<int> positions;
            for (int logical : visible) {
                int slot = pages.Slot(logical);
                for (int t = logical * 4; t < std::min(old + n, logical * 4 + 4); ++t) {
                    CHECK(device[slot][t % 4] == t + 1);
                    CHECK(device[slot][4 + t % 4] == 255 - t);
                    positions.push_back(t);
                }
            }
            // Compact causal masking is equivalent to the absolute mask for
            // every query row, including a chunk that crosses a page boundary.
            for (int row = 0; row < n; ++row) {
                int compactEnd = (int)positions.size() - n + row;
                for (int i = 0; i < (int)positions.size(); ++i)
                    CHECK((i <= compactEnd) == (positions[i] <= old + row));
            }
            CHECK(visible.front() == 0);
            CHECK(visible.back() == (old + n - 1) / 4);
            CHECK(pages.Stats().hostBytes <= config.hostBytes);
            int resident = 0;
            std::vector<int> occupied;
            for (int p = 0; p < (old + n + 3) / 4; ++p) {
                int slot;
                try { slot = pages.Slot(p); } catch (const std::out_of_range&) { continue; }
                ++resident;
                CHECK(std::find(occupied.begin(), occupied.end(), slot) == occupied.end());
                occupied.push_back(slot);
                for (int t = p * 4; t < std::min(old + n, p * 4 + 4); ++t) {
                    CHECK(device[slot][t % 4] == t + 1);
                    CHECK(device[slot][4 + t % 4] == 255 - t);
                }
            }
            CHECK(resident == pages.Stats().residentPages && resident <= 6);
            CHECK(resident >= (int)visible.size());
            old += n;
        }
        CHECK(pages.Stats().evictedPages > 0 && pages.Stats().restoredPages > 0);
        auto before = pages.Stats();
        Throws([&] { pages.Append(1, {}, read, write); });
        CHECK(pages.Stats().tokens == before.tokens);
    }
    Throws([&] { KvMemPages pages(config, 0, 8); });
    // Rejected tails must never enter immutable host backing, including a
    // transaction that completes the old writer page and creates another.
    {
        KvMemPages pages(config, 4, 8);
        std::vector<std::vector<uint8_t>> device(6, std::vector<uint8_t>(8));
        auto read = [&](const std::vector<KvMemPages::ReadPage> &batch) {
            for (auto page : batch) std::memcpy(page.data, device[page.slot].data(), 8);
        };
        auto write = [&](const std::vector<KvMemPages::WritePage> &batch) {
            for (auto page : batch) std::memcpy(device[page.slot].data(), page.data, 8);
        };
        int old = 0;
        for (int iteration = 0; iteration < 70; ++iteration) {
            int accepted = iteration % 5;
            pages.BeginTransaction(4);
            Throws([&] { pages.BeginTransaction(1); });
            Throws([&] { pages.Append(1, {}, read, write); });
            std::vector<float> scores(40, 0);
            scores[iteration % 20] = 100;
            pages.Append(4, scores, read, write);
            for (int t = old; t < old + 4; ++t) {
                auto &slot = device[pages.Slot(t / 4)];
                slot[t % 4] = t - old < accepted ? t + 1 : 255;
            }
            Throws([&] { pages.Append(4, {}, read, write); });
            Throws([&] { pages.FinishTransaction(5); });
            pages.FinishTransaction(accepted);
            old += accepted;
            CHECK(pages.Tokens() == old && !pages.InTransaction());
            int resident = 0;
            for (int p = 0; p < 40; ++p) {
                int slot;
                try { slot = pages.Slot(p); } catch (const std::out_of_range&) { continue; }
                CHECK(p * 4 < old);
                ++resident;
                for (int t = p * 4; t < std::min(old, p * 4 + 4); ++t)
                    CHECK(device[slot][t % 4] == t + 1);
            }
            CHECK(pages.Stats().residentPages == resident && resident <= 6);
            CHECK(resident >= (int)pages.Visible().size());
            if (old) CHECK(pages.Visible().back() == (old - 1) / 4);
            for (int p : pages.Visible()) for (int t = p * 4; t < std::min(old, p * 4 + 4); ++t)
                CHECK(device[pages.Slot(p)][t % 4] == t + 1);
        }
        pages.BeginTransaction(1);
        pages.FinishTransaction(0); // cancellation before any layer ran
        CHECK(pages.Tokens() == old);
        CHECK(pages.Stats().evictedPages > 0 && pages.Stats().restoredPages > 0);
        Throws([&] { pages.FinishTransaction(0); });
    }
    // Revisit a recent retrieval after one intervening selection. Its physical
    // page stays cached, but must not leak into the intervening attention view.
    {
        auto cached = config;
        cached.residentTokens = 32;
        KvMemPages pages(cached, 4, 8);
        std::vector<std::vector<uint8_t>> device(8, std::vector<uint8_t>(8));
        auto backup = [&](const std::vector<KvMemPages::ReadPage> &batch) {
            for (auto page : batch) std::memcpy(page.data, device[page.slot].data(), 8);
        };
        auto restore = [&](const std::vector<KvMemPages::WritePage> &batch) {
            for (auto page : batch) std::memcpy(device[page.slot].data(), page.data, 8);
        };
        auto append = [&](int preferred) {
            const int old = pages.Tokens();
            std::vector<float> scores(40);
            scores[preferred] = 100;
            pages.Append(4, scores, backup, restore);
            for (int t = old; t < old + 4; ++t)
                device[pages.Slot(t / 4)][t % 4] = t + 1;
            if (old >= 32) {
                CHECK(pages.Visible() == std::vector<int>({0, preferred, old / 4 - 1, old / 4}));
                CHECK(pages.Stats().residentPages == 8);
            }
            for (int p : pages.Visible()) for (int t = p * 4; t < p * 4 + 4; ++t)
                CHECK(device[pages.Slot(p)][t % 4] == t + 1);
        };
        while (pages.Tokens() < 80) append(1);
        const auto cold = pages.Stats().restoredPages;
        append(2);
        CHECK(pages.Stats().restoredPages == cold + 1);
        const int slot = pages.Slot(2);
        append(3);
        CHECK(pages.Slot(2) == slot);
        const auto before = pages.Stats().restoredPages;
        append(2);
        CHECK(pages.Slot(2) == slot && pages.Stats().restoredPages == before);
    }
    auto invalid = config; invalid.hostBytes = 1;
    Throws([&] { KvMemPages pages(invalid, 4, 8); });
    invalid = config; invalid.residentTokens = 12;
    Throws([&] { invalid.Validate(4); });
    KvMemPages failed(config, 4, 8);
    auto read = [](const std::vector<KvMemPages::ReadPage>&) { throw std::runtime_error("transfer failed"); };
    auto write = [](const std::vector<KvMemPages::WritePage>&) {};
    for (int i = 0; i < 6; ++i) failed.Append(4, {}, read, write);
    Throws([&] { failed.Append(4, {}, read, write); });
    Throws([&] { failed.Append(1, {}, [](const std::vector<KvMemPages::ReadPage>&) {}, write); });

    // Multi-page callbacks must back up old contents before any destination
    // slot is overwritten. A partial restore failure poisons the whole store.
    {
        auto batched = config;
        batched.residentTokens = 40;
        batched.retrievalTokens = 16;
        batched.prefillTokens = 8;
        KvMemPages pages(batched, 4, 8);
        std::vector<std::vector<uint8_t>> device(10, std::vector<uint8_t>(8));
        size_t maxRead = 0, maxWrite = 0;
        bool failRestore = false;
        auto backup = [&](const std::vector<KvMemPages::ReadPage> &batch) {
            maxRead = std::max(maxRead, batch.size());
            for (auto page : batch) std::memcpy(page.data, device[page.slot].data(), 8);
        };
        auto restore = [&](const std::vector<KvMemPages::WritePage> &batch) {
            maxWrite = std::max(maxWrite, batch.size());
            for (auto page : batch) {
                std::memcpy(device[page.slot].data(), page.data, 8);
                if (failRestore) throw std::runtime_error("partial restore failed");
            }
        };
        auto append = [&](int preferred) {
            const int old = pages.Tokens();
            std::vector<float> scores(40);
            for (int p = preferred; p < preferred + 4; ++p) scores[p] = 100;
            pages.Append(4, scores, backup, restore);
            for (int t = old; t < old + 4; ++t) {
                auto &slot = device[pages.Slot(t / 4)];
                slot[t % 4] = t + 1;
                slot[4 + t % 4] = 255 - t;
            }
            for (int p : pages.Visible()) for (int t = p * 4; t < std::min(old + 4, p * 4 + 4); ++t) {
                CHECK(device[pages.Slot(p)][t % 4] == t + 1);
                CHECK(device[pages.Slot(p)][4 + t % 4] == 255 - t);
            }
        };
        while (pages.Tokens() < 120) append(1);
        for (int preferred : {5, 1, 5, 1}) append(preferred);
        CHECK(maxRead >= 4 && maxWrite == 4);
        failRestore = true;
        Throws([&] { append(5); });
        failRestore = false;
        Throws([&] { append(1); });
    }
    std::cout << "KVMem page tests passed\n";
}
