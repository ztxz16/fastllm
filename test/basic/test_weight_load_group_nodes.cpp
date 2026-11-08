#include "utils/weight_load_group_nodes.h"
#include <atomic>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct Payload {
    std::string data;
    bool freed = false;
    void FreeSpace() { data.clear(); freed = true; }
};
struct Single { std::vector<std::string> inputs; std::string output; };
struct Rule { std::set<std::string> allInputs; std::vector<Single> rules; };
using Map = std::unordered_map<std::string, Payload>;
static void Require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void LegacyStructure() {
    Map weights; weights.emplace("input", Payload{"data"});
    weights.emplace("reader", Payload{"stable"});
    const auto count = weights.size(), buckets = weights.bucket_count();
    std::atomic<int> stage{0}; bool sawMutation = false;
    std::thread writer([&] {
        weights.emplace("merged", Payload{"data"});
        stage.store(1, std::memory_order_release);
        while (stage.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        weights.erase("input");
    });
    std::thread reader([&] {
        while (stage.load(std::memory_order_acquire) != 1) std::this_thread::yield();
        Require(weights.at("reader").data == "stable", "Reader fixture corrupted.");
        sawMutation = weights.size() != count || weights.bucket_count() != buckets;
        stage.store(2, std::memory_order_release);
    });
    writer.join(); reader.join();
    Require(!sawMutation, "Legacy worker inserts changed map structure during the worker phase.");
}
static void Contract() {
    std::vector<Rule> rules = {{{"gate", "up"}, {{{"gate", "up"}, "gateup"}}},
                               {{"other"}, {{{"other"}, "other-merged"}}}};
    const auto index = fastllm::BuildWeightMergeIndex(rules);
    Map weights; weights.emplace("gate", Payload{"G"}); weights.emplace("up", Payload{"U"});
    weights.emplace("reader", Payload{"stable"});
    const std::set<std::string> group = {"gate", "up"};
    Require(fastllm::PrepareMergeOutputNodes(weights, rules, index, group) == 1,
            "Current-group merged output was not prepared.");
    Require(weights.find("other-merged") == weights.end(), "Prepared another group's node.");
    Require(weights.at("gate").data == "G", "Preparation modified an existing payload.");
    weights.at("gateup").data = "preserved";
    Require(fastllm::PrepareMergeOutputNodes(weights, rules, index, group) == 0,
            "Duplicate preparation must preserve existing nodes.");
    Require(weights.at("gateup").data == "preserved", "Preparation overwrote a merged payload.");
    const auto count = weights.size(), buckets = weights.bucket_count();
    const auto *readerNode = &weights.at("reader");
    std::mutex lock; std::set<std::string> pending; std::atomic<bool> done{false}, stable{true};
    std::thread reader([&] {
        while (!done.load()) {
            if (weights.size() != count || weights.bucket_count() != buckets ||
                &weights.at("reader") != readerNode || weights.at("reader").data != "stable") stable = false;
        }
    });
    std::thread worker([&] {
        const auto &operation = rules[0].rules[0];
        weights.at(operation.output).data = weights.at(operation.inputs[0]).data + weights.at(operation.inputs[1]).data;
        std::lock_guard<std::mutex> guard(lock);
        fastllm::ReleaseMergedInputs(weights, operation.inputs, pending);
        done = true;
    });
    worker.join(); reader.join();
    Require(stable, "Worker phase changed map structure or invalidated node references.");
    Require(weights.at("gateup").data == "GU", "Merge result or input sequence changed.");
    Require(weights.at("gate").freed && weights.at("up").freed && weights.at("gate").data.empty(),
            "Consumed input payload was not released immediately.");
    Require(weights.count("gate") && pending == std::set<std::string>({"gate", "up"}),
            "Input metadata must remain until join.");
    fastllm::EraseConsumedWeightNodes(weights, pending);
    Require(!weights.count("gate") && !weights.count("up") && pending.empty(), "Join cleanup left consumed nodes.");
    std::vector<Rule> ordered = {{{"a"}, {{{"a"}, "oa"}}}, {{"b"}, {{{"b"}, "ob"}}}};
    const auto orderedIndex = fastllm::BuildWeightMergeIndex(ordered);
    Map sequence; sequence.emplace("a", Payload{"A"}); sequence.emplace("b", Payload{"B"});
    fastllm::PrepareMergeOutputNodes(sequence, ordered, orderedIndex, std::set<std::string>{"b", "a"});
    std::string trace;
    for (const auto &rule : ordered) {
        const auto &single = rule.rules[0];
        sequence.at(single.output).data = sequence.at(single.inputs[0]).data;
        trace += single.output;
        fastllm::ReleaseMergedInputs(sequence, single.inputs, pending);
    }
    Require(trace == "oaob" && sequence.at("oa").data == "A" && sequence.at("ob").data == "B",
            "Multiple-rule order or merged output changed.");
    fastllm::EraseConsumedWeightNodes(sequence, pending);
    Map mtp;
    const std::vector<std::string> mtpNames = {"mtp.expert.0.gateup", "mtp.expert.0.down"};
    for (const auto &name : mtpNames) mtp.try_emplace(name);
    const auto mtpCount = mtp.size(), mtpBuckets = mtp.bucket_count();
    for (const auto &name : mtpNames) mtp.at(name).data = "BF16-chunk";
    for (const auto &name : mtpNames) mtp.at(name).FreeSpace();
    Require(mtp.size() == mtpCount && mtp.bucket_count() == mtpBuckets, "MTP chunk modified map structure.");
    std::cout << "contract=PASS map_structure=STABLE payload_release=IMMEDIATE erased_after_join=PASS mtp=PASS\n";
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--legacy-structure") LegacyStructure();
        else { Require(argc == 1, "Usage: test [--legacy-structure]"); Contract(); }
    } catch (const std::exception &error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
    return 0;
}
