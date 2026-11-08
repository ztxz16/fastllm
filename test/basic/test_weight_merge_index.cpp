#include "utils/weight_merge_index.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

struct Rule { std::set<std::string> allInputs; };
using Ids = std::vector<std::size_t>;
static void Require(bool ok, const char *reason) {
    if (!ok) throw std::runtime_error(reason);
}
static Ids FullScan(const std::vector<Rule> &rules, const std::string &name) {
    Ids result;
    for (std::size_t i = 0; i < rules.size(); ++i)
        if (rules[i].allInputs.find(name) != rules[i].allInputs.end()) result.push_back(i);
    return result;
}
static void Contract() {
    std::vector<Rule> rules = {{{"a.gate", "a.up", "shared"}}, {{"shared", "b.gate"}},
                               {{"", "shared"}}, {{}}, {{"a.gate", "shared"}}};
    const auto index = fastllm::BuildWeightMergeIndex(rules);
    static_assert(std::is_same<fastllm::WeightMergeIndex::key_type, std::string_view>::value,
                  "Index keys must borrow the original strings.");
    for (const auto &name : {"a.gate", "a.up", "shared", "b.gate", "", "unknown",
                            "a.gate.extra", "prefix.a.gate", "a.gat"})
        Require(FullScan(rules, name) == fastllm::FindWeightMergeRules(index, name),
                "Full-scan/index mismatch.");
    Require(fastllm::FindWeightMergeRules(index, "shared") == Ids({0, 1, 2, 4}),
            "Shared-input rule order changed.");
    for (const auto &item : index) {
        const auto &original = *rules[item.second.front()].allInputs.find(std::string(item.first));
        Require(item.first.data() == original.data(), "Index copied a string payload.");
    }
    const std::vector<Rule> empty;
    Require(fastllm::BuildWeightMergeIndex(empty).empty(), "Empty rules must yield empty index.");
    auto mutableRules = rules;
    auto rebuilt = fastllm::BuildWeightMergeIndex(mutableRules);
    rebuilt.clear(); // Release borrowed keys before invalidating their owners.
    mutableRules.clear();
    mutableRules.push_back({{"changed", "shared"}});
    rebuilt = fastllm::BuildWeightMergeIndex(mutableRules);
    Require(fastllm::FindWeightMergeRules(rebuilt, "changed") == Ids({0}), "Rebuild missed new key.");
    Require(fastllm::FindWeightMergeRules(rebuilt, "a.gate").empty(), "Rebuild retained old key.");
    std::cout << "contract=PASS shared_order=PASS borrowed_keys=PASS rebuild=PASS\n";
}
static long long Micros(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
}
static std::size_t RssKiB(const std::string &field = "VmRSS:") {
    std::ifstream status("/proc/self/status");
    for (std::string line; std::getline(status, line);)
        if (line.compare(0, field.size(), field) == 0) return std::stoull(line.substr(field.size()));
    return 0;
}
static void Benchmark(const char *namesFile, int layers, int experts, bool memoryOnly = false) {
    std::ifstream input(namesFile);
    Require(bool(input), "Cannot read checkpoint-name evidence.");
    std::vector<std::string> names;
    for (std::string name; std::getline(input, name);) names.push_back(std::move(name));
    Require(!names.empty(), "Checkpoint-name evidence is empty.");
    std::vector<Rule> rules;
    rules.reserve((layers + 1) * (experts + 1));
    for (int layer = 0; layer <= layers; ++layer) {
        const std::string mlp = layer == layers ? "mtp.layers.0.mlp." :
            "model.language_model.layers." + std::to_string(layer) + ".mlp.";
        rules.push_back({{mlp + "shared_expert.gate_proj.weight",
                          mlp + "shared_expert.up_proj.weight"}});
        for (int expert = 0; expert < experts; ++expert) {
            const auto prefix = mlp + "experts." + std::to_string(expert) + ".";
            rules.push_back({{prefix + "gate_proj.weight", prefix + "up_proj.weight"}});
        }
    }
    const auto beforeIndexRss = RssKiB();
    auto start = std::chrono::steady_clock::now();
    const auto index = fastllm::BuildWeightMergeIndex(rules);
    const auto buildUs = Micros(start);
    const auto afterIndexRss = RssKiB();
    if (memoryOnly) {
        std::cout << "memory_only=1 names=" << names.size() << " rules=" << rules.size()
                  << " keys=" << index.size() << " index_build_us=" << buildUs
                  << " before_index_rss_kib=" << beforeIndexRss << " after_index_rss_kib=" << afterIndexRss
                  << " index_rss_delta_kib=" << (static_cast<long long>(afterIndexRss) - beforeIndexRss)
                  << " benchmark_peak_rss_kib=" << RssKiB("VmHWM:") << '\n';
        return;
    }
    std::vector<Ids> expected; expected.reserve(names.size());
    start = std::chrono::steady_clock::now();
    for (const auto &name : names) expected.push_back(FullScan(rules, name));
    const auto scanUs = Micros(start);
    std::size_t mismatches = 0, checksum = 0;
    start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto &found = fastllm::FindWeightMergeRules(index, names[i]);
        mismatches += found != expected[i];
        for (auto id : found) checksum += id + 1;
    }
    const auto lookupUs = Micros(start);
    std::cout << "names=" << names.size() << " rules=" << rules.size()
              << " keys=" << index.size() << " mismatches=" << mismatches
              << " index_build_us=" << buildUs << " fullscan_us=" << scanUs
              << " lookup_compare_us=" << lookupUs << " checksum=" << checksum
              << " before_index_rss_kib=" << beforeIndexRss
              << " after_index_rss_kib=" << afterIndexRss
              << " index_rss_delta_kib=" << (static_cast<long long>(afterIndexRss) - beforeIndexRss) << '\n';
    Require(mismatches == 0, "Checkpoint-name behavior mismatch.");
}
int main(int argc, char **argv) {
    try {
        Contract();
        if (argc == 5 && (std::string(argv[1]) == "--bench" || std::string(argv[1]) == "--memory"))
            Benchmark(argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::string(argv[1]) == "--memory");
        else Require(argc == 1, "Usage: test [--bench|--memory names-file layers experts]");
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
    return 0;
}
