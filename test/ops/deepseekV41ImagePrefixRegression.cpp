#include "deepseekv41.h"
#include <cstdlib>
#include <iostream>

using namespace fastllm;

static void require(bool ok, const char *message) {
    if (!ok) { std::cerr << message << std::endl; std::exit(1); }
}

int main() {
    DeepSeekV41HistoryCacheManager cache;
    std::vector<int> tokens(100, 42);
    DeepSeekV41ImageCacheKeys a = {{20, 6, 6, 8, 1, 2, 3, 4, 5, 6, 7, 8}};
    auto record = std::make_shared<DeepSeekV41HistoryMemory>();
    record->tokens = tokens; record->totalLen = 100; record->imageKeys = a;
    cache.Record(record);
    auto candidates = cache.GetCandidates(tokens, a);
    require(candidates.size() == 1 && candidates[0].second == 99, "same image should match");
    auto changed = a; changed[0][11]++;
    require(cache.GetCandidates(tokens, changed)[0].second == 20, "changed image must stop before its span");
    require(cache.GetCandidates(tokens)[0].second == 20, "missing image must not match placeholders");
    auto added = a; added.push_back({70, 6, 6, 8, 9, 2, 3, 4, 5, 6, 7, 8});
    require(cache.GetCandidates(tokens, added)[0].second == 70, "appended image should preserve earlier cache");
    auto resized = a; resized[0][1]++;
    require(cache.GetCandidates(tokens, resized)[0].second == 20, "image geometry is part of cache identity");
    auto moved = a; moved[0][0]--;
    require(cache.GetCandidates(tokens, moved)[0].second == 19, "moved image must stop at earlier position");
    auto partial = tokens; partial.resize(25);
    require(cache.GetCandidates(partial, changed)[0].second == 20, "partial image match must also verify identity");
    auto record2 = std::make_shared<DeepSeekV41HistoryMemory>();
    record2->tokens = tokens; record2->totalLen = 100; record2->imageKeys = changed;
    cache.Record(record2);
    require(cache.memorys.size() == 2, "different images with identical tokens must coexist");
    require(cache.GetCandidates(tokens, changed)[0].first == record2, "choose matching media snapshot");
    auto text = std::make_shared<DeepSeekV41HistoryMemory>();
    text->tokens = tokens; text->totalLen = 100;
    cache.Record(text);
    require(cache.GetCandidates(tokens)[0].first == text, "text-only prefix reuse must remain intact");
    std::cout << "DeepSeek-V4.1 image prefix regression: PASS" << std::endl;
}
