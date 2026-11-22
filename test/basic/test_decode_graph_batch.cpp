#include "utils/decode_graph.h"
#include "utils/persistent_worker_group.h"
#undef NDEBUG
#include <cassert>

int main() {
    using namespace fastllm;
    PersistentWorkerGroup workers;
    std::vector<int> keys{0, 1};
    std::vector<std::exception_ptr> errors(keys.size());
    auto check = [](bool allowed) {
        assert(GraphCaptureScope::Allowed() == allowed);
        GraphCaptureScope nested(true);
        assert(GraphCaptureScope::Allowed() == allowed);
    };
    {
        GraphCaptureScope serving(false);
        workers.Run(keys, [&](int) { check(false); }, errors);
        workers.RunWithCaller(keys, [&](int) { check(false); }, errors);
        std::thread unrelated([&] { check(true); }); unrelated.join();
        check(false);
    }
    check(true);
    workers.Run(keys, [&](int) { check(true); }, errors);
    for (auto error : errors) assert(!error);
    try { GraphCaptureScope serving(false); throw 1; } catch (int) {}
    check(true);
    for (int alignment : {1, 2, 8, 16})
    for (int maximum = 1; maximum <= 256; ++maximum) {
        auto buckets = DecodeGraphBatchBuckets(maximum, alignment);
        const int padded = maximum <= 4 ? maximum :
            (maximum + alignment - 1) / alignment * alignment;
        assert(buckets.back() == padded);
        assert(!DecodeGraphBatch::Select(0, buckets));
        assert(!DecodeGraphBatch::Select(padded + 1, buckets));
        for (int actual = 1; actual <= maximum; ++actual) {
            auto shape = DecodeGraphBatch::Select(actual, buckets);
            assert(shape && shape.actualRows == actual && shape.graphRows <= padded);
            for (int bucket : buckets)
                assert(bucket < actual || bucket >= shape.graphRows);
            if (actual <= 4) assert(shape.graphRows == actual);
            else assert(shape.graphRows % alignment == 0);
        }
    }
    auto buckets = DecodeGraphBatchBuckets(64);
    assert(DecodeGraphBatch::Select(39, buckets).graphRows == 40);
    assert(DecodeGraphBatch::Select(41, buckets).graphRows == 48);
    assert(DecodeGraphBatch::Select(44, buckets).graphRows == 48);
    assert(!DecodeGraphBatch::Select(1, {}));
    assert(DecodeGraphBatchBuckets(0).empty());
    assert(DecodeGraphBatchBuckets(64, 0).empty());
    assert(DecodeGraphBatchBuckets(INT_MAX, 8).empty());
    auto endpoint = DecodeGraphBatchBuckets(220);
    for (int actual = 217; actual <= 220; ++actual)
        assert(DecodeGraphBatch::Select(actual, endpoint).graphRows == 224);
}
