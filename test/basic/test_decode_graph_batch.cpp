#include "utils/decode_graph.h"
#undef NDEBUG
#include <cassert>

int main() {
    using namespace fastllm;
    for (int alignment : {1, 2, 8, 16})
    for (int maximum = 1; maximum <= 256; ++maximum) {
        auto buckets = DecodeGraphBatchBuckets(maximum, alignment);
        assert(buckets.back() == maximum);
        assert(!DecodeGraphBatch::Select(0, buckets));
        assert(!DecodeGraphBatch::Select(maximum + 1, buckets));
        for (int actual = 1; actual <= maximum; ++actual) {
            auto shape = DecodeGraphBatch::Select(actual, buckets);
            assert(shape && shape.actualRows == actual && shape.graphRows <= maximum);
            for (int bucket : buckets)
                assert(bucket < actual || bucket >= shape.graphRows);
            if (actual <= 4) assert(shape.graphRows == actual);
        }
    }
    auto buckets = DecodeGraphBatchBuckets(64);
    assert(DecodeGraphBatch::Select(39, buckets).graphRows == 40);
    assert(DecodeGraphBatch::Select(41, buckets).graphRows == 48);
    assert(DecodeGraphBatch::Select(44, buckets).graphRows == 48);
    assert(!DecodeGraphBatch::Select(1, {}));
    assert(DecodeGraphBatchBuckets(0).empty());
    assert(DecodeGraphBatchBuckets(64, 0).empty());
}
