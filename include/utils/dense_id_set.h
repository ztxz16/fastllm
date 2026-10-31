#ifndef FASTLLM_DENSE_ID_SET_H
#define FASTLLM_DENSE_ID_SET_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace fastllm {
    // Repeated, short-lived sets over a bounded dense ID space. Clearing normally
    // costs O(1), and inserts never allocate a node. Not shared between threads.
    template<class Stamp = uint32_t> class DenseIdSet {
        static_assert(std::is_unsigned<Stamp>::value, "The epoch must be unsigned.");
        std::vector<Stamp> marks;
        Stamp epoch = 0;
    public:
        void Reset(size_t capacity) {
            if (marks.size() < capacity) marks.resize(capacity, 0);
            if (++epoch == 0) {
                std::fill(marks.begin(), marks.end(), 0);
                epoch = 1;
            }
        }
        bool Contains(int id) const {
            return id >= 0 && (size_t)id < marks.size() && marks[id] == epoch;
        }
        void Insert(int id) {
            assert(id >= 0 && (size_t)id < marks.size() && epoch != 0);
            marks[id] = epoch;
        }
    };
}
#endif
