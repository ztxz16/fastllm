#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace fastllm::duo {
enum class Kind { Reference, Shared, Text, Noisy, Clean };
struct Group {
    Kind kind;
    int block, first, count;
};
bool visible(const Group &query, const Group &key, bool refiner = false);
struct RowRange {
    int first, count;
};

// All query rows within a Duo group have the same legal keys. Run one
// softmax over the union of those keys, preserving their original order.
struct AttentionDomain {
    RowRange queries;
    std::vector<RowRange> keys;
};
std::vector<AttentionDomain> attentionDomains(const std::vector<Group> &queries,
                                              const std::vector<Group> &keys, bool refiner = false);
int groupRows(const std::vector<Group> &);
std::vector<float> schedule(int steps, float shift);
// Integer media clocks: no accumulation of a rounded 708 ms interval.
int64_t frameStart(int64_t chunk);
int64_t sampleStart(int64_t chunk);
struct Layout {
    std::vector<Group> groups, refinerGroups;
    std::vector<int> video, audio, text, textSelection, tags, time;
    std::vector<float> positions;
    int rows() const { return int(tags.size()); }
    void validate(int sharedRows, int currentRows) const;
};
Layout prefixLayout(int latentHeight, int latentWidth, int sharedRows);
Layout chunkLayout(int latentHeight, int latentWidth, int sharedRows, int intentRows, int chunk,
                   bool clean = false);
// Explicit transaction lifecycle for generation, commit, cancellation and reset.
class Timeline {
  public:
    uint64_t begin(int chunk);
    void committed(uint64_t ticket);
    void cancel();
    void fail(uint64_t ticket);
    void reset();
    bool current(uint64_t ticket) const;
    int next() const { return next_; }
    bool failed() const { return failed_; }

  private:
    uint64_t generation_ = 0;
    int next_ = 0;
    bool busy_ = false, failed_ = false;
};
} // namespace fastllm::duo
