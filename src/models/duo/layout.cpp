#include "models/duo/layout.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fastllm::duo {
namespace {
void require(bool ok, const char *why) {
    if (!ok)
        throw std::invalid_argument(why);
}
void geometry(int h, int w) {
    require(h >= 2 && w >= 2 && h % 2 == 0 && w % 2 == 0 && h <= 256 && w <= 256,
            "Duo latent geometry must be even and in [2,256]");
}
void add(Layout &l, Kind kind, int block, int count, int tag, int time) {
    if (!count)
        return;
    const int first = l.rows();
    l.groups.push_back({kind, block, first, count});
    l.tags.insert(l.tags.end(), count, tag);
    l.time.insert(l.time.end(), count, time);
    l.positions.resize(size_t(l.rows()) * 3);
}
void spatial(Layout &l, int first, int frames, int h, int w, int chunk, bool reference) {
    const double area = std::sqrt(double(h) * w);
    const int starts[] = {0, 1, 5, 9, 13};
    for (int t = 0; t < frames; ++t)
        for (int y = 0; y < h / 2; ++y)
            for (int x = 0; x < w / 2; ++x) {
                int i = first + (t * h / 2 + y) * w / 2 + x;
                l.video.push_back(i);
                l.positions[i * 3] =
                    reference ? 0.f : float((frameStart(chunk) + starts[t]) * (1000.0 / 24.0) * .04);
                l.positions[i * 3 + 1] = float(((1 - h / area) / 2 + y * 2 / area) * 32);
                l.positions[i * 3 + 2] = float(((1 - w / area) / 2 + x * 2 / area) * 32);
            }
}
} // namespace
bool visible(const Group &q, const Group &k, bool refiner) {
    if (refiner)
        return k.kind == Kind::Shared || (q.kind == Kind::Text && k.kind == Kind::Text && q.block == k.block);
    if (q.kind == Kind::Reference || q.kind == Kind::Shared)
        return q.kind == k.kind;
    if (k.kind == Kind::Reference || k.kind == Kind::Shared)
        return true;
    if (k.kind == Kind::Text)
        return k.block == q.block;
    if (k.kind == Kind::Noisy)
        return q.kind == Kind::Noisy && k.block == q.block;
    if (k.kind == Kind::Clean)
        return k.block < q.block || (q.kind == Kind::Clean && k.block == q.block);
    return false;
}
int groupRows(const std::vector<Group> &groups) {
    int cursor = 0;
    for (const auto &g : groups) {
        require(g.first == cursor && g.count > 0 && g.count <= 1000000 - cursor,
                "Duo groups must partition contiguous rows");
        cursor += g.count;
    }
    return cursor;
}
std::vector<AttentionDomain> attentionDomains(const std::vector<Group> &queries,
                                              const std::vector<Group> &keys, bool refiner) {
    groupRows(queries);
    groupRows(keys);
    std::vector<AttentionDomain> domains;
    for (const auto &q : queries) {
        AttentionDomain domain{{q.first, q.count}, {}};
        for (const auto &k : keys) {
            if (!visible(q, k, refiner))
                continue;
            auto &ranges = domain.keys;
            if (!ranges.empty() && ranges.back().first + ranges.back().count == k.first)
                ranges.back().count += k.count;
            else
                ranges.push_back({k.first, k.count});
        }
        require(!domain.keys.empty(), "attention query has no visible keys");
        domains.push_back(std::move(domain));
    }
    return domains;
}
std::vector<float> schedule(int steps, float shift) {
    require(steps > 0 && steps <= 1000 && std::isfinite(shift) && shift > 0, "invalid Duo sampler");
    std::vector<float> out(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        const float t = float(i) / steps, noise = 1.f - t;
        out[i] = 1.f - shift * noise / (1.f + (shift - 1.f) * noise);
    }
    out.front() = 0;
    out.back() = 1;
    return out;
}
int64_t frameStart(int64_t k) {
    require(k >= 0 && k <= std::numeric_limits<int64_t>::max() / 68000, "invalid Duo chunk clock");
    return k * 17;
}
// Half-open audio spans use ceil(frame * 32000 / 24), matching Duo's media clock.
int64_t sampleStart(int64_t k) {
    frameStart(k);
    return (k * 68000 + 2) / 3;
}
Layout prefixLayout(int h, int w, int shared) {
    geometry(h, w);
    require(shared > 0 && shared <= 32768, "invalid shared Intent length");
    Layout l;
    add(l, Kind::Reference, -1, h * w / 4, 0, 0);
    spatial(l, 0, 1, h, w, 0, true);
    const int first = l.rows();
    add(l, Kind::Shared, -1, shared, 1, 0);
    for (int i = 0; i < shared; ++i) {
        l.text.push_back(first + i);
        l.textSelection.push_back(i);
        l.positions[(first + i) * 3] = -1;
    }
    l.refinerGroups.push_back({Kind::Shared, -1, 0, shared});
    l.validate(shared, 0);
    return l;
}
Layout chunkLayout(int h, int w, int shared, int text, int chunk, bool clean) {
    geometry(h, w);
    frameStart(chunk);
    require(shared > 0 && text > 0 && shared + text <= 32768, "invalid Duo Intent length");
    Layout l;
    add(l, Kind::Text, chunk, text, 1, 0);
    for (int i = 0; i < text; ++i) {
        l.text.push_back(i);
        l.textSelection.push_back(shared + i);
        l.positions[i * 3] = float(frameStart(chunk) * (1000.0 / 24.0) * .04);
    }
    int videoRows = 5 * h * w / 4, first = l.rows();
    add(l, clean ? Kind::Clean : Kind::Noisy, chunk, videoRows + 58, 0, clean ? 0 : 1);
    spatial(l, first, 5, h, w, chunk, false);
    const double area = std::sqrt(double(h) * w);
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 29; ++t) {
            int i = first + videoRows + c * 29 + t;
            l.audio.push_back(i);
            l.tags[i] = 2;
            l.positions[i * 3] = float(float(frameStart(chunk) * (1000.0 / 24.0) * .04) + float(t));
            l.positions[i * 3 + 2] = float(((1 - w / area) / 2 + (c ? w / 2 - 1 : 0) * 2 / area) * 32);
        }
    l.refinerGroups = {{Kind::Shared, -1, 0, shared}, {Kind::Text, chunk, shared, text}};
    l.validate(shared, text);
    return l;
}
void Layout::validate(int shared, int current) const {
    require(groupRows(groups) == rows() && positions.size() == size_t(rows()) * 3 &&
                time.size() == tags.size(),
            "invalid Duo layout");
    require(groupRows(refinerGroups) == shared + current && text.size() == textSelection.size(),
            "invalid Duo refiner selection");
    std::vector<int> assigned(rows());
    auto check = [&](const std::vector<int> &ids, int tag) {
        for (int i : ids) {
            require(i >= 0 && i < rows() && tags[i] == tag && !assigned[i]++,
                    "invalid Duo modality partition");
        }
    };
    check(video, 0);
    check(text, 1);
    check(audio, 2);
    for (int n : assigned)
        require(n == 1, "missing Duo modality row");
    for (int i : textSelection)
        require(i >= 0 && i < shared + current, "invalid selected Intent row");
    for (float v : positions)
        require(std::isfinite(v), "nonfinite Duo position");
}
uint64_t Timeline::begin(int chunk) {
    require(!failed_ && !busy_ && chunk == next_, "Duo chunk is out of order or session requires reset");
    busy_ = true;
    return ++generation_;
}
bool Timeline::current(uint64_t ticket) const {
    return busy_ && !failed_ && ticket == generation_;
}
void Timeline::committed(uint64_t ticket) {
    require(current(ticket), "stale Duo commit");
    busy_ = false;
    ++next_;
}
void Timeline::cancel() {
    ++generation_;
    busy_ = false;
    failed_ = true;
}
void Timeline::fail(uint64_t ticket) {
    if (current(ticket))
        cancel();
}
void Timeline::reset() {
    ++generation_;
    busy_ = false;
    failed_ = false;
    next_ = 0;
}
} // namespace fastllm::duo
