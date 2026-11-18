#include "models/duo/layout.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace fastllm::duo;
void check(bool v) {
    if (!v)
        throw std::runtime_error("Duo layout invariant failed");
}
template <class F> void rejects(F f) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw);
}
int main() {
    try {
        auto p = prefixLayout(8, 12, 3), s = chunkLayout(8, 12, 3, 4, 2);
        check(p.rows() == 27 && s.video.size() == 120 && s.audio.size() == 58 && s.text.size() == 4);
        Group text{Kind::Text, 2, 0, 1}, noise{Kind::Noisy, 2, 0, 1}, clean{Kind::Clean, 2, 0, 1},
            past{Kind::Clean, 1, 0, 1}, future{Kind::Clean, 3, 0, 1};
        check(!visible(text, noise) && visible(noise, text) && !visible(noise, clean) &&
              visible(clean, clean));
        check(visible(noise, past) && !visible(noise, future) && !visible(text, future));
        check(!visible(p.groups[0], p.groups[1]) && !visible(p.groups[1], p.groups[0]));
        check(!visible(p.groups[1], text, true) && visible(text, p.groups[1], true));
        for (int k = 0; k < 10000; ++k) {
            check(frameStart(k) == int64_t(k) * 17);
            check(sampleStart(k) == (int64_t(k) * 68000 + 2) / 3);
        }
        check(sampleStart(1) == 22667 && sampleStart(2) == 45334 && sampleStart(3) == 68000);
        auto t = schedule(4, 5);
        check(t.size() == 5 && t[0] == 0 && t[4] == 1 && std::abs(t[1] - .0625) < 1e-7);
        Timeline timeline;
        auto first = timeline.begin(0);
        rejects([&] { timeline.begin(0); });
        timeline.committed(first);
        rejects([&] { timeline.begin(0); });
        auto second = timeline.begin(1);
        timeline.cancel();
        rejects([&] { timeline.committed(second); });
        timeline.reset();
        auto reset = timeline.begin(0);
        rejects([&] { timeline.committed(first); });
        timeline.committed(reset);
        rejects([] { prefixLayout(3, 8, 1); });
        rejects([] { schedule(0, 5); });
        rejects([] { frameStart(-1); });
        // Reference and shared keys are adjacent. A prior text group separates them
        // from history that the current noisy query may attend to.
        std::vector<Group> keys{{Kind::Reference, -1, 0, 2}, {Kind::Shared, -1, 2, 3},
                                {Kind::Text, 1, 5, 2},       {Kind::Clean, 1, 7, 4},
                                {Kind::Text, 2, 11, 1},      {Kind::Noisy, 2, 12, 3}};
        auto domains = attentionDomains({{Kind::Noisy, 2, 0, 3}}, keys);
        check(domains.size() == 1 && domains[0].queries.count == 3);
        const auto &ranges = domains[0].keys;
        check(ranges.size() == 2 && ranges[0].first == 0 && ranges[0].count == 5 && ranges[1].first == 7 &&
              ranges[1].count == 8);
        rejects([] { attentionDomains({{Kind::Shared, -1, 0, 1}}, {{Kind::Noisy, 0, 0, 1}}); });
        std::cout << "Duo layout, causal masks, rational clocks and stale commits passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
