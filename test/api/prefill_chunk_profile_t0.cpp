#include "utils/prefill_chunk_profile.h"
#include <cstdio>
#include <cstdlib>

using Profile = fastllm::PrefillChunkProfile;
using Phase = Profile::Phase;
static uint64_t ticks[] = {120, 620, 650, 800, 900, 1100};
static std::size_t reads = 0;

static uint64_t ReadClock() {
    if (reads >= sizeof(ticks) / sizeof(ticks[0])) std::abort();
    return ticks[reads++];
}
static void Check(bool condition, const char *reason) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", reason);
        std::exit(1);
    }
}

int main() {
    static_assert(sizeof(Profile) <= 128, "Profile storage must stay bounded");
    Profile disabled(false, ReadClock);
    disabled.MarkAt(Phase::Begin, 100);
    for (std::size_t i = 0; i < static_cast<std::size_t>(Phase::Count); ++i)
        disabled.Mark(static_cast<Phase>(i));
    Check(reads == 0 && !disabled.Enabled() && !disabled.Complete(),
          "off/user collection must not read clocks");

    Profile active(true, ReadClock);
    active.MarkAt(Phase::Begin, 100); // Existing observed chunk timestamp.
    Check(!active.Complete(), "partial intervals must not be reported");
    active.Mark(Phase::TargetBegin);
    active.Mark(Phase::TargetEnd);
    active.Mark(Phase::DraftBegin);
    active.Mark(Phase::DraftEnd);
    active.Mark(Phase::CheckpointBegin);
    active.Mark(Phase::CheckpointEnd);
    Check(active.Complete() && reads == 6, "six additional boundary samples");
    Check(active.Duration(Phase::TargetBegin, Phase::TargetEnd) == 500,
          "target call interval");
    Check(active.Duration(Phase::DraftBegin, Phase::DraftEnd) == 150,
          "draft call interval");
    Check(active.Duration(Phase::CheckpointBegin, Phase::CheckpointEnd) == 200,
          "checkpoint call interval");
    Check(active.Duration(Phase::Begin, Phase::CheckpointEnd) == 1000,
          "whole chunk must retain setup and inter-call waits");
    Check(active.Duration(Phase::Begin, Phase::CheckpointEnd) >
              active.Duration(Phase::TargetBegin, Phase::TargetEnd) +
              active.Duration(Phase::DraftBegin, Phase::DraftEnd) +
              active.Duration(Phase::CheckpointBegin, Phase::CheckpointEnd),
          "unattributed waits must not disappear from whole-chunk time");
    Check(active.Duration(Phase::CheckpointEnd, Phase::Begin) == 0,
          "reversed interval must not underflow");

    Profile backwards(true, ReadClock);
    backwards.MarkAt(Phase::Begin, 100);
    backwards.MarkAt(Phase::TargetBegin, 99);
    Check(!backwards.Complete(), "clock regression must be invalid");
    Profile duplicate(true, ReadClock);
    duplicate.MarkAt(Phase::Begin, 100);
    duplicate.MarkAt(Phase::Begin, 101);
    Check(!duplicate.Complete(), "duplicate boundary must be invalid");
    Profile skipped(true, ReadClock);
    skipped.MarkAt(Phase::Begin, 100);
    skipped.MarkAt(Phase::DraftBegin, 101);
    Check(!skipped.Complete(), "missing target interval must not be invented");
    Profile noClock(true, nullptr);
    noClock.Mark(Phase::Begin);
    Check(!noClock.Complete(), "missing clock must be invalid");
    Profile zero(true, ReadClock);
    zero.MarkAt(Phase::Begin, 0);
    Check(!zero.Complete(), "unobserved startup timestamp must not be reported");
    Check(reads == 6, "invalid/disabled probes must not add clock reads");
    std::puts("PASS: bounded prefill phase accounting; disabled and invalid clocks; gaps preserved");
}
