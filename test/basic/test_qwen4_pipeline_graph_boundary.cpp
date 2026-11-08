#include "models/qwen4_pipeline_layout.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

int main() {
    using fastllm::qwen4_pipeline::VerificationDraftCount;
    assert(VerificationDraftCount(1, 126, 128, 2048, true) == 1);
    assert(VerificationDraftCount(1, 127, 128, 2048, true) == 0);
    assert(VerificationDraftCount(1, 383, 384, 2048, true) == 0);
    assert(VerificationDraftCount(3, 125, 128, 2048, true) == 2);
    assert(VerificationDraftCount(3, 126, 128, 2048, true) == 1);
    assert(VerificationDraftCount(3, 127, 128, 2048, true) == 0);
    assert(VerificationDraftCount(3, 128, 128, 2048, true) == 3);
    assert(VerificationDraftCount(3, 2046, 4096, 2048, true) == 1);
    assert(VerificationDraftCount(3, 2048, 4096, 2048, true) == 3);
    assert(VerificationDraftCount(3, 126, 128, 2048, false) == 3);
    // The policy is idempotent: the receiving stage cannot trim it again.
    for (int previous = 0; previous < 2200; ++previous) {
        for (int drafts = 0; drafts <= 8; ++drafts) {
            int count = VerificationDraftCount(drafts, previous, 384, 2048, true);
            assert(count >= 0 && count <= drafts);
            assert(VerificationDraftCount(count, previous, 384, 2048, true) == count);
        }
    }
    std::cout << "PASS PP2 graph candidate boundaries, MTP1-8 and idempotence\n";
}
