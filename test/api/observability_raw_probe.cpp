// CPU-only probe of the actual production event emitter (no model/CUDA).
#include "utils/prefix_cache_logging.h"
int main() {
    std::puts("[Fastllm] original loader line");
    fastllm::RegisterFastllmPublicRequestId(7, "req-host-7");
    fastllm::FastllmEvent("cache", 7, "cache_candidates", "l2_query_status=busy");
    fastllm::FastllmEvent("cache", 7, "cache_result", "source_tier=L1 effective_reused_tokens=128");
    fastllm::FastllmEvent("prefill", 7, "prefill", "tokens=3");
    fastllm::FastllmEvent("decode", 7, "dflash", "accepted=3 draft=8");
    fastllm::FastllmEvent("scheduler", 7, "state", "alive=1 pending=0");
    fastllm::FastllmEvent("future", 7, "backend_detail", "payload=7");
    fastllm::FastllmEvent("vision", 7, "vision_cache_lookup_result", "hit=0");
    fastllm::FastllmEvent("output", 7, "request_finished", "reason=eos output_tokens=12");
    fastllm::UnregisterFastllmPublicRequestId(7);
    return 0;
}
