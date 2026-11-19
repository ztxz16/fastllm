#ifndef FASTLLM_DISTRIBUTED_BATCH_INVARIANT_H
#define FASTLLM_DISTRIBUTED_BATCH_INVARIANT_H

#include <cstdint>
#include <vector>

namespace fastllm {
namespace distributed {

/** 返回协议中的版本、线程块、warp、指令块、流水阶段及 Split-K 分片数；用于核对 worker。 */
inline std::vector<uint32_t> MoeBatchInvariantConfig() {
    return {1, 64, 128, 32, 32, 64, 32, 8, 8, 4, 2, 1};
}
constexpr const char *kMoeBatchInvariantBackend = "cutlass_sm70_moe_experts_v1";

} // namespace distributed
} // namespace fastllm
#endif
