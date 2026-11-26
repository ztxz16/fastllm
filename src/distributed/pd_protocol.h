#ifndef FASTLLM_PD_PROTOCOL_H
#define FASTLLM_PD_PROTOCOL_H
#include "distributed/pd_native.h"
#include "process_protocol.h"
#include <cstring>

namespace fastllm { namespace distributed { namespace protocol {

/** 编码固定长度本地存储调用；网络身份不经此通道传递，由服务交接模块先校验。 */
inline std::vector<uint8_t> EncodePD(const PDCommand &c) {
    std::vector<uint8_t> out;
    Put32(out, uint32_t(c.action)); Put64(out, c.key);
    Put32(out, c.tokens); Put32(out, c.capacity); Put32(out, c.position);
    Put32(out, uint32_t(c.targetRank));
    Put32(out, c.tokenIds.size());
    for (auto id : c.tokenIds) Put32(out, id);
    return out;
}

/** 严格解析本地命令，不接收裸 GPU 地址。 */
inline PDCommand DecodePD(const std::vector<uint8_t> &in) {
    if (in.size() < 32 || Get32(in, 0) < 1 || Get32(in, 0) > 7 || !Get64(in, 4) ||
        Get32(in, 28) > 2048 || uint64_t(Get32(in, 28)) * 4 + 32 != in.size())
        throw std::runtime_error("invalid PD storage command");
    PDCommand c{PDAction(Get32(in, 0)), Get64(in, 4), Get32(in, 12), Get32(in, 16), Get32(in, 20), int32_t(Get32(in, 24)), {}};
    for (size_t offset = 32; offset < in.size(); offset += 4) c.tokenIds.push_back(Get32(in, offset));
    if ((c.action == PDAction::RestoreLocal) != !c.tokenIds.empty())
        throw std::runtime_error("invalid local-prefix token list");
    return c;
}

/** 编码 worker 自己生成的 region；固定记录尺寸避免结构体 padding 和 ABI 差异。 */
inline std::vector<uint8_t> EncodePDRegions(const std::vector<PDRegion> &regions) {
    constexpr size_t width = 144;
    if (regions.size() > (kMaxPayload - 4) / width) throw std::runtime_error("PD region count exceeds local protocol");
    std::vector<uint8_t> out;
    Put32(out, regions.size());
    for (const auto &r : regions) {
        for (auto v : {r.layer, r.component, r.tokenBegin, r.tokenEnd, r.headBegin, r.headEnd, r.headDim, r.pageSize}) Put32(out, v);
        for (auto v : {r.allocationBytes, r.offset, r.tokenStride, r.headStride}) Put64(out, v);
        out.insert(out.end(), r.memory.begin(), r.memory.end());
        out.insert(out.end(), r.device.begin(), r.device.end());
    }
    return out;
}

/** 解码可信本地 worker 的描述，并验证总长度；后续服务层仍须验证语义和授权。 */
inline std::vector<PDRegion> DecodePDRegions(const std::vector<uint8_t> &in) {
    if (in.size() < 4 || uint64_t(Get32(in, 0)) * 144 + 4 != in.size())
        throw std::runtime_error("invalid PD regions response");
    std::vector<PDRegion> out(Get32(in, 0));
    size_t offset = 4;
    for (auto &r : out) {
        for (auto *v : {&r.layer, &r.component, &r.tokenBegin, &r.tokenEnd, &r.headBegin, &r.headEnd, &r.headDim, &r.pageSize}) {
            *v = Get32(in, offset); offset += 4;
        }
        for (auto *v : {&r.allocationBytes, &r.offset, &r.tokenStride, &r.headStride}) {
            *v = Get64(in, offset); offset += 8;
        }
        std::memcpy(r.memory.data(), in.data() + offset, 64); offset += 64;
        std::memcpy(r.device.data(), in.data() + offset, 16); offset += 16;
    }
    return out;
}

} } }
#endif
