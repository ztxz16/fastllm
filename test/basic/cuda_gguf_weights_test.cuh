#pragma once

// Shared deterministic packed GGUF weight fixtures. Included after the GGUF types.
static std::vector<uint8_t> Weights(ggml_type type, int n, int k) {
    std::vector<uint8_t> bytes(size_t(n) * ggml_row_size(type, k));
    uint32_t random = 7821;
    for (auto &v : bytes) {
        random = random * 1664525U + 1013904223U;
        v = random >> 24;
    }
    const size_t blockSize = ggml_type_size(type);
    for (size_t i = 0; i < bytes.size(); i += blockSize) {
        void *p = bytes.data() + i;
        const half d = __float2half_rn(float(1 + (i / blockSize) % 4) / 16384.0f);
        switch (type) {
        case GGML_TYPE_IQ1_M: {
            auto *sc = reinterpret_cast<uint16_t *>(static_cast<block_iq1_m *>(p)->scales);
            uint16_t bits;
            std::memcpy(&bits, &d, 2);
            for (int j = 0; j < 4; ++j)
                sc[j] = (sc[j] & 0x0fff) | (((bits >> (4 * j)) & 15) << 12);
            break;
        }
        case GGML_TYPE_IQ3_S:
            static_cast<block_iq3_s *>(p)->d = d;
            break;
        case GGML_TYPE_IQ3_XXS:
            static_cast<block_iq3_xxs *>(p)->d = d;
            break;
        case GGML_TYPE_IQ4_XS:
            static_cast<block_iq4_xs *>(p)->d = d;
            break;
        case GGML_TYPE_IQ2_XXS:
            static_cast<block_iq2_xxs *>(p)->d = d;
            break;
        case GGML_TYPE_IQ2_S:
            static_cast<block_iq2_s *>(p)->d = d;
            break;
        case GGML_TYPE_IQ2_XS:
            static_cast<block_iq2_xs *>(p)->d = d;
            break;
        case GGML_TYPE_Q4_K: {
            auto *w = static_cast<block_q4_K *>(p);
            w->dm = __halves2half2(d, d);
            break;
        }
        case GGML_TYPE_Q2_K: {
            auto *w = static_cast<block_q2_K *>(p);
            w->dm = __halves2half2(d, d);
            break;
        }
        default:
            throw std::runtime_error("unhandled fixture type");
        }
    }
    return bytes;
}
