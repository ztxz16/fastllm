#pragma once

#include "fastllm.h"
#include "executor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fastllm::qwen4_pipeline {

struct TokenEmbeddingRows {
    uint64_t request_epoch = 0;
    // Independent row Data objects remain alive until the caller's upload
    // event completes; this object makes no CUDA allocation or transfer.
    std::unique_ptr<Data> bf16_rows;
    std::unique_ptr<Data> fp32_rows;
};

class SharedTokenEmbedding {
public:
    static SharedTokenEmbedding FromOwnedBf16(int vocab, int hidden,
                                               std::vector<uint16_t> payload,
                                               int max_rows) {
        if (vocab <= 0 || vocab > (1 << 24) || hidden <= 0 || max_rows <= 0 ||
            payload.size() != static_cast<uint64_t>(vocab) * hidden) {
            throw std::invalid_argument("invalid shared BF16 token embedding shape/payload");
        }
        return SharedTokenEmbedding(vocab,hidden,max_rows,std::move(payload));
    }

    uint64_t PayloadBytes() const { return payload_->size() * sizeof(uint16_t); }

    TokenEmbeddingRows GatherTokenRows(const std::vector<float> &token_ids,
                                       uint64_t request_epoch) const {
        if (token_ids.empty() || token_ids.size() > static_cast<size_t>(max_rows_)) {
            throw std::invalid_argument("embedding lookup exceeds the configured row budget");
        }
        for (float id : token_ids) {
            uint32_t bits;
            std::memcpy(&bits,&id,sizeof(bits));
            // Bit-level finiteness also holds when a consumer was compiled
            // with -ffast-math (which can optimize std::isfinite away).
            if ((bits & 0x7f800000u) == 0x7f800000u ||
                id < 0 || id >= vocab_ || std::floor(id) != id) {
                throw std::invalid_argument("token ID must be a finite integer inside the vocabulary");
            }
        }
        // EmbeddingDirect's global low-memory branch reads a file instead of
        // this immutable payload. Refuse it without changing process settings.
        if (GetLowMemMode()) {
            throw std::invalid_argument("shared CPU token embedding requires memory-table mode");
        }
        // Each thread owns an executor/profiler, and every invocation creates
        // new Data metadata. Reshape can only modify the temporary view.
        const auto payload_owner=payload_;
        thread_local Executor cpu_executor;
        Data table(BFLOAT16,{vocab_,hidden_},DataDevice::CPU,
                   const_cast<uint16_t *>(payload_owner->data()));
        table.lockInCPU=true;
        Data input(FLOAT32,{1,static_cast<int>(token_ids.size())},DataDevice::CPU,
                   const_cast<float *>(token_ids.data()));
        TokenEmbeddingRows output;
        output.request_epoch=request_epoch;
        output.bf16_rows=std::make_unique<Data>(BFLOAT16);
        cpu_executor.RunOnDevice("cpu","EmbeddingDirect",
            {{"input",&input},{"weight",&table},{"output",output.bf16_rows.get()}},{},{});
        output.fp32_rows=std::make_unique<Data>(FLOAT32,output.bf16_rows->dims);
        output.fp32_rows->Allocate(false);
        const auto *source=reinterpret_cast<const uint16_t *>(output.bf16_rows->cpuData);
        auto *dest=reinterpret_cast<float *>(output.fp32_rows->cpuData);
        for (uint64_t i=0;i<output.bf16_rows->Count(0);++i) {
            const uint32_t bits=static_cast<uint32_t>(source[i]) << 16;
            std::memcpy(dest+i,&bits,sizeof(bits));
        }
        return output;
    }

private:
    SharedTokenEmbedding(int vocab, int hidden, int max_rows,
                          std::vector<uint16_t> payload)
        : vocab_(vocab),hidden_(hidden),max_rows_(max_rows),
          payload_(std::make_shared<const std::vector<uint16_t>>(std::move(payload))) {}

    int vocab_;
    int hidden_;
    int max_rows_;
    // No caller retains a mutable alias to the moved vector allocation.
    const std::shared_ptr<const std::vector<uint16_t>> payload_;
};

} // namespace fastllm::qwen4_pipeline
