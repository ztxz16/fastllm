#pragma once

#include "fastllm.h"
#include "fastkv/fastkv.h"

namespace fastllm {

// Model adapters serialize logical state, never allocator addresses, page IDs,
// CUDA graphs, workspaces, or scheduler handles. Tensor payloads own their data
// independently of the request being exported.
class PDStateWriter {
  public:
    fastkv::Writer metadata;
    std::vector<fastkv::Buffer> buffers;
    void Tensor(const Data &tensor);
    fastkv::Message Finish();
};

class PDStateReader {
  public:
    explicit PDStateReader(const fastkv::Message &message);
    fastkv::Reader metadata;
    // Return a compact CPU tensor. The adapter chooses the destination worker
    // and allocator; the transport does not allocate live model cache pages.
    void Tensor(Data &tensor);
    void End();

  private:
    const fastkv::Message &message;
    size_t next = 0;
};

} // namespace fastllm
