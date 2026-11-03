#ifndef FASTLLM_PARALLEL_MODEL_H
#define FASTLLM_PARALLEL_MODEL_H
namespace fastllm { namespace parallel {
// Optional capability implemented by model adapters. Configuration is immutable
// after weights have been partitioned or requests have been admitted.
class AttentionParallelModel {
public:
    virtual ~AttentionParallelModel() = default;
    virtual void ConfigureAttentionParallel(int attentionDP) = 0;
};
}}
#endif
