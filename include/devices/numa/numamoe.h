//
// NUMA-aware MOE (Mixture of Experts) implementation
// Each expert's weights distributed across NUMA nodes
//

#ifndef FASTLLM_NUMAMOE_H
#define FASTLLM_NUMAMOE_H

#ifdef USE_NUMA

#include "fastllm.h"
#include "devices/numa/numathreadpool.h"
#include "devices/numa/numaweight.h"
#include <vector>

namespace fastllm {

    // Per-expert NUMA-distributed weights
    struct NumaExpertWeights {
        int expert_id;
        NumaMlpWeight mlp_weight;  // gate, up, down projections
        
        void Initialize(int eid, const Data& gate, const Data& up, const Data& down, int stride) {
            expert_id = eid;
            mlp_weight.Initialize(gate, up, down, stride);
        }
    };

    // NUMA-aware MOE layer manager
    class NumaMoeLayer {
    public:
        NumaMoeLayer() : initialized_(false), num_experts_(0) {}
        ~NumaMoeLayer() { Cleanup(); }

        // Initialize MOE layer with multiple experts
        // Each expert's weights will be distributed across NUMA nodes
        void Initialize(const std::vector<Data*>& gate_weights,
                       const std::vector<Data*>& up_weights,
                       const std::vector<Data*>& down_weights,
                       int stride);

        // Forward pass for MOE
        // routing: [n, num_experts] routing weights
        // input: [n, hidden_size] input data
        // output: [n, hidden_size] output data
        void Forward(const float *routing, const float *input, float *output,
                    int n, int hidden_size, int intermediate_size);

        int GetNumExperts() const { return num_experts_; }

    private:
        void Cleanup();

        bool initialized_;
        int num_experts_;
        int stride_;
        int hidden_size_;
        int intermediate_size_;
        
        std::vector<NumaExpertWeights> experts_;
        
        // Temporary buffers (per-NUMA-node to avoid false sharing)
        std::vector<std::vector<float>> node_gate_outputs_;
        std::vector<std::vector<float>> node_up_outputs_;
        std::vector<std::vector<float>> node_intermediate_;
        std::vector<std::vector<float>> node_down_outputs_;
    };

    // Helper function for NUMA-aware MOE merge operation
    void RunNumaMoeMerge(const std::vector<float*>& expert_outputs,
                        const float *routing_weights,
                        float *output,
                        int n, int num_experts, int hidden_size);

}  // namespace fastllm

#endif  // USE_NUMA

#endif  // FASTLLM_NUMAMOE_H

