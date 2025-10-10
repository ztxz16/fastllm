//
// NUMA-aware MOE implementation
// Based on lktransformers MOE design with per-expert NUMA distribution
//

#ifdef USE_NUMA

#include "devices/numa/numamoe.h"
#include "utils.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace fastllm {

    // Activation function (SiLU/Swish)
    static inline float silu_activation(float x) {
        return x / (1.0f + expf(-x));
    }

    void NumaMoeLayer::Initialize(const std::vector<Data*>& gate_weights,
                                  const std::vector<Data*>& up_weights,
                                  const std::vector<Data*>& down_weights,
                                  int stride) {
        if (initialized_) {
            Cleanup();
        }

        num_experts_ = gate_weights.size();
        stride_ = stride;
        
        AssertInFastLLM(gate_weights.size() == up_weights.size() && 
                       gate_weights.size() == down_weights.size(),
                       "NumaMoeLayer: All weight vectors must have same size");
        AssertInFastLLM(num_experts_ > 0, "NumaMoeLayer: Must have at least one expert");

        // Get dimensions from first expert
        hidden_size_ = gate_weights[0]->dims[1];
        intermediate_size_ = gate_weights[0]->dims[0];

        // Initialize each expert with NUMA-distributed weights
        experts_.resize(num_experts_);
        for (int i = 0; i < num_experts_; i++) {
            experts_[i].Initialize(i, *gate_weights[i], *up_weights[i], *down_weights[i], stride);
        }

        // Allocate per-NUMA-node temporary buffers
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        int num_nodes = pool.GetNumNodes();
        
        node_gate_outputs_.resize(num_nodes);
        node_up_outputs_.resize(num_nodes);
        node_intermediate_.resize(num_nodes);
        node_down_outputs_.resize(num_nodes);
        
        // Size buffers for maximum expected batch size
        const int max_batch = 128;
        for (int nid = 0; nid < num_nodes; nid++) {
            node_gate_outputs_[nid].resize(max_batch * intermediate_size_);
            node_up_outputs_[nid].resize(max_batch * intermediate_size_);
            node_intermediate_[nid].resize(max_batch * intermediate_size_);
            node_down_outputs_[nid].resize(max_batch * hidden_size_);
        }

        initialized_ = true;
    }

    void NumaMoeLayer::Forward(const float *routing, const float *input, float *output,
                              int n, int hidden_size, int intermediate_size) {
        AssertInFastLLM(initialized_, "NumaMoeLayer: Must initialize before forward");
        AssertInFastLLM(hidden_size == hidden_size_ && intermediate_size == intermediate_size_,
                       "NumaMoeLayer: Dimension mismatch");

        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        
        // Clear output
        memset(output, 0, n * hidden_size * sizeof(float));

        // Process each expert
        for (int expert_idx = 0; expert_idx < num_experts_; expert_idx++) {
            auto& expert = experts_[expert_idx];
            auto& mlp = expert.mlp_weight;
            
            // For each sample, check if this expert is used
            std::vector<int> active_samples;
            for (int i = 0; i < n; i++) {
                if (routing[i * num_experts_ + expert_idx] > 1e-6f) {
                    active_samples.push_back(i);
                }
            }
            
            if (active_samples.empty()) {
                continue;
            }

            int n_active = active_samples.size();
            
            // Step 1: Gate and Up projections (NUMA-parallelized)
            int nth_gate_up = intermediate_size / stride_;
            pool.DoNumaWork(1, nth_gate_up,
                nullptr,
                [&](int task_id) {
                    int numa_node = NumaThreadPool::GetThreadNumaNode();
                    const auto& gate_block = mlp.GetGateUpBlock(numa_node);
                    
                    int block_id = task_id;
                    if (block_id < gate_block.start_block || 
                        block_id >= gate_block.start_block + gate_block.num_blocks) {
                        return;  // Not this node's block
                    }
                    
                    int local_offset = block_id - gate_block.start_block;
                    
                    // Get NUMA-local weight pointers
                    void* gate_weight = mlp.GetGateBlock(block_id);
                    void* up_weight = mlp.GetUpBlock(block_id);
                    
                    if (!gate_weight || !up_weight) return;
                    
                    float* gate_out = node_gate_outputs_[numa_node].data();
                    float* up_out = node_up_outputs_[numa_node].data();
                    float* intermediate = node_intermediate_[numa_node].data();
                    
                    // Compute for all active samples
                    for (int idx = 0; idx < n_active; idx++) {
                        int i = active_samples[idx];
                        
                        // Gate projection
                        float gate_sum = 0.0f;
                        for (int l = 0; l < hidden_size; l++) {
                            gate_sum += input[i * hidden_size + l] * 
                                       ((float*)gate_weight)[local_offset * stride_ * hidden_size + l];
                        }
                        gate_out[idx * intermediate_size + block_id * stride_] = gate_sum;
                        
                        // Up projection
                        float up_sum = 0.0f;
                        for (int l = 0; l < hidden_size; l++) {
                            up_sum += input[i * hidden_size + l] * 
                                     ((float*)up_weight)[local_offset * stride_ * hidden_size + l];
                        }
                        up_out[idx * intermediate_size + block_id * stride_] = up_sum;
                        
                        // Activation: gate * silu(up)
                        intermediate[idx * intermediate_size + block_id * stride_] = 
                            silu_activation(gate_out[idx * intermediate_size + block_id * stride_]) *
                            up_out[idx * intermediate_size + block_id * stride_];
                    }
                },
                nullptr
            );

            // Step 2: Down projection (NUMA-parallelized)
            int nth_down = hidden_size / stride_;
            pool.DoNumaWork(1, nth_down,
                nullptr,
                [&](int task_id) {
                    int numa_node = NumaThreadPool::GetThreadNumaNode();
                    const auto& down_block = mlp.GetDownBlock(numa_node);
                    
                    int block_id = task_id;
                    if (block_id < down_block.start_block || 
                        block_id >= down_block.start_block + down_block.num_blocks) {
                        return;
                    }
                    
                    int local_offset = block_id - down_block.start_block;
                    void* down_weight = mlp.GetDownBlock(block_id);
                    if (!down_weight) return;
                    
                    float* intermediate = node_intermediate_[numa_node].data();
                    float* down_out = node_down_outputs_[numa_node].data();
                    
                    // Compute for all active samples
                    for (int idx = 0; idx < n_active; idx++) {
                        int i = active_samples[idx];
                        
                        float down_sum = 0.0f;
                        for (int l = 0; l < intermediate_size; l++) {
                            down_sum += intermediate[idx * intermediate_size + l] * 
                                       ((float*)down_weight)[local_offset * stride_ * intermediate_size + l];
                        }
                        
                        // Apply routing weight and accumulate to output
                        float routing_weight = routing[i * num_experts_ + expert_idx];
                        down_out[idx * hidden_size + block_id * stride_] = down_sum * routing_weight;
                    }
                },
                nullptr
            );

            // Step 3: Accumulate results to output
            for (int idx = 0; idx < n_active; idx++) {
                int i = active_samples[idx];
                for (int nid = 0; nid < pool.GetNumNodes(); nid++) {
                    float* down_out = node_down_outputs_[nid].data();
                    for (int j = 0; j < hidden_size; j++) {
                        output[i * hidden_size + j] += down_out[idx * hidden_size + j];
                    }
                }
            }
        }
    }

    void NumaMoeLayer::Cleanup() {
        if (!initialized_) {
            return;
        }

        experts_.clear();
        node_gate_outputs_.clear();
        node_up_outputs_.clear();
        node_intermediate_.clear();
        node_down_outputs_.clear();
        
        initialized_ = false;
    }

    // Helper function for MOE merge
    void RunNumaMoeMerge(const std::vector<float*>& expert_outputs,
                        const float *routing_weights,
                        float *output,
                        int n, int num_experts, int hidden_size) {
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        
        // NUMA-parallelized merge
        pool.DoWork(n * hidden_size,
            nullptr,
            [&](int task_id) {
                int i = task_id / hidden_size;
                int j = task_id % hidden_size;
                
                float sum = 0.0f;
                for (int e = 0; e < num_experts; e++) {
                    sum += expert_outputs[e][i * hidden_size + j] * routing_weights[i * num_experts + e];
                }
                output[i * hidden_size + j] = sum;
            },
            nullptr
        );
    }

}  // namespace fastllm

#endif  // USE_NUMA

