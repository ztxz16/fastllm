//
// NUMA-aware weight distribution and management
// Based on lktransformers weight sharding strategy
//

#ifndef FASTLLM_NUMAWEIGHT_H
#define FASTLLM_NUMAWEIGHT_H

#ifdef USE_NUMA

#include "fastllm.h"
#include "devices/numa/numathreadpool.h"
#include <vector>
#include <cstring>

namespace fastllm {

    // NUMA-distributed weight for linear operations
    // Weights are sharded across NUMA nodes for locality
    class NumaLinearWeight {
    public:
        NumaLinearWeight() : initialized_(false) {}
        
        ~NumaLinearWeight() {
            Cleanup();
        }

        // Initialize weight distribution across NUMA nodes
        // n: output dimension, m: input dimension, stride: block size
        void Initialize(const Data& weight, int stride);

        // Get weight pointer for a specific block on current NUMA node
        void* GetBlockWeight(int block_id);

        // Get block information for a NUMA node
        const NumaBlock& GetNumaBlock(int node_id) const {
            return numa_blocks_[node_id];
        }

        // Get total number of blocks
        int GetTotalBlocks() const { return total_blocks_; }

        // Get stride (block size)
        int GetStride() const { return stride_; }

        // Get weight data type
        DataType GetDataType() const { return data_type_; }

        // Get input dimension
        int GetInputDim() const { return input_dim_; }

        // Get output dimension
        int GetOutputDim() const { return output_dim_; }

    private:
        void Cleanup();

        bool initialized_;
        int stride_;
        int total_blocks_;
        int input_dim_;
        int output_dim_;
        size_t stride_bytes_;
        DataType data_type_;

        std::vector<NumaBlock> numa_blocks_;
        std::vector<void*> numa_weight_ptrs_;  // Per-node weight data
        std::vector<size_t> numa_weight_sizes_;  // Per-node sizes
    };

    // NUMA-distributed weights for MLP operations
    // Manages gate, up, and down projection weights
    class NumaMlpWeight {
    public:
        NumaMlpWeight() : initialized_(false) {}
        
        ~NumaMlpWeight() {
            Cleanup();
        }

        // Initialize MLP weight distribution
        void Initialize(const Data& gate_weight, const Data& up_weight, 
                       const Data& down_weight, int stride);

        // Get gate/up weight block
        void* GetGateBlock(int block_id);
        void* GetUpBlock(int block_id);
        
        // Get down weight block
        void* GetDownBlock(int block_id);

        // Get block information
        const NumaBlock& GetGateUpBlock(int node_id) const {
            return gate_up_blocks_[node_id];
        }
        
        const NumaBlock& GetDownBlock(int node_id) const {
            return down_blocks_[node_id];
        }

        int GetStride() const { return stride_; }
        int GetHiddenSize() const { return hidden_size_; }
        int GetIntermediateSize() const { return intermediate_size_; }

    private:
        void Cleanup();

        bool initialized_;
        int stride_;
        int hidden_size_;
        int intermediate_size_;

        std::vector<NumaBlock> gate_up_blocks_;
        std::vector<NumaBlock> down_blocks_;

        std::vector<void*> gate_numa_ptrs_;
        std::vector<void*> up_numa_ptrs_;
        std::vector<void*> down_numa_ptrs_;

        std::vector<size_t> gate_numa_sizes_;
        std::vector<size_t> up_numa_sizes_;
        std::vector<size_t> down_numa_sizes_;

        size_t stride_gate_bytes_;
        size_t stride_up_bytes_;
        size_t stride_down_bytes_;

        DataType gate_type_;
        DataType up_type_;
        DataType down_type_;
    };

    // Helper class for managing temporary NUMA-aware buffers
    class NumaBuffer {
    public:
        NumaBuffer() : data_(nullptr), size_(0), alignment_(64) {}
        
        ~NumaBuffer() {
            if (data_) {
                FreeAligned(data_);
            }
        }

        // Allocate buffer
        void Allocate(size_t size, size_t alignment = 64) {
            if (data_) {
                FreeAligned(data_);
            }
            size_ = size;
            alignment_ = alignment;
            data_ = AllocAligned(size, alignment);
        }

        // Get buffer pointer
        void* GetData() { return data_; }
        const void* GetData() const { return data_; }

        size_t GetSize() const { return size_; }

    private:
        void* data_;
        size_t size_;
        size_t alignment_;
    };

}  // namespace fastllm

#endif  // USE_NUMA

#endif  // FASTLLM_NUMAWEIGHT_H

