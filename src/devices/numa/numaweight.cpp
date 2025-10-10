//
// NUMA-aware weight distribution and management implementation
//

#ifdef USE_NUMA

#include "devices/numa/numaweight.h"
#include "utils.h"
#include <cstring>

namespace fastllm {

    // Helper function to get data type size
    static size_t GetDataTypeSize(DataType dtype) {
        switch (dtype) {
            case DataType::FLOAT32:
                return sizeof(float);
            case DataType::FLOAT16:
            case DataType::BFLOAT16:
                return sizeof(uint16_t);
            case DataType::INT8:
                return sizeof(int8_t);
            case DataType::INT4:
            case DataType::INT4_NOZERO:
                return 0.5;  // Will be handled specially
            case DataType::INT32PARAM:
                return sizeof(int);
            default:
                return sizeof(float);
        }
    }

    // Helper function to calculate weight bytes based on type
    static size_t CalculateWeightBytes(int rows, int cols, DataType dtype) {
        if (dtype == DataType::INT4 || dtype == DataType::INT4_NOZERO) {
            // INT4: 2 values per byte
            return (size_t)rows * cols / 2;
        }
        return (size_t)rows * cols * GetDataTypeSize(dtype);
    }

    // NumaLinearWeight implementation
    void NumaLinearWeight::Initialize(const Data& weight, int stride) {
        if (initialized_) {
            Cleanup();
        }

        AssertInFastLLM(weight.dims.size() == 2, 
                       "NumaLinearWeight: weight must be 2D (output_dim x input_dim)");

        output_dim_ = weight.dims[0];
        input_dim_ = weight.dims[1];
        stride_ = stride;
        data_type_ = weight.dataType;

        total_blocks_ = output_dim_ / stride_;
        stride_bytes_ = CalculateWeightBytes(stride_, input_dim_, data_type_);

        // Get NUMA configuration
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        numa_blocks_ = pool.CalculateNumaBlocks(total_blocks_);
        int num_nodes = pool.GetNumNodes();

        // Allocate weight memory on each NUMA node
        numa_weight_ptrs_.resize(num_nodes);
        numa_weight_sizes_.resize(num_nodes);

        for (int nid = 0; nid < num_nodes; nid++) {
            int n_blocks = numa_blocks_[nid].num_blocks;
            size_t node_size = n_blocks * stride_bytes_;

            numa_weight_sizes_[nid] = node_size;
            numa_weight_ptrs_[nid] = NumaAllocAligned(node_size, nid);

            if (!numa_weight_ptrs_[nid]) {
                ErrorInFastLLM("NumaLinearWeight: Failed to allocate memory on node " + 
                              std::to_string(nid));
            }

            // Copy weight data to NUMA node
            int start_block = numa_blocks_[nid].start_block;
            for (int ib = 0; ib < n_blocks; ib++) {
                int global_block = start_block + ib;
                uint8_t* src = weight.cpuData + global_block * stride_bytes_;
                uint8_t* dst = (uint8_t*)numa_weight_ptrs_[nid] + ib * stride_bytes_;
                memcpy(dst, src, stride_bytes_);
            }
        }

        initialized_ = true;
    }

    void* NumaLinearWeight::GetBlockWeight(int block_id) {
        if (!initialized_) {
            return nullptr;
        }

        // Determine which NUMA node this block belongs to
        int numa_node = NumaThreadPool::GetThreadNumaNode();
        if (numa_node < 0 || numa_node >= (int)numa_blocks_.size()) {
            // Fallback: find the node that contains this block
            for (size_t i = 0; i < numa_blocks_.size(); i++) {
                if (block_id >= numa_blocks_[i].start_block && 
                    block_id < numa_blocks_[i].start_block + numa_blocks_[i].num_blocks) {
                    numa_node = i;
                    break;
                }
            }
        }

        if (numa_node < 0 || numa_node >= (int)numa_blocks_.size()) {
            return nullptr;
        }

        // Calculate local offset within the NUMA node
        int local_offset = block_id - numa_blocks_[numa_node].start_block;
        if (local_offset < 0 || local_offset >= numa_blocks_[numa_node].num_blocks) {
            return nullptr;
        }

        return (uint8_t*)numa_weight_ptrs_[numa_node] + local_offset * stride_bytes_;
    }

    void NumaLinearWeight::Cleanup() {
        if (!initialized_) {
            return;
        }

        for (size_t i = 0; i < numa_weight_ptrs_.size(); i++) {
            if (numa_weight_ptrs_[i]) {
                NumaFreeAligned(numa_weight_ptrs_[i], numa_weight_sizes_[i]);
            }
        }

        numa_weight_ptrs_.clear();
        numa_weight_sizes_.clear();
        numa_blocks_.clear();
        initialized_ = false;
    }

    // NumaMlpWeight implementation
    void NumaMlpWeight::Initialize(const Data& gate_weight, const Data& up_weight,
                                   const Data& down_weight, int stride) {
        if (initialized_) {
            Cleanup();
        }

        AssertInFastLLM(gate_weight.dims.size() == 2 && 
                       up_weight.dims.size() == 2 && 
                       down_weight.dims.size() == 2,
                       "NumaMlpWeight: all weights must be 2D");

        hidden_size_ = gate_weight.dims[1];
        intermediate_size_ = gate_weight.dims[0];
        stride_ = stride;

        gate_type_ = gate_weight.dataType;
        up_type_ = up_weight.dataType;
        down_type_ = down_weight.dataType;

        // Calculate sizes
        int nth_gate_up = intermediate_size_ / stride_;
        stride_gate_bytes_ = CalculateWeightBytes(stride_, hidden_size_, gate_type_);
        stride_up_bytes_ = CalculateWeightBytes(stride_, hidden_size_, up_type_);

        int nth_down = hidden_size_ / stride_;
        stride_down_bytes_ = CalculateWeightBytes(stride_, intermediate_size_, down_type_);

        // Get NUMA configuration
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        gate_up_blocks_ = pool.CalculateNumaBlocks(nth_gate_up);
        down_blocks_ = pool.CalculateNumaBlocks(nth_down);
        int num_nodes = pool.GetNumNodes();

        // Allocate gate and up weights
        gate_numa_ptrs_.resize(num_nodes);
        up_numa_ptrs_.resize(num_nodes);
        gate_numa_sizes_.resize(num_nodes);
        up_numa_sizes_.resize(num_nodes);

        for (int nid = 0; nid < num_nodes; nid++) {
            int n_blocks = gate_up_blocks_[nid].num_blocks;
            
            gate_numa_sizes_[nid] = n_blocks * stride_gate_bytes_;
            up_numa_sizes_[nid] = n_blocks * stride_up_bytes_;

            gate_numa_ptrs_[nid] = NumaAllocAligned(gate_numa_sizes_[nid], nid);
            up_numa_ptrs_[nid] = NumaAllocAligned(up_numa_sizes_[nid], nid);

            if (!gate_numa_ptrs_[nid] || !up_numa_ptrs_[nid]) {
                ErrorInFastLLM("NumaMlpWeight: Failed to allocate gate/up memory on node " + 
                              std::to_string(nid));
            }

            // Copy weight data
            int start_block = gate_up_blocks_[nid].start_block;
            for (int ib = 0; ib < n_blocks; ib++) {
                int global_block = start_block + ib;
                
                uint8_t* gate_src = gate_weight.cpuData + global_block * stride_gate_bytes_;
                uint8_t* gate_dst = (uint8_t*)gate_numa_ptrs_[nid] + ib * stride_gate_bytes_;
                memcpy(gate_dst, gate_src, stride_gate_bytes_);

                uint8_t* up_src = up_weight.cpuData + global_block * stride_up_bytes_;
                uint8_t* up_dst = (uint8_t*)up_numa_ptrs_[nid] + ib * stride_up_bytes_;
                memcpy(up_dst, up_src, stride_up_bytes_);
            }
        }

        // Allocate down weights
        down_numa_ptrs_.resize(num_nodes);
        down_numa_sizes_.resize(num_nodes);

        for (int nid = 0; nid < num_nodes; nid++) {
            int n_blocks = down_blocks_[nid].num_blocks;
            down_numa_sizes_[nid] = n_blocks * stride_down_bytes_;
            down_numa_ptrs_[nid] = NumaAllocAligned(down_numa_sizes_[nid], nid);

            if (!down_numa_ptrs_[nid]) {
                ErrorInFastLLM("NumaMlpWeight: Failed to allocate down memory on node " + 
                              std::to_string(nid));
            }

            // Copy weight data
            int start_block = down_blocks_[nid].start_block;
            for (int ib = 0; ib < n_blocks; ib++) {
                int global_block = start_block + ib;
                uint8_t* down_src = down_weight.cpuData + global_block * stride_down_bytes_;
                uint8_t* down_dst = (uint8_t*)down_numa_ptrs_[nid] + ib * stride_down_bytes_;
                memcpy(down_dst, down_src, stride_down_bytes_);
            }
        }

        initialized_ = true;
    }

    void* NumaMlpWeight::GetGateBlock(int block_id) {
        if (!initialized_) return nullptr;

        int numa_node = NumaThreadPool::GetThreadNumaNode();
        if (numa_node < 0 || numa_node >= (int)gate_up_blocks_.size()) {
            return nullptr;
        }

        int local_offset = block_id - gate_up_blocks_[numa_node].start_block;
        if (local_offset < 0 || local_offset >= gate_up_blocks_[numa_node].num_blocks) {
            return nullptr;
        }

        return (uint8_t*)gate_numa_ptrs_[numa_node] + local_offset * stride_gate_bytes_;
    }

    void* NumaMlpWeight::GetUpBlock(int block_id) {
        if (!initialized_) return nullptr;

        int numa_node = NumaThreadPool::GetThreadNumaNode();
        if (numa_node < 0 || numa_node >= (int)gate_up_blocks_.size()) {
            return nullptr;
        }

        int local_offset = block_id - gate_up_blocks_[numa_node].start_block;
        if (local_offset < 0 || local_offset >= gate_up_blocks_[numa_node].num_blocks) {
            return nullptr;
        }

        return (uint8_t*)up_numa_ptrs_[numa_node] + local_offset * stride_up_bytes_;
    }

    void* NumaMlpWeight::GetDownBlock(int block_id) {
        if (!initialized_) return nullptr;

        int numa_node = NumaThreadPool::GetThreadNumaNode();
        if (numa_node < 0 || numa_node >= (int)down_blocks_.size()) {
            return nullptr;
        }

        int local_offset = block_id - down_blocks_[numa_node].start_block;
        if (local_offset < 0 || local_offset >= down_blocks_[numa_node].num_blocks) {
            return nullptr;
        }

        return (uint8_t*)down_numa_ptrs_[numa_node] + local_offset * stride_down_bytes_;
    }

    void NumaMlpWeight::Cleanup() {
        if (!initialized_) {
            return;
        }

        for (size_t i = 0; i < gate_numa_ptrs_.size(); i++) {
            if (gate_numa_ptrs_[i]) {
                NumaFreeAligned(gate_numa_ptrs_[i], gate_numa_sizes_[i]);
            }
            if (up_numa_ptrs_[i]) {
                NumaFreeAligned(up_numa_ptrs_[i], up_numa_sizes_[i]);
            }
            if (down_numa_ptrs_[i]) {
                NumaFreeAligned(down_numa_ptrs_[i], down_numa_sizes_[i]);
            }
        }

        gate_numa_ptrs_.clear();
        up_numa_ptrs_.clear();
        down_numa_ptrs_.clear();
        gate_numa_sizes_.clear();
        up_numa_sizes_.clear();
        down_numa_sizes_.clear();
        gate_up_blocks_.clear();
        down_blocks_.clear();
        initialized_ = false;
    }

}  // namespace fastllm

#endif  // USE_NUMA

