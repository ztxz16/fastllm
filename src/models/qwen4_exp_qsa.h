#pragma once
#include "qwen4_exp.h"
namespace fastllm {
    struct Qwen4QsaMirrorCompletion;
    void Qwen4AttentionInput(const Data &input, Data &output, DataType type);
    struct Qwen4ExpModel::QsaHostMirrorTransfer {
#ifdef USE_CUDA
        ~QsaHostMirrorTransfer();
        // Graph outputs stay alive through this scope. Fuse their small host
        // appends into batched mapped writes and one completion per device.
        struct Batch {
            struct Device {
                std::vector<void *> destinations;
                std::vector<const void *> sources;
                std::vector<size_t> sizes;
                std::set<Qwen4QsaMirrorCompletion *> waited;
                std::shared_ptr<Qwen4QsaMirrorCompletion> completion;
            };
            std::map<int, Device> devices;
            ~Batch();
            void Flush();
        };

        void Prepare(const std::vector<float> &rawKeys, const std::vector<float> &positions,
                     int expectedRows, int rowWidth);

        void ReserveForPrefill(int promptRows, int outputLimit, int rowWidth);

        void Materialize(std::vector<float> &rawKeys, std::vector<float> &positions);

        void MarkDeviceSynchronized();

        void Rollback(int rows);

        void Queue(const Data &rawKeys, const Data &positions, int previousRows, int rowCount,
                   int rowWidth, Batch *batch = nullptr);

      private:
        friend struct Qwen4PrefixCacheTestAccess;

        void Synchronize();

        void EnsureCapacity(int requiredRows, int rowWidth);

        void Release();

        float *rawStorage = nullptr;
        float *positionStorage = nullptr;
        float *mappedRaw = nullptr, *mappedPositions = nullptr;
        int rowCapacity = 0;
        int committedRows = 0;
        int headDim = 0;
        int device = -1;
        bool pending = false;
        std::shared_ptr<Qwen4QsaMirrorCompletion> completion;
#endif
    };
} // namespace fastllm
