#pragma once
#include "qwen4_exp.h"
namespace fastllm {
    struct Qwen4ExpModel::QsaHostMirrorTransfer {
#ifdef USE_CUDA
        ~QsaHostMirrorTransfer();

        void Prepare(const std::vector<float> &rawKeys, const std::vector<float> &positions,
                     int expectedRows, int rowWidth);

        void Reserve(int rows, int rowWidth);

        void Materialize(std::vector<float> &rawKeys, std::vector<float> &positions);

        void MarkDeviceSynchronized();

        void Rollback(int rows);

        void Queue(const Data &rawKeys, const Data &positions, int previousRows, int rowCount,
                   int rowWidth);

      private:
        void Synchronize();

        void EnsureCapacity(int requiredRows, int rowWidth);

        void Release();

        float *rawStorage = nullptr;
        float *positionStorage = nullptr;
        int rowCapacity = 0;
        int committedRows = 0;
        int headDim = 0;
        int device = -1;
        bool pending = false;
#endif
    };
} // namespace fastllm
