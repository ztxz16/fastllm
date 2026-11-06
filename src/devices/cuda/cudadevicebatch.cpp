//
// Created by huangyuyang on 7/19/23.
//

#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"

#include "fastllm-cuda.cuh"
#ifndef USE_ROCM
#include "devices/cuda/attention/fastllm-attention-backend.h"
#endif
#include "utils.h"

namespace fastllm {
    void CudaSplitBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = intParams.find("output___batch") != intParams.end()
            ? intParams.find("output___batch")->second : -1;
        AssertInFastLLM(part == input.dims[axis],
                        "CudaSplitBatchOp: output count must equal input size on split axis.\n");
        std::vector <int> dims = input.dims;
        dims[axis] = 1;
        for (int i = 0; i < part; i++) {
            outputs[i]->dataType = input.dataType;
            outputs[i]->Resize(dims);
        }
    }

    void CudaSplitBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = intParams.find("output___batch") != intParams.end()
            ? intParams.find("output___batch")->second : -1;
        AssertInFastLLM(part == input.dims[axis],
                        "CudaSplitBatchOp: output count must equal input size on split axis.\n");
        for (int i = 0; i < part; i++) {
            outputs[i]->Allocate();
        }
        FastllmCudaSplitBatch(input, outputs, axis);
    }

    void CudaCatBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **inputs = (Data**)(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = inputs[0]->dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = intParams.find("input___batch")->second;
        std::vector <int> dims = inputs[0]->dims;
        dims[axis] = part;
        output.dataType = inputs[0]->dataType;
        output.Resize(dims);
    }

    void CudaCatBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **inputs = (Data**)(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = inputs[0]->dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = intParams.find("input___batch")->second;
        output.Allocate();
        FastllmCudaCatBatch(inputs, output, axis);
    }

    void CudaMulBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **inputs = (Data**)(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        int batch = intParams.find("input___batch")->second;
        if (outputs[0]->Count(0) > outputs[0]->expansionSize) {
            for (int i = 0; i < batch; i++) {
                outputs[i]->Allocate();
                AssertInFastLLM(inputs[i]->dataType == DataType::FLOAT32,
                                "Mul error: Data's type should be float32.\n");
            }
        }

        FastllmCudaMulBatch(inputs, v, batch, outputs);
    }

    void CudaMatMulBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        int batch = intParams.find("input0___batch")->second;
        Data** input0s = ((Data**)datas.find("input0")->second);
        Data** input1s = ((Data**)datas.find("input1")->second);
        Data** outputs = ((Data**)datas.find("output")->second);

        if (input0s[0]->dims.size() == 3 && input1s[0]->dims.size() == 3) {
            AssertInFastLLM(input0s[0]->dataType == DataType::FLOAT32 && input1s[0]->dataType == DataType::FLOAT32,
                            "MatMul's input's type should be float32.\n");
            AssertInFastLLM(input0s[0]->dims[0] == input1s[0]->dims[0] &&
                            input0s[0]->dims[2] == input1s[0]->dims[1],
                            "MatMul's shape error.\n");
            for (int i = 0; i < batch; i++) {
                outputs[i]->dataType = input0s[i]->dataType;
                outputs[i]->Resize({input0s[i]->dims[0], input0s[i]->dims[1], input1s[i]->dims[2]});
            }
        } else {
            fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CudaMatMulOp());
            DataDict tempDatas = datas;
            for (int i = 0; i < batch; i++) {
                tempDatas["input0"] = ((Data **) datas.find("input0")->second)[i];
                tempDatas["input1"] = ((Data **) datas.find("input1")->second)[i];
                tempDatas["output"] = ((Data **) datas.find("output")->second)[i];
                op->Reshape("MatMulTransB", tempDatas, floatParams, intParams);
            }
            delete op;
        }
    }

    void CudaMatMulBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        int batch = intParams.find("input0___batch")->second;
        Data **input0s = (Data**)(datas.find("input0")->second);
        Data **input1s = (Data**)(datas.find("input1")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : -1;

        std::vector <void*> i0s, i1s, os;
        std::vector <int> ns, ms, ks, i0Strides, i1Strides;
        for (int i = 0; i < batch; i++) {
            auto &input0 = *input0s[i];
            auto &input1 = *input1s[i];
            auto &output = *outputs[i];
            output.Allocate();

            int input0Spatial = input0.Count(input0.dims.size() - 2);
            int input1Spatial = input1.Count(input1.dims.size() - 2);
            int input0Stride = input0.strides[input0.dims.size() - 2];
            int input1Stride = input1.strides[input1.dims.size() - 2];
            int n = input0.dims[input0.dims.size() - 2];
            int m = input0.dims.back();
            int k = input1.dims[input1.dims.size() - 1];
            int batch0 = input0.Count(0) / input0Spatial;
            int batch1 = input1.Count(0) / input1Spatial;
            int outputSpatial = output.Count(output.dims.size() - 2);

            for (int b = 0; b < batch0; b++) {
                i0s.push_back((float*)input0.cudaData + b * input0Spatial);
                i1s.push_back((float*)input1.cudaData + b * input1Spatial);
                os.push_back((float*)output.cudaData + b * outputSpatial);
                ns.push_back(n);
                ms.push_back(m);
                ks.push_back(k);
                i0Strides.push_back(input0Stride);
                i1Strides.push_back(input1Stride);
            }
        }

        FastllmCudaBatchMatMulBatch(i0s.data(), i1s.data(), os.data(),
                                    ns.data(), ms.data(), ks.data(),
                                    i0Strides.data(), i1Strides.data(), alpha, (int)i0s.size());
    }

    void CudaMatMulTransBBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        int batch = intParams.find("input0___batch")->second;
        Data **input0s = ((Data **) datas.find("input0")->second);
        Data **input1s = ((Data **) datas.find("input1")->second);
        Data **outputs = ((Data **) datas.find("output")->second);

        if (input0s[0]->dims.size() == 3 && input1s[0]->dims.size() == 3) {
            AssertInFastLLM(input0s[0]->dataType == DataType::FLOAT32 && input1s[0]->dataType == DataType::FLOAT32,
                            "MatMul's input's type should be float32.\n");
            AssertInFastLLM(input0s[0]->dims[0] == input1s[0]->dims[0] &&
                            input0s[0]->dims[2] == input1s[0]->dims[2],
                            "MatMul's shape error.\n");
            for (int i = 0; i < batch; i++) {
                outputs[i]->dataType = input0s[i]->dataType;
                outputs[i]->Resize({input0s[i]->dims[0], input0s[i]->dims[1], input1s[i]->dims[1]});
            }
        } else {
            fastllm::BaseOperator *op = (fastllm::BaseOperator *) (new CudaMatMulTransBOp());
            DataDict tempDatas = datas;
            for (int i = 0; i < batch; i++) {
                tempDatas["input0"] = ((Data **) datas.find("input0")->second)[i];
                tempDatas["input1"] = ((Data **) datas.find("input1")->second)[i];
                tempDatas["output"] = ((Data **) datas.find("output")->second)[i];
                op->Reshape("MatMulTransB", tempDatas, floatParams, intParams);
            }
            delete op;
        }
    }

    void CudaMatMulTransBBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        int batch = intParams.find("input0___batch")->second;
        Data **input0s = (Data**)(datas.find("input0")->second);
        Data **input1s = (Data**)(datas.find("input1")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : -1;

        std::vector <void*> i0s, i1s, os;
        std::vector <int> ns, ms, ks, i0Strides, i1Strides;
        for (int i = 0; i < batch; i++) {
            auto &input0 = *input0s[i];
            auto &input1 = *input1s[i];
            auto &output = *outputs[i];
            output.Allocate();
            int input0Spatial = input0.Count(input0.dims.size() - 2);
            int input1Spatial = input1.Count(input1.dims.size() - 2);
            int input0Stride = input0.strides[input0.dims.size() - 2];
            int input1Stride = input1.strides[input1.dims.size() - 2];
            int n = input0.dims[input0.dims.size() - 2];
            int m = input0.dims.back();
            int k = input1.dims[input1.dims.size() - 2];
            int batch0 = input0.Count(0) / input0Spatial;
            int batch1 = input1.Count(0) / input1Spatial;

            int outputSpatial = output.Count(output.dims.size() - 2);
            for (int b = 0; b < batch0; b++) {
                i0s.push_back((float*)input0.cudaData + b * input0Spatial);
                i1s.push_back((float*)input1.cudaData + b * input1Spatial);
                os.push_back((float*)output.cudaData + b * outputSpatial);
                ns.push_back(n);
                ms.push_back(m);
                ks.push_back(k);
                i0Strides.push_back(input0Stride);
                i1Strides.push_back(input1Stride);
            }
        }

        FastllmCudaBatchMatMulTransBBatch(i0s.data(), i1s.data(), os.data(),
                                         ns.data(), ms.data(), ks.data(),
                                         i0Strides.data(), i1Strides.data(), alpha, (int)i0s.size());
    }

    void CudaSoftmaxBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int batch = intParams.find("input___batch")->second;
        Data **inputs = (Data**)(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        for (int i = 0; i < batch; i++) {
            outputs[i]->Allocate();
        }
        FastllmCudaSoftmaxBatch(inputs, outputs, axis, batch);
    }

    void DoCudaCatDirectBatch(Data **input0s, Data **input1s, int batch, int axis) {
        std::vector <void*> dsts, srcs;
        std::vector <size_t> dpitchs, spitchs, widths, heights;
        dsts.resize(batch);
        srcs.resize(batch);
        dpitchs.resize(batch);
        spitchs.resize(batch);
        widths.resize(batch);
        heights.resize(batch);

        for (int b = 0; b < batch; b++) {
            Data &input0 = *input0s[b];
            Data &input1 = *input1s[b];

            if (input0.dims.size() == 0) {
                input0.Resize(input1.dims);
                AssertInFastLLM(input0.expansionDims.size() == input1.dims.size() &&
                                input1.dims[axis] <= input0.expansionDims[axis],
                                "CatDirect Error: input0's expansion size is not enough.\n");
                int outer = input1.Count(0) / input1.Count(axis);
                int input0Stride = input0.Count(axis);
                int input1Stride = input1.Count(axis);
                int inner = input0.strides[axis];
                int unitSize = input0.unitSize;

                dsts[b] = (input0.cudaData);
                dpitchs[b] = (input0Stride * unitSize);
                srcs[b] = (input1.cudaData);
                spitchs[b] = (input1Stride * unitSize);
                widths[b] = (input1.dims[axis] * inner * unitSize);
                heights[b] = (outer);
                continue;
            }

            int oldAxisDims = input0.dims[axis];
            input0.dims[axis] += input1.dims[axis];
            int outer = input0.Count(0) / input0.Count(axis);
            int input0Stride = input0.Count(axis);
            int input1Stride = input1.Count(axis);

            int inner = input0.strides[axis];
            int unitSize = input0.unitSize;

            dsts[b] = ((uint8_t *) input0.cudaData + oldAxisDims * inner * unitSize);
            dpitchs[b] = (input0Stride * unitSize);
            srcs[b] = (input1.cudaData);
            spitchs[b] = (input1Stride * unitSize);
            widths[b] = (input1.dims[axis] * inner * unitSize);
            heights[b] = (outer);
        }

        FastllmCudaMemcpy2DDeviceToDeviceBatch(dsts.data(), dpitchs.data(), srcs.data(),
                                               spitchs.data(), widths.data(), heights.data(), dsts.size());
    }

    void CudaCatDirectBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                   const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **input0s = (Data**)(datas.find("input0")->second);
        Data **input1s = (Data**)(datas.find("input1")->second);
        int batch = intParams.find("input0___batch")->second;
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        AssertInFastLLM((input0s[0]->dataType == DataType::FLOAT32 && input1s[0]->dataType == DataType::FLOAT32) ||
                        (input0s[0]->dataType == DataType::FLOAT16 && input1s[0]->dataType == DataType::FLOAT16) ||
                        (input0s[0]->dataType == DataType::BFLOAT16 && input1s[0]->dataType == DataType::BFLOAT16),
                        "Cat's input's type should be float32, float16 or bfloat16.\n");
        AssertInFastLLM(input0s[0]->dataDevice == input1s[0]->dataDevice,
                            "CatDirect error: inputs should use same device.\n");
        AssertInFastLLM(input0s[0]->dims.size() == 0 || input0s[0]->dims.size() == input1s[0]->dims.size(),
                            "Cat Error: input's shape's size should be same.\n");
        int dimsLen = input1s[0]->dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        for (int i = 0; i < dimsLen && i < input0s[0]->dims.size(); i++) {
            if (i != axis) {
                AssertInFastLLM(input0s[0]->dims[i] == input1s[0]->dims[i], "Cat Error: input's shape doesn't match.");
            }
        }

        DoCudaCatDirectBatch(input0s, input1s, batch, axis);
    }

    void CudaAppendKVCacheBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        int batch = intParams.find("caches___batch")->second;
        Data **caches = (Data**)(datas.find("caches")->second);
        Data &input = *(datas.find("input")->second);
        std::vector <void*> dsts, srcs;
        std::vector <size_t> dpitchs, spitchs, widths, heights;

        int heads = input.dims[1], dims = input.dims[2] * input.unitSize;
        for (int i = 0; i < batch; i++) {
            std::vector <int> cacheDims = caches[i]->dims;
            uint8_t *cur = (uint8_t*)input.cudaData + i * heads * dims;

            dsts.push_back((uint8_t *) caches[i]->cudaData + cacheDims[1] * dims);
            dpitchs.push_back(caches[i]->Count(1) * caches[i]->unitSize);
            srcs.push_back(cur);
            spitchs.push_back(dims);
            widths.push_back(dims);
            heights.push_back(heads);

            cacheDims[1]++;
            caches[i]->Resize(cacheDims);
        }
        FastllmCudaMemcpy2DDeviceToDeviceBatch(dsts.data(), dpitchs.data(), srcs.data(),
                                               spitchs.data(), widths.data(), heights.data(), dsts.size());
    }

    void DoCudaAttentionBatchReshape(Data **qs, Data **vs, Data **outputs, int batch) {
        for (int i = 0; i < batch; i++) {
            outputs[i]->dataType = qs[i]->dataType;
            outputs[i]->Resize({qs[i]->dims[0], qs[i]->dims[1], vs[i]->dims[2]});
        }
    }

    void CudaAttentionBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **qs = (Data**)(datas.find("q")->second);
        Data **ks = (Data**)(datas.find("k")->second);
        Data **vs = (Data**)(datas.find("v")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        int batch = intParams.find("q___batch")->second;

        Data &q = *qs[0], &k = *ks[0], &v = *vs[0];
        AssertInFastLLM(q.dims.size() == 3 && k.dims.size() == 3 && v.dims.size() == 3, "Attention: dims of q, k, v should be 3.\n");
        AssertInFastLLM(q.dims[2] == k.dims[2], "Attention: q.dims[2] should be equal to k.dims[2].\n");
        AssertInFastLLM(k.dims[1] == v.dims[1], "Attention: k.dims[1] should be equal to v.dims[1].\n");
        AssertInFastLLM(k.dims[0] == v.dims[0], "Attention: k.dims[0] should be equal to v.dims[0].\n");
        AssertInFastLLM(q.dims[0] == k.dims[0] * group, "Attention: q.dims[0] should be equal to k.dims[0] * group.\n");
        AssertInFastLLM(q.dataType == k.dataType && q.dataType == v.dataType,
                        "Attention: q, k, v's datatype should be same.\n");
        AssertInFastLLM(q.dataType == DataType::FLOAT32 ||
                        q.dataType == DataType::FLOAT16 ||
                        q.dataType == DataType::BFLOAT16,
                    "Attention's input's type should be float32, float16 or bfloat16.\n");
        DoCudaAttentionBatchReshape(qs, vs, outputs, batch);
    }

#ifndef USE_ROCM
    namespace {
        /**
         * 检查原有连续KV Cache实现是否接受当前调用。
         *
         * @param invocation 统一Attention参数；必须携带批量Q/K/V和输出数组。
         * @return 支持状态及拒绝原因。
         */
        CudaAttentionSupportResult SupportsLegacyContiguousAttention(
            const CudaAttentionInvocation &invocation) {
            if (invocation.cacheLayout != CudaAttentionCacheLayout::CONTIGUOUS) {
                return {false, "requires contiguous KV cache"};
            }
            if (invocation.qs == nullptr || invocation.ks == nullptr ||
                invocation.vs == nullptr || invocation.outputs == nullptr ||
                invocation.batch <= 0) {
                return {false, "missing contiguous attention tensors"};
            }
            if (invocation.dataType != DataType::FLOAT32 &&
                invocation.dataType != DataType::FLOAT16 &&
                invocation.dataType != DataType::BFLOAT16) {
                return {false, "requires float32, float16 or bfloat16"};
            }
            return {true, "supported"};
        }

        /**
         * 执行原有连续KV Cache Attention实现。
         *
         * BF16保持历史行为：逐请求转换Q/K/V到FP32，执行FP32 Attention后
         * 再转换输出；FLOAT32和FLOAT16继续调用原批量入口。本Backend作为
         * 重构基线和不兼容实现的显式fallback，不改变同步与数值语义。
         *
         * @param invocation 连续KV Cache Attention参数。
         * @return true表示原有实现完成执行；false表示参数不完整。
         */
        bool RunLegacyContiguousAttention(CudaAttentionInvocation &invocation) {
            Data **qs = invocation.qs;
            Data **ks = invocation.ks;
            Data **vs = invocation.vs;
            Data **masks = invocation.masks;
            Data **outputs = invocation.outputs;
            const int group = invocation.group;
            const float scale = invocation.scale;
            const int batch = invocation.batch;

            if (qs[0]->dataType == DataType::BFLOAT16) {
                for (int i = 0; i < batch; i++) {
                    outputs[i]->Allocate();
                    Data q32, k32, v32, mask32, output32;
                    ToDataType(*qs[i], q32, DataType::FLOAT32);
                    ToDataType(*ks[i], k32, DataType::FLOAT32);
                    ToDataType(*vs[i], v32, DataType::FLOAT32);
                    output32.dataType = DataType::FLOAT32;
                    output32.UpdateUnitSize();
                    output32.Resize(outputs[i]->dims);
                    output32.ToDevice(outputs[i]->dataDevice, false);
                    output32.Allocate();
                    if (masks != nullptr && masks[i] != nullptr && !masks[i]->dims.empty()) {
                        ToDataType(*masks[i], mask32, DataType::FLOAT32);
                    }
                    FastllmCudaAttention(q32, k32, v32, mask32, output32,
                                         group, scale, 0);
                    ToDataType(output32, *outputs[i], DataType::BFLOAT16);
                }
                return true;
            }

            for (int i = 0; i < batch; i++) {
                outputs[i]->Allocate();
            }
            FastllmCudaAttentionBatch(qs, ks, vs, masks, outputs, group, scale, batch);
            return true;
        }
    }

    /** 注册保持原数值和同步语义的连续KV Cache基线Backend。 */
    void RegisterCudaContiguousAttentionBackends() {
        static const bool registered = RegisterCudaAttentionBackend({
            "legacy_contiguous", 10,
            SupportsLegacyContiguousAttention,
            RunLegacyContiguousAttention,
        });
        AssertInFastLLM(registered,
                        "Failed to register legacy_contiguous attention backend.\n");
    }

    void DoCudaAttentionBatch(Data **qs, Data **ks, Data **vs, Data **masks,
                              Data **outputs, int group, float scale, int batch) {
        CudaAttentionInvocation invocation;
        invocation.cacheLayout = CudaAttentionCacheLayout::CONTIGUOUS;
        invocation.phase = qs[0]->dims[1] == 1
            ? CudaAttentionPhase::DECODE : CudaAttentionPhase::PREFILL;
        invocation.dataType = qs[0]->dataType;
        invocation.deviceId = FastllmCudaGetDevice();
        invocation.batch = batch;
        invocation.group = group;
        invocation.headDim = qs[0]->dims[2];
        invocation.scale = scale;
        invocation.qs = qs;
        invocation.ks = ks;
        invocation.vs = vs;
        invocation.masks = masks;
        invocation.outputs = outputs;

        std::string actualBackend;
        std::string lastError;
        if (!RunCudaAttentionBackend(invocation, actualBackend, lastError)) {
            ErrorInFastLLM("AttentionBatch backend selection failed: " + lastError + "\n");
        }
    }
#else
    void DoCudaAttentionBatch(Data **qs, Data **ks, Data **vs, Data **masks,
                              Data **outputs, int group, float scale, int batch) {
        if (qs[0]->dataType == DataType::BFLOAT16) {
            for (int i = 0; i < batch; i++) {
                outputs[i]->Allocate();
                Data q32, k32, v32, mask32, output32;
                ToDataType(*qs[i], q32, DataType::FLOAT32);
                ToDataType(*ks[i], k32, DataType::FLOAT32);
                ToDataType(*vs[i], v32, DataType::FLOAT32);
                output32.dataType = DataType::FLOAT32;
                output32.UpdateUnitSize();
                output32.Resize(outputs[i]->dims);
                output32.ToDevice(outputs[i]->dataDevice, false);
                output32.Allocate();
                if (masks != nullptr && masks[i] != nullptr && !masks[i]->dims.empty()) {
                    ToDataType(*masks[i], mask32, DataType::FLOAT32);
                }
                FastllmCudaAttention(q32, k32, v32, mask32, output32,
                                     group, scale, 0);
                ToDataType(output32, *outputs[i], DataType::BFLOAT16);
            }
            return;
        }
        for (int i = 0; i < batch; i++) {
            outputs[i]->Allocate();
        }
        FastllmCudaAttentionBatch(qs, ks, vs, masks, outputs, group, scale, batch);
    }
#endif

    void CudaAttentionBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        int batch = intParams.find("q___batch")->second;
        Data **qs = (Data**)(datas.find("q")->second);
        Data **ks = (Data**)(datas.find("k")->second);
        Data **vs = (Data**)(datas.find("v")->second);
        Data **masks = (Data**)(datas.find("mask")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        int group = intParams.find("group")->second;
        float scale = floatParams.find("scale")->second;

        DoCudaAttentionBatch(qs, ks, vs, masks, outputs, group, scale, batch);
    }
}
