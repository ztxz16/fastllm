//
// Created by huangyuyang on 8/2/24.
//

#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/multicudadevice.h"

#include "fastllm-multicuda.cuh"

#include "utils.h"

namespace fastllm {
    MultiCudaDevice::MultiCudaDevice(CudaDevice *cudaDevice) {
        this->cudaDevice = cudaDevice;
        this->deviceType = "multicuda";

        this->ops["MLP"] = (BaseOperator*)(new MultiCudaMLPOp());
        this->ops["Linear"] = (BaseOperator*)(new MultiCudaLinearOp());
        this->ops["MergeMOE"] = (BaseOperator*)(new MultiCudaMergeMOE());
        this->ops["MergeAttention"] = (BaseOperator*)(new MultiCudaMergeAttention());
    }

    bool MultiCudaDevice::Malloc(void **ret, size_t size) {
        *ret = FastllmCudaMalloc(size);
        return true;
    }

    bool MultiCudaDevice::Free(void *ret) {
        FastllmCudaFree(ret);
        return true;
    }

    bool MultiCudaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        FastllmCudaCopyFromHostToDevice(dst, src, size);
        return true;
    }

    bool MultiCudaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        FastllmCudaCopyFromDeviceToHost(dst, src, size);
        return true;
    }

    bool MultiCudaDevice::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            if (((BaseDevice*)this->cudaDevice)->ops.find(opType) == ((BaseDevice*)this->cudaDevice)->ops.end()) {
                return false;
            } else {
                return ((BaseDevice*)this->cudaDevice)->CanRun(opType, datas, floatParams, intParams);
            }
        } else {
            return this->ops[opType]->CanRun(opType, datas, floatParams, intParams);
        }
    }

    // 对某一个算子进行形状推理
    void MultiCudaDevice::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            ((BaseDevice*)this->cudaDevice)->Reshape(opType, datas, floatParams, intParams);
        } else {
            this->ops[opType]->Reshape(opType, datas, floatParams, intParams);
        }
    }

    // 对某一个算子进行推理
    void MultiCudaDevice::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            ((BaseDevice*)this->cudaDevice)->Run(opType, datas, floatParams, intParams);
        } else {
            this->ops[opType]->Run(opType, datas, floatParams, intParams);
        }
    }

    struct MultiCudaDoMergeMLPOp : MultiThreadBaseOp {
        uint8_t *oriCudaInput, *oriCpuInput, *partOutput;
        Data *input, *weight0, *bias0, *weight1, *bias1;
        Data *w1, *w2, *w3;
        Data *output;
        int deviceId;

        MultiCudaDoMergeMLPOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput, uint8_t *partOutput,
                            Data *input, Data *weight0, Data *bias0, Data *weight1, Data *bias1, 
                            Data *w1, Data *w2, Data *w3,
                            Data *output, int deviceId) : 
                oriCudaInput(oriCudaInput), oriCpuInput(oriCpuInput), partOutput(partOutput),
                input(input), weight0(weight0), bias0(bias0), weight1(weight1), bias1(bias1), 
                w1(w1), w2(w2), w3(w3), 
                output(output), deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            if (deviceId == 0) {
                input->cudaData = oriCudaInput;
            } else {
                input->Allocate();
                FastllmCudaCopyFromHostToDevice(input->cudaData, oriCpuInput, input->GetBytes());
            }

            int bsz = input->dims[0], seqlen = input->dims[1];            

            DoCudaLinearReshape(*input, *weight0, *w3);
            DoCudaLinear(*input, *weight0, bias0 == nullptr ? Data() : *bias0, *w3);

            DoCudaSwigluReshape(*w3, *w1);
            DoCudaSwiglu(*w3, *w1);

            DoCudaLinearReshape(*w1, *weight1, *output);
            if (deviceId == 0) {
                output->isFake = true;
                output->UpdateUnitSize();
                output->cudaData = partOutput;
                output->expansionSize = output->Count(0);
                output->expansionBytes = (output->Count(0) * output->unitSize - 1) / output->unitSizeDiv + 1;
            }
            DoCudaLinear(*w1, *weight1, bias1 == nullptr ? Data() : *bias1, *output);
            if (deviceId != 0) {
                FastllmCudaCopyFromDeviceToDevice(partOutput, output->cudaData, output->GetBytes());
            }
        }
    };

    struct MultiCudaCpuDoMergeMLPOp : MultiThreadBaseOp {
        uint8_t *oriCpuInput, *partOutput;
        Data *input, *weight0, *bias0, *weight1, *bias1;
        Data *w1, *w2, *w3;
        Data *output;
        int deviceId;

        MultiCudaCpuDoMergeMLPOp(uint8_t *oriCpuInput, uint8_t *partOutput,
                            Data *input, Data *weight0, Data *bias0, Data *weight1, Data *bias1, 
                            Data *w1, Data *w2, Data *w3,
                            Data *output, int deviceId) : 
                oriCpuInput(oriCpuInput), partOutput(partOutput),
                input(input), weight0(weight0), bias0(bias0), weight1(weight1), bias1(bias1), 
                w1(w1), w2(w2), w3(w3), 
                output(output), deviceId(deviceId) {}

        void Run() {
            input->Allocate();
            memcpy(input->cpuData, oriCpuInput, input->GetBytes());

            DoCpuLinearReshape(*input, *weight0, *w3);
            DoCpuLinear(*input, *weight0, bias0 == nullptr ? Data() : *bias0, *w3);

            DoCpuSwigluReshape(*w3, *w1);
            DoCpuSwiglu(*w3, *w1);

            DoCpuLinearReshape(*w1, *weight1, *output);
            DoCpuLinear(*w1, *weight1, bias1 == nullptr ? Data() : *bias1, *output);

            FastllmCudaCopyFromHostToDevice(partOutput, output->cpuData, output->GetBytes());
        }
    };

    void MultiCudaMLPOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight0 = *(datas.find("weight0")->second);
        Data &weight1 = *(datas.find("weight1")->second);

        AssertInFastLLM(weight0.dims.size() == 2 && weight1.dims.size() == 2, "MLP's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight0.dims[1], "MLP's weight's shape error.\n");
        AssertInFastLLM(weight0.dims[0] / 2 == weight1.dims[1], "MLP's weight's shape error.\n");
        AssertInFastLLM(weight0.dataType == weight1.dataType, "MLP's weight's data type error.\n");

        weight0.weightType = WeightType::LINEAR;
        weight1.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight1.dims[0];

        output.dataType = input.dataType;
        output.Resize(dims);
    }
    
    void DeviceGetInfos(int deviceId, std::string &specialId, int &mallocType) {
        static std::map <int, std::string> specialDeviceIds = {
            {99999, "cpu"}
        };
        specialId = "";
        if (specialDeviceIds.find(deviceId) != specialDeviceIds.end()) {
            specialId = specialDeviceIds[deviceId];
        }
        mallocType = 1;
        if (specialId == "cpu") {
            mallocType = 0;
        }
    }

    void MultiCudaMLPOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight0 = *(datas.find("weight0")->second);
        Data &bias0 = *(datas.find("bias0")->second);
        Data &weight1 = *(datas.find("weight1")->second);
        Data &bias1 = *(datas.find("bias1")->second);

        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);

        output.Allocate();
// auto st = std::chrono::system_clock::now();
        int mid = weight0.dims[0] / 2;
        int unit = weight0.groupCnt <= 0 ? 128 : weight0.groupCnt;
        if (weight0.dataType == fastllm::DataType::FP8_E4M3) {
            unit = weight0.blockM;
        }
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, false);
        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, mid, unit);

        DivisionScheme divisionScheme, divisionSchemeO;
        for (int i = 0; i < devices.size(); i++) {
            int st = points[i], end = points[i + 1];
            int deviceId = devices[i];

            divisionScheme[deviceId].push_back(std::make_pair(st, end));
            divisionScheme[deviceId].push_back(std::make_pair(mid + st, mid + end));

            divisionSchemeO[deviceId].push_back(std::make_pair(st, end));
        }
        SplitMultiCudaWeight(weight0, bias0, devices, divisionScheme, 0);
        SplitMultiCudaWeight(weight1, bias1, devices, divisionSchemeO, 1);
        CopyToMultiDevices(w1, devices, false);
        CopyToMultiDevices(w2, devices, false);
        CopyToMultiDevices(w3, devices, false);
        
        Data curOutput;
        CopyToMultiDevices(input, devices, false);
        curOutput.dataDevice = input.dataDevice;
        CopyToMultiDevices(curOutput, devices, false);
        std::vector <uint8_t> cpuInput;
        cpuInput.resize(input.GetBytes());
        FastllmCudaSetDevice(0);
        FastllmCudaCopyFromDeviceToHost(cpuInput.data(), input.cudaData, input.GetBytes());
        uint8_t *partOutput = (uint8_t*)FastllmCudaMalloc(output.GetBytes() * devices.size());
        
        // Launch cuda op
        auto *pool = fastllm::GetAlivePool();

        std::vector<fastllm::MultiThreadBaseOp*> ops;
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            std::string specialId = "";
            int mallocType;
            DeviceGetInfos(device, specialId, mallocType);

            if (specialId != "cpu") {
                ops.push_back(new MultiCudaDoMergeMLPOp (
                    (uint8_t*)input.cudaData, (uint8_t*)cpuInput.data(), partOutput + output.GetBytes() * i,
                    input.multiDeviceDatas[device], 
                    weight0.multiDeviceDatas[device], bias0.multiDeviceDatas[device], 
                    weight1.multiDeviceDatas[device], bias1.multiDeviceDatas[device], 
                    w1.multiDeviceDatas[device], w2.multiDeviceDatas[device], w3.multiDeviceDatas[device],
                    curOutput.multiDeviceDatas[device], device));
            }
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }

        // run cpu op
        auto temp = pool->curActivateThreadInterval;
        pool->curActivateThreadInterval = std::make_pair(ops.size(), pool->threads.size());
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            std::string specialId = "";
            int mallocType;
            DeviceGetInfos(device, specialId, mallocType);

            if (specialId == "cpu") {
                MultiCudaCpuDoMergeMLPOp (
                    (uint8_t*)cpuInput.data(), partOutput + output.GetBytes() * i,
                    input.multiDeviceDatas[device], 
                    weight0.multiDeviceDatas[device], bias0.multiDeviceDatas[device], 
                    weight1.multiDeviceDatas[device], bias1.multiDeviceDatas[device], 
                    w1.multiDeviceDatas[device], w2.multiDeviceDatas[device], w3.multiDeviceDatas[device],
                    curOutput.multiDeviceDatas[device], device).Run();
            }
        }
        pool->curActivateThreadInterval = temp;

        // wait cuda op
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
// printf("calc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        FastllmReduce((uint8_t*)output.cudaData, partOutput, output.Count(0), devices.size(), output.dataType);
        FastllmCudaFree(partOutput);
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
    }

    struct MultiCudaDoLinearOp : MultiThreadBaseOp {
        uint8_t *oriCudaInput, *oriCpuInput;
        Data *input, *weight, *bias;
        Data *output;
        int n, m, k, start, len;
        uint8_t *lastOutput;
        int deviceId;

        MultiCudaDoLinearOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput,
                            Data *input, Data *weight, Data *bias, Data *output, 
                            int n, int m, int k, int start, int len, uint8_t *lastOutput, int deviceId) : 
                oriCudaInput(oriCudaInput), oriCpuInput(oriCpuInput),
                input(input), weight(weight), bias(bias),
                output(output), 
                n(n), m(m), k(k), start(start), len(len), lastOutput(lastOutput),
                deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            if (deviceId == 0) {
                input->cudaData = oriCudaInput;
            } else {
                input->Allocate();
                FastllmCudaCopyFromHostToDevice(input->cudaData, oriCpuInput, input->GetBytes());
            }
            DoCudaLinearReshape(*input, *weight, *output);
            if (deviceId == 0 && n == 1) {
                output->isFake = true;
                output->UpdateUnitSize();
                output->cudaData = lastOutput;
                output->expansionSize = output->Count(0);
                output->expansionBytes = (output->Count(0) * output->unitSize - 1) / output->unitSizeDiv + 1;
            }
            DoCudaLinear(*input, *weight, bias == nullptr ? Data() : *bias, *output);
            if (deviceId != 0 || n > 1) {
                FastllmCudaMemcpy2DDeviceToDeviceAuto(lastOutput + start * output->unitSize, k * output->unitSize, output->cudaData, 
                    len * output->unitSize, len * output->unitSize, n, 0, deviceId);
            }
        }
    };

    bool MultiCudaLinearOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        Data &weight = *(datas.find("weight")->second);
        return weight.dims[0] > 10000 || weight.dims[1] > 10000;
    }

    void MultiCudaLinearOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate();
// auto st = std::chrono::system_clock::now();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        int unit = weight.groupCnt <= 0 ? 128 : weight.groupCnt;
        if (weight.dataType == fastllm::DataType::FP8_E4M3) {
            unit = weight.blockM;
        }
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, weight.dims[0], unit);

        DivisionScheme divisionScheme;
        for (int i = 0; i < devices.size(); i++) {
            int st = points[i], end = points[i + 1];
            int deviceId = devices[i];
            divisionScheme[deviceId].push_back(std::make_pair(st, end));
        }
        SplitMultiCudaWeight(weight, bias, devices, divisionScheme, 0);
        Data curOutput;
        CopyToMultiDevices(input, devices, false);
        curOutput.dataDevice = input.dataDevice;
        CopyToMultiDevices(curOutput, devices, false);
        std::vector <uint8_t> cpuInput;
        cpuInput.resize(input.GetBytes());
        FastllmCudaSetDevice(0);
        FastllmCudaCopyFromDeviceToHost(cpuInput.data(), input.cudaData, input.GetBytes());
        auto *pool = fastllm::GetAlivePool();
        std::vector<fastllm::MultiThreadBaseOp*> ops;
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            int start = points[i], len = points[i + 1] - points[i];
            ops.push_back(new MultiCudaDoLinearOp (
                (uint8_t*)input.cudaData, (uint8_t*)cpuInput.data(),
                input.multiDeviceDatas[device], 
                weight.multiDeviceDatas[device], bias.multiDeviceDatas[device], 
                curOutput.multiDeviceDatas[device], 
                n, m, k, start, len, (uint8_t*)output.cudaData, device));
        }
        for (int i = 0; i < devices.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        // ops[0]->Run();
        for (int i = 0; i < devices.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
// float spend = GetSpan(st, std::chrono::system_clock::now());
// float gops = (float)n * m * k / spend / 1e9;
// printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }
    
    void MultiCudaMergeAttention::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight1 = *(datas.find("weight1")->second);
        Data &output = *(datas.find("output")->second);
        std::vector <int> dims = input.dims;
        dims.back() = weight1.dims[0];
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    struct MultiCudaDoMergeAttentionOp : MultiThreadBaseOp {
        uint8_t *oriCudaInput, *oriCpuInput, *partOutput;
        Data *input, *weight0, *bias0, *weight1, *bias1;
        Data *qkv, *q, *k, *v;
        int qNum, kvNum, headDim, rotDim;
        float attentionScale;
        Data *positionIds, *sinData, *cosData;
        Data **keys, **values, **masks;
        Data *output;
        int batch;
        int deviceId;

        MultiCudaDoMergeAttentionOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput, uint8_t *partOutput,
                            Data *input, Data *weight0, Data *bias0, Data *weight1, Data *bias1, 
                            Data *qkv, Data *q, Data *k, Data *v,
                            int qNum, int kvNum, int headDim, int rotDim, float attentionScale,
                            Data *positionIds, Data *sinData, Data *cosData,
                            Data** keys, Data** values, Data** masks, 
                            Data *output, int batch, int deviceId) : 
                oriCudaInput(oriCudaInput), oriCpuInput(oriCpuInput), partOutput(partOutput),
                input(input), weight0(weight0), bias0(bias0), weight1(weight1), bias1(bias1), 
                qkv(qkv), q(q), k(k), v(v), 
                qNum(qNum), kvNum(kvNum), headDim(headDim), rotDim(rotDim), attentionScale(attentionScale),
                positionIds(positionIds), sinData(sinData), cosData(cosData),
                keys(keys), values(values), masks(masks), 
                output(output), batch(batch), deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            if (deviceId == 0) {
                input->cudaData = oriCudaInput;
            } else {
                input->Allocate();
                FastllmCudaCopyFromHostToDevice(input->cudaData, oriCpuInput, input->GetBytes());
            }

            if (batch > 1) {
                int bsz = batch, seqlen = input->dims[1];
                std::vector <Data*> vKeys, vValues, vMasks;
                std::vector <Data> curKs, curVs, curQs;
                Data curAttenOutput;
                curKs.resize(bsz);
                curVs.resize(bsz);
                curQs.resize(bsz);
                std::vector <Data*> pointersK, pointersV, pointersQ;
                pointersK.resize(bsz);
                pointersV.resize(bsz);
                pointersQ.resize(bsz);
                std::vector <Data*> qs, attns, contexts;
                qs.resize(bsz);
                attns.resize(bsz);
                contexts.resize(bsz);
                std::vector <Data> curContextLayer;
                curContextLayer.resize(bsz);

                vKeys.resize(bsz);
                vValues.resize(bsz);
                vMasks.resize(bsz);
                for (int i = 0; i < bsz; i++) {
                    vKeys[i] = keys[i];
                    vValues[i] = values[i];
                    vMasks[i] = masks[i];
                }
                DoCudaLinearReshape(*input, *weight0, *qkv);
                DoCudaLinear(*input, *weight0, bias0 == nullptr ? Data() : *bias0, *qkv);
                int per = qkv->dims.back() / (qNum / kvNum + 2);
                int qdim = per * (qNum / kvNum);
                DoCudaSplitReshape(*qkv, -1, 0, qdim, *q);
                DoCudaSplitReshape(*qkv, -1, qdim, qdim + per, *k);
                DoCudaSplitReshape(*qkv, -1, qdim + per, qdim + per * 2, *v);
                DoCudaSplit(*qkv, -1, 0, qdim, *q);
                DoCudaSplit(*qkv, -1, qdim, qdim + per, *k);
                DoCudaSplit(*qkv, -1, qdim + per, qdim + per * 2, *v);

                std::vector <int> qkvSize = {1, seqlen, -1, headDim};
                q->Reshape(qkvSize);
                k->Reshape(qkvSize);
                v->Reshape(qkvSize);

                FastllmCudaLlamaRotatePosition2D(*q, *positionIds, *sinData, *cosData, rotDim);
                FastllmCudaLlamaRotatePosition2D(*k, *positionIds, *sinData, *cosData, rotDim);

                int total = 0;
                q->Reshape({-1, q->dims[2], q->dims[3]});
                k->Reshape({-1, k->dims[2], k->dims[3]});
                v->Reshape({-1, v->dims[2], v->dims[3]});

                std::vector <int> qdims = {q->dims[1], 1, q->dims[2]};
                std::vector <uint64_t> qstrides = {(uint64_t)q->dims[2], (uint64_t)q->dims[2], 1};
                std::vector <int> kdims = {k->dims[1], 1, k->dims[2]};
                std::vector <uint64_t> kstrides = {(uint64_t)k->dims[2], (uint64_t)k->dims[2], 1};
                std::vector <int> vdims = {v->dims[1], 1, v->dims[2]};
                std::vector <uint64_t> vstrides = {(uint64_t)v->dims[2], (uint64_t)v->dims[2], 1};
                for (int b = 0; b < bsz; b++) {
                    curQs[b].dims = qdims;
                    curQs[b].strides = qstrides;
                    curQs[b].FakeFrom(*q, b * q->strides[0] * q->unitSize);
                    curKs[b].dims = kdims;
                    curKs[b].strides = kstrides;
                    curKs[b].FakeFrom(*k, b * k->strides[0] * k->unitSize);
                    curVs[b].dims = vdims;
                    curVs[b].strides = vstrides;
                    curVs[b].FakeFrom(*v, b * v->strides[0] * v->unitSize);
                }

                for (int b = 0; b < bsz; b++) {
                    pointersK[b] = (&curKs[b]);
                    pointersV[b] = (&curVs[b]);
                }

                DoCudaCatDirectBatch(vKeys.data(), pointersK.data(), bsz, 1);
                DoCudaCatDirectBatch(vValues.data(), pointersV.data(), bsz, 1);

                int embed_dim = weight1->dims[1];
                Data &attenOutput = *qkv;
                attenOutput.ToDevice(q->dataDevice);
                attenOutput.Resize({1, bsz, embed_dim});
                attenOutput.Allocate();
                for (int b = 0; b < bsz; b++) {
                    qs[b] = (&curQs[b]);
                    curContextLayer[b].FakeFrom(attenOutput, b * embed_dim * attenOutput.unitSize);
                    contexts[b] = (&curContextLayer[b]);
                }

                DoCudaAttentionBatchReshape(qs.data(), vValues.data(), contexts.data(), bsz);
                DoCudaAttentionBatch(qs.data(), vKeys.data(), vValues.data(), vMasks.data(), contexts.data(), 
                                qs[0]->dims[0] / values[0]->dims[0], attentionScale, bsz);                

                DoCudaLinearReshape(*qkv, *weight1, *output);
                if (deviceId == 0) {
                    output->isFake = true;
                    output->UpdateUnitSize();
                    output->cudaData = partOutput;
                    output->expansionSize = output->Count(0);
                    output->expansionBytes = (output->Count(0) * output->unitSize - 1) / output->unitSizeDiv + 1;
                }
                DoCudaLinear(*qkv, *weight1, bias1 == nullptr ? Data() : *bias1, *output);
                if (deviceId != 0) {
                    FastllmCudaCopyFromDeviceToDevice(partOutput, output->cudaData, output->GetBytes());
                }
            } else {
                int bsz = input->dims[0], seqlen = input->dims[1];

                DoCudaLinearReshape(*input, *weight0, *qkv);
                DoCudaLinear(*input, *weight0, bias0 == nullptr ? Data() : *bias0, *qkv);
                int per = qkv->dims.back() / (qNum / kvNum + 2);
                int qdim = per * (qNum / kvNum);
                DoCudaSplitReshape(*qkv, -1, 0, qdim, *q);
                DoCudaSplitReshape(*qkv, -1, qdim, qdim + per, *k);
                DoCudaSplitReshape(*qkv, -1, qdim + per, qdim + per * 2, *v);
                DoCudaSplit(*qkv, -1, 0, qdim, *q);
                DoCudaSplit(*qkv, -1, qdim, qdim + per, *k);
                DoCudaSplit(*qkv, -1, qdim + per, qdim + per * 2, *v);

                std::vector <int> qkvSize = {bsz, seqlen, -1, headDim};
                q->Reshape(qkvSize);
                k->Reshape(qkvSize);
                v->Reshape(qkvSize);

                FastllmCudaLlamaRotatePosition2D(*q, *positionIds, *sinData, *cosData, rotDim);
                FastllmCudaLlamaRotatePosition2D(*k, *positionIds, *sinData, *cosData, rotDim);

                DoCudaPermuteSelf(*q, {0, 2, 1, 3});
                DoCudaPermuteSelf(*k, {0, 2, 1, 3});
                DoCudaPermuteSelf(*v, {0, 2, 1, 3});

                qkvSize = {-1, seqlen, headDim};
                q->Reshape(qkvSize);
                k->Reshape(qkvSize);
                v->Reshape(qkvSize);

                Data &pastKey = *keys[0];
                Data &pastValue = *values[0];

                DoCudaCatDirect(pastKey, *k, 1);
                DoCudaCatDirect(pastValue, *v, 1);
                DoCudaAttentionReshape(*q, pastValue, *qkv);
                DoCudaAttention(*q, pastKey, pastValue, *masks[0], *qkv, q->dims[0] / pastKey.dims[0], attentionScale, 1);
                DoCudaPermuteSelf(*qkv, {1, 0, 2});
                qkv->Reshape({seqlen, bsz, -1});
                DoCudaPermuteSelf(*qkv, {1, 0, 2});
                DoCudaLinearReshape(*qkv, *weight1, *output);

                if (deviceId == 0) {
                    output->isFake = true;
                    output->UpdateUnitSize();
                    output->cudaData = partOutput;
                    output->expansionSize = output->Count(0);
                    output->expansionBytes = (output->Count(0) * output->unitSize - 1) / output->unitSizeDiv + 1;
                }
                DoCudaLinear(*qkv, *weight1, bias1 == nullptr ? Data() : *bias1, *output);
                if (deviceId != 0) {
                    FastllmCudaCopyFromDeviceToDevice(partOutput, output->cudaData, output->GetBytes());
                }
            }
        }
    };

    void MultiCudaMergeAttention::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight0 = *(datas.find("weight0")->second);
        Data &bias0 = *(datas.find("bias0")->second);
        Data &weight1 = *(datas.find("weight1")->second);
        Data &bias1 = *(datas.find("bias1")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sinData")->second);
        Data &cosData = *(datas.find("cosData")->second);
        Data &output = *(datas.find("output")->second);
        Data &qkv = *(datas.find("qkv")->second);
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        int qNum = intParams.find("qNum")->second;
        int kvNum = intParams.find("kvNum")->second;
        int headDim = intParams.find("headDim")->second;
        int rotDim = intParams.find("rotDim")->second;
        float attentionScale = floatParams.find("attentionScale")->second;
        Data **keys = (Data**)(datas.find("keys")->second);
        Data **values = (Data**)(datas.find("values")->second);
        Data **masks = (Data**)(datas.find("masks")->second);

        int batch = intParams.find("keys___batch")->second;
        output.Allocate();
// auto st = std::chrono::system_clock::now();
        int group = qNum / kvNum;
        int vDim = weight1.dims[0] / qNum;
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, kvNum, 1);
        DivisionScheme divisionScheme, divisionSchemeO;
        for (int i = 0; i < devices.size(); i++) {
            int st = points[i], end = points[i + 1];
            int deviceId = devices[i];
            int qgap = qNum * headDim, qkgap = (qNum + kvNum) * headDim;
            divisionScheme[deviceId].push_back(std::make_pair(st * group * headDim, end * group * headDim));
            divisionScheme[deviceId].push_back(std::make_pair(qgap + st * headDim, qgap + end * headDim));
            divisionScheme[deviceId].push_back(std::make_pair(qkgap + st * headDim, qkgap + end * headDim));

            divisionSchemeO[deviceId].push_back(std::make_pair(st * group * vDim, end * group * vDim));
        }
        SplitMultiCudaWeight(weight0, bias0, devices, divisionScheme, 0);
        SplitMultiCudaWeight(weight1, bias1, devices, divisionSchemeO, 1);
        CopyToMultiDevices(qkv, devices, false);
        CopyToMultiDevices(q, devices, false);
        CopyToMultiDevices(k, devices, false);
        CopyToMultiDevices(v, devices, false);
        CopyToMultiDevices(positionIds, devices, true);
        CopyToMultiDevices(sinData, devices, true);
        CopyToMultiDevices(cosData, devices, true);
        for (int i = 0; i < batch; i++) {
            CopyToMultiDevices(*keys[i], devices, true);
            CopyToMultiDevices(*values[i], devices, true);
            if (masks[i] != nullptr) {
                CopyToMultiDevices(*masks[i], devices, true);
            } 
        }
        std::map <int, std::vector <Data*> > curKeys, curValues, curMasks;
        for (int device : devices) {
            for (int i = 0; i < batch; i++) {
                curKeys[device].push_back(keys[i]->multiDeviceDatas[device]);
                curValues[device].push_back(values[i]->multiDeviceDatas[device]);
                curMasks[device].push_back(masks[i] == nullptr ? nullptr : masks[i]->multiDeviceDatas[device]);
            }
        }

        Data &curInput = *(datas.find("curInput")->second);
        Data &curOutput = *(datas.find("curOutput")->second);

        CopyToMultiDevices(input, devices, false);
        curOutput.dataDevice = input.dataDevice;
        CopyToMultiDevices(curOutput, devices, false);
        std::vector <uint8_t> cpuInput;
        cpuInput.resize(input.GetBytes());
        FastllmCudaSetDevice(0);
        FastllmCudaCopyFromDeviceToHost(cpuInput.data(), input.cudaData, input.GetBytes());
        uint8_t *partOutput = (uint8_t*)FastllmCudaMalloc(output.GetBytes() * devices.size());
        auto *pool = fastllm::GetAlivePool();
        std::vector<fastllm::MultiThreadBaseOp*> ops;

        for (int i = 0; i < devices.size(); i++) {
            int device = devices[i];
            FastllmCudaSetDevice(device);
            int bsz = batch, seqlen = input.dims[1];
            if (bsz > 1) {
                seqlen = 1;
            }
            
            int unitLen = 128;
            for (int i = 0; i < bsz; i++) {
                Data &pastKey = *keys[i]->multiDeviceDatas[device];
                Data &pastValue = *values[i]->multiDeviceDatas[device];
                while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || seqlen > pastKey.expansionDims[1]))
                    || (pastKey.dims.size() > 0 && pastKey.dims[1] + seqlen > pastKey.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector <int> {kvNum, ((seqlen - 1) / unitLen + 1) * unitLen, headDim};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((seqlen - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || seqlen > pastValue.expansionDims[1]))
                    || (pastValue.dims.size() > 0 && pastValue.dims[1] + seqlen > pastValue.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector <int> {kvNum, ((seqlen - 1) / unitLen + 1) * unitLen, headDim};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((seqlen - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
            }
        }
        FastllmCudaSetDevice(0);
        
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            ops.push_back(new MultiCudaDoMergeAttentionOp (
                (uint8_t*)input.cudaData, (uint8_t*)cpuInput.data(), partOutput + output.GetBytes() * i,
                input.multiDeviceDatas[device], 
                weight0.multiDeviceDatas[device], bias0.multiDeviceDatas[device], 
                weight1.multiDeviceDatas[device], bias1.multiDeviceDatas[device], 
                qkv.multiDeviceDatas[device], q.multiDeviceDatas[device], k.multiDeviceDatas[device], v.multiDeviceDatas[device], 
                qNum, kvNum, headDim, rotDim, attentionScale, 
                positionIds.multiDeviceDatas[device], sinData.multiDeviceDatas[device], cosData.multiDeviceDatas[device], 
                curKeys[device].data(), curValues[device].data(), curMasks[device].data(), 
                curOutput.multiDeviceDatas[device], batch, device));
        }
        for (int i = 0; i < devices.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        // ops[0]->Run();
        for (int i = 0; i < devices.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
// printf("calc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        FastllmReduce((uint8_t*)output.cudaData, partOutput, output.Count(0), devices.size(), output.dataType);
        FastllmCudaFree(partOutput);
// printf("FastllmReduce spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        for (int i = 0; i < batch; i++) {
            keys[i]->dims = keys[i]->multiDeviceDatas[devices[0]]->dims;
            keys[i]->expansionDims = keys[i]->multiDeviceDatas[devices[0]]->expansionDims;
            values[i]->dims = values[i]->multiDeviceDatas[devices[0]]->dims;
            values[i]->expansionDims = values[i]->multiDeviceDatas[devices[0]]->expansionDims;
        }
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
    }

    struct MultiCudaDoMergeMOEOp : MultiThreadBaseOp {
        uint8_t *oriCudaInput, *oriCpuInput, *partOutput;
        Data *input;
        Data **weights;
        Data *logits, *gateBias;
        Data *w1, *w2, *w3;
        int wBatch, topk, needNorm;
        float routeScale, sharedScale;
        Data *output;
        int deviceId;

        MultiCudaDoMergeMOEOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput, uint8_t *partOutput, 
                Data *input, Data **weights, Data *logits, Data *gateBias, 
                Data *w1, Data *w2, Data *w3, 
                int wBatch, int topk, int needNorm, float routeScale, float sharedScale,
                Data *output, int deviceId) : 
                oriCudaInput(oriCudaInput), oriCpuInput(oriCpuInput), partOutput(partOutput),
                input(input), weights(weights), logits(logits), gateBias(gateBias), 
                w1(w1), w2(w2), w3(w3),
                wBatch(wBatch), topk(topk), needNorm(needNorm), routeScale(routeScale), sharedScale(sharedScale),
                output(output), deviceId(deviceId) {}

        void Run() {
            FastllmCudaSetDevice(deviceId);
            if (deviceId == 0) {
                input->cudaData = oriCudaInput;
            } else {
                input->Allocate();
                FastllmCudaCopyFromHostToDevice(input->cudaData, oriCpuInput, input->GetBytes());
            }            
            if (deviceId == 0) {
                output->isFake = true;
                output->UpdateUnitSize();
                output->cudaData = partOutput;
                output->expansionSize = output->Count(0);
                output->expansionBytes = (output->Count(0) * output->unitSize - 1) / output->unitSizeDiv + 1;
            }
            
            std::vector <Data*> curWeights;
            curWeights.resize(wBatch);
            for (int i = 0; i < wBatch; i++) {
                if (weights[i] == nullptr) {
                    curWeights[i] = nullptr;
                } else {
                    curWeights[i] = weights[i]->multiDeviceDatas[deviceId];
                }
            }

            output->Resize(input->dims);
            DoCudaMergeMOE(*input, *output, *gateBias, *logits, *w1, *w2, *w3, curWeights.data(), nullptr, topk, needNorm, sharedScale, routeScale);

            if (deviceId != 0) {
                FastllmCudaCopyFromDeviceToDevice(partOutput, output->cudaData, output->GetBytes());
            }
        }
    };

    struct MultiCudaCpuDoMergeMOEOp : MultiThreadBaseOp {
        uint8_t *oriCpuInput, *partOutput;
        Data *input;
        Data **weights;
        Data *logits, *gateBias;
        Data *w1, *w2, *w3;
        int wBatch, topk, needNorm;
        float routeScale, sharedScale;
        Data *output;
        int deviceId;

        MultiCudaCpuDoMergeMOEOp(uint8_t *oriCpuInput, uint8_t *partOutput, 
                Data *input, Data **weights, Data *logits, Data *gateBias, 
                Data *w1, Data *w2, Data *w3, 
                int wBatch, int topk, int needNorm, float routeScale, float sharedScale,
                Data *output, int deviceId) : 
                oriCpuInput(oriCpuInput), partOutput(partOutput),
                input(input), weights(weights), logits(logits), gateBias(gateBias), 
                w1(w1), w2(w2), w3(w3),
                wBatch(wBatch), topk(topk), needNorm(needNorm), routeScale(routeScale), sharedScale(sharedScale),
                output(output), deviceId(deviceId) {}

        void Run() {
            // 注意weights里面的值，真正要使用的是weights[x]->multiDeviceDatas[deviceId]
            input->Allocate();
            memcpy(input->cpuData, oriCpuInput, input->GetBytes());

            int batch = input->dims[0];
            Data &bias = *gateBias;                  
            float *cpuRouterLogits = (float*)logits->cpuData;
            int m = logits->dims.back();

            if (batch == 1) {
                float *cur = cpuRouterLogits;
                std::vector <std::pair <float, int> > oriV; // (value, idx)
                for (int i = 0; i < m; i++) {
                    oriV.push_back(std::make_pair(-cur[i], i));
                }
                if (bias.dims.size() > 0) {
                    float *cpuBias = (float*)bias.cpuData;
                    for (int i = 0; i < m; i++) {
                        oriV[i].first -= cpuBias[i];
                    }
                }
                // sort(oriV.begin(), oriV.end());
                std::partial_sort(oriV.begin(), oriV.begin() + topk, oriV.end());
                float sum = 0.0;
                for (int j = 0; j < topk; j++) {
                    float value = cur[oriV[j].second];
                    sum += value;
                }
                if (!needNorm) {
                    sum = 1.0;
                }
                std::vector <std::pair <int, float> > v;
                v.resize(topk + 1);
                for (int j = 0; j < topk; j++) {
                    v[j] = std::make_pair(oriV[j].second + 1, cur[oriV[j].second] / sum * routeScale);
                }
                v.back() = (std::make_pair(0, sharedScale));
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    float value = v[j].second;
                    if (weights[idx * 2] == nullptr) {
                        continue;
                    }

                    DoCpuLinearReshape(*input, *weights[idx * 2]->multiDeviceDatas[deviceId], *w3);
                    DoCpuLinear(*input, *weights[idx * 2]->multiDeviceDatas[deviceId], Data(), *w3);
                    
                    DoCpuSwigluReshape(*w3, *w1);
                    DoCpuSwiglu(*w3, *w1);

                    DoCpuLinearReshape(*w1, *weights[idx * 2 + 1]->multiDeviceDatas[deviceId], *w2);
                    DoCpuLinear(*w1, *weights[idx * 2 + 1]->multiDeviceDatas[deviceId], Data(), *w2);

                    if (j == 0) {
                        output->Resize(w2->dims);
                        output->Allocate();
                        for (int i = 0; i < output->Count(0); i++) {
                            ((float*)output->cpuData)[i] = ((float*)w2->cpuData)[i] * value;
                        }
                    } else {
                        for (int i = 0; i < output->Count(0); i++) {
                            ((float*)output->cpuData)[i] += ((float*)w2->cpuData)[i] * value;
                        }
                    }
                }
            } else {
                Data attenPart, moePart;
                Data moeFinal = Data();
                moeFinal.Resize({0, input->dims[1]});
                moeFinal.Expansion(input->dims);
                attenPart.ToDevice(input->dataDevice);
                moePart.ToDevice(input->dataDevice);
                moeFinal.ToDevice(input->dataDevice);
                
                for (int b = 0; b < batch; b++) {
                    float *cur = cpuRouterLogits + b * m;
                    std::vector <std::pair <float, int> > oriV; // (value, idx)
                    for (int i = 0; i < m; i++) {
                        oriV.push_back(std::make_pair(-cur[i], i));
                    }
                    if (bias.dims.size() > 0) {
                        float *cpuBias = (float*)bias.cpuData;
                        for (int i = 0; i < m; i++) {
                            oriV[i].first -= cpuBias[i];
                        }
                    }

                    sort(oriV.begin(), oriV.end());
                    Data *currentData = input;
                    if (batch != 1) {
                        attenPart.Resize({1, input->dims.back()});
                        attenPart.dataType = input->dataType;
                        attenPart.Allocate();
                        memcpy (
                            attenPart.cpuData,
                            input->cpuData + b * input->dims.back() * input->unitSize,
                            attenPart.GetBytes()
                        );

                        currentData = &attenPart;
                    }
                        
                    moePart.Resize(currentData->dims);
                    moePart.Allocate(0.0f);

                    float sum = 0.0;
                    for (int j = 0; j < topk; j++) {
                        float value = cur[oriV[j].second];
                        sum += value;
                    }
                    if (!needNorm) {
                        sum = 1.0;
                    }

                    std::vector <std::pair <int, float> > v;
                    for (int j = 0; j < topk; j++) {
                        v.push_back(std::make_pair(oriV[j].second + 1, cur[oriV[j].second] / sum * routeScale));
                    }
                    v.push_back(std::make_pair(0, sharedScale));

                    for (int j = 0; j < v.size(); j++) {
                        int idx = v[j].first;
                        float value = v[j].second;
                        if (weights[idx * 2] == nullptr) {
                            continue;
                        }

                        DoCpuLinearReshape(*currentData, *weights[idx * 2]->multiDeviceDatas[deviceId], *w3);
                        DoCpuLinear(*currentData, *weights[idx * 2]->multiDeviceDatas[deviceId], Data(), *w3);
                        
                        DoCpuSwigluReshape(*w3, *w1);
                        DoCpuSwiglu(*w3, *w1);

                        DoCpuLinearReshape(*w1, *weights[idx * 2 + 1]->multiDeviceDatas[deviceId], *w2);
                        DoCpuLinear(*w1, *weights[idx * 2 + 1]->multiDeviceDatas[deviceId], Data(), *w2);

                        for (int i = 0; i < moePart.Count(0); i++) {
                            ((float*)moePart.cpuData)[i] += ((float*)w2->cpuData)[i] * value;
                        }
                    }

                    DoCpuCatDirect(moeFinal, moePart, 0);
                    moeFinal.expansionDims.clear();

                    output->Resize(moeFinal.dims);
                    output->Allocate();
                    memcpy(output->cpuData, moeFinal.cpuData, moeFinal.GetBytes());
                }
            }

            FastllmCudaCopyFromHostToDevice(partOutput, output->cpuData, output->GetBytes());
        }
    };

    void MultiCudaMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &gateBias = *(datas.find("gateBias")->second);
        Data &logits = *(datas.find("logits")->second);
        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);
        Data **weights = (Data**)(datas.find("weights")->second);
        Data **biass = (Data**)(datas.find("biass")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        int needNorm = intParams.find("needNorm") != intParams.end() ? intParams.find("needNorm")->second : 0;
        float sharedScale = floatParams.find("sharedScale") != floatParams.end() ? floatParams.find("sharedScale")->second : 1.0f;        
        float routeScale = floatParams.find("routeScale") != floatParams.end() ? floatParams.find("routeScale")->second : 1.0f;        
        output.Allocate();
        std::vector <int> devices;
        std::map <int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, false);
        int wBatch = intParams.find("weights___batch") != intParams.end() ? intParams.find("weights___batch")->second : (topk + 1) * 2;
        if (!weights[2]->multiDeviceData) {
            // 这里需要保证已经warmup过了，如果weights[2]切好就代表所有weight都已经切好了
            Data empty;
            for (int i = 0; i < wBatch; i += 2) {
                if (weights[i] == nullptr) {
                    continue;
                }
                int k = weights[i]->dims[0];
                std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, k / 2, weights[i]->groupCnt <= 0 ? 128 : weights[i]->groupCnt);
                DivisionScheme divisionScheme, divisionSchemeO;
                int mid = weights[i]->dims[0] / 2;
                for (int i = 0; i < devices.size(); i++) {
                    int st = points[i], end = points[i + 1];
                    int deviceId = devices[i];
                    divisionScheme[deviceId].push_back(std::make_pair(st, end));

                    divisionSchemeO[deviceId].push_back(std::make_pair(st, end));
                    divisionSchemeO[deviceId].push_back(std::make_pair(mid + st, mid + end));
                }
                SplitMultiCudaWeight(*weights[i], empty, devices, divisionSchemeO, 0);
                SplitMultiCudaWeight(*weights[i + 1], empty, devices, divisionScheme, 1);
            }
        }

        ToDataType(logits, DataType::FLOAT32);
        logits.ToDevice(DataDevice::CPU);
        if (gateBias.dims.size() > 0) {
            ToDataType(gateBias, DataType::FLOAT32);
            gateBias.ToDevice(DataDevice::CPU);
        }
        
        CopyToMultiDevices(w1, devices, false);
        CopyToMultiDevices(w2, devices, false);
        CopyToMultiDevices(w3, devices, false);

        Data &curInput = *(datas.find("curInput")->second);
        Data &curOutput = *(datas.find("curOutput")->second);

        CopyToMultiDevices(input, devices, false);
        curOutput.dataDevice = input.dataDevice;
        CopyToMultiDevices(curOutput, devices, false);
        std::vector <uint8_t> cpuInput;
        cpuInput.resize(input.GetBytes());
        FastllmCudaSetDevice(0);
        FastllmCudaCopyFromDeviceToHost(cpuInput.data(), input.cudaData, input.GetBytes());
        uint8_t *partOutput = (uint8_t*)FastllmCudaMalloc(output.GetBytes() * devices.size());

        auto *pool = fastllm::GetAlivePool();
        std::vector<fastllm::MultiThreadBaseOp*> ops;
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            std::string specialId = "";
            int mallocType;
            DeviceGetInfos(device, specialId, mallocType);

            if (specialId != "cpu") {
                ops.push_back(new MultiCudaDoMergeMOEOp(
                    (uint8_t*)input.cudaData, (uint8_t*)cpuInput.data(), partOutput + output.GetBytes() * i,
                    input.multiDeviceDatas[device], weights, &logits, &gateBias, 
                    w1.multiDeviceDatas[device], w2.multiDeviceDatas[device], w3.multiDeviceDatas[device], 
                    wBatch, topk, needNorm, routeScale, sharedScale, 
                    curOutput.multiDeviceDatas[device], device
                ));
            }
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }

        // run cpu op
        auto temp = pool->curActivateThreadInterval;
        pool->curActivateThreadInterval = std::make_pair(ops.size(), pool->threads.size());
        for (int i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            std::string specialId = "";
            int mallocType;
            DeviceGetInfos(device, specialId, mallocType);
            if (specialId == "cpu") {
                MultiCudaCpuDoMergeMOEOp (
                    (uint8_t*)cpuInput.data(), partOutput + output.GetBytes() * i,
                    input.multiDeviceDatas[device], weights, &logits, &gateBias, 
                    w1.multiDeviceDatas[device], w2.multiDeviceDatas[device], w3.multiDeviceDatas[device], 
                    wBatch, topk, needNorm, routeScale, sharedScale, 
                    curOutput.multiDeviceDatas[device], device).Run();
            }
        }
        pool->curActivateThreadInterval = temp;

        // wait cuda op
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
        FastllmReduce((uint8_t*)output.cudaData, partOutput, output.Count(0), devices.size(), output.dataType);
        FastllmCudaFree(partOutput);
    }
}