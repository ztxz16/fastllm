#include "ascenddevice.h"

#include "fastllm-acl.h"

#include "executor.h"
#include "utils.h"
#include <acl/acl.h>

namespace fastllm {
    AscendNpuDevice::AscendNpuDevice() {
        this->deviceType = "acl";
        npu::FastllmAclInit();
        this->ops["Linear"] = new AscendLinearOp();
        this->ops["Split"] = new AscendSplitOp();
        this->ops["LayerNorm"] = new AscendLayerNormOp();
        this->ops["SoftMax"] = new AscendSoftMaxOp();
        this->ops["Sigmoid"] = new AscendSigmoidOp();
        this->ops["Silu"] = new AscendSiluOp();
        this->ops["TanH"] = new AscendTanHOp();
        this->ops["Relu"] = new AscendReluOp();
        this->ops["Gelu"] = new AscendGeluOp();
        this->ops["GeluNew"] = new AscendGeluNewOp();
        this->ops["MulTo"] = new AscendMulToOp();
        this->ops["Swiglu"] = new AscendSwigluOp();
        this->ops["AddTo"] = new AscendAddToOp();
        this->ops["PermuteSelf"] = new AscendPermuteSelfOp();
    }

    AscendNpuDevice::~AscendNpuDevice() {
        // npu::FastllmAclFinalize();
    }

    bool AscendNpuDevice::Malloc(void **ret, size_t size) {
        *ret = npu::FastllmAclMalloc(size);
        return (*ret != nullptr);
    }

    bool AscendNpuDevice::Free(void *ret) {
        npu::FastllmAclFree(ret);
        return true;
    }

    bool AscendNpuDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        int result = npu::FastllmAclCopyFromHostToDevice(dst, src, size);
        return result == 0;
    }

    bool AscendNpuDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        int result = npu::FastllmAclCopyFromDeviceToHost(dst, src, size);
        return result == 0;
    }

    BaseAscendOperator::~BaseAscendOperator() {}

    bool BaseAscendOperator::RunSingleOp(const std::string &opType, const fastllm::OrderedData &inputData,
                                         const OrderedData &outputData, const fastllm::FloatDict &floatParams,
                                         const fastllm::IntDict &intParams, const std::map <std::string, bool> &boolParams) {
        std::vector<aclTensorDesc *> inputTensors;
        std::vector<aclDataBuffer *> inputBuffers;
        for (auto &pair : inputData)
            npu::FastllmAclToTensor(pair, inputTensors, inputBuffers);
        std::vector<aclTensorDesc *> outputTensors;
        std::vector<aclDataBuffer *> outputBuffers;
        for (auto &pair : outputData)
            npu::FastllmAclToTensor(pair, outputTensors, outputBuffers);
        aclopAttr *attr;
        npu::FastllmAclToOpAttribute(floatParams, intParams, boolParams, &attr);
        bool result = npu::FastllmAclExecute(opType, inputTensors, inputBuffers,
                               outputTensors, outputBuffers, attr);
        npu::FastllmAclDestoryTensors(inputTensors, inputBuffers, outputTensors, outputBuffers, &attr);
        return result;
    }

    bool BaseAscendOperator::CompileAndRunSingleOp(const std::string &opType, const OrderedData &inputData, const OrderedData &outputData,
                                                   const DynamicShapeDict &dynamicShapes, const FloatDict &floatParams,
                                                   const IntDict &intParams, const BoolDict &boolParams) {
        std::vector<aclTensorDesc *> inputTensors;
        std::vector<aclDataBuffer *> inputBuffers;
        for (auto &pair : inputData)
            npu::FastllmAclToTensor(pair, inputTensors, inputBuffers);
        std::vector<aclTensorDesc *> outputTensors;
        std::vector<aclDataBuffer *> outputBuffers;
        for (auto &pair : outputData)
            npu::FastllmAclToTensor(pair, outputTensors, outputBuffers);
        aclopAttr *attr;
        npu::FastllmAclToOpAttribute(floatParams, intParams, boolParams, &attr);
        if (warmUpMode) {
            std::vector<aclTensorDesc *> inputTensorsForCompile;
            std::vector<aclTensorDesc *> outputTensorsForCompile;
            for (auto &pair : inputData) {
                auto it = dynamicShapes.find(pair.first);
                if (it != dynamicShapes.end()) {
                    npu::FastllmAclCreateShape(pair, inputTensorsForCompile, it->second.first, it->second.second);
                } else {
                    npu::FastllmAclCreateShape(pair, inputTensorsForCompile);
                }
            }
            for (auto &pair : outputData) {
                auto it = dynamicShapes.find(pair.first);
                if (it != dynamicShapes.end()) {
                    npu::FastllmAclCreateShape(pair, outputTensorsForCompile, it->second.first, it->second.second);
                } else {
                    npu::FastllmAclCreateShape(pair, outputTensorsForCompile);
                }
            }
            npu::FastllmAclInitOp(opType, inputTensorsForCompile, outputTensorsForCompile, attr);
            npu::FastllmAclDestroyShape(inputTensorsForCompile);
            npu::FastllmAclDestroyShape(outputTensorsForCompile);
        }
        bool result = npu::FastllmAclExecuteAfterInit(opType, inputTensors, inputBuffers,
                                                      outputTensors, outputBuffers, attr);
        npu::FastllmAclDestoryTensors(inputTensors, inputBuffers, outputTensors, outputBuffers, &attr);
        return result;
    }

    bool BaseAscendOperator::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (!deviceOk)
            return false;
        Executor *executor = (Executor *) GetExecutor();
        this->warmUpMode = executor->isWarmUpMode();
        return true;
    }

    AscendLinearOp::AscendLinearOp() :
        BaseAscendOperator("BatchMatMulV2") {}

    void AscendLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");

        weight.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];

        output.dataType = input.dataType;
        output.Resize(dims);
        output.Allocate();
    }

    bool AscendLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (!deviceOk)
            return false;
        Executor *executor = (Executor *) GetExecutor();
        this->warmUpMode = executor->isWarmUpMode();

        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        if (input.dataType == DataType::FLOAT16 && weight.dataType == DataType::FLOAT16) {
            return true;
        } else if (input.dataType == DataType::FLOAT32 && weight.dataType == DataType::FLOAT32) {
            return true;
        } else if (input.dataType == DataType::FLOAT32 && weight.dataType == DataType::FLOAT16) {
            return true;
        }
        return false;
    }

    void AscendLinearOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        // 准备重整形的输入和输出数据
        Data fakeReshapedInput;
        fakeReshapedInput.dims = input.dims;
        fakeReshapedInput.strides = input.strides;
        fakeReshapedInput.FakeFrom(input, 0);
        fakeReshapedInput.Reshape({n, m});

        Data fakeReshapedOutput;
        fakeReshapedOutput.dims = output.dims;
        fakeReshapedOutput.strides = output.strides;
        fakeReshapedOutput.FakeFrom(output, 0);
        fakeReshapedOutput.Reshape({n, k});

        // 构建输入数据映射
        OrderedData orderedInputData;
        orderedInputData.push_back(std::make_pair("x1", &fakeReshapedInput));
        orderedInputData.push_back(std::make_pair("x2", &weight));
        if (bias.dims.size() > 0) {
            orderedInputData.push_back(std::make_pair("bias", &bias));
        }

        // 构建动态形状信息（用于warmup模式）
        DynamicShapeDict dynamicShapes;
        if (warmUpMode) {
            // 为x1添加动态形状范围
            std::vector<std::vector<int64_t>> shapeRanges(1, std::vector<int64_t>({1L, 2048L}));
            dynamicShapes["x1"] = std::make_pair(std::vector<int>{0}, shapeRanges);
            dynamicShapes["y"] = std::make_pair(std::vector<int>{0}, shapeRanges);
        }

        // 构建属性参数
        BoolDict boolParams = {{"adj_x1", false}, {"adj_x2", true}};
        if (input.dataType == DataType::FLOAT16) {
            if (weight.dataType == DataType::FLOAT16) {
                deviceOk = CompileAndRunSingleOp(this->name, orderedInputData, {{"y", &fakeReshapedOutput}},
                                                dynamicShapes, {}, {}, boolParams);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32 || weight.dataType == DataType::FLOAT16) {
                deviceOk = CompileAndRunSingleOp(this->name, orderedInputData, {{"y", &fakeReshapedOutput}},
                                                dynamicShapes, {}, {}, boolParams);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport input's dataType.\n");
        }
    }

    AscendSplitOp::AscendSplitOp() : 
        BaseAscendOperator("Slice") {}

    bool AscendSplitOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (!BaseAscendOperator::CanRun(opType, datas, floatParams, intParams))
            return false;
        Data* input = datas.find("input")->second;
        if (input->dataDevice != DataDevice::NPU)
            return false;
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        return (axis != 1);
    }

    void AscendSplitOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));
        std::vector <int> dims = input.dims;
        dims[axis] = end - start;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void AscendSplitOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));
        std::vector <int32_t> offsetDims(dimsLen);
        offsetDims[axis] = start;
        Data offsets(DataType::FLOAT32, {dimsLen});
        offsets.Allocate();
        for (int i = 0; i < dimsLen; i++) {
            ((int32_t*)offsets.cpuData)[i] = offsetDims[i];
        }
        offsets.ToDevice(input.dataDevice);
        offsets.dataType = DataType::INT32PARAM;
        std::vector <int32_t> sizeDims(input.dims);
        sizeDims[axis] = end - start;
        Data sizes(DataType::FLOAT32, {dimsLen});
        sizes.Allocate();
        for (int i = 0; i < dimsLen; i++) {
            ((int32_t*)sizes.cpuData)[i] = sizeDims[i];
        }
        sizes.ToDevice(input.dataDevice);
        sizes.dataType = DataType::INT32PARAM;
        OrderedData inputDict({{"x", &input}, {"offsets", &offsets}, {"size", &sizes}});
        if (axis == 1) {
            deviceOk = RunSingleOp(this->name, inputDict, {{"y", &output}}, {}, {}, {});
        } else {
            DynamicShapeDict dynamicShapes;
            dynamicShapes["x"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
            dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
            deviceOk = CompileAndRunSingleOp(this->name, inputDict, {{"y", &output}}, dynamicShapes, {}, {}, {});
        }
    }

    AscendSoftMaxOp::AscendSoftMaxOp() :
        BaseAscendOperator("SoftmaxV2") {}

    void AscendSoftMaxOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = datas.find("input")->second;
        Data *output = datas.find("output")->second;
        output->Allocate();
        int axis = intParams.find("axis")->second;
        if (axis < 0)
            axis = axis + input->dims.size();

        DynamicShapeDict dynamicShapes;
        dynamicShapes["x"] = std::make_pair(std::vector<int32_t>({0, 2, 3}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}, {1,2048}}));
        dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 2, 3}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}, {1,2048}}));
        deviceOk = CompileAndRunSingleOp(this->name, {{"x", input}}, {{"y", output}}, dynamicShapes, {}, {{"axis", axis}}, {});
    }

    AscendSigmoidOp::AscendSigmoidOp() :
        BaseAscendOperator("Sigmoid") {}

    void AscendSigmoidOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = datas.find("input")->second;
        Data *output = datas.find("output")->second;
        output->Allocate();
        AssertInFastLLM(input->dataType == DataType::FLOAT32 ||
                        input->dataType == DataType::FLOAT16,
                        "Sigmoid error: Data's type should be float32 or float16.\n");

        DynamicShapeDict dynamicShapes;
        dynamicShapes["x"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        deviceOk = CompileAndRunSingleOp(this->name, {{"x", input}}, {{"y", output}}, dynamicShapes, {}, {}, {});
    }

    AscendTanHOp::AscendTanHOp() :
        BaseAscendOperator("Tanh") {}

    void AscendTanHOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = datas.find("input")->second;
        Data *output = datas.find("output")->second;
        DynamicShapeDict dynamicShapes;
        dynamicShapes["x"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        deviceOk = CompileAndRunSingleOp(this->name, {{"x", input}}, {{"y", output}}, dynamicShapes, {}, {}, {});
    }

    AscendReluOp::AscendReluOp() :
        BaseAscendOperator("Relu") {}

    void AscendReluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = datas.find("input")->second;
        Data *output = datas.find("output")->second;
        DynamicShapeDict dynamicShapes;
        dynamicShapes["x"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        deviceOk = CompileAndRunSingleOp(this->name, {{"x", input}}, {{"y", output}}, dynamicShapes, {}, {}, {});
    }

    AscendGeluOp::AscendGeluOp() :
        BaseAscendOperator("GeluV2") {}

    void AscendGeluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = datas.find("input")->second;
        Data *output = datas.find("output")->second;
        DynamicShapeDict dynamicShapes;
        dynamicShapes["x"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        deviceOk = CompileAndRunSingleOp(this->name, {{"x", input}}, {{"y", output}}, dynamicShapes, {}, {}, {});
    }

    AscendGeluNewOp::AscendGeluNewOp() :
        BaseAscendOperator("Gelu") {}

    void AscendGeluNewOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = datas.find("input")->second;
        Data *output = datas.find("output")->second;
        DynamicShapeDict dynamicShapes;
        dynamicShapes["x"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        deviceOk = CompileAndRunSingleOp(this->name, {{"x", input}}, {{"y", output}}, dynamicShapes, {}, {}, {});
    }

    AscendSiluOp::AscendSiluOp() : 
        BaseAscendOperator("Swish") {}

    void AscendSiluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = datas.find("input")->second;
        Data *output = datas.find("output")->second;
        DynamicShapeDict dynamicShapes;
        dynamicShapes["x"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        deviceOk = CompileAndRunSingleOp(this->name, {{"x", input}}, {{"y", output}}, dynamicShapes, {{"scale", 1.0}}, {}, {});
    }

    AscendMulToOp::AscendMulToOp() : 
        BaseAscendOperator("Mul") {}

    bool AscendMulToOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (!BaseAscendOperator::CanRun(opType, datas, floatParams, intParams))
            return false;
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;
        return alpha == 1.0f;
    }

    void AscendMulToOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);

        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "MulTo error: Data's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims == input1.dims, "MulTo error: input's shape should be same.\n");
        DynamicShapeDict dynamicShapes;
        dynamicShapes["x1"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["x2"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        deviceOk = CompileAndRunSingleOp(this->name, {{"x1", &input0}, {"x2", &input1}}, {{"y", &input0}}, dynamicShapes, {}, {}, {});
    }

    AscendSwigluOp::AscendSwigluOp() :
        BaseAscendOperator("") {}

    bool AscendSwigluOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (!BaseAscendOperator::CanRun(opType, datas, floatParams, intParams))
            return false;
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;
        return alpha == 1.0f;
    }

    void AscendSwigluOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
         Data &input = *(datas.find("input")->second);
         Data &output = *(datas.find("output")->second);

         std::vector <int> dims = input.dims;
         dims[dims.size() - 1] /= 2;
         output.dataType = input.dataType;
         output.Resize(dims);
    }

    void AscendSwigluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Swiglu error: Data's type should be float32 or float16.\n");

        Data mulHalf(output);
        mulHalf.Allocate();

        int dimsLen = input.dims.size();
        std::vector <int32_t> offsetDims(dimsLen);
        Data swishOffsets(DataType::FLOAT32, {dimsLen});
        swishOffsets.Allocate();
        for (int i = 0; i < dimsLen; i++) {
            ((int32_t*)swishOffsets.cpuData)[i] = offsetDims[i];
        }
        Data mulOffsets(swishOffsets);
        ((int32_t*)mulOffsets.cpuData)[dimsLen - 1] = input.dims.back() / 2;
        swishOffsets.ToDevice(input.dataDevice);
        swishOffsets.dataType = DataType::INT32PARAM;
        mulOffsets.ToDevice(input.dataDevice);
        mulOffsets.dataType = DataType::INT32PARAM;

        Data swishSizes(DataType::FLOAT32, {dimsLen});
        swishSizes.Allocate();
        for (int i = 0; i < output.dims.size(); i++) {
            ((int32_t*)swishSizes.cpuData)[i] = output.dims[i];
        }
        swishSizes.ToDevice(input.dataDevice);
        swishSizes.dataType = DataType::INT32PARAM;
        Data mulSizes(swishSizes);

        std::vector<int32_t> shapeRangeDims = {0, 1};
        std::vector<std::vector<int64_t>> shapeRanges = {{1L,128L}, {1L,2048L}};
        DynamicShapeDict dynamicShapes;
        dynamicShapes["x"] = std::make_pair(shapeRangeDims, shapeRanges);
        dynamicShapes["y"] = std::make_pair(shapeRangeDims, shapeRanges);
        deviceOk = CompileAndRunSingleOp("Slice", {{"x", &input}, {"offsets", &swishOffsets}, {"size", &swishSizes}}, {{"y", &output}}, dynamicShapes, {}, {}, {});

        deviceOk = CompileAndRunSingleOp("Swish", {{"x", &output}}, {{"y", &output}}, dynamicShapes, {{"scale", 1.0}}, {}, {});

        deviceOk = CompileAndRunSingleOp("Slice", {{"x", &input}, {"offsets", &mulOffsets}, {"size", &mulSizes}}, {{"y", &mulHalf}}, dynamicShapes, {}, {}, {});
        dynamicShapes.clear();
        dynamicShapes["x1"] = std::make_pair(shapeRangeDims, shapeRanges);
        dynamicShapes["x2"] = std::make_pair(shapeRangeDims, shapeRanges);
        dynamicShapes["y"] = std::make_pair(shapeRangeDims, shapeRanges);
        deviceOk = CompileAndRunSingleOp("Mul", {{"x1", &output}, {"x2", &mulHalf}}, {{"y", &output}}, dynamicShapes, {}, {}, {});
    }

    AscendAddToOp::AscendAddToOp() :
        BaseAscendOperator("Axpy") {}

    void AscendAddToOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;

        AssertInFastLLM(input0.dataType == DataType::FLOAT32 || input1.dataType == DataType::FLOAT16,
                        "AddTo error: Data's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");

        int len = input0.Count(0);
        DynamicShapeDict dynamicShapes;
        dynamicShapes["x1"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["x2"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), std::vector<std::vector<int64_t>>({{1,128}, {1,2048}}));
        deviceOk = CompileAndRunSingleOp(this->name, {{"x1", &input0}, {"x2", &input1}}, {{"y", &input0}}, dynamicShapes, floatParams, {}, {});
    }

    AscendLayerNormOp::AscendLayerNormOp() :
        BaseAscendOperator("LayerNorm") {}

    bool AscendLayerNormOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                   const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (!BaseAscendOperator::CanRun(opType, datas, floatParams, intParams))
            return false;
        Data &input = *(datas.find("input")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int inner = input.strides[axis];
        return inner == 1;
    }

    void AscendLayerNormOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &gamma = *(datas.find("gamma")->second);
        Data &beta = *(datas.find("beta")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "LayerNorm error: input's data type should be float32 or float16.\n");
        AssertInFastLLM(gamma.dataType == DataType::FLOAT32 || gamma.dataType == DataType::FLOAT16,
                        "LayerNorm error: gamma's data type should be float32 or float16.\n");
        AssertInFastLLM(beta.dataType == DataType::FLOAT32 || beta.dataType == DataType::FLOAT16,
                        "LayerNorm error: beta's data type should be float32 or float16.\n");

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        output.Allocate();

        // LayerNorm 强制输出 y、mean、variance，其中 mean/variance 形状为
        // [dims[0], ..., dims[axis - 1], 1, ..., 1]，数据类型与 x 一致
        std::vector<int> meanDims = input.dims;
        for (int i = axis; i < dimsLen; i++) {
            meanDims[i] = 1;
        }
        Data tempMean(input.dataType, meanDims);
        Data tempVariance(input.dataType, meanDims);
        tempMean.Allocate();
        tempVariance.Allocate();
        tempMean.ToDevice(input.dataDevice);
        tempVariance.ToDevice(input.dataDevice);

        // 准备输入数据映射
        OrderedData orderedInputData;
        orderedInputData.push_back(std::make_pair("x", &input));
        orderedInputData.push_back(std::make_pair("gamma", &gamma));
        orderedInputData.push_back(std::make_pair("beta", &beta));

        // 构建动态形状信息（用于warmup模式），mean/variance 与 x 共享相同的动态维度
        DynamicShapeDict dynamicShapes;
        if (warmUpMode && dimsLen >= 2) {
            std::vector<std::vector<int64_t>> shapeRanges = {{1L, 128L}, {1L, 2048L}};
            dynamicShapes["x"] = std::make_pair(std::vector<int32_t>({0, 1}), shapeRanges);
            dynamicShapes["y"] = std::make_pair(std::vector<int32_t>({0, 1}), shapeRanges);
            dynamicShapes["mean"] = std::make_pair(std::vector<int32_t>({0, 1}), shapeRanges);
            dynamicShapes["variance"] = std::make_pair(std::vector<int32_t>({0, 1}), shapeRanges);
        }

        // LayerNorm 属性：begin_norm_axis / begin_params_axis / epsilon。
        // 注意不能把 fastllm 的 "axis" 传给 ACL，否则会因属性不匹配导致编译失败。
        float eps = 1e-10f;
        IntDict aclIntParams = {{"begin_norm_axis", axis}, {"begin_params_axis", axis}};

        deviceOk = CompileAndRunSingleOp(this->name, orderedInputData, {{"y", &output}, {"mean", &tempMean}, {"variance", &tempVariance}},
                                         dynamicShapes, {{"epsilon", eps}}, aclIntParams, {});
    }

    AscendPermuteSelfOp::AscendPermuteSelfOp() :
        BaseAscendOperator("Transpose") {}

    bool AscendPermuteSelfOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (!BaseAscendOperator::CanRun(opType, datas, floatParams, intParams))
            return false;
        Data *input = datas.find("input")->second;
        if (input->dataDevice != DataDevice::NPU)
            return false;

        // 检查数据类型
        if (input->dataType != DataType::FLOAT32 && input->dataType != DataType::FLOAT16) {
            return false;
        }

        // 检查维度匹配
        Data *axisData = datas.find("axis")->second;
        std::vector<int> axis;
        for (int i = 0; i < axisData->Count(0); i++) {
            axis.push_back(((int32_t *)axisData->cpuData)[i]);
        }
        
        if (axis.size() != input->dims.size()) {
            return false;
        }

        return true;
    }

    void AscendPermuteSelfOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &axisData = *(datas.find("axis")->second);

        std::vector<int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *)axisData.cpuData)[i]);
        }

        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "PermuteSelf error: input's data type should be float32 or float16.\n");
        AssertInFastLLM(axis.size() == input.dims.size(),
                        "PermuteSelf error: axis's size should be equal to input's dims size.\n");

        std::vector<int> newDims;
        for (int i = 0; i < axis.size(); i++) {
            newDims.push_back(input.dims[axis[i]]);
        }

        // 构建 perm 数据，作为 Transpose 的输入
        Data permData;
        permData.CopyFrom(axisData);
        permData.dataType = DataType::FLOAT32;
        permData.ToDevice(input.dataDevice);
        permData.dataType = DataType::INT32PARAM;

        // 临时输出，转置完成后拷回 input 的设备内存实现原地 permute
        Data tmpData(input.dataType, newDims);
        tmpData.Allocate();
        tmpData.ToDevice(input.dataDevice);

        // 构建动态形状信息 (用于warmup模式)
        // {0, 2, 1, 3}:
        // q/k/v [batch, len, num_head, head_dim]
        // qkv   [batch, num_head, len, head_dim]
        DynamicShapeDict dynamicShapes;
        if (warmUpMode && input.dims.size() >= 2) {
            std::vector<int> xDynamicDims = {0, 1, 2};
            std::vector<std::vector<int64_t>> shapeRanges = {{1L, 128L}, {1L, 2048L}, {1L, 2048L}};
            std::vector<int> yDynamicDims;
            for (int i = 0; i < axis.size(); i++) {
                if (std::find(xDynamicDims.begin(), xDynamicDims.end(), axis[i]) != xDynamicDims.end()) {
                    yDynamicDims.push_back(i);
                }
            }
            dynamicShapes["x"] = std::make_pair(xDynamicDims, shapeRanges);
            dynamicShapes["y"] = std::make_pair(yDynamicDims, shapeRanges);
        }

        deviceOk = CompileAndRunSingleOp(this->name, {{"x", &input}, {"perm", &permData}},
                                        {{"y", &tmpData}}, dynamicShapes, {}, {}, {});

        // 把转置结果拷回 input 的设备内存，并更新 shape
        if (deviceOk) {
            npu::FastllmAclCopyFromDeviceToDevice(input.deviceData, tmpData.deviceData, input.GetBytes());
            input.Resize(newDims);
        }
    }

}
