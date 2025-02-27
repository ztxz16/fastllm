//
// Created by huangyuyang on 24/2/25.
//

#include <sys/mman.h>
#include <fcntl.h>

#include "devices/tops/topsdevice.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/alivethreadpool.h"

#include <cstring>
#include <thread>
#include <cfloat>
#include <cmath>

#include "utils.h"

#include "tops/tops_ext.h"
#include "tops/tops_runtime.h"
//#include "topsdnn.h"
#include "topsaten/topsaten.h"
#include "topsaten/topsaten_ops.h"

#define TOPS_CHECK(call)                                                     \
  {                                                                          \
    const topsError_t error = call;                                          \
    if (error != topsSuccess) {                                              \
      printf("runtime Error: %s:%d, error=%d\n", __FILE__, __LINE__, error); \
      exit(1);                                                               \
    }                                                                        \
  }

#define TOPSATEN_CHECK(call)                                                    \
  {                                                                           \
    const topsatenStatus_t error = call;                                        \
    if (error != TOPSATEN_STATUS_SUCCESS) {                                     \
      printf("Topsaten Error: %s:%d, error=%d\n", __FILE__, __LINE__, error); \
      exit(1);                                                                \
    }                                                                         \
  }

#define TOPSDNN_CHECK(call)                                 \
  {                                                         \
    const topsdnnStatus_t error = call;                     \
    if (error != TOPSDNN_STATUS_SUCCESS) {                  \
      printf("topsdnn error: %s:%d\n", __FILE__, __LINE__); \
      exit(1);                                              \
    }                                                       \
  }


namespace fastllm {
    TopsDevice::TopsDevice() {
        this->deviceType = "tops";
        this->ops["Linear"] = (BaseOperator *) (new TopsLinearOp());
    }

    bool TopsDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t [size];
        return true;
    }

    bool TopsDevice::Free(void *ret) {
        delete[] (uint8_t *)ret;
        return true;
    }

    bool TopsDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        return true;
    }
    
    bool TopsDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        return true;
    }

    void TopsLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
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
    }

    bool TopsLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }

        Data &weight = *(datas.find("weight")->second);
        return true;
    }

    static topsStream_t stream;
    static bool topsIsInited = false;

    void TopsInit() {
        if (!topsIsInited) {
            topsIsInited = true;
            topsError_t status = topsSuccess;
            topsStream_t stream;
            // init
            TOPSATEN_CHECK(topsatenInit());
            status = topsStreamCreate(&stream);
        }
    }

    extern void Float16ToFloat32(uint16_t *float16, float *float32, int len);
    extern void Float32ToFloat16(float *float32, uint16_t *float16, int len);

    void TopsLinearOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
// auto st = std::chrono::system_clock::now();
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();
        void *d_lhs, *d_rhs, *d_out, *d_bias;

        TopsInit();
        AssertInFastLLM(weight.dataType == DataType::FLOAT16, "Tops.Linear: dtype should be float16.");

        if (weight.deviceData == nullptr) {
            TOPS_CHECK(topsMallocAsync(reinterpret_cast<void **>(&weight.deviceData), weight.GetBytes(), stream, 0));
            TOPS_CHECK(topsMemcpyAsync(weight.deviceData, weight.cpuData, weight.GetBytes(), topsMemcpyHostToDevice, stream));
        }
        d_rhs = weight.deviceData;

        TOPS_CHECK(topsMallocAsync(reinterpret_cast<void **>(&d_lhs), input.Count(0) * sizeof(uint16_t), stream, 0));
        TOPS_CHECK(topsMallocAsync(reinterpret_cast<void **>(&d_out), output.GetBytes(), stream, 0));
        TOPS_CHECK(topsMallocAsync(reinterpret_cast<void **>(&d_bias), k * sizeof(uint16_t), stream, 0));

        if (input.dataType == DataType::FLOAT32) {
            uint16_t *float16Input = new uint16_t[input.Count(0)];
            Float32ToFloat16((float*)input.cpuData, float16Input, input.Count(0));
            TOPS_CHECK(topsMemcpyAsync(d_lhs, float16Input, input.Count(0) * sizeof(uint16_t), topsMemcpyHostToDevice, stream));
            TOPS_CHECK(topsStreamSynchronize(stream));
            delete[] float16Input;
        } else {
            TOPS_CHECK(topsMemcpyAsync(d_lhs, input.cpuData, input.GetBytes(), topsMemcpyHostToDevice, stream));
        }
        
        if (bias.dims.size() > 0) {
            ToDataType(bias, DataType::FLOAT16);
            TOPS_CHECK(topsMemcpyAsync(d_bias, bias.cpuData, bias.GetBytes(), topsMemcpyHostToDevice, stream));
        } else {
            TOPS_CHECK(topsMemsetAsync(d_bias, 0, k * sizeof(uint16_t), stream));
        }
        
        // tensor
        std::vector<int64_t> lhs_shape = {n, m};
        std::vector<int64_t> lhs_stride = {m, 1};
        std::vector<int64_t> rhs_shape = {k, m};
        std::vector<int64_t> rhs_stride = {m, 1};
        std::vector<int64_t> out_shape = {n, k};
        std::vector<int64_t> out_stride = {k, 1};
        std::vector<int64_t> bias_shape = {k};
        std::vector<int64_t> bias_stride = {1};

        topsatenSize_t tensor_dims;
        topsatenSize_t tensor_strides;
        tensor_dims.data = lhs_shape.data();
        tensor_dims.len = lhs_shape.size();
        tensor_strides.data = lhs_stride.data();
        tensor_strides.len = lhs_stride.size();
        topsatenTensor t_lhs(tensor_dims, tensor_strides, TOPSATEN_DATA_FP16, d_lhs);

        tensor_dims.data = rhs_shape.data();
        tensor_dims.len = rhs_shape.size();
        tensor_strides.data = rhs_stride.data();
        tensor_strides.len = rhs_stride.size();
        topsatenTensor t_rhs(tensor_dims, tensor_strides, TOPSATEN_DATA_FP16, d_rhs);

        tensor_dims.data = out_shape.data();
        tensor_dims.len = out_shape.size();
        tensor_strides.data = out_stride.data();
        tensor_strides.len = out_stride.size();
        topsatenTensor t_out(tensor_dims, tensor_strides, TOPSATEN_DATA_FP16, d_out);

        tensor_dims.data = bias_shape.data();
        tensor_dims.len = bias_shape.size();
        tensor_strides.data = bias_stride.data();
        tensor_strides.len = bias_stride.size();
        topsatenTensor t_bias(tensor_dims, tensor_strides, TOPSATEN_DATA_FP16, d_bias);
// TOPS_CHECK(topsStreamSynchronize(stream));
// auto st = std::chrono::system_clock::now();
        TOPSATEN_CHECK(topsaten::topsatenLinear(t_out, t_lhs, t_rhs, t_bias,  stream));
// TOPS_CHECK(topsStreamSynchronize(stream));
// float spend = GetSpan(st, std::chrono::system_clock::now());
// float gops = (float)n * m * k / spend / 1e9;
// if (n > 1) printf("n = %d, m = %d, k = %d, spend %f s, gops = %f, outer = %f\n", n, m, k, spend, gops, spend);

        if (output.dataType == DataType::FLOAT32) {
            uint16_t *float16Output = new uint16_t[output.Count(0)];
            TOPS_CHECK(topsMemcpyAsync(float16Output, d_out, output.Count(0) * sizeof(uint16_t), topsMemcpyDeviceToHost, stream));
            TOPS_CHECK(topsStreamSynchronize(stream));
            Float16ToFloat32(float16Output, (float*)output.cpuData, output.Count(0));
            delete[] float16Output;
        } else {
            TOPS_CHECK(topsMemcpyAsync(output.cpuData, d_out, output.GetBytes(), topsMemcpyDeviceToHost, stream));
        }
        // release
        // Free device global memory
        TOPS_CHECK(topsFreeAsync(d_lhs, stream));
        TOPS_CHECK(topsFreeAsync(d_out, stream));
        TOPS_CHECK(topsFreeAsync(d_bias, stream));
        TOPS_CHECK(topsStreamSynchronize(stream));
// float spend = GetSpan(st, std::chrono::system_clock::now());
// float gops = (float)n * m * k / spend / 1e9;
// if (n > 0) printf("n = %d, m = %d, k = %d, spend %f s, gops = %f, outer = %f\n", n, m, k, spend, gops, spend - inner);
    }

    long long int TopsLinearOp::Ops(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        return (long long int) n * m * k;
    }
}