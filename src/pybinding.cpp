#include "model.h"
#include "factoryllm.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <unordered_map>

namespace py = pybind11;
using namespace pybind11::literals;  

namespace pyfastllm{
  // TODO GPU内存不释放的bug
  // 对接不断更新的后端接口
  // 需优化，减少内存拷贝
  
  fastllm::Data Embedding(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output){
    fastllm::Embedding(input, weight, output);
    // output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }
  fastllm::Data &RMSNorm(const fastllm::Data &input, const fastllm::Data &weight, float eps, fastllm::Data &output){
    // fastllm::Data output;
    // std::cout<<"run rms norm"<<std::endl;
    fastllm::RMSNorm(input, weight, eps, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data LayerNorm(fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, int axis){
    fastllm::Data output;
    fastllm::LayerNorm(input, gamma, beta, axis, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Linear(fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output){
    // fastllm::Data output;
    fastllm::Linear(input, weight, bias, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data MatMul(const fastllm::Data &input0, const fastllm::Data &input1, float alpha){
    fastllm::Data output;
    fastllm::MatMul(input0, input1, output, alpha);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Attention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v, const fastllm::Data &mask,
                   int group, float scale, int attentionType, fastllm::Data &output) {
    fastllm::Attention(q, k, v, mask, output, group, scale, attentionType);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Softmax(const fastllm::Data &input,int axis) {
    fastllm::Data output;
    fastllm::Softmax(input, output, axis);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Silu(const fastllm::Data &input) {
    fastllm::Data output;
    fastllm::Silu(input, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Gelu(const fastllm::Data &input) {
    fastllm::Data output;
    fastllm::GeluNew(input, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Swiglu(const fastllm::Data &input, fastllm::Data &output) {
    fastllm::Swiglu(input, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Mul(const fastllm::Data &input, float v, fastllm::Data &output){
    fastllm::Mul(input, v, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Add(fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
    // fastllm::Data output;
    fastllm::AddTo(input0, input1, alpha);
    input0.ToDevice(fastllm::DataDevice::CPU);
    return input0;
  }

  fastllm::Data Split(const fastllm::Data &input, int axis, int start, int end, fastllm::Data &output){
    fastllm::Split(input, axis, start, end, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Permute(const fastllm::Data &input, const std::vector<int> &axis, fastllm::Data &output){
    fastllm::Permute(input, axis, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data Cat(const fastllm::Data &input0, const fastllm::Data &input1, int axis) {
    fastllm::Data output;
    fastllm::Cat(input0, input1, axis, output);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data CatDirect(fastllm::Data &input0, const fastllm::Data &input1, int axis) {
    fastllm::CatDirect(input0, input1, axis);
    input0.ToDevice(fastllm::DataDevice::CPU);
    return input0;
  }

  fastllm::Data TopK(const fastllm::Data &input, int topk, fastllm::Data &output) {
    fastllm::TopK(input, output, topk);
    output.ToDevice(fastllm::DataDevice::CPU);
    return output;
  }

  fastllm::Data RotatePosition2D(fastllm::Data &input, const fastllm::Data &positionIds, fastllm::Data &sinData, fastllm::Data &cosData, int rotaryDim){
    fastllm::RotatePosition2D(input, positionIds, sinData, cosData, rotaryDim);
    // input.ToDevice(fastllm::DataDevice::CPU);
    return input;
  }

  fastllm::Data NearlyRotatePosition2D(fastllm::Data &input, 
                          const fastllm::Data &positionIds, 
                          fastllm::Data &sinData, 
                          fastllm::Data &cosData, 
                          int rotaryDim){
    fastllm::NearlyRotatePosition2D(input, positionIds, sinData, cosData, rotaryDim);
    input.ToDevice(fastllm::DataDevice::CPU);
    return input;
  }

  std::string String(const fastllm::Data &data){
    std::string ss;
    ss += "tensor([";
    int last_dim = data.dims.back();
    int n = data.Count(0) / last_dim, m = last_dim;
    for (int i = 0; i < n; i++) {
      if (i > 0) ss += "\n";
      for (int j = 0; j < 3 && j < m; j++) {
          if (j>0) ss += ", ";
          ss += std::to_string(reinterpret_cast<float*>(data.cpuData)[i*m+j]);
      }
      if (m > 3) {
          ss += "..., ";
          for (int j = 0; j < 3 && j < m; j++) {
            if (j>0) ss += ", ";
            ss += std::to_string(reinterpret_cast<float*>(data.cpuData)[i*m + (m-3+j)]);
          }
      }
      
    }
    ss += "])";
    return ss;
  }

  std::vector<int> GetDims(const fastllm::Data &data){
    return data.dims;
  }

  int GetSize(const fastllm::Data &data, int idx){
     int n = data.dims.size();
     idx = (idx + n) % n;
     return data.dims[idx];
  }
  

  fastllm::Data ToDevice(fastllm::Data &data, const std::string &devices){
    size_t pos = devices.find(":");
    int len = devices.length();

    std::vector<int>deviceIds;
    std::string deviceStr = devices; 
    int device = fastllm::DataDevice::CPU;
    int deviceNum = 0;

    if (pos != -1){
      int deviceNum = atoi(devices.substr(pos, len-pos-1).c_str());
      deviceStr = devices.substr(0, pos);
    }

    deviceIds = {deviceNum};
    std::cout<<deviceStr<<std::endl;

    if (deviceStr == "cuda"){
       data.ToDevice(fastllm::DataDevice::CUDA, deviceIds);
    }else{
       data.ToDevice(fastllm::DataDevice::CPU, deviceIds);
    }
    return data;
  }


  fastllm::Data ToCuda(fastllm::Data &data){
    std::vector<int>deviceIds{0};
    data.ToDevice(fastllm::DataDevice::CUDA, deviceIds);
    return data;
  }

  fastllm::Data ToCpu(fastllm::Data &data){
    std::vector<int>deviceIds{0};
    data.ToDevice(fastllm::DataDevice::CPU, deviceIds);
    return data;
  }


  // TODO:fix data double free bug
  template<typename data_t = float>
  py::array_t<data_t> ToNumpy(fastllm::Data &data){
      py::capsule free_when_done_d(data.cpuData, [](void* f) {
            delete[] f;
      });
      std::vector<uint64_t> newStrides(std::move(data.strides));
      for (auto &stride:newStrides){
        stride *= sizeof(data_t);
      }
      return py::array_t<data_t>(
                data.dims,  // shape
                newStrides,  // C-style contiguous strides for each index
                (data_t*)data.cpuData,  // the data pointer
                free_when_done_d
            );
  }


  class Tensor {
  public:

    Tensor(){}

    Tensor(const Tensor& rhs){
      this->ptr = rhs.ptr;
    }

    Tensor(fastllm::DataType type) {
      // auto *tensor = new fastllm::Data(type);
      ptr = std::make_shared<fastllm::Data>(type);
    }

    Tensor(fastllm::DataType type, const std::vector<int> &dims) {
      // auto *tensor = new fastllm::Data(type, dims);
      this->ptr = std::make_shared<fastllm::Data>(type, dims);
    }

    Tensor(fastllm::DataType type, const std::vector<int> &dims, const std::vector<float> &data){
        // auto *tensor = new fastllm::Data(type, dims, data);
        this->ptr = std::make_shared<fastllm::Data>(type, dims, data);
    }

    Tensor(fastllm::Data &data){
      // auto *tensor = new fastllm::Data(data);
      this->ptr = std::make_shared<fastllm::Data>(data);
    }

    py::buffer_info MemBuffer(){
      std::vector<uint64_t> newStrides(std::move(ptr->strides));
      for(auto &stride:newStrides){
        stride *= sizeof(float);
      }
      return py::buffer_info(
          ptr->cpuData,                               /* Pointer to buffer */
          sizeof(float),                          /* Size of one scalar */
          py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
          ptr->dims.size(),                                      /* Number of dimensions */
          ptr->dims,                 /* Buffer dimensions */
          newStrides                  /* Strides (in bytes) for each index */
      );
    }

    py::list ToList(){
      auto n = this->Count(0);
      py::list data(n);
      for (int i = 0; i < n; i++) 
        data.append(((float*)ptr->cpuData)[i]);
      return data;
    }

    uint64_t Count(int idx) const {
      return this->ptr->Count(idx);
    }

    std::vector<int> GetDims() const {
      return this->ptr->dims;
    }

    int GetSize(int idx) const {
      int n = ptr->dims.size();
      idx = (idx + n) % n;
      return ptr->dims[idx];
    }

    std::vector<int> GetExpansionDims () const{
      return this->ptr->expansionDims;
    }

    void Reshape(const std::vector<int> &dims){
      ptr->Reshape(dims);
    }

    void Expansion(const std::vector<int> &dims) {
      ptr->Expansion(dims);
    }

    std::string HostString(){
      std::string ss;
      ss += "tensor([";
      int last_dim = this->ptr->dims.back();
      int n = this->ptr->Count(0) / last_dim, m = last_dim;
      for (int i = 0; i < n; i++) {
        if (i > 0) ss += "\n";
        for (int j = 0; j < 3 && j < m; j++) {
            if (j>0) ss += ", ";
            ss += std::to_string(reinterpret_cast<float*>(this->ptr->cpuData)[i*m+j]);
        }
        if (m > 3) {
            ss += "..., ";
            for (int j = 0; j < 3 && j < m; j++) {
              if (j>0) ss += ", ";
              ss += std::to_string(reinterpret_cast<float*>(this->ptr->cpuData)[i*m + (m-3+j)]);
            }
        }
      }
      ss += "])";
      return ss;
    }

    std::shared_ptr<fastllm::Data> ptr;
  };

  fastllm::WeightMap &LoadWeights(const std::string &fileName){
    fastllm::WeightMap wm;
    wm.LoadFromFile(fileName);
    return wm;
  }

  // 浅拷贝
  template<typename data_t>
  fastllm::Data fromNumpy(pybind11::array_t<data_t> NpData){
    pybind11::buffer_info buf = NpData.request();
    // printf("%u \n", buf.ptr);
    data_t *ptr = (data_t*) buf.ptr;

    std::vector<int> dataSize;
    uint64_t dataNum = 1;
    for (auto sz:buf.shape){
      dataSize.emplace_back((int)sz);
      dataNum *= sz;
    }

    std::vector<data_t>Vdata;
    for (int i=0;i<dataNum;i++){
      Vdata.emplace_back(*(ptr+i));
    }

    int n = buf.strides.size();
    std::vector<size_t> newStrides(n);
    std::vector<int> newShape(n);
    for (auto i=0;i<buf.strides.size();i++){
      newStrides[i] = buf.strides[i] / sizeof(data_t);
      newShape[i] = int(buf.shape[i]);
    }

    // output.dims = newShape;
    // output.dataType = fastllm::DataType::FLOAT32;
    // // output.Resize(newShape)
    // output.UpdateUnitSize();

    // output.strides = newStrides;
    // // std::memcpy(output.cpuData, ptr, output.GetBytes());
    // output.cpuData = (uint8_t*)ptr;

    // Tensor data = Tensor(fastllm::DataType::FLOAT32, dataSize, Vdata);
    fastllm::Data data = fastllm::Data(fastllm::DataType::FLOAT32, dataSize, Vdata);
    // output(std::move(fastllm::Data(fastllm::DataType::FLOAT32, dataSize, Vdata))); 
    // = fastllm::Data(fastllm::DataType::FLOAT32, dataSize, Vdata);
    // printf("build end!!!\n");
    // std::memcpy(data.cpuData, ptr, data.GetBytes());
    return data;
  }
}

// #ifdef PY_API

namespace py = pybind11;
using namespace pybind11::literals;  

// template <typename... Args>
// using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;


using pastKV = std::vector<std::pair<fastllm::Data,fastllm::Data>>;
// PYBIND11_MAKE_OPAQUE(std::vector<std::pair<fastllm::Data,fastllm::Data>>);
PYBIND11_MAKE_OPAQUE(fastllm::Data);

PYBIND11_MODULE(pyfastllm, m) {
  m.doc() = "fastllm python bindings";
  
  py::class_<fastllm::GenerationConfig>(m, "GenerationConfig")
	  .def(py::init<>())
	  .def_readwrite("max_length", &fastllm::GenerationConfig::output_token_limit) 
	  .def_readwrite("last_n", &fastllm::GenerationConfig::last_n) 
	  .def_readwrite("repeat_penalty", &fastllm::GenerationConfig::repeat_penalty) 
	  .def_readwrite("top_k", &fastllm::GenerationConfig::top_k) 
	  .def_readwrite("top_p", &fastllm::GenerationConfig::top_p) 
	  .def_readwrite("temperature", &fastllm::GenerationConfig::temperature)
	  .def_readwrite("enable_hash_id", &fastllm::GenerationConfig::enable_hash_id)
	  .def("is_simple_greedy", &fastllm::GenerationConfig::IsSimpleGreedy); 

  // high level
  m.def("set_threads", &fastllm::SetThreads)
    .def("get_threads", &fastllm::GetThreads)
    .def("set_low_memory", &fastllm::SetLowMemMode)
    .def("get_low_memory", &fastllm::GetLowMemMode)
    .def("set_kv_cache", &fastllm::SetKVCacheInCPU)
    .def("get_kv_cache", &fastllm::GetKVCacheInCPU)
    .def("set_device_map", &fastllm::SetDeviceMap)
    .def("create_llm", &fastllm::CreateLLMModelFromFile);
  m.def("std_hash", [](std::string input) -> size_t {
		return std::hash<std::string>{}(input);
  }); 

  // low level
  m.def("get_llm_type", &fastllm::GetModelTypeFromFile);
  m.def("llm_sampling", &fastllm::LLMSampling)
    .def("embedding", &pyfastllm::Embedding)
    .def("rms_norm", &pyfastllm::RMSNorm)
    .def("layer_norm", &pyfastllm::LayerNorm)
    .def("linear", &pyfastllm::Linear)
    .def("split", &pyfastllm::Split)
    .def("cat", &pyfastllm::Cat)
    .def("cat_direct", &pyfastllm::CatDirect)
    .def("matmul", &pyfastllm::MatMul)
    // .def("matmul_transB", &fastllm::MatMulTransB)
    .def("softmax", &pyfastllm::Softmax)
    .def("silu", &pyfastllm::Silu)
    .def("gelu", &pyfastllm::Gelu)
    .def("swiglu", &pyfastllm::Swiglu)
    .def("mul", &pyfastllm::Mul)
    .def("attention", &pyfastllm::Attention)
    .def("mul_to", &fastllm::MulTo)
    .def("add_to", &fastllm::AddTo)
    .def("add", &pyfastllm::Add)
    // .def("attention_mask", &fastllm::AttentionMask)
    // .def("alibi_mask", &fastllm::AlibiMask)
    .def("permute", &pyfastllm::Permute)
    .def("permute_", &fastllm::PermuteSelf)
    .def("topk", &pyfastllm::TopK)
    .def("rotateposition2D", &pyfastllm::RotatePosition2D)
    .def("nearlyrotateposition2D", &pyfastllm::NearlyRotatePosition2D)
    .def("llama_rotateposition2D", &fastllm::LlamaRotatePosition2D)
    // .def("repeat_penalty", &fastllm::RepeatPenalty);
    // .def("load", &pyfastllm::LoadWeights)
    .def("from_numpy", &pyfastllm::fromNumpy<float>);

  py::enum_<fastllm::DataType>(m, "DataType")
    .value("float32", fastllm::DataType::FLOAT32)
    .value("bfloat16", fastllm::DataType::BFLOAT16)
    .value("int16", fastllm::DataType::INT16)
    .value("int8", fastllm::DataType::INT8)
    .value("int4", fastllm::DataType::INT4)
    .value("int2", fastllm::DataType::INT2)
    .value("float16", fastllm::DataType::FLOAT16)
    .value("bit", fastllm::DataType::BIT)
    .value("int32param", fastllm::DataType::INT32PARAM)
    .export_values();

  py::class_<fastllm::Data>(m, "Tensor", py::buffer_protocol())
    .def_buffer([](fastllm::Data &m) -> py::buffer_info {
        std::vector<uint64_t> newStrides(std::move(m.strides));
        for(auto &stride:newStrides){
          stride *= sizeof(float);
        }
        return py::buffer_info(
            m.cpuData,                               /* Pointer to buffer */
            sizeof(float),                          /* Size of one scalar */
            py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
            m.dims.size(),                                      /* Number of dimensions */
            m.dims,                 /* Buffer dimensions */
            newStrides
            // { sizeof(float) * m.dims[1],             /* Strides (in bytes) for each index */
            //   sizeof(float) }
        );
     })
    .def_readonly("shape", &fastllm::Data::dims)
    .def_readonly("expansionDims", &fastllm::Data::expansionDims)
    .def(py::init<>())
    .def(py::init<fastllm::DataType>())
    .def(py::init<fastllm::DataType, const std::vector<int>&>())
    .def(py::init<fastllm::DataType, const std::vector<int>&, const std::vector<float>&>())
    .def(py::init<fastllm::Data>())
    .def("copy_from", &fastllm::Data::CopyFrom)
    .def("count", &fastllm::Data::Count)
    .def("reshape", &fastllm::Data::Reshape)
    .def("expansion", &fastllm::Data::Expansion)
    .def("to_list", [](fastllm::Data& data){
      std::vector <float> vecData;
      for (int i = 0; i < data.Count(0); i++) {
            vecData.push_back(((float*)data.cpuData)[i]);
        }
        return vecData;
    })
    .def("__str__", &pyfastllm::String)
    .def("size", &pyfastllm::GetDims)
    .def("size", &pyfastllm::GetSize)
    .def("to", &pyfastllm::ToDevice)
    .def("cuda", &pyfastllm::ToCuda)
    .def("cpu", &pyfastllm::ToCpu);

  m.def("zeros", [](const std::vector<int> &dims, fastllm::DataType dtype)->fastllm::Data {
    int nums = 1;
    for (auto dim:dims){nums *= dim; } 
    std::vector<float>zero_data(nums, 0);
    auto data = fastllm::Data(dtype, dims, zero_data);
    return data;
  }, py::arg("dims"), py::arg("dtype"));


  py::class_<fastllm::Tokenizer>(m, "Tokenizer")
    .def_readonly("add_dummy_prefix", &fastllm::Tokenizer::addDummyPrefix)
    .def_readonly("remove_extra_whitespaces", &fastllm::Tokenizer::removeExtraWhitespaces)
    .def_readonly("byte_as_char", &fastllm::Tokenizer::byteAsChar)
    .def("encode", &fastllm::Tokenizer::Encode)
    // .def("decode", &fastllm::Tokenizer::Decode)
    .def("decode", &fastllm::Tokenizer::Decode, "Decode from Tensor")
    .def("decode", &fastllm::Tokenizer::DecodeTokens, "Decode from Vector")
    .def("decode_byte", [](fastllm::Tokenizer &tokenizer, const fastllm::Data &data){
      std::string ret = tokenizer.Decode(data);
      return py::bytes(ret);
    })
    .def("decode_byte", [](fastllm::Tokenizer &tokenizer, const std::vector<int>& data){
      std::string ret = tokenizer.DecodeTokens(data);
      return py::bytes(ret);
    })
    .def("clear", &fastllm::Tokenizer::Clear)
    .def("insert", &fastllm::Tokenizer::Insert)
    .def("set_special_tokens", &fastllm::Tokenizer::SetSpecialTokens);
  
  py::class_<fastllm::WeightMap>(m, "WeightMap")
    .def(py::init<>())
    .def_readonly("tokenizer", &fastllm::WeightMap::tokenizer)
    .def("load", &fastllm::WeightMap::LoadFromFile)
    .def("save_lowbit", &fastllm::WeightMap::SaveLowBitModel)
    .def("set_kv", &fastllm::WeightMap::AddDict)
    .def("set_weight", &fastllm::WeightMap::AddWeight)
    .def("__getitem__", [](fastllm::WeightMap &weight, std::string key){
        fastllm::Data &data = weight[key];
        return data; 
    }, py::return_value_policy::take_ownership)
    .def("keys", [](fastllm::WeightMap &weight){
      std::vector<std::string> keys;
      for (auto iter:weight.weight){
        keys.push_back(iter.first);
      }
      return keys;
    });
    
  py::class_<pyfastllm::Tensor>(m, "Tensor_", py::buffer_protocol())
    .def(py::init<>())
    .def(py::init<fastllm::DataType>())
    .def(py::init<fastllm::DataType, const std::vector<int>&>())
    .def(py::init<fastllm::DataType, const std::vector<int>&, const std::vector<float>&>())
    // .def(py::init<fastllm::Data>())
    .def_buffer(&pyfastllm::Tensor::MemBuffer)
    // .def("size", py::overload_cast<>(&pyfastllm::Tensor::GetDims))
    .def("size", &pyfastllm::Tensor::GetSize)
    .def("size", &pyfastllm::Tensor::GetDims)
    .def("expand_size", &pyfastllm::Tensor::GetExpansionDims)
    .def("count", &pyfastllm::Tensor::Count)
    .def("reshape", &pyfastllm::Tensor::Reshape)
    .def("expansion", &pyfastllm::Tensor::Expansion)
    .def("to_list", &pyfastllm::Tensor::ToList)
    .def("__str__", &pyfastllm::Tensor::HostString);
    // .def("numpy", &pyfastllm::ToNumpy<float>)
    // .def("to", &pyfastllm::ToDevice);

  // model classes
  py::class_<fastllm::basellm>(m, "basellm");

  py::class_<fastllm::ChatGLMModel, fastllm::basellm>(m, "ChatGLMModel")
    .def(py::init<>())
    .def_readonly("model_type", &fastllm::ChatGLMModel::model_type)
    .def_readonly("weight", &fastllm::ChatGLMModel::weight)
    .def_readonly("block_cnt", &fastllm::ChatGLMModel::block_cnt)
    .def_readonly("bos_token_id", &fastllm::ChatGLMModel::bos_token_id)
    .def_readonly("eos_token_id", &fastllm::ChatGLMModel::eos_token_id)
    .def_readonly("gmask_token_id", &fastllm::ChatGLMModel::gmask_token_id)
    .def("load_weights", &fastllm::ChatGLMModel::LoadFromFile)
    .def("make_input", &fastllm::ChatGLMModel::MakeInput)
    .def("make_history", &fastllm::ChatGLMModel::MakeHistory)
    .def("response", &fastllm::ChatGLMModel::Response)
    .def("batch_response", [](fastllm::ChatGLMModel &model, 
                              const std::vector <std::string> &inputs,
                               RuntimeResultBatch retCb,
							   fastllm::GenerationConfig config)->std::vector<std::string> {
      std::vector <std::string> outputs;
      model.ResponseBatch(inputs, outputs, retCb, config);
      return outputs;
    })
    .def("warmup", &fastllm::ChatGLMModel::WarmUp)
    .def("forward",
        [](fastllm::ChatGLMModel &model, 
           const fastllm::Data &inputIds, 
           const fastllm::Data &attentionMask,
           const fastllm::Data &positionIds, std::vector<std::pair<fastllm::Data, fastllm::Data>> &pastKeyValues,
           const fastllm::GenerationConfig &generationConfig, const fastllm::LastTokensManager &tokens) {

          int retV = model.Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
          return std::make_tuple(retV, pastKeyValues);
    })
    .def("launch_response", &fastllm::ChatGLMModel::LaunchResponseTokens)
    .def("fetch_response", &fastllm::ChatGLMModel::FetchResponseTokens)
    .def("save_lowbit_model", &fastllm::ChatGLMModel::SaveLowBitModel)
    .def("make_input", &fastllm::ChatGLMModel::MakeInput);

  py::class_<fastllm::MOSSModel, fastllm::basellm>(m, "MOSSModel")
    .def(py::init<>())
    .def_readonly("model_type", &fastllm::MOSSModel::model_type)
    .def_readonly("weight", &fastllm::MOSSModel::weight)
    .def_readonly("block_cnt", &fastllm::MOSSModel::block_cnt)
    .def_readonly("bos_token_id", &fastllm::MOSSModel::bos_token_id)
    .def_readonly("eos_token_id", &fastllm::MOSSModel::eos_token_id)
    .def("load_weights", &fastllm::MOSSModel::LoadFromFile)
    .def("make_input", &fastllm::MOSSModel::MakeInput)
    .def("make_history", &fastllm::MOSSModel::MakeHistory)
    .def("response", &fastllm::MOSSModel::Response)
    .def("batch_response", [](fastllm::MOSSModel &model, 
                              const std::vector <std::string> &inputs,
                               RuntimeResultBatch retCb,
							   fastllm::GenerationConfig config)->std::vector<std::string> {
      std::vector <std::string> outputs;
      model.ResponseBatch(inputs, outputs, retCb, config);
      return outputs;
    })
    .def("forward",
        [](fastllm::MOSSModel &model, 
           const fastllm::Data &inputIds, 
           const fastllm::Data &attentionMask,
           const fastllm::Data &positionIds, std::vector<std::pair<fastllm::Data, fastllm::Data>> &pastKeyValues,
           const fastllm::GenerationConfig &generationConfig, const fastllm::LastTokensManager &tokens) {
          int retV = model.Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
          return std::make_tuple(retV, pastKeyValues);
    })
    .def("launch_response", &fastllm::MOSSModel::LaunchResponseTokens)
    .def("fetch_response", &fastllm::MOSSModel::FetchResponseTokens)
    .def("save_lowbit_model", &fastllm::MOSSModel::SaveLowBitModel)
    .def("make_input", &fastllm::MOSSModel::MakeInput);

  py::class_<fastllm::LlamaModel, fastllm::basellm>(m, "LlamaModel")
    .def(py::init<>())
    .def_readonly("model_type", &fastllm::LlamaModel::model_type)
    .def_readonly("weight", &fastllm::LlamaModel::weight)
    .def_readonly("block_cnt", &fastllm::LlamaModel::block_cnt)
    .def_readonly("bos_token_id", &fastllm::LlamaModel::bos_token_id)
    .def_readonly("eos_token_id", &fastllm::LlamaModel::eos_token_id)
    .def("load_weights", &fastllm::LlamaModel::LoadFromFile)
    .def("make_input", &fastllm::LlamaModel::MakeInput)
    .def("make_history", &fastllm::LlamaModel::MakeHistory)
    .def("response", &fastllm::LlamaModel::Response)
    .def("batch_response", [](fastllm::LlamaModel &model, 
                              const std::vector <std::string> &inputs,
                               RuntimeResultBatch retCb,
							   fastllm::GenerationConfig config)->std::vector<std::string> {
      std::vector <std::string> outputs;
      model.ResponseBatch(inputs, outputs, retCb, config);
      return outputs;
    })
    .def("warmup", &fastllm::LlamaModel::WarmUp)
    .def("forward",
        [](fastllm::LlamaModel &model, 
           const fastllm::Data &inputIds, 
           const fastllm::Data &attentionMask,
           const fastllm::Data &positionIds, std::vector<std::pair<fastllm::Data, fastllm::Data>> &pastKeyValues,
           const fastllm::GenerationConfig &generationConfig, const fastllm::LastTokensManager &tokens) {
          int retV = model.Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
          return std::make_tuple(retV, pastKeyValues);
    })
    .def("launch_response", &fastllm::LlamaModel::LaunchResponseTokens)
    .def("fetch_response", &fastllm::LlamaModel::FetchResponseTokens)
    .def("save_lowbit_model", &fastllm::LlamaModel::SaveLowBitModel)
    .def("make_input", &fastllm::LlamaModel::MakeInput);

  py::class_<fastllm::QWenModel, fastllm::basellm>(m, "QWenModel")
    .def(py::init<>())
    .def_readonly("model_type", &fastllm::QWenModel::model_type)
    .def_readonly("weight", &fastllm::QWenModel::weight)
    .def_readonly("block_cnt", &fastllm::QWenModel::block_cnt)
    .def_readonly("bos_token_id", &fastllm::QWenModel::bos_token_id)
    .def_readonly("eos_token_id", &fastllm::QWenModel::eos_token_id)
    .def("load_weights", &fastllm::QWenModel::LoadFromFile)
    .def("make_input", &fastllm::QWenModel::MakeInput)
    .def("make_history", &fastllm::QWenModel::MakeHistory)
    .def("response", &fastllm::QWenModel::Response)
    .def("batch_response", [](fastllm::QWenModel &model, 
                                const std::vector <std::string> &inputs,
                                RuntimeResultBatch retCb,
                                fastllm::GenerationConfig config)->std::vector<std::string> {
        std::vector <std::string> outputs;
        model.ResponseBatch(inputs, outputs, retCb, config);
        return outputs;
    })
    .def("warmup", &fastllm::QWenModel::WarmUp)
    .def("forward",
        [](fastllm::QWenModel &model, 
            const fastllm::Data &inputIds, 
            const fastllm::Data &attentionMask,
            const fastllm::Data &positionIds, std::vector<std::pair<fastllm::Data, fastllm::Data>> &pastKeyValues,
            const fastllm::GenerationConfig &generationConfig, const fastllm::LastTokensManager &tokens) {

            int retV = model.Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
            return std::make_tuple(retV, pastKeyValues);
    })
    .def("launch_response", &fastllm::QWenModel::LaunchResponseTokens)
    .def("fetch_response", &fastllm::QWenModel::FetchResponseTokens)
    .def("save_lowbit_model", &fastllm::QWenModel::SaveLowBitModel)
    .def("make_input", &fastllm::QWenModel::MakeInput);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}

// #endif
