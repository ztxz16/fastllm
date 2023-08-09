#include "model.h"
#include "factoryllm.h"

#ifdef PY_API
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <unordered_map>

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

  py::enum_<fastllm::DataType>(m, "Dtype")
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

  py::class_<fastllm::Data>(m, "Tensor")
    .def_readonly("dims", &fastllm::Data::dims)
    .def(py::init<>())
    .def(py::init<fastllm::DataType>())
    .def(py::init<fastllm::DataType, const std::vector<int>&>())
    .def(py::init<fastllm::DataType, const std::vector<int>&, const std::vector<float>&>())
    .def(py::init<fastllm::Data>())
    .def("copy_from", &fastllm::Data::CopyFrom)
    .def("count", &fastllm::Data::Count)
    .def("to_list", [](fastllm::Data& data){
      std::vector <float> vecData;
      for (int i = 0; i < data.Count(0); i++) {
            vecData.push_back(((float*)data.cpuData)[i]);
        }
        return vecData;
    })
    .def("print", &fastllm::Data::Print)
    .def("to", static_cast<void (fastllm::Data::*)(void *device)>(&fastllm::Data::ToDevice));

  m.def("zeros", [](const std::vector<int> &dims, fastllm::DataType dtype)->fastllm::Data {
    int nums = 1;
    for (auto dim:dims){nums *= dim; } 
    std::vector<float>zero_data(nums, 0);
    auto data = fastllm::Data(dtype, dims, zero_data);
    return data;
  }, py::arg("dims"), py::arg("dtype"));

  m.def("cat", [](std::vector<fastllm::Data> datas, int dim)->fastllm::Data {
    // int pos_dim = 0;
    // // dim check
    // for (int i=0;i<datas[0].dims.size();i++){
    //   int cur_dim = datas[0].dims[i];
    //   for (auto data:datas){
    //     if (i == dim){
    //       pos_dim += data.dims[i];
    //       continue;
    //     }
    //     if (data.dims[i] != cur_dim){
    //       std::cout<<"dim not the same!!!"<<std::endl;
    //       return fastllm::Data();
    //     }
    //   }
    // }

    // auto newDims = datas[0].dims;
    // newDims[dim] = pos_dim;
    // TODO use memcpy cp data 
    // TODO add different dim cat

     std::vector <float> vecData;
     for (auto data:datas){
      for (int i = 0; i < data.Count(0); i++) {
            vecData.push_back(((float*)data.cpuData)[i]);
        }
     }
     int seqLen = vecData.size();
     return fastllm::Data(fastllm::DataType::FLOAT32, {1, seqLen}, vecData);
  });


  py::class_<fastllm::Tokenizer>(m, "Tokenizer")
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
    .def("insert", &fastllm::Tokenizer::Insert);
  
  py::class_<fastllm::WeightMap>(m, "WeightMap")
    .def_readonly("tokenizer", &fastllm::WeightMap::tokenizer)
    .def("save_lowbit", &fastllm::WeightMap::SaveLowBitModel)
    .def("set_kv", &fastllm::WeightMap::AddDict)
    .def("set_weight", &fastllm::WeightMap::AddWeight)
    .def("__getitem__", [](fastllm::WeightMap &weight, std::string key){
        return weight[key]; });


  // model classes
  py::class_<fastllm::basellm>(m, "basellm");

  py::class_<fastllm::ChatGLMModel, fastllm::basellm>(m, "ChatGLMModel")
    .def(py::init<>())
    .def_readonly("model_type", &fastllm::ChatGLMModel::model_type)
    .def_readonly("weight", &fastllm::ChatGLMModel::weight)
    .def_readonly("block_cnt", &fastllm::ChatGLMModel::block_cnt)
    .def_readonly("bos_token_id", &fastllm::ChatGLMModel::bos_token_id)
    .def_readonly("eos_token_id", &fastllm::ChatGLMModel::eos_token_id)
    .def("load_weights", &fastllm::ChatGLMModel::LoadFromFile)
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
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}

#endif
