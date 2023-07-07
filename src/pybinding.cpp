#include "model.h"
#include "factoryllm.h"

#ifdef PY_API
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace pybind11::literals;  
#endif


#ifdef PY_API

// PYBIND11_MAKE_OPAQUE(std::vector<std::pair<fastllm::Data,fastllm::Data>>);
PYBIND11_MAKE_OPAQUE(fastllm::Data);

PYBIND11_MODULE(pyfastllm, m) {
  m.doc() = "fastllm python bindings";
  
  // high level
  m.def("set_threads", &fastllm::SetThreads)
    .def("get_threads", &fastllm::GetThreads)
    .def("set_low_memory", &fastllm::SetLowMemMode)
    .def("get_low_memory", &fastllm::GetLowMemMode)
    .def("create_llm", &fastllm::CreateLLMModelFromFile);
  
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
  
  py::class_<fastllm::WeightMap>(m, "WeightMap")
    .def_readonly("tokenizer", &fastllm::WeightMap::tokenizer);

  py::class_<fastllm::Tokenizer>(m, "Tokenizer")
    .def("encode", &fastllm::Tokenizer::Encode)
    .def("decode", &fastllm::Tokenizer::Decode)
    .def("decode_byte", [](fastllm::Tokenizer &tokenizer, const fastllm::Data &data){
      std::string ret = tokenizer.Decode(data);
      return py::bytes(ret);
    });
    // .def("clear", &fastllm::Tokenizer::Clear)
    // .def("insert", &fastllm::Tokenizer::Insert);


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
    .def("batch_response", &fastllm::ChatGLMModel::ResponseBatch)
    .def("warmup", &fastllm::ChatGLMModel::WarmUp)
    /*.def("__call__",
        [](fastllm::ChatGLMModel &model, 
           const fastllm::Data &inputIds, 
           const fastllm::Data &attentionMask,
           const fastllm::Data &positionIds, 
           const fastllm::Data &penaltyFactor,
           std::vector<std::pair<fastllm::Data, fastllm::Data>> &pastKeyValues) {
          int retV = model.Forward(inputIds, attentionMask, positionIds, penaltyFactor, pastKeyValues);
          return std::make_tuple(retV, pastKeyValues);
    })*/
    .def("launch_response", &fastllm::ChatGLMModel::LaunchResponseTokens)
    .def("fetch_response", &fastllm::ChatGLMModel::FetchResponseTokens)
    .def("save_lowbit_model", &fastllm::ChatGLMModel::SaveLowBitModel);

  py::class_<fastllm::MOSSModel, fastllm::basellm>(m, "MOSSModel")
    .def(py::init<>())
    .def_readonly("model_type", &fastllm::MOSSModel::model_type)
    .def_readonly("weight", &fastllm::MOSSModel::weight)
    .def_readonly("block_cnt", &fastllm::MOSSModel::block_cnt)
    .def_readonly("bos_token_id", &fastllm::MOSSModel::bos_token_id)
    .def_readonly("eos_token_id", &fastllm::MOSSModel::eos_token_id)
    .def("load_weights", &fastllm::MOSSModel::LoadFromFile)
    .def("response", &fastllm::MOSSModel::Response)
    .def("batch_response", &fastllm::MOSSModel::ResponseBatch)
    /*.def("__call__",
        [](fastllm::MOSSModel &model, 
           const fastllm::Data &inputIds, 
           const fastllm::Data &attentionMask,
           const fastllm::Data &positionIds, 
           const fastllm::Data &penaltyFactor,
           std::vector<std::pair<fastllm::Data, fastllm::Data>> &pastKeyValues) {
          int retV = model.Forward(inputIds, attentionMask, positionIds, penaltyFactor, pastKeyValues);
          return std::make_tuple(retV, pastKeyValues);
    })*/
    .def("launch_response", &fastllm::MOSSModel::LaunchResponseTokens)
    .def("fetch_response", &fastllm::MOSSModel::FetchResponseTokens)
    .def("save_lowbit_model", &fastllm::MOSSModel::SaveLowBitModel);

  py::class_<fastllm::LlamaModel, fastllm::basellm>(m, "LlamaModel")
    .def(py::init<>())
    .def_readonly("model_type", &fastllm::LlamaModel::model_type)
    .def_readonly("weight", &fastllm::LlamaModel::weight)
    .def_readonly("block_cnt", &fastllm::LlamaModel::block_cnt)
    .def_readonly("bos_token_id", &fastllm::LlamaModel::bos_token_id)
    .def_readonly("eos_token_id", &fastllm::LlamaModel::eos_token_id)
    .def("load_weights", &fastllm::LlamaModel::LoadFromFile)
    .def("response", &fastllm::LlamaModel::Response)
    .def("batch_response", &fastllm::LlamaModel::ResponseBatch)
    .def("warmup", &fastllm::LlamaModel::WarmUp)
    /*.def("__call__",
        [](fastllm::LlamaModel &model, 
           const fastllm::Data &inputIds, 
           const fastllm::Data &attentionMask,
           const fastllm::Data &positionIds, 
           const fastllm::Data &penaltyFactor,
           std::vector<std::pair<fastllm::Data, fastllm::Data>> &pastKeyValues) {
          int retV = model.Forward(inputIds, attentionMask, positionIds, penaltyFactor, pastKeyValues);
          return std::make_tuple(retV, pastKeyValues);
    })*/
    .def("launch_response", &fastllm::LlamaModel::LaunchResponseTokens)
    .def("fetch_response", &fastllm::LlamaModel::FetchResponseTokens)
    .def("save_lowbit_model", &fastllm::LlamaModel::SaveLowBitModel);
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}

#endif