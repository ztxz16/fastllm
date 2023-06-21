#include "model.h"
#include "factoryllm.h"

#ifdef PY_API
#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace pybind11::literals;  
#endif


#ifdef PY_API

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


  py::class_<fastllm::basellm>(m, "basellm");

  py::class_<fastllm::ChatGLMModel, fastllm::basellm>(m, "ChatGLMModel")
    .def(py::init<>())
    .def("load_weights", &fastllm::ChatGLMModel::LoadFromFile)
    .def("response", &fastllm::ChatGLMModel::Response)
    .def("batch_response", &fastllm::ChatGLMModel::ResponseBatch)
    .def("warmup", &fastllm::ChatGLMModel::WarmUp)
    .def("save_lowbit_model", &fastllm::ChatGLMModel::SaveLowBitModel);

  py::class_<fastllm::MOSSModel, fastllm::basellm>(m, "MOSSModel")
    .def(py::init<>())
    .def("load_weights", &fastllm::MOSSModel::LoadFromFile)
    .def("response", &fastllm::MOSSModel::Response)
    .def("batch_response", &fastllm::MOSSModel::ResponseBatch)
    .def("save_lowbit_model", &fastllm::MOSSModel::SaveLowBitModel);

  py::class_<fastllm::LlamaModel, fastllm::basellm>(m, "LlamaModel")
    .def(py::init<>())
    .def("load_weights", &fastllm::LlamaModel::LoadFromFile)
    .def("response", &fastllm::LlamaModel::Response)
    .def("batch_response", &fastllm::LlamaModel::ResponseBatch)
    .def("warmup", &fastllm::LlamaModel::WarmUp)
    .def("save_lowbit_model", &fastllm::LlamaModel::SaveLowBitModel);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}

#endif