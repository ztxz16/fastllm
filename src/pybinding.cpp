#include "factoryllm.h"
#include "utils.h"

#ifdef PY_API
#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace pybind11::literals;  
#endif


const std::string VERSION_INFO = "0.0.1";

#ifdef PY_API

PYBIND11_MODULE(pyfastllm, m) {
  m.doc() = "fastllm python bindings";
  
  m.def("set_threads", &fastllm::SetThreads)
    .def("get_threads", &fastllm::GetThreads)
    .def("set_low_memory", &fastllm::SetLowMemMode)
    .def("get_low_memory", &fastllm::GetLowMemMode);


  py::class_<fastllm::ChatGLMModel>(m, "ChatGLMModel")
    .def(py::init<>())
    .def("load_weights", &fastllm::ChatGLMModel::LoadFromFile)
    .def("response", &fastllm::ChatGLMModel::Response)
    .def("warmup", &fastllm::ChatGLMModel::WarmUp)
    .def("save_lowbit_model", &fastllm::ChatGLMModel::SaveLowBitModel);

  py::class_<fastllm::MOSSModel>(m, "MOSSModel")
    .def(py::init<>())
    .def("load_weights", &fastllm::MOSSModel::LoadFromFile)
    .def("response", &fastllm::MOSSModel::Response)
    .def("save_lowbit_model", &fastllm::MOSSModel::SaveLowBitModel);

  py::class_<fastllm::VicunaModel>(m, "VicunaModel")
    .def(py::init<>())
    .def("load_weights", &fastllm::VicunaModel::LoadFromFile)
    .def("response", &fastllm::VicunaModel::Response)
    .def("warmup", &fastllm::VicunaModel::WarmUp)
    .def("save_lowbit_model", &fastllm::VicunaModel::SaveLowBitModel);
  
  py::class_<fastllm::BaichuanModel>(m, "BaichuanModel")
    .def(py::init<>())
    .def("load_weights", &fastllm::BaichuanModel::LoadFromFile)
    .def("response", &fastllm::BaichuanModel::Response)
    .def("warmup", &fastllm::BaichuanModel::WarmUp)
    .def("save_lowbit_model", &fastllm::BaichuanModel::SaveLowBitModel);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}

#endif