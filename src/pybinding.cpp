#include "factoryllm.h"
#include "utils.h"

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
    .def("causal_mask", &fastllm::MOSSModel::WarmUp)
    .def("warmup", &fastllm::MOSSModel::WarmUp)
    .def("save_lowbit_model", &fastllm::MOSSModel::SaveLowBitModel);

  py::class_<fastllm::VicunaModel>(m, "VicunaModel")
    .def(py::init<>())
    .def("load_weights", &fastllm::VicunaModel::LoadFromFile)
    .def("response", &fastllm::VicunaModel::Response)
    .def("causal_mask", &fastllm::VicunaModel::WarmUp)
    .def("warmup", &fastllm::VicunaModel::WarmUp)
    .def("save_lowbit_model", &fastllm::VicunaModel::SaveLowBitModel);
}

#endif