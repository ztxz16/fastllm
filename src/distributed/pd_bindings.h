#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fastllm {
namespace pd {
namespace py = pybind11;
using namespace pybind11::literals;
using O = py::object;
using D = py::dict;
using L = py::list;
using T = py::tuple;
using I = int64_t;
using Handler = std::function<O(D)>;
using Handlers = std::map<std::string, Handler>;

inline O module(const char *name) {
    return py::module_::import(name);
}
inline O get(py::handle o, const char *key) {
    return py::getattr(o, key);
}
inline O item(py::handle o, py::handle key) {
    return py::reinterpret_borrow<O>(o)[key];
}
inline O at(py::handle o, const char *key) {
    return item(o, py::str(key));
}
inline O optional(D d, const char *key, O fallback = py::none()) {
    return d.contains(key) ? at(d, key) : fallback;
}
inline std::string str(py::handle o) {
    return py::str(o).cast<std::string>();
}
inline I num(py::handle o) {
    return py::cast<I>(o);
}
inline double real(py::handle o) {
    return py::cast<double>(o);
}
inline bool truth(py::handle o) {
    int value = PyObject_IsTrue(o.ptr());
    if (value < 0)
        throw py::error_already_set();
    return value != 0;
}
inline bool equal(py::handle a, py::handle b) {
    int value = PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_EQ);
    if (value < 0)
        throw py::error_already_set();
    return value != 0;
}
inline O tuple(py::handle o) {
    return module("builtins").attr("tuple")(o);
}
inline O list(py::handle o) {
    return module("builtins").attr("list")(o);
}
inline O asdict(py::handle o) {
    return module("dataclasses").attr("asdict")(o);
}
inline O replace(py::handle o, D changes) {
    return module("dataclasses").attr("replace")(o, **changes);
}
std::string &Package();
inline O cls(const char *mod, const char *name) {
    return module((Package() + "." + mod).c_str()).attr(name);
}
O Invoke(const std::string &name, D arguments);
inline void put(py::handle o, const char *key, O value) {
    py::setattr(o, key, value);
}
inline void put(py::handle o, const char *key, I value) {
    put(o, key, py::int_(value));
}
inline void put(py::handle o, const char *key, bool value) {
    put(o, key, py::bool_(value));
}
inline O self(D a) {
    return at(a, "self");
}
inline O none() {
    return py::none();
}
inline double now() {
    return real(module("time").attr("monotonic")());
}
inline double wall() {
    return real(module("time").attr("time")());
}
inline O uuid() {
    return module("uuid").attr("uuid4")().attr("hex");
}
inline void fail(const std::string &message, PyObject *type = PyExc_ValueError) {
    PyErr_SetString(type, message.c_str());
    throw py::error_already_set();
}
inline void require(bool condition, const std::string &message) {
    if (!condition)
        fail(message);
}
inline void protocol(bool condition, const std::string &message) {
    if (!condition) {
        auto type = cls("protocol", "ProtocolError");
        PyErr_SetString(type.ptr(), message.c_str());
        throw py::error_already_set();
    }
}

/** 持有接口对象的可重入锁；Python 适配器负责线程/GIL，状态转换只在 C++ 中执行。 */
struct Lock {
    O lock;
    explicit Lock(O value) : lock(std::move(value)) {
        lock.attr("acquire")();
    }
    ~Lock() {
        try {
            lock.attr("release")();
        } catch (...) {
            PyErr_Clear();
        }
    }
};

I integer(O value, const std::string &name, I low = 0, I high = std::numeric_limits<I>::max());
O identifier(O value, const std::string &name);
O plan(O source, O target);
void BindProtocol(Handlers &handlers);
void BindRouting(Handlers &handlers);
void BindResources(Handlers &handlers);
void BindTasks(Handlers &handlers);
void BindNative(Handlers &handlers);
void BindMoe(Handlers &handlers);
void BindService(Handlers &handlers);
void BindRouter(Handlers &handlers);
void BindCuda(py::module_ &module);
} // namespace pd
} // namespace fastllm
