#include "pd_bindings.h"

namespace fastllm {
namespace pd {
/** 建立唯一原生调度表；未知入口拒绝，不执行 Python 核心回退。 */
Handlers MakeHandlers() {
    Handlers result;
    BindProtocol(result);
    BindRouting(result);
    BindResources(result);
    BindTasks(result);
    BindNative(result);
    BindMoe(result);
    BindService(result);
    BindRouter(result);
    return result;
}
std::string &Package() {
    static std::string package = "fastllm_pytools.pd";
    return package;
}
O Invoke(const std::string &name, D arguments) {
    static const Handlers handlers = MakeHandlers();
    auto found = handlers.find(name);
    if (found == handlers.end())
        throw std::runtime_error("unknown native PD entry: " + name);
    return found->second(std::move(arguments));
}
} // namespace pd
} // namespace fastllm

PYBIND11_MODULE(_pd_core, module) {
    using namespace fastllm::pd;
    module.attr("abi_version") = 1;
    module.attr("implementation") = "cpp";
    module.attr("compiler") = __VERSION__;
    module.attr("cpp_standard") = __cplusplus;
#if defined(__x86_64__)
    module.attr("host_arch") = "x86_64";
#elif defined(__aarch64__)
    module.attr("host_arch") = "aarch64";
#else
    module.attr("host_arch") = "other";
#endif
    module.def("configure", [](const std::string &package) {
        require(package == "fastllm_pytools.pd" || package == "ftllm.pd", "unsupported PD package name");
        Package() = package;
    });
    module.def("invoke", &Invoke);
    module.def("entries", [] {
        L names;
        for (const auto &entry : MakeHandlers())
            names.append(py::str(entry.first));
        return names;
    });
    module.def("cancel_stream", [](const std::string &name, D a) {
        if (a.contains("budget"))
            put(at(a, "budget"), "cancelled", true);
        if (a.contains("cancelled") && !at(a, "cancelled").is_none())
            at(a, "cancelled").attr("set")();
        if (name == "service.Service.generate" || name == "service.Service.generate_full") {
            O s = self(a), id = a.contains("identity")
                                    ? at(a, "identity")
                                    : cls("service", "identity_from")(at(at(a, "body"), "identity"));
            O r = get(s, "records").attr("get")(get(id, "transfer"));
            if (!r.is_none() && equal(get(r, "identity"), id))
                get(r, "cancelled").attr("set")();
        }
    });
    BindCuda(module);
}
