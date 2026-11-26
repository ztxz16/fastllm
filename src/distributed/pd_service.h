#pragma once
#include "pd_bindings.h"
namespace fastllm {
namespace pd {
inline bool flag(O o, const char *key) {
    return truth(get(o, key));
}
inline bool cancelled(O r) {
    return truth(get(r, "cancelled").attr("is_set")());
}
inline O decode(O value) {
    return module("json").attr("loads")(value);
}
inline O encode(O value) {
    return cls("http", "dumps")(value);
}
inline O rpc(O s, O url, const char *path, O body = py::none(), double timeout = -1) {
    return cls("http", "rpc")(url, path, body, "token"_a = get(s, "token"),
                              "timeout"_a = (timeout < 0 ? real(get(s, "timeout")) : timeout));
}
inline double deadline(O s, O r) {
    return now() + std::min(real(get(s, "timeout")), std::max(0.0, real(get(r, "expires_at")) - wall()));
}
inline O recordIdentity(O r) {
    return get(r, "identity");
}
inline O fields(O r) {
    return D("identity"_a = asdict(recordIdentity(r)));
}
inline void start(O callback, const char *name, T args = py::tuple()) {
    module("threading")
        .attr("Thread")("target"_a = callback, "args"_a = args, "daemon"_a = true, "name"_a = name)
        .attr("start")();
}
void BindServiceTransfer(Handlers &h);
void BindServiceOutput(Handlers &h);
} // namespace pd
} // namespace fastllm
