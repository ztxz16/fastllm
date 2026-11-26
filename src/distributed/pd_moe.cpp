#include "pd_bindings.h"

namespace fastllm {
namespace pd {
/** 按 rank 合并控制请求，一个原生联合轮次至多提交每个 rank 的一条命令。 */
void BindMoe(Handlers &h) {
    h["moe.MoePDGroup.__init__"] = [](D a) {
        O s = self(a), transport = cls("moe", "RoundTransport")(*at(a, "args"), **at(a, "kwargs"));
        put(s, "transport", transport);
        put(s, "config",
            module("json").attr("loads")(
                transport.attr("model").attr("__truediv__")("config.json").attr("read_text")()));
        put(s, "condition", module("threading").attr("Condition")());
        put(s, "pending", module("collections").attr("deque")());
        put(s, "stopping", false);
        put(s, "failure", none());
        put(s, "views", L());
        return none();
    };

    h["moe.MoePDGroup.__enter__"] = [](D a) {
        O s = self(a), transport = get(s, "transport");
        transport.attr("__enter__")();
        if (!equal(get(transport, "ready").attr("get")("pd_storage_protocol"), py::int_(1))) {
            transport.attr("close")();
            fail("MoE coordinator needs PD protocol; rebuild on server", PyExc_RuntimeError);
        }
        L views;
        for (I rank = 0; rank < I(py::len(get(transport, "devices"))); ++rank)
            views.append(cls("moe", "MoePDView")(s, rank));
        put(s, "views", views);
        O thread = module("threading")
                       .attr("Thread")("target"_a = get(s, "_run"), "name"_a = "pd-ep-joint-rounds",
                                       "daemon"_a = true);
        put(s, "thread", thread);
        thread.attr("start")();
        return s;
    };

    h["moe.MoePDGroup._run"] = [](D a) {
        O s = self(a);
        L batch;
        try {
            while (true) {
                {
                    Lock lock(get(s, "condition"));
                    get(s, "condition").attr("wait_for")(py::cpp_function([s] {
                        return truth(get(s, "pending")) || truth(get(s, "stopping"));
                    }));
                    if (truth(get(s, "stopping")) && !truth(get(s, "pending")))
                        return none();
                    batch = L();
                    std::set<I> occupied;
                    O rest = module("collections").attr("deque")();
                    while (truth(get(s, "pending"))) {
                        O call = get(s, "pending").attr("popleft")();
                        if (occupied.insert(num(get(call, "rank"))).second)
                            batch.append(call);
                        else
                            rest.attr("append")(call);
                    }
                    put(s, "pending", rest);
                }
                L slots;
                for (size_t i = 0; i < py::len(get(s, "views")); ++i)
                    slots.append(none());
                for (auto call : batch)
                    slots[num(get(call, "rank"))] = get(call, "request");
                O reply = get(s, "transport").attr("exchange")("pd", "requests"_a = slots),
                  results = reply.attr("get")("results");
                require(py::isinstance<L>(results) && py::len(results) == py::len(slots),
                        "invalid MoE PD joint reply");
                for (auto call : batch) {
                    put(call, "result", item(results, get(call, "rank")));
                    get(call, "done").attr("set")();
                }
                batch = L();
            }
        } catch (const std::exception &error) {
            O failure = module("builtins").attr("RuntimeError")(error.what());
            Lock lock(get(s, "condition"));
            put(s, "failure", failure);
            for (auto call : batch) {
                put(call, "error", failure);
                get(call, "done").attr("set")();
            }
            for (auto call : get(s, "pending")) {
                put(call, "error", failure);
                get(call, "done").attr("set")();
            }
            get(s, "pending").attr("clear")();
            get(s, "condition").attr("notify_all")();
        }
        return none();
    };

    h["moe.MoePDGroup.call"] = [](D a) {
        O s = self(a), call = cls("moe", "Call")(at(a, "rank"), at(a, "request"));
        {
            Lock lock(get(s, "condition"));
            if (truth(get(s, "stopping")) || !get(s, "failure").is_none())
                fail("EP coordinator unavailable", PyExc_RuntimeError);
            get(s, "pending").attr("append")(call);
            get(s, "condition").attr("notify_all")();
        }
        get(call, "done").attr("wait")();
        if (!get(call, "error").is_none())
            fail("EP joint operation failed: " + str(get(call, "error")), PyExc_RuntimeError);
        return get(call, "result");
    };

    h["moe.MoePDGroup.__exit__"] = [](D a) {
        O s = self(a);
        {
            Lock lock(get(s, "condition"));
            put(s, "stopping", true);
            get(s, "condition").attr("notify_all")();
        }
        get(s, "thread").attr("join")();
        if (!get(s, "failure").is_none())
            get(s, "transport").attr("_dispose")("failed");
        else
            get(s, "transport").attr("close")();
        return none();
    };

    h["moe.MoePDView.__init__"] = [](D a) {
        O s = self(a), owner = at(a, "owner"), transport = get(owner, "transport");
        put(s, "owner", owner);
        put(s, "rank", at(a, "rank"));
        L devices;
        devices.append(item(get(transport, "devices"), at(a, "rank")));
        put(s, "devices", devices);
        put(s, "timeout", get(transport, "timeout"));
        put(s, "ready", get(transport, "ready"));
        return none();
    };

    h["moe.MoePDView.state"] = [](D a) -> O {
        O s = self(a), owner = get(s, "owner");
        return get(owner, "failure").is_none() ? get(get(owner, "transport"), "state") : py::str("failed");
    };

    h["moe.MoePDView.pd_command"] = [](D a) {
        O s = self(a);
        return get(s, "owner").attr("call")(get(s, "rank"), at(a, "request"));
    };

    h["moe.MoePDView.generate"] = [](D a) -> O {
        O s = self(a), request = at(a, "request"), context = uuid(), cancel = at(a, "cancel_event"),
          callback = at(a, "on_token");
        cls("moe", "validate_request")(request, at(get(get(s, "owner"), "config"), "vocab_size"));
        D sampling;
        for (auto pair : D("temperature"_a = 0.0, "top_k"_a = 0, "top_p"_a = 1.0, "seed"_a = 0))
            sampling[pair.first] = request.attr("get")(pair.first, pair.second);
        L output;
        auto cleanup = [&] {
            if (str(get(s, "state")) == "ready") {
                s.attr("pd_command")(D("action"_a = "end"));
                s.attr("pd_command")(D("action"_a = "sampling_end", "context"_a = context));
            }
        };
        try {
            for (I index = 0; index < num(at(request, "max_new_tokens")); ++index) {
                if (!cancel.is_none() && truth(cancel.attr("is_set")()))
                    fail("full MoE request cancelled", PyExc_InterruptedError);
                D command = index == 0
                                ? D("action"_a = "full_prefill", "token_ids"_a = at(request, "token_ids"),
                                    "capacity"_a = I(py::len(at(request, "token_ids"))) +
                                                   num(at(request, "max_new_tokens")) - 1)
                                : D("action"_a = "step", "input"_a = output[py::len(output) - 1]);
                command["first"] = py::bool_(index == 0);
                command["context"] = context;
                command["sampling"] = sampling;
                O token = at(s.attr("pd_command")(command), "token");
                output.append(token);
                if (!callback.is_none())
                    callback(token);
                if (truth(request.attr("get")("eos_token_ids", L()).attr("__contains__")(token)))
                    break;
            }
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
        return D("token_ids"_a = output);
    };
}
} // namespace pd
} // namespace fastllm
