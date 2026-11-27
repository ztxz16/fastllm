#include "pd_bindings.h"

namespace fastllm {
namespace pd {
static O state(const char *name) {
    return cls("tasks", "State").attr(name);
}
static bool terminal(O task) {
    auto value = get(task, "state");
    return equal(value, state("READY")) || equal(value, state("FAILED")) || equal(value, state("CANCELLED"));
}
static O task(O s, O identity) {
    O value = get(s, "_tasks").attr("get")(get(identity, "transfer"));
    protocol(!value.is_none() && equal(get(value, "identity"), identity),
             "unknown transfer or stale identity");
    return value;
}

/** 汇总实际传输事件；终态与引用解除只允许发生在所有已提交访问结束之后。 */
void BindTasks(Handlers &h) {
    h["tasks.TransferManager.__init__"] = [](D a) {
        auto s = self(a);
        require(real(at(a, "poll_interval")) > 0, "poll interval must be positive");
        for (const char *key : {"owner", "backend", "poll_interval"})
            put(s, key, at(a, key));
        put(s, "notifications", module("queue").attr("SimpleQueue")());
        put(s, "_tasks", D());
        put(s, "_condition", module("threading").attr("Condition")(module("threading").attr("RLock")()));
        put(s, "_closing", false);
        put(s, "_statistics", D("completed_batches"_a = 0, "copied_bytes"_a = 0, "gpu_copy_ms"_a = 0.0));
        O thread =
            module("threading")
                .attr("Thread")("target"_a = get(s, "_run"), "name"_a = "ft-pd-transfer", "daemon"_a = true);
        put(s, "_thread", thread);
        thread.attr("start")();
        return none();
    };

    h["tasks.TransferManager.register"] = [](D a) {
        auto s = self(a), source = at(a, "source"), target = at(a, "target"), id = get(source, "identity");
        O expected = get(id, str(get(id, "mode")) == "pull" ? "target" : "source");
        protocol(equal(get(s, "owner"), expected), "transfer initiated by wrong instance or boot");
        O copies = plan(source, target);
        Lock lock(get(s, "_condition"));
        if (truth(get(s, "_closing")))
            fail("transfer manager is closing", PyExc_RuntimeError);
        D tasks = get(s, "_tasks");
        O old = tasks.attr("get")(get(id, "transfer"));
        if (!old.is_none()) {
            protocol(equal(get(old, "identity"), id) && equal(get(old, "source"), source) &&
                         equal(get(old, "target"), target),
                     "transfer replay changed data");
            return get(old, "state");
        }
        tasks[get(id, "transfer")] = cls("tasks", "Task")(id, source, target, copies, at(a, "release"));
        get(s, "_condition").attr("notify_all")();
        return state("WAITING");
    };

    h["tasks.TransferManager.publish_ready"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_condition"));
        auto t = task(s, at(a, "identity"));
        O pair = py::make_tuple(at(a, "rank"), at(a, "layer"));
        bool found = false;
        for (auto c : get(t, "copies")) {
            O source = get(c, "source");
            if (equal(get(source, "rank"), at(a, "rank")) && equal(get(source, "layer"), at(a, "layer")))
                found = true;
        }
        protocol(found, "unexpected ready rank/layer");
        if (equal(get(t, "state"), state("WAITING")) || equal(get(t, "state"), state("TRANSFERRING"))) {
            get(t, "ready").attr("add")(pair);
            put(t, "final", truth(get(t, "final")) || truth(at(a, "final")));
            get(s, "_condition").attr("notify_all")();
        }
        return none();
    };

    h["tasks.TransferManager.seal"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_condition"));
        put(task(s, at(a, "identity")), "final", true);
        get(s, "_condition").attr("notify_all")();
        return none();
    };

    h["tasks.TransferManager.cancel"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_condition"));
        O t = task(s, at(a, "identity"));
        if (!terminal(t)) {
            put(t, "state", state("CANCELLING"));
            put(t, "error", at(a, "reason"));
            get(s, "_condition").attr("notify_all")();
        }
        return get(t, "state");
    };

    h["tasks.TransferManager.status"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_condition"));
        return get(task(s, at(a, "identity")), "state");
    };

    h["tasks.TransferManager.failures"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_condition"));
        L out;
        for (auto p : py::cast<D>(get(s, "_tasks"))) {
            O t = py::reinterpret_borrow<O>(p.second);
            if (truth(get(t, "error")) && !equal(get(t, "state"), state("CANCELLING")) &&
                !equal(get(t, "state"), state("CANCELLED")))
                out.append(get(t, "error"));
        }
        return out;
    };

    h["tasks.TransferManager.statistics"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_condition"));
        return module("builtins").attr("dict")(get(s, "_statistics"));
    };

    h["tasks.TransferManager.forget"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_condition"));
        O id = at(a, "identity"), t = task(s, id);
        if (!truth(get(t, "notified")) || !truth(get(t, "released")))
            fail("cannot forget in-flight transfer", PyExc_RuntimeError);
        get(s, "_tasks").attr("pop")(get(id, "transfer"));
        return none();
    };

    h["tasks.TransferManager._task"] = [](D a) { return task(self(a), at(a, "identity")); };
    h["tasks.TransferManager._finish"] = [](D a) {
        auto s = self(a), t = at(a, "task");
        if (truth(get(t, "notified")))
            return none();
        if (truth(get(t, "pending")))
            fail("cannot finish with GPU accesses in flight", PyExc_RuntimeError);
        if (!truth(get(t, "released"))) {
            get(t, "release")();
            put(t, "released", true);
        }
        put(t, "state", at(a, "state"));
        put(t, "notified", true);
        get(s, "notifications")
            .attr("put")(cls("tasks", "Completion")(get(t, "identity"), at(a, "state"), get(t, "error")));
        return none();
    };

    h["tasks.TransferManager._advance"] = [](D a) {
        auto s = self(a), t = at(a, "task"), backend = get(s, "backend");
        D pending = get(t, "pending"), statistics = get(s, "_statistics");
        for (auto entry : list(pending.attr("items")())) {
            O pair = py::reinterpret_borrow<O>(entry), key = pair[py::int_(0)], handle = pair[py::int_(1)];
            if (truth(backend.attr("query")(handle))) {
                O error = py::getattr(handle, "error", py::str(""));
                backend.attr("retire")(handle);
                pending.attr("pop")(key);
                if (truth(error))
                    put(t, "error", error);
                else {
                    statistics["completed_batches"] = num(at(statistics, "completed_batches")) + 1;
                    statistics["copied_bytes"] =
                        num(at(statistics, "copied_bytes")) + num(py::getattr(handle, "bytes", py::int_(0)));
                    statistics["gpu_copy_ms"] = real(at(statistics, "gpu_copy_ms")) +
                                                real(py::getattr(handle, "gpu_ms", py::float_(0.0)));
                }
            }
        }
        if (equal(get(t, "state"), state("CANCELLING"))) {
            if (!truth(pending))
                s.attr("_finish")(t, state("CANCELLED"));
            return none();
        }
        if (truth(get(t, "error"))) {
            if (!truth(pending))
                s.attr("_finish")(t, state("FAILED"));
            return none();
        }

        D groups;
        I index = 0;
        for (auto entry : get(t, "copies")) {
            O copy = py::reinterpret_borrow<O>(entry), source = get(copy, "source");
            bool submitted = truth(module("operator").attr("contains")(get(t, "submitted"), index));
            bool ready = truth(module("operator")
                                   .attr("contains")(get(t, "ready"), py::make_tuple(get(source, "rank"),
                                                                                     get(source, "layer"))));
            if (!submitted && ready) {
                O key = py::make_tuple(get(source, "rank"), get(source, "layer"),
                                       get(get(copy, "target"), "rank"));
                if (!groups.contains(key))
                    groups[key] = L();
                groups[key].attr("append")(py::make_tuple(index, copy));
            }
            ++index;
        }
        for (auto group : groups) {
            L copies;
            for (auto entry : py::reinterpret_borrow<O>(group.second))
                copies.append(item(entry, py::int_(1)));
            O handle = backend.attr("submit")(tuple(copies));
            pending[group.first] = handle;
            for (auto entry : py::reinterpret_borrow<O>(group.second))
                get(t, "submitted").attr("add")(item(entry, py::int_(0)));
            put(t, "state", state("TRANSFERRING"));
        }
        if (truth(get(t, "final")) && py::len(get(t, "submitted")) == py::len(get(t, "copies")) &&
            !truth(pending))
            s.attr("_finish")(t, state("READY"));
        return none();
    };

    h["tasks.TransferManager._run"] = [](D a) {
        auto s = self(a), condition = get(s, "_condition");
        while (true) {
            Lock lock(condition);
            L active;
            for (auto p : py::cast<D>(get(s, "_tasks")))
                if (!truth(get(p.second, "notified")))
                    active.append(p.second);
            if (truth(get(s, "_closing")) && !truth(active))
                return none();
            if (!truth(active)) {
                condition.attr("wait")();
                continue;
            }
            for (auto entry : active) {
                O t = py::reinterpret_borrow<O>(entry);
                try {
                    s.attr("_advance")(t);
                } catch (const std::exception &error) {
                    put(t, "error", py::str(error.what()));
                    if (!truth(get(t, "pending"))) {
                        try {
                            s.attr("_finish")(t, state("FAILED"));
                        } catch (const std::exception &cleanup) {
                            put(t, "error",
                                py::str(std::string(error.what()) + "; release: " + cleanup.what()));
                        }
                    }
                }
            }
            condition.attr("wait")(get(s, "poll_interval"));
        }
    };

    h["tasks.TransferManager.close"] = [](D a) {
        auto s = self(a);
        {
            Lock lock(get(s, "_condition"));
            put(s, "_closing", true);
            for (auto p : py::cast<D>(get(s, "_tasks"))) {
                O t = py::reinterpret_borrow<O>(p.second);
                if (!truth(get(t, "notified"))) {
                    put(t, "state", state("CANCELLING"));
                    put(t, "error", py::str("service shutdown"));
                }
            }
            get(s, "_condition").attr("notify_all")();
        }
        get(s, "_thread").attr("join")(at(a, "timeout"));
        if (truth(get(s, "_thread").attr("is_alive")()))
            fail("PD transfer drain incomplete; KV remains protected", PyExc_TimeoutError);
        get(s, "backend").attr("close")();
        return none();
    };
}
} // namespace pd
} // namespace fastllm
