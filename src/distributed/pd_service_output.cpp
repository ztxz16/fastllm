#include "pd_service.h"
namespace fastllm {
namespace pd {
/** 首 Token 与后续 Decode 均由 D 产生，C++ 检查输出进度并通知统一输出入口。 */
void BindServiceOutput(Handlers &h) {
    h["service.Service.generate"] = [](D a) {
        O s = self(a), id = at(a, "identity"), r = s.attr("_record")(id), emit = at(a, "_emit"),
          closed = at(a, "_closed");
        double until = deadline(s, r);
        if (truth(closed.attr("is_set")()))
            get(r, "cancelled").attr("set")();
        while (!truth(get(r, "ready").attr("wait")(.05))) {
            if (cancelled(r) || truth(closed.attr("is_set")()))
                fail("cancelled while receiving KV", PyExc_InterruptedError);
            if (now() >= until) {
                s.attr("cancel")(id);
                fail("KV receive timeout", PyExc_TimeoutError);
            }
        }
        if (flag(r, "error") || cancelled(r))
            fail(flag(r, "error") ? str(get(r, "error")) : "request cancelled", PyExc_RuntimeError);
        {
            Lock lock(get(s, "lock"));
            protocol(str(get(s, "role")) == "d" && !flag(r, "generating") && !flag(r, "generation_done") &&
                         !flag(r, "released"),
                     "generation already started or released");
            put(r, "generating", true);
        }
        O replica = get(r, "replica");
        auto cleanup = [&] {
            if (get(r, "history_result").is_none())
                replica.attr("release")(id);
            replica.attr("finish")(id);
            Lock lock(get(s, "lock"));
            put(r, "generating", false);
            put(r, "generation_done", true);
            put(r, "released", true);
        };
        try {
            replica.attr("commit")(id, py::len(at(get(r, "request"), "token_ids")));
            L generated;
            O callback = py::cpp_function([generated, emit, id](D event) mutable {
                generated.attr("extend")(event["token_ids"]);
                event["request"] = get(id, "request");
                event["transfer"] = get(id, "transfer");
                emit(event);
            });
            Invoke("native.NativeReplica.generate",
                   D("self"_a = replica, "identity"_a = id, "request"_a = get(r, "request"),
                     "cancelled"_a = get(r, "cancelled"), "deadline"_a = until, "_emit"_a = callback,
                     "_closed"_a = closed));
            if (flag(s, "return_kv") && !cancelled(r)) {
                O tokens = at(get(r, "request"), "token_ids")
                               .attr("__add__")(generated[py::slice(0, I(py::len(generated)) - 1, 1)]);
                O manifest = replica.attr("export")(id, tokens, "optional"_a = true);
                if (!manifest.is_none()) {
                    get(s, "history")
                        .attr("retain")(get(id, "transfer"), manifest, tokens,
                                        py::cpp_function([replica, id] { replica.attr("release")(id); }));
                    put(r, "history_result",
                        D("ticket"_a = get(id, "transfer"), "peer"_a = asdict(get(s, "peer")),
                          "expires_at"_a = wall() + real(get(get(s, "history"), "ttl"))));
                }
            }
            emit(D("event"_a = "finished", "request"_a = get(id, "request"),
                   "transfer"_a = get(id, "transfer"), "history"_a = get(r, "history_result")));
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
        return none();
    };

    h["service.Service.generate_full"] = [](D a) {
        O s = self(a), closed = at(a, "_closed"), emit = at(a, "_emit");
        protocol(str(get(s, "role")) == "full", "complete generation requires a full service");
        O created = s.attr("_new")(at(a, "body")), r = item(created, py::int_(0)), replica;
        {
            Lock lock(get(s, "lock"));
            protocol(truth(item(created, py::int_(1))) && !flag(r, "preparing"),
                     "complete request already submitted");
            put(r, "preparing", true);
            put(r, "generating", true);
            replica = s.attr("_select_replica")(r);
        }
        O id = recordIdentity(r), request = get(r, "request");
        double until = deadline(s, r);
        if (truth(closed.attr("is_set")()))
            get(r, "cancelled").attr("set")();
        I index = 0;
        auto cleanup = [&] {
            get(r, "cancelled").attr("set")();
            replica.attr("finish")(id);
            Lock lock(get(s, "lock"));
            put(r, "released", true);
            put(r, "generation_done", true);
            put(r, "generating", false);
        };
        try {
            replica.attr("acquire")(id, until, get(r, "cancelled"));
            put(r, "admitted", true);
            replica.attr("wait_capacity")(py::len(at(request, "token_ids")), true,
                                          num(at(request, "max_new_tokens")) - 1, until, get(r, "cancelled"));
            O callback = py::cpp_function([&](O token) {
                if (cancelled(r) || truth(closed.attr("is_set")()) || now() >= until) {
                    get(r, "cancelled").attr("set")();
                    fail("complete request cancelled or timed out", PyExc_InterruptedError);
                }
                O finish = none();
                if (truth(at(request, "eos_token_ids").attr("__contains__")(token)))
                    finish = py::str("eos");
                else if (index + 1 == num(at(request, "max_new_tokens")))
                    finish = py::str("length");
                L ids;
                ids.append(token);
                emit(D("event"_a = "token_delta", "index"_a = index++, "token_ids"_a = ids,
                       "finish_reason"_a = finish, "request"_a = get(id, "request"),
                       "transfer"_a = get(id, "transfer")));
            });
            get(replica, "session")
                .attr("generate")(request, "cancel_event"_a = get(r, "cancelled"), "on_token"_a = callback);
            replica.attr("refresh_cache")();
            emit(D("event"_a = "finished", "request"_a = get(id, "request"),
                   "transfer"_a = get(id, "transfer")));
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
        return none();
    };

    h["service.Service.health"] = [](D a) -> O {
        O s = self(a);
        Lock lock(get(s, "lock"));
        D cached;
        I preemptions = 0, load = 0;
        bool healthy = !flag(s, "failure") && !truth(get(s, "closed").attr("is_set")()) &&
                       !truth(get(s, "transfers").attr("failures")());
        for (auto replica : get(s, "replicas")) {
            Lock replicaLock(get(replica, "condition"));
            cached.attr("update")(get(replica, "cached"));
            preemptions += num(get(replica, "preemptions"));
            healthy = healthy && str(get(get(replica, "session"), "state")) == "ready";
        }
        for (auto r : get(s, "records").attr("values")())
            if (!truth(get(r, "released")))
                ++load;
        if (!equal(cached, get(s, "cache_snapshot"))) {
            put(s, "cache_sequence", num(get(s, "cache_sequence")) + 1);
            put(s, "cache_snapshot", cached);
        }
        return D("peer"_a = asdict(get(s, "peer")), "role"_a = get(s, "role"),
                 "model"_a = get(item(get(s, "replicas"), py::int_(0)), "model_id"), "healthy"_a = healthy,
                 "load"_a = load, "preemptions"_a = preemptions,
                 "pending_control"_a = py::len(get(s, "control_outbox")),
                 "transfer_statistics"_a = get(s, "transfers").attr("statistics")(),
                 "cache_version"_a = get(s, "cache_sequence"), "cache_snapshot"_a = cached);
    };

    h["service.Service.evidence"] = [](D a) -> O {
        O s = self(a);
        L requests, replicas;
        {
            Lock lock(get(s, "lock"));
            for (auto r : get(s, "records").attr("values")())
                requests.append(
                    D("request"_a = get(get(r, "identity"), "request"),
                      "transfer"_a = get(get(r, "identity"), "transfer"),
                      "local_reused"_a = get(r, "local_reused"), "remote_reused"_a = get(r, "remote_reused"),
                      "prefill_done"_a = get(r, "prefill_done"), "completed"_a = get(r, "complete"),
                      "released"_a = get(r, "released"), "error"_a = get(r, "error")));
        }
        for (auto r : get(s, "replicas")) {
            O session = get(r, "session");
            replicas.append(D("model_type"_a = at(get(r, "config"), "model_type"),
                              "storage"_a = py::getattr(session, "storage", py::str("paged")),
                              "devices"_a = get(session, "devices"),
                              "workers"_a = get(session, "ready").attr("get")("workers", L()),
                              "states"_a = get(r, "_command")("cache"),
                              "usage"_a = get(r, "_command")("usage")));
        }
        return D("health"_a = s.attr("health")(), "requests"_a = requests, "replicas"_a = replicas);
    };

    h["service.Service.close"] = [](D a) {
        O s = self(a);
        put(s, "failure", py::str("service closing"));
        for (auto r : list(get(s, "records").attr("values")())) {
            try {
                s.attr("cancel")(get(r, "identity"));
            } catch (const py::error_already_set &) {
            }
        }
        get(s, "transfers").attr("close")("timeout"_a = get(s, "timeout"));
        double until = now() + real(get(s, "timeout"));
        while (true) {
            bool retained = truth(get(s, "history").attr("close")()), controls;
            L unfinished;
            {
                Lock lock(get(s, "lock"));
                controls = flag(s, "control_outbox");
                for (auto r : get(s, "records").attr("values")())
                    if (!truth(get(r, "released")))
                        unfinished.append(r);
            }
            if (retained && !py::len(unfinished) && !controls)
                break;
            if (now() >= until)
                fail("shutdown has unconfirmed remote accesses; KV must remain protected",
                     PyExc_TimeoutError);
            for (auto r : unfinished) {
                try {
                    s.attr("cancel")(get(r, "identity"));
                } catch (const py::error_already_set &) {
                }
            }
            get(s, "closed").attr("wait")(.05);
        }
        get(s, "closed").attr("set")();
        get(s, "notifier").attr("join")(get(s, "timeout"));
        get(s, "reaper").attr("join")(get(s, "timeout"));
        return none();
    };
}
} // namespace pd
} // namespace fastllm
