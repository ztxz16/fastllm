#include "pd_service.h"

namespace fastllm {
namespace pd {
/** 管理服务身份、布局协商及队列准入；HTTP 适配层不参与选择或生命周期判断。 */
void BindService(Handlers &h) {
    BindServiceTransfer(h);
    BindServiceOutput(h);
    h["service.identity_from"] = [](D a) {
        D value(module("builtins").attr("dict")(at(a, "value")));
        for (auto key : {"source", "target"})
            value[key] = cls("protocol", "_object")(cls("protocol", "Peer"), value[key]);
        return cls("protocol", "_object")(cls("protocol", "Identity"), value);
    };

    h["service.Service.__init__"] = [](D a) {
        O s = self(a);
        auto role = str(at(a, "role")), policy = str(at(a, "routing"));
        require((role == "p" || role == "d" || role == "full") && truth(at(a, "replicas")) &&
                    truth(at(a, "token")),
                "P/D service requires role, replicas and internal token");
        require(std::isfinite(real(at(a, "timeout"))) && real(at(a, "timeout")) > 0,
                "service timeout must be positive");
        integer(at(a, "long_threshold"), "long_threshold", 1);
        require(policy == "load" || policy == "round_robin" || policy == "prefix",
                "invalid internal DP policy");
        for (auto key :
             {"role", "peer", "replicas", "token", "timeout", "long_threshold", "routing", "return_kv"})
            put(s, key, at(a, key));
        put(s, "cursor", I(0));
        put(s, "cache_sequence", I(0));
        put(s, "history", cls("history", "History")(at(a, "peer"), at(a, "retention")));
        for (auto key : {"records", "layouts", "control_outbox", "cache_snapshot"})
            put(s, key, D());
        put(s, "agreements", py::set());
        put(s, "lock", module("threading").attr("RLock")());
        O backend = at(a, "backend");
        if (backend.is_none())
            backend = cls("cuda_ipc", "CudaIPC")();
        put(s, "transfers", cls("tasks", "TransferManager")(at(a, "peer"), backend));
        put(s, "closed", module("threading").attr("Event")());
        put(s, "failure", py::str(""));
        for (auto entry :
             {std::make_pair("notifier", "_notify"), std::make_pair("reaper", "_expire_history")}) {
            O thread = module("threading")
                           .attr("Thread")("target"_a = get(s, entry.second), "daemon"_a = true,
                                           "name"_a = entry.first);
            put(s, entry.first, thread);
            thread.attr("start")();
        }
        return none();
    };

    h["service.Service._record"] = [](D a) {
        O identity = at(a, "identity"), r = get(self(a), "records").attr("get")(get(identity, "transfer"));
        protocol(!r.is_none() && equal(get(r, "identity"), identity),
                 "unknown transfer, request or boot identity");
        return r;
    };

    h["service.Service._send_terminal"] = [](D a) {
        O s = self(a);
        try {
            rpc(s, at(a, "url"), str(at(a, "path")).c_str(), at(a, "body"),
                std::min(real(get(s, "timeout")), 5.0));
        } catch (const py::error_already_set &) {
            Lock lock(get(s, "lock"));
            get(s, "control_outbox")[uuid()] = py::make_tuple(at(a, "url"), at(a, "path"), at(a, "body"));
        }
        return none();
    };

    h["service.Service._expire_history"] = [](D a) {
        O s = self(a);
        while (!truth(get(s, "closed").attr("wait")(1))) {
            try {
                get(s, "history").attr("expire")();
                O controls;
                {
                    Lock lock(get(s, "lock"));
                    controls = list(get(s, "control_outbox").attr("items")());
                }
                for (auto pair : controls) {
                    O key = item(pair, py::int_(0)), value = item(pair, py::int_(1));
                    try {
                        rpc(s, item(value, py::int_(0)), str(item(value, py::int_(1))).c_str(),
                            item(value, py::int_(2)), std::min(5.0, real(get(s, "timeout"))));
                    } catch (const py::error_already_set &) {
                        continue;
                    }
                    Lock lock(get(s, "lock"));
                    get(s, "control_outbox").attr("pop")(key, none());
                }
                L expired;
                {
                    Lock lock(get(s, "lock"));
                    for (auto r : get(s, "records").attr("values")())
                        if (!truth(get(r, "released")) && real(get(r, "expires_at")) <= wall())
                            expired.append(get(r, "identity"));
                }
                for (auto identity : expired) {
                    try {
                        s.attr("cancel")(identity);
                    } catch (const py::error_already_set &) {
                    }
                }
                Lock lock(get(s, "lock"));
                for (auto pair : list(get(s, "records").attr("items")())) {
                    O key = item(pair, py::int_(0)), r = item(pair, py::int_(1));
                    if (!flag(r, "released") || real(get(r, "expires_at")) > wall() ||
                        (flag(r, "started") && !flag(r, "complete")))
                        continue;
                    if (flag(r, "started"))
                        get(s, "transfers").attr("forget")(recordIdentity(r));
                    get(s, "records").attr("pop")(key);
                }
            } catch (const py::error_already_set &) {
            }
        }
        return none();
    };

    h["service.Service.handshake"] = [](D a) -> O {
        O s = self(a), body = at(a, "body"), peer = get(s, "peer"),
          remote = cls("protocol", "_object")(cls("protocol", "Peer"), at(body, "source")),
          target = cls("protocol", "_object")(cls("protocol", "Peer"), at(body, "target"));
        O layout = cls("protocol", "_object")(cls("protocol", "Layout"), at(body, "layout")),
          own = get(item(get(s, "replicas"), py::int_(0)), "layout");
        protocol(equal(target, peer) && !equal(remote, peer) && equal(get(remote, "host"), get(peer, "host")),
                 "layout handshake belongs to another instance, boot or host");
        protocol(truth(own.attr("compatible")(layout)), "layout handshake model/dtype/shape mismatch");
        {
            Lock lock(get(s, "lock"));
            D layouts;
            for (auto pair : D(get(s, "layouts")))
                if (!equal(item(pair.first, py::int_(0)), get(remote, "instance")) ||
                    equal(item(pair.first, py::int_(1)), get(remote, "boot")))
                    layouts[pair.first] = pair.second;
            layouts[py::make_tuple(get(remote, "instance"), get(remote, "boot"), get(layout, "layout_id"))] =
                layout;
            put(s, "layouts", layouts);
        }
        return D("peer"_a = asdict(peer), "layout"_a = asdict(own), "layout_id"_a = get(own, "layout_id"));
    };

    h["service.Service._negotiate"] = [](D a) {
        O s = self(a), r = at(a, "record"),
          remote = get(recordIdentity(r), str(get(s, "role")) == "p" ? "target" : "source"),
          own = get(get(r, "replica"), "layout"), key = py::make_tuple(remote, get(own, "layout_id"));
        {
            Lock lock(get(s, "lock"));
            if (truth(get(s, "agreements").attr("__contains__")(key)))
                return none();
        }
        O reply =
            rpc(s, get(r, "peer_url"), "/internal/handshake",
                D("source"_a = asdict(get(s, "peer")), "target"_a = asdict(remote), "layout"_a = asdict(own)),
                std::min(real(get(s, "timeout")), std::max(.001, real(get(r, "expires_at")) - wall())));
        O layout = cls("protocol", "_object")(cls("protocol", "Layout"), at(reply, "layout"));
        protocol(equal(cls("protocol", "_object")(cls("protocol", "Peer"), at(reply, "peer")), remote) &&
                     equal(at(reply, "layout_id"), get(layout, "layout_id")) &&
                     truth(own.attr("compatible")(layout)),
                 "remote layout handshake changed identity or model semantics");
        {
            Lock lock(get(s, "lock"));
            py::set agreements;
            for (auto pair : get(s, "agreements")) {
                O peer = item(pair, py::int_(0));
                if (!equal(get(peer, "instance"), get(remote, "instance")) ||
                    equal(get(peer, "boot"), get(remote, "boot")))
                    agreements.add(pair);
            }
            agreements.add(key);
            put(s, "agreements", agreements);
        }
        return none();
    };

    h["service.Service._new"] = [](D a) -> O {
        O s = self(a), body = at(a, "body"), id = cls("service", "identity_from")(at(body, "identity")),
          request = at(body, "request"), expires = at(body, "expires_at");
        protocol(equal(get(id, str(get(s, "role")) == "p" ? "source" : "target"), get(s, "peer")) &&
                     str(get(id, "direction")) == "p_to_d",
                 "wrong service role, instance or boot");
        cls("service", "validate_request")(
            request, at(get(item(get(s, "replicas"), py::int_(0)), "config"), "vocab_size"));
        protocol((PyLong_CheckExact(expires.ptr()) || PyFloat_CheckExact(expires.ptr())) &&
                     std::isfinite(real(expires)) && real(expires) > wall(),
                 "request control lease has expired");
        Lock lock(get(s, "lock"));
        if (truth(get(s, "closed").attr("is_set")()) || flag(s, "failure"))
            fail(flag(s, "failure") ? str(get(s, "failure")) : "service closing", PyExc_RuntimeError);
        O history = flag(s, "return_kv") ? body.attr("get")("history") : none(),
          old = get(s, "records").attr("get")(get(id, "transfer"));
        if (!old.is_none()) {
            protocol(equal(recordIdentity(old), id) && equal(get(old, "request"), request) &&
                         equal(get(old, "peer_url"), at(body, "peer_url")) &&
                         equal(get(old, "history"), history),
                     "transfer identity replayed with changed data");
            return py::make_tuple(old, false);
        }
        O r = cls("service", "Record")(id, request, at(body, "peer_url"));
        put(r, "expires_at", expires);
        put(r, "long_prompt", I(py::len(at(request, "token_ids"))) >= num(get(s, "long_threshold")));
        put(r, "history", history);
        get(s, "records")[get(id, "transfer")] = r;
        return py::make_tuple(r, true);
    };

    h["service.Service._select_replica"] = [](D a) {
        O s = self(a), r = at(a, "record"), replicas = get(s, "replicas"), best;
        I count = py::len(replicas), cursor = num(get(s, "cursor")), chosen = cursor;
        double scoreBest = 0;
        auto policy = str(get(s, "routing"));
        for (I i = 0; i < count; ++i) {
            I ix = (cursor + i) % count;
            O replica = item(replicas, py::int_(ix));
            double score = 0;
            if (policy == "prefix")
                score = -real(cls("routing", "Registry")
                                  .attr("_prefix")(at(get(r, "request"), "token_ids"),
                                                   get(replica, "cached").attr("values")()));
            if (policy == "load") {
                I waiting = 0;
                for (auto pending : get(s, "records").attr("values")())
                    if (get(pending, "replica").is(replica) && !truth(get(pending, "admitted")) &&
                        !truth(get(pending, "released")))
                        ++waiting;
                score = real(replica.attr("score")(waiting));
            }
            if (i == 0 || score < scoreBest) {
                best = replica;
                chosen = ix;
                scoreBest = score;
            }
            if (policy == "round_robin")
                break;
        }
        put(s, "cursor", (chosen + 1) % count);
        put(r, "replica", best);
        return best;
    };

    h["service.Service.create"] = [](D a) -> O {
        O s = self(a), r = item(s.attr("_new")(at(a, "body")), py::int_(0)), id = recordIdentity(r);
        bool fresh;
        {
            Lock lock(get(s, "lock"));
            fresh = !flag(r, "preparing");
            put(r, "preparing", true);
        }
        if (!fresh) {
            if (!truth(get(r, "prepared").attr("wait")(get(s, "timeout"))))
                fail("duplicate request still preparing", PyExc_TimeoutError);
            if (flag(r, "error"))
                fail(str(get(r, "error")), PyExc_RuntimeError);
            protocol(!flag(r, "released") && !cancelled(r), "transfer already released or cancelled");
            return D("manifest"_a = decode(get(r, "manifest").attr("to_json")()));
        }
        double until = deadline(s, r);
        bool producer = str(get(s, "role")) == "p";
        try {
            O replica;
            {
                Lock lock(get(s, "lock"));
                replica = s.attr("_select_replica")(r);
            }
            s.attr("_negotiate")(r);
            replica.attr("acquire")(id, until, get(r, "cancelled"));
            put(r, "admitted", true);
            O tokens = at(get(r, "request"), "token_ids"), output = at(get(r, "request"), "max_new_tokens"),
              manifest;
            replica.attr("wait_capacity")(py::len(tokens), !producer, num(output) - 1, until,
                                          get(r, "cancelled"));
            if (producer && (flag(r, "long_prompt") || flag(r, "history"))) {
                manifest = replica.attr("prepare")(id, tokens, 1, "deadline"_a = until,
                                                   "cancelled"_a = get(r, "cancelled"));
                put(r, "local_reused", replica.attr("restore_local")(id, tokens));
            } else if (producer) {
                manifest = replica.attr("prefill")(id, tokens, "deadline"_a = until,
                                                   "cancelled"_a = get(r, "cancelled"));
                replica.attr("finish")(id);
                put(r, "prefill_done", true);
                py::set layers;
                for (auto region : get(manifest, "regions"))
                    layers.add(py::make_tuple(get(region, "rank"), get(region, "layer")));
                put(r, "ready_layers", layers);
            } else
                manifest = replica.attr("prepare")(id, tokens, output, "deadline"_a = until,
                                                   "cancelled"_a = get(r, "cancelled"));
            {
                Lock lock(get(s, "lock"));
                put(r, "manifest", manifest);
                get(r, "prepared").attr("set")();
            }
            if (producer && flag(r, "history"))
                s.attr("_restore_history")(r);
            if (producer && cancelled(r)) {
                replica.attr("finish")(id);
                put(r, "source_stopped", true);
                fail("cancelled before Prefill", PyExc_InterruptedError);
            }
            if (producer && flag(r, "long_prompt")) {
                put(r, "compute_active", true);
                start(get(s, "_prefill_long"), "ft-pd-prefill", py::make_tuple(r));
            } else if (producer && flag(r, "history")) {
                put(r, "compute_active", true);
                s.attr("_prefill_long")(r);
                if (flag(r, "error"))
                    fail(str(get(r, "error")), PyExc_RuntimeError);
            }
            if (cancelled(r))
                fail("cancelled during preparation", PyExc_InterruptedError);
            if (str(get(id, "mode")) == "push") {
                if (producer)
                    s.attr("_start_push")(r);
                else
                    rpc(s, get(r, "peer_url"), "/internal/register-target",
                        D("manifest"_a = decode(manifest.attr("to_json")())));
            }
            return D("manifest"_a = decode(manifest.attr("to_json")()));
        } catch (const std::exception &error) {
            {
                Lock lock(get(s, "lock"));
                put(r, "error", py::str(error.what()));
                get(r, "cancelled").attr("set")();
                get(r, "prepared").attr("set")();
                get(r, "ready").attr("set")();
            }
            O replica = get(r, "replica");
            if (producer && !replica.is_none() && !flag(r, "compute_active") && !flag(r, "history_active") &&
                str(get(get(replica, "session"), "state")) == "ready") {
                replica.attr("finish")(id);
                put(r, "source_stopped", true);
            }
            s.attr("cancel")(id);
            if (get(r, "manifest").is_none()) {
                if (!replica.is_none())
                    replica.attr("finish")(id);
                put(r, "released", true);
            }
            throw;
        }
    };
}
} // namespace pd
} // namespace fastllm
