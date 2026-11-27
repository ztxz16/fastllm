#include "pd_service.h"
namespace fastllm {
namespace pd {
/** 交接完成与源端解除占用分别判断，取消不绕过真实 CUDA 完成事件。 */
void BindServiceTransfer(Handlers &h) {
    h["service.Service._restore_history"] = [](D a) {
        O s = self(a), r = at(a, "record"), history = get(r, "history"),
          id = cls("protocol", "Identity")(get(recordIdentity(r), "request"), uuid(),
                                           cls("protocol", "Peer")(**at(history, "peer")), get(s, "peer"),
                                           "d_to_p", "pull");
        O release = D("ticket"_a = at(history, "ticket"), "identity"_a = asdict(id)), result;
        auto terminal = [&] {
            s.attr("_send_terminal")(at(history, "url"), "/internal/history/release", release);
        };
        try {
            result = rpc(s, at(history, "url"), "/internal/history/acquire",
                         D("ticket"_a = at(history, "ticket"), "identity"_a = asdict(id),
                           "token_ids"_a = at(get(r, "request"), "token_ids")));
        } catch (const py::error_already_set &) {
            terminal();
            return none();
        }
        if (result.attr("get")("manifest").is_none())
            return none();
        O manager = cls("tasks", "TransferManager")(get(s, "peer"), cls("cuda_ipc", "CudaIPC")());
        put(r, "history_active", true);
        auto cleanup = [&] {
            manager.attr("close")("timeout"_a = get(s, "timeout"));
            put(r, "history_active", false);
            terminal();
        };
        try {
            O source = cls("protocol", "Manifest").attr("from_json")(encode(at(result, "manifest")));
            if (num(get(source, "token_end")) > num(get(r, "local_reused"))) {
                O tokens = at(get(r, "request"), "token_ids")[py::slice(0, num(get(source, "token_end")), 1)];
                O target = cls("protocol", "prefix_manifest")(get(r, "manifest"), id, get(s, "peer"), tokens);
                if (flag(r, "local_reused")) {
                    source = cls("protocol", "suffix_manifest")(source, get(r, "local_reused"));
                    target = cls("protocol", "suffix_manifest")(target, get(r, "local_reused"));
                }
                manager.attr("register")(source, target, py::cpp_function([] {}));
                py::set ready;
                for (auto c : plan(source, target))
                    ready.add(py::make_tuple(get(get(c, "source"), "rank"), get(get(c, "source"), "layer")));
                for (auto pair : ready)
                    manager.attr("publish_ready")(id, item(pair, py::int_(0)), item(pair, py::int_(1)));
                manager.attr("seal")(id);
                double until = deadline(s, r), draining = -1;
                O completion;
                while (true) {
                    if (cancelled(r) || now() >= until) {
                        manager.attr("cancel")(id);
                        if (draining < 0)
                            draining = now();
                        else if (now() - draining >= real(get(s, "timeout"))) {
                            put(s, "failure",
                                py::str("history transfer could not drain; source and target remain "
                                        "quarantined"));
                            fail(str(get(s, "failure")), PyExc_TimeoutError);
                        }
                    }
                    try {
                        completion = get(manager, "notifications").attr("get")("timeout"_a = .05);
                        break;
                    } catch (const py::error_already_set &error) {
                        if (!error.matches(module("queue").attr("Empty")))
                            throw;
                    }
                }
                if (!equal(get(completion, "state"), cls("tasks", "State").attr("READY")))
                    fail(truth(get(completion, "error")) ? str(get(completion, "error"))
                                                         : "history transfer failed",
                         PyExc_RuntimeError);
                O replica = get(r, "replica"),
                  key = item(item(get(replica, "exports"), get(recordIdentity(r), "transfer")), py::int_(1));
                get(replica, "session")
                    .attr("pd_command")(D("action"_a = "restore_prefix", "key"_a = key,
                                          "tokens"_a = get(source, "token_end"), "capacity"_a = 0,
                                          "position"_a = 0));
                put(r, "remote_reused", num(get(source, "token_end")) - num(get(r, "local_reused")));
            }
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
        return none();
    };

    h["service.Service._prefill_long"] = [](D a) {
        O s = self(a), r = at(a, "record"), replica = get(r, "replica"), id = recordIdentity(r);
        try {
            replica.attr("prepared_prefill")(
                id, at(get(r, "request"), "token_ids"),
                py::cpp_function([s, r](O rank, O layer) { s.attr("_layer_ready")(r, rank, layer); }));
            replica.attr("finish")(id);
            Lock lock(get(s, "lock"));
            put(r, "prefill_done", true);
            py::set ready;
            for (auto region : get(get(r, "manifest"), "regions"))
                ready.add(py::make_tuple(get(region, "rank"), get(region, "layer")));
            put(r, "ready_layers", ready);
            if (flag(r, "started"))
                s.attr("_publish")(r);
            if (cancelled(r))
                s.attr("_release_source")(r);
        } catch (const std::exception &error) {
            Lock lock(get(s, "lock"));
            put(r, "error", py::str(error.what()));
            get(r, "cancelled").attr("set")();
        }
        put(r, "compute_active", false);
        if (cancelled(r)) {
            if (str(get(get(replica, "session"), "state")) == "ready") {
                replica.attr("finish")(id);
                put(r, "source_stopped", true);
            }
            s.attr("cancel")(id);
        }
        return none();
    };

    h["service.Service._layer_ready"] = [](D a) {
        O s = self(a), r = at(a, "record");
        Lock lock(get(s, "lock"));
        get(r, "ready_layers").attr("add")(py::make_tuple(at(a, "rank"), at(a, "layer")));
        if (flag(r, "started"))
            s.attr("_publish")(r);
        return none();
    };

    h["service.Service.readiness"] = [](D a) -> O {
        O s = self(a);
        Lock lock(get(s, "lock"));
        O r = s.attr("_record")(at(a, "identity"));
        return D("ready"_a = module("builtins").attr("sorted")(get(r, "ready_layers")),
                 "done"_a = get(r, "prefill_done"), "error"_a = get(r, "error"),
                 "cancelled"_a = cancelled(r));
    };

    h["service.Service._poll_source"] = [](D a) {
        O s = self(a), r = at(a, "record");
        try {
            while (!cancelled(r)) {
                O state = rpc(s, get(r, "peer_url"), "/internal/readiness", fields(r));
                if (truth(at(state, "error")) || truth(at(state, "cancelled")))
                    fail(truth(at(state, "error")) ? str(at(state, "error")) : "source cancelled",
                         PyExc_RuntimeError);
                {
                    Lock lock(get(s, "lock"));
                    py::set ready;
                    for (auto pair : at(state, "ready"))
                        ready.add(tuple(pair));
                    put(r, "ready_layers", ready);
                    put(r, "prefill_done", at(state, "done"));
                    s.attr("_publish")(r);
                }
                if (truth(at(state, "done")))
                    break;
                get(r, "cancelled").attr("wait")(.002);
            }
        } catch (const std::exception &error) {
            get(s, "transfers").attr("cancel")(recordIdentity(r), error.what());
        }
        return none();
    };

    h["service.Service.register_target"] = [](D a) -> O {
        O s = self(a), manifest = at(a, "manifest"), id = get(manifest, "identity"), r;
        protocol(str(get(s, "role")) == "p" && str(get(id, "mode")) == "push" &&
                     equal(get(manifest, "owner"), get(id, "target")),
                 "invalid push target registration");
        {
            Lock lock(get(s, "lock"));
            r = s.attr("_record")(id);
            protocol(!cancelled(r), "request cancelled");
            ValidatePairManifest(r, manifest, "target");
            protocol(get(r, "remote").is_none() || equal(get(r, "remote"), manifest),
                     "target registration changed");
            put(r, "remote", manifest);
        }
        s.attr("_start_push")(r);
        return D("registered"_a = true);
    };

    h["service.Service._start_push"] = [](D a) {
        O s = self(a), r = at(a, "record");
        Lock lock(get(s, "lock"));
        if (!flag(r, "started") && !get(r, "manifest").is_none() && !get(r, "remote").is_none() &&
            !cancelled(r))
            s.attr("_submit")(r, get(r, "manifest"), get(r, "remote"));
        return none();
    };

    h["service.Service.lease"] = [](D a) -> O {
        O s = self(a), id = at(a, "identity");
        bool acquire = truth(at(a, "acquire"));
        Lock lock(get(s, "lock"));
        protocol(str(get(s, "role")) == "p" && str(get(id, "mode")) == "pull",
                 "only P source exports accept pull leases");
        if (!acquire && equal(get(id, "source"), get(s, "peer")) &&
            !D(get(s, "records")).contains(get(id, "transfer")))
            return D("leased"_a = false, "ignored"_a = "no remaining export");
        O r = s.attr("_record")(id);
        if (acquire) {
            protocol(!cancelled(r) && !flag(r, "released") && !get(r, "manifest").is_none(),
                     "source export unavailable");
            put(r, "readers", I(1));
        } else {
            put(r, "readers", I(0));
            s.attr("_release_source")(r);
        }
        return D("leased"_a = acquire);
    };

    h["service.Service.pull"] = [](D a) -> O {
        O s = self(a), source = at(a, "source"), id = get(source, "identity");
        protocol(str(get(s, "role")) == "d" && str(get(id, "mode")) == "pull" &&
                     equal(get(source, "owner"), get(id, "source")),
                 "invalid pull source");
        Lock lock(get(s, "lock"));
        O r = s.attr("_record")(id);
        if (flag(r, "started")) {
            protocol(equal(get(r, "remote"), source), "source changed on repeated pull");
            return D("submitted"_a = true);
        }
        protocol(!get(r, "manifest").is_none() && !cancelled(r), "receiver unavailable");
        ValidatePairManifest(r, source, "source");
        put(r, "remote", source);
        try {
            rpc(s, get(r, "peer_url"), "/internal/lease", D("identity"_a = asdict(id), "acquire"_a = true));
            s.attr("_submit")(r, source, get(r, "manifest"));
            start(get(s, "_poll_source"), "ft-pd-source-ready", py::make_tuple(r));
        } catch (...) {
            if (flag(r, "started"))
                get(s, "transfers").attr("cancel")(id, "pull setup failed");
            else
                s.attr("_send_terminal")(get(r, "peer_url"), "/internal/lease",
                                         D("identity"_a = asdict(id), "acquire"_a = false));
            throw;
        }
        return D("submitted"_a = true);
    };

    h["service.Service._submit"] = [](D a) {
        O s = self(a), r = at(a, "record");
        ValidatePairManifest(r, at(a, "source"), "source");
        ValidatePairManifest(r, at(a, "target"), "target");
        py::set expected;
        for (auto c : plan(at(a, "source"), at(a, "target")))
            expected.add(py::make_tuple(get(get(c, "source"), "rank"), get(get(c, "source"), "layer")));
        put(r, "expected_layers", expected);
        get(s, "transfers").attr("register")(at(a, "source"), at(a, "target"), py::cpp_function([] {}));
        put(r, "started", true);
        s.attr("_publish")(r);
        return none();
    };

    h["service.Service._publish"] = [](D a) {
        O s = self(a), r = at(a, "record"),
          ready = get(r, "expected_layers")
                      .attr("intersection")(get(r, "ready_layers"))
                      .attr("difference")(get(r, "published_layers"));
        for (auto pair : ready)
            get(s, "transfers")
                .attr("publish_ready")(recordIdentity(r), item(pair, py::int_(0)), item(pair, py::int_(1)));
        get(r, "published_layers").attr("update")(ready);
        if (flag(r, "prefill_done") && !flag(r, "sealed")) {
            get(s, "transfers").attr("seal")(recordIdentity(r));
            put(r, "sealed", true);
        }
        return none();
    };

    h["service.Service._release_source"] = [](D a) {
        O s = self(a), r = at(a, "record"), id = recordIdentity(r);
        if (flag(r, "readers") || flag(r, "released") || get(r, "manifest").is_none() ||
            (!flag(r, "prefill_done") && !flag(r, "source_stopped")))
            return none();
        if (flag(r, "started")) {
            auto state = str(get(get(s, "transfers").attr("status")(id), "value"));
            if (state != "ready" && state != "failed" && state != "cancelled")
                return none();
        }
        get(r, "replica").attr("release")(id);
        put(r, "released", true);
        return none();
    };

    h["service.Service._release_target"] = [](D a) {
        O r = at(a, "record");
        if (flag(r, "released") || flag(r, "generating") || get(r, "manifest").is_none() ||
            (!flag(r, "complete") && (flag(r, "started") || str(get(recordIdentity(r), "mode")) == "push")))
            return none();
        get(r, "replica").attr("release")(recordIdentity(r));
        get(r, "replica").attr("finish")(recordIdentity(r));
        put(r, "released", true);
        return none();
    };

    h["service.Service.received"] = [](D a) -> O {
        O s = self(a), id = at(a, "identity"), state = at(a, "state");
        Lock lock(get(s, "lock"));
        auto name = str(state);
        protocol(str(get(s, "role")) == "d" && str(get(id, "mode")) == "push" &&
                     (name == "ready" || name == "failed" || name == "cancelled"),
                 "invalid completion notification");
        if (equal(get(id, "target"), get(s, "peer")) && !D(get(s, "records")).contains(get(id, "transfer")))
            return D("received"_a = false, "ignored"_a = "no active transfer");
        O r = s.attr("_record")(id);
        if (flag(r, "completion_state")) {
            protocol(equal(get(r, "completion_state"), state), "completion replay changed terminal state");
            return D("received"_a = true);
        }
        put(r, "completion_state", state);
        put(r, "complete", true);
        if (name != "ready")
            put(r, "error", truth(at(a, "error")) ? at(a, "error") : state);
        get(r, "ready").attr("set")();
        if (cancelled(r))
            s.attr("_release_target")(r);
        return D("received"_a = true);
    };

    h["service.Service._notify"] = [](D a) {
        O s = self(a);
        while (!truth(get(s, "closed").attr("is_set")())) {
            O event, r;
            try {
                event = get(get(s, "transfers"), "notifications").attr("get")("timeout"_a = .1);
            } catch (const py::error_already_set &error) {
                if (error.matches(module("queue").attr("Empty")))
                    continue;
                throw;
            }
            {
                Lock lock(get(s, "lock"));
                r = s.attr("_record")(get(event, "identity"));
                put(r, "complete", true);
                put(r, "error",
                    equal(get(event, "state"), cls("tasks", "State").attr("READY")) ? O(py::str(""))
                                                                                    : get(event, "error"));
            }
            try {
                if (str(get(s, "role")) == "p") {
                    s.attr("_send_terminal")(get(r, "peer_url"), "/internal/received",
                                             D("identity"_a = asdict(get(event, "identity")),
                                               "state"_a = get(get(event, "state"), "value"),
                                               "error"_a = get(r, "error")));
                    Lock lock(get(s, "lock"));
                    s.attr("_release_source")(r);
                } else
                    s.attr("_send_terminal")(
                        get(r, "peer_url"), "/internal/lease",
                        D("identity"_a = asdict(get(event, "identity")), "acquire"_a = false));
            } catch (const std::exception &error) {
                put(r, "error", py::str(error.what()));
            }
            get(r, "ready").attr("set")();
            if (str(get(s, "role")) == "d" && cancelled(r)) {
                Lock lock(get(s, "lock"));
                s.attr("_release_target")(r);
            }
        }
        return none();
    };

    h["service.Service.cancel"] = [](D a) -> O {
        O s = self(a), id = at(a, "identity"), r;
        bool drained;
        {
            Lock lock(get(s, "lock"));
            r = s.attr("_record")(id);
            get(r, "cancelled").attr("set")();
            if (get(r, "manifest").is_none() && !flag(r, "preparing"))
                put(r, "released", true);
            if (flag(r, "started") && !flag(r, "complete"))
                get(s, "transfers").attr("cancel")(id);
            else
                s.attr(str(get(s, "role")) == "p" ? "_release_source" : "_release_target")(r);
            drained = flag(r, "complete") || flag(r, "released") || !flag(r, "started");
        }
        if (str(get(s, "role")) == "d" && str(get(id, "mode")) == "push" && !flag(r, "complete")) {
            O result = rpc(s, get(r, "peer_url"), "/internal/cancel", D("identity"_a = asdict(id)),
                           std::min(5.0, real(get(s, "timeout"))));
            if (result.attr("get")("drained").is(py::bool_(true))) {
                Lock lock(get(s, "lock"));
                put(r, "complete", true);
                s.attr("_release_target")(r);
                drained = true;
            }
        }
        return D("cancelled"_a = true, "drained"_a = drained);
    };
}
} // namespace pd
} // namespace fastllm
