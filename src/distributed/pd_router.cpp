#include "pd_service.h"
namespace fastllm {
namespace pd {
/** 根据请求重试预算决定是否重新配对，输出已交付之后不再重试。 */
static void Generate(O s, D a, bool full) {
    O request = at(a, "request"), budget = at(a, "budget"), session = at(a, "session"), emit = at(a, "_emit"),
      closed = at(a, "_closed"), requestId = uuid(), registry = get(s, "registry");
    bool live = truth(at(a, "live"));
    while (true) {
        O transfer = budget.attr("begin")(), id = none(), p = none(), d = none(), delay = none(),
          response = none();
        L leases;
        auto remaining = [&] {
            if (truth(closed.attr("is_set")()))
                put(budget, "cancelled", true);
            return s.attr("_remaining")(budget);
        };
        auto cleanup = [&] {
            if (!response.is_none()) {
                response.attr("close")();
                response = none();
            }
            if (!id.is_none()) {
                if (!full)
                    s.attr("_cancel_pair")(id, p, d);
                else {
                    try {
                        rpc(s, get(d, "url"), "/internal/cancel", D("identity"_a = asdict(id)),
                            real(get(s, "health_timeout")));
                    } catch (const py::error_already_set &) {
                    }
                }
            }
            for (auto lease : leases)
                registry.attr("release")(lease);
        };
        try {
            remaining();
            if (!full)
                leases.append(registry.attr("select")("p", get(s, "model_id"), at(request, "token_ids"),
                                                      "session"_a = session));
            leases.append(registry.attr("select")(full ? "full" : "d", get(s, "model_id"),
                                                  at(request, "token_ids"), "session"_a = session));
            d = registry.attr("worker")(leases[py::len(leases) - 1]);
            if (!full)
                p = registry.attr("worker")(leases[0]);
            O source, target;
            {
                Lock lock(get(s, "lock"));
                target = item(get(s, "peers"), get(d, "url"));
                source = full ? cls("protocol", "Peer")("router", get(s, "boot"), get(target, "host"))
                              : item(get(s, "peers"), get(p, "url"));
            }
            if (!equal(get(target, "boot"), get(d, "boot")) ||
                (!full && !equal(get(source, "boot"), get(p, "boot"))))
                fail("selected worker restarted", PyExc_ConnectionError);
            id = cls("protocol", "Identity")(requestId, transfer, source, target, "p_to_d",
                                             full ? O(py::str("pull")) : get(s, "mode"));
            double expires = wall() + real(remaining());
            D dBody("identity"_a = asdict(id), "request"_a = request,
                    "peer_url"_a = (full ? O(py::str("")) : get(p, "url")), "expires_at"_a = expires);
            if (!full) {
                D pBody("identity"_a = asdict(id), "request"_a = request, "peer_url"_a = get(d, "url"),
                        "expires_at"_a = expires);
                if (truth(session) && flag(s, "return_kv")) {
                    Lock lock(get(s, "lock"));
                    O history = get(s, "histories").attr("get")(session);
                    if (!history.is_none() && real(at(history, "expires_at")) > wall())
                        pBody["history"] = history;
                }
                rpc(s, get(p, "url"), "/internal/announce", pBody, real(remaining()));
                bool pull = str(get(s, "mode")) == "pull";
                if (pull && I(py::len(at(request, "token_ids"))) < num(get(s, "long_threshold"))) {
                    O meta = rpc(s, get(p, "url"), "/internal/prefill", pBody, real(remaining()));
                    rpc(s, get(d, "url"), "/internal/prepare", dBody, real(remaining()));
                    rpc(s, get(d, "url"), "/internal/pull", meta, real(remaining()));
                } else {
                    // HTTP 线程只执行 I/O；两端的并发时机和结果依赖由此处控制。
                    O executor = module("concurrent.futures")
                                     .attr("ThreadPoolExecutor")("max_workers"_a = 2,
                                                                 "thread_name_prefix"_a = "ft-pd-pair"),
                      meta;
                    try {
                        O pf = executor.attr("submit")(cls("http", "rpc"), get(p, "url"), "/internal/prefill",
                                                       pBody, "token"_a = get(s, "token"),
                                                       "timeout"_a = remaining());
                        O df = executor.attr("submit")(cls("http", "rpc"), get(d, "url"), "/internal/prepare",
                                                       dBody, "token"_a = get(s, "token"),
                                                       "timeout"_a = remaining());
                        meta = pf.attr("result")("timeout"_a = remaining());
                        df.attr("result")("timeout"_a = remaining());
                    } catch (...) {
                        executor.attr("shutdown")("wait"_a = true);
                        throw;
                    }
                    executor.attr("shutdown")("wait"_a = true);
                    if (pull)
                        rpc(s, get(d, "url"), "/internal/pull", meta, real(remaining()));
                }
            }
            response = cls("http", "call")(get(d, "url"), full ? "/internal/full" : "/internal/generate",
                                           full ? O(dBody) : O(D("identity"_a = asdict(id))),
                                           "token"_a = get(s, "token"), "timeout"_a = remaining());
            I index = 0;
            bool finished = false;
            L buffered;
            for (auto raw : cls("http", "events")(response)) {
                O event = py::reinterpret_borrow<O>(raw);
                remaining();
                auto kind = str(event.attr("get")("event"));
                if (kind == "error")
                    fail(str(event.attr("get")("error", "generation failed")), PyExc_ConnectionError);
                require(equal(event.attr("get")("request"), requestId) &&
                            equal(event.attr("get")("transfer"), transfer),
                        "stale or mismatched internal result");
                if (kind == "finished") {
                    if (!full && truth(session) && flag(s, "return_kv")) {
                        Lock lock(get(s, "lock"));
                        O history = event.attr("get")("history");
                        if (truth(history)) {
                            D saved(module("builtins").attr("dict")(history));
                            saved["url"] = get(d, "url");
                            get(s, "histories")[session] = saved;
                        } else
                            get(s, "histories").attr("pop")(session, none());
                    }
                    finished = true;
                    break;
                }
                O tokens = event.attr("get")("token_ids");
                require(kind == "token_delta" && equal(event.attr("get")("index"), py::int_(index)) &&
                            py::isinstance<L>(tokens) && truth(tokens),
                        "missing or duplicate output position");
                for (auto token : tokens)
                    integer(py::reinterpret_borrow<O>(token), "generated token", 0);
                index += py::len(tokens);
                if (live)
                    emit(event);
                else
                    buffered.append(event);
            }
            if (!finished || !index)
                fail(full ? "complete stream ended without completion"
                          : "decode result stream ended without completion",
                     PyExc_ConnectionError);
            if (!live)
                for (auto event : buffered)
                    emit(event);
        } catch (const py::error_already_set &error) {
            if (truth(closed.attr("is_set")()))
                put(budget, "cancelled", true);
            bool http = error.matches(module("urllib.error").attr("HTTPError"));
            bool transport =
                !http && (error.matches(PyExc_TimeoutError) || error.matches(PyExc_ConnectionError) ||
                          error.matches(module("urllib.error").attr("URLError")));
            delay = budget.attr("delay")("status"_a = (http ? get(error.value(), "code") : none()),
                                         "transport_error"_a = transport);
            cleanup();
            if (delay.is_none())
                throw;
        } catch (...) {
            cleanup();
            throw;
        }
        if (delay.is_none()) {
            cleanup();
            return;
        }
        if (truth(get(s, "stopped").attr("wait")(delay)))
            fail("router stopped", PyExc_InterruptedError);
    }
}

/** 服务注册、健康探测、会话历史及八种策略共享同一 C++ Registry。 */
void BindRouter(Handlers &h) {
    h["router.Router.__init__"] = [](D a) {
        O s = self(a);
        auto mode = str(at(a, "mode"));
        require((mode == "pull" || mode == "push") && truth(at(a, "services")) && truth(at(a, "token")),
                "invalid router deployment");
        for (auto name : {"timeout", "health_interval", "health_timeout"})
            require(std::isfinite(real(at(a, name))) && real(at(a, name)) > 0,
                    "router timeouts must be finite and positive");
        std::set<std::string> roles;
        for (auto pair : at(a, "services"))
            roles.insert(str(item(pair, py::int_(0))));
        require(roles == std::set<std::string>{"full"} || roles == std::set<std::string>{"p", "d"},
                "router requires full workers or both P and D pools");
        for (auto key :
             {"registry", "services", "token", "model_id", "tokenizer", "model_path", "mode", "timeout",
              "attempts", "health_interval", "health_timeout", "long_threshold", "return_kv"})
            put(s, key, at(a, key));
        put(s, "full", roles.count("full") != 0);
        put(s, "boot", uuid());
        for (auto key : {"histories", "peers", "cache_versions"})
            put(s, key, D());
        put(s, "lock", module("threading").attr("RLock")());
        put(s, "stopped", module("threading").attr("Event")());
        put(s, "monitor",
            module("threading")
                .attr("Thread")("target"_a = get(s, "_health"), "name"_a = "ft-router-health",
                                "daemon"_a = true));
        return none();
    };

    h["router.Router._probe"] = [](D a) {
        O s = self(a), url = at(a, "url"), role = at(a, "role"),
          info = rpc(s, url, "/health", none(), real(get(s, "health_timeout"))),
          peer = cls("protocol", "Peer")(**at(info, "peer"));
        require(equal(at(info, "role"), role) && equal(at(info, "model"), get(s, "model_id")) &&
                    info.attr("get")("healthy").is(py::bool_(true)),
                "service role/model/health mismatch");
        Lock lock(get(s, "lock"));
        if (!truth(get(s, "services").attr("__contains__")(py::make_tuple(role, url))))
            return none();
        for (auto pair : D(get(s, "peers")))
            require(equal(pair.first, url) || !equal(get(pair.second, "instance"), get(peer, "instance")),
                    "service instance ID is already registered at another URL");
        O registry = get(s, "registry"), name = get(peer, "instance"), boot = get(peer, "boot");
        registry.attr("register")(name, url, role, at(info, "model"), boot);
        registry.attr("health_result")(name, boot, true);
        registry.attr("update_load")(name, boot, at(info, "load"));
        O key = py::make_tuple(name, boot), version = info.attr("get")("cache_version");
        if (!equal(get(s, "cache_versions").attr("get")(key), version)) {
            registry.attr("cache_snapshot")(name, boot, info.attr("get")("cache_snapshot", D()));
            get(s, "cache_versions")[key] = version;
        }
        get(s, "peers")[url] = peer;
        return none();
    };

    h["router.Router._health"] = [](D a) {
        O s = self(a);
        while (!truth(get(s, "stopped").attr("is_set")())) {
            O services;
            {
                Lock lock(get(s, "lock"));
                D histories;
                for (auto pair : D(get(s, "histories")))
                    if (real(at(pair.second, "expires_at")) > wall())
                        histories[pair.first] = pair.second;
                put(s, "histories", histories);
                services = list(get(s, "services"));
            }
            for (auto pair : services) {
                O role = item(pair, py::int_(0)), url = item(pair, py::int_(1));
                try {
                    s.attr("_probe")(role, url);
                } catch (const py::error_already_set &) {
                    O peer;
                    {
                        Lock lock(get(s, "lock"));
                        peer = get(s, "peers").attr("get")(url);
                    }
                    if (!peer.is_none())
                        get(s, "registry")
                            .attr("health_result")(get(peer, "instance"), get(peer, "boot"), false);
                }
            }
            get(s, "stopped").attr("wait")(get(s, "health_interval"));
        }
        return none();
    };

    h["router.Router.start"] = [](D a) {
        O s = self(a);
        for (auto pair : list(get(s, "services"))) {
            try {
                s.attr("_probe")(item(pair, py::int_(0)), item(pair, py::int_(1)));
            } catch (const py::error_already_set &) {
            }
        }
        get(s, "monitor").attr("start")();
        return none();
    };

    h["router.Router.register"] = [](D a) {
        O s = self(a), role = at(a, "role"), url = at(a, "url").attr("rstrip")("/");
        auto r = str(role);
        require(flag(s, "full") ? r == "full" : r == "p" || r == "d",
                "service role is incompatible with this router deployment");
        O address = module("urllib.parse").attr("urlsplit")(url);
        auto scheme = str(get(address, "scheme"));
        require((scheme == "http" || scheme == "https") && truth(get(address, "hostname")) &&
                    !truth(get(address, "username")) && !truth(get(address, "password")) &&
                    !truth(get(address, "query")) && !truth(get(address, "fragment")) &&
                    !truth(get(address, "path")),
                "worker URL must be an HTTP(S) origin");
        {
            Lock lock(get(s, "lock"));
            for (auto pair : get(s, "services"))
                require(!equal(item(pair, py::int_(1)), url) || equal(item(pair, py::int_(0)), role),
                        "service URL already registered with another role");
            O pair = py::make_tuple(role, url);
            if (!truth(get(s, "services").attr("__contains__")(pair))) {
                O services = list(get(s, "services"));
                services.attr("append")(pair);
                put(s, "services", services);
            }
        }
        s.attr("_probe")(role, url);
        return none();
    };

    h["router.Router.unregister"] = [](D a) {
        O s = self(a), url = at(a, "url").attr("rstrip")("/");
        Lock lock(get(s, "lock"));
        L services;
        for (auto pair : get(s, "services"))
            if (!equal(item(pair, py::int_(1)), url))
                services.append(pair);
        put(s, "services", services);
        O peer = get(s, "peers").attr("pop")(url, none());
        if (!peer.is_none()) {
            get(s, "registry").attr("unregister")(get(peer, "instance"));
            get(s, "cache_versions")
                .attr("pop")(py::make_tuple(get(peer, "instance"), get(peer, "boot")), none());
        }
        return none();
    };

    h["router.Router.close"] = [](D a) {
        O s = self(a);
        get(s, "stopped").attr("set")();
        get(s, "monitor").attr("join")(real(get(s, "health_timeout")) * py::len(get(s, "services")) + 1);
        return none();
    };

    h["router.Router._remaining"] = [](D a) -> O {
        O budget = at(a, "budget");
        double remaining = real(get(budget, "deadline")) - now();
        if (flag(budget, "cancelled") || remaining <= 0)
            fail("request cancelled or deadline exceeded", PyExc_TimeoutError);
        return py::float_(remaining);
    };

    h["router.Router._cancel_pair"] = [](D a) {
        O s = self(a), id = at(a, "identity");
        T workers = str(get(id, "mode")) == "pull" ? py::make_tuple(at(a, "d"), at(a, "p"))
                                                   : py::make_tuple(at(a, "p"), at(a, "d"));
        for (auto w : workers) {
            try {
                rpc(s, get(w, "url"), "/internal/cancel", D("identity"_a = asdict(id)),
                    real(get(s, "health_timeout")));
            } catch (const py::error_already_set &) {
            }
        }
        return none();
    };

    h["router.Router.generate"] = [](D a) {
        Generate(self(a), a, flag(self(a), "full"));
        return none();
    };

    h["router.Router._generate_full"] = [](D a) {
        Generate(self(a), a, true);
        return none();
    };
}
} // namespace pd
} // namespace fastllm
