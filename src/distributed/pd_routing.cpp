#include "pd_bindings.h"

namespace fastllm {
namespace pd {

static D dictfield(O o, const char *key) {
    return py::cast<D>(get(o, key));
}
static O Get(D d, O key) {
    return d.attr("get")(key);
}
static I Prefix(O tokens, O entries) {
    I best = 0;
    L input = list(tokens);
    for (auto entry : entries) {
        L ids = list(entry);
        I count = 0, limit = std::min(py::len(input), py::len(ids));
        while (count < limit && equal(input[count], ids[count]))
            ++count;
        best = std::max(best, count);
    }
    return best;
}
static O Hash(O text) {
    auto digest = module("hashlib").attr("sha256")(text.attr("encode")()).attr("digest")();
    return module("builtins").attr("int").attr("from_bytes")(digest[py::slice(0, 8, 1)], "big");
}
static bool Less(O a, O b) {
    int result = PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_LT);
    if (result < 0)
        throw py::error_already_set();
    return result != 0;
}
static void ClearBindings(O s, const std::string &name) {
    D bindings = get(s, "_bindings"), out;
    for (auto p : bindings)
        if (str(item(p.second, py::int_(0))) != name)
            out[p.first] = p.second;
    put(s, "_bindings", out);
}
static void Origin(O url) {
    auto address = module("urllib.parse").attr("urlsplit")(url);
    auto scheme = str(get(address, "scheme")), path = str(get(address, "path"));
    require((scheme == "http" || scheme == "https") && truth(get(address, "hostname")) &&
                !truth(get(address, "username")) && !truth(get(address, "password")) &&
                !truth(get(address, "query")) && !truth(get(address, "fragment")) &&
                (path.empty() || path == "/"),
            "service URL must be an HTTP(S) origin");
}

/** 注册、缓存与计数只在同一锁内更新，选路不能看到半份快照。 */
void BindRouting(Handlers &h) {
    h["routing.Registry.__init__"] = [](D a) {
        auto s = self(a);
        D policies;
        policies["full"] = at(a, "policy");
        policies["p"] = truth(at(a, "prefill_policy")) ? at(a, "prefill_policy") : at(a, "policy");
        policies["d"] = truth(at(a, "decode_policy")) ? at(a, "decode_policy") : at(a, "policy");
        auto allowed = cls("routing", "POLICIES");
        for (auto p : policies)
            require(truth(module("operator").attr("contains")(allowed, p.second)),
                    "unsupported router policy");
        for (const char *key :
             {"cache_threshold", "balance_absolute", "balance_relative", "session_ttl", "load_ttl"})
            require(std::isfinite(real(at(a, key))), "nonfinite router limit");
        require(real(at(a, "cache_threshold")) >= 0 && real(at(a, "cache_threshold")) <= 1 &&
                    real(at(a, "balance_absolute")) >= 0 && real(at(a, "balance_relative")) >= 1,
                "invalid cache thresholds");
        for (const char *key :
             {"failure_threshold", "success_threshold", "session_ttl", "history_limit", "load_ttl"})
            require(real(at(a, key)) > 0, "invalid router limit");
        put(s, "policies", policies);
        for (const char *key :
             {"cache_threshold", "balance_absolute", "balance_relative", "failure_threshold",
              "success_threshold", "session_ttl", "history_limit", "load_ttl"})
            put(s, key, at(a, key));
        put(s, "_clock", at(a, "clock"));
        put(s, "_rng", truth(at(a, "rng")) ? at(a, "rng") : module("random").attr("SystemRandom")());
        put(s, "_lock", module("threading").attr("RLock")());
        for (const char *key : {"_workers", "_bindings", "_leases", "_cursor"})
            put(s, key, D());
        put(s, "_next_ticket", I(0));
        return none();
    };

    h["routing.Registry.register"] = [](D a) {
        O s = self(a), name = at(a, "name"), url = at(a, "url"), role = at(a, "role"), model = at(a, "model"),
          boot = at(a, "boot");
        identifier(name, "service name");
        identifier(model, "model");
        identifier(boot, "boot");
        Origin(url);
        require(str(role) == "full" || str(role) == "p" || str(role) == "d", "invalid service role");
        Lock lock(get(s, "_lock"));
        D workers = dictfield(s, "_workers");
        O old = Get(workers, name), clean = url.attr("rstrip")("/");
        if (!old.is_none()) {
            if (equal(get(old, "url"), clean) && equal(get(old, "role"), role) &&
                equal(get(old, "model"), model) && equal(get(old, "boot"), boot))
                return none();
            require(!equal(get(old, "boot"), boot), "service identity changed without new boot ID");
        }
        s.attr("unregister")(name);
        workers[name] = cls("routing", "Worker")(name, clean, role, model, boot);
        return none();
    };

    h["routing.Registry.unregister"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_lock"));
        get(s, "_workers").attr("pop")(at(a, "name"), none());
        ClearBindings(s, str(at(a, "name")));
        return none();
    };

    h["routing.Registry.health_result"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_lock"));
        O w = Get(dictfield(s, "_workers"), at(a, "name"));
        if (w.is_none() || !equal(get(w, "boot"), at(a, "boot")))
            return none();
        if (truth(at(a, "success"))) {
            put(w, "failures", I(0));
            I n = num(get(w, "successes")) + 1;
            put(w, "successes", n);
            if (n >= num(get(s, "success_threshold")))
                put(w, "healthy", true);
        } else {
            put(w, "successes", I(0));
            I n = num(get(w, "failures")) + 1;
            put(w, "failures", n);
            if (n >= num(get(s, "failure_threshold"))) {
                put(w, "healthy", false);
                ClearBindings(s, str(at(a, "name")));
            }
        }
        return none();
    };

    h["routing.Registry.update_load"] = [](D a) {
        integer(at(a, "count"), "unfinished requests");
        auto s = self(a);
        Lock lock(get(s, "_lock"));
        auto w = Get(dictfield(s, "_workers"), at(a, "name"));
        if (!w.is_none() && equal(get(w, "boot"), at(a, "boot"))) {
            put(w, "backend_load", at(a, "count"));
            put(w, "load_at", get(s, "_clock")());
        }
        return none();
    };

    h["routing.Registry.cache_event"] = [](D a) {
        auto s = self(a);
        I sequence = integer(at(a, "sequence"), "cache sequence", 1);
        auto operation = str(at(a, "operation"));
        require(operation == "add" || operation == "remove" || operation == "reset",
                "invalid cache operation");
        identifier(at(a, "key"), "cache key");
        L ids;
        for (auto t : at(a, "tokens"))
            ids.append(py::int_(integer(py::reinterpret_borrow<O>(t), "token", 0, 0xffffffff)));
        Lock lock(get(s, "_lock"));
        auto w = Get(dictfield(s, "_workers"), at(a, "name"));
        if (w.is_none() || !equal(get(w, "boot"), at(a, "boot")) || sequence <= num(get(w, "cache_sequence")))
            return py::bool_(false);
        I previous = num(get(w, "cache_sequence"));
        if (operation != "reset" && (previous == -1 || sequence != previous + 1)) {
            get(w, "cached").attr("clear")();
            put(w, "cache_sequence", I(-1));
            return py::bool_(false);
        }
        if (operation == "reset")
            get(w, "cached").attr("clear")();
        else if (operation == "add") {
            require(py::len(ids) > 0, "cache add requires token IDs");
            dictfield(w, "cached")[at(a, "key")] = tuple(ids);
        } else
            get(w, "cached").attr("pop")(at(a, "key"), none());
        put(w, "cache_sequence", sequence);
        return py::bool_(true);
    };

    h["routing.Registry.cache_snapshot"] = [](D a) {
        D parsed;
        for (auto entry : py::cast<D>(at(a, "entries"))) {
            auto key = identifier(py::reinterpret_borrow<O>(entry.first), "cache key");
            L ids;
            for (auto t : py::reinterpret_borrow<O>(entry.second))
                ids.append(py::int_(integer(py::reinterpret_borrow<O>(t), "token", 0, 0xffffffff)));
            parsed[key] = tuple(ids);
        }
        auto s = self(a);
        Lock lock(get(s, "_lock"));
        auto w = Get(dictfield(s, "_workers"), at(a, "name"));
        if (w.is_none() || !equal(get(w, "boot"), at(a, "boot")))
            return py::bool_(false);
        put(w, "cached", parsed);
        put(w, "cache_sequence", std::max(I(0), num(get(w, "cache_sequence"))) + 1);
        return py::bool_(true);
    };

    h["routing.Registry._prefix"] = [](D a) { return py::int_(Prefix(at(a, "tokens"), at(a, "entries"))); };
    h["routing.Registry._hash"] = [](D a) { return Hash(at(a, "text")); };
    h["routing.Registry.select"] = [](D a) -> O {
        O s = self(a), role = at(a, "role"), model = at(a, "model"), session = at(a, "session");
        L input;
        for (auto t : at(a, "tokens"))
            input.append(py::int_(integer(py::reinterpret_borrow<O>(t), "token", 0, 0xffffffff)));
        if (!session.is_none())
            identifier(session, "session");
        Lock lock(get(s, "_lock"));
        double current = real(get(s, "_clock")());
        D bindings;
        for (auto entry : dictfield(s, "_bindings"))
            if (real(item(entry.second, py::int_(2))) > current)
                bindings[entry.first] = entry.second;
        put(s, "_bindings", bindings);
        std::vector<O> candidates;
        for (auto entry : dictfield(s, "_workers")) {
            O w = py::reinterpret_borrow<O>(entry.second);
            if (truth(get(w, "healthy")) && equal(get(w, "role"), role) && equal(get(w, "model"), model) &&
                !truth(module("operator").attr("contains")(at(a, "exclude"), get(w, "name"))))
                candidates.push_back(w);
        }
        if (candidates.empty())
            fail("no healthy service for requested role/model", PyExc_LookupError);
        std::sort(candidates.begin(), candidates.end(),
                  [](O x, O y) { return str(get(x, "name")) < str(get(y, "name")); });
        O cursorKey = py::make_tuple(role, model);
        D cursors = dictfield(s, "_cursor");
        size_t cursor = num(cursors.attr("get")(cursorKey, 0)) % candidates.size();
        auto ordered = candidates;
        std::rotate(ordered.begin(), ordered.begin() + cursor, ordered.end());
        O lowest = *std::min_element(ordered.begin(), ordered.end(), [](O x, O y) {
            return num(get(x, "inflight")) < num(get(y, "inflight"));
        });
        auto policy = str(at(dictfield(s, "policies"), str(role).c_str()));
        O chosen = none();
        O sessionKey = truth(session) ? O(py::make_tuple(role, model, session)) : none();
        L pool;
        for (auto w : candidates)
            pool.append(w);
        if (policy == "round_robin")
            chosen = ordered.front();
        else if (policy == "random")
            chosen = get(s, "_rng").attr("choice")(pool);
        else if (policy == "power_of_two") {
            L pair = get(s, "_rng").attr("sample")(pool, std::min(size_t(2), candidates.size()));
            double best = std::numeric_limits<double>::infinity();
            for (auto entry : pair) {
                O w = py::reinterpret_borrow<O>(entry);
                O cached = get(w, "backend_load");
                double load =
                    !cached.is_none() && current - real(get(w, "load_at")) <= real(get(s, "load_ttl"))
                        ? real(cached)
                        : real(get(w, "inflight"));
                if (load < best) {
                    best = load;
                    chosen = w;
                }
            }
        } else if (policy == "cache_aware" || policy == "kv_aware") {
            I maximum = 0, minimum = num(get(lowest, "inflight")), best = -1;
            O matched = none();
            for (auto w : ordered) {
                maximum = std::max(maximum, num(get(w, "inflight")));
                I count =
                    Prefix(input, get(w, policy == "cache_aware" ? "history" : "cached").attr("values")());
                if (count > best) {
                    best = count;
                    matched = w;
                }
            }
            bool imbalance = maximum - minimum > real(get(s, "balance_absolute")) &&
                             maximum > minimum * real(get(s, "balance_relative"));
            chosen = !imbalance && py::len(input) > 0 &&
                             double(best) / py::len(input) > real(get(s, "cache_threshold"))
                         ? matched
                         : lowest;
        } else if (policy == "sticky_least_loaded") {
            auto bound = sessionKey.is_none() ? none() : Get(bindings, sessionKey);
            for (auto w : candidates)
                if (!bound.is_none() && equal(get(w, "name"), bound[py::int_(0)]) &&
                    equal(get(w, "boot"), bound[py::int_(1)]))
                    chosen = w;
            if (chosen.is_none()) {
                I best = std::numeric_limits<I>::max();
                for (auto w : ordered) {
                    I count = 0;
                    for (auto binding : bindings) {
                        O key = py::reinterpret_borrow<O>(binding.first),
                          value = py::reinterpret_borrow<O>(binding.second);
                        if (equal(key[py::int_(0)], role) && equal(key[py::int_(1)], model) &&
                            equal(value[py::int_(0)], get(w, "name")))
                            ++count;
                    }
                    if (count < best) {
                        best = count;
                        chosen = w;
                    }
                }
                if (sessionKey.is_none())
                    chosen = lowest;
            }
            if (!sessionKey.is_none())
                bindings[sessionKey] = py::make_tuple(get(chosen, "name"), get(chosen, "boot"),
                                                      current + real(get(s, "session_ttl")));
        } else {
            O digest =
                module("hashlib").attr("sha256")(py::repr(tuple(input)).attr("encode")()).attr("hexdigest")();
            std::string key = str(truth(session) ? session : digest);
            if (policy == "rendezvous_hash") {
                O best = none();
                for (auto w : candidates) {
                    O hash = Hash(py::str(key + std::string(1, '\0') + str(get(w, "url"))));
                    if (best.is_none() || Less(best, hash)) {
                        best = hash;
                        chosen = w;
                    }
                }
            } else {
                std::vector<std::pair<O, O>> ring;
                for (auto w : candidates)
                    for (int i = 0; i < 128; ++i)
                        ring.emplace_back(
                            Hash(py::str(str(get(w, "url")) + std::string(1, '\0') + std::to_string(i))), w);
                std::sort(ring.begin(), ring.end(),
                          [](const auto &x, const auto &y) { return Less(x.first, y.first); });
                O hash = Hash(py::str(key));
                auto found = std::lower_bound(ring.begin(), ring.end(), hash,
                                              [](const auto &x, O y) { return Less(x.first, y); });
                chosen = (found == ring.end() ? ring.front() : *found).second;
            }
        }
        auto index = std::find_if(candidates.begin(), candidates.end(), [&](O w) { return w.is(chosen); }) -
                     candidates.begin();
        cursors[cursorKey] = (index + 1) % candidates.size();
        if (py::len(input)) {
            O history = get(chosen, "history"), ids = tuple(input);
            O key = module("hashlib").attr("sha256")(py::repr(ids).attr("encode")()).attr("hexdigest")();
            history[key] = ids;
            history.attr("move_to_end")(key);
            while (py::len(history) > size_t(num(get(s, "history_limit"))))
                history.attr("popitem")(py::arg("last") = false);
        }
        put(chosen, "inflight", num(get(chosen, "inflight")) + 1);
        I ticket = num(get(s, "_next_ticket")) + 1;
        put(s, "_next_ticket", ticket);
        O lease = cls("routing", "Lease")(get(chosen, "name"), get(chosen, "boot"), ticket);
        dictfield(s, "_leases")[py::int_(ticket)] = lease;
        return lease;
    };

    h["routing.Registry.worker"] = [](D a) {
        auto s = self(a), lease = at(a, "lease");
        Lock lock(get(s, "_lock"));
        O w = Get(dictfield(s, "_workers"), get(lease, "worker"));
        if (w.is_none() || !equal(get(w, "boot"), get(lease, "boot")))
            fail("service removed or restarted", PyExc_LookupError);
        return w;
    };

    h["routing.Registry.release"] = [](D a) {
        auto s = self(a), lease = at(a, "lease");
        Lock lock(get(s, "_lock"));
        D leases = dictfield(s, "_leases");
        O ticket = get(lease, "ticket");
        if (!equal(Get(leases, ticket), lease))
            return none();
        leases.attr("pop")(ticket);
        auto w = Get(dictfield(s, "_workers"), get(lease, "worker"));
        if (!w.is_none() && equal(get(w, "boot"), get(lease, "boot")))
            put(w, "inflight", num(get(w, "inflight")) - 1);
        return none();
    };

    h["routing.Registry.snapshot"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "_lock"));
        L out;
        for (auto entry : dictfield(s, "_workers")) {
            O w = py::reinterpret_borrow<O>(entry.second);
            D row;
            for (const char *key : {"name", "url", "role", "model", "boot", "healthy", "inflight"})
                row[key] = get(w, key);
            out.append(row);
        }
        return out;
    };
}
} // namespace pd
} // namespace fastllm
