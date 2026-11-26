#include "pd_bindings.h"
#include <atomic>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace fastllm {
namespace pd {
static bool stopped(O event) {
    return !event.is_none() && truth(event.attr("is_set")());
}
static O command(O s, const char *action, D fields = D()) {
    return s.attr("_command")(action, **fields);
}
static double deadline(O s, O value) {
    return value.is_none() ? now() + real(get(get(s, "session"), "timeout")) : real(value);
}
static void pause(O event, double seconds) {
    if (event.is_none())
        module("time").attr("sleep")(seconds);
    else
        event.attr("wait")(seconds);
}

/** 接收 worker 的层就绪报文；仅匹配本次 key/rank/layer，计算入口仍使用既有会话。 */
static void Prepared(O s, O identity, O tokens, O callback) {
    O key, manifest;
    {
        Lock lock(get(s, "condition"));
        O saved = item(get(s, "exports"), get(identity, "transfer"));
        protocol(equal(saved[py::int_(0)], identity) && equal(get(s, "active"), identity),
                 "prepared Prefill identity mismatch");
        key = saved[py::int_(1)];
        manifest = saved[py::int_(2)];
    }
    char pattern[] = "/tmp/ft-pd-XXXXXX";
    char *directory = mkdtemp(pattern);
    if (!directory)
        throw std::runtime_error("cannot create PD notification directory");
    std::string path = std::string(directory) + "/ready.sock";
    int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    if (fd < 0 || bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address))) {
        if (fd >= 0)
            close(fd);
        unlink(path.c_str());
        rmdir(directory);
        throw std::runtime_error("cannot bind PD notification socket");
    }
    uint64_t expected = std::stoull(str(key), nullptr, 16);
    I ranks = num(get(get(manifest, "layout"), "tp_size")),
      layers = num(get(get(manifest, "layout"), "layers"));
    std::atomic<bool> done{false};
    std::string failure;
    std::thread listener([&] {
        while (!done.load()) {
            pollfd p{fd, POLLIN, 0};
            int status = poll(&p, 1, 50);
            if (status <= 0)
                continue;
            uint64_t data[4]{};
            auto size = recv(fd, data, 25, 0);
            if (size != 24 || data[0] != expected || data[1] >= uint64_t(ranks) ||
                data[2] >= uint64_t(layers))
                continue;
            py::gil_scoped_acquire acquire;
            try {
                callback(data[1], data[2]);
            } catch (const std::exception &error) {
                failure = error.what();
                return;
            }
        }
    });
    auto cleanup = [&] {
        done.store(true);
        {
            py::gil_scoped_release release;
            listener.join();
        }
        close(fd);
        unlink(path.c_str());
        rmdir(directory);
    };
    try {
        get(s, "session")
            .attr("pd_command")(D("action"_a = "prepared_prefill", "key"_a = key, "token_ids"_a = tokens,
                                  "notify_path"_a = path));
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    if (!failure.empty())
        throw std::runtime_error(failure);
    s.attr("refresh_cache")();
}

void BindNative(Handlers &h) {
    h["native.NativeReplica.__init__"] = [](D a) {
        auto s = self(a);
        D config = at(a, "config");
        auto type = str(config.attr("get")("model_type"));
        require((type == "qwen3" || type == "qwen2_moe") && !truth(config.attr("get")("quantization_config")),
                "native PD requires unquantized Qwen3 or Qwen2 MoE");
        for (const char *key : {"session", "peer", "config", "model_id", "growth"})
            put(s, key, at(a, key));
        put(s, "condition", module("threading").attr("Condition")(module("threading").attr("RLock")()));
        put(s, "active", none());
        put(s, "exports", D());
        put(s, "waiters", L());
        put(s, "closed", false);
        put(s, "cached", D());
        put(s, "preemptions", I(0));
        put(s, "kv_usage", py::float_(0));
        return none();
    };

    h["native.NativeReplica.layout"] = [](D a) {
        auto s = self(a), session = get(s, "session");
        D cfg = get(s, "config");
        auto dim =
            cfg.attr("get")("head_dim", num(at(cfg, "hidden_size")) / num(at(cfg, "num_attention_heads")));
        return cls("protocol",
                   "Layout")(get(s, "model_id"), at(cfg, "num_hidden_layers"), at(cfg, "num_key_value_heads"),
                             dim, "float16", py::len(get(session, "devices")),
                             py::getattr(session, "storage", py::str("paged")), get(session, "page_size"));
    };

    h["native.NativeReplica.refresh_cache"] = [](D a) {
        auto s = self(a);
        O states = command(s, "cache"), common = none();
        for (auto entry : states) {
            auto entries = py::set();
            for (auto snapshot : at(at(entry, "prefix_cache"), "snapshots")) {
                auto ids = at(snapshot, "token_ids");
                if (num(at(snapshot, "position_begin")) == 0 && truth(ids))
                    entries.add(tuple(ids));
            }
            common = common.is_none() ? O(entries) : common.attr("intersection")(entries);
        }
        D cached;
        if (!common.is_none())
            for (auto ids : common)
                cached[cls("protocol", "token_digest")(ids)] = list(ids);
        Lock lock(get(s, "condition"));
        put(s, "cached", cached);
        return none();
    };

    h["native.NativeReplica._command"] = [](D a) {
        auto s = self(a);
        D fields = module("builtins").attr("dict")(at(a, "fields"));
        if (!at(a, "key").is_none()) {
            for (const char *key : {"tokens", "capacity", "position"}) {
                require(!fields.contains(key), "duplicate native command field");
                fields[key] = 0;
            }
            fields["key"] = at(a, "key");
        }
        fields["action"] = at(a, "action");
        auto reply = get(s, "session").attr("pd_command")(fields);
        if (str(at(a, "action")) == "usage")
            put(s, "kv_usage", at(reply, "kv_cache_usage"));
        return reply;
    };

    h["native.NativeReplica.score"] = [](D a) {
        auto s = self(a);
        double wait = real(at(a, "waiting"));
        return py::float_(wait + (!get(s, "active").is_none() ? 1 : 0) +
                          wait * 6 * std::max(0.0, real(get(s, "kv_usage")) - 0.5));
    };

    h["native.NativeReplica.acquire"] = [](D a) {
        auto s = self(a), condition = get(s, "condition"), cancel = at(a, "cancelled");
        Lock lock(condition);
        O ticket = module("builtins").attr("object")();
        get(s, "waiters").attr("append")(ticket);
        auto cleanup = [&] {
            get(s, "waiters").attr("remove")(ticket);
            condition.attr("notify_all")();
        };
        try {
            while (!get(s, "active").is_none() || !item(get(s, "waiters"), py::int_(0)).is(ticket)) {
                if (stopped(cancel) || truth(get(s, "closed")))
                    fail("request cancelled while queued", PyExc_InterruptedError);
                double remaining = real(at(a, "deadline")) - now();
                if (remaining <= 0)
                    fail("replica queue deadline exceeded", PyExc_TimeoutError);
                condition.attr("wait")(std::min(remaining, 0.05));
            }
            if (stopped(cancel) || truth(get(s, "closed")) || now() >= real(at(a, "deadline")))
                fail("request no longer admissible", PyExc_InterruptedError);
            put(s, "active", at(a, "identity"));
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
        return none();
    };

    h["native.NativeReplica.wait_capacity"] = [](D a) {
        auto s = self(a);
        I page = num(get(get(s, "session"), "page_size"));
        if (page <= 0)
            page = 128;
        I reserve = truth(at(a, "decode")) ? std::min(num(get(s, "growth")), num(at(a, "remaining"))) : 0;
        I needed = (num(at(a, "prompt")) + reserve + page - 1) / page;
        while (true) {
            O states = at(command(s, "usage"), "workers");
            bool fits = truth(states), impossible = false;
            for (auto entry : states) {
                I total = num(at(entry, "total_pages")), free = num(at(entry, "free_pages"));
                if (total && free < needed)
                    fits = false;
                if (total && needed > total)
                    impossible = true;
            }
            if (fits)
                return none();
            require(!impossible, "request exceeds KV pool capacity including growth budget");
            if (truth(at(a, "cancelled").attr("wait")(0.05)))
                fail("cancelled waiting for KV capacity", PyExc_InterruptedError);
            if (now() >= real(at(a, "deadline")))
                fail("KV admission deadline exceeded", PyExc_TimeoutError);
        }
    };

    h["native.NativeReplica._manifest"] = [](D a) {
        auto s = self(a), layout = get(s, "layout"), regions = at(a, "regions");
        if (!truth(regions))
            fail("native worker did not allocate receiving KV", PyExc_RuntimeError);
        L converted;
        for (auto entry : regions) {
            D fields = module("builtins").attr("dict")(entry);
            protocol(equal(fields.attr("pop")("head_dim"), get(layout, "head_dim")) &&
                         equal(fields.attr("pop")("page_size"), get(layout, "page_size")),
                     "native layout mismatch");
            O bytes = module("builtins").attr("bytes").attr("fromhex")(fields.attr("pop")("memory_hex"));
            fields["memory"] = module("base64").attr("b64encode")(bytes).attr("decode")();
            for (const char *key : {"allocation_bytes", "offset", "token_stride", "head_stride"}) {
                O value = at(fields, key), integerValue = module("builtins").attr("int")(value);
                protocol(equal(integerValue, value), "nonintegral native byte range");
                fields[key] = integerValue;
            }
            converted.append(cls("protocol", "Region")(**fields));
        }
        return cls("protocol",
                   "Manifest")(1, at(a, "identity"), get(s, "peer"), layout, 0, py::len(at(a, "tokens")),
                               cls("protocol", "token_digest")(at(a, "tokens")), tuple(converted));
    };

    h["native.NativeReplica.prefill"] = [](D a) {
        auto s = self(a);
        {
            Lock lock(get(s, "condition"));
            protocol(equal(get(s, "active"), at(a, "identity")), "prefill does not own replica");
        }
        command(s, "prefill", D("token_ids"_a = at(a, "tokens"), "capacity"_a = py::len(at(a, "tokens"))));
        s.attr("refresh_cache")();
        return s.attr("export")(at(a, "identity"), at(a, "tokens"), "deadline"_a = at(a, "deadline"),
                                "cancelled"_a = at(a, "cancelled"));
    };

    h["native.NativeReplica.prepared_prefill"] = [](D a) {
        Prepared(self(a), at(a, "identity"), at(a, "tokens"), at(a, "on_layer"));
        return none();
    };
    for (bool prepare : {false, true})
        h[prepare ? "native.NativeReplica.prepare" : "native.NativeReplica.export"] = [prepare](D a) -> O {
            auto s = self(a), id = at(a, "identity"), tokens = at(a, "tokens"), cancel = at(a, "cancelled");
            {
                Lock lock(get(s, "condition"));
                protocol(equal(get(s, "active"), id) &&
                             !py::cast<D>(get(s, "exports")).contains(get(id, "transfer")),
                         "allocation does not own request or already exported");
            }
            O key = uuid()[py::slice(0, 16, 1)];
            double end = deadline(s, at(a, "deadline"));
            O native;
            while (true) {
                if (stopped(cancel) || now() >= end)
                    fail("KV allocation cancelled or expired", PyExc_TimeoutError);
                native =
                    get(s, "session")
                        .attr("pd_command")(D(
                            "action"_a = prepare ? "prepare" : "export", "key"_a = key,
                            "tokens"_a = py::len(tokens),
                            "capacity"_a = prepare ? I(py::len(tokens)) + num(at(a, "output_tokens")) - 1 : 0,
                            "position"_a = 0));
                if (truth(at(native, "regions")))
                    break;
                if (!prepare && truth(at(a, "optional")))
                    return none();
                pause(cancel, 0.05);
            }
            O manifest;
            try {
                manifest = s.attr("_manifest")(id, tokens, at(native, "regions"));
            } catch (...) {
                s.attr("_command")("release", key);
                if (prepare)
                    command(s, "end");
                throw;
            }
            {
                Lock lock(get(s, "condition"));
                py::cast<D>(get(s, "exports"))[get(id, "transfer")] = py::make_tuple(id, key, manifest);
            }
            s.attr("refresh_cache")();
            return manifest;
        };

    h["native.NativeReplica.commit"] = [](D a) {
        auto s = self(a), id = at(a, "identity");
        Lock lock(get(s, "condition"));
        protocol(equal(get(s, "active"), id), "commit does not own replica");
        O saved = item(get(s, "exports"), get(id, "transfer"));
        protocol(equal(saved[py::int_(0)], id) &&
                     equal(get(saved[py::int_(2)], "token_end"), at(a, "tokens")),
                 "stale transfer commit");
        get(s, "session")
            .attr("pd_command")(D("action"_a = "commit", "key"_a = saved[py::int_(1)],
                                  "tokens"_a = num(at(a, "tokens")) - 1, "capacity"_a = 0, "position"_a = 0));
        s.attr("release")(id);
        return none();
    };

    h["native.NativeReplica.restore_local"] = [](D a) {
        auto s = self(a), id = at(a, "identity");
        O key;
        {
            Lock lock(get(s, "condition"));
            O saved = item(get(s, "exports"), get(id, "transfer"));
            protocol(equal(saved[py::int_(0)], id) && equal(get(s, "active"), id),
                     "local restore identity mismatch");
            key = saved[py::int_(1)];
        }
        auto reply = get(s, "session")
                         .attr("pd_command")(D("action"_a = "restore_local", "key"_a = key,
                                               "token_ids"_a = at(a, "tokens"), "tokens"_a = 0,
                                               "capacity"_a = 0, "position"_a = 0));
        I count = integer(at(reply, "restored_tokens"), "restored prefix", 0, py::len(at(a, "tokens")) - 1);
        return py::int_(count);
    };

    h["native.NativeReplica.release"] = [](D a) {
        auto s = self(a), id = at(a, "identity");
        Lock lock(get(s, "condition"));
        O exports = get(s, "exports"), saved = exports.attr("get")(get(id, "transfer"));
        if (saved.is_none())
            return none();
        protocol(equal(saved[py::int_(0)], id), "export release identity mismatch");
        s.attr("_command")("release", saved[py::int_(1)]);
        exports.attr("pop")(get(id, "transfer"));
        command(s, "usage");
        get(s, "condition").attr("notify_all")();
        return none();
    };

    h["native.NativeReplica.finish"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "condition"));
        if (!equal(get(s, "active"), at(a, "identity")))
            return none();
        command(s, "end");
        put(s, "active", none());
        command(s, "usage");
        get(s, "condition").attr("notify_all")();
        return none();
    };

    h["native.NativeReplica.generate"] = [](D a) {
        auto s = self(a), id = at(a, "identity"), cancel = at(a, "cancelled");
        D request = at(a, "request");
        D sampling("temperature"_a = request.attr("get")("temperature", 0.0),
                   "top_k"_a = request.attr("get")("top_k", 0), "top_p"_a = request.attr("get")("top_p", 1.0),
                   "seed"_a = request.attr("get")("seed", 0));
        O input = at(request, "token_ids"), token = input[py::int_(-1)],
          eos = request.attr("get")("eos_token_ids", L());
        I maximum = num(request.attr("get")("max_new_tokens", 32)),
          page = num(get(get(s, "session"), "page_size"));
        if (page <= 0)
            page = 128;
        double end = deadline(s, at(a, "deadline"));
        L generated;
        auto cleanup = [&] {
            if (str(get(get(s, "session"), "state")) == "ready")
                command(s, "sampling_end", D("context"_a = get(id, "transfer")));
        };
        try {
            for (I index = 0; index < maximum; ++index) {
                if (stopped(cancel) || stopped(at(a, "_closed")) || now() >= end)
                    fail("decode cancelled or timed out", PyExc_InterruptedError);
                protocol(equal(get(s, "active"), id), "decode does not own replica");
                I position = py::len(input) - 1 + index;
                if (str(py::getattr(get(s, "session"), "storage", py::str("paged"))) == "paged" &&
                    position % page == 0) {
                    bool full = false;
                    for (auto state : at(command(s, "usage"), "workers"))
                        if (num(at(state, "free_pages")) == 0)
                            full = true;
                    if (full) {
                        s.attr("finish")(id);
                        put(s, "preemptions", num(get(s, "preemptions")) + 1);
                        s.attr("acquire")(id, end, cancel);
                        O history = module("operator").attr("add")(input, generated);
                        s.attr("wait_capacity")(py::len(history), true, maximum - index - 1, end, cancel);
                        if (position)
                            command(s, "prefill",
                                    D("token_ids"_a = history[py::slice(0, I(py::len(history)) - 1, 1)],
                                      "capacity"_a = I(py::len(input)) + maximum - 1));
                        else {
                            s.attr("prepare")(id, history, maximum);
                            s.attr("commit")(id, 1);
                        }
                    }
                }
                token = at(command(s, "step",
                                   D("input"_a = token, "sampling"_a = sampling, "first"_a = index == 0,
                                     "context"_a = get(id, "transfer"))),
                           "token");
                generated.append(token);
                O finish = none();
                if (truth(module("operator").attr("contains")(eos, token)))
                    finish = py::str("eos");
                else if (index + 1 == maximum)
                    finish = py::str("length");
                at(a, "_emit")(D("event"_a = "token_delta", "index"_a = index,
                                 "token_ids"_a = list(py::make_tuple(token)), "finish_reason"_a = finish));
                if (!finish.is_none())
                    break;
            }
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
        return none();
    };
}
} // namespace pd
} // namespace fastllm
