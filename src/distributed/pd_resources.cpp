#include "pd_bindings.h"

namespace fastllm {
namespace pd {
static D records(O s) {
    return py::cast<D>(get(s, "records"));
}
static I available(O s) {
    I total = num(get(s, "capacity"));
    for (auto entry : records(s)) {
        O r = py::reinterpret_borrow<O>(entry.second);
        total -= num(get(r, "used")) + num(get(r, "growth"));
    }
    return total;
}

/** 请求预算、历史租约和重试边界由原生逻辑决定；回调仅执行外部资源操作。 */
void BindResources(Handlers &h) {
    h["retry.RetryBudget.__post_init__"] = [](D a) {
        auto s = self(a);
        integer(get(s, "attempts"), "retry attempts", 1, 100);
        double base = real(get(s, "base")), maximum = real(get(s, "maximum"));
        require(std::isfinite(real(get(s, "deadline"))) && std::isfinite(base) && std::isfinite(maximum) &&
                    base > 0 && base <= maximum,
                "invalid retry timing");
        return none();
    };

    h["retry.RetryBudget.begin"] = [](D a) {
        auto s = self(a);
        double current = at(a, "now").is_none() ? now() : real(at(a, "now"));
        if (truth(get(s, "cancelled")) || truth(get(s, "responded")) || current >= real(get(s, "deadline")) ||
            num(get(s, "issued")) >= num(get(s, "attempts")))
            fail("request retry budget exhausted", PyExc_TimeoutError);
        put(s, "issued", num(get(s, "issued")) + 1);
        return uuid();
    };

    h["retry.RetryBudget.delay"] = [](D a) -> O {
        auto s = self(a), status = at(a, "status");
        double current = at(a, "now").is_none() ? now() : real(at(a, "now"));
        bool retryable = truth(at(a, "transport_error")) ||
                         (PyLong_CheckExact(status.ptr()) &&
                          (num(status) == 429 || (num(status) >= 500 && num(status) <= 599)));
        if (!retryable || truth(get(s, "cancelled")) || truth(get(s, "responded")) ||
            num(get(s, "issued")) >= num(get(s, "attempts")))
            return none();
        double bound =
            std::min(real(get(s, "maximum")),
                     std::ldexp(real(get(s, "base")), int(std::max(I(0), num(get(s, "issued")) - 1))));
        O delay = at(a, "rng").attr("uniform")(0, bound);
        return current + real(delay) < real(get(s, "deadline")) ? delay : none();
    };

    h["admission.Admission.__init__"] = [](D a) {
        auto s = self(a);
        put(s, "capacity", integer(at(a, "capacity"), "KV capacity", 1));
        put(s, "growth", integer(at(a, "growth"), "growth budget"));
        put(s, "clock", at(a, "clock"));
        put(s, "records", D());
        put(s, "waiting", module("collections").attr("OrderedDict")());
        put(s, "lock", module("threading").attr("RLock")());
        return none();
    };

    h["admission.Admission.available"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "lock"));
        return py::int_(available(s));
    };

    h["admission.Admission.request"] = [](D a) {
        auto s = self(a), key = at(a, "key");
        I tokens = integer(at(a, "tokens"), "KV tokens", 1);
        I reserve = truth(at(a, "decode")) ? num(get(s, "growth")) : 0;
        if (!at(a, "remaining").is_none())
            reserve = std::min(reserve, integer(at(a, "remaining"), "remaining tokens"));
        require(tokens <= num(get(s, "capacity")) - reserve, "request exceeds total KV capacity with budget");
        Lock lock(get(s, "lock"));
        D r = records(s);
        O old = r.attr("get")(key);
        if (!old.is_none()) {
            require(num(get(old, "used")) == tokens && num(get(old, "growth")) == reserve,
                    "admission key reused");
            return py::bool_(true);
        }
        O waiting = get(s, "waiting"), want = py::make_tuple(tokens, reserve),
          existing = waiting.attr("get")(key);
        require(existing.is_none() || equal(existing, want), "waiting key reused");
        waiting.attr("setdefault")(key, want);
        auto first = module("builtins").attr("next")(module("builtins").attr("iter")(waiting));
        if (!equal(first, key) || tokens > available(s) - reserve)
            return py::bool_(false);
        waiting.attr("pop")(key);
        r[key] = cls("admission", "Allocation")(key, tokens, reserve, "touched"_a = get(s, "clock")());
        return py::bool_(true);
    };

    h["admission.Admission.ready"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "lock"));
        auto r = item(records(s), at(a, "key"));
        put(r, "state", py::str("running"));
        put(r, "touched", get(s, "clock")());
        return none();
    };

    h["admission.Admission.grow"] = [](D a) {
        auto s = self(a);
        I count = integer(at(a, "count"), "growth count", 1);
        Lock lock(get(s, "lock"));
        auto r = item(records(s), at(a, "key"));
        require(str(get(r, "state")) == "running", "only running request may grow");
        if (std::max(I(0), count - num(get(r, "growth"))) > available(s))
            return py::bool_(false);
        put(r, "used", num(get(r, "used")) + count);
        put(r, "growth", std::max(I(0), num(get(r, "growth")) - count));
        I wanted = num(get(s, "growth"));
        if (!at(a, "remaining").is_none())
            wanted = std::min(wanted, integer(at(a, "remaining"), "remaining"));
        put(r, "growth",
            num(get(r, "growth")) + std::min(std::max(I(0), wanted - num(get(r, "growth"))), available(s)));
        put(r, "touched", get(s, "clock")());
        return py::bool_(true);
    };

    h["admission.Admission.pin"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "lock"));
        auto r = item(records(s), at(a, "key"));
        put(r, "pins", num(get(r, "pins")) + 1);
        return none();
    };

    h["admission.Admission.unpin"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "lock"));
        auto r = item(records(s), at(a, "key"));
        require(num(get(r, "pins")) > 0, "duplicate KV unpin");
        put(r, "pins", num(get(r, "pins")) - 1);
        return none();
    };

    h["admission.Admission.cache"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "lock"));
        auto r = item(records(s), at(a, "key"));
        put(r, "state", py::str("cached"));
        put(r, "growth", I(0));
        put(r, "touched", get(s, "clock")());
        return none();
    };

    h["admission.Admission.victim"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "lock"));
        O cached = none(), running = none();
        for (auto entry : records(s)) {
            auto r = py::reinterpret_borrow<O>(entry.second);
            if (num(get(r, "pins")))
                continue;
            auto state = str(get(r, "state"));
            if (state == "cached" &&
                (cached.is_none() || real(get(r, "touched")) < real(get(cached, "touched"))))
                cached = r;
            if (state == "running")
                running = r;
        }
        auto chosen = !cached.is_none() ? cached : running;
        return chosen.is_none() ? none() : get(chosen, "key");
    };

    h["admission.Admission.retire"] = [](D a) {
        auto s = self(a), key = at(a, "key");
        Lock lock(get(s, "lock"));
        O r = records(s).attr("get")(key);
        if (!r.is_none() && num(get(r, "pins")))
            fail("KV is still in flight", PyExc_RuntimeError);
        records(s).attr("pop")(key, none());
        get(s, "waiting").attr("pop")(key, none());
        return none();
    };

    h["history.History.__init__"] = [](D a) {
        auto s = self(a);
        double ttl = real(at(a, "ttl"));
        require(std::isfinite(ttl) && ttl > 0, "invalid history retention");
        for (const char *key : {"peer", "ttl", "clock"})
            put(s, key, at(a, key));
        put(s, "lock", module("threading").attr("RLock")());
        put(s, "entries", D());
        put(s, "closing", false);
        return none();
    };

    h["history.History.retain"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "lock"));
        D entries = get(s, "entries");
        protocol(!truth(get(s, "closing")) && !entries.contains(at(a, "ticket")) &&
                     I(py::len(at(a, "tokens"))) == num(get(at(a, "manifest"), "token_end")),
                 "invalid history retention");
        entries[at(a, "ticket")] =
            cls("history", "Retained")(at(a, "manifest"), tuple(at(a, "tokens")),
                                       real(get(s, "clock")()) + real(get(s, "ttl")), at(a, "release"));
        return none();
    };

    h["history.History.acquire"] = [](D a) -> O {
        auto s = self(a), id = at(a, "identity");
        Lock lock(get(s, "lock"));
        O entry = get(s, "entries").attr("get")(at(a, "ticket"));
        protocol(str(get(id, "direction")) == "d_to_p" && str(get(id, "mode")) == "pull" &&
                     equal(get(id, "source"), get(s, "peer")),
                 "invalid history identity");
        if (truth(get(s, "closing")) || entry.is_none() ||
            real(get(s, "clock")()) >= real(get(entry, "expires")))
            return none();
        O tokens = at(a, "tokens"), saved = get(entry, "tokens");
        I count = 0, limit = std::min(I(py::len(saved)), I(py::len(tokens)) - 1);
        while (count < limit && equal(saved[py::int_(count)], tokens[py::int_(count)]))
            ++count;
        if (!count)
            return none();
        auto manifest = cls("protocol", "prefix_manifest")(get(entry, "manifest"), id, get(s, "peer"),
                                                           tokens[py::slice(0, count, 1)]);
        D readers = get(entry, "readers");
        O transfer = get(id, "transfer"), old = readers.attr("get")(transfer);
        protocol(old.is_none() || equal(old, manifest), "history lease replay changed");
        readers[transfer] = manifest;
        return manifest;
    };

    h["history.History.release"] = [](D a) {
        auto s = self(a), id = at(a, "identity");
        Lock lock(get(s, "lock"));
        O entry = get(s, "entries").attr("get")(at(a, "ticket"));
        if (entry.is_none())
            return none();
        D readers = get(entry, "readers");
        O old = readers.attr("get")(get(id, "transfer"));
        if (!old.is_none()) {
            protocol(equal(get(old, "identity"), id), "history lease identity mismatch");
            readers.attr("pop")(get(id, "transfer"));
        }
        s.attr("expire")();
        return none();
    };

    h["history.History.expire"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "lock"));
        D entries = get(s, "entries");
        L snapshot = list(entries.attr("items")());
        for (auto p : snapshot) {
            O pair = py::reinterpret_borrow<O>(p), entry = pair[py::int_(1)];
            if (real(get(s, "clock")()) >= real(get(entry, "expires")) && !truth(get(entry, "readers"))) {
                get(entry, "release")();
                entries.attr("pop")(pair[py::int_(0)]);
            }
        }
        return none();
    };

    h["history.History.close"] = [](D a) {
        auto s = self(a);
        Lock lock(get(s, "lock"));
        put(s, "closing", true);
        for (auto p : py::cast<D>(get(s, "entries"))) {
            O entry = py::reinterpret_borrow<O>(p.second);
            put(entry, "expires", py::float_(std::min(real(get(entry, "expires")), real(get(s, "clock")()))));
        }
        s.attr("expire")();
        return py::bool_(!truth(get(s, "entries")));
    };
}
} // namespace pd
} // namespace fastllm
