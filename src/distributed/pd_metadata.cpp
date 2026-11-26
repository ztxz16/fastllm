#include "pd_bindings.h"
#include <cctype>

namespace fastllm {
namespace pd {

/** 严格读取有界整数，拒绝 bool、截断和超出 int64 的协议值。 */
I integer(O value, const std::string &name, I low, I high) {
    protocol(PyLong_CheckExact(value.ptr()), name + " must be an integer");
    int overflow = 0;
    long long result = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (PyErr_Occurred())
        throw py::error_already_set();
    protocol(!overflow && result >= low && result <= high, name + " outside permitted range");
    return result;
}

O identifier(O value, const std::string &name) {
    protocol(py::isinstance<py::str>(value), "invalid " + name);
    const auto text = str(value);
    protocol(!text.empty() && text.size() <= 192 &&
                 std::all_of(text.begin(), text.end(),
                             [](unsigned char c) {
                                 return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                                        (c >= '0' && c <= '9') || c == '_' || c == '.' || c == ':' ||
                                        c == '-';
                             }),
             "invalid " + name);
    return value;
}

static I field(O o, const char *key, I low = 0, I high = std::numeric_limits<I>::max()) {
    return integer(get(o, key), key, low, high);
}

static O Handle(O value) {
    protocol(py::isinstance<py::str>(value) && py::len(value) <= 128, "invalid CUDA IPC handle");
    O bytes;
    try {
        bytes = module("base64").attr("b64decode")(value, py::arg("validate") = true);
    } catch (const py::error_already_set &) {
        protocol(false, "invalid CUDA IPC handle encoding");
    }
    protocol(py::len(bytes) == 64, "CUDA IPC handle must contain 64 bytes");
    return bytes;
}

static bool Compatible(O a, O b) {
    for (const char *key : {"model", "layers", "kv_heads", "head_dim", "dtype"})
        if (!equal(get(a, key), get(b, key)))
            return false;
    return true;
}

static I Element(O layout) {
    auto dtype = str(get(layout, "dtype"));
    protocol(dtype == "float16" || dtype == "bfloat16" || dtype == "float32", "unsupported KV dtype");
    return dtype == "float32" ? 4 : 2;
}

/** 在乘加前检查剩余范围，避免恶意元信息使地址计算溢出。 */
static I Offset(O region, I token, I head) {
    I base = field(region, "offset"), size = field(region, "allocation_bytes", 1);
    I dt = token - field(region, "token_begin"), dh = head - field(region, "head_begin");
    I ts = field(region, "token_stride", 1), hs = field(region, "head_stride", 1);
    protocol(dt >= 0 && dh >= 0 && base < size, "invalid KV address range");
    protocol(dt <= (size - 1 - base) / ts, "KV token offset exceeds allocation");
    base += dt * ts;
    protocol(dh <= (size - 1 - base) / hs, "KV head offset exceeds allocation");
    return base + dh * hs;
}

static void ValidateRegion(O r, O layout, I begin, I end) {
    field(r, "rank", 0, field(layout, "tp_size") - 1);
    identifier(get(r, "device"), "GPU UUID");
    field(r, "layer", 0, field(layout, "layers") - 1);
    I tb = field(r, "token_begin", begin, end - 1), te = field(r, "token_end", tb + 1, end);
    I hb = field(r, "head_begin", 0, field(layout, "kv_heads") - 1);
    I he = field(r, "head_end", hb + 1, field(layout, "kv_heads"));
    I size = field(r, "allocation_bytes", 1), base = field(r, "offset", 0, size - 1);
    I element = Element(layout), width = field(layout, "head_dim") * element;
    I ts = field(r, "token_stride", width), hs = field(r, "head_stride", width);
    auto component = str(get(r, "component"));
    protocol(component == "k" || component == "v", "KV component must be k or v");
    Handle(get(r, "memory"));
    protocol(base % element == 0 && ts % element == 0 && hs % element == 0, "misaligned KV region");
    // 除法形式同时避免 stride 乘法溢出。
    protocol((he - hb - 1 <= (ts - width) / hs) || (te - tb - 1 <= (hs - width) / ts), "overlapping KV rows");
    I last = Offset(r, te - 1, he - 1);
    protocol(width <= size - last, "KV region exceeds exported allocation");
}

static O Digest(O tokens) {
    auto hash = module("hashlib").attr("sha256")();
    for (auto token : tokens) {
        uint32_t value =
            static_cast<uint32_t>(integer(py::reinterpret_borrow<O>(token), "token", 0, 0xffffffff));
        char bytes[4];
        for (int i = 0; i != 4; ++i)
            bytes[i] = static_cast<char>(value >> (i * 8));
        hash.attr("update")(py::bytes(bytes, 4));
    }
    return hash.attr("hexdigest")();
}

/** 对已选定的两组布局生成精确覆盖的拷贝矩形，不选择 DP 副本。 */
O plan(O source, O target) {
    auto identity = get(source, "identity"), layout = get(target, "layout");
    protocol(equal(identity, get(target, "identity")) &&
                 equal(get(source, "owner"), get(identity, "source")) &&
                 equal(get(target, "owner"), get(identity, "target")),
             "source/target transfer identity mismatch");
    protocol(Compatible(get(source, "layout"), layout) &&
                 equal(get(source, "prefix_digest"), get(target, "prefix_digest")),
             "model, KV semantics or prompt prefix mismatch");
    protocol(equal(get(source, "token_begin"), get(target, "token_begin")) &&
                 equal(get(source, "token_end"), get(target, "token_end")),
             "source/target logical token range mismatch");

    auto overlap = [](O a, O b) {
        return field(a, "token_begin") < field(b, "token_end") &&
               field(b, "token_begin") < field(a, "token_end") &&
               field(a, "head_begin") < field(b, "head_end") && field(b, "head_begin") < field(a, "head_end");
    };
    std::vector<O> sources, targets;
    for (auto r : get(source, "regions"))
        sources.push_back(py::reinterpret_borrow<O>(r));
    for (auto r : get(target, "regions"))
        targets.push_back(py::reinterpret_borrow<O>(r));
    std::sort(sources.begin(), sources.end(), [](O a, O b) {
        return std::make_tuple(field(a, "rank"), field(a, "token_begin"), field(a, "head_begin")) <
               std::make_tuple(field(b, "rank"), field(b, "token_begin"), field(b, "head_begin"));
    });
    L copies;
    I width = field(layout, "head_dim") * Element(layout);
    auto Copy = cls("mapping", "Copy");
    for (size_t i = 0; i < targets.size(); ++i) {
        O dst = targets[i];
        for (size_t j = 0; j < i; ++j) {
            O other = targets[j];
            protocol(!(equal(get(dst, "rank"), get(other, "rank")) &&
                       equal(get(dst, "layer"), get(other, "layer")) &&
                       equal(get(dst, "component"), get(other, "component")) && overlap(dst, other)),
                     "overlapping target logical regions");
        }
        I tb = field(dst, "token_begin"), te = field(dst, "token_end"), hb = field(dst, "head_begin"),
          he = field(dst, "head_end");
        std::set<I> tokens{tb, te}, heads{hb, he};
        std::vector<O> candidates;
        for (auto src : sources) {
            if (equal(get(src, "layer"), get(dst, "layer")) &&
                equal(get(src, "component"), get(dst, "component")) && overlap(src, dst)) {
                candidates.push_back(src);
                tokens.insert(std::max(tb, field(src, "token_begin")));
                tokens.insert(std::min(te, field(src, "token_end")));
                heads.insert(std::max(hb, field(src, "head_begin")));
                heads.insert(std::min(he, field(src, "head_end")));
            }
        }
        for (auto t = tokens.begin(); std::next(t) != tokens.end(); ++t) {
            for (auto h = heads.begin(); std::next(h) != heads.end(); ++h) {
                auto found = std::find_if(candidates.begin(), candidates.end(), [&](O s) {
                    return field(s, "token_begin") <= *t && field(s, "token_end") >= *std::next(t) &&
                           field(s, "head_begin") <= *h && field(s, "head_end") >= *std::next(h);
                });
                protocol(found != candidates.end(), "missing source KV layer/token/head range");
                copies.append(Copy(*found, dst, *t, *std::next(t), *h, *std::next(h), width));
            }
        }
    }
    for (I layer = 0; layer < field(layout, "layers"); ++layer)
        for (const char *component : {"k", "v"}) {
            std::set<I> edges{field(target, "token_begin"), field(target, "token_end")};
            std::vector<O> regions;
            for (auto r : targets)
                if (field(r, "layer") == layer && str(get(r, "component")) == component) {
                    regions.push_back(r);
                    edges.insert(field(r, "token_begin"));
                    edges.insert(field(r, "token_end"));
                }
            for (auto t = edges.begin(); std::next(t) != edges.end(); ++t) {
                std::vector<std::pair<I, I>> spans;
                for (auto r : regions)
                    if (field(r, "token_begin") <= *t && field(r, "token_end") >= *std::next(t))
                        spans.emplace_back(field(r, "head_begin"), field(r, "head_end"));
                std::sort(spans.begin(), spans.end());
                I covered = 0;
                for (auto span : spans) {
                    if (span.first > covered)
                        break;
                    covered = std::max(covered, span.second);
                }
                protocol(covered == field(layout, "kv_heads"), "incomplete target KV layout");
            }
        }
    return tuple(copies);
}

void BindProtocol(Handlers &h) {
    h["protocol.integer"] = [](D a) {
        return py::int_(
            integer(at(a, "value"), str(at(a, "name")), num(at(a, "minimum")), num(at(a, "maximum"))));
    };

    h["protocol.identifier"] = [](D a) { return identifier(at(a, "value"), str(at(a, "name"))); };
    h["protocol.new_id"] = [](D) { return uuid(); };
    h["protocol.ipc_handle"] = [](D a) { return Handle(at(a, "value")); };
    h["protocol.token_digest"] = [](D a) { return Digest(at(a, "tokens")); };
    h["protocol._object"] = [](D a) {
        auto type = at(a, "cls"), value = at(a, "value");
        protocol(py::isinstance<D>(value), "invalid metadata object");
        auto fields = module("dataclasses").attr("fields")(type);
        protocol(py::len(value) == py::len(fields), "unexpected metadata fields");
        for (auto f : fields)
            protocol(py::cast<D>(value).contains(get(f, "name")), "missing metadata field");
        return type(**value);
    };

    h["protocol.Peer.__post_init__"] = [](D a) {
        for (const char *key : {"instance", "boot", "host"})
            identifier(get(self(a), key), key);
        return none();
    };

    h["protocol.Identity.__post_init__"] = [](D a) {
        O s = self(a), p = get(s, "source"), d = get(s, "target");
        identifier(get(s, "request"), "request");
        identifier(get(s, "transfer"), "transfer");
        protocol(py::isinstance(p, cls("protocol", "Peer")) && py::isinstance(d, cls("protocol", "Peer")),
                 "invalid peer identities");
        protocol(equal(get(p, "host"), get(d, "host")) && !equal(p, d),
                 "cross-host or same-instance transfer");
        auto mode = str(get(s, "mode")), direction = str(get(s, "direction"));
        protocol((mode == "pull" || mode == "push") && (direction == "p_to_d" || direction == "d_to_p") &&
                     (direction != "d_to_p" || mode == "pull"),
                 "invalid transfer mode/direction");
        return none();
    };

    h["protocol.Layout.__post_init__"] = [](D a) {
        auto s = self(a);
        identifier(get(s, "model"), "model");
        field(s, "layers", 1, 4096);
        field(s, "kv_heads", 1, 65536);
        field(s, "head_dim", 1, 65536);
        field(s, "tp_size", 1, 64);
        auto page = field(s, "page_size", 0, 1048576);
        Element(s);
        auto storage = str(get(s, "storage"));
        protocol(storage == "continuous" || (storage == "paged" && page > 0), "invalid KV storage");
        return none();
    };

    h["protocol.Layout.layout_id"] = [](D a) {
        auto text = module("json").attr("dumps")(asdict(self(a)), py::arg("sort_keys") = true,
                                                 py::arg("separators") = py::make_tuple(",", ":"));
        return module("hashlib").attr("sha256")(text.attr("encode")()).attr("hexdigest")();
    };

    h["protocol.Layout.compatible"] = [](D a) { return py::bool_(Compatible(self(a), at(a, "other"))); };
    h["protocol.Region.validate"] = [](D a) {
        ValidateRegion(self(a), at(a, "layout"), num(at(a, "begin")), num(at(a, "end")));
        return none();
    };

    h["protocol.Region.byte_offset"] = [](D a) {
        return py::int_(Offset(self(a), integer(at(a, "token"), "token"), integer(at(a, "head"), "head")));
    };

    h["protocol.Manifest.__post_init__"] = [](D a) {
        O s = self(a), id = get(s, "identity"), layout = get(s, "layout");
        field(s, "version", 1, 1);
        protocol(py::isinstance(id, cls("protocol", "Identity")) &&
                     py::isinstance(layout, cls("protocol", "Layout")),
                 "invalid identity/layout");
        protocol(equal(get(s, "owner"), get(id, "source")) || equal(get(s, "owner"), get(id, "target")),
                 "invalid manifest owner");
        I begin = field(s, "token_begin", 0, std::numeric_limits<I>::max() - 1),
          end = field(s, "token_end", begin + 1);
        auto digest = get(s, "prefix_digest");
        protocol(py::isinstance<py::str>(digest), "invalid prefix digest");
        auto text = str(digest);
        protocol(text.size() == 64 && text.find_first_not_of("0123456789abcdef") == std::string::npos,
                 "invalid prefix digest");
        auto regions = get(s, "regions");
        protocol(py::isinstance<T>(regions) && py::len(regions) > 0 && py::len(regions) <= 131072,
                 "invalid region count");
        for (auto r : regions) {
            protocol(py::isinstance(r, cls("protocol", "Region")), "invalid region type");
            ValidateRegion(py::reinterpret_borrow<O>(r), layout, begin, end);
        }
        return none();
    };

    h["protocol.Manifest.to_json"] = [](D a) {
        auto text = module("json").attr("dumps")(asdict(self(a)), py::arg("ensure_ascii") = true,
                                                 py::arg("allow_nan") = false,
                                                 py::arg("separators") = py::make_tuple(",", ":"));
        protocol(py::len(text.attr("encode")()) <= 16 * 1024 * 1024, "manifest too large");
        return text;
    };

    h["protocol.Manifest.from_json"] = [](D a) -> O {
        auto payload = at(a, "payload");
        protocol((py::isinstance<py::str>(payload) || py::isinstance<py::bytes>(payload)) &&
                     py::len(payload) <= 16 * 1024 * 1024,
                 "invalid metadata size");
        auto unique = py::cpp_function([](L pairs) {
            D result;
            for (auto pair : pairs) {
                auto p = py::reinterpret_borrow<O>(pair);
                auto key = p[py::int_(0)];
                protocol(!result.contains(key), "duplicate metadata key");
                result[key] = p[py::int_(1)];
            }
            return result;
        });
        try {
            O parsed = module("json").attr("loads")(payload, py::arg("object_pairs_hook") = unique);
            protocol(py::isinstance<D>(parsed), "invalid metadata object");
            D value = parsed;
            auto object = cls("protocol", "_object");
            D identity = at(value, "identity");
            for (const char *key : {"source", "target"})
                identity[key] = object(cls("protocol", "Peer"), at(identity, key));
            value["identity"] = object(cls("protocol", "Identity"), identity);
            value["owner"] = object(cls("protocol", "Peer"), at(value, "owner"));
            value["layout"] = object(cls("protocol", "Layout"), at(value, "layout"));
            L regions;
            for (auto r : at(value, "regions"))
                regions.append(object(cls("protocol", "Region"), r));
            value["regions"] = tuple(regions);
            return object(at(a, "cls"), value);
        } catch (const py::error_already_set &error) {
            if (error.matches(PyExc_KeyError) || error.matches(PyExc_TypeError) ||
                error.matches(PyExc_RecursionError) || error.matches(module("json").attr("JSONDecodeError")))
                protocol(false, std::string("invalid KV manifest: ") + error.what());
            throw;
        }
    };

    h["protocol.prefix_manifest"] = [](D a) {
        O m = at(a, "manifest"), tokens = at(a, "tokens");
        I end = py::len(tokens);
        protocol(field(m, "token_begin") == 0 && end > 0 && end <= field(m, "token_end"),
                 "prefix outside exported range");
        L regions;
        for (auto entry : get(m, "regions")) {
            O r = py::reinterpret_borrow<O>(entry);
            if (field(r, "token_begin") < end) {
                D change;
                change["token_end"] = std::min(field(r, "token_end"), end);
                regions.append(replace(r, change));
            }
        }
        return cls("protocol", "Manifest")(1, at(a, "identity"), at(a, "owner"), get(m, "layout"), 0, end,
                                           Digest(tokens), tuple(regions));
    };

    h["protocol.suffix_manifest"] = [](D a) {
        auto m = at(a, "manifest");
        I begin = integer(at(a, "begin"), "suffix begin", field(m, "token_begin"), field(m, "token_end") - 1);
        L regions;
        for (auto entry : get(m, "regions")) {
            O r = py::reinterpret_borrow<O>(entry);
            if (field(r, "token_end") > begin) {
                I tb = std::max(begin, field(r, "token_begin"));
                D change;
                change["token_begin"] = tb;
                change["offset"] = Offset(r, tb, field(r, "head_begin"));
                regions.append(replace(r, change));
            }
        }
        D change;
        change["token_begin"] = begin;
        change["regions"] = tuple(regions);
        return replace(m, change);
    };

    h["mapping.Copy.bytes"] = [](D a) -> O {
        O s = self(a);
        I tokens = field(s, "token_end") - field(s, "token_begin");
        I heads = field(s, "head_end") - field(s, "head_begin");
        I width = field(s, "vector_bytes", 1);
        protocol(tokens > 0 && heads > 0 && heads <= std::numeric_limits<I>::max() / width &&
                     tokens <= std::numeric_limits<I>::max() / (heads * width),
                 "invalid copy byte count");
        return py::int_(tokens * heads * width);
    };

    h["mapping.plan_transfer"] = [](D a) { return plan(at(a, "source"), at(a, "target")); };
}
} // namespace pd
} // namespace fastllm
