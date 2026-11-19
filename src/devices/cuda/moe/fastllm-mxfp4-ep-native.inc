// Native loader for build-time exported FlashInfer SM120 CuTeDSL MegaMoE.
// Kernel binaries retain upstream provenance; see third_party/flashinfer_sm120_ep.
// This file does not link to Torch, Python, Triton, FlashInfer or CuTe runtime.
#include "json11.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using J = json11::Json;
static void ck(CUresult r, const char *operation = nullptr) {
    if (r != CUDA_SUCCESS) {
        const char *s = nullptr;
        cuGetErrorString(r, &s);
        throw std::runtime_error((operation ? std::string(operation) + ": " : std::string()) +
                                 (s ? s : "CUDA driver failure"));
    }
}
static void rt(cudaError_t r) {
    if (r != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(r));
}
static std::string read(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot read " + path);
    return {std::istreambuf_iterator<char>(f), {}};
}
static size_t number(const J &v) {
    double x = v.number_value();
    if (!v.is_number() || !std::isfinite(x) || x < 0 || x >= double(SIZE_MAX) || std::floor(x) != x)
        throw std::runtime_error("Invalid nonnegative integer");
    return (size_t)x;
}
static std::vector<unsigned char> unhex(const std::string &s) {
    if (s.size() % 2)
        throw std::runtime_error("Invalid hex argument");
    auto nib = [](char c) {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        throw std::runtime_error("Invalid hex digit");
    };
    std::vector<unsigned char> v(s.size() / 2);
    for (size_t i = 0; i < v.size(); i++)
        v[i] = (nib(s[2 * i]) << 4) | nib(s[2 * i + 1]);
    return v;
}

struct DeviceRestore {
    int device = 0;
    DeviceRestore() { rt(cudaGetDevice(&device)); }
    ~DeviceRestore() { cudaSetDevice(device); }
};

// Buckets of one EP state execute serially (layers already share their bucket
// workspace). Keep each bucket's VA and persistent communication pages private,
// but map transient pages to the same physical storage. Physical growth only
// appends handles: existing graph/TMA addresses and mappings never change.
struct ScratchPool {
    CUmemAllocationProp prop{};
    size_t granularity = 0;
    std::vector<CUmemGenericAllocationHandle> pages;
    explicit ScratchPool(int device) {
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        ck(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
           "EP workspace granularity");
    }
    ~ScratchPool() {
        for (auto page : pages) cuMemRelease(page);
    }
    size_t aligned(size_t n) const {
        if (n > SIZE_MAX - (granularity - 1))
            throw std::runtime_error("EP workspace size overflow");
        return (n + granularity - 1) / granularity * granularity;
    }
    void reserve(size_t n) {
        // CUDA 12.x requires cuMemMap's handle offset to be zero. Independent
        // granules can be reused at arbitrary bucket offsets without remapping.
        const size_t count = aligned(n) / granularity;
        pages.reserve(count);
        while (pages.size() < count) {
            CUmemGenericAllocationHandle page = 0;
            ck(cuMemCreate(&page, granularity, &prop, 0), "EP shared physical page");
            pages.push_back(page);
        }
    }
};

struct Allocation {
    void *ptr = nullptr;
    size_t size = 0;
    std::shared_ptr<ScratchPool> pool;
    CUdeviceptr reservation = 0;
    size_t reservedBytes = 0;
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
    std::vector<CUmemGenericAllocationHandle> privateHandles;
    Allocation(size_t n, bool ones = false) : size(n) {
        if (!n || (ones && n % sizeof(float)))
            throw std::runtime_error("Invalid EP allocation size");
        rt(cudaMalloc(&ptr, n));
        try {
            if (ones) {
                std::vector<float> x(n / 4, 1.0f);
                rt(cudaMemcpy(ptr, x.data(), n, cudaMemcpyHostToDevice));
            } else
                rt(cudaMemset(ptr, 0, n));
        } catch (...) {
            cudaFree(ptr);
            ptr = nullptr;
            throw;
        }
    }
    Allocation(size_t n, const J &privateRanges, std::shared_ptr<ScratchPool> shared)
        : size(n), pool(std::move(shared)) {
        if (!n || !privateRanges.is_array())
            throw std::runtime_error("Invalid EP workspace sharing metadata");
        reservedBytes = pool->aligned(n);
        const size_t gran = pool->granularity;
        std::vector<bool> privatePages(reservedBytes / gran, false);
        for (const auto &range : privateRanges.array_items()) {
            size_t first = number(range["offset"]), count = number(range["bytes"]);
            if (!count || first >= n || count > n - first)
                throw std::runtime_error("EP persistent region exceeds workspace");
            for (size_t page = first / gran; page < pool->aligned(first + count) / gran; page++)
                privatePages[page] = true;
        }
        size_t privateCount = 0;
        for (bool p : privatePages) privateCount += p;
        pool->reserve(reservedBytes - privateCount * gran);
        // Reserve bookkeeping before acquiring mappings/handles so exception
        // cleanup cannot lose ownership after a successful CUDA call.
        mappings.reserve(privatePages.size());
        privateHandles.reserve(privateCount);
        try {
            ck(cuMemAddressReserve(&reservation, reservedBytes, gran, 0, 0), "EP workspace VA reservation");
            size_t sharedPage = 0;
            for (size_t page = 0; page < privatePages.size();) {
                size_t end = page + 1;
                while (end < privatePages.size() && privatePages[end] == privatePages[page]) end++;
                size_t count = (end - page) * gran;
                CUdeviceptr address = reservation + page * gran;
                if (privatePages[page]) {
                    CUmemGenericAllocationHandle h = 0;
                    ck(cuMemCreate(&h, count, &pool->prop, 0), "EP private communication pages");
                    privateHandles.push_back(h);
                    ck(cuMemMap(address, count, 0, h, 0), "EP private workspace mapping");
                    mappings.emplace_back(address, count);
                } else {
                    for (size_t i = page; i < end; ++i) {
                        ck(cuMemMap(address, gran, 0, pool->pages.at(sharedPage++), 0),
                           "EP shared workspace mapping");
                        mappings.emplace_back(address, gran);
                        address += gran;
                    }
                }
                page = end;
            }
            CUmemAccessDesc access{};
            access.location = pool->prop.location;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            ck(cuMemSetAccess(reservation, reservedBytes, &access, 1), "EP workspace access");
            ptr = (void *)reservation;
            // Bucket creation runs on the coordinator before peer work starts.
            rt(cudaMemset(ptr, 0, n));
        } catch (...) { releaseVirtual(); throw; }
    }
    void releaseVirtual() noexcept {
        for (const auto &m : mappings) cuMemUnmap(m.first, m.second);
        mappings.clear();
        if (reservation) cuMemAddressFree(reservation, reservedBytes);
        reservation = 0; ptr = nullptr;
        for (auto h : privateHandles) cuMemRelease(h);
        privateHandles.clear();
    }
    ~Allocation() {
        if (pool) releaseVirtual();
        else if (ptr) cudaFree(ptr);
    }
};
using Buffers = std::map<std::string, std::shared_ptr<Allocation>>;
// A barrier-only manifest predates scale isolation and needs dedicated backing.
static bool canShareLocalWorkspace(const J &buffer) {
    bool barrier = false, scales = false;
    for (const auto &range : buffer["private_ranges"].array_items()) {
        const auto &name = range["name"].string_value();
        barrier |= name == "nvlink_barrier_counter";
        scales |= name == "fc1_output_sf";
    }
    return barrier && scales;
}
__global__ void transformWeight(const unsigned char *input, unsigned char *weight, unsigned char *scale,
                                int n, int k, bool gate) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t bytes = (size_t)n * k / 2, scales = (size_t)n * (k / 32);
    if (i < bytes) {
        int r = i / (k / 2), c = i % (k / 2);
        int target = gate ? ((r % (n / 2)) / 8 * 2 + r / (n / 2)) * 8 + r % 8 : r;
        weight[(size_t)target * (k / 2) + c] = input[i];
    }
    if (i < scales) {
        int cols = k / 32, r = i / cols, c = i % cols;
        if (gate)
            r = ((r % (n / 2)) / 8 * 2 + r / (n / 2)) * 8 + r % 8;
        int paddedCols = (cols + 3) / 4 * 4;
        size_t dst =
            (((size_t)(r / 128) * (paddedCols / 4) + c / 4) * 128 + (r % 32) * 4 + (r % 128) / 32) * 4 +
            c % 4;
        scale[dst] = input[bytes + i];
    }
}

struct NativeEp {
    struct Rank {
        int device = 0;
        CUcontext primary = nullptr;
        std::map<std::string, CUmodule> modules;
        std::map<int, Buffers> weights;
        std::shared_ptr<ScratchPool> scratch;
        bool vmmSupported = false;
        ~Rank() {
            cudaSetDevice(device);
            for (auto &x : modules)
                cuModuleUnload(x.second);
        }
    };
    struct BucketRank {
        int device = 0;
        Buffers buffers;
        CUgreenCtx green[2]{};
        CUcontext contexts[2]{};
        ~BucketRank() {
            cudaSetDevice(device);
            for (auto g : green)
                if (g)
                    cuGreenCtxDestroy(g);
        }
    };
    struct Graph {
        int device = 0;
        CUgraph graph = nullptr;
        CUgraphExec exec = nullptr;
        ~Graph() {
            cudaSetDevice(device);
            if (exec)
                cuGraphExecDestroy(exec);
            if (graph)
                cuGraphDestroy(graph);
        }
    };
    std::string directory;
    J manifest;
    std::vector<std::unique_ptr<Rank>> ranks;
    std::map<int, J> specs;
    std::map<int, std::vector<std::unique_ptr<BucketRank>>> buckets;
    std::map<std::pair<int, int>, std::vector<std::unique_ptr<Graph>>> graphs;
    NativeEp(const std::string &path, const std::vector<int> &devices) : directory(path) {
        std::string error;
        manifest = J::parse(read(path + "/manifest.json"), error);
        if (!error.empty() || manifest["version"].int_value() != 1 ||
            manifest["ep"].int_value() != (int)devices.size())
            throw std::runtime_error("Incompatible native EP manifest: " + error);
        for (int dev : devices) {
            rt(cudaSetDevice(dev));
            cudaDeviceProp prop{};
            rt(cudaGetDeviceProperties(&prop, dev));
            if (prop.major * 10 + prop.minor != manifest["sm"].int_value() ||
                prop.multiProcessorCount != manifest["sm_count"].int_value())
                throw std::runtime_error("Native EP GPU specialization differs");
            auto r = std::unique_ptr<Rank>(new Rank);
            r->device = dev;
            rt(cudaFree(nullptr));
            ck(cuCtxGetCurrent(&r->primary));
            int vmm = 0;
            ck(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev));
            r->vmmSupported = vmm != 0;
            for (int peer : devices)
                if (peer != dev) {
                    int access = 0;
                    rt(cudaDeviceCanAccessPeer(&access, dev, peer));
                    if (!access)
                        throw std::runtime_error("EP requires peer access");
                    auto e = cudaDeviceEnablePeerAccess(peer, 0);
                    if (e != cudaErrorPeerAccessAlreadyEnabled)
                        rt(e);
                    cudaGetLastError();
                }
            ranks.push_back(std::move(r));
        }
        for (const auto &s : manifest["buckets"].array_items()) {
            int c = s["capacity"].int_value();
            if (c <= 0 || s["ranks"].array_items().size() != ranks.size())
                throw std::runtime_error("Invalid EP bucket");
            specs.emplace(c, s);
        }
        int h = manifest["hidden"].int_value(), mid = manifest["intermediate"].int_value(),
            experts = manifest["experts"].int_value(), topk = manifest["topk"].int_value();
        if (h <= 0 || mid <= 0 || h % 128 || mid % 64 || experts <= 0 || experts % devices.size() ||
            topk <= 0 || topk > experts)
            throw std::runtime_error("Invalid native EP model dimensions");
        if (specs.empty())
            throw std::runtime_error("Empty native EP package");
    }
    ~NativeEp() {
        int saved = 0;
        cudaGetDevice(&saved);
        for (auto &r : ranks) {
            cudaSetDevice(r->device);
            cudaDeviceSynchronize();
        }
        graphs.clear();
        buckets.clear();
        ranks.clear();
        cudaSetDevice(saved);
    }
    int capacity(int rows) const {
        auto i = specs.lower_bound(rows);
        if (i == specs.end())
            return specs.rbegin()->first;
        return i->first;
    }
    void bucket(int cap) {
        if (buckets.count(cap))
            return;
        std::vector<std::unique_ptr<BucketRank>> result;
        const auto &spec = specs.at(cap);
        for (int rank = 0; rank < (int)ranks.size(); rank++) {
            rt(cudaSetDevice(ranks[rank]->device));
            auto b = std::unique_ptr<BucketRank>(new BucketRank);
            b->device = ranks[rank]->device;
            const auto &entry = spec["ranks"][rank];
            for (const auto &v : entry["buffers"].object_items()) {
                if (v.first == "w13" || v.first == "w2" || v.first == "s13" || v.first == "s2")
                    continue;
                if (v.first == "local" && ranks[rank]->vmmSupported && canShareLocalWorkspace(v.second)) {
                    auto &pool = ranks[rank]->scratch;
                    if (!pool) pool = std::make_shared<ScratchPool>(ranks[rank]->device);
                    b->buffers[v.first] = std::make_shared<Allocation>(
                        number(v.second["bytes"]), v.second["private_ranges"], pool);
                } else {
                    // Older packages and devices without VMM retain dedicated
                    // allocations, with the same fixed-address graph contract.
                    b->buffers[v.first] = std::make_shared<Allocation>(
                        number(v.second["bytes"]), v.second["init"].string_value() == "float_one");
                }
            }
            CUdevResource sm{}, group{}, remainder{};
            unsigned count = 1;
            ck(cuDeviceGetDevResource(b->device, &sm, CU_DEV_RESOURCE_TYPE_SM));
            ck(cuDevSmResourceSplitByCount(&group, &count, &sm, &remainder, 0,
                                           entry["green_sms"][0].int_value()));
            if (count != 1 || group.sm.smCount != (unsigned)entry["green_sms"][0].int_value() ||
                remainder.sm.smCount != (unsigned)entry["green_sms"][1].int_value())
                throw std::runtime_error("Green Context partition differs");
            CUdevResource resources[] = {group, remainder};
            for (int i = 0; i < 2; i++) {
                CUdevResourceDesc desc;
                ck(cuDevResourceGenerateDesc(&desc, &resources[i], 1));
                ck(cuGreenCtxCreate(&b->green[i], desc, b->device, CU_GREEN_CTX_DEFAULT_STREAM));
                ck(cuCtxFromGreenCtx(&b->contexts[i], b->green[i]));
            }
            result.push_back(std::move(b));
        }
        buckets.emplace(cap, std::move(result));
    }
    void bind(int rank, int layer, const uint64_t *sources) {
        auto &r = *ranks.at(rank);
        if (r.weights.count(layer))
            return;
        rt(cudaSetDevice(r.device));
        const auto &sizes = specs.begin()->second["ranks"][rank]["buffers"];
        Buffers w;
        for (auto n : {"w13", "s13", "w2", "s2"})
            w[n] = std::make_shared<Allocation>(number(sizes[n]["bytes"]));
        int h = manifest["hidden"].int_value(), mid = manifest["intermediate"].int_value(),
            experts = manifest["experts"].int_value() / ranks.size();
        for (int e = 0; e < experts; e++)
            for (int part = 0; part < 2; part++) {
                int n = part ? h : 2 * mid, k = part ? mid : h;
                const char *wn = part ? "w2" : "w13", *sn = part ? "s2" : "s13";
                size_t bytes = (size_t)n * k / 2,
                       scales = (size_t)((n + 127) / 128 * 128) * ((k / 32 + 3) / 4 * 4);
                transformWeight<<<(bytes + 255) / 256, 256>>>(
                    (const unsigned char *)sources[e * 2 + part], (unsigned char *)w[wn]->ptr + e * bytes,
                    (unsigned char *)w[sn]->ptr + e * scales, n, k, !part);
                rt(cudaGetLastError());
            }
        rt(cudaDeviceSynchronize());
        r.weights.emplace(layer, std::move(w));
    }
    uintptr_t address(const J &ref, int cap, int layer, size_t bytes = 1) {
        int rank = ref["rank"].int_value();
        const auto &name = ref["buffer"].string_value();
        auto &b = buckets.at(cap).at(rank)->buffers;
        auto p = b.find(name);
        auto a = p != b.end() ? p->second : ranks.at(rank)->weights.at(layer).at(name);
        size_t offset = number(ref["offset"]);
        if (offset >= a->size || bytes > a->size - offset)
            throw std::runtime_error("EP relocation exceeds buffer");
        return (uintptr_t)a->ptr + offset;
    }
    CUfunction function(int rank, const J &node) {
        auto &r = *ranks.at(rank);
        const auto &key = node["binary"].string_value();
        if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::runtime_error("Invalid cubin identifier");
        auto it = r.modules.find(key);
        if (it == r.modules.end()) {
            auto bytes = read(directory + "/" + key + ".cubin");
            if (bytes.size() != number(manifest["binaries"][key]) || bytes.compare(0, 4,
                                                                                   "\x7f"
                                                                                   "ELF"))
                throw std::runtime_error("Invalid cubin artifact");
            CUmodule m;
            ck(cuModuleLoadData(&m, bytes.data()));
            it = r.modules.emplace(key, m).first;
        }
        CUfunction fn;
        ck(cuModuleGetFunction(&fn, it->second, node["name"].string_value().c_str()));
        if (node["shared"].int_value() > 0)
            ck(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  node["shared"].int_value()));
        return fn;
    }
    CUgraph graph(const J &recipe, int rank, int cap, int layer) {
        CUgraph g;
        ck(cuGraphCreate(&g, 0));
        try {
            const auto &ns = recipe.array_items();
            std::vector<CUgraphNode> nodes(ns.size());
            std::vector<bool> pending(ns.size(), true);
            size_t remaining = ns.size();
            while (remaining) {
                bool advanced = false;
                for (size_t i = 0; i < ns.size(); i++)
                    if (pending[i]) {
                        const auto &n = ns[i];
                        bool ready = true;
                        for (auto &v : n["dependencies"].array_items()) {
                            size_t d = number(v["node"]);
                            if (d >= nodes.size())
                                throw std::runtime_error("Invalid graph edge");
                            if (pending[d]) {
                                ready = false;
                                break;
                            }
                        }
                        if (!ready)
                            continue;
                        // Preserve CUDA 12.3+ edge ports in a separate pass.
                        const auto &type = n["type"].string_value();
                        if (type == "graph") {
                            auto child = graph(n["nodes"], rank, cap, layer);
                            auto e = cuGraphAddChildGraphNode(&nodes[i], g, nullptr, 0, child);
                            cuGraphDestroy(child);
                            ck(e);
                        } else if (type == "empty")
                            ck(cuGraphAddEmptyNode(&nodes[i], g, nullptr, 0));
                        else if (type == "memset") {
                            CUDA_MEMSET_NODE_PARAMS p{};
                            p.dst = address(n["destination"], cap, layer,
                                            number(n["element_size"]) * number(n["width"]));
                            p.value = n["value"].number_value();
                            p.elementSize = n["element_size"].int_value();
                            p.width = number(n["width"]);
                            p.height = 1;
                            ck(cuGraphAddMemsetNode(&nodes[i], g, nullptr, 0, &p,
                                                    ranks[rank]->primary));
                        } else if (type == "kernel") {
                            std::vector<std::vector<unsigned char>> bytes;
                            std::vector<void *> args;
                            CUfunction fn = function(rank, n);
                            size_t argIndex = 0;
                            for (const auto &a : n["args"].array_items()) {
                                auto data = unhex(a["hex"].string_value());
                                size_t expectedOffset = 0, expectedSize = 0;
                                ck(cuFuncGetParamInfo(fn, argIndex++, &expectedOffset, &expectedSize));
                                if (expectedSize != data.size())
                                    throw std::runtime_error("Kernel argument ABI differs");
                                for (const auto &rel : a["relocations"].array_items()) {
                                    size_t off = number(rel["arg_offset"]);
                                    const auto &kind = rel["kind"].string_value();
                                    if (kind == "tma") {
                                        if (data.size() != sizeof(CUtensorMap) || off)
                                            throw std::runtime_error("Invalid TMA argument");
                                        alignas(64) CUtensorMap map;
                                        memcpy(&map, data.data(), sizeof(map));
                                        ck(cuTensorMapReplaceAddress(&map, (void *)address(rel, cap, layer)));
                                        memcpy(data.data(), &map, sizeof(map));
                                    } else {
                                        uint64_t value = 0;
                                        if (kind == "pointer")
                                            value = address(rel, cap, layer);
                                        else if (kind == "peer_offset")
                                            value = (uintptr_t)buckets.at(cap)
                                                        .at(rel["rank"].int_value())
                                                        ->buffers.at("heap")
                                                        ->ptr -
                                                    (uintptr_t)buckets.at(cap)
                                                        .at(rel["local_rank"].int_value())
                                                        ->buffers.at("heap")
                                                        ->ptr;
                                        else
                                            throw std::runtime_error("Unknown relocation");
                                        if (off + 8 > data.size())
                                            throw std::runtime_error("Argument relocation out of range");
                                        memcpy(data.data() + off, &value, 8);
                                    }
                                }
                                bytes.push_back(std::move(data));
                            }
                            for (auto &a : bytes)
                                args.push_back(a.data());
                            CUgraphNodeParams p{};
                            p.type = CU_GRAPH_NODE_TYPE_KERNEL;
                            p.kernel.func = fn;
                            p.kernel.gridDimX = n["grid"][0].int_value();
                            p.kernel.gridDimY = n["grid"][1].int_value();
                            p.kernel.gridDimZ = n["grid"][2].int_value();
                            p.kernel.blockDimX = n["block"][0].int_value();
                            p.kernel.blockDimY = n["block"][1].int_value();
                            p.kernel.blockDimZ = n["block"][2].int_value();
                            p.kernel.sharedMemBytes = n["shared"].int_value();
                            p.kernel.kernelParams = args.data();
                            int green = n["green"].int_value();
                            if (green < -1 || green > 1)
                                throw std::runtime_error("Invalid Green Context index");
                            p.kernel.ctx =
                                green < 0 ? ranks[rank]->primary : buckets.at(cap).at(rank)->contexts[green];
#if CUDA_VERSION >= 13000
                            ck(cuGraphAddNode(&nodes[i], g, nullptr, nullptr, 0, &p));
#else
                            ck(cuGraphAddNode(&nodes[i], g, nullptr, 0, &p));
#endif
                        } else
                            throw std::runtime_error("Unknown graph node");
                        pending[i] = false;
                        --remaining;
                        advanced = true;
                    }
                if (!advanced)
                    throw std::runtime_error("Cyclic graph recipe");
            }
            for (size_t i = 0; i < ns.size(); i++)
                for (const auto &dep : ns[i]["dependencies"].array_items()) {
                    CUgraphNode from = nodes.at(number(dep["node"]));
                    CUgraphEdgeData edge{};
                    if (dep["edge"].array_items().size() != sizeof(edge))
                        throw std::runtime_error("Invalid graph edge ABI");
                    for (size_t b = 0; b < sizeof(edge); b++)
                        ((unsigned char *)&edge)[b] = dep["edge"][b].int_value();
                    ck(cuGraphAddDependencies_v2(g, &from, &nodes[i], &edge, 1));
                }
            return g;
        } catch (...) {
            cuGraphDestroy(g);
            throw;
        }
    }
    void prepare(int cap, int layer) {
        if (graphs.count({cap, layer}))
            return;
        bucket(cap);
        std::vector<std::unique_ptr<Graph>> gs;
        for (int r = 0; r < (int)ranks.size(); r++) {
            rt(cudaSetDevice(ranks[r]->device));
            auto g = std::unique_ptr<Graph>(new Graph);
            g->device = ranks[r]->device;
            g->graph = graph(specs.at(cap)["ranks"][r]["nodes"], r, cap, layer);
            ck(cuGraphInstantiate(&g->exec, g->graph, 0));
            gs.push_back(std::move(g));
        }
        graphs.emplace(std::make_pair(cap, layer), std::move(gs));
    }
};

std::string NativeEpPackage(const std::vector<int> &devices) {
    if (devices.empty())
        return {};
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, devices[0]) != cudaSuccess)
        return {};
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&FastllmCudaMxfp4EpGraphSupported), &info) || !info.dli_fname)
        return {};
    std::string path(info.dli_fname);
    auto slash = path.find_last_of('/');
    if (slash == std::string::npos)
        path = ".";
    else
        path.resize(slash);
    return path + "/kernels/mxfp4_ep/sm" + std::to_string(prop.major * 10 + prop.minor) + "_sms" +
           std::to_string(prop.multiProcessorCount) + "_ep" + std::to_string(devices.size()) +
           "_h4096_i2048_e256_k8";
}
} // namespace

struct FastllmMxfp4EpNativeState {
    std::unique_ptr<NativeEp> runtime;
    std::vector<std::vector<std::vector<uint64_t>>> sources;
};

std::shared_ptr<FastllmMxfp4EpNativeState>
FastllmCudaCreateMxfp4EpNative(const std::vector<int> &devices,
                               const std::vector<std::vector<std::vector<uint64_t>>> &weights) {
    DeviceRestore restore;
    auto state = std::make_shared<FastllmMxfp4EpNativeState>();
    state->runtime.reset(new NativeEp(NativeEpPackage(devices), devices));
    state->sources = weights;
    return state;
}

void FastllmCudaBindMxfp4EpNativeWeights(FastllmMxfp4EpNativeState &state, int layer) {
    DeviceRestore restore;
    auto &runtime = *state.runtime;
    for (size_t rank = 0; rank < runtime.ranks.size(); ++rank) {
        if (runtime.ranks[rank]->weights.count(layer)) continue;
        auto &weights = state.sources.at(rank).at(layer);
        if (weights.size() != 2 * runtime.manifest["experts"].int_value() / runtime.ranks.size())
            throw std::runtime_error("Invalid native MXFP4 expert pointer table");
        runtime.bind(rank, layer, weights.data());
        // bind synchronizes the transform before publishing native weights.
        // Do not retain addresses that the caller is now allowed to release.
        weights.clear();
    }
}

std::vector<FastllmMxfp4EpGraph> FastllmCudaPrepareMxfp4EpNative(FastllmMxfp4EpNativeState &state, int layer,
                                                                 int rows) {
    DeviceRestore restore;
    auto &runtime = *state.runtime;
    int capacity = runtime.capacity(rows);
    FastllmCudaBindMxfp4EpNativeWeights(state, layer);
    runtime.prepare(capacity, layer);
    std::vector<FastllmMxfp4EpGraph> result;
    for (size_t rank = 0; rank < runtime.ranks.size(); ++rank) {
        const auto &native = runtime.graphs.at({capacity, layer})[rank];
        const auto &io = runtime.specs.at(capacity)["ranks"][rank]["io"];
        FastllmMxfp4EpGraph graph;
        graph.capacity = capacity;
        graph.graph = native->graph;
        graph.graphExec = native->exec;
        graph.input = (void *)runtime.address(io["input"], capacity, layer);
        graph.indices = (void *)runtime.address(io["indices"], capacity, layer);
        graph.scores = (void *)runtime.address(io["scores"], capacity, layer);
        graph.output = (void *)runtime.address(io["output"], capacity, layer);
        result.push_back(graph);
    }
    return result;
}
