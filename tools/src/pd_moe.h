#ifndef FASTLLM_TOOLS_PD_MOE_H
#define FASTLLM_TOOLS_PD_MOE_H

/** 编码 CUDA IPC 字节；仅用于已认证的本地服务控制消息。 */
std::string MoePDHex(const uint8_t *bytes, size_t count) {
    const char digits[] = "0123456789abcdef";
    std::string value;
    for (size_t i = 0; i < count; ++i) { value += digits[bytes[i] >> 4]; value += digits[bytes[i] & 15]; }
    return value;
}

/** 读取字符串 key，避免 JSON 浮点数损失 uint64 精度。 */
uint64_t MoePDKey(const Json &value) {
    const auto &s = value.string_value();
    if (s.empty() || s.size() > 16 || s.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error("invalid MoE PD key");
    auto key = std::stoull(s, nullptr, 16);
    if (!key) throw std::runtime_error("zero MoE PD key");
    return key;
}

/**
 * 执行一批 Attention rank 的交接控制与计算；唯一协调器保持 EP 集合通信顺序。
 *
 * 无计算的 rank 用 Hold 参加本轮专家通信；既不结束其请求，也不读取未就绪 KV。
 * 存储命令只作用于目标 rank。Prefill 不采样，D 的 step 使用正式采样函数。
 * @param requests 每个 Attention rank 至多一个命令，null 表示本轮无新动作。
 * @param random 按交接编号隔离的 RNG，本地抢占不重置。
 * @return 逐 rank 结果；执行/协议错误必须使整个 EP 组失败。
 */
Json::array MoePDExecute(ProcessGroup &group, const Json::array &requests, bool graph, int timeout,
                        std::map<std::string, std::mt19937> &random) {
    std::vector<QwenMoeStep> steps(requests.size());
    Json::array output(requests.size());
    bool compute = false, decode = false;
    for (size_t rank = 0; rank < requests.size(); ++rank) {
        auto &step = steps[rank];
        step.action = QwenMoeAction::Hold;
        if (requests[rank].is_null()) continue;
        const auto &request = requests[rank];
        const auto op = request["action"].string_value();
        if (op == "usage") {
            PDCommand initialize;
            initialize.action = PDAction::Initialize;
            initialize.key = 1;
            initialize.targetRank = rank;
            group.PDStorage(initialize);
            auto usage = group.Qwen3KVUsages().at(rank);
            output[rank] = Json::object{{"kv_cache_usage", usage.totalPages ? 1.0-double(usage.freePages)/usage.totalPages : 0.0},
                {"workers", Json::array{Json::object{{"rank", 0}, {"total_pages", int(usage.totalPages)}, {"free_pages", int(usage.freePages)}}}};
        } else if (op == "cache") {
            output[rank] = Json::array{States(group).at(rank)};
        } else if (op == "sampling_end") {
            random.erase(request["context"].string_value());
            output[rank] = Json::object{{"released", true}};
        } else if (op == "prefill" || op == "prepared_prefill" || op == "step" || op == "end" || op == "full_prefill") {
            compute = true;
            if (op == "end") step.action = QwenMoeAction::End;
            else if (op == "step") {
                step.action = QwenMoeAction::Decode;
                step.tokenIds = {Integer(request["input"])};
                decode = true;
            } else {
                graph = false;
                if (!request["token_ids"].is_array()) throw std::runtime_error("missing MoE PD token IDs");
                for (const auto &id : request["token_ids"].array_items()) step.tokenIds.push_back(Integer(id));
                if (op == "prepared_prefill") {
                    step.action = QwenMoeAction::Prepared;
                    step.transferKey = MoePDKey(request["key"]);
                    step.notifyPath = request["notify_path"].string_value();
                } else {
                    step.action = QwenMoeAction::Begin;
                    step.capacity = Integer(request["capacity"]);
                }
            }
        } else {
            PDCommand c;
            c.targetRank = rank;
            c.key = MoePDKey(request["key"]);
            c.tokens = Integer(request["tokens"]);
            c.capacity = Integer(request["capacity"]);
            c.position = Integer(request["position"]);
            if (op == "prepare") c.action = PDAction::Prepare;
            else if (op == "export") c.action = PDAction::Export;
            else if (op == "commit") c.action = PDAction::Commit;
            else if (op == "restore_prefix") c.action = PDAction::RestorePrefix;
            else if (op == "restore_local") {
                c.action = PDAction::RestoreLocal;
                if (!request["token_ids"].is_array()) throw std::runtime_error("local prefix requires token IDs");
                for (auto token : request["token_ids"].array_items()) c.tokenIds.push_back(Integer(token));
            }
            else if (op == "release") c.action = PDAction::Release;
            else throw std::runtime_error("unknown MoE PD operation");
            auto regions = group.PDStorage(c).at(rank);
            if (op == "restore_local") {
                output[rank] = Json::object{{"restored_tokens", int(group.Qwen3Requests().at(rank).tokens)}};
                continue;
            }
            Json::array entries;
            for (const auto &r : regions) entries.push_back(Json::object{{"rank", 0}, {"layer", int(r.layer)},
                {"component", r.component ? "v" : "k"}, {"token_begin", int(r.tokenBegin)}, {"token_end", int(r.tokenEnd)},
                {"head_begin", int(r.headBegin)}, {"head_end", int(r.headEnd)}, {"head_dim", int(r.headDim)},
                {"page_size", int(r.pageSize)}, {"memory_hex", MoePDHex(r.memory.data(), 64)}, {"device", MoePDHex(r.device.data(), 16)},
                {"allocation_bytes", double(r.allocationBytes)}, {"offset", double(r.offset)},
                {"token_stride", double(r.tokenStride)}, {"head_stride", double(r.headStride)}});
            output[rank] = Json::object{{"regions", entries}};
        }
    }
    if (!compute) return output;
    const auto logits = group.StepQwenMoe(steps, graph && decode, timeout);
    for (size_t rank = 0; rank < requests.size(); ++rank) {
        const auto &request = requests[rank];
        const auto op = request["action"].string_value();
        if (op == "step" || op == "full_prefill") {
            const auto context = request["context"].string_value();
            if (context.size() != 32 || context.find_first_not_of("0123456789abcdef") != std::string::npos ||
                !request["first"].is_bool()) throw std::runtime_error("invalid MoE PD sampling identity");
            const auto &params = request["sampling"];
            Qwen3GenerationOptions s;
            s.topK = Integer(params["top_k"]);
            s.temperature = params["temperature"].number_value(); s.topP = params["top_p"].number_value();
            if (!params["temperature"].is_number() || !params["top_p"].is_number() ||
                !std::isfinite(s.temperature) || s.temperature < 0 || !std::isfinite(s.topP) || s.topP <= 0 || s.topP > 1)
                throw std::runtime_error("invalid MoE PD sampling parameters");
            if (request["first"].bool_value()) {
                if (random.count(context)) throw std::runtime_error("MoE sampling context already exists");
                random.emplace(context, std::mt19937(Integer(params["seed"])));
            }
            auto found = random.find(context);
            if (found == random.end()) throw std::runtime_error("MoE sampling context missing");
            // 沿用非 PD Attention DP/EP 的 32-bit RNG 与归一化，保持同 seed 的采样序列。
            const auto uniform = double(found->second()) / 4294967296.0;
            output[rank] = Json::object{{"token", static_cast<long long>(SampleQwen3(logits.at(rank), s, uniform))}};
        } else if (op == "prefill" || op == "prepared_prefill") output[rank] = Json::object{{"prefill_done", true}};
        else if (op == "end") output[rank] = Json::object{{"ended", true}};
    }
    return output;
}
#endif
