// Included inside namespace fastllm, alongside the existing request-state
// lifecycle and scheduler marker helpers.
std::string DeepSeekV41Model::PDLayout() const {
    return "deepseek-v41-state-v1:" + std::to_string(block_cnt) + ":" + std::to_string(dataType) + ":" +
           std::to_string(kvCacheDataType) + ":" + std::to_string(window_size) + ":" +
           std::to_string(v41DsparkEnabled);
}

fastkv::Message DeepSeekV41Model::ExportPDState(ResponseContext &context) {
    auto state = GetOrCreateState(context.pastKeyValues, false);
    if (state->totalLen != context.preTokens || state->totalLen <= 0)
        throw std::runtime_error("FastKV: incomplete DeepSeek prefill state");
    const auto &owner = state->attentionRanks.empty() ? state : state->attentionRanks.front();
    if (!owner || owner->layers.size() != size_t(block_cnt))
        throw std::runtime_error("FastKV: missing DeepSeek KV owner");
    PDStateWriter writer;
    writer.metadata.String(PDLayout());
    writer.metadata.U32(state->totalLen);
    writer.metadata.Ints(state->engramHistory);
    for (int layer = 0; layer < block_cnt; ++layer) {
        const auto &cache = owner->layers[layer];
        if (cache.totalLen != state->totalLen)
            throw std::runtime_error("FastKV: incomplete DeepSeek layer");
        writer.metadata.U32(cache.compressedBlocks);
        writer.metadata.U32(cache.rawTail);
        writer.Tensor(cache.windowKV);
        if (isKvSource[layer]) {
            writer.Tensor(cache.compressedKV);
            writer.Tensor(cache.indexK);
            if (cache.rawTail) {
                writer.Tensor(cache.rawTailKV);
                writer.Tensor(cache.rawTailScore);
            }
        }
    }
    writer.metadata.U32(bool(state->dspark));
    if (state->dspark) {
        const auto &draft = *state->dspark;
        if (!draft.pending.empty())
            throw std::runtime_error("FastKV: prefill has unconsumed draft outputs");
        writer.metadata.U32(draft.committed);
        writer.metadata.U32(draft.filled);
        writer.metadata.U32(draft.disabled);
        writer.metadata.U32(draft.layers.size());
        for (const auto &layer : draft.layers)
            writer.Tensor(layer.windowKV);
        // A proposal depends on D's sampler. Transfer committed draft KV only;
        // D builds its first proposal while consuming the last prompt token.
    }
    return writer.Finish();
}

void DeepSeekV41Model::ImportPDState(ResponseContext &context, const fastkv::Message &message) {
    PDStateReader reader(message);
    if (reader.metadata.String() != PDLayout())
        throw std::runtime_error("FastKV: DeepSeek state layout mismatch");
    auto state = std::make_shared<DeepSeekV41RequestState>();
    state->totalLen = state->restoredLen = reader.metadata.U32();
    if (state->totalLen != int(context.allTokens.size()) - 1 || state->totalLen <= 0)
        throw std::runtime_error("FastKV: DeepSeek prompt length mismatch");
    state->engramHistory = reader.metadata.Ints(size_t(max_positions));
    if (!engram_layer_ids.empty() && state->engramHistory.size() != size_t(state->totalLen))
        throw std::runtime_error("FastKV: incomplete Engram history");
    state->layers.resize(block_cnt);
    for (int layer = 0; layer < block_cnt; ++layer) {
        auto &cache = state->layers[layer];
        cache.totalLen = state->totalLen;
        cache.compressedBlocks = reader.metadata.U32();
        cache.rawTail = reader.metadata.U32();
        reader.Tensor(cache.windowKV);
        if (isKvSource[layer]) {
            if (cache.compressedBlocks != state->totalLen / compress_ratios[layer] ||
                cache.rawTail != state->totalLen % compress_ratios[layer])
                throw std::runtime_error("FastKV: inconsistent DeepSeek compressed prefix");
            reader.Tensor(cache.compressedKV);
            reader.Tensor(cache.indexK);
            if (cache.rawTail) {
                reader.Tensor(cache.rawTailKV);
                reader.Tensor(cache.rawTailScore);
            }
        }
    }
    if (reader.metadata.U32()) {
        state->dspark = std::make_shared<DeepSeekV41DsparkState>();
        auto &draft = *state->dspark;
        draft.committed = reader.metadata.U32();
        draft.filled = reader.metadata.U32();
        draft.disabled = reader.metadata.U32();
        const uint32_t layers = reader.metadata.U32();
        if (layers > uint32_t(v41DsparkLayers) || draft.committed != state->totalLen || draft.filled < 0 ||
            draft.filled > window_size)
            throw std::runtime_error("FastKV: invalid DeepSeek draft state");
        draft.layers.resize(layers);
        for (auto &layer : draft.layers)
            reader.Tensor(layer.windowKV);
    }
    reader.End();
#ifdef USE_CUDA
    PrepareAttentionState(state);
#endif
    V41UpdateStubKV(context.pastKeyValues[0].first, context.pastKeyValues[0].second, state->totalLen);
    std::lock_guard<std::mutex> lock(v41StateMutex);
    RegisterState(&context.pastKeyValues, &context.pastKeyValues[0].first, state);
}
