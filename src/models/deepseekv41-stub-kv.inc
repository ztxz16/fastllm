// Scheduler metadata only: the model's real KV lives in its request state.
// The marker keeps an unrelated caller-supplied tensor on the rebuild path.
void V41UpdateStubKV(Data &key, Data &value, int totalLen) {
    const int paddedLen = (std::max(totalLen, 1) / 128 + 1) * 128;
    auto reusable = [totalLen](const Data &data) {
        if (data.name != "v41.stub_kv" || !data.isKVCache || data.isFake ||
            data.isPagedKVCache || data.kvMemConfig || data.multiDeviceData ||
            data.dataType != DataType::FLOAT32 || data.dataDevice != DataDevice::CPU ||
            !data.cpuData || data.cudaData || data.mapFile || data.isPinned ||
            data.dims.size() != 3 || data.dims[0] != 1 || data.dims[2] != 1 ||
            data.dims[1] < 0 || totalLen < data.dims[1] || data.expansionDims.size() != 3 ||
            data.strides.size() != 3)
            return false;
        const int capacity = (std::max(data.dims[1], 1) / 128 + 1) * 128;
        return data.expansionDims[0] == 1 && data.expansionDims[1] == capacity &&
            data.expansionDims[2] == 1 && data.strides[0] == (uint64_t)capacity &&
            data.strides[1] == 1 && data.strides[2] == 1 &&
            data.expansionSize == (uint64_t)capacity &&
            data.expansionBytes == (uint64_t)capacity * sizeof(float);
    };
    if (reusable(key) && reusable(value) && key.dims[1] == value.dims[1]) {
        const int oldLen = key.dims[1];
        if (oldLen == totalLen) return;
        for (Data *data : {&key, &value}) {
            if (data->expansionDims[1] < paddedLen)
                data->Expansion({1, paddedLen, 1});
            std::memset(data->cpuData + (size_t)oldLen * sizeof(float), 0,
                        (size_t)(totalLen - oldLen) * sizeof(float));
            data->Resize({1, totalLen, 1});
        }
        return;
    }

    // Preserve the previous reset/foreign-layout behavior, including padding.
    std::vector<float> zeros((uint64_t)totalLen, 0.0f);
    Data stubKey(DataType::FLOAT32, {1, totalLen, 1}, zeros);
    Data stubValue(DataType::FLOAT32, {1, totalLen, 1}, zeros);
    stubKey.SetKVCache();
    stubValue.SetKVCache();
    stubKey.Expansion({1, paddedLen, 1});
    stubValue.Expansion({1, paddedLen, 1});
    key.FreeSpace();
    value.FreeSpace();
    key = Data();
    value = Data();
    key.CopyFrom(stubKey);
    value.CopyFrom(stubValue);
    key.SetKVCache();
    value.SetKVCache();
    key.name = value.name = "v41.stub_kv";
}
