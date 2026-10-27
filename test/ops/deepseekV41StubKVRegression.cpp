#include "fastllm.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
namespace {
#include "../../src/models/deepseekv41-stub-kv.inc"

void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void Verify(const Data &data, int length) {
    const int padded = (std::max(length, 1) / 128 + 1) * 128;
    Check(data.dims == std::vector<int>({1, length, 1}), "logical shape");
    Check(data.expansionDims == std::vector<int>({1, padded, 1}), "scheduler capacity");
    Check(data.strides == std::vector<uint64_t>({(uint64_t)padded, 1, 1}), "strides");
    Check(data.expansionSize == (uint64_t)padded &&
          data.expansionBytes == (uint64_t)padded * sizeof(float), "allocated capacity");
    Check(data.isKVCache && !data.isFake && data.dataDevice == DataDevice::CPU &&
          data.dataType == DataType::FLOAT32 && data.cpuData, "owned CPU placeholder");
    const auto *values = reinterpret_cast<const float *>(data.cpuData);
    for (int i = 0; i < length; ++i) Check(values[i] == 0.f, "nonzero placeholder payload");
}
} // namespace

int main() {
    try {
        int cases = 0;
        Data key, value;
        for (int n : {0, 0, 1, 1, 2, 7, 127, 128, 129, 255, 256, 4096, 4097, 262144, 262149,
                      262149, 262271, 262272, 1048576, 1048583, 1048583, 31, 31, 0, 1}) {
            auto *previous = key.cpuData;
            const int previousLen = key.dims.empty() ? -1 : key.dims[1];
            const int previousCap = key.expansionDims.empty() ? 0 : key.expansionDims[1];
            V41UpdateStubKV(key, value, n);
            Verify(key, n); Verify(value, n);
            if (n >= previousLen && n < previousCap && previousLen >= 0)
                Check(key.cpuData == previous, "unexpected allocation within capacity");
            ++cases;
        }
        // Foreign payloads/precisions must not be mistaken for an existing stub.
        for (auto type : {FLOAT32, FLOAT16, BFLOAT16}) {
            Data foreign(type, {1, 17, 1}, std::vector<float>(17, 3.f));
            key.CopyFrom(foreign); value.CopyFrom(foreign);
            key.SetKVCache(); value.SetKVCache();
            V41UpdateStubKV(key, value, 17);
            Verify(key, 17); Verify(value, 17);
            ++cases;
        }
        Data owner(FLOAT32, {1, 256, 1}, std::vector<float>(256, 9.f));
        key.FreeSpace(); key = Data(); key.FakeFrom(owner, 0);
        V41UpdateStubKV(key, value, 32);
        Verify(key, 32); Verify(value, 32);
        Check(reinterpret_cast<float *>(owner.cpuData)[0] == 9.f, "borrowed owner modified");
        ++cases;
        for (int request = 0; request < 8; ++request) {
            Data k, v;
            for (int n = 4096; n < 4355; ++n) {
                V41UpdateStubKV(k, v, n); V41UpdateStubKV(k, v, n);
                Verify(k, n); Verify(v, n);
                ++cases;
            }
        }
        std::cout << "PASS " << cases << " scheduler stub states\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
