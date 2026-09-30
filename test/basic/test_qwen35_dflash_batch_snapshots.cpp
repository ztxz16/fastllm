#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;

namespace {
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

void Fill(Data &result, DataType type, const std::vector<int> &shape,
          const std::vector<float> &values, int device) {
    result.dataType = type;
    result.Resize(shape);
    result.Allocate();
    Require(result.Count(0) == values.size(), "tensor size mismatch");
    for (size_t i = 0; i < values.size(); ++i) {
        if (type == FLOAT32) reinterpret_cast<float *>(result.cpuData)[i] = values[i];
        else reinterpret_cast<uint16_t *>(result.cpuData)[i] = float_to_half(values[i]);
    }
    result.ToDevice(DataDevice::CUDA, std::vector<int>{device});
}

Data Tensor(DataType type, const std::vector<int> &shape,
            const std::vector<float> &values, int device) {
    Data result;
    Fill(result, type, shape, values, device);
    return result;
}

std::vector<float> Read(const Data &data) {
    std::vector<uint16_t> raw(data.Count(0));
    FastllmCudaCopyFromDeviceToHost(raw.data(), data.cudaData, raw.size() * 2);
    std::vector<float> values(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) values[i] = half_to_float(raw[i]);
    return values;
}

void Near(const std::vector<float> &actual, const std::vector<float> &expected) {
    Require(actual.size() == expected.size(), "comparison size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            std::abs(actual[i] - expected[i]) > 0.002f + 0.002f * std::abs(expected[i])) {
            std::cerr << "index=" << i << " actual=" << actual[i]
                      << " expected=" << expected[i] << '\n';
            throw std::runtime_error("snapshot/reference mismatch");
        }
    }
}

void Run(int device, int batch, int length, int kHeads = 2,
         int vHeads = 4, int vDim = 128) {
    FastllmCudaSetDevice(device);
    // Two K heads/four V heads exercise grouped heads and rank-local layouts.
    constexpr int kDim = 128;
    const int channels = 2 * kHeads * kDim + vHeads * vDim;
    const int stateSize = vHeads * kDim * vDim;
    const int outSize = vHeads * vDim;
    const int slots = length - 1;
    const float scale = 1.0f / std::sqrt(float(kDim));
    Data norm = Tensor(FLOAT32, {kDim}, std::vector<float>(kDim, 0.9f), device);
    std::vector<float> aValues(vHeads), dtValues(vHeads);
    const float aPattern[] = {-0.7f, -0.5f, -0.4f, -0.6f};
    const float dtPattern[] = {0.1f, -0.2f, 0.3f, -0.1f};
    for (int h = 0; h < vHeads; ++h) {
        aValues[h] = aPattern[h % 4]; dtValues[h] = dtPattern[h % 4];
    }
    Data aLog = Tensor(FLOAT32, {vHeads}, aValues, device);
    Data dtBias = Tensor(FLOAT32, {vHeads}, dtValues, device);
    std::vector<float> conv(batch * length * channels), ba(batch * length * vHeads * 2);
    for (size_t i = 0; i < conv.size(); ++i)
        conv[i] = 0.13f * std::sin(float(i + 11 * length) * 0.019f);
    for (int b = 0; b < batch; ++b) for (int t = 0; t < length; ++t)
        for (int h = 0; h < vHeads; ++h) {
            int offset = (b * length + t) * vHeads * 2;
            ba[offset + h] = -6.0f + 0.013f * b + 0.07f * t - 0.02f * h;
            ba[offset + vHeads + h] = -0.3f + 0.03f * t + 0.02f * h;
        }
    Data convSequence = Tensor(FLOAT16, {batch, length, channels}, conv, device);
    Data baSequence = Tensor(FLOAT16, {batch, length, vHeads * 2}, ba, device);
    std::vector<Data> states(batch), snapshots(batch * slots), initialStates(batch);
    std::vector<Data> convKeys(batch), initialKeys(batch), convSnapshots(batch * slots);
    std::vector<Data*> keyPtrs(batch), initialKeyPtrs(batch), initialStatePtrs(batch);
    std::vector<Data*> convSnapshotPtrs(batch * slots);
    // Include an unused Z tail, as in the combined QKVZ projection.
    const int inputWidth = channels + 256;
    std::vector<float> inputValues(batch * length * inputWidth);
    for (size_t i = 0; i < inputValues.size(); ++i)
        inputValues[i] = 0.2f * std::sin(i * 0.031f);
    Data input = Tensor(FLOAT16, {batch, length, inputWidth}, inputValues, device);
    Data convWeight = Tensor(FLOAT32, {channels, 4}, std::vector<float>(channels * 4, 0.25f), device);
    Data convBias = Tensor(FLOAT32, {channels}, std::vector<float>(channels, 0.02f), device);
    for (int b = 0; b < batch; ++b) {
        Fill(convKeys[b], FLOAT16, {1, channels, 4},
             std::vector<float>(channels * 4, 0.01f * (b + 1)), device);
        initialKeys[b].CopyFrom(convKeys[b]);
        keyPtrs[b] = &convKeys[b]; initialKeyPtrs[b] = &initialKeys[b];
    }
    for (int i = 0; i < batch * slots; ++i) convSnapshotPtrs[i] = &convSnapshots[i];
    Require(FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16BatchPointers(
                keyPtrs, input, convWeight, convBias, convSequence,
                convSnapshotPtrs, slots, 0), "reference convolution rejected");
    conv = Read(convSequence);
    std::vector<Data *> statePtrs(batch), snapshotPtrs(batch * slots);
    std::vector<std::vector<std::vector<float>>> expectedStates(batch), expectedOutputs(batch);
    auto step = [&](int b, int t, Data &state) {
        const int row = b * length + t;
        Data c = Tensor(FLOAT16, {1, 1, channels},
                        {conv.begin() + row * channels, conv.begin() + (row + 1) * channels}, device);
        Data a = Tensor(FLOAT16, {1, 1, vHeads * 2},
                        {ba.begin() + row * vHeads * 2, ba.begin() + (row + 1) * vHeads * 2}, device);
        Data output;
        Require(FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
                    c, a, norm, aLog, dtBias, state, output,
                    kHeads, vHeads, kDim, vDim, 1e-6f, scale), "single-token reference rejected");
        return Read(output);
    };
    for (int b = 0; b < batch; ++b) {
        std::vector<float> initial(stateSize);
        for (int i = 0; i < stateSize; ++i)
            initial[i] = 0.025f * std::cos((i + b * 997) * 0.017f);
        Fill(states[b], FLOAT16, {1, vHeads, kDim, vDim}, initial, device);
        states[b].isLinearAttentionTransposed = true;
        statePtrs[b] = &states[b];
        initialStates[b].CopyFrom(states[b]);
        initialStatePtrs[b] = &initialStates[b];
        Data reference = Tensor(FLOAT16, {1, vHeads, kDim, vDim}, initial, device);
        reference.isLinearAttentionTransposed = true;
        for (int t = 0; t < length; ++t) {
            expectedOutputs[b].push_back(step(b, t, reference));
            expectedStates[b].push_back(Read(reference));
        }
    }
    for (int i = 0; i < batch * slots; ++i) snapshotPtrs[i] = &snapshots[i];
    Data output;
    Require(FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16BatchSnapshots(
                convSequence, baSequence, norm, aLog, dtBias, statePtrs, output,
                snapshotPtrs, slots, kHeads, vHeads, kDim, vDim, 1e-6f, scale),
            "batched snapshot kernel rejected");
    auto actual = Read(output);
    for (int b = 0; b < batch; ++b) {
        Near(Read(states[b]), expectedStates[b].back());
        for (int t = 0; t < length; ++t) {
            const int offset = (b * length + t) * outSize;
            Near({actual.begin() + offset, actual.begin() + offset + outSize}, expectedOutputs[b][t]);
            if (t < slots) {
                Require(snapshots[b * slots + t].isLinearAttentionTransposed, "lost snapshot layout");
                Near(Read(snapshots[b * slots + t]), expectedStates[b][t]);
            }
        }
    }
    // Cross-process fingerprints let the same executable compare the prepared
    // path with FASTLLM_CUDA_GDN_SEQUENCE_PREPARE=0, including every output and
    // prefix state. The independent single-token and compact-restore checks
    // below still validate the recurrence and mixed acceptance lengths.
    uint64_t digest = 14695981039346656037ULL;
    auto hashValues = [&](const std::vector<float> &values) {
        for (float value : values) {
            digest ^= float_to_half(value);
            digest *= 1099511628211ULL;
        }
    };
    hashValues(actual);
    for (int b = 0; b < batch; ++b) {
        hashValues(Read(states[b]));
        for (int t = 0; t < slots; ++t) hashValues(Read(snapshots[b * slots + t]));
    }
    for (int b = 0; b < batch; ++b) states[b].CopyFrom(initialStates[b]);
    Data withoutSnapshots;
    Require(FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16BatchSnapshots(
                convSequence, baSequence, norm, aLog, dtBias, statePtrs, withoutSnapshots,
                {}, 0, kHeads, vHeads, kDim, vDim, 1e-6f, scale),
            "no-snapshot sequence rejected");
    Require(Read(withoutSnapshots) == actual, "no-snapshot output not exact");
    hashValues(Read(withoutSnapshots));
    for (int b = 0; b < batch; ++b) hashValues(Read(states[b]));
    std::cout << "gdn_digest " << device << ':' << batch << ':' << length
              << ':' << kHeads << ':' << vHeads << ':' << vDim << '=' << digest << '\n';
    std::vector<Data> finalKeys(batch), finalStates(batch);
    for (int b = 0; b < batch; ++b) {
        finalKeys[b].CopyFrom(convKeys[b]); finalStates[b].CopyFrom(states[b]);
    }
    // Bit-exact against the original sequence snapshots, including a complete
    // prefix (must remain untouched) and prefix zero (pure rollback).
    for (int rotation = 0; rotation <= length; ++rotation) {
        std::vector<int> accepted(batch);
        for (int b = 0; b < batch; ++b) {
            convKeys[b].CopyFrom(finalKeys[b]); states[b].CopyFrom(finalStates[b]);
            accepted[b] = (b + rotation) % (length + 1);
        }
        Require(FastllmCudaDFlashRestoreLinearPrefixes(
                    input, convSequence, baSequence, norm, aLog, dtBias,
                    keyPtrs, statePtrs, initialKeyPtrs, initialStatePtrs, accepted,
                    kHeads, vHeads, kDim, vDim, 1e-6f), "compact recovery rejected");
        FastllmCudaSyncCurrentThreadStream();
        for (int b = 0; b < batch; ++b) {
            int n = accepted[b];
            const Data &expectedKey = n == 0 ? initialKeys[b] :
                (n == length ? finalKeys[b] : convSnapshots[b * slots + n - 1]);
            const Data &expectedState = n == 0 ? initialStates[b] :
                (n == length ? finalStates[b] : snapshots[b * slots + n - 1]);
            Require(Read(convKeys[b]) == Read(expectedKey), "compact convolution not exact");
            Require(Read(states[b]) == Read(expectedState), "compact recurrent state not exact");
            if (n > 0 && n < length) {
                Near(step(b, n, states[b]), expectedOutputs[b][n]);
                Near(Read(states[b]), expectedStates[b][n]);
            }
        }
    }
    // Reject malformed input before changing either cache.
    std::vector<int> badLengths(batch, length + 1);
    Require(!FastllmCudaDFlashRestoreLinearPrefixes(
                input, convSequence, baSequence, norm, aLog, dtBias,
                keyPtrs, statePtrs, initialKeyPtrs, initialStatePtrs, badLengths,
                kHeads, vHeads, kDim, vDim, 1e-6f), "invalid length accepted");
    // Different requests accept different prefixes. Cover every prefix length
    // and continue decoding, detecting lane swaps or restoring the wrong slot.
    for (int rotation = 0; rotation < slots; ++rotation) {
        std::vector<void *> destinations(batch);
        std::vector<const void *> sources(batch);
        std::vector<size_t> bytes(batch, stateSize * sizeof(uint16_t));
        for (int b = 0; b < batch; ++b) {
            const int prefix = 1 + (b + rotation) % slots;
            destinations[b] = states[b].cudaData;
            sources[b] = snapshots[b * slots + prefix - 1].cudaData;
        }
        Require(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                    destinations.data(), sources.data(), bytes.data(), batch), "batched restore failed");
        FastllmCudaSyncCurrentThreadStream();
        for (int b = 0; b < batch; ++b) {
            const int prefix = 1 + (b + rotation) % slots;
            Near(Read(states[b]), expectedStates[b][prefix - 1]);
            Near(step(b, prefix, states[b]), expectedOutputs[b][prefix]);
            Near(Read(states[b]), expectedStates[b][prefix]);
            Near(Read(snapshots[b * slots + prefix - 1]), expectedStates[b][prefix - 1]);
        }
    }
}
} // namespace

int main() {
    try {
        const int devices = FastllmCudaGetDeviceCount();
        if (!devices) return 77;
        int cases = 0;
        for (int device = 0; device < devices; ++device) {
            // Include shrinking batches after the largest one.
            for (int batch : {2, 4, 8, 10, 16, 24, 32, 3, 1}) for (int length = 2; length <= 8; ++length) {
                std::cout << "case " << device << ":" << batch << ":" << length << std::endl;
                Run(device, batch, length);
                ++cases;
            }
            std::cout << "device " << device << " PASS\n";
        }
        // Current TP2 heads, all verification lengths, small batches and V-tail
        // fallback. Return to smaller shapes after growth to exercise scratch reuse.
        for (int device = 0; device < devices; ++device) {
            for (int batch : {4, 8, 16, 3}) for (int length = 2; length <= 8; ++length) {
                Run(device, batch, length, 8, 24, 128);
                ++cases;
            }
            for (int vDim : {96, 192}) {
                Run(device, 4, 4, 2, 4, vDim);
                ++cases;
            }
        }
        std::cout << "DFlash batch snapshots: PASS (" << cases << " cases)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
