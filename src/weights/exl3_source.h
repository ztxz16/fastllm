#pragma once
#include "tensor_source.h"
#include "exl3.h"
#include "utils.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>

namespace fastllm {
    struct Exl3SourceRange {
        std::string file;
        uint64_t offset = 0, bytes = 0;
        void Read(uint8_t *destination) const {
            FILE *stream = fopen(file.c_str(), "rb");
            AssertInFastLLM(stream != nullptr, "EXL3: cannot open " + file);
#if defined(_WIN32) || defined(_WIN64)
            _fseeki64(stream, offset, SEEK_SET);
#else
            fseeko(stream, (off_t)offset, SEEK_SET);
#endif
            size_t got = fread(destination, 1, bytes, stream);
            fclose(stream);
            AssertInFastLLM(got == bytes, "EXL3: truncated tensor in " + file);
        }
    };
    struct Exl3SourcePart {
        Exl3SourceRange trellis, inputScale, outputScale;
        int inputs = 0, outputs = 0, twiceBits = 0;
        uint64_t Bytes() const { return Exl3PartBytes(outputs, inputs, twiceBits); }
        void Read(uint8_t *destination) const {
            trellis.Read(destination);
            inputScale.Read(destination + trellis.bytes);
            outputScale.Read(destination + trellis.bytes + inputScale.bytes);
        }
    };

    struct Exl3TensorSource final : TensorSource {
        std::vector<Exl3SourcePart> parts;
        DataType storedType = EXL3;
        uint64_t bytes = 0, len = 0;
        DataType StoredType() const override { return storedType; }
        void Configure(Data &weight) const override {
            if (storedType == EXL3) {
                weight.blockK = parts.front().twiceBits;
                weight.blockM = int(parts.size());
                weight.UpdateUnitSize();
            }
        }
        uint8_t *Read(DataType dstType) const override {
            uint8_t *buffer = nullptr;
            if (storedType == EXL3) {
                AssertInFastLLM(dstType == DataType::EXL3 && !parts.empty(),
                                "EXL3: compressed matrices must retain their source type.");
                buffer = new uint8_t[bytes];
                size_t offset = 0;
                for (const auto &part : parts) { part.Read(buffer + offset); offset += part.Bytes(); }
                AssertInFastLLM(offset == bytes, "EXL3: compressed projection size mismatch.");
                return buffer;
            }
            // Mixed-bitrate grouped slices retain the existing floating fallback.
            // Uniform groups and routed experts keep their compressed bytes.
            AssertInFastLLM(dstType == FLOAT32 || dstType == FLOAT16 || dstType == BFLOAT16,
                            "EXL3: grouped projection requires floating-point storage.");
            const size_t unit = dstType == FLOAT32 ? 4 : 2;
            buffer = new uint8_t[len * unit];
            size_t offset = 0;
            for (const auto &part : parts) {
                std::vector<uint8_t> packed(part.Bytes());
                std::vector<float> matrix(size_t(part.inputs) * part.outputs);
                part.Read(packed.data());
                Exl3Reconstruct(packed.data(), part.outputs, part.inputs, part.twiceBits, matrix.data());
                if (dstType == FLOAT32) memcpy(buffer + offset * unit, matrix.data(), matrix.size() * unit);
                else for (size_t i = 0; i < matrix.size(); ++i) {
                    uint16_t bits;
                    if (dstType == FLOAT16) bits = float_to_half(matrix[i]);
                    else { uint32_t word; memcpy(&word, &matrix[i], 4); word += 0x7fff + ((word >> 16) & 1); bits = word >> 16; }
                    reinterpret_cast<uint16_t *>(buffer)[offset + i] = bits;
                }
                offset += matrix.size();
            }
            AssertInFastLLM(offset == len, "EXL3: grouped projection size mismatch.");
            return buffer;
        }

    };

    template<class Item>
    void NormalizeExl3Sources(std::map<std::string, Item> &itmeDict) {
        std::map<std::string, std::vector<std::pair<int, Exl3SourcePart>>> grouped;
        std::vector<std::string> remove;
        auto range = [](const Item &item) {
            return Exl3SourceRange{item.fileName, item.data_offsets.at(0), item.bytes};
        };
        for (const auto &entry : itmeDict) {
            const std::string &name = entry.first;
            if (!StringEndWith(name, ".trellis")) continue;
            const auto &tr = entry.second;
            const std::string prefix = name.substr(0, name.size() - 8);
            auto su = itmeDict.find(prefix + ".suh"), sv = itmeDict.find(prefix + ".svh");
            auto marker = itmeDict.find(prefix + ".mul1");
            AssertInFastLLM(tr.dtype == "I16" && tr.intShape.size() == 3 &&
                su != itmeDict.end() && sv != itmeDict.end() && marker != itmeDict.end(),
                "EXL3: incomplete matrix or unsupported codebook: " + prefix);
            Exl3SourcePart part;
            part.inputs = tr.intShape[0] * 16;
            part.outputs = tr.intShape[1] * 16;
            AssertInFastLLM(tr.intShape[2] % 8 == 0, "EXL3: invalid trellis packing: " + prefix);
            part.twiceBits = tr.intShape[2] / 8;
            part.trellis = range(tr); part.inputScale = range(su->second); part.outputScale = range(sv->second);
            AssertInFastLLM(su->second.dtype == "F16" && sv->second.dtype == "F16" &&
                su->second.intShape == std::vector<int>{part.inputs} &&
                sv->second.intShape == std::vector<int>{part.outputs} &&
                tr.bytes + su->second.bytes + sv->second.bytes == part.Bytes(),
                "EXL3: inconsistent matrix/scales: " + prefix);
            AssertInFastLLM(marker->second.dtype == "I32" && marker->second.bytes == 4 &&
                marker->second.len == 1, "EXL3: invalid mul1 marker: " + prefix);
            uint32_t multiplier = 0; range(marker->second).Read((uint8_t*)&multiplier);
            AssertInFastLLM(multiplier == 0x83dcd12du, "EXL3: unsupported mul1 multiplier: " + prefix);
            std::string target = prefix + ".weight";
            int slice = -1;
            const size_t at = prefix.rfind(".wo_a.slice.");
            if (at != std::string::npos) {
                std::string number = prefix.substr(at + 12);
                // '.wo_a.slice.' has twelve characters.
                AssertInFastLLM(!number.empty() && number.find_first_not_of("0123456789") == std::string::npos,
                                "EXL3: invalid grouped projection suffix: " + prefix);
                slice = std::stoi(number);
                target = prefix.substr(0, at) + ".wo_a.weight";
            }
            grouped[target].push_back({slice, part});
            for (const char *suffix : {".trellis", ".suh", ".svh", ".mul1", ".scale", ".weight_scale"})
                remove.push_back(prefix + suffix);
        }
        AssertInFastLLM(!grouped.empty(), "EXL3 checkpoint contains no supported matrices.");
        for (const std::string &name : remove) itmeDict.erase(name);
        for (auto &entry : grouped) {
            auto &parts = entry.second;
            std::sort(parts.begin(), parts.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            const auto &first = parts.front().second;
            Item tensor;
            tensor.tensorName = entry.first;
            tensor.fileName = first.trellis.file;
            tensor.data_offsets = {first.trellis.offset, first.trellis.offset + first.trellis.bytes};
            const bool sliced = parts.front().first >= 0;
            if (sliced) {
                for (size_t i = 0; i < parts.size(); ++i)
                    AssertInFastLLM(parts[i].first == (int)i && parts[i].second.inputs == first.inputs &&
                                    parts[i].second.outputs == first.outputs,
                                    "EXL3: incomplete/inconsistent grouped projection: " + entry.first);
                const bool sameBits = std::all_of(parts.begin(), parts.end(), [&](const auto &p) { return p.second.twiceBits == first.twiceBits; });
                tensor.dtype = sameBits ? "EXL3" : "BF16";
                tensor.intShape = {(int)parts.size(), first.outputs, first.inputs};
                tensor.len = uint64_t(parts.size()) * first.outputs * first.inputs;
                tensor.bytes = sameBits ? parts.size() * first.Bytes() : tensor.len * 2;
                tensor.blockK = first.twiceBits; tensor.blockM = int(parts.size());
            } else {
                AssertInFastLLM(parts.size() == 1, "EXL3: duplicate matrix " + entry.first);
                tensor.dtype = "EXL3";
                tensor.intShape = {first.outputs, first.inputs};
                tensor.len = uint64_t(first.outputs) * first.inputs;
                tensor.bytes = first.Bytes();
                tensor.blockK = first.twiceBits; tensor.blockM = 1;
            }
            tensor.shape.assign(tensor.intShape.begin(), tensor.intShape.end());
            auto source = std::make_shared<Exl3TensorSource>();
            source->storedType = tensor.dtype == "EXL3" ? EXL3 : DATA_AUTO_NONE;
            source->bytes = tensor.bytes; source->len = tensor.len;
            for (const auto &part : parts) source->parts.push_back(part.second);
            tensor.source = std::move(source);
            itmeDict[entry.first] = tensor;
        }
        printf("[Fastllm] EXL3: %zu logical projections, original mul1 trellises retained.\n", grouped.size());
    }

}
