#include "gguf.h"
#include "model.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    template <typename T>
    void WritePod(std::ofstream &output, const T &value) {
        output.write(reinterpret_cast<const char*>(&value), sizeof(T));
        assert(output.good());
    }

    void WriteString(std::ofstream &output, const std::string &value) {
        WritePod(output, (uint64_t)value.size());
        output.write(value.data(), (std::streamsize)value.size());
        assert(output.good());
    }

    size_t AlignUp(size_t value, size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    void WritePadding(std::ofstream &output, size_t bytes) {
        const std::vector<char> zeros(bytes, 0);
        output.write(zeros.data(), (std::streamsize)zeros.size());
        assert(output.good());
    }

    struct PackedTensorSpec {
        std::string name;
        ggml_type type;
        int64_t rowElements;
        int64_t experts;
        size_t relativeOffset = 0;
        std::vector<uint8_t> payload;
    };

    std::vector<uint8_t> RepeatBlock(const void *block, size_t bytes,
                                     int experts) {
        std::vector<uint8_t> payload(bytes * (size_t)experts);
        for (int expert = 0; expert < experts; ++expert) {
            std::memcpy(payload.data() + (size_t)expert * bytes, block, bytes);
        }
        return payload;
    }

    class ScopedPackedFixture {
    public:
        ScopedPackedFixture() {
            constexpr int experts = 3;

            block_iq2_xs iq2{};
            iq2.d = 0x3c00;
            block_iq3_xxs iq3{};
            iq3.d = 0x3c00;
            block_mxfp4 mxfp4{};
            mxfp4.e = 128;
            for (int i = 0; i < QK_MXFP4 / 2; ++i) {
                mxfp4.qs[i] = (uint8_t)(((15 - i) << 4) | i);
            }

            specs = {
                {"blk.0.ffn_gate_exps.weight", GGML_TYPE_IQ2_XS,
                 QK_K, experts, 0,
                 RepeatBlock(&iq2, sizeof(iq2), experts)},
                {"blk.0.ffn_down_exps.weight", GGML_TYPE_IQ3_XXS,
                 QK_K, experts, 0,
                 RepeatBlock(&iq3, sizeof(iq3), experts)},
                {"blk.1.ffn_down_exps.weight", GGML_TYPE_MXFP4,
                 QK_MXFP4, experts, 0,
                 RepeatBlock(&mxfp4, sizeof(mxfp4), experts)},
            };

            size_t nextOffset = 0;
            for (auto &spec : specs) {
                spec.relativeOffset = AlignUp(nextOffset, kAlignment);
                nextOffset = spec.relativeOffset + spec.payload.size();
            }

            const auto nonce = std::chrono::high_resolution_clock::now()
                                   .time_since_epoch().count();
            path = std::filesystem::temp_directory_path() /
                   ("fastllm-dsv4-quant-" + std::to_string(nonce) + ".gguf");
            Write();
        }

        ~ScopedPackedFixture() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }

        std::filesystem::path path;
        std::vector<PackedTensorSpec> specs;
        size_t payloadStart = 0;

    private:
        static constexpr size_t kAlignment = 32;

        void Write() {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            assert(output.is_open());

            WritePod(output, (uint32_t)0x46554747);
            WritePod(output, (uint32_t)3);
            WritePod(output, (uint64_t)specs.size());
            WritePod(output, (uint64_t)1);

            WriteString(output, "general.alignment");
            WritePod(output, (uint32_t)GGUF_TYPE_UINT32);
            WritePod(output, (uint32_t)kAlignment);

            for (const auto &spec : specs) {
                WriteString(output, spec.name);
                WritePod(output, (uint32_t)3);
                WritePod(output, spec.rowElements);
                WritePod(output, (int64_t)1);
                WritePod(output, spec.experts);
                WritePod(output, (uint32_t)spec.type);
                WritePod(output, (uint64_t)spec.relativeOffset);
            }

            const size_t headerBytes = (size_t)output.tellp();
            payloadStart = AlignUp(headerBytes, kAlignment);
            WritePadding(output, payloadStart - headerBytes);

            size_t written = 0;
            for (const auto &spec : specs) {
                WritePadding(output, spec.relativeOffset - written);
                output.write(reinterpret_cast<const char*>(spec.payload.data()),
                             (std::streamsize)spec.payload.size());
                assert(output.good());
                written = spec.relativeOffset + spec.payload.size();
            }
        }
    };

    const fastllm::ReadGGUFTask &FindTask(
            const std::vector<fastllm::ReadGGUFTask> &tasks,
            const std::string &name) {
        for (const auto &task : tasks) {
            if (task.name == name) {
                return task;
            }
        }
        throw std::runtime_error("missing expected GGUF task: " + name);
    }

    void AssertAll(const std::vector<float> &values, float expected) {
        for (float value : values) {
            assert(std::fabs(value - expected) < 1e-6f);
        }
    }

    void TestQuantTraits() {
        static_assert(sizeof(block_mxfp4) == 17,
                      "MXFP4 must preserve the GGUF wire layout");
        assert((int)GGML_TYPE_MXFP4 == 39);
        assert(std::string(ggml_type_name(GGML_TYPE_MXFP4)) == "mxfp4");
        assert(ggml_blck_size(GGML_TYPE_MXFP4) == QK_MXFP4);
        assert(ggml_type_size(GGML_TYPE_MXFP4) == sizeof(block_mxfp4));
        assert(ggml_type_to_float(GGML_TYPE_MXFP4) != nullptr);
        assert(ggml_type_to_float(GGML_TYPE_IQ2_XS) != nullptr);
        assert(ggml_type_to_float(GGML_TYPE_IQ3_XXS) != nullptr);
        assert(ggml_row_size(GGML_TYPE_MXFP4, QK_MXFP4) == 17);
        assert(ggml_row_size(GGML_TYPE_IQ2_XS, QK_K) ==
               sizeof(block_iq2_xs));
        assert(ggml_row_size(GGML_TYPE_IQ3_XXS, QK_K) ==
               sizeof(block_iq3_xxs));
    }

    void TestKnownDequantVectors() {
        static const float mxfp4Values[16] = {
            0, 1, 2, 3, 4, 6, 8, 12,
            0, -1, -2, -3, -4, -6, -8, -12,
        };
        block_mxfp4 mxfp4{};
        mxfp4.e = 128;
        for (int i = 0; i < QK_MXFP4 / 2; ++i) {
            mxfp4.qs[i] = (uint8_t)(((15 - i) << 4) | i);
        }
        std::vector<float> mxfp4Output(QK_MXFP4);
        dequantize_row_mxfp4(&mxfp4, mxfp4Output.data(), QK_MXFP4);
        for (int i = 0; i < 16; ++i) {
            assert(mxfp4Output[i] == mxfp4Values[i]);
            assert(mxfp4Output[i + 16] == mxfp4Values[15 - i]);
        }

        mxfp4.e = 127;
        dequantize_row_mxfp4(&mxfp4, mxfp4Output.data(), QK_MXFP4);
        for (int i = 0; i < 16; ++i) {
            assert(mxfp4Output[i] == 0.5f * mxfp4Values[i]);
        }

        block_iq2_xs iq2{};
        iq2.d = 0x3c00;
        std::vector<float> iq2Output(QK_K);
        dequantize_row_iq2_xs(&iq2, iq2Output.data(), QK_K);
        AssertAll(iq2Output, 1.0f);

        block_iq3_xxs iq3{};
        iq3.d = 0x3c00;
        std::vector<float> iq3Output(QK_K);
        dequantize_row_iq3_xxs(&iq3, iq3Output.data(), QK_K);
        AssertAll(iq3Output, 1.0f);
    }

    void TestPackedExpertOffsetsAndShortRead() {
        ScopedPackedFixture fixture;
        std::vector<fastllm::ReadGGUFTask> tasks;
        fastllm::AppendGGUFTasks("deepseek_v4", fixture.path.string(), tasks);
        assert(tasks.size() == fixture.specs.size() * 3);

        const std::vector<std::string> suffixes = {
            "w1.weight", "w2.weight", "w2.weight",
        };
        for (size_t tensor = 0; tensor < fixture.specs.size(); ++tensor) {
            const auto &spec = fixture.specs[tensor];
            const size_t expertBytes = spec.payload.size() / 3;
            const int layer = tensor == 2 ? 1 : 0;
            for (int expert = 0; expert < 3; ++expert) {
                const std::string name =
                    "layers." + std::to_string(layer) + ".ffn.experts." +
                    std::to_string(expert) + "." + suffixes[tensor];
                const auto &task = FindTask(tasks, name);
                assert(task.tensor.type == spec.type);
                assert(task.tensor.dims ==
                       std::vector<int>({1, (int)spec.rowElements}));
                assert(task.offset == fixture.payloadStart +
                       spec.relativeOffset + (size_t)expert * expertBytes);
                assert(ggml_nbytes(&task.tensor) == expertBytes);
            }
        }

        const auto &last = FindTask(
            tasks, "layers.1.ffn.experts.2.w2.weight");
        std::filesystem::resize_file(
            fixture.path, std::filesystem::file_size(fixture.path) - 1);
        ggml_tensor tensor = last.tensor;
        std::string fileName = last.fileName;
        fastllm::Data weight;
        std::string message;
        try {
            fastllm::WeightImportGGUFTensor(
                &weight, &tensor, fileName, last.offset);
        } catch (const std::runtime_error &error) {
            message = error.what();
        }
        assert(message.find("Short payload read") != std::string::npos);
        assert(message.find("blk.1.ffn_down_exps.weight") !=
               std::string::npos);
    }
}

int main() {
    TestQuantTraits();
    TestKnownDequantVectors();
    TestPackedExpertOffsetsAndShortRead();
    std::cout << "DeepSeek V4 quantized GGUF tests passed.\n";
    return 0;
}
